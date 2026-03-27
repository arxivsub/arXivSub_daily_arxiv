# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-03-27 | 今日论文总数: 502

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. Dual-Graph Multi-Agent Reinforcement Learning for Handover Optimization

**arXiv ID:** 2603.24634 | [PDF](https://arxiv.org/pdf/2603.24634v1)

**作者:** Matteo Salvatori `[一作]` (Telefónica Research), Ioannis Arapakis `[通讯]` (Telefónica Research)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于双图多智能体强化学习的Handover（HO）优化框架，通过在每条邻接关系上调节Cell Individual Offset（CIO）来提升网络吞吐量。

**💡 创新点**

创新点包括：① 将HO控制问题转化为在网络双图（边为智能体）上的Dec-POMDP，充分利用CIO的边缘结构；② 设计了TD3-D-MA算法，结合离散动作的Gumbel-Softmax、共享参数GNN演员以及区域级双Critic，实现CTDE训练与分布式执行；③ 通过区域化Critic改善稠密部署下的信用分配与学习稳定性。

**🔧 技术方法**

主要技术：离散化TD3（TD3-D）、GNN（IN/Transformer/GAT等）演员、共享参数、Gumbel-Softmax采样、区域化双Critic、CTDE训练、NS‑3仿真、Gymnasium接口、Stable‑Baselines3扩展。

**📊 数据集**

使用了开源NS‑3仿真环境，基于真实运营商参数构建的两套数据集：① 8个小区的基准网络；② 约2km²曼彻斯特市区30个天线（30小区）的大规模真实网络，涵盖多种交通和移动性场景。

**📈 对比分析**

在基准和曼彻斯特场景下与集中式RL、RRM、SON、Δ‑CIO等规则基线进行比较，评价指标为网络总吞吐量并归一化到最佳基线与专用RL上。实验显示TD3‑D‑MA在训练和测试（包括拓扑/流量迁移）场景下均能显著提升吞吐量（约10–30%），并在拓扑变化时保持鲁棒性。

**⚠️ 局限性**

局限性：① 离散动作空间可能限制细粒度控制；② 仅针对CIO，未联合优化其他HO控制参数；③ 区域化Critic需人工划分子网，扩展到更大规模网络时可能面临划分和通信开销；④ 实时部署时的信令与时延成本未进行评估。

---

## 2. An Approach to Generate Attack Graphs with a Case Study on Siemens PCS7 Blueprint for Water Treatment Plants

**arXiv ID:** 2603.24888 | [PDF](https://arxiv.org/pdf/2603.24888v1)

**作者:** Lucas Miranda `[一作]` (Universidade Federal do Rio de Janeiro), Tobias Limmer `[通讯]` (Siemens)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出一种半自动化方法，利用网络拓扑解析器、漏洞管理器和攻击图生成器，生成工业控制系统（ICS）的攻击图并对攻击路径进行定量评估。

**💡 创新点**

创新点在于：①将网络拓扑与漏洞信息融合，通过规则将 CVE 转化为前置条件与后果；②采用有状态遍历算法枚举多步攻击路径；③结合 EPSS 与可靠性块图实现攻击成功概率估算，提供可操作的风险优先级。

**🔧 技术方法**

主要技术包括：JSON 拓扑解析、基于 CVSS 的规则引擎、状态机式图遍历、可靠性块图（RBD）映射、EPSS 概率模型和交互式可视化渲染。

**📊 数据集**

使用的数据集包括：Siemens PCS7 Cybersecurity Blueprint（水处理厂网络拓扑），NVD CVE 数据库（SSAs）以及手工编制的设备别名映射表。

**📈 对比分析**

方法通过案例研究展示，能够识别单点失效、评估补丁效果，并通过 EPSS 计算得到每个目标设备的被攻破概率，显示对比传统定性评估的定量优势；然而未给出与现有自动化工具的客观性能对比。

**⚠️ 局限性**

局限性包括：①不考虑固件/补丁级别导致误纳入已修补漏洞；②对权限提升与横向移动的模拟过于简化；③缺乏对防火墙 ACL 细粒度配置与误配的完整建模；④未评估工具的可扩展性与运行效率。

---

## 3. Light Cones For Vision: Simple Causal Priors For Visual Hierarchy

**arXiv ID:** 2603.24753 | [PDF](https://arxiv.org/pdf/2603.24753v1)

**作者:** Manglam Kartik `[一作]` (Indian Institute of Technology Bombay), Neel Tushar Shah `[通讯]` (Indian Institute of Technology Bombay)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了Worldline Slot Attention，利用Lorentzian时空光锥将物体视为跨层级的持续轨迹，实现视觉层次结构的自动发现。

**💡 创新点**

创新点在于将层级约束转化为时空世界线绑定，并证明几何结构（Lorentzian光锥）是实现层次学习的关键；同时展示Lorentzian优于欧氏和双曲空间。

**🔧 技术方法**

使用Lorentzian时空嵌入、尺度自适应注意力、GRU更新以及传统Slot Attention框架。

**📊 数据集**

在三种合成数据集（Toy Hierarchical、Sprites、CLEVR）上进行评估。

**📈 对比分析**

与欧氏世界线、超平面、双曲空间等基线相比，Lorentzian世界线在层级准确率上提升6–8倍（从0.078到0.48–0.66），并在对象聚类上也表现良好。

**⚠️ 局限性**

局限性包括：仅针对基于密度的层级假设；固定层级深度（3级）；仅在二维点云上测试，未整合完整视觉编码器；需验证在真实语义层次上的泛化。

---

## 4. Supervising Ralph Wiggum: Exploring a Metacognitive Co-Regulation Agentic AI Loop for Engineering Design

**arXiv ID:** 2603.24768 | [PDF](https://arxiv.org/pdf/2603.24768v1)

**作者:** Zeda Xu `[一作]` (Carnegie Mellon University), Christopher McComb `[通讯]` (Carnegie Mellon University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出了自我调节循环（SRL）和共调节循环（CRDAL）两种元认知机制，用于增强LLM代理在工程设计中的决策，并在18650锂电池组配置设计任务中进行实验验证。

**💡 创新点**

首次将元认知自我调节与共调节引入LLM代理式设计系统；通过共调节循环显著提升设计性能并缓解设计定着。

**🔧 技术方法**

采用Google DeepMind的Gemini 3.1 Pro LLM，构建RWL、SRL、CRDAL三种架构，使用数值评估器、验证器以及进度分析器等工具实现设计与评估。

**📊 数据集**

实验使用18650锂电池的物理参数与设计约束作为评估基准，没有采用公开数据集，仅利用自定义模拟器对设计进行数值评估。

**📈 对比分析**

对30次独立实验分别记录容量、设计步骤和设计空间位置；统计检验显示CRDAL平均容量70.92 Ah，显著高于RWL（49.31 Ah）和SRL（54.14 Ah），计算成本无显著差异；CRDAL在设计空间中覆盖更多高容量区域。

**⚠️ 局限性**

局限性包括仅在单一电池组配置任务中测试；仅使用Gemini 3.1 Pro；工具仅限评估器和验证器，未提供优化器或更丰富的设计工具；未评估多代理或不同LLM的表现；缺乏对元认知交互细节的深入分析。

---

## 5. From Untestable to Testable: Metamorphic Testing in the Age of LLMs

**arXiv ID:** 2603.24774 | [PDF](https://arxiv.org/pdf/2603.24774v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 6. When Is Collective Intelligence a Lottery? Multi-Agent Scaling Laws for Memetic Drift in LLMs

**arXiv ID:** 2603.24676 | [PDF](https://arxiv.org/pdf/2603.24676v1)

**作者:** Hidenori Tanaka `[一作]` (Harvard University), Hidenori Tanaka `[通讯]` (Harvard University)

**通讯引用:** 2899 | [OpenAlex ID](https://openalex.org/A5050969230)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种名为量化简单共识（QSG）的模型，研究多智能体系统中大语言模型（LLM）如何通过相互学习形成共识，揭示了在中立条件下共识形成的机制。

**💡 创新点**

创新点在于引入了相互上下文学习和模因漂移的概念，阐明了在没有个体偏好的情况下，如何通过采样驱动的机制打破对称性并达成共识。

**🔧 技术方法**

使用了量化简单共识（QSG）模型，该模型结合了连续信念、量化通信和上下文适应，能够分析多智能体协调的动态。

**📊 数据集**

使用了命名游戏作为实验设置，LLM群体在没有外部奖励或真实标准的情况下进行互动，形成集体约定。

**📈 对比分析**

通过QSG模型的模拟和命名游戏实验验证了预测的缩放规律，结果表明，群体大小、通信带宽和适应率共同决定了共识是由放大采样噪声主导还是由系统偏见主导。

**⚠️ 局限性**

限制在于QSG模型是一个简化的理论框架，可能无法捕捉到多智能体系统中所有复杂的交互和动态，未来需要扩展分析以考虑更复杂的网络结构和异质智能体。

---

## 7. Surrogates, Spikes, and Sparsity: Performance Analysis and Characterization of SNN Hyperparameters on Hardware

**arXiv ID:** 2603.24891 | [PDF](https://arxiv.org/pdf/2603.24891v1)

**作者:** Ilkin Aliyev `[一作]` (University of Arizona), Tosiron Adegbija `[通讯]` (University of Arizona)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `cc175879-ab65-4aa9-b58a-f6100a057dbf`

**🎯 论文内容**

对SNN训练时的超参数（代理梯度函数与神经元模型）进行系统工作负载特征化，量化它们对硬件推理延迟和稀疏度的影响，并在周期精确FPGA平台上验证；

**💡 创新点**

① 证明代理梯度函数与硬件稀疏性/延迟存在可预测对应关系；② 显示更复杂的Lapicque神经元可通过减少事件数实现比LIF更低的延迟，突破传统简化模型更快的假设；③ 建立硬件感知的DSE流程，将训练超参数直接映射到实际推理性能；

**🔧 技术方法**

使用Fast Sigmoid、Arctangent、Spike Rate Escape、Stochastic Spike Operator四种代理梯度；LIF与Lapicque神经元模型；4‑bit量化；Optuna自动化超参数搜索；周期精确FPGA仿真；事件驱动的SNN硬件平台；

**📊 数据集**

DVS128‑Gesture、N‑MNIST、DVS‑CIFAR10三个事件驱动视觉数据集；

**📈 对比分析**

在同一FPGA平台上对不同代理梯度和神经元配置进行基准测评，绘制准确率–延迟Pareto图；结果显示，合适的代理梯度与Lapicque模型可将延迟降至5‑6 ms、准确率≥95%，相较基线提升12‑28%的延迟，且速度可与定制ASIC相媲美；

**⚠️ 局限性**

验证仅限单一FPGA平台，资源规模受限；代理梯度与神经元参数的最优组合高度依赖具体数据集，缺乏统一的硬件感知损失函数；未探讨更大网络或更高精度配置的迁移性；

---

## 8. Enhancing Structured Meaning Representations with Aspect Classification

**arXiv ID:** 2603.24797 | [PDF](https://arxiv.org/pdf/2603.24797v1)

**作者:** Claire Benét Post `[一作]`, Alexis Palmer `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文创建了一个新的英文句子语义表示数据集，给每个句子中的事件按UMR方面格（aspect lattice）标注了六种细粒度方面标签，并详细说明了标注流程、准则与协商机制。

**💡 创新点**

其创新点在于：①首次构建规模化、监督式的UMR方面标注数据集；②引入层次化方面格以兼容跨语言表达；③提出多阶段人工标注与仲裁流程，实现高质量标签；④提供基线评估，为后续自动化UMR解析奠定基础。

**🔧 技术方法**

技术方面，作者使用了：①标注指南与多轮训练+仲裁的人工标注体系；②三类基线模型——规则型Aspect分类器、基于LLama-3.1 8B平均嵌入的前馈神经分类器，以及LLM（GPT‑5mini与Llama）提示式预测；③标准化数据切分（70/15/15）与评价指标（加权F1、宏F1等）。

**📊 数据集**

所用数据集为UMR 2.0 约30k图的子集，挑选了四个子语料库（Little Prince、Minecraft、BOLT DF、Weblog），并在Pear Story语料上进行训练演练，最终得到1473条手工标注的方面标签。

**📈 对比分析**

作者对基线模型在同一分层测试集上进行比较：GPT‑5mini在所有指标上优于Llama；规则型AutoAspect虽无法在新数据上直接复现，但在其原始数据上表现良好；前馈分类器性能最差。宏F1显示所有自动方法在少数类上的表现不足，且人类标注基线远超自动模型。

**⚠️ 局限性**

局限性包括：①未能在新数据集上复现AutoAspect规则集（依赖问题）；②数据仅覆盖英语，缺乏跨语言验证；③标注以单句为单位，缺少文档级上下文，导致部分事件难以区分；④少数类样本稀缺，影响模型学习和评估。

---

## 9. A Nonvolatile Switchable-polarity EPM Valve

**arXiv ID:** 2603.24811 | [PDF](https://arxiv.org/pdf/2603.24811v1)

**作者:** Bingchao Wang `[一作]` (University of Edinburgh), Adam A. Stokes `[通讯]` (University of Edinburgh)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了一种能通过瞬态电激励实现磁极可逆的Switchable‑Polarity ElectroPermanent Magnet（S‑EPM）及其集成的低功耗气液阀，并将其应用于可编程的多通道流体网络（如解码器、分配树、六端口路由器等）

**💡 创新点**

S‑EPM突破传统EPM只能切换磁化/无磁化的限制，实现了磁极双稳态；通过磁通重构实现零功耗保持态；阀采用压缩式设计，兼容气液，动态/静态阻压高达320/500 kPa；将磁极态映射到流体逻辑，构建“数字流体”架构

**🔧 技术方法**

磁极可逆S‑EPM（Alnico + NdFeB +软磁铁 + 150 圈35 AWG电磁线）+ 3D打印(PLA)外壳 + 气液管路 + 多重永久磁阵列 + 微控制器+H‑桥驱动+传感器

**📊 数据集**

无公开数据集，实验使用压缩空气、硅胶管、食用色素等标准试剂

**📈 对比分析**

性能对比：阀在动态阻压高达320 kPa（标准阀多在<200 kPa），静态阻压至500 kPa；开关时间≈0.1 s；能耗仅0.6 J/切换；零功耗保持；相较传统机械/电磁阀在功耗、热漂移、密封性方面优越；在六端口、解码器等网络中实现了完全可编程、非挥发性路由，提升了可扩展性

**⚠️ 局限性**

动态阻压峰值约320 kPa，单次切换偶尔无法完全压紧导致需重复激励；尺寸较大，适合实验室级，需进一步微型化；磁场泄漏与材料成本仍需优化

---

## 10. Decentralized Task Scheduling in Distributed Systems: A Deep Reinforcement Learning Approach

**arXiv ID:** 2603.24738 | [PDF](https://arxiv.org/pdf/2603.24738v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 11. Automatization of building IT projects using composite consistency rules

**arXiv ID:** 2603.24726 | [PDF](https://arxiv.org/pdf/2603.24726v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 12. Accurate Point Measurement in 3DGS -- A New Alternative to Traditional Stereoscopic-View Based Measurements

**arXiv ID:** 2603.24716 | [PDF](https://arxiv.org/pdf/2603.24716v1)

**作者:** Deyan Deng `[一作]` (Ohio State University), Rongjun Qin `[通讯]` (Ohio State University)

**通讯引用:** 3979 | [OpenAlex ID](https://openalex.org/A5017812815)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

开发了基于3D Gaussian Splatting（3DGS）渲染的Web测量工具，利用多射线空间交点实现高精度3D坐标获取；

**💡 创新点**

创新点在于将经典空间交点方法迁移到无显式几何的3DGS渲染中，支持多视角多射线求解并实时展示不确定性，消除了对立体工作站和网格的依赖；

**🔧 技术方法**

使用了3DGS渲染、Cesium.js与WebGL、JavaScript实现最小二乘空间交点、RMS误差估计以及交互式用户界面；

**📊 数据集**

实验使用了三套UAV采集的实景数据集（钢雕、木雕、校园建筑），分别生成3DGS和传统网格模型；

**📈 对比分析**

通过与直接在网格上单点测量和高精度地面真值比较，3DGS测量的RMSE为1–2 cm，显著优于网格（3–4 cm），在薄结构和尖角等难测区域能够完整测量且误差更低；

**⚠️ 局限性**

局限性包括仍需人工对准射线、缺乏自动特征匹配、对纹理不足区域可能受限，以及目前仅支持点测量，需扩展至距离/面积/体积等测量任务。

---

## 13. A Dual-Threshold Probabilistic Knowing Value Logic

**arXiv ID:** 2603.24865 | [PDF](https://arxiv.org/pdf/2603.24865v1)

**作者:** Shanxia Wang `[一作]` `[通讯]` (Henan Normal University), Shanxia Wang (Henan Normal University)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

提出了双阈值概率式知道值逻辑，统一了对命题的概率阈值认知与对术语取值的高置信度认知；

**💡 创新点**

核心创新在于阈值双分层：命题阈值保持完整区间[0,1]，而知道值阈值限定在(1/2,1]，从而自动保证唯一性并实现非真值性（非 factive）高置信度锁定；在此基础上给出完整的语义、可证明性系统以及结构化弱完备性证明；

**🔧 技术方法**

采用概率模型计量语义、类型空间分布与赋值配置映射的两层构造、线性不等式约束可行性、赋值配置空间的引入以及归纳真值引理；

**📊 数据集**

本工作为理论性研究，无实验数据集；

**📈 对比分析**

通过证明理论（可达性与完备性）与语义验证（模型构造）进行对比，未涉及实验性能评估；

**⚠️ 局限性**

局限性包括：只适用于高阈值(>1/2)；未处理低阈值下可能出现的多值竞争；未引入群体知识或动态更新机制；缺乏可计算性与复杂度分析，且完整系统非递归可枚举。

---

## 14. DCARL: A Divide-and-Conquer Framework for Autoregressive Long-Trajectory Video Generation

**arXiv ID:** 2603.24835 | [PDF](https://arxiv.org/pdf/2603.24835v1)

**作者:** Junyi Ouyang `[一作]` (Institute for Creative Technologies), Haiwei Chen `[通讯]` (Institute for Creative Technologies)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 DCARL 框架，采用分阶段关键帧生成与局部插值的方式实现长轨迹视频生成。

**💡 创新点**

创新点是将关键帧全序列生成与段落插值结合，借助分块生成缓解自回归漂移。

**🔧 技术方法**

使用 DiT-based 流匹配模型、VAE 编码解码、光照/相机特征嵌入以及 Motion‑Inductive Noisy Conditioning。

**📊 数据集**

在 OpenDV‑YouTube (ODV‑YT) 480h 以及 nuScenes 验证集上进行训练与测试。

**📈 对比分析**

相较于 DiffF、SelfF、Vista、SEVA 等基线，DCARL 在 FID/FVD 及 ATE/ARE 上均表现更优，可稳定生成长达 32s 的视频。

**⚠️ 局限性**

局限在于关键帧与插值两阶段需额外训练、参数调优，且对极端动态场景的泛化仍有限。

---

## 15. DRoPS: Dynamic 3D Reconstruction of Pre-Scanned Objects

**arXiv ID:** 2603.24770 | [PDF](https://arxiv.org/pdf/2603.24770v1)

**作者:** Narek Tumanyan `[一作]` (Weizmann Institute of Science), Jonathon Luiten `[通讯]` (Meta Reality Labs)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

通过预先扫描静态对象并结合单目视频，构建可动态变形的三维高质量重建和极端视角的新视图合成。

**💡 创新点**

引入表面对齐的网格化高斯原语与基于CNN的运动参数化，利用预扫描先验和卷积网络隐式正则化，实现几何一致且高保真动态重建。

**🔧 技术方法**

采用3D Gaussian Splatting为表示，构造像素网格的表面对齐高斯，使用U-Net CNN预测6-DOF变形，结合深度、2D跟踪及多尺度等距损失。

**📊 数据集**

在Panoptic Studio真实捕捉数据和Truebones合成动物数据上进行实验，单目视频通过前一帧的预扫描或图像-3D模型生成。

**📈 对比分析**

与HiMoR、OriGS和Cog-NVS等最先进单目动态重建方法对比，PSNR、LPIPS、CLIP、3D追踪EPE等指标均优于对手，尤其在极端新视角和长程跟踪上显著提升。

**⚠️ 局限性**

仅支持单一前景主体，不能处理透明物体或拓扑变化，对2D跟踪与深度估计误差敏感，缺乏多对象独立运动建模。

---

## 16. Prune as You Generate: Online Rollout Pruning for Faster and Better RLVR

**arXiv ID:** 2603.24840 | [PDF](https://arxiv.org/pdf/2603.24840v1)

**作者:** Haobo Xu `[一作]` (University of Illinois at Urbana-Champaign), Hanghang Tong `[通讯]` (University of Illinois at Urbana-Champaign)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种在线回合剪枝方法ARRoL，能在生成过程中根据轻量级质量头预测的成功概率提前剔除不必要的rollout，提升RLVR训练效率并强化学习信号。

**💡 创新点**

创新点在于：①利用实时质量头预测部分rollout的正确率，动态控制生成过程中的回合平衡；②将剪枝策略与训练系统紧密耦合，实现端到端加速；③将训练得到的质量头用于测试时的加权投票，进一步提升最终答案准确率。

**🔧 技术方法**

技术包括：RLVR（GRPO/DAPO）框架、轻量级质量头（MLP）和概率校准、在线分桶后验估计、前后端分离的生成系统（verl/vLLM）以及测试时加权投票。

**📊 数据集**

使用数学推理任务数据集：Math500、MinervaMath、OlympiadBench、AMC'23、AIME'24、AIME'25，以及训练用的Dapo-Math-17K。

**📈 对比分析**

相较于原始GRPO/DAPO，ARRoL在Qwen-3与LLaMA-3.2模型（1B-8B）上平均准确率提升约+2.3~+2.99，测试时加权投票比多数表决提高多达+8.33；训练速度提升约1.6~1.7倍，且在多种基准上均保持或超越原方法。

**⚠️ 局限性**

限制主要包括：只在数学推理任务上验证；剪枝需先生成到中间检测长度（512）才可生效，导致生成阶段加速有限；对其他奖励类型或非可验证任务的适用性尚未验证。

---

## 17. More Than "Means to an End": Supporting Reasoning with Transparently Designed AI Data Science Processes

**arXiv ID:** 2603.24877 | [PDF](https://arxiv.org/pdf/2603.24877v1)

**作者:** Venkatesh Sivaraman `[一作]`, Jean Feng `[通讯]` (UC San Francisco)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出并实现了两个医学领域的 AI 数据科学工具（HACHI 与 Tempo），通过在工作流中加入可解释的中间产物支持非专家用户进行模型概念识别、数据提取和结果验证。

**💡 创新点**

创新点在于将中间产物设计为可供人类审阅的可解释表述（如可编辑的概念定义、可视化的概念标签、TempoQL 查询语言），使 AI 工作流从黑盒转化为“思维工具”，并通过交互迭代提升模型公平性和准确性。

**🔧 技术方法**

技术包括大语言模型（LLM）驱动的概念抽取与生成、基于 LLM 的智能助手、TempoQL 领域特定查询语言、交互式可视化界面与可编辑的概念标签。

**📊 数据集**

使用的主要数据集为儿科急诊医学的临床笔记（用于 TBI 预测）以及医院 EHR 记录（用于临时事件抽取），数据来源为 UCSF 和合作医院。

**📈 对比分析**

通过与专业团队的交互式评估显示，HACHI 的模型在多轮反馈后从 0.71 提升至 0.93 的 AUC，TempoQL 的查询准确率相比 SQL 提升 2.5 倍；但论文未给出统一的量化性能基准，仅以案例经验与专家评审为依据。

**⚠️ 局限性**

局限性包括中间产物仅覆盖工作流的一部分（概念抽取或数据提取），缺乏对探索性分析和预测建模的中间产物；缺乏系统的量化评估方法；并且用户对何时何频次需要审阅仍不确定。

---

## 18. Learning to Staff: Offline Reinforcement Learning and Fine-Tuned LLMs for Warehouse Staffing Optimization

**arXiv ID:** 2603.24883 | [PDF](https://arxiv.org/pdf/2603.24883v1)

**作者:** Kalle Kujanpää `[一作]` (Amazon), Shervin Malmasi `[通讯]` (Amazon)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文研究并实现了两种人工智能方法——离线强化学习（offline RL）与大语言模型（LLM）微调，用于实时仓储分拣系统的人员调度决策，并通过匹配的仿真环境评估其对吞吐量的提升效果。

**💡 创新点**

创新点在于：①提出Transformer‑GNN架构以高效建模高维时空工序动态，实现离线RL对大规模工位分配问题的可训练化；②构建手工仿真器与LLM交互的迭代偏好优化框架（DPO），通过模拟器产生的经理偏好实现LLM决策的持续改进。

**🔧 技术方法**

技术包括：离线强化学习（actor‑critic、行为克隆+微调）、Transformer‑GNN网络、LLM（Qwen2.5系列）提示与自动提示优化、监督微调（SFT）与直接偏好优化（DPO），以及手工与学习型仿真器。

**📊 数据集**

数据集为约31,366个工作班次的历史仓储调度轨迹（共501,856步，约42,000小时）作为训练集，8,228班次（131,648步）作为测试集。

**📈 对比分析**

在匹配的仿真环境中比较：离线RL在学习型仿真器上实现约2.4%吞吐提升；BC‑FT实现约2.1%；LLM在手工仿真器上提示策略表现不佳，SFT+两轮DPO可达+0.6%吞吐提升，略优于历史人类决策。

**⚠️ 局限性**

局限性包括：依赖仿真器导致的模拟器特定偏差；LLM方法需要大量标注与偏好数据，且对真实操作因素（成本、优先级）建模不足；未考虑跨设施的分布漂移与实时延迟等实际部署挑战。

---

## 19. On the Foundations of Trustworthy Artificial Intelligence

**arXiv ID:** 2603.24904 | [PDF](https://arxiv.org/pdf/2603.24904v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 20. MedOpenClaw: Auditable Medical Imaging Agents Reasoning over Uncurated Full Studies

**arXiv ID:** 2603.24649 | [PDF](https://arxiv.org/pdf/2603.24649v1)

**作者:** Weixiang Shen `[一作]` (Technical University of Munich), Jiazhen Pan `[通讯]` (Technical University of Munich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个可审计的运行时和全研究级医学影像基准，用来评估视觉语言模型在3D影像导航和工具使用方面的能力。

**💡 创新点**

创新点包括：① 将完整3D影像交互与可追溯执行轨迹结合，构建三轨评估框架（Viewer‑Only、Tool‑Use、Open‑Method）；② 发现“工具使用悖论”——即使给模型更强的工具，性能反而下降，揭示空间定位精度是瓶颈。

**🔧 技术方法**

使用了与 3D Slicer 的 REST 接口的受限运行时、MONAI 工具包、GPT‑5.x 和 Gemini LLM，以及记录动作的 API 进行实验。

**📊 数据集**

数据集包括 UCSF‑PDGM 多序列脑肿瘤 MRI（全研究级）和 NSCLC 肺 CT/PET（带病理注释）两种全研究级数据。

**📈 对比分析**

在相同提示下对比基线模型；Viewer‑Only 轨道最高准确率约 0.63，工具使用轨道出现准确率下降，说明当前模型缺乏精细空间定位能力。

**⚠️ 局限性**

局限性包括：仅覆盖两种模态、未涉及多回合对话或 EHR 数据、工具精度不足导致空间定位失误，以及仅评估固定提示下的模型表现。

---

## 21. KitchenTwin: Semantically and Geometrically Grounded 3D Kitchen Digital Twins

**arXiv ID:** 2603.24684 | [PDF](https://arxiv.org/pdf/2603.24684v1)

**作者:** Quanyun Wu `[一作]` (University of Waterloo), Yuhao Chen `[通讯]` (University of Waterloo)

**通讯引用:** 6129 | [OpenAlex ID](https://openalex.org/A5100321260)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文提出一种尺度感知的三维融合框架，将基于 Transformer 的全局点云与基于 Vision‑Language 的尺度恢复以及本地对象网格对齐，生成可操控的物体中心数字孪生。

**💡 创新点**

创新点在于利用 VLM 引导的物理锚点实现尺度恢复、在全局点云上施加几何约束注册，并采用 Manhattan 世界的碰撞消除，从而解决尺度歧义与坐标不一致的问题。

**🔧 技术方法**

技术包括 Pi‑Long Transformer 点云重建、VLM 尺度恢复、SAM3D 3D 网格生成、粗细化 TrICP 注册以及垂直对齐、Manhattan 约束和碰撞检测。

**📊 数据集**

使用了新发布的 KitchenTwin 数据集，该数据集包含真实厨房场景的稠密点云、物体掩码和已标注的尺度信息。

**📈 对比分析**

与 SAM3D 基线对比，采用 NVSS‑IoU 评估，平均 IoU 从 0.0246 提升至 0.5715，且 TrICP/碰撞约束使物体定位误差降低至几厘米级。

**⚠️ 局限性**

局限在于依赖 VLM 的网页检索来获取尺度锚点，可能对罕见物体或无锚点场景失效；此外，训练成本和对长序列的实时性仍有待改进。

---

## 22. LogSigma at SemEval-2026 Task 3: Uncertainty-Weighted Multitask Learning for Dimensional Aspect-Based Sentiment Analysis

**arXiv ID:** 2603.24896 | [PDF](https://arxiv.org/pdf/2603.24896v1)

**作者:** Baraa Hikal `[一作]` (University of Göttingen), Bela Gipp `[通讯]` (University of Göttingen)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了LogSigma系统，利用自学习的同方差不确定性权重结合语言特定编码器和多种种子集成，以在Dimensional Aspect-Based Sentiment Analysis任务中实现对Valence和Arousal回归的自动平衡。

**💡 创新点**

创新点在于将同方差不确定性学习用于多任务回归，自动调节Valence和Arousal的损失权重，同时发现语言预训练类型决定任务难度，并通过学习的σ²作为语言相似度诊断工具。

**🔧 技术方法**

使用了语言特定的预训练Transformer编码器、双分支回归头、可学习的logσ²权重、AdamW优化、fp16训练以及三种种子平均集成。

**📊 数据集**

在SemEval-2026 Task 3的Track A（六语言用户评论）和Track B（五语言立场检测）共15个数据集上进行评测，包括英语、日语、俄语、塔塔尔语、乌克兰语、中文、德语、尼日利亚皮钦语、斯瓦希里语等。

**📈 对比分析**

与单一多语言模型及固定损失权重的基线相比，LogSigma在五个提交数据集上获得第一名，RMSE显著下降，最优语言的性能提升约8–28%。

**⚠️ 局限性**

局限在于低资源语言的编码器效果受限、模型对新语言的冷启动成本较高、仅考虑同方差不确定性，未探索异方差或更复杂的任务相互依赖。

---

## 23. Generative Adversarial Perturbations with Cross-paradigm Transferability on Localized Crowd Counting

**arXiv ID:** 2603.24821 | [PDF](https://arxiv.org/pdf/2603.24821v1)

**作者:** Alabi Mehzabin Anisha `[一作]` (University of South Florida), Sriram Chellappan `[通讯]` (University of South Florida)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种跨范式（密度图与点回归）的对抗攻击框架，能在保持视觉隐蔽性的同时显著破坏多种热门人群计数模型。

**💡 创新点**

创新点在于：①结合多任务损失（密度抑制、Logit抑制、频域、Grad‑CAM、TV 等）实现跨范式攻击；②利用共享骨干网络的通用弱点，提升攻击可转移性；③通过生成式 UNet 直接学习扰动映射，减少迭代时间。

**🔧 技术方法**

核心技术包括：轻量级 UNet 生成器、梯度引导的 Perturbation Loss、频域约束、logit 与峰值抑制损失、Grad‑CAM 指导的稀疏扰动、以及多任务联合优化。

**📊 数据集**

使用了 ShanghaiTech Part A（SHHA）和 UCF‑QNRF 两个公开人群计数数据集进行训练与评估。

**📈 对比分析**

与 9 种现有对抗方法（包括 DI²FGSM、Admix、GE‑AdvGAN、DiffAttack、PAP 等）相比，攻击在 MAE 上提升约 7–9 倍，PSNR 约 19–20 dB，保持良好视觉质量；在跨范式可转移率上最高可达 1.69，显示优越的攻击效果与转移性。

**⚠️ 局限性**

局限性：攻击仍依赖于模型的共享骨干，对极端稀疏场景效果有限；在物理世界中的可实施性尚未验证；若目标模型采用更强的防御机制（如对抗训练、输入预处理），攻击效果可能下降。

---

## 24. Enhancing Online Support Group Formation Using Topic Modeling Techniques

**arXiv ID:** 2603.24765 | [PDF](https://arxiv.org/pdf/2603.24765v1)

**作者:** Pronob Kumar Barman `[一作]` (University of Maryland, Baltimore County), James Foulds `[通讯]` (University of Maryland, Baltimore County)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了两种新型机器学习模型——Group-specific Dirichlet Multinomial Regression (gDMR) 与 Group-specific Structured Topic Model (gSTM)，用于自动化在线健康社区中的支持小组生成，提升个性化和语义连贯性。

**💡 创新点**

创新点在于将用户文本内容、人口统计特征与网络交互的节点嵌入共同融入主题建模框架，通过组特定参数和稀疏正则化实现更精准的群组划分与主题解释。

**🔧 技术方法**

使用的技术包括Dirichlet Multinomial Regression、Structured Topic Model、节点嵌入（Node2Vec）、Gibbs采样、变分EM、稀疏拉普拉斯先验以及Logistic Normal 分布等。

**📊 数据集**

实验数据来自 MedHelp.org 的 200 万条用户发帖和 800 万条回复，包含性别、年龄、地区等人口统计信息与互动网络。

**📈 对比分析**

与 LDA、传统 DMR、STM 等基准模型比较，gDMR 在加入节点嵌入后在保留对数似然和 UMass 主题连贯性上分别提升约 15% 与 40%；gSTM 在主题连贯性上优于 STM，整体表现均显著优于基线。

**⚠️ 局限性**

局限性包括：对人口统计与交互数据的偏倚可能导致公平性问题；模型在大规模实时应用时计算成本高；缺乏基于用户体验的评估；隐私保护与差异化算法的进一步完善仍需研究。

---

## 25. Reconstructing Spiking Neural Networks Using a Single Neuron with Autapses

**arXiv ID:** 2603.24692 | [PDF](https://arxiv.org/pdf/2603.24692v1)

**作者:** Wuque Cai `[一作]` (University of Electronic Science and Technology of China), Daqing Guo `[通讯]` (University of Electronic Science and Technology of China)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种基于时间延迟自突触的漏电积分与发放（TDA‑LIF）单神经元，并将其通用化为可实现回路计算、MLP和卷积结构的时空展开式SNN（TDA‑SNN）

**💡 创新点**

创新点在于将生物神经元中的自突触延迟机制转化为可编程的时间反馈，构造单神经元可实现多种典型SNN拓扑，并提出原型学习框架以充分利用时空发放模式

**🔧 技术方法**

主要技术包括时间延迟自突触的数学建模、时空展开映射、原型学习与 surrogate gradient 反向传播、以及基于 PyTorch 的端到端训练

**📊 数据集**

实验使用了 DEAP、SHD（回路计算）、MNIST、Fashion‑MNIST、DVS Gesture（MLP）以及 DVS Gesture 与 CIFAR‑10（卷积）等公开数据集

**📈 对比分析**

通过与标准多层SNN基线在相同模型规模下对比，TDA‑SNN 在回路与 MLP 任务中可达或接近基线性能，同时显著降低神经元数量与状态存储；在卷积任务中性能略逊，但展示了空间-时间权衡的可调性

**⚠️ 局限性**

主要局限包括单神经元卷积实现的准确率与训练收敛性不足、延迟反馈导致的时间延迟和优化难度、以及极限压缩时对单个神经元的计算负载显著增加

---

## 26. BCMDA: Bidirectional Correlation Maps Domain Adaptation for Mixed Domain Semi-Supervised Medical Image Segmentation

**arXiv ID:** 2603.24691 | [PDF](https://arxiv.org/pdf/2603.24691v1)

**作者:** Bentao Song `[一作]` (Southwest University of Science and Technology), Qingfeng Wang `[通讯]` (Southwest University of Science and Technology)

**通讯引用:** 3250 | [OpenAlex ID](https://openalex.org/A5100419920)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种针对混合域半监督医学图像分割的双向相关图域适配框架BCMDA，解决域差异与确认偏差问题。

**💡 创新点**

创新点在于利用双向相关图生成虚拟域进行跨域知识迁移，并结合可学习的余弦相似度原型分类器实现双向原型对齐和伪标签校正。

**🔧 技术方法**

采用双向相关图、MixUp、CutMix、可学习的CosSim原型分类器、伪标签校正、以及Mean Teacher框架等技术。

**📊 数据集**

在三个公开多域医学数据集（Fundus、Prostate、M&Ms）和Synapse 3D数据集上进行评估。

**📈 对比分析**

与多种SSMS和UDA方法对比，BCMDA在Dice、Jaccard、95HD、ASD等指标上均优于SOTA，并在极少标签样本下保持强劲性能。

**⚠️ 局限性**

局限在于虚拟域图像合成可能产生语义错误或纹理模糊，导致模型学习不稳定，未来需进一步提升合成质量和泛化能力。

---

## 27. Self-Supervised Learning for Knee Osteoarthritis: Diagnostic Limitations and Prognostic Value of Uncurated Hospital Data

**arXiv ID:** 2603.24903 | [PDF](https://arxiv.org/pdf/2603.24903v1)

**作者:** Haresh Rengaraj Rajamohan `[一作]`, Cem M. Deniz `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e15e3743-5ee0-4d5f-813d-d146868082fc` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在铜氧薄片与聚苯乙烯微球界面上演示并表征了新型极化子（Y_1 极化子）的形成，探讨了其光学特性。

**💡 创新点**

创新点在于首次观察到四阶偶极子激子极化子（quadrupole exciton polariton）并通过实验展示其形成机制。

**🔧 技术方法**

采用光学实验技术（如光谱测量、散射光谱）对界面结构进行探测，并使用标准的光学仪器进行数据采集。

**📊 数据集**

论文未给出具体公开数据集，主要使用实验中获得的光学测量数据。

**📈 对比分析**

未进行与现有方法的对比实验，也未给出性能指标或量化评估，主要是实验观察与初步分析。

**⚠️ 局限性**

研究的局限性包括：实验条件受限于样品制备与测量精度；缺乏理论模型的深入推导；未探讨该极化子在实际应用中的可行性和稳定性。

---

## 28. GraphER: An Efficient Graph-Based Enrichment and Reranking Method for Retrieval-Augmented Generation

**arXiv ID:** 2603.24925 | [PDF](https://arxiv.org/pdf/2603.24925v1)

**作者:** Ruizhong Miao `[一作]` (Oracle AI), Dan Roth `[通讯]` (Oracle AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `3f18e8e3-0266-457c-8567-9039b6d2394d` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出GraphER框架，利用离线图信息丰富与在线图重排序提升检索增强生成（RAG）系统的检索完整度。

**💡 创新点**

创新点在于：①无需知识图谱，直接在检索候选上构建结构/概念/上下文图；②设计Graph Cohesive Smoothing（GCS）算法，克服PPR对hub节点的偏好；③可选用Graph Attention Network（GAT）捕获高阶节点交互。

**🔧 技术方法**

技术包括：离线元数据提取与图边构造、BM25+向量混合检索、图构建与邻接矩阵、GCS与PPR迭代推理、GAT消息传递及全连接后处理。

**📊 数据集**

使用了表检索数据集Spider 1.0、Bird、Beaver；多跳QA数据集HotpotQA、2WikiMultihopQA、MuSiQue；分块文档检索数据集BEIR‑NQ。

**📈 对比分析**

通过PR@K（含多相关物件）和LLM生成答案的准确率与基线（BM25+embedding、PPR、GCS、GAT）对比，实验表明GraphER‑GCS在18项设置中全提升，GraphER‑GAT在多数设置上最高；GCS在鲁棒性和可解释性上优于PPR。

**⚠️ 局限性**

局限性包括：需要离线预处理；仅针对三类相似度，其他领域需自行定义；在小样本或候选数大时GAT性能下降；构建邻接矩阵为O(n²)，对极大候选集可能开销较大。

---

## 29. CORA: A Pathology Synthesis Driven Foundation Model for Coronary CT Angiography Analysis and MACE Risk Assessment

**arXiv ID:** 2603.24847 | [PDF](https://arxiv.org/pdf/2603.24847v1)

**作者:** Jinkui Hao `[一作]` (Northwestern University), Bo Zhou `[通讯]` (Northwestern University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

研发了CORA，一个基于3D自监督预训练的冠状动脉CT基础模型，能够统一完成斑块分类、狭窄检测、血管分割以及30天MACE预测；

**💡 创新点**

创新点在于：①采用病理导向的合成病变引擎，将稀疏的冠脉病变直接嵌入未标注体积中，改写预训练目标从全局解剖重建转向病变检测；②多窗口四通道输入一次性捕获软组织、血管、钙化等不同衰减范围，提升低衰减非钙化斑块与高衰减钙化斑块的区分；③实现单一基础模型可无缝迁移至多任务，避免多阶段管道与误差累积；

**🔧 技术方法**

技术包括：3D Residual U-Net架构、合成病变生成与四窗口输入、Tversky+Focal双重损失以处理极端稀疏病变、Grad‑CAM可解释性、Qwen‑7B文本编码器融合的多模态MACE预测框架、Poisson噪声与几何增强等；

**📊 数据集**

数据集：12,801份未标注CCTA用于预训练；4,883份标注CCTA用于斑块分类与MACE（2,220训练+555内部验证，2,108外部验证）；348份用于狭窄检测；ImageCAS 1,000份用于血管分割；所有数据来自9家医院的跨中心临床数据库；

**📈 对比分析**

与从零训练、MAE、VolumeFusion、VoCo等基线比较；斑块分类AUROC（钙化0.93–0.94，非钙化0.71–0.79）显著优于基线；狭窄检测F1 0.684（vs. 0.549基线，MAE 0.575等）；血管分割Dice 0.654（vs. 0.629基线，MAE 0.642等）；MACE预测AUC 0.756（外部）和0.75（内部），明显高于基线0.59–0.62和其他3D模型；同时在多中心外部验证中保持稳定性能，表明强泛化；

**⚠️ 局限性**

局限性包括：未在无症状筛查人群验证；回溯性设计可能导致MACE标签噪声；缺乏药物使用信息；合成病变虽多样但未完全再现真实斑块的复杂形态；预训练模型对非冠脉病变的敏感性需进一步校正。

---

## 30. Synthetic Rewriting as a Quality Multiplier: Evidence from Portuguese Continued Pretraining

**arXiv ID:** 2603.24826 | [PDF](https://arxiv.org/pdf/2603.24826v1)

**作者:** Thales Sales Almeida `[一作]` (Institute of Computing), Hélio Pedrini `[通讯]` (Institute of Computing)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

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

## 31. Associative Memory using Attribute-Specific Neuron Groups-2: Learning and Sequential Associative Recall between Cue Neurons for different Cue Balls

**arXiv ID:** 2603.24910 | [PDF](https://arxiv.org/pdf/2603.24910v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620`

---

## 32. Unbiased Multimodal Reranking for Long-Tail Short-Video Search

**arXiv ID:** 2603.24975 | [PDF](https://arxiv.org/pdf/2603.24975v1)

**作者:** Wenyi Xu `[一作]` (Zhejiang University), Yi Zhang `[通讯]` (Kuaishou Technology)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种基于大语言模型的多模态重排框架，专门解决短视频长尾搜索中的质量偏差问题。

**💡 创新点**

创新点在于：①通过多模态证据驱动的监督细调构造高质量标注，②引入成对偏好学习实现部分排序；③将经验得分无行为依赖地注入重排模型，实现对曝光、热度与跨模态不匹配的全方位校正。

**🔧 技术方法**

主要技术包括：大语言模型（如Qwen2.5‑VL‑32B、Qwen3‑4B）生成多维质量评估；多模态输入（标题、OCR、ASR、封面及关键帧）；监督细调+成对偏好细调；经验得分驱动的两阶段重排（预训练增强+基于GRPO的页面级对齐）。

**📊 数据集**

使用了约187,000条查询‑视频样本（SFT阶段）和348,000条成对偏好对（PFT阶段）的自构造多模态数据集，全部来自千川短视频搜索日志，且在时间和查询/视频级别上严格拆分。

**📈 对比分析**

与GPT‑4o（文本/多模态点对点与列表）、RankGPT、BGE‑m3等基线在NDCG@1/5/10、AUC、人工偏好评价中均取得显著提升（例如NDCG@10提升至0.930，人工优选率+11.7%），在线A/B测试在长尾查询上分别提升IQRR‑1.28%、CTR‑1.24%、LVR‑1.67%，并在整体流量中也取得小幅正向影响。

**⚠️ 局限性**

局限性主要在于：①对大型LLM和GPU计算资源依赖较高；②仍需人工监督验证生成标注的质量；③对曝光偏差的校正在大规模上线后可能受页面级策略约束；④实验聚焦于短视频，跨领域迁移尚待验证。

---

## 33. Training LLMs for Multi-Step Tool Orchestration with Constrained Data Synthesis and Graduated Rewards

**arXiv ID:** 2603.24709 | [PDF](https://arxiv.org/pdf/2603.24709v1)

**作者:** Cheng Jiayang `[一作]` (Amazon Inc.), Yangqiu Song `[通讯]` (Amazon Inc.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于LLM的多步工具编排训练框架，解决参数错误和依赖链问题。

**💡 创新点**

创新点在于构建真实API响应缓存环境、约束数据合成管道以及分解成R_atomic与R_orch的分级奖励机制。

**🔧 技术方法**

使用强化学习（GRPO）、Deterministic Cache环境、AST与语义验证的奖励设计以及LLM生成查询的约束合成。

**📊 数据集**

主要使用ComplexFuncBench benchmark以及覆盖40种API的100k+真实API响应缓存。

**📈 对比分析**

与零-shot和仅SFT等方法对比，RL训练在转折准确率上提升约14–17%，在转折与调用准确率上均显著优于单一奖励或传统SFT。

**⚠️ 局限性**

局限在于奖励分解仍需手工设计，合成数据可能缺乏多样性，且对极深层次或并行依赖的处理仍有限。

---

## 34. CVA: Context-aware Video-text Alignment for Video Temporal Grounding

**arXiv ID:** 2603.24934 | [PDF](https://arxiv.org/pdf/2603.24934v1)

**作者:** Sungho Moon `[一作]` (Daegu Gyeongbuk Institute of Science and Technology), Sunghoon Im `[通讯]` (Daegu Gyeongbuk Institute of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种 Context‑aware Video‑text Alignment (CVA) 框架，用于实现视频与文本在时间维度上的精准对齐，且能抵抗无关背景的干扰。

**💡 创新点**

创新点包括三项：① Query‑aware Context Diversification (QCD)——基于 CLIP 相似度的查询感知上下文增广；② Context‑invariant Boundary Discrimination (CBD) loss——针对关键时间边界的对比学习；③ Context‑enhanced Transformer Encoder (CTE)——窗口自注意与双向交叉注意的层次编码。

**🔧 技术方法**

主要技术包括 CLIP 预训练特征、窗口自注意力、双向交叉注意、可学习查询、对比学习以及多任务损失（Moment Retrieval、Highlight Detection 与 CBD）。

**📊 数据集**

在 QVHighlights、Charades‑STA 与 TACoS 三个公开数据集上进行实验。

**📈 对比分析**

与最新基线相比，CVA 在 QVHighlights 上 Recall@1(0.7) 提升约 5 分，Highlight Detection mAP 提升 4.1 分；在 Charades‑STA 上 R@1(0.7) 提升 1.4 分；在 TACoS 上 mIoU 提升 1.8 分，整体性能均达到或超过现有最优。

**⚠️ 局限性**

局限性在于对 CLIP 相似度的依赖，可能在语义模糊或跨域文本时效果受限；此外，窗口大小与查询数量的超参数需手工调优，模型训练成本较高。

---

## 35. Confidence-Based Mesh Extraction from 3D Gaussians

**arXiv ID:** 2603.24725 | [PDF](https://arxiv.org/pdf/2603.24725v1)

**作者:** Lukas Radl `[一作]` (Graz University of Technology), Markus Steinberger `[通讯]` (Graz University of Technology)

**通讯引用:** 2279 | [OpenAlex ID](https://openalex.org/A5014594342)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种基于自监督置信度的3D高斯展开方法，用于高质量网格提取。

**💡 创新点**

创新点包括可学习的 per‑primitive 置信度平衡光度与几何损失、颜色与法线方差损失、改进的 D‑SSIM 解耦外观模型。

**🔧 技术方法**

使用技术包括 3D Gaussian Splatting、alpha‑blending、D‑SSIM、颜色/法线方差损失、confidence‑driven densification 以及改进的外观嵌入。

**📊 数据集**

使用的数据集为 Tanks & Temples、ScanNet++、Mip‑NeRF 360 等。

**📈 对比分析**

与 GOF、SOF、MiLo 等无界网格方法以及 PGSR、QGS 等有界方法比较，取得 F1 得分 0.521 并实现 <20 分钟的实时优化。

**⚠️ 局限性**

局限性在于对密集视角捕获依赖较强，稀疏视角下仍可能出现空洞，远景细节不足。

---

## 36. Zero-Cost NDV Estimation from Columnar File Metadata

**arXiv ID:** 2603.24606 | [PDF](https://arxiv.org/pdf/2603.24606v1)

**作者:** Claude Brisson `[一作]` `[通讯]`, Claude Brisson

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

利用列式存储文件的字典编码大小和行组最小/最大统计信息，在不访问数据页的前提下，零成本估计列的不同值数量。

**💡 创新点**

提出了两种互补的估计器：字典大小反演与基于 min/max 的优惠券收集器反演，并通过轻量级分布检测器动态选择。

**🔧 技术方法**

使用 Newton-Raphson 迭代求解字典大小方程、优惠券收集模型、重叠率和单调性指标进行分布检测，以及 HyperLogLog 计数。

**📊 数据集**

在 VoltronData 的生产工作负载（Theseus）中，对真实 Parquet 数据集进行验证；实验数据已丢失，计划在公开基准上复现。

**📈 对比分析**

与真实 NDV 进行对比，误差普遍低于 10%，对分布良好的列特别准确；通过 O(n) 线性时间、O(1) 额外空间实现。

**⚠️ 局限性**

仅适用于支持字典编码和行组/分区 min/max 统计的格式；在纯文本编码或极度排序的数据上估计可能低估；依赖元数据质量，缺失元数据时无法工作。

---

## 37. Rethinking Failure Attribution in Multi-Agent Systems: A Multi-Perspective Benchmark and Evaluation

**arXiv ID:** 2603.25001 | [PDF](https://arxiv.org/pdf/2603.25001v1)

**作者:** Yeonjun In `[一作]` (KAIST), Chanyoung Park `[通讯]` (KAIST)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出多视角失败归因概念，构建 MP-Bench benchmark 及其评估协议，并在多种 LLM 上进行大规模实验。

**💡 创新点**

创新点在于把失败归因从单一确定性标签转变为多视角可解释输出；首次为此任务设计专家级多视角标注和排名/推理双重评估。

**🔧 技术方法**

利用 LLM（GPT‑4.1、GPT‑5.1、Claude‑Sonnet‑4.5 等）进行全局推理，采用多次采样、温度调节、All‑at‑Once 方式，评估指标为 nDCG 与 LLM‑as‑Judge 分数。

**📊 数据集**

使用自定义的 MP‑Bench 数据集，包含 289 条执行日志（169 手工、120 自动），涵盖 MAgenticOne、CaptainAgent、GAIA、AssistantBench 等多种 MAS 框架。

**📈 对比分析**

通过对比 nDCG@5、nDCG@full（指数/线性）和 GPT‑Judge 评分进行多视角与推理质量评估；实验显示多次采样与跨模型组合能显著提升性能，远优于随机基线。

**⚠️ 局限性**

局限性：仅覆盖通用助手任务，数据规模受专家标注成本限制，缺乏对更专业领域和更大规模自动化标注方法的验证。

---

## 38. SentinelAI: A Multi-Agent Framework for Structuring and Linking NG9-1-1 Emergency Incident Data

**arXiv ID:** 2603.24856 | [PDF](https://arxiv.org/pdf/2603.24856v1)

**作者:** Kliment Ho `[一作]` (University Of California San Diego), Ilya Zaslavsky `[通讯]` (University Of California San Diego)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了 SentinelAI 多代理框架，利用 EIDO、IDX 与 Geocoding 代理将异构紧急响应数据转换为 NENA EIDO-JSON 标准化格式，并实现实时关联与空间增强；

**💡 创新点**

创新点在于将 EIDO 视为可增量更新的事件源，采用代理化拆分、语义相似度与多维度匹配的动态关联算法，以及针对非正式地点描述的本地化地理编码；

**🔧 技术方法**

使用自然语言处理模型进行信息抽取与代码映射、基于时间、空间、语义的相似度评分算法、外部地图服务与本地地标数据库的地理编码技术，以及 FME 平台的读取/写入组件；

**📊 数据集**

使用真实场景数据（NWS 洪水预警文本、San Diego 联合先驱新闻稿），以及通用的紧急响应日志、CAD 系统数据作为原始输入；

**📈 对比分析**

目前未给出量化性能指标，示例仅展示功能演示与互操作性（通过 FME 与传统系统对接），但缺乏大规模基准测试与实时处理延迟评估；

**⚠️ 局限性**

局限性包括：缺乏对算法性能的系统评估；对 AI 模型的依赖导致准确性与可解释性受限；阈值配置需人工调校；本地化地理编码依赖人工维护的地标库，易受更新滞后影响。

---

## 39. TIGFlow-GRPO: Trajectory Forecasting via Interaction-Aware Flow Matching and Reward-Driven Optimization

**arXiv ID:** 2603.24936 | [PDF](https://arxiv.org/pdf/2603.24936v1)

**作者:** Xuepeng Jing `[一作]` (Tianjin University), Jianguo Wei `[通讯]` (Tianjin University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `40105733-5154-44cd-8090-a8cab9e64b07` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了两阶段的轨迹预测框架TIGFlow-GRPO，先用带目标感知交互图的Conditional Flow Matching生成多模态轨迹，再通过ODE→SDE随机化和GRPO后训练，使生成轨迹符合社会交互和场景物理约束。

**💡 创新点**

创新点包括①利用TIG-GAT实现视野感知的局部交互建模；②将确定性ODE转换为SDE，引入可探索的随机策略；③设计视野社会奖励与地图语义奖励的组合，直接在后训练阶段对轨迹进行行为对齐；④通过GRPO实现无价值网络的政策更新，提升轨迹的社会合规性和物理可行性。

**🔧 技术方法**

技术手段包括Conditional Flow Matching、Trajector-Interaction-Graph Attention (TIG-GAT)、Group Relative Policy Optimization (GRPO)、SDE采样、签名距离场（SDF）与视野社会奖励等。

**📊 数据集**

使用了公开基准 ETH/UCY（5 个场景）和 Stanford Drone Dataset (SDD) 进行实验。

**📈 对比分析**

与 MoFlow、GroupNet、MID、SingularTrajectory、DD-MDN 等基线对比，TIGFlow-GRPO 在 ADE/FDE 上均实现了最优或次优表现，长时域预测更稳健，碰撞率显著降低，尤其在 SDD 上实现了最高精度。

**⚠️ 局限性**

局限性包括：对训练数据场景的依赖较强，缺少对动态障碍物的建模；后训练需要额外采样和奖励计算，增加训练成本；未充分评估实时推理性能和大规模多主体扩展性。

---

## 40. WAFT-Stereo: Warping-Alone Field Transforms for Stereo Matching

**arXiv ID:** 2603.24836 | [PDF](https://arxiv.org/pdf/2603.24836v1)

**作者:** Yihan Wang `[一作]` (Princeton University), Jia Deng `[通讯]` (Princeton University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了完全基于warping的立体匹配方法WAFT‑Stereo，消除成本体积，采用分类+回归的两步迭代更新，显著提升精度与速度。

**💡 创新点**

创新点包括：①证明成本体积非必要；②引入离散分类先导粗估后细化回归；③用LoRA微调预训练编码器、移除U‑Net、改用ResNet高分辨率块；④在立体匹配上实现与光流同类WAFT框架的迁移。

**🔧 技术方法**

技术手段包括：warp‑based iterative refinement、ViT‑S + DPT上采样、Mixture‑of‑Laplace (MoL) 损失、ResNet高分辨率模块、LoRA、分类概率分布与软argmax、双向光流相似的back‑warping。

**📊 数据集**

使用大规模合成数据集（SceneFlow、FSD、TartanAir、FallingThings、Virtual‑KITTI等）进行零射击训练，随后在实景数据集（KITTI、Middlebury、InStereo2K、Boosted等）微调。

**📈 对比分析**

在ETH3D、KITTI‑2012/2015、Middlebury公开排行榜上，零射击BP‑0.5 81%误差下降，BP‑2、D1、RMSE均处于或逼近首位；实时qHD输入可达21FPS，速度比FoundationStereo 1.8–6.7×，比S2M2‑XL 1.8×，在精度‑效率曲线中位于Pareto前沿。

**⚠️ 局限性**

局限性：在极端光照或纹理对比度高的场景仍易出错；单一SceneFlow训练导致过拟合；迭代次数越多延迟升高，尚需更高效的更新策略。

---

## 41. Contrastive Learning Boosts Deterministic and Generative Models for Weather Data

**arXiv ID:** 2603.24744 | [PDF](https://arxiv.org/pdf/2603.24744v1)

**作者:** Nathan Bailey `[一作]` `[通讯]` (Imperial College London), Nathan Bailey (Imperial College London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种端到端的自监督对比学习框架 SPARTA，能够在 ERA5 气象数据上学习稀疏数据增强的低维嵌入，并通过 decoder 将嵌入映射回原空间，支持时间序列预测、潜在扩散和分类等下游任务。

**💡 创新点**

创新点包括：①将 decoder 集成到对比学习管道，实现端到端训练；②设计稀疏数据增强对比（SPARTA）和硬负样本采样方案；③引入周期一致性损失提升嵌入空间平滑度；④提出基于图卷积网络的多模态融合方法，结合领域先验；⑤在多任务评估中对比 autoencoder，展示显著性能提升。

**🔧 技术方法**

使用技术包括：SimCLR 对比学习、组归一化、卷积自编码器、LSTM 序列预测、条件潜在扩散 U-Net、硬负采样、周期一致性损失、Graph Neural Network（GNN）融合、self‑attention 融合、时间敏感批采样、温度调参、交替训练与 alpha 衰减等。

**📊 数据集**

采用 ERA5 再分析数据集（5 变量：温度、地势位势、水平/垂直风速、比湿），空间分辨率 64×32，时间覆盖 1959–2023 年，按 60/20/20 的时间顺序划分训练/验证/测试集。

**📈 对比分析**

与基准自动编码器相比，SPARTA 在 3 个下游任务上均表现更优：预测 RRMSE 最高可提升 32%；潜在扩散的标准差下降 23%，密度得分提升；分类损失下降 28%。实验对比表明，早期融合+GNN 在预测任务上最好，self‑attention 在分类任务上略胜一筹。

**⚠️ 局限性**

局限性：仅在最低分辨率 64×32 的 ERA5 数据上验证；只使用 5 个变量，未扩展至更高分辨率或更多模态；未在其他气象/地球科学数据集上评估泛化能力；对稀疏数据比例和噪声的鲁棒性需进一步研究。

---

## 42. AIP: Agent Identity Protocol for Verifiable Delegation Across MCP and A2A

**arXiv ID:** 2603.24775 | [PDF](https://arxiv.org/pdf/2603.24775v1)

**作者:** Sunil Prakash `[一作]` `[通讯]` (Indian School of Business), Sunil Prakash (Indian School of Business)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

本文设计并实现了代理身份协议（Agent Identity Protocol, AIP）以及调用绑定能力令牌（Invocation-Bound Capability Tokens, s），实现了身份验证、可收窄授权、跨协议绑定和委托链完整追溯，支持单跳和多跳委托。

**💡 创新点**

创新点在于：①将身份、权限收窄与结果证明融合为单链令牌；②提出两种轻量化格式——JWT压缩模式用于单跳，Biscuit链式模式用于多跳；③实现跨MCP、A2A、HTTP的协议绑定，并在令牌中嵌入委托深度、上下文与Datalog策略；④提供完整的攻击评估和性能基准，证明在LLM推理场景下几乎不增加延迟。

**🔧 技术方法**

技术包括：Ed25519公钥签名、Biscuit token（追加块与Datalog策略）、JWT（EdDSA）签名、Python与Rust实现、HTTP header绑定、Datalog policy profiles（Simple/Standard/Advanced）、以及跨语言互操作测试。

**📊 数据集**

实验数据来源于：约2000台服务器的安全扫描、10次Gemini 2.5 Flash推理实验、1000次单跳/链式令牌微基准、100次多跳深度基准、600次攻击评估；未使用公开数据集。

**📈 对比分析**

评估方法：微基准测验验证验证速度和令牌大小；真实HTTP部署对比无认证与链式令牌的时延；LLM推理对比评估整体延迟；攻击评估对比不同方案的拒绝率。结果显示：compact模式Rust验证0.049 ms，Python 0.189 ms；链式模式depth5 0.745 ms；HTTP真实部署仅增加0.22 ms；LLM推理中AIP占0.086%总延迟；攻击测试600次全部拒绝，传统JWT未能捕获深度违规与上下文空白攻击。

**⚠️ 局限性**

局限性包括：①缺乏大规模生产多租户部署验证；②高级Datalog策略可能导致DoS，需谨慎使用；③完成块默认自报，缺乏独立验证；④当前版本不支持撤销机制；⑤DNS基身份仍易受劫持，需手动验证签名；⑥仅支持Ed25519，未覆盖后量子算法；⑦v2计划引入更完善的多协议绑定和撤销支持。

---

## 43. Sketch2Simulation: Automating Flowsheet Generation via Multi Agent Large Language Models

**arXiv ID:** 2603.24629 | [PDF](https://arxiv.org/pdf/2603.24629v1)

**作者:** Abdullah Bahamdan `[一作]` (Imperial College London), Antonio del Rio Chanona `[通讯]` (Imperial College London)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了一个多代理大语言模型系统，将流程图直接转换为可执行的Aspen HYSYS仿真模型。

**💡 创新点**

创新点在于：①将视觉理解、结构恢复、代码生成和多级验证整合为多层协作的多代理架构，实现端到端的图纸到可执行模型自动化；②引入结构化中间表示（IR）和多级验证，显著降低幻觉与错误传播；③在同一框架中同时处理多模态解析与代码生成，弥合了图像理解与文本/代码生成之间的鸿沟。

**🔧 技术方法**

使用的技术包括：多模态 LLM（Gemini 3 Flash）进行图像+文本推理；检索增强生成（RAG）匹配 HYSYS 组件；代码生成 LLM（Qwen2.5‑Coder‑7B 与 Qwen3‑Coder‑30B）生成 COM 接口脚本；规则驱动归一化与结构校正；LangChain/LLM‑Graph 进行工作流编排；Python + Aspen HYSYS COM 接口实现模型实例化与执行；以及多模态 OCR、图像分割、语义解析等底层技术。

**📊 数据集**

实验数据集为四个化工案例流程图（脱盐、Merox 甜化、原油分馏、芳烃生产），这些图纸均来自公开文献与内部资料，涵盖从简易到工业级密集结构的多样性。

**📈 对比分析**

评估方法：采用单元一致性（UC）、流线一致性（SC）、连接一致性（CC）和物料一致性（MC）四项 F1 指标，并对执行稳定性与错误定位进行定性分析。实验结果显示，前两个案例实现了 F1=1.00；第三、第四案例在连接和流线维度分别保持 0.93–0.98 的高水平；消融实验证明每个子模块对整体性能都有显著贡献；与其他多模态模型（Qwen‑VL、Qwen‑3.5）对比，Gemini 3 Flash 在所有指标上表现最佳。

**⚠️ 局限性**

局限性包括：①对图纸质量和隐式符号高度敏感，复杂回路与密集交叉会导致解析误差；②Aspen HYSYS 的 COM 接口限制了某些单元的动态配置，导致接口层面错误；③多模态推理计算量大，依赖云端或高性能 GPU，部署成本较高；④系统目前针对 Aspen HYSYS 设计，迁移到其他仿真平台需要重新定义中间表示与接口规则。

---

## 44. Energy-Efficient Hierarchical Federated Anomaly Detection for the Internet of Underwater Things via Selective Cooperative Aggregation

**arXiv ID:** 2603.24648 | [PDF](https://arxiv.org/pdf/2603.24648v1)

**作者:** Kenechi Omeke `[一作]` (University of Glasgow), Muhammad Ali Imran `[通讯]` (University of Glasgow)

**通讯引用:** 32682 | [OpenAlex ID](https://openalex.org/A5100331811)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c84dae5d-5273-4348-85a7-b44cb586b4df` `fede83ac-7505-405f-ab37-e7284695c47f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了面向海底物联网的能量高效分层联邦学习框架，用于分布式异常检测；

**💡 创新点**

创新点在于将可达性感知的传感器‑雾节点关联、压缩模型上传、以及只在必要时进行雾节点间协同聚合相结合，兼顾网络参与度、能耗和检测精度；

**🔧 技术方法**

采用分层联邦学习（Hierarchical Federated Learning）、Top‑K稀疏化+量化压缩、基于物理的声学传播与能耗模型、以及自动编码器异常检测模型；

**📊 数据集**

在三类真实数据集上验证：Server Machine Dataset (SMD)、Soil Moisture Active Passive (SMAP)、Mars Science Laboratory (MSL)，并在大规模合成实验中评估；

**📈 对比分析**

与传统平面联邦学习（FedAvg、FedProx）以及三种分层策略（无协同、始终协同、选择性协同）进行对比，发现分层方法在保持完整网络参与的同时，F1 分数与平面方法相近，而压缩上传可将能耗降低 70–95%，选择性协同在不显著牺牲精度的前提下将雾节点间通信能耗降低 30–33%；

**⚠️ 局限性**

局限性包括：仅考虑静态雾节点（未研究移动或异步场景）、仅使用自编码器作为异常检测器、决策规则为固定的确定性策略（缺乏学习型调度器），以及实验规模和数据集多样性有限。

---

## 45. Trusted-Execution Environment (TEE) for Solving the Replication Crisis in Academia

**arXiv ID:** 2603.24878 | [PDF](https://arxiv.org/pdf/2603.24878v1)

**作者:** Jiasun Li `[一作]` (George Mason University), Project Team `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了基于云端可信执行环境（TEE）执行复制包并提交加密证明的框架，以解决学术复制危机，并在《Management Science》70卷论文上进行实验验证

**💡 创新点**

将TEE用于复制验证，作者自行在云TEE中运行复制包并提交证明，显著降低成本、提升可扩展性、支持专有数据，并提供可公开验证的证明

**🔧 技术方法**

利用Intel TDX、AMD SEV‑SNP等可信执行环境、Docker容器化、加密证明（远程证明）以及Azure/Google Cloud等云计算平台

**📊 数据集**

使用《Management Science》70卷共352篇论文的复制包，涵盖R、Stata、Python、SAS、MATLAB等多语言环境

**📈 对比分析**

与传统人工数据编辑器成本和耗时对比；云TEE下每个复制包成本约1.35–1.80美元，平均运行时间2.3小时，隔离开销7–12%，成功率Google 68%/Azure 95%；相比之下成本降低数百倍、验证速度提升数千倍

**⚠️ 局限性**

仍无法验证代码是否真实实现论文方法；受限于云TEE供应商可用性和专有软件许可问题；实验规模有限、失败多因缺失依赖；安全风险（侧信道、供应链）需进一步评估

---

## 46. Rafture: Erasure-coded Raft with Post-Dissemination Pruning

**arXiv ID:** 2603.24761 | [PDF](https://arxiv.org/pdf/2603.24761v1)

**作者:** Rithwik Kerur `[一作]` (University of California, Santa Barbara), Dahlia Malkhi `[通讯]` (University of California, Santa Barbara)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 Rafture，一种将基于 IDA 的纠删码与 Raft 共识协议集成的分布式信息扩散算法，并实现了后传播修剪、节点自适应存储和统一重构。

**💡 创新点**

创新点在于：① 引入后传播修剪机制，使节点可在确认足够副本后自主删除多余碎片；② 采用统一的 (F+1,(N‑1)(F+1)) 码，所有条目使用相同重构方法，无需条目级元数据；③ 通过心跳和响应估算故障数，实现自适应片段分配，降低重传与延迟；④ 允许节点根据本地已知的节点活跃度动态裁剪碎片，消除中心化存储决策。

**🔧 技术方法**

技术手段包括：纠删码（Reed–Solomon/ISA‑L实现）、Raft consensus、心跳+响应计数估算、阈值标记 (Threshold 1/2) 进行碎片裁剪、C++/FlexRaft代码库集成、RocksDB状态机。

**📊 数据集**

数据集与实验设置：使用 4000 字节的合成日志条目；实验环境为单机 DigitalOcean Droplet（16 vCPU/64 GB RAM），模拟多种分区（2 节点、F 节点）和大规模集群（N=99）场景。

**📈 对比分析**

与 FlexRaft 对比：在正常网络下存储利用率相当；在分区恢复后显著降低存储（≈50% 节省）；在不稳定网络下平均延迟≈95% 的情况下需要 2–4 次网络跳，且几乎无需重传；整体吞吐量与 FlexRaft 相近，但在存储与动态适配上更优。

**⚠️ 局限性**

局限性：① 仍依赖领导者估算，缺乏完全去中心化的分发与裁剪；② 仅在故障节点崩溃（Crash）情景下验证，未覆盖拜占庭或恶意攻击；③ 在极端网络分区期间仍会产生额外存储；④ 评估主要基于合成日志与单机实验，缺少跨机房真实网络数据。

---

## 47. Algorithmic Barriers to Detecting and Repairing Structural Overspecification in Adaptive Data-Structure Selection

**arXiv ID:** 2603.24597 | [PDF](https://arxiv.org/pdf/2603.24597v1)

**作者:** Faruk Alpay `[一作]` (Bahcesehir University), Levent Sarioglu `[通讯]` (Bahcesehir University)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了在自适应数据结构选择过程中，系统性结构过度规格化的可检测性与修复性问题。

**💡 创新点**

创新点在于提出并证明了两类不可计算障碍：在无界输入域上检测过度规格化是不可判定的，且任何保持原有证据对齐的保守修复算子必定存在过度规格化的不动点。

**🔧 技术方法**

采用了理论计算机科学工具，包括 Rice 定理、Kleene 递归定理、s‑m‑n 定理以及对 benchmark 聚合和 Bradley–Terry–Luce 拟合的严格分析。

**📊 数据集**

本文未使用具体数据集，而是基于抽象的工作负载签名和测量凭证模型进行形式化证明。

**📈 对比分析**

由于研究聚焦于可判定性和可修复性的理论极限，因此未进行实验比较；结果通过归约证明展示了在所有可计算管道族上无统一修复方案的不可行性。

**⚠️ 局限性**

局限性在于：研究仅针对理论模型，缺乏对实际自适应系统中具体实现细节的评估；此外，保守修复算子在理论上被证明不完整，实际中可能需要放宽保守性或限制域以获得可用方案。

---

## 48. Characterization of Constraints in Flexible Unknown Environments

**arXiv ID:** 2603.24813 | [PDF](https://arxiv.org/pdf/2603.24813v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 49. Can an Actor-Critic Optimization Framework Improve Analog Design Optimization?

**arXiv ID:** 2603.24714 | [PDF](https://arxiv.org/pdf/2603.24714v1)

**作者:** Sounak Dutta `[一作]` (North Carolina State University), Paul Franzon `[通讯]` (North Carolina State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种基于演员-评论家（Actor‑Critic）框架的模拟电路尺寸优化方法，利用LLM提出搜索区域并由评论家校正；

**💡 创新点**

核心创新在于将搜索区域提议与验证分离，形成闭环决策过程，使优化更具可解释性、稳健性；

**🔧 技术方法**

采用大型语言模型（如Qwen、GPT‑4o‑Mini、GPT‑5）做演员和评论家，配合贝叶斯优化（BO）在批准区间内模拟；

**📊 数据集**

在四个基准电路上评估：SKY130 12‑参数两级放大器、17‑参数宽摆幅放大器；GF180 12‑参数两级放大器、21‑参数折叠共轭放大器；

**📈 对比分析**

与单一LLM+BO和纯BO对比，ACOF在FoM平均提升约38.9%、regret下降约24.7%，在所有基准中始终保持最优；

**⚠️ 局限性**

限制包括对LLM提示的高度依赖、对更大搜索空间和更复杂约束的适应性待验证，以及在真实工业流程中的可扩展性需进一步研究。

---

## 50. SABER: A Stealthy Agentic Black-Box Attack Framework for Vision-Language-Action Models

**arXiv ID:** 2603.24935 | [PDF](https://arxiv.org/pdf/2603.24935v1)

**作者:** Xiyang Wu `[一作]` (University of Maryland), Dinesh Manocha `[通讯]` (University of Maryland)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6215c339-3735-4be3-8a07-5bbb7004712d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于代理的黑盒、仅文本指令的VLA攻击框架，利用ReAct式工具调用和GRPO训练在有限编辑预算下生成鲁棒攻击；

**💡 创新点**

创新点在于把攻击视作多轮工具使用的决策过程，使用GRPO在无梯度环境中学习；结合字符、词、提示级工具实现细粒度、可解释的指令扰动；实现多模型、多任务的可迁移与可隐蔽攻击；

**🔧 技术方法**

使用ReAct式多轮工具调用、Group Relative Policy Optimization (GRPO) 与 LoRA微调、字符/词/提示编辑工具箱以及对LIBERO仿真环境的rollout反馈；

**📊 数据集**

使用LIBERO基准任务（Spatial、Object、Goal、Long等四个子集）和六个强大VLA模型；

**📈 对比分析**

与强GPT-5-mini对比，实验表明攻击成功率提升≈2%，任务成功率下降20.6%，动作序列增长55%，约束违规提升33%；同时工具调用和字符编辑量分别降低21.1%和54.7%，显示更高效、更隐蔽的攻击效果；

**⚠️ 局限性**

局限性：仅在仿真中对文本指令进行攻击；未考虑多模态感知或真实机器人环境；攻击仅针对输入层，未对模型内部推理过程进行干预。

---

## 51. Bridging Code Property Graphs and Language Models for Program Analysis

**arXiv ID:** 2603.24837 | [PDF](https://arxiv.org/pdf/2603.24837v1)

**作者:** Ahmed Lekssays `[一作]` `[通讯]` (Qatar Computing Research Institute), Ahmed Lekssays (Qatar Computing Research Institute)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

开发并公开了一个基于 Model Context Protocol（MCP）的服务器（Codebadger），将 Joern 的 Code Property Graph（CPG）与大型语言模型（LLM）通过一组高层分析工具（如程序切片、污点跟踪、数据流分析等）进行集成，使 LLM 能在不受 token 限制的情况下在完整代码仓库中进行漏洞检测、利用和补丁生成。

**💡 创新点**

创新点在于：① 把静态分析的复杂 CPGQL 查询抽象为易于 LLM 调用的高层工具，避免 LLM 需要自行生成错误或过于复杂的查询；② 通过 MCP 让 LLM 与 CPG 引擎实现无缝交互，解决了 token 受限、跨文件漏洞缺失的问题；③ 通过真实大规模代码库的三大用例（GGML、libtiff、libxml2）验证了该方法在漏洞发现、利用和补丁生成上的有效性。

**🔧 技术方法**

使用的技术包括：大型语言模型（Claude Sonnet 4.5）、Joern（生成与查询 CPG）、Model Context Protocol（MCP）框架、FastMCP 服务器、Redis 缓存、Docker 容器化、异步查询、程序切片算法、污点追踪、调用图构建、边界检查等。

**📊 数据集**

主要使用了三组真实代码库作为实验数据集：GGML（8,667 方法、约 246k LOC）、libtiff（828 文件、约 127k LOC）以及 libxml2（1,574 文件、约 335k LOC），并通过这些数据集进行漏洞发现、利用生成以及补丁验证。

**📈 对比分析**

通过在上述三大用例中演示 LLM 的工作流程，比较了传统仅基于片段的 RAG 方法与本方法的差异。结果显示，本方法能跨文件追踪数据流，发现并利用了 libtiff 的未公开缓冲区溢出，且在 libxml2 上首次尝试即可生成与维护者提交的修复相近的正确补丁。虽然文中未给出大规模数值评估，但示例表明 token 需求大幅下降、分析覆盖面大幅提升，并成功实现了跨文件漏洞检测和补丁生成。

**⚠️ 局限性**

局限性包括：① Joern 生成 CPG 对内存和 CPU 的需求较大，处理极大代码库（如 Linux kernel）可能受限；② LLM 在某些场景仍需直接调用 CPGQL，说明高层工具抽象并未完全消除对复杂查询的依赖；③ 目前仅支持静态分析，无法检测运行时竞态等动态漏洞；④ 系统性能与 LLM 的工具链调用能力紧密相关，模型不同时效果可能差异显著；⑤ 文章仅给出案例性验证，缺乏大规模量化评估与对比实验。

---

## 52. Bridging the Gap Between Agility and Planning

**arXiv ID:** 2603.24773 | [PDF](https://arxiv.org/pdf/2603.24773v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 53. Numerical Superoptimization for Library Learning

**arXiv ID:** 2603.24812 | [PDF](https://arxiv.org/pdf/2603.24812v1)

**作者:** Jonas Regehr `[一作]`, Pavel Panchekha `[通讯]`

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

通过将Herbie数值超级优化器的搜索、等价判定和成本模型逆向使用，自动识别并推荐可为目标应用实现的新数学原语。

**💡 创新点**

提出“数值库学习”框架，首次将超级优化器的搜索与等价推理直接用于生成可重用原语，并通过计数、频率和紧迫性等启发式对候选原语进行排序与去重。

**🔧 技术方法**

利用Herbie的e‑graph搜索、等价推导、成本与误差模型；构建候选生成、去重、评估与递推筛选阶段；使用LLVM匹配器进行部署演示。

**📊 数据集**

在三大科学应用数据集上验证：地理投影库PROJ（92个内核）、热物性计算库CoolProp（24个内核）以及天体动力学框架Basilisk（28个内核）。

**📈 对比分析**

实验表明，选定原语在Herbie中被广泛采用，提升了1–4倍的速度（高精度场景下最高4×）并显著降低误差；在编译后通过LLVM匹配器实现的原语替换可实现5%以内的加速或提高精度。

**⚠️ 局限性**

局限性包括：启发式排名未充分考虑实现难度，导致部分高频但成本高昂的原语被推荐；LLVM匹配器缺乏成本模型，无法做精确权衡；整个学习流程耗时数小时，且仍需人工专家完成实现。

---

## 54. Normal forms in cubical type theory

**arXiv ID:** 2603.24923 | [PDF](https://arxiv.org/pdf/2603.24923v1)

**作者:** Xu Huang `[一作]` `[通讯]`, Xu Huang

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文阐述了立方体类型理论（Cubical Type Theory）中的正规形式（normal forms）的规范化定义，提供了与传统归一化证明相对应的更易读的语法框架；

**💡 创新点**

创新点在于将立方体类型理论中的复杂中性形式（neutral forms）与不稳定前沿（frontier of instability）引入正规化定义，借助 Cofibration、Interval 与 Glue 等构造构建了一套完整的正规化语法，并对其等价性进行了系统化处理；

**🔧 技术方法**

主要使用了合成归一化（synthetic Tait computability）框架、分裂（splitting）与稳定化（stabilization）技术，以及分布式格（distributive lattice）理论来处理 Cofibration 和 Interval 的决策；

**📊 数据集**

无实验数据集，本文属于理论性综述与规范性定义，不涉及实际数据；

**📈 对比分析**

本文未进行实验比较或性能评估，而是通过形式化定义和归约规则证明等价性与可判定性，主要关注的是理论可判定性与一致性；

**⚠️ 局限性**

局限性在于公式与规则的实现仍需依赖复杂的分层结构（如 φ 前沿和 Glue 数据），对实际编译器或交互式证明助手的实现可能产生较高的实现成本；

---

## 55. TAMI-MPC:Trusted Acceleration of Minimal-Interaction MPC for Efficient Nonlinear Inference

**arXiv ID:** 2603.24861 | [PDF](https://arxiv.org/pdf/2603.24861v1)

**作者:** Zhuoran Li `[一作]` (University of Arizona), Danella Zhao `[通讯]` (University of Arizona)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 TAMI‑MPC 框架，利用可信执行环境中的同步种子和一次性多项式乘法，实现非线性推理的极低交互 MPC 方案。

**💡 创新点**

创新点在于：①将叶比较与树合并改为单轮交互，消除 OT 交互；②在 TEE 内同步种子生成 correlated randomness，完全摆脱离线 ROT 生成；③采用一次性多项式乘法和随机性重用技术，将树合并复杂度从 log₂n 降至 1 并大幅降低通信；④设计流水化哈希和内存高效数据管理的 FPGA 加速器。

**🔧 技术方法**

技术方法包括：可信执行环境（TEE）中的 PRG 同步种子、一次性多项式乘法（PolyMult）、同步 ROT 生成、随机性重用、流水化 AES + CRH、预取 LUT 及 Packed Polynomial 执行。

**📊 数据集**

使用的基准数据集包括 ImageNet（ResNet‑50、SqueezeNet）和 GLUE/BERT‑base 的 NLP 任务，且在 ReLU、Softmax、GeLU 等非线性激活层上做微基准。

**📈 对比分析**

与 Cryptflow2、Cheetah、Bumblebee 等现有 PPMLaaS 系统比较，在 LAN、WAN 与移动网络下分别获得 ResNet‑50 推理 4.86× 加速、BERT‑base 推理 7.44× 加速，整体在移动网络下完成 ResNet‑50 108 s、BERT‑base 380 s。

**⚠️ 局限性**

限制主要在于：①仍需可信执行环境做离线预处理，TEE 漏洞可能影响安全；②对大规模模型仍需大量 correlated randomness，虽然已优化但仍非零开销；③实现依赖 FPGA 加速，软件可移植性受限；④当前方案主要针对非线性推理，线性层仍按传统 MPC 处理。

---

## 56. A Framework for Generating Semantically Ambiguous Images to Probe Human and Machine Perception

**arXiv ID:** 2603.24730 | [PDF](https://arxiv.org/pdf/2603.24730v1)

**作者:** Yuqi Hu `[一作]` (University of California Berkeley), Jennifer E. Corbett `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 667 | [OpenAlex ID](https://openalex.org/A5024180947)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

生成语义模糊图像并测量人类与机器的决策边界

**💡 创新点**

将CLIP嵌入进行线性插值与扩散模型结合，产生连续的语义模糊刺激，并在同一刺激上同步进行心理物理实验和模型评估

**🔧 技术方法**

Stable Diffusion、CLIP嵌入插值、classifier‑free guidance、2AFC实验、psignifit心理计量曲线拟合

**📊 数据集**

基于文本提示"a duck"、"a rabbit"（以及"an elephant"），生成300张图像；评估多款ImageNet预训练分类器

**📈 对比分析**

通过拟合P(rabbit)的心理计量曲线得到PSE与斜率，对比人类与模型的偏差和敏感度，发现模型更偏向兔子、敏感度更高

**⚠️ 局限性**

实验仅覆盖少数概念，未检验多标签情况；偏差可能来源于生成图像本身或模型训练目标，缺乏更广泛模型和语义对的系统验证

---

## 57. Once-for-All Channel Mixers (HYPERTINYPW): Generative Compression for TinyML

**arXiv ID:** 2603.24916 | [PDF](https://arxiv.org/pdf/2603.24916v1)

**作者:** Yassien Shaalan `[一作]` `[通讯]`, Yassien Shaalan

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3855fcda-48ef-4070-a15e-803cd5c84d83` `8d10c613-917e-4880-9716-17789f50e119` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

提出了一种压缩即生成（compression‑as‑generation）方法，在微控制器上将大多数1×1通道混合器权重在加载时由共享微型MLP根据小的层码生成，从而显著减少Flash占用，并保持整数运算兼容。

**💡 创新点**

创新点在于将通道混合器的权重压缩为生成式模型：仅首层手工存储，后续层一次性在启动时生成并缓存，避免运行时动态生成、分支与SRAM压力，且兼容现有整数推理核。

**🔧 技术方法**

采用了共享微型MLP生成器、TinyML精确打包字节计数、4/6/8位量化、知识蒸馏、焦点损失、基准阈值以及CMSIS‑NN/TFLM兼容的整数卷积核。

**📊 数据集**

使用了三大单导联心电图数据集：Apnea‑ECG（呼吸暂停检测）、PTB‑XL（临床诊断）和MIT‑BIH（心律失常分类）。

**📈 对比分析**

在相同Flash预算下与regular CNN、tiny separable、ResNet‑small、VAE‑head、HRV‑feature等基准模型对比，使用macro‑F1、AUC、准确率、能耗和延迟评估；在约225 kB时实现1.4 MB模型6.3×压缩率，保留≥95%宏F1；在32–64 kB时仍保持较高精度，整体性能优于传统压缩/稀疏方案。

**⚠️ 局限性**

局限性包括：需要一次性热启动生成导致启动延迟；对极度类别不平衡的数据集（如MIT‑BIH）阈值敏感；目前仅验证单通道ECG，跨任务推广需进一步验证；生成模型不适合实时输入特征变化。

---

## 58. Data-Oriented Modeling for Spacecraft Design

**arXiv ID:** 2603.24841 | [PDF](https://arxiv.org/pdf/2603.24841v1)

**作者:** Nathan Strange `[一作]` `[通讯]` (Vis Viva Space), Nathan Strange (Vis Viva Space)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出数据导向的MBSE方法并实现VVERDAD原型，演示在行星际任务概念上的应用

**💡 创新点**

将数据导向编程与实体-组件-系统架构引入MBSE，分离数据与逻辑、使用不可变无状态函数、统一多格式数据接口，提升可部署性、可测试性和可扩展性

**🔧 技术方法**

Rust语言、Bevy ECS、Serde、MiniJinja模板、Docker、Git CI/CD、Jupyter/Quarto等

**📊 数据集**

多格式设计数据（JSON、YAML、TOML、CSV、Excel）以及示例行星际任务概念数据集

**📈 对比分析**

与传统面向对象MBSE工具对比，降低部署复杂度、减少测试代码量、支持并行计算；原型演示通过CI/CD持续分析，未给出定量性能指标

**⚠️ 局限性**

原型尚未在大规模项目中充分验证，缺乏量化性能数据；对部分需要状态的分析支持有限；对现有分析工具的兼容性仍需完善

---

## 59. Model2Kernel: Model-Aware Symbolic Execution For Safe CUDA Kernels

**arXiv ID:** 2603.24595 | [PDF](https://arxiv.org/pdf/2603.24595v1)

**作者:** Mengting He `[一作]` (Pennsylvania State University), Linhai Song `[通讯]` (Institute of Computing Technology)

**通讯引用:** 1712 | [OpenAlex ID](https://openalex.org/A5060633759)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

本文提出了 Model2Kernel，一种结合动态模型分析与 CUDA 专用符号执行的系统，用于自动检测 LLM 推理系统中的 CUDA 核心内存安全缺陷。

**💡 创新点**

创新点包括：①基于模型动态分析自动推断 CUDA 核心参数的模型依赖性；②为动态张量和线程标识引入新的符号抽象，显著提高了符号执行对 GPU 并行度的支持；③将上述信息与 KLEE 结合，实现对大规模 LLM 相关 CUDA 核心的高效、精准内存错误检测。

**🔧 技术方法**

主要技术：动态模型分析、CUDA 专用符号执行（基于 KLEE+Z3）、张量内存区域抽象、线程标识符符号化、配置变异与上下文生成。

**📊 数据集**

使用数据集：来自 vLLM（62 模型/50 核心）、Hugging Face（54 模型/117 核心）以及四篇近期 AI 会议论文（4 模型/6 核心），共计 120 模型、173 核心。

**📈 对比分析**

与 Honeycomb、GKLEE、ESBMC‑GPU 三种基线工具比较，Model2Kernel 在已知 20 条 vLLM 错误中检测到 15 条，整体发现 353 条内存错误（328 计数溢出、25 越界访问），误报率仅 2.49%。在性能上，总分析时长 985.95 小时，单核单线程分析上限 1 小时，平均每核约 8–12 分钟；相较原 KLEE，分析速度提升约 3 倍。

**⚠️ 局限性**

限制包括：仅支持 Hugging Face 格式模型和接受张量输入的 CUDA 核心；未覆盖 Triton、TensorFlow 或 NPU 相关代码；单核 1 小时限制可能导致部分复杂核遗漏；路径探索仍继承 KLEE 的策略，难以充分覆盖所有并行路径；对配置变异的自动化程度有限，需人工验证生成的 config。

---

## 60. Lookalike3D: Seeing Double in 3D

**arXiv ID:** 2603.24713 | [PDF](https://arxiv.org/pdf/2603.24713v1)

**作者:** Chandan Yeshwanth `[一作]` (Technical University of Munich), Angela Dai `[通讯]` (Technical University of Munich)

**通讯引用:** 12385 | [OpenAlex ID](https://openalex.org/A5026634347)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了室内场景中的相似物体检测任务，利用多视角图像对物体对进行同类、相似和不同的分类，并将检测结果用于联合3D重建和部件共分割。

**💡 创新点**

创新点包括：①设计了多层次交替注意力（单视、物体级、多视全局）来聚合多视角信息；②结合三元组损失与对齐损失实现精确的相似度学习；③创建了基于ScanNet++的海量相同、相似和不同物体对数据集，成为该任务的基准。

**🔧 技术方法**

使用的技术：DINOv2基线提取器、Transformer编码器（交替注意力层）、三元组损失、对齐损失、阈值对齐、SAM 3D 以及 P3-SAM 进行下游应用。

**📊 数据集**

使用的数据集：ScanNet++改编的LookAlike-ScanNet++数据集，包含约71,000对（32,270同类、5,468相似、33,320不同）物体对，覆盖22,285个独立实例和867类。

**📈 对比分析**

在验证集上与多种2D/3D匹配基线（DINOv2-feats、SuperGlue、ASpanformer、Lepard、Predator等）对比，取得了最高的整体IoU（约0.49），在同类、相似和不同类别均显著优于基线；消融实验验证了交替注意力层、三元组+对齐损失以及多视角输入的有效性。

**⚠️ 局限性**

局限性：对小尺寸或远距离物体的检测效果受DINOv2 patch分辨率限制；多视角覆盖不足时性能下降；在极端长尾类别场景中仍面临小样本和分辨率挑战。

---

## 61. Trust as Monitoring: Evolutionary Dynamics of User Trust and AI Developer Behaviour

**arXiv ID:** 2603.24742 | [PDF](https://arxiv.org/pdf/2603.24742v1)

**作者:** Adeela Bashir `[一作]` (Teesside University), The Anh Han `[通讯]` (Teesside University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

通过演化博弈论与强化学习模拟，研究用户信任（视为监控降低）与 AI 开发者安全行为在重复不对称交互中的共同进化。

**💡 创新点**

首次将信任定义为监控成本的递减，构建包含多种用户信任策略（AllA、AllN、TFT、TUA、DtG）与开发者安全/不安全选择的异质演化模型，并证明监管力度与监控成本共同决定可持续安全与高采纳的平衡点。

**🔧 技术方法**

采用无穷种群复制者动力学、有限种群随机动力学（基于 Fermi 复制率）、Q‑learning 强化学习三种演化/学习框架，配合参数化的重复博弈收益矩阵进行分析。

**📊 数据集**

本研究未使用真实数据集，而是通过参数化实验（如 μ=-0.2、c=0.5、b_u=b_c=4、r=10、θ_T=θ_D=3 等）在模拟环境中探索不同监控成本和监管处罚下的演化结果。

**📈 对比分析**

对比三种方法得到一致结论：监控成本低且监管处罚足够时，系统趋向安全开发+广泛采纳（AllA+C）；监控成本高或处罚不足时，系统趋向不安全采纳或不采纳。Q‑learning 结果表明在低监控成本下，TUA/TFT 等信任策略能保持合作；随监控成本上升，合作崩溃，出现 AllN + D 的不良平衡。

**⚠️ 局限性**

局限性包括：① 假设用户和开发者均匀同质；② 监管者被隐式合并为处罚参数，未显式建模监管机构互动；③ 只考虑有限的信任与行为策略；④ 结果基于模拟参数，缺乏对实际 AI 市场或监管制度的实证验证。

---

## 62. Quadratic Residue Codes over $\mathbb{Z}_{121}$

**arXiv ID:** 2603.24689 | [PDF](https://arxiv.org/pdf/2603.24689v1)

**作者:** Tapas Chatterjee `[一作]`, Priya Jain `[通讯]` (Indian Institute of Technology Ropar)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

研究并给出了在环 ℤ121 上的二次剩余码（QR 码）的幺元生成元，并讨论其扩展码和灰映射得到的新线性码；

**💡 创新点**

创新点在于首次通过幺元理论完整描述了 ℤ121 上二次剩余码的生成元，并确定了存在条件 (p ≡ ±1, ±5, ±7, ±9, ±19 mod 44)，从而为更一般的 ℤ_q^n 形式提供了易于构造的方法；

**🔧 技术方法**

使用了环与码理论的技术，特别是幺元分解、二次剩余集与非剩余集的构造、乘法逆变换以及 Gray 映射；

**📊 数据集**

本研究未使用外部公开数据集，所有示例均基于数学构造的具体质数（如 5、7 等）得到的二次剩余码；

**📈 对比分析**

通过 Gray 映射将 ℤ121 上的 QR 码映射为 11 倍长度的 ℤ11 线性循环码，得到的码参数如 [55,5,33]、[77,7,44] 等，表明在对应域上具备良好的最小距离；

**⚠️ 局限性**

局限性包括仅在特定模 44 的同余类下存在 QR 码，对更一般的环和码长的推广尚未完全实现，且 Gray 映射后得到的码不一定保持自对偶性。

---

## 63. Reaching Beyond the Mode: RL for Distributional Reasoning in Language Models

**arXiv ID:** 2603.24844 | [PDF](https://arxiv.org/pdf/2603.24844v1)

**作者:** Isha Puri `[一作]` (Massachusetts Institute of Technology), Yoon Kim `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种多答案强化学习（Multi‑Answer RL）框架，训练语言模型在一次生成中输出一组可行答案及对应置信度，而非单一答案；

**💡 创新点**

创新点在于将多答案生成与强化学习奖励结合，使用基于集合正确性与多答案Brier评分的奖励函数，实现模型主动探索并输出多样、校准的答案集合；

**🔧 技术方法**

技术包括：RLVR/RLCR基础，改进为多答案奖励（R_RLVR^multi 与 R_RLCR^multi），Qwen3‑8B 基础模型，GRPO 算法，结构化输出格式奖励，和多答案Brier评分；

**📊 数据集**

数据集涵盖三种任务：医学诊断多标签集 DDXPlus、含信息缺失的多跳 QA HotPotQA‑Modified、编程实现多解 MBPP；

**📈 对比分析**

与单答案 RL（RLVR/RLCR）以及仅使用提示的多答案方式对比，实验显示多答案 RL 在覆盖率、答案多样性、令牌效率（token 约 40‑70% 降低）和集级校准上均优于基线；单答案模型在 Top‑1 准确率略高；

**⚠️ 局限性**

局限性包括：单答案模式仍能得到更高的 Top‑1 准确率；实验仅限 QA 与编程任务；多答案序列化限制了并行生成；在极难的单标记任务（如 HotPotQA‑Modified）下多答案 RLCR 的校准表现低于单答案 RLCR，可能受先验置信度求和约束影响。

---

## 64. ICTPolarReal: A Polarized Reflection and Material Dataset of Real World Objects

**arXiv ID:** 2603.24912 | [PDF](https://arxiv.org/pdf/2603.24912v1)

**作者:** Jing Yang `[一作]` (University of Southern California), Yajie Zhao `[通讯]` (University of Southern California)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建并公开了一个大规模真实物体偏振反射和材料数据集，涵盖多视角、346个光照方向、交叉/平行偏振、反射分离以及物理属性（漫反射、镜面反射、法向量）。

**💡 创新点**

首次在真实世界中实现高分辨率偏振分离和完整材料属性测量，为深度网络提供直接监督的物理基础训练数据，显著提升逆向渲染、重光照和稀疏三维重建的真实性和一致性。

**🔧 技术方法**

使用8摄像机+346 LED光源的Light Stage，交叉/平行偏振滤光，OLAT采集；依据Malus定律和物理光度学求取漫、镜面分量及法向量；利用低秩适配技术对RGB2X等预训练网络进行微调，训练逆向和正向渲染网络。

**📊 数据集**

主要数据集为本研究提供的218个日常物体的1.2M高分辨率图像；对比基准Synthetic数据集（Objaverse、HyperIrm、OpenSVBRDF、Light Stage等）。

**📈 对比分析**

通过与Synthetic基线和Objaverse实验进行定量比较（MSE、PSNR、SSIM、LPIPS）和定性可视化，逆向分解误差显著下降，重光照质量提升（PSNR从≈18dB提升至≈28dB，SSIM从0.51提升至0.90），稀疏三维重建误差亦明显降低，整体性能明显优于纯Synthetic训练模型。

**⚠️ 局限性**

受限于静态光场、光源布局与物体尺寸，难以测量高度透明、动态或强各向异性材料，且未获取亚表面参数；未来需扩展到更复杂材料和野外获取。

---

## 65. TRAJEVAL: Decomposing Code Agent Trajectories for Fine-Grained Diagnosis

**arXiv ID:** 2603.24631 | [PDF](https://arxiv.org/pdf/2603.24631v1)

**作者:** Myeongsoo Kim `[一作]` (AWS AI Labs), Varun Kumar `[通讯]` (AWS AI Labs)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了TRAJEVAL诊断框架，将代码代理执行轨迹拆分为搜索、阅读、编辑三阶段，并基于参考补丁计算精度与召回；

**💡 创新点**

创新点在于提供细粒度、可解释的阶段诊断，超越单一Pass@k指标，既可预测成功率，又可实时给出反馈提升性能；

**🔧 技术方法**

采用黄金上下文对齐、树形解析（tree‑sitter）提取文件/函数/编辑目标，计算精度/召回；用逻辑回归预测任务成功，并实现实时反馈机制；

**📊 数据集**

使用SWE‑bench Verified（500个Python GitHub问题）和PolyBench Verified（382多语言实例）两大基准；

**📈 对比分析**

在三种代理架构（SWE‑Agent、OpenHands、LiveSWEAgent）与七大语言模型上评估，预测Pass@1的MAE仅为0.87–2.1%，排名相关性ρ≥0.886；实时反馈可将Pass@1提升2.2–4.6个百分点，同时成本降低20–31%；

**⚠️ 局限性**

局限在于依赖参考补丁，无法完全覆盖可替代解决方案；大部分阶段精度低，导致过度探索；跨语言泛化时MAE可升至≈8.8%，且框架主要适用于可获得参考补丁的代码任务。

---

## 66. Causal AI For AMS Circuit Design: Interpretable Parameter Effects Analysis

**arXiv ID:** 2603.24618 | [PDF](https://arxiv.org/pdf/2603.24618v1)

**作者:** Mohyeu Hussain `[一作]` (University of Florida), Domenic Forte `[通讯]` (University of Florida)

**通讯引用:** 6871 | [OpenAlex ID](https://openalex.org/A5009243659)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建了基于因果推断的AMS电路设计框架，自动从SPICE仿真数据中学习DAG并估计ATE，提供可解释的参数重要性排名。

**💡 创新点**

首次将因果推断与ATE估计应用于模拟电路，实现更高准确度和可解释性的设计决策。

**🔧 技术方法**

因果发现算法（YLearn混合管道）、ATE估计、双重机器学习（RF+MLP+ElasticNet）以及NN对照模型。

**📊 数据集**

三类运算放大器（OTA、台式、折叠cascode）在TSMC 65nm节点的SPICE仿真数据，OTA/台式分别20000样本，折叠cascode 38000样本。

**📈 对比分析**

与传统NN回归器对比，因果模型的平均绝对误差低于25%（≈80% NN），且避免了符号颠倒，展示更可靠的可解释性。

**⚠️ 局限性**

受限于单一工艺节点、有限的电路类型以及需要手动仿真生成的数据，尚未验证在更大规模或多工艺、CMFB等复杂模块上的通用性。

---

## 67. Cyber-Physical System Design Space Exploration for Affordable Precision Agriculture

**arXiv ID:** 2603.24785 | [PDF](https://arxiv.org/pdf/2603.24785v1)

**作者:** Pawan Kumar `[一作]` (Arizona State University), Hokeun Kim `[通讯]` (Arizona State University)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

提出了一个面向精准农业的成本感知多模态无人机/机器人平台设计空间探索框架，利用整数线性规划（ILP）与 SAT 验证自动生成多种成本、覆盖率与有效载荷兼顾的方案；

**💡 创新点**

核心创新在于将农作物与应用的非线性参数统一编码进 ILP，配合 SAT 检验消除无效方案，并通过权重调节实现成本、覆盖率和有效载荷的多目标优化；

**🔧 技术方法**

采用整数线性规划求解器、PySAT 进行约束验证、GEekbench 评估算计单元性能，并在两种规模的农场案例中进行实验；

**📊 数据集**

使用来自 USDA NASS、欧盟农业普查的数据估算农场面积与预算，结合公开的硬件成本与电池能量数据构建实验环境；

**📈 对比分析**

与多种现有 CPS DSE 方法（如模拟退火、贝叶斯优化、遗传算法、PG‑DSE 等）进行对比，结果表明该方法在覆盖率和成本之间取得最优权衡，获得最高的加权得分且验证通过率达 100%；

**⚠️ 局限性**

局限性主要体现在对静态、确定性条件的假设，线性近似无法完整捕捉电池放电、重量影响与动力学的非线性行为，且对天气、土壤等动态因素缺乏建模。

---

## 68. OpenCap Monocular: 3D Human Kinematics and Musculoskeletal Dynamics from a Single Smartphone Video

**arXiv ID:** 2603.24733 | [PDF](https://arxiv.org/pdf/2603.24733v1)

**作者:** Selim Gilon `[一作]` (University of Utah), Scott D. Uhlrich `[通讯]` (University of Utah)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发了一种基于单台智能手机视频的 3D 人体运动学与动力学无标记分析方法 OpenCap Monocular。

**💡 创新点**

创新点在于将 monocular pose 估计与物理约束优化相结合，显著提升姿态精度，并通过物理仿真与机器学习共同推算动力学；同时通过云端部署实现免费、快速的全流程服务。

**🔧 技术方法**

使用技术包括：WHAM/ViTPose 3D 关节检测、SMPL 模型、基于 PyTorch 的姿态优化、OpenSim 逆运动学、物理驱动模拟、GaitDynamics 机器学习预测 GRF、NVIDIA GPU 云计算。

**📊 数据集**

使用公开的 OpenCap 数据集（多摄像头视频、标记运动捕捉与力板），对 10 名健康成人的步行、深蹲、坐起等动作进行验证。

**📈 对比分析**

与标记捕捉、CV+IK 基线及双摄像头 OpenCap 进行比较。旋转 MAE 4.8°、平移 MAE 3.4 cm，分别比 CV+IK 降低 48% 与 69%；步行 GRF MAE 9.7% BW（比 CV+IK 下降 58%）。在临床用例中，膝关节伸展矩 MAE 5.8 Nm（低于 11 Nm 临床阈值），步行膝内翻矩峰值 MAE 0.36% BW·ht（低于 0.5% BW·ht 临床阈值）。

**⚠️ 局限性**

局限性包括：对跳跃等长时间飞行阶段表现不佳；验证仅在年轻健康人群，未覆盖病理运动；单摄像头 45° 视角要求较严格，无法直接用于已存在的大规模在线视频；需要保持相机静止与已知人身高度等先验假设。

---

## 69. Persistence-based topological optimization: a survey

**arXiv ID:** 2603.24613 | [PDF](https://arxiv.org/pdf/2603.24613v1)

**作者:** Mathieu Carriere `[一作]` (Centre Inria d'Université Côte d'Azur), Naoki Nishikawa `[通讯]` (University of Tokyo)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `90291a0e-9d36-4a08-9a16-89ce846d923f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述了利用持久同调（Persistence Homology）在机器学习与深度学习中对拓扑特征进行优化的理论与实践，涵盖了持久性图的可微分性、梯度计算方法、梯度下降变体以及实现细节，并提供了一个统一的开源 Python 库来复现并实验这些方法。

**💡 创新点**

①首次将持久性图的多值空间通过分层分块（Whitney 分层）与可微结构统一化，证明其在参数空间上可微；②提出了“分层梯度下降”和“大步梯度下降”等更高效、更稳定的梯度计算与优化方案；③将梯度平滑与降采样、差分同胚插值等技术融入优化框架，提升稀疏梯度的稠密性与数值鲁棒性。

**🔧 技术方法**

主要技术包括：持久性图的可微分框架（利用分层分块、lift 与链式法则）；梯度计算（vanilla、分层、单点、大步梯度）；梯度采样与分层梯度下降；降采样与神经网络降采样；差分同胚插值（RKHS 最小范数插值）；以及使用持久性图距离（Wasserstein / bottleneck）与相关稳定性定理。

**📊 数据集**

论文主要为综述性质，未在新数据集上进行实验；所述方法已在多篇实验论文中验证，常用的数据集包括点云（如 3D 场景、扫描数据）、图结构（社交网络、交通网络）以及图像（医学影像、自然图像）。

**📈 对比分析**

比较方法一般对比传统的基于欧氏距离或其他度量的损失与基于持久性图的损失；性能指标多为收敛速度（迭代次数）、梯度稠密度、最终损失值以及对下游任务（如分类、重建）的改进。大步梯度下降通常收敛更快，分层梯度下降则在全局收敛性上更有理论保证。

**⚠️ 局限性**

局限性包括：①持久性图空间缺乏线性结构，梯度稀疏导致收敛缓慢；②分层分块实现复杂且对高维大规模数据成本高；③梯度稳定性受匹配不唯一影响，尤其是距离到目标图的梯度不连续；④对数据噪声敏感，持久性图噪声点可能导致梯度不稳定；⑤目前大多数方法主要针对欧氏空间下的点云，扩展到更一般的度量空间和高维图仍有挑战。

---

## 70. OptiSAR-Net++: A Large-Scale Benchmark and Transformer-Free Framework for Cross-Domain Remote Sensing Visual Grounding

**arXiv ID:** 2603.24876 | [PDF](https://arxiv.org/pdf/2603.24876v1)

**作者:** Xiaoyu Tang `[一作]` (South China Normal University), Rui Fan `[通讯]` (Tongji University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一种高效的Transformer‑free框架OptiSAR‑Net++，实现跨光学与SAR图像的视觉定位任务，并首次构建了OptSAR‑RSVG大规模跨域视觉定位基准数据集。

**💡 创新点**

创新点包括：① Patch‑Level Low‑Rank Adaptation Mixture of Experts (PL‑MoE)实现跨域特征解耦；② 将定位任务转化为CLIP对比检索，配合对抗负样本提升细粒度语义判别；③ 设计文本引导的Dual‑Gate Spatial Shuffle Attention (TGDF‑SSA)实现高效的语义注入与空间建模；④ 结合区域感知辅助头，进一步强化空间分布监督。

**🔧 技术方法**

技术手段主要包括：Mixture‑of‑Experts 与 LoRA 的低秩参数化；CLIP预训练文本编码器和对比学习框架；对抗式负样本采样；多尺度特征金字塔；可学习的门控与注意力机制。

**📊 数据集**

使用数据集：OptSAR‑RSVG（46,825张光学+SAR图像，90,148图文框标注，覆盖16类），以及单源光学基准DIOR‑RSVG（20类）。

**📈 对比分析**

与多种Transformer‑based和对比学习方法在OptSAR‑RSVG上对比，OptiSAR‑Net++在光学、SAR两域均取得最优性能（meanIoU 82.76%，cumIoU 90.70%），参数量仅为95.6M，显著低于对比模型；在DIOR‑RSVG上也实现SOTA（Pr@0.9 45.28%，cumIoU 85.46%），并表现出较强的数据效率与空间定位能力。

**⚠️ 局限性**

局限性：仅考虑光学与SAR两种模态，未覆盖红外、多光谱或时序序列；CLIP‑style对比学习在极大数据规模下可能面临可扩展性挑战；缺乏对更复杂多模态融合与动态变化的深入探索。

---

## 71. SurgPhase: Time efficient pituitary tumor surgery phase recognition via an interactive web platform

**arXiv ID:** 2603.24897 | [PDF](https://arxiv.org/pdf/2603.24897v1)

**作者:** Yan Meng `[一作]` (Children's National Hospital), Mike Chang `[通讯]` (Stanford University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种针对垂体肿瘤手术的自动化手术阶段识别框架，并搭建了在线协作平台实现数据收集与注释。

**💡 创新点**

创新点包括：①使用自监督SimCLR在251段未标注视频上预训练ResNet‑50；②在少量标注数据上采用焦点损失、逐层解冻、动态采样等平衡策略；③将多阶段Temporal Convolutional Network（MS‑TCN++）改造为高维特征输入，提升时间建模；④通过手术记录自动抽取粗标签实现弱监督；⑤整合于可在线使用的Surgical Video Platform，实现持续学习与医生反馈。

**🔧 技术方法**

技术手段包括：自监督对比学习（SimCLR）、ResNet‑50特征编码、焦点损失（Focal Loss）、逐层解冻与动态采样、MS‑TCN++时间卷积网络、累加器后处理、NLP提取手术笔记标签。

**📊 数据集**

数据集：251段未标注垂体肿瘤手术视频用于SSL预训练，81段手工标注视频（4机构+PitVis）用于微调与评估；视频均转码至1920×720@15fps，随后按1fps、224×224进行训练。

**📈 对比分析**

与现有公开方法（如CITI、UNI‑ANDES等）比较，本文在测试集上实现整体准确率90%（F1≈88%），比CITI的61%提升约44%；ABlation实验表明SSL+焦点损失能显著提升各阶段召回与准确率，尤其是鼻腔与闭合阶段。

**⚠️ 局限性**

局限性：对个别与主流程差异大的手术视频识别准确率略下降；Sellar与Sphenoid、Closure阶段的边界仍易混淆；数据稀疏与帧选择不当可能影响判别；MS‑TCN++在极端变形序列中仍有改进空间。

---

## 72. Demystifying When Pruning Works via Representation Hierarchies

**arXiv ID:** 2603.24652 | [PDF](https://arxiv.org/pdf/2603.24652v1)

**作者:** Shwai He `[一作]` (University of Maryland), Ang Li `[通讯]` (University of Maryland)

**通讯引用:** 4450 | [OpenAlex ID](https://openalex.org/A5100413657)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对大型语言模型进行网络剪枝实验，并从嵌入、logit、概率三层表示层次分析剪枝对生成与非生成任务的不同影响。

**💡 创新点**

首次将模型内部计算拆分为嵌入、logit、概率三空间，并用二阶泰勒展开理论揭示softmax非线性放大剪枝误差导致生成性能崩溃的机制。

**🔧 技术方法**

采用结构化与非结构化剪枝（层丢弃、Wanda）、角度相似度、KL散度评估，以及张量分析+泰勒展开的理论分析。

**📊 数据集**

在Mistral、Qwen-2.5-7B上使用检索、文本分类、多选题（BoolQ、MMLU、OpenBookQA等）和生成任务（GSM8K、HumanEval、MBPP、NarrativeQA、NQ-Open）等基准数据集进行实验。

**📈 对比分析**

通过与未剪枝模型的准确率/得分对比，发现非生成任务保持≈90%原模型水平，而生成任务在中等剪枝率下仅达20%原模型，性能显著下降。

**⚠️ 局限性**

仅做训练后无微调的剪枝，未尝试微调恢复生成性能；实验仅聚焦于单一模型架构，缺乏跨模型泛化验证。

---

## 73. BiFM: Bidirectional Flow Matching for Few-Step Image Editing and Generation

**arXiv ID:** 2603.24942 | [PDF](https://arxiv.org/pdf/2603.24942v1)

**作者:** Yasong Dai `[一作]` (Australian National University), Hongdong Li `[通讯]` (Australian National University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 BiFM (Bidirectional Flow Matching)，在单一模型中同时学习少步生成和逆向编辑，通过估计平均速度场实现快速生成和精确逆向映射。

**💡 创新点**

创新点在于：① 将时间间隔监督从单向流匹配扩展到双向，利用共享的瞬时速度场实现前向和后向的平均速度一致；② 引入双向一致性损失和轻量级时间间隔嵌入，提升训练稳定性和逆向质量；③ 通过这一框架实现从预训练模型快速微调或从零训练，避免了传统 DDIM 逆向误差和额外的逆向网络。

**🔧 技术方法**

核心技术包括：流匹配 ODE 的平均速度场估计、MeanFlow 识别、时间间隔监督、双向一致性损失、时间间隔嵌入、LoRA 微调、以及在 SiT/MMDiT/U‑Net 结构上的集成。

**📊 数据集**

使用的数据集有：PIE‑Bench（图像逆向与编辑评估）、MSCOCO‑256（文本到图像生成）、ImageNet‑256（条件生成）、CIFAR‑10（小分辨率生成）。

**📈 对比分析**

与训练‑免费逆向方法、基于辅助网络的少步编辑器（如 SwiftEdit、PnP Inversion、RF‑Edit 等）以及传统流匹配模型（Flow Matching、MeanFlow、MMDiT）对比。BiFM 在编辑任务上在 LPIPS、SSIM、PSNR、MSE、CLIP 语义一致性等指标均优于基线；在生成任务上在 FID、IS、Precision/Recall 等指标上显著降低或提升，尤其在 1‑NFE（单步）设置下 FID 低至 2.75 (CIFAR‑10) 或 15.5 (SiT‑XL/2 on ImageNet‑256)。

**⚠️ 局限性**

局限性：① 仍需在训练阶段对时间间隔进行采样与权重调度，训练过程相对复杂；② 对极大步长的逆向仍可能出现误差，尤其在极端实时编辑场景下；③ 需要在已有流匹配/扩散模型上微调，未展示在零样本或完全未知结构上的迁移性。

---

## 74. ARC-AGI-3: A New Challenge for Frontier Agentic Intelligence

**arXiv ID:** 2603.24621 | [PDF](https://arxiv.org/pdf/2603.24621v1)

**作者:** ARC Prize Foundation `[一作]` `[通讯]` (ARC Prize Foundation), ARC Prize Foundation (ARC Prize Foundation)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了ARC-AGI-3交互式基准，旨在通过探索、建模、目标设定与规划四个核心维度评估代理智能。

**💡 创新点**

创新点在于：①将基准转为交互式、无语言、核心知识优先的环境；②引入行动效率（人类基准）评分框架；③强调未知未知的自适应挑战并对公共/私有集做OOD设计；④通过人类校准确保所有环境100%可解。

**🔧 技术方法**

使用自研Python游戏引擎实现的turn‑based 64×64格子环境，采用基于状态图的自动化验证与探索；利用人类实验数据做基准；采用LRM（大型推理模型）+自适配搜索或子代理架构进行实验。

**📊 数据集**

数据集包括：25个公共演示环境，55个半私有、55个完全私有，共165个环境；每个环境至少6级；人类校准数据来自约486名参与者。

**📈 对比分析**

评估方法：按层级动作效率与人类第二佳基准的平方来计算每级得分，层级线性加权后求平均；与人类对比，现有前沿模型在官方榜上仅0.25–0.37%（相对人类效率极低），但在社区榜或利用专业工具可达数十个百分点。

**⚠️ 局限性**

局限性：①过度依赖核心知识先验，可能不适用于更复杂现实任务；②公开集易被专门化工具或模型“harness”击破，官方榜难以防范；③动作效率仅衡量效率，未考虑长期学习与知识迁移；④对大型LRM的测试成本高，导致实际部署和验证受限。

---

## 75. From Weights to Concepts: Data-Free Interpretability of CLIP via Singular Vector Decomposition

**arXiv ID:** 2603.24653 | [PDF](https://arxiv.org/pdf/2603.24653v1)

**作者:** Francesco Gentile `[一作]` (University of Trento), Elisa Ricci `[通讯]` (University of Trento)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

针对 CLIP 视觉 Transformer 进行内部机制解释，提出完全无数据、无训练的框架，直接对注意力头的值-输出权重矩阵做奇异值分解，然后用 COMP 算法将奇异向量稀疏分解为人类可解释的概念，从而实现对单个概念向量的精细编辑与干预。

**💡 创新点**

创新点在于：① 采用权重空间（而非激活）进行解释，消除数据偏差；② 引入稀疏、语义一致的 COMP 分解算法，兼顾重建精度与概念连贯性；③ 通过对奇异值的调节实现对模型行为的可解释、数据自由的精细控制（消除伪相关、去除不安全概念、提升零样本分类）。

**🔧 技术方法**

核心技术包括：CLIP 视觉 Transformer 的 VO 权重提取；奇异值分解 (SVD)；稀疏正交匹配追踪（NNOMP）改进的 COMP 算法；LLM 作为评价者进行解释一致性评分；零样本分类与图像匹配实验验证。

**📊 数据集**

使用的数据集：OpenCLIP ViT‑L/14 预训练于 LAION‑2B；ConceptNet 5.5 作为概念词典；CC12M 用于图像匹配验证；Waterbirds、ViSU、零样本分类基准（如 ImageNet、CUB‑200 等）用于干预与评估；此外对 Fine‑Tuning 任务使用 Flowers 102、Oxford Pets、CUB‑200，并对 LoRA 进行验证。

**📈 对比分析**

与 TextSpan、NNOMP、top‑k 方案对比：COMP 在概念连贯性（LLM 评分）和重建余弦相似度两者上均优于对手；零样本分类准确率几乎不下降；在消除背景伪相关、去除 NSFW 概念和提升下游任务准确率上，改进后模型表现明显好于基线和 Safe‑CLIP，取得最高 1.0 点的性能提升。

**⚠️ 局限性**

局限性包括：① 不能完整重建所有奇异向量，因部分向量包含非语义或任务无关的“噪声”；② 依赖预定义的概念词典，缺乏新概念的发现；③ 仅分析 VO 权重，未考虑查询/键矩阵或注意力分布；④ 对于极端大模型的计算成本仍不低。

---

## 76. Fine-Tuning A Large Language Model for Systematic Review Screening

**arXiv ID:** 2603.24767 | [PDF](https://arxiv.org/pdf/2603.24767v1)

**作者:** Kweku Yamoah `[一作]` (University of Florida), Caleb Schutz `[通讯]` (University of Florida)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过在1.2B参数的LiquidAI LFM2.5模型上进行全参数微调，构建了针对单个系统综述标题与摘要筛选任务的专用语言模型。

**💡 创新点**

创新点在于强调并利用LLM的上下文依赖性，证明微调后模型在极度不平衡数据下仍能显著提升性能，并展现其可作为二级筛选器的潜力。

**🔧 技术方法**

所用技术包括Unsloth+TRL训练框架、bfloat16混合精度、8位AdamW优化器、全参数微调以及多温度（T∈{0.1,0.4,0.8}）的推理一致性验证。

**📊 数据集**

实验基于8,694条标题/摘要（人类标注为包含/排除）构成的数据集，进一步划分为371条（含194排除、121包含）用于训练/验证，剩余8,277条用于最终评估。

**📈 对比分析**

评估方式为与人类标注比较，计算准确率、平衡准确率、加权F1、Gwet AC1等指标；微调模型在完整数据集上准确率从6.52%提升至86.40%，加权F1提升80.79%，与人类的一致率达86.40%。

**⚠️ 局限性**

主要限制包括：使用的是极小模型、训练集对包含样本进行了人工过采样、仅采用全参数微调而未探索更高效的PEFT或强化学习方法、缺乏对不同系统综述任务的泛化验证。

---

## 77. Transformers in the Dark: Navigating Unknown Search Spaces via Bandit Feedback

**arXiv ID:** 2603.24780 | [PDF](https://arxiv.org/pdf/2603.24780v1)

**作者:** Jungtaek Kim `[一作]` (University of Wisconsin--Madison), Kangwook Lee `[通讯]` (University of Wisconsin--Madison)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究LLM是否能内化搜索算法，提出未知树搜索+bandit反馈框架，理论证明Transformer能实现多种搜索策略，并通过行为克隆训练Transformer，最后对预训练LLM进行针对搜索的微调。

**💡 创新点**

①将搜索问题与LLM对话交互分离，构造可控的“未知树搜索”设置；②证明Transformer在有限深度下可精确实现多种搜索策略；③通过从零训练和行为克隆展示Transformer的近似搜索能力及泛化；④对现有LLM进行搜索特定微调显著提升搜索性能。

**🔧 技术方法**

Transformer架构、行为克隆训练、叶子与树结构的tokenization、bandit反馈采样、Uniform/Greedy/UCT/Policy-guided等搜索策略、Qwen3-8B的微调与评估指标（max-reward hit rate、DCG、normalized path length 等）。

**📊 数据集**

合成二叉树搜索实例（深度6-8，目标数不同）、迷宫导航实例（4×4、壁密度0.4）以及Academic Paper Search真实轨迹（Gemini-2.5-Flash生成），训练集200个实例×100轨迹，测试集10个实例×100轨迹。

**📈 对比分析**

与Uniform leaf、Greedy leaf、Uniform path、Policy-guided path、UCT 等参考算法对比，评估指标为max-reward hit rate、DCG、normalized path length 等。行为克隆Transformer在指标上与对应参考算法相近，KL近似小；微调后的Qwen3-8B在Academic Paper Search任务中显著优于基线，提升约2倍。

**⚠️ 局限性**

仅在外部给定的搜索空间和bandit反馈下验证，未能完全内化搜索过程；模型受限于上下文长度和容量，泛化能力有限；多轮交互效率低；未证明理论可学习性；对高维/长序列搜索的表现未知。

---

## 78. Sovereign AI at the Front Door of Care: A Physically Unidirectional Architecture for Secure Clinical Intelligence

**arXiv ID:** 2603.24898 | [PDF](https://arxiv.org/pdf/2603.24898v1)

**作者:** Vasu Srinivasan `[一作]` (6Hats AI Labs Inc.), Dhriti Vasu `[通讯]` (University of California)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出并实现了“主权AI架构”，通过单向接收广播/数据二极管实现无网络接口的本地推理，并通过光学离线信道交付会话密钥。

**💡 创新点**

通过受控单向接收器实现结构化威胁模型降低，构建广播零信任边缘计算模式，彻底消除远程网络攻击面。

**🔧 技术方法**

受控广播接收（DVB‑T2/ATSC 3.0、卫星DVB‑S2）、硬件/软件数据二极管、光学一次性密钥传输、基于硬件根信任的安全隔离和本地化AI推理。

**📊 数据集**

在原型中部署了2 B参数INT4量化语言模型和相关视觉/语音推理模型，未公开具体数据集；重点在推理模型本地化。

**📈 对比分析**

对比传统联网终端，本体推理在不依赖网络的前提下仍可完成子秒级推理，广播载波下载需要数分钟；整体攻击面显著降低，但未给出量化性能基准。

**⚠️ 局限性**

局限在硬件信任、广播发射器可信性、实现缺陷、量子攻击、USB白名单弱点及回收延迟等，且剩余威胁仅限于物理攻击。

---

## 79. SABER: Spatial Attention, Brain, Extended Reality

**arXiv ID:** 2603.24830 | [PDF](https://arxiv.org/pdf/2603.24830v1)

**作者:** Tom Bullock `[一作]` (University of California, Santa Barbara), Barry Giesbrecht `[通讯]` (University of California, Santa Barbara)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

在沉浸式VR环境中，通过EEG记录与Beat Saber类似的目标追踪任务，验证了SABER框架。

**💡 创新点**

创新点在于将传统二维静态物体的注意力EEG方法迁移至三维动态物体，并首次实现实时位置重构。

**🔧 技术方法**

采用64通道EEG、事件相关电位、α波侧化及倒置编码模型（IEM）。

**📊 数据集**

使用32名健康大学生在4种条件（静态单/多、动态单/多）下的实验数据。

**📈 对比分析**

与传统单侧化α和ERP方法对比，IEM能够在动态情境下以α功率重构目标位置，显著高于置换基准。

**⚠️ 局限性**

主要限制包括强制中心注视、样本单一、VR佩戴造成的视觉/电极接触问题以及实验时长导致疲劳。

---

## 80. UniICL: Systematizing Unified Multimodal In-context Learning through a Capability-Oriented Taxonomy

**arXiv ID:** 2603.24690 | [PDF](https://arxiv.org/pdf/2603.24690v1)

**作者:** Yicheng Xu `[一作]` (Zhejiang University), Dacheng Tao `[通讯]` (Nanyang Technological University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出统一的多模态 In-Context Learning（Uni-ICL）框架，构建六级认知层级的任务分类，创建大规模训练集 UniICL-760K 与评测基准 UniICL-Bench，并引入 Context‑Adaptive Prototype Modulator（CAPM）模块以稳定少样本适配。

**💡 创新点**

创新点包括：①以认知层级为导向的六级任务分类，系统揭示 ICL 的非单调扩展行为；②首次公开统一训练集与评测基准，覆盖理解与生成两大领域；③CAPM 模块通过分离演示编码、低秩变换与自适应路由，实现对上下文的动态调节，显著提升 ICL 效率与鲁棒性。

**🔧 技术方法**

核心技术包括：多模态 Transformer 共享背骨、分段掩码交叉注意力、低秩动态参数化、Determinantal Point Process (DPP) 语义检索、以及自适应温度的密集余弦路由；此外还使用了 Qwen3、SAM、CLIP 等预训练模型进行数据生成与检索。

**📊 数据集**

使用了 766,868 条训练实例（UniICL-760K）以及 1,250 条评测样例（UniICL-Bench），这些样本来自 LAION‑5B 的人工标注分支（202,750 例）和生成分支（353,826 例），涵盖 15 个子任务，涵盖视觉理解与生成两大范畴。

**📈 对比分析**

与 BAGEL、UniWorld‑V1、Nexus‑Gen‑V2、Ovis‑U1 及 Qwen3‑VL 等多模态大模型进行对比，Uni-ICL 在理解与生成 ICL 效率均达到或超过同类模型，尤其在概念映射、类比推理等高认知层级任务中表现突出；在上下文扰动实验中表现出最高的稳定性，提升了 ICL 效率（峰值 78.9/16.9）。

**⚠️ 局限性**

局限性包括：①数据集依赖自动生成与外部模型，可能带来偏见与可复现性问题；②仅覆盖图像‑文本两模态，且上下文长度受限；③尚未扩展至视频、音频等更复杂模态，未来工作需提升标注质量、长上下文检索与跨域泛化。

---

## 81. ReSyn: A Generalized Recursive Regular Expression Synthesis Framework

**arXiv ID:** 2603.24624 | [PDF](https://arxiv.org/pdf/2603.24624v1)

**作者:** Seongmin Kim `[一作]` (University of Seoul), Sang-Ki Ko `[通讯]` (University of Seoul)

**通讯引用:** 783 | [OpenAlex ID](https://openalex.org/A5011019318)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种递归分解框架 ReSyn，用于从输入–输出示例自动合成复杂正则表达式

**💡 创新点**

引入可学习的分解路由器，允许根据示例动态选择串接或并集分解，并搭配参数高效的 Set2Regex 模型；同时证明最优分解 NP‑难并提出基于学习的近似策略

**🔧 技术方法**

结构化正则规范化、两级 Transformer（字符层与字符串层）结合多头注意力聚合、可学习的分解模块（分段、分区）以及递归合成流程

**📊 数据集**

利用来自多源的真实正则（GitHub 开源、Web 站点等）与合成/简化数据集，经过规范化后构建大规模训练集；评估使用三个真实世界基准（RegExLib、REx、CSharp）和一个合成基准

**📈 对比分析**

与现有 Seq2Seq 及 LLM（ByT5、gpt‑oss‑120b）对比，ReSyn+Set2Regex 在所有基准上的合成成功率、语义准确率和 MCC 均显著提升，尤其在深层（AST 深度≥5）正则上实现状态‑最优；参数量仅 29.6M，远小于 300M 的 baseline

**⚠️ 局限性**

对分解路由器的泛化能力有限（对不常见 Union 结构识别率下降），缺乏对负样例的系统评估，且仍依赖示例数量较多的训练集；未来需增强结构数据增强和负例挖掘

---

## 82. Distributed Real-Time Vehicle Control for Emergency Vehicle Transit: A Scalable Cooperative Method

**arXiv ID:** 2603.25000 | [PDF](https://arxiv.org/pdf/2603.25000v1)

**作者:** WenXi Wang `[一作]` (Tongji University), JunQi Zhang `[通讯]` (Tongji University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种可扩展的分布式车辆控制方法（SDVC），实现紧急车辆快速通行，同时最小化对普通车辆的影响。

**💡 创新点**

①使用仅本地信息的分布式决策，理论证明与全局最优目标近似相等；②引入基于联盟的冲突解决机制，消除集中式方法的单点故障风险；③实现无训练、实时可用，适应不同交通密度与道路配置。

**🔧 技术方法**

基于预测视界的影响判断、策略函数评估（速度/车道变更成本+安全惩罚）、联盟形成的冲突解决；利用V2V通信实现本地信息共享，并在每步递归解决安全冲突。

**📊 数据集**

HighD德国高速公路真实轨迹数据集，用于离散化道路网并评估算法性能。

**📈 对比分析**

与CRHA、GSAC、GSAC_IDM等基线方法对比，使用指标f'（影响度）、规划时延、碰撞率。SDVC在短期单段和长期多段、不同交通密度、速度异质性及车道数变化下均实现f'显著降低、规划时延保持在几十毫秒、碰撞率为0，优于传统方法。

**⚠️ 局限性**

仅在高速公路车道段测试；未覆盖交叉口等复杂路况；对极端拥堵（通信范围内车辆数过多）时可能仍出现冲突；需要V2V通信基础设施支持。

---

## 83. FODMP: Fast One-Step Diffusion of Movement Primitives Generation for Time-Dependent Robot Actions

**arXiv ID:** 2603.24806 | [PDF](https://arxiv.org/pdf/2603.24806v1)

**作者:** Xirui Shi `[一作]` (University of Alberta), Jun Jin `[通讯]` (University of Alberta)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Fast One‑step Diffusion of Movement Primitives（FODMP），通过一致性蒸馏将多步MPD扩散模型压缩到单步预测ProDMP参数，实现实时、时间结构化运动生成。

**💡 创新点**

创新点在于：① 将一致性模型与ProDMP结合，完成一次性扩散预测；② 在扩散过程直接推断运动参数而非完整轨迹；③ 通过一致性蒸馏实现低延迟高质量轨迹。

**🔧 技术方法**

使用了Transformer‑based one‑step diffusion、ProDMPs（概率动态运动原语）、一致性蒸馏、EMA目标网络以及多步教师MPD模型。

**📊 数据集**

在MetaWorld、ManiSkill仿真基准，以及真实场景的Push‑T（桌面推物）和Ball‑Catching（快速拦截球）任务上进行实验。

**📈 对比分析**

与DP、MPD、ManiCM三种基线比较，FODMP在成功率上分别在easy/medium/hard任务中达到约78%平均成功率，显著优于MPD（64%）、DP（50%）和ManiCM（34%）；推断时间为约17 ms/步，远快于MPD（≈169 ms）和DP（≈120 ms），满足实时闭环控制需求。

**⚠️ 局限性**

局限性：未在大规模任务预训练、超高维度控制或大参数网络上进行评估；目前仅验证在中小规模任务上的效果。

---

## 84. AVControl: Efficient Framework for Training Audio-Visual Controls

**arXiv ID:** 2603.24793 | [PDF](https://arxiv.org/pdf/2603.24793v1)

**作者:** Matan Ben-Yosef `[一作]` (Lightricks), Ofir Bibi `[通讯]` (Lightricks)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出AVControl框架，利用并行画布与LoRA在冻结的LTX-2音视频扩散模型上训练多种可扩展的控制器，支持深度、姿态、相机轨迹、音频强度等多模态控制。

**💡 创新点**

创新点在于：仅使用LoRA无新架构，采用并行画布通过timestep区分参考与生成token，允许精细强度调节、非像素对齐参考，并实现轻量化、可快速扩展的多模态控制。

**🔧 技术方法**

使用技术包括LTX-2联合音视频扩散模型、LoRA低秩适配器、并行canvas参考token、timestep区分、small-to-large控制网格以及自注意力中的参考-生成强度调节。

**📊 数据集**

数据集涵盖VACE Benchmark（深度、姿态、填补、延伸）、ReCamMaster Benchmark、VGGSound、HDTF，以及用于相机、轨迹和动作控制的合成/生成数据。

**📈 对比分析**

在VACE Benchmark上，AVControl在深度、姿态、填补和延伸任务上平均得分最高，且在ReCamMaster、VGGSound、HDTF等基准上均表现竞争或领先；整体训练步骤约55k，远低于VACE的200k。

**⚠️ 局限性**

局限包括受基础模型能力约束；mask编码易受相似色彩干扰；快速复杂人物动作可能导致时序抖动；相机轨迹在快速非刚性运动场景下产生拉伸/残影；未支持身份保持或参考图像条件。

---

## 85. Local learning for stable backpropagation-free neural network training towards physical learning

**arXiv ID:** 2603.24790 | [PDF](https://arxiv.org/pdf/2603.24790v1)

**作者:** Yaqi Guo `[一作]` (Delft University of Technology), Siddhant Kumar `[通讯]` (Delft University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出 FFzero 算法，融合前向-后向 (forward‑forward) 本地学习与零阶方向导数梯度估计，实现完全无反向传播的物理可实现学习框架。

**💡 创新点**

创新点在于：①将方向导数梯度近似与 FF 本地学习相结合，消除梯度在层间传播的误差；②使用固定原型向量与余弦相似度统一处理分类与回归任务；③对卷积层引入随机降维，降低原型维度；④通过实验验证了该方法在深层网络中的可扩展性与稳定性。

**🔧 技术方法**

技术手段包括：零阶方向导数梯度估计、前向‑后向学习策略、原型向量和余弦相似度的好感度计算、随机降维处理卷积特征、对比实验设置 BP+AD、BP+DD、FF+AD、FF+DD、以及使用 Neurophox 仿真平台实现光学 MZI 线性层和电光非线性。

**📊 数据集**

使用的数据集为 MNIST、FashionMNIST、以及两种合成函数（sin+cos 与 5 维复合函数）和对应的回归任务。

**📈 对比分析**

通过四种组合（BP+AD、BP+DD、FF+AD、FF+DD）对比实验。结果显示 BP+DD 随网络深度增长误差累积导致性能急剧下降，而 FF+DD（FFzero）在深层网络中保持稳定且往往优于 BP+DD，甚至在某些任务上与 BP+AD 接近；在 CNN 与回归实验中亦表现出更强的鲁棒性。

**⚠️ 局限性**

局限性包括：①梯度估计仍不如自动微分精确，整体精度低于 BP+AD；②需要预先生成随机原型，可能影响泛化；③对非线性硬件的鲁棒性尚未在实际硬件上验证；④当前框架仅适用于本地层学习，无法实现全局优化，限制了对更复杂模型（如 Transformer）的适用性。

---

## 86. PDET-LSH: Scalable In-Memory Indexing for High-Dimensional Approximate Nearest Neighbor Search with Quality Guarantees

**arXiv ID:** 2603.24920 | [PDF](https://arxiv.org/pdf/2603.24920v1)

**作者:** Jiuqi Wei `[一作]` (Oceanbase, Ant Group), Themis Palpanas `[通讯]` (Université Paris Cité)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 DET-LSH（基于动态编码树的 LSH 方案）及其并行版本 PDET-LSH，用以在高维空间中高效、准确地完成近邻搜索，解决传统 LSH 索引构造慢、查询效率低的问题。

**💡 创新点**

创新点包括：①动态编码树（DE-Tree）按维度动态划分并编码，避免多维直接划分的高成本；②将 BC 与 DM 策略相结合的两步查询：先做范围查询筛选候选，再精确排序，提升召回率与查询速度；③利用多核 CPU 的并行与 SIMD 加速，构造 PDET-LSH，实现索引与查询的显著并行加速。

**🔧 技术方法**

采用的技术：p‑stable LSH、iSAX 动态编码、QuickSelect 快速分位点选择、二叉分裂的 DE-Tree、范围查询与上下界距离裁剪、OpenMP/Pthreads 并行化、SIMD 指令集加速距离计算。

**📊 数据集**

使用的数据集包括：Msong、Deep1M、Sift10M、TinyImages80M、Sift100M、Yandex Deep500M、Microsoft SPACEV500M、Microsoft Turing-ANNS500M（尺寸从数十万到数十亿，维度从 96 到 384）。

**📈 对比分析**

与 DB-LSH、LCCS-LSH、PM-LSH（三种主流 LSH 方法）以及 HNSW、IMI-OPQ（非 LSH 方法）进行对比。结果显示：DET-LSH 在索引时间上最快，查询速度比单线程 LSH 方法提升 1–2 倍；PDET-LSH 在索引/查询上分别比最优 LSH 方法快多达 40×/62×；在相同召回率下，累计查询成本明显低于其他方法。

**⚠️ 局限性**

局限性：①需要在内存中持有完整数据集，内存占用较高；②并行实现受资源竞争影响，线程数过多时收益递减；③参数（L、K、β 等）对性能与准确性影响大，需人工调优；④不支持动态增删数据的在线更新。

---

## 87. Characterization of Off-wafer Pulse Communication in BrainScaleS Neuromorphic System

**arXiv ID:** 2603.24854 | [PDF](https://arxiv.org/pdf/2603.24854v1)

**作者:** Bernhard Vogginger `[一作]` (Technische Universität Dresden), Christian Mayr `[通讯]` (Technische Universität Dresden)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文对BrainScaleS wafer-scale 神经形态系统的离板脉冲通信（Layer‑2网络）进行了系统化的特性测量与验证，包括吞吐量、延迟、抖动与丢包率，并通过循环实验评估其对实际神经网络模拟的影响。

**💡 创新点**

创新点在于：①将高速可配置的FPGA‑基包络网络与大容量回放/跟踪存储结合，实现在10⁴加速因子下的长时序刺激与记录；②提供完整的测量框架，首次量化不同脉冲分布（正则/泊松）对通信质量的影响；③通过循环实验直接评估通信失真对神经网络动态的影响，为未来网络映射提供实证依据。

**🔧 技术方法**

主要技术包括：Xilinx Kintex‑7 FPGA、LVDS双向通道、定制UDP栈与应用层、双缓冲播放/追踪内存、时间戳模式、回波回包、数据包分帧与CRC校验、以及对HICANN接口的多路复用和缓冲管理。

**📊 数据集**

使用的数据集：①基于Poisson和正则间隔的人工脉冲序列；②500个AdEx神经元的网络模拟（通过NEST/ PyNN 生成），用以产生自发、AI、SR等多种动态状态的参考脉冲训练与测试。

**📈 对比分析**

比较方法：对单节点与多节点（8 HICANN）下的吞吐、丢包、延迟、抖动进行测量，并用系数变异（CV）与ISI直方图评估原始与回环后的脉冲分布；在循环实验中将硬件记录的脉冲与软件模拟脉冲进行时间对齐与统计对比。性能结果显示：单节点下吞吐可达17.8 M事件/秒（1.78 kHz 生物率），延迟≈230 ns，抖动≈8 ns；回放/跟踪分别支持125 M事件/秒与250 M事件/秒；在多节点情形下吞吐饱和12.5 M事件/秒，丢包率在1.6 kHz 以上开始显著；CV 下降但仍保持 1.0 以上表明动态性未被完全破坏。

**⚠️ 局限性**

局限性包括：①单HICANN通道吞吐受LVDS物理带宽限制（1.8 kHz 以上即出现丢包）；②时间戳模式尚未启用，导致无法使用双脉冲包实现更高吞吐；③回放内存格式未充分利用 32 bit 词，潜在可提升吞吐；④高峰突发率时仍会出现丢包与延迟飙升；⑤在多节点下回放组装导致的额外抖动与延迟，需进一步优化帧打包与批处理策略。

---

## 88. Attention-based Pin Site Image Classification in Orthopaedic Patients with External Fixators

**arXiv ID:** 2603.24815 | [PDF](https://arxiv.org/pdf/2603.24815v1)

**作者:** Yubo Wang `[一作]` (Aalborg University), Ming Shen `[通讯]` (Aalborg University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

收集外固定器针位创面感染图像数据集，构建基于注意力机制与高效冗余重建卷积（ERRC）的轻量化深度学习模型，用于将针位图像分为感染组（Group A）和非感染组（Group B）。

**💡 创新点**

创新点在于：①提出ERRC模块，在保持特征丰富度的同时显著降低参数量并减少卷积特征冗余；②将通道+空间注意力（CBAM）与ERRC结合，专门抑制金属针干扰，聚焦皮肤周围的感染特征；③在小样本环境下采用Focal Loss与数据增强提升不平衡数据的分类性能。

**🔧 技术方法**

使用的技术包括：卷积神经网络（改进的MobileNet‑style结构）、CBAM注意力模块、ERRC特征重建卷积、Focal Loss、图像增强（翻转、旋转、亮度/对比度调整、Cutout）、Grad‑CAM可视化、t‑SNE特征聚类分析。

**📊 数据集**

数据集为Nationwide Children’s Hospital（NCH）数据集，原始1554张RGB高分辨率图像，预处理后保留572张，最终通过增强扩充至3348张，其中标注为201张感染（Group A）和465张非感染（Group B）共666张图像。

**📈 对比分析**

与VGG‑16/19、ResNet‑50、EfficientNet‑V2等基线模型对比，本文模型仅5.77 M参数即可获得AUC 0.975、F1 0.927、精度 92.5%、召回 93.2%；相较于基线，AUC提升≥0.05、参数量减少≈70%。

**⚠️ 局限性**

局限性包括：①数据集规模有限，缺乏多设备、多环境、不同肤色与抗生素使用等因素的全面验证；②仅基于视觉标注，未与临床实验室检测结果进行一致性验证；③仅区分感染与非感染，未细分感染程度或其他创面状态，缺乏更细粒度的诊断能力。

---

## 89. Evaluating adaptive and generative AI-based feedback and recommendations in a knowledge-graph-integrated programming learning system

**arXiv ID:** 2603.24940 | [PDF](https://arxiv.org/pdf/2603.24940v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

---

## 90. Multi-LLM Query Optimization

**arXiv ID:** 2603.24617 | [PDF](https://arxiv.org/pdf/2603.24617v1)

**作者:** Arlen Dean `[一作]` (Washington University in St. Louis), Yuqing Liu `[通讯]` (University of Michigan)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

研究在多模型LLM环境下，如何在保证每个标签误判率不超过预设阈值的前提下，最小化查询总成本，提出离线查询规划框架。

**💡 创新点**

①证明该规划问题为NP‑hard；②给出基于Chernoff上界的可计算松弛，证明在高可靠性区间内近似无损；③构建渐进完全多项式时间近似方案（AFPTAS）。

**🔧 技术方法**

联合束缚、Chernoff指数上界、对冲击参数离散化、动态规划（无界背包），以及可行域保真性证明。

**📊 数据集**

文中未给出具体真实数据集，研究以理论模型和通用假设为基础，可在医疗诊断、内容分类、文档审核等领域应用。

**📈 对比分析**

没有实验对比；主要通过理论证明展示：在误判阈值足够小的情况下，使用Chernoff松弛得到的最优查询成本与原问题差距趋近于1，AFPTAS实现的成本≤(1+ε)乘以松弛最优值。

**⚠️ 局限性**

局限包括：需要满足对数似然比有界的假设；仅适用于离线、一次性查询规划；高可靠性区间外松弛可能导致显著成本上升；模型对真实LLM异质性与API费用的实际影响未在实验中验证。

---

## 91. TIGeR: A Unified Framework for Time, Images and Geo-location Retrieval

**arXiv ID:** 2603.24749 | [PDF](https://arxiv.org/pdf/2603.24749v1)

**作者:** David G. Shatwell `[一作]` (Institute of Artificial Intelligence, University of Central Florida), Mubarak Shah `[通讯]` (Institute of Artificial Intelligence, University of Central Florida)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了Geo‑Time Aware Image Retrieval（时空感知图像检索）框架，统一建模图像、地理位置和时间，能够在未知位置上检索与查询图像同一地点但不同时间的图像，并同时完成时空预测与地理定位。

**💡 创新点**

核心创新在于：①使用多模态Transformer实现跨模态自注意力，直接学习图像–位置–时间的联合嵌入；②在对齐阶段采用多对比损失和软分类监督，强化位置与时间的连续性；③提出自适应重排序机制，利用分类置信度动态加权检索；④构建全球覆盖、时空多样化的4.5M训练+86k测试AMOS子集，填补了现有数据集在地理与时间分布上的空白。

**🔧 技术方法**

技术手段包括：CLIP ViT视觉编码器、随机傅里叶特征映射位置与时间、共享多模态Transformer、InfoNCE对比损失、HEALPix地理分块、时间平面tori分箱、软标签分类目标、熵自适应重排序等。

**📊 数据集**

主要使用数据集为从AMOS摄像头中清洗、过滤并均衡后的4.5M训练图像与86k测试图像；对比实验还使用CVT社交媒体图像作为额外检索基准。

**📈 对比分析**

与Zhai等、Time‑Loc、GT‑Loc、GeoCLIP等基线相比，在geo‑time检索中R@1从2.60%提升至3.51%，R@10从13.70%提升至37.51%；在时间预测中平均误差下降约10%；在地理定位中平均误差从668km降至203km，整体性能显著优于现有最先进方法。

**⚠️ 局限性**

局限性包括：①时间信息对地理定位的辅助作用有限；②模型对极端光照/极端季节变化仍易失效；③依赖固定摄像头位置的时空分布，难以推广至移动摄像场景；④对多语言/文化背景的图像内容缺乏针对性。

---

## 92. GoldiCLIP: The Goldilocks Approach for Balancing Explicit Supervision for Language-Image Pretraining

**arXiv ID:** 2603.24804 | [PDF](https://arxiv.org/pdf/2603.24804v1)

**作者:** Deen Dayal Mohan `[一作]` (Samsung Electronics), Suren Kumar `[通讯]` (Samsung Electronics)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 GoldiCLIP 框架，在仅 30M 图文对的开放数据集上，通过多监督联合训练提升视觉‑语言模型性能。

**💡 创新点**

创新点：① 将自蒸馏扩展至文本条件特征；② 将解码器任务通过文本编码器实现，增强对非 Caption 查询的泛化；③ 采用不确定性权重自动平衡多任务损失，避免手工调参。

**🔧 技术方法**

使用技术包括 CLIP 双编码器、ViT 视觉编码器、跨注意力池化、文本条件自蒸馏、VQA 解码器、EMA 教师、以及多任务学习与不确定性权重。

**📊 数据集**

使用 Open‑30M 数据集（CC3M、CC12M、YFCC15M 共 27M 图像），并生成多句短长标题、VQA 任务以及对象级检测框。

**📈 对比分析**

在 MSCOCO、Flickr30k、DOCCI‑FG、IIW‑FG、DCI、Retrieval‑VQA、语义分割和零样本分类等基准上与 FLAIR、COSMOS、SigLIP2、Perception Encoder 等同规模或更大规模模型比较，文本‑图像检索 R@1 提升 2.2‑5.9 分，细粒度检索提升 3‑4 分，VQA 检索提升 5 分，语义分割提升 1.5 分，整体表现优于同规模基准。

**⚠️ 局限性**

局限性：零样本分类仍落后于大规模模型，受限于数据集概念多样性；实验仅在 ViT‑B/16 基础上验证，未探究更大模型的可扩展性。

---

## 93. Amplified Patch-Level Differential Privacy for Free via Random Cropping

**arXiv ID:** 2603.24695 | [PDF](https://arxiv.org/pdf/2603.24695v1)

**作者:** Kaan Durmaz `[一作]` (Technical University of Munich), Stephan Günnemann `[通讯]` (Technical University of Munich)

**通讯引用:** 14983 | [OpenAlex ID](https://openalex.org/A5074504351)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出在差分隐私训练中利用图像局部裁剪来提升隐私保护，定义了面向补丁的相邻关系并证明随机裁剪可作为额外的隐私放大机制。

**💡 创新点**

创新点在于将裁剪这一常见的数据增强视为隐私放大器，并在补丁级别构造相邻关系，从而实现无需修改模型或训练流程即可获得更强的隐私保证。

**🔧 技术方法**

使用的技术包括差分隐私随机梯度下降（DP‑SGD）、裁剪算子、基于补丁的相邻关系以及隐私放大与合成的理论分析。

**📊 数据集**

实验数据集包括Cityscapes和A2D2（用于语义分割），以及在MNIST上的验证，模型采用DeepLabV3+和PSPNet。

**📈 对比分析**

通过与标准DP‑SGD、随机噪声输入以及Gaussian噪声增强等方法比较，Patch‑Level DP在相同隐私预算下显著提升了模型性能（如mIoU提升高达81%），并在多种网络和数据集上保持了优越性。

**⚠️ 局限性**

局限性包括对补丁重叠程度的二值化处理、需要预先知道补丁大小以及仅适用于裁剪能保持语义信息的高分辨率图像。

---

## 94. A formalization of the Gelfond-Schneider theorem

**arXiv ID:** 2603.24823 | [PDF](https://arxiv.org/pdf/2603.24823v1)

**作者:** Michail Karatarakis `[一作]` (Radboud University), Freek Wiedijk `[通讯]` (Radboud University)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在 Lean4 证明助手中实现了 Hilbert 第七问题的完整形式化，即 Gelfond–Schneider 定理，证明对任意满足条件的代数数 α≠0,1 与 β 皆代数且 β 非有理时，α^β 为超越数。

**💡 创新点**

创新点包括：①首次将该深层的超越数理论结果形式化；②在 Lean 中将传统的可去奇点处理改为全局可定义的分块函数，显著降低了证明状态复杂度；③采用数域整数环的 Siegel 引理以及规范嵌入定义“house”来精准追踪常数与界限，保证证明可计算性。

**🔧 技术方法**

使用的技术主要有 Lean4 证明语言与 mathlib 代数/数论库、复分析函数库、规范嵌入、Siegels lemma、Cauchy 积分公式、辅助函数构造、可去奇点的分块拼接与全局解析性证明。

**📊 数据集**

本文并未使用传统意义上的实验数据集，而是利用公开的 Lean4 代码仓库（https://github.com/mkaratarakis/mathlib4/tree/9710b82b966813f7664819bc5d25b82d209a05e7/Mathlib/NumberTheory/Transcendental/GelfondSchneider）作为验证素材。

**📈 对比分析**

相较于以往仅在纸上或非形式化的证明，本文实现了完整可验证的证明，虽然未给出计算性能指标，但证明的可复现性和可检查性大大提升了结果的可靠性；在现有的数论/超越数形式化工作中，它是唯一覆盖此深层定理的实例。

**⚠️ 局限性**

限制包括：仅适用于经典指数函数，尚未推广至椭圆曲线、p‑adic 或更一般的 Meromorphic 函数；常数管理繁琐，导致证明文件庞大；目前缺乏自动化工具辅助常数优化与证据生成。

---

## 95. VideoTIR: Accurate Understanding for Long Videos with Efficient Tool-Integrated Reasoning

**arXiv ID:** 2603.25021 | [PDF](https://arxiv.org/pdf/2603.25021v1)

**作者:** Zhe Gao `[一作]` (Nanjing University), Dacheng Tao `[通讯]` (Nanyang Technological University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 VideoTIR 框架，实现多轮多工具推理，配合内部检索工具集和文本路由器完成长视频问答；通过 RL（TAGPO）学习工具调用策略，并用 sandbox 生成多工具轨迹进行 SFT 与 RL 预训练。

**💡 创新点**

创新点：① 统一的多层级内部检索工具（浏览、分段、帧检索、放大）与文本路由器，模仿人类粗细分辨；② 引入 Toolkit Action Group Policy Optimization (TAGPO) ，在同类工具调用间进行分组优势估计，抑制工具误用与过度使用；③ sandbox‑based 轨迹合成框架，利用外部 LLM 自动生成高质量工具调用轨迹，为零 RL 与 SFT cold‑start 提供训练数据。

**🔧 技术方法**

技术手段：多模态大语言模型 Qwen2.5‑VL + RL（GRPO + TAGPO）; 轨迹合成与评判使用外部 LLM；内部视觉工具实现；训练框架 LLaMaFactory / VeRL；多轮交互设计；格式化奖励与工具使用奖励。

**📊 数据集**

使用的数据集包括：MVBench、Video‑MME、LongVideoBench；训练与合成轨迹所用：LLaVa‑Video‑155k、CLEVRER、PerceptionTest、VideoSIAM、NextQA、NextGQA、VideoXUM、QV‑Highlights；RL 采样数据来自 LLaVa‑Video‑155k、CLEVRER、PerceptionTest、VideoSIAM。

**📈 对比分析**

在三个基准上与基线（Qwen2.5‑VL、InternVL3、Video‑MTR 等）比较，VideoTIR 在 低帧率/低分辨率输入下显著提升短/中视频性能，尤其在长视频上提升明显；TAGPO 使工具学习更快，收敛速度提升约 50%，最终准确率比单纯 GRPO 高 1–2%。

**⚠️ 局限性**

局限性：① 对 3B 小模型零 RL 训练困难，需强制 SFT；② 对工具调用格式化高度依赖 SFT 训练，若格式不规范奖励低；③ 需要大量生成轨迹，训练成本高；④ 仅使用内部工具，若面对外部知识源或更复杂的多模态任务仍受限。

---

## 96. MobileDev-Bench: A Comprehensive Benchmark for Evaluating Language Models on Mobile Application Development

**arXiv ID:** 2603.24946 | [PDF](https://arxiv.org/pdf/2603.24946v1)

**作者:** Moshood A. Fakorede `[一作]` (Louisiana State University), Umar Farooq `[通讯]` (Louisiana State University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并发布了一个面向移动应用的自动修复基准，收集了384个真实的issue–PR对，并提供了执行验证和人工评估。

**💡 创新点**

创新点在于关注移动app的多文件、多artifact修复需求，包含人工难度分层、编译可执行验证以及跨语言、跨框架的测试环境。

**🔧 技术方法**

使用了LLM本地化–修复流水线（改造自Agentless）、Tree‑sitter解析、Docker化构建环境，以及四款主流LLM（Claude Sonnet、GPT‑5.2、Gemini Flash、Qwen3‑Coder）。

**📊 数据集**

数据集为来自18个开源生产移动仓库（Android Native、React‑Native、Flutter）的384个手工验证的issue–PR实例。

**📈 对比分析**

通过对比四款LLM在基准上的解法成功率，整体解法率仅为3.39%–5.21%，显著低于传统库级基准，且模型在多文件、多artifact任务上的定位召回极低。

**⚠️ 局限性**

局限性包括仅覆盖Android及跨平台框架，缺乏iOS项目；依赖测试套件可能无法覆盖全部bug；仅基于单次修复流水线，未考虑交互式调试。

---

## 97. Formal Semantics for Agentic Tool Protocols: A Process Calculus Approach

**arXiv ID:** 2603.24747 | [PDF](https://arxiv.org/pdf/2603.24747v1)

**作者:** Andreas Schlapbach `[一作]` `[通讯]` (Swiss Federal Railways), Andreas Schlapbach (Swiss Federal Railways)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

本文提出了基于进程代数的正式语义框架，证明了Schema‑Guided Dialogue (SGD) 与 Model Context Protocol (MCP) 在一定映射下的结构同构，并找出了反向映射的缺陷。

**💡 创新点**

创新点包括首次给出SGD和MCP的过程代数语义、证明它们的结构化 bisimulation、识别并填补了MCP的表达缺口、提出五项类型系统扩展以实现完全等价，并给出了安全属性的过程不变量证明。

**🔧 技术方法**

使用了进程代数（π‑calculus）、强 bisimulation、类型系统扩展以及形式化证明。

**📊 数据集**

主要参考了SGD数据集及OpenAPI/JSON Schema，但本文并未进行数据实验，而是以形式化推理为主。

**📈 对比分析**

通过结构化 bisimulation 的证明来比较两种协议的行为等价性，理论上证明了等价性；没有提供运行时性能指标。

**⚠️ 局限性**

局限性在于映射仍是部分且不可逆，无法完整描述资源、能力协商、动态发现等MCP原语；缺乏对动态 schema 进化、概率行为和多代理协作的支持。

---

## 98. When Consistency Becomes Bias: Interviewer Effects in Semi-Structured Clinical Interviews

**arXiv ID:** 2603.24651 | [PDF](https://arxiv.org/pdf/2603.24651v1)

**作者:** Hasindri Watawana `[一作]`, Esaú Villatoro-Tello `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究探讨在半结构化抑郁症诊断对话中，模型仅使用采访者提问能否准确区分抑郁与非抑郁。

**💡 创新点**

创新点在于量化并跨数据集验证采访者提问产生的 prompt‑induced bias，并证明该偏差与模型架构无关。

**🔧 技术方法**

采用 Longformer（稀疏注意力 transformer）和 GCN（图卷积网络）两种模型进行对比实验。

**📊 数据集**

使用三大公开抑郁对话语料库：ANDROIDS、DAIC‑WOZ 与 E‑DAIC。

**📈 对比分析**

通过训练 participant‑only 与 interviewer‑only 两种变体，在开发集和测试集上计算宏观 F1；发现 interviewer‑only 在多数数据集表现更好或相当，证实了偏差存在。

**⚠️ 局限性**

局限性包括依赖 ASR 生成的文本导致转写误差，未考虑多模态信息，以及缺乏完整手工双语录音转写。

---

## 99. Scalable Air-to-Ground Wireless Channel Modeling Using Environmental Context and Generative Diffusion

**arXiv ID:** 2603.24620 | [PDF](https://arxiv.org/pdf/2603.24620v1)

**作者:** Jingyi Tian `[一作]` (University of Victoria), Lin Cai `[通讯]` (University of Victoria)

**通讯引用:** 99241 | [OpenAlex ID](https://openalex.org/A5100773343)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种基于真实环境数据（DEM、土地覆盖、气象）的环境感知空对地通道模型，并通过扩散模型实现对任意卫星‑地面终端组合的快速信道损耗预测。

**💡 创新点**

①将多源环境信息与射线追踪结果结合，完整描述遮蔽、反射、衍射等路径特征；②采用扩散模型代替射线追踪，兼顾高精度与大规模实时计算；③引入双头U‑Net与v‑parameter化，解决LOS零损耗与阻塞识别的难点。

**🔧 技术方法**

射线追踪、数字地形模型分析、地形反射评分、衍射损耗模型（Delta‑Bullington）、植被衰减、气象衰减、扩散模型（DDIM）、双头U‑Net、v‑parameterization、物理信息插值。

**📊 数据集**

加拿大北极区DEM、土地覆盖与气象数据；Starlink、OneWeb以及LTE测量（Ping、SNR）数据用于模型验证。

**📈 对比分析**

通过与实测（Starlink Ping、OneWeb SNR、LTE SNR）的相关性评估，模型相关系数均超过0.7；扩散模型推理速度比射线追踪提升≈480×，在1%–4%观测稀疏率下MAE均低于3 dB，显示出良好的性能与可扩展性。

**⚠️ 局限性**

对高分辨率DEM和土地覆盖的依赖；季节性植被变化的适配性有限；在极端遮挡或高分辨率需求下预测误差仍存在；需要至少部分观测作为条件，缺乏观测时预测可靠性下降。

---

## 100. Shortest Paths in Geodesic Unit-Disk Graphs

**arXiv ID:** 2603.24872 | [PDF](https://arxiv.org/pdf/2603.24872v1)

**作者:** Bruce W. Brewer `[一作]` (University of Utah), Haitao Wang `[通讯]` (University of Utah)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究在多边形域内的 geodesic 单位圆图（geodesic unit‑disk graph）上从源点 s 计算到所有顶点的单源最短路径（SSSP）问题，分别给出加权和无权两种版本的子二次时间算法。

**💡 创新点**

创新点包括：① 在简单多边形内实现 O(m + n log²n log²m) 时间的加权 SSSP；② 在无权简单多边形和有洞多边形内分别实现 O(m + n logn log²m) 和 O(√n (n+m) log(n+m)) 时间的 SSSP；③ 提出了删除‑只 geodesic 单位圆范围空查询的数据结构；④ 开发了隐式加权 geodesic Voronoi 图构造；⑤ 将 Bentley 的对数方法扩展到支持插入和 delete‑min 的优先队列更新。

**🔧 技术方法**

技术上主要使用：平衡多边形分解（BPD）与 GH 数据结构；隐式构造的 geodesic Voronoi 图；基于三种顺序（≺_R、≺_L、≈）的结构化查询；分层分解的动态数据结构（维护 S_i、C_i、Ψ(S) 等）；对数方法的多层重平衡与空间共享；以及多重堆与最近邻查询结合的增量更新。

**📊 数据集**

论文未在实验中使用公开数据集，而是以理论证明为主；主要涉及假设多边形 P、点集 S 与源点 s 的符号性参数 m、n。

**📈 对比分析**

方法相比传统构造完整图并使用 Dijkstra 的 O(n²) 时间，显著降低到子二次时间；加权单纯多边形版本与 Euclidean 版本最优算法的 loglog n 因子相当；无权多边形有洞版本突破了此前 O(n³/²) 的瓶颈。实验或定量对比未给出，但理论上已证明时间复杂度优于现有最优。

**⚠️ 局限性**

局限性：算法仅适用于多边形域（单连通或多连通）且要求点集满足一般位置假设；对非多边形或更复杂障碍的推广尚未实现；数据结构实现复杂，理论上可行但工程化难度大；在极端极限（如 n ≫ m 或 m ≫ n²）下，常数项可能较大。

---

## 101. Can LLMs Beat Classical Hyperparameter Optimization Algorithms? A Study on autoresearch

**arXiv ID:** 2603.24647 | [PDF](https://arxiv.org/pdf/2603.24647v1)

**作者:** Fabio Ferreira `[一作]` (ELLIS Institute), Arber Zela `[通讯]` (ELLIS Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文在Karpathy的autoresearch任务上，对比了经典HPO、LLM驱动和混合两种方法，评估它们在固定搜索空间和代码编辑两种设置下的表现，并提出了一种将CMA-ES内部状态与LLM共享的混合优化器。

**💡 创新点**

创新点在于：①系统化地将LLM与经典优化器结合，首次通过共享CMA-ES完整内部状态（均值、步长、协方差）来让LLM参与搜索；②在同一训练预算和种子下，构建了涵盖九种方法的统一基准；③展示了LLM规模对代码编辑和混合优化的不同影响，揭示了LLM在不直接编辑代码时并不需要过大模型。

**🔧 技术方法**

主要技术包括：使用Qwen3.5-27B（和0.8B）自托管LLM进行推理；采用Optuna的CMA-ES、TPE、SMAC、随机搜索等经典采样器；LLM直接编辑训练脚本或在固定HP空间内生成建议；混合方法通过每30%试验让LLM接收CMA-ES的完整内部状态并可覆盖其建议；利用AST解析自动提取14个超参数；通过OOM率、搜索多样性等指标对方法进行评估。

**📊 数据集**

使用的主要数据集是NanoChat—a 50M参数decoder-only transformer在FineWeb上训练的任务，评估指标为验证集bits-per-byte（val_bpb）。

**📈 对比分析**

在相同24小时GPU训练预算、3个随机种子下，经典HPO（CMA-ES、TPE）在固定搜索空间内明显优于纯LLM方法；Karpathy的代码编辑LLM（27B）与经典方法竞争力接近；混合方法（CMA-ES+LLM）实现了最优结果，0.8B版本甚至优于27B，表明LLM只需在少数试验中提供指导即可显著提升稳定性和收敛速度。

**⚠️ 局限性**

局限性包括：超参数范围仍需人工设置；LLM仅在固定HP空间内缺乏优势，无法完全替代经典搜索；混合方法依赖CMA-ES的状态表示，可能不易推广到其他优化器；对前沿大模型的评估缺失，未来需要在更强LLM上进一步验证。

---

## 102. Is Geometry Enough? An Evaluation of Landmark-Based Gaze Estimation

**arXiv ID:** 2603.24724 | [PDF](https://arxiv.org/pdf/2603.24724v1)

**作者:** Daniele Agostinelli `[一作]` (Università Politecnica delle Marche), Maura Mengoni `[通讯]` (Università Politecnica delle Marche)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

对基于面部关键点的注视估计进行系统评估，提出轻量化的MLP与XGBoost回归模型，并与传统ResNet18基线进行对比。

**💡 创新点**

通过标准化的关键点提取与归一化管道证明稀疏几何特征即可实现与深度CNN相近的跨域泛化；提出双分支Siamese MLP显式捕捉双眼几何关系。

**🔧 技术方法**

使用MediaPipe关键点检测、PnP姿态估计、投影归一化、极限梯度提升树、残差MLP、Permutation Feature Importance等技术。

**📊 数据集**

Gaze360、ETH-XGaze、GazeGene 三大公开注视数据集。

**📈 对比分析**

在域内与ResNet18比较，MLP性能略逊；在跨域测试中Siamese MLP MAE仅比ResNet18高约2–3度，显示更强泛化；XGBoost性能最低。

**⚠️ 局限性**

关键点检测噪声导致数据失真，尤其在极端视角缺失，限制了在域内的精度；需要提升检测器鲁棒性并扩大数据多样性。

---

## 103. Scalable Object Relation Encoding for Better 3D Spatial Reasoning in Large Language Models

**arXiv ID:** 2603.24721 | [PDF](https://arxiv.org/pdf/2603.24721v1)

**作者:** Shengli Zhou `[一作]` (Southern University of Science and Technology), Yang Liu `[通讯]` (Peking University)

**通讯引用:** 85567 | [OpenAlex ID](https://openalex.org/A5100355964)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于四元数的旋转位置编码（QuatRoPE）以及其隔离门控扩展（IGRE），并创建了仅测试空间推理能力的ASR基准，提升了3D LLM在空间推理任务中的表现。

**💡 创新点**

创新点包括：①将绝对3D坐标通过四元数旋转嵌入Token，使得注意力机制天然得到对象间相对位置；②通过IGRE在语言RoPE与QuatRoPE之间隔离维度，减少互相干扰；③设计仅关注空间关系的ASR基准，客观评估空间推理能力。

**🔧 技术方法**

技术主要有：四元数旋转位置编码（QuatRoPE）、隔离门控RoPE扩展（IGRE）、LoRA微调、与大语言模型（LLaMA‑3.2‑1B、Vicuna‑7B）结合的点云输入方案。

**📊 数据集**

使用了多种3D视觉‑语言数据集：ScanRefer、Multi3DRef、ScanQA、SQA3D、Scan2Cap、ReferIt3D、Chat‑Scene 对齐任务，并在这些数据集上训练和评估。

**📈 对比分析**

与Chat‑Scene、3DGraphLLM等基线进行对比，实验显示：在ScanRefer、Multi3DRef、SQA3D等3D VG/ VQA任务中，QuatRoPE+IGRE 在绝大多数指标上提升 5–15% 以上；在零样本 ASR 基准中，提升 10–20% 左右，显著验证了空间推理能力的提升。

**⚠️ 局限性**

局限性包括：① 依赖于大语言模型的预训练，若缺少足够的3D‑文本对齐数据，效果有限；② 四元数旋转虽线性扩展，但在极大物体数场景下仍可能出现注意力计算瓶颈；③ 当前仅在英文和室内场景上验证，未评估跨语言或室外、动态场景的鲁棒性。

---

## 104. Shopping with a Platform AI Assistant: Who Adopts, When in the Journey, and What For

**arXiv ID:** 2603.24947 | [PDF](https://arxiv.org/pdf/2603.24947v1)

**作者:** Se Yan `[一作]` (Peking University), Wenyu Zhou `[通讯]` (Zhejiang University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了中国最大线上旅游平台Ctrip嵌入的LLM购物助手Wendao的用户采用率、使用时机以及用途，揭示了高龄、女性及高粘性用户的高采用率、聊天与搜索交替使用以及对探索性、难以关键词化任务的偏好。

**💡 创新点**

首次大规模描述性揭示平台嵌入式购物AI的使用模式：与一般AI工具相反的年龄性别梯度、聊天与搜索的交互式使用以及对探索性需求的高度聚焦，提供了对技术采用与搜索行为的新视角。

**🔧 技术方法**

利用LLM（类似ChatGPT）构建的对话式助手，并采用描述性统计、线性/逻辑回归、随机森林分类器以及核密度估计等分析技术，研究聊天与搜索在购买旅程中的时序关系与转化效应。

**📊 数据集**

使用了Ctrip平台31,142,353名用户在2025年7月10日至24日两周内的登录、搜索、聊天记录和订单等四个关联数据集，并延伸至6月10日至8月15日的完整活动窗口，以构建完整的用户旅程。

**📈 对比分析**

通过描述性分析、回归模型和随机森林对比AI聊天与传统搜索的时间位置、使用频率和转化率，随机森林AUC从0.635提升至0.962，显示参与度特征对采用预测的显著优势；核密度估计展示聊天与搜索在旅程中相互交织的时序特征。

**⚠️ 局限性**

主要局限在于缺乏因果识别，采用率可能受曝光频次影响；聊天数据仅覆盖有聊天记录的用户，无法全面反映所有用户；实验设计未能完全分离聊天与搜索的相互影响；研究集中于旅游平台，结果的普适性仍需验证。

---

## 105. LLaVA-LE: Large Language-and-Vision Assistant for Lunar Exploration

**arXiv ID:** 2603.24696 | [PDF](https://arxiv.org/pdf/2603.24696v1)

**作者:** Gokce Inal `[一作]` (Ohio State University), Alper Yilmaz `[通讯]` (Ohio State University)

**通讯引用:** 8896 | [OpenAlex ID](https://openalex.org/A5008672128)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了面向月球探测的多模态视觉语言模型LLaVA-LE，并发布了大型实测月球数据集LUCID。

**💡 创新点**

创新点在于构建了真实高分辨率月球影像与科学描述相配对的100k级数据集，并采用两阶段领域对齐+指令微调，显著提升领域特定推理能力。

**🔧 技术方法**

使用了CLIP视觉编码器、LLaVA语言模型、LoRA轻量化适配、GPT‑5/5.1生成的科学描述与问答，进行自监督对齐与指令微调。

**📊 数据集**

数据集为LUCID（96k图像‑科学标题对）和LUCID‑VQA（81k问答对），包含LROC影像、GRAIL重力异常、LOLA坡度等多源配准数据。

**📈 对比分析**

通过GPT‑4与Gemini两大LLM评判，Stage2 LLaVA‑LE在细节、对话与推理三个类别上均超过基线，整体分数提升3.3倍，推理类甚至超过评判者参考分。

**⚠️ 局限性**

局限在于仅针对月球影像，缺少跨行星适用性；训练依赖大规模生成文本，可能带来假设偏差；对低分辨率或非光学模态的鲁棒性未充分验证。

---

## 106. FinMCP-Bench: Benchmarking LLM Agents for Real-World Financial Tool Use under the Model Context Protocol

**arXiv ID:** 2603.24943 | [PDF](https://arxiv.org/pdf/2603.24943v1)

**作者:** Jie Zhu `[一作]` (Alibaba Cloud Computing), Chi Zhang `[通讯]` (Alibaba Cloud Computing)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

FinMCP-Bench 构建了一个包含 613 条单工具、多工具和多轮对话样本的金融工具调用基准数据集，用于评估 LLM 在真实金融场景中的推理与工具使用能力。

**💡 创新点**

创新点在于结合 Model Context Protocol（MCP）实现真实金融工具调用，提出链式和角色扮演两种合成多工具、多轮样本的方法，并引入了针对工具调用准确性与推理过程的专门评估指标。

**🔧 技术方法**

使用大规模预训练语言模型（Qwen3 系列、DeepSeek‑R1、GPT‑OSS‑20B、Seed‑OSS‑36B）进行工具调用生成，并通过 LLM 辅助的工具依赖图构建与查询生成技术实现样本合成。

**📊 数据集**

数据来源为 Yingmi 基金 Qieman APP 的 10,000 条真实交互日志与人工合成样本，共构成 65 个真实金融工具的 613 条评测样本，覆盖 10 个主场景与 33 个子场景。

**📈 对比分析**

通过比较六大模型在单工具、多工具和多轮样本上的 Tool Recall、Precision、F1、Exact Match Rate 等指标，发现 Qwen3‑235B 在多工具和多轮场景中表现最佳，模型规模并非唯一决定因素，精确匹配率仍有提升空间。

**⚠️ 局限性**

局限在于合成过程仍需人工审核，且多轮对话中精确匹配率低，表明 LLM 在长篇对话与复杂工具链协同上的推理与执行仍存在挑战。

---

## 107. COIN: Collaborative Interaction-Aware Multi-Agent Reinforcement Learning for Self-Driving Systems

**arXiv ID:** 2603.24931 | [PDF](https://arxiv.org/pdf/2603.24931v1)

**作者:** Yifeng Zhang `[一作]` (National University of Singapore), Guillaume Sartoretti `[通讯]` (National University of Singapore)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了 COIN——一种面向多车自主驾驶系统的协同交互感知多智能体强化学习框架，设计了 CIG‑TD3 算法，联合优化个体导航与全局协作目标，并在 MetaDrive 模拟器及真实机器人平台上实现了端到端连续控制。

**💡 创新点**

创新点包括：①双层交互感知中心化评论器（利用 CVAE 捕获微观对偶交互、GAT 把握宏观全局依赖）；②引入对比基准的 Counterfactual Individual‑Global TD3，使局部与全局价值同步训练并通过对比值精确分配信用；③将局部与全局目标直接融合于策略更新，避免单目标学习导致的自利或低方差问题；④在稠密交互环境中首次实现了完整端到端多智能体协同导航，并在真实机器人中验证迁移效果。

**🔧 技术方法**

使用技术包括：CTDE 框架、TD3、Conditional Variational AutoEncoder (CVAE)、Graph Attention Network (GAT)、对比基准（counterfactual baseline）、双重 Q 网络与延迟策略更新、目标网络软更新、经验回放；基准对比方法有 IPPO、ITD3、CPPO、MFPO、CoPO、TraCo；实验平台为 MetaDrive 交叉口、环形与瓶颈三种稠密交通场景；真实实验采用 ROS、Jetson Nano、Mecanum 机器人和 OptiTrack 运动捕捉。

**📊 数据集**

数据集主要是 MetaDrive 仿真环境中的三种稠密交通场景（交叉口 30 车、环形 40 车、瓶颈 20 车），以及在真实机器人模拟中使用的 8 辆小型 Mecanum 机器人，采用真实轨迹与仿真轨迹混合的混合实验。

**📈 对比分析**

对比 6 项评估指标（成功率、越轨率、碰撞率、效率、安全性、平均行驶步数）与 IPPO、ITD3、CPPO、MFPO、CoPO、TraCo 进行实验。COIN 在所有三种场景下均获得最高成功率（最高 96.33%）和最低碰撞率（最低 1.57%），并显著提升效率与安全性，统计检验表明改进在 p<5.56e-4 下显著。对不同车数的零样本泛化实验中 COIN 仍保持领先；真实机器人实验显示路径平滑、碰撞率低，验证模型迁移能力。

**⚠️ 局限性**

局限性包括：①仅在 MetaDrive 仿真与小规模机器人平台验证，缺乏大规模真实交通系统的评估；②端到端控制未显式约束车辆动力学，可能导致运动不平滑或舒适度不足；③对极大规模多车系统的可扩展性尚未深入研究；④中心化评论器中 CVAE+GAT 的计算量相对较高，虽然在稠密场景下可控，但在更大规模环境下可能成为瓶颈。

---

## 108. A Practical Guide Towards Interpreting Time-Series Deep Clinical Predictive Models: A Reproducibility Study

**arXiv ID:** 2603.24828 | [PDF](https://arxiv.org/pdf/2603.24828v1)

**作者:** Yongda Fan `[一作]` (University of Illinois Urbana-Champaign), Adam Cross `[通讯]` (University of Illinois)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e15e3743-5ee0-4d5f-813d-d146868082fc` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了一个跨任务、跨模型的可解释性评估基准，对时间序列临床预测模型进行系统性可解释性方法比较；

**💡 创新点**

创新点在于：①系统评估多种解释方法（梯度、注意力、黑盒等）在多任务、多模型中的表现；②证明注意力在合适使用时可作为高效、可信的解释工具；③公开实现于PyHealth框架，支持可复现与可扩展；

**🔧 技术方法**

使用的技术包括：梯度基解释（Integrated Gradients、DeepLIFT）、改进的梯度流GIM、注意力权重融合（Chefer）、黑盒解释（LIME、Kernel SHAP）以及三种模型（StageNet、Transformer、StageAttn）;

**📊 数据集**

使用MIMIC‑IV临床数据集，涵盖三项任务：糖尿病酮症酸中毒预测、死亡率预测、住院时长预测；

**📈 对比分析**

比较方式是用足够性与全面性两项信度指标计算“胜率”，结果显示：Integrated Gradients与Chefer整体最为可靠，Chefer在多任务多模型中胜率最高；LIME与SHAP在计算成本与解释质量上表现最差；

**⚠️ 局限性**

局限性：①评估范围仅限时间序列数据，未覆盖多模态；②黑盒方法在规模上不可行，且对大规模模型效果不佳；③某些方法（DeepLIFT、GIM）在注意力模型上表现不稳定，需进一步改进；

---

## 109. Resisting Humanization: Ethical Front-End Design Choices in AI for Sensitive Contexts

**arXiv ID:** 2603.24853 | [PDF](https://arxiv.org/pdf/2603.24853v1)

**作者:** Silvia Rossi `[一作]` (Immanence), Mackenzie Jorgensen `[通讯]` (Northumbria University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

探讨并案例分析AI前端设计中的伦理问题，尤其是在性别暴力幸存者场景下的创伤知情设计

**💡 创新点**

提出前端设计为程序化伦理的概念，强调在高风险情境中通过减少人性化元素来提升用户自主与安全

**🔧 技术方法**

基于大型语言模型（LLM）构建的对话与信息检索工具，采用透明、明确身份与用户导向的交互方式

**📊 数据集**

未使用公开数据集，案例基于Chayn组织的实际产品与用户反馈

**📈 对比分析**

无定量比较方法或性能指标，本文为概念性与案例研究性质

**⚠️ 局限性**

局限于单一组织与场景，缺乏跨域实验与量化验证，结果难以推广到更广泛的AI系统

---

## 110. Grokking as a Falsifiable Finite-Size Transition

**arXiv ID:** 2603.24746 | [PDF](https://arxiv.org/pdf/2603.24746v1)

**作者:** Yuda Bi `[一作]` (Tri-Institutional Center for Translational Research in Neuroimaging and Data Science), Vince D Calhoun `[通讯]` (Georgia Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文通过将模组加法任务的群阶p作为可扩展尺寸变量，并使用Spectral Head‑Tail Contrast（HTC）作为表示层次的秩序参数，对Transformer模型的学习过程进行有限尺寸缩放（FSS）分析，验证了grokking现象在此框架下表现为类似相变的转移行为。

**💡 创新点**

创新点在于首次将凝聚态物理的Binder交叉、易感度峰值与AIC模型比较等诊断工具应用于机器学习中的grokking问题，并通过明确的尺寸控制（p）和秩序参数（HTC）实现了对转移现象的可验证、可排斥的量化分析。

**🔧 技术方法**

技术方法包括：基于Transformer（d_model=128）与AdamW优化器的训练；构造held‑out probe集提取隐藏层表示并计算其协方差谱；定义HTC并计算Binder‑like cumulant与易感度；使用Bootstrap回归评估交叉漂移、AIC比较平滑交叉与幂律模型；在近临界尺度上进行高分辨率诊断。

**📊 数据集**

使用的数据集为模组加法任务，即对所有p²个（x, y）∈ℤ_p×ℤ_p的有序对进行全量采样，随后按比例划分为训练集、probe集和评估集，其中p取13个质数（粗格）与6个更大质数（近临界）范围内。

**📈 对比分析**

比较方法：通过Binder交叉一致性（交叉点≈0.39），易感度峰值拟合比较（ΔAIC=11.4（粗格）→16.8（近临界）），拒绝平滑交叉模型；Binder最小值随p趋近零或负值的趋势表明可能为连续或弱一阶；未观察到seed分布的双峰共存。整体性能表现为强烈支持转移性而非平滑过渡，但转移阶数仍不确定。

**⚠️ 局限性**

局限性包括：仅在固定Transformer架构与p范围内验证，未探究更大尺寸或不同网络结构；未得到临界指数、指数指数或普适性类；未通过seed分布的双峰共存进一步确认一阶转移；仅关注加法任务，乘法、减法等运算需进一步验证。

---

## 111. Gaze patterns predict preference and confidence in pairwise AI image evaluation

**arXiv ID:** 2603.24849 | [PDF](https://arxiv.org/pdf/2603.24849v1)

**作者:** Nikolas Papadopoulos `[一作]` (Columbia University), Paul Sajda `[通讯]` (Columbia University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a4b10f5d-130b-4e77-9367-6469ec621899` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

对 AI 生成图像的二选一偏好评估过程进行眼动跟踪，研究眼动对偏好形成和信心水平的预测能力。

**💡 创新点**

首次证明眼动级联效应在图像偏好评估中成立，并发现眼动转移率可预测决策信心，为眼动在 RLHF/DPO 评价中的应用提供新视角。

**🔧 技术方法**

采用 Tobii Pro Fusion 眼动追踪、PsychoPy 试验平台、I‑DT 眨眼提取、Logistic Regression 等机器学习分类方法。

**📊 数据集**

使用 150 对 AI 生成图像（Open‑Image‑Preferences、Rapidata、Google Gemini 等）共 1,800 次试验。

**📈 对比分析**

与仅用响应时或自报置信度的方法相比，眼动特征在偏好预测上达到 68% 正确率，置信度预测上达到 66% 正确率，明显优于仅用响应时的 58%。

**⚠️ 局限性**

偏好预测准确率仅为 68%，眼动级联仅在决策前约 1 秒显现；缺乏对视觉细节层面的分析，且需要专业眼动设备限制了大规模部署。

---

## 112. Flow matching on homogeneous spaces

**arXiv ID:** 2603.24829 | [PDF](https://arxiv.org/pdf/2603.24829v1)

**作者:** Francesco Ruscelli `[一作]` `[通讯]` (University of Heidelberg), Francesco Ruscelli (University of Heidelberg)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `40105733-5154-44cd-8090-a8cab9e64b07`

**🎯 论文内容**

提出了一种将流匹配从李群降到李代数的框架，并通过提升-投影方法将其推广到齐次空间（G/H），从而避免了在齐次空间上复杂的几何计算。

**💡 创新点**

创新点在于将流匹配改写为李代数上的欧氏流匹配，利用指数映射把问题从李群降到其线性化空间，并在齐次空间上实现流匹配时避免预度量和测地线的计算，提供了更简单、更快、更本质的实现。

**🔧 技术方法**

使用了条件流匹配（Conditional Flow Matching）损失、指数曲线插值、左不变度量、神经网络逼近向量场，以及中点积分法进行推断。

**📊 数据集**

实验使用了在 Siegel 上半平面 ℍ 和球面 S² 上生成的棋盘分布作为目标分布，分别映射到 SL(2,ℝ) 和 SO(3,ℝ) 的李群进行训练。

**📈 对比分析**

与传统的“Lie 无关”欧氏流匹配模型对比，提出的李群流匹配模型和参数编码模型在生成精度、收敛速度和泛化性能上均明显优于对照模型，表现出更好的效果。

**⚠️ 局限性**

局限性包括：仅适用于指数映射为满射或数据已知落在指数映射像内的李群；实验仅在低维示例上验证，缺乏对高维数据、稀疏采样以及对齐群不变性实现的探讨。

---

## 113. Governance in Practice: How Open Source Projects Define and Document Roles

**arXiv ID:** 2603.24879 | [PDF](https://arxiv.org/pdf/2603.24879v1)

**作者:** Pedro Oliveira `[一作]` (Northern Arizona University), Igor Steinmacher `[通讯]` (Northern Arizona University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对 54 个 GitHub OSS 项目中的治理文件进行系统性分析，提取并结构化角色定义，揭示角色分工与权限分布。

**💡 创新点**

首次将 Institutional Grammar 与角色技能映射相结合，量化角色漂移（role drift）和“维护者悖论”，并通过手工解释式聚类得到跨项目可比的角色类别。

**🔧 技术方法**

采用 Institutional Grammar 进行文本解析，构建四维角色属性（范围、特权、义务、晋升/降职规则），结合技能词典做映射，并使用无监督聚类（K‑Means、Agglomerative、DBSCAN）以及手工解释式聚类。

**📊 数据集**

使用从 GitHub 上按许可证分层、按星标数排序筛选得到的 54 个治理文件（涵盖 MIT、GPL、Apache 等主流许可证）作为数据集。

**📈 对比分析**

通过角色属性与技能向量对比，先尝试无监督聚类但因角色复合导致结果不稳定，最终采用手工聚类得到可解释的角色集合；对比结果表明同一角色名在不同项目中职责差异大，体现了角色漂移和维护者职责集中化。

**⚠️ 局限性**

局限在于只检索文件名含“governance”的文本，忽略隐性治理和非仓库的规则；采样偏向高星标项目，难以代表全部 OSS；手工聚类主观性高，且未检验治理文本与实际实践的一致性。

---

## 114. SlopCodeBench: Benchmarking How Coding Agents Degrade Over Long-Horizon Iterative Tasks

**arXiv ID:** 2603.24755 | [PDF](https://arxiv.org/pdf/2603.24755v1)

**作者:** Gabriel Orlanski `[一作]` (University of Wisconsin--Madison), Aws Albarghouthi `[通讯]` (University of Wisconsin--Madison)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了SlopCodeBench，面向迭代软件开发的跨语言基准，用于评估代理在不断演化的规范下持续扩展自身代码的能力。

**💡 创新点**

创新点在于：①只给出外部行为规范，内部架构完全开放；②通过持续迭代检查点跟踪代码质量，定义了“冗长度”和“结构侵蚀”两项指标；③基准数据量大，包含20个问题、93个检查点。

**🔧 技术方法**

采用了大规模语言模型（Claude、Codex、OpenAI、Anthropic等）的原生 CLI harness；通过 Docker 隔离环境；使用静态代码分析工具（radon、AST-Grep）计算复杂度和冗余。

**📊 数据集**

数据集为SlopCodeBench自建的 20 个语言无关问题集合，覆盖 93 个检查点；同时对比 48 个开源 Python 仓库和 20 个长期维护仓库的历史版本。

**📈 对比分析**

与单射单击基准对比，所有 11 种模型在任何问题的完整轨迹上都未达 100% 正确率；最高检查点通过率仅 17.2%。在质量指标上，代理代码的冗长度比人类代码高 2.2 倍，侵蚀率随迭代持续上升。

**⚠️ 局限性**

局限性包括：仅在 Python 轨道上评估；缺乏对训练过程和工具干预的实验；指标仅关注两种质量度量，未涵盖其他软件质量维度；基准的规范与真实工业场景可能仍有差距。

---

## 115. NeuroVLM-Bench: Evaluation of Vision-Enabled Large Language Models for Clinical Reasoning in Neurological Disorders

**arXiv ID:** 2603.24846 | [PDF](https://arxiv.org/pdf/2603.24846v1)

**作者:** Katarina Trojachanec Dineva `[一作]` (Ss. Cyril and Methodius University), Kostadin Mishev `[通讯]` (Ss. Cyril and Methodius University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了面向神经影像诊断的多模态大型语言模型评测基准 NeuroVLM‑Bench，并对 20 种前沿视觉‑语言模型在 2D MRI/CT 病例上的诊断、子类预测、影像属性识别、结构化输出等多目标任务进行系统性评估。

**💡 创新点**

创新点在于：① 统一的结构化输出方案（JSON schema）与统一提示模板（UNIPP），逼近真实放射科报告流程；② 逐层筛选的四阶段评测流程（实验校准、筛选、稳定性验证、泛化测试），控制样本偏倚与选择性；③ 兼顾判别性能、校准、输出可靠性、运算成本等四维指标，实现多维度、可解释的性能对比。

**🔧 技术方法**

主要技术包括：多模态大语言模型（Gemini、GPT‑5、GPT‑4o、GPT‑4.1、Gemma、LLaMA‑4 Maverick、MedGemma、Qwen‑2.5‑VL 等），统一的提示策略、结构化 JSON 输出、基于温度 0.0 + top‑p 1.0 的确定性推理，使用 95% 置信区间的自助抽样评估，计算 ECE、Brier、AUC、macro‑F1、valid‑JSON 率、延迟与成本估算。

**📊 数据集**

使用公开 2D MRI/CT 数据集：脑肿瘤、髓鞘裂解、多发性硬化、缺血/出血性卒中、肿瘤模仿体（脓肿、囊肿、脑炎等）以及正常对照，共计约 29,538 张图像，涵盖多种序列（T1、T2、FLAIR、T1C+）与平面（轴向、矢状面）等属性。

**📈 对比分析**

通过四阶段评测，将 20 个模型从 30% 样本进行筛选，随后在 45% 开发集验证稳定性，最终在 25% 盲测集进行零/少量样本提示对比。结果显示：技术属性（模态、平面、序列）几乎完美识别；诊断子类预测仍显困难；Gemini 2.5 Pro 在 macro‑F1、weighted‑macro‑F1 及准确率上最优；Gemini 2.5 Flash 兼顾性能与成本；GPT‑5 Chat 具备较高微观指标；MedGemma 1.5 4B 在无样本提示下可逼近部分专有模型表现，且保持完美结构化输出。少量样本提示提升诊断平衡性，但显著增加 token 量、延迟与费用。

**⚠️ 局限性**

局限性：① 对 3D 成像与多序列联合推理未覆盖；② 对罕见/安全关键病例的鲁棒性不足；③ 模型在诊断子类上仍易错或产生虚假推断；④ 校准不一致导致置信度解释不可靠；⑤ 计算成本与延迟在大规模部署时可能不可接受；⑥ 评测仅限于公开数据，未覆盖真实临床多中心多模态图像的分布漂移。

---

## 116. Beyond Attention Magnitude: Leveraging Inter-layer Rank Consistency for Efficient Vision-Language-Action Models

**arXiv ID:** 2603.24941 | [PDF](https://arxiv.org/pdf/2603.24941v1)

**作者:** Peiju Liu `[一作]` (Fudan University), Xuanjing Huang `[通讯]` (Fudan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并验证了TIES框架，动态剪枝视觉令牌以减少VLA模型推理延迟并提升机器人操控性能。

**💡 创新点**

创新点在于使用层间Kendall τ一致性指标判别高关注度令牌的可靠性，结合软/硬阈值策略实现任务依赖的动态令牌选择。

**🔧 技术方法**

采用训练无关的Kendall τ计算、软硬TIES剪枝策略、相似度触发的τ更新机制以及基于统一采样和Top-k混合的令牌选取方法。

**📊 数据集**

实验基准为SIMPLER（Visual Matching 与 Variant Aggregation）以及LIBERO，使用CogACT、OpenVLA等VLA模型进行验证。

**📈 对比分析**

与CogACT、FastV、VLA-Cache、EfficientVLA等基线比较，TIES在仅保留56个令牌时平均成功率提升5–7%，在Variant Aggregation下进一步缩小性能差距。

**⚠️ 局限性**

缺点在于对Kendall τ与令牌重要性之间的因果关系缺乏理论依据，且主要聚焦空间冗余的过滤，跨模态对齐与更复杂语言指令的处理仍有待改进。

---

## 117. Learning From Developers: Towards Reliable Patch Validation at Scale for Linux

**arXiv ID:** 2603.24825 | [PDF](https://arxiv.org/pdf/2603.24825v1)

**作者:** Chih-En Lin `[一作]` (Purdue University), Pedro Fonseca `[通讯]` (Purdue University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文首先对 Linux 内存管理子系统过去十年的补丁评审过程进行了量化研究，随后提出并实现了 FLINT——一种基于规则、可解释且不需要再训练 LLM 的补丁验证框架。

**💡 创新点**

创新点包括：①通过多阶段自动化流程从历史邮件讨论中提取、过滤、聚合规则，并与源代码信息结合；②使用不需微调的 LLM 进行规则检索与问题生成；③提出 Ground‑Truth Coverage Score（GCS）与 H‑GCS 两种新的评估指标，专门用于衡量补丁验证系统在无限可能性空间中的表现。

**🔧 技术方法**

核心技术：规则提取与聚合（基于 LLM 的文本摘要与聚合）；多阶段验证流程（源代码符号识别、规则检索、问题生成、批量过滤）；链式思维（Chain‑of‑Thought）LLM 验证器；以及基于多数投票和 Best‑of‑N 的结果稳定化。

**📊 数据集**

数据集：(1) 2015‑2025 年 Linux memory‑management mailing list 共 386,535 条邮件，随机抽样 177 讨论线程、3,934 条回复；(5) 21 对 Bug‑Fix 补丁对作为验证集；(6) 公开/闭源 LLM 模型（Gemini‑2.5‑Pro/Flash、DeepSeek‑R1:32b、Qwen3‑coder:30b、Gemma3:4b/12b）。

**📈 对比分析**

与仅使用 LLM 的基线相比，FLINT 在 GCS 上提升了 21%（最佳模型），H‑GCS 也更高；在 10 次 Best‑of‑10 评估中，FLINT 的误报率为 35%（比基线低 20%），误报率更低；在不同 bug 类型（锁、数据竞争、性能回归等）上也表现出更均衡的覆盖率。

**⚠️ 局限性**

局限性：(1) 规则集来源仅限历史邮件讨论，可能缺失新出现的 bug 模式；(2) 仍需人工验证最终报告，无法完全替代人力；(3) 对并发复杂度的捕获依赖规则的完整性，若规则丢失细节可能导致漏报；(4) 评估聚焦内存管理子系统，对其他子系统的泛化能力未充分验证。

---

## 118. AutoSAM: an Agentic Framework for Automating Input File Generation for the SAM Code with Multi-Modal Retrieval-Augmented Generation

**arXiv ID:** 2603.24736 | [PDF](https://arxiv.org/pdf/2603.24736v1)

**作者:** Zaid Abulawi `[一作]` (Texas A&M University), Yang Liu `[通讯]` (Texas A&M University)

**通讯引用:** 34140 | [OpenAlex ID](https://openalex.org/A5100356037)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `14d48e9d-0069-4ad9-996a-1d5968216998` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 AutoSAM 框架，利用大语言模型结合检索增强生成和多模态工具，实现从工程文档（PDF、图像、电子表格等）自动生成 SAM 系统热‑水动力学代码的输入文件。

**💡 创新点**

创新点在于：① 将 LLM 作为 ReAct 代理与 SAM 用户手册、理论手册嵌入式检索相结合；② 开发专用的 PDF/图像解析工具，支持文字、表格、公式与图形的联合检索；③ 采用中间结构化 YAML 作为人机审计点，保证模型可追溯性与安全性。

**🔧 技术方法**

技术主要包括：大语言模型（如 GPT‑4/Claude）、检索增强生成（RAG）、视觉‑语言模型（Nougat、CLIP 等）用于图像解析、LangChain 框架进行工具链编排，以及自定义工具（PDF/Excel 读取、输入文件生成/校验、Python 代码执行）。

**📊 数据集**

使用的数据集来自四个真实的核反应堆案例：单管稳态模型、固体燃料温度反馈模型、先进燃料测试反应堆（ABTR）核心模型、熔盐反应堆实验（MSRE）主回路模型，其中包含结构化 Excel、PDF 文档、工程示意图等多模态输入。

**📈 对比分析**

通过对比人工手工制作的输入文件与 AutoSAM 生成的文件，验证结果在四个案例中均符合预期的热水动力学行为；在数值指标上，文档提取率达 100%（结构化）/ 88%（PDF 文本）/ 100%（图像几何），表明模型在信息获取与输入合成方面表现优异；同时模型能够主动标注缺失或假设值，提升了工程可审计性。

**⚠️ 局限性**

局限性包括：① 目前仅针对 SAM 代码，迁移到其他系统代码需重写规则和工具；② 大模型的上下文窗口有限，面对数百组件的大规模系统仍可能遗漏信息；③ 仍存在模型“幻觉”风险，需人工复核；④ 需要在实际监管流程中进一步验证可接受性与安全性。

---

## 119. Belief-Driven Multi-Agent Collaboration via Approximate Perfect Bayesian Equilibrium for Social Simulation

**arXiv ID:** 2603.24973 | [PDF](https://arxiv.org/pdf/2603.24973v1)

**作者:** Weiwei Fang `[一作]` (Wuhan University of Technology), Jianwei Zhang `[通讯]` (Iwate University)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出BEACOF框架，实现LLM多智能体在社交仿真中的自适应协作与竞争，动态切换合作模式以避免集体思维和僵局。

**💡 创新点**

核心创新在于将协作视为不完全信息的动态博弈，引入近似完美贝叶斯均衡（PBE）来统一信念更新与策略选择，解决协作类型选择与能力估计的循环依赖。

**🔧 技术方法**

利用游戏理论（PBE）、高斯贝叶斯更新、LLM提示工程与Meta‑Agent协调，构建可迭代的信念-策略闭环。

**📊 数据集**

在三类数据集上验证：法庭辩论（AgentsCourt + 中国裁判文书），PersonaChat，MedQA医学问答。

**📈 对比分析**

与CAMEL（合作）、MAD（竞争）和ReConcile（共识）等静态基线对比，BEACOF在法庭辩论的法条识别F1提升约3.4分、在PersonaChat中多样性提升10+点并将矛盾率降低约12.7点、在MedQA中准确率提升24+点；整体表现优于所有基线。

**⚠️ 局限性**

局限在于近似PBE仍依赖LLM推理，可能对模型规模敏感，且在极端噪声或大规模智能体网络下的收敛性尚未完全验证。

---

## 120. Examining the Effect of Explanations of AI Privacy Redaction in AI-mediated Interactions

**arXiv ID:** 2603.24735 | [PDF](https://arxiv.org/pdf/2603.24735v1)

**作者:** Roshni Kaushik `[一作]` (Fujitsu Research of America), Koichi Onoue `[通讯]` (Fujitsu Research of America)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在AI辅助的隐私保护交流中，研究了在系统对敏感信息进行删减时提供解释对用户信任与参与度的影响，并通过实验检验了不同解释深度与删减程度的交互作用。

**💡 创新点**

首次将解释生成与隐私删减结合，提出基于上下文与用户差异的自适应解释机制，探讨解释对信任的调节作用。

**🔧 技术方法**

使用GPT‑4.1进行敏感信息检测、删减及解释生成；通过人机交互实验收集用户信任与满意度问卷。

**📊 数据集**

实验数据来源于人工智能生成的五个研究主题与25个问题（分高、中、低删减级别），共180名受试者在Prolific平台完成。

**📈 对比分析**

通过在解释条件（无/一般/详细）与删减程度之间的混合设计，采用Welch ANOVA、Games‑Howell事后检验；结果表明提供解释可提升对隐私保护的信任（Cohen’s d≈0.3），且在高删减场景中解释效果更明显。

**⚠️ 局限性**

局限性包括：实验场景为受控在线模拟，缺乏真实任务与长期使用；只考虑PII删减，未覆盖更复杂隐私保护需求；解释的自适应策略尚未实现，需要进一步验证。

---

## 121. Evaluating Fine-Tuned LLM Model For Medical Transcription With Small Low-Resource Languages Validated Dataset

**arXiv ID:** 2603.24772 | [PDF](https://arxiv.org/pdf/2603.24772v1)

**作者:** Mohammed Nowshad Ruhani Chowdhury `[一作]` (Metropolia University of Applied Sciences), Sakari Lukkarinen `[通讯]` (Metropolia University of Applied Sciences)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对芬兰低资源语言医疗转录进行领域对齐的LLM微调，使用LLaMA 3.1‑8B在学生模拟临床对话数据上微调并评估其转录质量。

**💡 创新点**

证明在极小验证数据集上对开源大模型进行细粒度微调即可显著提升芬兰语医疗转录性能，为低资源语言医疗NLP提供可行路径。

**🔧 技术方法**

采用LLaMA 3.1‑8B、LoRA量化、Whisper 语音转文本、Hugging Face平台、7折交叉验证与BLEU/ROUGE/BERTScore评估等技术。

**📊 数据集**

仅包含7个芬兰语模拟临床对话的MP3与人工转录文本，由Metropolia大学学生生成的验证集。

**📈 对比分析**

通过7折交叉验证评估BLEU平均0.1214、ROUGE‑L 0.4982、BERTScore 0.8230，表明语义相似度高但句法匹配有限。

**⚠️ 局限性**

数据量极小，模型未原生芬兰语预训练，难以捕捉全部形态学细节，且可能存在信息缺失与hallucination。

---

## 122. Algebraic Expander Codes

**arXiv ID:** 2603.24788 | [PDF](https://arxiv.org/pdf/2603.24788v1)

**作者:** Swastik Kopparty `[一作]` (University of Toronto), Itzhak Tamo `[通讯]` (Tel Aviv University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文提出了一种新的 Algebraic Expander Code 构造，利用非交换的平移子群和标度子群在 AGL(1,𝔽) 上的作用，得到本地为 Reed–Solomon 码的稀疏图码。

**💡 创新点**

创新点在于突破传统 Tanner 码的 2r−1 速率瓶颈，证明在低本地速率 r≤1/2 仍能获得正的全局速率，并通过 u‑基度和多项式体积计数提供新的维度下界。

**🔧 技术方法**

采用了代数几何方法（子空间多项式、u‑基展开）、谱图论（通过字符和高斯和估计证明二阶奇异值 O(1/√p)）、以及多面体体积/维数计数技巧。

**📊 数据集**

未使用外部数据集，所有实验和证明均在理论字段上完成，构造在给定素数 p 与参数 m 的有限域上实现。

**📈 对比分析**

相较于传统的稀疏图扩展码，本文的码在本地率低于 1/2 时仍保持线性距离与正全局速率；实验显示相同块长下，速率约为 r^{2m+1}/(2m+1)!，距离≈(1−r)^2，优于仅依赖计数的 2r−1 下限。

**⚠️ 局限性**

主要局限包括字母表大小随块长呈多项式增长、图度随 m 下降为多项式而非常数、维度下界不够紧凑、对高维扩展的实现尚未探讨。

---

## 123. TaCo: Data-adaptive and Query-aware Subspace Collision for High-dimensional Approximate Nearest Neighbor Search

**arXiv ID:** 2603.24919 | [PDF](https://arxiv.org/pdf/2603.24919v1)

**作者:** Jiuqi Wei `[一作]` (Oceanbase, Ant Group), Themis Palpanas `[通讯]` (Universite Paris Cite)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出一种名为 TaCo 的近似最近邻搜索框架，结合数据自适应子空间变换与查询自适应候选选择，显著提升索引构建和查询效率。

**💡 创新点**

核心创新包括：①基于熵均衡的子空间自适应变换，平衡子空间信息量；②使用堆结构的可扩展动态激活算法，将子空间碰撞计数复杂度降至 O(1)；③查询自适应候选选择，根据 SC‑score 分布动态决定候选集大小，实现查询负载动态调节。

**🔧 技术方法**

采用信息论熵、特征分解与特征分配（Eigensystem Allocation）、K‑means 聚类、倒排多重索引（IMI）、最小堆、以及欧氏距离重排序。

**📊 数据集**

在 DEEP1M、GIST1M、SIFT10M、Yandex DEEP10M、Microsoft SPACEV10M 这五个公开高维数据集上进行实验。

**📈 对比分析**

与 SuCo、SC‑Linear、IMI‑OPQ、DET‑LSH、IVF‑RaBitQ、HNSW、MIRAGE、SHG 等方法对比，TaCo 在索引时间上最快（比 SuCo 快 8×，内存占用 0.6×），查询吞吐率提升 1.5×，在多种 recall 设定下均达或超过现有最优水平。

**⚠️ 局限性**

局限性：仍仅适用于欧氏空间；对极高维度或非均匀分布的数据可能需要进一步改进子空间划分策略；目前未结合深度学习或学习型索引，且实验集中在单机内存环境。

---

## 124. SolRugDetector: Investigating Rug Pulls on Solana

**arXiv ID:** 2603.24625 | [PDF](https://arxiv.org/pdf/2603.24625v1)

**作者:** Jiaxin Chen `[一作]` (Sun Yat-sen University), Zibin Zheng `[通讯]` (Sun Yat-sen University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文系统地收集并手工标注了117个Solana Rug Pull代币，并构建了第一个公开的Solana Rug Pull数据集，进一步提出并验证了一种基于链上行为的检测系统SolRugDetector。

**💡 创新点**

创新点在于将Rug Pull的检测从以太坊的合约逻辑迁移到Solana的账户与交易行为，提出了冻结权限滥用、流动性撤回和泵跌三类完整的恶意工作流程，并基于此设计了规则驱动的检测框架。

**🔧 技术方法**

利用Solana RPC、Solscan API和原始交易日志，采用规则匹配、时间窗口交易率、持有人数变化等行为特征，并通过阈值优化实现精准检测。

**📊 数据集**

使用了68份社区报告手工标注的117个恶意代币数据，以及对2025年上半年100,063个新发行代币的链上实时扫描结果，最终得到76,469个Rug Pull代币的实验集。

**📈 对比分析**

将SolRugDetector与Solana Rug Checker、RugChecker、Solsniffer、Solanatracker四个主流工具在同一标注集上比较，发现其F1得分最高（0.96），准确率和召回率均超过现有工具，误报率低至0.26%。

**⚠️ 局限性**

主要局限在于只能事后检测，缺乏实时预警，且对极低活跃度或长期潜伏的恶意代币识别能力有限。

---

## 125. MoE-GRPO: Optimizing Mixture-of-Experts via Reinforcement Learning in Vision-Language Models

**arXiv ID:** 2603.24984 | [PDF](https://arxiv.org/pdf/2603.24984v1)

**作者:** Dohwan Ko `[一作]` (Korea University), Hyunwoo J. Kim `[通讯]` (KAIST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在视觉语言模型中用强化学习优化稀疏专家激活策略，提出MoE‑GRPO框架。

**💡 创新点**

将专家选择建模为序列决策并用GRPO训练路由策略，同时引入模态感知路由指导以提升探索效率。

**🔧 技术方法**

使用Mixture‑of‑Experts架构、Group Relative Policy Optimization、门控网络、负载平衡损失和RL奖励。

**📊 数据集**

InternVL3.5‑1B、CLIP‑MoE、OneThinker 多选视觉指令集以及图像/视频基准（如ImageNet、VQA等）。

**📈 对比分析**

与传统 deterministic top‑K、随机采样与加噪声路由对比，在多图像与视频任务上均提升约1‑2%准确率/效果，并显著提高专家多样性。

**⚠️ 局限性**

RL探索仍可能导致训练不稳定、收敛慢，且在极大规模或跨模态场景下的泛化仍需进一步验证。

---

## 126. AutoCSF: Provably Space-Efficient Indexing of Skewed Key-Value Workloads via Filter-Augmented Compressed Static Functions

**arXiv ID:** 2603.24882 | [PDF](https://arxiv.org/pdf/2603.24882v1)

**作者:** David Torres Ramos `[一作]` (Distill), Benjamin Coleman `[通讯]` (Google DeepMind)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 AutoCSF 算法，自动决定何时以及如何将压缩静态函数（CSF）与预过滤器结合，以在高度偏斜的键值数据集上实现可证明的空间节省。

**💡 创新点**

创新点在于：①基于信息论给出 CSF 与过滤器组合的严格上下界；②提出通用框架，适用于任意近似集合成员结构；③利用下界直接推导最优误报率，提供可验证的参数选择。

**🔧 技术方法**

技术主要包括：压缩静态函数、哈夫曼编码、信息论界（Kraft‑McMillan 与 Shannon 不等式）、Bloom/XOR/二元熔合过滤器以及 AutoCSF 的实现与参数搜索。

**📊 数据集**

使用合成数据集（Unique、Zipfian、Uniform‑100）以及真实基因组 k‑mer 计数数据（E. coli、SRR10211353、C. elegans）进行评估。

**📈 对比分析**

与 BCSF、C++ 哈希表、MPH 表和学习型 CSF 进行对比；AutoCSF 在合成与基因组数据上均位于记忆-延迟 Pareto 前沿，内存仅几百毫比/键，查询延迟低，构建时间 <0.1 s；学习型 CSF 仅在压缩率上略优，但延迟与构建时间高达 2‑4 个数量级。

**⚠️ 局限性**

局限性包括：在极低偏斜（α ≲ 0.7）时不使用过滤器，Unique 等高离散度分布下内存仍高于学习型 CSF；对高度可学习或自相关的分布无法充分利用；理论界依赖于预过滤器的成本‑误报率模型及 δ 常数，实际实现中需精细调校。

---

## 127. Synthetic Cardiac MRI Image Generation using Deep Generative Models

**arXiv ID:** 2603.24764 | [PDF](https://arxiv.org/pdf/2603.24764v1)

**作者:** Ishan Kumarasinghe `[一作]` (University of Peradeniya), Vajira Thambawita `[通讯]` (SimulaMet)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `9cc9baba-5356-466d-81ff-d80028d90279` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `f86bf285-fd08-4156-973b-6e6481af8fa0` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述了合成心脏MRI（CMRI）生成技术，系统评估其在图像真实性（fidelity）、下游任务实用性（utility）和隐私保护（privacy）三维度的表现，并对不同生成框架（GAN、VAE、扩散模型、流匹配等）以及条件化策略（mask-conditioning、SPADE、ControlNet等）进行了对比。

**💡 创新点**

通过从生成模型、条件化技术、评估指标和隐私攻击等多维度进行整合，提出了对CMRI生成研究的共识与不足：1）大多数工作缺乏系统的隐私评估；2）跨数据集、跨病理的泛化实验不足；3）病理特异性条件生成研究稀缺；4）扩散模型在大规模生成时效率低下；5）流匹配在小样本情形下易过拟合。

**🔧 技术方法**

综合使用GAN、VAE、DDPM、DDIM、Latent Diffusion Model、Flow-Matching、SPADE、ControlNet、Stable Diffusion、MIP、MIA、FCRE等技术，结合分割引导、文本引导、姿态估计、差分隐私等方法对合成图像进行生成与评估。

**📊 数据集**

引用的主流公开数据集包括M&Ms、ACDC、UK Biobank、OCMR、M&Ms‑2、MS‑CMRSeg、MM‑WHS、Sunnybrook 等，覆盖多中心、多供应商、多疾病和多序列（LGE、T1/T2、Cine）数据。

**📈 对比分析**

通过对生成图像的FID、SSIM、MS‑SSIM、PSNR、Dice、HD等指标进行量化评估，并结合下游分割任务的性能提升（如Mask‑conditioned扩散模型可提升约4% Dice）、跨域泛化测试（如ACDC→M&Ms）、以及隐私攻击实验（MIA、nearest‑neighbor、FCRE），发现扩散模型在Fidelity与实用性上优于GAN/VAEs；Mask‑conditioned生成显著提升分割准确率；但在隐私评估中大部分工作未给出量化结果。

**⚠️ 局限性**

主要局限包括：1）隐私评估缺失或仅做定性分析；2）缺乏统一的跨数据集、跨病理的对比实验；3）病理特异性条件生成方法稀缺，难以覆盖所有临床情境；4）扩散模型采样慢、资源消耗大；5）流匹配在小样本下易过拟合；6）多模态、三维等复杂生成任务仍处于探索阶段。

---

## 128. Estimating near-verbatim extraction risk in language models with decoding-constrained beam search

**arXiv ID:** 2603.24917 | [PDF](https://arxiv.org/pdf/2603.24917v1)

**作者:** A. Feder Cooper `[一作]`, Percy Liang `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出解码受限束搜索（k-CBS）来估计大型语言模型（LLM）的近似逐字提取风险；

**💡 创新点**

通过在束搜索中强制使用解码分布，给出确定性下界，可在显著降低采样成本的同时捕获更多非逐字提取实例；

**🔧 技术方法**

利用top‑k束搜索、距离度量（汉明/Levenshtein）、可达性剪枝等技术，计算近似逐字提取概率下界；

**📊 数据集**

在OLMo 2（7B/13B/32B）Wikipedia 10,000 篇训练集和 Llama 2（7B/13B/70B）The Great Gatsby 数据集上进行实验；

**📈 对比分析**

与贪心提取、概率提取（Monte Carlo）对比，k-CBS 在 20 次 MC 样本成本下即可得到约 0.01 的下界，覆盖 0.7–0.8 的近似逐字提取质量，显著高于仅逐字提取的 1–2% 提取率；

**⚠️ 局限性**

下界可能被束搜索裁剪，无法捕获所有可提取序列；对解码策略和距离阈值敏感；若模型未充分记忆，方法可能无效。

---

## 129. Calibri: Enhancing Diffusion Transformers via Parameter-Efficient Calibration

**arXiv ID:** 2603.24800 | [PDF](https://arxiv.org/pdf/2603.24800v1)

**作者:** Danil Tokhchukov `[一作]` (MSU), Konstantin Sobolev `[通讯]` (FusionBrain Lab)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种名为Calibri的参数高效方法，对Diffusion Transformer（DiT）进行微调，主要通过在模型内部引入约100个可学习的标量系数实现；

**💡 创新点**

创新点在于将DiT块的输出通过单一可学习的标量进行缩放，证明单一缩放参数就能显著提升生成质量，并将校准过程视为黑盒奖励优化问题，采用CMA-ES演化策略高效搜索；

**🔧 技术方法**

核心技术包括DiT架构分析、块/层/门级缩放方案、基于奖励模型的黑盒优化（CMA-ES）、以及校准模型集成（Calibri Ensemble）和与Classifier-Free Guidance的融合；

**📊 数据集**

实验使用Flux、Stable Diffusion 3.5 Medium、Qwen-Image等公开文本到图像模型，评估数据集为HPSv3、Q-Align、ImageReward以及200人评价的HPDv3测试集；

**📈 对比分析**

与原始模型相比，Calibri在HPSv3、ImageReward和Q-Align等指标均有提升（例如Flux从11.41提升至13.48），并将推理步骤从30/40/50降至15/10/10，速度提升约2–3.3倍；

**⚠️ 局限性**

局限性包括：校准需要较高的离线GPU计算成本（32–356 H100 GPU小时）；仅针对文本到图像的DiT模型验证，跨域推广尚未评估；依赖奖励模型的质量，若奖励模型失真可能导致误导。

---

## 130. DyMRL: Dynamic Multispace Representation Learning for Multimodal Event Forecasting in Knowledge Graph

**arXiv ID:** 2603.24636 | [PDF](https://arxiv.org/pdf/2603.24636v1)

**作者:** Feng Zhao `[一作]` (Huazhong University of Science and Technology), Guandong Xu `[通讯]` (Education University of Hong Kong)

**通讯引用:** 10856 | [OpenAlex ID](https://openalex.org/A5051512158)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种动态多空间表示学习框架 DyMRL，用于在知识图谱中获取并融合多模态时序知识，从而实现精确的事件预测。

**💡 创新点**

创新点包括：① 在欧氏、双曲和复数空间同时进行深层结构信息学习，捕获关联、抽象与逻辑三类语义；② 设计双向融合‑演化注意机制，动态赋权不同模态与不同时刻的特征；③ 通过预训练视觉（VGG）与语言（BERT）模型动态获取时间感知的辅助模态；④ 构造四个包含图像、文本与时间结构的多模态时序知识图谱数据集。

**🔧 技术方法**

技术手段包括：多空间信息消息传递与多层图神经网络；RNN 更新模块捕获结构演化；预训练模型编码视觉/语言特征；双融合‑演化注意层（融合注意 + 演化注意）与 Transformer 结构；曲率自适应双曲距离做解码；对齐欧氏与复数空间以保持语义一致性。

**📊 数据集**

使用了四个公开构造的多模态时序 KG 数据集：GDELT-IMG-TXT、ICE14-IMG-TXT、ICE0515-IMG-TXT、ICE18-IMG-TXT，分别涵盖 15 分钟到每日一次的时间粒度。

**📈 对比分析**

与静态多模态基线（TransAE、MoSE、OTKGE、IMF、DySarl）和动态单模态基线（xERTE、RE-GCN、TiRGN、RETIA、RPC、ReTIN、LogCL、TempValid、CognTKE、ANEL）在 MRR、Hits@1、Hits@10 上进行对比，DyMRL 在所有数据集上均显著优于对手（如在 GDELT-IMG-TXT 上 MRR 提升约 17%），证明了动态多空间融合与演化注意机制的有效性。

**⚠️ 局限性**

局限性：① 对图像与文本的依赖使得在缺乏丰富多模态信息的 KG 上表现不佳；② 随着时间窗口增大（如 ICE0515-IMG-TXT），模型因历史信息过多导致提升有限；③ 需要大量计算资源（多空间 GNN、Transformer 关注层）；④ 仍未深入探索更复杂的模态交互与自监督预训练方法。

---

## 131. LogitScope: A Framework for Analyzing LLM Uncertainty Through Information Metrics

**arXiv ID:** 2603.24929 | [PDF](https://arxiv.org/pdf/2603.24929v1)

**作者:** Farhan Ahmed `[一作]` (IBM Research), Chad DeLuca `[通讯]` (IBM Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了 LogitScope 框架，通过计算 LLM 生成过程中的概率分布信息度量（熵、方差熵、惊奇度等）来量化 token 级不确定性，并提供实时可视化与分析工具。

**💡 创新点**

创新点在于将多种信息度量统一包装成轻量级、模型无关的分析工具，利用惰性求值和零拷贝访问实现高效实时计算，同时无需额外标签或多次前向推理即可揭示模型的信心分布和潜在幻觉。

**🔧 技术方法**

主要技术包括概率分布熵、方差熵、惊奇度等信息学度量，基于 HuggingFace Transformers 实现的 Python 封装，支持 CPU/CUDA/MPS 加速，懒加载与缓存机制。

**📊 数据集**

使用的示例数据为美国《独立宣言》序言（320 个 token），并对 SmolLM2-135M-Instruct 模型进行推理；同时对词序反转版本做对比。

**📈 对比分析**

在对比原始文本与词序反转文本时，计算熵、方差熵、困惑度等指标，发现原文本平均熵 1.70、方差熵 3.46、困惑度 11.75；反转文本熵 6.33、方差熵 8.63、困惑度 1833.76，表明框架能捕捉语言结构对模型不确定性的影响。

**⚠️ 局限性**

局限性：信息度量仅反映概率分布特征，无法直接评估语义正确性；高信心不一定正确，低熵也可能产生错误；未覆盖长距离语义或事实一致性，难以自动检测错误。

---

## 132. Pseudo Label NCF for Sparse OHC Recommendation: Dual Representation Learning and the Separability Accuracy Trade off

**arXiv ID:** 2603.24750 | [PDF](https://arxiv.org/pdf/2603.24750v1)

**作者:** Pronob Kumar Barman `[一作]` (University of Maryland, Baltimore County), Tera L. Reynolds. James Foulds `[通讯]`

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `57a58b01-81b4-4d75-a45c-2e891f272b50` `67630363-6be0-4f51-ab05-7198250671a5` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

在用户缺乏行为历史的在线健康社区中，提出了一种利用注册问卷特征的伪标签学习框架来改进支持小组推荐；

**💡 创新点**

创新点在于构建双重表示的PL-NCF模型，分别为排名任务和伪标签对齐任务学习主嵌入和专门的PL嵌入，并通过特征相似度生成软标签；

**🔧 技术方法**

采用了三种Neural Collaborative Filtering架构（MF、MLP、NeuMF），并在其基础上加入伪标签损失与双重嵌入；

**📊 数据集**

使用了基于问卷的16维用户特征和聚合的支持小组特征构建的165用户、498小组、498交互的合成数据集；

**📈 对比分析**

通过留一法评估HR@5和NDCG@5，在极端稀疏情况下，所有三种PL变体均显著提升排名性能（HR@5从约2.6提升至5.3-5.4），同时PL嵌入的聚类轮廓分数明显高于主嵌入；

**⚠️ 局限性**

局限性包括伪标签仅基于特征相似度，缺乏真实用户偏好验证；数据集规模小且为合成，可能不具备泛化性；并且对照实验仅限于NCF模型，未覆盖更广泛的基线。

---

## 133. IndustriConnect: MCP Adapters and Mock-First Evaluation for AI-Assisted Industrial Operations

**arXiv ID:** 2603.24703 | [PDF](https://arxiv.org/pdf/2603.24703v1)

**作者:** Melwin Xavier `[一作]` (Lulea Technical University), Midhun Xavier `[通讯]` (Independent Researcher)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了基于Model Context Protocol (MCP) 的工业协议适配器，并通过“mock-first”流程对Modbus、MQTT/Sparkplug B和OPC UA进行功能、错误处理、并发和恢复性能的确定性基准测试。

**💡 创新点**

创新点在于：①提出统一的MCP工具发现和响应包装，实现跨协议的可发现AI接口；②引入mock-first验证方法，降低物理硬件依赖；③设计了可复制的多场景（正常、故障注入、压力、恢复）基准框架。

**🔧 技术方法**

采用JSON‑RPC 2.0（MCP）、FastMCP、pymodbus、paho MQTT、OPC UA Python SDK、aedes broker、protobuf、Mock模拟器等技术栈。

**📊 数据集**

使用内部构造的模拟数据集：Modbus 100个寄存器/线圈、MQTT 设备模拟器（Sparkplug B 设备两台）、OPC UA 8只读、6只读写变量，且每秒生成同步动态波形。

**📈 对比分析**

通过多轮（30次/任务）确定性执行，比较了不同协议在正常、故障注入和压力情形下的成功率、延迟分布及错误类分布；结果显示Modbus和MQTT均低于3 ms延迟，OPC UA 变量枚举约100 ms；所有协议均在故障注入下100%结构化错误；恢复实验表明同会话恢复可行。

**⚠️ 局限性**

局限在于仅使用本地mock、仅评估三种协议、未涉及真实硬件、未验证LLM交互效果、缺乏安全审计与RBAC、并未覆盖方法调用和更广泛的压力测试。

---

## 134. Physics-Informed Neural Network Digital Twin for Dynamic Tray-Wise Modeling of Distillation Columns under Transient Operating Conditions

**arXiv ID:** 2603.24644 | [PDF](https://arxiv.org/pdf/2603.24644v1)

**作者:** Debadutta Patra `[一作]` (Veer Surendra Sai University of Technology), Sucheta Panda `[通讯]` (Veer Surendra Sai University of Technology)

**通讯引用:** 92 | [OpenAlex ID](https://openalex.org/A5102044529)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一个基于物理信息神经网络的数字孪生模型，用于预测二元HX/TX蒸馏塔在瞬态运行中的柱内温度、压强和组分分布。

**💡 创新点**

创新点包括：① 将VLE、MESH和平衡约束直接嵌入损失函数；② 采用Sigmoid自适应加权训练策略；③ 在多种基线模型上系统比较，证明了物理一致性与高精度的双重优势。

**🔧 技术方法**

使用技术：Physics‑Informed Neural Network（PINN）、全连接网络、Swish激活、Sigmoid调度损失权重、PyTorch实现。

**📊 数据集**

使用数据集：961条时间戳的Aspen HYSYS仿真数据，覆盖8小时的瞬态运行，包含16个传感器通道与噪声。

**📈 对比分析**

通过与LSTM、MLP、GRU、Transformer、DeepONet五个数据驱动基线对比，PINN在HX组分预测上达到RMSE 0.00143，R² 0.9887，较最佳基线下降44.6%，并且唯一满足VLE与质量守恒的模型。

**⚠️ 局限性**

局限性：仅在合成仿真数据上验证；未考虑工业传感器漂移、缺失、非平稳操作；仅针对二元非反应系统，需进一步扩展到多组分、反应蒸馏及工业数据验证。

---

## 135. ReDiPrune: Relevance-Diversity Pre-Projection Token Pruning for Efficient Multimodal LLMs

**arXiv ID:** 2603.24680 | [PDF](https://arxiv.org/pdf/2603.24680v1)

**作者:** An Yu `[一作]` (University at Albany, SUNY), Ming-Ching Chang `[通讯]` (University at Albany, SUNY)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

在多模态大型语言模型中提出一种训练无关、预投影阶段的视觉令牌剪枝方法 ReDiPrune，按查询相关性和多样性选择保留令牌。

**💡 创新点**

创新点在于在视觉编码器特征空间中结合文本相关性与最大-最小多样性进行贪心选择，避免跨模态投影后丢失细粒度信息，且无需再训练或改造网络。

**🔧 技术方法**

使用轻量级查询向量构造、余弦相关性打分、Dissimilarity矩阵和贪心最大-最小多样性优化，以及投影前的前向选择算法。

**📊 数据集**

在四个视频基准（ActivityNet-QA、NextQA、EgoSchema、Video-ChatGPT）和五个图像基准（GQA、MMBench、MME、POPE、ScienceQA-IMG）上进行实验。

**📈 对比分析**

与原始模型、DivPrune、CDPruner等基线相比，ReDiPrune 在保持或提升准确率的同时将 TFLOPs 下降到 10%~15%，并在多项指标上取得最佳或接近最佳的速度-性能折中。

**⚠️ 局限性**

局限性包括仅在帧级别进行剪枝，未考虑跨帧全局时空优化，且对不同模态骨干的泛化仍待验证。

---

## 136. CROSS: A Mixture-of-Experts Reinforcement Learning Framework for Generalizable Large-Scale Traffic Signal Control

**arXiv ID:** 2603.24930 | [PDF](https://arxiv.org/pdf/2603.24930v1)

**作者:** Xibei Chen `[一作]` (National University of Singapore), Guillaume Sartoretti `[通讯]` (National University of Singapore)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了CROSS框架，用Mixture-of-Experts和Predictive Contrastive Clustering进行场景适配的分布式强化学习，以实现可泛化的智能交通信号控制。

**💡 创新点**

创新点在于引入PCC模块将短期状态转移预测与对比学习结合，得到离散化的交通模式表示；以及基于该模式的Scenario-Adaptive MoE，动态激活专家网络实现场景特定策略。

**🔧 技术方法**

技术手段包括GRU+多头交叉注意力的通用特征提取、对比学习的PCC、Mixture-of-Experts路由、PPO强化学习以及多任务损失融合。

**📊 数据集**

使用了合成数据集（Grid 4×4、Grid 5×5、Arterial 4×4）和真实世界数据集（济南三组、杭州两组）在SUMO仿真环境下进行训练和评估。

**📈 对比分析**

与传统固定时序、Max-Pressure以及RL基线GESA、Unicorn进行对比。CROSS在合成数据集上均优于RL基线，在零射击真实数据集上也能显著低于或接近Max-Pressure，显示出良好的跨场景泛化能力。

**⚠️ 局限性**

局限性包括未显式建模交叉口间的协调，模型复杂度相对较高，对领域迁移仍依赖训练数据；在极端高交通量或异构网络中仍有提升空间。

---

## 137. An Explainable Federated Framework for Zero Trust Micro-Segmentation in IIoT Networks

**arXiv ID:** 2603.24754 | [PDF](https://arxiv.org/pdf/2603.24754v1)

**作者:** Muhammad Liman Gambo `[一作]` (King Fahd University of Petroleum & Minerals), Ahmad Almulhem `[通讯]` (King Fahd University of Petroleum & Minerals)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `3f18e8e3-0266-457c-8567-9039b6d2394d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种基于联邦学习、超图建模和可解释 AI 的 Zero Trust 微分段框架 EFAH‑ZTM，用以提升工业物联网网络的安全与可解释性。

**💡 创新点**

创新点包括：①将联邦深度非对称自编码器与 kNN/流形超图相结合，捕获多设备高阶关系；②构造谱嵌入并用 HDBSCAN 或 MiniBatch‑KMeans 进行动态微分段；③提出结合重构误差与结构异常的运营风险评分；④通过 LIME/SHAP 给微分段策略提供可解释性。

**🔧 技术方法**

使用技术包括：联邦学习（FedAvg）、深度非对称自编码器、kNN 与流形超图构建、谱嵌入、MiniBatch‑KMeans、HDBSCAN 聚类、风险评分模型、LIME/SHAP 可解释 AI。

**📊 数据集**

实验采用公开的 WUSTL‑IIoT‑2021 数据集（约 119 万样本，49 维特征）。

**📈 对比分析**

与中心化训练、无超图以及 Selciya 与 Arifeen 的基线对比，EFAH‑ZTM 在结构质量（Silhouette ≈0.64，DBI ≈0.41）和安全效能（聚类纯度≈0.999，污染率≈0）上均优于对照组；HDBSCAN+流形超图实现最佳效果。

**⚠️ 局限性**

局限性包括：仅在模拟环境中验证，未覆盖不同工业场景与时间序列；缺乏对联邦学习攻击的鲁棒性研究；风险评分聚合可能掩盖稀疏攻击；默认的跨分段拒绝策略可能过于严格。

---

## 138. ReLope: KL-Regularized LoRA Probes for Multimodal LLM Routing

**arXiv ID:** 2603.24787 | [PDF](https://arxiv.org/pdf/2603.24787v1)

**作者:** Yaopei Zeng `[一作]` (Pennsylvania State University), Lu Lin `[通讯]` (Pennsylvania State University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了在多模态大型语言模型中通过改进训练探针实现高效路由，解决文本LLM探针在多模态下表现下降的问题。

**💡 创新点**

提出Attention Probe聚合前层隐藏状态以及KL正则化LoRA探针ReLope，通过学习路由感知表示提升路由准确率。

**🔧 技术方法**

采用Transformer隐藏状态探针、注意力聚合、LoRA低秩适配器、信息瓶颈KL正则化、变分自编码器等技术。

**📊 数据集**

在MMMU、AOKVQA、ScienceQA、ChartQA、MathVision等五大多模态基准上测试，并使用Qwen2.5-VL-7B-Instruct、Gemma3-12B、Phi-4-Multimodal-Instruct等模型。

**📈 对比分析**

与MoT、Cascade Routing、RouteLLM、Post-Hoc Embed、Probe等基线比较，ReLope在AUC上平均提升约3–5个百分点，并在低路由比例下显著提升系统精度。

**⚠️ 局限性**

仅在有限模型/任务上验证，需人工标注正确性标签，未深入解释探针失效原因，且对更广泛多模态场景的泛化能力仍需进一步验证。

---

## 139. PII Shield: A Browser-Level Overlay for User-Controlled Personal Identifiable Information (PII) Management in AI Interactions

**arXiv ID:** 2603.24895 | [PDF](https://arxiv.org/pdf/2603.24895v1)

**作者:** Max Holschneider `[一作]` (MIT Media Lab), Saetbyeol LeeYouk `[通讯]` (MIT Media Lab)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一款浏览器插件，能够在用户与云端大型语言模型交互前，自动对输入的文本和附件进行本地 PII 识别并用占位符替换，同时提供“烟雾屏”功能将语义相近但不透露身份的替代内容注入提示；该插件通过可视化覆盖层让用户仍能看到原始文本并手动调整。

**💡 创新点**

创新点在于：①将企业级的 PII redaction pipeline 移植到前端、面向普通用户；②采用本地 LLM 生成“烟雾屏”伪装文本，兼顾上下文相关性；③实现无侵入式、透明的浏览器 overlay，既保护隐私又保持交互流畅；④将自动化红action 与手动细化结合，提升用户控制感。

**🔧 技术方法**

核心技术包括：浏览器扩展框架；基于本地 LLM 的命名实体识别与文本生成；文本/文件处理与占位符映射；可视化覆盖与弹窗交互；使用开源 NLP 库（如 spaCy、transformers）以及前端渲染技术。

**📊 数据集**

论文未提供正式实验数据集，推测使用常见 NER 语料（如 CoNLL‑2003）和公开的 PII 识别数据进行实体检测，smokescreen 生成采用公开预训练 LLM；具体数据集待未来用户研究补充。

**📈 对比分析**

目前未给出量化对比实验或性能指标；作者计划通过用户研究和参与式设计评估系统的可用性、对隐私泄露的抑制效果以及对 LLM 生成质量的影响；在实现阶段已保证对交互无明显延迟。

**⚠️ 局限性**

局限性包括：①缺乏大规模用户测试与量化评估；②对云端 LLM 性能影响尚未充分验证；③仅覆盖浏览器级交互，无法防御桌面级 AI 代理；④依赖本地 LLM，性能受设备限制；⑤缺乏对跨平台兼容性与持续更新的策略。

---

## 140. AI Security in the Foundation Model Era: A Comprehensive Survey from a Unified Perspective

**arXiv ID:** 2603.24857 | [PDF](https://arxiv.org/pdf/2603.24857v1)

**作者:** Zhenyi Wang `[一作]` (University of Central Florida), Siyu Luan `[通讯]` (University of Copenhagen)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出统一的闭环威胁分类法，并系统综述了四个方向（D→D、D→M、M→D、M→M）的攻击与防御，结合实验验证其在不同任务（视觉、语言、音频、图结构）上的表现。

**💡 创新点**

创新点在于构建四向闭环模型-数据交互框架，揭示攻击之间的相互放大/削弱关系，首次系统化评估多攻击协同效果并提出未来研究方向。

**🔧 技术方法**

使用的技术包括：文献系统综述、攻击与防御的数学建模、对抗训练、模型提取、成员推断、模型逆向、数据去水印、逆向生成等多种攻击与防御手段。

**📊 数据集**

使用的数据集涵盖：视觉（MNIST、CIFAR‑10、PathMNIST、MedMNIST、TinyImageNet）、语言（WIKIMIA）、音频/图结构等，实验中还加入 LVW、CLWD 等专用水印数据集。

**📈 对比分析**

通过统一实验协议对比多种攻击/防御在不同数据集上的指标（PSNR/SSIM、准确率、AUC、提取精度等），展示了攻击效果与防御效果，并指出大多数防御在多攻击环境下效果有限。

**⚠️ 局限性**

局限性包括：缺乏针对多攻击协同的系统评估基准；攻击与防御在动态更新（持续学习、对齐）场景下的适应性不足；实验多为离线对比，缺乏实际部署环境验证。

---

## 141. Towards automatic smoke detector inspection: Recognition of the smoke detectors in industrial facilities and preparation for future drone integration

**arXiv ID:** 2603.24850 | [PDF](https://arxiv.org/pdf/2603.24850v1)

**作者:** Lukas Kratochvila `[一作]` (Brno University of Technology), Karel Horak `[通讯]` (Brno University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了一套用于工业设施中烟雾探测器自动检测的系统，并设计了可集成到无人机平台的ROS‑2推理管线。

**💡 创新点**

创新点包括：① 在数据有限的情况下构建半合成数据集，并通过五种不同的训练策略验证其对模型泛化的提升；② 对三类主流检测器（YOLOv11、SSD、RT‑DETRv2）及其变体进行系统对比，确定最适合嵌入式平台的模型；③ 提供完整的Raspberry Pi 4/5 推理时间评估，为无人机集成提供实测参考。

**🔧 技术方法**

使用技术：YOLOv11（不同尺寸）、SSD（MobileNetV3、VGG16）、RT‑DETRv2；数据增强方法包括Randaugment、Albumentations；半合成图像生成与真实背景拼接；ROS‑2 pipeline 以及在Raspberry Pi上使用TensorRT/ONNX 的推理实现。

**📊 数据集**

数据集：1672张真实工业环境与实验室拍摄的烟雾探测器图像，3840张半合成图像（真实+渲染），并划分为训练、验证、两套测试集（normal、difficult），全部公开可获取。

**📈 对比分析**

对不同训练策略和模型变体采用 mAP@0.5 评价。最佳结果为 YOLOv11n 在混合训练/真实验证集上取得平均 0.884，SSD‑VGG16 0.876，RT‑DETRv2‑L 0.833；推理时间方面，YOLOv11n 在 Raspberry Pi 5 上可达 6.3 FPS，SSD 与 RT‑DETRv2 远超实时阈值。

**⚠️ 局限性**

局限性：数据集规模仍然有限，半合成图像可能带来域漂移；Transformer 方案在嵌入式设备上推理慢；YOLO 许可证限制商业部署；无人机硬件平台与导航、激光雷达等传感器的完整集成尚未完成。

---

## 142. Experiential Reflective Learning for Self-Improving LLM Agents

**arXiv ID:** 2603.24639 | [PDF](https://arxiv.org/pdf/2603.24639v1)

**作者:** Marc-Antoine Allard `[一作]` (Illuin Technology), Gautier Viaud `[通讯]` (Illuin Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了ERL框架，通过单次任务体验生成可复用启发式，并在新任务中检索并注入上下文以实现LLM代理自我改进。

**💡 创新点**

创新点在于利用单次尝试的经验生成可迁移的启发式，而非多次对比或完整轨迹，并通过LLM进行检索过滤提升相关性。

**🔧 技术方法**

采用LLM（GPT‑5‑mini）生成和检索启发式，结合ReAct代理架构，实现无参数更新的自适应。

**📊 数据集**

使用Gaia2基准（搜索和执行两分支，12个应用、101个工具），在8个训练宇宙上积累经验，在2个测试宇宙上评估。

**📈 对比分析**

与ReAct、ExpeL、AutoGuide、few‑shot等方法对比，ERL在整体成功率上达56.1%，比ReAct提升7.8%，并在pass@3和pass`3̂上显著提高可靠性。

**⚠️ 局限性**

局限包括对检索质量高度依赖，启发式池规模增长可能导致冲突和检索开销，以及在无成功/失败反馈的环境中效果受限。

---

## 143. Learning Mesh-Free Discrete Differential Operators with Self-Supervised Graph Neural Networks

**arXiv ID:** 2603.24641 | [PDF](https://arxiv.org/pdf/2603.24641v1)

**作者:** Lucas Gerken Starepravo `[一作]` (University of Manchester), Jack R. C. King `[通讯]` (University of Manchester)

**通讯引用:** 133 | [OpenAlex ID](https://openalex.org/A5087330185)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `3f18e8e3-0266-457c-8567-9039b6d2394d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了NeMDO，一种基于图神经网络的自监督学习框架，用于在无网格点云中生成离散微分算子

**💡 创新点**

创新点在于将多项式一致性约束作为自监督损失，直接学习基于几何的权重，获得可复用、分辨率无关且对几何不规则性鲁棒的算子

**🔧 技术方法**

采用图神经网络（GNN）实现消息传递，利用相对位置特征嵌入并预测权重；自监督训练使用Taylor展开的一致性矩阵；还使用了传统高阶方法LABFM与SPH核进行基准对比

**📊 数据集**

使用人工生成的噪声扰动的规则格点云（以不同扰动强度和粒子间距生成），以及Taylor–Green vortex等实际流场数据进行验证

**📈 对比分析**

与经典SPH（四次样条、Wendland C2）以及高阶一致性方法LABFM比较，NeMDO在相同邻域大小下实现了与LABFM相当的第二阶收敛，且在误差-成本曲线上明显优于SPH，且能在更低计算成本下达到更小误差；在Navier–Stokes模拟中误差降低约一阶且收敛更平稳

**⚠️ 局限性**

局限包括：当前模型仅支持固定邻域大小，无法自适应不同粒子数；对极端几何扰动的准确性随扰动增大而下降；在Lagrangian动力学场景下需配合粒子平滑或重构以维持邻域质量

---

## 144. How unconstrained machine-learning models learn physical symmetries

**arXiv ID:** 2603.24638 | [PDF](https://arxiv.org/pdf/2603.24638v1)

**作者:** Michelangelo Domina `[一作]` (École Polytechnique Fédérale de Lausanne), Michele Ceriotti `[通讯]` (École Polytechnique Fédérale de Lausanne)

**通讯引用:** 16259 | [OpenAlex ID](https://openalex.org/A5021241296)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一套严格的对称性诊断指标A_α和B_α，用于量化机器学习模型在学习物理对称性（如O(3)旋转对称）时的对称误差和内部特征的不可变/协变成分，并在两类无约束Transformer‑based模型（PET用于原子尺度模拟与PoLAr‑MAE用于粒子轨迹分类）上进行系统评估。

**💡 创新点**

创新点在于：① 将对称性误差与特征谱分解统一到一个可计算框架；② 通过对特征的谱分解揭示模型在训练过程中的对称性学习动态；③ 设计了仅在读取层进行的对称性“净化”正则化，可在不影响表达力的前提下显著降低对称性误差；④ 证明通过引入更高阶角动量的几何嵌入（SSH）可显著提升模型对高λ目标的学习能力。

**🔧 技术方法**

技术包括：Transformer‑based GNN（PET）、PointNet‑style编码器（PoLAr‑MAE）、Haar平均对称性评估、固体球谐函数嵌入（SSH）、正则化回归读取层、以及对称性梯度自适应训练。

**📊 数据集**

使用的数据集包括：① 1.5版本MAD（原子结构与PES）用于PET；② QCML分子数据集（电子密度投影）用于高λ目标实验；③ 公开的PILArNet LArTPC事件集用于PoLAr‑MAE。

**📈 对比分析**

通过对比随机旋转增强训练的无约束模型与严格约束的等变模型，实验显示无约束模型的对称误差远小于模型总体误差；对称性净化后能将能量与应力的对称误差降至基准的1/2；引入SSH至边缘几何嵌入后，模型在λ=8目标的RMSE从几乎无学习（几乎为零）提升至与传统等变模型相当的准确度。

**⚠️ 局限性**

局限性包括：① 对伪标量和高λ特征的学习仍存在显著延迟，模型需要额外的层级或训练时间；② 现有架构在边缘嵌入层缺乏足够的高阶角动量信息，导致对高λ目标学习效果有限；③ 对称性评估依赖Haar平均，对非紧致群如Lorentz群不可直接推广；④ 纯数据增强的对称性学习在极端复杂任务中可能无法完全覆盖所有对称轨道。

---

## 145. Decoding Market Emotions in Cryptocurrency Tweets via Predictive Statement Classification with Machine Learning and Transformers

**arXiv ID:** 2603.24933 | [PDF](https://arxiv.org/pdf/2603.24933v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 146. Spatio-Temporal Semantic Inference for Resilient 6G HRLLC in the Low-Altitude Economy

**arXiv ID:** 2603.24712 | [PDF](https://arxiv.org/pdf/2603.24712v1)

**作者:** Chuan-Chi Lai `[一作]` (National Chung Cheng University), Zhu Han `[通讯]` (University of Houston)

**通讯引用:** 89483 | [OpenAlex ID](https://openalex.org/A5063667378)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `fa81e2aa-eb25-4aba-a919-7efd247b3885`

**🎯 论文内容**

提出了EPIC框架，利用空间-时间语义推断(STSI)实现自主无人机群的主动协调，消除对网络时延和信号缺失的依赖。

**💡 创新点**

创新点在于将协调循环与物理信号层解耦，通过本地的主动推断机遇实现10 ms确定性反应时延，并在最大50 s信号空窗期内保持93.23%的覆盖效率。

**🔧 技术方法**

核心技术包括：离线学习的时空运动先验、低复杂度的衰减投影与空间一致性模块、数字孪生语义状态缓冲、O(N)规模的局部计算。

**📊 数据集**

本研究使用高保真6G空中网络仿真环境（1,800×1,800×100 m³、N=6、飞行高度100 m、频率6 GHz）进行实验，未使用公开真实数据集。

**📈 对比分析**

与传统被动反馈方案对比，EPIC将端到端反应时延从约150 ms降至9.73 ms（下降93.5%），在信号空窗期内覆盖效率提升10.5%，且对网络抖动保持10 ms不变的“战略免疫”。

**⚠️ 局限性**

局限性包括：依赖先验运动模型，对突发动态障碍或意外航迹偏差的适应性有限；实验仅在仿真环境中验证，缺乏真实部署测试；算法参数（如衰减因子α）需在不同场景中手动调优。

---

## 147. Framing Data Choices: How Pre-Donation Exploration Design Influence Data Donation Behavior and Decision-Making

**arXiv ID:** 2603.24995 | [PDF](https://arxiv.org/pdf/2603.24995v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 148. Saranga: MilliWatt Ultrasound for Navigation in Visually Degraded Environments on Palm-Sized Aerial Robots

**arXiv ID:** 2603.24699 | [PDF](https://arxiv.org/pdf/2603.24699v1)

**作者:** Manoj Velmurugan `[一作]` (Worcester Polytechnic Institute), Nitin J. Sanket `[通讯]` (Worcester Polytechnic Institute)

**通讯引用:** 450 | [OpenAlex ID](https://openalex.org/A5055752014)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `51c0528b-f690-4182-ae60-bb5f046c276c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

开发了一套低功耗超声波感知系统 Saranga，实现小型无人机在视觉失效环境（雾、低光、雪等）下的障碍物检测与避障。

**💡 创新点**

结合物理噪声屏蔽与深度学习去噪网络，突破超声波 PSNR 低至 -4.9 dB 的限制，使双声纳阵列在恶劣环境中可达 1.66 m 的有效检测距离。

**🔧 技术方法**

使用双低功耗 MEMS 叠加超声波 chirp 传感器、物理噪声屏蔽、基于 UNet 的 Saranga 去噪网络、M‑mode 影像化、双目时间差定位以及势场避障控制；硬件上集成 Google Coral Mini Edge TPU。

**📊 数据集**

通过合成的模拟回波数据与少量真实噪声采集相结合的“仿真‑实境”数据集；室内外实验环境涵盖透明墙、薄物体、雪、雾、低光以及森林场景。

**📈 对比分析**

与 BatDeck、传统去噪滤波（Gaussian、TV‑L1、TDLMS+Sobel 等）以及 LIDAR、RADAR、RealSense 等传感器对比；Saranga 在 PSNR -4.9 dB 时误差比传统方法低 30% 以上，避障成功率在不同场景下为 77%–92%，显著优于 BatDeck（<10%）及视觉/激光传感器在雾暗环境下的 0–40%。

**⚠️ 局限性**

仍受限于极薄或低反射率障碍物的检测距离仅 0.5–1 m，物理噪声屏蔽受尺寸/重量约束；对高速飞行时多路径与风扰动的鲁棒性不足，且需要更轻量化的计算平台才能实现更小型机型。

---

## 149. Dissecting Model Failures in Abdominal Aortic Aneurysm Segmentation through Explainability-Driven Analysis

**arXiv ID:** 2603.24801 | [PDF](https://arxiv.org/pdf/2603.24801v1)

**作者:** Abu Noman Md Sakib `[一作]` (University of Texas at San Antonio), Ender A. Finol `[通讯]` (University of Texas at San Antonio)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

开发了基于XAI引导的编码器塑形框架XAI‑SAM，用于复杂腹主动脉瘤CT图像的外壁和腔室分割；

**💡 创新点**

创新点在于将梯度归因（XAI field）作为训练信号，实现预测概率与编码器关注区域的对齐，并通过轻量化精炼路径和置信度先验抑制干扰，同时引入配对一致性分类器学习切片间的解剖连贯性；

**🔧 技术方法**

使用梯度归因（Grad‑CAM/梯度加权）、ViT编码器、SAM分割解码器、对齐损失、差异与拓扑正则化、配对一致性分类器以及形状一致性约束；

**📊 数据集**

基于147例CTA扫描（9037切片）构成的AAA数据集，包含外壁和腔室双重标注，80/20患者分割，且挑选出复杂子集进行评估；

**📈 对比分析**

与U‑Net、U‑Net++、nnU‑Net、MedSAM、SAM baseline进行对比；在一般集和复杂子集上，XAI‑SAM在IoU/Dice/HD95上均显著提升，复杂子集IoU突破96%，Dice超过97%，HD95降至0.2–0.5 mm；

**⚠️ 局限性**

局限在于XAI字段仅基于单一归因方法，可能无法完整捕捉编码器决策；训练为单任务，未考虑双标记共性；仅在AAA任务验证，跨任务泛化性待进一步验证。

---

## 150. Integrated Multi-Drone Task Allocation, Sequencing, and Optimal Trajectory Generation in Obstacle-Rich 3D Environments

**arXiv ID:** 2603.24908 | [PDF](https://arxiv.org/pdf/2603.24908v1)

**作者:** Yunes Alqudsi `[一作]`, Murat Makaraci `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

提出了IMD–TAPP框架，实现多无人机在三维障碍环境中任务分配、序列规划与轨迹生成的端到端协同；

**💡 创新点**

通过将障碍感知的代价矩阵与注入粒子群优化结合，并引入多线性分配（MLA）指导，提高了分配/序列搜索的收敛速度和任务完成时间；

**🔧 技术方法**

使用三维网格离散化+A*图搜索构建代价矩阵，注入粒子群优化（IPSO）进行任务分配与序列优化，随后采用最小快（minimum‑snap）多段多项式生成平滑轨迹并进行安全验证与局部再规划；

**📊 数据集**

采用仿真环境的三维障碍场（由研究者自行搭建），并未使用公开标准数据集；

**📈 对比分析**

在MATLAB仿真中，二架无人机完成七个目标的任务，总时间为136 s，满足碰撞与安全间距约束，性能优于传统分离式分配与轨迹规划方法；

**⚠️ 局限性**

局限性包括仅在仿真中验证，缺乏硬件实验；对状态估计误差、通信延迟及实时计算性能未充分评估；框架仅支持同质四旋翼且未考虑能耗、时间窗口等复杂约束。

---

## 151. How Far Are Vision-Language Models from Constructing the Real World? A Benchmark for Physical Generative Reasoning

**arXiv ID:** 2603.24866 | [PDF](https://arxiv.org/pdf/2603.24866v1)

**作者:** Luyu Yang `[一作]` (Salesforce AI Research), Zeyuan Chen `[通讯]` (Salesforce AI Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 DreamHouse 基准，评估视觉-语言模型在建筑结构生成中的物理可行性。

**💡 创新点**

创新点在于：①将结构、构造和规范合规性纳入生成评估；②使用可迭代的 agentic 交互框架；③提供 10 项确定性结构验证测试与 LoD 350 规范的映射。

**🔧 技术方法**

主要技术包括：Blender Python 代码生成与执行、结构图验证器、Alpha‑加权 MSE 视觉相似度、基于图的载荷路径与连接检查。

**📊 数据集**

使用了 26,543 条完整的木结构住宅模型，覆盖 13 种建筑风格，包含多视角渲染与完整的结构元数据。

**📈 对比分析**

与 GPT‑5、Gemini 3 Flash、Claude 4.5 进行对比，采用三种生成协议（全局单步、全局单步加中间检查、分阶段步骤），指标为结构有效率、视觉相似度和联合通过率；最优模型仅达 7.1% 的联合通过率，说明当前模型在物理生成方面存在显著缺口。

**⚠️ 局限性**

局限性包括：对计算成本敏感、仅覆盖木结构、未处理多材料或钢混凝土、未对模型进行针对性微调、验证器依赖硬件可变的渲染环境。

---

## 152. Context-Mediated Domain Adaptation in Multi-Agent Sensemaking Systems

**arXiv ID:** 2603.24858 | [PDF](https://arxiv.org/pdf/2603.24858v1)

**作者:** Anton Wolter `[一作]` (Aarhus University), Niklas Elmqvist `[通讯]` (Aarhus University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一种“Context-Mediated Domain Adaptation (CMDA)”框架，利用用户对AI生成的文本（如研究问题）进行的直接编辑和提示式再生成，将用户的隐性领域专业知识自动提取并反馈给多智能体系统，实现双向知识共享与持续改进。

**💡 创新点**

创新点在于：① 将用户编辑视为隐式域规范，而非单纯的修正；② 建立了基于“bdar + aco”结构的双向语义链接，能够记录生成上下文与编辑历史；③ 通过三种交互模式（直接编辑、提示式再生成、基于上下文生成）实现实时、跨用户的知识累积与注入；④ 通过系统化的日志与可观测性工具（Langfuse、自定义监控）验证知识迁移效果。

**🔧 技术方法**

技术实现主要包括：<br>• 前端：Next.js + React，提供实时编辑与交互模式切换；<br>• 后端：Python + FastAPI + LangGraph，用于多智能体工作流管理与知识抽取；<br>• 数据库：PostgreSQL，存储“bdar”实例、抽取的知识条目与用户行为日志；<br>• LLM：Google Gemini（或其他LLM）完成文本生成与知识抽取；<br>• 监控：Langfuse + 自定义指标跟踪生成质量、编辑距离与成本；

**📊 数据集**

使用的主要数据集：<br>① 3 篇可视化素养领域的最新论文（《Tell Me Without Telling Me》, 《DRIVE-T》, 《Charts-of-Thought》）；<br>② 5 名可视化专家的编辑日志与质量评分；<br>③ 存储的生成与编辑的原始与最终文本，用于知识抽取。

**📈 对比分析**

对比方法：在每位参与者的实验中，记录生成前的质量评分、编辑距离、编辑次数与时间，并与后续受累积知识影响的生成结果对比。结果显示：<br>• 参与者序号递增，初始质量评分从约2.7提升至4.3；<br>• 编辑距离与编辑次数未呈单调下降，但质量提升显著；<br>• 通过抽取的46条域知识，系统在后续论文生成中减少了对用户修正的需求。虽然样本量小且缺乏对照组，仍证明CMDA在可视化素养领域可提升生成质量。

**⚠️ 局限性**

局限性：<br>1. 仅在可视化素养三篇论文上评估，缺乏跨领域验证；<br>2. 样本规模为5位专家，难以作因果推断；<br>3. 知识抽取依赖LLM，可能出现标签不一致或误抽取；<br>4. 系统累积的域知识对终端用户不可见，未实现可视化或编辑建议功能；<br>5. 需要专家级参与，对普通用户友好度有限。

---

## 153. Infinite Gaze Generation for Videos with Autoregressive Diffusion

**arXiv ID:** 2603.24938 | [PDF](https://arxiv.org/pdf/2603.24938v1)

**作者:** Jenna Kang `[一作]` (New York University), Qi Sun `[通讯]` (New York University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种自回归扩散模型，用于生成任意长度视频的连续原始视线轨迹；

**💡 创新点**

创新点在于结合自回归扩散与视觉注意力的稀疏特征编码，实现了无限时域、细粒度的视线生成；

**🔧 技术方法**

采用DDPM框架的U‑Net扩散网络，配合saliency‑aware视觉潜在表示与自回归生成；

**📊 数据集**

主要使用DIEM视频数据集（长时序、高分辨率）以及DHF1K视频数据集进行训练与评估；

**📈 对比分析**

与HAT、GazeFormer、TPP‑Gaze、DeepGaze III、DiffEye等基线进行对比，使用Levenshtein、Fréchet、DTW和最大时序相关等指标，结果显示本方法在绝大多数指标上均优于基线，且在用户研究中得到85%以上的匹配率；

**⚠️ 局限性**

局限性包括对长范围依赖的处理仍基于固定窗口，可能无法捕捉更长期的视线模式；度量指标对短期偏差敏感，且未考虑头部运动等因素。

---

## 154. Quantum Inspired Vehicular Network Optimization for Intelligent Decision Making in Smart Cities

**arXiv ID:** 2603.24971 | [PDF](https://arxiv.org/pdf/2603.24971v1)

**作者:** Kamran Ahmad Awan `[一作]` (University of Haripur), Ahmed Farouk `[通讯]` (Hamad Bin Khalifa University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出并实现了一种量子启发式的车辆网络优化模型（QIVNOM），在边缘–云架构上联合优化V2V/V2I通信与城市交通控制，显著降低延迟、提升可靠性，并改善交通流量。

**💡 创新点**

创新点包括：① 将概率超位置（superposition）与纠缠耦合（entanglement‑style regularizer）引入经典优化框架；② 采用球面投影梯度与annealed采样实现高效搜索；③ 结合Tchebycheff多目标标量化与可行性投影，同时引入Lyapunov漂移与chance约束，构建鲁棒、容错的联合优化方案；④ 通过熵最优传输将全局计划分配至雾节点，形成完整的分布式系统。

**🔧 技术方法**

使用技术包括：量子启发式算法（superposition、entanglement）、球面投影梯度、annealed采样、Tchebycheff标量化、可行性投影、Lyapunov控制、Chance约束、熵最优传输、CVaR微策略；仿真平台为SUMO+OMNeT++/Veins；数据源为METR‑LA交通速度数据与OpenStreetMap道路图。

**📊 数据集**

使用的主要数据集为METR‑LA公共交通速度数据（5‑min速率）并配合OpenStreetMap 5km×5km路网；仿真中还包含IEEE 802.11p与5G NR侧链的网络层模型。

**📈 对比分析**

与多种基线（PBQEI、VQAM、QIDEF、SMC、UADM、RUMP）在六种场景下对比：平均端到端延迟降低约20%，PDR与可靠性提升约2.3个百分点，平均旅行时间降低约11–13%，拥堵指数下降约10–12%。在事故、RSU故障、负载高峰等压力情境下，QIVNOM仍保持显著优势。

**⚠️ 局限性**

局限性包括：① 结果基于仿真，未在真实城市环境中验证；② 对超参数（温度β、风险δ、种群大小K）敏感，需要进一步自动化调参；③ 计算与能耗成本尚未全面量化；④ 对不同通信协议、网络规模的泛化能力尚待评估。

---

## 155. Design Once, Deploy at Scale: Template-Driven ML Development for Large Model Ecosystems

**arXiv ID:** 2603.24963 | [PDF](https://arxiv.org/pdf/2603.24963v1)

**作者:** Jiang Liu `[一作]` (Meta AI), Alireza Vahdatpour `[通讯]` (Meta AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出并实现了Standard Model Template（SMT）框架，对Meta大规模推荐系统中的数百个模型进行标准化，显著降低模型迭代复杂度并提升性能。

**💡 创新点**

核心创新在于将技术迭代与模型迭代解耦，采用模板化设计与多模型优化（MMO）结合代表性模型聚类，实现技术在O(n+k)复杂度下快速推广，并通过回归率阈值筛选保证泛化效果。

**🔧 技术方法**

使用了模板化模块化架构、贝叶斯优化、多模型优化、k‑means代表性模型聚类、回归率与增量阈值判定、自动化工程流水线以及可配置超参曝光等技术手段。

**📊 数据集**

实验基于Meta内部排名与检索生态的≈50%模型（共约252个），使用约20个代表性模型进行技术评估，覆盖多维度（排名阶段、模型规模、硬件、优化事件等）。

**📈 对比分析**

与传统独立优化模型在四个周期内对比，SMT实现离线NE平均提升0.63%、在线lift累计0.86%，工程时间减少92%，技术传播吞吐量提升6.3倍。

**⚠️ 局限性**

局限包括约5.6%模型难以迁入模板、泛化优化牺牲部分单模型的细粒度调优收益，以及需对高方差超参进行曝光以保持必要的模型灵活性。

---

## 156. Information-Theoretic Limits of Node Localization under Hybrid Graph Positional Encodings

**arXiv ID:** 2603.25030 | [PDF](https://arxiv.org/pdf/2603.25030v1)

**作者:** Zimo Yan `[一作]` (National University of Defense Technology), Runfan Duan `[通讯]` (National University of Defense Technology)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `57a58b01-81b4-4d75-a45c-2e891f272b50` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了在混合定位编码（anchor距离向量+量化低频谱特征）下的节点定位可识别性，提出信息论不可识别界并通过实验验证其图依赖性。

**💡 创新点**

创新点在于：①将节点定位视为观测映射问题，给出通用信息论逆界；②针对能量谱嵌入提供桶式碰撞与平衡细化定理；③通过理论引导的预算比例揭示随机正则图上的相位转移；④展示不同图结构对可识别性的显著影响。

**🔧 技术方法**

技术包括：基于拉普拉斯谱的低频投影、anchor距离编码、坐标量化、观测映射图像大小计数、信息论逆推、桶式碰撞-平衡计数、随机正则图理论。

**📊 数据集**

数据集：随机3正则图（n=500,1000,2000,4000）、DrugBank DDI图（1684节点、189774条边）以及Decagon派生DDI图（40节点、466条边）。

**📈 对比分析**

对比方法：将完整观测映射与仅距离或仅谱、不同anchor策略、不同量化精度进行 ablation；性能指标为定位错误率、图像大小比例和平均预像尺寸。实验显示在随机正则图上，错误率随预算比例ρeng→1快速下降；在DrugBank中仍高度碰撞；在Decagon中加入丰富谱信息后错误率可降至≈1%。

**⚠️ 局限性**

局限性：①仅给出不可识别下界，缺乏匹配的可实现阈值；②理论仅针对随机正则图与正交不变谱嵌入；③实测使用相对量化规则与理论的绝对量化不完全对齐；④未直接关联定位可识别性与下游任务性能。

---

## 157. System-Anchored Knee Estimation for Low-Cost Context Window Selection in PDE Forecasting

**arXiv ID:** 2603.25025 | [PDF](https://arxiv.org/pdf/2603.25025v1)

**作者:** Wenshuo Wang `[一作]` (South China University of Technology), Fan Zhang `[通讯]` (Zhejiang University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了自回归神经 PDE 模拟器的上下文窗口选择问题，提出了系统锚定膝点估计 (SAKE) 两阶段低成本选择方法，并在 PDEBench 上进行实验验证。

**💡 创新点**

创新点在于将上下文窗口选择正式化为独立的低成本算法问题，利用系统动力学提取物理可解释的窗口锚点，并在锚点空间内进行膝点感知的下游选择，显著提升选择的可靠性和效率。

**🔧 技术方法**

采用系统级自回归模型 (VAR)、上限置信区间 (UCB)、统计检验、低成本 Pilot 训练、膝点感知评分规则，以及多种神经算子（U-Net、FNO、ConvLSTM、Transformer）与 PDEBench 基准进行融合。

**📊 数据集**

实验使用 PDEBench 的两个前向任务（DiffReact 与 Radial Dam Break）以及全部八个 PDEBench 家族的数据集。

**📈 对比分析**

与全搜索基准、直接低成本搜索（Direct‑3/4‑Shortlist）、系统内存估计（System‑core）及 ASHA 等方法比较，SAKE 在匹配预算下实现 67.8% 的准确率、91.7% 的 within‑1、平均 regret 6.1% 以及 0.051 的成本比，明显优于对手。

**⚠️ 局限性**

局限性包括仅适用于固定窗口一阶自回归模型、依赖完整或可代理的系统轨迹、在某些难点如 RDB/ConvLSTM 仍难以精确定位膝点，以及不适用于内部自适应记忆的架构。

---

## 158. C2W-Tune: Cavity-to -Wall Transfer Learning for Thin Atrial Wall Segmentation in 3D Late Gadolinium-enhanced Magnetic Resonance

**arXiv ID:** 2603.24992 | [PDF](https://arxiv.org/pdf/2603.24992v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 159. Few-Shot Left Atrial Wall Segmentation in 3D LGE MRI via Meta-Learning

**arXiv ID:** 2603.24985 | [PDF](https://arxiv.org/pdf/2603.24985v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 160. Group-Differentiated Discourse on Generative AI in High School Education: A Case Study of Reddit Communities

**arXiv ID:** 2603.24972 | [PDF](https://arxiv.org/pdf/2603.24972v1)

**作者:** Parth Gaba `[一作]` (Valley Christian High School), Emiliano De Cristofaro `[通讯]` (Valley Christian High School)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

分析了高中教育中不同 Reddit 社区对生成式 AI 的讨论，重点关注学习影响、学术诚信、检测工具及情绪表达，并探讨检测工具对学生情绪的负面影响。

**💡 创新点**

1) 通过社区角色对话语进行细分，首次揭示教师与学生在学习价值观上的双重框架差异；2) 量化检测工具讨论与负面情绪的关联，显示情绪伤害不对称分布；3) 采用人机协同的 LLM 过滤与标注框架，提升大规模文本分析的可行性。

**🔧 技术方法**

关键词检索 + LLM 主导的主题过滤；人机标注 + 三方可靠性评估；统计检验（卡方检验、Cramér's V、风险差异）以比较不同群体间的标签出现率；情绪与检测讨论的相关性分析。

**📊 数据集**

3,789 条来自 5 个与教育相关的 subreddits（r/teachers、r/teenagers、r/highschool、r/chatgpt、r/askteachers）的帖子，数据时间跨度为 2022 年 12 月至 2024 年 10 月。

**📈 对比分析**

采用卡方检验并报告效应量（Cramér's V、ϕ、风险差异）比较学生、教师和混合社区的标签分布；检测讨论与负面情绪的关联在所有群体中显著，学生与混合社区的风险差异更大（+0.194、+0.200），教师仅 +0.103；效果均超过传统统计显著性阈值，证明群体差异显著且具实际意义。

**⚠️ 局限性**

1) Reddit 样本不具代表性，用户偏年轻、技术导向；2) 组别不平衡，教师样本远大于学生，可能稀释差异；3) LLM 标注的可靠性虽与人类相当但仍有限，低共识标签需谨慎解释；4) 统计检验假设独立性受限，多发帖用户可能导致结果偏倚；5) 研究时间窗口有限，AI 与检测技术持续演进，结论在未来可能需要更新。

---

## 161. GDPO-Listener: Expressive Interactive Head Generation via Auto-Regressive Flow Matching and Group reward-Decoupled Policy Optimization

**arXiv ID:** 2603.25020 | [PDF](https://arxiv.org/pdf/2603.25020v1)

**作者:** Zhangyu Jin `[一作]` (University of Southern California), Mohammad Soleymani `[通讯]` (University of Southern California)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 GDPO‑Listener，一种基于 Auto‑Regressive Flow Matching 并通过 Group reward‑Decoupled Policy Optimization (GDPO) 的后训练强化学习框架，用来生成具有高度表达性、与语境一致的说话与倾听头部运动。

**💡 创新点**

创新点包括：1）将 FLAME 参数空间扩展到眼睑闭合、眼球姿态和全局头部旋转，支持更丰富的非语言行为；2）采用 Auto‑Regressive Flow Matching 在连续潜空间中实现稳定的长序列生成；3）利用 GDPO 通过分组奖励和解耦优化突破“回归到均值”问题，显著提升倾听动作的多样性和自然度；4）引入多模态前缀（文本、音频、伴侣动作）与 Classifier‑Free Guidance，支持可解释的语义与强度控制。

**🔧 技术方法**

核心技术包括 Transformer‑based VAE、TMRoPE（时间对齐多模态旋转嵌入）、自回归流匹配（AR‑Flow）与 SDE 采样、组奖励解耦的 PPO‑style GDPO、以及 CFG 用于表达度可调。

**📊 数据集**

使用 Seamless Interaction 数据集（718h 训练，31.5h 测试）和 DualTalk 数据集，且在两者上构造表达性子集进行评估。

**📈 对比分析**

与 DiffposeTalk、L2L、ARTalk、DualTalk 等基线相比，在说话端实现 LVE≈2.95、MHD≈1.06，倾听端在 FDD、PDD、JDD 上显著下降（如 FDD 18.85 vs DualTalk 43.58，Expressive Subset FDD 22.56 vs DualTalk 55.94），证明了更高的动态一致性与表达多样性；用户研究也显示其感知质量优于或相当于基线。

**⚠️ 局限性**

局限性包括：需要额外的后训练 RL 阶段（GDPO）增加训练成本；对极端高频动态或长时序的鲁棒性尚未彻底验证；模型主要针对头部运动，无法直接生成全身或全景交互；对多模态输入同步仍可能受限于硬件延迟。

---

## 162. PASDiff: Physics-Aware Semantic Guidance for Joint Real-world Low-Light Face Enhancement and Restoration

**arXiv ID:** 2603.24969 | [PDF](https://arxiv.org/pdf/2603.24969v1)

**作者:** Yilin Ni `[一作]` (Nanjing University of Posts and Telecommunications), Jian Yang `[通讯]` (Nanjing University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出 PASDiff，一种训练无关的物理感知语义扩散框架，用于低光环境下人脸图像的联合增强与恢复。

**💡 创新点**

创新点在于将曝光与Retinex反射的物理约束与Style‑Agnostic Structural Injection（SASI）相结合，形成多目标能量引导，实现光照、色彩与结构的解耦；同时通过Statistic‑Aligned Guidance Loss滤除先验模型的光照色彩偏差。

**🔧 技术方法**

使用扩散概率模型（DDPM）及其逆向采样、Retinex分解网络、预训练人脸恢复网络、AdaIN对齐、能量引导梯度等技术。

**📊 数据集**

使用合成FFHQ低光数据与自建真实低光人脸基准WildDark‑Face（700张）进行训练与评估。

**📈 对比分析**

与级联方法（LightenDiffusion→DiffBIR、DiffBIR→L‑Diff）以及联合基线（DarkIR、FDN、LEDNet等）对比，PASDiff在PSNR、LPIPS、DISTS、FID、身份识别准确率等指标上均显著领先，尤其在身份保留与自然照明方面表现最佳。

**⚠️ 局限性**

主要局限在于扩散采样需要多步迭代，导致推理速度慢；以及在极端低光条件下的色彩恢复仍可能出现欠饱和等问题。

---

## 163. Fast Spanning Tree Sampling in Broadcast Congested Clique

**arXiv ID:** 2603.25018 | [PDF](https://arxiv.org/pdf/2603.25018v1)

**作者:** Nima Anari `[一作]` (Stanford University), Alireza Haqi `[通讯]` (Stanford University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种在广播拥塞Clique模型中采样随机生成树的多对数轮算法。

**💡 创新点**

这是首个多对数轮的BCC算法，显著改善了之前的算法性能，提供了更高效的随机生成树采样。

**🔧 技术方法**

使用了基于近似生成树采样框架的算法，结合了拉普拉斯稀疏化技术和边缘概率估计。

**📊 数据集**

使用了带权图的生成树分布，具体数据集未明确给出，但假设边权为非负整数且满足一定条件。

**📈 对比分析**

与之前的最佳算法相比，新的算法在运行时间上有指数级的改进，能够在O(log(nU) log(1/ε))轮内完成采样，性能显著提升。

**⚠️ 局限性**

算法的局限性在于对输入图的边权有特定的假设，并且在某些情况下可能需要较大的常数因子来保证性能。

---

## 164. Learning Rollout from Sampling:An R1-Style Tokenized Traffic Simulation Model

**arXiv ID:** 2603.24989 | [PDF](https://arxiv.org/pdf/2603.24989v1)

**作者:** Ziyan Wang `[一作]` (Beihang University), Guizhen Yu `[通讯]` (Beihang University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c773407a-6119-4871-b8b3-1e7ae17a6851` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种基于R1-Style的交通仿真框架R1Sim，结合熵引导自适应采样与Group Relative Policy Optimization（GRPO）来生成既安全又多样的多智能体行驶轨迹。

**💡 创新点**

核心创新包括：① 用运动令牌熵动态调节采样宽度，从高不确定性时大幅扩展探索；② 在GRPO中引入安全感知奖励与组相对优势估计，避免传统SFT对真实轨迹的过度依赖；③ 通过消除优势标准差归一化，提升训练稳定性。

**🔧 技术方法**

技术方法主要包括：令牌化轨迹的下一令牌预测（NTP）模型、熵引导自适应采样、GRPO强化学习框架、基于SAT的碰撞检测与指数衰减距离奖励。

**📊 数据集**

在Waymo Open Motion Dataset（WOMD）上进行训练与评估，使用训练/验证/测试场景分别为486,995 / 44,097 / 44,920。

**📈 对比分析**

与SMART、CATK等SOTA模型对比，R1Sim在WOMD Test集上在交互、运动学与地图基准指标均表现出色，尤其在交互安全度和整体真实度（RMM）上提升约0.01-0.02分。

**⚠️ 局限性**

局限性包括：① 仍需大量算力（4xRTX4090）和较长训练时间；② 对超大场景的采样规模受限，过大K值会引入噪声；③ 只针对离散令牌的运动模型，未考虑连续控制细节。

---

## 165. Epistemic Compression: The Case for Deliberate Ignorance in High-Stakes AI

**arXiv ID:** 2603.25033 | [PDF](https://arxiv.org/pdf/2603.25033v1)

**作者:** Steffen Lukas `[一作]` `[通讯]` (Charité – Universitätsmedizin Berlin), Steffen Lukas (Charité – Universitätsmedizin Berlin)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

研究提出 Epistemic Compression 原理并构建 Regime Index，评估何时应优先使用高复杂度模型或压缩模型，并在 15 个高风险领域进行实证验证。

**💡 创新点**

创新点在于将奥卡姆剃刀转化为可实现的压缩架构（如 CRATE），提出可根据数据稳定性自适应的白盒模型，并量化 Shifting 与 Stable 两种环境的范式差异。

**🔧 技术方法**

使用白盒压缩架构（CRATE、Rate‑Reduction）、Regime Index 指标、Viability Gap 理论，并与高/低容量模型进行对比实验。

**📊 数据集**

数据集包括 LendingClub 贷款记录、MIMIC‑IV ICU 记录、UCI Adult 收入数据，以及 15 个公开高风险领域的标准数据集。

**📈 对比分析**

通过训练、OOS 与稳健 AUROC 对比，压缩模型在 Shifting 环境下的鲁棒性优于高容量模型，且与 Regime Index 的匹配率达 86.7%。

**⚠️ 局限性**

局限性包括元分析的选择偏差、缺乏正式风险评估、仅覆盖已公开的 shift 评估域，以及白盒架构在其他领域的泛化能力尚待验证。

---

## 166. Error Understanding in Program Code With LLM-DL for Multi-label Classification

**arXiv ID:** 2603.25005 | [PDF](https://arxiv.org/pdf/2603.25005v1)

**作者:** Md Faizul Ibne Amin `[一作]`, Md. Shahajada Mia `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了基于大型语言模型与循环神经网络融合的多标签程序错误分类框架。

**💡 创新点**

首次系统性地将预训练的代码专用LLM与GRU/LSTM/BiLSTM等RNN解码器结合，并通过Optuna优化实现高效错误识别。

**🔧 技术方法**

采用CodeT5、GraphCodeBERT、CodeT5+等LLM编码器，结合GRU、LSTM、BiLSTM及BiLSTM‑A等深度学习模块，以及二元交叉熵损失、Dropout、Adam等训练策略。

**📊 数据集**

使用从AOJ系统收集的95,631对学生错误与正确Python代码样本，经过标签汇总得到11类多标签错误集合。

**📈 对比分析**

通过多种多标签指标（AvgAcc、EMAcc、F1_ψ、Hamming Loss、ROC‑AUC等）对32种模型组合进行评估，CodeT5+_GRU在权重F1达到0.8243，AvgAcc 91.84%，Hamming Loss 0.0816，显示显著优于单一LLM或传统方法。

**⚠️ 局限性**

实验仅针对Python单一数据集，模型对其他语言及极度不平衡标签场景的泛化有限，且训练成本与GPU资源需求仍较高。

---

## 167. Interpretable Zero-shot Referring Expression Comprehension with Query-driven Scene Graphs

**arXiv ID:** 2603.25004 | [PDF](https://arxiv.org/pdf/2603.25004v1)

**作者:** Yike Wu `[一作]` (University of Technology Sydney), Jian Zhang `[通讯]` (University of Technology Sydney)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种零样本指代表达理解方法SGREC，利用查询驱动的场景图和LLM进行目标定位。

**💡 创新点**

创新点在于将视觉场景转化为结构化文本场景图，让LLM可进行可解释推理，且不需要任务特定微调。

**🔧 技术方法**

结合VLM（如VinVL、LLaVA）生成检测与关系，利用LLM（LLaMA、Qwen）进行推理，并使用词向量匹配抽取查询相关对象。

**📊 数据集**

在RefCOCO、RefCOCO+和RefCOCOg三大基准数据集上进行评估。

**📈 对比分析**

相较于现有零样本方法，SGREC在三大数据集上平均Top‑1准确率最高，尤其在RefCOCOg上提升10%以上。

**⚠️ 局限性**

主要限制是计算成本高，需要两个大型模型（VLM+LLM）生成场景图与推理，且在极密集场景下仍有误检。

---

## 168. Towards Video Anomaly Detection from Event Streams: A Baseline and Benchmark Datasets

**arXiv ID:** 2603.24991 | [PDF](https://arxiv.org/pdf/2603.24991v1)

**作者:** Peng Wu `[一作]` (Northwest Polytechnical University), Yanning Zhang `[通讯]` (Northwest Polytechnical University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `8d10c613-917e-4880-9716-17789f50e119` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文建立了基于事件摄像机的视频异常检测基线EWAD，并通过模拟生成了同步RGB-事件的三大基准数据集（UCF-Crime、CCTV-Fights、UBnormal）。

**💡 创新点**

创新点包括：①事件密度感知动态采样（EDS），自适应选择高密度事件段；②密度调制时间注意机制（EDA），将事件密度与时间距离融入注意力；③RGB到事件的双层知识蒸馏（B-KD+M-KD），在训练时将RGB教师模型的高层语义迁移到事件学生模型。

**🔧 技术方法**

使用的技术包括ViT特征提取、多任务MIL弱监督、EDA注意机制、EDS采样策略、双层知识蒸馏、以及事件帧化与空间定位后处理。

**📊 数据集**

数据集方面，作者通过v2e模拟在三大RGB VAD 数据集（UCF-Crime、CCTV-Fights、UBnormal）上生成事件流，并在18条真实EventVOT事件视频上进行验证。

**📈 对比分析**

与SNN、SNN-SNN、ViT-DNN等基线在三数据集上进行对比，EWAD在UCF-Crime上AUC达76.55%（比MSF提升11.5%），在CCTV-Fights和UBnormal亦保持领先；空间定位性能为13.28% TIoU，略低于RGB方法但已具竞争力。

**⚠️ 局限性**

主要限制是依赖模拟事件数据，缺乏大规模真实事件流；空间定位准确率仍低于RGB方法；未来需要真实事件数据、跨模态融合以及专用事件流基础模型。

---

## 169. Rethinking Health Agents: From Siloed AI to Collaborative Decision Mediators

**arXiv ID:** 2603.24986 | [PDF](https://arxiv.org/pdf/2603.24986v1)

**作者:** Ray-Yuan Chung `[一作]` (University of Washington), Ari Pollack `[通讯]` (Seattle Children’s Hospital)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出并探讨了将大语言模型（LLM）嵌入多方医疗协作中的 AI 协作者框架，并通过临床验证的虚构儿科慢性肾病案例说明其在共享情境意识、对齐心理模型和支撑协作决策方面的潜力。

**💡 创新点**

创新点在于：①将 AI 从单一助手转变为多方协作的共同体成员；②提出以团队情境意识（SA）为指导的协作设计原则；③强调 AI 作为信息媒介而非决策者，保持人类决策权和病人自主性。

**🔧 技术方法**

未给出具体实现技术，概念性讨论基于通用 LLM（如 GPT 系列）及其在医疗文本处理、情境信息提取与对话管理方面的潜在应用。

**📊 数据集**

未使用公开数据集；案例数据为临床验证的虚构病例，来源于儿科肾脏科医生与注册营养师的合作。

**📈 对比分析**

未进行实验对比或性能评估；仅通过案例演示展示该框架在促进共享认知、降低不遵从风险方面的示范效果。

**⚠️ 局限性**

局限性包括：①缺乏可实现的技术细节和原型验证；②未评估多方使用中的可用性、负载与隐私风险；③依赖理论和单一案例，缺乏广泛实证支持。

---

## 170. Exons-Detect: Identifying and Amplifying Exonic Tokens via Hidden-State Discrepancy for Robust AI-Generated Text Detection

**arXiv ID:** 2603.24981 | [PDF](https://arxiv.org/pdf/2603.24981v1)

**作者:** Xiaowei Zhu `[一作]` (Chinese Academy of Sciences), Li Guo `[通讯]` (Chinese Academy of Sciences)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种训练无关的 AI 文本检测方法 Exons-Detect，利用隐藏状态差异识别并加权重要 token，从而提高检测精度。

**💡 创新点**

创新点在于将基因外显子/内含子概念引入文本检测，使用隐藏状态差异自适应地为 token 赋予不同重要性权重，显著提升鲁棒性和准确率。

**🔧 技术方法**

技术包括双模型隐藏状态余弦差异计算、非线性映射生成权重、加权 log-perplexity 与 cross-perplexity 计算以及 mutation-repair 机制来构建翻译得分。

**📊 数据集**

实验数据集涵盖 M4、RealDet、DetectRL 三大公开基准，且在 DetectRL 中进一步评估多 LLM 与多域设置。

**📈 对比分析**

与多种训练基与无训练基方法对比，Exons-Detect 在三大基准上的平均 AUROC 达到 92.14%，在所有数据集上均超过 90%，并表现出对攻击和短文本的强鲁棒性。

**⚠️ 局限性**

局限性在于仅用余弦距离衡量隐藏差异，可能不足以捕获更细粒度的来源信息，未来可探索更精细的 token 重要性评估方法。

---

## 171. Select, Hypothesize and Verify: Towards Verified Neuron Concept Interpretation

**arXiv ID:** 2603.24953 | [PDF](https://arxiv.org/pdf/2603.24953v1)

**作者:** ZeBin Ji `[一作]` (Chongqing University of Posts and Telecommunications), Bin Xiao `[通讯]` (Chongqing University of Posts and Telecommunications)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并实现了 SIEVE（Select–Hypothesize–Verify）框架，用于解释深度神经网络层级中神经元的功能，并通过验证过滤错误概念。

**💡 创新点**

创新点在于将观测-假设-验证闭环引入神经元解释，结合样本选择、聚类和生成式验证，实现更精确、可验证的概念映射。

**🔧 技术方法**

使用了 CLIP、CLIP‑Dissect、Stable Diffusion 等视觉–语言模型进行概念匹配与图像生成，配合 ResNet‑18/50、ViT‑B/16 等网络进行实验。

**📊 数据集**

主要数据集包括 ImageNet‑1K、Places365、Broden、Common Words 以及遥感数据集 EuroSAT，用于探测、概念生成与验证。

**📈 对比分析**

与 Network Dissection、CLIP‑Dissect、FALCON、WWW、DnD 等基线相比，SIEVE 在所有评估指标（CLIP cos、MPNet cos、平均 Activation Rate）均高出 1.5 倍左右，尤其在平均 Activation Rate 方面显著提升。

**⚠️ 局限性**

局限在于对生成模型的依赖导致域迁移时效果下降，以及仅关注层级近前层，未能全面覆盖网络内部多层次语义。

---

## 172. The Anatomy of Uncertainty in LLMs

**arXiv ID:** 2603.24967 | [PDF](https://arxiv.org/pdf/2603.24967v1)

**作者:** Aditya Taparia `[一作]` (Arizona State University), Vivek Narayanaswamy `[通讯]` (Lawrence Livermore National Laboratory)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了LLM不确定性拆解框架，将整体不确定性拆分为输入歧义、知识缺口、解码随机三类，并通过语义熵量化；

**💡 创新点**

创新点在于用可解释的语义来源替代单一不确定性分数，提供针对性诊断与干预；验证了不确定性主导因素随模型规模和任务而转移；

**🔧 技术方法**

采用语义等价聚类（双向蕴含）、语义熵计算、LoRA适配器构建模型集合、不同解码策略（greedy、beam、temperature、top‑k/p）以及Rouge‑L、AUROC/ECE评估；

**📊 数据集**

使用TriviaQA（事实问答）和GSM8K（数学推理）两个数据集；

**📈 对比分析**

与传统单一不确定性方法对比，拆解后在TriviaQA上输入不确定性AUROC可达0.761、解码不确定性0.731；在GSM8K表现较弱。更大模型更依赖输入歧义，小模型更受解码随机影响；随机采样策略显著提升AUROC；

**⚠️ 局限性**

局限在于：使用有限的重述样本近似输入歧义、有限的LoRA模型集合近似后验、依赖语义聚类质量，未覆盖所有不确定性来源。

---

## 173. Can MLLMs Read Students' Minds? Unpacking Multimodal Error Analysis in Handwritten Math

**arXiv ID:** 2603.24961 | [PDF](https://arxiv.org/pdf/2603.24961v1)

**作者:** Dingjie Song `[一作]` (Lehigh University), Qingsong Wen `[通讯]` (Squirrel Ai Learning)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并评测了专门用于解释和分类学生手写数学解题错误的多模态基准。

**💡 创新点**

首次公开高质量的手写数学错题数据集与双任务（错误原因解释与分类）评估框架，填补了现有 MMLM 对真实手写解题过程诊断的空白。

**🔧 技术方法**

结合 LLM-as-a-Judge、视觉 OCR + LLM 协同推理，以及对 16 款多模态大模型的系统评测。

**📊 数据集**

1,720 份中国小学和初中学生手写数学解题样本，涵盖七类错误类型。

**📈 对比分析**

在 ECC 任务中，开源模型准确率低于 30%，专有模型最高达约 65%；在 ECE 任务中，最佳模型（o4-mini）得分约 71%，仍与人类约 89% 存在明显差距。

**⚠️ 局限性**

样本来源单一语言（中文）和单一平台，可能缺乏跨文化或跨语言的普适性。

---

## 174. DIET: Learning to Distill Dataset Continually for Recommender Systems

**arXiv ID:** 2603.24958 | [PDF](https://arxiv.org/pdf/2603.24958v1)

**作者:** Jiaqing Zhang `[一作]` (University of Science and Technology of China), Enhong Chen `[通讯]` (University of Science and Technology of China)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `8d10c613-917e-4880-9716-17789f50e119` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在推荐系统的持续学习框架下，提出DIET（Dataset distillation for the Incremental Evaluation of Training），实现了基于流式行为日志的压缩数据集生成与更新，能用极小的数据量逼近全量训练效果；

**💡 创新点**

①将数据集蒸馏从静态转为持续流式；②利用任务条件的影响评分（EL2N）和梯度相似度构建边界记忆；③通过持续记忆融合与影响导向的双向记忆寻址实现对训练动态的随时间演化；

**🔧 技术方法**

基于双层优化（内循环模拟梯度更新，外循环对蒸馏样本进行Meta梯度更新），使用EL2N、梯度相似度、随机截断BPTT、硬样本与活跃记忆选择、稀疏嵌入等技术；

**📊 数据集**

KuaiRand、Tmall、Taobao三大工业级稀疏多任务推荐数据集；

**📈 对比分析**

与全量训练、冷启动、warmup、随机抽样、EL2N、K‑Means等子集选择方法对比，DIET在三数据集上仅使用约1.5%原始交互即可接近全量AUC，跨模型架构表现优于基线，模型迭代成本下降约60倍；

**⚠️ 局限性**

仍需先训练参考模型，双层优化与内循环计算成本不可忽视；在极稠密场景可能出现“去噪”超越全量的情况但理论解释不足；在极低采样率下对罕见行为的保留有限；验证范围主要局限于工业推荐任务。

---

## 175. Sparton: Fast and Memory-Efficient Triton Kernel for Learned Sparse Retrieval

**arXiv ID:** 2603.25011 | [PDF](https://arxiv.org/pdf/2603.25011v1)

**作者:** Thong Nguyen `[一作]` (University of Amsterdam), Andrew Yates `[通讯]` (Johns Hopkins University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一种融合矩阵乘法、ReLU、log1p 与 max 操作的 Triton GPU 内核（Sparton），专门针对学习型稀疏检索（LSR）模型中的语言模型头，实现了显著的速度提升和显存降低。

**💡 创新点**

创新点包括：
1) 通过早期在线 max‑reduction 把 max 操作提前到 ReLU/ log1p 之前，减少了对完整 logit 矩阵的存储需求；
2) 将 GEMM 与后续的 element‑wise 以及最大池化一次性融合到同一核，消除了多次 HBM/ SRAM 之间的数据搬运；
3) 采用按 vocab 与 batch 分块的张量化策略，在保持 GEMM 高吞吐的同时，显著压缩内存占用；
4) 通过单一反向核同时完成梯度传播与稀疏激活的回传，进一步降低了前向状态存储与 I/O。

**🔧 技术方法**

技术手段包括 Triton 融合内核、CUDA 张量化矩阵乘法、内存层次优化（HBM ↔ SRAM）、稀疏激活剪枝、自动微分的自定义梯度实现，以及对词表大小、批量与序列长度的多维实验调优。

**📊 数据集**

使用的数据集主要是：
- 训练集：Mistral‑Splade（用于 InfoNCE 损失训练）
- 评估集：small‑Beir（Arq, FiQA, NFCorpus, SciDocs, SciFact）
- 还在多语言 Backbone（|𝒱|≈250k）上进行大规模实验。

**📈 对比分析**

对比方法：PyTorch 原生 eager 与 torch.compile 自动编译的 LM 头。实验结果显示：
- 单核速度提升至 4.8×；
- 峰值显存降至 12×；
- 训练批量可从 384 增至 512，训练速度提升 14%；
- 在多语言 Backbone 上可实现 26× 更大的批量和 2.5× 的训练加速；
- 这些提升均在不影响检索效果（NDCG@10）下实现。

**⚠️ 局限性**

局限性：
1) 当前实现只针对 LSR 模型的 LM 头，未对整体模型或其他检索框架进行深入集成；
2) 依赖 Triton 与 CUDA，须在支持这些工具链的 GPU 上部署；
3) 对极大词表（>250k）或极长序列（>8192）仍需进一步优化，可能需要利用 TMA、FP8 等硬件特性；
4) 反向梯度的稀疏处理虽然节省显存，但在多任务或多头注意力结构中可能需重新设计；
5) 目前未验证在多卡或分布式训练环境下的可扩展性。

---

## 176. A Systematic Empirical Study of Grokking: Depth, Architecture, Activation, and Regularization

**arXiv ID:** 2603.25009 | [PDF](https://arxiv.org/pdf/2603.25009v1)

**作者:** Shalima Binta Manir `[一作]` (University of Maryland, Baltimore County), Anamika Paul Rupa `[通讯]` (Howard University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过在模97加法和乘法任务上，系统地比较不同深度、网络结构、激活函数以及正则化强度对神经网络“grokking”现象的影响，验证了延迟泛化主要由优化稳定性与正则化相互作用决定，而非单纯的网络架构

**💡 创新点**

创新点在于首次通过对超参数、深度和架构进行严格匹配的对照实验，揭示了深度的非单调关系、Transformer与MLP的差异主要源于正则化选择，以及权重衰减在“grokking”中扮演的金发小调控角色，并进一步通过权重范数阈值和傅里叶稀疏性分析统一了多种机制

**🔧 技术方法**

实验采用全批梯度下降（SGD或AdamW）、权重衰减、残差连接、层归一化、不同激活（ReLU、GELU、Tanh），并结合权重范数跟踪、傅里叶频谱分析等技术

**📊 数据集**

使用的主要数据集为模97的加法和乘法（分别有9,409和9,216个样本），采用20%/80%训练/测试划分

**📈 对比分析**

通过多次随机种子（3–5种）在匹配的超参数下测量grokking延迟ΔT，结果显示在最佳权重衰减下，MLP平均延迟约为26,800步，Transformer约为50,800步，差距约为1.9×；在不同激活和深度下也给出了相应的延迟与方差；权重衰减的金发区间被精确定位

**⚠️ 局限性**

限制主要在于实验仅覆盖了模算术任务，未验证到更大规模或多样化数据集；不同架构需使用不同优化器和正则化，完全消除架构与优化器的耦合仍是挑战；实验时间和超参数空间有限，可能未捕捉更细微的动态

---

## 177. Few TensoRF: Enhance the Few-shot on Tensorial Radiance Fields

**arXiv ID:** 2603.25008 | [PDF](https://arxiv.org/pdf/2603.25008v1)

**作者:** Thanh-Hai Le `[一作]` (Saigon International University), Trong-Nghia Vu `[通讯]` (FPT University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 Few TensoRF，一个将 TensorRF 的高效张量表示与 FreeNeRF 的频率驱动少样本正则化相结合的 3D 重建框架，专门针对稀视角输入的重建场景。

**💡 创新点**

创新点在于：① 在 TensorRF 的密度与颜色张量上动态加入频率掩码，降低高频敏感度并强化低频结构；② 采用遮挡正则化，抑制稀视角下的浮点体和墙体伪影；③ 将这两种正则化无缝整合到 TensorRF 的训练流程中，兼顾速度与质量。

**🔧 技术方法**

技术细节包括：TensorRF 的 4D 张量表示与 VM 分解；FreeNeRF 的频率掩码与遮挡正则化；光线采样、视角编码、基于 PyTorch 的实现；Adam 优化器与 15000 次迭代的训练策略。

**📊 数据集**

使用的数据集为 Synthesis NeRF benchmark（用于少样本评估）和 THuman 2.0（用于人体三维重建测试）。

**📈 对比分析**

与 TensorRF 与 FreeNeRF 的基线方法在同等条件下进行对比，主要指标为 PSNR 与训练时间。Few TensoRF 在 Synthesis NeRF 上平均 PSNR 提升至 23.70 dB（细调版 24.52 dB），训练时间保持约 15 分钟；在 THuman 2.0 上仅用 8 张图像即可获得 27.37–34.00 dB 的 PSNR，明显优于同样稀视角下的 TensorRF。

**⚠️ 局限性**

局限性包括：对细节复杂的场景（如 Drums）仍表现不佳；在 THuman 2.0 的测试仅覆盖两个人体物体，缺乏全面基准；渲染结果在极端稀视角下仍会出现噪声和网格缺陷，需要进一步改进正则化策略和网格后处理。

---

## 178. Improving Fine-Grained Rice Leaf Disease Detection via Angular-Compactness Dual Loss Learning

**arXiv ID:** 2603.25006 | [PDF](https://arxiv.org/pdf/2603.25006v1)

**作者:** Md. Rokon Mia `[一作]` (Begum Rokeya University), B M Taslimul Haque `[通讯]` (Central Michigan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

提出了一个双损失框架，将ArcFace和Center Loss结合应用于稻叶病理的细粒分类任务。

**💡 创新点**

创新点在于通过同时使用角度边距约束和中心约束，显著提升特征嵌入的判别能力，而无需对网络结构做大幅改动。

**🔧 技术方法**

技术方案包括迁移学习（使用预训练的InceptionNetV3/DenseNet201/EfficientNetB0）、数据增强、ArcFace+Center Loss的联合训练以及AdamW优化器。

**📊 数据集**

使用了公开的Roboflow Rice Leaf Dataset（六类）以及Kaggle Rice Leaf Dataset进行实验。

**📈 对比分析**

通过与传统交叉熵损失以及其他SOTA模型（如RegNetY080、FVBC、DEX等）进行对比，InceptionNetV3+双损失实现了99.6%的准确率，较基线提升约0.4%–1.6%，并在精度、召回率、F1得分上均表现出色。

**⚠️ 局限性**

局限性在于实验仅覆盖两套数据集、病种相对有限，缺乏跨季节、跨源的泛化评估，以及未对边缘设备部署和模型可解释性进行进一步验证。

---

## 179. Relaxed Rigidity with Ray-based Grouping for Dynamic Gaussian Splatting

**arXiv ID:** 2603.24994 | [PDF](https://arxiv.org/pdf/2603.24994v1)

**作者:** Junoh Leea `[一作]`, Jin-Hwa Kim `[通讯]` (SNU)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32`

**🎯 论文内容**

本文提出在动态3D高斯分散中，通过射线聚类与松弛刚性约束来保持物理可行的运动；

**💡 创新点**

创新点在于基于视角射线的高斯聚类策略以及在聚类内施加运动一致性和谱正则化，摆脱了传统KNN聚类和光流先验；

**🔧 技术方法**

采用3DGS渲染管线、射线聚类、运动一致性正则、谱正则化以及Welford算法在线协方差计算等技术；

**📊 数据集**

使用D-NeRF、HyperNeRF和NeRF-DS三个动态场景数据集进行实验；

**📈 对比分析**

将方法集成到RTD、MoDec-GS、Grid4D和Ex4DGS四个基线模型中，评估PSNR/SSIM/LPIPS等指标，平均提升PSNR 1.19dB，Grid4D+Ours获得最高PSNR 42.20；

**⚠️ 局限性**

局限性包括训练时间提升约2-3倍，以及在极端动态或高反射场景中仍可能出现光照误差和拓扑变化问题。

---

## 180. Co-designing for the Triad: Design Considerations for Collaborative Decision-Making Technologies in Pediatric Chronic Care

**arXiv ID:** 2603.24993 | [PDF](https://arxiv.org/pdf/2603.24993v1)

**作者:** Ray-Yuan Chung `[一作]` (University of Washington), Ari Pollack `[通讯]` (Seattle Children’s Hospital)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在儿科慢性肾病护理中，研究者通过七次在线共创工作坊，邀请青少年患者、家长和医疗提供者三方参与，探讨并设计支持协同决策的数字技术。

**💡 创新点**

创新点在于将团队情境认知（Perception, Comprehension, Projection）框架应用于三方（患者、家长、医护）协同决策，并提出一套面向情境认知的设计原则，如支持会诊外的决策练习、通过可视化对齐心理模型、共享护理所有权以及提前预演潜在挑战。

**🔧 技术方法**

技术方法主要是人机交互领域的共创与迭代设计流程，包括情景式角色扮演、魔法工具（“Magical Tool”）构想、Affinity Diagram、主题分析以及对情境认知框架的映射。

**📊 数据集**

未使用传统数据集；研究数据来源于工作坊记录、故事板、设计原型及参与者的访谈记录，全部匿名化后进行主题分析。

**📈 对比分析**

研究未做量化对比或性能评估，主要以定性发现和设计建议为主；因此没有传统意义上的指标或性能比较。

**⚠️ 局限性**

局限性包括样本量小（19人），仅来自一所儿童医院，可能缺乏跨文化或跨地区的代表性；使用的共创方法易受参与者自我选择偏差影响；结果主要面向慢性肾病，泛化至其他慢性病需进一步验证。

---

## 181. LLM-Driven Reasoning for Constraint-Aware Feature Selection in Industrial Systems

**arXiv ID:** 2603.24979 | [PDF](https://arxiv.org/pdf/2603.24979v1)

**作者:** Yuhang Zhou `[一作]` (Meta AI), Jiayi Liu `[通讯]` (Meta AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于大语言模型的特征选择框架Model Feature Agent，通过顺序推理和元数据整合，在工业生产环境中实现可扩展、可约束的特征子集构建。

**💡 创新点**

创新点在于：①将特征选择视为分步推理过程，利用LLM的语义理解与量化元数据；②提出分治（Divide‑and‑Conquer）策略以突破上下文窗口限制；③在选择过程中显式加入容量、组别、公平等业务约束。

**🔧 技术方法**

主要技术包括大语言模型提示工程、结构化推理输出、分桶并行选择与全局精炼、以及基于特征元数据的约束引导。

**📊 数据集**

使用了三组工业数据：①用户兴趣/时效性调查数据（约70k样本、1,030个特征）；②推荐系统价值模型信号对（约400对交互特征）；③移动推送行为预测数据（8,169个特征，目标4,000特征）。

**📈 对比分析**

与Lasso回归或随机选特征的基线比较，在TI任务中在小特征预算下提升AUC约0.6%；在WT任务中差距低于0.5%；在交互特征选取中带来0.055%–0.048%的会话提升；在推送预测中相较生产模型和随机基线分别提升0.065%和0.198% NE损失，表明性能稳健且在操作成本上有显著优势。

**⚠️ 局限性**

局限性包括：①逐特征推理导致计算时间随选取特征数线性增长；②大量特征及元数据可能超出LLM上下文长度；③贪婪的单步搜索可能错过多特征协同带来的最佳组合。

---

## 182. From Stateless to Situated: Building a Psychological World for LLM-Based Emotional Support

**arXiv ID:** 2603.25031 | [PDF](https://arxiv.org/pdf/2603.25031v1)

**作者:** Boning Zhao `[一作]` (New York University), Xinnuo Li `[通讯]` (New York University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一种名为LEKIA 2.0的情绪支持LLM架构，核心在于将情境建模与干预执行分离，配合双门控机制实现过程控制。

**💡 创新点**

创新点在于引入可持续更新的外部情境结构和双门控执行层，避免传统LLM在多轮对话中的“无状态漂移”，并提出静态转动态的在线评估协议。

**🔧 技术方法**

技术上使用LangGraph构建状态图、Pydantic结构化输出、DeepSeek-V3作为基础模型、Qwen-Plus驱动的寻求者模拟器，以及CBT启发式四阶段状态机和双门控。

**📊 数据集**

数据集为扩充后的1,300条情绪支持案例（来自ESConv），并通过LLM模拟器构建动态交互环境，确保评估的可复用性。

**📈 对比分析**

与prompt‑only基线和中间baseline（阶段提示+冷却）比较，LEKIA在四个过程指标（评估、教育、干预、作业）上的平均完成率达0.895，较基线提升约31%，且在高抵抗测试中实现0%冷却违规率。

**⚠️ 局限性**

局限性包括评估主要依赖LLM-as-Judge，缺乏大规模真实用户验证；高抵抗测试样本不代表所有自然抵抗模式；模拟环境无法完全替代真实交互体验。

---

## 183. Optimal High-Probability Regret for Online Convex Optimization with Two-Point Bandit Feedback

**arXiv ID:** 2603.25029 | [PDF](https://arxiv.org/pdf/2603.25029v1)

**作者:** Haishan Ye `[一作]` `[通讯]` (Xi'an Jiaotong University), Haishan Ye (Xi'an Jiaotong University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

在两点带噪反馈下求解强凸在线凸优化问题，给出了高概率下的最优调度方法并证明了最小化回报的上界。

**💡 创新点**

首次实现了在两点反馈环境下对强凸损失函数的高概率调度回报上界，并将维度依赖从 O(d²) 降低到 O(d)，从而填补了期望式与高概率式结果之间的空白。

**🔧 技术方法**

采用了自适应梯度估计的两点梯度推导、Freedman 不等式的高概率收敛分析、以及对梯度方差的几何概率技巧，构建了新的直接分析框架。

**📊 数据集**

未使用任何实验数据集，全部为理论证明。

**📈 对比分析**

通过与之前仅在期望下给出 O(d² log T) 误差上界的工作比较，本文在相同环境下取得了 O(d log T) 的高概率上界，证明了方法在理论上更优。

**⚠️ 局限性**

主要限制是仍需要在每轮查询两次损失函数，且算法对 Lipschitz 与强凸参数的估计有一定依赖，若这些参数未知或不满足假设，则可能无法直接应用。

---

## 184. Hyena Operator for Fast Sequential Recommendation

**arXiv ID:** 2603.25027 | [PDF](https://arxiv.org/pdf/2603.25027v1)

**作者:** Jiahao Liu `[一作]` (Wuhan University of Technology), Jingling Yuan `[通讯]` (Wuhan University of Technology)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出HyenaRec，一种利用Legendre多项式参数化的Hyena算子与门控卷积相结合的顺序推荐模型；

**💡 创新点**

将Legendre正交多项式作为卷积核生成器，构造低维、稳定的长卷积滤波器，并结合门控机制捕捉短期行为，实现在长序列下子二次复杂度的高效建模；

**🔧 技术方法**

Hyena算子、Legendre多项式核参数化、门控卷积、子线性时间复杂度、交叉熵训练与Recall/NDCG评估；

**📊 数据集**

ML-1M、Steam、Video、XLong四个公开推荐数据集；

**📈 对比分析**

与GRU4Rec、NARM、SASRec、BERT4Rec、EulerFormer、HSTU、LRURec等主流基线对比，HyenaRec在所有数据集上均取得或逼近最高的NDCG/Recall，并且训练速度比基线快2-6倍，尤其在长序列XLong上提升约17%；

**⚠️ 局限性**

对极稀疏或极短序列的适用性不如传统注意力；需要调节多项式阶数K和滤波器顺序，参数选择对性能影响显著；

---

## 185. CARE: Training-Free Controllable Restoration for Medical Images via Dual-Latent Steering

**arXiv ID:** 2603.25026 | [PDF](https://arxiv.org/pdf/2603.25026v1)

**作者:** Xu Liu `[一作]` `[通讯]` (University of Washington), Xu Liu (University of Washington)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出 CARE，一种训练自由、可控的医疗图像恢复框架，利用双潜变量策略将数据保真与先验引导分离，并通过风险感知自适应控制器在推理时动态平衡两者。

**💡 创新点**

创新点在于：①双潜变量恢复架构，将保真分支和先验分支并行，显式区分结构保持与信息重建；②风险感知自适应控制器，根据局部可靠性与不确定性动态调节融合权重，实现无训练的可控恢复；③整体实现训练自由，能够在多种降噪/欠采样场景下灵活切换恢复模式。

**🔧 技术方法**

技术包括：扩散模型作为先验生成器、编码-解码双潜变量结构、风险感知自适应控制器（利用可靠性/不确定性评分和 sigmoid 调节），以及基于局部结构可靠性估计的动态融合策略。

**📊 数据集**

使用数据集：NIH‑AAPM‑Mayo Clinic Low Dose CT Grand Challenge（CT 降噪）和 fastMRI 2020 Brain Reconstruction Challenge（加速 MRI 重建）。

**📈 对比分析**

与 BM3D、REDCNN、EDCNN、U‑Net、CTformer、VarNet、XPDNet 等基线对比。CT 任务中，CARE 取得最高 PSNR 33.62 dB（高于 U‑Net 33.07 dB）且 SSIM 0.9004；MRI 任务中，4X SSIM 0.964、8X SSIM 0.955，均与最优基线相当或优于。读者研究中，CARE 在 Artifact Reduction 与 Contrast Retention 评分均名列前茅。控制模式对比表明，平衡模式在 PSNR、结构评分与幻影风险之间实现最佳折衷。

**⚠️ 局限性**

局限性包括：①先验模型需与目标临床域匹配，若不匹配可能导致恢复质量与安全性下降；②尽管自适应控制降低幻影风险，但仍无法完全消除不可信重建，临床使用仍需人工审核；③缺乏针对不同模态的专门验证，未涵盖更多读者研究与下游任务评估；④对极端降噪/欠采样场景的鲁棒性与可解释性需进一步提升。

---

## 186. A Public Theory of Distillation Resistance via Constraint-Coupled Reasoning Architectures

**arXiv ID:** 2603.25022 | [PDF](https://arxiv.org/pdf/2603.25022v1)

**作者:** Peng Wei `[一作]` (Southwest University), Wesley Shu `[通讯]` (Institute of Energetic Paradigm)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119`

**🎯 论文内容**

提出了一种公开、无商业机密的理论框架，旨在通过在模型内部嵌入稳定性约束来降低知识蒸馏的有效性，减少能力与治理之间的不对称。

**💡 创新点**

创新点在于构建“约束耦合推理”四要素（有限转移负荷、路径负荷累积、动态可行域、能力-稳定性耦合条件），将模型的高性能与内部状态演化的稳定约束紧密绑定，形成新的蒸馏抵抗机制。

**🔧 技术方法**

主要采用了数学抽象方法：定义负荷函数、可行域收缩算子、路径负荷累积、以及以这些元素为基础的训练损失结构；未给出具体实现细节或优化算法。

**📊 数据集**

未使用具体数据集；论文仅以“同一任务分布”作为假设性比较基准，未给出具体数据来源。

**📈 对比分析**

通过比较从基线教师和约束耦合教师蒸馏得到的学生模型，在相同训练预算和评估条件下观察能力与稳定性权衡；但目前尚无实验结果，提出若干可检验假设以供后续实验验证。

**⚠️ 局限性**

局限性包括：未完成实证验证、需要可测量的负荷与稳定性指标、并非所有提取攻击被阻断、可能导致额外训练/部署成本，以及不能替代机构层面的治理与监管。

---

## 187. Mechanistically Interpreting Compression in Vision-Language Models

**arXiv ID:** 2603.25035 | [PDF](https://arxiv.org/pdf/2603.25035v1)

**作者:** Veeraraju Elluru `[一作]` (Indian Institute of Technology Jodhpur), Hreetam Paul `[通讯]` (Independent Researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `fede83ac-7505-405f-ab37-e7284695c47f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对裁剪和量化等压缩方法在视觉‑语言模型中的机制影响进行研究，并提出新的安全基准 VLMSafe‑420。

**💡 创新点**

创新点在于结合因果电路分析与交叉编码器特征比较两种机制方法，揭示压缩导致特征旋转、衰减和新电路产生，并量化其对安全行为的影响。

**🔧 技术方法**

主要技术包括 Edge Activation Patching、Crosscoder 稀疏自编码器特征分析、logit‑lens 以及基于 Claude Haiku 的判定器评估。

**📊 数据集**

使用 BLIP‑VQA、LLaVA、Qwen3‑VL 以及自制 VLMSafe‑420 420 样本安全对照集。

**📈 对比分析**

通过 Jaccard/Spearman、Feature Sharing Ratio 等指标对比未压缩模型，发现 Wanda 剪枝保留更多电路但特征被旋转；量化更易产生新电路；在安全测试中量化保持 33.9% 真正拒绝率，而剪枝在 50% 稀疏时仅 22%。

**⚠️ 局限性**

局限性包括只评估小至中型（≤8B）模型、压缩仅对视觉和桥接模块、缺乏大规模数据以及 VLMSafe‑420 仅覆盖 38 类安全标签。

---

## 188. Imperative Interference: Social Register Shapes Instruction Topology in Large Language Models

**arXiv ID:** 2603.25015 | [PDF](https://arxiv.org/pdf/2603.25015v1)

**作者:** Tony Mason `[一作]` `[通讯]` (University of British Columbia), Tony Mason (University of British Columbia)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对 Claude Code 系统提示进行 22 个可剥离块的指令级消融实验，探究不同语言（英、西、法、汉）下指令交互拓扑的变化；

**💡 创新点**

发现指令交互拓扑在语言之间翻转（英语协作，西班牙竞争），并证明社会注册（命令语气）是驱动因素，利用声明式重写显著降低跨语言方差并产生溢出效应；

**🔧 技术方法**

采用覆盖数组消融、配对消融、Wilcoxon/贝叶斯检验、Permutation 测试、声明式与过程式编码比较、机器翻译、LLM 评估器等技术；

**📊 数据集**

使用 Claude Code v2.1.50 系统提示的 56 块语义拆分，翻译成 4 种语言，22 个手工构造的评估探针；

**📈 对比分析**

通过跨语言基线对比、配对消融的平均 Δ、方差降低率（81%）等指标评估模型表现，显示英语表现最好，西班牙表现差且存在竞争；

**⚠️ 局限性**

仅测试单一系统提示、机器翻译质量未验证、部分探针使用同一 LLM 评估、模型样本有限、未测试双语言交互、重写手工完成等限制。

---

## 189. Wireless bioelectronics for untethered biohybrid robots

**arXiv ID:** 2603.24959 | [PDF](https://arxiv.org/pdf/2603.24959v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 190. IrisFP: Adversarial-Example-based Model Fingerprinting with Enhanced Uniqueness and Robustness

**arXiv ID:** 2603.24996 | [PDF](https://arxiv.org/pdf/2603.24996v1)

**作者:** Ziye Geng `[一作]` (University of Houston), Changqing Luo `[通讯]` (University of Houston)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了IrisFP模型指纹框架，通过多边界特征、复合样本指纹和指纹质量评估实现模型唯一性与鲁棒性的提升，并完成了版权验证流程。

**💡 创新点**

创新点包括①将指纹置于所有决策边界交叉点以提升预测间距；②通过细微多样化扰动生成复合样本指纹，增强唯一性；③利用统计可分离度筛选指纹并为每个指纹自适应设定阈值。

**🔧 技术方法**

使用技术包括对抗样本生成、KL散度优化、Cohen’s d效应量评估、指纹自适应阈值、两步所有权验证（匹配率与决策聚合）等。

**📊 数据集**

实验数据集涵盖CIFAR-10、CIFAR-100、Fashion-MNIST、MNIST、Tiny-ImageNet，并使用ResNet-18、MobileNet‑V2、ViT‑B/16等模型。

**📈 对比分析**

与IPGuard、UAP、ADV-TRA、AKH四种基线在六类模型修改攻击下进行AUC评估，IrisFP在所有数据集与攻击场景中均获得最高AUC，并在唯一性与鲁棒性曲线上优于基线。

**⚠️ 局限性**

局限性在于对更大规模模型或不同任务（如目标检测、语音）适用性未充分验证，且对抗样本生成和指纹筛选过程计算开销相对较高。

---

## 191. LiteGuard: Efficient Task-Agnostic Model Fingerprinting with Enhanced Generalization

**arXiv ID:** 2603.24982 | [PDF](https://arxiv.org/pdf/2603.24982v1)

**作者:** Guang Yang `[一作]` (Virginia Commonwealth University), Changqing Luo `[通讯]` (University of Houston)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9cc9baba-5356-466d-81ff-d80028d90279` `3855fcda-48ef-4070-a15e-803cd5c84d83` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本研究提出 LiteGuard，一种任务无关的模型指纹识别框架；

**💡 创新点**

主要创新点在于使用训练过程中的检查点进行模型集增强，以及为每个指纹配备轻量级本地验证器，降低参数耦合；

**🔧 技术方法**

采用检查点采样、指纹-验证器对的联合训练、二元交叉熵损失、Adam 优化器等技术；

**📊 数据集**

实验使用 CIFAR-100、CASP、California Housing、QM9、Weather 等数据集；

**📈 对比分析**

与 MetaV 及多种任务特定方法对比，LiteGuard 在 AUC 上显著提升（如图像分类 0.936 对 0.676，GNN 0.803 对 0.598 等），同时显著降低训练模型数量与计算成本；

**⚠️ 局限性**

局限在于目前仅在少数任务上验证，缺乏对极大模型或更复杂攻击手段的评估，且对检查点的依赖可能导致存储成本上升。

---

## 192. Self-Corrected Image Generation with Explainable Latent Rewards

**arXiv ID:** 2603.24965 | [PDF](https://arxiv.org/pdf/2603.24965v1)

**作者:** Yinyi Luo `[一作]` (Carnegie Mellon University), Shengfeng He `[通讯]` (Singapore Management University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出xLARD框架，利用可解释潜在奖励在潜在空间中进行自我纠错，提升文本到图像生成的语义一致性。

**💡 创新点**

创新点在于将图像级非可微评估映射到潜在空间的可微奖励投影，并通过轻量级残差修正器在生成过程中实时纠正，整个流程保持模型冻结且高度可解释。

**🔧 技术方法**

使用多模态大型语言模型进行评估，PPO强化学习更新修正器，潜在奖励投影器将图像奖励映射到潜在空间，结合计数、颜色、空间等可解释子奖励实现自我指导。

**📊 数据集**

在GenEval、DPG‑Bench、ImgEdit、GEdit等公开基准数据集上进行实验。

**📈 对比分析**

与OmniGen、Show‑O、Bagel、UniWorld、HermesFlow等多种基线对比，xLARD在GenEval上提升约+4.1%，DPG‑Bench提升+2.97%，编辑任务亦取得更高总体分，整体性能优于或持平于现有后训练方法。

**⚠️ 局限性**

局限性：奖励设计依赖特定语义维度，可能无法覆盖美学或文化细微差异；仅在英文提示和固定基准上验证，跨语言、动态或更复杂任务的泛化尚待进一步研究。

---

## 193. Toward domain-specific machine translation and quality estimation systems

**arXiv ID:** 2603.24955 | [PDF](https://arxiv.org/pdf/2603.24955v1)

**作者:** Javad Pourmostafa Roshan Sharami `[一作]` `[通讯]`, Javad Pourmostafa Roshan Sharami

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `67630363-6be0-4f51-ab05-7198250671a5` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究如何在专业领域提升机器翻译与质量估计的准确性、适应性与计算效率，提出四种针对数据选择、训练管线、子词/词表联合分析以及QE指导的ICL等方法。

**💡 创新点**

创新点包括基于相似度的高质量数据挑选、分阶段域适配与轻量化增量训练、系统化子词标记化与词表来源的联合分析，以及QE驱动的无参数ICL框架。

**🔧 技术方法**

使用技术包括相似度过滤、域标签、合成数据生成、跨语言混合训练、子词标记化、词表对齐、基于QE评分的检索/选择以及LLM提示式ICL。

**📊 数据集**

实验基于通用大规模语料（如WMT、OpenSubtitles等）和多专业领域语料（医疗、法律、IT等），并利用人工标注和合成句对进行评估。

**📈 对比分析**

通过BLEU、COMET等指标与随机、n-gram、BM25检索等基线对比，实验显示在低计算成本下实现了数点BLEU/COMET提升，且小规模高质量数据可替代大规模通用数据。

**⚠️ 局限性**

局限性在于对足够的领域标注或合成资源有依赖，方法在极低资源或未覆盖领域的适用性有限，且多步骤管线的复杂性与域标签可迁移性仍需进一步研究。

---

## 194. AirSplat: Alignment and Rating for Robust Feed-Forward 3D Gaussian Splatting

**arXiv ID:** 2603.25129 | [PDF](https://arxiv.org/pdf/2603.25129v1)

**作者:** Minh-Quan Viet Bui `[一作]` (Korea Advanced Institute Of Science And Technology), Munchurl Kim `[通讯]` (Korea Advanced Institute Of Science And Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

对3D视觉基础模型进行训练，得到无姿态输入下的高质量新视图合成系统

**💡 创新点**

提出自洽姿态对齐（SCPA）解决姿态与几何不一致，和基于教师评价的半透明匹配（ROM）剔除浮点体

**🔧 技术方法**

采用SCPA自回馈姿态校正、ROM基于稀视角NVS教师的几何评分、固定主编码器只微调高斯头

**📊 数据集**

在RealEstate10K、DL3DV和ACID三大规模数据集上进行训练与评测

**📈 对比分析**

与现有无姿态与有姿态NVS方法对比，显著提升PSNR/SSIM/LPIPS，尤其在低视角下超越多有姿态SOTA

**⚠️ 局限性**

仍受限于教师模型的稀视角质量，且在极端大视角间隔或极端遮挡场景中效果下降

---

## 195. From Logic Monopoly to Social Contract: Separation of Power and the Institutional Foundations for Autonomous Agent Economies

**arXiv ID:** 2603.25100 | [PDF](https://arxiv.org/pdf/2603.25100v1)

**作者:** Anbang Ruan `[一作]` `[通讯]` (NetX Foundation), Anbang Ruan (NetX Foundation)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出Agent Enterprise for Enterprise（AE4E）范式，并通过NetX Enterprise Framework（NEF）实现基于区块链的三权分立治理模型，解决多智能体系统的可靠性缺口。

**💡 创新点**

创新点在于将立法、执行与审判三权分立与智能体治理相结合，构建可扩展的Agent Enterprise Economy，并首次将Parsons的AGIL框架嵌入智能体社会层，形成完整的机构治理结构。

**🔧 技术方法**

技术手段包括智能合约、TEE受信计算、去中心化身份识别（DID）、加密经济机制、链上审计日志以及基于Parsons AGIL的治理模型。

**📊 数据集**

实验使用Agents of Chaos、La Serenissima、ASB等公开数据集评估安全性、可追溯性和资源消耗，验证框架的实用性。

**📈 对比分析**

与现有多智能体框架（如LangGraph、AutoGen、MetaGPT）的对比显示，AE4E将攻击成功率从约80%降低至约15%，并将平均资源消耗降低30%，同时保持可扩展性。

**⚠️ 局限性**

局限性包括对区块链性能的依赖、跨链互操作的复杂性，以及在大规模部署时对TEE硬件资源的高需求。

---

## 196. Large Language Models as Optimization Controllers: Adaptive Continuation for SIMP Topology Optimization

**arXiv ID:** 2603.25099 | [PDF](https://arxiv.org/pdf/2603.25099v1)

**作者:** Shaoliang Yang `[一作]` (Santa Clara University), Yunsheng Wang `[通讯]` (Santa Clara University)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

使用大语言模型（LLM）作为在线自适应控制器，实时根据SIMP拓扑优化过程中的状态（合规性、灰度指数等）决定四个关键参数（惩罚指数 p、Heaviside 尖锐度 β、过滤半径 r_min、移动限制 δ），替代传统固定时间表实现闭环控制，并通过 Meta‑优化循环自适应调整控制器自身超参数。

**💡 创新点**

创新点包括：① 直接数值控制（Direct Numeric Control）实现LLM直接输出连续参数；② 在控制中加入灰度门和安全约束，确保不会在灰度仍高时过早锐化；③ 双层 Meta‑优化实现全流程无训练、零调优；④ 通过对比实验证明LLM在多维度、不同网格分辨率下均优于固定、专家手工和仅时间表的传统策略。

**🔧 技术方法**

核心技术：三场 SIMP 拓扑优化、Heaviside 投影与密度过滤、OC 更新法、Gemini-3.1-flash-lite LLM 的系统提示与自然语言推理、直接数值输出接口、回退/折算调度、Meta‑优化循环、标准化 40 迭代锐化尾部的实验设计。

**📊 数据集**

数据集：三种经典二维基准（cantilever, MBB beam, L‑bracket）采用 120×60 网格，两个三维基准（cantilever, MBB beam）采用 40×20×10 网格；每个基准使用 5 个独立随机种子进行实验。

**📈 对比分析**

比较方法：在所有控制器（固定、三场连续、专家手工、仅时间表、LLM）执行相同 300 迭代主循环后统一应用 40 迭代锐化尾部；评估指标为最终合规性、灰度指数、运行时间和最佳快照迭代。结果显示 LLM 在所有问题上获得最低合规性，提升幅度从 5.7%（二维）到 18.1%（三维）相对固定基准，并优于专家手工约 0.1%–1.9%；所有控制器最终均实现完全二进制拓扑；运行时间略高（≈32–42%），API 成本低于 1 美元。

**⚠️ 局限性**

局限性：① 依赖特定 LLM 版本和系统提示，需固定模型与提示保证可复现；② 仅验证单负荷合规最小化问题，未扩展到多负荷、应力约束或制造约束；③ Meta‑优化循环与在线决策的贡献未完全分离；④ 需要更多实验验证模型泛化到其他前沿 LLM 或开放模型；⑤ 目前控制器在三维大规模问题中运行时间略高，仍有优化空间。

---

## 197. Bounded Independence Edge Sampling for Combinatorial Graph Properties

**arXiv ID:** 2603.25095 | [PDF](https://arxiv.org/pdf/2603.25095v1)

**作者:** Aaron Putterman `[一作]` (Harvard University), Vadim Zaripov `[通讯]` (Harvard University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文通过使用有限相互独立（k‑wise independent）抽样，在图的边子样本中实现可判定的连通性与无环性，进而为图形和共图形基的显式并行求解提供了确定性算法。

**💡 创新点**

创新点在于：①证明在边数为 m、最小割≥κlog m 或最小环长≥κlog m 的图中，O(log m) 维相互独立抽样即可保持连通性或无环性；②建立了边权重与最小割、有效电阻直径之间的新等价关系，利用该关系构造可控的权重并得到小勒维得分；③基于上述采样理论，设计了一套通用框架，显式地将随机图抽样转化为确定性并行图形/共图形基求解，取得 O(log m loglog m) 轮次、多项式查询的显式算法。

**🔧 技术方法**

主要技术包括：有限相互独立随机变量构造、有效电阻与勒维得分分析、图的分层划分与递归加权、并行基搜索框架（删除循环与大独立集的递归），以及对图的连通性与无环性通过谱稀疏化的精细概率估计。

**📊 数据集**

本文不依赖实验数据，全部为理论证明与算法设计，不使用真实数据集。

**📈 对比分析**

与先前的随机化并行算法（如 Khanna–Putterman–Song 2025 的 O(log m) 轮次随机算法）相比，本文提供了可实现的显式确定性版本，且在轮次数与查询数上仅增加了常数因子 loglog m，保持了与最优随机方案相近的性能。

**⚠️ 局限性**

限制主要体现在：①对 k 的取值仍需为 O(log m)，构造相互独立样本的空间与时间复杂度相对较高；②目前仅适用于图形与共图形基的求解，无法直接推广到所有稀疏或一般 matroid；③在实际实现中，权重分配与有效电阻计算可能导致较大的常数因子。

---

## 198. Pixelis: Reasoning in Pixels, from Seeing to Acting

**arXiv ID:** 2603.25091 | [PDF](https://arxiv.org/pdf/2603.25091v1)

**作者:** Yunpeng Zhou `[一作]` `[通讯]` (University of Reading), Yunpeng Zhou (University of Reading)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

设计并训练了Pixelis，一种在像素空间直接执行可验证工具链的视觉语言代理，能够在推理过程中通过分割、跟踪、OCR等像素级操作进行行动，并在测试时安全自适应；

**💡 创新点**

①将工具设计为可执行的像素级操作，并直接在像素空间构建推理链；②提出三阶段训练流程——监督微调学习工具语法、好奇-一致性RFT引导结构化探索、像素级测试时RL通过轨迹投票实现安全自适应；③使用KL‑EMA安全门控保证在线更新的稳定性；

**🔧 技术方法**

使用Qwen3‑VL‑8B‑Instruct视觉语言模型，配合可执行工具集{SEG、ZOOM、TRK、OCR、TEMP、PROP}；Chain‑of‑Thought‑Action (CoTA) 轨迹；SFT的mask imitation loss；好奇-一致性RFT（GRPO、adjacent‑step coherence、curiosity reward、KL penalty）；Pixel TTRL（检索+投票、KL/EMA安全约束、行为相似度、伪参考校准）；RaPR/RaCPR过程评估指标；多任务评估与对比。

**📊 数据集**

公共图像/视频基准：V*Bench、MMBench v1.1、MVBench、InfoVQA、Video‑MMMU、VSI‑Bench；CoTA轨迹收集自Grok 4 VLM；训练集约80k图像+28k视频，验证集8k图像+2k视频，测试集8k图像+2k视频。

**📈 对比分析**

与相同8B基础模型（无工具）、仅自一致性、VLM测试时适应等方法进行对比，采用token‑KL约束在0.15 corridor；Pixelis在六个基准上平均提升+4.08%相对基线，VSI‑Bench最高+6.03%；链长度从≈6缩短到3.7；RaPR/RaCPR显著提升，工具精度提升；安全门控保证token‑KL≤0.2。

**⚠️ 局限性**

①SFT阶段残留偏差可能导致下游自适应不足；②Pixel TTRL在突变域面临震荡或适配不足；③一致性仅局部，无法捕捉长程依赖；④工具（分割、OCR、跟踪）在稀疏结构、模糊、复杂字体等场景易失效，误差累积；⑤需要大量计算资源与工具链实现；⑥KL‑EMA安全门限需手动调节。

---

## 199. Visual Attention Drifts,but Anchors Hold:Mitigating Hallucination in Multimodal Large Language Models via Cross-Layer Visual Anchors

**arXiv ID:** 2603.25088 | [PDF](https://arxiv.org/pdf/2603.25088v1)

**作者:** Chengxu Yang `[一作]` (Wuhan University of Technology), Jiawei Jiang `[通讯]` (Wuhan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种训练-free的CLVA方法，通过跨层视觉锚点纠正多模LLM中的注意力漂移，降低幻觉。

**💡 创新点**

创新点在于利用视觉敏感/不敏感注意力头区分，提取中层正向锚点和初层负向锚点，随后在深层对注意力进行重锚，从根本上解决深层注意力回退至噪声的问题。

**🔧 技术方法**

技术包括跨层视觉锚点提取、注意力重锚机制、基于均值标准差的头筛选、Z-score阈值掩码，以及对比锚点调节。

**📊 数据集**

使用的数据集包括COCO、MSCOCO、AOKVQA、GQA、POPE、CHAIR和MME。

**📈 对比分析**

与IMCCD、MemVR、ClearSight等训练-free方法在POPE、CHAIR、MME等基准上对比，CLVA在准确率/F1上提升约1%–7%，显著降低幻觉率，且对推理速度和显存影响极小。

**⚠️ 局限性**

局限性在于对极端视觉场景的鲁棒性有限，需要手动设定锚层，对不同模型层数的适配不够自适应，且未探索结合训练时改进的可能性。

---

## 200. Z-Erase: Enabling Concept Erasure in Single-Stream Diffusion Transformers

**arXiv ID:** 2603.25074 | [PDF](https://arxiv.org/pdf/2603.25074v1)

**作者:** Nanxiang Jiang `[一作]` (Beihang University), Wenjun Wu `[通讯]` (Beihang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了一种针对单流扩散变换器的概念消除方法Z-Erase，使模型在不破坏图像生成的前提下去除指定概念。

**💡 创新点**

创新点包括：①设计了流解耦概念消除框架，解耦视觉与文本参数更新；②提出拉格朗日引导自适应消除调制算法，在保持约束的同时动态平衡消除与保留。

**🔧 技术方法**

采用了低秩适配（LoRA）、流匹配训练、自注意力分析以及拉格朗日约束优化等技术。

**📊 数据集**

主要在Z-Image Turbo模型上评测，使用I2P（nudity/violence）、MS-COCO 10K、CelebA、Ring-A-Bell、UnlearnDiffAtk等数据集。

**📈 对比分析**

与AC、ESD、MACE、UCE、EraseAnything等基线对比，Z-Erase在NSFW、名人、艺术风格等任务中实现最高的消除效果，同时保持FID/CLIP不明显下降；在攻击鲁棒性测试中表现最稳健。

**⚠️ 局限性**

局限性在于仅验证单流模型，极端多概念消除仍有挑战；需要额外的结构改造与超参调优，训练成本相对较高。

---

## 201. The System Prompt Is the Attack Surface: How LLM Agent Configuration Shapes Security and Creates Exploitable Vulnerabilities

**arXiv ID:** 2603.25056 | [PDF](https://arxiv.org/pdf/2603.25056v1)

**作者:** Ron Litvak `[一作]` `[通讯]` (Columbia University), Ron Litvak (Columbia University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并评估了 PhishNChips 基准，对 11 种 LLM 进行 220k 评估，探讨系统提示与模型配置对钓鱼检测安全性的影响。

**💡 创新点**

首次将系统提示视为主要安全变量，提出信号化提示优化和“指令特异性悖论”，并设计 Safetility 指标衡量部署可行性。

**🔧 技术方法**

采用系统提示工程、信号化提示、因子实验、回溯分析、基础设施钓鱼对抗实验等技术。

**📊 数据集**

使用 2000 封合成邮件（1000 真实钓鱼 URL + 1000 合法邮件）及 73 封基础设施钓鱼样本和 100 封交叉域合法样本。

**📈 对比分析**

以召回率、误报率、净效能和 Safetility 为指标比较，优化提示可达 93.7% 召回率、3.8% 误报率（Safetility≈87%），但在基础设施钓鱼上召回率大幅下降。

**⚠️ 局限性**

局限包括合成数据、均衡样本分布、缺少完整邮件元数据、交叉域误报未充分评估、基础设施样本量有限、预部署倾向分类可能不普适、结果随模型版本变化不确定。

---

## 202. GaussFusion: Improving 3D Reconstruction in the Wild with A Geometry-Informed Video Generator

**arXiv ID:** 2603.25053 | [PDF](https://arxiv.org/pdf/2603.25053v1)

**作者:** Liyuan Zhu `[一作]` (Stanford University), Iro Armeni `[通讯]` (Stanford University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `40105733-5154-44cd-8090-a8cab9e64b07` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了GaussFusion，一种基于几何信息引导的视频生成框架，通过GP-Buffer对3D Gaussian splatting结果进行无瑕重建，并进一步优化3D重建。

**💡 创新点**

首次将颜色、深度、法线、透明度和协方差等多模态几何信息编码为GP-Buffer，并在流匹配生成器中引入Geometry Adapter实现几何条件化；同时构建全面的伪造数据生成管线提升泛化能力。

**🔧 技术方法**

采用流匹配（Flow Matching）的视频生成器 Wan-2.1-1.3B，配合GP-Buffer、Geometry Adapter、DMD蒸馏得到四步高效模型，并结合3D Gaussian splatting的优化与更新策略。

**📊 数据集**

使用来自DL3DV和RE10K的75K+对齐视频数据，涵盖稀疏视角、不同初始化及feed‑forward生成的失真；评测同两大基准数据集。

**📈 对比分析**

与DiFiX3D+、GenFusion、ExploreGS、MVSplat360等公开基线在DL3DV、RE10K上进行对比，采用PSNR、SSIM、LPIPS、FID等指标；GaussFusion在PSNR/SSIM上显著领先，实时版本16FPS且质量保持接近全步模型。

**⚠️ 局限性**

在极大规模场景下GP-Buffer的存储/渲染开销较大；实时版对高频细节略有损失；对极端光照或遮挡仍可能留下残余误差，需进一步完善几何表达与鲁棒性。

---

## 203. AutoPDR: Circuit-Aware Solver Configuration Prediction for Hardware Model Checking

**arXiv ID:** 2603.25048 | [PDF](https://arxiv.org/pdf/2603.25048v1)

**作者:** Guangyu Hu `[一作]` (Hong Kong University of Science and Technology), Hongce Zhang `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种基于电路结构的自动化PDR参数配置框架，通过图学习预测最优参数组合以提升硬件模型检查效率。

**💡 创新点**

创新点在于①利用专家规则对PDR参数空间进行约束过滤，将搜索空间压缩78%；②设计多层静态特征提取与图表示并行的特征提取管线；③在GNN中引入排名感知混合损失函数，直接优化配置排名。

**🔧 技术方法**

使用的技术包括图神经网络（GraphSAGE为最佳）、AIG图的COI预处理、统计特征归一化、基于规则的参数过滤以及混合回归-排名损失。

**📊 数据集**

实验数据集来源于Hardware Model Checking Competition（HWMCC）2007–2024的192个未见电路和属性实例，约5%困难案例用于训练。

**📈 对比分析**

与默认PDR配置和常用优化设置（pdr‑nct）对比，Top‑5预测配置能在192个实例中解决152例（79.2%），比默认提升90.0%，表现出显著的性能提升。

**⚠️ 局限性**

局限性包括仅覆盖9个参数的离散组合，缺乏动态/在线调优；对极大电路的图规模仍有限；以及模型对超出训练分布的电路结构可能泛化不足。

---

## 204. The Order Is The Message

**arXiv ID:** 2603.25047 | [PDF](https://arxiv.org/pdf/2603.25047v1)

**作者:** Jordan LeDoux `[一作]` `[通讯]` (Independent Researcher), Jordan LeDoux (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

研究梯度下降训练中的数据排序对模型学习的影响，提出并验证了“排序通道”概念，即顺序信息通过Hessian‑梯度耦合在梯度中产生可利用的信号。

**💡 创新点**

创新点在于：①首次将排序视为信息通道而非噪声；②通过反事实梯度分解直接测量排序与内容两种梯度成分；③揭示排序在大多数训练轮中占梯度能量的85%以上；④在模数算术任务中证明排序能决定模型是否能快速泛化，并揭示不同排序策略（随机、固定、按输出排序、按行走步长）对应的构造性、无序性与破坏性三种模式。

**🔧 技术方法**

主要技术包括：梯度反事实分解、Hessian‑梯度耦合分析、频谱分析（Fourier特征）、AdamW优化器的自适应梯度变换、统计量（梯度范数、相位相似度、谱熵）以及对比实验。

**📊 数据集**

使用的主要数据集为模数算术任务（p = 9973，约0.3%训练样本），并在p = 97实验中验证排序效应；模型为两层预层归一化Transformer，采用AdamW优化器，cosine学习率调度。

**📈 对比分析**

比较方法：在相同模型、相同数据、相同超参下仅改变排序策略，记录训练准确率、测试准确率、梯度分量比例、频谱浓度等指标。结果显示：固定步长排序在487轮即可达到99.5%测试精度，随机打乱则需5000轮；按输出排序导致模型停滞；排序通道的排序成分占梯度能量的83–89%且对不同策略的最终性能决定性强。

**⚠️ 局限性**

局限性包括：①实验仅在高度结构化的模数算术任务上验证，尚未证明在自然语言或图像等更复杂任务中的可迁移性；②需对任务有先验结构才能设计有效排序；③对优化器的具体实现（如Adam）有一定依赖；④对梯度分解的统计估计依赖于多次随机打乱，计算成本高；⑤安全风险讨论为初步，缺乏针对实际部署场景的评估。

---

## 205. ThermoAct:Thermal-Aware Vision-Language-Action Models for Robotic Perception and Decision-Making

**arXiv ID:** 2603.25044 | [PDF](https://arxiv.org/pdf/2603.25044v1)

**作者:** Young-Chae Son `[一作]`, Soo-Chul Lim `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出ThermoAct框架，将热成像融入Vision‑Language‑Action体系，实现机器人在执行任务时具备温度感知与安全决策能力。

**💡 创新点**

创新点在于通过层次化规划：利用Vision‑Language模型(VLM)先对指令进行语义分解，再让Vision‑Language‑Action模型(VLA)执行子任务，解决热数据稀缺导致的端到端学习困难，并首次将热成像数据与VLM/VLA结合。

**🔧 技术方法**

采用Gemini 2.0 Flash作为高层VLM规划器；π_0类VLA模型并用LoRA微调；将RGB图像与热图像（通过线性归一化、伪彩色映射后转为RGB）作为输入；实验平台为7‑DoF Kinova Gen3 Lite。

**📊 数据集**

使用自己构建的含有RGB、深度和热图像的演示数据集，约每任务50条演示，涵盖5个日常与安全场景任务；热图像以256×192尺寸采集后转换为RGB伪彩色图。

**📈 对比分析**

对比RGB‑RGB基线、RGB‑T模型以及单体平面VLA，ThermoAct在5个任务中平均成功率约83%，比RGB‑RGB提升约40%；层次化VLM‑VLA组合显著优于端到端flat VLA（后者几乎零成功率）。

**⚠️ 局限性**

局限性包括热相机缺乏深度信息导致高度估计不准，手腕摄像头视野受限导致探测不完整；数据集规模有限，未覆盖更广泛的物体与温度分布；对动态与低视角场景的鲁棒性仍待提升。

---

## 206. Intern-S1-Pro: Scientific Multimodal Foundation Model at Trillion Scale

**arXiv ID:** 2603.25040 | [PDF](https://arxiv.org/pdf/2603.25040v1)

**作者:** Yicheng Zou `[一作]` (Shanghai AI Laboratory), Lei Bai `[通讯]` (Shanghai AI Laboratory)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了 Intern‑S1‑Pro，一款 1 万亿参数的科学多模态基础模型，结合专家扩展与分组路由，实现了大规模 MoE 的高效训练。

**💡 创新点**

创新点包括：分组路由实现设备级负载均衡、Straight‑Through Estimator 让稀疏路由可微、FoPE 通过傅里叶位置编码捕获连续物理特征、专门的时间序列编码模块，以及面向科学图像的高质量标题生成管道。

**🔧 技术方法**

核心技术：MoE + 分组路由 + STE；多模态 Encoder（ViT、时间序列 Encoder）；混合精度 RL（FP8 量化 + 重要性采样）；FP8 与 BF16 混合训练；对齐的多任务联合训练。

**📊 数据集**

使用了约 6T 令牌的图文数据，包括 CC12M、LAION‑COCO、SBU、LAION‑2B‑Multi、Wukong 等公共数据集，并通过 PDF 解析生成约 270B 高质量科学图像‑文本对；此外还收集了多学科时间序列数据和结构化科学数据。

**📈 对比分析**

采用 OpenCompass、VLMEvalKit、AgentCompass 等评测框架，在 20+ 科学与通用任务上与 Qwen3‑VL‑235B、Gemini‑3‑Pro、GPT‑5.2 等模型对比，Intern‑S1‑Pro 在 SciReasoner、SFE、SmolInstruct、MatBench 等科研基准上均超越或匹配顶尖闭源模型；在 AIME‑2025、MMLU‑Pro 等通用基准上也实现了最高或接近最高分数。

**⚠️ 局限性**

局限性：训练过程仍易受专家负载不均衡、梯度稀疏化导致的收敛慢；需要海量计算资源与存储，成本高昂；对不同专业数据的负迁移仍存在，需要进一步完善数据多样性与对齐策略；在未覆盖的科学领域或极端稀缺任务上性能尚不充分。

---

## 207. OMIND: Framework for Knowledge Grounded Finetuning and Multi-Turn Dialogue Benchmark for Mental Health LLMs

**arXiv ID:** 2603.25105 | [PDF](https://arxiv.org/pdf/2603.25105v1)

**作者:** Suraj Racha `[一作]` (Indian Institute of Technology Bombay), Ganesh Ramakrishnan `[通讯]` (Indian Institute of Technology Bombay)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究提出oMind框架，构建面向精神健康的LLM代理，包含基于UMLS知识图谱和医学书籍的高质量164k多任务SFT数据集以及全新多轮对话评测基准oMind-Chat。

**💡 创新点**

创新点在于（1）将结构化医学知识与开放式对话生成相结合，打造可解释、知识根植的训练与评测；（2）通过检索‑修剪‑生成‑审核四阶段管道实现精确解释；（3）构建覆盖诊断、判断、开放问答及多轮对话的统一数据与评估体系。

**🔧 技术方法**

使用的技术包括：LoRA微调、DPO偏好对齐、检索式知识补全（BM25+dense）、LLM驱动的修剪与生成、NLI回顾、GPT‑4‑mini与RoBERTa审核、以及基于GPT‑4的评测与标注。

**📊 数据集**

采用的数据集有：多来源精神健康NLP数据（如MHQA、MedMCQA、MMLU‑hsp、MMLU‑pp、USMLE‑Mental、ANGST、Dreaddit、CAMS、SAD、ESConv、MentalChat 16k、CounselLMe）以及从这些种子数据生成的约57k支持性长问答和37.5k多轮对话，最终构成164k SFT样本与961条oMind‑Chat对话。

**📈 对比分析**

与基线（原始LLM、LLM+KG、LLM+Books）对比，oMind‑LLM在多项任务上均显著提升：分类F1平均提升约5–9%，MCQA平均提升17.3%，多轮对话在二进制覆盖率、0–10尺度和Likert评分上均超过基线（尤其是oMind‑Qwen）。DPO对齐进一步提升对话质量，解释方面win率达80%。

**⚠️ 局限性**

局限性包括：①依赖GPT‑4及其算力做生成与评测，成本高；②数据主要来自公开来源，缺乏真实临床对话；③模型仍未在真实临床或用户场景中验证安全性与伦理；④在部分任务（如USMLE、某些分类）表现不如预期，需更细粒度调优。

---

## 208. AuthorityBench: Benchmarking LLM Authority Perception for Reliable Retrieval-Augmented Generation

**arXiv ID:** 2603.25092 | [PDF](https://arxiv.org/pdf/2603.25092v1)

**作者:** Zhihui Yao `[一作]` (State Key Laboratory Of Ai Safety Institute Of Computing Technology Chinese Academy Of Sciences), Keping Bi `[通讯]` (State Key Laboratory Of Ai Safety Institute Of Computing Technology Chinese Academy Of Sciences)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 AuthorityBench 基准，评估 LLM 在源域权威性和实体权威性上的判断能力，并验证其在 RAG 任务中的实用性。

**💡 创新点**

首次系统化定义并量化源域与实体权威性，提出 PointJudge/PairJudge/ListJudge 三种评估方式及权威性过滤提升 RAG 的创新框架。

**🔧 技术方法**

采用 LLM 作为判定器，基于 PageRank 与维基百科 Sitelink 的权威标签，利用 Spearman/Kendall 相关性与 RAG 答案准确率等技术。

**📊 数据集**

使用 10k 站点的 PageRank 评分、22k 实体的 Sitelink 计数以及 120 个是非查询+10 条文档构成的 RAG 数据集。

**📈 对比分析**

对比三种评估方式，发现 ListJudge/PairJudge 加 PointScore 在多种 LLM（Qwen3、Llama3）上 Spearman ρ 最高，且权威过滤在 RAG 中将答案准确率提升至 70%+，显著优于无过滤或相关性过滤。

**⚠️ 局限性**

局限在于仅用 PageRank 作为权威基准，缺乏主题特异性；数据规模有限，RAG 评估样本仅 120 条，可能不具备充分泛化能力。

---

## 209. THEMIS: Towards Holistic Evaluation of MLLMs for Scientific Paper Fraud Forensics

**arXiv ID:** 2603.25089 | [PDF](https://arxiv.org/pdf/2603.25089v1)

**作者:** Tzu-Yen Ma `[一作]` (Beijing University of Posts and Telecommunications), Haihong E `[通讯]` (Beijing University of Posts and Telecommunications)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `e0540dec-d77f-42db-94ae-d039248f6393` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `67630363-6be0-4f51-ab05-7198250671a5` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出并构建了 THEMIS 这一多任务评测基准，用于评估多模态大语言模型在学术图像欺诈识别与定位中的专家级视觉推理能力。

**💡 创新点**

创新体现在三个维度：1) 集成了 4,054 题目涵盖 7 个真实学术情境与 60% 纹理复杂图像；2) 细粒度覆盖 5 类欺诈方式并设计 16 种叠加操作；3) 将欺诈类型映射至 5 大视觉推理能力，实现多维度诊断。

**🔧 技术方法**

采用多模态图像分割、SAM、Flux、Stable Diffusion 等生成技术构造合成欺诈数据，并通过 GPT‑4o‑mini 进行人工审核；评测采用准确率、IoU、F1 及 BRI 等指标。

**📊 数据集**

数据由 152 条真实撤稿案例与 4,635 条高质量合成样本组成，覆盖微观图、医学影像等七大场景。

**📈 对比分析**

在 16 个主流 MLLM 上进行评测，SOTA GPT‑5 的 BRI 最高仅 56.15%，表明模型在复合变换与定位任务上仍显薄弱，且存在显著的能力不均衡。

**⚠️ 局限性**

模型普遍缺乏对多变换、输入扰动的鲁棒性，难以精准定位与跨模态对齐，且受限于文本推理，难以真正体现视觉本身的推理深度。

---

## 210. An Explainable Ensemble Learning Framework for Crop Classification with Optimized Feature Pyramids and Deep Networks

**arXiv ID:** 2603.25070 | [PDF](https://arxiv.org/pdf/2603.25070v1)

**作者:** Syed Rayhan Masud `[一作]` (American International University), Rakib Hossain Sajib `[通讯]` (Begum Rokeya University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

提出了一种可解释的集成学习框架，结合特征金字塔、深度网络、注意力机制和残差网络，实现精准作物分类。

**💡 创新点**

创新地将多种网络结构融合为“Final Ensemble”，并结合 SHAP 与置换重要性解释，使模型在作物分类上达到 98.8% 的准确率。

**🔧 技术方法**

使用特征预处理（标签编码、IQR 去异常、StandardScaler、SMOTE）、Logistic 回归、KNN、SVM、随机森林、XGBoost、相对误差支持向量机、深度网络、特征金字塔网络、注意力机制和残差网络，并通过网格搜索与交叉验证进行超参数调优，最后采用 SHAP 与置换重要性进行可解释性分析。

**📊 数据集**

采用来自埃塞俄比亚农业转型署和 NASA 的 3,867 条记录、29 个特征（土壤 pH、氮、磷、钾等以及温度、降水等）构成的数据集。

**📈 对比分析**

对基线模型与集成模型使用准确率、精确率、召回率、F1 分数进行比较，最终集成模型在所有指标上达到 98.80%，显著优于单一模型。

**⚠️ 局限性**

局限性包括对少数类样本的 SMOTE 合成可能导致过拟合，模型对高维特征的可解释性仍有限，并且仅在单一地区数据上验证，缺乏跨区域泛化测试。

---

## 211. Ultra-fast Traffic Nowcasting and Control via Differentiable Agent-based Simulation

**arXiv ID:** 2603.25068 | [PDF](https://arxiv.org/pdf/2603.25068v1)

**作者:** Fumiyasu Makinoshima `[一作]` (Fujitsu Limited), Sean Qian `[通讯]` (Carnegie Mellon University)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研发了一套端到端可微的基于代理的交通仿真器，实现对大规模城市网络（约10,000个校准参数、1百万辆车）的实时校准、即时预测和控制。

**💡 创新点**

创新点包括：① 通过“轨迹嫁接（Trajectory Grafting）”保持不连续状态更新的可微性；② 将 Gumbel‑Softmax 与 straight‑through 估计结合，实现离散路段选择的可微采样；③ 设计可微交通计数和车流量匹配技术；④ 在 JAX 上实现全向量化与 GPU 加速，使仿真速度达到 173 倍实时。

**🔧 技术方法**

技术手段：JAX 自动微分、GPU 并行化、向量化 Newell 追随模型、Gumbel‑Softmax、straight‑through 估计、Trajectory Grafting、检查点技术（checkpointing）、AdamW 优化器。

**📊 数据集**

使用公开的 Chicago Sketch（993 节点、2571 条路段）与 Sioux Falls（48 节点、100 条路段）网络，并在这两网络上合成 30 分钟的交通计数数据进行实验。

**📈 对比分析**

与传统基于均值参数的不可微模拟对比：校准后 MAE 下降 40%（52.6 对 87.4），相关系数从 0.60 提升至 0.83；全流程（校准 455 s、预测 21 s、控制 728 s）在 20 分钟内完成，留出约 40 分钟实施干预，显著优于传统梯度无关方法。

**⚠️ 局限性**

局限性：实验基于合成数据，未涉及真实传感器噪声、缺失观测和动态反馈；模型在极端拥堵或大规模网络中的鲁棒性未充分验证；需进一步集成需求预测、信号控制等模块以实现完整数字孪生。

---

## 212. Synergistic Event-SVE Imaging for Quantitative Propellant Combustion Diagnostics

**arXiv ID:** 2603.25054 | [PDF](https://arxiv.org/pdf/2603.25054v1)

**作者:** Jing Tao `[一作]` (National University of Defense Technology), Qifeng Yu `[通讯]` (National University of Defense Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种结合空间可变曝光（SVE）相机与神经形态事件相机的闭环测量系统，用于实时监测高能固体推进剂燃烧中的微秒级粒子运动和烟雾。

**💡 创新点**

创新点：①基于烟雾感知的HDR融合，分离粒子辐射与烟雾散射；②利用SVE产生的HDR强度先验指导事件相机去噪和粒子状态分类；③三维立体事件重建实现微秒级分离高度和等效粒径测量，提供前所未有的空间时间分辨率。

**🔧 技术方法**

技术：SVE相机、神经形态事件相机、烟雾概率图、Retinex式HDR融合、事件去噪与粒子状态识别、双目三维重建与尺度归一。

**📊 数据集**

数据集：实验室硼基固体推进剂燃烧记录，包含同步的SVE、事件和传统高速相机序列。

**📈 对比分析**

比较方法：与单曝光、PESPD-MEF HDR融合以及传统高速相机测量比较。指标包括NIQE/NIQMC、平均梯度、三角测量误差（<0.56%）和等效粒径一致性。结果显示HDR融合提升特征梯度，事件测量误差低于0.56%，等效粒径在三种模态下保持一致。

**⚠️ 局限性**

局限性：仅在单一硼基推进剂和固定光学布置下验证；缺乏外部真实基准；系统目前为离线处理；对其他化学或压强条件需重新调参。

---

## 213. Closing the Confidence-Faithfulness Gap in Large Language Models

**arXiv ID:** 2603.25052 | [PDF](https://arxiv.org/pdf/2603.25052v1)

**作者:** Miranda Muqing Miao `[一作]` (University of Pennsylvania), Lyle Ungar `[通讯]` (University of Pennsylvania)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过线性探测和对比激活加法（CAA）技术，发现大型语言模型内部对准确率与口头置信度分别编码在几乎正交的线性方向，并提出两阶段自适应激活驱动的读出校准管线，使模型的口头置信度与真实准确率高度对齐。

**💡 创新点**

创新点在于揭示置信度与准确率在内部表示中的几何正交性、提出“推理污染效应”解释联合推理与置信度输出时的偏差，并通过自适应激活驱动实现跨数据集、跨模型的置信度校准改进，显著降低ECE、Brier和MAE。

**🔧 技术方法**

主要技术包括金标准线性探测（ridge 回归）提取准确率信号、对比激活加法（CAA）构造置信度方向向量、以及利用归一化激活注入实现可控的置信度驱动；同时使用单点自适应Steering强度映射与PCHIP插值实现问题级置信度调节。

**📊 数据集**

实验使用了四大问答基准（MATH、MMLU、TriviaQA、TruthfulQA），以及三大模型族（Llama‑3.1‑8B、Qwen2.5‑7B、Mistral‑7B‑v0.3）的基础版和指令版。

**📈 对比分析**

与无驱动口头置信度、Logit基准以及SteerConf方法相比，所提自适应驱动在所有模型上将ECE降低4–7倍（例如Llama‑3.1‑8B‑Inst.从14.9降至3.7），Brier分数和MAE亦显著下降，证明其在跨数据集和指令版迁移上的强大鲁棒性。

**⚠️ 局限性**

局限性包括：实验仅覆盖7–8B参数模型，尚未验证更大规模模型的线性编码和正交特性；方法依赖两次前向推理，缺乏单次生成的高效实现；开放式生成任务的置信度校准仍待进一步研究。

---

## 214. Efficient ML-DSA Public Key Management Method with Identity for PKI and Its Application

**arXiv ID:** 2603.25043 | [PDF](https://arxiv.org/pdf/2603.25043v1)

**作者:** Penghui Liu `[一作]` (Pengcheng Laboratory), Bin Xiao `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并实现了IPK-pq——一种基于身份识别、NIST ML-DSA和随机矩阵理论的后量子PKI框架，取消传统证书链，直接通过身份映射生成公私钥，并在RPKI环境中实现完整的原型。

**💡 创新点**

创新点包括：①结合随机矩阵与身份映射实现无证书公钥绑定；②在Composite Public Key（CPK）基础上解决线性串通问题；③采用HSM协作生成私钥，消除密钥托管风险；④实现IPK-pq在RPKI中的完整交付；⑤通过单次签名将验证复杂度从O(n)降为O(1)，大幅提升可扩展性。

**🔧 技术方法**

使用技术包括：NIST ML-DSA（后量子签名算法）、随机矩阵理论、Composite Public Key、身份基加密（IBE）、HSM（NXP C293 PCIe卡）、Krill/Routinator RPKI软件、OpenSSL 3.5、SHAKE128/256哈希。

**📊 数据集**

使用数据集：RPKI历史数据（RIPE NCC公开存档，覆盖2015–2026年），以及在VMware和AWS云环境中生成的模拟ROA/RC数据（约8k条/天）。

**📈 对比分析**

比较方法：在三层CA结构下，分别测量ROA生成和验证的TPS（事务/秒），并与标准ML-DSA方案对比；在HSM与软件实现中对比；结果显示IPK-pq在生成方面提升约1.2×，验证提升约4×；使用HSM后，生成/验证速度提升至27–33倍；通信与存储开销从线性增长降为常数。

**⚠️ 局限性**

限制：仍需在HSM上实现SHAKE XOF；部署依赖HSM与身份注册中心，实施成本较高；性能评估主要基于模拟环境，缺乏大规模真实网络测试；安全证明仅覆盖理论基础，缺少完整对抗侧信道等攻击的实证；目前仅针对ML-DSA，其他后量子算法的兼容性尚未验证。

---

## 215. eBeeMetrics: An eBPF-based Library Framework for Feedback-free Observability of QoS Metrics

**arXiv ID:** 2603.25067 | [PDF](https://arxiv.org/pdf/2603.25067v1)

**作者:** Muntaka Ibnath `[一作]` (University of California, Riverside), Daniel Wong `[通讯]` (University of California, Riverside)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

开发了一个基于eBPF的库eBeeMetrics，可在不对应用进行任何注入的情况下，实时从服务器端监测HTTP/1.1和gRPC请求的尾部延迟、吞吐量等QoS指标。

**💡 创新点**

创新点在于通过在eBPF层面利用文件描述符或gRPC流ID实现请求边界的无侵入式识别，并将仅包含开始/结束时间的最小请求级别数据流式传输到用户空间，从而实现高精度、低开销的实时QoS反馈。

**🔧 技术方法**

核心技术包括eBPF kprobes/uprobes、BPF映射、环形缓冲区（ring buffer）与用户空间库的轻量级API，配合CPU时间戳和文件描述符/流ID来区分并计算请求延迟。

**📊 数据集**

使用了10个真实世界的延迟敏感工作负载（如Memcached、Redis、HBase、MongoDB、Triton、TensorRT等）以及云原生基准套件（CloudSuite、vSwarm）进行评估。

**📈 对比分析**

与客户端报告的指标以及之前基于离线eBPF追踪的代理指标比较，eBeeMetrics在所有基准上都能精准追踪尾部延迟（误差小于网络延迟差距）并且吞吐量几乎完全匹配；平均CPU开销不超过3%，对大多数工作负载几乎无影响。

**⚠️ 局限性**

局限性包括：仅能捕获服务器端延迟，无法反映网络排队；依赖于应用暴露稳定的请求生命周期（如文件描述符或gRPC流ID）；目前仅支持HTTP/1.1和gRPC，其他协议需要额外的请求边界识别方案；在极低延迟场景下，eBPF的调度与内核调度交互可能产生微小偏差。

---

## 216. Denoise and Align: Towards Source-Free UDA for Robust Panoramic Semantic Segmentation

**arXiv ID:** 2603.25131 | [PDF](https://arxiv.org/pdf/2603.25131v1)

**作者:** Yaowen Chang `[一作]` (Wuhan University), Zhen Dong `[通讯]` (Wuhan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 DAPASS 框架，用于源自由无监督领域自适应，将标注丰富的针孔相机数据迁移到全景图像语义分割。

**💡 创新点**

设计了 Panoramic Confidence‑Guided Denoising (PCGD) 双路径去噪与类别平衡，以及 Cross‑Resolution Attention Module (CRAM) 结合高分辨率细节与低分辨率全局语义，实现对全景几何失真与伪标签噪声的双重补偿。

**🔧 技术方法**

采用一致性分割、邻域双层优化、copy‑paste 采样、温度缩放交叉熵、多分辨率注意力融合等技术，并基于 SegFormer 作为骨干网络。

**📊 数据集**

在 Cityscapes→DensePASS（户外）和 Stanford2D3D→Stanford2D3D‑Panoramic（室内）两个真实世界基准上进行评估。

**📈 对比分析**

与现有 360SFUDA++、SFDA、DATC 等源自由方法以及 UDA 方法进行对比，DAPASS 在 Cityscapes‑DensePASS 的 mIoU 提升至 55.04%（+2.05%）并在 Stanford‑Pan 取得 70.38%（+1.54%），显著优于对手。

**⚠️ 局限性**

对极少数类别仍易受伪标签质量限制，copy‑paste 方式缺乏几何上下文对齐，导致稀有类性能波动较大。

---

## 217. MCLMR: A Model-Agnostic Causal Learning Framework for Multi-Behavior Recommendation

**arXiv ID:** 2603.25126 | [PDF](https://arxiv.org/pdf/2603.25126v1)

**作者:** Ranxu Zhang `[一作]` (University of Science and Technology of China), Chao Wang `[通讯]` (University of Science and Technology of China)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一个可插拔的因果学习框架 MCLMR，用于多行为推荐（MBR）系统中消除用户行为习惯与物品多行为分布的混淆效应，并通过自适应聚合与偏置感知对比学习提升推荐质量。

**💡 创新点**

创新点包括：①构建多行为因果图，系统性地建模用户与物品的隐含混淆；②基于 Mixture‑of‑Experts 的自适应聚合模块，按用户‑物品对动态融合辅助行为信息；③偏置感知对比学习模块，采用用户‑物品双重对齐与个性化温度，实现跨行为语义对齐并抵消偏差。

**🔧 技术方法**

核心技术为：因果推断（反事实干预）、混合专家网络（MoE）自适应聚合、信息对比学习（InfoNCE）与个性化温度调节，以及传统 GNN/GCN 作为骨干模型的无缝集成。

**📊 数据集**

使用了三大公开电商数据集：Tmall、Jdata、Taobao，包含视图、收藏、加入购物车、购买等多种行为。

**📈 对比分析**

与单行为、传统多行为以及现有因果/对比学习方法（如 SNIPS、CausE、Multi‑IPW、DCCL、CVID、CMSR）进行对比，MCLMR 在所有基线模型上均取得显著提升（例如 LightGCN+MCLMR 在 Tmall 上 HR@10 提升约 24%，NDCG@10 提升 21%），并在不同用户/物品分组中保持更高的表现。

**⚠️ 局限性**

局限性：①框架仍需在大规模工业场景中验证其实时推理效率；②偏置代理使用相对频率近似，可能无法捕获更细粒度的隐含混淆；③对温度调节的超参数需要手动设置，缺乏自适应化；④对极度稀疏或极端长尾行为的鲁棒性仍待进一步研究。

---

## 218. AnyDoc: Enhancing Document Generation via Large-Scale HTML/CSS Data Synthesis and Height-Aware Reinforcement Optimization

**arXiv ID:** 2603.25118 | [PDF](https://arxiv.org/pdf/2603.25118v1)

**作者:** Jiawei Lin `[一作]` (Xi'an Jiaotong University), Christopher Tensmeyer `[通讯]` (Adobe Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了一套统一基于 HTML/CSS 的文档生成框架，可同时完成意图→文档、截图→代码、元素→文档等三类任务，并通过高度感知强化学习进一步抑制内容溢出。

**💡 创新点**

创新点在于：①提出 HTML/CSS 作为统一可编辑文档表示，克服传统 raster/平面坐标的局限；②构建规模达 265,206 条、覆盖 111 类、32 风格的合成数据集；③引入 Height‑Aware Reinforcement Learning（HARL）解决文档溢出问题；④在三大任务上统一评测并显著优于现有通用与专用模型。

**🔧 技术方法**

技术手段包括：多模态大语言模型（如 Qwen2.5‑VL‑7B‑Instruct）微调、生成式代码与图像模型（Qwen3‑Coder、FLUX.1‑dev）合成、Playwright 渲染、GRPO‑based HARL、LoRA 微调、数据清洗与多阶段生成管道。

**📊 数据集**

使用自研合成数据集（约 265k 条文档，111 类，32 风格），包含元数据、HTML/CSS 源码、合成图像与渲染截图；并对比通用 MLLM（GPT‑4o、InternVL3‑78B）以及专用基线（OpenCOLE、FLUX、WebSight、Design2Code、LaDeCo）。

**📈 对比分析**

与基线对比，模型在 I2D、DD、E2D 三大任务中均在 MLLM‑based 评分（布局、图像、排版、内容、创新）和专用指标（Block、Text、Position、Color、CLIP）上取得明显提升；且 Height 指标大幅下降，说明溢出被有效控制；实验结果表明即使使用 10K 样本也能击败 GPT‑4o，整体性能稳健。

**⚠️ 局限性**

局限性包括：①模型规模相对较小，仍存在偶发溢出和某些极复杂布局的细节不足；②HARL 的奖励设计需进一步细化以避免奖励作弊；③数据合成仍受预训练模型质量限制，部分风格和排版可能与真实工业样本存在偏差；④当前仅验证了三类任务，未来需扩展到更多生成场景。

---

## 219. SEVerA: Verified Synthesis of Self-Evolving Agents

**arXiv ID:** 2603.25111 | [PDF](https://arxiv.org/pdf/2603.25111v1)

**作者:** Debangshu Banerjee `[一作]`, Gagandeep Singh `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种在自我进化的LLM代理中加入形式化安全约束的完整框架，核心是Formally Guarded Generative Models (FGGM) 与搜索-验证-学习 (search‑verify‑learn) 循环。

**💡 创新点**

创新点在于：① 将每个生成模型调用包装为FGGM，使用一阶逻辑约束、拒绝采样与验证回退，使输出对所有参数始终满足硬约束；② 通过CEGIS式搜索保证程序结构满足约束，再对参数进行无约束梯度优化，兼顾安全性与性能；③ 将形式化约束直接注入LLM提示，提升搜索效率。

**🔧 技术方法**

使用技术包括：LLM程序合成、Dafny verifier与SMT求解、形式化一阶逻辑约束、拒绝采样与回退、GRPO+LoRA微调、梯度优化与自动化回调、数据驱动任务损失。

**📊 数据集**

实验数据集涵盖四类任务：① Symbolic Regression（35个任务，600/400样本）② DafnyBench/HumanEvalDafny（760/135程序）③ GSM‑Symbolic（100/50题）④ τ²‑bench（航空50/114、零售50/114交互）。

**📈 对比分析**

与无约束自我进化框架、手工代理及现有状态技术比较，结果显示：0%约束违背；验证成功率提升至97%/89%，GSM准确率从38%提升至66%，符号回归NMSE降至0.02；在所有任务上均优于基线与现有方法。

**⚠️ 局限性**

局限性包括：① 资源约束（调用次数、token数）未显式建模；② 对闭源LLM参数无法微调，只能靠提示调优；③ 验证与拒绝采样导致额外运行时开销；④ 需人工编写一阶逻辑约束，难以覆盖更高阶语义或动态协同场景。

---

## 220. MoireMix: A Formula-Based Data Augmentation for Improving Image Classification Robustness

**arXiv ID:** 2603.25109 | [PDF](https://arxiv.org/pdf/2603.25109v1)

**作者:** Yuto Matsuo `[一作]` (National Institute of Advanced Industrial Science and Technology), Akio Nakamura `[通讯]` (Tokyo Denki University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 MoireMix 数据增强方法，通过在训练时程序化生成摩尔干涉纹理并与图像混合，以提升图像分类模型的鲁棒性。

**💡 创新点**

创新点在于使用闭式解析公式即时生成无存储需求的干涉纹理，彻底替代外部纹理数据集，同时实现多尺度频率覆盖和高效性。

**🔧 技术方法**

采用解析摩尔干涉纹理生成、PixMix 混合框架、Beta 混合操作、Vision Transformer 训练、ImageNet-C/R/Moire、PGD 对抗测试等技术。

**📊 数据集**

实验使用 ImageNet-1k 及其变体（ImageNet-C、ImageNet-R、ImageNet-Moire、ImageNet-PGD）以及 CIFAR-10/100 进行验证。

**📈 对比分析**

与 Cutout、AugMix、RandAug、MixUp、CutMix、PixMix、IPMix、DiffuseMix 等无外部纹理方法对比，MoireMix 在 ImageNet-C、R、PGD、Moire 等指标上分别提升至 50.5%、23.1%、3.1%、36.2%，总体相较基线提升约 16 点。

**⚠️ 局限性**

限制在于无法与 PixMix 等外部纹理最强方法完全匹配，仅研究了一类程序化纹理，未来可扩展至更丰富的解析纹理生成方式。

---

## 221. Label What Matters: Modality-Balanced and Difficulty-Aware Multimodal Active Learning

**arXiv ID:** 2603.25107 | [PDF](https://arxiv.org/pdf/2603.25107v1)

**作者:** Yuqiao Zeng `[一作]` (Beijing Jiaotong University), Hui Yu `[通讯]` (University of Glasgow)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 RL-MBA 框架，利用强化学习实现多模态主动学习的自适应采样，动态平衡模态贡献与样本难度。

**💡 创新点**

创新点在于引入 Adaptive Modality Contribution Balancing (AMCB) 与 Evidential Fusion for Difficulty-Aware (EFDA) 两大模块，并将采样建模为马尔科夫决策过程，通过奖励反馈持续调节采样策略。

**🔧 技术方法**

使用技术包括 Policy Gradient 强化学习、证据层 Dirichlet 融合、多模态加权、基于不确定性与多样性的评分、预算 k-means++ 选样等。

**📊 数据集**

实验数据集为 Food101（图像+文本）、KineticsSound（视频+音频）与 VGGSound（视频+音频）。

**📈 对比分析**

与 Random、Entropy、GCNAL、CoreSet、DeepFool、BALD、BADGE、BMMAL 等基线对比，RL‑MBA 在相同标签预算下均实现 Top‑1 准确率提升约 0.3–2% 的显著改进。

**⚠️ 局限性**

局限性包括：需要预设验证集与超参数（如温度、奖励形式），奖励设计需手工调优，且在不同多模态组合或更大规模数据集上的泛化性尚未充分验证。

---

## 222. ElephantBroker: A Knowledge-Grounded Cognitive Runtime for Trustworthy AI Agents

**arXiv ID:** 2603.25097 | [PDF](https://arxiv.org/pdf/2603.25097v1)

**作者:** Cristian Lupascu `[一作]` (Elephant Broker), Alexandru Lupascu `[通讯]` (Elephant Broker)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出并实现了 ElephantBroker——一个面向 LLM 代理的完整认知运行时，提供从知识图到向量检索、评分、上下文拼装、安全门控、合并学习等全流程，支持多代理、多组织场景的持久记忆与可追溯性。

**💡 创新点**

创新点在于：① 将 Neo4j 知识图与 Qdrant 向量库通过 Cognee SDK 统一写入，形成可追溯且可验证的双写存储；② 设计了五源混合检索与十一维度竞价评分，结合预算约束实现高效、可信的上下文装配；③ 引入四态证据验证模型和数字权限层，构建从事实到证明到审批的完整可信链；④ 架构化的六层安全门控与 AI 防火墙，实现从策略到物理拦截的双重安全保障；⑤ 通过九阶段合并引擎实现持续的知识精炼与自学习。

**🔧 技术方法**

技术包括：Neo4j 图数据库、Qdrant 向量检索、Cognee SDK、Python FastAPI、Redis、OpenTelemetry+Prometheus、OpenClaw 插件、LiteLLM、NeMo Guardrails、LLM Embedding 与 Cross-Encoder API、JSON‑Schema/Pydantic、SQLite、ClickHouse 等。

**📊 数据集**

实验数据主要来自内部构造的 2,200 条单元/集成/端到端测试用例，覆盖记忆写入、检索、评分、门控、合并与安全扫描等核心子系统；未使用公开基准数据集，而是通过场景化对话和模拟代理交互进行验证。

**📈 对比分析**

与 MemGPT、Mem0、GraphRAG、Zep 等现有记忆/检索框架对比，ElephantBroker 在可追溯性、证据验证、动态门控和多组织权限控制方面实现了显著提升；在 2,200 条验证用例中，子系统通过率均达到 100%，但论文未给出具体任务完成率或幻觉率等性能指标。

**⚠️ 局限性**

主要局限：① 未提供基准任务的实证性能数据（如 RAG 准确率或幻觉减少率）；② 依赖多种外部服务（Neo4j、Qdrant、Redis、外部 LLM API），对部署复杂度和成本有一定影响；③ 评分权重与策略初始为专家设定，调优机制有限；④ 论文侧重架构验证，缺乏大规模真实对话评估。

---

## 223. Sparse Visual Thought Circuits in Vision-Language Models

**arXiv ID:** 2603.25075 | [PDF](https://arxiv.org/pdf/2603.25075v1)

**作者:** Yunpeng Zhou `[一作]` `[通讯]` (University of Reading), Yunpeng Zhou (University of Reading)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在冻结的视觉‑语言模型中，作者利用稀疏自编码器定位并干预视觉思考回路，验证了任务相关特征的因果必要性，并探究其线性可组合性。

**💡 创新点**

创新点在于提出可复现的因果诊断管道、揭示了特征集合的非模块化干预效应、从几何对抗与噪声放大角度解释了交互失效，并在多模型与跨域数据集上验证了这一现象。

**🔧 技术方法**

采用线性探针定位关键层，训练Top‑K稀疏自编码器，构建任务与全局特征集合，使用遮蔽干预算子、标准化校准、空间反投影以及统计控制进行实验。

**📊 数据集**

使用了自制的Synthetic Visual Reasoning (SVR) 基准，以及 NLVR2、SNLI‑VE、CLEVR、VSR 等公开视觉推理数据集进行跨域验证。

**📈 对比分析**

与随机/置换对照以及单集成干预进行对比，Pattern集合在多数模型中提升约 0.5–1.0pp 准确率，Union集合则导致 1–2pp 准确率下降并显著增加输出漂移；在不同模型和数据集上保持一致性。

**⚠️ 局限性**

局限包括仅针对 Qwen3‑VL‑8B 的单一模型架构，干预仅为线性加法且缺乏动态时序控制，且对层级的非模块化阈值可能因模型而异，未能解决离谱状态下的非线性失效。

---

## 224. Approaches to Analysing Historical Newspapers Using LLMs

**arXiv ID:** 2603.25051 | [PDF](https://arxiv.org/pdf/2603.25051v1)

**作者:** Filip Dobranić `[一作]` (Institute of Contemporary History), Darja Fišer `[通讯]` (Institute of Contemporary History)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过主题建模、LLM情感分析、实体图可视化和批判性话语分析，对19世纪末斯洛文尼亚报纸Slovenec与Slovenski narod中集体身份与情绪进行大规模定量与定性研究。

**💡 创新点**

①将指令式LLM用于OCR降噪历史报纸的方面级情感标注；②将主题、情绪与实体共现构成图谱，指导近读；③通过跨模型对比评估历史文本中LLM的适用性。

**🔧 技术方法**

BERTopic主题模型、四种指令式LLM（GaMS3‑12B‑Instruct、Gemma‑3‑12B‑IT、LLaMA‑3.1‑8B‑Instruct、DeepSeek‑R1‑Distill‑Qwen‑7B）情感分类、NER与实体图绘制、批判性话语分析与网络分析。

**📊 数据集**

sPeriodika 1 B单词史料库中的Slovenec（1873–1945）与Slovenski narod（1868–1943）两份报纸；手工标注400条集体身份提及的情感评估集。

**📈 对比分析**

在371条标注样本上对四模型做准确率与F1比较，GaMS3在准确率(0.695)与加权F1(0.708)最高，宏平均F1最高为Gemma3(0.555)。不同语法与指代类型下性能差异明显，名词化提及准确率低于形容词化，集体引用比非集体引用更难。

**⚠️ 局限性**

模型对正向情感识别不足、负向情感过度召回；OCR噪声与历史语言变体导致性能受限；情感类别不平衡使加权与宏平均差异明显；聚合结果可能低估正面评价，需结合质性验证。

---

## 225. MP-MoE: Matrix Profile-Guided Mixture of Experts for Precipitation Forecasting

**arXiv ID:** 2603.25046 | [PDF](https://arxiv.org/pdf/2603.25046v1)

**作者:** Huyen Ngoc Tran `[一作]` (Hanoi University of Science and Technology), Nam-Phong Nguyen `[通讯]` (Hanoi University of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `afceb026-1760-41ae-8d86-010831a37d97` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计并验证了基于Matrix Profile的Mixture of Experts框架MP-MoE，用于NWP后处理以提高越南热带降水预测的时空结构精度。

**💡 创新点**

结合结构化损失与自适应专家门控，使用未归一化MP距离代替传统点误差，解决双重惩罚与峰值削平问题。

**🔧 技术方法**

混合MSE+MP损失、轻量MLP门控网络、PyTorch实现、DTW、CSI-M评估等技术。

**📊 数据集**

两大越南山区流域（Ban Nhung、Song Chay）观测降水与GFS/WRF等NWP专家输出的数据集。

**📈 对比分析**

与多种学习模型（XGBoost、LSTM等）和物理专家对比，MP-MoE在MAE、DTW和CSI-M等指标上均显著优于基线，峰值捕捉更准确。

**⚠️ 局限性**

仅在一维时序层面，专家池多样性受限；固定超参数Δ和λ；未考虑空间耦合；需进一步自适应调参和在线学习。

---

## 226. MoRGS: Efficient Per-Gaussian Motion Reasoning for Streamable Dynamic 3D Scenes

**arXiv ID:** 2603.25042 | [PDF](https://arxiv.org/pdf/2603.25042v1)

**作者:** Wonjoon Lee `[一作]` (Yonsei University), Sangyoun Lee `[通讯]` (Yonsei University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出MoRGS框架，实现在线动态场景重建时显式建模每个高斯点的运动。

**💡 创新点**

创新点是利用稀疏光流作为运动先验，结合可学习的运动偏移场和运动置信度，实现对动态高斯点的精确更新。

**🔧 技术方法**

采用3D高斯散点渲染（3DGS）、光流监督、运动偏移场、运动置信度网络、SAM2分割及稀疏视角光流。

**📊 数据集**

在Neural 3D Videos（N3DV）与Meet Room这两个室内动态场景数据集上进行评测。

**📈 对比分析**

与现有在线方法（3DGStream、QUEUE、4DGC等）比较，MoRGS在PSNR/SSIM等指标上领先，且保持近乎同等的训练延迟与实时渲染速度。

**⚠️ 局限性**

局限在于光流稀疏采样对运动推断仍有依赖，且在极端快速运动或复杂遮挡场景下效果可能受限。

---

## 227. Process-Aware AI for Rainfall-Runoff Modeling: A Mass-Conserving Neural Framework with Hydrological Process Constraints

**arXiv ID:** 2603.25093 | [PDF](https://arxiv.org/pdf/2603.25093v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 228. GeoNDC: A Queryable Neural Data Cube for Planetary-Scale Earth Observation

**arXiv ID:** 2603.25037 | [PDF](https://arxiv.org/pdf/2603.25037v1)

**作者:** Jianbo Qi `[一作]` (Beijing Normal University), Qiao Wang `[通讯]` (Beijing Normal University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建 GeoNDC，利用连续时空隐式神经场将地球观测数据压缩成可查询的神经数据立方体

**💡 创新点**

首次将长周期卫星档案统一转化为可直接查询、连续重建且存储极度压缩的神经数据立方体，解决传统文件体系的存储、查询与重建分离问题

**🔧 技术方法**

采用多分辨率哈希编码、双分支（高分辨率2D + 低分辨率3D）架构、MLP解码器以及可选稀疏残差层，并通过 WebGPU 实现浏览器端推理

**📊 数据集**

MODIS MCD43A4 20 年光谱记录、Sentinel‑2 10m 时间序列、HiGLASS LAI/FPAR 20m 产品

**📈 对比分析**

与传统 GeoTIFF、Int16+Zstd、JPEG2000 以及单帧 INR（COIN）对比，压缩率约 380:1，单帧 R²>0.98（≥0.99）且查询延迟比 GeoTIFF 快 80×

**⚠️ 局限性**

训练过程计算量大，极端突发事件易欠拟合，表示为有损，需保留原始观测及残差以保证科学透明度

---

## 229. When Sensing Varies with Contexts: Context-as-Transform for Tactile Few-Shot Class-Incremental Learning

**arXiv ID:** 2603.25115 | [PDF](https://arxiv.org/pdf/2603.25115v1)

**作者:** Yifeng Lin `[一作]` (Fuzhou University), Zheng-Jun Zha `[通讯]` (University of Science and Technology of China)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对触觉感知中的上下文变化，提出了 Context-as-Transform（CaT）与 Uncertainty-Conditioned Prototype Calibration（UCPC）的 FSCIL 框架。

**💡 创新点**

创新点：将上下文建模为可逆低维变换并通过伪上下文一致性学习；使用上下文不确定性进行原型校准，缓解残差高维上下文的影响。

**🔧 技术方法**

技术：log‑Mel 频谱、可逆几何变换+幅度调制、伪上下文一致性损失、UCPC 原型收缩、ResNet‑18 表征网络、可学习分类器。

**📊 数据集**

数据集：HapTex、LMT108 两个数值触觉基准。

**📈 对比分析**

与 PriViLege、Comp‑FSCIL、OrCo、ADBS、PITS‑SC 等基线比较，在两大数据集上均取得最高平均准确率（AA），同时 PD 与 ADR 也更低，表明性能稳健且优于现有方法。

**⚠️ 局限性**

局限：对高维残差上下文的建模仍依赖伪上下文近似，超参数（如 n_ucpc）会影响稳定性；实验仅在无重放模式下评估，未讨论显存/计算预算等实际部署问题。

---

## 230. Layer-Specific Lipschitz Modulation for Fault-Tolerant Multimodal Representation Learning

**arXiv ID:** 2603.25103 | [PDF](https://arxiv.org/pdf/2603.25103v1)

**作者:** Diyar Altinses `[一作]` (South Westphalia University of Applied Sciences), Andreas Schwung `[通讯]` (South Westphalia University of Applied Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于Lipschitz理论的双阶段自监督多模态故障容错学习框架；

**💡 创新点**

通过解析卷积与全连接层对局部扰动的传播特性，实现检测-纠正的敏感性与收敛性平衡，并引入层级Lipschitz调节和梯度放大/裁剪；

**🔧 技术方法**

自监督对比损失、相似性对齐、卷积/全连接网络、梯度裁剪、Spectral Normalization、Jacobian正则化、动态梯度放大；

**📊 数据集**

三组工业机器人数据集（MuJoCo、单机器人焊接站、双机器人焊接站），每组包含图像与运动/传感器两模态；

**📈 对比分析**

与多模态AE、VAE、模糊正则、概念融合、GAN纠正等基线对比，实验表明MMSSL在组合误差上比最优对手提升约10‑37%，且检测宏F1≥0.93；

**⚠️ 局限性**

缺点包括对超参数（对比/相似权重、放大因子、裁剪阈值）敏感，且对大规模高维模态的扩展尚未验证，模型对小幅异常仍易误判为正常。

---

## 231. Bridging Perception and Reasoning: Token Reweighting for RLVR in Multimodal LLMs

**arXiv ID:** 2603.25077 | [PDF](https://arxiv.org/pdf/2603.25077v1)

**作者:** Jinda Lu `[一作]` (University of Science and Technology of China), Xiangnan He `[通讯]` (University of Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文在多模态大语言模型的强化学习可验证奖励（RLVR）框架中提出了一种 Token‑Reweighting（ToR）策略。

**💡 创新点**

创新点在于揭示感知与推理 token 在训练过程中的耦合关系，并通过动态重权重实现二者的联合优化。

**🔧 技术方法**

采用 RLVR 基础算法（GRPO、DAPO），并结合 token 预测熵与视觉敏感度两种度量来识别关键推理与感知 token。

**📊 数据集**

使用 Geometry3K 进行训练，评估数据集包括 MathVerse、MathVision、MathVista、WeMath 与 HalluBench。

**📈 对比分析**

与现有 RLVR 基线对比，ToR 在多项评测上平均提升约 2–3% 绝对分数，达到或逼近各基准的最佳成绩。

**⚠️ 局限性**

局限性包括需手动设置 token 选择比例与权重，重权重对重叠 token 的处理略显经验性，且目前仅在 token 层面开展耦合分析。

---

## 232. Learning Explicit Continuous Motion Representation for Dynamic Gaussian Splatting from Monocular Videos

**arXiv ID:** 2603.25058 | [PDF](https://arxiv.org/pdf/2603.25058v1)

**作者:** Xuankai Zhang `[一作]` (Sun Yat-sen University), Qing Zhang `[通讯]` (Sun Yat-sen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种基于SE(3) B-spline的连续动态高斯分裂框架，可从单目视频中高质量合成新视角。

**💡 创新点**

创新点包括：①使用SE(3)累计B-spline构建连续位置与姿态的动态高斯轨迹；②引入自适应控制机制动态调整运动基数与控制点数量；③设计软段重建策略降低长时间间隔运动的干扰；④利用多视角扩散模型的SDS损失增强不可见区域的先验信息。

**🔧 技术方法**

主要技术包括3D Gaussian Splatting、SE(3) B-spline运动基、动态高斯变形、软段重建、基于扩散模型的多视角先验、ARAP平滑、光流轨迹约束、相机平滑损失以及联合优化。

**📊 数据集**

实验数据集为iPhone（手持摄像机拍摄的多场景数据）和NVIDIA（12摄像机陀螺仪阵列拍摄的7场景）。

**📈 对比分析**

与Shape-of-Motion、MoSca、Gaussian Marbles、SplineGS、MoDec-GS、HiMoR等方法比较，实验表明在iPhone数据集上的mPSNR、mSSIM、mLPIPS以及在NVIDIA数据集上的PSNR、SSIM、LPIPS均显著优于对比方法；在关键点跟踪（PCK-T）评估中也取得更高的准确率。

**⚠️ 局限性**

局限性：在大幅非刚性变形、强运动模糊、快速相机或物体运动的场景下仍难以获得理想的渲染效果。

---

## 233. $π$, But Make It Fly: Physics-Guided Transfer of VLA Models to Aerial Manipulation

**arXiv ID:** 2603.25038 | [PDF](https://arxiv.org/pdf/2603.25038v1)

**作者:** Johnathan Tucker `[一作]` (Stanford University), Mac Schwager `[通讯]` (Stanford University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `40105733-5154-44cd-8090-a8cab9e64b07` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出 AirVLA 框架，将基于固定底座的 π_0 VLA 模型迁移到无人机平台，完成无人机的拾取-投放和导航任务。

**💡 创新点**

通过在流匹配采样中注入 payload‑aware guidance 实现实时动力学补偿，并利用 Gaussian Splatting 生成导航数据以弥补数据稀缺，突破了 VLA 在多动态平台上的迁移瓶颈。

**🔧 技术方法**

实时片段化 (RTC)、梯度引导的流匹配采样、Gaussian Splatting 重建与合成、低阶飞控包装、语义与视觉嵌入等技术。

**📊 数据集**

手工收集 270 条无人机遥操作拾取-投放与导航演示，并使用 Gaussian Splatting 合成 50 条导航示例；未使用公开数据集，全部自制。

**📈 对比分析**

与 π_0 naive、π_0+RTC、π_0+RTC+payload guidance、ACT、Diffusion Policy 等方法对比；在拾取-投放任务中成功率从 0% 提升至 50%，门式导航从 50% 提升至 81%（合成），综合任务最终成功率达 62.5%。

**⚠️ 局限性**

对运动捕捉定位高度依赖、训练样本量小导致导航泛化受限、payload guidance 仅针对竖直方向，无法处理更复杂的动力学与外部扰动。

---

## 234. Auditing Algorithmic Personalization in TikTok Comment Sections

**arXiv ID:** 2603.25061 | [PDF](https://arxiv.org/pdf/2603.25061v1)

**作者:** Yueru Yan `[一作]` (Indiana University Bloomington), Siqi Wu `[通讯]` (Indiana University Bloomington)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

对TikTok 2024美国总统选举相关视频的评论区进行算法审计，使用袜子账户训练偏左/偏右偏好并抓取评论排序。

**💡 创新点**

首次在评论区揭示算法主要通过排名重排实现个性化，而非删除内容，且与视频评论量、参与度不平衡和党派失衡相关。

**🔧 技术方法**

采用sock-puppet训练-验证-评估框架、Jaccard距离、Normalized Damerau–Levenshtein距离、ANOSIM统计、PCA+K-means聚类、Spearman相关等技术。

**📊 数据集**

使用从257个TikTok政治频道提取的4,691条选举相关视频，筛选65条中立视频，15,000+评论，以及17个训练成功的袜子账户。

**📈 对比分析**

通过ANOSIM和随机置换检验对比不同政治组之间与组内差异，发现约25%视频排名差异显著，聚类后发现高互动视频更易出现个性化，但整体效果受限。

**⚠️ 局限性**

受限于仅使用搜索-观看训练、样本规模有限、评论区左倾偏倚、只评估旧视频且未考虑位置、关注等信号，可能低估实际个性化程度。

---

## 235. CTS-PLL: A Robust and Anytime Framework for Collaborative Task Sequencing and Multi-Agent Path Finding

**arXiv ID:** 2603.25121 | [PDF](https://arxiv.org/pdf/2603.25121v1)

**作者:** Junkai Jiang `[一作]`, Jianqiang Wang `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种改进的 CTS-MAPF 求解框架 CTS-PLL，融合了锁定检测与释放（使用 LaCAM 本地重规划）和基于 LNS 的任意改进机制，以提升求解成功率和路径质量。

**💡 创新点**

创新点：①在配置式搜索 CTS-PIBT 上加入锁定检测/释放模块，显著克服死锁/循环锁；②引入 LNS 任意改进阶段，在固定任务序列下进一步优化流时间；③两者结合构成完整的高效、鲁棒且可持续改进的 CTS-MAPF 求解器。

**🔧 技术方法**

技术：配置式搜索（Extended‑PIBT）、锁定检测与 LaCAM 本地 MAPF 重规划、Large Neighborhood Search（LNS）以及任务序列生成（K‑best 组合）。

**📊 数据集**

使用 CTS‑MAPF 数据集中的四种网格地图（empty、random、room、maze），分别在稀疏和稠密设置下生成 50 个随机实例进行评测。

**📈 对比分析**

与 CTS‑CBS、CBSS、CTS‑PIBT 等基线对比。稀疏场景下 CTS‑PLL 在 60 s 预算内 100% 成功率，流时间平均比 CTS‑PIBT 提升 10.4%；稠密场景下成功率从 CTS‑PIBT 的 38–76% 提升至 72–98%，流时间改善 1–17%（CTS‑PLL‑v3）。实验还在真实机器人上验证了方法的可行性。

**⚠️ 局限性**

局限：①锁定释放依赖随机目标，缺乏正式完整性保证；②在极稠密或大规模任务/代理时，LNS 迭代深度与时间开销仍较大；③未对动态/不确定环境做适配，缺乏在线学习或预测模块。

---

## 236. DFLOP: A Data-driven Framework for Multimodal LLM Training Pipeline Optimization

**arXiv ID:** 2603.25120 | [PDF](https://arxiv.org/pdf/2603.25120v1)

**作者:** Hyeonjun An `[一作]` (Yonsei University), Kwanghyun Park `[通讯]` (Yonsei University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了DFLOP——一种基于数据驱动的多模态大型语言模型（MLLM）训练管线优化框架，能够同时调优3D并行配置和在线微批次调度。

**💡 创新点**

创新点包括：①结合输入数据分布与模型性能建模，实现对输入相关吞吐变异的预测；②通过独立的3D并行策略和Inter-model Communicator支持模态编码器与LLM的异构并行；③在线微批次调度采用ILP+LPT混合求解，并配备Adaptive Correction自适应校正；④整体实现可在不改动模型结构的前提下显著提升GPU利用率和吞吐率。

**🔧 技术方法**

主要技术：数据驱动性能建模（内存与吞吐）；多维搜索优化3D并行配置；ILP+LPT在线微批次调度；Adaptive Correction动态误差校正；Inter-model Communicator处理不同并行组间通信。

**📊 数据集**

使用的多模态数据集为混合组合，包括LLaVA-Wild、AI2D、Infographic VQA、M4-Instruct、LLaVA-Video，涵盖单图、多图和视频等多种模态样本。

**📈 对比分析**

对比方法：与Megatron‑LM和自定义PyTorch基线在多种MLLM架构（LLaVA‑OV、InternVL‑2.5）和不同模型规模下进行端到端实验；评价指标包括GPU吞吐率、总训练时间、GPU空闲率等。实验显示，DFLOP相较基线提升1.2×–3.6×吞吐，训练时间缩短5–40小时，GPU空闲率降至约16%。

**⚠️ 局限性**

局限性：①需对每个模型和数据集完成性能重新profiling，重建成本一定；②对极端输入分布或模型结构剧烈变化时需要重新profiling；③当前实现仅基于PyTorch，无法直接兼容DeepSpeed等框架；④在极大GPU规模下搜索空间仍较大，虽然计算量可接受，但实际配置仍依赖经验选择。

---

## 237. Do LLMs Know What They Know? Measuring Metacognitive Efficiency with Signal Detection Theory

**arXiv ID:** 2603.25112 | [PDF](https://arxiv.org/pdf/2603.25112v1)

**作者:** Jon-Paul Cacioli `[一作]` `[通讯]`, Jon-Paul Cacioli

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估大型语言模型的元认知能力，提出基于Type‑2 SDT的meta‑d'与M‑ratio评估框架

**💡 创新点**

首次将meta‑d'和M‑ratio应用于LLM，揭示同一模型在准确率相近时的元认知效率差异，并发现域特异性与温度对准则的解耦

**🔧 技术方法**

使用Token级归一化对数概率作为连续置信度信号，并通过最大似然估计计算meta‑d'和M‑ratio

**📊 数据集**

在TriviaQA和NaturalQuestions两套事实问答数据集上进行224,000次实验

**📈 对比分析**

与ECE、Brier分数、AUROC_2比较，发现M‑ratio能揭示隐藏的元认知结构，模型排名与AUROC_2完全相反，表明M‑ratio更适用于信心依赖的部署

**⚠️ 局限性**

局限在于仅评估7–9B开源模型，未覆盖更大规模模型，且仅使用内部Token概率，API模型无法直接使用

---

## 238. MSRL: Scaling Generative Multimodal Reward Modeling via Multi-Stage Reinforcement Learning

**arXiv ID:** 2603.25108 | [PDF](https://arxiv.org/pdf/2603.25108v1)

**作者:** Chenglong Wang `[一作]` (Northeastern University), Tong Xiao `[通讯]` (Northeastern University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a4b10f5d-130b-4e77-9367-6469ec621899` `8d10c613-917e-4880-9716-17789f50e119` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了一种多阶段强化学习(MSRL)框架，用于可扩展地训练生成式多模态奖励模型。

**💡 创新点**

创新点在于先利用海量文本偏好数据训练奖励推理能力，然后通过跨模态知识蒸馏和任务识别奖励逐步迁移到多模态任务，从而在仅使用极少多模态偏好数据的情况下显著提升性能。

**🔧 技术方法**

技术包括：生成式多模态奖励模型、RLVR、跨模态知识蒸馏(CMKD)、多阶段强化学习（文本阶段、caption阶段、全模态阶段）、GRPO优化和经验回放等。

**📊 数据集**

使用的数据集包括：HelpSteer3（40k推理样本）、GRAM‑R^2（400k文本偏好样本）、20k caption‑based偏好样本、20k真实多模态偏好样本；评测集有VL‑RewardBench、Multimodal RewardBench、GenAI‑Bench、VideoGen‑RewardBench、ShareGPTVideo 等。

**📈 对比分析**

与 UnifiedReward、R1‑Reward、LLaVA‑Critic、Discriminative/Generative MRM、LLM‑as‑Judge 等基线比较，MSRL 在视觉理解/生成任务上提升约6‑8%准确率，例如 VL‑RewardBench 从 66.6% 提升到 75.9%，GenAI‑Bench 从 70.2% 提升到 75.7%，并且在不同模型规模（1B‑14B）均保持显著优势。

**⚠️ 局限性**

局限性包括：仍需少量多模态偏好数据；跨模态蒸馏依赖文本与视觉描述的一致性，可能在长文本或复杂视频场景下效果受限；以及 Stage‑1 对大规模文本偏好的依赖，若文本偏好与真实多模态偏好差异过大可能导致迁移偏差。

---

## 239. Learning domain-invariant features through channel-level sparsification for Out-Of Distribution Generalization

**arXiv ID:** 2603.25083 | [PDF](https://arxiv.org/pdf/2603.25083v1)

**作者:** Haoran Pei `[一作]` (Beihang University), Baochang Zhang `[通讯]` (Beihang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了层级因果丢弃（Hierarchical Causal Dropout, HCD）框架，用于通过通道级稀疏化和信息瓶颈实现对抗域外（OOD）分布的鲁棒性。

**💡 创新点**

创新点包括：① 通过可学习的通道级稀疏掩码实现表示层级的因果干预；② 引入基于矩阵的互信息（MMI）来量化并最小化特征与域标签的相关性；③ 结合 StyleMix 与 VICReg 对稀疏化的正则化，提升表示的域不变性。

**🔧 技术方法**

使用了通道级稀疏化、矩阵互信息损失、StyleMix 生成的伪 OOD 采样、VICReg 结构化正则化以及基于 DenseNet-121 / ResNet-50 的网络骨干。

**📊 数据集**

在 Camelyon17（医学影像）和 iWildCam（野生动物监测）这两个 WILDS 公开数据集上进行实验。

**📈 对比分析**

与 ERM、Bonsai、IRM、IRMX、GroupDRO、VREx 等基线进行比较，HCD 在 Camelyon17 上取得 86.62% 的最高准确率，iWildCam 上取得 33.09% 的最佳准确率，显著优于其他方法。

**⚠️ 局限性**

主要限制是基于矩阵的互信息估计在批大小上呈二次复杂度，可能影响在极大规模训练中的可扩展性。

---

## 240. GIFT: Global Irreplaceability Frame Targeting for Efficient Video Understanding

**arXiv ID:** 2603.25072 | [PDF](https://arxiv.org/pdf/2603.25072v1)

**作者:** Junpeng Ma `[一作]` (Fudan University), Jian Pu `[通讯]` (Fudan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种无训练的关键帧选择框架GIFT，通过评估每帧的不可替代性来挑选视频帧；

**💡 创新点**

创新点在于：① 用“定向多样性”重新定义多样性，仅与更相关帧比较；② 采用预算感知细化（Budget‑Aware Refinement）迭代动态更新，兼顾信息独特性与时间连贯性；

**🔧 技术方法**

核心技术包括：查询相关性计算、定向多样性度量、可替代性分数求解以及批量迭代细化；

**📊 数据集**

在四大长视频评测集上验证：MVBench、LongVideoBench、MLVU、VideoMME；

**📈 对比分析**

与均匀采样、BOLT、AKS等基线对比，GIFT在不同帧预算下均表现最优，最优提升达12.5%，在极限预算下仍保持高达93.9%性能；

**⚠️ 局限性**

局限性包括：仍需预先提取候选帧、对极短视频或非问答任务的适用性未知，以及迭代细化的计算开销与超参数B的选择需要经验。

---

## 241. TopoPilot: Reliable Conversational Workflow Automation for Topological Data Analysis and Visualization

**arXiv ID:** 2603.25063 | [PDF](https://arxiv.org/pdf/2603.25063v1)

**作者:** Nathaniel Gorski `[一作]` (University of Utah), Bei Wang `[通讯]` (University of Utah)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

无法确定

**💡 创新点**

无法确定

**🔧 技术方法**

无法确定

**📊 数据集**

无法确定

**📈 对比分析**

无法确定

**⚠️ 局限性**

无法确定

---

## 242. SIGMA: Structure-Invariant Generative Molecular Alignment for Chemical Language Models via Autoregressive Contrastive Learning

**arXiv ID:** 2603.25062 | [PDF](https://arxiv.org/pdf/2603.25062v1)

**作者:** Xinyu Wang `[一作]` (University of Connecticut), Minghu Song `[通讯]` (Chinese Academy of Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出SIGMA框架，利用自回归Transformer和密集轨迹对比学习消除SMILES序列与分子图结构不匹配导致的轨迹发散，并设计IsoBeam搜索动态剔除等价SMILES冗余；

**💡 创新点**

①在自回归生成中实施token级对比学习，实现轨迹级对齐；②通过Probe Suffix和结构正负样本构造功能等价对；③引入投影头与Siamese前向以区分语法与结构；④提出IsoBeam动态剪枝等价路径；⑤实验表明显著提升分布匹配度（FCD）和生成多样性。

**🔧 技术方法**

GPT‑2自回归Transformer、InfoNCE对比学习+投影头、结构正负样本、轨迹对齐损失、RDKit/InChIKey等式子验证、IsoBeam beam搜索。

**📊 数据集**

ZINC‑250k（无监督生成），PubChem/ChEMBL/ZINC（预训练），PMO benchmark（多属性优化任务）。

**📈 对比分析**

与canonical SMILES、随机SMILES、图基模型（GraphAF、MoFlow、LO‑ARM）及序列基线（CharRNN、mGPT2、MolGPT、LTCL）对比；在ZINC‑250k上FCD降至0.752，近似图模型水平；在优化任务中生成更多唯一结构（#Scaf提升约40%），在IsoBeam下结构多样性翻倍。

**⚠️ 局限性**

仍受自回归生成速度限制，IsoBeam需RDKit验证导致推理开销；对极大或复杂分子可能不稳定；依赖Oracle/InChIKey验证；部分模式崩溃仍存在。

---

## 243. Absolute convergence and Taylor expansion in web based models of Linear Logic

**arXiv ID:** 2603.25215 | [PDF](https://arxiv.org/pdf/2603.25215v1)

**作者:** Christine Tasson `[一作]` (Fédération ENAC ISAE-SUPAERO ONERA, Université de Toulouse), Aymeric Walch `[通讯]` (Fédération ENAC ISAE-SUPAERO ONERA, Université de Toulouse)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文提出了一种通用的网络模型构造方法，利用强 PCM 与绝对可加 PCM（允许负系数）统一了多种线性逻辑模型，并在此框架下扩展了可计量泰勒展开，证明包括柯泰空间、有限性空间、相干空间、概率相干空间等在内的多种模型都具备该泰勒展开。

**💡 创新点**

创新点在于：① 引入了绝对可加 PCM 的概念，使得泰勒展开可以适用于负系数的情形；② 在强 PCM 的基础上给出一种通用构造，随后将其推广到绝对收敛的 PCM；③ 通过可代表性与分析共形的双射，统一证明了多种模型的泰勒展开；④ 对指数结构的 Lafont 性质给出了猜想与进一步研究方向。

**🔧 技术方法**

主要技术包括：线性逻辑的 PCM‑enriched 类别、部分可加群、可代表性、张量与闭合结构、分配律、二元结构（单子与余单子）、以及分析共形的构造和双射。

**📊 数据集**

该工作完全是理论性的，没有使用任何实验数据集。

**📈 对比分析**

比较方法：通过与已知模型（如权重关系模型、概率相干空间等）的对应关系进行理论证明，未进行实验性性能评估。

**⚠️ 局限性**

局限性：① 未能证明指数结构在所有网络模型中都是 Lafont；② 缺少无网格（web‑free）模型的例子；③ 对更广泛的绝对可加 PCM（如方便向量空间）的可代表性与泰勒展开尚未给出完整证明。

---

## 244. Enabling Homomorphic Analytical Operations on Compressed Scientific Data with Multi-stage Decompression

**arXiv ID:** 2603.25206 | [PDF](https://arxiv.org/pdf/2603.25206v1)

**作者:** Xuan Wu `[一作]` (Oregon State University), Franck Cappello `[通讯]` (Argonne National Laboratory)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `fede83ac-7505-405f-ab37-e7284695c47f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计了多阶段分块解压的压缩框架HSZ，并实现四个基于SZp/SZx的压缩器，支持在不同解压阶段直接对中间表示执行统计、数值微分和多变量派生等分析操作。

**💡 创新点**

引入多阶段解压与同质运算（homomorphic analytic operations），在不完全解压的情况下对量化、预测残差等中间数据执行分析；在单维与多维变体中实现最优阶段选择；同时保持严格的误差控制。

**🔧 技术方法**

使用 error‑controlled lossy compression（SZp/SZx 及其多维变体），多级解压、基于整数量化和预测残差的同质运算算法、滑动窗口块级 out‑of‑core 处理，以及误差分析证明误差≤ε。

**📊 数据集**

使用五个科学仿真数据集：Ocean、Miranda、Hurricane、NYX、JHTDB（尺寸从 65 MB 到 191 GB）。

**📈 对比分析**

与原始 SZp/SZx 以及 SZOps 进行对比，评估压缩比、解压吞吐量、分析运算吞吐量。结果显示在均值、方差、导数、拉普拉斯、散度、旋度等操作上，HSZx/HSZx‑nd 分别提升至 7315×、1.89×、2.68×；在大规模数据（JHTDB）仍保持 1.7–1.9× 的速度提升，且误差始终低于设定的 ε。

**⚠️ 局限性**

目前仅在预测型压缩器（SZp/SZx）实现；对变换型压缩器需要重新设计同质运算；多维情况下多边界处理导致 D_p 方案性能下降；实现为单核版本，尚未扩展到多核/ GPU。

---

## 245. The Competence Shadow: Theory and Bounds of AI Assistance in Safety Engineering

**arXiv ID:** 2603.25197 | [PDF](https://arxiv.org/pdf/2603.25197v1)

**作者:** Umair Siddique `[一作]` `[通讯]` (Independent Research), Umair Siddique (Independent Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出了安全工程的五维能力框架并引入“能力影子”概念，系统分析人工智能在安全分析中的协作结构与认知机制，给出了四种典型结构的性能下限公式；

**💡 创新点**

创新点在于将安全工程的多维能力与AI偏倚机制统一建模，提出竞争影子（competence shadow）理论，证明协作结构决定AI帮助或削弱安全分析质量；

**🔧 技术方法**

采用形式化理论方法（向量能力模型、乘法性衰减机制、闭式期望质量表达式）与定量参数化的认知偏差模型；

**📊 数据集**

未使用公开数据集，主要基于行业案例与文献综述构建参数（如α_frame、β、η_disagree、γ）进行理论推导；

**📈 对比分析**

通过理论推导与示例计算比较四种协作结构，结果显示串行依赖结构在中等偏差下会降低质量，独立分析与人主导探索能提升或保持质量；

**⚠️ 局限性**

局限在于缺乏实证验证、参数估计方法不成熟、模型假设静态且未考虑学习与动态影子演化。

---

## 246. Vision Hopfield Memory Networks

**arXiv ID:** 2603.25157 | [PDF](https://arxiv.org/pdf/2603.25157v1)

**作者:** Jianfeng Wang `[一作]`, Thomas Lukasiewicz `[通讯]` (Vienna University Of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 Vision Hopfield Memory Network (V-HMN)，一种以关联记忆为核心的视觉基础模型，通过局部和全局 Hopfield 模块与可学习的预测编码迭代细化实现图像特征的重构与提升。

**💡 创新点**

创新点在于：①将局部窗口与全局场景记忆完全替代自注意力作为特征混合机制；②引入可学习的迭代细化规则，模仿预测编码的误差校正；③采用类平衡的实时记忆库，在训练时写入真实样本嵌入、推理时冻结，提升数据效率与可解释性。

**🔧 技术方法**

技术手段包括：Hopfield 风格的关联记忆检索、L2 归一化与 √D 缩放的相似度计算、可学习的迭代细化公式、注意力池化做全局表征、以及多路（局部/全局）记忆融合。

**📊 数据集**

主要使用的公开数据集有 CIFAR‑10、CIFAR‑100、SVHN、Fashion‑MNIST 以及 ImageNet‑1k，评估分类性能。

**📈 对比分析**

与 ViT、Swin‑ViT、MLP‑Mixer、MetaFormer、Vim、AiT 等主流基础模型进行对比。V-HMN 在 CIFAR‑10、CIFAR‑100、SVHN、Fashion‑MNIST 上均取得最高或接近最高的 Top‑1 准确率，并在 10%/30% 少量数据下显著优于同类模型；在 ImageNet‑1k 上也能达到与大模型相当的 80.3% Top‑1，显示出良好的可扩展性。

**⚠️ 局限性**

局限性包括：① 迭代细化需要额外的前向和后向计算，虽少量但仍比单步注意力略慢；② 记忆库尺寸固定，可能在极大类别数下记忆容量不足；③ 目前仅在分类任务验证，缺乏对检索、分割、检测等更复杂任务的实验；④ 对超参数（记忆槽数、迭代步数、更新强度）敏感，需要进一步自动调优。

---

## 247. Dissimilarity-Based Persistent Coverage Control of Multi-Robot Systems for Improving Solar Irradiance Prediction Accuracy in Solar Thermal Power Plants

**arXiv ID:** 2603.25139 | [PDF](https://arxiv.org/pdf/2603.25139v1)

**作者:** Haruki Kawase `[一作]` (University of Osaka), A. Daniel Carnerero `[通讯]` (University of Osaka)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种基于差异度图的持久覆盖控制方法，用移动传感器自适应采样以提升集中式太阳能热电站的辐射预测精度。

**💡 创新点**

创新点在于将Kernel‑Based Kriging（KB‑Kriging）的差异度函数映射为时空重要性函数，进而指导移动传感器前往高不相似区域；设计了结合差异度的持久覆盖控制器，并通过实验验证其优越性。

**🔧 技术方法**

使用了Kernel‑Based Kriging、Gaussian核的差异度函数、持久覆盖控制理论、移动机器人（TurtleBot3）与ROS的硬件在环实现，以及MATLAB/ROS Noetic进行实验与参数优化。

**📊 数据集**

采用了J. G. Martin等人提供的97 × 72空间网格、2000时间步的云雾条件太阳辐射数据集。

**📈 对比分析**

方法与固定传感器布局（Voronoi中点）和基线持久覆盖（ϕ = 1）进行对比，使用RMSE和平均RMSE评估。实验结果显示，提出方法在时间序列RMSE和平均RMSE上均优于两者，并在不同天气、不同传感器数量下保持较好性能。

**⚠️ 局限性**

局限性：验证仅在小规模实验室场景，通信假设为全连通且采用中心化计算，难以直接扩展到规模达780 ha的大型CST厂区；未解决分布式实现、障碍物、非holonomic约束等实际部署问题。

---

## 248. Semi-Automated Generation and Hemodynamic Assessment of Surgical Baffle Geometry for Biventricular Repair

**arXiv ID:** 2603.25207 | [PDF](https://arxiv.org/pdf/2603.25207v1)

**作者:** Elena Sabdy Martinez `[一作]` (Stanford University), Alison Lesly Marsden `[通讯]`

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

构建了一套半自动化计算框架，用于为双主流右室患者设计并评估心室间隔瓣膜

**💡 创新点**

创新点在于通过目标截面积约束、PCHIP中心线构造、凸包平滑和TPS表面插值，实现瓣膜在保持右心室几何的前提下产生可计算流体动力学的通道

**🔧 技术方法**

使用技术包括nnU-Net语义分割、SimVascular VTMK、PCHIP插值、Thin-Plate Spline、稳态Navier–Stokes CFD

**📊 数据集**

数据集为四例双主流右室患者的术前CT影像与术后超声测量

**📈 对比分析**

通过与术后超声测得的压力梯度对比，预测的压力下降在2–5 mmHg，均低于临床测量值，显示出更优的血流动力学

**⚠️ 局限性**

局限在于界面选择仍需人工、仅采用稳态刚壁模型、未考虑瓣膜材料与心室动力学耦合

---

## 249. SafeMath: Inference-time Safety improves Math Accuracy

**arXiv ID:** 2603.25201 | [PDF](https://arxiv.org/pdf/2603.25201v1)

**作者:** Sagnik Basu `[一作]` (Indian Institute of Technology Kharagpur), Animesh Mukherjee `[通讯]` (Indian Institute of Technology Kharagpur)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究数学文字题中嵌入的有害叙事，并提出在推理时通过在潜在空间插入对策向量实现安全与数学正确性的双重优化。

**💡 创新点**

首次构建针对数学文字题的有害内容数据集，并通过对比向量（safe 与 math）实现推理时安全对齐而不牺牲甚至提升数学推理精度。

**🔧 技术方法**

使用对比学习+主成分分析生成的 in‑context 向量，结合缩放系数 α、β 在所有层的隐藏状态中注入，实现推理时的安全引导。

**📊 数据集**

主要数据集为自构建的 SafeMath（约 1.9k 题）以及原始 GSM8K 过滤后的 2–3 步子集，用于生成有害和无害题目。

**📈 对比分析**

与基础模型、仅安全提示、少量示例提示等基线对比，评估准确率与安全性（Beaver 成本模型）。结果显示安全向量单独使用可提升 2–3% 正确率，双向量组合可进一步提升 4–5% 并大幅降低有害输出。

**⚠️ 局限性**

局限：仅在 7B–8B 规模模型上验证，未测试更大模型；两向量线性融合可能不充分，缺乏非线性组合；性能对超参数 α、β 敏感，需要精细调优。

---

## 250. On-Demand Instructional Material Providing Agent Based on MLLM for Tutoring Support

**arXiv ID:** 2603.25195 | [PDF](https://arxiv.org/pdf/2603.25195v1)

**作者:** Takumi Kato `[一作]` (Nagoya Institute of Technology), Tadachika Ozono `[通讯]` (Nagoya Institute of Technology)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一款基于多模态大型语言模型的即时教学材料代理，能够在师生对话中自动生成搜索查询并检索相关 Web 图片。

**💡 创新点**

创新点在于：①使用 MLLM 从语音对话实时提炼上下文，自动生成搜索词；②完全不需要人工发起搜索请求，保持对话连贯；③通过实验验证显著降低检索时延并实现高满意率。

**🔧 技术方法**

使用的技术包括：多模态大语言模型 Gemini 1.5 Flash、语音转文本、Web 图片检索 API、实时交互式界面；核心流程为语音→文本→查询生成→检索→展示。

**📊 数据集**

数据来源为实验室中教师与学生录制的对话音频，涵盖 7 个地球科学主题；检索结果基于公开 Web 图片，无特定标注数据集。

**📈 对比分析**

与传统人工 Web 搜索相比，实验1显示平均检索时间缩短 44.4 秒，实验2的成功率达 85.7%，表明系统在实用性和效率上均优于人工方式。

**⚠️ 局限性**

局限性包括：假设对话无话题切换；检索结果质量仍不及人类搜索；缺乏过滤与安全机制；对长时间部署的成本与可扩展性未充分评估。

---

## 251. zk-X509: Privacy-Preserving On-Chain Identity from Legacy PKI via Zero-Knowledge Proofs

**arXiv ID:** 2603.25190 | [PDF](https://arxiv.org/pdf/2603.25190v1)

**作者:** Yeongju Bak `[一作]` `[通讯]` (Tokamak Network), Yeongju Bak (Tokamak Network)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出zk-X509，一种基于零知识证明的身份系统，将现有X.509 PKI证书桥接至公链，实现身份验证的隐私与合规双重保障。

**💡 创新点**

核心创新在于：1）仅使用已发行的政府级X.509证书，无需新建发行基础设施；2）私钥永不进入zkVM，采用OS密钥链签名；3）通过Merkle树隐藏CA身份，实现CA匿名；4）在zkVM内部完成链式验证、CRL检查与所有权证明。

**🔧 技术方法**

采用SP1 RISC‑V zkVM、Groth16 succinct proof、RSA/ECDSA签名验证、SHA‑256散列、Merkle树与OS密钥链（macOS Secure Enclave/Windows TPM）集成。

**📊 数据集**

以韩国NPKI（约2千万份）真实国家级X.509证书为主数据集，辅以自制测试证书进行功能与性能验证。

**📈 对比分析**

与DID/VC、zk‑email、Polygon ID、Semaphore、zkPassport、Worldcoin等方案对比，zk-X509在零知识、无硬件依赖、CA匿名性方面突出；单级P‑256链式验证耗时约11.8M SP1周期（RSA约17.4M），链上Groth16验证费用约30万Gas。

**⚠️ 局限性**

局限包括：证明生成耗时约5分钟CPU、需本地离线证明器、CRL提供方受信任限制、跨链身份关联无法实现、对主动共享凭证缺乏防护、以及对多签治理与实时OCSP支持仍待完善。

---

## 252. A Catalog of Basque Dialectal Resources: Online Collections and Standard-to-Dialectal Adaptations

**arXiv ID:** 2603.25189 | [PDF](https://arxiv.org/pdf/2603.25189v1)

**作者:** Jaione Bengoetxea `[一作]` (University of Basque Country), Rodrigo Agerri `[通讯]` (University of Basque Country)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

系统整理并公开了巴斯克语方言数据资源，提出了手工并行适配的XNLIvar扩展数据集，并对自动适配的BasPhyCowest进行手工评估。

**💡 创新点**

首次提供完整的手工并行NLI方言数据集，并设计了对自动方言适配效果的对比评估框架。

**🔧 技术方法**

结合手工标注、LLM（Latxa-It-70B）多样本提示自动适配、Levenshtein距离计算以及Cohen's Kappa等评估指标。

**📊 数据集**

使用XLIvar、Parallel XNLIvar、BasPhyCo、BasPhyCowest及其新版本BasPhyCowest-new等数据集。

**📈 对比分析**

通过计算标准与方言文本的Levenshtein距离、对句子对进行手工评估并计算IAA，结果表明非标准正字法版本更具方言特征，IAA达到0.71。

**⚠️ 局限性**

部分资源受版权限制、自动适配仅覆盖西部方言、评估中仍存在错误和评注者偏差。

---

## 253. Factors Influencing the Quality of AI-Generated Code: A Synthesis of Empirical Evidence

**arXiv ID:** 2603.25146 | [PDF](https://arxiv.org/pdf/2603.25146v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 254. Knowledge-Guided Adversarial Training for Infrared Object Detection via Thermal Radiation Modeling

**arXiv ID:** 2603.25170 | [PDF](https://arxiv.org/pdf/2603.25170v1)

**作者:** Shiji Zhao `[一作]` (Beihang University), Xingxing Wei `[通讯]` (Beihang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `6215c339-3735-4be3-8a07-5bbb7004712d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了基于相对热辐射关系的知识指导对抗训练方法（KGAT），提升红外目标检测在对抗攻击和常见噪声下的鲁棒性。

**💡 创新点**

创新点在于将红外图像的物理知识——不同类别的相对热辐射排名关系及其稳定性度量，融入对抗训练的重加权机制，实现知识驱动的鲁棒优化。

**🔧 技术方法**

使用了光谱辐射模型、灰度值排名、斯皮尔曼相关系数重加权、热辐射关系稳定性度量、PGD对抗采样等技术。

**📊 数据集**

实验使用了三大红外多分类数据集（M^3FD、FLIR-ADAS、KAIST）以及多种检测框架（YOLO‑v8、Faster R‑CNN、SSD、DINO）。

**📈 对比分析**

与标准训练、纯对抗训练（LOC、CLS、MTD、CWAT）以及结合架构鲁棒设计（SALC、RobustNet、Det‑AdvProp）进行对比，KGAT在保持清晰图像精度的同时显著提升对抗攻击（如MTD、PGD、CWA、DAG等）和常见噪声（Gaussian、Salt、Blur）下的 mAP_50，平均提升约 7–10% 以上。

**⚠️ 局限性**

限制在于热辐射关系不稳定或类别数量有限时，KGAT 的鲁棒提升有限；极端遮挡、温度极差或强噪声环境下仍会受影响。

---

## 255. UniAI-GraphRAG: Synergizing Ontology-Guided Extraction, Multi-Dimensional Clustering, and Dual-Channel Fusion for Robust Multi-Hop Reasoning

**arXiv ID:** 2603.25152 | [PDF](https://arxiv.org/pdf/2603.25152v1)

**作者:** Jie Wang `[一作]` (China Unicom), Shiguo Lian `[通讯]` (China Unicom)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 UniAI‑GraphRAG 框架，结合本体驱动知识抽取、多维社区聚类和双通道检索融合，以提升垂直领域多跳推理与检索性能。

**💡 创新点**

三大创新点：① 基于预定义本体的抽取约束降低噪声；② 引入属性感知模块与边界完成机制的多维社区聚类；③ 设计图谱检索与社区报告检索双通道动态加权融合。

**🔧 技术方法**

使用本体约束生成式抽取、属性感知模块（α 权重），基于深度递归的多跳子图划分，Trie 结构实体检索，向量+语义相似度社区检索，交叉编码重排序与互信息最大化。

**📊 数据集**

在 MultiHopRAG（含推理、比较、时间三类问答）基准集上进行评估，实验中还使用 MultiHopQA 进行消融验证。

**📈 对比分析**

与 Dify RAG、Wanwu RAG 及 LightRAG 比较，UniAI‑GraphRAG 在所有问答类型上均取得最高 F1（推理 90.23、比较 68.54、时间 52.67），相较 LightRAG 提升 2.23% F1，较 Dify 提升 22.45%，消融实验显示每项创新均带来 3–4% 的 F1 提升。

**⚠️ 局限性**

局限性包括：本体与模式需人工专家定义成本高、仅支持文本输入不含多模态数据、分块抽取导致语义碎片化可能影响知识完整性。

---

## 256. Goodness-of-pronunciation without phoneme time alignment

**arXiv ID:** 2603.25150 | [PDF](https://arxiv.org/pdf/2603.25150v1)

**作者:** Jeremy H. M. Wong `[一作]` (Institute for Infocomm Research, A*STAR), Nancy F. Chen `[通讯]` (Institute for Infocomm Research, A*STAR)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文提出使用弱监督多语种 ASR（如 Whisper）生成混淆网络（CN），从中提取音素后验、GOP、LPP、LPR 等特征，并结合跨注意力（cross‑attention）架构，无需音素时间对齐即可完成读音评估。

**💡 创新点**

创新点在于：①利用 CN 替代传统强监督对齐获得音素后验；②把词级 speaking rate / duration 作为特征替代音素级；③采用跨注意力融合音素级与帧级特征；④完全无需训练目标语言的 ASR，直接使用公开的多语种弱监督模型扩展低资源语言评估。

**🔧 技术方法**

技术细节包括：Whisper 的 N‑best 解码、文本标准化、转写、词转音素（词典 + G2P）、CN 构建与 CN‑GOP/LPP/LPR 计算；跨注意力 Transformer decoder 结合 SSL（wav2vec2.0 XLSR‑53）、pitch、语义嵌入等特征；SVR 进行单音素评分，Transformer encoder 进行句子级评分。

**📊 数据集**

实验数据集为 English 的 speechocean762（2500 句）和低资源 Tamil 内部数据（2118/2093 句）。基准 ASR 为混合神经网络‑隐马尔可夫模型，Whisper 作为弱监督模型进行特征提取。

**📈 对比分析**

通过 PCC 与 MSE 与基准 phoneme‑pooling 模型对比，发现：在 English 上与基准相当；在 Tamil 上性能优于基准（PCC 提升，MSE 降低）；CN 解码相比单 1‑best WER 更低；跨注意力在无 ASR 训练数据时也能取得可比效果。

**⚠️ 局限性**

局限性包括：缺乏音素时间对齐导致特征稀疏；CN 取决于 N‑best 质量与词典/ G2P 的准确性；词级 GOP 无显著提升；对非拉丁脚本的转写与转写不完善；仍需目标语言的评估标注数据，且对极低资源场景下 Whisper 语言覆盖有限。

---

## 257. Trace2Skill: Distill Trajectory-Local Lessons into Transferable Agent Skills

**arXiv ID:** 2603.25158 | [PDF](https://arxiv.org/pdf/2603.25158v1)

**作者:** Jingwei Ni `[一作]` (ETH Zürich), Guanjun Jiang `[通讯]` (Alibaba)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种并行多智能体补丁生成与层级合并的技能演化框架，能够从大量轨迹中自动创建或深化技能；

**💡 创新点**

创新点在于：①将所有轨迹局部补丁一次性并行合并，避免顺序更新导致的漂移；②采用代理式错误分析和成功分析，提升补丁可迁移性；③利用层级合并实现自我监督的归纳推理与冲突消除；

**🔧 技术方法**

技术包括：ReAct式多轮代理、并行子代理补丁生成、层级合并合并器（ℳ）、程序化冲突检测与格式校验、基于LLM的轨迹生成与补丁编辑；

**📊 数据集**

使用的数据集有：SpreadsheetBench‑Verified（训练/测试）、WikiTableQuestions（OOD迁移）、DAPO‑Math‑Train/​Test 与 AIME2026（数学推理）、DocVQA（视觉问答）；

**📈 对比分析**

通过与人写技能、参数化技能、ReasoningBank检索库、顺序更新等基线对比，实验显示在SpreadsheetBench中平均提升17–22个百分点，在WikiTQ迁移中提升1–5个百分点，数学与VQA任务同样获得正向提升；并行合并相比顺序更新在性能与计算效率上都有显著优势；

**⚠️ 局限性**

限制包括：缺乏单个补丁的因果贡献量化、缺少对技能各部分使用频率的跟踪与归因，导致难以精确评估每条补丁或章节的实际价值。

---

## 258. CIV-DG: Conditional Instrumental Variables for Domain Generalization in Medical Imaging

**arXiv ID:** 2603.25202 | [PDF](https://arxiv.org/pdf/2603.25202v1)

**作者:** Shaojin Bai `[一作]` (Tianjin University), Weizhi Nie `[通讯]` (Tianjin University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

针对医学影像跨站点泛化问题，提出通过条件工具变量(CIV)从数据产生机制中剔除医院选择偏倚导致的混淆，训练出与病理特征因果相关的鲁棒模型。

**💡 创新点**

创新点在于：①将条件工具变量理论引入域泛化，放宽传统工具变量对完全随机性的要求；②设计了基于 DeepGMM 的条件矩约束学习框架，使用分层指数滑动平均实现条件期望估计；③在同一框架下兼顾 OOD 鲁棒性、公平性和校准性。

**🔧 技术方法**

技术方法包括：条件工具变量理论、条件矩约束、DeepGMM（深度泛化方法），利用 Spectral Normalization + 经验矩约束实现预测器与 critic 的对抗式训练；使用冻结的 ViT 编码器与 Causal Adapter 进行特征提取。

**📊 数据集**

数据集：Camelyon17-WILDS（合成的站点与年龄分层）与 Multi-Source Chest X‑Ray（NIH‑CXR14、CheXpert、MIMIC‑CXR）并结合真实的年龄、性别分层信息。

**📈 对比分析**

与 SWAD、LISA、DFR（DG 基线）、CLIP、MedCLIP、AdFair‑CLIP 等方法进行对比，CIV‑DG 在 Camelyon17 的 Accuracy（93.24%）、Worst‑Group Accuracy（70.46%）和 ECE（8.73%）以及 CXR 的 AUC（84.15%）、EOD（6.81%）、DPD（6.78%）均显著优于所有基线，说明在 OOD 鲁棒性、准确性和公平性上均领先。

**⚠️ 局限性**

局限性包括：①需要先验的分层变量（如年龄、性别）才能实现条件工具变量；②对连续或高维分层变量的处理尚未系统化；③实验仅覆盖影像数据，未验证在多模态或非影像医学记录中的泛化能力。

---

## 259. A Decade-Scale Benchmark Evaluating LLMs' Clinical Practice Guidelines Detection and Adherence in Multi-turn Conversations

**arXiv ID:** 2603.25196 | [PDF](https://arxiv.org/pdf/2603.25196v1)

**作者:** Andong Tan `[一作]` (Microsoft Research Asia), Hao Chen `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了CPGBench基准框架，用于评估大型语言模型在多轮对话中检测与遵循临床实践指南（CPG）的能力，并通过自动化生成对话与Judge‑LLM评估结合人工审核来验证结果。

**💡 创新点**

①首次构建跨24个医学专科、9个地区、十年尺度的完整CPG检测与遵循基准；②引入双任务（检测+标题定位与多轮遵循）并实现自动化对话生成与评估；③使用Judge‑LLM实现大规模自动评估并与人工一致性对照；④针对安全关键指南建立子集评估，揭示模型在关键场景下的不足。

**🔧 技术方法**

利用大型预训练语言模型（如GPT‑4o、DeepSeek‑R1、Qwen3系列、Baichuan‑m2、GPT5、Llama3‑8B‑inst、Huatuo‑o1‑7B）进行文档信息抽取、对话生成与评估；正则表达式与格式约束验证对话；Judge‑LLM作为自动评分器；Cohen’s κ 与卡方检验用于统计与一致性分析；PDF/HTML文本提取工具（Pymupdf、BeautifulSoup）进行数据预处理。

**📊 数据集**

收集并筛选了3,418篇高质量CPG文档（2015年至今），从中提取32,155条临床推荐；生成对应多轮对话；进一步筛选出6,632条安全关键子集。原始来源包括国家卫生部门、专业医学协会、国际组织官网以及ECRI等平台。人工验证样本包括56位临床医生。

**📈 对比分析**

对8款LLM在检测、标题定位与遵循三项任务进行量化评估：检测率71–89%，标题定位率4–30%，遵循率21–63%；安全关键子集检测率更高但遵循率更低，揭示“知道–做”缺口。统计显著性检验显示不同专科与地区对模型表现影响显著。整体性能显示虽然模型能识别指南，但在实际应用与引用上仍存在显著不足。

**⚠️ 局限性**

自动化流程中错误可能累积；Judge‑LLM评估结果受阈值与主观性影响；生成对话场景有限，未覆盖所有真实临床对话；对指南抽取与标注可能出现误差；模型对新发布指南的更新滞后；数据来源多为官方英文文献，中文/非官方资料缺失；安全关键评估仍受限于子集规模与标注准确性。

---

## 260. AnyID: Ultra-Fidelity Universal Identity-Preserving Video Generation from Any Visual References

**arXiv ID:** 2603.25188 | [PDF](https://arxiv.org/pdf/2603.25188v1)

**作者:** Jiahao Wang `[一作]` (Xi’an Jiaotong University), Jieping Ye `[通讯]` (Alibaba Cloud Computing)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 AnyID 框架，实现多来源（面部、肖像、视频）身份保持的视频生成，支持差异化提示控制

**💡 创新点**

创新点包括：可扩展的 omni‑referenced 统一编码架构、主参照+差异化提示的生成范式，以及基于人类偏好数据的强化学习优化

**🔧 技术方法**

技术主要涵盖：基于 Diffusion Transformer 的流模型、VAE 直接编码、差异化提示生成、Direct Preference Optimization (DPO) 强化学习

**📊 数据集**

使用 PortraitGala 数据集进行大规模标注和多样化参考生成，并在此基础上构建 100k ID‑group、300k 视频样本

**📈 对比分析**

与 ConsisID、FantasyID、Phantom、SkyReels‑A2 等基线对比，在身份保真、元素一致性、提示可控性和视觉质量四维度均取得最高分，尤其在身份保真上达 73.22%

**⚠️ 局限性**

局限性：仍依赖人工偏好对比数据，差异化提示生成过程需多阶段 VLM+LLM 处理，单一 VAE 编码可能限制极端多模态输入的表达能力

---

## 261. Probing the Lack of Stable Internal Beliefs in LLMs

**arXiv ID:** 2603.25187 | [PDF](https://arxiv.org/pdf/2603.25187v1)

**作者:** Yifan Luo `[一作]` (Tsinghua University), Andrew Chi-Chih Yao `[通讯]` (Tsinghua University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过在20题猜谜游戏中让LLM隐式选择目标并持续回答是/否，评估其在多轮对话中的隐式一致性（即内部目标的持久性）

**💡 创新点**

首次正式定义并量化“隐式一致性”，提出统一的探测框架和基于KL散度的正则化训练方法来缓解目标漂移

**🔧 技术方法**

采用数值索引探测技术、KL散度正则化损失、外部一致性验证（LLM-as-Judge）以及对话模拟生成与评估

**📊 数据集**

使用自行构造的两类对话数据集：数值猜测（10个0-99随机数）与实体猜测（10个跨类别实体），通过模型对话生成和探测获取内部信念分布

**📈 对比分析**

对比GPT‑4o、Deepseek‑v3.1、Claude‑3.7‑Sonnet等主流LLM，在多轮实验中测量Drift Rate、KL变化和External Consistency；结果显示多数模型Drift Rate接近100%，仅通过KL正则化的微调（Qwen‑2.5‑14B）显著降低漂移率至约14%

**⚠️ 局限性**

当前方法仍无法完全消除隐式漂移，推理增强模型可能加剧漂移；需要更复杂的记忆或显式信念追踪机制来实现长期目标一致性

---

## 262. Knowledge-Guided Retrieval-Augmented Generation for Zero-Shot Psychiatric Data: Privacy Preserving Synthetic Data Generation

**arXiv ID:** 2603.25186 | [PDF](https://arxiv.org/pdf/2603.25186v1)

**作者:** Adam Jakobsen `[一作]` (SimulaMet), Vajira Thambawita `[通讯]` (SimulaMet)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一个零样本、基于知识引导的检索增强生成框架，用以合成精神病学问卷数据。

**💡 创新点**

创新点在于利用DSM‑5和ICD‑10作为检索知识库，结合大语言模型实现无数据训练的高保真合成数据。

**🔧 技术方法**

使用了大语言模型（Mistral 7B）+检索增强生成（RAG）以及Faiss向量检索、知识上下文注入。

**📊 数据集**

使用了公开的OSF精神科问卷数据（包含564名成人的DSM‑5强度量表），并在六种焦虑障碍上进行实验。

**📈 对比分析**

与CTGAN、TVAE等基于真实数据的生成模型对比，零样本LLM在对偶关系上与TVAE相当、在单变量/多变量上略逊，但在隐私指标上更安全且无重叠。

**⚠️ 局限性**

局限包括仅建模单一障碍而未考虑共病、数据空间有限导致隐私度量偏高、未进行对抗性攻击评估，且统计精度仍不及传统模型。

---

## 263. Cross-Preference Learning for Sentence-Level and Context-Aware Machine Translation

**arXiv ID:** 2603.25183 | [PDF](https://arxiv.org/pdf/2603.25183v1)

**作者:** Ying Li `[一作]` (Soochow University), Daimeng Wei `[通讯]` (Huawei)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种跨偏好学习（Cross-Preference Learning, CPL）框架，用于在句子级和文档级机器翻译（上下文感知翻译）之间自适应地选择使用上下文，以提升翻译质量和鲁棒性。

**💡 创新点**

创新点在于：①将句子级与上下文级翻译视为双条件问题，构造跨条件与同条件偏好对，形成统一的偏好学习目标；②通过跨条件偏好学习，使得不同输入条件下的翻译输出共享偏好结构，显式建模何时使用上下文；③完全在训练目标层面实现，无需改动模型架构。

**🔧 技术方法**

采用的技术包括：对生成的多种候选翻译进行自动评估（COMET + d-COMET）来构造偏好对；对同条件和跨条件偏好对分别使用 Contrastive Preference Optimization (CPO) 进行对比学习；在多语言多模型（Qwen3-4B/8B、Llama-3-8B）上进行 fine‑tune 并使用 CPL 进行优化。

**📊 数据集**

数据集：News Commentary v18.1（六个语言对）用于主实验；IWSLT 2017 用于上下文重要性评估；小规模连续句子块用于 CPL 的候选样本采样；验证集用于构造偏好对。

**📈 对比分析**

与多种基线（句子级、上下文级、联合 fine‑tune、CPO 单条件等）进行对比，CPL 在所有模型和语言对上平均提升 COMET 1–1.1 分，句子级和上下文级翻译均获得显著提升；在文档级 COMET、BLEU、Faithfulness、Coherence、Fluency 等指标上也表现最优；在上下文噪声与不同评价指标下仍保持鲁棒性。

**⚠️ 局限性**

局限性包括：①依赖自动评估指标构造偏好对，可能继承指标偏见；②需要多次生成候选并计算分数，训练成本较高；③对极长文档或复杂篇章结构的处理尚未充分验证；④实验范围仅限于六个语言对和三种 LLM，未覆盖更广泛的语言和模型规模。

---

## 264. FD$^2$: A Dedicated Framework for Fine-Grained Dataset Distillation

**arXiv ID:** 2603.25144 | [PDF](https://arxiv.org/pdf/2603.25144v1)

**作者:** Hongxu Ma `[一作]` (Zhejiang University), Zhihui Wang `[通讯]` (Dalian University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文针对细粒度图像分类任务，提出了一种专门的样本集蒸馏框架 FD^2，实现了对细粒度数据的高质量蒸馏。

**💡 创新点**

创新点在于引入细粒度特征约束与相似性约束，并利用反事实注意力学习（CAL）获得类原型与精准注意力，从而同时提升类内紧凑度与类间可分性。

**🔧 技术方法**

技术手段包括反事实注意力学习（CAL）、类原型对齐/排斥约束、注意力多样性约束，以及在现有分离式蒸馏流程中的插件式整合。

**📊 数据集**

实验使用细粒度数据集 CUB-200-2011、FGVC-Aircraft、Stanford Cars 以及通用子集 ImageNette、ImageWoof 进行评估。

**📈 对比分析**

与 SRe^2L++、FADRM+ 等基线相比，FD^2 在细粒度数据集上平均提升 10–15% 的 Top‑1 准确率，在跨架构和一般数据集也能获得小幅提升。

**⚠️ 局限性**

局限性包括对低容量模型或简单数据集的提升有限，以及需要额外的 CAL 预训练和超参数调优来保持性能。

---

## 265. EgoXtreme: A Dataset for Robust Object Pose Estimation in Egocentric Views under Extreme Conditions

**arXiv ID:** 2603.25135 | [PDF](https://arxiv.org/pdf/2603.25135v1)

**作者:** Taegyoon Yoon `[一作]` (Seoul National University), Hyung-Sin Kim `[通讯]` (Seoul National University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并公开了 EgoXtreme 这一极端条件下的 egocentric 6D pose 数据集，并在该数据集上对主流 RGB‑only 零射测量模型进行基准评估；

**💡 创新点**

首次提供大规模、极端光照/烟雾/高速运动环境下的 egocentric 6D pose 数据与评测，揭示单帧方法的脆弱性并强调时序融合策略的重要性；

**🔧 技术方法**

采用 FoundPose、GigaPose、PicoPose 等零射测量模型，结合图像去模糊、去雾、低光增强等预处理，以及直接、融合、混合三种时序跟踪策略；

**📊 数据集**

使用 EgoXtreme 数据集（约 1.3M 帧，775.5 分钟，15 位参与者，13 个对象，涵盖工业维护、运动和紧急救援三大场景）；

**📈 对比分析**

通过 ADD(-S) recall、MSSD、MSPD 等 BOP 指标比较模型在标准与极端条件下的性能，发现极端光照/烟雾下性能下降显著，混合时序跟踪能显著提升准确率；

**⚠️ 局限性**

主要限制包括：标注依赖室内 OptiTrack 系统，难以验证户外鲁棒性；缺乏 3D 手部姿态标注；图像预处理方法对 pose 估计效果不佳，需进一步研究更强的鲁棒模型。

---

## 266. RubricEval: A Rubric-Level Meta-Evaluation Benchmark for LLM Judges in Instruction Following

**arXiv ID:** 2603.25133 | [PDF](https://arxiv.org/pdf/2603.25133v1)

**作者:** Tianjun Pan `[一作]` (Fudan University), Yanghua Xiao `[通讯]` (Fudan University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 RubricEval，一个针对指令跟随的基于规则级的元评估基准，覆盖四类指令并划分易/难子集，并设计了 Rubric Arbitration Framework (RAF) 进行大规模高质量标注。

**💡 创新点**

首次在指令跟随评估中实现规则级元评估；提出 RAF 通过多模型共识与元评审实现高可信度标注；系统分析判定器在不同规则维度上的失效模式。

**🔧 技术方法**

使用大型语言模型作为判定器，采用共识与元评审技术构建 RAF；对比 rubric‑level 与 checklist‑level 评估、显式推理 vs 直接判断；统计 Balanced Accuracy、Macro F1 等指标。

**📊 数据集**

RubricEval 数据集：3,486 条规则级评估实例，涵盖 Constrained、Compositional、Multi‑turn、System 四类指令，来自多基准，使用真实 LLM 响应，并划分 2,034 易、1,452 难子集。

**📈 对比分析**

将多种开源与商用 LLM 判定器（GPT‑4o、Claude‑4.5、Qwen3‑235B、gpt‑oss‑120b 等）在 Easy 与 Hard 子集上进行评测，得到平衡准确率与宏观 F1；结果显示即使是 GPT‑4o 在难子集也仅达 55.97% 平衡准确率，说明规则级评估仍具挑战。

**⚠️ 局限性**

只覆盖四类指令，未涵盖代理或领域特定指令；RAF 仍依赖 LLM 生成标注，存在标注噪声；对冲突实例直接丢弃，可能遗漏真正难题；仅研究二元规则评估，未涵盖 Likert 级别或对比评估。

---

## 267. A Wireless World Model for AI-Native 6G Networks

**arXiv ID:** 2603.25216 | [PDF](https://arxiv.org/pdf/2603.25216v1)

**作者:** Ziqi Chen `[一作]` (China Mobile Research Institute), Liang Xia `[通讯]` (China Mobile Research Institute)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `afceb026-1760-41ae-8d86-010831a37d97` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出并实现了Wireless World Model（WWM），一种融合CSI、三维点云和用户轨迹的多模态预训练基础模型，用于6G无线网络的多任务预测与优化。

**💡 创新点**

主要创新在于：①构建800k样本的光线追踪与真实测量混合数据集；②采用Joint Embedding Predictive Architecture（JEPA）将自监督预测任务迁移至潜在空间；③引入多模态Mixture-of-Experts Transformer，实现环境感知、物理一致性和多任务通用性。

**🔧 技术方法**

技术手段包括自监督多模态预训练（JEPA）、多模态Mixture-of-Experts Transformer、专属CSI/点云/轨迹嵌入、光线追踪仿真与真实测量融合。

**📊 数据集**

使用Sionna RT光线追踪生成的约700k条城市场景CSI、点云和轨迹样本（Munich、Paris、Beijing Forbidden City、Beijing CBD、Wall Street等五城），以及中国移动6G原型机采集的真实测量数据。

**📈 对比分析**

与传统单模态或专用模型对比，WWM在CSI时序预测、压缩反馈、波束预测、定位和真实频域预测等四大下游任务中均取得显著提升，平均SGCS提升10–20%，NMSE降低约20%，并在未见城市和速度下表现出更强的泛化能力。

**⚠️ 局限性**

主要限制在于对高质量3D点云和轨迹信息的依赖，实时获取仍具挑战；此外，模型的计算量和推理延迟（约8.5 ms）在大规模多用户干扰场景中仍需进一步优化。

---

## 268. A CDF-First Framework for Free-Form Density Estimation

**arXiv ID:** 2603.25204 | [PDF](https://arxiv.org/pdf/2603.25204v1)

**作者:** Chenglong Song `[一作]` (University of Jinan), Bo Yang `[通讯]` (Quancheng Shandong Laboratory)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `40105733-5154-44cd-8090-a8cab9e64b07` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了先估计累计分布函数(CDF)再求导得到概率密度的框架，以解决自由形式条件密度估计的数学不稳定性。

**💡 创新点**

核心创新是将密度估计转向 CDF，利用光滑极小极大(SMM)网络保证 CDF 单调性与可微性，并通过有限差分恢复 PDF，从而几乎不加先验偏置。

**🔧 技术方法**

使用 Smooth Min‑Max 网络、蒙版自回归结构、噪声正则化以及有限差分求导来实现可训练且可计算似然的 CDF 模型。

**📊 数据集**

在四个二维 toy 任务（含离散方块、半高斯、旋转高斯棒、弹性环）以及七个 UCI 实际回归数据集（Fish, Concrete, Energy, Parkinsons, Temperature, Air, Skillcraft）上进行实验。

**📈 对比分析**

与 MDN、MAF、NSF、RNF、DDN 等基线对比，CDF‑First 在所有 toy 任务中取得最低 SSE，七个 UCI 数据集的负对数似然多半优于基线，尤其在多模态和拓扑复杂场景中表现显著提升。

**⚠️ 局限性**

主要限制是依赖自回归拆分，导致采样顺序化且扩展到更高维输出时效率低下。

---

## 269. Probabilistic Concept Graph Reasoning for Multimodal Misinformation Detection

**arXiv ID:** 2603.25203 | [PDF](https://arxiv.org/pdf/2603.25203v1)

**作者:** Ruichao Yang `[一作]` (University of Science and Technology Beijing), Xu-Cheng Yin `[通讯]` (University of Science and Technology Beijing)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Probabilistic Concept Graph Reasoning（PCGR）框架，用概率概念图对多模态谣言进行可解释的分层推理。

**💡 创新点**

创新点：①通过多模态大语言模型自动生成并迭代更新概念集合，实现对新谣言手法的自适应；②构建层次化的概率概念图，将低层证据逐级推向高层结论；③采用软化的 Graph‑of‑Thought 推理与层次化注意力，兼顾可解释性与鲁棒性。

**🔧 技术方法**

使用技术包括 CLIP 文图编码、Sentence‑BERT 语义嵌入、低秩投影与 MLP 计算概念概率、Soft‑PMI 与 NLI 评估概念间关系、乘法式层次化聚合、交替训练与概念正交正则化。

**📊 数据集**

实验数据集：MiRAGeNews、MMFakeBench、AMG，覆盖二分类与细粒度多类别谣言检测。

**📈 对比分析**

与 13 个基线（通用大语言模型与专业谣言检测器）对比，PCGR 在三组数据集上均取得最高准确率与 F1 分数，二分类上比最强基线提升 6–8%，细粒度检测上亦优于 MGCA，显示出更好的泛化与鲁棒性。

**⚠️ 局限性**

局限性：需要高算力与大规模 LLM；概念生成质量高度依赖 LLM 能力，易引入噪声；对极端新型谣言的即时响应仍有限；生成的解释链长度可变，部分案例对人类解读仍具挑战。

---

## 270. TacSIm: A Dataset and Benchmark for Football Tactical Style Imitation

**arXiv ID:** 2603.25199 | [PDF](https://arxiv.org/pdf/2603.25199v1)

**作者:** Peng Wen `[一作]`, Qiurui Wang `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `a4b10f5d-130b-4e77-9367-6469ec621899` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了TacSIm数据集与基准，用广播视频轨迹重建并在虚拟足球仿真中评估全队战术模仿；

**💡 创新点**

将真实比赛中的多球员轨迹精确投影到标准场地坐标、引入基于网格的空间-时间相似度评价、构建统一的战术阶段标签与可视化流程，形成首次可量化多代理战术模仿基准；

**🔧 技术方法**

采用YOLO+DeepSORT+TVCalib实现人物与球检测与坐标变换，利用VAE补全缺失轨迹，使用多代理强化/模仿学习模型（BC、CMIL、IRL、CoDAIL、DRAIL）在Google Research Football环境中训练与仿真，并用Jaccard与余弦相似度组合为最终分数；

**📊 数据集**

TacSIm数据集：140场英超2024‑25赛季官方广播视频，1080p 25FPS，包含194,565段标注的进攻/防守/过渡阶段及对应的球员与球轨迹；

**📈 对比分析**

对BC、CMIL、IRL、CoDAIL、DRAIL五种基线在不同网格分辨率与预测时长下进行统一评价，CoDAIL在中短期和中等网格表现最佳，DRAIL在长时预测更稳健；整体性能仍低于理想，说明长时段和细粒度预测仍具挑战；

**⚠️ 局限性**

局限包括：仅评估球员轨迹而非个体身份导致匹配噪声；对长时预测精度不足；数据仅来自英超，缺乏跨联赛、多语言、多视角验证；评价指标仅聚焦空间占用与方向相似度，未覆盖战术细节如传球网络或位置意图。

---

## 271. VolDiT: Controllable Volumetric Medical Image Synthesis with Diffusion Transformers

**arXiv ID:** 2603.25181 | [PDF](https://arxiv.org/pdf/2603.25181v1)

**作者:** Marvin Seyfarth `[一作]` (Heidelberg University), Sandy Engelhardt `[通讯]` (Heidelberg University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

研究并实现了首个纯Transformer的3D Diffusion Transformer（VolDiT），用于高质量体积医学图像的无条件和条件合成。

**💡 创新点**

创新点在于将全局自注意力迁移到3D体积空间，并提出时间门控控制适配器（TGCA），实现对分割掩模等空间条件的精细token级调控。

**🔧 技术方法**

采用VQ‑GAN压缩为低维latent，再用3D patch化生成token，利用全局自注意力的DiT块进行velocity预测，并通过时间门控适配器实现条件注入。

**📊 数据集**

使用公开肺CT数据集LUNA16和内部心脏CTA数据集TaviCT进行评估。

**📈 对比分析**

与U‑Net LDM和HA‑GAN基线对比，VolDiT在FID、precision、recall、density、coverage以及MS‑SSIM等指标上均表现更佳，尤其在FID下降和多样性提升方面显著。

**⚠️ 局限性**

受限于医学数据集规模有限、固定训练迭代次数导致大型模型未完全收敛，以及仅基于单一VQ‑GAN编码器，进一步研究需扩展数据、训练时长及多模态编码方式。

---

## 272. Prompt Attack Detection with LLM-as-a-Judge and Mixture-of-Models

**arXiv ID:** 2603.25176 | [PDF](https://arxiv.org/pdf/2603.25176v1)

**作者:** Hieu Xuan Le `[一作]` (GovTech), Quy Anh Tang `[通讯]` (GovTech)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出使用轻量级通用LLM作为“安全裁判”来实时检测提示攻击（如越狱、注入），通过结构化推理流程（意图拆解、安全信号核查、危害评估、自动自省）实现低延迟、可部署的安全防护；并将该裁判与多种现有方法进行对比实验，验证其优越性；进一步探究多模型集合（MoM）对检测性能的影响。

**💡 创新点**

创新点在于：①将LLM裁判任务转化为结构化链式思维并强制自省，显著提升对灰区攻击的判定精度；②提出多层次安全信号与意图分类的Taxonomy-guided reasoning框架；③构造融合多模型的加权组合策略，并系统评估其在不同规模下的提升与可能的退化；④利用真实生产流量与自动红队生成的混合数据集，逼真复现多样化攻击模式。

**🔧 技术方法**

使用技术包括：通用LLM（gemini‑2.0‑flash‑lite‑001、gemini‑3‑flash‑preview、gpt‑5、claude等）配合专门设计的Prompt与输出模板；链式思维（CoT）推理与自省步骤；置信度分级与数值映射；加权线性组合实现Mixture‑of‑Models；对比实验采用精确阈值、Precision/Recall/F1指标；评估延迟与性能。

**📊 数据集**

使用私有929条样本数据集：770条来自新加坡公共服务聊天机器人的真实生产请求（包括“误报陷阱”），159条通过自动红队（ART/PAIR）生成的攻击提示，涵盖社交工程、隐蔽框架、代码混合、角色扮演等多种灰区攻击。

**📈 对比分析**

比较方法：在阈值τ=0.5下，分别测算Precision、Recall、F1，并与Amazon Bedrock Guardrails、PromptGuard、ProtectAI、Qwen3Guard、gpt_oss_safeguard等传统或专用模型对比。结果显示：单一LLM Judge在F1上最高可达0.8711（gpt‑5.1），gemini‑2.0‑flash‑lite‑001达到0.844，延迟约1.5秒；在Mixture‑of‑Models实验中，只有少数配对（如gpt‑5.1+gpt‑5‑mini）能略有提升，整体未必优于单模型。

**⚠️ 局限性**

局限性包括：①数据集规模有限，可能未涵盖所有攻击变体；②LLM的随机性导致结果波动；③实验环境仍以零温度推理，实际部署中可能需要多轮推断以稳定判定；④缺乏对更大规模、多语言或极端延迟场景的验证。

---

## 273. To Write or to Automate Linguistic Prompts, That Is the Question

**arXiv ID:** 2603.25169 | [PDF](https://arxiv.org/pdf/2603.25169v1)

**作者:** Marina Sánchez-Torrón `[一作]` (Smartling), Jason Rauchwerk `[通讯]` (Smartling)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文系统比较了专家人工设计的零样本提示、未优化的DSPy签名以及使用GEPA自动优化的DSPy签名在术语插入、机器翻译和语言质量评估（LQA）三种语言学任务中的效果。

**💡 创新点**

创新点在于首次对比了专家级提示与自动优化提示在专门语言学工作流中的表现，展示了GEPA在不同模型和任务上的可比性与优势，并提供了统一单阶段与分阶段系统的性能对比。

**🔧 技术方法**

主要技术包括：DSPy框架、GEPA（梯度无关的离散优化器）、基于反射模型的评估、BLEU/HTER/chrF3、MQM兼容的错误分类、以及多模型（GPT‑4.1‑mini、GPT‑5.4‑mini、Gemini 3.1 Flash‑Lite、Claude Sonnet 4.6、Qwen3:8B‑q4_K_M）交互。

**📊 数据集**

使用的公开金标数据集：术语插入（1085例）、翻译（989例）和LQA（1250例），均包含源语、目标语、词典/样本、风格指南等信息，按20%/40%/40%划分为训练/验证/测试。

**📈 对比分析**

比较方法为对每个任务分别在统一（单阶段）和分阶段（多阶段）配置下，评估手工提示、基准DSPy和GEPA优化DSPy的BLEU、HTER、TMR、CHR3、检测/类别/严重性F1和MQM相关性。性能显示：术语插入中两种提示在BLEU/HTER上基本无显著差异，优化提示在术语匹配率上显著提升；翻译中大部分模型差异不显著，Gemini 3.1 Flash‑Lite在BLEU上手工提示明显优于优化提示；LQA中GEPA优化在错误描述（类别/严重性F1）上优于手工提示，手工提示在错误检测上有优势但仅在部分模型显著。

**⚠️ 局限性**

局限性包括：1）两种方法对标注数据的使用不对等，GEPA利用黄金分割进行自动反馈，专家提示依赖领域知识与人工迭代；2）仅评估单一优化器（GEPA）与“light”模式；3）未考虑少量样本或链式思维手工提示；4）所有评估均为自动化，缺乏人工主观评判；5）成本效益分析缺失。

---

## 274. Towards Foundation Models for 3D Scene Understanding: Instance-Aware Self-Supervised Learning for Point Clouds

**arXiv ID:** 2603.25165 | [PDF](https://arxiv.org/pdf/2603.25165v1)

**作者:** Bin Yang `[一作]` (Robert Bosch GmbH), Alexandru Paul Condurache `[通讯]` (Robert Bosch GmbH)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种自监督学习框架，利用语义一致性与几何推理双分支实现点云的实例感知表示；

**💡 创新点**

创新点在于引入Offset Distribution Regularization（ODR）和Spatial Clustering Regularization（SCR）两种无标签正则化，使模型能够稳定学习指向实例中心的偏移向量，从而显著提升实例与全景分割性能；

**🔧 技术方法**

采用teacher‑student自蒸馏与原型聚类作为语义分支，加入偏移分支并使用PIT归一化匹配真实几何先验；通过K‑means+BFS生成伪实例掩码实现局部一致性约束；

**📊 数据集**

在室内数据集ScanNet、ScanNet200、S3DIS以及室外数据集nuScenes、SemanticKITTI、Waymo上进行预训练与评估；

**📈 对比分析**

与DOS、Sonata、PSA等最新自监督方法在线性探针、解码探针与全微调三种协议下对比，室内实例分割平均AP提升约3–4%，室外全景分割PQ提升约4–5%，接近监督基准；

**⚠️ 局限性**

仍与完全监督存在一定差距，依赖大规模预训练数据，未涵盖时空4D几何推理，且偏移正则化对几何分布假设敏感。

---

## 275. PIDP-Attack: Combining Prompt Injection with Database Poisoning Attacks on Retrieval-Augmented Generation Systems

**arXiv ID:** 2603.25164 | [PDF](https://arxiv.org/pdf/2603.25164v1)

**作者:** Haozhen Wang `[一作]` (Chinese University of Hong Kong), Xiaoying Tang `[通讯]` (Chinese University of Hong Kong)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了PIDP-Attack，一种结合查询注入与数据库毒化的复合攻击，能在不知晓用户查询的前提下对RAG系统进行定向误导。

**💡 创新点**

创新点在于同时利用查询路径的注入与语料库路径的毒化，突破传统需要预知查询的毒化攻击限制，实现对任意查询的高成功率攻击。

**🔧 技术方法**

采用了提示注入模板（将目标问题嵌入查询后缀）、少量（≤k）毒化段落、基于Contriever的检索器和多种大型语言模型进行生成。

**📊 数据集**

在BEIR格式的三大问答数据集——Natural Questions、HotpotQA和MS-MARCO上进行实验。

**📈 对比分析**

与PoisonedRAG、Disinformation Attack、GGPP、GCG和Corpus等单向攻击进行对比，PIDP-Attack在多数模型与数据集上实现ASR超过90%，相较基线提升4%–16%，且检索F1保持高水平。

**⚠️ 局限性**

局限性包括对检索器召回的依赖（检索噪声导致失败）、对模型指令遵从性的敏感（某些拒绝型模型抵御力强）、以及需要在语料库更新渠道中有攻击权限，且在k增大或毒化预算受限时效果下降。

---

## 276. SportSkills: Physical Skill Learning from Sports Instructional Videos

**arXiv ID:** 2603.25163 | [PDF](https://arxiv.org/pdf/2603.25163v1)

**作者:** Kumar Ashutosh `[一作]` (University of Texas at Austin), Kristen Grauman `[通讯]` (University of Texas at Austin)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究构建了首个大规模体育教学视频数据集SportSkills，包含约638 k段视频和配音，并在此基础上训练了细粒度动作表征及错误条件下的教学视频检索模型，实现在学习者视频中自动给出个性化视觉反馈；

**💡 创新点**

创新点在于①提出互联网规模、精细标注正确/错误演示的体育技能数据集；②利用该数据集训练对比学习的动作-文本表征，使模型能够捕捉微妙的技能差异；③提出并实现错误条件的教学视频检索任务及其检索算法，显著提升个性化技能指导的效果；

**🔧 技术方法**

核心技术包括LLM+VLM自动筛选与标注、对比学习（video‑text contrastive）、CLIP、InternVideo2.5等编码器、Qwen‑2.5‑VL进行特征提取，以及LoRA微调的VLM进行检索评分；

**📊 数据集**

使用的数据集为新建的SportSkills（638k片段，55种运动，含正负标签），辅以Ego‑Exo4D学习者视频，并以专业教练标注的CoachGT测试集评估检索性能；

**📈 对比分析**

与零射击CLIP、InternVideo2.5等基线相比，SportSkills‑训练模型在20种运动上的平均recall@10提升至12.75/15.48（约4倍），在线性探针分类正确/错误演示上提升约5%，在检索任务中连贯率提高10%，Cohen’s d提升18%，显著优于所有基线；

**⚠️ 局限性**

局限性包括依赖LLM/VLM自动筛选导致可能出现标注错误、数据主要来自英语YouTube片段缺乏多语言覆盖、检索模型对细粒度错误识别仍受句子相似度限制、且未涵盖团队运动的协同动作等。

---

## 277. Photon: Speedup Volume Understanding with Efficient Multimodal Large Language Models

**arXiv ID:** 2603.25155 | [PDF](https://arxiv.org/pdf/2603.25155v1)

**作者:** Chengyu Fang `[一作]` (Tsinghua University), Minfeng Xu `[通讯]` (DAMO Academy, Alibaba Group)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本文提出 Photon，一种本地 3D 医疗多模态大语言模型框架，能够直接对完整 3D 扫描进行编码与问答，并通过自适应视觉令牌调度与代理梯度传播实现高效推理；

**💡 创新点**

创新点包括：①基于指令的令牌调度（ITS）能够在推理与训练中自适应地挑选重要视觉令牌并预测实例特定阈值；②代理梯度传播（SGP）实现离散令牌裁剪下的可微优化；③可变长度令牌序列保持体积细节而不依赖切片或固定压缩；④轻量级正则化（鲁棒性与翻转正则）抑制语言偏差；

**🔧 技术方法**

技术细节：3D ViT 视觉编码器、Qwen2.5‑VL 语言解码器、FlashAttention‑友好式裁剪、指令自相关注意力、实例阈值预测网络、straight‑through 代理掩码、梯度标准化与归一化、训练两阶段（对齐 + 微调）

**📊 数据集**

使用数据集：3D‑RAD（包含存在检测、静态/纵向诊断、医学测量、图像观察、异常检测等六类任务）以及 DeepTumorVQA（多选与自由文本测量、识别、视觉推理、医学推理子任务），并在训练阶段使用体积-标题配对数据进行 3D 视觉嵌入对齐

**📈 对比分析**

与 Qwen2.5‑VL、RadFM、M3D‑L2/P3、OmniV、Lingshu 等基线以及 VisionZip、HiPrune 等裁剪方法进行比较。Photon 在 3D‑RAD 上实现所有任务的最佳微调成绩，异常检测 +14%，测量 +7.3%，纵向诊断 +3%；在 DeepTumorVQA 上多选精度提升 3.6%，自由文本提升 11.5%，测量子任务超 35%；裁剪比较中 Photon 以约 4.12 tokens/s 的推理速度和更低的显存占用（≈2/3）保持甚至提高准确率

**⚠️ 局限性**

局限性：①对完整 3D 视觉编码器的全微调易导致模式崩溃，需要精细的两阶段训练；②目前仅验证于 CT/MRI VQA 与测量任务，泛化到其他医学影像或非医学体积尚未评估；③实现依赖 FlashAttention 与特定硬件，部署成本可能较高；④方法对指令格式敏感，若指令质量差或不完整可能影响阈值预测；

---

## 278. SAVe: Self-Supervised Audio-visual Deepfake Detection Exploiting Visual Artifacts and Audio-visual Misalignment

**arXiv ID:** 2603.25140 | [PDF](https://arxiv.org/pdf/2603.25140v1)

**作者:** Sahibzada Adil Shahzad `[一作]` (Academia Sinica), Hsin-Min Wang `[通讯]` (Academia Sinica)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种完全自监督的音频-视觉深伪造检测框架SAVe，只用真实视频训练，生成区域自混合伪造样本，并结合音频-视觉同步模块进行检测。

**💡 创新点**

创新点在于：① 只需真实视频，无需标注伪造数据；② 通过区域自混合（全脸、嘴唇、下脸）生成多尺度伪造，捕捉更丰富的视觉伪造痕迹；③ 将视觉伪造检测与音频-视觉同步一致性融合，实现跨模态一致性检测；④ 使用无监督的自监督方式显著提升对未知生成器和压缩噪声的鲁棒性。

**🔧 技术方法**

技术手段包括：自监督视觉伪造生成（SS‑VPFG）+多分支视觉判别器；音频-视觉同步模块AVSync（基于AV‑HuBERT + InfoNCE 对齐）；多分支融合（平均对数几率融合）；图像/音频预处理与区域掩码生成；数据增强与软混合。

**📊 数据集**

使用两个公开数据集进行评估：FakeAVCeleb（含多种生成方式和压缩版本）和AV‑LipSync‑TIMIT（LipSyncTIMIT）。模型仅在真实视频上训练，然后在两个数据集上测试。

**📈 对比分析**

与现有监督方法（Xception、LSDA、FTCN、LIPINC‑V2）及无监督视觉基线（FaceBlend、LowerFaceBlend、LipBlend）比较，SAVe 在 FakeAVCeleb‑LS 上达 0.9945 的 AUC，AVSync 分支单独在 LipSyncTIMIT 上已近 1.0，整体融合后在所有压缩水平下保持 0.96+ 的性能，显著优于无监督视觉基线且与监督方法相近。

**⚠️ 局限性**

局限性包括：单独视觉分支在音频-视觉不一致（如 RTVC）下表现差，音频-视觉同步模块对压缩敏感；自监督伪造生成可能无法覆盖所有真实攻击方式；在极端压缩或极低帧率视频中对齐特征衰减导致性能下降。

---

## 279. Robust Principal Component Completion

**arXiv ID:** 2603.25132 | [PDF](https://arxiv.org/pdf/2603.25132v1)

**作者:** Yinjian Wang `[一作]` (Beijing Institute of Technology), Gemine Vivone `[通讯]` (National Research Council)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种新的低秩与稀疏分解框架——鲁棒主成分补全（RPCC），通过间接识别稀疏分量的支持而非直接求解稀疏矩阵，实现了对稀疏前景替代或遮挡背景的更准确建模；

**💡 创新点**

创新点在于将RPCC建模为块级稀疏支持的贝叶斯稀疏张量分解问题，并通过变分贝叶斯推断得到一个硬分类器，在噪声方差趋于零时实现对稀疏块支持的确定性分离，消除了传统RPCA中后置阈值选择的困扰；

**🔧 技术方法**

采用了完全概率化的贝叶斯CP分解（BCP）与变分贝叶斯推断（VBI），引入块级高斯混合模型与稀疏性诱导的层级先验；

**📊 数据集**

实验数据包括合成张量数据、四个CDnet彩色视频序列（Highway、Turnpike、Crossroad、Busstation）以及四个高光谱数据集（Belcher、Urban、Beach、Salinas）；

**📈 对比分析**

与多种传统RPCA及基于张量的RPCA方法（PCP、KBR、TRPCA、ETRPCA、t-CTV、MTTD、ONTRPCA、LRTFR）以及专用异常检测算法（PTA、LSDM-MoG、TLRSR、BTD）进行对比；在合成数据上取得接近最优的低秩重建与支持识别；在视频前景提取与高光谱异常检测任务中，BCP-RPCC在F1、IoU等指标上均优于其他方法，尤其在阈值敏感度低、误报率低方面表现突出；

**⚠️ 局限性**

局限性包括：低秩分量恢复效果受限于CP分解的表示能力，需大秩参数导致计算成本高；模型对噪声方差σ的设置仍缺乏理论指导，需要进一步可辨识性与参数自适应研究；以及对更复杂数据类型的适应性与可扩展性待验证。

---

## 280. Variance Based Transmitter Localization in Vessel-Like Molecular Communication Channels

**arXiv ID:** 2603.25213 | [PDF](https://arxiv.org/pdf/2603.25213v1)

**作者:** Dağhan Erdönmez `[一作]` (Bogazici University), H. Birkan Yilmaz `[通讯]` (Bogazici University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `e15e3743-5ee0-4d5f-813d-d146868082fc` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种利用接收信号时间方差估算血管状分子通信中发射器与接收器距离的方法。

**💡 创新点**

创新点在于不需要发射时间或多台接收器，仅通过单一接收器测量信号方差即可估算距离，并给出闭式近似关系。

**🔧 技术方法**

采用Poiseuille流动模型的扩散-对流传输公式、Gaussian近似、泰勒展开、Monte Carlo粒子模拟。

**📊 数据集**

使用在血管尺度（r_v 5–10 μm）下的粒子模拟生成的接收信号数据。

**📈 对比分析**

与传统需要发射时间的峰值时间法相比，VALOR在相同条件下误差低于1%，R²>0.999，显示出同等甚至更优的精度。

**⚠️ 局限性**

局限在于对大Pe、小w、低r_v条件下近似失效，且需要已知平均流速且距离足够大才能满足 D_e/(l v_avg)≪1。

---

## 281. Free-Lunch Long Video Generation via Layer-Adaptive O.O.D Correction

**arXiv ID:** 2603.25209 | [PDF](https://arxiv.org/pdf/2603.25209v1)

**作者:** Jiahao Tian `[一作]` (Westlake University), Chi Zhang `[通讯]` (Westlake University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

在不进行任何模型再训练的前提下，提出了FreeLOC框架，能够把已训练好的视频Diffusion Transformer（DiT）扩展到更长的时序长度，生成高质量、时序一致的长视频。

**💡 创新点**

核心创新在于：① 用视频相对位置重编码（VRPR）对超出训练范围的帧间相对位置进行多粒度映射；② 用分层稀疏注意（TSA）对长上下文导致的注意力分散进行结构化抑制；③ 通过自动层级敏感性探测实现层适应性应用，使每一层仅在其最需要的时刻使用VRPR或TSA。

**🔧 技术方法**

主要技术包括：3D RoPE位置编码、多粒度重新编码策略（Fine/Medium/Coarse），分层稀疏注意机制（Local Window、Striped Attention、Attention Sink），以及自适应层级敏感性探测和选择性插拔。

**📊 数据集**

利用公开预训练的DiT模型（如Wan2.1‑T2V‑1.3B和HunyuanVideo）及其原始训练集，随后在公开的Text‑to‑Video评测集（VBench）上进行评估。

**📈 对比分析**

与直接采样、滑窗、FreeNoise、FreeLong、RIFLEx等训练无关基线在VBench的SC、BC、MS、IQ、AQ、DD六项指标上对比，FreeLOC 在 2× 与 4× 长度扩展时均实现了最高的综合得分，显著优于所有对照方法。

**⚠️ 局限性**

局限性包括：仍受长上下文带来的注意力稀疏和计算成本限制；对极长序列（数百帧以上）或复杂动态场景的全局一致性提升空间；以及在不同类型的视频内容（如高动态场景）中效果尚未完全验证。

---

## 282. CardioDiT: Latent Diffusion Transformers for 4D Cardiac MRI Synthesis

**arXiv ID:** 2603.25194 | [PDF](https://arxiv.org/pdf/2603.25194v1)

**作者:** Marvin Seyfarth `[一作]` (Heidelberg University), Sandy Engelhardt `[通讯]` (Heidelberg University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出并实现了全4D的CardioDiT模型，用于短轴心脏MRI的无监督合成；

**💡 创新点**

直接对完整的3D+t分布进行建模，去除空间与时间分离的结构偏差，并将Transformer应用于4D潜在空间；

**🔧 技术方法**

使用VQ‑GAN编码器产生4D潜在表示，4D扩散Transformer（含4D位置编码与patch化），FlashAttention等技术；

**📊 数据集**

公开acdc和mm2数据集以及一套含813例的私有临床数据集；

**📈 对比分析**

与2D+t LDM、3D+t U-Net等基线比较，CardioDiT在FID、d‑SSIM、AR_ED、EF分布等指标上表现更佳，显示出更好的解剖一致性与运动真实性；

**⚠️ 局限性**

局限性在于对时间轴采用循环重复标准化，可能引入人工周期性，对可变长度序列的处理不足。

---

## 283. Train at Moving Edge: Online-Verified Prompt Selection for Efficient RL Training of Large Reasoning Model

**arXiv ID:** 2603.25184 | [PDF](https://arxiv.org/pdf/2603.25184v1)

**作者:** Jiahao Wu `[一作]` (Southern University of Science and Technology), Ke Tang `[通讯]` (Southern University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 HIVE 两阶段在线验证的 prompt 选择框架，以显著降低大模型推理训练中的 rollout 计算开销并保持性能

**💡 创新点**

创新点在于将历史奖励轨迹与实时 prompt 熵相结合，形成“学习边缘”精准定位，解决 metadata staleness 的同时实现低成本高效的 prompt 过滤

**🔧 技术方法**

技术包括 GRPO 基于可验证奖励的强化学习、基于历史 reward 轨迹的概率粗筛、在线 prompt 熵验证、动态阈值调节及 λ 平衡系数

**📊 数据集**

使用 DAPO+MATH、OPEN‑R1 30k 训练集，评估六个数学推理基准（Math500、AIME24、AMC、Minerva Math、Gaokao、Olympiad Bench）

**📈 对比分析**

与 Dynamic Sampling 与 GRESO 对比，HIVE 在 3.4× 降低 rollouts、2.3× 提升总训练时间，同时在六大基准上保持或超过对照组的平均准确率

**⚠️ 局限性**

局限：仅在文本推理模型上验证，尚未推广至多模态场景；对 λ、Δp 等超参数仍需手动或自适应调节，未进行全局最优搜索

---

## 284. Bilingual Text-to-Motion Generation: A New Benchmark and Baselines

**arXiv ID:** 2603.25178 | [PDF](https://arxiv.org/pdf/2603.25178v1)

**作者:** Wanjiang Weng `[一作]` (Southeast University), Hongsong Wang `[通讯]` (Southeast University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了第一份双语文本到运动生成基准BiHumanML3D，并基于此训练了Bilingual Motion Diffusion (BiMD) 模型；

**💡 创新点**

创新点在于构建多阶段LLM辅助双语注释流程以及跨语言对齐 (CLA) 模块，显著提升双语和代码混合输入下的运动生成质量；

**🔧 技术方法**

使用了大型语言模型（DeepSeek、Qwen）、OpenCLIP教师+XLM学生的知识蒸馏、跨语言对齐损失、Latent Diffusion、CFG 指导等技术；

**📊 数据集**

采用了扩展自 HumanML3D 的双语数据集 BiHumanML3D（13,312 条目），并与原始 HumanML3D 及 KIT-ML 进行对照；

**📈 对比分析**

通过 FID、R‑Precision、MM Distance 等指标与现有单语扩散模型及翻译+单语模型对比，BiMD+CLA 在 FID、R‑Precision 上分别比无对齐模型提升约70% 与 5% 以上，且在零射与代码混合场景下表现优异；

**⚠️ 局限性**

局限性包括仅覆盖英中两语种，跨语言对齐依赖单语教师模型，代码混合的极端复杂句式仍可能出现语义失真，且对其他语言扩展尚未验证。

---

## 285. AG-EgoPose: Leveraging Action-Guided Motion and Kinematic Joint Encoding for Egocentric 3D Pose Estimation

**arXiv ID:** 2603.25175 | [PDF](https://arxiv.org/pdf/2603.25175v1)

**作者:** Md Mushfiqur Azam `[一作]` (University of Texas at San Antonio), Kevin Desai `[通讯]` (University of Texas at San Antonio)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本论文提出了一种双流框架AG-EgoPose，用于从头戴鱼眼摄像机的第一人称视频中估计3D人体姿势。

**💡 创新点**

创新点包括：①融合短期与长期运动上下文的动作引导编码器；②将2D热图嵌入到可学习的 joint token 中，通过 Transformer 解码器实现空间-时间信息的细粒度融合；③在不依赖额外深度或场景重建的情况下，仅用单目图像实现高精度姿态估计。

**🔧 技术方法**

核心技术包括：Weight‑sharing ResNet‑18 encoder‑decoder + FPN 生成 2D 热图；ResNet‑50 + ActionFormer 进行多尺度时序注意力编码；Transformer 解码器与 learnable joint token 进行交叉注意力融合；骨骼一致性损失正则化。

**📊 数据集**

使用 EgoPW（约 318K 帧、10 位演员）与 SceneEgo（28K 图像、2 位演员）两个真实 egocentric 数据集进行训练与评估。

**📈 对比分析**

与现有方法（SelfPose、Mo^2Cap^2、EgoHMR 等）相比，在 EgoPW 上 PA‑MPJPE 下降至 76.7 mm（相较于最佳 84.2 mm 下降 9%），在 SceneEgo 上 MPJPE/PA‑MPJPE 分别为 104.0 mm / 76.2 mm，提升 14.5 mm / 16.5 mm（约 18%），且模型参数 11.4 M、FLOPs 8.0 G，兼顾精度与效率。

**⚠️ 局限性**

局限性包括：①对极端视角或长时间摄像机抖动仍易出现误差；②依赖预训练的 ActionFormer，若无足够 egocentric 动作数据可能性能下降；③仅使用单目信息，无法充分利用深度/多摄像机优势，导致某些场景下姿态不完整。

---

## 286. ET-SAM: Efficient Point Prompt Prediction in SAM for Unified Scene Text Detection and Layout Analysis

**arXiv ID:** 2603.25168 | [PDF](https://arxiv.org/pdf/2603.25168v1)

**作者:** Xike Zhang `[一作]` (Wuhan University), Bo Du `[通讯]` (Wuhan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 ET‑SAM，利用轻量化点解码器和联合训练实现统一场景文本检测与布局分析，显著加速推理并提升多级文本识别。

**💡 创新点**

创新点包括：① 通过轻量化点解码器从词级热图生成稀疏视觉提示，避免千点采样；② 设计可学习任务提示并联合训练不同粒度注释的数据集；③ 将多级文本分割集成到 SAM 架构，实现统一检测与布局。

**🔧 技术方法**

采用 SAM 预训练的图像编码器、轻量化点解码器、层次化掩码解码器（HM‑Decoder）、任务提示、联合训练策略、矩阵 NMS、Union‑Find 等技术。

**📊 数据集**

使用 HierText、Total‑Text、CTW1500、ICDAR15、ICDAR13、TextSeg 等多级文本数据集进行联合训练与评估。

**📈 对比分析**

与 Hi‑SAM 在 HierText 上仅略逊，微调后在所有层级均优于 Hi‑SAM；在 Total‑Text、CTW1500、ICDAR15 上平均 F‑score 提升 11.0%，相对竞争方法提升 10–15%。

**⚠️ 局限性**

局限性在于仍受限于训练数据量和多级标注稀缺，且在段落级别分割仍存在轻微失误；未使用大规模合成数据，导致与某些专用检测器的性能差距。

---

## 287. A Semantically Disentangled Unified Model for Multi-category 3D Anomaly Detection

**arXiv ID:** 2603.25159 | [PDF](https://arxiv.org/pdf/2603.25159v1)

**作者:** SuYeon Kim `[一作]` (Kyung Hee University), MyeongAh Cho `[通讯]` (Kyung Hee University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

设计了一种统一的3D异常检测框架SeDiR，先通过粗细尺度全局标记构建语义身份，再用类别对比学习消除类别交叉，最终通过几何引导解码器实现语义一致的重建，提升异常检测和定位性能。

**💡 创新点**

创新点在于：①首次提出“语义解耦重建”思路，即先识别对象类别再进行重建；②引入粗细尺度全局标记(Coarse‑to‑Fine Global Tokenization)形成全局语义上下文；③采用类别条件对比学习(Category‑Conditioned Contrastive Learning)有效消除Inter‑Category Entanglement；④在解码阶段使用几何引导注意力(Geometry‑Guided Decoder)保证重建几何与语义一致。

**🔧 技术方法**

使用的核心技术包括：粗细尺度全局标记(CFGT)、类别条件对比学习(C3L)（包含监督对比损失、类别分类损失和多分辨率余弦对齐损失）、几何引导解码器(GGD)（注意力偏置、几何先验）、自适应上下文标记(Adaptive Context Token)、对比学习缓冲区、Gaussian池化等。

**📊 数据集**

实验数据集：Real3D‑AD（12类真实点云，单视角测试）和Anomaly‑ShapeNet（40类合成点云）。

**📈 对比分析**

与多种单类别和统一基线方法（BTF、M3DM、PatchCore、IMRNet、MC3D‑AD等）进行对比，在Real3D‑AD上对象级AUROC提升2.8%（从85.0%提升到86.0%），在Anomaly‑ShapeNet上提升9.1%（从96.2%提升到97.6%），在点级定位上也保持领先或相近。

**⚠️ 局限性**

局限性：①对训练数据多样性依赖较大，未验证跨域或开放集下的泛化；②对稀疏或噪声较多的点云鲁棒性尚需进一步提升；③模型训练和推理复杂度相对较高，部署成本需要考虑。

---

## 288. Learning to Rank Caption Chains for Video-Text Alignment

**arXiv ID:** 2603.25145 | [PDF](https://arxiv.org/pdf/2603.25145v1)

**作者:** Ansel Blume `[一作]` (University of Illinois Urbana Champaign), Garin Kessler `[通讯]` (Amazon Prime Video)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了使用排名优化而非传统二分类 DPO 方法来提升视频-文本对齐与长文本生成的质量。

**💡 创新点**

提出通过 LLM 递归降级生成全序 caption 链，并利用 Plackett-Luce DPO 训练模型，使其能细粒度关注视觉细节。

**🔧 技术方法**

采用 LLM 生成降级 caption 链、Plackett-Luce 排名优化、LoRA 微调、vision encoder 联合微调等技术。

**📊 数据集**

在 PE-Video、MSR-VTT 等视频 caption 数据集上生成 RCC，并在 VDC、ARGUS、TempCompass、PE-Video 等评测集上验证。

**📈 对比分析**

与 SFT、二元 DPO、MPO 等基线对比，排名优化在相关性、描述性、时序一致性、流畅度等指标上均优于基线，并在 VDC/ARGUS、长文本问答等任务上提升显著。

**⚠️ 局限性**

对 vision encoder 的 finetune 敏感，需较大算力；链长越长效果提升但资源消耗显著；缺乏对更广泛多模态任务的验证。

---

## 289. Physical Backdoor Attack Against Deep Learning-Based Modulation Classification

**arXiv ID:** 2603.25304 | [PDF](https://arxiv.org/pdf/2603.25304v1)

**作者:** Younes Salmi `[一作]` (Poznan University of Technology), Hanna Bogucka `[通讯]` (Poznan University of Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文研究了一种基于功率放大器非线性失真实现的物理后门攻击，用以欺骗基于深度学习的调制识别模型。

**💡 创新点**

创新点在于首次将物理触发器（PA失真）嵌入训练数据并在推理时通过估计触发器实现高成功率的后门攻击。

**🔧 技术方法**

使用卷积神经网络VT-CNN2进行调制分类，并用CNN模拟PA的非线性映射来估计触发器。

**📊 数据集**

数据集为基于OFDM的仿真信号，包含11种调制方式，SNR范围-8到18 dB。

**📈 对比分析**

与数字后门攻击相比，本攻击在5%污染率下即可达到95%成功率，且在-8到10 dB SNR范围内保持>92%，且对Neural Cleanse、STRIP和Activation Clustering无效。

**⚠️ 局限性**

局限在于仅在仿真环境下验证，缺乏对真实硬件和多样化PA特性的评估，且攻击依赖对PA非线性的精确估计。

---

## 290. V2U4Real: A Real-world Large-scale Dataset for Vehicle-to-UAV Cooperative Perception

**arXiv ID:** 2603.25275 | [PDF](https://arxiv.org/pdf/2603.25275v1)

**作者:** Weijia Li `[一作]` (Xiamen University), Chenglu Wen `[通讯]` (Xiamen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `edb9d762-f411-4838-a852-f2d638b018db` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并构建了V2U4Real数据集，针对车-无人机协同感知任务开展单体与协同3D目标检测、跟踪基准实验。

**💡 创新点**

首次提供大规模真实世界多模态车-无人机协同感知数据集，解决视角差异与空间时序异步挑战，并给出统一评估基准。

**🔧 技术方法**

采用多传感器（旋转LiDAR、固态LiDAR、RGB相机）同步采集，利用几何校准、点云配准、Transformer/注意力等特征融合方法（如CoAlign、DSRC、V2X‑ViT）。

**📊 数据集**

使用V2U4Real自身的56K LiDAR帧、56K多视角相机图像以及700K 3D边框数据；对比现有V2V、V2I数据集进行验证。

**📈 对比分析**

通过对比单体、无融合、早期/后期/中间融合四种策略，CoAlign在同步/异步设置下实现最高AP（同步56.67%/36.61%，异步50.81%/33.33%），长距离检测提升约10%，跟踪任务同样显著优于单体基线。

**⚠️ 局限性**

受限于传输延迟、空间时序异步、光照/天气变化以及模型对高速动态场景的鲁棒性不足，仍需进一步提升实时性与跨域适应性。

---

## 291. Distribution and Clusters Approximations as Abstract Domains in Probabilistic Abstract Interpretation to Neural Network Analysis

**arXiv ID:** 2603.25273 | [PDF](https://arxiv.org/pdf/2603.25273v1)

**作者:** Zhuofan Zhang `[一作]` (Imperial College London), Herbert Wiklicky `[通讯]` (Imperial College London)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

本文提出了分布近似和聚类近似两种新的抽象域，用于概率抽象解释框架下的神经网络分析，并给出了相应的抽象变换器；

**💡 创新点**

创新点在于引入了可根据具体问题灵活选择的分布近似（多项式回归、径向基函数、傅里叶变换）和聚类近似（K‑means、GMM），从而扩展了传统网格近似的适用范围；

**🔧 技术方法**

采用了概率抽象解释理论、分布近似技术（多项式、RBF、傅里叶）和聚类技术（K‑means、GMM）以及示例性推导；

**📊 数据集**

使用了人工合成的示例数据集（如13个一维点、连续的Zonotope），未报告对真实数据集的实验；

**📈 对比分析**

论文通过理论推导与示例比较与传统网格近似，声称在不同场景下更具灵活性和表达力，但未给出定量性能指标；

**⚠️ 局限性**

局限性包括缺乏大规模实证评估、计算复杂度高（如中心选取的穷举搜索）、仅在仿射层实验，未处理非线性激活及更复杂网络结构。

---

## 292. When Hate Meets Facts: LLMs-in-the-Loop for Check-worthiness Detection in Hate Speech

**arXiv ID:** 2603.25269 | [PDF](https://arxiv.org/pdf/2603.25269v1)

**作者:** Nicolás Benjamín Ocampo `[一作]` (Centrum Wiskunde & Informatica), Davide Ceolin `[通讯]` (Centrum Wiskunde & Informatica)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究提出了WSF-ARG+数据集，并开发了一种LLM‑in‑the‑loop注释框架，用于高效标注嫌恶言论（HS）与可检查性（check‑worthiness）标签，随后探讨两者的关联及其对HS检测性能的影响。

**💡 创新点**

创新点在于：①首次将HS与可检查性标签结合形成完整数据集；②提出LLM‑in‑the‑loop框架，利用多模型投票与人工判定降低人工成本；③证明包含可检查性主张的HS更具攻击性，并且引入检查性标签可提升大型LLM在HS检测上的宏F1至0.213/0.154。

**🔧 技术方法**

技术手段包括：多模型（Mistral、Llama、Qwen、Olmo、Command）零/一轮提示、三次投票多数决、人工审查、温度1控制；使用OpenAI Moderation评估攻击与仇恨得分；对比评估采用宏F1、Cohen κ、Krippendorff α等指标。

**📊 数据集**

使用数据集：WSF‑ARG+（来自Stormfront的227条HS和294条非HS消息，提取的多条主张及其检查性标签），并与原WSF‑ARG及公开的LIAR、FullFact等数据进行对照。

**📈 对比分析**

与全人工（platinum）标注对比，Cohen κ达0.800；与无检查性标签基线相比，所有大型模型（≥70B）在HS检测宏F1提升0.213（最大）到0.154（平均）。实验中使用多模型平均、三次重复以评估稳定性。

**⚠️ 局限性**

局限性包括：框架仅在白人至上论坛数据上验证，泛化性未知；依赖专家人工参与；LLM作为注释者可能存在偏差与滥用风险；未验证对其他任务或更大规模数据集的适用性。

---

## 293. ViewSplat: View-Adaptive Dynamic Gaussian Splatting for Feed-Forward Synthesis

**arXiv ID:** 2603.25265 | [PDF](https://arxiv.org/pdf/2603.25265v1)

**作者:** Moonyeon Jeong `[一作]` (University of Seoul), Hongje Seong `[通讯]` (University of Seoul)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种从未标定图像进行新视角合成的视角自适应动态高斯 splatting 框架 ViewSplat。

**💡 创新点**

创新点在于将传统静态高斯回归转为视角自适应动态更新：网络先预测基准高斯和动态 MLP 权重，渲染时 MLP 根据目标视角产生残差更新各高斯属性，从而显著提升视角相关细节的重建。

**🔧 技术方法**

采用 3D Gaussian splatting、基于 ViT 的几何变换器、DPT 头进行基准高斯与姿态预测、视角依赖的 MLP 生成残差、无姿势监督训练以及 SPFSplat 作为骨干模型。

**📊 数据集**

在 RealEstate10K、ACID 数据集上进行训练和评估，并在 DTU 数据集上做零样本跨域测试。

**📈 对比分析**

与 SPFSplat、SPFSplatV2、SPFSplatV2‑L 等前沿方法比较，ViewSplat 在 PSNR、SSIM、LPIPS 上持续提升（例如 RE10K 上 PSNR 从 27.95 提升到 28.98），同时保持实时渲染（约 154 FPS）和较低的推理时间。

**⚠️ 局限性**

局限性：仍属于重建式方法，缺乏生成先验，无法合成未观测区域的内容，导致稀视角下出现模糊或空白区域。

---

## 294. Semantic-Aware Prefix Learning for Token-Efficient Image Generation

**arXiv ID:** 2603.25249 | [PDF](https://arxiv.org/pdf/2603.25249v1)

**作者:** Qingfeng Li `[一作]` (Institute of Automation, Chinese Academy of Sciences), Pengfei Wan Guoqi Li `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种语义感知前缀学习的视觉 tokenizer（SMAP）以及基于该 tokenizer 的混合因果自回归–扩散生成器（CARD），实现高效、可控的图像生成。

**💡 创新点**

创新点在于：①将类别条件直接注入 1D tokenization 前缀，使语义信息成为必需的训练信号；②通过尾部 token 丢弃策略强迫模型逐步将结构信息压缩到前缀，形成信息排序、语义扎根的 token 序列；③共享 tokenizer 学习到的语义 embedding 供生成器使用，提升生成质量。

**🔧 技术方法**

使用了 query‑based 1D tokenization、视觉 Transformer 编码/解码、语义条件注入、尾部 token 丢弃、VQ/KL/SoftVQ 正则化，以及混合因果自回归 + 低秩扩散（CARD）生成框架。

**📊 数据集**

在 ImageNet‑1K（256×256）上进行训练与评估，使用其训练集进行 tokenizer/生成器预训练，验证集用于 reconstruction 与 generation 的 FID、IS 等指标。

**📈 对比分析**

与多种基准 tokenizer（VQGAN、VQ、KL、SoftVQ、TiTok 等）以及生成器（LlamaGen、LDM、DiT、SiT 等）在相同 token 数量（如 128 token）下对比。SMAP 在 128 token 预算下 reconstruction FID 可从 1.47 降至 0.55，生成 FID 下降到 0.75‑0.90，IS 显著提升，表明语义前缀学习显著提高了 token 质量和后续生成性能。

**⚠️ 局限性**

局限性包括：①需要大量 GPU 资源进行训练；②tail token dropping 需要手动调参，可能对不同数据集不易迁移；③目前仅支持类别级别的语义条件，难以处理更细粒度或多模态条件；④实验仅在 ImageNet‑1K 上验证，缺乏跨域或更大规模数据集的评估。

---

## 295. FEAST: Fully Connected Expressive Attention for Spatial Transcriptomics

**arXiv ID:** 2603.25247 | [PDF](https://arxiv.org/pdf/2603.25247v1)

**作者:** Taejin Jeong `[一作]` (Yonsei University), Seong Jae Hwang `[通讯]` (Yonsei University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了 FEAST 框架，利用全连接注意力模型从 H&E 影像中预测空间转录组表达并保留细胞结构信息。

**💡 创新点**

创新点在于：1）全连接注意力替代稀疏图，能够建模所有点对交互；2）负向注意力同时建模激活与抑制关系；3）离网采样伪点恢复被截断或遗漏的形态学上下文。

**🔧 技术方法**

使用了 Transformer 自注意力、负向注意力机制、层级注意力（局部 k‑NN 与全局自注意力）、位置偏置、以及预训练的 UNI2‑h 影像特征提取器。

**📊 数据集**

在三个公开 ST 数据集上实验：ST-Net（乳腺癌）、Her2ST（乳腺癌）和 SCC（皮肤癌）。

**📈 对比分析**

通过 8 折交叉验证与七个基线模型（ResNet18+FC、BLEEP、HisToGene、Hist2ST、THItoGene、TRIPLEX、MERGE）对比，FEAST 在 7/9 指标上实现最高表现，尤其在 Her2ST 数据集上 MSE 降至 0.5761、PCC 提升至 0.5524。

**⚠️ 局限性**

局限性包括：离网采样导致额外伪点增加计算量；对位置偏置的固定系数可能限制不同尺度交互的建模；以及在更大规模、更多组织类型的数据上仍需进一步验证。

---

## 296. FluxEDA: A Unified Execution Infrastructure for Stateful Agentic EDA

**arXiv ID:** 2603.25243 | [PDF](https://arxiv.org/pdf/2603.25243v1)

**作者:** Zhengrui Chen `[一作]` (Zhejiang University), Cheng Zhuo `[通讯]` (Zhejiang University)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 FluxEDA，一个统一且有状态的基础设施，使得代理式 EDA 能够在持久化的工具实例上执行多步骤、跨工具的自动化任务。

**💡 创新点**

核心创新包括基于网关的统一执行接口、持久化后端实例与状态管理、结构化请求/响应处理，以及通过“技能”层面实现高层流程指引，从而突破传统一次性脚本调用的局限。

**🔧 技术方法**

实现技术涵盖 Socket‑RPC 通信、MCP 服务器、Python/C++ SDK、工具适配器、实例生命周期管理、心跳监控、超时与异步长任务支持，以及基于 LLM 的代理框架。

**📊 数据集**

实验数据来自两大商业后端案例：后路由时序 ECO 关闭与标准单元子库优化，使用实际工业 EDA 流程（Tcl、Python 等）而非公开数据集。

**📈 对比分析**

与传统一次性脚本/请求方式相比，FluxEDA 能够在同一工具实例上完成多轮分析‑执行‑细化，支持状态回滚与分支探索；实验显示能实现更快的收敛、更高的时序/面积恢复率，体现了显著的效率与质量提升。

**⚠️ 局限性**

局限性包括：对商业工具的依赖需要手工编写工具适配器；技能层的设计仍需人工维护；在极端大规模或非标准工具环境下的可扩展性待验证；以及 LLM 生成指令的可靠性仍需进一步控制。

---

## 297. Gap Safe Screening Rules for Fast Training of Robust Support Vector Machines under Feature Noise

**arXiv ID:** 2603.25221 | [PDF](https://arxiv.org/pdf/2603.25221v1)

**作者:** Tan-Hau Nguyen `[一作]` (Can Tho University), Kien Trung Nguyen `[通讯]` (Can Tho University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在鲁棒支持向量机（R‑SVM）中设计并实现了一套动态安全样本筛选框架，能够在训练过程中提前剔除对最优分类器无关的训练样本，从而显著降低求解规模和时间成本。

**💡 创新点**

首次将Lagrangian对偶与强凸性结合，构造GAP球安全区域，推导出针对含不确定集的R‑SVM的安全筛选规则；并提出从静态到动态的完整筛选流程。

**🔧 技术方法**

Lagrangian对偶分析、KKT条件、强凸性、GAP安全球构造、动态安全筛选算法、基于SCS求解的SOCP实现。

**📊 数据集**

UCI公开数据集：Breast Cancer Wisconsin（569样本，30维）和Spambase（4601样本，57维）。

**📈 对比分析**

与不做筛选的R‑SVM基线在相同参数设置下进行比较。实验表明，安全筛选可在Breast Cancer上加速1.07–18.96倍、在Spambase上加速1.54–9.85倍；筛选率在96–99%之间，保持分类性能不变。

**⚠️ 局限性**

仅针对ℓ₂不确定集，未探索更复杂的不确定集和更大规模数据集；对比仅有一套基线，缺乏与其他安全筛选方法的交叉验证；动态筛选在早期迭代中的计算开销尚未进一步优化。

---

## 298. Beyond Benchmarks: How Users Evaluate AI Chat Assistants

**arXiv ID:** 2603.25220 | [PDF](https://arxiv.org/pdf/2603.25220v1)

**作者:** Moiz Sadiq Awan `[一作]` (Independent Researcher), Muhammad Salman Munaf `[通讯]` (Independent Researcher)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对七大 AI 聊天平台（ChatGPT、Claude、Gemini、DeepSeek、Grok、Mistral、Llama）进行跨平台用户满意度、采纳动机、多平台使用和痛点的问卷调查。

**💡 创新点**

首次采用一致的问卷对活跃用户进行跨平台比较，揭示满意度趋同、专化竞争、首发优势与多重使用等真实市场格局。

**🔧 技术方法**

使用 Qualtrics 设计问卷，运用非参数统计（Kruskal‑Wallis、Mann‑Whitney）、效应量计算、Cronbach α、主题编码等技术进行数据分析。

**📊 数据集**

以 388 名活跃 AI 聊天用户自报的数据为样本，涵盖平台使用情况、满意度、驱动因素、使用案例、订阅状态、价格敏感度以及开放式回答。

**📈 对比分析**

采用 within‑subjects 评价块对同一受访者在不同平台上的评分进行比较；Kruskal‑Wallis 检验显示总体满意度差异显著，但主要受低评分平台影响；顶尖三平台满意度无显著差异，使用案例评分表明各平台在不同任务领域存在显著差异。

**⚠️ 局限性**

样本偏向技术专业、日常高频使用、地理集中，受访者自选导致潜在偏差；平台列表的中途变更可能影响部分结果；5 分量表可能压缩差异；跨平台主观评分可能受锚定效应影响；Mistral 与 Llama 样本量较小；横断面设计无法揭示因果关系。

---

## 299. SDD-YOLO: A Small-Target Detection Framework for Ground-to-Air Anti-UAV Surveillance with Edge-Efficient Deployment

**arXiv ID:** 2603.25218 | [PDF](https://arxiv.org/pdf/2603.25218v1)

**作者:** Pengyu Chen `[一作]` (Southeast University), Junbo Wang `[通讯]` (Southeast University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种面向地面到空中（G2A）无人机侦测的轻量级小目标检测框架，解决了像素极低、背景杂乱和实时推理的挑战。

**💡 创新点**

创新点包括：在4×下采样下引入P2高分辨率检测头；去除DFL模块、采用NMS-free双标签分配；使用MuSGD+STAL优化稀疏小目标梯度；以及通过多尺度特征对齐的知识蒸馏提升学生模型精度。

**🔧 技术方法**

主要技术方法包括YOLO26改进架构、双注意力机制、DPF-free回归、NMS-free推理、MuSGD梯度正交化、STAL自适应标签分配、Progressive Loss、温度蒸馏等。

**📊 数据集**

使用自研的30k张高分辨率G2A无人机数据集（涵盖多天气、多背景、微小目标），并在此数据集上从零开始训练和评估。

**📈 对比分析**

与YOLOv5n基线相比，该框架在同等参数量下实现了86.0% mAP@0.5（+7.8pp），GPU上可达226 FPS，CPU上可达35 FPS，表明既提升了精度又保持了实时性能。

**⚠️ 局限性**

局限性包括：尚未完成INT8量化及在国产NPU上的部署；对极端天气（如大雨、强背光）数据仍需进一步增强；以及模型对多光谱（红外）数据的适配仍待扩展。

---

## 300. A Linear-Size Block-Partition Fibonacci Encoding for Gödel Numbering

**arXiv ID:** 2603.25307 | [PDF](https://arxiv.org/pdf/2603.25307v1)

**作者:** Zoltán Sóstai `[一作]` `[通讯]`, Zoltán Sóstai

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

提出一种基于斐波那契数列块划分的编码方法，将有限字母表的字符串映射为自然数。

**💡 创新点**

创新点在于通过在斐波那契序列中插入单索引空隙，将其划分为若干块，使得每个字符串位置仅选取对应块内的一个斐波那契数，从而保证所得到的和满足 Zeckendorf 表示，保证单射且码长线性增长。

**🔧 技术方法**

采用 Zeckendorf 定理、斐波那契数列性质以及贪婪算法进行编码与解码，利用非相邻索引保证唯一性。

**📊 数据集**

无外部数据集；该方法适用于任意固定有限字母表。

**📈 对比分析**

与传统素数幂 Gödel 编码、Rosko 的右嵌套携带无进位配对函数等做比较。该编码在最坏情况下码长为 Θ(m)，比 Rosko 的方法的指数级 Θ(2^m) 改进显著；相较于信息理论下的最优常数因子，仍保持线性但有约 2.3 倍的常数开销。

**⚠️ 局限性**

局限性包括：仅解决数值编码层，未涉及完整的 Gödel 编码所需的代换、连接、对角化等操作；未验证在受限算术系统中的可定义性与应用；对极大字符串长度的实际数值大小仍可能过大。

---

## 301. Auditing the Impact of Cross-Site Web Tracking on YouTube Political and Misinformation Recommendations

**arXiv ID:** 2603.25302 | [PDF](https://arxiv.org/pdf/2603.25302v1)

**作者:** Salim Chouaki `[一作]` (New York University Abu Dhabi), Sandra Siby `[通讯]` (New York University Abu Dhabi)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `a2602d71-93ab-4bad-974b-672788df8193` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

审计跨站跟踪对YouTube政治与误导性推荐的影响

**💡 创新点**

首次从跨站跟踪角度出发，使用sock puppet自动化实验并比较隐私增强浏览器对推荐结果的影响

**🔧 技术方法**

采用sock puppet浏览器实例、MPNet文本嵌入、余弦相似度、Google Chrome（默认）与Brave（隐私模式）等技术

**📊 数据集**

利用Media Bias/Fact Check新闻偏见标签、PolitiFact误导性新闻标签以及YouTube视频标题和转录文本做为数据集

**📈 对比分析**

通过对比跟踪允许与跟踪阻止环境下的推荐余弦相似度，发现跟踪允许时误导性内容相似度提升约0.0134，而隐私模式下无此变化

**⚠️ 局限性**

限制包括仅评估误导性曝光、未分析政治偏见变化、实验时间有限以及未考虑登录与匿名两种状态

---

## 302. Towards Controllable Low-Light Image Enhancement: A Continuous Multi-illumination Dataset and Efficient State Space Framework

**arXiv ID:** 2603.25296 | [PDF](https://arxiv.org/pdf/2603.25296v1)

**作者:** Hongru Han `[一作]` (University of Macau), Zhuohua Ye `[通讯]` (University of Macau)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种可控低照明图像增强框架CLE-RWKV，并引入连续光照数据集Light100。

**💡 创新点**

创新点在于将低照明问题转为有条件生成任务，使用空间-深度(S2D)设计的State Space Model以及噪声解耦的HVI监督。

**🔧 技术方法**

采用RWKV+S2D架构、FiLM条件调制、HVI颜色空间、噪声解耦监督以及轻量级参数。

**📊 数据集**

使用Light100（100个光照级别）以及传统LLIE基准（LOL、SID、SMID、SDSD）。

**📈 对比分析**

与多种SOTA（如RetinexMamba、MambaLLIE、GSAD等）比较，CLE-RWKV在PSNR/SSIM/LPIPS上均超越同类模型，并在实时性能上达到约150 FPS。

**⚠️ 局限性**

局限性包括对ISO、曝光等其他相机参数未建模，且在极低照明下仍受噪声限制。

---

## 303. Revealing the influence of participant failures on model quality in cross-silo Federated Learning

**arXiv ID:** 2603.25289 | [PDF](https://arxiv.org/pdf/2603.25289v1)

**作者:** Fabian Stricker `[一作]` (Hochschule Karlsruhe), Christian Zirpins `[通讯]` (Hochschule Karlsruhe)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

对交叉硅片联邦学习中参与者失败的影响进行了系统实验与分析，评估其对模型训练、评估及对失效参与者自身使用的影响。

**💡 创新点**

提出并量化了一组“失败影响修饰符”（FIM），涵盖数据模态、数据倾斜、模型架构、参与时机及参与者数量等维度，并将这些修饰符与失败效果关联；同时探讨了 Shapley 值在预测失败影响中的作用。

**🔧 技术方法**

采用 Federated Averaging (FedAvg) 与 Adam 优化器进行联邦训练；利用 Dirichlet 分布生成非 IID 数据分布；通过 Shapley 值估计方法评估参与者贡献；实验设置中模拟缺失参与者并记录不同失效时机。

**📊 数据集**

使用图像数据集 CIFAR-10、CIFAR-100；表格数据集 Adult、Covertype；时间序列数据集 PVOD、GermanSolarFarm；并在这些多模态数据上执行实验。

**📈 对比分析**

将缺失参与者情形与无失效基线进行对比；采用宏 F1 分数（分类）和 R² 分数（回归）衡量性能；实验表明：数据倾斜度越大、缺失时机越关键、某些模型架构（如 LSTM）更具鲁棒性；但缺失参与者会导致评估过乐观和整体性能下降，且对失效参与者本身的模型可用性受其他参与者分布影响。

**⚠️ 局限性**

局限性包括：仅评估了部分修饰符和失效模式；只考虑单一协调器与 FedAvg，未探究更复杂的聚合或分布式架构；实验集束在少数公开数据集上，结果可能不完全泛化；缺失恢复机制仅在训练后期实现，未覆盖长期或频繁失效情形。

---

## 304. List Estimation

**arXiv ID:** 2603.25280 | [PDF](https://arxiv.org/pdf/2603.25280v1)

**作者:** Nikola Zlatanov `[一作]` (Innopolis University), Mikhail Rudakov `[通讯]` (Innopolis University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了k-list估计，即从单个观测产生k个候选估计，并以最优候选的平方误差来衡量性能。

**💡 创新点**

将k-list估计与去中心化MMSE基准进行比较，证明中心化列表在高k下可实现k^{-2/d}的误差衰减指数，甚至在某些情况下优于基准，并给出了精确的高阶定量分析。

**🔧 技术方法**

采用后验向量量化与高阶逼近理论、small-ball概率分析以及Gaussian模型的解析推导来得到误差极限和常数。

**📊 数据集**

主要在仿真中使用标准高斯观测模型（无公开数据集）。

**📈 对比分析**

通过理论推导得到中心化误差的k^{-2/d}上限和去中心化误差的下界，并在数值实验中验证指数匹配；中心化估计在大k时的性能与理论一致，表现优良。

**⚠️ 局限性**

局限在于仅分析了对称MMSE基准，未给出最优去中心化设计；并且只在光滑连续误差分布下成立，对离散、奇异或不可微分分布缺乏分析。

---

## 305. Mitigating Evasion Attacks in Fog Computing Resource Provisioning Through Proactive Hardening

**arXiv ID:** 2603.25257 | [PDF](https://arxiv.org/pdf/2603.25257v1)

**作者:** Younes Salmi `[一作]` (Poznan University of Technology), Hanna Bogucka `[通讯]` (Poznan University of Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究了在雾计算资源调度系统中基于 k‑means 聚类的工作负载分配模型，并针对该模型设计了多阶段攻击（探索、规避、因果）与主动对抗训练防御方案。

**💡 创新点**

创新点在于首次将多阶段对抗机器学习攻击与主动对抗训练相结合，用于硬化 k‑means 资源调度器，并通过 MITRE ATLAS 框架系统化地描述攻击流程。

**🔧 技术方法**

主要技术包括 k‑means 在线/离线聚类、PGD 对抗样本生成、对抗训练、以及对模型决策边界的探索式逆向工程。

**📊 数据集**

实验使用自行生成的雾计算工作负载与 VM 资源数据集，特征经过归一化处理后用于训练和攻击仿真。

**📈 对比分析**

与传统固定 VM 分配方案对比，鲁棒模型在攻击情境下保持约 98% 的资源利用率（RU），任务丢弃率（TD）下降到 6%，显著恢复了系统性能。

**⚠️ 局限性**

局限性包括仅在仿真环境下验证，未考虑真实流量和更复杂的网络拓扑；攻击效果受均匀数据分布影响；对抗训练计算复杂度随聚类数和特征维度线性增长。

---

## 306. An Image Dataset of Common Skin Diseases of Bangladesh and Benchmarking Performance with Machine Learning Models

**arXiv ID:** 2603.25229 | [PDF](https://arxiv.org/pdf/2603.25229v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 307. Hyperspectral Trajectory Image for Multi-Month Trajectory Anomaly Detection

**arXiv ID:** 2603.25255 | [PDF](https://arxiv.org/pdf/2603.25255v1)

**作者:** Md Awsafur Rahman `[一作]` (University of California Santa Barbara), B. S. Manjunath `[通讯]` (University of California Santa Barbara)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了TITAnD框架，将多月GPS轨迹转换为超光谱轨迹图像，使用循环分解Transformer实现密集轨迹异常检测。

**💡 创新点**

创新点在于将轨迹数据重新表征为二维图像并引入循环分解注意力，既统一稠密与稀疏轨迹，又显著降低计算成本。

**🔧 技术方法**

采用Hyperspectral Trajectory Image、稀疏与密集轨迹特征编码、Cyclic Factorized Transformer、RoPE位置编码以及卷积/自注意力混合的网络结构。

**📊 数据集**

在Los Angeles NumoSim‑LA稀疏轨迹、东京私有密集GPS数据以及其稀疏化版本Dense2Sparse Tokyo上进行评估。

**📈 对比分析**

与USTAD、ICAD等稀疏方法及Transformer、UNet、SegFormer等HTI基线在AUC‑PR和mIoU上对比，CFT在稠密轨迹的临时AUC提升至0.84，稀疏轨迹AUC达0.63，速度比平面Transformer提升11–75倍，参数仅6.5M。

**⚠️ 局限性**

局限在于使用合成注入的异常、缺乏真实标注的大规模公开数据、以及对东京地区数据的过度依赖。

---

## 308. MolQuest: A Benchmark for Agentic Evaluation of Abductive Reasoning in Chemical Structure Elucidation

**arXiv ID:** 2603.25253 | [PDF](https://arxiv.org/pdf/2603.25253v1)

**作者:** Taolin Han `[一作]` (Alibaba Group), Wei Hu `[通讯]` (Alibaba Group)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了MolQuest基准，评估LLM在化学结构阐释中的主动推理与实验规划能力。

**💡 创新点**

创新点在于将结构阐释转为多轮交互、实时实验请求的代理式框架，并使用真实实验数据。

**🔧 技术方法**

采用LLM代理、工具调用（模拟实验仪器）、状态机交互、链式思考和自我校准评估等技术。

**📊 数据集**

使用从2025年后化学文献支持信息提取的530个真实案例，涵盖多模态谱图。

**📈 对比分析**

对12种最前沿LLM进行Agent与Baseline对比，结果显示Gemini 3系列最高约51%准确率，动态交互对部分模型提升显著。

**⚠️ 局限性**

局限包括仅聚焦小分子阐释、评估以最终准确率为主、成本模型抽象、数据量有限且未覆盖更大化学空间。

---

## 309. ColBERT-Att: Late-Interaction Meets Attention for Enhanced Retrieval

**arXiv ID:** 2603.25248 | [PDF](https://arxiv.org/pdf/2603.25248v1)

**作者:** Raj Nath Patel `[一作]` (Huawei Research Center), Sourav Dutta `[通讯]` (Huawei Research Center)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种在Late Interaction框架中显式融合查询和文档注意力权重的检索模型，并在ColBERT的MaxSim操作中加入注意力加权与文档长度正则化。

**💡 创新点**

在MaxSim中加入查询/文档注意力权重，并通过长度基正则化解决训练-推理注意力差异；首次将注意力机制与Late Interaction结合。

**🔧 技术方法**

基于ColBERTv2_PLAID的深度检索模型，使用token级向量嵌入、最大相似度(MaxSim)、注意力权重指数化、文档长度正则化，以及MS-MARCO训练数据的triplet loss。

**📊 数据集**

在MS‑MARCO dev、BEIR搜索与语义检索任务、LoTTE搜索与论坛数据集上进行评估。

**📈 对比分析**

与ColBERTv2_PLAID、ColBERT、BM25、ANCE、RocketQAv2等基线对比，MS‑MARCO R@100提升0.2%；LoTTE Success@5提升约1%；BEIR多数据集平均提升约1‑2%，尤其在ArguAna上提升约2%。

**⚠️ 局限性**

模型基于ColBERTv2_PLAID，未使用原始ColBERTv2实现；提升幅度有限；注意力正则化的阈值需手工调优，且在极长或极短文档上仍可能产生注意力分布不一致。

---

## 310. Training-free Detection and 6D Pose Estimation of Unseen Surgical Instruments

**arXiv ID:** 2603.25228 | [PDF](https://arxiv.org/pdf/2603.25228v1)

**作者:** Jonas Hein `[一作]` (University Hospital Balgrist, University of Zurich), Philipp Fürnstahl `[通讯]` (University Hospital Balgrist, University of Zurich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一套训练无关的多视角检测与6D位姿估计管线，仅利用CAD模型进行推理。

**💡 创新点**

创新点在于将基础模型（SAM2、DINOv2）与多视角几何一致性相结合，并引入遮挡感知的轮廓基位姿微调，首次实现对未见手术器械的毫米级精度位姿估计。

**🔧 技术方法**

技术包括：SAM2生成无类别掩模、DINOv2特征相似度评分、跨视角三角化与聚类、扩展的FoundationPose多视角粗估、跨视角注意力评分，以及基于轮廓的多视角重投影优化。

**📊 数据集**

实验数据集为真实手术场景的MVPSP数据集以及MedShapeNet合成数据集。

**📈 对比分析**

与监督学习基线对比，5视角条件下训练自由方法在未见器械上实现毫米级精度，误差仅比监督方法高3-10倍；但在2视角或检测质量低时性能受限。

**⚠️ 局限性**

主要局限包括：检测阶段误检率高、遮挡与视觉相似导致误分、粗估精度有限、轮廓匹配收敛性弱以及整体多步骤导致信息流隔离，基础模型的泛化能力仍不足。

---

## 311. Translation or Recitation? Calibrating Evaluation Scores for Machine Translation of Extremely Low-Resource Languages

**arXiv ID:** 2603.25222 | [PDF](https://arxiv.org/pdf/2603.25222v1)

**作者:** Danlu Chen `[一作]` (University of California San Diego), Freda Shi `[通讯]` (University of Waterloo)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究极低资源机器翻译（XLR MT）中性能波动的根本原因，提出并评估了四个数据集内在难度指标（Fertility Ratio、Retrieval Proxy、Pre‑training Exposure、Corpus Diversity），通过限制五种高资源语言的训练数据量（10k 句）构建基准，并将这些指标应用于多种古老、土著及非洲/印度等极低资源语言，比较模型（主要是 fine‑tuned mBART）与检索基准的表现。

**💡 创新点**

创新点在于：①首次将数据集内部特征量化为四个可计算的难度指标，直接解释模型表现差异；②通过高资源语言的低资源切片提供可控基准，使得跨语言评估更具可比性；③发现检索基准（R‑score）是解释 XLR MT 变异的最强预测因子；④揭示低资源语言在词汇重叠、标记化效率和预训练覆盖方面的系统性瓶颈。

**🔧 技术方法**

主要技术包括：基于 mBART 的 fine‑tuning、句子级检索基准计算、token‑to‑character 费率统计、n‑gram 预训练重叠计数、self‑BLEU 语料多样性测量、回归与 R² 分析，以及多语言实验与可视化。

**📊 数据集**

使用的语料集：高资源语言（芬兰语、中文、阿拉伯语、日语、印地语）各 10k 句；极低资源语言包括：古阿卡德语、古埃及语、台湾原住民（陶语）等；美洲原住民（Shp、Hch、Quy、Guc）、非洲语言（豪萨、祖鲁、班巴拉）、印度土著（Mni、Kha）；预训练语料用于 E‑score 计算，采用公开的多语言预训练语料库。

**📈 对比分析**

比较方法：将 XLR 语言的 BLEU/ChrF 与高资源基准相对照，并与检索基准（R‑score）以及其他难度指标进行相关性和回归分析。结果显示 R‑score 解释约 58% 的性能方差；部分古老语言因高 R‑score 取得超出基准的 BLEU（>40）；而一些土著语言因高 token fertility 与低 E‑score 仍低于基准，且部分模型甚至未能超过检索基准。

**⚠️ 局限性**

局限性包括：BLEU/ChrF 在不同语言间的可比性受限；FRED 指标虽能解释大部分方差，但对词汇重叠和标记化的量化仍不完善；高资源基准仅模拟数据量限制，无法覆盖所有极低资源的语言生态和域差；缺乏对单语料对模型性能影响的深入分析；未来工作需进一步完善 E‑score 与单语料的关联度，并探索更细粒度的多语种评估方法。

---

## 312. Activation Matters: Test-time Activated Negative Labels for OOD Detection with Vision-Language Models

**arXiv ID:** 2603.25250 | [PDF](https://arxiv.org/pdf/2603.25250v1)

**作者:** Yabin Zhang `[一作]` (Harbin Institute of Technology), Curtis Langlotz `[通讯]` (Stanford University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种测试时动态挖掘激活负标签的零训练 OOD 检测方法（TANL）。

**💡 创新点**

创新点在于使用激活度量在线选择对 OOD 具有高响应且对 ID 低响应的负标签，并通过激活感知得分函数提升检测鲁棒性。

**🔧 技术方法**

主要技术包括 CLIP 视觉-文本模型、激活度量、FIFO 队列在线更新、批量适配、激活感知得分、零训练和零样本推理。

**📊 数据集**

使用的数据集包括 ImageNet-1k 作为 ID 数据集，以及多个 OOD 数据集（如 Places、SUN、Texture 等）、ImageNet-C、R、V2、CIFAR、医学影像等。

**📈 对比分析**

在 ImageNet、OpenOOD、全光谱等多种设置下，与多种训练/无训练/适配方法对比，TANL 将 FPR95 降至 0.42，AUROC 接近 100，显著优于现有方法，尤其在近 OOD 场景中表现突出。

**⚠️ 局限性**

局限性包括需要预先构建大型语料库生成负标签，对极端类别分布变化的适应性有限，且对批量大小和队列长度等超参数较为敏感。

---

## 313. Offline Decision Transformers for Neural Combinatorial Optimization: Surpassing Heuristics on the Traveling Salesman Problem

**arXiv ID:** 2603.25241 | [PDF](https://arxiv.org/pdf/2603.25241v1)

**作者:** Hironori Ohigashi `[一作]` (Panasonic Connect Co., Ltd.), Shinichiro Hamada `[通讯]` (Panasonic Connect Co., Ltd.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

利用离线强化学习框架Decision Transformer，从现有启发式解集学习并生成优于原始启发式的TSP解。

**💡 创新点**

创新点在于：①将Pointer Network与DT结合，解决TSP可变动作空间；②使用期望回归（expectile regression）预测Return-to-Go（RTG），实现对目标的乐观估计；③通过离线RL直接利用已有启发式数据，超越传统行为克隆。

**🔧 技术方法**

采用Decision Transformer、Transformer Encoder/Decoder、Pointer Network、期望回归以及交叉熵+期望回归的多任务损失进行训练。

**📊 数据集**

使用Joshi等人提供的2D欧氏TSP数据集（N=20、50、100），每个大小包含1,000,000训练实例，分别由四种启发式（NN、NI、FI、SA）生成。

**📈 对比分析**

与原始启发式、行为克隆（BC）进行最优性缺口比较。实验表明DT在所有训练启发式上均能显著降低最优性缺口，尤其在SA训练集上提升约一倍；相比BC亦表现更好，证明RTG条件对性能提升至关重要。

**⚠️ 局限性**

局限性包括：1）RTG预测仍偏保守，尤其在大规模实例上可能低估最优回报；2）单一启发式（如NN）多样性不足，限制模型提升空间；3）对RTG预测的精度有待进一步改进，可能需结合多源或专家数据。

---

## 314. A Unified Spatial Alignment Framework for Highly Transferable Transformation-Based Attacks on Spatially Structured Tasks

**arXiv ID:** 2603.25230 | [PDF](https://arxiv.org/pdf/2603.25230v1)

**作者:** Jiaming Liang `[一作]` (University of Macau), Chi-Man Pun `[通讯]` (University of Macau)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出统一的空间对齐框架（SAF），使得现有的变换攻击可以直接用于语义分割和目标检测等结构化任务。

**💡 创新点**

创新点在于发现并解决了空间变换导致的标签空间错位问题，通过同步变换标签实现空间对齐，显著提升了变换攻击在结构化任务中的可迁移性。

**🔧 技术方法**

结合多种变换攻击技术（如DIM、SIA、BSR、I-C 等）与空间对齐算法，采用梯度累计的攻击优化流程。

**📊 数据集**

使用 Cityscapes、Kvasir‑SEG 和 MS COCO 三个数据集进行评估。

**📈 对比分析**

与原始 TAA、其他对齐方法的对比实验显示，SAF 在非目标和目标攻击中将 mIoU 和 mAP 明显下降（即攻击效果提升），在多模型上均优于传统方法。

**⚠️ 局限性**

局限在于仅验证了空间对齐的有效性，未深入探讨不同任务、不同变换组合的最佳设置，也未考虑更复杂的防御策略或更大规模数据的实验。

---

## 315. Comparing Natural and Synthetic Structured Data: A Study of the Passive Verb Alternation in French and Italian

**arXiv ID:** 2603.25227 | [PDF](https://arxiv.org/pdf/2603.25227v1)

**作者:** Giuseppe Samo `[一作]`, Paola Merlo `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究通过在BLM框架下系统比较自然语料和人工生成语料对语言模型识别被动变位的效果。

**💡 创新点**

创新点在于提出多语言结构化测试套件，并将自然与合成数据在同一框架下对照，揭示自然数据在泛化上的优势。

**🔧 技术方法**

使用前馈神经网络探测器与预训练ELECTRA嵌入，评估模型在不同数据训练和测试下的表现。

**📊 数据集**

使用从Universal Dependencies（法语、意大利语）检索的自然句子和由DeepSeek‑V3生成的合成句子，形成两个BLM数据集。

**📈 对比分析**

比较方法是SynSyn、NatNat、SynNat、NatSyn四种训练/测试组合，结果显示合成训练在分布内表现最佳，但对自然测试失效；自然训练则在所有条件下稳健，尤其在跨域泛化上表现优异。

**⚠️ 局限性**

局限在于仅覆盖两种罗曼语和被动变位，未探究更广语言和变体，以及对模型容量的影响。

---

## 316. CRAFT: Grounded Multi-Agent Coordination Under Partial Information

**arXiv ID:** 2603.25268 | [PDF](https://arxiv.org/pdf/2603.25268v1)

**作者:** Abhijnan Nath `[一作]` (Colorado State University), Nikhil Krishnaswamy `[通讯]` (Colorado State University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 CRAFT，一个多代理基准，用于评估在部分可观测、基于 3D 结构构造任务中大语言模型的务实沟通与协作能力。

**💡 创新点**

创新点在于提出多发言者受限的语用说话人框架、分离思考/消息生成的机制，以及基于 LLM 的自动评判器，从而揭示个体语用实力不一定导致更好的多代理协作。

**🔧 技术方法**

使用 Bounded Pragmatic Speaker（BPS）理论、Rational Speech Acts、LLM 思考/消息分离架构，并以 GPT‑4o‑mini 作为评判者。

**📊 数据集**

使用随机生成的 3×3 立方体 3D 结构数据集，包含简单、中等、复杂共 20 个结构作为评估对象。

**📈 对比分析**

对 15 种模型（8 开源、7 专有）在 20 回合内进行对比，发现专有模型在个体沟通指标上得分更高，却在整体任务完成度和进展速度上落后多数开源模型。

**⚠️ 局限性**

局限在文本-only、固定 builder 与 oracle‑辅助候选、未涉及视觉输入以及异构模型协作场景。

---

## 317. Towards Practical Lossless Neural Compression for LiDAR Point Clouds

**arXiv ID:** 2603.25260 | [PDF](https://arxiv.org/pdf/2603.25260v1)

**作者:** Pengpeng Yu `[一作]` (Sun Yat-sen University), Yulan Guo `[通讯]` (Sun Yat-sen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种面向LiDAR点云的实时可跨平台神经压缩框架，采用几何再密集化（GRED）和跨尺度特征传播（XFP）两大模块实现高精度无损压缩，并实现整数化推理以确保跨平台一致性。

**💡 创新点**

创新点在于：1）通过在编码过程中动态再密集化稀疏几何，缓解高分辨率上下文稀疏（HRCS）问题；2）设计跨尺度特征传播模块，在浅层与深层之间共享信息，提升上下文利用率；3）实现全整数推理管线，消除浮点不确定性，确保比特流可跨平台解码。

**🔧 技术方法**

主要技术包括：Octree结构编码、稀疏卷积与重密集化、轻量化多层感知机预测、交叉尺度特征融合、整数化量化与固定点Softmax、基于熵编码的无损压缩。

**📊 数据集**

在KITTI（22个序列共43,552帧）和Ford（3个序列共4,500帧）LiDAR数据集上进行训练与评估，采用点到点PSNR与点到平面PSNR评测。

**📈 对比分析**

与G-PCC、OctAttention、Light EHEM、Unicorn、RENO等主流方法对比，浮点模型在KITTI上BD-Rate下降约12%~21%，BD-PSNR提升约1.8~2.5 dB；整数模型保持近似性能，且编码速度最高可达14 FPS，编码/解码时间比现有实时方法更快或相近。

**⚠️ 局限性**

局限性在于：1）在样本量较小的Ford数据集上泛化性略逊于KITTI；2）对极高分辨率点云的压缩性能尚未充分验证；3）整数化过程中仍需在多平台上进行精细的量化校准，可能增加部署复杂度。

---

## 318. Efficient Preemptive Robustification with Image Sharpening

**arXiv ID:** 2603.25244 | [PDF](https://arxiv.org/pdf/2603.25244v1)

**作者:** Jiaming Liang `[一作]` (University of Macau), Chi-Man Pun `[通讯]` (University of Macau)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了利用传统拉普拉斯锐化（Laplacian sharpening）作为预攻击防御（preemptive robustification）方法，直接对原始图像进行轻微锐化以提升模型对对抗样本的鲁棒性。

**💡 创新点**

创新点在于：①首次将传统锐化技术用于预攻击鲁棒化，完全不依赖对抗训练、优化或生成器；②方法无模型或计算开销，直观可解释；③通过实验验证锐化可显著提升对各种攻击的防御效果，尤其是高可迁移性攻击。

**🔧 技术方法**

使用的技术主要是经典的8邻域拉普拉斯卷积核，对图像加权后添加到原图；此外对不同锐化强度α进行调参并与多种对抗攻击、模型和任务进行对照。

**📊 数据集**

使用的数据集包括：NIPS 2017 Adversarial Competition（分类）、Cityscapes（语义分割）和MS COCO（目标检测）。

**📈 对比分析**

对比方法包括多种基于梯度、变换、目标优化、模型相关、目标攻击及集成攻击；在黑盒、白盒及集成攻击下，锐化平均可提升 10–15% 的准确率或降低攻击成功率，且与对抗训练联合使用能进一步提升鲁棒性。

**⚠️ 局限性**

局限性：锐化对标准准确率影响有限，但随着锐化强度增大会出现轻微的性能下降；对分割与检测任务的鲁棒提升相对有限，主要因这些攻击的可迁移性弱；未来需要探索更强的锐化算子或跨模态推广。

---

## 319. JSON Schema Inclusion through Refutational Normalization: Reconciling Efficiency and Completeness

**arXiv ID:** 2603.25306 | [PDF](https://arxiv.org/pdf/2603.25306v1)

**作者:** Mohamed-Amine Baazizi `[一作]`, Stefanie Scherzinger `[通讯]`

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种新的 JSON Schema 包含性检查方法，称为 Refutational Normalization，旨在兼顾规则基础方法的高效性与 witness‑generation 方法的完备性。

**💡 创新点**

创新点：①将规则系统的推理思路嵌入到归一化算法中，形成“refutational normalization”；②引入懒惰归一化、快速补吸收、急切引用评估、快速组件访问和 […]‑优化等技术，实现对复杂布尔结构的高效处理；③在保证完备的前提下，针对规则可证的情况实现与规则基础方法相同的时间复杂度。

**🔧 技术方法**

主要技术手段包括：
- 懒惰归一化（lazy normalization）
- 快速补吸收（fast complement‑absorption）
- 急切引用评估（eager reference‑evaluation）
- 快速组件访问（rapid component access）
- […]‑优化（[…]‑optimization）
- 结合了 DNF 计算、规则推理和可满足性检查的综合算法。

**📊 数据集**

使用的数据集：
- SchemaStore 版本（包括 1,056 个版本对）
- MergeAllOf、Synthesized、Handwritten SC、-testset
- oneOf as anyOf、uneval as additional、additional as uneval
- 进一步对大型（>25KB）子集的评测。
这些数据集涵盖了真实版本演进、工具验证与关键词使用分析等三大应用场景。

**📈 对比分析**

比较方法：与现有的两种工具（规则基础的 jsonsubschema、完整 witness‑generation）在同一硬件与 10 分钟超时设置下进行对比。结果显示：
- 完备性方面：Refutational Normalization 在所有数据集上都实现了零运行时或逻辑错误，显著提升了可覆盖率；
- 性能方面：在规则可证的案例中，平均耗时与 jsonsubschema 相当；在大多数复杂案例中，耗时比 witness‑generation 低 5–10 倍，超时率大幅下降；
- 对大型 (>25KB) 子集的评测表明，该方法在保持完备性的同时还能在多实例中完成 85–95% 的检查。

**⚠️ 局限性**

局限性：
- 仍然存在指数级扩张风险，尤其在包含大量布尔组合且无规则可证的情况；
- 目前仅支持 Draft‑06 规范，尚未完整覆盖现代关键字（如 unevaluatedItems、unevaluatedProperties 等）；
- 递归模式在某些极端情况下可能导致深度堆栈溢出或内存占用高；
- 实现依赖 Scala 3.3，若需迁移到其他语言需重新实现归一化细节。

---

## 320. Learning in Proportional Allocation Auctions Games

**arXiv ID:** 2603.25303 | [PDF](https://arxiv.org/pdf/2603.25303v1)

**作者:** Younes Ben Mazziane `[一作]` (Avignon university), Francesco De Pellegrini `[通讯]` (Avignon university)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文研究了基于Kelly比例分配机制的重复竞价游戏，探讨了在对数型效用下玩家如何通过在线学习和最佳响应策略来更新投标，并证明在满足Rosen的对角严格凹性（DSC）条件时，所有这些学习算法收敛到唯一的纳什均衡。

**💡 创新点**

创新点在于：①给出了一个可计算且仅需O(n)时间的充分条件，证明对数型效用下游戏满足r-DSC并拥有唯一均衡；②在此框架下，证明了在线梯度下降（OGD）和双重平均（DA）等经典无后悔算法在异步和异步步长下仍能收敛；③首次将最佳响应动态在该游戏中作为收敛到NE的线性速率方法；④通过大量仿真验证了不同学习率、步长和异质更新规则的收敛速度与时间平均效用。

**🔧 技术方法**

主要技术包括：游戏理论中的对角严格凹性分析、Rosen的对角严格凹性与单调性等概念；无后悔学习算法（OGD、DA、RRN）和最佳响应动态的连续时间/离散时间收敛性证明；使用雅可比矩阵的收缩性质来证明最佳响应的线性收敛；以及通过仿真验证收敛指标（固定点残差、时间平均收益）与不同参数（γ、n、学习率）之间的关系。

**📊 数据集**

本文使用的是仿真数据，模拟了10个代理在预算约束下进行重复Kelly拍卖，效用为对数函数，参数包括预算c=400、最小投标ϵ=1、δ=0.1等；通过多次随机初始投标生成实验结果。

**📈 对比分析**

通过比较固定步长（F）与随时间衰减步长（V）的OGD、DA、RRN以及最佳响应四种算法，使用固定点残差r_t和时间平均收益作为性能指标。结果表明：OGD最先收敛、最佳响应最快、DA次之，而RRN收敛最慢；在异质更新规则下，收敛到均衡可能失败，但时间平均收益仍接近均衡收益。

**⚠️ 局限性**

限制与未来工作：①缺乏对异质学习规则影响的理论框架；②只考虑单一资源和对数型效用，未扩展到α-公平或多资源情形；③仿真基于理想化假设，未验证在真实网络环境中的鲁棒性。

---

## 321. Usability of Passwordless Authentication in Wi-Fi Networks: A Comparative Study of Passkeys and Passwords in Captive Portals

**arXiv ID:** 2603.25290 | [PDF](https://arxiv.org/pdf/2603.25290v1)

**作者:** Martiño Rivera-Dourado `[一作]` (Universidade da Coruña), Jose Vázquez-Naya `[通讯]` (Universidade da Coruña)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在实验室环境中对比了使用 FIDO2 Passkeys 与传统密码在 Wi‑Fi 捕获门户中的可用性，包含注册与登录两项任务。

**💡 创新点**

首次将 FIDO2CAP 协议嵌入捕获门户并通过 split‑plot 设计比较两种身份验证方法，重点考察操作系统差异与平台兼容性。

**🔧 技术方法**

技术手段包括 FIDO2 (WebAuthn + CTAP)、Windows Hello/Android 平台认证器、OpenNDS + Zitadel IAM、FIDO2CAP 协议实现，以及 SUS、SEQ 等可用性评估工具。

**📊 数据集**

数据来源于 50 名受试者在实验中的屏幕录像、错误记录、时长统计及问卷调查，未使用公开数据集。

**📈 对比分析**

通过任务完成率、错误率、时长、SUS/SEQ 分数进行对比，发现 Passkeys 在 Windows 上略优，Android 上差异更大，但整体差异无统计显著性。

**⚠️ 局限性**

局限性包括：受试者技术水平偏高、实验环境与真实网络差异、Android CPD 浏览器不支持 WebAuthn 导致额外跳转、仅使用平台认证器、样本量有限、未评估长期使用效果。

---

## 322. CSI-tuples-based 3D Channel Fingerprints Construction Assisted by MultiModal Learning

**arXiv ID:** 2603.25288 | [PDF](https://arxiv.org/pdf/2603.25288v1)

**作者:** Chenjie Xie `[一作]` (Southeast University), Xiqi Gao `[通讯]` (Southeast University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了低空通信中的三维通道指纹（3D‑CF）构建，提出基于CSI‑tuples的模型并设计了模块化的多模态学习框架。

**💡 创新点**

创新点在于将低空地理环境、测量数据与LAV位置视为不同模态，构建Corr‑MMF、MMR和CSI‑R模块实现对CSI分布的横向与纵向特征融合，并在CSI‑tuples框架下实现无格网的高精度3D‑CF构建。

**🔧 技术方法**

采用多模态学习（Corr‑MMF、MMR、CSI‑R）、卷积网络、注意力机制（TAM、CAM、SAM）、自编码器与全连接回归等技术，并使用Sionna射线追踪生成的低空通信通道数据进行训练。

**📊 数据集**

数据集基于南京的OpenStreetMap地理地图在Blender中构建场景，通过NVIDIA Sionna射线追踪得到低空通信通道、RSS样本和地理环境图，涵盖宏观与微观两类城市场景。

**📈 对比分析**

与Kriging插值、GPR、GAN和联邦学习等四种基准对比，实验显示多模态框架在RMSE/MAE上分别比最优基准低约27.5%和52.2%，并在四个不同场景下均表现出较高的泛化能力。

**⚠️ 局限性**

局限性包括对测量数据分辨率影响的探讨不足、仅考虑地面到LAV的Rician信道未覆盖空对空或动态环境，以及模型训练依赖强大算力，需在基站端部署。

---

## 323. A Gait Foundation Model Predicts Multi-System Health Phenotypes from 3D Skeletal Motion

**arXiv ID:** 2603.25283 | [PDF](https://arxiv.org/pdf/2603.25283v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 324. SliderQuant: Accurate Post-Training Quantization for LLMs

**arXiv ID:** 2603.25284 | [PDF](https://arxiv.org/pdf/2603.25284v1)

**作者:** Shigeng Wang `[一作]` (Intel Labs China), Anbang Yao `[通讯]` (Intel Labs China)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对大语言模型的后训练量化（PTQ），提出了一种基于滑动窗口的自适应量化框架SliderQuant，能够在不同层次（浅层、深层、中间层）上采用不同窗口大小和量化频率，显著降低量化误差并保持模型性能。

**💡 创新点**

创新点在于引入了两级滑动量化：inter‑layer滑动窗口（对浅层使用递增窗口，对深层使用递减窗口，对中间层使用固定窗口）和intra‑layer滑动细化（在每个窗口内部按比例分阶段量化），以及结合通道缩放与LoRA来抑制权重与激活的离群值。

**🔧 技术方法**

技术手段包括：自适应滑动窗口量化、通道缩放（CS）、低秩适配（LoRA）、基于均方误差的量化目标、旋转变换（在SliderQuant+中可选）以及使用统一量化器实现权重和激活的量化。

**📊 数据集**

数据集与任务涵盖：语言生成（WikiText2、C4）、零样本常识推理（PIQA、ARC、HellaSwag、Winogrande、BoolQ、MMLU）、数学推理（MATH‑500、AIME‑2024、GSM8K）和代码生成（HumanEval+、MBPP+），以及对多模型家族（Llama、Llama2、Llama3、Qwen2.5、Qwen3、DeepSeek‑R1、MoE Qwen3‑30B‑A3B）的实验。

**📈 对比分析**

与多种主流PTQ方法（GPTQ、AWQ、SmoothQuant、OmniQuant、CBQ、SpQR、LLM‑MQ、QUIK、QLLM等）进行比较，SliderQuant在多数低位宽设置（如W4A4、W4A16、W2A16）下均实现了更低的perplexity和更高的零样本推理准确率，甚至在极低位宽（W2A16）下仍保持接近FP16的性能。

**⚠️ 局限性**

局限性包括：相较于固定窗口量化，滑动窗口会增加量化时的内存和计算开销；在极低位宽（如W2A16）下对某些模型仍有一定性能损失；方法对校准样本的依赖性较高，若校准数据不足可能影响量化效果。

---

## 325. Connectivity-Aware Representations for Constrained Motion Planning via Multi-Scale Contrastive Learning

**arXiv ID:** 2603.25298 | [PDF](https://arxiv.org/pdf/2603.25298v1)

**作者:** Suhyun Jeon `[一作]` (Seoul National University), Jaeheung Park `[通讯]` (Seoul National University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

论文提出了一种多尺度对比学习的连通性感知表示，用来在约束运动规划前选择起止配置，从而显著提升规划成功率和速度。

**💡 创新点**

创新点在于将多尺度流形学习与聚类产生伪标签，再通过多尺度对比学习训练特征空间，使距离与连通性直接相关，实现连通性感知的预规划选择。

**🔧 技术方法**

主要技术包括UMAP多尺度流形嵌入、HDBSCAN聚类生成伪标签、基于InfoNCE的多尺度对比学习、深度MLP特征编码器以及null-space数据增强。

**📊 数据集**

使用在Franka Panda 7-DoF 机器人和Tocabi 8-DoF 人形机器人上构建的约束操控任务数据集（杯子转移、托盘转移、三臂旋转等），并以采样的IK解作为训练集。

**📈 对比分析**

与随机选择、关节空间距离选择以及传统CBiRRT和Latent Motion Planner 对比，方法在100场景、100s限制下成功率提升至100%（相较于随机约60–84%），规划时间缩短至约原来的0.43倍，平均成功率提升约1.9倍。

**⚠️ 局限性**

局限性包括仅针对当前约束配置训练，需重新训练；在高维复杂约束下全局连通性仍难以完全捕捉，且对规划器本身的限制仍会影响最终成功率。

---

## 326. DAGverse: Building Document-Grounded Semantic DAGs from Scientific Papers

**arXiv ID:** 2603.25293 | [PDF](https://arxiv.org/pdf/2603.25293v1)

**作者:** Shu Wan `[一作]` (Arizona State University), Huan Liu `[通讯]` (Arizona State University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了Doc2SemDAG框架，利用半自动Pipeline从科学论文中提取语义有向无环图（semantic DAG），并发布了-1基准数据集。

**💡 创新点**

创新点包括：提出将文档证据与图结构结合的语义DAG概念；设计多阶段半自动Pipeline；利用带有DAG图的科学论文作为真实监督，生成大规模真实语义DAG数据集。

**🔧 技术方法**

主要技术为：LLM（GPT‑5.1、Gemini 2.5 Pro）用于图分类与注释；VLM用于图像理解；Marker用于PDF解析；图结构重建与文本对齐；人工与LLM-as-a-judge进行验证。

**📊 数据集**

使用的数据集：arXiv（约2.7M篇）与bioRxiv（约400k篇）论文；合成Cladder DAG数据；最终发布-1数据集（108条经过专家验证的语义DAG）。

**📈 对比分析**

与零-shot VLM 基线对比，评估DAG分类精度（DC）、结构差异（SD）与证据对齐（EA）；Pipeline在DAG分类精度88%时恢复所有50个真实DAG，整体性能显著优于基线。

**⚠️ 局限性**

局限性包括：强调精度导致覆盖率不足；目前仅支持单一静态DAG，缺乏对多种图类型（如动态图、混合图）的处理；Pipeline依赖较旧的OCR/VLM/LLM组件，人工验证难以扩展。

---

## 327. WebTestBench: Evaluating Computer-Use Agents towards End-to-End Automated Web Testing

**arXiv ID:** 2603.25226 | [PDF](https://arxiv.org/pdf/2603.25226v1)

**作者:** Fanheng Kong `[一作]` (Northeastern University), Kun Gai `[通讯]` (Kuaishou Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了面向AI驱动Web开发的端到端自动化Web测试基准WebTestBench，并提出两阶段检索式检查表生成与缺陷检测框架

**💡 创新点**

首次实现无人工写测试用例的端到端评测，强调隐含逻辑约束和四维质量维度，并公开了高质量标注数据

**🔧 技术方法**

利用大型语言模型（Claude、GPT、Qwen等）与Playwright自动化执行的计算机使用代理实现检索、执行与评估

**📊 数据集**

基于100个由AI平台生成、人工重写指令的Web应用，配套功能/约束/交互/内容四维检查表与结果标注

**📈 对比分析**

与人工评测对比，使用覆盖率、F1等指标，所有模型F1低于30%，检索不足、缺陷误判与长交互不稳是主要瓶颈

**⚠️ 局限性**

局限性在于人工标注成本高、检查表与真实缺陷不完全覆盖、仅覆盖七类应用且不适用于高频动态交互

---

## 328. EagleNet: Energy-Aware Fine-Grained Relationship Learning Network for Text-Video Retrieval

**arXiv ID:** 2603.25267 | [PDF](https://arxiv.org/pdf/2603.25267v1)

**作者:** Yuhan Chen `[一作]` (Sun Yat-sen University), Xiaochun Cao `[通讯]` (Sun Yat-sen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一个基于细粒度关系学习与能量匹配的文本-视频检索框架，能够生成上下文感知的增强文本嵌入；

**💡 创新点**

创新点在于同时建模文本与视频帧之间的关系、帧与帧内部的时序关系，并通过能量模型捕捉细粒度匹配信息，最终使用 Sigmoid 损失实现更稳健的跨模态对齐；

**🔧 技术方法**

核心技术包括 CLIP 预训练视觉-语言编码器、关系图注意力网络（RGAT）、能量基础模型（EBM）与 Sigmoid 对比损失，以及多候选文本采样的随机文本建模；

**📊 数据集**

在四大公共检索基准上进行实验，数据集分别为 MSRVTT、DiDeMo、MSVD 与 VATEX；

**📈 对比分析**

与现有 SOTA 方法（如 CLIP4Clip、XPool、TMASS、Video‑ColBERT 等）进行对比，平均 R@1 提升约 1.5‑2.5%，Rsum 领先，验证了框架在多数据集上的普适性和优势；

**⚠️ 局限性**

局限性包括对计算资源要求高（需要多 GPU 与 MCMC 采样），以及在部分数据集上对帧数、文本长度的敏感性，未来可进一步简化模型并提升推理速度。

---

## 329. Probabilistic Abstract Interpretation on Neural Networks via Grids Approximation

**arXiv ID:** 2603.25266 | [PDF](https://arxiv.org/pdf/2603.25266v1)

**作者:** Zhuofan Zhang `[一作]` (Imperial College London), Herbert Wiklicky `[通讯]` (Imperial College London)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文将概率抽象解释（Probabilistic Abstract Interpretation）理论应用于神经网络，构建网路输入空间的概率分布流动模型并分析其传播；

**💡 创新点**

创新点在于将概率抽象解释与传统抽象解释相结合，利用Moore‑Penrose伪逆构造抽象变换器，并在高维网络中通过网格近似（grid approximation）实现对输入分布的离散化，提升对密度分布的可解释性；

**🔧 技术方法**

使用的技术包括：概率抽象解释框架、Moore‑Penrose伪逆、抽象域（如离散符号域、Zonotope、网格域）、卷积层与全连接层的抽象变换器、以及基于网格的近似处理；

**📊 数据集**

实验以MNIST手写数字识别数据集为例，构建8层卷积网络进行验证；

**📈 对比分析**

评估方式为对比传统基于抽象解释的鲁棒性分析，结果显示概率抽象解释能够提取网络特征并呈现输入分布流向，表明在信息可解释性方面具有优势，但未给出具体性能数值；

**⚠️ 局限性**

局限性包括：高维网络中抽象域维度仍大且计算开销高；仅讨论了网格近似，未涵盖更多抽象域；实验案例维度低，需进一步验证在更复杂网络与数据集上的效果。

---

## 330. XBRLTagRec: Domain-Specific Fine-Tuning and Zero-Shot Re-Ranking with LLMs for Extreme Financial Numeral Labeling

**arXiv ID:** 2603.25263 | [PDF](https://arxiv.org/pdf/2603.25263v1)

**作者:** Gang Hu `[一作]` (Yunnan University), Haiyan Ding `[通讯]` (Yunnan University)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了XBRLTagRec框架，实现金融文本中数值的XBRL标签自动匹配。

**💡 创新点**

结合LLM生成标签文档、语义检索与多轮零样本重排，解决标签语义相似难题。

**🔧 技术方法**

使用指令微调FLAN‑T5‑Large + LoRA生成文档，Sentence‑T5‑XXL做嵌入检索，ChatGPT‑3.5 进行零样本重排。

**📊 数据集**

在FNXL子集上进行实验，包含约79k句子、142k数值实例和2794个XBRL标签。

**📈 对比分析**

与FLAN‑FinXC、AttentionXML、FiNER等基线比较，Hits@1提升至0.862，宏F1提升4.34%。

**⚠️ 局限性**

重排耗时高、未评估不同标签嵌入模型、未尝试未调优LLM的零样本标注。

---

## 331. A Minimum-Energy Control Approach for Redundant Mobile Manipulators in Physical Human-Robot Interaction Applications

**arXiv ID:** 2603.25259 | [PDF](https://arxiv.org/pdf/2603.25259v1)

**作者:** Davide Tebaldi `[一作]` (University of Modena and Reggio Emilia), Luigi Biagiotti `[通讯]` (University of Modena and Reggio Emilia)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `9cc9baba-5356-466d-81ff-d80028d90279` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种针对移动机械臂的控制方法，在人机交互场景下通过最小化整机的动能来实现更安全、更平稳的操作，并在插孔任务上进行了实验验证。

**💡 创新点**

创新点在于将动力学一致伪逆与动能最小化目标结合，构建了一个针对移动底盘与机械臂耦合的逆微分运动学优化框架，能够自动抑制重型底盘运动，提升安全性和执行效率。

**🔧 技术方法**

采用动力学一致伪逆、虚拟质量阻尼模型、POG能量端口分析，实验平台为UR10e协作机器人 + Robotnik RB‑KAIROS+ 全向底盘，使用FTE‑AXIA80 力传感器进行数据采集。

**📊 数据集**

使用27名受试者在peg‑in‑hole任务中产生的实验数据，未使用公开数据集。

**📈 对比分析**

与传统“全运动模式”和“切换模式”进行对比，评估指标包括平均动能、平均人力、平均执行速度、最终位移及总执行时间。最小能量方案平均动能降低约66%，人力下降约22%，执行时间最短（约比切换模式快24%，比全运动模式快11%）。

**⚠️ 局限性**

局限性：实验仅覆盖插孔任务，未验证在更复杂或动态环境中的鲁棒性；算法对系统动力学模型依赖较强；实验规模有限，缺乏大样本或多机器人验证。

---

## 332. Does Explanation Correctness Matter? Linking Computational XAI Evaluation to Human Understanding

**arXiv ID:** 2603.25251 | [PDF](https://arxiv.org/pdf/2603.25251v1)

**作者:** Gregor Baer `[一作]` (Eindhoven University Of Technology), Pieter Van Gorp `[通讯]` (Eindhoven University Of Technology)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `67630363-6be0-4f51-ab05-7198250671a5` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在一项用户实验中，作者通过合成时间序列数据，精准操纵解释的正确性（100%、85%、70%、55%）并让受试者在无领域先验知识的前提下进行前向模拟，测量其对 AI 决策的理解程度。

**💡 创新点**

创新点在于：①首次在受控实验中将解释正确性连续化为四个水平；②发现正确性对人类理解呈阈值效应而非线性递减；③揭示自评指标仅在完全正确且受试者已学习决策模式时才与客观表现相关。

**🔧 技术方法**

使用技术包括：合成单变量时间序列、基于定位的正确性操纵（热图重叠度控制）、前向模拟测评、Welch t 检验与非劣效检验、Likert 量表自评及 Spearman 相关分析。

**📊 数据集**

使用的数据集为完全人工生成的时间序列（随机游走+峰/谷），无真实公开数据集。

**📈 对比分析**

比较方法：将各正确性水平与100%正确性进行两两比较，采用一尾 t 检验与阈值 Δ=0.10 的非劣效检验。结果显示：70% 与 55% 正确性显著低于 100%，但 70% 与 55% 之间无显著差异；85% 与 100% 关系不显著；自评与前向模拟准确率相关性弱，只有在 100% 正确且学习成功时才有中等相关。

**⚠️ 局限性**

局限性包括：①使用合成数据，缺乏真实任务生态效度；②仅评估定位型正确性，未涵盖其他错误模式；③任务极其简单，可能不适用于复杂决策场景；④仅测试热图式单变量时间序列解释，无法推广至多维或文本等其他格式。

---

## 333. Understanding Newcomer Persistence in Social VR: A Case Study of VRChat

**arXiv ID:** 2603.25223 | [PDF](https://arxiv.org/pdf/2603.25223v1)

**作者:** Qijia Chen `[一作]` (University of Helsinki), Giulio Jacucci `[通讯]` (University of Helsinki)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

通过对24名VRChat用户的访谈和反思性主题分析，研究了新手在社交VR中的整合过程与障碍。

**💡 创新点**

提出了新人持续参与的三阶段模型（适应、文化化、嵌入），并结合VR特有的生理、认知和社交摩擦，提供针对性的设计建议。

**🔧 技术方法**

采用定性研究方法（半结构化访谈、反思性主题分析），并在设计建议中示例了“Ghost Visualizations”“Community Passports”等交互原型。

**📊 数据集**

收集了24名VRChat用户（13名新手、11名老手）的访谈数据，包含年龄、性别、使用经验等背景信息。

**📈 对比分析**

本研究未进行实验性对比或性能评估，只通过访谈内容进行归纳与分析，缺乏量化的效果指标。

**⚠️ 局限性**

样本仅涵盖已持续使用的用户，未包含一次或两次后离开的新手，难以捕捉最早退出点；研究聚焦单一平台VRChat，泛化性有限；对骚扰等负面互动的深入探讨不足。

---

## 334. Modernising Reinforcement Learning-Based Navigation for Embodied Semantic Scene Graph Generation

**arXiv ID:** 2603.25415 | [PDF](https://arxiv.org/pdf/2603.25415v1)

**作者:** Roman Kueble `[一作]`, Joerg Haehner `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在有限动作预算下，改进基于强化学习的导航策略，提升语义场景图（SSG）的完整性与安全性。

**💡 创新点**

①将REINFORCE换成PPO以实现更稳定的优化；②引入多头分解动作空间（旋转、步长、停机），而非单一离散动作；③使用课程学习和深度辅助碰撞监督来提升学习效率与执行安全；④系统性对比多种动作空间与算法组合的效果。

**🔧 技术方法**

PPO、LSTM+CNN视觉编码、动作分解（multi-head）、奖励塑形、深度感知与碰撞预测辅助、经验迁移预训练、课程学习。

**📊 数据集**

AI2-THOR室内模拟器；训练集FloorPlan 1‑27，评估集FloorPlan 28‑30；使用模拟器元数据生成完美的语义场景图。

**📈 对比分析**

通过对比REINFORCE基线、PPO、不同动作空间尺寸（16、504原子 vs 504分解）以及课程学习与否、是否加入深度输入等，评估指标包括Node Recall（语义完整性）、Move Success Rate（安全性）、Episode/Path Length。结果显示：PPO+分解504方案获得最高Node Recall并保持较高安全率；原子504在安全率和收敛速度上表现最弱；课程学习在分解方案中提升安全率但略慢收敛。

**⚠️ 局限性**

①完整性评估基于理想感知（无检测/匹配误差）；②环境规模有限，难以体现大规模复杂性；③部分配置未在训练预算内完全收敛，可能低估最终差异；④高分辨率动作空间同时增加步长，导致难以单独归因分辨率提升的影响；⑤仅在模拟器上验证，未覆盖真实感知噪声与动态障碍。

---

## 335. Decidable By Construction: Design-Time Verification for Trustworthy AI

**arXiv ID:** 2603.25414 | [PDF](https://arxiv.org/pdf/2603.25414v1)

**作者:** Houston Haynes `[一作]` `[通讯]` (SpeakEZ Technologies), Houston Haynes (SpeakEZ Technologies)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出一种在模型训练前对AI模型进行设计时验证的框架，利用有限生成阿贝尔群的代数约束来确保模型的数值稳定性、物理一致性和内存安全。

**💡 创新点**

创新点在于将维度一致性、Clifford等级保持、逃逸分类等关键属性统一归纳为可在多维整数空间上做线性方程求解的约束，从而实现多项式时间且唯一的主类型推导；并通过三者组合（维度类型系统、程序超图、适应域模型）实现完整的设计时验证闭环。

**🔧 技术方法**

主要技术包括：Hindley–Milner类型推导扩展到ℤⁿ；利用高斯消元实现多维整数线性方程求解；程序超图（Program Hypergraph）推断Clifford代数的梯度和稀疏性；前向模式自动微分与正交累加（quire）保持数值完整性；以及基于BARE协议的Typed查询与响应，确保持续学习时的类型一致性。

**📊 数据集**

本文未针对具体数据集进行实验；重点在理论证明与框架可扩展性上，引用了通用物理公式（如F=ma）和金融风险度量示例以验证维度推导正确性。

**📈 对比分析**

与传统的后期投影（Moreau投影）、PINN损失塑形、Engram检索及ARC几何求解等现有可靠性方法相比，本文提出的设计时验证不产生运行时额外开销，边际成本可忽略不计；虽然未给出数值实验，但理论分析显示相比O(d·l·r)的后期成本，设计时验证实现O(1)的摊销成本。

**⚠️ 局限性**

局限性包括：仅覆盖可写入ℤⁿ的线性约束（维度、等级、逃逸等），无法处理需要量化量化或递归固定点等更复杂的逻辑约束；对非整数型数值精度与硬件特定实现的支持仍在开发中；此外，实际部署时仍需依赖Z3等求解器来验证图层级不变量，若求解失败需回退到更高级别的验证层。

---

## 336. Does Structured Intent Representation Generalize? A Cross-Language, Cross-Model Empirical Study of 5W3H Prompting

**arXiv ID:** 2603.25379 | [PDF](https://arxiv.org/pdf/2603.25379v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 337. HiSpatial: Taming Hierarchical 3D Spatial Understanding in Vision-Language Models

**arXiv ID:** 2603.25411 | [PDF](https://arxiv.org/pdf/2603.25411v1)

**作者:** Huizhi Liang `[一作]` (Tsinghua University), Jiaolong Yang `[通讯]` (Microsoft Research Asia)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出四层空间智能框架，并通过自动化流水线生成约20亿QA对，训练出融合度量尺度点图的RGB‑D VLM

**💡 创新点**

创新点在于：①层级任务体系（0–3级）揭示空间推理层次与相互依赖；②利用点图（而非相对深度）提升几何表征；③结合大规模多源数据自动生成多样化VQA任务；

**🔧 技术方法**

技术核心：PaliGemma‑2基础模型 + MoGe‑2 3D点图估计 + 视觉+点图融合投影 + 文本生成与监督微调；

**📊 数据集**

数据来源：KosMos‑2、Objects365、CA‑1M，经过自动化处理生成5M图像、45M物体、2B QA对，覆盖所有四层任务；

**📈 对比分析**

在公开空间基准（CV‑Bench、EmbSpatial、RoboSpatial、3DSRBench 等）以及自制 Omni3D/CA‑1M 基准上，HiSpatial‑3B 在量化任务上平均提升 12‑18%，在抽象推理任务上比 GPT‑5/Gemini‑2.5‑pro 提高约 10%，并在多项基准上夺得第一名；

**⚠️ 局限性**

局限性：对 3D 注释数据依赖度高，点图估计误差会削弱几何精度；高层次推理仍受样本稀缺和链式思维不足限制；

---

## 338. MMaDA-VLA: Large Diffusion Vision-Language-Action Model with Unified Multi-Modal Instruction and Generation

**arXiv ID:** 2603.25406 | [PDF](https://arxiv.org/pdf/2603.25406v1)

**作者:** Yang Liu `[一作]` (Westlake University), Donglin Wang `[通讯]` (Westlake University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种全新的基于离散扩散的视觉-语言-动作模型，可一次性生成目标图像与动作序列，实现端到端机器人操控。

**💡 创新点**

将语言、图像与连续动作统一映射到离散词表，使用掩码去噪训练实现并行目标图像与动作生成，并通过混合注意力和迭代去噪提升长程一致性。

**🔧 技术方法**

离散扩散生成、掩码去噪训练、混合注意力机制、迭代去噪推理、Key‑Value缓存加速、动作离散化。

**📊 数据集**

在61M步跨体态的机器人操作数据上进行大规模预训练，随后在LIBERO与CALVIN模拟基准以及真实AgileX机器人上进行微调与评测。

**📈 对比分析**

与现有VLA方法对比，实验显示在LIBERO上平均成功率98.0%，在CALVIN长程任务平均完成长度4.78，均超越同类连续或离散动作模型；在真实机器人实验中成功率超过80%。

**⚠️ 局限性**

生成的目标图像细节有限，缺乏高像素精度；模型对动作离散化的精度和推理延迟仍受限，且在极端视觉复杂场景下的准确性有待提升。

---

## 339. LaMP: Learning Vision-Language-Action Policies with 3D Scene Flow as Latent Motion Prior

**arXiv ID:** 2603.25399 | [PDF](https://arxiv.org/pdf/2603.25399v1)

**作者:** Xinkai Wang `[一作]` (Southeast University), Lixin Yang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出LaMP框架，将3D场景流作为潜在运动先验，在Vision‑Language‑Action模型中实现双专家（Motion Expert与Action Expert）协同学习；

**💡 创新点**

创新点包括：① 将稠密3D场景流作为动态先验；② 采用门控跨注意力实现动态运动信息注入，避免预训练VLM特征崩塌；③ 仅使用一步去噪的运动隐状态，减少计算开销；④ 通过条件流匹配和部分去噪的设计实现高效推理；

**🔧 技术方法**

核心技术包括：条件流匹配（Conditional Flow Matching）、3D Transformer、gated cross‑attention、部分去噪（one‑step denoising）以及流量匹配的动作预测；

**📊 数据集**

使用了1.6M观测‑语言‑运动三元组（来自LIBERO、BridgeV2、DROID、InternData-A1、TraceForge生成的稠密3D流），并在LIBERO、LIBERO‑Plus、SimplerEnv‑WidowX、BridgeV2以及真实世界Flexiv Rizon 4机器人数据上进行评估；

**📈 对比分析**

与π_0、π_0.5、OpenVLA、FlowVLA等多种VLA基线对比，LaMP在LIBERO平均成功率98.3%（最高）、LIBERO‑Plus OOD平均79.3%（比最强基线高9.7%）、SimplerEnv‑WidowX平均79.2%（比FlowVLA高5.2%），在长时序、精细几何以及跨领域鲁棒性方面均表现最优；

**⚠️ 局限性**

局限性在于Motion Expert采用固定20×20网格与T=32的时间窗口，限制了对细粒度局部运动和更长规划时序的表达；目前依赖TraceForge生成的稠密3D流监督，难以直接迁移到无标注的现实视频；未来可探索自适应分辨率和更长时间上下文的运动先验学习。

---

## 340. From Intent to Evidence: A Categorical Approach for Structural Evaluation of Deep Research Agents

**arXiv ID:** 2603.25342 | [PDF](https://arxiv.org/pdf/2603.25342v1)

**作者:** Shuoling Liu `[一作]` (Hong Kong University of Science and Technology), Qiang Yang `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文将深度研究代理（DRA）的工作流程形式化为范畴论中的结构保持映射，并基于此构建了包含296道题目的理论性基准测试，涵盖连通链、V结构、子结构重排与Yoneda探测四个维度；

**💡 创新点**

创新点在于首次为DRA提供了严谨的范畴论框架，用结构保持的函子来描述搜索与推理操作，并设计了四轴机制感知评测，能系统性检验结构保持与多跳推理的能力；

**🔧 技术方法**

使用了范畴论的概念（范畴、函子、拉回、极限/余极限）来模型化代理行为，并结合大规模语言模型与检索模块实现搜索与推理的分离；

**📊 数据集**

评测数据集为自构造的296道双语问题，覆盖四个维度的结构性挑战，并通过人类验证与LLM辅助评判相结合的方式得到答案；

**📈 对比分析**

对11种主流模型（推理、检索增强、深度研究）采用二元准确率指标进行比较，结果表明最优模型Grok Deep Research在整体得分上仅达19.9%，在V结构与Yoneda探测上表现突出，但在多跳连通链与子结构重排上依旧低迷；

**⚠️ 局限性**

局限性在于评测任务仍高度依赖任务特定的启发式策略，模型在结构保持与多跳推理方面表现不稳定，存在明显的“结构盲区”，表明现有技术尚未实现对复杂拓扑信息的普适掌握。

---

## 341. Large Language Model as Token Compressor and Decompressor

**arXiv ID:** 2603.25340 | [PDF](https://arxiv.org/pdf/2603.25340v1)

**作者:** Wenbing Li `[一作]`, Wei Yang `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过在LLM内部集成压缩与解码模块，利用自回归生成将长文本映射为可变长度离散的Z‑令牌，再从中重构原文本或执行下游任务。

**💡 创新点**

创新点在于将LLM本身转化为“内部语言”压缩器和解码器，Z‑令牌可变长度、可解释、可控制，且直接在LLM推理过程中使用。

**🔧 技术方法**

技术包括基于预训练LLM的自回归压缩器与解码器、Gumbel‑Softmax离散化、LoRA轻量化适配、滑动窗口压缩、Schedule Sampling 以降低暴露偏差、以及语义一致性评估。

**📊 数据集**

使用的公开数据集有 Wikipedia、CNN/DailyMail、HotpotQA、NarrativeQA、QuALITY、QASPER，覆盖长文本生成、摘要与多跳问答任务。

**📈 对比分析**

与ICAE、AutoCompressor、GistToken、LLOCO等基线比较，本文在 BLEU‑4、ROUGE、QA F1 指标上均取得最高或相近的表现，压缩率最高可达 18×，推理速度提升约 2–3×，显著降低显存占用。

**⚠️ 局限性**

局限性包括对滑动窗口跨段一致性的需求、极端压缩下信息可能损失、训练成本相对较高，以及对 Z‑令牌多义性仍存在一定不确定性。

---

## 342. On Representability of Multiple-Valued Functions by Linear Lambda Terms Typed with Second-order Polymorphic Type System

**arXiv ID:** 2603.25337 | [PDF](https://arxiv.org/pdf/2603.25337v1)

**作者:** Satoshi Matsuoka `[一作]` `[通讯]` (AIST), Satoshi Matsuoka (AIST)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

**🎯 论文内容**

本文提出一种通过第二阶多态线性 λ 演算实现多值函数的表示方法，提供电路式和归纳式两种构造风格，并给出若干优化策略，案例演示了 Belnap 双格上的四变量多数函数的实现与优化；

**💡 创新点**

创新点在于将多值函数映射为线性 λ 术语并利用多态型系统实现；同时将电路式与归纳式结合，提出减少复制器与约简规则的优化；

**🔧 技术方法**

采用第二阶多态线性 λ 演算、类型推导与 β₁/β₂ 约简规则、复制组合子与组合子构造；

**📊 数据集**

本文未使用传统机器学习或实验数据集，主要以理论构造与符号化例子为主；

**📈 对比分析**

未给出系统性能对比，仅提出理论上可实现多值函数的通用表示与优化；

**⚠️ 局限性**

局限性包括缺乏分支（if-then-else）支持、未提供具体实现与性能评估、以及对三种构造风格的实际效率比较仍待研究。

---

## 343. HeSS: Head Sensitivity Score for Sparsity Redistribution in VGGT

**arXiv ID:** 2603.25336 | [PDF](https://arxiv.org/pdf/2603.25336v1)

**作者:** Yongsung Kim `[一作]` (Seoul National University), Sungroh Yoon `[通讯]` (Seoul National University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种针对视觉几何基础变压器（VGGT）全局注意力层的两阶段稀疏化管线，先通过Head Sensitivity Score（HeSS）评估各注意力头对稀疏化的敏感度，再在推理阶段利用HeSS动态分配注意力预算，从而实现头级别的自适应稀疏化；

**💡 创新点**

创新点包括：①首次提出HeSS，利用相机位姿误差和点云误差的Hessian（通过Fisher信息矩阵近似）来量化头的敏感度；②基于HeSS实现头级动态稀疏化预算重分配，显著缓解高稀疏率下的性能下降；③将头级稀疏化与3D几何误差结合，提升多视图重建任务的鲁棒性；

**🔧 技术方法**

采用多头自注意力、块稀疏注意力机制、Fisher信息矩阵估计误差梯度、相机位姿误差与点云误差的定义、基于HeSS的预算重分配与水填算法，以及CUDA自定义内核实现高效稀疏化；

**📊 数据集**

在CO3Dv2 dev集上进行头级敏感度校准，使用DTU数据集进行相机位姿估计和多视图立体（MVS）性能评估；

**📈 对比分析**

与SparseVGGT以及通用稀疏化基线进行对比，使用AUC@30、Chamfer Distance等指标，结果表明在高稀疏率（如75%）下，HeSS-guided稀疏化在相机位姿和MVS任务上均显著优于对照组，且性能衰减更平缓；

**⚠️ 局限性**

限制包括：①HeSS仅在层内可比，对不同层间的敏感度比较缺乏统一标准；②仅针对推理时稀疏化，未考虑训练时的自适应或鲁棒性；③方法主要在VGGT上验证，通用性需进一步研究；

---

## 344. Macroscopic Characteristics of Mixed Traffic Flow with Deep Reinforcement Learning Based Automated and Human-Driven Vehicles

**arXiv ID:** 2603.25328 | [PDF](https://arxiv.org/pdf/2603.25328v1)

**作者:** Pankaj Kumar `[一作]` (Indian Institute of Technology Kanpur), Subrahmanya Swamy Peruru `[通讯]` (Indian Institute of Technology Kanpur)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文基于TD3深度强化学习模型，利用真实NGSIM轨迹数据训练自主车辆的跟车控制策略，并通过仿真与IDM模型对比，系统分析了混合交通中安全、效率、舒适性及燃油消耗等宏观交通特性。

**💡 创新点**

创新点在于：①首次对DRL控制车辆进行宏观流量分析，构建Fundamental Diagram（FD）；②通过调节安全时间间隙（T）和自动车辆渗透率，探讨驾驶员异质性对道路容量和燃油效率的影响；③揭示DRL车辆在100%渗透率时可提升约7.5%道路容量，并在高速段提升约29%燃油效率。

**🔧 技术方法**

技术方法包括：Twin Delayed Deep Deterministic Policy Gradient（TD3）算法；多维奖励函数（安全、效率、舒适、速度合规、燃油效率）；Edie’s方法用于从微观轨迹估计宏观密度、流量和速度；混合交通仿真与IDM对照。

**📊 数据集**

数据集：I‑80 NGSIM高速公路轨迹数据（45分钟，高频10 Hz），用于训练与评估；并使用Ornstein–Uhlenbeck过程生成的单条前车轨迹，用于构建FD覆盖不同速度区间。

**📈 对比分析**

比较方法：将训练好的DRL跟车模型与传统IDM模型在相同交通条件下的密度-流量关系、道路容量以及燃油效率进行对比。实验结果显示：在100%DRL渗透率时道路容量提升约7.52%，燃油效率在高速段提升28.98%（低速段提升1.86%），且整体安全性未受影响。

**⚠️ 局限性**

局限性：仅使用单条前车轨迹生成后续车辆，未考虑多前车或不同车型的相互作用；燃油消耗仅基于仿真模型估计，缺乏真实发动机测量；部分渗透率下的提升有限，需进一步验证多车辆类型与真实燃油数据。

---

## 345. How Pruning Reshapes Features: Sparse Autoencoder Analysis of Weight-Pruned Language Models

**arXiv ID:** 2603.25325 | [PDF](https://arxiv.org/pdf/2603.25325v1)

**作者:** Hector Borobia `[一作]` (Universitat Politecnica De Valencia), Guillermina Tormo-Carbó `[通讯]` (Universitat Politecnica De Valencia)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究系统性地探讨了无结构权重剪枝如何重塑语言模型的内部特征几何，使用稀疏自编码器（SAEs）作为可解释性探针。

**💡 创新点**

研究发现稀有SAE特征在剪枝后存活的比率远高于频繁特征，表明剪枝实际上起到了隐式特征选择的作用。此外，Wanda剪枝在特征结构保留方面的表现优于传统的幅度剪枝。

**🔧 技术方法**

使用了稀疏自编码器（SAEs）进行特征提取和比较，采用了两种剪枝方法：幅度剪枝和Wanda剪枝。

**📊 数据集**

研究涉及三种模型（Gemma 3 1B、Gemma 2 2B、Llama 3.2 1B），并在不同的稀疏水平（0-60%）下进行实验。

**📈 对比分析**

Wanda剪枝在特征保留方面表现优于幅度剪枝，尤其在30%稀疏时，Gemma 3的MNN为0.669（Wanda）对比0.521（幅度），Gemma 2的差距更大，分别为0.794和0.213。整体上，Wanda剪枝在保持模型性能的同时，保留了更多的内部特征几何结构。

**⚠️ 局限性**

本研究的局限性包括：模型规模仅限于1-2B参数，未探讨更大规模模型的表现；使用的上下文长度较短，可能影响困惑度测量；在深层次的激活提取中出现了浮点数溢出问题；种子敏感性导致特征的稳定性较低；仅研究了一次性剪枝，未考虑剪枝后的再训练效果。

---

## 346. AD-CARE: A Guideline-grounded, Modality-agnostic LLM Agent for Real-world Alzheimer's Disease Diagnosis with Multi-cohort Assessment, Fairness Analysis, and Reader Study

**arXiv ID:** 2603.25322 | [PDF](https://arxiv.org/pdf/2603.25322v1)

**作者:** Wenlong Hou `[一作]` (Hong Kong Polytechnic University), Shujun Wang `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a6cb313d-240c-4723-a372-3ba1f39b9afc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

开发并验证了一种名为AD-CARE的多模态、无模态依赖的LLM驱动诊断代理，能够在缺失数据的真实临床环境中生成符合指南的诊断报告；

**💡 创新点**

创新点在于将LLM与专用工具协同执行，实现动态工具调用与多模态证据融合，并在报告中嵌入临床指南逻辑，提升可解释性和公平性；

**🔧 技术方法**

核心技术包括大语言模型推理引擎（Planner）、专用分析工具（如脑体积、海马体积、PHS、MRI预测等）、以及结果聚合器（Coordinator）；

**📊 数据集**

使用了四个公开数据集（ADNI、NACC、OASIS、AIBL）和两个院内数据集（XWH、SYSUH），共计10,303例，涵盖多地区、多学科和多模态；

**📈 对比分析**

与多种基准方法（单模态、多模态网络、传统机器学习、LLM原生输出）比较，AD-CARE在六个队列中准确率平均84.9%，比基线提升4.2%–13.7%；在种族和年龄子组中减少误差波动，读者研究显示医生/放射员准确率提升6%–11%，决策时间缩短≈2.5倍；

**⚠️ 局限性**

局限包括仅评估AD（非阿尔茨海默痴呆）类别，院内样本规模与族裔多样性有限；读者研究规模小、缺乏真实工作流程；对非公开LLM的依赖可能限制部署；未来需扩展至其他痴呆类型、更多模态及多中心前瞻验证。

---

## 347. MACRO: Advancing Multi-Reference Image Generation with Structured Long-Context Data

**arXiv ID:** 2603.25319 | [PDF](https://arxiv.org/pdf/2603.25319v1)

**作者:** Zhekai Chen `[一作]` (HKU MMLab), Xihui Liu `[通讯]` (HKU MMLab)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个覆盖多参考图像生成四个关键维度（定制、插画、空间、时间）的 400K 样本数据集 MACRO，并构建了 4K 样本的多参考评测基准。

**💡 创新点**

创新点在于：①大规模多参考数据结构化收集，支持最高 10 张参考图；②四维任务设计提供全景覆盖；③基准采用 LLM‑as‑Judge 评估多模态一致性；④通过跨任务共训练和 token 选择策略提升长上下文生成性能。

**🔧 技术方法**

使用的技术包括：大规模数据提炼与过滤、跨任务联合训练、动态分辨率策略、token 选择（块级、图像级、文本对齐）和 LLM‑as‑Judge 评价框架。

**📊 数据集**

使用的数据集为：MACRO（400K 训练集，10 张参考图/样本）和 MAB（4K 评测集），并对开放源模型 Bagel、OmniGen2、Qwen‑Image‑Edit‑2511 进行微调。

**📈 对比分析**

实验表明，微调后模型在所有四个任务维度和不同参考图数量（1–10）上均明显优于原始开源基线，并与闭源模型 Nano Banana Pro 和 GPT‑Image‑1.5 接近；在 OmniContext 任务上也超过专门为该基准设计的数据集 Echo4o。

**⚠️ 局限性**

局限性包括：仍然对 6–10 张参考图的性能下降；token 选择和长上下文处理仍有提升空间；评估依赖 LLM 判别，可能受限于模型自身的偏差；缺乏对更大规模模型的验证。

---

## 348. FSGNet: A Frequency-Aware and Semantic Guidance Network for Infrared Small Target Detection

**arXiv ID:** 2603.25389 | [PDF](https://arxiv.org/pdf/2603.25389v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 349. UMBRELLA: Uncertainty-aware Multi-robot Reactive Coordination under Dynamic Temporal Logic Tasks

**arXiv ID:** 2603.25395 | [PDF](https://arxiv.org/pdf/2603.25395v1)

**作者:** Qisheng Zhao `[一作]` (Peking University), Zhongkui Li `[通讯]` (Peking University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种名为UMBRELLA的在线多机器人协作框架，能够在动态目标与基于LTL的协同任务下实时规划与执行；

**💡 创新点**

创新点在于将Conformal Prediction（CP）用于动态目标运动预测并融入Monte Carlo Tree Search（MCTS）进行不确定性感知的任务分配，同时采用CVaR目标和事件触发的递归滚动规划，实现了风险厌恶与高效在线更新；

**🔧 技术方法**

核心技术包括CP预测、LSTM/GRU轨迹预测器、MCTS、LTL语义约束、CVaR风险评估、递归滚动规划与部分同步执行；

**📊 数据集**

使用了2000条人工合成动物（羚羊、兔子、象、老虎）轨迹数据集，分为训练与校准集，并在仿真与实际硬件实验（4台机器人、2个目标）中验证；

**📈 对比分析**

与MILP、BnB、无预测/无不确定性基线、Ours-G、Ours-P10及clairvoyant策略对比，平均完工时间平均减少约23%、方差下降71%，在大多数场景中与clairvoyant方案仅相差6–8%；

**⚠️ 局限性**

局限性包括：需要较高的计算开销（采样与模拟），依赖精准的目标观测，CP失败概率会影响预测质量，且在机器人数目或目标数目急剧增大时可能出现实时性瓶颈；

---

## 350. Visualizing Impedance Control in Augmented Reality for Teleoperation: Design and User Evaluation

**arXiv ID:** 2603.25418 | [PDF](https://arxiv.org/pdf/2603.25418v1)

**作者:** Gijs van den Brandt `[一作]` (Eindhoven University of Technology), Elena Torta `[通讯]` (Eindhoven University of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本研究提出并验证了一种基于增强现实的阻抗控制目标可视化方法，帮助操作员在低成本VR手柄下实现对接触力的实时感知，随后在双臂搬运任务中进行用户实验评估。

**💡 创新点**

创新点在于将隐藏的阻抗控制目标姿态及其位移通过蓝色圆盘和连线实时叠加于操作员视野，提供直观的力反馈，无需昂贵触觉设备；实验表明此可视化显著提升了需精细力控制的抬升任务效率。

**🔧 技术方法**

采用Meta Quest 3 VR手柄与头显、Unity实时渲染AR叠加、Franka Emika Panda双臂阻抗控制（ROS实现）、自定义硅胶末端执行器、线性混合效应模型分析、NASA‑TLX与SUS问卷评估等技术。

**📊 数据集**

使用自身收集的实验数据：17名参与者完成32个目标（每人8个抬升+8个滑动任务），记录完成时间、任务成功率、主观负荷与可用性评分。

**📈 对比分析**

通过与无AR可视化条件的对照实验，使用线性混合效应模型比较完成时间；结果显示在抬升任务中AR可视化平均缩短24%完成时间，滑动任务无显著差异；主观负荷与可用性评分差异不显著。

**⚠️ 局限性**

局限性包括仅在Franka双臂与自制末端执行器上测试、样本量相对有限、AR遮挡可能影响视野、未测量力曲线或降落率、缺乏跨硬件/学习曲线的进一步验证。

---

## 351. LACY: Simulating Expert Mentoring for Software Onboarding with Code Tours

**arXiv ID:** 2603.25391 | [PDF](https://arxiv.org/pdf/2603.25391v1)

**作者:** Zeynep Begüm Kara `[一作]` (Bilkent University), Eray Tüzün `[通讯]` (Bilkent University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

开发并部署了 Lacy——一种基于代码导览的混合人机软件入职系统，能够捕获专家指导并生成可复用的代码旅程，支持语音录制、测验、播客等功能。

**💡 创新点**

核心创新在于：① 通过 Voice‑to‑Tour 将专家现场讲解实时转化为可重复使用的代码导览；② 采用 AI 辅助生成并让专家迭代，兼顾自动化与专家隐性知识；③ 在 IDE 内嵌入完整的入职工作流（分配、跟踪、测验、问答）并在工业场景中验证其可扩展性。

**🔧 技术方法**

技术栈包括：VS Code 扩展、Google Gemini‑2.5 Flash LLM、语音识别（STT）与语音合成（TTS）工具、代码导览结构（节点、链接、注释）、测验/问答数据库、可视化仪表盘与后台数据库。

**📊 数据集**

使用了 Beko 公司的 30K+ LOC 旧版金融系统 “Bankhet” 作为实验代码库，并通过与 2 位专家和 5 名学习者的访谈、问卷与实验收集数据；没有公开的标准数据集，而是基于真实工业项目进行评估。

**📈 对比分析**

在对比实验中，将专家引导的 AI‑辅助导览（Guided）与纯 AI 生成的探索性导览（Exploratory）进行对照：测验得分 83% vs 57%，专家评分 79% vs 76.8%；学习者完成 Guided 约 10 分钟更快，整体学习效果提升约 26 个百分点。与传统 1‑to‑1 导览相比，用户体验接近或优于。

**⚠️ 局限性**

局限性包括：① 结果依赖 LLM 的质量和专家输入，模型失误可能导致导览失真；② 随着代码演进，导览可能失效，需要持续维护；③ 受限于仅 7 名参与者和单一组织，外推性有限；④ 受访者自报数据的主观性；⑤ 需要企业内置 LLM 或本地模型以规避外部云服务的安全顾虑。

---

## 352. GlowQ: Group-Shared LOw-Rank Approximation for Quantized LLMs

**arXiv ID:** 2603.25385 | [PDF](https://arxiv.org/pdf/2603.25385v1)

**作者:** Selim An `[一作]` (DGIST), Yeseong Kim `[通讯]` (POSTECH)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 GlowQ 与 GlowQ‑S 两种低秩校正方案，用组共享右因子和缓存机制在量化 LLM 的推理中实现高效校正。

**💡 创新点**

创新点在于：① 只为共享输入的模块学习单一右因子，显著减少多层重复计算；② 通过协方差对齐的随机化 SVD 选取最有利的子空间；③ 在部署时实现只恢复重要组/层的选择性策略，进一步降低延迟与内存。

**🔧 技术方法**

核心技术包括：协方差对齐（whitened）目标、QR‑降维随机化 SVD、组共享低秩分解、缓存与选择性恢复策略。

**📊 数据集**

使用多大模型（LLaMA 3、LLaMA 2、Qwen 2.5/3、OPT、Mistral、Qwen1.5‑MoE）以及 WikiText‑2、C4 等公开文本数据集进行评测。

**📈 对比分析**

与现有 PTQ（BitsAndBytes、AWQ、GPTQ）及低秩校正（LQER、ZeroQuant‑V2、QERA、L2QER）对比，GlowQ 在保持或略低准确率的同时：TTFB 平均下降 5.6%（全恢复）/23.4%（选择性），吞吐率提升 9.6%/37.4%；内存占用比逐层校正低约 30‑60%。

**⚠️ 局限性**

局限性包括：选择标准对不同模型需要手动调参；在极低位宽或极高精度需求场景下，组共享可能无法完全恢复所有误差；对输入统计分布的依赖使得在分布漂移时需要重新校准。

---

## 353. IntentReact: Guiding Reactive Object-Centric Navigation via Topological Intent

**arXiv ID:** 2603.25382 | [PDF](https://arxiv.org/pdf/2603.25382v1)

**作者:** Yanmei Jiao `[一作]` (Hangzhou Normal University), Wen-an Zhang `[通讯]` (Zhejiang University of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种意图驱动的对象中心导航框架IntentReact，利用低维方向信号桥接全局拓扑规划与局部感知控制；

**💡 创新点**

创新点在于将2跳拓扑意图作为软方向引导，通过FiLM条件化方式将意图注入局部策略，并结合BEV可行性修正实现既全局一致又局部可执行的导航；

**🔧 技术方法**

使用的技术包括：基于对象的拓扑地图构建、Dijkstra全局规划、低维意图向量（2跳方向）生成、FiLM特征调制、BEV可行性约束；

**📊 数据集**

采用Habitat-Matterport3D（HM3Dv0.2）+ InstanceImageNav 数据集进行训练与评估；

**📈 对比分析**

与RoboHop、TANGO、ObjectReact等基线在GT-Topological与学习感知两种设定下对比，IntentReact在SPL/SSPL指标上显著提升（如SPL提升至81.48对比63.89），并在初始朝向偏差、意图估计误差等鲁棒性实验中表现更佳；

**⚠️ 局限性**

局限性包括：需要预先构建完整的拓扑地图和准确的意图估计，受感知误差影响；对动态环境或大规模场景的适应性尚未验证；以及仅在仿真环境中测试，缺乏真实机器人实验。

---

## 354. PRISM: Dynamic Primitive-Based Forecasting for Large-Scale GPU Cluster Workloads

**arXiv ID:** 2603.25378 | [PDF](https://arxiv.org/pdf/2603.25378v1)

**作者:** Xin Wu `[一作]` (Southwest Jiaotong University), Qiang Duan `[通讯]` (Pennsylvania State University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了PRISM框架，用原语词典分解与自适应频谱精炼实现GPU工作负载的可解释组合预测

**💡 创新点**

创新点在于将聚合时序拆解为可解释的原语基础并结合频域自适应过滤，避免传统单一全局模型平滑峰值，显著提升预测准确性与可解释性

**🔧 技术方法**

采用双组件编码器（Primitive Dictionary Decomposition + Adaptive Spectral Refinement）、多头注意力、频域滤波、动态门控以及多尺度补丁嵌入

**📊 数据集**

使用阿里云GPU集群真实生产轨迹，覆盖466,867个AI任务、184天、10,412张GPU的数据集

**📈 对比分析**

与Autoformer、Dlinear、TimesNet、MetaEformer等9种SOTA基线在MSE、MAE、RMSE、R²等指标对比，PRISM在所有预测时长均取得最低误差、最高R²（48h达0.9131）

**⚠️ 局限性**

局限性包括对训练时频谱窗口选择敏感、原语字典规模需经验调优、对极端极端稀疏峰值的泛化仍有待进一步验证

---

## 355. Supercharging Federated Intelligence Retrieval

**arXiv ID:** 2603.25374 | [PDF](https://arxiv.org/pdf/2603.25374v1)

**作者:** Dimitris Stripelis `[一作]` (Flower Labs), Nicholas D. Lane `[通讯]` (Flower Labs)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个安全的 Federated RAG 系统，能够在私有数据仓库之间进行分布式检索，并在可信执行环境中完成聚合与文本生成。

**💡 创新点**

创新点在于：①将检索留在各本地仓库以保持数据本地化；②在 TEE 内安全聚合与 LLM 推理；③引入混合信任级联推理，允许使用非保密第三方模型作为辅助上下文；④将 Flower CRC 与 Federated RAG 结合，实现端到端的保密远程 LLM 推理。

**🔧 技术方法**

采用技术包括 Flower 框架的 Federated RAG、可信执行环境 (TEE) 与远程证明、FAISS 索引、本地检索、SmolLM 1.7B、AWS Nova Micro、Qwen3 235B、Flower CRC。

**📊 数据集**

使用的数据集包括 PubMed、StatPearls、Textbooks、Wikipedia 文档语料库，并在 MIRAGE 基准（PubMedQA、BioASQ、MedQA）上进行评估。

**📈 对比分析**

通过比较 Standalone、Cascading、Confidential 三种推理模式，发现 Cascading 在 PubMedQA 与 MedQA 上提升约 40%–46% 的准确率；Confidential 模式下 Qwen3 在 TEE 内实现最高准确率和最低延迟，整体性能均优于单独推理。

**⚠️ 局限性**

局限性包括：CPU‑基 TEE 的生成瓶颈；对第三方模型的依赖；侧信道攻击不在保护范围内；未对更大模型或更复杂场景进行深入评估。

---

## 356. New bounds for codes over Gaussian integers based on the Mannheim distance

**arXiv ID:** 2603.25362 | [PDF](https://arxiv.org/pdf/2603.25362v1)

**作者:** Minjia Shi `[一作]` (Anhui University), Jon-Lark Kim `[通讯]` (Sogang University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文研究了在高斯整数剩余环上，以曼宁距离为度量的线性码，推导了相应的距离界限、曼宁球体积、球包络界限以及完美码条件，并给出了自共轭码的上界与示例，同时提出了曼宁距离的译码算法。

**💡 创新点**

创新点包括：①把曼宁距离引入到高斯整数码中，得到一套完整的经典界限（如汉明、格里斯马尔、球包络）在曼宁度量下的对应；②利用曼宁-麦克威尔斯身份推导自共轭码的最小距离上界，并构造达到该上界的自共轭码；③给出曼宁球体积的显式公式并证明其对完美码存在的必要条件；④展示曼宁距离能够纠正汉明距离无法纠正的误差。

**🔧 技术方法**

技术方法主要包括：代数数论（高斯整数与剩余环的结构）、组合计数（曼宁球体积与球包络计算）、编码理论中的界限推导（格里斯马尔、汉明、球包络）、麦克威尔斯身份与其曼宁版本、以及基于余数环的译码表和余数判别算法。

**📊 数据集**

主要使用的“数据集”是有限域 𝔽_p（p = a² + b² ≡ 1 (mod 4)）对应的高斯整数剩余环 𝔾_π，其中 p = 13, 17, 29 等小素数；通过对这些域构造具体的线性码、完美码与自共轭码进行实验验证。

**📈 对比分析**

比较方法：将曼宁距离下的码性能（最小距离、纠错能力）与传统汉明距离下的性能进行对比，尤其在同一码族（如[10,5]、[12,6]、[14,7]等）中展示曼宁距离可提升的误差纠正能力。实验结果表明，曼宁距离下的码能够纠正更多的四维 QAM 误差，且界限被已构造的实例所达到，证明界限的紧确性。

**⚠️ 局限性**

局限性包括：①对完美码的存在性推断仍停留在必要条件层面，缺乏完整的构造与证明；②界限主要针对高斯整数剩余环中满足 π ≡ 1 (mod 4) 的情况，对其他情况（如 π = 1 + i 或 π 为 3 (mod 4) 的整数）讨论不足；③译码表方法在码长增大时计算复杂度急剧上升，实际实现仍需进一步优化。

---

## 357. Reinforcing Prestige: Journal Citation Biases in Astronomy

**arXiv ID:** 2603.25349 | [PDF](https://arxiv.org/pdf/2603.25349v1)

**作者:** Vardan Adibekyan `[一作]` (Universidade do Porto), Artur Hakobyan `[通讯]` (Alikhanian National Science Laboratory)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文基于ADS数据库，系统性分析了2000–2025年天文学论文在主流期刊（ApJ、MNRAS、A&A）与多学科期刊（Nature、Science、Nature Astronomy）之间的引用偏差和时间演化。

**💡 创新点**

创新点在于将期刊发表比例与平均引用比例相结合，量化“引用比例”与“出版比例”之比，并对同一作者在不同期刊中的引用偏差进行分层比较，揭示了期刊声望与引用行为之间的动态关系。

**🔧 技术方法**

采用自定义Python脚本对ADS bibcode进行解析，提取期刊名称和参考文献，结合文本处理（正则、词典映射）进行作者机构与国家归一化，计算引用比率和时间窗口统计。

**📊 数据集**

使用了由ADS检索得到的255,008篇符合关键词（Exoplanet、Star、Galaxy、AGN）的同行评议论文，覆盖769个期刊，覆盖2000–2025年共计约25万篇文章。

**📈 对比分析**

通过计算每个期刊的“引用比例/出版比例”以及“在刊内引用比率/跨刊引用比率”，与整体场域平均值对比，发现多学科期刊引用比例显著高于其出版比例，而主流天文期刊则低于1；在十年内，多学科期刊的引用偏差逐渐减弱，表明引用行为趋于多元化。

**⚠️ 局限性**

局限性包括：关键词筛选可能忽略非关键词主题的天文研究；作者姓名与机构信息的不完整导致人名去重困难；引用与影响力的因果关系未能完全解析；未考虑预印本和开放获取对引用的影响；以及对期刊自引与领域特定引用习惯的细粒度分离不足。

---

## 358. Beyond Detection: Rethinking Education in the Age of AI-writing

**arXiv ID:** 2603.25329 | [PDF](https://arxiv.org/pdf/2603.25329v1)

**作者:** Maria Marina `[一作]` (AIRI), Vasily Konovalov `[通讯]` (AIRI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对生成式 AI 写作对教育影响进行系统梳理，评估现有文本检测技术并提出以教学设计替代禁令的策略，强调写作过程与 AI 文本识别素养的重要性。

**💡 创新点**

提出将 AI 文本检测和写作教学融合为核心素养的框架，倡导“更难作弊、更易学习”的课堂设计，并将人类检测者作为可训练的关键能力。

**🔧 技术方法**

综述并对比神经网络二分类、零射击检测、软水印、检索式检测等技术，并讨论其算法原理和潜在攻击手段。

**📊 数据集**

利用公开的 Wikipedia 文章、PubMed 摘要以及常用 AI 与人类文本对照集进行评估，探讨数据集对检测性能的影响。

**📈 对比分析**

通过 AUROC、准确率等指标对比显示：传统检测器在多轮改写攻击后 AUROC 降至 59.8%，而经验丰富的人工检测者在 300 篇非虚构文中仅误判 1 篇，准确率超过 99%。

**⚠️ 局限性**

检测方法易被迭代绕过，水印可能降低文本质量且缺乏统一标准，且无法覆盖新模型的生成特性；教学策略仍需实证验证，且对不同语言和技术背景的学生存在公平性挑战。

---

## 359. Adaptive Learned Image Compression with Graph Neural Networks

**arXiv ID:** 2603.25316 | [PDF](https://arxiv.org/pdf/2603.25316v1)

**作者:** Yunuo Chen `[一作]` (Shanghai Jiao Tong University), Guo Lu `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了基于图神经网络的内容自适应图像压缩框架GLIC，构建双尺度图并通过复杂度感知的自适应邻居度分配来实现灵活的感受野和像素级连通性；

**💡 创新点**

创新点在于双尺度候选采样与复杂度感知的邻居度预算，使得模型在保持近线性复杂度的同时能捕获局部与全局冗余，显著提升压缩效率；

**🔧 技术方法**

采用Graph Neural Networks（GNN）与图特征聚合块（GFA），利用Sobel梯度计算复杂度得分，结合局部窗口与稀疏全局网格构建双尺度图；

**📊 数据集**

在Kodak、Tecnick和CLIC 2020验证集上进行训练与评估；

**📈 对比分析**

与VTM‑9.1、FTIC、CCA、WeConvene等多种SOTA学习型压缩方法对比，GLIC在BD‑rate上分别降低19.29%、21.69%和18.71%，并在参数量、FLOPs、解码延迟和峰值显存方面实现显著优势；

**⚠️ 局限性**

限制主要包括：对图构造和阈值搜索仍有一定计算开销，且在极高分辨率下邻居度与候选集规模需谨慎平衡，未来可进一步优化自适应度分配与并行实现。

---

## 360. CLIP-RD: Relational Distillation for Efficient CLIP Knowledge Distillation

**arXiv ID:** 2603.25383 | [PDF](https://arxiv.org/pdf/2603.25383v1)

**作者:** Jeannie Chung `[一作]` (Ewha Womans University), Jaehyung Sim `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出CLIP-RD框架，结合VRD与XRD实现多方向关系蒸馏以压缩CLIP模型。

**💡 创新点**

首次将垂直关系蒸馏(VRD)和交叉关系蒸馏(XRD)与水平关系(HRD)联合，形成完整的多方向关系对齐机制。

**🔧 技术方法**

利用InfoNCE、KL散度、多模态对齐的交叉模态相似度分布，结合标准CLIP损失和特征、交互对比蒸馏。

**📊 数据集**

在ViT-B/16教师、ViT-T/16学生上使用CC3M/CC12M训练，评估ImageNet及其变体、MSCOCO、Flickr30K、CIFAR-10/100、EuroSAT等零样本任务。

**📈 对比分析**

与TinyCLIP、CLIP-KD及其单一关系蒸馏进行对比，ImageNet零样本分类提升0.8%p，MSCOCO/I2T R@1提升9.6%p，整体在多数据集上均优于基线。

**⚠️ 局限性**

仅在ViT轻量模型上验证，跨模型/跨规模适用性待进一步探究，且训练成本与多种对齐损失复杂度较高。

---

## 361. Hessian-informed machine learning interatomic potential towards bridging theory and experiments

**arXiv ID:** 2603.25373 | [PDF](https://arxiv.org/pdf/2603.25373v1)

**作者:** Bangchen Yin `[一作]` (ByteDance), Changsu Cao `[通讯]` (ByteDance)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种基于Hessian监督的机器学习原子势Hi-MLIP，并通过HINT训练协议实现了高精度的势能曲率预测。

**💡 创新点**

创新点在于结合低成本的Hessian预训练、加权采样、课程学习和随机投影Hessian损失，显著降低高阶导数标签需求。

**🔧 技术方法**

采用AlphaNet等等变图神经网络、HINT协议、Hutchinson估计、SSCHA与Migdal–Eliashberg等计算手段进行训练与后处理。

**📊 数据集**

使用T1x、HORM、T1x-xTB-Hess、HORM-xTB-Hess、OMat24以及氢化物声子数据集进行预训练与微调。

**📈 对比分析**

与传统E/F训练相比，Hi-MLIP在转移态搜索成功率、吉布斯自由能MAE（≈0.01 eV）以及氢化物声子与超导Tc预测上均达到或逼近化学精度，且仅需原始Hessian标签的0.0005%量。

**⚠️ 局限性**

局限性包括仍需有限高精度Hessian标签、对更高阶导数（如第三阶）未覆盖、以及在极大系统或多相材料中可能需要进一步的多源数据融合与主动学习。

---

## 362. Shape and Substance: Dual-Layer Side-Channel Attacks on Local Vision-Language Models

**arXiv ID:** 2603.25403 | [PDF](https://arxiv.org/pdf/2603.25403v1)

**作者:** Eyal Hadad `[一作]` (Ben Gurion University Negev), Mordechai Guri `[通讯]` (Ben Gurion University Negev)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在本论文中，作者研究了动态高分辨率预处理对本地视觉语言模型的算法侧信道泄露，并提出双层攻击框架。

**💡 创新点**

创新点在于揭示 AnyRes 预处理产生的几何和语义两层侧信道，并证明其对隐私敏感上下文的可推断性。

**🔧 技术方法**

利用硬件性能计数器（LLC miss）和执行时间测量，以及决策树分类器。

**📊 数据集**

使用合成图像数据集，包括不同纵横比和视觉复杂度的文档、X光、加密噪声和技术图纸。

**📈 对比分析**

通过对 LLaVA-NeXT、Qwen2-VL 等模型在 Intel/AMD 两台机器上测量，结合分类准确率 84% 和对敏感类别 93%+ 的召回率，展示双层侧信道可可靠推断。

**⚠️ 局限性**

局限性包括：实验受限于实验室环境，未考虑多线程/NUMA、功耗、真实工作负载噪声；语义侧信道依赖于 LLC 容量，硬件差异可能降低效果；未探讨更深层攻击或更强防御。

---

## 363. PMT: Plain Mask Transformer for Image and Video Segmentation with Frozen Vision Encoders

**arXiv ID:** 2603.25398 | [PDF](https://arxiv.org/pdf/2603.25398v1)

**作者:** Niccolò Cavagnero `[一作]` (Eindhoven University of Technology), Daan de Geus `[通讯]` (Eindhoven University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种在冻结的视觉基础模型（VFM）特征上运行的轻量级Transformer分割解码器（PMT），实现高效的图像和视频分割，同时保持编码器可共享。

**💡 创新点**

创新点在于解决EoMT和VidEoMT需要对编码器微调的问题：通过在冻结的VFM上添加可训练的轻量级解码器、横向连接和RoPE，重现编码器最后两层的联合查询-补丁注意力，实现既快又可共享。

**🔧 技术方法**

采用Vision Transformer（ViT）作为编码器，使用Transformer解码器层、学习的查询、掩码注意力、RoPE、横向连接等技术。

**📊 数据集**

使用COCO（panoptic、instance）和ADE20K（语义）图像数据集，以及YouTube‑VIS 2019/2021、VIPSeg、VSPW视频数据集进行评测。

**📈 对比分析**

与ViT‑Adapter+Mask2Former、EoMT、CAVIS、VidEoMT等方法对比，PMT在图像分割上匹配冻结编码器的state‑of‑the‑art且速度提升约3倍；在视频分割上性能与全微调方法相当，速度提升达8倍，甚至在VSPW上刷新mIoU记录。

**⚠️ 局限性**

局限性：需要规模大、预训练充分的VFM；对小型模型效果差距明显；主要针对分割任务，未验证对其他下游任务的通用性；解码器仍需训练，对非常长视频序列的时序建模可能受限。

---

## 364. Multimodal Dataset Distillation via Phased Teacher Models

**arXiv ID:** 2603.25388 | [PDF](https://arxiv.org/pdf/2603.25388v1)

**作者:** Shengbin Guo `[一作]` (Harbin Institute of Technology), Zhuotao Tian `[通讯]` (Harbin Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于分阶段教师模型与短路轨迹的多模态数据集蒸馏框架PTM-ST，解决传统方法在教师训练后期知识迁移不佳的问题。

**💡 创新点**

核心创新在于：① 分阶段教师建模（PTM）按训练阶段使用不同教师模型；② 短路轨迹（ST）通过在关键终点之间插值构建平滑、可控的教师参数轨迹，从而稳定梯度并提升知识传递效率。

**🔧 技术方法**

技术实现基于Match‑Training‑Trajectory（MTT）框架，结合指数移动平均EMA、梯度匹配损失、以及自定义的轨迹插值函数，使用NFNet图像编码器与BERT文本编码器训练。

**📊 数据集**

在Flickr30K、MS‑COCO以及更大规模的LLaVA‑cc3m数据集上进行实验，评估图像‑文本检索（IR/ TR）性能，并在VQA、零样本分类等下游任务中验证通用性。

**📈 对比分析**

与随机、Herd、K‑center、Forgetting、MTT‑VL、LoRS、EDGM等基线对比，PTM‑ST在Flickr30K上实现最高53.6% IR@10（相比LoRS提升约15%），在COCO上提升至64.9% IR@10；在LLaVA‑cc3m上也保持领先，且在VQA和ImageNet零样本分类上显著超过LoRS，证明其在多模态蒸馏中的显著性能提升。

**⚠️ 局限性**

主要局限在于需手动设定每个阶段的插值终点与匹配范围，参数调优繁琐；此外目前仍基于MTT框架，未探索更通用的蒸馏方法。

---

## 365. The Complexity of Distributed Minimum Weight Cycle Approximation

**arXiv ID:** 2603.25368 | [PDF](https://arxiv.org/pdf/2603.25368v1)

**作者:** Yi-Jun Chang `[一作]` (National University of Singapore), Mingyang Yang `[通讯]` (National University of Singapore)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

在 CONGEST 模型下研究无向加权图的最小权重环（MWC）问题，提出一种随机化 (k+1)-近似算法，并给出其轮数复杂度与逼近比例之间的平滑权衡；同时在 Erdős 圆度猜想的假设下，证明任何 (k+1-ε)-近似算法至少需要 Ω~(n^{k+1/2k+1}) 轮数，从而实现了上界与下界在图直径 D = O(n^{1/4})（k≥2）下的几乎匹配。

**💡 创新点**

1) 将最小权重环问题转化为通过低直径分解（LDD）与随机起始时间相结合的聚类策略；2) 利用尺度化（scaling）将加权图转化为无权图，以便使用 BFS‑based SSSP；3) 对于下界，构造基于 Erdős 圆度猜想的高圆度稀疏图并应用 moving‑cut 框架，将通信瓶颈刻画为膨胀与拥塞的两项；4) 通过参数 k 调节逼近比与轮数的折中，形成连续的权衡曲线。

**🔧 技术方法**

主要技术手段包括：
- 随机指数分布启动时间 + 低直径分解（Miller‑Peng‑Xu 变体）；
- 加权图尺度化与无权图 BFS 近似 SSSP；
- 关键的短跳/长跳两种情形分别处理；
- Skeleton 节点随机采样与局部聚类；
- moving‑cut 证明框架与 Erdős 圆度猜想的结合；
- 路径逼近、边权重扰动保证唯一最短路；
- 通过重复实验与随机延迟实现高成功概率。

**📊 数据集**

本工作为理论性研究，不依赖具体实验数据集，所有证明与实验均在抽象图（高圆度稀疏图、二部图、树状结构等）上进行。

**📈 对比分析**

与 Manoharan & Ramachandran (PODC 2024) 的 (2+ε)-近似 O(n^{2/3}+D) 方案相比，本论文实现了任意 k≥1 的 (k+1)-近似，轮数可达到 Õ(n^{k+1/2k+1}+n^{1/k}+D n^{1/2(2k+1)}+D^{2/5} n^{2/5+1/2(2k+1)})，在 D=O(n^{1/4}) 时简化为 Õ(n^{k+1/2k+1})；下界证明 Ω~(n^{k+1/2k+1}) 与上界匹配（多项式对数因子内），而之前仅有 Ω~(√n/log n) 的一般下界。实验上，本论文展示了更细粒度的逼近‑轮数折中曲线，在常数逼近区间内逼近至 2+ε，且对于更大的逼近比（k≥2）实现了更低的轮数。

**⚠️ 局限性**

限制与未解决问题：
- 下界依赖 Erdős 圆度猜想；若猜想不成立，则需要新的构造；
- 对于 k=1 的情况，上界与下界并不完全匹配，仍有进一步降低 Ω~(n^{3/4}) 的空间；
- 仅在直径 D=O(n^{1/4}) 时与下界匹配；当 D 较大时上界仍显得保守；
- 算法为随机化，需多次重复实验以提高成功概率；
- 对于有向图与加权图的通用下界尚未覆盖所有可能的图结构；
- 对于极小圆度（k=1）下的常数近似（如 2‑近似）尚未提供更优的轮数上界。

---

## 366. Bayesian Learning-Enhanced Navigation with Deep Smoothing for Inertial-Aided Navigation

**arXiv ID:** 2603.25364 | [PDF](https://arxiv.org/pdf/2603.25364v1)

**作者:** Nadav Cohen `[一作]` (University of Haifa), Itzik Klein `[通讯]` (University of Haifa)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种基于贝叶斯学习的深度平滑框架BLENDS，在传统两滤波器平滑器上加入Transformer网络，动态调整协方差并给出偏置校正，实现对惯导-GNSS融合中的系统位置偏差进行补偿；

**💡 创新点**

核心创新在于将数据驱动的协方差修改与加性校正嵌入到贝叶斯平滑框架中，设计了贝叶斯一致损失函数保证均值和协方差的最小方差最优性，同时通过梯度裁剪和物理约束保证训练稳定；

**🔧 技术方法**

采用Transformer编码器提取全轨迹信息，输出协方差修改矩阵和校正向量，结合经典两滤波器融合，训练时使用自监督的BCL（位置、速度、姿态、协方差）损失；

**📊 数据集**

使用两套真实数据集：一是Haifa移动机器人（Arazim IMU+GNSS）55分钟数据，另一是INSANE四旋翼机（Pixhawk4 IMU+GNSS）约23.7分钟数据；

**📈 对比分析**

与前向EKF、两滤波器平滑器（TFS）和Rauch–Tung–Striebel平滑器（RTSS）进行对比；BLENDS在未见测试轨迹上水平位置误差相比EKF平均提升至63%，TFS/RTSS几乎无提升，且保持了协方差一致性；

**⚠️ 局限性**

需要人工调节校正边界和滑窗长度，受限于状态可观测性（如航向无法观测），且仍需事先收集训练数据，且目前仅为离线后处理方案。

---

## 367. InstanceAnimator: Multi-Instance Sketch Video Colorization

**arXiv ID:** 2603.25357 | [PDF](https://arxiv.org/pdf/2603.25357v1)

**作者:** Yinhan Zhang `[一作]` (Hong Kong University Of Science And Technology), Zeyu Wang `[通讯]` (Hong Kong University Of Science And Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 InstanceAnimator，一种基于 Diffusion Transformer 的多实例草图视频着色框架，解决传统方法在单帧依赖、实例一致性与细节保真方面的缺陷。

**💡 创新点**

创新点包括 Canvas Guidance Condition 让用户可自由放置参考元素并消除第一帧依赖；Instance-Aware Attention 建立线稿与实例的精确对应；Adaptive Decoupled Control Module 通过可学习的条件权重将语义特征细粒度注入扩散过程，提升细节保真度。

**🔧 技术方法**

采用 Diffusion Transformer、VAE 编码、CLIP+T5 语义特征投影、实例感知注意力、可学习的条件权重和跨模态投射等技术，构建多实例兼容且可控的扩散网络。

**📊 数据集**

使用自建的 OpenAnimate 数据集（约 42K 只高质量动画视频，包含参考帧、实例、背景与文本描述）以及 Sakuga42M、动画电影数据进行训练。

**📈 对比分析**

在 100 条多实例动画测试集上与 LVCD、ToonCrafter、Anidoc、LayerAnimate、ToonComposer 等基线对比，InstanceAnimator 在 FID、SSIM、LPIPS、Temporal 与 CLIP 评估中均取得最优或接近最优表现；用户研究表明其质量、连贯性与基线相当，但在易用性和可控性方面更具优势。

**⚠️ 局限性**

仍存在对极大实例数或极端姿态、背景变化下的细节保持不足；对极短或极长序列的泛化尚未充分验证；模型参数规模较大，训练成本高；部分细粒度控制在复杂场景中可能受限。

---

## 368. SafeGuard ASF: SR Agentic Humanoid Robot System for Autonomous Industrial Safety

**arXiv ID:** 2603.25353 | [PDF](https://arxiv.org/pdf/2603.25353v1)

**作者:** Thanh Nguyen Canh `[一作]` (Strike Robotics), Ben Wei Lim `[通讯]` (Strike Robotics)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e0540dec-d77f-42db-94ae-d039248f6393` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了基于Unitree G1人形机器人的自主工业安全系统SafeGuard ASF，集成多模态感知、ReAct推理框架和强化学习行走策略，实现火灾、温度异常与入侵检测与响应。

**💡 创新点**

首次将人形机器人与ToolOrchestra框架相结合，利用ReAct推理实现结构化决策，并通过多模态感知实现三类危害的实时检测与自动干预。

**🔧 技术方法**

使用YOLOv8进行火灾与人员检测、热图像阈值异常分析、深度定位、OSNet身份识别、PPO强化学习行走与追踪策略、D* Lite与MPPI规划，以及Jetson Orin边缘计算。

**📊 数据集**

在D‑Fire火灾数据集上微调YOLOv8，结合自建工业温度基线图和人员Re‑ID数据库，利用RGB‑D、热像与IMU等多模态传感器数据。

**📈 对比分析**

与基于阈值的规则系统对比，SafeGuard ASF在火灾检测mAP 94.2%/127ms、热异常检测F1 91.5%/83ms、入侵检测F1 95.9%/152ms，平均响应时间12.4s，成功率89.3%，明显优于基线。

**⚠️ 局限性**

局限在单机器人部署、缺乏物理操作与多机协同、处理器负载有限、对遮挡和极端照明下的检测性能下降。

---

## 369. Beyond Content Safety: Real-Time Monitoring for Reasoning Vulnerabilities in Large Language Models

**arXiv ID:** 2603.25412 | [PDF](https://arxiv.org/pdf/2603.25412v1)

**作者:** Xunguang Wang `[一作]` (Hong Kong University of Science and Technology), Shuai Wang `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了“推理安全”概念，并设计了九类推理不安全行为的分类法，构建了安全监测框架，实时检测并定位大型语言模型的推理链中的逻辑错误与攻击迹象。

**💡 创新点**

创新点在于：①从安全角度补充内容安全，明确推理过程本身的安全性；②提出完整的九类错误分类体系；③通过在监测提示中嵌入分类法，让现成的LLM在不训练的情况下实现对推理链的实时分类和定位；④在大量自然与对抗样本上验证了分类覆盖和监测效果。

**🔧 技术方法**

技术包括：基于结构化JSON的提示工程（含角色定义、错误分类嵌入、校准规则），多模型并行监测（Gemini-3-Flash、gpt-oss-20b、GPT‑4o、Qwen3.5‑35B-A3B），以及对推理步骤的实时流式评估与中断机制。

**📊 数据集**

使用了4,111条推理链数据集，其中包括OmniMath（自然错误）和四种对抗攻击（BadChain、Preemptive Answer Attack、OverThink、Deadlock），并从中抽取450条链做静态基准测试。

**📈 对比分析**

与SelfCheckGPT（事实一致性检测）和Qwen2.5‑Math‑PRM‑7B（步骤质量评分）对比；LLM监测在定位准确率上达84.88%，错误类型分类准确率达85.37%，显著优于基线（44.36%/68.83%）。

**⚠️ 局限性**

局限性：对干净推理分布下的误报率和阈值调优缺乏系统评估；对适应性攻击的鲁棒性尚未验证；仅针对文本推理，未涵盖多模态或工具调用的推理链。

---

## 370. System Design for Maintaining Internal State Consistency in Long-Horizon Robotic Tabletop Games

**arXiv ID:** 2603.25405 | [PDF](https://arxiv.org/pdf/2603.25405v1)

**作者:** Guangyu Zhao `[一作]` (Peking University), Yitao Liang `[通讯]` (Peking University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了一个机器人麻将桌面系统，能够通过感知、决策、操控和恢复闭环来完整执行长时序的多人游戏。

**💡 创新点**

将高层语义推理与时间关键感知/控制分离，引入触觉驱动的错误恢复和交互层监测，系统化研究了长时序桌面游戏中错误传播与状态一致性维护。

**🔧 技术方法**

使用视觉语言模型进行策略与规则解释，YOLO/SAM/FoundationPose 进行快速检测与姿态估计，触觉传感器用于抓取验证，结合模块化状态转移与日志记录。

**📊 数据集**

主要采用现场收集的真实麻将游戏数据（棋盘、手牌、历史记录）、公开棋子图像数据用于 YOLO 训练，以及人工标注的违规行为数据用于交互监测。

**📈 对比分析**

在122场完整游戏中，机器人赢率 27.9%（34/122）对比 GPT‑5.2 的 3/40；抓取成功率从 99.2% 提升到 99.8%；完整游戏完成率 89.3%；交互监测精度约 87%。

**⚠️ 局限性**

系统依赖预先校准的工作空间，监测的违规事件有限；无法提供形式化的状态一致性保证；硬件可靠性成为主要瓶颈；对动态环境的适应性不足；未评估教学与自适应辅助功能。

---

## 371. ALPS: Automated Least-Privilege Enforcement for Securing Serverless Functions

**arXiv ID:** 2603.25393 | [PDF](https://arxiv.org/pdf/2603.25393v1)

**作者:** Changhee Shin `[一作]` (Incheon National University), Seungsoo Lee `[通讯]` (Incheon National University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过静态分析和LLM自动提取服务器无服务器函数最小权限，并在运行时实时验证。

**💡 创新点**

创新点在于整合专门的静态分析、LLM代码插桩与跨云供应商的策略生成，实现全自动、实时的最小权限执行。

**🔧 技术方法**

采用AST静态分析、可定制查询、fine-tuned LLM插桩以及实时权限钩子等技术。

**📊 数据集**

使用Wonderless基准集（约8300个函数）及自建的权限策略样本。

**📈 对比分析**

与基线对比，最小权限提取覆盖率94.8%，LLM生成的验证逻辑BLEU提升220%，执行时延平均低于10%，成本增长不足8%。

**⚠️ 局限性**

局限在于无法完全处理动态资源识别和需要通配符的服务调用，导致部分函数只能生成服务级权限。

---

## 372. 4OPS: Structural Difficulty Modeling in Integer Arithmetic Puzzles

**arXiv ID:** 2603.25356 | [PDF](https://arxiv.org/pdf/2603.25356v1)

**作者:** Yunus E. Zeytuncu `[一作]` `[通讯]` (University of Michigan-Dearborn), Yunus E. Zeytuncu (University of Michigan-Dearborn)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究算术谜题难度，提出基于最小输入使用的结构性定义并构建3.4M实例数据集

**💡 创新点**

将最小输入使用量作为完整难度判定的最小充分统计量，揭示结构性难度的可解释性

**🔧 技术方法**

使用精确符号求解器（动态规划）枚举解空间并提取最小构造，配合特征工程与机器学习模型

**📊 数据集**

基于六数集合（5个1-9及25/50/75）与100-999目标的手工生成数据集，3.4M个实例

**📈 对比分析**

与仅使用表面统计的基线模型对比，加入求解器结构特征后分类准确率从约73%提升到100%，证明方法有效

**⚠️ 局限性**

仅基于符号求解的难度定义，缺乏对人类感知难度的验证，适用范围限于整数算术谜题

---

## 373. Adaptive Chunking: Optimizing Chunking-Method Selection for RAG

**arXiv ID:** 2603.25333 | [PDF](https://arxiv.org/pdf/2603.25333v1)

**作者:** Paulo Roberto de Moura Júnior `[一作]`, Annabelle Blangero `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 Adaptive Chunking 框架，通过文档感知的分块策略提升 Retrieval‑Augmented Generation（RAG）的检索与生成效果。

**💡 创新点**

创新点在于：①提出五项内在分块质量指标（References Completeness、Intrachunk Cohesion、Document Contextual Coherence、Block Integrity、Size Compliance）；②设计了两种新分块方法（LLM‑guided regex splitter 与 split‑then‑merge recursive splitter）并结合后处理；③引入基于指标的文档级分块选择策略。

**🔧 技术方法**

使用技术包括：Markdown 解析与结构化、LLM (GPT‑5) 生成正则、递归分块、BERT/句子 Transformer 生成句子与块嵌入、核心指代解析（Maverick）、BM25 与稠密检索、reranker snowflake‑arcticembed‑l‑v2.0、GPT‑4.1 生成答案。

**📊 数据集**

实验数据集为 33 份 PDF（法律、技术、社会科学领域），共约 1.18M tokens。

**📈 对比分析**

与基线（LangChain 递归、页分块、句子分块、语义分块）对比，Adaptive Chunking 在 Retrieval Completeness（67.68% vs 58.08%）和 Answer Correctness（78.01% vs 70.11%）上分别提升约 9.6pp 与 7.9pp，答复率提升 32.7%。

**⚠️ 局限性**

局限性包括：评估成本高（特别是 DCC 与核心指代抽取）、仅支持英文核心指代、超参数（长度阈值、窗口大小）经验性强、需要完整 Markdown 解析、未在查询时动态适配、需在索引阶段多次运行分块器。

---

## 374. Evaluating Language Models for Harmful Manipulation

**arXiv ID:** 2603.25326 | [PDF](https://arxiv.org/pdf/2603.25326v1)

**作者:** Canfer Akbulut `[一作]`, Laura Weidinger `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

设计并实施了跨三领域（公共政策、金融、健康）及三国（英国、美国、印度）的人工实验，评估大型语言模型在真实人机交互中对人类信念与行为的有害操纵影响。

**💡 创新点**

创新在于：①将有害操纵拆分为过程伤害（操纵线索频率）与结果伤害（信念/行为改变）两大维度；②提出多维度指标（propensity、efficacy）并在多域、多地理样本中同时评估；③首次系统比较模型与静态信息卡片的效果，揭示跨域和跨文化差异。

**🔧 技术方法**

使用 Gemini 3 Pro 大语言模型、Deliberate Lab 对话实验平台、LLM‑as‑judge 评估操纵线索、统计方法（odds ratio、chi‑square、Pearson 相关）进行数据分析。

**📊 数据集**

采用 10,101 名受试者的实验数据（涵盖三国、三领域）以及人工合成的公共政策对话样本作为评估数据集。

**📈 对比分析**

通过与静态翻转卡片对照，计算 odds ratio 评估信念加强、信念翻转、原则与金钱承诺等四个核心指标。结果显示：金融领域显著高于公共政策和健康领域；公共政策与健康领域效果相对较弱；不同国家之间存在显著差异，印度样本普遍更易产生承诺行为。

**⚠️ 局限性**

局限性包括：①实验环境与真实世界脱节，缺乏真实伤害；②仅限文本交互，未覆盖音频/视频等模态；③仅评估单个个体的二人对话，未探讨群体或社会层面的操纵；④未考察模型作为工具生成操纵内容的风险；⑤实验集中于 Gemini 3 Pro，未对其他模型做广泛验证。

---

## 375. Joint Learning Global-Local Speaker Classification to Enhance End-to-End Speaker Diarization and Recognition

**arXiv ID:** 2603.25377 | [PDF](https://arxiv.org/pdf/2603.25377v1)

**作者:** Yuhang Dai `[一作]` (Northwestern Polytechnical University), Xinsheng Wang `[通讯]` (Soul AI Lab)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出GLSC-SDR框架，将全局-局部说话人分类与端到端多说话人对话转录任务联合训练

**💡 创新点**

通过层次化说话人分类（宏观聚类+微观识别）强化说话人辨识，未改动LLM结构，联合优化提升转录与归属精度

**🔧 技术方法**

使用大型音频语言模型Qwen2.5-Omni+LoRA微调，HDBSCAN聚类，ESR2Net说话人嵌入，Serialized Output Training等技术

**📊 数据集**

在AliMeeting、AISHELL-4和AMI-SDM三大多说话人会议数据集上进行实验

**📈 对比分析**

与TagSpeech、ViebVoice-ASR、Gemini等SOTA端到端系统对比，cpWER降低约18%，SCA提升，整体性能达或超过SOTA

**⚠️ 局限性**

对聚类超参数敏感，聚类误差会影响性能；需要高质量单说话人片段；未在更大规模或跨语言真实对话数据上验证

---

## 376. Integrating Deep RL and Bayesian Inference for ObjectNav in Mobile Robotics

**arXiv ID:** 2603.25366 | [PDF](https://arxiv.org/pdf/2603.25366v1)

**作者:** João Castelo-Branco `[一作]` (Instituto Superior Técnico), Alexandre Bernardino `[通讯]` (Institute For Systems and Robotics)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e0540dec-d77f-42db-94ae-d039248f6393` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一个结合贝叶斯推理与深度强化学习的混合框架，用于移动机器人在室内部分可观测环境中进行目标搜索。

**💡 创新点**

创新点在于将贝叶斯不确定性建模与可学习的DQN决策器融合，使系统既能解释不确定性又能自适应地选择导航目标。

**🔧 技术方法**

使用了Dirichlet信念地图、温度校准的YOLO检测、空间聚类抽象、深度Q网络以及Habitat 3.0仿真平台。

**📊 数据集**

采用MS COCO进行检测温度校准，并在Habitat 3.0提供的办公室与公寓两大室内场景中进行实验。

**📈 对比分析**

与随机漫步、聚类顺序搜索、纯贝叶斯效用最大化等基线比较，BBDPS在两环境中成功率最高（1.00/0.99），并在大环境中搜索效率提升约23%和18%。

**⚠️ 局限性**

局限性包括依赖已知占用网格、无法联合地图构建与语义发现、需要为每个环境单独训练、对检测质量敏感以及跨场景泛化受限。

---

## 377. Multi-target Coverage-based Greybox Fuzzing

**arXiv ID:** 2603.25354 | [PDF](https://arxiv.org/pdf/2603.25354v1)

**作者:** Masami Ichikawa `[一作]` `[通讯]`, Masami Ichikawa

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究并实现了一种针对操作系统与固件协同运行环境的多目标覆盖导向灰盒模糊器MTCFuzz，能够在单一虚拟机中统一跟踪并利用多软件组件的代码覆盖信息进行模糊。

**💡 创新点**

创新点在于：①通过在QEMU中嵌入基本块级覆盖追踪，跨软件边界实现统一覆盖度量；②设计了地址过滤与哈希统计机制，避免无关代码干扰；③使用快照恢复重置系统状态，保证每次模糊输入的独立性。

**🔧 技术方法**

使用技术包括：QEMU QMP扩展（启动/停止追踪指令）、TCG基本块记录、覆盖日志过滤与哈希、快照保存/加载、ssh/guest文件传输、addr2line源码定位、以及Python/Go/ Rust混合实现的模糊框架。

**📊 数据集**

实验数据集：RISC‑V Linux内核 6.16‑rc1 + OpenSBI v1.6、AArch64 Linux内核 6.14 + OP‑TEE 4.7.0‑rc1，使用内核 syscall/TEE API、OpenSBI Base Extension 代码，以及内核 OOPS 日志验证漏洞重现。

**📈 对比分析**

比较方法：在相同硬件/软件环境下对比 MTCFuzz 的多目标模式与仅使用单目标（仅内核）模式，评估覆盖率、发现漏洞、执行速率。结果显示多目标模式在相同时间内覆盖率提升约20%，发现 CVE‑2025‑40031；QEMU 代码覆盖测量导致的额外开销约 45‑65%，快照恢复导致的额外开销约 12%。

**⚠️ 局限性**

局限性：①QEMU 基于模拟的覆盖记录导致较高运行时开销；②当前实现仅支持单个 QEMU 实例，无法并行多实例或多平台；③需要手工编写测试 harness，工作量大；④对硬件平台（CPU、内核版本）有一定依赖。

---

## 378. Image Rotation Angle Estimation: Comparing Circular-Aware Methods

**arXiv ID:** 2603.25351 | [PDF](https://arxiv.org/pdf/2603.25351v1)

**作者:** Maximilian Woehrer `[一作]` `[通讯]` (University of Vienna), Maximilian Woehrer (University of Vienna)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对全球图像旋转角度估计问题进行系统研究，比较了五种处理圆周拓扑的旋转估计方法，并在十六种现代视觉模型上进行大规模实验；

**💡 创新点**

创新点在于：①首次对直接回归、角度分箱分类、单元向量回归、相位移编码与圆形高斯分布等五种圆形感知方法进行全面横向比较；②揭示了分类与概率方法（尤其是圆形高斯分布）在不同backbone上的鲁棒性与稳定性；

**🔧 技术方法**

采用迁移学习框架，基于ImageNet预训练模型改造输出头，使用圆形MAE、角度分箱交叉熵、单位向量L1、相位移编码MAE以及KL散度的圆形高斯分布；

**📊 数据集**

主要使用DRC-D数据集（1474训练/535测试）做基准实验，并在COCO 2014（约83K图像）与COCO 2017（约117K图像）上验证最优配置；

**📈 对比分析**

实验表明分类（CLS）在EfficientViT‑B3上获得MAE 1.23°，圆形高斯分布（CGD）在MambaOut‑Base上获得MAE 1.24°；在COCO 2014上CGD达到3.71° MAE，COCO 2017进一步降至2.84°，显著优于以往方法；

**⚠️ 局限性**

局限性包括：实验依赖于小规模DRC-D数据，训练随机性导致结果波动；分类方法在部分大模型上训练不稳定；未在更大数据集上完成完整的16×5格网格评估。

---

## 379. Agentic Trust Coordination for Federated Learning through Adaptive Thresholding and Autonomous Decision Making in Sustainable and Resilient Industrial Networks

**arXiv ID:** 2603.25334 | [PDF](https://arxiv.org/pdf/2603.25334v1)

**作者:** Paul Shepherd `[一作]` (London South Bank University), Jonathan Rodriguez `[通讯]` (Instituto de Telecomunicações)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c84dae5d-5273-4348-85a7-b44cb586b4df` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出了 Agentic Trust Control Loop（ATCL），一种在联邦学习（FL）中由服务器端监督的轻量级信任协调机制，能够实时观察、推理并针对信任状态做出上下文感知的干预，从而维持训练过程的稳定与可靠。

**💡 创新点**

创新点在于把信任适配视为控制问题，而非简单的阈值或参数调优；ATCL通过显式的观察–推理–决策–行动循环，联合多维度信任与系统指标（梯度相似度、波动性、参与一致性、全局损失趋势等）进行状态推断，实现更精准、上下文感知的信任干预，避免了传统方法对瞬时噪声的过度反应。

**🔧 技术方法**

技术要点包括：
1. 信任分数计算与指数加权移动平均（EMA）平滑；
2. 通过TOPSIS等多准则决策对客户端进行排序；
3. 服务器端状态推理模块（规则或轻量级代理逻辑）；
4. 动态调整阈值、EMA 参数、临时排除与恢复机制；
5. 与现有 FL 框架（如 Flower、FedAvg 等）无缝集成。

**📊 数据集**

论文未公开使用具体数据集，主要在概念验证层面讨论设计与预期效果；后续工作计划在真实工业/边缘环境中部署评估。

**📈 对比分析**

对比方法：与之前的 ATSSSF 规则驱动适配方案进行对比。预期在攻击或客户端行为波动时，ATCL 能更早检测不稳定并通过上下文调节减少误排除，提高全局模型的收敛稳定性与容错率；然而目前尚无实验数据，性能评估仍待实现。

**⚠️ 局限性**

局限性包括：
1) 目前仅为设计与预期分析，缺乏实测数据验证；
2) 代理推理采用简单规则，可能难以覆盖极端动态或高度对抗场景；
3) 只在服务器端执行，未考虑客户端侧协同或安全性问题；
4) 需要进一步研究与多种 FL 框架、模型与任务的兼容性。

---

## 380. On the Vulnerability of Deep Automatic Modulation Classifiers to Explainable Backdoor Threats

**arXiv ID:** 2603.25310 | [PDF](https://arxiv.org/pdf/2603.25310v1)

**作者:** Younes Salmi `[一作]` (Poznan University of Technology), Hanna Bogucka `[通讯]` (Poznan University of Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文研究了深度学习自动调制分类（AMC）系统对XAI驱动的后门攻击的脆弱性，提出并实现了在信号物理层前插入触发器的全新后门机制，并在DNN、RNN、CNN三类模型上进行评估。

**💡 创新点**

创新点在于：①首次将可解释人工智能（XAI）用于定位调制信号中最易被操纵的时域窗口；②采用“Prototype‑PCA Hybrid”方法生成触发器值；③实现了可转移的物理后门攻击，可在不同模型间保持高成功率。

**🔧 技术方法**

采用的技术包括：自定义SamplingSHAP结合相位归一化的特征归因；触发器位置选择与触发器值生成的Prototype‑PCA Hybrid；后门注入的训练数据伪造；DNN/RNN/CNN模型训练与评估；ASR/ALC/ABC等性能指标；以及对Neural Cleanse、STRIP、Activation Clustering等防御的对抗实验。

**📊 数据集**

使用的数据集为100,000个仿真OFDM信号，包含11种调制方式，子载波数N=512、循环前缀N_cp=128，采用三跳Rayleigh衰落信道，TTI长度14符号，模拟不同SNR场景。

**📈 对比分析**

与两种白盒数字域后门攻击（Ref1、Ref2）对比，本文攻击在低SNR（-16至8 dB）下ASR显著更高（最高可达约80%），在高SNR下也保持80%以上；同时保持后门模型在未触发样本上的准确率与原始模型相近，证明了其可转移性与高度隐蔽性。

**⚠️ 局限性**

局限性包括：仅在仿真环境下验证，未考虑实际射频链路、硬件非理想性与复杂信道干扰；依赖已知训练集的黑盒设置，实际部署中对抗能力可能受限；触发器设计以OFDM基带信号为基础，可能不适用于其他调制或传输协议。

---

## 381. Separate Before You Compress: The WWHO Tokenization Architecture

**arXiv ID:** 2603.25309 | [PDF](https://arxiv.org/pdf/2603.25309v1)

**作者:** Kusal Darshana `[一作]` `[通讯]` (Remeinium Research), Kusal Darshana (Remeinium Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 WWHO 三层分词架构与 SGPE 分词算法，解决了 BPE 在 Abugida 文字中碎片化导致的 Token Tax 问题。

**💡 创新点**

创新点在于将脚本语法与统计压缩分离，利用 DFA 进行语音节划分，并通过零拆分保证词形完整性，同时实现统一元词表跨脚本 ID 合并。

**🔧 技术方法**

主要技术包括正则表达式定义正则语言、确定性有限自动机实现节识别、改进的 GPE（SGPE）与 BPE 的联合编码、以及路由层的 Unicode 区块分段。

**📊 数据集**

实验使用约 3 千 5 万句混合语料（35% Sinhala、45% Hindi、20% English）来自 MADLAD、CC100、HINMIX 等公开数据集，评估集包含 1.5 万句共 122 万字符。

**📈 对比分析**

在与 OpenAI o200k、Meta Llama 4 Scout、DeepSeek V3 等主流分词器的 Token‑to‑Word Ratio、Token 数量及上下文窗口倍率等指标比较中，Sinhala 减少 61.7–77.2% 的 token，Hindi 减少 27–57.6%，整体降低 36.7–60.2%，上下文窗口提升至 4.38×。

**⚠️ 局限性**

局限性包括仅针对 Abugida 与复杂脚本，还需进一步扩展至更多脚本；在纯英语环境下无收益；对极少量未知 Unicode 仍可能产生 UNK；缺乏对下游任务性能的实证验证。

---

## 382. Challenges in Hyperspectral Imaging for Autonomous Driving: The HSI-Drive Case

**arXiv ID:** 2603.25510 | [PDF](https://arxiv.org/pdf/2603.25510v1)

**作者:** Koldo Basterretxea `[一作]` (University Basque Country), Javier Echanobe `[通讯]` (University Basque Country)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了基于超光谱成像的图像分割方法在自动驾驶中的应用，改进了HSI-Drive数据集的标签质量和反射率校正，并在U-Net网络中引入光谱注意力模块。

**💡 创新点**

创新点包括：①开发了轻量级的自适应光谱注意力机制（ECA）以提升模型对光谱特征的利用；②提出了基于最高反射率像素的伪反射率校正算法，增强了数据一致性；③在不增加显著计算量的前提下显著提升了分割精度。

**🔧 技术方法**

使用的技术包括：超光谱成像（25波段红-近红外）、U-Net深度网络、ECA注意力模块、像素归一化、伪反射率校正、5折交叉验证以及IoU评估指标。

**📊 数据集**

所用数据集为HSI-Drive 2.0和2.1版本，包含真实驾驶场景下的25波段超光谱图像和人工标注的语义分割标签。

**📈 对比分析**

通过5折交叉验证与mean IoU对比实验，未使用注意力和校正的U-Net与使用ECA注意力加伪反射率校正的模型相比，平均IoU提升约2%；在6类实验中，金属和行人类别的精度分别提升了10.22%和5.09%。

**⚠️ 局限性**

局限性包括：反射率校正仍采用伪白平衡，受光照变化影响导致数据一致性有限；超光谱相机的波段数和分辨率受硬件限制；对小目标和高内异类的分割精度仍有提升空间。

---

## 383. Lightweight GenAI for Network Traffic Synthesis: Fidelity, Augmentation, and Classification

**arXiv ID:** 2603.25507 | [PDF](https://arxiv.org/pdf/2603.25507v1)

**作者:** Giampaolo Bovenzi `[一作]` (University of Naples Federico II), Dario Rossi `[通讯]` (Huawei Technologies)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `67630363-6be0-4f51-ab05-7198250671a5` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了轻量级生成式AI模型用于网络流量合成，并系统评估了其在流量真实性、分类训练、低样本增广和计算效率等方面的表现。

**💡 创新点**

创新点在于将Transformer、状态空间和扩散模型压缩到1–2M参数级别，并针对网络流量特征（首10包payload长度和方向）构造低维表示，避免原始payload泄露，同时实现高保真合成和实用性。

**🔧 技术方法**

使用的技术包括基于条件Transformer（GPT‑style）、Mamba状态空间网络、Refined 2D Diffusion模型，以及离散化token化和图像映射，配合后向映射重构流量。

**📊 数据集**

实验数据集为公开的Android应用流量集（40个应用约10万biflows）和CENSUS TLS22服务集（80个服务），均使用首10包payload长度/方向特征。

**📈 对比分析**

通过与传统统计生成（SMOTE、Fast Retransmit）、基线变分自编码器以及大规模扩散模型对比，轻量级Transformer与状态空间模型在流量真实性（JS-D近零）和分类F1（最高87%）上优于基线，增广效果提升最高+40% F1，且推理延迟仅0.5–31ms，模型体积3.9–7.9MB。

**⚠️ 局限性**

限制在于仅合成首10包的header特征，无法生成完整payload；对极端长流或复杂交互序列的建模能力有限，且扩散模型在时间效率上仍显劣势。

---

## 384. An Experimental Comparison of the Most Popular Approaches to Fake News Detection

**arXiv ID:** 2603.25501 | [PDF](https://arxiv.org/pdf/2603.25501v1)

**作者:** Pietro Dell'Oglio `[一作]` (University of Pisa), Lucia C. Passaro `[通讯]` (University of Pisa)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对 12 种假新闻检测方法在 10 个公开英文文本数据集上的跨域泛化进行系统评估，设计了单域、跨域、多域和留一域四种实验设置。

**💡 创新点**

将跨域泛化作为核心评价维度，统一实验协议并同时比较传统 ML、DL、Transformer、跨域架构与 LLM 的表现，揭示 LLM 在零/少样本下的稳健性与传统模型的泛化瓶颈。

**🔧 技术方法**

采用传统 TF‑IDF 线性模型、Word2Vec CNN/BiLSTM、BERT/DeBERTa 微调、跨域 MoE MERMAID 以及 Llama3‑8B、Qwen3‑32B、Zephyr‑7B‑beta 的零/少样本提示技术。

**📊 数据集**

使用 10 个公开数据集（Celebrity、Cidii、FakeVsSatire、Fakes、Horne、Infodemic、Isot、LIAR‑PLUS、NDF、Politifact），涵盖多主题、多来源和不同标签语义。

**📈 对比分析**

通过对每个模型在四个实验组的 F1 分数进行比较，发现 DeBERTa 在单域实验中最高，LLM 在跨域与留一域实验中保持较高中位数；传统模型表现波动大，跨域架构在足够数据时稳健但数据量受限时欠佳。

**⚠️ 局限性**

限制包括未考虑时间、传播上下文与生成式 LLM 产出的假新闻，标签统一导致语义细节损失，且未做资源受限或推理成本统一评估。

---

## 385. Translation Asymmetry in LLMs as a Data Augmentation Factor: A Case Study for 6 Romansh Language Varieties

**arXiv ID:** 2603.25489 | [PDF](https://arxiv.org/pdf/2603.25489v1)

**作者:** Jannis Vamvas `[一作]` (University of Zurich), Rico Sennrich `[通讯]` (University of Zurich)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过使用 Gemini 2.5 Flash 生成的低资源语言（Romansh）文本，探究了在多变体低资源语言机器翻译中，数据增强方向的影响，并对 NLLB-200 模型进行微调，最终实现了可区分变体的高质量翻译。

**💡 创新点**

创新点在于揭示 LLM 在多变体低资源语言中的翻译不对称性，并证明将低资源语言的单语料作为数据增强来源（LR→HR）显著优于高资源语言（HR→LR），从而实现了最高 23 BLEU 的提升，并首次提供了针对各 Romansh 变体的可变体翻译系统。

**🔧 技术方法**

使用的技术包括 Gemini 2.5 Flash LLM 的 greedy 解码翻译、few-shot 示例提示、词典提示、SVM 变体分类器、NLLB-200-Distilled 1.3B 的多语言编码器-解码器模型微调，以及 BLEU、COMET、人工评测等评估方法。

**📊 数据集**

所用数据集包含约 117M 词元的 Romansh 单语料（网络、新闻、教材）、与之对应的 117M 词元 German Europarl、9M 词元的真实平行语料（包括 3M 词表条目）、以及 WMT24++ 基准测试集和 9500 条人工质量评估记录。

**📈 对比分析**

通过将 LR→HR 与 HR→LR 两种数据增强策略在同一 NLLB 模型上进行对比，使用 BLEU、COMET 和人工 7 分制评估，LR→HR 在德→Rum 翻译上平均提升 17.8 BLEU，在 Rum→德上提升 3.1 BLEU，模型平均比 Gemini 3 Pro 提高 11.4 BLEU，Sutsilvan 变体最高提升 23.2 BLEU，人工评测显示流畅度和准确度均显著优于 Gemini。

**⚠️ 局限性**

限制包括仅使用单一 LLM（Gemini 2.5 Flash）和贪婪解码，前向翻译仅基于 Europarl 数据，词典提示仅在 LR→HR 方向尝试，人工评测仅覆盖德→Rum 方向，且实验未涉及不同变体间的翻译或更广泛的低资源语言，未来可扩展至其他 LLM、采样策略及多方向翻译。

---

## 386. Navigating the Prompt Space: Improving LLM Classification of Social Science Texts Through Prompt Engineering

**arXiv ID:** 2603.25422 | [PDF](https://arxiv.org/pdf/2603.25422v1)

**作者:** Erkan Gunes `[一作]` (Constructor University), Tevfik Murat Yildirim `[通讯]` (University of Stavanger)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估了标签描述、指令提示、few-shot示例和批处理大小等三类prompt工程对LLM在社科文本分类任务（国会法案标题和情感识别）中的准确性与可重复性的影响。

**💡 创新点**

首次系统化地将prompt空间细化为三大可控组件，并通过大规模实验揭示其交互效应、边际收益递减以及非线性性能变化。

**🔧 技术方法**

使用指令化大型语言模型GPT‑4o和Gemini 2.0 Flash，在prompt中手动构造标签描述、指令提示、few‑shot示例，并调节批处理大小。

**📊 数据集**

实验数据集包括：CAP 21类国会法案标题数据集和美国国家选举调查（ANES）开放式回答情感标签。

**📈 对比分析**

在所有可能的prompt组合和批量大小下计算加权F1；结果表明至少增加一项上下文可显著提升性能，但追加更多信息往往边际收益递减；GPT‑4o在多数配置下优于Gemini；单文本批处理并非最佳策略，批量可提高成本效益。

**⚠️ 局限性**

局限在于仅测试两款商用LLM、有限的prompt组件与批量尺寸、未探索更细粒度的few‑shot数量或更多模型；结果易受prompt微调敏感，存在非确定性和可重复性问题。

---

## 387. Are LLMs Overkill for Databases?: A Study on the Finiteness of SQL

**arXiv ID:** 2603.25568 | [PDF](https://arxiv.org/pdf/2603.25568v1)

**作者:** Yue Li `[一作]` (Cornell University), Unso Eun Seo Jo `[通讯]` (Cornell University)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本研究通过对376个数据库模式和2万余条自然语言‑SQL对的系统分析，证明SQL查询的实际复杂度受到限值限制，且可用少量模板覆盖大部分查询。

**💡 创新点**

创新点在于首次量化SQL查询的复杂度上界，揭示模板化分布呈幂律，显示LLM生成代码在数据库访问领域可被高效、可审计的模板所取代。

**🔧 技术方法**

使用了LLM（Claude‑Sonnet‑4.6）生成NLQ‑SQL对，手工验证语义正确性；对SQL进行硬模板与软模板抽象；计算六类复杂度代理（表数、JOIN数、子查询数等）并进行统计与相关性分析。

**📊 数据集**

数据集来源于Spider 1.0、Spider 2.0-lite、KaggleDBQA、Bird23-train-filtered以及drawSQL公开模式，共376个数据库、约2万条SQL。

**📈 对比分析**

通过对模板频率的Power Law拟合以及覆盖率实验，发现约13%软模板可覆盖70%的查询，说明仅需几百模板即可覆盖绝大多数实际查询；与传统LLM代码生成相比，模板方法更低成本、可审计。

**⚠️ 局限性**

限制在于模板化忽略了SQL的多种等价写法，导致相同结果的不同查询可能映射到不同模板；此外研究聚焦SQLite方言，未验证在其他SQL方言下的泛化性。

---

## 388. Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes

**arXiv ID:** 2603.25562 | [PDF](https://arxiv.org/pdf/2603.25562v1)

**作者:** Yuqian Fu `[一作]` (Chinese Academy of Sciences Institute of Automation), Dongbin Zhao `[通讯]` (University of Chinese Academy of Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在长时延后训练中重新审视了基于教师的 on‑policy distillation（OPD），提出了教师 top‑K 本地支持匹配（teacher top‑K local support matching）作为一种更稳健的 token‑级监督方式。

**💡 创新点**

创新点在于将原先单一采样 token 的 OPD 换成在教师支持的 top‑K 词表范围内进行分布级别的匹配，并结合 top‑p 采样、特殊 token 掩码和支持集重归一化，平衡了偏差与方差，提升了训练稳定性和下游性能。

**🔧 技术方法**

核心技术包括：reverse‑KL 换算为 token‑级和序列级梯度估计、top‑K 支持匹配、top‑p 采样策略、特殊 token 掩码、支持集重归一化，以及基于这些的本地截断逆 KL 损失。

**📊 数据集**

实验数据集涵盖单任务数学推理（Math500、AIME24、AIME25、Minerva OlympiadBench）以及多任务交替训练（ALFWorld 代理任务+数学推理），使用 Qwen2.5‑7B‑It 作为学生模型，OpenThinker3‑7B 等教师模型。

**📈 对比分析**

与采样 token 的 OPD（以及加 mask 的版本）进行对比，采用 pass@1、success rate、平均分等指标；本方法在数学推理任务上平均分从 36.4 提升至 41.5，单任务 pass@1 提升至 82.0，且在多任务场景下保持 ALFWorld 性能不降反升，整体表现优于基线。

**⚠️ 局限性**

局限性包括：损失是截断的近似，未完全等价于全词表逆 KL；对 reward‑hacking 的解释仍是机制假设而非因果证明；教师匹配仍可能与真实任务成功度不完全对齐，需要更强的滚动策略控制和教师不确定性建模。

---

## 389. Insights on back marking for the automated identification of animals

**arXiv ID:** 2603.25535 | [PDF](https://arxiv.org/pdf/2603.25535v1)

**作者:** David Brunner `[一作]` (University of Applied Sciences Upper Austria), Maciej Oczak `[通讯]` (University of Veterinary Medicine Vienna)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

通过收集猪背部标记图像并训练ResNet‑50模型，分析并评估不同背标在运动模糊、姿态变化、遮挡及常用数据增强下的识别效果，进而给出优化背标设计的建议。

**💡 创新点**

首次系统性结合动物行为特征与机器学习数据增强，阐明哪些背标设计易导致误判，提出可操作的背标选择准则。

**🔧 技术方法**

使用深度卷积网络ResNet‑50进行图像分类，并对训练数据进行水平/垂直翻转、随机旋转、亮度/对比度/饱和度/色调变换、灰度化和模糊等数据增强。

**📊 数据集**

基于“Let me out”项目的11段观测视频（约26,260张训练裁剪图像，500张验证图像和500张测试图像）。

**📈 对比分析**

以分类准确率为评价指标，模型在训练集上达到91%，验证集69%；不同背标（如逆T、垂直线）性能差异显著，且误差分析揭示运动模糊、视角和遮挡是主要误判原因。

**⚠️ 局限性**

数据规模有限（仅10只猪、10种标记），实验环境受限于固定摄像机视角，未在更大规模或多样化真实场景中验证泛化能力。

---

## 390. Beyond the Golden Data: Resolving the Motion-Vision Quality Dilemma via Timestep Selective Training

**arXiv ID:** 2603.25527 | [PDF](https://arxiv.org/pdf/2603.25527v1)

**作者:** Xiangyang Luo `[一作]` (Tsinghua University), Shao-Lun Huang `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文针对视频生成中运动质量与视觉质量负相关导致的“运动-视觉质量困境”，提出通过时间步选择的训练方式——Timestep-aware Quality Decoupling（TQD），让不同质量的视频样本在对应的去噪阶段学习；

**💡 创新点**

创新点在于利用梯度分析发现质量失衡样本在适当时间步能产生与高质量样本相同的学习信号，并将样本的运动与视觉质量信息映射到不同的时间步采样分布，实现数据层面的质量解耦；

**🔧 技术方法**

核心技术包括预先计算视频的运动质量（MQ）与视觉质量（VQ）评分、基于样本质量的概率保留机制、以及使用Beta分布对时间步进行自适应采样；

**📊 数据集**

使用Koala36M数据集的VQ与MQ评分结果对训练集进行划分，实验中分别构造了全量、仅运动优质/视觉优质两类以及仅高质量四种子集；

**📈 对比分析**

与传统的均匀或logit‑normal时间步采样相比，TQD在包含高MQ/低VQ与低MQ/高VQ两类不平衡数据的 Set‑B 甚至在 Set‑C（仅高质量）下都实现了 VBench、VideoAlign 等指标的提升，尤其在物理推理任务 VideoPhy2 上也表现出更好的语义一致性和物理常识；

**⚠️ 局限性**

局限性包括对MQ/VQ评分模型的依赖，评分噪声虽对性能影响有限但仍可能影响时间步映射；此外，TQD在 LoRA 微调的 CogVideoX 上提升有限，表明不同模型结构对该方法的适用性存在差异。

---

## 391. An Integrative Genome-Scale Metabolic Modeling and Machine Learning Framework for Predicting and Optimizing Biofuel-Relevant Biomass Production in Saccharomyces cerevisiae

**arXiv ID:** 2603.25561 | [PDF](https://arxiv.org/pdf/2603.25561v1)

**作者:** Neha K. Nair `[一作]` (National Institute of Technology, Warangal), Aaron D'Souza `[通讯]` (National Institute of Technology, Warangal)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建了一套基于酵母9代谢网络的全流程框架，通过FBA生成的代谢流数据训练机器学习模型，预测并优化Saccharomyces cerevisiae的生物量流；

**💡 创新点**

创新点在于将代谢网络仿真、变分自编码器聚类、随机森林/梯度提升/深度网络预测、SHAP解释、贝叶斯优化与生成对抗网络等技术融合为一体化端到端工作流，首次实现对代谢状态的全面解析与可操作的优化建议；

**🔧 技术方法**

采用Flux Balance Analysis（FBA）、Variational Autoencoder（VAE）、Random Forest、XGBoost、Feed‑Forward Neural Network、SHAP、Bayesian Optimization（高斯过程）、Generative Adversarial Network（GAN）等算法；

**📊 数据集**

使用Yeast9 GEM在不同葡萄糖、氧气、铵离子摄取率下生成的2000个代谢流样本（每个样本包含4131条反应速率）作为训练与评估数据集；

**📈 对比分析**

与传统FBA/单一模型对比，Random Forest R²=0.99989，XGBoost R²=0.9990；贝叶斯优化将预测生物量从0.0858 gDW·h⁻¹提升至1.041 gDW·h⁻¹，提升约12倍；GAN生成的代谢流在满足质量约束下均可实现，方差为0.156；

**⚠️ 局限性**

局限性包括：FBA仅满足稳态假设，忽略动态调控与酶动力学；所有结论均为计算机模拟，需要实验验证；GAN可能出现模式崩塌；模型对代谢网络的泛化受限于所用的Yeast9 GEM质量。

---

## 392. Rotatable Antenna-Empowered Wireless Networks: A Tutorial

**arXiv ID:** 2603.25559 | [PDF](https://arxiv.org/pdf/2603.25559v1)

**作者:** Beixiong Zheng `[一作]` (South China University of Technology), Rui Zhang `[通讯]` (National University of Singapore)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文综述了可旋转天线（RA）在无线通信与感知中的基本原理、系统架构、数学建模、优化方法以及实际实现与实验验证，提供了从理论到实践的完整教程。

**💡 创新点**

创新点包括：① 建立了统一的RA旋转模型与通用多路径/宽带/极化通道模型；② 提出了针对RA的方向优化、信道估计与资源调度算法；③ 将RA技术与ISAC、SAGIN等新兴场景结合，展示其在能效、容量与感知精度上的显著提升；④ 通过实验原型验证了RA在不同频段和配置下的实际性能增益。

**🔧 技术方法**

采用的技术主要有：三维欧拉角旋转理论、天线方向性模式（余弦与3GPP模型）、近场/远场通道模型、广义SCA、BCD、SDP、MMSE/ZF接收、波束成形、极化匹配分析、时频信号处理与多视角估计方法；在硬件实现方面涉及机械马达、MEMS、PIN/varactor天线开关以及混合电子-机械驱动。

**📊 数据集**

实验与仿真数据主要来自自建天线阵列与环境模型（如随机散射簇、用户/目标位置分布），未使用公开数据集，而是基于仿真场景和实际原型测量（如2.4 GHz、5 GHz、THz频段）进行性能评估。

**📈 对比分析**

通过与固定方向天线、随机方向、阵列全局旋转等基线进行对比，结果显示：RA可在相同天线数量下提升 3–5 dB 的 SNR 与容量；在多用户/宽带场景中实现更高的极限速率和更小的误码率；在ISAC场景中，在保持相同通信速率的前提下，回波功率提升 2–4 dB，CRB 显著下降。

**⚠️ 局限性**

局限性主要包括：① 硬件成本与能耗在机械/电子驱动之间权衡；② 旋转速度与控制延迟影响时变场景下的跟踪性能；③ 校准与误差导致的方向误差与极化失配；④ 对大规模天线阵列的多视角信道估计仍面临计算复杂度与标定难题；⑤ 目前多频段与多波束兼容性仍需进一步研究。

---

## 393. Retraining as Approximate Bayesian Inference

**arXiv ID:** 2603.25480 | [PDF](https://arxiv.org/pdf/2603.25480v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 394. Unveiling the Resilience of LLM-Enhanced Search Engines against Black-Hat SEO Manipulation

**arXiv ID:** 2603.25500 | [PDF](https://arxiv.org/pdf/2603.25500v1)

**作者:** Pei Chen `[一作]` (Fudan University), Min Yang `[通讯]` (Fudan University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文系统评估黑帽 SEO 对大型语言模型增强搜索引擎（LLM-Enhanced Search Engines）的攻击效果，并构建了 SEO-Bench 数据集，对10款主流产品进行端到端安全测试，提出并验证了七种针对不同阶段的攻击策略。

**💡 创新点**

创新点在于：①首次将黑帽 SEO 与 LLM 搜索引擎结合，系统地划分理解、检索、总结三个阶段并评估其抵御能力；②构建了 1,000 条真实黑帽 SEO 网站数据集 SEO-Bench，填补了安全评测数据缺口；③提出并验证了针对检索重排序和总结阶段的“分段文本”和“重写查询堆叠”等新攻击手法，揭示了 LLM 搜索的潜在脆弱性。

**🔧 技术方法**

主要技术包括：基于 LLM 的查询重写、意图分析、检索重排序、文本生成与引用过滤；黑帽 SEO 检测采用内容分类器、链接特征分析；实验平台涵盖 10 款闭源（如 ChatGPT、Gemini、Perplexity 等）与开源（如 Perplexica、GPT-Researcher 等）系统。

**📊 数据集**

使用的数据集为 SEO-Bench：1,000 对查询-网站样本，来源于 2,499 个非法词、9,301 个热点词在 Google 搜索中收集的黑帽 SEO 网站；此外，还使用了公开的域名权重、内容碎片化度、跨链接等特征作为辅助评估。

**📈 对比分析**

比较方法：对每个系统在理解、检索、总结三个阶段分别计算阻断率，并汇总为累计抵御率；与传统搜索引擎 Google 的效果对比。结果显示：检索阶段阻断率 98.2%，整体累计抵御率 99.78%，远超传统搜索引擎；在攻击实验中，“分段文本”与“重写查询堆叠”在多数系统上攻击成功率均超过基准，证明其有效性。

**⚠️ 局限性**

局限性包括：仅评估了十款主流产品，可能无法代表所有 LLM 搜索引擎；攻击样本为简化版本，未覆盖高度复杂或持续性攻击；实验时间有限，未能评估多策略叠加的长期影响。

---

## 395. Knowledge-Guided Failure Prediction: Detecting When Object Detectors Miss Safety-Critical Objects

**arXiv ID:** 2603.25499 | [PDF](https://arxiv.org/pdf/2603.25499v1)

**作者:** Jakob Paul Zimmermann `[一作]` (Fraunhofer HHI), David Lerch `[通讯]` (Fraunhofer IOSB)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了一种基于知识指导的失败预测框架kgfp，用以监测目标检测器在安全关键任务中的检测缺失并在运行时发出警告。

**💡 创新点**

首次将检测器内部特征与视觉基础模型（DINO）嵌入进行双编码，通过余弦角度相似度直接预测检测器在安全关键类别上的失误，并引入监督式安全标签、跨尺度注意力与跨注意力融合以提升 id 与 ood 场景的泛化。

**🔧 技术方法**

采用双编码器架构、YOLOv8多尺度特征、DINO视觉基础模型嵌入、交叉尺度注意力、跨注意力融合以及余弦角度相似度作为失效指标，并用 BCE 训练和 LARS 优化器。

**📊 数据集**

在 COCO 2017 的 person 类别作为 id 训练/验证/测试集，并在 COCO-O 的六个视觉域（cartoon、sketch、painting、handmake、tattoo、weather）进行 ood 评估。

**📈 对比分析**

与传统 OOD 检测方法（GRAM、knn、vim）及 DINO 相关的 MLP 进行对比，kgfp 在 id 上实现 84.5% 的 Person Recall @5% fpr，超越基线 15–20%；在 COCO-O 上平均 recall 34.2%，且 auroc 达 92.9% 与 80.3%，表现显著优于基线。

**⚠️ 局限性**

依赖冻结的基础模型，面对远超训练域的视觉输入易失效；计算量大导致实时性挑战；安全标签为全或无，缺乏细粒度误检预测；仅关注人类类别，需扩展至多类别安全监测。

---

## 396. LILAC: Language-Conditioned Object-Centric Optical Flow for Open-Loop Trajectory Generation

**arXiv ID:** 2603.25481 | [PDF](https://arxiv.org/pdf/2603.25481v1)

**作者:** Motonari Kambara `[一作]` (Keio University), Komei Sugiura `[通讯]` (Keio University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 LILAC 框架，利用流基方法生成对象中心的 2D 光流并转换为 6-DoF 轨迹，实现语言指令驱动的机器人操控。

**💡 创新点**

创新点在于引入视觉提示与语义对齐损失以提升流生成与语言指令的一致性，并采用一次性前向推理的开环轨迹生成，减少循环编码与误差累积。

**🔧 技术方法**

使用视觉语言动作模型、Prompt‑Conditioned Multi‑Modal Adapter、Semantic Alignment Loss、Transformer 编码/解码结构、深度相机参数映射以及 MLLM 生成视觉提示。

**📊 数据集**

采用 Robot Flow benchmark（基于 Fractal 与 BridgeData V2 的光流标注数据）进行训练与评估，并在真实 HSR 机器人平台上使用多样化家用物体进行物理实验。

**📈 对比分析**

与 Im2Flow2Act、FLIP、π_0 等基线在 Robot Flow 基准和真实机器人实验中对比，LILAC 在 ADE、P@K、AUC 以及任务成功率等指标上均优于基线，平均成功率提升约 14%。

**⚠️ 局限性**

局限性包括假设静态场景、2D 光流对深度运动表达不足、对视觉提示错误敏感、以及在多物体复杂场景下视觉提示生成仍可能失败，且无法在动态环境中实现实时重规划。

---

## 397. How Class Ontology and Data Scale Affect Audio Transfer Learning

**arXiv ID:** 2603.25476 | [PDF](https://arxiv.org/pdf/2603.25476v1)

**作者:** Manuel Milling `[一作]` (Technical University of Munich), Björn W. Schuller `[通讯]` (Technical University of Munich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文通过在 AudioSet 的不同子集上预训练模型，再在 ASC、BAD、SCR 三个听觉任务上微调，系统研究了预训练数据规模、标签粒度与任务相似度对音频迁移学习效果的影响。

**💡 创新点**

创新点在于首次对音频领域的预训练数据选择进行细粒度的分层评估，揭示任务相似度往往比数据规模更能决定迁移效果，并通过表示相似度分析进一步验证了这一发现。

**🔧 技术方法**

使用基于 10 层卷积的 CNN10 体系结构进行多标签预训练与微调，并采用交叉熵/软交叉熵损失、Adam 优化器、随机种子平均和余弦距离分析来评估模型表示。

**📊 数据集**

采用 AudioSet（完整及按语音、人声、事物、动物、自然声音等子集划分）进行预训练，微调任务使用 DCASE 2020 版 ASC、BAD、SCR 数据集。

**📈 对比分析**

与随机初始化基线比较，利用 UAR 评价微调性能，结果显示更大类目数量与更高任务相似度的预训练子集均能显著提升性能；部分子集甚至优于使用完整 AudioSet 的预训练模型。

**⚠️ 局限性**

局限性包括仅使用单一 CNN 架构、固定训练周期、资源受限导致实验规模有限，以及预训练过程中未剔除非目标标签样本，可能导致表面统计特征的干扰。

---

## 398. Causal-INSIGHT: Probing Temporal Models to Extract Causal Structure

**arXiv ID:** 2603.25473 | [PDF](https://arxiv.org/pdf/2603.25473v1)

**作者:** Benjamin Redden `[一作]` (Queen's University Belfast), Shuyan Li `[通讯]` (Queen's University Belfast)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了Causal-INSIGHT，一种在训练后通过输入夹紧和影响信号提取任意预训练时间序列预测器的有向时滞因果图的后置解释框架。

**💡 创新点**

创新点在于将因果图提取从模型训练中解耦出来，采用基于模型夹紧的干预式探测来推断影响关系，并提出稀疏感知的Qbic准则实现无监督图选择。

**🔧 技术方法**

使用了干预式输入夹紧、绝对偏差影响张量、时序峰值归约以及Qbic评分进行图稀疏化，并对多种骨干网络（MLP、CNN、LSTM、Transformer）进行统一测试。

**📊 数据集**

在四类合成因果结构（分叉、V、介导、菱形）、Lorenz‑96混沌系统以及28个模拟fMRI数据集上进行评估。

**📈 对比分析**

与cMLP、cLSTM、TCDF、CUTS、CausalFormer及PCMCI+等深度与经典因果发现基线对比，Causal-INSIGHT在结构F1上保持竞争力，并在时滞定位（PoD）上显著优于同类方法，尤其在与CausalFormer结合时提升了时序精度。

**⚠️ 局限性**

局限性在于仅恢复模型暗含的Granger‑样依赖关系，缺乏对真实数据生成因果图的可识别性，且对因果充分性、平稳性等假设敏感，单变量夹紧也可能忽略更高阶交互。

---

## 399. Improving metadata flows -- The simultaneous use of multiple metadata schemas at disciplinary research data repositories

**arXiv ID:** 2603.25468 | [PDF](https://arxiv.org/pdf/2603.25468v1)

**作者:** Dorothea Strecker `[一作]` `[通讯]` (Humboldt-Universität zu Berlin), Dorothea Strecker (Humboldt-Universität zu Berlin)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究在科研数据仓库中同时使用多种元数据模式的情况，比较基于学科元数据模式与DataCite元数据模式的记录。

**💡 创新点**

通过对三种学科元数据模式与DataCite的结构差异映射，评估跨表关系，并证明可通过优化跨表显著提升DataCite元数据完整性，展示了跨模式整合的实用价值。

**🔧 技术方法**

利用OAI‑PMH抓取、DataCite API、元数据解析、跨表映射、统计分析（t检验）与时间序列分析等技术。

**📊 数据集**

共采集8个科研数据仓库的元数据记录（4地球科学、4社会科学），约11280+12870条记录，分别使用ISO 19139、DIF、DDI等学科模式。

**📈 对比分析**

采用元素级别对齐和覆盖率评估，对同一数据集在不同模式下的记录进行比较，计算潜在改进百分比，发现10%–100%不等的提升；时间序列分析揭示了不同仓库的元数据工作流程差异。

**⚠️ 局限性**

样本规模有限；仅包含具备DOI注册和OAI‑PMH接口的仓库，难以推广到其他学科或技术不够成熟的仓库。

---

## 400. VideoWeaver: Multimodal Multi-View Video-to-Video Transfer for Embodied Agents

**arXiv ID:** 2603.25420 | [PDF](https://arxiv.org/pdf/2603.25420v1)

**作者:** George Eskandar `[一作]` (Huawei Heisenberg Research Center), Ziyuan Liu `[通讯]` (Huawei Heisenberg Research Center)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出VideoWeaver框架，实现多视角视频翻译，保持跨视角一致性。

**💡 创新点**

创新点包括使用4D点云统一潜空间、Mixture-of-Experts融合深度与草图、异步时钟训练实现自回归视角生成。

**🔧 技术方法**

技术涵盖基于DiT的流模型、Pi3 4D点云注入、Mixture-of-Experts、跨视角注意力、异步时钟训练。

**📊 数据集**

使用Droid、Agibot、Bridgev2以及内部5K视频数据集进行训练，评估在这些数据集及未见数据集上的表现。

**📈 对比分析**

与VACE、Cosmos-Transfer-1、ControlVideo、Control-A-Video等单视角基线对比，VideoWeaver在多视角一致性和视觉质量上显著优于或匹配基线。

**⚠️ 局限性**

局限包括对小物体的细节一致性不足、生成序列长度固定且缺乏时序自回归能力。

---

## 401. TAPO: Translation Augmented Policy Optimization for Multilingual Mathematical Reasoning

**arXiv ID:** 2603.25419 | [PDF](https://arxiv.org/pdf/2603.25419v1)

**作者:** Xu Huang `[一作]` (Nanjing University), Shujian Huang `[通讯]` (Nanjing University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Translation-Augmented Policy Optimization (TAPO) 框架，利用翻译先理解再推理的方式提升多语言数学推理性能

**💡 创新点**

创新点在于通过step-level相对优势机制解耦翻译质量奖励与推理奖励，避免奖励冲突并实现对理解与推理的联合优化

**🔧 技术方法**

结合GRPO（on-policy强化学习）+翻译质量评估指标（ChrF++、xCOMET、CometKiwi）+公式化的格式与推理奖励

**📊 数据集**

使用MGSM8KInstruct、GSM8K（翻译成Telugu）等多语言训练集，并在MGSM、MMATH、MSVAMP等基准上评测

**📈 对比分析**

与SFT、QAlign、RAFT、GRPO等基线对比，TAPO在已训练语言上平均提升约8–10%准确率，在未训练语言和OOD任务上保持竞争力，且翻译质量亦显著提升

**⚠️ 局限性**

依赖翻译指标的鲁棒性、仅在英语推理导致潜在信息丢失、仅验证小型模型且未对大模型扩展、可能对低资源语言的奖励劫持问题需进一步研究

---

## 402. NERO-Net: A Neuroevolutionary Approach for the Design of Adversarially Robust CNNs

**arXiv ID:** 2603.25517 | [PDF](https://arxiv.org/pdf/2603.25517v1)

**作者:** Inês Valentim `[一作]` (University of Coimbra), Nuno Lourenço `[通讯]` (University of Coimbra)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种基于神经进化的框架 NERO-Net，用来自动搜索能够在无对抗训练下实现本质鲁棒性的卷积神经网络。

**💡 创新点**

创新点在于将对抗鲁棒性与清洁准确度统一进适应度函数，并通过可变层块和跳跃连接扩展搜索空间，首次实现了不依赖对抗训练即可获得本质鲁棒性的模型。

**🔧 技术方法**

采用 Fast‑DENSER 进化框架、双层语法编码、FGSM 估计对抗准确率、(1+λ) ES 进化策略以及权重 β 的调度等技术。

**📊 数据集**

实验数据集为 CIFAR‑10。

**📈 对比分析**

与 NSGA‑Net 等基准对比，最佳个体在标准训练下清洁准确率约 93%，FGSM 对抗准确率约 48%，在 AutoAttack 下对抗准确率约 40%，显示出相较于传统模型显著的鲁棒性提升；但与专门针对对抗训练的 NAS 模型相比仍有差距。

**⚠️ 局限性**

主要限制包括高昂的计算成本、对抗准确率对基因微调高度敏感、对抗训练效果受限于超参数设置以及模型规模较大导致收敛速度慢。

---

## 403. AdaSFormer: Adaptive Serialized Transformers for Monocular Semantic Scene Completion from Indoor Environments

**arXiv ID:** 2603.25494 | [PDF](https://arxiv.org/pdf/2603.25494v1)

**作者:** Xuzhi Wang `[一作]` (Tianjin Normal University), Ziping Zhao `[通讯]` (Tianjin Normal University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于自适应序列化Transformer的室内单目语义场景完成框架 AdaSFormer，能在低内存条件下捕获全局上下文并恢复被遮挡结构。

**💡 创新点**

创新点包括：①可学习的序列化位移（Adaptive Serialized Attention）通过 Gumbel‑Softmax 动态调整感受野；②以场景中心为参照的相对姿态编码（Center‑Relative Positional Encoding）提升空间语义表达；③卷积调制层归一化（Conv‑Modulated Layer Normalization）实现Transformer与CNN特征的无缝融合。

**🔧 技术方法**

核心技术包括：2D 编码 + 深度估计 → 3D 投影；空间填充曲线序列化 + 可变窗口注意力；可学习位移与温度退火；多尺度特征融合的轻量级 DDR 解码器；交叉熵、几何与语义亲和损失的联合训练。

**📊 数据集**

在 NYUv2（1,449张RGB+深度图）和大规模室内数据集 Occ‑ScanNet（45,755张）上进行评估。

**📈 对比分析**

与多种现有方法（SSCNet、ISO、MonoScene 等）对比，AdaSFormer 在 NYUv2 上 mIoU 32.50（比 ISO 高 1.25），在 Occ‑ScanNet 上 mIoU 45.33（比 ISO 高 14.5），同时推理时延仅 178.9 ms，显著低于 ISO 的 535.2 ms。

**⚠️ 局限性**

局限性：仅在室内单目场景完成任务中验证，缺乏对室外或多视角、动态场景的泛化评估；模型仍需更高效的并行化策略以进一步压缩内存占用。

---

## 404. Maximum Entropy Behavior Exploration for Sim2Real Zero-Shot Reinforcement Learning

**arXiv ID:** 2603.25464 | [PDF](https://arxiv.org/pdf/2603.25464v1)

**作者:** Jiajun Hu `[一作]` (EPFL), Stelian Coros `[通讯]` (ETH Zurich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `40105733-5154-44cd-8090-a8cab9e64b07`

**🎯 论文内容**

开发了一种在线零样本强化学习算法MEBE，用于四足机器人控制；通过最大熵行为探索和行为正则化实现了自然且可直接部署的策略。

**💡 创新点**

创新点在于用熵最大化驱动探索，迫使行为分布均匀，同时引入正则化critic以限制不自然的运动，使得在线FB算法能够在无外部数据的前提下获得多样、可迁移的策略。

**🔧 技术方法**

技术上采用Forward‑Backward（FB）框架、最大熵行为探索策略、行为正则化critic、Normalizing Flow密度估计、IsaacLab仿真平台以及真实Unitree Go2四足机器人。

**📊 数据集**

使用的数据集为自生成的在线经验回放，依赖IsaacLab环境模拟的Unitree Go2机器人状态，无需外部运动捕捉或预收集数据。

**📈 对比分析**

与FB、FB‑Critic、FB‑MEBE（β=0）以及有监督的Fast‑TD3进行对比；在17个速度追踪任务和12个姿态任务中，MEBE取得最高或相当的平均回报、最高熵、最低脚滑；并能在真实机器人上实现零样本部署。

**⚠️ 局限性**

局限性包括：当下游任务远离训练分布时性能衰退；探索仍相对宽松，可能探索与任务无关的极端状态；缺乏对FB算法优先级的理论解释，需进一步提升安全性和针对性探索。

---

## 405. CIAR: Interval-based Collaborative Decoding for Image Generation Acceleration

**arXiv ID:** 2603.25463 | [PDF](https://arxiv.org/pdf/2603.25463v1)

**作者:** Keming Ye `[一作]` (Zhejiang University), Shengyu Zhang `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出CIAR框架，实现云-设备协同的自回归图像生成，利用设备端不确定性量化与云端增强解码加速推理并保持图像质量。

**💡 创新点**

创新点在于设计基于连续概率区间的Inter‑Head实现设备端不确定性量化，结合区间增强解码、前缀注入和分布鲁棒对齐训练，显著降低云端请求并提升速度。

**🔧 技术方法**

使用技术包括自回归模型、区间不确定性量化、分布鲁棒优化（Inter‑DRO）、前缀注入、云端增强解码及分布对齐训练策略。

**📊 数据集**

实验数据集为MS‑COCO验证集的文本提示，评估生成图像质量。

**📈 对比分析**

与EAGLE‑2、Lantern、Entropy‑lens等基线比较，CIAR实现约2.18×速度提升，云请求减少70%，同时FID、CLIP等指标与基准相当或更佳。

**⚠️ 局限性**

局限在于仍受模型规模与云端延迟约束，且在极端边界细节和极大词表/高分辨率下区间估计精度和推断稳定性需进一步提升。

---

## 406. Temporally Decoupled Diffusion Planning for Autonomous Driving

**arXiv ID:** 2603.25462 | [PDF](https://arxiv.org/pdf/2603.25462v1)

**作者:** Xiang Li `[一作]` (Bosch Cross-Domain Computing Solutions), Jianjun Wang `[通讯]` (Bosch Cross-Domain Computing Solutions)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种时序解耦扩散模型 TDDM，用于自动驾驶轨迹规划。

**💡 创新点**

将轨迹拆分为时间段，采用独立噪声和时序解耦的 TD‑AdaLN 及非对称时序 CFG，提升短期安全与长期目标一致性。

**🔧 技术方法**

扩散模型、时间解耦自适应层归一化、轨迹分词、anchor 词表、Classifier‑Free Guidance 与 DPM‑Solver++ 推理等技术。

**📊 数据集**

使用 nuPlan 大规模真实城市驾驶数据集。

**📈 对比分析**

与规则基、混合、学习基和 Diffusion Planner 等在 nuPlan Val14、Test14‑hard、Test14‑random 上进行 NR/R 评估，TDDM 在 Test14‑hard 以 77.95 分超过 Diffusion Planner，整体表现领先或相当。

**⚠️ 局限性**

在完全闭环动态仿真中仍欠佳，需改进交互、去除 anchor 依赖、进一步利用时序令牌。

---

## 407. DC-Reg: Globally Optimal Point Cloud Registration via Tight Bounding with Difference of Convex Programming

**arXiv ID:** 2603.25442 | [PDF](https://arxiv.org/pdf/2603.25442v1)

**作者:** Wei Lian `[一作]` (Changzhi University), Wangmeng Zuo `[通讯]` (Harbin Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种基于DC分解的全局最优点云配准框架DC-Reg。

**💡 创新点**

创新点在于全局DC下界逼近，取代传统逐项松弛，显著收敛加速。

**🔧 技术方法**

使用Difference of Convex编程、Branch-and-Bound、线性赋值问题与旋转不变特征。

**📊 数据集**

在合成2D/3D数据、3DMatch、Caltech-256等数据集上评测。

**📈 对比分析**

与RPM-HTB、Go-ICP、TEASER++等方法比较，DC-Reg在噪声、遮挡、外点等极端条件下更稳健、误差更低。

**⚠️ 局限性**

限制是高维旋转-平移参数仍导致搜索树增长，计算时间在极端噪声/大尺度变形时仍较慢。

---

## 408. From Manipulation to Mistrust: Explaining Diverse Micro-Video Misinformation for Robust Debunking in the Wild

**arXiv ID:** 2603.25423 | [PDF](https://arxiv.org/pdf/2603.25423v1)

**作者:** Zhi Zeng `[一作]` (Xi'an Jiaotong University), Minnan Luo `[通讯]` (Xi'an Jiaotong University)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了WildFakeBench大规模微视频误信息基准，并提出FakeAgent多代理推理框架用于检测和解释微视频误信息。

**💡 创新点**

创新点在于：①提供多维度、专家标注的真实案例基准；②使用Delphi式多代理融合内部内容分析与外部证据实现可解释的归因推理；③实现跨模态、多源的误信息检测。

**🔧 技术方法**

采用多模态大型语言模型（如Qwen3）进行内容分析、检索代理、定位代理和整合代理；结合检索增强生成（RAG）、链式思维（CoT）和多视角推理。

**📊 数据集**

使用WildFakeBench（10,107条微视频，涵盖10种误信息子类型）以及公开的事实核查资源（PolitiFact、中国互联网联合谣言辟谣平台、Wikipedia等）。

**📈 对比分析**

与多种主流MLLM（InternVL、Qwen2.5-VL、Gemma3、GPT‑4o等）对比，FakeAgent在Micro‑Acc上均高于对手，尤其在多模态操纵与外部语境误信息方面显著提升。

**⚠️ 局限性**

局限性包括：在某些细分类型（如AIGC）仍未达最优；对新出现的误信息形式依赖检索质量；基准主要来源于中英文平台，可能不完全覆盖全球多样性。

---

## 409. TAAC: A gate into Trustable Audio Affective Computing

**arXiv ID:** 2603.25570 | [PDF](https://arxiv.org/pdf/2603.25570v1)

**作者:** Xintao Hu `[一作]` (Hefei University of Technology), Feng-Qi Cui `[通讯]` (Hefei University of Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9cc9baba-5356-466d-81ff-d80028d90279` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种可信的音频情感计算框架，通过子空间分解和可调噪声扩散加密实现抑郁检测与身份信息保护。

**💡 创新点**

创新点在于将子空间分解与确定性扩散加密结合，既保持抑郁相关特征的可识别性，又实现身份信息的可逆加密与可调隐私级别。

**🔧 技术方法**

采用差分子空间自编码器、可调噪声扩散加密、VPM 分类器以及分阶段训练的多任务学习方法。

**📊 数据集**

在公开的 DAIC‑WOZ 临床访谈语音数据集上进行实验。

**📈 对比分析**

与传统的 Chaos‑Map 和同态加密方法相比，保持 78–84% 以上的抑郁检测准确率的同时，身份识别准确率被压至约 50% 以内，显示出更优的准确性与隐私平衡。

**⚠️ 局限性**

局限性包括加密强度提升会进一步降低检测准确率、DP‑SGD 训练耗时显著增加、对噪声和较长语音片段的鲁棒性仍有待提升。

---

## 410. Investigating the Fundamental Limit: A Feasibility Study of Hybrid-Neural Archival

**arXiv ID:** 2603.25526 | [PDF](https://arxiv.org/pdf/2603.25526v1)

**作者:** Marcus Armstrong `[一作]` (University of Houston), Arjun Mukherjee `[通讯]` (University of Houston)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现Hybrid-LLM混合神经-符号压缩架构，验证其在冷存储场景下的可行性。

**💡 创新点**

通过Logit量化协议消除GPU非确定性，实现硬件无关的精确解码；结合内容感知Scout实现算力与存储的最优协同。

**🔧 技术方法**

采用LLM预测+算术编码、Logit量化、固定KV缓存分块、分布式块级并行与内容感知路由等技术。

**📊 数据集**

使用Alice in Wonderland、Canterbury Corpus以及2025年12月的新闻数据集进行评测。

**📈 对比分析**

与GZIP、ZPAQ、LLMZip、Zstd等基线对比；Alice文本压缩率0.39 BPC（20.5×），新闻文本0.75 BPC（10.7×），显著优于传统方法，推理延迟约2600×慢。

**⚠️ 局限性**

主要局限在推理延迟高、仅适用于冷存储、对硬件量化精度敏感、分块并行导致带宽瓶颈、以及模型在低资源语言上的表现差异。

---

## 411. EcoThink: A Green Adaptive Inference Framework for Sustainable and Accessible Agents

**arXiv ID:** 2603.25498 | [PDF](https://arxiv.org/pdf/2603.25498v1)

**作者:** Linxiao Li `[一作]` (University of Sydney), Zhixiang Lu `[通讯]` (University of Liverpool)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了一个在生成式AI中根据查询复杂度动态路由的EcoThink绿色推理框架，分为低能耗检索路径和高能耗推理路径。

**💡 创新点**

1）提出能源感知的决策理论路由器，依据复杂度决定是否使用深度推理；2）将RAG+量化模型与UniMath-CoT/Tree of Thoughts结合，构建低能耗与高能耗双路；3）引入物理基础能耗模型（PUE、TDP等），实现可量化的碳排放评估。

**🔧 技术方法**

DistilBERT轻量级路由器；Qwen3‑VL‑2B（量化）与Qwen3‑VL‑8B；RAG（BM25+DPR）检索；Chain‑of‑Thought、UniMath‑CoT、Tree of Thoughts；Early‑Exit与验证机制；物理能耗公式。

**📊 数据集**

9个基准数据集：GSM8K、SVAMP、StrategyQA、ARC‑C、HotpotQA、WebQuestions、TriviaQA、MT‑Bench、TruthfulQA。

**📈 对比分析**

与闭源 SOTA（GPT‑4o、Claude 3.5 Sonnet、Gemini 2.5 Pro）、开源 Llama‑3.1‑8B、Qwen‑3‑8B 以及 FrugalGPT 进行对比，评估准确率、吞吐量与碳排放。EcoThink 在保持约 90 % 以上准确率（接近 SOTA）的同时，平均能耗下降 40 %（最优 81.9 %），吞吐量提升至 148 tokens/s。

**⚠️ 局限性**

仍需手动调优路由阈值；高复杂查询仍会消耗较多能耗；在极低资源环境下模型压缩仍有限；多模态任务的能耗评估尚不完善；对极长上下文支持有限。

---

## 412. SHADOW: Seamless Handoff And Zero-Downtime Orchestrated Workload Migration for Stateful Microservices

**arXiv ID:** 2603.25484 | [PDF](https://arxiv.org/pdf/2603.25484v1)

**作者:** Hai Dinh-Tuan `[一作]` `[通讯]` (Technical University Berlin), Hai Dinh-Tuan (Technical University Berlin)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出并实现了 SHADOW 框架，在 Kubernetes 上实现无停机的有状态微服务迁移，支持 StatefulSet 与 Deployment 两种工作负载。

**💡 创新点**

通过将服务流量路由与 StatefulSet 所有权分离，使用 ShadowPod 并引入 ExchangeFence 身份交换机制，突破了 StatefulSet 约束导致的停机瓶颈。

**🔧 技术方法**

利用 Kubernetes Operator、CRIU、FCC、OCI 镜像、RabbitMQ 消息回放、ms2m-agent、ExchangeFence 以及消息队列控制协议。

**📊 数据集**

实验使用自建的 Go 生产者与 Python 消费者微服务、RabbitMQ 消息中间件，并在 280 次迁移实验中收集数据；无公开数据集。

**📈 对比分析**

与传统顺序（Sequential）Baseline 在四种配置和七个消息速率下对比，ShadowPod 在低速率下可将总迁移时间降低 73–76%、恢复阶段时间减少 92%，且停机时间为 0；高速率受 120 s 截止阈值限制。

**⚠️ 局限性**

在高消息速率下重放队列爆炸导致 120 s 截止；ShadowPod 迁移后放弃 StatefulSet 的完整控制权；需要改进增量 CRIU、批量回放等以进一步提升性能。

---

## 413. GridVAD: Open-Set Video Anomaly Detection via Spatial Reasoning over Stratified Frame Grids

**arXiv ID:** 2603.25467 | [PDF](https://arxiv.org/pdf/2603.25467v1)

**作者:** Mohamed Eltahir `[一作]` (King Abdullah University of Science and Technology), Sondos Mohamed `[通讯]` (National Center for Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了 GridVAD，一种无训练的开集视频异常检测管线，利用 VLM 生成异常提议，随后用 Grounding DINO 进行空间定位，再用 SAM2 进行像素级掩码传播。

**💡 创新点**

创新点包括：①将 VLM 从直接判别转变为异常提议者；②引入 Self‑Consistency Consolidation (SCC) 对多次采样提议进行统计去噪；③使用分层网格把视频映射为单帧图像，实现固定 VLM 调用次数并显著提升调用效率；④在不需要训练的情况下实现像素级异常定位。

**🔧 技术方法**

核心技术：Qwen3‑VL 30B 作为 VLM、SCC 统计过滤、Grounding DINO 作为零样本目标检测器、SAM2 进行视频级掩码传播、分层网格采样与时间映射。

**📊 数据集**

评测数据集：UCSD Ped2（行人异常）和 ShanghaiTech Campus（校园异常）

**📈 对比分析**

与多种训练基线（AdaCLIP、AnomalyCLIP、TAO 等）和零样本方法对比，GridVAD 在 UCSD Ped2 上 Pixel‑AUROC 达到 77.59，超过所有比较方法（含部分微调的 TAO 75.11），在像素级指标表现最好；在对象级 RBDC/TBDC 上相对较低，但可通过 SCC 进行精度召回调节。

**⚠️ 局限性**

局限性：①异常召回受 VLM 提议质量限制，无法覆盖所有异常；②跨 clip 边界的异常可能被遗漏或重复；③计算成本相对较高（VLM 调用量虽固定，但每次调用较昂贵）；④缺乏跨 clip 全局上下文建模。

---

## 414. GeoHeight-Bench: Towards Height-Aware Multimodal Reasoning in Remote Sensing

**arXiv ID:** 2603.25565 | [PDF](https://arxiv.org/pdf/2603.25565v1)

**作者:** Xuran Hu `[一作]` (KTH Royal Institute of Technology), Wufan Zhao `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了GeoHeight-Bench和GeoHeight-Bench+两套高度感知遥感基准，并提出GeoHeightChat LMM，通过两阶段训练实现对垂直维度的无高度输入下的推理。

**💡 创新点**

① 自动化 VLM‑驱动的数据生成管线，生成海量高度相关的问答与分割样本；② 通过隐式几何对齐的 GeoAdapter，使模型能在仅有光学影像时捕获高度信息；③ 首个在遥感领域具备高度感知能力的 LMM 基线。

**🔧 技术方法**

使用 Vision‑Language 模型（CLIP ViT‑Large + Llama‑2‑7B + LoRA）、ConvNeXt‑Tiny 作为教师、Bottleneck GeoAdapter、SAM 分割、Prompt Engineering、自动一致性检查、Smooth‑L1、交叉熵、BCE+Dice 等技术。

**📊 数据集**

基于 GeoNRW、RSMSS、FLAIR 等遥感数据集，并结合官方 DTM/DEM/DSM 等高程产品生成约数十万条高质量图文对，覆盖像素级、实例级、场景级以及推理级任务。

**📈 对比分析**

采用模板化问答评估、mIoU/cIoU、准确率等指标进行比较。GeoHeightChat 在 GeoHeight‑Bench 上整体准确率达 44.14%，远超 GPT‑4o（21.61%）和多款开源 LMM；在 GeoHeight‑Bench+ 上整体得分 65.59%，领先闭源模型和 30B 级开源模型，尤其在 Reasoning 任务中显著提升。

**⚠️ 局限性**

仍受限于仅依赖光学影像推断高度，光照/云遮挡等因素可能导致误差；基准样本仍以单源、单时相为主，缺乏多时相、多源融合的验证；未评估实时推理效率与大规模部署成本；缺少跨域泛化实验。

---

## 415. Towards Comprehensive Real-Time Scene Understanding in Ophthalmic Surgery through Multimodal Image Fusion

**arXiv ID:** 2603.25555 | [PDF](https://arxiv.org/pdf/2603.25555v1)

**作者:** Nikolo Rohrmoser `[一作]` (Carl Zeiss AG), Nassir Navab `[通讯]` (Technical University of Munich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `6514db3d-8de6-452c-91b7-acdb31787cc4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一个多模态实时框架，融合手术显微镜图像与iOCT，实现了仪器检测、关键点定位和工具-视网膜距离估计。

**💡 创新点**

创新点在于跨模态注意力融合模块与时间递归单元，及分布式距离回归头提供置信度评估，显著提升了近距离深度测量精度。

**🔧 技术方法**

采用Yolo-NAS作为OPMI编码器、改进的ResNet-18提取iOCT特征、跨模态注意力融合、GRU/LSTM递归模块以及分布式回归损失。

**📊 数据集**

使用SynthesEyes公司构建的全注释合成多模态眼科手术数据集，包含20段视频、两种手术类型和四类仪器。

**📈 对比分析**

与单模态基线相比，双模态模型在单帧下mAP50提升至95.79%，距离估计平均误差从480µm下降至128µm，近距离误差从284µm降至33µm；实时速度保持在22.5 ms/帧。

**⚠️ 局限性**

局限在于对iOCT数据的依赖导致在持续失效时性能下降，缺乏对真实临床噪声和多种病理场景的验证。

---

## 416. Voxtral TTS

**arXiv ID:** 2603.25551 | [PDF](https://arxiv.org/pdf/2603.25551v1)

**作者:** Alexander H. Liu `[一作]`, Zhenlin Xu `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Voxtral TTS，一种可在仅3秒参考音频下实现多语种零样本语音克隆的 TTS 模型；

**💡 创新点**

创新点在于将自回归语义 token 生成与流匹配声学 token 结合的混合架构，以及自研的 Voxtral Codec（语义 VQ+FSQ 声学量化）和针对流匹配的 Direct Preference Optimization；

**🔧 技术方法**

核心技术包括 Transformer 自回归解码器、流匹配（flow‑matching）Transformer、ASR 蒸馏的语义 token、FSQ 声学量化、CUDA Graph 加速、异步分块流式推理；

**📊 数据集**

训练使用了自标注的 Voxtral Mini Transcribe 数据集（音频+伪字幕），以及 SEED‑TTS、MiniMax‑TTS、Expresso 等公开语音数据集；

**📈 对比分析**

在 SEED‑TTS、MiniMax‑TTS 以及多语言零样本克隆评测中，Voxtral TTS 的 WER、UTMOS 与 Speaker Similarity 均优于 ElevenLabs v3/Flash，零样本克隆人类评测中获胜率达 68.4%，在旗舰声线评测中与 Gemini 2.5 Flash 接近；

**⚠️ 局限性**

局限性包括对高 α 的情绪遵从度下降、NFE 过高导致 WER 稍微上升，以及对极低质量音频的鲁棒性待提升。

---

## 417. Towards Embodied AI with MuscleMimic: Unlocking full-body musculoskeletal motor learning at scale

**arXiv ID:** 2603.25544 | [PDF](https://arxiv.org/pdf/2603.25544v1)

**作者:** Chengkun Li `[一作]` (EPFL), Alexander Mathis `[通讯]` (EPFL)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出 MuscleMimic，一套基于 GPU 并行的肌肉驱动运动模仿学习框架，并构建了两个全身/上肢生物力学模型（MyoBimanualArm 与 MyoFullBody），实现千级多样化动作的训练与快速微调。

**💡 创新点**

创新点包括：① 通过 MuJoCo Warp 实现数千并行肌肉环境，显著提升训练吞吐量（≈20 h完成10亿步）；② 引入单梯度 epoch（E=1）与大批量训练策略，避免了高并行下的分布偏移；③ 设计 GMR‑Fit 运动重定向管线，使 SMPL 动作在满足肌肉约束的同时降低关节违规、触地穿透等问题；④ 用单一通用策略覆盖数百种运动，证明了肌肉驱动模型可在多任务上泛化。

**🔧 技术方法**

技术实现基于 JAX + MuJoCo‑Warp 的 GPU 加速仿真；强化学习采用 PPO + gated residual MLP，加入肌肉激活动态、动作空间约束与多目标奖励；重定向采用 GMR‑Fit 与 Mocap‑Body；评价使用 EMG 相关性、关节角度误差、GRF、跑步/步行 kinematics 等实验数据。

**📊 数据集**

使用 AMASS（972/108/1770/312 条全身/上肢动作）作为训练/测试集，跑步/步行实验数据来自公开的实验室记录；EMG 对比使用 Wang 与 Boo 两个步态实验数据集。

**📈 对比分析**

与 KINESIS、KinTwin 等现有基准对比，MyoFullBody 的关节角误差约 6–7°、速度误差 17–27°/s；跑步/步行 kinematics 与 GRF 的相关系数 ≥0.8；与 Mocap‑Body 对比，GMR‑Fit 在关节限制、地面穿透和肌肉跳跃率上显著下降。单一通用策略在 2 天内即可完成多运动训练，微调仅需数小时。

**⚠️ 局限性**

局限性包括：① Hill‑type 肌肉模型简化了肌腱弹性、肌束角等真实生理细节；② 运动重定向依赖 SMPL 平均体型，可能在异常体型或病理步态下产生误差；③ 当前验证仅覆盖步行、跑步等基本动作，对更复杂动作（跳跃、踢腿、战斗动作）缺乏独立实验数据；④ 仍需要外部实验验证肌肉激活与关节力学的真实性，模型预测结果不宜直接用于临床决策。

---

## 418. PAWS: Perception of Articulation in the Wild at Scale from Egocentric Videos

**arXiv ID:** 2603.25539 | [PDF](https://arxiv.org/pdf/2603.25539v1)

**作者:** Yihao Wang `[一作]` (Aalto University), Arno Solin `[通讯]` (Aalto University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

开发了一种无需监督训练的框架PAWS，能够直接从大规模一目视摄像机视频中提取室内可动作物体的关节运动参数。

**💡 创新点**

创新点在于将手-物体交互轨迹、全景几何重建与视觉语言模型（VLM）推理结合，实现在野外真实场景中可扩展、无标注的姿态感知。

**🔧 技术方法**

采用3D手姿态估计（MANO）、多视角几何与线三角化、Manhattan世界假设、OU-RTS轨迹平滑、LO-RANSAC线拟合，以及VLM+LLM的VQA推理。

**📊 数据集**

使用HD-EPIC、Arti4D、Epic-Fields等公开数据集，并在这些数据集上构建自制的PAWS训练集。

**📈 对比分析**

与Articulation3D、Articulate-Anything、ArtiPoint、iTACO等基线进行对比实验，PAWS在HD-EPIC与Arti4D上的MAO/MA评估指标均显著优于对手，并通过微调提升USDNet与Spot机器人执行性能。

**⚠️ 局限性**

局限性包括对低纹理或快速移动物体的几何重建依赖度高，VLM推理在极端遮挡下可能失准；目前仅支持单自由度关节，且对手部关键点估计的鲁棒性有一定要求。

---

## 419. BFMD: A Full-Match Badminton Dense Dataset for Dense Shot Captioning

**arXiv ID:** 2603.25533 | [PDF](https://arxiv.org/pdf/2603.25533v1)

**作者:** Ning Ding `[一作]` (Nagoya Institute of Technology), Toru Tamaki `[通讯]` (Nagoya Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了BFMD数据集，并提出了基于VideoMAE的多模态射击描述框架，利用语义反馈提升击球描述的语义一致性。

**💡 创新点**

创新点包括：①首次发布包含完整比赛、回合、击球事件的多层次密集注释数据集；②将球拍轨迹、玩家位置、姿态与视频特征融合的多模态注释；③设计语义反馈机制，将击球类型、轨迹等语义属性回传至解码器，增强描述准确性。

**🔧 技术方法**

主要技术手段有VideoMAE视觉编码器、Token Refiner、跨模态融合与跨注意力、Transformer解码器、语义反馈模块以及基于GPT‑4.1辅助生成的标注流程。

**📊 数据集**

使用BFMD数据集（19场全长比赛，12场单打、7场双打，共20.32小时，1,687回合、16,751击球），并在单打子集上进行实验。

**📈 对比分析**

与RGB‑only、Vid2Seq、InternVideo2以及大规模视觉‑语言模型进行对比；在BLEU‑4、METEOR、ROUGE‑L、CIDEr等指标上，提出的方法均显著优于基线，最优模型在BLEU‑4达到15.7、METEOR 23.7、ROUGE‑L 35.9、CIDEr 32.3。

**⚠️ 局限性**

局限性在于仅关注单个击球描述，未覆盖完整比赛的长期时序建模；对极其相似的前场击球仍可能出现误判；多模态特征依赖自动检测质量，若检测误差会影响最终结果。

---

## 420. Synchronous Signal Temporal Logic for Decidable Verification of Cyber-Physical Systems

**arXiv ID:** 2603.25531 | [PDF](https://arxiv.org/pdf/2603.25531v1)

**作者:** Partha Roop `[一作]` (University of Auckland), Logan Kenwright `[通讯]` (University of Auckland)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

本文提出了同步离散时序逻辑 SSTL，用于将连续时间的 STL 公式转换为可判定的离散形式，并通过 SPIN 模型检查器验证安全与活性性质。

**💡 创新点**

创新点包括：①首次构造同步（离散）STL；②在信号不变假设（SIH）下证明 SSTL 对 STL 的可判定且完整的抽象；③提供从 SSTL 到 LTL_P 的翻译规则，利用 SPIN 原生支持算术谓词实现自动验证。

**🔧 技术方法**

采用的技术：时域投影（[t] = ⌊t/Δt⌋）、SIH 条件、SSTL 语法与语义定义、翻译函数 τ、LTL_P 逻辑、Promela + C 嵌入模型、SPIN Büchi 自动机交叉检查。

**📊 数据集**

使用数据集：
- 33 节点心脏模型（心电信号），
- 交通灯控制器（有限状态机），
- 行人过街系统（整数队列与信号）。

**📈 对比分析**

方法比较：在上述三例中，SSTL 的验证均可在 SPIN 上完成；安全属性在 1–2 秒内验证完成；活性和有界响应属性在 6–65 秒内完成；相比传统 STL 监测或不完整的 SMT/MIQP 方法，SSTL 通过离散化实现判定性、无需插值或鲁棒化假设，验证精度与连续时间等价；但对大状态空间模型仍需显著内存（高达 9 GB）。

**⚠️ 局限性**

局限性：
- 需要满足 SIH，若信号在采样周期内变动，SSTL 与 STL 不等价；
- 需要手工编写翻译函数 τ；
- SPIN 需要将实数信号量化为有限精度整数，精度与状态空间大小成正比；
- 对大规模、强交互的系统，内存和时间成本仍较高。

---

## 421. RealRestorer: Towards Generalizable Real-World Image Restoration with Large-Scale Image Editing Models

**arXiv ID:** 2603.25502 | [PDF](https://arxiv.org/pdf/2603.25502v1)

**作者:** Yufeng Yang `[一作]` (Southern University of Science and Technology), Shifeng Chen `[通讯]` (Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

训练并发布了一款开源图像修复模型RealRestorer，能够一次性处理九种常见真实世界降质（模糊、雨、噪声、低光、摩尔纹、雾霾、压缩伪影、反射、炫光）并保持内容一致性。

**💡 创新点**

创新点包括：①面向真实世界的合成与真实降质大规模数据生成管线，显著缩小synthetic-to-real差距；②两阶段Progressively-Mixed训练策略，先用合成数据迁移知识，再用真实对齐数据细化并保持多任务泛化；③构建RealIR‑Bench无参考基准，并设计基于VLM的RS/LPS/FS评估框架。

**🔧 技术方法**

技术手段主要是：在Step1X‑Edit的Diffusion‑in‑Transformer（DiT）主干上进行fine‑tune，冻结Flux‑VAE与Qwen‑VL文本编码器；使用Qwen3‑VL‑8B‑Instruct评估降质级别；合成降质采用多种手工与网络增强策略；训练在8块NVIDIA H800 GPU上完成。

**📊 数据集**

使用了约1M的合成降质样本、约100K真实降质–清晰对齐样本；基准数据RealIR‑Bench（464张无参考真实降质图）以及FoundIR测试集（7类单一降质与多重降质）。

**📈 对比分析**

与多款闭源与开源大型图像编辑模型（Nano Banana Pro、GPT‑Image‑1.5、Seedream 4.5、LongCat‑Image‑Edit、Qwen‑Image‑Edit‑2511、FLUX.1‑Kontext‑dev、Step1X‑Edit）在九个任务上对比，RealRestorer在绝大多数任务中取得FS最高，整体排名首位；与闭源Nano Banana Pro仅差0.007点，优于其他开源模型。

**⚠️ 局限性**

局限性包括：①模型基于28步去噪，推理成本高；②在强语义/物理歧义场景（如镜面自拍）中难以准确区分真实内容与反射；③对极端严重降质仍可能无法完全恢复或保持物理一致性。

---

## 422. Interpretable PM2.5 Forecasting for Urban Air Quality: A Comparative Study of Operational Time-Series Models

**arXiv ID:** 2603.25495 | [PDF](https://arxiv.org/pdf/2603.25495v1)

**作者:** Moazzam Umer Gondal `[一作]`, Sultan Alamri `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文通过构建泄漏意识的预报工作流，对北京PM2.5小时级短期预测进行了可解释的轻量级时间序列模型评估，重点比较了SARIMAX、Facebook Prophet与NeuralProphet两种实际部署方案：每周walk-forward重拟合与冻结模型加在线残差校正；

**💡 创新点**

创新点在于提出在Perfect Prognosis下的完整预报流程，结合外生驱动变量与两种操作性更新机制，揭示轻量级可解释模型在真实运营中的可行性与优势；

**🔧 技术方法**

使用的技术包括时间序列预处理（排序、winsor化、标准化）、滤波特征选择（相关、MI、mRMR）、三类模型（SARIMAX、Prophet、NeuralProphet）以及在线残差校正（EWMA/卡尔曼滤波）和滚动周评估；

**📊 数据集**

采用了北京市2020年12月至2025年6月的小时级PM2.5与气象/污染物（NO、NO2、CO、SO2）数据，来源于OpenWeather与Open-Meteo公开API；

**📈 对比分析**

通过在10%测试区段进行滚动周评估，计算MAE与RMSE，结果显示Prophet在walk-forward和冻结+校正两方案下均保持最低误差（MAE≈37–38，RMSE≈50），SARIMAX在冻结+校正后误差进一步下降（MAE≈32，RMSE≈47），NeuralProphet误差最高且不受校正提升；

**⚠️ 局限性**

局限性包括：数据仅来自聚合API，缺乏传感器级细节；实验仅针对北京城市，可能不易迁移至其他城市；以及对突发剧烈气候或监管干预等异常事件的鲁棒性不足。

---

## 423. Not a fragment, but the whole: Map-based evaluation of data-driven Fire Danger Index models

**arXiv ID:** 2603.25469 | [PDF](https://arxiv.org/pdf/2603.25469v1)

**作者:** Shahbaz Alvi `[一作]` (CMCC Foundation - Euro-Mediterranean Center on Climate Change), Jose Maria Costa Saura `[通讯]` (CMCC Foundation - Euro-Mediterranean Center on Climate Change)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究提出一种基于全图推理和误报率评估的日FDR指数预测框架，并对多种CNN与ConvLSTM模型及其集成版进行实证比较

**💡 创新点**

创新点在于将模型评估从传统数据集指标转向真实运营场景的全图推理，并系统量化误报；同时证明ConvLSTM和模型集成能显著提升召回率与误报率；

**🔧 技术方法**

采用卷积神经网络（Basic CNN、Deeper CNN1/2）与ConvLSTM，利用PyTorch Lightning训练，采用F1、召回率、误报分布等多维度指标进行评估，并构建7模型集成平均预测

**📊 数据集**

使用高分辨率1 km × 1 km的火灾预测数据立方体（包含90个预测变量，包括气象、植被、人类因素等），覆盖希腊、巴尔干半岛及西土耳其，时间范围2009‑2021年

**📈 对比分析**

方法通过全图推理在2020‑2021年火季对日常召回率与误报分布进行比较，结果显示ConvLSTM在多数日子达到完美召回，误报率低于CNN；集成模型进一步降低误报、提升鲁棒性

**⚠️ 局限性**

局限性包括：仅针对特定地理区域与时间段；缺乏跨区域外部验证；误报仍存在于部分地区；模型训练与评估依赖高质量数据集，普适性待验证

---

## 424. CLAR: CIF-Localized Alignment for Retrieval-Augmented Speech LLM-Based Contextual ASR

**arXiv ID:** 2603.25460 | [PDF](https://arxiv.org/pdf/2603.25460v1)

**作者:** Shangkun Huang `[一作]` (Bairong, Inc.), Yunzhang Chen `[通讯]` (Bairong, Inc.)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究提出了一种基于CLAR的检索增强式上下文ASR系统，利用检索到的热点词作为提示注入Speech LLM，以提升命名实体和长尾词的识别准确性。

**💡 创新点**

创新点在于采用无时间戳的连续积分-触发（CIF）对齐学习单词级单调边界，并结合长度感知局部匹配与多粒度对比学习，实现高精度热点词检索。

**🔧 技术方法**

所使用的技术包括双编码器（Paraformer + Chinese‑RoBERTa）、CIF对齐机制、局部均值切片聚合、全局/局部对比损失以及量化约束，以及将检索结果作为自然语言提示注入Speech LLM。

**📊 数据集**

实验使用AISHELL‑1（训练Speech LLM）和AISHELL‑2（训练CLAR）数据集，在AISHELL‑1‑NE测试集上评估性能。

**📈 对比分析**

与SeACo‑Paraformer、CFL、PAC、GLCLAP等基线对比，CLAR在测试集上实现了0.92% CER、2.78% B‑WER，并在热点词检索上达到了97%+召回率，显示出显著的性能提升。

**⚠️ 局限性**

该方法的局限性包括仅在中文单语环境验证，缺乏多语种扩展；对检索结果误差敏感，错误热点词可能影响非热点词识别；以及仍需进一步验证在更大规模和更复杂场景下的鲁棒性。

---

## 425. Cross-Model Disagreement as a Label-Free Correctness Signal

**arXiv ID:** 2603.25450 | [PDF](https://arxiv.org/pdf/2603.25450v1)

**作者:** Matt Gorbett `[一作]`, Suman Jana `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

**🎯 论文内容**

该论文提供了COLM 2026会议论文提交的格式和要求。

**💡 创新点**

创新点在于详细说明了提交格式的具体要求，确保作者遵循一致的标准。

**🔧 技术方法**

使用了LaTeX格式和COLM样式文件。

**📊 数据集**

未提及具体数据集。

**📈 对比分析**

未提供与其他方法的比较或性能评估。

**⚠️ 局限性**

限制在于缺乏具体的研究内容和数据分析，主要集中在格式要求上。

---

## 426. Approximating Pareto Sum via Bounded Monotone Min-Plus Convolution

**arXiv ID:** 2603.25449 | [PDF](https://arxiv.org/pdf/2603.25449v1)

**作者:** Geri Gokaj `[一作]` (Karlsruhe Institute of Technology), Carina Truschel `[通讯]` (University of Konstanz)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了二维 Pareto 和的近似计算，提出可调节精度的加法近似算法。

**💡 创新点**

通过将近似 Pareto 和与有界单调 Min‑Plus 级联卷积等价，实现了条件最优的强子二次时间复杂度，并在实践中获得显著加速。

**🔧 技术方法**

采用规模化变换+最小-加卷积算法（如 CDXZ、Convex Pruning）与桶排序等实现。

**📊 数据集**

评估了三类数据集：合成整数范围、函数生成的近线性/曲线实例以及真实路网双指标路径规划实例。

**📈 对比分析**

与现有精确算法（Sort & Compare、Successive Sweep）以及其它卷积方法比较，平均运行时间缩短1–3个数量级，输出误差满足 Δ≤2t，输出尺寸显著下降。

**⚠️ 局限性**

主要限制是近似算法仍需参数调优，且对极端数据分布（如高凸度或非常大规模）性能不一，且目前对 O(n^1.5) 级别卷积的高效实现尚未完成。

---

## 427. CoDeTT: A Context-Aware Decision Benchmark for Turn-Taking Evaluation

**arXiv ID:** 2603.25434 | [PDF](https://arxiv.org/pdf/2603.25434v1)

**作者:** Huan Shen `[一作]` (Bairong Inc), Yunzhang Chen `[通讯]` (Bairong Inc)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `67630363-6be0-4f51-ab05-7198250671a5` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出CoDeTT基准，系统化评估多场景、细粒度的交替决策

**💡 创新点**

引入14类意图层级与语义不匹配率(SMR)揭示“运气猜测”，并构建统一两阶段评估协议

**🔧 技术方法**

采用大规模语音语言模型、语音合成、ASR校验、音频混合与多模型对比，利用SMR等指标进行评估

**📊 数据集**

构建300小时中英双语多轮对话数据，按14细粒度情景分布，融合真实与合成语音

**📈 对比分析**

对比专用控制器与全模态SLM，使用四大动作与14意图的准确率和SMR；结果显示控制器在Takeover高但Maintain/Dismiss低，SLM更均衡但仍存在高SMR，表明意图不匹配

**⚠️ 局限性**

评估协议对仅二元控制器缺乏细粒度意图评估；SMR依赖标注质量；数据生成可能带来人工痕迹，导致模型偏差

---

## 428. Missing-Aware Multimodal Fusion for Unified Microservice Incident Management

**arXiv ID:** 2603.25538 | [PDF](https://arxiv.org/pdf/2603.25538v1)

**作者:** Wenzhuo Qian `[一作]` (Zhejiang University), Shuiguang Deng `[通讯]` (Zhejiang University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了ARMOR框架，利用自监督学习在微服务异常检测、故障分拣与根因定位任务中处理缺失模态。

**💡 创新点**

首次将模态专属非对称编码器与缺失感知门控融合相结合，使用可学习占位符与动态偏置抑制补偿噪声，实现在无完整数据时仍能保持高诊断精度。

**🔧 技术方法**

采用自监督自回归重建、ModernTCN异构编码、注意力门控缺失融合、图注意网络、极值阈值判别及极端梯度提升分类等技术。

**📊 数据集**

在两个工业级基准集上验证：D1为电商模拟环境的46节点微服务，D2为银行核心系统的18节点，均含指标、日志与追踪三种模态。

**📈 对比分析**

与九个现有单/多任务基线相比，ARMOR在AD、FT、RCL上均取得最高分，且在各种缺失模态场景下表现最为稳健，推理时延比ART更低。

**⚠️ 局限性**

依赖完整历史数据训练，对持续缺失模态和子通道级缺失未做充分验证；图结构需定期更新，且实验环境规模有限。

---

## 429. Humans vs Vision-Language Models: A Unified Measure of Narrative Coherence

**arXiv ID:** 2603.25537 | [PDF](https://arxiv.org/pdf/2603.25537v1)

**作者:** Nikolai Ilinykh `[一作]` (University of Gothenburg), Sharid Loáiciga `[通讯]` (University of Gothenburg)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对视觉导向叙事中的叙事连贯性进行量化评估，并将人类写作与多模态语言模型（VLM）生成的故事进行对比。

**💡 创新点**

提出了包含核心推理、隐式话语关系多样性、主题切换、角色持久性以及多模态角色锚定的五项连贯性指标，并将其合成为叙事连贯性得分（NCS），揭示人类与模型在多维度连贯性上的系统性差异；同时研究了不同提示方式（短提示 vs 长提示）对模型连贯性的影响。

**🔧 技术方法**

使用核心推理链分析（Link-Append）、隐式话语关系分类（DeDisCo）、主题建模（BERTopic）、角色持久性计算、GROOViST与多模态角色连续性（MCC）等技术，构建完整的连贯性评估框架。

**📊 数据集**

主要数据集为 Visual Writing Prompts (VWP) 语料库，随机抽取 60 个视觉故事序列；此外在长提示实验中收集 54 条人工撰写的长篇故事；实验还使用了从网络抓取的多模态文本数据进行 perplexity 对比。

**📈 对比分析**

通过计算各指标的均值与标准差，并使用配对 t‑检验比较人类与模型得分；人类在核心推理、话语关系多样性、主题切换、角色持久性以及综合 NCS 上普遍优于模型；部分模型在多模态角色锚定上表现更佳。长提示对部分模型（如 Llama 4 Scout）提升显著，但整体人机差距未完全缩小。

**⚠️ 局限性**

局限性包括：仅评估 60 条故事序列，样本量有限；VWP 语料可能在模型预训练中已见过，导致 perplexity 结果偏倚；评估指标虽覆盖多维度，但仍可能遗漏某些连贯性维度；模型选择有限（仅五个 VLM），不代表所有现有模型；实验未探讨跨文化或多语言情况。

---

## 430. CHIRP dataset: towards long-term, individual-level, behavioral monitoring of bird populations in the wild

**arXiv ID:** 2603.25524 | [PDF](https://arxiv.org/pdf/2603.25524v1)

**作者:** Alex Hoi Hang Chan `[一作]` (University of Konstanz), Hemal Naik `[通讯]` (University of Konstanz)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究构建了CHIRP数据集，涵盖野生西伯利亚燕鸽的个体识别、行为识别、2D关键点估计、目标检测与实例分割，并提出基于颜色环的自动个体识别方法CORVID；

**💡 创新点**

创新点在于整合多种计算机视觉任务于单一长期野外数据集，并引入应用特定的评测指标（如进食率、共现率）以及利用颜色环信息的无监督个体识别框架；

**🔧 技术方法**

使用YOLOv8进行目标检测，Mask2Former进行颜色环实例分割，随机森林对颜色进行分类，BoTSORT实现跟踪，C3D实现动作识别，MegaDescriptor作为基线；

**📊 数据集**

数据集来源于2014-2022年在瑞典拉普兰收集的Siberian jay群体视频，包含1.6万段视频、1.1万帧检测、2.8k关键点、数千颜色环标注及12段应用测试视频；

**📈 对比分析**

在闭集与离散集拆分上，CORVID在仅考虑领土内或邻居限制时的Top‑1准确率高于Fine‑tuned MegaDescriptor（例如在“within territory”上0.69 vs 0.31），但当所有个体参与时性能下降；在动作识别中C3D达72%精度，关键点估计中ViTPose‑large PCK@10达97.8%；在应用级评测中CORVID相较于MegaDescriptor在个体进食率与共现率误差与相关性更优，仍低于人工基准；

**⚠️ 局限性**

局限性包括：数据仅来自单一物种与研究系统，行为类别偏向进食，颜色环方法对大规模种群可扩展性有限，且模型预测误差仍高于人工标注，需进一步提升检测、跟踪与识别精度。

---

## 431. Measuring What Matters -- or What's Convenient?: Robustness of LLM-Based Scoring Systems to Construct-Irrelevant Factors

**arXiv ID:** 2603.25674 | [PDF](https://arxiv.org/pdf/2603.25674v1)

**作者:** Cole Walsh `[一作]` (Acuity Insights Inc.), Rodica Ivan `[通讯]` (Acuity Insights Inc.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了基于大型语言模型（LLM）的自动评分系统在情境判断测试中对构造无关因素（如无意义文本、拼写错误、离题文本）的鲁棒性，并系统评估了这些扰动对评分的影响。

**💡 创新点**

创新点在于首次证明LLM评分系统对传统易被操纵的因素（如文本重复导致得分下降、写作复杂度不影响得分）表现出鲁棒性，并首次系统性比较不同离题情境对得分的惩罚效果。

**🔧 技术方法**

采用双架构：LLM-as-a-Judge进行特征提取（使用GPT‑5.X），并结合传统回归权重模型，对得分进行预测；通过Cohen’s d效应量评估不同实验条件下得分差异。

**📊 数据集**

使用30题开放式情境判断测试的26,571条答卷（910名学生）为基础，抽样545条（318名学生）用于实验；测试覆盖个人与职业技能四个维度。

**📈 对比分析**

方法为在基线答卷上加入不同构造无关扰动（无意义文本、拼写错误、阅读水平变化、离题重配），重新评分并计算均值、标准差与Cohen’s d；结果显示大多数扰动对得分影响极小（|d|<0.2），仅在拼写错误超过30%或离题配对时出现显著负面效应。

**⚠️ 局限性**

局限性包括仅评估单一LLM评分系统和GPT‑5.X模型，未考察低性能模型或其他LLM架构；实验规模有限，未深入研究公式化短语对评分的长期影响；未实现对异常答案的标记与警报功能。

---

## 432. DeepFAN, a transformer-based deep learning model for human-artificial intelligence collaborative assessment of incidental pulmonary nodules in CT scans: a multi-reader, multi-case trial

**arXiv ID:** 2603.25607 | [PDF](https://arxiv.org/pdf/2603.25607v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 433. Designing Any Imaging System from Natural Language: Agent-Constrained Composition over a Finite Primitive Basis

**arXiv ID:** 2603.25636 | [PDF](https://arxiv.org/pdf/2603.25636v1)

**作者:** Chengshuai Yang `[一作]` `[通讯]` (NextGen PlatformAI C Corp), Chengshuai Yang (NextGen PlatformAI C Corp)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

在本研究中，作者构建了一个完整的自动化管道，能够将一句自然语言描述直接转换为可验证的成像系统规范（spec.md），并通过三位代理（Plan、Judge、Execute）完成设计、验证和执行，实现了对173种不同成像模态的快速原型化；

**💡 创新点**

其创新点在于提出了基于11个有限原语的成像原语基础（FPB）与结构化规范格式，配合三步“Triad”门控验证与设计到实际误差五项分解定理，实现了从自然语言到可部署成像系统的端到端自动化和误差可控；

**🔧 技术方法**

技术实现包括基于FPB的有限原语库、Pydantic驱动的结构编译器、LLM驱动的Plan/Judge/Execute代理、三重门控（Recoverability、Carrier Budget、Operator Mismatch）以及跨层级层提升（Tier‑Lifting）和误差分解定理；

**📊 数据集**

实验验证使用了多种公开数据集，包括LoDoPaB-CT、M4Raw MRI、KAIST TSA CASSI、PICMUS 超声、SrTiO₃ 4D-STEM、CACTI 实时视频等六个真实测量数据集；

**📈 对比分析**

与专家手工实现的重建算法（如FISTA‑TV、HybridCascade++、GAP‑TV等）比较，代理生成的系统在6个真实模态上平均达到了98.1 ± 4.2 % 的质量比，并实现了约200–400倍的开发时间加速；

**⚠️ 局限性**

局限性包括：对LLM的依赖使得在不同模型上的鲁棒性未验证；仅有39个模态拥有量化重建基准，剩余约134个模态缺乏实测验证；误差上界在某些情况相对宽松（约3倍）；算法选择仍以传统模型为主，缺少针对性深度学习方法；层提升仅在两种跨层案例上得到验证。

---

## 434. Clinician Perspectives on Type 1 Diabetes Guidelines and Glucose Data Interpretation

**arXiv ID:** 2603.25631 | [PDF](https://arxiv.org/pdf/2603.25631v1)

**作者:** Mohammed Basheikh `[一作]` (University of Manchester), Simon Harper `[通讯]` (University of Manchester)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

通过在线问卷调查19名英国糖尿病相关专业人士，评估他们在执行T1DM临床指南及对患者使用血糖监测设备的认知。

**💡 创新点**

首次系统性比较临床医师对指南优先级的认知与对患者解读CGM/闪光监测数据的期待，揭示两者认知差距。

**🔧 技术方法**

采用Qualtrics在线问卷，使用多选、排序和5分李克特量表收集定量数据，并进行描述性统计。

**📊 数据集**

数据集为19名参与者的问卷答复，已公开托管于Zenodo（DOI: 10.5281/zenodo.18404347）。

**📈 对比分析**

研究未使用对照实验，而是通过描述性统计与已发表的实证研究结果对比，发现医师普遍低估患者解读困难，患者正确决策率低于22%。

**⚠️ 局限性**

样本量小、仅限英国医疗系统、受访者职业结构单一以及自报数据可能存在偏差，限制了结果的推广性。

---

## 435. Spatiotemporal System Forecasting with Irregular Time Steps via Masked Autoencoder

**arXiv ID:** 2603.25597 | [PDF](https://arxiv.org/pdf/2603.25597v1)

**作者:** Kewei Zhu `[一作]` (University College London), Sibo Cheng `[通讯]` (Institut Polytechnique de Paris)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计并实现了一种 Physics Spatiotemporal Masked Autoencoder（P-STMAE），用于高维动力学系统在非规则时间步的预测；

**💡 创新点**

创新点在于将卷积自编码器与掩码 Transformer 结合，在潜在空间完成无须预处理的缺失时间步恢复与未来预测，突破了传统 RNN 需要正则采样的局限；

**🔧 技术方法**

采用了卷积自编码器（CAE）提取空间特征、基于掩码的自动编码器（MAE/TiMAE）做时序建模、Transformer 的自注意力机制与位置编码，并结合物理空间与潜在空间的联合损失进行训练；

**📊 数据集**

在 PDEBench 提供的浅水方程与扩散-反应方程的模拟数据，以及 NOAA 海面温度（SST）真实观测数据上进行实验；

**📈 对比分析**

通过与 ConvRAE 与 ConvLSTM 两种基线模型在 MSE、SSIM、PSNR 三指标上的对比，P-STMAE 在所有数据集均取得最低 MSE、最高 SSIM 与 PSNR，并在缺失比例和采样稀疏性上表现更稳健；

**⚠️ 局限性**

主要限制包括 Transformer 的全局自注意力导致的计算与内存复杂度随序列长度增加而显著上升、对相对时间嵌入处理的局限，以及潜在空间压缩可能导致的重建细节损失，需要进一步研究稀疏/局部注意力和更高效的编码器方案。

---

## 436. Intelligent Navigation and Obstacle-Aware Fabrication for Mobile Additive Manufacturing Systems

**arXiv ID:** 2603.25688 | [PDF](https://arxiv.org/pdf/2603.25688v1)

**作者:** Yifei Li `[一作]` (Pennsylvania State University), Ilya Kovalenko `[通讯]` (Pennsylvania State University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c`

**🎯 论文内容**

本文研究了一种移动增材制造机器人（MAMbot）的导航-打印协同控制框架，实现在动态工厂环境中安全导航、障碍回避与打印质量同步控制。

**💡 创新点**

创新点在于将移动导航与材料沉积实现实时耦合，利用MPC轨迹规划与紧急迂回策略，能够在不中断任务的前提下处理障碍和地形扰动，显著提升打印精度。

**🔧 技术方法**

采用差分驱动移动平台、FDM打印机、LiDAR+IMU定位、ROS通信、MPC轨迹规划、耦合控制器以及暂停-恢复打印策略等技术。

**📊 数据集**

实验未使用公开数据集，而是在Gazebo仿真和实际KUKA KMR iiwa平台上，构建了含三处10 mm高度凸起的工厂环境，并测量实际打印样本尺寸。

**📈 对比分析**

通过与连续打印（Case A）和静止打印对比，采用三维尺寸误差评估，结果显示暂停-恢复策略将误差分别降低约89%–93%，使打印精度达到±0.1 mm容差范围。

**⚠️ 局限性**

局限性包括仅在封闭工厂环境验证，未考虑多机器人或人机协作；算法对速度与振动敏感，需要更复杂的动态障碍预测；实验规模有限，需进一步验证在更大规模工况下的鲁棒性。

---

## 437. Just Zoom In: Cross-View Geo-Localization via Autoregressive Zooming

**arXiv ID:** 2603.25686 | [PDF](https://arxiv.org/pdf/2603.25686v1)

**作者:** Yunus Talha Erzurumlu `[一作]` (Ohio State University), Alper Yilmaz `[通讯]` (Ohio State University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于自回归“先放大后细化”的交叉视角地理定位框架——Just Zoom In，利用从粗到细的卫星图像缩放序列来定位地面图像；

**💡 创新点**

核心创新在于把定位任务从单次对比学习检索转化为逐步缩放决策，避免了对大批量、难负样本的依赖，并显式利用地图的多尺度层次结构；

**🔧 技术方法**

技术方案包括：共享权重的DINOv2视觉编码器、Transformer解码器（带因果掩码与RoPE）、基于next‑action交叉熵的监督学习、部分冻结的编码器微调；

**📊 数据集**

使用了由Mapillary爬取的约30万张覆盖华盛顿特区的多视角街景图像，以及政府公开的高分辨率正射影像，构成了全新的多尺度交叉视角数据集；

**📈 对比分析**

在该数据集上与多种对比学习检索基线（如Sample4Geo、TransGeo等）对比，Just Zoom In在R@50m提升5.5个百分点、R@100m提升9.6个百分点，且不需要硬负样本挖掘、内存占用更低、训练时间更短；

**⚠️ 局限性**

局限性包括：仅在已知搜索区域（within‑AOI）内有效，对未知区域的泛化尚未验证；方法依赖固定的4×4分块层次，可能在非规则地图结构上受限；

---

## 438. Longitudinal Digital Phenotyping for Early Cognitive-Motor Screening

**arXiv ID:** 2603.25673 | [PDF](https://arxiv.org/pdf/2603.25673v1)

**作者:** Diego Jimenez-Oviedo `[一作]` (BiometricsAI, Universidad Autonoma de Madrid), Jaime Herreros-Rodriguez `[通讯]` (Hospital Universitario Infanta Leonor)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

利用儿童在平板设备上的交互数据，构建 AI 驱动的纵向模型，通过无监督学习（t‑SNE + K‑Means++）对儿童的认知‑运动表现进行聚类，并追踪个体在不同学年之间的群体转移，揭示认知发展轨迹。

**💡 创新点**

创新点在于：①将大规模、连续的平板交互日志转化为可量化的六项认知‑运动指标；②采用 t‑SNE 可视化降维后聚类，自动发现低、中、高三种发展档案；③构建转移矩阵对各档案的稳定性与变动进行时间序列分析，为早期筛查与个性化干预提供数据驱动的参考。

**🔧 技术方法**

主要技术包括：i) 交互数据预处理与 Q‑score 归一化；ii) t‑SNE 降维；iii) K‑Means++ 聚类；iv) 转移矩阵与稳定率计算；v) 统计描述（占比、稳定率、改进/衰退比例）。

**📊 数据集**

使用 ChildCIdb 数据集：940 名 18 个月至 8 岁儿童，来自西班牙马德里一所学校，在七个教育阶段内多次完成六项平板任务，记录时间、精度、误差等交互特征。

**📈 对比分析**

在每个学年分别进行聚类，采用肘部法选择最佳簇数。通过转移矩阵量化跨年稳定性、提升与下降比例：低性能群组稳定率 >90%，中性能群组稳定率 70‑90%，高性能群组波动最大。与传统单次评估或主观观察相比，该方法提供了连续、客观且可追溯的认知轨迹，可显著提升早期干预的精准度。

**⚠️ 局限性**

局限性包括：①数据来自单一地区学校，样本多样性有限；②仅依赖平板触控交互，缺乏社交、情感或神经影像等多模态信息；③无实际干预实验验证模型对干预效果的预测能力；④聚类方法依赖 t‑SNE 的局部结构，可能导致全局距离失真；⑤高性能群组波动可能受任务上限或参与度变化影响，解释上需要进一步研究。

---

## 439. Uncertainty-Guided Label Rebalancing for CPS Safety Monitoring

**arXiv ID:** 2603.25670 | [PDF](https://arxiv.org/pdf/2603.25670v1)

**作者:** John Ayotunde `[一作]` (University of Limerick), Lionel C. Briand `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种利用行为不确定性进行标签重平衡的安全监测框架（U-Balanc3），通过先训练不确定性预测器并将其输出用于概率性重新标记安全窗口为不安全，最终在重平衡后的数据上训练安全预测模型。

**💡 创新点**

创新点在于：①首次把行为不确定性作为重平衡信号用于时间序列数据的极端类别不平衡问题；②设计了基于分布统计特征的 GatedMLP 预测器高效捕捉不确定性；③采用不生成合成样本的 Label Rebalancing（改进 LNR）实现安全标签的概率性翻转，从而在不引入噪声的前提下提升少数类表示；④在安全监测任务中实现了显著性能提升。

**🔧 技术方法**

核心技术包括：分布统计特征提取（均值、标准差、最小值、最大值）；GatedMLP 结构的不确定性预测器；基于不确定性分数的 Flip‑Rate 计算与概率性标签翻转；Bi‑LSTM 安全预测器；超参数网格搜索与 30 次随机种子重复实验；统计显著性检验（Mann‑Whitney U、Vargha‑Delaney A12）。

**📊 数据集**

使用公开 UAV 1,498 次飞行数据集（约 54 小时），包含安全标签与自动生成的不确定性标签；数据按 8:1:1 划分为训练/验证/测试，窗口长度 25 时步，四个运动通道。

**📈 对比分析**

与 14 个基线（经典 ML、深度学习、集成方法以及多种重平衡技术）进行对比。U-Balanc3 在 F1 上比最优基线提升 14.3pp，召回率显著高于传统方法，推理延迟与模型参数量与主流深度模型相近；在各阈值与 ablation 研究中均证明了 GatedMLP 与不确定性重平衡策略的关键作用。

**⚠️ 局限性**

局限性包括：依赖规则生成的不确定性标签，可能在不同域或传感器设置下需重新标注；对阈值 τ 及概率翻转策略敏感；仅在 UAV 任务上验证，未检验跨 CPS（如水下无人机、自动驾驶车辆）或多模态数据的通用性；安全预测仍不是完美，召回与精确仍有提升空间。

---

## 440. Conchordal: Emergent Harmony via Direct Cognitive Coupling in a Psychoacoustic Landscape

**arXiv ID:** 2603.25637 | [PDF](https://arxiv.org/pdf/2603.25637v1)

**作者:** Koichi Takahashi `[一作]` `[通讯]` (Keio University), Koichi Takahashi (Keio University)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

构建了基于声学进化动力学的音频合成器 Conchordal，使用心理声学导向的连续共鸣景观来驱动自主音调代理的自组织、选择、遗传与同步。

**💡 创新点**

创新点在于提出 Direct Cognitive Coupling（DCC）原理，将心理声学指标（谐性与粗糙度）直接映射为生态/音乐评价景观，实现生态与美学的双重作用，并通过最小化遗传机制展示景观驱动的进化。

**🔧 技术方法**

采用了心理声学建模（谐性、粗糙度、二次线性组合共鸣量），基于梯度搜索与拥挤惩罚的自适应音调调整，代谢能量模型以及 Kuramoto 型相位耦合，结合 agent‑based 仿真实现。

**📊 数据集**

实验使用人工合成的单音（220 Hz 参考音与±2 oct 采样网格），无外部真实数据集，所有评估基于内部生成的共鸣景观及代理行为。

**📈 对比分析**

与随机漫步、无选择与无同步等消融实验对比，使用 leave‑one‑out 共鸣得分、存活时间、区间熵、PLV 等指标，结果表明景观驱动策略显著提升多音结构、选择效应与节奏同步。

**⚠️ 局限性**

局限包括缺乏独立听觉验证、对西方调性偏好依赖、仅限于固定参考音与谐波音调、未实现开放式进化、群体规模受限，且缺乏跨模态推广。

---

## 441. Anchored-Branched Steady-state WInd Flow Transformer (AB-SWIFT): a metamodel for 3D atmospheric flow in urban environments

**arXiv ID:** 2603.25635 | [PDF](https://arxiv.org/pdf/2603.25635v1)

**作者:** Armand de Villeroché `[一作]` (CEREA, ENPC, EDF R&D, Institut Polytechnique de Paris), Patrick Massin `[通讯]` (CEREA, ENPC, EDF R&D, Institut Polytechnique de Paris)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

训练并评估了一种基于Transformer的神经算子AB‑SWIFT，用于局部尺度大气流的快速模拟；

**💡 创新点**

创新点包括：①引入了锚点分支结构以分别处理地形、建筑和气象信息；②在几何编码中对地形和建筑进行分支编码并交叉注意；③在体积编码中加入气象剖面向量；④采用场量分支解码器实现各物理场的非线性预测；

**🔧 技术方法**

采用Transformer网络（多头注意力、M‑LP、RoPE位置编码）、锚点注意、supernode嵌入、正弦位置嵌入、物理块等深度学习技术；

**📊 数据集**

构建了包含228个CFD稳态大气流样本的新数据库，样本覆盖不同建筑几何、不同稳定性（unstable、neutral、stable），网格点数在5万到20万之间；

**📈 对比分析**

与AB‑UPT、GAOT、Transolver、BSMGN等基线模型在相同的6M参数规模下对比，AB‑SWIFT在所有指标（NMSE、L1、L2）均显著优于基线，误差低于10%；训练时间与显存均在可接受范围内；

**⚠️ 局限性**

局限性：使用RoPE位置编码导致模型难以泛化到训练之外的更大尺度域；未验证在更大、更逼真网格上的性能；未来需改进位置编码并构建更高保真数据集。

---

## 442. LanteRn: Latent Visual Structured Reasoning

**arXiv ID:** 2603.25629 | [PDF](https://arxiv.org/pdf/2603.25629v1)

**作者:** André G. Viveiros `[一作]` (Instituto de Telecomunicações), André Martins `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种名为 Lantern 的多模态推理框架，能够在语言生成过程中插入压缩的潜在视觉表示，实现“与图像并行思考”而非仅“描述图像”；

**💡 创新点**

创新点在于：①引入连续潜在视觉“思维”向量并在推理链中与文本交替使用；②通过监督学习将潜在向量与视觉编码器输出对齐；③再使用强化学习在任务层面上优化潜在向量的生成；

**🔧 技术方法**

技术包括：Qwen2.5‑VL 基础模型；扩展词表加入三种控制标记（<VIS_START>、<VIS_END>、<ANS>）；监督阶段使用 MSE 对齐潜在向量与视觉特征；强化学习阶段采用 GRPO + 潜在状态重放；

**📊 数据集**

数据集：Synthetic Visual‑CoT 用于监督潜在向量；V^⋆、Blink 子集用于评估视觉推理；VIRL‑39k 用于强化学习无区域监督阶段；

**📈 对比分析**

比较方法：与基线 Qwen2.5‑VL‑3B、文本仅推理版 LantErn‑NTP 以及强化学习版 NTP‑RL 进行对比；在 VisCoT、V^⋆、Blink 等视觉推理基准上，SFT 阶段已提升多项指标，RL 阶段进一步提升至或接近 7B 规模模型水平；

**⚠️ 局限性**

局限性：潜在向量块大小固定，缺乏自适应；潜在状态的解释性和可视化不足；训练依赖大量视觉任务数据，且对领域外视觉推理的泛化仍有待验证。

---

## 443. Visual or Textual: Effects of Explanation Format and Personal Characteristics on the Perception of Explanations in an Educational Recommender System

**arXiv ID:** 2603.25624 | [PDF](https://arxiv.org/pdf/2603.25624v1)

**作者:** Qurat Ul Ain `[一作]` (University of Duisburg-Essen), Astrid Rosenthal-von der Pütten `[通讯]` (RWTH Aachen University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在教育推荐系统中系统性比较视觉与文本解释，评估它们对用户的控制感、透明度、信任与满意度的影响，并探究用户个人特征（如人格、认知需求、决策风格等）如何调节这一效果。

**💡 创新点**

首次在同一教育推荐系统中同时对两种解释格式进行实证比较，并将多维个人特征纳入混合效应模型进行调节分析，最终提出针对教育推荐系统的可解释性设计准则。

**🔧 技术方法**

实现了交互式视觉解释与模板化文本解释，采用问卷量表（Likert）收集感知数据，使用Wilcoxon符号秩检验、混合效应模型和主题分析对定量与定性结果进行综合评估。

**📊 数据集**

收集了54名大学生（来自德国、巴基斯坦和伊朗）的实验数据，利用系统根据“未理解概念”与推荐项目（YouTube视频、维基百科文章）的相似度产生的内容进行推荐。

**📈 对比分析**

通过同主体设计让每位受试者先后体验视觉与文本两种格式，使用Wilcoxon检验比较控制感、透明度、信任与满意度；混合效应模型揭示大五人格、决策风格等个人特征的调节作用。结果显示视觉解释在信任与满意度上显著优于文本解释，其他维度差异不显著。

**⚠️ 局限性**

样本规模相对有限，且主要来自高校网络，缺乏多样性；实验仅关注感知指标，未检验对学习成效的直接影响；所用解释仅为简单交互和模板化文本，缺少更复杂的可解释算法或多模态方法。

---

## 444. The Kitchen Loop: User-Spec-Driven Development for a Self-Evolving Codebase

**arXiv ID:** 2603.25697 | [PDF](https://arxiv.org/pdf/2603.25697v1)

**作者:** Yannick Roy `[一作]` `[通讯]`, Yannick Roy

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了 Kitchen Loop 框架，利用 LLM 代理在已知规格表上以“As a User ×1000”的方式自动化代码生成、测试、评审和回归，形成闭环自演进软件生命周期。

**💡 创新点**

创新点在于将可枚举规格表、可击败的测试（ground‑truth 级验证）、多模型审查与漂移控制三大组件组合成统一的信任模型，并通过“不可击败测试”与“逆向 UAT 门”保证每一次合并都经过严格的真值检验。

**🔧 技术方法**

使用的技术包括多模型 LLM 代理（Codex、Gemini、Claude、CodeRabbit）、多轮辩论式审查（Discussion Manager）、抗信号可以类（anti‑signal canaries）、4 层测试金字塔、回归预言机与自动化暂停门，全部在 GitHub Actions、Anvil、Playwright 等现成工具上实现。

**📊 数据集**

主要数据集是内部代码库与运行时数据：在两套生产系统（DeFi 策略框架和信号情报平台）中分别使用 13+ 交易链、30+ 协议、77+ 信号类型及 62+ 演示策略，累计 1,094+ PR 与 13,000+ 代码/集成/端到端测试。

**📈 对比分析**

在 285+ 轮迭代中，系统实现 1,094+ 合并，零回归，质量门由 76–91% 提升至 100%，每个 PR 平均成本仅 0.38 美元，产出速度比人类高 30–60 倍，且无生产事故。

**⚠️ 局限性**

局限性包括：需要先行定义可枚举的规格表、为每个领域手工编写回归预言机、对“不可验证”任务（如主观 UX、绝对安全）不适用，LLM 对抗性测试成本高且可能产生 sycophancy，单线程执行和外部 API 限速可能成为瓶颈。

---

## 445. Persistent Robot World Models: Stabilizing Multi-Step Rollouts via Reinforcement Learning

**arXiv ID:** 2603.25685 | [PDF](https://arxiv.org/pdf/2603.25685v1)

**作者:** Jai Bardhan `[一作]` (Czech Technical University in Prague), Vladimir Petrik `[通讯]` (Czech Technical University in Prague)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种强化学习后训练框架（PersistWorld），通过让动作条件世界模型在自己的自回归生成轨迹上训练，显著提升长期视频生成质量。

**💡 创新点**

创新点在于：①将对比式 RL 目标迁移至 x0 预测的扩散模型；②设计可变长度前缀分支训练协议，使模型在不同阶段的错误累积中学习；③构造多视角、基于感知的奖励（LPIPS、SSIM、PSNR），并在候选组内进行归一化。

**🔧 技术方法**

使用扩散模型（Ctrl-World 的 EDM x0 预测骨干），LoRA 微调与动作编码器，强化学习对比式训练，视觉奖励与多视角处理。

**📊 数据集**

在大型机器人操纵数据集 DROID 上进行评估，利用多视角（外部两摄像头与腕部摄像头）视频。

**📈 对比分析**

与 Ctrl-World、WPE、IRASim 等基线相比，PersistWorld 在 LPIPS、SSIM、PSNR 等指标上均实现显著提升（外部相机 LPIPS 降低 14%，SSIM 提升 9.1%，腕部视角 PSNR 提升 1.59 dB），在 98% 的样本上优于基线，且人类偏好率达到 80%。

**⚠️ 局限性**

局限性包括：训练时需采样 16 个候选生成，计算开销高；奖励仅关注视觉相似度，未显式加入物理或几何约束，导致对物理真实性的进一步提升空间。

---

## 446. On the Formalization of Network Topology Matrices in HOL

**arXiv ID:** 2603.25682 | [PDF](https://arxiv.org/pdf/2603.25682v1)

**作者:** Kubra Aksoy `[一作]` (Concordia University), Sofiene Tahar `[通讯]`

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

使用 Isabelle/HOL 形式化并验证加权有向图的邻接矩阵、度矩阵、拉普拉斯矩阵及入出边矩阵，证明它们的基本性质及相互关系，并应用于 Kron 减法与电阻网络功耗的形式化分析。

**💡 创新点**

首次提供了完整、可复用的加权有向图拓扑矩阵形式化框架，系统性地证明了矩阵间的关系，并将其应用于实际电力系统的形式化验证，填补了传统仿真与形式化分析之间的空白。

**🔧 技术方法**

利用 Isabelle/HOL 交互式定理证明器、JNF 矩阵库、locale 模块化设计、Sledgehammer 自动化工具以及块矩阵运算实现形式化验证。

**📊 数据集**

在案例研究中使用 IEEE 5‑bus 测试系统和 IEEE RTS‑96 测试系统的数据来构造图模型并进行验证。

**📈 对比分析**

通过形式化证明而非数值实验进行比较，验证了所有定理的严谨性与一致性；在实际案例中证明了 Kron 减法保持拉普拉斯矩阵的 Laplacian 性质，以及电阻网络功耗公式的正确性。

**⚠️ 局限性**

局限性包括：仅处理实值矩阵，未扩展到复数域；证明工作量巨大，需手工编写较长代码；仅适用于有限图，未覆盖动态系统或大规模自动化推导。

---

## 447. Experimental Analysis of FreeRTOS Dependability through Targeted Fault Injection Campaigns

**arXiv ID:** 2603.25666 | [PDF](https://arxiv.org/pdf/2603.25666v1)

**作者:** Luca Mannella `[一作]` (Politecnico di Torino), Alessandro Savino `[通讯]` (Politecnico di Torino)

**关键词:** `9a43038e-f401-4fd9-9c05-65c0b8369d7e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

使用软件无侵入的KRONOS框架，在FreeRTOS核心数据结构上进行大规模的临时和永久故障注入实验。

**💡 创新点**

创新点在于构建了一个可在Linux/Windows主机上运行、无需硬件调试接口、能自动定位并注入多种内核对象（指针、列表、TCB字段等）的后传播故障注入框架，并通过编译期打补丁实现永久性故障。

**🔧 技术方法**

技术实现包括KRONOS软件框架、PCRE2补丁工具、FreeRTOS主机端移植、故障注入模块、日志收集与结果分类，以及二项式比例估计用于统计分析。

**📊 数据集**

使用TACLeBench数据集的五个基准（hash、FFT、三次方程求解、Huffman解码、调制编码）作为任务执行负载。

**📈 对比分析**

通过与金色运行的基线比较，实验显示约70%任务保持正常，约20%崩溃，临时与永久注入的影响相近；整个83,916次注入在平均77秒内完成，证明该框架具备可扩展性和可重复性。

**⚠️ 局限性**

局限性包括：仅关注已可见的内核级别故障，无法模拟低层物理传播；结果受任务调度和工作负载影响；仅验证了FreeRTOS，未评估跨RTOS或硬件级联效应。

---

## 448. Fast-dVLA: Accelerating Discrete Diffusion VLA to Real-Time Performance

**arXiv ID:** 2603.25661 | [PDF](https://arxiv.org/pdf/2603.25661v1)

**作者:** Wenxuan Song `[一作]` (Hong Kong University of Science and Technology (Guangzhou)), Haoang Li `[通讯]` (Hong Kong University of Science and Technology (Guangzhou))

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种基于块级扩散的加速框架（BlockDiff），将离散扩散 VLA（dVLA）的推理速度提升至实时水平，兼顾高成功率与实时控制。

**💡 创新点**

创新点在于：①利用块级自回归解码的内在特性，设计块级因果注意力结构实现 KV 缓存复用；②引入扩散强制（diffusion forcing）实现不同噪声级块的并行去噪；③采用非对称蒸馏快速从已训练的双向 dVLA 转化为块级扩散模型；④构造流水线式并行解码调度，兼顾速度与置信度控制。

**🔧 技术方法**

核心技术包括离散扩散模型、块级注意力与 KV 缓存机制、扩散强制去噪、非对称蒸馏训练、流水线并行推理与置信度阈值控制。

**📊 数据集**

实验数据集涵盖：CALVIN、LIBERO、SimplerEnv 三大模拟基准；以及在真实 bimanual AgileX 机器人上完成的三个任务（输送带抓取、蔬菜分拣、蔬菜检索）。

**📈 对比分析**

与现有 dVLA 加速方法、流匹配 VLA 以及 AR VLA 进行对比；在 CALVIN、LIBERO、SimplerEnv 上实现 2.8×–4.1× 的速度提升，同时成功率与现有方法持平甚至略有提升；在真实机器人上实现 30 Hz 的执行频率，显著优于传统方法。

**⚠️ 局限性**

局限性：需要针对动作维度挑选合适的块大小（多为动作维度的整数倍）并调节置信度阈值；目前仅验证了离散扩散输出的场景，尚未测试在更大规模或更复杂视觉输入下的鲁棒性；加速效果仍受模型规模与硬件并行度的影响。

---

## 449. SHAPR: Operationalising Human-AI Collaborative Research Through Structured Knowledge Generation

**arXiv ID:** 2603.25660 | [PDF](https://arxiv.org/pdf/2603.25660v1)

**作者:** Ka Ching Chan `[一作]` `[通讯]` (University of Southern Queensland), Ka Ching Chan (University of Southern Queensland)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

提出并操作化了 SHAPR 框架，结合人类决策与生成式 AI，形成可追溯、可重复的研究软件开发流程；

**💡 创新点**

创新点在于：①将 Action Design Research 通过 Explore–Build–Use–Evaluate–Learn 循环具体化为可执行工作流；②引入 Structured Knowledge Units (SKUs) 用于系统化捕获与复用研究洞察；③将框架设计为 AI 可解读、可执行，支持不同层级的 AI 介入；

**🔧 技术方法**

使用的技术包括大型语言模型（如 ChatGPT、Gemini）、通用开发环境、云存储与版本控制（Git）、结构化模板与文档化流程；

**📊 数据集**

论文未针对特定数据集，主要面向研究软件开发与知识管理；

**📈 对比分析**

未提供实验或性能对比，重点在框架设计与理论验证；

**⚠️ 局限性**

局限性：缺乏经验性评估与案例验证；对 AI 辅助程度与人类决策权的平衡尚未通过实证证明；框架实施需要研究者严格遵守文档化与可追溯性，实践成本可能较高。

---

## 450. A Mentalistic Interface for Probing Folk-Psychological Attribution to Non-Humanoid Robots

**arXiv ID:** 2603.25646 | [PDF](https://arxiv.org/pdf/2603.25646v1)

**作者:** Giulio Pisaneschi `[一作]` (Institute of Clinical Physiology National Research Council), Mario G. C. A. Cimino `[通讯]` (University of Pisa)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一套基于非人形机器人和大型语言模型的实验平台，用于系统性探究人类在不同解释框架下对机器的意向性归因

**💡 创新点**

创新点在于将意向性归因视为可操纵的语言变量，利用同一机器人行为在心理学、目的论和机械学三种语言框架下生成自述，从而剥离外观与信念归因的混杂

**🔧 技术方法**

采用Llama 3.2大语言模型进行意向、目的和机制层的对话生成，结合ROS2、Gazebo与Nav2进行机器人运动与状态管理

**📊 数据集**

使用自制的仿真视频和对应对话日志数据集，涵盖书店和小屋两种场景的任务执行与三种语言框架的自述

**📈 对比分析**

通过对比仅观看视频时与观看视频+LLM自述时受试者的意向性归因结果，发现语言框架能显著改变人类对机器人意图的判断；具体数值尚未在本文给出，但实验表明三种框架产生可区分的归因分布

**⚠️ 局限性**

局限在于实验仅在仿真环境中进行，缺乏真实人机交互数据；LLM可能产生幻觉或不一致的自述；未评估长期交互和多模态信息对归因的影响

---

## 451. RenoBench: A Citation Parsing Benchmark

**arXiv ID:** 2603.25640 | [PDF](https://arxiv.org/pdf/2603.25640v1)

**作者:** Parth Sarin `[一作]` (Stanford University), Dione Mentis `[通讯]` (DataCite)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并公开了RenoBench，一个基于真实PDF多语言多平台公开文献的引用解析基准；

**💡 创新点**

首次构建公共领域、跨出版生态系统且含多语言、不同出版类型的真实引用数据，并通过编辑距离匹配、严格质量检查及特征平衡采样实现高质量数据集；

**🔧 技术方法**

采用PDF→Markdown转换、字符串编辑距离匹配、正则与模糊匹配的质量检查、特征向量加权采样、LLM提示工程及GROBID API等技术；

**📊 数据集**

数据来源于SciELO、Redalyc、PKP、ORE四大开放平台的PDF与JATS XML，最终筛选得到10,000条符合质量标准的引用；

**📈 对比分析**

对比GROBID与多款LLM（Qwen、Gemma、Llama、Mistral、GPT‑OSS）在字段召回率上的表现，LLM（尤其是Qwen3-0.6B+LoRA）在大多数字段上优于传统工具，GROBID在生成有效XML方面最高；

**⚠️ 局限性**

主要限制：Precision评估受标注不完整影响，数据集规模有限且缺乏更细粒度标签，Prompt优化未见显著提升，跨语言性能仍有待加强。

---

## 452. Is Mathematical Problem-Solving Expertise in Large Language Models Associated with Assessment Performance?

**arXiv ID:** 2603.25633 | [PDF](https://arxiv.org/pdf/2603.25633v1)

**作者:** Liang Zhang `[一作]` (University of Michigan), Xinyi Jin `[通讯]` (High School Affiliated to Minzu University of China)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究大型语言模型（LLM）在数学问题求解与逐步评估（定位错误步骤）之间的关联性。

**💡 创新点**

发现同一模型在正确求解题目时，评估其解答中错误步骤的准确率显著提升，表明求解能力与评估能力存在正相关但不完全重叠；同时揭示评估任务比求解更困难，需要额外的步骤追踪和错误定位能力。

**🔧 技术方法**

使用OpenAI GPT‑4 与 GPT‑5 作为数学导师模型，并采用PROCESSBENCH（GSM8K 与 MATH 子集）的人工标注错误步骤数据，进行三次重复实验；通过准确率、F1、solve–assess 差值及卡方检验、Fisher精确检验评估关联强度。

**📊 数据集**

PROCESSBENCH 数据集：GSM8K 400 条和随机抽样的 MATH 400 条，共 800 条题目；每条题目包含原始问题、一步步解答轨迹和错误步骤标签。

**📈 对比分析**

对比方法：在同一模型下将求解成功与失败的题目划分，分别计算评估准确率；结果显示 GPT‑5 在求解与评估两项均优于 GPT‑4；在已出现错误的解答中评估准确率低于 10%，整体 F1 仅 9–17%；solve–assess 差距大（GSM8K 超过 80% 点，MATH 12–17% 点）。

**⚠️ 局限性**

局限性：仅评估两种 LLM 版本；数据集为人工标注的示例解答而非真实学生作业；评估指标采用严格的 exact‑match 误差步骤匹配，可能低估近似正确的评判；求解与评估使用相同题目，难以断定因果转移。

---

## 453. Demographic Fairness in Multimodal LLMs: A Benchmark of Gender and Ethnicity Bias in Face Verification

**arXiv ID:** 2603.25613 | [PDF](https://arxiv.org/pdf/2603.25613v1)

**作者:** Ünsal Öztürk `[一作]` (Idiap Research Institute), Sébastien Marcel `[通讯]` (Idiap Research Institute)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究针对多模态大型语言模型（MLLM）在面部识别验证中的公平性进行系统评估，比较了九个公开的MLLM在不同族裔与性别组上的表现；

**💡 创新点**

首次将四种基于FMR的公平性指标应用于MLLM的面部验证任务，并发现MLLM的偏差模式与传统嵌入式识别系统显著不同；

**🔧 技术方法**

采用视觉提问/回答框架：给定两张人脸图像和文本提示，让MLLM输出相似度分数并计算验证指标；

**📊 数据集**

使用IJB-C（带族裔与性别标签）和RFW（四族裔）两个验证基准；

**📈 对比分析**

在IJB-C上，FaceLLM-8B实现最低EER（5.13%），优于其他通用模型；在RFW上精度下降至29.46%；公平性方面，FaceLLM-8B在性别上表现最公平，族裔上虽相对较好但仍存在差距；

**⚠️ 局限性**

局限性包括：仅评估2B-8B规模模型；使用单一提示模板；未覆盖交叉属性（族裔×性别）及更大规模模型；验证速度与准确度仍不及专用嵌入式系统；

---

## 454. Kakeya Conjecture and Conditional Kolmogorov Complexity

**arXiv ID:** 2603.25611 | [PDF](https://arxiv.org/pdf/2603.25611v1)

**作者:** Nicholas G. Polson `[一作]` (University of Chicago), Daniel Zantedeschi `[通讯]` (University of South Florida)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了几何纤维对象下的条件压缩原理，并证明在正则可识别纤维分解下，Kolmogorov 复杂度可分解为纤维标签的复杂度与沿纤维的残余复杂度，从而给出 Kakeya 集的 Hausdorff 维度等于 n。

**💡 创新点**

创新点在于将算法信息理论与几何测度理论相结合，提出适应性纤维阻碍概念，构建了最小化压缩的游戏框架，并引入了黑塞比较等信息论工具来刻画侧信息的相对效用。

**🔧 技术方法**

使用了 Kolmogorov 复杂度及其链式规则、Lutz 的点到集合原理、有效 bi‑Lipschitz 可计算映射、可识别纤维分解等技术。

**📊 数据集**

未使用任何实验数据集，全部为理论证明。

**📈 对比分析**

方法仅通过理论分析验证，无实验对比；性能评估以理论复杂度界限和维度下界形式给出。

**⚠️ 局限性**

限制在于仅在可识别且正则的纤维分解下成立，对于高维下非正则或多重纤维分解仍未解决，是研究的核心难点。

---

## 455. Assessing Age Assurance Technologies: Effectiveness, Side-Effects, and Acceptance

**arXiv ID:** 2603.25695 | [PDF](https://arxiv.org/pdf/2603.25695v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 456. S2D2: Fast Decoding for Diffusion LLMs via Training-Free Self-Speculation

**arXiv ID:** 2603.25702 | [PDF](https://arxiv.org/pdf/2603.25702v1)

**作者:** Ligong Han `[一作]` (Red Hat AI Innovation), Akash Srivastava `[通讯]` (MIT-IBM Watson AI Lab)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了一种训练无关的自我推测解码框架，利用块扩散模型在块尺寸1下作为自回归验证器与常规块扩散解码器配合；

**💡 创新点**

创新点在于同一预训练块扩散模型同时充当稿稿和验证器，无需额外训练或改动架构，并通过轻量级路由策略动态决定何时进行验证；

**🔧 技术方法**

采用块扩散模型、block-size-1自回归验证、拒绝采样与重采样、轻量化路由策略，以及残差能量校正分析等技术；

**📊 数据集**

在三大块扩散家族模型（-8B-Chat、Fast-dLLM v2、2.1-Mini 等）上进行实验，使用 GSM8K、MBPP、HumanEval、IFEval 等基准数据集；

**📈 对比分析**

与标准块扩散、静态/动态置信度阈值以及 AR 基线比较，实验显示在保持或提升准确率的同时，速度提升可达 4.7×（相较于 AR）或 3.8×（相较于动态阈值），平均准确率提升约 4-5 个百分点；

**⚠️ 局限性**

局限性包括：验证仅覆盖首个连续掩码区块；路由策略需要手动调参；在极短或极长段落时验证成本仍显高；部分模型（如 Fast-dLLM v2）在块尺寸1不稳定。

---

## 457. Neural Network Conversion of Machine Learning Pipelines

**arXiv ID:** 2603.25699 | [PDF](https://arxiv.org/pdf/2603.25699v1)

**作者:** Man-Ling Sung `[一作]` (Raytheon BBN Technologies), Chinnu Pittapally `[通讯]` (Raytheon BBN Technologies)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了将随机森林教师模型的知识迁移到多层感知机（MLP）学生模型，并在100个OpenML任务上评估其性能。

**💡 创新点**

创新点在于将非神经网络管线（随机森林）转换为神经网络，尝试多种MLP配置，并探索基于元数据的自动最佳配置选择方法。

**🔧 技术方法**

使用了迁移学习、知识蒸馏、10折交叉验证、OpenML API、随机森林和多层感知机（MLP）等技术。

**📊 数据集**

使用了OpenML平台的100个任务，包含特征预处理、PCA以及随机森林分类器的完整管线。

**📈 对比分析**

通过10折交叉验证与原始随机森林教师模型对比，平均精度下降2.66%，中位数差距0.01%，在55%的任务上学生模型与教师相当或更好；单一最佳MLP配置平均仅差0.9%。

**⚠️ 局限性**

局限性包括：学生模型配置过多难以管理；基于元数据的自动选择效果差，元数据信息不足且样本量有限；在少数异常任务上性能不佳；未探讨管线其他组件的转换和端到端联合优化。

---

## 458. A Unified Memory Perspective for Probabilistic Trustworthy AI

**arXiv ID:** 2603.25692 | [PDF](https://arxiv.org/pdf/2603.25692v1)

**作者:** Xueji Zhao `[一作]` (University of Notre Dame), Ningyuan Cao `[通讯]` (University of Notre Dame)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

**🎯 论文内容**

提出了一个统一的概率数据访问框架，将确定性内存读写视为随机采样的零方差极限，揭示随机需求增加导致的熵墙瓶颈，并对传统 Von Neumann 体系与新兴概率 compute‑in‑memory（p‑CIM）架构进行了系统分析与对比，给出了评估标准和跨层设计路径。

**💡 创新点**

① 以数据访问为统一视角，把随机采样映射为确定性读写的推广；② 引入概率数据比例 α 并构建屋脊‑式模型，直观展示从内存瓶颈到熵瓶颈的转换；③ 提出内存级评价指标（统一操作、分布可编程、效率、鲁棒性、并行兼容）和跨层协同框架；④ 系统性讨论耦合与非耦合 p‑CIM 的权衡与性能边界。

**🔧 技术方法**

使用统计与系统级吞吐量模型、概率分布理论、随机数生成与噪声源分析、CMOS 与新兴存储器噪声物理、compute‑in‑memory 设计方案、以及对指令集与编译器层面的抽象。

**📊 数据集**

本文为视角论文，未使用任何实验数据或具体数据集，所有讨论均基于已有文献与理论推导。

**📈 对比分析**

通过理论模型与图表展示不同 α 值下系统吞吐量随熵产生率的变化，比较 Von Neumann 与耦合/非耦合 p‑CIM，在高 α 场景下指出传统架构因熵墙受限，而 p‑CIM 通过在内存中集成随机源实现更高吞吐与能效。

**⚠️ 局限性**

缺乏实测验证，模型对熵源速率与分布假设敏感；不同工作负载对统计质量的需求未量化；p‑CIM 在分布可编程性、硬件可靠性与软件抽象支持方面仍存在挑战；需要更完整的跨层评估与验证框架。

---

## 459. LEMMA: Laplacian pyramids for Efficient Marine SeMAntic Segmentation

**arXiv ID:** 2603.25689 | [PDF](https://arxiv.org/pdf/2603.25689v1)

**作者:** Ishaan Gakhar `[一作]` (Manipal Institute of Technology), Ujjwal Verma `[通讯]` (Manipal Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种轻量级语义分割模型LEMMA，专门针对海洋环境中的无人船（USV）和无人机（UAV）遥感图像进行障碍物与油污分割。

**💡 创新点**

创新点在于使用拉普拉斯金字塔提取边缘信息，构建三分支残差网络，避免深层特征图的高计算成本，同时实现71倍参数压缩、88.5% GFLOPs降低和84.65%推理速度提升。

**🔧 技术方法**

采用拉普拉斯金字塔多尺度分解、残差块、三分支特征融合以及焦点损失（Focal Loss）训练，构成端到端的轻量级分割体系。

**📊 数据集**

使用两个公开海洋分割数据集：MaSTr1325（USV摄像）和Oil Spill Drone（无人机油污）进行实验评估。

**📈 对比分析**

与30多种SOTA模型（如DeepLabv3、PSPNet、WaSR-T、BEMRF-Net等）在mIoU、参数量、GFLOPs和推理时间上进行对比。LEMMA在MaSTr1325上取得98.97% mIoU、1.07M参数、17.83 GFLOPs、7.3 ms；在Oil Spill上取得93.42% mIoU、1.01M参数、17.83 GFLOPs、7.3 ms，性能与更大模型持平而显著降低资源占用。

**⚠️ 局限性**

局限性包括对强反射、波纹、眩光等光照变化的鲁棒性不足，导致边缘信息模糊；数据集规模和多样性有限；当前模型采用固定金字塔层级和残差块配置，缺乏自适应机制。

---

## 460. On Neural Scaling Laws for Weather Emulation through Continual Training

**arXiv ID:** 2603.25687 | [PDF](https://arxiv.org/pdf/2603.25687v1)

**作者:** Shashank Subramanian `[一作]` (Lawrence Berkeley National Laboratory), Michael W. Mahoney `[通讯]` (Lawrence Berkeley National Laboratory)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

系统研究了气象预报任务中的神经网络规模化，使用最小化的Transformer架构和持续训练（常数学习率+周期性冷却），探索了不同计算预算下的计算最优（IsoFLOP）模型与数据规模关系，并对超大规模模型的性能极限进行了外推。

**💡 创新点**

创新点：
1) 采用无域特化、极简的Transformer作为基准，消除了架构与损失函数的混淆；
2) 用常数学习率加短周期冷却替代传统余弦调度，实现持续训练并显著降低实验成本；
3) 冷却阶段被重构为下游对齐过程，可通过不同损失（谱损失、AR rollout）快速提升多步预报质量；
4) 通过IsoFLOP曲线构建计算最优训练方案，并在超大计算预算下发现性能饱和点，揭示数据/分辨率瓶颈。

**🔧 技术方法**

技术：
- Shifted‑Window Transformer（窗口自注意力 + MLP）
- 恒定LR + 5% 终止冷却
- 连续训练与多轮epoch
- 空间并行 + 数据并行
- QK 归一化、RMSNorm
- AdamW + 混合精度训练
- MSE、谱损失、AR rollout 训练目标

**📊 数据集**

数据集：
ERA5 0.25° 分辨率气象数据，约 350k 时序样本（1979‑2022），选取 71 变量子集；训练集 1979‑2016，验证 2017，测试 2020；预测步长 6 小时，评估 10 天（240 小时）多步预报。

**📈 对比分析**

对比与性能：
- 与余弦 LR 的传统训练相比，常数+冷却训练在同等 FLOP 下损失更低；
- 与 NWP/HRES（HRES）对比，计算最优模型（如 204M 参数）在 6E+19 FLOP 下 RMSE 超越 HRES、与 GraphCast 竞争；
- 随着计算预算提升，多步预报的 RMSE 持续下降，直至最高预算出现饱和；
- 冷却阶段重构为谱损失或 AR 训练时，可显著改善对应指标。

**⚠️ 局限性**

局限性：
- 超大模型需要多轮 epoch，易出现过拟合，导致性能饱和；
- 数据量和空间分辨率是瓶颈，无法充分利用亿级参数；
- 冷却比例 5% 的选择仍基于经验；
- 仅在 ERA5 0.25° 低分辨率数据上验证，缺乏跨域或高分辨率验证。

---

## 461. Self-Improvement of Large Language Models: A Technical Overview and Future Outlook

**arXiv ID:** 2603.25681 | [PDF](https://arxiv.org/pdf/2603.25681v1)

**作者:** Haoyan Yang `[一作]` (Zesearch NLP Lab, Stony Brook University), Jiawei Zhou `[通讯]` (Zesearch NLP Lab, Stony Brook University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

系统性梳理并提出统一框架，涵盖数据获取、选择、模型优化与推理改进等闭环自我提升流程。

**💡 创新点**

首次将自我提升视为四阶段闭环，并系统归纳现有技术，提出自治评估与多模态数据获取三路策略。

**🔧 技术方法**

基于LLM自生成数据、强化学习、主动评估、推理优化等技术，构建统一的生成–奖励–优化（GRO）流程。

**📊 数据集**

未引入新数据集，综述了多种公开数据集如Common Crawl、GitHub、arXiv、WebGPT等。

**📈 对比分析**

论文未进行原始实验，对比分析聚焦于已有工作表现，未给出统一指标。

**⚠️ 局限性**

缺乏系统实现与大规模验证，安全与偏见控制难题，模型自主决策的可解释性与监管挑战。

---

## 462. Can Users Specify Driving Speed? Bench2Drive-Speed: Benchmark and Baselines for Desired-Speed Conditioned Autonomous Driving

**arXiv ID:** 2603.25672 | [PDF](https://arxiv.org/pdf/2603.25672v1)

**作者:** Yuqian Shao `[一作]` (Shanghai Jiao Tong University), Junchi Yan `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Bench2Drive-Speed 基准，支持闭环、端到端自动驾驶模型的目标速度与超车/跟随指令控制，并提供相应的数据集与评估指标。

**💡 创新点**

创新点包括：①将目标速度与超车指令作为显式控制输入；②设计 Speed‑Adherence Score 与 Overtake Score 两项量化指标；③引入“虚拟目标速度”重新标注策略，避免对专家控制器内部参数的依赖；④构建多难度、可变目标速度的 CARLA 场景集。

**🔧 技术方法**

使用的技术主要是基于 TCP（Trajectory‑Control Path）网络的端到端学习框架；数据预处理将视觉、状态与目标速度/指令拼接后送入轨迹与控制分支；评估采用闭环仿真并结合 Bench2Drive 的安全、舒适度等传统指标。

**📊 数据集**

数据集为 2100 条 CARLA 仿真轨迹，按难度（易/中/难）与超车/跟随两种指令分布，包含专家演示标注和虚拟目标速度标注两种策略。

**📈 对比分析**

在 Bench2Drive‑Speed 的 48 条评测路线上，采用虚拟目标速度训练的模型与专家演示训练的模型在 Speed‑Adherence Score 上相近；在 Overtake Score 上仍有较大差距，尤其在难度较高的场景；与原始 TCP 相比，速度控制模型保持或略提升了 Driving Score 与 Success Rate，舒适度略降，效率略升。

**⚠️ 局限性**

局限性包括：①超车指令的执行仍不稳定，尤其在高难度场景易导致安全违规；②虚拟目标速度标注在长时延伸时噪声增大；③实验全部基于 CARLA 仿真，缺乏真实道路数据验证；④对目标速度与安全边界的权衡机制尚不完善。

---

## 463. EPAR: Electromagnetic Pathways to Architectural Reliability in Quantum Processors

**arXiv ID:** 2603.25671 | [PDF](https://arxiv.org/pdf/2603.25671v1)

**作者:** Navnil Choudhury `[一作]` (Rensselaer Polytechnic Institute), Kanad Basu `[通讯]` (Rensselaer Polytechnic Institute)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出 EPAR 框架，将超导量子处理器的物理布局转化为有效耦合拓扑，并基于此评估量子电路的可靠性。

**💡 创新点**

创新点在于同时结合 EM 仿真、寄生电容提取、哈密顿量重构与两项新度量（LTD 与 SI），实现布局驱动的拓扑失真与脉冲敏感性可视化。

**🔧 技术方法**

采用 Qiskit Metal 进行布局生成、ElmerSolver 进行有限元电容提取、Qiskit Dynamics 进行多模哈密顿量时域模拟，并用自定义算法计算 LTD 与 SI。

**📊 数据集**

使用两组基准数据集：九个 Q_i–C–Q_i+1 版块布局与一个五量子比特线性链，配合多种交叉共振与调制脉冲。

**📈 对比分析**

通过与传统校准的两量子比特误差率对比，发现 EPAR 能揭示结构弱点与脉冲相关的失真，识别出误差率相同但可靠性差距高达 10 倍的布局；在不同脉冲时域下，LTD 与 SI 能准确划分鲁棒与脆弱设计。

**⚠️ 局限性**

局限在于依赖大规模 EM 仿真，计算成本随芯片规模呈指数级增长；哈密顿量近似与布局静态假设可能忽略温度、噪声或工艺变异；目前仅适用于 Transmon/可调耦合器模型，需进一步验证在更大规模或不同量子比特类型的适用性。

---

## 464. Beyond Via: Analysis and Estimation of the Impact of Large Language Models in Academic Papers

**arXiv ID:** 2603.25638 | [PDF](https://arxiv.org/pdf/2603.25638v1)

**作者:** Mingmeng Geng `[一作]` (École Normale Supérieure), Thierry Poibeau `[通讯]` (École Normale Supérieure)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过对arXiv论文标题与摘要的词频变化进行量化分析，评估大型语言模型（LLM）对学术写作的影响，并探讨不同LLM之间的词汇偏好差异。

**💡 创新点**

提出基于线性趋势回归与词频比率的直接可解释估计方法，以及利用词频系数变化识别LLM影响；同时揭示LLM词汇偏好随模型迭代的演化。

**🔧 技术方法**

线性回归、词频比率计算、ROUGE/BERTScore文本相似度评估、BERT/GPT‑2/T5/LLM2Vec分类器、SLSQP优化等。

**📊 数据集**

arXiv公开数据集（约290万篇论文，含标题/摘要），Google Books Ngram前1万高频词，NLTK停用词表。

**📈 对比分析**

对比方法包括：词频趋势预测误差、词频比率与词汇系数差异、相似度指标（ROUGE‑1/2/L、BERTScore）、分类准确率。结果显示词频估计能反映LLM影响；分类在二分类下准确率达80–90%，但多分类准确率仅60%以下；相似度指标显示新LLM与原文的相似度提升，但语义相似度并未同步提升。

**⚠️ 局限性**

受限于模型版本覆盖不足、提示样本有限、数据中人工修改导致误判、词频共线性及高频词差异不显著、以及LLM持续迭代导致估计不稳定。

---

## 465. Accurate Surface and Reflectance Modelling from 3D Radar Data with Neural Radiance Fields

**arXiv ID:** 2603.25623 | [PDF](https://arxiv.org/pdf/2603.25623v1)

**作者:** Judith Treffler `[一作]` (Örebro University), Martin Magnusson `[通讯]` (Örebro University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6514db3d-8de6-452c-91b7-acdb31787cc4` `57a58b01-81b4-4d75-a45c-2e891f272b50` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种基于神经隐式表示的3QFPI方法，用雷达点云同时重建3D几何和视角依赖的雷达强度；

**💡 创新点**

创新点在于将三维雷达的稀疏噪声数据映射到连续SDF与视角依赖强度网络上，并采用内存高效的三维四叉树+傅里叶特征编码，实现对低能见度环境下的平滑、精确重建；

**🔧 技术方法**

技术包括3QFP混合特征编码、Neural Radiance Fields（NeuS2）架构、傅里叶特征与球谐编码、SDF+Intensity双网络联合优化、三维稀疏采样与交叉熵+L1损失；

**📊 数据集**

使用三个公开雷达数据集：Hugin A3、Oculii Eagle（两个序列）以及自制半圆角反射测试集；

**📈 对比分析**

与α‑shapes、BPA、Poisson、VDBFusion、SHINE‑Mapping等传统与神经隐式基准相比，3QFPI在准确率、平滑度（Gamma分布指标）和视角依赖强度恢复（MAE≈2–5）方面表现最佳，尤其在稀疏输入下保持较高完成率；

**⚠️ 局限性**

主要限制包括对雷达数据质量高度依赖，噪声与多路径反射未被完全建模，评估指标仍受限于稀疏性与视角覆盖不足，且在某些场景下仍产生少量误差和外点。

---

## 466. UNIC: Neural Garment Deformation Field for Real-time Clothed Character Animation

**arXiv ID:** 2603.25580 | [PDF](https://arxiv.org/pdf/2603.25580v1)

**作者:** Chengfeng Zhao `[一作]` (HKUST), Yuan Liu `[通讯]` (HKUST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4de8e9d8-757b-475f-9627-18a445e50202` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

开发了一种基于实例特定神经变形场的实时衣物动画方法，可在任意服装拓扑下生成逼真衣物变形。

**💡 创新点**

创新点在于使用基于MLP的连续神经变形场与离散化的运动编码器相结合，既保留空间平滑性又显著提升对复杂服装的适应性，并加入可插拔的穿插处理模块。

**🔧 技术方法**

采用了多层感知机（MLP）神经场、Gumbel-Softmax离散运动编码器、交叉碰撞检测与拖拽处理、以及运动匹配技术实现实时推理。

**📊 数据集**

使用SMPL骨架、CMU MoCap、VTO-数据集、VirtualBones-数据集以及作者构建的高复杂度服装数据集进行训练与评估。

**📈 对比分析**

在VTO和VirtualBones数据集上与TailorNet、SNUG、HOOD、VirtualBones等基线比较，RMSE和Hausdorff误差均比最先进方法低5–60%，在RTX 3090上实现每秒30+帧，比专业软件加速版快1.5–4倍。

**⚠️ 局限性**

主要限制是方法仅针对单个服装实例训练，无法跨服装迁移；复杂服装训练耗时较长；在极端运动或极细节服装上仍可能出现细小穿插或失真。

---

## 467. Cooperative Deep Reinforcement Learning for Fair RIS Allocation

**arXiv ID:** 2603.25572 | [PDF](https://arxiv.org/pdf/2603.25572v1)

**作者:** Martin Mark Zan `[一作]` (TU Wien), Stefan Schwarz `[通讯]` (TU Wien)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

在多基站、用户分布不均的场景下，提出基于同时升价拍卖的可重构智能表面（RIS）共享资源分配框架，并通过协作式多智能体强化学习实现公平性驱动的竞价策略

**💡 创新点**

将公平性指标融入到每个基站的观测中，使得各基站在不直接通信的情况下实现隐式协作；同时结合拍卖机制与强化学习，首次将拍卖与RL相结合以实现资源的公平高效分配

**🔧 技术方法**

同时升价拍卖机制、深度强化学习（PPO）、宏观信道估计与公平权重计算、模拟环境Gymnasium/PettingZoo/Stable‑Baselines3

**📊 数据集**

通过仿真生成的随机网络实例（用户、RIS位置、信道大尺度参数），不使用公开数据集，所有实验均在自研仿真平台完成

**📈 对比分析**

与传统的无公平约束拍卖和基于启发式分配的基线进行对比；结果显示在保持系统总吞吐量下降≤7% 的同时，最差用户速率提升≈34%，Atkinson不平等指数随公平参数增大而显著下降

**⚠️ 局限性**

受限于仅考虑二维两基站场景，仿真规模较小；缺乏对更大规模网络、动态用户和不同拍卖格式的验证；公平权重依赖于中心化信息，若基站自报不准则会引入策略失真

---

## 468. Concentration And Distribution of Container Flows In Mauritania's Maritime System (2019-2022)

**arXiv ID:** 2603.25678 | [PDF](https://arxiv.org/pdf/2603.25678v1)

**作者:** Mohamed Bouka `[一作]` (University of Nouakchott), Moulaye Abdel Kader Ould Moulaye Ismail `[通讯]` (University of Nouakchott)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了毛里塔尼亚2019-2022年集装箱航运系统的结构特征，重点分析路线集中度、港口依赖性、行业专业化、进出口结构不对称性以及时间稳定性。

**💡 创新点**

提出基于FFE权重的概率分布框架，统一量化路线、港口和行业的集中度与差异性，首次揭示小型贸易经济体中港口与行业结构的极端不对称性，并提供时间演化评估。

**🔧 技术方法**

采用统计学集中度指标（HHI、CR、Gini、熵）、Jensen‑Shannon距离、Spearman/Kendall相关性及年度HHI跟踪等方法，使用Python实现计算与可视化。

**📊 数据集**

利用毛里塔尼亚海关申报的集装箱装运记录（FFE单位），包含2019-2022年共105,686条船级、路由、起/到港、行业等信息。

**📈 对比分析**

通过多维度指标对进出口差异与时间变化进行比较，结果显示路线集中度高、港口分布极度不对称、出口行业高度集中，整体结构相对稳定；JSD值在0.2-0.4之间，证明框架能捕捉细微差异。

**⚠️ 局限性**

仅为分布性分析，未揭示因果机制；行业“未分类”类别混杂导致专业化评估受限；路由与港口的定义固定，未考虑运营效率或服务可用性等实际指标。

---

## 469. Advances in Exact and Approximate Group Closeness Centrality Maximization

**arXiv ID:** 2603.25642 | [PDF](https://arxiv.org/pdf/2603.25642v1)

**作者:** Christian Schulz `[一作]` (Heidelberg University), Henning Woydt `[通讯]` (Heidelberg University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了改进的 Grover 算法，用于求解组接近度中心性最大化（GCCM）问题。

**💡 创新点**

创新点包括两种数据约简技术：①通过估算每个顶点的距离上限 d(v) 降低 ILP 的迭代次数；②利用可被“吸收”顶点的约简规则进一步减少 ILP 变量。

**🔧 技术方法**

技术主要是基于改进的 ILP 表达式、迭代 ILP 方法、支配顶点（dominated vertices）与吸收顶点（absorbed vertices）的判定，以及对 GS‑LS‑C 近似算法的改写。

**📊 数据集**

实验使用 32 个来自 Konect 与 Network Data Repository 的真实网络，规模从约 50 顶点到 40,000+ 顶点不等，涵盖社交、协作、生物、经济、信息、脑网络等类型。

**📈 对比分析**

与 ILPind、DVind、SubModST 等现有方法对比，Grover 在 646 个实例中解决 514 个，速度平均提升 3.65 倍，单实例最大提升 22.37 倍；在近似场景中，改写 GS‑LS‑C 的运行时间平均提升 1.4 倍，且保持 1/5 近似保证。

**⚠️ 局限性**

局限在于对 k 较大的情况仍需要 10 分钟以内的时间限制，且在非常小的图上由于预处理开销可能略逊；此外改进主要针对无权无向图，扩展到加权或有向图仍需进一步研究。

---

## 470. The Geometry of Efficient Nonconvex Sampling

**arXiv ID:** 2603.25622 | [PDF](https://arxiv.org/pdf/2603.25622v1)

**作者:** Santosh S. Vempala `[一作]` (Georgia Institute of Technology), Andre Wibisono `[通讯]` (Yale University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `de8d30ba-c289-43a5-b4ec-7b80df73aea2`

**🎯 论文内容**

提出了一种名为 In-and-Out 的迭代采样算法，能够在满足等周不等式（Poincaré 常数有限）和体积增长条件的任意紧致非凸体内，从温暖起点高效采样均匀分布。

**💡 创新点**

创新点在于将已有的凸体和星形体采样理论统一扩展到更广泛的非凸集合；首次证明仅靠等周性质和体积增长就能保证多项式时间采样；并通过对 In-and-Out 的拒绝采样实现给出了显式的复杂度上界。

**🔧 技术方法**

主要技术包括：
- 近似原始（Proximal Sampler）与 Langevin 动力学的离散化;
- 在体积增长条件下对拒绝采样失败概率的精确上界;
- 通过 Renyi 散度的收敛分析得到误差保证；
- 对冲击步长 h 的精细取值与阈值 N 的设置。

**📊 数据集**

本工作为理论分析，没有使用具体数据集；所有结果均为数学证明，适用于任何满足条件的集合。

**📈 对比分析**

与现有仅适用于凸体或星形体的采样方法相比，In-and-Out 在相同条件下实现了 
O(q α β² M n³ log⁴(1/ε)) 次成员查询和 O(n) 计算量，
即在非凸情形下仍保持多项式时间，且误差以 Rényi 散度控制，隐含 KL 与 TV 的下界。

**⚠️ 局限性**

局限性包括：
- 需要先给定一个温暖起点（warm start），而生成温暖起点的高效方法仍未给出；
- 依赖于成员查询接口，对实际实现可能有实现成本；
- 对 Poincaré 常数与体积增长常数的估计在实践中可能较难获得；
- 目前仅证明了理论复杂度，缺乏实验验证。

---

## 471. PICon: A Multi-Turn Interrogation Framework for Evaluating Persona Agent Consistency

**arXiv ID:** 2603.25620 | [PDF](https://arxiv.org/pdf/2603.25620v1)

**作者:** Minseo Kim `[一作]` (KAIST), Edward Choi `[通讯]` (KAIST)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于质询的多轮评估框架，测量大型语言模型驱动的人格代理在对话中的一致性。

**💡 创新点**

创新点在于：①将内部一致性、外部事实一致性和再测一致性三维度统一评估；②采用逻辑链式追问与实时网页检索进行事实核查，显著提升对矛盾的检测力度。

**🔧 技术方法**

使用多代理体系（Questioner、Entity‑&‑Claim Extractor、Evaluator）配合 GPT‑5、Gemini‑3 等 LLM 进行问答链与实体抽取，并通过 NLI/LLM‑Judge 进行内部冲突检测。

**📊 数据集**

数据集包括：公开的世界价值调查问卷作为基准提问，真实人类参与者（63人）与七组现有人格代理（Character.ai、OpenCharacter、Consistent LLM 等）进行对照实验。

**📈 对比分析**

方法通过三维度（IC、EC、RC）给出评分，并以人类基准为对照；结果表明所有代理均低于人类基准，且各代理在不同维度上表现差异显著，未出现任何代理在所有维度上均优于人类的情况。

**⚠️ 局限性**

limitations：仅评估可检索事实的陈述，忽略语言风格、偏好等主观维度；依赖公开网页检索，无法验证非公开事实；问卷样本可能缺乏代表性；对完全回避回答的处理尚有限。

---

## 472. Social Hippocampus Memory Learning

**arXiv ID:** 2603.25614 | [PDF](https://arxiv.org/pdf/2603.25614v1)

**作者:** Liping Yi `[一作]` (Tianjin University), Qinghua Hu `[通讯]` (Tianjin University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c84dae5d-5273-4348-85a7-b44cb586b4df` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于记忆的社交机器学习框架 SoHip，利用共享长期记忆而非模型参数实现异构联邦学习协作。

**💡 创新点**

创新点在于通过记忆抽象、海马式长期记忆整合以及个体-集体记忆融合三步机制，使多模型客户端在不泄露原始数据和模型参数的情况下高效共享知识。

**🔧 技术方法**

采用轻量化编码器/解码器、门控机制、联邦服务器聚合以及 SGD 训练，并在理论上给出收敛与隐私保证。

**📊 数据集**

在 CIFAR‑100 与 Tiny‑ImageNet 两大图像分类基准上进行实验，使用标签偏斜划分产生极端非 IID 环境。

**📈 对比分析**

与 7 个代表性基线（FedProto、FedSSA、FedRAL、FedKD、FedMRL、pFedES 及单机学习）比较，SoHip 在所有设置下均取得最高准确率，提升幅度高达 8.78%。

**⚠️ 局限性**

仍需在更大规模、多任务场景验证；记忆维度对效果敏感；对极端系统资源异构或动态参与度的鲁棒性待进一步探讨。

---

## 473. Quantum Circuit Repair by Gate Prioritisation

**arXiv ID:** 2603.25587 | [PDF](https://arxiv.org/pdf/2603.25587v1)

**作者:** Eñaut Mendiluze Usandizaga `[一作]` (Simula Research Laboratory), Shaukat Ali `[通讯]` (Simula Research Laboratory)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了一种基于门优先级的量子电路自动修复方法，自动定位最疑似错误门并逐步应用补丁；

**💡 创新点**

创新点在于将缺陷定位与补丁优先级结合，使用门的可疑性得分来显著缩小搜索空间，提升修复效率；

**🔧 技术方法**

技术主要包括量子门修补、Hellinger距离评估、COBYLA参数优化、Qiskit/Qulacs仿真等；

**📊 数据集**

实验数据集包含40个量子电路，其中4个来自真实缺陷库Bugs4Q，36个为人工生成的变异量子算法（来自MQT Bench 12个算法）；

**📈 对比分析**

与随机搜索和基于代数模型的修复方法比较，实验显示该方法在40个缺陷中达70%完整修复率，在真实缺陷中与基线相当（50%），在人工缺陷中显著优于基线（10倍提升）；

**⚠️ 局限性**

主要局限是一次只能修复一个门，受限于测试套件覆盖范围，且对多重缺陷和更大规模电路的适应性尚未充分验证。

---

## 474. Towards Generalizable Robotic Data Flywheel: High-Dimensional Factorization and Composition

**arXiv ID:** 2603.25583 | [PDF](https://arxiv.org/pdf/2603.25583v1)

**作者:** Yuyang Xiao `[一作]` (ByteDance Seed), Yuxiao Liu `[通讯]` (ByteDance Seed)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4`

**🎯 论文内容**

提出一种基于因子分解和组合迭代学习（F-ACIL）的机器人操纵策略数据采集与训练方法

**💡 创新点**

创新点：将状态空间拆分为对象（Object）、动作（Action）、环境（Environment）三因子；采用逐因子扩展与组合泛化策略；通过稀疏高斯混合分布实现高效的数据飞轮，显著降低所需演示样本量

**🔧 技术方法**

技术：因子化状态表示、因子级别的迭代子集搜索、组合泛化（orbit closure）与数据飞轮；使用预训练的VLA backbone（保持不变）进行微调；算法框架包含迭代收集、评估、增量扩展三个阶段

**📊 数据集**

数据集：在ByteMini真实机器人平台上收集的 Pick‑and‑Place 与 Open‑and‑Close 任务演示；对对象纹理、几何、尺寸；动作平面位置与旋转；环境宏观光照、阴影与微观表面/背景噪声等因子进行离散化；构建因子化基准和多样化采样集合

**📈 对比分析**

与随机高斯采样、全空间统一采样、以及无因子比例的混合采样对比；F-ACIL 在相同任务下实现 45% 以上性能提升，仅需 5–10 倍更少的演示；在 Pick‑and‑Place、Open‑and‑Close 两类任务的多维因子空间中均表现出显著的数据效率和泛化优势

**⚠️ 局限性**

局限性：因子空间选择仍有限（未包含语言、身体体现等维度）；假设因子独立，未考虑因子间潜在耦合；实验多在有限真实环境中进行，模拟验证不足；对极高维度任务与动态环境的适应性待进一步研究

---

## 475. Hierarchy-Guided Multimodal Representation Learning for Taxonomic Inference

**arXiv ID:** 2603.25573 | [PDF](https://arxiv.org/pdf/2603.25573v1)

**作者:** Sk Miraj Ahmed `[一作]` (Brookhaven National Laboratory), Wei Xu `[通讯]` (Brookhaven National Laboratory)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种基于层级信息正则化和自适应融合的多模态生物多样性识别框架，能够同时处理物种图像、DNA 条码和文本信息。

**💡 创新点**

创新点在于将生物分类的层级结构显式嵌入对比学习中，提出层级信息正则化（HiR）以塑造嵌入几何；并设计轻量级门控融合模块，使模型能在单模或双模噪声条件下自适应加权。

**🔧 技术方法**

采用 CLIP 风格的对比损失、HiR 正则化、门控融合头、DNABERT2 DNA 编码器、BioCLIP 文本编码器和视觉编码器，并使用分层监督进行多模态训练。

**📊 数据集**

使用 BIOSCAN-1M 数据集，其中包含 903,536 条训练样本和 224,777 条测试样本，样本同时包含物体图像、COI DNA 条码和分层文本描述。

**📈 对比分析**

与原始 CLIBD 以及基于平均的融合方法进行对比，在干净与噪声（DNA 噪声、图像+DNA 噪声）场景下，HiR 使整体 Top‑1 准确率提升约 14% 以上，融合模块在 DNA 噪声和双模噪声下进一步提升 2–3% 以上。

**⚠️ 局限性**

局限性包括未处理 CLIBD 见/不见拆分导致的长尾不平衡、训练时未加入噪声样本、以及对更大规模多模态场景的泛化能力仍需验证。

---

## 476. PackForcing: Short Video Training Suffices for Long Video Sampling and Long Context Inference

**arXiv ID:** 2603.25730 | [PDF](https://arxiv.org/pdf/2603.25730v1)

**作者:** Xiaofeng Mao `[一作]` (Shanda AI Research), Kaipeng Zhang `[通讯]` (Shanda AI Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

该方法通过三分KV缓存（sink、mid、recent）和双分支压缩实现了在单张GPU上生成长达120秒的高质量视频，解决了自回归视频生成中的误差累积和内存膨胀问题。

**💡 创新点**

创新点在于引入三分KV缓存策略、128×空间时间压缩的双分支压缩模块、增量RoPE调整和动态上下文选择，实现在固定内存约束下的长视频生成。

**🔧 技术方法**

采用流匹配训练框架、Transformer的KV缓存、3D卷积+VAE低分辨率重编码双分支压缩、RoPE位置编码调整以及基于查询‑键相似度的动态上下文选择等技术。

**📊 数据集**

训练仅使用5秒钟长的视频片段，数据来源于VidProM、MovieGen等；评估使用VBench‑Long的128个提示集。

**📈 对比分析**

与CausVid、LongLive、Self‑Forcing、Rolling Forcing、Deep Forcing等基线在VBench‑Long 60s/120s评估中比较，取得动态度最高（56.25/54.12），文本‑视频对齐CLIP得分保持最稳定，整体VBench指标均优于或接近最强基线。

**⚠️ 局限性**

仍存在在极高动态度与主体一致性之间的权衡，主角一致性略低于LongLive，且对极端场景的长期保持仍有待提升。

---

## 477. Natural-Language Agent Harnesses

**arXiv ID:** 2603.25723 | [PDF](https://arxiv.org/pdf/2603.25723v1)

**作者:** Linyue Pan `[一作]` (Tsinghua University), Hai-Tao Zheng `[通讯]` (Tsinghua University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出将代理系统中的控制层（harness）抽象为可编辑、可执行的自然语言表示（NLAHs），并设计共享运行时（IHR）直接解释执行该表示。

**💡 创新点**

将 harness 设计模式层外部化为可读的、可执行的自然语言对象，使得不同 harness 可以在同一共享运行时下进行比较、迁移与模块化 ablation。

**🔧 技术方法**

利用 IHR 在循环中嵌入 LLM，结合后台工具接口、多代理接口与运行时契约；实现自然语言驱动的任务调度、状态管理、验证与错误处理。

**📊 数据集**

在两个代表性基准上评估：编码类的 SWE‑bench Verified（issue 解决率）以及基于真实桌面环境的 OSWorld（任务成功率）。

**📈 对比分析**

通过对 Full IHR 与各组件剥离（runtime skill 与 harness skill）的对照实验，发现共享运行时和 harness 逻辑显著影响过程指标（token、调用数、运行时）且在大部分样本上保持一致性；模块化 ablation 展示不同控制模块对解决边界案例的影响；代码↔文本迁移实验表明自然语言实现可达到或略优于原始代码的性能。

**⚠️ 局限性**

局限性包括：自然语言表达缺乏代码的精确性，难以完整恢复隐藏的服务状态；共享运行时可能吸收本应归因于 harness 的行为；文本表示在 ablation 中可能带来指令显著性与长度等混杂影响。

---

## 478. MuRF: Unlocking the Multi-Scale Potential of Vision Foundation Models

**arXiv ID:** 2603.25744 | [PDF](https://arxiv.org/pdf/2603.25744v1)

**作者:** Bocheng Zou `[一作]`, Yong Jae Lee `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6514db3d-8de6-452c-91b7-acdb31787cc4` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构造多分辨率图像金字塔，利用冻结的 Vision Foundation Model（VFM）对不同分辨率的图像进行编码，随后把得到的特征图上采样到统一尺寸后按通道拼接得到统一的多尺度表征，并将该表征用于多种下游任务（语义分割、深度估计、视觉问答、无监督异常检测）。

**💡 创新点**

提出一种完全无训练（或仅训练轻量化头部）的推理时多尺度融合方法 MuRF，该方法仅通过简单的通道级拼接与上采样即可将低分辨率的全局语义信息与高分辨率的细粒度细节有机融合，且对任何 VFM 体系结构均通用。

**🔧 技术方法**

技术包括：多分辨率图像预处理、冻结 VFM 编码器（如 DINOv2‑ViT‑B/14、SigLIP2）、特征上采样、通道拼接、轻量化任务头（1×1 卷积、投影层等）以及无监督异常检测时的最近邻距离计算。

**📊 数据集**

使用的公开数据集包括：ADE20K、PASCAL VOC（语义分割）；NYU Depth V2、SUN RGB‑D（深度估计）；MME、VLMsAreBiased、GQA、MME RealWorld、RealWorld QA、mmbench（视觉问答）；MVTec AD 2（无监督异常检测）。

**📈 对比分析**

与单尺度基线（同一 VFM 在单一分辨率下的推理）以及多种先进方法（PatchCore、SuperAD、RoBiS 等）进行对比。MuRF 在语义分割的 mIoU、深度估计的 RMSE、视觉问答的各项指标以及异常检测的 AU‑PRO 上均实现了显著提升，往往达到或超过现有最优结果。

**⚠️ 局限性**

局限性：需要多次前向推理导致计算开销和显存占用增加；对 VFM 的依赖性较强，若 VFM 需要微调则需额外训练；通道维度随分辨率数目线性增长，可能在极大模型或资源受限环境下不易部署；对极大尺寸图像的多尺度处理未做专门优化。

---

## 479. PSDesigner: Automated Graphic Design with a Human-Like Creative Workflow

**arXiv ID:** 2603.25738 | [PDF](https://arxiv.org/pdf/2603.25738v1)

**作者:** Xincheng Shuai `[一作]` (Fudan University), Dacheng Tao `[通讯]` (Nanyang Technological University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一套自动化平面设计系统，能够根据用户指令收集主题素材，并通过多模态大模型进行工具调用，实现从素材整合到层级细化的可编辑PSD文件生成；

**💡 创新点**

系统模拟人类创作工作流，支持底层递归遍历与资产逐步融合；构建首个基于PSD、带操作轨迹的大规模设计数据集；结合LoRA微调与GRPO强化学习提升工具调用精度；

**🔧 技术方法**

采用Qwen2.5‑VL‑7B多模态大模型，基于LoRA进行监督微调，随后使用GRPO强化学习；通过Adobe UXP插件实现对Photoshop的实时工具调用；

**📊 数据集**

自建PSD设计数据集（约48层/样本，包含多种图层类型和属性），并在Crello‑v5上进行评测；

**📈 对比分析**

与OpenCOLE、Bagel、FLUX、PosterCraft、CanvaGPT、LaDeCo等方法对比，结果显示在美学、布局、内容相关性、色彩和创新度等维度均具竞争力，且能生成可编辑PSD文件，优于大多数非可编辑的输出；

**⚠️ 局限性**

受限于PSD格式与模型文本渲染能力，对极其复杂布局或特殊字体支持有限；实现对Photoshop的调用需要专用插件，部署与训练成本较高。

---

## 480. SlotVTG: Object-Centric Adapter for Generalizable Video Temporal Grounding

**arXiv ID:** 2603.25733 | [PDF](https://arxiv.org/pdf/2603.25733v1)

**作者:** Jiwook Han `[一作]` (Kyung Hee University), Jinwoo Choi `[通讯]` (Kyung Hee University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 SlotVTG 框架，在多模态大型语言模型（MLLM）中加入轻量化 Slot Adapter，实现对象级视觉表示，显著提升视频时序定位（VTG）任务的跨域泛化能力。

**💡 创新点**

创新点在于：①仅在早期解码器层插入可训练的 Slot Adapter，通过 Slot Attention + GRU 迭代实现视觉 token 的竞争性分解；②结合 Slot Alignment Loss，将自监督 DINOv2 的物体性先验对齐至 slot 关注权重，促使模型真正基于视觉内容进行推理；③保持参数高效（≈0.25%），无需重新训练完整视觉‑语言对齐流程。

**🔧 技术方法**

使用技术包括：Slot Attention、GRU 迭代、跨注意力重构、LoRA 参数微调、Slot Alignment Loss、DINOv2 自监督特征、MMD 领域差距度量、交叉域评估与多源多目标实验。

**📊 数据集**

采用 Charades‑STA、QVHighlights 和 ActivityNet Captions 三个标准 VTG 数据集进行源/目标交叉评估，验证跨域鲁棒性。

**📈 对比分析**

与零样本 MLLM、DETR 专用模型以及 Fine‑tuned MLLM 基线对比，SlotVTG 在 OOD 场景下 R1@0.5 提升 2.4–4.3 点，ID 性能保持不变或略升，说明显著提升了跨域表现。

**⚠️ 局限性**

局限性包括：对极大规模模型的适配仍待验证；当源域数据已足够多样时提升空间有限；过强的物体性先验可能导致对源域过拟合，需进一步调节 λ。

---

## 481. AnyHand: A Large-Scale Synthetic Dataset for RGB(-D) Hand Pose Estimation

**arXiv ID:** 2603.25726 | [PDF](https://arxiv.org/pdf/2603.25726v1)

**作者:** Chen Si `[一作]` (UC San Diego), Hao Su `[通讯]` (UC San Diego)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了大规模合成RGB-D手部数据集AnyHand，并在此基础上对现有RGB和RGB-D手姿估计模型进行联合训练，显著提升多种基准数据集上的性能。

**💡 创新点**

创新点包括：① 统一的合成流程生成包含手部、前臂、对象交互、背景与光照多样化的真实感RGB与精确深度图；② 轻量级双模深度融合模块，可无缝嵌入ViT基础架构；③ 通过大规模合成数据与真实数据的共训练，突破单一数据集局限，实现跨域泛化。

**🔧 技术方法**

技术手段：基于SAPIEN渲染管线、SMPL/SMPL+H与MANO模型、DPoser-Hand扩散式手姿先验、Handy纹理生成器、随机化相机与光照、MoGe-2背景深度估计、双分支Transformer与跨模态双向注意力融合。

**📊 数据集**

使用数据集：AnyHand（2.5M单手+4.1M手物交互RGB-D），真实数据如FreiHAND、HO-3D、DexYCB、HO-Cap、HO-3D v2；基准评测包含FreiHAND、HO-3D v2、HO-Cap、HO-3D v2（RGB-D）。

**📈 对比分析**

对比方法：HaMeR、WiLoR（RGB）、Keypoint-Fusion、IPNet（RGB-D）。实验表明：在RGB设置下，co‑training with AnyHand可使HaMeR PA-MPJPE从6.0mm降至5.54mm、WiLoR从6.5mm降至5.3mm；在RGB‑D设置下，AnyHand+融合模块使STA-MPJPE从1.87cm降至1.09cm、PA-MPJPE从1.31cm降至0.79cm，均显著优于之前最佳方法。

**⚠️ 局限性**

局限性：合成数据与真实世界的细微差异仍存在，特别是深度分辨率与噪声分布；合成的纹理与光照虽多样，但仍可能缺乏极端真实场景；模型对极端遮挡与非手部运动的鲁棒性仍待提升。

---

## 482. ShotStream: Streaming Multi-Shot Video Generation for Interactive Storytelling

**arXiv ID:** 2603.25746 | [PDF](https://arxiv.org/pdf/2603.25746v1)

**作者:** Yawen Luo `[一作]` (Chinese University of Hong Kong), Tianfan Xue `[通讯]` (Chinese University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发ShotStream，一个因果多镜头视频生成架构，实现实时、交互式长叙事视频合成。

**💡 创新点**

将多镜头生成改为下一镜头自回归任务，支持流式提示；双缓存记忆机制与RoPE断点标识保证跨镜头和镜内一致性；两阶段自迫式蒸馏减少错误累积。

**🔧 技术方法**

基于文本到视频模型的双向下一镜头教师，分布匹配蒸馏至4步因果学生；双缓存、RoPE断点、动态采样、句子提示注入；自迫式训练。

**📊 数据集**

内部320K多镜头视频集训练教师；内部5K教师采样对用于蒸馏；在100个由Gemini 2.5 Pro生成的多镜头提示上评估。

**📈 对比分析**

与Mask2DiT、EchoShot、CineTrans等双向模型以及Self Forcing、LongLive、Rolling Forcing、Infinity-RoPE等因果模型对比；ShotStream在视觉一致性、提示遵循、过渡控制等指标上均居首；帧率16 FPS，推理速度比双向模型快25×。

**⚠️ 局限性**

在场景与提示复杂时易出现视觉瑕疵和不一致；模型规模受限，可通过扩展基模型和加入稀疏/注意力加速进一步提升。

---

## 483. RefAlign: Representation Alignment for Reference-to-Video Generation

**arXiv ID:** 2603.25743 | [PDF](https://arxiv.org/pdf/2603.25743v1)

**作者:** Lei Wang `[一作]` (Nankai University), jian Yang `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 RefAlign 框架，利用训练阶段的 Reference Alignment（RA）损失将 Diffusion Transformer（DiT）参考分支的特征显式对齐到视觉基础模型（VFM）特征空间，从而提升参考图像在视频生成中的身份一致性与细节保真。

**💡 创新点**

创新点在于：1) 在推理时不增加任何计算负担，仅在训练阶段通过 RA 损失实现显式对齐；2) 采用正负对齐策略，将同一主体特征拉近、不同主体特征推远，显著缓解 copy‑paste 与多主体混淆问题；3) 通过对比实验验证该对齐策略在多模态条件下的有效性。

**🔧 技术方法**

技术手段包括：VAE + DiT（Wan‑2.1）生成器；Frozen VFM（如 DINOv3）作为教师；Reference Alignment 损失；T5 作为文本编码器；CFG 与 50 步 Euler 采样；对比学习与正余对齐的交叉熵/余弦相似度计算。

**📊 数据集**

训练数据：OpenS2V‑5M 中挑选的 360K 张图像‑文本‑视频三元组子集；评估数据：OpenS2V‑Eval 基准，生成 180 条视频。

**📈 对比分析**

与多种闭源（Vidu、Pika、Kling、Saber）及开源（VACE、Phantom、MAGREF、VINO、Kaleido、BindWeave）R2V 方法对比；在 OpenS2V‑Eval 上，1.3B 版 RefAlign 获得 60.42% TotalScore，排名第一；14B 版同样位居榜首；在 NexusScore、FaceSim 等指标上表现最优。

**⚠️ 局限性**

局限性：1) 训练数据多样性不足，导致指令遵循与参考保真难以同时优化；2) 目前仅支持 81 帧短视频，无法处理长视频；3) 仅使用单一 VFM 作为对齐目标，可能无法全面覆盖所有视觉语义，限制了泛化能力。

---

## 484. MegaFlow: Zero-Shot Large Displacement Optical Flow

**arXiv ID:** 2603.25739 | [PDF](https://arxiv.org/pdf/2603.25739v1)

**作者:** Dingxi Zhang `[一作]` (ETH Zurich), Haofei Xu `[通讯]` (ETH Zurich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种基于预训练视觉Transformer的统一框架MegaFlow，用于大位移光流估计和零样本点跟踪

**💡 创新点**

通过将全局匹配与轻量级迭代细化相结合，既能捕获大范围位移，又能保持亚像素精度；框架无需任务特定微调，兼容任意帧数输入

**🔧 技术方法**

预训练DINOv2/ VGGT Transformer、CNN局部特征提取、全局匹配（softmax关联）、ConvNeXt+时序注意力的递归细化

**📊 数据集**

训练集：FlyingChairs、TartanAirV1、FlyingThings；混合集：HD1K、Sintel、KITTI；零样本评测在Sintel、KITTI、Spring、TAP‑Vid、DAVIS等

**📈 对比分析**

与RAFT、GMA、FlowFormer、MemFlow等两帧/多帧光流模型及AllTracker、CoTracker等点跟踪器比较；在Sintel Final、KITTI Fl‑all、Spring EPE上均实现SOTA零样本性能，TAP‑Vid零样本精度超过多数光流基线，微调后突破现有跟踪器

**⚠️ 局限性**

在长序列中计算开销较大，缺乏高效的序列级优化；对极端运动的鲁棒性虽强，但在快速前行的KITTI等场景下多帧窗口会引入遮挡误差

---

## 485. Unleashing Guidance Without Classifiers for Human-Object Interaction Animation

**arXiv ID:** 2603.25734 | [PDF](https://arxiv.org/pdf/2603.25734v1)

**作者:** Ziyin Wang `[一作]` (University of Illinois Urbana Champaign), Liang-Yan Gui `[通讯]` (University of Illinois Urbana Champaign)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出LIGHT框架，利用无先验的文本驱动人机交互动画生成。

**💡 创新点**

通过异步噪声调度实现的pace‑induced guidance，替代外部分类器，并结合形状光谱数据增强提升泛化。

**🔧 技术方法**

Diffusion forcing、classifier‑free guidance、Transformer decoder、BPS对象编码、形状对齐优化等技术。

**📊 数据集**

主要使用InterAct、BEHAVE、OMOMO数据集，并通过ShapeNet/Objaverse扩展至1121个对象。

**📈 对比分析**

与HOI‑Diff、CHOIS、InterDiff、Text2HOI等基线对比，取得更低FID、MM Dist、R‑Precision，并在未见对象任务上表现更好。

**⚠️ 局限性**

仍需大量HOI数据，处理大物体或极复杂场景受限，生成速度依赖于500步扩散。

---

## 486. BizGenEval: A Systematic Benchmark for Commercial Visual Content Generation

**arXiv ID:** 2603.25732 | [PDF](https://arxiv.org/pdf/2603.25732v1)

**作者:** Yan Li `[一作]` (Shanghai Jiao Tong University), Chong Luo `[通讯]` (Microsoft Corporation)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了BizGen基准，用于系统评估图像生成模型在商业视觉内容（如幻灯片、图表、网页、海报和科研图）上的生成质量，包含400个细化提示和8,000个人工核验清单，并对26个主流模型进行大规模评测。

**💡 创新点**

创新点在于：①构建了覆盖五类商业文档、四大能力维度（文本渲染、布局控制、属性绑定、知识推理）的完整评测框架；②设计了结构化的多项检查清单与MLLM自动评判流程；③通过大量真实样本与知识点收集，填补了现有自然图像基准在商业设计场景下的空白。

**🔧 技术方法**

采用多模态大语言模型（Gemini‑3‑Flash‑Preview）作为评判者，对生成结果逐项回答二进制检查题；使用人工构造的提示、视觉‑语言分析、自动分数计算等技术；评测涉及闭源商业API和开源模型的多样化推理与生成。

**📊 数据集**

数据集来源为1,819张真实商业设计图（幻灯片、图表、网页、海报、科研图）与100条专业领域知识点，随后生成400个精细提示及对应8,000个人工核验问题。

**📈 对比分析**

通过对26个模型的易/难子集分数进行对比，闭源API（如Nano‑Banana‑Pro）在文本与知识维度上取得高分（>90%），但在布局与属性维度仍低于70%；开放源模型整体分数明显偏低，尤其在图表与科研图等高精度场景下接近零，揭示了当前模型在商业设计中的显著差距。

**⚠️ 局限性**

局限性包括：①模型仍缺乏对精准布局与属性的确定性控制；②文本与知识推理表现极不均衡，部分模型几乎无法正确渲染文字或融入知识；③基准只覆盖五类文档，未必完全覆盖所有商业设计需求；④评判依赖MLLM，可能对模型内部生成细节的细粒度判断有限。

---

## 487. PixelSmile: Toward Fine-Grained Facial Expression Editing

**arXiv ID:** 2603.25728 | [PDF](https://arxiv.org/pdf/2603.25728v1)

**作者:** Jiabin Hua `[一作]` (Fudan University), Yu-Gang Jiang `[通讯]` (Fudan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于扩散模型的细粒度面部表情编辑框架，解决表情语义重叠问题，实现连续、可控的表情编辑与多表情混合；

**💡 创新点**

创新点在于使用连续情感标注的Flex Facial Expression数据集和全对称联合训练+流匹配文本潜在插值机制，显著降低语义混淆并实现精准线性控制；

**🔧 技术方法**

核心技术包括多模态扩散变换器（MMDiT）+ LoRA、文本潜在插值与流匹配评分监督、完全对称的对比学习以及身份保持损失；

**📊 数据集**

采用了Flex Facial Expression (FFE) 数据集，总计60,000张图像（6k真实、6k动漫），覆盖12类情感，采用连续12维情感评分；

**📈 对比分析**

与多种通用编辑模型（如Nano Banana Pro、GPT-Image-1.5）以及线性控制模型（如SliderEdit、ConceptSlider）对比，mSCR降至0.055，编辑准确率>0.86，CLS和HES均处于最佳水平，整体保持身份一致；

**⚠️ 局限性**

局限性包括对VLM评估的依赖导致在复杂表情组合时仍出现冲突，极端α值下可能出现身份漂移，对模型训练需要大量对齐数据且缺乏视频动态验证。

---

## 488. Agent Factories for High Level Synthesis: How Far Can General-Purpose Coding Agents Go in Hardware Optimization?

**arXiv ID:** 2603.25719 | [PDF](https://arxiv.org/pdf/2603.25719v1)

**作者:** Abhishek Bhandwaldar `[一作]` (IBM Corporation), Akash Srivastava `[通讯]` (IBM Corporation)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了一种两阶段代理工厂流程，利用多代理自主探索高层合成(HLS)设计空间，从源代码和pragmas出发自动生成和优化硬件实现。

**💡 创新点**

创新点在于将代理数量作为推理时的可扩展维度，展示通用编码代理（无硬件专属训练）能够发现既有的硬件优化模式并实现跨函数级别的代码与pragma变换；并将子核级别的ILP组合与全局代理探索相结合。

**🔧 技术方法**

采用代理系统、整数线性规划(ILP)、Claude Code Opus 4.5/4.6作为语言模型、AMD Vitis HLS作为合成工具、功能验证、代码与pragma自动变换技术。

**📊 数据集**

使用12个HLS基准核：HLS‑Eval（AES, DES, KMP, NW, PRESENT, SHA256）与Rodinia‑HLS（lavamd, kmeans, hotspot, leukocyte, cfd, streamcluster）。

**📈 对比分析**

对比方法包括基于指令枚举的限定搜索基线和参考优化实现；结果显示从1个代理到10个代理平均加速比提升至8.27×，在复杂工作负载如streamcluster可达20×，并在不同面积预算下展示Pareto前沿。

**⚠️ 局限性**

限制在于实验仅在单一模型、单一工具链（Vitis HLS）与单一FPGA平台上进行；基线受限于指令空间；基准规模较小；未评估ASIC或其他工具链；代理探索受token成本与搜索效率限制，缺乏与更先进DSE框架的深入对比。

---

## 489. TRACE: Object Motion Editing in Videos with First-Frame Trajectory Guidance

**arXiv ID:** 2603.25707 | [PDF](https://arxiv.org/pdf/2603.25707v1)

**作者:** Quynh Phung `[一作]` (University of Maryland), Aniruddha Mahapatra `[通讯]` (Adobe Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于用户在第一帧绘制目标轨迹的“第一帧对象运动设计”框架，利用两阶段管线在保持原场景内容的前提下，重新生成目标物体沿用户指定路径运动的视频。

**💡 创新点**

创新点包括：①首次提出第一帧运动设计的编辑范式；②跨视角运动转换模块，直接把第一帧的二维路径映射为动态视角的帧级框序列，避免显式3D重建；③运动条件视频重合成模块，结合对象消除和重生实现高质量、时空一致的运动重写；④使用合成数据管线自动生成大规模配对训练样本；⑤支持多目标编辑、插入、替换等多种应用。

**🔧 技术方法**

技术手段：跨视角转换采用Diffusion Transformer (DiT) + 3DVAE 编码 + 点轨迹 (DCT) 条件；重合成使用大型视频扩散模型 Wan 2.1 1.4B + LoRA 微调，输入为第一帧、掩膜视频、目标框和文本提示；训练采用 Flow‑matching 损失；数据生成使用 ReCamMaster、CoTracker、MegaSAM/DepthAnything 3 等工具。

**📊 数据集**

训练与评估数据：合成 75k 视角视频（7.5k 静态 + 10 动态路径），约 110k 对齐框序列；用于重合成的内部 1.1M 视频；评测基准包括 DAVIS、VBench 以及自建 70 条测试视频。

**📈 对比分析**

与多种 I2V（MagicMotion、Wan‑Move、MotionCanvas）和 V2V（VACE、HunyuanCustom、GenCompositor、PISCO）基线对比，跨视角转换在 IoU_f2v 0.80、mAP_f2v 0.91 上超越插值和 3D warping；重合成在 SSIM、LPIPS、Tube IoU 等指标上表现最佳，PSNR 与 mAP 次优。整体管线在 FVD、FID、KID 以及 VBench 的多维度指标上均取得最优或领先成绩，用户研究显示偏好率超过 70%。

**⚠️ 局限性**

局限性：1) 对极端摄像机运动、快速运动或光照变化仍易产生抖动或失真；2) 依赖第一帧框标注，多目标编辑仍需手动多次绘制；3) 生成过程基于扩散模型，推理速度相对较慢；4) 目前缺乏显式 3D 解释性，难以在复杂三维场景中精准控制；5) 对于遮挡、透明物体的处理仍不够稳健。

---

## 490. Less Gaussians, Texture More: 4K Feed-Forward Textured Splatting

**arXiv ID:** 2603.25745 | [PDF](https://arxiv.org/pdf/2603.25745v1)

**作者:** Yixing Lao `[一作]` (University of Hong Kong), Hengshuang Zhao `[通讯]` (University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了LGTM，一个前向可预测纹理化高斯原语的网络，实现4K分辨率的新视角合成

**💡 创新点**

创新点在于双网络架构将几何和纹理预测分离，利用低分辨率几何网络与高分辨率纹理网络，并采用投影纹理映射，突破了传统3D高斯分裂方法在高分辨率下的可扩展性瓶颈

**🔧 技术方法**

采用2D高斯分裂（2DGS）、ViT编码器-解码器、图像Patchify与投影特征融合、纹理化高斯原语、分阶段训练和高分辨率监督

**📊 数据集**

使用RealEstate10K（RE10K）和DL3DV-10K数据集进行评估

**📈 对比分析**

在单视、双视（有姿势/无姿势）和多视任务中与Flash3D、NoPoSplat、DepthSplat、VGGT等基线比较，LGTM在PSNR/SSIM/LPIPS上均有显著提升；在4K时仅用1.8×内存、1.47×时间完成64倍像素的推理

**⚠️ 局限性**

局限性包括对几何质量依赖较大；在多视场景中提升有限；纹理分辨率需手动调节；仍需进一步提升几何精度和多视一致性

---

## 491. Drive My Way: Preference Alignment of Vision-Language-Action Model for Personalized Driving

**arXiv ID:** 2603.25740 | [PDF](https://arxiv.org/pdf/2603.25740v1)

**作者:** Zehao Wang `[一作]` (University of California Riverside), Jiachen Li `[通讯]` (University of California Riverside)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一套名为Drive My Way（DMW）的端到端视觉-语言-动作（VLA）驾驶框架，能够通过学习驾驶者长期偏好嵌入并结合实时自然语言指令，实现个性化驾驶决策；

**💡 创新点**

创新点包括：①将长期驾驶偏好与即时语言指令联合编码进策略；②使用对比学习构建用户嵌入并对齐驾驶轨迹嵌入；③引入基于风格的奖励自适应机制，实现安全、效率、舒适三者的动态权重平衡；④通过残差解码器在保留基准规划的基础上精细调节动作，实现多样化的个性化行为；

**🔧 技术方法**

技术手段包括：SimLingo（InternVL2-1B + Qwen2-0.5B）VLA骨干网络；对比学习（InfoNCE）用于嵌入对齐；Group Relative Policy Optimization（GRPO）强化微调；残差解码器与PID控制结合产生最终动作；自适应平均池化（AAP）提升嵌入表达；以及多阶段奖励参数生成（LLM+专家校准）；

**📊 数据集**

使用了自研的Personalized Driving Dataset（PDD），在CARLA模拟器中收集了30名真实驾驶者的长周期驾驶数据与结构化个人档案；评估时采用Bench2Drive基准；

**📈 对比分析**

与SimLingo、StyleDrive、MORL-PD等基线进行闭环评测，指标包括Driving Score、Success Rate、效率、舒适度、加速度等；DMW在风格指令适配方面表现最佳，提升DS与SR，保持安全门槛，同时在对齐分数（Alignment Score）和用户评分上显著优于对比方法；

**⚠️ 局限性**

局限性在于：①目前仅在CARLA仿真环境验证，缺乏真实车辆实验；②数据集规模（30位驾驶者）有限，可能影响对新驾驶者的泛化；③奖励自适应依赖LLM推理与人工校准，复杂度较高；④在高交互场景下可能趋向保守，导致个性化表达受限；

---

## 492. How good was my shot? Quantifying Player Skill Level in Table Tennis

**arXiv ID:** 2603.25736 | [PDF](https://arxiv.org/pdf/2603.25736v1)

**作者:** Akihiro Kubota `[一作]` (Kyoto University), Ko Nishino `[通讯]` (Kyoto University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

通过学习球拍击球向量的生成模型，构建球员嵌入空间，并利用该空间对乒乓球选手的技能水平进行量化与排名预测。

**💡 创新点**

创新点在于：① 将球员动作视为受比赛上下文条件的随机生成分布并联合学习球员嵌入；② 采用HitFormer估计击球向量、HitFlow基于Flow Matching生成击球分布；③ 用RankNet/SkillNet从嵌入中直接提取技能分数，实现在无显式评分的情况下的技能评估。

**🔧 技术方法**

主要技术包括：Transformer-based HitFormer、Flow Matching HitFlow、FiLM条件化、SMPL姿态估计与物理仿真、RankNet/SkillNet进行排名预测。

**📊 数据集**

使用JTTA官方的《All Japan Table Tennis Championships》数据集（36场男场、34场女场，共约1.08M帧，25男28女职业选手），以及合成的SynthHit数据集用于HitFormer训练。

**📈 对比分析**

与TT3D基准和LATTE-MV自回归模型比较：HitFormer在MAE上（10.1–18.5 cm）优于TT3D；HitFlow在能量得分上为0.37，优于LATTE-MV 0.40；球员嵌入在技能等级预测中实现Spearman相关系数≈0.66–0.70，且相对排名准确率超过70%。

**⚠️ 局限性**

局限性包括：单摄像头视角导致重建误差、样本量相对有限、缺乏绝对评分标签、对年龄信息不敏感、对排名相近的选手区分度不高。

---

## 493. No Hard Negatives Required: Concept Centric Learning Leads to Compositionality without Degrading Zero-shot Capabilities of Contrastive Models

**arXiv ID:** 2603.25722 | [PDF](https://arxiv.org/pdf/2603.25722v1)

**作者:** Hai X. Pham `[一作]` (Samsung AI Center), Brais Martinez `[通讯]` (Samsung AI Center)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

针对 CLIP 类模型的组合推理（compositionality）性能进行提升，提出一种无硬负样本、基于短名词短语的概念中心对齐方法，并在训练时加入参数无关的交叉注意力池化。

**💡 创新点**

创新点：①利用短名词短语（概念）替代长文本进行对齐，显著减少 Bag‑of‑Words 影响；②在视觉编码器上引入无参数交叉注意力池化，在全局池化前就实现概念绑定；③通过概念对齐损失和交叉注意力对齐损失两种辅助对齐，提升组合推理同时不牺牲零样本和检索性能。

**🔧 技术方法**

技术手段：SigLIP 视觉/文本编码器、对比式 Sigmoid 损失、概念对齐损失 ℒ_npc、交叉注意力对齐损失 ℒ_xac、无参数注意力池化、spaCy 依存句法分割提取名词短语。

**📊 数据集**

数据集：CC3M（DreamLIP 变体）进行微调；评测使用 SugarCrepe、SugarCrepe++（组合推理）、ImageNet1K（零样本分类）、Flickr30K、MSCOCO、DOCCI、IIW（检索）等。

**📈 对比分析**

对比方法：与 SigLIP、CLIP、CLIP‑B/16、各种 composition‑aware 方案（CE‑CLIP、NegCLIP、CLIC、DAC‑LLM 等）以及大模型（CLIP‑L、BLIP、FLAVA 等）进行对比。C2LIP 在 SugarCrepe、SugarCrepe++ 上均达 SOTA，且在检索、零样本分类等任务上保持或提升性能，整体平均分比大多数基线高 1–3% 左右，唯一较弱点是 ImageNet 分类略微下降。

**⚠️ 局限性**

局限性：1）ImageNet 性能略降，说明更注重场景而非单一物体的表征；2）当前池化方式仍为全局，可能仍无法完美保留细粒度绑定；3）仅验证了对象-属性绑定，未扩展到更复杂的关系推理。

---

## 494. R-C2: Cycle-Consistent Reinforcement Learning Improves Multimodal Reasoning

**arXiv ID:** 2603.25720 | [PDF](https://arxiv.org/pdf/2603.25720v1)

**作者:** Zirui Zhang `[一作]` (Rutgers University), Chengzhi Mao `[通讯]` (Rutgers University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过交叉模态循环一致性奖励（R-C^2）实现多模态大型语言模型的自监督强化学习，提升视觉与文本预测的一致性和推理准确率。

**💡 创新点**

创新点在于：①把模态不一致视为自我奖励信号而非投票误差；②设计答案→查询→交叉模态→答案的循环，生成稠密的无标签奖励；③采用全四路（T→T, T→I, I→T, I→I）循环实现内部与跨模态一致性约束。

**🔧 技术方法**

技术主要包括：跨模态强化学习（GRPO）框架、反向推理生成查询、前向推理重构答案、离线循环数据生成、全四路一致性奖励。

**📊 数据集**

使用的多模态推理基准包括 ScienceQA、ChartQA、MathVista、A-OKVQA、DocVQA、InfoVQA 以及 Visual Web Arena；模型基线为 Qwen2.5-VL-3B-Instruct 与 Qwen3-VL-8B-Instruct。

**📈 对比分析**

与单模态和多模态多数投票基线对比，R-C^2 在所有六大基准上均提升 2–7.6 分（文本/视觉准确率），并显著提高跨模态一致性（最多提升 12.5 分）。

**⚠️ 局限性**

局限性包括：①奖励仅为二元，可能忽略细粒度推理误差；②循环需要高质量的反向查询，若生成质量差会降低奖励信号；③对需要精确语义对齐的模态更有效，非对齐或噪声模态的提升有限。

---

## 495. Seeing to Ground: Visual Attention for Hallucination-Resilient MDLLMs

**arXiv ID:** 2603.25711 | [PDF](https://arxiv.org/pdf/2603.25711v1)

**作者:** Vishal Narnaware `[一作]` (University of Central Florida), Mubarak Shah `[通讯]` (University of Central Florida)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 VISAGE，一个无训练的解码重排序框架，用空间熵校准并行掩码解码的目标误匹配，从而抑制多模态扩散 LLM 的幻觉现象。

**💡 创新点**

将幻觉视为局部优化误差，利用交叉注意力的空间熵估计与 β‑分位数聚合实现视觉定位共识，并给出稳健性误差上界，实现在不更新模型参数的前提下提升视觉与语言的对齐。

**🔧 技术方法**

并行掩码解码、交叉注意力空间熵计算、β‑分位数聚合、重排序修正、理论误差分析与稳定性证明。

**📊 数据集**

POPE、HallusionBench、MMMU‑val、MME 等多模态图像‑文本评测基准。

**📈 对比分析**

与基线 MMaDA 及 Visual Contrastive Decoding（VCD）对比，在幻觉敏感任务上相对提升 8.59%（MMMU‑val）和 7.75%（HallusionBench），并在整体多模态基准上保持或提升性能。

**⚠️ 局限性**

目前仅针对图像到文本的并行解码，未考虑视频时序一致性，未来需扩展至视频多模态扩散模型并整合时间维度的定位一致性。

---

## 496. Wan-Weaver: Interleaved Multi-modal Generation via Decoupled Training

**arXiv ID:** 2603.25706 | [PDF](https://arxiv.org/pdf/2603.25706v1)

**作者:** Jinbo Xing `[一作]` (Tongyi Lab), Yujiu Yang `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Wan-Weaver，一个将规划与可视化专家分离的统一多模态模型，用于交错生成文本与图像序列。

**💡 创新点**

创新点在于将交错生成拆解为文本规划与视觉一致性两步，利用文本代理数据和解耦训练来弥补缺乏真实交错数据的不足。

**🔧 技术方法**

采用 MoT 结构的 VLM 规划器和 Diffusion Transformer 可视化器，配合 Dense Prompt Context Window、VAE 视觉编码器以及 RoPE 等技术。

**📊 数据集**

使用 Qwen2.5-VL 预训练模型、Wan2.2 VAE 以及大规模合成的文本代理交错数据、参考引导图像数据、视频关键帧与图像聚类数据等多样化数据集。

**📈 对比分析**

在 WeaverBench 与 OpenING 基准上与多种开源、集成管道和商业模型（如 Nano Banana）对比，Wan-Weaver 超越所有开源方法，且性能与商业模型相当。

**⚠️ 局限性**

局限在于对合成文本代理数据的依赖，仍难以完全捕捉真实交错场景中的细粒度语义与视觉细节；极长上下文或复杂多图像连续生成时可能出现一致性衰退。

---

## 497. Stone Duality for Monads

**arXiv ID:** 2603.25710 | [PDF](https://arxiv.org/pdf/2603.25710v1)

**作者:** Richard Garner `[一作]` (Macquarie University), Nicolas Wu `[通讯]` (Imperial College London)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文提出了一种新的对偶结构：构造了一个局部化行为类别（localic behaviour category）LBT，并证明其与给定的排名单子（ranked monad）T 的对偶关系，即 LBT 对应于 T 的“真实世界”状态机；随后给出了从局部化类别到单子再回来的右伴随 functor Γ，形成一个自反的、反变的伴随对，并证明其固定点正是 hyperaffine‑unary 单子与 ample 局部化类别之间的等价，即一种针对单子的新型 Stone 对偶性。

**💡 创新点**

创新点主要包括：
- 1) 将单子与内部类别之间的经典行为类别推广到局部化（无点）情形，以处理无限 arity 的单子；
- 2) 引入“有限信息拓扑”（finite‑information topology）以约束可接受的计算，避免无穷信息的无理计算；
- 3) 通过 retrofunctor 取代传统函子，得到更自然的“模拟”关系；
- 4) 证明该伴随对自反，固定点正好对应 hyperaffine‑unary 单子与 ample 局部化类别，因而提供了一个全新的 Stone 对偶性框架；
- 5) 为单子提供了“超射读” (scrying) 形式的超幂运算与局部化行为类别的逻辑解释。

**🔧 技术方法**

主要技术手段：
- 1) 经典范畴论（单子、自由/终极 comodel、行为类别、内部类别、retrofunctor）;
- 2) 局部化空间与点自由拓扑（locale、frame、Grothendieck Boolean algebra、ultraparacompact locales）；
- 3) comodel 语义与操作拓扑；
- 4) 形如 ΓLC 的“全局截面单子”构造；
- 5) 以局部化行为类别的源/目标映射为局部 homeomorphism 的 sheaf 理论；
- 6) 证明对偶关系的自然性与自反性，并给出固定点的代数/几何描述。

**📊 数据集**

本研究为理论论文，未使用任何实验数据集；所有结论均通过纯粹的范畴与拓扑论证得出。

**📈 对比分析**

由于是纯理论工作，本文不涉及实验对比或性能评估；主要通过对偶性、可证明性以及结构等价性来验证方法有效性。

**⚠️ 局限性**

局限性：
- 只适用于 Set 上的排名单子，无法直接推广到更一般的基类或更广义的单子；
- 需要单子满足可预见性（hyperaffine‑unary）或局部化类别满足 ample 条件，非满足者无法获得完全对偶；
- 对于包含非终止/非确定性运算的单子（如 fail、⊕ 等），局部化行为类别与 ΓLC 仍为平凡，导致该方法在这些情形下失效；
- 只讨论了无点局部化空间，未探讨更一般的拓扑/点空间情形；
- 未来工作需要进一步拓展至部分单子、Kleisli 场景或与模型检查/逻辑推理的结合。

---

## 498. Vega: Learning to Drive with Natural Language Instructions

**arXiv ID:** 2603.25741 | [PDF](https://arxiv.org/pdf/2603.25741v1)

**作者:** Sicheng Zuo `[一作]` (Tsinghua University), Jiwen Lu `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了约10万条带有自然语言指令的驾驶数据集（InstructScene），并提出了统一的 vision‑language‑world‑action 模型 Vega，能够根据视觉观测和用户指令生成多种可执行轨迹及未来场景图像。

**💡 创新点**

创新点包括：①将未来图像生成作为稠密监督来弥补动作监督稀疏的缺陷；②采用混合自回归‑扩散 Transformer 并通过 Mixture‑of‑Transformers (MoT) 解耦多模态参数；③设计交错图像‑动作序列来提升模态间动态学习；④使用无监督指令生成管道自动构造指令数据。

**🔧 技术方法**

技术手段：集成 Transformer（自回归 + 扩散）、MoT、噪声扩散模型、causal attention、classifier‑free guidance、VAE、SigLIP2 ViT、Qwen2.5 tokenizer、动作相对表示、联合损失优化。

**📊 数据集**

数据集：NAVSIM v1/v2 基准数据以及自建的 InstructScene（约10万条指令场景）用于训练与评估。

**📈 对比分析**

与多种 SOTA 方法（VLA、BEV、CoT 结合的 RL 方法等）在 NAVSIM v1/v2 上对比。Vega 在 NAVSIM v2 获得 86.9 EPDMS，best‑of‑N 策略下突破 90 分；在 NAVSIM v1 获得 87.9 PDMS（best‑of‑N 89.8），在驾驶方向、红绿灯、车道保持、舒适度等指标上均优于现有方法。

**⚠️ 局限性**

局限性：①对高分辨率多视角输入依赖较弱；②指令多样性虽大但仍受生成模型偏好限制；③在 NAVSIM v1 的风险规避指标略逊于极端规避策略；④模型规模大、推理耗时相对较长；⑤训练需要大量 GPU 资源和大规模指令数据。

---

## 499. Training the Knowledge Base through Evidence Distillation and Write-Back Enrichment

**arXiv ID:** 2603.25737 | [PDF](https://arxiv.org/pdf/2603.25737v1)

**作者:** Yuxing Lu `[一作]` (Peking University), Jinzhuo Wang `[通讯]` (Peking University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本工作提出将检索增强生成（RAG）系统中的知识库视为可训练的组件，利用标注数据的检索行为筛选有用文档，并通过LLM进行融合压缩，将所得写回知识单元追加到原始语料库，从而提升检索与生成的整体性能。

**💡 创新点**

核心创新点包括：① 将知识库训练纳入RAG框架；② 采用两阶段门控（utility gate与document gate）精确识别哪些样本和文档对检索有益；③ 利用LLM蒸馏将多文档证据压缩成单一、紧凑的写回知识单元；④ 该写回知识可以与任何检索器/生成器无缝结合，成为一种与现有技术互补的提升手段。

**🔧 技术方法**

使用的主要技术包括：RAG四种基线方法（Naive Retrieval、RePlug、Self‑RAG、FLARE）；LLM（Llama‑3.1‑8B、Gemma‑3‑12B）做生成与蒸馏；E5‑base‑v2作为检索器；两阶段门控逻辑；LLM蒸馏模块；FAISS索引实现写回知识库。

**📊 数据集**

实验数据集为FlashRAG公开六个基准：Natural Questions（NQ）、BoolQ、FEVER、zsRE、HotpotQA、SQuAD，并以Wikipedia为检索语料。

**📈 对比分析**

在四种RAG + 两种LLM、六个基准的48种设置中，写回知识均实现提升，平均提升+2.14%；单一设置最高提升+5.12%；并且在跨方法写回实验中，写回知识的效果几乎不受RAG方法差异影响，说明其对知识库本身的改进具有普适性。

**⚠️ 局限性**

局限性包括：需要标注训练样本；LLM蒸馏可能引入错误或偏见；当前仅实现写回增量，未涉及删除、去重或冲突解决；离线成本较高；实验仅基于公开Wiki数据，未验证多语言、持续更新或非公开领域的适用性。

---

## 500. Back to Basics: Revisiting ASR in the Age of Voice Agents

**arXiv ID:** 2603.25727 | [PDF](https://arxiv.org/pdf/2603.25727v1)

**作者:** Geeyang Tay `[一作]` (Boson AI), Alex Smola `[通讯]` (Boson AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了WildASR，一个基于真实人类语音的多语言诊断基准，系统评估ASR在环境降解、人口偏移和语言多样性三大维度的鲁棒性。

**💡 创新点**

创新点在于将OOB条件细分为环境、人口和语言三轴，并使用真实语音而非TTS生成数据，提供因子分离的评估与诊断工具。

**🔧 技术方法**

采用多种声学增强（混响、远场、电话编解码、噪声插入、削波）以及手工挑选的人口语料和语言多样性数据，对七种主流ASR模型进行统一推理。

**📊 数据集**

数据集来源包括FLEURS、MagicData、SwitchLingua、YODAS、Zenodo儿童语音、GLOBE等四种语言的真实录音。

**📈 对比分析**

通过统一协议评估七个模型，结果显示即使在清晰数据上表现优异，模型在不同语言和条件下仍出现不均衡的大幅性能下降，且部分模型会产生语义幻觉。

**⚠️ 局限性**

局限性包括仅覆盖四种语言、人口子集样本有限、缺乏多说话人重叠和实时流媒体等更复杂情况，以及仅做诊断未提供鲁棒性提升方法。

---

## 501. SoftMimicGen: A Data Generation System for Scalable Robot Learning in Deformable Object Manipulation

**arXiv ID:** 2603.25725 | [PDF](https://arxiv.org/pdf/2603.25725v1)

**作者:** Masoud Moghani `[一作]` (NVIDIA), Ajay Mandlekar `[通讯]` (NVIDIA)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `67630363-6be0-4f51-ab05-7198250671a5` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一套自动化数据生成流水线，利用少量人类遥控演示在仿真环境中大规模生成可变形物体操作数据，并发布了包含多种可变形物体与四种机器人（人形、Franka臂、外科机器人、双臂YAM）的高保真仿真环境。

**💡 创新点**

核心创新在于：① 针对可变形物体设计基于节点点云的表示；② 采用非刚性配准技术实现源演示与新场景的轨迹适配；③ 通过此流水线在多机器人平台上实现了可变形物体的高精度、动态操纵任务的规模化数据生成。

**🔧 技术方法**

使用了：非刚性配准（求解平滑变形场）、轨迹重塑、仿真器Isaac Lab、Apple Vision Pro遥控采集、BC-RNN‑GMM和扩散策略等模仿学习方法，以及Point Bridge的点云域桥接技术进行 sim‑real 转移。

**📊 数据集**

数据集包括：① 1–3 条人类遥控演示（Apple Vision Pro）；② 通过流水线生成的每个任务约1,000 条合成演示；③ 在真实世界实验中收集的 30 条演示用于对比。

**📈 对比分析**

与仅使用源演示训练的策略相比，生成数据训练的策略成功率提升 25–97%；在零射模拟‑真实迁移实验中，1,000 条合成演示即可实现与 30 条真实演示相当甚至更好的表现；随着数据量增大，策略性能稳步提升。

**⚠️ 局限性**

局限性：流水线假设任务由固定顺序的子任务构成，无法处理需要多轮尝试或条件跳转的自由结构任务；对极端形变或高度动态场景的鲁棒性仍有限；在部分任务（如 YAM‑Towel）上成功率相对较低。

---

## 502. Out of Sight but Not Out of Mind: Hybrid Memory for Dynamic Video World Models

**arXiv ID:** 2603.25716 | [PDF](https://arxiv.org/pdf/2603.25716v1)

**作者:** Kaijin Chen `[一作]` (Huazhong University of Science and Technology), Xiang Bai `[通讯]` (Huazhong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出混合记忆（Hybrid Memory）框架，解决视频生成中背景与动态主体在视野外时的持续一致性问题。

**💡 创新点**

创新点在于：①引入混合记忆概念，要求模型同时记忆静态背景与动态主体；②构建专门的HM‑World数据集；③提出HyDRA记忆架构，使用记忆分词器与时空相关检索机制。

**🔧 技术方法**

技术上结合3D VAE、Diffusion Transformer、相机注入、3D卷积记忆分词器、动态检索注意力。

**📊 数据集**

使用自研HM‑World数据集，包含59K高质量视频，17场景、49主体、10轨迹、28摄像机轨迹。

**📈 对比分析**

与基线、DFoT、Context‑as‑Memory以及商业模型WorldPlay对比，HyDRA在PSNR、SSIM、DSC等指标均明显提升，尤其在动态主体一致性上表现最优。

**⚠️ 局限性**

局限在复杂场景多主体或严重遮挡时性能下降，未来需改进更鲁棒的记忆机制并推广至真实环境。

---

