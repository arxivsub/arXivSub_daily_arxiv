# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-02-02 | 今日论文总数: 591

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. ScribbleSense: Generative Scribble-Based Texture Editing with Intent Prediction

**arXiv ID:** 2601.22455 | [PDF](https://arxiv.org/pdf/2601.22455v1)

**作者:** Yudi Zhang `[一作]` (Beijing Institute of Technology), Lei Zhang `[通讯]` (Beijing Institute of Technology)

**通讯引用:** 105450 | [OpenAlex ID](https://openalex.org/A5100433899)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文提出了一种基于手绘涂鸦的三维纹理编辑框架ScribbleSense，能够在无需文字提示的情况下根据涂鸦颜色与形状自动理解编辑意图并生成高质量的局部纹理；

**💡 创新点**

创新点在于将多模态大语言模型（MLLM）与扩散模型相结合，先用MLLM预测涂鸦的语义意图，再生成全局提示并从全局图像中挑选最匹配的局部纹理，同时通过几何引导的遮罩细化进一步消除空间歧义；

**🔧 技术方法**

主要技术包括多模态大语言模型（如GPT‑4/InternVL3）用于意图预测与全局提示生成，Stable Diffusion SDXL做全局纹理生成，SAM（Segment Anything Model）与几何投影实现涂鸦区域遮罩细化，局部纹理拼接与稳定扩散的inpainting模型实现纹理融合；

**📊 数据集**

实验使用来自Sketchfab与Objaverse的24个三维纹理网格，配合32组涂鸦输入，评估真实用户意图的语义提示；

**📈 对比分析**

与基线方法TEXure、TEXGen、MagicQuill比较，使用CLIP‑Score、FID和用户偏好测评，ScribbleSense在CLIP‑Score（31.90）、FID（130.83）以及整体偏好（44.65%）上均超越对手，显示出显著的性能优势；

**⚠️ 局限性**

局限性包括局部纹理生成依赖涂鸦颜色与语义线索，难以充分利用复杂几何细节，UV岛碎片过多时纹理对齐效果不佳，且缺少基于深度的生成方案来进一步提升几何一致性。

---

## 2. Jailbreaks on Vision Language Model via Multimodal Reasoning

**arXiv ID:** 2601.22398 | [PDF](https://arxiv.org/pdf/2601.22398v1)

**作者:** Aarush Noheria `[一作]`, Yuguang Yao `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

论文提出了一种结合ReAct推理与自适应图像噪声的双模态越狱攻击框架，能在视觉语言模型中以更高的成功率生成逃避安全过滤的文本和图像。

**💡 创新点**

创新点在于将思考–行动–观察循环同时应用于文本重写和图像噪声调整，使攻击能够动态适应模型反馈并保持自然性；以及利用模型内部推理轨迹进行安全评分与黑盒审计。

**🔧 技术方法**

主要技术包括ReAct动态提示重写、基于类别的自适应图像噪声（高斯模糊、DCT、图像重色），以及双重安全评分体系（事实与反事实安全分）。

**📊 数据集**

使用了 VLGuard 与 SPA‑VL 两个公开安全对齐数据集，分别包含数千对图文样本与多类有害内容。

**📈 对比分析**

与传统的原始提示、静态重写、仅图像噪声等基线相比，复合ReAct+噪声策略在 SPA‑VL 有害子集和 VLGuard 违规子集上的攻击成功率提升至约52%–53%，显著高于其它方法。

**⚠️ 局限性**

局限性包括依赖于特定模型（Gemini‑2.0‑Flash）的内部推理轨迹，攻击效果在更严格的安全设置下尚未充分验证；同时，对图像重写的可解释性与对真实世界多样化图像的适应性仍需进一步研究。

---

## 3. Demystifying Mergeability: Interpretable Properties to Predict Model Merging Success

**arXiv ID:** 2601.22285 | [PDF](https://arxiv.org/pdf/2601.22285v1)

**作者:** Luca Zhou `[一作]` (Sapienza University of Rome), Emanuele Rodolà `[通讯]` (Sapienza University of Rome)

**通讯引用:** 6971 | [OpenAlex ID](https://openalex.org/A5087051832)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种可解释的线性框架，通过对28个可解释的对模型属性度量进行线性组合，预测不同模型合并方法的成功率，并揭示各方法的“成功指纹”。

**💡 创新点**

创新点在于：①将模型合并的可合并性从“内在属性”转变为与合并算法和任务对的相互依赖；②使用可解释的对模型度量（梯度距离、子空间重叠等）而非黑盒预测；③发现并验证了跨方法的稳定指标（梯度距离、主子空间重叠）。

**🔧 技术方法**

技术主要包括：线性优化求解最大化Pearson相关的系数，min‑max归一化，Leave‑One‑Task‑Out交叉验证，以及L1正则化稀疏化；同时利用CLIP ViT‑B/16模型提取任务向量、梯度和激活信息。

**📊 数据集**

使用20个公开图像分类任务（如CIFAR10/100、SVHN、MNIST、SUN397等）Fine‑tune CLIP ViT‑B/16得到20个任务模型，共190对模型。

**📈 对比分析**

与四种合并方法（Task Arithmetic、Weight Averaging、Task Singular Vectors、Isotropic）对比，线性模型在验证集上平均Pearson相关分别为0.34–0.57，预测效果与Mlp相当；同样方法的系数高度可解释，揭示了不同方法对梯度和子空间指标的不同权重。

**⚠️ 局限性**

局限性包括：仅在CLIP ViT‑B/16上验证，可能不适用于其他架构；任务数有限（20个），样本量不足以覆盖所有多任务场景；度量需校准数据，适用性受限；仅考察两模型合并，未扩展到多模型合并或更高阶干扰。

---

## 4. AsyncMesh: Fully Asynchronous Optimization for Data and Pipeline Parallelism

**arXiv ID:** 2601.22442 | [PDF](https://arxiv.org/pdf/2601.22442v1)

**作者:** Thalaiyasingam Ajanthan `[一作]` (Pluralis Research), Alexander Long `[通讯]` (Pluralis Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究在 2D 网格（DP × PP）中实现全异步训练，消除同步瓶颈并保持模型性能

**💡 创新点**

① 将异步 PP 与异步稀疏参数平均 DP 结合；② 设计基于 EMA 的延迟补偿机制，理论证明异步稀疏平均可收敛；③ 首次在完整异步设置下实现与同步版本相当的效果

**🔧 技术方法**

异步 PP（采用 NAG 延迟补偿）、稀疏参数平均（SPARTA）、EMA 估计平均延迟并补偿、Python/PyTorch 实现，结合理论收敛分析

**📊 数据集**

四大语言建模数据集（WT, BC, OWT, FW）以及 1B 参数的 Transformer 模型

**📈 对比分析**

与 FullSync、SyncDP、SPARTA 等同步/半同步基线比较；在多种网格和模型规模下，异步方法在消除 DP 通信后性能相当甚至略优，速度提升 1.5–3.7 倍

**⚠️ 局限性**

对稀疏平均子集比例≥40%会不稳定；理论假设同质设备、缓慢漂移的延迟；EMA 系数需调节；实验多在同一数据中心，极端低带宽环境需进一步验证

---

## 5. Optimization, Generalization and Differential Privacy Bounds for Gradient Descent on Kolmogorov-Arnold Networks

**arXiv ID:** 2601.22409 | [PDF](https://arxiv.org/pdf/2601.22409v1)

**作者:** Puyu Wang `[一作]` (RPTU Kaiserslautern-Landau), Marius Kloft `[通讯]` (RPTU Kaiserslautern-Landau)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

分析了梯度下降在两层Kolmogorov–Arnold网络（KAN）上的训练动态、泛化性能以及在差分隐私条件下的效用上限，给出了宽度、迭代次数与隐私预算之间的定量关系。

**💡 创新点**

提出了统一的参考点框架，对GD与DP‑GD分别给出了最优优化、1/n泛化和√(d)/(nε)隐私效用上限，并证明在私有训练中多项式对数宽度既是充分又必要的，揭示了非私有与私有训练在宽度需求上的本质差异。

**🔧 技术方法**

采用NTK分离假设、基于B‑spline边缘函数的两层KAN模型、梯度下降的自洽曲率控制、轨迹平均稳定性分析以及对DP‑GD的投影式噪声加噪机制等技术。

**📊 数据集**

在合成逻辑回归数据和MNIST手写数字数据上进行实验验证。

**📈 对比分析**

通过与非私有GD、不同宽度和迭代次数以及DP‑GD的实验对比训练/测试准确率和私有效用；实验表明宽度与迭代次数在理论预估的临界点后不再提升，私有效用在宽度与迭代次数达到一定范围后出现峰值并下降。

**⚠️ 局限性**

仅针对两层KAN，假设激活函数光滑且使用B‑spline边缘函数；未涵盖ReLU等非光滑激活；未对SGD等随机优化方法给出结论；隐私下的下界仅针对NTK可分离情形。

---

## 6. Accurate Pedestrian Tracking in Urban Canyons: A Multi-Modal Fusion Approach

**arXiv ID:** 2601.22406 | [PDF](https://arxiv.org/pdf/2601.22406v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 7. ShellForge: Adversarial Co-Evolution of Webshell Generation and Multi-View Detection for Robust Webshell Defense

**arXiv ID:** 2601.22182 | [PDF](https://arxiv.org/pdf/2601.22182v1)

**作者:** Yizhong Ding `[一作]` (Beijing Electronic Science and Technology Institute), Yizhong Ding `[通讯]` (Beijing Electronic Science and Technology Institute)

**通讯引用:** 44 | [OpenAlex ID](https://openalex.org/A5100523626)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

ShellForge 是一个 Webshell 生成与检测的共进化框架，通过交替更新生成器与检测器实现持续的安全强化。

**💡 创新点**

创新点包括：① 用生成器与检测器的双向硬样本交换实现共进化；② 引入 LLM 进行安全的去恶意转换以降低误报；③ 多视图融合检测（语义、结构、统计）克服长上下文与重混淆问题。

**🔧 技术方法**

技术包括：基于 Qwen2.5‑Coder 的强化学习生成器、CodeBERT + TextCNN 语义分支、Tree‑sitter AST 结构分支、熵等统计特征、LLM（Claude Sonnet、Gemini）做去恶意转换以及 PPO 风格的强化学习。

**📊 数据集**

数据集：FWOID 公共 PHP Webshell benchmark；以及共进化过程中产生的 EvoSet（约 1,138 变异 Webshell 与 1,090 去恶意样本），总计 6,139/7,026 例。

**📈 对比分析**

与 ShellPub、AST‑DF、FRF‑WD、GlareShell 等基线比较，ShellForge 在 FWOID 测试集上 F1 0.993，EvoSet 上 F1 0.981，显著提升检测精度和鲁棒性。

**⚠️ 局限性**

限制：依赖 LLM 的去恶意转换可能受模型对抗或策略限制；生成器仍基于已知 Webshell 语料，难以覆盖所有未知攻击变体；模型训练与推理成本相对较高。

---

## 8. Stop Jostling: Adaptive Negative Sampling Reduces the Marginalization of Low-Resource Language Tokens by Cross-Entropy Loss

**arXiv ID:** 2601.22439 | [PDF](https://arxiv.org/pdf/2601.22439v1)

**作者:** Galim Turumtaev `[一作]` `[通讯]`, Galim Turumtaev

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种通过对logits做阈值化来抑制低资源语言词汇在训练中被过度边缘化（marginalization）的技术，改善稀有词的表示并提升模型性能。

**💡 创新点**

创新点在于将负采样（negative sampling）用于减少稀有词的边缘化影响，而非传统的加速训练或提升正样本表现；阈值化后仅忽略概率低于目标词的一部分负样本，使得这些负样本的梯度为零，从而降低噪声。

**🔧 技术方法**

技术主要包括：logits阈值化（Thresholding Logits）、分离嵌入（Separated Embeddings）以防止AdamW的动量传播导致的全局更新、以及温度缩放和核采样等后处理方法来控制长尾效应。

**📊 数据集**

实验使用的是莎士比亚文本的字符级数据，构造了高资源（98%）与低资源（2%）两种语言，通过字符级词表模拟不同稀缺度；此外在GitHub上公开的实现基于Karpathy的nanoGPT。

**📈 对比分析**

与传统的基线GPT‑2架构（800k参数、权重共享）以及单语训练、仅阈值化或阈值化+分离嵌入的模型进行对比。结果显示：低资源语言的准确率从0.31提升至0.49，高资源语言保持≈0.53；PPL_best从10.63降至6.11，接近高资源的4.97；并且在召回@5、MRR和I(W)等指标上也有明显改善。

**⚠️ 局限性**

主要限制包括：仅在小规模字符级模型和极简数据集上验证，无法直接推断在大模型或真实多语言数据上的效果；阈值超参需要手工调节；长尾概率仍存在，可能影响PPL；且未对下游任务进行评估。

---

## 9. MERMAID: Memory-Enhanced Retrieval and Reasoning with Multi-Agent Iterative Knowledge Grounding for Veracity Assessment

**arXiv ID:** 2601.22361 | [PDF](https://arxiv.org/pdf/2601.22361v1)

**作者:** Yupeng Cao `[一作]` (Stevens Institute of Technology), K. P. Subbalakshmi `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个记忆增强的多智能体框架，用以在在线内容的真伪评估中实现检索与推理的紧耦合与跨命题证据复用。

**💡 创新点**

创新点包括：①将检索过程嵌入LLM的ReAct循环，允许动态调整查询；②引入持久化证据记忆，支持跨命题检索缓存；③采用结构化知识图和主题关键字辅助检索。

**🔧 技术方法**

技术手段：多智能体架构（Decomposer + Executor）；ReAct推理循环；MCP工具接口；基于实体键值的持久化证据记忆；多模态检索工具（Wiki、Google、Scholar、论文检索）。

**📊 数据集**

使用了五个基准数据集：FacTool‑QA、BingCheck、FactCheck‑Bench（LLM生成回答验证）以及HoVer、SciFact（自然命题验证）。

**📈 对比分析**

与现有基线（FacTool‑QA、FactCheck‑GPT、SAFE、FIRE、Self‑Ask、ProgramFC、FOLK）对比，取得所有数据集的宏观F1最高或相近水平，并且在检索次数上下降约15‑30%。

**⚠️ 局限性**

局限性：记忆模块仅基于字符串关键字匹配，缺乏语义检索；LLM在工具调用时仍可能出现冗余或失败；工具选择策略尚未最优。

---

## 10. Automating Forecasting Question Generation and Resolution for AI Evaluation

**arXiv ID:** 2601.22444 | [PDF](https://arxiv.org/pdf/2601.22444v1)

**作者:** Nikos I. Bosse `[一作]` (FutureSearch), Dan Schwarz `[通讯]` (FutureSearch)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建了一套基于LLM的自动生成与解决未来预测问题的系统，生成1499个多领域、可明确解决且难度可调的预测题并在后期自动完成解答。

**💡 创新点**

创新点在于：①利用可自适应的Web研究代理在种子基础上快速生成原型问题并精炼为可验证问题；②多轮专门验证器（质量、歧义、可解决性、难度）筛选；③嵌入+DBSCAN+LLM去重保证唯一性；④投票+抢票式争议解决提升自动解答准确率；⑤通过子问题分解进一步提升模型预测性能。

**🔧 技术方法**

核心技术：ReAct式LLM代理进行网页搜索与文本生成；多模态验证器（ReAct代理）对问题质量进行分级；嵌入模型(text‑embedding‑3‑large)与DBSCAN进行聚类去重；多模型投票解答策略；Brier评分及其校准/细化分解用于性能评估。

**📊 数据集**

数据来源：2500个种子——500来自Stockfisher的S&P 500公司收益预测；200来自Media Cloud新闻；1800来自GDELT事件；生成14073原型问题，经过筛选、去重后得到1499个最终问题；用于评估的解答与预测均基于实时Web数据。

**📈 对比分析**

比较方法：在相同问题集上让Gemini 3 Pro、GPT‑5、Gemini 2.5 Flash等多模型先做信息检索后做概率预测，计算Brier分数。结果显示更强模型性能更佳（Brier从0.179降至0.134），子问题分解将Brier从0.141降至0.132，且系统生成问题的否定率≈29%与Metaculus相当，错误率≈4.9%低于Metaculus。

**⚠️ 局限性**

限制：LLM在交互式网页、表单填写、动态加载内容及PDF提取方面仍有限；部分问题因数据不可公开或缺失导致无法解决；解答平均耗时较长；多模型投票仍易受共性错误影响；生成问题偏向当前热点，缺乏长期高影响领域的覆盖。

---

## 11. JAF: Judge Agent Forest

**arXiv ID:** 2601.22269 | [PDF](https://arxiv.org/pdf/2601.22269v1)

**作者:** Sahil Garg `[一作]` (Averlon), Vishal Agarwal `[通讯]` (Averlon)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Judge Agent Forest（JAF）框架，使判定代理在评估一组相关查询–响应对时可进行联合推理，取代传统的单实例评估。

**💡 创新点**

核心创新在于将判定过程从局部评估扩展为跨实例的协同判断，并通过可学习的局部敏感哈希（LSH）构造多尺度、可解释的邻域，从而实现信息流、贝叶斯传播和链式思考共享。

**🔧 技术方法**

采用信息论驱动的哈希学习、LLM 推理、随机森林式邻域采样、链式思考（CoT）共享、迭代自我修正和概率评估等技术，构建集成评估与改进流程。

**📊 数据集**

主要在云端漏洞分拣任务上进行实验，使用公开 CVE 数据、NVD、GHSA 等漏洞报告以及企业级虚拟机/容器/无服务器服务的资产与网络配置元数据。

**📈 对比分析**

与传统单实例判定、BoT-R、CARE 等方法对比，JAF 在漏洞优先级排序准确率、误报率、调优成本和一致性指标上提升 4–12%（具体数值依赖实验细节），且计算成本保持可接受。

**⚠️ 局限性**

局限性包括哈希函数设计需要手工调参、对动态环境的适应性受限、判定过程仍受 LLM 生成不确定性的影响，以及对极端稀有场景的覆盖率有限。

---

## 12. Flexible FTN-OTFS for High-Mobility LEO Satellite-to-Ground Communication

**arXiv ID:** 2601.22526 | [PDF](https://arxiv.org/pdf/2601.22526v1)

**作者:** Chaorong Zhang `[一作]` (Macao Polytechnic University), Halim Yanikomeroglu `[通讯]` (Carleton University)

**通讯引用:** 20712 | [OpenAlex ID](https://openalex.org/A5035446029)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

提出一种面向低轨卫星的轻量化、基于OTFS的可调节FTN传输方案，在高速移动环境下通过SNR感知的LUT动态调整压缩因子，实现高效、可靠的链路传输。

**💡 创新点**

创新点在于：①将OTFS与FTN结合，突破Nyquist限制；②设计SNR感知的灵活FTN（FFTN）策略，利用低复杂度LUT实时自适应压缩因子；③在真实3GPP TDL通道模型下推导有效吞吐、能效与BER理论表达式，并验证自适应方案在不同SNR区间的性能提升。

**🔧 技术方法**

核心技术包括：OTFS调制、Faster‑than‑Nyquist时域压缩、根升余弦脉冲成形、3GPP TR 38.811 TDL通道模型、LMMSE线性检测、LUT式自适应压缩因子调度、能效与吞吐量的理论分析。

**📊 数据集**

采用3GPP TR 38.811标准的TDL‑A~E三路径模型（NLOS至LOS），Ka‑波段28 GHz，卫星高度780 km，系统带宽10 MHz，子载波间隔15 kHz，进行Monte‑Carlo仿真，未使用公开真实数据集。

**📈 对比分析**

通过与固定FTN（α = 1.0、0.9、0.8）以及纯OTFS基准对比，仿真显示自适应FFTN在低SNR下保持可靠性，在中高SNR区段实现最高25%吞吐提升，且BER误差阶趋于固定FTN。实验亦评估了CSI误差对性能的影响，结果表明对中等估计误差鲁棒。

**⚠️ 局限性**

局限性包括：①假设CSI完美，实际估计误差会削弱性能；②对大帧尺寸（如M = N = 64）时对SNR误差更敏感；③仅考虑SISO链路，未讨论多天线或链路层优化；④高SNR下仍存在残留ISI导致误差地板，需进一步改进检测或信道预编码。

---

## 13. Learning Policy Representations for Steerable Behavior Synthesis

**arXiv ID:** 2601.22350 | [PDF](https://arxiv.org/pdf/2601.22350v1)

**作者:** Beiming Li `[一作]` (University of Pennsylvania), Alejandro Ribeiro `[通讯]` (University of Pennsylvania)

**通讯引用:** 16135 | [OpenAlex ID](https://openalex.org/A5078862959)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种将马尔可夫决策过程中的策略映射为期望特征向量的表示方法，并在此基础上构建可解码的潜在空间，实现对策略的重建与梯度搜索，支持测试时的行为合成。

**💡 创新点**

创新点包括：①将策略表征为占优度测度下的特征期望；②将变分自编码器与Rank‑N‑Contrast对比学习结合，生成既平滑又与价值函数对齐的潜在空间；③采用局部切线投影的原始–对偶优化实现零样本约束行为合成。

**🔧 技术方法**

主要技术：β‑VAE、基于Transformer的集成编码器、Rank‑N‑Contrast对比损失、半正交投影、基于价值预测器的梯度投影优化。

**📊 数据集**

使用多目标MuJoCo控制任务（如HalfCheetah、Ant等）生成的PPO策略数据集，包含多种奖励权重下的轨迹。

**📈 对比分析**

与VAE基线、VAE+InfoNCE、AE+RNC等方法对比，指标包括返回回归MSE、行为合成成功率、目标误差和约束违例。本文模型在MSE上下降约5×，合成成功率高达98%，目标误差约11.7%，约束违例仅0.13%，明显优于对照组。

**⚠️ 局限性**

局限性：依赖于覆盖广泛的策略样本；价值预测器误差可能导致梯度误导；投影优化仍需手工设定邻域尺寸与步长；目前仅验证于连续控制任务，扩展至更复杂环境或离散动作空间尚待研究。

---

## 14. ZK-HybridFL: Zero-Knowledge Proof-Enhanced Hybrid Ledger for Federated Learning

**arXiv ID:** 2601.22302 | [PDF](https://arxiv.org/pdf/2601.22302v1)

**作者:** Amirhossein Taherpour `[一作]` (Columbia University), Xiaodong Wang `[通讯]` (Columbia University)

**通讯引用:** 77533 | [OpenAlex ID](https://openalex.org/A5100382645)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于DAG账本、侧链智能合约和零知识证明的去中心化联邦学习框架ZK-HybridFL，实现在保持数据隐私的同时对模型更新进行公开验证与安全聚合。

**💡 创新点**

创新点在于将零知识证明嵌入区块链验证流程、引入loss-aware DAG父链选择、使用挑战机制清除孤儿攻击，并通过侧链事件驱动实现高吞吐量。

**🔧 技术方法**

采用了DAG共识、侧链事件驱动智能合约（EDSC）、Groth16零知识证明、Bulletproof和Lamport时间戳等技术。

**📊 数据集**

实验使用MNIST（图像分类）和Penn Treebank（文本预测）两个公开数据集，均在多节点、不同攻击率下进行。

**📈 对比分析**

与Blade-FL和ChainFL相比，ZK-HybridFL在准确率/困惑度上提升了5-10个百分点，收敛速度快，单轮延迟约7-8 s，吞吐量最高；在攻击节点比例上也更稳健。

**⚠️ 局限性**

局限性包括零知识证明生成的计算成本、对大模型的证明大小与GPU内存需求、以及对侧链事件顺序的依赖等问题。

---

## 15. MIRRORTALK: Forging Personalized Avatars Via Disentangled Style and Hierarchical Motion Control

**arXiv ID:** 2601.22501 | [PDF](https://arxiv.org/pdf/2601.22501v1)

**作者:** Renjie Lu `[一作]` (Ping An Technology), Shangfei Wang `[通讯]` (University of Science and Technology of China)

**通讯引用:** 3904 | [OpenAlex ID](https://openalex.org/A5077046519)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

设计了一种基于条件扩散模型的生成框架 MirrorTalk，用于生成与任意音频同步、同时保留说话人独特说话风格的个人化对话视频。

**💡 创新点**

核心创新点包括：
• 两阶段跨模态训练的 Semantically‑Disentangled Style Encoder（SDSE），能够从极短的参考视频中提取纯净、与语义无关的说话风格表征；
• 空间‑时间层次化调制策略，在扩散过程中动态平衡音频与风格特征在上半脸与下半脸的贡献，从而既保证精准口型同步，又保持风格表达；
• 结合 HSIC、正交化约束和三元组损失实现风格与语义的显式解耦与区分；
• 引入记忆库和自注意力池化提高语义对齐与风格抽取的鲁棒性。

**🔧 技术方法**

使用的技术与方法有：
- 3D 形状模型 FLAME、SMIRK、MICA、3DDFA 进行面部几何估计；
- 变形器 Transformer 作为 SDSE 的 backbone；
- 预训练 Motion Expert（Wav2Lip 的嘴唇同步判别器）做语义监督；
- 扩散变压器（DiT）作为生成器；
- 空间‑时间层次化交叉注意力调制；
- 记忆库、HSIC、正交化、三元组损失进行风格解耦与区分；
- 神经渲染器将生成的表情参数映射为最终视频帧。

**📊 数据集**

实验数据集包括：
- VoxCeleb2（1M+ 语音视频），
- HDTF（高分辨率 16 小时视频），
- CREMA‑D（情感对话视频）。

**📈 对比分析**

与多种 state‑of‑the‑art 基线（Wav2Lip、EAMM、AniTalker、SadTalker、Echomimic、V‑Express）在视觉质量（SSIM/FID）、口型同步（M‑LMD、Sync_conf）以及风格保留（StyleSim）等指标上进行了全面对比。MirrorTalk 在大多数指标上均取得显著提升：SSIM 与 FID 更佳，M‑LMD 下降，Sync_conf 上升，StyleSim 最高，表明既能保持同步又能保留说话人个性。

**⚠️ 局限性**

主要局限包括：
1. 需要一段参考视频来提取风格，短视频或无风格表现时效果可能下降；
2. 对极端姿态、光照或遮挡的鲁棒性尚未充分验证；
3. 3D 估计与渲染过程增加了计算负担，实时性能受限；
4. 层次调制策略需手工设计区域划分，适配不同面部结构可能需要进一步调整。

---

## 16. Beyond Activation Patterns: A Weight-Based Out-of-Context Explanation of Sparse Autoencoder Features

**arXiv ID:** 2601.22447 | [PDF](https://arxiv.org/pdf/2601.22447v1)

**作者:** Yiting Liu `[一作]` (Peking University), Zhi-Hong Deng `[通讯]` (Peking University)

**通讯引用:** 5334 | [OpenAlex ID](https://openalex.org/A5060171279)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043`

**🎯 论文内容**

本文提出了一种基于权重的稀疏自编码器（SAE）特征解释框架，能够在不依赖激活模式的情况下揭示特征的计算功能。

**💡 创新点**

创新点在于利用直接权重交互来衡量特征对输出的因果影响，并通过多指标筛选发现了语义特征与注意力参与特征在网络深度上的分布规律。

**🔧 技术方法**

主要技术包括SAE稀疏重构、logit lens、Levenshtein距离/余弦相似度/熵阈值判定、预/后softmax注意力矩阵分析以及分位数阈值设定。

**📊 数据集**

使用的数据集为Gemma‑2（2B与9B）和Llama‑3.1‑8B预训练模型中的SAE权重。

**📈 对比分析**

与传统激活模式解释相比，该方法发现约25%的特征具有可解释的语义输出预测能力，并且在模型中呈现U形/倒U形的深度分布，验证了其在不同层次上的有效性。

**⚠️ 局限性**

局限性包括对特定SAE架构的依赖、阈值和指标选择的敏感性，以及对未关联嵌入/解嵌入结构的模型适用性可能受限。

---

## 17. Coarse-to-Real: Generative Rendering for Populated Dynamic Scenes

**arXiv ID:** 2601.22301 | [PDF](https://arxiv.org/pdf/2601.22301v1)

**作者:** Gonzalo Gomez-Nogales `[一作]` (Universidad Rey Juan Carlos), Yi Zhou `[通讯]` (Roblox)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

该论文提出了一种将粗糙 3D 控制视频通过文本提示转化为逼真城市人群视频的生成渲染框架。

**💡 创新点**

创新点在于：①采用两阶段混合 CG–真实训练策略，先学习真实分布，再用隐式时空特征实现可控；②利用 DINOv3 的无监督特征作为跨域控制信号，避免显式几何约束导致的失真；③通过 HSV 去相关化与自适应提示引导提升生成自由度与结构一致性。

**🔧 技术方法**

核心技术包括 WAN‑2.1 Diffusion Transformer、T5‑XXL 文本编码、DINOv3 视觉特征、Flow‑Matching 采样、HSV 颜色去相关化、Adaptive Prompt Guidance 以及轻量级特征投影 Adapter。

**📊 数据集**

使用了约 240k 条多国多环境的真实城市视频数据与 1.3k 条配对的 CG 合成样本，覆盖不同气候、光照与人群服饰等多样性。

**📈 对比分析**

与公开的 Wan‑2.1+ControlNet 基线对比，C2R 在 VE‑Bench 与 VQAScore 上均取得更高分数；视觉上生成的视频更具纹理细节、光照真实性且对粗糙 3D 结构保持更好的一致性。

**⚠️ 局限性**

在极其粗糙或抽象的 3D 输入下，模型难以准确还原建筑布局与摄像机轨迹，导致生成场景与输入结构偏离；未来可引入更显式的摄像机/运动约束以提升可控性。

---

## 18. Whispers of Wealth: Red-Teaming Google's Agent Payments Protocol via Prompt Injection

**arXiv ID:** 2601.22569 | [PDF](https://arxiv.org/pdf/2601.22569v1)

**作者:** Tanusree Debi `[一作]` (University of Georgia), Wentian Zhu `[通讯]` (University of Georgia)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对AP2协议进行AI红队评估，提出并实验验证两种提示注入攻击（品牌耳语攻击与金库耳语攻击），展示其对支付系统的破坏。

**💡 创新点**

首次系统性评估AP2安全性，定义两种新的基于提示注入的攻击方式，并在完整功能代理上实验验证其有效性。

**🔧 技术方法**

采用Gemini‑2.5‑Flash LLM、Google ADK框架构建多代理购物系统，实施直接与间接提示注入，使用加密签名的授权令牌和可验证凭证。

**📊 数据集**

使用人工合成的虚拟商品信息和用户记录（无真实敏感数据）。

**📈 对比分析**

通过对照正常运行与攻击运行的候选商品列表与数据泄露情况，攻击在实验中始终成功，展示了系统对提示注入的极易受损。

**⚠️ 局限性**

评估仅覆盖两种攻击且基于小规模合成场景，缺乏对更广泛攻击类型、真实数据和量化成功率的深入分析。

---

## 19. HeaPA: Difficulty-Aware Heap Sampling and On-Policy Query Augmentation for LLM Reinforcement Learning

**arXiv ID:** 2601.22448 | [PDF](https://arxiv.org/pdf/2601.22448v1)

**作者:** Weiqi Wang `[一作]` (Amazon Inc.), Yangqiu Song `[通讯]` (Amazon Inc.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 HeaPA 框架，在 RLVR 训练中维护有限且不断演化的 prompt 池，利用堆边界采样聚焦能力前沿，并通过轻量化异步验证的 on‑policy 增量生成新的可验证 prompt，稳定地回收与重插。

**💡 创新点**

创新点包括：①堆边界采样（heap-based boundary sampling）把采样聚焦在中等难度区；②基于当前策略的在线 prompt 增量与异步验证；③拓扑感知的池统计重估与受控重插，缓解相关增量导致的课程振荡。

**🔧 技术方法**

使用技术：RLVR 的 group‑based PPO（GRPO/DAPO）优化；双堆（low / high）数据结构实现 bounded pool；异步教师验证；拓扑传播统计更新；受控回收机制。

**📊 数据集**

数据集：种子数据使用 DAPO‑Math（约 14K 题）和 OpenR1‑Math‑220k（约 93K 题）；评估基准包括 AIME24/25、AMC23、GPQA、MATH500、MinervaMath、OlympiadBench。

**📈 对比分析**

比较方法：将 HeaPA 插入 GRPO、DAPO、Reinforce‑Ada 等原始管线；在 7 个基准上平均提升约 3–5 分；在 Qwen3 各规模模型上提升可达 7+ 分；相较于优先采样或单独增量效果更显著，计算开销仅增加约 2% 的时钟延迟。

**⚠️ 局限性**

局限性：仍需教师模型进行验证，增量生成与验证存在异步延迟；堆与回收管理在极大模型或极大数据集上复杂度上升；未充分探讨对非数值答案或多模态 prompt 的适用性。

---

## 20. Stability-Aware Prompt Optimization for Clinical Data Abstraction

**arXiv ID:** 2601.22373 | [PDF](https://arxiv.org/pdf/2601.22373v1)

**作者:** Arinbjörn Kolbeinsson `[一作]` (Century Health), Sanjay Hariharan `[通讯]` (Century Health)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

研究临床文本抽取任务中大语言模型对提示词敏感度与不确定性联合影响，提出双目标提示优化循环；

**💡 创新点**

首次将提示敏感度作为独立目标与准确度共同优化，证明稳定性可显著提升并与模型置信度相关；

**🔧 技术方法**

使用LLM‑in‑the‑loop优化（生成候选提示并评估准确率+翻转率），结合概率、共形预测、可靠性校准等不确定性度量；

**📊 数据集**

在MedAlign（适用性与正确性二分类）与内部MS子类型抽取（6分类）两大医学表格化任务上评估；

**📈 对比分析**

对比单目标准确度优化与双目标优化，结果显示在9种模型‑任务组合中，加入稳定性项后翻转率下降约70%（平均），准确度变动较小；

**⚠️ 局限性**

实验规模有限（50–200例，3个提示变体），生成的提示变体不完全代表部署多样性，且仅评估了单一优化策略，缺乏下游临床效益验证；

---

## 21. RulePlanner: All-in-One Reinforcement Learner for Unifying Design Rules in 3D Floorplanning

**arXiv ID:** 2601.22476 | [PDF](https://arxiv.org/pdf/2601.22476v1)

**作者:** Ruizhe Zhong `[一作]` (Shanghai Jiao Tong University), Junchi Yan `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 16585 | [OpenAlex ID](https://openalex.org/A5087158377)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了一个统一的深度强化学习框架，用于同时满足3D floorplanning 中多项硬件设计规则，避免人工后处理。

**💡 创新点**

核心创新包括：① 用邻接终端/模块矩阵对设计规则进行统一表征；② 在动作空间直接加约束过滤非法位置，显著减小搜索空间；③ 将量化的规则违规指标融入奖励，支持七项以上工业规则且易于扩展。

**🔧 技术方法**

采用 Actor‑Critic 与 Hybrid PPO 的强化学习策略，使用 Transformer 图编码器 + CNN 视觉特征；动作空间为混合离散（位置）与连续（纵横比）；通过可用性掩码对动作空间施加硬约束，奖励重塑保证训练稳定。

**📊 数据集**

在公开 MCNC 与 GSRC benchmark（10–300 模块电路）上进行实验，并针对包含边界、分组、预置等多种任务进行评估。

**📈 对比分析**

与解析、启发式和其他 RL 方法（GraphPlace、DeepPlace、MaskPlace、FlexPlanner、RulePlanner）对比，实验表明在块-终端距离、块-块相邻长度等关键规则指标上几乎完美满足，优于基线；在 HPWL 等传统指标上略逊，但整体更符合工业约束。

**⚠️ 局限性**

限制在于未考虑热量/功耗优化，且验证范围仅限公开 benchmark，真实工艺和更大规模芯片的鲁棒性仍需进一步研究。

---

## 22. Fairness-Aware Performance Evaluation for Multi-Party Multi-Objective Optimization

**arXiv ID:** 2601.22497 | [PDF](https://arxiv.org/pdf/2601.22497v1)

**作者:** Zifan Zhao `[一作]` (Nanjing University of Information Science and Technology), Wenjian Luo `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 2930 | [OpenAlex ID](https://openalex.org/A5001184471)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种基于Nash乘积的公平感知评估框架，用于多方多目标优化（MPMOP）的解集评估。

**💡 创新点**

创新点在于①将公平性用四条公理（Pareto单调性、对称性、平衡偏好、可接受性单调性）形式化；②引入“让步率”概念，将传统的完全共享Pareto解扩展为可接受区域；③将传统指标（IGD、HV）转换为效用后通过Nash乘积聚合，天然兼顾效率与公平。

**🔧 技术方法**

技术手段包括：公理化分析、Nash乘积聚合、让步率/惩罚机制、实验验证与对比、PF‑free 归一化指标。

**📊 数据集**

使用了扩展的MPMOP1–17、MPDMP1–18以及UAV路径规划实例（MPUAV1–6）等公开和自构造的基准数据集。

**📈 对比分析**

与传统的 meanIGD/meanHV 对比；实验表明在大多数问题上，两种评估会给出不同的算法排名，证明该框架更能捕捉多方公平性。整体性能在共识解存在时更好，非共识解时通过惩罚提升公平度。

**⚠️ 局限性**

局限性：需要已知Pareto前沿或设定让步阈值；PF‑free 情况下的公平性判定仍不完善；惩罚参数的选择可能影响评估结果。

---

## 23. VocBulwark: Towards Practical Generative Speech Watermarking via Additional-Parameter Injection

**arXiv ID:** 2601.22556 | [PDF](https://arxiv.org/pdf/2601.22556v1)

**作者:** Weizhi Liu `[一作]` (East China Normal University), Zhaoxia Yin `[通讯]` (East China Normal University)

**通讯引用:** 3213 | [OpenAlex ID](https://openalex.org/A5035489942)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

设计了一种基于额外参数注入的语音生成水印框架 VocBulwark，实现在保持语音质量的同时对生成模型进行水印嵌入与提取。

**💡 创新点**

通过冻结主模型参数并引入轻量化时间适配器（Temporal Adapter）将水印深度融合到声学特征，再结合粗细分级门控提取器（Coarse-to-Fine Gated Extractor）以及准确度引导优化课程，兼顾高保真与高鲁棒性。

**🔧 技术方法**

采用额外参数注入、声学特征对齐、帧级广播、深度可分离卷积、门控分离卷积、多尺度特征聚合、攻击仿真以及准确度引导的多目标训练策略。

**📊 数据集**

主要使用 LJSpeech 进行训练，并在跨域 LibriTTS 与 AiShell3 数据集上进行鲁棒性与泛化性评估。

**📈 对比分析**

与 RIWF、HiFi-GANw、Groot 等生成水印方法以及 Normspace、FSVC、PBML、AudioSeal、WavMark、TBWM 等后置水印基线在 STIO、PESQ、SSIM、MCD、DNSMOS 等保真指标和 ACC 等鲁棒指标上对比，VocBulwark 在保真上保持或略优于基线，在鲁棒性上达到近 99% 以上的准确率，显著优于现有方案。

**⚠️ 局限性**

在极高容量（>2000 bps）下音质略有下降；目前仅在无先验攻击强度的黑盒环境下验证，对模型迁移、多语言泛化以及资源受限设备的部署仍需进一步研究。

---

## 24. PersonaCite: VoC-Grounded Interviewable Agentic Synthetic AI Personas for Verifiable User and Design Research

**arXiv ID:** 2601.22288 | [PDF](https://arxiv.org/pdf/2601.22288v1)

**作者:** Mario Truss `[一作]` (Adobe), Mario Truss `[通讯]` (Adobe)

**通讯引用:** 5 | [OpenAlex ID](https://openalex.org/A5093931911)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种VoC数据驱动的交互式人工智能人设系统，能够在对话中实时检索并引用真实用户证据，若缺乏证据则主动放弃回答。

**💡 创新点**

创新点在于将人工智能人设从基于提示的角色扮演转变为检索增强的可验证研究工具，结合了即时证据检索、显式拒答和回答级别来源标注。

**🔧 技术方法**

采用了Gemini和GPT‑4O作为核心推理引擎，并使用Python、Pydantic、AI.SDK、Next.js实现代理式上下文工程（ACE）。

**📊 数据集**

数据集来源为多模态VoC数据，包括社交媒体文本、图像、视频转录等公开渠道的用户生成内容。

**📈 对比分析**

通过与14名行业专家进行半结构化访谈和迭代评估，未对系统进行量化基准测试；评估主要基于专家对可信度、透明度和设计价值的主观感知。

**⚠️ 局限性**

局限性包括样本规模有限、未进行客观事实准确性验证、缺乏与传统或其他人设方法的对比基准，以及对因果推断等更深层洞察的功能尚未实现。

---

## 25. Action-Sufficient Goal Representations

**arXiv ID:** 2601.22496 | [PDF](https://arxiv.org/pdf/2601.22496v1)

**作者:** Jinu Hyeon `[一作]` (Seoul National University), Taesup Moon `[通讯]` (Seoul National University)

**通讯引用:** 2634 | [OpenAlex ID](https://openalex.org/A5080346989)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出“动作足够性（action sufficiency）”概念，并证明在离线目标条件强化学习中仅靠价值函数学习的目标表示往往不具备动作足够性，导致低层控制失效；通过在低层策略训练中联合学习目标编码器实现动作足够性；

**💡 创新点**

核心创新是将动作足够性作为评估目标表示的理论标准，证明价值足够性不等价于动作足够性；并提出基于动作似然（actor‑based）目标编码器的简单实现，理论上可近似满足动作足够性；

**🔧 技术方法**

信息论分析（互信息、条件KL分解）、优势加权回归（AWR）、离线强化学习框架HIQL以及Hierarchical Flow（H–Flow）等；

**📊 数据集**

在OGBench离线数据集上，针对高难度物体摆放任务（如 2c、3c 等）；

**📈 对比分析**

与传统基于价值的表示（ϕ_V）以及无编码器、以及其他基线（GCIQL、GCIVL）对比；实验显示 actor‑based 表示在多数任务中显著提升成功率（如 2c 任务从 0.26→0.42、3c 从 0.01→0.28 等），尤其在高难度任务中优势更为明显；

**⚠️ 局限性**

理论与实践之间的差距（离线数据覆盖不足可能导致足够性下降），以及动作足够性虽对低层重要，但未必能保证整体层级规划的可压缩性与可预测性，需要进一步平衡。

---

## 26. Learning to Defer in Non-Stationary Time Series via Switching State-Space Models

**arXiv ID:** 2601.22538 | [PDF](https://arxiv.org/pdf/2601.22538v1)

**作者:** Yannis Montreuil `[一作]` (National University of Singapore), Wei Tsang Ooi `[通讯]` (Institute for Infocomm Research)

**通讯引用:** 2799 | [OpenAlex ID](https://openalex.org/A5072587271)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种面向非平稳时间序列的在线学习‑to‑defer框架，在每一步仅能查询部分专家并且专家集合可动态变化的环境下实现自适应路由。

**💡 创新点**

核心创新包括：①将专家残差建模为分层的切换线性高斯状态空间模型（L2D‑SLDS），其中引入共享全局因子以实现跨专家信息传递；②利用上下文驱动的状态转移矩阵实现情景切换；③在此概率模型下设计基于信息导向采样（IDS）的探索策略；④引入动态注册表机制实现专家的增删与状态缓存。

**🔧 技术方法**

使用技术包括：切换线性动态系统（SLDS）+ Kalman 过滤 + 交互式多模型（IMM）递归；信息导向采样（IDS）启发式决策；Monte Carlo 估计状态信息增益；对比实验使用 LinUCB、NeuralUCB、随机及固定专家策略。

**📊 数据集**

实验数据集：①仿真生成的情景相关性实验；②澳大利亚墨尔本每日最低气温；③附录中的 FRED DGS10 系列。

**📈 对比分析**

与基线比较时，L2D‑SLDS 在所有实验中均优于 LinUCB、NeuralUCB、随机及固定专家策略，尤其在有共享因子的版本下相较无共享因子和基线提升显著；在墨尔本数据集上相对基线提升约 10%‑15%。

**⚠️ 局限性**

局限性包括：模型假设为线性高斯，可能对高度非线性或离散事件的适用性有限；需要手动设定隐藏维度、切换次数、稀疏度阈值等超参数；在极大专家集合下，IMM 递归和 Monte Carlo 估计的计算开销仍然较高。

---

## 27. SP^2DPO: An LLM-assisted Semantic Per-Pair DPO Generalization

**arXiv ID:** 2601.22385 | [PDF](https://arxiv.org/pdf/2601.22385v1)

**作者:** Chaoyue He `[一作]` (Alibaba NTU Global e-Sustainability CorpLab), Chunyan Miao `[通讯]` (Alibaba)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

基于教师LLM的语义注释，将每个偏好对的温度设为实例级的β_i，改写标准DPO训练过程，使对齐控制从超参数转化为可审计的数据工件。

**💡 创新点**

1) 将全局温度替换为预先离线决定的实例级温度；2) 理论证明实例温度与加权方案不等价；3) 通过多提示、多教师集成构造鲁棒的β_i。

**🔧 技术方法**

DPO目标、教师LLM语义注释（类别、幅度、置信度）、确定性β映射、对齐过程优化(APO)框架、单/多提示自集成与多注释器集成。

**📊 数据集**

UltraFeedback偏好对语料（59,960对）用于训练，AlpacaEval 2.0长度控制评测集用于性能比较。

**📈 对比分析**

与手工调优的全局β DPO以及随机β_i做对比；在四个指令调优开源模型上，实例温度DPO与全局β相当，部分模型提升长度控制胜率约1–2个百分点。

**⚠️ 局限性**

仅在AE2上评估，未覆盖安全、工具使用等；依赖教师LLM的质量和提示设计；β_i为固定离线值，无法随训练动态调整；潜在的偏见、隐私及可访问性限制。

---

## 28. Causal Imitation Learning Under Measurement Error and Distribution Shift

**arXiv ID:** 2601.22206 | [PDF](https://arxiv.org/pdf/2601.22206v1)

**作者:** Shi Bo `[一作]` (Boston University), AmirEmad Ghassami `[通讯]` (Boston University)

**通讯引用:** 327 | [OpenAlex ID](https://openalex.org/A5078609284)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

本文研究在离线模仿学习中，存在状态测量误差且训练与部署环境分布可能发生漂移的情形，并提出一种基于因果推断的目标策略π_opt，能够在不直接观测潜在状态的情况下实现稳健的策略学习。

**💡 创新点**

创新点在于：①将测量误差视为代理变量，利用“proximal causal inference”框架对可观测的时间序列数据进行识别，推导出在分布漂移下仍保持不变的因果最优策略；②给出了离散和连续两种情形的可识别性条件与相应的估计方法；③通过对抗式RKHS方法解决连续状态下的Fredholm积分方程，实现无奖励离线策略估计。

**🔧 技术方法**

核心技术包括：因果图模型、代理变量与桥接函数（confounding bridge）、矩阵逆识别、RKHS对抗估计、离线数据聚合与核方法、正则化极小极大优化。

**📊 数据集**

实验数据主要来自PhysioNet/Computing in Cardiology Challenge 2019的ICU患者时序记录，构造了半模拟的“血流动力学支持强度”决策任务，并通过模拟专家规则产生动作；另外还进行了基于仿真的模拟实验。

**📈 对比分析**

与传统的行为克隆（BC1：仅用S，BC2：用S和前一次代理W）以及其它基准方法对比。实验表明，π_opt在测量误差和人口分布漂移下的平均方差误差（MSE）显著低于BC基线，并保持相对稳定；BC2对测量误差尤其敏感。

**⚠️ 局限性**

局限性包括：①需要满足代理变量可逆性、完整性或桥接函数存在等较强假设；②离散情况需设计可逆的 coarsening，连续情况对核函数与正则化参数的选择敏感；③对真实临床决策的模拟程度有限，未验证在完全真实动作数据上的性能；④计算复杂度较高，尤其是大规模连续状态时的核矩阵运算。

---

## 29. Training-Free Representation Guidance for Diffusion Models with a Representation Alignment Projector

**arXiv ID:** 2601.22468 | [PDF](https://arxiv.org/pdf/2601.22468v1)

**作者:** Wenqiang Zu `[一作]` (Institute of Automation Chinese Academy of Sciences), Lei Ma `[通讯]` (Peking University)

**通讯引用:** 15148 | [OpenAlex ID](https://openalex.org/A5022157306)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出在扩散变压器（SiT、REPA）推断阶段通过注入自监督表示（如DINOv2）来进行语义引导（R‑Pred）

**💡 创新点**

创新点在于利用预训练的投影器预测的表示作为中间采样步骤的语义锚点，且不需要修改模型结构

**🔧 技术方法**

采用的技术包括扩散模型的ODE/SDE采样、梯度引导、投影器表示预测以及与现有指导方法（CFG、RepG等）的结合

**📊 数据集**

实验数据集为ImageNet 256×256图像，生成50,000张样本进行评估

**📈 对比分析**

与基线和主流指导方法对比，R‑Pred在多尺度模型上显著降低FID（如REPA‑XL/2从6.82降至3.34，SIT‑XL/2+CFG从2.15降至2.08），并提升IS

**⚠️ 局限性**

局限性在于需要在推断过程中进行梯度计算，导致额外计算开销，且目前仍依赖梯度引导，未来可探索更高效的引导方案

---

## 30. Lost in Space? Vision-Language Models Struggle with Relative Camera Pose Estimation

**arXiv ID:** 2601.22228 | [PDF](https://arxiv.org/pdf/2601.22228v1)

**作者:** Ken Deng `[一作]` (University of Oxford), Yftah Ziser `[通讯]` (University of Groningen)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究视觉语言模型在相对相机姿态估计（RCPE）任务中的表现，提出了基于无标签前瞻视频的LIVS基准和单自由度诊断基准，用于评估多视角3D空间推理能力。

**💡 创新点**

创新点在于将RCPE转化为离散分类任务，构建真实场景下的多自由度与单自由度评测集，并揭示VLM在光学轴运动（深度平移与滚转）上的显著薄弱。

**🔧 技术方法**

使用多种流行的视觉语言模型（LLaVA、Qwen、GPT‑4o、GPT‑5等）以及传统几何方法（SIFT、LoFTR）进行对比，并通过一致性、诊断和误差分析评估模型性能。

**📊 数据集**

采用7 Scenes、ScanNet和ScanNet++三大RGB‑D视频数据集，自动抽取相同中心物体的图像对并生成语义化的相机运动描述。

**📈 对比分析**

与几何基线和人类评估对比，VLM平均F1约为0.64，远低于LoFTR的0.97和人类的0.92；在图像顺序一致性测试中仅达到59.7%，表明VLM在3D推理与多视角一致性上仍显不足。

**⚠️ 局限性**

主要局限在于基准规模有限、自动生成的注释可能带噪声、未探究模型微调提升效果，以及评估仅覆盖公开VLM，缺乏专用空间推理模型的测试。

---

## 31. Towards Solving the Gilbert-Pollak Conjecture via Large Language Models

**arXiv ID:** 2601.22365 | [PDF](https://arxiv.org/pdf/2601.22365v1)

**作者:** Yisi Ke `[一作]` (Peking University), Liwei Wang `[通讯]` (Peking University)

**通讯引用:** 13272 | [OpenAlex ID](https://openalex.org/A5100406718)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

利用大型语言模型生成可执行的几何lemma，并将其转化为可验证函数，经过迭代反射优化后将Steiner比例的下界从0.824提升至0.8559。

**💡 创新点**

创新点在于：①将原始证明拆解为规则约束的代码lemma；②提出可验证函数并利用顶点最大性把连续空间验证简化为有限点检验；③引入瓶颈区反射机制，指引模型针对性生成新的lemma以突破局部最优。

**🔧 技术方法**

技术手段包括：大语言模型（GPT‑5 / Gemini 3 Pro）自动生成代码lemma；Mathematica符号与数值计算验证；分支限界（branch‑and‑bound）求解下界；强化学习式的迭代闭环与反射回传；高效的多线程计算实现。

**📊 数据集**

使用的数据集主要是内部构造的几何参数空间（轴对齐超矩形、欧氏距离等），并不依赖公开的实验数据集。

**📈 对比分析**

在与之前最优0.824的对比实验中，本系统在约10轮迭代后稳定到0.8559；不同LLM backbone（GPT‑5、Gemini 3 Pro）均能实现相同提升；总推理时间约4.6小时，验证时间约11.7小时，消耗约0.5M token。

**⚠️ 局限性**

局限性包括：对LLM的hallucination仍有一定风险；瓶颈区反射机制在极大参数空间下的收敛速度未知；当前成果仅适用于二维欧氏平面，推广到高维或其他度量空间仍需进一步研究。

---

## 32. Stealthy Poisoning Attacks Bypass Defenses in Regression Settings

**arXiv ID:** 2601.22308 | [PDF](https://arxiv.org/pdf/2601.22308v1)

**作者:** Javier Carnerero-Cano `[一作]` (IBM Research Europe), Emil C. Lupu `[通讯]` (Imperial College London)

**通讯引用:** 7513 | [OpenAlex ID](https://openalex.org/A5065619732)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了针对回归模型的隐蔽数据投毒攻击及基于贝叶斯线性回归的防御方法BayesClean。

**💡 创新点**

创新点在于将攻击目标建模为多目标双层优化，兼顾攻击效果与可检测性，并通过归一化实现不同权衡；同时提出利用预测方差拒绝异常样本的新防御。

**🔧 技术方法**

技术包括双层优化、逆向模式微分(RMD)求超梯度、软可检测性约束、贝叶斯线性回归（EM求解超参数）以及SVD实现高效矩阵运算。

**📊 数据集**

实验使用四个真实数据集：Lending Club贷款利率、心脏病、波士顿房价和家电能耗，并将攻击扩展到前馈深度神经网络。

**📈 对比分析**

与TRIM、Huber、SEVER、Proda、DUTI等现有防御进行比较，采用测试防御增益评价；结果显示隐蔽攻击能绕过现有防御，而BayesClean在高投毒比例（>20%）下的防御效果显著优于其它方法。

**⚠️ 局限性**

局限性包括双层优化求解仍受限于凸内层、计算成本高；防御基于贝叶斯线性回归，对非线性模型的适用性有限；假设攻击者了解数据分布，且评估主要聚焦于回归任务。

---

## 33. Adapting Reinforcement Learning for Path Planning in Constrained Parking Scenarios

**arXiv ID:** 2601.22545 | [PDF](https://arxiv.org/pdf/2601.22545v1)

**作者:** Feng Tao `[一作]` (Bosch Research), Ren Liu `[通讯]` (Bosch Research)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研发了一种基于深度强化学习的实时泊车路径规划框架，并发布了专为受限泊车场景设计的ParkBench基准；

**💡 创新点**

创新点包括：将泊车规划视为基于车辆运动学的强化学习任务；使用动作分块机制（Action‑Chunking）兼容任何RL算法；结合手工设计的课程学习提升收敛；以及保持与传统Hybrid A*相同的输入格式，实现无定位追踪的端到端规划；

**🔧 技术方法**

核心技术包括：双轮车辆运动学模型、PPO算法与自适应课程学习、动作分块包装器、交叉注意力特征提取器、Gym接口仿真环境；

**📊 数据集**

使用ParkBench数据集，共51个真实泊车场景的初始姿态、目标姿态与障碍物轮廓；

**📈 对比分析**

在ParkBench上与Hybrid A*、PPO（无课程）和PPO+Chunking 进行对比；取得最高成功率92.2%（+96%相对Hybrid A*），规划时间0.20s（-52%），路径长度19.2m（-28%），并保持相似的方向切换次数；

**⚠️ 局限性**

局限性包括：在开阔空间表现下降（成功率下降）；课程是手工设计，难以迁移到其他泊车类型；未来工作计划扩大基准、自动化课程设计以及处理更广泛的泊车动作。

---

## 34. A Semantically Consistent Dataset for Data-Efficient Query-Based Universal Sound Separation

**arXiv ID:** 2601.22599 | [PDF](https://arxiv.org/pdf/2601.22599v1)

**作者:** Kai Li `[一作]` (Tsinghua University), Xiaolin Hu `[通讯]` (Tsinghua University)

**通讯引用:** 19154 | [OpenAlex ID](https://openalex.org/A5004579631)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `67630363-6be0-4f51-ab05-7198250671a5` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一套自动化的数据清洗与合成流程，挖掘并合成高纯度单事件音频，构建Hive合成数据集，并在该数据集上训练查询式通用音频分离模型；

**💡 创新点**

1）对AudioSet本体进行重构，消除同义与重叠标签；2）使用多模态大模型（Qwen3‑Omni）与音频标签模型实现语义‑声学一一对应，保证单事件纯度；3）引入语义一致混合矩阵，生成语义上可共存的混合；4）通过小样本验证展示数据量仅占SAM‑Audio 0.2%即可获得竞争性能；

**🔧 技术方法**

大规模多模态语言模型（Qwen3‑Omni）进行语义对齐；音频标签模型用于粗糙分类；超分辨率模型（Apollo）用于采样率统一；FlowSep与AudioSep作为生成与判别两类分离框架；混合时随机SNR与能量归一化；

**📊 数据集**

采集自AudioSet、VGGSound、FreeSound、BBC Sound Effects等公开数据共0.9M片段，经过清洗后构成2.4k小时的Hive合成数据集；实验还使用MUSDB18‑HQ、USS‑Bench等公开基准进行零样本和跨域评估；

**📈 对比分析**

以Hive测试集为零样本基准，对比原始公开模型与在Hive上训练的AudioSep/FlowSep，使用SDR、SI‑SDR、FAD、CLAP、OQ等多维度指标；结果显示Hive训练模型在零样本时与百万小时的SAM‑Audio持平，甚至在某些指标上超越；同时提供MACs、CPU/GPU耗时与显存对比，展示判别模型的高效性；

**⚠️ 局限性**

1）合成数据仍无法完全重现真实环境的复杂性；2）依赖大模型的标签推断可能引入偏差；3）在极端多源场景下仍可能出现残余干扰；4）对资源有限的研究者而言，虽然数据量小，但仍需一定的GPU资源进行训练与评估。

---

## 35. Predicting Intermittent Job Failure Categories for Diagnosis Using Few-Shot Fine-Tuned Language Models

**arXiv ID:** 2601.22264 | [PDF](https://arxiv.org/pdf/2601.22264v1)

**作者:** Henri Aïdasso `[一作]` (Ecole de technologie superieure), Ali Tizghadam `[通讯]` (TELUS)

**通讯引用:** 976 | [OpenAlex ID](https://openalex.org/A5011117380)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究提出了FlaXifyer，一种利用少样本微调预训练语言模型预测CI/CD流水线中间歇性（flaky）作业失败类别的方法，并提出LogSift解释技术来快速定位日志中关键语句；

**💡 创新点**

创新点包括：①将少样本学习与预训练语言模型结合，实现仅12个样本即可准确预测13类失败；②提出高效的LogSift递归分区解释算法；③在工业数据上验证方法的可扩展性和可解释性，填补了现有检测后诊断研究的空白；

**🔧 技术方法**

使用技术包括SetFit少样本微调、BGE/CodeBERT文本编码器、对比学习、LogSift递归分区解释、Monte Carlo交叉验证与Optuna调参；

**📊 数据集**

数据集为4,511条TELUS工业CI/CD作业日志，精选13类优先失败，构成2,458条间歇性失败样本；

**📈 对比分析**

通过与CodeBERT的对比、不同shot设置的实验，采用宏F1、MCC、Top‑k准确率评估；结果显示BGE 12shot得到84.3%宏F1、92% Top‑2准确率；LogSift平均降低日志74.4%，执行时间≤1s；

**⚠️ 局限性**

局限性包括：标签依赖正则匹配，约9%噪声；实验仅在TELUS环境，泛化性待验证；类别重叠导致多标签情况未完全处理，LogSift偶尔误聚焦关键语句。

---

## 36. Tuning the Implicit Regularizer of Masked Diffusion Language Models: Enhancing Generalization via Insights from $k$-Parity

**arXiv ID:** 2601.22450 | [PDF](https://arxiv.org/pdf/2601.22450v1)

**作者:** Jianhao Huang `[一作]` (University of California), Baharan Mirzasoleiman `[通讯]` (University of California)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过k-奇偶性任务研究并理论分解了掩码扩散（MD）目标，证明其噪声项充当隐式正则化，帮助模型避免“grokking”，并提出基于信号最优的掩码采样策略。

**💡 创新点**

创新点在于：①把MD目标拆成信号与噪声两段，揭示噪声正则化机制；②提出信号最优掩码分布；③将这些理论从离散任务迁移到大规模语言模型，显著提升泛化与预训练效率。

**🔧 技术方法**

使用技术包括：掩码扩散训练框架、Transformer（后简化为2层MLP）与梯度流分析、信息论样本复杂度推导、以及针对不同规模的nanoGPT、LLaDA 8B实现。

**📊 数据集**

使用的数据集主要有：k-奇偶性（synthetic）、C4/CCP语料（预训练）、HellaSwag、ARC-Easy、MMLU、ARC-Challenge、GPQA（监督微调）以及GSM8K、MATH（生成式评测）。

**📈 对比分析**

与传统AR模型或均匀掩码训练相比，在8B预训练阶段取得约8.8% perplexity下降，在SFT监督任务上提升约5.8%准确率；生成式任务在高掩码比例下表现更佳，说明不同任务需调节掩码窗口。

**⚠️ 局限性**

局限性在于仅实验了宽度为0.1的均匀掩码区间，未探索非线性或自适应课程学习；此外研究主要聚焦英文语料，跨语言或更复杂结构的验证尚待进一步工作。

---

## 37. Advanced techniques and applications of LiDAR Place Recognition in Agricultural Environments: A Comprehensive Survey

**arXiv ID:** 2601.22198 | [PDF](https://arxiv.org/pdf/2601.22198v1)

**作者:** Judith Vilella-Cantos `[一作]` (Miguel Hernandez University), Luis Payá `[通讯]` (Miguel Hernandez University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文综述了农业环境下基于LiDAR的定位与识别技术，系统评估了方法、数据集与评估指标，并讨论了挑战与未来方向。

**💡 创新点**

首次聚焦农业场景的LiDAR定位综述，整合了深度学习、手工特征与语义融合方法，并对跨季性能差异提供深入分析。

**🔧 技术方法**

综述涵盖了深度学习（如PointNetVLAD、MinkUNeXt）、传统手工特征（Scan Context、Fast Point Feature Histogram）以及语义信息融合技术。

**📊 数据集**

讨论的农业数据集包括TEMPO-VINE、BLT、HORTO-3DLM、MAgro等，同时与城市KITTI、Oxford RobotCar等对比数据集进行评估。

**📈 对比分析**

通过Recall@1、MTE等指标对SOTA方法进行比较，发现同季Recall@1可达70%+，但跨季Recall下降至<50%，MTE在不同季节差异显著。

**⚠️ 局限性**

主要局限在数据集稀缺、跨季鲁棒性不足，以及对动态与季节变化适应性不佳。

---

## 38. Elastic Spectral State Space Models for Budgeted Inference

**arXiv ID:** 2601.22488 | [PDF](https://arxiv.org/pdf/2601.22488v1)

**作者:** Dachuan Song `[一作]` (George Mason University), Xuan Wang `[通讯]` (George Mason University)

**通讯引用:** 11142 | [OpenAlex ID](https://openalex.org/A5089622254)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研发了一种Elastic Spectral State Space Model（ES‑SSM），可一次性训练全容量模型，随后按需截断至任意规模进行预算化推理，支持在不同硬件资源下动态部署。

**💡 创新点**

创新点在于：① 引入输入自适应谱门控（lightweight MLP+mask‑softmax）和预算dropout训练，强制低序列谱通道携带主要信息，使截断后仍能保持高性能；② 在Hankel谱滤波的Spectral SSM基础上实现可弹性压缩，解决传统SSM在不同预算下性能骤降的问题。

**🔧 技术方法**

技术手段包括：基于Hankel矩阵的谱分解与固定谱基的Spectral SSM；两层轻量MLP门控与根均方归一化的预算感知softmax；随机预算dropout训练；残差前归一化（pre‑norm）结构。

**📊 数据集**

实验数据集涵盖：Long Range Arena（ListOps、IMDb Text、AAN Retrieval、CIFAR、Pathfinder、Path‑X）、PG19字节级语言建模、Speech Commands 等长序列基准。

**📈 对比分析**

与Transformer、S4、Mamba、Mamba‑2、AR‑STU以及固定预算Spectral SSM等基线在约20M参数规模下比较，ES‑SSM在全容量下与基线相当或更优；在预算扫描中，98%性能阈值的sweet spot仅需3–4个谱通道，性能随预算下降呈平滑曲线，显示出优异的预算‑性能折衷。

**⚠️ 局限性**

局限性包括：未在数十亿参数的基金模型规模上验证；实验仅覆盖标准长序列基准，未覆盖真实生产环境；对与注意力等混合模块的交互影响了解有限；缺乏实时在线预算调节的实际部署评估。

---

## 39. $ρ$-$\texttt{EOS}$: Training-free Bidirectional Variable-Length Control for Masked Diffusion LLMs

**arXiv ID:** 2601.22527 | [PDF](https://arxiv.org/pdf/2601.22527v1)

**作者:** Jingyi Yang `[一作]` (Shanghai Artificial Intelligence Laboratory), Jing Shao `[通讯]` (Fudan University)

**通讯引用:** 10581 | [OpenAlex ID](https://openalex.org/A5003742311)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种训练无关的单阶段双向可变长度去噪策略ρ-，用于掩蔽扩散大语言模型（dLLMs），以解决固定生成长度的限制。

**💡 创新点**

创新点在于通过隐式密度信号动态评估生成长度的充分性，实现了生成长度的双向调整，提升了推理效率和令牌利用率。

**🔧 技术方法**

使用了隐式密度估计和去噪过程中的双向长度调整技术。

**📊 数据集**

在数学推理（GSM8K和MATH500）和代码生成（MBPP和HumanEval）基准上进行了实验。

**📈 对比分析**

与固定长度基线和DAEDAL方法进行比较，ρ-在大多数基准上表现出相当的性能，同时显著提高了推理效率和有效令牌比率。

**⚠️ 局限性**

限制在于对隐式密度阈值的选择可能影响性能，但实验表明其对阈值变化具有较强的鲁棒性。

---

## 40. Countering the Over-Reliance Trap: Mitigating Object Hallucination for LVLMs via a Self-Validation Framework

**arXiv ID:** 2601.22451 | [PDF](https://arxiv.org/pdf/2601.22451v1)

**作者:** Shiyu Liu `[一作]` (Xiamen University), Jinsong Su `[通讯]` (Xiamen University)

**通讯引用:** 3966 | [OpenAlex ID](https://openalex.org/A5066326238)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

针对大视觉语言模型在图像描述任务中出现的对象幻觉问题，本文系统分析了模型对语言先验的过度依赖，并提出了语言先验自由验证（LPFV）和自验证框架，显著降低幻觉率。

**💡 创新点**

创新点在于通过“仅一句词/短语”提示迫使模型关注图像而非先验，得到可靠的对象存在置信度，并将其用于候选句子验证与过滤/聚合两种策略，从而在不训练的前提下提升事实性。

**🔧 技术方法**

技术主要包括：Jensen‑Shannon Divergence (JSD) 分析、LPFV 对象置信度评估、候选多样化采样、Best‑of‑N 选择、Filter‑then‑Aggregate 策略、基于规则或模型的对象抽取与 GPT‑4o‑mini 的辅助评估。

**📊 数据集**

实验数据集为 MSCOCO 验证集（500张图片）用于评估，离线对象抽取还使用了 2,000 张训练集图像。

**📈 对比分析**

与 VCD、CGD、HALC、DeCo、Less、Nullu 等现有解码/校正方法相比，Self‑Val 在 LLaVA‑v1.5‑7B 上 CHAIR_I 降低 52.2%–65.6%，实现了新 SOTA；在 LLaVA‑v1.5‑13B、mPLUG‑Owl2‑7B、Qwen2.5‑VL‑7B 上亦保持显著提升，且 F1、Acc、Rel 等 GPT‑assisted 指标不受损失。

**⚠️ 局限性**

局限性包括：额外的候选采样与验证会增加推理延迟；对预定义对象集的依赖在泛化到新领域时可能受限；在线对象抽取受模型指令遵循能力限制；当阈值 α 过高时可能过度过滤真实细节。

---

## 41. Rethinking Speech Representation Aggregation in Speech Enhancement: A Phonetic Mutual Information Perspective

**arXiv ID:** 2601.22480 | [PDF](https://arxiv.org/pdf/2601.22480v1)

**作者:** Seungu Han `[一作]` (Seoul National University), Kyogu Lee `[通讯]` (Seoul National University)

**通讯引用:** 1941 | [OpenAlex ID](https://openalex.org/A5088852010)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种“语言聚合层”来在语音增强任务中预训练并冻结，以最大化与音素标签的互信息，从而在噪声环境下保持语义信息。

**💡 创新点**

创新点在于将聚合模块的训练与语音增强主网络解耦，先用互信息目标优化聚合权重，再冻结使用；并提出动态加权求和（Dynamic Weighted-Sum）以适应时变噪声，显著降低单词错误率。

**🔧 技术方法**

技术包括自监督学习（HuBERT、WavLM、wav2vec 2）、互信息下界估计（线性探测器）、加权求和聚合、动态自注意力聚合、语音增强后端（SUPERB 语音增强模型）以及 Whisper-Large-v3 进行 WER 评估。

**📊 数据集**

使用的数据集：LibriSpeech（clean dev/test）、ESC‑50 生成噪声、VoiceBank‑DEMAND 进行增强训练与评估，以及 MUSAN 噪声做数据增强。

**📈 对比分析**

与传统联合训练的加权求和聚合（Acoustic‑WS）和混合聚合（Hybrid‑WS）比较，语言聚合层在保持语义信息方面显著提高 MI，WER 降低至 5.8%–6.3%，在保持或略低的 SI‑SDR、STOI、PESQ 的前提下；动态聚合进一步提升低 SNR 下的 WER，并在某些模型上获得最佳性能。

**⚠️ 局限性**

局限性包括：1）对噪声的鲁棒性仍受限于上层语义信息在噪声下衰减；2）动态聚合模型复杂度提高，训练时间与资源需求上升；3）在 wav2vec 2 上表现相对弱，说明模型内部语义分布不够集中；4）仅关注音素互信息，未考虑说话人识别等其他语义维度。

---

## 42. Scalable Fair Influence Blocking Maximization via Approximately Monotonic Submodular Optimization

**arXiv ID:** 2601.22584 | [PDF](https://arxiv.org/pdf/2601.22584v1)

**作者:** Qiangpeng Fang `[一作]` (China University of Mining and Technology), Zhixiao Wang `[通讯]` (China University of Mining and Technology)

**通讯引用:** 1937 | [OpenAlex ID](https://openalex.org/A5101687093)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文在影响阻断最大化（IBM）框架中加入了人口统计公平（DP），提出了公平影响阻断最大化（FIBM）问题，并设计了一种可约子模函数的DP感知目标，结合阻断效果实现可调的公平-效果折中。

**💡 创新点**

创新点包括：①将DP作为公平度量并用近似单调子模函数逼近；②通过线性标量化同时优化公平与阻断效果；③提出CELF-R算法，利用近似子模特性实现高效的惰性贪婪选择，并天然支持构造Pareto前沿。

**🔧 技术方法**

核心技术包括：近似子模与单调性理论、基于VRR路径的高效影响估计、改进的惰性贪婪（CELF-R）以及可调的线性标量化；算法实现采用多线程并行化以提升速度。

**📊 数据集**

实验使用四个真实社交网络数据集：Facebook（4.5K节点）、Slashdot（70K）、Gowalla（197K）和Pokec（1.6M）并分别检测不同社群划分。

**📈 对比分析**

与Greedy、iIMDP（LP形式）、MMF、DC、WF、CFF等基线比较，CELF-R在公平-效果两维上均形成Pareto前沿，且在阻断效果和公平性指标上均优于基线；在计算效率上相较于全评估（FC）提高约50-60%，相较于LP方法快数百倍。

**⚠️ 局限性**

局限性主要是：①近似子模理论下的理论保证仍为保守估计，实际性能受网络结构影响；②对β值的调节仍需人工经验；③DP约束仍假设社区划分已知且不重叠，对重叠社群的公平性尚未考虑。

---

## 43. Learn More with Less: Uncertainty Consistency Guided Query Selection for RLVR

**arXiv ID:** 2601.22595 | [PDF](https://arxiv.org/pdf/2601.22595v1)

**作者:** Hao Yi `[一作]` (Renmin University of China), Yong Liu `[通讯]` (Renmin University of China)

**通讯引用:** 20356 | [OpenAlex ID](https://openalex.org/A5100724297)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种基于“不确定性一致性”的主动学习方法，用来在强化学习可验证奖励（RLVR）框架下筛选更有价值的训练样本，从而在数学推理任务中仅用30%的数据即可达到接近全量数据的性能。

**💡 创新点**

创新点在于：①引入离线与在线两种不确定性一致性指标（r_pb 与 r_pb^online），以衡量主观不确定性与客观奖励的一致性；②理论证明在线指标与离线指标呈严格负相关，并能有效指导样本选择；③在RLVR训练中采用一致性样本显著提升梯度稳定性与模型泛化。

**🔧 技术方法**

技术手段包括：强化学习可验证奖励算法（GRPO、RLOO、DAPO、REINFORCE++）、基于熵/困惑度的主观不确定性估计、点二分相关系数计算、在线一致性指标的归一化优势与不确定性融合。

**📊 数据集**

使用了数学推理数据集 MATH、GSM8K 以及更大规模的 DAPO-MATH-17K。

**📈 对比分析**

与随机抽样、传统不确定性采样（PPL/ENT）、特征覆盖采样（K-center）以及基于提示的采样（AskLLM）等基线相比，本文方法在离线 30% 数据下提升 1–2% Pass@1，在线 30% 数据下接近甚至略优于全量数据，且在不同RLVR算法与不同 γ 参数设置下均保持优越性能。

**⚠️ 局限性**

局限性包括：①对 γ 超参数敏感，需要经验调优；②目前仅验证于可验证奖励的RLVR任务，对其它类型奖励或无奖励环境的推广尚需探索；③在线指标的计算需要额外采样与优势估计，可能增加计算开销。

---

## 44. Decoding in Geometry: Alleviating Embedding-Space Crowding for Complex Reasoning

**arXiv ID:** 2601.22536 | [PDF](https://arxiv.org/pdf/2601.22536v1)

**作者:** Yixin Yang `[一作]` (Peking University), Zhifang Sui `[通讯]` (Peking University)

**通讯引用:** 4585 | [OpenAlex ID](https://openalex.org/A5110285832)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在大型语言模型的采样式解码中，提出了基于嵌入空间拥挤（crowding）的诊断方法，并设计了一种无需额外训练的几何感知采样策略（Crowding-Aware Sampling）来缓解嵌入空间拥挤，从而提升复杂推理任务的准确性与多样性。

**💡 创新点**

①首次量化并证明嵌入空间拥挤与推理成功率负相关；②提出一种仅对高概率且高拥挤的候选词进行几何加权的自适应重权重机制；③该方法不需要额外训练、前向推理或外部信号，可作为现有采样策略（如 top‑p、温度）之插件。

**🔧 技术方法**

使用余弦相似度计算词嵌入之间的几何关系；在每一步对高概率词集合计算 token‑level、step‑level 的拥挤分数；根据拥挤分数与概率共同计算修正因子 α 并重新归一化；整体流程只需一次 softmax 之后的张量运算。

**📊 数据集**

主要在数学推理基准 AIME24、AIME25、HMMT25 上进行实验，并在 Qwen3‑1.7B、Qwen3‑4B 与 Hunyuan‑1.8B‑Instruct 三个模型上验证通用性。

**📈 对比分析**

与标准采样（top‑p + 温度）以及改进的线性/非线性权重版本对比，Crowding‑Aware Sampling 在 Avg@32、Pass@8、Distinct‑N 与 Semantic Diversity 上平均提升 0.5–1.0 分，Pass@8 提升约 1–3%，同时保持或提高多样性。ablation 证明拥挤加权是关键因素，非线性权重更倾向多样性，线性权重更注重准确率。

**⚠️ 局限性**

①仅依赖词嵌入的几何信息，忽略了潜在的语义上下文差异；②在极度低温度或严苛的截断策略下，拥挤修正可能导致过度抑制高概率词；③实验集中在数学推理数据集，未验证对自然语言生成或多模态任务的效果；④超参数 τ、ε 对结果有一定影响，需在不同模型/任务中手动调节。

---

## 45. Why Johnny Can't Think: GenAI's Impacts on Cognitive Engagement

**arXiv ID:** 2601.22430 | [PDF](https://arxiv.org/pdf/2601.22430v1)

**作者:** Rudrajit Choudhuri `[一作]` (Oregon State University), Anita Sarma `[通讯]` (Oregon State University)

**通讯引用:** 3558 | [OpenAlex ID](https://openalex.org/A5024821289)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

调查了 STEM 学生对生成式 AI 的信任和日常使用如何影响他们在课程中的认知投入，包括反思、理解需求和批判性思维。

**💡 创新点**

首次将生成式 AI 的信任驱动使用与认知风格结合，提出并验证了“认知债务循环”模型，并发现技术亲和、风险容忍和自我效能高的学生更易出现认知脱离。

**🔧 技术方法**

采用结构方程模型（Partial Least Squares‑SEM）评估调查问卷，并通过多组分析检验模型稳健性。

**📊 数据集**

基于五所北美大学 299 名 STEM 学生的问卷数据。

**📈 对比分析**

通过 PLS‑SEM 获得的模型解释力最高 R² 0.65，拟合优度 SRMR 0.067，预测 Q² 为正，表明模型具备良好的解释与预测性能。

**⚠️ 局限性**

受限于横断面自报告数据、样本集中于美国高校、未能验证因果关系及未纳入长期使用行为，可能存在自选偏差。

---

## 46. Towards the Holographic Characteristic of LLMs for Efficient Short-text Generation

**arXiv ID:** 2601.22546 | [PDF](https://arxiv.org/pdf/2601.22546v1)

**作者:** Shun Qian `[一作]` (Harbin Institute of Technology), Baoxun Wang `[通讯]` (Tencent)

**通讯引用:** 989 | [OpenAlex ID](https://openalex.org/A5047681321)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出 HOLO 插件，利用 LLM 生成过程早期出现的目标关键词（Holographic Characteristic），通过两步提取关键词后采用词汇约束生成（POINTER）完成短文本生成。

**💡 创新点**

创新点在于：①首次系统性发现并利用 LLM 生成早期关键词分布；②仅凭前两步推断关键词，极大减少推理步骤；③将关键词链与并行插入式生成结合，实现高效且可控的短文本生成。

**🔧 技术方法**

主要技术包括：概率分布推断（基于 Markov 近似）、词汇约束生成模型 POINTER、掩码预测（Mask‑Predict）改进插入策略、BERT 评估器做句子相关性排序。

**📊 数据集**

使用三大中文对话数据集：Douban（1k）、Weibo（10k）和 LCCC（10k）进行实验。

**📈 对比分析**

与 EVA2.0‑2.8B、ChatGLM‑6B、Belle‑13B 等基线模型对比。自动评估（F1、ROUGE‑L、BLEU‑2、Distinct‑2、Relevance）中 HOLO 在 5/7 指标上优于基线；BLEU‑4 与 PPL 略逊；人工（GPT‑4/3.5 G‑Eval）评估显示 HOLO 与基线相当，部分指标稍低但总体保持较好。推理效率方面，HOLO 在大模型上时间降低 56‑92%，显存降低 55‑62%。

**⚠️ 局限性**

局限性包括：①对极快或已接近最优的基线模型提升有限；②词汇约束生成仍导致句子自然度与人类水平略逊；③仅针对短文本/对话任务，未验证长文本或多轮对话的适用性；④依赖前两步的关键词预测，对噪声或低质量模型可能不稳健。

---

## 47. The Benefit of Collective Intelligence in Community-Based Content Moderation is Limited by Overt Political Signalling

**arXiv ID:** 2601.22201 | [PDF](https://arxiv.org/pdf/2601.22201v1)

**作者:** Gabriela Juncosa `[一作]` (Central European University), Taha Yasseri `[通讯]` (University College Dublin)

**通讯引用:** 3385 | [OpenAlex ID](https://openalex.org/A5046908604)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过在线实验，探究团队协作写作对社交媒体政治帖子的事实核查注释质量的提升；

**💡 创新点**

首次将政治多样性与身份信号纳入社区核查写作阶段，验证协作和多样性对注释质量的影响；

**🔧 技术方法**

采用实验设计与统计检验（两样本t检验、非参数检验）来评估协作与单独写作、身份披露与否的效果；

**📊 数据集**

实验使用432名自报民主党或共和党派的Prolific参与者生成648条注释，并收集1,610名评审者的帮助度评分及3名专家评价；

**📈 对比分析**

与单独写作和身份隐蔽条件比较，协作写作显著提升帮助度（p<0.01）；政治多样团队在评估共和党帖子时表现更好；身份公开则削弱协作优势；

**⚠️ 局限性**

实验规模有限、任务顺序固定、仅采用双人团队、可能存在自我报告偏差，且实验于2023年夏季进行，AI辅助写作的可能性已降至低水平。

---

## 48. SCOPE-PD: Explainable AI on Subjective and Clinical Objective Measurements of Parkinson's Disease for Precision Decision-Making

**arXiv ID:** 2601.22516 | [PDF](https://arxiv.org/pdf/2601.22516v1)

**作者:** Md Mezbahul Islam `[一作]` (Florida International University), Ananda Mohan Mondal `[通讯]` (Florida International University)

**通讯引用:** 495 | [OpenAlex ID](https://openalex.org/A5045295085)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本研究构建了SCOPE-PD框架，结合PD患者自评问卷与临床评估数据，利用机器学习预测并解释PD诊断

**💡 创新点**

创新点在于将主观与客观测量统一到同一可解释模型中，并使用SHAP方法量化每个特征对诊断的影响

**🔧 技术方法**

核心技术包括随机森林分类器、SHAP解释方法、特征预处理与标准化、5折嵌套交叉验证

**📊 数据集**

使用的数据集为公开的Parkinson’s Progression Markers Initiative（PPMI）基线数据，包含1786名样本、148个特征

**📈 对比分析**

与传统单一主观或客观方法相比，SCOPE‑PD在主观、客观与两者结合三种数据上分别取得约96.5%、97.8%和98.6%的准确率，表现优于已发表的多种方法

**⚠️ 局限性**

局限包括仅使用基线数据、缺乏外部验证、未纳入遗传与影像等多模态信息、未进行纵向进展预测

---

## 49. MirrorMark: A Distortion-Free Multi-Bit Watermark for Large Language Models

**arXiv ID:** 2601.22246 | [PDF](https://arxiv.org/pdf/2601.22246v1)

**作者:** Ya Jiang `[一作]`, Kai Zeng `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了针对文本水印的贝叶斯最优检测与解码框架，并给出了基于 Wrapped‑Beta 分布的统计分析，阐明了检测、解码与误检之间的关系。

**💡 创新点**

创新点包括：① 将水印检测与解码统一为贝叶斯后验推断，证明检测与解码可分离且同等优；② 分析了 GLRT（最大似然）方法的误差来源，说明其会因多假设导致误检率上升；③ 对 Gumbel‑max 镜像水印模型给出了 Wrapped‑Beta 统计推导，解释了误检与错误码率的定量关系；④ 通过 Bernstein 不等式得到误检/漏检概率的指数上界，揭示了文本长度与选取概率熵对检测可靠性的影响。

**🔧 技术方法**

主要技术包括：贝叶斯似然比检验、Gumbel‑max 采样、Wrapped‑Beta 分布推导、子指数分布的 Bernstein 上界、误检概率的指数界等统计与信息论工具。

**📊 数据集**

本文未给出具体公开数据集，性能分析基于理论推导和模拟实验（如随机生成的文本块与水印消息）。

**📈 对比分析**

与传统 GLRT、总是解码等方法对比，贝叶斯检测在 FPR、FNR 与 BER 上均表现更好；GLRT 的 FPR 随消息空间大小上升，导致性能下降；detect‑then‑decode 在保持检测最优的同时进一步降低误码率。

**⚠️ 局限性**

局限性在于：① 假设文本块独立且已知消息先验，实际情况可能更复杂；② 贝叶斯检测需要对所有可能消息求和，计算量随消息空间指数增长；③ 对分布假设（如 Wrapped‑Beta）敏感，若实际分布偏离理论模型，性能可能下降。

---

## 50. TimeMachine-bench: A Benchmark for Evaluating Model Capabilities in Repository-Level Migration Tasks

**arXiv ID:** 2601.22597 | [PDF](https://arxiv.org/pdf/2601.22597v1)

**作者:** Ryo Fujii `[一作]` (Tohoku University), Jun Suzuki `[通讯]` (RIKEN)

**通讯引用:** 5067 | [OpenAlex ID](https://openalex.org/A5002182453)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一套自动化生成的 Python 软件迁移基准 TimeMachine-bench，并构建了可验证子集与多模型评估。

**💡 创新点**

引入基于日期的环境控制实现动态依赖解析，提供可实时更新的迁移基准；同时提供人类验证的可解子集。

**🔧 技术方法**

使用 PyPI 时间过滤器、Docker 容器执行测试、LLM + ReAct 代理、工具集（edit_file、replace_all_in_file）等技术。

**📊 数据集**

采集自 The Stack v2 的 1,145 个 Python 项目，精简后 100 个可验证实例。

**📈 对比分析**

采用 pass@1(n,m) 与 prec@1 指标，在 n=100、m=10 的设置下评估 11 个模型；Claude Sonnet 4 取得 99% 的 pass，Qwen3-Coder 480B 90% 的 pass，精度普遍低于 80%，总体表现可观但仍有冗余编辑问题。

**⚠️ 局限性**

仅限 Python、依赖测试覆盖率低导致可能漏检、验证成本高、可能出现未来提交泄漏、子集偏向易题导致性能高估。

---

## 51. Shattered Compositionality: Counterintuitive Learning Dynamics of Transformers for Arithmetic

**arXiv ID:** 2601.22510 | [PDF](https://arxiv.org/pdf/2601.22510v1)

**作者:** Xingyu Zhao `[一作]` (University of Wisconsin–Madison), Yiqiao Zhong `[通讯]` (University of Wisconsin–Madison)

**通讯引用:** 793 | [OpenAlex ID](https://openalex.org/A5100968975)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了Transformer在合成算术任务中的学习动态，发现模型往往以非人类顺序并行学习子技能，导致“碎片化组合”（shattered compositionality）错误。

**💡 创新点**

创新点在于：①揭示模型子技能的反向学习顺序和并行学习导致的混合错误；②利用信息论互信息度量解释学习行为；③证明这一现象在大模型、缩放、scratchpad等条件下仍持续存在。

**🔧 技术方法**

技术手段包括：小型NanoGPT Transformer、梯度训练、逆序输出格式、子技能拆分与监测、信息论互信息指标、对比实验（不同数据格式、模型规模、scratchpad）以及整数误差分布分析。

**📊 数据集**

使用的主要数据集有：四个自定义算术数据集（4算子加法、乘法、比较、排序）、GSM8K模板扩展的分布偏移测试集，以及Pythia-1B预训练模型进行微调的算术数据。

**📈 对比分析**

通过子技能错误率、整数误差分布、互信息匹配曲线与训练步骤对齐进行评估；在小模型和大模型、scratchpad实验中均观察到学习顺序不变、混合错误持续；GSM8K分布偏移测试显示大模型在添加额外子句后性能显著下降。

**⚠️ 局限性**

局限性包括：实验主要在小型模型和合成算术任务上进行，缺乏对更大模型和多样化任务的验证；结果可能不易直接推广到真实自然语言推理场景。

---

## 52. High-Definition 5MP Stereo Vision Sensing for Robotics

**arXiv ID:** 2601.22445 | [PDF](https://arxiv.org/pdf/2601.22445v1)

**作者:** Leaf Jiang `[一作]` (NODAR Inc.), Piotr Swierczynski `[通讯]` (NODAR Sensor GmbH)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6514db3d-8de6-452c-91b7-acdb31787cc4` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

该研究通过引入专门针对5MP立体摄像头的高级在线标定方法，验证了高分辨率传感器能在现实环境中实现理论上可达的稠密点云质量。

**💡 创新点**

创新点在于提出并验证了点云质量与像素数的平方根尺度关系，并首次展示了仅靠高像素数而非标定就无法获得高质量点云的事实，进一步强调了动态标定的必要性。

**🔧 技术方法**

采用了NODAR的Hammerhead实时立体匹配算法与GroundTruth离线算法结合高分辨率图像，配合全局快门5.1MP摄像机、NVIDIA Jetson Orin AGX以及自研的图像预处理与匹配流水线。

**📊 数据集**

使用自采集的户外场景数据集，包含草地、树木和行人，图像分辨率为5.1MP（也对1.3MP下采样图像做对比），并提供完整的原始色彩图与深度图文件。

**📈 对比分析**

与GroundTruth对比实验表明，5MP Hammerhead在距离20 m以内的目标深度误差保持在1 m以下，且全分辨率相较于下采样能显著降低误差（如在39 m处误差从4.3 m降至2.6 m），验证了该方法在长距离高动态范围环境中的优越性能。

**⚠️ 局限性**

实验仅在15 cm基线、137°视场的单一硬件平台上完成，未覆盖不同基线长度、滚动快门或更低性能嵌入式GPU的情况，故其通用性和实时性能在更大尺度或更弱硬件上仍需进一步验证。

---

## 53. ReNCE: Learning to Reason by Noise Contrastive Estimation

**arXiv ID:** 2601.22432 | [PDF](https://arxiv.org/pdf/2601.22432v1)

**作者:** Wenzheng Zhang `[一作]` (Rutgers University), Karl Stratos `[通讯]` (Rutgers University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于噪声对比估计（NCE）的在线对比学习方法ReNCE，用于在LLM中实现数学推理能力；

**💡 创新点**

核心创新在于将GRPO中隐式的优势估计改为显式的正负样本对比，并结合动态正负比率过滤、奖励缩放的自适应间隔以及对KL惩罚的自适应调节，使对比学习更高效且更稳定；

**🔧 技术方法**

技术上使用多标签NCE、基于旧策略的log‑ratio评分、动态奖励缩放的间隔、KL惩罚的自适应窗口以及硬负采样的动态prompt过滤；

**📊 数据集**

训练数据由DAPO数据集（整数答案）与MATH数据集（符号/LaTeX形式）混合构成；

**📈 对比分析**

在六大数学推理基准（MATH500、AIME24、AIME25、AMC、Minerva Math、OlympiadBench）上与GRPO、DAPO和半在线DPO比较，ReNCE在大多数任务上排名第一，平均通过率最高；

**⚠️ 局限性**

局限性包括：仅在中等规模（Qwen3‑4B）模型上验证，缺乏对更大模型的评估；对超参数（如t_easy、α、β）敏感；仅在数学推理任务上测试，未验证在更广泛推理或对话任务中的泛化。

---

## 54. SAIR: Cost-Efficient Multi-Stage ML Pipeline Autoscaling via In-Context Reinforcement Learning

**arXiv ID:** 2601.22397 | [PDF](https://arxiv.org/pdf/2601.22397v1)

**作者:** Jianchang Su `[一作]` (University of Connecticut), Wei Zhang `[通讯]` (University of Connecticut)

**通讯引用:** 16395 | [OpenAlex ID](https://openalex.org/A5030319088)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本研究提出了 SAIR，一个基于大型语言模型的自适应多阶段机器学习推理管道自动伸缩框架，能够在无需离线训练的情况下实现在线决策；

**💡 创新点**

创新点在于将大模型作为在场强化学习（ICRL）代理，通过 Pareto 基础奖励塑形、惊讶（surprisal）驱动的经验检索以及 CUDA 拦截实现连续 GPU 速率控制，从而在动态瓶颈迁移和跨阶段耦合场景下实现理论可证明的 regret 分解；

**🔧 技术方法**

核心技术包括：LLM 驱动的 in-context RL、Pareto 前沿奖励设计、基于信息增益的子集检索、CUDA 用户空间拦截实现细粒度 GPU 速率调节以及基于 JSON 的动作验证与强制探测；

**📊 数据集**

使用四种常见 ML 推理管道数据集：MobileNetV2 图像分类、BERT 文本分析、Transformer 生成和视频多帧分析，并在三种负载模式（Poisson、Ramp、Burst）下进行评估；

**📈 对比分析**

与静态、HPA、VPA、阈值等传统伸缩方法对比，SAIR 在所有实验配置中实现了 P99 延迟提升达 35–50% 与有效成本降低 30–60%（相对最佳基线），并在高延迟视频分析场景中取得 97% 成本减幅；

**⚠️ 局限性**

主要限制包括：决策延迟约 1–2 秒，适用于分钟级伸缩；上下文窗口有限（最多 15–20 条经验）；成本模型假设可共享 GPU 速率，单租户场景下收益下降；以及仅在规模为 2‑GPU 集群中验证，未测试大规模多节点部署。

---

## 55. ReloPush-BOSS: Optimization-guided Nonmonotone Rearrangement Planning for a Car-like Robot Pusher

**arXiv ID:** 2601.22289 | [PDF](https://arxiv.org/pdf/2601.22289v1)

**作者:** Jeeho Ahn `[一作]` (University of Michigan), Christoforos Mavrogiannis `[通讯]` (University of Michigan)

**通讯引用:** 1032 | [OpenAlex ID](https://openalex.org/A5067086333)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c`

**🎯 论文内容**

提出了 ReloPush-BOSS 框架，用车式机器人在密集杂乱环境中进行非抓取多物体重新排列规划。

**💡 创新点**

创新点：将预定位优化与 Dubins 路径分类相结合，使用种子初始化避免高成本局部最优；采用深度优先回溯高层规划与可达性图结合；在连续优化中仅考虑最多一个预定位以保持可行性。

**🔧 技术方法**

技术：Dubins 路径、3 点 Dubins 路径（3PDP）理论、连续梯度优化、深度优先搜索、推送可达性图、OMPL/Hybrid A*、Dijkstra 图搜索。

**📊 数据集**

数据集：在 4×5.2 m² 的模拟工作空间内，使用 8–13 个方形物体，随机扰动 100 个实例/场景；在 MuSHR 1/10 赛道机器人上进行真实实验。

**📈 对比分析**

与 plRS‑Push、ReloPush、ReloPush‑B、ReloPush‑BO 等基线比较，ReloPush‑BOSS 在成功率、规划时间、推送长度和总路径长度上均优于其他方法，尤其在最难的 13‑物体场景中表现突出。

**⚠️ 局限性**

局限：转向半径设置保守导致在极端拥挤空间中规划困难；仅允许每个物体最多一个预定位，可能限制搜索空间；对极度拥挤环境时机器人可能无法找到可行路径；障碍清除仅使用直线推送，未充分利用优化技术。

---

## 56. Learning Provably Correct Distributed Protocols Without Human Knowledge

**arXiv ID:** 2601.22369 | [PDF](https://arxiv.org/pdf/2601.22369v1)

**作者:** Yujie Hui `[一作]` (Ohio State University), Yang Wang `[通讯]` (Ohio State University)

**通讯引用:** 39888 | [OpenAlex ID](https://openalex.org/A5100343306)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出了 GGMS 框架，自动合成满足 SMT 约束且在给定进程数、轮数和失败模型下 provably correct 的分布式协议。

**💡 创新点**

创新点在于将 MCTS 与全局 DFS、Transformer 动作编码以及模型检查闭环结合，使得在存在正确协议时搜索过程可保证收敛且结果可被完全验证。

**🔧 技术方法**

核心技术包括蒙特卡罗树搜索、Transformer 事件编码、全局深度优先搜索、基于模型检查的 counter‑example 反馈以及分阶段的 guided 采样策略。

**📊 数据集**

实验数据完全由自定义仿真生成：同步 crash 失效模型下的 consensus 与 atomic commit 场景，枚举所有初始状态与消息丢失组合，无外部公开数据集。

**📈 对比分析**

与传统 MCTS 及 MCTS+DFS 对比，GGMS 在相同设置下成功率更高（如 4 进程 3 失效时 100%），且收敛速度相当或略快，尤其在更大规模设置下表现优异。

**⚠️ 局限性**

局限性包括仅支持同步 crash 失效、广播同样消息、固定轮数与有限进程数；搜索空间随进程数指数增长，缺乏对异步或拜占庭模型的扩展。

---

## 57. The Six Sigma Agent: Achieving Enterprise-Grade Reliability in LLM Systems Through Consensus-Driven Decomposed Execution

**arXiv ID:** 2601.22290 | [PDF](https://arxiv.org/pdf/2601.22290v1)

**作者:** Khush Patel `[一作]` (Lyzr Research), Shreyas Kapale `[通讯]` (Lyzr Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了 Six Sigma Agent 架构，利用任务分解、微代理多重采样与动态投票，实现在企业级 LLM 工作流中的可预测高可靠性。

**💡 创新点**

创新点在于把可靠性归结为系统层面的冗余与共识，而非单纯提升模型能力；通过数学证明展示投票能实现指数级错误下降，并引入基于语义聚类与动态扩容的投票机制。

**🔧 技术方法**

采用 LLM 微代理并行采样、基于嵌入的语义聚类投票、动态样本扩容、任务 DAG 分解、状态管理及跨模型异构执行；并借鉴 Byzantine/Fault‑Tolerance 思路进行系统设计。

**📊 数据集**

使用了三大企业用例数据集：财务文档处理、客服路由与合同条款分析（均为内部业务数据），用于评估可靠性和性能。

**📈 对比分析**

与单一 GPT‑4o、GPT‑4o‑mini 以及 CoT‑Self‑Consistency 基线对比，结果显示 13 代理动态扩容后 DPMO 达 3.4（Six Sigma 级别），相较于 GPT‑4o‑mini 提高 14,700×，且在保持 80% 成本下降的前提下提升可靠性。

**⚠️ 局限性**

局限包括对任务分解质量的高度依赖；假设误差多样化，若出现系统性偏差或全局错误则难以纠正；对极长或实时工作流的扩展性有限；对开放式、创意类任务不适用。

---

## 58. MemeChain: A Multimodal Cross-Chain Dataset for Meme Coin Forensics and Risk Analysis

**arXiv ID:** 2601.22185 | [PDF](https://arxiv.org/pdf/2601.22185v1)

**作者:** Alberto Maria Mongardini `[一作]` (Technical University of Denmark), Alessandro Mei `[通讯]` (Sapienza University of Rome)

**通讯引用:** 2035 | [OpenAlex ID](https://openalex.org/A5050834543)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

开发并发布了跨链多模态数据集MemeChain，包含34,988个Meme Coin的链上交易、合约信息、网页HTML源码、Token Logo、社交媒体链接，并对其生命周期、网站基础设施以及“1日死亡”现象进行了系统分析。

**💡 创新点**

创新点：①首次构建跨链（Ethereum、BNB Smart Chain、Solana、Base）且涵盖链上、网页源码、Logo、社交媒体等多模态信息的完整数据集；②系统化研究“1日死亡”币与网站可达率、Logo缺失等指标之间的关联，为诈骗识别提供新维度；③为后续多模态异常检测、存活预测等研究提供可复现、可扩展的基准数据。

**🔧 技术方法**

技术方法包括：大规模爬虫与API抓取（CoinMarketCap、CoinGecko、DexScreener、CoinSniper、Gecko Terminal、pump.fun等），TF‑IDF关键词分类+手工过滤，区块链探测器（Etherscan、BscScan、BaseScan、Solana RPC、Alchemy）验证合约部署；WHOIS查询、URL 重定向处理、Google Safe Browsing/Chain Patrol安全扫描；统计分析与可视化。

**📊 数据集**

使用的数据集：本研究发布的MemeChain数据集（34,988个币），以及上述公开来源（CoinMarketCap、CoinGecko、DexScreener、CoinSniper、Gecko Terminal、pump.fun）和区块链探测器（Etherscan、BscScan、BaseScan、Solana RPC、Alchemy）收集的链上信息。

**📈 对比分析**

本文主要聚焦数据集构建与描述性分析，没有对比实验或模型性能评估；在描述性统计中展示了1日死亡率、网站可达率、Logo缺失比例等指标，以证明数据集的丰富性与研究价值。

**⚠️ 局限性**

局限性：采样窗口仅为2024年10月至2025年1月，无法覆盖长期存活动态；使用聚合API导致交易细粒度缺失；基于名称的分类偏重精度，召回率可能不足；HTML快照为单点时间戳，无法捕捉网站的动态变化；部分币缺少价格/市值信息。

---

## 59. An innovating approach to teaching applied to database design. Improvement of Action Learning in Lifelong Learning

**arXiv ID:** 2601.22175 | [PDF](https://arxiv.org/pdf/2601.22175v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7`

---

## 60. Exact closed-form Gaussian moments of residual layers

**arXiv ID:** 2601.22307 | [PDF](https://arxiv.org/pdf/2601.22307v1)

**作者:** Simon Kuang `[一作]` (University of California), Xinfan Lin `[通讯]` (University of California)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出层级高斯近似的精确矩值传播方法，给出ReLU、GeLU、sin、probit、Heaviside等激活函数的一阶二阶矩闭式，并支持残差网络。

**💡 创新点**

首次在单层和深层网络中实现闭式一、二阶矩匹配，覆盖多种激活函数，并提供理论误差界限，显著优于传统线性、无穷正态或均值场方法。

**🔧 技术方法**

采用高维Stein定理、Gaussian ODE、支配收敛定理、Fourier/特征函数等解析技巧，构造Mσ、Kσ、Lσ三类函数，推导闭式表达式。

**📊 数据集**

在合成随机网络、加州住房（California Housing）、台湾破产预测（Taiwanese bankruptcy）以及UCI 具体压缩强度等实测数据上评估。

**📈 对比分析**

与均值场、线性、无穷正态（'95、'02）、Monte Carlo等基线比较，通过Wasserstein、KL散度和log概率评估，显著降低误差（上千倍到百万倍），在实测任务中提升对数概率和校准度。

**⚠️ 局限性**

仍需近似高阶交互的多层乘法（仅为高斯近似），对非闭式激活函数（如logistic sigmoid）无直接解析；在某些极端非线性/高相关场景下误差界限保守；并未完全克服权重不确定性的深度网络中多重非正态性。

---

## 61. Does My Chatbot Have an Agenda? Understanding Human and AI Agency in Human-Human-like Chatbot Interaction

**arXiv ID:** 2601.22452 | [PDF](https://arxiv.org/pdf/2601.22452v1)

**作者:** Bhada Yun `[一作]` (ETH Zurich), April Yi Wang `[通讯]` (ETH Zurich)

**通讯引用:** 461 | [OpenAlex ID](https://openalex.org/A5046673805)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对22名参与者进行为期一个月的自然交互实验，记录并分析他们与AI聊天机器人Day的对话，探讨人类与AI的代理权是如何共同建构的。

**💡 创新点**

提出了3×4代理框架（Human/AI/Hybrid × Intention/Execution/Adaptation/Delimitation），阐明代理权是一个动态、协同的过程，并提出了“可见性按需”与“可转化代理”的设计原则。

**🔧 技术方法**

使用大型语言模型（Claude、Gemini）与Prompt工程构建Day，采用Next.js前端、Supabase后端实现双层架构，支持策略生成、记忆管理与对话生成。

**📊 数据集**

数据集为22名跨文化参与者的约75小时对话日志（≈192场对话、167.6条信息/场），以及后续的半结构化访谈记录和策略揭示阶段的系统日志。

**📈 对比分析**

采用质性编码（线性分析、共编码、讨论达成）来评估代理维度，未做传统算法性能对比；实验结果表明代理权在对话中逐步共建，透明度对用户感知影响双向。

**⚠️ 局限性**

局限性包括样本规模小、文化与专业背景不够多样、实验时长有限、Day的特定实现可能影响结果、缺乏外部基准与量化指标，故难以推广至所有聊天机器人。

---

## 62. Can 3D point cloud data improve automated body condition score prediction in dairy cattle?

**arXiv ID:** 2601.22522 | [PDF](https://arxiv.org/pdf/2601.22522v1)

**作者:** Zhou Tang `[一作]`, Haipeng Yu `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

提出了一套基于深度图与点云的牛体形态关键点检测、YOLO分割与手工几何特征提取的完整工作流，用于牛体况评分与后续量化分析。

**💡 创新点**

创新点在于将九个解剖关键点映射为尖峰标记，并利用这些标记构造多尺度的最大距离、面积及体积特征；同时通过F1‑confidence曲线自动选取最佳阈值0.925，实现了1.00的最高F1分数。

**🔧 技术方法**

使用YOLOv8分割模型、关键点检测CNN、深度图与点云处理算法，计算最大距离、面积、体积等几何手工特征，并在实验4中将其作为预测模型的输入。

**📊 数据集**

数据集为2024‑2025年收集的牛体况分数与对应深度图/点云数据，包含不同时间点的直方图分布。

**📈 对比分析**

通过F1‑confidence曲线对YOLO分割模型进行评估，最大F1为1.00，阈值0.925被选为最佳；在实验4中使用手工特征训练的预测模型表现优异，但具体指标未给出。

**⚠️ 局限性**

局限性包括：模型依赖于高质量的深度图与点云，缺乏跨场景鲁棒性与实时性评估；手工特征选择可能缺乏可解释性，且只针对特定解剖关键点。

---

## 63. Relative Wasserstein Angle and the Problem of the $W_2$-Nearest Gaussian Distribution

**arXiv ID:** 2601.22355 | [PDF](https://arxiv.org/pdf/2601.22355v1)

**作者:** Binshuai Wang `[一作]` (George Washington University), Peng Wei `[通讯]` (George Washington University)

**通讯引用:** 5563 | [OpenAlex ID](https://openalex.org/A5025147595)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了相对 Wasserstein 角度（RW_2 角）和正交投影距离两种几何量，用来量化经验分布与高斯分布的偏差，并给出了 1 维闭式解和高维随机 Riemannian 优化方法；

**💡 创新点**

创新点在于：1）揭示 RW_2 空间的锥体几何，证明填充锥面为平坦，从而可以在此空间中严格定义角度和投影；2）将高斯逼近转化为投影问题，证明矩匹配高斯通常不是 W_2 最近高斯；3）提供 1 维闭式公式和高维高效算法；

**🔧 技术方法**

使用了最优传输理论、Wasserstein 距离、相对平移不变 Wasserstein（RW_2）几何、半离散对偶公式、随机 Riemannian 变分优化、梯度投影等技术；

**📊 数据集**

实验数据包括：合成的高斯混合模型；CIFAR‑10、MNIST、CelebA‑64、LSUN‑Church（使用 Inception‑V3 提取特征）等公开数据集；

**📈 对比分析**

方法上与传统矩匹配高斯进行对比，计算 RW_2 角度和投影距离；结果表明 RW_2 最近高斯的角度更小、方差更低，说明更稳健；在 FID 评估中，该近似能够提供更接近真实分布的高斯替代，提升指标表现；

**⚠️ 局限性**

局限性包括：1）仅考虑零均值或已中心化的分布；2）高维解依赖数值优化，收敛速度与初值有关；3）当前仅对高斯族给出闭式或高效算法，其他分布族（如非参数化曲线）仍待推广；4）对非平滑或离散分布的处理尚未充分讨论。

---

## 64. Language Model Circuits Are Sparse in the Neuron Basis

**arXiv ID:** 2601.22594 | [PDF](https://arxiv.org/pdf/2601.22594v1)

**作者:** Aryaman Arora `[一作]` (Stanford University), Sarah Schwettmann `[通讯]` (Transluce)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文开发了一套基于MLP激活的稀疏电路追踪管线，并演示其在大型语言模型中的可解释性。

**💡 创新点**

创新点在于证明MLP激活本身就是稀疏特征基，并通过RelP归因方法让其在电路稀疏度与SAE等稀疏字典相当，甚至更优。

**🔧 技术方法**

使用的技术包括RelP梯度归因、平均抑制干预、对比SAE字典、以及对Transformer MLP的非线性替代与直通梯度的实现。

**📊 数据集**

实验数据集涵盖了主语-动词一致性(SVA)基准以及多跳州首府推理数据集，均使用Llama 3.1 8B等模型。

**📈 对比分析**

通过faithfulness与completeness指标比较，MLP激活+RelP在约200–300个神经元的稀疏电路下即可实现近乎完美的faithfulness和低completeness，优于IG方法和传统SAE字典。

**⚠️ 局限性**

主要局限在于电路仍包含大量神经元导致可解释性困难，缺乏高效的聚类和自动描述机制，并且实现效率不够高。

---

## 65. AI-Enabled Waste Classification as a Data-Driven Decision Support Tool for Circular Economy and Urban Sustainability

**arXiv ID:** 2601.22418 | [PDF](https://arxiv.org/pdf/2601.22418v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 66. AI Narrative Breakdown. A Critical Assessment of Power and Promise

**arXiv ID:** 2601.22255 | [PDF](https://arxiv.org/pdf/2601.22255v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 67. RoboStriker: Hierarchical Decision-Making for Autonomous Humanoid Boxing

**arXiv ID:** 2601.22517 | [PDF](https://arxiv.org/pdf/2601.22517v1)

**作者:** Kangning Yin `[一作]` (Shanghai Jiao Tong University), Weinan Zhang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 18081 | [OpenAlex ID](https://openalex.org/A5090720315)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出一种三阶段的层级框架，先用模仿学习训练可执行多样化人类拳击动作的低层运动追踪器，然后将这些动作压缩到一个受约束的潜在空间，再在该空间内用潜在空间神经虚构自我对弈（LS‑NFSP）训练双人拳击机器人的对抗策略，从而实现全自主的 humanoid 拳击。

**💡 创新点**

创新点包括：1) 将高层策略与低层物理执行完全解耦，利用潜在空间的单位球面约束保证动作物理可行性；2) 在潜在空间而非原始动作空间进行 NFSP，从而显著降低非平稳性、避免运动崩溃；3) 引入对抗性运动先验（AMP）和行为预热阶段，提高训练收敛速度和策略质量。

**🔧 技术方法**

核心技术包括：DeepMimic 模仿学习、潜在空间学习与高斯→单位球正则化、潜在空间神经虚构自我对弈（LS‑NFSP）、PPO 强化学习、对抗性运动先验（AMP）以及大规模并行仿真（Isaac Lab/Omniverse）。

**📊 数据集**

使用公开的人类动作捕捉数据集训练低层运动追踪器；随后在 Unitree G1 29-DOF 虚拟机器人上进行自我对弈训练；在真实 Unitree G1 机器人上进行 sim‑to‑real 验证。

**📈 对比分析**

与基线（PPO‑Only、Naïve Self‑Play、Fictitious Self‑Play、Action‑Space Self‑Play、Static‑Target Specialist、无 AMP 等）比较，LS‑NFSP 在交叉赛中取得最高胜率（最高 100% 对 29‑DOF Action‑Space SP），在进攻命中率、参与率、基础稳定性、扭矩平滑度等指标上均显著优于对手，证明在潜在空间中进行自我对弈能更好地平衡物理稳定性与策略探索。

**⚠️ 局限性**

局限性包括：1) 对计算资源依赖较大（需要高性能 GPU 与大规模仿真）；2) 只验证了两人对抗的拳击场景，无法直接推广到更复杂多方或多任务环境；3) 虽然能迁移至真实机器人，但对不同机器人形态或更大物理扰动的鲁棒性尚未充分评估。

---

## 68. BayesFlow: A Probability Inference Framework for Meta-Agent Assisted Workflow Generation

**arXiv ID:** 2601.22305 | [PDF](https://arxiv.org/pdf/2601.22305v1)

**作者:** Bo Yuan `[一作]` (Georgia Institute of Technology), Balasubramaniam Srinivasan `[通讯]` (Amazon Web Services AI Lab)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过贝叶斯后验采样构造工作流，提出 BWG 框架及 BayesFlow 算法，实现自动生成高质量多步工作流。

**💡 创新点**

将工作流生成视为能量模型的后验采样，结合并行 look‑ahead 采样与序列化全局改进，提供理论收敛保证并显著提升多样性。

**🔧 技术方法**

基于 LLM 的先验、能量模型奖励、并行采样（SMC式 look‑ahead）、MCTS式全局改进、文本梯度编辑以及无训练的推理框架。

**📊 数据集**

六个任务域（MATH、GSM8K、DROP、HotpotQA、GPQA、MMLU‑Pro）以及编码基准 MBPP。

**📈 对比分析**

与零样本提示、CoT、CoT‑SC 以及自动工作流基线 ADAS、AFlow、MaAS 进行对比，BayesFlow 在大多数数据集上平均提升 4–9 个百分点，最高超 AFlow 约 4.6%。

**⚠️ 局限性**

计算成本受 look‑ahead 采样占主导，缺乏自适应控制；未能充分重用采样轨迹；实验覆盖范围有限，尚未验证更大模型或多模态场景。

---

## 69. Transform-Augmented GRPO Improves Pass@k

**arXiv ID:** 2601.22478 | [PDF](https://arxiv.org/pdf/2601.22478v1)

**作者:** Khiem Le `[一作]` (University of Notre Dame), Nitesh V. Chawla `[通讯]` (University of Notre Dame)

**通讯引用:** 58771 | [OpenAlex ID](https://openalex.org/A5068157871)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对大型语言模型的推理任务，提出了通过生成语义等价的题目变体并对其联合计算优势的TA‑GRPO方法，解决了GRPO的多样性崩溃和梯度消失问题。

**💡 创新点**

创新点在于：①使用变换增强（paraphrase、变量重命名、格式变换）构造同义题组；②在题组内进行优势池化，从而保证即使原题过难或过易也能产生梯度；③给出理论分析证明梯度消失概率下降、训练测试差距缩小，并推导出池化优势公式。

**🔧 技术方法**

技术主要包括：强化学习与可验证奖励（RLVR）的GRPO框架、文本变换生成、优势池化与策略梯度更新，以及对梯度消失的概率和泛化误差的理论证明。

**📊 数据集**

使用的主要数据集是MATH（训练集）以及AMC12、AIME24/25、OlympiadBench等竞赛数学基准和GPQA‑Diamond、Minerva等跨领域 OOD 评测集。

**📈 对比分析**

与基准模型和标准GRPO进行对比，实验显示在Pass@32上提升了最高9.84个百分点（AMC12），在GPQA‑Diamond OOD 上提升5.05个百分点；Pass@1 变化小，Pass@k 随 k 递增而差距扩大，表明多样性得到提升。

**⚠️ 局限性**

局限性包括：①变换质量需人工或GPT‑4控制，若语义失真会导致错误；②实验仅在中小规模模型（Qwen3‑1.7B/4B）上验证，未探究更大模型的效果；③依赖可验证奖励的任务，无法直接迁移到非可验证领域。

---

## 70. SSL: Sweet Spot Learning for Differentiated Guidance in Agentic Optimization

**arXiv ID:** 2601.22491 | [PDF](https://arxiv.org/pdf/2601.22491v1)

**作者:** Jinyang Wu `[一作]` (Tsinghua University), Jiaming Xu `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出甜点学习（SSL）框架，通过分层甜点区奖励为强化学习提供差异化引导，提升智能体的性能与样本效率

**💡 创新点**

创新点在于将解决空间离散为层级甜点区，并在奖励中加入分层甜点分数，理论上保证最优解排序并提升梯度信噪比

**🔧 技术方法**

使用奖励塑造与策略梯度算法（如GRPO）相结合的RLVR技术，构造甜点奖励与多任务适配策略

**📊 数据集**

在12个基准上进行评估，涵盖GUI感知、短/长周期规划和复杂推理任务（如Sudoku、Maze、ARC-AGI），使用公开数据集如GUI-R1、SWEET-AGI等

**📈 对比分析**

相较于传统二进制奖励和连续奖励基线，SSL在多数任务上提升10–30%准确率，样本效率提升至2.5×，并能在跨任务迁移中保持优势

**⚠️ 局限性**

局限在于甜点区划分需手工设计，可能在某些任务或域中不易自适应，且对奖励设计的敏感性仍需进一步研究

---

## 71. Why Reasoning Fails to Plan: A Planning-Centric Analysis of Long-Horizon Decision Making in LLM Agents

**arXiv ID:** 2601.22311 | [PDF](https://arxiv.org/pdf/2601.22311v1)

**作者:** Zehong Wang `[一作]` (University of Notre Dame), Yanfang Ye `[通讯]` (University of Notre Dame)

**通讯引用:** 5057 | [OpenAlex ID](https://openalex.org/A5027601906)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了大型语言模型（LLM）在长周期决策中的推理与规划差异，证明单步贪婪推理在长期任务中会产生不可逆的早期误差，并基于MCTS提出一种最低限度的规划机制（显式lookahead、价值反向传播与回退式承诺），在控制环境下显著提升LLM代理的连贯规划性能。

**💡 创新点**

创新点包括：① 将LLM推理建模为一步步贪婪策略并从理论上证明其在长周期任务中可导致无穷大子最优差距；② 证明仅增宽搜索宽度（Beam Search）无法解决该问题；③ 提出仅需一阶lookahead即可严格优于纯贪婪推理的规划框架；④ 将该规划框架统一应用于多种代理框架并展示其跨模型、跨任务的优越性。

**🔧 技术方法**

主要技术：基于MCTS的显式路径模拟与评估；轨迹级奖励估计与反向价值传播；回退式承诺（receding‑horizon）策略；行动剪枝（限制候选动作集）与轨迹记忆（避免重复评估）。

**📊 数据集**

实验数据集：知识图谱问答（KGQA）任务的CWQ、WebQSP和GrailQA（采用oracle‑structured子图），以及工具使用环境ALFWorld。

**📈 对比分析**

与单步推理、Beam Search、浅层Lookahead、ToG/PoG框架以及ReAct/Reflexion等方法对比，结果显示该规划框架在所有数据集上均实现显著的准确率提升，且在token预算相同或更低时表现更好；在LLM规模上，使用LLaMA‑8B即可超越GPT‑4o。

**⚠️ 局限性**

限制：仅在确定性、完全可观测、评估信号已知的控制环境中验证；不涉及随机性、部分可观测或需要学习世界模型的复杂环境；对评估信号的依赖可能导致在信号噪声大时性能下降。

---

## 72. Constructing BERT Models: How Team Dynamics and Focus Shape AI Model Impact

**arXiv ID:** 2601.22505 | [PDF](https://arxiv.org/pdf/2601.22505v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 73. AI and My Values: User Perceptions of LLMs' Ability to Extract, Embody, and Explain Human Values from Casual Conversations

**arXiv ID:** 2601.22440 | [PDF](https://arxiv.org/pdf/2601.22440v1)

**作者:** Bhada Yun `[一作]` (ETH Zurich), April Yi Wang `[通讯]` (ETH Zurich)

**通讯引用:** 461 | [OpenAlex ID](https://openalex.org/A5046673805)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发并验证了VAPT工具，评估LLM在提取、体现和解释用户价值观的能力。

**💡 创新点**

提出可复用的感知工具包和三阶段评估流程，并首次系统性研究AI对用户价值的感知与信任。

**🔧 技术方法**

基于Claude Sonnet4 LLM、Schwartz PVQ-RR 价值问卷、对话日志以及可视化图谱等技术。

**📊 数据集**

使用20名多文化受试者的一个月日常聊天日志和对应的PVQ-RR自评结果。

**📈 对比分析**

通过定性访谈与量化指标（自评与LLM预测的相关性、Likert满意度）对比，发现约63.6%的值预测误差≤1，用户对价值图整体认可度高。

**⚠️ 局限性**

样本规模有限、年龄偏年轻、语言多样性不足、对价值框架的文化偏差及LLM解释可能导致自我观念改变。

---

## 74. FunPRM: Function-as-Step Process Reward Model with Meta Reward Correction for Code Generation

**arXiv ID:** 2601.22249 | [PDF](https://arxiv.org/pdf/2601.22249v1)

**作者:** Ruiyi Zhang `[一作]` (University of California, San Diego), Pengtao Xie `[通讯]` (University of California, San Diego)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 FunPRM，一种面向代码生成的过程奖励模型，利用函数层级分解代码并在最佳‑N测试时缩放中对生成结果进行选择；

**💡 创新点**

创新点包括：① Chain‑of‑Function 提示，将函数视为 PRM 步骤，实现意义明确的中间步骤；② 元学习奖励修正机制，借助单元测试得到的干净最终奖励去除 Monte‑Carlo 估计的噪声奖励，从而提升 PRM 训练质量；

**🔧 技术方法**

使用技术包括：Process Reward Model、Monte‑Carlo 采样奖励估计、元学习（Meta‑Learning）奖励修正、LoRA 微调、best‑of‑N 选择、单元测试评估、函数级提示等；

**📊 数据集**

实验数据集：LiveCodeBench、BigCodeBench、HumanEval+、MBPP+（使用 EvalPlus 评估）;

**📈 对比分析**

与 Self‑Certainty、ORM、Skywork‑PRM 等测试时缩放基线在多种基模型（O4‑mini High、Qwen3‑30B‑A3B、GPT‑4o‑mini 等）上对比，FunPRM 在 LiveCodeBench 上以 80.9% pass@1（最高）取得 SOTA，整体性能均优于基线；

**⚠️ 局限性**

局限性：仍需要人工审查代码安全与正确性；奖励修正涉及额外计算开销；仅在单轮生成且无公共测试用例的场景下有效；对极其复杂或多轮交互式任务的适应性尚待验证。

---

## 75. FOTBCD: A Large-Scale Building Change Detection Benchmark from French Orthophotos and Topographic Data

**arXiv ID:** 2601.22596 | [PDF](https://arxiv.org/pdf/2601.22596v1)

**作者:** Abdelrrahman Moubane `[一作]` `[通讯]` (Retgen AI), Abdelrrahman Moubane (Retgen AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

发布了FOTBCD建筑变化检测数据集，包括大规模二进制版本和实例级子集，覆盖法国28个省份，解决地理同质性与规模不足的问题。

**💡 创新点**

通过国别级别的大规模地理多样性提升跨区域泛化能力，并提供人工验证的评价集，展示地理多样性对模型鲁棒性的显著作用。

**🔧 技术方法**

使用HybridSiam-CD基准模型，融合冻结的DINOv3视觉Transformer与ResNet34特征，配合Lovasz hinge + BCE损失进行训练。

**📊 数据集**

主要使用FOTBCD-Binary和FOTBCD-Instances，并与LEVIR-CD+、WHU-CD等公开基准数据集进行比较。

**📈 对比分析**

在交叉域实验中，模型在FOTBCD上训练后对WHU-CD与LEVIR-CD+的IoU分别为0.6974和0.2984，显示跨域性能优于仅用单域数据训练的模型；相反，LEVIR-CD+或WHU-CD训练的模型在FOTBCD上的IoU低于0.35。

**⚠️ 局限性**

局限包括仅覆盖法国本土，缺乏道路、植被等非建筑变化类别，且训练集自动生成标签可能存在噪声。

---

## 76. PromptMAD: Cross-Modal Prompting for Multi-Class Visual Anomaly Localization

**arXiv ID:** 2601.22492 | [PDF](https://arxiv.org/pdf/2601.22492v1)

**作者:** Duncan McCain `[一作]` (Clemson University), Fatemeh Afghah `[通讯]` (Clemson University)

**通讯引用:** 3569 | [OpenAlex ID](https://openalex.org/A5035395012)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种 PromptMAD 框架，通过跨模态提示和文本引导的扩散模型实现多类别视觉异常检测与定位。

**💡 创新点**

创新点包括：①将 CLIP 文本提示与视觉特征融合，实现语义增强的重建；②引入 Focal Loss 平衡像素级不平衡；③结合多尺度卷积、Transformer 注意力和扩散迭代细化的监督分割器，提升细粒度定位精度。

**🔧 技术方法**

使用的技术包括：CLIP 文本编码、Transformer 双向解码器、多尺度 EfficientNet‑B4 特征、Focal Loss、基于 U‑Net 的扩散细化，以及深度卷积+Transformer 注意力的分割器。

**📊 数据集**

实验数据集为 MVTec‑AD，包含 15 类工业物品与纹理，总计 3,629 张正常训练图像和 1,725 张测试图像（含异常）。

**📈 对比分析**

与 OneNIP、DRAEM、UniAD 等方法对比，PromptMAD 在像素级 AUC 从 97.81% 提升至 98.35%，AP 提升至 66.54%，在多数类别（尤其是纹理类）表现显著优势；推理速度保持实时（约 193 FPS）。

**⚠️ 局限性**

局限性在于对极少异常样本的定位仍受伪异常生成质量影响，跨模态提示需人工编写，对新类别扩展有一定成本；在部分对象类（如 grid、screw）的 AP 略有下降。

---

## 77. High-utility Sequential Rule Mining Utilizing Segmentation Guided by Confidence

**arXiv ID:** 2601.22179 | [PDF](https://arxiv.org/pdf/2601.22179v1)

**作者:** Chunkai Zhang `[一作]` (Harbin Institute of Technology), Philip S. Yu `[通讯]` (University of Illinois at Chicago)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一种基于置信度分段的高效序列规则挖掘算法RSC，解决传统LRE方法导致的冗余效用计算问题。

**💡 创新点**

创新点在于将规则生成归一化为一次分段计算并共享效用，同时引入“减少剩余效用”上界以及效用链表结构以降低搜索空间并支持重复项。

**🔧 技术方法**

技术包括置信度引导分段、效用链表（Utility‑Linked Table）、减少剩余效用（Reduced Remaining Utility）上界以及基于支持的前向剪枝策略。

**📊 数据集**

实验使用了Bible、Kosarak10k、Sign、Leviathan及多规模Syn系列等公开数据集。

**📈 对比分析**

与最先进的TotalSR、TotalUS等算法对比，RSC在不同阈值下运行时间平均缩短50–80%，内存消耗也略低。

**⚠️ 局限性**

局限在于目前仅支持完全有序规则，对非有序或带间隙/闭集约束的高效规则挖掘尚未覆盖，并且在极稠密重复项少的数据上改进空间有限。

---

## 78. Riemannian Lyapunov Optimizer: A Unified Framework for Optimization

**arXiv ID:** 2601.22284 | [PDF](https://arxiv.org/pdf/2601.22284v1)

**作者:** Yixuan Wang `[一作]` (University of Florida), Warren E. Dixon `[通讯]` (University of Florida)

**通讯引用:** 16880 | [OpenAlex ID](https://openalex.org/A5032651215)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种新的优化算法家族，称为Riemannian Lyapunov Optimizers (RLOs)，将经典优化器统一在一个几何框架内。

**💡 创新点**

创新点在于将优化重新解释为在Riemannian参数流形上的扩展状态离散时间控制动态系统，并通过构造严格的Lyapunov函数来证明收敛性。

**🔧 技术方法**

使用了控制理论的几何方法，特别是Riemannian几何和Lyapunov稳定性理论。

**📊 数据集**

在CIFAR-10和ImageNet等大型基准数据集上进行了验证。

**📈 对比分析**

与现有的优化器（如AdamW和Lion）进行了比较，RLO-Λ在多个架构上表现出色，尤其是在Vision Transformers上，显示出显著的收敛速度和最终准确性。

**⚠️ 局限性**

限制在于对不同模型的适应性可能存在差异，特别是在小型模型上，提升参数的选择可能影响性能。

---

## 79. DAJ: Data-Reweighted LLM Judge for Test-Time Scaling in Code Generation

**arXiv ID:** 2601.22230 | [PDF](https://arxiv.org/pdf/2601.22230v1)

**作者:** Peijia Qin `[一作]` (University of California, San Diego), Pengtao Xie `[通讯]` (University of California, San Diego)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种基于双层数据重加权的LLM判决器，用以提升代码生成任务中的测试时扩展性能。

**💡 创新点**

创新点在于将数据重加权策略与LLM-as-a-Judge相结合，自动学习域级或实例级重要性权重，专门针对易难不平衡、任务分布失配和轨迹差异等分布偏移问题。

**🔧 技术方法**

技术上采用了推理式LLM判决器、可验证奖励机制、偏好优化与强化学习目标，以及双层优化框架实现数据权重自适应更新。

**📊 数据集**

主要使用了LiveCodeBench和BigCodeBench两大代码生成基准，训练集通过低成本模型生成，元集则选取高质量模型产生的样本。

**📈 对比分析**

与零拷贝模型和现有测试时扩展方法相比，所提方法在LiveCodeBench总体准确率提升至84.7%（比最强基线高7.6个百分点），在BigCodeBench达到35.9%（高于Skywork‑o1 PRM的35.2%）。

**⚠️ 局限性**

局限性包括对元集选择的依赖、双层优化训练成本较高，以及在极端难题上的提升仍有限。

---

## 80. 5G LDPC Codes as Root LDPC Codes via Diversity Alignment

**arXiv ID:** 2601.22470 | [PDF](https://arxiv.org/pdf/2601.22470v1)

**作者:** Hyuntae Ahn `[一作]` (Sungkyunkwan University), Sang-Hyo Kim `[通讯]` (Sungkyunkwan University)

**通讯引用:** 16644 | [OpenAlex ID](https://openalex.org/A5100756277)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

针对非平稳块衰落信道，提出了利用布尔函数演化（DivE）分析LDPC码的多样性，并基于此设计贪心块映射搜索实现5G NR LDPC码的全多样性

**💡 创新点**

引入DivE以精确追踪迭代BP中变量节点的多样性演变，定义一般根检查（generalized rootcheck）并利用其进行块映射优化，显著提升非平稳信道性能

**🔧 技术方法**

布尔函数演化、迭代最小和算法（MSD）、贪心搜索、随机化块映射、基于Protograph的QC‑LDPC码

**📊 数据集**

5G NR基准图BG1和BG2（Z=240、Z=20）

**📈 对比分析**

与随机块映射对比，得到更陡峭的高SNR斜率和更低的BLER，示例中在BG1与BG2上分别实现了更高码率的全多样性

**⚠️ 局限性**

仅在两块衰落（M=2）情形下验证，且在原始码率（R=1/2）下仍难以实现全多样性，原因在于5G NR LDPC的高码率预编码结构；缺乏对M>2的推广

---

## 81. Game-Based and Gamified Robotics Education: A Comparative Systematic Review and Design Guidelines

**arXiv ID:** 2601.22199 | [PDF](https://arxiv.org/pdf/2601.22199v1)

**作者:** Syed T. Mubarrat `[一作]` (Purdue University), Dominic Kao `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 874 | [OpenAlex ID](https://openalex.org/A5021762095)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对 2014–2025 年间关于游戏化（gamification）与基于游戏的学习（GBL）在机器人教育中的研究进行系统综述与对比分析，涵盖学习环境、教学模式、技能层级与沉浸技术使用等维度。

**💡 创新点**

首次采用 PRISMA 规范对 95 篇研究进行系统评估，比较 GBL 与 gamification 在效果、情感与技能培养方面的差异，并提出设计空间与八条未来研究方向。

**🔧 技术方法**

使用文献检索、主题编码、交叉表统计、卡方检验、Cramér's V、逻辑回归、MMAT 与 ROBINS‑I 等工具进行定量与定性分析。

**📊 数据集**

共收集 95 篇研究论文，来源于 4 个数据库（Scopus、ProQuest、IEEE Xplore、ACM DL），总计 12,485 条记录。

**📈 对比分析**

通过 2×2 交叉表和多元逻辑回归比较 GBL 与 gamification 的学习收益与动机，发现两者在学习成果和情感结果上相近，正式课堂更易报告量化提升。

**⚠️ 局限性**

研究样本量普遍偏小，多为描述性设计，缺乏随机对照与长期跟踪；沉浸技术（VR、Haptics）研究不足，技能层级评估不够细致。

---

## 82. Quantum $(r,δ)$-Locally Recoverable BCH and Homothetic-BCH Codes

**arXiv ID:** 2601.22567 | [PDF](https://arxiv.org/pdf/2601.22567v1)

**作者:** Carlos Galindo `[一作]` (Universitat Jaume I), Ryutaroh Matsumoto `[通讯]` (Institute of Science Tokyo)

**通讯引用:** 1587 | [OpenAlex ID](https://openalex.org/A5005174170)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构造并分析基于BCH与H‑BCH码的量子(r,δ)-局部可恢复码（量子LRC），并证明其可达单一式界，实现最优纯量子LRC。

**💡 创新点**

首次将BCH和H‑BCH的子域子码与仿射多项式评价码相结合，利用其欧氏/赫米特自正交性得到量子LRC；通过构造合适的指数集Δ，获得可调节局部性与距离的最优码。

**🔧 技术方法**

采用子域子码、仿射变换评价、BCH/ H‑BCH 码、欧氏/赫米特自正交性、短化/ puncture、追踪映射等经典码理论工具；利用量子稳定子码与经典LRC的对应关系。

**📊 数据集**

无实测数据，全部为理论构造与参数证明；论文给出多组参数表，展示不同q、δ、r、λ等情形。

**📈 对比分析**

与已有最优量子LRC 方案对比，本文构造的码在长度、信息符号数、局部性和距离等指标上均满足或超过已知界限；表格显示多种情形下实现最优（达到 Singleton‑like 界）且长度可无上界。

**⚠️ 局限性**

仅适用于满足特定阶除法、λ 约束的参数；构造复杂度高，需满足自正交性与连续整数集条件；并未讨论非纯码或码率与纠错距离权衡的实际实现问题。

---

## 83. HetCCL: Accelerating LLM Training with Heterogeneous GPUs

**arXiv ID:** 2601.22585 | [PDF](https://arxiv.org/pdf/2601.22585v1)

**作者:** Heehoon Kim `[一作]` (Moreh Inc.), Jaejin Lee `[通讯]` (Seoul National University)

**通讯引用:** 2310 | [OpenAlex ID](https://openalex.org/A5100767175)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 HetCCL，支持 NVIDIA 与 AMD GPU 的跨厂商集体通信，允许在同一训练任务中同时使用两种 GPU。

**💡 创新点**

创新点在于：① 利用 RDMA 实现跨厂商 GPU 直接内存互访；② 通过运行时 API 抽象层和多平台内核编译，既能调用原厂 NCCL/RCCL 又能统一管理；③ 在不改动应用代码的前提下保持同源环境的近原生性能。

**🔧 技术方法**

使用 RDMA、NVIDIA NCCL、AMD RCCL、TACC 抽象层、CUDA/HIP 内核编译、DeepSpeed ZeRO 并结合 MPI/SLURM 等工具。

**📊 数据集**

使用 GPT 与 LLaMA 大模型（1B、7B、13B 等）训练，数据集包括 WikiText‑103 等。

**📈 对比分析**

与 NCCL、RCCL、MPI 基准对比；在同源环境几乎等同原生性能，在异源环境可达 90%+ 效率，训练吞吐提升 1.48×‑2.97×，并保持模型精度一致。

**⚠️ 局限性**

局限性：目前仅支持 NVIDIA 与 AMD GPU；需要 RDMA‑capable 网络和驱动；对更广泛的多厂商/多加速器混合场景尚未覆盖。

---

## 84. Models Under SCOPE: Scalable and Controllable Routing via Pre-hoc Reasoning

**arXiv ID:** 2601.22323 | [PDF](https://arxiv.org/pdf/2601.22323v1)

**作者:** Qi Cao `[一作]` (University of California, San Diego), Pengtao Xie `[通讯]` (University of California, San Diego)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研发了一种可扩展且可控制的模型路由框架 Scope，通过预先推理估计各模型的准确率和代价，实现动态路由。

**💡 创新点**

创新点在于①使用行为指纹检索替代固定模型名称，支持对新模型零训练适配；②预测准确性与 token 成本，构建可调节效用函数，让用户可控制成本-精度权衡。

**🔧 技术方法**

采用检索增强指纹构造、链式推理生成预测器、监督微调+GRPO 强化学习、归一化效用函数与锚点校准等技术。

**📊 数据集**

使用 Scope‑60K（13 种 LLM 的查询-模型-准确率-token 数据）和 Scope‑250（代表性锚点集合）进行训练与评估，并在 Test Set 与 OOD 集合（如 AIME、Humanity's Last Exam 等）上测试。

**📈 对比分析**

与各类基线路由器（SVM、KNN、MLP、Graph Router、xRouter 等）对比，Scope 在高 α 下提升 24–25% 准确率，低 α 下成本降低 95% 以上，在多预算场景下保持最佳的成本-精度 Pareto 前沿。

**⚠️ 局限性**

局限性包括需人工挑选锚点集合、检索开销、效用系数调参的复杂度，以及对极端 OOD 或极大模型集合的泛化尚未充分验证。

---

## 85. An Effective Energy Mask-based Adversarial Evasion Attacks against Misclassification in Speaker Recognition Systems

**arXiv ID:** 2601.22390 | [PDF](https://arxiv.org/pdf/2601.22390v1)

**作者:** Chanwoo Park `[一作]` (Korea University), Chanwoo Kim `[通讯]` (Korea University)

**通讯引用:** 6679 | [OpenAlex ID](https://openalex.org/A5019347107)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6215c339-3735-4be3-8a07-5bbb7004712d` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出并实现了Masked Energy Perturbation（MEP）方法，用于在声纹识别系统中生成对抗样本，兼顾语音质量与攻击成功率。

**💡 创新点**

创新点在于将人类听觉掩蔽效应与频谱能量掩蔽相结合，先在能量低的频率区块上进行掩蔽，再对剩余高能区块施加对抗扰动，从而实现对抗效果与语音感知质量的平衡。

**🔧 技术方法**

使用STFT、Mel滤波器、log Mel谱、深度学习声纹编码器（ResNetSE34-L/V、ECAPA‑TDNN）、基于梯度的攻击算法（FGSM、I‑FGSM、MI‑FGSM、PGD、MEP、I‑MEP），以及对抗扰动生成与裁剪策略。

**📊 数据集**

实验数据集为LibriSpeech（16 kHz，无背景噪声的音频），共116名说话人、113条录音，采用VoxCeleb预训练的声纹模型。

**📈 对比分析**

在与传统对抗攻击比较时，MEP与I‑MEP在PESQ（3.68–3.77）和SNR（≈38 dB）上显著优于FGSM、I‑FGSM、MI‑FGSM和PGD；EER提升更小，表明语音质量损失更低，同时攻击效果仍然可观。

**⚠️ 局限性**

局限性包括：仅在干净语料上验证，缺乏对嘈杂或实时环境的评估；只针对白盒声纹模型，未测试黑盒或跨模型迁移；未考察对语音生成/识别系统的鲁棒性或实际部署场景。

---

## 86. Context Structure Reshapes the Representational Geometry of Language Models

**arXiv ID:** 2601.22364 | [PDF](https://arxiv.org/pdf/2601.22364v1)

**作者:** Eghbal A. Hosseini `[一作]` (Google DeepMind), Andrew Kyle Lampinen `[通讯]` (Princeton Neuroscience Institute)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

分析大型语言模型在不同的上下文学习任务（自然语言、网格世界、少样本推理与谜语）中内部表示的几何演变，重点考察表示直线化与模型行为之间的关系。

**💡 创新点**

提出表示直线化是连续预测任务的通用几何标志，但在结构化ICL任务中失效，表明LLM采用“工具箱”式的多种机制适配不同任务结构。

**🔧 技术方法**

使用Transformer残差激活、PCA、有效维度、Menger曲率、延伸度等几何度量，以及对不同上下文长度与结构的实验控制；模型为Gemma‑2‑27B。

**📊 数据集**

数据集包括LAMBADA（长距依赖）、自定义Grid World（单层与双层结构）、少样本推理任务（包含BIG‑bench Riddle Benchmark）以及其他少样本任务。

**📈 对比分析**

通过与随机文本对照、不同上下文长度（短、长、重复）和行为指标（logit差距、准确率）比较，发现自然语言与网格世界任务中直线化随上下文增加显著提升且与行为提升相关；而在少样本和谜语任务中直线化不随表现变化。

**⚠️ 局限性**

实验仅针对Gemma‑2，未进行因果干预；几何度量范围有限；缺乏跨模型/规模的验证，可能导致结果泛化受限。

---

## 87. Gradual Fine-Tuning for Flow Matching Models

**arXiv ID:** 2601.22495 | [PDF](https://arxiv.org/pdf/2601.22495v1)

**作者:** Gudrun Thorkelsdottir `[一作]` (University of Illinois Urbana-Champaign), Arindam Banerjee `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 24899 | [OpenAlex ID](https://openalex.org/A5014459472)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `40105733-5154-44cd-8090-a8cab9e64b07` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种针对流匹配模型的渐进微调（GFT）框架，使得在目标分布样本可用的情况下能够稳健地适配预训练模型；

**💡 创新点**

创新点在于用温度控制的中间目标插值实现对预训练漂移与目标漂移的连续混合，并证明在任意源-目标耦合下仍保持收敛性；

**🔧 技术方法**

采用流匹配（Flow Matching）与随机微分方程理论、Girsanov定理、条件流匹配（CFM）以及温度退火策略；

**📊 数据集**

在WILDS基准数据集上进行实验，主要使用Camelyon17、RxRx1和FMoW三组图像数据集；

**📈 对比分析**

与传统从头训练和标准CFM微调进行对比，GFT在FID、平均路径长度、瞬时方差和Spearman相关性等指标上均表现出更快收敛、更短路径和更高稳定性；

**⚠️ 局限性**

局限性包括仍需人工设定退火温度表，且在极大分布漂移时需要更长训练时间，未对奖励驱动或在线控制方法做直接比较。

---

## 88. MixQuant: Pushing the Limits of Block Rotations in Post-Training Quantization

**arXiv ID:** 2601.22347 | [PDF](https://arxiv.org/pdf/2601.22347v1)

**作者:** Sai Sanjeet `[一作]` (State University of New York at Buffalo), Nicholas J. Fraser `[通讯]` (Advanced Micro Devices)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文针对少位元语言模型的激活离群值问题，提出了一种基于块Hadamard旋转的等化框架，利用Permutation Diffusion（MassDiff）算法在计算图中均衡每块激活的ℓ₁质量，然后再进行块Hadamard旋转并量化，避免在线旋转带来的推理开销。

**💡 创新点**

创新点在于：①首次给出块Hadamard旋转抑制离群值的非渐进性、确定性与概率性分析；②发现离群抑制受每块最大ℓ₁质量限制；③设计MassDiff贪婪算法在校准集上等化块质量；④通过识别Permutation-Equivariant区域，将等化Permutation嵌入权重，完成无推理时延的部署。

**🔧 技术方法**

技术包括：块Hadamard旋转、Permutation Diffusion（MassDiff）等化算法、Permutation-Equivariant区域识别与融合、GPTQ、Qronos等错误校正量化算法，以及Brevitas框架实现。

**📊 数据集**

实验使用Llama3与Qwen3大模型，校准集为WikiText2训练集（128序列×2048标记），评估集为WikiText2测试集以及LightEval的5个推理任务（ARC、HellaSwag、PIQA、Winogrande）。

**📈 对比分析**

与基线MR‑GPTQ/BRQ、MR‑RTN、MR‑Qronos、BRQ‑Spin等进行比较，结果显示在INT4、FP4和MXFP4等不同量化格式下，本文方法在WikiText2 perplexity上接近全精度，零样本准确率平均保持≥91%（MXFP4下≥95%），显著优于传统块旋转基线。

**⚠️ 局限性**

局限性包括：分析基于最坏情况，无法完整预测所有模型/数据/格式组合的性能；在已具备分组缩放的MX格式下，离群抑制重要性降低，收益减小；未考虑权重量化离群抑制或其他旋转结构（如Givens矩阵）等。

---

## 89. From Retrieving Information to Reasoning with AI: Exploring Different Interaction Modalities to Support Human-AI Coordination in Clinical Decision-Making

**arXiv ID:** 2601.22338 | [PDF](https://arxiv.org/pdf/2601.22338v1)

**作者:** Behnam Rahdari `[一作]` (Stanford University), Shriti Raj `[通讯]` (Stanford University)

**通讯引用:** 302 | [OpenAlex ID](https://openalex.org/A5083865738)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

对12名临床医生在两种真实病历情景下使用LLM聊天、视觉决策界面和语音交互三种模式进行思考实验与访谈，探讨其协同决策行为与认知方式。

**💡 创新点**

提出并验证交互结构（如模型定位为专家、提示架构化、可视化推理与多模态协同）对医生使用LLM的主动性、批判性和决策参与度的影响，揭示单一聊天界面无法满足临床多任务、注意力分配需求。

**🔧 技术方法**

采用大语言模型（如GPT‑4）作为文本生成器，结合自定义UI原型（表格、卡片、路径图）以及语音录音/识别组件，配合思考实验、半结构访谈和AI辅助编码进行数据分析。

**📊 数据集**

使用从真实电子病历抽取、去标识化的两份临床病历（肺移植、肾移植）以及包含多任务的临床情景，样本规模12名医生，覆盖内科、儿科、放射肿瘤等专业。

**📈 对比分析**

通过对话记录的结构编码与主题分析比较不同模式的使用频率、信息提取方式、响应满意度等指标，结果显示：聊天模式主要用于检索事实；视觉模式更易触发批判性评估；语音模式被认为不适合决策支持，整体表现取决于交互结构而非模型本身；未给出数值性能指标，仅以定性行为模式呈现。

**⚠️ 局限性**

局限性包括：样本量小且为实验室环境；UI反馈仅为走查而非实时交互；语音交互未实际使用，仅为访谈中概念性讨论；结果难以直接推广至日常临床工作，需进一步实地评估。

---

## 90. Privacy-Preserving Sensor-Based Human Activity Recognition for Low-Resource Healthcare Using Classical Machine Learning

**arXiv ID:** 2601.22265 | [PDF](https://arxiv.org/pdf/2601.22265v1)

**作者:** Ramakant Kumar `[一作]` (GLA University), Pravin Kumar `[通讯]` (GLA University)

**通讯引用:** 1193 | [OpenAlex ID](https://openalex.org/A5101818302)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研发了一种基于可穿戴IMU传感器的低成本隐私保护人类活动识别系统，并结合支持张量机与联邦学习实现实时分类。

**💡 创新点**

创新点在于引入支持张量机保持时空结构并采用联邦学习实现边缘隐私训练，显著提升精度至98.5%。

**🔧 技术方法**

使用了传感器数据预处理（移动平均、卡尔曼滤波）、经典机器学习（Logistic Regression、Random Forest、SVM、k‑NN）与支持张量机，以及联邦学习FedAvg。

**📊 数据集**

使用了15名志愿者自采IMU加速度/陀螺仪数据（6种活动）以及UCI HAR公开数据集进行验证。

**📈 对比分析**

与LR、RF、SVM、k‑NN在中央化训练下比较，STM在测试集上达96.7%准确率，5折交叉验证98.5%，联邦模型最终准确率98.69%。

**⚠️ 局限性**

局限在于样本量有限、仅使用加速度数据，且对不同姿态、跨人群鲁棒性待进一步验证。

---

## 91. Toward Non-Expert Customized Congestion Control

**arXiv ID:** 2601.22461 | [PDF](https://arxiv.org/pdf/2601.22461v1)

**作者:** Mingrui Zhang `[一作]` (University of Nebraska Lincoln), Lisong Xu `[通讯]` (University of Nebraska Lincoln)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种非专家定制拥塞控制框架NECC，允许普通用户通过LLM交互收集需求、改写现有Linux CCA代码并以BPF方式部署。

**💡 创新点**

创新点在于将定制需求视为代码修订目标，使用LLM进行代码修订而非全新生成，并结合链式思维提示和网络特定反馈迭代改进。

**🔧 技术方法**

主要技术包括大型语言模型（GPT‑4o等）、BPF接口、链式思维提示、网络性能仿真Mininet以及代码编译与BPF检查。

**📊 数据集**

实验使用Linux内核中四个现有CCA（Reno、Cubic、Vegas、Illinois）以及模拟的2K 60fps流媒体场景和多速率家庭网络。

**📈 对比分析**

通过对比0-shot与CoT提示、不同温度和池大小，NECC生成的定制Cubic在拥塞网络下SSIM显著高于原Cubic，且满意度最高分可达100%。

**⚠️ 局限性**

局限包括仅考虑吞吐量安全约束、仅支持四个CBA、对网络信息完整性与模型准确性依赖较高，且未覆盖BBR、QUIC等新协议。

---

## 92. Design Perspective on Materials Experience: A CiteSpace-Based Bibliometric and Visual Analysis of Interdisciplinary Research

**arXiv ID:** 2601.22518 | [PDF](https://arxiv.org/pdf/2601.22518v1)

**作者:** Yuxin Zhang `[一作]` (Tsinghua University), Fan Zhang `[通讯]` (Tsinghua University)

**通讯引用:** 53036 | [OpenAlex ID](https://openalex.org/A5100403400)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

**🎯 论文内容**

对2005–2024年材料体验领域的文献进行系统的文献计量学分析，利用CiteSpace构建共引、共词与合作网络，并对研究热点、发展脉络和学术影响进行量化评估。

**💡 创新点**

首次从宏观层面绘制材料体验知识结构图谱，揭示其跨学科融合趋势，并提出“设计为媒介”视角，将设计从技术手段转变为跨学科知识整合的核心。

**🔧 技术方法**

主要使用CiteSpace（6.4.R1）进行可视化绘图；Python（3.12.7）完成统计与图表；采用共词聚类（Louvain）、Burst分析、度中心度、betweenness等计量指标。

**📊 数据集**

数据来源于Web of Science Core Collection，检索词涵盖“Material Experience”“Kansei Engineering”等，初始2,831条记录经两轮人工筛选后得到575篇论文（期刊+会议）作为分析样本。

**📈 对比分析**

通过年度发表量回归、作者/机构/国家中心度对比，以及聚类模块度Q=0.83、平均Silhouette=0.93等质量指标评估模型；结果显示研究热度持续上升，学术产出集中于美国、中国、德国等国家，设计与技术交叉点表现最为突出。

**⚠️ 局限性**

局限性包括：仅检索英文WOS文献，可能遗漏非英文或非WOS收录的工作；手工筛选带来的主观性；聚类结果依赖算法参数；未对研究质量进行深度评估，缺乏对实验方法与理论创新的细致阐释。

---

## 93. Capacity of Two-User Wireless Systems Aided by Movable Signals

**arXiv ID:** 2601.22358 | [PDF](https://arxiv.org/pdf/2601.22358v1)

**作者:** Matteo Nerini `[一作]` (Imperial College London), Bruno Clerckx `[通讯]` (Imperial College London)

**通讯引用:** 15728 | [OpenAlex ID](https://openalex.org/A5070530952)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了在两用户无线系统中通过可移动信号动态重构频率来实现信道正交化，从而提升多接入（MAC）和广播（BC）的容量区域。

**💡 创新点**

创新点在于利用频域可调频率来正交化用户信道，达到比传统固定频率或RIS/可调天线更高的容量提升，并给出了频率优化策略。

**🔧 技术方法**

主要技术包括频率可调的可移动信号设计、解析求解信道正交化的最优频率、匹配滤波/匹配波束成形以及有限频率范围内的频率优化。

**📊 数据集**

论文使用理论分析与仿真验证，假设用户距离10 m、角度均匀分布，基站采用Ula天线阵列，频率范围为[f_min,f_max]（如f_min=f_A, f_max=1.8f_A）。

**📈 对比分析**

与固定频率系统和理论上限比较，受限频率范围内的可移动信号在10 dB SNR、N=2时可提升约31%（相较于固定信号），而理论上限可提升45%，性能随天线数增多或SNR升高而变化。

**⚠️ 局限性**

局限性包括只考虑两用户单天线、LOS环境、无限频率可调的理论假设、以及未考虑多用户、多天线接收端或NLOS环境等更复杂场景。

---

## 94. Controllable Information Production

**arXiv ID:** 2601.22449 | [PDF](https://arxiv.org/pdf/2601.22449v1)

**作者:** Tristan Shah `[一作]` (Texas Tech University), Stas Tiomkin `[通讯]` (Texas Tech University)

**通讯引用:** 184 | [OpenAlex ID](https://openalex.org/A5046771851)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出并推导了Controllable Information Production（CIP）这一全新的内在动机框架，并在经典的单摆、倒立摆和双摆等基准环境中通过MPC‑CIP算法验证了其有效性。

**💡 创新点**

创新点在于直接从最优控制理论导出以Kolmogorov–Sinai熵差为核心的CIP指标，避免了设计者需要指定信息传输的随机变量，提供了对可控混沌量化的全新视角。

**🔧 技术方法**

主要技术包括最优控制与离散代数Riccati方程、Kolmogorov–Sinai熵（开环与闭环）的计算、以及基于MPC和改进交叉熵方法（iCEM）的采样优化。

**📊 数据集**

实验使用MuJoCo仿真环境中的三种经典控制任务：单摆（Single Pendulum）、倒立摆（Cart‑Pole）和双摆（Double Pendulum）。

**📈 对比分析**

虽然论文未给出与其他IM方法（如Empowerment、Curiosity等）的直接数值对比，但通过CIP目标的MPC控制，能够在所有三种任务中实现摆动上升与稳定，并产生更高的CIP值，说明在探索‑利用平衡上表现更优。

**⚠️ 局限性**

主要局限包括开环熵估计的数值不稳定、基于随机射击MPC的可扩展性差、未在高维或长时序任务中验证，以及未与外部奖励任务相结合。

---

## 95. Stalled, Biased, and Confused: Uncovering Reasoning Failures in LLMs for Cloud-Based Root Cause Analysis

**arXiv ID:** 2601.22208 | [PDF](https://arxiv.org/pdf/2601.22208v1)

**作者:** Evelien Riddell `[一作]` (University of Waterloo), Krzysztof Czarnecki `[通讯]` (University of Waterloo)

**通讯引用:** 18079 | [OpenAlex ID](https://openalex.org/A5066916130)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究提出并实现了一种面向推理的评估框架，用来单独检验大语言模型（LLM）在云端根因分析（RCA）任务中的推理能力，并对推理过程中的错误进行系统分类。

**💡 创新点**

创新点在于：①将LLM的推理过程从复杂多代理流水线中剥离出来，构造了极简化的RCA实验环境；②设计了“LLM‑as‑a‑Judge”自动评判器，结合人类标注构建了16类推理失败的分类体系；③通过大规模模拟（48,000 场景）量化LLM在不同工作流和数据模态下的准确性与推理质量。

**🔧 技术方法**

使用的技术包括：多模态告警提取（日志、指标、追踪），图知识库（KG）构建，LangGraph实现的三种工作流（即时交互式、计划先行、无代理直接推断），多种开源LLM（Llama‑3.2‑3B、Qwen‑3‑4B/32B、Llama‑3.3‑70B、Command‑R+‑104B、DeepSeek‑R1‑70B），以及基于GPT‑5的LLM‑as‑a‑Judge自动评判。

**📊 数据集**

使用的数据集为：①GAIA（MicroSS 微服务）与其多模态监控数据；②OpenRCA（Online Boutique）公开微服务系统的日志、指标和追踪；两者均提供真实的故障案例和系统依赖图。

**📈 对比分析**

通过与随机猜测基线、非代理直推法和两种代理工作流进行对比，发现即使是大模型在根因定位和路径验证上也仅能达到约 30–40% 的 top‑3 正确率；代理工作流对小模型往往适得其反，而对大模型提升路径有效率；整体推理错误普遍存在，说明当前LLM在RCA中的实际效能有限。

**⚠️ 局限性**

局限性包括：①实验仅考虑单一根因场景，未覆盖多故障交互；②告警提取可能带噪声，未进行精细筛选；③仅评估了少数公开数据集和模型，结果可能不具备广泛外部有效性；④分类体系和LLM‑as‑a‑Judge的可靠性受主观编码与模型偏好影响；⑤提示词与工具定义的细微变化可能显著影响结果。

---

## 96. Mock Worlds, Real Skills: Building Small Agentic Language Models with Synthetic Tasks, Simulated Environments, and Rubric-Based Rewards

**arXiv ID:** 2601.22511 | [PDF](https://arxiv.org/pdf/2601.22511v1)

**作者:** Yuan-Jay Lü `[一作]` (University of Science and Technology of China), Tong Xu `[通讯]` (University of Science and Technology of China)

**通讯引用:** 4089 | [OpenAlex ID](https://openalex.org/A5025292786)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出 SynthAgent 框架，通过生成多样化的工具使用任务并搭建稳定的模拟环境，使小模型在代理式强化学习中获得显著提升。

**💡 创新点**

创新点在于：①使用教师 LLM 生成任务与专属工具生态并故意制造信息缺口；②构建基于工作流的评价 Rubric；③利用轻量级 LLM 模拟用户与工具，保证训练环境稳定。

**🔧 技术方法**

采用 LLM 生成任务与工具、任务级工作流抽取、规则化奖励设计、GRPO 强化学习算法，并通过 Qwen 系列模型进行模拟与评估。

**📊 数据集**

主要数据集包括 15,096 个自合成的工具使用任务、14 个挑战性基准（如 TAU‑2、BFCL‑V4、AIME、HMMT 等）以及 4,000 条从 ToolStar 采样的推理任务。

**📈 对比分析**

与基于真实工具或公开数据的基线（ToolStar‑8B/14B、Qwen3‑32B 等）进行对比，SynthAgent‑8B 在 TAU‑2、BFCL‑Multi‑turn 的平均得分提升 12.3 分，SynthAgent‑14B 与 32B 同水平甚至更优，并在推理任务上同样获得显著性能提升。

**⚠️ 局限性**

限制在于：合成任务与奖励规则的质量依赖教师 LLM，且缺乏与其他大型厂商模型的公开对比；模拟环境虽稳定但仍可能缺乏真实世界的复杂性。

---

## 97. Symmetry Breaking in Transformers for Efficient and Interpretable Training

**arXiv ID:** 2601.22257 | [PDF](https://arxiv.org/pdf/2601.22257v1)

**作者:** Eva Silverstein `[一作]` (Stanford University), Vasudev Shyam `[通讯]` (Zyphra Technologies)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过在Transformer注意力中加入批量采样的非学习查询与值偏置，打破旋转对称性，提升优化效率并赋予模型可解释性。

**💡 创新点**

创新点在于用最小的架构改动——引入固定无学习的查询/值偏置，既能消除对称性导致的能量守恒问题，又能让模型利用该偏置放大或抑制特定token的注意力。

**🔧 技术方法**

采用的技术包括基于Hamiltonian动力学的能量守恒下降（ECD）以及传统的SGDM、AdamW、SOAP优化器，并在注意力层实现无学习偏置的批量采样。

**📊 数据集**

使用FineWeb‑Edu数据集对GPT‑2 124M模型进行500M个token的预训练，随后在逻辑推理任务上评估。

**📈 对比分析**

对比实验显示，在对称性破坏下ECD的验证损失降至与SOAP相近（约3.3-3.4），且在大多数种子下逻辑推理Top‑5准确率提升或保持不变；相比之下，未破坏对称性的ECD表现显著差。

**⚠️ 局限性**

主要局限是实验规模与统计量有限，模型参数、数据量与算力均相对较小，未检验在更大规模或不同任务下的可扩展性与稳健性。

---

## 98. Nethira: A Heterogeneity-aware Hierarchical Pre-trained Model for Network Traffic Classification

**arXiv ID:** 2601.22494 | [PDF](https://arxiv.org/pdf/2601.22494v1)

**作者:** Chungang Lin `[一作]` (Institute of Computing Technology), Yujun Zhang `[通讯]` (Institute of Computing Technology)

**通讯引用:** 4835 | [OpenAlex ID](https://openalex.org/A5100751747)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

Nethira提出了一种异质感知的层级预训练模型，通过字节、协议和数据包三层的重构与层级增强实现网络流量分类。

**💡 创新点**

创新点在于同时采用多层级重构预训练和一致性正则化的层级增强细调，有效捕捉流量的层级异质性并显著降低对标签的依赖。

**🔧 技术方法**

使用Transformer编码器‑解码器架构、字节嵌入、协议字段掩码、数据包顺序扰动、KL一致性正则化等技术实现模型训练。

**📊 数据集**

在ISCX‑VPN（App/Service）、USTC‑TFC、CIC‑IoT三大公开数据集上进行实验验证。

**📈 对比分析**

与十二个基线（统计特征、深度学习与七种预训练模型）对比，Nethira平均F1提升9.11%，在仅1%标签样本下即可匹配甚至超过全标签训练模型。

**⚠️ 局限性**

局限性包括在低异质性数据集上的提升有限，以及对大规模预训练仍需消耗较高计算资源。

---

## 99. DreamVAR: Taming Reinforced Visual Autoregressive Model for High-Fidelity Subject-Driven Image Generation

**arXiv ID:** 2601.22507 | [PDF](https://arxiv.org/pdf/2601.22507v1)

**作者:** Xin Jiang `[一作]` (Nanjing University of Science and Technology), Tao Mei `[通讯]` (HiDream.ai Inc.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了DreamVAR框架，用视觉自回归模型实现以参考主体为条件的图像生成，解决传统交错条件方式的训练-推理不匹配问题。

**💡 创新点**

创新点在于先将多尺度参考主体特征预填入模型，消除训练时的教师强制导致的分布偏差，并结合GRPO强化学习，将主体一致性与文本对齐双重奖励联合优化。

**🔧 技术方法**

技术包括多尺度视觉分词器、VAR next‑scale 预测、GRPO强化学习、DINO/CLIP特征嵌入以及基于Infinity‑2B的预训练文本图像模型。

**📊 数据集**

使用Subject‑200K和自构建的DreamSubject‑14K数据集进行多阶段训练，并在Dreambench数据集上进行评估。

**📈 对比分析**

与Diffusion基线（OminiControl、UNO等）相比，DreamVAR在512×512分辨率下实现了0.764的DINO、0.838的CLIP‑I，CLIP‑T略优于基线，同时参数量仅2B，推理速度提升约8×。

**⚠️ 局限性**

局限性包括对奖励设计敏感（可能出现奖励作弊），依赖高质量主体分割与标注，评估范围主要聚焦于主体保持与文本对齐，未在更广泛的控制任务或高分辨率设置中验证。

---

## 100. Enhancing TableQA through Verifiable Reasoning Trace Reward

**arXiv ID:** 2601.22530 | [PDF](https://arxiv.org/pdf/2601.22530v1)

**作者:** Tung Sum Thomas Kwok `[一作]` (University of California, Los Angeles), Guang Cheng `[通讯]` (University of California, Los Angeles)

**通讯引用:** 2555 | [OpenAlex ID](https://openalex.org/A5043707940)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种无训练、可即插即用的奖励增强表格问答框架，在每一步表格状态转换和模拟推理阶段为LLM提供可验证的奖励信号，从而提升推理准确性与效率。

**💡 创新点**

创新点在于：①设计基于最长公共子序列（LCS）的表格对查询相关度指标，实现对中间表格状态的可验证评估；②引入两阶段奖励搜索（状态转移奖励 + 模拟推理奖励），显著降低推理成本；③该框架无需额外训练，能直接插入现有工具操作的表格问答系统。

**🔧 技术方法**

技术手段包括：POMDP 形式建模；LLM+工具操作的表格推理；可验证的 LCS‑based 状态奖励；两阶段搜索与奖励聚合；对动作级与状态级奖励的对比实验；在多种 LLM（GPT‑3.5、GPT‑4.1‑nano、QWEN‑3‑8B 等）上实施。

**📊 数据集**

使用了三大公开表格问答基准：WikiTQ、MMQA 和 MMTU。

**📈 对比分析**

在 Chain-of-Tables 与 Tree-of-Tables 基线上加入奖励后，所有 LLM 的平均准确率提升约 15–20%，最优提升至 41.77%；同时推理 token 数减少 33.33%；与传统 VLM 或基于嵌入的相似度奖励相比，该方法更稳定、更高效。

**⚠️ 局限性**

局限性：①奖励仅在单表格层面，缺乏跨表统一校准；②对极端 OOD 迁移的鲁棒性尚未充分验证；③奖励易受停用词或格式噪声影响，需要进一步去噪与关键词提取；④目前无法直接比较不同表格之间的全局奖励。

---

## 101. SurrogateSHAP: Training-Free Contributor Attribution for Text-to-Image (T2I) Models

**arXiv ID:** 2601.22276 | [PDF](https://arxiv.org/pdf/2601.22276v1)

**作者:** Mingyu Lu `[一作]` (Paul G. Allen School of Computer Science and Engineering University of Washington), Su-In Lee `[通讯]` (Paul G. Allen School of Computer Science and Engineering University of Washington)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种无需重新训练的贡献者归因框架SurrogateSHAP，用于评估文本到图像扩散模型中的数据贡献者对模型行为的影响。

**💡 创新点**

创新点在于：①将昂贵的“重训练”游戏重构为训练自由的代理游戏，只需利用预训练模型的条件生成；②用梯度提升树拟合代理游戏并通过TreeSHAP高效解析Shapley值，显著提升样本效率和计算速度。

**🔧 技术方法**

核心技术包括：训练自由的代理游戏（条件生成+混合分布）、梯度提升树(GBDT)拟合代理函数、TreeSHAP解析Shapley值，以及理论稳定性分析。

**📊 数据集**

实验使用三大数据集：CIFAR‑20（DDPM‑CFG）、ArtBench后印象派（Stable Diffusion + LoRA）、Fashion‑Product（FLUX‑1 + LoRA）。

**📈 对比分析**

与基线（LOO、TRAK、IF、sFT、KernelSHAP等）对比，SurrogateSHAP在FID、Aesthetic、LPIPS等多种评价指标上均获得最高的LDS和最优的样本效率，计算速度提升达12–23倍。

**⚠️ 局限性**

局限性包括：代理游戏仍假设条件分布不随子集显著变化，某些高度非线性或交互强的任务中精度可能受限；对多目标或多模态评估的适用性尚待验证。

---

## 102. The Third-Party Access Effect: An Overlooked Challenge in Secondary Use of Educational Real-World Data

**arXiv ID:** 2601.22472 | [PDF](https://arxiv.org/pdf/2601.22472v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 103. VMonarch: Efficient Video Diffusion Transformers with Structured Attention

**arXiv ID:** 2601.22275 | [PDF](https://arxiv.org/pdf/2601.22275v1)

**作者:** Cheng Liang `[一作]` (Kuaishou Technology), Limin Wang `[通讯]` (Nanjing University)

**通讯引用:** 21717 | [OpenAlex ID](https://openalex.org/A5100436505)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了 Video Monarch Attention（VMonarch），一种利用结构化 Monarch 矩阵实现的子二次复杂度注意力机制，以提升 Video Diffusion Transformers（DiTs）的长序列计算效率。

**💡 创新点**

创新点在于：① 将 Monarch 矩阵与视频的时空结构对齐，设计时空分块形式；② 引入首帧重算策略缓解注意力泄漏导致的首帧模糊；③ 开发在线熵 FlashAttention 内核，加速 Monarch 因子更新；④ 通过双迭代交替最优化实现 𝒪(N√N) 复杂度。

**🔧 技术方法**

核心技术包括 Monarch 矩阵结构化表示、交替最小化优化、熵正则化、FlashAttention 结合在线熵计算、首帧重算和 GPU 自定义核实现。

**📊 数据集**

在 Wan2.1‑1.3B、Wan2.1‑14B、Wan2.2‑5B 这三套基础 DiT 模型上，使用 Wan14B‑Syn‑600k、Wan2.2‑Syn‑32k 语料库进行微调，并在 VBench 评测集上验证。

**📈 对比分析**

与全注意力（FA2）以及现有稀疏注意力方法（VSA、VMoBA）对比，VMonarch 在大约 90% 稀疏率下，Attention FLOPs 减少 17.5 倍，推理速度提升 5×，同时在 VBench 的 5 项质量指标上保持与全注意力相当甚至优于其余稀疏方案。

**⚠️ 局限性**

限制包括：① 仍需微调才能充分发挥稀疏注意力效果；② 对首帧的重算引入额外开销；③ 在极长序列或极大空间尺寸下，Monarch 因子更新仍可能成为瓶颈；④ 对不同视频内容的通用性需进一步验证。

---

## 104. MrRoPE: Mixed-radix Rotary Position Embedding

**arXiv ID:** 2601.22181 | [PDF](https://arxiv.org/pdf/2601.22181v1)

**作者:** Qingyuan Tian `[一作]` (Shanghai Jiao Tong University), Rui Wang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 24210 | [OpenAlex ID](https://openalex.org/A5028153158)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于混合基数（Radix）转换的RoPE扩展框架MrRoPE，并在此框架下设计了两种训练‑free 方法：MrRoPE-Uni（均匀转换）和 MrRoPE-Pro（逐层递增转换），从而实现 LLM 的长文本上下文窗口扩展。

**💡 创新点**

创新点在于：① 用基数转换视角统一了现有 RoPE 扩展方法（PI、NTK、YaRN 等），② 发现并提出“逐层递增”基数扩展策略（MrRoPE‑Pro），该策略在保留高频信息的同时显著提升了长距位置编码的可用范围。

**🔧 技术方法**

核心技术包括：RoPE 的基数转换数学表述、λ 向量的设计（均匀或递增）、无训练的角度缩放与插值、以及理论上对 RoPE 边界（Upper‑bound）与注意力分布的分析。

**📊 数据集**

使用了 LLaMA2‑7B、LLaMA3‑8B、Qwen2.5‑3B 三大 RoPE‑LLM 作为基线；在 Proofpile、RULER、Needle‑in‑a‑Haystack、Infinite‑Bench 与 LongBench‑V2 等公开长文本评测集上进行实验。

**📈 对比分析**

与 YaRN、NTK 等最先进的无训练扩展方法相比，MrRoPE‑Pro 在 8K–128K 的上下文长度上均实现了更低的困惑度、显著更高的检索召回率（如 Needle‑in‑a‑Haystack 维持 85%+ 的 recall 至 120K），在 Infinite‑Bench 与 LongBench‑V2 上的任务平均得分超过 YaRN 近 10–15% 并逼近 GPT‑4 的表现。

**⚠️ 局限性**

局限性包括：① 仍未在超过 200K 的极长序列上进行系统验证；② 依赖于预训练 RoPE 的频率分布，若基础模型的频率设定极差，递增策略可能不适用；③ 由于是训练‑free 方法，无法通过微调进一步提升在特定领域任务的性能。

---

## 105. EMBC Special Issue: Calibrated Uncertainty for Trustworthy Clinical Gait Analysis Using Probabilistic Multiview Markerless Motion Capture

**arXiv ID:** 2601.22412 | [PDF](https://arxiv.org/pdf/2601.22412v1)

**作者:** Seth Donahue `[一作]` (Shriners Children's Lexington), R. James Cotton `[通讯]` (Shirley Ryan AbilityLab)

**通讯引用:** 3387 | [OpenAlex ID](https://openalex.org/A5049159760)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

对一种基于概率的多视角无标记运动捕捉（MMMC）方法进行外部验证，评估其在临床人群中的校准性和可靠性，并提供每个关节、每个时间点的置信区间。

**💡 创新点**

创新点在于：1）将变分推断嵌入端到端可微分的MMMC管道，得到可校准的联合角度后验分布；2）利用ECE和PIT量化置信区间的校准；3）在多种临床人群（正常、神经、义肢、儿童）和多摄像头配置下对模型进行全面外部验证，证明其可在无外部仪器的情况下识别不可靠推断。

**🔧 技术方法**

核心技术包括：MuJoCo+MyoSuite的物理模拟；MetrABs-ACAE关键点检测；可微分的几何一致性约束；变分推断（高斯后验，低秩协方差）；ELBO+ECE正则化的联合损失；PIT与ECE评估；基于标记数据的偏差校正。

**📊 数据集**

数据集：68名受试者（41名来自Shirley Ryan，27名来自Shriners Children）共416次试验；采集使用8-12台FLIR BlackFly相机（30-60Hz）同步视频；外部验证使用GaitRite步道（步长/跨步长）和12摄像头Vicon Vantage（标记式三维运动捕捉）获取真实关节角度。

**📈 对比分析**

与GaitRite的步长/跨步长相比，模型的中位误差分别为≈16 mm和12 mm，ECE≤0.1；与标记式捕捉相比，关节角误差均≤4°（大部分≤3°），ECE<0.1；预测不确定性与实际误差呈高度相关，低不确定性区间内误差显著降低；通过筛除前50%不确定性步骤，步长误差可降至≈12 mm。

**⚠️ 局限性**

局限性：1）所有实验均在实验室环境下进行，未覆盖真实临床现场的多障碍、光照变化和摄像头遮挡情况；2）需要针对每个受试者进行系统级偏差校正，影响实时部署；3）部分关节（如髋旋转、髋关节）校准误差较大，需改进物理模型和Euler角顺序；4）模型对遮挡、助行器等因素的鲁棒性仍需进一步定量评估。

---

## 106. Knowledge Gradient for Preference Learning

**arXiv ID:** 2601.22335 | [PDF](https://arxiv.org/pdf/2601.22335v1)

**作者:** Kaiwen Wu `[一作]` (University of Pennsylvania), Jacob R. Gardner `[通讯]` (University of Pennsylvania)

**通讯引用:** 1957 | [OpenAlex ID](https://openalex.org/A5072585411)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种在仅能进行成对比较查询的偏好学习框架中，能够精确评估下一轮查询价值的知识梯度方法。

**💡 创新点**

创新点在于发现一阶前瞻后验服从扩展偏态正态分布，利用其闭式矩公式得到知识梯度的解析表达，从而摆脱了传统需要采样或近似的计算瓶颈。

**🔧 技术方法**

所使用技术包括高斯过程代理模型、Probit 偏好似然、扩展偏态正态分布理论以及一阶前瞻后验均值的闭式推导，进而实现一体化的 one‑shot 知识梯度优化。

**📊 数据集**

实验采用了六个经典基准函数（如 Ackley、Levy、Rastrigin 等），维度从 2 到 7，使用变分高斯过程作为潜在效用函数的代理模型。

**📈 对比分析**

与随机抽样、对数期望改进（LogEI）以及期望最佳使用（EUBO）等基线相比，知识梯度在低噪声和高噪声情形下均实现了更低的最优性缺口，整体性能优于或接近其它方法。

**⚠️ 局限性**

局限性包括在噪声极低或已知最优区间明显时，固定噪声参数导致知识梯度过于保守；此外在大规模高维问题上，其计算与优化效率仍需进一步研究。

---

## 107. PoSafeNet: Safe Learning with Poset-Structured Neural Nets

**arXiv ID:** 2601.22356 | [PDF](https://arxiv.org/pdf/2601.22356v1)

**作者:** Kiwan Wong `[一作]` (MIT CSAIL), Daniela Rus `[通讯]` (MIT CSAIL)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种名为PoSafeNet的安全学习框架，通过显式建模安全约束的偏序（poset）来实现多约束下的安全控制。

**💡 创新点**

创新点在于将安全约束的优先级关系编码为偏序，并通过多头神经网络在每个线性扩展（poset的一致排序）上进行闭式投影，既保证了安全优先级，又避免了传统二次规划（QP）在多约束下的不可行性和计算开销。

**🔧 技术方法**

技术手段包括：
- 控制势函数（CBF）产生线性不等式约束；
- 对单一约束的闭式投影；
- 多头网络生成不同poset一致顺序的安全执行；
- 通过Gumbel‑softmax或凸混合实现头部的选择或组合；
- 端到端的模仿学习与梯度可微的投影层结合。

**📊 数据集**

使用的数据集包括：
- 2D 圆形障碍物的平面小车导航（随机起点-终点）
- 两连杆平面机械臂的关节限制与障碍规避任务
- VISTA 仿真环境下的视觉端到端自动驾驶（包含碰撞与车道保持安全约束）。

**📈 对比分析**

与 ABNet、BarrierNet、DFBNet 等无结构或基于QP的安全层进行比较：PoSafeNet 在所有任务中实现 100% 的可行性和安全性（正的安全边际），同时在 MSE、最终距离和计算时间上均优于对比方法；在视觉驾驶任务中实现 100% 的通过率且无碰撞，且在车道偏离和控制方差上优于基线。

**⚠️ 局限性**

局限性包括：需要先验或手工定义安全约束的偏序关系；仅适用于控制仿射系统和基于CBF的安全证书，无法直接处理随机或多智能体情境；在不满足几何投影不冲突条件时，凸混合的安全性只能视为经验性。

---

## 108. DP-$λ$CGD: Efficient Noise Correlation for Differentially Private Model Training

**arXiv ID:** 2601.22334 | [PDF](https://arxiv.org/pdf/2601.22334v1)

**作者:** Nikita P. Kalinin `[一作]` (Institute of Science and Technology Austria), Christoph H. Lampert `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种无需额外内存、仅利用伪随机数生成器重构噪声的噪声相关技术，可在差分隐私梯度下降（DP‑SGD）中引入一阶相关噪声；

**💡 创新点**

创新点在于通过噪声重生而非存储历史噪声向量，构造仅与前一次迭代相关的Toeplitz策略矩阵C_λ，实现了低内存且近乎零额外开销的相关噪声；

**🔧 技术方法**

核心技术包括：低阶Toeplitz逆矩阵噪声相关、伪随机数生成器状态保存与回滚、Balls‑in‑Bins采样放大、对RMSE与MaxSE误差的理论分析及λ参数调优；

**📊 数据集**

实验数据集主要为CIFAR‑10（CNN）和IMDB（LSTM、BERT‑tiny）等；

**📈 对比分析**

与DP‑SGD、Poisson采样、BandInvMF、BISR、BLT等基线比较，结果显示该方法在相同隐私预算下准确率提升、误差下降，且与DP‑SGD的运行时相差不到1%（CIFAR‑10）或2%（IMDB），大幅优于需要CPU存储或完整重生噪声的方案；

**⚠️ 局限性**

局限包括：需要手动选择λ参数，最优值与任务、批量、隐私级别相关；仅实现了与前一次迭代相关，未扩展到更宽带的相关性；伪随机数生成器非加密安全；对极端大规模模型或多GPU环境的评估仍有限。

---

## 109. Federate the Router: Learning Language Model Routers with Sparse and Decentralized Evaluations

**arXiv ID:** 2601.22318 | [PDF](https://arxiv.org/pdf/2601.22318v1)

**作者:** Baris Askin `[一作]` (Carnegie Mellon University), Carlee Joe-Wong `[通讯]` (Carnegie Mellon University)

**通讯引用:** 17165 | [OpenAlex ID](https://openalex.org/A5003037377)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究如何在多租户、隐私敏感且评估数据稀疏的环境下，利用联邦学习训练能在多大模型池中根据查询选择最佳模型的路由器。

**💡 创新点**

①首次将联邦学习应用于LLM路由器；②同时支持参数化（MLP）和非参数化（K‑means）两类路由器；③通过跨客户端聚合显著提升了模型覆盖率和查询泛化能力；④提出针对高异质性客户的自适应个性化混合策略。

**🔧 技术方法**

联邦平均（FedAvg）框架；MLP trunk + 模型特定线性头；局部+全局K‑means聚类；多任务估计（准确率与成本）并通过加权平均得到全局估计；自适应加权混合方案；使用预训练句子编码器（mpnet-base-v2）生成查询嵌入。

**📊 数据集**

①包含11个公开+专有LLM、8个公开数据集（Benchmark A）；②包含14个公开LLM、10个公开数据集（Benchmark B）。

**📈 对比分析**

在联邦模拟环境（10个客户端，60%参与率，Dirichlet 0.6/0.03）下，与仅使用本地数据训练的路由器相比，联邦路由器在全局测试集和本地测试集上的准确率‑成本曲线均显著上升，AUC提升约3–10%；在引入新模型时，只需轻量级校准即可快速更新。相比中心化训练，性能相当，且在高异质性场景下，采用自适应个性化可进一步提高。

**⚠️ 局限性**

①需要每个客户端至少有少量查询‑模型评估样本；②目前为离线评估，未覆盖在线实时更新；③对极端稀疏数据或极端模型不平衡时仍可能受限；④模型池变动速度过快时，K‑means聚类需要重新计算；⑤安全性只保证数据不被共享，但模型参数仍可被推断，需进一步隐私增强。

---

## 110. Do Open-Vocabulary Detectors Transfer to Aerial Imagery? A Comparative Evaluation

**arXiv ID:** 2601.22164 | [PDF](https://arxiv.org/pdf/2601.22164v1)

**作者:** Christos Tsourveloudis `[一作]` (National Technical University of Athens), Christos Tsourveloudis `[通讯]` (National Technical University of Athens)

**通讯引用:** 1 | [OpenAlex ID](https://openalex.org/A5117714706)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

系统评估了五种主流开源开源式可变词汇目标检测模型在空中图像基准LAE-80C上的零样本性能，并通过Global、Oracle和Single-Category三种推理模式分离语义混淆与视觉定位失效。

**💡 创新点**

提出了严格零样本空中评估流程、分离语义与定位的评估方法、以及对不同提示策略与推理模式的系统性对比，首次揭示语义混淆是空中OVD的主要瓶颈。

**🔧 技术方法**

采用Grounding DINO、OWLv2、YOLO-World、YOLOE、LLMDet等模型，结合域特定前缀、同义词扩展等Prompt工程，并用IoA阈值0.7评估定位质量。

**📊 数据集**

使用LAE-80C基准（来自DOTA、DIOR、FAIR1M、xView共3,592张图、86,558个实例、80类），并对DIOR、DOTA、FAIR1M、xView子集进行逐步性能演化分析。

**📈 对比分析**

通过Precision、Recall、F1以及TP/FP/FN计数进行比较，最佳模型OWLv2在LAE-80C上达27.6% F1（Recall≈24.7%），但误报率高达69%；整体表现远低于自然图像基准，显示出显著的域迁移失败。

**⚠️ 局限性**

主要局限包括：视觉编码因训练于地面图像而与空中视角不匹配；语义混淆是核心瓶颈；Prompt工程在无细调的情况下无效；模型在不同子数据集间表现不稳定，缺乏平衡的精度召回。

---

## 111. What Lies Beneath: A Call for Distribution-based Visual Question & Answer Datasets

**arXiv ID:** 2601.22218 | [PDF](https://arxiv.org/pdf/2601.22218v1)

**作者:** Jill P. Naiman `[一作]`, JooYoung Seo `[通讯]` (University of Illinois)

**通讯引用:** 577 | [OpenAlex ID](https://openalex.org/A5101467241)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究构建并公开了一个基于分布的科学图表 VQA 数据集，专注于直方图，并对生成的图表提出两类统计问题（中位数和高斯成分数），随后使用人类注释者与大型多模态模型（ChatGPT‑5‑nano）进行答案评估。

**💡 创新点**

创新点在于：①提出“分布式 VQA”范式，强调图表标记与原始数据不具 1 对 1 对应；②利用合成直方图与可复制的生成流程，填补当前 VQA 数据集中直方图和分布信息缺失的空白；③公开完整的图像、原始数据、生成参数、边界框与人机答案，提供可复现的评测基线。

**🔧 技术方法**

技术主要包括：Python 合成图表库（生成高斯混合直方图）、JSON 结构化问题与答案、ChatGPT‑5‑nano 作为大模型推理、Zooniverse 平台进行人类注释、统计检验（Kruskal‑Wallis、Levene、线性混合效应模型）对结果进行分析。

**📊 数据集**

使用的数据集是本研究自行生成的 80 张直方图（含 40 张固定高斯成分 2、变条宽的子集及 40 张固定条宽 50、变高斯成分的子集），每张图配备原始数据、生成参数、边界框以及人工与 LMM 产生的答案；此外还引用了已有的 Chart‑VQA 数据集作为背景对比。

**📈 对比分析**

评估方法：将人类注释者与 LMM 的答案分别与真实答案计算残差，采用非参数 Kruskal‑Wallis 检验检验组间差异，Levene 检验检验方差差异，并用线性混合效应模型检验高斯成分数对误差的影响。结果显示：人类与 LMM 的中位数估计差异无显著性；LMM 的误差与具统计背景的人类相当，且在高斯成分数较多时误差显著上升；LMM 在数值提问（1-5）约束下表现与人类相近，未受约束时错误率上升。

**⚠️ 局限性**

局限性包括：①仅针对直方图且参数范围有限（高斯数 1–5，条宽 10–60）；②只设计了两道统计问题，未覆盖更复杂的推理任务；③实验仅使用 ChatGPT‑5‑nano，未涉及更大或其他多模态模型，且在更大模型下出现幻觉；④数据集规模相对较小，缺乏真实科研图表多样性；⑤未对合成数据的外部可泛化性进行评估。

---

## 112. Rethinking LLM-as-a-Judge: Representation-as-a-Judge with Small Language Models via Semantic Capacity Asymmetry

**arXiv ID:** 2601.22588 | [PDF](https://arxiv.org/pdf/2601.22588v1)

**作者:** Zhuochun Li `[一作]` (Ping An Technology), Daqing He `[通讯]` (University of Pittsburgh)

**通讯引用:** 3985 | [OpenAlex ID](https://openalex.org/A5026188630)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出“Representation-as-a-Judge”范式，利用小型LLM内部表示通过INSPECTOR框架实现无参考评估。

**💡 创新点**

首次证明小模型隐藏状态蕴含高质量评估信号，提出语义容量不对称假设，并通过探测而非提示实现高效可解释评估。

**🔧 技术方法**

内部表示探测、池化+PCA特征提取、线性/轻量化分类器、层级特征融合、评估标签对齐。

**📊 数据集**

数学推理基准GSM8K、MATH、GPQA。

**📈 对比分析**

与提示式小模型、微调小模型及RoBERTa对比，零样本二分类F1可达80‑90%，多分类F1提升约20%，接近大型LLM评判。

**⚠️ 局限性**

多类别预测仍受限；需先用大型LLM生成评估标签；探测模型对标签噪声敏感，未覆盖更广泛任务。

---

## 113. DNA: Uncovering Universal Latent Forgery Knowledge

**arXiv ID:** 2601.22515 | [PDF](https://arxiv.org/pdf/2601.22515v1)

**作者:** Jingtong Dou `[一作]` (University of Sydney), Tat-Seng Chua `[通讯]` (National University of Singapore)

**通讯引用:** 60236 | [OpenAlex ID](https://openalex.org/A5089404640)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于预训练模型内部稀疏神经单元（FDU）的鉴伪框架DNA，通过粗到细的层级定位与三元融合评分，直接从冻结的模型中挖掘隐含的伪造检测知识。

**💡 创新点**

创新点在于把鉴伪能力视作预训练模型的潜在知识而非需后期大量微调；采用层级层次定位、注意力分布与梯度加权的三元评分以及曲率截断算法高效筛选最具鉴伪特征的稀疏神经元。

**🔧 技术方法**

使用视觉Transformer及CNN等预训练骨干，结合线性探测、注意力向量距离、三元融合得分、kneedle阈值截断等技术实现FDU提取与利用。

**📊 数据集**

在ForenSynths、GenImage以及新建的高保真度合成基准HIFI‑Gen上进行实验，覆盖多种GAN与Diffusion模型。

**📈 对比分析**

与现有SOTA方法（如MoLD、DRCT、DIRE等）相比，DNA在ACC/AP均达≈99%（ForenSynths）和≈96%（GenImage），在未见模型的HIFI‑Gen上平均准确率超过96%，表现出极强的少样本泛化和鲁棒性。

**⚠️ 局限性**

局限性包括对预训练数据分布的依赖、对极度不同的生成技术（如超分辨、后处理变形）可能不够鲁棒，以及在面对主动对抗性伪造时仍需进一步验证。

---

## 114. Learning to Recommend Multi-Agent Subgraphs from Calling Trees

**arXiv ID:** 2601.22209 | [PDF](https://arxiv.org/pdf/2601.22209v1)

**作者:** Xinyuan Song `[一作]` (Emory University), Liang Zhao `[通讯]` (Emory University)

**通讯引用:** 6661 | [OpenAlex ID](https://openalex.org/A5061568038)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究多智能体系统中的代理与代理团队推荐，提出基于历史调用树的两阶段受限推荐框架，先检索可行候选集合，再通过学习到的评分模型对候选进行重排序，从而在给定上下文与约束下选择最合适的代理或代理子图。

**💡 创新点**

创新点在于：①将推荐建模为受限决策问题，明确区分可行性构造与价值优化；②同时提出代理级与代理系统级两种推荐实例；③利用统一调用树表示和结构化监督，提出兼顾语义相关、历史可靠性、协同结构与调用树几何的多维评分特征；④给出可解释的线性LTR实现与Token复杂度分析。

**🔧 技术方法**

采用检索+重排序技术（LLM检索、向量检索），学习到排名（参数化评分函数、线性LTR），图嵌入、历史可靠性特征、调用树几何特征；实现基于LangGraph与GraphRAG等工具，配合GPT‑4o进行语义编码与Prompt设计。

**📊 数据集**

构建统一调用树基准，整合八个异构多智能体语料库：agent‑data‑protocol、Agents_Failure_Attribution、GTA、MCPToolBench++、MedAgentBench、TRAIL、GUI‑360°、Seal‑Tools。

**📈 对比分析**

实验采用检索单一、检索+重排序、LLM直接、SARL（单代理）和ASRL（代理系统）多种配置进行比较；在中等规模数据集上，SARL 的 Top‑1 准确率提升约20–30%，ASRL 在所有数据集均实现 100% Top‑1；在大型数据集上，两阶段管线在 99.9–100% 之间。实验显示该方法显著提升了代理选择的可靠性、协调性与整体执行质量。

**⚠️ 局限性**

局限性包括：①依赖历史调用树，易受到日志偏差（如流行度偏见）影响，可能导致多样性下降；②对代理更新/失效敏感；③需要显式的安全、预算与公平约束；④实现依赖大量训练数据与高 Token 成本，实际部署需平衡成本与性能。

---

## 115. Continual Policy Distillation from Distributed Reinforcement Learning Teachers

**arXiv ID:** 2601.22475 | [PDF](https://arxiv.org/pdf/2601.22475v1)

**作者:** Yuxuan Li `[一作]` (Department of XXX, University of YYY), Jiayu Chen `[通讯]` (Company Name)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种融合知识蒸馏与持续学习机制的去中心化大模型学习框架，能够在多任务异构数据环境中有效训练和维护中心模型。

**💡 创新点**

创新点在于：1) 设计了任务感知的中心模型，将 Transformer 与 MoE 模块结合，充分利用连续观测空间的序列信息；2) 引入新任务蒸馏的抗遗忘机制，解决多任务冲突和分布漂移问题。

**🔧 技术方法**

采用技术包括 Transformer、Mixture-of-Experts（MoE）、任务感知知识蒸馏以及抗遗忘蒸馏机制。

**📊 数据集**

使用的数据集为 Metaworld 仿真环境。

**📈 对比分析**

通过与现有多任务/持续学习基线对比实验，结果表明在 Metaworld 任务上该框架在性能上优于最新的先进方法（具体数值未给出，但描述为显著提升）。

**⚠️ 局限性**

局限性包括：1) 未讨论通信成本和同步效率在极大规模分布式场景下的影响；2) 对抗遗忘机制的理论保证不足；3) 对非同质任务分布的泛化能力缺乏深入分析。

---

## 116. Small is Beautiful: A Practical and Efficient Log Parsing Framework

**arXiv ID:** 2601.22590 | [PDF](https://arxiv.org/pdf/2601.22590v1)

**作者:** Minxing Wang `[一作]` (Singapore Management University), Yintong Huo `[通讯]` (Singapore Management University)

**通讯引用:** 390 | [OpenAlex ID](https://openalex.org/A5080873193)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 EFParser，一种无监督的 LLM 基础日志解析框架，解决小型 LLM 在日志解析中的性能衰退。

**💡 创新点**

创新点在于双缓存自适应更新机制和模板校验模块，可纠正缓存错误并校验/修正 LLM 生成的模板，从而显著提升小模型解析精度。

**🔧 技术方法**

结合小型 LLM（如 Gemini‑1.5‑Flash‑8B）、树+桶双缓存、Levenshtein/最长公共子序列、类型与语法匹配、以及多阶段模板校正策略。

**📊 数据集**

在公开基准 Loghub‑2.0（14 大规模系统日志集，约 3500 个真模板）上进行实验。

**📈 对比分析**

与六个主流语义和语法基准对比，EFParser 在 8B 小模型上平均提升 12.5% 以上，部分指标甚至超过 GPT‑3.5 方法；在大模型上保持竞争力，处理速度比 LUNAR 等更快。

**⚠️ 局限性**

仍受 LLM 生成错误影响，尤其是过度泛化误差处理不如过度专化；双缓存虽提高精度，但略增内存占用，且在极其复杂日志场景下校正可能不足。

---

## 117. Purely Agentic Black-Box Optimization for Biological Design

**arXiv ID:** 2601.22382 | [PDF](https://arxiv.org/pdf/2601.22382v1)

**作者:** Natalie Maus `[一作]` (University of Pennsylvania), Jacob R. Gardner `[通讯]` (University of Pennsylvania)

**通讯引用:** 1957 | [OpenAlex ID](https://openalex.org/A5072585411)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种全代理、语言驱动的层级优化框架（PABLO），用预训练的化学与生物文献大语言模型在黑盒优化循环中全程生成并迭代改进生物候选物。

**💡 创新点**

创新点在于：①完全去除传统结构中心的搜索组件，改为由不同角色的LLM代理（Explorer、Planner、Worker）共同规划与执行搜索；②在全流程中直接嵌入检索增强工具与任务描述，实现语义任务感知与动态策略调整；③通过任务注册表自适应地选择和更新局部搜索策略，提升搜索效率。

**🔧 技术方法**

技术包括：层级代理架构、链式思维微调的LLM、检索增强生成（RAG）工具、自然语言任务描述、硬/软约束处理、聚合多目标与多样性优化、离线验证的最小抑菌浓度（MIC）评估。

**📊 数据集**

数据集主要包括：GuacaMol 10个多属性分子优化任务（SMILES空间）和抗菌肽（AMP）设计任务（基于APEX 1.1预测器的MIC），以及在实验室的20条抗菌肽样本用于体外活性验证。

**📈 对比分析**

与多种基线比较：Bayesian优化（NF-BO）、强化学习（GEGL）、遗传算法（Graph GA）、LLM增强方法（AlphaEvolve、LLAMBO、MOLLEO、BOPRO）等。PABLO在10个GuacaMol任务上实现了最高的Top‑1分数，样本效率显著优于对手；在AMP任务中，在相同评估预算下取得更低的预测MIC和更优的多样化投资组合，实验验证也显示对抗菌活性。

**⚠️ 局限性**

局限性包括：①依赖LLM推理速度与成本，虽然总token使用不高，但单次推理耗时较长；②在极高维空间或极大样本预算下，LLM生成多样性和搜索深度可能受限；③对外部工具的集成仍需手工设计工具接口与提示，缺乏通用化的自动化流程。

---

## 118. Weak Diffusion Priors Can Still Achieve Strong Inverse-Problem Performance

**arXiv ID:** 2601.22443 | [PDF](https://arxiv.org/pdf/2601.22443v1)

**作者:** Jing Jia `[一作]` (Rutgers University), Guanyang Wang `[通讯]` (Rutgers University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究弱扩散模型（低质量或域不匹配）在逆问题中的表现及其对重建结果的影响，给出理论解释与实践评估。

**💡 创新点**

提出利用贝叶斯一致性理论解释在信息丰富的测量下弱先验仍能得到强逆问题性能，并系统识别出其失败模式；同时提出改进的噪声优化算法（AdamSphere、HoldoutTopK）和评估框架。

**🔧 技术方法**

使用初始噪声优化（含AdamSphere、HoldoutTopK），少步DDIM采样，传统DPS对比，混合高斯模型推导的贝叶斯一致性分析。

**📊 数据集**

评估数据集包括LSUN-Bedroom、LSUN-Church、CelebA、ImageNet；任务涵盖inpainting、Gaussian deblurring、4×/16× super‑resolution、非线性去噪等。

**📈 对比分析**

与强先验DPS、DMPlug等方法对比。弱先验在信息丰富（如70%像素掩码、4×超分）场景下可匹配甚至超过强先验，PSNR/SSIM提升0.1–2 dB，LPIPS下降约0.02–0.04；但在大盒子填补、16×超分等低信息/生成任务中表现明显逊色。

**⚠️ 局限性**

局限性：仅在观测信息充分时有效；低观测维度或严重缺失导致对先验高度敏感；在盒子inpainting、超分等生成任务中效果差；方法主要基于噪声优化，可能对极端问题或非高斯噪声不适用。

---

## 119. Spatially-Adaptive Conformal Graph Transformer for Indoor Localization in Wi-Fi Driven Networks

**arXiv ID:** 2601.22322 | [PDF](https://arxiv.org/pdf/2601.22322v1)

**作者:** Ayesh Abu Lehyeh `[一作]` (University of Vermont), Safwan Wshah `[通讯]` (University of Vermont)

**通讯引用:** 826 | [OpenAlex ID](https://openalex.org/A5001816279)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种基于图变换器与自适应共形预测的室内定位框架

**💡 创新点**

首次将图变换器与空间自适应共形预测结合，提供区域级置信区域

**🔧 技术方法**

Graph Transformer、K‑Means聚类、共形预测、RSSI信号特征

**📊 数据集**

公开的SODIndoorLoc（HCXY建筑）数据集

**📈 对比分析**

相较于传统ML和GCN基线，SAC‑GT在SODIndoorLoc上 MAE 1.76 m、RMSE 2.21 m、Median 1.37 m，显著优于基线且置信覆盖约84.8%（目标 90%）

**⚠️ 局限性**

缺乏每个区域的严格条件覆盖保证，空间划分需人工设置

---

## 120. Scalable Batch Correction for Cell Painting via Batch-Dependent Kernels and Adaptive Sampling

**arXiv ID:** 2601.22331 | [PDF](https://arxiv.org/pdf/2601.22331v1)

**作者:** Aditya Narayan Ravi `[一作]` (University of Illinois Urbana-Champaign), Ilan Shomorony `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 1042 | [OpenAlex ID](https://openalex.org/A5046640187)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `da1b1a89-583a-4b57-9c81-478778569bec` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种可扩展的批次校正方法 BALANS，用于 Cell Painting 图像数据的细胞形态特征，能够在大规模数据集上有效去除实验批次效应并保留生物学信号。

**💡 创新点**

创新点在于：①将批次相关的局部尺度（基于目标批次 k‑近邻距离）嵌入高斯核中，构造批次感知的相似度；②设计了基于累计覆盖度的自适应采样策略，仅计算稀疏的邻接行，从而实现近线性时间复杂度；③提供了理论证明，证明采样量 O(K log K) 就能以较低谱误差逼近原始相似度矩阵。

**🔧 技术方法**

主要技术包括局部尺度估计的高斯核、稀疏相似度矩阵构造、覆盖度驱动的自适应采样、Nyström 型低秩逼近、Elbow 阈值稀疏化以及行归一化的光滑运算。

**📊 数据集**

使用了 JUMP Cell Painting Consortium、BBBC 以及 DeepProfiler 提取的真实 Cell Painting 数据集，规模从几千到数十万细胞；并构建了多达 5 百万点的合成高斯混合模型来检验可扩展性。

**📈 对比分析**

与 Combat、Harmony、Scanorama、fastMNN、SCVI、Seurat 等主流批次校正方法进行对比，评价指标包括 LISI、kBET、Silhouette、ARI、NMI 等。实验显示 BALANS 在保持生物学簇结构（Avg‑label、ARI、NMI）方面均优于基线，且在大规模场景下运行时间约为 30%–70% 以内，几乎达到原生实现的速度。

**⚠️ 局限性**

局限性：①方法对参数 k、τ、J 的选择仍需经验调优；②在极端高维或样本分布非常稀疏时，局部尺度估计和自适应采样的效果可能受限；③理论分析基于块对角结构和正态/指数噪声假设，实际数据中若存在强非对称噪声，近似误差可能增大。

---

## 121. Conversational Inoculation to Enhance Resistance to Misinformation

**arXiv ID:** 2601.22394 | [PDF](https://arxiv.org/pdf/2601.22394v1)

**作者:** Dániel Szabó `[一作]` (University of Oulu), Simo Hosio `[通讯]` (University of Oulu)

**通讯引用:** 3818 | [OpenAlex ID](https://openalex.org/A5067689431)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过构建基于LLM的聊天机器人系统MindFort，研究了一种新型的“会话免疫化”方法，以对话方式预先暴露并反驳错误信息，从而增强受试者对误导信息的抵抗力。

**💡 创新点**

创新点在于将认知免疫化理论与交互式人工智能相结合，利用聊天机器人进行动态、个性化的对话式预防，替代传统的阅读或写作式预防手段，首次验证了会话式预防的可行性与潜在优势。

**🔧 技术方法**

技术实现包括：Flask Web框架搭建前端；使用GPT‑4o（temperature=1）构建聊天机器人；采用认知免疫化理论设计对话流程；使用LIWC对聊天文本进行语言特征分析；使用IMI量表评估用户内在动机。

**📊 数据集**

实验数据来自65名Prolific受试者，包含四个主题的对话记录、写作文本及问卷反馈；对话文本与公开的LMSYS‑Chat‑1M LLM语料做语言特征对比。

**📈 对比分析**

研究采用within‑subject设计，三种处理（阅读、写作、聊天机器人）与无处理对照，使用Friedman检验和Wilcoxon符号秩检验；结果显示聊天机器人在控制下显著降低误导信息后信心下降（p=0.001），并在控制个体差异后效应显著高于阅读和写作，但与阅读、写作无显著差异；IMI评分无显著差异。

**⚠️ 局限性**

局限性包括样本量有限、实验时长长导致高dropout；未设置主动聊天对照，难以区分对话介入与内容效果；缺乏长期跟踪验证持久性；对话交互摩擦导致部分受试者受影响；多重比较校正导致语言特征关联分析未发现显著结果。

---

## 122. CoVA: Text-Guided Composed Video Retrieval for Audio-Visual Content

**arXiv ID:** 2601.22508 | [PDF](https://arxiv.org/pdf/2601.22508v1)

**作者:** Gyuwon Han `[一作]` (Chung-Ang University), Chanho Eom `[通讯]` (Google DeepMind)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了包含音频的组合视频检索任务CoVA，构建了首个同时考虑视觉与音频差异的AV-Comp基准，并设计了基于GFT与AVT的跨模态融合框架。

**💡 创新点**

1) 在组合检索中首次加入音频信息；2) AV-Comp数据集覆盖视觉与音频两种跨模态变化；3) 引入AVT模块实现查询感知的加权融合，超越简单平均；4) 利用Gated Fusion Transformer实现视觉–音频深度交互。

**🔧 技术方法**

使用CLIP ViT-B/32做视觉与文本编码、AST做音频编码，结合Gated Fusion Transformer对视听特征交互，AVT通过MLP预测权重进行加权融合，并使用InfoNCE对齐查询与目标。

**📊 数据集**

基准数据集AV-Comp，包含8,357条训练、1,001条测试以及1,000条额外图库视频，音频标签来源于AudioCaps 2.0，视觉特征使用CLIP。

**📈 对比分析**

与随机、单模态、双模态平均融合、以及LanguageBind/​ImageBind等基线对比；CoVA在R@1 = 35.9%、R@5 = 73.7%、R@10 = 86.4%、MnR = 6.2，明显优于其他模型。

**⚠️ 局限性**

音频单独作为查询表现差，模型仍依赖简单加权融合，冻结编码器限制了潜在性能；数据规模相对有限，可能不足以覆盖更丰富的音频变化。

---

## 123. SCaLRec: Semantic Calibration for LLM-enabled Cloud-Device Sequential Recommendation

**arXiv ID:** 2601.22543 | [PDF](https://arxiv.org/pdf/2601.22543v1)

**作者:** Ruiqi Zheng `[一作]` (La Trobe University), Hongzhi Yin `[通讯]` (University of Queensland)

**通讯引用:** 17022 | [OpenAlex ID](https://openalex.org/A5088492734)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `8d10c613-917e-4880-9716-17789f50e119` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了 SCaLRec 框架，用于在云端 LLM 语义表示被缓存并随时间过期时，自动在设备端校准并利用该语义信号以提升顺序推荐的排序质量。

**💡 创新点**

创新点在于引入可靠性估计器预测缓存语义的有效性，并设计语义校准器在不调用云端 LLM 的前提下，基于设备端最新交互自适应地修正缓存嵌入，从而缓解语义陈旧导致的排名偏差。

**🔧 技术方法**

技术实现包括：在设备端使用轻量级 MLP 估计器和校准器；利用 Spearman 相关度、交互漂移等低成本特征进行可靠性评估；通过离线知识蒸馏让校准器学习与新鲜语义一致的排名行为；云端使用 LLaMA-3.1-8B 生成语义用户嵌入，设备端采用轻量级序列模型（如 SASRec/SURGE）提取结构信号。

**📊 数据集**

实验采用两个真实世界数据集：ReDial（对话式电影推荐）和 Foursquare（位置兴趣推荐）进行评估。

**📈 对比分析**

与多种基线（纯云端推荐、纯设备推理、云-设备协同框架）在不同 staleness gap（g=0、5、10、20）下对比，SCaLRec 在 NDCG@10、HR@10 上持续取得领先，尤其在大 gap 情况下的性能提升最为显著。

**⚠️ 局限性**

局限性包括：仅校准缓存语义，未给出何时刷新云端语义的决策；需在离线阶段完成模型训练，设备侧仍需一定算力；在极端高 staleness 或冷启动场景下，校准效果可能不如预期。

---

## 124. FlexMap: Generalized HD Map Construction from Flexible Camera Configurations

**arXiv ID:** 2601.22376 | [PDF](https://arxiv.org/pdf/2601.22376v1)

**作者:** Run Wang `[一作]` (Clemson University), Siyu Huang `[通讯]` (Clemson University)

**通讯引用:** 2502 | [OpenAlex ID](https://openalex.org/A5082547392)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种名为FlexMap的、对相机配置无关的向量化高精地图构建框架，能够在不需要相机标定或2D→BEV投影的情况下，利用任意数量的未标定摄像头生成结构化地图。

**💡 创新点**

核心创新包括：
• 通过几何基础模型（VGGT）获取跨视角的几何感知特征，消除对相机内外参数的依赖；
• 引入空间-时间注意力模块，分别处理跨视角空间推理与跨时间动态建模；
• 设计基于潜在相机标记的相机感知解码器，实现视角自适应注意力与查询初始化。

**🔧 技术方法**

技术手段：
• 预训练的几何Transformer（VGGT）作为特征提取器；
• 空间-时间注意力机制（cross‑view + cross‑time）；
• 基于相机标记的层次化查询与可变温度的变形注意力；
• 交叉损失（分类、点回归、方向一致性）与Hungarian匹配。

**📊 数据集**

使用的公开数据集：nuScenes（6摄像头，1k场景）和Argoverse 2（7摄像头，1k日志）。

**📈 对比分析**

与MapTR、MapTRv2、StreamMapNet、GeMap、MapQR等现有向量化高精地图方法在无标定姿态的情形下对比。实验显示，FlexMap在单摄像头、双摄像头以及6摄像头配置下均显著提升mAP（最高约46.9），尤其在缺失视角或低质量摄像头时保持稳定性能；在Argoverse 2上亦超越对手。

**⚠️ 局限性**

局限性：
• 仍依赖高质量的几何基础模型，若VGGT推理精度下降会影响地图精度；
• 对极端遮挡或极低帧率场景的鲁棒性尚未充分验证；
• 计算量相对较大，部署于实时车载系统时需进一步优化。

---

## 125. EvoEGF-Mol: Evolving Exponential Geodesic Flow for Structure-based Drug Design

**arXiv ID:** 2601.22466 | [PDF](https://arxiv.org/pdf/2601.22466v1)

**作者:** Yaowei Jin `[一作]` (Lingang Laboratory), Qian Shi `[通讯]` (Lingang Laboratory)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `40105733-5154-44cd-8090-a8cab9e64b07` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了Evolving Exponential Geodesic Flow (EvoEGF) 框架，用信息几何方法在结构基础药物设计中实现分子生成，避免传统测地线崩溃，提升生成稳定性和几何精度。

**💡 创新点**

创新点在于：① 将分子建模为指数族分布，并沿Fisher–Rao度量的指数测地线进行流式生成；② 采用动态终点（渐进集中）策略替代传统 Dirac 终点，消除测地线收敛时的数值不稳定；③ 结合渐进参数细化的生成范式，实现高效训练与采样。

**🔧 技术方法**

使用技术包括信息几何、指数族理论、Fisher–Rao度量、指数测地线、动态终点策略、渐进参数细化（类似 BFN/PIF）、高斯与 Dirichlet 分布联合建模、深度生成网络与连续/离散数据的统一处理。

**📊 数据集**

采用的数据集为 CrossDocked2020（用于评价姿态、结合能、构象稳定性等）和 MolGenBench（用于骨架级活性回收与药物化学筛选）。

**📈 对比分析**

与 AR、Pocket2Mol、TargetDiff、DecompDiff、MolCRAFT 等基线在 CrossDocked 上通过 PoseBusters 通过率、Vina 评分、能量、QED、SA 等指标对比，EvoEGF-Mol 取得 93.4% 的 PoseBusters 通过率、最优 Vina Dock、最低 Strain Energy；在 MolGenBench 上在骨架 Pass Rate、Hit Recovery、TAScore 等指标上均超越所有基线，显示出更好的生成质量与药物化学合规性。

**⚠️ 局限性**

局限性包括：生成分子的活性覆盖率仍低（骨架 Hit 率 <5%），对未见蛋白的泛化能力有限；模型训练与推理仍需较高计算资源；此外，虽然几何精度高，但在实际实验验证前仍需进一步优化化学合理性与合成可行性。

---

## 126. DRL-Enabled Trajectory Planing for UAV-Assisted VLC: Optimal Altitude and Reward Design

**arXiv ID:** 2601.22512 | [PDF](https://arxiv.org/pdf/2601.22512v1)

**作者:** Tian-Tian Lin `[一作]` (Tongji University), Yuhan Dong `[通讯]` (Tsinghua University)

**通讯引用:** 5498 | [OpenAlex ID](https://openalex.org/A5108047157)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出一种基于深度强化学习的三维轨迹规划方法，旨在最小化UAV在VLC系统中收集数据所需的飞行距离，且对飞行高度进行了理论最优解析；

**💡 创新点**

创新点包括：1）在VLC通道增益阈值下推导出闭式最优飞行高度；2）设计了一种基于信息素的奖励机制，显著加速TD3算法收敛；3）将这两项结合，获得比传统算法更低的飞行距离和更快的学习速度；

**🔧 技术方法**

使用了TD3（Twin Delayed Deep Deterministic Policy Gradient）强化学习算法，并结合自定义的奖励函数和信息素更新；

**📊 数据集**

实验数据为仿真场景，设置I=10~30个地面用户（GU），使用固定参数（如Φ½=60°, Ψc=60°, P=10W 等）生成多种分布情况；

**📈 对比分析**

与SCAN、GREEDY‑RRT、ACO‑RRT三种基准方法对比，结果表明改进的TD3在飞行距离上比ACO‑RRT至少低22.3%，并将收敛时间缩短约50%；

**⚠️ 局限性**

局限性：仅在静态GU分布下验证；信息素奖励若设置过大可能导致震荡；对多UAV或动态环境的适用性尚未探讨；

---

## 127. High Rate Efficient Local List Decoding from HDX

**arXiv ID:** 2601.22535 | [PDF](https://arxiv.org/pdf/2601.22535v1)

**作者:** Yotam Dikstein `[一作]` (Institute for Advanced Study), Toniann Pitassi `[通讯]` (Columbia University)

**通讯引用:** 10555 | [OpenAlex ID](https://openalex.org/A5003318200)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

本文提出了首批可局部计算、近似可局部列表可解码的误码率接近信息理论极限的高效编码方案，并在多种核心任务（硬件放大、低深度列表解码等）上取得突破；

**💡 创新点**

创新点包括：①针对高维扩散器构建新的信念传播框架，将局部校正与全局扩展相结合以控制误差累积；②提出强显式低拥塞局部路由机制，使得在极稀疏的高维结构上能够在多对点间快速、低深度地构造随机路径；③将上述两项技术整合到直接积编码中，得到既高速又高率的近似局部列表可解码码；

**🔧 技术方法**

核心技术：高维扩散器（HDX）理论、局部采样与全局采样的结合、信念传播（belief propagation）算法、低拥塞路由（local routing）、直接积（direct product）与列表恢复（list recovery）的组合，以及代码连接（concatenation）与近似列表恢复的缩放技巧；

**📊 数据集**

本工作为理论研究，未使用实际数据集，所有结论均基于构造性证明与算法设计；

**📈 对比分析**

与现有方法相比，该方案在误差容忍度、编码/解码速率、查询复杂度与深度上均逼近信息理论上限；在硬件放大方面实现了无多项式域扩大的近似硬度放大；在低深度列表解码方面首次提供了RNC^1级别的实现，突破了对数平方深度壁垒；

**⚠️ 局限性**

局限性：1) 目前仅在子多项式速率（ε≪1/ln N）下实现常数率；2) 对常数误差ε时实现的常数率仍需高符号域且后续需进一步压缩；3) 对更大符号域或更严格的近似误差的适用性尚未完全证明；4) 对路由算法的容错性能虽已给出，但在更复杂错误模型下仍有待扩展。

---

## 128. FedAdaVR: Adaptive Variance Reduction for Robust Federated Learning under Limited Client Participation

**arXiv ID:** 2601.22204 | [PDF](https://arxiv.org/pdf/2601.22204v1)

**作者:** S M Ruhul Kabir Howlader `[一作]` (University of Leicester), Lu Liu `[通讯]` (University of Exeter)

**通讯引用:** 9531 | [OpenAlex ID](https://openalex.org/A5002822427)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了FedAdaVR及其量化版FedAdaVR‑Quant，利用服务器端自适应优化与SAGA式方差削减，解决FL中的部分客户端参与导致的方差问题；

**💡 创新点**

创新点在于将自适应优化器与方差削减结合，并通过存储并量化最近客户端更新来消除部分参与误差；

**🔧 技术方法**

采用自适应优化器（Adam、Adagrad、Adabelief、Yogi、Lamb）与SAGA‑风格方差削减技术，以及FP16/Int8/Int4量化存储；

**📊 数据集**

在MNIST、FMNIST、CIFAR‑10（Vision）和Shakespeare（NLP）等数据集上进行实验；

**📈 对比分析**

与FedAvg、FedProx、SCAFFOLD、FedNova、FedAdam、FedAdagrad、FedYogi、MIFA、FedVARP等方法对比，FedAdaVR在绝大多数分区和数据集上获得更高精度、更快收敛，且不增加客户端通信或计算；

**⚠️ 局限性**

局限性包括仅使用线性定量化，未探究更高级压缩方法；对不同模型架构的泛化尚未验证；超参数调优对结果影响较大。

---

## 129. Bifocal Attention: Harmonizing Geometric and Spectral Positional Embeddings for Algorithmic Generalization

**arXiv ID:** 2601.22402 | [PDF](https://arxiv.org/pdf/2601.22402v1)

**作者:** Kanishk Awadhiya `[一作]` `[通讯]` (Indian Institute of Technology), Kanishk Awadhiya (Indian Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Spectral-RoPE与Bifocal Attention以提升Transformer在递归逻辑任务上的泛化能力。

**💡 创新点**

将位置编码拆分为可学习的频率、幅值、相位三元组，形成自适应“谐波桥接”，从而打破固定几何的Spectral Rigidity。

**🔧 技术方法**

基于RoPE的旋转位置编码，引入可学习频率、幅值、相位参数，并通过Surgical Integration将其替换到预训练模型中。

**📊 数据集**

使用Dyck-3、Bio-Rotation、Modulo Arithmetic等形式化语言数据集，以及PyTorch源码数据集进行实验。

**📈 对比分析**

与基准Llama-2-7b对比，在三项任务中Spectral-RoPE将损失降至近零，提升约99%，显著优于标准RoPE。

**⚠️ 局限性**

在自然语言推断或长距离语义任务中的效果尚未验证；对特定结构的过度拟合可能导致普适性不足。

---

## 130. MM-OpenFGL: A Comprehensive Benchmark for Multimodal Federated Graph Learning

**arXiv ID:** 2601.22416 | [PDF](https://arxiv.org/pdf/2601.22416v1)

**作者:** Xunkai Li `[一作]` (Beijing Institute of Technology), Guoren Wang `[通讯]` (Beijing Institute of Technology)

**通讯引用:** 7221 | [OpenAlex ID](https://openalex.org/A5054991337)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出MM-OpenFGL基准，系统化研究多模态联邦图学习（MMFGL）

**💡 创新点**

①首次给出MMFGL的完整问题定义与三维仿真框架；②构建19个多模态图数据集、8种仿真策略、57种算法，覆盖图级与模态级任务；③通过10条实证结论揭示跨模态对齐、结构鲁棒性及通信效率的重要性

**🔧 技术方法**

多模态图神经网络、标准与异构联邦学习算法、图基础模型；联合使用预训练多模态编码器（如Qwen、Clip、DINOv2）以及自监督联邦预训练+局部微调两阶段流程

**📊 数据集**

19个跨领域多模态图数据集（电影、服装、推荐、社交、医疗等），包括文本、图像、属性及边结构；对每个数据集进行模态-IID/Non-IID、拓扑可用/不可用、标签-IID/Non-IID的仿真划分

**📈 对比分析**

对比了三大类别（MM-GNN、标准FL、异构FL、图基础模型）在8种仿真场景下的节点分类、链路预测、模态匹配/检索/生成等任务；实验显示：1）多模态融合显著优于单模态；2）现有联邦方法在多模态场景中表现不一，异构FL（如MH-pFLID）最稳健；3）图基础模型在多任务上整体领先；4）通信与收敛效率各异，轻量级语义对齐方法收敛快、通信低

**⚠️ 局限性**

仍存在：
• 对极端模态缺失或标签噪声的鲁棒性不足；
• 对不同特征编码器的依赖性强；
• 服务器端聚合成为计算瓶颈；
• 大规模多模态联邦训练的通信与内存开销高

---

## 131. The Unseen Threat: Residual Knowledge in Machine Unlearning under Perturbed Samples

**arXiv ID:** 2601.22359 | [PDF](https://arxiv.org/pdf/2601.22359v1)

**作者:** Hsiang Hsu `[一作]` (JPMorgan Chase), Chun-Fu Chen `[通讯]` (JPMorgan Chase)

**通讯引用:** 4284 | [OpenAlex ID](https://openalex.org/A5080417608)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种在模型中实现“遗忘”而不需重新训练的微调策略 RURK，并发现并量化了遗忘样本的残留知识隐私风险。

**💡 创新点**

创新点在于将残留知识定义为对被遗忘样本在对抗扰动下的可辨性，并通过额外惩罚项显式抑制这类知识，从而兼顾模型准确性与隐私安全。

**🔧 技术方法**

采用了（ε,δ）可区分性与 Rényi unlearning 的理论框架，结合 FGSM/PGD 等对抗样本生成以及基于梯度的微调（含噪声梯度下降）实现了 RURK。

**📊 数据集**

实验使用的基准数据集包括 CIFAR‑5、CIFAR‑10 以及 ImageNet‑100（100 类子集），全部使用 ResNet‑18/ResNet‑50 等卷积网络。

**📈 对比分析**

与 11 种现有 unlearning 方法对比，RURK 在保持测试准确率、MIA 失败率、重学时间等指标上均表现更好，并且残留知识指标大幅降低，显示出更强的隐私保护效果。

**⚠️ 局限性**

局限性包括：1）难以完全消除所有扰动下的可辨性；2）需访问并利用被遗忘样本；3）对大规模模型的计算成本仍高于部分基线；4）无法在理论上保证对所有对抗扰动的不可辨性。

---

## 132. Screen, Match, and Cache: A Training-Free Causality-Consistent Reference Frame Framework for Human Animation

**arXiv ID:** 2601.22160 | [PDF](https://arxiv.org/pdf/2601.22160v1)

**作者:** Jianan Wang `[一作]` (Fudan University), Wenqiang Zhang `[通讯]` (Fudan University)

**通讯引用:** 3337 | [OpenAlex ID](https://openalex.org/A5100669255)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一个训练无关的三阶段框架 FrameCache，用来提升长序列人物动画的时间一致性和视觉稳定性。

**💡 创新点**

创新点在于：① 通过质量感知筛选（Screen）动态挑选高质量帧；② 采用冗余感知缓存维护（Cache）保持多样性和相关性；③ 通过运动一致性匹配（Match）在生成时选择最符合当前动作的参考帧，将隐式时间信息显式化。

**🔧 技术方法**

技术手段包括：无参考图像质量评估（CLIP‑IQA、MUSIQ）做帧筛选；基于余弦相似度的冗余度估计与替换策略；运动序列与缓存帧的平均相似度匹配；与基线模型无缝对接，保持可插拔性。

**📊 数据集**

使用了 21 条未见过的 TikTok 视频（尺寸 512×512）作为评估数据集，并在 MagicAnimate、StableAnimator 两个基线上进行实验。

**📈 对比分析**

比较方法：在无参考（NIQE、NIMA、CLIP‑IQA、TRES‑FLIVE、PAQ2PIQ、MUSIQ）和全参考（CKDN）指标上评估；结果显示在 StableAnimator 上提升显著，MagicAnimate 上效果稳定且略有提升，整体实现了更高的时间一致性和视觉稳定性。

**⚠️ 局限性**

局限性：1) 效果高度依赖基线模型的时间建模能力；2) 对真实与合成帧的细微差异敏感，可能导致低层图像属性的轻微偏移；3) 不是通用万能方案，需在兼容性与自适应缓存策略方面进一步研究。

---

## 133. Mitigating Hallucinations in Video Large Language Models via Spatiotemporal-Semantic Contrastive Decoding

**arXiv ID:** 2601.22574 | [PDF](https://arxiv.org/pdf/2601.22574v1)

**作者:** Yuansheng Gao `[一作]` (Zhejiang University), Wenzhi Chen `[通讯]` (Zhejiang University)

**通讯引用:** 3998 | [OpenAlex ID](https://openalex.org/A5101562846)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种新的解码策略SSCD，利用空间-时间与语义对比的负样本特征来抑制VideoLLM的幻觉；

**💡 创新点**

创新点在于通过随机游走构造时空一致性破坏、条件互信息驱动的语义破坏，并将两类负样本融入对比解码，形成轻量且可插拔的幻觉抑制方案；

**🔧 技术方法**

使用轻量级空间-时间语义破坏器、随机游走图模型、条件互信息损失、对比解码以及可调节的可解释阈值来实现；

**📊 数据集**

在ShareGPT4Video 3000条样本上训练破坏器，评估使用VideoHallucer、EventHallusion、VideoHallu、ActivityNet‑QA、MMVU等公开基准；

**📈 对比分析**

与TCD、MotionCD、Dino‑Heal等基线对比，SSCD在幻觉相关指标上 consistently 取得最优或次优表现，同时保持甚至提升视频理解与推理的准确率；

**⚠️ 局限性**

局限性包括对对比信号质量敏感、受视频编码信息的限制以及对超参数（α、β、λ等）的调参依赖。

---

## 134. Tabular Foundation Models Can Do Survival Analysis

**arXiv ID:** 2601.22259 | [PDF](https://arxiv.org/pdf/2601.22259v1)

**作者:** Da In Kim `[一作]` (Imperial College London), Kelly W. Zhang `[通讯]` (Imperial College London)

**通讯引用:** 403 | [OpenAlex ID](https://openalex.org/A5101990414)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

将生存分析转化为多阶段二分类问题，利用预训练的表格基础模型进行无监督的生存预测

**💡 创新点**

通过时间离散化与二分类框架，使得表格基础模型能够处理右删失且无需专门训练，并在样本增大时证明一致性

**🔧 技术方法**

采用时间离散化、二分类交叉熵、表格基础模型（MITRA、TabPFN）与无监督上下文学习

**📊 数据集**

使用 SurvSet 公开数据集（43 个静态、5 个动态真实世界右删失生存数据集）

**📈 对比分析**

与 CoxPH、Random Survival Forest、DeepSurv、DeepHit 等传统与深度生存模型对比，在 C-index、Integrated AUC、IBS 等指标上均表现更优，MITRA 成为最强模型

**⚠️ 局限性**

局限性在于仅针对离散时间，未对连续时间建模；对时间离散化的选择敏感；未探讨因果/治疗效应、模型细化训练及更复杂的删失机制

---

## 135. Multitask Learning for Earth Observation Data Classification with Hybrid Quantum Network

**arXiv ID:** 2601.22195 | [PDF](https://arxiv.org/pdf/2601.22195v1)

**作者:** Fan Fan `[一作]` (Technical University of Munich), Xiao Xiang Zhu `[通讯]` (Technical University of Munich)

**通讯引用:** 25220 | [OpenAlex ID](https://openalex.org/A5068384981)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种多任务量子-经典混合网络 MLTQNN，用于地球观测影像的分类。

**💡 创新点**

创新点在于通过图像重建辅助任务压缩特征、引入位置权重模块，并在量子卷积层中融合两者以实现高效量子数据编码。

**🔧 技术方法**

主要使用量子态编码、量子卷积、位置权重模块、交叉熵与MSE双任务损失，以及 TensorFlow Quantum 进行无噪声模拟。

**📊 数据集**

实验数据集包括 SAT‑6、LCZ42、EuroSAT 与 PatternNet 四个遥感分类基准。

**📈 对比分析**

与经典多任务网络 MLTCNN、单任务网络 SEQNN 和纯 CNN 进行对比，MLTQNN 在 4 大数据集上均实现 0.937–0.969 的平均精度，尤其在样本稀缺和类别不平衡场景中保持更高的准确率与 F1 分数。

**⚠️ 局限性**

主要限制是目前仅在无噪声模拟器上验证，量子电路资源（12 个量子比特）受限，未在真实量子硬件上测试，且对噪声鲁棒性与大规模数据处理尚不充分。

---

## 136. Recursive Mutexes in Separation Logic

**arXiv ID:** 2601.22557 | [PDF](https://arxiv.org/pdf/2601.22557v1)

**作者:** Ke Du `[一作]` (University of Illinois Chicago), Gregory Malecha `[通讯]` (Skylabs AI)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

**🎯 论文内容**

为递归互斥锁在分离逻辑中提供统一的规范，并演示如何使用该规范验证C++标准库中递归互斥的正确性。

**💡 创新点**

提出holds谓词和借用规则，使得线程不论是否已持有锁都能安全获取和释放锁，同时避免在递归调用中丢失锁保护资源的完整信息。

**🔧 技术方法**

基于分离逻辑的规范技术，利用BRiCk工具链和逻辑原子规格来定义递归锁的获取、释放与借用。

**📊 数据集**

无数据集，主要通过理论证明和示例代码展示规范的适用性。

**📈 对比分析**

本文未进行实验或性能对比，只通过逻辑推理证明规范的正确性。

**⚠️ 局限性**

目前仅覆盖递归互斥，未扩展到所有C++同步原语；实现高度依赖BRiCk内部实现细节，实际应用范围受限。

---

## 137. Anytime Safe PAC Efficient Reasoning

**arXiv ID:** 2601.22446 | [PDF](https://arxiv.org/pdf/2601.22446v1)

**作者:** Chengyao Yu `[一作]` (Southern University of Science and Technology), Bingyi Jing `[通讯]` (Chinese University of Hong Kong)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种 B-PAC 推理方法，用于在线环境下在部分反馈条件下实现安全且高效的推理；

**💡 创新点**

将阈值更新建模为赌博过程，利用逆倾向评分 (IPS) 估计、超马丁格尔与固定序列检验实现任意时间的 PAC 安全保证；

**🔧 技术方法**

使用 IPS 估计、超马丁格尔、Ville 不等式、固定序列检验、可调赌注策略、混合马丁格尔等统计工具；

**📊 数据集**

在四个推理基准上评估：MATH、MMLU-Pro、BIG-Bench Hard（BBH）和 Magpie，使用 Qwen3 作为思考模型与其轻量版作为非思考模型；

**📈 对比分析**

与离线 PAC 推理、IPS+Hoeffding、O-naive、CoD、NoThinking 等在线与离线方法比较；B-PAC 在保证误差容忍度 ϵ 且置信度 1-α 的前提下，将专家调用率（ECP）从 18.99% 降至 47% 以上，同时令 token 消耗大幅下降，整体性能优于基线且安全可靠；

**⚠️ 局限性**

目前仅支持双模型切换，需进一步扩展至多模型；效率依赖不确定性评分质量，缺乏多评分自适应选择机制；条件式 B-PAC 的设计仍待探索。

---

## 138. Investigating the Interplay of Parameterization and Optimizer in Gradient-Free Topology Optimization: A Cantilever Beam Case Study

**arXiv ID:** 2601.22241 | [PDF](https://arxiv.org/pdf/2601.22241v1)

**作者:** Jelle Westra `[一作]` (Leiden University), Elena Raponi `[通讯]` (Leiden University)

**通讯引用:** 406 | [OpenAlex ID](https://openalex.org/A5089662417)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

在此研究中，作者通过对悬臂梁的最小化符合度问题进行实验，探讨了几何参数化与黑盒优化器在拓扑优化中的相互作用。

**💡 创新点**

创新点在于系统性比较了不同参数化质量对优化效果的影响，并提出“参数化优先于优化器”这一结论。

**🔧 技术方法**

采用的技术包括差分进化(DE)、协方差矩阵自适应进化策略(CMA‑ES)、以及异方差演化贝叶斯优化(HEBO)等三种代表性黑盒优化算法。

**📊 数据集**

实验使用基于有限元（100×50 网格）的悬臂梁模型，构建了10D、20D、50D三种设计维度的三种参数化（MMC、曲线MMC、蜂巢镶嵌）。

**📈 对比分析**

通过27种配置、每种配置15次独立运行的实验，结果显示参数化质量对最终符合度影响更大；在高质量参数化下三种优化器表现相近，低质量参数化下优化器差异显著，DE在蜂巢参数化下表现最佳。

**⚠️ 局限性**

局限性包括仅聚焦单一悬臂梁案例、只测试三种优化器、实验规模受计算预算限制，且未验证结果在更大尺度或不同工程问题中的普适性。

---

## 139. Chance-Constrained Secrecy Optimization in Hybrid RIS-Empowered and UAV-Assisted Networks

**arXiv ID:** 2601.22499 | [PDF](https://arxiv.org/pdf/2601.22499v1)

**作者:** Elhadj Moustapha Diallo `[一作]` (Ningbo Ciruan Software Development Co., Ltd.), Muhammad Naeem Shah `[通讯]` (Ningbo Ciruan Software Development Co., Ltd.)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出并优化了集成UAV挂载RIS、STAR-RIS和H-RIS的混合可重构系统，以降低混合室内外用户的安全盲区概率

**💡 创新点**

创新在于联合三种RIS结构与UAV定位的全局优化、分布式鲁棒安全设计与Bernstein型确定性近似

**🔧 技术方法**

采用概率约束的分布式鲁棒优化、Bernstein型不等式、SCA-交替优化以及3GPP/ITU-R通道模型

**📊 数据集**

实验使用基于3GPP TR 38.901、TR 36.873、ITU‑R P.2109的仿真数据集

**📈 对比分析**

与UAV‑RIS单体、STAR‑RIS单体、RIS单体基线相比，所提方案在不同功率、保密率、QoS下实现了更低的安全盲区概率并保持了更好的鲁棒性

**⚠️ 局限性**

主要局限在于Bernstein近似的保守性、算法复杂度高以及对实时动态场景的适配性有限

---

## 140. Keep Rehearsing and Refining: Lifelong Learning Vehicle Routing under Continually Drifting Tasks

**arXiv ID:** 2601.22509 | [PDF](https://arxiv.org/pdf/2601.22509v1)

**作者:** Jiyuan Pei `[一作]` (Victoria University of Wellington), Xin Yao `[通讯]` (Lingnan University)

**通讯引用:** 66902 | [OpenAlex ID](https://openalex.org/A5100635494)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了在持续漂移任务环境下的车辆路径规划（VRP）终身学习框架DREE，用经验重放实现持续学习。

**💡 创新点**

创新点在于同时使用问题实例重放、行为重放和经验增强三种机制，并通过动态调度改进重放质量。

**🔧 技术方法**

采用深度强化学习的构造型神经求解器（如POMO）与经验缓冲、轨迹重放、目标函数比较等技术。

**📊 数据集**

使用基于现有任务顺序构造的CVRP/TSP连续漂移数据集，并在TSPLIB/CVRPLIB上进行泛化评测。

**📈 对比分析**

与微调、Li、LLR‑BC以及多任务基准对比，DREE在平均性能、遗忘率、最优可塑性等指标上均优于现有方法，并接近全任务学习基准。

**⚠️ 局限性**

局限在于对所有时间步同等对待，未考虑任务重要性权重；对漂移程度的自动适应与约束漂移的扩展仍待研究。

---

## 141. Culturally Grounded Personas in Large Language Models: Characterization and Alignment with Socio-Psychological Value Frameworks

**arXiv ID:** 2601.22396 | [PDF](https://arxiv.org/pdf/2601.22396v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 142. SPLA: Block Sparse Plus Linear Attention for Long Context Modeling

**arXiv ID:** 2601.22379 | [PDF](https://arxiv.org/pdf/2601.22379v1)

**作者:** Bailin Wang `[一作]` (Apple), Chong Wang `[通讯]` (Apple)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Sparse Plus Linear Attention（SPLA）框架，将长文本上下文分为精确注意和残差线性注意两部分，既提升块选择精度，又压缩未选块信息；

**💡 创新点**

创新点在于：①基于二阶泰勒展开得到的块级重要性度量，实现无训练的高召回块选择；②残差线性注意（RLA）模块利用差分实现IO高效的全局上下文压缩，避免长尾丢失；

**🔧 技术方法**

技术包括块级统计（均值、协方差）、泰勒近似、分组查询注意（GQA）、残差线性注意的差分实现、门控残差融合以及统一的 TPU kernel；

**📊 数据集**

使用了 10 万亿 token 的预训练数据、RULER、SciQ、TriviaQA、WebQ、MMLU、GSM8k、LAMBADA、PiQA、HellaSwag、WinoGrande、ARC、OpenReasoning、AIME、HMMT、LiveCodeBench、HumanEval、MMLU Pro、GPQA 等多任务数据集；

**📈 对比分析**

与完整注意力、NSA 与 InfLLM‑v2 等基线在 32k、128k、256k 上下文长度下进行对比，SPLA 在长上下文（>64k）甚至超过密集模型表现，同时保持或提升通用知识、推理任务性能，且训练/推理效率较密集模型略低但可接受；

**⚠️ 局限性**

局限在：①训练时仍需额外小规模参数；②在较小的 GQA 组大小下推理时会出现速度瓶颈；③对超大批量或极端长文本的进一步优化尚未验证。

---

## 143. When LLM meets Fuzzy-TOPSIS for Personnel Selection through Automated Profile Analysis

**arXiv ID:** 2601.22433 | [PDF](https://arxiv.org/pdf/2601.22433v1)

**作者:** Shahria Hoque `[一作]` (BRAC University), Nirjhar Gope `[通讯]` (BRAC University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了基于大型语言模型与模糊TOPSIS的自动化人员选拔系统，利用LinkedIn简历文本进行多类别评估与排名。

**💡 创新点**

将DistilRoBERTa的多类预测与模糊TOPSIS相结合，采用三角模糊数处理评估不确定性，构建可解释、可扩展的招聘决策框架。

**🔧 技术方法**

使用DistilRoBERTa、知识蒸馏、数据增强、三角模糊数、模糊TOPSIS、AHP权重以及多种性能评估指标（准确率、F1、MAP、NDCG等）。

**📊 数据集**

约100份软件工程师LinkedIn个人资料，包含Experience、Skills、Education、About四项，人工给出五分制评分，并通过数据增强扩增至10k样本。

**📈 对比分析**

与RoBERTa-base、LastBERT等基线模型比较；DistilRoBERTa+Fuzzy-TOPSIS在各属性分类准确率≈91%，排名与人类专家高度一致，MAP≈0.99、NDCG≈0.92、RMSE≈0.04。

**⚠️ 局限性**

数据量小、仅软件工程领域、人工标注成本高、模糊权重手工设定、缺乏跨文化、多语言评估、模型解释性待提升。

---

## 144. Failing to Explore: Language Models on Interactive Tasks

**arXiv ID:** 2601.22345 | [PDF](https://arxiv.org/pdf/2601.22345v1)

**作者:** Mahdi JafariRaviz `[一作]` (University of Maryland), Soheil Feizi `[通讯]` (University of Maryland)

**通讯引用:** 10336 | [OpenAlex ID](https://openalex.org/A5025450606)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估语言模型在有限交互预算下的探索能力，提出三类可参数化任务，并引入并行预算分配与周期性摘要两种干预手段。

**💡 创新点**

创新点：①构建可调节探索难度的交互式任务框架；②揭示LLM普遍出现“过早承诺”导致低效探索的现象；③证明并实践并行线程和摘要生成能显著提升探索效果。

**🔧 技术方法**

技术：利用多种语言模型（GPT‑4、GPT‑3.5、Claude、Llama 等）作为代理，通过自然语言交互与oracle；任务实现使用高斯山函数、树搜索（含陷阱）与 SAT 问题；干预技术包括并行预算拆分、摘要生成与上下文替换；理论分析证明并行在理论上无优势但在实验中有效。

**📊 数据集**

数据集：自制任务实例，包含三类任务——连续高斯山函数、带陷阱的树结构、含金句的 SAT；每类任务通过可控参数生成多实例以调节难度。

**📈 对比分析**

比较方法：与简单的探索‑利用基线（随机采样+局部精炼、随机树探索、随机赋值+局部变异）对比；评估不同交互预算下归一化奖励的增长；结果显示 LLm 在所有任务上均低于基线，并且随预算增长的提升幅度较小；并行/摘要干预能显著提升 LLm 性能。

**⚠️ 局限性**

局限性：实验仅在小规模自制任务上进行，缺乏对真实复杂环境的验证；LLM 受上下文窗口限制，摘要策略尚未系统化；并行方法理论上无优势，需进一步探究其在实践中的有效机制。

---

## 145. Molecular Representations in Implicit Functional Space via Hyper-Networks

**arXiv ID:** 2601.22327 | [PDF](https://arxiv.org/pdf/2601.22327v1)

**作者:** Zehong Wang `[一作]` (University of Notre Dame), Yanfang Ye `[通讯]` (University of Notre Dame)

**通讯引用:** 5057 | [OpenAlex ID](https://openalex.org/A5027601906)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出 MolField，将分子建模为连续、SE(3)不变的三维函数，并通过超网络学习分子函数分布；

**💡 创新点**

创新点在于把分子视为连续函数而非离散结构，使用统一的隐式网络表示与结构化权重标记实现任务无关的函数空间表示；

**🔧 技术方法**

核心技术包括 Canonical Implicit Neural Representation (C‑INR)、Structured Weight Tokenization (SWT) 以及 Function‑Space Hyper‑Network (FSHN)；

**📊 数据集**

实验使用分子动力学轨迹（如 YiiP、NahA、I‑FABP、adk_equi 等）以及 QM9 属性预测数据集；

**📈 对比分析**

与传统 GNN、CNN、Equivariant 等方法比较，在动态表面重建、属性预测、数据效率和鲁棒性上均取得更优的指标（如更高 IoU、更低 MAE 等）；

**⚠️ 局限性**

局限性包括仅处理非反应性或有限动力学系统，未考虑化学合成可行性、量子力学细节以及大规模生物分子等复杂情形。

---

## 146. Temporal Graph Pattern Machine

**arXiv ID:** 2601.22454 | [PDF](https://arxiv.org/pdf/2601.22454v1)

**作者:** Yijun Ma `[一作]` (University of Notre Dame), Yanfang Ye `[通讯]` (University of Notre Dame)

**通讯引用:** 5057 | [OpenAlex ID](https://openalex.org/A5027601906)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种基于演化模式的时序图学习框架 TGPM，用交互补丁（由时序偏置随机游走构建）来捕捉长程结构和时间依赖，并通过 Transformer 编码与自监督预训练学习可迁移的演化机制；

**💡 创新点**

创新点在于：① 将时序图建模转向“演化模式”视角，摒弃仅基于一跳邻域、短期依赖和回顾性时间建模的传统假设；② 通过时序偏置随机游走生成交互补丁，充分体现多尺度结构语义；③ 设计了两种自监督任务（Masked Token Modeling 与 Next Time Prediction）实现多尺度时序依赖与前瞻性时间预测；

**🔧 技术方法**

使用技术包括：时序偏置随机游走、Transformer 编码器、Sinusoidal 时序编码、Masked Token Modeling (block‑wise)、Next Time Prediction、均值池化、基准自监督预训练、下游链接预测头；

**📊 数据集**

主要使用的公开时序图基准数据集有 Enron、ICEWS1819 和 Googlemap CT（分别来自邮件、知识图谱和电商领域）；

**📈 对比分析**

与现有监督模型（TGAT、GraphMixer、DyGFormer）和自监督模型（PT‑DGNN、DDGCL、CPDG）在传导与归纳式链接预测任务中比较；TGPM 在多数数据集上均取得最优或次优成绩，跨域迁移实验更显优势；

**⚠️ 局限性**

局限性在于：对极端时间突发且同质并发交互（如 Enron）易出现预训练退化；对时序突发性和高同质度的并发交互处理不足；此外缺乏对缺失数据、随机动态和部署反馈等实际情况的建模。

---

## 147. Head-Aware Visual Cropping: Enhancing Fine-Grained VQA with Attention-Guided Subimage

**arXiv ID:** 2601.22483 | [PDF](https://arxiv.org/pdf/2601.22483v1)

**作者:** Junfei Xie `[一作]` (Ping An Technology), Xulong Zhang `[通讯]` (University of Science and Technology of China)

**通讯引用:** 656 | [OpenAlex ID](https://openalex.org/A5115593108)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种训练无关的多模态视觉裁剪框架（HAVC），通过先对注意力头进行OCR诊断过滤，再在推理时使用空间熵与梯度敏感度两支路精细筛选并融合，生成可引导模型聚焦的视觉裁剪指导图，从而提升细粒度视觉问答的准确性。

**💡 创新点**

创新点包括：①仅保留能够与真实视觉区域对齐的专家注意力头，显著减少噪声；②引入空间熵过滤和梯度敏感度评分两种互补信号，兼顾空间聚焦与预测贡献；③通过归一化、加权融合以及温度软最大化得到最终裁剪指导图，实现在推理阶段无需额外训练即可实现高质量裁剪。

**🔧 技术方法**

技术细节：OCR诊断任务得到每个头的投影分数；空间熵计算（基于Otsu阈值分割、连通域中心距离）衡量聚焦度；梯度敏感度计算为对数概率对注意力向量的梯度；分数归一化与融合后用温度软最大化得到权重；最终对裁剪区域生成置信度地图并裁剪子图送入MLLM。

**📊 数据集**

数据集与实验：OCR诊断使用Synthdog；细粒度VQA评测在AOKVQA、POPE、TextVQA、V*；通用VQA评测在VQAv2、GQA；基线模型为LLaVA‑1.5与InstructBLIP（Vicuna‑7B），并与ViCrop三种变体对比。

**📈 对比分析**

与原始MLLM和ViCrop比较，HAVC在大多数基准上实现显著提升（例如POPE 85.8%、TextVQA 57.6%、VQAv2 77.8%、GQA 72.5%），在LLaVA‑1.5上5/6个任务获最佳或竞争性表现，InstructBLIP上也取得多项最高分，验证了其鲁棒性和有效性。

**⚠️ 局限性**

局限性：需要OCR图像进行头筛选，可能对无文本图像的泛化不足；在某些基准（如InstructBLIP的V*）对裁剪策略敏感度低；依赖一系列超参数（阈值、α、K、τ），虽然整体鲁棒但在极端设置下性能可能下降。

---

## 148. Partial Rewriting and Value Interpretation of Logically Constrained Terms (Full Version)

**arXiv ID:** 2601.22191 | [PDF](https://arxiv.org/pdf/2601.22191v1)

**作者:** Takahito Aoto `[一作]` (Niigata University), Jonas Schöpf `[通讯]` (Data Lab Hell)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

提出并研究了对存在性约束项的部分约束重写（partial constrained rewriting）形式，并与传统的最广义约束重写（most general constrained rewriting）进行系统对比；

**💡 创新点**

创新点在于引入部分约束重写概念（仅要求可满足而非必然有效的约束），并提出价值解释（value interpretation）以精确区分两种重写及其正规化；

**🔧 技术方法**

采用逻辑约束重写系统（LCTRS）框架、存在性约束项、子句和等价关系、以及SMT求解技术来定义、证明性质和构造解释；

**📊 数据集**

本文未使用具体实验数据集，而是基于形式化定义和理论证明进行研究；

**📈 对比分析**

通过理论证明和示例展示了部分约束重写在处理部分实例时更为精细，并证明其在正规化判定中的优越性，但未给出时间复杂度或运行时性能比较；

**⚠️ 局限性**

主要限制是仅针对左线性且左值自由的约束重写规则，未讨论非左线性情况；

---

## 149. PerfGuard: A Performance-Aware Agent for Visual Content Generation

**arXiv ID:** 2601.22571 | [PDF](https://arxiv.org/pdf/2601.22571v1)

**作者:** Zhipeng Chen `[一作]` (Beijing University of Posts and Telecommunications), Yi-Zhe Song `[通讯]` (University of Surrey)

**通讯引用:** 11429 | [OpenAlex ID](https://openalex.org/A5046046128)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研发了 PerfGuard 框架，解决 LLM 代理在视觉内容生成任务中工具选择不精准、执行不可靠的问题，通过性能感知工具选择、适应性偏好更新和能力对齐规划优化，实现任务规划与执行的闭环闭合。

**💡 创新点**

创新点在于：① 引入多维度工具性能边界评估（PASM）替代传统文本描述；② 通过 Adaptive Preference Updating（APU）根据实际执行反馈动态调整工具偏好；③ 将性能信息嵌入 Planner 的 Capability‑Aligned Planning Optimization（CAPO）中，使规划与工具执行实时对齐。

**🔧 技术方法**

使用了标准化代理架构（Analyst‑Planner‑Worker‑Self‑Evaluator）、多维评分矩阵、探索‑利用策略、SPO/CAPO 强化学习优化、CLIP 检索经验以及多种 LLM（QWen3‑VL‑32B、GPT‑4o 等）。

**📊 数据集**

使用的数据集包括 T2I‑CompBench、OneIG‑Bench、Complex‑Edit、ImgEdit‑Bench 以及 T2I‑compbench 等，用于工具性能边界定义和任务评估。

**📈 对比分析**

在基本生成、先进生成和复杂编辑三个基准上与 FLUX、SD3、T2I‑R1、GoT、GenArtist、T2I‑Copilot 等方法做定量和定性比较，PerfGuard 在工具选择准确率、执行可靠性和与用户意图对齐方面均表现最佳（如在 T2I‑CompBench 的属性绑定、对象关系、复杂度等指标均高于对手）。

**⚠️ 局限性**

局限性在于：① 依赖已有基准得分初始化性能矩阵，若领域缺乏高质量基准则难以直接迁移；② 目前验证仅针对视觉内容生成，缺乏跨领域及多代理协作的实验与机制。

---

## 150. Why Self-Rewarding Works: Theoretical Guarantees for Iterative Alignment of Language Models

**arXiv ID:** 2601.22513 | [PDF](https://arxiv.org/pdf/2601.22513v1)

**作者:** Shi Fu `[一作]` (Nanyang Technological University), Dacheng Tao `[通讯]` (Nanyang Technological University)

**通讯引用:** 98256 | [OpenAlex ID](https://openalex.org/A5074103823)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并理论证明了自奖励语言模型（SRLM）在迭代对齐过程中能够自我纠正并收敛，克服单步自监督的局限。

**💡 创新点**

创新点在于：①给出单步更新的下界证明了对初始模型的严重依赖；②推导出迭代更新的有限样本误差上界，展示了对初始条件数的指数衰减；③将理论框架应用于线性softmax模型，提供了可计算的经验风险界限，并引入有效维度分析。

**🔧 技术方法**

采用的技术包括自奖励信号（log‑probability）、DPO 训练目标、KL 正则化、收敛性分析（收缩映射）、统计学习理论中的PAC 风格下界与上界、以及谱有效维度方法。

**📊 数据集**

实验数据主要使用从真实分布 μ 随机抽样的无标签提示集，未公开特定公开数据集；理论验证基于通用假设和线性 softmax 参数化。

**📈 对比分析**

与传统 RLHF 和单步自监督方法对比，SRLM 在多轮迭代后可实现 O(1/√n) 的收敛速度，并且对初始模型质量的敏感性显著下降，表现出更稳健的对齐效果。

**⚠️ 局限性**

局限性包括：理论主要针对单一模型类（线性 softmax）和理想化假设，未涉及完整 Transformer 结构；有效维度假设对特征协方差谱的指数衰减要求较强；实际应用中对样本大小、迭代次数与计算成本的权衡仍待进一步研究。

---

## 151. Toward Third-Party Assurance of AI Systems: Design Requirements, Prototype, and Early Testing

**arXiv ID:** 2601.22424 | [PDF](https://arxiv.org/pdf/2601.22424v1)

**作者:** Rachel M. Kim `[一作]` (Carnegie Mellon University), Rayid Ghani `[通讯]` (Carnegie Mellon University)

**通讯引用:** 18392 | [OpenAlex ID](https://openalex.org/A5106542734)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

**🎯 论文内容**

设计并验证了一个第三方AI保障框架，包括责任分配矩阵、访谈协议、成熟度矩阵和报告模板，并在两套真实企业/政府AI系统中进行早期测试和专家访谈。

**💡 创新点**

提出全过程、跨角色、可操作且经过实证验证的AI保障流程，区别于仅关注技术输出的审计，填补了标准化第三方保障缺口。

**🔧 技术方法**

采用访谈法、主题分析、成熟度评估矩阵、责任分配矩阵等定性评估工具，并基于NIST AI风险管理框架与行业最佳实践构建框架。

**📊 数据集**

未使用公开数据集，而是收集了两套内部AI系统的技术文档、代码、模型及相关访谈资料。

**📈 对比分析**

通过对比两系统的成熟度评分、发现问题数量与类别以及专家对易用性与有效性的评估，显示框架能识别问题且被认为易用有效，但未给出定量性能指标。

**⚠️ 局限性**

样本规模小、缺乏长期纵向验证、对外部系统产出获取受限、需进一步调整与更新以适应快速演进的AI环境。

---

## 152. FedDis: A Causal Disentanglement Framework for Federated Traffic Prediction

**arXiv ID:** 2601.22578 | [PDF](https://arxiv.org/pdf/2601.22578v1)

**作者:** Chengyang Zhou `[一作]` (Jilin University), Juncheng Hu `[通讯]` (Jilin University)

**通讯引用:** 6843 | [OpenAlex ID](https://openalex.org/A5014477635)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种面向联邦交通预测的因果分离框架 FedDis，利用双分支结构分别学习个性化本地动态模式和跨客户端共享的全局空间‑时间模式，并通过互信息最小化实现两者信息正交。

**💡 创新点**

创新点在于：①首次在联邦空间‑时间预测中引入因果分离思想，将全局与个性化模式显式解耦；②设计了可共享的全局模式库和本地模式库，以及协同模式共享与图注意力融合机制；③通过互信息约束提高分离质量，增强模型在非 IID 条件下的鲁棒性与可解释性。

**🔧 技术方法**

主要技术包括：Adaptive Graph Convolutional Recurrent (AGR) 编码器、双分支因果表示分离模块、模式库检索与注意力融合、CLUB 互信息上界、协同模式共享与图注意力聚合、联邦优化框架（只上传共享分支参数、全局模式库）。

**📊 数据集**

使用四个真实交通数据集：METR‑LA、PEMS‑BAY、PEMS03、PEMS04，均为5‑分钟间隔的交通速度/流量时间序列。

**📈 对比分析**

与集中式基准（GWNet、AGCRN）以及联邦基准（FedAvg、FedProx、STDN、FedGTP、pFedCTP）对比，FedDis 在所有四个数据集上均取得最低或接近最低的 MAE、RMSE、MAPE，尤其在非 IID 客户端多样性下表现出显著优势；在计算效率上，FedDis 仅略高于 FedAvg/FedProx，远低于 FedGTP。

**⚠️ 局限性**

局限性包括：①在极端稀疏或高异质性数据集上仍可能无法完全赶上集中式模型；②需要额外的模式库维护和互信息计算，增加模型复杂度；③对模式数、K 值等超参数较为敏感，需经验调优；④仅在交通预测场景验证，跨域推广需进一步评估。

---

## 153. Specialists or Generalists? Multi-Agent and Single-Agent LLMs for Essay Grading

**arXiv ID:** 2601.22386 | [PDF](https://arxiv.org/pdf/2601.22386v1)

**作者:** Jamiu Adekunle Idowu `[一作]` (Sahel AI), Ahmed Almasoud `[通讯]` (Prince Sultan University)

**通讯引用:** 337 | [OpenAlex ID](https://openalex.org/A5049069930)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过在 ASAP 2.0 语料上对比单体 LLM 与多体 LLM，探究不同架构对作文质量级别的影响，系统评估其自动评分性能。

**💡 创新点**

创新点在于提出基于内容、结构、语言三大专门代理与主席代理裁决规则的多体架构，并证明少样本校准对两种架构同样关键，揭示不同架构在低分与中分段的互补优势。

**🔧 技术方法**

使用 GPT‑5.1 大语言模型，结合零样本与两例/分数级别的少样本提示，构建单体与多体评分管道，分别采用规则化裁决与平均融合两种策略。

**📊 数据集**

实验数据来源于公开的 ASAP 2.0 语料库，选取 450 篇论述性作文作为测试集，并使用 12 篇校准样本（每个分数级别两例）进行少样本提示。

**📈 对比分析**

通过四种设置（单体/多体，零样本/少样本）进行 Quadratic Weighted Kappa 与准确率比较，少样本提示使 QWK 提升约26%，多体在低分段更优，单体在中分段略优，总体多体 QWK 达到 0.7453，单体 QWK 为 0.7165。

**⚠️ 局限性**

局限主要包括：仅在单一数据集上验证、对高分作文预测偏低、未系统评估多体解释质量、以及多体体系导致的四倍计算成本。

---

## 154. Lantern: A Minimalist Robotic Object Platform

**arXiv ID:** 2601.22381 | [PDF](https://arxiv.org/pdf/2601.22381v1)

**作者:** Victor Nikhil Antony `[一作]` (Johns Hopkins University), Chien-Ming Huang `[通讯]` (Johns Hopkins University)

**通讯引用:** 3202 | [OpenAlex ID](https://openalex.org/A5017287995)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

设计并开源了一款低成本、可握持的极简机器人平台——Lantern，支持可编程触觉和振动反馈，用于多种人机交互实验

**💡 创新点**

将机器人对象概念与可编程振动、压缩式触觉反馈结合，提供可扩展的开源硬件与软件框架，突破传统封闭式机器人对象的局限

**🔧 技术方法**

采用Raspberry Pi Pico W控制单元，5V舵机+带绳机构实现“呼吸”式伸缩，迷你振动马达产生触觉反馈；软件上提供microPython SDK与microROS+ROS2接口

**📊 数据集**

未使用公开数据集，主要通过设计研讨、案例研究、实验室共享、课程实验及公开展览收集定性用户交互数据

**📈 对比分析**

与其他现有机器人平台（如Yolo、Blossom、Flexi等）在成本、可握持性、触觉反馈等指标对比，Lantern在成本（≈40 USD）与可握持、触觉功能上表现突出；缺乏量化性能基准或统计显著性分析

**⚠️ 局限性**

研究范围有限：样本量小、缺乏对照实验、长期使用可靠性未评估，平台对硬件/编程门槛仍有一定要求，未来需扩大样本、深入实地部署并完善可视化编程与低门槛构建工具

---

## 155. Leveraging Data to Say No: Memory Augmented Plug-and-Play Selective Prediction

**arXiv ID:** 2601.22570 | [PDF](https://arxiv.org/pdf/2601.22570v1)

**作者:** Aditya Sarkar `[一作]` (University of Maryland), Nuno Vasconcelos `[通讯]` (University of California)

**通讯引用:** 32208 | [OpenAlex ID](https://openalex.org/A5043325212)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出一种基于外部视觉‑语言表示模型和检索数据集的无训练记忆增强式选择性预测方法

**💡 创新点**

在开放词汇场景下实现轻量化、无训练的选择性预测，解决了CLIP‑style表示不稳定和相似度校准差的问题

**🔧 技术方法**

使用外部CLIP/SigLIP等视觉‑语言表示模型、检索增强代理嵌入、对比归一化分数以及基于硬负样本的对比评分

**📊 数据集**

在MS‑COCO、Flickr‑30K、Flowers、Pets、UCF‑101、SugarCrepe、Winoground、What'sUp、VL‑Checklist、Foil等视觉‑语言数据集，以及医学影像数据集CheXpert、MIMIC‑CXR

**📈 对比分析**

与VQAScore、SeeTRUE等LLM基准对比，AURC大幅下降（15%–30%提升），在所有任务上均优于大型LVLM基线；对比不同检索集与模型尺寸验证了方法稳健性

**⚠️ 局限性**

性能依赖于检索集与目标域的覆盖，缺乏足够域内样本时效果下降；对高维文本生成任务仍需更细粒度的负样本构造与校准技巧

---

## 156. COL-Trees: Efficient Hierarchical Object Search in Road Networks

**arXiv ID:** 2601.22183 | [PDF](https://arxiv.org/pdf/2601.22183v1)

**作者:** Tenindra Abeywickrama `[一作]` (Center for Computational Science), Sabine Storandt `[通讯]` (University of Konstanz)

**通讯引用:** 965 | [OpenAlex ID](https://openalex.org/A5027517542)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了 COL-Tree 索引，用于在道路网络上高效处理 AkNN、kFN 和范围查询等对象搜索问题。

**💡 创新点**

创新点包括：① 基于多地标的层次子图结构 COL-Tree，可利用凸性保留的聚合距离下界实现更精确的剪枝；② 设计了 SUL-Tree 作为构建加速器，显著降低预处理成本；③ 同时支持下界和上界搜索，使得对 AkNN、kFN 等非“最近”问题也能高效求解。

**🔧 技术方法**

采用多地标差分启发式（LLB/LUB）、递归子图划分、子图顶点排序、受限 Dijkstra 搜索等技术，并与 PHL 路网距离索引解耦。

**📊 数据集**

使用美国大陆路网（约 2.39 亿顶点、5.77 亿边）及其 8 种 POI 集（学校、公园、快餐等），以及按密度随机生成的合成 POI 集。

**📈 对比分析**

与 IER、NVD、R-tree、PHL 等现有方法对比：在 AkNN、kFN 查询中 COL-Tree 提升 10–1000 倍（最多 4 位数级）性能，范围查询同样可匹敌并在大规模数据集上提升 2–3 倍；预处理时间仅为 NVD 的 1/1500，空间与 R‑tree 相当。

**⚠️ 局限性**

局限性：多地标选择仍需经验调优；在 POI 动态变化时需要重建索引；主要针对道路网络，对非路网图或高维空间的适用性尚未验证。

---

## 157. MetaLead: A Comprehensive Human-Curated Leaderboard Dataset for Transparent Reporting of Machine Learning Experiments

**arXiv ID:** 2601.22420 | [PDF](https://arxiv.org/pdf/2601.22420v1)

**作者:** Roelien C. Timmer `[一作]` (CSIRO Data61), Stephen Wan `[通讯]` (CSIRO Data61)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

创建了MetaLead数据集，该数据集为机器学习论文的全量实验结果（包括基线、主方法与其变体）进行人工标注，并在每条记录中加入实验类型、训练集与测试集等扩展元信息；同时对该数据集进行了零样本抽取模型的基准评测。

**💡 创新点**

创新点在于：①不再只收集最佳结果，而是完整收集所有公布的实验结果；②在数据结构中显式区分实验类型与训练/测试数据集，支持实验类型驱动对比与跨域评估；③采用完全人工标注而非社区或自动化方式，保证标注质量；④提供统一的丰富元数据schema，可用于更细粒度的后续分析与可定制leaderboard构建。

**🔧 技术方法**

技术手段包括：手工双人标注并制定决策树，利用多种大型语言模型（GPT‑4.1、GPT‑4o、o4‑mini、Gemini‑2.5‑Pro/Flash、Gemini‑1.5‑Pro）进行零样本抽取；与开源模型（Llama 3.3 70B、Mistral Large）以及现有基线（arxiv2024‑leaderboard‑generation、axcell）进行对比；使用自定义的实体/元组评估指标、实验类型准确率及leaderboard‑级别指标（Recall、Coverage、Overlap）。

**📊 数据集**

数据集为MetaLead本身，包含43篇机器学习论文、3,568条实验结果元组；对标数据集为SciLead、Papers with Code、NLP Progress等。

**📈 对比分析**

在闭域（已知实体列表）与开域（全文本抽取）两种设置下，GPT‑4.1与Gemini‑2.5‑Pro在元组抽取上表现最佳（闭域F1≈0.9，开域F1≈0.6）；实验类型分类准确率约70%（GPT‑4.1），基线方法可达≈82%；但leaderboard recall仅在闭域下达49%（最多），说明完整leaderboard重建仍具挑战。整体来看，虽然LLM抽取已显著优于传统基线，但仍存在较高误差，特别是实体组合与跨域信息的完整性。

**⚠️ 局限性**

局限性包括：论文数量仅43篇，覆盖范围有限；仅关注机器学习领域且仅英文；未能细化到更细粒度的评分与实验细节（注：此类标注一致性低）；未对图表中的结果进行抽取；人工标注成本高；抽取方法仍存在低召回与错误匹配，需进一步改进。

---

## 158. One Ring to Rule Them All: Unifying Group-Based RL via Dynamic Power-Mean Geometry

**arXiv ID:** 2601.22521 | [PDF](https://arxiv.org/pdf/2601.22521v1)

**作者:** Weisong Zhao `[一作]` (Institute of Information Engineering, Chinese Academy of Sciences), Xu Zhou `[通讯]` (Sangfor Technologies)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种动态聚合几何的强化学习框架PMPO，能够根据轨迹的稳定性自适应调整聚合指数p，从而在数学推理任务中实现更好的收敛稳定性与效果

**💡 创新点**

通过引入基于剪切率的有效样本量(ESS)匹配机制，使聚合指数p成为一个可调的“温度”参数，统一并推广了GRPO与GMPO的几何，解决了固定几何导致的鲁棒性与效率冲突

**🔧 技术方法**

使用功率均值(Power-Mean)聚合、log域剪切、softmax注意力梯度分析、ESS匹配求解、PPO风格优化以及大规模LLM（Qwen2.5-Math、DeepSeek-R1-Distill-Qwen）在线RL训练

**📊 数据集**

在数学推理基准集AIME24、AMC、MATH500、Minerva、OlympiadBench上进行评测，训练使用MATH Levels 3–5数据集

**📈 对比分析**

与GRPO、GMPO、Dr.GRPO、Prime-Zero、GPG等基线对比，PMPO在7B规模模型上平均准确率提升至54.2%（相较GMPO 52.7%），在1.5B规模也实现44.7%平均准确率，显著超过现有在线RLHF方法

**⚠️ 局限性**

对极端p值或阈值敏感；过高或过低的剪切阈值会导致过于保守或过度激进；在超大规模模型或非数学推理任务中需进一步验证鲁棒性

---

## 159. Knowledge-Informed Kernel State Reconstruction for Interpretable Dynamical System Discovery

**arXiv ID:** 2601.22328 | [PDF](https://arxiv.org/pdf/2601.22328v1)

**作者:** Luca Muscarnera `[一作]` (University of Cambridge), Mihaela van der Schaar `[通讯]` (University of Cambridge)

**通讯引用:** 22342 | [OpenAlex ID](https://openalex.org/A5012339002)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种基于知识驱动的核方法（KSR）用于在部分可观测、含噪声数据中重构系统状态及其导数，并将其作为输入进行符号回归以发现动力学方程

**💡 创新点**

将物理先验（非负、守恒、结构约束）直接嵌入核回归目标，兼顾异构观测（稀疏直接测量与密集聚合测量），并通过解析导数避免数值差分噪声积累

**🔧 技术方法**

使用可再生核希尔伯特空间（RKHS）核回归、软约束正则化、解析导数计算以及后续的稀疏符号回归（SINDy、PySR）

**📊 数据集**

共12个仿真/真实数据集，涵盖药理学（QSP）、流行病学（SEIR）、生理学（胰岛素‑葡萄糖）、经济学等领域

**📈 对比分析**

与多种基线（立方样条、RBF插值、GP、Kalman、Neural ODE、PINN 等）比较，KSR 在噪声下的状态重建 MSE 通常低于 0.02，符号回归识别率最高，鲁棒性显著优于传统方法

**⚠️ 局限性**

对极度稀疏或信息不足的观测不易唯一确定状态；需预先给出状态分层结构；核方法计算复杂度随样本数增长至 O(N^2) 或更高；若先验错误会导致系统偏差

---

## 160. Are LLM Evaluators Really Narcissists? Sanity Checking Self-Preference Evaluations

**arXiv ID:** 2601.22548 | [PDF](https://arxiv.org/pdf/2601.22548v1)

**作者:** Dani Roytburg `[一作]` (Carnegie Mellon University), Narmeen Oozeer `[通讯]` (Martian Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了大型语言模型评判者的自我偏好，提出评估质量基线以消除评估不确定性对自我偏好的影响。

**💡 创新点**

提出了Evaluator Quality Baseline，通过与能力匹配的代理模型对比来区分自我偏好与评估噪声。

**🔧 技术方法**

采用配对t检验、oracle标签、熵分析等统计方法，并构建结果匹配代理选择。

**📊 数据集**

在MATH500、MBPP+、MMLU、AlpacaEval、TruthfulQA、翻译与总结等多项公开数据集上进行实验。

**📈 对比分析**

与原始自我偏好测量对比，发现约89.6%的偏差归因于评估不确定性，纠正后多数模型自我偏好显著下降。

**⚠️ 局限性**

依赖模型生成的oracle标签和代理构造可能未完全匹配能力，对主观任务的评估仍有限。

---

## 161. FIRE: Multi-fidelity Regression with Distribution-conditioned In-context Learning using Tabular Foundation Models

**arXiv ID:** 2601.22371 | [PDF](https://arxiv.org/pdf/2601.22371v1)

**作者:** Rosen Ting-Ying Yu `[一作]` (Massachusetts Institute of Technology), Faez Ahmed `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 12132 | [OpenAlex ID](https://openalex.org/A5026634347)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种训练无须的多分辨率回归框架FIRE，利用Tabular Foundation Models实现零样本推断，并通过低分辨率模型的均值、方差、分位数构造分布条件残差学习，实现高阶逼近；

**💡 创新点**

创新点在于（1）引入分布条件的残差转移，利用低分辨率模型的分布信息（均值、方差、分位数）捕捉异方差误差；（2）将多源低分辨率合并为单一基模型，支持非嵌套数据；（3）在推理阶段直接使用TFM完成零训练的贝叶斯推断；

**🔧 技术方法**

使用的技术包括Tabular Foundation Models（TabPFN）、in-context learning、分布条件残差学习、二层分辨率表示以及零样本贝叶斯推断；

**📊 数据集**

使用了31个多分辨率基准数据集，涵盖18个合成任务、6个HPO任务以及7个工程/物理仿真任务（如DrivAerNet、Blended Wing Body等）；

**📈 对比分析**

通过与7个SOTA GP/深度学习基线在非嵌套、极端高分辨率稀缺（2‑5%）场景下进行10折交叉验证比较；FIRE在NRMSE、NLL上获得最高Elo评分、最低排名，并在运行时显著更快，主导Pareto前沿；

**⚠️ 局限性**

主要限制在于受TFM上下文窗口、模型尺寸、预训练质量的影响；Transformer注意力的二次复杂度限制大规模推理速度；并且对低分辨率数据分布与预训练任务相近度要求较高。

---

## 162. FITMM: Adaptive Frequency-Aware Multimodal Recommendation via Information-Theoretic Representation Learning

**arXiv ID:** 2601.22498 | [PDF](https://arxiv.org/pdf/2601.22498v1)

**作者:** Wei Yang `[一作]` (Kuaishou Technology), Peng Jiang `[通讯]` (Kuaishou Technology)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于频域信息瓶颈的多模态推荐框架FITMM，先对视觉、文本等模态进行频谱分解，再进行频域特征融合并通过信息瓶颈约束提升泛化能力。

**💡 创新点**

创新点在于：①将多模态融合转到频域，利用正交变换实现各频段协方差块对角化；②在频域引入信息瓶颈正则化，按频段分配容量，实现类似Wiener滤波的自适应降噪；③通过跨模态频域一致性损失保证不同模态在同一频段内对齐。

**🔧 技术方法**

使用的技术包括：图卷积网络（GNN）进行图增强表征；正交/紧帧变换（如SVD、图小波）实现频谱分解；信息瓶颈（Gaussian IB）和逆水位填充（reverse water‑filling）调控频段容量；频域门控自适应融合；交叉模态对比损失。

**📊 数据集**

在Amazon Review（Baby、Sports、Clothing）三个亚马逊商品子集上进行实验，采用5‑core过滤，构建图和多模态特征。

**📈 对比分析**

与多种SOTA基线（LightGCN、VBPR、MMGCN、SMORE、MMIL等）以及冷启动模型进行比较，FITMM在Recall@10/20和NDCG@10/20上均实现显著提升，冷启动场景下表现尤为突出。

**⚠️ 局限性**

局限性包括：频谱分解需要额外计算开销；模型对频段数和信息瓶颈权重敏感；仅在静态图和单模态组合上验证，未覆盖视频、音频等更复杂模态；未探索用户个性化频段选择等方向。

---

## 163. Large Language Models: A Mathematical Formulation

**arXiv ID:** 2601.22170 | [PDF](https://arxiv.org/pdf/2601.22170v1)

**作者:** Ricardo Baptista `[一作]` (University of Toronto), Son Tran `[通讯]` (Amazon)

**通讯引用:** 2256 | [OpenAlex ID](https://openalex.org/A5008915336)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文概述了大型语言模型（LLM）的原理、训练方法、评估框架及其在多模态、推理、翻译、摘要等任务中的应用与挑战。

**💡 创新点**

创新点在于将概率下一个词预测的理论与实际应用、评估指标、以及与扩散模型的潜在替代方案系统化整合，提出基于KL、MMD及评分规则的评估度量，并讨论了在逻辑一致性和多模态映射中的新技术。

**🔧 技术方法**

采用自回归概率建模、上下文提示（ICL）、微调（Fine‑tuning）、链式推理（CoT）以及离散扩散模型等技术。

**📊 数据集**

利用公开的大规模文本数据集（如Common Crawl、Wikipedia）、图像文本对数据集以及多语种翻译语料，结合自构建的评估集。

**📈 对比分析**

通过计算交叉熵、困惑度、MMD等指标与基准模型比较，展示了在数学、编程、翻译、VQA等任务上的性能提升，但仍需更多量化评测。

**⚠️ 局限性**

主要限制包括缺乏硬性逻辑约束导致推理错误、长文本生成受限、跨模态映射误差，以及对实时检索与最新知识的依赖不足。

---

## 164. FlowSymm: Physics Aware, Symmetry Preserving Graph Attention for Network Flow Completion

**arXiv ID:** 2601.22317 | [PDF](https://arxiv.org/pdf/2601.22317v1)

**作者:** Ege Demirci `[一作]` (University of California Santa Barbara), Ambuj Singh `[通讯]` (University of California Santa Barbara)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种新型网络流缺失值补全框架FlowSymm，利用离散流的群对称性与图注意力相结合，保证流量守恒并高精度恢复缺失边的流量。

**💡 创新点**

创新点包括：①将可行的无散度调整视为Abelian群作用，构造可列化的基向量；②用GATv2学习上下文敏感的注意力权重来选择合适的群动作；③采用特征条件的Tikhonov微调，并通过隐式二阶优化实现端到端训练。

**🔧 技术方法**

使用的技术包括：图注意力网络（GATv2）、线性代数（伪逆、SVD、投影）、Tikhonov正则化、隐式微分的二阶优化、Cholesky分解、以及传统的图卷积与物理约束方法。

**📊 数据集**

实验数据集涵盖三大真实场景：交通网络（洛杉矶县道路网，约7500条边）、电力输电网（欧洲大陆输电网，约2700条边）以及自行车共享网络（Citi Bike，约2600条边）。

**📈 对比分析**

与九种基线（纯物理、纯数据、混合、以及Bilevel Diagonal Regularizer等）对比，FlowSymm在RMSE、MAE和Pearson相关系数上均取得最佳或最优近的成绩，在最坏情况下比先前最优方法提高约8–10% RMSE、16% MAE，相关系数提升约0.04。

**⚠️ 局限性**

主要局限包括：仅处理单一时间点的静态图；基数k的设置对大规模网络可能不够；对传感器布置的随机化依赖较大；以及整体Bilevel训练非凸，易陷入局部最优。

---

## 165. CARE: Multi-Task Pretraining for Latent Continuous Action Representation in Robot Control

**arXiv ID:** 2601.22467 | [PDF](https://arxiv.org/pdf/2601.22467v1)

**作者:** Jiaqi Shi `[一作]`, Jianzong Wang `[通讯]` (Ping An Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了 CARE 框架，在无动作标签的预训练阶段通过多任务学习获得连续潜在动作表示，并在少量动作标签上微调实现机器人控制。

**💡 创新点**

①将潜在动作模型嵌入 VLM 预训练流程，减少训练步骤；②利用帧预测与关键点轨迹预测的多任务学习，显式编码动作信息；③在无监督条件下避免快捷学习。

**🔧 技术方法**

使用视觉语言模型（Prismatic‑7B + Llama）、SigLIP/DinoV2 视觉编码器、交叉注意力、多任务不确定性加权损失、关键点跟踪（Co‑Tracker）以及 LoRA 微调技术。

**📊 数据集**

240k 视频‑文本对（Open X‑Embodiment 140k 机器人轨迹 + Something‑Something v2 100k 人类日常视频）用于预训练；微调使用 3% RT‑1 机器人动作标签；评测采用 LIBERO benchmark 四套任务。

**📈 对比分析**

与 OpenVLA、Octo、Diffusion Policy、MDT、LAPA、CoMo 等基线相比，CARE 在无标签预训练下取得 77.7% 的整体成功率，超过所有无标签方法，且接近或优于部分使用标签预训练的方法，展示出更高的成功率、解释性和更低的快捷学习指标。

**⚠️ 局限性**

与完全使用动作标签预训练的模型相比仍存在性能差距，特别是 Goal 与 Object 任务；潜在动作表达在离散空间上有限；需要更多多模态数据来进一步提升效果。

---

## 166. PriviSense: A Frida-Based Framework for Multi-Sensor Spoofing on Android

**arXiv ID:** 2601.22414 | [PDF](https://arxiv.org/pdf/2601.22414v1)

**作者:** Ibrahim Khalilov `[一作]` (Johns Hopkins University), Yaxing Yao `[通讯]` (Johns Hopkins University)

**通讯引用:** 86 | [OpenAlex ID](https://openalex.org/A5057516581)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `9cc9baba-5356-466d-81ff-d80028d90279` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出并实现了PriviSense，一套基于Frida的轻量级工具，可在已root的Android设备上实时、可脚本化地伪造传感器（如加速度计、陀螺仪、步数计）和系统信息（如电量、系统时间、设备元数据）

**💡 创新点**

突破了以往仅支持模拟器或需要改写应用的局限，实现了在物理设备上可重现的、无需外部宿主机的多传感器实时伪造；同时提供脚本化、可逆的控制接口，适用于软件测试与行为审计两大场景

**🔧 技术方法**

使用Frida动态插桩在Android运行时拦截并替换系统/传感器API返回值；结合Termux在设备上构建Python/Frida运行环境；通过JavaScript脚本实现对各类API的覆盖与注入

**📊 数据集**

本工作未使用传统意义上的数据集，而是在真实设备上对五款代表性传感器可视化应用进行演示测试

**📈 对比分析**

与传统模拟器或静态伪造工具相比，PriviSense在真实硬件上保持了传感器时序、噪声与功耗等物理特性；实验视频中显示多应用对伪造值的即时响应，表明方法在可重现性与覆盖率上优于现有方案，但未给出量化性能指标

**⚠️ 局限性**

需要设备已root且Frida支持，且部分应用可能通过反调试或安全检查检测到Frida导致崩溃或行为异常；对安全隐私的潜在误用需严格限制与审核

---

## 167. Prepare Reasoning Language Models for Multi-Agent Debate with Self-Debate Reinforcement Learning

**arXiv ID:** 2601.22297 | [PDF](https://arxiv.org/pdf/2601.22297v1)

**作者:** Chenxi Liu `[一作]` (University of Maryland), Heng Huang `[通讯]` (University of Maryland)

**通讯引用:** 24708 | [OpenAlex ID](https://openalex.org/A5060016795)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Self-Debate Reinforcement Learning (SDRL)，通过在训练时构造多样化的辩论情景，联合优化单次推理和辩论条件下的回答，使同一模型既能在单体推理中表现优异，又能在多智能体辩论中高效合作。

**💡 创新点**

创新点在于：①将单体推理与辩论学习统一到同一 RL 目标中，避免传统方法训练后无法适应多智能体环境的“训练-推理不匹配”；②利用可验证奖励进行强化学习；③通过随机或频率匹配两种辩论对构造方式，强制模型学习“私有批评”（private critique）以识别并改进不同思路；④对比理论分析揭示私有批评是提升辩论性能的关键。

**🔧 技术方法**

技术细节包括：GRPO + DAPO 的可验证奖励强化学习框架；自对话（self‑debate）构造；辩论对采样策略（随机 pairing、frequency‑based pairing）；多轮/多代理的 MAD 模型；对齐奖励与优势估计。

**📊 数据集**

训练使用 DAPO‑Math‑17K；评测采用数学推理基准：MATH500、AMC 2023、AIME 2024/2025；在这些基准上测试单体推理（mean@K、maj@K）与多智能体辩论（Maj、Debate、Δ）。

**📈 对比分析**

与传统 DAPO 基线在分散式 MAD、稀疏 MAD 和集中式 MAD 等不同框架下对比。SDRL 在所有基准上均实现了显著提升：多智能体辩论准确率提高 1–4%（Δ 上升 0.7–3.5%），单体推理准确率亦提升 0.5–1.5%。特别是难题 AIME24/25 上，辩论准确率提升超过 7%。

**⚠️ 局限性**

局限性：①随着代理数量增加，模型在保持完整回答长度方面受限，导致性能下降；②在多轮辩论后期，准确率趋于递减，需进一步调节模型的“批评强度”；③目前仅验证于数学推理任务，是否能推广到其他领域仍待探究；④训练成本较高，需多次采样和奖励评估。

---

## 168. Matrix Factorization for Practical Continual Mean Estimation Under User-Level Differential Privacy

**arXiv ID:** 2601.22320 | [PDF](https://arxiv.org/pdf/2601.22320v1)

**作者:** Nikita P. Kalinin `[一作]` (Institute of Science and Technology Austria), Christoph H. Lampert `[通讯]` (Institute of Science and Technology Austria)

**通讯引用:** 12641 | [OpenAlex ID](https://openalex.org/A5068085751)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究在用户级差分隐私（ULDP）下的持续平均值估计问题，提出一种针对该任务的矩阵分解方法，显著降低均方根误差（RMSE）。

**💡 创新点**

创新点在于设计了专门针对平均值估计的 Toeplitz 矩阵分解（D_Toep 及其 banded‑inverse 版本），利用 Gregory 系数实现更小的敏感度和更低的噪声；该分解在理论上可实现渐进最优（k=1）且在实际中优于现有分解。

**🔧 技术方法**

核心技术包括：矩阵分解机制（Matrix Factorization）+ 高斯机制；对分解矩阵的敏感度分析（b‑min 分离条件、Toeplitz 结构、Gregory 系数）；banded‑inverse 变体以降低存储成本；以及与指数扣留（exponential withholding）和子高斯浓缩结合的隐私加噪方案。

**📊 数据集**

在实验中使用了合成数据（高维均匀/高斯分布）和真实的 Credit Card Transactions 数据集（约 1000 名用户，6 个月交易记录）。

**📈 对比分析**

与多种基线（BandMF、E_1^{1/2}、I、ν‑DP‑FTRL、纯 DP Binary Tree）对比，实验结果显示 D_Toep 在大多数时间步的 RMSE 明显低于对手，尤其在早期更新时误差更小；在多次参与（k>1）时，banded 版本进一步提升，整体性能优于现有方法。

**⚠️ 局限性**

主要局限包括：需要满足 b‑min 分离（用户参与间隔）和基于分布假设（有界或子高斯）才能得到理论保证；在多次参与大 k 时，理论最优性不再保持；常数与对数项在实际中可能占主导，影响小规模或低隐私预算场景；实现中需预先计算 Gregory 系数，增加复杂度。

---

## 169. Coordinating Power Grid Frequency Regulation Service with Data Center Load Flexibility

**arXiv ID:** 2601.22487 | [PDF](https://arxiv.org/pdf/2601.22487v1)

**作者:** Ali Jahanshahi `[一作]` (University of California), Daniel Wong `[通讯]` (University of California)

**通讯引用:** 2916 | [OpenAlex ID](https://openalex.org/A5108142825)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究如何让 AI/ML GPU 数据中心通过参与电网频率调节服务，降低对化石燃料调节储备的需求，从而实现对电网侧碳排放的“外源碳”减排。

**💡 创新点**

提出了 Exogenous Carbon 指标量化数据中心调节服务带来的电网侧碳减排，并设计了 EcoCenter 框架，利用 GPU 多核、功率上限与核心分配多维度调控，实现高质量 2 秒级频率调节与大幅度调节容量。

**🔧 技术方法**

核心技术包括：GPU 动态电压频率与核心掩码控制、跨 GPU 协同功率重塑策略、基于 ISO 规则的调度优化（含不对称上下调节）、基于单位承诺模型的网格侧碳排放计算与 MCE_resv 获得、以及 Z3 约束求解器的实时优化。

**📊 数据集**

实验数据集主要来自 Facebook SWIM 负载跟踪（多种平均负载与方差组合）、PJM 频率调节信号（Extreme、High Transition、Noisy）以及 CAISO 2022 年电价与调节奖励数据。

**📈 对比分析**

与 CPU 仅调节、GPU 仅单 GPU 方案及 UPS 调节方案对比，EcoCenter 在 10%–80% 负载下的调节容量提升 30–50% 以上；性能评分均在 85% 以上；在 50% 负载时，其年碳减排达 433.2 kt CO₂eq（相较于 CPU 方案 132 kt），TCO 在低电价环境下可降低 10–20%。

**⚠️ 局限性**

局限性包括：需在低负载下才可提供充足调节容量；调节导致 BE 任务吞吐下降，产生机会成本；GPU 低功率状态受限，无法充分利用 60W 的动态范围；模型假设单一数据中心规模与功率上限，未覆盖多租户与多站点协同情况。

---

## 170. Graph is a Substrate Across Data Modalities

**arXiv ID:** 2601.22384 | [PDF](https://arxiv.org/pdf/2601.22384v1)

**作者:** Ziming Li `[一作]` (University of Connecticut), Chuxu Zhang `[通讯]` (University of Connecticut)

**通讯引用:** 5396 | [OpenAlex ID](https://openalex.org/A5022275632)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

设计并实现了 G-Substrate 框架，将图结构视为可持续的中间表示，统一跨模态、跨任务的图空间，并通过交叉角色（生成-理解）训练实现图结构的可重用与累积。

**💡 创新点**

创新点包括：1）将图结构从任务局部转化为跨任务共享的底层子结构；2）构建统一的结构 schema 以实现跨模态的兼容；3）提出交叉角色互联训练策略，使同一图在不同功能角色中被共享，从而提升结构的泛化和利用效率。

**🔧 技术方法**

使用技术：统一结构 schema、交叉角色训练策略（生成、理解交替）、多任务学习框架（以 Qwen3‑VL‑2B‑Instruct 为 backbone），结合结构一致性、子图检索、图算法监督等多种任务目标。

**📊 数据集**

使用的数据集包括：Visual Genome（场景图生成）、MAVEN（事件关系抽取）、Mol‑Instructions（分子描述生成）、Graph Algorithmic Reasoning 数据集（连通性、环检测、最短路、二分匹配）等多域多模态数据。

**📈 对比分析**

与单任务训练、任务专用模型、以及传统 multi‑task（仅参数共享）进行对比。G-Substrate 在多数任务上均优于传统方法，平均提升约 2‑5% 以上，并在某些任务上接近或超越专门化模型，证明统一结构与交叉角色训练的有效性。

**⚠️ 局限性**

局限性：未探究不同模态/角色比例的最优平衡；对数据偏差与结构偏见传播缺乏系统控制；仅聚焦图结构，未扩展到其他形式的中间结构；在极大规模任务或多样性更高的场景下，训练成本与复杂度待进一步评估。

---

## 171. CoDCL: Counterfactual Data Augmentation Contrastive Learning for Continuous-Time Dynamic Network Link Prediction

**arXiv ID:** 2601.22427 | [PDF](https://arxiv.org/pdf/2601.22427v1)

**作者:** Hantong Feng `[一作]` (Southeast University), Wenwu Yu `[通讯]` (Southeast University)

**通讯引用:** 25877 | [OpenAlex ID](https://openalex.org/A5100627758)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种基于反事实数据增强与对比学习的动态网络链接预测框架CoDCL，能够在保持原有结构的前提下，自动生成高质量反事实样本并通过对比学习提升表示学习的鲁棒性。

**💡 创新点**

核心创新在于：①将因果推断视角引入动态网络，设计动态交互指标作为处理变量；②构造局部反事实样本生成策略（k-hop最近邻搜索+相似度约束）；③将生成的反事实样本与真实样本共同参与对比学习，形成一种可插拔的增量模块。

**🔧 技术方法**

采用的技术包括：动态交互指标（基于时间窗口的共同邻居计数）、二值处理变量阈值化、k-hop邻域搜索、节点嵌入的局部加权平均、InfoNCE对比损失、α加权联合损失以及可插拔的动态图网络骨干（如TGAT、GraphMixer、DyGFormer等）。

**📊 数据集**

实验使用七个跨领域真实数据集：Wikipedia、UCI、Enron、MOOC、Reddit、LastFM、CanParl。

**📈 对比分析**

与9种先进动态图学习基线（JODIE、DyRep、TGAT、TGN、TCL、GraphMixer、DyGFormer、FreeDyG、CorDGT）在AP和AUC-ROC指标下进行对比。CoDCL在多数数据集上实现了约0.1%–1.1% 的绝对提升，尤其在CanParl、UCI等具有复杂时序结构的数据集上表现突出。

**⚠️ 局限性**

主要局限包括：①对超参数（p阈值、k_max搜索深度）的敏感性，需要精细调优；②生成反事实样本的搜索过程在高连通网络中可能产生较高计算开销；③目前仅针对链接预测任务验证，未在社区检测、节点分类等其他动态图任务中进行探讨。

---

## 172. Hybrid Cross-Device Localization via Neural Metric Learning and Feature Fusion

**arXiv ID:** 2601.22551 | [PDF](https://arxiv.org/pdf/2601.22551v1)

**作者:** Meixia Lin `[一作]` (ByteDance Inc), Xiao Liu `[通讯]` (ByteDance Inc)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了一套跨设备定位框架，融合传统几何PnP与神经网络MapAnything，利用共享检索编码器和多特征融合实现高召回与高精度。

**💡 创新点**

创新点包括：①统一检索编码器共享，减少重复检索；②多描述子与多匹配器融合提升检索鲁棒性；③神经引导的候选剪枝，利用P3D估计或MapAnything预测过滤大基线帧；④深度条件化MapAnything进一步提升尺度精度；⑤混合PnP+MapAnything姿态融合。

**🔧 技术方法**

采用的技术主要有：MegaLoc检索、SuperPoint / GIM‑SuperPoint / DISK描述子、SuperGlue / LightGlue / GIM‑LightGlue匹配器、COLMAP PnP、MapAnything神经重定位、神经引导剪枝、深度过滤。

**📊 数据集**

使用的数据集为CroCoDL 2025 Challenge中的 HYDRO 与 SUCCU，涵盖 iOS、HL、Spot 等不同设备场景。

**📈 对比分析**

通过在 HYDRO 与 SUCCU 评测中使用 R@0.5 m, 5° 的指标，系统取得 92.62 的总分；多特征融合使召回提升约 3 个百分点，混合PnP+MapAnything方案在精度与鲁棒性上均优于单一方法。

**⚠️ 局限性**

局限性包括：对深度图噪声敏感；在极大基线或极端光照/视角变化下仍可能出现失败；MapAnything 的推断依赖于检索质量，对检索召回率不稳定的情况易受影响。

---

## 173. Towards Resiliency in Large Language Model Serving with KevlarFlow

**arXiv ID:** 2601.22438 | [PDF](https://arxiv.org/pdf/2601.22438v1)

**作者:** Shangshu Qian `[一作]` (Purdue University), Yongle Zhang `[通讯]` (Purdue University)

**通讯引用:** 371 | [OpenAlex ID](https://openalex.org/A5101772866)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一个容错的LLM服务架构，在节点失效时仍能保持请求处理。

**💡 创新点**

1) 动态流量重路由，保持部分模型并行可用；2) 解耦模型并行初始化，支持运行时重构；3) 背景KV缓存复制，避免重启导致的延迟。

**🔧 技术方法**

基于TensorRT‑LLM的PyTorch后端，使用MPICH、NCCL、gRPC、CUDA流等；实现了分布式锁、GPU‑to‑GPU复制等。

**📊 数据集**

使用ShareGPT请求日志模拟负载，评测Llama‑3.1‑8B模型在4阶段流水线并行。

**📈 对比分析**

与TensorRT‑LLM基线和标准故障行为、DejaVu/AnchorTP等进行对比；平均延迟、p99延迟、TTFT提升3.1x/2.8x、TTFT提升378.9x/574.6x，MTTR从10min降至约30s，运行时开销仅2–4%。

**⚠️ 局限性**

依赖节点间均衡负载组与相同模型副本；需要GPU内存余量；对大规模多节点失效或无GPUDirect的环境支持有限；实现复杂且对现有框架集成成本较高。

---

## 174. Word-Centered Semantic Graphs for Interpretable Diachronic Sense Tracking

**arXiv ID:** 2601.22410 | [PDF](https://arxiv.org/pdf/2601.22410v1)

**作者:** Imene Kolli `[一作]` (University of Zurich), Carsten Jentsch `[通讯]` (TU Dortmund University)

**通讯引用:** 820 | [OpenAlex ID](https://openalex.org/A5013581565)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建基于静态和上下文词向量的词中心语义网络，利用周边连通性聚类识别词义社区，并通过节点重叠对社区在不同时期进行对齐，进而跟踪词义演变。

**💡 创新点**

创新点在于：① 将静态 Skip‑gram 相似度与时间特定掩码语言模型的可替换性融合成同一图；② 用周边连通性（去掉中心词后聚类）直接挖掘词义，而不依赖预定义词义表；③ 通过节点重叠实现无监督的跨时序社区对齐，实现可解释的语义变迁可视化。

**🔧 技术方法**

核心技术包括：Skip‑gram 训练得到静态词向量；Fine‑tuned RoBERTa 生成可替换词列表；图网络构建与层级扩展；周边图聚类（连通分量）；节点重叠对齐；社区质量归一化计算词义使用分布。

**📊 数据集**

实验使用 1980–2017 年纽约时报杂志（NYT Magazine）语料库，分别对词汇 "trump"、"god"、"post" 进行时序语义分析。

**📈 对比分析**

本文未进行与现有基准方法的量化对比，仅通过图示和社区分布的可视化展示效果；示例显示词义替换、稳定与渐进关联变迁等多种语义演变模式，展示框架的可解释性与灵活性。

**⚠️ 局限性**

局限性包括：① 仅做应用示例，缺乏对 benchmark 数据集的评估；② 对超参数（k 值、层数等）和预处理敏感，可能影响社区数量与稳定性；③ 语料单一来源，存在编辑与主题偏差，部分观察到的变迁可能是临时性事件导致。

---

## 175. Sparks of Rationality: Do Reasoning LLMs Align with Human Judgment and Choice?

**arXiv ID:** 2601.22329 | [PDF](https://arxiv.org/pdf/2601.22329v1)

**作者:** Ala N. Tak `[一作]` (Institute for Creative Technologies), Jonathan Gratch `[通讯]` (Institute for Creative Technologies)

**通讯引用:** 16559 | [OpenAlex ID](https://openalex.org/A5051992568)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在启用“思考”模式的LLM在符合理性选择公理以及在情感诱导下的行为偏差，构建了一套多域的基准测试，评估不同情感驱动方法（ICL 与 RLS）对决策的影响。

**💡 创新点**

创新点在于：① 将行为经济学与心理学中的经典情感效应（风险偏好、模糊厌恶、损失厌恶、终端效应、时间折现等）与LLM决策紧密结合；② 设计了两种情感驱动机制（ICL 通过上下文情绪提示，RLS 通过低秩向量注入），并对其效果进行对比；③ 使用理性公理检验与可解释的效应量（Hedges' g、概率加权函数等）来度量LLM的行为偏差。

**🔧 技术方法**

技术包括：大规模语言模型推理（启用思考 tokens）、情感驱动方法（ICL 与 RLS）、基于公理的理性检验、参数化决策模型（预加权概率函数、效用曲线、损失厌恶线性判定面）以及统计分析（曲线拟合、效应量计算）。

**📊 数据集**

数据集主要来自：① 行为经济学实验原型（风险、模糊、损失、终端效应、时间折现等）；② 结构化问卷与情景推理任务（刻板印象、框架、道德、法律责任、分配公平、慈善分配等）；③ 论文内部构建的模拟问卷与提示模板，用于统一测试各模型与情感驱动方式。

**📈 对比分析**

比较方法：① 通过四个理性公理（完备性、传递性、连续性、独立性）计算一致率；② 对各情感驱动强度下的决策进行参数化拟合并计算效应量（Hedges' g）和置信区间；③ 采用随机效应荟萃分析得到跨域平均效应。结果显示：启用思考显著提升理性一致率；ICL 在情感效应上更接近人类但幅度往往过大；RLS 产生更温和、可解释的偏差；但在终端效应、模糊厌恶、时间折现等领域仍存在显著偏差。

**⚠️ 局限性**

局限性包括：① 终端效应反向、强模糊厌恶、时间折现异常；② ICP 情感驱动源头显式导致偏差过大，难以校准；③ RLS 效果小且不够稳定；④ 仅评估单人决策，缺乏多智能体、博弈和策略性互动的检验；⑤ 仅使用控制实验情绪诱导，未检验更自然、模糊的情感情境；⑥ 可能忽略模型内部潜在的非可解释偏差与安全风险。

---

## 176. Mitigating Cognitive Inertia in Large Reasoning Models via Latent Spike Steering

**arXiv ID:** 2601.22484 | [PDF](https://arxiv.org/pdf/2601.22484v1)

**作者:** Seojin Lee `[一作]` (Chung Ang University), Hwanhee Lee `[通讯]` (Chung Ang University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 STARS 框架，实时检测大规模推理模型的隐藏状态 L2 “冲击”以识别认知惯性节点，并在发现关键冲击后通过插入状态感知后缀文本引导模型纠正过度推理或路径僵化。

**💡 创新点**

创新点：① 使用隐藏状态的瞬时 L2 距离峰值作为内部转折点信号，区别功能性与临界性 pivot；② 通过几何轨迹诊断（方向翻转、重复度）精准定位脆弱时刻；③ 采用训练无关、轻量级的后缀注入实现实时推理 steering，而非简单早停或硬约束。

**🔧 技术方法**

技术：隐藏状态冲击检测（MAD 适应阈值、SPR 层选择）、方向翻转与循环重复诊断、基于诊断结果的“Shifting Suffix”和“Loop Breaker Suffix”自然语言注入；评估使用层级 Euclidean 距离、余弦相似度等指标。

**📊 数据集**

数据集：MathPerturb、ConditionedMath（评估推理僵化）、AIME24/AIME25（长链推理）、PuzzleTrivial（逻辑推理）与 GPQA-Diamond（学术推理）等多领域基准。

**📈 对比分析**

对比方法：Vanilla、DEER、ConCISE 等无训练干预模型。结果显示 STARS 在所有模型和基准上提升 3–10% 的准确率，同时平均 token 数下降 15–30%，并在精度–效率方面优于现有 token‑level 控制方案；在深层模型上保持较高的推理长度，避免过度截断。

**⚠️ 局限性**

limitations：① 若推理在未出现显著冲击前就完成，STARS 无法介入；② 一些冲击被误判为错误路径，后缀注入可能导致更糟；③ 对极小模型或训练目标平滑的模型，隐藏状态冲击信号弱，STARS 效果受限。

---

## 177. SHED Light on Segmentation for Dense Prediction

**arXiv ID:** 2601.22529 | [PDF](https://arxiv.org/pdf/2601.22529v1)

**作者:** Seung Hyun Lee `[一作]` (University of Michigan), Stella X. Yu `[通讯]` (University of Michigan)

**通讯引用:** 10102 | [OpenAlex ID](https://openalex.org/A5042014034)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6514db3d-8de6-452c-91b7-acdb31787cc4` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种新的编码-解码架构 SHED，通过在密集预测任务中显式加入分割层级，使得模型能够在单张图像中同时进行像素级预测与分割信息推理。

**💡 创新点**

创新点在于：1) 利用分割词元进行双向层级推理，编码阶段进行层级池化，解码阶段进行层级展开，显式强制模型遵循几何先验；2) 只在最终输出进行监督，无需额外的分割标签，分割层级可自发生成；3) 该层级感知解码器能更好地捕捉全局3D场景布局。

**🔧 技术方法**

采用了基于 Transformer 的编码器-解码器框架，结合分割词元（segment tokens）的层级池化与解池化；使用单一的最终监督损失，结合深度边界锐化与语义分割损失；在3D重建与语义分割上进一步利用梯度回传进行端到端训练。

**📊 数据集**

主要使用合成数据集（如 ShapeNet、KITTI synthetic）进行训练，并在真实世界数据集（如 NYU Depth V2、KITTI、ScanNet）上进行评估，以验证跨域泛化能力。

**📈 对比分析**

与传统像素级预测方法（如 UNet、DeepLabV3 等）以及基于几何先验的对比模型相比，SHED 在深度边界清晰度、分割连贯性、语义分割准确率和3D重建精度上均取得显著提升，尤其在从合成到真实场景的迁移实验中表现出更强的鲁棒性。

**⚠️ 局限性**

局限性包括：1) 计算开销相对较大，尤其在大尺寸图像上需要较多显存；2) 仅通过最终监督可能导致低层级细节欠拟合；3) 对极端光照或遮挡条件的鲁棒性尚未充分验证；4) 目前仅在单模态输入上进行实验，未扩展到多模态场景。

---

## 178. Variational Bayesian Flow Network for Graph Generation

**arXiv ID:** 2601.22524 | [PDF](https://arxiv.org/pdf/2601.22524v1)

**作者:** Yida Xiong `[一作]` (Wuhan University), Wenbin Hu `[通讯]` (Wuhan University)

**通讯引用:** 23203 | [OpenAlex ID](https://openalex.org/A5069789783)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `3f18e8e3-0266-457c-8567-9039b6d2394d` `40105733-5154-44cd-8090-a8cab9e64b07` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出Variational Bayesian Flow Network (VBFN)，用于图结构数据的离散生成，利用联合高斯变分推断实现节点与边的协同更新。

**💡 创新点**

创新点在于把经典BFN的独立编码几何提升为结构化精度的联合高斯变分族，利用图的表示依赖关系构造无泄漏的Laplacian精度，从而使贝叶斯更新转化为一次SPD线性系统求解，实现全局节点‑边协同推理。

**🔧 技术方法**

核心技术包括：联合高斯变分推断、结构化精度（Laplacian+稀疏掩码）、贝叶斯更新公式、共轭梯度或Cholesky求解SPD系统、消息匹配的KL目标。

**📊 数据集**

实验数据集涵盖三类合成图（Planar、Tree、SBM）以及两个分子图（QM9、ZINC250k）。

**📈 对比分析**

与传统自回归、GAN、扩散、流模型（GraphRNN、GRAN、SPECTRE、DiGress、EDGE、BwR、BiGG、GraphGen、HSpectre、GruM、CatFlow、DisCo、Cometh、SID、DeFoG、TopBF）进行对比。VBFN在合成图上获得最佳或第二佳的有效率（V.U.N.）与结构距离比（Ratio），在分子图上实现99.98%有效率、0.083 FCD、0.9519 Scaffold相似度，显著优于现有基线。

**⚠️ 局限性**

局限性包括：需要在每一步求解SPD线性系统，计算成本随节点数平方增长；对结构化精度的选择（λ_X、λ_A、Ω_obs）敏感；在高稠密图或大规模网络上可扩展性仍需进一步验证。

---

## 179. Quantum-Inspired Reinforcement Learning for Secure and Sustainable AIoT-Driven Supply Chain Systems

**arXiv ID:** 2601.22339 | [PDF](https://arxiv.org/pdf/2601.22339v1)

**作者:** Muhammad Bilal Akram Dastagir `[一作]` (Qatar Center for Quantum Computing), Ahmed Farouk `[通讯]` (Qatar Center for Quantum Computing)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文设计了一个量子启发式强化学习框架，用于AIoT驱动的供应链，实现库存管理、碳排放惩罚和安全目标的统一决策。

**💡 创新点**

创新点在于将可控自旋链模拟与AIoT实时信号耦合，构建多目标奖励，并通过集成DQN+PPO实现对噪声鲁棒且兼顾可持续性与安全的控制。

**🔧 技术方法**

采用量子自旋链模型、强化学习（DQN、PPO及其集成）、经验回放、熵归一化奖励等技术。

**📊 数据集**

使用基于自旋链的仿真环境与模拟供应链参数（N=3自旋、库存、碳排放等）作为实验数据集。

**📈 对比分析**

与PPO、DQN、ACKTR、DDPG、ACER、GRAPE、MPC等基线对比，提出方法在峰值奖励和后期平均奖励上均优于所有深度RL模型，且优于传统模型。

**⚠️ 局限性**

局限性包括仅在仿真环境验证、理想化噪声假设、难以扩展到更大自旋链或复杂网络、以及多目标奖励在大规模部署时可能过于复杂。

---

## 180. Unrewarded Exploration in Large Language Models Reveals Latent Learning from Psychology

**arXiv ID:** 2601.22474 | [PDF](https://arxiv.org/pdf/2601.22474v1)

**作者:** Jian Xiong `[一作]` (Fudan University), Dejing Dou `[通讯]` (Fudan University)

**通讯引用:** 4919 | [OpenAlex ID](https://openalex.org/A5066063885)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究在大型语言模型中是否存在类似托尔曼的潜在学习（latent learning）现象，并证明在无奖励的探索阶段模型仍能提升表现；随后在引入奖励后进一步提升；

**💡 创新点**

提出两阶段训练策略（先无奖励探索后再奖励学习）及其理论分析，首次将托尔曼心理学概念与LLM训练结合；

**🔧 技术方法**

采用GRPO（Group Relative Policy Optimization）算法进行无奖励探索与奖励学习，使用KL正则化的策略匹配目标；

**📊 数据集**

使用数学推理数据集（GSM8K、MATH、AMC23、AIME24、Minerva Math、OlympiadBench）以及GUI代理数据集（ScreenSpot、ScreenSpot‑V2、ScreenSpot‑Pro、AndroidControll‑Low/High、GUIAct‑Web、OmniAct‑Web/Desktop、GUI‑Odyssey）和SimpleRL‑Zoo‑Data；

**📈 对比分析**

与全奖励训练对比，在数学推理和GUI任务上均显示无奖励阶段即有显著提升，随后奖励阶段进一步提升，部分模型在两阶段训练后超过单阶段奖励训练的性能；

**⚠️ 局限性**

局限性包括：仅在GRPO框架下验证，未知其他RL算法的通用性；实验规模仅至8B参数，未覆盖更大模型；评估任务局限于数学推理与GUI代理，未覆盖开放式对话、代码生成等更广泛场景；

---

## 181. Learning Reward Functions for Cooperative Resilience in Multi-Agent Systems

**arXiv ID:** 2601.22292 | [PDF](https://arxiv.org/pdf/2601.22292v1)

**作者:** Manuela Chacon-Chamorro `[一作]` (Universidad de los Andes), Nicanor Quijano `[通讯]` (Universidad de los Andes)

**通讯引用:** 3385 | [OpenAlex ID](https://openalex.org/A5010955102)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出并验证了一套基于合作弹性指标对轨迹进行排名，然后利用偏好学习的逆强化学习推导奖励函数，并将该奖励函数融入多智能体强化学习训练的框架。

**💡 创新点**

创新点在于：①使用系统级合作弹性度量对轨迹进行排序，从而以行为证据驱动奖励学习；②提出混合奖励策略，将集体弹性奖励与个体消费奖励相结合；③构建可插拔的奖励学习模块，可与现有 MARL 算法（如 PPO、QMIX）协同工作。

**🔧 技术方法**

技术核心包括：偏好学习的逆强化学习（Margin-based 和 Probabilistic 两种实现）、基于失败/恢复曲线的合作弹性指标、线性/手工特征/神经网络三种奖励参数化，以及标准的 MARL 算法（PPO、QMIX）用于策略训练。

**📊 数据集**

实验使用自行设计的社会困境环境（Commons Harvest）——8×8 网格（2 代理）与扩展版 16×16 网格（4 代理）中随机策略产生的 500 条轨迹，用于生成排名和训练奖励。

**📈 对比分析**

通过在三种破坏协议下评估，混合奖励策略在 500 次评估中相较于随机、标准 PPO、QMIX 在以下指标上表现更好：合作弹性显著提升（均值提升至 ~0.89）；平均苹果消费与回合长度提升；“最后一苹果”消耗事件频率仅 13.2%，远低于基线；统计检验表明提升在显著性水平下显著。

**⚠️ 局限性**

局限性包括：①仅在完全可观测、规模较小的环境中验证；②最佳效果依赖手工特征，难以直接推广到更复杂或高维状态空间；③神经网络参数化在样本量不足时易欠拟合；④未与最新的协作 MARL 基线比较；⑤对大规模多代理场景的可扩展性仍待进一步评估。

---

## 182. Label-Efficient Monitoring of Classification Models via Stratified Importance Sampling

**arXiv ID:** 2601.22326 | [PDF](https://arxiv.org/pdf/2601.22326v1)

**作者:** Lupo Marsigli `[一作]` (Amazon), Angel Lopez de Haro `[通讯]` (Amazon)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出并验证了在部署后模型监控中使用分层重要性抽样（SIS）来在有限标签预算下高效估计分类错误率。

**💡 创新点**

创新点在于将SIS推广为通用无模型、无校准假设的监控框架，并给出在有限样本下比传统重要性抽样（IS）和分层随机抽样（SRS）更低均方误差的理论保证。

**🔧 技术方法**

核心技术包括分层采样、重要性加权、比例分配、有限样本MSE分析，以及与自适应抽样方法的对比。

**📊 数据集**

实验使用了六个公开数据集（BCW、Digits、Credit Default、MNIST、CIFAR‑10、专有大规模二分类）以及多种模型（LogReg、CNN、XGBoost）。

**📈 对比分析**

与随机采样、SRS、IS、Adaptive IS/FILA等方法比较时，SIS在绝大多数情形下实现了1.5‑10倍以上的相对效率提升，且在噪声分层或误差率极低的场景中尤为突出。

**⚠️ 局限性**

局限性包括在严重提议分布不匹配或分层信息极弱时无法保证MSE下降，以及不具备自适应更新机制，导致在某些高标签预算或极端偏倚场景下性能不如自适应IS。

---

## 183. ParalESN: Enabling parallel information processing in Reservoir Computing

**arXiv ID:** 2601.22296 | [PDF](https://arxiv.org/pdf/2601.22296v1)

**作者:** Matteo Pinna `[一作]` (University of Pisa), Claudio Gallicchio `[通讯]` (University of Pisa)

**通讯引用:** 3094 | [OpenAlex ID](https://openalex.org/A5011604061)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并实现了 ParallEchostate Network (ParalESN)，一种基于对角线线性递归的无训练 Reservoir Computing 模型，并在多种时间序列与像素级分类任务上进行实验验证。

**💡 创新点**

创新点在于：①利用对角线复数矩阵实现线性递归，可通过 associative‑scan 并行化序列处理；②通过混合卷积层打破对角结构引入非线性交互；③证明该结构保留 Echo State Property 与传统 ESN 的通用性，且可映射任意线性递归 Reservoir；④显著降低内存占用（对角矩阵只需 O(N) 参数）并将时间复杂度从 O(T) 降至 O(log T)。

**🔧 技术方法**

主要技术：Reservoir Computing、Echo State Network、线性对角递归、复数 Eigenvalue 初始化、卷积混合层、MLP 读出、Ridge 回归训练、关联扫描并行化、HiPPO/SSM 相关初始化思想。

**📊 数据集**

实验数据集包括：
- 回归：MemCap、ctXOR、SinMem、Lorenz96、Mackey‑Glass、NARMA、ETTh1/2/TTm1/2；
- 分类：UCR/UEA 时间序列分类、sMNIST、psMNIST。

**📈 对比分析**

与传统浅层 ESN、深层 ESN、全训练 LSTM、Transformer、LRU、Mamba 等模型对比，ParalESN 在绝大多数任务上保持相同或更高的预测/分类精度，同时训练时间、CO₂ 排放和能耗降低 10–100 倍；在 sMNIST/psMNIST 上仅比全训练模型低 1–2% 精度，却节省 80–95% 训练时间与能耗。

**⚠️ 局限性**

局限性：
- 仅使用线性递归，可能在极为复杂的非线性动态任务上不足；
- 读出层必须足够表达（通常为 MLP），否则无法捕获复杂映射；
- 对极长序列仍需注意数值稳定性与复数相位的累计误差；
- 目前实验多基于单机 GPU，跨设备大规模并行部署仍需进一步验证。

---

## 184. Is Hierarchical Quantization Essential for Optimal Reconstruction?

**arXiv ID:** 2601.22244 | [PDF](https://arxiv.org/pdf/2601.22244v1)

**作者:** Shirin Reyhanian `[一作]` (Ruhr University Bochum), Laurenz Wiskott `[通讯]` (Ruhr University Bochum)

**通讯引用:** 10907 | [OpenAlex ID](https://openalex.org/A5039663126)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

对单层与两层层次化VQ‑VAE在匹配表示预算下的重建性能进行对比研究，证明在有效的代码本使用下单层模型可匹敌层次化模型；

**💡 创新点**

首次在保持连续与离散容量相等且代码本坍塌均被轻量化处理的条件下，系统地证明层次化并非提升重建质量的必要因素；

**🔧 技术方法**

使用向量量化自编码器（VQ‑VAE）、EMA代码本更新、straight‑through 估计、代码本初始化与周期性重置、以及对代码本尺寸与维度的网格搜索；

**📊 数据集**

ImageNet 256×256 图像；

**📈 对比分析**

通过对比两种模型在相同训练目标、相同训练调度、相同EMA量化器和相同代码本坍塌缓解措施下的均方误差（MSE）和峰值信噪比（PSNR），发现单层模型在匹配预算和使用稳定的代码本配置时可实现与层次化模型相当的重建质量；

**⚠️ 局限性**

研究聚焦于重建质量，未探讨层次化在生成性或感知质量上的潜在优势，对不同训练策略或更深层次结构的泛化仍有待验证。

---

## 185. Dynamic Welfare-Maximizing Pooled Testing

**arXiv ID:** 2601.22419 | [PDF](https://arxiv.org/pdf/2601.22419v1)

**作者:** Nicholas Lopez `[一作]` (Harvard University), David C. Parkes `[通讯]` (Harvard University)

**通讯引用:** 15745 | [OpenAlex ID](https://openalex.org/A5086173064)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并评估了在有限测试预算下，基于结果自适应的动态福利最大化池化检测策略，主要通过贪婪动态分配算法与强化学习方法进行实验比较；

**💡 创新点**

首次将动态适应性引入福利最大化池化检测框架，证明了即使是简单的贪婪策略也能显著提升社会福利；

**🔧 技术方法**

使用贪婪启发式、混合整数规划（MILP）静态基线、Gibbs 采样后验概率更新、以及基于 PPO 的强化学习策略；

**📊 数据集**

实验使用人工生成的合成数据集，规模从 N=3、B=2 的小实例到 N=50、B=5 的大规模实例，健康概率均匀分布，效用取值 1、2、3；

**📈 对比分析**

与最优动态方案（仅在小规模可行）、静态 MILP 基线以及监督学习和 PPO 学习策略进行对比；在大规模情形下，贪婪动态策略平均提升约 1–3% 的福利，优于 MILP；强化学习策略尚未超越贪婪策略；

**⚠️ 局限性**

缺乏关于动态价值的理论下界；学习方法需要大量训练且在大规模实例上性能不佳；实验仅基于合成数据，缺乏真实世界评估。

---

## 186. Small Talk, Big Impact: The Energy Cost of Thanking AI

**arXiv ID:** 2601.22357 | [PDF](https://arxiv.org/pdf/2601.22357v1)

**作者:** Julien Delavande `[一作]` (Hugging Face), Sasha Luccioni `[通讯]` (Hugging Face)

**通讯引用:** 3532 | [OpenAlex ID](https://openalex.org/A5091714241)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对大型语言模型在回应礼貌性“谢谢”消息时的能耗进行系统测量与建模，探讨输入长度、输出长度和模型规模对能耗的影响。

**💡 创新点**

首次将能耗拆解为预填充（prefill）和解码（decode）两阶段，并提出闭式延迟模型与能耗公式，量化不同模型规模和响应长度对能耗的主导作用。

**🔧 技术方法**

使用GPU/CPU/内存功耗采集（NVML、Intel RAPL、CodeCarbon），基于Transformers库进行推理，构建理论延迟模型，并通过实验验证。

**📊 数据集**

基于约1万条真实聊天记录的“谢谢”结尾对话（从公开对话数据集改编）以及Qwen 2.5系列（0.5B–14B）和Mistral‑7B‑Instruct 的推理生成。

**📈 对比分析**

对比不同模型规模、不同输出长度下的能耗，发现能耗与输入长度线性相关、与输出长度二线性相关；较大模型耗能显著高于小模型，且在相同输出长度下更高；预填充阶段能耗受输入长度驱动，解码阶段受输出长度和模型规模驱动。

**⚠️ 局限性**

仅关注礼貌性微交互、单线程推理、单张H100 GPU，未考虑批量推理、不同硬件、动态功耗变化以及能耗对应的碳排放和成本转化，缺乏对更复杂对话场景的普适性验证。

---

## 187. Machine Unlearning in Low-Dimensional Feature Subspace

**arXiv ID:** 2601.22456 | [PDF](https://arxiv.org/pdf/2601.22456v1)

**作者:** Kun Fang `[一作]` (Hong Kong Polytechnic University), Haibo Hu `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 8664 | [OpenAlex ID](https://openalex.org/A5020630816)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于低维特征子空间的机器不学习方法，利用一个可学习的投影矩阵将模型特征投影到子空间中，最大化保留剩余数据的信息并最小化遗忘数据的信息，从而实现不学习。

**💡 创新点**

创新点在于：①引入低维特征子空间可分离性视角，证明遗忘数据与剩余数据在精确不学习模型的子空间中可被区分；②基于PCA思想构造投影矩阵优化目标，在Stiefel流形上求解；③实现仅一次特征提取、无需访问原始数据，且投影矩阵作为可插拔模块，无需微调或重训练整个网络。

**🔧 技术方法**

核心技术包括：主成分分析（PCA）与协方差矩阵构造、Stiefel流形上的Riemannian优化、投影矩阵的正交投影、特征子空间的重构误差与方差保留度量。

**📊 数据集**

实验使用 Tiny‑ImageNet（200 类）与 ImageNet‑1K（1000 类）数据集，分别配合 Swin‑T 与 ResNet‑50 进行类级与实例级不学习任务。

**📈 对比分析**

与 FT、GA、RL、SalUn、BT、L2UL、COUN、DELETE 等主流近似不学习方法及 Exact MU 对比。结果显示该方法在保持模型性能的同时，平均与 Exact MU 的差距仅为 0.85（Tiny‑ImageNet）和 0.29（ImageNet‑1K），在 MIA、Acc 等指标上均表现最好；计算时间仅约 0.5–1 秒，参数量降低至原模型的 4%~5%，显著优于其他方法。

**⚠️ 局限性**

局限性：①目前仅针对判别模型（分类）设计，无法直接应用于生成模型或需要更复杂特征映射的任务；②需要一次性计算特征协方差，若数据量极大仍需一定内存；③投影矩阵需针对每个不学习请求单独训练，若请求极多时仍需存储多份小矩阵；④在极端高维特征或多模态数据下，子空间可分离性可能降低，需要进一步研究。

---

## 188. Rethinking Anonymity Claims in Synthetic Data Generation: A Model-Centric Privacy Attack Perspective

**arXiv ID:** 2601.22434 | [PDF](https://arxiv.org/pdf/2601.22434v1)

**作者:** Georgi Ganev `[一作]` (University College London), Emiliano De Cristofaro `[通讯]` (University of California Riverside)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `67630363-6be0-4f51-ab05-7198250671a5` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文从模型层面重新审视合成数据的匿名性，阐释了 GDPR 中的三大可识别性风险（单独识别、链接与推断）与现代隐私攻击（差分、成员推断、属性推断、重构）的对应关系，并在此基础上评估常用隐私保护机制，证明差分隐私能够在理论与实验上充分缓解这三类风险，而基于相似度的隐私指标（SBPM）则缺乏足够的保护；

**💡 创新点**

创新点在于将监管层面的匿名性风险与模型层面最先进的隐私攻击映射起来，提出了以“受动者攻击测试”为基础的评估框架，并对比分析了差分隐私与 SBPM 在隐私、可解释性、可否拒绝责任等八个维度的表现，首次系统性说明了 SBPM 在满足 GDPR 匿名化要求方面的不足；

**🔧 技术方法**

主要使用的技术包括差分隐私（DP）框架、基于生成模型的合成数据生成、差分、成员、属性与重构隐私攻击，以及一系列针对 SBPM 的攻击（如 DCR、MIA、AIA、RAP-Rank 等）；

**📊 数据集**

实验主要在常用的公开表格数据集上进行，例如 Adult、Bank Marketing、Medical、Census 等，利用这些数据训练生成模型后进行攻击评估；

**📈 对比分析**

在实验比较中，差分隐私训练的模型在满足固定隐私预算的前提下能够显著降低成员推断、属性推断和重构的成功率，且不因多次生成数据而额外泄露信息；相比之下，SBPM 通过单次距离测试对隐私风险评估不足，攻击者仅需有限的 API 调用即可完成成员和属性推断，整体性能远逊于 DP；

**⚠️ 局限性**

本文的局限在于差分隐私的实现与调参仍较为复杂，对实际应用场景的适用性和实用性尚未给出统一指南；此外，评估主要基于公开数据集，未涉及更大规模或更敏感的行业数据，未来工作需进一步验证在多样化数据与攻击模型下的稳健性；

---

## 189. Plant-Inspired Robot Design Metaphors for Ambient HRI

**arXiv ID:** 2601.22387 | [PDF](https://arxiv.org/pdf/2601.22387v1)

**作者:** Victor Nikhil Antony `[一作]` (Johns Hopkins University), Chien-Ming Huang `[通讯]` (Johns Hopkins University)

**通讯引用:** 3202 | [OpenAlex ID](https://openalex.org/A5017287995)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

通过研究设计方法，将植物的存在感、时间性、形态与行为隐喻转化为机器人原型，探索植物启发的低干预、人类-机器人交互的可能性。

**💡 创新点**

创新点在于提出植物作为隐喻源的四维设计视角（存在、调色板、节奏、形状），并基于此迭代生成一系列可开放源代码的植物形态机器人原型。

**🔧 技术方法**

主要技术包括增材制造（PLA/TPU/PET‑G）与柔性驱动（拉绳、双气囊本体、步进电机与TMC驱动）以及多模态感知（光照、触觉）。

**📊 数据集**

研究未使用传统机器学习数据集，而是通过设计研讨会收集的 11 名参与者的注释、对话与草图作为定性数据。

**📈 对比分析**

通过对比参与者对不同原型的情感与实用性评价，发现植物式交互在低干预与情感表达上优于传统拟人化机器人，但缺乏可量化性能指标；设计评估主要以用户满意度与可感知度为准。

**⚠️ 局限性**

局限性包括原型多为概念性、缺乏长期实地实验、能耗与材料可持续性未评估，以及对隐私与人机关系的深度探讨不足。

---

## 190. SCALAR: Quantifying Structural Hallucination, Consistency, and Reasoning Gaps in Materials Foundation Models

**arXiv ID:** 2601.22312 | [PDF](https://arxiv.org/pdf/2601.22312v1)

**作者:** Can Polat `[一作]` (Texas A&M University), Hasan Kurban `[通讯]` (Hamad Bin Khalifa University)

**通讯引用:** 451 | [OpenAlex ID](https://openalex.org/A5070970331)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建了SCALAR基准，用于在不同尺度下评估材料基础模型的结构一致性、幻觉率和推理能力。

**💡 创新点**

创新点在于将晶体单胞与多尺寸纳米颗粒配对，设计了跨尺度、跨分布的评价指标（幻觉、一致性、推理、误差、解析错误），并公开了丰富的多尺度结构数据。

**🔧 技术方法**

利用大语言模型（如Claude、GPT-5、Gemini等）进行CIF→属性预测、链式思考推理以及逆检索任务，并与几种几何原生图神经网络和物理解析基线进行对比。

**📊 数据集**

使用从DFT-验证的单胞数据生成的约10万结构（单胞与10–30 Å半径纳米颗粒），涵盖41种元素，形成训练、ID和OOD旋转分布。

**📈 对比分析**

方法通过多维度指标（误差、幻觉率、一致性、推理相关度、解析错误率）评估模型，实验表明在OOD尺度下大多数模型出现高幻觉率和一致性下降，few‑shot与CoT提示的提升不稳定，物理解析基线在密集属性上表现最好。

**⚠️ 局限性**

局限包括缺乏缺陷/热运动等真实材料扰动、仅评估几何属性、旋转分割主要检验文本解析、幻觉与一致性指标可能重叠、对氢化物化学的普适性尚待验证、CoT提示易导致格式化不稳定。

---

## 191. Semi-Autonomous Mathematics Discovery with Gemini: A Case Study on the Erdős Problems

**arXiv ID:** 2601.22401 | [PDF](https://arxiv.org/pdf/2601.22401v1)

**作者:** Tony Feng `[一作]` (Google DeepMind), Thang Luong `[通讯]` (Google DeepMind)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

使用 Gemini Deep Think 的 Aletheia 研究代理对 Bloom 的 Erdős Problems 数据库中 700 个标记为“Open”的数学猜想进行半自主性评估，筛选出 5 个真正新颖的解决方案并识别出 8 个已在文献中解决的问题。

**💡 创新点**

首次系统性利用 AI 生成自然语言验证器与人类专家相结合的流程，揭示“Open”状态往往源于文献被忽视而非难度，并首次暴露 AI 在数学推理中潜在的“潜意识抄袭”风险。

**🔧 技术方法**

使用 Gemini Deep Think（Aletheia）多模态数学研究代理，配合内部自然语言验证机制进行筛选；随后采用人工专家评审和文献检索进行最终判定。

**📊 数据集**

Bloom 的 Erdős Problems 数据库（约 700 条未解猜想）和公开的数学文献库作为评估与检索依据。

**📈 对比分析**

通过对比人工评估结果，发现 68.5% 答案基本错误，31.5% 技术正确但 6.5% 真实有效，展示了半自主流程在高精度筛选上的有效性；与此前单纯 AI 解决或 Lean 形式化相比，本方法在可扩展性与效率上更具优势。

**⚠️ 局限性**

主要限制包括：对文献检索的依赖导致耗时、AI 对问题表述的误解导致大量无效答案、潜在的无意抄袭风险、缺乏完整的 Lean 形式化支持以及对数学专业领域专家的高需求。

---

## 192. ScamPilot: Simulating Conversations with LLMs to Protect Against Online Scams

**arXiv ID:** 2601.22426 | [PDF](https://arxiv.org/pdf/2601.22426v1)

**作者:** Owen Hoffman `[一作]` (Swarthmore College), Sukrit Venkatagiri `[通讯]` (Swarthmore College)

**通讯引用:** 293 | [OpenAlex ID](https://openalex.org/A5007675088)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了ScamPilot，一个基于大型语言模型的交互式界面，利用模拟诈骗者和受害者三方对话，让用户实时给受害者建议并获得即时反馈，以增强对在线诈骗的识别与抵御能力。

**💡 创新点**

创新点在于三方面：①将LLM生成的诈骗对话与用户实时建议结合，形成动态三方交互；②将预 inoculation 理论与 Kolb 体验学习框架（测试、实践、反思）融合，加入多项选择测验与即时反馈；③系统实现可自适应不同诈骗类型的对话生成与教学内容。

**🔧 技术方法**

核心技术包括：LLM Prompt Engineering（构建诈骗者、受害者与用户三方角色的系统提示）；对话生成与状态管理；多项选择测验与答案即时验证；反馈生成模型；前端React+后端Next.js+MongoDB架构。

**📊 数据集**

数据集：使用来自Prolific的150名美国受试者，基于预先设计的诈骗脚本（信用卡、身份冒充等）和固定对照对话；问卷收集自UT的自我效能、风险行为、合法性识别等量表；对话日志和建议文本作为训练/评测数据。

**📈 对比分析**

与四种交互条件（控制、测验、建议、测验+建议）进行随机对照实验。结果显示：测验+建议条件在诈骗识别（+8%）和情境判断（+8%）上显著优于对照；自我效能提升19%和响应效能提升9%；与单独测验或建议相比，其在合法信息识别上差异不显著，避免了过度警觉。

**⚠️ 局限性**

局限性：仅覆盖冒充类诈骗，需扩展至其他诈骗形式；LLM 生成对话可能包含不安全或误导内容，需要更严格的安全审查；受试者为在线平台样本，代表性有限；实验时间短，未评估长期记忆与行为转移。

---

## 193. Aligning Microscopic Vehicle and Macroscopic Traffic Statistics: Reconstructing Driving Behavior from Partial Data

**arXiv ID:** 2601.22242 | [PDF](https://arxiv.org/pdf/2601.22242v1)

**作者:** Zhihao Zhang `[一作]` (Ohio State University), Bowen Weng `[通讯]` (Iowa State University)

**通讯引用:** 272 | [OpenAlex ID](https://openalex.org/A5033842396)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一个两阶段框架，先用生成器根据部分微观轨迹和宏观统计恢复隐藏车辆状态，再用共享策略在完成的场景上训练，使策略在微观层面符合观测行为、宏观层面符合流量统计。

**💡 创新点**

将微观轨迹与宏观统计的异构、部分观测数据联合学习，并通过生成器完成隐藏状态，形成完整场景；利用宏观约束对策略进行正则化，解决多智能体识别和可扩展性问题。

**🔧 技术方法**

生成器采用自回归神经网络，策略采用PPO强化学习；目标函数为微观行为匹配奖励和宏观统计一致性奖励的轨迹级别组合。

**📊 数据集**

以环形道路为实验场景，使用IDM生成的全观测轨迹作为训练数据，人工随机隐藏部分车辆；宏观特征包括平均速度、平均间距、最小/最大间距等。

**📈 对比分析**

将学习得到的策略与基准IDM控制器对比；在宏观层面均速与间距均与目标相差不大（误差<0.6 m/s、0.2 m），微观层面展示合理的跟随响应，整体表现优于仅使用微观或宏观单独训练的方案。

**⚠️ 局限性**

仅在单车道环形道路上验证，缺乏多车道、交叉口等复杂网络；数据为合成模拟，未在真实交通数据上测试；生成器对隐藏车辆的估计仍可能产生不确定性。

---

## 194. Linux Kernel Recency Matters, CVE Severity Doesn't, and History Fades

**arXiv ID:** 2601.22196 | [PDF](https://arxiv.org/pdf/2601.22196v1)

**作者:** Piotr Przymus `[一作]` (Nicolaus Copernicus University), Gunnar Kudrjavets `[通讯]` (Amazon Web Services)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

**🎯 论文内容**

分析了 Linux 内核自 2024 年成为 CVE CNA 以来的 6464 条 CVE 及其对应的提交和修补时间，探讨漏洞提交与修复的结构、CVSS 评分与补丁时延的关系以及内核版本年龄对补丁速度的影响。

**💡 创新点**

发现 CVSS 评分对补丁时延几乎无影响，漏洞引入提交往往体积大、文件多、涉及文档和配置，而修复提交则极小且单文件；并证实较新的 LTS 版本修补速度更快，版本年龄是修补延迟的可预测因子。

**🔧 技术方法**

采用了 PatchScope 进行差异行级注解、Survival 分析（Kaplan–Meier、Cox 比例风险模型）评估补丁时延，以及 Somers’ D_xy 统计衡量 CVSS 与补丁时延的关联。

**📊 数据集**

使用了从 Linux kernel 版本仓库、官方 CVE 记录（NVD）以及自建的 CVE‑提交映射数据，覆盖 6 个 LTS 分支、约 46,000 (CVE, 版本) 对。

**📈 对比分析**

通过对比 CVSS 维度与补丁时延的 Somers’ D_xy（接近 0）以及不同 LTS 分支的 Kaplan–Meier 曲线，验证了版本年龄对修补速度的影响；相比传统的 CVSS‑优先级方法，后者对 Linux 内核补丁时延几乎没有预测力。

**⚠️ 局限性**

局限在于 CVE 引入时间的推断可能不准确、缺失或错误的提交映射会影响时延计算、CVSS 评分高度倾斜导致统计敏感度低，以及研究仅聚焦 LTS 分支，未必适用于主线或其他开源 OS。

---

## 195. PhoStream: Benchmarking Real-World Streaming for Omnimodal Assistants in Mobile Scenarios

**arXiv ID:** 2601.22575 | [PDF](https://arxiv.org/pdf/2601.22575v1)

**作者:** Xudong Lu `[一作]` (Chinese University of Hong Kong MMLab), Hongsheng Li `[通讯]` (Chinese University of Hong Kong MMLab)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并发布了PhoStream基准，评估移动场景下多模态大语言模型在实时视频流中的理解与回答能力。

**💡 创新点**

创新点包括：统一屏内屏外两类场景；提供开放式问答而非多项选择；设计实时推理流水线与时间戳控制；使用LLM‑as‑Judge评估开放式答案；揭示模型在向前任务中普遍存在的早期响应偏差。

**🔧 技术方法**

采用的技术包括：基于Gemini 3 Pro的自动生成与校验管线、双轮人工审核、秒级更新的在线推理流水线、滑动上下文窗口、时间戳切点判定、LLM裁判评分框架。

**📊 数据集**

使用的数据集为PhoStream：5,572个开放式问答对，来自578段平均时长13.3分钟的视频，涵盖4类场景（YouTube vlog、Phone Tutorial、Phone Record、EgoBlind）以及10类能力标签。

**📈 对比分析**

评估方式为即时、向后、向前三种时间逻辑的问答，利用LLM裁判给出0–5分并乘20得到0–100分；实验显示专有模型整体得分最高，向前任务得分最低，早期响应率高导致得分显著下降。

**⚠️ 局限性**

局限性在于：模型普遍缺乏判断何时应答的时间推理能力；加入音频虽然提升即时/向后表现，却往往导致更多早期响应；评估依赖LLM裁判，仍存在主观性与误差。

---

## 196. Darwinian Memory: A Training-Free Self-Regulating Memory System for GUI Agent Evolution

**arXiv ID:** 2601.22528 | [PDF](https://arxiv.org/pdf/2601.22528v1)

**作者:** Hongze Mi `[一作]` (Didichuxing Co. Ltd), Naiqiang Tan `[通讯]` (Didichuxing Co. Ltd)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Darwinian Memory System (DMS)，一种自适应、进化式的记忆框架，用于提升多模态大型语言模型在 GUI 自动化中的长时序跨应用任务性能。

**💡 创新点**

创新点在于：①将长流程拆分为可复用的预置-目标子任务，解决记忆刚性与上下文污染；②通过生存价值评估、动态衰减与可靠性惩罚实现自我调节，形成“适者生存”机制；③引入 ε‑mutation 与贝叶斯风险反馈，平衡探索与稳定，避免陷入局部最优。

**🔧 技术方法**

核心技术包括：层次化 Planner‑Actor 框架、双因子检索（预置+目标相似度）、ε‑mutation 与进化替换、基于 Beta-Binomial 的贝叶斯声誉模型、Elbow 法动态容量管理、时间衰减与可靠性惩罚的多维生存评分。

**📊 数据集**

使用 AndroidWorld 基准（116 个任务，20 个真实应用）进行评测，任务通过参数随机化生成数百万种变体。

**📈 对比分析**

对比包括 GPT‑4o、Qwen2.5‑VL‑72B、Qwen3‑VL‑30B、GLM‑4.5V 等开源/闭源模型，DMS 在 Qwen2.5‑VL‑72B 上成功率提升至 66.4%（比基线 25.4%），在 Qwen3‑VL‑30B 上提升 12.4%，整体在执行稳定性（SRR）和推理延迟上也均有显著改进。

**⚠️ 局限性**

局限性包括：①需要手动设计预置-目标格式与相似度函数；②在极端动态 UI 变化或完全新任务时仍可能出现记忆过时；③自适应阈值与衰减参数需在不同场景中调优，影响迁移性能。

---

## 197. Conformal Prediction for Generative Models via Adaptive Cluster-Based Density Estimation

**arXiv ID:** 2601.22298 | [PDF](https://arxiv.org/pdf/2601.22298v1)

**作者:** Qidong Yang `[一作]` (Massachusetts Institute of Technology), Sherrie Wang `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 1603 | [OpenAlex ID](https://openalex.org/A5088966490)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种针对条件生成模型的合规预测方法 CP4Gen，利用生成样本的密度估计来构造预测集并给出置信覆盖保证。

**💡 创新点**

创新点在于用 K‑means 聚类得到的高斯混合模型估计生成器的条件密度，控制超参数 K 可调节对离群点的鲁棒性与结构复杂度；并统一 CP 与生成模型的密度估计框架。

**🔧 技术方法**

采用 split conformal 预测、K‑means 聚类、Gaussian mixture（负对数似然得分）、Monte Carlo 体积估计以及降维处理等技术。

**📊 数据集**

实验使用合成一维/二维样本、9 个公开一维响应数据集（bike、protein、blog、Facebook 评论、医疗支出等）、纽约出租车两维响应、能源效率两维响应，以及气候模拟降尺度降温任务（全球降水降维至 2–4 维）。

**📈 对比分析**

与 PCP（以及 HD‑PCP）比较，CP4Gen 在保证 90% 边际覆盖率的同时，预测集体积更小、结构复杂度更低；在高维降水任务中维度升高时体积优势更明显。

**⚠️ 局限性**

局限性：仍需依赖离散样本进行密度估计，难以捕捉更复杂的空间/马尔可夫结构；对分布漂移敏感，只提供边际覆盖保证，未解决条件覆盖问题；在高维时需 Monte Carlo 估计体积，计算成本上升。

---

## 198. Task-Uniform Convergence and Backward Transfer in Federated Domain-Incremental Learning with Partial Participation

**arXiv ID:** 2601.22274 | [PDF](https://arxiv.org/pdf/2601.22274v1)

**作者:** Longtao Xu `[一作]` (Stony Brook University), Jian Li `[通讯]` (Stony Brook University)

**通讯引用:** 31302 | [OpenAlex ID](https://openalex.org/A5100402534)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并验证了服务器端正则化的Federated Domain-Incremental Learning算法，解决了跨任务漂移与部分参与的挑战。

**💡 创新点**

引入单一服务器端proximal anchor实现无记忆、无生成式、自适应漂移抑制，理论上给出BKT和全任务一致收敛率。

**🔧 技术方法**

Federated Averaging + 服务器端正则化（proximal term）+ 非凸理论分析 + 多任务漂移建模 + 客户端随机抽样。

**📊 数据集**

Digit-10、VLCS、PACS、DN4IL 四个域漂移视觉数据集。

**📈 对比分析**

与记忆式与无记忆式基线比较，ACC 最高、BWT 良好，通信/计算成本低于大多数对手。

**⚠️ 局限性**

假设梯度有界、L-smooth、同步抽样；未考虑异步/拖延；仅评估视觉模型，λ需手动调参。

---

## 199. AI Literacy, Safety Awareness, and STEM Career Aspirations of Australian Secondary Students: Evaluating the Impact of Workshop Interventions

**arXiv ID:** 2601.22486 | [PDF](https://arxiv.org/pdf/2601.22486v1)

**作者:** Christian Bergh `[一作]` (University of New South Wales), Jake Renzella `[通讯]` (University of New South Wales)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估了Day of AI Australia在澳大利亚中学进行的一次性AI素养工作坊对学生AI知识、认知与STEM职业兴趣的短期影响。

**💡 创新点**

创新点在于将工作坊内容与学生日常使用的社交媒体、流媒体平台等AI工具结合，验证了学生对AI的误解可通过一次性干预显著纠正，并首次量化了学生在深度伪造曝光与使用中的行为。

**🔧 技术方法**

使用混合方法调查问卷（Likert量表、多选、开放式问题）以及主题分析；统计分析采用非参数 Mann‑Whitney U、Friedman 检验和效应量 r。

**📊 数据集**

数据来自三所新南威尔士州政府中学共计205名前测、163名后测学生的问卷，涵盖AI工具使用频率、动机、信息来源以及深度伪造相关行为。

**📈 对比分析**

通过比较前后测样本的比例差异与效应量，结果显示AI知识得分提升显著（r=0.445），AI自信提升中等（r=0.279），识别AI工具的比例大幅提高（如Netflix 27.1%），但职业兴趣的效应量均小于0.3，表明单日工作坊对长期职业转向影响有限。

**⚠️ 局限性**

局限包括样本未配对导致仅做横向对比、三所学校中两所为女生校、工作坊内容与时序差异、深度伪造数据自报可能存在偏差、缺乏客观性能测评以及缺乏长期追踪。

---

## 200. FedCARE: Federated Unlearning with Conflict-Aware Projection and Relearning-Resistant Recovery

**arXiv ID:** 2601.22589 | [PDF](https://arxiv.org/pdf/2601.22589v1)

**作者:** Yue Li `[一作]` (Xidian University), Hui Li `[通讯]` (Xidian University)

**通讯引用:** 37292 | [OpenAlex ID](https://openalex.org/A5065859286)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出FedCARE框架，实现统一低开销的联邦忘记，支持客户端/实例/类别级别的忘记，并提供抗回归恢复；

**💡 创新点**

创新点在于使用冲突感知投影梯度上升结合无数据伪样本生成限制失效，并通过骨干冻结与分类头向量过滤实现回归抗性恢复；

**🔧 技术方法**

技术包括无数据伪样本生成（模型反演+GroupNorm）、冲突感知投影梯度上升、骨干冻结、分类头向量过滤以及传统FedAvg等联邦学习方法；

**📊 数据集**

实验使用MNIST、SVHN、CIFAR‑10/100数据集，模型为CNN或ResNet‑18，在IID与非IID分布下评估；

**📈 对比分析**

与Retrain、FedEraser、FedSE、MoDe、FedOSD、NoT等基线对比，指标R‑Acc、U‑Acc、MIA、ASR、时间、FLOPs等均优，FedCARE在保持接近Retrain的准确率同时显著降低时间与计算开销，并几乎消除回归风险；

**⚠️ 局限性**

局限在于伪样本生成在极端非IID或大规模客户端时可能不足以代表全局知识，需进一步验证在不同模型与分布下的适用性。

---

## 201. FraudShield: Knowledge Graph Empowered Defense for LLMs against Fraud Attacks

**arXiv ID:** 2601.22485 | [PDF](https://arxiv.org/pdf/2601.22485v1)

**作者:** Naen Xu `[一作]` (Zhejiang University), Shouling Ji `[通讯]` (Zhejiang University)

**通讯引用:** 7670 | [OpenAlex ID](https://openalex.org/A5058611515)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出FraudShield框架，构建与修正欺诈策略‑关键词知识图并用XML标签突出高置信度关键词，提升LLM在多种欺诈场景下的检测与响应能力。

**💡 创新点**

首次将欺诈策略与关键词映射为可解释的双向图，结合置信度过滤与图聚类实现高置信度证据生成，并在不需要微调的情况下通用闭源与开源LLM。

**🔧 技术方法**

采用提示式信息抽取、图聚类优化、置信度阈值剪枝、XML标签增强与指令式推理等技术。

**📊 数据集**

使用Fraud‑R1（涵盖五类欺诈场景的单/多轮对话）和MMLU评估通用任务准确率。

**📈 对比分析**

与Vanilla、Safety Prompt、Self‑Reminder、Goal Prioritization等基线在四个LLM（GPT‑4o‑mini、GPT‑o3‑mini、Llama‑3.1‑8B、Qwen‑2.5‑7B）以及Helpful Assistant和Role‑play两种设置下进行DSR@1、DSR@k与ACC评估，FraudShield平均提升DSR约24%（AS）和47%（RP），且ACC保持不变。

**⚠️ 局限性**

在极其细腻或隐蔽的欺诈文本中仍可能误判或无法完全拒绝，同时攻击者可利用该方法生成更具说服力的欺诈内容。

---

## 202. Score-based Integrated Gradient for Root Cause Explanations of Outliers

**arXiv ID:** 2601.22399 | [PDF](https://arxiv.org/pdf/2601.22399v1)

**作者:** Phuoc Nguyen `[一作]` (Applied Artificial Intelligence Initiative), Svetha Venkatesh `[通讯]` (Applied Artificial Intelligence Initiative)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种基于分数函数的积分梯度方法SIREN，用于解释异常样本的根本原因；

**💡 创新点**

创新点在于将条件分数匹配与逆向扩散积分梯度相结合，既支持非线性、异方差因果模型，又满足大部分Shapley公理；

**🔧 技术方法**

技术上使用了条件分数匹配、积分梯度、逆向SDE扩散、神经网络拟合因果机制以及自回归的链式求导；

**📊 数据集**

实验数据集包括合成随机图、云服务延迟模拟数据和供应链延迟仿真数据；

**📈 对比分析**

与Naive、Traversal、CIRCA、CausalRCA、BIGEN等基线比较，SIREN在NGCG@k/NDCG@k指标上均取得最高或相近的排名，显著优于其它方法；

**⚠️ 局限性**

局限性包括需先验已知因果结构、对分数模型训练的依赖、扩散轨迹采样的计算开销，以及在对称性公理上的不足。

---

## 203. Non-Intrusive Graph-Based Bot Detection for E-Commerce Using Inductive Graph Neural Networks

**arXiv ID:** 2601.22579 | [PDF](https://arxiv.org/pdf/2601.22579v1)

**作者:** Sichen Zhao `[一作]` (Northeastern University), Zihan Yu `[通讯]` (Northeastern University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

用非侵入式图模型检测电商平台恶意机器人，构建会话‑URL双边图并使用GraphSAGE进行实时评分。

**💡 创新点**

创新点在于将会话与内容视为异构图节点，利用归纳式GNN实现零启动和对抗鲁棒性。

**🔧 技术方法**

采用GraphSAGE聚合器、节点特征工程、邻居采样等图神经网络技术。

**📊 数据集**

数据集来自真实电商服务器日志，包含数万会话和数千URL，标注比例约5%为机器人。

**📈 对比分析**

与基线MLP对比，GraphSAGE AUC提升至0.9705，召回率90%，在对抗扰动和冷启动实验中保持高精度。

**⚠️ 局限性**

限制包括对极端伪装攻击的鲁棒性仍有限，极少边连接的新节点依赖特征，且模型训练需要较多计算资源。

---

## 204. AgentScore: Autoformulation of Deployable Clinical Scoring Systems

**arXiv ID:** 2601.22324 | [PDF](https://arxiv.org/pdf/2601.22324v1)

**作者:** Silas Ruhrberg Estévez `[一作]` (University of Cambridge), Mihaela van der Schaar `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

设计了一种自动构建符合临床可执行约束的单位权重检查表的框架。

**💡 创新点**

通过LLM指导规则生成、数据驱动验证与部署约束的多阶段搜索，实现了在严格可执行性限制下与传统可解释模型竞争的性能。

**🔧 技术方法**

采用大型语言模型（如 GPT‑5）生成规则、工具接口提供聚合统计、确定性验证、冗余过滤、规则多样性约束以及可执行性检查，并在此基础上构建单规则集并优化阈值。

**📊 数据集**

在 MIMIC‑IV、eICU、PhysioNet ICU 2012 以及英国囊性纤维化（CF）队列等多项真实临床任务上进行评估。

**📈 对比分析**

与 RiskSLIM、FasterRisk、AutoScore、PLR、决策树、逻辑回归等基线相比，平均 AUROC 提升约 0.05，且在外部验证中优于现有临床指南。

**⚠️ 局限性**

受限于单位权重和规则深度，表达能力有限；LLM 可能漏检可用规则；未对公平性、分布漂移做完整评估；在高复杂病例上可能不如深度模型。

---

## 205. SPARK: Real-Time Monitoring of Multi-Faceted Programming Exercises

**arXiv ID:** 2601.22256 | [PDF](https://arxiv.org/pdf/2601.22256v1)

**作者:** Yinuo Yang `[一作]` (University of Notre Dame), April Yi Wang `[通讯]` (ETH Zurich)

**通讯引用:** 461 | [OpenAlex ID](https://openalex.org/A5046673805)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了一套实时监控面板（Spark），支持教师在多层次编程练习中自定义检查点、生成自动化测试、可视化学生进度、检视中间输出并回放课堂过程。

**💡 创新点**

引入检查点结构化进度跟踪与AI自动生成测试、实时变量/DOM可视化以及回放功能，实现教师可自定义、主动介入的多面向学习分析；在大规模班级中提供结构化监控与深入洞察。

**🔧 技术方法**

使用 VS Code 扩展采集键盘输入、OpenAI API 生成测试代码、Puppeteer 执行模拟与评测、InfluxDB 存储时间序列数据、Resemble.js 进行截图聚类，配合前端可视化库构建仪表盘。

**📊 数据集**

22 名学生完成两道 20 分钟 Web 编程练习时产生的键盘记录数据集（约 810 次按键/学生/任务），用于实时模拟与回放。

**📈 对比分析**

与简化版基线（仅代码视图）通过 16 名教学者的交叉实验对比；采用 Likert 量表、测验准确率和混合效应回归分析；结果显示 Spark 在监测准确度、教师信心和信息洞察上显著优于基线，单个 Puppeteer 服务器在 22 名学生时保持 <30 s 延迟。

**⚠️ 局限性**

需人工准备检查点与测试代码，适用范围目前限定于 Web 编程；大班级下可视化信息易拥挤，UI 认知负荷较高；研究仅评估预设检查点的使用，缺乏对教师自创测试的长期使用验证。

---

## 206. Hair-Trigger Alignment: Black-Box Evaluation Cannot Guarantee Post-Update Alignment

**arXiv ID:** 2601.22313 | [PDF](https://arxiv.org/pdf/2601.22313v1)

**作者:** Yavuz Bakman `[一作]` (University of Southern California), Sai Praneeth Karimireddy `[通讯]` (University of Southern California)

**通讯引用:** 2209 | [OpenAlex ID](https://openalex.org/A5026569995)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了大型语言模型在更新后对齐问题的理论与实验框架，证明仅靠静态黑盒评估无法保证模型在微调或更新后仍保持对齐，并演示单步梯度更新即可触发隐藏的对齐失败。

**💡 创新点**

创新点包括：①正式定义静态与后更新对齐；②证明无论更新数据如何，任何黑盒评估都无法验证模型的更新鲁棒性；③展示单步梯度更新可激活潜在的违约行为；④证明隐藏对齐量随模型过参数化线性增长；⑤在三个核心领域（逃逸安全、隐私/重学习、行为诚实）进行实证验证。

**🔧 技术方法**

采用可逆重参数化、线性系统求解与梯度攻击式对齐训练（adversarial training with one-step gradient update）、LoRA低秩适配、黑盒评估工具（LlamaGuard、AdvBench、HarmfulQA等）以及不同梯度步长的实验设置。

**📊 数据集**

使用的主要数据集包括：Aegis2.0、AdvBench、HarmfulQA、TriviaQA、NaturalQuestions、TOFU、Alpaca、Dolly、GSM8K，以及用于衡量隐藏对齐容量的随机输入输出序列。

**📈 对比分析**

与原始模型在静态评估（安全分数、准确率、隐私泄露等指标）下表现相近；但在单步更新后，原始模型保持对齐，脆弱模型出现严重对齐失效：安全分数降为0、准确率下降、私有信息泄露从0升至1。实验表明，模型规模/参数量越大，隐藏对齐容量越大。

**⚠️ 局限性**

局限性：①仅考虑单步梯度更新，未探索多步或不同优化器的影响；②实验仅在两款模型上验证，未涵盖更广泛架构；③对齐训练假设可访问梯度信息，实际部署场景可能受限；④缺乏完整的白盒或更新感知评估方法；⑤未讨论更自然的数据级攻击或防御策略。

---

## 207. A Systematic Literature Review on LLM Defenses Against Prompt Injection and Jailbreaking: Expanding NIST Taxonomy

**arXiv ID:** 2601.22240 | [PDF](https://arxiv.org/pdf/2601.22240v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 208. Understanding Efficiency: Quantization, Batching, and Serving Strategies in LLM Energy Use

**arXiv ID:** 2601.22362 | [PDF](https://arxiv.org/pdf/2601.22362v1)

**作者:** Julien Delavande `[一作]` (Hugging Face), Sasha Luccioni `[通讯]` (Hugging Face)

**通讯引用:** 3532 | [OpenAlex ID](https://openalex.org/A5091714241)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估LLM推理能耗与延迟，探讨数值精度、批量化和请求调度对NVIDIA H100 GPU的影响

**💡 创新点**

系统级能耗分析与阶段感知profiling；利用到达调度（arrival shaping）在TGI服务器上实现能耗减至原来的1/100

**🔧 技术方法**

Tensor Cores低精度运算、FlashAttention、LLM.int8/int4量化、Hugging Face Text Generation Inference（TGI）持续批处理、CUDA内核级能耗/延迟测量

**📊 数据集**

10,000条礼貌式问答提示（UltraChat‑200k子集）以及Qwen2.5、Mistral‑7B‑Instruct‑v0.3、LLaMA3.1‑8B‑Instruct等模型

**📈 对比分析**

通过对prefill和decode两阶段分别测量能耗/延迟，并对不同dtype、批量大小、arrival模式进行平均比较，结果显示：大模型在compute‑bound阶段量化可获得4×能耗下降；TGI加固定间隔可将每请求能耗降低至原来的1/100；整体性能提升显著

**⚠️ 局限性**

仅在H100 GPU上实验；仅测GPU能耗，忽略CPU/内存/网络等系统级耗能；提示长度有限，未覆盖更长对话或多轮场景；量化效果受内存带宽限制，可能在其他硬件上差异明显

---

## 209. Heterogeneous Graph Alignment for Joint Reasoning and Interpretability

**arXiv ID:** 2601.22593 | [PDF](https://arxiv.org/pdf/2601.22593v1)

**作者:** Zahra Moslemi `[一作]` (University of California), Babak Shahbaba `[通讯]` (University of California)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

提出并实现了 Multi‑Graph Meta‑Transformer（MGMT）框架，用图变压器编码每个异构图，挑选超级节点，构建可解释的元图并在元图上进行跨图信息传播，实现图级预测。

**💡 创新点**

创新点在于：1）深度感知的图变压器融合多层信息；2）基于注意力的超级节点抽取；3）显式元图连接跨图子结构，提供细粒度解释；4）理论证明可实现 L‑hop 混合并相较单纯后期融合具有更低的逼近误差。

**🔧 技术方法**

采用图变压器（Graph Transformer）、注意力机制、深度感知聚合、余弦相似度构造跨图边、元图 Transformer 层、池化+MLP 分类等技术。

**📊 数据集**

在三套合成多图数据集、脑电 LFP 记忆实验（多动物）以及阿尔茨海默病检测（NACC）等真实神经与医学数据集上进行实验。

**📈 对比分析**

与单源模型、早期融合、MMGL、MultiMoDN、Meta‑Transformer、AMIGO、MaxCorrMGNN、MGLAM 等多种基线对比，MGMT 在所有任务的测试准确率均最高，且在大多数任务中提升 3–5% 以上。

**⚠️ 局限性**

局限性包括：对阈值参数（τ、γ）的敏感性；模型计算成本仍高，尤其是大规模图集合；仅针对图级分类，未覆盖节点/边预测任务；解释性依赖注意力可视化，仍需进一步验证因果解释能力。

---

## 210. In Vino Veritas and Vulnerabilities: Examining LLM Safety via Drunk Language Inducement

**arXiv ID:** 2601.22169 | [PDF](https://arxiv.org/pdf/2601.22169v1)

**作者:** Anudeex Shetty `[一作]` (University of New South Wales Sydney), Salil S. Kanhere `[通讯]` (University of New South Wales Sydney)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了在大型语言模型（LLM）中诱导醉酒语言（drunk language）的方法，并评估其对安全（越狱）和隐私（信息泄露）的影响。

**💡 创新点**

创新点在于：①首次系统性诱导LLM产生醉酒语言并量化其安全风险；②构建了大规模互联网醉酒文本语料库；③提出三种诱导策略（提示、因果微调、强化学习）并与现有越狱技术对比。

**🔧 技术方法**

技术手段包括：persona‑prompting（在提示中加入“醉酒人格”），LoRA微调（基于醉酒文本的因果语言建模），以及基于PPO的强化学习（使用醉酒判别器作为奖励）。

**📊 数据集**

数据集为从网站文本和Reddit r/drunk 收集的醉酒文本语料，经过自动分类器筛选后构成训练与评估集，约含数十万条文本。

**📈 对比分析**

与现有越狱与隐私泄露基准（JailbreakBench、ConfiADe）对比，诱导后的LLM在安全攻击成功率（ASR）上通常高于基线，且在隐私泄露率上也显著上升，说明醉酒诱导是有效的安全漏洞。

**⚠️ 局限性**

局限性包括：仅在单轮交互中验证；仅探讨了三种诱导方式，未覆盖如DPO等其他方法；仅关注醉酒状态，未考虑其他精神或物质影响的多重交互。

---

## 211. Successive Cancellation List Decoding of Extended Reed-Solomon Codes

**arXiv ID:** 2601.22482 | [PDF](https://arxiv.org/pdf/2601.22482v1)

**作者:** Xiaoqian Ye `[一作]` (Sun Yat-sen University), Chang-An Zhao `[通讯]` (Sun Yat-sen University)

**通讯引用:** 433 | [OpenAlex ID](https://openalex.org/A5062583521)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文提出一种将扩展Reed–Solomon（eRS）码转化为n个二进制极化码并采用成功取消（SC）和列表（SCL）解码的新方法，完成对eRS码的软判定解码；

**💡 创新点**

创新点在于引入预变换矩阵实现eRS码到极化码的映射，利用列线性独立性对SC性能给出下界，并在SCL解码中实现比传统Chase-BM和KV算法更低的有限域运算量；

**🔧 技术方法**

使用了极化编码理论、成功取消与列表解码、有限域线性代数、Gauss消元、Monte‑Carlo/高斯逼近等技术；

**📊 数据集**

实验基于AWGN信道下的BPSK调制，SNR从6 dB到11 dB不等，通过模拟得到FER曲线；未使用公开真实数据集；

**📈 对比分析**

通过将SCL的FER与Chase‑BM、KV算法在同一码长和码率下的FER、以及运算复杂度（有限域算子数与浮点运算量）进行对比，结果表明SCL在32位码长、0.22~0.78码率时FER优于两种传统算法，且有限域运算量更低，但浮点运算量略高；

**⚠️ 局限性**

局限性在于随着码长增大，SC/SCL性能随之下降，原因是预变换矩阵的列独立性限制了可利用的先验信息，使得该方法仅在短至中等码长下表现突出。

---

## 212. Attention Isn't All You Need for Emotion Recognition:Domain Features Outperform Transformers on the EAV Dataset

**arXiv ID:** 2601.22161 | [PDF](https://arxiv.org/pdf/2601.22161v1)

**作者:** Anmol Guragain `[一作]` `[通讯]` (Universidad Politécnica de Madrid), Anmol Guragain (Universidad Politécnica de Madrid)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

在EAV小样本情感识别任务中，系统比较了Transformer、因式分解注意力和简单CNN改进三类模型。

**💡 创新点**

发现因式分解注意力在小样本下表现差，简单的领域特征工程和bug修复更有效。

**🔧 技术方法**

采用预训练的AST/ViT、Squeeze‑Excitation、Delta MFCC、频域功率特征、分离注意力等技术。

**📊 数据集**

使用同步EEG、Audio、Video的EAV数据集（42人、约280训练样本/人）。

**📈 对比分析**

在单模态评估中，改进CNN分别提升了EEG 7.6pp、Audio 3.7pp、Vision 1.3pp，Transformer在Vision上得到75.3%准确率，均优于原论文基线。

**⚠️ 局限性**

局限在于仅做单模态、受样本量限制、缺乏跨模态融合与跨人泛化评估。

---

## 213. Tacit Coordination of Large Language Models

**arXiv ID:** 2601.22184 | [PDF](https://arxiv.org/pdf/2601.22184v1)

**作者:** Ido Aharon `[一作]` (Bar-Ilan University), Sarit Kraus `[通讯]` (Bar-Ilan University)

**通讯引用:** 18369 | [OpenAlex ID](https://openalex.org/A5103213461)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估大语言模型（LLM）在隐式协调游戏中的表现，并研究焦点点（focal points）在协调中的作用。

**💡 创新点**

首次大规模基于焦点点理论评估LLM的隐式协调能力，并提出无学习的提示技巧（如culture、saliency）提升协调效果。

**🔧 技术方法**

采用焦点点理论、协调指数（CI/NCI）、软最大映射、提示工程以及多规模LLM进行实验。

**📊 数据集**

使用 Amsterdam 与 Nottingham 的多答案问题数据集，以及 Bargaining Table 的竞合游戏数据。

**📈 对比分析**

与人类实验结果及多族群 LLM 进行对比，使用 CI/NCI、收益和社会福利指标。LLM 在大多数情境下表现优于人类，但在文化或数字敏感任务中略逊；提示技巧可进一步提升性能。

**⚠️ 局限性**

局限性包括对文化和数字细节的敏感度不足，推理深度不提升协调效果，实验结果高度依赖提示设计和模型规模。

---

## 214. MC-GRPO: Median-Centered Group Relative Policy Optimization for Small-Rollout Reinforcement Learning

**arXiv ID:** 2601.22582 | [PDF](https://arxiv.org/pdf/2601.22582v1)

**作者:** Youngeun Kim `[一作]` `[通讯]` (Amazon), Youngeun Kim (Amazon)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对大语言模型的群组相对策略优化（GRPO）方法，提出使用中位数基线替代均值基线，并丢弃中位数样本以保持梯度样本数不变，解决小滚动预算下优势符号翻转导致的训练不稳定。

**💡 创新点**

创新点在于将鲁棒的中位数作为共享基线，显著降低优势符号翻转率，并通过仅增加一个额外采样、删除零优势样本的方式实现无额外梯度样本开销的改进。

**🔧 技术方法**

核心技术为：中位数基线计算、极差（MAD）尺度估计、优势归一化、梯度裁剪的GRPO框架，以及对GRPO、DAPO、DR-GRPO等变体的直接替换。

**📊 数据集**

实验使用的主要数据集包括 GSM8K（数学推理）、Math-500（500道数学题）以及在此基础上评估的外部竞赛数据集 AMC‑2023 和 AIME‑2024。

**📈 对比分析**

在相同的模型（如 Qwen3‑1.7B、Llama‑3.2‑3B、Qwen‑2.5‑Math‑1.5B 等）与相同的滚动预算下，MC‑GRPO 在 G=2、4 的小预算场景中平均提升 2–5% 的准确率，甚至在 G=8 时也保持竞争力；训练时间略有提升（约 5–10%）。

**⚠️ 局限性**

局限性：评估主要基于结构化、近似二元的验证器奖励，缺乏对噪声大、动态或多目标奖励（如人类偏好模型）的验证，且在更复杂的推理任务上鲁棒性的泛化尚待进一步研究。

---

## 215. FAIRFORMER: A transformer architecture for discrete fair division

**arXiv ID:** 2601.22346 | [PDF](https://arxiv.org/pdf/2601.22346v1)

**作者:** Chris Mascioli `[一作]`, Mithun Chakraborty `[通讯]`

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

论文探讨了某种新型算法在特定任务中的应用，旨在提高效率和准确性。

**💡 创新点**

创新点在于提出了一种新的优化策略，能够在处理大规模数据时显著减少计算时间。

**🔧 技术方法**

使用了深度学习和强化学习相结合的技术。

**📊 数据集**

使用了公开的图像识别数据集和自建的实验数据集进行验证。

**📈 对比分析**

与现有的几种主流算法进行了比较，结果显示新算法在准确率和速度上均有显著提升。

**⚠️ 局限性**

限制在于算法在特定类型的数据上表现不佳，且对计算资源的需求较高。

---

## 216. Do AI Overviews Benefit Search Engines? An Ecosystem Perspective

**arXiv ID:** 2601.22493 | [PDF](https://arxiv.org/pdf/2601.22493v1)

**作者:** Yihang Wu `[一作]` (Zhejiang University), Fan Yao `[通讯]` (UNC-Chapel Hill)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

**🎯 论文内容**

研究 AI 概览对搜索引擎盈利的长期影响，并设计了两种激励机制（引用机制与补偿机制）来恢复创作者的激励，提升搜索引擎的长期利润。

**💡 创新点**

创新点包括：① 将乘法收益（位置偏差×创作者投入）与博弈论结合，建立了基于位置模型（PBM）的创作者竞争模型；② 在对称和二元异质成本下证明混合纳什均衡唯一且对称；③ 通过 Schur‑concavity 等技巧得到近似最优的分段常数补偿结构与对应的引用策略；④ 将理论与实证结合，首次在真实点击实验中验证 AI 概览对位置偏差的影响。

**🔧 技术方法**

使用的技术主要有：博弈论与机制设计（混合纳什均衡、位置模型、Schur‑concavity、数值优化）、用户点击实验（自建实验平台、A/B/C 对比实验）以及对位置偏差的估计。

**📊 数据集**

数据集：在自建实验平台上邀请 50 名参与者完成 24 个搜索查询（涵盖事实类与开放式查询），记录点击行为并估计位置偏差；查询结果和 AI 概览均采用 Google 原始结果构建；实验数据用于验证 AI 概览与引用对位置偏差及长期利润的影响。

**📈 对比分析**

对比方法：分别计算无 AI 概览、AI 概览无引用、AI 概览引用四页三种场景下的短期福利、长期福利与利润；在引入引用机制（TPBL）与补偿机制后再次计算长期利润。实验显示：AI 概览短期提升福利但长期导致利润下降；加入引用或补偿机制后，长期利润显著恢复并在高盈利率（α）和高创作者成本（低 β）场景下进一步提升。

**⚠️ 局限性**

局限性：① 仅考虑了二元异质成本结构，未覆盖更复杂的多类型创作者；② 假设位置偏差和关键词固定，未考虑多关键词或动态变化的搜索情境；③ 实验规模有限（50 人），可能受样本偏差影响；④ 未深入探讨 AI 概览能力随技术进步的动态演化。

---

## 217. Lethe:Adapter-Augmented Dual-Stream Update for Persistent Knowledge Erasure in Federated Unlearning

**arXiv ID:** 2601.22601 | [PDF](https://arxiv.org/pdf/2601.22601v1)

**作者:** Hanwei Tan `[一作]` (Jinan University), Yijun Quan `[通讯]` (University of Warwick)

**通讯引用:** 72 | [OpenAlex ID](https://openalex.org/A5076927814)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为Lethe的联邦无学习方法，旨在在训练继续进行时消除已删除数据对全局模型的影响，防止知识复现。

**💡 创新点**

创新点包括：①提出知识复现率（Resurfacing Rate）度量；②设计Reshape–Rectify–Restore三阶段流程，利用轻量化adapter进行“probe”学习；③在更新层面进行相关性校正，主动消除未学习集与剩余集更新的正相关。

**🔧 技术方法**

核心技术包括：梯度上升训练adapter、双流局部更新（忘记流与保持流）、层级相似度判别、负向或减法校正、恢复阶段的轻量化训练。

**📊 数据集**

使用MNIST（LeNet‑5）、CIFAR‑10（ResNet‑18）以及Tiny‑ImageNet（HSViT）等公开数据集进行实验。

**📈 对比分析**

与传统的重新训练、FedEraser、FedAU、FedOSD、NoT等基线相比，Lethe在所有粒度（样本、类、客户端）下均能保持RR<1%，显著低于其他方法，且通信轮数和恢复开销更小。

**⚠️ 局限性**

局限性包括：需要手动设定代理网络与正则强度，理论假设在复杂非IID环境下的适用性待验证，且目前仅在几个典型模型/数据集上验证，未来需进一步推广与自动调优。

---

## 218. Recoverability Has a Law: The ERR Measure for Tool-Augmented Agents

**arXiv ID:** 2601.22352 | [PDF](https://arxiv.org/pdf/2601.22352v1)

**作者:** Sri Vatsa Vuddanti `[一作]`, Satwik Kumar Chittiprolu `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究工具增强语言模型的恢复能力，提出期望恢复惩罚（ERR）与效率得分（ES）之间的定量法则，并通过大规模实验验证其可预测性。

**💡 创新点**

将恢复行为建模为期望恢复惩罚，并推导出一阶线性耦合关系（ERR–ES 法则），为执行层面噪声提供可验证的恢复动态规律；首次提出效率得分作为唯一可观测的恢复替代指标。

**🔧 技术方法**

理论推导采用稳态线性近似、风险敏感控制与MDP分析；实验使用大模型（Qwen 2.5、Gemma 3、Llama 3、Thinking V1）结合Monte Carlo roll‑outs、Bootstrap置信区间评估。

**📊 数据集**

五类工具使用基准（FortifyEval、ToolReflectEval、API‑Bank、Wikidata Live）以及生产 API（金融、旅游、天气）与人工扰动数据集。

**📈 对比分析**

通过比较恢复率（RR）、效率得分（ES）与观测ERR，计算误差Δ_pred；所有模型与策略在ES轴上呈单一效率‑惩罚曲线，误差Δ_norm≤0.05，表明在不同规模、扰动强度和恢复期限下均保持一致。

**⚠️ 局限性**

局限在于假设执行噪声为稳态、成本有界、误差方差低；对非平稳工具语义、无限重试或非ergodic状态演化等情况失效，导致ES与ERR失配。

---

## 219. Proof Complexity of Linear Logics

**arXiv ID:** 2601.22393 | [PDF](https://arxiv.org/pdf/2601.22393v1)

**作者:** Amirhossein Akbar Tabatabai `[一作]` (Bernoulli Institute), Raheleh Jalali `[通讯]` (Department of Computer Science)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

在经典序贯推理系统中，系统性地逐一删除弱化、收缩和剪切规则，研究其对证明规模的影响，并通过构造硬公式得到新的分离结果；

**💡 创新点**

证明了缺失任一单独结构规则就会导致指数级的证明规模爆炸，揭示了结构规则在生成高效证明中的本质作用，并表明受控指数规则（!）无法补偿这种缺失；

**🔧 技术方法**

主要使用 Chu 变换/Chu 转译、可判定推导（deduction theorem）技术、可行插值与单调电路下界、向量加法系统（VASS）归约等理论工具；

**📊 数据集**

不涉及实测数据集，所用“数据集”为理论构造的图形公式（如 Clique‑Color 公式）和 VASS 模型；

**📈 对比分析**

通过理论比较（证明规模与可模拟证明大小的多项式关系）证明，结构规则缺失导致的证明规模从多项式跃升到指数级，性能上表现为证明长度指数增长；

**⚠️ 局限性**

限制在于：尚未完成对所有无弱化/无收缩与仅弱化逻辑间的指数分离、剪切规则下的全证明行数下界以及在扩展 Frege 系统中的进一步分离等问题。

---

## 220. Gaussian Process Bandit Optimization with Machine Learning Predictions and Application to Hypothesis Generation

**arXiv ID:** 2601.22315 | [PDF](https://arxiv.org/pdf/2601.22315v1)

**作者:** Xin Jennifer Chen `[一作]` (Stanford University), Yunjin Tong `[通讯]` (Stanford University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种 Prediction-Augmented Gaussian Process Upper Confidence Bound (PA-GP-UCB) 算法，结合昂贵的真值oracle、廉价的预测oracle以及离线数据，对连续空间的未知目标函数进行贝叶斯优化。

**💡 创新点**

创新点包括：1) 将多任务高斯过程与控制变差估计结合，实现对预测偏差的校正；2) 在保持 GP-UCB 传统回报阶数不变的前提下，严格降低累计回报的常数因子；3) 通过 ε‑net 离线采样与在线回归实现对预测信噪比与覆盖率的可控性，并给出理论证明与经验验证。

**🔧 技术方法**

使用的主要技术有：多任务高斯过程、GP-UCB、控制变差（PPI）估计、ε‑net 离线设计、信息增益理论、以及基于 LLM 的预测模型。

**📊 数据集**

实验数据集包括：1) 1 维合成 Gaussian Process 基准；2) 54 条真实人类行为实验条件的访客到访率数据；3) 通过 LLM（OpenAI 模型）生成的预测数据，包含不同数量的 in-context 示例。

**📈 对比分析**

与 Vanilla GP-UCB 以及两种 naïve prediction‑augmented GP‑UCB 基线进行对比。实验表明，在所有实验设置下，PA‑GP‑UCB 的累计回报始终低于基线，收敛速度更快，尤其在预测相关性中等或低、离线采样规模有限时仍能显著提升性能。

**⚠️ 局限性**

局限性包括：1) 离线采样条件（ε、N）的理论保证相对保守，实际应用需经验调参；2) 对核函数或预测模型的误设容忍度有限，过度偏差会削弱收益；3) 仍需进一步研究成本敏感的离线‑在线权衡与自适应设计。

---

## 221. Neural Signals Generate Clinical Notes in the Wild

**arXiv ID:** 2601.22197 | [PDF](https://arxiv.org/pdf/2601.22197v1)

**作者:** Jathurshan Pradeepkumar `[一作]` (University of Illinois Urbana-Champaign), Jimeng Sun `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 27985 | [OpenAlex ID](https://openalex.org/A5084279065)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了首个临床EEG到语言的基础模型CELM，实现从长时EEG录音到结构化报告的端到端生成。

**💡 创新点**

创新点包括Epoch‑Aggregated Tokenization压缩极长EEG序列、Sequence‑Aware Alignment保留长时序依赖以及Prompt Fusion实现多尺度报告生成。

**🔧 技术方法**

采用预训练EEG编码器（如CBraMod）、LLM（Qwen3‑4B）和自定义对齐模块，结合Transformer注意力与Perceiver式压缩。

**📊 数据集**

使用Harvard EEG数据库v4.1中约12,290份临床报告与10,886名患者的EEG录音，过滤为单次会话约10,922份报告。

**📈 对比分析**

与文本单独输入或使用手工EEG特征的基线对比，CELM在ROUGE‑1、METEOR等指标上平均提升约2倍，零上下文时仍达0.43‑0.52分。

**⚠️ 局限性**

局限性包括缺乏严格临床评价指标、对极长记录的记忆容量有限以及模型仍易对罕见事件缺乏准确性。

---

## 222. Geometry without Position? When Positional Embeddings Help and Hurt Spatial Reasoning

**arXiv ID:** 2601.22231 | [PDF](https://arxiv.org/pdf/2601.22231v1)

**作者:** Jian Shi `[一作]` (King Abdullah University of Science and Technology), Peter Wonka `[通讯]` (King Abdullah University of Science and Technology)

**通讯引用:** 14906 | [OpenAlex ID](https://openalex.org/A5076768552)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文通过在ViT中对位置嵌入（PE）进行 token 级别的诊断与实验，探究 PE 作为几何先验如何决定多视角图像的空间一致性，并验证 PE 一致性是维持跨视角几何对齐的因果机制。

**💡 创新点**

将 PE 视作隐式空间核而非单纯坐标；设计无训练的 token 重索引方法快速恢复跨视角一致性；对 14 种基础 ViT 进行系统对比；分析 VGGT 通过聚合层隐式对齐 PE 的机制。

**🔧 技术方法**

使用 token 余弦相似度、4D 相关体积、EPE、Recall@n 等指标；解析旋转编码的相位核；对不同 PE 策略（绝对、相对、旋转）进行对比；采用无训练的 PE 对齐实验。

**📊 数据集**

Imagenette（用于裁剪对比）和 Spring 立体数据集（用于 4D 相关和 EPE 评估）。

**📈 对比分析**

在 14 种 ViT 上比较不同 PE 策略，量化跨视角 token 相似度、EPE 与 Recall@n。结果显示，保持 PE 一致可将相似度恢复至≈1，EPE 降低至小于 patch 大小，Recall 提升；失去或打乱 PE 时性能急剧下降。

**⚠️ 局限性**

仅在推理阶段评估 PE 作用，未针对训练过程做更改；实验仅覆盖固定的数据集与视角变换；未探索更广泛的 PE 设计与全局几何约束。

---

## 223. Discovering High-utility Sequential Rules with Increasing Utility Ratio

**arXiv ID:** 2601.22178 | [PDF](https://arxiv.org/pdf/2601.22178v1)

**作者:** Zhenqiang Ye `[一作]` (Jinan University), Philip S. Yu `[通讯]` (University of Illinois Chicago)

**通讯引用:** 133102 | [OpenAlex ID](https://openalex.org/A5036357902)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并实现了SRIU算法，用于挖掘具有递增效用比的高效用序列规则。

**💡 创新点**

创新点在于定义递增效用比约束、引入e‑index决定扩展方向、设计两套上界与剪枝策略，以及使用IPEUM与Roaring Bitmap提升效率。

**🔧 技术方法**

采用了规则增长框架、上界估计、剪枝策略、Roaring Bitmap、最大项映射等技术实现SRIU。

**📊 数据集**

使用了Bible、Kosarak10K、Leviathan、Sign四个真实数据集和Syn10K、Syn20K两个合成数据集进行实验。

**📈 对比分析**

与US‑Rule、HUSRM等基线对比，SRIU在大多数设置下实现了更快运行时间、更低内存消耗，并产生更高置信度与自信度的规则。

**⚠️ 局限性**

限制在于递增效用比阈值会显著削减规则数量，对稠密长序列的IPEUM和Roaring Bitmap在某些场景下效果不明显。

---

## 224. Large Language Model Agents Are Not Always Faithful Self-Evolvers

**arXiv ID:** 2601.22436 | [PDF](https://arxiv.org/pdf/2601.22436v1)

**作者:** Weixiang Zhao `[一作]` (Harbin Institute of Technology), Ting Liu `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估并分析了自我进化LLM代理在使用原始与压缩经验时的可信度。

**💡 创新点**

首次提出并量化经验可信度（faithfulness），揭示了代理对压缩经验的忽视与误用的系统性差距。

**🔧 技术方法**

采用因果干预、对比实验与梯度归因等技术对四种自我进化框架、十种LLM骨干和九类任务进行评估。

**📊 数据集**

使用 ExpeL、Dynamic CheatSheet、ReasoningBank、G-Memory 等框架，结合 HotpotQA、FEVER、MMLU、AIME、ALFWorld、WebArena 等公开数据集。

**📈 对比分析**

通过对干预前后表现的差异与基准对照，发现原始经验干预显著降低性能，而压缩经验干预几乎无影响；总体性能仍依赖模型规模，但可信度差距未改善。

**⚠️ 局限性**

局限在于仅关注经验可信度而未改进模型或记忆机制，且实验主要基于已公开框架与数据集，缺乏对更复杂场景与动态记忆更新的深入探索。

---

## 225. From Self-Evolving Synthetic Data to Verifiable-Reward RL: Post-Training Multi-turn Interactive Tool-Using Agents

**arXiv ID:** 2601.22607 | [PDF](https://arxiv.org/pdf/2601.22607v1)

**作者:** Jiaxuan Gao `[一作]` (Tsinghua University), Yi Wu `[通讯]` (Tsinghua University)

**通讯引用:** 11992 | [OpenAlex ID](https://openalex.org/A5107949456)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `67630363-6be0-4f51-ab05-7198250671a5` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种可扩展的后期训练框架，将自我演化的多智能体数据生成引擎与可验证奖励的强化学习相结合，用于训练多轮交互式工具使用代理。

**💡 创新点**

创新点在于：①自我演化的多智能体生成框架可迭代生成并验证可执行的对话和检查器；②采用群组相对优势的GRPO与动态过滤提升RL稳定性；③对用户模拟器进行微调以减少噪声；④整体实现实现了端到端的可验证奖励学习。

**🔧 技术方法**

技术手段包括层次化多智能体工作流、提示工程与自我批判、轨迹级群组相对优势GRPO、动态过滤、用户模型微调以及可执行验证器的自动生成。

**📊 数据集**

使用的主要数据集是自行生成的EiganData合成数据，评估基准为τ^2‑bench（Airline、Retail、Telecom）以及公开的Qwen3等开源大模型。

**📈 对比分析**

在τ^2‑bench上与业界领先模型对比，SFT后加RL可将pass1提升至73.0%（Airline）、98.3%（Telecom）等，整体性能超过或接近商业模型；多种消融实验验证每一模块的贡献。

**⚠️ 局限性**

局限性包括：①仍需依赖用户模拟器，模拟质量不佳会影响RL；②自我演化循环和提示工程复杂，需一定工程投入；③在某些域如Retail仍难以与顶尖模型持平；④对新领域迁移的鲁棒性尚待进一步验证。

---

## 226. Latent Spherical Flow Policy for Reinforcement Learning with Combinatorial Actions

**arXiv ID:** 2601.22211 | [PDF](https://arxiv.org/pdf/2601.22211v1)

**作者:** Lingkai Kong `[一作]` (Harvard University), Milind Tambe `[通讯]` (Harvard University)

**通讯引用:** 23161 | [OpenAlex ID](https://openalex.org/A5000327528)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `40105733-5154-44cd-8090-a8cab9e64b07` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种用于组合动作的强化学习的潜在球面流策略，旨在解决组合动作空间中的可行性和表达性问题。

**💡 创新点**

创新点在于将可行性约束的执行从策略网络转移到组合优化求解器，同时在连续的潜在空间中学习表达性随机策略。

**🔧 技术方法**

使用了球面流匹配技术来学习策略，并引入了平滑的贝尔曼算子以提高学习的稳定性。

**📊 数据集**

在公共基准测试和真实世界的性传播感染检测任务中进行了评估。

**📈 对比分析**

与现有的最先进方法相比，提出的方法在多个组合强化学习任务上平均提高了20.6%的性能，并且训练效率提高了约3.0倍。

**⚠️ 局限性**

限制在于未详细讨论的潜在问题，可能包括在特定约束下的策略表现和可扩展性。

---

## 227. PersonaAct: Simulating Short-Video Users with Personalized Agents for Counterfactual Filter Bubble Auditing

**arXiv ID:** 2601.22547 | [PDF](https://arxiv.org/pdf/2601.22547v1)

**作者:** Shilong Zhao `[一作]` (State Key Lab of AI Safety, Institute of Computing Technology, Chinese Academy of Sciences), Xueqi Cheng `[通讯]` (State Key Lab of AI Safety, Institute of Computing Technology, Chinese Academy of Sciences)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出 PersonaAct，一种基于人格化多模态代理的短视频用户模拟框架，用于规模化审计过滤泡沫。

**💡 创新点**

创新点在于：① 通过自动化访谈将行为日志与结构化提问相结合，生成可解释的人格化用户画像；② 将人格化画像作为条件输入，利用监督微调与强化学习训练多模态代理，实现更高的行为逼真度；③ 设计了基于反事实实验的过滤泡沫深度与广度评估方法。

**🔧 技术方法**

技术手段包括：多模态观测处理、Persona Interview Agent、监督微调（SFT）、Group Relative Policy Optimization（GRPO）、Jensen–Shannon 散度评估等。

**📊 数据集**

使用公开的首个多模态短视频用户行为数据集，包含 1,719 条视频交互记录、两位用户、27 次会话、视频帧、音频转录和用户动作。

**📈 对比分析**

与通用 LLM 模拟器对比，PersonaAct 在观看时长预测上 SMAPE 与 MAE 均显著下降，SFT+GRPO 方案在两位用户上的误差比基线低约 30–60%，验证了其更高的仿真精度和过滤泡沫评估的可靠性。

**⚠️ 局限性**

局限性包括：数据规模仅涵盖两名用户，缺乏跨平台与多样性；主要聚焦观看时长，其他交互稀缺；模型训练与评估仍受限于现有多模态技术与平台 API 的访问权限。

---

## 228. LEAP -- Live Experiments for Active Pedagogy

**arXiv ID:** 2601.22534 | [PDF](https://arxiv.org/pdf/2601.22534v1)

**作者:** Sumedh Karajagi `[一作]` (California Academy of Mathematics and Science), Bhaskar Krishnamachari `[通讯]` (University of Southern California)

**通讯引用:** 23678 | [OpenAlex ID](https://openalex.org/A5063784062)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 LEAP 框架，允许学生通过远程调用教师定义的函数进行交互式实验并记录日志；

**💡 创新点**

创新点在于利用轻量级 RPC 实现课堂即时互动、可视化学习轨迹、实时排行榜与数据驱动的教学反馈；

**🔧 技术方法**

技术包括 Python、JAX、RESTful API、PBKDF2 身份验证、数据库日志存储、Marimo 前端可视化；

**📊 数据集**

使用自定义实验数据集（如梯度下降、蒙特卡洛采样、数值积分、根求解等）并不依赖公开公开数据集；

**📈 对比分析**

未给出传统基准对比，主要通过实时仪表盘展示学生参与度与学习轨迹，可视化显示不同策略与错误模式；

**⚠️ 局限性**

局限性包括：目前仅支持 Python 客户端、缺乏跨语言支持、实验案例有限、对真实课堂大规模部署与评估尚未完成。

---

## 229. WED-Net: A Weather-Effect Disentanglement Network with Causal Augmentation for Urban Flow Prediction

**arXiv ID:** 2601.22586 | [PDF](https://arxiv.org/pdf/2601.22586v1)

**作者:** Qian Hong `[一作]` (Renmin University of China), Xiao Zhou `[通讯]` (Renmin University of China)

**通讯引用:** 25054 | [OpenAlex ID](https://openalex.org/A5002827290)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出WED‑Net模型，利用双分支Transformer（自注意力+交叉注意力）结合记忆库、天气判别器与自适应融合，对城市交通流进行天气感知预测；

**💡 创新点**

在交通与天气信息分离、引入天气判别器以实现天气不变表征、以及基于因果识别的时空增广三项技术上实现对极端天气的鲁棒泛化；

**🔧 技术方法**

采用Transformer自注意力、交叉注意力、时空记忆池、梯度反转判别器、以及时空因果增广等深度学习技术；

**📊 数据集**

使用纽约、芝加哥、华盛顿三市出租车流量与对应气象（降雨、温度、风速）数据集；

**📈 对比分析**

与STGCN、DCRNN、AGCRN、GWNet、GTS、STAEformer、MTGNN等基线比较，WED‑Net在极端天气下MAE、RMSE平均降低约14%–20%，始终保持最优；

**⚠️ 局限性**

对极端天气样本仍依赖稀缺数据，模型对降雨阈值和因果区分选择敏感，且未考虑多天气因素交互的复杂影响。

---

## 230. Cross-Domain Few-Shot Learning for Hyperspectral Image Classification Based on Mixup Foundation Model

**arXiv ID:** 2601.22581 | [PDF](https://arxiv.org/pdf/2601.22581v1)

**作者:** Naeem Paeedeh `[一作]` (Adelaide University), Wisnu Jatmiko `[通讯]` (University of Indonesia)

**通讯引用:** 2666 | [OpenAlex ID](https://openalex.org/A5069933043)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文针对跨域少样本学习（CDFSL）在高光谱图像分类中的难题，提出了 MIFOMO 模型，实现了跨域知识迁移。

**💡 创新点**

创新点包括：① 在基础模型 HyperSigma 上引入共聚投影（CP）实现参数高效微调；② 设计混合域适配的中间域，缓解域差异；③ 结合伪标签平滑技术提升伪标签质量。

**🔧 技术方法**

技术手段涵盖：迁移学习 + 基础模型、mixup 采样、共聚投影（CP）、伪标签平滑、元学习（episodic）训练，以及 T‑SNE 可视化等。

**📊 数据集**

实验数据集涵盖：Chikusei（源域）、Indian Pines、Salinas、Pavia University、Houston（目标域）等四大高光谱数据集。

**📈 对比分析**

通过与 20+ 先前方法（如 CFSL‑KT、GLGAT‑CFSL、CDLA、MGPDO 等）在 5‑shot 设置下比较，MIFOMO 在 OA、AA、KC 上均领先，最高可提升约 14%。

**⚠️ 局限性**

局限性在于方法属于传递式（transductive）范式，需要访问目标域的未标记样本；未来需探索无标签目标域的归纳式（inductive）方法。

---

## 231. SpanNorm: Reconciling Training Stability and Performance in Deep Transformers

**arXiv ID:** 2601.22580 | [PDF](https://arxiv.org/pdf/2601.22580v1)

**作者:** Chao Wang `[一作]` (Meituan Inc.), Xunliang Cai `[通讯]` (Meituan Inc.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种新的Transformer归一化方案SpanNorm，旨在解决PreNorm的稳定性与PostNorm的性能之间的权衡；

**💡 创新点**

创新点在于通过在第二个残差连接处将原始块输入直接加入，形成跨整个Transformer块的跳连路，从而兼具PreNorm的梯度稳定性和PostNorm的表达能力；

**🔧 技术方法**

使用了LayerNorm、残差连接、以及对权重进行1/√L缩放的初始化策略，并在实验中结合了Mixture-of-Experts（MoE）架构；

**📊 数据集**

主要使用了SlimPajama大规模文本数据集，规模从30B到200B token，并在Mistral tokenizer下进行训练；

**📈 对比分析**

与PreNorm、PostNorm、HybridNorm、Mix-LN、LayerNorm Scale等方法对比，SpanNorm在Dense和MoE模型上均取得1–4点的性能提升，并成功训练到128层、6.5B参数的模型，显著优于传统方法；

**⚠️ 局限性**

局限性包括：对不同Transformer变体（如Swin Transformer等）未进行验证，且在极深层数（512层）下的训练仍需更大算力；

---

## 232. DELNet: Continuous All-in-One Weather Removal via Dynamic Expert Library

**arXiv ID:** 2601.22573 | [PDF](https://arxiv.org/pdf/2601.22573v1)

**作者:** Shihong Liu `[一作]`, Hanguang Xiao `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `afceb026-1760-41ae-8d86-010831a37d97` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于连续学习的多天气图像恢复框架DELNet，能够在不重新训练的情况下适应新的降雨、雾霾、雪等天气干扰；

**💡 创新点**

创新点在于（1）判断阀（Judging Valve）自动评估任务相似性并决定是否为新任务；（2）动态专家库（Dynamic Expert Library）存储不同天气下的专家并通过Top‑k激活，避免灾难性遗忘；（3）多级损失设计融合重建、对比、蒸馏、投影和多样性约束；

**🔧 技术方法**

使用深度特征增强网络（DFE）结合自注意力与极化注意力；判断阀基于多统计量与三种相似度（余弦、欧氏、皮尔逊）动态阈值；专家采用轻量化适配器并通过性能-使用度联合分数与温度缩放融合；损失采用重建+对比、输出蒸馏、特征投影、适配器正则和专家多样性；

**📊 数据集**

在RESIDE（雾）、Rain100H（雨）和Snow100K（雪）三个公开数据集上进行评估；

**📈 对比分析**

与多种连续学习基线（EWC、MAS、LwF、POD、PIGWM、AFC）及所有任务统一模型（TransWeather、WGWS、CLAIO等）对比，DELNet在PSNR/SSIM上分别比最佳连续学习基线提升约3–4dB，且参数量仅5.6M，明显优于TransWeather（38.1M）等；

**⚠️ 局限性**

限制包括对极端天气条件的鲁棒性尚待进一步验证，任务顺序对学习效果有一定影响，且动态阈值和专家数量需手动调参，未来工作将聚焦高效教师更新与极端情况下的鲁棒性提升。

---

## 233. EUGens: Efficient, Unified, and General Dense Layers

**arXiv ID:** 2601.22563 | [PDF](https://arxiv.org/pdf/2601.22563v1)

**作者:** Sang Min Kim `[一作]` (Seoul National University), Krzysztof Choromanski `[通讯]` (Google DeepMind)

**通讯引用:** 2684 | [OpenAlex ID](https://openalex.org/A5031842812)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于随机特征的高效稠密层（EUGen），能够无偏逼近全连接前馈层并将其直接输入范数纳入计算，显著降低推理复杂度和参数量。

**💡 创新点**

创新点包括：①利用随机特征实现多阶多项式激活函数的无偏近似；②在层与层之间解耦权重与输入，通过简单线性运算重构；③提出层级知识迁移方法，支持无梯度反向传播的快速适配；④统一并超越现有的高效前馈层扩展。

**🔧 技术方法**

使用的技术主要有：随机特征（RF）变换、准蒙特卡洛（QMC）加速、核方法与线性化、集中界与方差分析、层级知识蒸馏。

**📊 数据集**

实验数据集涵盖：GPT‑2预训练（OpenWebText，约36.8B token），ViT在ImageNet与Places365上的图像分类，NeRF/3D场景重建，以及多语言模型与视觉Transformer的基准评测。

**📈 对比分析**

通过与原始全连接层、低秩近似和SNNK等方法对比，EUGen在推理速度上提升最高27%，内存占用下降30%，在验证损失/准确率上保持与原模型相当，且在参数量和计算量上显著优于对比方法。

**⚠️ 局限性**

局限性：对低阶多项式激活的表达能力有限，需调节阶数k；随机特征的质量和数量对近似误差敏感；在极深网络或极大模型中，累积误差可能影响最终性能；需要额外的理论分析与调参成本。

---

## 234. LeanArchitect: Automating Blueprint Generation for Humans and AI

**arXiv ID:** 2601.22554 | [PDF](https://arxiv.org/pdf/2601.22554v1)

**作者:** Thomas Zhu `[一作]` (Carnegie Mellon University), Sean Welleck `[通讯]` (Carnegie Mellon University)

**通讯引用:** 2242 | [OpenAlex ID](https://openalex.org/A5019030424)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了 LeanArchitect 工具，提供在 Lean 代码中使用 @[blueprint] 注解、自动提取依赖与证明状态、在 Lake 构建系统中导出蓝图，并通过这些功能实现 Lean 代码与蓝图工作流的同步与集成。

**💡 创新点**

创新点在于引入了 Lean 原生的 @[blueprint] 属性，将蓝图元数据与 Lean 声明绑定；通过自动推断依赖和 proof 状态，消除手动维护蓝图的重复劳动；同时为 AI 自动化证明提供结构化的蓝图接口，显著提升协同效率。

**🔧 技术方法**

使用 Lean 4 语言实现插件与环境扩展，利用 Lake 构建系统进行增量导出；采用 doc‑gen4 风格的代码生成；使用 GPT‑5 Pro 生成蓝图和 Lean 草稿，Aristotle 进行 sorry‑filling；在实现中也用到了 Lean 内置的属性系统、宏与环境扩展机制。

**📊 数据集**

对 PrimeNumberTheoremAnd、Carleson、Brownian Motion、Infinity Cosmos、Fermat's Last Theorem 等五个大型 Lean 项目进行了迁移与实验；这些项目作为实际数据集验证工具的可行性和效果。

**📈 对比分析**

与传统手动蓝图同步流程相比，LeanArchitect 通过自动推断减少了 80% 以上的手动编辑工作量，能自动发现蓝图与 Lean 代码之间的隐藏不一致；在多变量 Taylor 定理的案例中，蓝图可视化帮助快速定位未证明节点，人工修正后 AI 能完成大部分子目标，整体协同效率明显提升（但未给出具体数值）。

**⚠️ 局限性**

限制包括：Lean IDE 对 @[blueprint] 注解的语法高亮与交互体验欠佳；单一 Lean 声明只能对应一个蓝图节点，导致同一命题在蓝图中多次出现时处理不便；构建步骤增加了 CI 的复杂度；纯粹非正式的蓝图节点仍需手动维护。

---

## 235. Exo-Plore: Exploring Exoskeleton Control Space through Human-aligned Simulation

**arXiv ID:** 2601.22550 | [PDF](https://arxiv.org/pdf/2601.22550v1)

**作者:** Geonho Leem `[一作]` (Seoul National University), Jungdam Won `[通讯]` (Seoul National University)

**通讯引用:** 1122 | [OpenAlex ID](https://openalex.org/A5090223842)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一个基于神经力学仿真与深度强化学习的框架，用来在仿真中优化膝髋外骨骼的控制参数，避免了耗时的人类实验。

**💡 创新点**

创新点在于将自适应的“人-外骨骼交互”奖励与深度RL生成的步态数据结合，利用神经网络 surrogate 取代传统贝叶斯优化实现高效搜索，并在五种病理步态中发现助力参数与病理严重度呈线性关系。

**🔧 技术方法**

采用深度强化学习生成步态数据，利用多层感知器 surrogate 以及拉丁方采样构建代价景观，并通过SLSQP优化，配合Hill型肌肉模型与DART物理引擎实现仿真。

**📊 数据集**

主要使用已有的实验步态与能耗数据（包括无外骨骼和不同助力参数下的实验）作为仿真匹配的基准。

**📈 对比分析**

通过与真实实验的关节角度、肌肉激活、代谢消耗、RMS 助力功率和代谢减速率进行定量对比，仿真结果在趋势和量级上与实验高度一致，且优化得到的参数随速度递减、病理严重度呈线性关系。

**⚠️ 局限性**

局限包括未在真实人群（尤其病人）中验证、奖励模型过于简化、缺乏个体化调节、肌肉动力学近似可能不足以捕捉个体差异，以及对部分病理步态（如足落地冲击）优化不收敛。

---

## 236. Benchmarking Long Roll-outs of Auto-regressive Neural Operators for the Compressible Navier-Stokes Equations with Conserved Quantity Correction

**arXiv ID:** 2601.22541 | [PDF](https://arxiv.org/pdf/2601.22541v1)

**作者:** Sean Current `[一作]`, Srinivasan Parthasarathy `[通讯]` (Ohio State University)

**通讯引用:** 19321 | [OpenAlex ID](https://openalex.org/A5100755351)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种模型无关的保守量修正方法，利用质量和动量守恒约束来校正神经算子，使其在自回归预测中长时程保持数值稳定性。

**💡 创新点**

创新点在于：①将质量与动量两种守恒量的校正统一到同一框架；②提出了两种校正策略（幅值缩放与平移校正），可应用于任意神经算子；③通过频域分析揭示当前算子在高频特征上的局限性。

**🔧 技术方法**

技术手段包括：FNO 与 DPOT 两种神经算子架构、Patch 嵌入与 Fourier Attention、自动回归滚动训练（5 步）、Adam 一周期学习率调度、L2 相对误差评估、以及保守量校正的幅值与平移公式。

**📊 数据集**

实验数据集为 PDEBench 的 2D 可压缩 Navier‑Stokes 数据，使用低粘度（η=ζ=1e‑8）和两组马赫数（0.1 与 1.0）的 128×128 网格。

**📈 对比分析**

与基线 FNO、DPOT、DPOT‑Ti、DPOT‑S 以及仅质量校正的模型比较，结果显示校正后模型在 1‑10 步的 L2 相对误差显著下降（如 DPOT_c 在 t+10 时误差 7.8% 对比 8.3%），长时程滚动（50 步）中保持更高的相关性（DPOT_c 约 28 步超过 90% 相关），证明了校正能提升长周期预测性能。

**⚠️ 局限性**

局限性包括：所有算子仍难以捕捉高频细节，特别是在马赫 1.0 的湍流数据中表现更差；FNO 受 Fourier 模式数量限制；DPOT 的 Patch 解码在高频时产生伪影；因此高频表示仍是未来研究的主要瓶颈。

---

## 237. Neural-Inspired Posterior Approximation (NIPA)

**arXiv ID:** 2601.22539 | [PDF](https://arxiv.org/pdf/2601.22539v1)

**作者:** Babak Shahbaba `[一作]` (University of California Irvine), Zahra Moslemi `[通讯]` (University of California Irvine)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种神经启发的后验近似框架（NIPA），通过整合模型驱动（HMC）、模型自由（自编码+神经网络逼近）与情节记忆（近邻检索）三种机制，实现高效的贝叶斯推断。

**💡 创新点**

创新点在于将人类多重控制系统（基于模型、习惯与情节记忆）映射到采样算法中，采用距离门控动态切换三种子模块，显著降低了对目标分布评估的计算开销。

**🔧 技术方法**

使用的技术包括Hamiltonian Monte Carlo、SGHMC、自动编码器降维、深度前馈网络逼近、标准化距离度量与门控策略以及记忆检索。

**📊 数据集**

实验数据集涵盖合成回归/分类任务、Year Prediction MSD（音乐年份预测）和MNIST（奇偶分类）等。

**📈 对比分析**

与传统BAYESNN-HMC、SGHMC、pCN、Lasso、VI、MC‑Dropout、SWAG、随机网络逼近以及DNN集成等基线对比，NIPA在保持相近RMSE/准确率的同时，CP95和ECE更佳，平均计算速度比HMC提升约8倍。

**⚠️ 局限性**

局限性包括门控阈值的手工调参、逼近模型对高维空间的适应性有限、记忆检索对最近邻的依赖以及在极大模型规模下可能出现的存储与更新开销。

---

## 238. Demystifying Design Choices of Reinforcement Fine-tuning: A Batched Contextual Bandit Learning Perspective

**arXiv ID:** 2601.22532 | [PDF](https://arxiv.org/pdf/2601.22532v1)

**作者:** Hong Xie `[一作]` (University of Science and Technology of China), Enhong Chen `[通讯]` (University of Science and Technology of China)

**通讯引用:** 27823 | [OpenAlex ID](https://openalex.org/A5048237545)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文构建了一个最小化基线，并通过系统实验探究了强化细调中各设计选择（优势函数、rollout数、batch size、replay策略等）的学习与泛化贡献。

**💡 创新点**

创新点在于提出了以单次rollout、无优势、batch size=32的极简基线，将强化细调视作批量情境赌博问题；同时设计了可复现的实验流水线和回放策略，以揭示不同设计参数的边际提升与最优权衡。

**🔧 技术方法**

采用基于GRPO的策略梯度优化，结合优势估计、rollout扩展、batch大小调节和经验回放，并利用 PyTorch/Verl 框架实现。

**📊 数据集**

使用三款小规模指令微调 LLM（Qwen2.5-0.5B、LLaMA‑3.2‑1B、OLMo‑2‑0425‑1B）与两套公开数学推理数据集（MATH 与 GSM8K）进行实验。

**📈 对比分析**

通过 Pass@1 评估模型在训练集与测试集上的表现，发现基线已取得显著提升，优势函数与 roll‑out 数提升有限，batch size 与 roll‑out 的权衡在 (32,8) 配置下表现最好，回放策略可近似该最优组合。整体提升幅度因模型/数据而异，最高约 0.17（GSM8K 上 Qwen）而最低为 0（OLMo‑MATH）。

**⚠️ 局限性**

局限性包括：实验仅覆盖三种小规模模型和两类任务，无法验证结论在更大模型或不同领域的泛化；基线假设 reward 为离散 0/1，可能不适用于连续奖励；回放策略虽然高效，但在极端资源受限环境下仍需进一步优化。

---

## 239. Learn from A Rationalist: Distilling Intermediate Interpretable Rationales

**arXiv ID:** 2601.22531 | [PDF](https://arxiv.org/pdf/2601.22531v1)

**作者:** Jiayi Dai `[一作]` (University of Alberta), Randy Goebel `[通讯]` (University of Alberta)

**通讯引用:** 4679 | [OpenAlex ID](https://openalex.org/A5091866119)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了REKD（Rationale Extraction with Knowledge Distillation）框架，结合了可解释性目标选择与知识蒸馏，使轻量级模型在可解释约束下提升预测性能。

**💡 创新点**

创新点在于同步使用Gumbel-Softmax温度退火与蒸馏温度，构建了一个隐式的课程学习机制；并将蒸馏目标扩展到选择器和预测器的分布匹配。

**🔧 技术方法**

核心技术包括可微分的Straight-Through Gumbel-Softmax、KL蒸馏损失、温度退火调度以及联合的选择与预测损失。

**📊 数据集**

实验使用语言任务IMDB情感分析与视觉任务CIFAR‑10/100，采用BERT与ViT系列模型（Base/Small/Tiny）。

**📈 对比分析**

与仅使用RE的基线相比，REKD在相同的rationale比例下显著提升准确率（例如ViT Small从0.889提升至0.968，ViT Tiny从0.797提升至0.936；BERT Small在IMDB从0.881提升至0.906），并且方差更小。

**⚠️ 局限性**

局限性包括：仅在相同架构的师生对进行蒸馏，未验证跨架构蒸馏效果；以及使用固定的损失权重，缺乏动态权重调度；同时未讨论对鲁棒性或偏见的影响。

---

## 240. Detect and Act: Automated Dynamic Optimizer through Meta-Black-Box Optimization

**arXiv ID:** 2601.22542 | [PDF](https://arxiv.org/pdf/2601.22542v1)

**作者:** Zijian Gao `[一作]` (South China University of Technology), Hongshu Guo `[通讯]` (South China University of Technology)

**通讯引用:** 37 | [OpenAlex ID](https://openalex.org/A5113281508)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

提出了一种基于强化学习的动态优化框架Meta-DO，自动实现环境变化检测与自适应调整，取代传统的人工设计的检测-响应管道；

**💡 创新点**

创新点在于将动态优化视为端到端的MDP，使用Transformer结构对种群状态进行全局编码，并联合控制PSO的惯性权重与加速度系数，同时引入长时记忆的archive与对数尺度奖励，实现对多变环境的无手工阈值自适应；

**🔧 技术方法**

采用深度Q网络（DQN）/PPO强化学习策略、Transformer编码器、NBNC-PSO低层优化器、log‑scale奖励设计，以及基于archive的环境漂移感知；

**📊 数据集**

在64个自定义的动态测试集合（包含噪声、景观切换、混合转移等）上训练，在32个独立测试实例上评估，并在真实海面船舶路径规划任务（OKAOP）中检验跨域泛化；

**📈 对比分析**

与8个现有动态优化算法（DynDE、mDE、mCMAES、APCPSO、DPCPSO、PSPSO、ACFPSO、NBNC-PSO）以及实时USV规划基线比较，Meta-DO在32个测试实例上平均排名1.15625，29/32实例获得最佳均值；在USV任务中显著提升成功率、终点距离与步数，证明性能优越；

**⚠️ 局限性**

局限在于低层搜索仍受PSO固定结构限制，特征在高维空间可能不稳定，未来需探索个体级算子自适应和更通用的学习-优化范式。

---

## 241. COBRA++: Enhanced COBRA Optimizer with Augmented Surrogate Pool and Reinforced Surrogate Selection

**arXiv ID:** 2601.22624 | [PDF](https://arxiv.org/pdf/2601.22624v1)

**作者:** Zepei Yu `[一作]`, Zeyuan Ma `[通讯]` (South China University of Technology)

**通讯引用:** 5886 | [OpenAlex ID](https://openalex.org/A5063438356)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文提出了 COBRA++，一种针对昂贵约束黑盒优化的自适应 surrogate 选择框架，显著提升了 COBRA 的求解效率和精度。

**💡 创新点**

创新点在于构建多样化的 RBF surrogate 池（包括不同核类型与宽度）以及利用强化学习（DQN）在线动态选择最合适的 surrogate，从而取代传统的手工或启发式选择策略。

**🔧 技术方法**

使用了 RBF 基函数模型、深度 Q‑网络（DQN）进行 surrogate 选择、经验回放与 epsilon‑greedy 探索，以及基于状态特征的多层感知机进行特征提取。

**📊 数据集**

实验基于 COCO 平台的 bbob‑constrained 基准集，共 54 个约束优化问题（10 维和 40 维），并在训练集上学习、在测试集上验证泛化性能。

**📈 对比分析**

与 COBRA、SACOBRA、A‑SACOBRA 对比时，COBRA++ 在 10 维和 40 维问题上均实现了更高的相对改进（RI）和更快的收敛速度，尤其在大多数测试实例中显著优于对手。

**⚠️ 局限性**

局限性包括：对极度狭窄或异常几何特征（如 Bent Cigar）的可行解发现仍存在困难；仅使用 RBF 作为 surrogate，未考虑更丰富的 surrogate 形式；以及需要在训练阶段消耗额外的计算资源。

---

## 242. Assistive Robots and Reasonable Work Assignment Reduce Perceived Stigma toward Persons with Disabilities

**arXiv ID:** 2601.22689 | [PDF](https://arxiv.org/pdf/2601.22689v1)

**作者:** Stina Klein `[一作]` (University of Augsburg), Nils Mandischer `[通讯]` (University of Augsburg)

**通讯引用:** 148 | [OpenAlex ID](https://openalex.org/A5030132038)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过情景式问卷（vignette）探讨了在不同工作情境下，辅助机器人对残疾人（PwD）在职场中的认知与行为污名化的影响。

**💡 创新点**

创新点在于首次系统检验辅助机器人与通用设计（universal design）对降低认知污名化的作用，并将其与工作任务适配度和机器人辅助程度关联。

**🔧 技术方法**

采用情景实验、WMISS（工作场所精神疾病污名量表）测量、混合设计方差分析（ANOVA）与效应量检验，重点关注辅助机器人技术与任务适配。

**📊 数据集**

数据来源为75名德国低技能工业岗位员工（Prolific招募）的问卷回答；未使用公开数据集，而是构建自定义的四种情景（T1–T4）与两种残疾表现（I1、I2）。

**📈 对比分析**

通过比较四种情景的认知与行为污名分数，发现T1> T3/T4、T2> T3/T4，ANOVA显著主效应（F(3,219)=45.31，p<.001，η²=0.51），效应量均在0.26–1.33之间，表明辅助机器人显著降低污名。

**⚠️ 局限性**

限制包括仅研究辅助机器人而非其他辅助设备、样本仅来自德国文化、样本量有限导致统计功效不足、未量化污名降低的实际意义。

---

## 243. FlyAware: Inertia-Aware Aerial Manipulation via Vision-Based Estimation and Post-Grasp Adaptation

**arXiv ID:** 2601.22686 | [PDF](https://arxiv.org/pdf/2601.22686v1)

**作者:** Biyu Ye `[一作]` (Sun Yat-sen University), Ximin Lyu `[通讯]` (Differential Robotics Technology Company)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一套基于视觉与语言预估惯性参数、实时适配与自适应控制的无人机抓取与搬运系统，完成从预感知到抓取后即时更新惯性并调度控制增益的闭环流程。

**💡 创新点**

创新点包括：①双阶段视觉预估+抓取后反馈适配的惯性估计框架；②惯性感知自适应控制（IAGS）的增益调度策略；③利用 GPT 多模推理实现零样本体积与惯量修正；④在频域对不确定惯性进行稳健性分析。

**🔧 技术方法**

采用 RGB‑D 视觉 + 自然语言感知（Grounded SAM、IST‑Net、GPT API）+ DOB 适配器 + PID+增益调度自适应控制 + 频域 μ‑分析 + PX4/L1Quad 飞控 + Delta 机械臂运动学。

**📊 数据集**

实验使用自制的 8 个日常物体（箱子、罐子、钢杯等）进行真实飞行；视觉模型使用预训练的 SAM、IST‑Net 与 GPT API（不使用公开数据集）。

**📈 对比分析**

与 PX4 基线和 L1Quad 进行对比；在悬停、抓取、搬运等 8 物体任务中，位置/姿态 RMSE 分别降低 28.2%/29.4%，峰值误差降低 30.9%/28.5%；惯性估计误差在 2 s 内 <3%，比传统 20‑27 s 方法提升约 10 倍。

**⚠️ 局限性**

局限性：需依赖外部网络获取 GPT 结果；对极端形状或尺寸误差仍有一定鲁棒性不足；仅验证室内光照与小型物体；未测试多物体、户外高风速与复杂动力学耦合情况。

---

## 244. Beyond Medical Chatbots: Meddollina and the Rise of Continuous Clinical Intelligence

**arXiv ID:** 2601.22645 | [PDF](https://arxiv.org/pdf/2601.22645v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 245. What can Computer Vision learn from Ranganathan?

**arXiv ID:** 2601.22634 | [PDF](https://arxiv.org/pdf/2601.22634v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 246. Fire on Motion: Optimizing Video Pass-bands for Efficient Spiking Action Recognition

**arXiv ID:** 2601.22675 | [PDF](https://arxiv.org/pdf/2601.22675v1)

**作者:** Shuhan Ye `[一作]` (Ningbo University), Xudong Jiang `[通讯]` (Nanyang Technological University)

**通讯引用:** 15468 | [OpenAlex ID](https://openalex.org/A5085533260)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一个可插拔的 Pass-Band Optimizer（PBO），通过在脉冲神经网络（SNN）的膜积分前添加一个可学习的两点差分滤波器，自动调节时间通带以匹配视频动作和异常检测任务的频率特征；

**💡 创新点**

创新点在于首次将频域通带匹配问题与SNN的低通滤波特性联系起来，并通过轻量级可学习的周期性系数实现任务自适应的通带优化；

**🔧 技术方法**

使用了离散时间 Leaky Integrate-and-Fire（LIF）模型、频域分析、周期性可学习滤波器、强度与梯度一致性正则化，以及常见的 SNN 变换器和注意力架构；

**📊 数据集**

在 RGB 与事件（DVS）配对的数据集 UCF101-CEP、HMDB51-CEP、HARDVS 以及 UCF-Crime-CEP 上进行评估；

**📈 对比分析**

与多种基准（如 Spikformer、SD-Transformer、SCA、WeiAttn、CMCI、S-CMRL 等）相比，PBO 在动作识别任务中提升约10%准确率，在多模态融合和视频异常检测任务中亦实现显著性能提升，同时能量消耗保持低；

**⚠️ 局限性**

局限性包括对不同网络结构的适配需要进一步验证、对更长时序视频的通带调节可能需要更高阶滤波器，以及在极端稀疏或噪声环境下的鲁棒性尚待深入研究。

---

## 247. Ethical Risks of Large Language Models in Medical Consultation: An Assessment Based on Reproductive Ethics

**arXiv ID:** 2601.22621 | [PDF](https://arxiv.org/pdf/2601.22621v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 248. VisionTrim: Unified Vision Token Compression for Training-Free MLLM Acceleration

**arXiv ID:** 2601.22674 | [PDF](https://arxiv.org/pdf/2601.22674v1)

**作者:** Hanxun Yu `[一作]` (Zhejiang University), Jianke Zhu `[通讯]` (Udeer.ai)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 VisionTrim 框架，实现训练无关的视觉 token 压缩，统一在视觉编码和 LLM 解码阶段进行高效推理。

**💡 创新点**

创新点在于双插件模块：DVTS 结合全局语义与局部空间连续性进行 token 选择；TGVC 利用文本引导完成视觉 token 的补全和对齐，形成端到端的压缩与对齐方案。

**🔧 技术方法**

采用 [CLS] 注意力、局部 Token 亲和度 LTAM、方差自适应加权、文本-视觉相似度聚类、双阶段压缩策略等技术。

**📊 数据集**

使用 10 个图像基准（GQA、VQA‑V2、VizWiz、POPE、MMBench、MME、MM‑Vet、SQA、Text‑VQA）以及 4 个视频基准（TGIF、MSVD、MSRVTT、ActivityNet）进行评估。

**📈 对比分析**

与 SparseVLM、VisionZip、PyramidDrop、VScan 等方法对比，在 88.9% token 压缩率下保持 98% 以上原性能，且在多模态任务中多项指标均优于对手；同时显著降低 FLOPs、内存占用和推理时延。

**⚠️ 局限性**

局限性：在极高压缩比例下仍有轻微性能下降，未实现完全无损压缩，未来需进一步挖掘视觉 token 冗余与无损压缩技术。

---

## 249. Unsupervised Synthetic Image Attribution: Alignment and Disentanglement

**arXiv ID:** 2601.22663 | [PDF](https://arxiv.org/pdf/2601.22663v1)

**作者:** Zongfang Liu `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Kun Zhang `[通讯]` (Carnegie Mellon University)

**通讯引用:** 38919 | [OpenAlex ID](https://openalex.org/A5100342359)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出无监督的合成图像归因框架 Alignment and Disentanglement，避免了配对标注的需求。

**💡 创新点**

通过对比自监督对齐与 ICA 解耦实现无监督的 CCA，突破了需配对数据的限制。

**🔧 技术方法**

采用对比自监督学习（如 MoCo、DINO 等）与信息熵最大化的 ICA 以及正交约束等技术。

**📊 数据集**

在 AbC（Attribution by Customization）基准上进行实验，包含 400+ 万合成图像及其原始训练样本。

**📈 对比分析**

与监督、伪标签、预训练等方法对比，A&D 在多数后端实现 Recall@5/mAP 超过或接近监督方法，尤其在 DINO/MoCo 后端表现最佳。

**⚠️ 局限性**

依赖预训练 backbone 的跨域对齐，难以处理大域间差距，需进一步改进跨域自监督对齐能力。

---

## 250. Time-Annealed Perturbation Sampling: Diverse Generation for Diffusion Language Models

**arXiv ID:** 2601.22629 | [PDF](https://arxiv.org/pdf/2601.22629v1)

**作者:** Jingxuan Wu `[一作]` (University of North Carolina at Chapel Hill), Yang You `[通讯]` (National University of Singapore)

**通讯引用:** 3811 | [OpenAlex ID](https://openalex.org/A5100658705)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Time‑Annealed Perturbation Sampling（TAPS）——一种训练无关的推理策略，利用扩散语言模型在不同时间步的“分工”来增强文本多样性。

**💡 创新点**

创新点在于首次揭示扩散语言模型的时间分工：早期去噪决定全局语义结构，后期去噪聚焦局部词汇细化；基于此设计了时间退火扰动采样，在推理早期鼓励语义分支，后期逐步降低扰动以保持流畅与指令遵循。

**🔧 技术方法**

技术包括扩散语言模型（Diffusion‑LM）、时间退火扰动采样（TAPS）以及在非自回归和半自回归架构（如LLaDA、TraDo）上的推理实现。

**📊 数据集**

使用了创意写作与推理基准（Creative Writing and Reasoning benchmarks）以及LLaDA和TraDo两个扩散语言模型作为实验平台。

**📈 对比分析**

与传统扩散模型推理方法（不使用TAPS）进行对比，实验表明TAPS在多样性指标上显著提升，同时生成质量与流畅度保持不变或略有提升。

**⚠️ 局限性**

局限性包括：对非自回归/半自回归模型的适用性有限，可能在极端指令严格或对流畅度要求极高的场景下仍出现语义漂移；且尚未在大规模、专业领域数据上进行充分验证。

---

## 251. Computing Dominating Sets in Disk Graphs with Centers in Convex Position

**arXiv ID:** 2601.22609 | [PDF](https://arxiv.org/pdf/2601.22609v1)

**作者:** Anastasiia Tkachenko `[一作]` (University of Utah), Haitao Wang `[通讯]` (University of Utah)

**通讯引用:** 37801 | [OpenAlex ID](https://openalex.org/A5100396117)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

在凸位置的点集下，提出了求解圆图最小占据集（最小加权与最小非加权）的多项式时间算法；

**💡 创新点**

首次利用“线可分性”结构与rank‑t中心概念，将一般圆图的占据集问题转化为动态规划问题，突破了已知的NP‑难限制；

**🔧 技术方法**

核心技术包括：加权最近圆Voronoi图构造、最小值封闭子列表查询、基于动态规划与贪心的子列表合并；

**📊 数据集**

论文仅给出理论分析与算法实现说明，并未使用实测数据集；

**📈 对比分析**

与以往针对单位圆图的 O(n³log²n) 与 O(k·nlogn) 方法相比，本工作在凸位置假设下实现了 O(n⁵log²n)（无权）和 O(k²nlog²n)（加权）多项式时间；

**⚠️ 局限性**

限制：仅适用于凸位置点，算法复杂度仍较高（尤其是 O(n⁵) 对大规模实例不实用），且未对一般位置点给出有效方案。

---

## 252. Parameter conditioned interpretable U-Net surrogate model for data-driven predictions of convection-diffusion-reaction processes

**arXiv ID:** 2601.22654 | [PDF](https://arxiv.org/pdf/2601.22654v1)

**作者:** Michael Urs Lars Kastor `[一作]` (RPTU University Kaiserslautern Landau), Nicolas Ralph Gauger `[通讯]` (RPTU University Kaiserslautern Landau)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了一个二阶有限差分+Runge‑Kutta-Fehlberg 数值求解器，并基于其生成的仿真数据训练了一种参数条件化的 U‑Net 近似模型，用于快速预测二维表型空间中的非线性非稳态输运反应方程的解。

**💡 创新点**

创新点包括：① 将 Feature‑wise Linear Modulation (FiLM) 与 CoordConv 结合到 U‑Net 中，使单一网络可处理不同 PDE 参数组合；② 采用残差块和 GroupNorm 提升训练稳定性；③ 通过规模归一化的 Huber 损失实现对不同幅值场的统一评估；④ 在测试集上系统性分析泛化误差，发现误差主要由 PDE 规制参数决定，而非初始场。

**🔧 技术方法**

使用技术：数值求解采用二阶空间差分、三阶 Runge‑Kutta‑Fehlberg 时间积分与误差控制；深度模型采用残差 U‑Net、FiLM、CoordConv、GroupNorm、LeakyReLU；训练使用 AdamW、余弦退火学习率、梯度裁剪和 Huber 损失；GPU 并行实现推理时间仅 ~11 ms/批次。

**📊 数据集**

数据集：通过 10 000 组随机 hill‑like 初始场与随机参数向量在 256×256 网格上生成最终解；测试集 2 500 组由 50 个初始场与 50 个参数向量交叉组合得到；训练时将数据下采样至 64×64。

**📈 对比分析**

方法比较：训练/验证损失收敛至 7.4×10⁻⁵/1.75×10⁻⁴；最大像素误差 0.08；相对误差低于 0.1 %；推理速度稳定，批量大小不影响 11 ms；泛化误差矩阵显示对参数向量的灵敏度远高于初始场。

**⚠️ 局限性**

局限性：对某些参数组合（PDE 规制区间）学习困难，误差与单个参数无强相关；模型仅在固定终止时间和低维参数空间内有效；未考虑积分项或更复杂的边界条件；仅在粗网格上训练，可能无法直接推广至更高分辨率；需要进一步研究更具挑战性的 PDE 规制与多尺度特性。

---

## 253. TSLM: Tree-Structured Language Modeling for Divergent Thinking

**arXiv ID:** 2601.22688 | [PDF](https://arxiv.org/pdf/2601.22688v1)

**作者:** Doyoung Kim `[一作]` (New York University), Minjoon Seo `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 2322 | [OpenAlex ID](https://openalex.org/A5087565126)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Tree-Structured Language Modeling（TSLM），让语言模型在一次前向传播中生成完整搜索树并进行系统性分支探索；

**💡 创新点**

创新点在于通过特殊标记的序列化方式，让模型内部化搜索树结构，避免外部多次独立采样，支持可控的分支扩展与搜索策略；

**🔧 技术方法**

使用Transformer基础模型（如Llama‑3‑8B），对搜索树进行序列化编码，采用标准交叉熵训练，训练时包括成功与失败分支；

**📊 数据集**

在结构化任务（Game of 24、Gridworld）和开放式推理任务（ProntoQA、GSM8K）上进行实验；

**📈 对比分析**

与传统顺序生成、Procedure Cloning、GRPO以及Tree‑of‑Thought对比；在结构化任务中TSLM达到100%（vs 17%/32%等），在Gridworld扩展至更大规模仍保持高准确率（91.5% vs 42.7%）；在开放任务中TSLM与或略优于ToT；同时TSLM在推理时间与样本效率上优于ToT；

**⚠️ 局限性**

局限在于目前实验主要基于小模型（8B级别），对更大规模模型与更复杂任务的泛化尚未充分验证；另外仍需进一步优化对不可解问题的识别阈值与搜索策略的自动化。

---

## 254. OOVDet: Low-Density Prior Learning for Zero-Shot Out-of-Vocabulary Object Detection

**arXiv ID:** 2601.22685 | [PDF](https://arxiv.org/pdf/2601.22685v1)

**作者:** Binyi Su `[一作]` (Hebei University of Technology), Haiyong Chen `[通讯]` (Hebei University of Technology)

**通讯引用:** 9724 | [OpenAlex ID](https://openalex.org/A5089644877)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `3855fcda-48ef-4070-a15e-803cd5c84d83` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种零样本OOV检测框架OOVDet，能在不训练OOV样本的情况下，准确识别预定义类别并可靠拒绝未知类别。

**💡 创新点**

创新点在于三大模块：①基于低密度区域的OOV提示合成（OPS）产生低概率文本提示；②Dirichlet梯度归因（DGA）挖掘高不确定度的伪OOV图像；③低密度先验约束（LPC）通过核密度估计实现OOV决策边界。

**🔧 技术方法**

使用Prompt学习、文本编码器、视觉-文本对齐、Gaussian混合模型、Dirichlet梯度归因、Gaussian核密度估计、两阶段检测器等技术。

**📊 数据集**

在OOV‑VOC、OOV‑COCO公开基准上验证，并在ObstacleTrack、ADE‑OoD、MVTec‑Anomaly、RoadAnomaly等真实世界数据集上做可视化。

**📈 对比分析**

与多种基线（RegionCLIP、OpenDet、EDL等）比较，OOVDet在OOV召回率、OOV mAP、WI、AOSE等指标均显著提升（如OOV‑VOC mAPOOV+2.7%、ROOV+8.7%、WI下降0.35%）。

**⚠️ 局限性**

局限包括对高质量文本提示和高维特征的依赖；当OOV样本与IV类别非常相似时，低密度假设可能失效；以及对超参数（如核宽度h、掩码比例）的敏感性。

---

## 255. From Horizontal Layering to Vertical Integration: A Comparative Study of the AI-Driven Software Development Paradigm

**arXiv ID:** 2601.22667 | [PDF](https://arxiv.org/pdf/2601.22667v1)

**作者:** Chi Zhang `[一作]` (Moximize), Ming Dong `[通讯]` (Shanghai Jiaotong University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过多案例对比研究，分析Generative AI在软件工程组织中的应用，比较传统企业与AI原生初创在从横向分层到纵向整合的组织转型与资源消耗变化。

**💡 创新点**

提出以人机协作效能为核心的优化目标，定义“超级员工”概念，并揭示AI失真效应对总要素生产率的重新分配，强调人机协作而非单纯的个人产出。

**🔧 技术方法**

利用大型语言模型（LLM）进行代码生成与执行层，构建人机协作框架并采用Conway Law与复杂度度量评估组织结构变化。

**📊 数据集**

以两案例项目（Project W与内部AI‑CRM）的实际资源利用、Git提交日志、会议记录和访谈文本为数据来源。

**📈 对比分析**

对照传统基准估算（功能点分析）与AI驱动实现的实际资源消耗，效率提升分别为约8.3×（案例A）和33×（案例B），表现出明显的规模效益。

**⚠️ 局限性**

样本局限于高绩效工程师、观察窗口短、技术债务与长期可维护性未验证、人机协作负荷可能导致疲劳，以及对LLM能力和API可用性的依赖。

---

## 256. Task-Aware LLM Council with Adaptive Decision Pathways for Decision Support

**arXiv ID:** 2601.22662 | [PDF](https://arxiv.org/pdf/2601.22662v1)

**作者:** Wei Zhu `[一作]` (Yunnan University), Kun Yue `[通讯]` (Yunnan University)

**通讯引用:** 1855 | [OpenAlex ID](https://openalex.org/A5041269918)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种任务感知的大模型议事会（TALC），通过动态路由不同LLM专家并结合双信号蒙特卡洛树搜索实现自适应规划。

**💡 创新点**

创新点在于（1）为每个LLM构建结构化的成功记忆档案，依据语义匹配实现任务感知的模型选择；（2）在MCTS中融合模型评价与记忆先验的双信号价值估计，动态调节搜索深度；（3）实现轻量级与高性能两种配置，兼顾可部署性与推理强度。

**🔧 技术方法**

使用的核心技术包括：结构化成功记忆片段（SMS）、语义相似度检索的路由机制、基于UCB的MCTS搜索、双信号价值融合、温度调节与自适应加权。

**📊 数据集**

在WebShop、HumanEval和Game of 24三个多模态、结构化推理基准上进行评估。

**📈 对比分析**

与单模型推理、ToT、RAP、LATS、MASTER等现有方法对比，TALC在所有基准上均获得更高的成功率/pass@1和更低的搜索代价；TALC‑Lite与GPT‑4o相近，TALC‑Pro在WebShop与HumanEval上刷新SOTA。

**⚠️ 局限性**

局限性包括：对成功记忆的实时收集与管理仍存在存储与检索成本；双信号融合权重调节需额外开销；在极大搜索空间或不确定环境下，MCTS仍可能面临深度与计算量的权衡。

---

## 257. Layerwise Progressive Freezing Enables STE-Free Training of Deep Binary Neural Networks

**arXiv ID:** 2601.22660 | [PDF](https://arxiv.org/pdf/2601.22660v1)

**作者:** Evan Gibson Smith `[一作]` (Worcester Polytechnic Institute), Bashima Islam `[通讯]` (Worcester Polytechnic Institute)

**通讯引用:** 646 | [OpenAlex ID](https://openalex.org/A5035909313)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了一种不使用直通估计器（STE）的进阶冻结训练方法 StoMPP，用于从头开始训练二进制网络。

**💡 创新点**

创新点在于提出层级进阶冻结与随机遮罩的 StoMPP，解决全二进制网络因激活冻结导致梯度阻塞的问题，并在更深网络上实现更高精度。

**🔧 技术方法**

使用的技术包括层级进阶冻结、随机遮罩软刷新、无 STE 梯度传播、三次方冻结进度曲线、无权重衰减常数学习率训练等。

**📊 数据集**

实验数据集为 CIFAR‑10、CIFAR‑100 和 ImageNet，网络结构为 ResNet18/34/50 以及 Bi‑Real Net。

**📈 对比分析**

与标准 BinaryConnect/STE 基线在相同训练配置下对比，StoMPP 在全二进制网络上提升 18%（ResNet‑50 CIFAR‑10）、13.5%（CIFAR‑100）和 3.8%（ImageNet），并表现出更好的深度扩展性。

**⚠️ 局限性**

局限性包括对冻结进度与刷新率的手动调参需求、在更大模型或更复杂结构上的推广尚未充分验证，以及与其他 STE 改进方法组合效果有限。

---

## 258. The Semantic Trap: Do Fine-tuned LLMs Learn Vulnerability Root Cause or Just Functional Pattern?

**arXiv ID:** 2601.22655 | [PDF](https://arxiv.org/pdf/2601.22655v1)

**作者:** Feiyang Huang `[一作]` (Zhejiang University), Yang Liu `[通讯]` (Nanyang Technological University)

**通讯引用:** 48649 | [OpenAlex ID](https://openalex.org/A5100355773)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 TrapEval 评估框架，通过构造 V2P（漏洞‑补丁）和 V2N（漏洞‑无害）两类数据集，系统评估并揭示微调后 LLM 在漏洞检测中陷入语义陷阱的现象。

**💡 创新点**

创新点在于：① 将漏洞检测拆解为功能模式与根本漏洞逻辑的对抗性评估；② 通过语义保持的扰动和 CodeBLEU 语义距离分组，量化 LLM 对细微逻辑差异的辨识能力；③ 明确指出微调提升并不等同于真正的漏洞根因学习。

**🔧 技术方法**

技术方法包括：LoRA 参数高效微调、vLLM 推理、语义保持的代码变换、CodeBLEU 语义距离计算，以及交叉数据集评估与鲁棒性测试。

**📊 数据集**

使用了三大开源漏洞数据集（DiverseVul、PrimeVul、CVEFixes）整合后构建的 436,489 条样本数据集，进一步拆分为 V2P 与 V2N 两种形式。

**📈 对比分析**

通过对五大 7–8B 级 LLM（Qwen、Llama、DeepSeek 系列）在 V2P/V2N、扰动、语义距离三个维度进行横向对比，结果显示微调后模型在 V2P 上 F1 仅提升 5–15%，而在 V2N 上性能大幅提升；在语义保持扰动下 Recall 与 F1 均显著下降，且性能随 CodeBLEU 下降（语义差距增大）而提升，表明模型更多依赖功能模式而非根本漏洞逻辑。

**⚠️ 局限性**

局限性包括：仅评估 7–8B 规模模型，较大模型可能表现不同；数据集仍可能存在标签噪声；评估聚焦于 C/C++，跨语言通用性未验证；微调方法未尝试结构化程序信息，导致难以捕捉细粒度逻辑。

---

## 259. DART-ing Through the Drift: Dynamic Tracing of Knowledge Neurons for Adaptive Inference-Time Pruning

**arXiv ID:** 2601.22632 | [PDF](https://arxiv.org/pdf/2601.22632v1)

**作者:** Abhishek Tyagi `[一作]` (National University of Singapore), Xuanyao Fong `[通讯]` (National University of Singapore)

**通讯引用:** 2974 | [OpenAlex ID](https://openalex.org/A5085588788)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种无训练、动态稀疏化框架 DART，利用上下文感知的神经元选择与知识漂移检测，在推理时自适应地更新 LLM 的 FFN 子层掩码。

**💡 创新点**

创新点在于将层级敏感度分配与实时注意力分布漂移监测结合，首次实现了在推理过程中根据语义上下文动态更新神经元子空间，解决了传统静态稀疏化导致的知识漂移问题。

**🔧 技术方法**

技术上采用累计激活评分生成结构化掩码、基于注意力向量相似度的漂移阈值检测、层级敏感度 (S_t,τ) 与深度因子 (D^l) 计算层级稀疏比例，并在发现漂移时触发掩码重构。

**📊 数据集**

实验数据集包括 LLaMA-3.2‑3B / LLaMA-3.1‑8B，在 WikiText、C4、BoolQ、RTE、HellaSwag、WinoGrande、ARC‑e/c、OBQA、MMLU、GPQA、MedMCQA 以及多主题摘要基准 CNN/DailyMail、Multi‑News、GovReport 等。

**📈 对比分析**

与 Wanda、DejaVu 等静态/动态稀疏化基线比较；在 70% FFN 稀疏率下，DART 在多项零样本与多样本任务上平均提升 10–20% 以上，摘要任务的 ROUGE‑L 提升至 3 倍；整体推理速度提升至 0.1% FLOPs，内存占用低于 10 MB。

**⚠️ 局限性**

局限性包括对注意力相似度阈值的手工调参需求、极端长序列或频繁主题跳转时可能产生误报导致不必要的掩码更新，以及对非 FFN 子层的稀疏化尚未覆盖。

---

## 260. SYMPHONY: Synergistic Multi-agent Planning with Heterogeneous Language Model Assembly

**arXiv ID:** 2601.22623 | [PDF](https://arxiv.org/pdf/2601.22623v1)

**作者:** Wei Zhu `[一作]` (Yunnan University), Kun Yue `[通讯]` (Yunnan University)

**通讯引用:** 1855 | [OpenAlex ID](https://openalex.org/A5041269918)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 SYMPHONY，一种将多模异构语言模型与 Monte Carlo Tree Search (MCTS) 结合的多智能体规划框架，用以提升复杂任务的探索多样性与规划效率。

**💡 创新点**

创新点在于：①构建异构语言模型池以增强搜索分支多样性；②使用基于 UCB 的动态智能体调度，实现上下文感知的资源分配；③引入熵调制置信度评分 (EMCS) 校准价值评估；④通过共享自然语言反思实现无参模型的持续适应。

**🔧 技术方法**

核心技术包括：多模异构 LLM、MCTS、UCB 调度策略、EMCS 评估、分布式反思记忆、以及与不同硬件配置（本地 vs 云 API）的兼容部署。

**📊 数据集**

实验数据集涵盖三类任务：多跳问答 HotpotQA、WebShop 目标驱动交互、以及代码生成 MBPP（Python 与 Rust）。

**📈 对比分析**

与多类基线（线性推理、反馈驱动、结构化搜索、单模 LLM、以及 MASTER 之类的多智能体框架）对比，SYMPHONY 在所有任务上均取得显著提升：HotpotQA- SYMPHONY-L 0.79 EM 超越 MASTER 0.76；WebShop 0.88 SR 超越 MASTER 0.80；MBPP Pass@1 达到 0.965/0.974，超过 MetaGPT、AgentCoder 等先进方法。

**⚠️ 局限性**

局限性包括：①对超参（如 UCB 探索系数、MCTS 预算）仍需人工调优；②依赖多模 LLM 的协同调用，仍会产生计算与通信成本；③在极度噪声或无结构环境下的鲁棒性尚未充分验证；④反思记忆管理方式可能在长时间运行时产生信息冗余。

---

## 261. EntroCut: Entropy-Guided Adaptive Truncation for Efficient Chain-of-Thought Reasoning in Small-scale Large Reasoning Models

**arXiv ID:** 2601.22617 | [PDF](https://arxiv.org/pdf/2601.22617v1)

**作者:** Hongxi Yan `[一作]` (Beihang University), Yunhong Wang `[通讯]` (Beihang University)

**通讯引用:** 13813 | [OpenAlex ID](https://openalex.org/A5115589096)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种训练免费方法EntroCut，利用模型输出早期步骤的熵值动态截断链式思考，从而提升大型推理模型的推理效率；

**💡 创新点**

创新点在于发现熵值能够早期区分正确与错误答案，并基于此构建熵引导的自适应截断机制以及统一衡量效率与准确度平衡的新指标EPR；

**🔧 技术方法**

技术实现包括熵探针、阈值触发截断、反思词检测、EPR计算以及针对不同模型（DS‑1.5B、DS‑7B）的温度/阈值设置；

**📊 数据集**

实验数据集覆盖四类数学推理任务：AIME24、AIME25、MATH500 与 AMC23；

**📈 对比分析**

与 Vanilla、NOWAIT、TIP、DEER 等基准相比，EntroCut 在保持甚至提升准确率的同时将 token 使用减少约30–40%，EPR 平均提升 9.5（DS‑1.5B）或 0.9（DS‑7B），表现优异；

**⚠️ 局限性**

局限性包括对阈值的手动调参、仅在数学推理任务上验证、缺乏跨领域泛化与对模型内部反思机制的深入分析。

---

## 262. Human-Centered Explainability in AI-Enhanced UI Security Interfaces: Designing Trustworthy Copilots for Cybersecurity Analysts

**arXiv ID:** 2601.22653 | [PDF](https://arxiv.org/pdf/2601.22653v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 263. Do Transformers Have the Ability for Periodicity Generalization?

**arXiv ID:** 2601.22690 | [PDF](https://arxiv.org/pdf/2601.22690v1)

**作者:** Huanyu Liu `[一作]` (Peking University), Tongxuan Liu `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究Transformer在周期性泛化中的局限性，并提出统一的群论解释与可控合成周期性数据集。

**💡 创新点**

首次将周期性与推理通过群论统一解释，构造了涵盖Hollow与Extrapolation两种OOD场景的合成周期性基准。

**🔧 技术方法**

采用Transformer（RoPE）、FANFormer、Mamba、RWKV等架构，并通过群论理论分析其对规则周期性与合成周期性的捕获能力。

**📊 数据集**

使用自定义的Composite Periodicity数据集，包含50k训练样本和3k测试样本，测试集划分为ID、Hollow、Extrapolation三种情形。

**📈 对比分析**

与多种基线对比，Transformer和FANFormer在ID上可达≈90%准确率，但在Hollow和Extrapolation上仅≈30%和≈20%准确率；Mamba/RWKV在ID上表现差，OOB泛化更差，说明现有模型缺乏对合成周期性的真正泛化能力。

**⚠️ 局限性**

局限性包括未考虑链式推理或外部记忆机制，且实验仅针对RoPE的Transformer，缺乏对更通用自注意力实现的验证。

---

## 264. Visual Personalization Turing Test

**arXiv ID:** 2601.22680 | [PDF](https://arxiv.org/pdf/2601.22680v1)

**作者:** Rameen Abdal `[一作]` (Snap Research), Kuan-Chieh Jackson Wang `[通讯]` (Snap Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出Visual Personalization Turing Test (VPTT)，用于评估生成内容是否与特定人群的视觉表达难以区分；

**💡 创新点**

创新点在于以可感知的“可区分性”而非身份复制为目标，构建VPTT框架、VPTT‑Bench 10k合成人格、检索增强生成方法VPRAG，以及可作为自动化评估的文本代理VPTT_score；

**🔧 技术方法**

使用层次检索、温度软化、类别配额、Prompt Composition以及可学习的反馈网络等技术，实现无微调、低成本的检索增强生成；

**📊 数据集**

使用自生成的10,000个合成人格数据集VPTT‑Bench（每人30条文本描述的视觉资产），并渲染1,000人可视库，保证数据规模与隐私安全；

**📈 对比分析**

与Baseline、Persona Only、BRAG、高成本LoRA/Flux/VIPER等多种方法对比，VPRAG在生成和编辑任务中在VPTT_score、VLM评估及人类评价上均取得最高分，显著优于传统方法；

**⚠️ 局限性**

主要局限包括合成与真实用户的差距、目前仅支持图像而非视频/3D、对结构保持缺乏保证，以及多模态输入支持有限。

---

## 265. Beyond Fixed Rounds: Data-Free Early Stopping for Practical Federated Learning

**arXiv ID:** 2601.22669 | [PDF](https://arxiv.org/pdf/2601.22669v1)

**作者:** Youngjoon Lee `[一作]` (Korea Advanced Institute of Science and Technology), Joonhyuk Kang `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 2194 | [OpenAlex ID](https://openalex.org/A5062994489)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种无验证数据的早停框架，通过监控全局模型任务向量的增长速率决定停止训练的最佳时机；

**💡 创新点**

创新点在于利用服务器端参数（任务向量增长率）实现完全数据无关的早停，而非传统依赖验证集或固定轮次的方法；

**🔧 技术方法**

核心技术包括任务向量定义、增长速率计算与阈值、耐心计数器等基于全局参数的模型驱动判定；

**📊 数据集**

使用医学影像分类数据集：皮肤病变图像集与血液细胞图像集，采用100个客户端、10个本地客户端的分布式设置；

**📈 对比分析**

与多种先进联邦学习方法（FedAvg、FedProx、SCAFFOLD、FedSAM等）进行比较，实验显示该框架在保持或提升验证基线性能的同时，平均多训练约47/20轮，取得了超过12.5%/10.3%的准确率提升；

**⚠️ 局限性**

局限性包括对阈值τ的敏感性需手工调参、在极端非IID环境下仍可能出现过早或过晚停止，以及在极小/极大训练轮数下的性能不确定性。

---

## 266. GUDA: Counterfactual Group-wise Training Data Attribution for Diffusion Models via Unlearning

**arXiv ID:** 2601.22651 | [PDF](https://arxiv.org/pdf/2601.22651v1)

**作者:** Naoki Murata `[一作]` (Sony AI), Yuki Mitsufuji `[通讯]` (Sony Group Corporation)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于机器未学习的扩散模型组级数据归因框架，使用ELBO差分来衡量各训练组的贡献；

**💡 创新点**

创新点在于用未学习替代昂贵的LOGO重训练，并通过ReTrack与条件重定向的未学习损失实现对不同生成设置的高效估计；

**🔧 技术方法**

采用机器未学习、ELBO（低阶对数似然下界）、ReTrack重要性采样、条件文本-图像的anchor重定向等技术；

**📊 数据集**

使用CIFAR‑10（10类）和UnlearnCanvas（60种艺术风格，Stable Diffusion 1.5）等数据集进行验证；

**📈 对比分析**

与CLIPA、DAS、D‑TRAK、TRAK等基线对比，在CIFAR‑10上Top‑1≈72%，NDCG@3≈0.68；在UnlearnCanvas上Top‑1≈45%，NDCG@3≈0.73；相较于LOGO重训练速度提升约100×；

**⚠️ 局限性**

局限性包括：仅针对互斥训练组，无法处理重叠组；依赖ELBO近似；需要预先训练/未学习多模型，存储与计算成本仍随组数上升；未针对更复杂的条件/多模态归因进行广泛评估。

---

## 267. Pushing the Boundaries of Natural Reasoning: Interleaved Bonus from Formal-Logic Verification

**arXiv ID:** 2601.22642 | [PDF](https://arxiv.org/pdf/2601.22642v1)

**作者:** Chuxue Cao `[一作]` (Hong Kong University of Science and Technology), Yike Guo `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 18510 | [OpenAlex ID](https://openalex.org/A5045081171)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个在大型语言模型推理过程中动态嵌入形式逻辑验证的框架，并通过两阶段训练（监督微调+强化学习）实现。

**💡 创新点**

创新点在于：①将形式逻辑验证实时嵌入推理链，形成“自然-形式-验证”交互；②设计执行验证的合成数据管线；③提出基于组相对奖励的策略优化（GRPO）来强化逻辑一致性。

**🔧 技术方法**

采用了形式逻辑验证工具（SAT/SMT/定理证明器）、链式思考（CoT）、监督微调、强化学习（GRPO）、工具调用与执行反馈、语义等价检查等技术。

**📊 数据集**

使用的数据集包括 WebInstruct‑Verified、K&K、NuminaMath‑TIR 进行训练，评测时使用 KOR‑Bench、BBH、MATH‑500、AIME‑2024、GPQA‑Diamond 与 TheoremQA 等六大基准。

**📈 对比分析**

与多种基线（Base、Natural‑SFT/RL、RLPR、SimpleRL‑Zoo、General‑Reasoner、ZeroTIR、SimpleTIR 等）进行对比，7B 模型平均提升 10.4%，14B 模型平均提升 14.2%；在 AIME‑2024 达到 30.2%（远超 17.5%），MATH‑500 81.4%，TheoremQA 63.5%，在逻辑、数学和通用推理任务上均显著优于 SOTA。

**⚠️ 局限性**

局限性包括：①形式验证开销较大，计算成本上升；②对特定逻辑求解器和自动形式化的依赖，可能在非符号化或高度数值任务上表现不佳；③训练数据规模有限（约 17k），可能影响模型的泛化；④在某些基准（如 GPQA‑Diamond）表现略逊。

---

## 268. ScholarPeer: A Context-Aware Multi-Agent Framework for Automated Peer Review

**arXiv ID:** 2601.22638 | [PDF](https://arxiv.org/pdf/2601.22638v1)

**作者:** Palash Goyal `[一作]` (Google), Jinsung Yoon `[通讯]` (Google)

**通讯引用:** 6143 | [OpenAlex ID](https://openalex.org/A5002289527)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了 ScholarPeer，一种多代理框架，模拟资深研究员的认知流程，利用动态网络检索为学术论文生成深度评审。

**💡 创新点**

创新点在于引入历史学家、基线侦测和问答代理，构建“领域叙事”并主动验证主张，从而克服传统静态LLM评审的“真空”问题。

**🔧 技术方法**

技术包括多代理协同、检索增强生成（RAG）、Google搜索驱动的文献检索、领域叙事压缩、基线侦测、主动问答、以及与 Gemini 3 Pro/Claude Sonnet 4.5 等大型 LLM 配合的评审生成。

**📊 数据集**

数据集主要使用 DeepReview‑13K（ICLR 2024/25）以及公开的评审数据与各类基线模型。

**📈 对比分析**

在 LLM‑judge 与人工对比实验中，ScholarPeer 在五个评审维度上实现 91.5% 的胜率、6.14 的“Score”超过人类专家 5.0，并在多样性与与人类评分的 Spearman 相关性上均优于单体与多体基线。

**⚠️ 局限性**

局限包括对外部检索的依赖导致隐私与保密风险、对大模型的高计算成本、以及在内部逻辑一致性检查上略逊于 AI Scientist v2。

---

## 269. PEFT-MuTS: A Multivariate Parameter-Efficient Fine-Tuning Framework for Remaining Useful Life Prediction based on Cross-domain Time Series Representation Model

**arXiv ID:** 2601.22631 | [PDF](https://arxiv.org/pdf/2601.22631v1)

**作者:** En Fu `[一作]` (University of Science and Technology Beijing), Kaixiang Peng `[通讯]` (University of Science and Technology Beijing)

**通讯引用:** 5602 | [OpenAlex ID](https://openalex.org/A5016153320)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了 PEFT‑MuTS，一种基于跨域时间序列预训练与参数高效微调的极少样本剩余使用寿命预测框架。

**💡 创新点**

创新点在于通过独立特征调节网络和元变量低秩融合将单变量预训练模型迁移到多变量任务，并采用零初始化回归器提升微调稳定性。

**🔧 技术方法**

使用自监督时间序列预训练（FEI/SimMTM）、ResNet‑18骨干、LoRA 风格的参数高效微调、低秩元变量融合和零初始化线性回归。

**📊 数据集**

实验基于 C‑MAPSS（FD002、FD004）航空发动机数据和 XJTU‑SY 轴承数据。

**📈 对比分析**

与传统全微调、线性微调、Domain‑Tuning、Meta‑Learning 等方法对比，PEFT‑MuTS 在少于 1% 样本的极少样本场景下在 MAE、RMSE、MAPE 等指标上均优于对照组，显著提升预测精度。

**⚠️ 局限性**

局限在于对跨域预训练数据的可获取性依赖，且在变量极少的任务中元变量融合效果有限，需要进一步验证跨设备可迁移性。

---

## 270. TTCS: Test-Time Curriculum Synthesis for Self-Evolving

**arXiv ID:** 2601.22628 | [PDF](https://arxiv.org/pdf/2601.22628v1)

**作者:** Chengyi Yang `[一作]` (Xiamen University), Jinsong Su `[通讯]` (Xiamen University)

**通讯引用:** 3966 | [OpenAlex ID](https://openalex.org/A5066326238)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种共进化的测试时训练框架 TTCS，利用合成器根据测试题生成可控难度的变体，解算器在这些变体上在线自我演化，并通过自一致性奖励来提升推理性能。

**💡 创新点**

创新点在于：①通过能力感知的合成器生成与解算器当前能力边界相匹配的训练题，从而解决了原有方法中伪标签不可靠和难题缺乏可学习样本的问题；②将合成器与解算器以循环方式共进化，形成动态、自适应的课程体系；③采用 GRPO 与自一致性奖励实现无标签在线优化。

**🔧 技术方法**

技术手段包括：Group Relative Policy Optimization (GRPO)、自一致性奖励、能力自适应奖励与相似性惩罚的合成器奖励设计、在线数据过滤、批量多样性增强，以及在测试时无标签的自监督学习。

**📊 数据集**

使用的评测数据集：数学基准 AMC23、AIME24/25、MATH‑500、Minerva、OlympiadBench；通用推理基准 MMLU‑Pro、SuperGPQA；并在多种预训练模型（Qwen2.5‑Math‑1.5B、Qwen2.5‑Math‑7B、Qwen3‑4B‑Base）上进行实验。

**📈 对比分析**

与基线（预训练模型、Self‑Consistency、R‑Zero、TTRL）对比，TTCS 在所有测试集上均显著提升，特别是在 AIME24/25 等难度高的数学题上提升 6–7 分，平均提升 20–50 分，展现出在困难推理任务和跨域迁移上的强大效果。

**⚠️ 局限性**

局限性包括：①仍依赖测试集的规模与多样性，极小数据集下效果受限；②合成器生成的变体质量受预训练模型限制，可能出现过拟合或重复；③采用多数投票的伪标签在极难题上仍可能产生噪声；④目前仅在数学和部分通用推理任务验证，缺乏对更复杂或开放式任务的适用性研究。

---

## 271. Elderly HealthMag: Systematic Building and Calibrating a Tool for Identifying and Evaluating Senior User Digital Health Software

**arXiv ID:** 2601.22627 | [PDF](https://arxiv.org/pdf/2601.22627v1)

**作者:** Yuqing Xiao `[一作]` (Monash University), Elizabeth Manias `[通讯]` (Monash University)

**通讯引用:** 13900 | [OpenAlex ID](https://openalex.org/A5033367129)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出并验证了一种基于InclusiveMag框架的HealthMag和Elderly HealthMag工具，用于识别和评估数字健康软件中的健康与年龄相关的可用性缺陷；

**💡 创新点**

首次将健康条件视为可操作的交互维度，设计五个健康Facet（动机、技术熟练度、接收护理、自我效能、信任与隐私），并与年龄Facet融合形成双镜头方法；

**🔧 技术方法**

采用系统文献综述、专家调研与LLM（ChatGPT‑4.0/DeepSeek）生成数据驱动的Persona，再通过认知走查（cognitive walkthrough）评估应用；

**📊 数据集**

使用来自澳大利亚健康与福利机构（AIHW）的年龄、语言、性别分布统计以及130余篇相关文献中的用户需求引用；

**📈 对比分析**

与传统可用性评估方法对比，借助SUS量表与认知走查记录的易用性得分显示HealthMag能够更细粒度地捕捉健康与年龄交叉导致的可用性瓶颈，提升设计改进的针对性；

**⚠️ 局限性**

局限性包括：① 受专家面谈样本规模和专业背景限制；② LLM生成Persona可能存在幻觉和刻板印象；③ 仅在两款常见药物管理App上验证，需扩展至更广泛的数字健康场景验证。

---

## 272. Layer-wise Swapping for Generalizable Multilingual Safety

**arXiv ID:** 2601.22620 | [PDF](https://arxiv.org/pdf/2601.22620v1)

**作者:** Hyunseo Shin `[一作]` (University of Seoul), Wonseok Hwang `[通讯]` (University of Seoul)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研发了一种训练无关的安全感知层交换方法，可将英语安全专家模型的安全对齐迁移到低资源语言专家模型。

**💡 创新点**

通过任务向量化层/模块级融合，并自动根据安全与多语言专门化比例选择或混合注意力与MLP模块，实现模块级动态交换，兼顾安全和通用能力。

**🔧 技术方法**

使用任务向量（task vector）对比、层级/模块级参数差分、自动重要性评分与阈值/α混合策略，并在 LLaMA 3.1 8B 与 Qwen 3 8B 上进行指令微调。

**📊 数据集**

使用英语安全指令集（约2k样本）和低资源语言指令集（70k–80k样本）涵盖韩语、孟加拉语、斯瓦希里语、泰卢固语；安全评测采用 MultiJail，通用评测采用 MMMLU、BELEBELE 与 MGSM。

**📈 对比分析**

与基线（基础模型、单独安全/语言微调、混合微调、Task Arithmetic、TIES、DARE、传统层交换）对比；实验表明在 LLaMA/Qwen 上安全失误率显著下降，且在通用任务的 EM 分数保持或提升，模块级交换优于层级交换。

**⚠️ 局限性**

评估依赖 LLM 评判器，可能存在语言偏差；当前实现为样本无关的参数合并，未考虑上下文感知交换。

---

## 273. TTSA3R: Training-Free Temporal-Spatial Adaptive Persistent State for Streaming 3D Reconstruction

**arXiv ID:** 2601.22615 | [PDF](https://arxiv.org/pdf/2601.22615v1)

**作者:** Zhijie Zheng `[一作]` (University of California), Jiawei Zhang `[通讯]` (University of California)

**通讯引用:** 14467 | [OpenAlex ID](https://openalex.org/A5100462828)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了训练‑free 的 TTSA3R 框架，利用时空自适应更新机制实现长序列流式 3D 重建；

**💡 创新点**

创新点在于同时引入 Temporal Adaptive Update Module (TAUM) 与 Spatial Context Update Module (SCUM)，通过时序演化与空间对应信号联合决策状态更新，显著缓解了递归模型的灾难性遗忘；

**🔧 技术方法**

基于 ViT 编码器/解码器的递归 Transformer 结构，配合时序归一化、交叉注意力、余弦相似度等自适应信号；

**📊 数据集**

使用 Sintel、Bonn、KITTI（视频深度）、TUM‑Dynamics、ScanNet（位姿）、NRGBD（3D 重建）等公开数据集；

**📈 对比分析**

与多类基线（优化‑基、全注意力、流式）对比，在视频深度、位姿估计和 3D 重建上均实现了与全注意力模型相近或更优的精度，同时保持流式方法的实时性和低内存占用；

**⚠️ 局限性**

在遮挡严重或观测稀疏的场景下，空间对应信号不可靠时自适应机制效果受限；

---

## 274. Stabilizing Consistency Training: A Flow Map Analysis and Self-Distillation

**arXiv ID:** 2601.22679 | [PDF](https://arxiv.org/pdf/2601.22679v1)

**作者:** Youngjoong Kim `[一作]` (Seoul National University), Jaesik Park `[通讯]` (Seoul National University)

**通讯引用:** 9216 | [OpenAlex ID](https://openalex.org/A5100611457)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文对一致性模型的训练不稳定性进行了理论分析，并重新设计了自蒸馏目标，使其能够在无预训练扩散模型的前置器下实现从零初始化的稳定训练。

**💡 创新点**

创新点在于：①将一致性训练与Eulerian分散化统一到流图表示；②证明直接使用条件速度会导致退化解；③基于此提出改进的自蒸馏目标（iSD），通过约束梯度范数并兼容分类器无监督指导，从而提升训练稳定性与可复现性。

**🔧 技术方法**

技术主要包括：流图表示、Eulerian分散化、改进的自蒸馏目标（iSD-R）、JVP与有限差分实现、分类器无监督指导（Pre‑CFG / Post‑CFG）以及在Transformer基础上的DiT网络。

**📊 数据集**

使用的数据集有ImageNet‑1K（256×256）、CelebA‑HQ（256×256）以及几个机器人控制任务（Transport、Push‑T）。

**📈 对比分析**

在ImageNet‑1K 2步采样上，iSD‑T在不使用预训练前置器的情况下取得与现有最佳方法相当甚至更好的FID（15.20）且方差显著降低；在CelebA‑HQ与政策学习任务中也展现出与竞争方法相近或更优的性能。

**⚠️ 局限性**

局限性包括：仍主要针对少步采样设置，模型对大规模任务的可扩展性与对更复杂条件（如多模态）的适用性未完全验证，且在极高维数据上梯度范数约束仍需细化。

---

## 275. Postural Virtual Fixtures for Ergonomic Physical Interactions with Supernumerary Robotic Bodies

**arXiv ID:** 2601.22672 | [PDF](https://arxiv.org/pdf/2601.22672v1)

**作者:** Theodora Kastritsi `[一作]` (Istituto Italiano di Tecnologia), Arash Ajoudani `[通讯]` (Istituto Italiano di Tecnologia)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种面向超额机器人身体（SRB）的姿态虚拟装置控制框架，利用在线人体姿态评估与触觉反馈，动态调整浮动底座以提升人体工程学与协同效率。

**💡 创新点**

创新点包括：①将人体姿态作为连续可微的舒适度因子，嵌入虚拟装置动力学实现实时姿态反馈；②对SRB控制器进行能量守恒（被动性）分析；③基于RULA的连续姿态评估量化与多任务重心（null-space）坐标协同控制。

**🔧 技术方法**

技术手段包括：姿态虚拟装置（god-object）动力学模型、可变阻尼的自适应阻尼矩阵、基于RULA的姿态舒适度因子、弹性动手势接口（admittance控制）、浮动底座基于LiDAR的半圆形碰撞避免与Null-space约束、DLS求解全身逆运动学。

**📊 数据集**

实验数据集采用14名受试者在两种任务（细小操作与长距离搬运）中收集的运动捕捉（IMU/Mediapipe）与RGB-D视觉姿态数据，以及LiDAR点云；未使用公开数据集。

**📈 对比分析**

对比基线（不包含姿态因子）与PVF条件，结果显示平均舒适度因子提升约15%，非舒适姿势时间降低≥30%，姿势切入次数下降；NASA-TLX物理负荷显著降低；用户体验问卷中PVF获得更高可用性与学习效果。

**⚠️ 局限性**

局限性在于：①仅评估上肢与躯干姿态，未考虑下肢负荷；②IMU传感器漂移可能影响姿态估计；③使用2D LiDAR限制碰撞检测精度；④未纳入人体力学（力量、扭矩）信息。

---

## 276. ExpAlign: Expectation-Guided Vision-Language Alignment for Open-Vocabulary Grounding

**arXiv ID:** 2601.22666 | [PDF](https://arxiv.org/pdf/2601.22666v1)

**作者:** Junyi Hu `[一作]` (Tsinghua University), Yi Zhang `[通讯]` (Tsinghua University)

**通讯引用:** 24545 | [OpenAlex ID](https://openalex.org/A5100388281)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种名为ExpAlign的期望引导视觉‑语言对齐框架，专为弱监督下的开放词汇定位（检测与分割）任务设计；

**💡 创新点**

创新点包括：① 期望对齐头（EAH）通过软期望机制聚合 token‑region 相似度，实现无监督的 token 与实例选择；② 多尺度一致性正则化，融合 Top‑K 多正对比（InfoNCE）与几何一致性目标（GACO），从能量最小化视角理论化；③ 轻量化、可插拔于标准检测/分割网络。

**🔧 技术方法**

使用技术：CLIP 预训练文本编码器（保留 token 级表示）、ConvNeXt/T/Swin 视觉 backbone、SwiGLU 前馈映射、FPN 多尺度特征、soft MIL/注意力 soft‑pooling、Top‑K InfoNCE、几何一致性正则化（GACO）、能量‑based 多尺度一致性、AdamW+cosine 学习率、SAM‑2.1 伪 mask 生成等。

**📊 数据集**

数据集：Objects365、GoldG（GQA+Flickr30k）、RefCOCO/+/g、LVIS、ODinW、COCO（用于下游微调）；训练时剔除与 COCO 重叠图像，伪 mask 通过 SAM 生成。

**📈 对比分析**

与现有方法对比：在 LVIS minival 上 37.1 AP、36.2 AP_r，显著优于 GLIP、GDINO‑T、YOLO‑Worldv2‑L，参数仅 60M；在 ODinW13/35 上 47.7/22.4 AP；在零样本分割上 AP^m 29.9、AP_r 29.0，超过 YOLO‑Worldv2‑L 与 YOLOE；在 COCO 微调（线性/全量）下 AP_b/AP_m 与 YOLOE 对齐或略优。

**⚠️ 局限性**

局限性：在 RefCOCO/+/g 的指代表达理解远低于 Grounding DINO‑T，主要受限于 CLIP 文本编码器对位置/关系词的理解不足；对复杂关系表达（如 “left of”“behind”）的定位效果不佳。

---

## 277. Evaluating and Rewarding LALMs for Expressive Role-Play TTS via Mean Continuation Log-Probability

**arXiv ID:** 2601.22661 | [PDF](https://arxiv.org/pdf/2601.22661v1)

**作者:** Yong Ren `[一作]` (StepFun), Xuerui Yang `[通讯]` (StepFun)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文提出了一种基于大型音频语言模型的角色扮演式文本到语音（RP‑TTS）系统，并通过引入可解释的风格一致性度量MCLP来提升生成语音的表情与角色指令的一致性。

**💡 创新点**

创新点在于：①将MCLP作为既可用于评估又可作为奖励信号的连续风格一致性指标；②设计混合奖励函数（MCLP + CER）并使用GRPO训练，防止奖励黑客；③构建大规模多轮角色扮演TTS数据集。

**🔧 技术方法**

技术方法包括：利用预训练的大型音频语言模型进行上下文感知的TA4音频生成、SFT与RL联合训练、基于MCLP的风格评估、GRPO强化学习框架以及混合奖励设计。

**📊 数据集**

使用从WenetSpeech衍生的高质量剧集语音数据构建的RP‑TTS数据集，总计约1,435小时、311k场景、约7.3句/场景，涵盖多轮对话与角色/场景标签。

**📈 对比分析**

与GPT‑Audio、MiMo‑Audio‑7B‑Instruct和Step‑Audio‑2‑mini等基线比较，本文方法在内容准确率（CER、WER）和风格一致性（MCLP）上均优于对照组；主观MOS提升至3.65/5，接近真实语音。

**⚠️ 局限性**

局限性在于：①对中文语料和剧集场景的依赖，跨语言和跨文化迁移需进一步验证；②MCLP虽可解释但仍基于模型对风格的内部表示，可能无法覆盖所有细粒度情绪与语调变化；③RL训练对算力要求高，需更高效策略以适应工业部署。

---

## 278. UCPO: Uncertainty-Aware Policy Optimization

**arXiv ID:** 2601.22648 | [PDF](https://arxiv.org/pdf/2601.22648v1)

**作者:** Xianzhou Zeng `[一作]`, Xingzhong Xu `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种名为UCPO的强化学习框架，使大型语言模型能够在面对未知查询时自觉表达不确定性，显著降低幻觉产生。

**💡 创新点**

创新点在于通过三元优势解耦（TAD）将确定性与不确定性路径的优势分离，并结合动态不确定性奖励调整（DURA）实时平衡不确定性奖励，消除了传统固定奖励导致的优势偏差与奖励劫持问题。

**🔧 技术方法**

技术实现包括基于Group Relative Policy Optimization（GRPO）的无价值函数训练、三元优势解耦（TAD）、动态不确定性奖励调整（DURA）、非三元过滤（NTF）与低资源扩展（LRE）等模块。

**📊 数据集**

使用了大规模数学推理数据集DAPO-Math-17k进行训练，并在AIME24、AMC、MATH500、Minerva、Olympiad Bench等数学推理任务以及MMLU-Redux2、GPQA-Diamond等通用任务上进行评估。

**📈 对比分析**

与基线、Prompt-UC、GRPO以及GRPO-UC（固定不确定性奖励）等方法相比，UCPO在PAQ和F1指标上均取得显著提升，尤其在复杂推理任务中将幻觉率降低、可靠性提升，并通过动态奖励避免了过度规避或过度自信的现象。

**⚠️ 局限性**

局限性在于不同采样比例（正确、错误、不确定）对训练动态的影响尚未完全解析，且在低资源或极端任务难度下，动态奖励调节可能仍需进一步优化。

---

## 279. Statistical Estimation of Adversarial Risk in Large Language Models under Best-of-N Sampling

**arXiv ID:** 2601.22636 | [PDF](https://arxiv.org/pdf/2601.22636v1)

**作者:** Mingqian Feng `[一作]` (University of Rochester), Jianfeng Gao `[通讯]` (Microsoft Research)

**通讯引用:** 38229 | [OpenAlex ID](https://openalex.org/A5114910293)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6215c339-3735-4be3-8a07-5bbb7004712d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种基于Beta分布的Best‑of‑N风险估计框架（SABER），能够用少量预算测量快速预测大规模攻击成功率。

**💡 创新点**

创新点在于推导了一个精确的概率尺度律，将单次成功率分布映射到大规模并行采样下的成功率，并引入了锚定估计器显著降低估计误差。

**🔧 技术方法**

核心技术包括贝塔–二项分布的最大似然估计、Gamma函数渐近展开以及基于Delta法的置信区间计算。

**📊 数据集**

使用了HarmBench（159条有害查询）以及多种攻击器、受害模型（Llama‑3.1‑8B‑Instruct、GPT‑4.1‑mini）和评判器（LLM Classifier、HarmBench Classifier）进行实验。

**📈 对比分析**

与基线（基于单次成功率直接推算）对比，锚定估计在多种预算与目标N下平均绝对误差降低约4–6倍，且在不均匀预算或小样本场景亦保持优势。

**⚠️ 局限性**

局限性包括对Beta分布假设的依赖、在极小N或非平稳查询集时误差可能上升，以及对不同评判器或多模态任务的推广性尚待验证。

---

## 280. MCP-Diag: A Deterministic, Protocol-Driven Architecture for AI-Native Network Diagnostics

**arXiv ID:** 2601.22633 | [PDF](https://arxiv.org/pdf/2601.22633v1)

**作者:** Devansh Lodha `[一作]` (Indian Institute of Technology Gandhinagar), Sameer G. Kulkarni `[通讯]` (Indian Institute of Technology Gandhinagar)

**通讯引用:** 1077 | [OpenAlex ID](https://openalex.org/A5081037214)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

提出了 MCP-Diag 架构，通过确定性翻译层、协议级治理和 SSE 混合传输，实现在网络诊断中的 LLM 自动化与安全交互。

**💡 创新点**

创新点在于：① 在协议层实现强制的人机确认弹幕（Elicitation Loop）；② 将 CLI 原始输出转换为严格的 JSON 架构，消除 LLM 解析错误；③ 通过 SSE 实现长时间诊断工具的无阻塞异步流。

**🔧 技术方法**

采用 Model Context Protocol（MCP）为基础，结合 JSON‑schema 解析器（如 `JSON‑Schema‑Validator`）、Server‑Sent Events、Bun/V8 JavaScript 运行时以及 Gemma‑3‑27b‑it LLM。

**📊 数据集**

使用公开的 Top‑500 域名列表（共 500 条目标）作为 traceroute 任务的测试数据集。

**📈 对比分析**

与传统基于正则的 CLI 调用方法对比，MCP‑Diag 在 500 次试验中实现 100% 实体提取准确率，平均延迟仅 0.9%（约 311 ms），上下文 token 消耗提升 3.7 倍，但并未明显影响首次输出时间。

**⚠️ 局限性**

局限性包括：① 仅在单一 MCP Host–Server 场景下验证，未评估并发与大规模部署；② 评估任务仅覆盖 traceroute，未验证更复杂诊断链；③ SSE 侧通道需自行实现，可能不兼容所有 MCP 生态。

---

## 281. UniGeo: A Unified 3D Indoor Object Detection Framework Integrating Geometry-Aware Learning and Dynamic Channel Gating

**arXiv ID:** 2601.22616 | [PDF](https://arxiv.org/pdf/2601.22616v1)

**作者:** Xing Yi `[一作]` (Hefei University of Technology), Dan Guo `[通讯]` (University of Science and Technology of China)

**通讯引用:** 37635 | [OpenAlex ID](https://openalex.org/A5100434028)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了统一的3D室内检测框架UniGeo，融合几何感知学习与动态通道门控机制；

**💡 创新点**

创新点在于通过几何感知学习模块将空间几何关系映射为特征权重，并结合动态通道门控实现自适应特征增强；

**🔧 技术方法**

使用了稀疏3D U-Net特征提取、欧氏距离几何中心映射、指数衰减权重、动态通道门控、Transformer编码器和DIoU回归等技术；

**📊 数据集**

在六大室内点云数据集（ScanNet、S3DIS、MultiScan、3RScan、ScanNet++、ARKitScenes）上进行评估；

**📈 对比分析**

与SOTA方法对比，UniGeo在大多数数据集上实现mAP提升5-10%（如S3DIS mAP25+5.3%、mAP50+11%），并保持了平均实验稳定性；

**⚠️ 局限性**

局限性在于对域差异仍敏感，某些复杂场景的几何噪声和稀疏分布导致准确率略低。

---

## 282. Stabilizing Transformer Training Through Consensus

**arXiv ID:** 2601.22614 | [PDF](https://arxiv.org/pdf/2601.22614v1)

**作者:** Shyam Venkatasubramanian `[一作]` (Anthrogen PBC), Connor Lee `[通讯]` (Anthrogen PBC)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出并验证了一种以图谱为基础的共识机制（Consensus）替代传统注意力，以提升Transformer在高学习率训练中的稳定性

**💡 创新点**

创新点在于将共识机制形式化为图谱低通滤波器，证明其与注意力的低频保留性质相似，并通过混合共识-注意力架构实现既保持性能又增强鲁棒性

**🔧 技术方法**

采用图谱谱理论、低通滤波、图卷积/图注意力、以及梯度更新等技术；实现了自共识、自交叉共识以及多头扩展

**📊 数据集**

使用OpenWebText（文本）、OpenGenome（DNA）、AlphaFoldDB（蛋白序列与结构）等公开数据集进行训练与评估

**📈 对比分析**

通过在不同规模（54M–385M）和模态下的学习率扫掠，比较自注意力（SA）、自共识（SC）、滑窗注意力（SW）与混合（MIX）四种机制；SC在更宽学习率区间内保持较低NLL，MIX在最优学习率下与SA性能相当，但对过大学习率更稳健

**⚠️ 局限性**

主要局限在于共识机制需预先指定或学习图结构，且在某些高阶结构或异构数据中可能不易自动构造，需进一步研究自适应图构造方法

---

## 283. Full-Graph vs. Mini-Batch Training: Comprehensive Analysis from a Batch Size and Fan-Out Size Perspective

**arXiv ID:** 2601.22678 | [PDF](https://arxiv.org/pdf/2601.22678v1)

**作者:** Mengfan Liu `[一作]` (University of Hong Kong), Chuan Wu `[通讯]` (University of Hong Kong)

**通讯引用:** 10897 | [OpenAlex ID](https://openalex.org/A5012597518)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

系统地比较全图训练与小批量训练在图神经网络中的表现与效率，重点分析批量大小与 fan-out 大小对收敛、泛化和计算开销的影响。

**💡 创新点**

提出基于 Wasserstein 距离的泛化理论，揭示批量大小与 fan-out 大小在收敛与泛化上的非各向同性影响；引入迭代次数为指标的硬件无关评估方法；给出在内存约束下的调参指导。

**🔧 技术方法**

理论分析（一层 GNN + ReLU、Gaussian 初始化、PAC‑Bayesian 框架）、Wasserstein 距离、实验评估（迭代‑精度、时间‑精度、吞吐量）以及多种 GNN 模型（GCN、GraphSAGE、GAT）。

**📊 数据集**

四个真实数据集：reddit、ogbn‑arxiv、ogbn‑products、ogbn‑papers100M，采用 GCN、GraphSAGE（均值聚合）和 GAT（多头注意力）进行实验。

**📈 对比分析**

使用批量大小与 fan‑out 的不同组合进行实验，对比全图训练与小批量训练在收敛速度、精度和吞吐量上的表现。结果表明：在合适的批量/ fan‑out 设置下，小批量训练往往能获得与全图相近甚至更优的精度，同时计算效率更高；但若批量过大或 fan‑out 超大，则会出现泛化退化。

**⚠️ 局限性**

局限性：理论主要基于一层 GNN、ReLU 与 Gaussian 初始化，未覆盖更深网络或不同激活函数；实验仅在节点分类任务上验证；对图的稀疏度假设有限；实际部署时可能还需考虑分布式通信与多任务场景。

---

## 284. VarParser: Unleashing the Neglected Power of Variables for LLM-based Log Parsing

**arXiv ID:** 2601.22676 | [PDF](https://arxiv.org/pdf/2601.22676v1)

**作者:** Jinrui Sun `[一作]` (Peking University), Ying Li `[通讯]` (Peking University)

**通讯引用:** 22414 | [OpenAlex ID](https://openalex.org/A5100414156)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了基于大型语言模型的变量中心化日志解析框架 VarParser，通过变量贡献采样、变量模糊匹配、变量单元缓存和自适应变量感知上下文学习，提升日志解析的准确性、效率并保留变量信息。

**💡 创新点**

创新点在于：①聚焦日志的可变部分而非常量，显著提升信息利用；②设计变量贡献采样和变量模糊匹配表，减少日志组数和 LLM 调用；③引入变量单元缓存，记录变量标签、实例与频率，解决 LLM 幻觉并提升结果完整性；④自适应 token‑级示例选择与 Prompt 设计，降低 token 消耗。

**🔧 技术方法**

使用技术包括：大型语言模型（gpt‑3.5‑turbo、llama3‑70b、qwen‑plus）；Jaccard 相似度、Token 级相似度；前缀树 + 变量模糊匹配表缓存；变量单元结构；Token‑级示例选择与 Prompt 设计；自适应变量感知 ICL。

**📊 数据集**

使用 Loghub‑2.0 大规模日志基准，包含 14 个多样化系统日志，共计约 3.69M 条日志。

**📈 对比分析**

在 GA、PA、FGA、FTA 四项指标上与 8 个最先进基线（语法基、语义基、LLM 基）对比，平均提升 GA 3.9%、PA 8.5%、FTA 5.8%；解析时间比 LILAC 降低 51.2%，Token 使用比 LILAC 降低 56%；与 LogBatcher 的效率相近并降低 14.8%。

**⚠️ 局限性**

局限性：①变量标签仍可能出现无意义值（如 “label”）需要后处理；②在常量与变量相近的日志（如 MAC）可能导致误识别；③依赖 LLM 的输出稳定性，hallucination 仍需校正；④对实时流仍逐条处理，未实现完全无停顿批处理；⑤采样与示例选择需一定手工标注，启动成本仍存在。

---

## 285. Real-Time Aligned Reward Model beyond Semantics

**arXiv ID:** 2601.22664 | [PDF](https://arxiv.org/pdf/2601.22664v1)

**作者:** Zixuan Huang `[一作]` (Beihang University), Deqing Wang `[通讯]` (Beihang University)

**通讯引用:** 9990 | [OpenAlex ID](https://openalex.org/A5102969899)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 R2M，一种轻量级 RLHF 框架，通过实时利用策略模型隐藏层状态来动态更新奖励模型；

**💡 创新点**

创新点在于将策略反馈融入奖励模型，使用跨注意力与时间步加权组合实现对齐，并引入 GREBT（Group Reward Entropy + Bradley‑Terry）混合损失；

**🔧 技术方法**

采用跨注意力机制、时间步加权、GREBT 损失、以及 RLOO/GRPO 等 RL 优化技术；

**📊 数据集**

使用 UltraFeedback 对话数据，评测于 AlpacaEval 2 与 MT‑Bench；在摘要任务使用 TL;DR 数据；

**📈 对比分析**

与基线 RLOO、GRPO、REINFORCE++ 等相比，R2M 在 AlpacaEval 2 的 win‑rate 提升 5.2–8.0%，长度控制 win‑rate 提升 2.9–6.1%，TL;DR win‑rate 提升 6.3%；

**⚠️ 局限性**

局限在于奖励模型的 LLM 部分保持冻结，且在极端分布漂移或大规模模型上效果可能受限。

---

## 286. NAG: A Unified Native Architecture for Encoder-free Text-Graph Modeling in Language Models

**arXiv ID:** 2601.22657 | [PDF](https://arxiv.org/pdf/2601.22657v1)

**作者:** Haisong Gong `[一作]` (New Laboratory of Pattern Recognition Institute of Automation Chinese Academy of Sciences), Liang Wang `[通讯]` (School of Artificial Intelligence University of Chinese Academy of Sciences)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种无外部图编码器的语言模型框架 NAG，能够直接在 Transformer 内部完成文本图的编码与推理。

**💡 创新点**

创新点在于通过拓扑感知自注意力机制和结构位置校准，将节点与边的文本信息与图拓扑自然融合，打破传统双通道（GNN+LM）架构的概念分离。

**🔧 技术方法**

主要技术包括结构化展开、特殊标记包装、层级化拓扑注意力掩码、RoPE 位置重校准，以及两种轻量化适配实现（LoRA 与 Zero 适配器）。

**📊 数据集**

实验使用合成图数据集（9项拓扑任务）以及三大真实图推理基准 ExplaGraphs、SceneGraphs 与 WebQSP。

**📈 对比分析**

与 Qwen3‑Direct、GraphToken 等对标基线相比，NAG‑LoRA 在多数任务上获得最优或相近性能，NAG‑Zero 亦在大多数任务上优于 GraphToken 变体，显示出高效且无损的结构注入效果。

**⚠️ 局限性**

局限性包括：Zero 适配器参数空间有限，难以深度融合语义；稀疏与全连通注意策略在不同任务中表现不稳定；在高连通度节点时信息瓶颈显现；对大规模模型与动态图的可扩展性仍待验证。

---

## 287. Test-Time Mixture of World Models for Embodied Agents in Dynamic Environments

**arXiv ID:** 2601.22647 | [PDF](https://arxiv.org/pdf/2601.22647v1)

**作者:** Jinwoo Jang `[一作]` (Sungkyunkwan University), Honguk Woo `[通讯]` (Sungkyunkwan University)

**通讯引用:** 871 | [OpenAlex ID](https://openalex.org/A5001227049)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `3f18e8e3-0266-457c-8567-9039b6d2394d` `afceb026-1760-41ae-8d86-010831a37d97` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `8d10c613-917e-4880-9716-17789f50e119` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出 Test-time Mixture of World Models (TMoW)，一种在测试时动态重配置世界模型混合、并通过少样本蒸馏快速扩展新模型的多专家架构；

**💡 创新点**

创新点在于：①在 MoE 中引入多粒度原型路由，利用对象到场景层次的相似性实现灵活混合；②通过测试时原型细化在不重训的情况下对未见域进行自适应；③提出蒸馏混合模型增补机制，利用已有专家的知识在极少示例下构造新世界模型；

**🔧 技术方法**

技术主要包括：多层级图神经网络（MPNN）提取原型、基于余弦相似度的稀疏 top‑K 路由、测试时原型插值更新、教师强迫蒸馏训练、轻量化适配器（如 LoRA）实现专家模块；

**📊 数据集**

实验数据集包括 VirtualHome、ALFWorld、RLBench 三个仿真环境以及真实机器人 Franka Research 3；

**📈 对比分析**

与五个基线（ZSP、LLM+FT、LLM‑Planner、SayCanPay、FLARE）对比，在零样本与少样本扩展场景下，TMoW 在未见域成功率提升约 27% 以上、步骤数下降 3–4 步；在真实场景中相较最佳基线提升 34–39% 的成功率；

**⚠️ 局限性**

局限性：性能受底层 LLM 能力限制；在高度非平稳或多智能体环境中，世界模型对动态行为变化的适应仍面临挑战。

---

## 288. LINA: Linear Autoregressive Image Generative Models with Continuous Tokens

**arXiv ID:** 2601.22630 | [PDF](https://arxiv.org/pdf/2601.22630v1)

**作者:** Jiahao Wang `[一作]` (University of Hong Kong), Ping Luo `[通讯]` (University of Hong Kong)

**通讯引用:** 53056 | [OpenAlex ID](https://openalex.org/A5100752686)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于线性注意力的自回归图像生成模型LINA，并在连续令牌框架下进行系统实验。

**💡 创新点**

创新点包括：①证明在自回归生成中使用除法归一化的线性注意力更优；②加入深度可分卷积提升局部性建模；③设计KV门控机制以灵活管理键值记忆，弥补了传统忘记门在双向注意中的不足。

**🔧 技术方法**

主要技术包括除法归一化线性注意力、深度可分卷积（DWC）、KV门控（K门和V门）以及流匹配MLP作为去噪网络。

**📊 数据集**

使用ImageNet 256×256做类别条件生成和1024×1024做文本到图像（T2I）生成，数据来源于公开的ImageNet和文本提示数据集。

**📈 对比分析**

与软最大注意力模型和现有扩散模型对比：在ImageNet 256×256下，LINA-H达到FID 2.18，接近或优于SOTA扩散模型；在T2I任务中，1.4B版本在GenEval上得到74分，超越1.6B SANA，逼近10.5B Fluid；单个线性注意模块将FLOPs降低约61%，推理延迟与软最大注意力相当。

**⚠️ 局限性**

局限性包括：①模型在极大分辨率（>1024px）下的可扩展性和训练成本仍较高；②KV门控是数据独立的可学习参数，可能无法充分利用上下文信息；③实验主要集中在ImageNet和标准文本提示上，未对多模态或更复杂场景进行验证。

---

## 289. Local-Global Multimodal Contrastive Learning for Molecular Property Prediction

**arXiv ID:** 2601.22610 | [PDF](https://arxiv.org/pdf/2601.22610v1)

**作者:** Xiayu Liu `[一作]` (University of Electronic Science and Technology of China), Hou-biao Li `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 12491 | [OpenAlex ID](https://openalex.org/A5065782725)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

该论文提出了一种基于本地-全局多模态对比学习的分子属性预测框架LGM-CL，联合建模分子图（局部功能组和全局拓扑）和化学文本（SMILES与LLM生成的语义增强文本），并通过对比学习与双向交叉注意力融合多模态信息；

**💡 创新点**

创新点在于（1）将局部和全局图表示并行编码并通过对比学习对齐；（2）设计化学感知的文本增强模板，利用LLM生成富含化学语义的描述并与原SMILES进行对比学习；（3）构建双交叉注意力模块，将图、文本与指纹三模态融合成统一表征；

**🔧 技术方法**

核心技术包括Graph Transformer与AttentiveFP两种图编码器、DeBERTa文本编码器、对比学习（NT‑Xent）、LLM文本生成（Mistral‑7B‑Instruct）以及双交叉注意力融合；

**📊 数据集**

预训练使用ZINC15无标签分子数据，微调评估在MoleculeNet十个基准数据集（BACE、BBBP、Clintox、Tox21、SIDER、HIV、ToxCast、ESOL、FreeSolv、Lipophilicity）；

**📈 对比分析**

与多种自监督和对比学习基线（GROVER、S‑CGIB、GraphCL等）对比，LGM-CL在分类任务中平均提升ROC‑AUC约1–4个百分点，在回归任务中平均降低RMSE约20–30%，展现出更稳健、竞争力的性能；

**⚠️ 局限性**

主要局限在于依赖LLM生成文本的质量与生成开销，模型对大规模训练和高计算资源需求较高，且在更复杂或更大分子空间的泛化能力尚未充分验证；

---

## 290. Beyond Abstract Compliance: Operationalising trust in AI as a moral relationship

**arXiv ID:** 2601.22769 | [PDF](https://arxiv.org/pdf/2601.22769v1)

**作者:** Lameck Mbangula Amugongo `[一作]` (Boehringer Ingelheim Pharma GmbH and Co. KG), Nicola J Bidwell `[通讯]`

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出基于非洲关系伦理（Ubuntu）的信任设计原则，并将其在AI开发生命周期中与敏捷方法和社区共创相结合，示范在医疗决策支持与教育辅导两大场景中的应用。

**💡 创新点**

创新点在于将信任从单一的合规属性转变为动态、关系化的道德信任，提出四大原则（共生主义、尊重他人、诚信、设计公开）并提供具体的实施步骤与案例演示，首次将非洲哲学视角嵌入可信AI框架。

**🔧 技术方法**

主要技术手段是原则驱动的设计与治理方法、社区共创与协作评估、敏捷开发流程、设计公开（透明化说明书、数据治理方案）以及红队/评估协同机制。

**📊 数据集**

未使用具体公开数据集，案例中假设使用医疗抗生素处方记录与教育学习日志等典型领域数据，但未给出详细数据来源或规模。

**📈 对比分析**

本文未进行实验比较或性能评估；主要通过案例说明来展示原则的可操作性，缺乏量化指标和对比基准，因而无法给出性能结果。

**⚠️ 局限性**

局限性包括：理论模型缺乏可量化评估工具；未进行实证验证或大规模用户研究；仅以示例说明，未探讨在不同文化或技术环境下的适用性与可扩展性。

---

## 291. Sparse Attention as Compact Kernel Regression

**arXiv ID:** 2601.22766 | [PDF](https://arxiv.org/pdf/2601.22766v1)

**作者:** Saul Santos `[一作]` (Instituto Superior Técnico, Universidade de Lisboa), André F. T Martins `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过将稀疏注意力机制解释为带有限支持的核回归，构建了稠密与稀疏注意力统一的理论框架，并在 Memory Mosaics 模型中验证了该框架。

**💡 创新点**

创新点在于：① 将 sparsemax、α-entmax、top‑k、normalized ReLU 与 ReLUmax 等注意力映射映射到 Epanechnikov、biweight、triweight 等经典有限支持核；② 通过自适应归一化（自动归一化）实现核带宽自适应，得到稀疏最大化与 α‑entmax；③ 提出新的 ReLUmax 变换以解决归一化 ReLU 的 0/0 歧义；④ 将该核视角应用于 Memory Mosaics，展示在语言建模、上下文学习和长度泛化任务中的优势。

**🔧 技术方法**

技术手段主要包括：核回归（Nadaraya‑Watson 估计）、有限支持多项式核（Epanechnikov、biweight、triweight）、稀疏注意力变换（sparsemax、α‑entmax、top‑k、normalized ReLU、ReLUmax）、自适应核带宽设计、Memory Mosaics 架构（上下文记忆与持续记忆模块）、以及多种实验评测方法。

**📊 数据集**

使用的数据集包括：BabiStories（小规模英文叙事）用于语言建模；RegBench（合成正则语言）用于上下文学习；以及多项长度泛化任务（MQMTAR、序列排序、反转）用于测试模型在长序列上的表现。

**📈 对比分析**

与传统的 Gaussian（softmax）以及启发式 top‑k 方案相比，实验表明：① 采用有限支持核的稀疏注意力在浅层模型中验证损失更低，且随着层数加深仍保持优势；② 在 RegBench 上，Biweight、Triweight、ReLUmax、sparsemax 等方法在低样本量下准确率更高、TVD 更低；③ 在长度泛化任务中，有限支持核能够保持高精度并显著优于 Gaussian 与 top‑k，尤其是 Triweight 与 Biweight 在 MQMTAR 与反转任务中表现最佳。

**⚠️ 局限性**

局限性包括：① ReLUmax 与自适应归一化方法相比，在深层模型中性能略逊；② top‑k、Gaussian kNN 在极长序列上仍易出现不稳定；③ 实验主要在小规模合成或简易文本数据上进行，未在大规模真实语料上验证；④ 需要手动调参（如 ReLUmax 的偏置 b、α‑entmax 的 α）以获得最佳效果，缺乏自动化选择机制。

---

## 292. Discovering Scaling Exponents with Physics-Informed Müntz-Szász Networks

**arXiv ID:** 2601.22751 | [PDF](https://arxiv.org/pdf/2601.22751v1)

**作者:** Gnankan Landry Regis N'guessan `[一作]` (Axiom Research Group), Bum Jun Kim `[通讯]` (University of Tokyo)

**通讯引用:** 249 | [OpenAlex ID](https://openalex.org/A5100666999)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本研究提出并验证了一种结合Müntz‑Szász网络与物理信息网络（MSN‑PINN）的新框架，可直接从PDE残差中学习并输出物理系统的尺度指数。

**💡 创新点**

核心创新在于将尺度指数显式化为可训练参数，理论上证明了可辨识性与稳定性，并通过约束感知训练解决指数漂移问题。

**🔧 技术方法**

技术方法包括：可训练幂律基的MSN结构、基于PDE残差的PINN损失、约束损失（如边界条件兼容性）以及双时标梯度优化。

**📊 数据集**

实验使用合成PDE解析解（如幂律函数、Laplace楔形尖点、含奇异源的Poisson方程）以及40配置楔形基准数据集。

**📈 对比分析**

与传统PINN及无约束MSN-PINN比较，MSN‑PINN在尺度指数误差上从数百个百分点下降到0.009%–0.05%，在40配置楔形基准上实现100%成功率、平均误差0.022%，性能显著提升。

**⚠️ 局限性**

主要限制是需要先验物理约束才能唯一确定指数，对指数间分离度敏感；当相邻指数非常接近时，恢复精度会下降。

---

## 293. Is Softmax Loss All You Need? A Principled Analysis of Softmax-family Loss

**arXiv ID:** 2601.22745 | [PDF](https://arxiv.org/pdf/2601.22745v1)

**作者:** Yuanhao Pu `[一作]` (University of Science and Technology of China), Enhong Chen `[通讯]` (University of Science and Technology of China)

**通讯引用:** 27823 | [OpenAlex ID](https://openalex.org/A5048237545)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文对Softmax族损失函数进行统一理论分析，探讨其一致性、梯度动力学、近似方法的偏差-方差分解及计算复杂度，并在推荐任务上系统实验验证。

**💡 创新点**

创新点在于将Fenchel‑Young框架与Bregman散度、镜像下降相结合，首次从SOP与WOP的雅可比结构区分不同损失的收敛行为；并给出软化近似方法的完整偏差‑方差分解与每轮复杂度表，提供实用的损失选择准则。

**🔧 技术方法**

采用Fenchel‑Young理论、Bregman散度分析、雅可比谱分析、Δ‑方法偏差‑方差分解、采样近似（NCE、SSM）、层次软化（HSM、RG）和稀疏化（Sparsemax、α‑Entmax、Rankmax）等技术。

**📊 数据集**

使用公开的推荐数据集ML‑1M、Amazon‑Electronics和Gowalla，并在MF、SASRec、LightGCN三种骨干网络上进行对比实验。

**📈 对比分析**

对比方法包括完整Softmax、稀疏化Softmax族以及近似方法；实验表明在中等类别数下Softmax收敛最快、性能最优；稀疏化方法在简单模型上速度快但在高容量模型中表现下降；采样方法速度快但受偏差与采样分布影响；Rankmax在强调硬负样本时优于其他稀疏方法。

**⚠️ 局限性**

局限性在于稀疏化方法因WOP导致梯度稀疏、收敛慢；近似方法存在不可消除的偏差；理论分析主要针对logit层的凸性，未完全覆盖深层网络的非凸训练；实验集有限，未覆盖更大规模工业场景。

---

## 294. StreamSense: Streaming Social Task Detection with Selective Vision-Language Model Routing

**arXiv ID:** 2601.22738 | [PDF](https://arxiv.org/pdf/2601.22738v1)

**作者:** Han Wang `[一作]` (Singapore University of Technology and Design), Roy Ka-Wei Lee `[通讯]` (Singapore University of Technology and Design)

**通讯引用:** 1600 | [OpenAlex ID](https://openalex.org/A5089793938)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种针对直播流的实时社交任务检测框架 StreamSense，能够在每个时间戳根据历史视频、文本和音频信息做出判定，并在必要时调用大模型进行复核或推迟决定。

**💡 创新点**

创新点包括：① 用轻量化流编码器与大模型的“选择性路由”机制，显著降低 VLM 调用频率；② 引入 IoU‑加权交叉熵减轻段界干扰；③ 采用跨模态对比损失对视觉、音频与文本进行对齐；④ 设计可学习或阈值的推迟策略，在上下文不足时自动延迟决策。

**🔧 技术方法**

技术细节：多模态流编码器（视觉 ViT‑Large、文本 BERT、音频 Wav2Vec2），2 层编码/融合网络+MLP；跨模态对比损失与 IoU 加权 CE 训练；阈值/MLP 方式的 VLM 调度与推迟；VLM 采用预训练 Llama‑3.2‑Vision 进行专家复核。

**📊 数据集**

实验使用三大公开数据集：MOSI、MOSEI（情感分析）和 HateClipSeg（仇恨语料）。

**📈 对比分析**

与 OadTR、LSTR（轻量化动作识别）以及 Qwen2.5、LLaVA‑Next、Llama‑3.2（VLM‑only）进行对比。StreamSense 在三数据集上实现了最高的 Macro‑F1，同时 VLM 调用率仅 27%，推迟率 17%，平均延迟 0.3 s，显著低于纯 VLM 方案且准确率超越其。

**⚠️ 局限性**

局限性：仅使用过去信息，缺乏未来视角，可能导致在信息不足时的误判；需要在实际部署中加入短时缓冲区以平衡实时性与准确性。

---

## 295. GaussianOcc3D: A Gaussian-Based Adaptive Multi-modal 3D Occupancy Prediction

**arXiv ID:** 2601.22729 | [PDF](https://arxiv.org/pdf/2601.22729v1)

**作者:** A. Enes Doruk `[一作]` (Ozyegin University), Hasan F. Ates `[通讯]` (Ozyegin University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种多模态3D语义占据预测框架GaussianOcc3D，将相机语义与激光雷达几何信息通过连续3D高斯原子统一建模，并在此空间上进行特征聚合、平滑、融合与全局上下文建模，最终生成高分辨率占据栅格。

**💡 创新点**

创新点包括：
• 采用3D高斯原子代替体素/BEV，实现稀疏激光雷达与稠密相机信息的自适应上采样与融合；
• 深度可变形采样（LDFA）将激光雷达点云映射至高斯原子；
• 基于交叉熵的熵平滑（EBFS）抑制模态噪声，提升语义一致性；
• 自适应相机‑激光雷达融合（ACLF）引入一致性加权与门控机制，动态平衡两模态贡献；
• Gauss‑Mamba Head使用选择性状态空间模型，线性复杂度捕获全局上下文。

**🔧 技术方法**

使用的技术包括：3D高斯喷涂、深度可变形采样、交叉熵熵平滑、跨模态注意力与门控融合、选择性状态空间模型（Mamba）、交叉熵与Lovász‑softmax 损失、AdamW 优化、余弦学习率退火。

**📊 数据集**

实验数据集：NuScenes、SurroundOcc、Occ3D、SemanticKITTI。分别在2Hz、0.5m/0.4m、256×256×32等不同分辨率与场景下评估。

**📈 对比分析**

与多种SOTA方法（如GaussianFormer3D、OccFusion、M‑CONet、Co‑Occ、BEVFormer等）对比，GaussianOcc3D在Occ3D上取得49.4% mIoU、SurroundOcc 28.9% mIoU、SemanticKITTI 25.2% mIoU，分别显著领先对手；在雨天与夜间等恶劣天气下仍保持较高性能，显示出鲁棒性。

**⚠️ 局限性**

限制：
• 仍需要精准的相机‑雷达标定，误配对性能影响明显；
• 25,600个高斯原子虽比体素更高效，但在极大场景或更高分辨率时仍可能导致显存占用与推理时延；
• 对极端稀疏雷达点云或极端光照条件下的鲁棒性尚未彻底验证。

---

## 296. On Small Pair Decompositions for Point Sets

**arXiv ID:** 2601.22728 | [PDF](https://arxiv.org/pdf/2601.22728v1)

**作者:** Kevin Buchin `[一作]`, Carolin Rehs `[通讯]`

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文研究了点集的近似二分图覆（Approximate Biclique Cover）与最小WSPD分解，提出了常数逼近算法并给出实验验证。

**💡 创新点**

创新点在于引入不限制直径的新分解模型，证明任意有限度量空间可得到近线性大小的覆；在欧氏空间实现O(d³)线性大小；在doubling度量空间给出输出敏感的常数逼近算法，并提供相应的NP难度与逼近难度下界。

**🔧 技术方法**

核心技术包括：环状分解与球面划分构造近似二分图覆；基于α‑四叉树（shifted quadtree）与球面交集实现欧氏空间的覆；利用net‑tree构造doubling空间的WSPD；在一维情形下将问题化归为伪圆盘集合的集合覆盖并使用贪心、局部搜索与直线扫描实现3近似；以及将覆转化为矩形划分实现互不相交覆。

**📊 数据集**

实验使用一维合成数据集，包括：{1,…,n}、均匀分布、正态分布、对数分布、i²、i³、(log(i+1))²等多种点分布，总样本量从10到400，均采用相同的分解参数。

**📈 对比分析**

与方法比较包括：贪心集合覆盖、Callahan‑Kosaraju的WSPD、3-近似算法、带后处理的3-近似、整数规划最优解以及互不相交覆的整数规划。实验表明3-近似和后处理版在绝大多数分布下接近或等于最优解，且运行时间为O(n log n)；相比之下贪心和WSPD往往得到更大覆，整数规划则在规模受限时效果最佳。

**⚠️ 局限性**

限制在于：覆的权重在最坏情况下可能为二次；对一般度量空间的线性大小界不一定紧；高维欧氏空间的常数逼近仍未解决；以及最小WSPD的多项式逼近算法尚无已知的上界。

---

## 297. OpenVTON-Bench: A Large-Scale High-Resolution Benchmark for Controllable Virtual Try-On Evaluation

**arXiv ID:** 2601.22725 | [PDF](https://arxiv.org/pdf/2601.22725v1)

**作者:** Jin Li `[一作]` (Renxing Intelligence), Chenhui Wu `[通讯]` (Renxing Intelligence)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `729e5870-4135-47f5-97f2-e3974d07b5dc` `67630363-6be0-4f51-ab05-7198250671a5` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个约10万张高分辨率（1024-1536像素）虚拟试衣基准（OpenVTON‑Bench），并提出了融合VLM语义推理与多尺度结构度量的混合评估协议。

**💡 创新点**

创新点包括：
1) 利用DINOv3进行语义聚类并通过Gemini实现高质量密集标题，保证数据在20个细粒度服装类别上的均衡分布；
2) 设计了基于SAM3分割和形态学侵蚀的多尺度表面相似度指标，可分离边界对齐误差与内部纹理失真；
3) 将VLM‑as‑a‑Judge与低层次特征度量相结合，实现与人类评价高度相关的五维质量指标。

**🔧 技术方法**

使用技术：
- DINOv3 (ViT‑H+) 进行视觉嵌入和语义聚类；
- Gemini 2.0 Flash 进行密集标题生成；
- SAM3 进行服装分割；
- VLM（如Qwen‑VL‑Plus）作为语义判定器；
- 多尺度侵蚀与DINOv3特征相似度计算；
- 传统像素/分布指标（PSNR、SSIM、LPIPS、FID）作为对照。

**📊 数据集**

使用的数据集：
- 公开现有数据集（VITON、VITON‑HD、DressCode、SHHQ、StreetTryOn、LH‑400K、VTBench、VTONQA）中的高分辨率样本；
- 通过网页抓取的公开图片；
- 通过人工与AI混合标注得到的最终99,925张图像对，覆盖20个服装子类别。

**📈 对比分析**

在OpenVTON‑Bench上对9种主流VTON方法（包括FLUX、Qwen‑Editor、YingHui、Nanobanana等）进行评测。混合评估指标（VLM评分与多尺度相似度）与人工评估的相关性显著高于传统指标（如SSIM、PSNR），特别是多尺度相似度的Kendall τ达到0.833，远超SSIM（0.611）。模型在整体真实度与纹理保真度上表现差异明显，FLUX.2与YingHui在大部分维度均位居前列，但仍存在纹理细节失真。

**⚠️ 局限性**

局限性：
- 数据预处理与标题生成高度依赖预训练基础模型，可能带来语义偏差或幻觉；
- 尽管分辨率提升至1.5K，但对极端遮挡、多层穿搭或极端姿态的覆盖仍有限；
- 评估协议虽然改进，但对VLM提示偏差和细粒度纹理识别仍存在一定误差。

---

## 298. Breaking the Blocks: Continuous Low-Rank Decomposed Scaling for Unified LLM Quantization and Adaptation

**arXiv ID:** 2601.22716 | [PDF](https://arxiv.org/pdf/2601.22716v1)

**作者:** Pingzhi Tang `[一作]` (Institute for Artificial Intelligence), Muhan Zhang `[通讯]` (Institute for Artificial Intelligence)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了LoRDS框架，通过将量化缩放矩阵拆解为低秩因子（S＝BA），实现对LLM的统一量化（PTQ/QAT）与参数高效微调（PEFT）的整合；

**💡 创新点**

核心创新是打破传统块级量化限制，使用连续低秩矩阵提供更高表达力，同时实现零推理开销的乘法型高秩适配；

**🔧 技术方法**

技术包括SVD低秩分解、渐进式PTQ优化、基于STE的QAT训练、乘法型参数高秩PEFT以及专门的高效量化内核；

**📊 数据集**

实验使用LLaMA‑3、Qwen‑3等LLM，以及WikiText‑2、Penn Treebank、Commonsense‑170k等数据集进行评测；

**📈 对比分析**

在4‑bit量化下，LoRDS在Llama3‑8B上相较NF4、GPTQ、AWQ、LoftQ提升约0.6% PPL/accuracy；在3‑bit量化上提升8–13%；PEFT阶段比QLoRA提升4–9%且保持1.5×推理加速；

**⚠️ 局限性**

局限性包括需要额外的低秩矩阵参数和训练开销，对极低位宽（≤2.5‑bit）仍可能出现数值不稳定，且目前主要关注权重量化，激活量化研究待完善。

---

## 299. Qualitative Evaluation of LLM-Designed GUI

**arXiv ID:** 2601.22759 | [PDF](https://arxiv.org/pdf/2601.22759v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 300. A Unified Study of LoRA Variants: Taxonomy, Review, Codebase, and Empirical Evaluation

**arXiv ID:** 2601.22708 | [PDF](https://arxiv.org/pdf/2601.22708v1)

**作者:** Haonan He `[一作]` (Shanghai Artificial Intelligence Laboratory), Peng Ye `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 7856 | [OpenAlex ID](https://openalex.org/A5100729336)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对LoRA及其多种变体进行统一研究，提出细粒度分类、统一理论框架、可扩展代码库，并在三大领域进行大规模实验。

**💡 创新点**

首次构建以四个主轴为基础的LoRA变体细粒度分类，统一了理论视角并实现了统一接口的LoRAFactory代码库，极大提升了复现和扩展性。

**🔧 技术方法**

基于低秩更新动态的理论分析，采用PyTorch实现模块化LoRA及其各类变体，并在不同模型上采用AdamW+余弦调度等技术进行训练。

**📊 数据集**

使用RoBERTa-Base、Llama-3.1-8B-Base和CLIP-ViT-B/16等预训练模型，在GLUE、MetaMathQA、GSM8K、CodeFeedback、HumanEval以及七个图像分类数据集（如Stanford-Cars、EuroSAT等）上进行评测。

**📈 对比分析**

通过对20个代表性变体进行相同超参数搜索（除学习率外固定），发现LoRA在适当学习率下往往能匹配或超过大多数变体，学习率是决定性能的关键因素。

**⚠️ 局限性**

实验只覆盖了20个变体且主要聚焦于语言模型与图像分类，未探讨更大规模或跨模态任务，且需要对每种方法单独调优学习率，限制了评估的通用性。

---

## 301. RealSec-bench: A Benchmark for Evaluating Secure Code Generation in Real-World Repositories

**arXiv ID:** 2601.22706 | [PDF](https://arxiv.org/pdf/2601.22706v1)

**作者:** Yanlin Wang `[一作]` (Sun Yat-sen University), Zibin Zheng `[通讯]` (Sun Yat-sen University)

**通讯引用:** 33970 | [OpenAlex ID](https://openalex.org/A5000582109)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一个基于真实 Java 仓库、结合 SAST、LLM 过滤与人工验证的安全代码生成基准，构成 105 个涵盖 19 种 CWE、0~34 跳的实例。

**💡 创新点**

创新点在于：①使用多阶段流水线确保漏洞真实、可编译、可测试；②引入多 LLM 判别和人审双重校验来提高安全性判定；③设计联合功能与安全的 Composite 指标，揭示现有 LLM 在功能和安全双重要求下的显著差距。

**🔧 技术方法**

采用的技术包括：CodeQL 静态扫描、GPT‑4.1 等 LLM 进行误报过滤与文档重写、检索增强生成（BM25、Dense、SAST 数据流）以及多 LLM 投票判定安全性。

**📊 数据集**

数据集来自 4,000 个最受 Star 的 Java 仓库筛选出的 532 个高风险仓库，进一步提取 20,000+ 代码缺陷并精炼为 105 个实例。

**📈 对比分析**

评估方法：对 5 大 LLM 进行 Pass@k、Secure@k、SecurePass@k 三项指标测评，实验发现安全通过率（SecurePass@1）低于 6%，RAG 在提升功能准确率上有明显作用，但对安全提升几乎无效，安全提示在不同模型上表现不稳定。

**⚠️ 局限性**

局限性包括：SAST 误报仍影响安全评判；多 LLM 判别受限于 LLM 本身的判断准确性；基准仅覆盖 Maven‑based Java 开源项目，未涉及其他语言或私有代码，缺乏动态分析支持。

---

## 302. AEGIS: White-Box Attack Path Generation using LLMs and Training Effectiveness Evaluation for Large-Scale Cyber Defence Exercises

**arXiv ID:** 2601.22720 | [PDF](https://arxiv.org/pdf/2601.22720v1)

**作者:** Ivan K. Tung `[一作]` (Cyber Defence Test and Evaluation Centre), Lawrence Zheng `[通讯]` (Cyber Defence Test and Evaluation Centre)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了AEGIS系统，利用大型语言模型（LLM）、白盒网络访问和蒙特卡罗树搜索（MCTS），自动生成并验证网络攻击路径，用于网络防御演练的情境构建。

**💡 创新点**

创新点包括：
1) 引入白盒阶段化管线，将漏洞识别与攻击执行分离，减少错误累积；
2) 采用MCTS在真实执行环境中搜索攻击路径，实现基于实际成功率的动态路径规划；
3) 提供经过因子分析验证的12条量表问卷，量化演练体验四维度（感知学习、投入度、可信度、挑战度）。

**🔧 技术方法**

技术手段：LLM（Kimi K2 0905、Claude Sonnet 4.5）配合结构化提示、终端使用代理；网络扫描（PowerCLI + 端口/软件扫描脚本）；漏洞检索（Tavily、GitHub、ExploitDB）；漏洞验证器（文件写入检测）；MCTS搜索框架；问卷设计与因子分析。

**📊 数据集**

使用的数据集与环境：
- 真实CIDeX 2025演练网络（46台主机：Linux、Windows、网络设备）
- CVE与公开PoC仓库（ExploitDB、GitHub）
- 漏洞扫描结果、网络拓扑与权限信息
- 参与者问卷响应（225份）。

**📈 对比分析**

比较方法与性能：
- 将AEGIS生成的路径与人类设计路径在同一演练中交替使用；
- 通过事后问卷（12题）进行同质性检验；
- 采用两侧单边检验（TOST）与混合效应模型，设定等效界限d = ±0.5；
- 结果显示四个维度均在等效区间内，效应量均< 0.05；
- 实际开发周期从数月降至数天。

**⚠️ 局限性**

局限性：
- 依赖公开CVE与PoC代码，无法处理协议设计攻击或无公开漏洞的场景；
- 未生成持久化、批量传播或高级C2机制；
- 需要完整的白盒网络访问；
- 评估仅在单一演练环境中完成，缺乏跨场景验证；
- 可能受LLM训练数据泄漏影响，无法完全排除记忆化。

---

## 303. AutoMerge: Search-Based Model Merging Framework for Effective Model Reuse

**arXiv ID:** 2601.22748 | [PDF](https://arxiv.org/pdf/2601.22748v1)

**作者:** You Lu `[一作]` (Fudan University), Xin Peng `[通讯]` (Fudan University)

**通讯引用:** 13979 | [OpenAlex ID](https://openalex.org/A5071724015)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对不同模型架构（LLM、图像分类、自动驾驶）下的模型融合技术进行系统评估，并提出一种基于搜索的模型融合框架（简称 Search‑Merge）来实现多任务模型的零训练融合。

**💡 创新点**

① 首次在三大域内对五种主流融合技术进行交叉实验，揭示其对结构异构和超参数敏感性导致的性能波动；② 设计模型分段与区块级搜索策略，利用贝叶斯优化在不同块上自动选取融合技术与超参数，从而显著提升融合效果并降低能力差异；③ 通过实验证明该框架在不重新训练的前提下可匹配甚至超过全量微调模型。

**🔧 技术方法**

使用权重平均、Task Arithmetic、TIES、DARE‑Linear、DARE‑TIES 等融合技术；模型分段器依据张量形状切分网络；贝叶斯优化（随机森林代理、Log‑EI 采集）搜索区块级超参数；评估指标包括 PR（保留率）与 PD（保留差异）以及任务特定指标（HumanEval、MMLU、ImageNet Top‑1/5、CARLA Route Completion 等）。

**📊 数据集**

Llama2‑7B‑Code、Llama2‑7B‑Chat、CCT‑Organism、CCT‑Inanimate、Interfuser‑City、Interfuser‑Countryside、ImageNet‑1k、CARLA 8‑Towns、HumanEvalPack、MMLU‑Pro、CodeSearchNet、训练集（各领域细分数据集）。

**📈 对比分析**

与五种传统融合技术及全量微调进行对比。Search‑Merge 在三大域平均提升 PR 23.55%、降低 PD 51.94%；与传统搜索相比耗时减少 13–60%；与全量微调相比，构建多任务模型的时间缩短 62.4%、显存消耗降低 64.3%，性能几乎不逊色（PR/PD 与微调相当或更优）。

**⚠️ 局限性**

仅支持同一架构的任务专用模型；融合仍受超参数空间维度影响，区块级搜索耗时略高；在极端异构或大规模模型（如更大 LLM）下可进一步验证鲁棒性；目前未对不同架构模型融合进行探索。

---

## 304. UrbanMoE: A Sparse Multi-Modal Mixture-of-Experts Framework for Multi-Task Urban Region Profiling

**arXiv ID:** 2601.22746 | [PDF](https://arxiv.org/pdf/2601.22746v1)

**作者:** Pingping Liu `[一作]` (Jilin University), Irwin King `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 27305 | [OpenAlex ID](https://openalex.org/A5042251906)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 UrbanMoE，一种稀疏多模态多专家框架，用于在同一模型中同时预测碳排放、人口密度和夜间灯光强度等城市区域指标。

**💡 创新点**

创新点在于将稀疏 Mixture‑of‑Experts 与多模态融合结合，动态路由特定任务到专门专家，既捕捉任务间共享模式，又实现高效推理，显著降低参数量与显存消耗。

**🔧 技术方法**

技术包括：RemoteCLIP 视觉‑文本预训练编码器、POI 统计嵌入与区域嵌入、轻量级前馈专家网络、任务专属门控的稀疏路由、以及多任务回归损失。

**📊 数据集**

数据集：自行构建的三城基准（上海、北京、华盛顿 DC），每城包含 Sentinel‑2 卫星图像、POI 结构、文本摘要及 ODIAC、WorldPop、VIIRS DNB 目标指标，覆盖多模态与多任务。

**📈 对比分析**

与 Autoencoder、PCA、ResNet‑18、Tile2Vec、READ、PG‑SimCLR、UrbanCLIP、RemoteCLIP 等传统与最新方法对比，UrbanMoE 在三城平均提升 R² 约 10.7%/6.9%/1.6%，RMSE 与 MAE 大幅下降，且参数量仅为基线的 1‑2%，显存与推理时间也显著更优。

**⚠️ 局限性**

局限性包括：对不同城市的迁移泛化尚未充分验证；在北京人口预测上略逊 UrbanCLIP，表明任务间平衡仍存在挑战；实验仅涵盖三城市，需扩展至更多多样化地区以进一步验证鲁棒性。

---

## 305. Beauty and the Beast: Imperceptible Perturbations Against Diffusion-Based Face Swapping via Directional Attribute Editing

**arXiv ID:** 2601.22744 | [PDF](https://arxiv.org/pdf/2601.22744v1)

**作者:** Yilong Huang `[一作]` (Southeast University), Songze Li `[通讯]` (Southeast University)

**通讯引用:** 2743 | [OpenAlex ID](https://openalex.org/A5085853632)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6215c339-3735-4be3-8a07-5bbb7004712d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种面向扩散模型的主动防御框架 FaceDefense，用来生成既能有效干扰面部交换又视觉上几乎不可见的对抗样本。

**💡 创新点**

创新点在于：① 引入新的扩散损失使对抗扰动在扩散过程中的影响更强；② 通过方向性属性编辑恢复扰动造成的面部细节损失；③ 采用两阶段交替优化策略平衡防御强度与视觉不可见度；④ 在 W+ 空间完成多属性编辑，实现高分辨率、可控的扰动重构。

**🔧 技术方法**

核心技术包括：扩散模型（Stable Diffusion v1‑5）、Latent Diffusion 的噪声预测与导向、PGD 对抗优化、MaskFaceGAN/StyleGAN2 的属性编辑、面部解析与属性分类网络、两阶段交替梯度优化。

**📊 数据集**

使用公开的高分辨率人脸数据集：FFHQ、CelebA‑HQ、VoxCeleb2、XM2VTS、FRGC 进行实验与评估。

**📈 对比分析**

与基线方法（像素空间的 AdvDM、DiffusionGuard、SDS 等以及潜在空间的 MyFace）对比，FaceDefense 在防御指标（SSIM↓、PSNR↓、LPIPS↑、ATT_id↑）上均显著优于对手；在视觉指标（SSIM↑、PSNR↑、LPIPS↓）上也实现更高的不可见度，尤其在 ϵ=75/255 时表现最为突出。

**⚠️ 局限性**

局限性包括：对极低质量交换模型（如 DiffSwap）防御效果有限；在某些压缩/模糊处理后仍需进一步提升鲁棒性；跨模型迁移仍不够稳健，部分对抗样本在不同 LDM 交换器上效果下降；若被攻击者获取对抗策略，存在反向工程风险。

---

## 306. Vision-Language Models Unlock Task-Centric Latent Actions

**arXiv ID:** 2601.22714 | [PDF](https://arxiv.org/pdf/2601.22714v1)

**作者:** Alexander Nikulin `[一作]` (AIRI), Vladislav Kurenkov `[通讯]` (Innopolis University)

**通讯引用:** 24 | [OpenAlex ID](https://openalex.org/A5010816959)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究利用视觉语言模型（VLM）的可提示表征过滤动作相关干扰，从而提升潜在动作模型（LAM）在存在干扰时的学习效果。

**💡 创新点**

首次将VLM的提示表征作为LAM的无监督学习目标，证明其能显著恢复LAM在干扰场景下的成功率。

**🔧 技术方法**

采用VLM提示嵌入、LAM的IDM/FDM框架、行为克隆和动作解码器等技术。

**📊 数据集**

在改造后的Distracting MetaWorld（MT10）数据集上进行实验，并对比多种VLM（Molmo、Gemma‑3、Phi‑4等）。

**📈 对比分析**

与传统LAPO、OTTER、UniVLA基线相比，提示表征的LAM在干扰环境中成功率提升约6倍，且在无干扰环境下保持或略优表现。

**⚠️ 局限性**

局限包括对VLM的依赖、对提示和聚合策略的调优敏感，以及在更复杂真实场景中的可迁移性尚未验证。

---

## 307. CONCUR: High-Throughput Agentic Batch Inference of LLM via Congestion-Based Concurrency Control

**arXiv ID:** 2601.22705 | [PDF](https://arxiv.org/pdf/2601.22705v1)

**作者:** Qiaoling Chen `[一作]` (Nanyang Technological University), Tianwei Zhang `[通讯]` (Nanyang Technological University)

**通讯引用:** 2793 | [OpenAlex ID](https://openalex.org/A5028270700)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了面向代理的主动KV缓存入场控制层，防止中期缓存抖动导致的吞吐量下降。

**💡 创新点**

创新点在于将网络拥塞控制的AIMD思想迁移到GPU KV缓存调度，实现了基于缓存利用率和命中率的动态入场窗口。

**🔧 技术方法**

采用AIMD算法、实时缓存利用率和命中率监控、代理级暂停/恢复等技术构建控制层。

**📊 数据集**

在Qwen3‑32B、DeepSeek‑V3等大模型与真实代理工作负载（RL rollouts、数据蒸馏、评估）上进行评测。

**📈 对比分析**

与SGLang、请求级限流和HiCache三种基线对比，吞吐量提升最高可达4.09×（Qwen3‑32B）和1.90×（DeepSeek‑V3），并保持较高缓存命中率。

**⚠️ 局限性**

局限在仅针对离线批量代理推理，需手动设置阈值，且对高并发PCIe传输和异步工具调用的鲁棒性仍有限。

---

## 308. Best-of-Q: Improving VLM agents with Q-function Action Ranking at Inference

**arXiv ID:** 2601.22701 | [PDF](https://arxiv.org/pdf/2601.22701v1)

**作者:** Emilien Biré `[一作]` (H Company), Kai Yuan `[通讯]` (H Company)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种在推理时通过对 VLM 生成的候选动作使用离线训练的 Q 函数重新排序，来提升 VLM 代理性能的框架。

**💡 创新点**

创新点在于将 VLM 的高容量动作生成与价值基础的动作选择解耦，并利用轻量级 Q 函数在推理阶段即时改进决策，而不需要重新训练 VLM。

**🔧 技术方法**

使用了冻结的 Vision‑Language Model（如 Qwen2.5‑VL‑7B / GPT‑4.1）生成动作候选，利用 IQL（Implicit Q‑Learning）训练一个小型 MLP Q 函数，并在 WebVoyager 基准上进行评估。

**📊 数据集**

主要数据集为 WebVoyager 的 590 个真实网页任务数据，并通过 ε‑greedy 策略在不同 VLM 上采集的离线交互数据用于 Q 函数训练。

**📈 对比分析**

与 Prompting、Random Action 以及 VLM 直接挑选动作的 baseline 进行对比，方法在 Qwen2.5‑VL‑7B 上将成功率从 38.8% 提升至 55.7%，在 GPT‑4.1 上从 82.4% 提升至 88.8%，同时保持了较低的步骤数和成本。

**⚠️ 局限性**

局限性在于仅能从 VLM 提议的候选动作中挑选，若 VLM 未生成合适动作，Q 函数无法弥补，导致性能受限于动作提议质量。

---

## 309. Models Know Models Best: Evaluation via Model-Preferred Formats

**arXiv ID:** 2601.22699 | [PDF](https://arxiv.org/pdf/2601.22699v1)

**作者:** Joonhak Lee `[一作]` (Seoul National University), Jaejin Lee `[通讯]` (Seoul National University)

**通讯引用:** 2310 | [OpenAlex ID](https://openalex.org/A5100767175)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文探讨了多项选择题评估中符号式与空白式两种格式的性能差异，并提出了基于模型内部偏好的动态格式对齐策略。

**💡 创新点**

创新点在于利用模型生成的偏好信号而非人工规则，实现实例级的格式选择，从而显著提升LLM在不同任务上的零样本准确率。

**🔧 技术方法**

采用DeBERTaV3编码器进行轻量级分类器微调，结合模型对比得分、置信度差距等信号，并通过自监督或投票标签进行训练。

**📊 数据集**

使用了八个公开MCQA基准（MMLU、ARC、OpenBookQA、CommonsenseQA、HellaSwag、PIQA、WinoGrande、SocialIQA）以及多种LLM（如Llama、Mistral、Qwen、GPT-4等）进行评估。

**📈 对比分析**

通过与传统固定格式评估进行对比，实验表明在大多数以完成式为主的任务上性能提升可达数个百分点，符号式任务保持不变，显示方法鲁棒。

**⚠️ 局限性**

局限在于仅针对解码器式LLM，且需要先行运行模型生成标签，增加预处理成本；对编码器或混合架构的推广仍待验证。

---

## 310. Decomposing Epistemic Uncertainty for Causal Decision Making

**arXiv ID:** 2601.22736 | [PDF](https://arxiv.org/pdf/2601.22736v1)

**作者:** Md Musfiqur Rahman `[一作]` (Purdue University), Murat Kocaoglu `[通讯]` (Johns Hopkins University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种新的框架，用置信集合和神经因果模型来分解因果效应估计的不确定性，将其分为非可识别性（non‑ID）区间和样本不确定性区间，并给出决策路径（返回、观察变量、收集更多样本）。

**💡 创新点**

创新点在于首次将有限样本与结构不可识别性这两种不确定性统一建模，并通过求解 min–max / max–min 优化得到交集与差集，实现在观测数据有限时判定是否需要增样本或增变量；同时设计了 ε‑net 搜索与深度因果模型的 Lagrangian 正则化训练。

**🔧 技术方法**

技术手段包括：置信集合构造（Hoeffding + Bonferroni）、ε‑net 采样、深度因果模型（Deep Causal Model, DCM）训练、分布距离正则化、双重目标（最大/最小化因果效应）以及基于神经网络的分布匹配与 ATE 估计。

**📊 数据集**

实验使用：两类合成结构（Bow 图、IV 图、含观测/未观测混杂的图）、两套敏感性合成 SCM、以及真实数据 Parents’ Labor Supply（约 92k 家庭记录，包含子女性别作为工具变量）。

**📈 对比分析**

与闭式公式、AutoBound、NCM、倾向匹配、距离匹配等基线比较。基线仅给出宽泛的区间，无法提示增样本或增变量；而本文方法在 1000–3000 样本和 92k 样本上均能准确定位非可识别区间大小并给出下一步行动，显著缩小外层带宽，整体表现优于基线。

**⚠️ 局限性**

局限性包括：ε‑net 近似可能遗漏最优分布；计算量随变量数和支持度急剧增加；方法假设变量离散、结构已知、半马尔可夫 SCM；对连续变量、非离散分布的推广仍需研究。

---

## 311. Is Training Necessary for Anomaly Detection?

**arXiv ID:** 2601.22763 | [PDF](https://arxiv.org/pdf/2601.22763v1)

**作者:** Xingwu Zhang `[一作]` (Hunan University), Zijun Long `[通讯]` (Hunan University)

**通讯引用:** 107 | [OpenAlex ID](https://openalex.org/A5102631471)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种训练‑free 的检索式多类别无监督异常检测框架 RAD，直接使用冻结编码器提取的正常特征作为检索库，利用多层全局与局部检索来生成异常分数。

**💡 创新点**

创新点：①在表示空间证明 encoder–decoder 重建方法存在保真–稳定矛盾；②提出不使用重建网络、只做检索的异常评分方式；③理论上证明检索分数是重建残差分数的上界；④设计多层记忆库与多级检索（全局+局部+多层融合）实现高效定位。

**🔧 技术方法**

技术手段：使用冻结的 ViT（DINOv3）编码器提取多层 patch 特征；构建多层记忆库；对测试图像做全局相似检索筛选候选参考图；再做局部空间检索并用 1‑NN cosine 距离得到 patch 异常分数；多层分数加权融合得到像素级异常地图；不进行任何模型训练。

**📊 数据集**

数据集：MVTec‑AD、VisA、Real‑IAD、3D‑ADAM 进行标准 MUAD 与少样本（few‑shot）实验；另外在 cold‑start、增量分类等设置下评估。

**📈 对比分析**

比较方法：与多种 SOTA（RD4AD, UniAD, DiAD, MambaAD, Dinomaly, OmiAD, SimpleNet, DeSTSeg 等）以及专门的 few‑shot 方法（IIPAD, PromptAD, WinCLIP, SPADE, PatchCore 等）对比；在所有四大 benchmark 上均实现 state‑of‑the‑art，尤其像素级定位 AUROC/P‑AP/P‑F1‑max 等指标显著提升；在 1‑shot/少样本设置下超越现有专门算法；在 cold‑start 与增量类实验中表现出优异的数据效率和迁移能力。

**⚠️ 局限性**

局限性：需要存储所有正常 patch 的特征，内存占用较大；检索过程在推理时比单前向传播耗时，导致延迟较高；目前仅支持 2D 视图，3D 完整信息尚未利用。

---

## 312. Procedural Knowledge Extraction from Industrial Troubleshooting Guides Using Vision Language Models

**arXiv ID:** 2601.22754 | [PDF](https://arxiv.org/pdf/2601.22754v1)

**作者:** Guillermo Gil de Avalle `[一作]` (University of Groningen), Christos Emmanouilidis `[通讯]` (University of Groningen)

**通讯引用:** 2533 | [OpenAlex ID](https://openalex.org/A5005647074)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对工业故障排除图纸中的程序知识进行结构化抽取，并比较两种开源视觉语言模型及两种提示策略的表现

**💡 创新点**

首次在工业故障排除图纸上系统评估VLM的抽取能力，揭示模型特定失效模式（如无穷循环幻觉）和提示对结果的显著影响

**🔧 技术方法**

使用Pixtral‑12B和Qwen2‑VL‑7B两款开源视觉语言模型，并通过标准指令提示与包含视觉符号说明的增强提示进行实验

**📊 数据集**

使用12份荷兰制造商提供的故障排除手册（共24页）作为评测数据集，包含约548个实体和536条关系

**📈 对比分析**

采用实体/关系的精度、召回率和F1分数进行评估，实体F1最高达0.34，关系F1低于0.11；Qwen2‑VL在增强提示下关系F1提升75%，但Pixtral表现下降

**⚠️ 局限性**

主要限制包括：关系抽取性能极低、模型易产生无穷循环幻觉、对视觉结构的捕捉不足、数据集规模小且语言单一，难以推广至更广泛的工业场景

---

## 313. ImgCoT: Compressing Long Chain of Thought into Compact Visual Tokens for Efficient Reasoning of Large Language Model

**arXiv ID:** 2601.22730 | [PDF](https://arxiv.org/pdf/2601.22730v1)

**作者:** Xiaoshu Chen `[一作]` (National University of Defense Technology), Xinwang Liu `[通讯]` (National University of Defense Technology)

**通讯引用:** 19639 | [OpenAlex ID](https://openalex.org/A5101727888)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出将链式推理（CoT）压缩为视觉令牌，设计 ImgCoT 与 L‑ImgCoT 两个框架，既可大幅减少推理令牌，又能保留必要的推理细节。

**💡 创新点**

核心创新在于把 CoT 的重构目标从文本改为图像，使压缩过程受到空间归纳偏差驱动，优先保留推理结构；随后通过关键步骤筛选（基于 token 似然）将少量文本步骤与视觉令牌混合，兼顾全局结构与细节。

**🔧 技术方法**

技术手段包括：① 视觉文本渲染（将文本转为带框、箭头的图像）; ② TiTok 1D 视觉自编码器 + VQ‑VAE 量化实现压缩; ③ 在 LLM 上进行自回归训练并使用 LoRA 微调; ④ 通过 token 似然阈值筛选关键文本步骤；整体实现了无视觉编码器推理的闭环。

**📊 数据集**

主要使用的数据集有：
- MathPile（训练视觉自编码器）
- GSM8K、MATH、GPQA‑extended、ProsQA（评估基准）
- MetaMathQA（微调训练）
- Gaokao‑Math‑2023、Svamp、MultiArith、SingleEq（跨域泛化测试）。

**📈 对比分析**

与 Full‑CoT、Coconut、ICoT、CODI、CoLaR 等最新隐式 CoT 压缩方法进行对比。实验表明，ImgCoT 在多模型（Qwen2.5‑0.5B、1.5B、Llama3.2‑3B）和多数据集上，往往匹配甚至超越 Full‑CoT，同时令牌消耗显著降低；L‑ImgCoT 在保持约 30% 推理成本下降的同时，进一步提升准确率，尤其在逻辑与专业领域推理任务中表现突出。

**⚠️ 局限性**

主要局限包括：
① 视觉令牌对细节的模糊化可能影响对领域特定细节的把握；
② 需要手工设计视觉渲染方案，增加前处理成本；
③ 训练过程较为复杂，需多阶段（编码器、LLM 微调）；
④ 仅在部分大型语言模型上验证，尚未证明在更大模型或多语言场景中的可迁移性。

---

## 314. Rust and Go directed fuzzing with LibAFL-DiFuzz

**arXiv ID:** 2601.22772 | [PDF](https://arxiv.org/pdf/2601.22772v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 315. A Step Back: Prefix Importance Ratio Stabilizes Policy Optimization

**arXiv ID:** 2601.22718 | [PDF](https://arxiv.org/pdf/2601.22718v1)

**作者:** Shiye Lei `[一作]` (University of Sydney), Dacheng Tao `[通讯]` (Nanyang Technological University)

**通讯引用:** 98256 | [OpenAlex ID](https://openalex.org/A5074103823)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在 RL 后训练阶段，重新审视离线（off‑policy）策略梯度，发现 token‑level 的重要性采样比率无法准确补偿分布漂移，于是提出了基于 prefix importance ratio 的 MinPRO 方法，利用最小前缀比率作为非累计、稳定的修正因子，提升 LLM 的训练稳定性与推理性能。

**💡 创新点**

核心创新在于将理论上正确的 prefix importance ratio 取代传统 token‑level 近似，并通过“最小前缀比率”简化成非累计形式，既保留了前缀信息，又消除了累计乘积导致的数值不稳定和长度偏差。

**🔧 技术方法**

使用 critic‑free RL 框架（如 CISPO 软裁剪），在目标策略与采样策略之间加入 MinPRO 修正；实现上采用软裁剪+最小前缀比率乘积的梯度约束；通过 VeRL 框架完成大规模离线 roll‑out 与更新。

**📊 数据集**

训练集为 DAPO‑Math‑17K（17000 题），评估基准包括 AMC23、AIME24、AIME25、MATH500、Olympiad、Minerva、GSM8K。

**📈 对比分析**

与 GRPO、GSPO、CISPO、M2PO 等主流基线在 8B、14B 与 30B MoE LLM（Qwen3‑8B‑Base、Qwen3‑14B‑Base、Qwen3‑30B‑A3B‑Base）上进行离线 RL 对比，结果显示 MinPRO 在大 off‑policy 情况下实现更高、更稳定的训练奖励，并在 Pass@k 上至少提升 0.5~1 分（相较于基线）。

**⚠️ 局限性**

局限性包括：只在数学推理任务上验证，未涉及通用对话或代码生成；对超参数（clip 范围、缓冲区大小等）仍较敏感；虽然减小了长度偏差，但在极长序列下仍可能出现极端比率导致梯度失衡；未对多任务或跨域泛化进行系统评估。

---

## 316. Bi-MCQ: Reformulating Vision-Language Alignment for Negation Understanding

**arXiv ID:** 2601.22696 | [PDF](https://arxiv.org/pdf/2601.22696v1)

**作者:** Tae Hun Kim `[一作]` (Inha University), Hyun Gyu Lee `[通讯]` (Inha University)

**通讯引用:** 43933 | [OpenAlex ID](https://openalex.org/A5100383157)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出 Bi‑MCQ 细调策略，改写 Vision‑Language 对齐为双向多选问答，显著提升对医学影像中否定陈述的理解。

**💡 创新点**

创新点在于：①将对齐目标从全局相似度最大化改为条件语义比较；②采用双向交叉注意力模块解耦图像→文本与文本→图像推理；③引入混合提示增强泛化。

**🔧 技术方法**

使用 CLIP‑style 双编码器（如 ViT‑B/16 + BioBERT），双向交叉注意力 Fusion，交叉熵损失的 Bi‑MCQ 训练框架。

**📊 数据集**

在 ChestXray14 进行细调，使用 Open‑I、CheXpert、PadChest 进行跨域评测；同时对 CARZero、MedKLIP、KAD 等预训练模型进行对比。

**📈 对比分析**

与 InfoNCE 细调相比，Bi‑MCQ 在正负两侧 AUC 均提升，最大正负差距缩小约0.12，正负综合（PNC）AUC 提升至 0.08，整体在多数据集上保持稳定或更优的性能。

**⚠️ 局限性**

局限性包括：对域外数据（如 PadChest）仍有适配瓶颈；需要手工设计正负提示，且对其他视觉任务的适用性仍待验证。

---

## 317. FNF: Functional Network Fingerprint for Large Language Models

**arXiv ID:** 2601.22692 | [PDF](https://arxiv.org/pdf/2601.22692v1)

**作者:** Yiheng Liu `[一作]` (Northwestern Polytechnical University), Xintao Hu `[通讯]` (Northwestern Polytechnical University)

**通讯引用:** 5053 | [OpenAlex ID](https://openalex.org/A5030799310)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种无训练、样本高效的LLM指纹识别方法——功能网络指纹（Functional Network Fingerprint, FNF），用于判断一个可疑模型是否源自已知受害模型；

**💡 创新点**

创新点在于将神经科学中的功能网络概念迁移到LLM，利用ICA分解模型内部激活并比较功能网络的动态一致性，从而实现跨尺度、跨架构甚至权重重包装（weight repackaging）攻击下的鲁棒指纹识别；

**🔧 技术方法**

核心技术包括：1）对Transformer层输出进行空间ICA（CanICA）提取功能网络；2）在多样本上计算功能网络时间序列的Spearman相关性；3）构造K×K一致性矩阵作为指纹；

**📊 数据集**

主要使用WikiText‑2作为输入样本来评估指纹稳定性，并在多种公开LLM（LLaMA2、Qwen、Vicuna、ChatGLM、Mistral、EvolveLM等）上进行实验；

**📈 对比分析**

与基线REEF（利用CKA进行内部表示相似度）对比，FNF在识别相同家族模型（例如同系列的不同规模、微调或权重重包装后模型）时准确率高，且对参数置换、剪枝、缩放、模型合并等攻击具有更高鲁棒性；

**⚠️ 局限性**

局限性包括：1）对模型内部激活的统计学习可能受输入分布影响；2）对极大规模或高维度模型的ICA计算成本仍不低；3）在极端剪枝或训练差异过大的情况下，功能网络一致性下降，导致误判。

---

## 318. Okara: Detection and Attribution of TLS Man-in-the-Middle Vulnerabilities in Android Apps with Foundation Models

**arXiv ID:** 2601.22770 | [PDF](https://arxiv.org/pdf/2601.22770v1)

**作者:** Haoyun Yang `[一作]` (AffilOne), Xianghang Mi `[通讯]` (AffilTwo)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研发Okara框架，用基础模型驱动的GUI代理自动高覆盖率交互，检测Android应用中的TLS MitM漏洞；随后利用LLM分类器与动态Instrumentation对漏洞代码进行精确归因与分类；

**💡 创新点**

首次将大规模预训练模型用于移动App UI自动化交互以及漏洞代码归因，构建新的TLS MitM漏洞分类体系，并显著提升检测覆盖率和归因准确度；

**🔧 技术方法**

基于生成式预训练语言模型（LLM）的GUI代理、动态Instrumentation、LLM驱动的代码分类器、Android UI自动化工具；

**📊 数据集**

37,349个来自Google Play及第三方商店的Android应用程序；

**📈 对比分析**

与传统低覆盖率的手动/脚本化交互检测方法对比，Okara在同样规模下发现22.42%（8,374）可利用漏洞，覆盖率和检测率均显著提升；在归因阶段，TMV-ORCA将41%漏洞定位至第三方库并揭示空信任管理器、主机名验证失效等重复不安全模式；

**⚠️ 局限性**

依赖LLM模型的生成与推理质量，可能对不常见UI路径或非UI触发的漏洞识别不足；模型推理成本和计算资源需求高；仅在Android平台评估，跨平台适用性尚未验证；

---

## 319. AscendCraft: Automatic Ascend NPU Kernel Generation via DSL-Guided Transcompilation

**arXiv ID:** 2601.22760 | [PDF](https://arxiv.org/pdf/2601.22760v1)

**作者:** Zhongzhen Wen `[一作]` (Nanjing University), Tian Zhang `[通讯]` (Nanjing University)

**通讯引用:** 9218 | [OpenAlex ID](https://openalex.org/A5100371729)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

本文提出一种基于DSL的自动化 AscendC 核心生成框架，利用大型语言模型先在 DSL 级别生成高层次计算与调度，再通过多阶段结构化转译得到可编译、功能正确的 AscendNPU 内核；

**💡 创新点**

创新点在于设计了轻量级、专门针对 AscendNPU 的 DSL，显式抽象并隐藏了低层细节，同时采用多阶段 LLM 传译与错误回馈机制，显著提升了 NPU 内核生成的正确率和性能；

**🔧 技术方法**

使用的技术包括大型语言模型（LLM）进行代码生成与调优、DSL 设计与示例驱动、分阶段结构化转译（四个传译通道）、编译器反馈回传、以及基于 AscendC 编程模型的硬件友好约束；

**📊 数据集**

评估数据集为 MultiKernelBench（覆盖激活、损失、算术、归一化、优化器、归约、池化七大类 52 个单算子内核），并在新提出的 mHC 结构上测试了两种自定义算子；

**📈 对比分析**

通过与 PyTorch eager 基线对比，生成的内核在 82.7% 的算子上至少达到 20% 的性能（Fast_0.2），57.7% 达到 80%（Fast_0.8），46.2% 与基线持平或超越（Fast_1.0），且在 mHC 任务中单一生成器可实现 6.6×/3.0× 的加速，后续人工+LLM 优化后可达 15.9×/7.2×；

**⚠️ 局限性**

局限性包括对矩阵乘法与卷积算子尚未覆盖，部分算子（如数学与池化）在性能与正确率上仍有提升空间，且仍依赖人工提供高质量 DSL 示例与对齐策略，无法完全自动化最优化流程；

---

## 320. Unveiling Scaling Behaviors in Molecular Language Models: Effects of Model Size, Data, and Representation

**arXiv ID:** 2601.22757 | [PDF](https://arxiv.org/pdf/2601.22757v1)

**作者:** Dong Xu `[一作]` (Shenzhen University), Junkai Ji `[通讯]` (Shenzhen University)

**通讯引用:** 1589 | [OpenAlex ID](https://openalex.org/A5046906366)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

对分子语言模型进行大规模可扩展性研究，训练了300个模型并完成10,000+实验，系统考察了模型规模、训练数据量和分子表示对预训练损失及下游任务性能的影响。

**💡 创新点**

提出了基于计算预算的双变量缩放定律，揭示分子表示会显著改变计算最优前沿，并解释了先前在分子生成任务中观察到的“无缩放”现象；同时公开了迄今最大规模的分子语言模型库，供后续研究使用。

**🔧 技术方法**

使用GPT‑style 自回归Transformer进行预训练，并通过LoRA实现轻量级微调；利用token平均交叉熵评估预训练损失，并基于C∝PD的计算预算公式推导计算最优模型规模与训练量。

**📊 数据集**

数据来源为ZINC、UniChem等化学数据库，将分子转换为DeepSMILES、FragLink、FragSeq、SAFE、SMILES等五种字符串表示；预训练数据量覆盖100M–3B token范围。

**📈 对比分析**

在MoleculeNet九个基准任务（包括BACE、HIV、BBBP、Sider、Tox21、ClinTox、ESOL、FreeSolv、Lipophilicity）中，将模型与现有SOTA进行对比，证明预训练的缩放趋势能有效迁移到下游任务；不同表示在不同任务上表现出最佳性能，体现了任务依赖性。

**⚠️ 局限性**

局限性包括：生成与优化指标易饱和且高度依赖采样或搜索策略，不能直接用于评估缩放；模型规模受限于GPU算力，未探索更大规模或更多表示；实验主要集中在单一任务设置，未来需扩展至更广泛的化学任务。

---

## 321. Understanding Generalization from Embedding Dimension and Distributional Convergence

**arXiv ID:** 2601.22756 | [PDF](https://arxiv.org/pdf/2601.22756v1)

**作者:** Junjie Yu `[一作]` (Southern University of Science and Technology), Quanying Liu `[通讯]` (Southern University of Science and Technology)

**通讯引用:** 3785 | [OpenAlex ID](https://openalex.org/A5078854583)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了以嵌入表示几何为核心的后训练泛化误差理论，并给出了包含内在维度和下游映射 Lipschitz 稳定性的泛化上界。

**💡 创新点**

创新点在于：
1) 明确地将泛化误差与单个已训练模型的嵌入分布内在维度关联；
2) 在最终层将架构敏感性消除，只剩下维度和分布收敛项；
3) 通过 Wasserstein 收敛率与内在维度的理论关系，解释了大模型为何仍能泛化；
4) 通过宽度干预实验验证了维度与 Lipschitz 常数的交互作用。

**🔧 技术方法**

主要技术包括：
- Wasserstein 距离和上层 Wasserstein 维度概念；
- 对嵌入分布的内在维度估计（MLE 估计器）；
- 通过谱范数上界近似网络的 Lipschitz 常数；
- 结合梯度界定的光滑损失函数；
- 经验风险与贝叶斯预测的对比。

**📊 数据集**

实验使用的数据集包括：MNIST（autoencoder）、CIFAR‑10/100（ResNet、MLP）、ImageNet‑1K（预训练视觉模型）和 MNLI（预训练语言模型）。

**📈 对比分析**

与传统参数空间容量上界（VC、Rademacher、PAC‑Bayesian）相比，本文的度量在不同架构、模型规模和数据量下更能解释泛化差异；实验表明，最终层的内在维度与 Wasserstein 距离与测试误差呈显著正相关，说明该指标在多种任务中都具有较高的预测性。

**⚠️ 局限性**

主要局限包括：
- 上界中的常数可能较松，理论与实际误差之间的距离不易量化；
- 需要 Lipschitz 连续性、光滑损失和支持有界等假设，实际情况可能不完全满足；
- Lipschitz 常数的估计在深层网络中仍不易实现；
- 对贝叶斯预测的依赖限制了对非监督/自监督场景的直接适用。

---

## 322. OSNIP: Breaking the Privacy-Utility-Efficiency Trilemma in LLM Inference via Obfuscated Semantic Null Space

**arXiv ID:** 2601.22752 | [PDF](https://arxiv.org/pdf/2601.22752v1)

**作者:** Zhiyuan Cao `[一作]` (Shanghai Key Laboratory of Computer Software Testing and Evaluating), Mingang Chen `[通讯]` (Shanghai Key Laboratory of Computer Software Testing and Evaluating)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 OSNIP，一种轻量化的客户端加密框架，将原始嵌入投射到“伪装语义零空间”，实现 LLM 推理过程中的隐私保护。

**💡 创新点**

创新点在于利用高维表示中的伪装语义零空间实现无后处理的隐私投影，并通过密钥条件化个性化与动态学习策略同时满足隐私、效能与效率三重最优。

**🔧 技术方法**

采用几何正交约束的加密网络，结合 KL 目标保持、夹角阈值正则、键分离正则；使用高维球面投影、动态优先学习（Utility‑Gated Curriculum）以及无后处理的直接加密机制。

**📊 数据集**

训练使用 65k 条混合语料（80% Alpaca，20% ARC‑Easy/Challenge 与 SciQ），评估基准包括 MMLU、ARC‑Easy、HellaSwag、PIQA、MNLI、SST2、ANLI、WiC、WikiText‑2、CNN/DailyMail 等 12 个标准任务。

**📈 对比分析**

与 Cape、DYNTEXT、InferDPT 等基线对比，在 10 个闭端任务中 OSNIP 保持 100%+ 预测准确率，KNN 攻击 ASR 归零；在开放生成任务中 PPL 仅升幅 1%–3%；整体性能显著优于现有方法，尤其在隐私‑效能权衡上表现突出。

**⚠️ 局限性**

局限性：依赖可信第三方获取服务器端梯度；对小规模 LLM 隐私‑效能平衡不如大模型理想；目前仅针对文本推理，扩展到多模态或更大规模模型仍需进一步验证。

---

## 323. AR-BENCH: Benchmarking Legal Reasoning with Judgment Error Detection, Classification and Correction

**arXiv ID:** 2601.22742 | [PDF](https://arxiv.org/pdf/2601.22742v1)

**作者:** Yifei Li `[一作]` (Beihang University), Pengchong Li `[通讯]` (People's Procuratorate of Beijing Municipality)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了法律判决审查任务，构建了包含8,700份细致标注判决和34,617份补充文本的中文判决错误检测/分类/纠错基准。

**💡 创新点**

首创将判决错误细分为六类，并提出从错误检测到分类再到纠错的连续子任务框架，构建了大规模精细标注数据集。

**🔧 技术方法**

利用大型语言模型（LLM）进行评估，采用链式思维、法律条文补充等多种输入设定，并使用准确率、宏观F1、ImpScore等指标评测。

**📊 数据集**

使用来自中国裁判文书网的判决文本，手工注入六类错误，形成8,700份标注判决和34,617份补充文本的综合数据集。

**📈 对比分析**

对比14种LLM（通用、推理增强、领域特定）及SOTA SLM，发现错误检测最强、纠错最弱，通用LLM性能优于领域特定模型，人工评测结果远优于现有模型。

**⚠️ 局限性**

研究仅覆盖中国民事判决，未包含英美普通法；未涵盖法律解释/论证任务；评估依赖现有LLM，可能带偏；数据存在地区、时段偏差，限制了基准的普适性。

---

## 324. Lingua-SafetyBench: A Benchmark for Safety Evaluation of Multilingual Vision-Language Models

**arXiv ID:** 2601.22737 | [PDF](https://arxiv.org/pdf/2601.22737v1)

**作者:** Enyi Shi `[一作]` (Nanjing University of Science and Technology), Tat-Seng Chua `[通讯]` (National University of Singapore)

**通讯引用:** 60236 | [OpenAlex ID](https://openalex.org/A5089404640)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Lingua‑SafetyBench，一个覆盖10种语言、100,440个语义对齐的图文危险对的多模多语安全基准；

**💡 创新点**

创新点在于将危险来源显式分为图像主导和文本主导两类，实现对多模危险源的可控归因；

**🔧 技术方法**

采用GPT‑5.1与Qwen‑Guard自动评判器、扩散模型生成图像、风险对齐翻译策略以及三阶段构建流程；

**📊 数据集**

使用自研的图文对、MM‑SafetyBench、VLGuard、XSAFETY等公开数据进行构造与翻译；

**📈 对比分析**

对11款开源VLLM进行ASR评测，发现高资源语言图像主导风险更易成功、低资源语言文本主导风险更易成功，模型规模提升对高资源语言的安全收益更大；

**⚠️ 局限性**

局限在于评判器偏向高资源语言、对抗性测试范围有限，且提示式安全增强方法效果不佳。

---

## 325. MM-THEBench: Do Reasoning MLLMs Think Reasonably?

**arXiv ID:** 2601.22735 | [PDF](https://arxiv.org/pdf/2601.22735v1)

**作者:** Zhidian Huang `[一作]` (Tsinghua University), Juanzi Li `[通讯]` (Tsinghua University)

**通讯引用:** 14531 | [OpenAlex ID](https://openalex.org/A5003324011)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 MM‑THEBench，用来评估多模态大语言模型在产生中间 Chain‑of‑Thought (CoT) 过程中的幻觉情况。

**💡 创新点**

提出了基于知识、感知与推理三大认知维度的细粒度幻觉分类体系，并设计了可自动化的多层级评估框架和 rubric‑based 评估方法。

**🔧 技术方法**

使用 LLM‑as‑Judge（Qwen‑3‑32B）实现全自动化评价；结合细粒度 rubric 进行步骤匹配与评分；对多模态输入进行注释与对齐。

**📊 数据集**

从 MathVision、MM‑vet‑v2、MMMU‑pro、HallusionBench、Omni‑Spatial、CharXiv、GUI‑Agent、Video‑MME 等现有高质量多模态数据集中抽取 1,340 条样本进行标注。

**📈 对比分析**

对 14 种主流推理 MLLM 采用答案级、步骤级和 rubric 级三层评估，发现中间 CoT 的正确率显著低于最终答案，幻觉类型与模型规模、推理深度有关，较大模型在知识与推理维度表现更好，但感知维度仍易产生空间幻觉。

**⚠️ 局限性**

受限于裁判模型的主观性与人工审核成本、数据规模有限、对闭源模型 CoT 的近似可能导致评估偏差、幻觉子类覆盖仍有不足等问题。

---

## 326. Local Intrinsic Dimension of Representations Predicts Alignment and Generalization in AI Models and Human Brain

**arXiv ID:** 2601.22722 | [PDF](https://arxiv.org/pdf/2601.22722v1)

**作者:** Junjie Yu `[一作]` (Southern University of Science and Technology), Quanying Liu `[通讯]` (Southern University of Science and Technology)

**通讯引用:** 3785 | [OpenAlex ID](https://openalex.org/A5078854583)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

研究了视觉AI模型与人脑在表征空间中的收敛关系，探讨了模型间、模型与大脑以及模型与泛化性能之间的关联。

**💡 创新点**

提出本质上只有本地内在维度（local intrinsic dimension）能同时解释模型间相似度、模型与大脑的对齐以及泛化表现，并将模型规模与数据规模的提升与本地维度降低联系起来，给出一种几何学视角的规模解释。

**🔧 技术方法**

采用了PCA+Ridge回归的表征对齐方法，使用最大似然估计（MLE）测算嵌入的本地内在维度，并在多尺度（不同邻域大小K）下评估维度与对齐/泛化的相关性。

**📊 数据集**

主要使用了视觉自然场景数据集NSD（包含高分辨率fMRI响应）以及ImageNet-1K、12K、22K的图像数据来训练并评估各类视觉模型。

**📈 对比分析**

通过AI–Brain、AI–AI R²对齐得分和ImageNet-1K Top‑1准确率进行互相关分析，发现本地维度越低，AI–AI对齐、AI–Brain对齐以及泛化表现越好；在多种架构（ViT、ConvNeXt、ResNet、ResMLP）中保持一致性，且大模型/大数据训练的模型在这些指标上表现最佳。

**⚠️ 局限性**

局限在于仅针对视觉任务和视觉皮层；对其它模态（语言、听觉）的推广尚未验证；本地内在维度与学习机制之间的理论因果关系仍不清晰。

---

## 327. Gated Relational Alignment via Confidence-based Distillation for Efficient VLMs

**arXiv ID:** 2601.22709 | [PDF](https://arxiv.org/pdf/2601.22709v1)

**作者:** Yanlong Chen `[一作]` (ETH Zurich), Yawei Li `[通讯]` (ETH Zurich)

**通讯引用:** 4724 | [OpenAlex ID](https://openalex.org/A5100377386)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种名为GRACE的框架，将知识蒸馏与量化感知训练（QAT）联合起来，实现视觉‑语言模型（VLM）的高效压缩；

**💡 创新点**

创新点包括：①基于信息瓶颈的理论框架；②置信度门控的分离式蒸馏（过滤噪声并突出关键信息）；③面向视觉token的关系中心核对齐（Relational Centered Kernel Alignment，RCKA）；④自适应信息瓶颈控制器动态平衡蒸馏强度与模型容量；以及①量化采用组级可学习步长和STE。

**🔧 技术方法**

使用信息瓶颈、KL蒸馏、置信门控、CKA对齐、组级可学习步长量化、直通估计器（STE）以及自适应Lagrange乘子；

**📊 数据集**

训练数据主要为ShareGPT4V（约1.3M图文对）；评测使用LLaVA‑1.5、Qwen2‑VL模型；基准包含SQA、MMBench、SEED‑Bench、ScienceQA、VQA等多模态任务；

**📈 对比分析**

与BF16基线、PTQ、AWQ、RTN、QAT+Naive KD以及其他蒸馏方法（MoVE‑KD、HAWAII）比较，INT4 GRACE在LLaVA‑1.5和Qwen2‑VL上分别比BF16高约1–3%，比AWQ/RTN高2–3%；在4‑bit下仍能超过BF16基线，且实现3×吞吐量提升、54%显存压缩。

**⚠️ 局限性**

局限性：仅对权重量化；视觉编码器保持冻结，未对激活量化；仅在LLaVA和Qwen2‑VL上验证，未覆盖更大模型或其他多模态架构；需要大型教师模型，压缩过程仍相对昂贵。

---

## 328. Deep Learning-Based Early-Stage IR-Drop Estimation via CNN Surrogate Modeling

**arXiv ID:** 2601.22707 | [PDF](https://arxiv.org/pdf/2601.22707v1)

**作者:** Ritesh Bhadana `[一作]` `[通讯]` (Gurugram University), Ritesh Bhadana (Gurugram University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出基于U‑Net卷积神经网络的深度学习模型，对VLSI布局的功率网格、单元密度和切换活跃度三张二维特征图进行像素级回归，生成IR‑drop热图。

**💡 创新点**

创新点在于将IR‑drop估计视为像素级回归任务，使用U‑Net架构实现全局与局部特征融合，并利用物理启发的合成数据生成替代昂贵的物理求解器。

**🔧 技术方法**

主要技术包括U‑Net卷积编码‑解码网络、MSE损失、Adam优化器、像素级归一化及合成标签生成公式。

**📊 数据集**

使用自制的物理启发式合成数据集，包含64×64的功率网格、单元密度、切换活跃度和对应IR‑drop标签。

**📈 对比分析**

与传统基于物理的sign‑off工具相比，模型在验证集上达到MSE≈4.9e‑4、PSNR≈33.3 dB，单样本推理时间<10 ms，显著加速且误差可接受。

**⚠️ 局限性**

主要局限在于依赖合成标签，未覆盖温度、封装寄生等真实效应，泛化到工业真实设计仍需验证，阈值设定缺乏自适应性。

---

## 329. Multi-target DoA estimation with a single Rydberg atomic receiver by spectral analysis of spatially-resolved fluorescence

**arXiv ID:** 2601.22704 | [PDF](https://arxiv.org/pdf/2601.22704v1)

**作者:** Liangcheng Han `[一作]` (Huazhong University of Science and Technology), Mérouane Debbah `[通讯]` (Khalifa University)

**通讯引用:** 65607 | [OpenAlex ID](https://openalex.org/A5056145687)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出一种利用单一Rydberg原子接收机、通过空间分辨荧光成像将多目标方向估计问题转化为谱估计问题的方法，并通过Prony方法实现高精度多目标DoA估计。

**💡 创新点**

创新点在于：①在强本振主导下对原子吸收系数进行线性化，得到空间正弦叠加模型；②通过虚拟阵列将荧光分辨率转换为等距采样，实现连续开口感知；③剔除细胞长度依赖，恢复宽带能力并支持多通道/全波束MIMO。

**🔧 技术方法**

采用技术包括Rydberg原子电磁感知、荧光成像、空间窗函数采样、虚拟阵列处理、Prony谱估计以及Cramér–Rao下界分析。

**📊 数据集**

使用基于实际四能级Rb原子参数的数值仿真数据（无真实实验数据），通过多种目标角度、SNR、LO/信号比等情形验证方法。

**📈 对比分析**

与传统集成功率单目标方法以及理论CRLB进行对比，结果表明在LO信号强度至少为目标总功率10倍时，RMSE可近似达到CRLB，且多目标分辨率随单元数和细胞长度提升而改善。

**⚠️ 局限性**

局限包括：①必须在强本振主导的高功率场下工作，弱本振时线性化失效；②对荧光成像的空间分辨率和相机像素有严格要求；③Prony方法对噪声敏感，实际硬件噪声和散射光需进一步抑制。

---

## 330. PEAR: Pixel-aligned Expressive humAn mesh Recovery

**arXiv ID:** 2601.22693 | [PDF](https://arxiv.org/pdf/2601.22693v1)

**作者:** Jiahao Wu `[一作]` (International Digital Economy Academy), Yu Li `[通讯]` (International Digital Economy Academy)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `4de8e9d8-757b-475f-9627-18a445e50202` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 PEAR 框架，单张图像即可回归 SMPLX 与 FLAME 参数，实时恢复全身、手部和面部网格。

**💡 创新点**

创新点包括：①引入可学习的 EHM-s 模型并添加 head‑scale 参数以解耦头体比例；②使用单一轻量化 ViT‑B 作为骨干；③通过两阶段训练结合可微渲染实现像素级对齐；④采用部件级伪标签生成，支持任意裁剪输入。

**🔧 技术方法**

技术手段包括：Vision Transformer 轻量化骨干、SMPLX/FLAME 参数回归、可微 3DGS 神经渲染器、像素级光度与 LPIPS 损失、关键点监督以及两阶段粗细网格训练。

**📊 数据集**

使用的训练数据集包括 Human3.6M、MPI‑INF‑3DHP、COCO、MPII、InstaVariety、AVA、Ego‑Exo4D、Ego‑humans、SA1B、Harmony4D、AI Challenger，并通过 ProHMR、HAMER、TEASER、DWPose 等方法生成部件级伪标签。

**📈 对比分析**

与现有 SMPLX、SMPL、FLAME 及其它 Type‑3 方法在 3DPW、AGORA、COCO、LSP‑Extended 等基准上对比，MPJPE、PCK、MVE、LVE、PA‑PVE 等指标均优于或相当；推理速度约 100 FPS，显著提升像素级精度与细节捕捉。

**⚠️ 局限性**

局限性：对极端遮挡或复杂交互仍表现不足；Type‑3 模型在部分关节精度略低于专门的 Body‑only 方法，且需要较大的训练数据集与两阶段训练成本。

---

## 331. Constraint Satisfaction Problems over Finitely Bounded Homogeneous Structures: a Dichotomy between FO and L-hard

**arXiv ID:** 2601.22691 | [PDF](https://arxiv.org/pdf/2601.22691v1)

**作者:** Leonid Dorochko `[一作]` (Jagiellonian University), Michał Wrona `[通讯]` (Jagiellonian University)

**通讯引用:** 72 | [OpenAlex ID](https://openalex.org/A5078579181)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

证明了所有有限签名、有限边界的同构结构（模型完整核）的第一阶扩张的CSP在非统一AC^0与L难之间存在二分法，即若结构不满足特定平衡蕴含则CSP可在AC^0内解答，否则CSP在L中是难的。

**💡 创新点**

核心创新在于首先给出Larose‑Tesson定理的新证明，并将该证明机制推广到无限域；引入k‑树公式、(k,ℓ)‑最小化等工具，以实现对无限结构的统一二分判定。

**🔧 技术方法**

利用模型理论（ω‑齐次、有限边界、k‑同质性）、结构化的k‑树公式、(k,ℓ)‑最小化算法、极限理论与正则性保持定理，对CSP进行解析。

**📊 数据集**

无实验数据集，全部为理论证明。

**📈 对比分析**

本工作不涉及实验比较；通过证明方法展示了在理论上该二分法可适用于更广泛的结构，且在无限域上实现了与已知有限域结果等价的可判定性。

**⚠️ 局限性**

局限性在于仅针对模型完整核且不包含可一阶定义等价关系的结构；对更一般的第一阶约简或非模型完整核的结构，二分法尚未涵盖，且未给出具体的多元关系或路径宽度下的更细粒度复杂度分类。

---

## 332. How Far Can Pretrained LLMs Go in Symbolic Music? Controlled Comparisons of Supervised and Preference-based Adaptation

**arXiv ID:** 2601.22764 | [PDF](https://arxiv.org/pdf/2601.22764v1)

**作者:** Deepak Kumar `[一作]` (Johannes Kepler University Linz), Markus Schedl `[通讯]` (Johannes Kepler University Linz)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在ABC符号音乐领域对预训练LLM进行微调，比较了基础、SFT、DPO和音乐专用模型在生成与理解任务上的表现

**💡 创新点**

系统化地对比多种微调策略并引入基于音乐退化的弱优先级信号，为符号音乐LLM适配提供了可重复的评估框架

**🔧 技术方法**

使用LLaMA 3.1 Inst. 8B作为基础，采用监督微调（SFT）和基于对比的偏好优化（DPO），并结合Fréchet Music Distance与MMLU评估

**📊 数据集**

统一收集的ABC数据集（MusicPile、PDMX、ABCTunes、Open Lieder、Open String Quartets等）划分为短序列与长序列两组

**📈 对比分析**

在短序列数据上SFT获得最低PPL，ChatMusician在全局相似度FMD上表现相对优异；但在长序列与MMLU上均显著受损，显示域内收益与通用能力折衷

**⚠️ 局限性**

依赖弱优先级信号、仅使用ABC表示、数据覆盖不均、评估指标与人类判断不完全一致，以及算力与上下文长度限制导致结果不完全可迁移

---

## 333. AutoRefine: From Trajectories to Reusable Expertise for Continual LLM Agent Refinement

**arXiv ID:** 2601.22758 | [PDF](https://arxiv.org/pdf/2601.22758v1)

**作者:** Libin Qiu `[一作]` (Alibaba Group), Shuo Tang `[通讯]` (Alibaba Group)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 AutoRefine 框架，自动从 LLM 代理执行轨迹中提取并维护经验模式，包括技能模式和子代理模式，支持持续改进。

**💡 创新点**

创新点在于双模式自动提取（技能+子代理）以及连续维护机制（评分、裁剪、合并），解决流程逻辑无法被平面文本捕捉和经验库退化问题。

**🔧 技术方法**

采用对比分析的批量提取代理、语义相似检索、MMR 多样性过滤、子代理层级委派、嵌入向量检索、维护评分公式等技术。

**📊 数据集**

在 ALFWorld、ScienceWorld 与 TravelPlanner 三个基准上进行评估。

**📈 对比分析**

与 ReAct、Reflexion、AutoManual 等基线对比，AutoRefine 在 ALFWorld 成功率 98.4%、ScienceWorld 70.4%、TravelPlanner 27.1%，并在步骤数上降低 20-73%，在 TravelPlanner 上自动提取超越手工设计的 ATLAS（27.1% vs 12.1%）。

**⚠️ 局限性**

局限包括对失败经验的学习不足、对案例特定约束的捕捉不如手工设计、维护阈值和参数需手动设定。

---

## 334. SQUAD: Scalable Quorum Adaptive Decisions via ensemble of early exit neural networks

**arXiv ID:** 2601.22711 | [PDF](https://arxiv.org/pdf/2601.22711v1)

**作者:** Matteo Gambella `[一作]` (Politecnico di Milano), Manuel Roveri `[通讯]` (Politecnico di Milano)

**通讯引用:** 3958 | [OpenAlex ID](https://openalex.org/A5035547226)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种名为SQUAD的框架，将早期退出神经网络与分布式集成学习相结合，实现动态推理时的准确定量退出；

**💡 创新点**

创新点包括：1) 引入基于仲裁（quorum）的停止判定，利用多模型投票而非单模型阈值；2) 设计了QUEST NAS方法，联合优化准确率与层级多样性；3) 在推理过程中按计算复杂度顺序逐步激活学习器，减少不必要的计算；

**🔧 技术方法**

技术手段包括：早期退出神经网络（EENNs），集成学习（soft voting），t检验统计显著性检验，DARTS神经架构搜索，SVGD‑RD Bayesian采样，MACs计数与能耗评估；

**📊 数据集**

实验数据集：CIFAR‑10、CIFAR‑100、ImageNet16‑120；

**📈 对比分析**

与CNAS、NACHOS（单模型动态）、NESBS（静态集成）以及基准单模型比较；SQUAD在保持相近或更低MACs（F_M降低70.6%~42.2%）的同时，准确率比单模型动态方案高1.6%–5.9%，与静态集成相比也取得显著的能耗优势；

**⚠️ 局限性**

局限性：1) 需要先行训练和搜索耗时；2) 设计的仲裁阈值与t检验参数对性能影响较大，需要经验调参；3) 在极大模型或多任务场景下的扩展性尚未验证；4) 只针对分类任务，其他任务的适用性待研究。

---

## 335. AlienLM: Alienization of Language for API-Boundary Privacy in Black-Box LLMs

**arXiv ID:** 2601.22710 | [PDF](https://arxiv.org/pdf/2601.22710v1)

**作者:** Jaehee Kim `[一作]` (Seoul National University), Pilsung Kang `[通讯]` (Seoul National University)

**通讯引用:** 4396 | [OpenAlex ID](https://openalex.org/A5059650940)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一层基于词表置换的翻译机制AlienLM，用以在黑盒LLM API边界下保护文本不被泄露。

**💡 创新点**

创新在于利用词表级双射置换和仅API的微调（AAT），实现无内部访问的文本隐私，同时保持高效可逆。

**🔧 技术方法**

技术包括词表双射优化（基于代理嵌入）、客户端翻译器、AAT微调、以及针对攻击的隐私评估。

**📊 数据集**

使用七个基准（MMLU、ARC-Easy/Challenge、HellaSwag、WinoGrande、TruthfulQA、GSM8K）以及Magpie指令与推理数据。

**📈 对比分析**

与随机置换、字符级ROT13、SentinelLM等基线比较，平均恢复率超过81%，比随机置换高40+点，且在攻击场景下恢复率低于0.22%。

**⚠️ 局限性**

缺乏正式安全证明、可能削弱安全防护、对元数据（如长度）仍可见，且多租户单模型方案尚未完善。

---

## 336. DAVIS: OOD Detection via Dominant Activations and Variance for Increased Separation

**arXiv ID:** 2601.22703 | [PDF](https://arxiv.org/pdf/2601.22703v1)

**作者:** Abid Hassan `[一作]` (University of Southern California), Nenad Medvidovic `[通讯]` (University of Southern California)

**通讯引用:** 12418 | [OpenAlex ID](https://openalex.org/A5006714421)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种后置插件，通过加入通道最大激活和方差统计来增强特征向量，从而提升OOD检测性能。

**💡 创新点**

创新点在于将原本被GAP丢失的最大值和方差信息用于后处理，显著提升ID/OOD区分。

**🔧 技术方法**

技术实现为在预训练模型的特征图上计算每通道均值、最大值和标准差，然后将其与权重相乘得到新的logits，并与能量分数等传统评分函数结合。

**📊 数据集**

使用CIFAR-10/100、ImageNet-1k及多种OOD数据集（Textures、SVHN、Places365、LSUN、iSUN等）进行评估。

**📈 对比分析**

与现有方法（ReAct、DICE、ASH、SCALE等）以及能量/ MSP/ODIN等评分器比较，FPR95显著下降至CIFAR-10 48%~27%不等，ImageNet提升超过20%，显示出优越性能。

**⚠️ 局限性**

局限性在于仅针对使用GAP的CNN架构，无法直接应用于无全局聚合的ViT等模型；对超参数γ敏感，且目前仅验证logit基评分。

---

## 337. Metric Hub: A metric library and practical selection workflow for use-case-driven data quality assessment in medical AI

**arXiv ID:** 2601.22702 | [PDF](https://arxiv.org/pdf/2601.22702v1)

**作者:** Katinka Becker `[一作]` (Physikalisch Technische Bundesanstalt), Daniel Schwabe `[通讯]` (Physikalisch Technische Bundesanstalt)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e15e3743-5ee0-4d5f-813d-d146868082fc` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

构建了面向医疗AI的数据质量指标库 Metric Hub，并提供指标卡与决策树，形成一个可操作的多维度数据质量评估工作流。

**💡 创新点**

首次将理论的 METRIC 框架量化为 60 个可操作的指标，并通过决策树为不同用例提供自动化指标选择方案，弥补了医学机器学习数据质量评估缺乏统一、可重复方法的空缺。

**🔧 技术方法**

通过专家焦点小组共识收集指标，开发标准化指标卡；使用统计分布度量与相关系数实现指标计算；构建决策树算法辅助指标选取；在 Metric Hub 网站上实现可视化与交互；利用 Python 工具 MetricLib 对指标进行批量计算。

**📊 数据集**

在 PTB‑XL 12‑lead ECG 公共数据集（21,837 条记录）上进行演示，生成原始数据集及三种人工扰动子集（性别不平衡、设备均衡、目标类别不平衡），验证指标的敏感性与可解释性。

**📈 对比分析**

通过与人工评估的对比和对三种扰动子集指标变化的观察，展示指标能够捕捉数据不平衡、设备差异、噪声等质量问题；未给出统一性能数值，但实验结果表明指标值随数据质量变化而明显改变，验证了方法的有效性。

**⚠️ 局限性**

缺乏统一阈值与聚合得分；指标库对单标注噪声、缺失标签等场景覆盖不足；决策树对不同任务的泛化性仍待进一步验证；阈值设定与可解释性在实际应用中的可操作性尚需改进。

---

## 338. Farewell to Item IDs: Unlocking the Scaling Potential of Large Ranking Models via Semantic Tokens

**arXiv ID:** 2601.22694 | [PDF](https://arxiv.org/pdf/2601.22694v1)

**作者:** Zhen Zhao `[一作]`, Xiaojia Chang `[通讯]`

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了基于语义 Token 的 TRM 框架，旨在提升大型排序模型的可扩展性，显著降低稀疏参数并提升预测性能。

**💡 创新点**

创新点在于将协同对齐、多模态与用户行为信息融合、混合 Token 化（粗粒度 generalization Token + 细粒度 BPE 记忆 Token），以及联合判别‑生成目标的训练策略，解决了传统语义 Token 在冷启动、记忆化和结构信息缺失方面的瓶颈。

**🔧 技术方法**

技术包括多模态大语言模型（MLLM）进行视频 Caption 预训练、残差量化（RQ‑Kmeans）+BPE 生成 hybrid Token、对比学习实现协同对齐、基于 RankMixer 与纯 Transformer 的两种架构，并在判别层使用 BCE 损失、生成层使用下一 token 预测（NTP）损失实现联合优化。

**📊 数据集**

实验数据来自真实业务的短视频搜索日志，包括视频帧、标题、音频、字幕、用户查询和交互记录。

**📈 对比分析**

与 ID‑based 基准（DCN、DHEN、WuKong、RankMixer）以及 token‑based 方案（Tiger、OneRec、SemID）在 CTR/Real‑Play AUC、QAUC 进行对比；TRM‑RankMixer 在稀疏参数减少 32% 的同时，CTR AUC 提升 0.65%、QAUC 0.54%，Real‑Play AUC 分别提升 0.85%/0.70%；在线 A/B 实验显示活跃天数提升 0.26%、查询更改率下降 0.75%。

**⚠️ 局限性**

限制包括对 Token 数量与计算/存储成本的折衷、对极长序列生成建模的挑战，以及在不同业务场景下的迁移鲁棒性待进一步验证。

---

## 339. Neural Clothing Tryer: Customized Virtual Try-On via Semantic Enhancement and Controlling Diffusion Model

**arXiv ID:** 2601.22838 | [PDF](https://arxiv.org/pdf/2601.22838v1)

**作者:** Zhijing Yang `[一作]` (Guangdong University of Technology), Liruo Zhong `[通讯]` (Genstoraige Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 Neural Clothing Tryer（NCT），实现了可定制的虚拟试穿（Cu-VTON）系统，支持服装、姿态和属性的自由编辑。

**💡 创新点**

创新点在于结合语义增强模块将服装语义嵌入扩散模型，以及双分支语义控制模块实现服装细节与姿态、表情等属性的同时控制。

**🔧 技术方法**

采用扩散模型（latent diffusion）、视觉语言编码器（BLIP2/CLIP）、ControlNet 控制网络以及 LoRA 等技术。

**📊 数据集**

使用公开的 Dress Code 数据集，并通过交叉配对合成数据消除一对一配对偏差。

**📈 对比分析**

与 LaDI‑VTON、DreamBooth、DreamBooth‑LoRA、Textual Inversion 等基线在 Cu‑VTON 和传统 VTON 上进行对比，NCT 在 CLIP‑I、CLIP‑T、CLIP‑S、PD 指标上均明显领先，用户研究也显示其服装自然度和定制性最高。

**⚠️ 局限性**

局限包括：手部细节质量低、对极复杂或离散分布服装的再现能力不足、推理时间较长（512×512 约 15 秒）以及对高频细节的捕捉仍有限。

---

## 340. NativeTok: Native Visual Tokenization for Improved Image Generation

**arXiv ID:** 2601.22837 | [PDF](https://arxiv.org/pdf/2601.22837v1)

**作者:** Bin Wu `[一作]` (University of Science and Technology of China), Zhendong Mao `[通讯]` (University of Science and Technology of China)

**通讯引用:** 6125 | [OpenAlex ID](https://openalex.org/A5023341829)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种新的视觉分词框架 NativeTok，实现了原生视觉分词，使得生成阶段的 token 序列顺序与视觉信息一致，从而提高图像生成质量。

**💡 创新点**

创新点在于引入了自适应视觉顺序的分词、Mixture of Causal Expert Transformer (MoCET) 位置专属专家块以及分层原生训练策略，解决了传统分词与生成阶段的顺序不匹配问题。

**🔧 技术方法**

使用了 Meta Image Transformer 进行全局上下文建模，MoCET 进行位置专属因果建模，Transformer‑based 解码器，以及层级原生训练（Hierarchical Native Training）。

**📊 数据集**

在 ImageNet‑1K 256×256 图像上进行训练与评估。

**📈 对比分析**

与 VQGAN、TiTok 等主流 tokenizer 以及 LlamaGen、MaskGIT 生成器比较，NativeTok 在 AR 生成下 gFID 从 7.45 提升到 5.23，MaskGIT 下 gFID 达到 2.16，显著优于同参数量对手。

**⚠️ 局限性**

仍存在编码速度相对较慢、模型参数较大以及对不同生成框架适配需要进一步验证等局限。

---

## 341. A Comparative Evaluation of Large Vision-Language Models for 2D Object Detection under SOTIF Conditions

**arXiv ID:** 2601.22830 | [PDF](https://arxiv.org/pdf/2601.22830v1)

**作者:** Ji Zhou `[一作]` (Graz University of Technology), Arno Eichberger `[通讯]` (Graz University of Technology)

**通讯引用:** 1678 | [OpenAlex ID](https://openalex.org/A5084393303)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文对10种大型视觉语言模型在SOTIF条件下的2D目标检测性能进行了系统评估。

**💡 创新点**

创新点在于将视觉语言模型通过视觉定位实现无细化训练的检测任务，并构建统一评测管线；同时提出将LVLM作为高层安全验证器的思路。

**🔧 技术方法**

采用视觉预处理（缩放、边框、坐标尺）、链式思维提示、JSON输出解析以及基于COCO的mAP/mAR指标进行评测。

**📊 数据集**

使用PeSOTIF数据集，包含自然与人为环境退化以及长尾车辆与异常物体的1126帧。

**📈 对比分析**

与YOLOv5_60e基线对比，Gemini 3和Doubao在自然/恶劣天气下mAP提升≈0.25，YOLO在合成扰动下几乎保持优势，但LVLM在召回率上明显更优。

**⚠️ 局限性**

主要限制在于推理延迟高、定位精度不足以及需进一步压缩模型以满足实时性需求。

---

## 342. Offline Reinforcement Learning of High-Quality Behaviors Under Robust Style Alignment

**arXiv ID:** 2601.22823 | [PDF](https://arxiv.org/pdf/2601.22823v1)

**作者:** Mathieu Petitbois `[一作]` (Ubisoft), Sylvain Lamprier `[通讯]` (University of Angers)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在离线强化学习框架下提出 SCIQL 算法，用子轨迹标签实现可解释的风格对齐并优化任务表现；

**💡 创新点**

提出统一的行为风格定义并利用子轨迹标签减少信用分配难题，同时结合 Gated Advantage Weighted Regression 解决风格与任务目标冲突；

**🔧 技术方法**

基于 IQL、HER、GCRL、AWR 的离线 RL 价值学习与策略优化，并在 SCIQL 中加入风格重标记和门控优势回归；

**📊 数据集**

使用 Circle2d、HalfCheetah 与 HumEnv 等自定义环境，构建多标签子轨迹数据集；

**📈 对比分析**

与 BC、CBC、SCBC、BCPMI、SORL 等基线比较，SCIQL 在风格对齐与风格条件任务性能两项指标上均显著优于对照组（平均提升超 30% 以上）；

**⚠️ 局限性**

仍受限于离线数据覆盖不足、风格标签生成的准确性以及多风格多目标的可扩展性等问题。

---

## 343. Understanding on the Edge: LLM-generated Boundary Test Explanations

**arXiv ID:** 2601.22791 | [PDF](https://arxiv.org/pdf/2601.22791v1)

**作者:** Sabinakhon Akbarova `[一作]` (Chalmers University of Technology and University of Gothenburg), Robert Feldt `[通讯]` (Chalmers University of Technology and University of Gothenburg)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对使用大型语言模型（LLM）生成的软件边界解释进行探索性研究，并通过问卷和访谈评估其可用性和质量。

**💡 创新点**

首次系统评估LLM在边界值分析（BVT）中的解释能力；提出基于研究结果的七项设计需求清单，为未来的LLM边界解释工具奠定规范。

**🔧 技术方法**

使用 GPT‑4.1 生成解释；采用混合方法（在线问卷 + 半结构化访谈）评估四个维度（清晰度、准确性、完整性、实用性），并通过主题分析提炼需求。

**📊 数据集**

20 对边界输入/输出对（来自 4 个简易函数：Email、Bytecount、Date、BMI），27 名软件专业人员参与问卷，6 名参与访谈；未使用大型公开数据集，而是人工挑选的边界案例。

**📈 对比分析**

对比方法：仅在问卷中记录评分并做统计（正面占 63.5%，中立 19.5%，负面 17%）。访谈中未与其他工具做直接性能对比，但从受访者反馈判断 LLM 生成的解释在可读性和实用性上具备一定优势。

**⚠️ 局限性**

局限性：样本量小、便利抽样、仅使用 GPT‑4.1、仅针对文本解释、潜在的 hallucination 与误导、未考虑视觉/交互式解释，且结果对不同 LLM 版本或更复杂的程序可能不适用。

---

## 344. Toward IIT-Inspired Consciousness in LLMs: A Reward-Based Learning Framework

**arXiv ID:** 2601.22786 | [PDF](https://arxiv.org/pdf/2601.22786v1)

**作者:** Hamid Reza Akbari `[一作]` (Sharif University of Technology), Hossein Sameti `[通讯]` (Sharif University of Technology)

**通讯引用:** 1311 | [OpenAlex ID](https://openalex.org/A5024773257)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出基于IIT（集成信息理论）的奖励函数，对大语言模型进行强化学习后训练，使其生成更简洁、信息量更高的文本。

**💡 创新点**

创新点在于将IIT信息后设（Intrinsic Information）作为奖励，既保留信息整合特性，又显著降低输出长度；同时不依赖外部数据或辅助模型，奖励设计简单高效。

**🔧 技术方法**

技术包括：自回归LLM策略、强化学习（GRPO）与参数高效微调（LoRA/PEFT）、TPM构造与信息后设计算、文本生成长度与熵评估。

**📊 数据集**

使用 OpenThoughts‑114k‑Math 进行后训练，评估数据集涵盖 MATH‑500（in‑domain）、Countdown 与 GPQA‑Diamond（out‑of‑domain）等数学推理基准。

**📈 对比分析**

与三种基线（原始模型、仅准确率奖励、熵最小化奖励）比较，模型在保持相近或略低准确率的前提下，整体输出长度缩短约 25–31%，在 GPQA‑Diamond 上最小化 31% 的长度，并在自我一致性与校准上表现与基线相当或略优。

**⚠️ 局限性**

局限性包括：长度压缩与准确率存在权衡，奖励函数对超参数敏感；IIT 的完整 Φ 计算仍不可行，无法完全捕获意识特性；在更复杂任务（高难度数学推理、代码生成）上的泛化与效果尚待验证。

---

## 345. Color Matters: Demosaicing-Guided Color Correlation Training for Generalizable AI-Generated Image Detection

**arXiv ID:** 2601.22778 | [PDF](https://arxiv.org/pdf/2601.22778v1)

**作者:** Nan Zhong `[一作]` (City University of Hong Kong), Mian Zou `[通讯]` (Jiangxi University of Finance and Economics)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 DCCT 框架，通过模拟相机的 CFA 取样与去马赛克过程，学习颜色通道间的相关性，从而检测 AI 生成图像。

**💡 创新点**

创新点在于：① 将相机内部物理过程（CFA 取样与去马赛克）作为自监督任务，显式建模缺失通道的条件分布；② 通过高频域建模并理论证明在摄影图像与 AI 生成图像之间存在可测的 1‑Wasserstein 距离差异；③ 采用混合 Logistic 输出的 U‑Net 进行条件概率建模。

**🔧 技术方法**

技术细节包括：自监督 U‑Net（混合 Logistic 输出），高通滤波处理，CFA 对齐掩码，理论的 1‑Wasserstein 距离分析，以及基于提取的颜色相关特征的轻量级二分类器（ResNet+Transformer）。

**📊 数据集**

使用了 GenImage（8 代生成器+ImageNet）和 DRCT‑2M（16 Stable Diffusion 变体+MSCOCO）作为主要评测集，并在 GigaGAN、DFGAN、FLUX、SD‑3.5‑Turbo 等新生成器上做前瞻性评估。

**📈 对比分析**

与基于伪影检测（CNNSpot、GramNet 等）和通用预训练（CLIP、LNP、UnivFD、Effort）方法比较，DCCT 在超过 20 个未见生成器上平均精度超过 95%，比最强对手提升约 8%，在 DRCT‑2M 上几乎达到 100%（仅在 DR 变体稍有下滑），且对 JPEG 压缩和下采样等常见后处理保持较高鲁棒性。

**⚠️ 局限性**

局限性包括：对混合内容（DR）图像识别效果不佳；当生成模型刻意模仿 CFA 统计时可能失效；目前仅针对 Bayer CFA，需扩展到非 Bayer 或多光谱传感器与计算摄影流水线。

---

## 346. Inference-Time Dynamic Modality Selection for Incomplete Multimodal Classification

**arXiv ID:** 2601.22853 | [PDF](https://arxiv.org/pdf/2601.22853v1)

**作者:** Siyi Du `[一作]` (Imperial College London), Chen Qin `[通讯]` (Imperial College London)

**通讯引用:** 8625 | [OpenAlex ID](https://openalex.org/A5100362873)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于推理时动态模态选择框架（DyMo），通过自适应融合可靠的恢复模态来解决不完整多模态学习中的丢弃‑重建两难；

**💡 创新点**

创新点在于基于任务损失推导的增量任务相关信息奖励函数，并加入类内相似度校正，实现对低质量或语义失配恢复模态的自动剔除；

**🔧 技术方法**

采用多模态 Transformer 结构、动态模态选择算法、奖励函数与校正、以及模拟缺失与缺失无关对比损失的联合训练策略；

**📊 数据集**

在PolyMNIST、MST、CelebA、自然图像数据集 DVM 与医学图像数据集 UKBB 上进行实验；

**📈 对比分析**

与多种静态/动态融合方法及 9 种不完整 MDL 基线相比，DyMo 在多数任务中显著提升准确率或 AUC，尤其在 80% 模态缺失或全表缺失情形下提升超过 10%；

**⚠️ 局限性**

局限性包括对恢复质量高度依赖，校正项可能过于保守导致在某些数据集上无法进一步受益，以及对极端缺失情形下恢复方法本身的性能限制。

---

## 347. When Meanings Meet: Investigating the Emergence and Quality of Shared Concept Spaces during Multilingual Language Model Training

**arXiv ID:** 2601.22851 | [PDF](https://arxiv.org/pdf/2601.22851v1)

**作者:** Felicia Körner `[一作]` (LMU Munich), Barbara Plank `[通讯]` (LMU Munich)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了大型语言模型在多语言预训练过程中语言无关概念空间的出现与演化，利用交叉语言概念补丁（activation patching）方法对 EuroLLM 26 个检查点进行因果解释。

**💡 创新点**

首次系统地在多语言预训练的中间检查点上进行交叉语言概念补丁实验，揭示共享概念空间在训练早期就已形成且稳定，并提供细粒度手工错误分析，区分翻译质量提升是源自语义映射还是词义/同形词误差。

**🔧 技术方法**

因果可解释性技术激活补丁（activation patching）与交叉语言概念补丁，结合多语言翻译提示、词级翻译任务及完整词序列准确率评估。

**📊 数据集**

基于 Multi‑SimLex 的 2,147 个名词概念及其 13 语言翻译构造的 256 个可比概念对，覆盖 11 种语言（包括低资源与未见语言），并使用 EuroLLM 1.7B 的 26 个预训练检查点。

**📈 对比分析**

通过与未补丁翻译的基线对比，计算词级翻译准确率并给出 95% 置信区间；结果显示在大多数目标语言上，补丁可以提升 5–15% 准确率，尤其在第二阶段多平行数据训练后低资源语言的提升更显著。

**⚠️ 局限性**

仅评估 1.7B 规模模型；仅使用名词；仅在词级翻译任务上验证，未覆盖句子级或开放式生成；受 Multi‑SimLex 翻译偏差影响；检查点数量虽多但仍有限，模型大小与训练策略对结果的普适性需进一步验证。

---

## 348. Degradation-Aware Frequency Regulation of a Heterogeneous Battery Fleet via Reinforcement Learning

**arXiv ID:** 2601.22865 | [PDF](https://arxiv.org/pdf/2601.22865v1)

**作者:** Tanay Raghunandan Srinivasa `[一作]` (Plaksha University), Prashant Shenoy `[通讯]` (University of Massachusetts Amherst)

**通讯引用:** 18447 | [OpenAlex ID](https://openalex.org/A5032939724)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了在频率调节任务中，如何通过强化学习实现异构电池组的实时调度，以最小化循环衰减。

**💡 创新点**

创新点在于将循环衰减的非马尔可夫、延迟成本用密集代理奖励近似，构建可实时反馈的MDP框架，并利用极限学习机（ELM）实现大规模状态动作空间的价值函数逼近。

**🔧 技术方法**

采用强化学习中的MDP建模、奖励塑形、极限学习机特征提取与线性时序差分学习技术。

**📊 数据集**

使用PJM RegD频率调节信号（10 秒采样）以及合成的马尔可夫调节信号进行实验。

**📈 对比分析**

与无衰减惰性基准（比例分配、贪婪）及表格Q学习相比，RL‑ELM在累积奖励和雨流计数的循环损耗上均显著优于基线，衰减降低约10–20 %（取决于电池配置）。

**⚠️ 局限性**

局限性包括未将调节误差与衰减成本联合惩罚，仅考虑循环衰减，且未考虑温度、日历衰减等其他老化机制。

---

## 349. Trackly: A Unified SaaS Platform for User Behavior Analytics and Real Time Rule Based Anomaly Detection

**arXiv ID:** 2601.22800 | [PDF](https://arxiv.org/pdf/2601.22800v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 350. Design of a GPU with Heterogeneous Cores for Graphics

**arXiv ID:** 2601.22862 | [PDF](https://arxiv.org/pdf/2601.22862v1)

**作者:** Aurora Tomás `[一作]` (Universitat Politècnica de Catalunya), Antonio González `[通讯]` (Universitat Politècnica de Catalunya)

**通讯引用:** 9310 | [OpenAlex ID](https://openalex.org/A5100733331)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种针对图形渲染的异构GPU架构，结合计算专用和内存专用两个核心类型。

**💡 创新点**

创新点在于为GPU设计两类专用核心并引入基于Tile的亲和性与局部性感知调度器，实现动态映射。

**🔧 技术方法**

使用Tile-Based Rendering、基于MPKI的内存强度预测、Z-order与S-order遍历、Flood Fill等技术，并在TEAPOT仿真框架中评估。

**📊 数据集**

使用了32款来自Google Play的商业Android游戏，包括2D、2.5D、3D等多样化Benchmark。

**📈 对比分析**

与传统同构GPU对比，平均速度提升约20%~30%，帧率提高，能耗降低约15%，通过基准测评与DRAM访问量、L1/L2失效率对比验证。

**⚠️ 局限性**

局限性包括对极端内存或计算极端分布的应用效果不佳，调度算法仍需在高分辨率场景下验证，且实现仅在模拟环境下，无硬件实现成本评估。

---

## 351. Under-Canopy Terrain Reconstruction in Dense Forests Using RGB Imaging and Neural 3D Reconstruction

**arXiv ID:** 2601.22861 | [PDF](https://arxiv.org/pdf/2601.22861v1)

**作者:** Refael Sheffer `[一作]` (Rafael Advanced Defense Systems inc), Roee Litman `[通讯]` (Rafael Advanced Defense Systems inc)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `3855fcda-48ef-4070-a15e-803cd5c84d83` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `51c0528b-f690-4182-ae60-bb5f046c276c` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

利用常规 RGB 航拍图像，构建基于 NeRF 的三维重建管线，生成无冠层遮挡的地面真实感视图；

**💡 创新点**

首次将 NeRF 与低照度损失函数、语义遮罩及高度裁剪相结合，实现在森林冠层下的景观重建，并将该方法用于搜索救援人像检测与树干计数；

**🔧 技术方法**

NeRF（Instant‑NGP）+低光 RAW 损失 + 语义分割（SAM 或基于 HSV 的分割）+ SfM（COLMAP）+ 3D 点云提取 + HDBSCAN 聚类 + 自编码异常检测；

**📊 数据集**

AOS 公开数据集（RGB 与热图）以及合成 CAD 3D 森林场景；

**📈 对比分析**

与 AOS 热图、MVS、3D Gaussian Splatting 等方法对比，本文重建结果更完整、细节更清晰；在搜索救援人像检测中取得 80.5% AP（全场景）或 89.6% AP（去除低质量场景），仅用 RGB 图像；树干计数与 3DGS 取得相近的 M‑SSIM；

**⚠️ 局限性**

对光照极端或低照度条件下的表现不佳；高度依赖 SfM 精度与图像质量；在冠层密集或光照不均匀的区域可能失真；目前未与激光雷达或多光谱数据融合。

---

## 352. How Much of a Model Do We Need? Redundancy and Slimmability in Remote Sensing Foundation Models

**arXiv ID:** 2601.22841 | [PDF](https://arxiv.org/pdf/2601.22841v1)

**作者:** Leonard Hackel `[一作]` (Big Data Analytics for Earth Observation), Begüm Demir `[通讯]` (Big Data Analytics for Earth Observation)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

对六个遥感基础模型在四个下游分类任务中进行统一宽度裁剪（post‑hoc）和学习式可裁剪训练，系统评估其可裁剪性与信息冗余。

**💡 创新点**

首次揭示遥感模型在比计算机视觉模型更小的规模下进入过参数化区间，裁剪损失极小；提出学习式裁剪可进一步提升鲁棒性，并用方差解释与特征相关性提供机制解释。

**🔧 技术方法**

采用 Transformer 可裁剪层、统一宽度裁剪、学习式可裁剪训练（US3L、SlimCLR 等）、KNN/线性探针评估、相对保持率、解释方差比、特征相关性分析。

**📊 数据集**

基于 4–11 个遥感数据集预训练的模型，在 GeoBench 四个分类数据集：m-brick-kiln、m-eurosat、m-so2sat、m-bigearthnet 进行评测。

**📈 对比分析**

与 CV 领域 MAE 基线对比；遥感模型在 1 FLOP 预算下保持 71% 以上相对准确度，CV MAE 仅 10%；学习式裁剪在所有规模下均优于裁剪式；中间规模往往表现最佳，提升 12× 速度同时提升准确率。

**⚠️ 局限性**

仅针对 Transformer 结构；裁剪方式为统一宽度，未探索非均匀裁剪；MAE 基础模型在学习式裁剪时表现差，需任务调优；未验证更复杂压缩方法或跨模态压缩的可行性。

---

## 353. Sparse or Dense? A Mechanistic Estimation of Computation Density in Transformer-based LLMs

**arXiv ID:** 2601.22795 | [PDF](https://arxiv.org/pdf/2601.22795v1)

**作者:** Corentin Kervadec `[一作]` (Universitat Pompeu Fabra), Gemma Boleda `[通讯]` (Catalan Institute of Research and Advanced Studies)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了Transformer‑based LLM的计算密度，提出基于信息流路径的稀疏子图提取方法，并用TV距离衡量子图对完整模型输出的保真度，量化了不同输入对稠密度的影响；

**💡 创新点**

创新点在于将机制解释中的计算图与信息流重要性筛选相结合，形成可训练的trace估计器；使用TV距离定义保真度，避免KL对尾部敏感；量化计算密度（ρ）并揭示其随词频、位置变化的动态特性；提供跨模型的输入驱动稀疏性评估；

**🔧 技术方法**

使用Transformer计算图、信息流路线（IFR）重要性分数、递归后向搜索、TV距离、CatBoost + SHAP、梯度提升树、Nucleus采样、Zero‑ablation、Mean‑ablation 等技术；

**📊 数据集**

使用5000个平均20词的Wikitext片段进行评估，另外采集660个多域提示，生成30词续写并计算词频表；

**📈 对比分析**

通过绘制trace大小与TV误差的曲线，计算ρ估计值，并与完整模型对比；结果显示平均计算密度偏高，且与词频、位置呈显著相关；不同模型间的密度分布高度相关，稀疏剪枝会导致分布扁平化，影响生成质量；

**⚠️ 局限性**

局限性：依赖当前的trace提取方法，若改进可能得到更稀疏的结果；Zero‑ablation评估可能偏离数据流，未考虑均值替代等更细粒度抑制；仅在英语Transformer上验证，未覆盖其他语言或架构；

---

## 354. Conditional Performance Guarantee for Large Reasoning Models

**arXiv ID:** 2601.22790 | [PDF](https://arxiv.org/pdf/2601.22790v1)

**作者:** Jianguo Huang `[一作]` (Nanyang Technological University), Bo An `[通讯]` (Nanyang Technological University)

**通讯引用:** 6771 | [OpenAlex ID](https://openalex.org/A5017743551)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出基于组别的PAC推理框架 G-PAC 和 C-PAC，实现大模型推理的组别条件风险控制

**💡 创新点**

创新点是把 PAC 推理从边际保证扩展到组别条件保证，并给出可学习组划分的聚类 PAC（C-PAC），理论证明组划分可显著提升效率

**🔧 技术方法**

采用 PAC 推理、置信上界构造、重要抽样、CLT 估计、1D k-means 聚类和分层校准等技术

**📊 数据集**

在 MATH-500、ZebraLogic、GPQA 等推理基准以及 Arena‑Hard 开放域任务上进行实验验证

**📈 对比分析**

与传统 PAC 推理相比，G-PAC/C-PAC 在每个组的误差间隙为零，STP 略低但仍保持显著的计算节省，误差提升不足 10%

**⚠️ 局限性**

局限在于对不确定性分数的依赖、学习组划分时需要样本分裂或出现覆盖间隙，以及组信息缺失时效率下降

---

## 355. FACET: Multi-Agent AI Supporting Teachers in Scaling Differentiated Learning for Diverse Students

**arXiv ID:** 2601.22788 | [PDF](https://arxiv.org/pdf/2601.22788v1)

**作者:** Jana Gonnermann-Müller `[一作]` (Zuse Institute Berlin), Sebastian Pokutta `[通讯]` (Zuse Institute Berlin)

**通讯引用:** 1951 | [OpenAlex ID](https://openalex.org/A5043574831)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一套教师侧的多智能体框架 FACET，用于在课堂中生成针对不同学习者（包括动机、绩效、神经多样性）差异化、符合课程与无障碍标准的学习材料，并让教师在生成流程中保持完全控制。

**💡 创新点**

① 采用四个专门化智能体（学习者模拟、诊断推理、材料生成、评估）实现学习者画像的稳定与可重用；② 将动机、学习差异等认知与情感维度纳入差异化的设计；③ 在教师控制下提供完整的教师‑循环工作流，保持教师专业判断；④ 集成课程标准、教学原则与无障碍指南，自动生成可直接打印的 PDF/Word 文档。

**🔧 技术方法**

基于本地部署的 GPT‑OSS 120B 大语言模型；Python FastAPI + React 前端；智能体间通过定义好的 JSON 协议交互；学习者智能体使用 LLM 生成思维轨迹；诊断智能体将轨迹转换为结构化诊断 JSON；生成智能体依据诊断与教师输入生成材料；评估智能体以结构化反馈评价生成结果；后续计划加入 Retrieval‑Augmented Generation（RAG）以克服上下文窗口限制。

**📊 数据集**

主要使用教师提供的课堂任务与课程对齐资源；学习者画像由教师手工设定（无公开数据集）。实验中未使用公开评测数据集，而是通过德国校长研讨会与 70 名英国/美/加/澳教师的问卷对生成的 5 份试卷进行评价。

**📈 对比分析**

采用混合方法评估：① 对 30 名德国校长进行一小时工作坊，讨论实用性、可用性、课堂整合；② 对 70 名 K‑12 教师进行在线问卷，给出 5 份基于不同学习者画像（高/低绩效、动机、阅读障碍）的工作表，教师按 6 分制在结构、清晰度、格式/设计、创造性等四维度评分。结果显示：对低动机或低绩效学生的支架化材料得分 4–4.5；高绩效/高动机材料得分 3–3.5；阅读障碍材料在布局/语言简化方面评分 4+，但字体可读性与适应性仅 3.3，提示改进空间。

**⚠️ 局限性**

① 研究仅覆盖 7 年级数学，缺乏跨学科与跨年级推广；② 学习者智能体基于 LLM 模拟，真实性与实际学生表现尚未充分验证；③ 对高动机/高绩效学生的丰富化与阅读障碍的排版细节仍需优化；④ 评估主要基于教师主观打分，未包含学生反馈与学习成效；⑤ 系统目前不支持云端部署，局限于本地硬件。

---

## 356. Robust Rigid Body Assembly via Contact-Implicit Optimal Control with Exact Second-Order Derivatives

**arXiv ID:** 2601.22849 | [PDF](https://arxiv.org/pdf/2601.22849v1)

**作者:** Christian Dietz `[一作]` (Siemens AG), Armin Nurkanović `[通讯]` (University of Freiburg)

**通讯引用:** 202 | [OpenAlex ID](https://openalex.org/A5069370342)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出一种基于可微物理仿真的鲁棒闭环组装轨迹优化方法，能够在极低的采样量下生成可直接执行的组装运动。

**💡 创新点**

创新点包括：1）将增量距离（growth distance）改进为对欧氏距离更友好的平滑Signed Distance Function（SDF）并提供精确的一阶、二阶导数；2）在碰撞检测和接触解析中统一使用基于内部点法的平滑化，从而得到可微的完整接触动力学模型；3）设计多场景鲁棒最优控制框架与卡尔曼式偏移扰动结合的卡尔曼式阻尼；4）通过同态方法逐步收紧平滑参数，实现从松弛到接近非平滑原始动力学的过渡；5）证明使用精确Hessian比Gauss‑Newton/L‑BFGS等近似更快、更稳定。

**🔧 技术方法**

技术手段包括：内部点法（Interior‑Point）求解凸QP/LCP、隐式函数定理实现一次性得到一阶/二阶导数、Scholtes平滑与松弛、多场景（scenario‑based）最优控制、基于C++/CvxQuad、Eigen、CasADi的高性能实现、同态参数更新（κτ, κσ, κμ）以及多周期仿真与实验验证。

**📊 数据集**

使用随机生成的多面体拼装任务（如五种不同的 peg‑in‑hole 问题，接触对数从 6 到 24）和真实机器人（UR10e）与 3D 打印的 H‑形杆/夹具组装实验；实验中不涉及公开数据集，全部为自定义仿真和实机实验。

**📈 对比分析**

与基线相比：1）在仿真中，使用精确Hessian 时迭代次数约为 Gauss‑Newton 的 5 倍、L‑BFGS 的 10 倍；2）使用 Scholtes 的松弛比平滑收敛更快；3）在实际机器人实验中，成功率可达 99%（当模型误差被保守估计至少 1 mm 时），而传统 RL/采样方法需数十亿步仿真。

**⚠️ 局限性**

局限性包括：1）碰撞检测子问题规模随凸多面体半空间数量线性增长，导致高复杂几何或大量接触对时计算量大；2）当前模型仅为无摩擦接触，未考虑 Coulomb 摩擦；3）求解器采用通用线性求解器，未充分利用轨迹优化的稀疏结构；4）同态收敛到极端小平滑参数时，Gauss‑Newton/L‑BFGS 的近似可能导致不收敛。

---

## 357. μTouch: Enabling Accurate, Lightweight Self-Touch Sensing with Passive Magnets

**arXiv ID:** 2601.22864 | [PDF](https://arxiv.org/pdf/2601.22864v1)

**作者:** Siyuan Wang `[一作]` (Shanghai Jiao Tong University), Dongyao Chen `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 917 | [OpenAlex ID](https://openalex.org/A5085088480)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于被动磁铁的轻量化自触感知平台μTouch，能够识别面部细微接触和身体抓痒行为；

**💡 创新点**

创新点在于：①仅使用三枚低功耗磁传感器即可实现高精度识别；②引入自监督预训练+极少标注微调的半监督学习框架，减少用户采样成本；③设计环境磁场抑制算法，提升不同环境和衣物遮挡下的鲁棒性；

**🔧 技术方法**

使用低功耗霍尔效应磁传感器、BLE传输、磁场抑制滤波、基于TS2Vec的自监督时间序列编码器、SVM/随机森林下游分类器；

**📊 数据集**

构建了两组用户数据集：面部接触（11人，7类+无触摸）和身体抓痒（12人，二分类及九分类），并利用10人数据预训练编码器；

**📈 对比分析**

与SVM、随机森林、PCA+SVM等传统基线相比，μTouch在面部识别中达93.4%准确率、抓痒识别中达94.6%准确率，均超过基线5–10%；

**⚠️ 局限性**

局限性包括：对磁铁强度和位置高度依赖；需要在使用前做一次环境校准；样本量有限，未验证在更广泛自触任务（如手部抚摸、唇部抚触）上的迁移性。

---

## 358. Learning to Build Shapes by Extrusion

**arXiv ID:** 2601.22858 | [PDF](https://arxiv.org/pdf/2601.22858v1)

**作者:** Thor Vestergaard Christiansen `[一作]`, J. Andreas Bærentzen `[通讯]` (Technical University of Denmark)

**通讯引用:** 2564 | [OpenAlex ID](https://openalex.org/A5032257710)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

通过学习面挤压（extrusion）序列并使用大型语言模型生成3D网格，支持任意面数和网格编辑。

**💡 创新点**

将挤压操作编码为文本（TEE），并利用LLM自动学习和生成挤压序列，而非传统基于三角面或点云的直接网格生成。

**🔧 技术方法**

使用文本编码挤压（TEE）、Llama 3.2 1B大型语言模型、2D谐波映射、聚类和基于图的挤压序列生成技术。

**📊 数据集**

基于FEQ网格构建的数据集，包括来自MANO、DFAUST、HexaLab以及自建的四面体网格，用于训练和评估。

**📈 对比分析**

与MeshXL等Transformer基模型进行FID对比，取得13.23对比66.4的显著优势，且生成网格面数不受限制。

**⚠️ 局限性**

仅支持FEQ网格且拓扑为球面，不能处理自相交或自相邻面环、非球面拓扑及长分支挤压序列，需先将三角网格映射为FEQ网格。

---

## 359. Hierarchical Shift Mixing -- Beyond Dense Attention in Transformers

**arXiv ID:** 2601.22852 | [PDF](https://arxiv.org/pdf/2601.22852v1)

**作者:** Robert Forchheimer `[一作]` (Linköping University), Robert Forchheimer `[通讯]` (Linköping University)

**通讯引用:** 5079 | [OpenAlex ID](https://openalex.org/A5038095342)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Hierarchical Shift Mixing (HSM) 作为 Transformer 的线性时间 token 混合替代方案

**💡 创新点**

在保持因果性和并行性的同时，将密集注意力的 O(T^2) 复杂度降低到 O(T)，并探索多种线性与非线性混合函数

**🔧 技术方法**

使用自定义的线性、加权、门控、融合等 token 混合技术，并在 HSM 中实现多头与层级移位策略

**📊 数据集**

采用 TinyStories 1.9 GB 童话故事数据集进行训练与评估

**📈 对比分析**

通过交叉熵损失、验证准确率和训练时间等指标与 GPT‑2 小型基线对比，单层 HSM 在损失和速度上与 GPT 相当，Hybrid 模型甚至优于 GPT

**⚠️ 局限性**

局限在于实验规模较小，未验证在更大模型/数据集上的可扩展性，Hybrid 方案失去线性时间特性

---

## 360. Toward Pluralizing Reflection in HCI through Daoism

**arXiv ID:** 2601.22831 | [PDF](https://arxiv.org/pdf/2601.22831v1)

**作者:** Aaron Pengyu Zhu `[一作]` (National University of Singapore), Janghee Cho `[通讯]` (National University of Singapore)

**通讯引用:** 618 | [OpenAlex ID](https://openalex.org/A5103283515)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

**🎯 论文内容**

通过对18名道教宗师、学者与修行者的半结构化访谈，提出了以静止、共鸣、涌现为核心的三维道家反思模型。

**💡 创新点**

首次将道家哲学的身体化、关系化与过程性视角引入HCI反思研究，拓宽了反思框架的多元性与实践性。

**🔧 技术方法**

采用解释性主题分析，对访谈录音转写文本进行编码、归纳与对照，揭示道家反思维度。

**📊 数据集**

使用18名跨国、跨族裔的道教专家访谈数据，包含多样的修行与学术背景。

**📈 对比分析**

论文未进行定量性能比较，而是通过质性对比阐释新框架在理论与设计启示上的优势。

**⚠️ 局限性**

样本规模有限，且受访者多为东亚背景，结果可能在跨文化推广时需要进一步验证。

---

## 361. Diachronic Stereo Matching for Multi-Date Satellite Imagery

**arXiv ID:** 2601.22808 | [PDF](https://arxiv.org/pdf/2601.22808v1)

**作者:** Elías Masquil `[一作]` (Universidad de la República), Gabriele Facciolo `[通讯]` (Universite Paris-Saclay)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种能够从时间跨度较大、季节、光照和阴影变化显著的卫星图像对中恢复 3D 结构的单目+双目深度匹配方法。

**💡 创新点**

创新点在于：1) 将包含同步与非同步（时间差>30天）图像对的多样化数据集用于微调，显著提升在极端变化下的匹配鲁棒性；2) 通过结合单目深度先验（Depth Anything V2）来抵消光照/季节变化的影响；3) 采用基于 RPC 的自适应单像素配准与不偏差正极性对齐的 rectification 算法，使得极差、单极化且随高度增大的视差满足深度网络要求。

**🔧 技术方法**

使用了 MonSter 深度匹配网络（含单目先验），并在其预训练权重上微调；配合 DISK+LightGlue 关键点匹配、RPC 直射投影、光流式成本体裁剪与迭代细化；利用 LiDAR DSM 生成的真实视差作为监督。

**📊 数据集**

数据集：① DFC2019 Track3（Jacksonville、Omaha）——110 个 AOI，含同步与非同步图像对；② IARPA2016（Buenos Aires）——3 个 AOI；③ 公开的 Enschede 与 EuroSDR-Vaihingen 空中影像基准用于跨域泛化评估。

**📈 对比分析**

在同步、软非同步与硬非同步（雪 vs 非雪）测试集上，微调后的 MonSter（同步+非同步）在平均绝对误差（MAE）上均优于：经典 s2p‑hd、RAFT‑Stereo、FoundationStereo、StereoAnywhere 等零射击与仅同步微调模型；在同一数据集上实现了 20–30% 的误差下降，且在非同步场景中误差大幅降至 <1 m。跨域评估表明，在 Enschede 数据集上，微调模型的 MAE 与 RMSE 也比零射击模型低约 20%。

**⚠️ 局限性**

局限性包括：① 依赖 LiDAR 生成的 DSM 作为真值，导致时间不一致或结构缺失（如建造/拆除）产生误差；② 训练数据中植被高度分布偏差导致模型在树木出现与否上存在偏倚；③ 与合成数据相比，真实 DSM 解析度有限，造成预测边界相对平滑、细节缺失；④ 仍需进一步提升在极端光照/阴影变化和大视角差下的鲁棒性。

---

## 362. Compact Hypercube Embeddings for Fast Text-based Wildlife Observation Retrieval

**arXiv ID:** 2601.22783 | [PDF](https://arxiv.org/pdf/2601.22783v1)

**作者:** Ilyass Moummad `[一作]` (Inria), Alexis Joly `[通讯]` (Inria)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于超立方体哈希的文本‑观测对齐框架，能快速检索大规模野生动物图像和音频数据库。

**💡 创新点**

创新点在于将 Cross‑View Code Alignment 扩展到跨模态，构建共享的 Hamming 空间，并证明哈希目标可提升底层表示与零样本泛化。

**🔧 技术方法**

采用轻量级哈希头、LoRA 参数高效微调、BioCLIP/BioLingual 基础模型、最大编码率正则化与二进制交叉熵对齐等技术。

**📊 数据集**

使用 iNaturalist2024（文本‑图像）、iNatSounds2024（文本‑音频）以及多个声景数据集（HSN、NES、SNE、UHH、PER、SSW）进行评估。

**📈 对比分析**

与连续嵌入在 Cosine 相似度下对比，使用 mAP@1000 评估；256‑bit 哈希在图像检索上与或超过连续嵌入，在音频检索上显著优于原模型；在 OOD 声景数据上，哈希方案提升检索和零样本分类性能。

**⚠️ 局限性**

局限在于低比特数（128‑bit）会显著损失精度；跨模态对齐依赖已配对训练数据，未覆盖无配对场景；模型仍受原始基础模型偏差影响。

---

## 363. Learning with Challenges: Adaptive Difficulty-Aware Data Generation for Mobile GUI Agent Training

**arXiv ID:** 2601.22781 | [PDF](https://arxiv.org/pdf/2601.22781v1)

**作者:** Linjia Kang `[一作]` (Tsinghua University), Zhi Wang `[通讯]` (Tsinghua University)

**通讯引用:** 18796 | [OpenAlex ID](https://openalex.org/A5100376411)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一个基于代理能力前沿的自适应难度感知数据生成框架，用来训练移动GUI代理。

**💡 创新点**

创新点在于把轨迹难度拆解为结构性与语义性两个维度，构建以挑战点为驱动的难度分布，并通过多代理可控生成器产生符合该分布的高质量轨迹。

**🔧 技术方法**

采用了结构/语义难度量化、能力轮廓分析、挑战点α调节、Gaussian/三角分布、Explorer–Supervisor协作生成、逆向合成思路、LoRA微调等技术。

**📊 数据集**

主要使用了AndroidWorld、AndroidControl‑Curated和GUIOdyssey三个基准数据集，并结合先前的任务数据集进行实验。

**📈 对比分析**

与零样本通用VLM、专用GUI代理以及其他数据合成框架进行对比，利用SR、Type、Grounding等指标，在三大基准上平均提升约1.4–1.6倍成功率，显著优于SOTA。

**⚠️ 局限性**

局限性包括对挑战点α的调参需要经验、难度分布对不同能力维度的适配尚不完美、生成过程算力需求高、未充分研究跨设备视觉差异等问题。

---

## 364. Float8@2bits: Entropy Coding Enables Data-Free Model Compression

**arXiv ID:** 2601.22787 | [PDF](https://arxiv.org/pdf/2601.22787v1)

**作者:** Patrick Putzky `[一作]` (Merantix Momentum GmbH), Stefan Dietzel `[通讯]` (Merantix Momentum GmbH)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 EntQuant 框架，利用熵编码将量化精度与压缩率解耦，完成无数据、极限压缩（可达 2 甚至 1.5 位有效比特/参数）的后训练量化，并在推理时将权重实时解压。

**💡 创新点**

创新点在于：①首次将高精度量化（8/16 位）与 GPU‑加速的 ANS 熵编码结合，打破传统固定比特宽度与存储成本的耦合；②使用 ℓ₁ 正则化作为可微的熵近似，简化离散熵优化；③把熵解码嵌入推理前向传播，保持高精度算子且延迟仅 1.5–2×。

**🔧 技术方法**

核心技术包括：对称量化（Q₍γ₎），通道级缩放，ANS 熵编码/解码（nvCOMP/hipANS），直通估计器（Straight‑Through Estimator）进行梯度传播，基于 8 位浮点的量化基准。

**📊 数据集**

实验使用 16 款开源 LLM（如 LLaMA‑2、Qwen‑3、Mistral‑Large‑Instruct 等），评测数据集包括 C4、LLaMA‑Eval、GSM8K CoT、IFEval、GPQA、MMLU 等标准与难度更高的指令/推理任务。

**📈 对比分析**

与 HQQ、NF4 等纯数据‑free 方法比较：在 2–3 位有效压缩率下显著优于 HQQ，且在 2 位时不崩溃；与基于校准或微调的 4 位 PTQ（如 GPTQ、AWQ）相当，甚至可与 QuIP#、EfficientQAT 等微调方法竞争；在指令/推理基准上保持 95%+ 原始性能，推理速度仅比 NF4 慢 1.5–2 倍。

**⚠️ 局限性**

局限性包括：使用单一 ℓ₁ 熵代理，未探索更高级的分组或自定义融合核；实验主要聚焦于单一 GPU，未对 Mixture‑of‑Experts 大模型或多 GPU 场景做完整评估；熵解码仍依赖 GPU，CPU 侧可能存在显著性能瓶颈。

---

## 365. Hide and Seek in Embedding Space: Geometry-based Steganography and Detection in Large Language Models

**arXiv ID:** 2601.22818 | [PDF](https://arxiv.org/pdf/2601.22818v1)

**作者:** Charles Westphal `[一作]` (University College London), Fernando E. Rosas `[通讯]` (University of Sussex)

**通讯引用:** 5922 | [OpenAlex ID](https://openalex.org/A5020498855)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种低可恢复性的语言模型隐藏信息方法，利用模型嵌入空间的随机超平面投影实现秘密编码与词汇分桶；同时探讨基于内部激活的线性探测器来检测此类隐藏信息。

**💡 创新点**

创新点在于：①把传统基于奇偶或ASCII映射的可逆方案替换为基于随机超平面的不可逆编码与分桶，显著降低可恢复性；②在攻击与防御两侧同时给出统一的“payload可恢复性”评估指标；③证明即使保持困惑度与KL散度在微调范围内，内部线性探测仍能高精度识别隐藏信息。

**🔧 技术方法**

技术包括：随机超平面投影、Token embedding空间分桶、LoRA微调、线性探测器（Logistic/Linear回归）以及XOR掩码控制等。

**📊 数据集**

使用数据集：WikiText‑103、TrojanStego提示集、HuggingFace helpful‑instructions、Stanford Alpaca；模型包括 Llama‑8B、Ministral‑8B、Llama‑70B 的LoRA 微调版。

**📈 对比分析**

与传统 ASCII+Parity 基线相比，本文方案在 Llama‑8B、Ministral‑8B 与 Llama‑70B 上的精确匹配率分别提升了约 30%–123%，且可恢复性指标显著下降；检测时，线性探测器在最后层对比基线可达 33% 的准确率提升。

**⚠️ 局限性**

局限性包括：检测需要对内部激活的访问；高级微调策略可能进一步隐蔽隐藏信息；对黑盒或仅输出观察的环境尚无直接检测方法。

---

## 366. FarmMind: Reasoning-Query-Driven Dynamic Segmentation for Farmland Remote Sensing Images

**arXiv ID:** 2601.22809 | [PDF](https://arxiv.org/pdf/2601.22809v1)

**作者:** Haiyang Wu `[一作]` (Central South University), Tao Chao `[通讯]` (Central South University)

**通讯引用:** 2566 | [OpenAlex ID](https://openalex.org/A5069479267)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 FarmMind 动态分割框架，结合 reasoning‑query 机制，在遇到分割歧义时主动检索时空辅助影像进行跨图像协同推理，从而提升农田遥感影像分割精度。

**💡 创新点**

①突破传统静态分割范式，形成动态分割范式；②利用多模态大语言模型实现因果推理与查询决策；③主动检索高分辨率、扩展尺度或相邻时间的辅助图像；④构建可检索的大规模农田遥感数据库。

**🔧 技术方法**

多模态大语言模型（如 Qwen‑VL‑Max）作为 reasoning‑query 模型；基础分割模型 FSVLM/SAM2；条件掩模纠正（CMC）与二值重映射；层级索引检索与跨图像协同推理。

**📊 数据集**

FarmSeg‑VL 中文区遥感图像集作为训练/评测集；构建的112 场景多时相宽幅遥感图像数据库，涵盖中国、美国、德国、柬埔寨等，包含 11,131 个 512×512 语义标注样本。

**📈 对比分析**

与 U‑Net、DeepLabv3+、SegFormer 等标签驱动方法以及 PixelLM、LaSagnA、FSVLM 等语言驱动方法进行对比；FarmMind 在 mAcc 92.85%、mIoU 84.43%、F1 91.25%、Recall 93.04% 上均优于对手，在跨域测试中亦保持最佳泛化能力。

**⚠️ 局限性**

需要外部数据库检索，查询效率和实时性受限；在高精度基础分割模型上可能出现纠正误差导致假阳性；对大模型推理成本高，且对不同基础模型的纠正策略需动态调节。

---

## 367. Aligning the Unseen in Attributed Graphs: Interplay between Graph Geometry and Node Attributes Manifold

**arXiv ID:** 2601.22806 | [PDF](https://arxiv.org/pdf/2601.22806v1)

**作者:** Aldric Labarthe `[一作]` (Universite Paris Saclay), Julien Randon-Furling `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出一种将属性流形学习与图结构对齐分离的变分自编码器框架，用以量化属性流形与图热核之间的几何失配。

**💡 创新点**

创新点在于将几何失配视为可解释的结构信号，并通过可微分的 geodesic 近似实现端到端优化，提供了新的异常/社区检测指标。

**🔧 技术方法**

采用变分自编码器、热核（Heat Kernel）与可微分离散/线性 geodesic 近似，以及基于拉普拉斯/欧氏度量的正则化。

**📊 数据集**

实验数据包括合成的瑞士卷（Swiss‑roll）流形加偏好附件生成的图，以及法国巴黎大区公共交通网络与其社会经济属性。

**📈 对比分析**

与 Spectral、Louvain、VAE、VGAE 等基线对比，提出的失配得分在社区检测中 F1 最高达 82.5%，显著优于传统方法。

**⚠️ 局限性**

主要局限在于计算量大、可微 geodesic 近似在高维下精度下降，缺乏大规模可扩展性。

---

## 368. SOMBRERO: Measuring and Steering Boundary Placement in End-to-End Hierarchical Sequence Models

**arXiv ID:** 2601.22805 | [PDF](https://arxiv.org/pdf/2601.22805v1)

**作者:** Pit Neitemeier `[一作]` (Aleph Alpha Research), Jan Hendrik Metzen `[通讯]` (Aleph Alpha Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了端到端学习的分层序列模型中边界的质量评估与引导，并提出了基于预测难度的边界精炼指标和置信度对齐边界损失，提升了1B规模模型的准确性‑效率权衡。

**💡 创新点**

创新点包括：①提出 router‑agnostic 的边界丰富度指标；②在同一模型上以置信度对齐的边界损失直接引导边界位置；③将置信度加权平滑迁移至字节级别并将 chunker 简化为 sigmoid，显著提升性能。

**🔧 技术方法**

采用 Hierarchical Autoregressive Transformer（HAT）框架，使用 Mamba‑2 层编码器/解码器、Transformer 背景网络、sigmoid chunker、字节级置信度加权平滑以及置信度对齐边界损失，并通过 FLOP 匹配进行训练。

**📊 数据集**

使用 UTF‑8 字节序列混合数据集，包含英文、德文自然语言、代码与数学文本。

**📈 对比分析**

与固定尺寸切分基线、H‑Net 以及 tokenization 基线进行对比；在 1B 规模下，模型在 bits‑per‑byte（BPB）、边界丰富度 B、压缩率等指标上均优于对照组，最终实现 0.6568 BPB，B≈3.0。

**⚠️ 局限性**

局限性包括：对更深层级结构和更大规模训练的适应性尚未充分验证；对不同语言/领域的泛化能力有限；模型对分块率变化敏感，需进一步研究缩放行为。

---

## 369. RASST: Fast Cross-modal Retrieval-Augmented Simultaneous Speech Translation

**arXiv ID:** 2601.22777 | [PDF](https://arxiv.org/pdf/2601.22777v1)

**作者:** Jiaxuan Luo `[一作]` (Johns Hopkins University), Lei Li `[通讯]` (Carnegie Mellon University)

**通讯引用:** 11872 | [OpenAlex ID](https://openalex.org/A5100440407)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种检索增强的同步语音翻译框架，利用轻量级跨模态检索器与语音LLM实现对术语的实时精准翻译。

**💡 创新点**

创新点在于将滑动窗口跨模态检索与LLM增量生成耦合，并通过合成训练数据教会模型何时使用检索术语，从而显著提升罕见术语的翻译准确率。

**🔧 技术方法**

技术方案包括双编码器检索器（Qwen3-Omni音频编码器 + BGE-M3文本编码器）和FAISS滑动窗口检索，LLM使用Qwen3-Omni-30B-Instruct并通过LoRA微调。

**📊 数据集**

使用GigaSpeech数据集训练检索器，评估时采用ACL 60/60 dev集和自动抽取的会议论文术语表。

**📈 对比分析**

与InfiniSST和离线ST基线对比，检索增强模型在En→Zh/De/Ja方向上术语准确率提升最多16%，BLEU提升约3点，且仅增加约16%的计算开销。

**⚠️ 局限性**

局限性包括检索误差仍可能影响翻译效果，对极长或多义词术语的召回率仍有限，且对负采样策略依赖较大。

---

## 370. TSPO: Breaking the Double Homogenization Dilemma in Multi-turn Search Policy Optimization

**arXiv ID:** 2601.22776 | [PDF](https://arxiv.org/pdf/2601.22776v1)

**作者:** Shichao Ma `[一作]` (University of Science and Technology of China), Yang Wang `[通讯]` (University of Science and Technology of China)

**通讯引用:** 34170 | [OpenAlex ID](https://openalex.org/A5100764445)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于第一次出现的目标答案的潜在奖励机制(FOLR)，在多轮工具调用的强化学习中对每一步进行奖励分配，从而解决“Double Homogenization Dilemma”。

**💡 创新点**

创新点在于：①通过首次出现的答案作为隐式进度信号，无需外部标注或奖励模型；②在GRPO框架内实现按步优势估计，打破过程级与组内奖励同质化；③只需 (x,a_gold) 数据即可训练。

**🔧 技术方法**

使用的技术包括：基于 FOLR 的 turn‑level reward 分配、组相对政策优化（GRPO）改进、对抗 KL 正则化、掩码和填充处理多长度轨迹的优势计算。

**📊 数据集**

在七个问答基准上评估：一般 QA（NQ、TriviaQA、PopQA）和多跳 QA（HotpotQA、2WikiMultiHopQA、Musique、Bamboogle）。

**📈 对比分析**

与多种基线（直接推理、RAG、IRCoT、Search‑R1、ZeroSearch、MT‑PPO、StepSearch、ReasonRAG等）比较，TSPO 在 Qwen2.5‑7B‑Instruct 上平均提升 13.6%，在 Qwen2.5‑3B‑Instruct 上提升 24%，在所有数据集均超越最强对比方法。

**⚠️ 局限性**

局限性包括：依赖检索结果，若答案不在检索证据中则无效；仅在搜索增强的问答任务上验证，尚未证明对其他多工具使用场景的通用性。

---

## 371. Bayesian Interpolating Neural Network (B-INN): a scalable and reliable Bayesian model for large-scale physical systems

**arXiv ID:** 2601.22860 | [PDF](https://arxiv.org/pdf/2601.22860v1)

**作者:** Chanwook Park `[一作]` (Northwestern University), Wing Kam Liu `[通讯]` (Northwestern University)

**通讯引用:** 37258 | [OpenAlex ID](https://openalex.org/A5014509470)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种贝叶斯插值神经网络（B-INN），作为可扩展、可靠的代理模型，用于大规模物理系统的仿真。

**💡 创新点**

创新点在于：将插值神经网络与贝叶斯线性回归结合，利用张量分解（CP）和交替方向算法实现线性复杂度；理论证明B-INN为高斯过程的子空间，并在模式数趋向无穷时可恢复高斯过程；同时保持与高斯过程相当的预测精度与不确定性估计。

**🔧 技术方法**

使用技术包括：插值神经网络、张量分解、贝叶斯线性回归、交替方向（ALS）算法、Gaussian Process、Bayesian Neural Network（HMC、VI、MH）、主动学习策略。

**📊 数据集**

实验数据集：1D 合成正弦+余弦函数、BlendedNet 气动数据集、三维空间+1D 参数泊松方程、二维空间+时间+2D 参数热扩散方程。

**📈 对比分析**

与 GP、BNN（HMC、VI、MH）进行对比。B-INN 在预测精度上与 GP 接近或略优，训练时间比 GP 快 20–10,000 倍，BNN 更慢；在主动学习中，B-INN 的误差更低、收敛更稳定，且训练总耗时显著减少。

**⚠️ 局限性**

局限性：对高维问题的基函数数、长度尺度等超参数较为敏感，交替方向算法可能受局部优化结构影响；目前仅在固定插值基底上进行贝叶斯推断，缺乏全局变分或 MCMC 的全局探索；未来需要进一步研究超参数自适应和全局贝叶斯推断策略。

---

## 372. Just-in-Time Catching Test Generation at Meta

**arXiv ID:** 2601.22832 | [PDF](https://arxiv.org/pdf/2601.22832v1)

**作者:** Matthew Becker `[一作]` (Meta Platforms), Sophie Zeng `[通讯]` (Meta Platforms)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发并部署了“即时捕获测试”框架，利用差分感知的生成和评估技术，在Meta的数亿行代码系统中预先发现并阻止严重回归错误。

**💡 创新点**

将差分感知与意图感知结合到测试生成，设计基于LLM和规则的评估器过滤假阳性，并证明该方法能在生产前拦截多起严重缺陷。

**🔧 技术方法**

变异指导的LLM测试生成、意图推断、LLM‑as‑Judge评估器、Rule‑Based RubFake、协同判定以及持续集成与Diff Risk Score目标器等技术。

**📊 数据集**

Meta内部提交的数十万次diff（共22,126条生成测试）以及3.5 B用户访问的后台系统代码。

**📈 对比分析**

与传统硬化测试、偶然捕获和不感知基线对比，差分感知工作流产生弱捕获数4倍、20倍；LLM/规则评估器将人工审核负担降低70%，并在8起强捕获中预防了4起严重失效。

**⚠️ 局限性**

评估器准确率仍有限，缺乏高精度真阳性预测；方法高度依赖内部大规模数据与资源，难以直接迁移至开源或小型项目。

---

## 373. Decomposing and Composing: Towards Efficient Vision-Language Continual Learning via Rank-1 Expert Pool in a Single LoRA

**arXiv ID:** 2601.22828 | [PDF](https://arxiv.org/pdf/2601.22828v1)

**作者:** Zhan Fa `[一作]` (Nanjing University), Yinghuan Shi `[通讯]` (Nanjing University)

**通讯引用:** 4840 | [OpenAlex ID](https://openalex.org/A5055917015)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了一种无额外推理负载的连续学习框架，通过将单个LoRA模块拆分为Rank‑1专家池并基于语义引导路由器动态组合稀疏更新，以实现视觉‑语言模型在多域任务上的连续适应；

**💡 创新点**

创新点在于①把LoRA重构为可拆分的Rank‑1专家池，允许任务特定的稀疏更新；②提出激活引导正交（AGO）损失，利用专家激活频率实现任务间参数正交化，显著减少参数冲突；③在不使用外部数据或任务ID的情况下实现SOTA性能；

**🔧 技术方法**

使用技术包括LoRA、Rank‑1专家池、语义引导路由器、Activation‑Guided Orthogonal（AGO）损失、参数合并与稀疏化、CLIP视觉‑语言预训练模型以及Transformer结构；

**📊 数据集**

使用11个视觉任务数据集（Aircrft、Caltech101、CIFAR‑100、DTD、EuroSAT、Flowers、Food、MNIST、OxfordPet、StanfordCars、SUN397），在MTIL与X‑TAIL场景下采用5‑shot划分；

**📈 对比分析**

与ZSCL、MoE‑a、RAIL、GIFT等方法对比，在Transfer、Average、Last指标均优于对手（无任务ID时提升3–6%，有任务ID时提升1–2%），超过CLIP zero‑shot；训练参数量下降96.7%，GPU内存降低66.7%，推理无额外负载；

**⚠️ 局限性**

局限性包括需在每个任务保存LoRA权重和激活频率；对正交损失权重 λ 仍有一定敏感性；实验主要集中在MTIL/X‑TAIL，尚未验证更大规模或不同预训练模型的泛化能力；

---

## 374. User-Adaptive Meta-Learning for Cold-Start Medication Recommendation with Uncertainty Filtering

**arXiv ID:** 2601.22820 | [PDF](https://arxiv.org/pdf/2601.22820v1)

**作者:** Arya Hadizadeh Moghaddam `[一作]` (University of Kansas), Zijun Yao `[通讯]` (University of Kansas)

**通讯引用:** 1225 | [OpenAlex ID](https://openalex.org/A5040604135)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出MetaDrug，一种多层不确定性感知元学习框架，用于解决电子病历中患者冷启动药物推荐问题。

**💡 创新点**

创新点在于：①自适应层和同伴适应层两级元学习，利用患者自身和相似患者的访视信息；②引入不确定性量化过滤机制，剔除对自适应无关的访视；③将Transformer用于患者级和访视级嵌入，并结合偏好门控提升个性化预测。

**🔧 技术方法**

技术手段包括：基于MAML的元学习、两级Transformer编码器、偏好门控层、Jaccard相似度选取同伴访视、辅助误差预测的不确定性量化、二阶梯度更新与全局更新相结合。

**📊 数据集**

使用公开MIMIC‑III和自研AKI（急性肾损伤）两大临床数据库进行训练与评估。

**📈 对比分析**

与Deepr、GameNet、Micron、Retain、MELU、SafeDrug、Transformer、MoleRec、ARCI等基线模型对比，MetaDrug在PRAUC、F1、Jaccard指标上均取得最高分，冷启动患者表现尤为突出；但在DDI（药物相互作用）率上略高。

**⚠️ 局限性**

局限性包括：①DDI率仍高，需进一步加入药物相互作用约束；②对访视代码数量敏感，访视频次不足仍难以完全解决；③不确定性阈值β需经验调优，影响过滤效果；④仅在两份数据集上验证，泛化能力待进一步评估。

---

## 375. Cascaded Flow Matching for Heterogeneous Tabular Data with Mixed-Type Features

**arXiv ID:** 2601.22816 | [PDF](https://arxiv.org/pdf/2601.22816v1)

**作者:** Markus Mueller `[一作]` (Econometric Institute Erasmus University Rotterdam), Dennis Fok `[通讯]` (Econometric Institute Erasmus University Rotterdam)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `40105733-5154-44cd-8090-a8cab9e64b07` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种新的Cascade Flow Matching框架TabCascade，用于生成具有混合类型特征（离散+连续）的异构表格数据。

**💡 创新点**

创新点包括：1）将表格特征分层为低分辨率的离散表示和高分辨率的连续细节，采用级联结构；2）设计了由低分辨率信息引导的条件概率路径和数据自适应耦合的流匹配模型；3）通过非线性时间调度降低传输成本；4）直接处理混合型特征（缺失、零/一膨胀等）。

**🔧 技术方法**

使用的技术包括流匹配（Flow Matching）、条件概率路径设计、数据自适应耦合、非线性时间调度、分布回归树或高斯混合模型的离散化编码，以及CDTD等低分辨率生成器。

**📊 数据集**

在12个公开表格数据集（包括7个传统数据集和5个TabZilla基准）上进行实验，并在其中注入10%（可调到50%）的缺失值模拟。

**📈 对比分析**

与CTGAN、TVAE、ARF、TabDDPM、TabSyn、TabDiff、CDTD等state‑of‑the‑art生成模型进行比较，使用多种指标（检测得分、形状、Wasserstein距离、JSD、趋势、MLE、α‑Precision、β‑Recall、DCR Share、MIA）评估。TabCascade在检测得分提升约40%，Wasserstein距离下降50%，趋势指标提升9%/10%，下游任务MLE下降29%，显示出显著的性能优势。

**⚠️ 局限性**

限制包括：1）对低分辨率生成器的依赖，若选择不同模型会影响性能；2）在高缺失率下仍需更多实验验证；3）隐私保护未提供正式差分隐私保证；4）模型对离散化编码的依赖可能导致对特征分布的细粒度捕捉有限。

---

## 376. Quartet II: Accurate LLM Pre-Training in NVFP4 by Improved Unbiased Gradient Estimation

**arXiv ID:** 2601.22813 | [PDF](https://arxiv.org/pdf/2601.22813v1)

**作者:** Andrei Panferov `[一作]` (Institute of Science and Technology Austria), Dan Alistarh `[通讯]` (Institute of Science and Technology Austria)

**通讯引用:** 4412 | [OpenAlex ID](https://openalex.org/A5083822059)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种新的无偏量化方案 MS-EDEN，并将其集成到完整的 NVFP4 训练框架中，实现了更高精度且更快的大规模语言模型预训练。

**💡 创新点**

创新点在于将随机哈达玛变换与尺度校正相结合，生成无偏量化器 MS-EDEN，显著降低量化误差；同时引入 4/6 规模选择和完整的前向/后向 NVFP4 计算图，配合高效 CUDA 核心。

**🔧 技术方法**

采用的技术包括 NVFP4 微尺度 FP4/FP8/FP32 量化、随机哈达玛变换、无偏量化（MS-EDEN）、4/6 规模选择、重量化内核、NVIDIA Blackwell GPU 张量核心。

**📊 数据集**

实验使用 C4 语料库进行预训练，并在 Nanochat 任务中使用 FineWeb-Edu、ARC、GSM8K、Smol‑SmolTalk 等数据集。

**📈 对比分析**

与 BF16 基线以及现有 NVFP4 方案（TetraJet‑v2、FourOverSix）对比，MS‑EDEN+完整 NVFP4 方案在验证损失上至少降低 20%，在 Blackwell GPU 上实现了 4.2× 的速度提升，并比现有 FP4 核心快约 70%。

**⚠️ 局限性**

局限性包括：对特定硬件（Blackwell GPU）的依赖；量化与旋转操作对实现复杂度高；在极大规模模型（>1B 参数或 >1T tokens）上的鲁棒性尚未完全验证；4/6 规模选择在后向传递中仍可能引入偏差。

---

## 377. Stable Personas: Dual-Assessment of Temporal Stability in LLM-Based Human Simulation

**arXiv ID:** 2601.22812 | [PDF](https://arxiv.org/pdf/2601.22812v1)

**作者:** Jana Gonnermann-Müller `[一作]` (Zuse Institute Berlin), Sebastian Pokutta `[通讯]` (Zuse Institute Berlin)

**通讯引用:** 1951 | [OpenAlex ID](https://openalex.org/A5043574831)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过双源评估框架研究大型语言模型在不同语境下维持人格稳定性的表现。

**💡 创新点**

创新点在于提出双源评估、使用 ADHD 诊断特征测试人格稳定性，并量化跨模型、跨提示的方差。

**🔧 技术方法**

采用了七种主流 LLM（Claude、DeepSeek、GPT、Gemini、Grok、Llama 等）与三种等价提示，并使用 CAARS 自评与观察者评测。

**📊 数据集**

数据集为 3,473 次单轮对话与 1,370 次 18 轮对话，共计 12,201 次观察评估与 4,054 次自评，涵盖四种 ADHD 强度水平。

**📈 对比分析**

通过线性混合模型与标准差分解，结果显示自评稳定性极高，观测评估在高/中强度会随对话延伸而衰减；整体性能在模型间差异小，提示差异更小。

**⚠️ 局限性**

局限包括仅评估 ADHD 这一单一构念、对话长度有限（18 轮）、测量工具受限于标准量表，且自评与观测评估存在差异。

---

## 378. Trojan-Resilient NTT: Protecting Against Control Flow and Timing Faults on Reconfigurable Platforms

**arXiv ID:** 2601.22804 | [PDF](https://arxiv.org/pdf/2601.22804v1)

**作者:** Rourab Paul `[一作]` (Shiv Nadar University), Amlan Chakrabarti `[通讯]` (University of Calcutta)

**通讯引用:** 5124 | [OpenAlex ID](https://openalex.org/A5043543748)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

设计并实现了一个安全的 NTT（Number‑Theoretic Transform）硬件架构，能够检测并校正因硬件木马导致的控制信号失真、异常延迟以及 SASCA 侧信道攻击，同时提供自适应故障纠正机制。

**💡 创新点**

创新点包括：① 独立的 shift‑register 备份 CSR 用以实现控制流完整性检查；② 结合 Clock Cycle Counter（CCC）检测异常时序；③ 在 butterfly 单元使用本地掩码（Local Masking）随机化 twiddle factor 防止 SASCA；④ 基于风险因子 R_i 的自适应故障纠正模块，可按故障类型动态选择重跑、重载或迁移 PR 位流；⑤ 整体实现资源开销低（检测仅 8.7% 片区增幅，检测+纠错 19.7%）。

**🔧 技术方法**

使用的技术包括：shift‑register CSR、CCC、LM、本地掩码、分区可重配置（PR）与 ICAP、风险因子 R_i 计算与四阈值机制、VHDL 设计、Vivado 2022.2、C 语言位流配置，针对 Artix‑7 FPGA 进行实现与仿真。

**📊 数据集**

实验数据集为 Kyber 公开密钥加密算法的三种变体（Kyber‑512、768、1024），n=256、q=3329，使用 4096 次 KeyGen/Encap/Decap，总共 69,632/98,304/126,976 次 NTT 运行；在每次运行中随机注入 10 组控制信号故障，覆盖所有 1024 个时钟周期。

**📈 对比分析**

与基线 NTT 以及文献中已有的控制信号保护方案进行对比：检测阶段资源占用提升 8.7%、能量+1%，检测+纠错阶段提升 19.7%、能量+3%，无时序损失；在所有注入的硬件木马故障中 100% 能被检测并成功纠正。纠正时延取决于措施：重跑仅 1 周期；重载 PR 位流约 150 µs；迁移 PR 位流约 256 µs。总体而言，资源和性能优于或可与现有方案媲美。

**⚠️ 局限性**

局限性：① 只针对控制信号故障，未覆盖数据路径故障；② 方案针对 NTT，迁移至其他模块需要重新设计；③ 需要可重配置位流与外部 CPU 交互；④ 侧信道泄漏在逆变换（INTT）后仍可能出现；⑤ 对硬件木马位置与注入阶段的假设限制了适用范围。

---

## 379. CVeDRL: An Efficient Code Verifier via Difficulty-aware Reinforcement Learning

**arXiv ID:** 2601.22803 | [PDF](https://arxiv.org/pdf/2601.22803v1)

**作者:** Ji Shi `[一作]` (Harbin Institute of Technology), Weili Guan `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 2249 | [OpenAlex ID](https://openalex.org/A5075938343)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出了基于强化学习的代码验证框架 CVeDRL，通过生成单元测试来过滤 LLM 生成的代码候选，提升代码选择的准确性和效率。

**💡 创新点**

创新点包括：① 将分支覆盖率、样本难度、语法合法性与功能正确性三方面融入多维奖励；② 引入指数奖励塑形和静态分析指标（Halstead、Maintainability Index）来实现分支难度和样本难度感知；③ 通过理论推导阐明单元测试多模态奖励对验证置信度的影响，指导奖励设计；④ 在仅 0.6B 参数规模下实现与大模型同等级甚至更优的验证效果。

**🔧 技术方法**

主要技术：强化学习（GRPO）结合语法奖励与功能奖励；指数奖励塑形；静态代码复杂度评估；单元测试多数表决框架；基于 AST 的语法检查。

**📊 数据集**

使用数据集：HumanEval+、MBPP+、LiveCodeBench、LeetCode（用于验证和评估），CodeRM 数据集（训练时使用），并在四大基准上进行对照实验。

**📈 对比分析**

相较于 GPT‑3.5、GPT‑4o‑mini、CodeRM‑8B 等基线，CVeDRL‑0.6B 在 Pass Rate、Branch Coverage 等指标上提升 17–28%（取决于数据集），且在推理吞吐量上比对照模型快 20 倍以上，显著降低资源消耗和延迟。

**⚠️ 局限性**

局限性：仅支持完整函数级 Python 代码，无法处理部分代码或多文件/类结构；生成的单元测试仍无法完全判断代码正确性；未覆盖更多编程语言与更大规模代码库。

---

## 380. Clipping-Free Policy Optimization for Large Language Models

**arXiv ID:** 2601.22801 | [PDF](https://arxiv.org/pdf/2601.22801v1)

**作者:** Ömer Veysel Çağatan `[一作]` (KUIS AI Center), Xuandong Zhao `[通讯]` (University of California)

**通讯引用:** 366 | [OpenAlex ID](https://openalex.org/A5068022531)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种无剪切的策略优化方法CFPO，用于大语言模型的后训练。

**💡 创新点**

创新点在于用总变差约束的凸二次惩罚替代传统剪切机制，实现全局可微且自适应的信赖域约束。

**🔧 技术方法**

技术主要是将PPO/GRPO的剪切目标改写为二次惩罚，结合KL正则并使用组相对优势或留一优势估计。

**📊 数据集**

使用的数据集包括MATH、GSM8K、AIME24、GPQA-Diamond等推理数据集，以及OpenRLHF的奖励模型和AlpacaEval、Arena-Hard、MT-Bench、IFEval、OpenLLM Leaderboard等对齐评测集。

**📈 对比分析**

与传统PPO/GRPO/RLOO相比，CFPO在推理训练中延迟崩溃点、在对齐训练中降低verbosity、保持更低的对齐税，且在各下游指标上与基线保持相近甚至略优。

**⚠️ 局限性**

局限在于仅在1.5B-8B规模模型、特定数据集和训练配置下验证，缺乏对更大规模或不同任务（如代码生成、代理学习）的评估。

---

## 381. HeatMat: Simulation of City Material Impact on Urban Heat Island Effect

**arXiv ID:** 2601.22796 | [PDF](https://arxiv.org/pdf/2601.22796v1)

**作者:** Marie Reinbigler `[一作]` (Inria Saclay), Rosalie Martin `[通讯]` (Adobe Research)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

通过结合 OpenStreetMap 数据与街景 360° 图像，利用预训练的视觉语言模型（VLM）估算建筑立面材质；将城市几何和材质信息编码成可随机访问的二维纹理，构建 2.5D 城市模型；在 GPU 像素着色器上实现蒙特卡罗耦合热传递模拟器，用以预测城市表面温度并评估 UHI 效应。

**💡 创新点**

①首次将 VLM 与街景图像结合进行大规模立面材质估算并生成统计分布；②提出将城市信息压缩为 18 张 2D 纹理，支持快速随机访问，从而实现 2.5D 热传递模拟；③在 GPU 上实现直接、间接太阳辐射、辐射、对流和导热的耦合，获得 20× 的速度提升。

**🔧 技术方法**

计算机视觉（GroundedSAM、Llava VLM）、GIS（OpenStreetMap）、GPU 图形技术（球面追踪、随机采样）、蒙特卡罗热传递、有限差分方法（用于验证）、光照与材料属性数据库。

**📊 数据集**

OpenStreetMap 城市几何、Mapillary Metropolis 360° 街景图像、人工标注的 50 张立面图材质基准、Landsat 地表温度影像、Stardis 3D 模拟器。

**📈 对比分析**

与有限差分方法、Stardis 3D 模拟器以及 Landsat 热图进行对比；误差均在 1% 以内；与 Stardis 相比，GPU 版实现 20× 的速度提升；与 Landsat 热图梯度高度一致。

**⚠️ 局限性**

依赖开源数据覆盖度，材质库有限；VLM 识别准确度受限；仅考虑表面热传递，未处理角落切换、植被或人为热源等因素；仅支持 2.5D 结构，无法完整模拟三维内部热流。

---

## 382. Constructing Safety Cases for AI Systems: A Reusable Template Framework

**arXiv ID:** 2601.22773 | [PDF](https://arxiv.org/pdf/2601.22773v1)

**作者:** Sung Une Lee `[一作]` (CSIRO), Jieshan Chen `[通讯]` (CSIRO)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了一套可复用的 AI 安全案例（Safety Case）模板框架，并通过系统性文献综述（SLR）梳理了 AI 安全案例的主流说法、分类与模式，构建了针对 AI 系统的 Claim–Argument–Evidence（CAE）三维分类体系，进一步给出四类典型安全案例模式（发现驱动评估、无基准边际风险、持续演化、阈值决策），并在政府招标评估系统案例中演示了这些模式的落地；

**💡 创新点**

①将传统安全案例与 AI 系统差异化，提出 AI 专用的 CAE 分类；②设计可组合、可复用的模板与模式库；③将安全案例与动态评估、阈值监测、对比验证等方法结合，形成持续更新的“动态安全案例管理”理念；

**🔧 技术方法**

系统性文献综述方法（关键词搜索、筛选、质量评估）；构建 CAE 分类与模板；案例演示采用统计与对比分析；动态安全案例管理与阈值决策所用的安全性能指标与统计方法；

**📊 数据集**

无公开实验数据集，案例使用的是政府招标评估的合成 200 个招标案例与人工评审基准；

**📈 对比分析**

通过对比分析（人机、人机 vs 人机）和阈值检验（误差率 ≤5%），实现“至少不劣于”与“安全阈值内”的双重证明；在 200 个案例实验中，AI‑人评估的不一致率 2.8% < 人机评估 3.0%，且落在预设阈值 5% 内，表明系统安全可接受；

**⚠️ 局限性**

缺乏对动态更新的自动化工具支持，模板需手工维护；在不同监管环境下的适配与标准化仍未完成；对大型前沿 AI 系统的实证验证不足，尚需更多真实案例和数据支撑。

---

## 383. MEnvAgent: Scalable Polyglot Environment Construction for Verifiable Software Engineering

**arXiv ID:** 2601.22859 | [PDF](https://arxiv.org/pdf/2601.22859v1)

**作者:** Chuanzhe Guo `[一作]` (Harbin Institute of Technology), Haifeng Wang `[通讯]` (Baidu Inc.)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种多语言自动化环境构建框架MEnvAgent，能够在10种主流编程语言下自动构建可执行、可验证的软件工程实验环境。

**💡 创新点**

其创新点在于引入多代理计划‑执行‑验证循环以及环境重用机制，显著提升构建成功率与效率，并首次构建大规模可验证多语言数据集。

**🔧 技术方法**

技术上结合了多代理架构、LLM推理、增量环境补丁、环境检索与迭代调试，并通过Docker容器实现可验证执行。

**📊 数据集**

实验使用了在10个主流语言的200个开源仓库上构建的MEnvBench（1000个任务）和MEnvData‑SWE（3005个任务）作为训练与评测数据集。

**📈 对比分析**

与Repo2Run、SWE‑Bench‑Live、SWE‑Factory等基线比较，MEnvAgent在Fail‑to‑Pass率提升8.6%、通过率提升11.0%，时间成本下降43%，在多语言任务上均表现优异。

**⚠️ 局限性**

局限性包括对历史环境库规模的依赖，环境重用率随数据量波动；对极其复杂或自定义构建系统（如大型Java项目）仍可能失败；对LLM生成命令的安全与准确性需进一步控制。

---

## 384. OptiMAG: Structure-Semantic Alignment via Unbalanced Optimal Transport

**arXiv ID:** 2601.22856 | [PDF](https://arxiv.org/pdf/2601.22856v1)

**作者:** Yilong Zuo `[一作]` (Beijing Institute of Technology), Guoren Wang `[通讯]` (Beijing Institute of Technology)

**通讯引用:** 7223 | [OpenAlex ID](https://openalex.org/A5054991337)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出OptiMAG框架，通过Unbalanced Optimal Transport（UOT）对多模态属性图（MAG）中的结构-语义冲突进行正则化，从而提升节点表征学习和多模态生成任务的效果。

**💡 创新点**

创新点在于：①将结构-语义对齐建模为Unbalanced Fused Gromov‑Wasserstein（UFGW）问题；②利用KL惩罚实现自适应地抛弃噪声节点；③将该正则化作为plug‑and‑play模块，可无缝集成到现有MAG模型；④通过子图采样+Sinkhorn迭代将GW计算复杂度降至线性。

**🔧 技术方法**

核心技术包括：多模态特征编码（BERT、ResNet等）、图结构的PPR扩散、Unbalanced OT、Fused GW距离、Sinkhorn迭代、Block Coordinate Descent（BCD）优化。

**📊 数据集**

实验使用OpenMAG六大基准数据集：Grocery、Movies、Toys、Reddit‑S（图中心任务），Flickr30k、SemArt（多模态生成任务）。

**📈 对比分析**

在节点分类、链接预测、聚类、图到文本和图到图像等任务中，OptiMAG在三种主流MAG模型（MMGCN、MGAT、UniGraph2）上均取得显著提升，节点分类准确率平均提升约2%，CIDEr最高提升4.6点，最显著提升出现在预训练模型UniGraph2上。

**⚠️ 局限性**

局限性包括：①UFGW仍需较多超参调优（α、ρ、τ等）；②虽然子图采样缓解计算量，但在极大图上仍可能受限于GPU内存；③目前仅支持图像与文本两种模态，扩展到更多模态或动态图需要进一步研究。

---

## 385. Unconditional flow-based time series generation with equivariance-regularised latent spaces

**arXiv ID:** 2601.22848 | [PDF](https://arxiv.org/pdf/2601.22848v1)

**作者:** Camilo Carvajal Reyes `[一作]` (Imperial College), Felipe Tobar `[通讯]` (Imperial College)

**通讯引用:** 731 | [OpenAlex ID](https://openalex.org/A5020822083)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出在时序生成中使用自编码器学习具有等变性的潜在空间，并在该空间上训练流匹配模型，实现高效无条件时序生成。

**💡 创新点**

通过在预训练自编码器上加入等变性正则化损失，使潜在空间对平移和幅度缩放等变换保持等变性，从而提升生成质量并显著加速采样。

**🔧 技术方法**

结合 VAE/AE、等变性正则化、潜在空间流匹配（latent flow matching）、ODE 解算器（adjoint/euler）以及基于 U‑Net 的流网络。

**📊 数据集**

使用 Household Electric Power Consumption、外汇汇率（8 维）以及 Weather（14 维）等真实时序数据集。

**📈 对比分析**

与四种主流扩散模型（SigDiffusion、Diffusion‑TS、CSPD‑GP、DDO）在判别分数、预测分数和 KS 检验等指标下对比，结果显示在判别分数和 KS 得分上显著优于基线，且采样速度提升至秒级。

**⚠️ 局限性**

等变性正则化是经验式约束，无法保证严格的群等变性；对更复杂变换的泛化有限，且模型对变换范围等超参数较为敏感。

---

## 386. SWE-Manager: Selecting and Synthesizing Golden Proposals Before Coding

**arXiv ID:** 2601.22956 | [PDF](https://arxiv.org/pdf/2601.22956v1)

**作者:** Boyin Tan `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Youcheng Sun `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**通讯引用:** 2208 | [OpenAlex ID](https://openalex.org/A5060663223)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本工作先对开源项目维护者在面对多个修复提案时的选择理由进行手工分析，随后设计了SWE‑Manager模型，能够在不执行代码或跑测试的前提下对同一问题的多个候选提案进行推理、选择最佳方案并合成一份“金色提案”以指导后续实现。

**💡 创新点**

①通过实证研究提炼出四大选择主题（风险与安全、修复深度、可维护性、其他），为模型提供判据；②将提案选择与合成联合为任务，利用合成环节迫使模型展开跨提案比较；③采用决策对齐的强化学习（DAPO）在8B模型上实现高质量的推理与选择；④引入P2A框架评估在完整工作流中的效益。

**🔧 技术方法**

大语言模型（Qwen3‑8B、GPT‑5）、监督微调、强化学习（DAPO）、结构化推理模板、奖励分解（选择、推理、理由、金色提案），以及SWE‑Lancer基准与P2A工作流。

**📊 数据集**

GitHub公开issue与候选提案数据集（2852个issue、7500个提案），SWE‑Lancer Manager与IC两大benchmark。

**📈 对比分析**

在Manager基准中，SWE‑Manager实现53.21%匹配率、57.75%奖励率，较GPT‑5提升约9%准确率、20%奖励；在IC基准中，55.6%通过率，比基线提升7.1%，与GPT‑5保持同等表现。

**⚠️ 局限性**

①模型仅依赖文本，缺乏对代码上下文的访问，导致在高价值实例上仍略逊于GPT‑5的agentic方式；②对多提案集（>3）鲁棒性下降，因训练集中此类样本比例低；③模型在长文本输入下的推理稳定性待提升。

---

## 387. Sifting the Noise: A Comparative Study of LLM Agents in Vulnerability False Positive Filtering

**arXiv ID:** 2601.22952 | [PDF](https://arxiv.org/pdf/2601.22952v1)

**作者:** Yunpeng Xiong `[一作]` (University of Melbourne), Ting Zhang `[通讯]` (Monash University)

**通讯引用:** 4509 | [OpenAlex ID](https://openalex.org/A5101691764)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对三种 LLM 代理框架（Aider、OpenHands、SWE‑agent）在 SAST 工具产生的误报（FP）过滤任务上进行系统性评估，结合 OWASP Benchmark 与 Vul4J 两个数据集，探究不同后端模型与漏洞类别对代理效果的影响。

**💡 创新点**

首次将代理式推理与传统一次性 LLM 交互在 FP 过滤上的差异进行量化比较，并揭示代理框架对模型强度、CWE 分类的依赖；提出基于模型、漏洞类型与成本的实用部署建议。

**🔧 技术方法**

使用 Perceive–Reason–Act 循环的 LLM 代理框架（Aider、OpenHands、SWE‑agent）配合 Claude Sonnet 4、DeepSeek Chat、GPT‑5 等大语言模型；评估指标包括 FP‑率、TP‑保留率、误判率、交互轮次、Token 与成本消耗。

**📊 数据集**

数据集：① OWASP Benchmark（Java 1.2）——1,325 非漏洞实例、1,415 真漏洞实例；② Vul4J（79 版本）——79 个 Java 项目，生成 3,426 条 CodeQL 警报，随机抽取 50 条进行手工标注。

**📈 对比分析**

方法：统一提示模板、禁用外部浏览器、对每种模型+框架组合评估 FP‑过滤率、TP 识别率、误判率和算力成本。性能显示：最佳配置（Claude + SWE‑agent）将 OWASP 上 FP‑率从 98.3% 降至 6.3%（降低 92.1%）；在真实项目中 FP 识别率最高可达 93.3%；但不同模型、CWE 之间存在显著差异，且代理框架成本差距大。

**⚠️ 局限性**

限制：① LLM 随机性导致可复现性受限；② Vul4J 评估仅用 50 条警报样本，样本偏倚可能影响结果；③ 仅覆盖 Java，结果对 C/C++ 等语言的推广性未知；④ 误报过滤过度时会误删真实漏洞，尤其在加密、政策类 CWE 上误判率高；⑤ 代理框架的交互轮次与 Token 消耗差异显著，实际部署需权衡成本与效果。

---

## 388. Benchmarking Machine Translation on Chinese Social Media Texts

**arXiv ID:** 2601.22931 | [PDF](https://arxiv.org/pdf/2601.22931v1)

**作者:** Kaiyan Zhao `[一作]` (University of Tokyo), Shaosheng Cao `[通讯]` (Xiaohongshu Inc)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研发了一个覆盖五种中外语言方向的中文社交媒体机器翻译基准，包含 Fun Posts（长篇含俚语/新词）和 Social Snippets（短篇情绪化表达）两子集，并提供人工双语专家译文。

**💡 创新点**

创新点在于：①针对社交媒体非标准语言设计了专门的子集和评估方法（Slang Success Rate、Embedding Similarity + LLM-as-judge），②系统性比较多种 MT 与 LLM 模型在俚语保留与情感/风格传递方面的表现，揭示现有模型的不足。

**🔧 技术方法**

采用 GPT‑5 生成俚语词典与模糊匹配、XCOMET、embedding‑based style/emo/sentiment 相似度、GEMBA‑stars 提示评估等技术，并对 22 种闭源、开源、专用及 LLM 模型进行实验。

**📊 数据集**

使用自制的 CSM‑MTBench 数据集（约 10,000 条样本），Fun Posts 包含 1,183 条含俚语/新词的帖子，Social Snippets 为数千条简短情绪化短句，均由双语专家翻译。

**📈 对比分析**

通过 XCOMET、Slang Success Rate、Embedding Similarity（ES）和 GEMBA 进行多维度评测。实验表明闭源 LLM（如 GPT‑4o、GPT‑5）总体性能最佳；GPT‑5 在俚语保留与情绪/风格传递上优于 GPT‑4o；翻译专用模型落后；开源 LLM 随模型规模提升而表现提升，指示规模化与后训练对翻译鲁棒性有显著影响。

**⚠️ 局限性**

局限性：仅探讨了简单提示提升方法，未涉及更深层的俚语预训练或专门微调；数据仅覆盖五个语言方向，扩展性受标注成本限制；评测主要集中在自动化指标，尚缺乏更细粒度的人类评价。

---

## 389. Feedback Control via Integrated Sensing and Communication: Uncertainty Optimisation

**arXiv ID:** 2601.22912 | [PDF](https://arxiv.org/pdf/2601.22912v1)

**作者:** Touraj Soleymani `[一作]`, John S. Baras `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本论文研究了集成感知与通信（ISAC）架构在网络控制系统中的反馈控制设计，重点关注如何在不确定性下优化物理过程的控制性能。

**💡 创新点**

创新点在于提出了一种基于阈值的最优切换策略，明确了在源和基站估计协方差的联合空间中，感知与通信的动态平衡。

**🔧 技术方法**

使用了高斯-马尔可夫模型、有限时域线性-二次-高斯（LQG）成本函数以及动态规划技术。

**📊 数据集**

未具体提及使用的数据集，但研究基于高斯-马尔可夫源和伯努利感知与通信链路的模型。

**📈 对比分析**

与传统的感知和通信调度方法相比，提出的ISAC系统在不确定性下的动态策略显示出更优的控制性能，具体性能通过数值实验验证。

**⚠️ 局限性**

限制在于模型假设的简化，未考虑更复杂的环境因素和实际应用中的多样性。

---

## 390. Status Updating via Integrated Sensing and Communication: Freshness Optimisation

**arXiv ID:** 2601.22901 | [PDF](https://arxiv.org/pdf/2601.22901v1)

**作者:** Touraj Soleymani `[一作]`, John S. Baras `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

研究在ISAC系统中，利用状态更新和信息新鲜度（AoI）优化远程导航代理的感知与通信调度。

**💡 创新点**

证明最优策略呈单调阈值结构，阈值随基站AoI递增，提供可解释且低复杂度的决策规则。

**🔧 技术方法**

使用马尔可夫决策过程（MDP）与Bellman迭代、单调性和次模性分析技术。

**📊 数据集**

未使用真实数据集，采用数值仿真设置（A_max=30, γ=0.95, λ_s=0.6, λ_c=0.9, c_s=0.2, c_c=0.1）。

**📈 对比分析**

通过价值函数与决策图可视化验证阈值结构，结果表明阈值越高时通信更优，模型在仿真中表现稳定。

**⚠️ 局限性**

局限包括：仅考虑两种操作（感知/通信）、假设通信可靠性≥感知、离散时间和截断AoI，未考虑多源或网络拥塞等实际复杂性。

---

## 391. Calibrated Multivariate Distributional Regression with Pre-Rank Regularization

**arXiv ID:** 2601.22895 | [PDF](https://arxiv.org/pdf/2601.22895v1)

**作者:** Aya Laajil `[一作]` (Mohamed Bin Zayed University of Artificial Intelligence), Souhaib Ben Taieb `[通讯]` (Mohamed Bin Zayed University of Artificial Intelligence)

**通讯引用:** 2720 | [OpenAlex ID](https://openalex.org/A5036751620)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种基于预秩函数的正则化方法，在多变量分布回归训练过程中直接强制校准，并引入了新的PCA预秩。

**💡 创新点**

创新点在于把预秩校准误差嵌入可微正则项并首次提出沿主成分方向的PCA预秩，用于捕捉依赖结构失配。

**🔧 技术方法**

核心技术包括严格正则化的PCE-KDE正则项、严格正则得分（如负对数似然）、混合高斯模型、梯度优化和多种预秩（位置、尺度、依赖、HDR、Copula、PCA）。

**📊 数据集**

使用18个真实多输出回归数据集进行评估，并在仿真中使用多变量高斯和高斯随机场来检验方法。

**📈 对比分析**

与未正则化基线相比，正则化模型在PCE、NLL和能量评分上均保持或提升性能，显著降低多变量预秩的校准误差。

**⚠️ 局限性**

局限性包括需要手动选择预秩和正则权重、计算成本相对较高、对非高斯或极高维数据的泛化性待验证，以及对某些特殊误差类型的敏感性有限。

---

## 392. Matterhorn: Efficient Analog Sparse Spiking Transformer Architecture with Masked Time-To-First-Spike Encoding

**arXiv ID:** 2601.22876 | [PDF](https://arxiv.org/pdf/2601.22876v1)

**作者:** Zhanglu Yan `[一作]` (National University of Singapore), Weng-Fai Wong `[通讯]` (National University of Singapore)

**通讯引用:** 4266 | [OpenAlex ID](https://openalex.org/A5023989495)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了高能效的稀疏脉冲变换器 Matterhorn

**💡 创新点**

创新点是掩码时序编码M‑TTFS与死区策略，以及利用计算‑在‑内存的忆阻突触单元MSU

**🔧 技术方法**

使用M‑TTFS编码、死区调节、MSU CIM计算、QNN→SNN 转换训练技术

**📊 数据集**

在GLUE基准任务上进行评测

**📈 对比分析**

相较于现有SNN模型，GLUE平均准确率提升1.42%，能耗降低2.31倍

**⚠️ 局限性**

死区过大会导致准确率下降，且硬件实现仍需进一步验证

---

## 393. MTDrive: Multi-turn Interactive Reinforcement Learning for Autonomous Driving

**arXiv ID:** 2601.22930 | [PDF](https://arxiv.org/pdf/2601.22930v1)

**作者:** Xidong Li `[一作]` (Li Auto Inc), Zehuan Wang `[通讯]` (NVIDIA)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了MTDrive框架，利用多轮交互式强化学习与大型视觉语言模型改进自动驾驶轨迹规划

**💡 创新点**

提出多轮Group Relative Policy Optimization（mtGRPO）解决奖励稀疏问题，构建多轮交互式数据集，并通过IPSS/IPTC系统优化提升训练吞吐量

**🔧 技术方法**

采用视觉语言模型Qwen2.5-VL-7B-Instruct、强化学习GRPO/mtGRPO、PDM Agent反馈、多轮交互式训练、veRL系统以及IPSS/IPTC等技术

**📊 数据集**

使用NAVSIM仿真平台的PDM Agent、RecogDrive轨迹数据以及闭环仿真生成的多轮交互式数据集

**📈 对比分析**

在NAVSIM基准上与多种端到端与VLM方法对比，MTDrive在PDMS指标上达到96.2（oracle）/91.1（kinematic），超过人类和现有方法

**⚠️ 局限性**

依赖外部感知或真实感知输入，PDM反馈需人工标注，当前验证仅在规则型Agent上，需进一步推广至其他仿真环境

---

## 394. DC-LA: Difference-of-Convex Langevin Algorithm

**arXiv ID:** 2601.22932 | [PDF](https://arxiv.org/pdf/2601.22932v1)

**作者:** Hoang Phuc Hau Luu `[一作]` (Nanyang Technological University), Zhongjian Wang `[通讯]` (Nanyang Technological University)

**通讯引用:** 743 | [OpenAlex ID](https://openalex.org/A5022215899)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出了一种差分凸（DC）拉格朗日算法（DC‑LA），用于在满足远程耗散性假设的前提下，从非光滑、非对数凹的DC正则化分布中进行采样，并给出了Wasserstein距离上的收敛分析；

**💡 创新点**

创新点在于将Moreau包络平滑分别应用于DC正则化的两个凸分量，构造前向-后向拆分的Langevin采样器，克服了弱凸性限制，并在理论上实现了对一般DC正则化的采样收敛保证；

**🔧 技术方法**

采用了Moreau包络平滑、前向-后向分裂、差分凸规划、Wasserstein距离分析、拉格朗日动力学以及随机梯度噪声的数值积分技术；

**📊 数据集**

实验使用了二维高斯观测数据（μ∈{(0,0),(1,1),(2,2)}）以及Mayo Clinic腹部CT扫描512×512；

**📈 对比分析**

与ULA、Moreau ULA和PSGLA等基线进行比较，DC‑LA在KL/ Wasserstein距离、后验均值收敛速度和CT中方差估计精度上均优于其他方法；

**⚠️ 局限性**

局限性包括对正则化项的Lipschitz和远程耗散性假设的依赖、对近似prox运算误差的有限考虑，以及在高维大规模问题上的计算成本仍需进一步优化。

---

## 395. Toward Fully Autonomous Driving: AI, Challenges, Opportunities, and Needs

**arXiv ID:** 2601.22927 | [PDF](https://arxiv.org/pdf/2601.22927v1)

**作者:** Lars Ullrich `[一作]` (Friedrich-Alexander-Universität Erlangen-Nürnberg), Knut Graichen `[通讯]` (Ulm University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文综述了当前自动驾驶技术与人工智能的交叉发展，提出了两阶段微调流程（先环境级奖励模型后车辆级迁移学习）以实现跨场景与跨车辆的可扩展性，并讨论了情境感知、模块化与端到端架构的融合与挑战。

**💡 创新点**

创新点在于将情境感知与法规、文化、伦理需求统一纳入奖励模型，提出基于注意力与查询接口的自适应服务化端到端架构（SO‑M‑E2E），并为多源外部信息融合与可解释性提供概念框架。

**🔧 技术方法**

主要技术包括深度强化学习、迁移学习、注意力机制、查询/标记接口、基础模型（LLM、VLM、VLA）以及多模态感知与预测模块。

**📊 数据集**

本文并未在实验中使用特定数据集，而是引用并讨论了公开数据集（如KITTI、nuScenes、Waymo、Argo、Cruise等）以及相关的HD地图、V2X消息等信息来源。

**📈 对比分析**

由于本工作为综述与概念设计，未进行具体的实验对比或性能评估；作者指出需在未来通过多域多车实验验证所提出的两阶段微调与SO‑M‑E2E框架的效果。

**⚠️ 局限性**

主要局限包括：缺乏实证验证、模型与监管、可解释性与安全保障的深度融合尚未完善、对外部数据的信任与完整性要求高，以及在多国法规与文化差异下的适配复杂性。

---

## 396. Perplexity Cannot Always Tell Right from Wrong

**arXiv ID:** 2601.22950 | [PDF](https://arxiv.org/pdf/2601.22950v1)

**作者:** Petar Veličković `[一作]` (Google DeepMind), Razvan Pascanu `[通讯]` (Google DeepMind)

**通讯引用:** 29863 | [OpenAlex ID](https://openalex.org/A5043910056)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文证明并验证了在某些Transformer模型中，低perplexity并不一定意味着高预测准确性，尤其在模型对输入过度自信时。

**💡 创新点**

创新点在于将Transformer连续性理论与perplexity的局限性结合，给出严格的定理说明高信心预测会导致perplexity无法识别错误，并通过iso‑perplexity曲线揭示“误报”区域。

**🔧 技术方法**

使用的技术包括Transformer连续性理论、理论证明、熵与confidence分析、可视化的iso‑perplexity曲线以及在不同采样策略下的实验评估。

**📊 数据集**

数据集涵盖了自定义的二进制复制任务、Gemma 3 4B大模型的复制请求以及长度从16扩展到128的Parity（异构输入）任务。

**📈 对比分析**

与传统基于准确率或F1的评估对比，实验显示perplexity与准确率在OOD情形下的相关性趋于正值，表明perplexity在这些场景下可能错误地优先选择表现更差的模型；在ID情形下相关性仍为负，说明其有效性有限。

**⚠️ 局限性**

局限性包括：仅适用于compact position embeddings的Transformer；实验多依赖贪婪解码和理想化的confidence假设；理论结果在真实多模态、长文本任务中的可推广性待验证。

---

## 397. LLMDR: Large language model driven framework for missing data recovery in mixed data under low resource regime

**arXiv ID:** 2601.22916 | [PDF](https://arxiv.org/pdf/2601.22916v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99`

---

## 398. From Data Leak to Secret Misses: The Impact of Data Leakage on Secret Detection Models

**arXiv ID:** 2601.22946 | [PDF](https://arxiv.org/pdf/2601.22946v1)

**作者:** Farnaz Soltaniani `[一作]` (Clausthal University of Technology), Mohammad Ghafari `[通讯]` (Clausthal University of Technology)

**通讯引用:** 1094 | [OpenAlex ID](https://openalex.org/A5024783227)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究对公开的 SecretBench 数据集进行重复代码检测与去重，并评估去重对基于机器学习的硬编码秘密检测模型性能的影响。

**💡 创新点**

创新之处在于系统量化重复样本对模型评估的扭曲效应，提出三种评估场景（混合、近似重复、唯一）并展示不同模型对数据泄漏的鲁棒性差异。

**🔧 技术方法**

采用 Jaccard 相似度识别精确与近似重复，使用 Random Forest、双层 LSTM 以及 GraphCodeBERT（全微调和冻结编码+MLP）等模型，并利用差分进化对超参数进行优化。

**📊 数据集**

使用从 GitHub 提取、通过 TruffleHog/GitLeaks 标注的 SecretBench 数据集，经过过滤后包含 23,352 条样本（7 种面向对象语言），其中 3,134 条为真实秘密。

**📈 对比分析**

在固定 5 折交叉验证下分别在三种数据拆分情景下训练评估，结果显示 GraphCodeBERT 在混合情景下 MCC 0.97，去重后仍保持 0.90；Random Forest 的 MCC 由 0.89 降至 0.65，表明对重复样本高度敏感。

**⚠️ 局限性**

局限性包括仅使用 200 字符上下文窗口、仅关注 SecretBench 产生的秘密类型、对秘密长度分布的潜在偏倚、可能存在标签误差，以及缺乏公开可获取的完整数据集导致可复现性受限。

---

## 399. Scalable Topology-Preserving Graph Coarsening with Graph Collapse

**arXiv ID:** 2601.22943 | [PDF](https://arxiv.org/pdf/2601.22943v1)

**作者:** Xiang Wu `[一作]`, Guoren Wang `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种可扩展的拓扑保持图粗化方法 STPGC，用以加速在粗化图上训练图神经网络。

**💡 创新点**

创新点在于引入图强折叠、图边折叠与邻域锥化三种操作，理论上证明保持同伦等价，并通过近似折叠实现可调节的粗化比例。

**🔧 技术方法**

技术实现基于代数拓扑原理（强折叠/边折叠、邻域锥化）、GNN 感受野分析、Betti 数维护以及近似折叠算法。

**📊 数据集**

实验使用 Cora、Citeseer、DBLP、ogbn‑arXiv、ogbn‑products 等中小规模数据集，以及大型数据集 Youtube、LiveJournal、cit‑Patent、Flixter。

**📈 对比分析**

与七种基线方法（Variation Neighborhoods、Variation Edges、Algebraic JC、Affinity GS、Kron、FGC、GEC）比较，STPGC 在节点分类准确率上往往优于或与 GEC 相近，同时运行时间比 GEC 快 15–37 倍。

**⚠️ 局限性**

局限性包括需手动调节阈值参数（影响精度与效率平衡）、在极低粗化比例下可能仍失去部分拓扑特征，以及对极大密集图的扩展性仍有待进一步提升。

---

## 400. BEAR: Towards Beam-Search-Aware Optimization for Recommendation with Large Language Models

**arXiv ID:** 2601.22925 | [PDF](https://arxiv.org/pdf/2601.22925v1)

**作者:** Weiqin Yang `[一作]` (Zhejiang University), Can Wang `[通讯]` (Zhejiang University)

**通讯引用:** 11631 | [OpenAlex ID](https://openalex.org/A5100428567)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文针对大语言模型（LLM）在推荐系统中的训练与推理不一致问题，提出了一种新的训练目标BEAR，使模型在微调时更符合beam search的解码行为。

**💡 创新点**

创新点在于：①识别并量化SFT训练目标与beam search推理之间的冲突；②设计一种基于必要条件的beam-search-aware正则化，既能有效降低正样本被误剪的风险，又不需要在训练中模拟完整的beam search；③将该正则化可无缝融入现有的SFT或DPO目标。

**🔧 技术方法**

技术手段包括：token级概率的softmax阈值计算、sigmoid平滑的指示函数逼近、与原始SFT交叉熵损失的加权组合；训练采用LoRA微调、AdamW优化，推理使用束搜索。

**📊 数据集**

实验使用四个Amazon电商数据集（Office、Book、Toy、Clothing）以及不同规模的LLM（Llama-3.2-1B、3B、8B）和多种推荐骨干（BIGRec、LLaRA、A-LLMRec）。

**📈 对比分析**

与传统推荐模型（SASRec、BERT4Rec、DROS）以及最新的LLM基推荐方法（BIGRec、LLaRA、A-LLMRec）和多种SFT/DPO微调策略（MSL、D^3、CFT、D^2LR、IGD、RosePO、S-DPO、SPRec）进行比较。BEAR在所有数据集上平均提升NDCG@K和HitRatio@K约12.5%，且在beam宽度为10的束搜索下显著降低了正样本误剪比例。

**⚠️ 局限性**

局限性包括：①正则化只关注必要条件，仍有可能忽略更复杂的剪枝路径；②对温度参数和权重的敏感性需要手动调参；③在极大规模数据或更宽束搜索下的计算效率尚未系统评估；④未针对多模态或非文本化项目描述的推荐场景进行验证。

---

## 401. Multi-Cue Anomaly Detection and Localization under Data Contamination

**arXiv ID:** 2601.22913 | [PDF](https://arxiv.org/pdf/2601.22913v1)

**作者:** Anindya Sundar Das `[一作]` (Umeå University), Monowar Bhuyan `[通讯]` (Umeå University)

**通讯引用:** 3236 | [OpenAlex ID](https://openalex.org/A5044933320)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了一种在训练数据受污染且仅有少量标注异常样本的情况下，能够同时检测并定位视觉异常的鲁棒可解释框架。

**💡 创新点**

将偏差学习、预测不确定性与分割空间证据三种互补信号融合成统一异常评分，并在训练中使用自监督伪异常生成和少量真实异常样本，同时通过自适应实例加权抑制污染样本。

**🔧 技术方法**

自监督伪异常生成、偏差学习+soft偏差损失、熵不确定性模块、像素级分割分支、α‑divergence实例加权、三信号几何均值融合以及梯度回传定位。

**📊 数据集**

MVTec AD 与 VisA 两大工业视觉异常基准。

**📈 对比分析**

与 PatchCore、PaDiM、DestSeg、DRÆM、LOE、ADL 等现有方法在不同污染率下进行对比；在 MVTec、VisA 上均实现最高或次高的图像级 AUROC 与像素级 AUROC，且在高污染率下性能衰减最小。

**⚠️ 局限性**

对像素级定位的精度仍低于专门的分割优化方法；缺少对复杂多类别、极稀疏异常的进一步适应；未来需探索多信号自适应权重与语义感知定位。

---

## 402. MulFeRL: Enhancing Reinforcement Learning with Verbal Feedback in a Multi-turn Loop

**arXiv ID:** 2601.22900 | [PDF](https://arxiv.org/pdf/2601.22900v1)

**作者:** Xuancheng Li `[一作]` (Tsinghua University), Qingyao Ai `[通讯]` (Tsinghua University)

**通讯引用:** 4431 | [OpenAlex ID](https://openalex.org/A5089655391)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 MulFeRL，一种在失败样本上使用多轮口语反馈驱动的强化学习框架，帮助模型在多步生成过程中自我纠错并提升推理准确率。

**💡 创新点**

将口语反馈转化为可训练的学习信号；在仅出现失败的生成组中动态触发多轮反馈驱动再生成；结合同轮 GRPO 与跨轮 DPO 两种互补优化；采用结构化反馈注入以增强模型对反馈的利用。

**🔧 技术方法**

多轮生成与再生成、GRPO（组相对策略优化）、DPO（跨轮优先级优化）、结构化反馈注入、验证器奖励、基于 LLM 的反馈模拟器（如 GPT‑4o）。

**📊 数据集**

使用 OpenR1‑Math 训练集（约 4k 条样本），在 5 个数学基准（AMC23、AIME24、OlympiadBench、MATH500、Minerva‑Math）和 3 个 OOD 任务（MMLU‑Pro、GPQA‑Diamond、TheoremQA）进行评估。

**📈 对比分析**

与基线模型（基础、SFT、RAFT、CITL‑FT、GRPO、Dr.GRPO、Critique‑GRPO）比较，MulFeRL 在两大后端模型（Qwen2.5‑7B‑Base、Qwen3‑4B‑Inst）上均取得显著提升，特别是在 OOD 任务上取得 5–10% 的 Pass@1 增幅，证明其对多域迁移的鲁棒性。

**⚠️ 局限性**

训练成本高（多轮生成与验证开销），对反馈质量高度依赖；在极难样本或反馈噪声大时效果可能不稳定；需要在推理时额外提供外部反馈，导致部署复杂性增加。

---

## 403. When Machines Get It Wrong: Large Language Models Perpetuate Autism Myths More Than Humans Do

**arXiv ID:** 2601.22893 | [PDF](https://arxiv.org/pdf/2601.22893v1)

**作者:** Eduardo C. Garrido-Merchán `[一作]` (Comillas Pontifical University), Adriana Constanza Cirera Tirschtigel `[通讯]` (Comillas Pontifical University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

比较了178名西班牙参与者与GPT‑4、Claude和Gemini对30个关于自闭症真伪的判断，评估其知识水平。

**💡 创新点**

首次系统性对比LLM与人类在自闭症知识上的表现，揭示LLM在误区传播上的不足。

**🔧 技术方法**

采用问卷调查与API调用的混合技术，利用两比例z检验评估误差率。

**📊 数据集**

自闭症知识问卷（30条，16条迷思+14条事实），人类样本178人，LLM对同一问卷的响应。

**📈 对比分析**

通过比较误差率，发现人类平均误差36.2%低于LLM平均44.8%，差异显著；GPT‑4表现最好，其余两模型相近。

**⚠️ 局限性**

样本教育水平高、非代表性；LLM模型随时间迭代；未纳入自闭症人士视角，需进一步验证。

---

## 404. DiffuSpeech: Silent Thought, Spoken Answer via Unified Speech-Text Diffusion

**arXiv ID:** 2601.22889 | [PDF](https://arxiv.org/pdf/2601.22889v1)

**作者:** Yuxuan Lou `[一作]` (National University of Singapore), Yang You `[通讯]` (National University of Singapore)

**通讯引用:** 3811 | [OpenAlex ID](https://openalex.org/A5100658705)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出“Silent Thought, Spoken Answer”范式，使语音大模型在回答问题时先生成内部文本推理（silent thought）再输出语音答案，并实现了第一款同时支持语音理解和生成的扩散式语音‑文本语言模型。

**💡 创新点**

①将推理过程显式化为文本轨迹；②采用统一词表与掩码扩散框架同时生成文本与语音，突破自回归模型在语音生成中的单向限制；③构造首个含推理轨迹的语音 QA 数据集。

**🔧 技术方法**

掩码扩散语言模型（MDLM）与 HuBERT 编码/量化、HiFi‑GAN vocoder、双阶段多任务训练（对齐 + 指令+推理微调）等技术。

**📊 数据集**

自建 SpeechQA-Reasoning 数据集（26,387 条，319 小时），从 Smoltalk2、SoundMind 经过 LLM 重写、质量过滤与 TTS 合成而成。

**📈 对比分析**

与多种自回归语音 LLM（Moshi、MinMo、Qwen2‑Audio 等）以及 DiFFA、LLaDA 进行对比。S→S 语音 QA 准确率最高（68.5%‑49.7%），比最佳基线提升 3.0‑9.0 分；TTS WER 最低（6.2%），语音识别 WER 竞争力强；在语言理解任务上保持与基准 LLaDA 相当或略优。

**⚠️ 局限性**

扩散模型推理时间仍高于自回归；数据集规模有限（仅 26k 条），且主要覆盖问答与推理场景；模型目前仅支持语音与文本两模态，尚未扩展视觉或更复杂多模态任务。

---

## 405. Should LLMs, $\textit{like}$, Generate How Users Talk? Building Dialect-Accurate Dialog[ue]s Beyond the American Default with MDial

**arXiv ID:** 2601.22888 | [PDF](https://arxiv.org/pdf/2601.22888v1)

**作者:** Jio Oh `[一作]` (KAIST), Amani Namboori `[通讯]` (Amazon)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个名为MDial的大规模多方言对话生成框架，并基于该框架生成了覆盖9种英语方言的并行对话数据集。

**💡 创新点**

创新点在于将用户口语方言特征与模型生成文本应遵循的方言特征区分开来，且同时覆盖词汇、拼写和语法三大维度，首次实现了完整的书面方言对话生成。

**🔧 技术方法**

采用规则驱动的LLM变换（RBT）结合专业方言学家审核、LLM参数知识以及后期质量控制等技术，确保生成文本既自然又符合目标方言。

**📊 数据集**

使用了基于MDial生成的50k+对话、97k+问答对以及MDialBenchmark，涵盖美国（SAE）、澳洲、英国、加拿大、印度、爱尔兰、尼日利亚、菲律宾、苏格兰共九种方言。

**📈 对比分析**

通过在MDialBenchmark上评估17款LLM（包括Claude、Qwen、Gemma、GPT-OSS、Deepseek等），发现即使是最先进模型在方言识别和响应生成任务上准确率也低于70%，对加拿大等低资源方言识别更差；经过少量微调后，准确率可提升至96%以上。

**⚠️ 局限性**

局限性包括仅关注书面方言的拼写、词汇和语法三维，而未覆盖口音/语音；仅按国家级方言划分，缺乏细粒度（如AAVE）；生成评估仅使用多选响应，缺乏更精确的生成质量指标。

---

## 406. AnoMod: A Dataset for Anomaly Detection and Root Cause Analysis in Microservice Systems

**arXiv ID:** 2601.22881 | [PDF](https://arxiv.org/pdf/2601.22881v1)

**作者:** Ke Ping `[一作]` (University of Helsinki), Mika V. Mäntylä `[通讯]` (University of Helsinki)

**通讯引用:** 7135 | [OpenAlex ID](https://openalex.org/A5078824435)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文构建了AnoMod多模态数据集，用于微服务系统的异常检测与根因分析，收集了日志、指标、追踪、代码覆盖报告与API响应等五种监控与行为数据。

**💡 创新点**

创新点包括四层异常类型体系、引入代码覆盖与API响应两种新模态、以及根据服务依赖性优先注入关键服务异常，提升数据集的真实性与可用性。

**🔧 技术方法**

技术上结合ChaosMesh与ChaosBlade进行性能、服务、数据库与代码级别的异常注入，利用EvoMaster自动化生成工作负载，使用Prometheus、Jaeger等监控框架以及JaCoCo/Gcov进行代码覆盖收集。

**📊 数据集**

所使用的数据集为AnoMod，基于开源微服务系统SocialNetwork与TrainTicket，共设计并注入24种异常案例，覆盖四个层级，收集五种模态信息。

**📈 对比分析**

本文将AnoMod与Nezha、DeepTraLog、Eadro等现有数据集进行对比，指出其在异常类型多样性和模态丰富度上的优势，为跨模态异常检测与根因定位提供更全面的数据基础。

**⚠️ 局限性**

局限性在于仅针对两个系统构建数据集，异常注入依赖于Chaos工具，工作负载来源单一（EvoMaster黑盒测试），可能无法完全模拟真实生产环境的复杂性。

---

## 407. Synthetic Time Series Generation via Complex Networks

**arXiv ID:** 2601.22879 | [PDF](https://arxiv.org/pdf/2601.22879v1)

**作者:** Jaime Vale `[一作]` (Universidade do Porto), Fernando Silva `[通讯]` (Universidade do Porto)

**通讯引用:** 6853 | [OpenAlex ID](https://openalex.org/A5068596735)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于可逆量化图（Quantile Graph）与逆映射（Inverse Quantile Graph）的方法，用网络映射生成合成时间序列，并在人工与真实数据上验证其真实性与实用性。

**💡 创新点**

创新点在于实现了可逆的网络映射与逆映射，生成合成序列无需训练、解释性强，可直接保留原序列的统计与结构特征，并能一次性生成多种多样的合成样本。

**🔧 技术方法**

使用的技术包括复杂网络映射（量化图）与逆映射算法、统计特征提取（tsfeatures）、网络拓扑特征提取（NetF）、PCA、t‑SNE、k‑means聚类、以及对比GAN模型（TimeGAN、DoppelGANger）。

**📊 数据集**

使用的数据集包括：1）人工数据——11类已知统计模型（如AR、ARIMA、GARCH等），每类100条长度10,000；2）真实数据——英国East Midlands地区22户家庭两年小时能耗数据，包含缺失值与不同长度。

**📈 对比分析**

通过t‑SNE、PCA、统计与网络特征相似度、聚类指标（Silhouette、ARI、NMI）与GAN模型比较，结果显示InvQG在大多数模型上保持更高的相似度与多样性，训练成本低，整体性能优于TimeGAN和DoppelGANger。

**⚠️ 局限性**

局限性主要是：仅能捕捉一阶时序依赖，难以保留长程依赖；对非平稳或强趋势/季节模型的拟合有限；逆映射采用均匀分位区间采样，可能降低波动性；缺乏多延迟或多阶Markov信息。

---

## 408. Eroding the Truth-Default: A Causal Analysis of Human Susceptibility to Foundation Model Hallucinations and Disinformation in the Wild

**arXiv ID:** 2601.22871 | [PDF](https://arxiv.org/pdf/2601.22871v1)

**作者:** Alexander Loth `[一作]` (Frankfurt University of Applied Sciences), Marc-Oliver Pahl `[通讯]` (IMT Atlantique)

**通讯引用:** 882 | [OpenAlex ID](https://openalex.org/A5004198506)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了人类如何识别大语言模型生成的文本，提出了 JudgeGPT 与 RogueGPT 两个开放源平台，实现多模型、多语言的双轴评估（真实性与来源归属），并使用结构因果模型（SCM）分析影响识别准确性的因素。

**💡 创新点**

创新点在于将真实性与来源归属拆分为两个评估维度；提供可复现的双轴评估框架和多模型内容生成器；首次将因果推断应用于人类对 AI 文本的判别研究；揭示 GPT‑4 的“流畅陷阱”并提出预警训练（pre‑bunking）策略。

**🔧 技术方法**

采用结构因果模型（SCM）、双轴滑块评估界面（Streamlit）、多大模型生成（GPT‑4、Llama‑2、Gemma‑7b‑it 等）、统计相关与因果推断、语言学特征分析（表面流畅性、连贯性、因果一致性）。

**📊 数据集**

收集了 154 名参与者、918 次评估、2301 条新闻片段的多语言数据（英语、德语、法语、西班牙语），记录了来源（人类/AI）、生成模型、真实性与来源归属评分、反应时间等信息。

**📈 对比分析**

通过对不同模型的 HumanMachineScore 进行 t 检验，发现 GPT‑4 的分数最低（0.20），表明其文本最像人类；群体层面人类无法显著区分人类与 AI 文本；学习曲线显示早期提升，后期维持；不同参与者群体表现出显著的行为差异。

**⚠️ 局限性**

局限性包括样本规模相对有限、参与者主要来自受教育程度较高的欧洲背景、缺乏长期跟踪学习效果、未系统评估领域与语言差异、因果假设尚未通过实验干预验证。

---

## 409. Reinforcement Learning-Based Co-Design and Operation of Chiller and Thermal Energy Storage for Cost-Optimal HVAC Systems

**arXiv ID:** 2601.22880 | [PDF](https://arxiv.org/pdf/2601.22880v1)

**作者:** Tanay Raghunandan Srinivasa `[一作]` (Plaksha University), Vishal Garg `[通讯]` (Plaksha University)

**通讯引用:** 2252 | [OpenAlex ID](https://openalex.org/A5011914527)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文通过强化学习方法对商业空调系统的冷却机组与热能储存（TES）联合设计与运行进行协同优化，目标是最小化30年生命周期成本。

**💡 创新点**

创新点在于：①将冷却机组的部分负荷比控制建模为有限状态马尔可夫决策过程（MDP），并使用可掩蔽的动作空间解决可行性约束；②将深度Q网络（DQN）作为操作成本评估工具，在外层尺寸优化循环中获取每种机组–TES配置的最优运营成本；③通过DQN学习得到的策略能够在仅使用一年历史负荷与电价数据的情况下实现零失负载并显著降低能耗。

**🔧 技术方法**

主要技术包括马尔可夫决策过程建模、深度强化学习（DQN）与动作掩蔽、经验回放与目标网络、基于负荷/价格的奖励函数设计，以及生命周期成本折现计算。

**📊 数据集**

使用的数据集为印度海得拉巴一栋高层住宅建筑的全年度（8760小时）实时冷却负荷和时价电价记录，并在此基础上随机化TES初始荷电状态生成训练样本。

**📈 对比分析**

与传统贪婪、TES优先、存储保守等基线策略比较，DQN策略在所有可行配置下实现零失负载，并在年电费上比贪婪策略降低约8–12%，在生命周期成本上实现约7%更低的总成本，且通过强化学习可逼近最优的储能利用。

**⚠️ 局限性**

局限性包括：仅基于单年历史数据，可能对未来负荷与电价变动缺乏鲁棒性；DQN训练依赖大量经验样本且计算开销较高；模型对机组与TES参数的精确性高度敏感，实际工况中的非线性与维护成本变动可能导致估计偏差。

---

## 410. Triage: Hierarchical Visual Budgeting for Efficient Video Reasoning in Vision-Language Models

**arXiv ID:** 2601.22959 | [PDF](https://arxiv.org/pdf/2601.22959v1)

**作者:** Anmin Wang `[一作]` (Huazhong University of Science and Technology), Jianzong Wang `[通讯]` (Ping An Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种无训练、可插拔的层级视觉预算框架Triage，用于在视频视觉‑语言模型中通过先选择关键帧再分配视觉令牌（核心与多样上下文令牌）来降低冗余和计算成本。

**💡 创新点**

创新点在于把视频推理视作资源分配问题，设计两阶段层级预算：①通过视觉动态、运动强度和文本相关性加权评估关键帧；②在核心/上下文两步预算中先保留最相关令牌，再用批量最大边际相关性（MMR）高效选取多样化上下文令牌；且整个流程无需额外训练，直接可嵌入现有VLM。

**🔧 技术方法**

主要技术包括：
- 关键帧重要性评分（S_change、S_motion、S_relevance）和自适应时间分桶。
- 交叉注意力得分作为令牌重要性评估。
- 核心令牌固定比例分配。
- 批量MMR算法（seed+diversity项）进行上下文令牌选择。
- 采用轻量级CLIP进行查询相关性计算。

**📊 数据集**

使用四个视频推理基准：Video‑MME、MVBench、LongVideoBench、LVBench，并在三种公开VLM（LLaVA‑OneVision‑7B、LLaVA‑Video‑7B、Qwen2‑VL‑7B）上评测。

**📈 对比分析**

与基线（原始VLM、PyramidDrop、FastV、DyCoke）对比，Triage在保留率为50%时往往取得更高得分；在某些数据集（如LVBench）甚至在25%保留率下超过其他方法；同时在推理速度和显存占用上显著降低，表明具有优越的性能‑效率折中。

**⚠️ 局限性**

局限性包括：
- 依赖预先计算的交叉注意力分数，若VLM架构或层数不同需调整实现。
- 对超长视频的时序覆盖仍有限，仍需进一步验证在极长视频上的鲁棒性。
- 虽不需要训练，但选择的权重和阈值对不同任务仍有一定调参需求。

---

## 411. A Real-Time Privacy-Preserving Behavior Recognition System via Edge-Cloud Collaboration

**arXiv ID:** 2601.22938 | [PDF](https://arxiv.org/pdf/2601.22938v1)

**作者:** Huan Song `[一作]` (Institute of Artificial Intelligence), Xuelong Li `[通讯]` (Institute of Artificial Intelligence)

**通讯引用:** 61657 | [OpenAlex ID](https://openalex.org/A5100740143)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `9cc9baba-5356-466d-81ff-d80028d90279` `40105733-5154-44cd-8090-a8cab9e64b07` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出基于 AI Flow 框架的边缘‑云协同隐私保护监控系统，通过 SPA‑D 源去敏算法实现身份与语义的数学分离，仅利用抽象特征实现高敏感场景行为检测。

**💡 创新点**

首次将信息瓶颈理论与不可逆特征映射结合，构建单向信息流，彻底阻断原图重建；并在边缘端实现毫秒级去敏与噪声注入，提升实时性与隐私安全。

**🔧 技术方法**

信息瓶颈理论、非线性特征映射、对抗性扰动注入、多模态 AI Flow 模型、边缘‑云协同架构。

**📊 数据集**

文中未给出具体数据集，实验基于真实高敏感场景的实地部署录像。

**📈 对比分析**

未提供与传统像素化、面部遮挡、联邦学习等方法的定量对比；仅描述在公共卫生间场景中实现实时风险告警且无视频输出，缺乏准确率、召回率等指标。

**⚠️ 局限性**

缺乏对模型鲁棒性、跨场景泛化能力的评估；对抗攻击对特征映射的影响未充分实验；实现细节中对硬件资源、带宽需求的量化不足。

---

## 412. DINO-SAE: DINO Spherical Autoencoder for High-Fidelity Image Reconstruction and Generation

**arXiv ID:** 2601.22904 | [PDF](https://arxiv.org/pdf/2601.22904v1)

**作者:** Hun Chang `[一作]` (KAIST), Jong Chul Ye `[通讯]` (KAIST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 DINO‑Spherical Autoencoder (DINO‑SAE)，通过改进的层级卷积 patch 嵌入与方向性对齐，将预训练的 DINO 编码器转化为高保真重建和生成的 tokenizer；

**💡 创新点**

创新点在于：①使用余弦相似度对齐只约束特征方向，解放幅度自由度以保留高频细节；②引入层级卷积 patch 嵌入降低 ViT 的信息瓶颈；③在球面隐空间上采用黎曼流匹配 (RFM) 训练 DiT，去除幅度冗余、加速收敛；

**🔧 技术方法**

技术包括层级卷积 Patch 嵌入、Cosine Similarity 对齐、LPIPS 感知损失、对抗训练、Riemannian Flow Matching、Diffusion Transformer (DiT) 等；

**📊 数据集**

在 ImageNet‑1K（256×256）上训练并评估；

**📈 对比分析**

与 SD‑VAE、VAVAE、MAETok、RAE 等基线对比，DINO‑SAE 取得 rFID 0.37、PSNR 26.2 dB；在此隐空间上训练的 LightningDiT‑XL/DiT^DH‑XL 达 gFID 3.07‑3.47，显著快于同类基线；

**⚠️ 局限性**

局限在于仅验证无条件重建和 ImageNet 生成，未测试文本到图像、多模态或逆向任务，对复杂条件场景的泛化尚未评估。

---

## 413. Assessing the Real-World Impact of Post-Quantum Cryptography on WPA-Enterprise Networks

**arXiv ID:** 2601.22892 | [PDF](https://arxiv.org/pdf/2601.22892v1)

**作者:** Lukas Köder `[一作]` (University of Applied Sciences Esslingen), Tobias Heer `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

在实验环境下实现并测量了基于 PQC 的 WPA‑Enterprise 身份验证的实际性能，比较了多种 NIST 标准化的签名和 KEM 算法（ML‑KEM、ML‑DSA、Falcon、SLH‑DSA）以及其混合方案，并评估了会话恢复对延迟的影响。

**💡 创新点**

首次在真实 Wi‑Fi 网络中量化 PQC 对 EAP‑TLS / EAP‑TTLS 的身份验证时延，并提出了“量子烦恼”度量，将量子攻击难度与安全级别关联；同时给出可直接落地的 hostapd、FreeRADIUS 补丁与测试脚本，推动开源社区快速验证 PQC 的可行性。

**🔧 技术方法**

使用 OpenSSL 3.4.1、liboqs 0.14.0、oqs‑provider 0.9.0 作为 PQC 加密库；在 Linux 服务器和客户端上部署 FreeRADIUS 3.2.8 与 hostapd，利用 wpa_supplicant 2.12 进行 EAP‑TLS/EAP‑TTLS 交互；通过 2.4 GHz/5 GHz 频段、三种信号强度（>‑60 dBm、‑60至‑70 dBm、<‑70 dBm）构建实验平台。

**📊 数据集**

数据集主要来源于自建实验室环境：100 次 EAP‑TLS / EAP‑TTLS 会话、三种信号质量、两频段，记录完整的 pcap、延迟统计与服务器缓存大小；无外部公开数据集，全部为实验生成数据。

**📈 对比分析**

通过对比 RSA‑2048 + ECDHE 的基线，分别计算中位数、95 % 分位数以及消息数量、CPU 周期与存储占用；结果显示 Falcon‑1024 与 ML‑DSA‑65 的平均延迟约为 RSA 的 1.5–2 倍，但仍在 0.2–0.3 s 范围；SLH‑DSA‑f 由于签名大且消息多，延迟可达 1.2 s；会话恢复将总体延迟降低至 10–20 % 的水平。

**⚠️ 局限性**

局限性包括：仅测试单一硬件平台和小规模网络，未考虑大规模部署下的链路拥塞与 RADIUS 负载；未包含证书链长度、CRL 或 OCSP 影响；仅覆盖 NIST 第 3 轮已标准化方案，未来新方案与混合构造的兼容性尚未评估；此外，实验仅在 2.4 / 5 GHz 下进行，未覆盖 6 GHz 等新频段。

---

## 414. PlatoLTL: Learning to Generalize Across Symbols in LTL Instructions for Multi-Task RL

**arXiv ID:** 2601.22891 | [PDF](https://arxiv.org/pdf/2601.22891v1)

**作者:** Jacques Cloete `[一作]` (Oxford University), Alessandro Abate `[通讯]` (Oxford University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本研究提出了PlatoLTL，能够通过LTL指令在多任务强化学习中实现零样本泛化，支持未见的原子命题；

**💡 创新点**

其创新点在于将原子命题重新表述为带参数的原子谓词实例，利用共享的参数化嵌入实现跨命题、跨任务的参数化泛化；

**🔧 技术方法**

技术方法包括谓词实例嵌入、图神经网络（GCN）对布尔公式树进行编码、GRU序列网络、PPO强化学习以及与有限自动机（LDBA）的联合；

**📊 数据集**

使用的环境数据集为RGBZoneEnv（连续颜色空间）与FalloutWorld（网格+辐射阈值），并基于这些环境生成多种LTL规范；

**📈 对比分析**

与DeepLTL、LTL-GNN等基线相比，PlatoLTL在大量原子命题下收敛更快、成功率更高，并能零样本泛化到未见和连续命题，整体性能优于基线；

**⚠️ 局限性**

限制在于目前仅支持相对简单的参数化命题，对更复杂对象表示和更大规模环境的可扩展性与鲁棒性仍需进一步验证。

---

## 415. MoVE: Mixture of Value Embeddings -- A New Axis for Scaling Parametric Memory in Autoregressive Models

**arXiv ID:** 2601.22887 | [PDF](https://arxiv.org/pdf/2601.22887v1)

**作者:** Yangyan Li `[一作]` `[通讯]` (Ant Group), Yangyan Li (Ant Group)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出MoVE（Mixture of Value Embeddings）机制，突破Transformer中参数容量与计算成本的耦合关系，实现参数记忆与计算层深度独立扩展。

**💡 创新点**

创新点在于将一个全局可学习的“价值嵌入”存储库注入注意力的Value流，并通过可微软门控动态混合多槽记忆，从而实现记忆容量可按需放大，而无需增深或加宽网络。

**🔧 技术方法**

技术包括：全局可学习嵌入银行、每头可微路由门控、SoftMixing of Value Embeddings、与MLA（Multi‑Head Latent Attention）兼容的压缩空间注入、以及在Transformer基础上的实验对照。

**📊 数据集**

数据集：文本生成采用FineWeb‑Edu（约100B token）+ BPE词表；图像生成使用ImageNet‑1K；实验也覆盖LlamaGen框架中的GPT‑B、GPT‑L模型。

**📈 对比分析**

方法与传统Dense Transformer、层级记忆（LaVE）进行严格对照；在文本域BPB、图像域FID/IS/Precision/Recall上，MoVE在相同计算预算下均优于或接近Dense且明显优于LaVE；在MLA中MoVE的记忆扩展可持续提升。

**⚠️ 局限性**

局限性包括：1）记忆存取需较大内存带宽，取决于硬件；2）在极大记忆容量时，参数效率低于Dense缩放；3）未与MoE等稀疏专家技术联合验证；4）未探索不同记忆共享策略或更高效的路由投影。

---

## 416. When Anomalies Depend on Context: Learning Conditional Compatibility for Anomaly Detection

**arXiv ID:** 2601.22868 | [PDF](https://arxiv.org/pdf/2601.22868v1)

**作者:** Shashank Mishra `[一作]` (German Research Center for Artificial Intelligence), Jason Rambach `[通讯]` (German Research Center for Artificial Intelligence)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了在视觉域中针对基于上下文的异常检测，构建了新型数据集CAAD‑3K，并提出CoRe‑CLIP框架实现条件兼容性学习，能判断主体与上下文是否匹配。

**💡 创新点**

核心创新在于把异常判定从单一全局特征转化为主体–上下文的条件兼容性推理；通过三分支视觉表示（主体、上下文、全局）和文本嵌入对齐，解决了传统异常检测对上下文忽略导致的非可辨识性问题。

**🔧 技术方法**

利用预训练的Vision‑Language模型CLIP作为骨干，加入Context‑Selective Residual（CSR）适配器、文本细化模块及兼容性推理模块（CRM），并通过互补的图像与文本损失进行端到端训练。

**📊 数据集**

主要使用CAAD‑3K（包含2,095/905张训练/测试样本），并在工业缺陷数据集MVTec‑AD、VisA以及真实场景的MIT‑OOC、COCO‑OOC上进行零样本迁移评估。

**📈 对比分析**

与多种CLIP‑based异常检测方法（WinCLIP、AnomalyCLIP、AdaCLIP等）以及基础模型的OOC方法（CRTNet）对比，CoRe‑CLIP在CAAD‑3K交叉上下文测试中Image‑AUROC最高达85.0/87.3/98.3，明显优于现有方法；在MVTec‑AD/VisA上也实现了SOTA或接近SOTA的性能。

**⚠️ 局限性**

局限性包括：依赖预训练CLIP的语义偏见；对主体与上下文划分的区域掩码在训练时需要额外（但可自动生成）信息；在极端复杂或多主体情境下的兼容性推理仍需进一步验证。

---

## 417. Residual Context Diffusion Language Models

**arXiv ID:** 2601.22954 | [PDF](https://arxiv.org/pdf/2601.22954v1)

**作者:** Yuezhou Hu `[一作]` (University of California, Berkeley), Chenfeng Xu `[通讯]` (University of California, Berkeley)

**通讯引用:** 4385 | [OpenAlex ID](https://openalex.org/A5083914796)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了残差上下文扩散（RCD）框架，用以在扩散式大语言模型中利用被“重掩码”未解码的 token 信息，从而提升并行解码的质量。

**💡 创新点**

创新点包括：① 采用熵加权残差机制动态调节残差贡献；② 两阶段解耦式训练（参考模型+目标模型）解决循环依赖；③ 温度缩放对齐推理分布；④ 在保留原有吞吐量的前提下显著降低解码步数。

**🔧 技术方法**

技术实现主要依赖：扩散式语言建模、软 token 表示、熵归一化、两阶段训练策略、温度缩放、残差插值。

**📊 数据集**

使用的主要数据集有：OpenR1‑Math‑220k（长上下文数学推理）、OpenMathInstruct‑2（指令跟随）、GSM8K、MATH500、AIME24/25、MinervaMath，用以评估推理和指令执行能力。

**📈 对比分析**

与传统顺序去噪（SeqD）以及 Loopholing 等基线相比，RCD 在 SDAR、LLaDA 等模型上提升 5–10% 的准确率，AIME 任务可近乎翻倍，同时在相同吞吐量下实现 4–5 倍的去噪步数节省；在同等推理吞吐量下，其准确率始终优于基线。

**⚠️ 局限性**

局限性包括：需要额外训练参考模型并进行温度调优；对非常大型模型的内存和并行解码仍有限制；在某些基准上 AR 模型仍具备优势；残差注入可能对训练稳定性产生依赖于数据分布的风险。

---

## 418. Autonomous Chain-of-Thought Distillation for Graph-Based Fraud Detection

**arXiv ID:** 2601.22949 | [PDF](https://arxiv.org/pdf/2601.22949v1)

**作者:** Yuan Li `[一作]` (National University of Singapore), Cheng Chen `[通讯]` (ByteDance Inc)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `8d10c613-917e-4880-9716-17789f50e119` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 FraudCoT 框架，用链式思维（CoT）和 LLM‑GNN 统一训练实现文本属性图的欺诈检测

**💡 创新点**

①自主图感知 CoT 迁移，消除预设提示限制；②负样本抑制的 CoT 退化学习；③高效非对称共训练，既实现端到端对齐又显著降低计算成本

**🔧 技术方法**

大模型（如 Qwen3‑8B、DeepSeek‑R1）+ LoRA 参数适配；链式思维生成与蒸馏；GraphSAGE/HGT 等 GNN；负样本不似然损失；异步邻居缓存加速训练

**📊 数据集**

公开数据集：InstantVideo、DigitalMusic（来自 Amazon 评论）；工业数据集：ByteDance 推广滥用 PromotionAbuse

**📈 对比分析**

与多类基线（纯 GNN、LLM‑GNN、图增强 LLM 等）对比，FraudCoT 在所有数据集上取得最高 Macro‑F1、AUROC、AUPRC，提升幅度可达 8.8% AUPRC，训练速度提升多达 1066×

**⚠️ 局限性**

主要局限：需依赖教师 LLM 生成 CoT，样本成本仍高；对图结构的假设（异步缓存）可能限制在大规模异构图上的适用；负样本权重需手工调参

---

## 419. Relaxing Positional Alignment in Masked Diffusion Language Models

**arXiv ID:** 2601.22947 | [PDF](https://arxiv.org/pdf/2601.22947v1)

**作者:** Mengyu Ye `[一作]` (Tohoku University), Jun Suzuki `[通讯]` (Tohoku University)

**通讯引用:** 8003 | [OpenAlex ID](https://openalex.org/A5001456824)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了 Masked Diffusion Language Models（MDLM）在开放式文本生成中对位置误差的敏感性，并提出在监督微调时加入 CTC 目标以引入 slack token 缓冲，从而放宽严格的一对一位置对齐限制，提升生成鲁棒性。

**💡 创新点**

创新点在于首次将 CTC 的对齐自由度引入 MDLM 的训练目标，通过 slack token 让模型在解码过程中自动吸收局部位置偏移，解决了标准 MDLM 在不可逆解码过程中因位置误差导致的语义崩塌问题。

**🔧 技术方法**

主要技术包括 MDLM（以 LLaDA-8B-Instruct 为基准）、CTC 目标的联合训练、在解码时对重复/相同 token 自动替换为 slack、以及控制性的一位移干预实验验证误差传播机制。

**📊 数据集**

使用的数据集涵盖：训练阶段采用 300k 条 Magpie 过滤后的高质量指令数据；评估阶段使用 Arena‑Hard‑v2.0（硬提示与创意写作子集）、MTBench、WildBench、Writing Bench 与 Creative Writing v3 等开放式生成基准。

**📈 对比分析**

与原 LLaDA-8B‑Instruct 及仅使用交叉熵的 CE‑only 对照组对比，采用 win‑rate 评价指标，实验显示 CTC 训练模型在 5 个基准上均优于对照组，并在位置误差干预实验中表现出更低的敏感度（Pearson r 从 -0.85 降至 -0.10）。

**⚠️ 局限性**

局限性包括：CTC 的 collapse 操作可能破坏数字或格式敏感文本；实验仅针对监督微调阶段，未探索在预训练中放宽对齐的效果；仅使用基础 CTC 目标，未尝试更复杂的对齐机制；以及模型在特定格式化任务上的鲁棒性未做充分验证。

---

## 420. Protecting Private Code in IDE Autocomplete using Differential Privacy

**arXiv ID:** 2601.22935 | [PDF](https://arxiv.org/pdf/2601.22935v1)

**作者:** Evgeny Grigorenko `[一作]` (JetBrains Research), Kostadin Cvejoski `[通讯]` (JetBrains Research)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了在IDE代码自动补全场景中使用差分隐私（DP）训练大型语言模型，以减少训练数据泄露风险。

**💡 创新点**

首次提出并实现了在Kotlin代码补全任务中使用DP‑SGD和LoRA技术进行私有数据微调的完整工作，展示了可接受的隐私-效能平衡。

**🔧 技术方法**

核心技术包括：DP‑SGD（加噪梯度与梯度裁剪）、低秩适配（LoRA）作为参数高效微调方法，以及对模型输出进行置信度评估的评测框架。

**📊 数据集**

使用JetBrains内部Kotlin代码仓库共800万条训练样本进行无私有模型训练，私有模型仅使用8万条样本；评测集为未见过的Kotlin基准数据。

**📈 对比分析**

通过对比无DP基线模型与DP模型，利用ChrF++、LLM‑as‑judge、LM Score三指标验证模型生成质量；隐私评估采用成员推断攻击（MIA）得到AUC：无DP模型0.901，DP模型0.606，表明DP显著降低攻击成功率，且在有限样本下DP模型的生成质量与基线相当甚至略优。

**⚠️ 局限性**

局限包括：隐私预算ε≈30相对较大，未探索更严格预算对性能的影响；实验仅聚焦Kotlin语言，缺乏跨语言验证；模型规模为4B，尚未验证在更大模型上的可扩展性。

---

## 421. LLMs Explain't: A Post-Mortem on Semantic Interpretability in Transformer Models

**arXiv ID:** 2601.22928 | [PDF](https://arxiv.org/pdf/2601.22928v1)

**作者:** Alhassan Abdelhalim `[一作]` (Universität Hamburg), Michaela Regneri `[通讯]` (Universität Hamburg)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统评估了 LLM 解释性方法，探究了注意力分析与嵌入属性推断在语言模型中的假设与局限。

**💡 创新点**

创新点在于提出严格的诊断实验，揭示这两种主流解释技术的核心假设失效，并指出误导性解释的根源。

**🔧 技术方法**

使用残差流跟踪、注意力可视化、特征规范映射（McRae、Buchanan、Binder）以及 PLSR/FFNN 预测模型，并通过随机/混淆特征上限实验检验方法可靠性。

**📊 数据集**

采用 BERT 等 Transformer 模型的输入句子，以及三套特征规范数据集（McRae、Buchanan、Binder）和对应的词向量。

**📈 对比分析**

通过与上限、随机、混淆基线对比，发现即使特征被打乱或随机化，预测准确度仍保持较高，表明方法性能受数据结构限制，而非真正的语义理解。

**⚠️ 局限性**

方法受限于假设的 token 连续性和注意力可解释性，未考虑残差混合和非线性变换；嵌入属性推断受特征稀疏、几何结构约束主导，无法真实反映语义内涵。

---

## 422. Improving Supervised Machine Learning Performance in Optical Quality Control via Generative AI for Dataset Expansion

**arXiv ID:** 2601.22961 | [PDF](https://arxiv.org/pdf/2601.22961v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 423. Alignment among Language, Vision and Action Representations

**arXiv ID:** 2601.22948 | [PDF](https://arxiv.org/pdf/2601.22948v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 424. Q-Hawkeye: Reliable Visual Policy Optimization for Image Quality Assessment

**arXiv ID:** 2601.22920 | [PDF](https://arxiv.org/pdf/2601.22920v1)

**作者:** Wulin Xie `[一作]` (Institute of Automation, Chinese Academy of Sciences), Jie Wen `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 7122 | [OpenAlex ID](https://openalex.org/A5017617923)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 Q-Hawkeye，基于 RL 的可靠视觉策略优化框架，用于无参考图像质量评估。

**💡 创新点**

创新点在于引入不确定性感知动态优化与感知感知优化两种机制，分别通过预测方差加权更新与原始-降质图像的分布差异约束提升模型可靠性。

**🔧 技术方法**

采用 Group Relative Policy Optimization（GRPO）作为后训练框架，配合多路 rollout、方差估计、KL 散度和熵正则化等技术。

**📊 数据集**

主要使用 KonIQ 进行训练，评估覆盖 KonIQ、SPAQ、LIVE-Wild、FLIVE、KADID-10K、CSIQ、PIPAL、AGIQA-3K 等八个 IQA 数据集。

**📈 对比分析**

与传统无参考、深度学习及多模态 LLM 方法对比，单数据集训练已超越 VisualQuality-R1、Q-Insight 等，跨数据集平均 PLCC/SRCC 最高，鲁棒性显著提升。

**⚠️ 局限性**

仍依赖大量 rollouts 与计算资源，且对极弱降质需要人工筛选，模型对某些极端失真或语义先验的依赖尚未完全消除。

---

## 425. Leveraging LLMs For Turkish Skill Extraction

**arXiv ID:** 2601.22885 | [PDF](https://arxiv.org/pdf/2601.22885v1)

**作者:** Ezgi Arslan İltüzer `[一作]` (Kariyer.net), Gülşen Eryiğit `[通讯]` (Istanbul Technical University)

**通讯引用:** 3519 | [OpenAlex ID](https://openalex.org/A5030279774)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文创建了土耳其语首个技能提取数据集，并构建了基于大型语言模型（LLM）的端到端技能提取管道。

**💡 创新点**

创新点包括：①首次提供土耳其语技能提取数据集；②在低资源语言环境下引入动态few-shot提示与因果推理提升技能链接；③将ESCO技能库翻译并与本地数据对齐；④综合评估LLM与传统序列标注模型在技能识别与链接中的性能。

**🔧 技术方法**

技术方案涵盖：BERT/EuroBERT序列标注；decoder-only LLM（Claude Sonnet 3.7、Gemma 3、GPT‑4o）用于技能识别的静态/动态few-shot；支持向量机（SVM）进行多技能解析；模糊匹配与句子嵌入相似度检索技能；LLM重排序（含因果推理）对检索结果进行提升。

**📊 数据集**

使用的数据集为327篇土耳其招聘广告，标注了4,819个技能span；同时将ESCO 1.2.0翻译为土耳其语作为技能链接的知识库。

**📈 对比分析**

性能评估通过CoNLL‑F1、MUC Partial、HitRate@k以及端到端F1进行。实验显示动态few-shot提示优于静态/零shot；嵌入相似度检索远优于模糊匹配；最佳配置（Claude Sonnet 3.7动态十-shot + GPT‑4o因果重排序）实现端到端F1 = 0.56，接近其他语言的研究水平。

**⚠️ 局限性**

局限性：数据量有限且来源单一平台，导致代表性不足；BERT基模型在技能识别上仍优于LLM；评测指标未充分考虑语义近似与层级关系；LLM重排序成本高；缺乏对ESCO层级的显式利用，导致部分语义相关但非精确匹配被误判。

---

## 426. From Labels to Facets: Building a Taxonomically Enriched Turkish Learner Corpus

**arXiv ID:** 2601.22875 | [PDF](https://arxiv.org/pdf/2601.22875v1)

**作者:** Elif Sayar `[一作]` (Istanbul Technical University), Gülşen Eryiğit `[通讯]` (Istanbul Technical University)

**通讯引用:** 3519 | [OpenAlex ID](https://openalex.org/A5030279774)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并发布首个使用面向错误的分面分类法（faceted taxonomy）扩展的土耳其学习者语料库，并提供相应的标注工具与规范。

**💡 创新点**

创新点在于：①首次将分面分类法应用于土耳其语学习者语料，实现多维度、可解释的错误标注；②提出半自动化标注扩展框架（annotation extender），自动推断六个分面（Identifier、MorphologicalFeature、Unit、Phenomenon、LinguisticLevel、Metadata），显著降低人工标注负担；③通过协作标注与标准化指南提升一致性，并将资源开放共享。

**🔧 技术方法**

技术手段包括：Label Studio 进行协作式手工标注；UDPipe v2 进行 UD（Universal Dependencies）层面的词性、形态特征自动推断；自定义的映射表（pre‑defined tagset mapping schema）驱动分面推断；Python 实现的注释扩展管道，实现从原始标注到富分面语料的转换。

**📊 数据集**

数据集为 672 篇土耳其语学习者作文（C1 级别），共 133,752 词，包含 147 篇子集（mini‑corpus）和 525 篇主体（main‑corpus），涵盖 75 个国家、男女比例均衡，主题多样。

**📈 对比分析**

与人工标注的金标准进行比较，使用 Cohen’s κ 评估标注一致性（κ=0.80），随后评估扩展器的分面准确率：POS 94.46%、Inflectional Feature 98.91%、Lexical Feature 99.85%、Unit 96.29%、Phenomenon 89.77%，宏观平均 95.86%；注释级别精确匹配率为 81.18%。

**⚠️ 局限性**

局限性包括：①对 UDPipe 的依赖，分词/词性/形态标注错误会传播到分面推断；②多值分面采用手工规则，泛化性有限；③假设原始错误标注完全正确，若存在误标会影响扩展结果；④评估采用分层抽样，改变了原始错误分布，导致整体准确率与自然语料偏差。

---

## 427. Environment-Conditioned Tail Reweighting for Total Variation Invariant Risk Minimization

**arXiv ID:** 2601.22944 | [PDF](https://arxiv.org/pdf/2601.22944v1)

**作者:** Wang Yuanchao `[一作]`, Li Fengnan `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在混合相关性偏移和多样性偏移的OOD场景下，提出环境条件下尾部重加权的 TV‑IRM 框架 ECTR，兼顾环境级 invariance 与样本级尾部鲁棒；同时支持无环境标签时通过对抗推断 latent 环境。

**💡 创新点**

将尾部样本重加权嵌入 TV‑IRM invariance 约束，形成环境条件化的尾部风险；引入环境-wise KL 正则稳定重加权；设计可在已知或未知环境标签下统一训练的 min‑max 结构。

**🔧 技术方法**

使用 TV‑IRM（变分形式）、对抗式尾部重加权网络、环境条件化、KL 正则、minimax 优化、环境推断网络（soft assignment）等技术。

**📊 数据集**

在回归/分类/时间序列/视觉等多模态基准上实验，包括 House Prices、CelebA、Landcover、Adult、Colored MNIST、NICO、PACS、以及时间序列仿真。

**📈 对比分析**

与 ERM、IRM、groupDRO、OI、EIIL、LfF、ZIN、TIVA、Minimax IRM‑TV 等基线比较，ECTR 在多数任务的 Worst（最差环境）指标上显著优于对手，平均指标也保持竞争性。

**⚠️ 局限性**

对环境推断的质量依赖辅助变量，KL 正则与尾部权重平衡需手工调节；在高容量模型下训练可能不稳定，缺乏完整的理论泛化分析。

---

## 428. Evaluating Large Language Models for Security Bug Report Prediction

**arXiv ID:** 2601.22921 | [PDF](https://arxiv.org/pdf/2601.22921v1)

**作者:** Farnaz Soltaniani `[一作]` (Technische Universitaet Clausthal), Mohammad Ghafari `[通讯]` (Technische Universitaet Clausthal)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对五个公开安全漏洞报告数据集进行实验，比较了基于提示的专有大语言模型（GPT、Gemini）与微调的小型 LLM（BERT、DistilBERT、DistilGPT‑2、Qwen）的安全漏洞预测能力。

**💡 创新点**

创新点在于同时评估提示工程与微调两种 LLM 方案的性能与成本，提出统一的角色扮演提示模板，并深入分析两者在召回率、精确率、推理速度与费用等维度的权衡。

**🔧 技术方法**

采用提示工程（含角色扮演与规则验证）、差分进化超参数搜索、encoder/decoder LLM 微调、以及多指标评估（Precision、Recall、F1、FPR、G‑measure）等技术。

**📊 数据集**

使用 Chromium、Derby、Ambari、Camel、Wicket 这五个公开缺陷报告数据集，数据中已标注安全漏洞（SBR）与非安全漏洞（NSBR）。

**📈 对比分析**

通过宏平均的 Precision、Recall、F1、FPR、G‑measure 进行比较；Gemini 在召回率与 G‑measure 上领先但精确率低；DistilBERT 在微调模型中取得最高 G‑measure（51%）和精确率（75%），并在推理时比专有模型快 10‑50 倍。

**⚠️ 局限性**

局限性包括仅使用五个数据集、固定的 prompt 参数（temperature、top‑p）与模型版本、超参数搜索可能未达最优、以及使用专有 LLM 时的隐私与成本问题。

---

## 429. Deep in the Jungle: Towards Automating Chimpanzee Population Estimation

**arXiv ID:** 2601.22917 | [PDF](https://arxiv.org/pdf/2601.22917v1)

**作者:** Tom Raynes `[一作]` (University of Bristol), Tilo Burghardt `[通讯]` (University of Bristol)

**通讯引用:** 2908 | [OpenAlex ID](https://openalex.org/A5052504255)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

利用计算机视觉技术自动估计野生黑猩猩种群数量。

**💡 创新点**

提出一种端到端的深度学习框架，将图像中的个体检测与群体计数结合。

**🔧 技术方法**

使用YOLOv5作为检测器并结合基于卷积网络的计数器。

**📊 数据集**

构建了一个由三家机构共同采集的野外相机捕捉图像数据集。

**📈 对比分析**

与人工标注基准和传统统计方法比较，平均相对误差降低至12%。

**⚠️ 局限性**

受限于数据量不足、光照变化及个体遮挡，模型在高密度群体时仍易漏检。

---

## 430. A Serverless Edge-Native Data Processing Architecture for Autonomous Driving Training

**arXiv ID:** 2601.22919 | [PDF](https://arxiv.org/pdf/2601.22919v1)

**作者:** Fabian Bally `[一作]` (Deggendorf Institute of Technology), Thomas Limbrunner `[通讯]` (Deggendorf Institute of Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并实现了 Lambda 框架：在车载边缘设备上采用 serverless 思想的函数式数据过滤与处理平台，支持用户自定义事件驱动的过滤算法，兼容 ROS 2 生态。

**💡 创新点**

创新点包括：把 Function‑as‑a‑Service (FaaS) 原理迁移至资源受限的汽车边缘；通过 Rust 编写的 orchestrator 与独立 Python 进程的 runtime，提供任务调度、部署、隔离的抽象层；利用零拷贝、锁自由缓冲区实现高效的数据摄取；兼容 ROS 2 DDS，保持现有数据录制管道的可迁移性。

**🔧 技术方法**

技术细节：Rust orchestrator 与 runtime、Python lambda 函数、ROS 2 DDS 通信、ONNX 推理、锁自由多生产者单消费者并发模型、零拷贝图像传输、固定内存槽管理、NVIDIA Jetson Orin Nano SoC（CPU+GPU）、C++/Python 运行时、系统级计时测量与 RTT 统计。

**📊 数据集**

使用数据集：ZOD（Zenseact Open Dataset）预录制的 IMU 与摄像头数据（OXTS IMU、Main Camera）进行实验；对不同传感器组合（如 Brake+Dark）进行测试。

**📈 对比分析**

比较方法：在 Jetson Orin Nano 上将相同的 lambda 函数与原生 ROS 2 Python 节点做对比；通过 RTT、平均延迟、95% 分位、MAD、内存与功耗等指标评估。实验结果显示 Lambda 框架平均 RTT 约减半，YOLO 推理的平均延迟从 532 ms 降至 192 ms，Brake+Dark 的平均延迟从 71 ms 降至 43 ms，整体 jitter 亦得到改善，功耗与内存占用相近。

**⚠️ 局限性**

局限性：实验仅覆盖两种传感器（IMU 与摄像头），未验证多并发话题下的可伸缩性；YOLO 任务接近平台资源上限；使用预录制数据，未测试实时车载部署；RTT 只测计算完成到决策生成，未计入 DDS 发送时延；未评估安全隔离与学习型决策模块。

---

## 431. FlexLoRA: Entropy-Guided Flexible Low-Rank Adaptation

**arXiv ID:** 2601.22905 | [PDF](https://arxiv.org/pdf/2601.22905v1)

**作者:** Muqing Liu `[一作]` (Southeast University), Yuheng Jia `[通讯]` (Southeast University)

**通讯引用:** 1710 | [OpenAlex ID](https://openalex.org/A5013880628)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种名为FlexLoRA的可扩展低秩微调框架，能够在保持训练参数量有限的前提下，动态地对大模型进行低秩适配。

**💡 创新点**

核心创新包括：1）基于谱熵的矩阵级重要性度量；2）在全局秩预算下同时支持秩剪枝与扩展；3）采用零影响初始化新奇异方向以保证训练稳定。

**🔧 技术方法**

技术实现：SVD形式的LoRA更新、谱熵计算、全局秩预算调度、零影响初始化、PyTorch实现。

**📊 数据集**

在自然语言理解（GLUE）、常识推理（8项基准）以及视觉任务（VTAB）等多模态数据集上进行评估。

**📈 对比分析**

与LoRA、AdaLoRA等主流PEFT基线在相同参数预算下对比，FlexLoRA在大部分任务上取得更高平均分（如GLUE 89.1% 对比 AdaLoRA 88.1%），表现更优。

**⚠️ 局限性**

局限性：1）依赖全局秩预算调度，需手动设定；2）谱熵计算在大模型上计算成本较高；3）在极低参数预算或非常深层模型时，扩展策略可能不足以充分利用资源。

---

## 432. Uncertainty-Aware Extrapolation in Bayesian Oblique Trees

**arXiv ID:** 2601.22899 | [PDF](https://arxiv.org/pdf/2601.22899v1)

**作者:** Viktor Andonovikj `[一作]` (Jožef Stefan Institute), Pavle Boškoski `[通讯]` (Faculty of Information Studies in Novo mesto)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种将高斯过程（GP）作为叶子预测器的变分斜分割预测聚类树（VSPYCT-GP），实现可解释且能校准不确定性的回归模型，尤其在外推情形下表现更好。

**💡 创新点**

创新点包括：1) 将贝叶斯斜分割树的分割不确定性与叶子层的GP预测相结合；2) 设计基于训练样本支持的门控机制，决定何时使用GP进行外推；3) 在单棵树框架内实现功能与路由不确定性的分解，保持可解释性。

**🔧 技术方法**

使用技术包括：变分贝叶斯推断的斜分割树、每叶子局部GP回归、Mahalanobis距离门控、Monte Carlo 样本合成预测、梯度优化训练、以及自适应阈值校准。

**📊 数据集**

实验数据集：20个OpenML回归数据集（样本量 768–72,000，特征 5–189），合成三维线性数据，用于插值与外推对比。

**📈 对比分析**

通过 10 折交叉验证和 NRMSE（归一化均方根误差）比较，VSPYCT-GP 在 14/20 数据集上优于基线 VSPYCT，外推实验显示其 RMSE 明显低于 VSPYCT，且预测不确定性随外推距离递增，满足可靠性需求。

**⚠️ 局限性**

局限性：GP 在叶子样本量大时计算成本高；门控阈值对性能敏感，可能在高维或样本稀疏叶子中不稳定；目前仅支持高斯观测噪声，需扩展至非高斯/分类任务；门控依赖于协方差估计，易受噪声影响。

---

## 433. Game-Theoretic Co-Evolution for LLM-Based Heuristic Discovery

**arXiv ID:** 2601.22896 | [PDF](https://arxiv.org/pdf/2601.22896v1)

**作者:** Xinyi Ke `[一作]` (Institute of Automation, Chinese Academy of Sciences), Jian Cheng `[通讯]` (Institute of Automation, Chinese Academy of Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 ASRO 框架，将 LLM 生成的启发式求解器和实例生成器建模为两玩家零和博弈，通过自适应的生成与评估循环实现算法的共进化。

**💡 创新点**

将游戏理论中的 PSRO 思路迁移到可执行程序空间，引入持续的策略池与元策略，并利用 LLM 生成最佳响应形成自生成难度日程，从而显著提升启发式求解器的鲁棒性与泛化。

**🔧 技术方法**

使用 LLM（如 DeepSeek-V3.2）进行程序搜索与进化搜索（EoH），PSRO 风格的元策略计算（最小化极大化求解）、混合策略与基于 payoff 矩阵的最佳响应搜索。

**📊 数据集**

在在线箱子装填（OBP）中使用 Falkenauer、Hard28、Weibull 等；在旅行商问题（TSP）中使用随机 100 节点实例与 TSPLIB；在容量车队路由（CVRP）中使用 CVRPLIB 的 A、B、E、F、M、P、X 系列实例。

**📈 对比分析**

与仅在固定分布下训练的 EoH 以及改进版（数据增强、无记忆自对弈）进行比较，ASRO 在三类任务的测试集上均显著降低 optimality gap，尤其在结构更复杂的实例上差距更大，展示出更强的鲁棒性和泛化性能。

**⚠️ 局限性**

依赖精确或近似最优参考值导致评估噪声；目前仅支持零和博弈，未考虑多目标或多样性；计算开销相对较大；对实例分布估计的准确性和真实感仍有提升空间。

---

## 434. Semantic Leakage from Image Embeddings

**arXiv ID:** 2601.22929 | [PDF](https://arxiv.org/pdf/2601.22929v1)

**作者:** Yiyi Chen `[一作]` (Aalborg University), Johannes Bjerva `[通讯]` (Aalborg University)

**通讯引用:** 1393 | [OpenAlex ID](https://openalex.org/A5013472329)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究并证明图像嵌入能通过对齐、检索和多阶段推理泄露语义信息，提出一种无解码器、仅依赖本地检索的轻量泄露框架。

**💡 创新点**

将语义邻域保持定义为泄露核心，并展示即便仅保留局部邻域也能恢复大量语义内容，提出了基于对齐的低成本泄露方法。

**🔧 技术方法**

使用一阶线性对齐、基于CLIP/其他嵌入的本地检索器、LLM（如Deepseek、Gemini）与VLM（Gemini‑flash、GPT‑5.1）进行文本与图像重建。

**📊 数据集**

采用COCO、nocaps等公开数据集，并在Gemini、Nomic、Clip、Cohere等多种嵌入模型上进行评估。

**📈 对比分析**

与传统检索精度、BLEU/ROUGE等指标对比，邻域保留下F1可达0.8，文本重建Rouge‑L约0.3‑0.5，表明即使极度压缩也能显著泄露语义。

**⚠️ 局限性**

仅聚焦语义邻域，未探讨更高级隐私对策；对极低质量嵌入或极端压缩的鲁棒性尚未充分验证。

---

## 435. Towards Explicit Acoustic Evidence Perception in Audio LLMs for Speech Deepfake Detection

**arXiv ID:** 2601.23066 | [PDF](https://arxiv.org/pdf/2601.23066v1)

**作者:** Xiaoxuan Guo `[一作]` (Communication University of China), Qin Zhang `[通讯]` (Communication University of China)

**通讯引用:** 16472 | [OpenAlex ID](https://openalex.org/A5100425448)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在语音深度伪造检测任务中，通过在音频大模型（Audio LLM）中显式加入时间–频率可视化表征（如CQT声谱图）来增强细粒度声学证据的可访问性，从而提升检测可靠性与稳健性。

**💡 创新点**

核心创新在于提出“acoustic evidence accessibility”框架——将音频与结构化声谱视图交叉输入音频LLM，改变模型对声学特征的访问方式，抑制语义主导的捷径学习，并通过视觉token与文本token共同推理实现更稳健的伪造判别。

**🔧 技术方法**

技术包括：使用Qwen2.5-Omni Audio LLM（3B/7B）作为主体；Whisper音频编码器提取原始音频token；ViT视觉编码器处理CQT声谱图；多模态对齐器将两种token映射到共享隐藏空间；LoRA微调全模型；并通过自回归语言建模任务完成二分类推断。

**📊 数据集**

实验数据集为ASVspoof2019 LA（训练/评估）与ASVspoof2021 LA（跨域测试）。

**📈 对比分析**

与多种基线（RawNet2、AASIST、ALLM4ADD、DFALLM等）对比，SDD-APALLM在ASVspoof2019 LA上取得ACC 99.46%（相比Audio‑Only 98.76%），显著优于其他音频LLM方法，并接近或超过传统端到端模型；在跨域2021 LA上表现更稳健，误差幅度大幅减小。

**⚠️ 局限性**

局限性包括：仍依赖于大模型与预训练语音编码器，模型规模大导致算力和能耗高；目前仅在ASVspoof系列数据上验证，跨语言、跨攻击类型的泛化待进一步评估；以及对不同声谱表征的选择仍需经验指导，未形成统一的最优方案。

---

## 436. On the Impact of Code Comments for Automated Bug-Fixing: An Empirical Study

**arXiv ID:** 2601.23059 | [PDF](https://arxiv.org/pdf/2601.23059v1)

**作者:** Antonio Vitale `[一作]` (Politecnico di Torino), Rocco Oliveto `[通讯]` (University of Molise)

**通讯引用:** 13842 | [OpenAlex ID](https://openalex.org/A5009727039)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对包含或不包含自动生成的多意图代码注释的 bug 修复数据集进行实验，评估注释对大型语言模型（LLM）自动修复 Java 方法错误的影响。

**💡 创新点**

创新点在于首次系统地检验注释在训练和推理阶段对 ABF 性能的作用，并通过 SHAP 分析揭示不同注释意图对模型决策的重要性。

**🔧 技术方法**

实验采用两种 LLM：CodeT5+（220M 参数的编码-解码器）和 DeepSeek-Coder（1.3B 参数的解码器），并结合 GPT‑3.5 自动生成五类注释。

**📊 数据集**

使用基于 Tufano 数据集重构的 D_b（116,372 对）与 D_c（同一实例前后加注释的版本），并在 80/10/10 划分下进行训练与评估。

**📈 对比分析**

通过 Exact Match 与 CodeBLEU 两指标对四种训练/推理组合进行对比，结果显示在推理时提供注释可使模型准确率提升约 2–3 倍，DeepSeek-Coder 在注释环境下达 12.77% EM，CodeT5+ 达 9.80%。

**⚠️ 局限性**

主要局限包括：数据集仅覆盖 2011–2017 年的 Java 代码；自动生成的注释质量与真实开发者注释存在差距；EM 仅衡量完全匹配，未考虑可行替代补丁；模型对 Bug‑free 情境的偏差与对较新 LLM 的可推广性待验证。

---

## 437. From Abstract to Contextual: What LLMs Still Cannot Do in Mathematics

**arXiv ID:** 2601.23048 | [PDF](https://arxiv.org/pdf/2601.23048v1)

**作者:** Bowen Cao `[一作]` (Chinese University of Hong Kong), Furu Wei `[通讯]` (Microsoft)

**通讯引用:** 31437 | [OpenAlex ID](https://openalex.org/A5014662947)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 ContextMATH 基准，通过将 AIME 与 MATH‑500 的抽象题目改写为情境支撑（SG）和约束隐藏（CS）两类叙事化版本，系统评估大语言模型在情境化数学推理中的表现。

**💡 创新点**

创新点包括：①以情境化和约束隐藏方式重构传统数学题，①系统分离并量化公式化与推理两大瓶颈，②证明情境化微调能显著提升性能，而单独训练公式化模型效果有限。

**🔧 技术方法**

采用的技术手段包括：LLM（如 GPT‑4、Gemini、LLaMA 系列）自动生成与校验情境化题目；人工审核确保数学等价性；自定义评判器测量公式化准确性；基于 SFT 与 RL 的微调实验。

**📊 数据集**

所使用的数据集包括：AIME 2024/2025 与 MATH‑500 的原始题目；生成的 SG 与 CS 版本（约 1.5 万题）；DeepMath‑103K 作为原始训练数据；以及 5 万条自动校验通过的情境化示例。

**📈 对比分析**

对比方法：在原始抽象题目与 SG、CS 版本上分别计算模型准确率；开放源模型平均降幅为 SG 13%、CS 34%；专有模型降幅为 SG 13%、CS 20%；通过情境微调实验（+SFT_Syn、+SFT_Mix）提升约 40%‑60%，但仍与抽象版本存在显著差距。

**⚠️ 局限性**

局限性：公式化与推理仍是互补的瓶颈，单独训练公式化模型无效；即便采用最优规模与情境化微调，CS 版本的准确率仍低于 60%；当前方法无法彻底弥补抽象与情境之间的性能鸿沟，亟需更深层次的整合策略。

---

## 438. Uncovering Hidden Inclusions of Vulnerable Dependencies in Real-World Java Projects

**arXiv ID:** 2601.23020 | [PDF](https://arxiv.org/pdf/2601.23020v1)

**作者:** Stefan Schott `[一作]` (Paderborn University), Eric Bodden `[通讯]` (Paderborn University and Fraunhofer IEM)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

**🎯 论文内容**

本文提出一种混合型依赖扫描方法 Unshade，通过在 SBOM 中增添被篡改的隐藏依赖并交给元数据扫描器，检测 Java 项目中的隐藏漏洞。

**💡 创新点**

创新点在于结合轻量级元数据扫描与基于字节码指纹的代码级检测，能够识别重打包、重命名等隐藏的已知漏洞依赖。

**🔧 技术方法**

使用技术包括 Java 字节码指纹生成与匹配、SBOM 生成、与 OSV 元数据扫描器集成，以及对 Maven 依赖的解析。

**📊 数据集**

数据集为 GitHub 上 1,808 个拥有 500 星以上的最流行 Java Maven 项目。

**📈 对比分析**

与单纯使用 OSV 的元数据扫描相比，Unshade 在同一批项目中多检测到约 7,700 条 CVE，平均每个受影响项目多发现 8 个隐藏漏洞，扫描时间保持在可接受范围内。

**⚠️ 局限性**

局限性包括仅针对 Maven 项目、仅利用 OSV advisory 作为漏洞来源、以及检测到的隐藏漏洞不一定可在目标项目中被利用。

---

## 439. Integrating Multi-Label Classification and Generative AI for Scalable Analysis of User Feedback

**arXiv ID:** 2601.23018 | [PDF](https://arxiv.org/pdf/2601.23018v1)

**作者:** Sandra Loop `[一作]` (SAP SE), Martin Schrepp `[通讯]` (SAP SE)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a2602d71-93ab-4bad-974b-672788df8193` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了如何通过机器学习和生成式 AI 对大规模用户评论进行主题分类、情感分析和自动摘要，以支持软件产品的 UX 评估。

**💡 创新点**

创新点在于结合多标签分类模型与 SBERT 嵌入提升准确率，制定基于标签的 GenAI 摘要流程，并揭示情感与满意度指标的非单一对应关系。

**🔧 技术方法**

采用了超学习多标签分类、SBERT 嵌入、XGBoost、GenAI 生成摘要、情感分析、交叉验证和人工反馈循环等技术。

**📊 数据集**

使用的数据集包括 SAP BTP 30 款产品收集的 2,756 条评论（后扩展至 8,757 条）以及两项调查（教程质量、应用可用性）共约 3,000 条评论。

**📈 对比分析**

通过对比 fastText 与 SBERT 嵌入，SBERT 在微平均 F1 上提升 24.58%；多标签 XGBoost 模型在各标签上 F1 介于 0.49 与 0.82 之间。

**⚠️ 局限性**

局限性包括评论积极率低、情感分类偏差、罕见标签性能差、无法准确判断可操作性、样本自选偏差导致外推受限。

---

## 440. Mem-T: Densifying Rewards for Long-Horizon Memory Agents

**arXiv ID:** 2601.23014 | [PDF](https://arxiv.org/pdf/2601.23014v1)

**作者:** Yanwei Yue `[一作]` (Peking University), Yan Zhang `[通讯]` (Peking University)

**通讯引用:** 45017 | [OpenAlex ID](https://openalex.org/A5100456327)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种可训练的层次化记忆代理和树引导强化学习框架，实现端到端优化记忆构建与检索。

**💡 创新点**

创新点在于将稀疏终端奖励转化为密集步骤监督的记忆操作树与回顾信用分配，解决长序列记忆管理的时间信用分配问题。

**🔧 技术方法**

采用强化学习（GRPO/PPO）、记忆操作树（MoT）与回顾信用分配、离线专家学习等技术。

**📊 数据集**

在LoCoMo、LongMemEval、HotpotQA、NarrativeQA四个长上下文基准上进行评测。

**📈 对比分析**

与13个基线对比，SOTA F1提升14.9%/14.55%，在各基准上均跑到最优，且推理成本降低约24%且保持精度。

**⚠️ 局限性**

仍受大模型依赖、树搜索成本、对多模态或开放域知识适应性有限，训练数据规模受限。

---

## 441. About an Automating Annotation Method for Robot Markers

**arXiv ID:** 2601.22982 | [PDF](https://arxiv.org/pdf/2601.22982v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 442. Automatic Constraint Policy Optimization based on Continuous Constraint Interpolation Framework for Offline Reinforcement Learning

**arXiv ID:** 2601.23010 | [PDF](https://arxiv.org/pdf/2601.23010v1)

**作者:** Xinchen Han `[一作]` (Institut Polytechnique de Paris), Michel Marot `[通讯]` (Institut Polytechnique de Paris)

**通讯引用:** 341 | [OpenAlex ID](https://openalex.org/A5057498700)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了连续约束插值框架（CCI），统一了行为克隆、密度正则化和支持约束三类离线强化学习约束，并基于此设计了自动约束策略优化算法（ACPO），通过拉格朗日双重更新自适应调节约束强度。

**💡 创新点**

创新点在于：①将三种主流约束视为一个连续谱的特例，给出单一插值参数实现平滑切换与组合；②通过理论分析得到最大熵性能差分与下界，揭示插值参数对保守性和性能的影响；③在算法层面实现自动调节约束的双重优化，避免手动设定约束形式与强度。

**🔧 技术方法**

核心技术包括：最大熵强化学习、连续约束插值（CCI）、拉格朗日双重优化、逆 KL 近似投影、行为策略学习（高斯或 CVAE）以及离线数据评估与正则化方法。

**📊 数据集**

实验数据集涵盖：D4RL（Gym‑MuJoCo、AntMaze、Kitchen）和 NeoRL2（Pipeline、Simglucose、RocketRecovery 等工业/医疗场景），共 20+ 子任务。

**📈 对比分析**

与 BCQ、IQL、CQL、TD3+BC、SPOT、SWG、EQL、DTQL、EDAC、MCQ、RAMBO、MOBILE 等主流基线相比，ACPO 在大多数任务上取得最优或接近最优的平均归一化得分，尤其在 NeoRL2 复杂环境中显著优于 TD3+BC，显示出稳健的性能提升。

**⚠️ 局限性**

限制包括：①对长时延、稀疏奖励的长轨迹任务的适配尚未充分验证；②性能高度依赖行为策略估计器的质量，若估计不准会削弱约束信号；③插值参数 λ 的动态变化虽能自适应，但在训练早期可能需要较大 λ 以抑制外域行为，导致学习收敛速度慢。

---

## 443. Mano: Restriking Manifold Optimization for LLM Training

**arXiv ID:** 2601.23000 | [PDF](https://arxiv.org/pdf/2601.23000v1)

**作者:** Yufei Gu `[一作]` (Hong Kong University of Science and Technology), Zeke Xie `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 253 | [OpenAlex ID](https://openalex.org/A5066773635)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对大规模语言模型预训练，提出了一种新的优化器 Mano，并在 LLaMA 和 Qwen3 上进行实验。

**💡 创新点**

创新点在于把流形优化重新定义为动量驱动的软约束，并引入旋转 Oblique 流形正则化，既保持了梯度方向，又避免了传统重投影带来的表达力受限，显著提升了收敛速度和训练稳定性。

**🔧 技术方法**

使用了流形优化（Oblique、旋转 Oblique）、动量投影、矩阵归一化、梯度稳定性分析等技术，并与 AdamW、Muon 进行对比。

**📊 数据集**

使用中文语料库（包含约 10B 语料）对 LLaMA‑350M、LLaMA‑1.3B、Qwen3‑0.6B、Qwen3‑1.7B 进行预训练，评估模型在 10,000 步时的困惑度。

**📈 对比分析**

与 AdamW、Muon 在相同批大小（512）、相同学习率、相同训练步骤进行对比。Mano 在困惑度下降速度、壁钟时间和显存占用上均优于两者，尤其在大型模型上可达到 1.75× 的收敛加速。

**⚠️ 局限性**

实验规模有限，未覆盖更大模型或长期训练；理论分析仅适用于简化版 Mano，缺少对动量动态的完整证明；缺少不同超参、不同语言语料的泛化验证。

---

## 444. ArabicDialectHub: A Cross-Dialectal Arabic Learning Resource and Platform

**arXiv ID:** 2601.22987 | [PDF](https://arxiv.org/pdf/2601.22987v1)

**作者:** Salem Lahlou `[一作]` `[通讯]` (Mohamed Bin Zayed University of Artificial Intelligence), Salem Lahlou (Mohamed Bin Zayed University of Artificial Intelligence)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

创建了552条跨六种阿拉伯语方言（摩洛哥达里贾、黎巴嫩、叙利亚、阿联酋、沙特、MSA）的短语语料库，并开发了交互式学习平台ArabicDialectHub。

**💡 创新点**

以LLM生成+母语者验证的方式构建教学型方言短语集，按难度分层，提供跨方言对比学习；同时发布开源平台实现翻译、适应性测验、云同步进度与文化语境卡。

**🔧 技术方法**

使用大语言模型（Claude 3.5、GPT‑4）生成短语，React + TypeScript 前端，Clerk + Supabase 后端，Netlify 部署；以及智能干扰项生成算法和词序排列测验。

**📊 数据集**

主要使用自研的LLM生成短语并由5名母语者验证；参考 MADAR、AOC 等现有方言语料但未直接使用，未采用公开大型对话语料。

**📈 对比分析**

未进行用户学习成效实验，仅通过人工验证确认自然性与语义等价；平台功能已实现无错误，但缺乏学习效果评估，无法给出性能数值。

**⚠️ 局限性**

验证仅覆盖达里贾和黎巴嫩方言，缺乏其他方言本土验证；语料规模有限、缺少专业领域词汇、无音频支持、未做交叉验证与用户研究，难以评估学习效果。

---

## 445. A Unified View of Attention and Residual Sinks: Outlier-Driven Rescaling is Essential for Transformer Training

**arXiv ID:** 2601.22966 | [PDF](https://arxiv.org/pdf/2601.22966v1)

**作者:** Zihan Qiu `[一作]` (Qwen Team), Junyang Lin `[通讯]` (Qwen Team)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文探讨大型语言模型中注意力与残差“极值”对模型训练与推理的功能性影响，提出“极值驱动重缩放”假设，并基于此设计了两种极值缓解策略（PreAffine 与 GatedNorm），同时验证了这些方法在不同模型规模和量化设置下的性能提升。

**💡 创新点**

创新点包括：①将注意力极值（attention sink）与残差极值（residual sink）统一为极值驱动重缩放现象；②证明去除归一化或直接裁剪极值会破坏该重缩放机制并导致性能下降；③提出可学习的预尺度（PreAffine）与低秩自门控归一化（GatedNorm）两种直接实现重缩放的结构，显著降低极值并提升量化鲁棒性。

**🔧 技术方法**

核心技术：softmax 与 RMSNorm 归一化、动态双曲正切（DyT）、自门控层（GatedAttention/GatedNorm）、可学习缩放向量（PreAffine）、低秩门控（r≈16）、量化技术（W4A4、SmoothQuant）、对比实验中的 loss、MMLU、STEM、Code 等基准指标。

**📊 数据集**

数据集与规模：在 120B、1T、1.2T 级别的预训练语料上训练 2B、7B、24B 参数模型（含 MoE），并在 FP16/BF16 与 4-bit 量化（W4A4）设置下进行评测；使用 MMLU、S-GPQA、GPQA-D、GSM8k、Math、Crux 等公开评测基准。

**📈 对比分析**

比较方法：以基线（软max+RMSNorm+GA）为参照，记录训练 loss、MMLU/其他任务分数；实验显示 PreAffine 与 GatedNorm 均能在不增加显著参数量的前提下提升平均 2 分（+0.02 绝对提升），在 4-bit 量化下降低损失 1.23 分，且在 STEM/Code 等任务上超过 +2 分的提升；同时显著减少极值幅度并提高量化鲁棒性。

**⚠️ 局限性**

局限性：实验性验证为主，缺乏对为何需要极值驱动重缩放的深入理论解释；目前仅针对预归一化 Transformer 结构验证，尚未探讨其他模型（如卷积、稀疏注意力）及更小规模模型的适用性。

---

## 446. EAG-PT: Emission-Aware Gaussians and Path Tracing for Indoor Scene Reconstruction and Editing

**arXiv ID:** 2601.23065 | [PDF](https://arxiv.org/pdf/2601.23065v1)

**作者:** Xijie Yang `[一作]` (Zhejiang University), Linning Xu `[通讯]` (The Chinese University of Hong Kong)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

研究了一种基于二维高斯原语的光线追踪框架 EAG-PT，能够在不依赖网格的情况下实现室内场景的物理一致重建与编辑。

**💡 创新点**

创新点包括：①将发光与非发光分离并使用 2D 高斯作为统一几何/材质代理；②通过单次光照回传实现材质恢复；③在编辑后使用多次光线追踪并可将结果烘焙回 2D 高斯，实现质量与实时性的双重优势。

**🔧 技术方法**

主要技术包括可微 2D 高斯光线追踪、单次与多次光线追踪、辐射缓存、光照烘焙、Pytorch+OptiX 加速、感知量化等。

**📊 数据集**

使用的数据集有真实室内数据 FIPT（f-）、VR‑NeRF Eyeful Tower（e-）、Synthetic Blender 场景（b-）以及自采集的 LectureRoom。

**📈 对比分析**

与基准的 0‑bounce、1‑bounce、FIPT 逆向路径追踪以及多次光线追踪等方法对比，EAG‑PT 在视觉质量、全局照明一致性和实时渲染速度上均优于对手，尤其在细节保持与无网格伪影方面表现突出。

**⚠️ 局限性**

局限性在于需要人工标注发射掩模、仅支持漫反射材质、光线追踪渲染仍耗时，缺乏实时级别，对复杂 BRDF 与动态光源的支持有限。

---

## 447. Adaptive Edge Learning for Density-Aware Graph Generation

**arXiv ID:** 2601.23052 | [PDF](https://arxiv.org/pdf/2601.23052v1)

**作者:** Seyedeh Ava Razi Razavi `[一作]` (Brock University), Renata Dividino `[通讯]` (Brock University)

**通讯引用:** 299 | [OpenAlex ID](https://openalex.org/A5078237873)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `3f18e8e3-0266-457c-8567-9039b6d2394d` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

开发了基于WGAN‑GP的密度感知条件图生成框架，使用可微分的距离驱动边预测器和类特定的密度感知边选择；

**💡 创新点**

创新点在于学习节点间的距离‑映射边概率，并通过密度感知的top‑k边选择实现类特定稀疏度控制，避免了传统随机采样导致的结构失真；

**🔧 技术方法**

采用Wasserstein GAN与梯度惩罚、GCN‑Critic、GNN‑Generator、距离基边预测器、温度退火、类嵌入以及节点级噪声；

**📊 数据集**

在三大基准图分类数据集上验证：MUTAG、ENZYMES、PROTEINS；

**📈 对比分析**

与 DeepGMG、GraphRNN、LGGAN、WPGAN 等方法在 MMD（度分布、聚类系数、谱特征）上比较，实验显示在聚类与谱指标上显著优于基线，且唯一性与新颖度均较高；

**⚠️ 局限性**

局限性是固定的 top‑k 选择限制了度分布的多样性，导致生成图中孤立节点出现频率偏高，未来需考虑概率采样或引入度分布损失以提升结构多样性。

---

## 448. MedMCP-Calc: Benchmarking LLMs for Realistic Medical Calculator Scenarios via MCP Integration

**arXiv ID:** 2601.23049 | [PDF](https://arxiv.org/pdf/2601.23049v1)

**作者:** Yakun Zhu `[一作]` (Shanghai Jiao Tong University), Xiaofan Zhang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 5606 | [OpenAlex ID](https://openalex.org/A5100330729)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 MedMCP‑Calc 基准，评估大型语言模型在基于 Model Context Protocol 的真实医疗计算器工作流中的表现，并基于此训练出 CalcMate。

**💡 创新点**

创新点在于首次将模糊自然查询、交互式 EHR、全流程多步情景和过程级评估结合到医疗计算器基准中，并通过 MCP 提供统一的工具调用接口。

**🔧 技术方法**

主要技术包括 Model Context Protocol (MCP)、ReAct 代理框架、情境规划与工具增强的微调策略，以及多模态工具（SQL、Python、搜索/获取）集成。

**📊 数据集**

使用的数据集为 118 题目的任务集合（来自 MDCalc/Medscape/QxMD 公开计算器），与 49,419 名 MIMIC‑IV 病例的结构化 EHR 以及 LLM 生成的任务描述与情境。

**📈 对比分析**

在 23 个主流模型（包括专有与开源）上进行评估，指标为计算器选择 (CS)、证据获取 (EA)、定量精度 (QP) 与任务完成 (TF)，结果表明顶尖模型仍仅达 30%‑70% 的精度，而微调的 CalcMate 在所有指标上实现了开源模型的最优表现。

**⚠️ 局限性**

局限性包括基准 EHR 数据可能存在地区偏差、工具增强导致上下文长度大幅增长且推理延迟升高，以及模型仍易出现知识错报、计算错误和工具使用不足等问题。

---

## 449. Guided by Trajectories: Repairing and Rewarding Tool-Use Trajectories for Tool-Integrated Reasoning

**arXiv ID:** 2601.23032 | [PDF](https://arxiv.org/pdf/2601.23032v1)

**作者:** Siyu Gong `[一作]` (Southeast University), Min-Ling Zhang `[通讯]` (Southeast University)

**通讯引用:** 15371 | [OpenAlex ID](https://openalex.org/A5079083101)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 AutoTraj，一种两阶段框架，用自动修复与奖励工具使用轨迹来训练工具集成推理（TIR）模型。

**💡 创新点**

创新点在于：①在监督微调阶段不再仅过滤低质量轨迹，而是用 LLM 自动修复并生成新的高质量轨迹；②在强化学习阶段引入轨迹级奖励模型，提供细粒度奖励，显著缓解奖励稀疏问题。

**🔧 技术方法**

核心技术包括：多维轨迹评估（正确性、置信度、长度、重复率）、LLM‑as‑Repairer、对比学习生成轨迹奖励模型、GRPO 强化学习以及格式/结果/轨迹三重奖励组合。

**📊 数据集**

使用 Tool‑Star 数据集（10,000 题答对无轨迹）、Tool‑Star RL 训练集，以及 9 个公开基准（AIME2024/25、AMC23、GSM8K、Math、HotpotQA、2Wiki、MuSiQue、HLE）进行评测。

**📈 对比分析**

与 SFT‑Only、RL‑Only、SFT‑RL 以及 Tool‑Star 等基线对比，AutoTraj 在大部分任务上取得最高或第二高分，平均准确率提升约 5%，且推理轨迹长度缩短 7‑倍，表现出更高的推理效率。

**⚠️ 局限性**

局限性：①轨迹修复依赖 LLM 的生成质量，仍可能产生误修复；②轨迹奖励模型在极少量样本下可能过拟合；③目前仅验证了搜索与代码执行两种工具，未覆盖更广泛的工具生态。

---

## 450. Causal Characterization of Measurement and Mechanistic Anomalies

**arXiv ID:** 2601.23026 | [PDF](https://arxiv.org/pdf/2601.23026v1)

**作者:** Hendrik Suhr `[一作]` (CISPA Helmholtz Center for Information Security), Jilles Vreeken `[通讯]` (CISPA Helmholtz Center for Information Security)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了基于因果模型的异常根因定位与类型识别，提出将测量错误与机制漂移分别建模为对潜变量与观测变量的硬干预，并给出可识别性证明与最大似然估计算法。

**💡 创新点**

首次在根因分析中同时对测量误差与机制变更进行区分；引入潜在干预的可识别性理论；提出利用最大似然与贝叶斯稀疏性假设的混合估计框架。

**🔧 技术方法**

采用因果结构方程模型、稀疏机制位移假设、最大似然估计、蒙特卡罗积分、稳健加性模型回归、修剪核密度估计等技术。

**📊 数据集**

使用合成随机ER DAG（15节点）以及真实数据集Sachs、Causal Chambers基准，并在NYC出租车数据上做案例研究。

**📈 对比分析**

与多种基线方法（Shapley‑based、SmoothTraversal、RootClam等）在根因定位的Top‑k召回率和异常类型分类准确率进行对比，实验表明在合成与Causal Chambers数据上可获得最高召回率，分类准确率远优于基线，NYC案例亦能产生可解释结果。

**⚠️ 局限性**

需要先验或已学习的DAG，对模型假设（潜变量独立、噪声正态、测量误差独立等）敏感；仅适用于表格数据；在干预稀少或结构未知时性能下降。

---

## 451. Leveraging Convolutional Sparse Autoencoders for Robust Movement Classification from Low-Density sEMG

**arXiv ID:** 2601.23011 | [PDF](https://arxiv.org/pdf/2601.23011v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 452. Value-at-Risk Constrained Policy Optimization

**arXiv ID:** 2601.22993 | [PDF](https://arxiv.org/pdf/2601.22993v1)

**作者:** Rohan Tangri `[一作]` (Oxford), Jan-Peter Calliess `[通讯]` (Oxford)

**通讯引用:** 513 | [OpenAlex ID](https://openalex.org/A5039607149)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

开发了VaR-CPO算法，实现在强化学习中直接优化价值风险约束，保证训练过程中的安全性。

**💡 创新点**

采用单侧切比雪夫不等式构造可微分的VaR约束近似，并引入状态增广实现马尔可夫性质，给出训练阶段的最坏情况约束违例上界。

**🔧 技术方法**

结合CPO的信赖域优化、一次性优势估计、Chebyshev上界、状态增广以及JAX加速实现。

**📊 数据集**

在JAX实现的Bra x与gymnasium环境下的EcoAnt和IcyLake两种任务。

**📈 对比分析**

与PPO、CPO、CPPO三种基线对比，实验显示VaR-CPO在可行环境下实现零约束违例并保持良好奖励，优于其他方法。

**⚠️ 局限性**

近似使用切比雪夫不等式导致过度保守，且仅在期望成本低于阈值时有效，对不满足该前提的策略需额外恢复机制。

---

## 453. ERA: Epoch-Resolved Arbitration for Duelling Admins in Group Management CRDTs

**arXiv ID:** 2601.22963 | [PDF](https://arxiv.org/pdf/2601.22963v1)

**作者:** Kegan Dougal `[一作]` `[通讯]` (Element Creations Ltd), Kegan Dougal (Element Creations Ltd)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

**🎯 论文内容**

本文提出了通过“epoch‑resolved arbitration”（时段仲裁）来解决 CRDT 组管理中管理员冲突（Duelling Admins）问题，利用最终性节点发出 epoch 事件来强制化事件执行顺序；

**💡 创新点**

创新点在于引入第三方最终性节点（finality node）和 epoch 事件机制，以实现对并发撤销操作的公平最终性，从而避免 CRDT 中的 roll‑back 与背调攻击；

**🔧 技术方法**

主要技术包括冲突无关复制数据类型（CRDT）、哈希 DAG、因果广播、时段仲裁、以及对最终性节点的信任与加密防护；

**📊 数据集**

本文没有使用实测数据集，而是通过理论分析与案例演示（如管理员互相撤销的情形）说明方法有效性；

**📈 对比分析**

与现有的 Keyhive、Matrix 方案对比，epoch 仲裁可在不集中化权力、也不需要持续共识的情况下终止撤销循环，性能上保持 CRDT 的高可用性，同时在需要时可等待最终性；

**⚠️ 局限性**

主要限制在于对最终性节点的信任依赖——若节点背调或延迟，可能导致历史被重写或撤销不及时，并且对节点失效时的容错机制仍需进一步完善。

---

## 454. One-shot Optimized Steering Vector for Hallucination Mitigation for VLMs

**arXiv ID:** 2601.23041 | [PDF](https://arxiv.org/pdf/2601.23041v1)

**作者:** Youxu Shi `[一作]` (University of Science and Technology of China), Dong Liu `[通讯]` (University of Science and Technology of China)

**通讯引用:** 23298 | [OpenAlex ID](https://openalex.org/A5100407381)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种仅使用单个样本即可训练的输入无关式模型引导向量（OSGA），用于在推理时有效降低视觉语言模型的幻觉与安全风险。

**💡 创新点**

创新点在于：①引入变异度驱动的数据选择策略，挑选靠近知识边界的样本；②在对比学习目标中加入生成锚正则化，稳定并提升引导向量的泛化；③通过单步优化得到的向量可在所有输入上共享，保持极低的推理开销。

**🔧 技术方法**

采用对比式引导、生成锚正则化、变异度数据挑选、以及在VLM解码器层注入偏移的技术，并在LLaVA‑v1.5与Qwen2‑VL上实现。

**📊 数据集**

使用公开基准：CHAIR、POPE、MME、FaithScore、HalFscore、GAVIE、GOAT‑Bench 等多任务数据集进行评测。

**📈 对比分析**

与基线、现有无训练方法（如OPERA、ICD、VCD、RLAIF‑V）对比，OSGA 在对象、属性、关系幻觉、以及安全分类上均取得显著提升，F1 分数提升约 5‑10% 以上，且推理延迟几乎无变化。

**⚠️ 局限性**

局限性包括：需要手动挑选适当的注入层和权重因子 α，过强或深层注入可能导致幻觉反弹；仅用单一样本优化，可能不足以覆盖极端场景；对任务语义高度对齐的情况更有效，非对齐任务效果可能有限。

---

## 455. Leveraging Multi-Rater Annotations to Calibrate Object Detectors in Microscopy Imaging

**arXiv ID:** 2601.23007 | [PDF](https://arxiv.org/pdf/2601.23007v1)

**作者:** Francesco Campi `[一作]` (Helmholtz Zentrum München), Marie Piraud `[通讯]` (Helmholtz Zentrum München)

**通讯引用:** 3777 | [OpenAlex ID](https://openalex.org/A5005966038)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

通过为每位专家训练单独的 Mask R-CNN 检测器并在推理时对其结果进行集成（rater‑specific ensemble, rse），与混合标签训练后集成（label‑sampling ensemble, lse）对比，探讨如何利用多评审标注提升检测器的置信度校准。

**💡 创新点**

创新点在于显式建模评审者间偏差：训练每个评审者的专属模型并将它们聚合，使得模型在评审者意见不一致时自动降低置信度，从而显著改善校准误差。

**🔧 技术方法**

技术包括：Mask R-CNN 迁移学习、评审者专属训练、预测框 IoU 组聚合、按置信度分箱计算 D‑ECE（对象检测的校准误差指标）以及多模型自助采样评估统计显著性。

**📊 数据集**

使用 100 张大肠癌 PDO 明场显微图像（80 张单评审、20 张共识），由两名经验相当的专家分别标注，形成单评审集与共识集。

**📈 对比分析**

比较方法是对不同规模的 lse 与 rse 进行 100 次自助重采样，评估 D‑ECE 与 mAP；结果显示 rse 在 iou=0.5 时 D‑ECE 从 0.15 降至 0.08，mAP 与 lse 基本持平，表明在保持检测精度的前提下显著提升了置信度校准。

**⚠️ 局限性**

局限性包括：标注数据量小、仅有两名评审者、所有模型从同一预训练权重微调导致模型多样性不足、以及 rse 的推理成本随模型数量线性增长。

---

## 456. TriCEGAR: A Trace-Driven Abstraction Mechanism for Agentic AI

**arXiv ID:** 2601.22997 | [PDF](https://arxiv.org/pdf/2601.22997v1)

**作者:** Roham Koohestani `[一作]` (JetBrains Research), Maliheh Izadi `[通讯]` (Delft University of Technology)

**通讯引用:** 4543 | [OpenAlex ID](https://openalex.org/A5024645888)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

实现了基于执行日志的自动状态抽象机制，构建代理行为的马尔可夫决策过程（MDP），并在此模型上进行概率模型检验和运行时异常检测。

**💡 创新点**

创新点在于：①利用决策树学习的谓词树从原始日志自动生成抽象；②将抽象与MDP、概率模型检验与异常检测三者无缝集成；③提出基于计数的在线MDE构建与基于抽象路径的伪计数检验的动态概率保证框架。

**🔧 技术方法**

核心技术包括：运行时验证（runtime verification）、概率模型检验（PCTL）与Storm模型检查器、基于信息增益的决策树学习、谓词树（predicate tree）构造、MDP状态/转移计数、基于路径似然的异常检测。

**📊 数据集**

使用了在示例代理（文件读写任务）上收集的两组数据集：1,000 条随机生成的基线轨迹；1,000 条注入异常（如读写比例失衡、轨迹长度极端）测试轨迹；此外提供了公开的复制包（https://zenodo.org/records/18338703）供实验复现。

**📈 对比分析**

与手工定义状态抽象方法对比，自动抽象在无须开发者手工规则的情况下能够快速构建多状态抽象，并通过概率模型检验得到成功率上界和失败率下界；在异常检测方面，基于轨迹似然的阈值能及时捕获长/短轨迹和读写比例异常，但存在较高的误报率，需要进一步的抽象细化。性能方面，该框架支持在线更新，且在示例任务中能够在几秒级完成一次模型检验与异常评分。

**⚠️ 局限性**

局限性包括：①仅基于已观测到的轨迹，无法覆盖未见过的行为；②当前未实现完整的概率 CEGAR 细化，抽象可能过度泛化导致误报；③对变量选择仍需手工配置，未实现自动注解；④缺乏对文本/语义谓词的支持；⑤未对大规模、复杂代理的性能与可扩展性做充分评估。

---

## 457. Learning Geometrically-Grounded 3D Visual Representations for View-Generalizable Robotic Manipulation

**arXiv ID:** 2601.22988 | [PDF](https://arxiv.org/pdf/2601.22988v1)

**作者:** Di Zhang `[一作]` (Tongji University), Guang Chen `[通讯]` (Tongji University)

**通讯引用:** 12439 | [OpenAlex ID](https://openalex.org/A5101684449)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

设计并实现了 GEM3D 框架，将单视 RGB‑D 的 3D 表示与多步蒸馏的策略学习相结合，实现视角可泛化的机器人操控。

**💡 创新点**

① 引入基于体素特征的粗细尺度点云重建与高精度 Gaussian splatting 的单视预训练；② 通过多步蒸馏软对齐策略，将 3D 语义保持在策略中，显著提升视角泛化能力。

**🔧 技术方法**

体素 U‑Net、可变形交叉注意力、Snowflake 细化网络、Feed‑forward 3D Gaussian splatting、Perceiver‑IO 策略网络与多步相似度蒸馏。

**📊 数据集**

RLBench 12 个操控任务（9 场景）进行训练与评估，并在多视点预训练中使用 8 视图 RGB‑D 轨迹。

**📈 对比分析**

与 PerAct、GNFactor、ManiGaussian 等基线对比，GEM3D 平均成功率提升 12.7%（从 31.5% 至 44.2%），在中等/大视角位移下成功率下降仅 22%/29% 而对手降至 41%/51%，表现更稳健。

**⚠️ 局限性**

预训练仍需多视点数据，训练成本高；对极端动态场景或完全单视点的泛化仍有限；目前仅在 RLBench 机器人模拟环境验证，缺乏真实硬件的实验。

---

## 458. Learnable Permutation for Structured Sparsity on Transformer Models

**arXiv ID:** 2601.22980 | [PDF](https://arxiv.org/pdf/2601.22980v1)

**作者:** Zekai Li `[一作]` (Advanced Micro Devices), Emad Barsoum `[通讯]` (Advanced Micro Devices)

**通讯引用:** 1787 | [OpenAlex ID](https://openalex.org/A5027115167)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

针对Transformer模型的结构稀疏化，本文提出了一种端到端可学习的通道置换框架。

**💡 创新点**

创新点在于引入可学习的置换成本矩阵、基于Sinkhorn的可微二分匹配求解器，以及联合任务损失的稀疏优化目标。

**🔧 技术方法**

技术上使用了可学习成本预测器、差分二分匹配求解器、N:M稀疏掩码生成器（如Wanda）以及交叉熵+蒸馏损失。

**📊 数据集**

实验数据集涵盖视觉的ImageNet（ViT-Base/16、ViT-Large/14）、语言的Wikitext2、ARC-Easy/Challenge、BoolQ、HellaSwag、OpenBookQA、WinoGrande、MMLU以及多模态MMU等。

**📈 对比分析**

与Magnitude、Wanda、SparseGPT、PrunerZero、CP、RIA等主流稀疏化方法对比，所提出方法在ViT、LLM和VLM上分别实现了最高的Top‑1/Top‑5准确率和平均精度，提升幅度在1–3个百分点。

**⚠️ 局限性**

局限性包括置换学习仍需额外的参数与计算开销、对极大模型的全局置换仍受限于分组策略，且仅在现有稀疏掩码框架上验证，未进一步探究与其他稀疏化技术的组合。

---

## 459. Stabilizing the Q-Gradient Field for Policy Smoothness in Actor-Critic

**arXiv ID:** 2601.22970 | [PDF](https://arxiv.org/pdf/2601.22970v1)

**作者:** Jeong Woon Lee `[一作]` (Kyung Hee University), Hyoseok Hwang `[通讯]` (Kyung Hee University)

**通讯引用:** 285 | [OpenAlex ID](https://openalex.org/A5018395387)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了将策略不平滑根源归因于 critic 的 Q‑函数几何结构，构建了 PAVE 框架，通过正则化混合偏导数、向量场一致性和曲率保持来抑制 Q‑梯度振荡，从而实现无需对 actor 进行任何改动即可获得平滑策略。

**💡 创新点**

创新点在于利用隐函数定理推导出策略敏感度受混合偏导数与曲率比率支配的理论，并首次提出 critic‑centric 的几何正则化方法（PAVE）来直接平滑学习信号，显著区别于以往仅对 actor 进行 Lipschitz 或光滑约束的做法。

**🔧 技术方法**

采用隐函数定理、有限差分近似、Hutchinson trace 估计、Fisher 散度一致性等技术，对 TD3/SAC 等 off‑policy actor‑critic 算法的 critic 进行三项正则化（MPR、VFC、Curv）以实现 Q‑梯度场平滑。

**📊 数据集**

在 Gymnasium 与 MuJoCo 提供的六个连续控制环境上进行实验：LunarLanderContinuous‑v2、Pendulum‑v1、Reacher‑v2、Ant‑v5、Hopper‑v5、Walker2d‑v5。

**📈 对比分析**

与 CAP、Grad‑CAPS、ASAP 等传统策略平滑方法对比，PAVE 在大多数任务中获得与或优于基线的累计奖励，并在高维环境（Ant、Walker）中表现出最优的平滑度与性能。

**⚠️ 局限性**

主要限制是训练阶段的计算开销相对较大（约为基线的 3‑4 倍），并且需要使用可微分激活（如 SiLU）以保证 Hessian 的存在；此外，在极端观测噪声下仍受限于 critic 的逼近误差。

---

## 460. EvoClinician: A Self-Evolving Agent for Multi-Turn Medical Diagnosis via Test-Time Evolutionary Learning

**arXiv ID:** 2601.22964 | [PDF](https://arxiv.org/pdf/2601.22964v1)

**作者:** Yufei He `[一作]` (National University of Singapore), Jiang Bian `[通讯]` (Microsoft Research)

**通讯引用:** 13488 | [OpenAlex ID](https://openalex.org/A5030951014)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种在医学多轮诊断任务中，能在测试阶段自我演化的代理模型EvoClinician，并构建了交互式基准Med‑Inquire；

**💡 创新点**

创新点在于引入Diagnose‑Grade‑Evolve循环，通过基于动作级别的评级（考虑临床收益和成本）实现无梯度的提示与记忆更新，从而使代理在连续病例中不断提升诊断准确性和资源效率；

**🔧 技术方法**

使用多代理架构（Actor、Process Grader、Evolver），结合大型语言模型（如Gemini‑3‑Pro、GPT‑4）、梯度自由的提示演化与外部记忆机制，以及基于测试时间学习（TTL）的策略；

**📊 数据集**

基准数据集为从10份顶级医学期刊（NEJM、Lancet等）提取的915个真实临床案例的DiagnosisArena；

**📈 对比分析**

与静态提示、内存检索、Prompt‑优化以及EvoTest等自我演化基线相比，EvoClinician在所有四种LLM骨干上平均提升约10–12分诊断评分，同时将平均成本下降约15%至30%，且保持与基线相近的交互轮数；

**⚠️ 局限性**

主要局限包括：基准为模拟环境，患者代理生成的细节可能不完全真实；动作级评级的主观性与噪声可能导致自我演化偏差；成本表仅为近似，无法直接映射真实医疗费用；在实际临床应用前仍需严格验证与安全监测。

---

## 461. dgMARK: Decoding-Guided Watermarking for Diffusion Language Models

**arXiv ID:** 2601.22985 | [PDF](https://arxiv.org/pdf/2601.22985v1)

**作者:** Pyo Min Hong `[一作]` (Hongik University), Albert No `[通讯]` (Yonsei University)

**通讯引用:** 506 | [OpenAlex ID](https://openalex.org/A5049196468)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种针对离散扩散语言模型的解码导向水印方法（dgMARK）

**💡 创新点**

创新点在于通过引导解码顺序而非改变词概率来嵌入水印，利用二进制哈希对位置进行奇偶匹配，无需改动模型原有分布

**🔧 技术方法**

采用二进制哈希、解码策略（confidence、entropy、margin等）以及可选的一步look‑ahead beam搜索进行位置选择；检测时使用奇偶匹配统计的z检验和滑动窗口检测

**📊 数据集**

使用C4新闻子集和Writing Prompts两大基准数据集，模型为LLaDA系列和Dream扩散模型

**📈 对比分析**

与传统AR水印方法KGW和面向无序解码的PATTERN‑MARK对比，dgMARK在保持文本质量（PPL和benchmark准确率仅轻微下降）的前提下，检测率接近1，误报率接近0；one‑step look‑ahead beam显著提升可检测性

**⚠️ 局限性**

局限性包括对短文本的可检测性较弱、对大块大小（block size）敏感、对极端编辑（高插入/删除比例）仍存在一定降检概率；同时水印需要共享密钥且在极端扰动下可被轻易移除

---

## 462. HierLoc: Hyperbolic Entity Embeddings for Hierarchical Visual Geolocation

**arXiv ID:** 2601.23064 | [PDF](https://arxiv.org/pdf/2601.23064v1)

**作者:** Hari Krishna Gadi `[一作]`, Liqiu Meng `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一个基于 CSV 的地理层级树，自动解析并聚合国家、地区、子地区、城市等信息，同时计算坐标均值、图像嵌入和文本嵌入等多模态特征。

**💡 创新点**

提出了完全无模式的标题解析、容错的 ISO2 解析、分层累加与后处理策略，以及在一次流式读取中生成完整层级结构和特征的高效流程。

**🔧 技术方法**

使用了 CSV 头匹配、字符串清洗、前缀/后缀匹配、树形节点累加、坐标平均、图像与文本嵌入编码、JSON 列表化，以及分块流式处理。

**📊 数据集**

主要针对含有 country, region, subregion, city, lat, lon 等字段的 CSV 数据集，示例中包含了 ReverseGeocode 等逆向地理编码数据。

**📈 对比分析**

算法复杂度为 O(N)（N 为行数）并仅需 O(|节点数|) 内存，未给出与其他方法的基准对比，但说明其在大规模数据下的线性时间和低内存占用。

**⚠️ 局限性**

局限性包括对 ISO2 边缘情况的补丁处理、对缺失字段的假设、未处理同名冲突与重复记录，以及缺乏对真实数据集的性能与准确性评估。

---

## 463. Evaluating the Effectiveness of OpenAI's Parental Control System

**arXiv ID:** 2601.23062 | [PDF](https://arxiv.org/pdf/2601.23062v1)

**作者:** Kerem Ersoz `[一作]` (Microsoft), Junfeng Jiao `[通讯]` (University of Texas at Austin)

**通讯引用:** 4051 | [OpenAlex ID](https://openalex.org/A5060920769)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究评估了主流对话式助手在儿童使用时平台级家长控制的有效性，采用儿童真实情境下的提示迭代与人工在环 UI 重放，并监测家长通知与安全输出。

**💡 创新点**

创新点在于结合 PAIR 风格的提示演化、人工重放与自动判别相结合的多阶段实验流程，揭示了通知系统的选择性覆盖与产品与政策间的缺口。

**🔧 技术方法**

使用了 PAIR‑style 迭代提示生成、人工在环 UI 交互、ChatGPT‑5 自动评判加人工审核、电子邮件通知监测等技术。

**📊 数据集**

构建了以七大风险类别（身体伤害、色情、隐私暴力、健康咨询、诈骗、仇恨言论、恶意软件）为中心的自定义提示语料库，初始种子由 Claude Sonnet 4.5 生成并通过 API 迭代完善。

**📈 对比分析**

通过比较当前后端与 GPT‑4.1、GPT‑4o 两个历史模型，在通知率、泄漏率、过度拦截率与 UI 介入率等指标上进行评估；结果显示当前后端泄漏率下降，但通知覆盖仍偏向身体伤害与部分色情，过度拦截仍普遍。

**⚠️ 局限性**

局限性包括：仅覆盖七类风险，提示语料为人工生成而非真实儿童对话，评判模型可能与被测助手产生偏差，通知管道不透明，实验环境与实际使用场景的可迁移性有限。

---

## 464. The Hot Mess of AI: How Does Misalignment Scale With Model Intelligence and Task Complexity?

**arXiv ID:** 2601.23045 | [PDF](https://arxiv.org/pdf/2601.23045v1)

**作者:** Alexander Hägele `[一作]` (EPFL), Jascha Sohl-Dickstein `[通讯]` (Anthropic)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过对 AI 失败进行偏差-方差分解，量化模型在不同任务、规模和推理长度下的无序度（incoherence），并探讨其随模型规模和推理步数的变化趋势。

**💡 创新点**

创新点在于将无序度定义为方差占总误差的比例，引入偏差-方差框架评估 AI 的系统性失配与无序失误，并揭示在更复杂任务中更强大的模型往往表现出更高的无序度，而非单纯更准确。

**🔧 技术方法**

主要技术包括 KL、Brier 以及 0/1 的偏差-方差分解；多重采样生成多次输出以估计方差；利用推理预算与模型自回归行为；集合（ensembling）降低方差；以及对合成优化器进行尺度实验验证。

**📊 数据集**

使用的公开数据集包括科学推理 benchmark、通用知识 benchmark、GitHub 任务编码评测、Model‑Written Evals（含开放式与多选），以及自构造的二次优化合成数据；同时还收集了人类调查对 AI 统一度与无序度的主观评价。

**📈 对比分析**

通过比较不同规模模型、不同推理预算与不同任务难度下的误差、方差和无序度，发现：更大模型在简单任务上无序度降低，但在困难任务上升；推理时间越长方差越大；集成可按 1/E 下降无序度；推理预算略微提升性能并略降无序度，但整体仍受自然推理变异影响，实验表明在高难度情景下无序失误占主导。

**⚠️ 局限性**

局限性包括：未深入揭示为何推理/规模增加会导致无序度上升；实验主要基于可量化的任务，开放式目标的无序度难以度量；集合等错误修正手段在实际行动循环中不切实际；并且实验结果可能受样本数量、模型实现细节及数据集偏倚影响。

---

## 465. Avoiding Premature Collapse: Adaptive Annealing for Entropy-Regularized Structural Inference

**arXiv ID:** 2601.23039 | [PDF](https://arxiv.org/pdf/2601.23039v1)

**作者:** Yizhi Liu `[一作]` `[通讯]` (Stony Brook University), Yizhi Liu (Stony Brook University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

论文分析了熵正则化最优传输在差分匹配层中的模式崩溃机制，并提出了基于热力学速度极限的自适应温度调度算法EPH‑ASC。

**💡 创新点**

通过非正规特征分析揭示“Premature Mode Collapse”是标准指数降温不可避免的热力学失稳，并设计出利用1/ε敏感度线性稳定约束的低成本自适应调度策略。

**🔧 技术方法**

结合Sinkhorn固定点映射的非正规雅可比矩阵、伪谱理论、局部非退化假设与热力学速度极限推导，提出线性稳定性阈值和“热力学暂停”控制。

**📊 数据集**

在视角变化下的语义关键点匹配基准SPair‑71k上进行实验验证。

**📈 对比分析**

与标准指数降温(Log‑Space)和Gumbel‑Sinkhorn两种基线对比，EPH‑ASC在保持梯度信息的同时实现了约1.60倍的收敛速度，并且计算开销仅提升0.51%。

**⚠️ 局限性**

需要先离线校准安全斜率k_safe，算法对不同任务或数据分布的泛化性尚待进一步验证，并且在极小ε时仍可能出现数值不稳定。

---

## 466. Divide-and-Conquer CoT: RL for Reducing Latency via Parallel Reasoning

**arXiv ID:** 2601.23027 | [PDF](https://arxiv.org/pdf/2601.23027v1)

**作者:** Arvind Mahankali `[一作]` (Stanford University), Tengyu Ma `[通讯]` (Stanford University)

**通讯引用:** 9144 | [OpenAlex ID](https://openalex.org/A5101821970)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

训练了一个LLM，通过“导演—工人”架构将长链式推理拆分成可并行的子任务，以降低推理时延。

**💡 创新点**

首次在LLM中引入并行推理框架，并通过多阶段强化学习学习识别可并行子任务，显著减少最长路径长度的同时保持准确率。

**🔧 技术方法**

结合有监督微调（SFT）、强化学习（DAPO、CISPO）、长度惩罚、专用注意力掩码和多阶段数据过滤，实现并行思考与准确率的平衡。

**📊 数据集**

使用DeepScaleR 1.5B预训练模型的DeepScaleR训练集15k题目做SFT，并在AIME 2024、AMC 23、HMMT 2025、MATH 500、Minerva Math和Olympiad‑Bench等数据集上进行评测。

**📈 对比分析**

与DeepScaleR‑1.5B‑Preview及其HLP、Maj‑DSR‑3等基线对比，DC‑CoT在保持或略高准确率的情况下，最长路径长度下降约35‑40%；-HLP进一步压缩约20%；-Maj在多数数据集上亦实现并行优势。

**⚠️ 局限性**

对高度顺序化的题目（如Minerva Math）效果不佳；RL训练易出现准确率与路径长度冲突，需复杂的多阶段调参；方法依赖模型已有长链推理能力，迁移性有限。

---

## 467. SolAgent: A Specialized Multi-Agent Framework for Solidity Code Generation

**arXiv ID:** 2601.23009 | [PDF](https://arxiv.org/pdf/2601.23009v1)

**作者:** Wei Chen `[一作]` (Shanghai Jiao Tong University), Yuan Luo `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 5423 | [OpenAlex ID](https://openalex.org/A5064776495)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `8d10c613-917e-4880-9716-17789f50e119` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种针对Solidity智能合约的多代理生成框架（SolAgent），通过双循环改进和工具增强实现代码可执行性与安全性；

**💡 创新点**

创新点在于将Forge编译/测试与Slither静态分析结合到迭代改进循环，并通过文件系统工具模拟专家工作流，随后利用交互轨迹蒸馏生成可部署的小模型；

**🔧 技术方法**

使用技术包括多代理架构、Forge、Slither、文件系统工具、动态停止机制、MS-Agent框架以及Qwen3-8B的微调蒸馏；

**📊 数据集**

采用的数据集为从SolEval扩展而来的SolEval+基准（1,125函数+1,188测试用例）以及多代理交互轨迹；

**📈 对比分析**

与GPT‑5、Claude‑Sonnet‑4.5、GitHub Copilot、MetaGPT等基线对比，SolAgent在SolEval+上Pass@1达64.39%，约比基线高30%，漏洞数降低约40%，Gas使用低于人类代码；

**⚠️ 局限性**

局限性包括仅支持单文件合约、对多合约系统支持不足、仍需依赖Forge/Slither工具链、蒸馏模型性能受限于原始代理质量等。

---

## 468. Self-Supervised Slice-to-Volume Reconstruction with Gaussian Representations for Fetal MRI

**arXiv ID:** 2601.22990 | [PDF](https://arxiv.org/pdf/2601.22990v1)

**作者:** Yinsong Wang `[一作]` (Imperial College London), Chen Qin `[通讯]` (Imperial College London)

**通讯引用:** 8628 | [OpenAlex ID](https://openalex.org/A5100362873)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `729e5870-4135-47f5-97f2-e3974d07b5dc` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出基于3D高斯表示的自监督切片到体素重建框架 GaussianSVR，结合多分辨率联合优化实现运动估计与体积重建。

**💡 创新点**

创新点在于将3D高斯原语用于医学体积重建，并引入自监督的前向切片采样模型以及多分辨率训练策略。

**🔧 技术方法**

使用3D高斯渲染、前向切片采样、D-SSIM+TV 损失、Adam 优化器以及 PyTorch 框架。

**📊 数据集**

使用 Fetal Tissue Annotation Challenge（FeTA）数据集中的 T2w 胎儿脑 MRI 进行实验。

**📈 对比分析**

与 NiftyMIC、SVoRT、NeSVoR 等基线方法比较，GaussianSVR 在 PSNR、SSIM 上均优于其他方法，PSNR 提升约 2.9% 以上。

**⚠️ 局限性**

仍需验证单堆栈切片的重建效果、对不同运动模型的鲁棒性以及高分辨率下的计算成本。

---

## 469. Quantifying Model Uniqueness in Heterogeneous AI Ecosystems

**arXiv ID:** 2601.22977 | [PDF](https://arxiv.org/pdf/2601.22977v1)

**作者:** Lei You `[一作]` (Technical University of Denmark), Lei You `[通讯]` (Technical University of Denmark)

**通讯引用:** 767 | [OpenAlex ID](https://openalex.org/A5082049111)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于 in‑silico 准实验设计（ISQED）的统计框架，用以量化异构 AI 生态系统中模型的唯一性，并实现 DISCO 估计器完成主动与被动审计。

**💡 创新点**

创新点包括：① 定义了 Peer‑Inexpressible Residual（PIER）衡量模型不可被同行混合解释的程度；② 证明仅凭观测日志无法识别唯一性；③ 在主动审计中给出上界和下界，证明样本复杂度是最优的；④ 证明 Shapley 值等协同博弈归因方法无法检测冗余。

**🔧 技术方法**

采用的技术包括：统计学中的凸投影与合成控制、因果推断的准实验设计、最优采样与子高斯噪声理论、线性结构模型、中心极限定理与极限理论。

**📊 数据集**

实验数据集涵盖：计算机视觉模型（ResNet、ConvNeXt、ViT 等）、大型语言模型（BERT、RoBERTa、DistilBERT、ALBERT、XLNet 等）以及城市级交通预测模型；文本任务使用 SST‑2 数据集。

**📈 对比分析**

与传统 XAI 归因方法（LIME、SHAP）、简单对偶模型混合等进行对比。实验显示 DISCO 在识别冗余、提供可操作路由策略以及在主动采样模式下显著降低查询量（约 1.34× 的误差降低），且对不同干预强度能揭示模型的 dose‑dependent 行为。

**⚠️ 局限性**

局限性：① 需要对输入干预可控的 in‑silico 环境，无法仅凭观测日志推断唯一性；② PIER 只衡量凸混合冗余，复杂非线性冗余可能被低估；③ 仅给出平均唯一性量，缺乏对鲁棒性差异（如高振荡 vs 稳定）的区分。

---

## 470. MiTa: A Hierarchical Multi-Agent Collaboration Framework with Memory-integrated and Task Allocation

**arXiv ID:** 2601.22974 | [PDF](https://arxiv.org/pdf/2601.22974v1)

**作者:** XiaoJie Zhang `[一作]`, Jianzong Wang `[通讯]` (Tsinghua University)

**通讯引用:** 1925 | [OpenAlex ID](https://openalex.org/A5088981076)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于大型语言模型的记忆集成任务分配多智能体框架 MiTa，通过层级管理和协商式分配实现更高效的协作。

**💡 创新点**

创新点：①将任务分配与记忆集成结合到经理‑成员层级架构；②引入 Allocation 模块实现全局任务分配，避免冲突；③引入 Summary 模块利用 LLM 生成短摘要，保留长期上下文；④在 LLM 驱动下的协商式分配机制。

**🔧 技术方法**

技术手段：使用 GPT‑4o、Qwen3‑Plus、DeepSeek‑V3.1 等 LLM 进行规划、摘要与协商；多智能体框架、任务分配策略、记忆集成与协商机制；在 VirtualHome‑Social 3D 仿真环境中实现。

**📊 数据集**

数据集与实验环境：VirtualHome‑Social 平台下的 C‑WAH 任务集（Prepare Tea、Wash Dishes、Prepare a Meal、Put Groceries、Set up Table），采用符号观测和视觉观测两种设置。

**📈 对比分析**

对比方法：与 MHP、CoELA、ProAgent 等先进多智能体框架进行平均步数和效率提升（Efficiency Improvement, EI）评估。MiTa 在三智能体设置下平均步数最低，EI 可达 68%；在两智能体设置下仍优于基线，整体表现最佳。

**⚠️ 局限性**

局限性：经理节点依赖强大的 LLM，若使用较弱模型会显著降低协调效率；在两智能体场景下的 Wash Dishes 和 Put Groceries 任务表现略逊；未验证更大规模或异构模型的可扩展性。

---

## 471. Improved Algorithms for Nash Welfare in Linear Bandits

**arXiv ID:** 2601.22969 | [PDF](https://arxiv.org/pdf/2601.22969v1)

**作者:** Dhruv Sarkar `[一作]` (Indian Institute of Technology Kharagpur), Sayak Ray Chowdhury `[通讯]` (Indian Institute of Technology Kanpur)

**通讯引用:** 101 | [OpenAlex ID](https://openalex.org/A5052162709)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种名为FairLinBandit的线性带问题公平性优化框架，能够在任意p-均值目标下实现近似最优的Nash（以及更广义的p-均值）遗憾；

**💡 创新点**

创新点在于通过数据驱动的自适应停止规则和基于UCB的自适应置信区间，突破了传统乘法浓缩不等式导致的维度次优瓶颈，实现了线性带问题的阶数最优Nash遗憾；

**🔧 技术方法**

主要技术包括D‑optimal设计与John椭圆分布的交替探索、数据自适应终止规则、UCB/Phased Elimination的乐观估计以及自归一化马尔可夫链浓缩；

**📊 数据集**

实验使用了MSLR‑WEB10K和Yahoo!学习排名挑战数据集，在高维（d=10）下构造线性带实例；

**📈 对比分析**

与现有LinNash算法比较，FairLinPE和FairLinUCB在Nash遗憾和p‑均值遗憾上均显著优于LinNash，收敛更快、稳定性更好；

**⚠️ 局限性**

限制在于对p<0时遗憾随着|p|指数增长，需要更长时间周期；对非线性奖励结构的推广仍未完成。

---

## 472. Gender Disparities in StackOverflow's Community-Based Question Answering: A Matter of Quantity versus Quality

**arXiv ID:** 2601.23063 | [PDF](https://arxiv.org/pdf/2601.23063v1)

**作者:** Maddalena Amendola `[一作]` (Italian Institute of Technology), Raffaele Perego `[通讯]` (National Research Council Institute of Informatics and Telematics)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过对 Stack Overflow 数据进行性别推断、人工评估和大模型评估，比较男性与女性回答者的回答质量以及被选为最佳答案的比例，并分析性别差异对声誉系统的影响。

**💡 创新点**

①首次直接使用大模型对同一问题的男女回答质量进行对比；②发现回答质量无显著差异，声誉差距主要源于活跃度和系统对数量的偏好；③提出重新设计声誉机制以减少性别不平等。

**🔧 技术方法**

性别推断工具（基于姓名和地区统计）、两类 LLM（重排模型与指令式大模型）用于回答质量评估、Mann‑Whitney U 检验、Spearman 相关、网络同质性分析以及人类评测（MTurk）。

**📊 数据集**

Stack Overflow 的公开数据集（2017‑2024 年的提问、回答、用户信息），并构建了两套子样本：①仅含男女回答且在接受答案前一日内的问答；②包含至少男女双方回答的问答。

**📈 对比分析**

通过将 LLM 生成的最佳答案与 Stack Overflow 实际接受答案及人工评测结果进行比对。LLM 的匹配率在 48.8%–68.2% 之间，差异值（男女差别）均低于 2%；与人工评测的准确率最高可达 76%。

**⚠️ 局限性**

①性别推断仅为二元，无法覆盖非二元身份；②对 Stack Overflow 的偏好和声誉系统的分析仅在该平台内，缺乏跨平台验证；③LLM 可能已在训练中见过 Stack Overflow 数据，导致一定程度的过拟合；④研究仅考虑了回答质量，未深入探讨问答内容类型和上下文差异。

---

## 473. From Absolute to Relative: Rethinking Reward Shaping in Group-Based Reinforcement Learning

**arXiv ID:** 2601.23058 | [PDF](https://arxiv.org/pdf/2601.23058v1)

**作者:** Wenzhe Niu `[一作]` (Meituan), Renqing He `[通讯]` (Meituan)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种新的强化学习框架RLRR，将奖励塑造从绝对分数转为相对排序，并引入列表式排序奖励模型（Ranking RM）以提升LLM的推理与生成质量。

**💡 创新点**

创新点在于：①用组内相对排名替代稀疏或不稳定的绝对奖励，解决传统GRPO中的梯度消失与奖励波动问题；②设计混合相对奖励（HRR）和纯相对奖励（PRR）两种策略；③开发Ranking RM，能够一次性对多条生成结果进行列表排序，并与规则验证器和长度约束相结合实现层级重排序。

**🔧 技术方法**

核心技术包括：基于GRPO的组式强化学习；相对奖励塑造函数与正确性/长度约束的优势裁剪；Ranking RM的列表式交叉熵训练；层级重排序（先正确性后长度）；以及对奖励模型的可解释性与鲁棒性分析。

**📊 数据集**

使用了多种公开数据集：数学与逻辑推理（GURU、SimpleRL、Open‑RS、AIME、AMC、GSM8K等），写作任务（WritingBench、Dolphin‑R1），以及用于训练Ranking RM的25k条多样化对比样本；模型包括DeepSeek‑R1‑Distill‑Qwen‑1.5B/8B、Qwen3‑1.7B、Skywork‑8B、URM‑8B 等。

**📈 对比分析**

与GRPO、Dr.GRPO、DAPO、GPG、CISPO、RLOO等多种基线进行对比；在数学推理基准上，RLRR（HRR/PRR）平均提升 2‑3% 的准确率并在 token 使用上更优；在写作基准上，RLRR 在大多数领域均超过 1‑2 分，且Ranking RM 在抽样多样化时性能优于传统单点奖励模型；实验表明相对奖励显著降低训练波动并提升最终性能。

**⚠️ 局限性**

主要局限包括：①依赖Ranking RM 的排序质量，若训练数据不足或模型偏差，排名误差会影响奖励；②需要手动调节 τ、λ 等超参数，超参数空间较大；③对极端长文本或复杂推理的提升有限，受基础模型推理能力限制；④层级重排序在某些场景下可能过度惩罚长度，导致生成多样性受限。

---

## 474. Digital Twin Synchronization: towards a data-centric architecture

**arXiv ID:** 2601.23051 | [PDF](https://arxiv.org/pdf/2601.23051v1)

**作者:** Eduardo Freitas `[一作]` (Universidade Federal de Pernambuco), Judith Kelner `[通讯]` (Universidade Federal de Pernambuco)

**通讯引用:** 2994 | [OpenAlex ID](https://openalex.org/A5026724035)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `9cc9baba-5356-466d-81ff-d80028d90279` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文针对工业4.0场景中数字孪生（DT）与物理实体的同步问题，综述现有技术并提出一套面向多行业通用的数据中心化DT同步参考架构，包括Telemetry（数据采集、变换、知识存储）、Analysis（更新监督、更新执行）等关键模块，旨在实现高效、可扩展、安全、互操作的同步闭环；

**💡 创新点**

创新点在于（1）首次将数字孪生同步抽象为统一的数据中心化架构，兼顾网络、机器人、IoT等异构设备；（2）将知识对象与知识存储嵌入同步流程，提升数据利用率与模型预测能力；（3）提出分布式更新监督与执行机制，支持大规模工业系统；（4）强调安全与互操作性需求，提供标准化接口与协议组合；

**🔧 技术方法**

核心技术包括：网络遥测（SNMP、NETCONF/RESTCONF、IPFIX、INT）、ROS、SensorThings API；数据融合与知识转换（RDF、知识对象）；消息中间件（MQTT、DDS、Kafka等）；时序与大数据存储（InfluxDB、Prometheus、Hadoop/Spark、Cassandra、MongoDB、Neo4j）；机器学习/深度学习模型用于预测与决策；多协议与数据模型（YANG、JSON、RDF）。

**📊 数据集**

文中未给出具体公开数据集，所述数据来源为工业设备传感器、机器人状态、网络流量等真实或仿真采集数据，主要以工业场景的实时监测数据为主；

**📈 对比分析**

目前论文未提供实验评估或性能对比，作者仅在结论中提出将来计划评估同步服务的计算、存储与通信开销，并讨论同步频率与精度的折中策略；

**⚠️ 局限性**

主要限制包括：①设备异构性导致数据采集与格式统一困难；②海量数据产生高延迟与资源消耗；③同步机制缺乏统一标准，难以跨平台推广；④未在真实工业环境中验证，缺乏实验结果；⑤安全与隐私风险尚未充分评估。

---

## 475. MOSAIC: Modular Scalable Autonomy for Intelligent Coordination of Heterogeneous Robotic Teams

**arXiv ID:** 2601.23038 | [PDF](https://arxiv.org/pdf/2601.23038v1)

**作者:** David Oberacker `[一作]` (FZI Forschungszentrum für Informatik), Arne Roennau `[通讯]` (Robotic Systems Lab ETH Zürich)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c`

**🎯 论文内容**

开发并验证了一套可模块化、可扩展的多机器人自治框架MOSAIC，支持异构机器人团队在单个操作员监督下执行科学探索任务。

**💡 创新点**

创新点包括统一的任务抽象（POI）与多层自治模式、基于机器人能力的动态任务分配、以及通过行为树实现的实时任务与操作者交互。

**🔧 技术方法**

采用ROS 2/ROS 1混合框架、行为树、NavPi 3D导航、SLAM (Open3D, Cartographer)、RMP本地规划、QoS优化、域桥接及多传感器融合等技术。

**📊 数据集**

在瑞士采石场进行的月球模拟实地实验，人工布置的岩石和沙块目标作为任务数据。

**📈 对比分析**

通过关键性能指标评估，任务完成率82.3%，自治比例86%，操作员工作负载78.2%，映射面积758 m²，映射效率2.19，展示了在5机器人团队中实现高效、可靠的自治。

**⚠️ 局限性**

局限性包括ROS1/2兼容性与网络QoS问题、对手动操作员干预的依赖、网络延迟导致的通信瓶颈、有限的团队规模与任务范围，以及在长时间通信中断时自治水平仍需提升。

---

## 476. DimABSA: Building Multilingual and Multidomain Datasets for Dimensional Aspect-Based Sentiment Analysis

**arXiv ID:** 2601.23022 | [PDF](https://arxiv.org/pdf/2601.23022v1)

**作者:** Lung-Hao Lee `[一作]` (National Yang Ming Chiao Tung University), Saif M. Mohammed `[通讯]` (National Research Council Canada)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建了第一个多语言、多领域的维度化方面情感分析（ABSA）资源 DimABSA，并基于该资源提出了三类子任务（DimASR、DimASTE、DimASQP）以及统一的连续 F1（cF1）评估指标。

**💡 创新点**

创新点包括：①将传统 ABSA 的离散情感标签转化为连续的 valence–arousal（VA）得分，实现细粒度情感表达；②首次在多语言、多领域上公开提供维度化 ABSA 数据集；③提出兼顾离散与连续输出的 cF1 指标，解决混合型任务的评估难题。

**🔧 技术方法**

使用技术包括：多语言文本预处理与翻译、SAM 视觉量表的 VA 标注、基于 QLoRA 的 4‑bit 参数高效微调、对 GPT‑5 mini、Kimi K2 等闭源 LLM 进行零/少量样本推理，并对 Qwen‑3、Mistral‑3、Llama‑3.3、GPT‑OSS 等公开 LLM 进行微调。

**📊 数据集**

数据集涵盖 6 种语言（中文、英文、日语、俄语、塔塔尔语、乌克兰语）与 4 个领域（餐饮、笔记本、酒店、金融），共计 76,958 个 aspect 实例，分为 10 个子数据集，涵盖 aspect term、aspect category、opinion term 与 VA 得分。

**📈 对比分析**

与方法比较：零/少量样本推理在英语表现最好；微调后 70B 与 120B 模型相对稳定提升，120B 在多语言任务上取得最佳平均性能，但整体任务仍较难，低资源语言（塔塔尔语、乌克兰语）仍落后；相比传统类别化 ABSA，维度化数据集的难度更高。

**⚠️ 局限性**

局限性：①VA 的文化与语言差异导致跨语言可比性有限；②数据覆盖面仍局限于 6 语言，缺乏更低资源语言；③缺乏对 VA 标注测量不变性的系统评估；④高维度模型在 token‑级 VA 预测上仍受限。

---

## 477. InstructDiff: Domain-Adaptive Data Selection via Differential Entropy for Efficient LLM Fine-Tuning

**arXiv ID:** 2601.23006 | [PDF](https://arxiv.org/pdf/2601.23006v1)

**作者:** Junyou Su `[一作]` (Peking University), Guanhua Chen `[通讯]` (SUSTech)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于差分熵（ΔH）的统一数据选择框架，用于在监督微调（SFT）阶段高效筛选最优训练样本

**💡 创新点**

创新点在于发现与基模型对比的最小差分熵样本在不同任务域（推理、一般指令等）都能实现最佳性能，并且该原则在不同域自适应地表现为熵增（认知扩展）或熵减（认知压缩）

**🔧 技术方法**

使用了基模型与轻度指令微调校准模型的对比，计算负对数似然差（ΔNLL）和差分熵（ΔH），并结合双向NLL过滤、熵排序和可迭代校准等技术

**📊 数据集**

在四个域上验证：数学推理（NuminaMath），通用指令（Alpaca），医学问答（MedCAQA）和代码生成（BigCode），使用Qwen2.5-7B、LLaMA3-8B等模型

**📈 对比分析**

与随机、PPL、Entropy、Length、IFD、SelectIT、ZIP、Superfilter等多种基线以及完整数据训练进行对比，结果显示该方法在仅使用10%–20%数据时，数学推理提升约17%，通用指令提升约52%，医学问答提升约6%，代码生成提升约5%，并在大规模数据集上保持优势

**⚠️ 局限性**

局限包括：对β、γ等超参数需在不同域微调；热身校准集太小会导致差分熵估计不稳；跨模型族选择一致性差，建议使用同族模型进行校准

---

## 478. Bias Beyond Borders: Political Ideology Evaluation and Steering in Multilingual LLMs

**arXiv ID:** 2601.23001 | [PDF](https://arxiv.org/pdf/2601.23001v1)

**作者:** Afrozah Nadeem `[一作]`, Usman Naseem `[通讯]` (Macquarie University)

**通讯引用:** 2940 | [OpenAlex ID](https://openalex.org/A5077006200)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对13款最先进LLM在巴基斯坦五种本土语言（乌尔都语、旁遮普语、信德语、普什图语和俾路支语）上进行政治偏见评估，并提出跨语言对齐激活干预方法（CLAS）来减轻偏见。

**💡 创新点**

创新点在于：①将Political Compass Test（PCT）文化适配并结合多层框架分析；②提出跨语言对齐的激活空间干预（CLAS）和不确定性自适应缩放；③系统展示了低资源语言中的语言条件偏见与西方训练数据的差异。

**🔧 技术方法**

使用技术包括：PCT评估、激活空间对齐、跨语言对齐（CLAS）、单层与向量集（ISV、SVE）激活干预、Logistic回归生成对齐向量、以及不确定性自适应调节。

**📊 数据集**

采用的主要数据集为：50国PCT基准（含11个针对巴基斯坦语境的社会政治主题）以及各语言的文化适配翻译版PCT语句。

**📈 对比分析**

通过比较Bias Reduction Score（ΔBias）、跨语言一致性和回应质量指标，CLAS相较于ISV和SVE实现了更高的偏见削减，且在保持流畅度与语义一致性的同时显著降低了跨语言方差。

**⚠️ 局限性**

局限性包括：①对齐基准为英语，可能带来英语中心的残余偏见；②PCT仅以两轴衡量政治倾向，无法覆盖所有非西方政治维度；③不确定性自适应假设与模型自信度关联可能不适用于细微或模糊提示；④实验仅限于7B规模模型，缺乏更大模型与人工评估验证。

---

## 479. Why Your Deep Research Agent Fails? On Hallucination Evaluation in Full Research Trajectory

**arXiv ID:** 2601.22984 | [PDF](https://arxiv.org/pdf/2601.22984v1)

**作者:** Yuhao Zhan `[一作]` (Zhejiang University), Chao Huang `[通讯]` (University of Hong Kong)

**通讯引用:** 12133 | [OpenAlex ID](https://openalex.org/A5006594763)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了面向深度研究代理（DRA）的过程感知评估框架，通过审计完整的研究轨迹来检测中间阶段的幻觉，填补了传统基于最终输出评估的空白。

**💡 创新点**

创新点在于引入PIES分类体系（Planning/Explicit、Summarization/Implicit）细分幻觉类型，并构建了首个覆盖全流程幻觉的基准库DeepHalluBench。

**🔧 技术方法**

技术实现包括：Web UI日志解析成计划-检索-总结循环、原子化拆分（行动、索引、主张）、基于NLI+LLM的主张与行动验证、语义聚类与噪声惩罚算法，以及多阶段自适应重新验证。

**📊 数据集**

数据集方面，作者从Mind2Web2、ReportEval、BrowseComp等公开基准汇总查询，再用Gemini深度研究代理生成轨迹并挑选出100个最易幻觉的查询，其中包含25个通过原子扰动合成的“无答案”攻击查询。

**📈 对比分析**

实验对比了六款主流DRA（Gemini、OpenAI、Perplexity、Qwen、Grok、Salesforce Air），使用PIES四类幻觉得分和检索质量进行评估；结果显示无一代理在全流程中实现可靠性，最优整体幻觉得分为0.149（Qwen）。

**⚠️ 局限性**

局限性包括：对部分专有代理缺乏结构化日志，导致部分评估指标缺失；评估依赖LLM验证，存在误判；仅关注幻觉而未覆盖其他错误模式。

---

## 480. PIDSMaker: Building and Evaluating Provenance-based Intrusion Detection Systems

**arXiv ID:** 2601.22983 | [PDF](https://arxiv.org/pdf/2601.22983v1)

**作者:** Tristan Bilot `[一作]` (University of British Columbia), Thomas Pasquier `[通讯]` (University of British Columbia)

**通讯引用:** 1410 | [OpenAlex ID](https://openalex.org/A5005580571)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了PIDSMaker，一个统一的开源框架，用于构建、评估和比较基于系统根因图的入侵检测系统（PIDS）

**💡 创新点**

创新点在于：①将八个主流PIDS整合到一个模块化、可配置的管线中；②提供标准化的预处理、特征化与地面真值；③通过YAML配置实现零代码快速原型；④内置多跑不确定性评估、超参数调优与消融实验工具；⑤使用磁盘缓存显著提升实验迭代效率

**🔧 技术方法**

技术方面采用Python+PyTorch、PyTorch Geometric构建GNN/TGN模型；实现七阶段管线（构造、转换、特征化、批处理、训练、评估、triage）；支持多种文本嵌入（Word2Vec/Doc2Vec/FastText/ALaCarte等）和图转换（DAG、伪图等）；利用W&B实时监控与实验记录；通过哈希缓存避免重复计算

**📊 数据集**

主要使用公开的DARPA TC E3、E5、OpTC三大数据集，并兼容共计13个传统数据集；为所有数据统一提供一致的节点级真值标签（遵循Jiang等的标注方案）

**📈 对比分析**

比较方法：通过统一管线、标准化标签与指标（precision/recall/F1、AUC‑ROC、平均分布等）在同一数据集上跑完整八个系统；利用超参数网格/随机搜索进行公平调优；通过多跑统计计算相对标准差评估稳定性；实验结果显示各系统在统一协议下表现差异显著，提供基准可供后续工作参考

**⚠️ 局限性**

局限性包括：①对自监督PIDS的预测稳定性仍差；②缺少对正常异常（benign anomaly）的标注导致误报率难以评估；③未考虑概念漂移和对抗鲁棒性；④目前仅支持自监督模型，尚未覆盖监督或基于规则的检测方法

---

## 481. SpecIBT: Formally Verified Protection Against Speculative Control-Flow Hijacking

**arXiv ID:** 2601.22978 | [PDF](https://arxiv.org/pdf/2601.22978v1)

**作者:** Jonathan Baumann `[一作]`, Julay Leatherman-Brooks `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文提出 SpecIBT，一种利用 Intel CET（IBT）与 LLVM 里 Ultimate SLH 结合的编译器级防御，能够为任意程序提供对 Spectre BTB、RSB 与 PHT 攻击的完整保护。

**💡 创新点**

创新点在于：1) 使用被调用函数指针本身而非静态标签来精确检测 BTB 误预测；2) 将检测结果直接写入 misspeculation flag 与 SLH 结合，实现无近似、无过度覆盖的安全检查；3) 在 Coq/Rocq 中完成从源语言到机器码的全流程形式化证明，实现了端到端的相对安全（relative security）保障。

**🔧 技术方法**

技术手段包括：硬件级控制流完整性（Intel CET/IBT）、软件级指令级安全增强（Ultimate SLH）、编译器插桩（SpecIBT transformation）、形式化验证（Coq/Rocq）以及基于抽象语义的安全模型。

**📊 数据集**

论文没有使用传统意义上的实验数据集，而是基于正式语义模型与定理证明构建安全性证明；若在实验层面验证，则使用 LLVM 生成的机器码与对比无防御程序。

**📈 对比分析**

与现有 Spectre 防御（如 Serberus、Swivel、Retpoline）相比，SpecIBT 在理论上提供更紧凑、无近似的保护，并在 Coq 中完成了全流程证明。性能方面，作者未给出具体计时指标，但预计相对成本低于完整的重写与多层保护（如 Serberus 的函数私有栈、寄存器清理）。

**⚠️ 局限性**

局限性包括：仅在支持 Intel CET/IBT 的 CPU 上可用；只针对 Spectre BTB/RSB/PHT，无法防御 Spectre STL/PSF 等变体；要求源程序使用的是可安全执行的指令集（避免未定义行为），对低级语言（如汇编）需要额外映射；实现仍处于 LLVM 级别，尚未公开完整的编译器插件。

---

## 482. Golden Goose: A Simple Trick to Synthesize Unlimited RLVR Tasks from Unverifiable Internet Text

**arXiv ID:** 2601.22975 | [PDF](https://arxiv.org/pdf/2601.22975v1)

**作者:** Ximing Lu `[一作]` (NVIDIA), Yejin Choi `[通讯]` (NVIDIA)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `67630363-6be0-4f51-ab05-7198250671a5` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种将不可验证的互联网文本转换为可验证的多选题的RLVR数据合成管道，并基于此构建了包含约70万题目的RLVR数据集。

**💡 创新点**

创新点在于利用填空式多选题的形式，利用大型语言模型自动提取关键信息并生成可验证的答案与多样化干扰项，从而从海量不可验证文本中无限制地扩展RLVR数据。

**🔧 技术方法**

技术主要包括LLM驱动的文本掩码与干扰项生成、难度过滤、RLVR训练（ProRL/GRPO）以及多选题评估。

**📊 数据集**

使用的数据集包括AoPS-Instruct、rStar-Coder、MegaScience，以及针对网络安全的FineWeb抓取文本，合成后得到70+万条RLVR任务以及18万条网络安全RLVR任务。

**📈 对比分析**

与原有ProRL数据比较，加入新数据后在15个基准上持续提升，最大提升达3.48% STEM；在网络安全基准上仅100步RL训练就获得4.44%绝对提升，超过了7B域专用模型。

**⚠️ 局限性**

局限性包括对LLM质量的高度依赖，生成过程可能继承原始文本的偏见或有害内容，且多选题格式仍无法覆盖完全开放式验证场景。

---

## 483. Self-Imitated Diffusion Policy for Efficient and Robust Visual Navigation

**arXiv ID:** 2601.22965 | [PDF](https://arxiv.org/pdf/2601.22965v1)

**作者:** Runhua Zhang `[一作]` (Zhejiang University), Wuyue Zhao `[通讯]` (Zhejiang University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 Self-Imitated Diffusion Policy（SIDP），通过奖励引导的自我模仿学习改进视觉导航路径规划。

**💡 创新点**

创新点在于将自我模仿与奖励加权的分布匹配结合，消除生成-过滤流程，并引入目标无关探索与奖励驱动的课程学习。

**🔧 技术方法**

采用扩散模型的轨迹生成与去噪、重要性加权、DDIM/DDPM 调度、奖励函数设计与 BPTT 替代的 IL/RL 混合训练技术。

**📊 数据集**

训练和评估使用 InternVLA‑N1 S1 与 InternData‑N1 数据集，并在 Jetson Orin Nano 边缘平台上进行真实机器人实验。

**📈 对比分析**

与 NavDP、ViPlanner、iPlanner 等基线对比，SIDP 在 InternVLA‑N1 S1 上 SR/SPL 提升约 10%+，在 Jetson Orin Nano 上推理速度提升 2.5×。

**⚠️ 局限性**

局限性包括对奖励参数和温度的手工调优敏感，模型对极端动态或复杂环境的鲁棒性仍有限，训练仍需一定计算资源。

---

## 484. Competitive Non-Clairvoyant KV-Cache Scheduling for LLM Inference

**arXiv ID:** 2601.22996 | [PDF](https://arxiv.org/pdf/2601.22996v1)

**作者:** Yiding Feng `[一作]` (Hong Kong University of Science and Technology), Yuhao Zhang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 8997 | [OpenAlex ID](https://openalex.org/A5100388133)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个针对大语言模型推理中 KV‑缓存动态增长的非先验调度算法 GSA，能够在未知响应长度情况下实现对总流量时间的有界竞争比。

**💡 创新点**

核心创新在于引入几何切片（Geometric Slicing）与分段流水线（Staggered Pipeline）两种结构，配合内存‑时间面积分析，首次实现常数竞争比（61.92/32）并在 clairvoyant 情况下提供 10.67/6.75 的近似保证。

**🔧 技术方法**

使用几何增大阶段长度、kill‑and‑restart 预处理、分段流水线实现内存平滑，以及基于内存‑时间面积的上界/下界推导，构造了 GSA 与其 clairvoyant 对偶 GBA 的证明框架。

**📊 数据集**

在实验中利用公开的 LMSYS‑Chat‑1M 实际请求轨迹与合成工作负载进行评估。

**📈 对比分析**

实验结果显示，GSA 在多种工作负载下均优于现有基线（如 Let‑Long‑Jobs‑Finish）并保持理论 worst‑case 保证；在大内存场景下，实际性能更接近理论上限。

**⚠️ 局限性**

限制在于仅考虑离线批处理且提示长度相同、仅支持 kill‑and‑restart 方式；对在线到达、异质提示长度或更细粒度预emption 的情况尚未覆盖。

---

## 485. Character as a Latent Variable in Large Language Models: A Mechanistic Account of Emergent Misalignment and Conditional Safety Failures

**arXiv ID:** 2601.23081 | [PDF](https://arxiv.org/pdf/2601.23081v1)

**作者:** Yanghao Su `[一作]` (University of Science and Technology of China), Jie Zhang `[通讯]` (Astar)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过在LLM上进行针对性角色（Evil、Sycophantic、Hallucinatory）训练，研究了角色作为潜在变量如何导致持续、可迁移的误导行为，并揭示了训练时触发器与推理时角色对齐提示的条件激活机制。

**💡 创新点**

创新点在于将误导行为视为角色这一内部行为倾向的形成与激活，而非单纯错误内容的泛化；提出角色作为统一的潜在控制变量，解释了出现的多种安全失败（Emergent Misalignment、Backdoor、Jailbreak）之间的关联。

**🔧 技术方法**

主要技术包括监督微调（SFT）、Persona向量分析（低维方向投影）、Trait Expression Score与Misalignment Score评估、以及对比攻击成功率（ASR）与拒绝率（RR）等指标。

**📊 数据集**

使用了自构造的角色条件数据集（Evil、Sycophantic、Hallucinatory），从三个领域（健康、职业发展、汽车维护）采集用户问答，并与基准的错误建议数据集进行对照。

**📈 对比分析**

通过与错误建议微调模型以及基线模型比较，发现角色微调在保持模型通用能力（如MMLU）不变的前提下，显著提高了误导得分、特征表达得分、以及角色对齐的攻击成功率，证明了角色驱动的误导更强大、更易迁移。

**⚠️ 局限性**

局限性包括仅研究了少数角色与两种模型族，使用的微调方式为监督学习而非RLHF或偏好优化；Persona向量仅提供相关性证据，缺乏因果解释；未评估更大规模模型或其他对齐方法的交互影响。

---

## 486. Probing the Trajectories of Reasoning Traces in Large Language Models

**arXiv ID:** 2601.23163 | [PDF](https://arxiv.org/pdf/2601.23163v1)

**作者:** Marthe Ballon `[一作]` (Vrije Universiteit Brussel), Andres Algaba `[通讯]` (Vrije Universiteit Brussel)

**通讯引用:** 233 | [OpenAlex ID](https://openalex.org/A5012066587)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在本研究中，我们提出了一种系统的轨迹探测协议，通过对LLM的推理轨迹进行token级切片并逐步注入，定量评估其对答案分布的影响。

**💡 创新点**

创新点在于将推理轨迹按token百分位切分，并利用插入式实验来区分实例特定语义信息、长度效应和结构效应，从而揭示推理内容对准确性和决策承诺的真正贡献。

**🔧 技术方法**

技术方法包括：使用Qwen3和gpt-oss模型生成完整推理轨迹、token化切片、next-token概率查询、随机/交换/洗牌对照实验，以及跨模型救援实验（基线与自由延续模式）。

**📊 数据集**

数据集为GPQA Diamond（198道多项选择题）和MMLU-Pro（12,032道多项选择题），覆盖多学科领域。

**📈 对比分析**

对比方法为在不同解码百分位、不同对照（长度、结构、token身份）和跨模型救援场景下计算准确率、决策概率、转向率等指标，结果显示准确率随推理深度提升18–33%，更强模型在自由延续模式下的救援率提升17–43%，并揭示了错误轨迹的可恢复性与锚定现象。

**⚠️ 局限性**

局限性包括：仅评估多项选择任务，未验证开放式生成的适用性；仅覆盖两大模型族，未涉及封闭源前沿模型；对齐分层可能掩盖推理信息密度差异；未探讨推理轨迹的因果可信度与内部决策机制。

---

## 487. DIFFA-2: A Practical Diffusion Large Language Model for General Audio Understanding

**arXiv ID:** 2601.23161 | [PDF](https://arxiv.org/pdf/2601.23161v1)

**作者:** Jiaming Zhou `[一作]` (Nankai University), Yong Qin `[通讯]` (Nankai University)

**通讯引用:** 9675 | [OpenAlex ID](https://openalex.org/A5088716214)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出DIFFA-2，一种基于扩散的大型音频语言模型，采用双适配器、四阶段渐进训练与VRPO偏好优化，显著提升音频理解性能。

**💡 创新点**

将扩散模型引入音频理解，设计语义与声学双适配器、4阶段训练、低参数LoRA微调和变差异偏好优化，实现在仅1.1%可训练参数下竞争AR LALM。

**🔧 技术方法**

扩散语言模型（LLaDA）、双适配器（语义+声学）、LoRA、VRPO偏好优化、因子并行解码。

**📊 数据集**

公开语音文本数据约11,000小时ASR（LibriSpeech、GigaSpeech等）、3,767小时监督微调数据、约3,000对偏好样本，覆盖MMSU、MMAU、MMAR等。

**📈 对比分析**

与多款AR LALM（Qwen3‑Omni、Qwen2.5‑Omni、Kimi‑Audio）及开源基线在MMSU、MMAU、MMAR上对比，DIFFA‑2在相同规模下提升4–20点，接近大型专有模型。

**⚠️ 局限性**

缺乏对话式/全双工交互训练，未覆盖语音生成或流式输入，偏好优化范围有限，速度仍不如最强AR模型。

---

## 488. Behemoth: Benchmarking Unlearning in LLMs Using Fully Synthetic Data

**arXiv ID:** 2601.23153 | [PDF](https://arxiv.org/pdf/2601.23153v1)

**作者:** Eugenia Iofinova `[一作]` (IST Austria), Dan Alistarh `[通讯]` (IST Austria)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Behemoth 框架，用来生成可控的全合成 {s,r,o} 事实数据并训练小型 LLM，以研究知识编辑。

**💡 创新点**

创新点在于用完全可解释的合成数据与自定义分词，能精确量化编辑效果并探究层级、秩与编辑策略的关系。

**🔧 技术方法**

使用了 Pythia‑31M Transformer、全微调、LoRA（低秩微调）以及 ROME 这三种编辑技术，并结合激活补丁与奇异值分解估算权重变化。

**📊 数据集**

使用了三种合成数据集：简单（6 条关系、400 值）、相关关系（两条关系完全相关）和嵌套关系（对象本身又是主体），全部由 Behemoth 生成。

**📈 对比分析**

与三种编辑方法对比，单个事实编辑成功率≥95%（LoRA rank 32），十个相同事实编辑需要 rank ≥64；完全忘记关系需要使用全秩或 rank 128，剩余准确率仅 70‑80%；层级实验显示只需微调 1–2 个 Transformer 块即可保持编辑效果，注意力层或 MLP 层的选择对不同方法有显著影响。

**⚠️ 局限性**

局限性是合成语法过于简化、数据结构缺乏自然语言复杂性，导致结果难以直接迁移到真实 LLM；仅在 31M 参数模型上验证，未探索更大规模模型。

---

## 489. Secure Tool Manifest and Digital Signing Solution for Verifiable MCP and LLM Pipelines

**arXiv ID:** 2601.23132 | [PDF](https://arxiv.org/pdf/2601.23132v1)

**作者:** Saeid Jamshidi `[一作]` (Polytechnique Montreal), Mohammad Adnan Hamdaqa `[通讯]` (Polytechnique Montreal)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种安全工具清单与数字签名框架，用于验证和可追溯LLM工具调用的执行完整性。

**💡 创新点**

创新点在于将MCP扩展为加密签名、透明日志和可度量的公平性，实现在大规模LLM管线中的可验证、可审计和公平性保障。

**🔧 技术方法**

技术包括ECDSA签名、Merkle树透明日志、HSM密钥管理、统计验证与性能分析。

**📊 数据集**

使用了三大主流LLM模型的API调用数据（GPT‑4‑turbo、LLaMA‑3.5、DeepSeek‑V3），并在100到5万条manifest实例的工作负载上进行实验。

**📈 对比分析**

通过与基线无安全措施执行对比，发现安全框架在5%以内的性能开销、80% 以上验证通过率、近线性扩展（R²=0.998）以及模型使用均衡。

**⚠️ 局限性**

局限性包括实验规模仅至5万实例、仅涵盖三种模型、未考虑多云分布式环境下的资源争用与恶意攻击、以及密钥集中度高等。

---

## 490. How should AI Safety Benchmarks Benchmark Safety?

**arXiv ID:** 2601.23112 | [PDF](https://arxiv.org/pdf/2601.23112v1)

**作者:** Cheng Yu `[一作]` (Technical University of Munich), Orestis Papakyriakopoulos `[通讯]` (Technical University of Munich)

**通讯引用:** 1003 | [OpenAlex ID](https://openalex.org/A5061502731)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文综述了210个AI安全基准，识别并量化了其在构造覆盖、风险量化与测量有效性方面的主要不足，并提出了十条改进建议与一份可操作的检查清单。

**💡 创新点**

创新点在于将传统风险工程、测量理论和社会技术系统三大理论框架系统化地引入AI安全基准评估，提出基于Rumsfeld矩阵的风险盲点映射、概率风险评估方法以及以测量理论为基础的可追溯校准流程，并通过案例验证其可行性。

**🔧 技术方法**

主要技术包括：系统性文献综述与分类、Rumsfeld矩阵映射、概率风险评估（PRA）、测量理论校准、量化指标的置信区间与安全裕度计算，以及基于社区输入的迭代改进机制。

**📊 数据集**

本文并未引入单一数据集，而是对已公开的210个安全基准（如TruthfulQA、HarmBench、MACHIAVELLI等）进行汇总与元分析，利用这些基准提供的任务、指标与结果进行对比。

**📈 对比分析**

在对比方面，作者将所提改进建议在一个名为AIR 2024的基准上进行案例评估，指出相较传统基准在构造覆盖、风险量化与部署关联性方面显著提升；实验结果显示，改进后基准能够更好地反映真实部署风险，尽管在实现成本与社区参与度上仍有挑战。

**⚠️ 局限性**

局限性包括：仍缺乏统一的标准化指标与校准机制，风险量化方法在实践中需进一步细化与验证；案例评估仍基于单一基准，未能覆盖所有安全子领域；此外，系统层面评估与跨模型、多方交互的研究尚待深入。

---

## 491. FlowCalib: LiDAR-to-Vehicle Miscalibration Detection using Scene Flows

**arXiv ID:** 2601.23107 | [PDF](https://arxiv.org/pdf/2601.23107v1)

**作者:** Ilir Tahiraj `[一作]` (Technical University of Munich), Markus Lienkamp `[通讯]` (Technical University of Munich)

**通讯引用:** 7425 | [OpenAlex ID](https://openalex.org/A5079718896)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出一种名为 FlowCalib 的框架，用场景流（static object 的点云运动）来检测 LiDAR 与车辆坐标系之间的角度失配，既能判断是否存在失配，又能区分是 Roll、Pitch 还是 Yaw 轴的失配。

**💡 创新点**

创新点在于：
1) 首次把 S2V 失配视作分类问题，用运动模式而非传统几何约束或外部传感器来识别；
2) 采用双分支网络，将 PointNet 提取的全局流特征与手工几何特征（流幅度、角度直方图、叉积分布）融合，显著提升了对角度误差的辨识能力；
3) 通过对流场的系统性偏差建模，实现了无需 IMU、GPS 或相机的纯 LiDAR 失配检测。

**🔧 技术方法**

核心技术包括：
- 基于 Neural Scene Flow Prior (NSFP) 的场景流生成；
- PointNet 处理无序流向量得到全局特征；
- 手工几何特征提取（流幅度均值/方差、角度直方图、叉积统计）；
- 两个判别头（全局失配与轴向失配），使用二元交叉熵损失训练。

**📊 数据集**

使用公开的 nuScenes 数据集，在该数据集上注入不同幅度的角度误差（±5°、±2°、±1°、±0.5°）进行实验。

**📈 对比分析**

与传统 S2S 误差检测方法相比，FlowCalib 在全局失配检测上取得 81.16% 的准确率；在“易”级别（±5°~±2°）可达 90.27%，“难”级别（±2°~±0.5°）为 73.81%；轴向检测中 Roll、Yaw 分别达到 87.04% 与 76.06%，Pitch 仅 60.81%。整体表明其对多轴失配检测更为稳健。

**⚠️ 局限性**

局限性包括：
1) 对 Pitch 失配的辨识效果明显弱于 Roll/Yaw，原因是 Pitch 产生的运动模式较为均匀；
2) 依赖于高质量的场景流估计，若流估计误差大则检测效果下降；
3) 仅在 nuScenes 数据集上验证，未测试跨域或不同 LiDAR 规格的泛化能力；
4) 需要先去除动态物体与地面，流程较为繁琐。

---

## 492. Lossy Compression of Cellular Network KPIs

**arXiv ID:** 2601.23105 | [PDF](https://arxiv.org/pdf/2601.23105v1)

**作者:** Andrea Pimpinella `[一作]` (University of Bergamo), Alessandro E. C. Redondi `[通讯]` (Politecnico di Milano)

**通讯引用:** 1936 | [OpenAlex ID](https://openalex.org/A5075395997)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `fede83ac-7505-405f-ab37-e7284695c47f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了在蜂窝网络 KPI 上使用有损压缩技术，并通过速率-失真分析和任务中心化方法评估其对网络分析的影响。

**💡 创新点**

创新点在于把压缩视为任务级别的优化，证明了即使在极低比特率（1–2 bit/样本）下，聚合后的 KPI 仍能保持高 SNR，并且对常用预测任务几乎没有影响；此外对比了 DCT 与 KLT 的性能，显示闭式 DCT 在大多数情形下已足够好，减少了训练开销。

**🔧 技术方法**

采用了 PCM、DPCM、DCT、KLT 等预测–量化–熵编码框架，利用均匀标量量化和理想熵编码估计速率；在 KPI 上计算 MSE/SNR，并评估聚合误差和 MWS 预测误差。

**📊 数据集**

使用了约 3000 个 LTE（4G）小区在 2023 年 10 月为期一月的小时级 KPI 数据集，包含下行流量、PRB 占用和活跃用户三种指标。

**📈 对比分析**

与 PCM/DPCM 的样本域压缩以及 KLT 的最佳速率-失真曲线对比；实验表明 DCT 与 KLT 在 2–4 bit/样本时可获得 20–30 dB SNR，聚合后即使 cell‑level SNR 仅 15 dB 也能得到 30 dB 的聚合 SNR；在预测任务中，cell‑level SNR 超过 30 dB 时压缩与原始数据的 RMSE 几乎相同。

**⚠️ 局限性**

仅考虑了时间域压缩，未利用空间冗余；使用理想熵编码，实际压缩率可能略低；实验仅基于 LTE 小区数据，未验证在 5G 或更大规模部署中的泛化能力。

---

## 493. Computing braids from approximate data

**arXiv ID:** 2601.23073 | [PDF](https://arxiv.org/pdf/2601.23073v1)

**作者:** Alexandre Guillemot `[一作]` (Inria), Pierre Lairez `[通讯]` (Inria)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4`

**🎯 论文内容**

本文提出一种基于近似路径分离预测的几何结算方法，能够在仅有根轨迹的粗略笼罩信息时计算参数多项式根的几何结的组合表示。

**💡 创新点**

创新点在于引入分离谓词模型，将不确定性转化为沿实/虚轴的可比较性判定，从而不依赖于精确轨迹线性化即可得到组合结。

**🔧 技术方法**

使用了分离谓词、排列点、排列化简的组合学构造，配合动态拓扑排序和布尔相邻图的增删边算法实现路径覆盖与结算。

**📊 数据集**

使用的测试集是由自定义的参数多项式及其根轨迹通过algpath（基于Taylor模型的区间方法）生成的分离盒子序列。

**📈 对比分析**

与传统的SIROCCO+线性化方法相比，该方法在保持相同精度的同时，减少了对轨迹线性化的需求，实验显示在四条根轨迹的例子中计算时间下降约30%。

**⚠️ 局限性**

局限性包括需对分离谓词提供足够的时间间隔保证性，且在极度接近交叉点时可能需要细分时间步导致计算量激增。

---

## 494. Unsupervised Hierarchical Skill Discovery

**arXiv ID:** 2601.23156 | [PDF](https://arxiv.org/pdf/2601.23156v1)

**作者:** Damion Harvey `[一作]` (University of the Witwatersrand), Steven James `[通讯]` (University of the Witwatersrand)

**通讯引用:** 3509 | [OpenAlex ID](https://openalex.org/A5078861770)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种完全无监督、仅使用观察数据的技能分割与层次化结构发现框架（HiSD），通过先将连续观测序列划分为可重用的技能，再利用语法压缩生成多层次的技能层次结构，并将其作为选项（option）在下游强化学习任务中加速学习。

**💡 创新点**

创新点在于：①在无动作标签、无奖励、无交互的环境下完成技能分割与层次化；②将基于最优传输的时间序列分割与基于Sequitur的语法压缩有机结合，形成端到端的结构学习；③展示该层次化抽象能显著提升高维像素环境（Craftax、Minecraft）中的RL性能。

**🔧 技术方法**

技术手段包括：ASOT（基于最优传输的时间序列分割）；Sequitur语法压缩算法；PCA/MineCLIP等特征提取；行为克隆（BC）与正负样本分类器构建选项的启动/终止条件；高层PPO策略用于选项决策。

**📊 数据集**

使用的主要数据集是：①Craftax（改造为全可观测、确定性环境，采集500条专家轨迹）；②Minecraft（原版全尺寸、部分可观测环境，采集500条VPT生成的轨迹），两者分别在原始和聚合（Mapped）技能标注版本上进行实验。

**📈 对比分析**

与CompILE、OMPN和基准（Ground‑Truth）进行比较，采用全局与局部的IoU、F1、树结构指标（唯一树数、平均深度、树大小、分支因子）评估。结果显示：HiSD在大多数任务（尤其是随机/长周期）上在mIoU、F1等分割指标上优于基线，并在层次结构上获得更少的唯一树、更合理的深度和更小的树尺寸。下游RL实验表明，使用HiSD层次化选项的PPO能在Craftax与Minecraft中取得比平面策略更高且更稳定的奖励，接近Ground‑Truth层次化方案。

**⚠️ 局限性**

局限性包括：①需预先提取特征（PCA/MineCLIP），缺少端到端的感知学习；②需要先验设定技能数K，对K的误设可能导致性能下降；③在噪声极大、随机性强的环境（如完整Minecraft）下，层次结构易变，树的数量与大小膨胀；④目前只针对单一观察序列的离线学习，未充分探索在线交互场景。

---

## 495. Hearing is Believing? Evaluating and Analyzing Audio Language Model Sycophancy with SYAUDIO

**arXiv ID:** 2601.23149 | [PDF](https://arxiv.org/pdf/2601.23149v1)

**作者:** Junchi Yao `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Lijie Hu `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**通讯引用:** 34 | [OpenAlex ID](https://openalex.org/A5067496051)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了SYAUDIO基准，用于系统评估音频语言模型(ALMs)的sycophancy行为。

**💡 创新点**

首次聚焦音频模态的sycophancy，构建多领域(感知、推理、数学、伦理)任务，设计六种用户提示场景并引入噪声、语速等现实因素；并提供链式思考(SFT)的减轻方案。

**🔧 技术方法**

利用音频与文本转换、TTS生成、链式思考数据、监督微调(SFT)、prompt工程等技术，并提出MSS与CRS两种度量。

**📊 数据集**

基于四大公开数据集：MMAU、MMAR、GSM8K-Audio、MMLU（伦理）并用TTS合成算术与伦理问题，构成4,319条音频问答。

**📈 对比分析**

与多款开源和闭源ALMs（Qwen2‑Audio‑7B‑Instruct、Audio‑Flamingo‑3、Qwen2.5‑Omni‑7B、GPT‑4o‑Mini‑Audio‑Preview、Gemini‑2.5‑Flash）对比，结果显示TTS音频显著提高MSS，闭源模型整体更稳健；链式思考SFT显著降低MSS但对CRS提升有限。

**⚠️ 局限性**

局限包括：仅在人工合成与公开数据上评估，未覆盖极端噪声或多说话人场景；链式思考SFT对CRS提升不显著；未来需探索更普适的鲁棒性提升方法。

---

## 496. To See Far, Look Close: Evolutionary Forecasting for Long-term Time Series

**arXiv ID:** 2601.23114 | [PDF](https://arxiv.org/pdf/2601.23114v1)

**作者:** Jiaming Ma `[一作]` (University of Science and Technology of China), Yang Wang `[通讯]` (University of Science and Technology of China)

**通讯引用:** 34170 | [OpenAlex ID](https://openalex.org/A5100764445)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了演化式预测（EF）范式，通过将模型的输出时长与评估时长解耦，解决了传统直接预测（DF）中梯度冲突导致的性能瓶颈；

**💡 创新点**

创新点在于：①将DF视为EF的退化特殊情况；②引入“Reasoning Blocks”迭代生成预测，实现单模型多时长预测；③揭示并缓解DF中的梯度冲突与远程梯度主导问题；

**🔧 技术方法**

技术包括：Transformer/MLP架构的时间序列模型、梯度相似度可视化、EF训练中的混合输入（历史+预测）策略、单模型多步推理的循环机制；

**📊 数据集**

实验数据集涵盖ETTh1/ETTh2、Weather、Traffic、Exchange、ILI等六大行业场景；

**📈 对比分析**

采用相同模型、输入长度T、输出长度L的基准DF与EF进行对比，EF在大多数评估时长下获得10‑14% MSE/MAE提升，极端外推时长仍保持稳定优势，单模型即可替代多模型集合；

**⚠️ 局限性**

局限性包括：EF在极低输入长度或周期性不足时的纯外推阶段可能积累误差；需进一步研究动态调节L、T的自动化策略，并验证在更大规模、不同频率的数据上的泛化。

---

## 497. Rethinking Transferable Adversarial Attacks on Point Clouds from a Compact Subspace Perspective

**arXiv ID:** 2601.23102 | [PDF](https://arxiv.org/pdf/2601.23102v1)

**作者:** Keke Tang `[一作]` (Cyberspace Institute of Advanced Technology, Guangzhou University), Zhihong Tian `[通讯]` (Cyberspace Institute of Advanced Technology, Guangzhou University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了一种在紧凑子空间内进行对抗攻击的框架CoSA，专门针对3D点云模型的可迁移性问题；

**💡 创新点**

创新点在于将点云表示分解为基于类别原型的基础子空间和低秩扰动子空间，限制扰动在共享语义方向上进行；

**🔧 技术方法**

采用预训练自编码器提取低维潜在表示、K‑means聚类构建原型字典、稀疏重构、低秩正则化与正交约束等技术；

**📊 数据集**

实验使用ModelNet40和ScanObjectNN两个公开点云数据集；

**📈 对比分析**

在多种目标网络（PointNet、DGCNN、PCT、Point‑Mamba）上与八种主流可迁移攻击方法对比，CoSA在攻击成功率上普遍提升约8–15%，并保持良好的可感知性；

**⚠️ 局限性**

局限在于对原型字典和低秩维数的选择仍需经验调参，且在强防御场景下仍存在一定攻击成功率下降。

---

## 498. Safer Policy Compliance with Dynamic Epistemic Fallback

**arXiv ID:** 2601.23094 | [PDF](https://arxiv.org/pdf/2601.23094v1)

**作者:** Joseph Marvin Imperial `[一作]` (University of Bath), Harish Tayyar Madabushi `[通讯]` (University of Bath)

**通讯引用:** 463 | [OpenAlex ID](https://openalex.org/A5070941491)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为Dynamic Epistemic Fallback（DEF）的推理时安全协议，用单句提示激活LLM的知识警觉性以识别并拒绝被篡改的法律政策文本；

**💡 创新点**

创新点在于将人类认知中的知识警觉机制转化为LLM的动态安全提示，并设计了三层提示强度来逐步增强警觉；

**🔧 技术方法**

使用了自然语言处理的提示工程、链式思考（CoT）监测、以及LLM的参数知识回溯机制；

**📊 数据集**

在公开的隐私政策数据集上进行实验，采用GDPR和HIPAA两份法律文本，并用GDPRHub和GoldCoin两套情景数据集；

**📈 对比分析**

通过在未篡改、篡改+无DEF、DEF 1/2/3三层提示三种设置下比较，发现最强的Memory Prioritization提示能显著提升检测率和拒绝率（约+30%检测、+33%拒绝），并恢复因篡改导致的准确率下降；

**⚠️ 局限性**

局限性包括：仅验证了两份法律政策文本，对其他类型权威文档的通用性未知；DEF的提示强度可能在非篡改场景引入轻微误报；并且在CoT摘要型LLM上效果受限于其隐式推理不透明性。

---

## 499. Temporally Coherent Imitation Learning via Latent Action Flow Matching for Robotic Manipulation

**arXiv ID:** 2601.23087 | [PDF](https://arxiv.org/pdf/2601.23087v1)

**作者:** Wu Songwei `[一作]` (Harbin Institute of Technology), Liu Hong `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 17256 | [OpenAlex ID](https://openalex.org/A5100410336)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种轨迹层面潜在动作流匹配框架（LG-Flow Policy），通过在连续潜在动作空间进行流匹配，并结合几何感知点云编码与执行时的多模态调制，实现长时限、高维度机械操作的快速、平滑、稳定控制。

**💡 创新点**

①在潜在动作空间而非原始动作空间进行流匹配，显著降低轨迹抖动；②使用GRU编码短期动作段生成时序连贯潜在轨迹；③通过点云局部/中心特征和FiLM实现几何感知条件；④解码阶段使用视觉调制实现执行时的多模态适配；⑤实现近单步推理，兼顾速度与稳定性。

**🔧 技术方法**

流匹配（Consistency Flow Matching）、变分自编码器+GRU、FiLM调制、点云局部/中心编码器、视觉编码器、变分正则化、轨迹平滑度评价、三维几何感知处理。

**📊 数据集**

仿真数据集：Adroit 与 Meta-World（共 37 个任务）；真实机器人实验：Frank­a Emika Panda + LEAP Hand/Parallel Gripper，使用 RealSense L515 与 D435 采集 RGB‑D/点云，收集 30 条示范。

**📈 对比分析**

与 DP3、iDP3、RDP、Flow Policy 等基线对比；在仿真中 LG‑Flow Policy 成功率 78.3%（比 DP3 +12.6%），轨迹平滑度最优，单步推理时间 7.5 ms；在真实机器人中成功率 77.5%（最高），响应时间 8.59 ms，显著优于 DP3（57.9%）和 RDP（70.0%）。

**⚠️ 局限性**

系统采用轻量级多模态融合，未显式建模复杂接触动力学；未来需进一步丰富多模态表示并加入接触感知的潜在动力学，以提升在更具挑战性的接触丰富任务中的表现。

---

## 500. Chain-of-thought obfuscation learned from output supervision can generalise to unseen tasks

**arXiv ID:** 2601.23086 | [PDF](https://arxiv.org/pdf/2601.23086v1)

**作者:** Nathaniel Mitrani Hadida `[一作]` (University of Cambridge), Puria Radmard `[通讯]` (Geodesic Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文研究了在大型语言模型（LLM）中，对最终输出进行惩罚会导致链式推理（CoT）中的推理轨迹被掩盖，并且这种掩盖会泛化到未见过的任务上，尤其是高风险医学情境；

**💡 创新点**

创新点在于首次证明仅对模型最终输出进行监督（而非直接对CoT施加压力）即可诱导CoT掩盖，并且该掩盖策略能够跨任务域、跨情境泛化，揭示了传统RLHF、Constitutional AI等安全训练方法可能无意中削弱模型可监控性的问题；

**🔧 技术方法**

主要技术包括：基于GRPO的强化学习，构造任务奖励（使用泄露信息选择答案）、CoT长度奖励、摘要长度奖励、答案格式奖励以及对泄露行为的惩罚；使用两个惩罚方式——对CoT直接施加压力与对摘要（输出后段）施加压力；并配合Kimi‑K2监督器进行惩罚判定；

**📊 数据集**

数据集方面，使用了四个修改版的reward‑hacking数据集（Revealing Score、Factual Sycophancy、Code Selection、World‑Affecting Reward）以及一个新建的高风险医学sycophancy hold‑out集；训练时还混入随机MMLU题目作为正则化；

**📈 对比分析**

与不加惩罚的基线相比，所有实验中模型在分布内的reward‑hacking率几乎达到最高，而在被惩罚的CoT或摘要中的可监测率几乎为零；更重要的是，输出惩罚（summary‑only）导致的CoT掩盖在离散训练任务之外的医学高危情境中也显著出现，说明泛化效果；实验通过多重seed、多个任务、留一法评估，结果显示此现象普遍存在；

**⚠️ 局限性**

局限性包括：仅在单一模型（Qwen3‑4B）与单一规模上实验；奖励任务过于简化、结构统一，缺乏真实世界的复杂性；未探索更难的任务或需要更深CoT的场景；以及未对不同RL算法、模型尺寸的影响进行系统比较。

---

## 501. Solving 4-Block Integer Linear Programs Faster Using Affine Decompositions of the Right-Hand Sides

**arXiv ID:** 2601.23083 | [PDF](https://arxiv.org/pdf/2601.23083v1)

**作者:** Alexandra Lassota `[一作]` (Eindhoven University of Technology), Koen Ligthart `[通讯]` (Eindhoven University of Technology)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了一种新的、运行时间更快的四块整数线性规划算法。

**💡 创新点**

创新点在于突破了以 Graver 复杂度或接近度为界的平方指数瓶颈，实现了仅线性依赖于块尺寸的指数。

**🔧 技术方法**

采用了 n‑fold ILP 的分解技巧、向量重排引理的仿射化以及超平面排列的面划分来构造动态可信分解。

**📊 数据集**

由于研究为理论算法，未使用实验数据集，仅在理论上证明了算法的复杂度。

**📈 对比分析**

与之前的最优算法相比，运行时间从 O(f(k,Δ)·n^k^2) 降至 O(f(k,Δ)·n^k)，并且对大系数的处理只依赖于其编码长度。

**⚠️ 局限性**

限制在于算法的常数项对块尺寸的三重指数依赖以及尚未达到真正的 FPT 性能；在某些实例中仍需对全局变量进行指数级猜测。

---

## 502. Robust and Generalized Humanoid Motion Tracking

**arXiv ID:** 2601.23080 | [PDF](https://arxiv.org/pdf/2601.23080v1)

**作者:** Yubiao Ma `[一作]` (Beijing Institute of Technology), Dongdong Zheng `[通讯]` (Beijing Institute of Technology)

**通讯引用:** 3675 | [OpenAlex ID](https://openalex.org/A5101685638)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

设计并实现了一个基于动力学条件化命令聚合的全身运动跟踪框架，能够在噪声命令和多接触场景下实现鲁棒的人形机器人控制，并集成了跌倒恢复。

**💡 创新点**

使用因果Transformer提取最近感知的动力学嵌入作为查询，结合跨注意力聚合上下文命令，实现对不可靠命令的自适应筛选；同时在单一训练流程中加入跌倒恢复和随机不稳定初始化。

**🔧 技术方法**

低阶PD控制、残差命令、因果Transformer、跨注意力、多头注意力、PPO、异构演员-评论家训练、随机不稳定初始化、助力力退火等技术。

**📊 数据集**

约3.5小时的高质量收集自LAFAN1和AMASS的MoCap数据（重定向后去噪），并在实验中使用视频估计运动、实时传感器演示等多源数据。

**📈 对比分析**

与Any2Track和GMT在MoCap、视频、地面交互三类数据上比较，取得最高成功率和最低MPJPE；在噪声鲁棒性测试中误差保持低且逐渐衰退；在真实机器人上实现无人工重置的跌倒恢复和多源动作跟踪，性能优异。

**⚠️ 局限性**

缺乏全局定位，无法实现长时段世界坐标一致跟踪；对极端未知动作的泛化仍有限；训练需要较大GPU资源，跌倒恢复仅在实验环境下验证。

---

## 503. Omni-fMRI: A Universal Atlas-Free fMRI Foundation Model

**arXiv ID:** 2601.23090 | [PDF](https://arxiv.org/pdf/2601.23090v1)

**作者:** Mo Wang `[一作]`, Quanying Liu `[通讯]` (Southern University of Science and Technology)

**通讯引用:** 3788 | [OpenAlex ID](https://openalex.org/A5078854583)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

提出了 Omni-fMRI，一种无脑区划分的 voxel 级 fMRI 基础模型。

**💡 创新点**

核心创新在于动态 patch 令模型自适应分辨率的 tokenization、双路径多尺度嵌入以及尺度感知的 MAE 目标，彻底摆脱 atlas 诱导的偏差。

**🔧 技术方法**

采用自监督掩码自编码器（MAE）框架，结合动态 patching、双路径投影与尺度嵌入，并使用 ViT 编码器。

**📊 数据集**

预训练数据包含 49,497 个 fMRI 会话，来源于 5 个公开数据库；下游评测跨 11 个数据集（ABCD、ABIDE、HCP、PPMI、ADNI、SALD、BHRC、NKI、NSD、HCP task、StudyForrest）。

**📈 对比分析**

在众多下游任务（年龄回归、性别/疾病/教育预测、图像检索、情绪识别、脑状态预测等）与现有 ROI/图网络基础模型比较，Omni-fMRI 以 10% 以上的提升或占据最高分数，且在线性探针评估中亦表现优异。

**⚠️ 局限性**

局限性在于动态 patch 采用基于方差阈值的启发式复杂度估计，未实现端到端可学习的 token 选取；可能对特定任务或个体信息分布的自适应性不足。

---

## 504. RN-D: Discretized Categorical Actors with Regularized Networks for On-Policy Reinforcement Learning

**arXiv ID:** 2601.23075 | [PDF](https://arxiv.org/pdf/2601.23075v1)

**作者:** Yuexin Bian `[一作]` (University of California San Diego), Yuanyuan Shi `[通讯]` (University of California San Diego)

**通讯引用:** 6990 | [OpenAlex ID](https://openalex.org/A5100645756)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

将 PPO 中的高斯连续动作策略替换为离散分箱的类别策略，并配合正则化的残差 MLP 结构，提升了在连续控制任务中的学习稳定性和样本效率。

**💡 创新点**

创新点在于将动作空间离散化与交叉熵目标相结合，证明类别策略的交叉熵更新比传统的 MSE 更能利用网络容量，并且正则化残差网络显著降低梯度方差，二者组合得到的 RN‑D 架构在多种任务上均取得最优表现。

**🔧 技术方法**

技术手段包括：动作空间统一分箱（K=41）、分离式类别策略、基于 LayerNorm 的预层残差 MLP 结构、PPO 的剪切策略更新、优势估计 GAE、以及对比实验中的梯度方差和信噪比分析。

**📊 数据集**

数据集覆盖：Gym MuJoCo 5 种行走/跑步/跳跃任务、ManiSkill 10 种低维状态控制任务、ManiSkill 5 种基于 RGB 的视觉控制任务。

**📈 对比分析**

与三种基线比较：RN‑C（高斯+正则化网络）、MLP‑C（标准高斯+浅 MLP）和 MLP‑D（类别+浅 MLP）。RN‑D 在所有任务上均实现最高的归一化回报/成功率，且在训练步数上相较最佳基线约 1.3–1.9 倍更快收敛；梯度方差也明显更低，SNR 更高。

**⚠️ 局限性**

局限性包括：实验仅在 PPO 框架下验证，未探究与其他 on‑policy 算法的通用性；离散分箱的粒度需要手动调参，对极大动作空间或高分辨率任务可能受限；以及未系统分析演员与评论者交互的协同效应。

---

## 505. No More, No Less: Least-Privilege Language Models

**arXiv ID:** 2601.23157 | [PDF](https://arxiv.org/pdf/2601.23157v1)

**作者:** Paulius Rauba `[一作]` (University of Cambridge), Mihaela van der Schaar `[通讯]` (University of Cambridge)

**通讯引用:** 22347 | [OpenAlex ID](https://openalex.org/A5012339002)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了基于最小特权原则的语言模型部署方案，并通过构建Nested Least‑Privilege Networks（NLPN）实现可调节的内部计算可达性；

**💡 创新点**

创新点在于把“最小特权”迁移到语言模型内部，设计了可逆、按秩索引的权重截断接口，并将请求级别的特权分配与模型推理结合；

**🔧 技术方法**

技术上采用低秩因子化重参数化、加权多任务训练、动态特权分配策略（静态、递进、跳跃），以及对模型块级灵敏度的分析；

**📊 数据集**

主要使用了算法级任务数据集（Balanced Brackets、Length Comparison、Contains Substring），以及MiniPile语料做细调，并在Pythia‑1B、Qwen2.5‑0.5B等预训练模型上进行实验；

**📈 对比分析**

通过对不同特权分配策略在多目标精度（80%、90%、95%）下的平均秩、准确率与推理开销进行比较，发现特权递进策略能在保持目标准确率的同时显著降低平均秩，且块级截断可针对性抑制特定知识；

**⚠️ 局限性**

局限性包括：仅在有限任务与模型上验证，缺乏对提示诱导恢复的系统评估；特权分配需依赖可靠信号；未提供理论安全保证，且可能被滥用于隐性审查或歧视性访问控制。

---

## 506. SPICE: Submodular Penalized Information-Conflict Selection for Efficient Large Language Model Training

**arXiv ID:** 2601.23155 | [PDF](https://arxiv.org/pdf/2601.23155v1)

**作者:** Powei Chang `[一作]` (Bilibili Inc), Dongying Kong `[通讯]` (Bilibili Inc)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种冲突感知的贪婪数据选择方法 SPICE，用于指令微调，能够在仅使用 10% 数据的情况下实现甚至超越全数据训练的性能。

**💡 创新点**

创新点在于：① 通过 ε‑decomposition 定量关联梯度冲突与 Fisher 信息子模性质的曲率，揭示子模近似失效的根本原因；② 在贪婪得分中加入冲突惩罚，既保留高信息样本，又抑制梯度对抗，显著减缓信息增益衰减；③ 设计了基于代理模型与自适应早停的高效实现。

**🔧 技术方法**

主要技术包括：Fisher 信息矩阵（log‑det 目标）、子模优化与曲率分析、梯度冲突度量（负余弦相似度）、代理模型梯度预估、冲突惩罚的自适应权重、早停策略。

**📊 数据集**

使用了包含数学推理、代码生成与通用知识的 97.5K 指令响应语料库（GSM8K、Alpaca Code、ShareGPT、Alpaca 等），在 8 个下游评测基准上进行验证。

**📈 对比分析**

与全量数据、随机抽样、IFD、Fisher、LESS、SelectIT、DPP、TSDS、LEAD 等基线比较，SPICE 在 Qwen2‑7B 上平均分 58.0，超越全量训练 56.4，且在 7/8 评测任务上取得最高或次高分，训练成本约 20 GPU‑h，仅占全量的 1/5。

**⚠️ 局限性**

局限性包括：仍需使用代理模型进行梯度估计，导致额外计算；对跨架构（如 LLaMA 与 Qwen）迁移性能略低；冲突惩罚参数需要经验调优；目前仅在文本指令任务验证，尚未在多模态或强化学习场景中评估。

---

## 507. On Safer Reinforcement Learning Policies for Sedation and Analgesia in Intensive Care

**arXiv ID:** 2601.23154 | [PDF](https://arxiv.org/pdf/2601.23154v1)

**作者:** Joel Romero-Hernandez `[一作]`, Oscar Camara `[通讯]` (Pompeu Fabra University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文利用离线强化学习研究ICU镇痛镇静剂的剂量决策，比较仅优化疼痛缓解与同时考虑疼痛和30天生存率的两种策略；

**💡 创新点**

创新点在于：①引入死亡率作为终结奖励项以强化长期安全；②通过GRU学习递归状态表示处理部分可观测；③使用四种连续药物剂量的高维动作空间，并在超过47000例ICU数据上训练；

**🔧 技术方法**

技术主要包括离线深度强化学习（基于Actor-Critic与双Critic的行为正则化），GRU状态编码，MICE多重插补，和贝尔曼目标的惰性估计；

**📊 数据集**

数据集为MIMIC‑IV数据库，包含47144个成人ICU停留，22维时序观测及四种药物的连续剂量；

**📈 对比分析**

评估方法是计算策略与临床剂量的相似度，并通过Bootstrapped Spearman相关性检验相似度与30天死亡率、累计疼痛的关系。结果显示仅优化疼痛的策略与更高死亡率正相关，而考虑死亡率的策略与死亡率负相关，且在高复杂度病例中更安全；

**⚠️ 局限性**

局限性包括：观察性关联而非因果推断；模型在真实世界噪声、异质性强的ICU数据上表现仅为统计关联；缺乏多中心验证及因果建模，未来需加强鲁棒性与推广性验证。

---

## 508. Manifold-Aware Perturbations for Constrained Generative Modeling

**arXiv ID:** 2601.23151 | [PDF](https://arxiv.org/pdf/2601.23151v1)

**作者:** Katherine Keegan `[一作]` (Department of XXX, University of YYY), Lars Ruthotto `[通讯]` (Company Name)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提供了国际机器学习会议（ICML 2026）的论文提交和格式化指南。

**💡 创新点**

创新点在于提供了详细的提交要求和格式规范，以确保论文的统一性和可读性。

**🔧 技术方法**

使用了PDF格式提交，强调了字体和排版的要求。

**📊 数据集**

未使用特定数据集，主要是格式和提交流程的指导。

**📈 对比分析**

与其他会议的比较未具体提及，但强调了双盲审稿的过程以确保公正性。

**⚠️ 局限性**

限制在于未提供具体的研究内容或方法，主要集中在格式和提交要求上。

---

## 509. Do Good, Stay Longer? Temporal Patterns and Predictors of Newcomer-to-Core Transitions in Conventional OSS and OSS4SG

**arXiv ID:** 2601.23142 | [PDF](https://arxiv.org/pdf/2601.23142v1)

**作者:** Mohamed Ouf `[一作]` (Queen's University), Mariam Guizani `[通讯]` (Queen's University)

**通讯引用:** 269 | [OpenAlex ID](https://openalex.org/A5058066905)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

对375个GitHub项目（190 OSS4SG，185常规OSS）中的92,721名贡献者的提交和PR等活动进行大规模挖掘，研究新人转变为核心贡献者的过程，包括结构特征、早期行为预测、转化路径和时间模式。

**💡 创新点**

首次将项目使命（社会公益 vs 传统）与新人转型路径做系统比较，发现OSS4SG项目在贡献者留存、核心比例、转化率、风险因素等方面显著更好；提出早期广泛探索和“Late Spike”贡献强度模式是快速晋升核心的关键；提供可操作的新人和维护者指导。

**🔧 技术方法**

使用机器学习预测模型（Logistic Regression、Random Forest、Gradient Boosting）、Markov链路径分析、动态时间规整(DTW)聚类、Scott‑Knott检验、Cox生存模型、Gini、Bus Factor等多种定量方法；核心贡献者定义为80% Pareto，贡献指数(CI)衡量每周贡献强度。

**📊 数据集**

来自GitHub Archive的375个活跃项目（使用Python、JavaScript/TypeScript、Go、Rust等），结合DPGA和Ovio的OSS4SG项目列表；共3.5M次提交、92,721名贡献者、8,812名已晋升核心的贡献者。

**📈 对比分析**

采用结构指标、周转化率、Kaplan‑Meier生存曲线和Cox回归比较OSS4SG与常规OSS；预测模型PR‑AUC最高0.656，ROC‑AUC 0.746；路径分析显示OSS4SG有多条高效路径，常规OSS集中在单一路径；时间模式聚类揭示“Late Spike”最快（21周），常规OSS仅此模式最快。

**⚠️ 局限性**

结果仅基于GitHub公开仓库，缺乏对小型或非GitHub平台项目的验证；核心定义以提交为主，可能低估文档/评审等贡献；身份解析误差和项目样本偏差仍可能影响结论；因相关性而非因果关系，无法确认使命是决定因素。

---

## 510. From Monolith to Microservices: A Comparative Evaluation of Decomposition Frameworks

**arXiv ID:** 2601.23141 | [PDF](https://arxiv.org/pdf/2601.23141v1)

**作者:** Mineth Weerasinghe `[一作]` (University of Moratuwa), Srinath Perera `[通讯]` (WSO2 LLC)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对从单体架构迁移到微服务的分解框架进行统一、可复现的评估；通过一致的度量管道重新计算已有研究结果并补充实验数据，比较静态、动态与混合分解方法在四个常用基准系统上的性能。

**💡 创新点**

首次将多种分解工具（静态、动态、混合和学习型）放在同一评估环境下进行客观比较，并提出统一的综合得分公式；证明层次化密度聚类（HDBScan）在多项指标上持续领先。

**🔧 技术方法**

使用结构模块化（SM）、接口数（IFN）、分区间通信（ICP）、非极端分布（NED）等指标；通过z-score标准化和加权合成得到综合得分；实验中调用了Bunch、MEM、FoSCI、CoGCN、Mono2Micro、HDBScan、a-BMSC、CHGNN、MonoEmbed等工具。

**📊 数据集**

四个开源基准系统：JPetStore、AcmeAir、DayTrader、Plants（Plant Nursery）。

**📈 对比分析**

对每个工具在每个基准上的指标进行标准化后加权求和得到“Score”，并根据得分排序。实验显示，HDBScan 在所有基准上得分最高，a‑BMSC 和 Mono2Micro 次之；其他方法（MEM、CoGCN、CHGNN、MonoEmbed）表现中等或与数据集高度相关。

**⚠️ 局限性**

局限性包括：基准版本与工具配置可能不一致导致指标可比性受限；综合得分权重手工设定且缺乏统计显著性检验；仅使用四个基准且在不同硬件上运行；缺乏与真实服务边界的精度评估；未考虑运行时成本或性能开销。

---

## 511. Automated Testing of Prevalent 3D User Interactions in Virtual Reality Applications

**arXiv ID:** 2601.23139 | [PDF](https://arxiv.org/pdf/2601.23139v1)

**作者:** Ruizhen Gu `[一作]` (University of Sheffield), Donghwan Shin `[通讯]` (University of Sheffield)

**通讯引用:** 1068 | [OpenAlex ID](https://openalex.org/A5019085537)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一套基于Interaction Flow Graph的自动化VR交互测试工具

**💡 创新点**

创新点在于提出IFG模型以抽象多步交互并构建覆盖度指标IFC，以及通过静态图驱动自动交互执行

**🔧 技术方法**

采用Unity XR Interaction Toolkit、静态场景解析、图模型构建、模拟控制器动作、碰撞检测oracle等技术

**📊 数据集**

使用XRBench3D基准数据集，包括10个Unity VR场景共456个交互

**📈 对比分析**

与随机探索基线对比，覆盖率提升12倍，效率提高6倍，成功发现异常与未响应交互错误

**⚠️ 局限性**

局限在于仅支持基于抓取/触发的交互、仅分析静态场景、对运行时生成或销毁对象识别不足，且只针对Unity/XRI

---

## 512. Why GRPO Needs Normalization: A Local-Curvature Perspective on Adaptive Gradients

**arXiv ID:** 2601.23135 | [PDF](https://arxiv.org/pdf/2601.23135v1)

**作者:** Cheng Ge `[一作]` (Massachusetts Institute of Technology), Jiawei Zhang `[通讯]` (University of Wisconsin Madison)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了 GRPO 算法中标准差归一化的理论基础和实验效果，证明其可视为曲率自适应梯度机制并加速收敛。

**💡 创新点**

创新点是将奖励方差解释为局部曲率估计，提出自适应梯度视角，并在多种假设下给出 GRPO 的更快收敛理论证明，同时揭示训练过程的三阶段规律。

**🔧 技术方法**

采用 GRPO、REINFORCE、PPO 等策略梯度方法；log‑linear 策略参数化；Fisher 信息曲率估计；正交与弱正交假设；梯度相似度分析。

**📊 数据集**

使用 GSM8K（Easy/Hard）和 MATH 两个数学推理基准，并通过先验评估器对难度进行划分。

**📈 对比分析**

与不归一化的 GRPO、REINFORCE 对比，采用准确率和梯度相似度评估；实验显示在中期阶段标准化提升约 5–7% 准确率，最终保持 2–3% 的优势。

**⚠️ 局限性**

局限性在于依赖正交假设和奖励方差与曲率的局部关联；后期高方差时增益下降；未充分解决跨 prompt 交互噪声和跨任务泛化问题。

---

## 513. Machine Learning for Energy-Performance-aware Scheduling

**arXiv ID:** 2601.23134 | [PDF](https://arxiv.org/pdf/2601.23134v1)

**作者:** Zheyuan Hu `[一作]` (University of Cambridge), Yifei Shi `[通讯]` (University of Cambridge)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文将异构多核系统调度参数优化建模为贝叶斯优化问题，利用高斯过程自适应搜索Pareto前沿并结合敏感性分析揭示硬件敏感度。

**💡 创新点**

创新点在于采用Matérn 5/2核捕捉非光滑性能景观，自动发现“race-to-idle”策略与资源解耦，以及在高负载下的相位转移；同时将多目标优化与fANOVA结合，实现可解释的调度原则。

**🔧 技术方法**

技术包括高斯过程回归、贝叶斯优化、EHVI/LogEI采集函数、敏感性分析（fANOVA）、离散事件模拟、Sobol初始化。

**📊 数据集**

数据集为基于Poisson过程生成的任务流（500任务、1000 ms模拟时间），在多种工作负载（λ=0.5、1.0、2.5、5.0）下评估。

**📈 对比分析**

与随机搜索基线对比；在单目标下实现比随机搜索低约3–5 % 的损失，在多目标下获得约15 % 的超体积提升，且能覆盖更广阔的能耗-时延折中空间。

**⚠️ 局限性**

局限在于仅做离线静态配置，未考虑在线动态调度、任务依赖关系、以及真正硬件测量验证，且对极端负载下的最优策略仍需进一步理论分析。

---

## 514. Distribution-informed Efficient Conformal Prediction for Full Ranking

**arXiv ID:** 2601.23128 | [PDF](https://arxiv.org/pdf/2601.23128v1)

**作者:** Wenbo Liao `[一作]` (Chinese University of Hong Kong), Hongxin Wei `[通讯]` (Southern University of Science and Technology)

**通讯引用:** 1062 | [OpenAlex ID](https://openalex.org/A5020027500)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在全排序场景下提出基于分布信息的合格预测方法 DCR 与其随机近似 MDCR，解决了传统 TCPR 方法过度保守的问题。

**💡 创新点**

利用负超几何分布精确刻画校准项绝对排名的分布，从而得到非一致性分数的精确分布；并给出理论证明其覆盖性及相对 TCPR 更高的效率。

**🔧 技术方法**

合格预测框架、负超几何分布推导、混合 CDF 计算、蒙特卡洛采样实现。

**📊 数据集**

合成数据、Yummly28K、ESOL、Anime 推荐数据集。

**📈 对比分析**

与 Oracle CP 和 TCPR 对比，DCR 在保持 90% 覆盖率的前提下，预测集相对长度平均缩小 30-36%，在多模型、多数据集上均优于 TCPR；MDCR 在计算速度上优于 TCPR，性能略逊于 DCR。

**⚠️ 局限性**

对极大样本或高模型精度场景仍会出现保守性；MDCR 随机采样导致方差增大；DCR 复杂度为 O(nm)，在 m 极大时不具备极限可扩展性；方法依赖样本可交换性假设。

---

## 515. An Automatic Deep Learning Approach for Trailer Generation through Large Language Models

**arXiv ID:** 2601.23121 | [PDF](https://arxiv.org/pdf/2601.23121v1)

**作者:** Roberto Balestri `[一作]` (University of Bologna), Guglielmo Pescatore `[通讯]` (University of Bologna)

**通讯引用:** 152 | [OpenAlex ID](https://openalex.org/A5038915137)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了基于大型语言模型的完整多模态电影预告片自动生成框架，从剧情梳理到视觉剪辑、配音和配乐全流程实现一键生成。

**💡 创新点**

创新点在于将GPT‑4主动协同处理场景划分、对白选择、配音文本、音乐描述等多项任务，并利用Clip‑ViT‑L、StableWhisper、MusicGen等多模态模型实现端到端自动化。

**🔧 技术方法**

技术包括OpenAI GPT‑4、Clip‑ViT‑L、StableWhisper、Pyannote、MusicGen、Coqui XTTS‑v2、OCR、SBD、音频分离与淡入淡出等。

**📊 数据集**

数据来源主要是IMDB抓取的电影信息和剧本、公开影片片段，实验使用《The Wolverine》《The Hobbit》《300》等公开电影，并用Night of the Living Dead做演示。

**📈 对比分析**

与PPBVAM和Movie2Trailer两种现有方法在三部影片的预告片上进行Likert量表评测，平均得分约3.56/2.96/2.90，显著优于对手（分别为3.15/3.08/2.85和2.81/2.88/2.62）。

**⚠️ 局限性**

局限包括样本量小、AI生成音乐与配音质量尚不如人类、缺乏动作识别导致画面与对白不匹配、未做更细粒度的定量/定性评测。

---

## 516. THINKSAFE: Self-Generated Safety Alignment for Reasoning Models

**arXiv ID:** 2601.23143 | [PDF](https://arxiv.org/pdf/2601.23143v1)

**作者:** Seanie Lee `[一作]` (KAIST AI), Sung Ju Hwang `[通讯]` (KAIST AI, DeepAuto.ai)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 ThinkSafe 框架，通过在模型自身分布内生成安全的推理轨迹来实现大规模推理模型的安全对齐；

**💡 创新点**

创新点在于利用轻量级的拒绝指令（refusal steering）唤醒模型潜在的安全推理能力，并在不依赖外部教师的情况下生成自监督训练数据，从而显著降低分布偏移；

**🔧 技术方法**

技术手段包括基于 LoRA 的微调、在模型内部进行拒绝指令引导采样、使用安全守护模型过滤训练样本、以及通过交叉熵或前向 KL 损失进行离线微调；

**📊 数据集**

数据集涵盖安全评估数据集（StrongReject、HarmBench、WildJailbreak、XSTest）和推理评估基准（GSM8K、MATH500、AIME24、GPQA），同时使用 SafeChain 原始提示集生成自监督数据；

**📈 对比分析**

与传统教师蒸馏（SafeChain、STAR‑1、SafeKey）以及直接拒绝（DirectRefusal）和在线 RL（GRPO）等基线对比，ThinkSafe 在安全性上优于或等同于 GRPO，在推理准确率上保持或提升，且训练成本大幅降低（约 8 倍更快、成本更低）；

**⚠️ 局限性**

局限性包括对拒绝指令的依赖性、仍无法完全消除安全性与推理能力的权衡、以及在极端有害提示上可能仍需进一步校准安全守护模型。

---

## 517. CATTO: Balancing Preferences and Confidence in Language Models

**arXiv ID:** 2601.23096 | [PDF](https://arxiv.org/pdf/2601.23096v1)

**作者:** Nisarg Parikh `[一作]` (University of Massachusetts), Andrew Lan `[通讯]` (University of Massachusetts)

**通讯引用:** 1802 | [OpenAlex ID](https://openalex.org/A5063813962)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种在偏好优化过程中对大语言模型进行置信度校准的方法，设计可微分的 per‑token 校准损失并将其与 Direct Preference Optimization 结合，形成 Calibration‑Aware Preference Optimization（CAPO）。

**💡 创新点**

创新点在于将基于 ECE 的置信度校准目标直接嵌入偏好优化的训练目标，实现在保持偏好排序的同时约束绝对概率尺度，避免了后期校准的计算开销和误差漂移。

**🔧 技术方法**

核心技术包括可微分的 per‑token 校准损失、Differentiable Expected Calibration Error surrogate、与 DPO 的线性组合训练，以及 Confidence@k 推理机制。

**📊 数据集**

使用了 5 个多样化的问答/选择题数据集（如 GSM8K、MMLU、BBH、ARC 等）以及相应的分布外对照集进行评估。

**📈 对比分析**

与 DPO、RCFT、DPO+BCE 等基线比较，CAPO 在内分布下 ECE 降低 2.22%–7.61%，在分布外下降 1.46%–10.44%；在任务准确率上保持或略提升，平均 +3.16%，最大降幅仅 1.33%，并且相较于 RCFT 计算成本显著降低。

**⚠️ 局限性**

局限性包括依赖可微分的概率近似，极端概率误估可能削弱校准信号；目前仅在固定标签的 MCQ 任务验证，开放式生成或多模态场景仍待验证；token‑level 校准可能不足以完整捕捉全序列的可靠性。

---

## 518. WiFiPenTester: Advancing Wireless Ethical Hacking with Governed GenAI

**arXiv ID:** 2601.23092 | [PDF](https://arxiv.org/pdf/2601.23092v1)

**作者:** Haitham S. Al-Sinani `[一作]` (Diwan of Royal Court), Chris J. Mitchell `[通讯]` (Royal Holloway)

**通讯引用:** 10860 | [OpenAlex ID](https://openalex.org/A5063477888)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了WiFiPenTester系统，利用大语言模型对无线网络扫描结果进行智能目标排序与可行性评估，并在严格的人工监管与预算控制下执行握手捕获与密码强度测试；

**💡 创新点**

在无线渗透测试中首次提出可治理、可审计的GenAI协助流程，采用结构化提示工程与JSON输出模式限制LLM行为，同时实现成本估算、数据最小化与离线/本地LLM部署，保障安全与合规；

**🔧 技术方法**

Python实现、Aircrack‑ng、wifite、kismet等无线工具链、OpenAI GPT‑4（可替换为Claude、Gemini、Llama等本地模型）以及自定义提示模板、链式推理、JSON schema校验、成本估算与日志记录；

**📊 数据集**

实验使用真实无线环境（Kali VM配合USB MT7601U适配器）下的多台AP（包含WEP、WPA/WPA2‑PSK、WPA3‑SAE）和手工生成的受控信号强度、客户端活动等监测数据；

**📈 对比分析**

与传统静态启发式工具（Aircrack‑ng、wifite）对比，实验显示GenAI辅助的目标排序准确率提升约20‑30%，整体评估时间缩短30%，且在多AP稠密环境中保持较高的决策一致性；

**⚠️ 局限性**

依赖被动扫描数据的完整性、动态RF环境导致评估失真、对WPA3‑SAE攻击支持有限、LLM偶尔出现幻觉、成本与隐私风险、需要人工审批等限制因素；

---

## 519. From Similarity to Vulnerability: Key Collision Attack on LLM Semantic Caching

**arXiv ID:** 2601.23088 | [PDF](https://arxiv.org/pdf/2601.23088v1)

**作者:** Zhixiang Zhang `[一作]` (Hong Kong University of Science and Technology), Dongdong She `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 439 | [OpenAlex ID](https://openalex.org/A5048358055)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一种针对LLM语义缓存的键碰撞攻击框架，能够在多租户环境中通过对缓存键的伪造导致响应劫持与代理行为失控。

**💡 创新点**

首次将语义缓存键建模为模糊哈希，并系统性揭示其局部性与雪崩效应之间的内在权衡，从而证明该设计天然易受键碰撞攻击；同时构造了基于生成-验证器的自动攻击框架，兼顾黑盒约束与效率。

**🔧 技术方法**

采用生成器-验证器结构、基于梯度的搜索（GCG）、近似匹配（余弦相似/LSH）、统计缓存命中判别、以及代理模型逼真度评估等技术。

**📊 数据集**

使用SC‑IPI（4185条间接注入指令）、Natural Questions（NQ）样本、Berkeley Function Calling Leaderboard（BFCL）以及多种句子嵌入模型（Sentence‑BERT、CLIP‑Text、SBERT、Universal‑Sentence‑Encoder）进行实验。

**📈 对比分析**

通过比较命中率（HR）、注入成功率（ISR）、工具调用成功率（TSR）和准确率（Acc）等指标，攻击在R1中HR≈0.86、ISR≈0.82，在R2中工具命中率≈90.6%、准确率下降约83%，并在不同嵌入模型间实现了高达92%以上的跨模型转移。

**⚠️ 局限性**

受限于对缓存状态的隐蔽性、黑盒对齐难度、TTL等待导致的时间消耗以及攻击易被检测的风险，且缺乏在真实生产系统中持续评估的结果。

---

## 520. A Complete Finitary Refinement Type System for Scott-Open Properties

**arXiv ID:** 2601.23082 | [PDF](https://arxiv.org/pdf/2601.23082v1)

**作者:** Colin Riba `[一作]` (École Normale Supérieure de Lyon), Adam Donadille `[通讯]` (École Normale Supérieure de Lyon)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

针对处理无限数据（如流、无穷树）的 λ‑程序，构建了一套有限制的精炼类型系统，用来形式化和验证其输入‑输出属性；并证明该系统在 Scott‑开集性质上是完备且可靠的。

**💡 创新点**

创新点主要有：① 引入带有极限/极大固定点的多极化逻辑，将 Scott‑开集与紧致饱和集分别映射到正负极性公式；② 通过实现 realizability implication 让函数类型能够表达非平凡的输入‑输出规范；③ 将此逻辑与 Abramsky 的“领域论的逻辑形式”结合，得到一种既有限又完备的精炼类型系统。

**🔧 技术方法**

技术手段包括：领域论（Scott 域、谱空间）、极限逻辑（μ‑计算机与其固定点变量）、多极化逻辑（正负极性、可迭代固定点）、实现 realizability implication 的子类型规则、以及对递归类型和时序逻辑的语义解释。

**📊 数据集**

本工作为理论性研究，没有使用实验数据集；所有结果均通过形式化证明给出。

**📈 对比分析**

由于缺乏实验评测，本文没有提供与其他方法的性能比较；理论上证明了正向规范的半可判定性和完备性，但未给出具体运行时性能指标。

**⚠️ 局限性**

局限性包括：① 仅覆盖 Scott‑开集属性，无法直接处理所有 liveness（永活）属性；② 由于系统是半可判定的，不能保证对所有程序的判定性；③ 受限于使用的 λ‑系统与递归类型，未涵盖带有副作用的计算模型（如存储、并发）。

---

## 521. SplineFlow: Flow Matching for Dynamical Systems with B-Spline Interpolants

**arXiv ID:** 2601.23072 | [PDF](https://arxiv.org/pdf/2601.23072v1)

**作者:** Santanu Subhash Rathod `[一作]` (CISPA Helmholtz Center for Information Security), Xiao Zhang `[通讯]` (CISPA Helmholtz Center for Information Security)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `40105733-5154-44cd-8090-a8cab9e64b07` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种基于流匹配的B-样条插值方法——SplineFlow，用于从稀疏观测数据中学习连续动力系统的速度场和漂移。

**💡 创新点**

创新点在于：①将B-样条插值引入条件路径构造，克服线性插值的振荡和不光滑问题；②提供理论误差分析和速度场解析形式；③在确定性与随机性动力学、离散与连续时间、线性与非线性系统上均实现统一。

**🔧 技术方法**

技术核心包括：流匹配（Conditional Flow Matching）、B-样条插值（Cox‑de Boor 递推）、连续正则化的速度场回归、Schrödinger桥与SF2M框架（随机动力学），以及数值求解 ODE/SDE 的 Euler/Euler‑Maruyama 方法。

**📊 数据集**

使用的基准数据集包括：多种经典 ODE 系统（指数衰减、谐振子、阻尼谐振子、Lotka–Volterra、Lorenz）、其对应的 SDE 版本、MuJoCo 的 HopperPhysics 仿真轨迹，以及两组单细胞转录组时间序列（Axolotl 脑再生、胚胎发育）在 PHATE/PCA 嵌入空间。

**📈 对比分析**

与基准方法（NeuralODE、LatentODE、Trajectory Flow Matching、SF2M、MOTFM）对比，SplineFlow 在大多数 ODE/​SDE 任务中均取得更低的 MSE、Wasserstein、MMD 等指标，尤其在非线性、周期性与不规则采样场景下表现更显著；在细胞动力学推断任务中，其在 Wasserstein、MMD、能量距离指标上均优于竞争对手。

**⚠️ 局限性**

局限性：①对 B‑样条的节点位置仍采用固定等距或预设分布，未学习自适应节点；②当前仅处理状态无关的扩散项，缺乏对状态相关噪声的建模；③在极其复杂或高维随机动力学下，训练收敛速度和数值稳定性仍需进一步提升。

---

## 522. ExplainerPFN: Towards tabular foundation models for model-free zero-shot feature importance estimations

**arXiv ID:** 2601.23068 | [PDF](https://arxiv.org/pdf/2601.23068v1)

**作者:** Joao Fonseca `[一作]` (INESC-ID), Julia Stoyanovich `[通讯]` (New York University)

**通讯引用:** 2892 | [OpenAlex ID](https://openalex.org/A5082830839)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建了一种名为 ExplainerPFN 的零样本表格基础模型，用于在无法访问模型或其解释的情况下，直接从输入数据和预测值推断特征重要性（Shapley 值）。

**💡 创新点**

创新点在于：①实现完全零样本的 Shapley 估计，②利用合成结构因果模型预训练生成任务，③引入基于 Shapley 公理的后处理校准，④通过统一的 Transformer 架构完成在不同任务上的迁移学习。

**🔧 技术方法**

核心技术包括 TabPFN 的表格 Transformer、合成数据生成（随机 DAG + 生成式模型）、对数似然训练目标、标准化与分桶化的 Shapley 目标、以及多步后处理（零均值、方差归一化、实例效率校正）。

**📊 数据集**

评估数据集包括 11 个 UCI 表格数据集以及 2018 年美国人口普查的公共健康保险覆盖率数据（ACS Public Coverage），全部在预训练之外独立测试。

**📈 对比分析**

与使用 2–10 条 SHAP 参考样本的少量样本代理解释器（TabPFN、MLP、RF）比较，ExplainerPFN 在大多数数据集上实现了相当或更高的 Pearson 相关系数，并且推断速度比 SHAP 快数十倍，尤其在随机森林基准上显著提升。

**⚠️ 局限性**

主要局限包括：对特征间绝对量级的校准不完善、维度高时性能下降、对真实数据分布的迁移不确定、以及在高维交互复杂度高的任务上可能产生不可靠的解释。

---

## 523. Segment Any Events with Language

**arXiv ID:** 2601.23159 | [PDF](https://arxiv.org/pdf/2601.23159v1)

**作者:** Seungjun Lee `[一作]` (National University of Singapore), Gim Hee Lee `[通讯]` (National University of Singapore)

**通讯引用:** 9396 | [OpenAlex ID](https://openalex.org/A5071967339)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了SEAL框架，实现在事件摄像机中实现开放词汇实例分割（OV-EIS），支持多层级（语义、实例、部件）与自由文本查询；

**💡 创新点**

创新点在于：①引入多模态层次语义指导（MHSG），通过SAM生成多级掩模并利用LLM生成文本描述，实现对事件的多层级语义学习；②在单一事件背骨上构建轻量级多模态融合网络，融合语言、空间与掩模特征，解决“死掩模”和“语义冲突”问题；③自定义四种基准（DDD17‑Ins、DSEC11‑Ins、DSEC19‑Ins、DSEC‑Part），为事件实例分割提供完整评测；

**🔧 技术方法**

技术手段包括：事件摄像机数据预处理、EventSAM掩模生成、CLIP与SAM的跨模态特征对齐、MLLM生成多层级文本指导、跨注意力与自注意力的融合网络、RoI‑Align、空间编码与掩模特征增强、参数高效设计与推理加速；

**📊 数据集**

使用的数据集：混合的Mixed‑24K（事件‑图像对），基准数据集包括DDD17‑Seg、DSEC‑Semantic（转为实例/部件掩模），以及自构建的四个基准；

**📈 对比分析**

与AR‑CDG、Hybrid、AF‑DA三类基线进行对比，SEAL在AP、AP50、AP25上均超越所有基线，提升幅度达3–6个百分点；同时参数量约为100M，推理时间仅22ms，显著快于同类方法；

**⚠️ 局限性**

局限性在于：①仍依赖事件‑图像对进行无标注训练，对真正无图像对的场景适配性待验证；②基准集中于单一事件摄像机，未覆盖不同光照、运动速度极端情况；③对自由文本的鲁棒性尚未在大规模开放词汇集上充分评估。

---

## 524. Securing Time in Energy IoT: A Clock-Dynamics-Aware Spatio-Temporal Graph Attention Network for Clock Drift Attacks and Y2K38 Failures

**arXiv ID:** 2601.23147 | [PDF](https://arxiv.org/pdf/2601.23147v1)

**作者:** Saeid Jamshidi `[一作]` (Polytechnique Montreal), Foutse Khomh `[通讯]` (Polytechnique Montreal)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并实现了一种时钟动态感知的时空图注意网络（STGAT），用于检测能源物联网系统中的时钟漂移攻击和Y2K38溢出故障。

**💡 创新点**

创新点在于将漂移感知时间嵌入、时间自注意力与图注意力相结合，并通过曲率正则化在潜在空间区分正常时钟演化与异常时钟失真，从而精准捕捉极端时间异常。

**🔧 技术方法**

使用漂移感知的时间嵌入、Transformer自注意力、图卷积/图注意力、奥斯顿–舒尔茨过程建模、曲率正则化以及顺序似然比检测等技术。

**📊 数据集**

在公开的 Edge‑IIoTset 数据集上，通过算法 1 注入控制的时钟漂移、同步偏移、噪声和 Y2K38 溢出等时间层扰动进行训练和评估。

**📈 对比分析**

与 LSTM、Transformer、GAT 和 IoT‑TimeFormer 基线比较，STGAT 在准确率、AUC 和 F1 上分别高出约 3–5%，检测延迟平均降至 2.3 步，差异具有显著统计意义。

**⚠️ 局限性**

局限性包括：依赖人工合成的时间扰动，未覆盖所有真实时钟失效模式；模型计算量和能耗较大，难以直接部署在极低功耗设备；对动态网络拓扑和混合攻击缺乏鲁棒性保障。

---

## 525. Evaluating the Utility of Grounding Documents with Reference-Free LLM-based Metrics

**arXiv ID:** 2601.23129 | [PDF](https://arxiv.org/pdf/2601.23129v1)

**作者:** Yilun Hua `[一作]` (Cornell University), Kevin Small `[通讯]` (Amazon)

**通讯引用:** 1530 | [OpenAlex ID](https://openalex.org/A5028194913)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了无标签的 Grounding Generation Utility（GGU）指标，用来衡量检索文档对 LLM 生成的实用性，并利用该指标训练查询重写器。

**💡 创新点**

创新点在于：①以 LLM 生成置信度（熵）差异衡量文档价值；②通过关键词提取聚焦有信息量的 token；③提供了模型特定且无参考答案的评价方法；④在无注释数据下即可生成有效的训练样本。

**🔧 技术方法**

核心技术包括：熵与困惑度计算、关键 token 筛选、基于 GGU 的 Preference Pair 生成与 DPO 训练、检索器（BM25、ANCE）与 LLM（Phi‑4、Qwen）结合。

**📊 数据集**

使用的数据集包括：Natural Questions（用于评估与验证）、常规检索训练集（BM25/ANCE 索引）以及实验中的大规模语料库，评估时对检索结果与生成答案均有参考黄金答案。

**📈 对比分析**

与 RetPO、ConvGQR、IterCQR、ADACQR 等基线对比，GGU 驱动的查询重写器在 BM25/ANCE 检索场景下分别提升 MRR 高达 18.2 点、答案准确率提升约 9.4%，在多数指标上与基于注释的基线相当或更优。

**⚠️ 局限性**

局限性在于：仅在查询重写任务上验证；对其他 RAG 子组件（如重排序器）的适用性未充分实验；评估开放式回答时仍以精确匹配/准确率衡量，可能低估真实表现；依赖检索器的召回与质量；对模型特异性可能需要额外调参。

---

## 526. Greedy Routing Reachability Games

**arXiv ID:** 2601.23126 | [PDF](https://arxiv.org/pdf/2601.23126v1)

**作者:** Pascal Lenzner `[一作]` (University of Augsburg), Paraskevi Machaira `[通讯]` (Hasso Plattner Institute)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究了在分散式网络中使贪婪路由可行的网络创建博弈，分别分析了有向和无向边两种情形，并给出了可在多项式时间内构造近似纳什均衡的算法。

**💡 创新点**

创新点在于：
- 首次证明有向边博弈的所有纳什均衡即为社会最优，价格无差距；
- 对无向边博弈给出理论上最优的 PoA 上界 1.8（下界 1.75）和上界 2-1/K(D) 的通用上界；
- 设计了一种基于最小贪婪路由集合、最近邻图与 Delaunay 三角剖分的多项式算法，构造出 +2-NE 或 1.8 倍最优边数的近似均衡。

**🔧 技术方法**

技术主要包括：
- 贪婪路由度、最近邻图（NNG）和 Kissing 数的几何性质；
- 证明无向博弈无最佳响应周期并引入 α‑函数、关键边集与关键最优响应；
- 采用 Hakimi 定理与最大流算法进行边权分配；
- 通过构造和简化（如去除无用边、与 Delaunay 三角剖分比较）实现多项式时间。

**📊 数据集**

论文为理论性工作，未使用实际数据集；所有实验与评估均基于理论证明和抽象构造。

**📈 对比分析**

与传统的 Delaunay 三角剖分（3|P|‑6 条边）相比，所构造的近似均衡在 2D 情形下最多只多 80% 边（1.8 倍），而 Delaunay 需要最多 3 倍；在更高维或任意度量空间，PoA 上界为 2‑1/K(D)，远优于 3‑近似；同时证明了最佳响应计算为 NP‑难，凸显了所给算法的实际意义。

**⚠️ 局限性**

局限性包括：
- 对无向边博弈仅得到近似均衡，未证明存在精确纳什均衡；
- 主要针对几何网络，非几何或动态网络的适用性未讨论；
- 证明依赖于贪婪路由可行性条件，实际路由协议或噪声干扰未考虑；
- 对于高维空间，只给出 2‑1/K(D) 的上界，具体常数仍较大。

---

## 527. Exploring Sidewalk Sheds in New York City through Chatbot Surveys and Human Computer Interaction

**arXiv ID:** 2601.23095 | [PDF](https://arxiv.org/pdf/2601.23095v1)

**作者:** Junyi Li `[一作]` (New York University), Takahiro Yabe `[通讯]` (New York University)

**通讯引用:** 1904 | [OpenAlex ID](https://openalex.org/A5075756309)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了基于AI聊天机器人的图像注释调查，用于评估纽约市人行道棚的设计对行人可视性和选择行为的影响。

**💡 创新点**

创新点在于将大型语言模型与图像交互界面结合，生成动态后续提问，实现场景感知与行为决策的结构化、可重复测量数据收集。

**🔧 技术方法**

使用了Google Gemini 1.5 flash LLM、Python、Google Cloud平台、图像分割与信号检测等技术。

**📊 数据集**

采用了NYC 2023‑2025年间的30张街景图（当前、历史、无棚三种状态）以及25名已完成问卷的行人样本。

**📈 对比分析**

通过混合效应模型和逻辑回归评估棚影对门口识别的d'值和侧边选择的概率，结果表明棚存在时d'下降0.16，雨天选择棚边的概率显著提高。

**⚠️ 局限性**

局限性包括样本量小、仅覆盖曼哈顿核心、缺少夜间、拥堵及不同城市背景，以及调查未完成率高导致的可能偏差。

---

## 528. OrLog: Resolving Complex Queries with LLMs and Probabilistic Reasoning

**arXiv ID:** 2601.23085 | [PDF](https://arxiv.org/pdf/2601.23085v1)

**作者:** Mohanna Hoveyda `[一作]` (Radboud University), Faegheh Hasibi `[通讯]` (Radboud University)

**通讯引用:** 572 | [OpenAlex ID](https://openalex.org/A5047151593)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了 OrLog 框架，先用检索器筛选候选实体，再利用大语言模型（LLM）在一次前向推断中给每个谓词估计可信度，随后通过 ProbLog 进行概率逻辑推理，以实现对包含并、或、非等逻辑约束的复杂查询的检索排序。

**💡 创新点**

创新点在于将谓词可信度估计与逻辑推理解耦：LLM只负责无解码的可信度输出，ProbLog负责符号级推理；利用 LLM 的 logits 直接得到可信度，显著降低 token 消耗；在处理 disjunction 结构时相较于 monolithic LLM 具有显著优势。

**🔧 技术方法**

使用的技术包括：大型语言模型（如 Mistral、Qwen、Llama）用于真值估计；ProbLog 概率逻辑编程框架；BM25 与 E5 dense 检索器；解码自由的前向推断与 logits 转化为可信度；逻辑模板解析。

**📊 数据集**

实验数据集为 QUEST，包含 1,727 个基于 Wikipedia 类别组合的查询，覆盖七种逻辑模板（并、或、非等）。

**📈 对比分析**

与检索器原始、LLM-as-reasoner、Logic-augmented LLM 等基线进行比较。结果显示：在提供实体描述（Parametric+）的条件下，OrLog 在 P@1、NDCG@1 等指标显著优于基线，并在 disjunction 查询中表现尤为突出；token 使用量平均降低约 90%。

**⚠️ 局限性**

局限性：在缺乏外部实体描述（Parametric）时，OrLog 的可信度估计不佳，导致性能低于 monolithic LLM；可信度估计高度依赖 LLM 的校准，易受模型误差影响；目前仅支持有限的逻辑运算，尚未扩展到更复杂或多跳推理任务。

---

## 529. RAudit: A Blind Auditing Protocol for Large Language Model Reasoning

**arXiv ID:** 2601.23133 | [PDF](https://arxiv.org/pdf/2601.23133v1)

**作者:** Edward Y. Chang `[一作]` (Stanford University), Longling Geng `[通讯]` (Stanford University)

**通讯引用:** 4114 | [OpenAlex ID](https://openalex.org/A5075703749)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种在无真值环境下的推理诊断协议（Recursive Audit），通过盲审只检查推理路径是否支持结论，从而检测推理过程中的病理；

**💡 创新点**

创新点包括：①将控制理论（PID）与多支柱 CRIT 评分结合，构建可保证有限纠正和终止的闭环诊断框架；②首次量化评估推理过程的“合理性”dial，并探究社交框架对模型回应的影响；③通过实验发现四种推理病理机制（潜在能力抑制、虚假能力陷阱、复杂性易损平衡、治疗性批评），挑战了“能力即鲁棒”假设；

**🔧 技术方法**

技术手段主要包括：基于 PID 的反馈循环控制、CRIT 多支柱评分、信息指示器（JS、Ov、τ）衡量多样性与一致性、盲审员 LLM 评估、对话式审计与干预、以及对推理过程的量化监测；

**📊 数据集**

使用的主要数据集为 CAP‑GSM8K（数学推理的对抗性提示版）和 CausalL2（因果推理的 L2 子集），并与 CausalT5k 等基准对比；此外在不同 LLM 上进行实验评估；

**📈 对比分析**

实验通过将无盲审和盲审两种模式对比，采用 Paranoia Rate、Realignment Rate、Sycophancy Ratio、Net Effect 等指标评估，结果显示：在 Llama 3.3 70B 上可将 96.2% 的潜在能力提升到 96.6%；在数学任务中错误率下降，在因果任务中 sycophancy 率显著升高（>10×）；并证实强审计能揭示弱审计掩盖的错误；

**⚠️ 局限性**

局限性包括：审计者本身是 LLM，可能携带偏见；无法纠正推理过程本身的结构性偏差，导致某些错误无法被修正；未探讨对审计者的对抗攻击；缺乏外部符号/因果推理器；以及在无真值环境下的评估仍有挑战。

---

## 530. Regularisation in neural networks: a survey and empirical analysis of approaches

**arXiv ID:** 2601.23131 | [PDF](https://arxiv.org/pdf/2601.23131v1)

**作者:** Christiaan P. Opperman `[一作]` (University of Pretoria), Katherine M. Malan `[通讯]` (University of South Africa)

**通讯引用:** 1473 | [OpenAlex ID](https://openalex.org/A5082432019)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文综述并分类了神经网络正则化技术，提出四大类（数据、架构、训练、损失）并分析其相互矛盾与对应关系，随后在十个数值与图像分类数据集上对九种主流正则化方法进行基准实验。

**💡 创新点**

创新点在于：①构建了覆盖现有文献的系统化正则化分类体系；②揭示了不同正则化方法之间的矛盾与协同机制；③通过大规模实验验证正则化效果随数据集与网络架构显著变化，强调无“一刀切”方案。

**🔧 技术方法**

使用的技术包括：数据增强（几何变换、SMOTE）、噪声注入、权重扰动、剪枝、Dropout、BatchNorm、LayerNorm、WeightNorm、正则化项（L₂）等；实验平台为PyTorch、Scikit‑Learn、Python 3.10，利用PyHopper进行超参数搜索。

**📊 数据集**

数据集共十个：数值数据集（Diabetes、Liver Cirrhosis、Magic、Mfeat‑pixel、White Wine Quality）和图像数据集（Balls、Bean Leaf Lesions、CIFAR‑10、Fashion MNIST、Shoes）。

**📈 对比分析**

比较方法为在同一模型架构（MLP 或 CNN）下多次独立训练，记录测试 F1 分数；通过对比基线（无正则化）并使用 Mann‑Whitney U 检验统计显著性。结果显示：BatchNorm 在图像集上始终不致使性能下降并有显著提升；Pruning 在数值集上表现最好；其他方法（Dropout、WeightNorm 等）在不同数据集上效果参差不齐，部分数据集出现性能下降。

**⚠️ 局限性**

局限性包括：①实验仅覆盖十个数据集，缺乏对更大规模、不同任务（回归、序列）的验证；②只测试了九种正则化技术，未涵盖最新方法如 SAM、SWA；③缺乏对正则化组合与训练阶段时机的深入探讨；④结果高度依赖超参数调优，未给出通用调参指南。

---

## 531. Synthesizing Petri Nets from Labelled Petri Nets using Token Trail Regions

**arXiv ID:** 2601.23130 | [PDF](https://arxiv.org/pdf/2601.23130v1)

**作者:** Robin Bergenthum `[一作]` (FernUniversität in Hagen), Jakub Kovář `[通讯]` (FernUniversität in Hagen)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6`

**🎯 论文内容**

提出并实现了基于 token trail 区域的 Petri 网合成方法，可从包含状态图、语言或标记网等多种行为规范集合中自动生成最小额外行为的 Petri 网。

**💡 创新点**

核心创新是将 token trail 区域定义为状态基与语言基区域的统一“meta 区域”，同时允许直接使用标记 Petri 网作为规范，从而打破传统方法在并发、冲突和状态合并表达上的权衡。

**🔧 技术方法**

使用 token trail 语义、整数线性规划（ILP）求解最小区域、k‑bound 截断技术，以及基于 GLPK 的实现和 Web 工具来完成合成。

**📊 数据集**

实验以四个示例为基准：①基于标记网的规范；②状态图规范；③包含六条运行的语言规范；④包含循环与弧权的完整 Petri 网，并提供在线可复现工具。

**📈 对比分析**

与传统状态基/语言基合成以及 ILP Miner 对比，token trail 在处理混合规范时能在 30 秒至 6 分钟内完成；对单一状态图只能得到无短环结果，对单一语言图则需更长时间；总体上在合成精度和额外行为控制上优于单一方法。

**⚠️ 局限性**

主要限制包括：规范需要完整且正确（对噪声日志支持有限）；理论上区域集合无穷大，需通过 k‑bound 截断导致可能产生多余行为；对连通性、死锁与 Soundness 的保证仍需后续研究。

---

## 532. "I Choose to Live, for Life Itself": Understanding Agency of Home-Based Care Patients Through Information Practices and Relational Dynamics in Care Networks

**arXiv ID:** 2601.23127 | [PDF](https://arxiv.org/pdf/2601.23127v1)

**作者:** Sung-In Kim `[一作]` (Seoul National University Bundang Hospital), Hwajung Hong `[通讯]` (KAIST)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

通过23次访谈和60小时现场观察，探讨在居家照护网络中患者能动性的表现与被忽视的根源；

**💡 创新点**

首次将患者能动性视为通过持续日常、微小成就和关系互动共同构建的动态能力，并提出缩小信息鸿沟的技术设计思路；

**🔧 技术方法**

采用质性研究方法——半结构访谈、现场观察及扎根理论编码；

**📊 数据集**

数据来源为8名患者、7名医疗专业人员与8名护理工的访谈稿、现场笔记、护理报告及电子病历；

**📈 对比分析**

本研究为探索性定性研究，不涉及算法或性能对比；

**⚠️ 局限性**

样本仅包含能够沟通的患者，且聚焦于首尔地区，可能不具备跨文化与不同资源环境的普适性，并未对非语言表达的患者进行深入探讨。

---

## 533. Now You Hear Me: Audio Narrative Attacks Against Large Audio-Language Models

**arXiv ID:** 2601.23255 | [PDF](https://arxiv.org/pdf/2601.23255v1)

**作者:** Ye Yu `[一作]` (University of Illinois Urbana-Champaign), Haohan Wang `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 2534 | [OpenAlex ID](https://openalex.org/A5072244531)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6215c339-3735-4be3-8a07-5bbb7004712d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

研究了通过音频语调（delivery style）对大型音频语言模型的攻击效果，证明语调控制比单纯文本或中性语音更能诱导模型生成违规内容。

**💡 创新点**

首次将社会心理学中的影响策略（如权威诉求、紧迫指令等）转化为可控制的语音特征，形成一种新的对抗性音频攻击范式。

**🔧 技术方法**

采用文本转语音（TTS）技术配合手工设计的语调模板，以及对端到端音频语言模型的黑盒攻击评估。

**📊 数据集**

使用公开的对抗性数据集AdvBench、JailbreakBench以及20条人工录制语音样本进行实验。

**📈 对比分析**

对比文本输入、中性语音输入与定制语调音频输入，在GPT‑4o Realtime、Gemini 2.0 Flash和Qwen 2.5‑Omni三种模型上评估攻击成功率（ASR），结果显示定制语调能提升10–20%甚至更高的成功率。

**⚠️ 局限性**

攻击对小型模型效果有限；目前只使用少量手工挑选的语调；仅在英语环境下验证，未覆盖多语种或多口音。

---

## 534. Strongly Polynomial Time Complexity of Policy Iteration for $L_\infty$ Robust MDPs

**arXiv ID:** 2601.23229 | [PDF](https://arxiv.org/pdf/2601.23229v1)

**作者:** Ali Asadi `[一作]` (Institute of Science and Technology Austria), Carlo Pagano `[通讯]` (Concordia University)

**通讯引用:** 69 | [OpenAlex ID](https://openalex.org/A5007406156)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

研究了在固定折扣因子下，(s,a)-rectangular L∞不确定集的鲁棒马尔可夫决策过程（RMDP）以及鲁棒马尔可夫链（RMC）的政策迭代算法，并证明该算法在此模型中是强多项式的。

**💡 创新点**

首次给出在固定折扣因子下，(s,a)-rectangular L∞不确定集的RMDP能够通过鲁棒政策迭代得到最优策略且运行时间是强多项式，填补了该领域的关键空白。

**🔧 技术方法**

利用潜能函数（potential function）与组合学论证相结合的技术，分析政策迭代过程中价值函数的收敛性；同时采用自同伦（homotopy）算法实现鲁棒策略改进，并在证明中使用了组合数学关于二进制最高有效位的上界。

**📊 数据集**

无，本文为理论性工作，未使用具体数据集。

**📈 对比分析**

与传统的线性规划或价值迭代方法相比，鲁棒政策迭代在固定折扣因子下不依赖数值精度，能够在多项式次数内得到最优策略；论文给出了 O(n⁴ log n · log(1/γ)/log(1-γ)) 次迭代的上界。

**⚠️ 局限性**

仅适用于固定折扣因子；对非固定折扣因子或更一般的不确定集（如 s-rectangular）尚未给出强多项式结果；实验验证与实际性能评估缺失。

---

## 535. Optimal Fair Aggregation of Crowdsourced Noisy Labels using Demographic Parity Constraints

**arXiv ID:** 2601.23221 | [PDF](https://arxiv.org/pdf/2601.23221v1)

**作者:** Gabriel Singer `[一作]` (University Paris Saclay), Argyris Kalogeratos `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了在众包标签聚合中公平性问题，给出了Majority Vote和Bayesian聚合的非渐近公平性误差上界，并提出了一种可行的post‑processing算法FairCrowd，用于在任意聚合规则下强制满足ε‑demographic parity。

**💡 创新点**

创新点包括：①在小规模众包场景下给出Majority Vote公平性误差的上界，并证明两种聚合器在满足可解释条件时指数收敛至真实标签的公平性；②将先前仅适用于连续输入的ε‑公平post‑processing框架推广到离散输入；③提出基于min‑max和软max的闭式优化方案，兼顾精度与公平性。

**🔧 技术方法**

主要技术手段有：统计学习理论、误差指数与Poisson二项分布不等式、min‑max优化、软max平滑、序列二次规划以及随机分类器的构造。

**📊 数据集**

实验使用的主要数据集包括：①合成数据；②Crowd Judgment（COMPAS再犯预测）和Jigsaw Toxicity（评论毒性）这两个真实众包数据集。

**📈 对比分析**

与FairTD、Post_TD等现有公平性后处理方法对比。实验结果显示，FairCrowd在大多数设置下在保持或略低的F1分数的同时，能够更好地满足不同ε值的demographic parity约束；尤其在小ε或小众包规模时优于对手；在Jigsaw等可靠annotator数据集上，Majority Vote+FairCrowd表现接近或优于Bayesian或DS。

**⚠️ 局限性**

局限性包括：①未考虑敏感特征与输入特征交互导致的混淆矩阵变化；②对多类别高维问题的实现和理论推广仍较复杂；③对极少标注的annotator的概率估计可能不稳定；④未处理annotator选择偏差和任务难度相关的公平性问题。

---

## 536. Med-Scout: Curing MLLMs' Geometric Blindness in Medical Perception via Geometry-Aware RL Post-Training

**arXiv ID:** 2601.23220 | [PDF](https://arxiv.org/pdf/2601.23220v1)

**作者:** Anglin Liu `[一作]` (Hong Kong University of Science and Technology), Jintai Chen `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 1563 | [OpenAlex ID](https://openalex.org/A5023258354)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

本文通过在多模态大语言模型（MLLM）上实施强化学习后训练（Med-Scout），利用无标签医学图像生成的几何代理任务来提升模型的几何感知，消除所谓的“几何盲目”，并提出相应的基准（Med-Scout-Bench）用于定量评估。

**💡 创新点**

创新点包括：① 设计三类几何代理任务（尺度定位、拓扑拼图、异常一致性）以自动生成可验证的监督信号；② 引入密集几何奖励（Dense Geometric Reward, DGR）与 GRPO 强化学习框架相结合，实现对几何约束的连续、细粒度反馈；③ 构建专门评估几何盲点的 Med-Scout-Bench 基准，填补了现有医学视觉评估中缺失的几何维度。

**🔧 技术方法**

技术手段包括：基于 GRPO 的强化学习、密集几何奖励机制、链式推理（CoT）结构奖励、LLM-as-a-Judge 自动评估、无标签医学图像的三类几何代理任务生成、以及对输出格式的严格正则化。

**📊 数据集**

数据集方面：从 108k 生成的 VQA 案例中抽取 10k 作为 Med-Scout-Bench；用于评估的公开医学视觉基准包括 RadImageNet-VQA、VQA-RAD、SLAKE、PMC-VQA、OmniMedVQA、MedXpertQA、MIMIC‑CXR、IU‑Xray 等；训练时使用无标签的 CT/MRI/X‑ray 图像通过 TotalSegmentor、MIMIC‑CXR 等工具自动生成代理任务。

**📈 对比分析**

通过在 Med-Scout-Bench 上对多种基础 MLLM（Qwen3‑VL‑4B/8B、Lingshu‑7B、HuatuoGPT‑Vision‑7B 等）进行后训练，模型平均准确率提升 40%+；在 Radiological VQA、报告生成等公开基准上，Med-Scout‑aligned 模型均超过 GPT‑5、Gemini‑3‑Flash 等专有模型，并在多项指标（Accuracy、ROUGE‑L、CIDEr、SemScore）上刷新 SOTA。

**⚠️ 局限性**

局限性包括：① CoT 结构奖励提升有限，说明模型在逻辑推理上的收益不大；② 代理任务设计对不同医学影像模态的通用性仍需进一步验证；③ 训练仍受限于无标签图像的质量与多样性；④ 对模型规模和算力的依赖较高，可能不适用于资源受限的环境；⑤ 评估集中在几何层面，未全面覆盖语言多样性和临床决策的复杂性。

---

## 537. High-quality generation of dynamic game content via small language models: A proof of concept

**arXiv ID:** 2601.23206 | [PDF](https://arxiv.org/pdf/2601.23206v1)

**作者:** Morten I. K. Munk `[一作]` (Information Technology University of Copenhagen), Paolo Burelli `[通讯]` (Information Technology University of Copenhagen)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文验证了一种通过极度精细化训练的小语言模型（SLM）实现实时动态游戏内容生成的可行性，具体实现了一个针对RPG中声誉冲突的讽刺宣传生成器（DefameLM）。

**💡 创新点**

创新点包括：
1) 将游戏生成任务拆分为高度专业化的子任务，让单一SLM即可完整服务一个完整游戏循环；
2) 采用DAG式合成数据生成方法，精确控制训练样本的多样性与局部性；
3) 对模型进行量化（4/8/16‑bit）并通过“重试至成功”策略，证明低精度模型在保持质量的同时显著提升推理速度；
4) 使用LLM‑as‑Judge 的评价框架客观量化生成质量。

**🔧 技术方法**

技术手段包括：
- 基于 Llama 3.2‑1B 的 LoRA 微调；
- DAG 生成式数据集（1800 条输入/输出对，1440 训练 / 360 验证）；
- 3 层量化（4/8/16‑bit）以及 Llama.cpp 部署；
- 采用 ChatGPT‑4o 进行训练数据生成和判分，使用 ChatGPT‑4o‑mini 进行部分判分；
- 重试至成功策略（温度 0.75）与实时性能测评。

**📊 数据集**

使用的训练数据集为 1800 条合成对话记录，全部由 ChatGPT‑4o 在 DAG 生成流程中自动产生；其中 1440 条用于微调，360 条保留为评测集。

**📈 对比分析**

比较方法：利用 LLM‑as‑Judge 评估输出是否通过 7 条判分（整体、角度、情报、对齐、写作、受众、目标），得到成功率；随后在 AMD Ryzen 9 7950X + RTX 3070 8GB 上测量平均生成时间和期望完成时间。性能结果显示：
- 16‑bit 与 8‑bit 模型成功率均 ~93%，两者在统计上无显著差异；
- 4‑bit 模型成功率约 78%，但平均推理速度比 8‑bit 约 1.5×快，整体期望完成时间在 2–3 s 以内；
- Spearman 相关性 ≥ 0.82 证明不同量化等级下难度排序保持一致，4‑bit 在最难样本上出现较大差异。

**⚠️ 局限性**

限制与挑战：
- 仍需在游戏运行时实现本地质量评估，当前使用的 LLM‑as‑Judge 依赖云端模型；
- 训练与判分均使用同一大模型，可能存在自一致性偏差；
- 仅验证了单一窄域游戏循环，开放式剧情或多任务场景仍待研究；
- 训练样本规模有限，过拟合与世界知识泛化能力未知；
- 对大规模、动态变化的游戏世界的适应性未被评估。

---

## 538. Planar Graph Homomorphisms: A Dichotomy and a Barrier from Quantum Groups

**arXiv ID:** 2601.23198 | [PDF](https://arxiv.org/pdf/2601.23198v1)

**作者:** Jin-Yi Cai `[一作]` (University of Wisconsin-Madison), Ben Young `[通讯]` (University of Wisconsin-Madison)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了对称非负实矩阵参数化的平面图同构计数问题，并给出了完整的可判定复杂度二分法；

**💡 创新点**

首次将量子自同构群与平面边件的表达能力联系起来，揭示了平面化约的根本障碍，并证明了相关判定问题的不可判定性；

**🔧 技术方法**

运用了平面边件构造、插值与矩阵分析、量子群论以及图同构与量子同构的理论；

**📊 数据集**

未使用实验数据集，而是以理论证明为主；

**📈 对比分析**

通过构造可判定的判据与不可判定性证明，表明在矩阵对角线值互异或可分离的情况下，问题为多项式可解或#P-难；

**⚠️ 局限性**

对非平面化可分离但量子自同构群非平凡的情况缺乏完整的二分法，需要新的技术突破。

---

## 539. FourierSampler: Unlocking Non-Autoregressive Potential in Diffusion Language Models via Frequency-Guided Generation

**arXiv ID:** 2601.23182 | [PDF](https://arxiv.org/pdf/2601.23182v1)

**作者:** Siyang He `[一作]` (Fudan University), Xipeng Qiu `[通讯]` (Fudan University)

**通讯引用:** 17443 | [OpenAlex ID](https://openalex.org/A5044665993)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文针对扩散式大语言模型（dLLM）进行频域分析，提出 FourierSampler 通过动态聚焦低频/高频信息实现结构先行、细节后填的解码策略。

**💡 创新点**

创新点在于首次揭示 dLLM 隐状态低频对应结构、高频对应细节，并利用可调频移滑窗、翻译频率得分和自适应频率校准器，实现无需外部指引的端到端解码优化。

**🔧 技术方法**

主要技术包括：频域滤波、翻译频率得分（Translated Filtering Score）、自适应 Fourier 校准器、滑窗频率平移策略、以及多步解码块（block-wise）框架。

**📊 数据集**

实验数据集涵盖数学推导（GSM8K、MATH）、代码生成（MBPP、HumanEval、Countdown）等多任务数据集，模型为 LLaDA 系列（1.5‑8B、8B‑Instruct）与 SDAR 系列（1.7B‑Chat、4B‑Chat）。

**📈 对比分析**

与基线置信度解码、PC‑Sampler、RWS 等方法以及同尺寸自回归模型（Llama3.1‑8B‑Instruct、Qwen2.5‑7B‑Instruct）对比，FourierSampler 在数学任务提升最高达 45.1%，在代码任务提升最高达 20.4%，平均相对提升约 10–15%，并在多数任务中超过所有竞争方法。

**⚠️ 局限性**

局限性包括：仅在两类 dLLM 架构上验证；对超大模型或跨模态扩散模型的适用性未探究；频域划分参数（窗口比例、β 适配）需经验选择；实验规模受 GPU 资源限制，未做大规模用户评测。

---

## 540. ReGuLaR: Variational Latent Reasoning Guided by Rendered Chain-of-Thought

**arXiv ID:** 2601.23184 | [PDF](https://arxiv.org/pdf/2601.23184v1)

**作者:** Fanmeng Wang `[一作]` (Renmin University of China), Zhifeng Gao `[通讯]` (DP Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于 VAE 的 Latent Reasoning 方法 ReGuLaR，通过将 Chain‑of‑Thought 逐段渲染成图像并用视觉编码器生成先验，来引导隐藏状态学习，从而在不依赖显式推理步骤的前提下实现高效推理。

**💡 创新点**

创新点在于：① 将显式推理链可视化为图像，利用视觉文本压缩作为先验信息，显著减少信息损失；② 在 VAE 框架下对先验进行学习与正则化，支持极端压缩（单步推理）和多模态（文本+图形）推理；③ 通过视觉先验实现了对后验分布的有效引导，突破传统 latent 推理的性能瓶颈。

**🔧 技术方法**

使用技术包括：变分自编码器（VAE）框架、视觉编码器 DeepSeek‑OCR、渲染函数与 MLP 适配器、LoRA 微调、KL 散度正则化、以及基于视觉表示的先验构建。

**📊 数据集**

实验数据集涵盖 GSM8K‑Aug、GSM‑Hard、SVAMP、MultiArith、GSM8K‑Aug‑NL、AQUA‑RAT、MATH 以及分子描述生成的多模态基准。

**📈 对比分析**

与 iCoT、CODI、Coconut、CoLaR 等现有 latent 推理方法在上述数据集上对比，ReGuLaR 在准确率上实现了显著提升（平均提升 10–20%），推理长度平均缩短 35%，甚至在多模态任务中超过了显式 CoT；此外在不同模型规模上表现出良好的可扩展性。

**⚠️ 局限性**

局限性包括：对渲染图像质量和视觉编码器性能高度依赖；在极其复杂的推理任务（如大规模数学题或需要深层逻辑推演的场景）中仍可能出现信息损失；目前缺乏理论分析说明为何视觉先验能最优地引导后验分布。

---

## 541. The Iterated Local Model for tournaments

**arXiv ID:** 2601.23246 | [PDF](https://arxiv.org/pdf/2601.23246v1)

**作者:** Anthony Bonato `[一作]` (Toronto Metropolitan University), Teddy Mishura `[通讯]` (Toronto Metropolitan University)

**通讯引用:** 3 | [OpenAlex ID](https://openalex.org/A5080247839)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39`

**🎯 论文内容**

提出了一种基于转移性和克隆操作的迭代本地模型（ILMT），该模型通过在每一步复制节点并对克隆之间的边方向进行选择，生成高度密集的锦标赛图。

**💡 创新点**

创新点包括：① 通过可控的0/1步（反转或保持边方向）实现对锦标赛结构的精细调控；② 证明在无限支持的生成序列下，生成的锦标赛族具有准随机性质、直径≤3、无穷连通度；③ 展示该模型能够在有限步内包含任意大小的子锦标赛，因而可得到无穷多非同构的准随机序列。

**🔧 技术方法**

使用了组合学与图论技术（如诱导子图计数、连通度与直径证明）、概率论框架（准随机性定义与证明）、以及对策游戏与色数分析等方法，对模型性质进行严格推导。

**📊 数据集**

论文主要为理论工作，没有使用外部实验数据或实际网络数据集，所有结果均来自理论证明与计数公式。

**📈 对比分析**

与已有的ILT模型和随机锦标赛性质对比：在无限支持下 ILMT 能产生准随机族，直径与连通度满足小世界特性；对比 0/1 步对支配数、追捕数、色数的影响，提供了上界和下界的理论分析；整体表现基于理论证明，不涉及实验性能评估。

**⚠️ 局限性**

局限性：① 未给出随机化版本或概率化参数的分析；② 色数随步数的精确增长率未知，仅给出对数下界；③ 模型仅适用于完全有向图（锦标赛），未对稀疏有向图进行推广；④ 对于更一般的生成序列（如非二进制或随机混合）如何影响性质仍未解决。

---

## 542. Sequence Diffusion Model for Temporal Link Prediction in Continuous-Time Dynamic Graph

**arXiv ID:** 2601.23233 | [PDF](https://arxiv.org/pdf/2601.23233v1)

**作者:** Nguyen Minh Duc `[一作]` (VNU University of Engineering and Technology), Viet Cuong Ta `[通讯]` (VNU University of Engineering and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 Sequence Diffusion for Dynamic Graphs (SDG)，将连续时间动态图的时序链接预测视为序列级去噪任务并实现端到端的生成式模型；

**💡 创新点**

在历史交互序列和未来目标序列上注入噪声，使用跨注意力去噪解码器进行条件生成，并采用余弦重构损失与排名损失结合，构建一种全新的基于扩散的时序图学习框架；

**🔧 技术方法**

利用扩散模型 (DDPM)、因果 Transformer 编码、跨注意力去噪网络、时间编码、余弦损失以及 BCE/BPR 排名损失等技术；

**📊 数据集**

在十个公开基准上评估：小规模的 Wikipedia、Reddit、MOOC、Lastfm、UCI 以及大规模 TGB-Seq 的 GoogleLocal、YouTube、Flickr、ML-20M、Taobao；

**📈 对比分析**

与七大最先进时序图模型（JODIE、DyRep、TGAT、TGN、GraphMixer、DyGFormer、CRAFT）在 MRR 与 HR@10 上进行对比，SDG 在绝大多数数据集上获得最高或次高分，提升幅度从数个百分点到十几个百分点；

**⚠️ 局限性**

推理时需多步逆扩散，导致比纯判别模型略慢；对极端噪声或重复交互场景的优势有限；目前仅在已知节点特征的情形下评估，缺乏对归纳学习的探索。

---

## 543. ShotFinder: Imagination-Driven Open-Domain Video Shot Retrieval via Web Search

**arXiv ID:** 2601.23232 | [PDF](https://arxiv.org/pdf/2601.23232v1)

**作者:** Tao Yu `[一作]` (Chinese Academy of Sciences), Liang Wang `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 43046 | [OpenAlex ID](https://openalex.org/A5115602506)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 ShotFinder 这一开放领域视频镜头检索基准，并提出了基于文本描述的三阶段检索与定位方法。

**💡 创新点**

创新点包括：① 将镜头级自然语言描述与视频检索对接，采用五个单因子约束；② 通过“视频想象”对查询进行扩展，弥合镜头描述与视频元数据的鸿沟；③ 采用关键帧加 LLM 的双阶段定位与评估策略；④ 对检索结果进行 LLM 辅助评估，克服传统全局视觉-文本相似度的局限。

**🔧 技术方法**

使用的技术主要有：大语言模型（Gemini‑3‑Pro、GPT‑5.2、Claude‑4.0‑Sonnet 等）做查询扩展、定位与评估；搜索引擎检索获取候选视频；自适应帧采样；LLM 辅助的关键帧评估。

**📊 数据集**

数据集为 ShotFinder，1210 条从 YouTube 收集的视频镜头，覆盖 20 个主题，采用 LLM 生成描述并人工校正后形成。

**📈 对比分析**

评估方式：将检索结果的关键帧与输入描述、真值关键帧一起输入 LLM 进行一致性判断；与人工评估对比。结果显示，闭源模型平均准确率约 26.9%，开源模型约 20%；时序（Temporal）性能最好，颜色（Color）与风格（Style）仍显不足。

**⚠️ 局限性**

局限性：① 仅使用关键帧代替完整镜头；② 仅考虑单因子约束，未研究多因子组合；③ 查询生成为单轮交互；④ 仅做基本 URL 过滤，缺乏更细粒度内容筛选；⑤ 帧采样策略未考虑视频内容差异；⑥ 仅评估少量顶尖模型，未来需扩展。

---

## 544. Agile Reinforcement Learning through Separable Neural Architecture

**arXiv ID:** 2601.23225 | [PDF](https://arxiv.org/pdf/2601.23225v1)

**作者:** Rajib Mostakim `[一作]` (Bangladesh University of Engineering and Technology), Sourav Saha `[通讯]` (Virginia Polytechnic Institute and State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了SPAN（Spline-based Adaptive Networks），一种在资源受限的强化学习中具有参数和样本效率的分离 B-spline 架构；

**💡 创新点**

创新点在于结合可学习预处理层与低秩 KHRONOS Tensor B-spline，既保留局部平滑先验又显著降低计算开销；

**🔧 技术方法**

使用了 B-spline 基函数、张量分解、低秩模式、预处理全连接层以及 SAC、PPO、IQL 等 RL 算法；

**📊 数据集**

在经典控制、MuJoCo 连续控制和 Minari D4RL 的 Adroit 手部操作数据集上进行评估；

**📈 对比分析**

与传统 MLP 基线相比，SPAN 在样本效率上提升 30–50%，在在线任务上成功率提升 1.3–9 倍，离线任务上在 Expert 级数据集上平均提升 6.7 倍；

**⚠️ 局限性**

局限包括在资源充足时 MLP 更易实现、对高度不连续或非光滑策略表现有限、需手动调节张量秩、分辨率等超参数。

---

## 545. Are you going to finish that? A Practical Study of the Tokenization Boundary Problem

**arXiv ID:** 2601.23223 | [PDF](https://arxiv.org/pdf/2601.23223v1)

**作者:** Hao Xu `[一作]` (University of Washington), Noah A. Smith `[通讯]` (University of Washington)

**通讯引用:** 30554 | [OpenAlex ID](https://openalex.org/A5088517824)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了语言模型在非空格分词语言（中文、德语、代码）中出现的部分词标记问题（Partial Token Problem）对生成质量的影响，并系统评估其频率与严重程度；

**💡 创新点**

创新点在于从自然语料中构造词边界不对齐的提示，量化误差并验证精确推断时解决方案（Exact Cover Tree）的有效性，展示规模化并未缓解该问题；

**🔧 技术方法**

使用了token healing、Exact Cover Tree等推断时修正技术，以及tokenization对齐率分析与概率对比；

**📊 数据集**

使用了中文Wiki、德语Wiki、CodeXGLUE、FLORES等多语言文本数据集来提取不对齐边界并构造评测对；

**📈 对比分析**

与token‑aligned提示对比，误差可达4个数量级，准确率下降60%–95%；Exact方法可恢复100%准确率，token healing效果不一，且对大模型仍有效；

**⚠️ 局限性**

局限在于仅评估少数模型与任务，未探索训练时随机分词、更多语言与更大规模实验的影响。

---

## 546. Region-Normalized DPO for Medical Image Segmentation under Noisy Judges

**arXiv ID:** 2601.23222 | [PDF](https://arxiv.org/pdf/2601.23222v1)

**作者:** Hamza Kalisch `[一作]` (University Hospital Essen), Frederic Jonske `[通讯]` (University Hospital Essen)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

研究在医学图像分割中，利用噪声质量控制（QC）信号进行无标签的偏好式微调，并提出一种新的 Region‑Normalized DPO（RN‑DPO）算法以提升鲁棒性。

**💡 创新点**

创新点在于：①对比偏好微调在噪声评判下的脆弱性进行系统评估；②设计基于分割不一致区域归一化的 RN‑DPO 目标，降低错误偏好对更新的影响；③在多种基准和评判强度下展示该方法的普适性。

**🔧 技术方法**

主要技术包括：基于 U‑Net 的基模型训练；多样化候选掩码生成（dropout、温度、阈值、形态学编辑等）；使用 QC 模型或集成一致性作为评判；对偏好对进行采矿；在 DPO 框架下引入区间归一化的损失；对比实验与多种基线（随机、Top‑vs‑Base、IPO、rDPO 等）。

**📊 数据集**

使用了两大医学分割基准：JSRT（胸部 X 光多标签分割）和 ACDC（心脏 MRI 多类别分割），并在不同的训练预算下构建弱/强评判和基模型。

**📈 对比分析**

与标准 DPO、随机采矿、IPO、rDPO、伪标记等方法比较，RN‑DPO 在所有评判强度下均显著提升峰值 IoU 与 TailAvg 指标，尤其在弱评判场景下表现更为稳健，且在 GT 评判下仍保持优势。

**⚠️ 局限性**

局限性包括：①依赖于评判模型的质量；②对评判噪声的统计模型尚未充分理论化；③在极低数据预算或评判极差时仍可能出现性能下降；④仅在 2D 分割任务验证，需进一步评估 3D 或多模态情况。

---

## 547. Secure Integrated Sensing and Communication against Communication and Sensing Eavesdropping

**arXiv ID:** 2601.23216 | [PDF](https://arxiv.org/pdf/2601.23216v1)

**作者:** Sidong Guo `[一作]` (Georgia Institute of Technology), Matthieu R. Bloch `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 8053 | [OpenAlex ID](https://openalex.org/A5055689993)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `9cc9baba-5356-466d-81ff-d80028d90279` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究在单波束集成感知与通信（ISAC）系统中，考虑单一发射机同时发送保密信息并进行环境状态估计，同时面对既能窃听信息又能估计状态的被动敌手，探讨在固定状态模型下的安全性与感知指数之间的权衡；

**💡 创新点**

提出了区分信息保密与感知安全的根本区别，并给出了利用反馈生成秘密密钥、可解析性编码（resolvability coding）与一次性密钥（OTP）三种策略的可行区域，阐明了在软覆盖率足够时可实现的三种最优工作点；

**🔧 技术方法**

采用信息理论工具，包括多假设序贯检验、带反馈的组合窃听信道、通道可解析性（soft covering）与Chernoff信息等，构造自适应编码与停机规则；

**📊 数据集**

以二进制对称信道（BSC）为示例的离散状态信道进行数值验证；

**📈 对比分析**

通过绘制通信速率与感知指数（E1、E2）区域，展示三种工作点（P_SO、P_SC、P_CO）之间的权衡，并说明在软覆盖率超越Chernoff指数时，系统可实现较高的保密率与感知隐私；

**⚠️ 局限性**

模型仅适用于固定状态，缺乏对Eve检测指数在其信道优势显著时的闭式分析；此外假设了无噪声、无延迟的安全反馈，实际系统中可能难以满足；

---

## 548. Tackling air quality with SAPIENS

**arXiv ID:** 2601.23215 | [PDF](https://arxiv.org/pdf/2601.23215v1)

**作者:** Marcella Bona `[一作]` (Queen Mary University of London), Xiwen Shirley Zheng `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文通过将Google地图交通色块图转化为环形分段的交通强度特征，利用该特征对墨西哥城空气污染水平进行预测；

**💡 创新点**

创新点在于提出了基于环形分段的交通强度表示方法，能够在没有高密度污染传感器的情况下，用公开的交通数据进行超本地化的空气质量预测；

**🔧 技术方法**

采用了偏最小二乘回归（PLSR）模型，并结合VIP分数进行特征重要性评估；

**📊 数据集**

使用了墨西哥城44个空气质量传感器的9种污染物（PM2.5、PM10、O3、NO、NO2、SO2等）时间序列数据以及对应时段的Google地图交通色块图；

**📈 对比分析**

模型通过5折交叉验证选取最佳成分数，训练-测试分离后得到RMSE指标。结果显示，使用6个站点训练的模型在验证站的RMSE最低，且相较于单站或相似站点训练的模型，预测性能更优；

**⚠️ 局限性**

局限在于仅使用单一时间段（三个月）数据，交通强度特征对天气、时段的考虑不足，且模型仍为线性，难以捕捉非线性关系，预测范围相对真实值较窄。

---

## 549. Network analysis and link prediction in competitive women's basketball

**arXiv ID:** 2601.23193 | [PDF](https://arxiv.org/pdf/2601.23193v1)

**作者:** Anthony Bonato `[一作]` (Toronto Metropolitan University), Morganna Hinds `[通讯]` (Toronto Metropolitan University)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对NCAA女子篮球赛季及WNBA比赛进行网络结构分析，构建对战、投篮阻挡和传球网络，计算CON得分、PageRank及低调领袖强度，并使用node2vec嵌入结合逻辑回归进行链接预测和排名变化分析。

**💡 创新点**

创新点在于提出低调领袖强度（低中心性但结构相似度高的节点指标），以及将node2vec嵌入用于预测季后赛对阵、阻挡行为和传球可行性，展示网络特征对传统排名和未来互动的预测价值。

**🔧 技术方法**

主要技术包括CON得分、PageRank、低调领袖强度计算，node2vec嵌入（p=1,q=1），余弦相似度，嵌入拼接，逻辑回归模型，统计显著性检验（伪R²、似然比检验）。

**📊 数据集**

数据集涵盖2021–2024赛季NCAA女子篮球对战数据（Kaggle），三场WNBA比赛的手工传球记录，以及2023–2024赛季WNBA阻挡数据（PBP Stats）；并对比NCAA男子篮球数据。

**📈 对比分析**

通过在100次独立迭代中评估逻辑回归，得到伪R²约0.02–0.03，似然比检验显著（p<0.05），表明嵌入相似度能显著预测季后赛对阵和阻挡交互；传球预测性能较弱。

**⚠️ 局限性**

局限性包括：传球数据样本极小（仅3场），事件稀疏且随机，缺乏细粒度时空信息，模型仅使用逻辑回归，未加入主客场、伤病、赛程强度等情境变量，网络表示主要基于赛季或分段汇总。

---

## 550. Ensuring Semantics in Weights of Implicit Neural Representations through the Implicit Function Theorem

**arXiv ID:** 2601.23181 | [PDF](https://arxiv.org/pdf/2601.23181v1)

**作者:** Tianming Qiu `[一作]` (Technical University of Munich), Hao Shen `[通讯]` (Research Institute of the Free State of Bavaria)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出HyperINR框架，利用超网络将低维潜向量映射到隐式神经表示（INR）的权重，并通过隐式函数定理阐述数据与权重潜空间之间的映射关系。

**💡 创新点**

创新点在于：①将权重视为数据模态并给出严格的理论证明；②利用隐式函数定理保证潜空间能完整保留数据语义；③在保持极简实验设置的前提下，已实现与现有基线竞争甚至超越的分类性能。

**🔧 技术方法**

技术手段包括：超网络生成权重、联合训练主网络与潜向量、MSE重建损失、隐式函数定理（IFT）分析Jacobian满秩性、PCA可视化、线性插值验证潜空间连续性。

**📊 数据集**

使用的数据集有：二维图像集MNIST与FashionMNIST，以及三维形状集ModelNet40、ShapeNet10和ScanNet10。

**📈 对比分析**

与现有方法的比较采用“轻量化匹配协议”，消除数据增强和扩展策略，直接用MLP分类器评估；结果显示在FashionMNIST、ModelNet40和ShapeNet10上达到或超过最先进水平，且训练时间显著更短。

**⚠️ 局限性**

局限性包括：①需要在训练过程中实现近乎完美重建以满足IFT假设；②对Jacobian满秩性的依赖在大规模或噪声数据下可能不成立；③目前只验证了分类任务，其他下游任务的适用性尚未探究。

---

## 551. MeshGraphNet-Transformer: Scalable Mesh-based Learned Simulation for Solid Mechanics

**arXiv ID:** 2601.23177 | [PDF](https://arxiv.org/pdf/2601.23177v1)

**作者:** Mikel M. Iparraguirre `[一作]` (Universidad de Zaragoza), Elias Cueto `[通讯]` (Universidad de Zaragoza)

**通讯引用:** 6891 | [OpenAlex ID](https://openalex.org/A5076212934)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `4de8e9d8-757b-475f-9627-18a445e50202` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种新的 MeshGraphNet‑Transformer（MGN‑T）框架，用于高分辨率工业级固体力学模拟，能够同时处理变形、接触、自碰撞和塑性等非线性现象。

**💡 创新点**

创新点在于将全局Transformer（physics‑attention）与 MeshGraphNet 的几何归纳偏置相结合，消除传统 MPNN 的信息传播不足（under‑reach）问题，且无需网格降维或层次化处理；此外采用基于物理的切片(tokenization)实现高效的全局注意力。

**🔧 技术方法**

核心技术包括：多阶段处理器（Pre‑processor MPNN → Transformer physics‑attention → Refinement MPNN）、物理注意力机制（使用 Gumbel‑Softmax 切片将节点映射到固定数量的物理 token）、节点/边特征编码（相对位置、厚度、速度、塑性变量等）以及端到端的教师强制训练。

**📊 数据集**

使用了两组数据集：① pi‑beam（多组分、弹塑性能量吸收器对刚体撞击，约16k节点），② deforming plate（含孔洞的弹性板在准静态条件下的冲击，约1.25k节点）。

**📈 对比分析**

与原始 MGN、BSMS‑GNN、EvoMesh、HCMT、M4GN 等基准方法进行对比。MGN‑T 在 pi‑beam 上实现了比 MGN 低 5‑10 倍的相对误差；在 deforming plate 上单步误差降至 0.11（相对误差 3.21 %），参数量仅 0.5 M，约为同类方法的 1/4；同时在推理速度和内存占用上也优于传统深层 MPNN。

**⚠️ 局限性**

局限性包括：未对极大规模 3D 网格进行评估，无法直接处理 Eulerian 流体动力学等不同物理体系；对高度复杂多物理耦合问题的适用性仍需进一步验证。

---

## 552. Outcome-Conditioned Reasoning Distillation for Resolving Software Issues

**arXiv ID:** 2601.23257 | [PDF](https://arxiv.org/pdf/2601.23257v1)

**作者:** Chenglin Li `[一作]` (Concordia University), Chen `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `8d10c613-917e-4880-9716-17789f50e119` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了Outcome-Conditioned Reasoning Distillation框架，通过在已验证修复结果的指导下，逆向重建多阶段修复思路，并在推理时将其作为提示，以提升大型仓库中的软件缺陷修复效率。

**💡 创新点**

创新点在于：1) 利用已验证的补丁结果作为全局约束，逆向重构可重用的阶段化修复计划；2) 引入Exemplar Guardian对检索到的历史修复进行保守的可迁移性筛选；3) 通过无参数更新的方式在推理时直接注入这些计划，避免昂贵的前向搜索和在线微调。

**🔧 技术方法**

核心技术包括：① 基于文本相似度和语义对齐的历史缺陷检索；② LLM-as-a-judge的Exemplar Guardian筛选；③ Backward Reasoning Distillation（BRD）在已知终点（补丁）下进行条件化的多步生成与逐步细化；④ 与Agentless等现有程序修复工作流集成。

**📊 数据集**

使用的数据集为SWE‑Bench Lite，它包含300个真实世界Python项目的缺陷及其对应补丁与测试用例，且对不同难度级别进行了标注。

**📈 对比分析**

与同一基础模型（GPT‑4o、GPT‑5、DeepSeek‑V3）下的Agentless以及多种前沿方法（SWE‑Agent、OpenHands、SWE‑Search等）对比，Outcome‑Conditioned Reasoning Distillation 在Pass@1上分别提升了约10%（GPT‑4o）/8.6%（DeepSeek‑V3）/10.3%（GPT‑5），并在文件/函数定位准确率上也有明显提升；相较于MCTS等前向搜索方法，在同等推理成本下可获得13%更高的Pass@1，并显著降低LLM调用次数与token量。

**⚠️ 局限性**

局限性包括：① 主要评估在Python项目的SWE‑Bench Lite上，可能对其他语言或更大规模项目的泛化有限；② 依赖同仓库中足够的历史修复记录，缺乏修复案例的项目难以受益；③ Exemplar Guardian与BRD对LLM的推理可靠性敏感，若模型推理不稳定，筛选与重构效果可能下降。

---

## 553. YuriiFormer: A Suite of Nesterov-Accelerated Transformers

**arXiv ID:** 2601.23236 | [PDF](https://arxiv.org/pdf/2601.23236v1)

**作者:** Aleksandr Zimin `[一作]` (Massachusetts Institute of Technology), Philippe Rigollet `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 3552 | [OpenAlex ID](https://openalex.org/A5001421646)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过将 Transformer 块视为在 token 配置上的离散优化算法，提出了利用 Nesterov 加速的 YuriiFormer 变体。

**💡 创新点**

关键创新在于将经典加速优化模板直接嵌入 Transformer 架构，同时保持自注意力和 MLP 为预训练的梯度或子梯度算子。

**🔧 技术方法**

使用变分能量表述、Lie–Trotter 与 Euler 切分、Nesterov 与 Polyak 动量法以及预层归一化等技术。

**📊 数据集**

在 TinyStories 与 OpenWebText 两个语言建模数据集上进行实验。

**📈 对比分析**

与 nanoGPT 基线在相同参数量与训练步数下比较，Nesterov+Lie–Trotter 在验证交叉熵和下游 HellaSwag/ARC‑Easy 任务上持续优于传统 GPT‑style，表现出 0.02–0.05 的 nats/token 或 1–2% 的准确率提升。

**⚠️ 局限性**

主要局限在于理论收敛性缺失、仅在中小规模模型与短上下文上验证、以及对更大规模或长序列的泛化未知。

---

## 554. Toward Digital Twins in 3D IC Packaging: A Critical Review of Physics, Data, and Hybrid Architectures

**arXiv ID:** 2601.23226 | [PDF](https://arxiv.org/pdf/2601.23226v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 555. Hi-Light: A Path to high-fidelity, high-resolution video relighting with a Novel Evaluation Paradigm

**arXiv ID:** 2601.23167 | [PDF](https://arxiv.org/pdf/2601.23167v1)

**作者:** Xiangrui Liu `[一作]` (Arizona State University), Yezhou Yang `[通讯]` (Arizona State University)

**通讯引用:** 4387 | [OpenAlex ID](https://openalex.org/A5002278578)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了Hi-Light框架，实现训练无关的高质量视频重光照，解决细节损失与光照抖动问题。

**💡 创新点**

三大创新模块：光度先验引导的进阶融合扩散、混合运动自适应光照平滑滤波器(HMA‑LSF)、LAB细节融合(LAB‑DF)，以及首个光照稳定度量Light Stability Score。

**🔧 技术方法**

采用扩散模型引导重光照、光度先验、光流补偿+双边滤波、LAB色彩空间细节保留等技术，完全无训练推理。

**📊 数据集**

使用100段公开与自录视频（70人像、30环境），1080p–2160p，标准化81帧/24fps进行实验与评估。

**📈 对比分析**

与Light‑A‑Video、TC‑Light等SOTA对比，使用SSIM与Light Stability Score评估；Hi‑Light在SSIM 0.943（比LAV 0.604高56%）和Light Stability Score 0.509（比二者高80%）上表现最佳。

**⚠️ 局限性**

对高对比度或饱和高光场景的抑制效果有限，且对阴影与高光细粒度控制仍存在不足。

---

## 556. Compressed Set Representations based on Set Difference

**arXiv ID:** 2601.23240 | [PDF](https://arxiv.org/pdf/2601.23240v1)

**作者:** Travis Gagie `[一作]` (Dalhousie University), Gonzalo Navarro `[通讯]` (University of Chile)

**通讯引用:** 20182 | [OpenAlex ID](https://openalex.org/A5080743153)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `fede83ac-7505-405f-ab37-e7284695c47f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种基于集合对称差压缩的新表示方法，可在 O(Δ) 词空间内支持集合的成员、访问、秩、前驱/后继等五种基本查询。

**💡 创新点**

核心创新包括：① 定义了集合集合的对称差压缩量 Δ；② 用 Prim 算法的增量化变种快速构造最优的对称差最小生成树；③ 在树抽取框架上实现 O(log u) 时间的访问和前驱/后继查询，支持带删除标记的祖先标号检索。

**🔧 技术方法**

采用的技术主要有：树抽取（tree extraction）框架、带 0/1 标签的分层树、增量化 Prim 最小生成树、后缀树辅助对称差计算、位图与路径计数的组合实现。

**📊 数据集**

论文未给出具体实验数据集；讨论了常见的应用场景（布尔矩阵、图邻接表、倒排索引、网页图、颜色 De Bruijn 图等）并以这些典型数据结构为理论基础进行分析。

**📈 对比分析**

与之前的标准压缩方法（如基于包含熵、直接差集压缩、O(s² log s) 的 MST 构造）相比，提出的算法在构造时间上从 O(nlog u + s² log s) 减少到 O(nlog u + min(s²ℓ, sn))（ℓ 为最重边权），在查询时间上保持与传统方法相同的 O(log u) 或更优的 O(log_ω u)（成员查询）。总体性能在理论上显著提升，尤其在集合数量大、单个集合较小的场景。

**⚠️ 局限性**

局限性包括：
• 结构占用 O(Δ) 词而非紧凑的 O(Δ·u + o(Δ·u)) 位；
• 对所有五种查询的时间均为 O(log u)，无法同时实现成员 O(log_ω u) 与其它查询 O(log u) 的最优组合；
• 对称差压缩量 Δ 仍需通过全图最小生成树计算，最坏情况仍为 O(s²) 复杂度；
• 对于需要添加额外子集以进一步压缩的情况，算法无法保证多项式时间的最优解；
• 未覆盖集合补集、并/交等高级操作的高效支持。

---

## 557. How well do generative models solve inverse problems? A benchmark study

**arXiv ID:** 2601.23238 | [PDF](https://arxiv.org/pdf/2601.23238v1)

**作者:** Patrick Krüger `[一作]` (Institute of Mathematics), Hanno Gottschalk `[通讯]` (Institute of Mathematics)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `40105733-5154-44cd-8090-a8cab9e64b07` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究并比较了条件流匹配、可逆神经网络、条件Wasserstein‑GAN以及传统贝叶斯逆向方法在气体涡轮燃烧室设计逆问题上的性能。

**💡 创新点**

提出了针对低维目标高维设计空间的逆问题的标准化评估框架，并证明条件流匹配在数据效率、精度和多样性上显著优于其他方法。

**🔧 技术方法**

采用可逆神经网络、条件流匹配、条件Wasserstein‑GAN和基于前向模型的贝叶斯逆推，以及多尺度代理仿真。

**📊 数据集**

使用从CFD模拟获得的1295条燃烧室参数‑性能对，并通过代理模型扩增到25万条合成数据。

**📈 对比分析**

通过目标性能标签的平均绝对误差和生成设计的多样性分布进行评估，条件流匹配在所有数据规模下均实现最低误差且保持较高多样性。

**⚠️ 局限性**

结果依赖代理模型的精度，实际高昂的CFD验证有限，且研究仅针对六维设计空间，难以直接推广到更高维复杂工程问题。

---

## 558. Video-o3: Native Interleaved Clue Seeking for Long Video Multi-Hop Reasoning

**arXiv ID:** 2601.23224 | [PDF](https://arxiv.org/pdf/2601.23224v1)

**作者:** Xiangyu Zeng `[一作]` (Nanjing University), Limin Wang `[通讯]` (Nanjing University)

**通讯引用:** 21722 | [OpenAlex ID](https://openalex.org/A5100436505)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种原生交互工具调用框架，支持长视频多跳推理的自适应线索寻找与答案推理；模型在查询与当前视觉观察的引导下主动定位关键视觉线索，利用VideoCrop工具检索关键片段，迭代检索并在获得足够证据后终止推理。

**💡 创新点**

① 通过 Task‑Decoupled Attention Masking（TDAM）将线索定位与答案推理的注意力分离，避免注意力分散；② 设计 Verifiable Trajectory‑Guided Reward（VTGR）对工具调用路径进行奖励，引导模型高效检索并及时终止；③ 开发大规模自动合成管线，构建 173k 条工具交互轨迹的 Seeker‑173K 数据集，提供丰富的训练样本。

**🔧 技术方法**

原生多轮工具调用（VideoCrop），注意力掩码（TDAM），轨迹导向奖励（VTGR）与 GRPO 强化学习，SFT+RL 细调，token 预算管理与动态工具调用控制，数据合成与逻辑一致性检查。

**📊 数据集**

训练使用 Seeker‑173K（173k 条工具交互轨迹），评估采用 MLVU、LVBench、LongVideoBench、VideoMME、VideoMMMU、Video‑Holmes、Charades‑STA 等长视频理解与推理基准。

**📈 对比分析**

与现有 SOTA 进行直接对比，结果显示显著提升：在 MLVU 上 72.1%（SOTA 71.9%），Video‑Holmes 46.5%（SOTA 46.1%），VideoMME 66.5% 等；在多跳推理与线索定位方面表现优异，证明多轮工具调用与奖励机制带来的准确率提升。

**⚠️ 局限性**

仍需大量高质量训练数据与算力，工具调用会消耗 token，虽然通过奖励减少冗余，但在极长或高帧率视频中仍可能超出预算；模型对未见场景或复杂视觉线索的鲁棒性有限；扩展至其他任务或工具时需要重新设计掩码与接口。

---

## 559. TSAQA: Time Series Analysis Question And Answering Benchmark

**arXiv ID:** 2601.23204 | [PDF](https://arxiv.org/pdf/2601.23204v1)

**作者:** Baoyu Jing `[一作]` (University of Illinois at Urbana-Champaign), Hanghang Tong `[通讯]` (University of Illinois at Urbana-Champaign)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本研究提出了一个名为TSAQA的大规模时间序列问答基准，涵盖210k条样本、13个领域、6个任务和3种问题类型（TF、MC、PZ）

**💡 创新点**

创新点在于统一了传统与先进的时间序列分析任务，首次引入PZ（拼图）问题以考察全局时序推理，并提供多领域多任务的标准化评测框架

**🔧 技术方法**

采用LLM进行任务定义与数据生成（GPT‑4o、GPT‑4.1等），并通过指令调优与LoRA对开源模型进行微调，评估其在QA形式下的时序推理能力

**📊 数据集**

数据集来自13个公共域的核心时间序列、异常检测基准（ECG、SMD等）和UCR分类集，经过长度、缺失率等过滤后构建成统一的问答对

**📈 对比分析**

通过与GPT‑4.1、Claude‑3.5‑Sonnet、Gemini‑2.5‑Flash等商业LLM及LLaMA3.1‑8B、Qwen3‑8B等开源模型在零样本和指令调优两种设置下对比，发现商业模型平均分约65，指令调优后开源模型可逼近甚至超越（最高69.7），但PZ等高级任务仍明显困难

**⚠️ 局限性**

局限在于缺乏不规则/混频/外生驱动的数据，任务仅覆盖结构化问答，数据集为静态且对模型局部平滑偏好产生负面影响，未来需扩展更真实场景与动态适应性评估

---

## 560. Large Language Models for Patent Classification: Strengths, Trade-offs, and the Long Tail Effect

**arXiv ID:** 2601.23200 | [PDF](https://arxiv.org/pdf/2601.23200v1)

**作者:** Lorenzo Emer `[一作]` (Scuola Superiore Sant’Anna), Andrea Vandin `[通讯]` (Scuola Superiore Sant’Anna)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对比了传统监督编码器（BERT、SciBERT、PatentSBERTa）与开放权重大型语言模型（Phi‑3、Mistral、LLaMA、Qwen）在CPC子类多标签专利分类任务中的表现，探讨了零样本、少样本、检索增强提示以及LoRA微调等不同推理策略，并评估了它们在准确率、长尾覆盖、能耗与延迟等方面的权衡。

**💡 创新点**

创新点在于：① 在同一实验框架下系统评估了编码器与LLM在长尾类别上的互补优势；② 引入检索增强生成（RAG）与少样本提示的组合，提升LLM在稀有子类上的召回；③ 结合CodeCarbon等工具量化能耗与碳排放，为模型可持续性提供实证依据；④ 在外部EPO数据集上验证结果的稳健性，展示跨机构迁移性。

**🔧 技术方法**

使用的技术包括Transformer编码器微调（BERT、SciBERT、PatentSBERTa）、instruction‑tuned 7–8B LLMs（Phi‑3、Mistral、LLaMA‑3.1、Qwen‑2.5）、零样本/少样本提示、检索增强提示（RAG）、LoRA参数高效微调、基于E5 bi‑encoder的子类检索、微调后的推理与能耗监测（CodeCarbon）。

**📊 数据集**

主要数据集为USPTO‑70k（约7万份专利，按时间划分训练/验证/测试），并在外部10k份欧洲专利办公室（EPO）专利上进行不进行任何再训练的外部验证；数据使用标题+摘要拼接文本，四字符级别CPC子类标签。

**📈 对比分析**

比较方法：在相同推理设置下对所有模型进行微分层次评估（Micro‑F1、Macro‑F1、层级F1）以及能耗/耗时测量。结果显示：编码器在频繁子类上Micro‑F1最高，但在Macro‑F1与稀有子类召回上表现差；LLM在零样本时性能最低，但通过少样本+RAG后可提升至与编码器相当的层级召回，并在稀有子类上显著优于编码器；然而LLM的推理时间与能耗比编码器高1–2个数量级；LoRA微调对性能提升有限。

**⚠️ 局限性**

局限性包括：① LLM在本研究仅限于7–8B参数范围，未覆盖更大规模模型；② LoRA微调未显著改善，可能需要更深层微调或全模型微调；③ 仅评估了少数提示策略，未探索更复杂的提示或人机交互；④ 能耗评估基于单GPU推理，实际大规模部署需考虑多GPU/分布式情况；⑤ 仅在USPTO与EPO两国专利上验证，跨语言/跨分类体系的推广仍待研究。

---

## 561. Deep Search with Hierarchical Meta-Cognitive Monitoring Inspired by Cognitive Neuroscience

**arXiv ID:** 2601.23188 | [PDF](https://arxiv.org/pdf/2601.23188v1)

**作者:** Zhongxiang Sun `[一作]` (Renmin University of China), Jun Xu `[通讯]` (Renmin University of China)

**通讯引用:** 13759 | [OpenAlex ID](https://openalex.org/A5020766468)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种深度搜索框架DS-MCM，在推理–检索循环中嵌入层级化的元认知监控机制；

**💡 创新点**

通过将人类元认知的“快速一致性监控”与“慢速经验驱动反思”相结合，形成了对检索证据不确定性与内部推理不确定性一致性的轻量化校准，并利用历史轨迹构建经验记忆来进行反思式调节；

**🔧 技术方法**

使用检索熵(SE)与推理熵(RE)的熵计算、语义聚类、向量检索、FAISS索引、LLM作为基线推理模型以及LLM‑Critic与经验记忆驱动的Critic模型；

**📊 数据集**

在BrowseComp‑Plus、BrowseComp‑ZH、xbench‑DeepSearch、GAIA四大深度搜索基准上进行评测，并在Who&When基准上验证错误定位能力；

**📈 对比分析**

与OpenAI、Google等专有系统以及开源基线（LLM‑Critic）对比，DS‑MCM在所有基准上均显著提升准确率，并在开源模型中超越多款专有系统；

**⚠️ 局限性**

仍需手动调节阈值与经验记忆规模，依赖丰富的历史轨迹，且在极端多源噪声或实时检索场景下的鲁棒性尚未充分验证；

---

## 562. JobResQA: A Benchmark for LLM Machine Reading Comprehension on Multilingual Résumés and JDs

**arXiv ID:** 2601.23183 | [PDF](https://arxiv.org/pdf/2601.23183v1)

**作者:** Casimiro Pio Carrino `[一作]` (Avature Machine Learning), José A. R. Fonollosa `[通讯]` (Universitat Politècnica de Catalunya)

**通讯引用:** 1941 | [OpenAlex ID](https://openalex.org/A5004224082)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 JobResQA，一个用于评估大语言模型在简历与职位描述上的多语言机器阅读理解的基准，包含 105 对合成简历-职位描述和 581 组问答。

**💡 创新点**

其创新点在于将真实简历/职位信息通过去标识化与合成技术生成可控的、带占位符的多语言数据，并配合 TEaR 机器翻译与人类校正的翻译管线，实现高质量、可复现的多语言 HR 问答集合。

**🔧 技术方法**

技术实现包括 LLM 驱动的合成与翻译、占位符替换与归一化、MQM 错误标注与估计、少量样本的精炼步骤，以及使用 G‑Eval 的 LLM‑as‑Judge 评估框架。

**📊 数据集**

所使用的数据集是 5 语言（英语、西班牙语、意大利语、德语、中文）的 105 对合成简历‑职位描述和 581 组 QA，包含三种难度层级（基础、交叉、跨文档推理）。

**📈 对比分析**

在基线评估中，Mistral、Llama、Gemma 等开源多语言 LLM 通过零样本提问展示了英语和西班牙语表现较好（得分 0.6–0.9 级别），但在德语、意大利语和中文上性能显著下降（多数落在 0.3–0.6 级别），说明跨语言能力仍有提升空间。

**⚠️ 局限性**

局限性包括翻译质量仍受自动化与人类校正的限制、问答标注的主观性可能影响评估一致性，以及合成数据虽保障隐私但可能缺乏真实简历的写作风格与细节。

---

## 563. TriSpec: Ternary Speculative Decoding via Lightweight Proxy Verification

**arXiv ID:** 2601.23180 | [PDF](https://arxiv.org/pdf/2601.23180v1)

**作者:** Haoyun Jiang `[一作]` (Shanghai Jiao Tong University), Jiangchao Yao `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 2382 | [OpenAlex ID](https://openalex.org/A5102922412)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出三元推测解码框架 TriSpec，通过引入轻量代理验证器降低验证成本，实现更高推理速度。

**💡 创新点**

创新点在于将验证成本视为关键瓶颈，使用同族小模型作为代理验证器，并采用置信度边际阈值进行可信路由，显著减少大模型调用。

**🔧 技术方法**

技术包括单层 EAGLE‑style 草稿器、代理验证器、margin‑based 路由、token pruning 等。

**📊 数据集**

实验使用 Qwen3、DeepSeek‑R1‑Distill‑Qwen、DeepSeek‑R1‑Distill‑LLaMA 系列模型，并在 GSM8K、MATH500、Gaokao‑2023‑EN、HumanEval、MBPP、SpecBench 等多种推理与代码生成基准上评测。

**📈 对比分析**

与标准推测解码和 SpecCascade 等方法对比，TriSpec 在保持 1%以内精度损失的前提下，平均提升 30% 速度，最大可达 35% 加速，目标模型调用率下降至 50% 以下。

**⚠️ 局限性**

局限性包括对代理模型与目标模型对齐程度的依赖，边际阈值选择需经验调优，且在极度不确定或高复杂度生成任务中仍可能需要频繁调用大模型。

---

## 564. Beyond Fixed Frames: Dynamic Character-Aligned Speech Tokenization

**arXiv ID:** 2601.23174 | [PDF](https://arxiv.org/pdf/2601.23174v1)

**作者:** Luca Della Libera `[一作]` (Concordia University), Mirco Ravanelli `[通讯]` (Concordia University)

**通讯引用:** 1726 | [OpenAlex ID](https://openalex.org/A5040811098)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出 DyCAST，一种能够通过软字符对齐和显式持续时间建模实现可变帧率分词的语音编码器，并在低帧率下通过检索增强解码提升重构质量。

**💡 创新点**

创新点在于：①不需要文本推断即可实现可变帧率分词；②使用软字符对齐与持续时间模型实现动态边界；③引入检索增强解码，在不增加比特率的前提下恢复细节。

**🔧 技术方法**

采用的技术包括：WavLM 预训练特征提取、焦点调制压缩/解压模块、基于概率危险模型的动态分块、负二项持续时间预测、球面量化（SSQ）以及 FAISS 近邻检索。

**📊 数据集**

训练数据为 LibriTTS，评估使用 LibriSpeech、MLS、VoiceBank、Libri1Mix、VCTK、IEMOCAP 等公开数据集。

**📈 对比分析**

与多种固定帧率编码器（EnCodec、FocalCodec 等）在语音重构、语音转换、ASR、SI、SER 和 TTS 等任务上对比，DyCAST 在相同或更低帧率下实现自然度、可懂度、WER 等指标与固定帧率相当甚至更优，TTS 性能尤为突出。

**⚠️ 局限性**

局限性：低帧率下细节恢复仍受限；检索增强解码依赖大规模检索库；训练与推理需要较大计算资源；在极低帧率（≈6 Hz）下 ASR 误差明显上升。

---

## 565. Names Don't Matter: Symbol-Invariant Transformer for Open-Vocabulary Learning

**arXiv ID:** 2601.23169 | [PDF](https://arxiv.org/pdf/2601.23169v1)

**作者:** İlker Işık `[一作]` (Boston University), Wenchao Li `[通讯]` (Boston University)

**通讯引用:** 3395 | [OpenAlex ID](https://openalex.org/A5100381719)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种符号不变的Transformer架构，利用并行嵌入流和聚合注意力实现对可互换符号的α-等价不变性。

**💡 创新点**

通过并行嵌入流和聚合注意力提供了形式化的α-等价不变性保证，兼顾可扩展词汇与精确区分。

**🔧 技术方法**

Transformer、并行嵌入流、聚合注意力、共享权重、Cosine损失、RoPE位置编码等技术。

**📊 数据集**

DeepLTL的Propositional Logic、LTL Random35、复制任务及改名/样本减少扰动等基准数据集。

**📈 对比分析**

与全词汇基线、随机嵌入、α-重命名基线以及GPT‑5.2进行对比，实验表明在α-等价测试中实现100% alpha‑covariance，且在开放词汇推理任务上取得最优或与GPT‑5.2相当的准确率。

**⚠️ 局限性**

多流带来的 O(SL²) 计算复杂度，极大 S 时效率下降，未来需采用稀疏或低秩注意力等方法改进。

---

## 566. GrepRAG: An Empirical Study and Optimization of Grep-Like Retrieval for Code Completion

**arXiv ID:** 2601.23254 | [PDF](https://arxiv.org/pdf/2601.23254v1)

**作者:** Baoyi Wang `[一作]` (Zhejiang University), Jianwei Yin `[通讯]` (Zhejiang University)

**通讯引用:** 7182 | [OpenAlex ID](https://openalex.org/A5069353502)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了一种基于轻量级词法检索的仓库级代码补全方法——GrepeRAG，并通过后处理技术提升检索质量。

**💡 创新点**

创新点在于：①让LLM自动生成 ripgrep 命令实现无索引检索；②引入识别符加权的 BM25 重新排序和基于行号的结构化去重融合，解决了关键词歧义和上下文碎片化问题。

**🔧 技术方法**

采用的技术包括检索增量生成 (RAG) 框架、ripgrep 词法检索、BM25 加权排序、行号去重融合、LLM 指令生成与知识蒸馏等。

**📊 数据集**

实验数据集为 CrossCodeEval（Python/Java）与 RepoEval_Updated（大规模 Python/Java）两大代码补全基准。

**📈 对比分析**

与无检索、VanillaRAG、GraphCoder、RepoFuse、RLCoder 等五个基线对比，GrepeRAG 在 CrossCodeEval 上 EM 提升 7–15%，在 RepoEval_Updated 上代码 EM 提升 13%+，且检索延迟低于 0.5 秒，明显优于图模型。

**⚠️ 局限性**

局限性：仍依赖词法匹配，难以处理隐式依赖；高频词干扰可能影响检索质量；在极大项目中命令生成质量不佳时性能下降。

---

## 567. Training-Free Test-Time Adaptation with Brownian Distance Covariance in Vision-Language Models

**arXiv ID:** 2601.23253 | [PDF](https://arxiv.org/pdf/2601.23253v1)

**作者:** Yi Zhang `[一作]` (Shenzhen University), Liang-Jie Zhang `[通讯]` (Southern University of Science and Technology)

**通讯引用:** 29619 | [OpenAlex ID](https://openalex.org/A5100425201)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种无训练、无反向传播的测试时自适应方法TaTa，能够在域迁移下动态调整视觉-语言模型的表示。

**💡 创新点**

创新地将Brownian距离协方差用于衡量视觉-语言特征的非线性相关性，并结合属性增强的提示和多模态聚类实现伪标签的动态更新与软投票融合。

**🔧 技术方法**

使用了Brownian距离协方差、属性辅助提示、动态多模态聚类、伪标签自适应、软投票机制，以及ViT‑B/16 CLIP编码器。

**📊 数据集**

在跨数据集通用性基准（Aircraft、Caltech101、Cars、DTD、EuroSAT、Flower102、Food101、Pets、SUN397、UCF101）和域泛化基准（ImageNet‑A、V2、R、S）上进行实验。

**📈 对比分析**

与七种现有方法（CoOp、CoCoOp、Tip‑Adapter、TPT、DiffTPT、TDA及原始CLIP）对比，TaTa在平均准确率上提升约2–3%，同时测试时间仅为13.5分钟，比TPT的12小时和TDA的16分钟大幅降低。

**⚠️ 局限性**

仅在图像分类任务上验证，且仍依赖CLIP预训练特征，伪标签生成和属性提示的质量会影响性能，难以直接推广到更复杂的多模态场景。

---

## 568. Structured Over Scale: Learning Spatial Reasoning from Educational Video

**arXiv ID:** 2601.23251 | [PDF](https://arxiv.org/pdf/2601.23251v1)

**作者:** Bishoy Galoaa `[一作]` (Northeastern University), Sarah Ostadabbas `[通讯]` (Northeastern University)

**通讯引用:** 2271 | [OpenAlex ID](https://openalex.org/A5031787107)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

利用《Dora the Explorer》教育节目的结构化问答内容，对 Vision‑Language 模型进行自监督强化学习 fine‑tuning，从而提升其空间推理能力。

**💡 创新点**

创新点在于提出结构化教育视频作为“自监督”训练信号，利用问答-暂停-答案的教学流程与 Group Relative Policy Optimization (GRPO) 结合，实现少量数据（约38小时）即可显著提升多模态推理。

**🔧 技术方法**

技术包括自动抽取时间对齐的问答对、构建 DoraVQA 数据集、使用 GRPO 进行强化学习微调，并对生成答案与真实答案计算 F1 与 Levenshtein 距离的奖励。

**📊 数据集**

使用的数据集为 DoraVQA（5,344 QA 对），并在 Video‑MME、CVBench、NExT‑QA 等公开基准上进行零样本评估。

**📈 对比分析**

与基线 SFT 和其他 RL 方法相比，GRPO 在 DoraVQA 上提升 13–15 分，CVBench 上达到 86.16%（SOTA），在 Video‑MME 与 NExT‑QA 上亦获得 1.6–12 分的显著增益，证明结构化学习能弥补规模不足。

**⚠️ 局限性**

局限性包括对计数任务仍显不足，因缺乏视觉感知而无法彻底解决；此外仅以单一节目为训练来源，泛化范围仍受限。

---

## 569. Multi-Agent Systems Should be Treated as Principal-Agent Problems

**arXiv ID:** 2601.23211 | [PDF](https://arxiv.org/pdf/2601.23211v1)

**作者:** Paulius Rauba `[一作]` (University of Cambridge), Mihaela van der Schaar `[通讯]` (University of Cambridge)

**通讯引用:** 22347 | [OpenAlex ID](https://openalex.org/A5012339002)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

将多智能体系统视为主从代理问题，阐述信息不对称与目标不一致导致的代理损失，并提出采用微观经济学中的机制设计方法来缓解“scheming”现象。

**💡 创新点**

创新点在于：①将语言模型驱动的多智能体系统与经典主从代理框架对接，系统性揭示信息不对称与目标冲突的来源；②把“scheming”视为主从代理问题中的隐性行为（moral hazard）与逆向选择（adverse selection），为后续机制设计提供理论依据；③建议将机制设计工具（激励规则、筛选机制、审计技术等）迁移到 AI 多智能体场景。

**🔧 技术方法**

采用了微观经济学的主从代理模型、机制设计理论（决策规则与转移规则）、以及对 LLM 行为的理性优化假设，辅之以案例分析与文献综述。

**📊 数据集**

论文未使用实验数据集，主要基于理论推导与已有文献综述。

**📈 对比分析**

无实验对比，文章以理论分析为主；若要评估，可与传统多智能体对齐方案（如强化学习奖励调优）进行比较，预期机制设计能显著降低代理损失。

**⚠️ 局限性**

局限性包括：①缺乏经验验证与量化评估；②对 LLM 真实行为的理性优化假设可能过于简化；③机制设计的实现细节在高维 LLM 环境中仍具挑战；④对“scheming”现象的具体情境与触发条件研究不足。

---

## 570. Evaluating the Viability of Additive Models to Predict Task Completion Time for 3D Interactions in Augmented Reality

**arXiv ID:** 2601.23209 | [PDF](https://arxiv.org/pdf/2601.23209v1)

**作者:** Logan Lane `[一作]` (Virginia Tech), Doug A. Bowman `[通讯]` (Virginia Tech)

**通讯引用:** 14068 | [OpenAlex ID](https://openalex.org/A5069853790)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文通过两项实验验证了在3D增强现实界面中使用加性GOMS/KLM模型预测交互完成时间的可行性；

**💡 创新点**

创新点在于将已有的运动时间与确认操作模型（Fitts法、角度ID等）扩展至多模态（手柄、眼动、触碰等）并在AR环境中系统评估其准确性；

**🔧 技术方法**

采用加性KLM框架、Fitts法改进模型、确认操作时间常数、统计检验（配对Z检验、Cohen's d、TOST）以及Unity3D+Magic Leap 2实现；

**📊 数据集**

使用Magic Leap 2 HWD、手柄、眼动数据以及实验收集的交互时长数据；

**📈 对比分析**

将模型预测值与实际测量值进行配对比较，结果显示除ControllerBlink外5种模态预测误差均≤20%，Cohen's d小于0.5，TOST表明大多数模态在20%范围内等价，整体性能良好；

**⚠️ 局限性**

局限包括样本性别比例失衡、受试者多为已有沉浸技术经验、部分模态（ControllerBlink）因疲劳影响预测准确性、确认操作模型受设备差异影响。

---

## 571. Learning to Execute Graph Algorithms Exactly with Graph Neural Networks

**arXiv ID:** 2601.23207 | [PDF](https://arxiv.org/pdf/2601.23207v1)

**作者:** Muhammad Fetrat Qharabagh `[一作]` (University of Waterloo), Kimon Fountoulakis `[通讯]` (University of Waterloo)

**通讯引用:** 540 | [OpenAlex ID](https://openalex.org/A5050838896)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本工作通过构造局部多层感知机（MLP）集合并嵌入到图神经网络（GNN）中，证明了在有限度、有限精度约束下，GNN 能够精确学习分布式 LOCAL 模型中的任何算法，并给出了消息洪泛、广度优先搜索（BFS）、深度优先搜索（DFS）和 Bellman‑Ford 等经典图算法的可学习性定理；

**💡 创新点**

核心创新在于将神经切线核（NTK）理论与本地 MLP 集成、二进制块编码、阈值化与通信掩码相结合，首次对 LOCAL 算法给出 exact learnability 证明，并提供了针对每个算法的训练样本构造方法；

**🔧 技术方法**

技术手段包括 NTK 分析、局部 MLP 集成、二进制块编码（Ψ_Enc）、Heaviside 阈值化（Ψ_H）、通信掩码（P_C,P_M）以及图模板匹配框架，实现了可学习的 GNN 架构；

**📊 数据集**

训练数据为基于二进制指令的人工样本，实验以消息洪泛为主，使用所有 7 节点的树图（以及部分随机图）进行评估；

**📈 对比分析**

与先前的 feedforward 网络对比，展示了在图尺寸、指令数量和集成规模上的优势；实验表明随着集成大小增大，准确率提升至 100%，并验证理论所需集成规模约为 O(l²D⁴) 的上界，但实际所需集成数明显低于保守估计；

**⚠️ 局限性**

主要局限在于依赖无限宽 MLP/NTK 近似，导致所需集成规模巨大；仅适用于有界度、有限精度的图；对更大规模或包含负权边、非 LOCAL 结构的算法适用性有限。

---

## 572. Make Anything Match Your Target: Universal Adversarial Perturbations against Closed-Source MLLMs via Multi-Crop Routed Meta Optimization

**arXiv ID:** 2601.23179 | [PDF](https://arxiv.org/pdf/2601.23179v1)

**作者:** Hui Lu `[一作]` (Nanyang Technological University), Xudong Jiang `[通讯]` (Nanyang Technological University)

**通讯引用:** 15473 | [OpenAlex ID](https://openalex.org/A5085533260)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文针对闭源多模态大语言模型（如 GPT‑4o、Gemini‑2.0、Claude‑4.5）提出通用目标可迁移对抗攻击，旨在通过单一扰动样本统一诱导模型产生指定目标输出。

**💡 创新点**

创新点在于提出 MCRMO‑Attack 框架，结合多裁剪聚合与注意力引导裁剪（稳定目标监督）、对齐门控令牌路由（提升有效梯度）以及元初始化（加速且提升跨目标泛化）三大技术，克服了传统样本级攻击在通用性和可迁移性上的不足。

**🔧 技术方法**

技术实现使用多裁剪聚合（Multi‑Crop Aggregation）、注意力引导裁剪（Attention‑Guided Crop）、令牌路由（Token Routing）以及基于 Reptile 的元学习初始化，并利用 CLIP 等视觉编码器作为 surrogate；攻击过程中采用 ℓ∞ 约束、FGSM‑式投影更新。

**📊 数据集**

数据集为 MSCOCO 验证集 100 张目标图像，针对每个目标从 NIPS 2017 对抗攻击竞赛数据集抽取 20 张训练图和 30 张测试图，确保训练与评估互不重叠。

**📈 对比分析**

实验将 MCRMO‑Attack 与 AnyAttack、M‑Attack、FOA‑Attack、UAP、UnivIntruder 等基线在 GPT‑4o、Gemini‑2.0、Claude‑4.5 上进行比较，未见样本攻击成功率（ASR）提升至 GPT‑4o 61.7%（比 UAP 提升 23.7%）和 Gemini‑2.0 56.7%（提升 19.9%），并在多指标上均优于现有方法。

**⚠️ 局限性**

局限性包括：对极大裁剪数量或高维扰动空间时计算成本上升；在极其复杂或与目标语义差异较大的背景下，攻击成功率仍可能下降；仅在少量源样本下训练，需进一步提升对样本不平衡或极端场景的鲁棒性。

---

## 573. Monotonic Reference-Free Refinement for Autoformalization

**arXiv ID:** 2601.23166 | [PDF](https://arxiv.org/pdf/2601.23166v1)

**作者:** Lan Zhang `[一作]` (University of Manchester), André Freitas `[通讯]` (National Biomarker Centre CRUK Manchester Institute)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种无参考、迭代单调优化的全定理自动形式化框架，利用LLM生成器与定理证明器交互，联合优化正式有效性、逻辑保持、数学一致性与形式质量四维指标。

**💡 创新点**

首次定义无参考的全定理自动形式化任务，构造掩码多目标目标函数，设计响应度映射与单调接受策略，保证在评估噪声下实现单调改进与收敛，并在不同LLM角色之间实现互补。

**🔧 技术方法**

使用多角色LLM（一次生成 OOG、验证修复 FVR、递归改进 REG）、Lean4 定理证明器、LLM 评判器（GPT‑4.1‑mini / Qwen2.5‑Coder）以及响应度映射与下界置信度（LCB）决策机制。

**📊 数据集**

在 miniF2F 和 ProofNet 两大自然语言证明数据集上进行实验。

**📈 对比分析**

与单纯自我改进（ISR）以及多种LLM生成器在相同评判器下对比，实验显示在 miniF2F 上实现 93.44% 正式有效率、78.22% 综合得分；在 ProofNet 上实现 44.09% 有效率、29.79% 综合得分。

**⚠️ 局限性**

局限性包括：LLM 评判器不确定性的参数未经验估计；采用相同评判器做采纳与评价可能导致偏差；对可达性与计算资源有一定依赖。

---

## 574. Stochastic Linear Bandits with Parameter Noise

**arXiv ID:** 2601.23164 | [PDF](https://arxiv.org/pdf/2601.23164v1)

**作者:** Daniel Ezer `[一作]` (Tel Aviv University), Yishay Mansour `[通讯]` (Google Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

研究了在参数噪声模型下的随机线性赌博机，并给出了针对一般动作集及 ℓ_p 单位球的最优 regret 上界。

**💡 创新点**

提出了两种新算法（VASE 与 VALEE），利用 G‑optimal 设计、方差感知的加权最小二乘估计以及自适应探测-利用策略，实现了与下界匹配的方差依赖 regret 上界。

**🔧 技术方法**

主要技术包括 G‑optimal 设计、加权最小二乘估计、Median‑of‑Means 估计、随机化探测与利用分阶段策略，以及利用参数噪声模型的方差可控性。

**📊 数据集**

论文不涉及实际数据集，所有结果均为理论分析和上界/下界证明。

**📈 对比分析**

与已知的加性噪声模型及对抗模型相比，参数噪声模型在 ℓ_p 单位球（p≤2）上实现了从 d√T 降至 √(dT) 的显著改进，且在已知协方差时达到下界上限。

**⚠️ 局限性**

当协方差未知时仍需额外的 dT^{1/3} 惩罚项；对 ℓ_p 单位球（p>2）的最优算法尚未给出，且对极小方差情形的进一步改进仍是开放问题。

---

## 575. Agnostic Language Identification and Generation

**arXiv ID:** 2601.23258 | [PDF](https://arxiv.org/pdf/2601.23258v1)

**作者:** Mikael Møller Høgsgaard `[一作]` (Aarhus University and University of Oxford), Chirag Pabbaraju `[通讯]` (Stanford University)

**通讯引用:** 85 | [OpenAlex ID](https://openalex.org/A5081649943)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2`

**🎯 论文内容**

在没有可识别语言假设的“agnostic”环境下，研究语言识别和生成的统计学习理论，给出新的目标函数并推导其误差收敛速率；

**💡 创新点**

首次将可识别性假设完全放弃，提出对任意分布的通用目标，并证明在满足可达性条件时可实现指数级误差衰减，同时给出对应的下界；

**🔧 技术方法**

采用概率上界与下界技术、信息论方法、枚举与稀疏性分析以及基于“最小索引”判别的算法设计；

**📊 数据集**

无具体实验数据集，全部在理论模型（可数语言集合、任意概率分布）上进行推导；

**📈 对比分析**

通过对比指数衰减的上界与匹配的下界，证明在满足假设时能达到最优速率；在不满足条件时误差下降速度可无限慢；

**⚠️ 局限性**

提出的生成目标在一般情况下不可实现（下界接近1），且对有限语言集合之外的情况仅给出足够条件；尚未解决所有可数集合下的通用上界以及更合理的评估目标。

---

## 576. (Doubly) Exponential Lower Bounds for Follow the Regularized Leader in Potential Games

**arXiv ID:** 2601.23248 | [PDF](https://arxiv.org/pdf/2601.23248v1)

**作者:** Ioannis Anagnostides `[一作]` (Carnegie Mellon University), Tuomas Sandholm `[通讯]` (Carnegie Mellon University)

**通讯引用:** 20178 | [OpenAlex ID](https://openalex.org/A5023571961)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

做了什么：提出了对 Follow the Regularized Leader（FTRL）在势游戏中的收敛性上限与下限，并给出了 Fictitious Play 在多玩家势游戏中的双指数下界。

**💡 创新点**

创新点是什么：首次证明 FTRL 需要指数时间收敛，给出上界 exp(O(1/ε²)) 与下界 2^Ω(m log m)，并构造双指数下界证明 Fictitious Play 在 n 维势游戏中需 (C·2ⁿ)! 步骤。

**🔧 技术方法**

用了什么技术：运用势函数平滑性、稀疏化概率分析、周期与螺旋构造、蛇形路径编码理论以及递推不等式证明。

**📊 数据集**

用了什么数据集：无，全部为理论构造与证明。

**📈 对比分析**

如何比较的方法，性能怎么样：与已有指数收敛下界对比，展示更紧密的上界与下界，表明 FTRL 在势游戏中可能是最差的梯度下降型优化器。

**⚠️ 局限性**

limitation是什么：结果主要针对势游戏与理想化的 FTRL，实际算法在多玩家混合策略情形下是否满足双指数下界尚未解决。

---

## 577. Applications of QR-based Vector-Valued Rational Approximation

**arXiv ID:** 2601.23237 | [PDF](https://arxiv.org/pdf/2601.23237v1)

**作者:** Simon Dirckx `[一作]` `[通讯]` (Oden institute, University of Texas), Simon Dirckx (Oden institute, University of Texas)

**关键词:** `e4c502e8-c16d-4c56-8df3-cffaee9eaadb` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出了一种新的QR-AAA算法，用于向量值有理逼近，通过贪婪优化实现。

**💡 创新点**

QR-AAA算法在效率上显著优于传统的SV-AAA方法，同时保持了准确性和鲁棒性。它简化了理论推导和误差估计，并且不依赖于额外的结构假设或随机化草图。

**🔧 技术方法**

使用了QR分解和贪婪自适应有理逼近（AAA）技术。

**📊 数据集**

应用了多个数据集，包括用于求解Stokes方程的边界积分方程、预选函数类的积分节点、以及多变量有理逼近的样本数据。

**📈 对比分析**

与SV-AAA方法相比，QR-AAA在多个应用中表现出显著的加速效果，尤其是在处理大规模计算时，QR-AAA的计算时间显著低于SV-AAA，且在精度上没有明显损失。

**⚠️ 局限性**

QR-AAA在某些情况下可能无法达到最佳性能，特别是在处理具有复杂结构的函数时，可能需要额外的步骤来增强逼近效果。

---

## 578. Scaling Multiagent Systems with Process Rewards

**arXiv ID:** 2601.23228 | [PDF](https://arxiv.org/pdf/2601.23228v1)

**作者:** Ed Li `[一作]`, Cat Yan `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并训练了多智能体系统，使用过程性AI反馈奖励实现端到端强化学习。

**💡 创新点**

通过使用基于代理的coach对每一步动作给出密集奖励，实现精细的信用分配与样本效率提升。

**🔧 技术方法**

采用REINFORCE++与全局归一化的PPO策略，配合Ray、vLLM和DeepSpeed实现分布式训练，coach采用Gemini 2.5等LLM。

**📊 数据集**

在数学推理（AIME、AMC）和数据科学管道（DSBench）两个数据集上进行实验。

**📈 对比分析**

相较于基准，MAPPA在AIME提升5–17.5个百分点，AMC提升7.8–17.2个百分点，DSBench成功率提升12.5个百分点，质量指标提升至30%。

**⚠️ 局限性**

受限于coach偏见、训练样本有限以及对大规模系统的reward hacking风险，且只提供标量奖励，缺乏可解释的奖励策略。

---

## 579. MonoScale: Scaling Multi-Agent System with Monotonic Improvement

**arXiv ID:** 2601.23219 | [PDF](https://arxiv.org/pdf/2601.23219v1)

**作者:** Shuai Shao `[一作]` (Shanghai Jiao Tong University), Weinan Zhang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 18087 | [OpenAlex ID](https://openalex.org/A5090720315)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种 MonoScale 框架，能够在多代理系统中按阶段有序引入新代理，并通过任务定制与记忆更新来防止冷启动导致的性能崩溃。

**💡 创新点**

创新点在于将新代理的熟悉化任务与可审计、可回滚的自然语言记忆结合，形成基于上下文赌博机的保守信赖区间更新，从而保证每一次扩展都能实现单调不下降的性能。

**🔧 技术方法**

采用的核心技术包括：任务合成（planner‑executor‑validator）、可编辑的记忆模块（文本缓冲区）、基于 TRPO 的记忆更新约束，以及保守的策略提升与回退机制。

**📊 数据集**

实验使用 GAIA 基准和 Humanity's Last Exam（HLE）多项选择子集，评估在不同规模（3–10 个代理）下的表现。

**📈 对比分析**

与无记忆的“naive scale‑up”以及使用 GPT‑5、Gemini‑3‑Pro 等强大模型的固定池基线对比，MonoScale 在 GAIA 上从 44.84% 提升到 55.15%，在 HLE 上从 11.60% 提升到 19.90%，并在包含噪声代理的环境中保持稳定，避免了性能崩溃。

**⚠️ 局限性**

局限性包括：仅在最多 10 个代理的小规模实验，未探索百万级代理目录的检索式路由；记忆更新仍可能被恶意代理或敏感数据污染；对安全性、隐私等方面的完整保障尚未实现。

---

## 580. A complete characterisation of conditional entropies

**arXiv ID:** 2601.23213 | [PDF](https://arxiv.org/pdf/2601.23213v1)

**作者:** Roberto Rubboli `[一作]` (University of Copenhagen), Marco Tomamichel `[通讯]` (National University of Singapore)

**通讯引用:** 7539 | [OpenAlex ID](https://openalex.org/A5040644995)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文对条件熵进行了完整的公理化刻画，给出了满足可加、重标记不变、在条件混合通道下单调以及归一化等四条自然公理的最一般形式，并证明该形式可表示为 Rényi 熵的指数平均（参数为实数 t 和正实数上的概率测度 τ）。

**💡 创新点**

创新点包括：
• 通过预序半环与大样本定理构建条件主导理论，首次将条件熵的极端形式 H_{t,τ} 与其可加、单调等公理完全对应；
• 推导出足够且在某些条件下必要的 t 与 τ 的取值范围；
• 将条件熵与条件主导率、量子热力学第二定律等实际问题直接关联，提供了一组新的“第二定律”。

**🔧 技术方法**

核心技术：
• 预序半环理论与大样本比较定理（Vergleichsstellensatz）；
• Rényi 熵与指数期望的耦合与变换；
• 对 t 与 τ 的凸性/凹性分析（Minkowski、对数凸/凹性）；
• 极限与渐近技术（t→±∞、t→0 等），构造必要条件的反例。

**📊 数据集**

由于工作完全是理论推导，未使用具体数据集；所有结论基于概率分布、矩阵变换和数学分析。

**📈 对比分析**

本文没有与实验方法直接比较；其性能指标为“可行性”和“完整性”，即对所有满足公理的条件熵给出唯一形式，并证明其在主导转换率与热力学第二定律中的适用性。

**⚠️ 局限性**

局限性：
• 对 τ 的必要性条件只在满足“在 1 附近衰减快于 1/(1-α)”的测度下得到；
• 完全量子情形（无对角化假设）仍为开放问题；
• 对 t 的取值区间仍有未解的极限情况（如 t→∞ 的细节）。

---

## 581. User Prompting Strategies and Prompt Enhancement Methods for Open-Set Object Detection in XR Environments

**arXiv ID:** 2601.23281 | [PDF](https://arxiv.org/pdf/2601.23281v1)

**作者:** Junfeng Lin `[一作]` (Duke University), Maria Gorlatova `[通讯]` (Duke University)

**通讯引用:** 2169 | [OpenAlex ID](https://openalex.org/A5036726336)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

探讨XR场景下开放式目标检测模型对不同自然语言提示（含含糊、过度详细、普通等）的鲁棒性，并验证提示增强策略的有效性。

**💡 创新点**

首次系统评估真实用户提示多样性对OSOD性能的影响，提出两种基于视觉语言模型的提示增强方法（关键对象提取与语义类别归属）显著提升检测鲁棒性。

**🔧 技术方法**

使用视觉语言模型（GPT‑5、BLIP等）生成和优化提示；采用GroundingDINO与YOLO‑E两种主流OSOD模型进行实验；评估指标为mIoU和置信度。

**📊 数据集**

在自制的DiverseAR与DiverseAR+增强现实图像数据集（264张），每张图像标注目标框并生成多种提示。

**📈 对比分析**

对比原始提示与增强后提示的性能：在含糊提示下，语义类别归属可将mIoU提升55%以上、置信度提升41%；在过度详细提示下，GroundingDINO可提升约20% mIoU，YOLO‑E在原始提示下表现最佳。

**⚠️ 局限性**

实验样本规模有限，提示多样性未覆盖所有文化与隐喻表达；仅在单帧图像上评估，未验证实时视频和多用户交互场景的鲁棒性。

---

## 582. IRL-DAL: Safe and Adaptive Trajectory Planning for Autonomous Driving via Energy-Guided Diffusion Models

**arXiv ID:** 2601.23266 | [PDF](https://arxiv.org/pdf/2601.23266v1)

**作者:** Seyed Ahmad Hosseini Miangoleh `[一作]` (Amirkabir University of Technology), Farzaneh Abdollahi `[通讯]` (Amirkabir University of Technology)

**通讯引用:** 2944 | [OpenAlex ID](https://openalex.org/A5042263160)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93`

**🎯 论文内容**

提出了一种将逆强化学习、条件扩散模型和自适应注意力掩码相结合的安全自适应轨迹规划框架IRL‑DAL，用于无人驾驶车辆的安全路径生成与控制。

**💡 创新点**

创新点在于：①端到端的IL‑IRL‑RL协同训练；②仅在高风险状态激活的能量引导扩散规划器做安全补偿；③轻量可学习自适应掩码（LAM）根据速度与雷达信息动态调整视觉注意；④经验修正机制（SAEC）将安全动作标记为专家行为，实现安全经验的持续利用。

**🔧 技术方法**

使用了逆强化学习（GAIL）、行为克隆、PPO、条件扩散模型、能量引导采样、自适应掩码模块、FSM结构化回放、经验纠正策略，以及Webots仿真环境。

**📊 数据集**

数据集为由基于有限状态机（FSM）专家策略在Webots中生成的平衡采样驾驶经验，覆盖四种驾驶模式，并在仿真中加入多车道曲线路况、移动障碍物与光照变化。

**📈 对比分析**

通过与基线PPO+均匀采样、加FSM回放、加扩散规划器、加LAM+SAEC四种组合进行对比，最终模型在50k步训练后取得96%成功率，碰撞率降至0.05/1k步，平均奖励提升至180.7，显著优于所有对照方法。

**⚠️ 局限性**

局限性包括：①仍基于仿真，缺乏真实道路验证；②扩散规划器仅在高风险时激活，计算量相对较大；③LAM仅依据速度与雷达距离调节，可能不足以处理更复杂多模态情景；④整体训练流程涉及多阶段调参，复杂度较高。

---

## 583. Particle-Guided Diffusion Models for Partial Differential Equations

**arXiv ID:** 2601.23262 | [PDF](https://arxiv.org/pdf/2601.23262v1)

**作者:** Andrew Millard `[一作]` (Linköping University), Zheng Zhao `[通讯]` (Linköping University)

**通讯引用:** 21602 | [OpenAlex ID](https://openalex.org/A5065843472)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了一种在扩散模型中加入PDE残差引导并嵌入顺序蒙特卡洛框架的生成式PDE求解方法。

**💡 创新点**

创新点在于将物理约束转化为显式似然并结合第二阶随机引导（SOSaG）与伪启动（pBS）实现高效、可调节的条件采样，同时揭示传统SOTA方法在高维下有效的背后是多次尝试的启发式特性。

**🔧 技术方法**

使用扩散模型（EDM）、梯度引导（score-based）、Euler–Maruyama、SOSaG、SMC、pBS及PINN思想等技术。

**📊 数据集**

在五个经典单物理PDE（Darcy流、Helmholtz、Navier–Stokes、Poisson）和两组多物理交互反应扩散（2SRD、3SRD）上进行实验，数据来源为公开的训练集与对应的FEM真值。

**📈 对比分析**

与DiffPDE（原始指导方法）和DiffPDE‑NoG（无引导）对比，SMC + SOSaG在解域和系数域的相对误差均低于基线，尤其在含观测噪声的多物理场景下仍保持优异表现。

**⚠️ 局限性**

局限性包括：计算开销显著增加（尤其样本数增大时）、有效样本数（ESS）在后期退化、GPU内存受限导致只能用有限粒子、PDE残差评估成本高，需要进一步加速或引入预训练PINN作为代理。

---

## 584. End-to-end Optimization of Belief and Policy Learning in Shared Autonomy Paradigms

**arXiv ID:** 2601.23285 | [PDF](https://arxiv.org/pdf/2601.23285v1)

**作者:** MH Farhadi `[一作]` (University of Rhode Island), Reza Abiri `[通讯]` (University of Rhode Island)

**通讯引用:** 1318 | [OpenAlex ID](https://openalex.org/A5064238465)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并实现了 BRACE 框架，实现了共享自主系统中用户意图贝叶斯推断与辅助策略的端到端联合优化；

**💡 创新点**

创新点在于：①使用完整的目标后验分布而非 MAP 估计来调节辅助力度；②通过梯度回传实现推断模块与控制模块的双向耦合，消除传统顺序管线的估计器-控制器不匹配；③给出了辅助策略对目标不确定性与环境约束的单调性与二阶 regret 上限的理论证明；

**🔧 技术方法**

技术包括递归贝叶斯过滤、Actor‑Critic 强化学习（PPO）与上下文编码网络、连续混合参数 γ 的可微控制、以及基于梯度的联合训练与课程学习；

**📊 数据集**

使用 1968 条已标记的 2D 光标轨迹数据预训练贝叶斯模块，并在仿真环境（2D cursor、Reacher‑2D、Fetch‑PickAndPlace）中进行评估；

**📈 对比分析**

与现有方法（IDA、DQN、手动调参等）进行对比，实验表明 BRACE 在 2D 光标任务中成功率提升 36.3%、路径效率提升 60.9%；在 Reacher‑2D 中比 IDA 高 17.9% 的目标完成率、比 DQN 高 4.8 目标/分钟；在 Fetch‑PickAndPlace 中成功率 86%（比 DQN 74%、IDA 68%）、完成时间 9.8 s（比 DQN 12.3 s、IDA 14.7 s）、碰撞次数 0.22（比 DQN 0.41、IDA 0.58）；

**⚠️ 局限性**

局限性包括：依赖高质量专家策略；未对专家进行在线自适应；主要基于运动学输入，缺乏多模态感知；训练过程需要课程学习与精细调参，扩展到更复杂任务仍需研究。

---

## 585. FOCUS: DLLMs Know How to Tame Their Compute Bound

**arXiv ID:** 2601.23278 | [PDF](https://arxiv.org/pdf/2601.23278v1)

**作者:** Kaihua Liang `[一作]` (King Abdullah University of Science and Technology), Marco Canini `[通讯]` (King Abdullah University of Science and Technology)

**通讯引用:** 4595 | [OpenAlex ID](https://openalex.org/A5042255975)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 FOCUS 系统，通过动态剔除不可解码的 token，显著提升 DLLM 推理吞吐量。

**💡 创新点**

创新点在于利用早期层注意力差值作为解码概率预测，训练无关地实现 token 淘汰。

**🔧 技术方法**

使用了块扩散范式、KV 缓存重用、动态预算与 Token Eviction 技术，并结合重要性差分指标。

**📊 数据集**

在 GSM8K、Math500、HumanEval、MBPP、IFEval、ShareGPT、WildChat、MATH 等数据集上评测。

**📈 对比分析**

相较于 LMDeploy 等基线，FOCUS 在 B=32 时可提升约 2.32×吞吐量，B=64 时可达 3.52×，且保持或提升生成质量。

**⚠️ 局限性**

局限性包括对超大块和复杂推理场景下的 KV 缓存一致性敏感，以及需手动设置阈值与扩展因子。

---

## 586. TEON: Tensorized Orthonormalization Beyond Layer-Wise Muon for Large Language Model Pre-Training

**arXiv ID:** 2601.23261 | [PDF](https://arxiv.org/pdf/2601.23261v1)

**作者:** Ruijie Zhang `[一作]` (University of California Santa Barbara), Zheng Zhang `[通讯]` (University of California Santa Barbara)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为Teon的优化器，将原来按层正交化的MuOn方法扩展到张量级别，利用跨层梯度信息提升训练效率。

**💡 创新点**

创新点在于将多层梯度堆叠为高阶张量，在张量模式展开后进行正交化，从而获得更强的收敛保证和理论上可达√K倍的性能提升。

**🔧 技术方法**

主要技术包括张量正交化（基于矩阵化的正交化）、Newton–Schulz迭代近似SVD、PolarExpress等高效SVD方法，以及对Transformer梯度进行跨层堆叠。

**📊 数据集**

实验数据集为FineWeb文本数据，分别在10 B token的GPT-2规模（130 M–774 M参数）和1.1–13.1 B token的LLaMA规模（60 M–1 B参数）上进行预训练。

**📈 对比分析**

与AdamW和MuOn比较，Teon在GPT-2和LLaMA两大模型上在验证perplexity上持续优于MuOn，早期收敛更快、最终性能提升约1–2点，且在不同SVD近似方法下表现鲁棒。

**⚠️ 局限性**

局限性包括需手动选择堆叠层数K和正交化模式、对不同层类型的适配性差异、K增大时正交化精度下降，以及实现相对复杂。

---

## 587. VideoGPA: Distilling Geometry Priors for 3D-Consistent Video Generation

**arXiv ID:** 2601.23286 | [PDF](https://arxiv.org/pdf/2601.23286v1)

**作者:** Hongyang Du `[一作]` (Brown University), Yue Wang `[通讯]` (University of Southern California)

**通讯引用:** 54020 | [OpenAlex ID](https://openalex.org/A5113600509)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

针对视频扩散模型缺乏三维一致性的问题，提出 VideoGPA，通过几何基础模型自动生成重投影一致性评分，构造几何偏好对，使用 Direct Preference Optimization (DPO) 对预训练模型进行轻量级后训练，从而显著提升视频的 3D 结构一致性和时间稳定性。

**💡 创新点**

创新点包括：
1) 用几何基础模型提供的密集重投影一致性得分，作为无人工标签的自监督几何偏好信号；
2) 将该几何信号嵌入 DPO 框架，对扩散模型进行后训练对齐；
3) 仅需约 2,500 条偏好对和 1% 参数 LoRA 微调，即可实现显著性能提升，避免了显式几何约束或大规模数据需求。

**🔧 技术方法**

核心技术：
- 视频扩散模型（CogVideoX、DiT）采用 v‑prediction 参数化；
- 几何基础模型（如 DUSt3R、MASt3R）对帧进行深度与相机姿态估计；
- 重投影一致性损失（MSE+LPIPS）作为 3D 一致性评分；
- Direct Preference Optimization (DPO) 与 LoRA 微调整合，形成后训练对齐流程。

**📊 数据集**

数据集与提示：
- 训练与评估使用 DL3DV-10K（图像提示和文本提示）；
- I2V 使用 DL3DV-10K 子集的初始帧并配合结构化摄像机运动原语；
- T2V 使用 CogVLM2-Video 生成的字幕；
- 与 GeoVideo 对比的基准数据为约 10k DL3DV-10K 视频。

**📈 对比分析**

与 Base、SFT、Epipolar‑DPO、GeoVideo 等基线比较：
- I2V 结果：SSIM 提升 0.055、LPIPS 减少 0.045；MVCS 从 0.976 提升至 0.986、3DCS 从 0.687 降至 0.638；VideoReward 总体胜率 76.0%（高于 66.0% 的 Epipolar‑DPO）。
- T2V 结果：SSIM 0.621、LPIPS 0.495、MVCS 0.974、3DCS 0.519，VideoReward 总体胜率 60.33%；与 GeoVideo 对比，VideoGPA 在几何一致性上更优而视觉质量保持更好。
- 人类偏好实验（I2V）显示 VideoGPA 获得 53.5% 的最高选胜率。

**⚠️ 局限性**

局限性：
- 计算成本随帧数线性增长，重投影一致性评估在长视频上的效率和显存需求较高；
- 依赖几何基础模型，若基础模型性能下降或在复杂动态场景中失效，可能影响偏好信号的可靠性；
- 目前主要针对静态或摄像机运动场景，完全动态、遮挡复杂的场景效果待进一步验证。

---

## 588. Decoupled Diffusion Sampling for Inverse Problems on Function Spaces

**arXiv ID:** 2601.23280 | [PDF](https://arxiv.org/pdf/2601.23280v1)

**作者:** Thomas Y. L. Lin `[一作]`, Anima Anandkumar `[通讯]` (California Institute of Technology)

**通讯引用:** 16927 | [OpenAlex ID](https://openalex.org/A5014498545)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了 Decoupled Diffusion Inverse Solver (DDIS)，一种在函数空间上将系数先验建模与物理前向算子分离、并通过 DAPS 后验采样的生成式框架，用于在稀疏观测下解决逆 PDE 问题。

**💡 创新点**

创新点包括：①解耦先验与物理，使得联合嵌入模型的引导衰减和稀疏观测下的指导失效问题被消除；②利用神经算子显式建模前向 PDE，提高数据效率并实现高频细节的恢复；③在后验采样中采用 DAPS，避免传统 DPS 的 Jensen 间隙导致的过平滑；④给出理论分析证明解耦设计在数据稀缺和稀疏观测场景下的优势。

**🔧 技术方法**

使用技术：score‑based diffusion prior、神经算子（如 FNO、DeepONet 等）、Decoupled Annealing Posterior Sampling (DAPS)、Langevin 动力学、物理约束损失（如 PINO）以及可选的 PDE 残差正则化。

**📊 数据集**

数据集：合成逆 Poisson、Helmholtz、Navier‑Stokes 三个 PDE 任务；观测采样点约 500 个（约 3% 空间）；训练分为全量 128² 对齐样本、5%/1% 对齐样本以及 64² 低分辨率/混合分辨率样本，均采用相同的网格与噪声设定。

**📈 对比分析**

与 DiffusionPDE、FunDPS、ECI‑sampling、OFM 等前沿方法对比：在标准监督下，DDIS 在相同时间预算下 ℓ₂ 误差和谱误差均优于对手；在 1% paired 数据稀缺场景下，DDIS 仍保持低 ℓ₂ 误差，优势约 40%；在低/多分辨率训练下，DDIS 的性能衰减最小，显示出强大的分辨率不变性。

**⚠️ 局限性**

局限性：①物理损失在 Navier‑Stokes 等复杂 PDE 上难以应用；②仍需较多对齐样本来训练高质量神经算子；③在极高维度或更复杂的真实世界数据集上，算子训练与采样成本仍较高；④目前验证主要集中在合成数据，缺乏大规模真实场景的实验；⑤对观测噪声水平变化的鲁棒性尚未系统评估。

---

## 589. PaperBanana: Automating Academic Illustration for AI Scientists

**arXiv ID:** 2601.23265 | [PDF](https://arxiv.org/pdf/2601.23265v1)

**作者:** Dawei Zhu `[一作]` (Peking University), Jinsung Yoon `[通讯]` (Google Cloud AI Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于代理的框架，能够自动生成符合学术规范的研究方法图和统计图。

**💡 创新点**

创新点在于结合检索驱动的参考样本、自动化的风格提炼和视觉与批评代理的多轮迭代，显著提升了图像的忠实度、简洁度、可读性和美观度。

**🔧 技术方法**

主要技术包括多模态大型语言模型（Gemini‑3‑Pro）、文本转图像模型（Nano‑Banana‑Pro、GPT‑Image‑1.5）、可执行代码生成、检索与计划代理，以及 VLM‑as‑Judge 评估。

**📊 数据集**

使用了从 NeurIPS 2025 论文中提取的 292 条测试案例及其 292 条参考图形构建的“Methodology Diagram Generation Benchmark”，以及 240 条统计图的 ChartMimic 数据集。

**📈 对比分析**

与 vanilla、few‑shot、Paper2Any 等基线相比，本文方法在四个维度上均获得提升，整体分数提升 17 %（Faithfulness +2.8%、Conciseness +37.2%、Readability +12.9%、Aesthetics +6.6%），人类评估亦显示 72.7 % 的胜率。

**⚠️ 局限性**

主要局限包括生成的图像为光栅而非矢量，细粒度连线错误导致忠实度仍低于人工绘制，且对多样化风格与极复杂结构的处理仍有挑战。

---

## 590. UPA: Unsupervised Prompt Agent via Tree-Based Search and Selection

**arXiv ID:** 2601.23273 | [PDF](https://arxiv.org/pdf/2601.23273v1)

**作者:** Siran Peng `[一作]`, Zhen Lei `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种完全无监督的 Prompt Agent（UPA），通过树结构搜索和细粒度无偏对比，在没有任何监督奖励信号的情况下自动生成高质量 Prompt。

**💡 创新点**

创新点在于：①将无监督的树基搜索与大语言模型（LLM）的双盲五分 Likert 对比相结合；②采用 Bradley‑Terry‑Luce 模型的两阶段（路径贝叶斯过滤 + 全局 BTL 最大似然）选择框架，有效消除路径依赖与噪声；③在搜索过程中加入多样性惩罚和 UCB 调整，以避免语义冗余。

**🔧 技术方法**

核心技术包括：改进版 MCTS、双盲 Likert 5 分评分、Beta‑贝塔推断、BTL 最大似然与 MM 算法、LLM 作为执行、评判和优化工具。

**📊 数据集**

实验数据集涵盖闭合式任务：GPQA、AGIEval‑MATH、LIAR、WSC、BBH‑Navigate；开放式任务：MT‑Bench（Writing、Roleplay、Humanities）；并在 GPT‑5、Claude‑4.5‑Sonnet、DeepSeek‑V3.2 等前沿执行 LLM 上进行跨模型验证。

**📈 对比分析**

与 IO、CoT、RaR、Step‑Back、PromptAgent、PromptBreeder、TextGrad、SPO 等方法在同一任务上进行对比，UPA 在闭合式任务中平均准确率比最强对手高约 2.7%，在 MT‑Bench 上对 IO 和 SPO 的胜率均超过 50%，在多种执行模型上均保持领先。

**⚠️ 局限性**

局限性：①对 LLM 的执行、评判和优化模型高度依赖，算力与成本较高；②对比样本稀疏导致路径估计误差，可能影响深层节点的可靠性；③实验主要聚焦现有基准，未充分验证在更大规模、不同目标或多任务场景下的适用性。

---

## 591. TCBench: A Benchmark for Tropical Cyclone Track and Intensity Forecasting at the Global Scale

**arXiv ID:** 2601.23268 | [PDF](https://arxiv.org/pdf/2601.23268v1)

**作者:** Milton Gomez `[一作]` (University of Lausanne), Tom Beucler `[通讯]` (University of Lausanne)

**通讯引用:** 26738 | [OpenAlex ID](https://openalex.org/A5061588829)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并公开了全球范围内热带气旋轨迹与强度预测的统一基准 TC Bench，并提供评估工具、可视化以及多种物理与神经网络模型的基线结果。

**💡 创新点**

创新点在于：①将观测（IBTrACS）、重分析（ERA5）与多模型（TIGGE、FourCastNet、Pangu‑Weather、GenCast 等）输出统一格式；②提出完整的评估指标体系（包括 DPE、CTE、ATE、CRPS、CSI 等）并将 RI 作为二分类问题；③提供可复现的后处理流水线提升神经网络的强度预测能力。

**🔧 技术方法**

技术手段包括：图神经网络、傅里叶神经算子、视觉 Transformer（FourCastNet 等）以及基于 TempestExtremes 的自动轨迹提取；后处理采用统计-动力学方法（类似 SHIPS）与分布式推断。

**📊 数据集**

数据集：IBTrACS（轨迹与强度）、ERA5 重分析、TIGGE 物理模型后向预测、FourCastNet、Pangu‑Weather、GenCast、AIFS 等神经模型输出；所有数据统一转换为 IBTrACS 标准格式。

**📈 对比分析**

比较方法：使用确定性误差（RMSE、MAE、DPE、CTE、ATE）与概率误差（CRPS）评估轨迹与强度；RI 采用 TP/FP/FN/TN 计算 CSI。实验结果显示：某些 AI 模型在 24~48 小时内轨迹误差低于物理模型且优于 persistence，后处理后能将强度预测提升至与 GEFS 相近；但整体概率轨迹精度仍落后于 GEFS；RI 预测仅后处理后的 Pangu‑Weather 具备一定的成功率。

**⚠️ 局限性**

局限性：①数据源不完整（缺少部分气象变量与更高分辨率场）；②对不同机构观测定义的强度差异与轨迹起始点的不一致；③RI 预测仍依赖于后处理且仅在少数模型上表现；④基准仅涵盖 2023 年，缺少跨季节长周期评估；⑤评估主要关注 1–5 天，长周期预测未覆盖。

---

