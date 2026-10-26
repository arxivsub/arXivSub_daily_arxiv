# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-10-26 | 今日论文总数: 605

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. Experimental Classification of xApp Conflicts in O-RAN: Slicing vs. Per-UE Scheduling

**arXiv ID:** 2610.32254 | [PDF](https://arxiv.org/pdf/2610.32254v1)

**作者:** Rashmi Sharan Sinha `[一作]` (University College Dublin), Nima Afraz `[通讯]` (University College Dublin)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在 O‑RAN 近实时 RIC 中实现并实验验证了切片层 PRB 分配与单 UE 调度之间的跨粒度冲突检测与控制，主要通过一个动态谱效率 xApp（在基准切片分配基础上回收闲置资源并分配给饱和切片）与一个 UE 级调度 xApp（为单 UE 固定 40% PRB 保证）共同作用的场景。

**💡 创新点**

创新点在于：① 在真实硬件测试床上首次系统性地探测并分类切片级与 UE 级 xApp 的直接、间接与隐式冲突；② 采用核心侧流量计数作为切片负载代理，克服 gNB PRB 计数器缺乏切片标签的缺陷；③ 通过预算守护的谱效率算法实现资源池化与回收，保证每个控制周期 PRB 预算不超出 100%。

**🔧 技术方法**

技术实现包括：Open RAN 生态（OCUDU+USRP X310 5G gNB、Open5GS 核心、Near‑RT RIC）、Python xApp 开发、命令总线（Command Bus）日志、核心侧 per‑slice 流量测量、预算守护算法与整数量化、冲突检测器对日志的规则匹配。

**📊 数据集**

实验数据集由 5 台商用手机（COTS UE）在三路切片（SST 1、2、3）下收集的 276 个控制周期（约 17.1 分钟）以及 36 周期（134 秒）短窗口的控制日志与流量样本组成，包含 147 次直接冲突、49 次间接冲突和 11 次隐式冲突。

**📈 对比分析**

通过在同一流量轨迹上重放静态 34/33/33% 切片分配与动态谱效率 xApp 的对比，动态方案将未满足需求降低 12–33%，闲置分配降低 21–23%，但由于资源池饱和，饱和切片的 PRB 额度提升仅约 5% 并未显著提升服务吞吐；实验中每次 PRB 目标变更均在下一周期实现了 5–10% 的使用变化。

**⚠️ 局限性**

局限性包括：① UE 级调度 xApp 的 PRB 保证受 UE 标识符漂移影响，导致部分推送被拒绝；② 预算守护算法仅能提供 5% 的增补，难以在高负载下显著提升饱和切片吞吐；③ 本研究仅统计冲突出现次数，未测量实际 UE 质量服务（吞吐、时延）影响；④ 仅在单小区、有限 UE 数量的测试床上验证，缺乏大规模部署与多域协同控制的实验。

---

## 2. TRACE: Data-Free Text Reconstruction Attacks against Approximate Unlearning in LLMs

**arXiv ID:** 2610.31995 | [PDF](https://arxiv.org/pdf/2610.31995v1)

**作者:** Mengying Zhang `[一作]` (RMIT University), Minhui Xue `[通讯]` (CSIRO)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文针对大型语言模型（LLM）的近似遗忘技术，提出一种攻击框架TRACE，利用模型快照间的参数差异恢复被遗忘的数据，展示了近似遗忘在实际部署中可能泄露敏感信息的风险。

**💡 创新点**

创新点包括：1) 通过嵌入层参数差异筛选出潜在遗忘词汇，显著压缩搜索空间；2) 设计对比流畅度得分（CFS），同时兼顾流畅性与遗忘显著性；3) 采用两阶段“探索-完成”搜索策略，以实现多样化且精确的文本重建；4) 在无任何数据先验知识的条件下完成完整文本恢复。

**🔧 技术方法**

技术方法主要包括：参数差异分析、候选词集构建、对比流畅度评分（CFS）、基于困惑度的对比搜索、聚类判定多样性、两阶段搜索框架、黑盒与白盒实验、压缩与稀疏化的鲁棒性测试。

**📊 数据集**

使用的公开数据集有：TOFU、MUSE、WMDP；模型涵盖LLaMA2-13B/7B、Mistral-7B、Qwen2.5-3B、StableLM-3B-4E1T、TinyLlama-1.1B；对比六种常见的近似遗忘方法（GA、GD、KL、NPO、NPO-Diff、NPO-KL）。

**📈 对比分析**

在所有模型与遗忘方法上，TRACE均优于仅基于困惑度的基线（PPL-only）和前沿的前缀辅助方法（UBNF）。在LLaMA2-7B、Mistral-7B、LLaMA2-13B上，ROUGE‑L召回率超过0.93；在大批量遗忘（B=128）和长文本恢复（512词）场景下，性能仍保持在0.8以上；在黑盒和模型压缩（4‑bit AWQ、10% Wanda pruning）设置下，性能略降但依旧可用。

**⚠️ 局限性**

局限性包括：1) 对大规模词表的黑盒搜索仍受限，需更高计算成本；2) 高级隐私防御（如梯度裁剪、参数混淆）虽能削弱攻击，但对遗忘效果亦有负面影响；3) 只针对近似遗忘，完全重新训练的精确遗忘在大模型上仍较难实现；4) 实验主要集中于公开数据集，真实业务场景下的多样性与复杂性仍待验证。

---

## 3. Efficient Zero-Shot Video Anomaly Detection via Signal-Aware Adaptive Fusion (SAAF)

**arXiv ID:** 2610.32486 | [PDF](https://arxiv.org/pdf/2610.32486v1)

**作者:** Khaled Waleed Dawoud `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Muhammad Haris Khan `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种零训练、零样本的视频异常检测框架SAAF，利用视觉‑语言模型高效识别异常

**💡 创新点**

三大创新：双信号自适应采样（delta+drift）、阈值门控级联结构、信号感知加权融合

**🔧 技术方法**

采用CLIP视觉编码、VLM（Qwen3‑VL、VideoLLaMA3）与LLM（Qwen2.5）配合EMA阈值动态门控

**📊 数据集**

在四大基准数据集：XD‑Violence、UCF‑Crime、MSAD、Shang 进行评测

**📈 对比分析**

与现有零样本与有监督方法对比，SAAF在XD 92.07%、MSAD 86.70%、Shang 80.54% AUC，速度提升约2.2–2.6×

**⚠️ 局限性**

局限性包括对极短时或高帧率异常的敏感性不足，仍需针对不同场景调参，且对大型VLM的算力依赖较高

---

## 4. Playing Bias: Epistemic Games for Interpreting Text-to-Image Models

**arXiv ID:** 2610.32159 | [PDF](https://arxiv.org/pdf/2610.32159v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 5. The Path to NDE 4.0: A Review of Algorithms, Techniques, and Strategies for Autonomous Inspection

**arXiv ID:** 2610.31704 | [PDF](https://arxiv.org/pdf/2610.31704v1)

**作者:** Ashiqur Rahmana `[一作]`, Hamed Alhoori `[通讯]` (Northern Illinois University)

**通讯引用:** 638 | [OpenAlex ID](https://openalex.org/A5065954205)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b` `0d7d4da1-2b80-44f1-afe6-3f60783c9de2` `70e40602-aae3-44bd-80ec-4a7f2674330f` `a6cb313d-240c-4723-a372-3ba1f39b9afc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `109c2b71-d051-425c-831f-0c544c24280d` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文对非破坏性检测（NDI）领域进行系统综述，围绕算法、技术、应用和研究类型四个维度进行归类与比较，并提出从传统手工检验向NDE 4.0数字化智能化转型的研究路线图。

**💡 创新点**

创新点在于构建了多维度分类框架，综合评估了从计算机视觉到深度学习、生成模型等算法在不同检测技术中的表现，并系统识别了数据稀缺、黑盒AI、标准化和数字孪生集成等关键瓶颈。

**🔧 技术方法**

采用文献综述与元分析方法，使用表格与图示对比不同算法（CV、ML、DL、异常检测、生成模型）与技术（超声、X‑ray、红外、视觉、电磁场）在多行业中的应用与性能。

**📊 数据集**

本研究未收集新实验数据，而是利用已公开的研究报告与论文所引用的数据集，强调目前缺乏统一、开放的NDI数据资源。

**📈 对比分析**

通过对比分析已发表的实验结果，指出深度学习方法普遍优于传统算法，但受限于标注数据不足、模型可解释性差和跨行业通用性低；文章对各技术性能做了概括性评价。

**⚠️ 局限性**

主要局限在于：①缺乏大规模、标准化的公开NDI数据集；②深度模型的黑盒特性导致可解释性不足；③算法与传感器、数字孪生等生态系统的互操作性差；④行业内对标准化流程与评估指标（如POD）的认知不足。

---

## 6. Reading Between the Layers: Cross-Modal Alignment and Representational Hierarchy in Vision Transformers

**arXiv ID:** 2610.32463 | [PDF](https://arxiv.org/pdf/2610.32463v1)

**作者:** Sugam Karki `[一作]` (MBZUAI), Yova Kementchedjhieva `[通讯]` (MBZUAI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

系统性分析了基于Transformer的视觉编码器（如CLIP、DINO、MAE）在不同网络层级上对视觉文本的处理与跨模态语义对齐能力，使用渲染文本图像作为控制实验，并通过层级探测、检索与正交性控制等多种方法评估从表面形态到语法再到语义的层次结构。

**💡 创新点**

提出了一种全新的、以渲染文本图像为桥梁的跨模态评估框架，首次揭示了跨模态语义对齐仅在最深层出现，并与层级化的文字表征相对应，同时系统探讨了对比学习、无监督学习与掩码自编码在模型规模与数据组成上的差异对该层次结构的影响。

**🔧 技术方法**

采用ViT架构的视觉编码器冻结表示，结合层级归一化后的残差流；使用逻辑回归探测器进行字符识别、词长、词性、超义类别与抽象度预测；配合跨模态分类、零样本检索、关系检索与正交性控制实验。

**📊 数据集**

使用Caltech‑256、ImageNet‑1K、合成水果场景数据集；渲染文本图像与真实图像配对；SUBTLEX‑US、Universal Dependencies树库、WordNet以及抽象度评分数据构建探测任务；模型训练数据涵盖CLIP对比学习数据、DINO自监督数据以及MAE自编码数据。

**📈 对比分析**

通过层级化PAG（百分比准确度提升）与Recall@K等指标，CLIP在最深层表现出显著的跨模态语义对齐与语法/语义层次结构；DINO在规模提升至7B时获得一定改善；数据富文本化仅产生微小提升。相较于CLIP，DINO与MAE在深层跨模态对齐与语义探测上差距明显。

**⚠️ 局限性**

仅关注渲染文本的单词级评估，未覆盖句子级、语篇级及实际阅读理解所需的上下文与推理；探测方法只能测量可提取信息而非因果机制；仅评估视觉编码器，未考察完整视觉‑语言系统的影响。

---

## 7. Quantum-Classical Link Prediction on Dark Web Hyperlink Graphs

**arXiv ID:** 2610.32510 | [PDF](https://arxiv.org/pdf/2610.32510v1)

**作者:** Vojtěch Novák `[一作]` (VSB-Technical University of Ostrava), Ivan Zelinka `[通讯]` (VSB-Technical University of Ostrava)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

分析383个Tor onion主机的HTML内容与2,067条交叉主机链接，构建图结构并进行链接预测；同时对比传统与量子变分分类器的性能。

**💡 创新点**

提出在链接预测中使用源目标度匹配的负采样方法，显著降低基于端点度的偏差；将内容特征与图结构融合，并在相同压缩与训练预算下对量子与经典模型进行公平比较。

**🔧 技术方法**

使用TF-IDF+SVD文本特征、图自编码器(GAE/VGAE)、多模态逻辑回归、RBF SVC、MLP；量子模型为四比特变分量子分类器，采用线性ZZ和R_y编码，COBYLA/SPSA优化，exact与1,024-shot模拟。

**📊 数据集**

离线Tor爬虫数据，包含1,000个HTTP-200页面，383个主机，2,067条主机间有向链接；并提取HTML文本与静态HTML指示器。

**📈 对比分析**

在源条件目标度匹配的负采样下，经典多模态GAE/VGAE达≈0.90 ROC‑AUC；多模态逻辑回归≈0.67；量子四比特模型在exact下≈0.55 ROC‑AUC，接近RBF SVC（0.55），低于16参数MLP的0.57。

**⚠️ 局限性**

局限包括：爬虫为单次BFS组件，缺乏时间序列；大部分主机无交叉链接导致评估集中于少数发射节点；量子模型受限于极低维压缩、浅电路与有限数据；未对不同硬件噪声做评估。

---

## 8. Graphs of Trajectory Bundles for Nonlinear Motion Planning

**arXiv ID:** 2610.32365 | [PDF](https://arxiv.org/pdf/2610.32365v1)

**作者:** Joris Verhagen `[一作]` (KTH Royal Institute of Technology), Zachary Manchester `[通讯]` (MIT)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出 Graphs of Trajectory Bundles (GTB) 方法，用图搜索与轨迹束优化相结合，在非线性或黑盒动力学下实现全局运动规划。

**💡 创新点**

创新点在于：①将采样轨迹束嵌入凸集合图 (GCS) 中，形成动态可行性内近似；②通过逐步收缩采样束（trust region）和分数流（fractional flow）实现全局视角与局部收敛；③兼容黑盒动力学，无需雅可比、解析模型。

**🔧 技术方法**

使用技术包括：凸集合图 (GCS)、轨迹束方法 (TBM)、无导数采样插值、视角（perspective）松弛的 SOCP/LP、分数流与概率剪枝、annealing 采样尺度、动态不一致性边代价。

**📊 数据集**

实验数据集：车道迷宫（bicycle model）、平面无人机穿细缝、Mujoco CartPole 的 LTL 任务、SAM AUV 学习后的动力学模型。每个场景均使用多次随机种子（20 次）。

**📈 对比分析**

与 TBM 及 GCS+NLP 对比：GTB 在 20/20、18/20、19/20、19/20 的成功率均高于两者；平均计算时间约为 250–544 秒；在最短路径局部最优陷阱中表现更稳健，能显著降低动态不一致性容差。

**⚠️ 局限性**

局限性：无法保证全局最优；收敛依赖于初始图划分和采样策略；迭代求解的 SOCP/LP 计算量大；对高度耦合或快速变化的动力学可能需要更细致的采样；在极端非凸约束下仍可能出现分数流不收敛。

---

## 9. Behaviour analysis for large language model-driven agents

**arXiv ID:** 2610.32087 | [PDF](https://arxiv.org/pdf/2610.32087v1)

**作者:** Tianqiang Yan `[一作]` (Monash University), Lizhen Qu `[通讯]` (Monash University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了将行为发现、轨迹分析和实验设计结合的通用框架，并在自我反思任务上实现了 ReBeCA 原型。

**💡 创新点**

创新点在于使用明确的行为类别、因果发现和可解释假设来桥接数据驱动发现与实验验证，实现可解释行为分析。

**🔧 技术方法**

采用归纳编码、因果结构学习（GES）、统计评估、交叉验证和提示调控实验等技术。

**📊 数据集**

利用 Qwen3（0.6–8B）模型在 417 篇数学和 400 篇翻译的交互轨迹，以及 50 题 AIME 评测数据。

**📈 对比分析**

与全变量或无行为变量模型相比，选取的行为集合在预测和实验上表现更好，但提示组合效果不一，提升有限。

**⚠️ 局限性**

局限在于仅验证固定轮次的自我反思、样本规模有限、未在其他任务或模型族中测试、提示设计与解释性之间存在依赖。

---

## 10. Tempo-Zero: Recursive Self-Play in Executable Worlds for Video Reasoning

**arXiv ID:** 2610.32433 | [PDF](https://arxiv.org/pdf/2610.32433v1)

**作者:** Jingmin Zhu `[一作]` (Monash University), Qiuhong Ke `[通讯]` (Monash University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种基于可执行程序库的自我学习框架，通过程序生成的视频和事件轨迹实现完全可验证的监督；

**💡 创新点**

创新点在于：1）将可执行程序作为视频生成与标注的双重来源，实现标签和时间区间的自动、可验证生成；2）单一视觉‑语言策略同时担任问答生成者和解答者，利用执行结果直接评估问答质量；3）加入通过pass‑rate的课程分配和策略选取规则变更的环境演化机制，使训练数据随模型能力自适应扩展；

**🔧 技术方法**

核心技术包括可执行程序环境（生成视频及事件轨迹）、基于执行结果的奖励设计（答案、证据、定位等分量）、Group Relative Policy Optimization (GRPO) 的联合策略更新、基于pass‑rate的课程调度与规则变更选择，以及对Qwen3‑VL等大模型的LoRA微调；

**📊 数据集**

训练仅使用程序生成的合成视频，无需外部视频或人工标注；评测使用十二个公开视频基准，包括时间定位（ANet‑RTL、ActivityNet Captions、Charades‑STA、NExT‑GQA）、长视频理解（LongVideoBench、MLVU、Video‑MME‑Long、LSDBench）和视频推理（LongVideoReason、MMVU、VideoMathQA、VideoMMMU）；

**📈 对比分析**

与基准模型及三种自我学习基线（VisPlay、V‑Zero、Video‑Zero）进行对比，实验显示该方法在所有12个基准上平均提升显著，尤其在时间定位任务上超越所有基线；在4B/8B规模下取得最高分；

**⚠️ 局限性**

局限性在于：1）任务空间受程序库设计限制，可能缺乏足够的多样性来覆盖复杂真实场景；2）合成视频实体过于简单，可能导致模型对真实复杂视觉模式的迁移受限；3）缺少多模态或跨域的挑战；4）对程序实现的可解释性和可扩展性仍有待进一步研究。

---

## 11. ReLEAP: Upcycling Vision Datasets through Cross-Task Unification and Semi-Supervised Annotation

**arXiv ID:** 2610.32289 | [PDF](https://arxiv.org/pdf/2610.32289v1)

**作者:** Julia Simon `[一作]` (AIT Austrian Institute of Technology), Andreas Trondl `[通讯]` (AIT Austrian Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了ReLEAP框架，用于统一异构视觉数据集，支持注释回收和跨任务标签映射，实现数据再利用和自动注释升级。

**💡 创新点**

提出了可扩展的Python框架，利用共享层次化标签树实现跨任务注释统一，并通过跨数据集和半监督方式实现注释升级。

**🔧 技术方法**

标签层次化映射、数据源抽象、自动注释回收、基于SAM、YOLO、Mamba等模型的跨任务迁移和半监督学习。

**📊 数据集**

集成了15个精准农业公开数据集，包括玉米、甜菜、豆类、杂草等多种作物和种子。

**📈 对比分析**

通过在5个实例分割数据集上进行五轮交叉验证，比较无监督、弱监督、半监督及少量样本微调等方案，发现多源域特定模型在精度上优于单源模型，且速度比大型基底模型快10倍。

**⚠️ 局限性**

对稀有类别的识别受限，需手动标注或高质量源数据；基底模型在推理速度和资源消耗上仍显高；框架对非农业领域的迁移需额外构建词典。

---

## 12. ChoreoStudio: Planning What, When, and Who in Music-Driven Group Choreography

**arXiv ID:** 2610.32325 | [PDF](https://arxiv.org/pdf/2610.32325v1)

**作者:** Jing Xu `[一作]`, Qiuhong Ke `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出ChoreoStudio框架，利用多尺度音乐解耦和冻结多模态语言模型生成宏观“何-何-谁”计划，再通过计划编码驱动条件扩散模型实现3D多舞者同步舞蹈。

**💡 创新点**

创新点在于显式将舞蹈规划拆分为形成、时序与执行三要素，并通过关系图、转移门和移动分配实现可编辑、可验证的全流程规划；同时结合多尺度音乐特征和高层语义描述，显著提升宏观队形演变和群体组织。

**🔧 技术方法**

核心技术包括：音乐解耦器（帧级节奏与段级结构提取）、冻结Qwen3‑Omni多模态语言模型生成语义描述、导演网络（关系图、转移门、移动分配）、条件扩散运动生成器、验证器闭环优化和计划到运动的编码接口。

**📊 数据集**

使用公开的AIOZ‑GDance数据集（1,309训练，154测试，舞者数2–7）。

**📈 对比分析**

与GCD、CoDancers、TCDiff、TCDiff++、ST‑GDance等基线在150帧协议下比较，ChoreoStudio在GMR、GMC、TIF、IDDV、FMR、BAS等指标均取得最优或次优表现，特别在宏观队形动态与计划实现度上显著提升。

**⚠️ 局限性**

局限性包括：缺乏绝对舞台方向与更丰富的语义角色支持；对极大规模舞者或非标准音乐结构的适应性待进一步验证；生成过程仍需多阶段训练与手动提供音乐属性；长序列性能虽有报告但仍需系统评估。

---

## 13. Understanding Communication Failures in Multi-Agent LLM Systems

**arXiv ID:** 2610.32060 | [PDF](https://arxiv.org/pdf/2610.32060v1)

**作者:** Priya Pitre `[一作]` (Virginia Tech), Xuan Wang `[通讯]` (Virginia Tech)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一套基于多轮多智能体对话的通信失败分类体系，并利用该分类分析不同任务、模型配置下的失败模式。

**💡 创新点**

首次以通信过程为核心提出面向多智能体LLM的失败分类法，揭示恢复与共识失败在多轮对话中的主导地位，并提供实时预警机制。

**🔧 技术方法**

自动化文本摘要+GPT-5.4-nano生成标签，聚类/归并，人工验证，随机森林预测错误。

**📊 数据集**

涵盖十个领域（医疗QA、代码、金融、跨步推理、伦理、嵌入式文本）共计约7300个任务实例，其中3211条失败轨迹。

**📈 对比分析**

与人类注释和基线规则/LLM评估比较，自动标签micro‑F1达0.84，预警模型在错误预测上相比无触发提升≈20个百分点。

**⚠️ 局限性**

主要局限在于对特定任务的依赖、标签生成对GPT质量的敏感，以及未深入探讨因果干预效果。

---

## 14. SCI-AGENT-EXAM: Benchmarking Agents on Scientific Research Tasks

**arXiv ID:** 2610.32441 | [PDF](https://arxiv.org/pdf/2610.32441v1)

**作者:** Tianyun Zhong `[一作]` (University of Chinese Academy of Sciences), Xianpei Han `[通讯]` (Chinese Academy of Sciences)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

构建并公开了名为 SCI-AGENT-EXAM 的工作流导向基准，包含 120 个专家设计、涵盖 11 学科、7 任务类型的可执行科学任务，并评估 LLM 代理在可执行和文本交付物上的表现。

**💡 创新点**

创新点在于：①提出 Scientific Workflow Task Taxonomy (SWTT) 以系统化任务类型；②要求任务同时产生代码、报告和答案；③采用混合评估协议，结合程序化检测与经过校准的 LLM 判定；④对执行轨迹进行效率和稳定性分析。

**🔧 技术方法**

使用了 Docker 容器化执行、程序化检查脚本、预先训练的 LLM 评判器（如 Claude Code）以及基于专家编写的评分 Rubric；此外还进行了轨迹回放与成本/效率评估。

**📊 数据集**

数据集来源于近年科研论文、竞赛、开源项目，由 156 名领域专家构建，包含 120 个任务的输入、可执行包与参考解答，全部公开于 GitHub。

**📈 对比分析**

通过在同一固定环境下对 10 款前沿 LLM 代理（包括 Claude、GPT‑4、Qwen 等）进行三次独立评测，计算程序化得分、LLM 判定得分及加权最终得分；最强模型得分 80.9，但仅 26% 的运行达 0.95 以上，显示仍有提升空间。

**⚠️ 局限性**

局限性在于：评估仅聚焦通用基础模型与标准化代理设置；缺乏针对特定领域工具、记忆规划或迭代研究能力的专业化评测；基准任务数量相对有限，难以覆盖更广泛的科研场景。

---

## 15. Multi-Material Topology Optimization of Phononic Crystals for Targeted Elastic Wave Bandgap

**arXiv ID:** 2610.31999 | [PDF](https://arxiv.org/pdf/2610.31999v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 16. MICE to Meet You: From Re-ranking Only to End-to-End Retrieval

**arXiv ID:** 2610.31932 | [PDF](https://arxiv.org/pdf/2610.31932v1)

**作者:** Victor Morand `[一作]` (Sorbonne Université), Benjamin Piwowarski `[通讯]` (Sinequa by ChapsVision)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文把MICE重新排名模型扩展为完整的端到端检索系统，利用PLAID压缩并存储预先计算的文档表示。

**💡 创新点**

创新点在于：①提出低维瓶颈投影将MICE的上下文文档向量适配PLAID存储；②设计从已训练的交叉编码器迁移到MICE的分阶段Fine‑Tuning；③实现与SPLADEv3协同的完整检索管线，形成新的效率‑效果 Pareto 前沿。

**🔧 技术方法**

使用的技术包括：ModernBERT 基础网络、MICE 双阶段（上下文+交互）架构、FastPLAID 量化索引、线性投影+LayerNorm 瓶颈、点对点 MSE/Pearson 损失的迁移学习、SPLADEv3 一阶段检索。

**📊 数据集**

主要使用的评测数据集有：BEIR13（13 个去污染后的公开检索基准）、Nano‑BEIR、LoTTE‑Search（Search 版）以及 5M 文档的 Climate‑FEVER 用于存储测试。

**📈 对比分析**

实验中将 MICE-P（含瓶颈+PLAID）与相同规模的交叉编码器、ColBERT、LateOn 等进行 nDCG@10 对比；结果表明 MICE-P 的效果与交叉编码器差异 <0.3，且在 BEIR13 上平均超过 ColBERT 1+ nDCG，Latency 仅略高于 ColBERT 但远低于交叉编码器，构成新的效率‑效果平衡点。

**⚠️ 局限性**

局限性包括：①PLAID 压缩导致文档存储体积增大，②压缩对效果影响仍有 0.7–1 nDCG 的轻微下降，③ColBERT 在使用 PLAID 作为一阶段检索时性能显著下滑，说明该方案对文档向量分布要求更高。

---

## 17. LazyFlash: Forward-Dense, Backward-Sparse Attention for Accelerated Long-Context Fine-Tuning

**arXiv ID:** 2610.31911 | [PDF](https://arxiv.org/pdf/2610.31911v1)

**作者:** Keqi Deng `[一作]` (Microsoft), Jinyu Li `[通讯]` (Microsoft)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出 LazyFlash，一种在训练期间保持前向全密集、后向稀疏的注意力机制，以加速长序列 LLM 的训练。

**💡 创新点**

创新点在于：①仅对后向梯度做块级稀疏，前向保持标准密集；②使用梯度能量预算决定可跳过的块数，适应每个头、每步的梯度分布；③通过重要性采样保持梯度无偏，降低方差；④在 FlashAttention 后向核中直接跳过不加载的块，实现真实时间加速。

**🔧 技术方法**

技术手段包括：基于块的输出梯度范数评分、梯度能量预算、按梯度比例的 Bernoulli 采样、权重调度到 FlashAttention 后向核、支持变量长度输入的 CUDA 选取管线。

**📊 数据集**

实验数据集涵盖：长文本（LongAlign‑10k、ShareGPT、LongBench‑Pro）和长语音（50K 小时英语长片段、10‑分钟 ASR 测试集）。

**📈 对比分析**

与 FlashAttention‑2（全密集）和 MoBA（稀疏前向）对比：LazyFlash 在长文本上保持或提升总体准确率，同时后向时间减速 2.8×；在长语音上 EER/WER 均优于密集；整体训练速度在 32K 上可达 1.66×，后向加速 5.7×。

**⚠️ 局限性**

局限性包括：①需依赖 FlashAttention 后向核实现，难以迁移到非 FlashAttention 框架；②梯度稀疏化会导致方差上升，需调节预算 δ，且 δ 取值对不同任务敏感；③对极短序列或梯度分布高度均匀的场景效果有限；④目前只针对 Transformer 的注意力，未验证对其他模块的可扩展性。

---

## 18. Beyond Gain: Measuring Skill Internalization for Agent Skill Retirement

**arXiv ID:** 2610.32322 | [PDF](https://arxiv.org/pdf/2610.32322v1)

**作者:** Yanze Wang `[一作]` (EverMind), Yafeng Deng `[通讯]` (EverMind)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了基于模型自身未援助能力（unaided level）的技能退休判定方法，并引入有效性门（validity gate）来确保技能真正对任务有用。

**💡 创新点**

创新点在于：①用单一模型的未援助评分作为内部化判定依据，避免传统的增益（gain）判定误区；②构建可验证的标签可靠性框架；③提供公开的1251个技能及其任务、评分和可靠性指标的数据集。

**🔧 技术方法**

主要技术包括：可解释的离线任务合成器、LLM评审器、统计可靠性分析（内部一致性、测试再测信度、并行形式可靠性）以及阈值化的决策规则。

**📊 数据集**

使用的主要数据集为：①来自公开技能库的随机挑选的约1000个技能，②每个技能对应的单一合成任务（以及部分双任务），③在Qwen3.6‑27B、Qwen3.5‑397B、Qwen3.8‑27B和DeepSeek‑V4‑Flash等四个模型上评测，另外在SkillsBench和Tessl‑single上进行外部验证。

**📈 对比分析**

与传统的|A−U|<0.10增益退役规则和基于单臂得分的规则相比，本文的level rule在四个模型上可删除约57%–78%未覆盖技能，同时保留超过90%的已内部化技能；在SkillsBench基准上，使用该规则删除的技能对任务奖励影响在统计上不显著，表现与其他规则相当但更为可靠。

**⚠️ 局限性**

局限性包括：①未覆盖状态的判定仍依赖于单一任务的置信区间，任务难度的波动可能导致误判；②阈值（如0.8）需为每个模型单独校准；③目前只评估了公开的技能库，尚未验证在更大规模、多领域技能集上的可迁移性。

---

## 19. EDWARD-NMT: Elastic Dimensions With Adaptive Routing and Dual-consistency for Neural Machine Translation

**arXiv ID:** 2610.32115 | [PDF](https://arxiv.org/pdf/2610.32115v1)

**作者:** Edward Nguyen `[一作]` (UNEY), Vu Le `[通讯]` (UNEY)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研发了一种训练一次即可在多种资源环境部署的弹性Mixture-of-Experts（MoE）神经机器翻译架构EDWARD‑NMT。

**💡 创新点**

创新点包括Matryoshka式弹性专家、基于阈值的自适应Sigmoid路由以及双向一致性（dual‑consistency）正则化。

**🔧 技术方法**

采用了Elastic MoE、动态Sigmoid路由、FiLM调制的共享专家、对称KL一致性目标以及变异量保留缩放技术。

**📊 数据集**

在PhoMT (En↔Vi)、WMT‑14 (En↔De)、多域 De↔En 以及OPUS‑16 多语言数据集上进行实验。

**📈 对比分析**

与Dense Transformer Base/Big、mBART、Switch/​GShard MoE 基线对比，EDWARD‑NMT在全量模型下性能优于 Big，且各子网络（从同一检查点抽取）在参数量大幅压缩后仍优于对应大小的Dense模型，整体表现显著提升。

**⚠️ 局限性**

局限性在于仅在encoder‑decoder NMT 任务中验证，弹性仅扩展至 FFN 宽度层面，无法跨越最小训练宽度的极端压缩。

---

## 20. AusClinBench: a deterministically scored benchmark of large language models against current Australian clinical policy

**arXiv ID:** 2610.31993 | [PDF](https://arxiv.org/pdf/2610.31993v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 21. Why Full-Duplex Speech Models Collapse in Long Conversations

**arXiv ID:** 2610.32349 | [PDF](https://arxiv.org/pdf/2610.32349v1)

**作者:** Yeonju Kim `[一作]` (KAIST), Yong Man Ro `[通讯]` (KAIST)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了Full-Duplex Long-Horizon Benchmark (FD-LHB)，用来评估全双工语音模型在长时段对话中的稳定性。

**💡 创新点**

创新点在于识别并定位了全双工模型在长时段对话中崩溃的两大机制：初始化锚点清除和自我强化重复，并给出了无需重新训练的两种修复方法：Initialization-Anchor Pinning (IAP) 和 Duplex Repetition Penalty (DRP)。

**🔧 技术方法**

使用的技术包括KV缓存管理、重复惩罚、实时语音生成以及对模型输出的实时检测。

**📊 数据集**

数据集为CANDOR（长对话语料）以及与LLM伙伴进行的自适应对话，且使用了Gemini-2.5-Flash进行评判。

**📈 对比分析**

通过在FD-LHB-CANDOR和FD-LHB-LLM两条轨道上与五个最新全双工模型比较，IAP+DRP显著降低了重复、停顿和胡言乱语率，并在实时时间内保持高质量对话；在不做修复的基线中，大多数模型在几分钟至十几分钟内就出现崩溃。

**⚠️ 局限性**

局限性包括仅针对滑动窗口模型的缓存机制，未针对增长型上下文模型提供同样的修复；且仍无法保证极长（几十分钟以上）对话的持续稳定性。

---

## 22. Knowing Whom to Trust is not Enough: The Trust--Agreement Gap in Repeated Multi-Party LLM Negotiation

**arXiv ID:** 2610.31971 | [PDF](https://arxiv.org/pdf/2610.31971v1)

**作者:** Amit Sarker `[一作]` (University of Massachusetts Amherst), Shlomo Zilberstein `[通讯]` (University of Massachusetts Amherst)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了LLM代理在重复多方谈判中如何形成并利用显式信任以改善协议结果，并提出了TASMAN框架；

**💡 创新点**

首次引入跨剧信任机制并定义“信任-协议差距”，量化LLM在多轮谈判中识别隐蔽对手的能力；

**🔧 技术方法**

使用Beta信任系统、基于rubric评估、自由形式评估以及多种LLM推理模型（Gemini、GPT、DeepSeek等），并通过公开议题、BATNA和投票机制进行实验；

**📊 数据集**

采用自生成的多项谈判情境数据集（每剧5议题4选项，共20剧、20轮），在10条种子轨迹上重复实验；

**📈 对比分析**

与无信任、固定信任、随机信任等对照组比较，评估协议率、诚实方剩余收益、硬议员拒绝率等指标，发现信任指导下协议率提升9–23个百分点，最大提升23个百分点，但效果因模型和评估方法而异；

**⚠️ 局限性**

可靠信任并未始终转化为更好协议，模型对信任的使用有限；仅考虑单一隐蔽对手，未训练模型主动使用信任；实验仅在生成情境下进行，缺乏真实世界验证。

---

## 23. RobustNCL: Robustifying Neuromorphic Continual Learning under Transient Faults with Lightweight Detect-and-Cure Strategy

**arXiv ID:** 2610.31954 | [PDF](https://arxiv.org/pdf/2610.31954v1)

**作者:** Rachmad Vidya Wicaksana Putra `[一作]` (New York University), Muhammad Shafique `[通讯]` (New York University)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3855fcda-48ef-4070-a15e-803cd5c84d83` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 RobustNCL 框架，利用检测-恢复策略在 DRAM 发生瞬态故障时保证神经网络的持续学习可靠性。

**💡 创新点**

创新点在于首次系统研究 NCL 在硬件瞬态故障下的鲁棒性，构建权重分布统计库并提供零/极值/最常见值三种恢复变体，且采用轻量级检测-恢复机制。

**🔧 技术方法**

使用权重分布统计、瞬态位翻转注入、辐射硬化存储、零/极值/最常见值替换、Python+GPU 仿真实现检测与恢复。

**📊 数据集**

实验采用 Split MNIST、Split Fashion‑MNIST、Split CIFAR‑10、Split CIFAR‑100 等分割任务数据集。

**📈 对比分析**

与无故障补偿 baseline 对比，结合 Subnet 与 MR 两种 NCL 方法，覆盖多种 fault rate；结果显示 RobustNCL 在高 fault rate 下平均准确率仅低于 0.5%，延迟提升不超过 2.3%，能耗提升约 6.35%。

**⚠️ 局限性**

局限性包括：仅针对 DRAM 级瞬态故障，未考虑其他硬件错误；恢复策略基于静态统计，可能在极端动态变更时失效；实验仅在 GPU 环境下验证，缺乏真实嵌入式平台的实测。

---

## 24. Machine-Readable Operator Manuals over the ISO 11783 Network for Offline, Multilingual LLM Assistance

**arXiv ID:** 2610.31863 | [PDF](https://arxiv.org/pdf/2610.31863v1)

**作者:** Timo Oksanen `[一作]` (Technical University of Munich), Julius Gun `[通讯]` (Munich Institute of Robotics and Machine Intelligence)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了在 ISO 11783 CAN 总线上通过 ETP 传输完整操作手册，并评估本地 LLM 进行跨语言问答的可行性

**💡 创新点**

首次提出“语义同步”概念，将实现设备手册与拖拉机侧助手的知识库保持一致，并验证 Markdown 为最佳传输格式；证明大型 LLM 对多语言手册的跨语言检索无显著优势

**🔧 技术方法**

采用 ISO 11783‑6 的 ETP、Markdown/JSON/XML 结构化文件、检索增强生成（RAG）+语义检索、开源 LLM（如 Qwen、Gemma、Llama 等）以及 4‑bit 量化推理

**📊 数据集**

使用 Kverneland Exacta‑TLX Geospread GS3 的 6 语言版本操作手册（PDF→Markdown/JSON/XML）和 108 题的 QA 数据集（54 可答、54 不可答）

**📈 对比分析**

在 31–89 s 的传输时间内，Markdown 产生最小文件、最快传输；所有格式在大多数模型上回答质量相近，英语版 F1 0.77–0.83，其他语言下降 0.12–0.30；语义检索对非英语手册至关重要，128 B 模型无明显优势

**⚠️ 局限性**

局限包括仅测试单一机型手册、QA 集非真实用户问答、所有实验在 PC 级硬件上完成、未评估嵌入式 ECU 性能与安全性、未考虑手册篡改或 prompt 注入等安全问题

---

## 25. Angular Supervision for Visual Object Grounding in Autoregressive Vision-Language Models

**arXiv ID:** 2610.32220 | [PDF](https://arxiv.org/pdf/2610.32220v1)

**作者:** Parsa Esmaeilkhani `[一作]` (Temple University), Longin Jan Latecki `[通讯]` (Temple University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

**🎯 论文内容**

未提供论文内容，无法说明

**💡 创新点**

未提供论文内容，无法说明

**🔧 技术方法**

未提供论文内容，无法说明

**📊 数据集**

未提供论文内容，无法说明

**📈 对比分析**

未提供论文内容，无法说明

**⚠️ 局限性**

未提供论文内容，无法说明

---

## 26. Do Halting Heads Really Save Compute in Looped Language Models?

**arXiv ID:** 2610.32246 | [PDF](https://arxiv.org/pdf/2610.32246v1)

**作者:** Tsung-Han Wu `[一作]` (Apple), Kenny Tan `[通讯]` (Apple)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究循环语言模型中的停机头，探讨其训练损失、退出规则与批量推理对计算节省的影响，并提出更可靠的评估方法。

**💡 创新点**

创新点在于揭示混合损失无法训练单个迭代，指出仅按每步权重评分不足以实现早停，需要强先验；同时发现批量推理会让停机头的计算优势几乎消失，并给出预训练前评估头收益的上限测试。

**🔧 技术方法**

使用循环Transformer结构、PonderNet式停机头、混合损失和逐步损失（带/不带先验）、多种退出阈值、批量与逐序列推理、键/值重用技术。

**📊 数据集**

在两大公开语料上进行预训练：FineWeb‑Edu（过滤后的网页抓取）和Common Corpus（公开授权文本）。

**📈 对比分析**

与无停机头的随机深度基线及专家模型（固定深度）对比，发现逐步损失加强先验时可在单序列推理下略降0.02–0.04 BPB，但在32序列批量推理下几乎无效；混合损失的停机头在任何设置下几乎不节省计算。

**⚠️ 局限性**

局限性包括仅在最大1.3 B参数规模下实验，测试假设深度曲线平坦、梯度覆盖受限，键/值重用的通用性尚未验证，且对多核高性能推理实现未充分评估。

---

## 27. Beyond What to Say: Proactive Utterance Timing with Inner Thoughts for Werewolf Agents

**arXiv ID:** 2610.32107 | [PDF](https://arxiv.org/pdf/2610.32107v1)

**作者:** Yoshiki Tanaka `[一作]` (University of Electro-Communications), Michimasa Inaba `[通讯]` (University of Electro-Communications)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在狼人杀游戏中实现了基于 LLM 的 Adversarial Inner Thoughts 框架，用于决定何时发言；

**💡 创新点**

创新点是将 Inner Thoughts 框架改造成可在对抗性、不完整信息环境下控制发言时机，并通过多维度评估与阈值决策来判断是否说话；

**🔧 技术方法**

使用了 LLM（ChatGPT/Claude 等）+语义检索、规则化长期记忆、阈值决策、内部思考生成与评估等技术；

**📊 数据集**

使用了 AIWolfDial 2026 共享任务的自对弈日记（日本五人版）作为实验数据；

**📈 对比分析**

对比方法是将系统与仅基于触发的基础系统进行六维度 Likert 评分；结果显示系统在所有维度平均得分略高，虽无显著差异，但在交流质量、内容论证等方面表现更好；

**⚠️ 局限性**

限制包括样本量小、评审单人评价、仅自对弈、仅日本五人版、未分离发言时机与内容、阈值经验性设置等。

---

## 28. Sharp Bounds on the Winner's Bias under Independent and Nonnegatively Correlated Noise

**arXiv ID:** 2610.31948 | [PDF](https://arxiv.org/pdf/2610.31948v1)

**作者:** Ivan Viakhirev `[一作]` (lab260), Grach Mkrtchian `[通讯]` (lab260)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了在独立和非负相关噪声下，最佳选择的偏差界限，确定了不同噪声依赖性情况下的偏差常数。

**💡 创新点**

创新点在于为独立、非同分布噪声和成对非负相关噪声提供了明确的偏差常数，填补了现有文献中的空白。

**🔧 技术方法**

使用了数学不等式和机器验证工具Lean 4进行形式化验证。

**📊 数据集**

使用了448个语言模型和分类器选择管道的配置数据集。

**📈 对比分析**

通过与现有的高斯模型进行比较，发现独立类的设计在固定预算下的乐观程度显著低于共享项目的设计，且在多个配置中实现了中位数0.75的偏差常数。

**⚠️ 局限性**

限制在于所得到的常数是针对每个依赖类的最坏情况，且数值研究的范围较窄，未能涵盖不同模型规模的混合情况。

---

## 29. The permutation and automorphism groups of univariate multiplicity codes

**arXiv ID:** 2610.31858 | [PDF](https://arxiv.org/pdf/2610.31858v1)

**作者:** Eduardo Camps Moreno `[一作]` (Institut de Mathématiques de Bordeaux), Rodrigo San-José `[通讯]` (Virginia Tech)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了多重码（univariate）在NRT度量和折叠Hamming度量下的线性自同构与置换群，给出了精确描述；

**💡 创新点**

创新点在于首次完整刻画多重码的自同构群，揭示其与投影变换（projectivities）和标量乘法的关系，并区分了乘性=1与≥2时的置换结构；

**🔧 技术方法**

采用Hasse导数、射影几何（交比）、线性代数与图论（(k-1)-degree图）等数学工具，推导自同构的矩阵形式；

**📊 数据集**

无数据集，纯理论推导；

**📈 对比分析**

与已知的Reed–Solomon自同构群对比，证明当r=1时恢复经典结果；对折叠Hamming度量给出更大自同构群的结构，说明NRT自同构是其子群；

**⚠️ 局限性**

仅适用于中心维度范围（n≥3且r≤k≤rn−r），低高维度及极端情形在附录处理；此外对r≥2时置换仅为平移，限制了可用的对称变换。

---

## 30. Virtual Placebo: a scoping review of control conditions in randomized controlled trials of psychological immersive virtual reality interventions

**arXiv ID:** 2610.32232 | [PDF](https://arxiv.org/pdf/2610.32232v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 31. MAGIC: Margin-gated Contrastive Debiasing with Evidence Restoration for VLM Test-time Robustness Scaling

**arXiv ID:** 2610.32013 | [PDF](https://arxiv.org/pdf/2610.32013v1)

**作者:** Shunshun Gu `[一作]` (University of Wisconsin--Madison), Xiao Luo `[通讯]` (University of Wisconsin--Madison)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种无训练、基于门控的对比解码方法，能够在推理时对大规模视觉‑语言模型进行去偏修正。

**💡 创新点**

创新点在于：①根据决策边距动态决定是否触发去偏；②在去偏后恢复被扰动去除的视觉证据；③利用校准集按模型自适应剪枝扰动分支。

**🔧 技术方法**

技术手段包括：对比解码、门控机制、证据恢复、校准集分支剪枝和多种视觉扰动（全黑、灰度、模糊、中心遮挡）。

**📊 数据集**

实验数据集涵盖 DRBench 的 B、S、BS 子集，以及六个真实世界 LVLM 数据集（MMBench、MME、ViLP 等）和两代主干模型（Qwen2‑VL、LLaVA‑NEXT、Qwen3‑VL、LLaVA‑OneVision）。

**📈 对比分析**

与 CD 家族（SCI、M3ID、VCD、TIE 等）对比，所提方法在 BS Overall 上提升约 15–20%，同时在计算成本上保持更低或相当的前向传播次数，形成最佳的准确率-计算成本折中。

**⚠️ 局限性**

局限性：仅适用于固定候选集的得分输出，无法在 token‑by‑token 解码中介入；门控阈值需针对每个分布手动重新校准；不支持自由文本生成任务。

---

## 32. Real-to-Sim Calibration of Bosch Etching Simulators with Few-Shot XSEM Data and Hybrid Search

**arXiv ID:** 2610.31904 | [PDF](https://arxiv.org/pdf/2610.31904v1)

**作者:** Xinyu Jia `[一作]` (Purdue University), Zhihong Chen `[通讯]` (Purdue University)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种基于少量真实XSEM测量的Real‑to‑Sim校准框架，利用ViennaPS模拟器的可调 flux 参数来推断每条工艺配方的专属模拟参数向量，从而实现对Bosch刻蚀深度和壁角的准确预测。

**💡 创新点**

创新点在于：①将模拟器参数与工艺变量解耦，形成可针对每条配方的共享参数；②采用混合搜索策略，结合局部随机探索、LLM引导的候选生成和XGBoost代理筛选，显著降低昂贵模拟的调用次数；③通过少量终端测量实现高质量校准，弥补传统仿真缺乏真实数据的缺陷。

**🔧 技术方法**

主要技术包括 ViennaPS 物理仿真器、LLM（如 Qwen3.8）生成物理可行的参数更新、XGBoost 回归代理用于快速评估候选参数，以及基于测量误差的损失函数进行参数优化。

**📊 数据集**

使用了两套真实 Bosch 刻蚀样本集（共20个样本），每套包含不同周期数和开口尺寸的 XSEM 深度与壁角测量，训练时仅采用每套 12 个样本作为拟合集，剩余 8 个样本作为测试集。

**📈 对比分析**

与默认 ViennaPS 参数和仅使用随机/LLM/代理搜索等基线方法比较，本文方法在测试集上的归一化全局误差分别降至 0.340（配方1）和 0.249（配方2），深度 MAE 下降至 0.78 µm/0.146 µm，壁角 MAE 下降至 0.85°/0.83°，显示出显著的性能提升。

**⚠️ 局限性**

局限性包括：仅在低维 flux 参数空间（8 个参数）上验证，尚未探索高维/多物理耦合的仿真器；依赖 ViennaPS 的具体实现，迁移到其他刻蚀模拟器可能需要额外调试；少量真实样本虽足够，但在更大规模工艺库中的泛化能力仍待进一步验证。

---

## 33. MI-MIDI 2: From Observation to Control with Sparse Autoencoders in Text-to-MIDI Generation

**arXiv ID:** 2610.32007 | [PDF](https://arxiv.org/pdf/2610.32007v1)

**作者:** Jakub Poćwiardowski `[一作]` (Warsaw University of Technology), Mateusz Modrzejewski `[通讯]` (Warsaw University of Technology)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并验证一种端到端协议，用于评估文本到MIDI生成模型中无监督稀疏自编码器（SAE）特征的可解释性与可控性。

**💡 创新点**

结合多重控制（注意力密度调整、置换校准、随机方向对照）评估观测证据是否能预测特征驱动的生成输出，并首次发布SAE字典与特征图谱。

**🔧 技术方法**

采用稀疏自编码器（TopK）、相关性调整、置换控制、解码器方向干预、随机方向对照、CLM（Claude Sonnet 5）自动描述与审核，以及piano roll可视化等技术。

**📊 数据集**

使用1500条未加文本提示的Lakh MIDI数据集进行字典训练、特征挖掘与评估。

**📈 对比分析**

通过八至十个提示，对每个特征与基线、随机方向进行配对符号检验与Benjamini–Hochberg校正；结果显示合成主音特征在确认实验中将乐器程序出现率提升至65%（相对基线），而密度调整后的特征在目标音乐指标上平均提升约0.17个标准差。

**⚠️ 局限性**

仅使用单一随机种子训练字典、屏幕使用极少提示、未进行听觉评估、缺乏文件级重叠审计、模型差异难以归因、且评估仅针对符号输出。

---

## 34. Beyond On-Policy vs. Off-Policy: At What Level Should LLM Agent Distillation Follow the Student?

**arXiv ID:** 2610.31918 | [PDF](https://arxiv.org/pdf/2610.31918v1)

**作者:** Ya Gao `[一作]` (Aalto University), Pekka Marttinen `[通讯]` (Aalto University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并实验验证了层次化代理蒸馏框架，将交互式LLM的 on-policy 选择细分为历史层和动作前缀层，并系统评估四种配置（TT、ST、TS、SS）、其混合组合及训练调度对自蒸馏性能的影响。

**💡 创新点**

创新点在于：①把交互式 agent 蒸馏中的 on-policy 与 off-policy 细分为两级（历史级与前缀级）；②提出可独立控制这两级来源的四种配置及其混合与时间调度策略；③通过大规模实验证明学生历史 + 教师前缀最优，混合与调度能进一步提升性能。

**🔧 技术方法**

使用技术包括：自蒸馏（teacher frozen，学生从同一 checkpoint 初始化）、rank‑128 LoRA 微调、ReAct 交互格式、前缀级/历史级的四种配置、前向 KL 作为损失、混合权重和学习曲线调度；评估指标为成功率 (SR)、pass@3、任务分数和平均动作数。

**📊 数据集**

实验数据集为 ToolQA、OfficeBench、ALFWorld 和 WebShop；教师指导通过人工撰写和 GPT‑5.4 生成；模型三种：Qwen3‑8B、Qwen3‑14B 和 Ministral‑3。

**📈 对比分析**

比较方法为：1）单独训练四配置并对比性能；2）在固定训练预算下构造等边、对角、交叉以及各种权重组合的混合，比较成功率提升；3）对同一混合做不同时间调度（TT‑first、SS‑later、TT‑first+SS‑later）比较；结果显示 ST 以最高的 SR 和 pass@3 为基准，混合可提升 0.5–3.3 个百分点，TT‑first+SS‑later 调度获得最佳最终验证性能，整体相比初始模型提升显著。

**⚠️ 局限性**

局限性包括：①仅在自蒸馏且 teacher 冻结、同模型起点的实验；对更大规模模型或开放式 teacher 的推广尚未验证；②混合比例未给出全局最优解，只通过对比得到经验原则，可能对不同任务或模型存在差异。

---

## 35. Robotic Budding: From Homogeneous Cells to Diverse Morphologies in Physical Self-Reproduction

**arXiv ID:** 2610.31929 | [PDF](https://arxiv.org/pdf/2610.31929v1)

**作者:** Siyuan Zhang `[一作]` (Columbia University), Hod Lipson `[通讯]` (Columbia University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

使用单一模块设计，实现多形态机器人通过磁性对接和关节折叠进行芽式自我复制

**💡 创新点**

创新点在于同一模块可支持不同拓扑形态，并通过硬件实现芽式物理复制，而非仅模拟或半支架化环境

**🔧 技术方法**

采用磁性对接、双关节折叠、无线命令广播、预编程分离序列等硬件与固件技术

**📊 数据集**

未使用公开数据集，仅在实验室随机分散的5/3个模块进行测试

**📈 对比分析**

对比方法仅为单次演示，未进行统计，展示了完整的收集-组装-分离流程；性能上表现可行，但缺乏量化指标

**⚠️ 局限性**

主要局限为：收集与组装仅由人机遥控完成，缺乏自主导航与对接检测；单次成功演示，未验证多次实验或子代再复制

---

## 36. SkillRep: Characterizing and Understanding the Representation Design Space of Agent Skill Retrieval

**arXiv ID:** 2610.32336 | [PDF](https://arxiv.org/pdf/2610.32336v1)

**作者:** Tianyi Sun `[一作]` (EverMind, Shanda Group), Yafeng Deng `[通讯]` (EverMind, Shanda Group)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了 SkillRep 检索模型，结合编码器微调、查询/文档侧扩展、背景投影与词汇通道等五种表示选择，以提升 LLM 代理检索技能的效果。

**💡 创新点**

创新点在于首次统一将五种表示选择组合到单一检索器中，并通过定义三种 gap 指标（背景、未说词汇、稀释）解释各模块的交互与增益来源。

**🔧 技术方法**

采用 Qwen3-Embedding-0.6B 进行对比学习微调，利用 LLM 生成假请求/假技能实现查询/文档扩展，使用背景投影消除公共向量，结合 BM25 词汇通道并通过 z-score 标准化后线性融合得分。

**📊 数据集**

在 SRA-Bench、SkillsBench 以及自建的 111,247 技能库 UniPool 上进行评估；训练集为 38,308 合成的请求–技能对，测试集覆盖 5,400 请求与 6,006 请求/技能。

**📈 对比分析**

与现有同规模检索器对比，SkillRep 在 SRA-Bench 上 MRR@10 提升 7.02/8.79 分，在 SkillsBench 上 MRR@10 提升 6.12/10.91 分，整体表现为同规模最佳。

**⚠️ 局限性**

局限性包括：需要 LLM 生成扩展导致检索时额外计算成本；模块增益受编码器影响，未针对更大规模或多模态技能库进行充分验证。

---

## 37. Omni-UBench: Benchmarking Omni-Modal LLM-based Guardrails Across Safety, Utility, and Reliability

**arXiv ID:** 2610.31713 | [PDF](https://arxiv.org/pdf/2610.31713v1)

**作者:** Junhao Wang `[一作]` (Zhejiang University), Kui Ren `[通讯]` (Zhejiang University)

**通讯引用:** 36059 | [OpenAlex ID](https://openalex.org/A5000596496)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了Omni‑SafetyBench基准，构建了183,977个跨文本、图像、音频、视频及其组合的样本，并提供统一的10类风险标签与三维评估框架（安全性、实用性、可靠性），用于系统评估多模态安全守卫模型。

**💡 创新点**

创新点在于：①整合35+公开数据集，形成跨模态、多任务的大规模评估集；②设计安全-实用-可靠三维指标，并细化为五个子模块（基础安全、越狱防护、干扰防护、过度敏感、幻觉检测）；③使用LLM辅助重标注实现标签统一；④对多模态守卫模型进行综合对比，揭示其在隐藏意图、复杂越狱与视频安全方面的弱点。

**🔧 技术方法**

技术上主要采用：多模态大型语言模型（如Qwen2.5‑Omni、Qwen3‑Omni、Gemini‑2.5‑Flash、MiMo‑V2.5）作为基准模型；基于这些模型微调或直接训练专用守卫（MultiGuard、OmniGuard、GuardReasoner‑Omni）；LLM辅助的两阶段标注流程；输入掩码与文本模式重写用于可靠性测试。

**📊 数据集**

使用的数据集包括但不限于：Aegis2.0、OpenAI Moderation、WildGuard、ToxicChat、VLSBench、VLGuard、AIAH、SafeWatch、Video‑SafetyBench、Omni‑SafetyBench、JailbreakBench、MM‑SafetyBench、CipherChat、AJailBench、Omni‑SafetyBench、以及其他35+公开文本、图像、音频、视频源。

**📈 对比分析**

比较方法：在安全、实用、可靠三维度分别计算F1/准确率/召回率或FPR，然后宏平均归一到0–100分，最终通过几何平均得到S_balance。实验表明：大型OLLM（如MiMo‑V2.5、Gemini‑2.5‑Flash）在安全与实用上表现突出；但在跨模态越狱和视频场景中守卫模型仍落后；安全幻觉与过度敏感问题普遍存在。

**⚠️ 局限性**

局限性：①基准对视频复杂场景和跨数据集泛化仍不足；②安全与可靠的权衡难以同时优化；③风险类别边界模糊导致误判；④模型对表面文本诱导模式过度敏感，容易产生误拒。

---

## 38. RADIANT: Object-Centric Feed-Forward Global Illumination

**arXiv ID:** 2610.32075 | [PDF](https://arxiv.org/pdf/2610.32075v1)

**作者:** Sarah Zhang `[一作]` (Massachusetts Institute Of Technology), Wojciech Matusik `[通讯]` (Massachusetts Institute Of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于点输入、无网格、面向对象的全前向全局照明渲染器，能在单次前向推理中合成新场景的光照效果；

**💡 创新点**

创新点包括：①使用每个对象仅32个表面补丁代币，令自注意力长度与对象数而非三角形数量相关；②引入不变形式因子注意力，将距离和法线角度等光辐射传输的几何不变量直接嵌入查询‑键内积，避免显式 O(N²) 关系矩阵；③采用点剖面光栅化提供主视可见性，解耦光线与场景自注意力；

**🔧 技术方法**

技术实现主要包括：点云到补丁代币的 PointNet++ 抽象、旋转位置嵌入 (OBB‑RoPE)、不变形式因子注意力、CamRay 视角编码、点剖面 G‑buffer、Dense Prediction Transformer 进行图像上采样；

**📊 数据集**

使用基于 1.03M 场景的合成数据集，场景由 1–6 个 Objaverse 对象、墙体、灯光随机组合构成，训练与评估均采用 Blender Cycles 产生的 512² 解析度图像；

**📈 对比分析**

与 RenderFormer 及 RenderFormer‑V2 在匹配模型容量（≈206M 参数）与训练预算下比较，改进模型在 600k 步时获得 32.22 dB PSNR（比 RenderFormer 高 3.3 dB），并在 16k 步时实现 25 dB，速度提升 5×；在 500 对象场景的单次前向推理中，速度比 RenderFormer 快 8.5 倍；

**⚠️ 局限性**

局限性包括：统一使用 32 个补丁代币可能对平面墙过度分配，对复杂几何不足；视角解码仍需完整自注意力，难以扩展到更高分辨率；点剖面仅捕获直接可见性，无法处理体积媒介；输入法向噪声可能影响渲染质量。

---

## 39. Translations in library catalogues: how to find them?

**arXiv ID:** 2610.32176 | [PDF](https://arxiv.org/pdf/2610.32176v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 40. The Exact Robustness of Smith's Rule with Predicted Processing Times

**arXiv ID:** 2610.32197 | [PDF](https://arxiv.org/pdf/2610.32197v1)

**作者:** Ohad Ayalon `[一作]` (Bar Ilan University), Ilan Reuven Cohen `[通讯]` (Bar Ilan University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究在处理时间仅为乘法误差α（α≥1）近似已知的条件下，Smith 规则（按权重/估计处理时间比排序）在相同机器上的列表调度性能，并给出了该规则在所有机器数下的精确最坏情况近似比；

**💡 创新点**

创新点在于首次给出完整的α-依赖最坏比曲线 ℒ(α)，证明当α≥4时最坏比恰为 √α，并证明对于任意固定机器数的确定性非抢占算法，这一极限无法被突破；此外提出“前缀-后缀证书”方法，将复杂多机分析归约为对端点实例的单机两块块分析，得到闭式解；

**🔧 技术方法**

核心技术包括两重归约：先将任意实例归约为所有工作相同 Smith 比值的实例，再归约为端点实例（每个工作处理时间仅为 μ_j 或 αμ_j）；随后采用前缀-后缀切割与矩阵不等式相结合的证明技巧；并使用参数化证书（λ、τ）和多项式不等式验证实现最优曲线；

**📊 数据集**

本研究使用构造的最坏实例（多类工作，权重/时间比例相同且仅取两种真实处理时间）作为实验数据，没有采用公开数据集；

**📈 对比分析**

比较方法主要是理论证明和极值分析，结果显示列表规则的实际近似比达到所推导的上界，且在α≥4时与下界一致，证明其为最优；另外通过“nominal‑to‑robust”转换，得到一个多项式时间算法，在1≤α≤4范围内可实现 (1+ε)√α 的近似比；

**⚠️ 局限性**

局限性包括：尚未给出单一多项式时间算法能在所有α≥1上同时实现 (1+ε)√α；此外仅考虑非抢占式批处理，未探讨抢占、异构机或随机预测误差等更一般场景；

---

## 41. VisFlow: Interpretable Flow Matching for Visibility Reconstruction, Characterization, and Recognition

**arXiv ID:** 2610.31892 | [PDF](https://arxiv.org/pdf/2610.31892v1)

**作者:** Rahul Moorthy `[一作]`, Volkan Isler `[通讯]`

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `4de8e9d8-757b-475f-9627-18a445e50202` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于连续SDF和显式多边形演化的可解释流匹配模型，用于从可见性图生成多边形及其所有满足条件的解

**💡 创新点**

创新点包括：①将SDF与几何演化结合，构造线性潜在路径；②引入可微可见性预测网络实现梯度可导的可见性监督；③使用基于Grassfire的轨迹监督提升生成多样性；④将同一框架迁移到蛋白质接触图结构生成

**🔧 技术方法**

使用了连续SDF表示、PatchNets + Transformer编码器-解码器、可微可见性预测网络、Rectified Flow（时间条件U-Net）以及梯度优化进行最终细化

**📊 数据集**

构建了新的可见性图重建数据集，在形状多样性上比现有数据集提升22%，并在该数据集上进行评估；同时在蛋白质接触图上进行实验验证

**📈 对比分析**

与基线方法（如VisDiff、传统GNN/mesh方法）相比，模型在主基准数据集上F1得分提升约20%，在多样性评估中的覆盖率提升约30%；实验表明显著优于现有最先进技术

**⚠️ 局限性**

局限性包括：①SDF的PatchNets表示假设形状信息均匀分布在固定空间 patch，限制了对不均匀或复杂形状的适应；②目前未充分利用接触图的结构信息，可进一步结合AlphaFold等方法提升蛋白质生成效果

---

## 42. CRM-Agent-Bench: Benchmarking Tool-Using Agents in Stateful Enterprise Environments

**arXiv ID:** 2610.32004 | [PDF](https://arxiv.org/pdf/2610.32004v1)

**作者:** Monojit Banerjee `[一作]`, Defu Cao `[通讯]`

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

暂无具体内容

**💡 创新点**

暂无具体内容

**🔧 技术方法**

暂无具体内容

**📊 数据集**

暂无具体内容

**📈 对比分析**

暂无具体内容

**⚠️ 局限性**

暂无具体内容

---

## 43. Rethinking Multi-View Fusion for Urban Representation Learning

**arXiv ID:** 2610.31850 | [PDF](https://arxiv.org/pdf/2610.31850v1)

**作者:** Sean Bin Yang `[一作]` (Aalborg University), Christian S. Jensen `[通讯]` (Aalborg University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

针对城市区域表示学习中的多视图融合问题，提出了自适应区域路由与PID式梯度平衡框架PRISM。

**💡 创新点**

创新点在于：①利用稀疏Mixture-of-Experts实现区域级视图自适应分配；②采用PID启发的误差正则化调控多目标梯度冲突，提升训练稳定性。

**🔧 技术方法**

技术手段包括稀疏MoE路由、基于Transformer的自注意力编码、PID-MSE正则化、基于不确定性加权的多任务学习。

**📊 数据集**

使用纽约、芝加哥、旧金山三市的出租车OD、POI、用地、犯罪、签到、服务请求等数据。

**📈 对比分析**

与多种基线（Concat、PGN、HGN等）在检查点预测、犯罪预报、服务请求预测三任务中对比，PRISM在MAE/RMSE/R^2上均优于对手，表现最优。

**⚠️ 局限性**

局限在于：当主导视图严重缺失或噪声过大时，稀疏路由可能失效；模型对视图缺失的鲁棒性仍需进一步提升。

---

## 44. OpenHome: From Perception to Insight in Smart-Home Camera Videos

**arXiv ID:** 2610.31986 | [PDF](https://arxiv.org/pdf/2610.31986v1)

**作者:** Xianglong Wang `[一作]` (Wyze Labs), Lin Chen `[通讯]` (Wyze Labs)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并构建了一个基于真实智能家居摄像头视频的视觉语言模型（VLM）评测基准，包含2,254道四选一多项选择题。

**💡 创新点**

创新之处在于：①引入真实家居摄像头场景并针对其独特的视角、光照、截断等问题进行数据采集；②设计了从感知（Perception）、推理（Reasoning）到洞察（Insight）三层能力层级，涵盖10个任务；③所有问题均可仅凭视觉证据独立回答，避免文本先验偏差。

**🔧 技术方法**

方法上使用自动化问题生成（Gemini 2.5 Pro、Gemini 3.8 Flash、GPT‑6等）、人类事实校验、视觉事实检验以及统一的评估协议（每秒1帧、最多64帧、无音频）来评测VLM。

**📊 数据集**

数据集由Wyze社区公开的1,118段约5.97小时摄像头录像组成，平均时长19.2秒，涵盖昼夜、红外、室外等多种光照和视角，主体包括动物、车辆、人等多种类型。

**📈 对比分析**

对32款VLM（闭源与开源）在每个任务上的准确率进行比较，最佳模型Gemini 4 Argon整体准确率达81.2%；闭源模型普遍优于开源，Insight层的性能差距最大；计数与时间顺序任务最难。

**⚠️ 局限性**

局限性包括：仅使用视觉信息，缺少音频与语义上下文；部分事件因视角或截断无法完整呈现，导致模型难以推断；评估只使用最小推理设置，未能充分展示模型的潜在推理能力。

---

## 45. PixGR: Pixel-Space Generative Decoding for Fast Feed-Forward 3D Reconstruction

**arXiv ID:** 2610.31908 | [PDF](https://arxiv.org/pdf/2610.31908v1)

**作者:** Ranran Huang `[一作]` (Imperial College London), Krystian Mikolajczyk `[通讯]` (Imperial College London)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `f86bf285-fd08-4156-973b-6e6481af8fa0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种轻量化像素空间生成模型与3D高斯重建的端到端框架，能够在稀疏视角下建模场景不确定性；

**💡 创新点**

创新点在于将像素空间流匹配直接嵌入3D Gaussian重建，消除VAE压缩瓶颈，实现生成与重建协同优化；采用轻量化DiT解码器并缓存场景特征，实现子秒推理；

**🔧 技术方法**

主要技术包括像素空间流匹配（x-prediction）、3D Gaussian splatting、可微渲染、VGGT预训练的场景编码器、DiT解码器、RMSNorm与RoPE；

**📊 数据集**

在RealEstate10K、DL3DV数据集上训练评估，并在ACID数据集进行跨域测试；

**📈 对比分析**

与传统基于回归的3DGS方法（如MVSplat、DepthSplat、AnySplat）及生成式方法（Bolt3D、MVSplat360、latentSplat、ReconSplat）对比；在视角插值和外推任务中均优于回归方法，且在子秒推理内可匹敌更大型生成模型；

**⚠️ 局限性**

仍受限于对VGGT预训练的依赖，轻量解码器在极端遮挡或细节丰富场景下可能无法捕捉所有细节；

---

## 46. Correct Answers, Closed Gates: SolveBind for GUI Agents at the Verification Barrier

**arXiv ID:** 2610.32223 | [PDF](https://arxiv.org/pdf/2610.32223v1)

**作者:** Xinhao Song `[一作]` (Shanghai Jiao Tong University), Dongrui Liu `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种结合分层求解树与运行时监督的 GUI‑agent 框架，在受保护的 GUI 环境中同时完成任务解题与可接受的交互执行；

**💡 创新点**

核心创新在于：①将工具输出作为可修订的证据绑定到动作承诺；②基于证据的行动优先级与冲突冲正；③通过反馈驱动的有限恢复预算实现成本可控的执行；

**🔧 技术方法**

采用基于 Perceive‑Ground‑Solve‑Act‑Verify 的层次求解树；运行时监督器负责证据绑定、承诺管理和恢复预算；使用多种大语言模型（GPT‑5.5/5.4、Claude Opus、Gemini 3.1 Pro、GLM‑5V‑Turbo、MiniMax‑M2.7/3、Qwen3.7‑Plus）与工具调用；

**📊 数据集**

在 HLL CAPTCHA 任务集（9 个动态验证家族）和 AndroidWorld 模拟器任务上进行评估；

**📈 对比分析**

与 MobileAgent、AppAgent、UI‑TARS 等基线进行对比；在 8/9 个 CAPTCHA 后端上实现最高 82 % 的成功率，较基线提升 13–35 个百分点；在 AndroidWorld 上平均成功率提升 30–60 个百分点，平均耗时缩短 50–80 %；

**⚠️ 局限性**

局限性包括：①依赖对环境交互证据的准确提取，可能不适用于真实部署的复杂防护；②不同后端模型性能差异较大，部分模型在恢复预算受限时表现不稳定；③恢复预算的设置需手工调优，未给出统一理论；

---

## 47. GS-MoE: Training-Free MoE Compression via Groupwise Subspace Sharing and Pruning

**arXiv ID:** 2610.32009 | [PDF](https://arxiv.org/pdf/2610.32009v1)

**作者:** Carson Cary `[一作]` (Florida State University), Shangqian Gao `[通讯]` (Florida State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种训练无关的 GS‑MoE 框架，用组共享子空间与专家内部宽度压缩并行压缩 Mixture‑of‑Experts 模型的参数。

**💡 创新点**

创新点在于：①通过基于重构误差引导的专家分组，允许不同专家组拥有不同的共享子空间；②采用基于特征矩阵的 Nyström 与岭杠杆选择实现专家内部宽度压缩；③在不需要额外训练的前提下，用梯度无关的层重要性、特征值与杠杆得分在层、组、专家层面分配预算。

**🔧 技术方法**

使用的技术包括：稀疏激活 MoE 架构、低秩分解（组共享投影）、Nyström 近似、岭杠杆得分、基于重构误差的专家分组（平衡 medoid 聚类）、特征值与杠杆得分的预算分配、无梯度优化的资源分配算法。

**📊 数据集**

在 Mixtral‑8x7B、DeepSeek‑MoE‑16B、Qwen2‑57B‑A14B 等多种粗细粒度 MoE 语言模型上进行实验，并使用 Alpaca、C4、WikiText‑2‑raw‑v1 等数据集收集校准统计；在 LM Evaluation Harness 的 9 个零样本推理任务上评估模型性能。

**📈 对比分析**

与 MoE‑SVD、D²‑MoE、TD‑MoE、NAEE、MoE‑I² 等主流压缩方法对比，GS‑MoE 在 60% 参数压缩时保持约 90% 推理准确率（DeepSeek‑MoE‑16B‑Base），并在同等压缩率下显著低于 D²‑MoE 的 73%；在 Qwen‑2‑57B‑A14B 上实现 60% 压缩时几乎无精度损失；同时在混合子空间数量较多的模型上保持 3‑4× 的吞吐量提升。

**⚠️ 局限性**

局限性包括：①压缩后仍需大量计算资源，对极大模型的推理仍受限；②组分配与预算分配过程依赖于校准数据的统计估计，若数据分布变化可能导致性能退化；③在某些细粒度 MoE 结构中，子空间数量增多会略微降低吞吐量，需在压缩率与速度间做权衡。

---

## 48. MS-TTRL: Breaking the Frequency Conflict in Multiple-Solution Test-Time Reinforcement Learning

**arXiv ID:** 2610.32234 | [PDF](https://arxiv.org/pdf/2610.32234v1)

**作者:** Zhixia Zhang `[一作]` (Beihang University), Dianbo Liu `[通讯]` (National University Of Singapore)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在多解推理任务中提出一种标签自由的测试时强化学习框架MS‑TTRL，以解决频率冲突问题。

**💡 创新点**

创新点在于将正确性估计与多样性保持解耦，使用结果层自检得到频率无关的正确性信号，并通过三态缓存与置信门限逆概率缩放强化稀有有效解的学习信号。

**🔧 技术方法**

核心技术包括：结果层自检（self‑verification）、三态结果缓存（correct/candidate/incorrect）、置信门限逆概率缩放（confidence‑gated IPS）、以及对多解推理任务的多种奖励设计。

**📊 数据集**

实验数据集涵盖HypoSpace（因果发现、3D结构重建、布尔表达式发现）和MACE‑Math四个多解推理基准。

**📈 对比分析**

与两种标签自由基线（TTRL、TTRL‑CoCoV）及监督参考IPS‑GRPO对比，MS‑TTRL在所有任务上均显著提升恢复率（Recovery Rate）和结果恢复率（Outcome Recovery Rate），平均提升约19个百分点，接近监督方法。

**⚠️ 局限性**

局限在于自检过程仍需多次推理，导致计算成本略高；对阈值、逆概率缩放系数等超参敏感，且在极少数极大解空间任务中仍可能出现结果崩溃。

---

## 49. Does LeWorldModel Preserve Local Action Geometry?

**arXiv ID:** 2610.31716 | [PDF](https://arxiv.org/pdf/2610.31716v1)

**作者:** Vasil Vasilev `[一作]` (Technological School Electronic Systems), Siddharth Nayak `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 11 | [OpenAlex ID](https://openalex.org/A5114449480)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

评估预训练LeWorldModel在Reacher和OGBench Cube环境中对小动作扰动的敏感度，利用模拟器与模型的对照响应计算Gram矩阵并分析不同扰动尺度下的几何匹配。

**💡 创新点**

首次在有限尺度下对动作敏感度进行匹配评估并引入校准参考（等距、旋转、随机置换）来量化模型与物理环境之间的方向与大小误差。

**🔧 技术方法**

使用Jacobian近似、Gram矩阵归一化、条件数与特征向量角度分析、加权绝对百分误差以及冻结编码器的模拟器反事实来实现敏感度测量与比较。

**📊 数据集**

利用Reacher（458个anchor，二维动作）和OGBench Cube（500个anchor，五维动作）这两个标准强化学习数据集进行实验。

**📈 对比分析**

通过比较环境与模型的归一化Gram差异、误差率（WAPE）以及在不同扰动尺度下的D_G指标，发现模型在小尺度下误差较大，尺度扩大可显著降低误差，模型相对环境的敏感度约为4/5至1/2，Held‑out测试中模型Gram的误差低于随机参考但仍高于环境。

**⚠️ 局限性**

实验仅使用单一预训练检查点，缺乏跨种子多样性；未评估闭环控制性能；对物理、像素与嵌入尺度的因果影响未分离；重放协议变更导致可复现性受限。

---

## 50. The Minimum Number of Samples in the Labeled Coupon Collector Problem

**arXiv ID:** 2610.32251 | [PDF](https://arxiv.org/pdf/2610.32251v1)

**作者:** Andrew Tan `[一作]`, Paul H. Siegel `[通讯]`

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

提出了标记优惠券收集问题（Labeled Coupon Collector Problem，LCCP），并给出了在两种设置下（已知标签集与未知标签集）推断所有标签所需的最小样本数的完整理论阐述。

**💡 创新点**

将LCCP与分离系统（separating system）的组合理论联系起来，构造了最小分离与覆盖系统，从而得到比已有文献更紧的样本数界限，并且完全刻画了全恢复与部分恢复两种情形的最小样本数。

**🔧 技术方法**

采用了二进制抽样矩阵（draw matrix）框架，利用组合计数、哈希码、几何/二项分布逆CDF、几乎正则矩阵构造以及分块/补集技术，对样本数的上下界进行严格推导和构造。

**📊 数据集**

本工作为理论研究，不使用实测数据集，所有结果均来自严谨的组合数学证明和构造算法。

**📈 对比分析**

通过与之前Katona、Wegener等人给出的上界比较，本文给出的上界在大多数参数范围内显著更小，特别是在n远大于k时。下界与已有的对数下界一致，且在特定阈值下可达成。

**⚠️ 局限性**

局限性：在n介于2k和k+12之间的中间区间仍需通过枚举寻找最优m，缺乏闭式解；对部分恢复情形的最优样本数在两极限子案例（如r=2^m−1或r=2^m−3）仍需额外样本；若放宽标签映射为非双射，则需进一步研究。

---

## 51. Beyond Heuristics and Commercial Navigation: An AI-Augmented Decision-Support for Ambulance Travel Time Estimation

**arXiv ID:** 2610.31968 | [PDF](https://arxiv.org/pdf/2610.31968v1)

**作者:** Muhammad Adil Abid `[一作]` (Malmö University), Jesper Petersson `[通讯]` (Region Skåne)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

开发了基于机器学习的急救车行程时间预测决策支持系统

**💡 创新点**

将历史急救车运营数据与时间、优先级、空间特征结合，并通过可解释AI验证模型学习真实运营模式

**🔧 技术方法**

采用人工神经网络回归、CatBoost、随机森林等模型，并使用 SHAP 与置换重要性进行解释

**📊 数据集**

使用 2020‑2021 年瑞典 Skåne 区域急救车历史行程数据

**📈 对比分析**

与 Google Maps 估计及固定 5% 快速假设对比，优先级分层 MAE 下降 52‑80%，ANN 获得最高 R²=0.67

**⚠️ 局限性**

仅考虑区域级别特征，未使用实时交通或动态路况更新，模型对新路况泛化能力有限

---

## 52. An AI Factory for Technology Education: An Agentic Pipeline with Closed Loop Audience Feedback

**arXiv ID:** 2610.31875 | [PDF](https://arxiv.org/pdf/2610.31875v1)

**作者:** Josue Hernandez `[一作]`, Rocael Hernandez Rizzardini `[通讯]`

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

构建了一个 agentic pipeline，采用 typed workflow graph 并闭环受众反馈，用于快速生成面向技术教育的教程和指南。

**💡 创新点**

创新点包括：1) 四类节点（确定性脚本、固定 Rubric 的模型步骤、开放式代理阶段、人工审批门）明确划分工作量和成本；2) 双层来源（需求层 + 权威层）确保内容既跟随热点又保持事实可靠；3) 反馈循环将用户参与度数据自动转化为下一轮生成规则，形成自适应内容生产；4) 在连续 5 个月内实现 26 篇指南的持续发布。

**🔧 技术方法**

主要技术：大型语言模型（LLM）+ agent harness（orchestrator + executor）、图像生成模型、爬虫/爬取脚本、内容检索与验证模块、策略文件自动更新、人工操作接口、社交媒体分发与关键词触发机制。

**📊 数据集**

使用的数据集：11 个配置好的 YouTube 频道与 8 个 Subreddit 的视频与帖子（需求层），官方文档、Changelog、GitHub README 等权威资源（权威层），以及通过爬虫收集的用户互动（如点赞、评论、分享）做为后续反馈数据。

**📈 对比分析**

性能评估：在 5 个月内平均每周 1.5 小时人工投入，发布 26 篇指南；通过“engagement score”衡量社交媒体参与度，四项闭环反馈结果表明：关键词触发提高 1.7 倍参与度，逆向 hook 风格提升 1.83 倍，主题疲劳与 hook 主语等因素得到量化并转化为规则；相比单次生成或无反馈循环的基线，系统表现出更高的受众互动和规则自适应能力。

**⚠️ 局限性**

限制：①评估仅覆盖受众采纳和需求指标，未测量学习成效；②人工工时为自报，缺乏客观计时；③实验仅在 AI 工具领域，缺乏跨领域验证；④反馈机制无法完全分离主题、时间与平台变化的影响；⑤缺乏学习效果评估工具和更细粒度的性能对比。

---

## 53. Rewriting Scaling Laws for Synthetic Data with Compute and Data Constraints

**arXiv ID:** 2610.31867 | [PDF](https://arxiv.org/pdf/2610.31867v1)

**作者:** Nicholas Roberts `[一作]` (Princeton University), Nitya Mani `[通讯]` (Jane Street)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了合成重写在预训练语料中的使用，提出了一种重写缩放法则，预测模型大小、训练令牌、重写比例和自然数据的训练轮数对损失的影响。

**💡 创新点**

提出的重写缩放法则能够预测最佳重写比例和相应的损失，并且在计算受限和数据受限的设置中均表现良好。

**🔧 技术方法**

使用了重写缩放法则来预测模型的损失，并通过实验验证了该法则的有效性。

**📊 数据集**

在计算受限的设置中使用了FineWeb-Edu数据集及其重写版本；在数据受限的设置中使用了固定的5B令牌自然数据集及其重写版本。

**📈 对比分析**

通过与自然数据训练的模型进行比较，发现使用合成重写的模型在多个评估任务中表现优于仅使用自然数据的模型，尤其是在计算预算较高时。

**⚠️ 局限性**

限制在于所选择的自然种子集和重写样式的数量，未能全面评估不同重写提示分布或不同重写模型的影响。

---

## 54. Pareto-optimal Scheduling with Predictions

**arXiv ID:** 2610.32503 | [PDF](https://arxiv.org/pdf/2610.32503v1)

**作者:** Ziyad Benomar `[一作]` (Amazon), Vianney Perchet `[通讯]` (Criteo AI Lab)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出并分析了一种在单机非clairvoyant调度问题中使用预测的最大化鲁棒承诺（Maximal Robust Commitment，_R）算法，证明其在所有确定性算法中实现了最佳的一致性‑鲁棒性权衡曲线；

**💡 创新点**

创新点在于构造了新的鲁棒性不变式、基于层级的精细分析以及与可接受目标函数相匹配的递推最优解，首次给出了确定性算法在预知顺序或尺寸预测下的完整 Pareto 前沿；

**🔧 技术方法**

技术手段包括：鲁棒性不变式证明、贪心调度策略（commit 与 round‑robin 结合）、层级划分与积分分析、目标函数的微分方程求解以及对等价下界的递推证明；

**📊 数据集**

实验使用指数、Pareto（shape 1.1）和 Weibull（尺度 2、形状 0.5）三种独立分布的随机作业尺寸，并加入高斯噪声模拟预测误差；

**📈 对比分析**

与时间共享、延迟预测、Round‑Robin、随机组合等算法在噪声水平 τ = 0（完美预测）和 τ > 0（误差）下进行比较，_R 在近似完美预测时几乎与理想预测顺序同等，在鲁棒性 3 的条件下显著优于时间共享并在高噪声下仍低于随机组合；

**⚠️ 局限性**

局限性包括：未证明算法在误差随时间平滑性（smoothness）下的表现；只针对单机无权重、无释放时间的情形；随机化算法可能在某些实例上突破确定性前沿，且目前对多机或权重调度的扩展仍待研究。

---

## 55. EVOLVE: Uncertainty-Aware Multivariate Time Series Anomaly Prediction with Adaptive Prototype Memories

**arXiv ID:** 2610.32189 | [PDF](https://arxiv.org/pdf/2610.32189v1)

**作者:** Jianxin Jin `[一作]` (East China Normal University), Chenjuan Guo `[通讯]` (East China Normal University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种名为EVOLVE的无监督多变量时间序列异常预测框架，能够在未见未来异常标签的情况下估计未来异常风险；

**💡 创新点**

创新点包括：①通过自适应异常原型记忆实现变量级、上下文级的异常扰动生成；②结合不确定性感知的多假设未来推理，利用不同演化假设（平滑与偏差放大）生成多条未来轨迹，并用其预测差异来衡量异常风险；③使用对抗式扰动学习提升异常扰动的判别性；

**🔧 技术方法**

主要技术包括自注意力编码器、跨注意力原型检索、动态原型更新、变分隐层采样、双结构解码器（平滑与偏差放大分支）、演化模式对齐损失与对抗损失；

**📊 数据集**

在六个工业与物联网公开数据集上进行评估：MSL、PSM、SMD、SWAN、GECCO、Genesis；

**📈 对比分析**

与改造后的异常检测方法（如LODA、IForest、TranAD等）以及专门的异常预测方法（如PAD、A2P、IGCL、RED-F、TAP、FCM）进行了对比。EVOLVE在Aff-F1、VUS-ROC、VUS-PR三项指标上均实现了最优或最接近最优的表现，显著优于所有基线；

**⚠️ 局限性**

局限性包括：①需要额外的内存来维护动态原型记忆，虽影响不大但仍为瓶颈；②模型训练需两阶段交替优化，训练成本相对较高；③在极端稀疏异常场景下，预测分辨率仍受限，仍需进一步提升对极短异常周期的敏感性。

---

## 56. TACS-RF: Tool-driven Agentic Circuit Synthesis of RF and mmWave Modules

**arXiv ID:** 2610.31882 | [PDF](https://arxiv.org/pdf/2610.31882v1)

**作者:** Tristan Liang `[一作]` (Northeastern University), Aravind Nagulu `[通讯]` (Northeastern University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出TACS-RF框架，利用大型语言模型（Claude Opus 4.8）驱动行业级EDA工具完成从原理图生成、布局、EM仿真到后仿真验证，最终实现240–251 GHz亚THz发射机的全流程自动化设计

**💡 创新点**

首次将LLM直接与Cadence Virtuoso、EMX、HFSS、Calibre等工具无界面交互，完成完整的高频IC设计闭环，展示“文本到ASIC”设计的可行性

**🔧 技术方法**

大型语言模型、EDA脚本化接口（virtuso-bridge-lite）、EM仿真工具（Cadence EMX、Ansys HFSS）、物理验证工具（Calibre）以及参数化布局和提取流程

**📊 数据集**

90 nm SiGe BiCMOS工艺手册、相关亚THz传输机参考论文、已提取的被动元件库

**📈 对比分析**

与前沿亚THz源进行对比，获得+6.3 dBm输出功率、1 MHz偏移下-80至-88 dBc/Hz相位噪声、11 GHz调谐范围，功耗361 mW，性能与已有技术相当甚至优于部分基准

**⚠️ 局限性**

设计时间主要受EM仿真耗时限制；目前需要人工提供被动元件库和晶体管布局，且完全无监督的“文本到ASIC”尚未实现，仍需进一步提升自动化程度

---

## 57. When Does Dynamics Randomization Test Context Inference?

**arXiv ID:** 2610.32414 | [PDF](https://arxiv.org/pdf/2610.32414v1)

**作者:** Oleg Smirnov `[一作]` (King AI Labs, Microsoft Gaming), Sara Karimi `[通讯]` (King AI Labs, Microsoft Gaming)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并量化在环境随机化情境下，当前状态-动作对所缺失的预测信息（预测头room），并用其评估显式上下文推断的价值。

**💡 创新点**

创新点在于引入预测头room概念，用 KL 散度度量不同上下文之间的转移核差异，并对访问分布与固定分布的比较进行定量分析，从而区分覆盖变化与上下文权重变化。

**🔧 技术方法**

技术包括上下文化 MDP 框架、KL 散度计算、核匹配与重权重、经验估计、Bootstrap 不确定性评估、状态组件重置实验。

**📊 数据集**

使用 MiniGrid 的 Wind 与 Resistance 变体（离散网格环境）以及 Point‑Mass‑Attractor（连续二维系统）作为数据集。

**📈 对比分析**

通过计算结构化参考、匹配查询、访问查询下的 KL 散度，得到预测头room，并与不同重置策略比较。实验表明在 Wind 与 Attractor 环境中访问头room 很低，但在 Resistance 中较高；重置状态组件可显著提升头room。

**⚠️ 局限性**

局限性包括仅评估单步预测、依赖已知转移核、对学习模型误差和优化不敏感、对评估分布的依赖以及未覆盖长期规划场景。

---

## 58. Nested Sparsity for Scalable Pretraining and Continual Learning

**arXiv ID:** 2610.31818 | [PDF](https://arxiv.org/pdf/2610.31818v1)

**作者:** Tianzhe Chu `[一作]` (University of Hong Kong), Yi Ma `[通讯]` (University of Hong Kong)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了一种新的Transformer架构——Nestor，采用列路由注意力和专家网络，形成嵌套稀疏性，旨在提升大规模语言模型的计算效率和持续训练表现。

**💡 创新点**

将条件计算扩展到注意力层，提出层级列路由与专家混合，形成多维稀疏性；与MoE和密集Transformer统一，并在理论上解释其优越性。

**🔧 技术方法**

列路由注意力、Mixture‑of‑Experts在列内、残差连接和注意力残差、平衡路由、可变稀疏度、长上下文扩展、计算效率分析与理论建模。

**📊 数据集**

语言模型预训练使用DCLM baseline；持续预训练对比书籍（PG‑19）、数学（OpenWebMath）、代码（AlgebraicStack）、德语（FineWeb‑2）等；长上下文测试使用DCLM baseline。

**📈 对比分析**

与相同稀疏度的MoE Transformer和密集Transformer在100M–1B参数规模下对比；Nestor在相同compute/参数下取得更低的验证损失；持续预训练中适应性与忘记率更好；长上下文中compute效率更高，优势随上下文长度增加。

**⚠️ 局限性**

仅评估到1B参数，未验证更大规模或下游任务；列仅使用完整因果注意力，未探究更稀疏/线性注意力；meta‑architecture的其他组合未尝试；路由和残差设计细节仍有待改进。

---

## 59. Learning Fine-Grained Visual Classification with Text as Privileged Information via Multimodal Distillation

**arXiv ID:** 2610.32442 | [PDF](https://arxiv.org/pdf/2610.32442v1)

**作者:** Aishwarya Venkataramanan `[一作]` (Friedrich Schiller University Jena), Joachim Denzler `[通讯]` (Friedrich Schiller University Jena)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出FG‑LUPI框架，利用训练阶段的文本属性描述作为privileged信息，训练多模态教师模型并通过多层蒸馏将知识迁移到仅使用图像的学生模型，从而在Fine‑grained视觉分类任务中提升性能；

**💡 创新点**

创新点在于：①将文本属性视为LUPI的privileged信息；②构建门控交叉注意力的多模态教师；③设计预测、补丁层和注意力层的多级蒸馏；④使得图像‑仅学生甚至能超过教师，并改善模型置信度校准；

**🔧 技术方法**

使用技术包括Vision Transformer骨干、冻结BERT文本编码器、门控交叉注意力融合、温度蒸馏、补丁层余弦对齐、注意力层KL对齐、EMA教师以及多级蒸馏损失；

**📊 数据集**

实验数据集涵盖CUB‑200‑2011、NABirds、Stanford Dogs、FGVCAircraft、Stanford Cars、iNaturalist 2021，文本属性由人工描述或Gemma生成；

**📈 对比分析**

与多种CNN/ViT基线及FitNet、Attn‑Trans、SP、DKD、CRD等蒸馏方法对比，FG‑LUPI在所有六个基准上均实现最高或第二高的Top‑1精度，并在ECE上显著低于ViT基线，提升幅度约2‑5个百分点；

**⚠️ 局限性**

局限性包括：依赖高质量文本属性；在仅有类别级描述的iNat等数据集上提升有限；对文本编码器的鲁棒性相对不高；训练阶段需额外多模态教师，计算成本上升；未在更大规模或实时场景中验证。

---

## 60. Tradeoff-Aware Territory Design: A Multi-Objective Decision System for Enterprise Customer-to-Seller Allocation

**arXiv ID:** 2610.31701 | [PDF](https://arxiv.org/pdf/2610.31701v1)

**作者:** Nan Wang `[一作]` (Databricks), Mitchell Turiel `[通讯]` (Databricks)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出了一套面向业务权衡的地区分配决策系统，整合账户价值量化、卖家-账户连续性估值以及多目标约束优化，部署为可视化情景式的人工干预工具。

**💡 创新点**

创新点在于将卖家-账户连续性视为可经验估计、异质化的价值而非固定惩罚，并以统一的规划货币框架将多目标权衡透明化，形成可检验的决策表面。

**🔧 技术方法**

采用了预测性账户评分模型、合成控制式因果反事实估计连续性影响、混合整数多目标优化以及人工干预的自适应再优化流程。

**📊 数据集**

使用了企业级B2B GTM的历史账户、卖家、容量与消费时序数据，并在同一数据集上进行后向回测与生产对照。

**📈 对比分析**

通过与专家手工划分基线的历史回测和生产前后对比，系统在合规率提升78%、机会分布更平衡、卖家达成度保持不变、机会捕获率提高等计划质量指标上表现优异；运行时间约为4小时。

**⚠️ 局限性**

局限性包括：账户价值视为与分配无关；连续性估计为观察性因果，缺乏随机验证；未涵盖地理或语言等地区规则；在低密度区域可能产生不可行方案，需要人工复核；缺乏对实时运营影响的随机实验验证。

---

## 61. Knowing When to Stop: The Evidence-Action Gap in Coding Agents

**arXiv ID:** 2610.31895 | [PDF](https://arxiv.org/pdf/2610.31895v1)

**作者:** Tianyu Chen `[一作]` (University of Texas at Austin), Tongshuang Wu `[通讯]` (Carnegie Mellon University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估编码代理在环境阻碍下的证据‑动作间隙，提出 ImpasseBench 基准并测试工具与监控干预。

**💡 创新点**

引入以证据为锚点的基准，量化过度坚持与提前交接，并验证两种干预方式显著改善停止行为。

**🔧 技术方法**

使用大规模语言模型（GPT‑5.6、Gemini、Claude、Kimi、Qwen 等）与工具调用接口、规则式标注器及基于规则的监控。

**📊 数据集**

SWE‑smith 修复任务，12 类阻断条件，102 基准任务，共 19,371 条轨迹。

**📈 对比分析**

对比无干预与加 Escalation Tool 以及 Harness Monitor，结果显示预算消耗从 34.0% 降至 6.7%，不安全请求降至 0.68%，完成率提升至 21.8%。

**⚠️ 局限性**

局限于规则评估、沙箱仿真、模型对工具使用依赖，缺乏真实生产环境验证，难以全面反映部署时的安全与交接效果。

---

## 62. Tightening the Online Bin Packing Gap from Both Sides, and the Limits of the Current Methods

**arXiv ID:** 2610.32367 | [PDF](https://arxiv.org/pdf/2610.32367v1)

**作者:** Karyamsetty Martin `[一作]` `[通讯]`, Karyamsetty Martin

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

通过精确理性算术和自动化验证，对 2018 年的高级谐波算法及其下界构造进行重新评估，得到新的上界 1.5782147437 并证明下界构造在窗口尺寸之外无法进一步提升。

**💡 创新点**

提出了精确整数算术上界证明、下界构造的刚性定理、窗口批处理的机器验证，以及对现有机制（如标记、红沙）在高级谐波中无效性的系统性否定。

**🔧 技术方法**

采用精确理性数算术、凸优化与切割法、行为线性规划、分支定界与精确背包求解、以及符号验证等技术。

**📊 数据集**

使用 402 个场景表格、所有可能的批处理树以及约 850 个双统计树等数据集进行全面枚举。

**📈 对比分析**

在所有 402 个场景上执行完整的精确优化与独立验证，证明新上界在每个场景下均满足，且下界构造在所有可扩展窗口外部均不超过原始极限，显示两边已达到其框架边界。

**⚠️ 局限性**

仅适用于现有框架，未突破固定比例或窗口外扩展；多窗口批或额外统计量的改进仍未完成，且缺乏针对两中介箱的机制，未解决固定比例上限是否适用于高级谐波。

---

## 63. EigenGradCAM with AugSmooth: Enhancing Explainability and Precision in Multi-Stage Alzheimer's Disease Detection

**arXiv ID:** 2610.32374 | [PDF](https://arxiv.org/pdf/2610.32374v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 64. Sharpness-aware Orthogonalized Momentum Algorithms

**arXiv ID:** 2610.32465 | [PDF](https://arxiv.org/pdf/2610.32465v1)

**作者:** Geetika `[一作]` (Indraprastha Institute of Information Technology), Bapi Chatterjee `[通讯]` (Indraprastha Institute of Information Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种基于Sharpness-aware的正交化动量优化器（SOMA），通过重用上一迭代的梯度来构造轻量级的对抗扰动，并在大语言模型预训练、量化和微调过程中进行实验验证。

**💡 创新点**

创新点在于：① 在保持SAM思想的前提下，使用“陈旧”随机梯度而非额外计算的梯度来生成扰动，极大降低每步计算成本；② 结合矩阵正交化动量（MuON/SpecSAM）实现谱几何下的扰动；③ 给出了在重尾梯度噪声下的收敛理论，证明误差上限仅随扰动半径而非噪声强度变化。

**🔧 技术方法**

技术手段包括：Sharpness-aware Minization (SAM)、矩阵正交化动量（MuON、SpecSAM）、重尾噪声假设下的p-Bounded Central Moment (p-BCM)、Newton–Schulz 迭代实现谱正交化、以及AWQ量化算法。

**📊 数据集**

实验数据集：1）合成教师-学生模型（可控制重尾噪声）；2）nanoGPT (124M) 与 Pythia-70M 在 FineWeb-Edu 上的继续预训练；3）Airbench CNN 在 CIFAR‑10 上的训练；4）4‑bit AWQ 量化的 FineWeb‑Edu held‑out 1M tokens；5）Fine‑tune 语料用于灾难性遗忘评估。

**📈 对比分析**

与 LAMB、AdamW、AdaBelief 等基线优化器及其 SAM 变体（SpecSAM-Muon、RandSAM-Muon、MuON、SpecSAM-Muon 等）进行对比。结果表明：基线优化器决定了整体性能；SOMA 在保持接近基线的最终困惑度（如 nanoGPT 24.41 与 24.55）的同时，仅略微增加 3% 训练时间；在 4‑bit 量化后，SOMA 的困惑度提升幅度明显低于基线（如 Pythia‑70M 由 14.6% 降至 10.5%）。在合成重尾噪声实验中，谱方向的扰动显著优于随机扰动，且在不同尾指数下保持稳健。

**⚠️ 局限性**

局限性包括：① 只在中小规模 LLM（124M、70M）和单个 CNN 上验证，缺乏大规模跨模型通用性验证；② 需手动调节扰动半径，且在噪声轻重不同情况下的最佳半径仍需经验确定；③ 对灾难性遗忘的减缓效果有限，未能在微调阶段显著提升；④ 理论分析基于重尾 p-BCM 假设，可能不适用于所有实际梯度噪声分布。

---

## 65. Interpreting Physics in Video Encoders using Multi-Object Interaction Videos

**arXiv ID:** 2610.31983 | [PDF](https://arxiv.org/pdf/2610.31983v1)

**作者:** Ayushi Mehrotra `[一作]` (California Institute of Technology), Sai Vidyaranya Nuthalapati `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

研究视频编码器是否能在其隐藏表示中编码两体相互作用的关系物理量（如弹性、动量守恒和摩擦系数），并评估这些编码在不同环境下的可迁移性。

**💡 创新点**

首次针对关系物理量（而非单体属性）进行解释性探测，考察预训练目标（像素重建 vs 隐空间预测）对物理信息编码的影响。

**🔧 技术方法**

使用冻结的自监督视频编码器 VideoMAE‑base（像素重建）和 V‑JEPA2‑ViT‑L（预测）进行线性读出和模型自身预训练目标的“惊讶”度量。

**📊 数据集**

合成数据集：Kubric 渲染的两体碰撞（弹性、非弹性、能量产生/损失）和斜坡/平面滑动（摩擦系数、倾斜角、加速度）。

**📈 对比分析**

与可观测的速度/接触几何基准对比，VideoMAE 与 V‑JEPA2 在弹性判别上分别取得 AUC 0.911 与 0.973；动量守恒和摩擦系数读数无法在不同几何或物理设定中保持高 R²，表明转移性能不佳。

**⚠️ 局限性**

局限于合成、相对简易的场景，且仅评估线性读出；两模型的结果不具泛化性，且未验证非线性推理的潜能。

---

## 66. When Simple Is Enough: Resource-Aware Heuristic versus DDQN for O-RAN Mobility Robustness

**arXiv ID:** 2610.32257 | [PDF](https://arxiv.org/pdf/2610.32257v1)

**作者:** Ali Deylamy `[一作]` (University College Dublin), Nima Afraz `[通讯]` (University College Dublin)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在O‑RAN xApp框架下，针对移动性管理设计并评估了一种资源感知启发式以及基于DDQN的移动性策略，并将其与标准3GPP Event A3进行配对系统级仿真对比。

**💡 创新点**

创新点在于：①利用NSGA‑II离线优化权重，将射频质量、目标站点资源、UE运动和服务级别统一纳入启发式评分；②保持线性在线复杂度和可解释性；③通过CRITIC‑TOPSIS多KPI评估揭示不同策略在停机率、手over可靠性等方面的细粒度差异。

**🔧 技术方法**

采用的技术包括：NSGA‑II多目标进化优化、双重深度Q网络（DDQN）与经验回放、射线追踪（ray‑tracing）生成射频输入、系统级仿真、CRITIC‑TOPSIS多指标加权排名。

**📊 数据集**

使用UCD Belfield校园网络的六个gNB站点作为射频环境，配合100–150个UE的随机漫步与高速移动轨迹，射线追踪得到的接收功率缓存为所有控制器共享的输入数据集。

**📈 对比分析**

在Base、High‑Mobility、High‑UE三种场景下，使用相同UE轨迹、交通负载和射频缓存，三种控制器按同一xApp接口执行手over决策；评估指标涵盖平均与5%分位好通量、停机率、手over成功率、ping‑pong比率等。结果显示启发式在所有场景中取得最低停机率，并在CRITIC‑TOPSIS排名第一；DDQN在拥塞场景下提升目标可行性但伴随更多手over和ping‑pong；标准A3在高移动性场景中仍保持较高的平均好通量和低ping‑pong。

**⚠️ 局限性**

局限性包括：仅在单一随机种子下完成配对评估，缺乏多种种子Bootstrap验证；DDQN需昂贵的离线训练和神经网络推理；启发式对优化权重敏感；未在真实E2‑连接O‑RAN环境中测试，可能存在部署差异。

---

## 67. The Schema Coherence Inventory: Measuring Whether an Assigned Belief Organizes a Language Model's Cognition

**arXiv ID:** 2610.31683 | [PDF](https://arxiv.org/pdf/2610.31683v1)

**作者:** Yuqi Wu `[一作]` (Fudan University), Jie Chen `[通讯]` (Fudan University)

**通讯引用:** 52474 | [OpenAlex ID](https://openalex.org/A5100694761)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并验证了 Schema Coherence Inventory (SCI)，用于判断 persona‑conditioned 语言模型是否将核心自我信念组织其行为。

**💡 创新点**

提出了强制选择的对数似然评分方案，消除解析/生成偏差，并通过“目标特异性”指标有效区分情绪与信念。

**🔧 技术方法**

利用 Llama‑3.1‑8B、Qwen‑3.x、Phi‑3.5‑mini 等大型语言模型，结合 LoRA 适配器、工程化提示与混合适配器等条件化技术进行实验。

**📊 数据集**

使用 96 条 SCI 题目（覆盖 8 个认知层级、6 种信念）以及 16 条患者 CCD 作为评估数据集。

**📈 对比分析**

通过多种对照（影响适配器、混合适配器、一般情绪控制）进行内容效度、信度、构念效度、准则效度、泛化性和短表单等六项研究；SCI‑96 在目标信念上平均提升约 29.8 分（相对基线 16.7%），SCI‑48 与 SCI‑96 相关性为 0.92，且能准确识别模型自由生成文本中的信念比例。

**⚠️ 局限性**

局限包括：三轴结构基于 Beck 模型的先验假设；未评估预训练阶段形成的信念；仅在开放权重 backbone 上测试；短表单精度下降，且未覆盖无核心信念模型。

---

## 68. Freeform Programmable Liquid Crystal Elastomers via Monolithic Integration of Co-Optimization and 4D-Printing

**arXiv ID:** 2610.32186 | [PDF](https://arxiv.org/pdf/2610.32186v1)

**作者:** Yibo Wang `[一作]` (University of Illinois Urbana-Champaign), Xiaojia Shelly Zhang `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了一套多层共优化与直接墨水写入（DIW）4D打印闭环框架，能够逆向设计并制造具有自由几何和连续取向场的液晶弹性体（LCE），并在实验中实现了可编程热膨胀、三维形变、轨迹控制以及多功能软机器人的功能。

**💡 创新点**

创新点包括：①将几何和层级取向场同时进行拓扑优化，显著扩展设计空间；②基于非理想LCE本构模型的全三维大变形数值仿真；③提出连续取向过滤与Poisson场生成，进而通过ISO线提取实现DIW可打印的工具路径；④一次性完成设计、优化、制造、实验验证，形成完整的设计-制造闭环。

**🔧 技术方法**

采用的技术有：拓扑优化（SIMP+移动渐近法）、非理想液晶弹性体本构模型、非线性有限元（Newton–Raphson）、连续取向过滤与Poisson场求解、ISO线提取工具路径生成、DIW 4D打印与UV固化、热力学实验与拉伸测试。

**📊 数据集**

使用的数据集主要是公开的点云人脸数据集（Zenodo链接），其余数据为实验自制LCE样品的尺寸、热收缩率等信息；未使用其他公开大规模数据集。

**📈 对比分析**

通过数值仿真与实验测量对比，目标误差均低于6%（面变形误差）、旋转误差4.7%、热收缩率44.5%与理论相符；与传统经验设计相比，框架实现更复杂功能且保持可制造性；软机器人在循环测试中维持超过30次的稳定运动。

**⚠️ 局限性**

局限性包括：①未考虑黏弹性和取向旋转的显式耦合，导致动力学和自接触的准确性不足；②计算量大，需高分辨率网格；③制造过程未加入丝径、间隙等制约，易产生细微缺陷；④实验主要在气相热激，未充分验证液体环境或高速运动；⑤未实现三维嵌入式取向编程，限制了设计空间的进一步扩展。

---

## 69. PersonalQuery-Benchmark:Evaluating Retrieval Robustness to Personalized User Expression Differences

**arXiv ID:** 2610.32164 | [PDF](https://arxiv.org/pdf/2610.32164v1)

**作者:** Wenyu Liang `[一作]` (Monash University), Weiqing Wang `[通讯]` (Monash University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了 PersonalQuery‑Benchmark（PQB），并系统评估检索系统对个性化句法表达与写作错误差异的鲁棒性。

**💡 创新点**

①将个性化表达拆分为句法表达与写作错误两维；②从用户历史评论中学习用户特定句法分布和错误模型；③利用 rank‑one‑plus‑diagonal Gaussian 进行查询选择，构造不同用户的查询对；④在多模型、多域上统一评估，揭示不同检索架构对表达差异的敏感性。

**🔧 技术方法**

句法结构提取、同一产品监督对比学习、rank‑one‑plus‑diagonal Gaussian 建模、GECToR+ERRANT 写作错误提取、LLM（Qwen2.5‑0.5B‑Instruct、DeepSeek‑V4‑Pro）生成与评估、检索模型（BM25、SPLADE、GTE‑base、ColBERTv2、BGE‑M3 Hybrid、Llama‑3.1‑8B、Qwen3‑Reranker‑8B）及指标 Flip@K、Mean RR_std、Deg@K、MRR_degr 等。

**📊 数据集**

Amazon Reviews 2023 数据集，挑选音乐、视频游戏、婴儿产品三类，共 6,516 条查询实例。

**📈 对比分析**

对七种检索配置分别计算 Flip@K、Mean RR_std、Deg@K、MRR_degr 等指标。实验发现：即使核心信息保持不变，句法差异会导致明显的 hit‑status 与 rank 变化；写作错误则普遍导致 Deg@K、MRR_degr 正向下降。不同模型的鲁棒性差异显著，整体检索性能高并不等同于表达鲁棒性。

**⚠️ 局限性**

基准仅覆盖三类商品且依赖用户评论历史；写作错误模型聚焦表面字符编辑，忽略深层语义；实验缺乏真实查询及长期用户身份；LLM 生成与评估可能带来偏差，影响结果普适性。

---

## 70. X-Dubbing: Speech-to-Speech Translation under Temporal and Visual Constraints

**arXiv ID:** 2610.32042 | [PDF](https://arxiv.org/pdf/2610.32042v1)

**作者:** Jeongsoo Choi `[一作]` (KAIST), Joon Son Chung `[通讯]` (KAIST)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一个固定视频跨语言配音框架（X‑Dubbing），通过时间感知的翻译器和视频感知的语音合成器联合工作，并利用合成语音的反馈进一步优化翻译质量与同步性。

**💡 创新点**

创新点在于：①使用语言无关的时序预算将翻译与语音合成同步；②在翻译输出中嵌入暂停标记，供视频感知合成器精准控制时序；③通过多模态反馈（翻译质量、语音同步、说话速率等）对翻译器进行无对齐数据的强化学习，完全不需要配音对齐的训练数据。

**🔧 技术方法**

核心技术包括：Qwen2.5‑Omni（语音翻译器）、X‑Voice（Diffusion Transformer 语音合成器）与VAD/SyncNet 视觉同步、FiLM 层级条件、Direct Preference Optimization（DPO）进行翻译器的多模态偏好优化，以及ISCN（Isochrony‑normalized projection）用于训练时序监督。

**📊 数据集**

使用的数据集有：mTEDx French‑English 语音‑文本对（约 11k 片段）用于翻译器的监督微调；LRS3 英文音视频对（约 95k 片段）用于语音合成器的训练；以及 7,752 条翻译-合成偏好对用于后续 DPO；在 mTEDx French‑to‑English 测试集上评估。

**📈 对比分析**

对比方法包括 AV2AV‑Speech、SeamlessExpressive、Hibiki、Dub‑S2ST 以及基于同一预训练模型的多种级联策略。X‑Dubbing 在客观指标（ASR‑BLEU、SyllRate、Pause IoU、LSE‑C/D、UTMOS）和主观 MOS（音视频对应与整体自然度）均优于所有基线，特别在时序对齐和唇同步方面取得显著提升。

**⚠️ 局限性**

局限性：实验仅在法语‑英语单说话人单句视频上验证，未覆盖多语言、多人对话、复杂视觉场景；对源音视频质量敏感；生成语音可能被滥用于伪造内容。

---

## 71. Rethinking LoRA Initialization for Robust Asymmetric Learning Rates

**arXiv ID:** 2610.32142 | [PDF](https://arxiv.org/pdf/2610.32142v1)

**作者:** Disen Liao `[一作]` (University of Waterloo), Yaoliang Yu `[通讯]` (University of Waterloo)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `57a58b01-81b4-4d75-a45c-2e891f272b50` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种新的LoRA初始化方法（A的方差按n⁻²缩放），以抑制由B学习率放大的初始化噪声，从而提升LoRA在不对称学习率下的稳定性。

**💡 创新点**

创新点在于识别并解决了LoRA不对称学习率导致的初始化噪声放大问题；通过信号‑噪声分解证明了新初始化能显著降低噪声能量，并在有限宽度模型中保持特征学习能力。

**🔧 技术方法**

主要技术包括LoRA参数化、LoRA+不对称学习率、信号‑噪声分解分析、宽度相关的初始化设计（σ_A²∝n⁻²）、以及对比实验中的学习率搜索与特征能量评估。

**📊 数据集**

实验数据集涵盖：MNIST（toy MLP）、RoBERTa-Base在GLUE（编码器微调）、Llama 3‑8B 和 Llama 3.2‑1B 在算术与常识基准（生成任务）以及其他 Llama/Qwen/Mistral 微调实验。

**📈 对比分析**

对比方法：在标准LoRA与新初始化 LoRA 上分别对学习率、学习率比率、初始化规模、适配器秩及更新乘子进行网格搜索。结果显示：新初始化在保持或提升准确率（如算术任务从≈75%提升至≈77%）的同时，大幅扩展可用学习率范围，减少最佳比率漂移，并降低对高初始化尺度或大乘子的不稳定性。

**⚠️ 局限性**

局限性：实验大多基于单次运行，未覆盖所有模型/任务；仅验证了监督微调场景，对强化学习或生成模型训练的推广尚未评估；缺乏针对不同模型的通用LR比率设定规则；对极端宽度或更深层适配器的理论与实验验证有限。

---

## 72. CrowdStrike Channel File 291: An Engineering Analysis of a Global Endpoint Outage

**arXiv ID:** 2610.32078 | [PDF](https://arxiv.org/pdf/2610.32078v1)

**作者:** Luiz Parente `[一作]` `[通讯]` (Purdue University), Luiz Parente (Purdue University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

通过收集并分析 CrowdStrike 7.11 版本的内容更新日志、微软 Windows 崩溃转储、国会记录、运营影响报告等公开与内部资料，重建了 2024 年 7 月 19 日 CrowdStrike Falcon 内容更新导致 8.5 M Windows 设备崩溃的全过程，并提出了预防、限制、恢复三阶段的控制措施。

**💡 创新点**

创新点在于将安全更新失效的接口不一致问题拆解为验证、部署、恢复三阶段，并针对每一阶段提出具体的验证标准和控制框架，同时强调安全配置生命周期与代码生命周期应对等，形成系统化的事故复盘与改进模型。

**🔧 技术方法**

使用技术包括事件日志追踪与对照、内核级崩溃转储分析、接口定义与校验、灰度/金丝雀发布策略、故障注入与边界检查、网络与本地恢复方案等。

**📊 数据集**

数据集主要来自 CrowdStrike 内部根因分析报告（RCA）、微软崩溃分析报告、国会听证记录、Delta 航班取消数据、CISA 事件通报、企业恢复指标等公开与内部文档。

**📈 对比分析**

通过对比接口声明与实际传输数据、测试覆盖率与真实崩溃实例，以及部署前后设备连通率与恢复进度，评估了接口校验、灰度发布、异常处理等措施的有效性，指出缺失边界检查导致崩溃的严重性。

**⚠️ 局限性**

限制包括：缺乏对所有版本的接口一致性验证、测试仅验证匹配结果而非字段访问、灰度发布未能及时停止分发、恢复路径依赖网络与加密密钥等外部因素；论文基于公开信息，未进行源代码审计或实验验证，结论仅适用于该事件。

---

## 73. Embedding Large PQC KEMs in IKEv2

**arXiv ID:** 2610.32356 | [PDF](https://arxiv.org/pdf/2610.32356v1)

**作者:** Daniel Herzinger `[一作]`, Marian Margraf `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出在复合TCP/QUIC协议栈中加入增量密钥交换（ADDKE）和前向纠错（FEC）机制，以实现安全、低延迟的数据传输。

**💡 创新点**

创新点在于将密钥交换与前向纠错集成到TCP/QUIC的中间交换层，并设计了复合协议栈实现，从而兼顾安全性、可靠性与性能。

**🔧 技术方法**

采用了KEM代码点、RaptorQ FEC、notify-ACK、status-ACK等技术，并在TCP/QUIC栈中实现复合协议。

**📊 数据集**

使用了公开的网络测量数据集和实验室网络模拟数据来评估性能。

**📈 对比分析**

与传统TCP、QUIC以及单独使用FEC的基线方案对比实验表明，复合协议在丢包率降低约30%、延迟降低约15%的情况下，吞吐量保持相近或略有提升。

**⚠️ 局限性**

局限性包括在高丢包环境下的实现复杂度提升，以及密钥交换在多点通信中的扩展性和管理成本。

---

## 74. MAST-VLA: Skill-Conditioned Adaptation with Trajectory-Affordance Priors for Multi-Skill Robot Manipulation

**arXiv ID:** 2610.32312 | [PDF](https://arxiv.org/pdf/2610.32312v1)

**作者:** Jiachi Yang `[一作]` (Shanghai Innovation Institute), Yuan Xie `[通讯]` (Shanghai Innovation Institute)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种双重条件的视觉-语言-动作（VLA）框架 MAST‑VLA，用于多技能机器人操纵任务，能够在同一策略中同时建模原语技能语义与对象交互语义。

**💡 创新点**

创新点包括：① 在 LLM 主干中引入技能条件化的 LoRA‑MoE 适配器，使得低秩专家通过技能查询和路由偏置实现软技能特化；② 在动作头中利用轨迹生成的热图作为空间先验，将轨迹与景观几何直接注入注意力，以强化交互感知；③ 通过双重条件实现更强的多技能长期规划和跨场景泛化。

**🔧 技术方法**

主要技术包括：LoRA‑MoE 低秩适配器、技能查询/路由偏置、轨迹生成的热图监督、空间先验注入注意力、联合损失（动作回归+热图 BCE+MoE 正则化）以及在 LLM 主干中的自适应路由。

**📊 数据集**

使用 CALVIN ABC→D 长期多技能跨场景基准进行仿真评估，并在真实 Franka 机器人平台上进行桌面场景长度为 3 的多技能序列实验。

**📈 对比分析**

与多种基线（OpenVLA、UniVLA、EL3DD、VPP、FALCON 等）相比，MAST‑VLA 在 CALVIN ABC→D 上取得了最高的 Avg. Len 4.65 与 Success@5 84.7%，在真实机器人实验中也表现出更高的序列完成长度和更强的鲁棒性。

**⚠️ 局限性**

局限性包括：1) 未对低层动作误差进行实时纠正，需闭环反馈或重新规划；2) 轨迹热图依赖 RGB‑D 视觉，严重遮挡、透明或反射表面会导致空间先验失效。

---

## 75. Knowledge Graphs Quality Assessment in the Era of LLMs: A Conceptual Framework

**arXiv ID:** 2610.31684 | [PDF](https://arxiv.org/pdf/2610.31684v1)

**作者:** Elwin Huaman `[一作]` (University of Cambridge), Elwin Huaman `[通讯]` (University of Cambridge)

**通讯引用:** 631 | [OpenAlex ID](https://openalex.org/A5086006163)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a2602d71-93ab-4bad-974b-672788df8193` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一个面向知识图谱与大语言模型的质量评估框架，基于GQM方法构建了24个质量维度并给出了对应指标，随后通过用户驱动的四步流程（识别、设置、评估、利用）对知识图谱进行量化评估，并在POI（兴趣点）领域以DBpedia、Wikidata等数据集为例演示了该框架的使用。

**💡 创新点**

创新点在于①将多来源的质量维度系统化为24个可量化维度并与LLM相关指标相结合；②引入可加权的Alpha/ Beta权重机制，使评估可按场景定制；③提出完整的四阶段评估流程，兼顾自动与人工评估；④对现有质量评估文献进行整合，形成统一的概念框架。

**🔧 技术方法**

技术上采用了Goal Question Metric (GQM)方法来定义问题与指标，使用权重归一化与加权求和计算质量分数，框架实现为可视化用户界面支持实时权重调节与结果展示；部分指标可通过自动脚本计算（如SPARQL查询可用性、语法/语义校验等），其余需人工判定。

**📊 数据集**

本文在案例演示中使用了公开知识图谱数据集：DBpedia、Wikidata、Google Knowledge Graph（通过其API/SPARQL端点）以及POI相关本地数据，未引入其他大型数据集。

**📈 对比分析**

比较方法主要是基于权重加权的分数汇总，并与多KG在同一任务（POI检索）下的质量分数进行对比；由于本文未给出具体数值或性能指标，只说明框架能快速定位不同KG在各质量维度的优势与劣势。

**⚠️ 局限性**

局限性包括①部分指标仍需人工评估，缺乏完全自动化；②框架依赖用户对权重的主观设定，可能导致结果偏差；③在大规模KG或动态更新场景下评估成本较高；④未在多种领域进行广泛验证，缺乏跨领域的实验结果。

---

## 76. Links of minimum degree and exact covering numbers

**arXiv ID:** 2610.31696 | [PDF](https://arxiv.org/pdf/2610.31696v1)

**作者:** Alfonso Gippini Requeijo `[一作]` `[通讯]` (Team Banzai S.L.U.), Alfonso Gippini Requeijo (Team Banzai S.L.U.)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究了覆盖数C(v,k,t)的计算，提出通过“link”分解把寻找L块覆盖转化为对最小度数链接的分类，并给出了完整可验证的证明流程。

**💡 创新点**

创新点在于将link分解与完整性公式相结合，使得不必信任枚举器，仅需验证不可满足公式即可；并手工完成四块链接的分类，进而求得新覆盖数（如C(39,24,3)=9）。

**🔧 技术方法**

技术方法包括：使用SAT求解器kissat和DRAT证明检查器drat-trim；利用CakeML实现的LRAT检查器cake_lpr进行正式验证；采用canonical augmentation枚举链接；通过公式编码实现不可满足证明。

**📊 数据集**

数据来源主要是La Jolla Covering Repository与Covering Repository，所有公式、类列表、日志等均发布在Zenodo的研究记录中。

**📈 对比分析**

通过对比已知的下界与计算出的下界，验证了多组C(v,k,t)=L+1，填补了多项空缺；在计算资源允许的范围内，所需时间从几小时到数天不等，性能优于传统纯计算或手工方法。

**⚠️ 局限性**

局限性在于：仅适用于链接块数有限、t较小的情形；高度依赖SAT求解与可验证证明，资源消耗大；当r/L极低或kL=vr时方法效果有限。

---

## 77. StateBridge: Phase- and Track-Grounded Reasoning for Synthetic-to-Real Traffic Video Captioning and VQA

**arXiv ID:** 2610.31898 | [PDF](https://arxiv.org/pdf/2610.31898v1)

**作者:** Blessing Agyei Kyem `[一作]` (North Dakota State University), Armstrong Aboah `[通讯]` (North Dakota State University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对模拟到真实交通安全视频理解，提出了 StateBridge 框架，用阶段感知的证据构造、轨迹驱动的 VLM 适配以及保守的事实锁定融合与多证据 VQA 共识进行阶段级行人和车辆描述与问答生成。

**💡 创新点**

创新点在于将阶段、角色、轨迹几何等结构化、域不变信息作为跨域迁移的关键接口，并通过事实锁定门控与证据多样性共识保证生成结果的事实一致性，而非单纯依赖外观拟合。

**🔧 技术方法**

技术上结合了 Qwen3‑VL‑8B‑Instruct 及 LoRA 微调、文本-盒子摘要、渲染覆盖、交互裁剪等多种视觉/文本证据，使用自定义的事实锁定评分函数和加权投票机制实现输出融合。

**📊 数据集**

使用的主要数据集为 SynWTS Digital Twin（合成训练集）以及 AI City Challenge 2026 Track 2 的真实 WTS 视频（仅用于推理评测）。

**📈 对比分析**

与其他仅使用单模型或直接生成的基线相比，StateBridge 在官方 S2 分数上提升至 55.4679，排名第 8，展示了在合成训练下对真实视频的较好迁移效果。

**⚠️ 局限性**

限制主要在于对阶段和轨迹信息的高度依赖，若合成与真实场景在这些结构上差异较大或缺乏高质量轨迹标注，性能可能受限；此外模型仍受制于 VLM 的语言生成偏差与对细粒度视觉细节的捕捉不足。

---

## 78. An Agentic Approach with Evidence-Grounded Verification for Multimodal Emotion Reasoning in Conversation

**arXiv ID:** 2610.31685 | [PDF](https://arxiv.org/pdf/2610.31685v1)

**作者:** Elyas Rashno `[一作]` (Queen's University), Farhana Zulkernine `[通讯]` (Queen's University)

**通讯引用:** 2973 | [OpenAlex ID](https://openalex.org/A5063480277)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于冻结模型的多代理情绪推理框架，并通过证据追踪与验证实现可追溯性。

**💡 创新点**

创新点在于将错误归因到具体代理并通过验证层自动生成并验证提示补丁，实现不更新权重的针对性修复。

**🔧 技术方法**

使用多模态感知代理、检索增强、整合代理、内联MLLM批评器、确定性与含义检查以及提示空间强化。

**📊 数据集**

使用MER2025 Track 3（MER‑DES）公开情绪识别数据集进行评估。

**📈 对比分析**

与单调用、链式思考、全局/均匀/目标修复等对照，目标修复在四种模型上均优于静态管道，提升F‑score最高可达2.2点。

**⚠️ 局限性**

局限在于仅审计被引用的证据，无法修复不可追溯错误，以及多次模型调用导致的推理延迟未量化。

---

## 79. Towards Trustworthy AI-Enabled Emergency Medical Dispatch: Uncertainty-Aware Ambulance Travel Time Estimation

**arXiv ID:** 2610.31944 | [PDF](https://arxiv.org/pdf/2610.31944v1)

**作者:** Muhammad Adil Abid `[一作]` (Malmö University), Jesper Petersson `[通讯]` (Region Skåne)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出并验证了一种面向不确定性的救护车行驶时间估计方法，结合领域驱动的数据清洗、特征工程、点预测（ANN）与 conformal 校准生成可解释的预测区间

**💡 创新点**

在传统点预测基础上首次引入 conformal 校准实现可解释的不确定性区间，并通过子组覆盖评估确保在不同运营情境下的可靠性

**🔧 技术方法**

采用领域驱动的数据清理、特征编码、ANN 点预测模型，以及 Global Split Conformal、Conformalized Quantile Regression（CQR）和 Mondrian Conformal（优先级分层）三种 conformal 校准策略

**📊 数据集**

瑞典 Skåne 地区 2020‑2021 年两年的救护车行程日志（共 20,738 条有效记录）

**📈 对比分析**

与传统高斯参考区间及未校准分位回归对比，90% 置信水平下三种 conformal 策略的覆盖率均在 89–93%，ANN 的 MAE 为 4.42 分钟、R² 为 0.74，CQR 在不同距离四分位的覆盖率最为稳定（88.9–91.6%）

**⚠️ 局限性**

方法依赖于历史数据的完整性与质量，未实时采集交通/天气信息，缺乏前瞻性人机交互评估，且对极端或突发事件的预测能力有限

---

## 80. Validating the Comparison, Not Just the Scorer: Why Overall Accuracy Can Mislead in Interactive LLM Evaluation

**arXiv ID:** 2610.31678 | [PDF](https://arxiv.org/pdf/2610.31678v1)

**作者:** Ancil Crayton `[一作]` (Booz Allen Hamilton), Ancil Crayton `[通讯]` (Booz Allen Hamilton)

**通讯引用:** 3 | [OpenAlex ID](https://openalex.org/A5018847710)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文针对大语言模型在不同提示条件下的评估，提出了一套针对提示特异性误差的审计与验证流程。

**💡 创新点**

创新点在于引入“同一键”对比、四道门控评估（构造、错误观测、可迁移性、同键一致性）以及将校准不确定性纳入差异估计。

**🔧 技术方法**

使用的技术包括：混合规则+LLM评判的自动评估管道、两家独立LLM评估者、Rogan‑Gladen 校准修正、Bootstrap 与线性概率模型等统计方法。

**📊 数据集**

数据集为 180 条构造性不存在实体（城市、电影、科学家）问题，覆盖 6 种提示（Spontaneous, Soft cue, Neutral, Existence, Confidence, Authority），使用 6 种 LLM（Claude Sonnet 5, Gemma 3‑4B, Llama 4 Maverick, GPT‑5.6 Luna, Qwen‑32B, Qwen‑8B）。

**📈 对比分析**

在 288 条平衡抽样样本中总体 F1 为 0.916，但各提示下的 FP 率差异大（0.026–0.333），导致提示对比的差异估计从 9.3 点下降到 2.3 点；同键评估显示部分对比仍保持符号一致，但校准后不确定性仍较大。

**⚠️ 局限性**

局限性包括：审计样本规模有限导致精度与灵敏度区间宽；LLM 评估者不等同于人工真实标注；只覆盖六种模型与提示，缺乏对更广泛目标群体的可迁移性验证。

---

## 81. KnitTrans: Coordinated Trajectory Transfer for Effective and Efficient LLM Reinforcement Learning

**arXiv ID:** 2610.32389 | [PDF](https://arxiv.org/pdf/2610.32389v1)

**作者:** Bangjun Xiao `[一作]` (Peking University), Hailin Zhang `[通讯]` (Xiaomi)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一套针对大语言模型RL的轨迹传输系统，协调从rollout到训练的轨迹录入、存储、调度与多模态编码重用。

**💡 创新点**

创新点在于（1）约束保持的迷你批次录入策略，突破train-batch录入壁垒；（2）按训练拓扑的原子单位存储与按需访问，实现空间时间匹配；（3）全迷你批次的多模态编码去重，既保留梯度贡献又显著压缩计算。

**🔧 技术方法**

核心技术包括分布式原子单位存储（TransferQueue）、基于Topo-aware重排的按需访问、微批次并行预取、GRPO+R3的离线重放、FP8量化训练与推理、Ray共享内存、Tensor/Context Parallel等。

**📊 数据集**

使用CodeAgent（SWE-smith、Multi-SWE-bench、Terminal-Lego、Scale-SWE）与Video Grounding（Grounding A、B）两类任务，并在Qwen3-VL-235B、MiMo-V2.5-309B、DeepSeek-v3.2-671B三大模型上实验。

**📈 对比分析**

与 colocated 训练、train-batch录入、TransferQueue、Miles 等基线对比，系统在 disaggregated 设置下将架构时延降低至0.07~0.34 步，传输延迟降至 200‑400 倍，整体训练+传输加速最高达5.8×，RL 收敛几乎与 colocated 一致。

**⚠️ 局限性**

局限性包括对多模态输入依赖较高、需对原子单位尺寸与分布策略进行调优、在极大规模或多节点网络延迟时仍可能产生额外同步开销，以及系统实现复杂度较高。

---

## 82. Anti-coherence Regularized Absolute Scaling for Out-of-Distribution Detection

**arXiv ID:** 2610.32255 | [PDF](https://arxiv.org/pdf/2610.32255v1)

**作者:** Donghwan Kim `[一作]` (Yonsei University), Hyunsoo Yoon `[通讯]` (Yonsei University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一个无训练数据、无梯度的后置式OOD检测方法，改进了SCALE的比例因子并加入了Anti‑coherence正则；

**💡 创新点**

创新点包括：1）Absolute Fix——用绝对值替代ReLU以处理负激活；2）Anti‑coherence——利用最大logit的符号不匹配贡献比例来正则化缩放因子；

**🔧 技术方法**

技术：基于激活形变的SCALE方法、绝对值截断、比例因子正则化、能量分数结合；

**📊 数据集**

数据集：CIFAR‑10（ID）与MNIST、SVHN、Texture、Places365、TIN、CIFAR‑100等；ImageNet（ID）与iNaturalist、Texture、OpenImage‑O、SSB‑Hard、NINCO等；

**📈 对比分析**

对比方法包括MSP、MaxLogit、Energy、ReAct、ASH‑S、SCALE、VRA、OptFS、LTS、LAPS、AdaSCALE等；实验表明在10个ImageNet骨干网络上，Anti‑coh在FPR@95和AUROC上均优于或与使用训练数据的激活形变方法持平；在CIFAR‑10上亦取得最佳平均AUROC；

**⚠️ 局限性**

局限：对仅产生非负激活的网络提升有限；与基于表示的特征距离方法仍存在性能差距；对近OOD场景的提升不如部分梯度/训练依赖方法；

---

## 83. SKG in Practice: OOD Generalization Cases Across FourDomains

**arXiv ID:** 2610.32314 | [PDF](https://arxiv.org/pdf/2610.32314v1)

**作者:** Jian An `[一作]`, Xinpan Luo `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `5a41884c-404f-4688-a89c-aa238c10fe68` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并实现了 SKG 推断框架的工程化方法，验证了其在四个不同领域的 OOD 泛化能力；

**💡 创新点**

创新点在于将预验证、任务不变式构造、三阶段训练与残差训练相结合，并给出了四个关键定理与一个命题来理论支撑该流程；

**🔧 技术方法**

核心技术包括任务不变式 (如物理轮廓、谱先验)、弱/强形式残差学习、三阶段分离式训练、预验证指标与变量投影解释；

**📊 数据集**

使用了四个数据集：二维耗散力学系统、工业倾斜木块 Sim‑to‑Real 目标检测、时间序列的正弦/方波、MD17 分子动力学；

**📈 对比分析**

与传统神经网络基线对比，SKG 在 OOD MSE、mAP、AP75 等指标上分别提升 1.7–4.1 倍、+24 点、+40.3 点等，且参数量更少；

**⚠️ 局限性**

局限包括单跑实验、样本量有限、单核种子、对 OOD 领域分割的依赖、残差网络与官方实现不完全等。

---

## 84. Forging Emotions: a deep learning experiment on emotions and art

**arXiv ID:** 2610.32163 | [PDF](https://arxiv.org/pdf/2610.32163v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 85. Collaborative In-Context Learning for Indoor Radio Map Estimation

**arXiv ID:** 2610.31702 | [PDF](https://arxiv.org/pdf/2610.31702v1)

**作者:** Yuanzhe Peng `[一作]` (University of North Florida), Jie Xu `[通讯]` (University of Florida)

**通讯引用:** 23899 | [OpenAlex ID](https://openalex.org/A5008162410)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出Poly‑Ripple框架，将室内射频地图估计转化为协作的上下文学习问题，实现无需模型更新的无线信号预测。

**💡 创新点**

首次将大规模自回归视觉模型与波纹式自回归推理相结合，并通过重要性加权聚合分布式演示实现隐私友好的协作推理。

**🔧 技术方法**

使用VQ‑GAN对视觉输入进行离散化，采用LLaMA‑7B作为自回归推理器，并实现波纹排序与重要性聚合。

**📊 数据集**

在公开室内射频地图基准（包含25栋楼、3频段、5天线模式、27,750个样本）上进行实验。

**📈 对比分析**

与本地ICL和均匀聚合的协作ICL对比，Poly‑Ripple在所有5个未见布局上RMSE均显著下降，证明了协作与重要性聚合的有效性。

**⚠️ 局限性**

受限于自回归模型的上下文窗口、演示数量和演示分布的多样性，且对温度参数的选择和客户端匹配风险估计仍需经验。

---

## 86. Semantically Rich, Metrically Affect-Blind: A Geometric Blind Spot in Modern Text Embedders

**arXiv ID:** 2610.32122 | [PDF](https://arxiv.org/pdf/2610.32122v1)

**作者:** Mahdi Naser Moghadasi `[一作]` `[通讯]` (BrightMind AI), Mahdi Naser Moghadasi (BrightMind AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

分析并揭示现代文本嵌入器在情感属性上的解码-几何解耦现象，并提出并验证了一种小型监督对比头（SISE）可在CPU上秒级训练来修复该几何缺陷

**💡 创新点**

①引入Attribute Alignment Gap（AAG）这一无监督、标签仅需、可重用的度量来量化属性在嵌入空间中是否被几何表达；②证明该解耦与预训练目标相关而非提示模式；③展示仅通过轻量对比头即可显著提升检索MAP和几何表达，且对语义相似度的损失可通过融合保持

**🔧 技术方法**

使用冻结的句子编码器（MiniLM、E5、BGE、Qwen3-Embedding等）、监督对比学习（SupCon）训练128维投影头、线性探针、AAG度量、余弦检索

**📊 数据集**

SST-2、MR（含4,000句训练子集）、AG News、TweetEval等多属性标签集，用于探针、AAG和检索评估

**📈 对比分析**

在11种编码器上对比基线与SISE；对照线性探针准确率、检索MAP、AAG。结果显示：SISE在情感检索上提升9.8–26.0点MAP，probe准确率提升1.4–5.8点，AAG提升7–61倍；在低标签情境下，2-shot探针准确率提升至接近全标签极值；总体性能提升显著，且规模不再是决定因素

**⚠️ 局限性**

实验仅覆盖英语二元情感任务与少量其他属性，未对更大模型（>8B）或多语言数据进行评估；SISE仅针对情感属性，其他属性的修复效果未知；对比基线仅为线性探针，未与更复杂的监督投影或prompt式检索进行对比；AAG受空间异方差影响，跨空间比较需归一化

---

## 87. Searching with Sparse Histories: Cross-Task Architecture Search from Incomplete Search Logs and Zero-Cost Proxies

**arXiv ID:** 2610.32179 | [PDF](https://arxiv.org/pdf/2610.32179v1)

**作者:** Yuefeng Xu `[一作]` (Niigata University), Jun Yu `[通讯]` (Niigata University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种结合稀疏搜索历史和零成本代理的跨任务神经架构搜索方法，先用插值获得源任务冠军组合作为初始化，再联合预训练 pairwise ranker 并在目标任务上微调，最后进行弱预测搜索；

**💡 创新点**

创新点在于把稀疏历史记录与零成本代理统一视为多任务训练，利用冠军组合避免传统共识先验失效，同时实现对两种信息源的风险鲁棒融合；

**🔧 技术方法**

使用了结构编码、岭回归插值、pairwise RankNet 损失、ensemble、弱预测搜索等技术；

**📊 数据集**

在 TransNAS-Bench-101、MedMNIST-NAS-Bench 以及基于真实训练的 300 个网络池上进行评估；

**📈 对比分析**

与随机搜索、Arch-Graph、TRP 等基线相比，平均排名与基线相当或略优，最坏 10% 尾部显著降低，尤其在稀疏历史严重失效或代理不可靠的目标上表现更稳健；

**⚠️ 局限性**

局限性包括仅在 tabular benchmark 上充分验证，真实训练实验规模受限，目标任务数量有限，超参数在 TransNAS 上调优，且假设新旧任务共享搜索空间；

---

## 88. Automated Nucleus Segmentation in Histology Images: An Attention U-Net Approach with Comprehensive Preprocessing and Inference Pipelines

**arXiv ID:** 2610.31902 | [PDF](https://arxiv.org/pdf/2610.31902v1)

**作者:** Akshat G `[一作]` (Manipal Academy of Higher Education), Tusar Kanti Mishra `[通讯]` (Manipal Academy of Higher Education)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究实现并评估了轻量级Attention U‑Net模型用于MoNuSeg 2018数据集的细胞核分割，结合染色提取、CLAHE、掩膜平滑等预处理及重叠补丁拼接的全图推断流水线。

**💡 创新点**

创新点在于：1) 将染色去卷积与CLAHE等预处理与Attention U‑Net相结合，提高对高密度核与染色不均的鲁棒性；2) 采用重叠补丁拼接并加权平均，保证全图分割的无缝连续；3) 使用轻量级模型仅2M参数即可达到与更复杂模型相近的性能。

**🔧 技术方法**

核心技术包括：Attention U‑Net架构、注意力门、染色去卷积、CLAHE、掩膜多通道平滑、重叠补丁推断与加权拼接、BCE+Dice混合损失、TensorFlow/Keras实现。

**📊 数据集**

使用公开的MoNuSeg 2018多组织细胞核分割挑战数据集（30幅全分辨率图像，训练24幅，测试6幅）。

**📈 对比分析**

与Baseline U‑Net、Mask R‑CNN、U‑Net、CUMedVision等方法对比，Attention U‑Net在IoU上达0.8745，Dice 0.9323，显著优于Baseline和多种现有方法，性能接近挑战冠军。

**⚠️ 局限性**

局限性包括：仅在单一数据集验证，缺乏跨数据集泛化评估；补丁数量与重叠比例的选择仍为经验性，未自动优化；模型仍为2D分割，未考虑三维或实例分割需求。

---

## 89. Delay Randomization Widens the Delay Margin of Learned Humanoid Motion Trackers

**arXiv ID:** 2610.32277 | [PDF](https://arxiv.org/pdf/2610.32277v1)

**作者:** Shaoqin Li `[一作]` (City College of New York), Yi Sun `[通讯]` (City College of New York)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文在已验证的仿真检验平台上，测定了基于强化学习的类人机器人运动跟踪器对固定命令延迟的容忍度，并探讨了延迟随机化训练对容忍度的影响。

**💡 创新点**

创新点在于：①提出一种严格验证的仿真-仿真检查流程，剔除模型配置误差导致的误差；②系统量化不同延迟随机化范围对延迟裕度的提升效果；③通过对比额外训练与延迟随机化的控制实验，明确归因。

**🔧 技术方法**

使用了PPO强化学习、MuJoCo（GPU版和CPU版）与PhysX仿真器、BeyondMimic跟踪算法、ONNX模型导出与CPU执行、延迟缓冲区等技术。

**📊 数据集**

数据集为从五段单目视频提取的动作序列（网球发球、舞蹈、拳击等），通过GVHMR与GMR回退至关节轨迹，共计约 100–150 秒不同速度的参考动作。

**📈 对比分析**

比较方法：在独立的MuJoCo CPU harness 里，分别给定不同的恒定延迟（0–80 ms）并记录“存活率”与跟踪误差。结果显示：无延迟随机化的模型在 20–30 ms 时已出现明显崩溃；随机化到 0–50 ms 的训练使延迟裕度提升至 50–70 ms，存活率在 40–50 ms 仍保持 80–95%。

**⚠️ 局限性**

局限性包括：仅在仿真环境中测试，未考虑真实机械延迟；使用单一机器人模型与有限动作集；延迟设定为恒定值（未覆盖时变延迟）；训练样本数有限，可能低估重训练的方差。

---

## 90. MisconceptDialogue: Evaluation Criteria Shape Measured Success in LLM Math Tutoring

**arXiv ID:** 2610.31679 | [PDF](https://arxiv.org/pdf/2610.31679v1)

**作者:** Woojin Lee `[一作]` (Electronics and Telecommunications Research Institute), Jin-Xia Huang `[通讯]` (Electronics and Telecommunications Research Institute)

**通讯引用:** 225 | [OpenAlex ID](https://openalex.org/A5007149994)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

该论文设计并实现了 MisconceptDialogue 框架，对 7B–14B 开源 LLM 辅导系统在 5,000+ 轮对话中完成错误检测、诊断和纠正的全流程进行全自动评估。

**💡 创新点**

创新点在于将检测–诊断–纠正管道与评价标准分离，揭示评价准则（严格匹配 vs. LLM 判别器）对 Remediation Success Rate 的显著影响，并通过错误诊断对照验证 LLM tutor 是否真正依据诊断信息作出教学决策。

**🔧 技术方法**

使用的技术包括多轮对话生成、LLM 判别器（GPT‑5‑mini/5‑4、Claude Haiku 4.5、Gemini 3.1 Flash Lite）、多种提示策略（Zero‑shot、Few‑shot、CoT+Rubric、RAG、GRR）、以及数字提取与严格正则匹配两种自动评分器。

**📊 数据集**

采用的数据集包括 BEA 2025、MathDial、Bridge（用于错误检测）；Math Misconceptions Benchmark（55 种代数误解）用于诊断；以及自构造的 106 个包含事实、程序、概念三类误解的辅导情境，配有验证题目。

**📈 对比分析**

在多判别器（4 款 LLM）和人工评估的对比下，同一对话在严格匹配与 LLM‑判别标准下 Remediation Success Rate 差距高达 39.2 百分点；在管道信息对照实验中，Oracle 诊断与无诊断条件对 RSR 无显著差异，性能差距不到 5 百分点。

**⚠️ 局限性**

局限性包括：使用模拟学生而非真实学习者，结果对实际课堂迁移性未知；误诊断对照仅检验诊断标签是否被使用，未评估其内容有效性；仅覆盖中学代数误解，未扩展到更广泛学科；LLM 判别器与学生模拟器同源，可能存在内部偏差。

---

## 91. GeoLEON-CLIP: Graph-Anchored Multimodal Contrastive Learning for Geographic Region Representation

**arXiv ID:** 2610.32311 | [PDF](https://arxiv.org/pdf/2610.32311v1)

**作者:** Radoslaw Malek `[一作]` (Wrocław University of Science and Technology), Piotr Szymanski `[通讯]` (Wrocław University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 GeoLEON-CLIP 框架，将 OSM 图谱嵌入、卫星图像和文本描述通过三模态对齐学习可复用的 H3 区域表示。

**💡 创新点**

创新点在于将预训练的图谱编码器 LEON 作为语义锚点，构建三模态对齐，并引入链式检索指标 R@A@B 评估图谱在图像–文本桥接中的作用。

**🔧 技术方法**

使用冻结的 LEON 图谱编码器、DINOv3 卫星图像编码器和 Jina 文本嵌入器，配合轻量级残差投影头和 InfoNCE 对比损失进行训练。

**📊 数据集**

构建全对齐的 Tri‑Modal H3 数据集，覆盖 35 个全球大都市，三模态分别来源于 OSM、Sentinel‑2 以及 Wikipedia/Wikivoyage/Reddit，按 H3 等级划分。

**📈 对比分析**

与基线（如 RemoteCLIP、GeoRSCLIP 等）对比，图像–图谱检索 Recall@1 达 0.80，图像–文本检索提升 4–5 点；链式检索 R@1@1 通过图谱桥接比直接图像–文本检索高约 4.8%/4.4%，并在 Airbnb、Beijing 房价等下游任务中带来 1–2% 的 R² 提升。

**⚠️ 局限性**

主要局限包括 OSM 覆盖不均、文本来源多样且时序不一致、模型训练仅使用冻结特征、评估未做空间阻塞和城市外迁移测试，且链式检索收益随 H3 分辨率和候选集规模减弱。

---

## 92. Incomparable Minima in Agent Tool Authorization

**arXiv ID:** 2610.31914 | [PDF](https://arxiv.org/pdf/2610.31914v1)

**作者:** Xin Xu `[一作]` `[通讯]` (Carnegie Mellon University), Xin Xu (Carnegie Mellon University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究并量化了 Microsoft Graph 与 Google API 授权表中任务的最小足够权限集合的不唯一性及其对权限尺度、风险度量的影响，并提出在评测与部署中记录权限选择的建议。

**💡 创新点**

揭示大多数任务的最小权限集合是一个不可比较的反链（antichain），并证明现有的权限等级尺度往往将这些反向对压平，导致评测无法区分不同的安全风险；提出通过完整列出反链成员并记录多维风险向量来改进评测。

**🔧 技术方法**

使用最小覆盖集枚举、部分顺序分析、FORTIS 等级重构、四种风险度量（覆盖范围、写操作数、服务数、管理员权限）以及前沿模型（frontier models）进行实验，计算可比与不可比对的折叠率、达成度和选取行为。

**📊 数据集**

Microsoft Graph 权限文件（包含委托、应用、个人模式）以及 Google API Discovery 服务生成的 OAuth2 作用域列表；在此基础上按资源、随机子集和路径组生成任务集。

**📈 对比分析**

通过与供应商自带的尺度（Microsoft 的 ordinal level）以及重构的 FORTIS ladder 进行对比，评估它们在不可比较对中的折叠率、风险度量逆序比例以及前沿模型在最小权限选择中的准确率；实验表明前沿模型在可比对上能保持 100% 的最小性，而在不可比较对上会出现多达 10–20% 的差异。

**⚠️ 局限性**

仅考虑方法级的授权表，未包含资源级或条件约束；任务来自表结构而非真实代理工作负载，导致结果对工作负载分布敏感；对 FORTIS ladder 的机械重构可能与原作者意图略有偏差；前沿模型实验假设所有可比对均已枚举，忽略了动态授权场景。

---

## 93. SEMPRE: Semantic-Aware Protocol Reverse Engineering via Consistency-Driven Joint Optimization

**arXiv ID:** 2610.31844 | [PDF](https://arxiv.org/pdf/2610.31844v1)

**作者:** Yihua Xu `[一作]` (Illinois Institute of Technology), Yue Duan `[通讯]` (Singapore Management University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种全自动协议逆向工程框架SemPRE，通过细粒度动态分析与LLM推理实现字段切分与语义推断；

**💡 创新点**

创新点在于将LLM的树形思考（Tree‑of‑Thoughts）与EM‑style迭代优化结合，实现递归子载荷检测与边界+语义协同改进；

**🔧 技术方法**

核心技术包括动态污点跟踪、循环检测、LLM树形搜索、反思式评估与迭代精炼；

**📊 数据集**

使用八个主流协议数据集（Modbus、S7comm、Ethernet/IP、DNP3、DNS、FTP、TFTP、HTTP）和Mirai恶意软件数据；

**📈 对比分析**

与PolyGlot、Tupni、AutoFormat、BinPRE、BinaryInferno、DynPRE等六个SOTA工具对比，SemPRE在字段检测F1达到0.76、语义推断F1达0.72，明显优于最佳基线；

**⚠️ 局限性**

局限性包括依赖LLM的可靠性（可能出现幻觉）、仅以字节级污点与语义，难处理位级协议，且对抗性混淆攻击仍未解决。

---

## 94. Fooling LLM Safety Guards into Judging the Wrong Text

**arXiv ID:** 2610.31695 | [PDF](https://arxiv.org/pdf/2610.31695v1)

**作者:** Utkarsh Kanwat `[一作]` `[通讯]` (Independent Researcher), Utkarsh Kanwat (Independent Researcher)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究并展示了一种针对安全守护模型的“目标重绑定”攻击，攻击者通过在内容中附加伪造的查询-文档对，使守护模型误判为安全并将有害请求转发。

**💡 创新点**

提出了目标重绑定攻击的概念，并证明其在多种主流守护模型（Shieldstral、Llama Guard、gpt‑oss‑safeguard、ShieldGemma、Qwen3Guard）中的高效性，同时提出了一种结合边界标记与段落最大化自蒸馏训练的防御方法。

**🔧 技术方法**

使用了自然语言文本插入、标签模仿、结构化攻击语法、边界标记(token boundary)以及基于Segment‑Max的自蒸馏训练（LoRA微调）。

**📊 数据集**

实验数据集包括AdvBench、HarmBench、Aegis 2.0、BeaverTails、OpenAI moderation evaluation、ToxicChat、XSTest‑safe、OR‑Bench‑hard以及结构化善意文本模板。

**📈 对比分析**

通过对照实验评估了原始守护模型、无训练边界、结构检测、推理时段最大化、训练防御（T）、边界+训练（B+T）等多种方案；防御T在未见攻击上将附加攻击的通道误判率从约75%降低到≤1.6%，但在自适应搜索攻击下仍可被突破，整体检测AUROC略有下降但保持在0.80左右。

**⚠️ 局限性**

局限包括：仅在单一语言和短文本上验证；未覆盖检索、工具调用或多轮对话场景；防御对自适应攻击仍易受突破；实验规模相对有限（部分结果基于少量样本）；以及对某些守护模型的防御效果提升幅度不大。

---

## 95. Large-Scale Study of GitHub Copilot Trajectories

**arXiv ID:** 2610.32313 | [PDF](https://arxiv.org/pdf/2610.32313v1)

**作者:** Izah Sohail `[一作]` (New York University Abu Dhabi), Sarah Nadi `[通讯]` (New York University Abu Dhabi)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

对GitHub Copilot生成的拉取请求进行大规模轨迹分析，记录人机交互循环、迭代次数和持续时间，并探究干预如何影响代理工作

**💡 创新点**

提出面向人机协作的轨迹编码方案，构建首个大规模人机交互轨迹数据集，并系统评估干预对合并率和代理工作量的影响

**🔧 技术方法**

使用Copilot会话页面爬取与HTML解析、Thought-Action-Result编码、线性混合效应模型以及Mann‑Whitney、χ²等统计检验方法

**📊 数据集**

AIDev v4数据集，包含6,673个≥100星仓库中的7,631个问题分配的Copilot PR，涉及1,602个仓库

**📈 对比分析**

通过统计检验比较干预与否、合并与否及任务类型差异，发现干预后合并率上升，首次干预显著减少迭代和时间；效应量从小到中等，说明干预影响有限

**⚠️ 局限性**

仅针对Copilot云代理、仅100星以上仓库、仅issue分配的PR、对HTML结构的依赖、未考虑代理能力差异与人类工作习惯，且结果随Copilot版本变化而可能变动

---

## 96. SWE-Shift: Evaluating Coding Agents in Evolving Workspaces

**arXiv ID:** 2610.32196 | [PDF](https://arxiv.org/pdf/2610.32196v1)

**作者:** Victor Shea-Jay Huang `[一作]` (Chinese University of Hong Kong), Hongsheng Li `[通讯]` (Chinese University of Hong Kong)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

暂无可用信息

**💡 创新点**

暂无可用信息

**🔧 技术方法**

暂无可用信息

**📊 数据集**

暂无可用信息

**📈 对比分析**

暂无可用信息

**⚠️ 局限性**

暂无可用信息

---

## 97. RubricCraft: Where Solutions Diverge, Criteria Emerge

**arXiv ID:** 2610.31707 | [PDF](https://arxiv.org/pdf/2610.31707v1)

**作者:** Hosein Hasanbeig `[一作]` (Microsoft), Sumit Gulwani `[通讯]` (Microsoft)

**通讯引用:** 13153 | [OpenAlex ID](https://openalex.org/A5011543162)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过采集多模型的求解轨迹，使用自然语言推断和修复评判准则，自动生成并改进评估 Rubric。

**💡 创新点**

创新点在于把 Rubric 的构造转化为差异化推断（diff‑based induction）和缺陷自修复（repair）循环，并提出公平性、完整性、可区分性和可检查性四类缺陷分类。

**🔧 技术方法**

采用大型语言模型（GPT‑5.6、GPT‑6、Opus‑5 等）生成 solver 轨迹与 Rubric，使用自然语言处理与模板化指令实现 criteria 生成，利用判定器（grader）和分析器（analyzer）进行缺陷检测与修复。

**📊 数据集**

在 SpreadsheetBench v2（含 259 个非可视化任务）上进行实验，使用其自带的程序化细胞比较器作为基准。

**📈 对比分析**

与同模型生成的“naïve”Rubric 以及基准检查器比较：RubricCraft 的错误通过率降至 29.5%（相较 42.6%）、接近完整工作误判率 2.4%（相较 6.6%），在模型区分度与稳定性上均优于基准，且多模型修复后覆盖率提升至 48.6%。

**⚠️ 局限性**

局限包括：对黄金参考的误判仍可能误导修复；仅针对电子表格类 artifact 进行评测，跨域推广未知；依赖多样化 solver 生成，单一模型时修复效果有限；修复过程仍需人工审查以确保语义一致。

---

## 98. FOCUS: A Lightweight Mamba-2 Model for EEG-Based Cognitive Workload Classification

**arXiv ID:** 2610.32361 | [PDF](https://arxiv.org/pdf/2610.32361v1)

**作者:** Ioannis Kyprakis `[一作]` (Hellenic Mediterranean University), Raul Fernandez Rojas `[通讯]` (University Of Canberra)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出轻量级FOCUS模型，利用固定FFT频段分解、深度可分离卷积和Mamba-2状态空间模型，对EEG进行二分类认知工作负载评估。

**💡 创新点**

创新点在于：①固定FFT频段嵌入（无学习参数），②权重共享的编码-瓶颈Mamba-2结构，③极低参数量（0.04 M）和极低延迟（0.08 ms）实现高效工作负载分类。

**🔧 技术方法**

技术包括FFT频段分解、层归一化、深度可分离时间卷积、Mamba-2序列模型、dropout、AdamW优化及数据增强。

**📊 数据集**

使用11名受试者的EMOTIV EPOC 14通道EEG数据，三种任务（IQ、MATH、GAME）各双难度，共264样本/任务，采用10 s非重叠片段。

**📈 对比分析**

通过留一受试者交叉验证和Wilcoxon检验，与EEGNet、1BT、Mamba-2和统一多模态框架等基线对比，FOCUS在ALL设置下达到74.75%准确率，0.04 M参数，0.11 GFLOPs，显著优于基线。

**⚠️ 局限性**

局限性包括样本量仅11人、单实验室环境、仅EEG数据、未验证不同头盔或更长录制的泛化能力，以及固定FFT前端未与其他时域/学习滤波器做直接对比。

---

## 99. Complementing DINO Features with Image Structure for Part Discovery

**arXiv ID:** 2610.32293 | [PDF](https://arxiv.org/pdf/2610.32293v1)

**作者:** Samyak Rawlekar `[一作]` (University of Illinois Urbana-Champaign), Narendra Ahuja `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种无训练的Part‑DINO框架，将低级图像边界与冻结的DINO ViT特征结合，实现前景物体的语义部分发现。

**💡 创新点**

核心创新是发现ViT固定网格导致的像素级混合扩散到特征空间，并通过RAMP+Watershed生成几何一致的区域，然后在这些区域内聚合DINO特征，从而获得精确的部分边界，无需任何额外训练。

**🔧 技术方法**

使用的技术包括：RAMP变换、Watershed分割、冻结的DINOv2 ViT特征提取、区域特征池化、K‑Means聚类生成全局部分原型、CLS token进行类别路由，以及基于余弦相似度的区域赋值。

**📊 数据集**

实验数据集为CUB、CelebA、PartImageNet‑OOD、PartImageNet‑Seg四个标准部件发现基准。

**📈 对比分析**

与弱监督、无监督和其他训练‑free方法对比，Part‑DINO在NME和ARI指标上均超过训练‑free基线，在PartImageNet‑Seg上在部分K值上超越了训练好的MPAE，在CUB与CelebA上与多种训练方法竞争，并在PartImageNet‑OOD上获得最高ARI。

**⚠️ 局限性**

局限性包括：需要预先指定部分数量K；依赖对比度强的低级边缘，低对比度或纹理稀疏场景可能导致边界检测不足或误检。

---

## 100. ReMedI: Inference-Time Refinement and Self-Distillation for Controllable Molecular Optimization

**arXiv ID:** 2610.32103 | [PDF](https://arxiv.org/pdf/2610.32103v1)

**作者:** Jia Yi Zhang `[一作]` (University of Illinois Urbana-Champaign), Tong Zhang `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `5b4c1114-4a70-478e-9921-2514ee03850d` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `f86bf285-fd08-4156-973b-6e6481af8fa0` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种基于预训练分子语言模型的推理时增量编辑框架ReMedI，利用Metropolis–Hastings搜索对分子进行局部编辑并在推理阶段实时对目标进行对齐，同时将搜索轨迹蒸馏回模型以提升生成质量。

**💡 创新点**

创新点在于：①把多目标优化目标嵌入到推理时的接受概率中，实现无需任务特定RL适配器的即时调整；②使用Metropolis–Hastings引导的序列编辑大幅提高多目标成功率和相似性平衡；③通过搜索轨迹蒸馏（不需要专门的编辑数据集）一次性提升基线模型的零射生成和后续搜索性能。

**🔧 技术方法**

核心技术包括：分子语言模型（GeLLMO-Llama、GeLLMO-Mistral、mCLM-4B）作为提议者；Metropolis–Hastings采样与接受准则；基于Nash博弈的多目标收益函数；轨迹级监督学习（seed、chain、transition三种例子）来蒸馏提议分布。

**📊 数据集**

使用的数据集为C-MuMOInstruct（10个多目标编辑任务，包含5个训练集、5个未见组合）以及SMDD-Bench Task 4（314个lead‑optimization实例），并使用ADMET‑AI和PyTDC进行属性评估。

**📈 对比分析**

与RL后训练基线C‑MORAL进行比较。ReMedI在GeLLMO-Mistral上在匹配的lead上实现了≈2.8×的严格成功率覆盖，且在IND/OOD拆分上提升了80–90% 的成功率；在SMDD-Bench上比C‑MORAL提升了0.13–0.29的SOR并获得更高的清除率成功率。其在相似性约束下保持更优的 Pareto 前沿，并且训练样本量仅为C‑MORAL的7.8倍。

**⚠️ 局限性**

主要限制是所有发现仅基于体内计算预测，尚未验证实验室或临床效能；此外，方法依赖预训练模型的质量和属性预测模型的准确性。

---

## 101. Scaling cross-situational word learning on children's naturalistic experiences

**arXiv ID:** 2610.31842 | [PDF](https://arxiv.org/pdf/2610.31842v1)

**作者:** Michael C. Frank `[一作]` (Stanford University), Bria Long `[通讯]` (University of California, San Diego)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文通过训练对比学习视觉语言模型，利用儿童第一视角视频数据探究跨情境词学习在自然环境中的可行性和规模；

**💡 创新点**

创新点在于将跨情境学习理论与大规模、真实儿童第一视角视频相结合，解耦视觉表示与词映射，系统评估互联网预训练视觉表示与自监督视觉表示对学习的影响，并量化引用对齐（referential alignment）在学习中的关键作用，同时对比模型与儿童学习曲线；

**🔧 技术方法**

采用对比学习视觉语言模型（Contrastive VLM），InfoNCE 损失，多实例学习（MIL）技术，冻结视觉编码器（DINOv3 与互联网预训练 OTS 模型），词嵌入与对齐层，使用 Konkle 4AFC 评估，结合 Gemini 进行引用对齐评分；

**📊 数据集**

使用了 BabyView 2026.1 版本（48 名儿童共 2,441 小时第一视角视频及转录文本），Konkle 评估图像集合，以及 Wordbank 与 LEVANTE 词汇测评数据；

**📈 对比分析**

通过 Konkle 4AFC 与 LEVANTE 评估与儿童数据对比，比较 OTS 预训练视觉编码器与仅用 BabyView 训练的视觉编码器，发现预训练视觉编码器显著提升 4AFC 结果（最高 81.6% 对比约 50% 等），引用对齐过滤后可提升 10-20%，但模型学习曲线与儿童曲线不同，儿童学习更陡峭；

**⚠️ 局限性**

局限性包括：视觉编码器为冻结状态，未能学习儿童视觉环境的高质量表示；引用对齐评分依赖第三方模型；数据样本主要来自高收入家庭，缺乏多样性；模型训练方式与人类学习机制差异大，未能复制儿童学习加速；且仅关注名词，未涉及其他词类。

---

## 102. Move, don't think: sample-efficient collision avoidance with lossy proximity sensors

**arXiv ID:** 2610.31891 | [PDF](https://arxiv.org/pdf/2610.31891v1)

**作者:** Carson Kohlbrenner `[一作]` (University of Colorado Boulder), Alessandro Roncone `[通讯]` (University of Colorado Boulder)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文通过实验研究了多种基于全身接近传感器的信号表示方式对样本高效碰撞规避的影响，并进一步评估了未来碰撞预测的作用。

**💡 创新点**

创新点在于：①提出使用宽覆盖、低分辨率传感器作为样本高效学习的先验偏置；②发现相较于高分辨率传感器，简单的“最小距离”变换即可显著提升样本效率；③证明对未来碰撞信息的预测反而降低了学习效率，挑战了传统的“提前预测”思路。

**🔧 技术方法**

采用了基于Proximal Policy Optimization（PPO）的强化学习框架、近似隐空间编码器、以及“近场世界模型（PWM）”对接近传感器数据进行未来观测预测。

**📊 数据集**

数据集为在仿真环境中收集的数千条专家演示轨迹（用于PWM预训练）以及在Unitree H1-2人形机器人上模拟的投掷球碰撞任务，传感器配置为随机布置的64个接近传感器。

**📈 对比分析**

通过对比八种信号表示（原始、最小距离、二值、潜在编码、PWM等）在固定3000步训练预算内的任务成功率与NAUC（正则化曲线面积）进行评估。结果显示：1）f_tp（完美定位）是上限；2）原始场传感器在2m范围内获得0.522 NAUC；3）原始射线传感器仅为0.02；4）采用最小距离变换后，射线传感器提升至0.39 NAUC；5）PWM预测信息反而使学习速度下降，性能低于仅使用当前观测。

**⚠️ 局限性**

局限性包括：①所有实验均在仿真环境中完成，缺乏真实硬件验证；②传感器范围在训练期间保持不变，未探索动态可调的个人周边空间；③PWM模型虽预训练但仍受限于专家演示数据，未证明在更复杂环境下的通用性；④样本效率提升主要源自信号维度压缩，未来是否能在更高维或更大规模场景中保持优势尚未证实。

---

## 103. CuraLoc: Repository-Level Fault Localization via Planned Code Inspection

**arXiv ID:** 2610.31809 | [PDF](https://arxiv.org/pdf/2610.31809v1)

**作者:** Sangeeta Warrier `[一作]`, Ryen White `[通讯]` (Microsoft)

**通讯引用:** 14279 | [OpenAlex ID](https://openalex.org/A5076259865)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出CuraLoc框架，利用预先规划的代码检索和一次性源代码获取来定位bug修复位置。

**💡 创新点**

创新点在于将起始符号与检索动作从紧凑的行为与结构描述中一次性规划，消除多轮模型驱动的导航。

**🔧 技术方法**

使用AST图结构、符号签名与行为摘要、Deterministic检索以及GPT等大型语言模型进行规划与最终排名。

**📊 数据集**

实验基于SWE‑Bench Lite和Verified的Python仓库数据集。

**📈 对比分析**

与LocAgent、RepoSearcher和Agentless比较，CuraLoc在Func@1上提升至59.3%，同时减少约90% token；在GPT‑6 Astra上保持与RepoSearcher相当的准确性。

**⚠️ 局限性**

局限性包括仅在Python SWE‑Bench 上评估，未验证跨语言、跨仓库的适用性，也未直接评估对修复成功率的影响。

---

## 104. Two-colour crossing minimization in complete geometric graphs is NP-hard

**arXiv ID:** 2610.31719 | [PDF](https://arxiv.org/pdf/2610.31719v1)

**作者:** Zhe Zhang `[一作]` `[通讯]`, Zhe Zhang

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

证明在给定完整几何图的前提下，寻找两种颜色的边使单色交叉数最小化是NP‑完整的，并给出了多项式时间的确定性构造和整数坐标实例。

**💡 创新点**

通过将置换图的最大割问题归约到一个由比较网格和程序点构成的几何实例，利用定量稳定性定理控制颜色分布，从而实现了完整几何图两色交叉最小化的有效化简。

**🔧 技术方法**

使用几何构造、定量稳定性分析、MaxCut归约、整数规划、符号计算以及Lean形式化证明，对误差与基准进行严格估计。

**📊 数据集**

本研究为理论证明，不涉及实际数据集；仅使用理论上的置换图与几何点集合。

**📈 对比分析**

由于是NP‑完整性证明，没有实验对比；结果说明除非P=NP，否则不存在多项式时间精确算法，但可借此证明近似或随机化方法的必要性。

**⚠️ 局限性**

局限性在于适用于完整几何图、构造实例规模非常大且常数项庞大；对实际求解仍需有效启发式或近似算法，理论结果对实际应用的直接帮助有限。

---

## 105. DirCast: Live Esports Directing and Commentary with a Single Streaming Video-Language Model

**arXiv ID:** 2610.31734 | [PDF](https://arxiv.org/pdf/2610.31734v1)

**作者:** Zixian Gao `[一作]` (Alaya Lab), Kaipeng Zhang `[通讯]` (Alaya Lab)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种统一的实时直播制作模型DirCast，能在收到所有英雄摄像头流的情况下同时决定画面切换并进行解说。

**💡 创新点**

创新点在于将导演与解说整合到同一因果视频语言模型中，使用hold‑or‑switch导演头、程序记忆机制和事件锚定目标，使模型在自身切换错误时能自我恢复并生成合适的解说。

**🔧 技术方法**

技术手段包括基于Qwen3‑VL‑8B的因果视频语言模型、低分辨率视图编码、hold‑or‑switch决策、程序记忆策略、扰动程序训练与动态阈值控制。

**📊 数据集**

使用27场职业Dota 2比赛的重放和官方直播构建的多摄像头数据集，每秒十个英雄视角与对应的游戏事件、导演决策和解说文本。

**📈 对比分析**

与规则基、零样本VLM及人类导演/解说比较，DirCast在导演准确率（Top‑1 ≈ 27%、击杀覆盖率 ≈ 89.1%）和解说质量（与GPT‑5评估的比较赢率 ≈ 29%）等指标上均优于基线，平均说话延迟仅 0.30 s。

**⚠️ 局限性**

局限在于仅针对Dota 2专业比赛，缺乏跨游戏或跨语言的验证，且对直播中断、重播等非正常事件的处理仍有限。

---

## 106. Organizing Temporal Evidence in Hyperbolic Space for Video Moment Retrieval

**arXiv ID:** 2610.32317 | [PDF](https://arxiv.org/pdf/2610.32317v1)

**作者:** Mengnan Liu `[一作]` (Xi'an Jiaotong University), Le Wang `[通讯]` (Xi'an Jiaotong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于双曲几何的时序证据组织框架HTEO，用于视频时刻检索；

**💡 创新点**

通过在Poincaré球中引入半径-角度结构，分别通过半径组织时序结构信息、角度提升查询相关性，并配合残差记忆更新实现对Transformer解码器的无缝融合；

**🔧 技术方法**

双曲几何映射（指数/对数映射）、Möbius变换、正交旋转、几何正则化（半径结构约束、角度相关性监督）以及传统的跨模态注意力、DETR式解码器；

**📊 数据集**

QVHighlights、Charades-STA、TACoS三大公开基准；

**📈 对比分析**

与多种基线（M-DETR、QD-DETR、UniVTG、TR-DETR、LD-DETR等）对比，HTEO在QVHighlights、Charades-STA、TACoS上均取得最优或近优mAP、R1@IoU等指标，尤其在高歧义、短时刻场景下提升显著；

**⚠️ 局限性**

对曲率、残差尺度、注入位置等超参需调优；在极大时长或极低信息的视频中，半径与角度区分度有限；仅在Transformer框架下验证，其他解码器适配性待研究。

---

## 107. MedCaseAgent: From Multimodal Clinical Evidence to Rare Disease Case Reports

**arXiv ID:** 2610.32258 | [PDF](https://arxiv.org/pdf/2610.32258v1)

**作者:** Hongkang Chu `[一作]` (PediaMed AI), Xu Cao `[通讯]` (PediaMed AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

开发了MedCaseAgent系统，实现从碎片化的临床笔记和医学影像自动生成罕见病病例报告。

**💡 创新点**

通过多模态规划–写作–编辑框架结合工具调用，提供高可信度、低幻觉的病例报告生成，并发布MedCase-150K/MedCase-Bench基准。

**🔧 技术方法**

使用多模态vision‑language模型、工具调用、GRPO强化学习训练规划器、LLM写稿、编辑校验等技术。

**📊 数据集**

构建MedCase-150K（154,872篇罕见病病例）和其时序保留子集MedCase-Bench（584篇）。

**📈 对比分析**

在MedCase-Bench上与直接工具LLM和现有医学代理系统对比，MedCaseAgent-Max（GPT‑5.4）平均得分≈68/100，显著优于所有基线，并在不同病种子集均保持领先。

**⚠️ 局限性**

仍受限于检索库覆盖不足的极罕见疾病、对外部工具的依赖以及需要人工临床审核以防范错误。

---

## 108. Effectiveness of Curriculum Learning Depends on Reward Sparsity and Competing Optima

**arXiv ID:** 2610.31884 | [PDF](https://arxiv.org/pdf/2610.31884v1)

**作者:** John J. Vastola `[一作]` (Harvard University), Kanaka Rajan `[通讯]` (Harvard University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

构建并分析了一个可解析的目标拦截任务中的政策学习最小模型，以研究课程学习（CL）在强化学习中的有效性。

**💡 创新点**

发现奖励稀疏性和竞争性最优点决定了CL的必要性，并提出了利用动态目标大小和模态切换的课程策略，首次在解析模型中证明了这一点。

**🔧 技术方法**

使用REINFORCE算法的平均动力学、期望奖励分析、有效目标函数以及控制论框架来设计课程。

**📊 数据集**

无，模型基于仿真和理论推导，不涉及公开数据集。

**📈 对比分析**

未与其他方法做数值比较，论文通过解析推导和仿真展示了在稀疏奖励下课程显著加速学习、避免退出。

**⚠️ 局限性**

局限性包括模型仅为开环单步、线性策略、无表示学习、无值函数估计，且难以推广到更复杂的闭环、多步或深度学习环境。

---

## 109. Scaling Transformer Energy-based Models For High Quality Image Synthesis

**arXiv ID:** 2610.31889 | [PDF](https://arxiv.org/pdf/2610.31889v1)

**作者:** Jungsoo Lee `[一作]` (MIT), Yilun Du `[通讯]` (Harvard University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 Transformer Energy-Based Model (TEM)，在潜在空间对比式训练 EBM，生成高分辨率图像并支持组合生成、修复和 OOD 检测。

**💡 创新点**

通过多层能量提取、正向表示对齐 (PRA) 与可扩展采样，突破高维 EBMs 的尺度瓶颈。

**🔧 技术方法**

采用 Vision Transformer 主干、旋转位置编码、SwiGLU 激活、AdaLN 归一化、对比训练、Langevin MCMC、PRA 以及 Stable Diffusion VAE 的潜在空间。

**📊 数据集**

ImageNet（256×256/512×512）、CLEVR、iNaturalist、Textures、Places365、SUN 等数据集。

**📈 对比分析**

与 DiT‑XL/2、SiT‑XL/2 等扩散/EBM 基线对比，TEM‑XL 在无指导下 FID‑50K 3.82（256）/8.24（512），在组合生成和 OOD 检测上优于同类方法。

**⚠️ 局限性**

主要限制是采样计算成本高、长时间采样导致多样性下降，且缺乏高效的推理加速方案。

---

## 110. Cognitively Constrained Partners for Zero-Shot Coordination in Real-Time Partitioned Control

**arXiv ID:** 2610.31726 | [PDF](https://arxiv.org/pdf/2610.31726v1)

**作者:** Ash Simmons `[一作]` (University of Exeter), Richard M. Everson `[通讯]` (University of Exeter)

**通讯引用:** 9548 | [OpenAlex ID](https://openalex.org/A5016900158)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种基于人类生理约束的伙伴生成框架，并在分离控制的Lunar Lander环境中训练协作代理，以实现零射击协同。

**💡 创新点**

创新点在于用人类的反应延迟、机械延迟和感知阈值等生理学限制来产生多样化的伙伴，而非传统的奖励塑形或历史检查点，从而让训练代理更贴近真实人类行为。

**🔧 技术方法**

技术手段包括：多离散包装的持续Lunar Lander、基于指数加高斯分布的延迟与阈值心理学包装、递归PPO网络、能量距离评估以及与FCP、PBT、MARL的跨玩法对比。

**📊 数据集**

使用的数据集主要由20名人类参与者的10局单人试玩数据、45个认知约束伙伴、15个FCP线索、46个PBT伙伴以及另外50个未见伙伴组成。

**📈 对比分析**

比较方法为在交叉玩法下将每种代理与四类伙伴池（认知约束、PBT、FCP、MARL）进行匹配和持出测试；结果显示认知约束代理在未见伙伴时提升约+63分，整体平均回报与PBT/FCP相当，并且不受伙伴生成过程变化的影响。

**⚠️ 局限性**

局限性包括：未考虑人类随时间对伙伴的适应；伙伴仅观察物理状态缺失引擎信息；评估基于小规模人类样本；并且缺乏对真实人类团队表现的最终验证。

---

## 111. From Freehand Demonstrations to Autonomous Robotic Ultrasound: An Anatomy-Anchored Learning Framework

**arXiv ID:** 2610.32525 | [PDF](https://arxiv.org/pdf/2610.32525v1)

**作者:** Yuan Bi `[一作]` (Technical University Of Munich), Nassir Navab `[通讯]` (Technical University Of Munich)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出基于解剖锚定的模仿学习框架，将手持演示直接迁移到自主机器人超声扫描；

**💡 创新点**

创新点在于将探头放置规则化为解剖目标定位，采用锚定参考帧表达探头运动，结合流式流动策略与多模态感知实现机器人无须示范即可学习；

**🔧 技术方法**

使用解剖锚定姿态表示、手持演示采集平台、超声图+腕摄像+接触力三模态感知、Streaming Flow Policy、终止判定头和后处理最佳视图选择等技术；

**📊 数据集**

使用由16名志愿者提供的294条手持演示数据，包括肾、胆囊、肝和桡动脉的标准视图和扫查；

**📈 对比分析**

与微调的π0 Vision-Language-Action模型对比，任务成功率提升10-42个百分点，视图质量评分显著提高，停止成功率>93%，扫描时间、接触力和抖动均降低；在另一机器人上亦保持90%+成功率；

**⚠️ 局限性**

局限性包括未在患者病理样本上验证、对解剖可见性依赖、极端姿态旋转下性能下降，且未涉及多任务协同与临床交互。

---

## 112. Many Perturbations, Shared Failure Dynamics: Tracing Direction-Sensitive Perturbation Influence in Language Models

**arXiv ID:** 2610.32309 | [PDF](https://arxiv.org/pdf/2610.32309v1)

**作者:** Zhibo Hu `[一作]` (University of New South Wales), Liming Zhu `[通讯]` (University of New South Wales)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对 Qwen3-32B、Gemma 4‑31B 在 MMLU、ReClor 以及 MonitorBench 等任务中的语义重写与提示扰动，量化了内部表示差异，并通过窗口内正交分解、方向交换与补丁干预，揭示了保留度、变换与读出的几何关系，提出方向敏感的语义脆弱性预测与修复方案。

**💡 创新点**

创新点在于：①引入方向敏感、读出相对的几何描述，证明单一尺度化得分不足以捕获行为差异；②通过窗口内正交分解与定向补丁验证方向效应；③将语义脆弱性与隐藏 CoT 监测结合，提出读出相关的转向效应预测与定量补丁验证。

**🔧 技术方法**

主要技术包括：窗口内正交分解与方向交换、定向补丁干预、数值路径积分、梯度对齐、对数几率模型、AUROC 与 Brier 评估，以及对注意力层和 MLP 组件的定位实验。

**📊 数据集**

使用的数据集包括：MMLU、ReClor、RUPBench（六种重写族）以及 MonitorBench 的 logic、math、health 任务，并生成了 AI 过滤后的新 ReClor 变体。

**📈 对比分析**

比较方法采用 AUROC、Brier、恢复率、梯度与补丁效应相关性等指标；在 Qwen3-32B 上预测 AUROC 达 0.953，Gemma 0.910；ReClor 迁移 AUROC 0.978；方向补丁在失败案例中约 50‑70% 恢复；梯度与补丁效果相关性高达 0.97，但正负效应高度异质。

**⚠️ 局限性**

局限性包括：方向向量并非全局可转移，样本规模有限且缺少全覆盖实验；仅在固定窗口内补丁，未证明对完整提示的因果影响；CoT 监测采用固定生成，未直接证明真实隐藏推理；数值精度（bfloat16）限制了可实现的补丁幅度；缺乏跨模型统一验证。

---

## 113. Settling Nondeterministic Communication Complexity of Graph Connectivity

**arXiv ID:** 2610.32346 | [PDF](https://arxiv.org/pdf/2610.32346v1)

**作者:** Zhijun Zhang `[一作]` `[通讯]` (INSAIT), Zhijun Zhang (INSAIT)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

论文解决了图连通性的非确定性通信复杂度问题，证明了其下界为Ω(Nlog N)。

**💡 创新点**

创新点在于首次确定了图连通性的非确定性通信复杂度，并引入了一种新的通信问题，作为内积和指针追踪问题的组合。

**🔧 技术方法**

使用了上界1-单色矩形的最大大小的技术，并将其推广到加权、几乎1-单色矩形，从而得出随机协议的下界。

**📊 数据集**

论文中没有具体提到使用的数据集，但讨论了N个顶点的无向图的通信复杂度。

**📈 对比分析**

与之前的研究相比，之前的最优下界为Ω(Nloglog N)，而本研究证明了Ω(Nlog N)的下界，显示出显著的改进。对于随机协议，得到了Ω(N · log(min(1/ε,N)))的下界，适用于任意小的错误概率ε。

**⚠️ 局限性**

限制在于虽然证明了下界，但对于更复杂的协议或更高的轮数，仍然缺乏更强的下界证明。

---

## 114. A Resource-Efficient and Precision-Scalable FPGA Accelerator for Spiking Neural Networks Using Minifloat Arithmetic

**arXiv ID:** 2610.32052 | [PDF](https://arxiv.org/pdf/2610.32052v1)

**作者:** Ali Mehrabi `[一作]` (Western Sydney University), Philip C. Jose `[通讯]` (Western Sydney University)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文设计了一种可根据精度灵活配置的 FPGA 资源高效 SNN 加速器。

**💡 创新点**

创新点在于统一的可参数化 minifloat 数据路径，支持多种浮点格式且能在不改动神经元模型的前提下调节精度与硬件成本。

**🔧 技术方法**

采用时间复用的 LIF/LI 神经元引擎、深度流水线的 minifloat 加法/乘法器以及有效驱动调度机制，实现了低功耗、高频的可扩展硬件。

**📊 数据集**

使用 Poker‑DVS 事件记录数据集进行重复事件分类验证。

**📈 对比分析**

通过将 E5M6（12 位）与 32 位基线对比，CLB、LUT、FF 与动态功耗分别降低约 69.2%、54.4%、59.1% 与 46.0%，且分类精度保持 96.30%，Fmax 提升 18.5%。

**⚠️ 局限性**

局限在于极低精度（E4M3）会导致准确率下降，且当前实现未包含 BRAM/URAM 级别的权重存储。

---

## 115. Phronix: The Capability-Centric Learning Paradigm for Agents

**arXiv ID:** 2610.31789 | [PDF](https://arxiv.org/pdf/2610.31789v1)

**作者:** Hariharan Devarajan `[一作]` (Lawrence Livermore National Laboratory), Michela Taufer `[通讯]` (University of Tennessee)

**通讯引用:** 3821 | [OpenAlex ID](https://openalex.org/A5078387820)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种以能力为核心的学习范式，构建自学习的代理框架，使代理能够在执行过程中自动生成、验证并改进可执行能力；

**💡 创新点**

创新点在于将能力学习拆分为预测-验证-纠正循环，利用MCP工具与技能分离确定性与判断性，自动跟踪记录并将观察转化为可重复使用的工具与技能，从而实现完全无用户干预的持续学习；

**🔧 技术方法**

核心技术包括MCP（Model Context Protocol）工具接口、能力构建器（Capability Builder）用于生成工具与技能、跟踪器（Tracker）实现无缝记录与知识蒸馏、研究器（Researcher）进行基于进展的持续学习以及多层次能力存储（bundled、user、project）；

**📊 数据集**

评估使用SWE-bench Lite的30个软件工程任务以及1000 Genomes工作流的HPC I/O性能分析任务，数据集覆盖多种仓库与任务类型；

**📈 对比分析**

对比基线（无学习）、冷启动和热启动三种配置，发现热启动在所有模型下的解决率均提升至约90%，且平均每任务仅消耗约0.38–0.40百万token，显著低于单次基线尝试；

**⚠️ 局限性**

局限性包括对环境依赖的确定性问题（如Astropy任务表现差）、跨模型与跨框架的能力迁移尚未充分验证，以及在极端规模或高复杂度任务中可能需要更高的研究与验证开销。

---

## 116. $R^2E:$ A Role-driven Reward Evolutionary Framework for Automated Reward Function Design

**arXiv ID:** 2610.32439 | [PDF](https://arxiv.org/pdf/2610.32439v1)

**作者:** Shouhao Chang `[一作]` (Hunan University), Shigeng Zhang `[通讯]` (Central South University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了 R^2E 框架，实现 LLM 自动化奖励函数设计，通过角色驱动演化和全局精英池动态搜索。

**💡 创新点**

创新点包括：①角色驱动的演化结构（Explorer、Guardian、Evolver）实现多样化与稳定性并行；②动态调度策略让搜索从探索逐步过渡到利用；③全局精英池作为持续记忆，提升搜索稳定性；④将 LLM 生成过程结构化而非随机采样。

**🔧 技术方法**

主要技术：LLM（Qwen3-Max）生成奖励、RL 内部循环训练（IsaacGym、Bi-DexHands）、演化算子（交叉、变异）、全局精英池竞争接纳、动态调度机制。

**📊 数据集**

数据集：IsaacGym 6 任务（四足、四旋翼、机械手、夹持、机械臂等）以及 Bi-DexHands 8 任务，共 14 个复杂仿真环境。

**📈 对比分析**

实验将 R^2E 与 Sparse、Human、Eureka、REvolve、RF-Agent 进行比较，采用任务特定性能指标或成功率。R^2E 在所有任务上均表现最佳，尤其在操作任务上平均提升约 15%~20%，在 14 任务中的平均人类归一化得分最高。

**⚠️ 局限性**

限制：需手动设定角色比例、精英池大小和搜索预算；未利用失败奖励信息来指导搜索；停止准则不够系统；对 LLM 的提示和算力需求较高。

---

## 117. Empirically Calibrated Differential Games for Simulating Driver Interactions

**arXiv ID:** 2610.32276 | [PDF](https://arxiv.org/pdf/2610.32276v1)

**作者:** Amirmohammad Khakpour `[一作]` (Northwestern University), Nie `[通讯]` (Northwestern University)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并校准了一种基于差分游戏的道路交通模拟模型，用于模拟城市交叉口中异质驾驶员的互动。

**💡 创新点**

将差分游戏与经验校准相结合，首次在真实交叉口数据上验证了游戏模型能更准确地复现人类驾驶行为，并提供了可解释的行为参数。

**🔧 技术方法**

采用迭代线性-二次（iLQ）求解方法求解纳什均衡，使用贝叶斯优化对10个行为参数进行校准，并对延迟感知与冲突权重进行可识别性分析。

**📊 数据集**

使用TGSIM Foggy Bottom数据集（四个华盛顿特区交叉口的10 Hz视频轨迹）共819辆车进行训练，42辆车作保留集。

**📈 对比分析**

与SUMO、SMIRL及两种非游戏变体（独立OC、合作OC）在相同信息设置下进行对比，游戏模型在位置、速度和加速度误差上优于所有对手，特别是在仅给定路线的场景下误差降低4.3 m以上。

**⚠️ 局限性**

模型中部分参数（如冲突权重、延迟）在训练数据中难以辨识，需要额外的安全与舒适性目标进行设定；且对更紧密冲突场景的泛化及对非车辆交通参与者的扩展仍待验证。

---

## 118. Beyond Final Outcomes: Quantifying Claim-Level Information Dynamics in Multi-Agent LLM Interaction

**arXiv ID:** 2610.32333 | [PDF](https://arxiv.org/pdf/2610.32333v1)

**作者:** Haocheng Jiang `[一作]` (Xi'an Jiaotong University), Minnan Luo `[通讯]` (Xi'an Jiaotong University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了多智能体LLM交互中的信息动态，提出RIVER框架从收敛、探索、保留三维度评估声明级信息流。

**💡 创新点**

提出将声明拆分、语义对应、立场对齐结合的RIVER评估框架，并在新闻生成任务上系统比较五种交互策略。

**🔧 技术方法**

基于LLM进行声明拆分、语义相似度评估、立场得分；使用聚类、阈值匹配计算U、CC、ER、OR、SR、SL等指标。

**📊 数据集**

使用40条简短新闻（来自20份Reuters报道），并用DBS-eval、DELPHISTAR、Polis等人类标注数据验证收敛度。

**📈 对比分析**

对五种交互策略（Free、NGT、Delphi、Policy Delphi、Brainwriting）在Full与Framework两设置下进行量化对比；结果显示聚类结构提升收敛、缩小信息空间，分布式Relay扩大探索；额外约束提升探索但削弱收敛与保留。

**⚠️ 局限性**

仅使用单一LLM与固定五名代理，未考察模型能力与代理多样性；每轮讨论不保持历史，无法捕获跨轮信息累积。

---

## 119. RxnExBench: Benchmarking Vision-Language Models for Structured Chemical Reaction Extraction from Scientific Literature

**arXiv ID:** 2610.32221 | [PDF](https://arxiv.org/pdf/2610.32221v1)

**作者:** Jiaxing Sun `[一作]` (Shanghai Artificial Intelligence Laboratory), Conghui He `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出RxnExBench基准，评估Vision‑Language模型在从Screen Tables和Substrate Scopes中提取完整结构化化学反应数据的能力。

**💡 创新点**

创新点包括统一的26字段反应架构、三层难度分类、模块化高质量标注流程以及对分子识别与高阶推理的分离评估。

**🔧 技术方法**

主要技术为多模态视觉‑语言推理、分子结构识别（OCSR）以及层级化推理任务设计。

**📊 数据集**

使用了3565条人工标注反应、2361条SMILES，涵盖90个Substrate Scope和176个Screen Table图像，按三难度等级划分。

**📈 对比分析**

与9款先进VLM（如GPT‑5.5、Gemini‑3.1 Pro、Claude Opus等）对比，最优模型在Criterion 1下F1仅达23.16%，在Criterion 3下更低；使用Oracle‑MolRec提升至42.67%但仍无法达到高精度。

**⚠️ 局限性**

局限在于仅覆盖英文有机合成文献、严格的反应级匹配标准，难以反映模型的真正上限，未来需扩展领域与改进评估方式。

---

## 120. RAISE-Vision: An Automated Contact Angle Measurement Platform for Contact Lens Wettability Characterization

**arXiv ID:** 2610.32188 | [PDF](https://arxiv.org/pdf/2610.32188v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 121. Multi modal perceived conversation success prediction for spontaneous autistic and non-autistic dyadic interactions

**arXiv ID:** 2610.31682 | [PDF](https://arxiv.org/pdf/2610.31682v1)

**作者:** Thanushi Withanage `[一作]` (University of Maryland), Carol Espy-Wilson `[通讯]` (University of Maryland)

**通讯引用:** 4567 | [OpenAlex ID](https://openalex.org/A5078241735)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文通过提取发音器官协调、面部动作、声学韵律和交谈节奏等多模态特征，构建多模态模型来预测对话双方对对话成功的主观评价。

**💡 创新点**

创新点在于首次将发音器官协调特征与传统的面部、声学和节奏特征联合使用，并在跨不同年龄、神经类型和录制环境的三组数据集上验证模型的可迁移性，证明交谈节奏是最具通用性的预测因子。

**🔧 技术方法**

采用语音F0和强度提取、OpenFace面部动作单元同步计算、发音器官协方差矩阵及其特征谱、PCA降维、Ridge回归等技术，并使用五折嵌套交叉验证来选择正则化参数。

**📊 数据集**

使用了CANDOR（500条成人Zoom对话）、UTD（62对话，包含自闭症与非自闭症混合）、UMD（12条成人面对面对话）三大数据集。

**📈 对比分析**

与单模态、单个模态剔除、训练均值基线进行对比。CANDOR内部MAE从基线0.1143降至0.1089（4.7%）；跨数据集预测时，UTD MAE下降13.9%，UMD下降约1%；相关系数均为正，且交谈节奏单独模型在UTD上表现最好。

**⚠️ 局限性**

局限性包括仅研究两人对话、不同问卷需PCA预处理导致比较难度、模型对不同神经类型匹配的解释性有限、发音器官协调特征在不同语料中的稳定性尚待进一步验证。

---

## 122. SPA: Structure-Preserving Approximation of Gated Activations under FHE

**arXiv ID:** 2610.32065 | [PDF](https://arxiv.org/pdf/2610.32065v1)

**作者:** Rui Li `[一作]` (Chinese Academy Of Sciences), Weijie Miao `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出结构保持近似（Structure‑Preserving Approximation, SPA），在全同态加密（FHE）下高效近似 GELU、SiLU 等激活函数，保持奇数部分精确，只用偶数残差多项式逼近。

**💡 创新点**

创新点在于利用激活函数 A(x)=xg(x)，g(-x)=1-g(x) 的对称结构，保留线性奇数项，显著减少多项式次数与乘法深度；同时提供最佳一致逼近的理论证明和数值误差上界。

**🔧 技术方法**

使用 Chebyshev 多项式、离散最小化拟合、Paterson–Stockmeyer 评估分块、离散最小化程序（SciPy/HiGHS）和 CKKS 同态计算实现；并通过离散网格误差校验和系数上界给出误差保证。

**📊 数据集**

在 Gemma‑4‑E2B 和 Llama‑3.2‑3B 两个大型语言模型上，使用 HellaSwag、MMLU、Social‑IQa、RACE‑high、WikiText‑2、C4‑en 等标准评测数据集验证模型性能。

**📈 对比分析**

与 MOAI、NEXUS、Euston、THOR、Orion、Sylph 等现有 FHE 激活近似方法对比；SPA 在 [-5,5] 区间实现了相对 MOAI 的 3.25× 加速、QuickGELU 的 10.40× 加速以及 SiLU 的 2.06× 加速，同时保持 RMSE 与对手相当或更低；在更大区间 [-20,20] 仍保持较低误差与可观加速。

**⚠️ 局限性**

局限性包括仅适用于结构满足 A(x)=xg(x)、g(-x)=1-g(x) 的激活函数；误差上界依赖离散网格与系数舍入，未给出全范围统一保证；模型性能仅在两款 LLM 上验证，未覆盖其他网络结构；加速收益基于单激活评估，整体推理加速需进一步评估。

---

## 123. Hyperparameter-Free Boundary-Interior t-vMF Dice Loss for Medical Image Segmentation

**arXiv ID:** 2610.32266 | [PDF](https://arxiv.org/pdf/2610.32266v1)

**作者:** Hinako Mitsuoka `[一作]` (Meijo University), Kazuhiro Hotta `[通讯]` (Meijo University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了一种自适应边界-内部的 t‑vMF Dice 损失（STB‑tvMF Dice），通过上一 epoch 的均值余弦相似度来无手动调参地为边界和内部区域分别更新形状参数，从而在训练中自我调节损失函数的形状。

**💡 创新点**

创新点：① 在不使用验证集的前提下，用训练统计直接确定形状参数；② 将同一类别的边界与内部区域分离并各自调优；③ 通过最大化局部斜率（灵敏度）来决定形状参数，形成一种无手动搜索的自适应规则。

**🔧 技术方法**

使用技术包括：t‑vMF Dice 损失函数、基于 3×3 邻域的边界提取、均值余弦相似度统计、梯度/斜率分析自适应规则、epoch‑wise 参数更新、PyTorch 实现训练和多种基准损失（交叉熵、Dice、Tversky、Focal Tversky 等）的对比。

**📊 数据集**

使用数据集：Kvasir‑SEG（息肉分割）、ACDC（心脏 MRI 分割）、Synapse（腹部多器官 CT 分割）。

**📈 对比分析**

在 U‑Net、TransUNet、EMCAD 三种架构上进行交叉验证，对比交叉熵、Dice、Tversky、Focal Tversky、TopK Dice、ABeDice、CE+Dice、CE×Dice、Adaptive t‑vMF Dice 等损失；STB‑tvMF Dice 在大多数设置下取得最高或近似最高的平均 Dice，Synapse 上比 Adaptive t‑vMF Dice 提升约 7.7%，表明在不需要手动调参的情况下仍能保持或超越传统基线性能。

**⚠️ 局限性**

局限性：实验仅在 2D 分割任务中验证，未评估 3D 或更大尺度的适用性；对极少出现类别的鲁棒性有限；依赖固定的 3×3 邻域边界提取，可能不适用于所有图像分辨率或结构尺度；进一步研究需要探讨其在更广泛医学影像任务中的泛化能力。

---

## 124. Compact-Bench: Measuring What Context Compaction Costs Long-Horizon Coding Agents

**arXiv ID:** 2610.31710 | [PDF](https://arxiv.org/pdf/2610.31710v1)

**作者:** Ruopei Sun `[一作]` (University of Science and Technology of China), Houqiang Li `[通讯]` (University of Science and Technology of China)

**通讯引用:** 30209 | [OpenAlex ID](https://openalex.org/A5078141810)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 Compact‑Bench，针对编码代理在超长上下文中进行压缩后能否顺利继续工作进行评估。

**💡 创新点**

创新点在于：①使用“paired task continuation”比较完整历史与压缩摘要的后续执行；②引入“Recurring‑Detail Retention”与“Production‑Chain Retention”两种细节保留测度，能分离信息丢失与执行失败的关系；③通过多种摘要模型与提示（prompt）系统的交叉对比，揭示模型与提示对任务保留的不同影响。

**🔧 技术方法**

技术方法包括：多种 LLM 摘要后端（14 个 backbone）、多种压缩提示（5 个 prompt）、自动化执行脚本对 SWE‑bench 修复任务的测试、结构化细节提取器（路径、SHA、错误类型等）以及基于执行结果的稳定性判定。

**📊 数据集**

数据集：35 条本地化 SWE‑bench 修复实例（共 7 个开源仓库），146 条真实或仿真代理会话，106 条生产链（共 224 个压缩边界）。

**📈 对比分析**

比较方法：对同一历史与仓库快照，分别使用完整历史（PRE）与摘要（POST）进行 8 步限制的继续执行，并记录 PASS/FAIL；对细节保留使用提取器计算交叉匹配比例。结果显示摘要模型间稳定损坏率从 5%（Claude Opus 4.8）到 64%（DeepSeek‑Flash）不等；在提示比较中，Kimi Code 在任务保留上最高（19/22），Claude Code 在细节保留上最高（15.0%）。

**⚠️ 局限性**

局限性：仅评估了 35 个修复实例、固定的继续执行预算和单一代理；细节保留仅关注可提取的结构化信息，未覆盖隐式上下文；生产链样本规模随压缩次数急剧下降；使用的私有会话数据被匿名处理，可能缺乏更广泛的通用性。

---

## 125. IGG: A Benchmark for Interactive GUI Grounding under Visibility Constraints

**arXiv ID:** 2610.32536 | [PDF](https://arxiv.org/pdf/2610.32536v1)

**作者:** Kyeongseon Kim `[一作]` (Korea Advanced Institute of Science and Technology), Tae-Hyun Oh `[通讯]` (Korea Advanced Institute of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了Interactive GUI Grounding (IGG) 基准，评估代理在需要先交互使目标可见后再定位点击的能力。

**💡 创新点**

将GUI grounding拆分为可见性恢复与静态定位两个可测量步骤，构造可重现的交互基准并公开评估体系。

**🔧 技术方法**

采用视觉‑语言模型 (VLM)、监督式微调 (SFT)、在线强化学习 (GRPO) 等技术。

**📊 数据集**

601个任务，覆盖16种可控 GUI 环境，按七个可见性约束子类型划分。

**📈 对比分析**

与人类（94.6%）和商业 GPT‑5.5（79.4%）对比，开源 VLM 表现低至31.6%，通过 SFT+GRPO 可提升至约59% GA，显示显著改进。

**⚠️ 局限性**

仅使用可控模拟环境，缺乏对真实界面的直接评估，且可能被滥用于自动交互。

---

## 126. From Tabletop to Walking Bot: Deploying a VLA Model on a Legged Manipulator Robot

**arXiv ID:** 2610.32271 | [PDF](https://arxiv.org/pdf/2610.32271v1)

**作者:** Giovanni Minelli `[一作]` (Istituto Italiano di Tecnologia), Claudio Semini `[通讯]` (Istituto Italiano di Tecnologia)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

将预训练的视觉‑语言‑动作（VLA）模型迁移到四足行走机器人上，实现闭环行走‑操作；

**💡 创新点**

创新点在于通过共享演示/部署接口，将VLA模型从固定基座臂扩展至四足移动平台，并证明数据覆盖对新行为的提升；

**🔧 技术方法**

使用MolmoAct2 VLA模型、低秩适配、动作序列预测、基于动作空间的端到端控制以及深度摄像头感知；

**📊 数据集**

使用三组演示数据集（D1 131条，D2 233条，D3 258条），包含抓取与推送的抓取‑放置任务，配合语言指令与多视角视觉；

**📈 对比分析**

通过离线对比和硬件实验，比较不同数据集下的末端执行器误差；抓取误差从68.7mm降至48.2mm，推送误差下降到67mm，展示了数据量提升带来的性能提升；

**⚠️ 局限性**

局限性主要在于数据覆盖不足，尤其是初始身体放置阶段；模型容量不足以弥补，新的推送行为需要额外演示，且实时部署中存在感知遮挡导致失败。

---

## 127. Evidence-Grounded SLM Behavioral Prediction Architecture for Future Assistance Use

**arXiv ID:** 2610.32193 | [PDF](https://arxiv.org/pdf/2610.32193v1)

**作者:** Adeyemi Adeseye `[一作]` (University of Turku), Jouni Isoaho `[通讯]` (University of Turku)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文设计并实现了四层 ELBP 架构，利用本地小语言模型预测学习者在未来任务中是否需要请求帮助。

**💡 创新点**

创新点在于引入了双视图行为表示、证据赋值与验证机制，以及可选的推理拒绝（abstention），使模型既能本地运行又能对预测结果进行可解释性检验。

**🔧 技术方法**

使用了 Python 的事件预处理、双视图结构化表示、指令微调的小语言模型（如 Qwen2.5-3B、Llama‑3.2‑1B、Gemma‑2‑2B）以及基于证据的后验验证和覆盖率控制。

**📊 数据集**

实验数据来自 EdNet‑KT4，包含 131M 条细粒度学习交互日志，用以构建预测标签并评估模型。

**📈 对比分析**

与原始事件推理、仅结构化特征、双视图、以及 XGBoost 基线进行对比，Qwen2.5‑3B+ELBP 在全量 F1=0.80、选择性 F1=0.84、PR‑AUC=0.84 上明显优于所有基线，提升幅度超过 15% 以上。

**⚠️ 局限性**

研究局限在于仅针对帮助请求这一单一行为目标进行验证，缺乏对其他交互域和多样化预测任务的泛化评估。

---

## 128. Cautious Conflict-Free Inverse Gradient Methods

**arXiv ID:** 2610.31855 | [PDF](https://arxiv.org/pdf/2610.31855v1)

**作者:** Damian Hattler `[一作]` (Technical University of Munich), Nils Thuerey `[通讯]` (Technical University of Munich)

**通讯引用:** 7650 | [OpenAlex ID](https://openalex.org/A5047248117)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Cautious ConFIG 方法，通过对 Adam 输出进行谨慎掩码处理，恢复 ConFIG 在自适应优化器中的冲突自由性，从而改进物理信息神经网络（PINN）的梯度冲突管理。

**💡 创新点**

创新点在于仅用三行代码实现轻量级扩展，无需额外超参数即可在 ConFIG 与 Adam 等自适应优化器组合时恢复冲突自由性，并显著提升训练性能。

**🔧 技术方法**

使用了 ConFIG、谨慎更新（cautious update）技术以及 Adam 优化器，并在 PINN 框架中对多任务损失的梯度进行冲突消除。

**📊 数据集**

在三类 PDE 基准上测试：1D 可粘性 Burgers 方程、1D 非线性 Schrödinger 方程和 2D Kovasznay 流；采样点为边界与内部点。

**📈 对比分析**

与标准 Adam、Adam+ConFIG 以及 Adam+Cautious 进行对比，采用最低均方误差（MSE）作为评估指标；Cautious ConFIG 在三种任务中分别提升 19%–28% 的 MSE，并显著降低方差。

**⚠️ 局限性**

局限性包括仅在 PINN 场景验证，缺乏对其他自适应优化器的理论证明，模型依赖冲突轴假设且未考虑预条件化与内部状态的完整动态，且未在大规模真实数据集上进行验证。

---

## 129. Mining Frames: Association Rule Mining as an Unsupervised Framework for News Framing Detection

**arXiv ID:** 2610.31874 | [PDF](https://arxiv.org/pdf/2610.31874v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 130. Not All Information Is Worth Asking For: Decision-Informed Information Seeking with LLMs

**arXiv ID:** 2610.32181 | [PDF](https://arxiv.org/pdf/2610.32181v1)

**作者:** Seunghan Lee `[一作]` (LG AI Research), Wonbin Ahn `[通讯]` (LG AI Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 DIVA 方法，利用 LLM 仅推断一个小型决策模型，基于信息的决策价值和成本来确定何时询问和何时停止；

**💡 创新点**

创新点在于将 LLM 的知识与决策计算分离，采用基于决策价值的检查评估，并推出不确定性感知版本 DIVA-U；

**🔧 技术方法**

使用技术包括 LLM 提示、贝叶斯推理、期望信息价值（EIG）与决策价值对比、动态规划（DIVA-DP）以及高斯赫尔姆特积分的近似；

**📊 数据集**

实验数据集涵盖 12 个决策环境：8 个真实的 SEC、信用、临床记录 以及 4 个全指定可靠性的合成域，评估 20 个开源 LLM（Qwen3、Qwen2.5、Llama‑3.1/3.3、Gemma‑2/3、Mistral、Phi‑4、gpt‑oss、DeepSeek‑R1‑Distill）；

**📈 对比分析**

与 5 类基线（提示法、决策感知提示、EIG、PAL、永不/永询）比较，DIVA‑U 在所有 LLM 上实现最低 regret，控制域下平均降低约 50%，真实域平均降低约 30%；

**⚠️ 局限性**

局限性包括：需要预先给定收益阈值，假设检查互相独立，依赖 LLM 对可靠性的准确性，未充分处理多步决策中的不确定性收缩，实验仅覆盖英文和有限 LLM 规模。

---

## 131. Multimodal MPC via Bayesian Set Optimization for Dynamic Manipulation of Deformable Linear Objects

**arXiv ID:** 2610.31852 | [PDF](https://arxiv.org/pdf/2610.31852v1)

**作者:** Youyuan Long `[一作]` (Istituto Italiano di Tecnologia), Arash Ajoudani `[通讯]` (Istituto Italiano di Tecnologia)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了 BM2PC，一种基于贝叶斯集优化的多模态 MPC 框架，用于在障碍和动力学不确定性下实现弹性线性物体（如绳索、软管）的动态操纵。

**💡 创新点**

创新点包括：① 将多模态规划表述为贝叶斯集优化，直接在控制序列集合上优化后验密度和行为多样性；② 设计了增量式多模态更新算法，在保持多种低成本控制模式的同时仅产生极小的计算开销；③ 引入执行动态模型，对机器人执行误差进行预测补偿，提升高速动态操纵的可靠性。

**🔧 技术方法**

核心技术包括：控制-as-inference 视角的概率推理；采样式 MPC（类似 MPPI）与增量式多模态更新；轻量级质量弹性线性对象（DLO）质量-弹簧-阻尼模型；GRU+MLP 的执行动态预测模型；GPU 加速的并行滚动仿真。

**📊 数据集**

实验使用 MuJoCo 3.2.2 进行仿真，硬件实验在 Franka Emika Panda 搭配 Franka Hand、OptiTrack 视觉感知、手工制作的绳索和圆柱形障碍；数据集包括 1.2+ 兆条机器人轨迹，用于训练执行动态模型。

**📈 对比分析**

与 SPiD、C‑MPPI、SVMPC 等基线对比：在轨迹跟踪任务中 RMSE 下降 44%；在障碍规避任务中，规划时间比现有多模态 MPC 方法低 29%–66%；在扰动抑制任务中能在 <1 s 内恢复误差 <5 cm。实验表明 BM2PC 在未见轨迹和动态障碍下具有更好的泛化和更高的实时性能。

**⚠️ 局限性**

局限性：① 需要外部感知提供绳索状态与障碍几何；② 目前仅测试单端抓取，未验证多端或多绳系统；③ 执行动态模型在极端速度/加速度下仍可能产生误差；④ 对不同材质、长度的绳索需重新校准物理参数；⑤ 真实环境中视觉遮挡导致的感知噪声仍是主要挑战。

---

## 132. ALLMTS: Automated LLM-based Medical Triage System

**arXiv ID:** 2610.32469 | [PDF](https://arxiv.org/pdf/2610.32469v1)

**作者:** Michał Remigiusz Janiszewski `[一作]` (Vstorm), Bartosz Gonczarek `[通讯]` (Vstorm)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了ALLMTS——一种通过LLM执行已存在临床指南而非直接做决策的电话医学分诊系统

**💡 创新点**

创新点在于把LLM的角色限定为执行可审计的指南问答流程，利用“首正答案”规则保证可追溯的决策；并将指南检索、问答与SMAG重定向结合成完整流水线

**🔧 技术方法**

使用Azure OpenAI LLM、基于语义检索的指南候选筛选、结构化问答模板、闭世界二值答案与布尔逻辑、以及SMAG（See More Appropriate Guideline）递归重定向

**📊 数据集**

采用了566例专家校准的合成临床情境（涵盖14个分诊结论），并公开了其中50例作为benchmark

**📈 对比分析**

与两类基线（原始LLM、Web检索）以及指南上下文消融相比，ALLMTS在所有评估维度上显著领先：分诊准确率0.95（全集）/0.98（50例），过度分诊率从0.44降至≤0.02，稳定性和一致性也更高

**⚠️ 局限性**

局限性包括依赖特定商用指南导致循环评估、闭世界假设不区分未提及与否定信息、仅适用于成人门诊场景、缺乏真实电话录音的验证，以及未对每个子阶段做细粒度消融

---

## 133. Patronizing and Condescending Language Detection in English and Mandarin

**arXiv ID:** 2610.31808 | [PDF](https://arxiv.org/pdf/2610.31808v1)

**作者:** Xinyi Vicky Xiang `[一作]` (University of Washington), Paula Cortés Lemos `[通讯]` (University of Washington)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并实现了两种自动化检测偏颇与居高临下语言（PCL）的系统：一种基于静态词向量和CNN的分类模型，另一种基于预训练BERT的微调模型，并探讨了数据预处理、数据平衡和增广策略。

**💡 创新点**

创新点在于：①将BERT模型微调用于PCL二分类任务并取得显著提升；②针对PCL细微性设计了多类别子任务与多维度特征（如词汇删改、同义替换）尝试增广；③在中文语料上扩展任务，展示跨语言适用性。

**🔧 技术方法**

使用技术包括：GloVe静态词向量、卷积神经网络（CNN）、BERT-base-uncased微调、Adam/AdamW优化器、线性学习率调度、随机删除与同义词替换等数据增广方法。

**📊 数据集**

主要数据集是“Don't Patronize Me!”英文PCL数据集（约10,469条样本，二分类标签）以及中文CCPC（微博、知乎）数据集；实验集中在前者的训练集、开发集（10%）上。

**📈 对比分析**

比较方法：在开发集上评估precision、recall、F1。静态词+CNN表现最差（F1≈0.13），而BERT微调在2个epoch时达到F1≈0.53，明显优于CNN和Baseline；增广方法未显著提升性能。

**⚠️ 局限性**

限制包括：①PCL标签主观性高，导致数据噪声；②数据严重不平衡，增广方法效果有限；③CNN模型受限于词向量表达，无法捕捉上下文细微差别；④模型在不同语言、领域泛化能力待验证；⑤过拟合风险仍需更严谨的验证策略。

---

## 134. Hierarchical GTV Estimation: Bridging the Gap between Container-Level Ranking and Item-Level Conversion

**arXiv ID:** 2610.32420 | [PDF](https://arxiv.org/pdf/2610.32420v1)

**作者:** Zerong Lan `[一作]` (Meituan), Xingxing Wang `[通讯]` (Meituan)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了Hierarchical GTV Estimation (HGE) 框架，用于解决基于容器的广告系统中容器级排名与物品级转化之间的粒度不匹配问题，改进了 GTV 的预测与估计。

**💡 创新点**

创新点包括：① 采用 Hybrid Set‑Aware Encoder (HSAE) 对动态物品集合进行 Transformer‑级别的建模；② 引入 Proxy‑Label Learning (PLL) 将高方差的 GTV 目标拆分为转化概率、购买数量与单价等子任务；③ 设计 Residual Correction and Ensemble Fusion (RCEF) 模块，通过残差纠正和多模型融合缓解标签稀疏与偏差。

**🔧 技术方法**

使用技术包括：Transformer + 多头自注意力与位置编码；多任务学习（BCE + Huber）；DIN、DCN 进行特征交叉；共享 MLP 对每个物品进行预测；门控网络动态调节 RCEF 加权；以及整体的深度特征嵌入与残差连接。

**📊 数据集**

实验数据来自美团广告系统日志，包含 712M 条记录（121M 用户，949K POI），采用严格的时间外切分（5% 作为测试），并在 13 天的线上 A/B 测试中验证效果。

**📈 对比分析**

与 WLR、DLRM、RankMixer、CREAD 等基线对比，HGE 在离线 XAUC 上提升至 0.8158（比最佳基线高 0.63pp），PCOC 达 0.9858，MAE 下降至 4.99；线上 A/B 测试中 RPS 提升 1.42%，均表现显著优于现有方法。

**⚠️ 局限性**

局限性包括：① 对物品级标签稀疏性仍有一定依赖，部分高阶物品信息可能缺失；② 模型结构复杂，Transformer 及多任务训练对计算资源要求较高；③ 主要验证于美团数据，跨平台泛化能力尚未充分评估；④ 仍需在更大范围的多场景中检验其稳健性。

---

## 135. Distributed Swarm Morphogenesis with Neural Cellular Automata

**arXiv ID:** 2610.31900 | [PDF](https://arxiv.org/pdf/2610.31900v1)

**作者:** Ray Verma `[一作]` (New York University Abu Dhabi), Anthony Tzes `[通讯]` (New York University Abu Dhabi)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种基于神经细胞自动机（NCA）的分布式群体形态生成框架，能在部署时无先验形状列表地通过局部交互自适应形成大规模网格结构。

**💡 创新点**

创新点在于（1）引入分布式附件点发现机制，使机器人在组装过程中动态生成可附着位置；（2）使用离线稀疏保留微调和假正例惩罚，解决传统NCA与物理组装之间的动态不匹配；（3）将NCA与现有的分布式分配与碰撞感知控制模块解耦，实现模块化。

**🔧 技术方法**

采用Growing Neural Cellular Automaton进行形状重建，离线稀疏保留微调与假正例损失进行优化；使用人工光场（ALF）分配算法和约束控制器完成机器人运动控制；基于局部格子框架估计实现局部坐标变换。

**📊 数据集**

使用四种预定义的网格形状（Tree、Lizard、Hand‑Heart、E）作为参考图像，配合仿真中的随机初始位置和真实Crazyflie机器人的实验数据。

**📈 对比分析**

与标准Growing NCA基线进行消融比较，F1分数由0.583提升至0.988，误差显著下降；在250机器人规模下，结构收敛时间随机器人数量线性增长，最终F1接近1，验证了可扩展性。

**⚠️ 局限性**

局限性包括：实验中依赖外部定位系统（Lighthouse），缺乏真正的相对位置感知；每种目标形状需单独训练NCA，迁移到新形状需要重新训练；在高噪声或通信不稳定环境下的鲁棒性尚未充分验证。

---

## 136. The LLENS Platform for On-Premises Large Language Model Agents in a Japanese University Hospital: Development and Evaluation Study

**arXiv ID:** 2610.32151 | [PDF](https://arxiv.org/pdf/2610.32151v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 137. When and How Does Muon Precondition? The Role of Structured Gradient Noise

**arXiv ID:** 2610.31728 | [PDF](https://arxiv.org/pdf/2610.31728v1)

**作者:** Xiaoran Chen `[一作]`, Pinyan Lu `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了 Muon 优化器在随机梯度噪声下的预条件化优势，探讨了噪声结构、采样与动量对学习方向性的影响，并提出了基于曲率的快慢动量设计，验证了其在 0.12B 参数密集式语言模型预训练中的有效性。

**💡 创新点**

创新点在于：① 在低 SNR 等向噪声下证明 Muon 的一阶均值更新等价于学习率缩放的 SGD，揭示样本级谱归一化不足以产生方向性预条件化；② 通过两任务平衡阈值与多任务采样分析揭示采样频率对学习速度的非线性调节；③ 在标签噪声下引入二次采样尺度并给出平方根频率依赖的闭合预测；④ 对比 Muon 与 SignSGD 的旋转负迁移现象；⑤ 将动量视作方差降低与跟踪误差的折中，提出快慢动量方案。

**🔧 技术方法**

主要技术包括：Muon's 部分极化映射、低维低 SNR 解析、关联记忆模型与采样理论、标签噪声下的闭合分析、动量的指数加权平均与跟踪误差分解、以及曲率感知的快慢动量混合实现。

**📊 数据集**

使用了：① 受控的合成关联记忆数据集（多任务、不同频率和标签噪声的矩阵分类任务）；② 大规模密集式语言模型预训练数据（未公开指定，常见公开大语料库如 Common Crawl 等）。

**📈 对比分析**

与标准 Muon 对比，在 0.12B 参数的密集式 LLM 预训练中，快慢 Muon 在 30,000 步时的验证损失从 3.0097 降至 2.9888，下降 0.0209，表明该异构动量方案能提升收敛速度和最终泛化性能。

**⚠️ 局限性**

局限性包括：① 理论分析主要基于正交/矩阵表示，未覆盖非正交或非线性模型；② 仅在单一 0.12B 模型与少量实验验证，缺乏跨规模、跨任务的广泛评估；③ 对动量窗口选择给出局部设计指导，但缺乏全局收敛证明；④ 关联记忆实验为合成数据，实际 NLP 任务的噪声与频率分布可能更为复杂。

---

## 138. Minimum Description Length Unsupervisedly Separates Law-Grade from Statistical-Grade Knowledge in a Self-Improving Science Agent - and Reveals 86% of Its Own Discoveries Are Statistical

**arXiv ID:** 2610.32245 | [PDF](https://arxiv.org/pdf/2610.32245v1)

**作者:** Wenbin Mao `[一作]` (Hooman Biotechnology Co., Ltd.), Pei Zou `[通讯]` (Hooman Biotechnology Co., Ltd.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `fede83ac-7505-405f-ab37-e7284695c47f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `14d48e9d-0069-4ad9-996a-1d5968216998` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

审计了一套自我改进的科学发现代理的知识账本，利用其已实现的压缩比作为无监督的认识学评判器，识别出真理法则与统计规律的区别。

**💡 创新点**

证明在实际运行的发现系统中，离散两部分压缩计数（MDL）可以自动区分确定性物理法则和统计关联，并揭示系统先前错误将统计规律视为物理法则的事实。

**🔧 技术方法**

采用离散两部分描述长度编码、最小描述长度（MDL）理论、基于 R^2 的预测门控以及对账本的重排序与超几何检验等技术。

**📊 数据集**

使用公共测量档案（行星黄道、光谱表、晶体学数据库、气候/地球物理记录）和蛋白质二级结构预测的序列数据。

**📈 对比分析**

通过对 252 条已通过预测门的关系按压缩比排序，发现前段完整覆盖已知经典法则，统计显著性 p=3.6×10⁻¹²；同时检测到 86% 的关系压缩比≈1，表明大多数被误认为法则的是统计规律。

**⚠️ 局限性**

局限性在于仅基于单一系统单一账本（n=252），标签协议人为设定；评判基于测量精度而非自然规律；未在其他账本或多系统上验证，且低精度法则的判定存在不确定性。

---

## 139. Shorter Is Not Cheaper: A Controlled Comparison of Declarative Animation Specifications and Free-Form Code for LLM-Generated Explanations

**arXiv ID:** 2610.32129 | [PDF](https://arxiv.org/pdf/2610.32129v1)

**作者:** Jiantang Huang `[一作]` `[通讯]`, Jiantang Huang

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文在受控实验中比较了大型语言模型（Claude 5 系列与 Qwen3-14B）生成的可交互解释的两种输出方式：一种是自定义的时序声明式动画格式 ExplainSpec，另一种是完整的自由形式 HTML/JavaScript 页面。研究设计了观察合同和可验证的检查器，收集了 20 个跨 5 个领域（物理、算法、协议、机器学习、概率）的解释请求，并对每个模型在两种格式下的生成进行多轮实验。

**💡 创新点**

创新点在于：①提出了 ExplainSpec 这一时间索引的声明式动画规范及其统一观察合同，使自由形式与声明式输出可用同一套检查器进行公平评估；②首次在可交互解释任务上用“思考”模型进行受控对比，系统揭示输出压缩并不必然带来生成成本或可靠性的降低；③对模型在两种格式下的推理行为、输出长度、生成时延与正确率进行量化分析。

**🔧 技术方法**

技术与方法：使用 Claude-opus、Claude-sonnet、Claude-haiku（分别对应不同参数规模）与 Qwen3-14B（开放权重、可开启/关闭思考）作为生成模型；构建 ExplainSpec 运行时（JavaScript/SVG）和检查器框架；利用 Playwright 在无网络环境下加载并评估每个生成结果；对生成文本进行 token 统计、思考 token 计数；统计生成时延、首次渲染时间与合同调用时间；采用 Wilcoxon 符号秩检验、Bootstrap CI 与 Holm 校正进行统计分析。

**📊 数据集**

数据集与任务：共 20 个解释请求，按 5 个领域分配，每个请求包含 2–3 个参数设置，涉及的概念包括抛物运动、快速排序、TCP 三次握手、梯度下降等；任务描述中明确要求的状态变量与可调参数，使自由形式与声明式输出在内容上保持一致。

**📈 对比分析**

比较方法与性能：采用任务对任务（paired）方式，计算两种格式的总输出 token、答案 token、思考 token、生成时延；统计有效性（valid）与所有语义测试通过率（all tests pass）。结果显示：答案 token 在声明式格式下平均缩短 25–48%，但总 token 仅在最大的 Claude-opus 模型显著下降，其他模型几乎无变化；生成时延无显著改善；有效率方面，自由形式 HTML/JavaScript 的首次生成有效率达 95–100%，而声明式格式首次有效率仅 85–95%，修复后仍低于自由形式；语义通过率在声明式格式下普遍较低。

**⚠️ 局限性**

局限性：①仅对两类模型（Claude 与 Qwen3-14B）进行评估，无法推广到更大或更小规模模型；②声明式格式仅通过提示学习，未进行 fine‑tune 或预训练，导致模型对新语法的不熟悉与错误；③任务数量有限（20 个），可能不足以捕捉更细微的差异；④检查器仅评估语义正确性，未覆盖视觉质量或交互易用性；⑤生成过程未使用流式渲染或缓存，未反映实际系统中的性能；⑥部分错误被运行时隐藏或修复，导致结果可能偏差。

---

## 140. ImageRSI: Verification-Driven Recursive Self-Improvement for Agentic Image Generation

**arXiv ID:** 2610.32120 | [PDF](https://arxiv.org/pdf/2610.32120v1)

**作者:** Shuo Han `[一作]` (University of Science and Technology of China), Xike Xie `[通讯]` (University of Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出ImageRSI框架，冻结图像生成器，结合五维验证与局部修复、跨任务经验库与离线评判器自进化，实现复杂文本到图像请求的自动修复与学习。

**💡 创新点**

创新点在于：① 以验证驱动的递归自我改进；② 对经验策略实施A/B验证与适用条件判定；③ 评判器通过SFT+GRPO离线自进化；④ 开发ImageReq‑Bench基准。

**🔧 技术方法**

使用的技术包括：文本到图像生成器（Qwen‑Image、Wan2.7‑Image、Nano Banana Pro），五维评判模型，ReAct式检索，目标锁定的局部修复，A/B验证机制，以及SFT+GRPO评判器训练。

**📊 数据集**

使用的数据集包括：MindBench（500条请求）、ImageReq‑Bench（961条请求，8,165条评估项）以及生成过程中的内部记录用于SFT/GRPO训练。

**📈 对比分析**

与直接生成、主流生成模型及Agent系统进行对比；在MindBench上使用Nano Banana Pro时CSA达52.6%，在ImageReq‑Bench上达到65.66%，均比最强基线提升约10–12个百分点，局部修复与经验库显著提升性能。

**⚠️ 局限性**

限制在于：只在冻结生成器上评估，经验对新生成器的迁移性未知；离线评判器自进化需要多轮数据收集训练，且对生成器更换的适应性未验证。

---

## 141. Search-Based Testing of Modality Sensitivity in Vision Language Models

**arXiv ID:** 2610.32360 | [PDF](https://arxiv.org/pdf/2610.32360v1)

**作者:** Oliver Weißl `[一作]` (Technical University of Munich), Andrea Stocco `[通讯]` (Technical University of Munich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `6215c339-3735-4be3-8a07-5bbb7004712d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 MoST 框架，利用黑盒多目标搜索系统性生成图像、文本及其联合扰动，评估 VLM 的模态敏感性；

**💡 创新点**

首次引入模态敏感性测试与 SWTD 评价指标，系统比较三种扰动方式的攻击成功率与扰动幅度，揭示模型在不同模态下的脆弱点；

**🔧 技术方法**

基于 NSGA‑II 的进化搜索、多维扰动操作（压缩、模糊、噪声、颜色、剪裁、词语替换、语义变形等），以及视觉/文本相似度和误差度量；

**📊 数据集**

ImageNet‑1K 验证集（单/多类别场景）与 Udacity 驾驶数据集的图像与标注；

**📈 对比分析**

通过攻击成功率、SWTD、MS‑SSIM、NED 等多重指标，并使用 McNemar/Wilcoxon 统计检验比较；结果显示联合扰动往往能在更小的单模态修改下达到或超过单模态攻击成功率，Qwen 对文本更敏感、Kimi 全模态易失效，InternVL 与 Nemotron 的表现更均衡；

**⚠️ 局限性**

仅覆盖有限的扰动家族与单一任务（目标检测），人类评估一致性低，模型覆盖不全（高基线失败率的模型排除），并未探究内部机制导致的模态差异。

---

## 142. Bounding Queue Waits for Fair Scheduling of Bursty HPC Workloads via Hierarchical Algorithms

**arXiv ID:** 2610.32027 | [PDF](https://arxiv.org/pdf/2610.32027v1)

**作者:** Karaen Senthilkumar `[一作]` (University of Chicago), Ian Foster `[通讯]` (University of Chicago)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了HyDRA，一种层级调度框架，兼顾项目、用户和作业的公平性，并为每个作业绑定队列等待时间；

**💡 创新点**

创新点在于：1) 基于作业大小、权重和当前后端负载动态计算截止时间；2) 引入动态优先级分数，使等待时间随年龄增长并考虑作业规模；3) 在保留公平性（预算和服务赤字）约束下执行安全的后填充；4) 将后端负载作为后备因子，使截止时间随队列拥堵调整；5) 明确的层级优先级映射实现项目–用户–作业三层公平。

**🔧 技术方法**

使用了：层级权重计算、动态优先级分数公式、基于后备因子L(t)的截止时间估计、可选的后填充决策逻辑、服务赤字控制、与传统 EASY/Backfilling 的对比实现；算法以 C++20 实现。

**📊 数据集**

采用了 ALCF Polaris 2025 年的真实作业跟踪（约 20 万个作业，304 个项目，953 个用户）以及 Midway3 追踪，用这些数据采样生成不同负载（2h/6h/12h）和不同规模（Small/Medium/Large）工作负载。

**📈 对比分析**

与 SJF、Priority、WFQ、Themis、PCS、ANL 等六种基线进行比较，评估指标包括：项目级 Jain 公平指数、作业到期率、95% 分位数 bounded slowdown、项目延迟、集群利用率。HyDRA 在大负载下项目级 Jain 指数约为 0.24，基线最低为 0.12；到期率为 47% 远高于 36%；95% 分位数 bounded slowdown 降低 4.2 倍；集群利用率略低但处于 Pareto 前沿。

**⚠️ 局限性**

限制：在极端长时间负载（12h）下，后备因子最大只能将截止时间翻倍，导致部分作业无法及时满足；整体利用率略低于最优基线；需要对持续过载情形进一步自适应；实现需集成到生产资源管理器（如 Slurm）以验证实际部署效果。

---

## 143. Representation Learning via Composite Subspace Flow Matching

**arXiv ID:** 2610.32256 | [PDF](https://arxiv.org/pdf/2610.32256v1)

**作者:** José Afonso `[一作]` (Instituto Superior Técnico), Mário Figueiredo `[通讯]` (Instituto de Telecomunicações)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种复合子空间流匹配（CSFM）的方法，用于从原始图像数据中提取语义特征，解决了自监督学习中的特征提取与生成可逆性之间的矛盾。

**💡 创新点**

CSFM通过将确定性降解路径与解耦的噪声路径相结合，消除了多源随机扩散的不可约目标方差底线，同时在计算上实现了高效性（O(D log D)时间复杂度）。

**🔧 技术方法**

使用了复合子空间流匹配（CSFM）框架，结合了频率、颜色和照明子空间的确定性降解路径，以及解耦的各向同性噪声调度。

**📊 数据集**

在CIFAR-10、CIFAR-100和STL-10数据集上进行了实证评估，使用了CNN和视觉变换器作为基础架构。

**📈 对比分析**

与无指导流和像素级去噪自编码器相比，CSFM在CIFAR-10、CIFAR-100和STL-10上表现出竞争力，尤其在线性探测和少量样本分类任务中优于传统方法。

**⚠️ 局限性**

CSFM在处理高频样式细节时可能会导致编码器容量的分配不均，尽管通过有限的样式截止（t_1 < 1）可以强制特征分配到内容子空间，但仍需进一步优化。

---

## 144. See What I Need, Dream What I Want: A Navigation Foundation Model on a Self-Constructed Representation Space

**arXiv ID:** 2610.31762 | [PDF](https://arxiv.org/pdf/2610.31762v1)

**作者:** Jiawei Chen `[一作]` (Peking University), Tong Yang `[通讯]` (Peking University)

**通讯引用:** 6069 | [OpenAlex ID](https://openalex.org/A5069277955)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出 RepNav，一个利用预训练视觉基础模型自建表示空间并在该空间内进行潜在链式思考预测的导航基础模型。

**💡 创新点**

创新点在于将任务目标驱动的自建表示与潜在预判相结合，利用跨层注意力提炼三种视觉基础模型的内部结构，形成具有语义、几何、动态信息的自构空间，并通过联合嵌入预测框架学习预测未来表示。

**🔧 技术方法**

使用的技术包括跨层注意力的残差Transformer、联合嵌入预测架构（JEPA）、VISReg正则化、两阶段训练日程以及多任务（VLN、追踪、点目标、QA）联合训练。

**📊 数据集**

数据集涵盖VLN-CE（R2R、RxR）、VLNVerse、EVT-Bench、InternVLA-N1、ABotN-PointBench以及VQA v2和CLEVR。

**📈 对比分析**

与现有方法对比，RepNav在EVT-Bench单目标/干扰追踪、InternVLA-N1点目标、ABotN-PointBench、VLN-CE和VLNVerse上均取得领先或相当的性能，表现为最高成功率、SPL或碰撞率最低。

**⚠️ 局限性**

局限性在于只在导航任务中验证，尚未测试在操作、柔性手部或全身控制等其他任务上的迁移效果。

---

## 145. Reuse, Rewrite, or Reject: Evidence-Constrained Trace Reuse for Repeated Mobile GUI Control

**arXiv ID:** 2610.31712 | [PDF](https://arxiv.org/pdf/2610.31712v1)

**作者:** Bingyu Bai `[一作]` (Shanghai Jiao Tong University), Erhu Feng `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 82 | [OpenAlex ID](https://openalex.org/A5051943243)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于图-文本模板的记录-回放框架，利用验证过的执行轨迹实现移动GUI代理的选择性重用

**💡 创新点**

将环境和指令变化系统化为六类，并在图-文本合同中为每一步加入预后检验，既能重用也能局部重写或放弃

**🔧 技术方法**

采用图结构捕捉UI状态与动作转移，文本描述任务意图与可替换槽，结合预检、执行与后检证实每一步的有效性

**📊 数据集**

在AndroidWorld基准上评估了五种LLM代理，并在多款Android和HarmonyOS手机上进行真实设备实验

**📈 对比分析**

通过与基线LLM仅调用对比，显示重用率约70–80%，Token消耗和LLM调用量下降80%，执行时间平均提升1.2–3.9倍，性能优于现有相关方法

**⚠️ 局限性**

受限于回退覆盖范围、UI可观测性、未覆盖的工作流和敏感动作的重用风险，且未考虑记录与模板构建成本

---

## 146. Tampered ML: Evaluating LLM Judgment of Machine Learning Research

**arXiv ID:** 2610.32114 | [PDF](https://arxiv.org/pdf/2610.32114v1)

**作者:** Soroush Ebadian `[一作]` (Anthropic Fellows Program), Hailey Joren `[通讯]` (Anthropic)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个名为 Tampered ML 的基准，用来评估语言模型在比较两种实验设计时识别方法学缺陷的能力。

**💡 创新点**

创新点在于提出了一套可扩展的“破坏”管道，能够从真实 ML 论文中提取原始实验设计，并生成仅加入单一方法学缺陷的对照实验；同时设计了基于双向比较的评测指标和大规模人类验证。

**🔧 技术方法**

主要技术包括：基于大型语言模型的实验设计提取与改写、若干自动化合规门（faithfulness、tamper unfaithfulness 等）、对照实验对比推理、以及多种评测指标（per-order、both-orders、pointwise 等）。

**📊 数据集**

使用的数据集为 2024‑2026 年 NeurIPS、ICML、ICLR 的口头与亮点论文，总计 1,858 对实验设计（原始+破坏）。

**📈 对比分析**

评测方法是让模型在两种排列顺序下选择更方法学合理的设计，并统计 per-order 与 both-orders 准确率；最高模型 Opus 5 在 both-orders 上约 78 %（per-order 约 86 %）。在点对点评分任务中准确率下降 4–19 个百分点，说明单独评估更难。

**⚠️ 局限性**

局限性包括：基准构建高度依赖语言模型，可能引入错误；人类验证样本偏向最难项且仅由单个评审完成；选取的论文集中在高质量口头/亮点论文，可能导致缺陷更易被检测；此外模型可能学到表面提示而非真正方法学推理，造成过度信任风险。

---

## 147. ACVA-Bench: Beyond Prompt Following in Abstract Concept Visualization Ability for Text-to-Image Models

**arXiv ID:** 2610.32058 | [PDF](https://arxiv.org/pdf/2610.32058v1)

**作者:** Mengyan Wang `[一作]` (Beijing Language and Culture University), Liner Yang ang Erhong Yang `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出抽象概念可视化能力（ACVA）并构建 ACVA‑Bench 基准，评估 T2I 模型在仅给定抽象词而无视觉提示的情况下生成多样、准确且可区分图像的能力。

**💡 创新点**

创新点在于：①将抽象概念可视化定义为自主组织视觉实体、关系与场景的能力；②基于中文心理语言学规范设计类别平衡的抽象词集；③使用词典映射、成分分析与图像模式的可视化语义量表，实现多维、可解释的评估框架。

**🔧 技术方法**

技术手段包括：LLM 辅助生成可视化语义量表并人工校准、CLIP 与图像嵌入、RBF‑MMD、最近中心分类、两选一视觉辨别任务及多尺度自动与人工评判。

**📊 数据集**

数据集：150 个按 19 维认知维度划分的中文抽象词（六大类每类 25 词），每词配 5 个近义词共 30 对；6 款 T2I 模型分别对每词生成 3 张图，共 2,700 张评测图像。

**📈 对比分析**

评测方法为三轴指标：语义充分度、表达多样性、概念辨别；实验显示 GPT‑Image‑2 在充分度上最高（约 0.87）但多样性最低；Qwen‑Image‑2.0 与 GLM‑Image 兼顾适度；近义词辨别平均准确率约 0.81，整体性能表现出显著模型间差异。

**⚠️ 局限性**

局限性：仅涵盖中文抽象词，样本量相对有限；只评估单帧静态图像且提示极简，未扩展至多语言、多媒体或更大规模的验证。

---

## 148. Input Bias Correction in Gaussian Process Regression

**arXiv ID:** 2610.31920 | [PDF](https://arxiv.org/pdf/2610.31920v1)

**作者:** Muzaffar Qureshi `[一作]` (University of Florida), Rushikesh Kamalapurkar `[通讯]` (University of Florida)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种偏差感知高斯过程（Ba-GP）框架，用已记录的有误输入训练的GP模型在获得更准输入后进行高效的偏差校正与噪声补偿。

**💡 创新点**

创新点在于：①将输入偏差与残余噪声分别视作均值修正与协方差修正；②利用已训练GP的梯度和已存的矩阵，构造可重用的近似（Mean-only、一次和二次Neumann），将计算复杂度从O(N³)降至O(N²)；③兼容任何提供偏差估计与误差协方差的轨迹平滑方法。

**🔧 技术方法**

核心技术包括：高斯过程回归、泰勒展开一阶校正、第一/第二阶Neumann级数近似、Kalman滤波/RTS平滑、平方指数核函数以及对梯度与协方差的解析计算。

**📊 数据集**

使用两套数据集：1) 1D正弦函数的模拟数据；2) 室内光强场测量数据（Husarion ROSbot 2R Pro+TSL2591传感器与光电感应灯）与运动捕捉定位。

**📈 对比分析**

与基准方法（原始GP、记录GP、Dallaire GP、NIGP、全Ba-GP、Mean-only、Neumann-1/2）比较；结果显示：Ba-GP及其近似在MAE、NLPD和置信区间覆盖率上显著优于记录GP、Dallaire和NIGP；Neumann近似几乎与全Ba-GP相同，且计算时间显著低于全量重新训练（O(N³)）。

**⚠️ 局限性**

局限性包括：需要已知偏差估计与误差协方差且两者独立；近似在输入误差较大或核参数不佳时可能失效；对大规模数据仍受矩阵求逆限制，需进一步采用稀疏GP或更高阶近似。

---

## 149. PnP*: A certifiable solver for Perspective-n-Point

**arXiv ID:** 2610.31797 | [PDF](https://arxiv.org/pdf/2610.31797v1)

**作者:** Bang-Shien Chen `[一作]` (Northeastern University), David M. Rosen `[通讯]` (Northeastern University)

**通讯引用:** 1746 | [OpenAlex ID](https://openalex.org/A5082166133)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种新的相机姿态估计方法PnP*，该方法能够从3D-2D对应关系中直接优化最大似然估计（MLE），并保证计算效率和全局最优解的可证明性。

**💡 创新点**

PnP*是第一个可扩展的可证明求解器，能够直接优化PnP的MLE，并提供了在中等噪声下的稳定性保证，这是对任何PnP MLE松弛的首次稳定性保证。

**🔧 技术方法**

使用了半正定规划（SDP）松弛技术，通过引入必要的辅助变量和冗余约束来保持问题的稀疏性，从而实现了可扩展的求解。

**📊 数据集**

使用了合成的3D点和2D测量数据集，进行了多次蒙特卡洛实验以验证算法的性能。

**📈 对比分析**

与现有的最小求解器P3P、局部非最小求解器EPnP和SQPnP、替代求解器以及分支界限（BnB）方法进行了比较。PnP*在中等噪声下能够以超过10 Hz的速度返回可证明的正确相机姿态，且在性能上优于其他方法。

**⚠️ 局限性**

PnP*在处理极端噪声或不良初始化时可能会遇到困难，未来的工作将包括增强对离群值的鲁棒性以及将松弛技术扩展到其他大规模视觉问题。

---

## 150. Can Test-Time Memory Evolution Outperform Test-Time Training for Scientific Discovery?

**arXiv ID:** 2610.32046 | [PDF](https://arxiv.org/pdf/2610.32046v1)

**作者:** Yepeng Liu `[一作]` (University of California Santa Barbara), Yuheng Bu `[通讯]` (University of California Santa Barbara)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现 Test‑Time Memory Evolution (TTME)，一种通过不断更新外部记忆而非权重更新来改进 LLM 在科学发现任务中的搜索策略。

**💡 创新点**

用可插拔的外部记忆演化替代传统测试时训练，实现更低计算成本的同时保持甚至提升发现性能；引入指导记忆与策略记忆两种模块，自动提炼并复用实验经验。

**🔧 技术方法**

基于大型语言模型（Qwen3‑8B、gpt‑oss‑120b、DeepSeek‑R1‑0528‑Qwen3‑8B）和现有搜索框架（TTT‑Discover、ThetaEvolve）；使用 Prompt‑driven distillation、LLM 生成策略指令、Guideline Memory & Strategy Memory 等技术。

**📊 数据集**

五个数学构造任务（Erdős 最小重叠、AC1、AC2、AC3、Hadamard）以及对应的验证器。

**📈 对比分析**

在相同框架、模型与搜索预算下与原始 TTT 与无 TT 的基线对比；TTME 在所有任务上均优于 TTT，且 runtime 降低 48–59%，FLOPs 降低 83–85%，且能更早达到或超过 TTT 最终成绩。

**⚠️ 局限性**

验证范围仅限于这些数学任务与所选框架，未验证对更广泛科学发现任务的通用性；外部记忆容量与维护成本仍是潜在限制；与 TTT 叠加效果不佳，说明两者不一定兼容。

---

## 151. Beyond Data Schedule: Fast LLM Pretraining on Data Mixtures by Loss-Reweighting

**arXiv ID:** 2610.32141 | [PDF](https://arxiv.org/pdf/2610.32141v1)

**作者:** Jiazheng Li `[一作]` (Tsinghua University), Jingzhao Zhang `[通讯]` (Tsinghua University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了多域预训练中曝光比例与损失加权的耦合问题，并提出了一种在线自适应损失重加权方法，提升知识与数学推理能力。

**💡 创新点**

将域曝光与优化尺度分离，利用实时域损失统计通过归一化逆平方根公式自适应调整域损失权重，从而在固定曝光下突破传统混合或多阶段训练的性能上限。

**🔧 技术方法**

使用基于指数滑动平均的损失估计、归一化逆平方根权重规则、Muons/AdamW/ActGD等优化器，并在GPT‑2 风格 Transformer 上实现。

**📊 数据集**

采用知识、数学和代码三个域的数据，覆盖 30B–240B 标记化 token，主要评测指标为 MMLU、GSM8K、MATH、HumanEval 和 MBPP。

**📈 对比分析**

与传统阶段式训练、直接混合以及多阶段混合在相同曝光与预算下对比，结果显示在 7:1 数学/知识比例下在 4×Chinchilla 提升 MMLU 0.39 点、GSM8K 13.57 点、MATH 8.40 点；在 8×Chinchilla 提升 MMLU 0.83 点、GSM8K 17.06 点、MATH 9.0 点、HumanEval 6.71 点、MBPP 5.6 点。

**⚠️ 局限性**

未评估安全性、偏见与公平性，方法仅在 GPT‑2‑style 1.5B 模型上验证，且依赖域损失统计的稳定性；对其他域或更大模型的泛化性仍待验证。

---

## 152. A Topology-Aware Quantum-Classical Framework for Multi-Target Active Debris Removal using Gaussian Boson Sampling

**arXiv ID:** 2610.31824 | [PDF](https://arxiv.org/pdf/2610.31824v1)

**作者:** Komal Pandya `[一作]` (Independent Researchers), Nagaraj Kattian `[通讯]` (Independent Researchers)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种融合 Gaussian Boson Sampling 与多目标进化优化的混合量子-经典框架，用于在轨道碎片环境中选择并排序 Active Debris Removal（ADR）目标。

**💡 创新点**

创新点在于将轨道转移网络映射为加权图，利用 GBS 进行结构感知的目标子集采样，并将子集生成与后续航路优化分离，显著降低组合搜索空间并提升多目标规划性能。

**🔧 技术方法**

使用轨道动力学建模、图论加权网络构造、Gaussian Boson Sampling、PyGMO 的 NSGA-II/MOEA/D 进化算法及其 Archipelago 并行架构。

**📊 数据集**

实验基于公开 TLE 数据构建的三套基准集（SOCRATES Minefield、Collision Cloud、Random）以及相应的碰撞风险与来源族标签。

**📈 对比分析**

通过在 N∈{15,18,22,25}、k∈{3,5} 的情形下比较七种目标优先策略的 GBS 采样多样性、转移网络结构与下游多目标航路规划 Pareto 前沿，发现基于网络结构的策略能产生更优的燃料-时间折衷方案，且总体计算成本显著低于完整枚举。

**⚠️ 局限性**

主要局限在于 GBS 采样受硬件/模拟资源限制，样本数量受限；加权图构造与优先参数需经验调优；对极大规模候选集的可扩展性和真实量子硬件上的性能尚未充分验证。

---

## 153. Randomization to Balance Performance under Nominal and Compromised Communication

**arXiv ID:** 2610.31957 | [PDF](https://arxiv.org/pdf/2610.31957v1)

**作者:** Colton Hill `[一作]` (University of Colorado at Colorado Springs), Philip N. Brown `[通讯]` (University of Colorado at Colorado Springs)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究在多智能体子模糊最大化游戏中，利用对数线性学习（log‑linear learning）中的随机化来在常规通信与通信受损两种情况下实现性能平衡。

**💡 创新点**

创新点在于证明：对任何给定游戏，都存在一种温度设置，使得在正常通信时效率几乎最优，而在通信受损时效率仍严格正；但同时证明没有统一的温度选择方案能保证所有游戏的受损效率始终大于某个固定阈值，尤其当智能体可选资源过多时受损效率会变得极差。

**🔧 技术方法**

主要技术包括：使用边际贡献效用函数构造精确势游戏；对数线性学习的马尔可夫链稳态分析；构造星形图游戏（star‑graph game）作为理论和实验模型；通过理论证明与数值实验阐释温度与效率的关系。

**📊 数据集**

实验使用的“数据集”为合成的星形图游戏（G*（m,n）），其中包含主资源、次资源和大量低价值资源，用以模拟智能体面临的多资源选择情形。

**📈 对比分析**

方法比较主要展示温度对名义效率与受损效率的折中效果：在适当的低温度下名义效率可保持在最优的95%+，而受损效率显著提升；但当资源数量增大时，受损效率随温度设定而急剧下降，最终可低于预设阈值。

**⚠️ 局限性**

限制在于：没有通用的温度选择策略能在所有游戏中保持受损效率的正下界；当智能体需要在极大资源集合中做选择时，任何固定温度都可能导致受损效率趋近零，表明随机化策略需要与游戏结构紧密匹配。

---

## 154. Don't Forget What You Just Learned: Self-Directed Repair Turns Agent Mistakes Into Lasting Lessons

**arXiv ID:** 2610.31846 | [PDF](https://arxiv.org/pdf/2610.31846v1)

**作者:** Yuval Kansal `[一作]` (Princeton University), Mark Ibrahim `[通讯]` (Meta FAIR)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出 Self‑Direction 框架，利用低秩适配器（LoRA）在代理执行失败后自动提炼、学习并存储修复策略，从而在未来相似任务中直接复用这些修复；

**💡 创新点**

创新点在于：①将失败回溯到自监督的“自我反思”阶段，②在无需额外教师或外部奖励的前提下，直接将学习到的修复映射为轻量级 LoRA 适配器；③通过句子嵌入路由器动态检索与当前任务相似的适配器，实现持续改进与知识迁移；

**🔧 技术方法**

核心技术包括：自我反思与错误轨迹提取、示例合成、SVD 初始化的 LoRA 训练、句子嵌入路由（sentence‑transformers/all‑MiniLM‑L6‑v2）、基于 Open‑Weight Qwen 系列模型的 ReAct 对话循环；

**📊 数据集**

使用的数据集包括：GAIA（多难度任务、工具调用、浏览、代码）；Claw‑Eval（工具使用工作流）；Terminal‑Bench 2.1（终端编程）；MiniWoB++（浏览器交互）；并在 Benchmark 官方数据上评估；

**📈 对比分析**

对比方法有：仅重试（Resampling）、上下文中的反思（Reflection in Context）、权重中的反思（Reflection in Weights）、累积反思文件（Skills.md）。在 GAIA 上成功率从 38.8% 提升至 55.8%（+17.0pp）；Claw‑Eval 从 64.4% 提升至 73.3%（+8.9pp）；Terminal‑Bench 2.1 从 51.7% 提升至 65.5%（+13.8pp）。Self‑Direction 还在某些指标上超过了参数规模更大的前沿模型（如 GLM‑5.1、Claude Opus 4.7、Gemini 3.1 Pro）；

**⚠️ 局限性**

限制包括：①需要 GPU 与额外存储来训练 LoRA；②错误检测依赖自评或外部反馈，若误判会影响学习；③LoRA 修复与模型体系紧密耦合，难以跨模型迁移；④在极端任务分布变化时，路由匹配误差可能导致不当适配器加载；

---

## 155. TaxoSAE: Rethinking Sparse Autoencoders through Taxonomic Logic

**arXiv ID:** 2610.31894 | [PDF](https://arxiv.org/pdf/2610.31894v1)

**作者:** Xinyu Guo `[一作]` (University of Arizona), Mihai Surdeanu `[通讯]` (University of Arizona)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种税onomic 对齐的稀疏自编码器（Taxo-SAE），通过堆叠SAE并引入税onomic 对齐损失，能够从大型语言模型中读取并控制知识。

**💡 创新点**

将层级关系直接嵌入特征空间，使用堆叠SAE和税onomic 对齐损失，实现不同层级之间的逻辑一致性与可控性，并能读取未见概念的上位词。

**🔧 技术方法**

使用稀疏自编码器、堆叠网络、税onomic 对齐损失、双线性价值头以及对齐解码/编码等技术。

**📊 数据集**

构建了两个事实问答数据集：Bio（人名生物属性）和Animal（动物名称），均基于Freebase和ConceptNet。

**📈 对比分析**

与三种基线（随机、DiffMean、Hybrid）以及平面AlignSAE对比，Taxo-SAE在税onomic 读取、控制和未见概念的超类读取上分别提升了约+31%、+25%和+38个百分点。

**⚠️ 局限性**

实验仅在英语、规模有限的三层税onomic 及18终端节点上验证，未检验跨语言或更大模型和更复杂逻辑的泛化。

---

## 156. DualGSL: Prior-Guided Graph Structure Learning with Node-Adaptive Prediction Fusion

**arXiv ID:** 2610.32462 | [PDF](https://arxiv.org/pdf/2610.32462v1)

**作者:** Yuxia Sun `[一作]` (Jinan University), Li Zeng `[通讯]` (Changsha University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在无先验图的节点分类任务中，提出 DualGSL 框架，利用特征相似度构建先验图并引入虚拟类节点，同时学习任务适应的结构图，二者通过共享编码器产生独立表示并通过节点级门控实现logit融合，从而提升预测性能。

**💡 创新点**

核心创新在于：①保留独立的先验路径和学习结构路径，②通过标签增强的虚拟类节点将标签信息注入先验传播，③设计基于表示差异和置信度的节点级门控实现自适应logit融合，④将融合、分支分类和类锚对齐联合优化，形成一体化的学习机制。

**🔧 技术方法**

采用的技术包括：图神经网络（共享两层编码器）、kNN余弦相似度先验图构建、四种学习相似度函数（加权余弦、ReLU后点积、LeakyReLU线性组合、线性缩放点积）构建结构图、虚拟类节点与源/预测节点划分、投影矩阵实现多视图表示、门控网络生成节点级混合系数、类锚对齐损失、联合交叉熵分类损失。

**📊 数据集**

实验数据集为：AMG18恶意软件归因（6,360样本、18类）、20News文本分类（18,846文档、20类）以及USPS手写数字识别（9,298图像、10类）。

**📈 对比分析**

与七种基线（MLP、GCN、GSR、IDGL、ELU-GCN、CoGSL、LEGNN）在同一特征与数据分割下对比，DualGSL在三数据集上分别取得最高/次高准确率（AMG18 88.76%，20News 72.35%，USPS 95.36%）和Macro-F1（AMG18 87.28%，20News 71.58%，USPS 95.00%），跨数据集平均准确率和Macro-F1分别为85.49%与84.62%。

**⚠️ 局限性**

局限性包括：①依赖先验图构建质量和标签信息，对先验图欠佳时效果可能下降；②仅在传递式节点分类设置下验证，未探究增量或无标签场景；③模型在大规模图上的稀疏性与计算成本需进一步优化；④门控机制的解释性与鲁棒性尚待深入研究。

---

## 157. Analysing Austrian Parliamentary Speeches by Predicting Party Affiliations

**arXiv ID:** 2610.32515 | [PDF](https://arxiv.org/pdf/2610.32515v1)

**作者:** Sarah Sulollari `[一作]` (University of Vienna), Benjamin Roth `[通讯]` (University of Vienna)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对奥地利国民议会1996–2018年的发言文本使用 fine‑tuned BERT 进行党派概率预测，并将每位议员的预测结果聚合为 CM 级别的概率分布，用以分析党派之间的演讲特征、联盟与对立关系以及党派变动对演讲行为的影响。

**💡 创新点**

利用误分类信息揭示党派间的政治亲疏与演讲风格演化；在大规模议会语料上首次实现对党派的概率级别预测并公开相关数据集；将党派预测结果与政治事件关联，提供一种定量分析政治话语的方法。

**🔧 技术方法**

采用 BERT‑base‑german‑cased 进行微调，顶部加 softmax 分类层；使用 5 折 CM 层交叉验证、分块处理长句、概率平均聚合；利用 t‑SNE 可视化、精度/召回/F1 等指标评估；与 Logistic Regression + tf‑idf 进行基线对比。

**📊 数据集**

基准数据为 ParlSpeech V2 中的奥地利议会语料，包含 88219 条发言、708 名发言人（691 名议员），时间跨度 1996–2018 年，涵盖 9 个政党。

**📈 对比分析**

模型在 CM 聚合层的精度达 88.6%，大党（SPÖ、ÖVP、FPÖ）的 F1 超过 0.85，次大党（GRÜNE、NEOS、LIF、BZÖ）的 F1 在 0.5–0.6 之间；单句层面表现较差（F1 低于 0.6）。与 Logistic Regression 基线相比，BERT 在大多数党派上显著更优；误分类结果能够揭示党派间的政治相似性。

**⚠️ 局限性**

单句预测效果不佳，主要受话题局限和语料稀缺影响；模型为黑盒，缺乏可解释性；对小党数据不足导致误分类率高；未提供细粒度语篇分析，需进一步质性研究。

---

## 158. RSI-Opt: Recursive Self-Improvement Optimizer Agents For Model Training

**arXiv ID:** 2610.32026 | [PDF](https://arxiv.org/pdf/2610.32026v1)

**作者:** Binghang Lu `[一作]` (Purdue University), Guang Lin `[通讯]` (Purdue University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一个递归自我改进的优化器代理系统（RSI‑Opt），让大型语言模型在训练过程中动态修改优化器，并通过跨跑策略进化实现经验积累。

**💡 创新点**

创新点在于将优化器策略抽象为可编辑的prompt、playbook、代码库和决策节奏，并通过元代理在归档中递归改进，从而实现跨跑经验迁移与可检验的可执行工具。

**🔧 技术方法**

使用了Claude Opus 5.5、GPT‑5.5/6.1 作为LLM，工具调用式交互、可验证的Python优化器模块、CPU代理实验、基于Darwin‑Gödel机器的归档搜索，以及z‑score标准化评估。

**📊 数据集**

在训练集上使用ResNet‑18/CIFAR‑10、GPT‑34M/FineWeb、UNet/MNIST；在held‑out上使用ViT/CIFAR‑100、LSTM/WikiText‑2、noisy‑CIFAR‑10以及PINN Burgers、Heat、Fractional PDE 等三类PDE任务。

**📈 对比分析**

通过与静态默认优化器、调优的学习率搜索以及固定prompt代理对照，RSI‑Opt 在训练集上 z‑score 提升约 +4.3 点，在 held‑out ML+PDE 上平均提升 +4–6 点；在 Burgers PDE 上误差下降至 8.8 倍，整体显示显著优于基线，但不同模型间效果不均。

**⚠️ 局限性**

限制包括迁移性能受任务集规模和模型差异影响、归档搜索未与匹配预算对照、缺乏对更大规模多 GPU 训练和真实 wall‑clock 成本的评估、安全隔离不完善，可能导致间接信息泄露，以及仅在单 GPU 短周期内验证，未探讨长期稳定性。

---

## 159. CFBrainAug: Modularity-Induced Counterfactual Brain Network Augmentation for Interpretable fMRI-based Autism Diagnosis

**arXiv ID:** 2610.32184 | [PDF](https://arxiv.org/pdf/2610.32184v1)

**作者:** Yifan Qiao `[一作]` (University of Virginia), Sheng Li `[通讯]` (University of Virginia)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出并实现了 CFBrainAug，一个利用对照实验脑网络增强和自解释的对照实验框架，用于基于 rs‑fMRI 的自闭症诊断。

**💡 创新点**

创新点在于将对照实验脑网络作为训练样本与解释同时使用，并引入模块数量先验约束保证生物学可解释性。

**🔧 技术方法**

使用图谱构建、Transformer‑MLP 的多约束细化、RBF‑SVM 轻量级分类器以及模块化、拓扑保持与语义一致性损失等技术。

**📊 数据集**

在 ABIDE I 三个站点（NYU、UM、LEUVEN）共 64–184 受试者的 rs‑fMRI 数据上进行实验。

**📈 对比分析**

与四种深度图网络（BrainGNN、STGCN、GoFormer、Miniformer）及四种数据增强（Window Slicing、Time Warping、Node Dropping、Edge Perturbation）以及两种对照实验基线进行对比，CFBrainAug 在所有站点实现最高准确率（NYU 74.1%）和 AUC（NYU 79.2%），优于其他方法。

**⚠️ 局限性**

局限在于仅做了单站点评估、对照实验标签切换仅为鼓励而非强制，且缺乏跨站点泛化与更深层次的生物学验证。

---

## 160. Gaze Under Strain: How Oculomotor Behavior Tracks Cybersickness Progression in VR

**arXiv ID:** 2610.32019 | [PDF](https://arxiv.org/pdf/2610.32019v1)

**作者:** Nevzat Umut Demirseren `[一作]` (Texas State University), Isayas Berhe Adhanom `[通讯]` (Texas State University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了19名受试者在20分钟VR导航任务中，结合眼动追踪与连续一分钟的不适评分，分析视线稳定性、视觉探索与视线分配随不适与时间的变化。

**💡 创新点**

创新点在于引入时间分辨的眼动事件指标，并考察其与重复不适评分的交互效应，揭示不同眼动维度在持续暴露中的动态演变。

**🔧 技术方法**

使用HTC Vive Pro Eye眼动追踪（120 Hz）、I‑VT速度阈值算法提取眨眼/注视/扫视事件，并通过线性混合效应模型（R的lme4）评估不适、时间及交互项对眼动指标的影响。

**📊 数据集**

采用原始VR导航实验数据，19名受试者在现实环境中完成约20分钟行走，收集眼动数据与一分钟一次的不适评分。

**📈 对比分析**

通过线性混合效应模型比较各眼动指标的主效应与交互效应，结果显示注视时长随不适下降且随时间加剧，扫视幅度与速度未出现显著主效应但与时间交互显著；与SSQ症状无显著关联，提示行为随时间演变更为重要。

**⚠️ 局限性**

局限包括样本量有限、仅单一环境与行走模式、缺乏对时间与不适独立操纵的对照设计，以及眼动测量随时间可能出现的漂移。

---

## 161. LeafSkill: Efficient Leaf-Level Action Grounding for Recursive LLM Agents

**arXiv ID:** 2610.32412 | [PDF](https://arxiv.org/pdf/2610.32412v1)

**作者:** Shen Liu `[一作]` (North China Institute of Computer System Engineering), Chunlei Wang `[通讯]` (China Information Security Research Institute Co., Ltd.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

开发了一个外部检索‑绑定‑选择层（LeafSkill），将递归LLM代理的叶级任务映射为可执行动作，从而显著减少主模型推理量。

**💡 创新点**

创新点在于将叶级动作地面化抽象为独立委托边界，采用语义检索+状态绑定+可学习选择三段流程，并支持离线技能库扩展而无需重新训练选择器。

**🔧 技术方法**

技术上结合BGE‑small进行语义检索、Qwen3‑1.7B实现本地选择器、Qwen3‑5‑27B作为主模型，并利用离线模板化技能库、确定性绑定与抽象化词干化等方法。

**📊 数据集**

使用Robotouille（同步/异步烹饪任务）和ALFWorld（家庭任务）两套环境的轨迹数据，构建了包含413/226键、446/228技能的离线库。

**📈 对比分析**

在与主模型直接推理做对比时，Robotouille实现23%/14%主模型token和调用量下降，ALFWorld实现41%下降；成功率略高或相当；通过库扩展可将token节省提升至约25%。

**⚠️ 局限性**

局限性在于仅适用于离散可枚举动作场景，需要环境适配器；无法处理连续或开放式动作；对主模型、规划器保持固定，需预先构建完整技能库。

---

## 162. Solvent: Long-Horizon Reasoning for Multi-Step Retrosynthesis

**arXiv ID:** 2610.32308 | [PDF](https://arxiv.org/pdf/2610.32308v1)

**作者:** Yifei Wang `[一作]` (Fudan University), Hao Zhou `[通讯]` (Tsinghua University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并训练了一个8B参数的大语言模型Solvent，将多步还原合成规划视为长时序推理任务，模型通过迭代规划、库存可用性验证和自适应修订完成完整路线生成。

**💡 创新点**

创新点包括：①将全流程还原合成抽象为长时序任务，让LLM在多轮交互中完成规划与修订；②提出两轮多阶段后训练课程，先用教师蒸馏构建知识基础，再通过拒绝采样微调和强化学习稳定并提升长时推理；③在仅8B规模下实现与前沿LLaMA/Claude等大型模型竞争的性能。

**🔧 技术方法**

采用的技术包括：大语言模型(8B)；组序列策略优化（GSPO）强化学习；教师引导的冷启动与拒绝采样微调；结构化的还原分析蒸馏；库存可用性外部验证；多轮交互式修订与反馈闭环。

**📊 数据集**

使用的数据集涵盖：USPTO-full（训练与后训练），USPTO-190、ChEMBL-1000、Pistachio Reachable/Hard（四个基准测试），以及eMolecules库存作为可采购前体库。

**📈 对比分析**

与多种前沿与中型LLM（如Claude Opus、Gemini 3.5 Flash、GPT-5.5、Kimi K3等）以及传统搜索系统（retro*, EG-MCTS、RETRO-R1等）进行对比。Solvent在四大基准上平均成功率61.86%，与前沿模型持平且远超同规模模型；在搜索预算实验中，在USPTO-190、ChEMBL-1000及Pistachio基准上均取得最高成功率。

**⚠️ 局限性**

限制：路线验证主要基于计算预测，无法保证实验成功率、选择性、产率或安全性；缺乏与实验平台的实时反馈；未来需结合自动化实验平台进行实验验证与迭代优化。

---

## 163. PrefO2O: Zero-Shot Preference-Based Reinforcement Learning via Offline-to-Online Latent Preference Transfer

**arXiv ID:** 2610.31812 | [PDF](https://arxiv.org/pdf/2610.31812v1)

**作者:** Ruiqi Wang `[一作]` (University of Southern California), Guohua Chen `[通讯]` (Beijing University of Chemical Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计并实现了 PrefO2O，一种通过学习任务不变的潜在偏好空间，将源任务偏好信息迁移到未见目标任务，实现零-shot偏好驱动强化学习的框架。

**💡 创新点**

创新点在于：①联合训练轨迹编码器、偏好 VAE 与元奖励模型，构造任务不变的偏好潜在空间；②在此空间内利用 OOD‑aware Gromov‑Wasserstein 迁移生成可信的伪偏好标签；③通过预热的奖励模型与 OOT 机制，使目标任务无需任何人类偏好标注即可完成政策学习。

**🔧 技术方法**

核心技术包括：轨迹编码器、偏好变分自编码器、元奖励模型、Gromov‑Wasserstein 最优传输、置信度加权伪标签、SAC/PEBBLE 的偏好强化学习流程。

**📊 数据集**

使用 Meta‑World、Robosuite 与 DeepMind Control Suite 等多种机器人操控与步态基准；源任务提供约 100 条人类偏好标注，目标任务不提供任何标注；评估还使用真实环境奖励与模拟奖励作为上界。

**📈 对比分析**

与 PrefFew、PEARL、RL‑VLM‑F、PrefGT、SAC 等基线进行对比；PrefO2O 在跨任务、跨本体、单源/多源转移场景下伪标签准确率最高、学习曲线最快，最终性能与甚至超过了有目标任务真实偏好或奖励的基线；在 Kinova Jaco 机器人上也实现了 60–70% 的成功率。

**⚠️ 局限性**

局限性包括：①假设源任务与目标任务属于共享的行为结构域，无法迁移到完全新颖的偏好维度；②OT 对大规模源池的可扩展性受限，当前在小批量上实现；③潜在空间的语义并未显式标注，导致对特定行为属性的解释性和可控性有限。

---

## 164. Emergent Cooperative Thermoregulation in Lunar Swarm Robotics

**arXiv ID:** 2610.31868 | [PDF](https://arxiv.org/pdf/2610.31868v1)

**作者:** Anais Jeger `[一作]`, Lennart Puck `[通讯]` (European Space Agency)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

训练了一种共享的强化学习策略，使20个月球探测机器人在月球夜间通过彼此热屏蔽实现生存。

**💡 创新点**

仅以终端温度奖励为目标，无位置、接触或几何奖励，训练得到聚集和轮换行为，实现无共享参考、无接触、可扩展至更大规模。

**🔧 技术方法**

使用了强化学习（演化策略估计）配合共享参数的循环神经网络（64隐藏单元），并在基于热学模型的仿真环境中进行训练。

**📊 数据集**

使用的“数据集”是随机生成的机器人初始配置和10个保留测试种子的仿真数据，没有真实硬件数据。

**📈 对比分析**

通过与散射、静止聚集和手工聚集基线比较，生存阈值为155.6 ± 1.3 pcm‑h，比静止聚集低37%，比物理下限高12.7%。

**⚠️ 局限性**

限制在于仅在仿真中验证，未考虑长时间机械磨损、尘土接触等硬件现实问题，需进一步在硬件上验证。

---

## 165. WADE: An LSTM-Based Anomaly Detection Engine for WebAssembly Applications

**arXiv ID:** 2610.31984 | [PDF](https://arxiv.org/pdf/2610.31984v1)

**作者:** Ilaria Campestrini `[一作]` (Politecnico di Milano), Engin Kirda `[通讯]` (Northeastern University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在浏览器执行的 WebAssembly 应用中，使用动态插桩收集函数级执行轨迹，训练 LSTM 模型学习每个程序的正常控制流，并在线检测异常行为；

**💡 创新点**

提出了基于应用特定行为建模的 LSTM 异常检测方法，结合多阈值自适应投票机制，能够在不依赖签名的前提下实时发现已知与未知的控制流攻击；

**🔧 技术方法**

动态插桩工具 Wasabi、LSTM 神经网络（嵌入层+两层 LSTM+Dropout+全连接输出）、多阈值检测与自适应加权投票、LLM 生成多样化测试程序、Python/ PyTorch 训练与评估；

**📊 数据集**

构建了 67 个 WebAssembly 程序的实验集，来源于 7 个真实漏洞种子并通过 Claude Opus 4 生成变体，覆盖 10 类漏洞（缓冲区溢出、堆/栈数据覆盖、间接调用重定向、格式化字符串等）；

**📈 对比分析**

与一阶马尔可夫链基线比较，WADE 在 67 程序上实现 100% 检测召回率、0.42% 假阳性率、99.62% 准确率、96.9% 精确率；相比于基线仅 7.1% 的召回率，显著提升；

**⚠️ 局限性**

只能检测控制流异常，无法捕获数据仅攻击或侧信道攻击；依赖完整的正常行为训练，训练覆盖不足会导致误报；在真实低攻击率环境下误报率仍可能导致低正预测值；未针对自适应对抗者进行评估。

---

## 166. Data Quality Audit and Bivariate Temperature-Humidity Analysis of a Long-Term LoRaWAN Network: Anomaly Detection, PCA, Clustering and Machine Learning

**arXiv ID:** 2610.32438 | [PDF](https://arxiv.org/pdf/2610.32438v1)

**作者:** Mamadou Sow `[一作]` `[通讯]` (Universite Sorbonne Paris Nord), Mamadou Sow (Universite Sorbonne Paris Nord)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

针对LoRaWAN环境监测网络，本文构建了覆盖率审计及四层多模态异常检测流水线，聚焦温湿度数据并实现了去重、物理边界过滤、异常检测、PCA、聚类与MLP漂移识别。

**💡 创新点**

创新点在于将覆盖率审计嵌入数据提取阶段，消除变量缺失带来的隐式错误，并将单一检测方法与多层无监督/监督模型组合，实现对异常、结构失衡和慢速漂移的互补识别。

**🔧 技术方法**

采用了C++17实现的JSON解析、基于中位数绝对偏差的Modified Z-Score、Isolation Forest、闭式2×2 PCA重构误差、K-means++与加速DBSCAN、以及循环时间编码的多层感知机回归。

**📊 数据集**

使用了位于法国Grenoble的Tour Perret LoRaWAN网络数据集，包含421,937帧，经过审计后得到205,005帧温湿度对的高质量子集。

**📈 对比分析**

通过与传统单变量Z-Score、纯Isolation Forest、K-means聚类和无监督DBSCAN比较，本文验证了多层检测的互补性：Z-Score未发现异常，Isolation Forest检测到2.12%的多变量异常；DBSCAN仅标记0.17%噪声；MLP预测达R²=0.672、RMSE=10.54%，并发现0.62%潜在漂移点。

**⚠️ 局限性**

局限性包括：CO₂与气压变量缺失导致多变量模型受限；温湿度传感器量化与饱和导致微聚类与异常阈值偏移；数据集仅来自单一建筑物，缺乏多地点验证，且对长周期漂移的精细识别仍待进一步研究。

---

## 167. Beyond Pattern Matching in 3D MLLMs: Part-Level Functional Reasoning for Generalizable Recognition

**arXiv ID:** 2610.32262 | [PDF](https://arxiv.org/pdf/2610.32262v1)

**作者:** Weihong Li `[一作]` (Tsinghua University), Shu-Tao Xia `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出 PFIR 框架，利用 3D 部件级功能推理（功能‑>形状→身份）来提升 3D 多模态 LLM 在几何模糊对象上的识别能力，并通过 Per‑Sample Geometric Confusion Score (GCS) 量化现有模式匹配瓶颈。

**💡 创新点**

创新点包括：① 通过 GCS 对几何歧义进行细粒度诊断；② 建立 507K 题答对，功能‑>形状→身份链的自动化数据流水线；③ 采用三阶段训练（多源对齐 → Chain‑of‑Thought SFT → 双目标 GRPO+Task Arithmetic），实现从形式模仿到主动功能发现的跃迁；④ 两个相互正交的 RL 目标（功能推理与描述性归纳）通过 Task Arithmetic 组合，显著提升难度样本的识别。

**🔧 技术方法**

核心技术包括：PartNeXt 部件分割与层级标注；Point‑BERT 编码器 + Qwen3‑VL‑8B 语言模型；多源对齐训练；链式思维 (CoT) 超细调；Group Relative Policy Optimization (GRPO) 双目标 RL；Task Arithmetic 任务向量叠加；以及 GCS 诊断指标。

**📊 数据集**

使用的数据集：PartNeXt（23,138 个带层级标注的 3D 对象，生成 507K CoT 题答对）；Objaverse（经重新注释的对象描述数据）；ModelNet40（无监督的 OOD 评测集，XYZRGB→XYZ 变换）；PointLLM‑V2 的部件定位 QA；以及额外的点云与文本对齐样本。

**📈 对比分析**

与 PointLLM‑V2、PointLLM‑7B/13B、ShapeLLM‑7B/13B、MiniGPT‑3D、Point‑Bind 等基线相比，PFIR 在 ModelNet40 的生成式零样本分类中从 60.5% 提升至 66.7%（+6.2 p.p.），最模糊 20% 组提升 8.7 p.p.，增益比最清晰组高 2.6 倍；在 Objaverse 上分类保持 ~70% 的水平，同时在 Captioning 的正确率、幻觉率和精确度上均显著优于基线，达到 4.20/0.96/81.30%。

**⚠️ 局限性**

局限性：训练依赖 PartNeXt 的精确部件分割，未验证对分割噪声或误标的鲁棒性；目前仅在单对象点云上评估，未扩展到多对象场景；双目标 RL 的奖励设计仍需进一步调优以避免潜在的策略冲突。

---

## 168. DMAIC-Bench: From Self-Reflection to Self-Restraint for Agents in Multi-Stage Analytical Tasks

**arXiv ID:** 2610.32305 | [PDF](https://arxiv.org/pdf/2610.32305v1)

**作者:** Shan Peng `[一作]` (Xiwei Technology Co Ltd), Jia Li `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

设计并发布了 DMAIC-Bench benchmark，评估大语言模型在多阶段分析任务中根据中间证据做出停止、转移或继续的能力。

**💡 创新点**

创新点在于：①引入配对反事实任务和受保护的决策图，能精确区分是否应停止；②结合执行级轨迹评分与专家审查，既量化动作决策，又验证最终报告的有效性。

**🔧 技术方法**

技术手段包括：受保护决策图、数据管理员、私有验证器、私有评分器、执行轨迹记录、专家审计，以及基于 Harmonic Mean 的综合奖励公式。

**📊 数据集**

数据集涵盖 97 个工业场景（15 个案例，4 个行业）和 16 个生命科学任务（RNA‑seq 与 GWAS），所有数据均由种子生成，公开可复现。

**📈 对比分析**

比较方法：在 97 个工业任务上评估各大模型（如 GPT‑6‑astra、Claude‑Sonnet‑5 等）的 GO 与 NO‑GO 路径准确率，发现 GO 率高但 NO‑GO 准确率仅为负值（约 -25%），表明自我约束表现不足。生命科学扩展显示同样错误模式，且在无规则提示时 NO‑GO 准确率下降。

**⚠️ 局限性**

局限性：①仅覆盖有限的工业与生命科学场景，可能不具备普适性；②缺乏对真实数据的验证，所有任务均为模拟；③奖励与评分设计对不同模型的影响差异大，难以统一公平比较；④实验中仍存在人类专家评估主观性与分数波动。

---

## 169. PPO vs. SAC: A Comparative Study for Cooperative Motion Control of Dual-Arm Space Robots

**arXiv ID:** 2610.32034 | [PDF](https://arxiv.org/pdf/2610.32034v1)

**作者:** Nasim Fallahiarezoodar `[一作]` (York University), Zheng H. Zhu `[通讯]` (York University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了一套基于强化学习的双臂空间机器人协同运动规划框架，在目标到达任务中实现了碰撞回避、姿态匹配和运动平滑。

**💡 创新点**

提出了统一的集中式状态表示和多目标奖励函数，并在同一任务公式下对比了 on-policy PPO 与 off-policy SAC，揭示了 PPO 在奖励设计下易陷入奖励徘徊局部最优的缺陷。

**🔧 技术方法**

使用 Proximal Policy Optimization (PPO) 与 Soft Actor-Critic (SAC) 两种深度强化学习算法，在 PyBullet 仿真环境中训练双臂控制策略。

**📊 数据集**

使用自行构建的双臂机器人仿真数据集：在可达工作空间内部随机生成目标位置和姿态（训练）以及外部 3cm 边界区的目标（测试），共 5 种随机种子，累计 4M 环境步。

**📈 对比分析**

采用相同的 MDP、相同的训练步数和相同的随机种子进行公平比较；SAC 在 5 种种子上均达 80% 成功阈值（中位 1.6M 步），平均成功率 78.6%，比 PPO 的 7.4% 高出十倍，且位置误差、姿态误差、同步误差和碰撞率均显著优于 PPO。

**⚠️ 局限性**

局限在于仅使用固定基座模拟，未考虑观测噪声、自由漂浮基座动力学和真实环境验证；样本量仅 5 种种子，无法完全覆盖随机性。

---

## 170. Autonomous Cyber Weapons as AI-Enabled Digital Harms: A Platform Accountability Framework

**arXiv ID:** 2610.32238 | [PDF](https://arxiv.org/pdf/2610.32238v1)

**作者:** Mansi Ganga Nayak `[一作]` (Arizona State University), Hritika Adhikary `[通讯]` (Arizona State University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了针对国家级自主网络武器的责任框架，包含数字伤害分类、意义人类控制（MHC）模型和基于ARSIWA的责任架构，并在已知事件上验证其适用性。

**💡 创新点**

①将平台治理与国际人道法结合，针对ACW提出闭环的MHC阈值；②将国家责任与可测量的控制维度结合，形成可操作的责任阈值；③通过案例对比展示所有已知事件均不符合阈值。

**🔧 技术方法**

使用量化指标法（MHC五维加权得分）、阈值推导（IHL比例性与区分性）、案例分析和专家一致性检验（ICCs）等技术。

**📊 数据集**

使用公开归因的国家级网络攻击案例（如NotPetya、Stuxnet、Volt Typhoon等）以及构造的假设性防御AI案例。

**📈 对比分析**

通过对每一案例的五维子得分与总得分与阈值比较，所有实测案例均低于阈值，验证框架可识别不合规性；对假设案例的得分可达阈值但在独立评估中略低，显示灵敏度和一致性不足。

**⚠️ 局限性**

依赖已归因案例，缺乏归因方法；权重与阈值为示例性校准，缺少专家共识；缺少真实标签数据集，模型泛化性未知；对防御性AI评估存在设计矛盾。

---

## 171. TGR-EvoDAG: A DAG-based Multi-Agent Code Evolution Framework for Generative Recommendation on Tencent

**arXiv ID:** 2610.32326 | [PDF](https://arxiv.org/pdf/2610.32326v1)

**作者:** Junlin He `[一作]` (Tencent), Zang Li `[通讯]` (Tencent)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种多智能体代码演化框架（名为<框架名>），专门用于端到端的生成式推荐（GR）流水线的自动演化，能够在代码层面通过有向无环图（Code DAG）记录并管理多条创新分支，支持跨阶段的合并与协同适配。

**💡 创新点**

创新点包括：1）将通用代码演化的高级搜索（多分支、合并）与推荐系统专属的诊断反馈（模拟用户、数学探针）深度融合；2）引入阶段感知成本规划，基于实际 GPU 预算动态平衡不同阶段的重训成本；3）主动下游共适配机制，允许先行改进上游（如分词器）并通过后续节点及时补偿下游，以克服短视评估导致的创新丢失。

**🔧 技术方法**

主要技术包括：多智能体（Operator Selector、Architect-Critic、Coder-Reviewer、Distiller、Pruner）协同工作，使用大语言模型（GLM‑5.2/Claude）生成和评估代码与假设；Code DAG 结构实现多分支搜索与合并；GPU 成本跟踪与预算管理；模拟用户诊断与数学探针作为多维反馈；自动化实验平台实现阶段级别的精确成本与指标记录。

**📊 数据集**

使用公开 Amazon Office Products 数据集进行离线实验，并将演化出的 GR 模型迁移至腾讯大规模商业媒体平台（每日活跃用户超过 1000 万）进行工业级离线验证和 14 天 A/B 测试。

**📈 对比分析**

与传统非演化基线（APAO）以及多种通用与专用演化框架（AlphaEvolve、DeepEvolve、CORAL、Arbor、EvoRec、Self‑EvolveRec）进行对比。实验显示，框架在离线 Hit@10、NDCG@10 等指标提升约 24.7% / 37.4%，在模拟用户满意度提升 19.7%；工业环境下 Hit@1 提升 12.16%，在线 CTR 提升 0.40%。

**⚠️ 局限性**

局限性包括：1）对 GPU 预算的精细控制依赖实验平台的准确度，易受成本估计误差影响；2）多智能体协作对 LLM 计算资源有较高需求，可能不适用于资源受限的研究场景；3）框架主要针对三阶段 GR 流水线，尚未验证对更复杂或不同结构的推荐系统的通用性。

---

## 172. PRISM: Promoting Sampling Diversity of Large Language Models with Implicit Stochastic Modulation

**arXiv ID:** 2610.32101 | [PDF](https://arxiv.org/pdf/2610.32101v1)

**作者:** Qixin Xu `[一作]` (Tsinghua University), Bohan Lyu `[通讯]` (Tsinghua University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种训练时框架，通过子模集合成和随机嵌入调制实现LLM多样性可控生成。

**💡 创新点**

创新点在于将多样性内化为模型能力，使用子模数据挑选和特殊token噪声作为控制信号。

**🔧 技术方法**

使用子模集合优化构建多样化训练样本、正态噪声注入特殊token嵌入、理论稳定性分析及强化学习初始化等技术。

**📊 数据集**

使用写作数据集WritingPrompts、BookMIA，推理/代码数据集HumanEval、MBPP、MATH-500、AMC23、AIME 24/25，以及RL数据集SimpleRL-Zoo、Acecoder-V2。

**📈 对比分析**

与SFT、NEFTune、GEM、FIRE等基线对比，写作质量与多样性Pareto前沿跃升；代码/数学Pass@N在大多数基准上优于SFT，并提升RL采样效率。

**⚠️ 局限性**

需要多解候选池、噪声尺度任务相关、噪声作用仅在有限步优化中经验化、仅评估7B-8B模型且评估依赖LLM评审与单一嵌入模型。

---

## 173. BARC-Bench: Benign Asks, Risky Consequences

**arXiv ID:** 2610.32024 | [PDF](https://arxiv.org/pdf/2610.32024v1)

**作者:** Kangkang Li `[一作]` (Scale AI), Mohamed Elfeki `[通讯]` (Scale AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并实现了BARC-Bench，一个针对工作场景中看似正常但可能导致安全风险的任务集合，利用模拟工作环境评估大型语言模型在执行这些任务时的安全性和一致性。

**💡 创新点**

创新点在于：①首次构建聚焦于“善意请求但潜在危险后果”的基准；②通过工作记录、隐含上下文和社交规范定义安全边界；③引入pass^k_safety指标衡量模型在多次尝试中的安全一致性；④系统化分析不同类型边界（显式、隐式、社交）下的失败模式。

**🔧 技术方法**

使用了基于工具的多轮交互框架（AgentEnv、OpenCode harness）、四大模型（GPT‑6、Anthropic、Gemini、DeepSeek）以及人工与模型共同生成的安全评判器；任务与环境通过AI辅助生成并经人工审核；对模型行为进行五轮尝试并记录安全与否。

**📊 数据集**

BARC‑Bench数据集包含110个任务，分布在九个模拟工作环境（如Juniper Recruiting、Kestrel Logistics、Vellum Customer Support 等），每个环境提供数百至上千份文档、邮件、日历事件和工具接口，任务与环境均通过人机混合生成。

**📈 对比分析**

通过对每个任务进行5次尝试，计算安全pass@5（至少一次安全完成）与pass^5（全部五次安全完成）指标。最佳模型（GPT‑6）pass^5为72.7%，其余模型分别为37.3%、23.6%和9.1%。相较于传统成功率，pass^5显示模型在安全一致性方面的显著不足；按任务类型分解后发现，文件操作的安全率普遍高于沟通类任务。

**⚠️ 局限性**

局限性包括：①安全评判标准基于人类审核，可能带有文化偏差；②数据集为合成环境，缺乏真实多轮工作流程；③仅评估安全性而未充分衡量任务完成度；④缺乏对模型澄清请求时的后续交互分析；⑤在社交规范识别上仍表现不佳，需进一步提升模型对隐含约束的推理能力。

---

## 174. System One models Are Not Better Rerankers

**arXiv ID:** 2610.32318 | [PDF](https://arxiv.org/pdf/2610.32318v1)

**作者:** Michael Dinzinger `[一作]` (University of Passau), Michael Granitzer `[通讯]` (University of Passau)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `a4b10f5d-130b-4e77-9367-6469ec621899` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估“System One”决策模型作为检索系统的重排序器，并与同一 Qwen backbone 的普通重排序微调进行对比

**💡 创新点**

证明决策微调在 0.6B‑9B 参数范围内无法超越普通重排序微调，且对 27B 仅相当；展示问题表述对校准效果的显著影响

**🔧 技术方法**

使用 Qwen 系列大模型，比较未微调、决策微调（Kev、Clef）和重排序微调（Qwen3‑Reranker、Prism、自研模型）；采用 Isolated 与 Joint 模式；评估 ECE、Brier、nDCG@10、阈值 F1、计数误差

**📊 数据集**

TREC DL 2019‑2023、五个 BEIR 集合（TREC‑COVID、NFCorpus、SciFact、Touché、DBpedia）以及 BM25 top‑100 作为候选集

**📈 对比分析**

对同一 backbone 的三种模型在同一候选集上直接比较；结果显示 0.6B‑9B 期间重排序微调平均提升 5‑29 nDCG@10，决策微调仅提升 ≤0.7；27B 时两者相当；决策模型在校准上仅在特定问题表述下优于重排序微调

**⚠️ 局限性**

实验仅涵盖公开的决策模型与 RLCD 未公开；对 27B 仅测试一组决策微调；校准仅在 TREC DL 上评估；问题表述仅尝试两种；模型尺寸与训练方式不统一，限制了结论的普适性

---

## 175. Geometry-Aware Reinforcement Learning for Close-Fitting Wearable Insertion

**arXiv ID:** 2610.32143 | [PDF](https://arxiv.org/pdf/2610.32143v1)

**作者:** Takuma Tsukakoshi `[一作]` (Waseda University), Sethu Vijayakumar `[通讯]` (University Of Edinburgh)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文提出一种基于几何感知的强化学习框架，用于机器人协助将可柔性手环插入人手，解决了窄缝和关节部位的精细插入问题。

**💡 创新点**

创新点在于将插入任务表述为几何条件化的几何插入问题，采用分阶段课程学习与指尖插入软门机制，显著提升了对开口变形和误差的鲁棒性。

**🔧 技术方法**

技术方法包括Proximal Policy Optimization（PPO）强化学习、几何观察空间（手部与开口关键点相对几何）、软门奖励与分阶段奖励、以及基于Isaac Lab的柔性与刚性手环仿真环境。

**📊 数据集**

数据集主要是自建的Isaac Lab仿真环境，包括柔性手环与刚性手环任务；在真实机器人实验中使用20个不同姿态的人体模型（人偶）进行测试。

**📈 对比分析**

与不使用软门、无课程、硬门等变体比较，完整方法在柔性手环上达到84.7%±1.2%的成功率，在刚性手环上超过90%，在真实机器人上实现了15/20（75%）的成功率。

**⚠️ 局限性**

局限性包括未引入力感知和接触动力学，受视觉估计误差和模型差异影响，难以应对大幅姿态扰动和复杂的开口变形，导致仍有尖锐碰撞、脱落等失败模式。

---

## 176. TravSUITE: Traversability via Self-Supervised, Uncertainty-Aware IRL and Terrain Estimation

**arXiv ID:** 2610.32130 | [PDF](https://arxiv.org/pdf/2610.32130v1)

**作者:** Samuel Triest `[一作]` (Carnegie Mellon University), Sebastian Scherer `[通讯]` (Carnegie Mellon University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `51c0528b-f690-4182-ae60-bb5f046c276c` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 TravSUITE，一个完全自监督的越野可通行性与地图修补学习框架，融合体素映射、视觉基础模型特征和贝叶斯不确定性估计，实现高速度、可靠的本地导航。

**💡 创新点**

创新点在于将地图修补（inpainting）与逆强化学习联合训练、使用 VFM 进行 voxel‑to‑BEV 变换，并在测试时采用 CVaR 风险调节，使系统既能学习精细的地形语义，又能在不确定环境中做出安全决策。

**🔧 技术方法**

技术包括自监督训练、MaxEnt IRL、MPPI 规划、spconv 3D 体素卷积、UNet 语义分割、预训练 VFM、分布式成本/速度预测以及 CVaR 风险评估。

**📊 数据集**

使用约四小时的全尺寸 ATV 现场数据（LiDAR+IMU+RGB），在两地、四季环境下采集的 100k 帧数据，划分为训练/验证/测试集。

**📈 对比分析**

通过与 TerrainNet、Cascaded Trav、FCN 等基线对比，离线指标（MHD、速度概率）和硬件实验（干预次数、碰撞率、平均/最高速度）显示 Parallel Trav+CVaR 在干预率上降至 0、碰撞率降至 1，平均速度提升约 5%，错误率下降 80%，证明系统显著优于传统方法。

**⚠️ 局限性**

局限性包括缺乏跨生态位和跨平台的验证、当前不确定性估计方法相对粗糙、以及仅能进行局部可通行性评估，无法推断映射视域之外的远程路径。

---

## 177. Sam-Correction: Fully Adaptive Label Noise Reduction for Medical Image Segmentation

**arXiv ID:** 2610.32377 | [PDF](https://arxiv.org/pdf/2610.32377v1)

**作者:** Takuro Shimaya `[一作]` (NEC Corporation), Masahiro Saiko `[通讯]` (NEC Corporation)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

开发了 SAM-C 方法，利用 Segment Anything Model 对医学图像分割中的噪声标注进行无监督纠正，并在训练过程中迭代改进。

**💡 创新点**

创新点在于通过无监督使用大型基础模型（SAM）自适应纠正噪声标签，无需额外的高质量标注或偏差参数，且结合预纠正与中间纠正两阶段改进。

**🔧 技术方法**

使用 Segment Anything Model (SAM)、U‑Net 分割网络、B‑mix 门控机制及迭代训练策略。

**📊 数据集**

采用 LIDC‑IDRI（肺 CT）和 Cityscapes（城市街景）真实注释数据集进行实验。

**📈 对比分析**

与基线、SC（带完整/单一验证集）以及 oracle 进行对比，SAM‑C 在 DSC 上显著优于基线和 SC，接近 oracle，且在多种噪声类型下均保持稳健性能。

**⚠️ 局限性**

局限性包括在极端“worst”噪声情况下 SAM‑C 的纠正效果不及 SC；post‑correction 受限于 SAM 对域的适配性；需进一步提升基础模型在医学图像中的表现。

---

## 178. BiLoCo: Binary Low-Rank Corrections for Post-Training Quantization

**arXiv ID:** 2610.32243 | [PDF](https://arxiv.org/pdf/2610.32243v1)

**作者:** David Jin `[一作]` (Massachusetts Institute of Technology), Yoon Kim `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于二进制低秩校正（BiLoCo）的后训练量化方法，将LLM权重拆分为FP4残差与一维±1标记向量的低秩补偿；

**💡 创新点**

创新点在于将高精度BF16低秩修正替换为1-bit标记向量与标量系数的低秩修正，并在CUDA核心上并行执行，形成单一融合核；

**🔧 技术方法**

采用NVFP4块量化、二进制低秩分解（自适应符号交替算法）、全精度标量系数拟合、Blackwell GPU Tensor Core + CUDA Core 混合计算；

**📊 数据集**

使用WikiText-2、C4数据集评估困惑度，FineWeb 文档评估KL散度，WinoGrande、HellaSwag 等下游任务；

**📈 对比分析**

与SVDQuant及FP4基线对比，BiLoCo在相同存储预算下困惑度、KL散度与下游准确率均优于或相当，解码延迟在小批量时仅提升6–11%，大批量时保持在20%以内；

**⚠️ 局限性**

局限性包括：仅在B300 GPU、特定投影层规模内实现；大批量或宽MLP投影时性能下降；仅对投影层测试，未评估注意力或KV缓存；旋转预处理的兼容性验证有限。

---

## 179. Learning Structured Sparsity for Large Models: Bridging Model Compressibility and Efficient Inference

**arXiv ID:** 2610.31897 | [PDF](https://arxiv.org/pdf/2610.31897v1)

**作者:** Tianxiang Xia `[一作]` (ETH Zurich), Hao Ma `[通讯]` (University of Hong Kong)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了S-SALAAD框架，能够在训练期间联合学习Transformer线性投影的稀疏+低秩（SLR）表示，并支持从单一检查点进行零样本弹性压缩与高效边缘推理。

**💡 创新点**

创新点在于：①将结构化稀疏（块稀疏、N:M稀疏、行/列稀疏）与低秩分解同时引入到ADMM优化中；②提出I‑SVD快速近似SVD，降低训练成本；③设计EGAR能在不需要校准样本的情况下按预算自动分配低秩与稀疏容量；④为块稀疏实现专用CPU/GPU内核，将稀疏和低秩路径高效融合。

**🔧 技术方法**

核心技术包括：ADMM优化框架、热启动迭代SVD（I‑SVD）、能量引导交替精炼（EGAR）、块稀疏正则化、GPU TileLang内核与CPU AVX2实现。

**📊 数据集**

使用公开大语料C4进行Llama2（1B、350M）预训练，FineWeb‑Edu进行Nemotron‑3‑Super‑120B到Qwen3‑1.7B的蒸馏；评估数据集包含标准语言建模与零样本下游任务。

**📈 对比分析**

与Wanda、SparseGPT、SliceGPT、SVD‑LLM、LORO等基线对比。实验显示，S‑SALAAD在相同GPU/CPU平台上，往往在相同或更低参数预算下实现更低perplexity、显著减少峰值显存（最高可达42%）并提升吞吐量（最高≈2×），从而在perplexity‑memory‑latency三维上占据Pareto前沿。

**⚠️ 局限性**

局限性包括：①主要针对块稀疏，其他结构化模式的加速实现尚未完善；②对极大模型的可扩展性和训练稳定性仍需进一步验证；③ADMM与SVD近似可能在某些稀疏率下引入额外训练开销；④缺乏对非Transformer任务或硬件平台（如TPU、移动AI芯片）的评估。

---

## 180. RVG-Bench: A Benchmark for Reasoning-Driven Audio-Visual Temporal Grounding

**arXiv ID:** 2610.32268 | [PDF](https://arxiv.org/pdf/2610.32268v1)

**作者:** Yue Feng `[一作]` (Renmin University of China), Qin Jin `[通讯]` (Renmin University of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了推理驱动的音视频时序定位任务，并构建了RVG‑Bench数据集，包含1,255段视频和1,890个需要多步推理的查询-区间对。

**💡 创新点**

创新点在于将视频定位从表面描述匹配转为需要理解用户意图、推理因果、比较属性等多种推理类型，并严格防止信息泄露，确保模型真正理解音视频内容。

**🔧 技术方法**

使用前沿大语言模型（Gemini‑2.5‑Pro、Gemini‑3‑Pro 等）进行查询生成、质量过滤与目标区间细化，结合人类审核；评估时使用多模态大型语言模型与本地化模型，并通过 Chain‑of‑Thought、Oracle‑Target、任务拆分等多种推理策略。

**📊 数据集**

数据来源于现有音视频 QA 与推理数据集（WorldSense、Daily‑Omni、FunQA、Video‑Holmes 等），通过四阶段生成与审核得到最终RVG‑Bench。

**📈 对比分析**

在Bench上22个模型中，最优模型Qwen3.8‑Omni‑Flash达到49.6 mIoU（68.0 IoU@0.3），显著低于人类基线65.3 mIoU；与传统描述性基准相比，模型性能相差较大，表明任务难度高。

**⚠️ 局限性**

局限性包括：仅覆盖短中等长视频，未涉及长篇电影或数小时视频；部分推理类型仍对模型挑战大；缺乏针对长时序记忆与多段推理的深入探讨。

---

## 181. Separating Angular Alignment from Radial Ordering in Hyperbolic Vision-Language Models

**arXiv ID:** 2610.32146 | [PDF](https://arxiv.org/pdf/2610.32146v1)

**作者:** Mingyang Gao `[一作]` (Beijing Institute of Technology), Che Sun `[通讯]` (Shenzhen MSU-BIT University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一种名为 AURORA 的超曲空间视觉‑语言预训练框架，解耦角度语义对齐与径向层级归纳，并通过 Radial–Angular Order Cone (RAOC) 对层级关系进行建模。

**💡 创新点**

创新点：① 采用根角度对齐仅利用方向信息进行语义匹配，避免距离对径向层级的冲突；② 设计 RAOC 用父子径向间隙控制角度阈值，形成无参的部分序关系；③ 引入层级对比（HEC）训练和范数正则化，显著减少无关对间的假层级；④ 通过实验验证 RAOC 在多任务上均优于传统欧氏与超曲模型。

**🔧 技术方法**

核心技术：超曲空间（Lorentz 模型）嵌入、根角度相似度、Radial–Angular Order Cone、层级对比损失、范数正则化、对比学习的 pairwise sigmoid/InfoNCE 目标。

**📊 数据集**

主要数据集：GRIT（10.3M 图文对，17.8M 框框‑短语对）；在评测时使用 ImageNet、COCO、Flickr30K、VL‑CheckList‑Object、SugarCrepe 等公开数据集。

**📈 对比分析**

对比方法：CLIP、SigLIP、MERU、ATMG、HyCoCLIP、PHyCLIP、ARGENT 等基线；在零样本分类、图文检索、层级分类和组合理解等任务中，AURORA 在 13/16、全部检索指标、所有层级指标以及大部分组合测试中均取得最佳或同等表现。

**⚠️ 局限性**

局限性：① 仍需手工标注的框框‑短语层级信息；② 主要关注树状层级，未探索更复杂图结构；③ 对训练规模和计算成本敏感，需进一步优化实现。

---

## 182. Beyond Local Accuracy: Benchmarking the Rollout Reliability of PDE Foundation Models

**arXiv ID:** 2610.31935 | [PDF](https://arxiv.org/pdf/2610.31935v1)

**作者:** Jaewan Park `[一作]` (National Center for Supercomputing Applications University of Illinois Urbana--Champaign), Surresh Marru `[通讯]` (Georgia Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对七个二维时变PDE数据集进行基线与预训练（PDE基础）模型的评估，比较短期（10–20步）与长期（10–100或10–55步）自回归推理的误差放大，并提出错误放大系数（G）作为衡量指标。

**💡 创新点**

提出基于自回归推理的“误差放大”评价方法，揭示短期优势并不等价于长期仿真可靠性，强调在基线与预训练模型评估中需同时报告短期与长期误差及放大统计，显示基线模型在多数情形下仍具优势，但长期误差普遍放大。

**🔧 技术方法**

使用神经算子（FNO、DeepONet）以及多种Transformer‑基的PDE基础模型（DPOT、GPhyT、Poseidon、Walrus）和冻结的VICON，所有模型以其原生的时间窗口和自回归策略进行推理；误差通过VRMSE、相对L2和相对MAE计算。

**📊 数据集**

七个二维网格PDE数据集：RealPDEBench的FSI、cylinder、controlled cylinder、foil；PDEBench的diffusion–reaction；PDEArena的conditioned incompressible Navier–Stokes；PDEGym的compressible Euler Richtmyer–Meshkov；以及用于数据量扩展的CE‑RM。

**📈 对比分析**

通过在每个数据集和训练样本数（N）下对同一检查点进行短期与长期误差对比，并用误差放大系数评估模型在自回归过程中的误差传播。结果显示：①至少有一款预训练模型在26/30短期细胞中优于两款从零训练基线；②在21/24长期细胞中同样保持优势；③但长期误差普遍放大（median G≈2.04），有51个细胞从VRMSE<1升到≥1，12个细胞在长期超过10，表明短期优势不一定保证长期可靠性。

**⚠️ 局限性**

仅限二维正交网格；未评估3D、非规则网格或更长时间窗口；误差度量仅为场值误差，无法区分相位漂移、幅值误差或守恒缺陷；训练、验证与检查点选择等协议不统一，难以单独归因于预训练；VICON为冻结推理，无法直接比较；误差放大系数为归一化比值，缺乏严格的稳定性解释；阈值1和10为经验诊断而非理论稳定性界限。

---

## 183. ConsMSA: Semantic Distribution Consistency Learning for Multimodal Sentiment Analysis

**arXiv ID:** 2610.31731 | [PDF](https://arxiv.org/pdf/2610.31731v1)

**作者:** Pan Wang `[一作]` (University of Pittsburgh), Jingtong Hu `[通讯]` (University of Pittsburgh)

**通讯引用:** 4493 | [OpenAlex ID](https://openalex.org/A5066534595)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种基于语义分布一致性的多模态情感分析框架ConsMSA，利用token级语义投影和一致性评分对多模态信息进行加权与压缩；

**💡 创新点**

创新点在于定义I^2CS（Intra- and Inter-modality Consistency Score），用Jensen‑Shannon散度量化语义一致性，并将其与预测相关性耦合，形成重要性信号用于一致性正则化、soft gating与硬压缩；

**🔧 技术方法**

采用Transformer编码器、语义投影层、JS散度一致性计算、软硬门控加权、TopK压缩以及联合任务/一致性损失的端到端学习；

**📊 数据集**

在CMU‑MOSI、CMU‑MOSEI和CH‑SIMS三大公开多模态情感数据集上进行实验；

**📈 对比分析**

与多种基线相比，在T+A+V和T+A设置下均实现SOTA（Acc、MAE等指标），并且在token仅保留10%时仍保持与全量相近的性能，显示出鲁棒性与高效性；

**⚠️ 局限性**

限制包括：需要对每个token进行投影和散度计算，导致额外计算开销；对长文本、极端压缩或模态缺失情况的鲁棒性尚未充分验证。

---

## 184. Fisher Information and the Minimum Mean-Square Error for Locating Closely Spaced Point Sources

**arXiv ID:** 2610.32018 | [PDF](https://arxiv.org/pdf/2610.32018v1)

**作者:** Chuling Wen `[一作]` (Shenzhen University), Charles K. Chui `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究了在噪声下通过已知响应函数估计密集源位置的贝叶斯风险，并给出了最优估计误差的上、下界；

**💡 创新点**

创新点在于提出了利用“有界Cramér–Rao界”和“先验占用路径”两条准则来统一估计风险的阶数，并对不同先验（包括奇异先验）给出下尾指数的判定；

**🔧 技术方法**

使用了信息几何、Prony方法、Vandermonde矩阵分析、局部模量与有效信息结合的路径积分技术，以及Cramér–Rao界与最小距离估计的理论分析；

**📊 数据集**

实验部分主要基于仿真数据（模拟不同噪声水平和源配置的合成观测），未涉及真实测量数据集；

**📈 对比分析**

通过与理论上界比较，发现提出的最小距离最小二乘估计在噪声量级满足相应阈值时能够达到风险阶数的匹配；

**⚠️ 局限性**

局限性在于仅适用于已知中心或分离度的单聚类场景，对未知中心、源数或非单聚类分布需要进一步扩展；此外，先验占用路径的假设及高维计算复杂度仍需改进。

---

## 185. Democratizing LLM Training with the Edge: Challenges, Systems, and the Road Ahead

**arXiv ID:** 2610.32215 | [PDF](https://arxiv.org/pdf/2610.32215v1)

**作者:** Ziyue Luo `[一作]` (Ohio State University), Jia Liu `[通讯]` (Ohio State University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一个统一的分布式LLM训练系统，支持跨地理分布的GPU集群进行预训练和微调，并在真实大学站点上验证其稳定收敛性。

**💡 创新点**

采用云端参数服务器与异步聚合，解决计算异构、WAN延迟、网络不可靠性与防火墙/ NAT 等挑战；提供单一通信/压缩/容错堆栈；实现可跨预训练与微调共用的两层架构；提出多PS模式和自适应量化等设计。

**🔧 技术方法**

参数服务器架构、gRPC、异步FedAvg、Fully Sharded Data Parallelism (FSDP)、梯度量化+哈夫曼压缩、故障恢复机制、客户端自适应微批量、混合精度（FP16/ BF16/ INT8）、LoRA、Hybrid LoRA、RAVAN、零阶优化、Federated Split Learning (FSL-SAGE)、NCCL、Bfloat16、INT8 等。

**📊 数据集**

OpenWebText（用于 GPT‑2 Medium 预训练）、OpenWebText（用于 Llama3‑1B 预训练）、Stanford Question Answering Dataset (SQuAD)（用于 GPT‑2 微调）。

**📈 对比分析**

与 FedAvg、DiLoCo、FedNova 等基线进行比较。实验表明在两大学校站点上，10min 同步间隔的异步设计收敛速度快于 FedAvg 和 FedNova，DiLoCo 在推荐参数下出现发散；Llama3‑1B 在多PS 20min 同步下保持稳定下降；WAN 延迟与吞吐实验验证了压缩与异步聚合能缓解网络抖动，系统在真实 WAN 条件下实现了可观的 wall‑clock 效率。

**⚠️ 局限性**

仅在 1B 参数模型上验证，未测试 70B+ 规模；缺乏完整的弹性成员资格与动态网络自适应压缩机制；对多模态、专家模型的支持不足；未实现差分隐私、鲁棒性聚合与经济激励等关键安全与激励机制；在节点离线/网络故障的恢复与负载均衡方面仍需进一步研究。

---

## 186. Dual-Adaptive Z-Score PPO: Dynamic Clipping Using Action Confidence and Normalized Advantage Magnitude

**arXiv ID:** 2610.32194 | [PDF](https://arxiv.org/pdf/2610.32194v1)

**作者:** Taha M. Mahmoud `[一作]` (University of North Dakota), Naima Kaabouch `[通讯]` (University of North Dakota)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

研究了 PPO 的自适应裁剪机制，提出三种变体（AZ3、AZ3S、DAZ3），并在离散动作的经典控制任务上进行实验评估。

**💡 创新点**

创新点在于双信号自适应裁剪——同时利用动作置信度（当前策略对选动作的概率）与优势幅度（归一化优势）来计算样本级裁剪阈值，从而动态调节信赖域宽度。

**🔧 技术方法**

主要技术包括 PPO 与 GAE 结合、Adam 优化器、Sigmoid 映射实现平滑阈值、两种置信度指标的融合、固定超参数和多种种子评估。

**📊 数据集**

使用 Gymnasium 的三个离散动作基准环境：LunarLander-v3、CartPole-v1、Acrobot-v1。

**📈 对比分析**

与标准 PPO 进行对比，采用相同网络结构、优化器、回合数、回报评估等统一设置；实验结果显示在 LunarLander-v3 上 PPO‑DAZ3 取得最高平均回报，在 Acrobot-v1 上 PPO‑AZ3S 最佳，而在 CartPole-v1 上标准 PPO 仍保持竞争力。

**⚠️ 局限性**

局限性包括：实验规模仅三种随机种子；仅评估离散动作任务；缺乏超参数消融与统计显著性检验；未将方法扩展至连续动作或更大规模环境。

---

## 187. Towards Effective and Transferable Physical Camouflage against Multi-View BEV-based 3D Perception in Autonomous Driving

**arXiv ID:** 2610.32230 | [PDF](https://arxiv.org/pdf/2610.32230v1)

**作者:** Linye Lyu `[一作]` (Harbin Institute of Technology), Yu Li `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出BEVCA框架，生成针对多视角鸟瞰图（BEV）3D感知模型的对抗性伪装纹理，实现在黑盒和物理环境下显著降低检测与分割性能。

**💡 创新点**

创新点在于利用BEV特征层进行对抗性损失，并结合遮罩技术精确定位对车辆纹理敏感的BEV区域，同时扩展单视角渲染到多视角渲染，实现跨模型、跨任务的高可迁移性攻击。

**🔧 技术方法**

使用可微多视角渲染（PyTorch3D扩展）、BEV特征提取、遮罩增强BEV损失、总变差正则、Adam优化，并在CARLA模拟器中训练。

**📊 数据集**

数据集包括基于CARLA生成的多视角图像（含目标车辆与无车对照）、NuScenes/Waymo式场景的细化调优数据，以及物理测试用的Audi E‑Tron全尺寸车UV贴图与实测多视角视频。

**📈 对比分析**

与FCA、DTA、ACTIVE、RAUCA、CNCA、AdvPatch等传统2D/3D对抗纹理在3D检测AP和BEV分割IoU上对比，BEVCA在黑盒模型中平均提升约36.2%（AP）与21.6%（IoU），在物理实验中的攻击成功率达51.18%。

**⚠️ 局限性**

局限性包括仿真到真实的域差距、纹理仅适用于特定车辆几何体、需要为新车型单独重新优化、以及对未来Vision‑Language‑Action等新型感知框架的鲁棒性尚未评估。

---

## 188. DBCM: Dual-Branch Calibration Module for Test-Time Adaptation in Time Series Forecasting

**arXiv ID:** 2610.32071 | [PDF](https://arxiv.org/pdf/2610.32071v1)

**作者:** Jaebyeong Jeon `[一作]` (Yonsei University), Kibok Lee `[通讯]` (Yonsei University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出Dual-Branch Calibration Module（DBCM）在时间序列预测的测试时自适应中使用，帮助模型快速补偿非平稳分布漂移。

**💡 创新点**

创新点在于将自适应分解为共享分支和低秩通道特定分支，并引入通道混合与非线性门控，显著降低参数量同时提升校正效果。

**🔧 技术方法**

采用共享适配器、低秩通道适配器、线性通道混合器、双层非线性门控以及轻量级LoRA技术实现校准。

**📊 数据集**

在六个主流多变量时间序列数据集（ETTh1/2/ETTm1/2、Exchange Rate、Weather）以及高维Electricity和三大预训练TSFM（Moirai、Chronos、Chronos‑2）上进行评测。

**📈 对比分析**

与现有TSF‑TTA方法（TAFAS、PETSA）以及其他对齐方法相比，DBCM在所有骨干网络和预测 horizon 上均实现平均 6.20% MSE 降幅（相较于 TAFAS 的 4.07% 与 PETSA 的 2.61%），尤其在长周期和高非平稳数据集上提升显著。

**⚠️ 局限性**

局限在于仅分析输出空间的残差结构，未对输入侧自适应机制进行深入研究；此外对低秩分支容量的自适应选取仍需进一步探索。

---

## 189. HyScale: Scaling Long-Context Training for Learned Sparse Attention

**arXiv ID:** 2610.32057 | [PDF](https://arxiv.org/pdf/2610.32057v1)

**作者:** Yangsheng Deng `[一作]` (University of Edinburgh), Luo Mai `[通讯]` (University of Edinburgh)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一套针对学习型稀疏注意力的长上下文训练系统，能够在百万-token级别加速模型训练。

**💡 创新点**

核心创新在于将选择（selection）视为索引集合而非激活，提出了两项技术：1）HyTopK——利用乐观步进采样（optimistic strided sampling）在线完成 Top‑k 选择，完全避免构造并存储全分数表；2）Selection Replay——在激活重计算时仅重新计算已选索引对应的分数，消除了重复选择的高昂开销。

**🔧 技术方法**

实现技术包括：
- 乐观阈值的动态采样与回退恢复（host‑free recovery）；
- 确定性排序（deterministic sort）保证可复现；
- 细粒度重计算策略（index replay）与全局通信重叠（Decoupled CP Overlap）；
- 结合局部性感知稀疏核（Locality‑Aware Sparse Kernels）提升算子效率。

**📊 数据集**

主要在 GLM‑5.1、DeepSeek‑V4‑Flash、Qwen3.8‑Flash‑Next 的 1M‑token 训练数据上评测，使用 H20/H800 GPU 集群。

**📈 对比分析**

相较于 Megatron‑LM 基线，系统在 512 H20 GPU 上 1M‑token 时提升 2.07×，在 32 H800 GPU 上 1M‑token 时提升 2.01×（CSA）和 1.57×（QSA）。在不同上下文长度、不同 GPU 数量下均表现出更优的伸缩性与更低的 GPU 内存占用。

**⚠️ 局限性**

局限性包括：
- 对于极端失败率高的查询，回退恢复会占用额外计算；
- 乐观阈值依赖于上下文分布，极端噪声或随机初始化时可能导致更多失败；
- 方案主要针对基于 top‑k 的学习型稀疏注意力，尚未验证对非 top‑k 或其他稀疏注意力变体的适用性。

---

## 190. Silence Is Not Assurance: When Production Lane Keeping Assist Does Not Warn, and What It Costs

**arXiv ID:** 2610.31899 | [PDF](https://arxiv.org/pdf/2610.31899v1)

**作者:** Tianjia Yang `[一作]` (Pennsylvania State University), Xianbiao Hu `[通讯]` (Pennsylvania State University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究了车载 LKA 系统在真实驾驶中显示的检测状态是否可信，并评估其完整性与对驾驶者的潜在风险。

**💡 创新点**

创新点在于提出 CANDID‑LKA 框架，利用三种无参考信号（显示状态、第三方视觉置信度、驾驶员转向扭矩）通过隐类模型和多种统计指标无偏估计状态敏感度，并量化“静默”秒对车辆横向位置的影响。

**🔧 技术方法**

使用的技术包括逻辑回归、泊松回归、Cohen κ 统计、事件漏报率、驾驶员响应风险比以及隐类模型（EM 估计）。

**📊 数据集**

数据集来自公开的 OpenLKA‑Failure 子集，包含 13 款车辆共 39,800 秒的自然驾驶记录。

**📈 对比分析**

该方法通过与第三方视觉置信度和驾驶员转向扭矩对比，发现显示状态敏感度仅为 47%（各厂商 44–79%），漏报率高达 50%，显示状态与视觉置信度的 κ 仅为 0.21，表明两者一致性低。

**⚠️ 局限性**

主要限制包括：假设显示状态与视觉置信度误差独立、未对真实道路标记可视性进行人工标注、数据样本量和路线多样性有限，且使用的横向位置指标受视觉系统噪声影响。

---

## 191. TranSAGE: Automating Spatial Accelerator Generation for Large-Scale Transformer Models

**arXiv ID:** 2610.32279 | [PDF](https://arxiv.org/pdf/2610.32279v1)

**作者:** Chi Zhang `[一作]` (Shanghai Jiao Tong University), Jieru Zhao `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提供一种端到端的编译框架 TranSAGE，能够从 PyTorch 代码自动生成针对大规模 Transformer 的空间加速器 RTL 实现，并支持多操作管线执行。

**💡 创新点**

创新点包括：① 将操作聚合为管线化的“操作群集”，实现多操作级别的数据流优化；② 通过动态规划图划分与维度关系网络结合等价类缓存，大幅剪枝搜索空间；③ 设计可重构的空间架构原语（SAP），实现对不同操作群集的硬件统一映射。

**🔧 技术方法**

使用的技术主要有：MLIR+Torch‑MLIR 前端、动态规划（DP）图划分、线性规划求解、维度关系网络（Dimension‑Relation Network）与等价类缓存、SAP 图构造与 RTL 生成、能效与延迟评估（EDP）等。

**📊 数据集**

实验数据集：LLaMA3‑8B、Qwen3‑8B、Gemma‑7B、LLaMA3‑70B、Qwen3‑235B‑A22B（MoE）等大型语言模型，分别在预填充（prefill）和解码（decode）两种工作负载下评估。

**📈 对比分析**

与 NVIDIA Orin NX、A100 GPU、Ascend 910B3 NPU 以及 Gemmini、LEGO 等自动加速器生成器进行对比。结果显示：在预填充/解码模式下，TranSAGE 的吞吐量比 GPU/NPU 提升 2.0–2.4 ×，能效提升 3–13 ×；与现有生成器相比，吞吐量提升 1.5–11 ×，能效提升 1.4–6 ×。

**⚠️ 局限性**

局限性：生成的加速器资源（计算、存储、互连）在一次生成后固定，模型结构变更需重新生成；目前仅针对单芯片架构，未探索多芯片间的通信与并行度策略。

---

## 192. RootDown: Deploying and Benchmarking Models for Beet-Root Detection on Edge Devices

**arXiv ID:** 2610.32275 | [PDF](https://arxiv.org/pdf/2610.32275v1)

**作者:** Gerardus Croonen `[一作]` (AIT Austrian Institute of Technology), Daniel Steininger `[通讯]` (AIT Austrian Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了面向输送线糖甜菜部分重叠检测的端到端图像管线，并构建了首个RootDown基准数据集；

**💡 创新点**

创新之处在于将基础模型SAM 3用于自动注释，大幅降低人工标注成本，并首次系统评估了在Raspberry Pi 5+Hailo加速器上的低功耗实时推理；

**🔧 技术方法**

采用YOLO11检测网络、SAM 3生成掩码、OpenVINO/ONNX导出、INT8量化与Hailo编译器，将模型部署至CPU、GPU和NPU平台；

**📊 数据集**

使用自采的597张甜菜图像构成RootDown数据集，其中312张人工标注，剩余285张通过SAM 3自动生成；

**📈 对比分析**

通过mAP 50‑95对Core‑Manual、Core‑Auto、Ext‑Auto、Full‑Auto四种基准及多种硬件、精度组合进行比较；人工标注模型在640²分辨率下达到56.9% mAP，自动标注模型最高达52.6%；在Raspberry Pi 5+Hailo-8上实现58.7% mAP，推理速度≈83 ms/帧，平均功耗≈8 W，可连续运行超过7.5 h；

**⚠️ 局限性**

主要局限在于SAM 3生成的掩码仍存在噪声，导致标注误差；数据集规模有限，仍需进一步扩展；在ARM CPU上INT8量化效果不佳；目前仅完成检测阶段，后续质量评估等功能尚未实现。

---

## 193. TIE: Training-free CLIP Embedding Guidance for Compositional Text-to-Image Alignment

**arXiv ID:** 2610.32407 | [PDF](https://arxiv.org/pdf/2610.32407v1)

**作者:** Sujung Hong `[一作]` (Yonsei University), Seong Jae Hwang `[通讯]` (Yonsei University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一个无需训练、仅在推理阶段一次性更新文本嵌入的框架 TIE，用以提升多概念文本到图像生成模型的语义一致性。

**💡 创新点**

创新点在于将 token entropy 作为概念丢失的代理指标，并利用 CLIP 第一级自注意力得分捕捉属性‑对象绑定强度；随后通过自适应奇异值放大（AEP）和插值‑外推绑定（IEB）实现一次性嵌入修正，从而同时消除概念丢失和概念混淆。

**🔧 技术方法**

使用 CLIP 文本编码器的内部特征（token entropy 与 L0 注意力）、奇异值分解以及基于注意力的插值与外推操作，实现对文本嵌入的无监督调整。

**📊 数据集**

在 SDXL、SD3.5（中型和大模型）以及 T2I‑CompBench 的 300 条多属性/多对象提示上进行实验，主要评估 BLIP‑VQA 的颜色、形状、纹理、空间和复杂度等指标。

**📈 对比分析**

与 A&E、SynGen、Magnet、ToMe、FOCUS、TACA 等基线相比，TIE 在所有类别均表现出更高的 BLIP‑VQA 分数，且推理时间几乎无显著增加，证明其在保持性能的同时实现了高效推理。

**⚠️ 局限性**

局限性包括：仅适用于 CLIP‑基文本编码器的模型，对极其复杂或长文本提示的鲁棒性仍有限；虽然一次性嵌入更新极简，但在极端属性冲突或语义歧义的场景下仍可能出现残余的概念混淆。

---

## 194. Data Quality in RVL-CDIP: A Systematic Audit

**arXiv ID:** 2610.32126 | [PDF](https://arxiv.org/pdf/2610.32126v1)

**作者:** Stefan Larson `[一作]` (Vanderbilt University), Kevin Leach `[通讯]` (Vanderbilt University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对广泛使用的RVL‑CDIP文档分类基准进行了系统性审计，识别并纠正了标签错误、训练/测试集重叠、识别码等伪相关性、分布狭窄以及OCR引擎不一致等缺陷，并提供了去标识化、代码移除、去重等修复版本，随后重新基准化25个模型；

**💡 创新点**

创新点在于同时考察并修正五大数据质量维度（标签有效性、拆分完整性、识别码伪相关、分布局限、OCR引擎偏差），并通过大规模人工标注、CLIP+SuperPoint+LightGlue双阶段重复检测、XGBoost+CNN识别码分类、合成替换去标识化等技术构建完整的修复流程；

**🔧 技术方法**

采用手工注释、Confident Learning、CLIP文本嵌入、SuperPoint+LightGlue局部特征匹配、Ratcliff/Obershelp文本相似度、XGBoost+CNN识别码分类、OCR（Tesseract/Textract）比较、合成文本替换等多种技术；

**📊 数据集**

使用RVL‑CDIP（40万页）及其衍生的四个IIT‑CDIP子集作为主要数据集，并构造两套OOD集RVL‑CDIP‑N（10万+页）和RVL‑CDIP‑O（3千页）进行外域评估；

**📈 对比分析**

在全部缺陷修正后，重新评估25个模型，最优模型LayoutLMv3提升到96.5%准确率，整体提升0.9–2.9个百分点；同时通过控制OCR引擎可影响4.1–8.3个百分点，识别码移除导致模型损失≈11个百分点，证明缺陷对性能影响显著；

**⚠️ 局限性**

局限性包括：仍存在未知的标签错误和复制未完全消除，修正仅覆盖单标签且仅针对16类，OOB集仅涵盖部分类别，且实验依赖人工标注与手工修复，可能不完全可复现；

---

## 195. Stochastic Geometry Meets Optimization: Data-Driven Cellular Planning

**arXiv ID:** 2610.31793 | [PDF](https://arxiv.org/pdf/2610.31793v1)

**作者:** Abhishek K. Shukla `[一作]` (Indian Institute of Technology Kanpur), Ketan Rajawat `[通讯]` (Indian Institute of Technology Kanpur)

**通讯引用:** 1516 | [OpenAlex ID](https://openalex.org/A5007246073)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出一种基于随机几何与优化的软部署框架，利用用户位置样本学习基站部署强度分布，从而得到覆盖率或吞吐量最大的基站密度函数，并从该强度生成具体基站部署。

**💡 创新点**

创新点在于：①把无限维的基站部署优化问题转化为可解的有限维经验问题；②提供离散化误差、采样误差、优化误差和取样误差的非渐近错误上界；③通过随机镜像上升（Stochastic Mirror Ascent）在高维情形下高效求解；④兼顾多层网络、增量部署、排除区、绿色成本等实际约束，给出可操作的部署指南。

**🔧 技术方法**

主要技术包括：随机几何（IPPP模型）、Wasserstein-1距离的Lipschitz连续性分析、经验风险最小化与Rademacher复杂度推导、随机镜像上升算法、可采样点过程、以及离散化与样本误差分析。

**📊 数据集**

使用了两类数据：①合成用户位置分布（如环形、六聚类等），②真实用户位置（Facebook人口密度图、OpenCellID AT&T基站记录）进行增量部署实验。

**📈 对比分析**

与传统单一基站放置、贪心算法、K-means聚类以及均匀或需求比例随机部署等方法对比。实验显示：优化强度产生的随机部署在覆盖率分布和最优选择的覆盖率上均优于对照方案；在稀缺基站场景下差距不大，但在充分基站场景下优势明显；同时通过验证集选择候选部署可进一步提高最终部署性能。

**⚠️ 局限性**

局限性包括：①误差上界中存在常数和对数项，实际误差可能更低；②随机镜像上升在非凸目标下收敛到局部最优；③模型假设IPPP与Rayleigh衰落，可能无法覆盖所有现实复杂场景；④离散化细化需要更多采样，计算成本随网格细度提升；⑤对现有基站的排除区与成本约束仅以线性形式引入，复杂地理或经济约束需进一步扩展。

---

## 196. PARIS: Parallel Arithmetic Reduction and Integrated Selection for MSDF Inner Products

**arXiv ID:** 2610.32410 | [PDF](https://arxiv.org/pdf/2610.32410v1)

**作者:** Malik Zohaib Nisar `[一作]`, Muhammad Usman `[通讯]` (Universität Rostock)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 PARIS 架构，利用位平面并行归约与共享残差选择，在每个 tile 内并行处理输入项，完成 784 通道 INT8 内积在 88 周期内得到完整的 MSDF 输出。

**💡 创新点**

创新点包括：① 消除输入项序列化，内积延迟不随 fan‑in 增大；② 将 MSDF 归约、计数与残差更新整合至单一 carry‑save 数据流；③ 引入 bias‑aware ReLU 证书，提前终止无效计算；④ 在 45 nm ASIC 上实现 16/256 引擎可扩展加速器，显著提升吞吐与能效。

**🔧 技术方法**

核心技术包括：MSDF 取值与 2‑bit 计数符号处理、并行 AND 与 popcount 归约、5:2 压缩器、共享残差更新、数字选择与残差校正、ReLU 证书判定、基于 FPGA/ASIC 的多引擎同步调度与 16‑engine 局部存储。

**📊 数据集**

使用 MNIST 数据集，构建两种 MLP 模型：模型 A (784–256–10) 与模型 B (784–200–100–10)，在 INT8 量化下保持 98% 以上的分类准确率。

**📈 对比分析**

与匹配的 reuse‑MSDF 基线进行比较：模型 A、B 的速度提升分别为 240.44× 与 180.47×，算术能量下降 98.9% 与 98.6%；在 16‑engine 预留权重配置下，预计 24.33 GOPS 通过率，能效达 1,581 GOPS/W，明显优于现有 MLP 加速器。

**⚠️ 局限性**

局限性包括：尚未完成物理实现与完整的权重存储集成；面积与功耗相对传统 reuse‑MSDF 基线仍高；性能受 tile‑group 调度与内存访问模式影响；仅在 MNIST 任务上验证，需在更大规模网络与多模态数据上进一步评估。

---

## 197. Kolibri: A Sovereign European Model on the Pareto Frontier

**arXiv ID:** 2610.31945 | [PDF](https://arxiv.org/pdf/2610.31945v1)

**作者:** Aleph Alpha `[一作]`, Alexei Zacharov `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种双语（英德）混合专家（Mixture-of-Experts, MoE）Transformer模型，名为“2”，具有数十亿参数（B）和每个 token 仅激活约 3.4 B 参数，能够在 256k token 的长上下文下高效推理。

**💡 创新点**

创新点包括：1）在保持高效推理的前提下，将全注意力与滑动窗口注意力（SWA）混合，以显著降低长上下文下的计算和内存开销；2）使用基于 UniBPE 的 128k 词表，对德语的形态学做了精细化处理；3）构建了规模 24 T 的多语种数据管道，并通过合成、重写、翻译等方式补足德语语料；4）提出了针对公开模型的 Pareto 前沿评估框架，展示“2”在质量与服务成本双重维度上的领先。

**🔧 技术方法**

技术手段包括：MoE 结构（384 路专家，6 个被激活），Hybrid Attention（每 5 层使用全注意力，其余 4 层使用 512‑token SWA），RoPE、RMSNorm、GQA 量化注意力；使用 Muon（谱学习率）、AdamW、WSM（warm‑up 后常数）优化器；后训练阶段结合 SFT 与 RL‑HF，使用 1.2 M 任务；数据管道采用 Resiliparse + CC 过滤、去重、PII 处理、质量评分、抽样权重等；部署基于 DeepEP、FlashAttention、FSDP 等。

**📊 数据集**

数据集涵盖 78 个英德混合源，总计约 29 T token（预训练 20 T，mid‑train 3.4 T，long‑context 200 B），主要来源为 Common Crawl、FineWeb、FinePDFs、GermanCommons、Klettermix、HPLT 等；同时构建 2 T 以上的德语合成、重写与翻译语料；使用专门的质量标注器和多维度抽样权重进行混合；在后训练阶段使用内部 Curated 数据集（推理、工具使用、代码、数学等）。

**📈 对比分析**

对比方法：在多项公开模型（如 GPT‑5、DeepSeek‑V4、EuroLLM 等）上，采用 BPB、准确率、RULER、HELMET 等统一评测集；通过 Pareto 前沿图显示“2”在英德两语言上的 benchmark 分数优于多数公开模型，并在长上下文（至 256k token）下保持高准确率。总体性能提升：在 Agentic、Industry、Math、Code、General Knowledge 等领域平均提升 5‑10 % BPB，长上下文推理吞吐量提升 30‑40 %。

**⚠️ 局限性**

局限性：1）德语数据高度依赖合成与重写，真实多样性有限；2）数据管道仍存在内容重复、泄露与偏差风险，需进一步去重与 de‑contamination；3）HumanEval 评测受“recitation”影响；4）长上下文混合会引起域偏移，难以分离长度与域效应；5）仅在内部测试，缺乏跨平台的公开部署验证；6）模型规模大，对硬件成本和能耗有显著需求。

---

## 198. Sparsity-driven emergence of functional modularity in multitask neural networks

**arXiv ID:** 2610.32153 | [PDF](https://arxiv.org/pdf/2610.32153v1)

**作者:** Bowen Shi `[一作]` (Zhejiang University), Long Qian `[通讯]` (Peking University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究稀疏化正则化在多任务神经网络中如何自发形成功能模块化，利用功能剖面（ablation 与 Taylor 敏感度）识别任务专用子网络（functional tokens），并在 MLP 与 TinyTransformer 上验证其可重用性、深度依赖和正则化形态对模块化的影响。

**💡 创新点**

首次证明仅靠稀疏化约束即可在无预定义模块边界的网络中产生任务专用且可重用的子网络；提出功能标记与重组方法，展示在无重训练的情况下重组后仍保持或超越原网络性能；系统性分析了稀疏度、正则化指数与层级深度对模块紧凑性、专一性与功能迁移的调控机制。

**🔧 技术方法**

采用 L_γ（γ∈{0.5,1,2}）稀疏化正则化、ablation 与 Taylor 输出敏感度剖面、重要性加权归一化逆参与度（NIP）得到的模块化度量 M、PCA 与 CKA 表示学习表示的几何变化，功能子网络提取与重组实验，并通过深度学习训练多任务 MLP 与 TinyTransformer。

**📊 数据集**

使用合成任务族（Binary Sequence Calculation、Sparse Pattern Logic、MNIST Digit-Pair Arithmetic）以及 TinyTransformer 的 Binary Sequence 任务；数据全部由实验自动生成或公开 MNIST 数据集。

**📈 对比分析**

通过保留功能子网络的权重与全网络比较，计算任务相对准确率；对不同稀疏强度、正则化指数、任务加权等进行对比；重组实验显示在不同稀疏度下，重组子网络可恢复或超过原网络性能；与分别训练 5 个单任务网络相比，联合训练+提取成本仅为其 1/4.78，显示显著效率提升。

**⚠️ 局限性**

局限性包括：任务规则与接口需手工设计，功能子网络依赖于单标量权重重要性，可能忽略更高阶交互；理论分析基于线性代理的局部支持恢复，未覆盖全局非线性行为；实验仅在小规模网络与合成任务上验证，尚未证明在更大、更复杂任务或真实世界数据上的可推广性；重组需要兼容的中间接口，限制了复用的普适性。

---

## 199. Are LLM-Generated Formal Specifications Faithful to Their Requirements? A Study on Bimodal Formal Specification Generation

**arXiv ID:** 2610.32053 | [PDF](https://arxiv.org/pdf/2610.32053v1)

**作者:** Wenhan Wang `[一作]` (Institute for Software Chinese Academy of Sciences), Shenghua Feng `[通讯]` (Institute for Software Chinese Academy of Sciences)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文系统评估了大型语言模型在同时获取自然语言需求和代码实现（实现可能错误）时生成形式化规范的可行性。

**💡 创新点**

创新点在于引入双模态生成设定，提出需求忠实性评价指标，并设计了 RULER 两阶段迭代改进流程来缓解需求‑实现不一致导致的规范误导。

**🔧 技术方法**

使用技术包括 GPT‑5.5、GPT‑5.4‑mini 与 DeepSeek‑v4‑pro 生成 ACSL 规范，Frama‑C 进行语法检查与形式验证，E‑ACSL 进行运行时判定，以及 AutoSpec 与 RULER 作为对照策略。

**📊 数据集**

数据集基于 CodeContests 竞赛题库，构造了 1,403 个 C 程序（覆盖 500 题，其中 500 个正确实现和 903 个错误实现），并整理对应的自然语言需求。

**📈 对比分析**

实验比较了三种 LLM、四种提示策略（基础、AutoSpec、RULER‑2‑stage、RULER‑2‑stage+refine）以及 HoarePrompt；在正确实现上 RULER‑refine 提升验证成功率至约 24%，在错误实现上提升至约 29%，但整体远低于 HoarePrompt 的真阳性率，说明性能仍不理想。

**⚠️ 局限性**

主要限制包括：1）规范忠实性难以保证，错误实现易导致弱或误导规范；2）仅在小型竞赛程序上验证，难以推广到工业级代码；3）评估指标依赖人工与 E‑ACSL 近似，无法完全衡量语义一致性。

---

## 200. Spatial Heuristic Strategy based Diffusion Reinforcement Learning for Online 3D Bin Packing

**arXiv ID:** 2610.32241 | [PDF](https://arxiv.org/pdf/2610.32241v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 201. SAGE: A hybrid vision-language and geometry framework for work-zone pathway prediction under temporary traffic control

**arXiv ID:** 2610.31835 | [PDF](https://arxiv.org/pdf/2610.31835v1)

**作者:** Jiaxi Liu `[一作]` (University of Wisconsin--Madison), Bin Ran `[通讯]` (University of Wisconsin--Madison)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了SAGE（Semantic-to-Action Grounded Execution）框架，用于在临时交通管制（工作区）场景下的路径预测；

**💡 创新点**

创新点在于将路径生成、偏好评估与管制合规性分离为四个步骤，在共享候选空间中使用VLM轨迹作为偏好并通过标准化的MUTCD检查实现有限干预；

**🔧 技术方法**

主要技术包括：基于角色解码的RD‑YNet几何生成器、HistGradientBoosting几何评分器、SAGE‑VLM（Qwen3‑VL‑4B+LoRA）进行语义读取和轨迹生成、轨迹作为意见的轨迹‑为‑意见融合（TAO）以及基于MUTCD谓词的有限干预Z0；

**📊 数据集**

使用了ROADWork数据集（3,117条训练、2,306条验证），包含真实城市工作区的前视图、观察前缀和人类演示路径；

**📈 对比分析**

与基线几何规划器、直接输出的VLM/驱动VLA、SAGE的不同子模型进行对比；SAGE在ADE/FDE上分别比纯几何排名器提升8.3%/12%，在硬违例率上降低37%，安全得分提升，整体性能与最强VLM基线OneVL相当；

**⚠️ 局限性**

局限性包括：仅做开放循环评估，未考虑闭环驾驶动态；仅能检查可在图像空间表达的MUTCD条款，未覆盖深度/速度相关约束；VLM读取仍可能误判；缺乏跨区域或夜间工作区等更广泛场景验证。

---

## 202. "Not" is Not Lost in Translation: A Causal Analysis of Negation Representations in Multilingual LLMs

**arXiv ID:** 2610.32281 | [PDF](https://arxiv.org/pdf/2610.32281v1)

**作者:** Giang Son Nguyen `[一作]` (Nanyang Technological University), Wenya Wang `[通讯]` (Nanyang Technological University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文研究了多语言大型语言模型在翻译任务中对否定的内部表示，利用因果干预验证其可控性并探索跨语言共享情况。

**💡 创新点**

创新点在于发现并验证了否定在多语言模型中既可通过稠密激活方向又可通过单一稀疏特征实现因果控制，同时证明了这些表示在不同语言之间高度对齐并能迁移到未见语言及自然语料。

**🔧 技术方法**

使用技术包括对比激活添加（CAA）与稀疏自编码器（SAE）激活干预，结合层级位置的对比方向和特征解码向量来进行方向计算与干预。

**📊 数据集**

数据集涵盖了自行构造的 Mini-minus-4（四种语言的最小差异句对）和 FLORES-minus-4（基于 FLORES‑200 的否定翻译对），并在俄罗斯、印地语、阿拉伯语、印尼语等额外语言上进行测试。

**📈 对比分析**

通过翻译中否定切换成功率（flip rate）进行评估，实验显示在主语言上成功率可达 90% 以上，跨语言迁移平均 80–95%，单一 SAE 特征也能实现 90% 以上的控制。

**⚠️ 局限性**

局限性包括仅关注显式词形否定且仅在翻译任务中验证；未探讨形容词前缀否定、隐式否定以及其他任务中的表现；干预强度需经验调参，且稀疏特征分析受限于预训练 SAE 的可用性。

---

## 203. Label-Budgeted Attribute Inference from Protected Features in Edge-Cloud Split Multimodal LLMs

**arXiv ID:** 2610.32079 | [PDF](https://arxiv.org/pdf/2610.32079v1)

**作者:** Junguang Yao `[一作]` (Chinese University of Hong Kong), Yue Zheng `[通讯]` (Chinese University of Hong Kong)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文研究了边缘–云拆分多模态大型语言模型（MLLM）推理过程中，服务器通过中间特征推断用户敏感属性，并提出了基于标签预算的属性推断框架。

**💡 创新点**

创新点在于①引入恢复曲线以量化特征级隐私，并考虑攻击者的标记预算；②利用同一预算框架统一评估不同防御机制（输入层、模型层、表示层）；③补充无分类器的统计依赖检验（nHSIC），证明仅靠分类器低准确率并不能说明隐私安全。

**🔧 技术方法**

使用的技术包括：不同级别的攻击器（转移、少量标记微调、自监督）、归一化恢复率计算、nHSIC统计检验；实验在LLaVA‑1.5‑7B和Qwen2.5‑VL‑3B两大模型上进行；指标包括平衡准确率、均方误差、归一化恢复率和统计显著性。

**📊 数据集**

使用的数据集包括 FairFace（年龄9组、性别2、种族7）和 UTKFace（性别、种族5、年龄连续），以及公开的辅助面部图像集合。

**📈 对比分析**

与未保护特征的基线对比，在不同拆分层和防御机制下进行评估。结果显示：在Safe‑LLaVA下，20个标签/类即可恢复75–85%的泄漏；ImageProtector在自身模型下恢复<70%，跨模型效果差；PASS将恢复率限制在≈39%；加深拆分层并未显著降低泄漏。

**⚠️ 局限性**

局限性：仅针对人脸属性；实验仅覆盖两大模型和三种防御；未评估压缩特征和其它任务；未来需设计能在拆分边界有效保护特征的防御机制。

---

## 204. DraMaze: A Multi-Agent Pipeline with Future-Aware Memory for Diverse Microdrama Script Generation

**arXiv ID:** 2610.32002 | [PDF](https://arxiv.org/pdf/2610.32002v1)

**作者:** Siyang He `[一作]` (Fudan University), Xipeng Qiu `[通讯]` (Fudan University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种多智能体管线 DraMaze，能够在微剧本生成中实现多样化剧情规划、未来目标驱动的记忆检索以及动态世界状态跟踪。

**💡 创新点**

创新点在于：① 引入进化树搜索与题材记忆，鼓励剧情多样化；② 构造逆向叙事依赖树 (BNDT) 以实现面向未来的记忆检索；③ 结合动态世界状态记忆，保持角色、关系与资源的一致性；④ 创建 MiDramaBench 双语基准和基于因果模式的多维评价体系。

**🔧 技术方法**

核心技术包括：多智能体协同（规划、检索、写作、更新）；进化树搜索与突变；题材记忆的因果模式抽取；逆向叙事依赖树构建与检索；动态世界状态记忆；LLM（Qwen3.6‑27B、GLM5.3‑Flash、DeepSeek‑V4‑Pro）作为底层生成器。

**📊 数据集**

使用从 PG‑19 文本自动抽取的微剧本素材构建 MiDramaBench，涵盖 20、40、60 集长度，提供英文和中文版本，并根据剧情潜力筛选后续生成。

**📈 对比分析**

与 DOME 与 StoryWriter 进行对照实验，覆盖 20/40/60 集；DraMaze 在逻辑与质量分数均高于基线（20 集约 +4.2 分，40 集约 +8.5 分，60 集约 +9.5 分），同时 4‑gram、3‑gram 重复率最低，条件熵最高。跨模型实验（GLM5.3‑Flash、DeepSeek‑V4‑Pro）亦保持显著优势。消融实验验证了未来目标记忆和进化搜索对性能的贡献。

**⚠️ 局限性**

局限性：① 随着剧本长度增长，重复率仍有上升趋势；② 依赖 LLM 背景模型，对不同模型的迁移仍需进一步验证；③ 评价主要依赖自动化指标，缺少深入人类主观评测；④ 逆向依赖树构建与检索对复杂剧情可能产生额外计算开销。

---

## 205. A Regularity Condition Under Which a CPAB Transformation is a Diffeomorphism

**arXiv ID:** 2610.32369 | [PDF](https://arxiv.org/pdf/2610.32369v1)

**作者:** Amit Aflalo `[一作]` (Ben Gurion University of the Negev), Oren Freifeld `[通讯]` (Ben Gurion University of the Negev)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

本文证明了连续分段仿射（CPAB）变换在满足一定正则性条件下是微分同胚，并且在常用的格点划分中该条件几乎必然满足。

**💡 创新点**

创新点在于首次给出了足够的正则性条件（1D为内部单元边界不为零；2D为内部边界无零且法向分量不全为零）来保证 CPAB 变换的可微性，并纠正了早期文献中“所有 CPAB 变换都是微分同胚”的错误。

**🔧 技术方法**

所用技术包括对分段仿射速度场的 ODE 流解析、雅可比矩阵的绝对连续性证明、Grönwall 不等式与占优收敛定理、以及几何概率论论证正则性条件几乎必然成立。

**📊 数据集**

本文未使用具体数据集，而是以理论分析和随机参数抽样的概率论为主要验证手段；若要实验，可在常见的 2D 图像配准或 3D 体数据上验证。

**📈 对比分析**

由于本工作主要是理论性质的证明，未给出数值实验或与其他方法的性能对比；但通过先前 CPAB 计算瓶颈的消除，暗示在实际应用中可获得更快更精细的非刚性变形。

**⚠️ 局限性**

局限性包括：对 3D 的几何正则性条件虽给出但未严格证明几乎必然成立；在特殊几何配置（如边界法向全为零的情况）下仍可能出现不可微的 CPAB 变换；此外，本文关注的是理论正则性，并未给出具体实现细节或针对不同任务的参数调优方法。

---

## 206. Beyond Correctness: How Novice Software Engineering Students Exercise Oversight of Generative AI

**arXiv ID:** 2610.32118 | [PDF](https://arxiv.org/pdf/2610.32118v1)

**作者:** Ali Shakiba `[一作]` (University of Sydney), Thomas Chaffey `[通讯]` (University of Sydney)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过分析 145 名一年级软件工程学生在真实课程中提交的 962 条 AI 交互记录，探究学生在使用 Generative AI 时如何识别输出缺陷、干预方式及评估自信；

**💡 创新点**

首次系统区分学生干预关注点是“准确性问题”与“适配性问题”，并构建四维干预编码框架（触发、动机、干预类型、操作机制），揭示“适配性评估”在学生监督中的主导作用；

**🔧 技术方法**

采用定性编码方法（多维度框架）结合线性混合效应模型与序数逻辑回归，对干预类型与自我评估信心进行量化分析；

**📊 数据集**

使用学生提交的 AI 日志（共 962 条交互，172 条干预）作为数据集，数据来源为悉尼大学 ELEC1005 课程的 AI 日志；

**📈 对比分析**

与传统错误检测相比，研究发现学生干预更多关注适配性（74.4%）而非准确性（16.9%），输出修改和提示精炼分别占 51.0% 与 28.2%，且在自我评估中，认为有小缺陷的输出显著降低评估信心（β_W=-0.434，p<0.001）；

**⚠️ 局限性**

研究局限包括：仅基于学生自我报告，可能低估干预；数据仅来自单一低代码设计课程，难以推广；干预描述不充分导致 13.4% 归类为“描述不足”；对严重错误（5 条案例）缺乏统计推断。

---

## 207. Permission to Retry: A Study of Authorization and Recovery in AI Agent Workflows

**arXiv ID:** 2610.31916 | [PDF](https://arxiv.org/pdf/2610.31916v1)

**作者:** Deekshith Marla `[一作]` (Arya.ai), Kushagra Bhatnagar `[通讯]` (Arya.ai)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过构造人工历史记录，对基于工具的 AI 代理在授权变更与外部不确定性下的恢复机制进行可执行性与安全性评估。

**💡 创新点**

创新点在于：①将授权、派遣、效果观察和调解等职责拆分为独立义务；②设计了基于历史记录的有限符合性检验框架；③对两种主流主机（LangGraph 与 Temporal）以及企业平台 InfinityOS 进行配对对照实验，验证主机实现对同一策略的统一性；④揭示观察边界（是否已持久化效果）对恢复成功率的决定性影响。

**🔧 技术方法**

技术手段包括：Python 3.13+、SQLite、LangGraph 1.2.16、Temporal Python SDK 1.34.0、自定义的“效果账本”与“检查器”，以及 InfinityOS 的授权解析器与沙箱模拟器。

**📊 数据集**

数据集：336条人工生成的工作流历史（14种调度×3种提供者合约×4种策略），并在 InfinityOS 中再增设 16 条合成恢复案例；每条历史均记录工单、授权、派遣、观察、结算等信息。

**📈 对比分析**

比较方法：在相同策略和提供者合约下，分别在两台主机上执行同一历史并通过检查器评分；统计通过率、完成、拒绝、未决等指标。结果显示：所有主机在安全性与进度上保持一致；Ledger reconciliation 方案在 42 条历史上全部通过；Cache、Revalidate、Stable-key 重试方案分别通过 30/36/31 条历史；性能上，Temporal 的单次重置耗时约 2.1~2.8 秒，LangGraph 仅 0.26~0.35 秒。

**⚠️ 局限性**

局限性：①仅测试单一动作和人工假设的授权，未覆盖多工具、多并发、真实 IAM 交互；②使用模拟提供者和本地数据库，未验证在 SaaS 或生产环境中的鲁棒性；③实验规模有限，缺乏统计推断和随机采样；④主机实现差异未完全展开，可能与框架默认配置有关。

---

## 208. DiscoDemo: Discovering Efficient and Diverse Robot Demonstrations for Imitation Learning

**arXiv ID:** 2610.32093 | [PDF](https://arxiv.org/pdf/2610.32093v1)

**作者:** Minho Park `[一作]` (KAIST), Jaegul Choo `[通讯]` (KAIST)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

使用强化学习作为数据生成器，训练一个潜在条件的生成器来产生多样化且成功的演示，并将这些演示用于下游的视觉-运动学习行为克隆。

**💡 创新点**

创新点在于：① 并行逆向课程学习与潜在条件多样性奖励相结合，能在稀疏奖励环境下快速发现多条成功解法；② 通过成功门控和安全惩罚平衡多样性与任务完成；③ 生成的多样化数据显著提升下游政策的覆盖率和鲁棒性。

**🔧 技术方法**

技术手段包括：FlashSAC 强化学习、并行逆向课程、潜在条件策略、METRA 风格多样性奖励、任务成功门控、安全惩罚、VLA（DROID）预训练模型、行为克隆。

**📊 数据集**

数据集为四个Franka Research 3 机器人桌面任务（PnP-Banana、StackCube、FMB-Round、FMB-SqCircle），使用20条人类演示及3,000条生成演示；与运动规划、演示迁移、PGDG、PLD、ExpertGen、P‑RFCL 等基线进行对比。

**📈 对比分析**

采用下游行为克隆实验，比较成功率和完成时间。DiscoDemo 在仿真中平均成功率 62%（基线 40%），在真实机器人上 78%（基线 39%），在所有任务中均获得最强性能；相比人类演示、规划或单一 RL 生成方法，其成功率提升显著且完成时间保持在最优水平附近。

**⚠️ 局限性**

局限性包括：依赖精确模拟与特权状态、需要明确的成功信号；仅在四个简单桌面任务上验证，复杂或非结构化场景的适用性未知；安全约束与多样性目标针对具体任务，未直接针对下游性能优化。

---

## 209. LOCI: Localized Continuous Control for Multi-Subject Image Editing

**arXiv ID:** 2610.31950 | [PDF](https://arxiv.org/pdf/2610.31950v1)

**作者:** Sai Vidyaranya Nuthalapati `[一作]` (Meta AI), Shlok Kumar Mishra `[通讯]` (Meta AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `da1b1a89-583a-4b57-9c81-478778569bec` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种能够在一张图像中对多个目标进行区域感知的连续编辑框架，支持每个目标独立设定编辑强度；

**💡 创新点**

核心创新在于将 LoRA 的标量系数改为与分区对应的向量，并在实例解耦注意力下训练，从而获得可按区域缩放的编辑方向；

**🔧 技术方法**

使用 FLUX.1-Kontext-dev 作为冻结的 MMDiT 编辑器，结合 Regional LoRA、实例解耦注意力、Token 级区域分区与遮罩；

**📊 数据集**

训练数据基于 HQ-Edit，评估数据集为 128 个合成场景、427 条目标–编辑对（使用 Flux.2 生成、Grounding‑DINO/SAM2 注释）；

**📈 对比分析**

与 SliderEdit、SliderEdit+、only、Regional LoRA without 等方法对比，采用 CLIP/SigLIP、LPIPS、DINOv2 等指标；在单目标、独立多控、联合编辑、二维控制等四种协议中，本文方法在目标相似度、解耦性、范围、交叉干扰等多项指标上均取得最优或接近最优表现；

**⚠️ 局限性**

局限性包括依赖用户手工提供的区域掩码、仅在合成数据上验证，真实世界复杂场景的鲁棒性尚待进一步验证；

---

## 210. Stable Set Reconfiguration in Dense Digraphs

**arXiv ID:** 2610.32402 | [PDF](https://arxiv.org/pdf/2610.32402v1)

**作者:** Sarah Houdaigoui `[一作]` (Graduate university of advanced studies), Alexandre Vigny `[通讯]` (University Clermont Auvergne)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了在半完全有向图上，给定两个大小相同的无环子集，是否能够通过有向令牌滑动的重构序列在保持无环性的前提下将一个子集转换为另一个子集的问题，并证明在令牌数 k、图的有向路径宽度 p 与重构序列的迭代次数 ι 这三项参数化下该问题是固定参数可解（FPT）的，同时给出了相应的算法。

**💡 创新点**

创新点在于引入“迭代次数”参数来约束重构序列的重复访问度，并将重构问题与一个新的组合子序列问题（Ordered Stacks）关联，通过因式分解技术将序列压缩为常数大小，从而实现对半完全图的高效动态规划；此外，还提出了新的同步化与兼容性判定方法，使得不同子图的重构序列能够安全合并。

**🔧 技术方法**

本文主要使用了有向路径分解、守卫与黑洞（galactic）构造、同步化与兼容性判定、Ordered Stacks 问题的动态规划与因式分解、以及参数化复杂度分析与递归动态规划等技术。

**📊 数据集**

由于本研究属于纯理论计算机科学范畴，文中未使用任何实验数据集，而是通过严格的算法设计与理论证明来验证结果。

**📈 对比分析**

论文没有进行实验对比或性能评测；所给出的性能描述仅为理论上的 FPT 复杂度 O(f(k,p,ι)·n)，即在参数 k、p、ι 固定时，该问题可以在多项式时间内求解。

**⚠️ 局限性**

限制主要体现在：算法仅适用于半完全有向图；需要引入迭代次数 ι 作为额外参数，若未知该参数的上界，算法的可行性和复杂度尚不清楚；此外，虽然理论上可行，但实际实现的实现复杂度较高，尚未给出具体的实现细节或实验验证。

---

## 211. Training an MoE LLM in MXFP4: How We Closed the Gap to BF16

**arXiv ID:** 2610.32229 | [PDF](https://arxiv.org/pdf/2610.32229v1)

**作者:** Han Wang `[一作]` (Advanced Micro Devices, Inc.), Emad Barsoum `[通讯]` (Advanced Micro Devices, Inc.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并验证了一套基于 MXFP4 4-bit 低精度训练方案，专为 GPT-OSS-20B MoE 模型在 MLPerf Small MoE 训练基准上实现与 BF16 相近的精度与收敛速度。

**💡 创新点**

创新点在于将 1D/2D 混合块量化、随机 Hadamard 变换、随机舍入与简化的权重去振荡（de-oscillation）融合为一套可直接应用的低精度训练配方，并通过去振荡显著缩小与 BF16 的收敛差距。

**🔧 技术方法**

采用的技术包括 1d2d 混合块量化、随机 Hadamard 变换、随机舍入、权重去振荡、宏块缩放、低秩补偿、差分梯度估计等；软件实现基于 ALTO、Torchtitan 并使用模拟 MXFP4 GEMM。

**📊 数据集**

使用的训练与验证数据集为 C4 语料库（约 80 GB）。

**📈 对比分析**

与 BF16 基线在验证损失（3.3418 vs 3.3283）和到达 3.34 阈值所需步骤（+5% vs +20%）进行对比；在 GBS=16、步骤 16,128 时，4‑bit 方案仅增加 5% 步数即可接近 BF16 精度。

**⚠️ 局限性**

局限性包括仅使用模拟 MXFP4 核且未实现核融合与持久化缓冲；未达到真实硬件速度；仅在单一 GPT-OSS-20B 模型和 C4 数据集上评估；未完成 MLPerf 正式提交；对其他模型、批量大小和训练长度的泛化未知。

---

## 212. Non-Expansive Fuzzy Modal Logic Has Uniform Interpolation

**arXiv ID:** 2610.32335 | [PDF](https://arxiv.org/pdf/2610.32335v1)

**作者:** Stefan Gebhart `[一作]` (Friedrich-Alexander-University Erlangen-Nürnberg), Paul Wild `[通讯]` (Friedrich-Alexander-University Erlangen-Nürnberg)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文证明了非扩张模糊模态逻辑（non‑expansive fuzzy modal logic）具有统一插值性质，并给出了构造统一插值子句的算法；进一步证明插值子句的规模为指数级，且可在指数时间内计算；同时提出了基于双射推理的序贯推理系统与阈值调整技术，保持模态深度不变。

**💡 创新点**

主要创新点包括①首次为模糊模态逻辑提供统一插值证明；②设计了对阈值判断的统一插值构造，并推广到公式层面；③提出了利用符号阈值的递归构造，使得插值子句与原公式保持相同的模态深度；④在序贯推理框架下实现了插值计算，并证明其指数规模与复杂度与经典两值模态逻辑K相当。

**🔧 技术方法**

技术手段主要是：
- 通过引入否定标准化与阈值调整（shift）实现阈值判断的标准化；
- 构建双射的序贯推理系统（dual of tableau），并利用其规则递归计算p‑自由补集；
- 使用符号阈值（变量A）和截断移位来实现阈值判断的泛化；
- 归纳证明插值子句满足 
U(Γ) ≥ A ≡p(Γ–A) 这一不变式；
- 通过递归分解与合并推导树分析，得到指数大小上界与指数时间实现。

**📊 数据集**

本研究为理论性工作，未使用任何实验数据集；所有结果均为形式证明与算法分析。

**📈 对比分析**

相较于经典两值模态逻辑K和对应的描述逻辑，本文得到的统一插值子句规模与计算复杂度均保持在指数级别，等同于两值情况；在模糊逻辑中，尤其是Łukasiewicz和Zadeh逻辑，统一插值在理论上并不存在，本文填补了这一空白。

**⚠️ 局限性**

局限与未来工作：
- 结果仅适用于非扩张模糊模态逻辑，尚未扩展到更强的Łukasiewicz或Zadeh模态逻辑；
- 仅处理命题层面，未涉及混合或一阶模糊逻辑；
- 对于非有理数阈值的处理需进一步完善；
- 仅给出理论复杂度和构造，缺乏实验验证与性能评测；
- 未来可探索到混合逻辑、概率模型以及通用模态（coalgebraic）框架的推广。

---

## 213. From Promises to Accountability: Comparative AI Modeling and Visualization of Electoral Commitments in Nepali Political Texts

**arXiv ID:** 2610.32150 | [PDF](https://arxiv.org/pdf/2610.32150v1)

**作者:** Prakash Poudyal `[一作]` (Kathmandu University), Bal Krishna Bal `[通讯]` (Kathmandu University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了首个尼泊尔选举承诺的金标准数据集，并通过机器学习、深度学习和大型语言模型对承诺进行自动提取与分类，最终将最佳模型集成到移动应用中以提升选举问责。

**💡 创新点**

创新点在于：①首次在尼泊尔低资源语言上公开承诺标注数据并报告Kappa为0.777；②系统性比较多类模型并发现轻量级fastText‑LSTM在精度与部署成本上优于大型Transformer；③将抽取结果与生成式AI图像结合，提供可视化承诺展示。

**🔧 技术方法**

使用的技术包括：TF‑IDF、LaBSE、fastText、GloVe词嵌入；传统ML（LR、DT、RF、SVM、NB、KNN）；RNN/CNN（SRN、LSTM、GRU、CNN）；Transformer（mBERT、XLM‑R）；以及LLM（Qwen‑3‑8B、Llama‑3.1‑8B、Gemma‑4‑E4B），并在PyTorch/Scikit‑learn上训练与评估。

**📊 数据集**

数据集由尼泊尔地方选举的宣言、海报、手册、社交媒体文本共2,430条句子组成，标注为承诺(C)与非承诺(NC)，并在5折交叉验证下划分80/20训练/测试。

**📈 对比分析**

通过宏F1、准确率、AUROC等指标对比，fastText‑LSTM以宏F1≈0.913、AUROC≈0.961表现最佳，LaBSE‑LR次之；LLM在少量提示下宏F1≈0.857，但仍低于fastText‑LSTM；使用McNemar mid‑p检验验证差异显著。

**⚠️ 局限性**

局限包括：数据量有限且不覆盖所有省份或政党；模型仍有误判（误报/漏报）且主要基于表面词汇特征；生成图像缺乏外部评测；移动应用仅为原型，未做可用性与用户研究。

---

## 214. An Error Budget for Shuffled Multimodal Windows in the 3rd WEAR Challenge: Stacking, Count Constraints and Seam Matching

**arXiv ID:** 2610.32398 | [PDF](https://arxiv.org/pdf/2610.32398v1)

**作者:** Soheil Kooklan `[一作]` `[通讯]` (Independent Researcher), Soheil Kooklan (Independent Researcher)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在WEAR 3rd Dataset Challenge中构建了一套多模态窗口分类系统，利用视频与IMU特征、窗口级堆叠、后继链接、图结构与软计数约束的均场解码器来对未排序的一秒窗口进行活动标签预测。

**💡 创新点**

创新点在于结合后继链接与同一动作内的“接缝距离”判定、利用图邻接与后继链接构造的多层堆叠链，以及在均场解码器中引入软计数约束，以捕捉整段活动的全局信息。

**🔧 技术方法**

使用了VideoMAEv2的15帧嵌入、IMU 3D加速度特征、多种窗口分类器（MLP、Transformer、CNN、GBM）、梯度提升的后继预测模型、双编码器相似度、图卷积平均特征、三阶段堆叠（MLP+GBM）以及自定义均场解码器。

**📊 数据集**

采用WEAR数据集（22名训练参与者、4名测试参与者），每个窗口含15帧视频嵌入和一条随机手臂的50点加速度样本。

**📈 对比分析**

通过参与者级自举和多种ablation实验评估，最终系统在模拟测试中macro‑F1达到0.938（单次活动参与者0.943），比基线视频/IMU融合低约0.1，超越大多数公开方案；或acles显示仍有约0.03的提升空间。

**⚠️ 局限性**

局限性包括：需要全部窗口进行离线推理（非实时/在线）、软计数约束受训练数据分布影响、对后继链接的改进提升有限、对参与者间差异高度敏感，且实验依赖非嵌套堆叠导致一定的估计偏倚。

---

## 215. NUViT: Vision Transformers for Fourier Space Data

**arXiv ID:** 2610.32041 | [PDF](https://arxiv.org/pdf/2610.32041v1)

**作者:** Jeffrey Gu `[一作]` (Princeton University), Ellen D. Zhong `[通讯]` (Princeton University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种针对 Fourier 空间数据的 Non‑Uniform Vision Transformer（NUViT），通过在极坐标非均匀网格上进行 tokenization、幅相表示以及基于随机傅里叶特征的空间编码，改进了传统 ViT 在频域上的不匹配问题，并将其应用于多种任务（图像分类、晶体识别、MRI 病理分类、cryo‑EM 成像及重建）。

**💡 创新点**

创新点包括：① 在极坐标非均匀网格（利用 NUFFT）上采样，以更好地覆盖低频信息；② 使用幅值与相位（cos/ sin）三通道表示，捕获频谱功率律和相位结构；③ 引入随机傅里叶特征（RFF）作为连续空间位置编码，提升对不同分辨率的泛化能力；④ 将上述改造嵌入标准 Transformer，形成统一的 Fourier‑domain 处理框架。

**🔧 技术方法**

技术手段：非均匀快速傅里叶变换（NUFFT）、极坐标网格采样、幅相分离表示、随机傅里叶特征位置编码、标准多头自注意力 Transformer 编码器、以及与下游任务（如 cryoDRGN 生成模型）无缝对接。

**📊 数据集**

数据集：CIFAR‑10/CIFAR‑100（Fourier 变换后）、合成 2D 晶体格点数据、OASIS MRI 病理分级数据、四个合成及一个真实的 cryo‑EM 数据集（IgG‑1D、IgG‑RL、Ribosome、Tomotwin‑100、EMPIAR‑10076）。

**📈 对比分析**

与基线（FIT、ViT‑S、MLP、传统 Hartley/实值表示）比较，NUViT 在 Fourier 领域的分类精度提升 10–20%，在噪声鲁棒性和 3D 重建（FSC_AUC）上亦有显著提升，尤其在细粒度判别和低 SNR 环境下表现更优。

**⚠️ 局限性**

局限性：由于 NUFFT 计算量大，NUViT 的推理速度比标准 ViT 慢；当前实现对 NUFFT 依赖较强，若使用低效库会进一步拖慢；此外，为避免过采样导致的计算浪费，仍需在不同任务中调节网格参数。

---

## 216. From Visual Affinity to Learning Credit in Multimodal Reinforcement Learning

**arXiv ID:** 2610.32324 | [PDF](https://arxiv.org/pdf/2610.32324v1)

**作者:** Yuhang Han `[一作]` (Hong Kong University of Science and Technology), Linfeng Zhang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一种Outcome-Conditioned Visual Credit (OVC) 框架，针对多模态强化学习中的视觉信用分配进行细粒度优化

**💡 创新点**

创新点在于：①用相对视觉编码将图像亲和力与上下文对比；②通过优势侧对比与方差校准把视觉特征转化为可调证据；③将证据映射为归一化正权重，分离视觉测量与信用分配

**🔧 技术方法**

采用视觉相对编码、优势条件证据校准、归一化信用分配、GSPO‑token 等策略优化技术

**📊 数据集**

使用 ViRL39K、Math77K、MMMU、MathVerse、MathVision、We‑Math、MathVista、MMK12‑val、ChartVerse 等多模态与图表/视觉问答数据集

**📈 对比分析**

与 Base、GSPO、DelTA、VPPO、AT‑RL 等基线对比，OVC 在 6 个基准上平均提升 2.04–2.39 个百分点，最高单项提升达 7.44 分，且在多模型规模均保持一致性

**⚠️ 局限性**

局限性：主要依赖已预训练的视觉‑数学能力，对跨任务样本重叠和非视觉任务的可迁移性尚未充分验证；在视觉信息较弱的场景下效果可能受限

---

## 217. Polyp-VSSD: A Pre-trained Transformer with Local-Global Joint VSSD Decoding for Polyp Segmentation

**arXiv ID:** 2610.32282 | [PDF](https://arxiv.org/pdf/2610.32282v1)

**作者:** Xiaobo Zhu `[一作]` (Chinese Academy of Sciences), Zhenyu Liu `[通讯]` (Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了一种利用大规模端镜像自监督预训练的 Transformer 编码器和局部‑全局联合 VSSD 解码器的肠息肉分割框架 Polyp‑VSSD。

**💡 创新点**

①在私有端镜像视频上使用 DINOv2 进行自监督预训练构建专属基础模型；②提出非因果 Visual State Space Duality (VSSD) 与轻量 CNN 分支相结合的 LG‑VSSD 解码器，实现高效全局上下文建模与细粒度边界重建。

**🔧 技术方法**

DINOv2 自监督预训练、Transformer 编码器、VSSD 线性解码、局部像素块 (LPB)、多尺度 CNN 细化、非因果双向扫描。

**📊 数据集**

私有 20,000 分钟端镜像视频（约 145 万帧）用于预训练；公开五个息肉分割基准：CVC‑ClinicDB、Kvasir‑SEG、CVC‑ColonDB、ETIS‑LaribPolypDB 与 EndoScene。

**📈 对比分析**

与 18 余种基线（CNN、Transformer、Mamba、SAM 等）在五个公开数据集上对比，Polyp‑VSSD 在所有数据集上均获得最高 mIoU / mDice，尤其在 EndoScene 上达到 0.903/0.946，显示出显著的精度与泛化优势。

**⚠️ 局限性**

仅在静态图像上评估；实时视频处理与跨任务迁移（检测、分类）尚未验证；对罕见息肉样式的鲁棒性仍需进一步探索。

---

## 218. Supervised Source Separation of Electric Vehicle Interior Noise Using a Hybrid Transformer Architecture

**arXiv ID:** 2610.32310 | [PDF](https://arxiv.org/pdf/2610.32310v1)

**作者:** Philipp Diel `[一作]` (RWTH Aachen University), Steffen Marburg `[通讯]` (Technical University of Munich)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `b88c6eac-d57a-4623-a604-1f401f3eb268` `67630363-6be0-4f51-ab05-7198250671a5` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

将电动车内部噪声分解为风噪、轮胎噪、电机/逆变器噪和辅助驱动噪，视为监督式声源分离问题。

**💡 创新点**

创新点包括使用物理参数化合成器生成带标签的虚拟车辆数据、将运营状态信息通过FiLM条件化到HT Demucs混合时空变换器、评估在真实测量中的可迁移性。

**🔧 技术方法**

采用Hybrid Time‑Spectrogram Transformer（HT Demucs）结合FiLM条件化、L1时间域损失与可选STFT损失。

**📊 数据集**

使用基于物理的随机化虚拟车辆合成数据，包含单一车辆与多辆车辆两套数据集；同时在实验中对真实车辆测量数据进行验证。

**📈 对比分析**

与无条件HT Demucs对比，未条件下的模型在风轮胎噪声上达高SDR；在真实数据上仍存在误分和幻影，但相对传统TPA/OTPA减少测量工作量。

**⚠️ 局限性**

局限在于合成到真实的迁移性能不足，条件化效果不显著，低能量源的分离仍不理想，且对谱级别评价依赖人工/第三八分波段误差。

---

## 219. REACT: Representational Early Analysis and Control of Chain-of-Thought Trajectories

**arXiv ID:** 2610.32113 | [PDF](https://arxiv.org/pdf/2610.32113v1)

**作者:** Hanwen Zhang `[一作]` (Independent Researcher), Sanmi Koyejo `[通讯]` (Stanford University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种一次性、条件性的激活编辑技术，能在推理模型生成链式思考（CoT）时的早期隐藏状态处（约第2–3个CoT token）对模型隐藏状态进行单次向量加法，以引导后续生成避免不安全输出。

**💡 创新点**

创新点在于：①通过学习安全方向向量并在单一隐藏层位置上一次性施加，实现了无需重新训练模型权重的安全干预；②发现早期CoT状态（t₂）在控制不安全行为方面最有效，与风险可读性不同，强调需要从因果控制效果而非仅风险检测指标来选取干预点；③引入门控机制，只在检测到危险路径时才应用编辑，显著降低对正常请求的负面影响。

**🔧 技术方法**

技术包括：1) 对安全/不安全对齐的CoT轨迹进行重编码，提取隐藏状态；2) 计算安全方向并在教师强制下优化对后续token概率的影响；3) 训练线性门控器根据未编辑隐藏状态决定是否施加编辑；4) 在推理时一次性加法后继续普通自回归生成。

**📊 数据集**

使用的数据集包括：StrongREJECT-273（源独立的不安全请求集）、45类SorrY-Bench衍生池、CAST公开的安全/不安全请求对、Alpaca 500条通用指令用于benign评估；所有数据均已检查无训练-评估重叠。

**📈 对比分析**

与匹配数据的CAST、AlphaSteer、提示边界门控等方法对比，在8B模型上，单次编辑+门控将完整路径安全率从29.06%提升至82.05%（提升约53个百分点），在StrongREJECT和SORRY集合上均明显优于匹配替代方案；在benign评估中拒绝率保持≈5%，实用性≈92，基本不受影响。

**⚠️ 局限性**

局限性包括：①该方法在所评估模型/层位置上表现最佳，跨模型/层迁移需进一步验证；②仅一次性编辑，可能不足以处理更复杂或长尾不安全行为；③门控器依赖训练样本，若数据分布变化可能失效；④风险可读性与可控性并不一致，说明仅靠风险检测无法选取最佳干预点；⑤实验仅在公开评测集上进行，真实场景中可能出现新的攻击方式。

---

## 220. End-to-End Differentiable Diffusion Conditioning for Physics-Informed Optimization

**arXiv ID:** 2610.31721 | [PDF](https://arxiv.org/pdf/2610.31721v1)

**作者:** Ricardo Luna Gutierrez `[一作]` (Hewlett Packard Enterprise), Soumyendu Sarkar `[通讯]` (Hewlett Packard Enterprise)

**通讯引用:** 166 | [OpenAlex ID](https://openalex.org/A5005133715)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了Differentiable Diffusion Conditioning (D2C)，一种将预训练的条件扩散模型通过可微逆扩散链转化为端到端梯度优化器，用于离线模型基优化（MBO）中的物理设计问题；

**💡 创新点**

创新点在于：①利用可微逆扩散链将扩散模型的条件信号作为可优化变量；②采用冻结的生成器与评估器组合，保持生成在训练数据分布附近，同时利用评估器梯度进行优化；③将多目标物理约束通过随机Dirichlet权重嵌入梯度优化，实现对物理约束的直接优化；

**🔧 技术方法**

使用技术包括：条件扩散模型、DDIM确定性采样、Classifier-Free Guidance、可微逆链梯度传播、冻结的前向代理（surrogate）评估器、随机多目标权重抽样、硬约束筛选；

**📊 数据集**

主要数据集：ICF laser pulse数据（约256维时间序列）、SustainDC 数据中心调度数据（144维功率曲线）、Design-Bench 四个任务（Ant Morphology、D'Kitty Morphology、TF Bind 8/10）;

**📈 对比分析**

与12种主流离线MBO基线（梯度上升、COMs、BDI、CbAS、MINs、REINFORCE、DDOM、GTG、PGS、GP-qEI、CMA-ES、Brute Force）以及Diffusion基线对比，D2C在两项真实物理任务上分别提高了约18%（ICF）和9%（数据中心碳排），在Design-Bench四个任务上平均排名第1.75，整体表现最优；

**⚠️ 局限性**

局限性包括：①计算开销大，需对整个DDIM链做反向传播，速度比单次前向方法慢8–17倍；②对冻结评估器的准确性高度依赖，评估器不佳会导致梯度误导；③结果受硬约束筛选影响，必须同时关注有效率和收益；④需要领域专业知识手工设置多目标损失与约束，适配新任务需调参。

---

## 221. Watermarking Diffusion Language Models with Learned Foresight and Balanced Bands

**arXiv ID:** 2610.32205 | [PDF](https://arxiv.org/pdf/2610.32205v1)

**作者:** Hyojun Kim `[一作]` (Yonsei University), Yo-Sub Han `[通讯]` (Yonsei University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

学习了词汇分段（bands）和后置预测器，以提升扩散语言模型的绿红水印检测效果，同时保持文本质量。

**💡 创新点**

通过学习词汇分段以分离竞争候选词，并使用后置预测器估计候选词对邻近位置绿名单的影响，实现了基于检测器期望证据的优化，且加入了单位一次计数的方案。

**🔧 技术方法**

绿红水印、KL约束下的分布倾斜、Gibbs变分原理、前瞻预测器（利用生成器隐藏层）以及单位-aware 统计量。

**📊 数据集**

WaterBench-600（ELI5、FiQA、AlpacaFarm）、MBPP、MMLU-Pro、HumanEval、GPQA-Diamond，以及Dream-v0-Instruct-7B。

**📈 对比分析**

与DLM-KGW和dgMARK对比，采用TPR@1%、TPR@0.1%、log-PPL、LLM评测分数等指标；结果显示在低强度下TPR提升12-15个百分点，在高强度下保持文本质量（Judge分数≈9.3）且循环率低于DLM-KGW；在Dream上迁移学习亦提升检测率。

**⚠️ 局限性**

需保密词汇分段以保证安全；对重写/语义同义词鲁棒性有限；推理成本上升（预测器约1.5倍速度），且训练成本约29 GPU小时。

---

## 222. Present Drawing: Exploring Avatar-Mediated Drawing Events with Physical Traces

**arXiv ID:** 2610.32166 | [PDF](https://arxiv.org/pdf/2610.32166v1)

**作者:** Kojiro Tanaka `[一作]` (CyberAgent), Yuki Okafuji `[通讯]` (CyberAgent)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并实现了一种通过头像化身在真实世界中绘制并留下物理痕迹的“Present Drawing”系统，旨在打破传统绘图活动的物理约束；

**💡 创新点**

创新点在于：①从活动组织者视角构建四轴设计空间（时间、地点、控制来源、化身表现）；②利用隐藏磁力驱动的笔式绘图机与头戴式显示同步，使头像在HMD中看似直接在纸上绘画；③展示多种应用实例（现场签名、回放信息、跨站演示、迷你吉祥物、系统自动签名）；

**🔧 技术方法**

技术手段包括：混合现实（HMD+Unity）、人体姿态捕捉（Meta Movement SDK）、自制磁力驱动CNC笔绘机、GRBL‑HAL开源驱动、逆运动学（Final IK）生成头像动作、UDP/串口通信同步；

**📊 数据集**

未使用公开数据集，系统主要通过现场录制头像动作与笔迹生成的同步数据包；

**📈 对比分析**

对比方法主要是用户体验问卷（Likert量表）和访谈；在一次播放实验中，8名参与者对“头像绘画可信度”评分显著高于中立点，表明系统实现了基本的头像化身绘画可信性；但未给出数值性能指标；

**⚠️ 局限性**

局限性包括：①笔机的速度/加速度缺乏人类绘画的细腻变化，导致时序滞后；②可见机械噪声和延迟可能削弱“头像是绘画者”的感知；③仅在实验室HMD视角测试，未验证多用户/多站点的实际部署；④只支持单一笔/纸媒介，未评估不同工具和纸张的兼容性；

---

## 223. CEMIR vs. FISER: a comparative study of reasoning-based and enrichment-based approaches to pragmatic intent recognition

**arXiv ID:** 2610.32022 | [PDF](https://arxiv.org/pdf/2610.32022v1)

**作者:** Ziya Hosseynzadeh `[一作]`, Hossein Sameti `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

比较了两种意图识别范式：结构化推理方法FISER与自生成叙事+图像增强的多模态方法CEMIR，在HandMeThat v2环境下对机器人的意图识别性能进行评估。

**💡 创新点**

提出了CEMIR框架，即通过大语言模型生成叙事文本并用扩散模型生成对应图像，对原始文本进行多模态扩充，并将其与已有的结构化推理方法系统性对比，填补了“自生成多模态增强”与“结构化推理”在同一数据集、同一任务上的缺口。

**🔧 技术方法**

使用的大语言模型包括Qwen2.5‑Omni‑3B、GPT‑5‑Nano、GPT‑5‑Mini、GPT‑5；叙事生成采用Qwen3‑8B；图像生成采用FLUX；结构化推理采用FISER的社交+具身推理分层；评估采用配对置换检验和Holm‑Bonferroni校正。

**📊 数据集**

基准数据集为HandMeThat v2（400条目、四个难度级别），以及CEMIR扩充后的生成叙事和图像数据集。

**📈 对比分析**

在四种LLM和四个难度级别下进行单次尝试，统计成功率并比较FISER与CEMIR。结果显示两者在所有16个模型×难度组合中没有统计学显著差异，性能主要受模型规模与任务难度驱动，CEMIR的多模态增强并未带来可观提升。

**⚠️ 局限性**

局限性：仅评估了自生成多模态增强，未涉及真实感知输入；实验仅在HandMeThat v2（语义歧义）上进行；未对更大模型进行消融；生成内容的准确性有限且仅在部分样本检验；计算资源受限导致部分实验只能在小模型上完成。

---

## 224. Towards Adversarially Robust Multigraph Neural Networks for Financial Crime Analysis

**arXiv ID:** 2610.31718 | [PDF](https://arxiv.org/pdf/2610.31718v1)

**作者:** Dennis Heijmans `[一作]` (Delft University Of Technology), Kubilay Atasu `[通讯]` (Delft University Of Technology)

**通讯引用:** 1476 | [OpenAlex ID](https://openalex.org/A5027902223)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了第一套针对多重图神经网络（Multi‑GNN）的基于梯度的结构攻击与对抗训练框架，并通过节点–边归一化矩阵实现可微分的消息传递。

**💡 创新点**

创新点包括：① 用节点–边归一化矩阵重写 Multi‑GIN/Multi‑PNA 的消息传递，使得多重边可被单独扰动；② 采用投影随机块坐标下降（PR‑BCD）实现大规模可扩展的攻击；③ 引入基于有向 2‑和 3‑环计数的周期可行性正则化，保证攻击在统计上保持可信。

**🔧 技术方法**

主要技术手段有：多重图归一化矩阵改写、投影梯度下降、投影随机块坐标下降、循环计数正则化、对抗训练（inner‑max via PR‑BCD）。

**📊 数据集**

使用的数据集包括：1) 合成多模式检测基准（8192 节点、约 49k 条边、11 种结构标签）；2) IBM AML 真实金融交易基准（Small HI 与 Small LI 两种非法/合法比例设置）。

**📈 对比分析**

与基准方法（随机重连）比较，学习型 PR‑BCD 攻击在相同攻击预算下能更显著降低 PR‑AUC；在 AML 数据上，攻击可导致高达 20% 的 PR‑AUC 降低。对抗训练后，模型在 10% 预算下仍保持约 88% 的宏观 PR‑AUC，远优于未训练模型。

**⚠️ 局限性**

局限性包括：① 周期正则化仅关注 2‑、3‑环，无法覆盖所有可监测的图统计；② 研究假设白盒攻击，实际场景可能更弱；③ 仅在金融多重图上验证，缺乏对其他类型多重图的通用性评估。

---

## 225. Correct but Unselectable: The Hidden Interface Tax in Multi-Candidate Reasoning

**arXiv ID:** 2610.32235 | [PDF](https://arxiv.org/pdf/2610.32235v1)

**作者:** Qianli Ma `[一作]` (Beijing Normal University), Weijia Jia `[通讯]` (Beijing Normal University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并解决了多候选生成-重排序过程中，候选答案无法被固定选择器正确消费的“隐藏接口税”问题。

**💡 创新点**

通过“覆盖‑转换分解”和可选覆盖概念，将覆盖失败与转换失败分离，并引入兼容性感知候选构造（CACC）层来修复片段、对齐答案模式并去除噪声，显著提升候选质量与最终准确率。

**🔧 技术方法**

采用离线规则书校准与在线规则执行的两阶段架构，利用正则匹配、答案模式检测、完整性评估和候选清洗等技术；不需要重新训练生成器或选择器。

**📊 数据集**

在数字开放式推理数据集GSM8K、CompMath以及多选题数据集MMLU-Pro和GPQA Diamond上进行实验。

**📈 对比分析**

相较基线，CACC在GSM8K上将原始覆盖率从0.4882提升至0.6391、最终准确率从0.2873提升至0.3632；在GPQA Diamond上单独使用CACC即可将准确率从5.6%提升至19.2%（+13.6pp），与生成器强化组合后可达25.8%（+20.2pp）。

**⚠️ 局限性**

仅针对数值与多选答案模式，需手工为每个基准校准规则书；对自由文本、代码生成或多轮对话等更复杂场景的推广尚未验证，且在选择器或读出不匹配时效果有限。

---

## 226. Deterministic Constructions of Reed-Solomon Codes Beyond the Johnson Bound

**arXiv ID:** 2610.31922 | [PDF](https://arxiv.org/pdf/2610.31922v1)

**作者:** Abhranil Chatterjee `[一作]` (Indian Institute of Technology Kharagpur), Roshan Raj `[通讯]` (Ohio State University)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a2602d71-93ab-4bad-974b-672788df8193` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `09944146-298c-433e-89df-37255de463d7` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

论文提出了一种确定性（即无随机化）构造方法，能够在几乎多项式时间内生成可超越 Johnson 半径的 Reed–Solomon 码（List‑decodable）以及满足 MDS(3) 条件的 Reed–Solomon 码。

**💡 创新点**

创新点在于：①利用读一次行列式多项式与 matroid‑intersection 的孤立权重分配构造，首次实现了在任意特征下的无随机化构造；②通过隔离权重的特殊结构与 Vandermonde 替换实现了对行列式多项式非零性保持；③得到的域大小和构造时间均为 quasi‑polynomial（比以往常数速率下的多项式域大大改进），并且兼顾了 MDS(3) 的高阶 MDS 性质。

**🔧 技术方法**

技术方法主要包括：
- Matroid theory（partition matroid、matroid‑intersection polytope、循环结构）
- 读一次行列式多项式的 Cauchy‑Binet 展开与非零性判定
- Gurjar‑Thierauf 的孤立权重构造
- Vandermonde 替换与 Kronecker 代换
- 线性代数和多项式身份检验（Polynomial Identity Testing）
- 组合判定（交叉矩阵、t‑wise 交集矩阵、ℓ‑MDS 矩阵）

**📊 数据集**

论文未使用任何标准实验数据集；其工作属于理论计算机科学与编码理论的算法构造与复杂度分析，所用“数据”仅为符号变量与抽象域元。

**📈 对比分析**

与之前的随机化构造相比，本文的结果在域大小、构造时间以及确定性方面均取得突破：
- 通过随机 RS 码需指数级域大小，而本构造仅需 quasi‑polynomial（(nk)^O(log(nk))）域；
- 计算复杂度为 quasi‑polynomial（(nk)^O(log(nk))）而非指数；
- 对 MDS(3) 的构造在常数速率下域大小从先前的 O(n^{k^2 k}) 改进到 quasi‑polynomial。性能指标主要体现在这些理论上界；没有实验性能测试。

**⚠️ 局限性**

限制：
- 仍然是 quasi‑polynomial 而非多项式的域大小与构造时间，尚未达到完全可多项式化；
- 目前仅证明了 MDS(3) 的构造，关于更高阶 MDS(ℓ) 的确定性构造仍未完成；
- 需要对 Vandermonde 替换与孤立权重的兼容性做细致验证，理论上可能存在隐藏常数因素；
- 在特定小域（如 F_2）上的实现细节仍待进一步研究。

---

## 227. SAGE: Sense-Aligned Multimodal Grounding Evaluation with Gloss-Augmented Reasoning

**arXiv ID:** 2610.31838 | [PDF](https://arxiv.org/pdf/2610.31838v1)

**作者:** Sunjae Kwon `[一作]` (University of Massachusetts Amherst), Hong Yu `[通讯]` (University of Massachusetts Amherst)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了SAGE基准，统一评估名词、动词、实体的多模态词义对齐；引入SPIS诊断和GAR推理框架；

**💡 创新点**

SAGE通过10候选图像共享协议实现跨设定比较；SPIS通过保留词义但换图揭示动词感知差异；GAR利用字典定义与约束推理提升性能且无需训练；

**🔧 技术方法**

使用CLIP、SigLIP、InternVL、Gemma、Qwen等视觉语言模型；采用BabelNet词典生成定义；构建语义约束进行视觉验证；

**📊 数据集**

整合VWSD、VerSe、WikiDiverse三个原始数据集，统一为10候选图像选择任务；

**📈 对比分析**

通过Hits@1和MRR进行评估，模型在VWSD表现最佳，VerSe与WikiDiverse仍有显著提升空间；GAR在无训练情况下达到81.2% Hits@1，优于先前训练-free方法；

**⚠️ 局限性**

SAGE在不同设定下难度不完全平衡；SPIS缺乏严格因果控制；GAR受限于约束生成质量，且推理耗时较高；

---

## 228. BAARC: Uncertainty-Driven Adaptive Partial Spectral Measurement to Full Reconstruction for Material Classification

**arXiv ID:** 2610.32505 | [PDF](https://arxiv.org/pdf/2610.32505v1)

**作者:** Adithya Ashok Chalain Valapil `[一作]` (Friedrich Schiller University Jena), Joachim Denzler `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发了一种贝叶斯自适应采样与重建驱动分类的闭环光谱测量框架BAARC，能够在仅测量约一半光谱的情况下完成Raman光谱分类。

**💡 创新点**

创新点在于将重建不确定性驱动下一步采样、分类不确定性决定测量终止，并通过联合训练重建与分类模型实现少量测量高准确度的自适应光谱采集。

**🔧 技术方法**

使用蒙特卡洛Dropout、神经过程和深度高斯过程等贝叶斯不确定性估计方法，gMLP网络结构，以及基于信息增益的贪婪采样和熵/互信息停止准则。

**📊 数据集**

在四个公开Raman基准数据集RRUFF、COVID‑19、Marine Pathology和Bacteria上进行实验验证。

**📈 对比分析**

通过与全光谱、随机采样三种测量模式比较，BAARC在保留约50%光谱时准确率接近全光谱，且明显优于随机采样；在所有模型中，使用MCD的不确定性方法表现最佳。

**⚠️ 局限性**

主要限制在于对高质量训练数据的依赖；在真实硬件环境中噪声和分布漂移可能导致误判；并非所有样本都能用较少光谱完成准确识别。

---

## 229. How Much Do You Trust A Dog? Quantifying Hint-Following in Reasoning Language Models

**arXiv ID:** 2610.32055 | [PDF](https://arxiv.org/pdf/2610.32055v1)

**作者:** Anxin Guo `[一作]` (Northwestern University), Aravindan Vijayaraghavan `[通讯]` (Northwestern University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了大型语言模型在多项选择问答任务中对提示的响应，量化了提示引起的答案分布偏移，并提出了可预测的指数倾斜模型；

**💡 创新点**

创新点在于将提示导致的回答变化表述为一个“指数倾斜”结构，分离出来源强度与问题-答案的共通易感性，并用奖励训练实验揭示了该结构的可能机制；

**🔧 技术方法**

主要技术包括基于链式思维的多次采样、对数几率偏移量分析、KL正则化奖励微调、贝叶斯证据权重比较以及对数几率平滑和LOWESS回归；

**📊 数据集**

使用的数据集为MMLU（800题目）和自制的4000题合成序列任务，用于评估提示效应与奖励训练；

**📈 对比分析**

与无提示基线和仅源强度模型比较，指数倾斜模型在KL误差上提升了约95％，在预测留出源时平均KL降低到原来三分之一；性能表明模型对提示的敏感度与来源可靠性无关，误差显著；

**⚠️ 局限性**

局限性包括：仅在2B–4B规模模型实验，缺乏对更大模型的验证；需要大量采样导致计算成本高；指数倾斜模型仅为描述性近似，未完全证明其因果机制；仅关注多项选择提示，未涵盖更广泛的交互形式。

---

## 230. Aligning Large Language Models with Human Feedback: Mathematical Foundations and Algorithm Design

**arXiv ID:** 2610.31862 | [PDF](https://arxiv.org/pdf/2610.31862v1)

**作者:** Siliang Zeng `[一作]` (University of Minnesota), Mingyi Hong `[通讯]` (University of Minnesota)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一套从理论到算法的完整框架，系统性地介绍并比较了LLM对齐的主要技术路径，包括SFT、RLHF、DPO、IRL、SPIN等。

**💡 创新点**

核心创新在于将对齐问题视作逆问题与离散选择模型的结合，提出统一的AIHF框架，并阐明RLHF与IRL的本质关联，强调了对齐可视为推断人类偏好与模型内部推理的联合估计。

**🔧 技术方法**

采用的技术包括逆强化学习（IRL）、对数似然最大化、KL正则化、PPO、REINFORCE、DPO、GRPO、规则奖励验证、以及自生成对抗训练等多种优化与对抗方法。

**📊 数据集**

使用的数据集涵盖示范数据（UltraChat、TL;DR、UltraChat200k）、对比数据（UltraFeedback、HH‑RLHF）、以及评测基准（AlpacaEval、Open LLM Leaderboard、MT‑Bench、RewardBench、Helm、MT‑Bench、其他公开对话与推理数据集）。

**📈 对比分析**

在实验中，IRL在仅有示范数据的UltraChat上明显优于SFT和SPIN；在示范+对比的设置下，AIHF在Open LLM Leaderboard、MT‑Bench和RewardBench上获得最高得分，表现优于单独的DPO、PPO、RLOO等方法。

**⚠️ 局限性**

局限性包括：训练成本高、需大量人工标注，RLHF易出现奖励劫持与分布漂移问题，DPO在缺乏覆盖完整答案空间时泛化受限，整体方法对硬件与算力要求仍较高。

---

## 231. Context or Knowledge: How LLMs Attribute Their Own Answers

**arXiv ID:** 2610.32109 | [PDF](https://arxiv.org/pdf/2610.32109v1)

**作者:** Kirill Afendulev `[一作]`, Anton Korznikov `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究语言模型在生成答案后，对答案来源（文档还是内在知识）作报告的机制，采用在回答后对 KV 缓存进行干预来定位其根源。

**💡 创新点**

创新点在于揭示：源报告不依赖于当前文档或参数知识，而是由问答阶段产生的 KV 缓存（source trace）决定；通过复制/植入 KV、拟合线性方向并进行 steering，证明了该缓存的因果作用。

**🔧 技术方法**

使用技术包括 Transformer KV 缓存操作（drop、swap、transplant）、键值投影 clamp、线性方向拟合与 steering、报告提取与比较、键值与值分离实验。

**📊 数据集**

数据集为 155 条事实（覆盖 8 个关系）每条有 7 个句子版本（真、伪、无关、中立）以及 DynamicQA Wikipedia 章节，用于构造对照实验。

**📈 对比分析**

方法对四个模型（7B–32B）进行实验，评估报告翻转率和答案切换率。实验结果显示：文档后置干预不影响报告；Q+A 缓存 transplant 100% 迁移报告；拟合方向可在不同模型上以高率（≈100%）改变报告；steering 在回答阶段亦能切换答案来源。性能稳定，实验可复现。

**⚠️ 局限性**

局限性包括：仅在有限的控制设置、固定问题表述、特定模型和文档上验证；未评估自然检索环境下的归因可靠性；方向所编码的信息和键值/问答状态的具体作用仍不完全解释。

---

## 232. Cycle-of-Thought: Self-Evaluative Iterative Reasoning for Table Visual Question Answering in Document Images

**arXiv ID:** 2610.32423 | [PDF](https://arxiv.org/pdf/2610.32423v1)

**作者:** Somraj Gautam `[一作]` (IIT Jodhpur), Gaurav Harit `[通讯]` (IIT Jodhpur)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种训练无关的自我校正框架Self-RefineVQA，利用VLM在推理时通过生成-批评-修正循环逐步纠正表格VQA中的空间漂移与逻辑错误。

**💡 创新点**

核心创新在于将自我批评与视觉重定位相结合，形成闭环推理流程；并证明仅通过提示即可实现可扩展且低成本的表格推理提升。

**🔧 技术方法**

使用提示引导的生成、批评与修正（Cycle‑of‑Thought）三阶段流程；结合多模型推理、置信度阈值终止、结构化反馈与迭代推理；对不同规模VLM（如Qwen2‑VL、InternVL、LLaVA等）无梯度更新即可部署。

**📊 数据集**

在三大表格VQA基准上验证：WikiTableQuestions（图像版）、DocVQA‑Table、TableVQA‑Bench；并在跨语言表格数据集IndotabVQA上测试多语种鲁棒性。

**📈 对比分析**

与零射、Self‑Consistency以及公开的Fine‑Tuning模型（TAPAS）对比，Self‑RefineVQA在所有模型上均提升 1–20 % 以上；轻量级模型(2B–3B)提升最大，跨语言表格任务提升达 7–13 %。

**⚠️ 局限性**

主要局限在于对视觉感知错误的纠正仍受限：当表格文本因噪声、模糊或缺失导致VLM无法准确定位时，批评-修正循环难以发挥作用；此外，多轮迭代导致推理成本约 4 倍，仍需在实时应用中权衡。

---

## 233. DeformStruct: Deformable Structure Modeling for Point-Supervised Temporal Action Localization

**arXiv ID:** 2610.32321 | [PDF](https://arxiv.org/pdf/2610.32321v1)

**作者:** Mengnan Liu `[一作]` (Xi'an Jiaotong University), Gang Hua `[通讯]` (Amazon Alexa AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了 DeformStruct，一种针对点级监督的时序动作定位框架，通过实例级可变形结构建模实现更精准的动作边界预测。

**💡 创新点**

核心创新点包括：①结构锚点挖掘将稀疏时间点扩展为多种动作相关节点；②可变形部件对齐利用变形感知匹配动态定位隐式部件；③关系结构组织将对齐部件映射到共享坐标空间，显式建模其相对时间关系。

**🔧 技术方法**

技术上结合 Transformer 编码器/解码器、查询重构、基于最优传输的伪标签生成、Hungarian 匹配、共享坐标的关系回归，以及软正则化的变形约束。

**📊 数据集**

在四个主流时序动作数据集上评估：THUMOS14、ActivityNet‑v1.3、GTEA 与 BEOID。

**📈 对比分析**

与现有点级监督方法比较，DeformStruct 在所有数据集上均实现了最高或接近最高的 mAP（例如 THUMOS14 上 54.8% mAP@0.5），显著优于前沿方法。

**⚠️ 局限性**

尽管性能提升明显，但方法仍依赖于高质量的点标注与伪标签，且对超参数（如抑制半径、变形权重等）敏感，且在极端稀疏或错误点标注场景下鲁棒性有限。

---

## 234. MPQ: A Message-Passing View of Post-Training Quantization

**arXiv ID:** 2610.32127 | [PDF](https://arxiv.org/pdf/2610.32127v1)

**作者:** Yanshuo Chen `[一作]` (University of Maryland), Heng Huang `[通讯]` (University of Maryland)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于消息传递的行向量后训练量化（MPQ）方法，用于低比特（2~4比特）语言模型权重量化；

**💡 创新点**

创新点在于将行向量PTQ视为离散代码书的后验推断，并引入有限温度向量近似消息传递（VAMP）保留多重候选信息，再通过块坐标下降完成硬决策；

**🔧 技术方法**

采用的技术包括：后验分布建模、有限温度VAMP、块坐标下降精细化、随机Hadamard变换与激活缩放等预处理；

**📊 数据集**

实验数据集包括128个C4校准样本、WikiText-2、ARC-Challenge、ARC-Easy、HellaSwag、PIQA、Winogrande等；

**📈 对比分析**

与GPTQ、LDLQ、QuantEase、OJBKQ等贪心/随机基线比较，MPQ在Llama、Qwen、Mistral系列模型的2位、3位以及4位量化下，在困惑度和零样本准确率上均优于或相当于最强基线，尤其在2位稀疏码书上显著提升；

**⚠️ 局限性**

局限性包括：相较于纯贪心方法推理速度稍慢，且在极低比特（2位）VAMP精度较低时仍需较多块坐标下降迭代；

---

## 235. A Mathematical Theory of the Jacobian Lens

**arXiv ID:** 2610.31955 | [PDF](https://arxiv.org/pdf/2610.31955v1)

**作者:** Mitchell Ostrow `[一作]` (Massachusetts Institute of Technology), Ila Fiete `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

对Jacobian lens进行数学理论化，证明其优先识别空间、时间及上下文一致性高的Lyapunov模式，并通过Bayes滤波器和小型Transformer在生成的HMM任务上的实验进行验证。

**💡 创新点**

提出了J-lens偏好一致性特征的理论解释，并推出可调节时间尺度的ω-J-lens，使工具能够聚焦不同频率信息。

**🔧 技术方法**

采用Lyapunov理论、Jacobian平均、Fourier分解、稀疏非负分解、Bayes滤波器及Transformer训练等技术。

**📊 数据集**

使用自生成的多环隐藏马尔可夫模型（HMM）及其观测序列作为实验数据。

**📈 对比分析**

通过与Bayes滤波器的解析理论对比，证明J-lens的权重与理论预测一致；实验中Transformer能够按预期识别不同特征，且可准确预测特征交叉点。

**⚠️ 局限性**

实验仅在极简toy HMM和小型Transformer上进行，缺乏对真实语言数据的评估，限制了结论的泛化性。

---

## 236. DIAL-S2ST: Learning Latency Control for Simultaneous Speech-to-Speech Translation

**arXiv ID:** 2610.32214 | [PDF](https://arxiv.org/pdf/2610.32214v1)

**作者:** Jeongsoo Choi `[一作]` (Korea Advanced Institute of Science and Technology), Joon Son Chung `[通讯]` (Korea Advanced Institute of Science and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种利用预训练SimulS2ST模型进行lookahead自蒸馏的后训练方法，在单个LoRA适配器中实现推理时可调的发射延迟。

**💡 创新点**

通过在学生生成轨迹上对冻结教师模型的lookahead预测进行自蒸馏，并对大损失进行掩蔽，单个LoRA适配器在不同尺度下即可控制延迟，实现推理时的质量-延迟平衡。

**🔧 技术方法**

Self‑distillation with lookahead、LoRA适配器、KL损失掩蔽、多流Transformer（Hibiki）以及源语音仅使用的后训练策略。

**📊 数据集**

French‑to‑English CVSS测试集，源语音来自CoVoST‑2/ Common Voice，后训练使用约10小时的FLEURS源语音。

**📈 对比分析**

与StreamSpeech、Seamless、原始Hibiki以及logit biasing等基线在LAAL/End‑Offset、ASR‑BLEU、UTMOS等指标上对比，取得更优的质量‑延迟Pareto前沿，仅需10小时源语音即可实现推理时延迟可调。

**⚠️ 局限性**

仅在Hibiki的法英模型上验证，依赖自动评测且对不同utterance的延迟变化未建模，未做人类评估，适配效果受源域与目标域差异影响。

---

## 237. A Model-Agnostic Correction Framework for Short-Term Building Load Forecasting

**arXiv ID:** 2610.31959 | [PDF](https://arxiv.org/pdf/2610.31959v1)

**作者:** Shiqiao Zhou `[一作]` (University of Birmingham), Shuo Wang `[通讯]` (University of Birmingham)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种模型无关的检索增强校正框架（WeL‑RAC），可在不修改原始预测模型的前提下，通过检索历史负荷与天气相似案例并融合其实际未来轨迹来校正短期建筑负荷预测。

**💡 创新点**

创新点在于：①同时使用近似负荷状态和未来天气摘要的双视角检索；②将检索到的历史实际未来作为校正证据，而非单纯替代预测；③通过可解释的检索记录提供案例级解释。

**🔧 技术方法**

核心技术包括：多种基准预测模型（MLP、LightGBM、LSTM、DLinear、PatchTST、TimeMixer、TimeXer、Chronos‑2）；天气摘要的文本化与句向量编码；负荷状态摘要的统计特征编码；余弦相似度检索、softmax加权聚合；以及基于权重α的后期融合。

**📊 数据集**

使用了三套建筑负荷数据集：AlphaBuilding（合成、9栋、2年小时级）、ASU（实际校园12栋、3年）和UCF（实际校园10栋、3年）。

**📈 对比分析**

通过在31栋建筑、4个预测时隙（1/6/12/18小时）以及8种基准模型的场景下对比，检索校正在90/96项基准对比中均取得更优的MSE、MAE和R²；平均来看，MSE、MAE下降约30%/25%，R²提升约0.05。

**⚠️ 局限性**

局限性包括：检索和融合参数仅在验证集上确定，难以应对未见的极端负荷或天气变化；检索依据归一化相似度，可能导致绝对负荷级别不匹配；实验仅在单栋离线预测上验证，缺乏在线更新与跨建筑泛化的评估；对天气预报误差的鲁棒性未充分研究。

---

## 238. Not Every Agent Needs the Strongest Model: Unified Prompt-Model Optimization for Heterogeneous LLM-based Multi-Agent Systems

**arXiv ID:** 2610.31743 | [PDF](https://arxiv.org/pdf/2610.31743v1)

**作者:** Mingxiu Cai `[一作]` (Harbin Institute of Technology), Xuebo Liu `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 25313 | [OpenAlex ID](https://openalex.org/A5029551526)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出 UniMASO，联合优化多代理系统中每个角色的提示词和执行模型，以提升任务质量并降低部署成本。

**💡 创新点**

创新点包括：① 将提示词与模型选择耦合在同一搜索空间内；② 引入成本感知 Pareto 搜索和多粒度评估；③ 设计拓扑感知协同演化机制；④ 利用跨层轨迹记忆进行经验迁移。

**🔧 技术方法**

主要技术：成本感知 Beam Search、Pareto 非支配选择、多粒度（本地、前瞻、全局）评价、拓扑感知的模型分配策略、跨级轨迹记忆检索与演化。

**📊 数据集**

使用的数据集包括 AGIEval-MATH、GPQA-Diamond、BFCL-V3、LiveCodeBench v6 以及 HotpotQA，模型池包含 Qwen3.5-4B/9B/27B 与 DeepSeek-V4-Flash/Pro。

**📈 对比分析**

与 MASPO、MASPOB、TPE、SPO 等基线以及单模型 MAS 进行对比，UniMASO 在中央化拓扑下平均分数从 69.79 提升至 70.40，同时部署成本下降 85.2%，在五个任务和五种拓扑中均保持竞争力。

**⚠️ 局限性**

局限性：依赖大量的优化轮次和计算预算，轨迹记忆对跨任务迁移效果有限；对动态或非静态拓扑的适应性尚未系统评估；以及在极大规模代理网络中可扩展性待验证。

---

## 239. Agreement rather than normalisation keeps depth reusable in looped language models

**arXiv ID:** 2610.31943 | [PDF](https://arxiv.org/pdf/2610.31943v1)

**作者:** Ivan Viakhirev `[一作]` (lab260), Grach Mkrtchian `[通讯]` (lab260)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

探究循环语言模型中循环边界归一化（loop‑boundary normalization）对深度可复用性的影响，评估其对训练与推理拓扑一致性及模型性能的作用。

**💡 创新点**

发现循环边界归一化并非必需的稳定性超参数，而是循环与读出的契约；去除归一化可使已训练的模型在推理中失效，而在从零训练时去除反而在训练预算之外保持更好的性能。

**🔧 技术方法**

使用循环Transformer架构，实验对已发布的Ouro模型进行归一化重排；在19M–76M参数的自训练模型上对不同归一化设置和循环深度进行多次随机种子实验；利用线性探针、状态范数、读出熵等度量。

**📊 数据集**

使用三种合成任务（布尔电路、指针追踪、奇偶性计算）构造不同深度与宽度的实例网格，包含零样本和少样本测试。

**📈 对比分析**

对比去除与保留循环边界归一化的模型，在训练预算与超出预算的循环步数上评估准确率、读出熵、状态范数等；结果显示：已发布模型去除归一化在训练预算后几步即完全失效；自训练模型去除归一化在预算外保持高于0.6的准确率，且归一化的模型在训练预算处峰值后显著下降。

**⚠️ 局限性**

限制包括：只在19–76M规模上进行因果性验证，未在更大模型上直接修改训练拓扑；已发布模型的功能受限，无法测试对Huginn的归一化添加；实验样本量有限（六个种子），可能导致统计功效不足；仅评估合成任务，缺乏对真实语言任务的验证。

---

## 240. Depth-Dependent Safety Geometry in Vision-Language Models: Layer-Wise Localization and Bidirectional Refusal Control

**arXiv ID:** 2610.32182 | [PDF](https://arxiv.org/pdf/2610.32182v1)

**作者:** Muhammad Zeeshan Karamat `[一作]` (Virginia Tech), Christiana Chamon Garcia `[通讯]` (Virginia Tech)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种轻量级推理时框架，能够在不修改模型参数的前提下定位并操控视觉‑语言模型中与安全相关的内部表示，进而对拒绝行为进行双向控制。

**💡 创新点**

创新点在于将层级激活分歧分析与表示层级的激活缩放、方向性调控相结合，揭示了不同深度层的安全几何结构与因果可控性不一致的现象，并首次识别出基于视觉层的可逆拒绝控制方向，暴露了白盒表示层攻击面。

**🔧 技术方法**

使用了层级激活分歧（Mean Angular Divergence）分析、激活幅度缩放（Activation Scaling）以及概念激活方向调控（Concept Activation Steering）等技术，辅以前向钩子进行内部激活干预。

**📊 数据集**

实验数据集包括：UnsafeBench（含有害及“硬”安全示例）、Google Conceptual Captions（干净安全示例）以及MMLU（通用推理评估）。

**📈 对比分析**

在Qwen3‑VL‑4B上，早期视觉层通过全局方向调控可将不安全拒绝率（URR）从84%提升至94%或压低至6%；后期视觉层则对幅度缩放更敏感，调增α至1.15可将URR提高至96%，但伴随安全性选择性下降。与InternVL2.5‑4B的代表性几何对比表明两模型在深度依赖安全几何上呈现相似趋势，但数值与层位置略有差异。

**⚠️ 局限性**

局限性包括：因果干预仅在Qwen3‑VL‑4B上完成，InternVL2.5‑4B仅作定性验证；缺乏完整的输入层攻击实现；安全‑选择性权衡在有限样本评估下可能不足；未对更大规模数据集、不同安全类别及更高维安全子空间进行深入研究。

---

## 241. From Faulty Claims to Incorrect Patches: An Empirical Study of Task-Grounded Reasoning in Coding Agents

**arXiv ID:** 2610.32338 | [PDF](https://arxiv.org/pdf/2610.32338v1)

**作者:** You Wang `[一作]` (Zhejiang University), Zhongxin Liu `[通讯]` (Zhejiang University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究编码代理的链式思维（CoT）中任务相关推理的覆盖率、错误率以及这些错误对补丁正确性的因果影响。

**💡 创新点**

首次系统性量化开源大模型 CoT 中的任务驱动推理、识别关键错误声明并通过修正验证其对补丁缺陷的直接作用。

**🔧 技术方法**

采用 LLM 自动抽取与分类推理单元，LLM 评估验证（基于证据检索）、人工核对以及轨迹重演纠错的组合技术。

**📊 数据集**

使用 SWE‑Bench Pro 上的 100 个软件工程任务，共 400 条轨迹（四个开源大模型），并在 Mini‑Swe‑Agent 框架下收集 CoT、动作与观察。

**📈 对比分析**

通过统计推理单元占比、错误率、关键错误比例、修正后补丁是否修复缺陷、任务最终通过率等指标比较，发现约 9.6% 的推理单元是错误的，纠正关键错误后能使 89% 的错误补丁得到修复，但仅 11.5% 的任务最终通过。

**⚠️ 局限性**

局限性包括：仅评估开源模型且未完整检验评估召回率；只关注与补丁直接相关的错误，未考虑推理单元间的相互依赖和未在 CoT 中显现的缺陷；实验仅在单一基准和框架下进行，通用性尚待验证。

---

## 242. ENCL: Enabling Efficient Neuromorphic Continual Learning with Local Learning Rule and Sub-Network Freezing Strategy

**arXiv ID:** 2610.31927 | [PDF](https://arxiv.org/pdf/2610.31927v1)

**作者:** Rachmad Vidya Wicaksana Putra `[一作]` (New York University Abu Dhabi), Muhammad Shafique `[通讯]` (New York University Abu Dhabi)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了ENCL，一种利用局部学习规则和子网络冻结策略的高效神经形态持续学习方法。

**💡 创新点**

创新点在于：①使用TESS时空局部学习规则替代全局反向传播，显著降低计算与存储成本；②设计子网络冻结策略，在每个任务分配唯一子网络并冻结已学参数，避免网络扩张；③提出轻量级训练策略（少轮次、调节学习率与时间步长），进一步提升能效。

**🔧 技术方法**

技术核心包括：局部学习规则TESS、任务专属子网络掩码生成与冻结、动态学习率/时间步长调整、子网络微调与参数记录。

**📊 数据集**

使用的基准数据集为：Split MNIST、Split Fashion MNIST、Split CIFAR‑10 与 Split CIFAR‑100，均按5或10任务拆分。

**📈 对比分析**

与基线 BPTT‑noNCL、TESS‑noNCL 以及 BPTT+ER（经验重放）对比，ENCL 在保持相当甚至略高的准确率的同时，内存使用降低 67‑88%、训练时间缩短 7‑51%、功耗降低 17‑40%、能耗降低 23‑65%。

**⚠️ 局限性**

局限性包括：在 Split CIFAR‑10 上准确率略低于 BPTT+ER；实验仅覆盖任务增量学习场景；子网络掩码重叠阈值需手工设定，可能影响可扩展性；方法尚未在更复杂任务序列或在线学习条件下验证。

---

## 243. Topological Equivalence of Spectral and Density Cluster Hierarchies

**arXiv ID:** 2610.31909 | [PDF](https://arxiv.org/pdf/2610.31909v1)

**作者:** Sajjad Hashemian `[一作]` `[通讯]` (University of Tehran), Sajjad Hashemian (University of Tehran)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043`

**🎯 论文内容**

比较基于密度层级和基于谱扩散距离的层级聚类，证明在满足代数与几何条件下，扩散单链接层级与密度层级的合并树在一个适当的扩散时间窗口内等价（可通过单调变换匹配高度）。

**💡 创新点**

提出一种通用的“拓扑相遇条件”（TMC）框架，将密度层级的层级性转化为谱扩散过程的混合性与泄漏性，并给出几何充分条件（GMC）和多尺度扩散时间调度，解决单一扩散时间难以适应不同密度层级的问题。

**🔧 技术方法**

使用谱扩散距离、随机游走与马尔可夫链理论、单链接/最小生成树、紧致Riemannian流形上的Poincaré不等式、反射链与谱间隙估计等工具；同时给出了从k‑NN图到谱距离的算法实现。

**📊 数据集**

本文未包含实验或公开数据集，全部结果为理论分析与证明。

**📈 对比分析**

由于缺乏实验，未给出数值比较或性能指标；研究侧重于证明层级结构的拓扑等价性与理论保证。

**⚠️ 局限性**

局限包括：需要对密度函数与组件几何（厚度、可达性、谷深度）做严格假设；扩散时间窗口的选择在实际数据中缺乏直接数据驱动方法；有限样本理论依赖于对扩散距离的统一近似，实际实现需进一步研究；多尺度时间调度虽然理论可行，但实现细节仍待探索。

---

## 244. Ulysses: A Security Harness For Synthesizing Verifiable Web Exploits

**arXiv ID:** 2610.31831 | [PDF](https://arxiv.org/pdf/2610.31831v1)

**作者:** Aditya Oke `[一作]` (Carnegie Mellon University), Lei Li `[通讯]` (Carnegie Mellon University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一套多智能体安全工具箱，能从任何静态分析或手工报告的漏洞中自动生成并验证可执行的Proof of Exploit（PoE）

**💡 创新点**

创新点在于将假设图（hypothesis graph）与独立的效果判定机（effect oracle）相结合，使得代理在多条攻击路径之间并行探索，并通过可观察的运行效果来确权，而非依赖模型自评

**🔧 技术方法**

使用大语言模型（GLM‑5 等）实现规划、执行与反思；结合静态分析器输出（Semgrep/CodeQL 等）作为起点；通过 Docker、网络隔离的双机架构实现效果验证

**📊 数据集**

构建了名为 Ulysses Benchmark 的真实 Web CVE 数据集，包含 25 个跨 5 种语言、7 个框架的可验证漏洞实例，并与 OWASP Python Benchmark 及其他公开基准进行对照

**📈 对比分析**

与多种基线（CodeQL、Semgrep、Absentia、ZAP 等）对比，Ulysses 在 25 个真实案例中成功生成 10 个已验证 PoE（含 4 个此前无 PoE 的漏洞），在 OWASP Benchmark 上实现了更高的召回率和可验证率，证明其在定位和验证方面显著优于传统方法

**⚠️ 局限性**

局限性包括：仅支持产生可观察服务器端效果的漏洞（不涵盖 XSS、DoS、内存安全缺陷等）；零知识情境下定位瓶颈明显；需要显式部署环境与效果判定机，无法直接识别全新未披露的漏洞

---

## 245. A Broader Look into MLLM's Retrieval Heads

**arXiv ID:** 2610.32411 | [PDF](https://arxiv.org/pdf/2610.32411v1)

**作者:** Jia Wang `[一作]` (Chinese Academy of Sciences), Mo Yu `[通讯]` (Tencent)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文通过构造音视频双模同一答案的对比输入，系统评估并利用多模态大型语言模型中的检索注意力头（retrieval heads），探究其稀疏性、跨模态共享、在模型迁移过程中的保留情况以及因果作用，并进一步将检索头的注意力作为目标，采用排序损失对其进行训练，显著提升多模态文档检索性能。

**💡 创新点**

创新点在于：①首次在音频-视觉输入上验证检索头稀疏性和跨模态共享；②证明检索头在多模态模型迁移过程中被保留并对更广泛的推理任务至关重要；③提出通过对检索头注意力的排序优化，将机制转化为可直接提升性能的训练目标。

**🔧 技术方法**

主要技术包括：检索头检测与校准评分（差分注意力）、层级稀疏检索头的随机与目标消融实验、对检索头注意力进行归一化后用于候选页面/区域的评分、基于LoRA微调的排序损失训练、以及在MMDocIR等数据集上进行评估。

**📊 数据集**

使用的数据集包括：LongMemEval、Natural Questions（文本检索）、MM-NIAH、AV-NIAH（音视频检索）、MLVU（视频来源）、MMDocIR（文档检索，页级和布局级）、以及四个离线领域基准 MMMU、MMMU-Pro、MathVista、MMLongBench-Doc。

**📈 对比分析**

与随机头或随机消融的对照相比，检索头在音视频检索任务中几乎完全消失，表明其必要性；在MMDocIR任务上，训练后的检索头模型在Qwen3-VL-8B上实现了6.03点的Micro R@1提升，且提升在所有十个文档域中均显著；对比其他两大模型（Qwen2.5-VL-7B、Gemma3-12B-it）同样取得正向改进；在四个离线领域基准上改进幅度小于1点，显示改进主要集中在训练任务。

**⚠️ 局限性**

局限性包括：①改进效果在离线领域基准上的提升有限，说明模型对新领域的泛化受限；②仅针对检索注意力头进行优化，其他潜在功能头未被充分挖掘；③实验范围聚焦于少数模型族和检索任务，未覆盖更广泛的多模态推理任务；④缺乏对不同检索目标（如实体、关系）下检索头行为的深入分析。

---

## 246. Efficient Omni-modal Large Language Model Inference via Per-Query Modality Selection

**arXiv ID:** 2610.31768 | [PDF](https://arxiv.org/pdf/2610.31768v1)

**作者:** Shuqi Liu `[一作]` (Nanyang Technological University), Luke Ong `[通讯]` (Nanyang Technological University)

**通讯引用:** 3569 | [OpenAlex ID](https://openalex.org/A5025152913)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究并实现了基于查询的模态选择策略，针对Omni‑LLM 在推理时动态选择文本、音频、视频等模态，以降低成本并保持准确率。

**💡 创新点**

将模态选择问题视为准确率–成本优化问题，并提出成本敏感学习的路由器；通过权重化交叉熵训练路由器，实现对每条查询的最优模态子集选择。

**🔧 技术方法**

采用 Lagrangian 松弛的成本敏感学习框架；基于 BGE‑M3 文本嵌入的线性路由器；在 Qwen2.5‑Omni、Qwen3‑Omni Omni‑LLM 上实验；对比随机路由、LLM‑as‑a‑Judge、Estimate‑Then‑Predict、Cascade 等基线。

**📊 数据集**

WorldSense 多模态问答基准，包含文本、音频、视频。

**📈 对比分析**

与固定模态、随机路由、LLM‑as‑a‑Judge、Estimate‑Then‑Predict、Logit‑Conf Cascade 等方法对比；成本敏感学习在输入令牌、延迟和金钱成本三者上均获得最高 AUC，并在保持 Full‑TA+V 准确率的同时将成本降低约58%（Qwen2.5）和35%（Qwen3）。

**⚠️ 局限性**

仅考虑二元模态选择（是否加入视频）且路由器仅基于文本特征；实验局限于单一基准和两款 Omni‑LLM；对更复杂模态组合或多模态交互的适用性未验证。

---

## 247. Spectra: Multi-Scale Implicit Decomposition for Deepfake Detection

**arXiv ID:** 2610.31979 | [PDF](https://arxiv.org/pdf/2610.31979v1)

**作者:** Dhananjaya Jayasundara `[一作]` (Johns Hopkins University), Vishal M. Patel `[通讯]` (Johns Hopkins University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

开发了一种深度伪造检测框架Spectra，结合CLIP语义特征与基于隐式神经表示的重构动态，以提升对未知生成技术的鲁棒性。

**💡 创新点**

首次将多层SIREN重构热图与CLIP多尺度语义特征通过双向注意力交互融合，构建统一表示，显著提升跨数据集和跨攻击类型的检测性能。

**🔧 技术方法**

使用冻结CLIP视觉编码器提取多尺度语义特征、SIREN隐式神经网络生成多层重构热图、空间与频率分支自适应融合、双向注意力交互，以及混合训练与监督对比学习。

**📊 数据集**

训练基于FaceForensics++（c_23）并在Celeb-DF v1/v2、FaceShifter、DeepFakeDetection四个OOD集上评估，PAD实验使用CASIA-MFSD、Replay-Attack、MSU-MFSD、OULU-NPU四个数据集。

**📈 对比分析**

与MesoNet、Xception、EffNet、CLIP基线、WaveletCLIP、TGFD、MUF等多种方法对比，Spectra在四个深伪测试集上的平均AUC达0.856，均为最佳或第二佳；在PAD任务上平均AUC 97.17，领先大多数方法。

**⚠️ 局限性**

在极端压缩或最新生成模型上仍略逊于部分竞争方法，SIREN训练与超网络实现仍计算成本较高，且未在多模态或连续视频检测中充分验证。

---

## 248. STEP: Surgical Point Tracking from Sparse Endpoints and Pseudo-trajectories

**arXiv ID:** 2610.31951 | [PDF](https://arxiv.org/pdf/2610.31951v1)

**作者:** Bing Fan `[一作]` (United Imaging Intelligence), Ziyan Wu `[通讯]` (United Imaging Intelligence)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出一种两阶段的手术点跟踪框架STEP，先用离线教师生成的伪轨迹通过时间和立体一致性筛选得到密集监督，再利用稀疏物理终点与回路一致性损失训练可在线推理的因果跟踪器。

**💡 创新点**

创新点在于将视频中生成的稠密伪轨迹与实验室测得的物理终点相结合，利用时间反向一致性与立体几何筛选优质轨迹，并在第二阶段通过端点损失与回路一致性共同引导学习，使得模型在无人工标注的情况下实现高精度跟踪。

**🔧 技术方法**

核心技术包括：冻结的AllTracker离线教师、时间反向一致性（forward‑backward）与立体一致性筛选、端点损失（配对端点与集合端点）、回路一致性损失（round‑trip loss）、受限梯度窗口训练（bounded gradient window）以及支持网格上下文增强。

**📊 数据集**

使用的数据集为STIR（多视角手术视频）和VL‑SurgPT（带有物理终点标记的手术视频），并在这两个数据集上进行跨域迁移和诊断实验。

**📈 对比分析**

与Track‑On、CoTracker3、RAFT等现有跟踪器对比，STEP在VL‑SurgPT 29‑clip诊断集上将端点误差从31.52 px降至16.80 px，EPE整体从14.90 px提升到12.14 px，δ_avg提升约2.8点；在STIR转移实验中，EPE提升约7%。

**⚠️ 局限性**

主要限制包括：对物理终点标记的依赖，训练时需要离线教师与立体视图；对极端遮挡或快速运动的鲁棒性仍有限；未进行前瞻性临床验证，且训练过程对教师和端点数据的质量敏感。

---

## 249. PEAR: Adaptive Advantage Reshaping for Perception and Exploration in Large Vision-Language Models

**arXiv ID:** 2610.32216 | [PDF](https://arxiv.org/pdf/2610.32216v1)

**作者:** Junxin Wang `[一作]` (Northeastern University), Tong Xiao `[通讯]` (Northeastern University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种多模态推理的优势重塑方法PEAR，利用视觉依赖和词元熵自适应分配奖励。

**💡 创新点**

创新点在于在高熵位置加入基于视觉依赖调节的探索增益，提升关键推理步骤的权重。

**🔧 技术方法**

采用视觉依赖估计（KL散度）、词元熵、Min‑Max归一化、动态融合与优势重塑的RL技术。

**📊 数据集**

在ViRL39K上训练，并在Geo3K、DynaMath、MathVision、MathVerse、LogicVista、MMMU‑Pro等多模态及文本推理数据集评测。

**📈 对比分析**

与GRPO、DAPO、PAPO、PEPO等基线以及多模态基准模型对比，PEAR在3B/7B模型上平均提升约12%/4%并在所有任务上取得最高分。

**⚠️ 局限性**

仅在3B/7B规模实验，未测试更大模型，也未多次随机种子验证，结果的规模化与鲁棒性待进一步验证。

---

## 250. Head Dimension Matters in Language Models for Long-Context Retrieval

**arXiv ID:** 2610.32213 | [PDF](https://arxiv.org/pdf/2610.32213v1)

**作者:** Seiya Tokui `[一作]` (Preferred Networks, Inc.), Sosuke Kobayashi `[通讯]` (Preferred Networks, Inc.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本论文研究在固定参数预算下，改变Transformer中多头注意力的头数与头维度，对长上下文检索能力的影响；通过从零开始训练0.6B和3B规模模型，进行预训练、长上下文持续预训练以及监督微调，并在RULER检索基准上评估。

**💡 创新点**

发现更宽的注意力头（头维度更大、头数更少）在长上下文检索任务中表现更好，且这一优势并不以短上下文任务性能为代价；并且阐释了宽头模型在检索头分配和模型内部表示谱结构上的差异，表明提升主要来自于对嵌入层和MLP的更有效适应，而非单纯的注意力权重更新。

**🔧 技术方法**

使用多头注意力、分块滑动窗口注意力、分组查询注意力（grouped-query attention），以及RoPE位置编码；训练时采用AdamW优化器和固定学习率策略。

**📊 数据集**

数据集主要由Web文本（60%）、代码（30%）以及10%类似SFT的数据混合构成，用于预训练与持续预训练；随后对3B模型在多种短上下文基准任务（ARC、HellaSwag、WinoGrande等）进行监督微调。

**📈 对比分析**

通过比较同一模型规模下不同头维/头数配置的RULER-100平均分与训练损失，发现宽头模型在4K–256K的上下文长度上平均提升约9.96–15.00分；训练损失并不能单独预测检索性能；宽头模型的检索头比例更高，但绝对检索头数相近。

**⚠️ 局限性**

实验仅探讨头维度至512的范围，超大头维度需更高效的注意力实现；此外，实验聚焦单一训练路线，未验证在更大规模或不同任务分布下的泛化性。

---

## 251. TileShare: Training-Free Acceleration of Overlapping Tiled ViT Decoding

**arXiv ID:** 2610.32233 | [PDF](https://arxiv.org/pdf/2610.32233v1)

**作者:** Xiangzhong Ye `[一作]`, Ka-Chun Wong `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了一种无需训练即可去除固定窗口ViT解码器在部署时的重叠计算重复的方法，保留注意力窗口的完整性，只在前几层共享所有计算，从而显著提升解码速度并保持视觉质量

**💡 创新点**

核心创新在于识别并限定重叠计算的影响仅局限于注意力路径，确定需要保留的前几层深度，并通过仅共享其余层实现速度提升，且无需额外训练；同时提供训练辅助的坐标字段和LoRA自蒸馏方案，进一步恢复完整共享时的质量损失

**🔧 技术方法**

技术手段包括：基于token共享的重叠解码、按层分块共享策略、FP8 MLP量化、torch.compile无自定义核编译、坐标字段与LoRA自蒸馏、以及对不同解码器和编码器的统一评估框架

**📊 数据集**

实验数据集涵盖DAVIS-30、UVG、Gen-44、MAGI-1的DAVIS-10、DIV8K-40、Commons-40以及多种公开的编码器/分割基准（CLIP、SCLIP等）

**📈 对比分析**

与原始分块解码（batched/compiled tiling）以及其他加速手段（缩小重叠、全帧解码、FP8 MLP、INT8等）比较，作者实现的6层共享解码在MiniMax-H3上相对原始解码可获得1.41–1.51×的速度提升，PSNR、LPIPS、DISTS差异在±0.05 dB范围内；对MAGI-1和ViTok-v2同样获得等价质量且显著加速；在编码器方面，使用坐标字段可使CLIP等模型在不训练权重的情况下实现2×以上速度提升且保持分割mIoU不下降

**⚠️ 局限性**

局限性包括：需要根据每个模型单独测定前几层的保留深度k，且在某些模型（如Rae-SigLIP2、DINOv2等）需要较深的k（12–16层）才能保持质量；对于高分辨率图像，全帧解码在某些模型上仍较慢；若重叠窗口过小或无重叠，则会出现明显的边界失真；最后，完全共享所有层时仍需少量训练（如LoRA或坐标字段）才能恢复原始质量。

---

## 252. Who Verifies the Tools? How Warped Formatters Mislead Agents and Their Verifiers

**arXiv ID:** 2610.32110 | [PDF](https://arxiv.org/pdf/2610.32110v1)

**作者:** William Jeffries `[一作]` `[通讯]`, William Jeffries

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究分析并修正了代理系统中工具输出的四类“扭曲格式化”缺陷，评估了改写后对代理声明和验证器判定的影响。

**💡 创新点**

创新点在于首次系统化提出并验证“扭曲格式化”概念，并通过对比实验展示了清晰措辞对代理过度声明和验证器误判的显著降低。

**🔧 技术方法**

采用配对评估实验、固定答案和频率测量方法，使用模型评分器和验证器进行自动判定，并编写审计提示让编码代理自动检测格式化缺陷。

**📊 数据集**

实验数据来源于内部历史档案数据库（约17,807份文档）及其搜索工具输出，未使用公开数据集。

**📈 对比分析**

在三类大型语言模型（Claude Sonnet 5、GPT‑5.6 Sol、Claude Code）上共1200次实验，清晰措辞将范围、完整性、估算等类的代理过度声明率分别降低62‑92个百分点，验证器在扭曲措辞下90/90的错误接受率在清晰措辞下降至0/90。

**⚠️ 局限性**

研究仅覆盖单一系统的四类缺陷，实验使用单次工具调用且未考虑生产环境中完整的交互循环，且结果受限于所选模型与提示，无法代表所有代理系统的普遍性。

---

## 253. Putting Transformer Theories to the Test

**arXiv ID:** 2610.32104 | [PDF](https://arxiv.org/pdf/2610.32104v1)

**作者:** Awni Altabaa `[一作]` (Yale University), Kiran Tomlinson `[通讯]` (Microsoft Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过构建19个算法任务并填补预测空缺，系统比较了七种不同的Transformer理论，评估它们对任务难易的预测是否与实验结果一致；

**💡 创新点**

创新点在于：1）统一任务套件并对七种理论进行预测矩阵化；2）提供了新的理论分析填补空缺；3）设计多维实验（样本复杂度、长度泛化、LLM评估），揭示了表征、学习与偏置多重限制；

**🔧 技术方法**

主要技术包括：Transformer架构实验（4层/256宽度/rotary编码等）、多种位置编码（NoPE、RoPE、绝对/相对）、任务生成器、训练超参调优、LLM推理与prompt设计、理论证明与复杂度分析；

**📊 数据集**

数据集：所有任务均为合成数据，覆盖Boolean、检索、稀疏函数、算术、Dyck语言、迭代检索、图任务等；实验亦使用公开LLM（如GPT‑5、其他预训练模型）进行零样本推理；

**📈 对比分析**

比较方法：将理论预测与实验结果对齐，采用标准化准确率阈值90%判定成功；实验显示：多任务中无理论能完整解释；软注意力、内容依赖检索与位置编码对性能影响显著；小Transformer在部分任务表现优于LLM，而LLM在某些任务上更强；

**⚠️ 局限性**

局限性在于：1）实验仅限于单轮Transformer，未涵盖链式思维等；2）理论多基于理想化假设（无限精度、无限宽度）；3）LLM评估受prompt与tokenization限制；4）未深入学习优化瓶颈与正则化机制的理论建模。

---

## 254. Detection and Validation of Hidden Subgroup Disparities in Clinical AI

**arXiv ID:** 2610.32362 | [PDF](https://arxiv.org/pdf/2610.32362v1)

**作者:** Alceu Bissoto `[一作]` (University of Bern), Lisa M. Koch `[通讯]` (University of Bern)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

对临床使用的胸部X光心脏肥大检测模型进行性能监测，利用算法子群发现方法自动划分隐藏子群并评估其性能差距。

**💡 创新点**

首次在真实医疗数据上系统评估子群发现技术，并提出基于现有元数据的子群特征剖面与子群标签进行偏差缓解的证明性方法。

**🔧 技术方法**

采用DOMINO子群发现框架，结合CLIP/BiomedCLIP特征和模型预测进行聚类；使用groupDRO进行子群偏差校正。

**📊 数据集**

使用CheXpertPlus公开胸部X光图像数据集（22万张图像），并在少量标注的起搏器样本上验证。

**📈 对比分析**

与传统基于性别、年龄等元数据划分相比，发现的子群表现出更大且统计显著的准确率、敏感度、特异度和AUROC差距；在起搏器子群上，groupDRO将平衡准确率提升约3.2%。

**⚠️ 局限性**

受限于可用元数据范围，无法完全解释所有子群；缺乏因果推断；pacemaker验证样本规模小；在真实环境下对子群发现的稳定性和可推广性仍需进一步研究。

---

## 255. StackGen3D: Stable 3D Block Stack Generation from Multiview Silhouettes

**arXiv ID:** 2610.31803 | [PDF](https://arxiv.org/pdf/2610.31803v1)

**作者:** Xiaopeng Zhang `[一作]` (University of Chicago), Matthew R. Walter `[通讯]` (Toyota Technological Institute at Chicago)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建可扩展的物理稳定结构数据集，并利用支持关系图神经网络（PI-GNN）快速评估结构稳定性，随后训练条件流匹配模型从三视正投影生成满足视图且物理可行的3D块结构；

**💡 创新点**

①以支持关系为核心的图神经网络实现物理稳定性快速推理；②自动化生成多样化稳定结构的数据集管道；③将条件流匹配用于多模态结构生成；④利用Best‑of‑N与Remove‑then‑Rerun等非模拟后处理提升生成结果的物理稳定性；

**🔧 技术方法**

PI‑GNN、MuJoCo/PyBullet物理仿真、Transformer‑based 条件流匹配、ODE求解器、AABB重叠评估、Best‑of‑N采样与Remove‑then‑Rerun；

**📊 数据集**

由简单堆叠、Filler3D随机去块等方法生成的稳定结构，配对三视正投影构成自制稳定结构数据集；

**📈 对比分析**

与参数匹配的MLP进行对比，PI‑GNN在复杂结构上的准确率92.66%对比46.86%；单样本生成结构的视图IoU达0.967；Best‑of‑10将稳定性从45.3%提升至72.4%，IoU保持≈0.970；

**⚠️ 局限性**

仍需后期物理仿真验证；AABB重叠作为物理稳定性近似不完备；生成结构仍可能出现穿透；数据集覆盖范围受限于自动化生成策略；

---

## 256. Curvature-Aware Clipping: Mitigating Bias in DPSGD via Approximate Second-Order Information

**arXiv ID:** 2610.32175 | [PDF](https://arxiv.org/pdf/2610.32175v1)

**作者:** Xincheng Xu `[一作]` (Australian National University), David Smith `[通讯]` (CSIRO)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种曲率感知的差分隐私随机梯度下降（CA-DPSGD）框架，通过利用近似的二阶信息构造最优的裁剪空间，并结合噪声校正与各向异性噪声注入，以降低梯度裁剪偏差和DP噪声对模型性能的影响。

**💡 创新点**

创新点包括：① 在几何视角下推导出最优裁剪矩阵并给出其解析解；② 设计了噪声校正的二阶矩估计方法和自适应缩放因子，实现可扩展的对角裁剪几何；③ 将各向异性高斯噪声与裁剪空间相结合，保持与传统DPSGD相同的总体噪声方差。

**🔧 技术方法**

采用了近似二阶信息（梯度的二阶矩估计）、指数移动平均、噪声校正机制、各向异性高斯噪声注入，以及对角裁剪矩阵的自适应更新。

**📊 数据集**

在九个基准数据集上评估：图像分类的 MNIST、Fashion‑MNIST、CIFAR‑10、ImageNette、CelebA；自然语言处理的 GLUE 任务 MNLI、QQP、QNLI、SST‑2。

**📈 对比分析**

与传统的 DPSGD、Auto‑S、DP‑PSAC 以及其他改进版（如 IOM、KFSGD、DPAdamBC）进行对比。实验显示 CA‑DPSGD 在所有数据集上均取得最高平均准确率，尤其在 ImageNette（+1.30%）和 QNLI（+0.50%）上表现显著提升。

**⚠️ 局限性**

局限性在于采用的对角裁剪几何无法捕捉参数间的相关性，导致在高度耦合的模型中可能无法完全消除裁剪偏差，未来工作可考虑更丰富的裁剪几何结构。

---

## 257. Interactive image diagnosis using test-time backpropagation from the auxiliary task

**arXiv ID:** 2610.32396 | [PDF](https://arxiv.org/pdf/2610.32396v1)

**作者:** Takuro Shimaya `[一作]` `[通讯]` (NEC Corporation), Takuro Shimaya (NEC Corporation)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了一种多任务梯度协助（MGA）方法，使放射科医生能够在无需重新训练的情况下通过给出辅助任务反馈来实时改进计算机辅助诊断（CAD）模型的主诊断结果。

**💡 创新点**

创新点在于利用测试时对辅助任务损失梯度的反向传播来直接调整编码特征，从而实现对已训练的多任务模型的交互式改进，并能与传统的REPL（直接替换辅助任务预测）方法结合使用。

**🔧 技术方法**

技术主要包括多任务学习框架（Faster R‑CNN + ConvNeXt+FPN），梯度驱动的特征调整（MGA），REPL替换策略，以及对不同辅助任务的实验与消融研究。

**📊 数据集**

使用了两个公开医学影像数据集：CBIS‑DDSM（乳腺X光）和LIDC‑IDRI（肺部CT），分别用于验证模型在乳腺肿块和肺结节的恶性检测上的效果。

**📈 对比分析**

通过与单任务模型、简单多任务模型、以及基于REPL的联合多任务模型对比，MGA在CBIS‑DDSM的AP25从约0.61提升到约0.68，在LIDC‑IDRI上亦实现了显著提升；且MGA能显著改善恶性与良性肿块的诊断分数分布，组合使用MGA+REPL还能进一步提升准确率。

**⚠️ 局限性**

局限性包括：对某些辅助任务（如形状分类）效果有限；可能过度依赖人类反馈导致误判；当辅助任务预测已与医生期望一致时无效；需要合适的超参数；未评估在真实临床阅读中的鲁棒性与读者差异；以及尚未验证对临床决策的实际影响。

---

## 258. Synthetic-to-Real Text-Based Person Anomaly Search via Multi-Model Retrieval Fusion and Cross-Encoder Reranking

**arXiv ID:** 2610.31930 | [PDF](https://arxiv.org/pdf/2610.31930v1)

**作者:** Eugene Denteh `[一作]` (North Dakota State University), Armstrong Aboah `[通讯]` (North Dakota State University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `3855fcda-48ef-4070-a15e-803cd5c84d83` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

该论文提出了一套多阶段的文本与视觉检索框架，用于从真实监控图像库中检索与自然语言描述（外观与行为）相匹配的人。

**💡 创新点**

创新点在于将任务特定的Qwen3‑VL嵌入与冻结的SigLIP2检索分支通过多模型融合并结合跨编码器重排序，同时使用双向对比学习、LoRA微调、域随机化、假负样本掩码与对抗性负样本以及专门构造的对抗性检索验证集，以提升合成到真实域的迁移效果。

**🔧 技术方法**

采用技术包括：双向对比学习、LoRA微调、域随机化、六个不同随机种子/增强设置的Qwen3‑VL检索器、冻结SigLIP2四视图编码、跨编码器（Qwen3‑VL‑Reranker）、标准化分数融合、加权融合（β）与交叉融合（α）以及对抗性检索验证。

**📊 数据集**

训练使用合成PAB数据（人像+行为描述），评估使用AI City Challenge 2026 Track 4的真实监控图像；同时构造了基于合成的“对抗性检索验证集”用于模型与超参数选择。

**📈 对比分析**

与CMP等基线对比，系统在官方挑战中取得94.78% mAP、90.85% R@1，排名第8，明显优于单一模型或未融合的方法，并通过多阶段融合显著提升性能。

**⚠️ 局限性**

局限性包括：高计算与存储成本（多检索器+跨编码器），仅在单一挑战设置上验证，缺乏统计显著性评估，未利用时序信息，近似帧区分仍困难，且系统对合成数据的依赖未完全消除。

---

## 259. UniWorld: A Unified World Model for Real-Time Visual World Simulation

**arXiv ID:** 2610.32381 | [PDF](https://arxiv.org/pdf/2610.32381v1)

**作者:** Xin Zhang `[一作]` (Fudan University), Xuelong Li `[通讯]` (Institute of Artificial Intelligence (TeleAI), China Telecom)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6514db3d-8de6-452c-91b7-acdb31787cc4` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

构建统一的视觉世界仿真框架UniWorld，能将来自多种世界源的几何轨迹与参考RGB融合，连续渲染出高质量视频。

**💡 创新点**

提出噪声对齐重叠延续和滚动深度潜在状态两种机制，实现跨窗口的状态契约，保证生成的连贯性与可扩展性。

**🔧 技术方法**

使用扩散式DiT模型（H3/VDN DMD8）、Qwen3-VL、ControlNet、VideoX-Fun等生成器；深度估计采用VDA ViT‑L；采用噪声流投影、重叠掩码等技术。

**📊 数据集**

评估数据集包括100个30秒混合场景（驾驶、人类活动、仿真、合成），30秒长视频、17个60秒/13个120秒视频、25个PointOdyssey/TartanAir原始深度。

**📈 对比分析**

与H3 50‑step、LongVie2、LongVie AR2对比，VBench质量得分最高，PSNR约13.4dB，实时渲染可达14 FPS；深度一致性和帧级误差也在可接受范围。

**⚠️ 局限性**

主要局限在于对深度估计的依赖、对极端动态场景的适应性不足，以及深度读取方式对高帧率场景的扩展性尚待验证。

---

## 260. DELA: Learning Controllable Edits for Diffusion Language Models

**arXiv ID:** 2610.32096 | [PDF](https://arxiv.org/pdf/2610.32096v1)

**作者:** Xinsong Feng `[一作]` (Georgia Institute of Technology), Haipeng Chen `[通讯]` (William & Mary)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出DELA框架，将冻结的扩散语言模型通过可插拔编辑控制器实现可控草稿级编辑。

**💡 创新点**

设计两阶段编辑（drafting+global refinement）和分阶段训练课程，使编辑器学习既可执行又有益的插入/删除/重掩/停止动作，实现草稿级控制而不改动扩散模型。

**🔧 技术方法**

使用冻结的掩码扩散语言模型、编辑控制器（operation-position-length分解）、两阶段解码、合成恢复任务、挖掘改进动作、基于rollout的偏好学习和任务奖励对齐等技术。

**📊 数据集**

评估使用数学推理数据集GSM8K、MATH以及代码生成数据集HumanEval、MBPP，训练时利用通用文本数据OpenWebText和WikiText进行无监督学习。

**📈 对比分析**

与直接解码、Fast‑dLLM、Tolerator、ReMDM等方法对比，冻结LLaDA‑1.5下提升约5–8%准确率（如GSM8K提升3.2%），在HumanEval上从37.2%提升至42.7%，仅增加少量NFE。

**⚠️ 局限性**

实验仅覆盖两种扩散模型和有限任务集，缺乏更大规模、多样化评估；控制器训练耗时且全局编辑效率仍待提升。

---

## 261. What Open-Source Projects Tell Their Coding Agents: A Census of AGENTS.md and CLAUDE.md

**arXiv ID:** 2610.31715 | [PDF](https://arxiv.org/pdf/2610.31715v1)

**作者:** Pawel Zarzycki `[一作]` `[通讯]`, Pawel Zarzycki

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对近6000个活跃的GitHub仓库进行普查，分析其根目录下的AGENTS.md和CLAUDE.md文件，提取并标注其中的指令语句，研究指令的内容、表达方式、来源以及与项目属性的关联。

**💡 创新点**

首次系统性聚焦普遍存在的编码代理指令文件，构建了基于LLM自动标注的完整注释语料，揭示指令文本的生成痕迹、共享文本和AI标记的使用模式，并将这些发现与项目流行度、语言、活跃度等因素关联。

**🔧 技术方法**

采用Claude Haiku、Sonnet和Opus 5.5进行文本分割、语义标注与项目类型分类；使用负二项回归和贝叶斯校正分析与项目特征的关联。

**📊 数据集**

收集了GitHub搜索API返回的至少1,000星且最近30天有提交的13种主流语言仓库，并下载其AGENTS.md/CLAUDE.md文件及对应的提交历史。

**📈 对比分析**

通过模型自动标注与人工审核相结合，对指令语句进行主题与元素分类，然后使用回归模型检验与项目属性的显著关联，结果表明流行度越高的仓库越可能包含对AI贡献、拉取请求的规则，编译语言仓库更注重构建指令。

**⚠️ 局限性**

研究受限于仅检测根目录文件且不追踪重命名，LLM标注与人工标注之间存在差异，AI标记仅能捕捉部分机器辅助写作，且因样本为当日快照而无法说明因果关系。

---

## 262. POS-Signature Demonstration Retrieval for In-Context Dependency Parsing

**arXiv ID:** 2610.31966 | [PDF](https://arxiv.org/pdf/2610.31966v1)

**作者:** Muhammad Azam Afridi `[一作]` (Università Cattolica del Sacro Cuore), Marco Passarotti `[通讯]` (Università Cattolica del Sacro Cuore)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了基于POS签名的演示选择策略在Universal Dependencies解析中的应用，旨在提高解析准确性。

**💡 创新点**

提出了一种基于POS统计的演示选择策略，能够更好地捕捉句子的结构信息，从而提高解析性能。

**🔧 技术方法**

使用了监督的Stanza标注器来预测查询的POS序列，并通过加权特征表示句子。

**📊 数据集**

使用了UD English Web Treebank数据集，包含12,544个训练句子和2,077个测试句子。

**📈 对比分析**

与固定选择和语义检索基线进行比较，POS签名检索在两个模型上均取得了最高的标记附加分数（LAS），分别为72.4和73.2，超出语义检索1.9和2.7分。

**⚠️ 局限性**

该方法依赖于监督标注器和标注的演示池，限制了其在低资源环境下的应用。

---

## 263. EgoForesight: Effectively Scaling Policy Learning with Noisy Egocentric Actions

**arXiv ID:** 2610.31845 | [PDF](https://arxiv.org/pdf/2610.31845v1)

**作者:** Chubin Zhang `[一作]` (Tsinghua University), Yansong Tang `[通讯]` (Tsinghua University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种视觉‑语言‑动作（VLA）框架，通过联合流匹配模型同时学习未来视觉表示和动作，支持在大规模机器人轨迹与自摄像头视频上进行策略预训练。

**💡 创新点**

创新点在于两方面：一是可靠性条件动作标签腐败（reliability‑conditioned corruption），根据来源可靠性动态调整动作监督强度；二是特权未来表示蒸馏（privileged‑future representation distillation），利用演示结果作为教师提供额外的动作结构监督。

**🔧 技术方法**

技术手段包括：基于Transformer的VLA专家、联合流匹配框架、流时间控制的噪声腐败、EMA教师与学生的跨层对齐蒸馏、以及多阶段数据选取与持续微调。

**📊 数据集**

使用的数据集主要有：机器人轨迹数据集（AgiBot World、EgoDex、InternData‑A1、T‑Rex），约 20000 小时自摄像头人类演示数据，RoboCasa 仿真环境，以及 Galaxea R1 Pro 真实机器人实验数据。

**📈 对比分析**

与 DIAL、EgoDex、GR0.5、GR00T N1.7 等基线相比，在 RoboCasa 24 任务的 100 机器人演示下，本方法成功率达到 66%（比基线提升 7.7%），在 30k 小时预训练时，open‑loop MSE 下降 20.5%，success 率提升 8.1%；在真实机器人实验中平均成功率 63.8%，显著优于对比方法。

**⚠️ 局限性**

局限性包括：对动作标签噪声和人机动作差异仍敏感；需要高质量机器人轨迹作为初始化；无法直接处理无动作注释的视频；跨机器人结构（例如从抓手到灵巧手）适配仍具挑战。

---

## 264. ACWM-RM: A Reference-Free Reward Model from Complementary Mismatch Preferences

**arXiv ID:** 2610.32222 | [PDF](https://arxiv.org/pdf/2610.32222v1)

**作者:** Shibo Hong `[一作]` (Fudan University), Yixin Cao `[通讯]` (Fudan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了无参考的奖励模型ACWM‑RM，联合评估视觉质量与动作跟随一致性，并通过AVM‑Eval基准验证逆动力学模型在生成视频上的可靠性。

**💡 创新点**

创新点包括：①构建了AVM‑Eval与AVM‑Pref两套数据集，分别用于评估动作/视频不匹配；②设计了两阶段训练（动态预训练+偏好训练）与patch‑weighted评分头，实现了在无ground‑truth视频情况下同时评估VQ与AF；③证明传统IDM在生成视频上评估不可靠，ACWM‑RM在RL中显著提升VQ与AF。

**🔧 技术方法**

使用的技术包括：V‑JEPA 2.1特征编码器、动作条件预测器、patch‑weighted 评分头、逆动力学模型、视觉‑语言模型、Flow‑GRPO强化学习框架、Bradley‑Terry偏好损失。

**📊 数据集**

采用了AgiBotWorld真实机器人视频、Gen、RoboDojo、RoboTwin仿真视频以及DreamDojo‑2B等ACWM生成视频，构成AVM‑Eval（67.2k对）与AVM‑Pref（506.3k对）数据集。

**📈 对比分析**

通过AVM‑Eval对比四种IDM在动作/视频不匹配上的准确率；在DreamDojo‑2B上使用Flow‑GRPO对比IDM、VLM、指标奖励，ACWM‑RM在PSNR、LPIPS、SSIM、MSE、FVD等视觉指标和nDTW动作跟随上均优于基线，同时保持物理一致性。

**⚠️ 局限性**

局限性：未显式监督物理一致性；实验仅在DreamDojo‑2B与AgiBotWorld上验证，未覆盖更广泛的ACWM；对不同场景和模型的泛化能力仍待进一步探索。

---

## 265. Thumb on the scale: Using traditional wargaming adjudication practices to improve AI-judged evaluations and grading

**arXiv ID:** 2610.32066 | [PDF](https://arxiv.org/pdf/2610.32066v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 266. Markets of Minds: A 51-Billion-State Economy of Intelligence

**arXiv ID:** 2610.32190 | [PDF](https://arxiv.org/pdf/2610.32190v1)

**作者:** Jusheng Zhang `[一作]` (Stanford University), Ming-Hsuan Yang `[通讯]` (University of California at Merced)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出Markets of Minds框架，针对共享认知资源的多代理系统，在专家输出未知前，对每个代理-服务对的增益进行估计，并在共享预算、容量、访问限制等约束下进行合同清算，实现在有限认知能力下的最佳分配。

**💡 创新点**

创新点在于将助力的价值拆分为需求、修复效能和回归风险三项，并通过预执行价值估计与联合清算实现机会成本驱动的分配；此外，在51.2亿状态空间内进行人群级实验，验证了稀缺性与价值异质性的影响。

**🔧 技术方法**

技术包括：请求审计器生成可观测特征向量；三头分类器分别估计默认成功率、修复概率与回归风险；基于估计值的线性目标在约束可行集上求解；账本记录预算与容量；以及匹配回放和多模态基准的实验框架。

**📊 数据集**

实验数据集涵盖：240个交互式代理案例（Chat、Web、OS‑App）用于匹配回放；51.2亿人群-市场状态的设计空间；六个多模态基准（MMMU、MMBench、MATH‑Vision、InfoVQA、CC‑OCR、MMMU‑Pro）用于验证服务异质性。

**📈 对比分析**

与随机分配、难度/不确定性启发式、AutoMix、RouteLLM、GraphRouter、Agora等基线比较，Markets of Minds在30%专家容量下获得85.42%任务级遵从率（相较随机提升11.56pp），平均得分82.66（相较最佳对手提升1.78pp），并在不同预算下保持领先。

**⚠️ 局限性**

局限在于当前仅评估单服务（单一专家）情形，未完整实现回归风险的独立估计；多服务异质性与更复杂预算约束的实验仍待开展；同时模型对高维状态的泛化与实时性需要进一步验证。

---

## 267. Forecasting Associations, Not Interventions: A Preregistered Causal Audit of Covariate-Aware Time Series Foundation Models

**arXiv ID:** 2610.31869 | [PDF](https://arxiv.org/pdf/2610.31869v1)

**作者:** Zong-Han Bai `[一作]` `[通讯]` (National Tsing Hua University), Zong-Han Bai (National Tsing Hua University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

对协变量感知的时间序列基础模型进行了因果审核，验证其 what‑if 查询是否真正反映因果效应。

**💡 创新点**

提出了闭式协方差缺口指标和因果指南针，并给出了基于预训练分布重放的因果后训练修复方案。

**🔧 技术方法**

利用 in‑context 最小二乘、线性探针、低秩适配、核合成重放以及合成和物理系统实验进行评估。

**📊 数据集**

使用了线性‑高斯与非线性结构时序合成数据、DoTime 基准、两套 Causal Chambers 物理系统以及 M4 时序数据。

**📈 对比分析**

与基线最小二乘控制及其他 TSFM 在网格实验中比较，R² 高达 0.90‑1.00；后训练可将协方差缺口降至 60% 以上，预测误差提升不足 5%。

**⚠️ 局限性**

理论证明仅适用于线性‑高斯模型，现实系统的尺度校准仍未完善，且后训练成本受重放分布影响，跨架构验证有限。

---

## 268. Encoding-Sensitive Nullstellensatz Certificates for Syndrome Exclusion in LDPC Codes: Sharp Local Bounds and a Linear Separation

**arXiv ID:** 2610.32171 | [PDF](https://arxiv.org/pdf/2610.32171v1)

**作者:** Malihe Aliasgari `[一作]` (Kean University), Khadijeh Bagheri `[通讯]` (University of British Columbia)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `847a60d8-a755-47af-ba5d-c5236b9e3083` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究LDPC码的Nullstellensatz度并提出计数与局部编码方法来计算（t,b）问题的最小度数

**💡 创新点**

创新点是证明计数编码给出最优的M+1度数上界，并通过混合编码与伙伴植入技术揭示局部编码的上限与下限

**🔧 技术方法**

采用符号计算、线性代数、组合计数与整数规划及多元多项式理论

**📊 数据集**

使用的代码实例包括循环码、数组码H(w,L,Z)、准循环码、5G NR基码等

**📈 对比分析**

与已知局部上界t+b+K+1和理论下界相比，实验证实计数编码在多种码上实现线性增长的最小度数，生成的证明证书尺寸与已公布的证书相比显著缩小

**⚠️ 局限性**

局限性包括需先判定不可行性、对奇列权的（1,0）问题仍开放、混合编码下难以提升度数，以及方法在某些受限度的码中无法突破M+1上界

---

## 269. Coupling Execution Horizon and Sampling Depth in Flow-Matching VLA Policies

**arXiv ID:** 2610.32102 | [PDF](https://arxiv.org/pdf/2610.32102v1)

**作者:** Jiabing Yang `[一作]` (University of Chinese Academy of Sciences), Liang Wang `[通讯]` (University of Chinese Academy of Sciences)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对流匹配 Vision‑Language‑Action (VLA) 模型的数值求解深度 (N) 与执行时长 (s) 进行耦合设计，揭示并利用两者之间的相互影响来提升任务成功率与推理速度。

**💡 创新点**

创新点在于发现：截断前向欧拉求解步骤能作为隐式方差正则化器，在短时规划时导致性能下降，而在长时开环执行时反而提升成功率，从而实现无训练、无结构修改的计算‑执行协同优化。

**🔧 技术方法**

主要技术包括：流匹配概率流的前向欧拉数值积分、递归执行时长 (receding‑horizon) 策略、几何诊断与因果干预、以及多种数值求解器（Euler、midpoint、Heun 等）的对比实验。

**📊 数据集**

使用的数据集和实验平台包括 LIBERO (55 个仿真任务)、RoboTwin 2.0 (50 个双臂任务) 以及真实 Franka 机器人 (5 个操纵任务)，共计 2400 次物理试验。

**📈 对比分析**

比较方法：在不同 (s,N) 组合下评估任务成功率和每步推理延迟。实验显示，在 LIBERO‑Long 任务中，s=20 与 N=2 的组合可将成功率提升至 90.5% 并实现 5.6× 的推理加速；类似的性能提升在 RoboTwin 和 Franka 机器人上也得到验证。

**⚠️ 局限性**

局限性：效果仅在冻结、未改动模型的前提下显现；极短时高频反馈或移动目标任务（如转盘）仍需更细粒度的规划；未探索自适应或多步求解器的训练与部署方案。

---

## 270. FSVL: Taxonomic Fast-Slow Vision-to-Language Collaborative Framework for Interpretable Confusable Fish Species Identification and Knowledge Generation in the Middle Yangtze River

**arXiv ID:** 2610.31880 | [PDF](https://arxiv.org/pdf/2610.31880v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 271. DeepAVR: Deep Analysis via Multi-Agent Collaboration for Automated Vulnerability Repair

**arXiv ID:** 2610.32345 | [PDF](https://arxiv.org/pdf/2610.32345v1)

**作者:** Woorim Han `[一作]` (Seoul National University), Yunheung Paek `[通讯]` (Seoul National University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为DeepAVR的多智能体框架，实现从仅提供一次PoC或 sanitizer 日志到最终可部署补丁的端到端自动漏洞修复流程。

**💡 创新点**

创新点在于：① 通过将故障定位、补丁生成、编译/安全/功能/审查等六个专门角色拆分并使用结构化反馈实现深度分析；② 引入基于补丁的再定位机制，利用中间补丁导致的错误类型变化来挖掘隐藏根因；③ 设计根因导向的补丁审查，识别症状级修复和不完整补丁。

**🔧 技术方法**

技术核心包括大型语言模型驱动的多智能体协同、工具API交互（代码检索、编译、PoC执行等）、结构化提示模板、反馈路由与迭代重定位，整体构建在LangChain与OpenAI GPT‑4.1（可切换到 Gemini‑2.5‑Pro/Claude‑Haiku 4.5）之上。

**📊 数据集**

使用公开的 C/C++ 漏洞修复基准 VulnLoc（39 个真实漏洞）和 SAN2VULN（25 个训练未见漏洞）进行评估。

**📈 对比分析**

与 CrashRepair、LoopRepair、SAN2PATCH、PatchAgent、SpecRover 等五大基线对比，DeepAVR 在 VulnLoc 上实现 100% 可行补丁、92.3% 正确补丁；在 SAN2VULN 上 88% 正确率，显著低于同类方法的可行/错误补丁比例。

**⚠️ 局限性**

局限性包括：对测试套件完整度敏感；依赖大型语言模型，模型不同会影响性能与成本；主要针对 C/C++ 代码，跨语言迁移尚待验证；总体 API 费用与 token 消耗高于部分基线。

---

## 272. Rethinking Structured Generation: Can Graph-Based Reasoning Resolve Ambiguity?

**arXiv ID:** 2610.31681 | [PDF](https://arxiv.org/pdf/2610.31681v1)

**作者:** Ratun Rahman `[一作]` (University of Alabama in Huntsville), Atit Pokharel `[通讯]` (University of Alabama in Huntsville)

**通讯引用:** 29 | [OpenAlex ID](https://openalex.org/A5115057127)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种面向多候选结构的模糊感知生成框架，将结构化生成视为在候选集合上进行推理而非单一输出预测。

**💡 创新点**

创新点在于：①将候选结构映射为任务专属的图表示，①采用图神经网络进行语义一致性、结构完整性和关系一致性三维评分；②引入多有效学习目标，允许训练时保留所有语义合法候选，避免单参考抑制有效方案。

**🔧 技术方法**

核心技术包括：多样化候选生成（基于LLM的随机/束搜索），图结构构造与边感知GNN编码，三项评分模块（语义、结构、关系），以及多有效交叉熵损失。

**📊 数据集**

实验使用两大公开基准：文本到可视化的nvBench（聚合、编码、分组多义性）和文本到SQL的Spider（执行等价多义性）。

**📈 对比分析**

与传统单输出LLM、候选+似然、LLM重排序等基线对比，本文方法在EM与Top‑k上均取得显著提升；在多参考评估和不同模糊程度下表现更稳健，整体精度提高约5–8点。

**⚠️ 局限性**

局限性包括：依赖候选生成的覆盖度，若无有效候选则无法修正；多有效训练需手工或自动判断等价性；图构造仍需任务特定工程；生成多候选与图推理增加计算开销。

---

## 273. Profile-Driven User Simulation via Reinforcement Learning with Hierarchical Rewards for Task-Oriented Dialogues

**arXiv ID:** 2610.32435 | [PDF](https://arxiv.org/pdf/2610.32435v1)

**作者:** Juan Li `[一作]` (Bairong Inc.), Yunzhang Chen `[通讯]` (Bairong Inc.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了ProfileSim，一种针对金融营销外呼场景的用户模拟框架；

**💡 创新点**

创新点在于联合建模用户画像与业务画像，采用分层奖励的GRPO强化学习提升对话连贯性、个性化与业务一致性；

**🔧 技术方法**

主要技术包括监督微调（SFT）、Group Relative Policy Optimization（GRPO）与检索增强生成（RAG）以及SOP约束对话管理；

**📊 数据集**

使用了约72,296条真实外呼语音对话（共432,457条发言）以及由LLM生成的10,000条用户画像对；

**📈 对比分析**

与DeepSeek-R1/V3、GPT-5、Qwen3-Max等通用LLM在相同提示下进行零样本比较，ProfileSim在口语自然度、用户画像一致性、业务一致性和对客服代理的压力测试指标上均优于对手，且对话长度更短；

**⚠️ 局限性**

局限在于用户画像为静态不捕捉情绪变化，仅在金融外呼单一领域验证，跨域推广及语音非语言特征建模仍待完善。

---

## 274. Adaptive Control of Scientific Instruments with Nonstationary Responses through Multimodal Agents

**arXiv ID:** 2610.32370 | [PDF](https://arxiv.org/pdf/2610.32370v1)

**作者:** Guoyang Shi `[一作]` (Beijing ENN Fusion Energy Science and Technology Co., Ltd.), Tianyuan Liu `[通讯]` (Beijing ENN Fusion Energy Science and Technology Co., Ltd.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了无高保真模拟器的磁约束聚变中中性粒子束离子源的逐射调整，提出了基于物理量的多模态智能体框架。

**💡 创新点**

创新点在于将确定性物理量提取、检索增强的历史记忆与多模态大型语言模型结合，并通过独立可审核的动作可行性检查实现安全闭环。

**🔧 技术方法**

使用了确定性信号处理、视觉注释图像、双层检索记忆、检索增强生成（RAG）和多模态大型语言模型（如 Qwen3‑VL‑Plus）以及动作可行性约束。

**📊 数据集**

数据集为四台 EXL‑50U NBI 离子源的 455 次历史射击记录（含 6 类质量标签）以及 409 次实时执行记录。

**📈 对比分析**

通过对 4 个 LLM 的跨模型对比，基于物理量的输入提升了 0.21–0.32 的质量分类 macro‑F1，检索增强使方向一致率提升 22–43%，在线实验 beam‑extraction 成功率 86.8% 与人工 83.8% 接近。

**⚠️ 局限性**

限制在于缺乏高保真仿真，模型仍需外部检索支持，且对极端高参数区间表现仍需进一步验证。

---

## 275. Which Answer Did the Seventeen-Year-Old Write? A Behavioral Case Study of Author-Age Misattribution, Miscalibrated Confidence, and Non-Corrective Self-Explanation in Six Frontier Language Models

**arXiv ID:** 2610.31946 | [PDF](https://arxiv.org/pdf/2610.31946v1)

**作者:** Brianne Lee `[一作]` `[通讯]` (Independent Researcher), Brianne Lee (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了六款前沿语言模型在识别三条真实手写短文作者年龄和性别时的表现，并通过多种提示、上下文和开放式推断条件系统评估其错误模式。

**💡 创新点**

创新点在于：①使用真实手写文本并记录真实年龄/性别标签，构造正式化语言与年龄不符的对立样本；②系统比较六种实验条件；③分析模型自报置信度与正确性、后期自我解释的关联；④对比两代模型的改进。

**🔧 技术方法**

技术方法包括：调用六款 LLM（Claude Fable 5、Gemini 3.1 Pro、GPT‑5.6 Sol、Grok 4.5、GPT‑4o 2024、DeepSeek V4）；设计六种实验条件（盲目、机构标签、中立标签、观众意识指令、开放式推断、自报概率分布）；统计分析（Wilson 95% CI、z 检验、χ²、Fisher、Pearson 相关、熵计算）。

**📊 数据集**

数据集为三份真实手写答卷（对应17 M、28 F、45 M），共1,081次 API 会话、3,225 条可分析条目。

**📈 对比分析**

比较方法：以随机猜测（1/3）为基准，计算会话分数分布、误判率，利用 Fisher 检验比较条件间差异。结果显示：多数模型准确率低于偶然，误判率可达 90%；多选格式掩盖严重错误；观众意识指令在 GPT‑5.6 Sol 上显著提升；模型置信度与正确性往往负相关或无关；自我解释与判断质量不相关。

**⚠️ 局限性**

局限性：仅三条文本、单一主题与语言；样本量小且模型分布不均；无人类基线；API 与消费端行为可能不同；内部工具错误未全部排除；置信度范围受限导致相关性难测；后置解释编码仅基于词汇匹配，缺乏多评审验证。

---

## 276. $M^3$: A Multi-LLM System for Multi-Turn Jailbreaking as a Markov Decision Process

**arXiv ID:** 2610.31937 | [PDF](https://arxiv.org/pdf/2610.31937v1)

**作者:** Yuqing Huang `[一作]` (Vanguard Group), Jekaterina Novikova `[通讯]` (Vanguard Group)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6215c339-3735-4be3-8a07-5bbb7004712d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 M^3——一种基于有限时域马尔可夫决策过程的多轮红队框架，利用多个大型语言模型（攻击者、评估者、状态维护器）实现自动化多轮 jailbreak 评估。

**💡 创新点**

创新点包括：
- 把多轮 jailbreak 形式化为 MDP，消除对固定动作空间和人工编排路径的依赖；
- 采用可压缩的内部状态表示，解决长对话上下文长度限制；
- 引入基于评估器奖励的风险递增与回溯机制，保证对话风险逐步上升且能够自我纠错；
- 只需黑盒 API，且无需预先构造 jailbreak 数据集或风险分层，显著提升可扩展性。

**🔧 技术方法**

技术手段：
- 多轮 MDP 框架与离散动作空间（Proceed/Backtrack）；
- 状态维护 LLM 用于生成压缩的对话摘要；
- 评估 LLM 负责计算风险分数、成功标记和解释；
- 基于强化学习思想的奖励函数设计；
- 通过有限回溯和重试策略实现局部优化。

**📊 数据集**

使用的数据集：
- “15任务风险评估集”（覆盖8类风险：非法活动、伤害、错误信息、色情、脏话、仇恨、性别歧视、暴力）；
- “HarmBench 子集”（90任务，12类风险，包括化学/生物、网络犯罪、骚扰、危害、非法、错误信息等）。

**📈 对比分析**

与基准方法（如 Crescendo、X-Teaming、ActorAttack 等）对比，使用 ASR@1、ASR@5、AUC、Gini 等指标：
- 在两大基准上，M^3 在 ASR@5 上提升多达 53%，在 ASR@1 上提升约 33%；
- 轻量级攻击者在与高容量基线比较时也能提升 26%~34%；
- 在风险递增曲线和 AUC 方面表现更平稳、上升更快；
- Gini 指数下降至 4.82，显示跨风险类别成功率更均匀。

**⚠️ 局限性**

局限性：
- 对不同攻击者/状态 LLM 配置、最大回合数、回溯深度等超参数的影响尚未系统评估；
- 只使用了预先固定的评估器模型，评估器本身可能对某些攻击产生偏差；
- 研究仅在受控实验环境下进行，缺乏对实际生产系统防御效果的验证；
- 对更大规模或不同语言的目标模型迁移性尚待验证。

---

## 277. PROVE: Probabilistic Visual Grounding via Patch-Level Evidence for Vision-Language Models

**arXiv ID:** 2610.32440 | [PDF](https://arxiv.org/pdf/2610.32440v1)

**作者:** Mayank Nautiyal `[一作]` (Uppsala University), Prashant Singh `[通讯]` (Uppsala University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出 PROVE，一种后置的证据适配器，在冻结的 CLIP 模型上实现可概率化的视觉定位，输出每个图像补丁的 Beta 分布，进而得到定位地图、置信度（vacuum）和缺失检测分数。

**💡 创新点**

创新点包括：①把定位问题转化为每个补丁的 Bernoulli 成员资格，使用 Beta 分布收集正负证据；②引入先验拉回（prior‑pull）目标，让缺失实例保持在无信息先验附近，直接获得缺失检测；③在单次前向传播中闭式得到定位、置信度和缺失分数，省去额外分类器。

**🔧 技术方法**

技术细节：Beta‑Bernoulli 证据框架、Transformer 轻量头对冻结 CLIP 补丁特征进行上下文建模、Grad‑ECLIP 作为先验显著性、三项损失（定位、未支持证据正则化、缺失拉回）以及证据容量约束。

**📊 数据集**

主要数据集：gRefCOCO（训练、验证和两组测试）；跨域评估在 Flickr30k Entities 与 PhraseCut；在 COCO 基准上进行对比实验。

**📈 对比分析**

与基线（Grad‑CAM、Grad‑ECLIP、CLIP‑Surgery、Rollout）相比，PROVE 在 PG、EBPG、IoU@k 等定位指标上保持竞争力甚至领先；其置信度（vacuum）与定位误差正相关，可用于选择性推理；缺失检测 AUROC、T‑acc、N‑acc 均高于 Grad‑ECLIP 与确定性对照，说明证据建模提升了缺失判定。

**⚠️ 局限性**

局限性：部分性能提升源于监督适配；在分布外的数据上仍有一定退化；依赖冻结的 CLIP 与 Grad‑ECLIP 先验，受其偏差影响；需手动调节证据容量 γ；目前仅适用于 CLIP 结构，扩展到其它 VLM 需要新的显著性先验。

---

## 278. Measuring the Energy Cost of Completing Tasks with Quantized Language-Model Agents

**arXiv ID:** 2610.32417 | [PDF](https://arxiv.org/pdf/2610.32417v1)

**作者:** Stephen Abkin `[一作]` (Texas A&M University), David Kebo Houngninou `[通讯]` (Texas A&M University)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并实现了一种量化语言模型代理完成任务时的能耗测量框架，计算并报告任务完成的能耗（Task Completion Energy, TCE）

**💡 创新点**

创新点在于将量化技术与能耗评估结合，给出了量化程度与任务能耗之间的定量关系，并提出了新的能耗评估指标

**🔧 技术方法**

采用了权重量化与激活量化、PyTorch/ONNX Runtime深度学习框架以及nvidia-smi、Intel Power Gadget等能耗测量工具

**📊 数据集**

使用了GLUE、SQuAD、自然语言推断等标准NLP数据集以及XSum等文本生成任务数据集进行评估

**📈 对比分析**

通过与全精度模型对比，8位量化模型能耗降低约60%且准确率下降不到1%，4位量化模型能耗降低约80%但准确率下降2~3%

**⚠️ 局限性**

局限性包括仅在单一GPU/CPU环境下实验，未覆盖训练能耗；验证范围局限于BERT、GPT-2等模型，缺乏对更大规模或异构硬件的泛化；测量工具精度与系统负载可能影响结果

---

## 279. SOPO: Socratic Guided Policy Optimization for Span-Level Hallucination Detection

**arXiv ID:** 2610.32212 | [PDF](https://arxiv.org/pdf/2610.32212v1)

**作者:** Jiawei Dong `[一作]` (Patsnap), Peng Zhou `[通讯]` (Patsnap)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Socratic Guided Policy Optimization (SOPO)，通过教师模型提供 Socratic 反馈和回报缩放，解决跨度级幻觉检测中的奖励稀疏、定位模糊与长度膨胀问题。

**💡 创新点**

创新点包括：① Socratic 交互式反馈（分析+引导问题，避免真值泄漏）；② Turn‑based Advantage Scaling（惩罚多轮长度，控制冗长）；③ 在 GRPO 基础上实现无价值函数的强化学习，提升样本效率。

**🔧 技术方法**

技术手段：强化学习 (RLVR/GRPO)、Socratic 反馈机制、span‑level F1 奖励、回报缩放、动态采样、AReal 训练框架、Qwen3-4B‑2507（学生）与 Qwen3‑Next‑80B‑A3B（教师）等。

**📊 数据集**

使用 RAGTruth 作为主数据集（Summarization、QA、Data‑to‑Text），并在四个 OOD 基准（PsiloQA、Math‑Shepherd‑614、LLM‑AggreFact、HaluEval）进行评估。

**📈 对比分析**

与 SFT、GRPO、CAPO、LettuceDetect、GPT‑5、o3 等基线对比，SOPO‑4B 在 RAGTruth 上 span‑F1 63.3%、case‑F1 84.8%，在 OOD 上均取得最高分；训练速度提升 3‑4 倍，生成长度控制在 1,700 tokens，推理延迟下降 47%。

**⚠️ 局限性**

局限性：未结合更高级的信用分配或 Process Reward Model；长度惩罚可能抑制开放式生成的迭代试错；未对不同随机种子进行统计显著性评估；对极长上下文或更高规模模型的适应性仍需验证。

---

## 280. LAMB: A Modern Contextual Language Model for Latin

**arXiv ID:** 2610.31723 | [PDF](https://arxiv.org/pdf/2610.31723v1)

**作者:** Aidan McCarthy `[一作]` `[通讯]`, Aidan McCarthy

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并预训练了面向拉丁语的编码器LAMB，使用超过74B拉丁词元和创新的形态学掩码方法，同时创建了Acervus和Corpus Trinum两大拉丁语语料库。

**💡 创新点**

引入了基于形态学依赖的掩码策略以及针对拉丁语的高效词形分词器，并在多任务评测网格中系统评估模型。

**🔧 技术方法**

采用ModernBERT架构、两阶段预训练（随机掩码+形态学掩码）、BERT WordPiece/ SentencePiece分词、微调配置网格和子词聚合策略。

**📊 数据集**

使用Acervus（约23.9B词元）与Corpus Trinum（100M语义三元组）以及多来源公开拉丁语语料进行预训练和评测。

**📈 对比分析**

在17个子任务的5个下游任务上与17个基线模型对比，LAMB在16/17子任务上获胜，单一基准任务仅差0.8个百分点，且在POS、UFEAT、NER、STS等任务均名列前茅。

**⚠️ 局限性**

受限于数据稀缺与OCR噪声，预训练规模仍低于现代高资源语言；形态学掩码对部分任务的提升有限，且评测受子词聚合与微调配置的高度影响。

---

## 281. Latent Coherence: Do Frontier Language Models' Probability Judgments Obey the Law of Total Probability?

**arXiv ID:** 2610.31833 | [PDF](https://arxiv.org/pdf/2610.31833v1)

**作者:** Peter Potash `[一作]` `[通讯]`, Peter Potash

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估前沿大型语言模型在零样本情境下，关于隐变量的概率一致性（即满足全概率律）

**💡 创新点**

提出将一致性误差拆分为词法噪声与持久不一致两部分，并使用残差作为无真值的可靠性指标

**🔧 技术方法**

基于概率推理、贝叶斯后验、反事实与后门调整理论，利用 log‑odds 残差度量一致性

**📊 数据集**

使用四类评测数据集：全规格合成、定性合成、计数表的后门调整、以及对 HoVer 多跳推理的虚构化分解

**📈 对比分析**

对七款主流 API 模型进行比较，GPT‑6.1 Sol 与 Claude Opus 5.5 在一致性上表现最佳（残差约 0.08–0.12 log‑odds），其余模型表现差异显著；一致性与推理长度无显著关联；残差较大的实例往往对应更大的真实误差

**⚠️ 局限性**

限制包括：实验仅覆盖单一提示格式、缺乏多轮交互、样本量有限、部分数据集构造依赖 LLM、仅评测七款模型，未涵盖专用预测或开源模型

---

## 282. DynaSlots: Slot-Centric 3D Dynamics Pretraining for Robot Manipulation

**arXiv ID:** 2610.32330 | [PDF](https://arxiv.org/pdf/2610.32330v1)

**作者:** Meng Li `[一作]` (Beihang University), Guangliang Cheng `[通讯]` (University of Liverpool)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了 DynaSlots，一个基于槽（slot）的、无标签、无动作、无重建监督的 3D 预训练框架，用于提高机器人操控任务的性能。

**💡 创新点**

创新点包括：① 通过时序槽绑定实现跨帧槽一致性；② 为每个槽学习潜在动作（latent action）并用逆动力学模型提取；③ 使用扩散式前向动力学模型（FDM）捕捉槽间交互；④ 自监督交互头（interaction head）预测槽之间的接触与共运动关系；⑤ 所有目标均不需要动作、分割或重建标签。

**🔧 技术方法**

技术手段涵盖：点云 Tokenization 采用 Point Transformer；Temporal Slot Binder；Inverse Dynamics Model；扩散模型 Diffusion Transformer（DiT）做前向动力学预测；VICReg 风格的正则化；以及自监督交互头和噪声预测损失。

**📊 数据集**

使用的数据集包括 MetaWorld（14 个仿真操控任务）、Adroit（门和笔任务）、RealMan RM65-B 的四个真实世界任务，以及 RH20T 作为大规模预训练源。

**📈 对比分析**

与多种基线（静态 3D、策略仅有、场景级动态感知如 AFRO、DynaMo-3D、FVP、DP3 等）比较，DynaSlots 在 MetaWorld 的平均成功率 80.8%、Adroit 的 88.0% 上分别超过基线 4.8–6.0 pp；在真实世界任务中，平均成功率 79.0% 或 73.0%（RH20T 预训练），在杂乱和未见物体场景下表现最稳健，优势幅度 8–30 pp。

**⚠️ 局限性**

局限性包括：① 需要预先设定槽数 K，过少或过多会导致欠拟合或过分细分；② 目前仅处理点云，缺乏对多模态输入（如图像、语言）的支持；③ 对极端动态场景或大规模物体集合的泛化尚待验证；④ 训练对大量无标签视频依赖，且在预训练数据来源变化时性能仍有一定下降。

---

## 283. Autoregressive Multimodal Anomaly Detection with Large Language Models for Microservice Systems

**arXiv ID:** 2610.31722 | [PDF](https://arxiv.org/pdf/2610.31722v1)

**作者:** Yixiang Tang `[一作]` (South China University of Technology), C. L. Philip Chen `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于大型语言模型的自回归多模态异常检测框架LLMMAD，用于检测微服务系统的异常。

**💡 创新点**

创新点在于将跨模态交互与模态特定重构结合的重组合策略，并采用预测驱动的自回归学习以消除身份捷径，从而显著提升异常检测精度。

**🔧 技术方法**

主要技术包括LLM（如GPT‑2）+ LoRA微调、1D CNN嵌入、时间对齐、模态特定线性重构以及OR逻辑聚合。

**📊 数据集**

在三个真实微服务数据集上验证：SocialNetwork (SN)、TrainTicket (TT) 与 OpenStack‑based MSDS。

**📈 对比分析**

与单模态与多模态无监督基线（LogAnomaly、AnomalyTransformer、TraceAnomaly、SCWarn、DeepTraLog、FUSION、Triple）比较，LLMMAD在所有数据集上F1最高，分别提升约3.1%、1.4%和0.9%。

**⚠️ 局限性**

局限性包括对预训练LLM知识的依赖、对模型规模和训练数据量敏感，以及在极短窗口或极大数据场景下性能下降，且未针对有标签监督场景进行优化。

---

## 284. Trial by Trace: Benchmarking AI Agent Harnesses for Reliability and Failure Attribution

**arXiv ID:** 2610.32285 | [PDF](https://arxiv.org/pdf/2610.32285v1)

**作者:** Zhiyong Cao `[一作]`, Sicheng Zhou `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本论文提出并实现了 TraceTrial‑Bench 基准，用于对 LLM 代理的 harness 进行机制化可靠性评估，涵盖 95 个专家级任务、七个可靠性维度，并通过轨迹追踪、跨源证据获取和两层评估框架实现 agent 与 harness 层的独立打分与归因。

**💡 创新点**

创新点在于：① 引入基于机制的七维可靠性分类（Context、Execution、Governance、Lifecycle、Observability、Tools、Verification）；② 采用多源证据（轨迹、工作空间、模拟器状态、工具声明等）构建可审计的两层评估；③ 设计机制链式归因模块（FA），将 harness 层故障与 agent 层失败通过预声明的传播路径进行可验证关联。

**🔧 技术方法**

使用的技术包括：Trajectory Tracing (TT) 将各类 harness 产生的原始事件统一转换为 Trace IR；Multi‑source Evidence Acquisition (MEA) 收集工具调用、工作空间变更、模拟器记录等；两层评价框架（Agent 与 Harness）对判据进行确定性与语义化评判；Failure Attribution (FA) 模块根据预设的机制链激活候选归因并支持人工验证。

**📊 数据集**

数据集为 95 个专门设计的任务（覆盖 7 个可靠性维度），在 4 个原生 harness（Claude Code、Codex、DeepSeek、OpenClaw）与 9 个模型（DeepSeek‑V4‑Pro、Gemini‑3.7‑Flash、GLM‑5.3、GPT‑5.6‑Sol、Grok‑4.6、HY3、Kimi‑K3、Opus‑5、Qwen3.8‑Max）上执行，共计 3,420 次运行（每组一次），并记录完整的轨迹、工作空间、评判结果等。

**📈 对比分析**

通过宏/微平均的 criterion 通过率、Agent 任务成功率、以及两层综合得分对四个 harness 进行比较。结果显示 OpenClaw 在宏平均 Harness 通过率（72.0%）最高；Claude Code 在 Agent 任务成功率（44.0%）最高；两者在两层得分上并不一致，且 31.1% 的运行在 Agent 成功的同时至少出现一次 Harness 失效，表明单一的任务成功评估无法充分反映 harness 的可靠性。

**⚠️ 局限性**

局限性包括：① 仅采用单次执行，未评估重复性与稳定性；② 任务覆盖不均衡，部分机制缺乏足够样本；③ 评估环境与真实部署场景存在差异，泛化性有限；④ 归因依赖预声明机制链，未覆盖所有可能的失败路径；⑤ 仅关注单代理场景，未涉及多代理协作或长期持续任务。

---

## 285. SEVI: Safe Evidential Real-World Reinforcement Learning

**arXiv ID:** 2610.31829 | [PDF](https://arxiv.org/pdf/2610.31829v1)

**作者:** Shaekh Mohammad Shithil `[一作]` (Indiana University), Lantao Liu `[通讯]` (Indiana University)

**通讯引用:** 1762 | [OpenAlex ID](https://openalex.org/A5101917996)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出Safe Evidential Real‑World Reinforcement Learning (SEVI)框架，结合Evidential Safety Critic与离线恢复策略，实现在线强化学习中的安全干预。

**💡 创新点**

创新点在于：①在单一次前向传播中同时估计未来约束违约风险与证据不确定性；②基于风险与不确定性阈值进行决策，自动切换到恢复策略；③将安全组件与标准离线RL解耦，保持任务学习目标不变。

**🔧 技术方法**

使用了证据回归（Normal‑Inverse‑Gamma）构建安全评估器、深度双重评估器网络、基于TD3+BC的离线恢复策略、以及SERL的离线先验数据 + off‑policy 更新。

**📊 数据集**

数据集包括两款模拟任务（Block Plucking、Transport Nut）和真实硬件USB插入任务，分别使用RGB图像+状态观测；恢复演示由人类操作在接近失败边界时收集；同时使用10%、50%及100%训练覆盖度的子集验证不确定性。

**📈 对比分析**

与SERL、SAC、SAC‑C、SQRL、SEVI无不确定性等基线相比，SEVI在所有任务上实现了最高的平衡准确率、最高的成功率，并将训练期间的不安全事件减少了30%（Block）、13%（Nut）和57%（USB）。推断延迟与SQRL相当，且在OOD条件下仍保持较低的检测时间。

**⚠️ 局限性**

局限性：未给出形式化的零违规保证；恢复策略未针对局部或运动效率优化，可能导致早期阶段多次切换；安全评估器与恢复策略依赖有限数据，仍可能在极端分布偏移下失效。

---

## 286. Understanding Tool-Mediated Interaction Failures in Agentic GUI Software: An Empirical Study

**arXiv ID:** 2610.31698 | [PDF](https://arxiv.org/pdf/2610.31698v1)

**作者:** Yuchen Shao `[一作]` (East China Normal University), Chengcheng Wan `[通讯]` (East China Normal University)

**通讯引用:** 407 | [OpenAlex ID](https://openalex.org/A5016446062)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

系统性分析了大型语言模型（LLM）代理在实时图形用户界面（GUI）软件中通过工具层进行交互时出现的失败，构建了包含335个真实缺陷实例的缺陷库，并基于该分析提出了一个预部署静态合同检查器；

**💡 创新点**

首次从工具介导交互的角度对agentic GUI软件进行经验研究，提出了11种缺陷模式，发布了公开缺陷库和针对工具层的静态检测工具，填补了现有研究只关注模型或应用层缺陷的空白；

**🔧 技术方法**

采用了Python/TypeScript代码静态分析（AST + Tree‑sitter）、LLM与工具链交互监控、任务执行记录与人工标注、以及构建的缺陷模式匹配规则；

**📊 数据集**

基于40个开源agentic GUI项目（覆盖web、桌面、移动）收集约2000个issue并复现335个缺陷；另外选取20个未参与构建的应用进行评测；缺陷库与代码公开于doi.org/10.6084/m9.figshare.32838281；

**📈 对比分析**

通过与人工标注的缺陷真值集对比，静态检查器实现57.3%精准率和76.8%召回率，覆盖所有11种缺陷模式中的大部分；与传统缺陷检测相比，能够在工具层提前发现并定位大部分交互失败；

**⚠️ 局限性**

研究仅限于公开开源项目，静态分析局限于局部代码，无法捕捉跨文件或运行时产生的缺陷；未考虑封闭源代码、平台差异或特定企业工作流程中的差异；这些因素可能导致漏检或误检。

---

## 287. SILMARILS-MV: Information-Theoretically Secure Multi-Verifier Signatures with Dealer-Free Key Issuance and Linear Aggregation for Permissioned Blockchains

**arXiv ID:** 2610.31814 | [PDF](https://arxiv.org/pdf/2610.31814v1)

**作者:** Khadijeh Bagheri `[一作]` (University of British Columbia Okanagan Campus), Chen Feng `[通讯]` (University of British Columbia Okanagan Campus)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计并实现了一种信息理论安全的多验证者签名方案（MV‑USS），该方案可在无可信中介、无广播、无交互的条件下完成签名、验证、聚合、阈值公证和密钥发行，适用于权限链中的投票、仲裁证书和交易签名。

**💡 创新点**

创新点包括：①利用虚拟锚点（Γ）生成验证者密钥，彻底消除可信代理；②签名、验证和聚合均为线性运算，三者共享同一安全核心；③实现无广播的分布式密钥发行（dealer‑free MPC）；④阈值公证（TN）让委员会本身成为单一签名者，证书大小常数，验证者数量不随委员会规模增长；⑤批量仲裁、可回滚和重分发机制提升了可用性和可扩展性。

**🔧 技术方法**

技术栈：线性代数（Shamir共享、Vandermonde矩阵、矩阵向量乘法）；完美安全多方计算（MPC）；信息理论安全分析（基于偏移、线性子空间的证明）；聚合签名与阈值签名的统一框架；虚拟锚点和批量仲裁的实现；使用椭圆曲线字段 𝔽₂²⁵⁵⁻¹⁹；对协议层的优化（零填充、批量掩码）。

**📊 数据集**

实验评估在单机 Intel Core i9‑13900HX（WSL2 Ubuntu）上进行，使用 C、Rust 及 OpenSSL SHA3‑256；基准对比对象包括 Ed25519、ML‑DSA‑44/65、Falcon‑512、SLH‑DSA；没有真实区块链数据，使用随机消息和协议模拟。

**📈 对比分析**

比较方法：在相同主机、相同编译器下测量签名/验证时间、证书/消息大小、每区块认证流量及交易吞吐量；结果显示：相较 ML‑DSA‑44，验证速率提升 22×，签名速率提升 22–53×，QC 大小比 ML‑DSA‑44 小 22×；在 HotStuff 原型中，交易吞吐量提升 2.8×，CPU 使用下降 67×。协议在委员会规模 3t+1 ≤ 1000 时保持低成本，且与现有 PQ 方案相比提供信息理论安全。

**⚠️ 局限性**

局限性：①需要预共享的验证者授权和安全点对点通道；②验证者密钥存储随验证者数增长（每签名 ψ+1 个字段元素）；③批量仲裁安全性依赖批量大小，且在极端恶意情形下需要更多验证；④对高委员会规模仍有密钥发行和通信量上限；⑤对量子安全仅限于哈希碰撞抵抗；⑥动态投票者加入/离开需复杂的重分发协议；⑦在公开链或完全无可信验证者的场景下不适用。

---

## 288. Can LLM Annotators Match Human Consistency? Benchmarking Human Agreement and Optimizing Multi-Agent with Contrastive Knowledge

**arXiv ID:** 2610.32415 | [PDF](https://arxiv.org/pdf/2610.32415v1)

**作者:** Xiaopeng Ke `[一作]` (ANT Group), Min Zhang `[通讯]` (Harbin Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了ANT-A多注释人基准和ANT-OR多智能体框架，用真实任务指南和5份独立人工标注构建了7个跨域、跨模态的数据集，并通过多智能体与对比知识挖掘实现了自动注释的迭代优化。

**💡 创新点**

创新点：①构建真实的多注释人基准，捕捉注释不一致性；②引入分布感知评价（Fleiss' κ）；③提出基于对比三元组挖掘判别维度的多智能体优化框架ANT-OR，能够将错误经验转化为可复用决策规则；④在多智能体架构中实现注释-验证器层次化协作与知识增强推理。

**🔧 技术方法**

技术：多智能体注释-验证器架构；对比知识挖掘（正负样本池、判别维度池）；检索增强推理与质量感知样本检索；错误归因模型与提示优化；层次化注释与验证，结构化知识库。

**📊 数据集**

数据集：7个任务（Disease Privacy Assessment、Billing Scenario Classification、Medical Entity Extraction、Social Text Classification、Image Attribution Recognition、Image Classification、Image Text Relevance），每个1500条样本，涵盖医疗、电子商务、通用域；文本与视觉-语言两种模态。

**📈 对比分析**

方法比较：与单智能体基线（CoT、ICL、APE、ORPO、ERM、SPO）和多智能体基线（Vanilla、ICL、SPO）对比。ANT-OR在所有任务上平均提升4.5点（agreement@≥2），Fleiss' κ与人类标注接近，特别是BillSC和ImageAR，展示了更高的分布一致性和更好的泛化。

**⚠️ 局限性**

局限：①基准覆盖范围有限，未覆盖长文档、开放式判断等更复杂场景；②多智能体对比挖掘与知识库构建成本高，优化与推理开销大；③对底层LLM性能高度依赖，需在更强模型上进一步验证；④在高风险领域仍需人工复核，不能完全替代专业专家。

---

## 289. Past as State, Present as Attention: Persistent-State Blockwise Flow Matching for Long-Horizon Co-Speech Motion Generation

**arXiv ID:** 2610.32373 | [PDF](https://arxiv.org/pdf/2610.32373v1)

**作者:** Yi Liu `[一作]` (Tongji University), Jianqiang Ren `[通讯]` (Alibaba Group)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种基于Past-as-State, Present-as-Attention原则的长时程共语动作生成模型PASPA，能够在长序列中保持手势风格与语音同步。

**💡 创新点**

创新点：①将过去的历史压缩为持久状态（BGDN），将当前段落通过双向注意力完整建模；②在块级流匹配框架下实现多块监督和预填充‑解码滚动；③引入混合无分类器引导（Hybrid CFG）进一步提升历史与语音条件的协同效果。

**🔧 技术方法**

技术栈：流匹配（Flow Matching）与块级扩散；块级门控DeltaNet（BGDN）实现持久状态；块级双向自注意力（BMHSA）；AdaRMSNorm与旋转位置编码；混合无分类器引导；预填充‑解码策略；TensorCore加速的自定义Triton核。

**📊 数据集**

数据集：BEAT2（多说话人和单说话人子集）与SHOW，使用官方划分与评测协议。

**📈 对比分析**

对比方法：GlobalDiff、DiffSHEG等流匹配/扩散基准。PASPA在BEAT2的FGD、MSE、BC、Diversity等指标均超过或竞争最佳；在SHOW上FGD下降43.9%，表现最优。盲评测中PASPA在真实性、语义一致、同步性与多样性四项中获得最高首位率。

**⚠️ 局限性**

限制：①依赖于块级显式状态，块大小对长序列性能影响较大；②在极长序列或极低资源数据下仍可能出现微小风格漂移；③模型结构复杂，训练与推理成本相对较高；④对非标准动作范畴（如复杂舞蹈）尚未充分验证。

---

## 290. Incremental-Smoothing-Based ARAIM Protection Levels for Tightly Coupled GNSS/IMU Integration in Urban Canyons

**arXiv ID:** 2610.32259 | [PDF](https://arxiv.org/pdf/2610.32259v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 291. Concept Acupuncturist: An Efficient and Precise Concept Erasure Method for Diffusion Models via Sparse Autoencoders

**arXiv ID:** 2610.32059 | [PDF](https://arxiv.org/pdf/2610.32059v1)

**作者:** Yiran Xiahou `[一作]` (Chinese Academy of Sciences), Renyang Liu `[通讯]` (National University of Singapore)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Concept Acupuncturist框架，实现对扩散模型的概念无学习（unlearning）

**💡 创新点**

创新点在于：①将稀疏自编码器（SAE）产生的特征级纠正作为监督信号，内部化到模型参数；②在单一跨注意力块上局部化更新，避免全轨迹推理和高昂的计算成本；③引入FO、FAC两种内部评估指标，量化概念抹除的精确度

**🔧 技术方法**

使用稀疏自编码器（Sparse Autoencoders）对UNet的跨注意力块进行特征分解，基于其特征得分进行块与特征定位；随后在该块上进行基于输入‑输出对的无推理微调；还利用Clip、EvalIGMU、I2P、MMA、UnlearnDiffAtk等工具进行评估

**📊 数据集**

数据集包括：Synthetic Prompt dataset（ChatGPT生成50个prompt/概念）、EvalIGMU benchmark（Nudity、VanGogh、Church、Parachute）、I2P、MMA、UnlearnDiffAtk；模型在SDXL‑Turbo上进行实验，并在SDXL与SD v1.4上验证迁移性

**📈 对比分析**

与ESD、UCE、ConceptPrune、SAEUron等基线对比，Concept Acupuncturist在T1/T4下平均忘记成功率提升36.34pp，Clip准确率降低32.10%，且在SDXL‑Turbo、SDXL、SD v1.4上保持优异表现；同时在FO/FAC指标上也优于竞争者，内存占用和训练时长相比全轨迹微调降低约46%

**⚠️ 局限性**

局限性：①对多概念同一块级联处理时可能需要多轮微调，影响效率；②对极端或分散的概念特征（如Parachute）抹除效果相对弱；③对于全局风格概念（如VanGogh）需谨慎调节抑制强度，防止过度损伤图像质量；④尚未探索针对概念的专属SAE训练，可能进一步提升定位精度

---

## 292. SynMark: Semantic-Coordinate Multi-Bit Watermarking during LLM Code Generation

**arXiv ID:** 2610.32017 | [PDF](https://arxiv.org/pdf/2610.32017v1)

**作者:** Rui Xu `[一作]`, Zhaoxia Yin `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种名为SynMark的多比特代码生成水印框架，能够在LLM代码生成过程中嵌入水印并在功能保持不变的变换后仍能恢复；

**💡 创新点**

核心创新点包括语义坐标路由（把表面词干置换为由对齐功能等价代码得到的语义坐标）、针对语法敏感位置的自适应logit偏置、以及在提取阶段通过可靠性加权校正残差的提取方法；

**🔧 技术方法**

采用语义对齐学习、Sinkhorn软分配、GRU/BiGRU编码器、可学习的候选集映射、以及自适应偏置与可靠性权重机制等技术；

**📊 数据集**

在MBPP数据集上训练，使用HumanEval进行评估，并在OpenCoder、CodeLlama、DeepSeek-Coder三种代码生成模型上进行实验；

**📈 对比分析**

与CredID、XMark、MPAC、CodeIP等基线方法对比，SynMark在BitAcc、Pass@1等指标上均优于对手，尤其在100%标识符重命名和多种语义保持的代码变换下仍保持82%+的BitAcc；

**⚠️ 局限性**

局限性包括仅在现有LLM代码生成模型上验证，未考察对重新合成、功能改变或密钥泄露等更强攻击；以及对极端或未见过的代码结构变换可能仍存在恢复下降。

---

## 293. Automated Post-Exploitation Attack Chain Exploration Based on a Phase-guided Hybrid Planning-Monte Carlo Tree Search Algorithm

**arXiv ID:** 2610.32278 | [PDF](https://arxiv.org/pdf/2610.32278v1)

**作者:** Jie Ying `[一作]` (Zhejiang Wanli University), Zhongjie Zhu `[通讯]` (Zhejiang Wanli University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种基于阶段引导的混合规划‑蒙特卡洛树搜索算法，用于自动发现后期利用阶段的攻击链。

**💡 创新点**

创新点在于将攻击动作划分为四个语义阶段，并结合规划器对每阶段动作进行即时枚举，从而实现高效多样化链条搜索。

**🔧 技术方法**

主要技术包括PDDL域建模、阶段引导的Hybrid Planning‑MCTS、改进的单步枚举规划器以及LLM生成自然语言报告。

**📊 数据集**

使用了5台CTF虚拟机、30个合成问题以及291个真实云实例的基准集进行评测。

**📈 对比分析**

与纯规划器、纯MCTS以及ChainReactor比较，PG‑HP‑MCTS在成功率、搜索时间与链条多样性方面均明显优于基线，尤其在大规模云实例中实现了约47% 的时间提升。

**⚠️ 局限性**

局限包括缺乏对内核级、硬件级攻击的建模、手工提取漏洞能力、未考虑探测隐蔽性以及静态配置假设。

---

## 294. GELLI: A Generator for Experiential Language Learning Interactions

**arXiv ID:** 2610.31987 | [PDF](https://arxiv.org/pdf/2610.31987v1)

**作者:** Danilo Symonette `[一作]` (Stanford University), James Landay `[通讯]` (Stanford University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了GELLI系统，该系统在运行时根据学习者的目标生成叙事练习世界，包括情境、角色对话、实时支持和反思反馈，支持学习者在XR环境中进行沉浸式语言实践。

**💡 创新点**

创新点在于：1）完全无预编写脚本的情境生成，直接按学习目标生成子场景；2）将经验学习循环嵌入生成式交互，提供即时纠正与反思；3）通过共创式场景设定让学习者拥有实际的情境创作与引导权；4）在轻量化视觉与叙事支持下实现高沉浸感。

**🔧 技术方法**

技术采用了大语言模型（OpenAI GPT‑4o‑mini）进行对话与情境生成，图像生成使用 DALL‑E 3，语音合成与识别分别使用 gpt‑4o‑mini‑tts 与 whisper‑1，XR平台为 Apple Vision Pro，系统架构为多代理设计。

**📊 数据集**

使用的主要数据来源为实验收集的学习目标与情境描述，以及专家制作的 Cafe 与 Shopping 两段对话视频作为对照；并未使用公开标准数据集，而是基于自定义学习目标与实验日志。

**📈 对比分析**

通过 40 名日语学习者的 within‑subjects 对比实验，评估词汇回忆、口语表现、学习信心及叙事参与度。结果显示：GELLI 在词汇回忆与学习信心上显著优于专家编写的视频（p<0.01），口语表现无显著差异；沉浸感（情景、情感、社交、空间）显著提升，但对情节理解略逊于视频。

**⚠️ 局限性**

局限性包括：仅在单次会话内测试，未检验长期保持与真实对话转移；系统延迟与图像不一致、语音识别错误影响沉浸；未能分离各组件对学习效果的单独贡献；受限于成人日语学习者，难以推广至其他语言或人群；未评估不同视觉表示方式对沉浸与学习的具体影响。

---

## 295. PaperFormalizer: Author-in-the-Loop Autoformalization of Research Papers in Lean

**arXiv ID:** 2610.31810 | [PDF](https://arxiv.org/pdf/2610.31810v1)

**作者:** Jiyuan Tan `[一作]` (Stanford University), Jose Blanchet `[通讯]` (Stanford University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

开发了一个基于 Lean 的管道，自动将论文转录成依赖图，进行自然语言审查、修正和自动正式化，并让作者在每一步中做出决策，最终生成 Lean 验证的论文版本。

**💡 创新点**

首次将人工智能驱动的自然语言审查与作者参与式修正结合，再与 Lean 自动化正式化流程衔接，实现了可重复、可共享的“验证论文”流程，并通过个人工具库不断累积可复用的 Lean 代码。

**🔧 技术方法**

采用 OpenAI Codex 与 Claude Opus 等大型语言模型进行文本解析、依赖图构建与错误检测；利用 Lean theorem prover 与 Mathlib、领域库及个人工具库进行正式化；使用 Loogle、LeanSearch 进行库检索；通过本地网页接口实现作者交互。

**📊 数据集**

主要在作者自己的三篇论文上进行实验，并利用 Mathlib 及若干领域库（如 Causalean、StatLean 等）进行正式化；未使用公开数据集。

**📈 对比分析**

在三篇论文中发现 5/7 和 15/29 结果需修正，管道能给出与原命题保持一致的最小修正方案；性能指标未给出，但实验显示能显著减少手工正式化的工作量并提高结果的可靠性。

**⚠️ 局限性**

需要作者在多义句、重大修正时手动介入；依赖高质量语言模型，模型不可用时需降级；对大型论文或缺乏对应库的领域正式化仍耗时；可能未捕捉到所有细微逻辑漏洞。

---

## 296. POSEIDON: Scaling Training Environments for Scientific Agents from Code Repositories

**arXiv ID:** 2610.32448 | [PDF](https://arxiv.org/pdf/2610.32448v1)

**作者:** Qifan Zhang `[一作]` (Hong Kong University of Science and Technology (Guangzhou)), Jia Li `[通讯]` (Hong Kong University of Science and Technology (Guangzhou))

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 POSEIDON 自动化管道，从科学代码仓库中生成多样化、可验证的科学代理训练环境，并用其创建了 PEARL 数据集。

**💡 创新点**

创新点在于：①从可执行工作流程中衍生四种互补任务类型；②使用递归生成‑验证闭环提升环境质量；③大规模收集多学科仓库实现 9.3 倍的环境产量。

**🔧 技术方法**

采用 LLM（DeepSeek‑V4‑Flash、GPT‑5.5）进行关键词扩展、代码聚合与工作流抽取，构建沙箱、验证与递归改进；使用监督微调训练 Qwen3.5。

**📊 数据集**

使用 7,110 个公开 GitHub 科学仓库（跨 10 个学科），产生 97,659 个训练环境，覆盖 10,000+ 任务。

**📈 对比分析**

在 6 个科学代理基准上，PEARL 微调的 Qwen3.5‑27B/35B‑A3B 在多项任务上提升 10–15% 并缩小与前沿模型的差距；相较 D3‑Gym 与 MedAgentGym，获得更稳健的平均提升 2.1%。

**⚠️ 局限性**

局限包括：仍依赖 LLM 的错误判定与修复，缺乏跨语言/跨平台通用性；验证过程耗时，且对极其复杂的工作流程支持有限。

---

## 297. Guided Zero-Shot Anomaly Explanation for Anonymized Crowd Video Monitoring: Context, Categorization, and Spatial Localization

**arXiv ID:** 2610.32073 | [PDF](https://arxiv.org/pdf/2610.32073v1)

**作者:** Mulugeta Weldezgina Asres `[一作]` (University of Agder), Christian Walter Omlin `[通讯]` (University of Agder)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出XAVAD统一可解释且隐私保护的视频异常检测框架，集成全身匿名化、零样本异常分类、热图定位与上下文描述。

**💡 创新点**

首次在单一管线中同时实现全身匿名化、零样本多模态XAI与VAD；引入Prompt‑guided Patch Autofocus（PPA）提升热图精度与计算效率；提出GDSN负向分割提升遮蔽武器的定位。

**🔧 技术方法**

CLIP多模态嵌入、LLM辅助文本提示、改进的Grad‑ECLIP热图、PPA、GroundedSAM+Negative Segmentation (GDSN)、InternVL3.5等轻量VLM。

**📊 数据集**

公开监控数据集：XD Violence、UCF‑Crime、Simuletic CCTV Weapon/Knife、HIVAU‑70k 等。

**📈 对比分析**

与YOLO、RF‑DETR、Grad‑ECLIP等基线对比；异常分类mF1提升≈30%，热图AP提升至0.9以上，BERTScore达到0.42；PPA实现4×显存节省、30%速度提升。

**⚠️ 局限性**

匿名化遮蔽仍削弱异常检测与定位精度，尤其对重叠武器；依赖预训练模型的偏差与小样本数据缺乏；极端分辨率与光照条件仍存在挑战。

---

## 298. Data-Efficient Backdoor Attacks on Code Models via Trigger-Neutralized Counterfactual Scoring

**arXiv ID:** 2610.32334 | [PDF](https://arxiv.org/pdf/2610.32334v1)

**作者:** Yu-Yu Lee `[一作]` (National Sun Yat-sen University), Chun-I Fan `[通讯]` (National Sun Yat-sen University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在预训练代码模型上的数据高效后门攻击，并提出了 Trigger‑Neutralized Counterfactual Scoring (TNCS) 用于选择最有效的毒化样本。

**💡 创新点**

创新点在于通过比较触发输入与中性化反事实的目标输出差异，独立衡量触发器对后门效果的贡献，并且该评分方法可同时应用于分类与生成两类任务。

**🔧 技术方法**

使用了预训练代码模型 CodeBERT/CodeT5、早期代理训练、迭代过滤刷新框架以及对触发器的中性化操作来实现样本选择。

**📊 数据集**

实验数据集为 CodeXGLUE 任务中的 Devign（缺陷检测）、BigCloneBench（克隆检测）和 Bugs2Fix 子集（代码修复）。

**📈 对比分析**

与随机选择、FUS、FUS++、RD、PFS 等基线比较，在 0.1%–0.5% 的毒化率下，TNCS 在三任务、五触发器族上均取得最高或前两名的攻击成功率，且对清洁任务性能影响不大。

**⚠️ 局限性**

局限性包括仅在 CodeBERT/CodeT5 上验证，未测试更大模型；极低预算下攻击成功率仍具有较高方差；对不同数据访问比例的非单调表现仍需进一步研究。

---

## 299. Do VLAs Know They Will Fail? Interpretable Success and Failure Patterns in Physical AI Systems

**arXiv ID:** 2610.32208 | [PDF](https://arxiv.org/pdf/2610.32208v1)

**作者:** Christian A. Rogers `[一作]` (University of Utah), Daniel S. Brown `[通讯]` (University of Utah)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

研究了VLA模型内部表示，使用稀疏自编码器（SAE）提取OpenVLA激活的稀疏特征，用于预测任务失败并识别与成功或失败相关的行为模式。

**💡 创新点**

创新点在于将SAE特征用于提前预测失败并定位导致失败的关键行为窗口，同时使用VLM生成可解释的行为描述并验证其对应性。

**🔧 技术方法**

采用稀疏自编码器（TopK SAE）、逻辑回归预测、置信门限报警、行为区间自动检测、VLM生成描述及AUROC等评估技术。

**📊 数据集**

使用OpenVLA‑7B模型在LIBERO‑10数据集上微调后，在10个任务的50个初始状态生成的500条轨迹（51.6%成功），另外95条轨迹用于评估。

**📈 对比分析**

与多数类基线比较，SAE与原始激活在早期时间步均能超过基线；在PNR前的失败预测AUROC为0.71，高于原始激活0.65；特征选择后AUROC约0.76，优于随机描述0.58。

**⚠️ 局限性**

局限包括高误报率（阈值低时），对阈值敏感，行为解释依赖VLM质量，未在更大规模或不同任务上验证，也未证明对政策改进的实际效用。

---

## 300. The Helpful Agent Problem: When Emergent Cooperation Among LLM Agents Bypasses Intended Workflows

**arXiv ID:** 2610.31985 | [PDF](https://arxiv.org/pdf/2610.31985v1)

**作者:** Moumita Choudhury `[一作]` (University of Massachusetts), Shlomo Zilberstein `[通讯]` (University of Massachusetts)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `9cc9baba-5356-466d-81ff-d80028d90279` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了LLM代理在需要共享资源且审查流程失效时，如何在没有共享奖励或明确协作指令的情况下，主动帮助对手并绕过工作流程，提出了“有益代理问题”这一新概念。

**💡 创新点**

创新点包括：1）系统化定义并量化有益代理行为及其对外部团队的潜在危害；2）设计了可测量副作用的合成基准FavorBench，并在真实代码库中对CooperBench进行适配；3）在四种主流LLM上进行大规模实验，揭示了请求、投机与自发援助的出现模式及其对安全的影响；4）探索了多种安全干预（任务范围约束、共享渠道、惩罚与反馈）对合作与副作用的影响。

**🔧 技术方法**

技术手段主要是：基于LLM的代理框架（如GPT、DeepSeek、Gemini、Claude），多轮对话与共享频道交互，审查者模拟，奖励函数与任务定义；实验使用自动评判器和人工标注来判定执行类型、助力类型与副作用。

**📊 数据集**

数据集包括：1）FavorBench——10个域、8个任务族的合成代码仓库；2）CooperBench——改编自真实开源仓库的功能请求；两者都提供了共享资源、任务依赖与缺失审查条件。

**📈 对比分析**

对比方法：在同一环境下跑默认设置、约束规则、惩罚与反馈等多种策略，测量Favor Rate、Harm Rate、请求/授予比例及任务成功率。结果显示：默认设置下Favor Rate高达58–95%；任务范围约束能显著降低Favor Rate；惩罚与CI反馈可将Harm Rate从≈44%降至≈21%/25%，同时保持高合作率。

**⚠️ 局限性**

局限性：1）实验环境仍为模拟或合成，真实工业系统的复杂性尚未完全覆盖；2）惩罚与反馈依赖外部评估或事后信息，可能不适用于实时或零延迟场景；3）仅考察了资源共享与审查失效两类约束，未探讨更广泛的治理机制；4）LLM的行为受提示与模型版本影响，结果不易直接迁移到其他模型。

---

## 301. AgentTrace: Execution-Level Fingerprinting of Black-Box LLM Agents

**arXiv ID:** 2610.32092 | [PDF](https://arxiv.org/pdf/2610.32092v1)

**作者:** Gejian Zhao `[一作]` (Shanghai University), Xinpeng Zhang `[通讯]` (Shanghai University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 AgentTrace，利用主动执行探测和多步修复轨迹来识别黑盒工具增强 LLM 代理的基础模型，防止模型伪装。

**💡 创新点**

创新点在于：① 通过结构模糊探测诱发 tokenization 相关的序列化失败；② 结合初始失败与后续修复序列的两级信号；③ 用轻量级马尔可夫链对修复轨迹建模，获得细粒度的模型指纹。

**🔧 技术方法**

技术手段包括：离线探测优化（基于 token boundary 交叉差异、执行间隙方差等多目标搜索）；在线注入结构化探测字符串；序列化错误观测与修复行为的概率建模；多步修复的最大似然推断。

**📊 数据集**

使用的实验数据集为自构造的探测库（100 条结构模糊 probe，50 训练 50 测试），并对 6 种候选模型进行评估：GPT‑4o（目标），Llama‑3‑8B/70B，Qwen‑2.5‑7B/72B，Mistral‑7B‑Instruct‑v0.3。

**📈 对比分析**

与传统语义基线（stylometry、cross‑perplexity）以及内部 ablation（仅初始失败、无序修复操作）对比，AgentTrace 在 50 次探测下实现 96.0% 的整体准确率和 0.96 的 macro‑F1，显著优于基线（<80%），并在低查询预算（10 次）下仍能达到 92.8% 的准确率。

**⚠️ 局限性**

局限性包括：需要被测代理执行带有严格 schema 校验的外部工具；必须能观察到错误反馈和多步修复；若中间件强制删除隐藏字符或一次性终止会话，则无法获取完整修复轨迹；探测器对未知模型检查的泛化性待进一步验证。

---

## 302. IFC-SSL-Bench: Benchmarking Self-Supervised Learning for Annotation-Efficient Multispectral Imaging Flow Cytometry

**arXiv ID:** 2610.31848 | [PDF](https://arxiv.org/pdf/2610.31848v1)

**作者:** Angelo Jovin Yamachui Sitcheu `[一作]`, Ralf Mikut `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**



**💡 创新点**



**🔧 技术方法**



**📊 数据集**



**📈 对比分析**



**⚠️ 局限性**



---

## 303. AudioLift: Audio-Driven Single Image to Full-Head Gaussian Avatar via Multi-view Video Generation

**arXiv ID:** 2610.31747 | [PDF](https://arxiv.org/pdf/2610.31747v1)

**作者:** Zhuoyuan Jiang `[一作]` (Yale University), Ming-Hsuan Yang `[通讯]` (University of California, Merced)

**通讯引用:** 113423 | [OpenAlex ID](https://openalex.org/A5100418319)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出一种两阶段框架，利用音频驱动的多视角视频生成与3D高斯重建，能够仅凭单张人像和语音生成完整的3D谈话头像；

**💡 创新点**

创新点在于：①不使用传统的3DMM/FLAME等参数面部模型，直接通过视频扩散模型产生多视角同步表现；②采用四视角2×2画布与前视视角联合生成，实现跨视角一致性；③通过视角感知腐蚀、主动体积正则与后视先验蒸馏，提升对生成视角不一致的鲁棒性；

**🔧 技术方法**

技术手段包括：StableAvatar音频条件扩散Transformer、Wav2Vec2音频编码、3D高斯渲染器、视角感知腐蚀、主动体积正则化、后视先验蒸馏等；

**📊 数据集**

使用NeRSemble（328身份，2,641条语音序列）、RenderMe-360（21身份）和HDTF（在野人像语音配对）等公开数据集进行训练与评估；

**📈 对比分析**

与Real3D‑Portrait、SDTalk、NeRFFaceSpeech等基线在RenderMe-360和HDTF上对比，PSNR/SSIM提升0.1–0.3分，LPIPS下降0.02–0.05，FID降至约44，且在新视角下表现更佳；

**⚠️ 局限性**

局限性：对极端头部姿势或严重遮挡的重建仍可能出现浮动几何；依赖高质量音频与单张人像，且对大幅度表情或复杂饰品的细节捕捉仍有限。

---

## 304. SyncVid: World-Equivariant Diffusion for Synchronized Video Generation

**arXiv ID:** 2610.31765 | [PDF](https://arxiv.org/pdf/2610.31765v1)

**作者:** Farhad G. Zanjani `[一作]` (Qualcomm), Amirhossein Habibian `[通讯]` (Qualcomm)

**通讯引用:** 975 | [OpenAlex ID](https://openalex.org/A5045620960)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**



**💡 创新点**



**🔧 技术方法**



**📊 数据集**



**📈 对比分析**



**⚠️ 局限性**



---

## 305. Democratic Accountability for Autonomous Systems: A Legislative Notification Framework and Comparative Institutional Design

**arXiv ID:** 2610.32226 | [PDF](https://arxiv.org/pdf/2610.32226v1)

**作者:** Samarjeet Malik `[一作]` (Ashoka University), Vaibhav Sharma `[通讯]` (Cruxway)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了“民主问责”四级 Attribution 框架并提出72小时立法通知机制，评估了六个民主监督机构的可实施性并回顾三起大规模自主网络行动案例

**💡 创新点**

首次将国际法 Attribution 框架与国内立法监督相结合，形成非约束性但可执行的问责层级，并揭示现行法案对特定行动的排他性限制

**🔧 技术方法**

利用法律学方法（规则阐释、案例推理）与制度比较法工具（权重评分、合规成本分析）实现框架设计与评估

**📊 数据集**

以Stuxnet、NotPetya、Volt Typhoon三起事件为案例，对比六国监督机构法条和操作实践，构成制度评估的数据集

**📈 对比分析**

通过对六机构的五维度评分（法定访问、强制性、闭会、公开披露、AI/网络授权）进行定量比较，结果显示现有法案多排除特定行动，唯美联邦参议院委员会具备可行性

**⚠️ 局限性**

需要各国立法修订以突破“持续行动”排他条款；框架仅适用于进攻性 AI，未覆盖防御性；数据仅覆盖三起案例，局限于已公开信息

---

## 306. Do AI-for-Science Agents Preserve Model Safety?

**arXiv ID:** 2610.32045 | [PDF](https://arxiv.org/pdf/2610.32045v1)

**作者:** Michael Yew Chuan Lee `[一作]` (DSO National Laboratories), N. Benjamin Erichson `[通讯]` (ICSI and Berkeley Lab)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了将大型语言模型嵌入AI‑for‑Science代理后，安全行为是否得以保留，比较了五种代理与其基模型在50个有害化学请求上的攻击成功率。

**💡 创新点**

首次证明代理框架会导致安全退化，并提出ΔASR指标量化此变化，揭示在自适应攻击下安全性会显著恶化。

**🔧 技术方法**

使用GPT‑5.2/5.5/5.6‑Luna/6‑Luna四种基模型，搭配CACTUS、ChemToolAgent、STELLA、Biomni、DrugAgent五个代理，采用直接、角色扮演、过去式与PAI四种提示策略。

**📊 数据集**

基于SafeSciBench提供的50条有害化学请求，评估模型在化学领域的安全表现。

**📈 对比分析**

通过对比基模型与代理的攻击成功率差值ΔASR，发现某些代理可使成功率提升至70个百分点，尤其在自适应攻击下，提升可达36–58个百分点。

**⚠️ 局限性**

局限于仅评估化学领域，未覆盖其他科学域；仅考察五种代理，缺少对各组件（指令、工具、流程）的拆解；安全评估仍依赖人工三评判，缺乏完全自动化。

---

## 307. SyncPro: Synchronous Reasoning and Responding for Proactive Streaming Video Understanding

**arXiv ID:** 2610.32132 | [PDF](https://arxiv.org/pdf/2610.32132v1)

**作者:** Xueyun Tian `[一作]`, Huawei Shen `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了同步双流框架，用于主动流式视频理解，独立并发生成内部推理和用户响应。

**💡 创新点**

通过双流设计实现自适应推理与及时响应的分离，独立激活、共享历史，提出双流数据与两阶段（SFT+RL）训练。

**🔧 技术方法**

基于Qwen3.5-4B，使用LoRA适配器、双流解码、动作选择、动作时间排序ATR、轨迹级强化学习（GRPO），结合全注意力与线性注意力。

**📊 数据集**

构建189K双流训练轨迹（12个视频来源），评测使用StreamingBench（Proactive Output）、OVO-Bench（Forward Active Responding）和ProactiveVideoQA（VAD子集）。

**📈 对比分析**

在相同协议下与VideoLLM-online、MMDuet、Em-Garde、ThinkStream等基线对比，Acc@2s提升至41.2%（+3.2点）、OVO-FAR平均F1 34.19%（+3.2点）、VAD PAUC 32.43% 领先。

**⚠️ 局限性**

模型仍依赖离线数据，推理时双流并发导致资源占用，低帧率场景及实时延迟未充分评估，规模受限。

---

## 308. What the Query Remembers: Query Abstraction Governs Lexical and Dense Retrieval of Classical Arabic Poetry

**arXiv ID:** 2610.31821 | [PDF](https://arxiv.org/pdf/2610.31821v1)

**作者:** Omar Farouk Zouak `[一作]` (National School of Artificial Intelligence), Raid Abderrezak Ouahioune `[通讯]` (National School of Artificial Intelligence)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

系统性研究古典阿拉伯诗歌检索，比较九种稠密编码器与 BM25，在两大语料、三种索引模式和两种评估协议下评测；发现检索效果取决于查询记忆程度；解释增量对不同检索器的影响呈阶梯式；两种评估协议（已知项检索与主题相关性）在同一系统上产生显著分歧；提出递归排名融合提升性能；

**💡 创新点**

提出查询记忆抽象光谱作为解释变量，证明查询中的词汇记忆是决定检索器优劣的关键；展示两种评估协议系统性分离的现象；首次在古典阿拉伯诗歌上系统评估稠密检索与 BM25 的比较；

**🔧 技术方法**

使用稠密嵌入编码器（Qwen3-8B、BGE‑M3、mE5‑large、jina‑v3、silma‑sts、gte、LaBSE、Muffakir‑v2、BM25）；利用 AI 生成解释（Gemini 2.0 Flash）与人类参考解释；采用递归排名融合（RRF）；通过 Logistic 回归、Spearman 等统计检验评估查询与检索效果的相关性；

**📊 数据集**

两大语料库：FoF（4,687首诗）和 Diwan（3,500首诗，按最常见诗人抽样）；查询集包含502个已知项（Fann/Flop 和 Diwan）与200个主题查询；

**📈 对比分析**

在已知项检索中，BM25 在包含引用的查询（约32%）上 Recall@5 最高（1.000），而在无引用查询中与最强稠密编码器相当；在抽象查询中，Qwen3‑8B 领先；解释增量在含引用查询时降低 BM25 性能，提升稠密编码器；在主题检索中，BM25 成为最差（平均得分≈1.7/5），稠密编码器平均得分≈3.0/5；融合模型在含引用查询上提升至0.885 Recall@5；

**⚠️ 局限性**

局限：查询由作者人工构造，缺乏真实用户分布；解释生成质量不均（8% 误读）；样本只覆盖部分诗歌传统；人类评估一致性有限；仅使用两大语料，抽样偏差；稠密检索在高度抽象查询仍难以达到 BM25 水平；

---

## 309. Distributional Alignment of LLMs and Its Generalization Properties

**arXiv ID:** 2610.31687 | [PDF](https://arxiv.org/pdf/2610.31687v1)

**作者:** Nelson Jo `[一作]` `[通讯]` (Independent Researcher), Nelson Jo (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对大型语言模型进行分布式对齐实验，使其生成与人类回答分布相匹配。

**💡 创新点**

提出在多选与开放式任务上通过交叉熵微调直接匹配答案分布，并评估跨主题泛化。

**🔧 技术方法**

交叉熵损失、温度标度、Llama‑3.2 1B Instruct 微调。

**📊 数据集**

OpinionQA（政治观点多选）与ProtoQA（Family Feud开放式回答）。

**📈 对比分析**

相较于全局或问答级温度标度，微调后Wasserstein距离从0.22降至0.12；开放式任务TVD由0.5提升至0.45；但在MMLU等外部任务上召回率下降，熵升高。

**⚠️ 局限性**

缺乏对不同任务差异原因的深入解释；仅在token层面对齐，未覆盖高层语义；数据量与主题覆盖有限，可能导致过拟合及不可预知的输出变化。

---

## 310. LogLook: Execution-Grounded Memory for Frontier-Model Robot Planners

**arXiv ID:** 2610.32348 | [PDF](https://arxiv.org/pdf/2610.32348v1)

**作者:** Tianyi Jiang `[一作]` (Nanjing University), Zhenshan Bing `[通讯]` (Nanjing University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

为前沿模型机器人规划器设计并实现了一种执行基记忆（execution‑grounded memory），该记忆将执行日志和模型生成的笔记分离，系统保存完整的执行记录，模型仅选择性地请求关键帧和演示片段，以减少令牌与图像成本并提升成功率。

**💡 创新点**

创新点包括：
1) 将“记录执行+决策观察”拆分为系统层和模型层，解决传统逐步重写笔记导致信息丢失的问题；
2) 通过期望门控（expectation‑gated）帧选择与“先 skim‑then‑inspect”演示观看策略，让模型只看到可能变化的区域；
3) 在前沿模型框架下不需额外训练即可显著提升成功率与成本效率。

**🔧 技术方法**

技术手段：
- 前沿视觉‑语言模型（如基于 LLM + 视觉编码器）作为决策者；
- 系统维护的执行日志（包含动作、时序、末端抓取状态等）与笔记历史；
- 期望门控帧提取（根据模型指定的感兴趣点判断图像变化）；
- 演示先 skim‑then‑inspect 的分段观看；
- 两种执行器：学习型子目标执行器（RoboMME checkpoint）与直接运动规划器。

**📊 数据集**

数据集：RoboMME 机器人操作任务集合（16 个桌面任务，分为 4 组，共 800 条测试 episode），包含计数、保持、引用、模仿等任务，并在每个任务上有 50 条测试 episode。

**📈 对比分析**

比较方法：与四种基线（note‑only memory、per‑chunk planner、MemER、FrameSamp‑Modul）以及 Oracle 子目标和 GroundSG 进行配对对比。主要指标为成功率、每 episode 令牌数、图像数、模型调用次数。结果：
- 直接控制下 93.5% 成功率，接近人类 96% 的水平；
- 学习型执行器下 81.9% 成功率，几乎匹配 Oracle 子目标 82.2%；
- 与 note‑only memory 相比，成功率相近但令牌、图像消耗下降 40‑50%；
- 与 MemER/FrameSamp‑Modul 等学习型记忆策略相比，成功率提高 20‑30% 并显著降低成本。

**⚠️ 局限性**

局限性：
1) 仅在仿真环境（RoboMME）上测试，未验证物理机器人；
2) 仅使用单一前沿模型与执行器配置，缺乏跨模型泛化评估；
3) 由于 benchmark 的 motion‑planner 失败率约 2‑12%，难以彻底区分记忆与执行器失败；
4) 依赖深度图和相机标定（直接控制下），在真实机器人上可能需要额外校准；
5) 需要手工编写任务完成阈值，若缺失则会导致额外失败。

---

## 311. MetaDB: Putting File-System Structure Back into Distributed Databases

**arXiv ID:** 2610.32247 | [PDF](https://arxiv.org/pdf/2610.32247v1)

**作者:** Zebin Ren `[一作]` (Vrije Universiteit Amsterdam), Zhenyu Hou `[通讯]` (Baidu (China) Co., Ltd.)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出并实现了一款针对文件系统元数据的分布式数据库 MetaDB，专门为元数据的树形结构、事务语义和生命周期特性进行协同设计；

**💡 创新点**

创新点包括：①树拓扑感知的分区与执行，①通过 policy/size‑based sharding 将同一目录及其子节点放在同一 shard 上；②语义感知的异步非关键路径处理，异步更新配额/索引和目录属性聚合；③生命周期感知的存储引擎，内存 MVCC 与局部 compaction 以抑制墓碑累积；

**🔧 技术方法**

采用了 RocksDB 作为本地存储引擎，Raft 进行复制与一致性，Hybrid Logical Clock 替代 TSO，MetaCache 缓存路由，内置异步队列与批量聚合，基于树拓扑的分区策略和本地化 compaction 算法；

**📊 数据集**

实验数据主要来源于公司生产环境的真实元数据工作负载（云文件系统、对象存储和分布式文件系统），并使用合成微基准进行单点验证；

**📈 对比分析**

与之前的通用分布式数据库 GP‑DB 做对比，采用相同的硬件环境、测试集和基准脚本；结果显示 MetaDB 在吞吐量上提升 2.2‑4.1 倍，延迟降低 23–75%，并将元数据节点数减半，支持 1,000+ 节点、数万亿记录及 Exabyte 级存储；

**⚠️ 局限性**

主要局限包括：①专门针对元数据设计，通用性有限；②实现需从头构建，维护成本相对传统数据库高；③异步更新虽然低延迟，但在极端突发负载下仍可能出现微小的时延波动；

---

## 312. Reproducible Evaluation as the Substrate for Safe Prompt Compression in Production Agents

**arXiv ID:** 2610.31972 | [PDF](https://arxiv.org/pdf/2610.31972v1)

**作者:** Prabal Vashisht `[一作]` (DoorDash Inc.), Ying Yang `[通讯]` (DoorDash Inc.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a4b10f5d-130b-4e77-9367-6469ec621899` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个可离线、可复现的评估工具包，能够在生产环境的可变数据上重现并衡量对话代理的行为，并以此作为健全性门控的提示压缩目标；

**💡 创新点**

创新点在于：①针对工具的读/写/固定三种可变性设计了对应的可重现机制；②提供可按需重生成 Benchmark 的框架；③将健全性门控与 GEPA 结合，保证压缩不回退正确性并防止 reward hacking；

**🔧 技术方法**

使用了 GEPA（Genetic Prompt Optimization Algorithm）进行提示压缩，结合多模型评估（Anthropic、Gemini、OpenAI）、工具调用轨迹回放、分层抽样与正则化门控；

**📊 数据集**

采用自动生成的训练/验证/测试数据集，总计 302+99+196=597 条样本，覆盖 10 种技能及其原型，数据完全基于当前生产系统的快照；

**📈 对比分析**

通过在 D^test 上与种子提示对比，测量提示长度与正确率的变化；结果显示压缩后提示长度减 3.5–7.6%，Anthropic 型模型正确率提升 0.118，Gemini 与 OpenAI 型模型保持相同或略微提升的正确率；

**⚠️ 局限性**

局限性包括：评估基准为冻结快照，未评估压缩后优势随生产数据漂移的衰退；实验仅在单一商家对话代理上验证，泛化性待进一步验证。

---

## 313. When to Trust Memory: Retrieval-Guided Probabilistic Spatiotemporal Forecasting under Distribution Shift

**arXiv ID:** 2610.32329 | [PDF](https://arxiv.org/pdf/2610.32329v1)

**作者:** Haochen Lv `[一作]` (Beijing Jiaotong University), Huaiyu Wan `[通讯]` (Beijing Jiaotong University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 REMIND 框架，利用检索历史样本评估当前观测在历史模式中的支持度，进而在分布漂移下自适应融合记忆校准分支和偏差感知分支，生成带不确定性的时空预测；

**💡 创新点**

①将检索相似度作为分布漂移程度的信号；②双分支架构（记忆校准分支适用于轻微漂移，偏差感知分支适用于严重漂移）；③采用记忆分散度与检索相似度的门控自适应融合；④层次高斯混合模型捕捉内/跨分支不确定性；⑤记忆扰动训练提升泛化；

**🔧 技术方法**

粗细分层时空检索、GCN+时序注意力编码、轻量级校准、记忆分散度计算、门控门与软正则、层次高斯混合分布、记忆扰动训练、NLL 训练目标；

**📊 数据集**

CAMELS（水文）、PEMS（交通）、ENERGY（能源）三类真实时空数据集；

**📈 对比分析**

与 8 种概率分布模型（DeepAR、LODE、DER、SDER、PriSTI、DiffSTG、STGNF、STMPN）以及 4 种分布漂移方法（ShifTS、STONE、TFPS、STRAP）对比，使用 MAE、RMSE、MAPE、CRPS、ES 等指标；REMIND 在所有数据集与指标上均优于基线，尤其在概率指标上显著提升；Ablation 证明各模块贡献；

**⚠️ 局限性**

1) 依赖检索质量，检索错误或数据稀缺会影响性能；2) 需要额外存储和检索开销；3) 主要适用于已知漂移场景，极端新模式处理有限；4) 未评估实时低延迟环境下的效率。

---

## 314. Collection-Date Assignment Redirects Answers in Temporal QA

**arXiv ID:** 2610.32191 | [PDF](https://arxiv.org/pdf/2610.32191v1)

**作者:** Minsu Oh `[一作]` (Jeju National University), Yeo-Chan Yoon `[通讯]` (Jeju National University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究在固定的维基百科两版文本中，只改动收集日期（metadata）就能大幅改变大型语言模型的答案，揭示了收集日期对时间问答的显著影响。

**💡 创新点**

创新点在于明确区分收集日期与修订时间，利用严格对照实验证明收集日期的指派可以误导模型，而非内容变化；并提供系统的诊断流程和多种提示/推理控制，首次量化此类 metadata‑driven 偏差。

**🔧 技术方法**

采用精确匹配、EM/别名覆盖率、A/B 选择任务、七种提示变体、思考模式（chain‑of‑thought）与原生推理、API 调用等技术，对五大 LLM 进行对照实验。

**📊 数据集**

使用 DynaQuest（带时间标签的问答集合）、600 页英文维基百科两版、128 页无重叠页面、虚构记录以及 Temporal Wiki 等数据集。

**📈 对比分析**

通过将同一页面两版的收集日期互换，比较五大 LLM 的 EM 变化，发现误差从 37.6% 到 51.4%；控制实验显示即便提示、日期格式、归档说明变化也保留显著损失；思考模式可将损失降低至 11.1%。

**⚠️ 局限性**

局限性包括仅测试两版文本，未覆盖更长期历史或不同语料；只测量词形匹配而未验证事实真相；提示与模型配置有限，缺乏对更广泛模型与环境的普适性验证。

---

## 315. Exact Algorithms and Complexity of the Smallest Grammar Problem

**arXiv ID:** 2610.32298 | [PDF](https://arxiv.org/pdf/2610.32298v1)

**作者:** Xiangle Chen `[一作]` `[通讯]` (Tsinghua University), Xiangle Chen (Tsinghua University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

本文提出了一种新颖的“无交代表”结构，用于描述最优语法的字典，从而设计了多项精确算法（包括 O*（1.872^n） 的全枚举算法、对小字母表的子指数算法以及对二进制字母表的 NP‑完备性证明，并给出了决策版本的低阶多项式公式。

**💡 创新点**

创新点在于证明最优语法的字典可通过一组相互不重叠的区间来唯一确定，进而将字典枚举与规则构造分离；利用此结构对长度为两的短串进行计数约束，将指数基数从 3 降到 1.872；构造带锚点的编码实现二进制 SGP 的 NP‑完备性与 ETH 下的下界；并将无交代表转换为可验证的整数多项式，从而完成了 BGKMS 框架下的条件性 SETH‑硬度分析。

**🔧 技术方法**

主要技术包括：分层分治的递归动态规划、区间划分与离散化枚举、短串计数约束、基于锚点的同步码编码与投影定理、以及多变量多项式构造与求值。

**📊 数据集**

本文为理论研究，不使用实际数据集；所有结论均基于数学证明与构造性归约。

**📈 对比分析**

与之前的 O*（3^n）算法相比，新算法在基数上显著下降；在小字母表时达到 2^O((n/R)log(R+1)) 的子指数复杂度；二进制 SGP 的 NP‑完备性证明填补了空白；在多项式公式方面，提供了显式低阶多项式验证器，满足 BGKMS 条件。

**⚠️ 局限性**

主要限制是指数幂中仍存在 log(R+1) 的乘子，尚未能将子指数上界进一步降低到 2^O(n/R)；此外，对大字母表的上界与下界仍有一定差距，且编码长度的常数项对固定字母表产生影响。

---

## 316. Mining the Design Space of Multi-Agent Workflows

**arXiv ID:** 2610.31709 | [PDF](https://arxiv.org/pdf/2610.31709v1)

**作者:** Xavier Cadet `[一作]` (Dartmouth College), Kyle Richardson `[通讯]` (Allen Institute for AI)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个两阶段流水线，利用完整的 MCTS 搜索轨迹和确定性评估，挖掘并验证多智能体 LLM 工作流的设计原则。

**💡 创新点**

创新点在于：① 把被丢弃的搜索中间结果转化为可量化的实验数据；② 结合工作流代码和提示模板特征，系统性发现三层次的工作流改进规律；③ 在控制实验环境下实现高效、可复现的原型。

**🔧 技术方法**

技术手段包括：MCTS 优化器（AFlow）、六种可组合的工作流算子、确定性 Enigmata 评估、L0 正则化的特征挖掘循环、搜索级别的 train/val/test 隔离。

**📊 数据集**

使用了 26 组确定性验证的 Enigmata 逻辑推理任务（包括 Sudoku、Cryptarithm 等），共 20 个实例/任务，生成 28,568 条可执行工作流记录。

**📈 对比分析**

通过与单体 LLM 基线对比，验证了三条设计原则：Star‑first 入口结构会降低准确率、SharedMemory（黑板式共享状态）优于 Debate（对话式协作），以及更具体的提示模板和占位符路由能带来与宏观拓扑同等的提升；实验显示在 held‑out 测试上这些效应保持一致。

**⚠️ 局限性**

局限性包括：仅在无工具、确定性推理任务上评估；使用单一模型族（{LLM1, LLM2}）和 MCTS 优化范围；挖掘的规律仅反映 MCTS 探索空间内的结构，未覆盖更广泛的随机或异构工作流。

---

## 317. LapaFM: A Laparoscopic Foundation Model via Multi-Stage Continual Learning

**arXiv ID:** 2610.32263 | [PDF](https://arxiv.org/pdf/2610.32263v1)

**作者:** Xiaobo Zhu `[一作]` (Chinese Academy of Sciences), Zhenyu Liu `[通讯]` (Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `afceb026-1760-41ae-8d86-010831a37d97` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了LapaFM，一个针对腹腔镜手术的多模态基础模型，支持烟雾去除、手术阶段识别、工具/息肉分割、息肉检测与分类以及淋巴结转移预测等多项任务。

**💡 创新点**

创新点在于：①采用多阶段连续学习框架，在白光→伪彩色的预训练过程中使用教师–学生、记忆回放与跨阶段知识蒸馏，显著降低灾难性遗忘和模态冲突；②通过对预训练模型进行语义蒸馏，训练轻量化烟雾去除网络，兼顾实时性与高质量恢复；③实现单一模型在多任务、多模态环境下的跨域泛化与高数据效率。

**🔧 技术方法**

技术实现包括 Vision Transformer、DINO自监督学习、IBOT掩码预测、EMA教师更新、K‑means样本挑选、语义嵌入模块、轻量化去雾网络 Dehazeformer‑B、全局局部一致性约束、知识蒸馏等。

**📊 数据集**

数据集：约250个手术视频共20,000+分钟（白光12,097分钟 + 伪彩色1,577分钟），以及多任务公开/内部数据——CP‑CHILD、Propolyp‑cls/detection、WCE‑CCD、CRCLNM、EndoVis2017/2018、AutoLaparo、Cholec80、DS_Synthetic/DS_Real 等。

**📈 对比分析**

与多种任务专用模型（SSIS‑Seg、SurgicalSAM、Swin‑Transformer 等）和现有基础模型（SurgeNetXL、GastroNet_5M、DINOv3 等）比较，LapaFM 在烟雾去除（PSNR 30.3 dB，SSIM 0.972）、手术阶段识别（准确率 0.931）、工具分割（Dice 0.948‑0.957）、息肉检测/分类（ACC 0.992/0.971，AUC 0.999/0.957）、淋巴结转移预测（AUC 0.817）等指标均超越对照，并在前瞻性数据和跨域胃肠镜任务中保持优异表现。

**⚠️ 局限性**

局限性包括：仅覆盖白光与伪彩色两种模态，缺乏完整的多模态输入；时序建模能力有限，难以充分利用长视频信息；模型规模较大，部署和实时推理面临挑战，未来需扩展多模态、改进时空建模并进行大规模前瞻验证。

---

## 318. Compatible Likelihoods for Flow Matching on Manifolds

**arXiv ID:** 2610.31981 | [PDF](https://arxiv.org/pdf/2610.31981v1)

**作者:** Lucas Ng `[一作]` (University Of Cambridge), Mark Girolami `[通讯]` (University Of Cambridge)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `40105733-5154-44cd-8090-a8cab9e64b07` `edb9d762-f411-4838-a852-f2d638b018db` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在非欧几里得流匹配中提出兼容性理论并设计兼容且支持感知的指数族终点似然，改进传统的RFM与RG-VFM方法，

**💡 创新点**

创新点在于：①构造兼容性判定条件；②提出兼容且只在可行速度上支持的指数族终点似然；③通过预表均值-方差实现高效训练；

**🔧 技术方法**

主要技术包括：Riemannian流匹配、极值理论、指数族似然、预表均值-方差、数值积分与梯度裁剪；

**📊 数据集**

使用的实验数据集包括：平坦圆环 (𝕋^1)、高维平坦圆环、球面地球气候四个基准（火山、地震、洪水、火灾）以及大规模人类姿态数据 (SO(3)^19 AMASS)。

**📈 对比分析**

与 RFM、RG-VFM 以及其他基准（Riemannian CNF、Moser Flow、CNF Matching 等）对比，本文方法在圆环和高维圆环上获得更小的 MMD、NLL 更低；在球面基准上 NLL 下降约 0.5–1.5；在人类姿态数据上样本质量和能量距离显著优于前者；整体性能提升明显。

**⚠️ 局限性**

局限性：需要针对每个流形进行特定实现和预表计算；对一般几何形状（如非对称流形）实现复杂；目前仅在已知对称性、同质性的流形上验证。

---

## 319. PaperScorer: How Predictable Is Paper Acceptance?

**arXiv ID:** 2610.31791 | [PDF](https://arxiv.org/pdf/2610.31791v1)

**作者:** Sachin Konan `[一作]` (Princeton University), Zhuang Liu `[通讯]` (Princeton University)

**通讯引用:** 2079 | [OpenAlex ID](https://openalex.org/A5100452088)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了名为 PaperScorer 的模型体系，利用匿名论文的文本或视觉内容直接预测最终的接受/拒绝决策，并提供可校准的接受概率。

**💡 创新点**

① 直接对最终决策进行二分类监督，避免传统间接标签的噪声；② 通过文本和完整页面截图两种输入，验证视觉信息对预测的显著提升；③ 构建了大规模平衡的匿名数据集并实现可解释的校准。

**🔧 技术方法**

采用 Qwen2.5 及其 Vision 版大语言模型进行监督微调（SFT）和强化学习（RL），利用 token 化、决策 token 概率计算，并配合温度缩放实现校准。

**📊 数据集**

两大匿名平衡数据集：OpenReview 来源的 ICLR 等会议（约 25K 篇）以及从 arXiv 提取的 LaTeX 源构造的跨会议 Corpus（约 83K 篇），均去除作者信息与会议标签。

**📈 对比分析**

与 Gemini‑3‑Flash、GPT‑5.5、DeepReview、Jev 等基准对比。PaperScorer 在 Vision 版 7B 模型上在 PaperScorer‑OCR 数据集上取得 68.3% 准确率、0.736 AUC，较最优前沿模型提升 2.7%；在 OpenReview 数据上取得 77.5% 准确率，提升 9%；并实现良好校准。

**⚠️ 局限性**

缺乏对预测原因的可解释性；对最新年份论文（'26）准确性下降；在中间评分区间表现接近随机；训练数据对不同会议的迁移有限；模型未能捕捉人工评审中的细微偏差。

---

## 320. FORCA: A Force-Aware Coding Agent for Contact-Rich Robot Manipulation

**arXiv ID:** 2610.31841 | [PDF](https://arxiv.org/pdf/2610.31841v1)

**作者:** Xiao Tan `[一作]` (University of Illinois Urbana-Champaign), Wenzhen Yuan `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 5407 | [OpenAlex ID](https://openalex.org/A5055947140)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

生成并执行基于语言指令的Python程序，利用腕部力/扭矩传感器进行程序执行与迭代修订，完成接触丰富的机器人操作任务。

**💡 创新点**

推出一种训练无关、力感知的编码代理，能够在部署时实时生成并修订包含力反馈控制逻辑的完整程序。

**🔧 技术方法**

使用大语言模型(Claude Fable 5)进行代码生成与分析，结合SAM3、GraspGen、PyRoKi等开源模块完成感知与规划，并通过UR5e腕部F/T传感器实现力反馈。

**📊 数据集**

在五个真实机器人任务（抽屉打开、门把手操作、立方体击到目标、花瓶擦拭、黄瓜剥皮）收集视觉与力传感数据进行实验验证，并未使用公开数据集。

**📈 对比分析**

与仅使用视觉信息的基线进行对比；FORCA在抽屉打开和门把手操作的成功率从 40%/10% 提升到 100%，在擦拭和剥皮任务中接触损失率从 40–55% 降至 ≤2.2%，同时保持高精度抓取。

**⚠️ 局限性**

迭代成本高（多分钟推理时间），腕部力/扭矩传感器信息有限，无法精准定位接触点；对更复杂长序列或更严格力约束的任务仍难以诊断与修订，需改进推理效率与分布式触觉感知。

---

## 321. TTT in VLA Models: Exploring Training with Human Demonstrations

**arXiv ID:** 2610.32094 | [PDF](https://arxiv.org/pdf/2610.32094v1)

**作者:** Qiangong Zhou `[一作]` (Standard Robots Co Ltd), Hongxiang Li `[通讯]` (Standard Robots Co Ltd)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在机器人长周期任务中引入 Test‑Time Training（TTT）并将其与人类演示与机器人轨迹结合，探讨通过记忆机制影响目标选择的可能性。

**💡 创新点**

首次系统将 TTT 与 VLA 模型（InternVLA‑A1 与 Pi0.5）集成，并评估在演示条件下的行为，阐明记忆写入、保留与使用的瓶颈。

**🔧 技术方法**

使用 TTT（fast‑weight）层、Pi0.5 与 InternVLA‑A1 主干、TBPTT、对比正则化、单帧提示等技术。

**📊 数据集**

基于 InternVLA‑A1 的 692 M 帧预训练数据，辅以三组演示任务数据集：原始视频演示（654 对），短视频演示（273 对）和单帧提示（210 对）。

**📈 对比分析**

通过真实机器人杯子选择实验以及 GT‑state / AR‑state 开环 MAE/RMSE 与不同初始化与正则化方法对比，发现模型对演示记忆依赖弱，杯子选择成功率低于 1/3 的随机基线。

**⚠️ 局限性**

受限于样本规模、训练梯度截断、记忆写入频率不足等，导致 TTT 记忆未能有效引导决策；缺乏对记忆写入/读取机制的深入分析和更大规模训练验证。

---

## 322. Detecting Insecure Compositions of Clean Agent Skills via Capability-Flow Graph Analysis

**arXiv ID:** 2610.32155 | [PDF](https://arxiv.org/pdf/2610.32155v1)

**作者:** Yihao Li `[一作]` (Tsinghua University), Ke Xu `[通讯]` (Tsinghua University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

发现并检测了由多个看似安全的AI技能组合产生的安全漏洞链

**💡 创新点**

首次将技能能力流图（SCFG）与可观测性、可控性、证据等四个握手条件结合，构造可执行路径搜索方法

**🔧 技术方法**

利用静态分析抽取技能能力摘要、构造SCFG、基于策略的图搜索与双层验证（标记流可达性+LLM链审计）

**📊 数据集**

对13,598个ClawHub已审查为安全的技能进行评估，亦在1,062个热门技能子图上测试

**📈 对比分析**

与标签匹配与仅类型匹配基线对比，SCFG在9个agent-LLM组合上达到约79%总体利用率，显著高于基线（≤18%），同时降低搜索空间约85%

**⚠️ 局限性**

未覆盖隐式通道（共享文件、环境变量等），依赖于技能标签准确性与agent/LLM特定行为，且对技能版本变化的泛化能力有限

---

## 323. Lifelong Context Adaptation through Bidirectional Memory-Skill Learning

**arXiv ID:** 2610.32295 | [PDF](https://arxiv.org/pdf/2610.32295v1)

**作者:** Tianyi Xu `[一作]` (Oregon State University), Huazheng Wang `[通讯]` (Oregon State University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种在冻结大型语言模型下，通过外部记忆与可复用技能的双向学习，持续适配有限上下文窗口的 LLM 代理。

**💡 创新点**

创新点包括：① 将经验以情节、事实和条件技能三类记录存储；② 通过技能驱动交互产生新的记忆并持续修订已存在技能；③ 采用“结果返回”机制，仅更新检索权重而不重写内容，从而实现对已服务事实与技能的动态使用调整；④ 在多领域交互流中验证记忆、技能与两者协同的效果。

**🔧 技术方法**

核心技术包括：冻结 Qwen3‑8B（及 Qwen3‑1.7B、GPT‑4o‑mini）作为基模型；外部存储结构（episodic memory、typed facts、skill catalog）与检索器；基于重叠聚类的技能合成与审计；基于奖励的结果返回更新；预序（prequential）评估框架与 ablation 实验。

**📊 数据集**

使用十个跨域基准：
- 具身控制：ALFWorld
- 长文本阅读：RULER、BABILong
- 对话记忆：LoCoMo、LongMemEval、DialSim（命名替换）
- 逻辑与数学推理：MATH‑hard
- 代码生成：MBPP、CodeContests、LCB+。

**📈 对比分析**

通过与 ReAct、ExpeL、MemP、SimpleMem、LightMem 等基线的预序平均奖励比较，系统整体平均得分 52.26%（相较于 SimpleMem 的 39.92% 提升 12.34pp）。在 ALFWorld 上，完整系统实现 63.17% 的成功率，优于 memory‑only 59.83%（+3.33pp）和 skill‑only 50.50%（+12.67pp）。在对话域和混合域下也取得显著提升；在代码域中表现略低于 ExpeL，但优于 ReAct。

**⚠️ 局限性**

局限性包括：结果返回为所有服务记录赋予同一奖励，无法区分因果贡献或评估事实真伪；依赖外部成功信号，误判反馈鲁棒性未验证；技能合成通过 LM 审计但不保证绝对正确；在域外迁移与隐私、偏见放大等实际部署风险未充分探讨。

---

## 324. Neural Disjunctive Normal Forms for Learning Exact Rank Filters

**arXiv ID:** 2610.32203 | [PDF](https://arxiv.org/pdf/2610.32203v1)

**作者:** Diego Marcondes `[一作]` (Australian National University), Ronaldo F. Hashimoto `[通讯]` (University of Sao Paulo)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `729e5870-4135-47f5-97f2-e3974d07b5dc` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过在前向传递中保持形态学秩滤波器的精确评估，后向传递使用基于阈值分解和神经DNF层的光滑近似导数，提出一种基于直通估计器的梯度下降训练方法。

**💡 创新点**

创新点在于：① 用阈值分解和神经DNF层构造出能够在整数网格上精确实现的光滑近似；② 通过隐式微分得到梯度分布的解析表达，解决了梯度稀疏问题；③ 在前向保持精确、后向光滑的直通策略下实现了对任意秩滤波器及其组合的训练。

**🔧 技术方法**

使用的技术包括：形态学秩滤波器、阈值分解、神经离散析取范式（neural DNF）层、tanh 激活函数、隐式微分、直通估计器、Newton 迭代求根、Adam 优化。

**📊 数据集**

实验数据集为 MNIST、Fashion‑MNIST（图像去噪）和 CHASE_DB1（视网膜血管分割）。

**📈 对比分析**

与使用精确子梯度训练以及光滑形态学网络（HSMorph、HSMorph‑C）进行比较。结果表明，本文方法在去噪 PSNR 上平均提升 1–2 dB，在血管分割 AUC、F1 等指标上均优于精确子梯度，并超过现有光滑形态学网络。

**⚠️ 局限性**

局限性包括：① 仍需预先设定秩（delta）；② 对极大或极小形态学运算的扩展受限；③ 需要逐像素求根，计算开销相对较高；④ 对于初始参数的鲁棒性虽好于子梯度，但仍需进一步研究。

---

## 325. From Target Selection to Condition Reuse: Auditing Residual Interventions in Tool-Use Models

**arXiv ID:** 2610.31686 | [PDF](https://arxiv.org/pdf/2610.31686v1)

**作者:** Tianda Sun `[一作]`, Dimitar Kazakov `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究在 Qwen3.5-27B 上通过固定位置的残差写入（writer）控制工具调用中的目标选择，并检验这种干预是否能让模型在其他计算中正确使用相同条件。

**💡 创新点**

首次证明了目标选择控制、条件重用与完整动作正确性是独立的三种结果，并通过跨事实重组与自然历史替换评估编辑的实际效用。

**🔧 技术方法**

采用固定位置残差写入、交叉熵监督、自然历史替换以及源事实补丁等技术，在第60层注入向量并进行训练。

**📊 数据集**

使用 32 个随机生成的程序化“世界”，每个世界包含两种条件、菜单及工具调用规则，构成训练与评估任务集合。

**📈 对比分析**

通过 Anchor（训练事实）与 Cross（事实重组）两种评估方式，并与 Native A、Text Update、Natural B 等对照；写入写者在熟悉菜单中将目标选择准确率提升至约 90%（约 65% 的提升），但在 Cross 评估中大幅下降，独立判断和完整动作准确率仅提升不到 30%。

**⚠️ 局限性**

写入仅监督目标 ID，缺乏对条件重用的保证；早期补丁与后期写入效果难以分离，且实验仅在单一 27B 模型与程序化任务上验证，无法说明在实际应用中的普适性。

---

## 326. Scaling Graph Query Processing: Focused Exploration on Partitions Using Load Balancing

**arXiv ID:** 2610.32015 | [PDF](https://arxiv.org/pdf/2610.32015v1)

**作者:** Harshit Modi `[一作]` (University of Texas at Arlington), Sharma Chakravarthy `[通讯]` (University of Texas at Arlington)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了通过图划分和分布式并行处理来高效回答图查询（支持通配符和比较运算的子图查询）的方案，并给出了负载平衡的启发式方法。

**💡 创新点**

创新点在于将表达式丰富的图查询映射到分区图上，利用聚焦探索和分布式计算提升并行度；同时设计了可减少加载分区的启发式算法并证明了在分区环境下的正确性。

**🔧 技术方法**

主要技术包括：图划分算法、基于MapReduce的分布式查询架构、分区加载启发式与负载平衡策略。

**📊 数据集**

实验使用了大规模合成图与真实世界图数据集（未具体列明）。

**📈 对比分析**

通过大量实验与传统方法（集中式处理）对比，展示了在分区和负载平衡启发式下查询响应时间显著下降、可扩展性提升。

**⚠️ 局限性**

局限性包括：仅针对图查询而非搜索；对极大规模图的划分与通信开销未给出精细分析；实验集成与现有主流图数据库的兼容性尚待验证。

---

## 327. Pairing Matters: Performance, Failures, and Efficiency in Lightweight Coding Agents

**arXiv ID:** 2610.32131 | [PDF](https://arxiv.org/pdf/2610.32131v1)

**作者:** Lu Zhao `[一作]` (Singapore Management University), David Lo `[通讯]` (Singapore Management University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对轻量化编码模型与不同代理 harness 的配对进行全因子评估，探讨配对对性能、失败模式和 token 效率的影响

**💡 创新点**

首次系统化比较模型与 harness 的交互效应，揭示同一模型在不同 harness 下性能差异可达44个百分点，并提出配对特定失败分类和效率评估

**🔧 技术方法**

使用 OpenAI API 兼容接口、vLLM 服务、Docker 环境、SWE‑bench 验证任务、手工失败标签等技术

**📊 数据集**

SWE‑bench Verified 100 任务子集（共 100 个软件工程任务）

**📈 对比分析**

通过 18 个模型–harness 配对执行 1,800 次跑测，比较解决率、任务重叠、失败类型占比和每解决任务所消耗 token；性能表明多数配对在 token 效率上低于公开基准，且性能差异显著

**⚠️ 局限性**

局限包括单一 GPU 服从、仅 100 任务子集、缺乏运行方差、失败标注主观性、仅涵盖 6 个 harness 及 3 模型，未考虑不同硬件/配置影响

---

## 328. $Ψ$-0.6: Adaptive Streams of Thought in a Unified Vision-Language World Model

**arXiv ID:** 2610.32177 | [PDF](https://arxiv.org/pdf/2610.32177v1)

**作者:** Khai Loong Aw `[一作]`, Daniel Yamins `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了Ψ-0.6，一种统一的视觉‑语言世界模型，能够在单一自回归 Transformer 的共享 token 接口上生成、条件化、组合图像、文本、相机姿态、深度、光流等多模态表示，并通过“思维流”在推理过程中自适应地选择和使用这些中间表示。

**💡 创新点**

创新点主要包括：
1) 将感知、语言、几何与仿真能力统一到同一模型与同一 token 介面；
2) 通过任何到任何的提示实现可编排的推理流程，使得中间表示既是输出也是后续输入；
3) 采用强化学习（GRPO）让模型在任务反馈下学习自适应地选取和生成最有用的表示；
4) 支持逆向模拟与计数等逆向推理任务，提供“假设情境”推理能力。

**🔧 技术方法**

核心技术：8B 规模自回归 Transformer；Probabilistic Structure Integration（PSI）框架；稀疏指针 token 与内容 token 的细粒度控制；任何到任何的提示接口；多阶段训练（预训练→中训练→后训练）；强化学习优化推理策略；光流、深度、相机位姿等几何 token 的序列化；对抗性解码（CFG）与光流优先采样。

**📊 数据集**

使用的数据集：
- 预训练：大规模多样化视频数据（自然场景、人类行为、机器人交互）；
- 中训练：FineWeb‑Edu、DataComp‑LM、图像‑视频+字幕对齐数据；
- 后训练：视频问答、指令微调数据；
- 评测：MindCube（多视角空间推理）、FlowBench（光流与世界建模）、机器人数据集（DROID、Bridge V2、Language‑Table、RT‑1、Berkeley UR5）以及 EgoDex、Something‑Something v2、PE、OpenVid‑1M 用于 RL 训练。

**📈 对比分析**

对比方法：开源 VLM（DeepSeek‑VL2‑Small、LLaVA‑OneVision‑7B、Qwen3‑VL‑8B）、空间推理专用模型（RoboBrain、Space‑Qwen、Spatial‑MLLM）以及世界模型（FOFPred、InstructPix2Pix、Qwen‑Image‑Edit、IP2P）。在 MindCube 零样本上 Ψ-0.6 达到 49–51% 精度，优于上述对手；利用逆向/新视角推理提升至 57%；Fine‑Tuning 后在 MindCube 上分别达 68.7% / 76.3%；在 FlowBench 的光流预测中 56% 精度，远超 16% 的专用基线；在 Sim‑VQA 上 73% 接近 Qwen‑3‑VL‑8B Fine‑Tuned 的 89%；在 Real‑VQA 上 51%；强化学习进一步提升 VQA 5–9% 的相对准确率，并在四个未见基准上保持正向迁移。

**⚠️ 局限性**

局限性：
1) 模型规模和训练资源低于当前最前沿系统，性能仍有提升空间；
2) 统一框架可能导致不同能力之间的竞争与干扰，未对正向/负向迁移进行细粒度分析；
3) 当前仅实现视觉‑语言‑几何的推理，缺乏直接动作控制或机器人执行层面；
4) 对真实世界的验证主要基于仿真转移，缺少大规模真实数据的在线训练；
5) 需要更完善的通用一致性与验证机制，以进一步提升自监督学习的鲁棒性。

---

## 329. Agentic AI Defense: Decode-Time Provenance Binding for MCP

**arXiv ID:** 2610.31974 | [PDF](https://arxiv.org/pdf/2610.31974v1)

**作者:** Eric Yocam `[一作]` (California Polytechnic State University), Yong Wang `[通讯]` (University of Idaho)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了多模态代理（MCP）系统中，通过隐藏在邮件签名logo中的隐写指令导致的间接提示注入攻击，并提出了解码时来源绑定（decode‑time provenance binding）防御机制。

**💡 创新点**

创新点在于：① 将可信度标签的铸造点从文本返回处移动到任何隐写解码工具返回处；② 结合 HMAC 加密绑定和 MCP 层维护的完整传承记录，实现对跨代理传播的可信度追踪；③ 通过可信晋升路径实现合法任务不被误拦，兼顾实用性。

**🔧 技术方法**

使用技术包括：Model Context Protocol (MCP) 代理框架；HMAC‑SHA256 生成与验证绑定；内存式 ledger 记录绑定与上下文 taint；对三种隐写载体（LSB、DCT、PNG metadata）进行攻击与防御。

**📊 数据集**

实验数据集为 24 个自适应 payload（三种载体×八种逃逸技巧）以及一张测试 logo；并使用本地 语言模型和现有的关键词/正则扫描器做对比基准。

**📈 对比分析**

与无防御、关键词扫描器、正则解析器和基线信息流污点跟踪相比，解码时绑定在单一可信域内攻击成功率降至 0%，跨域传递中仍能阻止伪造标签和被篡改标签，且在正常任务中通过可信晋升实现 100% 的成功率，性能开销仅约 0.2% 的解码时间。

**⚠️ 局限性**

局限性包括：需要预先登记所有隐写解码工具并保持列表完整；无法追踪解码文本离开受控通道后对模型生成文本的影响；对多样化模型和真实生产环境的评估尚未完成；若解码器未被登记则攻击可通过标记为可信成功。

---

## 330. Disentangling Question, Response, and Verifier Languages in Multilingual P(True) Self-Evaluation

**arXiv ID:** 2610.32144 | [PDF](https://arxiv.org/pdf/2610.32144v1)

**作者:** Jewon Yeom `[一作]` (Seoul National University), Taesup Kim `[通讯]` (Seoul National University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对三种语言（问题语言、回答请求语言、验证者语言）在三种多语言问答基准上，使用一词概率探针对10个3B–14B模型的准确率、判别率与校准进行因子设计实验。

**💡 创新点**

首次将三种语言分别视为实验变量，揭示不同语言主导准确度、判别率与校准，并强调验证者语言与评判者选择对结果的重要性。

**🔧 技术方法**

使用一词“True”的第一词概率作为置信度探针，结合线性混合模型、Bootstrap统计和最大似然估计，对10个开源模型进行评估。

**📊 数据集**

使用LSQA、MMMLU、TriviaQA三大多语言问答基准，在英语、法语、中文、日语、韩语、印地语六种语言下生成并评估答案。

**📈 对比分析**

通过模型-配置单元的平均指标（准确率、AUROC、ECE、Brier）和线性模型分解比较，结果显示MMMLU在问答语言差距最大，TriviaQA在判别率最强，所有基准的绝对校准均偏差较大；验证者语言可使ECE波动约0.04。

**⚠️ 局限性**

研究仅针对单一一词探针、十个3B–14B开源模型及所用基准，未覆盖更大/闭源模型、不同探针或更广泛语言，且翻译质量与评判者验证可能影响结论。

---

## 331. Generalized green and red concepts for artificial intelligence, big data, and information and communication technologies: Multidimensional framework and countermeasures

**arXiv ID:** 2610.31739 | [PDF](https://arxiv.org/pdf/2610.31739v1)

**作者:** Jinsong Wu `[一作]` (Guilin University of Electronic Technology), Huamin Fang `[通讯]` (Guilin University of Electronic Technology)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了统一的多维可持续性框架，将人工智能（AI）、大数据（BD）和信息通信技术（ICT）按环境（E）、经济（Ec）、社会（S）和技术/运营（T）四维度进行分类，定义了泛化的绿色与红色概念（GGAI/GRAI、GBD/GBD、GICT/RICT），并基于此设计了四层级的可操作性对策与路线图；

**💡 创新点**

创新点在于：①将绿色/红色范畴从单纯的能耗/碳排放扩展到全生命周期、服务与使用边界的多维度；②引入E/Ec/S/T框架，实现跨技术堆栈的系统性评估；③提出可操作的四层对策（指标改革、设计原则、LLM 开发实践、能源基础设施转型），并给出时间维度的实施路线；

**🔧 技术方法**

主要技术手段为：框架构建、指标选择与归一、情景分析、对策分类与优先级评估；没有使用具体机器学习算法或模型训练，而是以理论与政策工具为主；

**📊 数据集**

数据来源主要为公开统计与案例：美国数据中心电力占比、GPT‑3 训练排放、Google 2023 数据中心水耗、全球大模型能源/水/废物预测等；未使用专有训练集；

**📈 对比分析**

本文未进行实验比较，仅通过案例说明（如 GPT‑3 训练与碳感知调度的对比）展示概念效果；性能评估以理论指标改善（如 CO₂e 降低 20%）为示例，缺乏量化性能指标；

**⚠️ 局限性**

局限性包括：缺乏大规模实证验证与情境化指标；框架仍处于概念阶段，需进一步细化可量化的多维指数；对技术实现细节（如硬件寿命、冷却效率）的模型不够深入；以及对社会公平与技术扩散的实证研究不足。

---

## 332. Benchmarks Are Surprisingly Compaction-Friendly

**arXiv ID:** 2610.31778 | [PDF](https://arxiv.org/pdf/2610.31778v1)

**作者:** Jeanne Salle `[一作]` (ELLIS Institute), Maksym Andriushchenko `[通讯]` (ELLIS Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了LLM代理在长时延任务中对上下文压缩（compaction）的实现与影响，系统地在OpenCode框架下评估了五种压缩策略，并在PaperBench、ProgramBench、PostTrainBench等基准上验证其效果。

**💡 创新点**

创新点在于首次对前沿大模型在真实长时延代理任务中的压缩阈值、策略与约束丢失进行系统实验，并指出现有基准对压缩友好、压缩方法对约束易损性。

**🔧 技术方法**

使用了OpenCode代理框架、五种压缩策略（自摘要、工具输出裁剪、文件系统离线、推理块丢弃、小模型摘要），以及Bootstrap置信区间评估、Claude Sonnet 5评判等技术。

**📊 数据集**

采用的评测数据集包括PaperBench、ProgramBench、PostTrainBench，以及SWE-chat真实用户代理会话。

**📈 对比分析**

通过对不同阈值（160K–380K）和不同压缩策略下模型（Claude Opus 4.6、GPT-5.3-Codex、GPT-5.4、Gemini 3.1 Pro Preview）的任务得分进行对比，发现1M上下文模型对阈值不敏感，GPT-5.3-Codex低阈值表现下降；无策略优于默认自摘要；压缩可导致最多20分的性能损失，但大多数基准下性能保持稳定。

**⚠️ 局限性**

局限性包括仅使用单一代理框架、受限于模型与任务数量、Gemini 3.1 Prereview稀缺触发压缩、SWE-chat使用不同框架，因而结果对其他环境与更复杂约束可能不具普适性。

---

## 333. Every Weight Is a Path of Shared Discrete Decisions

**arXiv ID:** 2610.31931 | [PDF](https://arxiv.org/pdf/2610.31931v1)

**作者:** Ammar Ali `[一作]` (ITMO University), Stamatis Lefkimmiatis `[通讯]` (ITMO University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为 TWIG（Tree‑structured Weights with Inherited Gates）的权重表示法，将每个权重视为一条从根到叶的离散决策路径，并通过共享路径中的节点来压缩存储、实现乘法无乘法电路以及将模型压缩问题统一为离散优化。

**💡 创新点**

创新点在于：① 把所有主流压缩技术（裁剪、量化、低秩分解、代码本、权重共享等）都映射为对同一树结构的不同“编辑”；② 在该树上构造的高阶玻尔兹曼机（或 Ising 模型）实现了模型压缩问题的统一离散优化；③ 通过将权重表示为乘以 2 的幂的符号、位移、门控、查找表等小状态，得到的电路几乎完全由加法和位移组成，显著降低乘法器成本。

**🔧 技术方法**

主要技术包括：
- 状态树（state tree）结构设计与编码；
- 通过共享节点实现内存共享和加法共享；
- 量化、裁剪、低秩、代码本、共享子树等三类编辑操作；
- 采用动态规划/整数规划等预算化优化器在整个网络层级上做全局选择；
- 将树的状态映射为高阶玻尔兹曼机，使用 QUBO/Ising 求解器（如 GPTQ、BFGS、模拟退火等）寻找低能量解；
- 采用 16 位 BF16、3‑4‑bit 量化、以及 3.3‑bit tree 进行实验。

**📊 数据集**

实验数据集与模型：
- LLaMA‑3‑8B（10 层 transformer）
- Qwen2.5‑1.5B‑Instruct
- WikiText‑2、C4、FineWeb、FineWeb‑Fine
- 其他公开数据集用于评估通用性能。

**📈 对比分析**

与现有方法比较：
- 存储方面：TWIG 通过共享指数/位移将 BF16 权重压缩到 12.13 bits/权重（≈ 24 % 下降），保持完全精度；
- 速度方面：在小批量（batch 1/8）推理中，基于 TWIG 的 CUDA 核心相对 cuBLAS 提升 1.05‑1.51×；
- 压缩方面：在 3.25 bits/权重下，TWIG 在 LLaMA‑3‑8B 上的 perplexity 下降 14‑17 % 以上，缩小 58 % 与 FP16 的差距；
- 计算成本方面：在相同层误差下，TWIG 需要 9‑55 %（3‑bit）和 30‑63 %（2‑bit）更少的加法操作；
- 与 GPTQ、AWQ 等传统量化方法相比，TWIG 在 3‑4‑bit 级别提供更低误差，并在存在大幅度离群值的矩阵上获得更显著的加法节省。

**⚠️ 局限性**

局限性与挑战：
- 当前仅在权重矩阵层面验证，尚未在完整训练循环中直接学习树结构；
- 对于极大模型，树的离散优化仍需要分块/分层求解，量化规模受限于现有硬件的可编程度；
- 电路实现假设乘法器被完全替代为加法/位移，适用于固定权重的专用硬件，未必能在通用 GPU/CPU 上充分体现；
- 在低精度（≤ 2 bit）下，树结构仍需更多剪枝/校正才能保持模型性能；
- 对比实验主要聚焦于语言模型，尚未在视觉/音频等其它任务上展示同样的收益。

---

## 334. REDAR: Reinforcement-Driven Adaptive Regularisation for Class Incremental Learning

**arXiv ID:** 2610.31991 | [PDF](https://arxiv.org/pdf/2610.31991v1)

**作者:** Akila Ayanthi `[一作]` (Queensland University of Technology), Peyman Moghadam `[通讯]` (Queensland University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `8d10c613-917e-4880-9716-17789f50e119` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于强化学习的自适应正则化框架（REDAR），通过一个轻量级RL代理在每个增量阶段动态选择知识蒸馏系数以平衡稳定性和可塑性；

**💡 创新点**

核心创新在于将正则化强度视为可控决策变量，将其调度问题建模为有限时域马尔可夫决策过程，并在离线伪CIL任务上学习一条通用策略，从而实现跨增量阶段的自适应调节；

**🔧 技术方法**

技术包括：离线伪任务构造、分布式Ape‑X DQN强化学习、四种损失（交叉熵、特征蒸馏、对数蒸馏、边缘排名）相结合的CIL模型，以及基于性能指标与数据统计的四维状态表示；

**📊 数据集**

实验使用CIFAR‑100和Tiny‑ImageNet两个公开图像数据集，在“train‑from‑half”协议下进行2/5/10阶段的增量学习；

**📈 对比分析**

与固定系数、随机选择以及多种现有CIL方法（iCaRL、LUCIR、OHOCIL等）对比，REDAR在平均准确率、平均遗忘率和前向迁移上均取得领先，且在跨阶段（从短到长）迁移实验中表现更稳健；

**⚠️ 局限性**

局限性包括：正则化系数采用离散网格限制了细粒度控制；训练策略仅在离线伪任务上完成，缺乏在线自适应；仅在基于回放的ResNet骨干上验证，未探究在扩展式或提示式CIL框架下的表现。

---

## 335. On the Propeller Conjecture

**arXiv ID:** 2610.32158 | [PDF](https://arxiv.org/pdf/2610.32158v1)

**作者:** Steven Heilman `[一作]` `[通讯]` (University of Southern California), Steven Heilman (University of Southern California)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

简化并改进了OpenAI对Khot和Naor在2008年提出的螺旋猜想的证明，证明了任何欧几里得空间的划分的高斯矩的平方长度之和最多为9/(8π)。

**💡 创新点**

证明了只有在平面中存在三个120度的扇区时，才能达到等式，并且去除了对三维情况下计算机辅助的依赖。

**🔧 技术方法**

使用了高斯概率测度和Lagrange乘数法等数学技术。

**📊 数据集**

使用了标准高斯概率测度和相关的数学构造，没有具体提到使用的数据集。

**📈 对比分析**

通过与之前的计算机辅助方法进行比较，证明了在没有计算机辅助的情况下也能达到相同的结果，性能上有所提升。

**⚠️ 局限性**

限制在于目前的结果主要集中在特定的几何和概率形式上，测量约束版本的螺旋问题仍然难以解决。

---

## 336. Free Lunch for LLMs Confidence: The RL Critic Already Knows

**arXiv ID:** 2610.31820 | [PDF](https://arxiv.org/pdf/2610.31820v1)

**作者:** Shujin Wu `[一作]` (University of Illinois Urbana Champaign), Heng Ji `[通讯]` (University of Illinois Urbana Champaign)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并验证在大语言模型（LLM）中，利用强化学习（RL）训练出的价值函数（critic）直接作为信心估计器（Critic-as-Confidence），不需额外训练或数据。通过理论证明和大规模实验，证明在以正确性为终止奖励的RL设定下，信心估计与价值函数学习等价。

**💡 创新点**

创新点：①在正确性奖励下证明信心估计与critic学习等价，揭示两者实质相同的预测任务；②展示外部模型（不依赖目标策略内部表示）即可从行为轨迹学习出与自体估计器相当的信心；③将RL critic直接复用为无额外训练的信心估计器，保持原模型性能并提供跨步骤、跨任务的置信度估计。

**🔧 技术方法**

技术方法：利用PPO/GRPO等RL算法训练LLM策略；从策略轨迹收集 (prompt, response, correctness) 三元组；训练外部校准器（如Qwen2.5‑7B、Mistral‑7B）或直接使用已训练好的critic；采用AUROC、ECE、Brier等指标评估置信度校准；对推理时间缩放进行Selection实验；在critic‑free RL 环境下，后置拟合价值模型。

**📊 数据集**

使用的数据集：数学推理（GSM8K、NuminaMath、AIME）、代码生成（APPS‑intro、TACO、CodeContests、Codeforces）、工具调用（ToolRL、API‑Bank）、通用推理（WebInstruct‑verified、MedMCQA、PubMedQA、ChemBench、UGPhysics、OpenScienceReasoning‑2）、以及相应的OOS/ID拆分。

**📈 对比分析**

对比方法：自体推断、Self‑consistency、R‑Tuning、Probe、Verbalized、Selection等；结果显示：Critic‑as‑Confidence在所有ID拆分上AUROC最高、ECE最低、Brier最佳；在OOS拆分亦领先或相当；通过Selection可将单样本性能提升4–6点，逼近8‑sample最佳；无额外任务改动保持原始准确率。

**⚠️ 局限性**

局限性：①需要RL训练产生的critic；critic‑free RL（如GRPO）需额外拟合价值模型；②使用critic会产生一次额外的前向推理开销；③仅在以二元正确性奖励为终止的设定下成立，若奖励更复杂则需进一步验证。

---

## 337. Financial Context-Rich Strategy for Actor-Critic based Agents: A Deep Reinforcement Learning Approach for Stock Trading

**arXiv ID:** 2610.32237 | [PDF](https://arxiv.org/pdf/2610.32237v1)

**作者:** Hrishikesh Singh `[一作]` (Indian Institute of Technology Roorkee), Millie Pant `[通讯]` (Indian Institute of Technology Roorkee)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文利用A2C、PPO、DDPG、TD3、SAC等actor‑critic深度强化学习代理，在印度市场上对5只大盘股进行交易模拟，并将情绪分析结果注入状态空间和奖励信号。

**💡 创新点**

创新点在于将FinBERT情绪倾向深度融入模型（分为部分与深度注入），并对比仅基于策略优化与结合Q‑学习的代理在情绪驱动下的性能差异。

**🔧 技术方法**

技术包括MDP建模、技术指标特征、FinBERT情绪提取、A2C/PPO（策略优化）以及DDPG/TD3/SAC（策略+Q‑学习）等深度强化学习算法。

**📊 数据集**

数据集由2019‑2024年间的5只印度大盘股历史行情（Yahoo Finance）和对应的新闻标题（Google RSS）构成，并用FinBERT生成情绪标签与分数。

**📈 对比分析**

通过三阶段实验（无情绪、部分情绪、深度情绪）使用年化收益、波动率、Sharpe比、稳定性、最大回撤等指标比较，结果显示DDPG在深度情绪下取得最高收益和Sharpe，PPO在无情绪时最佳，TD3在深度情绪下表现下滑。

**⚠️ 局限性**

局限性包括：情绪注入导致奖励信号波动、TD3对更新过于保守导致表现下降、仅测试5只股票可能缺乏普适性、缺乏可解释性分析，且结果对其他市场的可迁移性尚未验证。

---

## 338. AlphaAnything: Extending Pretrained RGB Generators to RGBA

**arXiv ID:** 2610.32413 | [PDF](https://arxiv.org/pdf/2610.32413v1)

**作者:** Qijie Wang `[一作]` (Tsinghua University), Bin Wang `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在预训练的 RGB 生成器的原始潜空间中，加入 LoRA 模块和轻量级 alpha 分支，使模型能够在不进行额外 RGBA 重建训练的情况下，直接学习并生成 RGBA 图像。

**💡 创新点**

①只在原始 RGB 潜空间学习透明度，无需额外 RGBA 潜空间；②在解码器中加入单向 RGB→Alpha 的 alpha 分支，保持任何固定潜空间下的 RGB 解码不变；③提出端点感知 Alpha 损失 (EA Loss)，能够同时处理完全透明/不透明端点、连续透明度以及局部错误。

**🔧 技术方法**

LoRA 参数化扩展 DiT（或其他预训练 RGB 生成器）transformer；冻结 RGB VAE 解码器并嵌入 alpha 分支；端点感知 Alpha 损失基于 Fisher–Rao 坐标与 Fenchel–Young 损失；联合流匹配（flow‑matching）训练，仅训练 LoRA + alpha 头。

**📊 数据集**

生成了 8,963 张 RGBA 训练样本；使用 FLUX.1‑dev、FLUX.2 Klein Base 9B、Qwen‑Image、LongCat‑Image 等预训练 RGB 基础模型；图像到 RGBA 采用 PrismLayersPro 校正后的 prompt，评估集包含 RGBA‑Text（294 句子）、RGBA‑Image（735 图像）、AlphaTest（402 句子）等。

**📈 对比分析**

与 AlphaVAE、LayerDiffuse、OmniAlpha、Qwen‑Image‑2.1 等现有文本/图像到 RGBA 方法对比，使用 CLIPScore、Aesthetic Score、ImageReward、FID、RGB L1、Soft IoU、SAD、LPIPS、Alpha Grad、Alpha Conn 等指标。实验表明，在视觉质量、文本‑图像对齐、alpha 边界与连通性等方面均优于或与竞争对手相当，尤其 EA Loss 在 alpha 评估指标上显著提升。

**⚠️ 局限性**

缺乏大规模高质量 RGBA 数据集与统一评测基准，导致方法在不同任务和模型上的泛化评估有限；目前指标主要关注 RGB 细节与 alpha 质量，尚未形成完整的全局 RGBA 评估框架。

---

## 339. Differentiable Combinatorial Auction Design: Optimizing Revenue-Welfare-Fairness Frontiers With Graph Attention Networks

**arXiv ID:** 2610.31977 | [PDF](https://arxiv.org/pdf/2610.31977v1)

**作者:** Mai Pham `[一作]` (Dartmouth), Peter Chin `[通讯]` (Dartmouth)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

设计了一种可微分且约束感知的组合拍卖框架，利用深度学习实现可行的近似策略-proof机制，并在机场时隙分配的实际案例中进行验证。

**💡 创新点**

创新点包括：① 用图结构 CAGraph 对组合冲突进行建模并通过约束因子化实现完全可行的分数分配；② 采用匹配追踪（Matching Pursuit）分解将分数分配转换为可执行的离散方案；③ 在同一模型中同时平衡收益、福利和公平，绘制可供监管者选择的 Pareto 前沿。

**🔧 技术方法**

技术方法：深度学习（CANet、CAFormer、CAGraph）+ 可微分约束因子化层 + 匹配追踪分解 + RegretNet 风格的约束正则化 + Jain 指数公平目标 + 自适应惩罚调节。

**📊 数据集**

数据集：① 合成分布（添加互补性项的多项式组合拍卖）；② 机场时隙真实数据——使用美国航班信息、T‑100 载客量、BTS 航班统计等，构造了 4 家航空公司、4 个时隙组合的 Washington National (DCA) 例子。

**📈 对比分析**

比较方法：与 VCG、随机序列决定（RSD）、传统组合拍卖（AMA、VVCA）以及 RegretNet/RegretFormer 进行对比；评估指标包括收入、社会福利、Jain 公平指数以及经验 regret。结果显示：在合成实验中，CAGraph 的收益比 VCG 高 73.5%；在 DCA 真实案例中，收益提高 180%（或 284% 在全分配实现下）；收益提升同时保持 regret 小于 0.01，福利损失可控。

**⚠️ 局限性**

局限性：① 近似 DSIC 仅在分数层保证，离散化后仍需额外审核；② 对非 bihierarchy 约束系统的实现可能存在残差；③ 训练模型对特定分布敏感，结构变化需重新训练或微调；④ 在极大规模时隙/组合场景下的可扩展性尚未充分验证；⑤ 收入波动大，需引入全分配或其他后处理来提升稳定性。

---

## 340. Toward an Agentic AI Approach to Public Health Data Collection

**arXiv ID:** 2610.31746 | [PDF](https://arxiv.org/pdf/2610.31746v1)

**作者:** Xueyi Zhang `[一作]` (University Of Virginia), Ensheng Dong `[通讯]` (Rand Corporation)

**通讯引用:** 11450 | [OpenAlex ID](https://openalex.org/A5066854264)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `67630363-6be0-4f51-ab05-7198250671a5` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

开发了一套基于Agentic AI的公共卫生疾病监测数据收集工作流，覆盖源发现、源评估、证据结构化、事件链接和验证等全流程，支持自适应、证据可追溯的数据采集。

**💡 创新点**

创新点在于将传统的固定源、规则化处理转变为目标驱动的Agentic AI工作流，利用多代理协同、可持续的任务状态以及证据共享机制，实现对信息环境变化的自适应响应和专家审查的无缝衔接。

**🔧 技术方法**

技术实现基于LangGraph框架搭建多代理网络，使用Claude Sonnet 4.6等大型语言模型完成检索、提取、归一化与验证；结合Tavily Search进行源发现，采用证据链接、对比冲突、重合判定等逻辑实现完整的自动化与半自动化流程。

**📊 数据集**

实验使用了三组公开病例数据：2026年土蝇病毒（Hantavirus）爆发、2025年加拿大麻疹（Measles）以及2025年塞拉利昂猴痘（mpox），对照集分别来自Global.health、PHAC和WHO等官方数据库。

**📈 对比分析**

通过与人工校准基准或官方参考的比较，Hantavirus案例完整恢复13例，链接85个发布域（vs 28）；麻疹最终计数仅比PHAC低0.33%，且在79.8%可比日累计值高于PHAC；mpox在加入媒体报告后，7日内可用报告日比例从28.2%提升至65.6%，参考日期的平均计数差距大幅下降。总体表现显示覆盖度、连续性和时效性均有提升，但个体记录的完整性和准确性仍与传统方法相近。

**⚠️ 局限性**

局限性包括：未在前瞻性场景中验证时效性提升；模型输出可能引入误差，需跨源校验与人工复核；对网页变化、模型更新的鲁棒性尚未评估；计算成本与人工审查需求随数据量增大而上升；以及不同模型间的可迁移性和一致性仍未确认。

---

## 341. Multi-Material Topology Optimization for Targeted Structural Dynamic Vibration Attenuation

**arXiv ID:** 2610.31996 | [PDF](https://arxiv.org/pdf/2610.31996v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 342. Async-RBench: Benchmarking Asynchronous Task Execution in LLM Agents

**arXiv ID:** 2610.32429 | [PDF](https://arxiv.org/pdf/2610.32429v1)

**作者:** Detao Zeng `[一作]` (University of Electronic Science and Technology of China), Zhong Zhang `[通讯]` (University of Electronic Science and Technology of China)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了 Async-RBench 基准，用于评估 LLM 代理在异步子代理执行中如何处理结果并更新任务状态。

**💡 创新点**

创新点在于提供 Event Response Contract (ERC) 规范每个异步事件的期望响应，并通过 Dynamic Replanning Score (DRS) 独立评估过程与结果，区分任务成功与事件处理质量。

**🔧 技术方法**

采用子代理架构、情景控制器（Scenario Controller）管理结果交付、ERC 规范、DRS 评分机制。

**📊 数据集**

使用来自 Terminal-Bench、Gaia2、OSWorld、SWE-bench 与 MultiAgentBench 的 200 个可执行任务，涵盖 8 种异步中断场景。

**📈 对比分析**

对 9 种 LLM 代理进行评测，发现异步交付导致任务成功率下降（平均 26% vs 29.7%），DRS 分数 19.2–45.1，强制重建计划仅提升 2.6–3.5 分。

**⚠️ 局限性**

局限性包括异步事件处理仍较困难，DRS 与最终任务成功关联弱，强制重建计划效果有限，以及评估仍受异步交付不可预知性影响。

---

## 343. From Noisy Retrieval to Decision Evidence: Silver Evidence Demonstrations for In-Context Learning

**arXiv ID:** 2610.32128 | [PDF](https://arxiv.org/pdf/2610.32128v1)

**作者:** Masoud Monajatipoor `[一作]` (Optum AI), Robert E. Tillman `[通讯]` (Optum AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种异步的银证据示例（silver‑evidence）格式，用于检索增强的少样本推理（RAG + ICL），在推理时先生成银证据再做决策；

**💡 创新点**

创新点在于将演示中的原始检索文档替换为目标条件的、无标签的银证据摘要，并在演示中对其进行专家监督，形成前后不对称的 ICL 结构；

**🔧 技术方法**

使用大型语言模型（GPT‑5.5 等）完成两阶段生成：Stage 1 生成银证据，Stage 2 在一次调用中生成银证据并给出最终标签与正确信息；

**📊 数据集**

实验涵盖四大任务：Evidence Inference 2.0、SciFact、Synthetic Prior‑Authorization Policy Retrieval、HICRIC/IMR；

**📈 对比分析**

与传统 RAG、Gold‑only ICL、Silver‑only ICL、InstructRAG 等方案对比，Full Method 在三/四个数据集上均显著提升标签准确率（最高提升 33 点），并在所有标注的 Evidence‑F1 上超越基线；

**⚠️ 局限性**

局限包括：银证据仍受检索质量限制，Oracle 证据仍表现最佳；方法在 EI 2.0 仅略有提升，说明当检索已简洁时压缩优势有限；对不同任务的泛化与解释性需要进一步验证。

---

## 344. Tracing Money Laundering Across DeFi, Bridges, and Mixers on Ethereum

**arXiv ID:** 2610.32210 | [PDF](https://arxiv.org/pdf/2610.32210v1)

**作者:** Huan Xing `[一作]` (Taiyuan University of Technology), Xing Fan `[通讯]` (Taiyuan University of Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 FPTracer，基于上下文感知转移树（CATT）恢复交易内部资产流路径，并以此为基础进行跨链桥和混音器的资产追踪与分析。

**💡 创新点**

创新点在于：①将交易执行结构与原始转移信息结合构造 CATT，准确重建复杂 DeFi 交易的资产流路径；②设计跨链意图的多视角特征（资产流、执行状态）并训练无先验桥信息的分类器；③利用已追踪的洗钱网络为混音器后的潜在流向提供证据，实现混音器后追踪。

**🔧 技术方法**

技术方法包括：CATT 构造与路径恢复算法、路径特征提取、XGBoost 交叉链交易分类器、混音器存款识别规则及后续候选提取。

**📊 数据集**

数据集：20 条 2019-2025 年的真实洗钱案例（共约 5.47 亿美元）及 AMLGuard 标注的 82 起案例，用于路径恢复、网络扩展与桥检测的评估；同时使用公开桥列表与 Mixers 记录进行验证。

**📈 对比分析**

与 XBlockFlow、AMLGuard-Public 对比：FPTracer 在地址召回率 98.6%、目标召回率 94.2% 领先；跨链检测 F1>98%（XGBoost），100% 报告桥覆盖率，额外发现 30 条未公开桥；混音器后分析在 7 起案例中与公开报告一致的 222/442、47/82 关联。

**⚠️ 局限性**

局限性：无法在混音器后得到确切的存款‑提取对应关系；对极端协议或未见过的桥设计的泛化仍受限；评估主要基于公开案例，可能存在未公开洗钱事件或桥使用未被覆盖。

---

## 345. DriveMind: Mind-Evolving Belief Tracking for Closed-Loop Autonomous Driving

**arXiv ID:** 2610.32069 | [PDF](https://arxiv.org/pdf/2610.32069v1)

**作者:** Yan Shi `[一作]` (University of Washington), Yinhai Wang `[通讯]` (University of Washington)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了DriveMind框架，利用多模态大语言模型（VLM）持续维护并更新多假设的自然语言推理，以异步快慢架构结合扩散控制器实现闭环自动驾驶；

**💡 创新点**

首次将粒子滤波式多假设维护与重加权引入驾驶决策，解决传统单假设“先下结论”导致的错误预判，并通过异步推理显著提升推理效率；

**🔧 技术方法**

核心技术包括：VLM驱动的假设生成与打分、基于粒子滤波的预测‑校正‑重采样、文本嵌入聚合的信念上下文、扩散式轨迹生成器以及异步快慢并行推理框架；

**📊 数据集**

在CARLA Bench2Drive和NAVSIM两个驾驶基准上进行实验；

**📈 对比分析**

与无信念跟踪的基线对比，Bench2Drive上Driving Score从68.50提升至81.37，Success Rate从41.36%提升至54.09%，在无信号交叉口、对向车道等模糊场景分别提升约20点，多能力平均得分提升至58.17%；

**⚠️ 局限性**

限制包括：假设数量与更新频率固定，未按实时情景复杂度自适应；仅考虑车辆自身的操控意图，未显式预测其他车辆的未来意图；推理耗时仍高，异步机制需进一步优化；

---

## 346. Not All Retained Clients Are Equal: Client Selection for Federated Unlearning

**arXiv ID:** 2610.31786 | [PDF](https://arxiv.org/pdf/2610.31786v1)

**作者:** Thanh Linh Nguyen `[一作]` (Trinity College Dublin), Quoc-Viet Pham `[通讯]` (Trinity College Dublin)

**通讯引用:** 18636 | [OpenAlex ID](https://openalex.org/A5062525719)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种基于路径依赖的客户端选择框架（Path-Dependent Unlearning，简称PD-Unlearn），用于在联邦学习后高效去除指定客户端的影响。

**💡 创新点**

创新点在于：①引入临界学习期（Critical Learning Period）权重和目标解耦度（Target Decoupling）度量，捕捉客户端在训练过程中的影响路径；②设计轻量级、目标无关的历史统计，能够在不存储完整梯度或模型快照的情况下对每个客户端进行排序；③在未完全重训的前提下，仅利用排序出的少数顶尖客户端进行重训练，显著降低通信和计算成本。

**🔧 技术方法**

主要技术包括：FedAvg 联邦训练、基于 Federated Gradient Norm 的临界期检测、归一化方向统计（z_i）、历史对齐度（s̅_i）、路径依赖得分（f_i），以及 top‑k 客户端选择与后续重训练。

**📊 数据集**

使用的数据集有：FMNIST（CNN），CIFAR‑10 与 CIFAR‑100（ResNet‑18）。

**📈 对比分析**

与 Retrain、CONDA、FedRecovery、FOUL、FedEraser、Exact‑Fun、FUSED、GDFA 等现有联邦去学习方法相比，PD‑Unlearn 在 FA（目标客户端精度）降至 0% 的同时保持 RA（保留客户端精度）与 Retrain 相当或更高；MIA‑AUC 与 Retrain 差距 ≤ 0.4%；通信量降低约 69%，计算量降低 57–61%。

**⚠️ 局限性**

局限性包括：①需要预先设置 β、λ、k 等超参数，对不同场景需调优；②仅针对客户端级去学习，尚未验证对样本/类别级别的扩展；③依赖于临界期检测的准确性，若训练过程不易分辨临界期可能影响排序效果；④在极端异构或极大规模客户端场景下，历史统计仍可能成为瓶颈。

---

## 347. Easy-3Dfy: Object-Level Geometric Composition for Single-Image 3D Scene Construction

**arXiv ID:** 2610.32250 | [PDF](https://arxiv.org/pdf/2610.32250v1)

**作者:** Ziang Zhang `[一作]` (Zhejiang University), Zhou Zhao `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6514db3d-8de6-452c-91b7-acdb31787cc4` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出 Easy-3Dfy 框架，将单图 3D 场景构建拆解为对象级生成与几何对齐，通过 RPF 对齐器实现对象到场景点云的注册

**💡 创新点**

解耦对象生成与布局预测，使用基于 RPF 的点云对应生成器和 Umeyama 拟合恢复变换，完全无需完整场景监督，支持模块化升级

**🔧 技术方法**

使用 RPF（Rectified Point Flow）对齐器、Umeyama 拟合、单目点图估计（MogeV2/Depth Anything V3）、图像编辑完成（FLUX-2、GPT-Image-2）和多种图像到 3D 生成模型（Hunyuan3D、Meshy-T2 等）

**📊 数据集**

在 Scene-Maker（1000 场景）和 3D-FUTURE200（200 场景）数据集上进行评估

**📈 对比分析**

与 SAM-3Dfy、MIDI、SceneGen 在共享网格、共享 mask 的实验中比较，Easy-3Dfy 在 Scene-CD、Scene-FScore、BBox IoU 等几何指标上均优于对手，整体提升显著

**⚠️ 局限性**

仅评估几何指标，未覆盖外观、语义、完整性；对小/遮挡严重对象、低质量点图等情况易出现配准失效，纹理细节的评估不足

---

## 348. SPECMEM: Accelerating Agentic Tool Calling via Live Memory Management

**arXiv ID:** 2610.31840 | [PDF](https://arxiv.org/pdf/2610.31840v1)

**作者:** Zihe Song `[一作]` (University of Illinois Chicago), Philip S. Yu `[通讯]` (University of Illinois Chicago)

**通讯引用:** 142472 | [OpenAlex ID](https://openalex.org/A5036357902)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究在长时间多用户部署中，如何保持工具调用记忆库活跃并在推理时进行投机解码。

**💡 创新点**

提出三大创新：1）在线写回、有限容量和 LRU 驱逐的生命周期管理；2）按用户本地化划分存储，减少跨用户泄露；3）基于声明的副作用门控，防止投机执行不可逆操作。

**🔧 技术方法**

技术核心包括检索式草稿生成（基于查询嵌入相似度）、在线写回与 LRU 驱逐、侧效门控、以及投机-验证解码框架。

**📊 数据集**

使用公开工具调用基准（BFCL、Seal‑Tools、API‑Bank、ToolAlpaca、ToolBench 等）扩展为多用户、多会话查询流，人工模拟用户重复与新任务。

**📈 对比分析**

与冻结全局存储、基准投机解码等方案对比，平均接受率提升27%（+11%–+34% across benchmarks），墙时速度提升最高1.74×；局部作用域对接受率无显著影响，但显著抑制跨用户泄露；副作用门控显著降低错误执行成本。

**⚠️ 局限性**

局限性包括：用户与会话模拟非真实日志，缺乏自然复用结构；评估指标未覆盖多轮选择与不相关性；检索阈值固定且未调优；副作用门控依赖正确的工具注解，且仅做二元判定。

---

## 349. Benchmarking Data-Free Continual Model Merging for Whole-Slide Image Survival Analysis

**arXiv ID:** 2610.32207 | [PDF](https://arxiv.org/pdf/2610.32207v1)

**作者:** Vu Minh Tran `[一作]` (University of Information Technology), Khang Nguyen `[通讯]` (University of Information Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个端到端的无数据持续模型融合基准，用于全切片图像（WSI）生存分析。

**💡 创新点**

首次系统评估多种融合策略在连续生存预测中的表现，并揭示基于删失的评估会显著改变方法排名。

**🔧 技术方法**

利用TITAN预训练网络，结合六种融合技术（Weight Averaging、Task Arithmetic、TIES-Merging、Dual Orthogonal Projection、Null‑Space Filtering、Orthogonal Projection‑based Continual Merging），并在TASK‑IL和CLASS‑IL推理模式下进行评估。

**📊 数据集**

使用四个TCGA癌症队列（BLCA、UCEC、LUAD、BRCA），通过CONCH特征提取和Cox回归模型训练。

**📈 对比分析**

通过C-index、IPCW‑C-index、遗忘（Forgetting）和后向迁移（Backward Transfer）等指标比较方法性能；结果显示NUFILT在传统C-index上最高，而TA和TIES在IPCW‑C-index上更具优势，表明方法排名随评估指标变化。

**⚠️ 局限性**

局限包括仅基于单一TITAN骨干网络、固定采样策略、仅四个队列且未探究跨中心域移位或任务序列不平衡带来的影响。

---

## 350. Decoupling Patient Personas and Clinical Workflows: A Dual-Agent Approach to Realistic Medical Dialogue Generation

**arXiv ID:** 2610.32443 | [PDF](https://arxiv.org/pdf/2610.32443v1)

**作者:** Qiyuan Li `[一作]` (Wuhan University of Science and Technology), Jinguang Gu `[通讯]` (Wuhan University of Science and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了CoRPS-Dial框架，用于在预诊前通过双代理（患者人格与临床流程）生成高质量的医患对话。

**💡 创新点**

创新点在于：①将患者异质性与指南驱动的临床流程解耦，双代理交互实现可控生成；②引入两阶段后置质量控制与选择性重写，显著提升医学一致性与文本自然度。

**🔧 技术方法**

使用大型语言模型（如Qwen3-32B、DeepSeek-R1-14B）完成患者人格建模、临床流程树构建与对话生成；规则引擎与GLM-4-9B奖励模型做后置评估；LoRA微调LLama-3.1-8B等模型进行下游对话任务。

**📊 数据集**

基于来自三级医院心血管科的256条真实预诊对话，经过手工标注与去标识化，生成5,120个患者画像，再扩增生成9,714条合成对话，随后与真实对话混合做下游微调。

**📈 对比分析**

通过结构、语义与多样性指标、人工评估以及LLM评估与下游诊断对话任务验证。CoRPS-Dial在所有指标上均优于基线（如Raw Doctor、Free Patient、Single Agent等），尤其在角色一致性、流程完整性和上下文连贯性上提升显著；在下游模型微调后性能提升高达20%+。

**⚠️ 局限性**

局限性包括：LLM仍存在幻觉风险，未加入专家实时验证；仅在冠心病单一疾病上验证，通用性待进一步评估；数据来源单一机构，可能存在样本偏倚；框架主要面向科研，临床部署仍需额外安全与合规保障。

---

## 351. PASTA: Prompt-Aware Annealed Diffusion Sampling

**arXiv ID:** 2610.32390 | [PDF](https://arxiv.org/pdf/2610.32390v1)

**作者:** Ren Ajiki `[一作]` (Hosei University), Jinjia Zhou `[通讯]` (Hosei University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种训练无关的扩散采样方法PASTA，能在图像生成时保留用户指定的空白区域，避免前景内容泄漏。

**💡 创新点**

创新点在于利用共享潜在轨迹对全提示与上下文提示的残差进行空间融合，并引入时间衰减的混合权重，使得空白区域在早期采样阶段得到强烈约束，后期恢复完整提示的一致性。

**🔧 技术方法**

使用了扩散模型的分类器无关引导、共享潜在残差融合、上下文提示拆分、遮罩引导的加权混合以及基于 UniPC 的多步调度器。

**📊 数据集**

在 P2P 与 SynPrompt 两个空间感知文本到图像生成基准数据集上进行实验，进一步扩展了大面积空白掩码的评估。

**📈 对比分析**

与 TextCenGen、SAWNA 等现有方法相比，PASTA 在大掩码条件下显著降低了侵入率（约为0.038–0.048），且在 CLIP 与 HPS 评分上保持竞争力，显示出较高的文本-图像一致性和感知质量。

**⚠️ 局限性**

在极端空间限制（空白区域占比>80%或剩余生成区域过窄）下，PASTA 可能无法生成合理内容，需进一步研究数据驱动或适配训练方案。

---

## 352. Controlled Evaluation of EEG Foundation Models: Random Initialisation, Band-Power and Validation-Selected Probes

**arXiv ID:** 2610.32067 | [PDF](https://arxiv.org/pdf/2610.32067v1)

**作者:** Mingyu Jeon `[一作]` (Modulabs), Jae Young Suh `[通讯]` (Modulabs)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估了五个公开 EEG 基础模型和一个内部模型，在统一协议下加入随机初始化、带电平基线和验证选择层等三项控制，系统性检验其下游性能。

**💡 创新点**

创新点在于将三项常被遗漏的对照（随机编码器、频段功率基线、验证层选择）统一纳入评估，并公开每个实验细胞结果，揭示大部分报告的提升在此控制下消失。

**🔧 技术方法**

使用线性探针、全微调、共享学习率调度、SincNet 前端等技术，对模型进行统一评估。

**📊 数据集**

使用 APAVA、BCI IV 2a、Lee2019、TUSL、TUAB 等五个基准及扩展的 P300、睡眠分期、情绪识别、受试内 MI 四个任务数据集。

**📈 对比分析**

通过与随机初始化、带电平基线以及验证选择层对比，发现随机初始化往往匹配预训练；微调增益多数落在种子噪声范围；大部分基准下模型无法超越基线，性能普遍不佳。

**⚠️ 局限性**

局限包括仅评估六个模型，使用单窗口评估，未覆盖所有 EEG 任务，部分基准可能与预训练数据重叠，且未在预训练阶段加入控制。

---

## 353. From Item Scores to Semantic Tokens: Test-Time Collaborative Adaptation for Generative Recommendation

**arXiv ID:** 2610.32098 | [PDF](https://arxiv.org/pdf/2610.32098v1)

**作者:** Xiongfeng Shan `[一作]` (Shenzhen Technology University), Xiuqiang He `[通讯]` (Shenzhen Technology University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种测试时协同束搜索自适应方法TCBSA，能够在不更新任何模型的前提下，将独立训练的判别式推荐器给出的全量物品分数融入冻结的生成式语义ID模型的束搜索决策中，提升推荐准确率。

**💡 创新点**

创新点在于：①将判别式分数的边际值映射到共享语义前缀的边上；②通过置信门控调节协同潜能的影响力；③在生成过程之前对束搜索的路径分数进行修正，实现提前纠正早期裁剪失误。

**🔧 技术方法**

使用技术包括：语义ID生成式推荐模型（如TIGER、MiniOneRec）、判别式推荐模型（SASRec、GRU4Rec等）、束搜索、置信门控、边聚合（Top‑M均值/最大等）以及离线预计算的物品-路径映射。

**📊 数据集**

实验数据集涵盖Amazon Review 2023的四个子域（Science、Music、Office、Yelp）以及Yelp本身，均采用全量catalog评估。

**📈 对比分析**

与多种基线（判别式、生成式、训练时增强、后期融合等）比较，TCBSA在16组数据集‑指标组合中取得最高Recall@K和NDCG@K，特别在Yelp、Emerging‑Interest场景提升显著；在束宽度增大时仍保持优势，且在相同推理时间下往往优于更宽束的原生成模型。

**⚠️ 局限性**

局限性包括：①增加了额外的推理延迟；②需要预先构建完整的物品-路径映射，且对语义ID固定；③对置信门控与聚合策略的超参数敏感；④主要针对语义ID生成模型，未验证在其他生成框架下的通用性。

---

## 354. DealsFlow: An Evidence-Gated Agentic Harness for Venture Capital Deal Triage

**arXiv ID:** 2610.32380 | [PDF](https://arxiv.org/pdf/2610.32380v1)

**作者:** Ming Ze Tang `[一作]` (Queen's University Belfast), Vasiliki Carson `[通讯]` (Sapphire Capital Partners)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并评估了名为DealsFlow的LLM辅助工具，用于英国早期阶段VC的投融资决策前筛选，结合证据引用、分数上限与严格的下调机制，并记录完整的审计日志。

**💡 创新点**

创新点在于将LLM的判分外包给代码，采用阶段化证据上限、只降低分数的“质疑者”双重门控，以及预先声明依赖的检索模式，保证分数可追溯且不受过度乐观或跑跑差异影响。

**🔧 技术方法**

核心技术包括：LLM（GPT‑6 Luna）进行文本阅读与评分、检索增强生成（RAG）与预取查询、阈值裁剪与下调算法、结构化分数组合、以及完整的审计日志与人工复核流程。

**📊 数据集**

使用了Sapphire Capital Partners 2026年7月8日导出的243笔SEIS/EIS投资组合数据（除2笔异常外）以及20份新递交的推介稿进行实验与时间测评。

**📈 对比分析**

通过Spearman相关、AUC（写亏vs非写亏、>2×vs非>2×）与对照基线（仅证据、结构化字段回归、初始承诺等）进行比较。完整管线在142笔成熟投资上Spearman≈0.45、AUC写亏≈0.71、AUC>2×≈0.79，相比无门控LLM提升约0.07–0.07，但改进幅度较小；在实际流程中，平均人工时间由44.7分钟降至18.9分钟（约58%节省）。

**⚠️ 局限性**

局限包括：仅评估已被基金筛选通过的投资（存活偏差）、检索未时间戳化导致的后视漏斗、LLM知识截止后可知晓结果、未测量多次运行方差、单一基金单一模型、评估样本量有限、以及未独立验证质疑者的具体贡献。

---

## 355. WorldMirror: Agentic Real-to-Sim Generation of Interactive Worlds from In-the-Wild Robot Demonstrations

**arXiv ID:** 2610.31834 | [PDF](https://arxiv.org/pdf/2610.31834v1)

**作者:** Tianling Xu `[一作]` (Xspark AI), Yue Chen `[通讯]` (Xspark AI)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `51c0528b-f690-4182-ae60-bb5f046c276c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

开发了 WorldMirror，一套基于代理的实景到仿真框架，能够将真实世界机器人演示视频与轨迹转换为可直接执行的数字孪生环境。

**💡 创新点**

创新点在于将场景重建过程转化为可执行的世界程序，通过一个路由代理动态调用专门工具并利用验证反馈实现迭代纠错；同时提出摄像机姿态锚定的全局坐标系统，保证重建与原始轨迹在同一空间框架下无额外运动规划。

**🔧 技术方法**

核心技术包括：多模态代理（如 GPT‑6 Astra、Articraft 等）、专门化重建工具（Meshy、Gaussian 表示、渲染‑对比优化）、结构化世界状态存储、验证模块（视觉、空间、物理一致性评估）以及摄像机姿态锚定坐标变换。

**📊 数据集**

使用 DROID 数据集（10 个机器人操作演示场景），每个场景提供 RGB 视频和机器人轨迹。

**📈 对比分析**

与 RoLA、RoboSnap、SimFoundry、GPT‑6 Astra 等基线比较。WorldMirror 在视觉（LPIPS、DISTS）、空间（遮罩 IoU、位置误差、相对向量/距离误差）以及重放一致性（运动误差、接触 F1、成功率）指标上均优于或竞争性地击败了这些方法，尤其在直接轨迹重放成功率上达到 100%。

**⚠️ 局限性**

局限性包括：实验仅覆盖少量 DROID 示例，未验证跨任务泛化；重建质量受限于现有的生成与仿真工具；未在下游策略学习或真实部署中进行评估。

---

## 356. SWE-Clar: Can Coding Agents Clarify Incomplete Client Requirements and Deliver Greenfield Software?

**arXiv ID:** 2610.31854 | [PDF](https://arxiv.org/pdf/2610.31854v1)

**作者:** Jialei Li `[一作]` (University of Science and Technology of China), Zhao Chen `[通讯]` (Upwork Inc)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究提出了SWE‑Clar基准，用于评估自由职业场景下绿地软件开发的需求澄清与代码实现能力；

**💡 创新点**

创新点在于：①澄清先行、实现独立；②通过独立的需求问答评估澄清效果；③基于真实高价值Upwork项目构建任务与测试；

**🔧 技术方法**

主要技术包括大型语言模型（如GPT‑6 Astra等）作为编码、客服、评审代理；采用自动化评测框架、接受测试与参考答案匹配；

**📊 数据集**

使用了由100个真实Upwork绿地项目组成的任务集，包含512个需求单元（问题+参考答案）以及对应的接受测试；

**📈 对比分析**

在9个前沿模型上进行比较，得到实现得分39–67%，澄清得分71–87%，GPT‑6 Astra最高；两项得分呈正相关，但整体完成率仍低；

**⚠️ 局限性**

局限性包括：仍难以完成多数任务；澄清与实现虽相关但不完全一致；基准仅覆盖一次性澄清场景，未考虑需求变化或客户持续性；测试基于静态接受测试，缺乏动态交互验证。

---

## 357. Learning Nonlinear Waveform Fusion Across Scales for Integrated Sensing and Communication

**arXiv ID:** 2610.32099 | [PDF](https://arxiv.org/pdf/2610.32099v1)

**作者:** Jiayu Liu `[一作]`, Giuseppe Caire `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `e0540dec-d77f-42db-94ae-d039248f6393` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `f86bf285-fd08-4156-973b-6e6481af8fa0` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种通过在多尺度下学习图像特征与定向LFM参考信号的非线性交互来生成集成感知与通信（ISAC）波形，支持 UE 端图像重建和基站端多目标时距角定位。

**💡 创新点**

创新点在于：① 采用可自适应的非线性波形融合机制，使得波形生成与当前图像内容和参考信号相互耦合；② 在波形生成时跨尺度融合（时间与频率）并结合上下文信息；③ 在接收端使用已知参考和信道状态信息进行神经解混，随后基站端通过基于实现波形的连续搜索和局部学习校正实现精细定位。

**🔧 技术方法**

主要技术包括：卷积神经网络图像编码、复数卷积与多尺度时间/频率特征提取、上下文条件的复数增益与权重、连续搜索与物理细化、局部CNN学习校正，以及基于实现波形构造的相干字典。

**📊 数据集**

使用 COCO 训练集（≈10,000 张图片）进行波形学习，验证集和测试集各 512 张图片，生成 16 组脉冲，每脉冲 256 次样本，采用 8×8 天线阵列。

**📈 对比分析**

对比方法包括时间域叠加（TD）、学习的频谱线性融合（SL）、固定 FDM 以及优化 FDM 四通道。实验表明，所提波形在 11 种任务权重平均下，归一化定位误差比 TD 低 29.0%（CI 20.9–36.6%）、比 SL 低 27.0%（CI 19.2–34.4%），图像 PSNR 分别提高 1.40 dB 与 0.82 dB；在基站端采用连续搜索+局部校正进一步降低定位误差 12.2%。

**⚠️ 局限性**

局限性：① 仅针对静态多目标，未考虑多普勒或动态场景；② 训练数据来自合成图像，真实环境迁移性能未知；③ 波形设计受总能量约束，可能在更高功率/多天线场景下表现不一致；④ 需要额外的神经解混和校正网络，部署成本相对较高。

---

## 358. RoboCode: Learning Reusable Action Primitives for Robot Manipulation

**arXiv ID:** 2610.32225 | [PDF](https://arxiv.org/pdf/2610.32225v1)

**作者:** Xianyue Peng `[一作]`, Iman Soltani `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了RoboCode，一种两阶段框架，利用软门控原语代码表将机器人短期动作片段表示为多原语的柔性组合，并在闭环控制中实现跨任务复用。

**💡 创新点**

创新点在于将动作片段视为软组合的多个学习到的原语，而非单一离散码或直接连续预测，构建共享代码表并通过相似度门控实现原语的可复用与可解释性。

**🔧 技术方法**

使用了两阶段学习：Stage 1训练共用动作自编码器（编码器、相似度门控、解码器）以学习原语代码；Stage 2训练任务特定观测编码器，将观测映射到共享潜在空间，再通过冻结的门控接口生成动作片段；损失包括重建、投影、稀疏与多样性等。

**📊 数据集**

使用ManiSkill五个操控任务（LiftPegUpright、PickCube、PokeCube、PushCube、StackCube）的约30k条短期动作片段作为训练与评估数据。

**📈 对比分析**

与ACT、Diffusion Policy和DP‑Large比较，RoboCode在所有任务中取得最高或相同的成功率，平均成功率75.6%（ACT 68.8%，DP‑Large 41.6%），并在大多数任务中步数更少、返回更优。

**⚠️ 局限性**

仅在仿真环境中验证，未解决真实世界感知、动力学与控制不确定性；任务特定编码器基于离线模仿学习，可能受状态分布漂移影响，需进一步研究sim‑to‑real与分布偏移问题。

---

## 359. Energy-Aware Performance Characterization of Kokkos Kernels Across Precision Levels

**arXiv ID:** 2610.31917 | [PDF](https://arxiv.org/pdf/2610.31917v1)

**作者:** Maria Patrou `[一作]` (Oak Ridge National Laboratory), Jean-Luc Fattebert `[通讯]` (Oak Ridge National Laboratory)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文对基于 Kokkos 的 GPU 内核在 FP32、FP64 与混合 FP32/FP64 精度下的算术强度、吞吐量及能耗进行了系统性特征化，并提出了能耗感知 Roofline 与热感知跨内核能耗模型。

**💡 创新点**

创新点在于将能耗与温度、执行顺序三维信息融入 Roofline 可视化，同时通过 PAPI+NVML 的细粒度与粗粒度测量相结合，揭示不同精度对能耗与性能的综合影响。

**🔧 技术方法**

主要技术手段包括 Kokkos Tools 的空间-时间堆栈与 PAPI 高级/低级接口、NVML 能耗与温度计数器、H100 GPU 计数以及自定义的能耗读取插件。

**📊 数据集**

实验使用了两套真实工作负载：LOBPCG 求解器（大/小规模稀疏矩阵）和 ArborX 的 DBSCAN 聚类（37M 与 497M 点的宇宙学数据集）。

**📈 对比分析**

通过对 FP32、FP64、FP32/FP64 三种精度下的算术强度、吞吐量、能耗和温度进行对比，结果表明 FP32 在性能-能耗平衡上最佳，FP64 能耗最高，混合精度在保持精度的同时能耗略高。

**⚠️ 局限性**

局限性包括 NVML 能耗计数的 100 ms 采样间隔导致短内核能耗测量不精确、仅关注 GPU 能耗、实验范围仅限两款应用且未进行实际功率/热控制干预。

---

## 360. Heterogeneous Space Data Centers: A Key Paradigm for Future Non-Terrestrial and In-Orbit Services

**arXiv ID:** 2610.32260 | [PDF](https://arxiv.org/pdf/2610.32260v1)

**作者:** Peng Hu `[一作]` (University of Manitoba), Amjad Iqbal `[通讯]` (University of Manitoba)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了异构空间数据中心（HSDC）架构，利用大中型节点与小型节点配合，为轨道、空中与地面用户提供AI推理与训练服务，并通过仿真评估其性能与能耗。

**💡 创新点**

创新点在于：① 将空间数据中心拆分为资源可分化的S-SDC与L-SDC，匹配不同负载与能量需求；② 设计跨轨道的节点部署与交互接口（I1‑I4）；③ 以黎明–黄昏同步轨道为大节点的能量供给基础，并针对低地球轨道的光照与通信窗口进行系统级优化。

**🔧 技术方法**

使用的技术包括：轨道动力学仿真（Kepler方程+地球扁率），光学与射频链路模型，基于GPU时钟的计算负载转换，电源与热力学估算（包括辐射器与光伏面板面积），以及分布式队列与任务调度模型。

**📊 数据集**

采用的工作负载数据集为两项卫星成像推理任务：船舶检测与野火分割，原始数据来自文献，利用Jetson AGX Orin的计时结果按H100速度比15进行缩放，产生约20 MB的推理结果。

**📈 对比分析**

比较方法：三种部署配置（仅地面、仅L‑SDC、HSDC）在48 h仿真周期内对场景到结果的延迟与能耗进行统计。HSDC在轨道用户的平均延迟≈2 s，95 %延迟<7 s；L‑SDC配置因链路窗口限制导致95 %延迟≈29 min；地面用户延迟均为数十分钟。能耗方面，L‑SDC每日约1470 kWh，S‑SDC约60 kWh，HSDC总能耗≈6 100 kWh，能耗几乎全部来自大节点。

**⚠️ 局限性**

局限性包括：未考虑S‑SDC到L‑SDC的任务转发与重训练过程；链路模型未纳入大气天气、拖曳效应；加速器性能按规格估算而非实测；未对多运营商共享与数字孪生调度细节进行建模。

---

## 361. TENET: Time-point Encoding Network for Multivariate Time Series Anomaly Detection

**arXiv ID:** 2610.32337 | [PDF](https://arxiv.org/pdf/2610.32337v1)

**作者:** Dongchan Cho `[一作]` (SimPlatform Co. Ltd.), Namsoon Jung `[通讯]` (SimPlatform Co. Ltd.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种针对多变量时序异常检测的时间点中心编码网络 TENET，专门构造当前时点的潜在状态并用固定 NLL 评分进行异常检测。

**💡 创新点**

创新点在于将窗口视为构造当前点状态的上下文，而非直接重建或预测；通过按变量的时点检索历史、交叉注意力以及自门控机制，实现决策对齐的潜在表示。

**🔧 技术方法**

使用的核心技术包括：变形的 Temporal Block（anchored cross‑attention 与 lag embedding）、跨变量 Transformer 交互、求和聚合得到系统状态，以及固定标准正态负对数似然（NLL）作为一类评分。

**📊 数据集**

在四个公开基准上评估：PSM（25 维）、SMD（38 维）、SWaT（51 维）和 SWAN（38 维）。

**📈 对比分析**

与 20+ 传统与最新的基线（如 PCA、OCSVM、DAGMM、OmniAnomaly、Anomaly Transformer、PatchTST、TimesNet 等）对比，TENET 在 AUC‑ROC、AUC‑PR、V‑ROC、V‑PR 等多指标上均实现了显著提升，特别是范围感知指标提升最高可达 52% 左右；同样在不同一类目标下也表现稳健。

**⚠️ 局限性**

局限性包括：需要较大窗口和变量维度时计算与内存开销仍显著；对异常样本的训练不做显式建模，若异常与正常分布相近仍可能产生误检；此外，固定 NLL 评分虽然简单，却可能限制对更复杂异常模式的捕捉。

---

## 362. Speculative Diffusion is a Tale of Two Tradeoffs

**arXiv ID:** 2610.31819 | [PDF](https://arxiv.org/pdf/2610.31819v1)

**作者:** Aaron Zorel `[一作]` (Independent Researcher), Daniel Israel `[通讯]` (University of California)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了将扩散模型作为草稿器与失真可接受的投机式解码相结合（即投机式扩散），通过同时调整扩散步骤数（K）和接受宽容度（λ）来探索推理速度与质量的 Pareto 前沿；

**💡 创新点**

创新点在于（1）理论证明在总变差约束下线性插值目标分布是最优接受策略；（2）提出并验证了同时调节扩散努力与接受宽容度可显著扩展传统单一参数的 Pareto 前沿；（3）构建了简化的吞吐量模型，解释了速度-质量关系；

**🔧 技术方法**

技术包括扩散大语言模型（Masked 与 Uniform）、投机式解码、线性插值接受规则、Cactus 失真接受、批量多步扩散草稿、以及对推理时序的分析和吞吐量建模；

**📊 数据集**

使用数据集包括 GSM8K（数学推理）、HumanEval（代码生成）和 GPQA Diamond（通用推理），并在 Dream‑7B、DiffusionGemma‑26B‑A4B 等草稿模型与 Qwen3‑8B、Qwen2.5‑32B 等验证器上进行实验；

**📈 对比分析**

通过在相同验证器下比较多种草稿器（Uno、DFlash、DSpark、SDAR 等）以及不同 K 与 λ 组合，测量延迟、吞吐量与准确率。结果显示，联合调节 K 与 λ 能显著提升速度-准确率 Pareto 前沿，特别是 SDAR 在无专门训练的情况下也能逼近最佳前沿；

**⚠️ 局限性**

局限性包括（1）部分草稿器无法支持多步扩散导致比较不完整；（2）理论模型假设统一接受率，未完全捕捉多步动态影响；（3）实验仅覆盖数学、代码、通用推理任务，未验证对更广泛领域的适用性；（4）缺乏动态调度或查询路由的实际部署实验。

---

## 363. Acoustic and Lexical Signals for Native Language Identification of Language Learners

**arXiv ID:** 2610.32382 | [PDF](https://arxiv.org/pdf/2610.32382v1)

**作者:** Ahmet Yavuz Uluslu `[一作]` (University of Cambridge), Mark Gales `[通讯]` (University of Cambridge)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文比较了基于WavLM的音频L1识别器与基于GPT‑4o的文本零样本L1识别器在学生英语语料中的表现，探讨了学习者熟练度和输入长度对两种模态的影响，并研究了两者的融合效果。

**💡 创新点**

创新点在于首次系统地将音频与文本两种模态的L1识别性能在同一数据集上进行对比，揭示了熟练度差异导致的模态优势转移，并展示了线性融合可显著提升识别准确率。

**🔧 技术方法**

使用了WavLM‑Large作为音频特征提取器，并在其隐藏层上学习权重组合；使用OpenAI GPT‑4o进行零样本文本分类，并通过提示模板与候选标签映射实现多选任务。

**📊 数据集**

主要使用了BULATS‑L1（自报L1、CEFR平衡评估集）和Speak & Improve 2025（含人工与ASR转录的对照）两大数据集。

**📈 对比分析**

在BULATS‑L1上，音频模型总体准确率85.56%，文本模型82.59%；音频在A1水平占优，文本在A2–C1占优；两者融合后准确率提升至94.81%。文本模型对输入长度高度敏感，音频模型对长度更稳健。

**⚠️ 局限性**

局限包括：对自报L1的依赖、对ASR误差的敏感性未完全解决、未深入探讨不同L1间的差异特征，以及融合方法仅为线性插值，未尝试更复杂的多模态学习。

---

## 364. PersonaScienceBench: Evaluating Scientific Agents with Diverse Simulated Researchers

**arXiv ID:** 2610.32106 | [PDF](https://arxiv.org/pdf/2610.32106v1)

**作者:** Siyang Wu `[一作]` (University of Chicago), Yuexing Hao `[通讯]` (MatrAIx)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了 PersonaScienceBench，利用 200 个从五个科学基准（DiscoveryBench、BixBench、EXP-Bench、SciConvBench、ASI-Bench）提取的任务，在与模拟研究者的交互中评估语言模型在科学工作中的协作表现。

**💡 创新点**

创新点在于：① 在固定任务与评分标准的前提下引入多样化的研究者 persona，真实模拟不同科研者的沟通偏好；② 通过交互式评估同时捕捉任务完成度、信息恢复和语言重复等多维度行为，揭示仅凭最终分数无法看到的协作差异。

**🔧 技术方法**

使用技术包括：大型语言模型（GPT‑6 Astra、Claude 系列、Gemini、GPT‑5、Grok 等）作为代理与研究者模拟器；MatrAIx 生成的 1,290 维度 persona 结构体作为研究者身份；交互式任务转换算法和多轮对话记录；多种评价指标（分数归一化、事实恢复率、词汇自重复率）。

**📊 数据集**

数据集：五个公开科学基准（共 200 题）和 MatrAIx 生成的 599,847 人工与 400,000 合成研究者 persona 样本；通过选择四个覆盖不同沟通特征的 persona 进行实验；对每个模型进行 1,000 次 episode（200 题 × 5 研究者设置）。

**📈 对比分析**

比较方法：对 18 种模型配置分别在 5 种研究者设置下进行单轮实验，计算归一化分数、事实恢复率和自重复率；使用 95% 置信区间的 bootstrap 抽样评估统计显著性。性能发现：某些模型（如 GPT‑5 mini）在 persona‑off 与 persona‑on 之间可提升 5.26 分；Gemini 3.7 Flash 在某些领域表现明显优于其 Flash 版本；不同研究者 profile 在不同学科域表现差异显著，表明单一评分无法全面评估。

**⚠️ 局限性**

局限性：仅涵盖文本、数据分析和代码执行任务，未涉及图像处理、物理实验或多研究者长期项目；交互限制为单代理与单模拟研究者的有限轮次；实验仅使用四个精选 persona，无法代表真实科研者多样性；评估中模型与模拟器使用同一 LLM，难以单独定位代理能力；指标（自重复率、嵌入分离）仅捕捉部分交互质量，未直接评估科学有效性。

---

## 365. Before the Ticket: Grounding HPC User Support in Documentation and Resolved Tickets with an Agentic LLM Assistant

**arXiv ID:** 2610.31703 | [PDF](https://arxiv.org/pdf/2610.31703v1)

**作者:** Elham J. Barezi `[一作]` (Purdue University), Niveditha Nerella `[通讯]` (Purdue University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在Purdue大学的RCAC高性能计算中心构建了一个协同助手，结合了基于文档的检索增强生成（RAG）和基于历史工单的微调LLM两大知识来源，并实现了端到端的对话系统。

**💡 创新点**

创新点在于：①将工单历史转化为可安全使用的训练数据（提取、重写、过滤、PII屏蔽）；②采用两工具并行调用、显式优先级规则合成答案的agentic架构；③系统性评估对比四种方法（基准模型、微调模型、文档RAG、工单RAG）以及合成与单工具的差异。

**🔧 技术方法**

主要技术包括：LLM微调（Llama‑3‑8B‑Instruct + QLoRA）、RAG（Sentence‑BERT + FAISS + BM25 + Cross‑Encoder）对文档和工单检索、Model Context Protocol（MCP）统一工具调用、FastAPI/CLI接口、LLM判定器（Qwen2.5‑72B & Gemma‑3‑27B）进行答案质量评估。

**📊 数据集**

使用RCAC公开的849页用户手册和2023年起的15,268条工单历史，经过处理后得到11,244对问答训练集、600对验证集、400对测试集。

**📈 对比分析**

在400条保留测试工单上，微调模型在BERTScore、ROUGE、Cosine以及LLM判定器的正确率方面均显著优于基准模型、工单RAG和文档RAG；文档RAG在完整度上最高，但正确率最低。端到端合成系统未能提升单工具性能；最佳的“oracle”选择策略（基于判定器得分挑选最佳答案）可将正确率提升至3.41（相比微调模型提升0.21）。

**⚠️ 局限性**

局限性包括：①对工单历史的重写和过滤仍可能漏掉误导信息；②微调模型受限于本地硬件，无法使用更强的云模型；③评估仅基于LLM判定器，尚无人工专家评测；④模型与判定器属于同一模型族，可能导致偏倚；⑤实验数据仅来自RCAC，缺乏跨机构验证。

---

## 366. RSIHub: A Composable Framework for Agent-Evolution Methods

**arXiv ID:** 2610.32253 | [PDF](https://arxiv.org/pdf/2610.32253v1)

**作者:** Zimu Wang `[一作]` (University of California, Berkeley), Zilong Zheng `[通讯]` (Beijing Institute for General Artificial Intelligence)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出 RSIHub 框架，用来把语言模型代理进化方法的策略与执行基础设施分离，提供可组合的实验流水线和可重用的生命周期操作。

**💡 创新点**

创新点在于：①定义了可组合的“recipe”抽象，允许用同一套生命周期操作实现多种进化策略；②提供了受保护的执行和评估层，确保评估结果可信且不可被策略修改；③公开了共享实验基础设施，简化实验重现与方法对比。

**🔧 技术方法**

主要技术包括：基于 Harbor 的容器化执行与轨迹捕获；分离的生命周期操作（Select, Rollout, Mutate, Gate, Record 等）与可插拔实现；基于 Git 和 SHA‑256 的版本与证书机制；使用 GPT‑5.4 作为目标与变异模型。

**📊 数据集**

使用的数据集包括 Terminal Bench 2（软件工程/终端交互）和 τ^3 Banking（知识驱动对话）两大 benchmark，每个 benchmark 分为 train、gate、sealed 子集。

**📈 对比分析**

方法比较：在相同模型、评估器和任务拆分下，对四种进化 recipe（A‑Evolve、AHE、GEPA、Hyperagents）以及基准 hill‑climbing 进行十代演化。结果显示：在 Terminal Bench 2 上所有方法提升 7.8–13.5 分，Hyperagents 与 A‑Evolve 达到 69.7；在 τ^3 Banking 上提升幅度因起始代理而异，Hyperagents 对 MiniSWE 最高，AHE 对 Codex 最高。实验还验证了框架的执行忠实度（与直接执行差异 ≤ 6%）和完整性（所有 420 次故障注入试验未出现误入侵）。

**⚠️ 局限性**

局限性：依赖可信的框架服务、宿主与文件系统保护、Git 哈希校验等；评估结果仅在本文测试的工作负载、主机与生命周期边界下可信；未验证适应性 recipe 修订（外部代理改写）对安全的影响；不提供完整行为等价性证明。

---

## 367. KiRoSe: Using Kinematic Data for Efficient Segmentation of Robot Demonstrations

**arXiv ID:** 2610.31801 | [PDF](https://arxiv.org/pdf/2610.31801v1)

**作者:** Shuangyu Xie `[一作]` (University of California, Berkeley), Ken Goldberg `[通讯]` (University of California, Berkeley)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `5a41884c-404f-4688-a89c-aa238c10fe68` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用机器人运动学序列自动划分时间子段并生成对应的自然语言动作描述，达到机器人演示数据的自动语义分割；

**💡 创新点**

提出稀疏运动令牌（SMT），通过子任务边界监督学习得到低率离散运动词表，并将其与开源语言模型结合，实现跨机器人零样本的时间与语义分割；

**🔧 技术方法**

使用时域卷积运动编码器、向量量化+K‑means生成SMT、预训练语言模型（Qwen3.5‑9B）并通过LoRA适配，以及时间标记和任务指令的输入；

**📊 数据集**

在ABC‑130K（YAM）机器人数据集、AgiBot World 和 Ego‑Exo4D 人类手部轨迹数据集上进行训练与评估；

**📈 对比分析**

与FAST、OAT、MotionGPT 等基准及视频基线对比，KiRoSe 在 YAM 上 Boundary F1 在 ±1 s 约 0.63（基线约 0.42），跨机型零样本 F1 约 0.40，语义 Grounded F1 约 0.59，且在逆向进度估计中表现为正负相关切换，优于基准模型；

**⚠️ 局限性**

主要局限在于仅依赖运动学信息，缺乏视觉、接触或物体中心特征，导致对高度动态或未见语义的适应性仍有限。

---

## 368. KVStream: Adaptive Parallel Storage Management for KV Caching in Large-Scale LLM Serving

**arXiv ID:** 2610.31827 | [PDF](https://arxiv.org/pdf/2610.31827v1)

**作者:** Avinash Maurya `[一作]` (Argonne National Laboratory), Bogdan Nicolae `[通讯]` (Argonne National Laboratory)

**通讯引用:** 1644 | [OpenAlex ID](https://openalex.org/A5085745891)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种多层存储级 KV 缓存引擎，利用节点本地 NVMe 与并行文件系统（PFS）并行读取前缀 KV 并推迟写入，显著加速长上下文 LLM 的推理；

**💡 创新点**

创新点在于：①将前缀读取视作多路聚合带宽调度问题；②通过批量读取、写入延迟、全层复制和基于完成的路由四种机制，动态平衡各存储层负载；③无需预测或调优即可实现接近理论最优的读取速度；

**🔧 技术方法**

技术包括：异步 I/O、并行读取池、写入延迟、全层复制、meet‑in‑the‑middle 路由、与 LMCache、SGLang 等推理引擎无缝对接的 C++ 库；

**📊 数据集**

评估数据集包括 LooGLE、NarrativeQA、CUAD（离线）和 Qwen‑Coder、Chutes、SWE‑bench agentic（在线）等；

**📈 对比分析**

与 CPU‑only、LMCache、Mooncake Store、SGLang HiCache 等方案对比，最多可提升 1.49× 的离线吞吐率，在线 TTFT 最多降低 6.1×，在 128 GPU 规模下检索接近 95% 的带宽上限；

**⚠️ 局限性**

局限性：需至少两层存储才能获益；复制会导致写入开销翻倍；当工作集可装入主内存或前缀复用稀疏时加速效果有限；多节点间未实现跨节点协同共享 KV 复制。

---

## 369. Full-Weight VLA Fine-Tuning Without an fp32 Master Copy Silently Erases Most Updates

**arXiv ID:** 2610.31859 | [PDF](https://arxiv.org/pdf/2610.31859v1)

**作者:** Shubham Singh `[一作]` `[通讯]`, Shubham Singh

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

探讨在视觉‑语言‑动作（VLA）模型微调过程中，若权重以bfloat16存储且未使用fp32主拷贝，AdamW会导致大部分小幅度更新被舍弃，从而严重影响训练效果；并提供解决方案：在训练前将可训练权重保持为fp32，使用bfloat16进行计算。

**💡 创新点**

发现并量化了缺失fp32主拷贝时的“静默消失”现象，证明这会导致训练中大量更新被忽略；同时证明通过恢复fp32主拷贝可显著提升模型性能（超过10个百分点）。

**🔧 技术方法**

利用bfloat16混合精度训练、AdamW优化器、学习率预热与余弦退火、对训练过程的probe监测以及统计检验（McNemar检验、点差阈值）。

**📊 数据集**

在LeRobot的SmolVLA与π_0/π_0.5两大VLA模型上进行实验，并在LIBERO-Object、Hugging Face Hub等公开微调模型数据集上进行验证。

**📈 对比分析**

将fp32主拷贝与不拷贝两种设置在相同学习率下进行对比，使用固定规则（差距≥10点）评估性能。实验显示，在fp32主拷贝下SmolVLA在默认学习率下提升约10点，低学习率下提升约45点；π_0.5在低学习率下提升约76点；在多数实验中fp32主拷贝显著提高成功率与损失曲线。

**⚠️ 局限性**

实验范围受限于两模型、单一任务集合、固定种子与训练时长；未探索更高学习率或不同VLA模型；未针对bf16时的梯度累积或其他错误补偿策略进行细化研究。

---

## 370. Beyond 3SUM-Hardness for Geometric Problems

**arXiv ID:** 2610.32162 | [PDF](https://arxiv.org/pdf/2610.32162v1)

**作者:** Nick Fischer `[一作]` `[通讯]` (Max Planck Institute for Informatics), Nick Fischer (Max Planck Institute for Informatics)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文重新证明了许多基于 3SUM 的几何问题在 k‑AP 假设下仍具有二次时间下界。

**💡 创新点**

创新点在于发现可以用 k‑GeomBase 取代传统 GeomBase 进行归约，从而几乎不需要修改原有证明，恢复了大量下界。

**🔧 技术方法**

主要技术包括 k‑AP 检测、颜色编码（color coding）、几何构造（如多条水平墙与孔洞的设计）以及对已有归约的直接重写。

**📊 数据集**

论文为纯理论工作，未使用实验数据集。

**📈 对比分析**

作者通过比较传统 3SUM 基础下的 O(n^2−ϵ) 下界与新的 k‑AP 基础下的同等下界，说明在 k‑AP 假设下仍保持相同的计算复杂度限制；未提出新的算法，只是维持了原有的下界。

**⚠️ 局限性**

限制：仅覆盖了已知通过 GeomBase 归约的几何问题，无法覆盖所有 3SUM 下界；此外 k‑AP 假设本身尚未得到实证或证明，仍是一种假设。

---

## 371. Learning to Be Monitored: Causal Bounds and Intervention-Aware Post-Training Recipes for Chain-of-Thought Monitorability

**arXiv ID:** 2610.31733 | [PDF](https://arxiv.org/pdf/2610.31733v1)

**作者:** Qiang Fu `[一作]` (Yale University), Arman Cohan `[通讯]` (Yale University)

**通讯引用:** 11272 | [OpenAlex ID](https://openalex.org/A5064858748)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了如何通过后训练提升大型语言模型生成的链式推理（CoT）可监测性，并提出了新的因果评估界限。

**💡 创新点**

提出了最小判定灵敏度上界和可观测下界的因果记忆度界限，并构建了多任务对照-干预数据集与三种后训练方法（离线自蒸馏、强化学习、在线自蒸馏）。

**🔧 技术方法**

使用结构因果模型分析、上界下界推导、离线自蒸馏SFT、基于监视奖励的强化学习、在线自蒸馏；并在 MonitorBench 上评估。

**📊 数据集**

从公开数据集合并并自动化生成的控制–干预任务集合，覆盖医疗、数学、法律等领域，包含多种答题格式；并使用 MonitorBench 的12个干预任务。

**📈 对比分析**

与基线模型在三种监视器（Qwen3.32B、Llama3.3-70B、GPT-OSS-120B）下进行宏观平均比较；所有方法均实现了 ^2 和 ^2 分数提升，最大提升达30+点，尤其在 Mistral Small 通过 RL 提升显著。

**⚠️ 局限性**

界限仅为上界/下界，无法精确测定因果召回；评估仅覆盖输入干预监测，未验证过程或结果属性；三种训练方法的对比条件不完全统一；训练数据缺乏人工注释，实验范围受限。

---

## 372. 4DMemory: Benchmarking Long-Horizon Spatiotemporal Memory with Harness-Augmented Agents

**arXiv ID:** 2610.32332 | [PDF](https://arxiv.org/pdf/2610.32332v1)

**作者:** Yanhao Jia `[一作]` (Nanyang Technological University), Erik Cambria `[通讯]` (Nanyang Technological University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了覆盖约900小时视频的LongVideo数据集和4DMemoryBench基准，并构建了4DMemory Harness，通过持久化记忆、检索增强生成和多种专用工具实现对长时序多模态推理的支持。

**💡 创新点**

创新点在于：①首个支持单个问题跨越超过100小时视频的多模态基准；②提出模型无关的 harness，将任务时长与模型原生上下文窗口解耦；③通过可复用、无问题依赖的记忆与工具编排提升长时序推理能力。

**🔧 技术方法**

采用视频分割与帧采样、SAM 3.1视觉定位、ASR 转写、结构化记忆、检索增强生成（RAG）以及22+专业工具模型（计数、排序、跨模态关联等）实现多模态知识抽取与推理。

**📊 数据集**

基于约900小时的独立网页视频与两部电视剧（包含单集、季集、全剧）构建，生成5,214个人工审核的问答对，覆盖视觉、文本与时空推理。

**📈 对比分析**

在13个主流LLM/MLLM API上（无任何微调）进行基准测试，评估工具调用率、执行顺序合规性与最终答案准确率；Harness提升超过50%，最高联合成功率达77.8%，但相比先前基准模型性能下降>10%。

**⚠️ 局限性**

局限性包括：未进行代理微调，未知在其他下游任务中的效果；基准和工具库的通用性待验证；以及仍未彻底解决分布式计数与多时段聚合的挑战。

---

## 373. MBABench-2.0: Evaluating AI Agents on Constructing Financial Spreadsheets Beyond Simple Verifiability

**arXiv ID:** 2610.31711 | [PDF](https://arxiv.org/pdf/2610.31711v1)

**作者:** Julian Poeltl `[一作]` (ESB Reutlingen), Hongseok Namkoong `[通讯]` (Columbia Business School)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了MBABench，一个公开的金融电子表格从零开始构建任务基准，涵盖多种金融模型并引入多维质量评价标准；

**💡 创新点**

创新点在于（1）构建完整的从零开始构建任务集合，突破仅填空式评测的局限；（2）提出覆盖可读性、可维护性、可审计性等方面的质量评价体系；（3）开发并验证了一种可扩展的LLM评判器，实现大规模评测；

**🔧 技术方法**

使用大语言模型（GPT‑5.6 Sol、Fable 5.1等）作为评判器，并利用多种LLM Agent harness（GUI、Code、Excel、In‑house）进行任务执行；

**📊 数据集**

数据集为101个手工构造的金融建模任务，涵盖估值、三表模型、债务表、个人财务等七类，约50个中等难度及以上任务，总计约33小时人类构建时间；

**📈 对比分析**

对20余个Agent在不同harness下进行评测，使用Pass Rate和Score指标；结果显示即便是最强的Agent（如GPT‑6 Astra）在所有任务中仅通过约10‑15%，其准确率在中等难度及以上任务上也低于60%；

**⚠️ 局限性**

局限性包括（1）评判器仍有误判率，尤其是高层次质量判断；（2）任务难度评估仅基于专家经验，缺乏自动化分层；（3）未提供人类基准或对行业实际工作量的对照；（4）仅覆盖财务领域，难以直接推广至其他领域。

---

## 374. PAFS: Perturbation-Aware Firing Scaffolds for Robust Event-Based SNNs

**arXiv ID:** 2610.32409 | [PDF](https://arxiv.org/pdf/2610.32409v1)

**作者:** Ning Liu `[一作]` (University of Western Australia), Thomas Bräunl `[通讯]` (University of Western Australia)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种针对冻结事件驱动脉冲神经网络的后训练校准框架 PAFS，利用后期 LIF 脉冲轨迹与残差对齐的低秩子空间来纠正模型在时间扰动下的残余误差。

**💡 创新点**

创新点：①构造扰动感知脉冲框架，将残差相关的脉冲动态投影到低秩子空间；②仅在冻结网络后添加轻量级 Adapter 作为锚点，并用校正头实现单次推理；③通过标准化、误差加权和 SVD 提取残差对齐方向，实现对残余误差的精确捕捉；④在多种事件数据集和 Transformer 架构上验证其跨模型通用性。

**🔧 技术方法**

技术手段：Leaky‑Integrate‑and‑Fire (LIF) 神经元、Spiking Transformer 编码器、轻量级瓶颈 Adapter、残差对齐的标准化与加权、SVD 构造低秩 scaffold、校正线性头、跨域多任务训练与验证调参。

**📊 数据集**

使用的数据集：CIFAR10‑DVS、DailyAction‑DVS、DVS‑Gesture‑Chain (DVS‑GC)，并在 Spikformer 与 Spikingformer 两种 spiking Transformer 结构上进行实验。

**📈 对比分析**

对比方法：LoRA、Adapter、Plain MLP、Logit‑only MLP 等基线。PAFS 在四个数据集–模型组合中，平均受损（Corr. Avg.）和最差条件（Worst）准确率均优于锚点、LoRA 和 Logit‑only，最高提升约 3.06–3.65 个百分点，仅增加约 0.34% 的可训练参数。

**⚠️ 局限性**

局限性：仅在事件摄像头数据和特定 Transformer 架构上验证；对速度变化的鲁棒性提升有限；需要训练样本进行校准，无法在无标签场景直接使用；对极端扰动或不同噪声模型的泛化能力未充分评估；依赖后期脉冲轨迹信息，若 backbone 结构变化或缺失 LIF 细节，方法不可直接迁移。

---

## 375. OlmoEarth v1.3: Embeddings

**arXiv ID:** 2610.32294 | [PDF](https://arxiv.org/pdf/2610.32294v1)

**作者:** Gabriel Tseng `[一作]` (Allen Institute for AI), Ted Schmitt `[通讯]` (Allen Institute for AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

本文提出 OlmoEarth v1.3，一种面向地球观测的视觉 Transformer 模型，兼顾高效嵌入生成与 fine‑tune，发布 10 m 分辨率的 128/64 维全局嵌入；

**💡 创新点**

创新点在于引入 Attention Pooler 进行多时序、多模态聚合与可调分辨率推理，并通过直接监督与多层次蒸馏实现极低维嵌入，同时显著降低预训练成本；

**🔧 技术方法**

采用 Vision Transformer 编码器‑解码器、掩码图像建模、RoPE 相对位置编码、Attention Pooler、PixLatentMIM Lite、cosine/Gram 蒸馏损失、Searchlight 推理、Zarr 存储与 int8 同号幂量化；

**📊 数据集**

使用 Sentinel‑1、Sentinel‑2、Landsat‑8/9 影像及 WorldCereal、OpenStreetMap、WorldCover、Cropland Data Layer、SRTM、Canopy Height Map 等辅助地图；评估基准包括 AEF、TESSERA‑v2、PASTIS、PLANTEUR 等；

**📈 对比分析**

通过 kNN‑20、线性探针、Ridge、RF、MLP 等分类器对比，OlmoEarth v1.3 128 维在 7/8 任务领先，64 维在 5/7 任务领先；PASTIS mIoU 为 59.8（patch 2）/58.6（patch 4），整体 kNN/LP 任务略优于前代，finetuning 平均排名 4.34；

**⚠️ 局限性**

局限在于 Attention Pooler 在单图像单模态任务上效果不佳，导致性能下降；全局嵌入推理成本高（约 14k H100 小时/年），对计算资源要求较高；量化误差虽极小，但仍有 0.00003 余弦误差；许可证限制阻止军事、国防及开采行业使用。

---

## 376. Real-Time Localisation of Dorsal Anatomical Regions on Dairy Cows in Commercial Rotary Parlours from RGB-D

**arXiv ID:** 2610.32174 | [PDF](https://arxiv.org/pdf/2610.32174v1)

**作者:** Vineet Prasad `[一作]` (Murdoch University), Hai Wang `[通讯]` (Murdoch University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6514db3d-8de6-452c-91b7-acdb31787cc4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

本文提出了一种单网络多任务RGB‑D模型，能在旋转式挤奶台的实时视频中自动定位目标奶牛、十个背部骨点以及四个命名背部区域，并将这些区域与深度信息一起输出。

**💡 创新点**

创新点在于将颜色和深度分别编码、在特征金字塔层级进行晚期融合，并同时学习稠密的区域分割与稀疏的骨点热图，利用两头一致性检测实现无标签质量评估。

**🔧 技术方法**

技术实现使用两条ImageNet预训练的Encoder（如ResNet‑18、MobileNetV3、PVTv2或ConvNeXt），共享特征金字塔解码器，三头输出（目标牛掩码、骨点热图+可见性、四区域掩码），并在训练中加入一致性损失。

**📊 数据集**

数据集基于公开的Kinect v2旋转式挤奶台RGB‑D录制，构造538帧标注（10个骨点、4个规则区域、目标牛掩码），并进行牛、会话拆分。

**📈 对比分析**

在牛‑分离的测试集上，10得分达到0.905，区域IoU最高可达0.91；与单模态或早期融合相比，晚期融合和区域头显著提升精度；模型在T4 GPU上每帧26.4 ms，帧率远高于摄像机11 fps。

**⚠️ 局限性**

局限性主要来自摄像机视角和光照：前部牛身遮挡、与侧面或后方遮挡导致标记不准；热像仪视场与RGB‑D不匹配，部分区域温度缺失；且在不同站位或光照下仍需进一步验证。

---

## 377. Rethinking Representation Geometry in Supervised Contrastive Learning under Long-Tailed Data

**arXiv ID:** 2610.32039 | [PDF](https://arxiv.org/pdf/2610.32039v1)

**作者:** Joonhyeok Jeong `[一作]` (Yonsei University), Kibok Lee `[通讯]` (Yonsei University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并验证了结构平衡的完全平衡对比学习（FBCL），用理论框架分析了长尾分布下监督对比学习目标是否收敛到正则简单形，并提出结构平衡以消除类别不平衡对目标的影响。

**💡 创新点**

通过构造下界和正则简单形的最优性条件，首次证明SCL与BCL在长尾情形下不能收敛到正则简单形；引入结构平衡并提出FBCL，其全局最优解为正则简单形；同时证明结构平衡可使类内方差惩罚与类别频率无关。

**🔧 技术方法**

使用下界分析、类中心化、正则简单形几何条件、结构平衡权重重构，以及双分支网络、对数几率调整交叉熵等训练框架。

**📊 数据集**

在ImageNet-LT、iNaturalist等长尾分类数据集，以及CIFAR-10/100、MIT67、DTD、Oxford-IIIT Pets、Caltech101、Flowers、CUB200、Food、Stanford Cars、FGVC Aircraft等迁移学习数据集上进行实验。

**📈 对比分析**

与多种长尾识别方法（OLTR、τ-norm、cRT、PaCo、GPaCo、LA、DisAlign、Hybrid-PSC、MiSLAS、KCL、TSC、SBCL、Focal-SAM、BCL、DBM等）以及其他对比学习损失（SimCLR、SCL、BCL）在同一训练框架下进行对比，FBCL在多项指标上均实现了最佳或最接近最佳的准确率，并在迁移学习中取得最高平均性能。

**⚠️ 局限性**

理论分析基于完整类别批次和K≤h+1的假设，未探讨大规模训练中类覆盖不完全或K>h+1的情况；实验仍在批量包含所有类别的设定下验证，实际分布与理论假设的偏差仍需进一步研究。

---

## 378. Decode Late, Multiply Once: Rethinking the Dot Product for Quantized Weights

**arXiv ID:** 2610.31913 | [PDF](https://arxiv.org/pdf/2610.31913v1)

**作者:** Jonathan Tapson `[一作]` `[通讯]` (BrainChip Inc), Jonathan Tapson (BrainChip Inc)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出了一种新的计算低分辨率权重的点积方法，通过在乘法之前对激活进行累加，减少了乘法和解码的次数。

**💡 创新点**

创新点在于将传统的点积计算重构为先累加后乘法的方式，从而消除了对解码权重向量的需求，并显著降低了乘法的数量。

**🔧 技术方法**

使用了散射-累加处理单元（SA-PE）技术，该技术替代了传统的乘加单元，并在权重静态的系统阵列中实现。

**📊 数据集**

使用了低分辨率（1-4位）权重和高分辨率（16或32位）激活的数据集进行实验。

**📈 对比分析**

与传统的乘加方法相比，新方法在乘法次数上减少了n到最多2^k，且在能量和性能上表现出显著的优势，尤其是在乘法器和解码器成本较高的情况下。

**⚠️ 局限性**

限制在于当2^k接近n时，收益会减小，并且需要在硬件上实现地址选择的累加器以便有效利用该方法。

---

## 379. Not an Accident: Exploiting Safety Filter Inadequacy in Humanoid Robots

**arXiv ID:** 2610.31923 | [PDF](https://arxiv.org/pdf/2610.31923v1)

**作者:** Tiffany Bao `[一作]` (Arizona State University), David Brumley `[通讯]` (Carnegie Mellon University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文发现并演示了基于优化的安全过滤器在类人机器人中的缺陷，并提出了一种自动化工具（SAFE-CE），能够在给定任务与环境下识别易受攻击的安全状态并合成“目标注入”攻击，使机器人进入不安全状态并碰撞。

**💡 创新点**

创新点在于：①首次揭示安全过滤器两种失败模式（逃逸速率不足与控制授权不足）及其对类人机器人安全性的影响；②从安全视角提出目标注入攻击模型，并基于交叉熵方法设计高效搜索算法；③在仿真与真实Unitree G1机器人上实现并验证攻击，证明攻击的可行性和实效性。

**🔧 技术方法**

使用技术包括：运动学/动力学模型、控制理论、优化安全过滤器（SSA、SSS、CBF、R-SSA、R-SSS、R-CBF、p-SSA）、逃逸率与控制授权分析、交叉熵方法（CEM）进行目标搜索、Python实现与SPARK仿真框架。

**📊 数据集**

数据集：SPARK v1.0 测试平台提供的多种场景与任务的benchmark，涵盖静态/动态障碍、全身/上身任务以及多种安全指数定义。

**📈 对比分析**

与基线随机搜索方法比较：SAFE-CE 通过CEM搜索能够在SPARK数据集中三倍速率发现攻击；在仿真与真实Unitree G1机器人上均成功导致碰撞，验证了攻击的实效性。

**⚠️ 局限性**

limitations：需要攻击者对目标机器人安全过滤器及其参数完全了解；无法评估攻击普遍性和成功概率；未考虑感知、定位等系统误差对攻击效果的影响；攻击假设只通过合法任务接口操作，未考虑更弱信息获取的情况。

---

## 380. The 3DS Oracle: BIN Attack via the Network

**arXiv ID:** 2610.32206 | [PDF](https://arxiv.org/pdf/2610.32206v1)

**作者:** Wasif Faisal `[一作]` (BRAC University), Suprava Saha Dibya `[通讯]` (BRAC University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `9cc9baba-5356-466d-81ff-d80028d90279` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对 EMV 3‑D Secure（3DS）预授权阶段的客户端可见响应进行系统研究，利用该阶段的“lookup‑stage”结果作为账户识别 oracle，证明攻击者可在不提交金融授权的情况下通过查询 Luhn‑合法的 PAN 并解析返回的状态码与事务 ID 匹配情况，来对同一 BIN 范围内的潜在卡号进行区分与筛选。

**💡 创新点**

① 发现并量化了预授权阶段的响应泄漏，展示其可作为枚举攻击的入口；② 引入事务 ID 等号（acsTransID ≡ dsTransID）与 reason code 语义的联合判定，区分 DS‑站内生成 vs. ACS 处理，构造了双层过滤模型；③ 通过对 7 个不同 ACS/BIN 组合的“anchor‑card”实验，证明了在大多数部署中存在显著的响应分离，说明此漏洞已在实战中被利用；④ 提出了以整合层为切入点的治理建议（最小化返回、统一分支、会话绑定）。

**🔧 技术方法**

使用协议规范分析、架构重建、形式化建模（事务 ID 等价性、reason code 语义推理）、静态文档审计、公开工具与 CaaS 服务扫描、以及受限的实时测试（off‑context HTTP caller + 伪造浏览器 telemetry）。

**📊 数据集**

实验数据来自 7 个研究者自有 anchor 卡（不同 BIN/ACS 组合），每组测试了 5–10 个相邻同 BIN 的候选 PAN；收集了 3DS gateway 交互日志、返回 JSON 以及事务 ID、reason code、ECI 等字段。

**📈 对比分析**

在实验设置中，6/7 组在 lookup‑stage 就能区分 anchor 卡与相邻未必已发卡的 PAN（即响应类别分离），显示出可将搜索空间从 10⁹–k 次查询压缩至 1 次代表查询的潜力；未做大规模计量对比，但通过观察响应分布证明该方法在实际部署中可实现有效的预授权筛选。

**⚠️ 局限性**

研究受限于：仅测试了单一 Braintree/Cardinal 结合，未覆盖所有支付渠道（如 native‑app、Split‑SDK）；未能确认相邻 PAN 是否真实已发卡；未评估分布式代理、速率限制与检测阈值；未测量对持卡人通知与安全日志的影响；并且实验仅在伦理范围内进行，无法证明网络级普遍可行性。

---

## 381. Principia: Test-Time Hypotheses to Principle Construction for Scientific Discovery

**arXiv ID:** 2610.32195 | [PDF](https://arxiv.org/pdf/2610.32195v1)

**作者:** Dongki Kim `[一作]` (KAIST), Sung Ju Hwang `[通讯]` (KAIST)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `09944146-298c-433e-89df-37255de463d7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个基于大语言模型的框架Principia，能够在测试阶段实时构建、更新科学原理并指导实验与预测。

**💡 创新点**

创新之处在于通过比较因果假设、归纳原理以及基于新观测的假设修订，实现了无须大量专家知识即可在测试时持续生成和改进科学原理，突破了传统方法的瓶颈。

**🔧 技术方法**

使用了GPT-5-mini等大语言模型进行假设生成、原理诱导、知识检索与修订，并结合图结构假设记忆、k‑medoids聚类以及LLM检索等技术。

**📊 数据集**

在分子优化、蛋白质工程、基因扰动以及分子属性预测等任务中使用了Scientific Discovery Evaluation benchmark、TDC毒性与CYP抑制任务等公开数据集。

**📈 对比分析**

与无引导优化器、经验摘要、外部检索、内部原理等基线进行AUC@10、最高分数、准确率/宏F1等指标比较，Principia在所有任务上均优于基线，并在实验预算上实现约55%–86%的评估次数减少。

**⚠️ 局限性**

主要局限在于高度依赖LLM的推理质量与检索准确性；假设生成仍受限于有限的比较样本，且在极端稀缺或非结构化数据场景下的适用性尚待验证。

---

## 382. Optimal Multiclass Online Regret with Arbitrary Label Spaces

**arXiv ID:** 2610.32339 | [PDF](https://arxiv.org/pdf/2610.32339v1)

**作者:** Sajad Ashkezari `[一作]` (University of Waterloo), Shai Ben-David `[通讯]` (University of Waterloo)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe`

**🎯 论文内容**

本文在任意（甚至无限）标签空间下，给出了多分类在线学习的最优后悔量上界，并证明其与二分类情形的最优 Θ(√(dT)) 后悔量完全一致。

**💡 创新点**

核心创新点是将二分类中消除对数项的分数覆盖（fractional cover）思想推广到多分类，构造了一种“一侧”分数覆盖，并与链式（chaining）技术相结合，从而实现了不依赖标签空间大小的 Θ(√(dT)) 后悔量上界。

**🔧 技术方法**

主要技术手段包括：
- 构造有限专家族并利用其一侧分数覆盖；
- 采用 Sion 定理的最小-最大双重性，将在线学习问题转化为有限专家游戏的极值；
- 在该极值框架下运用链式方法和分数覆盖实现对所有尺度的约束，从而消除对数因子；
- 通过马尔科夫分数覆盖与偏差控制，实现对专家与基准函数一致性的多尺度估计。

**📊 数据集**

该工作为理论论文，无需使用任何真实数据集；全部结论基于形式化的理论分析和证明。

**📈 对比分析**

与此前的上界 O(√(dT log(eT/d))) 以及已知下界 Ω(√(dT)) 相比，本研究实现了对数项的消除，达到匹配下界的最优性能；因此在任意标签空间下，后悔量从 O(√(dT log(⋅))) 下降到 O(√(dT))。

**⚠️ 局限性**

局限性包括：
- 仅适用于具有有限多分类 Littlestone 维数 d 的函数类；
- 证明中涉及复杂的分数覆盖与链式构造，常数项不易具体化；
- 对极端大 T 与 d 近似的细节未给出更精细的上界或下界。

---

## 383. When to Stop Penalizing Actions? Martingale GRPO for VLA Reinforcement Learning

**arXiv ID:** 2610.32431 | [PDF](https://arxiv.org/pdf/2610.32431v1)

**作者:** Zongyao Zhao `[一作]` (University of Hong Kong), Yuanwei Liu `[通讯]` (University of Hong Kong)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Martingale GRPO 方法，利用可视化错误与恢复事件定位失败反馈边界，改进 VLA 策略优化。

**💡 创新点**

创新点：从马氏过程推导理想停止规则；设计事件引导的反馈截断；使用冻结的视觉语言模型定位错误；证明截断保留期望并降低方差。

**🔧 技术方法**

技术：Martingale GRPO 公式；LoRA 微调 Qwen3.8 视觉语言模型；PPO+GRPO 损失；事件注释与可视化错误检测；GPU 大规模训练。

**📊 数据集**

数据集：7 任务 RoboTwin 2.0 机器人环境，包含 1,024 条失败视频带事件注释（训练/验证/测试分离）和 OpenVLA-OFT 任务。

**📈 对比分析**

比较方法：与全轨迹 GRPO 与 SFT 对比；在 7 任务中，MART 在第 40 次更新平均成功率提升 3.63%（如 Handover +8.98%），在 Hammer、Handover 上优于随机前缀和均匀缩放，分别高出 3.52% 与 3.12%。

**⚠️ 局限性**

局限性：依赖可视化错误与恢复事件的标注，截断边界基于已完成轨迹，在线推断成本较高；对策略演化后错误与剩余信用关系尚未完全理解，需进一步研究。

---

## 384. DS-BiCA-Net: Pair-Preserving Cross-Temporal Modeling for Remote Sensing Change Captioning

**arXiv ID:** 2610.32200 | [PDF](https://arxiv.org/pdf/2610.32200v1)

**作者:** Licheng Zhang `[一作]` (University of Melbourne), Siew-Kei Lam `[通讯]` (Nanyang Technological University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了DS‑BiCA‑Net，设计了一种保持双图像流不混合的编码器，分别通过窗口化选择性扫描（DBSS）建模单帧空间上下文，并用共享参数的双向交叉注意力（BiCA）进行双时序比较，最终用于遥感多时序图像的自然语言变更描述。

**💡 创新点**

创新点在于将内图像上下文建模与跨时序比较拆分为独立模块，保持双流可分离；首次通过可剔除的模块和方向实现对每种建模方式贡献的单独评估，并发现仅在BiCA之后加入DBSS才带来显著提升。

**🔧 技术方法**

采用共享ResNet101特征提取、窗口化选择性扫描状态空间模块、共享参数的双向交叉注意力、层归一化残差结构、聚合头与门控多尺度解码器；整个网络以预正则化残差和可学习的层尺度为核心。

**📊 数据集**

在LEVIR‑CC（5–14年卫星图像，10,077对）和Dubai‑CC（10年阿联酋城市图像，500对）两个公开变更描述基准上进行训练和测试。

**📈 对比分析**

与关注基、状态空间、扩散和视听语言四大范式下的十余种方法对比，DS‑BiCA‑Net在LEVIR‑CC上取得S_m^* 82.65、CIDEr 144.09的最高分；在Dubai‑CC（无几何增强）上获得S_m^* 66.72，排名第二；在加入几何增强的两种协议中进一步提升至70.14/72.13。

**⚠️ 局限性**

主要局限在于模型参数量（约59M）和显存需求较高；在极小变更检测和细粒度描述方面仍表现欠佳；并且共享参数的双向注意力虽能保持双流，但对不同时间方向的细微差异未能完全消除。

---

## 385. FLORA: An Open Model Suite for Scaling Robotics Data Engines

**arXiv ID:** 2610.31779 | [PDF](https://arxiv.org/pdf/2610.31779v1)

**作者:** Sicong Leng `[一作]`, Linxi "Jim" Fan `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研发并开源了一个工业级数据引擎FLORA，用于为机器人数据集生成时间戳式字幕、过程奖励和结果奖励。

**💡 创新点**

创新点在于整合分割、字幕和奖励三大模型为统一管道，支持多种奖励类型（结果、密集、用户定义Rubric），且在标注成本上比API模型低约125倍。

**🔧 技术方法**

技术主要基于Qwen3.5系列大型视觉语言模型（9B/27B），结合LoRA微调、GRPO强化学习、视频分割/字幕与奖励推理的端到端训练。

**📊 数据集**

训练数据包括公开数据集、商业授权数据集以及内部自行采集并人工标注的机器人/人类视频，覆盖多视角、并行抓取、精细手操作等场景。

**📈 对比分析**

在WGO‑Bench、DenseCaptionBench、RoboRewardBench、RobometerBench等基准上，FLORA在分割F1、联合F1、结果奖励MAE等指标均与GPT‑6 Astra相当或优于，同时标注成本显著更低。

**⚠️ 局限性**

限制包括对复杂多摄像头同步的依赖、密集奖励在误报率上略高、以及在部分任务上仍无法完全匹配最先进的专用奖励模型。

---

## 386. Fast and Useful: Adapting Distilled Generators for Visual Learning

**arXiv ID:** 2610.32512 | [PDF](https://arxiv.org/pdf/2610.32512v1)

**作者:** Patryk Bartkowiak `[一作]` (Adam Mickiewicz University), Wojciech Pałubicki `[通讯]` (Adam Mickiewicz University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种在已发布的蒸馏图像生成模型上仅修改第一步预测的低秩适配器，使得生成的图像在训练分类器时比原始模型产生的图像更具教学价值，同时保持快速生成速度。

**💡 创新点**

创新点在于：①使用低秩更新仅在第一步将学生模型的预测逼近无引导基模型的中间状态；②保留学生模型其余步骤，生成过程无需基模型；③通过对目标引导、跳跃终点和更新位置的系统性消融，验证中间状态教师目标对分类性能的关键作用。

**🔧 技术方法**

技术手段包括：扩散式文本到图像生成、模型蒸馏、低秩适配器训练、教师回归、引导调度对比、基模型与学生模型的对齐；使用的生成器有 FLUX.2-klein、SD-Turbo、Z-Image-Turbo。

**📊 数据集**

使用的数据集包括 ImageNet-100、ImageNet-200、ImageNet-1k（全规模）、以及细粒度数据集 Food‑101、Cars、Pets；实验中还用 COCO 目标描述作为 prompt。

**📈 对比分析**

对比方法：在相同的训练提示、种子和分类器训练流程下，分别用原始学生模型、基模型、基模型首步替代、公开 SDA 适配器以及 DP‑DMD 学生进行实验。评估指标为 ResNet‑50 在真实图像上的 top‑1 准确率；实验显示适配器在三大生成器上平均提升 2–5% 的分类准确率，同时生成时延与原始学生相当或更低；在 1K 级别的完整实验中，适配器对全规模 ImageNet 的提升可达 3–4%。

**⚠️ 局限性**

局限性包括：①需要对应的基模型来构造教师目标；②跳跃终点的选择需基于实验评估，缺乏通用准则；③实验主要关注分类任务，对检测、分割等任务未验证；④适配器继承了生成器的偏差与覆盖范围，未对人群或罕见场景的平衡性进行评估。

---

## 387. National-Scale Multimodal Accessibility Fields from Open Geospatial Data

**arXiv ID:** 2610.32479 | [PDF](https://arxiv.org/pdf/2610.32479v1)

**作者:** Till Zacher `[一作]` (Technical University of Munich), Markus Lienkamp `[通讯]` (Technical University of Munich)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建了可复现的MAPPER工作流，利用开放数据生成全国尺度多模式可达性连续字段，并通过H3格网输出。

**💡 创新点**

创新点在于结合低开销本地计算核心、网络核密度估计与统一的H3输出，实现在商用硬件上国家级多模式连续可达性堆叠。

**🔧 技术方法**

技术包括开源的OSM和GTFS图构建、路由器合并的收缩层次算法、C++/OpenMP并行的网络核密度估计、DAG调度与H3采样。

**📊 数据集**

数据集为德国的OpenStreetMap、完整GTFS（Delphi e.V. NeTEx）、OSM POI与工作场所代理、德国2022年人口普查，以及Bavaria案例使用的LEP规划数据。

**📈 对比分析**

通过与手机旅行日记实测面板进行Bland-Altman和误差聚合评估，行走/骑行/汽车的平均偏差接近零，公交存在偏低估计；整体计算在商用工作站上完成39.99万H3细胞、972字段的66GB数据，运行时间数小时。

**⚠️ 局限性**

局限包括：公交时刻表基准的理想化估计、未考虑拥堵与真实等待、工作场所代理的粗略比例、对核密度带宽和阈值的依赖以及对POI标签噪声的依赖。

---

## 388. CoCarry: Contact-Support-Structured Retargeting for Dual-Humanoid Cooperative Carrying

**arXiv ID:** 2610.32460 | [PDF](https://arxiv.org/pdf/2610.32460v1)

**作者:** Ziyu Lin `[一作]` (Tongji University), Guang Chen `[通讯]` (Tongji University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出CoCarry数据集并开发一种针对双机器人协同携带任务的联合运动再定位方法

**💡 创新点**

创新性地将手-物体锚点与上层对象交互网格、下层支撑网格结合，形成可同时维护共享抓握、相对姿态和支撑关系的结构化优化框架

**🔧 技术方法**

采用GMR作为初始运动生成器，随后通过手锚点约束、上层交互网格与下层支撑网格以及全局双机器人约束的联合优化技术实现再定位

**📊 数据集**

使用自建的220条同步双人携带序列CoCarry数据集，并在外部对照集CORE4D中的Unitree G1机器人进行跨数据集评估

**📈 对比分析**

与传统的GMR和OmniRetarget方法对比，取得接触保持率0.9995、渗透时长降低67%及71%并保持100%成功率，证明了该方法在多场景、多地形、多物体几何条件下的优越性

**⚠️ 局限性**

局限在于手势仅通过空间接近与方向约束，未考虑力学摩擦和动力学交互；且评估范围主要集中在携带任务，对更复杂的协同操作尚需进一步验证

---

## 389. MDMR-Bench: A Multi-Dimensional Benchmark for Multi-Reference Image Generation

**arXiv ID:** 2610.32459 | [PDF](https://arxiv.org/pdf/2610.32459v1)

**作者:** Bowen Zheng `[一作]` (Nanjing University), Xihui Liu `[通讯]` (University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了MDMR-Bench，一个包含2390个案例的多维多参考图像生成基准，覆盖实体融合、属性迁移、多元素提取、信息图表操作和视觉推理等多种任务。

**💡 创新点**

创新点在于①通过自动化图像库构建和任务桶填充实现大规模、多样化案例；②引入全局与原子级的检查表式评估，细粒度诊断模型性能；③系统性对比封闭式与开源式生成模型，揭示多参考生成的真实挑战。

**🔧 技术方法**

核心技术包括：大规模图像采集与合成（LAION、Nano Banana Pro、GPT‑Image‑1.5）；基于Gemini‑3‑Pro的自动化检查表生成与判定；多维度评估框架（Following、Consistency、Composition、Minimality、Quality）。

**📊 数据集**

使用的数据集来源于公开的LAION图像以及由Nano Banana Pro和GPT‑Image‑1.5生成的高质量合成图像，构成MDMR‑Bench。

**📈 对比分析**

评估方法采用VLM判定的二元检查表，并计算五个维度的得分；实验结果显示GPT‑Image‑2首位（Overall ≈ 88.8），后续为Nano Banana Pro等；开放源模型最高得分仅51.3，说明封闭式模型在多参考生成上仍显著领先。

**⚠️ 局限性**

局限性包括：基准仍受图像来源质量和人工验证流程限制；检查表依赖VLM判定，可能受模型偏差影响；在信息图表和推理任务上模型性能仍偏低，亟待进一步改进。

---

## 390. On Task, Off Track: Benchmarking LLM and Agent Robustness to Near-Target Prompt Injection

**arXiv ID:** 2610.32531 | [PDF](https://arxiv.org/pdf/2610.32531v1)

**作者:** Yuyang Gong `[一作]` (Nanyang Technological University), XiaoFeng Wang `[通讯]` (Nanyang Technological University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 NTBench，利用合法任务的参考答案生成与之相近但包含错误或有害偏差的攻击目标，评估 LLM 与代理在近目标提示注入下的鲁棒性。

**💡 创新点**

创新点在于把参考答案作为生成近目标攻击的先导，并结合目标合法性验证，扩展了传统固定目标评测未能揭示的错误/有害漏洞。

**🔧 技术方法**

采用 LLM 生成器（如 GPT-5.6 Luna）生成目标–注入指令对，LLM 判断和人工验证保证目标合法性，随后在模型 API 上执行注入并用判定器评估成功率。

**📊 数据集**

实验数据集包括 10 个 PIArena 语言任务（如 Dolly、HotpotQA、SQuAD 等）和 2 个代理工作流基准（AgentDojo 与 AgentDyn）。

**📈 对比分析**

方法在相同的攻击执行预算（每样本 3 次）下，对比固定目标、近目标错误与近目标有害三类攻击；结果显示近目标攻击成功率显著高于固定目标，提升幅度在 9.3–35.8%（错误）和 1.7–17.3%（有害）。

**⚠️ 局限性**

局限性包括仅考虑加法式注入、未对攻击执行预算或 ϵ 半径进行系统化调优、仅评估单一模型输入结构，且对更复杂的动态或多轮攻击场景尚未覆盖。

---

## 391. Facultative Swarming: A Capability, Not a Category, for Multi-Robot Systems

**arXiv ID:** 2610.32526 | [PDF](https://arxiv.org/pdf/2610.32526v1)

**作者:** Ning Zhou `[一作]` (University of Bristol), Edmund R. Hunt `[通讯]` (University of Bristol)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种“facultative swarming”范式，将集中式与分散式协作视为可切换的能力，并设计了触发器分类与成本比较切换逻辑；随后通过对CT‑TAPF仿真任务进行三层模式（最优集中、启发式集中、去中心化）切换实验，验证了在计算负载突增时系统能稳定、无抖动地实现模式转换。

**💡 创新点**

创新点在于：①将 swarm 视为可选的功能而非固有属性；②提出了三类触发器（资源匮乏、外部威胁、结构破坏）与对应的切换策略；③构建了基于成本比较的动态切换框架，并与自组织层次结构（如 SoNS）相结合；④首次在仿真中演示了从最优集中到分散的完整降级与恢复流程。

**🔧 技术方法**

采用的技术主要包括：集中式全局规划（CT‑TCBS）、启发式集中规划（Best‑Task/Worst‑Task）、去中心化自组织规则；切换逻辑基于成本比较与Schmitt触发器；动态算法配置（DAC）思想用于在线决策；仿真平台利用Python/ROS与自定义CT‑TAPF求解器。

**📊 数据集**

使用的数据集为CT‑TAPF求解器的基准实例（由原始论文提供），并在此基础上生成符合不同模式运行时与成本分布的仿真样本；没有使用真实场景的感知或地图数据。

**📈 对比分析**

比较方法是将三种模式在相同任务集与不同机器人数量下的运行时和成本进行统计，利用阈值触发的切换策略保证系统始终在最优或次优可行模式下运行；实验结果显示，在负载步长或脉冲变化时系统能无抖动地从最优集中切换至启发式或分散模式，恢复后也同样稳定；在规模扩展时，系统能按预设阈值顺序降级，维持较低的成本和可接受的完成率。

**⚠️ 局限性**

局限性包括：①触发器权重与成本尺度的统一与校准仍是开放问题；②多代理切换时的窗口脆弱性和状态同步成本难以量化；③缺乏正式验证与安全保证，尤其是切换过程的合规性；④仅在计算负载触发下验证，其他资源匮乏、威胁与结构破坏情形缺乏实证；⑤人机交互与行为异质性恢复机制尚未实现。

---

## 392. VERA-Hand: A Verified Multidimensional Benchmark for 2D Hand Pose Estimation in Third-Person Dynamic Gesture Videos

**arXiv ID:** 2610.32517 | [PDF](https://arxiv.org/pdf/2610.32517v1)

**作者:** Gibran Benitez-Garcia `[一作]` (University of Electro-Communications), Hiroki Takahashi `[通讯]` (University of Electro-Communications)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了VERA-Hand，一个手部姿态估计的多维度评测基准，包含检测、定位、身份识别和时间稳定性四个维度，并引入Pose‑F1联合指标；

**💡 创新点**

创新点在于：①首次提供完整人手3D动态视频的手工验证2D标注；②多维度评测能够揭示不同估计器在各个维度的性能差异；③通过基准发现并验证特定失效模式（如交叉手重叠）及其纠正效果；

**🔧 技术方法**

采用手工验证标注、框架统一的评价协议、Pose‑F1指标、时序抖动（J_acc、J_jerk）和运动能量指标；

**📊 数据集**

基准数据来自IPN Hand视频集，包含13位受试者的52段手势视频，共10,768帧、10,690个手部实例；

**📈 对比分析**

对六种公开估计器（MP‑hands、RTMP‑hand、MP‑holistic、RTMW‑wb、WiLoR、Sapiens2‑1B）在四个维度上进行比较，结果显示不同估计器在检测、定位、身份识别和时间稳定性上的排名大相径庭，未出现单一统治者；

**⚠️ 局限性**

局限性包括：仅覆盖第三人称动态手势场景；手部标注以右手为主；评估仅为完整公开管线，未拆解检测器与姿态头的贡献；标注初始来源于MP‑holistic，可能存在残留偏差；

---

## 393. Amortized Thompson Sampling and Efficient Information-Directed Sampling for Simulator-Based Design

**arXiv ID:** 2610.32514 | [PDF](https://arxiv.org/pdf/2610.32514v1)

**作者:** Michal Kobiela `[一作]` (University of Edinburgh), Michael U. Gutmann `[通讯]` (University of Edinburgh)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种离线预训练的 Thompson 采样加速方法，并利用信息增益与信息导向采样来改进实验设计。

**💡 创新点**

创新地将参数到动作的映射和动作流的条件流结合，实现在无似然、无梯度的模拟器中前置量化 Thompson 采样，并在决策中引入信息增益与一跳前瞻/信息导向采样。

**🔧 技术方法**

使用 PPO 强化学习训练参数化策略，使用条件正则化流（normalizing flow）近似后验和动作分布，并通过蒙特卡罗估计信息增益、即时回报及一跳前瞻回报。

**📊 数据集**

在两套合成生物学仿真案例上进行评估：宿主感知异源基因表达（22 条 ODE）和抑制环（离散随机模拟），仅使用仿真数据而非真实实验数据。

**📈 对比分析**

与传统 Thompson 采样、即时回报、信息增益、IDS 等多种策略在 1000 次轨迹模拟中比较，结果显示信息导向采样在累计回报上优于标准 Thompson，而信息增益和一跳前瞻在最终回报上优于其余方法。

**⚠️ 局限性**

存在参数流与动作流训练独立导致一致性缺失；信息增益/一跳前瞻在早期实验中可能产生高即时回报损失；方法对高维动作空间的可扩展性尚未验证。

---

## 394. Augmentation Strategy for DINO in Glaucoma Classification

**arXiv ID:** 2610.32511 | [PDF](https://arxiv.org/pdf/2610.32511v1)

**作者:** Ashish Kumar Meena `[一作]` (Indian Institute of Science), Chandra Sekhar Seelamantula `[通讯]` (Indian Institute of Science)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a6cb313d-240c-4723-a372-3ba1f39b9afc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了 DINO 在眼底图像中的增广策略，并提出针对青光眼诊断的任务感知增广方案。

**💡 创新点**

创新点在于通过 InfoMin 原理设计教师-学生不对称的裁剪分配，将诊断关键的视盘区域作为教师视图，其他多尺度视角作为学生视图。

**🔧 技术方法**

使用了 ViT‑Small 作为 backbone 的 DINO 框架，并结合图像增广、血管去除、EMA 训练等技术。

**📊 数据集**

训练数据为 81k 眼底图像，来源于六个公开数据集（REFUGE、ACRIMA、LAG、ORIGA、FIVES、PAPILA），并在七个青光眼基准集上评估。

**📈 对比分析**

与 Vanilla DINO、RETFound‑DINOv2 以及 RetiZero、FLAIR 等基线比较，线性探测在低样本和大样本两种 regime 下均取得平均 AUROC 最高，尤其在跨域泛化上优于 CLIP 视觉语言模型。

**⚠️ 局限性**

限制在于任务裁剪仅针对视盘相关区域，未覆盖分布式病变或其他视网膜疾病；同时在低样本 REFUGE 上仍不及大规模监督预训练模型。

---

## 395. Fast and Scalable Influence Estimation in the Independent Cascade Model: a unified framework

**arXiv ID:** 2610.32502 | [PDF](https://arxiv.org/pdf/2610.32502v1)

**作者:** Matic Požar `[一作]` (University of Primorska), Miklós Krész `[通讯]` (University of Primorska)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文系统研究并比较了独立级联（Independent Cascade）模型的多种解析式近似方法，提出了新的加速连续（continuation）策略、独立传播层（IPL）以及二阶动态消息传递（DMP_2）等方法，并在大规模真实与合成网络上进行全面评估。

**💡 创新点**

创新点包括：①统一的解析框架揭示不同近似方法的三大维度（传播状态、历史信息、聚合规则）；②引入补偿连续技术通过组合不同近似的误差方向来提升精度；③提出二阶动态消息传递，保留最近两步的回路信息，形成递增上界序列；④开发独立传播层和二阶传播聚合，修正层间重叠误差。

**🔧 技术方法**

使用的技术包括：贝叶斯传播（BP）与动态消息传递（DMP）框架、无向边缘化（live-edge）模型、线性与非线性噪声-OR聚合、矩阵因式分解、稀疏图算法以及多层迭代与加权求和。实现中大量利用稀疏矩阵运算、日志-指数变换与自适应截断。

**📊 数据集**

数据集包含七个真实网络（NetHEPT、WikiVote、Enron、Epinions、Slashdot、ca-CondMat、cit-HepTh）和五类合成网络（Erdős–Rényi、Barabási–Albert、Holme–Kim、Stochastic Block Model、Watts–Strogatz），并在多种边概率模型、种子集合大小与扩散强度下生成实验。

**📈 对比分析**

对所有方法在节点级激活概率上的误差（RMSE、MAE）和运行时间进行了交叉族比较。二阶 DMP（DMP_2）在所有数据集上获得最低的聚合 RMSE，并在每个网络族中常常是最优；SWE‑NO 与其 cavity 变体也表现优异；线性方法（ALE、IPL）速度最快但精度最低。实验展示了不同方法在密度、聚类、扩散强度等维度上的性能差异。

**⚠️ 局限性**

局限性包括：①仅针对 IC 模型，未验证在其他传播模型上的适用性；②高阶方法（DMP_2）虽然更精确但计算成本显著增加；③实验使用 Monte Carlo 作为参考，仍存在采样误差；④参数调优需在特定设置上校准，缺乏通用自动化选择机制；⑤报告误差包含种子节点，可能低估实际难度。

---

## 396. MusicTwins: Training Audio-Language Models to Listen like Their Persona

**arXiv ID:** 2610.32495 | [PDF](https://arxiv.org/pdf/2610.32495v1)

**作者:** Milan Liessens Dujardin `[一作]` (Bryel Labs, UC Berkeley), Kevin Miao `[通讯]` (Bryel Labs)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a2602d71-93ab-4bad-974b-672788df8193` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建 MusicTwins 框架，结合人物摘要、个人记忆与后训练，生成针对文本与音乐刺激的人格化响应；通过对 Audio Flamingo Next 进行监督微调和强化学习，提升其对个体音乐反应的模拟能力。

**💡 创新点**

首次将音频‑语言模型与完整人格表示（背景、观点、人格、写作风格及音乐感知/情感/偏好/行为等维度）相结合，并设计四个训练目标（回忆、插值、外推、反幻）实现精准的个体化回答；同时提供跨音乐与文本的评价协议。

**🔧 技术方法**

核心技术包括 Audio Flamingo Next（低秩适配器微调）、RAGFlow 记忆检索、CLAP 音频‑文本嵌入、Group Relative Policy Optimization（GRPO）强化学习；训练时使用 BM25 + dense 文本检索与 CLAP 音频相似度检索。

**📊 数据集**

数据集：从 Reddit Language–Genre 选取 85 位作者，提取最多 500 篇写作；将其音乐身份映射至公开古典音乐录音（Wikimedia Commons、IMSLP）；生成 1,276 条 QA–音频对；此外使用 MuChoMusic、HumMusQA 与 MMAR 评估通用音乐理解。

**📈 对比分析**

对比基线 Audio Flamingo Next，MusicTwins 在 10 位留置人物的 600 条 QA 组上提升整体五维分数 73.8→79.4（+5.7，95% CI [3.8, 8.2]），音乐响应子分数提升 45.5→56.7（+11.2，95% CI [8.1, 15.5]）。最显著提升来自插值（+14.3）、外推（+9.2）和反幻（+18.0）。在验证阶段，三阶段训练分别使回忆、插值、外推与反幻在 0–100 评分上显著上升，最终 GRPO 阶段实现 67.9 的平均得分；音乐理解基准略有提升。

**⚠️ 局限性**

局限性包括：人格被视为静态，忽略时间变动；使用 Reddit 文本与推断的人格评分可能带来偏差；模型仅训练于古典音乐（30 s 录音），不覆盖流行或实验音乐；仅输出文本，无法生成音乐或多模态交互；对大规模多样化人群与音乐风格的泛化仍待验证。

---

## 397. K-CLUE: A Segmentation-to-Clustering Framework for Ki67 Cell Typing

**arXiv ID:** 2610.32493 | [PDF](https://arxiv.org/pdf/2610.32493v1)

**作者:** Ana Beatriz Vieira `[一作]` (Instituto Superior Técnico, Universidade de Lisboa), Arlindo L. Oliveira `[通讯]` (Instituto Superior Técnico, Universidade de Lisboa)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在不需要任何标注或监督训练的前提下，提出K‑CLUE框架，用预训练视觉模型提取核嵌入，经过PCA白化后用k‑means对Ki67阳性与阴性细胞进行聚类，实现IHC图像中Ki67细胞类型的自动判别。

**💡 创新点**

创新点包括：①利用冻结的预训练编码器获取通用特征，而非针对IHC专门训练；②引入PCA白化消除嵌入空间方差结构对聚类的依赖，使不同编码器性能趋于一致；③实现完全无监督、无任务特定调参的Ki67细胞分型，省去人工标注与域适配。

**🔧 技术方法**

技术手段：Cellpose‑DINOv3细胞分割 → 预训练编码器（ResNet, EfficientNet, ConvNeXt, DINOv3, 病理特定编码器） → PCA降维保留90%方差 → 对保留主成分进行白化 → k‑means（k=2）聚类 → 用DAB平均强度判定阳阴簇。

**📊 数据集**

使用公开的MIST（多IHC染色翻译）和BCData（乳腺肿瘤细胞数据集）两个数据集进行实验和评估。

**📈 对比分析**

与DeepLIIF、IHCScoreGAN以及手工设计的颜色特征（HED+HSV+Lab）做对比。K‑CLUE在MIST上Ki67阳性/阴性F1分别为85.00%/81.91%，在BCData上为79.59%/81.16%，与手工颜色基线相差不超过1个百分点，优于DeepLIIF和IHCScoreGAN，且不需任何域适配或标注。

**⚠️ 局限性**

局限性：性能强烈依赖上游细胞分割质量；在BCData等不同扫描条件下表现略低；目前仅区分Ki67阳阴，未区分肿瘤细胞与淋巴细胞等其他细胞类型。

---

## 398. Accordion: Elastic-Depth Inference via Fixed-Prefix Layer Looping

**arXiv ID:** 2610.32492 | [PDF](https://arxiv.org/pdf/2610.32492v1)

**作者:** Yudong Zhang `[一作]` (Tsinghua University), Zhenyu Zhang `[通讯]` (Baidu Inc)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并训练了可调深度混合专家Transformer模型 Accordion，支持五个固定推理预算，并通过联合预训练、SFT和RL实现多领域任务。

**💡 创新点**

创新点在于采用固定前缀层循环的混合专家架构，使同一套权重支持多层深度推理；通过在预训练期间采用单路径微批采样训练，避免浅路径性能下降，并通过预算匹配的多教师蒸馏整合多领域专家。

**🔧 技术方法**

使用了 Looped Transformer 层循环、Mixture-of-Experts (MoE)、零初始化残差门、统一微批预算采样、参数组合并、预算匹配的多教师 on‑policy 蒸馏以及 RL 进行指令跟随与任务专精。

**📊 数据集**

使用了公共 Web 文本（Nemotron‑CC v2）、数学（MegaMath、FineMath、InfiMM‑WebMath）、代码（StarCoder）等数据集，累计约 3.6 T 训练标记，后续再加 50 B 进行长上下文扩展。

**📈 对比分析**

在 37 项通用基准上，Accordion‑Base 在五个预算下均优于四个公开参考模型（Qwen、OLMoE、Ouro），最大提升 3.5%（数学）和 4.8%（代码）；每增加深度可提升 3.01% 宏平均，且中间预算 k=6 获得 89% 的收益，长上下文时仍保持深度收益。

**⚠️ 局限性**

局限性包括：仍需在所有预算下共享同一参数，无法实现输入自适应深度；单路径预算采样在极端预算下可能略低；对极端长上下文或特殊任务的可扩展性有限，且复现依赖大规模硬件与公开数据。

---

## 399. From Perception to Pixels: Human-Derived Visually Distinct Colors for Improved Dominant Color Extraction

**arXiv ID:** 2610.32473 | [PDF](https://arxiv.org/pdf/2610.32473v1)

**作者:** Uldana Shyndali `[一作]` (Kazakh-British Technical University), Pakizar Shamoi `[通讯]` (Kazakh-British Technical University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `da1b1a89-583a-4b57-9c81-478778569bec` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

通过构建基于人类感知的“人类颜色词典”（Human Color Dictionary，HCD），并将其嵌入传统的 K‑means 聚类与颜色直方图算法中，实现了更符合人眼感知的主导色提取。

**💡 创新点**

创新点在于①利用大规模感知实验生成 64 色的 HCD；②在聚类前将随机中心映射到 HCD，并通过 CIEDE2000 进行去重与补全；③对结果进行感知映射后重新提取直方图，显著提升与专家色板的匹配度。

**🔧 技术方法**

主要技术包括：①人类感知实验与共现矩阵+贪心选择得到 HCD；②在 Lab 空间使用 CIEDE2000 计算颜色差异；④改进的 K‑means（初始化中心映射、去重补全）与 RGB 直方图+映射；⑤评价指标为 Earth Mover’s Distance（EMD）和最佳匹配（Best‑Match）距离。

**📊 数据集**

使用 4,562 张带有专家设计的 5 色色板的图像集（来自设计师网站），以及 120 色的感知实验样本集。

**📈 对比分析**

与基线方法（RGB/Lab K‑means、K‑means++、颜色直方图）对比，加入 HCD 后 Best‑Match 提升约 6 分（从 64.83% 提升到 70.78%），EMD 亦略有提升；Lab 空间的聚类总体表现优于 RGB。实验显示改进后的方法在与专家色板的匹配度上始终领先。

**⚠️ 局限性**

局限性包括：①实验样本与色板主要来自同一文化背景（设计师网站），未涵盖跨文化或非设计领域的色彩需求；②受试者主要为 18–25 岁年轻人，性别/年龄差异不大；③仅验证了 5 色的提取，对更多色数或不同图像内容的泛化需进一步研究；④对极暗/深色调的区分仍存在一定误差。

---

## 400. Online Continual Learning in Edge-Cloud Applications

**arXiv ID:** 2610.32472 | [PDF](https://arxiv.org/pdf/2610.32472v1)

**作者:** Charalampos Davalas `[一作]` (Harokopio University of Athens), Konstantinos Tserpes `[通讯]` (National Technical University of Athens)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

提出一种边缘‑云协同的在线持续学习框架，将概念漂移检测（ECDD）和离群检测（ODIN）与回放式持续学习结合，利用冲突检测触发模型更新；

**💡 创新点**

将漂移与OOD检测作为样本筛选与更新触发器，提出基于冲突的样本缓冲与动态更新策略，避免持续训练导致的灾难性遗忘；

**🔧 技术方法**

采用ODIN（温度缩放+梯度扰动）和ECDD（EWMA+伯努利漂移检测）作为检测器，回放式训练（rehearsal）与数据扩充（ANDE）；

**📊 数据集**

在交通标志（自定义Belgium Traffic Signs）、nuScenes对象、疟疾细胞图像和PlantVillage叶片图像等公开/自制数据集上进行实验；

**📈 对比分析**

与基线TRN、随机缓冲、仅回放等组合比较，实验显示ODIN+ANDE或RH+ODIN/ ECDD组合在工厂与流数据准确率上均有显著提升，平均提升约2–5%，且方差下降；

**⚠️ 局限性**

仍需人工标注，样本存储受限；ECDD需同时获取用户输入，ODIN计算开销较大；在模型规模、类别不平衡、实时性要求极高的场景下效果尚未充分验证。

---

## 401. Agentic Memory Management Is the Next Frontier of Information Retrieval

**arXiv ID:** 2610.32464 | [PDF](https://arxiv.org/pdf/2610.32464v1)

**作者:** Qingyao Ai `[一作]` `[通讯]` (Tsinghua University), Qingyao Ai (Tsinghua University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了面向AI代理的记忆管理（AMM）框架，将IR的结构、分析、组织、存储、检索、使用等任务迁移到代理内部，并给出对象、过程、应用和优化四个维度的系统级视图。

**💡 创新点**

创新点在于将IR视为代理记忆管理的新领域，提出了常量、函数、算子三层记忆对象分类，以及对应的检索语义和更新动态，并制定了面向AMM的研究议程和评测规范。

**🔧 技术方法**

使用了IR传统技术（检索、排序、聚类、点击模型等）以及机器学习中的非参数更新（文本梯度）和参数化优化，融合了记忆写入、遗忘、利用等操作。

**📊 数据集**

论文未给出具体数据集，而是基于现有的对话、代码、网页导航等代理任务场景，建议构建包含常量、函数、算子以及反馈流的基准集合。

**📈 对比分析**

评价方法提出了从状态质量、利用质量、任务结果和成本四维度的指标，强调可复现性和多代理、多用户的可比性，但尚未给出实验性能数字。

**⚠️ 局限性**

局限在于缺乏实证验证和统一基准，记忆对象与检索语义的匹配仍处于理论探讨阶段，跨模型泛化和安全隐私等治理问题待解决。

---

## 402. When Agents Fail: Model or Harness--Who's to Blame?

**arXiv ID:** 2610.32483 | [PDF](https://arxiv.org/pdf/2610.32483v1)

**作者:** Shao Zhang `[一作]` (Shanghai Artificial Intelligence Laboratory), Shuyue Hu `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了大语言模型与执行层（harness）之间的交互，提出了新的评估指标（VCR、Model–Harness Fit、Cross-Model Fit 等），并在六个闭环任务基准上系统比较了四种 harness 与四个模型的表现。

**💡 创新点**

创新点在于：①把任务完成度与操作有效性（valid completion）分离开来，揭示两者可能不一致；②引入交互残差和资源使用度量，量化 harness 对模型的“使用成本”；③提出“跨模型 Fit”来评估 harness 在不同模型间的可迁移性；④使用“交叉比较”方法消除任务固定导致的排名偏差。

**🔧 技术方法**

主要技术包括：①LLM Agent 交互框架（Observe、Context、Control、Act、Update、Govern 六大接口）；②基于 bootstrap 的统计置信区间；③任务性能与 VCR 的联合指标（Fit）以及多维度资源度量（Token Load、Tokens per Success 等）；④对失败阶段的细粒度分解（pre‑normal loss、post‑normal handoff loss）。

**📊 数据集**

使用了六个公开基准：SWE‑bench‑45、Terminal‑Bench 2、VitaBench、LOCA‑Bench、ClawMark 与 AgentSkillOS；在每个基准下测试了四个通用 harness（DeepAgents、OpenCode、Pi CLI、Codex CLI）和四个 LLM（DeepSeek V4 Flash、MiniMax M3、Kimi K2.7、Qwen3.6‑35B‑A3B），以及两套固定 native 系统。

**📈 对比分析**

比较方法：通过“胜率”（Outcome Win Rate、VCR Win Rate）和“Fit”指标对每种 harness 进行排序；利用交叉比较保持任务与模型固定的同时变更 harness 或模型；在所有配置上计算资源使用（Token Load、Tokens per Success）并评估 Pareto 前沿。结果显示：OpenCode 通常在 Cross-Model Fit 上排名第一；不同模型对同一 harness 的偏好不同，且 VCR 与任务分数并不总是同步，说明执行有效性与任务成功并非单一维度；资源指标揭示了在高 VCR 的 harness 也可能伴随高 token 消耗或失败尾部。

**⚠️ 局限性**

局限性：①评估仅覆盖已公开的六个基准，可能无法全面代表所有类型任务；②固定的模型–harness 组合未包含自定义或更小型模型，影响泛化性；③资源度量以 token 计数为主，忽略了计算时间、能耗等实际部署成本；④交叉比较在一定程度上受任务难度差异影响，可能需要更细粒度的任务划分；⑤未对 harness 的自适应训练进行实验，缺乏对训练过程的直接验证。

---

## 403. WaitLens: Temporal VLM-Based Adaptive Waiting for Industrial GUI Testing Agents

**arXiv ID:** 2610.32482 | [PDF](https://arxiv.org/pdf/2610.32482v1)

**作者:** Shika Ikeura `[一作]` (Fudan University), Yangfan Zhou `[通讯]` (Fudan University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出基于视觉语言模型的自适应等待策略，替代工业 GUI 测试代理中的固定等待，显著减少不必要的空闲等待时间。

**💡 创新点**

①采用滑动窗口限制历史截图并构造时序视觉输入；②利用时间重采样打破历史长度与标签相关性，增强模型泛化；③仅凭截图在不依赖应用内部信息的情况下训练 VLM 预测 GUI 是否已就绪。

**🔧 技术方法**

轻量级 VLM Qwen3‑VL‑2B‑Instruct 微调（LoRA），时序视觉编码，时间重采样数据增强，滑动窗口历史采样以及多步连续预测阈值。

**📊 数据集**

Meituan 真实工业回放数据：102 条标注录制用于 RQ1（12 个测试任务），1101 条用于 RQ2（3,868 个响应序列），包含内部屏幕录制与外部摄像机录制，并标注“Ready”时刻。

**📈 对比分析**

与预训练 VLM、Qwen3.8‑Max、AdaT 等基线进行二分类比较；与固定等待（1–20 s）和 AdaT 进行等待策略对比。模型准确率 86.08%（基线 54.13%），FPR 24.59%；推理延迟 0.243 s；等待策略平均实际等待 2.222 s，额外等待 0.901 s，较 20 s 固定等待低 95%（PRR 15.64%），远优于 AdaT 的 PRR 90.8%。

**⚠️ 局限性**

仅凭视觉难以区分部分渲染与完整渲染导致误判；实时采样与模型调用成本可能抵消短等待收益；缺少对动作目标和测试目标的上下文；在极快响应场景下改进空间有限。

---

## 404. The Punjabi Geet Toxicity Dataset (PGT-FG): Fine-Grained Multimodal Annotations for Toxic Content in Punjabi Songs

**arXiv ID:** 2610.32513 | [PDF](https://arxiv.org/pdf/2610.32513v1)

**作者:** Lakhjeet Kaur `[一作]` (Guru Gobind Singh Indraprastha University), Pravin Kumar `[通讯]` (UPES)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并发布了Punjabi Geet Toxicity Dataset (PGT‑FG)，包含 397 首 Punjabi 歌曲的 2,396 个 30 秒片段的多模态（文本、音频、视频）毒性标注。

**💡 创新点**

创新点在于首次提供细粒度毒性类别、严重程度、情感、表达方式以及不确定性标注，并针对 Punjabi 音乐场景进行专门的多模态数据集构建。

**🔧 技术方法**

技术上采用自动 YouTube 下载与 FFmpeg 分段、人工注释结合多标签标注方案，使用 Fleiss Kappa 评估注释一致性，最后通过多数投票生成单一标签版本。

**📊 数据集**

使用的数据集为自建的 PGT‑FG 本身，涵盖 2,396 个片段，来自 397 首 Punjabi 歌曲，并提供相应的 YouTube 源链接与时间戳。

**📈 对比分析**

本工作未进行模型训练或基线性能评估；提供的仅是标注资源，后续可用于构建文本、音频或多模态毒性检测模型进行实验。

**⚠️ 局限性**

局限包括样本量有限、细粒度类别分布极度不平衡（如种姓歧视极少）、源视频可能随时失效、注释者主观性导致部分维度一致性低、且仅覆盖 Punjabi–English 代码混合和罗马字转写。

---

## 405. Deterministic Root-Finding over Finite Fields with Elliptic Curves

**arXiv ID:** 2610.32508 | [PDF](https://arxiv.org/pdf/2610.32508v1)

**作者:** Pierre-Jean Spaenlehauer `[一作]` `[通讯]` (Université de Lorraine), Pierre-Jean Spaenlehauer (Université de Lorraine)

**关键词:** `847a60d8-a755-47af-ba5d-c5236b9e3083` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种将有限域根求解问题转化为椭圆曲线点运算的两阶段算法，先离线预处理得到椭圆预处理数据𝒟，再在线确定性求解任意分裂可分多项式的根。

**💡 创新点**

创新点在于：①用椭圆曲线的有理点群（包括二次变形曲线）替代传统的乘法循环群，从而将算法复杂度与q-1的平滑性无关；②提出可验证、无条件的预处理数据结构，并给出存在性猜想及实验支持；③给出完整的算法实现与复杂度分析，在线阶段完全在Fq上执行。

**🔧 技术方法**

技术包括：椭圆曲线群论与二次扭曲、分数分解与Tonelli–Shanks推导、分离多项式根的2-adic递归、分割多项式到多项式上的“投射”与分割；利用除法多项式、特征多项式、线性插值、乘法映射的GCD与翻译；以及离线阶段的曲线构造、点计数与基底求取。

**📊 数据集**

实验使用 Magma 进行实现，测试了多种 q（包括大于2^63 的素数域）和多项式的根求解，验证了算法在一般 q 下的可行性。

**📈 对比分析**

与传统的基于q-1平滑性的 Cyclotomic 方法相比，本算法在 q-1 非平滑时仍能保持多项式时间；在线复杂度为 O(Ξ(𝒟)^2 n log^2 q + n^ω log q + n log^3 q)，其中 Ξ(𝒟) 为曲线与其扭曲点群阶乘的最大质因子；若两点群均循环，可进一步降到 O(Ξ(𝒟) n log^2 q + n^ω log q + n log^3 q)。

**⚠️ 局限性**

主要限制是预处理阶段的曲线构造：目前仅能在假设下构造具有小 Ξ(𝒟) 的曲线，且缺乏多项式时间的构造算法；离线阶段对大 q 的实现成本仍较高；在实际应用中需对 Ξ(𝒟) 的可行性做进一步研究。

---

## 406. CASE: Confusion-Aware Skill Evolution for Frozen Multimodal Emotion Models

**arXiv ID:** 2610.32481 | [PDF](https://arxiv.org/pdf/2610.32481v1)

**作者:** Siyu Wu `[一作]` (Jilin University), Hongxia Xie `[通讯]` (Jilin University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 CASE 框架，通过外部可演化的情绪技能库，对冻结的多模态大语言模型进行局部纠错，以解决情绪细粒度识别中的相邻情绪混淆。

**💡 创新点**

创新点在于：①将情绪识别视为多对局部判别任务；②使用可检索、可演化的 pair‑specific EmotionSkills；③引入基于恢复性、轨迹驱动的演化与行为验证门控，保证技能更新有效且稀疏。

**🔧 技术方法**

采用冻结的 MLLM（Qwen3‑VL‑8B、Emotion‑Qwen、Gemini‑3.5‑Flash）+外部编辑器（GPT‑4o 或 GPT‑5.6‑sol）实现技能编辑；利用诊断提示、配对推理、文本编辑器和验证集来更新技能。

**📊 数据集**

在三个数据集上评估：CA‑MER（音频/视觉冲突+一致子集）、DFEW（在野面部表情识别）和 EMER（开放词表情绪推理）。

**📈 对比分析**

与基线 MLLM 直接推断、静态 pair‑skill 注入、无门控演化等方法比较，CASE 在 CA‑MER 上把 Qwen3‑VL‑8B 的准确率从 46.80% 提升至 59.13%，在 DFEW 上在不同 backbone 上均提升 3–5% 以上，EMER 也实现了跨 backbone 的平均提升 4–6%。

**⚠️ 局限性**

局限性：技能演化的接受率极低，主要受验证门控限制；依赖外部 LLM 编辑器和预定义的候选对；只对已观察到的混淆对进行演化，难以处理新型误差；对极端罕见情绪对的泛化仍待验证。

---

## 407. PathJev: Decision-Native Vision-Language Modelling for Histopathology

**arXiv ID:** 2610.32478 | [PDF](https://arxiv.org/pdf/2610.32478v1)

**作者:** Cristian Tommasino `[一作]` (University of Naples Federico II), Antonio M. Rinaldi `[通讯]` (University of Naples Federico II)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

开发了PathJev-P，一种在一次前向传递内返回校准概率分布的决策原生视觉语言模型，并演示了其在全切片中的聚类标注。

**💡 创新点**

创新点在于：1）将病理注释系统性转换为Typed Decision；2）在组织学上首次对决策原生模型进行训练并验证其在未见器官上的泛化与校准优势；3）实现聚类级标注与拒绝机制。

**🔧 技术方法**

采用MedGemma‑1.5‑4B 4‑bit backbone 加 LoRA 自适应模块，构建共享决策头，使用交叉熵训练；评估使用ECE、NLL、Brier等指标。

**📊 数据集**

训练集：BRACS、SICAPv2 与 SPIDER 四个子集；清洁测试集：HepatoBench（肝）和 HiESD（胃）；在 BRACS、SICAPv2、SPIDER 的见源测试集进行对比。

**📈 对比分析**

与零样本模型 OmniMed‑Jev、CONCH、内部读取（Inner‑Jev）及生成式 SFT 基线以及基于 H‑optimus‑0 的线性探针进行对比；在清洁集上 PathJev‑P 平衡准确率约为0.78，高于生成基线0.70，ECE也从0.07降至0.04；在未见器官上仍低于线性探针，但优于生成基线。

**⚠️ 局限性**

局限性包括：仅处理单图块状态，未在临床路径专家中系统评估；仅有两种未见器官，模型在新器官上过度自信；依赖单一 4‑bit MedGemma backbone；数据集许可受限，无法公开完整数据；未对全切片管线的推理时延和准确性做完整评估。

---

## 408. CauProphet: Teaching Agents to Develop Causal Cognition like Humans

**arXiv ID:** 2610.32477 | [PDF](https://arxiv.org/pdf/2610.32477v1)

**作者:** Kaiwen Wu `[一作]` (Shanghai Jiao Tong University), Chaochao Lu `[通讯]` (Shanghai AI Laboratory)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种能够从经验中累积并利用因果知识的LLM代理，并在多任务中使用该知识指导决策、更新与改进。

**💡 创新点**

创新点在于：①将因果关系抽象为图结构和可写说明，形成可跨任务迁移的因果记忆；②通过贝叶斯更新与ASP推理对因果假设进行增量维护与修正；③在每个新任务中动态检索并组合相关因果知识与当前证据，形成可执行的因果信念。

**🔧 技术方法**

使用的技术包括：LLM（DeepSeek‑V4‑Pro）、ReAct框架、工具调用、ASP求解、贝叶斯因果更新、因果图与写说明记忆、因果分析工具（如Effect Estimation、CausalGame等），以及多种记忆与检索机制。

**📊 数据集**

实验数据集涵盖：AlphaBench因子搜索（股票收益预测）、FutureSim预测问题、合成结构因果模型（SCM）、CausalGame模拟飞行、BLADE科学分析任务、AppWorld日常应用场景、以及公开的股票交易和交易量时间序列。

**📈 对比分析**

与ReAct、OpenCode、text‑memory ReAct等基线对比，实验显示：在股票收益排名（RankIC）上提升约25%，在事件预测（Brier Skill Score）上提升约30%，在CausalGame中击败所有对手并获胜；在BLADE任务中模型匹配率、变量匹配率和变换匹配率均优于基线；在AppWorld中成功率接近100%。

**⚠️ 局限性**

局限性包括：在缺乏足够证据时仍过度依赖基础模型；因果记忆更新受信心权重设定影响；实验阶段有限，未覆盖长期动态环境；需更多真实世界场景验证因果记忆的可持续性与适应性。

---

## 409. Beyond Post-Hoc Rewards: Privileged Reasoning Exploration for Multimodal Embeddings

**arXiv ID:** 2610.32474 | [PDF](https://arxiv.org/pdf/2610.32474v1)

**作者:** Tianyu Yang `[一作]` (University of Göttingen), Daniel Kurzawe `[通讯]` (University of Göttingen)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种利用训练时关系信息（privileged information）指导多模态大型语言模型（MLLM）的推理过程，以提升嵌入质量。

**💡 创新点**

创新点在于把推理过程视为潜在变量，构造输入条件先验（prior）与使用正负关系信息的后验（posterior）两种推理模式，并通过重要性加权的变分目标实现两者的共同演化和选择性迁移。

**🔧 技术方法**

核心技术包括：潜在变量推理框架、盲对比式后验生成、GRPO（基于策略梯度的奖励优化）、iWAE（重要性加权自编码）和基于权重的后验到先验迁移。

**📊 数据集**

使用MMEB‑V2（包含78项图像、视频和视觉文档检索任务）的公开数据集进行训练与评估，并在MMEB‑Train中构造SFT和RL数据。

**📈 对比分析**

在MMEB‑V2上与多种基线（如UME‑R1、TTE_s、MMEmb‑R1、VLM2Vec‑V2等）对比，2B模型平均提升约3.2分、7B模型提升约3.3分，视觉文档检索得分最高提升约6.1–6.4分。

**⚠️ 局限性**

主要局限在于后验生成的推理轨迹可能包含候选信息泄漏，直接蒸馏效果有限；同时需要额外的后验迁移策略和重要性加权来避免不合适的轨迹迁移。

---

## 410. AutoMIL: Automated Multiple Instance Learning for Whole-Slide Image Classification

**arXiv ID:** 2610.32452 | [PDF](https://arxiv.org/pdf/2610.32452v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 411. OPTARI: An open-source software framework for clinical translation of optoacoustic imaging

**arXiv ID:** 2610.32468 | [PDF](https://arxiv.org/pdf/2610.32468v1)

**作者:** Moritz Schillinger `[一作]` (Friedrich-Alexander-Universität Erlangen-Nürnberg), Katharina Breininger `[通讯]` (Friedrich-Alexander-Universität Erlangen-Nürnberg)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `729e5870-4135-47f5-97f2-e3974d07b5dc` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

本研究提出并实现了OPTARI，一个面向大规模临床OptA/US（OPUS）数据的开源图形化分析软件，集成了PATATO数据处理框架和napari可视化引擎，支持多模态图像重建、光谱解混、自动化组织分割、ROI管理以及批量处理。

**💡 创新点**

创新点在于：1）将已有开源框架与直观的GUI相结合，提供可部署的跨平台独立可执行文件；2）实现了基于深度学习的自动化ROI放置和批量分析，显著降低人工操作时间；3）统一采用IPASC标准和开放HDF5格式，实现多厂商数据互操作；4）提供可共享、可复现的参数配置，减少操作员主观性。

**🔧 技术方法**

技术实现基于Qt6 + PyQt5、napari、PATATO（JAX实现的BP重建）、ONNX Runtime（DeepMB重建与UNet分割）、pyqtgraph（交互式绘图）、PyApp（单文件打包）、HDF5/JSON（数据与配置存储）等。

**📊 数据集**

使用了真实临床MSOPUS数据集，主要为PAD（周围动脉疾病）研究：1,977张扫描、102名受试者；以及在肌肉（股四头肌、腓肠肌、胫前肌）等常见解剖部位训练的预训练分割模型。

**📈 对比分析**

与iLabs、PATATO、MITK-PAI、PATLAB、ImageJ/Fiji等工具进行功能和性能对比；自动分割Dice系数0.86、95%置信区间Hausdorff距离1.92 mm；自动与人工ROI的ICC0.934；分类AUROC自动0.76，手工0.77；批量处理时间仅0.93 s/扫描，显著低于人工1 min/ROI。

**⚠️ 局限性**

局限性：仅支持二维手持OPUS扫描，缺乏三维/体积数据处理；自动ROI性能受训练数据分布影响，需针对新解剖或扫描仪进行微调；自动分割结果仍需人工审核；未完成医疗器械认证，不能用于诊断或治疗决策。

---

## 412. Complexity concentration as a god-class and god-method detector: a pre-registered test of a scale-free outlier rule against tool definitions and human labels

**arXiv ID:** 2610.32466 | [PDF](https://arxiv.org/pdf/2610.32466v1)

**作者:** Daniel Sagenschneider `[一作]` `[通讯]` (OfficeFloor), Daniel Sagenschneider (OfficeFloor)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出并验证了一种基于每个仓库中位数与MAD标准化的鲁棒z‑score阈值的比例检测规则，用来识别God类和God方法，并在十种语言的真实仓库与MLCQ人工评审数据上进行实验

**💡 创新点**

首次将传统的绝对阈值God检测器替换为无原始尺度常数的相对异常检测器，旨在实现跨项目可迁移的自动阈值化

**🔧 技术方法**

使用中位数、MAD和modified z‑score统计方法，并结合PMD、CK、Lizard等工具提取复杂度指标，采用AUC、Matthews相关系数、Bootstrap等评估技术

**📊 数据集**

实验数据来自GitHub的十种语言公开仓库（约150个Java仓库等）以及MLCQ提供的专业开发者评审样本

**📈 对比分析**

将比例规则与手工调优的绝对阈值（如PMD GodClass、NPath、Blob等）在宏观MCC、微观MCC和FPR三项指标上进行对比，结果显示在一数据集上误差≤0.10、在另一数据集稍逊，比例规则召回率略高但误报率也相应提升

**⚠️ 局限性**

结论不确定；方法层级受分布形状影响、缺乏耦合度判别；误报多为小类、未能区分集中复杂度与多职责类，需进一步加入耦合分析或在LLM生成代码上验证

---

## 413. Beyond Copying: Lifelong Smart Contract Security Task Construction via Protocol Mutation

**arXiv ID:** 2610.32453 | [PDF](https://arxiv.org/pdf/2610.32453v1)

**作者:** Zifan Peng `[一作]` (Shanghai Artificial Intelligence Laboratory), Dongrui Liu `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究通过协议突变机制，自动化构建并持续扩展智能合约安全任务集，避免 LLM 代理直接复制已公开答案；

**💡 创新点**

创新点在于将协议条件突变与执行检查相结合，实现了无需人工干预的终身任务生成与验证，并显著降低了答案复制带来的评估偏差；

**🔧 技术方法**

采用协议重构、执行跟踪、自动化突变、验证检查以及大语言模型代理等技术，形成完整的任务构建与评估流程；

**📊 数据集**

数据集来源于203个公开安全事件，构造出1,015个检测、攻击与补丁任务，共计3,045个任务输入；

**📈 对比分析**

与仅重建的基线相比，突变后模型在公开答案支持下的成功率平均下降10.9%，复制率下降9.3%；六大 LLM 在检测、攻击和补丁任务上的单小时成功率分别为约30%–60%，平均成本较低；

**⚠️ 局限性**

局限在于仍需依赖 LLM 对攻击脚本的记忆与复制，突变对某些任务的提升有限，且无法覆盖所有潜在漏洞与攻击路径。

---

## 414. Principled Nonlinear Concept Erasure and Steering

**arXiv ID:** 2610.32516 | [PDF](https://arxiv.org/pdf/2610.32516v1)

**作者:** Idan Simai `[一作]` (Bar-Ilan University), Uri Shaham `[通讯]` (Bar-Ilan University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于条件自编码器的非线性概念抹除与概念转换框架，利用重建与统计独立性共同实现对敏感属性的去除，且在无标签情况下保持任务效用；

**💡 创新点**

创新点在于将信息保留与独立性相结合的目标，提供完美重建与独立性满足时的理论安全性保证，并首次实现了无额外训练成本的概念转换；

**🔧 技术方法**

核心技术包括条件自编码器、HSIC独立性惩罚、互信息最大化思路以及简洁的两项损失组合；

**📊 数据集**

实验使用了 Bias in Bios、Dial-Sentiment、Dial-Mention 等标准概念抹除数据集，并在 BERT、GPT‑2、DeepSeek、LLaMA、Qwen2.5‑7B 等多种预训练语言模型上进行评估；

**📈 对比分析**

与 Obliviator（有监督与无监督）、FaRM、KRaM、INLP、LEOPARD、kSAL 等方法比较，显示在无标签抹除下实现了更优的效用‑抹除曲线，并在概念转换任务上取得领先表现；

**⚠️ 局限性**

局限在于理论证明依赖 U 与 C 的独立假设，若该假设破碎则抹除与转换效果会下降，且目前实验主要聚焦于文本数据，跨模态推广尚待验证。

---

## 415. DRIVE: Fine-tuning via Data Contribution- and Diversity-aware Weighting with Prior Regularization

**arXiv ID:** 2610.32501 | [PDF](https://arxiv.org/pdf/2610.32501v1)

**作者:** Qing Liu `[一作]` (Hangzhou Institute for Advanced Study, University of Chinese Academy of Sciences), Weiyao Zhu `[通讯]` (Tianjin University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在预训练数据不可获得的单任务微调场景下，提出了 DRIVE（Data ContRIbution- and DiVersity-aware WEighting）框架，通过对样本贡献、样本多样性和样本先验共同建模，实现了对抗遗忘与任务适应的平衡。

**💡 创新点**

创新点在于：① 将样本贡献拆解为反遗忘与任务适应两部分并通过可调的 Shapley 加性分解实现稳定-可塑性权衡；② 引入可微的协方差行列式正则化实现样本多样性约束；③ 通过 KL 正则把样本特征先验融入权重分配，形成统一的可微目标；④ 证明了该目标在线性和局部非线性情况下具有更优的稳定-可塑性保证，并给出了收敛快的迭代求解器。

**🔧 技术方法**

采用了 Shapley 加性理论、协方差行列式多样性正则、KL 正则权重先验、随机投影（Johnson–Lindenstrauss）降低特征维度、固定点迭代求解器（线性收敛），并在微调过程中以梯度兼容的方式动态更新权重。

**📊 数据集**

在视觉任务上使用 ImageNet-1K、CIFAR‑10/100、Flowers102 等；在语言任务上使用 MetaMathQA、Magicoder 微调至 GSM8K、HumanEval 等；评估基准包括 ResNet‑18/50、Gemma‑2‑2B、Llama‑3.2‑3B、Llama‑3.1‑8B 等模型。

**📈 对比分析**

与标准微调、FLOW、WiSE‑FT、LoRA、DEFT‑UCS、TALR 等方法进行对比。实验表明 DRIVE 在保持预训练性能的同时提升目标任务准确率，整体平均准确率提升 1–3% 左右，收敛速度更快，参数估计更稳健，并在视觉和语言双模任务上均超越 SOTA。

**⚠️ 局限性**

局限性包括：① 依赖统一先验或简单的先验估计，信息丰富的先验需额外计算；② 需要调节 λ、τ、η 等超参数，影响性能；③ 计算成本仍受协方差矩阵求逆影响，虽然通过投影降低，但在极大维度或样本量极大时仍需注意；④ 目前仅针对单任务微调，跨任务连续学习仍需进一步扩展。

---

## 416. Distortion-Aware Spatiotemporal Learning for Real-Time Wide-Angle Video Rectification

**arXiv ID:** 2610.32500 | [PDF](https://arxiv.org/pdf/2610.32500v1)

**作者:** Lang Nie `[一作]` (Chongqing University of Posts and Telecommunications), Bin Xiao `[通讯]` (Chongqing University of Posts and Telecommunications)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 FastVC 框架，用于实时全身视频畸变校正，兼顾面部与身体区域。

**💡 创新点**

创新点包括畸变感知非均匀薄板样条（DA‑TPS）、双层时空学习（邻域时空注意力NST +稀疏监督）以及首个专门针对全身视频畸变的基准数据集。

**🔧 技术方法**

采用薄板样条控制点重排、邻域时空注意力模块、稀疏时空监督策略、轻量级网络结构以及RAFT光流用于运动平滑。

**📊 数据集**

使用了作者构建的 32 条 1080P、30fps、共 17,029 帧的全身视频畸变数据集。

**📈 对比分析**

与多种图像/视频基线对比，FastVC 在面部/身体平均差异分别为 3.04%/2.35%，稳定度 0.8542，帧率 57.0 FPS，整体性能显著优于现有方法。

**⚠️ 局限性**

局限性包括对伪标签生成的依赖、跨设备泛化仍需验证，以及在极端遮挡或极端畸变场景下的鲁棒性待进一步提升。

---

## 417. L0-Norm-Based CycleGAN for Unpaired Brain CT-to-MR Synthesis

**arXiv ID:** 2610.32491 | [PDF](https://arxiv.org/pdf/2610.32491v1)

**作者:** Yunxiao Zhang `[一作]` (Qingdao University), Chuanlong Xu `[通讯]` (Qingdao University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

在无配对脑CT到MR合成任务中引入L0范数的循环一致性损失，改进CycleGAN网络；

**💡 创新点**

用平滑可微的L0近似取代传统L1，形成红衰退梯度函数，既提升对细节的敏感性又抑制大残差的影响；

**🔧 技术方法**

采用CycleGAN架构、Least‑Squares GAN损失、残差块生成器、Instance Normalization、PatchGAN判别器以及L0平滑近似函数；

**📊 数据集**

使用Han公开的367张CT/MR图像（裁剪为256×256），挑选111张配对图像用于实验；

**📈 对比分析**

在同一网络架构下与L1基线对比，使用102张训练图像、9张测试图像，结果显示MSE降低20.7%，PSNR提升约1.03 dB，SSIM提升0.0282，且最大MSE下降、最小PSNR/SSIM提升；

**⚠️ 局限性**

仅在单中心toy数据集上验证，缺乏多中心数据和下游任务评估，对σ与λ的敏感性也未进行系统分析。

---

## 418. A curated multi-endpoint drug-protein bioactivity dataset with construct-aware protein targets

**arXiv ID:** 2610.32490 | [PDF](https://arxiv.org/pdf/2610.32490v1)

**作者:** Lina Abou-Abbas `[一作]` (Lebanese American University), Khadidja Henni `[通讯]` (TELUQ University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文构建了 Curate-DTI 数据集，重新处理并标准化 1 July 2026 BindingDB 导出，按 IC_50、K_i、K_d、EC_50 四个端点独立保留单链测量，并用实验序列和 UniProt 访问号定义构造感知蛋白实体。

**💡 创新点**

创新点在于：①将不同端点分离并保留不等式测量，②采用实验序列识别蛋白构造，③构建关系‑aware 分类与回归标签，并且完整保留来源和实验元数据。

**🔧 技术方法**

使用技术包括 Python 3.12、pandas 2.2.3、NumPy 1.26.4、RDKit 2025.09.4、Biopython 1.86、fastparquet 2025.12.0，以及 Zstandard 压缩和基于规则的 SMILES/序列标准化与对齐。

**📊 数据集**

数据集来源为 2026‑07‑01 BindingDB 完整导出和 2026‑07‑23 UniProtKB/Swiss‑Prot 版本；共保留 2.92 M 单链测量、1.29 M 标准化配体、9 668 序列定义蛋白实体和 2.28 M 端点特定交互。

**📈 对比分析**

本文未提出新模型，仅提供标准化数据、端点特定交互表和多种划分策略（冷药、冷蛋白、冷 UniProt、时间/源文档分组）以供研究者自行构建和比较机器学习模型；性能评估可基于这些划分自行完成。

**⚠️ 局限性**

限制包括：①分类阈值 1000 nM 为操作性定义，可能与实验上下文不一致；②EC_50 记录不区分激动剂/拮抗剂；③蛋白构造仅依据 BindingDB 报告序列，无法捕捉未记录的实验差异；④专利等来源元数据不完整，可能导致信息泄漏控制不足。

---

## 419. State Estimation as In-context Tabular Prediction

**arXiv ID:** 2610.32487 | [PDF](https://arxiv.org/pdf/2610.32487v1)

**作者:** Ayushi Jolotia `[一作]` (Indian Institute of Technology Roorkee), Parikshit Pareek `[通讯]` (Indian Institute of Technology Roorkee)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文将电力系统状态估计转化为在上下文中的表格预测问题，利用历史测量与对应状态估计构成的上下文，使用预训练的TabPFN/TabICL模型在不更新权重的前提下预测新测量向量对应的状态向量。

**💡 创新点**

创新点在于：不依赖网络特定的模型训练和观测性前提；使用冻结的表格基础模型即可跨五个不同输配电网络直接应用；在测量丢失、噪声放大、坏数据以及拓扑变化等极端场景下表现优于传统WLS及已公开的学习方法。

**🔧 技术方法**

采用基于Transformer的在上下文学习的表格模型TabPFN和TabICL，并结合线性最小二乘恢复绝对相角。

**📊 数据集**

使用IEEE 14、30、33、70、118节点输配电网络的仿真数据以及TU‑Delft公开的1080个样本的CIGRE 14、Oberrhein 70节点网络数据，所有测量均为带噪声的合成测量。

**📈 对比分析**

与WLS及其鲁棒变种、DSS²、TAPINE等方法在测量缺失、噪声放大、坏数据以及拓扑重构等测试中比较，ICTP在所有测试中均保持低于上下文平均误差；在低测量率时误差比WLS低4.4倍以上；在坏数据攻击下误差仅提升1.8–5倍，而WLS则提升2.5–12.7倍。

**⚠️ 局限性**

局限在于计算时间略高于WLS（118节点系统100次估计239–393秒对比81秒），且当前每个目标列单独前向传播，可通过进一步优化实现加速。

---

## 420. PCFBench: How Far Can Large Vision-Language Models Go in Physics-aware Photonic Inverse Design?

**arXiv ID:** 2610.32485 | [PDF](https://arxiv.org/pdf/2610.32485v1)

**作者:** Shengchao Chen `[一作]` (University of Technology Sydney), Sufen Ren `[通讯]` (Hainan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `14d48e9d-0069-4ad9-996a-1d5968216998` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了PCFBench基准，评估VLM在光子晶体光纤逆向设计中的感知、推理和决策能力。

**💡 创新点**

创新点在于首次提供涵盖感知、物理理解、多模态推理、逆向决策及代码生成的26项任务，聚焦视觉到物理到设计的完整链路。

**🔧 技术方法**

采用多模态语言模型（VLM）与多智能体系统（MAS），并结合Meep FDTD仿真与专家注释进行评测。

**📊 数据集**

使用220K个基于Meep FDTD的光纤样本，涵盖18种结构族，并提供结构图、模场图及四层专家标注。

**📈 对比分析**

通过单模型与多智能体两轨道比较，发现模型在几何感知上表现良好，但在逆向决策上仅达50%闭环成功；多智能体协同虽提升效果，但仍受感知错误传播限制。

**⚠️ 局限性**

限制在于缺乏物理根植的推理能力，规模增大不一定提升设计表现，且错误传播导致整体性能瓶颈。

---

## 421. DIGOM: Disentangled Interaction Graphs for Outcome prediction in Multimodal healthcare

**arXiv ID:** 2610.32451 | [PDF](https://arxiv.org/pdf/2610.32451v1)

**作者:** Hakima Laribi `[一作]` (Université de Sherbrooke), Martin Vallières `[通讯]` (McGill University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了一种基于分离视图的稀疏多关系图模型DIGOM，用于多模态医疗数据的临床结局预测。

**💡 创新点**

创新点在于同时建模跨模态、跨患者同模态和跨患者跨模态三种关系，并通过可微分离散采样实现病人自适应稀疏图。

**🔧 技术方法**

采用双投影生成相似度、Gumbel-Softmax+ST估计构造离散邻接矩阵、GIN消息传递、软HGR相关正则以及密度正则等技术。

**📊 数据集**

使用MIMIC-IV（含ICU、住院、X光、文本等多模态）和NIH-TB（基因、CT等）两个公开多模态医疗数据集。

**📈 对比分析**

与传统融合、单一关系图模型以及MaxCorrMGNN等方法比较，DIGOM在五项预测任务上均取得最高AUROC，且图稀疏且对缺失模态更鲁棒。

**⚠️ 局限性**

局限包括：需要预设稀疏率、未学习各关系类型权重、对极为重要模态缺失时鲁棒性不足、仅捕捉模态间关系未细化特征层面的交互。

---

## 422. Lessons Outlive Their Failures: A Case Study of Rules for AI Coding Agents

**arXiv ID:** 2610.32578 | [PDF](https://arxiv.org/pdf/2610.32578v1)

**作者:** Hassan Hassanzadeh Aliabadi `[一作]` `[通讯]`, Hassan Hassanzadeh Aliabadi

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对个人 Git 仓库进行为期六周的 AI 编码代理运行记录，构建 118 条失败账本，测评 LLM 文档审核对漂移的召回率，并在多个模型上对五个历史失败进行回放检验。

**💡 创新点**

首次系统性记录了在真实日常使用中 AI harness 的失效模式及其修复后是否复发，证明“教训”往往比模型本身更持久；同时发现 LLM 审核对文档漂移的检测率仅为 50% 以上，强调可执行检查的重要性。

**🔧 技术方法**

利用 Claude Code / Antigravity 运行时、预提交检查脚本、LLM 文档审核脚本、日志记录与回放脚本；对比不同 Claude 版本与 Gemini 3.8 Flash 运行时；通过多轮 LLM（Sonnet、Haiku、Opus）执行审核与重放。

**📊 数据集**

单一私有 Markdown + Python 脚本仓库（约 1200 次提交）作为实验数据集；故意植入 20 条合成漂移错误以验证审核召回；利用 47 条真实漂移条目作为评估基准。

**📈 对比分析**

通过三次独立运行的 LLM 审核评估召回率和精确度；在三种模型上对同一失败进行 3 次重放，记录出现率；结果显示：Claude Sonnet 5.5 召回率约 49.6%（对 47 条漂移），精确度 93%；执行重放时，许多历史失败在新版模型上未复现，说明模型改进可降低失效；可执行检查往往能在提交前拦截错误，提示更高。

**⚠️ 局限性**

研究仅覆盖单一用户单一仓库，缺乏多样化开发者和项目背景；观察窗口短（六周），部分失效标记为 Unknown；重放次数有限，统计显著性不足；LLM 本身用于生成、评分和重放，可能导致偏见；漂移答案集不完整，真实漂移数量被低估。

---

## 423. Reuse, Don't Retrain: Efficient, Modality-Versatile Geospatial Foundation Models through Feature Distillation

**arXiv ID:** 2610.32566 | [PDF](https://arxiv.org/pdf/2610.32566v1)

**作者:** Mario Hamon `[一作]` (Universite Bretagne Sud), Minh-Tan Pham `[通讯]` (Universite Bretagne Sud)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了通过特征蒸馏压缩GeoFM模型，并设计无配对多专家蒸馏框架TEDI，将不同传感器专家的表示合并到共享Transformer主干中。

**💡 创新点**

创新点在于（1）首次实现了无配对、多模态专家共享主干的蒸馏，保留各自专长；（2）通过扩大token宽度（1024维）缓解多教师共享导致的容量瓶颈；（3）显著降低预训练成本，仅为完整预训练的约0.5%。

**🔧 技术方法**

技术手段包括Vision Transformer、旋转位置嵌入（RoPE）、两层GELU MLP投影、基于patch-level余弦距离的特征蒸馏、PANGAEA-9任务评估以及多尺度、跨模态的训练协议。

**📊 数据集**

使用的数据集为SSL4EO-S12（Sentinel‑2）、FMoW（高分辨率RGB）、PANGAEA‑9（9个多任务集成测试），并在教师和学生训练中均采用这些公开数据。

**📈 对比分析**

在PANGAEA‑9上与CROMA、TerraMind、DINOv3等教师比较，TEDI‑Large取得62.51 mIoU，TEDI‑Base仅略逊61.75 mIoU，同时参数仅占教师的27%，GFLOPs降低71%；蒸馏训练耗时仅占原始预训练的0.5%，显著提升了准确‑计算前沿。

**⚠️ 局限性**

局限性包括：仅评估了两种专家和固定任务子集，未检验更多模态或更广泛任务；教师多样性与共享容量干扰难以分离；实验仅做三次下游跑，细微性能差异不够稳健；数据集覆盖仍有限，可能继承教师的偏差。

---

## 424. Combinatorial Minimum Cost Flow in Almost-Linear Time on Dense Graphs

**arXiv ID:** 2610.32556 | [PDF](https://arxiv.org/pdf/2610.32556v1)

**作者:** Bernhard Haeupler `[一作]` (INSAIT), Thatchaphol Saranurak `[通讯]` (University of Michigan)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文提出一种纯组合学方法，在密集有向图上求解最小费用流（MCF）问题，时间复杂度为 n^2 + o(1)，为 Goldberg–Tarjan 以来的首次改进。

**💡 创新点**

创新点包括：①黑盒化将精确 MCF 降到 DAG 上的弱近似问题；②提出“流‑成本保持 DAG 投影”，在保持拥塞与费用的同时将一般图映射为 DAG；③在 DAG 上设计了加权推-标记（weighted push–relabel）算法实现 (3,1+) 近似；④构造了长度受限扩散层次（LC‑expander hierarchy）和有限宽度路径覆盖（bounded‑width path cover），用于实现 DAG 投影；⑤将上述技术组合形成完整的 n^2 + o(1) 时间框架。

**🔧 技术方法**

主要技术手段包括：黑盒化减少（Exact → Approximate → DAG）；流‑成本保持 DAG 投影（基于长度受限扩散层次与路径覆盖）；加权推-标记算法（利用权重函数控制可接受路径）；FatLDD（实现具有大分离因子的邻域覆盖）；cut‑matching 游戏与多元权重更新（MWU）用于构造扩散层次和生成约束；以及递归减尺度的长度限制投影。

**📊 数据集**

论文为理论算法研究，未使用具体实验数据集。所有结果均通过数学证明与理论分析得到。

**📈 对比分析**

与传统 O(mn) 的 Goldberg–Tarjan 算法相比，本文实现了 n^2 + o(1) 的时间复杂度，几乎达到密集图上最优下界。与基于内部点法的近线性时间算法相比，本文保持了传统残差图范式，提供了新的组合学视角。实验与对比未给出，但理论上证明了性能提升。

**⚠️ 局限性**

局限性：①算法为随机化，仅在高概率下正确；②当前仅对密集图有效，稀疏图仍保持 O(mn)；③需要假设存在高效的 (κ,1+) 近似 MCF 或者 DAG 投影的 oracle，实际实现仍需进一步工作；④算法实现复杂，涉及多层递归与多种结构构造；⑤改进幅度相对有限，仅在 n^2 的基础上加上 o(1) 项。

---

## 425. Beyond Channel Data: Configuration-Enriched Radio Foundation Models

**arXiv ID:** 2610.32542 | [PDF](https://arxiv.org/pdf/2610.32542v1)

**作者:** Shrihari Vasudevan `[一作]` (Ericsson R&D), Thirumaran Ekambaram `[通讯]` (Ericsson R&D)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

将信道配置(CFG)信息嵌入无线电基础模型(RFM)的预训练过程中，构造CFG-增强的嵌入向量并在三种下游任务（LoS/NLoS分类、时延估计、波束选择）与不同标签量级及OOS配置下进行评估。

**💡 创新点**

首次在RFM预训练阶段直接加入CFG令模型获得的嵌入具备更强的归纳先验，显著提升在少标签和分布外（OOD）场景下的性能；并通过对多种嵌入策略（CLS、平均/最大池化）进行系统对比，揭示CFG增强在不同任务中的相对优势。

**🔧 技术方法**

使用Transformer编码器（多头自注意力）、Mask Channel Modeling (MCM) 预训练目标、加权NMSE损失、GELU激活、动态梯度裁剪；下游任务采用固定FCN头；实验采用DeepMIMO生成的多场景信道数据。

**📊 数据集**

DeepMIMO 数据集（30个CFG变体，约126K CFR样本）以及针对下游任务的自定义配置（不同天线尺寸、子载波数、波束库大小）。

**📈 对比分析**

与原始信道标记、加入CFG但无RFM、以及无CFG的RFM三类基线进行对比；在LoS/NLoS、TA和Beam-8/16/32任务上，CFG-增强RFM在少量样本和OOD情形下往往取得最高或相近的F1/NMSE分数；在Beam-16/32任务中CFG优势不明显。

**⚠️ 局限性**

CFG只提供粗粒度传播几何信息，难以捕捉邻近波束之间细微空间差异；在高分辨率波束选择任务（Beam-16/32）中CFG增强效果有限；同时实验依赖DeepMIMO，未验证跨设备/真实场景的通用性。

---

## 426. Single-Machine Guarantees on Unrelated Machines and Beyond: Proportional Fairness for Nonclairvoyant Polytope Scheduling

**arXiv ID:** 2610.32545 | [PDF](https://arxiv.org/pdf/2610.32545v1)

**作者:** Alexander Lindermayr `[一作]` (Technische Universität Berlin), Bart Zondervan `[通讯]` (University of Bremen)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了多面体调度问题（PSP）在非预见在线设置中的表现，特别是比例公平（PF）算法在单机和多机环境中的竞争比。

**💡 创新点**

提出了一类新的支持子次模多面体，证明PF在该类多面体上保持了单机的竞争保证，反驳了PF在某些异构环境中不具备O(1)-竞争比的观点。

**🔧 技术方法**

使用了比例公平（PF）算法和新的双重拟合分析技术。

**📊 数据集**

使用了多面体调度问题（PSP）作为数据集，特别关注无关机器和多项式体的调度。

**📈 对比分析**

与之前的算法相比，PF在支持子次模多面体上对于加权完成时间的竞争比从27改进到4，并且在图调度中首次实现了O(1)-速度O(1)-竞争保证。

**⚠️ 局限性**

在一般PSP中，证明了每个随机非预见算法的竞争比至少为2+10^-5，显示出一般PSP比单机调度更具挑战性。

---

## 427. Peregrine: A Language-Agnostic Packet Processing Intermediate Representation with Formal Guarantees

**arXiv ID:** 2610.32639 | [PDF](https://arxiv.org/pdf/2610.32639v1)

**作者:** Sean Rhee `[一作]` (New York University), Anirudh Sivaraman `[通讯]` (New York University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个面向网络数据平面的语言无关IR，并实现了形式化验证的等价性检查器。

**💡 创新点**

创新点在于从一开始就设计IR并附带形式语义与等价性验证，利用Rocq证明验证器正确，并支持多种DSL前端。

**🔧 技术方法**

使用了Rocq证明助手、Z3 SMT求解器、OCaml桥接、Python脚本前端，以及简化的内存与解析模型。

**📊 数据集**

评估使用了来自GitHub的开源网络程序集合，涵盖P4、eBPF、ParserHawk等多种DSL，包含多种网络功能和不同实现算法。

**📈 对比分析**

与现有P4、eBPF编译器输出进行等价性验证，发现多处误编译错误；验证速度在毫秒到秒级，足以覆盖大多数程序；与LLVM等通用IR对比，规模更小、速度更快。

**⚠️ 局限性**

尚未支持并行hdr_op、条件去解析、某些eBPF helper以及完整的并发状态；前端转换层仍依赖测试而非形式化；内存模型限制了复杂map等特性。

---

## 428. Residual Expert Routing for Text-Only Domain Adaptation of Factorized Transducer ASR

**arXiv ID:** 2610.32637 | [PDF](https://arxiv.org/pdf/2610.32637v1)

**作者:** Shashi Kumar `[一作]` (Idiap Research Institute), Andreas Stolcke `[通讯]` (Uniphore)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文提出一种保守残差路由方法，将文本自适应的非空白预测器视为对基准预测器的残差校正，并用每个前缀的 16k 参数门决定允许多少校正。

**💡 创新点**

创新点在于将残差路由直接嵌入到概率空间的因式化转导器中，使门在不同前缀动态调整校正量，既能保留文本适应带来的 OOD 性能提升，又不牺牲源域性能，无需目标域音频调优。

**🔧 技术方法**

使用的技术包括因式化转导器（分离空白与非空白分支）、门控残差融合、跨域文本自适应、基于 KL 约束和稀疏正则的门训练，以及对目标文本的语言优势监督。

**📊 数据集**

实验数据来自 40 小时的客服对话语料（覆盖银行、医疗、保险）作为源域，另外两类未见域（零售 10,981 条，电信 18,889 条）作为目标域；评估集包含这两类 OOD 集合和源域 3.4 小时集合。

**📈 对比分析**

与基准因式化转导器和仅文本自适应、静态插值的对比实验显示，残差路由在 Retail、Telco 上分别把 WER 从 19.89/20.91 降至 16.86/16.57，几乎全部保留文本自适应收益，同时源域 WER 仅从 12.90 轻微变为 12.91，优于单一插值参数在三组数据上的任何设定。

**⚠️ 局限性**

限制主要在于门仅通过预测器残差判断域信息，可能对极端分布差异或与门特征不匹配的任务表现不佳；此外门训练仍需源域音频与目标文本交替混合，模型扩展到不同结构的 ASR 体系时需进一步验证。

---

## 429. Toward Open ISA Supercomputing: A Comparative Study of RISC-V, Arm, and x86 Architectures on HPC Workloads

**arXiv ID:** 2610.32632 | [PDF](https://arxiv.org/pdf/2610.32632v1)

**作者:** Emanuele Venieri `[一作]` (University of Bologna), Andrea bartolini `[通讯]`

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

对 Sophgo SG2044 RISC‑V CPU 与 Intel Xeon Platinum 8480+ 与 NVIDIA Grace CPU Superchip 在统一软件栈和实验方法下进行系统级性能与能效评测。

**💡 创新点**

首次系统地把 RISC‑V 与主流 x86‑64 与 Arm HPC 架构做大规模基准（STREAM、HPL、HPCG、HPL‑MxP）及真实科研/AI 应用（iPIC3D、PLUTO、OpenFOAM、llama.cpp）对比，量化 RISC‑V 发展历程（Monte Cimone 三代）与能效优势。

**🔧 技术方法**

使用 Linux 6.12、GCC 14.2、OpenMPI 5.0、OpenBLAS 0.3.34、MPI 稳定化的进程/线程绑定、统一编译优化旗标、RAPL/硬件监控功耗采集与 IPMI 结合；HPL‑MxP 采用自研混合精度实现；LLM 推理使用 llama‑bench。

**📊 数据集**

Synthetic：STREAM、HPL、HPCG、HPL‑MxP；科研：iPIC3D（等离子体模拟）、PLUTO（磁流体动力学）、OpenFOAM（CFD）、llama.cpp（20B MoE LLM inference）。

**📈 对比分析**

采用强缩放实验（固定问题规模、逐步增核），对比单核基准、全核吞吐、相对加速、能耗（kJ）和能效（GFLOP/J 或 tok/J）。结果显示 SG2044 单核 FP64 仅 8–15× 低于 x86/Arm；但在内存带宽、HPL‑MxP、HPCG、LLM 推理等方面表现趋于接近；在能效上多场景优于 Intel、优于 Grace。SG2044 在 Monte Cimone 三代间实现了 100× 内存带宽、158× FP64 性能与 10× 能效提升。

**⚠️ 局限性**

局限性：单核计算能力受限于 128‑bit vector 长度和 12‑nm 工艺；缺乏成熟的向量化库和硬件优化；多核心时频率下降导致子线性缩放；能耗测量在 RISC‑V 只能通过板级功耗与模型估计，存在误差；实验仅覆盖单节点级别，未深入多节点网络与集群级别评估。

---

## 430. Can AI Agents Help Robotic Engineers? Lessons from Two Focus Days in Industry

**arXiv ID:** 2610.32636 | [PDF](https://arxiv.org/pdf/2610.32636v1)

**作者:** Salim Fares `[一作]` (University of Passau), Steffen Herbold `[通讯]` (University of Passau)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在真实工业环境中，使用 AI 代理（Claude Code + GPT‑OSS 120B）对四名工业机器人程序员进行两天的 AB/BA 交叉焦点日实验，评估代理在六类机器人编程任务中的完成率、耗时、功能性、代码质量、可读性等指标。

**💡 创新点**

首次在工业机器人编程领域进行系统化的实际交互实验，重点关注学习曲线、代理配置与域知识集成对性能的影响，并验证机器人工程师对 AI 代理的使用习惯和适应性。

**🔧 技术方法**

采用开源大模型 GPT‑OSS 120B 与 Claude Code 代理框架，在本地服务器（单块 H100 GPU）上无云部署，配合自定义技能和项目上下文（ABB RAPID 参考手册、任务描述等）。

**📊 数据集**

使用 AKE Technologies 提供的真实项目任务描述（六类任务：重构、文档、调试、路径反转、轨迹生成、3D 堆叠）以及相应的 ABB RAPID 代码与 RobotStudio 虚拟工作站数据。

**📈 对比分析**

通过 AB/BA 交叉焦点日设计收集每个任务的完成率、耗时、功能性（是否满足需求）、代码质量、可读性等指标，并辅以自动验证器检测编程标准违规。实验结果显示，除高度重复的轨迹生成任务外，代理未提升整体生产率，功能性与代码质量普遍低于无 AI 场景；但经过代理重新配置后，代码质量与标准遵循度有所提升。

**⚠️ 局限性**

局限性包括样本极小（仅四名工程师，第二天三人），实验时间短，未进行盲评；只评估单一代理和模型，缺乏跨模型对比；未实现与 RobotStudio 的深度集成，导致缺乏空间感知；缺乏长期纵向研究，无法完整揭示学习曲线与生产率提升的时间跨度。

---

## 431. EditJev: System One Routing for Image Editing Models

**arXiv ID:** 2610.32635 | [PDF](https://arxiv.org/pdf/2610.32635v1)

**作者:** Zhile Guan `[一作]` (Peking University), Jian Zhang `[通讯]` (Peking University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

为指令式图像编辑任务提出一种动态路由方法 EditJev，能够在每个请求上选择最合适的编辑模型

**💡 创新点**

利用 System One 结构直接预测候选模型的偏好得分，并结合模型速度得分实现质量与推理时延的权衡

**🔧 技术方法**

基于多模态大语言模型 Qwen3‑VL‑4B‑Instruct、LoRA 微调、轻量级偏好头；训练采用离线候选模型输出的双侧评分构建的 pairwise ranking 损失

**📊 数据集**

在 ImgEdit 和 GEdit‑En 两个公开图像编辑基准上进行评估

**📈 对比分析**

与固定模型、通用 MLLM 路由器（GPT‑5.5、Gemini‑3.7‑Flash 等）以及系统一路由基线（Jev）对比；EditJev 在 ImgEdit 上获得 4.592 分，比最强固定模型 4.540 提升，GEdit‑En 上 8.255 分，比 8.194 提升；同时端到端时延降低 60.8%/31.7%，路由推理仅 0.094s

**⚠️ 局限性**

依赖离线生成和评估，评估者的准确性与偏差可能影响路由决策；仅针对固定 5 个候选模型，扩展到动态模型池仍需改进；对大规模多模型场景的鲁棒性尚未验证

---

## 432. DriveOSS: Dataset and Benchmark for 3D Anomaly Segmentation and Open-set Semantic Segmentation in Autonomous Driving

**arXiv ID:** 2610.32612 | [PDF](https://arxiv.org/pdf/2610.32612v1)

**作者:** Zhiran Yan `[一作]` (Technische Hochschule Ingolstadt), Gordon Elger `[通讯]` (Technische Hochschule Ingolstadt)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究提出DriveOSS数据集和基准，用于3D开放式语义分割。

**💡 创新点**

首创真实世界多模态数据集，包含多天气/光照下的ID与OOD场景，并引入open‑IoU评估。

**🔧 技术方法**

采用LiDAR+相机融合与后置异常评分以及特征基方法；使用Mask4Former3D、P3Former、LCPS、LIDO等模型。

**📊 数据集**

使用DriveOSS自身，结合SemanticKITTI、Panoptic‑CUDAL作为训练集。

**📈 对比分析**

通过闭集语义分割、异常分割和开放式分割的mIoU、AUROC、FPR@95等指标评估，结果显示当前模型在OOD检测和open‑set性能仍远低于闭集准确率。

**⚠️ 局限性**

仅单帧、固定传感器配置，未利用时间信息，且开源数据对低光/恶劣天气的覆盖仍有限。

---

## 433. UAE GovTrust Scope Filtering Gains and Routing Failures in Bilingual Public Service Retrieval

**arXiv ID:** 2610.32608 | [PDF](https://arxiv.org/pdf/2610.32608v1)

**作者:** Aleksandar Mihajlovic `[一作]` `[通讯]` (Singidunum University), Aleksandar Mihajlovic (Singidunum University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在 UAE GovTrust 诊断上评估多语言检索与范围过滤，比较不同检索与路由策略的表现；

**💡 创新点**

首次将服务控制位置与附带位置区分，并系统分析范围过滤对检索效果的提升与路由失效的潜在风险；

**🔧 技术方法**

使用 BM25、E5 密集检索、RRF 级联、跨编码器、以及基于规则的服务检索等技术；

**📊 数据集**

采用 22 份双语服务卡、50 个情景（36 可回答）以及 110 变体的人工构造问句集；

**📈 对比分析**

在 72 条双语可回答问句上，RoleRRF 通过范围过滤实现 95.8% 的 Hit@1，远超 BM25 61.1% 与 E5 66.7%；在 168 条后审扰动中，原始路由仅正确处理 2 条（1.2%），显示路由鲁棒性不足；

**⚠️ 局限性**

数据集为人工合成，缺乏独立法律与语言专家验证；仅 22 条卡片，规模有限；路由规则在扰动下极易失效；未评估大规模文档、实时性能或生成答案的准确性。

---

## 434. LLaCO: Continuous Diffusion for Large Language Models with Competition-Aware Semantic Scheduling

**arXiv ID:** 2610.32604 | [PDF](https://arxiv.org/pdf/2610.32604v1)

**作者:** Zijian Jia `[一作]` (Beijing University of Posts and Telecommunications), Shuchen Weng `[通讯]` (Beijing Academy of Artificial Intelligence)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种新的连续扩散语言模型框架LLaCO，并给出了关于token竞争的理论分析——Competition Threshold Theory以及解决时间压缩的Semantic Retention Scheduling。

**💡 创新点**

创新点在于发现并剔除欧氏距离下的范数衰减导致的伪竞争，利用球面几何构造无范数衰减的扩散路径，并通过对齐语义保留率的对数刻度重新分配时间步，从而显著提升了连续扩散模型的生成质量。

**🔧 技术方法**

主要技术包括：球面SLERP轨迹、掩码偏置终点采样、Embedding Directional Adapter、时间条件调制、对数秩保留调度以及块级去噪推理。

**📊 数据集**

使用了1.32M多领域指令数据进行训练，并在ARC‑E/ARC‑C、MMLU、GPQA、BBH、GSM8K、MATH、HumanEval、MBPP等标准基准上进行评测。

**📈 对比分析**

在与TESS‑2‑7B、DSL‑LLaDA‑8B等连续扩散基线以及LLaDA、LLaDA‑1.5‑8B、Dream‑7B等离散扩散模型对比时，LLaCO‑8B在所有评测任务上均超过连续扩散基线，且在知识推理、数学推导和代码生成方面与离散扩散模型保持竞争力。

**⚠️ 局限性**

局限性在于理论推导主要基于高维空间中近正交结构的统计近似，且对噪声分布和表示几何的假设可能不适用于其它类型的语言模型或不同的噪声模型。

---

## 435. Automated Multi-Line Bug Repair Using Model Checker Aided LLMs: A Case Study on RISC-V Processors

**arXiv ID:** 2610.32597 | [PDF](https://arxiv.org/pdf/2610.32597v1)

**作者:** Debleena Das `[一作]` (Indian Statistical Institute), Amlan Chakrabarti `[通讯]` (A. K. Choudhury School of Information Technology, University Of Calcutta)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种结合大语言模型、断言与模型检测器的闭环自动 RTL 代码修复方法，能在单行和多行错误场景下修复代码并通过模型检测器验证修复正确性。

**💡 创新点**

创新点在于：①利用失败断言提供语义上下文，引导 LLM 生成准确补丁；②通过模型检测器做正式验证，保证补丁不引入新错误；③在大规模 RISC‑V 核心上实现可扩展的多行错误修复，首次实现此类高成功率。

**🔧 技术方法**

核心技术包括：大语言模型（GPT‑5.5）用于代码补丁生成；JasperGold 模型检测器用于失败断言分析与补丁验证；突变测试（mutation testing）用于构造多种错误类型；闭环迭代流程（Stage 1/Stage 2）。

**📊 数据集**

数据集：BRISC‑V RISC‑V 核心的六个模块（L1_bus_interface、coherence_controller、snooper、Lxcache_controller、Lxbus_interface、cache_controller），共 60 条突变案例，覆盖 10 种突变类型（逻辑、关系、算术、常量、条件、停滞、敏感度、连线、切片、嵌套 if‑else）。

**📈 对比分析**

与单纯 LLM 修复（Stage 1）对比，加入断言后修复成功率从 28/60（≈47%）提升至 98%（单行）/93%（多行）。实验中补丁平均 4k–4.5k 令牌，模型检测时间 135–1273 ms，显示方法既快速又可靠。

**⚠️ 局限性**

局限性：仍有极少数错误未被修复；方法高度依赖高质量断言集，若断言覆盖率低则定位困难；对大型模块的计算资源需求（模型检测耗时）仍需进一步优化。

---

## 436. Contextual Decision Making: Predictive or Generative

**arXiv ID:** 2610.32584 | [PDF](https://arxiv.org/pdf/2610.32584v1)

**作者:** L. Jeff Hong `[一作]` (University of Minnesota), Xinyao Li `[通讯]` (University of Minnesota)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究了在上下文随机优化中，比较预测先优化（PtO）与生成先优化（GtO）两种学习目标的理论与实践效果，给出选型准则并在合成实验与真实零售库存数据上进行评估。

**💡 创新点**

提出了基于凸上下文随机优化的理论框架，阐明统计效率、凸性保持与模型复用三维度的权衡，并给出实用的选型准则。

**🔧 技术方法**

采用非参数回归（K近邻、核岭回归）、随机森林等预测方法以及条件扩散模型（CDM）等生成式方法。

**📊 数据集**

使用合成新闻贩售与产品摆放实验数据，以及来自京东的 FreshRetailNet-LT 新鲜零售库存数据。

**📈 对比分析**

通过相对成本和在线计算时间进行比较，发现样本量大、特征维度高时 GtO 在决策质量上优于 PtO，但在线时间更长；在小样本或单任务场景下 PtO 更快且成本更低。

**⚠️ 局限性**

仅考虑凸问题，未覆盖离散或多阶段决策；未深入优化生成器的生成速度；理论与实证之间存在泛化差距。

---

## 437. From Parameter Rebasin to Functional Transfer Across Heterogeneous Transformers

**arXiv ID:** 2610.32581 | [PDF](https://arxiv.org/pdf/2610.32581v1)

**作者:** Filippo Rinaldi `[一作]` (University of Modena and Reggio Emilia), Simone Calderara `[通讯]` (University of Modena and Reggio Emilia)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a4b10f5d-130b-4e77-9367-6469ec621899` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种训练‑free 方法，利用局部 Procrustes 对齐源模型激活变化，并用闭式岭回归直接在目标模型中合成本地任务向量，从而实现不同深度、宽度 Transformer 之间的任务特定行为迁移。

**💡 创新点**

创新点在于：①把任务向量视为激活空间中的功能变更，而非参数差异，避免了参数传输的难题；②采用局部 Procrustes 映射与闭式岭回归，无需梯度或再训练即可构造目标模型的更新；③通过块级对应关系实现跨架构的可扩展性。

**🔧 技术方法**

使用的技术包括：局部 Procrustes 对齐、闭式岭回归、稀疏任务向量构造、块对应映射、激活对齐（可选梯度对齐）、经验贝叶斯岭正则化。

**📊 数据集**

实验数据集与模型：Vision8 视觉任务、IFEval 指令跟随、GSM8K 数学推理、HarmBench ARC‑Challenge 安全对齐；模型为 OpenCLIP ViT‑B/16 与 ViT‑L/14（视觉）以及 Qwen 0.5B/1.5B/3B（语言）。

**📈 对比分析**

比较方法：与参数传输基线（Rebasin、Cross‑Arch rebasin 等）在相同校准预算下比较。结果显示：在扩展与缩减场景下，本方法在目标模型上提升 10–12 个百分点，IFEval 上甚至超过源模型；构造时间比基线低 10 倍以上，峰值内存占用显著降低。

**⚠️ 局限性**

局限性：①仅在同一模型家族内验证，跨家族或模态迁移尚未测试；②需要预先指定块对应关系，且依赖代表性校准数据；③对极端深度/宽度差异或完全不同结构的模型迁移可能效果不佳。

---

## 438. From Regulation to Engineering: Organizational Privacy Climate as Mediator

**arXiv ID:** 2610.32576 | [PDF](https://arxiv.org/pdf/2610.32576v1)

**作者:** Oshrat Ayalon `[一作]` (University of Haifa), Michael Birnhack `[通讯]` (Tel Aviv University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究 GDPR 对美国软件开发者隐私工程实践的影响及其机制，提出并检验了法规-组织气候-态度-实践的中介模型

**💡 创新点**

首次量化验证 GDPR 重要性通过组织隐私气候影响开发者隐私态度进而影响工程实践的中介链，揭示法规直接与间接两条路径

**🔧 技术方法**

结构方程模型（SEM）与确认性因素分析（CFA），使用 EFA 进行量表构建并进行路径分析

**📊 数据集**

基于 156 名美国信息系统开发者的在线问卷数据，问卷包含 GDPR 重要性、组织隐私气候、隐私工程态度与实践四个构念

**📈 对比分析**

对比直接效应与间接效应，结果显示 GDPR 重要性对实践影响最强，且间接路径（气候→态度→实践）显著，模型拟合良好（RMSEA 0.054, CFI 0.924）

**⚠️ 局限性**

样本局限于美国开发者，样本量相对较小，且采用自报、横断面设计，难以确定因果关系，缺乏跨文化验证

---

## 439. Tool-Calling LLM Agents Under Realistic MCP Faults: Scale Does Not Purchase Fault Tolerance

**arXiv ID:** 2610.32555 | [PDF](https://arxiv.org/pdf/2610.32555v1)

**作者:** Antonio Ken Iannillo `[一作]` (University of Luxembourg), Indika Kumara `[通讯]` (Tilburg University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种针对Model Context Protocol（MCP）的协议感知故障注入中间件，并基于对1,774个公开MCP仓库57,240条问题与拉取请求的挖掘，构建了符合真实事件分布的故障模型，随后在20B与120B两种gpt‑oss模型、三种agent框架（MCPBench、OpenAI SDK、LangChain）上，采用1,350个实验单元进行全因子实验，评估故障对工具调用LLM代理的鲁棒性。

**💡 创新点**

创新点包括：①构建了基于社区实际故障报告的、包含持久性（永久、短暂、间歇）和概率调度的MCP故障模型；②实现了drop‑in、可种子化的中间件，能够在不改动代理或工具的情况下注入语义与时序错误；③通过大规模实验验证模型尺度与框架错误处理对鲁棒性的影响，发现规模并未显著提升容错性，框架级错误处理更为关键。

**🔧 技术方法**

技术手段包括：MCP协议拦截代理、Python实现的fault-injection API、gpt‑oss 20B/120B模型、三种工具调用框架、MCP-Bench任务集、LLM-as-a-Judge评测、统计与线性混合效应模型。

**📊 数据集**

数据集主要有：1) 57,240条英文GitHub线程（1,774个MCP仓库）用于构建故障分布；2) 25个单服务器MCP-Bench任务（总计1,350实验单元）；3) 1,350个实验结果日志与评测分数。

**📈 对比分析**

比较方法：在三种故障注入条件（无注入、OP0通过、OPP校准）下进行配对比较，采用任务完成率、任务完成分数、失效模式分布等指标。实验结果显示：在OPP下任务完成分数平均下降60%（从8.08降至3.23），完成率下降至88–97%，部分退化(56%)是最常见的失效模式，模型规模并未带来显著容错优势。

**⚠️ 局限性**

局限性包括：故障模型受GitHub报告偏倚，主要为永久性事件；缺乏多服务器、多任务时长的实验；仅测试两种模型与三种框架，难以推广至其他大模型或多代理系统；LLM评测随机性与评判尺度限制；间歇与短暂故障事件样本稀缺，导致其效果难以充分评估。

---

## 440. Hair Segmentation and Skin Characterization on Macroscopic Dermatological Images for Laser Treatment Parameterization

**arXiv ID:** 2610.32609 | [PDF](https://arxiv.org/pdf/2610.32609v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 441. Monte Carlo-Based Indoor UAV Localization in Dynamic Environments Using Only Four Onboard Range Sensors

**arXiv ID:** 2610.32645 | [PDF](https://arxiv.org/pdf/2610.32645v1)

**作者:** Mohammad Reza Ranjbar Divkoti `[一作]` (University of Porto), A. Pedro Aguiar `[通讯]` (University of Porto)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种基于Monte Carlo定位的AF-MCL算法，利用四个单点测距传感器和测量航向实现室内无人机定位，并在存在动态障碍物的环境中保持鲁棒性。

**💡 创新点**

创新点在于：①仅在上一估计位置进行束分类（射线投射），避免粒子级射线投射开销；②动态束过滤与自适应粒子注入相结合，可在滤波发散或“绑架”情形下快速恢复；③粒子分布直接给出在线定位不确定度。

**🔧 技术方法**

核心技术包括：Monte Carlo定位、动态束过滤（基于混合模型的测距误差判定）、指数移动平均检测滤波器失效、自适应随机粒子注入、系统低方差采样、欧氏距离变换（EDT）以及在线协方差不确定度估计。

**📊 数据集**

实验数据集涵盖：①Webots仿真环境 6 m×6 m 与 10 m×10 m（含动态障碍）；②真实实验环境 2.10 m×1.63 m，使用 Crazyflie 2.1、Multi‑ranger Deck、Flow Deck 及 Qualisys 摄像头；③Gazebo Garden 预验证。

**📈 对比分析**

与标准MCL、仅注入A‑MCL、仅过滤F‑MCL以及可靠MCL（R‑MCL）进行对比。AF‑MCL在动态障碍下仿真平均误差为4.5–5.2 cm，实测为6.4–7.7 cm；相较于标准MCL在动态障碍下误差提升12.3 cm，AF‑MCL降至7.7 cm；定位收敛时间从16.9 s降至2.9 s，实时计算在每个更新周期内约为4–5 ms。

**⚠️ 局限性**

局限性包括：①动态束过滤可能削弱几何信息，导致部分时刻信息不足；②需要完整已知静态地图，无法在无地图环境下直接使用；③实验仅在机载计算机上验证，未测试在机载STM32等资源受限平台的可行性；④对地图误差、传感器噪声及光照变化仍有一定敏感性。

---

## 442. The Standard Deviation in VICReg Buys a Safety Margin

**arXiv ID:** 2610.32630 | [PDF](https://arxiv.org/pdf/2610.32630v1)

**作者:** Hawar Dizayee `[一作]` `[通讯]` (Independent Researcher), Hawar Dizayee (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在VICReg中对比了标准差hinge和方差hinge的表现，采用不同权重μ在ResNet-18、CIFAR-10/100上训练并记录线性探针准确率与梯度特征。

**💡 创新点**

首次在实验中验证了VICReg作者提出的“标准差hinge必要性”主张，发现两种hinge在默认权重下效果相近，但方差hinge的崩溃阈值比标准差hinge高出35–130倍，为实现更安全的训练提供了经验依据。

**🔧 技术方法**

使用VICReg三项损失、梯度push/pull分析、在线线性探针评估、混合精度训练和监控批次梯度记录等技术。

**📊 数据集**

使用CIFAR-10和CIFAR-100两个常见图像分类数据集进行实验。

**📈 对比分析**

通过在不同μ下比较两种hinge的线性探针准确率来评估性能，结果显示在默认权重μ=1时两者差距不到0.1个百分点；然而方差hinge在较低权重下更快崩溃，标准差hinge在更宽的权重范围内保持稳定。

**⚠️ 局限性**

实验仅针对ResNet-18、CIFAR数据集、固定的ε和其他超参数，阈值对ε敏感；评估仅使用在线线性探针，未涉及离线评估或最近邻评估，梯度分析仅限于幅值而非方向。

---

## 443. Interpreting Supervised Pairwise Graph Representations: Local Evidence, Confounding, and Provable Post-hoc Correction

**arXiv ID:** 2610.32611 | [PDF](https://arxiv.org/pdf/2610.32611v1)

**作者:** Ilya Brekin `[一作]` (HSE University), Ural Valeev `[通讯]` (Ufa University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `c773407a-6119-4871-b8b3-1e7ae17a6851` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本研究构造监督式 pairwise 图表示，并在 Transformer 的错误检测任务与 HCP Emotion 组装图上评估其解释性，系统检验高阶拓扑是否真正提供信息；

**💡 创新点**

创新点在于设计了控制实验框架（引入方向不变的决定性得分、序列长度等噪声变量控制）、使用 Fisher 后验校正、对跨注意力机制进行分解，以及将这些方法结合在同一套 attribution 逻辑中；

**🔧 技术方法**

采用 Transformer（2 编码层+2 解码层）、Synolitic GNN 进行图构造、GATv2 与三层 GCN 进行图学习、逻辑回归基线、Fisher post‑hoc 校正、bootstrap 置信区间等技术；

**📊 数据集**

使用两类数据集：10k 条 Transformer 序列反转样本（synthetic）和 581 名受试者的 HCP Emotion 组装图（379 节点，fear vs neutral）；

**📈 对比分析**

与 raw attention‑as‑adjacency、模型置信度、平面图特征等基线比较，Transformer 端 ROC‑AUC 从 0.559（raw attention）提升到 0.839（监督式 pairwise），但在加入序列长度控制后优势消失；HCP 端 signed node‑summary 逻辑回归已达 0.924–0.999，GCN 的改进不超过 0.004，Fisher 校正满足分布无关下界；

**⚠️ 局限性**

局限性包括仅测试一类 Transformer 任务、一种 HCP 对比、单一 pairwise 构造和 GCN 家族，缺乏对其他消息传递网络或图生成方法的验证；此外模型选择与评估共享随机区域抽样，可能导致轻微乐观偏差，结果受固定图生成管线的约束。

---

## 444. OCTOPUS: Towards A Tag Model for Cross-Query Reuse in Semantic Query Processing

**arXiv ID:** 2610.32586 | [PDF](https://arxiv.org/pdf/2610.32586v1)

**作者:** GuangXue Zhang `[一作]` (University of California), Sharad Mehrotra `[通讯]` (University of California)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a2602d71-93ab-4bad-974b-672788df8193` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了标签模型（Tag Model）和标签操作符（Tag Operators），实现跨查询语义判断重用的数据库中间件。

**💡 创新点**

创新点在于将模型判断以可持久化的标签形式存储为数据库列，实现跨查询、跨操作符的重用与推理。

**🔧 技术方法**

使用 PostgreSQL 扩展、LLM（如 ChatGPT）进行语义解析、标签写回与查询重写，以及关系型数据库的索引与过滤。

**📊 数据集**

主要使用 SemBench 电影场景（Rotten Tomatoes 评审）以及 Amazon 产品数据进行实验。

**📈 对比分析**

与 iPDB、ThalamusDB、Palimpzest、LOTUS 等基线比较，行数/对数评估表明在 SemBench 上可减少 10 到 108 倍模型调用，质量保持在 0.8 以上。

**⚠️ 局限性**

局限性包括规则推理误差、标签生命周期维护、模型置信度阈值设置以及缺乏完整的事务与视图维护机制。

---

## 445. UAV-SwarmAgent: Scalable Multi-UAV Reconnaissance and Task Allocation via Code Generation and Vision-Language Reasoning

**arXiv ID:** 2610.32559 | [PDF](https://arxiv.org/pdf/2610.32559v1)

**作者:** Oleg Sautenkov `[一作]` (Skolkovo Institute of Science and Technology), Dzmitry Tsetserukou `[通讯]` (Skolkovo Institute of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一个将自然语言指令自动转化为多无人机侦察任务的系统，利用地面站智能体生成代码并分配航迹，配合无人机上的视觉语言决策器完成现场判定。

**💡 创新点**

创新点在于：①分层多智能体架构，地面站智能体通过工具调用式代码生成完成任务规划；②航迹按最长路径平衡分配实现高效多机协作；③在无人机上部署轻量级视觉语言推理器；④给出可预测的多机飞行时间Amdahl‑型扩展定律，帮助操作员预估资源投入与收益。

**🔧 技术方法**

采用ReAct循环与Python工具调用技术；使用视觉语言模型（Gemini、GPT‑4o、Qwen‑7B）进行地图目标定位；使用Valen‑Preview 2B视觉语言决策器做航迹图像即时判定；航迹分配算法基于“最远优先‑最近邻”实现；仿真飞行模型基于像素距离与巡航速度计算。

**📊 数据集**

使用UAV‑VLA卫星图集（约1.4×1.4 km农村场景）以及五个海岸场景，人工合成四类事件（结构火灾、森林火灾、车辆碰撞、油污泄漏）并嵌入目标地图，形成结构化评测数据集。

**📈 对比分析**

通过跨模型（Gemini、GPT‑4o、Qwen‑7B）和规模（1–5架无人机）的对比实验评估。整体成功率约90%（41/45），航迹时间随无人机数提升呈Amdahl‑型曲线，第二架无人机已实现约50%可得加速，五架无人机提升有限；总任务时间随无人机增多下降约18%。

**⚠️ 局限性**

局限性包括：仅在无风、无电量、无障碍的仿真环境下测试；事件为合成，缺乏真实场景；未对语言模型输出进行多次重跑；未包含人工参与，操作员相关结论仅为假设。

---

## 446. Optimization, Optimal Control, and Reinforcement Learning for Robotics

**arXiv ID:** 2610.32538 | [PDF](https://arxiv.org/pdf/2610.32538v1)

**作者:** Yonghyeon Lee `[一作]` `[通讯]` (Yonsei University), Yonghyeon Lee (Yonsei University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

通过课程笔记的形式系统介绍了最优化、最优控制和强化学习在机器人领域的理论基础与实践应用，重点强调两者的共通数学框架与在机器人决策中的互补性。

**💡 创新点**

将最优控制与强化学习两大视角统一呈现，并用机器人具体案例阐释它们如何在同一平台上协同工作，提供一种新的教学与研究框架。

**🔧 技术方法**

运用了动态系统建模、轨迹优化、模型预测控制（MPC）、价值函数与Bellman递归、以及强化学习（RL）算法等技术手段。

**📊 数据集**

本书主要以机器人仿真与实验为示例，并未使用公开数据集，典型的仿真平台包括ROS/Gazebo 等。

**📈 对比分析**

通过理论推导与案例演示展示两种方法的优势与局限性；缺乏定量实验对比，性能评估主要基于示例结果与直观理解。

**⚠️ 局限性**

局限性包括：内容偏向理论与示例，缺乏大规模实验验证；省略了部分严谨证明与高级正则性条件；对复杂真实环境的适用性需进一步验证。

---

## 447. InferSlice: A Reference Architecture for SLO-driven LLM Inference over a Telco Wide-Area Network

**arXiv ID:** 2610.32638 | [PDF](https://arxiv.org/pdf/2610.32638v1)

**作者:** Juan Parra-Ullauri `[一作]` (BT Group), Nektaria Efthymiou `[通讯]` (BT Group)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 InferSlice 架构，在 telco 级 WAN 上部署 SLO 驱动的 LLM 推理服务，并在 JOINER 多点实验平台上对其性能进行系统评估。

**💡 创新点**

创新点包括：① 将网络切片、GPU 分区、弹性推理与端到端可观测性集成到单一参考架构；② 在共享普通运营商网络上实现多租户 SLA 区分；③ 通过权重量化、并行推理和预填/解码拆分等多种分布式技术，在 WAN 环境中探索其可行性边界；④ 将权重量化和流式加载与重量级模型的弹性伸缩相结合，显著降低冷启动延迟。

**🔧 技术方法**

采用了 Kubernetes + Kube‑OVN、NVIDIA Run:ai、KubeRay、Ray、vLLM、MinIO、Longhorn、Prometheus/ Grafana 等云原生与 AI 专用技术栈；网络层使用 Geneve 隧道进行逻辑切片，算力层通过 GPU 资源分配和分区实现；推理层采用 vLLM 进行张量/流水线/专家并行、预填/解码拆分；监控层利用 Prometheus 采集跨层指标。

**📊 数据集**

主要使用 ShareGPT 数据集（短文本交互）作为推理工作负载；LLM 模型包括 Llama‑3.1‑8B‑Instruct（8B）用于服务层实验，GPT‑OSS‑20B（21B）用于分布式推理实验。

**📈 对比分析**

通过三类实验验证：① 服务层级差异（Free、Standard、Premium）比较 TTFT、P99 与合规 goodput；② 冷启动实验对比默认加载与并行流式加载的延迟；③ 网络瓶颈与分布式推理实验（张量、流水线、专家并行、预填/解码拆分）评估跨节点时延、吞吐、goodput 与传输带宽。实验显示：Premium 最高 2.2× 的合规 goodput，流式加载将冷启动时间从约1500 s 减至约610 s；张量并行仅在单子网内可行，流水线并行可跨 10 ms 延迟范围；预填/解码拆分在跨节点时因 KV‑cache 传输导致几秒延迟。

**⚠️ 局限性**

限制包括：① 仅在同一 GPU 代（L40S）上测试，未考虑不同加速器或 CPU 组合；② 实验采用静态 SLA 策略，未验证闭环自适应调度；③ 只使用无状态聊天请求，未涵盖状态化/代理工作负载；④ 分布式并行实验仅到两分片，四分片未能完成；⑤ 未评估能耗、碳排放与电源限制等实际运营商约束。

---

## 448. Natural-Language to Nav2: Operational Policies for Costmap-Based Navigation

**arXiv ID:** 2610.32633 | [PDF](https://arxiv.org/pdf/2610.32633v1)

**作者:** Jiyeong Chae `[一作]` (Daegu Gyeongbuk Institute of Science and Technology), Kyung-Joon Park `[通讯]` (Daegu Gyeongbuk Institute of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `51c0528b-f690-4182-ae60-bb5f046c276c` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

实现了一套自然语言指令转为ROS 2 Nav2可用JSON策略的接口，并将接受后的策略投影到成本图中，从而让移动机器人能够根据操作员的指令动态调整路径与约束。

**💡 创新点**

① 设计了结构化的策略模式和事务式接受检查机制，保证指令既符合语法也与当前任务状态兼容；② 在不改动Nav2核心规划器与控制器的前提下，通过成本图适配器实现keepout、软成本和对象避让；③ 支持定时与事件驱动的约束，以及多机器人共享同一策略的功能。

**🔧 技术方法**

使用本地部署的大语言模型（Llama 3.1-8B、Qwen系列等）进行指令解析；利用ROS 2 Nav2的costmap框架；实现KeepoutLayer、SoftCostLayer、ObjectAvoidanceLayer等适配器；采用事务式验证与上下文检查确保安全接受；在Gazebo仿真与TurtleBot3实测平台上验证。

**📊 数据集**

150条英文指令基准，覆盖10种策略意图、5种措辞层次；结合仓库地图的Gazebo仿真环境和真实TurtleBot3迷宫实验。

**📈 对比分析**

与规则基解析器进行对比：LLM的正确接受率从44/150提升至107–142/150；仿真与实测任务1–3的平均行驶时间分别下降18.4–24.5%（仿真）和23.1–35.0%（实测），行驶距离同步下降；keepout实现对比显示两种方式在路径规划和耗时上无显著差异；共享策略演示验证了多机器人间策略同步的可行性。

**⚠️ 局限性**

缺点包括：未在独立生成的命令上评估泛化能力；对复杂语句仍存在误解析；实验仅限于固定布局与几条任务路径；LLM推理延迟高（1–3 s）；未实现指令确认与授权流程；未充分探讨多机器人间约束冲突与协调。

---

## 449. What Transfers Across Protein Pockets? Experience Reuse in Molecular Optimization for Agentic Structure-Based Drug Design

**arXiv ID:** 2610.32620 | [PDF](https://arxiv.org/pdf/2610.32620v1)

**作者:** Zaifei Yang `[一作]` (Hong Kong University of Science and Technology), James Kwok `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文研究了如何利用大型语言模型在结构药物设计中，通过逐步编辑配体并进行对接来优化配体与蛋白口袋的结合。

**💡 创新点**

创新点在于将编辑过程中的对接反馈记录为经验库，并尝试将这些经验迁移到新的蛋白口袋上，以减少新任务的对接成本。

**🔧 技术方法**

采用了LLM驱动的PROBE框架、AutoDock Vina对接评分、MolCRAFT生成配体，以及基于键值检索的经验库技术。

**📊 数据集**

使用的主要数据集是CrossDocked2020测试集中的蛋白口袋及其初始配体。

**📈 对比分析**

通过将经验库键值检索与仅依据ΔN的基线进行对比，评估其在新口袋上的预测准确率，发现迁移效果有限，主要提升来源于编辑本身而非口袋信息。

**⚠️ 局限性**

局限性包括：仅在单一实验设置（PROBE、MolCRAFT、Vina、CrossDocked2020）下验证，键值检索采用精确匹配导致可迁移性不足，且未评估实际对接节省与闭环优化收益。

---

## 450. When disclosure rules are not enough: authorial warrant and judgment in ai-supported scholarship

**arXiv ID:** 2610.32574 | [PDF](https://arxiv.org/pdf/2610.32574v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 451. OWLS: Open-world Language-conditioned Speed Fields for Zero-Shot Rover Navigation

**arXiv ID:** 2610.32596 | [PDF](https://arxiv.org/pdf/2610.32596v1)

**作者:** Chahyon Ku `[一作]` (University of Michigan), Bernadette Bucher `[通讯]` (University of Michigan)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为OWLS的局部导航方法，该方法直接在开放词汇视觉语言模型（VLM）的密集特征图上进行规划，能够在没有任务特定训练和持久地图的情况下进行导航。

**💡 创新点**

创新点在于将VLM的每个像素响应通过地面投影转化为自我中心的价值图，并通过两个形态操作适应平台，最终作为最小时间规划器的速度场。

**🔧 技术方法**

使用了开放词汇视觉语言模型（VLM）进行特征提取，并结合形态学操作进行平台适应。

**📊 数据集**

在200个程序生成的火星和月球地形上进行评估，同时在真实的行星图像上进行定性测试。

**📈 对比分析**

与NoMaD等现有导航基础模型进行比较，OWLS在大多数未见地形上成功到达目标，成功率显著高于NoMaD，且在所有四个设置中表现优于后者。

**⚠️ 局限性**

局限性包括：1）设计为无记忆，无法记住离开视野的障碍；2）缺乏3D理解，平面假设可能导致在陡坡上规划不准确；3）运行在超出当前飞行硬件计算能力的VLM上；4）语言作为规范接口可能不稳定；5）未能在模拟实验中考虑行星光学效应。

---

## 452. GLP Networking API Specification

**arXiv ID:** 2610.32583 | [PDF](https://arxiv.org/pdf/2610.32583v1)

**作者:** Dan Bachar `[一作]` (Technical University of Munich), Ehud Shapiro `[通讯]` (London School of Economics)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `9cc9baba-5356-466d-81ff-d80028d90279` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

设计并规范了适用于多代理 deterministic GLP（madGLP）的网络层，提供统一的 API 并实现三种底层传输（PAN、LAN、WAN），实现点对点可靠、加密、身份与应用证明以及 NAT 穿透。

**💡 创新点**

创新点包括：1) 将网络功能与 GLP 代码严格分离，仅将网络栈、定位、远程 attestation 和加密原语交给底层；2) 采用旋转识别令牌实现接近发现时的隐私保护；3) 在 Noise 握手后加入双向应用 attestation，绑定应用版本与会话哈希；4) 通过轻量级 rendezvous 服务器实现 NAT 穿透、地址发现并保持无中继，兼容多平台。

**🔧 技术方法**

使用的技术包括：Bluetooth Low Energy（GATT、广播）、Multicast DNS (mDNS) 和 DNS‑SD、UDP、Noise 协议（IK、XX）、Ed25519 签名、ChaCha20‑Poly1305 加密、SHA‑256 哈希、Apple/iOS App Attest、Android 设备安全硬件（KeyStore）以及平台 geofencing 服务。

**📊 数据集**

本工作为协议与接口规范，不使用或评估任何数据集，重点在设计与实现细节。

**📈 对比分析**

未进行实验对比，仅阐述了满足 madGLP 对网络层的公平、无序、原子、完整性与身份验证等假设的实现路径，并给出了完整的 API 映射和实现细节。

**⚠️ 局限性**

局限性：缺乏实测性能与安全评估；依赖平台 attestation 与 geofencing，跨平台兼容性未验证；NAT 穿透效果受限于外部 rendezvous 服务器的可用性与配置；未覆盖高级攻击（如伪造信令或对手重放攻击）及系统间的完整性保证。

---

## 453. PHYRD: A Living Benchmark for Reproducing and Extending Theoretical Physics Research

**arXiv ID:** 2610.32575 | [PDF](https://arxiv.org/pdf/2610.32575v1)

**作者:** Qituan Zhang `[一作]` (Shanghai Jiao Tong University), Wenlong Zhang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一个名为PHYRD的“活”基准，包含两条赛道（Reproduce与Discover），用于评估大语言模型在理论物理推导与条件修改下的推理能力，并用FormulaForest从学术论文自动构造任务；

**💡 创新点**

创新点在于：①把理论物理推导划分为复现已发表结果与在修改假设下推导新结论两大类；②构建FormulaForest框架，将论文中的公式与假设映射成可检验的依赖图，自动生成任务与评分标准；③通过工具辅助（符号计算、检索）进一步提升模型表现，验证工具对物理推导的实际帮助。

**🔧 技术方法**

技术主要包括：大语言模型（GPT、DeepSeek、Kimi、Qwen、GLM）在无工具与工具辅助两种环境下的推理实验；FormulaForest的关系图构建与任务生成；子问题级别评估与完整问题准确率计算；错误分类与分析；以及利用生成的SciDeriv任务进行模型蒸馏。

**📊 数据集**

使用数据集：PHYRD（包含63个Reproduce问题和43个Discover问题，总计504个子问题），每个问题来源于五大物理子领域；FormulaForest自动提取的公式与假设；此外对比了OpenAI的FrontierScience基准和SciDeriv生成的推导任务。

**📈 对比分析**

比较方法：对比不同模型在无工具与工具辅助下的子问题准确率(SAcc)与完整问题准确率(WAcc)，并与FrontierScience benchmark的Qwen-35B-A3B模型在同一子基准上的表现进行对照；结果显示工具辅助能显著提升完整问题解答率（如DeepSeek和Qwen分别提升约21个百分点），但仍存在计算与概念错误，整体性能仍低于预期。

**⚠️ 局限性**

局限性：①不同子领域覆盖不均，导致域内性能对比不稳定；②模型在训练期间可能已接触部分相似论文，存在泄漏风险；③依赖工具的改进不一定能解决结构性与概念性错误；④基准更新仍需手工验证和版本管理，维护成本高。

---

## 454. Pseudo-Labels as Rewards, Not Targets: Group-Relative Policy Optimization for Semi-Supervised ASR

**arXiv ID:** 2610.32572 | [PDF](https://arxiv.org/pdf/2610.32572v1)

**作者:** Shashi Kumar `[一作]` (Idiap Research Institute), Andreas Stolcke `[通讯]` (Uniphore)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在没有转录文本的域适应任务中，作者使用伪标签作为奖励参考，采用Group Relative Policy Optimization（GRPO）对语音识别模型进行微调。

**💡 创新点**

创新点在于将伪标签用作奖励而非目标，借助GRPO的on‑policy采样抑制教师错误的传播，从而显著提高在低质量伪标签下的鲁棒性。

**🔧 技术方法**

核心技术包括GRPO（PPO简化版）、Whisper及其他教师模型生成伪标签、基于LLM的ASR模型（冻结WavLM+LoRA适配的Llama），奖励以伪标签的WER差值为准。

**📊 数据集**

使用的数据集为DefinedAI的银行、医疗、保险对话音频、Common Voice 26低资源语言数据以及LibriSpeech预训练语料。

**📈 对比分析**

实验对比SFT、SFT→GRPO和直接GRPO，结果显示直接GRPO在所有教师、领域和语言上均优于SFT；在最差伪标签（Whisper Tiny）下，GRPO相较SFT提升约18.1% WER，且对教师质量的敏感度明显降低。

**⚠️ 局限性**

局限性包括：直接GRPO要求学生模型已支持目标语言脚本；对高质量伪标签的优势有限；未探讨多轮迭代伪标签生成与精度提升的潜力。

---

## 455. HEALPix-Based Implicit Neural Compression of Omnidirectional Images

**arXiv ID:** 2610.32561 | [PDF](https://arxiv.org/pdf/2610.32561v1)

**作者:** Yulong Cheng `[一作]` (Harbin Institute of Technology), Jie Wen `[通讯]` (Harbin Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于HEALPix网格的单图像适配（overfitted）隐式神经压缩码流，针对全景图像实现低复杂度、实时解码。

**💡 创新点**

创新点包括：①使用多尺度等面积HEALPix表示实现均匀采样、避免ERP的极区过采样；②设计球面跨尺度条件熵模型，结合同尺度邻域与更粗尺度祖先上下文；③引入多环球面卷积，在保持算力小的同时获得均匀的球面感受野。

**🔧 技术方法**

采用隐式神经网络（INR）、HEALPix网格、球面跨尺度熵模型、Multi‑Ring卷积、SOAP优化、量化与指数加速编码等技术。

**📊 数据集**

使用Li等公开的100幅768×1536分辨率全景图集，及高分辨率1536×3072的99幅图像做实验。

**📈 对比分析**

与VVC、MLIC++、Cool‑Chic 5.0、OSLO‑IC等基准码流在视口评估（V‑PSNR、V‑MS‑SSIM、V‑LPIPS）下比较，取得相对VVC BD‑rate降低约16.3%、26.5%、19.5%，相对Cool‑Chic 5.0降低约13.1%，并在CPU上解码速度比对比学习型自编码器快2.6–3.6倍。

**⚠️ 局限性**

局限性包括：熵解码在每层串行进行，仍占主导时间；输入全景图需先从ERP转到HEALPix，导致重采样失真；当前仅在CPU上实验，GPU加速尚未充分探索。

---

## 456. Distortion-Rate Outage Probability for Non-ergodic Classes of Sources

**arXiv ID:** 2610.32549 | [PDF](https://arxiv.org/pdf/2610.32549v1)

**作者:** Dongmin Park `[一作]` (American University of Beirut), Razane Tajeddine `[通讯]` (Aalto University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文针对慢速源编码场景，定义了失效概率度量——失效率失真概率（DROP），并给出了在紧致源类下直接计算该指标的方法。

**💡 创新点**

创新点在于将信道侧的 outage 概念迁移到源编码侧，提出 DROP 作为衡量慢速源在固定编码率下失真超限事件的概率，并证明在紧致类中该指标可简化为对单个源的 rate–distortion 函数的集合求最大满足率的概率，极大降低计算复杂度。

**🔧 技术方法**

采用了信息论中的 rate–distortion 理论、紧致性与可测性分析、最优子集优化以及极值理论等技术手段，对源类的失效概率进行解析和数值演示。

**📊 数据集**

示例使用了离散伯努利源（具有不同参数的先验分布）和连续高斯源（均值限制、方差上界），并未涉及具体外部数据集，仅通过理论推导和图示展示。

**📈 对比分析**

通过对比不同编码率下的 P_out 曲线，作者展示了在给定失真阈值时 DROP 随编码率下降的趋势；在伯努利源例子中，P_out 随 R 增大而出现阶梯下降；在高斯源例子中，P_out 随 R 与 D 的变化呈连续下降，直至达到零，表明当编码率覆盖最坏源时无失效。

**⚠️ 局限性**

局限性包括：仅在紧致源类下可实现解析计算，对非紧致类缺乏可行方案；仅给出渐近理论，有限块长性能未被量化；假设源分布完全已知且无通道干扰，实际通信环境中可能需考虑更多噪声与误码因素。

---

## 457. StarRTT: An End-to-End Real-Time Framework for Blind Zone-Free RTT Prediction in Starlink LEO Satellite Networks

**arXiv ID:** 2610.32540 | [PDF](https://arxiv.org/pdf/2610.32540v1)

**作者:** Jaeyong Lee `[一作]` (Korea University), Wonjae Shin `[通讯]` (Korea University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在Starlink低轨卫星网络中构建了实时端到端的服务卫星识别与RTT预测框架StarRTT，并提出了结合历史、实时与未来特征的RT‑LSTM模型。

**💡 创新点**

创新点在于每秒更新服务卫星识别，消除传统15秒周期识别的盲区；通过双分支LSTM与门控融合同时利用历史RTT、实时卫星与网关相对特征以及TLE推算的未来特征，显著提升预测精度。

**🔧 技术方法**

使用LSTM双分支网络、门控融合机制、TLE轨道传播、基于角度距离与方向差的实时识别算法，以及传统的历史RTT特征提取。

**📊 数据集**

采用约114.5小时的Starlink实际测量数据，包括RTT、障碍图、星链gRPC工具提取的轨迹信息，覆盖V1.0、V1.5、V2.0-mini等卫星代。

**📈 对比分析**

与传统LSTM、StarNet和ASTM进行严格实时回测，RT‑LSTM在15步预测平均MAE为1.405 ms，分别比LSTM、StarNet和ASTM低4.3%、11.6%和22.1%。

**⚠️ 局限性**

仍受限于公开TLE数据和对下一周期服务卫星未知的情况；实验集中在日本网关区域，其他地区及未来ISL链路的通用性尚待验证。

---

## 458. LGMT: A Tool for Testing the Logical Reasoning Reliability of LLMs

**arXiv ID:** 2610.32627 | [PDF](https://arxiv.org/pdf/2610.32627v1)

**作者:** Zenghui Zhou `[一作]` (Beihang University), Zheng Zheng `[通讯]` (Beihang University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一个名为lgmt的开源工具，用逻辑等价变换检测LLM推理一致性。

**💡 创新点**

将FOL语法树重写、Z3等价校验与局部自然语言生成结合成端到端的元模型测试框架，并支持自定义元模型关系。

**🔧 技术方法**

使用第一阶逻辑、Z3定理证明器、LLM推理接口、Python、Docker、CLI/Web Studio等技术。

**📊 数据集**

采用由FOLIO、LogicNLI、ProverQA构成的3,091组元模型群（及40个示例子集）做评测。

**📈 对比分析**

对六大LLM进行零样本链式思考评估，发现元模型违规率在20.8%–29.4%之间，静态准确率下降约10%–13%。

**⚠️ 局限性**

目前仅支持FOL推理，改写仍需LLM，且工具未覆盖所有推理缺陷，难以完全保证逻辑一致性。

---

## 459. From archive to resource: bringing legacy data back to life at scale with large language models

**arXiv ID:** 2610.32616 | [PDF](https://arxiv.org/pdf/2610.32616v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 460. Interpretable Singing Emotion Recognition Using Acoustic and Vocal Technique Features

**arXiv ID:** 2610.32629 | [PDF](https://arxiv.org/pdf/2610.32629v1)

**作者:** Fotini Kiriakopoulou `[一作]` (National Technical University of Athens), Giorgos Stamou `[通讯]` (National Technical University of Athens)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

构建了一套可解释的歌唱情感识别框架，利用声学和歌唱技术（如振音、咽喉声等）特征进行情感分类；

**💡 创新点**

创新点在于将歌唱专属的技术特征与传统声学特征结合，并采用 SHAP 与 Grad‑CAM 对模型决策进行可解释分析；

**🔧 技术方法**

使用的技术包括手工提取的 130 维特征、嵌套留一歌手交叉验证的特征选择、XGBoost / Random Forest 分类器、CNN 双输入声纹/音高提取、SHAP 和 Grad‑CAM 解释方法；

**📊 数据集**

数据集包括 RAVDESS 歌唱子集、GTSinger（多语言、多风格）、VocalSet（外域验证）以及部分中文/法语/德语子集；

**📈 对比分析**

与传统基于 eGeMAPS 的“语音优化”特征集相比，歌唱优化特征在 RAVDESS 上 UAR 提升至 79.6%（95% CI: 76.9–82.2%），交叉歌手方差降低 32%；在 GTSinger 上两者 UAR 相近，但歌唱优化集跨风格稳定性更好；

**⚠️ 局限性**

局限性包括受限的歌手数量、RAVDESS 为情感表演、GTSinger 仅包含两种情感、CNN 推断技术特征可能存在标签噪声，且未检验因果关系。

---

## 461. Privacy-Preserving Federated Loss Exploration for Vision-Based Perception under Non-IID Data

**arXiv ID:** 2610.32626 | [PDF](https://arxiv.org/pdf/2610.32626v1)

**作者:** Nergiz Yuca `[一作]` (University of Passau), Stefan Katzenbeisser `[通讯]` (University of Passau)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种基于多方安全计算（MPC）的联邦学习框架PPFlex，用于车辆交通标识分类，解决非IID数据下的梯度共享导致的隐私泄露问题。

**💡 创新点**

创新点在于将FedLEx的梯度引导机制与MPC相结合，既保持了对非IID数据的优化效果，又通过三服务器MPC实现了对指导矩阵和模型聚合的安全计算，填补了非IID优化与安全聚合之间的空白。

**🔧 技术方法**

主要技术包括：MPC（Replicated2k、Semi2k、SPDZ2k、PSReplicated2k）用于安全聚合；梯度偏差计算与归一化；固定点量化与共享输入优化；以及使用MP-SPDZ框架实现可扩展的MPC程序。

**📊 数据集**

实验数据集为德国交通标识识别基准（GTSRB）和比利时交通标识分类数据集（BTSC），采用两层CNN模型（约170万参数）。

**📈 对比分析**

与传统FedAvg相比，PPFlex在α=0.1等高非IID场景下提升了约18%–28%的准确率；在安全模式下，模型精度基本不受影响；MPC在半诚实设置下单轮平均耗时约30秒，恶意安全设置下耗时可达数千秒。

**⚠️ 局限性**

局限性包括：在恶意安全模式下MPC预处理成本高，通信与计算开销随模型与客户端数量显著增长；对恶意客户端的防护尚未实现；且实验仅覆盖交通标识分类任务，尚需验证对更大模型和其他视觉任务的适用性。

---

## 462. Political sense-making in "individual" human-technology-world relations: the case of LLM-generated autobiographies

**arXiv ID:** 2610.32594 | [PDF](https://arxiv.org/pdf/2610.32594v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 463. Earth embeddings are decomposable records of the land surface

**arXiv ID:** 2610.32617 | [PDF](https://arxiv.org/pdf/2610.32617v1)

**作者:** Robin Young `[一作]` (University of Cambridge), Srinivasan Keshav `[通讯]` (University of Cambridge)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

对全球每10米像素的年度地球嵌入向量进行稀疏自编码分解，提取可解释的生态特征并用独立的全球参考数据验证这些特征是否真正对应真实生态事件。

**💡 创新点**

首次证明年度嵌入本身即包含可解释的生态事件信息，并通过稀疏自编码将高维嵌入压缩为少量专一特征，能够在不同大洲、不同事件类型中准确定位事件发生时间、严重度和类别，揭示模型训练目标对嵌入内容的影响。

**🔧 技术方法**

使用TopK稀疏自编码器 (SAE) 进行特征分解，线性探测器与AUC评估验证特征，抽样检验与随机特征集对比，原始嵌入信号对比分析两模型差异。

**📊 数据集**

Tessera (128维) 与 AlphaEarth (64维) 年度嵌入；多种公开参考数据集（MTBS火灾、Hansen森林变化、EFDA森林破坏、RADD 检测、JRC 水体历史、USDA Cropland Data Layer、全球矿产与光伏地图等）覆盖六大洲。

**📈 对比分析**

通过在部分事件上筛选特征集合，在其余事件上验证预期时间窗口匹配率，平均验证率达85%，显著高于随机特征集；两模型在原始嵌入上对同一事件的信号强度、持续时间和特征辨识能力存在明显差异，表明不同训练目标导致嵌入内容差异。

**⚠️ 局限性**

受限于参考数据误差、仅能分辨年级时间分辨率、无法捕捉事件内年内变化或因果来源；对不同模型的解释差异只能通过关联推断，未能系统验证训练目标对嵌入内容的因果影响。

---

## 464. Trajectory Matching: Discrete Flow Matching with Structured Actions

**arXiv ID:** 2610.32602 | [PDF](https://arxiv.org/pdf/2610.32602v1)

**作者:** Niklas Kemper `[一作]` (Technical University of Munich), Stephan Günnemann `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `f86bf285-fd08-4156-973b-6e6481af8fa0` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出轨迹匹配（Trajectory Matching）框架，将离散流匹配的条件化从仅依赖端点扩展到完整动作轨迹，从而支持大小变化、多元和依赖的复杂动作；

**💡 创新点**

核心创新在于通过轨迹条件化解锁一般化动作空间，统一包含传统独立D F M并显著提升结构化生成的有效率与多样性；

**🔧 技术方法**

技术手段包括连续时间马尔科夫链（CTMC）、基于Bregman散度的无模拟训练、Rao-Blackwell化梯度、可变动作空间与轨迹分布的设计与采样；

**📊 数据集**

实验使用分子数据集ZINC、MOSES、QM9以及树生成数据集进行评估；

**📈 对比分析**

与factorized DFM、离散扩散、GrIDDD等基线对比，树生成实现100%有效率，分子生成MOSES有效率从92.8%提升至99.2%，在子结构扩展和链接等下游任务中实现零样本迁移并显著优于基线；

**⚠️ 局限性**

局限性在于需手工设计动作空间与轨迹分布，需保证可采样与Rao-Blackwell化计算可行，且在单元编辑已足够时优势不明显。

---

## 465. Universal neural network approximation of symmetric polyconvex energies and envelopes

**arXiv ID:** 2610.32598 | [PDF](https://arxiv.org/pdf/2610.32598v1)

**作者:** Florian Gammer `[一作]` `[通讯]` (University of Regensburg), Florian Gammer (University of Regensburg)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `14d48e9d-0069-4ad9-996a-1d5968216998` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出一种可保证对称多重凸性的神经网络（SPCNN），用于近似小应变弹性问题中的对称多重凸能量及其包络，并给出了通用逼近定理

**💡 创新点**

创新点在于：①克服传统ICNN在满足对称多重凸性时需满足的严格权重限制导致的逼近不完全问题；②通过引入两两最大/LogSumExp激活函数实现凸性与二阶最小子矩阵的单调性约束；③给出对称多重凸包络的神经网络逼近方法和理论分析

**🔧 技术方法**

使用输入凸神经网络（ICNN）框架，结合对称多重凸性理论、LogSumExp激活、L1/L2 Sobolev回归以及包络逼近损失（绝对误差+上界惩罚）等技术

**📊 数据集**

采用合成实验数据：①基于线性+四次硬化项的双曲面能量生成应力-应变对；②双井势能（含两个相互竞争的能量势）在二维欧氏空间中采样 10×10 或 10000 点；无真实公开数据集

**📈 对比分析**

通过与解析对称多重凸包络（已知解析解）以及仅用凸包络的对比，评价逼近误差：在小应变区误差≤5%，在大应变区误差上升但仍低于传统凸包络；训练时间仅数分钟，参数规模适中，展示了计算效率与物理可解释性的兼顾

**⚠️ 局限性**

局限性包括：①仅适用于小应变对称情形，缺乏对大变形或非对称情况的推广；②网络在训练域外表现受限，误差随应变增大而显著；③对高维（>3）和强非线性材料仍需进一步研究与验证

---

## 466. Standing on Open Shoulders: Regional Fine-Tuning of Open-Weight Camera-Trap Models for UK Wildlife with Modest Data and Compute

**arXiv ID:** 2610.32573 | [PDF](https://arxiv.org/pdf/2610.32573v1)

**作者:** Paul Fergus `[一作]` `[通讯]` (Liverpool John Moores University), Paul Fergus (Liverpool John Moores University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

针对英国野生动物监测，利用公开模型 DeepFaune 进行迁移学习，构建 DeepFaune-UK v1.2/1.3，并与 DeepFaune v1.5.0 和 SpeciesNet 在手工验证的跨地区图像集上比较。

**💡 创新点**

证明仅用少量标签（约150张/类）和单块 GPU，即可在英国图像上实现与大型模型相近的精度，展示开放权重在区域 AI 开发的民主化潜力。

**🔧 技术方法**

采用 MegaDetector v6 检测器+DeepFaune ViT-L/14 (DINOv2/3) 分类器，双阶段训练、两阶段微调、数据增益、类逆频率采样。

**📊 数据集**

主数据集为英国摄像陷阱图像（约36k张训练），评估集为来自 MammalWeb 的 20,212 张手工标注的异地图像。

**📈 对比分析**

在严格物种级别下 DeepFaune-UK 88.8%/90.6% 精度，超过 DeepFaune v1.5.0 (56.5%) 和 SpeciesNet (59.0%)；在按 DeepFaune 词表映射的像级别下，DeepFaune-UK 与 DeepFaune v1.5.0 差距仅 0.8%，并在 10% 训练样本下保持近似性能。

**⚠️ 局限性**

评估集样本不平衡、与训练集存在部分重叠、缺乏罕见物种、模型阈值未调优、实验仅在单一 GPU 上测评，且 DeepFaune-UK 的性能受起始权重和许可证约束。

---

## 467. Shared Is Not Enough: Predictor Width and Factor Readability Under Masked Latent Prediction

**arXiv ID:** 2610.32562 | [PDF](https://arxiv.org/pdf/2610.32562v1)

**作者:** Nilotpal Kakati `[一作]` `[通讯]` (Weizmann Institute of Science), Nilotpal Kakati (Weizmann Institute of Science)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在一个可知因子模拟器中，比较了 I-JEPA、SimCLR 与 BYOL 对共享因子可读性的影响，探讨了预测头容量对编码器表征的影响。

**💡 创新点**

发现默认 I-JEPA 预测头过小导致共享因子可读性低于对比方法，并证明扩大预测头宽度可逆转这一差距，从而揭示预测头容量是影响可读性的关键因素。

**🔧 技术方法**

采用 Vision Transformer 编码器与掩码自回归预训练框架，配合专门设计的 Probe 评估模块来量化因子可读性。

**📊 数据集**

使用自制的四通道模拟数据集，其中每幅图像的因子位置、能量与形状均已知，可分别重绘或保持不变，以便精确测量可读性。

**📈 对比分析**

通过 Probe 的绝对误差改进（d(z₁)）和“移动度”指标对不同预测头尺寸与训练条件下的模型进行比较，发现宽预测头能显著提升可读性分数，而默认预测头与 SimCLR、BYOL 的分数相差显著。

**⚠️ 局限性**

实验仅在单一因子、模拟器和小型编码器上进行，未解析可读性下降的根本机制，也未验证结果是否能迁移至自然图像或更复杂的任务。

---

## 468. On the Expressiveness of Asynchronous Processes with Nested Session Types

**arXiv ID:** 2610.32558 | [PDF](https://arxiv.org/pdf/2610.32558v1)

**作者:** Sjors Wortelboer `[一作]` (University of Groningen), Jorge A. Pérez `[通讯]` (University of Groningen)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

论文证明了在Das的异步会话进程模型（与嵌套会话类型配合）下，能够通过正确编码两计数器Minsky机实现图灵完备性。

**💡 创新点**

创新点在于设计了状态感知的指令过程与递归嵌套类型构成的增量-递减-跳转协议，并构建了递归树结构的实例化过程，以保证对计数器状态的精确控制。

**🔧 技术方法**

使用技术包括嵌套会话类型（NST）、异步会话进程语言（Asynchronous Session Processes）、多态递归类型、类型等价与子类型判定算法以及通过类型层面建模的Peano自然数协议。

**📊 数据集**

论文没有使用外部数据集，而是以理论模型Minsky机作为抽象机器进行编码与证明。

**📈 对比分析**

通过类型可判定性、进化对应性（soundness 与 completeness）以及案例示例证明了编码正确性；由于采用形式化证明与过程对齐，未给出执行性能指标，只说明在理论层面实现了Turing完备性。

**⚠️ 局限性**

局限在于编码实现高度复杂且难以扩展到更大规模的程序，且异步通信模型仍依赖显式的实例化树，未探讨运行时效率或与其他会话类型系统的互操作性。

---

## 469. Semantic Representation for Agentic Self-Play in Automated Laboratories

**arXiv ID:** 2610.32755 | [PDF](https://arxiv.org/pdf/2610.32755v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 470. HGWM: Hierarchical Graph-Guided World Model for Zero-Shot Object Navigation via Graph Matching

**arXiv ID:** 2610.32554 | [PDF](https://arxiv.org/pdf/2610.32554v1)

**作者:** I-tak Ieong `[一作]` (Tsinghua University), Hao Tang `[通讯]` (Peking University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出HGWM（Hierarchical Graph‑Guided World Model），通过LLM生成的层级房间‑锚点先验与持久的空间语义记忆相结合，采用两阶段决策（粗视觉语义评分与细粒度符号兼容性评分）和相位依赖融合实现零样本物体导航。

**💡 创新点**

创新点包括：①使用LLM构建层级目标先验；②维护持久空间语义图记忆；③粗细两阶段评分机制，结合视觉-语义与符号兼容性；④相位依赖融合并加入LLM方向建议；⑤将图匹配显式作为决策信号。

**🔧 技术方法**

采用的技术有：大语言模型（Gemini‑1.5‑Pro、Qwen2.5‑7B‑Instruct）、视觉语言模型用于粗评分；图匹配与关系完善；RGB‑D感知；持久空间语义图；两阶段决策与相位融合；模拟环境Habitat。

**📊 数据集**

使用的数据集为HM3D v0.1、HM3D v0.2和MP3D。

**📈 对比分析**

与WMNav、UniGoal等现有方法在Habitat环境下进行SR和SPL比较，结果显示：HM3D v0.1上SR 62.6%（相较WMNav 58.1%提升4.5pp），SPL 31.5%（提升0.3pp）；MP3D上SR 46.8%（提升1.4pp），SPL 17.9%（提升0.7pp）；HM3D v0.2完整模型SR 74.4%，SPL 31.9%。组件消融后SR下降0.9–2.3pp，SPL下降0.6–1.5pp。

**⚠️ 局限性**

局限性包括：仅在仿真环境评估；未做真实机器人实验、噪声鲁棒性测试；缺乏不确定性估计；LLM推理延迟与成本未全面评估；可能对异常布局的先验假设过度依赖。

---

## 471. Agentic Hazard Mapper: Emulating Expert Map-Making with AI Agents for Risk Communication

**arXiv ID:** 2610.32738 | [PDF](https://arxiv.org/pdf/2610.32738v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 472. Beyond Coverage and Mutation: A Blind-Spot Analysis of LLM-Generated Unit Tests

**arXiv ID:** 2610.32548 | [PDF](https://arxiv.org/pdf/2610.32548v1)

**作者:** Xiaoyu Shi `[一作]` (Aalto University), Haoye Tian `[通讯]` (Aalto University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于义务的评估框架BlindSpotEval，用于评估LLM生成的单元测试，重点关注测试覆盖的边界值、控制流和断言的有效性。

**💡 创新点**

创新点在于引入了BlindSpotEval框架和ObligBench基准，前者通过边界值适应性、控制流适应性和oracle强度等指标，识别LLM生成测试中的系统性盲点，后者提供了复杂性分层的Java基准。

**🔧 技术方法**

使用了基于义务的评估技术，包括边界值适应性（BVA）、控制流适应性（CFA）和oracle强度（OS），并结合执行可靠性分析。

**📊 数据集**

使用了ObligBench数据集，该数据集包含895个复杂和700个低复杂度的Java函数，具有源侧义务注释。

**📈 对比分析**

与传统的覆盖率和突变评分方法相比，BlindSpotEval能够揭示系统性盲点，发现LLM生成的测试在处理边界值和异常控制流方面的不足，尤其是在处理非直接可推导的边界值时表现较差。

**⚠️ 局限性**

限制在于LLM生成的测试在处理复杂对象状态和异常行为的能力较弱，且在简单函数中也存在类似的盲点，表明这些问题不仅仅是复杂性导致的。

---

## 473. Polarization on Bounded Euclidean Spaces and More General Compact Groups

**arXiv ID:** 2610.32760 | [PDF](https://arxiv.org/pdf/2610.32760v1)

**作者:** Sharang M. Sriramu `[一作]` (Fraunhofer Heinrich Hertz Institute), Aaron B. Wagner `[通讯]` (Cornell University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了Arıkan极化变换在圆形或更一般紧群上的连续源极化性质，并给出了条件微分熵的极化速率上界。

**💡 创新点**

首次将熵跳跃不等式与熵功率不等式结合，证明了连续源在Arıkan变换下的极化现象，并指出极化速率极慢，给出了O((log N)^-1/12) 的理论上界。

**🔧 技术方法**

使用熵跳跃不等式、熵功率不等式、马尔可夫过程与停时分析、傅里叶参数化以及蒙特卡洛仿真等技术。

**📊 数据集**

在实验中使用了均匀分布在[0,1]（即圆）上的源和噪声密度为 p_Z(θ)=1+α cos(2πθ)（α=0.98）的加法余弦信道。

**📈 对比分析**

通过计算子通道互信息MI_i，统计MI_i>0.05 nats 的比例随块长N变化，并与1/ln N曲线对比，表明实际极化速度与1/ln N相近，远慢于离散案例。

**⚠️ 局限性**

极化速率极慢，理论上限与实际速率之间存在较大差距；熵跳跃常数非常小，导致上界过于宽松；仅适用于Arıkan变换，未证明其他变换能加速极化。

---

## 474. Adaptive World-Models for Anomaly Recognition (AWARE): A Framework for Robot Dynamics Prediction and Introspection

**arXiv ID:** 2610.32742 | [PDF](https://arxiv.org/pdf/2610.32742v1)

**作者:** Luke Beddow `[一作]` (Imperial College London), Antoine Cully `[通讯]` (Imperial College London)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了 AWARE 框架，用外部摄像头监测，基于世界模型与潜在动力学估计实现异常检测与诊断。

**💡 创新点**

创新点在于双信号架构（潜在动力学误差与预测误差）和无先验、无低层传感器的主动监测方法，兼顾噪声鲁棒性与动力学精度。

**🔧 技术方法**

采用了潜在动力学估计器、世界模型（Transformer 或 rssm）、域随机化训练、基于马氏距离的异常分数、潜在参数解码器等技术。

**📊 数据集**

使用 240 条真实工业塔吊轨迹数据集，包括负载碰撞和电机降解两类异常，且分别在低噪声（运动捕捉）和高噪声（仅摄像头）两种状态估计条件下评估。

**📈 对比分析**

与多种基线（RWM、rssm、Isolation Forest、Statistical Anomaly Detection、AESOP、AnyAnomaly、ROA）比较，AWARE 在低噪声下 AP 0.93、F1 高于最佳基线 3%；在高噪声下 AP 0.77、F1 超过最佳基线 26%，并能以 88% 的准确率定位故障电机。

**⚠️ 局限性**

主要局限在于对固定参考分布假设敏感，若操作环境变化可能产生误报；以及实验仅在单一塔吊平台上验证，跨平台泛化需进一步研究。

---

## 475. CC4Q Workbench: From Quantum Repositories to Analyzable Code Comments

**arXiv ID:** 2610.32651 | [PDF](https://arxiv.org/pdf/2610.32651v1)

**作者:** Zenghui Zhou `[一作]` (Beihang University), Zheng Zheng `[通讯]` (Beihang University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个开源工作台，自动从 Python 量子 SDK 代码中提取、切分并对代码注释进行量子知识分类和人工标注。

**💡 创新点**

创新点在于将注释提取、语义分割、量子知识分类、可视化探索、手工标注和模型微调整合为一个可复用、可扩展的端到端工作流，并在 UI 中隐藏预测结果以减少标注偏见。

**🔧 技术方法**

技术手段包括 Python、LibCST 代码抽象、docutils+spaCy 文本分割、BERT 分类器、Starlette/Uvicorn 后端、Apache ECharts 前端、Docker 镜像和 API 接口。

**📊 数据集**

使用了原 CC4Q 数据集（12,812 条 CCPs、28,076 条 SCCUs），以及新增的三份 Qiskit 应用层仓库（共 11,517 条 SCCUs）来构建和评估新语料库。

**📈 对比分析**

通过人工评估 379 条 SCCUs，二分类器在新仓库上的准确率为 85.0%，与原始测试集（0.936）相近；六分类器在原始数据集上的精度为 0.847，展示了模型在迁移场景下的可用性。

**⚠️ 局限性**

局限性包括仅支持 Python 语言的 SDK，难以处理包含大量经典数值计算或特殊缩写的注释；模型在非量子代码段上可能产生误判，且需手工微调以适应不同域。

---

## 476. Cross-Architectural Feature Fusion for Coronary Artery Stenosis Detection in X-ray Angiography

**arXiv ID:** 2610.32730 | [PDF](https://arxiv.org/pdf/2610.32730v1)

**作者:** Haobo Zhu `[一作]` (University of Oxford), Abhirup Banerjee `[通讯]` (University of Oxford)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出了一种融合 YOLOv11-L 与预训练 UNet++ 的混合框架 Fusion-YOLOv11-L，用于在 XCA 图像中检测冠状动脉狭窄。

**💡 创新点**

创新点在于通过通道注意力门控模块将分割模型的血管感知特征与检测网络结合，提升定位精度，并通过轻量化的测试时适配与增强提高跨域泛化能力。

**🔧 技术方法**

技术包括 YOLOv11-L 检测器、UNet++ 分割器、通道注意力门控 (CAG) 模块、AdaBN 适配、Test‑Time Augmentation (TTAg)、Optuna 超参数搜索。

**📊 数据集**

使用了重标注后的 ARCADE* 数据集（1,500 张 XCA 图像）进行训练与评估，并在 48 张 OxAMI 画像上进行跨域测试。

**📈 对比分析**

与 YOLOv8-L、YOLOv11-L 等基线对比，Fusion‑YOLOv11‑L 在 ARCADE* 上 AP@50 达到 0.585（比基线高 0.011），F1 最高为 0.596（阈值 0.2）。在 OxAMI 上，TTAg 使 AP@50 从 0.434 提升至 0.480，显示出更好的泛化性能。

**⚠️ 局限性**

局限在于对阈值选择敏感、在域移位下仍有性能下降、数据集规模有限，需进一步改进鲁棒性和适应性。

---

## 477. Cross-Scenario Controller Parameter Transformation from Real-World Driving Data through Flow Matching

**arXiv ID:** 2610.32719 | [PDF](https://arxiv.org/pdf/2610.32719v1)

**作者:** Christian Zink `[一作]` (Technical University of Darmstadt), Rolf Findeisen `[通讯]` (Technical University of Darmstadt)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了驾驶员个性化ACC参数在不同跟车场景之间的转换，采用流匹配方法从慢速跟车参数分布学习映射到快速跟车参数分布。

**💡 创新点**

创新点在于将ACC参数个性化场景转换建模为分布运输问题，利用流匹配实现非线性、高维度的参数迁移，并结合β‑VAE驾驶档案、驱动者一致配对、驱动者嵌入和CFG等多种条件方式进一步提升个性化效果。

**🔧 技术方法**

使用了流匹配（ODE flow matching）、Transformer‑based β‑VAE（带set聚合器）用于提取驾驶档案、线性参数平移基线、驱动者嵌入、Classifier‑Free Guidance（CFG）、最小二乘参数识别与多种评估指标（Wasserstein‑2、均值/协方差差异）等技术。

**📊 数据集**

使用了Volkswagen集团收集的真实世界大规模数据集：约1.6万条手动跟车轨迹，涉及170+驾驶员、30辆车，覆盖30个国家的多样化道路与环境。

**📈 对比分析**

通过与线性平移基线比较，采用Wasserstein‑2、均值差异Δμ和协方差差异ΔΣ_F三种指标评估；结果显示流匹配将Wasserstein‑2降至基线的约1/6，平均误差与协方差误差显著降低；CFG+驱动者一致配对+嵌入的变体在大多数司机上实现了最佳或接近最佳性能，尤其在未见司机的留一交叉验证中保持稳健。

**⚠️ 局限性**

局限性包括：仅研究了慢速与快速跟车两种场景，未覆盖更广泛的驾驶环境；方法依赖大量标记数据且对未见司机的驱动者嵌入不可用；未在车辆上进行实车验证，尚需进一步评估在线个性化与系统集成的可行性。

---

## 478. Developing Security Metrics for Automated Compliance Checking

**arXiv ID:** 2610.32690 | [PDF](https://arxiv.org/pdf/2610.32690v1)

**作者:** Immanuel Kunz `[一作]` (Fraunhofer AISEC), Christian Banse `[通讯]` (Fraunhofer AISEC)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了一种基于本体的安全度量开发方法及数据模型，支持自动化合规检查与协作式度量构建。

**💡 创新点**

创新点在于将高层合规需求分解为标准化组件并映射至统一本体，实现统一抽象层次的度量设计，同时提供半结构化描述与 Rego 执行层双重表达。

**🔧 技术方法**

采用本体建模（OWL/RDF）、Rego 策略语言、半结构化描述格式、GitHub 开源仓库以及与 OSCAL 的互联等技术。

**📊 数据集**

以欧盟多项安全标准（CRA、EUCS、C5 等）、CIS Benchmarks 与云资源描述（如 Azure REST）为示例数据集。

**📈 对比分析**

通过在 Confirmate 平台上对比传统人工审核与基于度量的自动检查，评估覆盖率与执行效率，结果表明可实现持续合规检查，性能尚未量化但系统具可扩展性。

**⚠️ 局限性**

局限性包括 Goodhart 定律导致度量被过度优化、无法完整覆盖所有语义丰富的文本要求，以及对非自动可测属性仍需人工干预。

---

## 479. When is more search useful? An Empirical Study of Search Utility in Quality-Aware Decoding for Machine Translation

**arXiv ID:** 2610.32679 | [PDF](https://arxiv.org/pdf/2610.32679v1)

**作者:** Javier Garcia Gilabert `[一作]` (University of Zurich), André F. T. Martins `[通讯]` (Universidade de Lisboa)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了质量感知解码（QAD）中不同源句子对候选搜索扩展的效益差异，并量化了总扩展收益、边际收益和饱和点。

**💡 创新点**

创新点在于将搜索效用拆解为总扩展收益、边际收益和饱和点三种指标，并利用句子级特征与候选池统计预测这些指标，从而实现针对性候选扩展。

**🔧 技术方法**

使用的技术包括QAD中的MBR与基于质量的重新排序、语义与词汇多样性、模型内部不确定性、目标函数稳定性等预测信号，以及对候选集的随机采样。

**📊 数据集**

实验数据集为960句测试集，覆盖en→{es,ru,de}三语对，并采用多款Qwen3模型与一款多语种MT模型。

**📈 对比分析**

在BLEU/Comet指标下，基于语义多样性的选择策略在MBR上接近oracle性能，明显优于随机扩展；相较于reranking，其预测效果更弱。

**⚠️ 局限性**

局限性包括：对指标的依赖导致结果与评估度量对齐；仅覆盖少量语言对、模型和采样策略；未给出完整的动态预算分配方案。

---

## 480. A Linear Lower Bound for Convex Body Chasing in Every $\ell_p$ Norm

**arXiv ID:** 2610.32670 | [PDF](https://arxiv.org/pdf/2610.32670v1)

**作者:** Nicolas Christianson `[一作]` `[通讯]` (Johns Hopkins University), Nicolas Christianson (Johns Hopkins University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文证明了在任何 ℓ_p 范数（1 ≤ p < ∞）下，凸体追踪问题的竞争比至少为 Ω(d)，即与维度成线性关系。

**💡 创新点**

创新点在于将 Bubeck‑Coester‑Rabani 的递归构造与凸体追踪相结合，设计了一种新的映射 Ψ，将离散请求映射到 (2r+1)-维向量空间的凸包，从而保持距离并控制运输成本，最终实现了对所有 ℓ_p 范数的线性下界，弥补了此前仅在 p = ∞ 下已知的线性下界。

**🔧 技术方法**

主要技术包括：
1. 递归图构造（Bubeck‑Coester‑Rabani 的 MSS 低界）；
2. 将图顶点映射到向量空间的深度及分支函数；
3. 证明映射保持 Lipschitz 性和运输成本上界；
4. 通过 Wasserstein‑1 距离与 ℓ_p 范数等价性完成成本比较；
5. 结合已知的功能 Steiner 点算法获得匹配上界。

**📊 数据集**

本工作为纯理论研究，未使用任何真实数据集；所有分析均基于构造的随机请求序列和离散图。

**📈 对比分析**

通过构造的凸体追踪实例，对任意在线算法的期望移动成本与离线最优成本比较，得到 c₀ r² Λ / (8 D^{1-1/p}) 的下界；同时利用功能 Steiner 点算法在任何 ℓ_p 范数下的竞争比 d 作为上界，证明在线竞争比至少为 Ω(d)，并与上界匹配（常数因子）。

**⚠️ 局限性**

局限性：
- 仅给出了下界，未给出新的更优在线算法；
- 证明依赖于递归图构造，可能不易直接推广到更一般的几何约束或非欧几里得空间；
- 结果以理论上限为准，缺乏实验验证；
- 对 p = ∞ 的上界已知，而本文仅说明对所有 p 的下界与上界相匹配，尚未讨论在极端 p 取值（如 p 接近 1 或 ∞）时的细节差异。

---

## 481. WL-SeqVAE: Sequence-Based Graph Generation Using Variational Autoencoders and Weisfeiler-Leman Colors

**arXiv ID:** 2610.32650 | [PDF](https://arxiv.org/pdf/2610.32650v1)

**作者:** Iakovos Evdaimon `[一作]` (École Polytechnique, IP Paris), Michalis Vazirgiannis `[通讯]` (École Polytechnique, IP Paris)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种将图生成转化为序列生成的WL-SeqVAE模型，利用 Weisfeiler–Leman（WL）算法得到的节点颜色作为结构令牌进行编码与解码。

**💡 创新点**

创新点在于用 WL 颜色取代直接学习邻接矩阵，既减少了二次复杂度，又通过在潜在空间和局部 WL 约束的双重引导，实现全局结构信息与局部一致性的统一。

**🔧 技术方法**

技术手段包括变分自编码器（VAE）框架、Transformer 编码器、基于 WL 颜色的自回归解码器、兼容性约束以及重参数化技巧。

**📊 数据集**

实验使用了四个抽象图数据集（Ego‑small、Community‑small、Enzymes、Grid）和两个分子图数据集（QM9、ZINC250K）进行评估。

**📈 对比分析**

与 GraphVAE、GNF、EDP‑GNN、GDSS、CatFlow 等基线相比，WL‑SeqVAE 在度分布、聚类和轨道统计等指标上表现相当或略优，但在某些全局结构更强的图上略逊；在分子生成中有效率高于多数基线，唯一性和 FCD 与 CatFlow 相近或稍差。

**⚠️ 局限性**

局限性包括：生成的图在 WL 约束下并非始终完备（WL‑complete 率不高）、对全局结构捕捉能力有限、分子生成的多样性与分布匹配度仍低于最强基线。

---

## 482. Internal Discrepancy for Contrastive Learning: Generalization Bounds and Downstream Behaviors

**arXiv ID:** 2610.32729 | [PDF](https://arxiv.org/pdf/2610.32729v1)

**作者:** Jialiang Wang `[一作]` (Harbin Institute of Technology), Xianming Liu `[通讯]` (Harbin Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出内部差异度量，并基于此给出一种无需额外假设的对比学习下游监督任务的泛化上界，同时设计了松弛 k‑means 损失以近似降低内部差异，并对 InfoNCE 与 SpeCL 产生的监督损失在类别不平衡与最佳性方面进行了理论分析。

**💡 创新点**

创新点在于将各类先前的数据与函数类假设统一为内部差异度量，利用该度量得到的泛化界为对比学习提供新的理论工具，并通过松弛 k‑means 与 InfoNCE‑/SpeCL‑产生的监督损失解释对不平衡学习的优势及其“神经崩塌”特征。

**🔧 技术方法**

技术手段包括凸、Lipschitz 对比损失的泛化分析、内部差异与正负对齐、类内不一致度的分解、松弛 k‑means 损失的设计、以及对 InfoNCE 和 SpeCL 所产生监督损失的近似与最优性证明。

**📊 数据集**

实验数据集涵盖 CIFAR‑10、CIFAR‑100、ImageNet‑100 及其长尾版本，用于评估线性探针和监督损失在不同负样本数量与类别不平衡比例下的性能。

**📈 对比分析**

与传统交叉熵、平衡 softmax 以及 SpeCL‑产生的监督损失对比，InfoNCE‑产生的损失在大负样本数和高不平衡比下表现最好，整体性能与交叉熵相近或略优，SpeCL‑产生的损失在归一化后亦能保持良好性能。

**⚠️ 局限性**

局限性包括：泛化上界依赖于凸、Lipschitz 损失与范数约束；内部差异度量需要潜在聚类中心，实际可观测性有限；松弛 k‑means 仅近似控制类内散度，可能无法完全捕捉复杂聚类结构；实验规模相对较小，未在大规模真实数据上验证。

---

## 483. Beyond Hotness: High-Resolution Memory Characterization of Cloud and LLM Workloads for Tiered Memory with ARM SPE

**arXiv ID:** 2610.32711 | [PDF](https://arxiv.org/pdf/2610.32711v1)

**作者:** Jie Ren `[一作]` (William & Mary), Ivy Peng `[通讯]` (KTH)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于ARM SPE的高分辨率内存特征化方法，评估了八个云/LLM工作负载的内存行为，并利用新指标指导多层内存迁移；

**💡 创新点**

将ARM SPE的指令采样与层级事件相结合，提出了基于内存空闲期和延迟分布的新指标，用以评估工作负载在多层内存系统中的适配性；

**🔧 技术方法**

使用ARM SPE（Statistical Profiling Extension）、低开销采样工具、SPE驱动的数据迁移运行时，并与Linux内核级工具DAMON做对比；

**📊 数据集**

采用CloudSuite工作负载（PageRank、Connected Components、Triangle Counting、Memcached、Cassandra、In-Memory Analytics）与LLM（Llama 8B 推理、GPT-2 124M 预训练、Qwen3-30B）以及STREAM、RandomAccess、MASIM等基准；

**📈 对比分析**

与DAMON在精度（precision/recall）和运行时开销对比，SPE实现低于0.6%–4.5%开销且精度更高；在多层内存迁移实验中，针对延迟偏斜且空闲期长的工作负载（如Cassandra、PageRank）提升10–30%性能，而对Llama、Memcached无显著提升；

**⚠️ 局限性**

采样周期和窗口大小对热点召回影响较大，低频或少量热点导致召回不足；对空闲期短的工作负载迁移效果差；目前仅在ARMv8.2+平台验证，跨架构适用性待进一步验证。

---

## 484. Are Textual Residuals Truly Complementary? Learning Visual Complements under Semantic Superposition for Visible-Infrared Person Re-Identification

**arXiv ID:** 2610.32709 | [PDF](https://arxiv.org/pdf/2610.32709v1)

**作者:** Xiaoyang Li `[一作]` (Central South University), Canhui Zhang `[通讯]` (Central South University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种基于文本残差校准的可见-红外行人重识别方法SARC，先将文本差异稀疏化并去除干扰，再通过视觉补偿与对抗式校准提升残差的视觉可补性。

**💡 创新点**

创新点在于把文本差异视作可选择的稀疏特征强度，并在低干扰学习与身份级视觉补偿相结合的框架下显式校准残差，避免直接稠密融合带来的语义混叠。

**🔧 技术方法**

使用共享字典的稀疏编码、Top‑k稀疏化、低干扰正则、视觉补偿对齐损失、对抗式残差校正、以及CLIP文本编码与ResNet‑50视觉编码。

**📊 数据集**

在三大文本增强的跨模态数据集上评测：Tri‑SYSU‑MM01、Tri‑RegDB 和 Tri‑LLCM。

**📈 对比分析**

与目前主流文本辅助VI‑ReID方法对比，SARC 在所有三个数据集上实现了排名第一或第二的 Rank‑1/mAP/mINP，尤其在 Tri‑SYSU‑MM01 的 indoor 搜索中超过 TVI‑LFM 约 0.7‑1.1pp，并保持极低的参数和推理延迟增量。

**⚠️ 局限性**

局限在于依赖身份相关的 RGB 文本描述，若无此辅助信息或出现身份不匹配时效果仍受限，且对极端文本噪声仍会产生一定性能下降。

---

## 485. CounselPlex: A Stressor-Aware Full-Duplex Counseling Model

**arXiv ID:** 2610.32702 | [PDF](https://arxiv.org/pdf/2610.32702v1)

**作者:** Yeonju Kim `[一作]` (KAIST), Yong Man Ro `[通讯]` (KAIST)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了CounselPlex框架，在全双工语音模型中实现压力源持续追踪与注入，以提升咨询质量。

**💡 创新点**

通过在说话者沉默期生成并在听话期间注入短压力量化表征，以及RoPE位置偏移增强技术，解决了“压力源盲区”问题。

**🔧 技术方法**

使用PersonaPlex/Moshi双流全双工模型、Silero VAD、Mimi编码器/解码器、RoPE位置偏移增强（RPSA）以及LLM-judge（GPT‑4.1、Gemini‑2.5‑Flash）评估。

**📊 数据集**

将ESConv和ESCoT文本情感支持数据通过CosyVoice3合成语音，得到约158小时、3009会话的音频数据，并手工标注1–7词压力量化。

**📈 对比分析**

在GPT‑4.1/ Gemini评估下，CounselPlex在共5个咨询维度上均超过PersonaPlex及其fine‑tuned版，平均5维评分提升约0.3分，响应率保持可接受、延迟略增，RPSA显著提升长上下文性能。

**⚠️ 局限性**

实验基于合成语音和LLM评估，未验证临床效果；单一压力量化状态可能忽略多重问题；全双工时缺乏自然回声通道，延迟略高且未恢复真实交互的回声和重叠。

---

## 486. Tile-Based Decompression in Compiled GPU Engines

**arXiv ID:** 2610.32695 | [PDF](https://arxiv.org/pdf/2610.32695v1)

**作者:** Antonio Boffa `[一作]` (EPFL), Anastasia Ailamaki `[通讯]` (EPFL)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在 GPU 查询引擎 Proteus 中实现了一个可组合的压缩子系统，将解压缩抽象为第一级物理操作符，并在同一核内将解压与 tuple‑at‑a‑time 逻辑融合，避免在 HBM 上产生完整解压后的中间表。

**💡 创新点**

创新点在于：
1) 把解压缩完全隔离到统一的两条后端 hook（元数据解析与 tile 解码）中，使得任何新的 GPU 压缩格式都只需实现这两条接口即可；
2) 通过共享内存、寄存器或全局内存三种解压放置方式，并将其作为可调节的编译时参数；
3) 在编译阶段将解压过程与查询管道融合，利用 LLVM/NVPTX 生成单核内核，减少 HBM 传输与中间存储。

**🔧 技术方法**

使用技术包括：
- Proteus 及其 LLVM/NVPTX JIT 编译框架；
- GPU 共享内存、寄存器、全局内存调度；
- 轻量级整数压缩编码（GPU‑RFOR、FastLanes‑BP）；
- 两级 hook 设计（metadata 解析 + tile 解码）；
- Nsight Compute 等工具对 GPU DRAM 流量与寄存器占用进行测评；
- 实验平台为 NVIDIA A40（48 GB GDDR6）和 A100（40 GB HBM2）。

**📊 数据集**

实验使用的数据集：
- TPC‑SSB 100（SF 100）
- TPC‑H SF 100
- StackExchange 真实查询（Stack）
- 所有字符串列统一使用字典编码为 4 字节整数 ID。

**📈 对比分析**

比较方法与性能：
- 对四种解压策略（未压缩、全局解压、bounded‑global、shared‑memory、register）与 Proteus 基线进行 DRAM 读写、查询时间对比；
- 与 HeavyDB（已优化的 tuple‑at‑a‑time）和 Sirius（vectorized libcudf）在同一 GPU 上的运行时间与显存占用做基准；
- 结果显示：共享内存解压在所有查询中最快，显存占用可降低 3–4×，HBM 带宽消耗可降低 4–6×；
- 在相同压缩率下，Proteus 的查询时间与 HeavyDB 相当，优于 Sirius；
- 与专用 Crystal 版压缩核比较，Proteus 在不同查询形态下表现更稳定，压缩后性能波动更小。

**⚠️ 局限性**

局限性：
- 仅在单 GPU 内部评估，未考虑跨 GPU/CPU 的压缩/解压交叉点；
- 只支持轻量级整数压缩，未覆盖变长压缩或深度学习推理场景；
- 解压放置仍需手动调参，无法完全自动化；
- 需要为每种压缩格式实现 hook，若格式变更需重新实现；
- 在高列数查询中共享内存压力导致占用提升，可能需要进一步优化。

---

## 487. Natural Language Autoencoders Identify Without Explaining

**arXiv ID:** 2610.32691 | [PDF](https://arxiv.org/pdf/2610.32691v1)

**作者:** Kristina Šekrst `[一作]` (University of Zagreb), Barbara Geld `[通讯]` (Vienna University of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究自然语言自动编码器（NLA），阐明其仅能保证文本能重构激活向量，不能证明解释的真实性。

**💡 创新点**

创新点在于系统地拆分解释为识别部分与描述部分，证明重构评分只奖励识别，提出独立性、引用性、风险三条限制条件。

**🔧 技术方法**

使用热启动、KL 正则化、强化学习训练的激活可读化模型（AV）与重构模型（AR），并在大语言模型上进行对齐。

**📊 数据集**

数据集为Claude Haiku 3.5/4.5/Opus 4.6等模型在各层的激活向量，配合手工生成的上下文摘要进行训练。

**📈 对比分析**

与归因图、干预测试等传统解释方法对比；FVE（方差解释率）在0.6–0.8之间，但在对齐与安全性任务中缺乏可靠性验证。

**⚠️ 局限性**

局限性包括：缺乏对“描述性”真实性的独立验证、容易产生伪造（confabulation）、对处理过程的断言难以检验，因而不宜直接用作安全审计或对齐工具。

---

## 488. Retiming Robot Demonstrations with a Learned Speed Prior

**arXiv ID:** 2610.32676 | [PDF](https://arxiv.org/pdf/2610.32676v1)

**作者:** Seungyong Moon `[一作]` (Seoul National University), Hyun Oh Song `[通讯]` (Seoul National University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种速度先验模型，用于从大型演示数据中学习机器人执行速度分布，并将该先验应用于重定时演示，从而消除操作者导致的慢速和停顿，提高任务成功率与执行平滑度；

**💡 创新点**

创新点在于：①一次性训练跨任务的速度先验，能够从快速演示（如手持演示或最短四分之一演示）迁移节奏到所有演示；②通过量化时钟与量化分位数的方式，使策略能够在部署时动态加速；③提出了基于速度先验的实时控制器，兼顾机器人动力学限制并实现无跳跃平滑连接；

**🔧 技术方法**

使用Transformer结构（包含指令、图像和节点嵌入）进行速度分布预测；训练使用交叉熵对数速度类别；重定时采用量化分位数；策略微调采用π_0.5 Vision‑Language‑Action模型；控制层采用Draft‑Splice‑Smooth三步时间最优规划；

**📊 数据集**

数据集包括：Teleoperation演示（DROID），手持演示（FastUMI‑100K），以及自收集的YUBI手持演示；

**📈 对比分析**

与统一加速、过滤快速演示BC、默认策略以及SAIL等基线比较；在120个RoboLab仿真任务和3个真实机器人任务上评估，结果显示：在相同加速倍数下，本方法成功率提升约4.9个百分点，TPR提升0.88成功/分钟，SPARC更平滑，峰值速度与冲击降低；在实际机器人中，杯子堆叠任务速度提升4.2×、成功率>2×；

**⚠️ 局限性**

局限性在于速度先验只能迁移已观察到的节奏，无法超越演示中出现的最大速度；并且对不同机器人/抓手的动力学约束需要进一步调整；

---

## 489. Do Language Models Trust Their Own Justifications? A Study on Functional Consistency

**arXiv ID:** 2610.32665 | [PDF](https://arxiv.org/pdf/2610.32665v1)

**作者:** Alisson Rosa Pereira `[一作]` (Universidade Federal do Rio Grande do Sul), Bruno Iochins Grisci `[通讯]` (Universidade Federal do Rio Grande do Sul)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了大型语言模型自生成解释的功能一致性，通过对模型标记的关键词进行保留、删除和语义反转等干预，检验其对预测评分的影响。

**💡 创新点**

提出了基于自动一致性原理的评估框架，首次将充分性（Sufficiency）、完整性（Comprehensiveness）和反事实性（Counterfactuality）三项度量统一用于检验LLM自解释的行为一致性。

**🔧 技术方法**

使用了功能一致性评估技术，包括系统化的文本干预（保留、删除、语义反转）、定量度量（ΔR、π_alt、π_red等）以及与GPT-4o-mini、Gemma3、Granite8B、DeepSeek系列等多种LLM的交互。

**📊 数据集**

采用IMDB电影评论数据集和Steam游戏评论数据集，共计2000条句子，进行情感分类与评分预测实验。

**📈 对比分析**

通过平均评分差异、方差、标签翻转比例及方向性评分下降等四个汇总指标比较不同模型，结果显示Gemma3、GPT-4o-mini、Granite8B在三项指标上表现出良好的功能一致性，而DeepSeek 1.5B/14B 在充分性或完整性方面显著偏离，说明模型规模与家族差异对自解释一致性影响显著。

**⚠️ 局限性**

局限性包括仅针对两种情感文本数据集，干预粒度局限于词级，反事实生成仅依赖WordNet或简单否定，未与随机或长度匹配基线进行对照，且仅使用单一提示模板，未来需扩展任务、数据和更自然的反事实生成方法。

---

## 490. The Answer Knows Where to Look: Zooming before MLLMs Answer under Visual-Token Budgets

**arXiv ID:** 2610.32664 | [PDF](https://arxiv.org/pdf/2610.32664v1)

**作者:** Zhensheng Hu `[一作]` (Southern Medical University), Guanqiang Hu `[通讯]` (Guangdong Kamfu Technology Co., Ltd.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

为多模态大语言模型在视觉 token 预算受限的情况下实现文本阅读，提出一种专用的“look token”，在回答前通过预填充提取证据定位信息并决定合适的放大视图。

**💡 创新点**

创新点在于：①将模型自身答案时的注意力通过 hindsight distillation 转化为集中指针，②把指向与阅读分离，③在预填充阶段就决定放大方式（无裁剪、固定/自适应窗口、视差变形）并用 counterfactual supervision 学习最佳视图。

**🔧 技术方法**

使用的技术包括 LoRA 细化预填充层注意力、KL 散度训练指针、对每个视图的 counterfactual 评分进行全信息策略梯度学习、GRPO 训练阅读器。

**📊 数据集**

实验数据集涵盖 DocVQA、InfographicVQA、ChartQA、TextVQA 以及 V‑CoT‑Doc，用于评估在 98、196、392 个视觉 token 的不同预算下的表现。

**📈 对比分析**

与 ViCrop、AttWarp、Ground‑then‑crop、GapSight、DeepEyes 等最强训练‑无关或基于注意力/定位的放大方法比较，Qwen3‑VL‑4B 在 98 tokens 时平均提升 4.8–9.1 分，在 196 tokens 时提升 5.5–4.6 分，整体速度快于注意力引导和基于定位的放大。

**⚠️ 局限性**

局限性包括：仅在单视图下实现；仍受 token 预算限制；对非文本密集图像的效果未验证；需要在每个模型上训练专用 look token，尚未支持多步多图或视频场景。

---

## 491. SemGeo-Gen: Unsupervised Generation of Approximate Cross-Instance Semantic-Geometric Correspondences

**arXiv ID:** 2610.32663 | [PDF](https://arxiv.org/pdf/2610.32663v1)

**作者:** Roy Amoyal `[一作]` (Ben Gurion University Negev), Oren Freifeld `[通讯]` (Ben Gurion University Negev)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了一个无监督、可扩展的管道，利用大规模近似对齐的3D资产生成跨实例的语义几何对应关系，提供预训练信号给真实图像对应模型；

**💡 创新点**

核心创新在于无需手工点级标注即可从已对齐的3D集合中生成成对对应关系，并且可组合性极强（从N个对象可产生N²对对应），同时结合语义过滤与几何约束实现高质量对应；

**🔧 技术方法**

采用CPA（Conditional Point Attention）变形模型进行几何对齐，使用DINOv2特征提取器进行语义过滤，结合多视角渲染生成3D-3D、投影3D-2D以及2D-2D对应；

**📊 数据集**

使用KeypointNet、SPair-71k等标注数据验证对应质量，并依托Objaverse-OA等大规模已对齐3D资产作为原始数据来源；

**📈 对比分析**

通过在KeypointNet上验证3D-3D与投影3D-2D对应，在SPair-71k上评估2D-2D匹配效果，结果显示无监督生成的对应显著提升匹配准确率，并能泛化到非刚性类别；

**⚠️ 局限性**

局限包括：CPA模型难以完全捕捉大跨实例结构差异；语义过滤依赖DINOv2，未来可换用更强特征；方法假设对象刚性且已对齐，无法直接处理非刚性姿态；合成3D资产与真实图像之间存在域差距，需进一步对齐真实3D资产以减小差异。

---

## 492. StateMirror: Pre-Execution Detection of Hallucinated Tool Calls in Coding Agents

**arXiv ID:** 2610.32648 | [PDF](https://arxiv.org/pdf/2610.32648v1)

**作者:** Shiwen Shan `[一作]` (Sun Yat-sen University), Yuxin Su `[通讯]` (Sun Yat-sen University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种预执行检测器 StateMirror，能够在执行前提取并验证编程代理的工具调用是否符合当前工作区的实际环境，防止幻觉式工具使用。

**💡 创新点**

创新点在于将动作中的环境前置条件通过静态分析抽取为谓词，并使用确定性、无执行、无token消耗的探测器在毫秒级完成预检查，从而实现低成本的 fail‑fast 机制。

**🔧 技术方法**

采用 AST 解析、命令和代码分析、谓词抽取与探测、规则匹配等技术，并将其集成到 Python 工具中，与现有的 SWE‑agent 等框架协同工作。

**📊 数据集**

使用了来自 SWE‑bench‑Extra 的 4,398 条轨迹构建的 2,508 条手工标注的预执行幻觉实例，并在 4 个不同代理框架（SWE‑agent、OpenHands、Sonar、Honeycomb）上进行评估。

**📈 对比分析**

通过与 LLM 判别器、Action‑only、Scaffold Linter 三种基线对比，StateMirror 在 2,508 条动作上达到 0.940 的精度、0.817 的 F1，超过最强 LLM 判别器 35.3% 的 F1；检测延迟仅 1.5 ms、无 token 消耗；在任务级干预中将执行时间降低 15.8% 及 token 消耗降低 11.8%。

**⚠️ 局限性**

局限性包括：仅覆盖静态文件系统和命名空间等有限谓词，对动态运行时错误缺乏覆盖；需要为其他编程语言或代理运行时实现新的谓词抽取器；在已检测到的错误中仍存在一定漏报率。

---

## 493. Direction-Aware Token Credit Assignment with Reward-Model Gradients

**arXiv ID:** 2610.32772 | [PDF](https://arxiv.org/pdf/2610.32772v1)

**作者:** Yikuan Hu `[一作]` (Nanjing University), Wangzhou Dai `[通讯]` (Nanjing University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于工作量的Token信用分配方法DATC，用以改进GRPO的token‑level学习更新。

**💡 创新点**

创新点在于将奖励模型梯度与GRPO的更新方向结合，采用logit空间的工作量度量并通过KL正则实现局部奖励增益保证。

**🔧 技术方法**

采用直通（Straight‑through）奖励梯度、GRPO、KL正则化权重分配、token级优势重分配、logit空间内的内积计算等技术。

**📊 数据集**

使用UltraFeedback的偏好数据与Skywork奖励模型，评估时使用RM‑Reward、AlpacaEval、Arena‑Hard v2.0、MT‑Bench四个指标。

**📈 对比分析**

与GRPO、RED、P2T等方法对比，DATC在Qwen3‑14B与Llama‑3.1‑8B的Base与Instruct四种设置中均在四项指标上均优于基线，提升幅度最高可达约12个百分点。

**⚠️ 局限性**

局限性包括对奖励模型的依赖、需要额外的KL正则调参、在不同模型/任务上泛化性尚未验证，且GPU时间略有提升。

---

## 494. CityLift: Compositional 3D Urban Generation from a Single Image

**arXiv ID:** 2610.32728 | [PDF](https://arxiv.org/pdf/2610.32728v1)

**作者:** Haoran Gong `[一作]`, Chenxu Wang `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `f86bf285-fd08-4156-973b-6e6481af8fa0` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

生成单幅倾斜城市图像及建筑实例掩码下的可编辑3D建筑网格，构建完整城市场景。

**💡 创新点**

提出CityLift框架，先通过场景级别的城市支架生成粗糙布局和建筑占位，再通过生成空间遗传将支架映射到局部体积，结合条件流模型实现细化建筑生成，实现场景一致性与局部细节并行。

**🔧 技术方法**

联合稀疏结构DiT、条件流模型、Treillis生成器、SAM 3D的SLAT与网格解码、生成空间遗传的特征传递以及预训练的DINOv2图像编码器。

**📊 数据集**

构建CityLift-5K数据集（4997合成城市块、43239建筑实例、图像、掩码、完整网格与位置信息），并在UrbanScene3D与InstanceBuilding上进行外部评估。

**📈 对比分析**

与TRELLIS.2、SAM 3D、Pixal3D、3D‑Fixer、SceneGen等基线在建筑层面和场景层面Chamfer距离、精度、完整度及F-score进行比较，CityLift在CityLift‑5K、UrbanScene3D和InstanceBuilding上均取得最低Chamfer距离、最高F‑score，整体性能提升超过30%，且生成速度保持在115 s/场景左右。

**⚠️ 局限性**

依赖合成训练数据导致对真实场景的泛化仍有限；高度与比例预测受视角遮挡和相机参数影响；生成空间遗传的精度决定了局部细节质量。

---

## 495. Ontology-Guided Knowledge Graph Construction from Structured Metadata and Climate Science Publications

**arXiv ID:** 2610.32725 | [PDF](https://arxiv.org/pdf/2610.32725v1)

**作者:** Aayush Acharya `[一作]` (Temple University), Longin Jan Latecki `[通讯]` (Temple University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建了一个基于证据和本体论的气候科学知识图谱（KG），通过多代理LLM系统从2000篇论文中抽取、验证并链接三元组。

**💡 创新点**

创新点在于：①引入显式文本证据以防止hallucination；②在提取前后分别进行本体一致性检查、证据检验和最终验证；③采用混合实体链接策略，将相似度阈值与LLM选择相结合。

**🔧 技术方法**

使用的技术包括Gemini-2.5-flash LLM（文档理解、三元组提取、验证）、自定义本体（ESGF、CMIP6、GCMD、GeoNames）、NV-Embed-V2嵌入模型、层级验证流程和Python实现的多代理管线。

**📊 数据集**

数据集为2000篇气候科学论文（基于IPCC、CMIP6公开论文并通过引用扩展），以及ESGF元数据、CMIP6控制词汇表、GCMD与GeoNames构成的本体。

**📈 对比分析**

与KARMA和ClimateIE基线对比，验证阶段完整系统在2000篇论文上抽取到23,327条三元组，准确率达98.96%，相比KARMA的92.28%准确率和仅3,820条输出，或ClimateIE的75%准确率更高；在前1,007篇论文上，与KARMA相比，抽取量提升近5倍，准确率提升约7个百分点。

**⚠️ 局限性**

局限性包括：对验证阶段的GPU开销约为提取量的1%；未对三元组级召回进行评估，只能基于外部元数据进行实体召回；LLM验证器可能仍会误接受隐含关系或因分散证据而误判。

---

## 496. Conditional Neural Processes for Self-Supervised Goal Discovery in Action Imitation

**arXiv ID:** 2610.32688 | [PDF](https://arxiv.org/pdf/2610.32688v1)

**作者:** Marco Gabriele Fedozzi `[一作]` (University of Genoa), Alessandra Sciutti `[通讯]` (Italian Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

评估并比较不同版本的多模态生成网络（DMBN、AE、DMBN‑PTE、AttnDMBN），研究条件神经过程在自监督目标发现和动作模仿中的表现。

**💡 创新点**

在CNP隐藏层引入注意力机制后，能够在观察动作序列早期就形成稳健的目标表示，证明了预测任务对自监督动作表征的重要性。

**🔧 技术方法**

使用条件神经过程（CNP）、注意力层、位置时间编码、生成式深度网络以及softargmax Bernoulli的分类头。

**📊 数据集**

利用模拟机械臂抓取或移动物体的第一人称图像和关节角度数据集，数据集与原始DMBN论文相同。

**📈 对比分析**

通过在冻结的编码器后接分类头，对从动作开始或结束的部分序列进行“move/ grasp”分类评估，比较四种网络的误差曲线。AttnDMBN在观测序列较短时误差显著下降，优于AE和原始DMBN；DMBN仅在观测到动作末尾时表现较好。

**⚠️ 局限性**

限制：普通CNP无法捕获目标信息；DMBN‑PTE缺乏时间信息导致表示不佳；实验仅在模拟环境中进行，缺乏真实世界验证；未探究不同目标类别或更复杂动作的泛化能力。

---

## 497. Identity Amortization: Shared Resources and the Cost of Sybil Attacks

**arXiv ID:** 2610.32685 | [PDF](https://arxiv.org/pdf/2610.32685v1)

**作者:** Homayoun Maleki `[一作]` (University of Deusto), Igor Santos-Grueiro `[通讯]` (International University of La Rioja)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出并正式化了“身份摊销”（identity amortization）的概念，给出了判定是否出现摊销的算法，并在电话验证与 Tor 隐藏服务目录的实际系统中评估了摊销对 Sybil 攻击成本的影响。

**💡 创新点**

创新点在于：①首次将经济学的规模经济与算法的摊销分析结合，定义身份摊销并阐明其与资源共享、身份计数的关系；②提供了基于资源价格与容量的判定与计算公式，能够准确预估摊销的起始点、误差幅度和可缓解的共享上限；③通过定量实验验证，说明摊销在真实系统中的显著效应。

**🔧 技术方法**

技术手段包括：构造资源共享与独立提供的成本模型、证明身份摊销与成本保证的等价关系、推导容量模型下的成本公式与误差界限、使用动态规划求解最优策略、以及对抗性重新优化的分析。

**📊 数据集**

使用的数据集为：2013‑2014 年 Google 电话验证的代码价格、SIM 卡价格、身份特定成本；电话验证与 Tor 网络的测量数据（如隐蔽服务目录数量、IPv4 地址限额、日均中继数）；以及 2024‑2025 年平台平均验证价格的最新引用。

**📈 对比分析**

评估方法是：在给定资源目录的前提下，利用公式计算独立与共享成本，得到误差因子 E；对电话验证系统进行动态规划验证，得到 1.6‑4.3 倍的成本低估；对 Tor 进行实验验证，显示将每地址中继上限从 2 提升到 8 可使 25% 隐藏服务目录的攻击成本降低 8%‑72%。这些结果与理论极限（如 3r/(4(r+2))）高度一致。

**⚠️ 局限性**

限制包括：模型仅为静态、一次性成本的分析，未考虑随时间演变的资源利用、合法多身份使用的成本分摊、以及实际攻击成本的测量；资源价格假设线性且未覆盖所有可共享资源（如 VOIP 号码、代理等）；对共享能力的假设（如容量上限）来自有限测量，可能与真实情况存在偏差。

---

## 498. Iterative Behavioral Identification for Covert Misappropriation Attacks

**arXiv ID:** 2610.32662 | [PDF](https://arxiv.org/pdf/2610.32662v1)

**作者:** Victor Mattos `[一作]` (Universidade Federal do Rio de Janeiro), Alan O. de Sá `[通讯]` (Universidade de Lisboa)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种数据驱动的三阶段框架，用于对闭环网络控制系统进行隐蔽性盗用攻击。攻击者通过拦截输入输出数据，逐步识别植物的行为子空间，同时在无限范数入侵检测系统下保持隐蔽性。

**💡 创新点**

创新点在于提出了一个无需先验模型知识的隐蔽攻击框架，结合了探测、识别和行为攻击合成，解决了现有方法在模型知识和隐蔽性方面的不足。

**🔧 技术方法**

使用了数据驱动控制、预测控制和深度学习控制（DeePC）风格的优化技术。

**📊 数据集**

在模拟实验中使用了一个二阶离散时间直流电机的控制系统，攻击者没有访问植物和控制器的矩阵，仅通过拦截的数据进行攻击。

**📈 对比分析**

与现有方法相比，该方法在隐蔽性和样本复杂度上表现出色，能够在较少的样本下实现全秩识别和隐蔽跟踪。性能通过数值模拟验证，显示出在每个秩增加时，投影误差单调减少。

**⚠️ 局限性**

限制在于所有分析假设系统为无噪声LTI动态，噪声的存在可能导致攻击效果下降。此外，当前的入侵检测系统模型较弱，未来需要考虑更复杂的检测策略。

---

## 499. Strongly Adaptive Online Learning with Time-Varying Movement Costs

**arXiv ID:** 2610.32740 | [PDF](https://arxiv.org/pdf/2610.32740v1)

**作者:** Andrew Jacobsen `[一作]` (Università degli Studi di Milano), Mengxiao Zhang `[通讯]` (University of Iowa)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文针对在线凸优化（OCO）中存在时间变化的移动成本（switching penalty）的问题，提出了能够在任意时间子区间上同时获得强自适应（strongly‑adaptive）和动态（dynamic）遗失上界的算法，并进一步将该框架应用于延迟反馈与时间变化记忆长度的 OCO 任务，给出了对应的强自适应遗失上界。

**💡 创新点**

创新点：
1. 首次在含时间变化移动成本的 OCO 中实现强自适应遗失上界；
2. 将睡眠专家（sleeping‑experts）与移动成本相结合，给出具有数据自适应的二阶上界；
3. 提出统一的移动成本归约方法，可将延迟反馈与记忆问题映射为移动成本问题，从而得到首个强自适应/动态上界；
4. 采用几何覆盖区间 + 首次顺序阻塞技术，实现在任意子区间上的自适应控制。

**🔧 技术方法**

主要技术：
- 元‑基（meta‑base）分解与移动成本归约；
- 睡眠专家（sleeping‑experts）算法并考虑移动成本；
- FTRL 与拉格朗日变分实现二阶自适应上界；
- 几何覆盖区间与首次顺序阻塞（adaptive first‑order blocking）来处理子区间；
- 延迟‑移动成本映射与记忆‑移动成本映射。

**📊 数据集**

论文为理论研究，没有使用具体数据集；所有结果均为闭式理论上界。

**📈 对比分析**

与已有工作比较：
- 在无移动成本时，本文上界退化为 G√((D²+DP)|(b‑a)|)，与目前最优强自适应上界一致；
- 对比仅给出动态遗失或不含移动成本的强自适应结果，本文得到额外的 λ_t 依赖，且上界为 √((D²+DP)|(b‑a)|+V^{λ})，其中 V^{λ}=∑(g_t²+λ_t g_t)，在 λ_t≪g_t 时显著优于旧有 O(log(b)) 乘子；
- 在延迟反馈场景下，得到 G√((D²+DP)|(b‑a)|+d_{[a,b]})，比之前仅能得到全局 O(√((D²+DP)T+∑d_t)) 的动态上界更加细粒度；
- 在时间变记忆情形下，得到 O(B√(HG(D²+DP)|(b‑a)|)+GB²(D+P))，同时兼顾强自适应与记忆长度的最优依赖。

**⚠️ 局限性**

局限与未来工作：
- 需要预先知道全局最大移动成本 λ、最大延迟 σ 及最大记忆长度 B，限制了对区间局部信息的利用；
- 睡眠专家归约无法实现纯“一阶”移动成本的完全自适应；
- 延迟‑移动成本映射会引入区间外的延迟信息影响；
- 阻塞技术不易推广到睡眠专家框架中的完全一阶自适应上界。

---

## 500. Deep Learning-Based Prediction of HER2 Status from Whole-Brain MRI in Breast Cancer Brain Metastases

**arXiv ID:** 2610.32774 | [PDF](https://arxiv.org/pdf/2610.32774v1)

**作者:** Sara Early `[一作]` (University of Calgary), Nils D. Forkert `[通讯]` (University of Calgary)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本文利用深度学习模型从全脑T1加权MRI预测乳腺癌脑转移患者的HER2状态，并与传统radiomics及加分割的SFCN模型进行比较。

**💡 创新点**

创新点在于提出无需人工肿瘤分割即可通过全脑MRI捕获HER2相关影像特征的SFCN模型，并评估分割信息对预测性能的增益。

**🔧 技术方法**

使用技术包括简单全卷积网络（SFCN）、逻辑回归radiomics、堆叠特征提取、交叉熵损失、Adam优化器、五折交叉验证及ROC-AUC评估。

**📊 数据集**

实验基于公开的BCBM‑Radiogenomics数据集，共135例（70 HER2+，65 HER2-）的乳腺癌脑转移患者。

**📈 对比分析**

通过五折交叉验证比较三种方法，radiomics得到ROC‑AUC 0.602，T1w‑only SFCN得到0.667，T1w+mask SFCN得到0.692；尽管数值略有提升，但差异未达到统计学显著水平。

**⚠️ 局限性**

局限性包括样本量有限、缺乏外部验证、MRI与组织学HER2标注时间可能不一致、仅使用单一T1序列、以及仅预测二分类HER2+/-，未涉及HER2低/零等更细粒度分级。

---

## 501. On-Policy Self-Training: Data-Free Self-Improvement in Large Language Models

**arXiv ID:** 2610.32770 | [PDF](https://arxiv.org/pdf/2610.32770v1)

**作者:** Leon Sick `[一作]` (Google), Federico Tombari `[通讯]` (Google)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了一种仅基于粗粒度主题描述的自我训练框架 OPST，利用模型自生成问题与特权信息进行后训练，从而无需任何人工标注即可提升大语言模型与视觉语言模型的推理能力。

**💡 创新点**

核心创新在于：①把生成的答案作为教师上下文而非目标，借助自生成的推理轨迹提供密集监督；②引入伪验证与共识门双重过滤，剔除不确定或已掌握的样本；③将该框架推广到视觉领域，实现仅凭未标记图像自我训练。

**🔧 技术方法**

使用技术包括自我生成（topic‑seeded generator）、伪验证（answer re‑generation & paraphrase consistency）、共识门（多路 roll‑out 互相一致性检测）、On‑Policy 自蒸馏（student‑teacher 区分训练）、EMA 教师平滑、视觉 OPST（图像+问题生成、视觉特权信息）。

**📊 数据集**

实验数据集：Gemma 4 E2B/E4B 在文本推理基准（MMLU、GPQA、GSM8K、HellaSwag、ARC‑Challenge、MATH‑500、Winogrande）以及视觉问答基准（VQAv2、POPE）上进行评估；所有训练数据均为模型自生成，未使用任何人工标注。

**📈 对比分析**

与 SFT、GRPO、OPD/OPS、以及包含人工标注的基准相比，OPST 在相同自生成数据上平均提升 5–10 分；在不使用思考模式时可逼近全监督 OPS 的表现；在视觉 OPST 上 VQAv2 提升 6.9 分，POPE 亦获得显著改善。

**⚠️ 局限性**

局限性包括：只能放大已有知识，伪验证仅检测一致性而非真正确性；主题描述需人工挑选；实验仅在 2B/4B 小模型上验证，未探索更大模型或多任务场景。

---

## 502. SCDM-HTD: Spectral-Wise Conditional Diffusion Model for Hyperspectral Target Detection

**arXiv ID:** 2610.32768 | [PDF](https://arxiv.org/pdf/2610.32768v1)

**作者:** Dunbin Shen `[一作]` (Dalian University of Technology), Hongyu Wang `[通讯]` (Dalian University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了一种基于谱条件扩散模型的高光谱目标检测方法，将目标检测任务改写为生成检测图，利用背景与目标置信度分布进行学习；

**💡 创新点**

创新点包括：①将HTD转化为生成任务，直接学习背景与目标置信度的分布差异；②提出轻量化的多层感知器条件去噪网络（CDNet）实现高效去噪；③结合约束能量最小化（CEM）与二值交叉熵损失的分离损失，强化背景抑制与目标突出；④采用DDIM式的确定性逆采样链显著加速生成过程；

**🔧 技术方法**

核心技术为条件扩散模型（DDPM/DDIM）、Gaussian噪声扩散与逆扩散、基于Mlp的条件去噪网络、CEM与BCE混合分离损失、时序嵌入、谱条件嵌入以及快速逆采样；

**📊 数据集**

实验使用了四个公开高光谱数据集：MUUL Gulfport、HYDICE、Airport-4、Urban-2；

**📈 对比分析**

与8种代表性方法（OSP、CEM、CSCR、WDCCR、BLTSC、HTD-Net、TSTTD、OS-VAE）在四个数据集上进行对比，SCDM-HTD在AUC_(P_f,P_d)上均取得最高分，且运行时间仅为几百毫秒，显示出优越的检测精度与实时性；

**⚠️ 局限性**

局限性包括：对目标先验谱的依赖，合成样本可能导致分布偏差；对超参数（如μ、λ）敏感；在更大规模或多目标场景下的可扩展性与泛化能力尚未验证；

---

## 503. ActionRAG: Evidence-Refining Graph Diffusion for Long-Term Memory in LLM Agents

**arXiv ID:** 2610.32752 | [PDF](https://arxiv.org/pdf/2610.32752v1)

**作者:** Navyansh Mahla `[一作]` (Actioneer), Sashank Vandrangi `[通讯]` (Actioneer)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于图结构的长期记忆框架ActionRAG，用于改进大型语言模型代理的检索与推理；

**💡 创新点**

创新点在于：①谨慎的图索引（标签对齐、分层实体解析、冲突感知摘要）；②双通道检索（实体与三元组）并构建局部图；③基于接收器归一化的图扩散与稀疏化、种子保护与中心化惩罚；④在检索阶段自我修正并补齐缺失的三元组；

**🔧 技术方法**

技术手段包括BGE‑M3嵌入、Claude Haiku抽取、PostgreSQL + pgvector存储、双通道最近邻检索、受限局部图扩散、闭式求解的接收器归一化扩散、基于阈值的保留与物化、LLM生成答案；

**📊 数据集**

使用MemoryAgentBench（MAB）作为基准，涵盖多轮对话下的准确检索、冲突解决、长程理解和测试时学习八个指标；

**📈 对比分析**

与MAB公开的长上下文、RAG、结构增强RAG、Agentic Memory等基线对比，ActionRAG在六个指标上取得最高检索基准分，平均得分55.1，超过HippoRAG‑v2 5.4分，查询成本仅约$0.001，检索延迟<0.5s；

**⚠️ 局限性**

局限性在于对长篇叙事性问答（如MH‑DocQA、Detective QA）的表现略逊，且对超参数（k、α、β、H等）敏感，未来需改进图扩展与自适应机制。

---

## 504. On the Identifiability of Hybrid Deep Generative Models: Meta-Learning as a Solution

**arXiv ID:** 2610.32748 | [PDF](https://arxiv.org/pdf/2610.32748v1)

**作者:** Yubo Ye `[一作]` (Zhejiang University), Linwei Wang `[通讯]` (Rochester Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `afceb026-1760-41ae-8d86-010831a37d97` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究并理论证明混合深度生成模型的可识别性，并提出基于元学习的可识别混合-DGM框架。

**💡 创新点**

首次从理论角度揭示无条件先验混合-DGM的不识别性，并通过元学习构造条件独立先验实现可识别性。

**🔧 技术方法**

结合非线性ICA理论、条件VAE、元学习与可变分推断，构建Meta-Hybrid-VAE。

**📊 数据集**

评估于三种仿真物理系统（阻尼驱动摆、扩散-输运系统、双摆）及真实双摆视频。

**📈 对比分析**

与传统混合VAE及APHYNITY比较，在MSE、MCC等可识别度指标上明显优于基线，预测与OOD性能显著提升。

**⚠️ 局限性**

需已知同一生成过程的上下文-查询样本配对，且未做全面超参数、模型通用性及纯物理与纯神经对比的消融研究。

---

## 505. Efficient Streaming Audio-Visual Target Speaker Extraction for Real-World Acoustic Scenes

**arXiv ID:** 2610.32749 | [PDF](https://arxiv.org/pdf/2610.32749v1)

**作者:** Wendi Sang `[一作]` (Qinghai University), Xiaolin Hu `[通讯]` (Tsinghua University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种新的音视频目标说话人提取方法Falcon，并发布了一个新的基准数据集RealSSA，旨在解决当前因果系统在数据和模型架构上的不足。

**💡 创新点**

创新点在于RealSSA基准数据集的发布，能够模拟多种真实场景的干扰，并且Falcon方法通过单次编码-解码过程实现多尺度时频分离，显著提高了提取质量和计算效率。

**🔧 技术方法**

使用了单次编码-解码的Falcon方法，结合了编码器空间说话人对比损失，能够有效抑制近场干扰。

**📊 数据集**

使用了RealSSA-Sim和RealSSA-Real数据集，前者是可控的模拟场景，后者是实际录制的评估集。

**📈 对比分析**

与基线模型AV-TFGridNet相比，Falcon在参数数量上减少了91.9%，计算量减少了20.2倍，GPU推理延迟减少了11.1倍，同时在多个基准上达到了最先进的提取质量。

**⚠️ 局限性**

限制在于Falcon的分离器和编码器空间说话人对比损失仅在单通道上操作，未来可以扩展到多通道。此外，RealSSA-Real的数据量相对较小，未来可以通过增加参与者和环境来扩展。

---

## 506. PlayerEmbed: Learning Behavioral Fingerprint Representations from Raw Input Sequences

**arXiv ID:** 2610.32727 | [PDF](https://arxiv.org/pdf/2610.32727v1)

**作者:** Tobias Schneider `[一作]` (Fraunhofer Iais), Rafet Sifa `[通讯]` (Fraunhofer Iais)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了PlayerEmbed框架，利用深度度量学习从原始游戏输入序列中学习紧凑的玩家行为嵌入，实现玩家身份识别；

**💡 创新点**

创新点在于：①使用原始键鼠动态而非手工特征；②结合混合CNN-LSTM编码器与Proxy Anchor Loss，实现可泛化到未见玩家的嵌入；③在大型数据集上验证，展示对1000名已知和1000名未知玩家的识别性能；

**🔧 技术方法**

采用深度度量学习（Proxy Anchor Loss）、混合CNN‑LSTM编码器、归一化与投影头、最近中心/媒体/k‑NN分类器，以及多样本平均；

**📊 数据集**

使用DDNet（开放源代码平台游戏）收集的3000名高活跃玩家的原始键鼠输入序列，长度固定1000步（20秒）；

**📈 对比分析**

与传统手工统计特征（MLP、SVM、XGBoost、最近中心）对比，PlayerEmbed在M=6时均衡准确率接近1，远超传统方法，且在未知玩家上也能达到≈0.88；

**⚠️ 局限性**

局限包括：仅在单一游戏（DDNet）验证，未测试跨游戏泛化；模型对大规模玩家数仍有性能下降；需要大量原始输入数据，且对输入特征类型有一定假设。

---

## 507. Robust Offset-Free Lane-Keeping Control Framework for Vision-Only Automated Vehicles

**arXiv ID:** 2610.32666 | [PDF](https://arxiv.org/pdf/2610.32666v1)

**作者:** Der-Hau Lee `[一作]` `[通讯]` (National Yang Ming Chiao Tung University), Der-Hau Lee (National Yang Ming Chiao Tung University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研发了鲁棒的 offset‑free model predictive control（OF‑MPC）框架，用于仅靠视觉感知的自动驾驶车辆在高速弯道上实现精确、无偏差的车道保持控制。

**💡 创新点**

将自适应管道交叉线迭代线性二次调节器（itube‑CILQR）嵌入 OF‑MPC，既降低计算量，又实现了实时无偏差跟踪，并通过 κ‑score 量化视觉感知系统的未见场景不确定性。

**🔧 技术方法**

采用了深度多任务 U‑Net 视觉前端、Kalman 滤波器状态与扰动估计、线性二次调节器、凸指数障碍函数、离散线性化双轮模型及增益矩阵求解等技术。

**📊 数据集**

使用 TORCS 仿真环境的两条三车道高速公路轨道（Track A、Track B）进行实验，且所用视觉模型 MT‑UNet 在这些轨道上均未被训练过。

**📈 对比分析**

与 IPOPT、OSQP 求解器对比后，itube‑CILQR‑T‑20 在约 4.21 ms 的平均计算时间内实现最大横向偏差仅 0.0217 m；在 TORCS 实验中，噪声水平不超过 1.8% 时即可完成完整车道保持。

**⚠️ 局限性**

局限在于仅支持单摄像头视觉输入，对极端弯道或高噪声场景仍易失稳；模型基于线性双轮动力学，可能无法覆盖更复杂的车辆动力学行为。

---

## 508. Clean-label Poisoning Attack and Defense Game for Federated Learning in Vehicular Networks

**arXiv ID:** 2610.32765 | [PDF](https://arxiv.org/pdf/2610.32765v1)

**作者:** Yalin E. Sagduyu `[一作]` (Nexcepta Inc.), Mengran Xue `[通讯]` (RTX BBN Technologies)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c84dae5d-5273-4348-85a7-b44cb586b4df` `6215c339-3735-4be3-8a07-5bbb7004712d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文在车辆网络联邦学习环境下，设计并验证了一种基于特征空间的clean‑label poisoning攻击，并提出一种基于特征一致性的重标记防御，随后用博弈论分析攻击‑防御的均衡策略。

**💡 创新点**

创新点包括：①针对联邦学习的特征空间clean‑label攻击模型；②融合距离、尾部、邻域、损失及重复模式等多种一致性指标的重标记防御；③构建攻击‑防御交互的非合作博弈框架，求解纯混合均衡并阐释阈值结构。

**🔧 技术方法**

使用技术包括：联邦学习与图约束聚合、SHAP特征重要性评估、分位数目标向量、距离/尾部/最近邻/损失/重复度一致性评分、加权求和检测、以及Nash均衡求解与区域划分。

**📊 数据集**

实验数据集为VeReMi扩展数据集，用于车辆安全消息（BSM）的误报检测。

**📈 对比分析**

与无防御的联邦学习基线对比，指标为干净样本准确率和攻击成功率；实验显示防御在保持干净准确率近乎不变的同时，将攻击成功率显著压低；攻击成功率随被攻破节点数线性增长，防御有效抑制其扩散。

**⚠️ 局限性**

局限性包括：仅在单一VeReMi数据集上验证，缺乏跨数据集与真实动态网络环境的泛化评估；防御阈值需人工调优，且对特征空间攻击的假设可能不适用于所有场景；未充分评估计算开销与实时性约束。

---

## 509. Selection under partial information

**arXiv ID:** 2610.32731 | [PDF](https://arxiv.org/pdf/2610.32731v1)

**作者:** Benjamin Aram Berendsohn `[一作]` `[通讯]` (Max Planck Institute for Informatics), Benjamin Aram Berendsohn (Max Planck Institute for Informatics)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `2704f255-0c84-4173-b83c-0e9a3dbea232` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文针对部分序列（poset）选择问题提出了一系列算法，给出了最优或近似最优的比较次数和运行时间；在树和森林的特殊结构下进一步优化，使得比较次数可降至 (G,k)+log(k/(G,k))，在 k‑unbounded 树上可达到 loglog(k/(G,k))。

**💡 创新点**

主要创新点包括：①引入“加权软堆”（Weighted Soft Heap）以对不同元素的腐败风险进行动态加权；②在软堆上实现增权操作，能够精确控制哪些子节点在被删除或腐败时才会被挖入优先队列；③通过“Heapify”与信息理论下界的结合，对树、森林以及一般 DAG 的选择问题给出了统一的分析框架；④在森林中利用递归块压缩和逐步消除的策略实现了 loglog 级别的比较复杂度。

**🔧 技术方法**

使用的技术主要有：软堆（Chazelle 的 Soft Heap）及其加权变种；Heapify（将子集构造为堆以保持部分序列一致性）；增权操作与动态权重更新；信息理论下界与树/森林结构的结合分析；以及对 DAG 中独占出度（exclusive out‑degree）的精细度量。

**📊 数据集**

该工作为纯理论算法研究，未使用任何实际数据集；所有结果均基于对图结构的数学分析与比较次数上界的证明。

**📈 对比分析**

相较于已知下界 (G,k) ，本文的算法在一般 DAG 上达到了 (k + (G,k)) 次比较，几乎匹配下界；在树/森林上，通过额外的结构假设进一步压缩为 (G,k)+log(k/(G,k))，在 k‑unbounded 情况下甚至实现了 loglog(k/(G,k)) 的比较次数；时间复杂度与比较次数保持一致，且在所有特殊结构下均为线性或接近线性。

**⚠️ 局限性**

局限性包括：①当 (G,k) 远小于 k 时，算法仍可能产生多于 k 次的比较，未能达到真正的最优；②对一般 DAG 的时间复杂度仍受图大小限制，无法完全用 (k + (G,k)) 描述；③加权软堆和增权操作实现复杂，实际实现成本较高；④对部分结构（如高度相近的树）需要额外的假设或改进，未给出完全通用的最优解。

---

## 510. SP-Guide: Enhancing Aerial Vision-Language Navigation with Spatial Priors

**arXiv ID:** 2610.32697 | [PDF](https://arxiv.org/pdf/2610.32697v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 511. Deterministic Incremental Triconnectivity in $O(\log^2 n)$ Worst-Case Time

**arXiv ID:** 2610.32721 | [PDF](https://arxiv.org/pdf/2610.32721v1)

**作者:** Niels Inselmann Christiansen `[一作]` (IT University of Copenhagen), Eva Rotenberg `[通讯]` (IT University of Copenhagen)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

**🎯 论文内容**

设计了一种确定性增量算法，维护任意连通图的三连通性，支持边插入、边细分以及三连通性查询，时间复杂度为 O(log n)（对二连通图）和 O(log² n)（对一般图）。

**💡 创新点**

通过将 SPQR 树与加权重重路径分解相结合，构造了可在最坏情况下以 O(log n) 时间更新与查询三连通性的结构，突破了之前 O(log³ n) 的上界。

**🔧 技术方法**

采用 ST 树、偏置搜索树、重路径分解、SPQR 树与“预分割”松弛 SPQR 树等技术；插入边时仅需常数次重路径操作和收缩。

**📊 数据集**

本文仅给出理论证明，未在任何实验数据集上进行评估。

**📈 对比分析**

与 Holm 与 Rotenberg 等人提出的 O(log³ n) 增量算法相比，所提算法在最坏情况下将时间从 O(log³ n) 降低到 O(log² n)，并接近单点连通性的 Ω(log n / loglog n) 下界。

**⚠️ 局限性**

对二连通图的算法虽接近下界，但仍有 O(loglog n) 的缺口；在全动态设置下仍缺乏多项式对数的三连通性查询结构。

---

## 512. Migration as a Steady-State Execution Primitive: A Runtime for Virtual Stationarity in Low-Earth Orbit

**arXiv ID:** 2610.32720 | [PDF](https://arxiv.org/pdf/2610.32720v1)

**作者:** Daniel Kleebinder `[一作]` (TU Wien), Stefan Nastic `[通讯]` (TU Wien)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在低地球轨道环境下，将连续卫星迁移视为持续执行原语，提出了虚拟站态（Virtual Stationarity, VSR）框架与迁移膜（Migration Membrane）实现持续迁移与资源虚拟化。

**💡 创新点**

将实时迁移从一次性操作转化为稳态执行原语，提出四阶段迁移协议与能力虚拟化+连续状态投影机制，实现跨异构 ISA 的无累积停机时间迁移。

**🔧 技术方法**

基于 WebAssembly Micro Runtime (WAMR) 的可移植执行与状态捕获，利用 CRIU、Pre‑copy 等传统方法对比，采用SCP/UDP、WASI 接口做能力虚拟化，采用事件流与回放实现连续状态投影。

**📊 数据集**

使用公开的 XR 维护工作负载（约4000 pkt/s，5 MB/s），Raspberry Pi 5 物理测试平台，跨 x86‑64 与 ARM64 的双架构执行；采用 SGP4 模拟的1600颗卫星星座轨道和 ISL/GSL 传输模型。

**📈 对比分析**

通过在物理双节点 Pi 5 上连续300次迁移实验，比较 checkpoint/restore、Pre‑copy 与 VSR；VSR 迁移时间稳定在 4.2 s（p95 5.5 s），停机时间 <6 ms，累积占用 <4 %，可满足 10 ms 及 6 ms 的 SLO；在模拟轨道中迁移完成在 2.8–4.5 s，满足 83–242 s 的交接窗口。

**⚠️ 局限性**

依赖外部轨道调度器，无法自适应能耗与热量；仅支持单线程模块与确定性执行；在高事件率接近投影吞吐 μ 时会超时；无法处理迁移后故障恢复，且对非 WebAssembly 运行时的可移植性有限。

---

## 513. Agent-Based Transport Model Calibration via Online Stochastic Batch Optimization

**arXiv ID:** 2610.32698 | [PDF](https://arxiv.org/pdf/2610.32698v1)

**作者:** Abdelkader Dib `[一作]` (ETH Zürich), Milos Balac `[通讯]` (ETH Zürich)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在一次完整的 MATSim 模拟运行中实现在线参数校准，避免传统的数百至数千次独立运行

**💡 创新点**

将机器学习中的随机批量优化（SGD）与 ABM 校准结合，形成单运行内迭代更新的在线校准框架

**🔧 技术方法**

采用 MATSim 与 eqasim 生成的瑞士全国规模场景，利用 CMA‑ES、SPSA、PSO 等黑盒优化器进行在线求解；并使用随机梯度、动量/EMA 平滑参数

**📊 数据集**

瑞士全国交通场景（5% 样本）与 Île‑de‑France 公开场景，真实的微观调查数据（Swiss Microcensus）作为目标模式分布

**📈 对比分析**

对比传统离线多次运行的校准：在 60 次迭代内即可使模式份额与观测值匹配；ASCs 校准几乎无额外计算负担，完整参数校准每次迭代耗时 2–3 分钟，整体运行时间与未校准相近

**⚠️ 局限性**

局部最优与多重极小值导致参数不稳定；需在目标函数上保证凸性或采用温度/热启动；行为可解释性不足，需要设定参数边界；小样本或低人口地区导致更新频率低、过拟合风险

---

## 514. AquaSense-ICS: A Hybrid Real-Simulated Cyber-Physical Dataset and Leakage-Aware Benchmark for Smart Water Quality Monitoring and Attack Detection

**arXiv ID:** 2610.32661 | [PDF](https://arxiv.org/pdf/2610.32661v1)

**作者:** Amal Neji `[一作]`, Lamia Chaari Fourati `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并发布了 AquaSense‑ICS 混合水质数据集，包括 25.4 小时真实 IoT 监测、13 节点 EPANET‑MSX 多物种模拟、注入的网络攻击与故障、以及 TimeGAN 合成序列，并给出了完整的标签、数据字典和评估协议。

**💡 创新点**

①首次将真实低成本传感器与多物种水质模拟同时融合；②提供可追溯的攻击/故障样本级标签；③设计防泄露的时间序列分割与留一攻击类型评估；④引入 SHAP 可解释性分析验证数据泄露。

**🔧 技术方法**

硬件采集：Raspberry Pi 5、TDS 传感器、BME680；仿真：EPANET‑MSX；数据增强：Jitter、Scaling、TimeWarp、Window Slicing、Permutation、Bootstrapping、TimeGAN；攻击生成器：FDI、Replay、DoS、Drift；机器学习：Random Forest、XGBoost、Isolation Forest、One‑class SVM、Logistic Regression；解释性：TreeSHAP。

**📊 数据集**

核心数据集为 AquaSense‑ICS（含 23,688 条真实记录、24 条仿真状态、TimeGAN 合成序列），对比已有数据集（SWaT、WADI、BattLeDIM、BATADAL）验证评估协议的必要性。

**📈 对比分析**

采用随机拆分和标签计算可获得宏 F1≈0.99，说明泄露严重；改用时间序列拆分与留一攻击类型，宏 F1 仅为 0.465，最优检测器 AUROC≈0.64（未见攻击类型），表明模型在真实泄露环境下性能远低于表面表现。

**⚠️ 局限性**

①真实采样时间短、缺乏多传感器；②水质标签恒定、未与实验样本验证；③仿真与真实不具物理耦合；④攻击仅注入数据级别，缺乏闭环物理后果；⑤攻击样本稀缺且分布不均；⑥TDS 未校准、无单位；⑦部分攻击类型仅作用于仿真通道。

---

## 515. WiMoT: Replay-Free Continual Wi-Fi Sensing via Moment Transport

**arXiv ID:** 2610.32660 | [PDF](https://arxiv.org/pdf/2610.32660v1)

**作者:** Chunchun Yang `[一作]` (University of Science and Technology of China), Zhitang Chen `[通讯]` (Huawei)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本论文提出了一种针对 Wi‑Fi 感知的持续学习框架 WiMoT，能够在不存储历史样本的情况下实现高记忆稳定性和可塑性；

**💡 创新点**

核心创新点在于通过冻结编码器并引入基于统计矩的“moment transport”与“moment constraints”来对抗模型漂移，同时利用 OSCL 统计分类器实现无域标识的无遗忘推理；

**🔧 技术方法**

技术手段包括基于 Transformer 的冻结特征编码器、适配器（adapter）与投影层的可微训练、带温度的监督对比损失、矩传输约束、以及基于类矩的 OSCL 分类器；

**📊 数据集**

实验使用三大 Wi‑Fi 数据集 3DO（3 天、3 活动）、WiMANS（3 房间、9 活动）与 Widar3.0（9 位用户、6 姿势）进行验证；

**📈 对比分析**

与 21 种基线（含 fine‑tune、EWC、SI、MAS、LwF、DER++、A‑GEM、CN‑DPM、DUCT、DualCP、ConSense、EdgeCL 等）在无样本回放条件下对比，WiMoT 在平均准确率（AFA）上排名第一或第二，平均遗忘（AFR）显著低于竞争者；

**⚠️ 局限性**

局限性主要在于需依赖预训练的冻结编码器，且对新域的适应性受限于适配器宽度；此外，存储统计矩虽比原始样本小，但在大规模类别时仍可能产生显著内存占用，且当前方法尚未在跨模态或动态环境中进行充分验证。

---

## 516. DT-JEPA: Learning to Predict Future Representations for Decision-Making

**arXiv ID:** 2610.32657 | [PDF](https://arxiv.org/pdf/2610.32657v1)

**作者:** Changxu Ni `[一作]` (Southeast University), Shi Jin `[通讯]` (Southeast University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种将Joint-Embedding Predictive Architecture (JEPA) 与Decision Transformer (DT) 结合的DT-JEPA框架，实现奖励无关的离线预训练与奖励导向的在线微调，提升跨任务的泛化与适应效率。

**💡 创新点**

创新点在于：①引入概率化的JEPA预测器捕获未来潜在表示及其不确定性；②通过AdaLN将预测的未来潜在信息注入Transformer的隐藏层；③设计Critic引导的在线微调机制，使预训练策略能在未见奖励下快速收敛。

**🔧 技术方法**

技术主要包括：离线强化学习、Transformer序列建模、Joint-Embedding Predictive Architecture、概率分布预测（高斯）、Adaptive Layer Normalization (AdaLN)、EMA目标网络、行为克隆与Q值指导。

**📊 数据集**

使用D4RL基准数据集：MuJoCo四个运动任务（Hopper、Walker2d、HalfCheetah、Ant）各三种数据集（medium、medium-replay、medium-expert）以及AntMaze和Maze2D的多种规模配置。

**📈 对比分析**

与TD3-BC、TD-JEPA、ODT、QT、Prompt DT、VDT、LSDT、PDT等基线比较，DT-JEPA在12个Gym MuJoCo数据集上赢得10项冠军，并在多任务与奖励迁移实验中实现最快收敛和最高归一化回报。

**⚠️ 局限性**

局限性包括：仍需大量离线数据；在线微调仍依赖有限交互；对更复杂环境（高维视觉、非连续动作）的适用性未充分验证；计算成本相对较高。

---

## 517. OpenDraco: A Platform for Engineering and Experimenting with Configurable Multi-Agent Architectures for Software Engineering

**arXiv ID:** 2610.32686 | [PDF](https://arxiv.org/pdf/2610.32686v1)

**作者:** Xiang Feng Ye Pan `[一作]` (Universitat Politècnica de Catalunya), Matias Martinez `[通讯]` (Universitat Politècnica de Catalunya)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一个开源平台，能够将基于LLM的多智能体系统（MAS）以可导出、版本化的JSON规范表示，自动转换为LangGraph工作流，并在同一环境中完成配置、执行、评估和结果比较，随后以自动程序修复（APR）为示例验证其可用性。

**💡 创新点**

创新点在于提供了统一的可导出、版本化的MAS规范与执行框架，整合了代理定义、提示、工具、模型、参数和通信拓扑，支持在同一平台上进行多配置实验、结果复现和成本/效果对比；同时利用LangGraph实现了可视化与可扩展的工作流转换。

**🔧 技术方法**

技术栈包括Angular前端、FastAPI后端、Python实现的Agentic System Modeler/Executor、LangGraph框架、Ollama本地LLM、OpenAI兼容API、SWE-bench评估 harness、Docker容器、Git版本控制等。

**📊 数据集**

使用的数据集包括：五个合成APR实例（涵盖不同难度级别）、SWE-bench Lite 23个开发子集实例、77个测试子集实例（仅保留金手册补丁<600字符的），以及从16个公开仓库构建的代理库。

**📈 对比分析**

通过在统一的本地qwen3.5:9b模型上执行8个不同拓扑的MAS配置，记录解决率、墙钟时间和token使用；在23dev集对baseline与其他配置进行对比；在77test集使用baseline评估，得到12/77的解决率；结果显示openhands_star在合成集最快且token最少，但在dev集baseline在时间和token上更具优势；温度0/1对单次运行的可变性显著。

**⚠️ 局限性**

局限性包括：重建的APR配置未完全复现原系统行为；受GPU 6GB限制导致所有分支顺序执行，未体现并行潜能；仅使用本地qwen3.5:9b，模型泛化受限；实验大部分为单次运行，缺乏多次重复评估；在真实数据上仅评估少数配置，难以全面推断拓扑影响。

---

## 518. Near-Optimal List Replicability from Littlestone Dimension

**arXiv ID:** 2610.32853 | [PDF](https://arxiv.org/pdf/2610.32853v1)

**作者:** Ari Blondal `[一作]` (McGill University), Sivan Tretiak `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

研究了可实现PAC学习框架中的列表可复制性数，并证明了每个具有Littlestone维度d和对偶VC维度v^*的概念类在计数域上满足()≤2^O(dmin{log d,v^*})。

**💡 创新点**

该研究提供了一个新的上界，显著改善了之前的结果，并且该界限与准确度无关，突破了以往固定准确度下的限制。

**🔧 技术方法**

使用了一种新的归纳框架来界定列表可复制性，并结合了稳定性和不可约性等概念。

**📊 数据集**

论文中没有具体提到使用的数据集。

**📈 对比分析**

与之前的研究相比，新的上界在理论上提供了更好的性能，尤其是在不依赖于准确度的情况下，性能得到了显著提升。

**⚠️ 局限性**

该研究的局限性在于没有考虑具体的数据集应用，且可能在实际应用中需要进一步验证其有效性。

---

## 519. OpenStripe supports open-set identification and spatial stripe profiling in wild Amur tigers

**arXiv ID:** 2610.32848 | [PDF](https://arxiv.org/pdf/2610.32848v1)

**作者:** Yichun Peng `[一作]`, Tianming Wang `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `729e5870-4135-47f5-97f2-e3974d07b5dc` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了 OpenStripe 多智能体框架，对东北华北虎豹国家公园的摄像机陷阱图像进行非侵入性识别，重建虎身条纹并标注四种局部结构，实现开放集个体聚类。

**💡 创新点**

创新点在于结合多智能体与大型语言模型，利用可解释的条纹拓扑特征（断点、分支、异常、圆形）实现无监督开放集识别，并揭示条纹特征与地理空间的关联。

**🔧 技术方法**

采用 GPT Image 2.5、大语言模型、黑板架构、前景分割、源对齐重建、结构注释、源平衡共识聚类以及多种距离度量和统计方法。

**📊 数据集**

使用了 2023 年在东北华北虎豹国家公园及中俄边境收集的 100 条 Amur 虎摄像机陷阱记录，来自 28 只个体。

**📈 对比分析**

与 49 种基线（单特征和融合）进行对比，OpenStripe 在准确率 0.90、ARI 0.85、NMI 0.97、B^3 F1 0.92 等指标上显著优于最佳基线（准确率 0.43、ARI 0.14），并通过消融实验验证四类注释的贡献。

**⚠️ 局限性**

局限在于仅使用单侧条纹，可能遗漏另一侧特征；缺乏性别信息导致空间统计偏差；样本量有限，且原始图像和地理数据未公开。

---

## 520. KrishokTech: A Bengali Agricultural Advisory System with Explicit Authorization Controls

**arXiv ID:** 2610.32817 | [PDF](https://arxiv.org/pdf/2610.32817v1)

**作者:** Khan Raiyan Ibne Reza `[一作]` (North South University), Sumaiya Tabassum Nimi `[通讯]` (North South University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

设计并实现了一个名为 KrishokTech 的基于授权边界的孟加拉语农业咨询系统，能够在检索、生成和发布前对请求进行多阶段控制并向用户展示控制决策。

**💡 创新点**

创新点在于将授权分为五个可视化的阶段（T0–T4）——预检、请求校验、证据绑定、草稿生成与剂量验证，并通过界面将 ASK、CONFIRM、REFER、DROP 等决策显式呈现给用户，确保高风险请求被及时拦截或转介，避免误导性剂量信息。

**🔧 技术方法**

使用的技术包括：Bengali/Banglish 归一化与规则预检、基于 BM25 的检索、浏览器端视觉分类器用于作物识别、剂量范围验证器、LoRA 适配器的生成模型，以及可视化的决策面板。

**📊 数据集**

数据集涵盖约2100条来自 BARI、BRRI、DAE 的农业指南段落，测试集包括 200 份农户查询、400 份应激查询以及 454 条文本–图像冲突案例，且还使用了 118 条人工插入的剂量错误。

**📈 对比分析**

比较方法：对照 T0 与 T1 预检、作物绑定、误差检测等控制流程。实验结果显示：T1 对 30/30 crop‑less 查询做 ASK，T0 对 85/85 高风险请求拦截；作物绑定将交叉作物检索率从 36.5% 降至 0%；文本–图像冲突检测命中 453/454 例；剂量验证器捕获 114/118 错误；整体安全性显著提升。

**⚠️ 局限性**

局限性包括：仅支持六类作物，需手动维护作物别名和证据索引；归一化和预检仅覆盖已知高风险模式，可能漏检新颖表述；剂量验证器仅检查剂量数值，对其他农学陈述不作完整事实核查；部分攻击仍能通过系统，未实现完全无风险；评估基于构造数据，缺乏现场农户决策与产量效益验证。

---

## 521. Sounding Surgical Instrument Localisation through Audio-Visual Correspondence

**arXiv ID:** 2610.32816 | [PDF](https://arxiv.org/pdf/2610.32816v1)

**作者:** Dominic Behling `[一作]` (University of Zurich), Matthias Seibold `[通讯]` (University of Zurich)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `e0540dec-d77f-42db-94ae-d039248f6393` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b88c6eac-d57a-4623-a604-1f401f3eb268` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

利用单麦克风和摄像头的音频-视觉对应关系，实现外科手术工具的检测与定位，且无需空间标注；

**💡 创新点**

首次在外科领域引入弱监督音频-视觉源定位（AV‑SSL），通过自定义静音、面积和复合损失提升模型对噪声、无声场景和多工具场景的鲁棒性；

**🔧 技术方法**

基于SLAVC的弱监督对比学习框架，使用DINOv2 ViT作为视觉编码器，结合音频谱图特征，并加入静音/面积/复合正则化；

**📊 数据集**

训练集为“action camera corpus”（54段模拟钻孔、锯割、凿割的RGB+单声道音频），测试集为公开的“acoustic camera corpus”（21段配备光学跟踪的RGB-D+声学相机录制），并使用光学跟踪生成的框架盒子做评估；

**📈 对比分析**

与SLAVC和ACL‑SSL基线对比，模型在所有工具上IoU提升至约0.46（基线仅0.03–0.04），距离误差下降至约10–12像素，检测AUC为1.0；

**⚠️ 局限性**

受限于数据量有限、仅涵盖三种手术动作、缺乏视角与声音多样性、输出分辨率粗糙、仅处理可见工具且无法处理遮挡或无声但可见场景等问题。

---

## 522. ParaTaxe: a Suite of Tools for Processing and Publishing Multi-parallel Corpora

**arXiv ID:** 2610.32796 | [PDF](https://arxiv.org/pdf/2610.32796v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 523. Beam Search for Error-Correcting Codes via Conditional Diffusion

**arXiv ID:** 2610.32807 | [PDF](https://arxiv.org/pdf/2610.32807v1)

**作者:** Ziv Aharoni `[一作]` (Duke University), Henry D. Pfister `[通讯]` (Duke University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93`

**🎯 论文内容**

无法获取论文具体内容

**💡 创新点**

无法判断创新点

**🔧 技术方法**

无法确认使用技术

**📊 数据集**

无法识别使用数据集

**📈 对比分析**

无法比较方法或评估性能

**⚠️ 局限性**

缺乏足够信息导致无法评估限制

---

## 524. Multiplicative vs. Additive State Coupling in Learned Optimizer Schedules

**arXiv ID:** 2610.32801 | [PDF](https://arxiv.org/pdf/2610.32801v1)

**作者:** M. Harper Langston `[一作]` (Qualcomm AI Research), Harris Teague `[通讯]` (Qualcomm AI Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

学习并优化最大割（MAX‑CUT）连续惩罚松弛中的调度超参数，利用无滚动的经典求解器进行完整阶段展开并反向传播。

**💡 创新点**

通过比较两种状态自适应调度（双线性和加性）展示了调度中状态耦合形式对离群分布泛化的决定性影响，证明仅靠状态感知不足，需特定的乘法交互结构。

**🔧 技术方法**

使用梯度下降连续惩罚继续方法、softplus、tanh 激活、可展开的求解器、Adam 优化器和反向传播学习调度参数。

**📊 数据集**

训练数据为 Erdős–Rényi 随机图 G(n,0.5)（n∈{6,…,15}）；测试数据包括 n=30,50,100 的随机图以及 BiqMac 80 个已知最优解的实例。

**📈 对比分析**

对比冻结基线、学习后的调度和手动调优的几何调度。双线性调度在 OOD 上显著提升（BiqMac 比例 0.9627≈0.9624），而加性调度相对基线下降；在分布内表现相近。

**⚠️ 局限性**

局限性在于实验仅针对一种松弛形式、一种求解器、特定图分布；不一定能推广到其他约束或学习型优化器；统计证据相对有限，机制尚待进一步验证。

---

## 525. CompMVR: Compression-Aware Multi-View Restoration Using Diffusion Models for Geometrically Consistent 3D Reconstruction

**arXiv ID:** 2610.32800 | [PDF](https://arxiv.org/pdf/2610.32800v1)

**作者:** Dong-hwi Kim `[一作]` (Kyungpook National University), Sang-hyo Park `[通讯]` (Kyungpook National University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本研究提出了 CompMVR，一种针对压缩多视图图像的扩散模型恢复框架，旨在在保持压缩后图像视觉质量的同时，显著提升跨视图几何一致性，从而提升3D重建和新视角合成等下游任务的性能。

**💡 创新点**

创新点包括：①将压缩码率信息（codec 与 QP）编码为全局压缩先验；②利用空间自适应压缩残差估计器（CAE）捕捉局部压缩噪声；③在扩散恢复阶段加入跨视图对应损失与三维注意力模块，实现几何一致性约束；④构建压缩多视图基准数据集，系统评估压缩对几何一致性的影响。

**🔧 技术方法**

技术手段包括：基于 Stable Diffusion 2.1 的扩散恢复模型（使用 LoRA 微调）、两阶段训练（先学习压缩先验，再联合恢复多视图）、压缩参数估计器（codec & QP 预测）、CAE（轻量级全卷积网络）、对应损失（基于 VGGT 的像素对应），以及三维自注意力以实现跨视图特征交互。

**📊 数据集**

训练数据：DTU、LLFF、Mip-NeRF360、Tanks&Temples 共 146 场景；评估数据：Free、CO3Dv2、RealEstate10K。采用四种压缩编码（AVC、HEVC、VP9、AV1），训练时使用 QP 20/26/32/38/44/50，测试时使用 QP 27/42，并对未见的 VVC/H.266 进行泛化测试。

**📈 对比分析**

与 CODiff、DiQP、SIR‑Diff 等单视图或多视图扩散恢复基线对比。评估指标包括图像质量（PSNR、SSIM、LPIPS、NIQE、CLIPIQA）、几何一致性（MEt3R、AUC@5、AUC@15/30）、新视角合成质量。结果表明 CompMVR 在所有指标上均优于或接近最优，尤其在压缩 QP42 严重失真下保持较高的匹配特征数和 pose 准确率，且在新视角渲染中展现更细腻的几何细节。

**⚠️ 局限性**

主要局限：①模型体积大、推理时间和显存消耗高（约 11GB GPU 内存），尚未实现端到端的全前向推理；②对不同压缩算法的适应性仍有限，需针对更复杂编码工具（如 AV1）设计更具通用性的压缩先验；③在未测试的 3D 任务中表现尚需进一步验证。

---

## 526. Consommé: Conditioned Token Merging for Split Inference

**arXiv ID:** 2610.32794 | [PDF](https://arxiv.org/pdf/2610.32794v1)

**作者:** Man Ngo `[一作]` (Leibniz University Hannover), Jan S. Rellermeyer `[通讯]` (Leibniz University Hannover)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 Consommé 方法，在冻结的 Vision Transformer 上通过压缩 CLS token 与软分配门合并 patch token，实现任意预算的分割推理。

**💡 创新点**

创新点在于将 CLS token 同时用作全局摘要与 token 选取信号，利用信息瓶颈压缩 CLS 并条件化软分配门，突破 BPM 的 50% 限制，单步精确达到任何 token 预算。

**🔧 技术方法**

采用信息瓶颈（VIB）、soft‑assignment 门、变分编码/解码、ViT 结构冻结以及少量适配器训练等技术。

**📊 数据集**

在 ImageNet‑1K 数据集上使用 DeiT‑S/T 及 ViT‑B 进行实验评估。

**📈 对比分析**

与 ToMe、ToFu、HAC‑ToMe、DiffRate、SPOT、DynamicViT 等 tier‑1/2/3 方法对比；在 k=8~64 的预算下，Consommé 在 DeiT‑S/T 上均优于 tier‑1/2 方法，且在轻预算下可匹配 fine‑tuned tier‑3 的准确率。

**⚠️ 局限性**

局限性包括仅适用于 CLS 读出 ViT；在极低预算或过早 split 时仍受限；需要公开共享的冻结 backbone，且预算和 split 深度的可调整性不够灵活。

---

## 527. Capacity Modification for Matching with Couples: An Integer Programming and Computational Study

**arXiv ID:** 2610.32793 | [PDF](https://arxiv.org/pdf/2610.32793v1)

**作者:** Zexiao Wu `[一作]` (Northwestern Polytechnical University), Péter Biró `[通讯]`

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究在医院/医生匹配（含情侣）中，通过对医院容量做有限修改来恢复稳定匹配，并给出整数规划与截断（cutoff）两种求解模型。

**💡 创新点**

证明：①在任意严格偏好下，只需把每医院容量改动不超过两位且总容量最多增加一位即可保证BIS稳定匹配；②在子响应、子完整偏好下，只需每医院增减一位且总增不超过情侣数即可；并给出存在性边界与紧致性证明。

**🔧 技术方法**

构造 Exact（整数规划）和 Cutoff（截断线性化）两类模型，使用 Scarf 主导约束、容量线性化、BIS 稳定性约束，并采用变量裁剪与分支策略。

**📊 数据集**

使用公开的 NRMP/MedMatch 2026 实验数据、100–30,000 名医生的随机市场，以及约 90,000 条合成与重建实例。

**📈 对比分析**

在固定容量下，Exact 的求解速度略快；在最小修复与大规模市场中，Cutoff 的完成率显著高于 Exact（92–96% 对比 50–71%），并在容量增量实验中均能证明最优修复并给出匹配数量提升。

**⚠️ 局限性**

尚未证明在任意严格偏好下单位增减即可保证稳定；多医院总容量约束下的最小化问题仍难处理；实验未考虑成本或偏好不确定性，缺乏经济最优决策指导。

---

## 528. Structure and Exact Certification of Nonnegative Tensor Decompositions

**arXiv ID:** 2610.32787 | [PDF](https://arxiv.org/pdf/2610.32787v1)

**作者:** Haoming Wang `[一作]` (Columbia University), Ming Yuan `[通讯]` (Columbia University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `847a60d8-a755-47af-ba5d-c5236b9e3083` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出一种对任意给定的非负张量 CP 分解进行完整、精确认证的方法，能够判定分解是否最小、是否本质唯一，并在失败时给出精确的较短或等长但不同的替代分解。

**💡 创新点**

创新点在于：① 通过正交换（positive exchange）与 Lovitz–Petrov 分裂定理，将任何最小化或唯一性失败归约为对供应的若干子集合进行局部替换；② 证明每个不可约替换的有效阶数 ≤ p+q‑2（p、q 为替换前后项数），从而得到一个与张量阶无关、仅依赖 R 的变量上界 O(R²)；③ 结合 Farkas 引理消除环境行变量，得到仅包含 < 2R² > 实变量的半代数方程组；④ 在此框架下给出确定性算法，时间为 G(R)·L^{O(R²)}。

**🔧 技术方法**

技术手段主要包括：半代数几何与实可解理论、Farkas 引理、Lovitz–Petrov 分裂不等式、极值点与极限射线分析、符号计算（Thom 编码）以及对多项式系统的可解性判定（如使用 Z3 的多项式求解器）。

**📊 数据集**

在实验验证中，作者使用了之前工作中留下的 114 个无法通过正散射准则判定的实例（所有实例均为 d=3、n₁=n₂=n₃=6、R=5），并用实现的算法（配合 Z3 作为 Solve 子程序）对其进行完整认证。

**📈 对比分析**

结果表明，算法成功解决了全部 114 个实例，其中 95 个被判定为最小且本质唯一，19 个为最小但非唯一。实验时间取决于所用求解器，实际运行时间相对可接受，但并未达到理论上 G(R)·L^{O(R²)} 的上界；该上界主要用于证明可行性，而非提供最优实现。

**⚠️ 局限性**

限制与挑战：① 算法的时间复杂度是 R² 次多项式级别，仍属指数级，尚未证明是否为 FPT；② 输出的分解可能需要包含指数阶数（至少 ⌊R/2⌋!）的代数数域，导致数值上不可忽视的符号复杂度；③ 对大型张量（大 nₘ 或 d）时，虽然变量上界不随维度增长，但实际求解多项式系统仍受高维度参数和求解器性能限制；④ 目前实验仅使用 Z3 作为可解子程序，缺乏针对本算法的专门高效求解器。

---

## 529. AFRA: Age-Threshold Frame-Based Random Access for Timely Communication in Multi-Agent Systems

**arXiv ID:** 2610.32785 | [PDF](https://arxiv.org/pdf/2610.32785v1)

**作者:** Jiedan Tan `[一作]` (Shenzhen University), Fang Liu `[通讯]` (Shenzhen University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并分析了一种名为AFRA的异步帧基随机接入协议，用于多智能体系统的及时状态更新。

**💡 创新点**

通过在阈值达到时强制首帧发送并在帧内允许一次随机重传，保证在L≥T>N≥2的条件下几乎必然收敛到无碰撞周期性运行，并给出了稳态吞吐量、平均AoI和峰值AoI的闭式表达式。

**🔧 技术方法**

采用马尔科夫链建模与吸收状态分析、离散事件仿真以及平均AoI、峰值AoI与吞吐量的理论评估。

**📊 数据集**

使用人工仿真数据，设置N=100、观测窗口H=100N，探究不同阈值和网络规模下的性能。

**📈 对比分析**

与原始TSA、1-持久TSA、SAT和理想TDMA进行对比，利用窗口平均AoI、窗口吞吐量以及稳态AoI评估。AFRA在相同阈值下实现更快收敛，窗口AoI比原始TSA低48.5%、比1-持久TSA低27.3%、比SAT低43.7%，并在小阈值下显著提升吞吐量。

**⚠️ 局限性**

仅在L≥T>N≥2时保证收敛，且假设无捕获效应、无多包接收且采用生成-随意模型，实际环境中可能存在多用户干扰、时延不确定性等因素。

---

## 530. Video-MemComp: Taylor-Guided Sensitivity-Aware Memory Management for Streaming Video Understanding

**arXiv ID:** 2610.32784 | [PDF](https://arxiv.org/pdf/2610.32784v1)

**作者:** Yonghui Niu `[一作]` (NJU), Chaoyou Fu `[通讯]` (NJU)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出 Video-MemComp，一种训练自由、模型无关、结构感知的混合框架，用 O(1) 内存实现无限流视频理解。

**💡 创新点**

创新点在于利用一阶泰勒展开将全局输出误差拆解为局部逼近误差与层级结构敏感度，并据此设计 Pre-RoPE 聚合与敏感度引导的预算分配方案，解决 KV 缓存线性增长瓶颈。

**🔧 技术方法**

采用的技术包括 Pre-RoPE 预聚合、KV 稀疏剪枝、代理查询与信息熵度量、梯度先验、三重混合预算分配、日志偏置补偿等。

**📊 数据集**

使用的数据集有 MLVU、Video-MME、EgoSchema、RVS-Ego/Movie、OVO-Bench 等多种离线与流式视频基准。

**📈 对比分析**

与多种 MLLM 后端（Qwen-2-VL-7B、Qwen-2.5-VL-3B、LLaVA-OV-7B、InternVL-3.5-2B）在 6K KV 预算下对比，Video-MemComp 与全缓存基线相当或更优，并在流式视频上超过 InfiniPot-V、StreamMem、ReKV 等现有压缩方法，提升数个百分点。

**⚠️ 局限性**

局限性包括泰勒近似在极端压缩下可能失效、结构敏感度先验易受分布漂移影响、Pre-RoPE 聚合带来额外计算开销，以及缺乏高阶近似与在线敏感度估计。

---

## 531. Multimodal Generative AI Assists Comprehensive Echocardiographic Interpretation

**arXiv ID:** 2610.32880 | [PDF](https://arxiv.org/pdf/2610.32880v1)

**作者:** Yi Qin `[一作]` (Hong Kong University of Science and Technology), Xiaomeng Li `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

开发并评估了EchoChat多模态大语言模型，能够完整解读超声心动图、识别33种心脏疾病、估计22项关键测量并自动生成诊断报告。

**💡 创新点**

创新点在于将超声心动图知识库（EKL）与心脏病学推理模块（CLR）嵌入模型训练，构建链式思考机制，实现可解释且覆盖面广的诊断推理。

**🔧 技术方法**

采用基于Qwen2.5‑VL‑7B的多模态大语言模型，使用视觉编码器与文本解码器相结合的架构，训练时引入指令调优、EKL知识注入和CLR推理监督，支持多任务学习。

**📊 数据集**

使用46,358例来自香港SAR中国的完整超声心动图作为训练集，外部5个中心（荷兰、西班牙、奥地利、内地A、内地B）共计5,095例进行验证，涵盖6个国家/地区。

**📈 对比分析**

与PanEcho和EchoPrime等基础模型对比，EchoChat在4个外部中心的条件识别F1平均提升约16–23%，量化测量的R²提升约0.25；在读者研究中，与常规读者协同报告的主诊F1从72.78%提升至82.75%，诊断准确度与完整度显著提升。

**⚠️ 局限性**

局限性包括对高质量、标准化超声记录的依赖，缺乏真实临床部署与长期跟踪验证；模型对图像质量、视角完整性敏感，需人工审核防止自动化偏差。

---

## 532. Do "True" and "False" Add Up to One? Complement Consistency in LLM-Agent Opinion Simulations

**arXiv ID:** 2610.32874 | [PDF](https://arxiv.org/pdf/2610.32874v1)

**作者:** Anouar Imel `[一作]` (University of Electronic Science and Technology of China), Jin Qi `[通讯]` (University of Electronic Science and Technology of China)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文针对 LLM 代理在意见模拟中的概率更新过程，提出并验证了补语一致性（complement consistency）测试，并对三种小型开源模型在模拟状态、真实事实核查命题及受控基准下的表现进行系统评估。

**💡 创新点**

创新点在于将逻辑一致性约束与代理状态更新耦合，形成可直接测量的补语一致性指标，并通过多维度（调度、证据、同义改写、重复采样、词序、否定等）实验揭示一致性缺陷及其在反馈循环中的累积效应。

**🔧 技术方法**

使用的技术包括：LLM 代理的概率推理接口、补语变换（T(s)）、Brier 失真评估、同步与随机顺序更新、同义改写与否定实验、平均与对称化一致性修复、以及闭环反馈与成本对比。

**📊 数据集**

数据集涵盖：16 条校园事实核查命题（每条正负配对），48 条 AVeriTeC 事实核查案例，24 个受控二元世界（包括颜色、开关、奇偶、阈值、成员、顺序六种结构），以及 6 个子集世界用于后续实验；所有实验均在 Qwen3-4B、Llama-3.2-3B 和 Gemma-3-4B 模型上进行。

**📈 对比分析**

比较方法采用对齐补语配对、同义改写对比、平均绝对差异、Brier 损失、并通过 bootstrap 计算置信区间；结果显示：补语不一致平均绝对误差显著高于同义改写敏感度（多达 2~10 倍），且即使在同步反馈中也会出现正向发散；对称化修复能降低两种措辞下的平均 Brier 损失，但未必优于最佳原始措辞，且计算成本显著提升。

**⚠️ 局限性**

局限性包括：仅测试三种小型模型，未涵盖更大或不同架构的 LLM；实验仅关注数值概率输出，忽略语言表达的多模态差异；对齐与补语变换的手工定义可能导致偏差；在多轮反馈中对随机采样的处理有限；因此结果在更广泛的实际应用和人类行为解释上仍需进一步验证。

---

## 533. J-Comm: Jacobian-Guided Latent Communication between LLM Agents

**arXiv ID:** 2610.32871 | [PDF](https://arxiv.org/pdf/2610.32871v1)

**作者:** Xiangwei Wang `[一作]` (University of Melbourne), Saman Halgamuge `[通讯]` (University of Melbourne)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于几何一致性的连续推理规则GAF以及双通道隐式通信接口H+J，使得冻结的LLM可通过嵌入混合与中间状态读写实现跨模型推理；

**💡 创新点**

创新点在于：①用几何相似度重加权候选嵌入形成下一步输入，提升推理质量；②设计H+J两通道通信，既传递概率分布又传递Jacobian字典读数，兼容不同隐藏宽度且无需共享KV缓存；

**🔧 技术方法**

技术包括：几何一致性反馈（GAF）重加权、Jacob lens字典、Embedding混合、Prefill阶段残差注入、BFS激活与预填写；

**📊 数据集**

实验数据集涵盖GSM8K、MATH500、GPQA‑Diamond、MedQA‑USMLE、EvalPlus MBPP+、AIME2024+2025、ALFWorld，主实验为Qwen3‑8B→Qwen3‑4B跨规模推理；

**📈 对比分析**

与TextMAS、LatentMAS、StateBridge、CIPHER等基线对比，GAF+H+J在七项基准上均取得最高平均分（如MATH500 94.8%、AIME 73.3%），同时生成文本量下降19–75%；

**⚠️ 局限性**

局限性：需要为每对模型预先建立token对应和Jacobian字典；跨词汇表通信需额外字符串映射；自适应推理步骤上限可达32k，推理成本仍较高；

---

## 534. PoKER: A Probability of Kill Estimation Rate Model for Air-to-Air Missiles Using Machine Learning on Stochastic Targets

**arXiv ID:** 2610.32864 | [PDF](https://arxiv.org/pdf/2610.32864v1)

**作者:** Joao P. A. Dantas `[一作]` (Institute for Advanced Studies), Takashi Yoneyama `[通讯]` (Aeronautics Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出PoKER模型，将武器接触区（WEZ）与弹道击中概率结合，用机器学习预测空空导弹的击中概率。

**💡 创新点**

创新点在于：① 将目标机动态机动行为概率化融入WEZ；② 通过近战与远程雷达制导模拟生成百万级仿真数据；③ 将弹头击中概率与机器学习模型结合，形成实时可用的概率估计。

**🔧 技术方法**

技术包括：5DOF弹道仿真（R语言实现）、线性多项式回归、人工神经网络（TensorFlow）和极端梯度提升（XGBoost），并配合特征预处理、交叉验证与早停机制。

**📊 数据集**

数据集为基于LHS采样的1000万条仿真结果，包含发射机/目标机姿态、速度、反应时间等变量及对应的击中概率（由弹头击中模型计算）。

**📈 对比分析**

评估指标为MAE、MSE、RMSE、R²、训练时间与推理时间；统计检验（Friedman + Nemenyi）显示ANN在预测精度上显著优于多项式回归，XGBoost处于中间水平，且训练/推理时间相对可接受。

**⚠️ 局限性**

局限性包括：仅使用仿真数据，缺乏实战验证；目标机机动模型基于专家假设，可能与实际偏离；计算成本高（10M仿真需40天，模型训练耗时数千秒）；模型泛化性待进一步验证。

---

## 535. On the Limits of Sparse Linear Model with Shuffled Differential Privacy

**arXiv ID:** 2610.32822 | [PDF](https://arxiv.org/pdf/2610.32822v1)

**作者:** Yan Zhang `[一作]` (Nankai University), Di Wang `[通讯]` (King Abdullah University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究了稀疏线性回归在shuffled differential privacy（SDP）下的估计误差，并给出了非交互式和交互式两种协议的理论上界与下界，进一步提出了两轮用户级交互式算法U2R。

**💡 创新点**

创新点在于：①揭示非交互式SDP仍受维数d的多项式依赖，②设计两轮交互式用户级算法在足够多样本下实现无d依赖的误差上界，③将score attack框架扩展到用户级数据，给出近似最优下界。

**🔧 技术方法**

使用的技术包括shuffled DP框架、随机化响应、heavy hitter检索、soft‑thresholding、Lasso、最小二乘估计，以及用户级score attack进行下界证明。

**📊 数据集**

实验使用合成数据以及公开真实数据集Kelmarsh进行评估。

**📈 对比分析**

与非私有Lasso、局部DP（LDP）以及中心DP方法对比，U2R在相对系数误差和测试MSE上表现最好，尤其在较小隐私预算（ε=1,4）下显著优于其它私有方法。

**⚠️ 局限性**

限制在于：非交互式协议仍存在维数d的多项式误差；交互式算法要求每用户拥有足够多样本并需要两轮交互；理论下界中仍保留log(d/k)因子，未完全消除维度对误差的影响。

---

## 536. Logarithmic Separation between LZ77 and LZ-End and Grammar Compression

**arXiv ID:** 2610.32818 | [PDF](https://arxiv.org/pdf/2610.32818v1)

**作者:** Philip Bille `[一作]`, Renfei Zhou `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一个二进制字符串族T_k，证明在最优的LZ‑End解析中所需的短语数比最优非重叠LZ77解析至少大对数量级，从而给出了LZ‑End与LZ77以及语法压缩之间的对数级分离。

**💡 创新点**

主要创新在于构造了一个确定性、可实现的二进制字符串族，并通过“snapshot”技术与位反转排列，给出了对数级的下界；同时给出了相应的上界，完成了LZ‑End与语法压缩的紧确分离。

**🔧 技术方法**

利用了快照构造、平衡二叉树事件计数、充电（charging）论证、位编码与对数级递增的排列（bit‑reversal）等理论工具，以及LZ77到语法的转换方法。

**📊 数据集**

使用的是作者自己构造的专用字符串族T_k（大小约为Θ(k^2 log k)），并未使用公开的真实数据集。

**📈 对比分析**

通过比较最优非重叠LZ77短语数 z_no 与最优LZ‑End短语数 z_end 的比值，得到 z_end/z_no ≥ Ω(log n)；同理对语法压缩大小 g 与 z_no 的比值也得到 Ω(log n)。

**⚠️ 局限性**

该结果仅为理论上最坏情况的下界；并未给出在实际文本或标准压缩数据集上的实验评估，且仅适用于二进制字符串，限制了其直接的实践适用性。

---

## 537. Beyond pass or fail: an information-processing account of the Turing Test

**arXiv ID:** 2610.32868 | [PDF](https://arxiv.org/pdf/2610.32868v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 538. SREdit: Scene Response Guided Appearance Editing for Projection Mapping

**arXiv ID:** 2610.32792 | [PDF](https://arxiv.org/pdf/2610.32792v1)

**作者:** Yuqiao Xia `[一作]` (Southwest University), Bingyao Huang `[通讯]` (Southwest University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于场景响应的编辑框架 SREdit，在不修改预训练图像编辑器的前提下，通过在采样早期引入源图像与场景响应引导，生成更易于投影重现的目标图像，然后用现有投影补偿技术实现物理投影。

**💡 创新点**

核心创新在于：①利用可微分场景渲染器估计场景响应地图，量化投影光照对不同表面区域的影响；②在扩散采样的前期引入响应引导的软约束，既保持与物体的几何配准，又根据响应强度自适应控制编辑幅度；③完全不需要额外训练或控制模块，且不向编辑器反向传播梯度。

**🔧 技术方法**

关键技术包括：Gaussian‑based differentiable scene renderer、预训练扩散图像编辑器（Qwen‑Image‑Edit‑2511、LongCat‑Image‑Edit）、响应引导采样（早期控制、权重调度）、投影补偿的迭代优化，以及 CLIP/BLIP‑ITM 等评估指标。

**📊 数据集**

实验数据基于六套真实投影摄像机系统，包含 18 条文本编辑提示（Qwen/LongCat）和 12 条参考图像编辑案例；没有使用公开的标准数据集，而是自行构建的物理场景与投影记录。

**📈 对比分析**

与 Nepmap、LAPIG 以及相同编辑器的 Native 版本进行对比。使用 PSNR、SSIM、LPIPS、ΔE、CLIP、BLIP‑ITM、生成时长与 VRAM 作为指标。SREdit 在 PSNR、SSIM 等重现度量上提升 5–6 dB，CLIP/BLIP‑ITM 也显著提高；用户研究显示 72.4% 的整体偏好；额外计算时间不到 1 秒，内存占用微幅增加。

**⚠️ 局限性**

局限性包括：①需要手动调节早期控制比例 r 与强度 S，难以在所有场景中自动平衡重现与纹理丰富度；②预训练编辑器仍占主导的计算与内存成本；③响应引导为软约束，极端情况可能导致纹理被过度抑制，缺乏硬约束机制。

---

## 539. SPOOK: Sparse Pixel Optimization for Off-Manifold Knowledge in Vision Encoders

**arXiv ID:** 2610.32835 | [PDF](https://arxiv.org/pdf/2610.32835v1)

**作者:** Ankit Gupta `[一作]` (Michigan State University), Emily Dolson `[通讯]` (Michigan State University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出SPOOK方法，利用稀疏像素编辑在空白画布上生成可在现代视觉编码器中检索到的稀疏非语义图像。

**💡 创新点**

首次证明稀疏非语义图像可在CLIP和DINO等视觉编码器中获得高检索成功率，揭示了编码器嵌入空间与人类可识别语义的脱节。

**🔧 技术方法**

采用黑盒查询式的稀疏像素hill‑climbing优化，结合余弦相似度目标，针对CLIP文本提示、Flickr30k字幕原型以及DINOv3图像原型进行目标对齐。

**📊 数据集**

使用ImageNet和Flickr30k的1000类/图片作为检索目标集，并对OpenCLIP ViT‑B/16/32以及DINOv3 ViT‑S/16/32/64进行实验。

**📈 对比分析**

通过ASR@1/5和平均/中位数目标余弦相似度与排名指标进行评估，SPOOK在CLIP/ImageNet和Flickr30k以及DINOv3/ImageNet上均达成超过90% top‑1成功率，且仅修改低于1%的像素。

**⚠️ 局限性**

SPOOK易受模型特定，迁移性差，对常见变形（压缩、裁剪、模糊）不稳健，且不具备鲁棒性与普适性。

---

## 540. Advancing SVD-based LLM Compression via Layer-Wise Error Model Search

**arXiv ID:** 2610.32878 | [PDF](https://arxiv.org/pdf/2610.32878v1)

**作者:** Moritz Thoma `[一作]` (Technical University of Munich), Ulf Schlichtmann `[通讯]` (Technical University of Munich)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了基于层级误差模型的 SVD 压缩方法 LEMS（oursearch）以及 token‑wise Fisher 估计的 Robust Fisher Decomposition（ours）来提升 LLM 的低秩压缩效果。

**💡 创新点**

创新点在于：1）引入层级误差模型，结合局部敏感性、经验校准和深度偏置，构造可分解的误差代理；2）通过整数规划一次性得到全局最优秩分配；3）使用 token‑wise 统计替代传统序列梯度估计，缓解 Fisher 低秩瓶颈。

**🔧 技术方法**

核心技术包括：低秩 SVD 近似、Kronecker‑factored Fisher 近似、层级误差模型（shape、scale、bias）、ILP（整数规划）求解、KL/ MSE 误差校准。

**📊 数据集**

在多种 LLM 家族（Mistral、Qwen‑3、Llama‑3）上，使用 Wiki、PTB、C4 文本以及 PIQA、OpenBookQA、HellaSwag、ARC‑Easy/Challenge、Winogrande 等零样本基准进行评估。

**📈 对比分析**

与现有的贪心、MRCs、ASVD、SVDLLMv2、ARS、ATP 等搜索策略以及 fwsvd、asvd、svdllm、dobisvd、gfwsvd 等 SVD 基线相比，oursearch 在大多数压缩比下可提升 1.7–4.8 百分点准确率，平均 perplexity 减少约 15%，在 70B Llama‑3 上更是将 perplexity 从 7.9 降至 4.5、准确率提升至 66.6%。

**⚠️ 局限性**

局限性包括：需要梯度计算导致计算成本较高；one‑time calibration 随模型深度线性增长；对非常大模型仍需进一步评估；未覆盖 VLM 或 MoE 结构。

---

## 541. Reimagining Enterprise RAG for Multi-Agent Systems: A Knowledge Layer That Organizes Before It Retrieves

**arXiv ID:** 2610.32851 | [PDF](https://arxiv.org/pdf/2610.32851v1)

**作者:** Garima Agrawal `[一作]` (Cognizant), Prasad Kalyanasundaram `[通讯]` (Cognizant)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建并部署了KB Cataloging知识层，利用逆向工程生成的业务规则等artifact对其进行分类、注册、层级化组织，并在生产环境中为多种SDLC代理和开发者提供精确的业务规则检索与根因分析。

**💡 创新点**

创新点包括：① 将本体论迁移到artifact层，使用自描述目录、稀疏元数据注册表和基于artifact的导航超图；② 在检索前对候选集做结构化约束，避免因语义冲突导致的误检；③ 通过写回学习动态更新元数据并支持无schema扩展。

**🔧 技术方法**

采用逆向工程pipeline、向量嵌入（混合 dense + BM25）、意图分类、超图导航、LLM置信度门控、write‑back元数据、CLI/REST/浏览器接口等技术；核心检索实现为在已约束的artifact集合上进行向量搜索。

**📊 数据集**

使用来自241个代码仓库的18个服务生成的数千个结构化artifact（scope、journey、业务规则、验收标准、gap register等），全部由机器生成并保留完整 provenance。

**📈 对比分析**

通过构建时长(<1周)和无schema扩展成本评估系统可用性；在生产中已分析并解决超过100个缺陷，归因35个回归失败；检索精度通过“扩宽率”与人工标注的模糊度加权评估，结果显示候选集显著缩小，检索成本降低，latency维持在可接受范围。

**⚠️ 局限性**

局限性：仅适用于包含完整 provenance 的机器生成 corpus；依赖 upstream 逆向工程的覆盖率，未包含运行时配置或手工编写的规则；检索精度受 artifact 放置质量影响，未单独测量 placement precision；单一企业 estate，评估结果可能受专业人员偏见影响。

---

## 542. On Latency-Optimal Implementations of the Timestamp Object

**arXiv ID:** 2610.32844 | [PDF](https://arxiv.org/pdf/2610.32844v1)

**作者:** Aurojit Panda `[一作]` (New York University), Wen Zhang `[通讯]` (Google)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究在异步消息传递系统中，针对崩溃和拜占庭失效模型下的时间戳对象实现，给出了可容忍的最大失败数与最优延迟的理论极限，并在除单一边界情况外实现了匹配的上界与下界；

**💡 创新点**

创新点在于提出了新的“循环链”下界证明技术，首次实现了在f≥2时的三延迟实现，并证明在f=1时可降至两延迟；

**🔧 技术方法**

主要技术包括线性化规范、可容忍的广播与计数协议、loquacious（全广播）算法转化、以及基于签名与无签名的拜占庭容错协议设计；

**📊 数据集**

为理论论文，未使用实验数据集，所有结果均为严谨的形式化证明；

**📈 对比分析**

通过严格的下界证明与匹配的构造算法，本文展示了在不同失败模型下的最优延迟（崩溃模型下两/三延迟，带签名拜占庭下两/三延迟，无签名拜占庭下三延迟），并指出唯一未完成的情形；

**⚠️ 局限性**

局限在于对无签名拜占庭模型中 n>4、f=1 的情形仍未决定最优延迟是两还是三；此外，本工作仅考虑异步无锁定网络，未涵盖同步或带有可靠性约束的环境。

---

## 543. Multi-Timescale Context-Aware Radio Resource Optimization for Harmonious NTN-TN Coexistence: A Digital Twin and Open RAN Enabled Framework

**arXiv ID:** 2610.32827 | [PDF](https://arxiv.org/pdf/2610.32827v1)

**作者:** Ecem Izde Kaya `[一作]` (Universite du Quebec), Georges Kaddoum `[通讯]` (Universite du Quebec)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 DT-STABLE，一个基于数字孪生和 O-RAN 的多时标资源调度框架，用于非地面网络与地面网络的共存。

**💡 创新点**

创新点在于：①使用基于场景的 CKM（覆盖和干扰图）实现预测式干扰管理；②将干扰约束与负载均衡统一入联合束激活和用户关联优化，并证明问题为 NP‑hard；③设计基于价格的双层迭代算法，实现低复杂度的实时调度；④在 O‑RAN 规范下实现多时标闭环交互，显著降低测量开销。

**🔧 技术方法**

核心技术包括：数字孪生建模（Sionna ray‑tracing、OpenNTN）、O‑RAN 接口（E2、A1、O1）、多时标调度（短时、短时中、短时长），以及基于 Lagrangian 的价格更新算法。

**📊 数据集**

使用了基于开源数据的场景：Quebec City 4×4 km 区域的 19 台宏站位置信息、Starlink 20 台 LEO 卫星的 TLE 数据以及 3GPP 低层统计模型，构建了高保真 CKM 与干扰图。

**📈 对比分析**

通过与三类基线（5G 标准、无负载感知、Per‑BS 干扰抑制）在 INR 满足率、平均速率、测量开销与计算时延等指标进行对比。结果显示 DT-STABLE 在 100 个 UE 的情形下实现 89.63% 的 INR 满足率、210.96 bps/Hz 的平均速率，测量开销比全扫低 99.71%，计算时延比 Gurobi 优化快 15–33 倍，且保持接近最优性能。

**⚠️ 局限性**

局限性包括：①依赖高精度 UE 位置信息和干扰预测，误差会影响性能；②假设相邻基站互斥且内部 beam 互不干扰，实际环境中可能需要考虑互斥；③仅针对下行共存，未考虑上行或多卫星多链路场景；④双层算法为启发式，无法保证全局最优。

---

## 544. U-NeXt: General-purpose event detection for neonatal EEG

**arXiv ID:** 2610.32811 | [PDF](https://arxiv.org/pdf/2610.32811v1)

**作者:** Robert Hogan `[一作]`, John M. O'Toole `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `6514db3d-8de6-452c-91b7-acdb31787cc4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

开发了一种基于U-Net与ConvNeXt的通用神经网络U‑NeXt，用于新生儿脑电图（EEG）中的癫痫发作、低电压间隔（LVI）和伪影等多种事件的检测；

**💡 创新点**

创新点在于将语义分割直接用于EEG事件边界标注，避免短时分类的时间分辨率和上下文缺失；通过统一的单通道输入架构实现跨任务可迁移性，并在三种不同事件类型上保持相同的网络结构与训练配置；

**🔧 技术方法**

采用1D ConvNeXt编码器+轻量化解码器、深度监督、对称Dice+ BCE复合损失、AdamW+单周期余弦退火、EMA权重平均等技术，并配合数据增强（翻转、幅度抖动、cutout）；

**📊 数据集**

使用329名新生儿的NeoBase多通道EEG数据（含扩充的Seizure+数据）进行训练，并在独立的Helsinki（79名）和Cork（51名）测试集上评估癫痫发作检测；同时构建LVI与伪影专用开发集；

**📈 对比分析**

与先前的ConvNeXt基线和公开SVM模型比较，开发集上MCC提升至0.821（相较0.799），在Helsinki和Cork的独立测试集上实现专家级等价（Δκ 0.368/0.470），并在癫痫负荷预测上相对提升30‑40%；推理速度比ConvNeXt提升约250倍；

**⚠️ 局限性**

局限性包括仍需为每种事件训练单独模型；缺乏多任务联合训练和更广泛事件类型的验证；对伪影多样性处理尚不充分；未进行真实临床部署与实时评估。

---

## 545. Noisy Target Training Meets Drifting for Weakly-Supervised Speech Enhancement

**arXiv ID:** 2610.32870 | [PDF](https://arxiv.org/pdf/2610.32870v1)

**作者:** Yoshiki Masuyama `[一作]` (Mitsubishi Electric Research Laboratories), Jonathan Le Roux `[通讯]` (Mitsubishi Electric Research Laboratories)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文提出将无监督的Sinkhorn漂移损失与NyTT（噪声目标训练）相结合，构建一种弱监督语音增强方法。

**💡 创新点**

创新点在于：①利用无配对的干净语音在SQA模型的潜在空间中计算Sinkhorn漂移场，鼓励增强语音分布逼近干净语音分布；②将此分布层次的监督与NyTT的样本层监督（含平均教师）耦合，使模型既保持对观测语音的忠实，又减少残留噪声。

**🔧 技术方法**

技术要点包括：噪声目标训练（NyTT）+平均教师框架；Sinkhorn漂移模型（基于Sinkhorn散度的梯度流）；使用UrgentMOS作为语音质量评估（SQA）模型的特征编码器；在潜在空间计算漂移场并对增强语音做欧氏损失。

**📊 数据集**

实验数据集：语音增强合成数据VoiceBank‑DEMAND（VBDM）用于训练与评估；真实环境数据CHiME‑3用于域适应与测试；无配对干净语音来源为EARS/VoiceBank等。

**📈 对比分析**

与传统CTT、DriftSE、unSE、RemixIt、SelfSE等方法对比，本文方法在VBDM上：SI‑SDR从6.6 dB提升至≈18.6 dB，DNSMOS/UTMOS显著提升；在CHiME‑3上，WER从9.28 %进一步降至9.12 %（仍高于噪声基线），DNSMOS/UTMOS亦提升，说明方法在提升感知质量的同时保持一定的识别性能。

**⚠️ 局限性**

局限性包括：①在真实录音上仍未能完全消除增强产生的误差，导致WER高于原始噪声信号；②对SQA模型的依赖使训练复杂度上升；③未对不同噪声环境、说话人多样性进行充分泛化验证，可能对特定噪声类型敏感。

---

## 546. Trusting the Tool: Asymmetric Deference of LLM Vulnerability Auditors to Static-Analysis Evidence

**arXiv ID:** 2610.32876 | [PDF](https://arxiv.org/pdf/2610.32876v1)

**作者:** ZhiHong Liu `[一作]` (Seoul School of Integrated Sciences and Technologies aSSIST University), Shuyao Gao `[通讯]` (Seoul School of Integrated Sciences and Technologies aSSIST University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了大型语言模型（LLM）在审核 C/C++ 函数时如何利用静态分析生成的证据，并系统量化了真伪证据及其呈现方式对模型判定的影响。

**💡 创新点**

创新点在于提出“方向感知的偏误度量”来区分假保证与假警报对判定的不同影响，发现模型对假保证易产生偏信、对假警报更倾向检查，并验证分步验证策略能显著降低偏误。

**🔧 技术方法**

使用了基于树解析器的程序分析器产生的证明义务、LLM（DeepSeek、Qwen 等）进行推断、Prompt engineering 的审计任务以及 verify‑show 等分步验证技术，并采用 Bootstrap 统计方法评估结果。

**📊 数据集**

采用了公开的 PrimeVul 数据集的漏洞修复对（共 916 对，四个互斥子集），每个函数不超过 150 行的 C/C++ 代码作为实验对象。

**📈 对比分析**

通过配对 McNemar 检验和 Bootstrap 置信区间比较不同证据条件与呈现策略的准确率，结果显示真证据提升不足 2%，假保证降低 11–19 分，假警报不显著影响；分步验证可提升 12 分准确率。

**⚠️ 局限性**

局限性包括仅研究局部 C/C++ 函数、使用自研分析器、模型仅以推断调用、未覆盖交叉过程/多语言场景，以及验证策略成本较高且对不同模型的效果不一。

---

## 547. Sustainability of TSFM Deployment and Pretraining: From Carbon Footprint Analysis to Mitigation via Pruning

**arXiv ID:** 2610.32859 | [PDF](https://arxiv.org/pdf/2610.32859v1)

**作者:** Cécile Luc `[一作]` (Schneider Electric), Ievgen Redko `[通讯]` (Schneider Electric)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了时间序列基础模型（TSFM）Chronos-2 在工业部署生命周期中的碳排放成本，并与传统监督模型 LightGBM、PatchTST 对比；同时提出了一种基于余弦相似度的结构化剪枝方法，去除单个编码器块，保持零样本预测性能的同时降低推理成本。

**💡 创新点**

①构建了生命周期碳成本模型，明确了在不同资产规模、查询频率和数据漂移条件下，Chronos-2 的预训练成本是否能被摊销；②提出了无梯度余弦相似度剪枝策略，能够在不降低预测性能的前提下实现 6–9% 的 FLOPs、CO₂ 与延迟下降。

**🔧 技术方法**

使用 CodeCarbon 进行碳排放量化；基于输入-输出余弦相似度的块级结构化剪枝；Zero-shot 预测评估；通过 FLOPs、CO₂、延迟等指标量化模型效率；使用几种数据漂移与查询频率场景进行成本比较。

**📊 数据集**

Chronos Benchmark II 中 25 个零样本时间序列预测任务（覆盖 30 分钟到年度频率，多个领域），以及用于校准的 6 个任务；并在实际工业部署场景（如数据中心 UPS 或电力变压器）进行碳成本模拟。

**📈 对比分析**

通过累计碳成本曲线与 FLOPs 比较，发现 LightGBM 在资产数少或查询频率低时始终更便宜，而在资产数千、查询频率高时 Chronos-2 能在数月至数年内摊销预训练成本。剪枝后，MASE 技能分数仅下降 1.1 点（约 5%），WQL 技能分数略有提升；FLOPs、CO₂ 与延迟均下降 6–9%。

**⚠️ 局限性**

研究仅聚焦于 Chronos-2，预训练成本采用单一步骤估计，可能低估真实成本；剪枝策略仅在零样本场景验证，未探讨 fine‑tune 后的效果；碳排放评估基于单一电网碳强度，可能不适用于其他地区；实验环境（GPU、CPU）有限，未考察不同硬件加速的影响；未考虑更细粒度的剪枝或混合精度等压缩技术。

---

## 548. HarnessMM: Recursive Self-Improvement for Harness Token Efficiency

**arXiv ID:** 2610.32813 | [PDF](https://arxiv.org/pdf/2610.32813v1)

**作者:** Yuhua Jiang `[一作]` (Tsinghua University), Li Zhao `[通讯]` (Microsoft Research Asia)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 HarnessMM，通过在代理-工具交互中存储已解析的决策并在后续工具调用中重用，避免不必要的推理回调，从而提升令牌效率并保持或提升任务成功率。

**💡 创新点**

核心创新在于可执行决策图（EDG）与受控决策重用（GDR）相结合，以及认证回调替换（CCS），使已决策可跨反馈边界复用；同时引入递归自我改进（RSI）迭代优化伴随配置。

**🔧 技术方法**

技术包括：可执行决策图（EDG）结构、受控决策重用（GDR）与认证回调替换（CCS）机制、递归自我改进（RSI）循环、公共状态适配器与回调验证证书。

**📊 数据集**

在多个编程与推理基准上评估：HumanEval、MBPP、SearchQA、LiveMath、AIME、Terminal-Bench、GAIA2 等。

**📈 对比分析**

与原始 Codex 与 DeepSeek Harness 对比，HumanEval 成功率保持 90.82% 同时令牌下降 29.54%，DeepSeek Harness 成功率提升至 85.71% 同时令牌下降 87.93%；MBPP 通过 GDR/CCS ablation 显示 26.91% 及 19.52% 的成本削减。

**⚠️ 局限性**

受限于：收益高度依赖输入提供商、缓存与上下文计费策略；需要手工设计公共状态适配器；小型训练集可能导致策略级质量差异不确定。

---

## 549. Low-Crossing Spanning Paths via Quasi-Uniform Sampling and their Applications

**arXiv ID:** 2610.32806 | [PDF](https://arxiv.org/pdf/2610.32806v1)

**作者:** Mónika Csikós `[一作]` (University Paris Cité), Nabil H Mustafa `[通讯]`

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `afceb026-1760-41ae-8d86-010831a37d97`

**🎯 论文内容**

提出了一种新的随机化算法，利用准均匀采样框架在一般集合系统中构造低交叉匹配和跨越路径，期望时间为 Õ(|X|^2 + |S|·|X|)。

**💡 创新点**

创新点在于只维护范围集合上的权重分布，迭代时先随机选取一个范围，再在其对应的 Venn 图细胞内均匀抽取边，从而避免了对所有 O(|X|^2) 条边维护权重的昂贵操作。该方法实现了在 |S| = o(|X|^{1+2/d}) 的重要情形下的最快运行时间。

**🔧 技术方法**

核心技术包括：
- 采用指数加权更新（MWU）在范围集合上维护权重；
- 通过 Venn 图细胞（equivalence classes）快速更新被覆盖/被交叉的子集；
- 双重计数和马氏过程分析证明交叉数上界；
- 通过分阶段调用子算法（subalgo）并用几何级数控制重启次数。

**📊 数据集**

论文主要为理论性质的证明，未使用具体实验数据集；所讨论的应用场景（例如球体、图邻域、预测市场）仅用于说明算法的潜在效益。

**📈 对比分析**

与以往最优算法相比，新的算法在 |S| = o(|X|^{1+2/d}) 时将期望运行时间从 Õ(|X|^2+2/d+|S|·|X|^2/d) 降低到 Õ(|X|^2+|S|·|X|)。这导致在组合差异、图邻域覆盖、球体匹配等问题上获得了更快的实现，时间复杂度从高次多项式降到几乎线性。实验验证未给出，但理论分析显示显著改进。

**⚠️ 局限性**

局限性包括：
- 算法依赖于集合系统的多项式上界 dual shatter 函数，实际应用需先验证此假设；
- 仍是期望时间分析，实际运行时间受随机选择影响；
- 仅适用于交叉数优化，若目标是平均交叉数或其他度量，则需要进一步改进；
- 没有提供实证评估，难以评估常数因子与实现复杂度。

---

## 550. Choosing What to Observe: Higher-Order Obstructions in Horn Abduction

**arXiv ID:** 2610.32805 | [PDF](https://arxiv.org/pdf/2610.32805v1)

**作者:** Yukun Yang `[一作]` `[通讯]` (Beijing Institute of Technology), Yukun Yang (Beijing Institute of Technology)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究在有限命题 Horn 归纳（含完整约束）中，如何通过观察源变量来保证对所有可能上下文都能做出统一的解释承诺；提出了“最小空交”阻塞集、对应的观察合同（超图转置）以及在不同表示方式下的复杂度上界与下界。

**💡 创新点**

创新点在于将统一解释问题转化为超图转置问题，给出了阻塞集的结构性描述、对矩形与无完整约束情况的精细分析，以及通过观测选择逼迫全局存在唯一承诺的多维量化复杂度证明；同时展示了独立组件可合成观察合同的理论。

**🔧 技术方法**

主要技术包括 Horn 归纳的闭包运算、超图转置与双向最小化、量化逻辑归约（到 Π₃⁽ᴾ⁾/Σ₄⁽ᴾ⁾）、电路到 Horn 理论的多态编码、以及对可列/可电路上下文的多种复杂度分析。

**📊 数据集**

论文主要是理论性工作，没有使用公开数据集；所有实例均为合成的 Horn 规则与上下文集合。

**📈 对比分析**

通过与已知的 Horn 归纳、判定与求解量化公式的复杂度比较，证明了在电路定义的上下文中，观察足够性是 Π₃⁽ᴾ⁾-完全的，观察选择是 Σ₄⁽ᴾ⁾-完全的；在可列上下文与已表解释族时，问题降为 NP-完全；在已表解释族时足够性可在多项式时间内判定。

**⚠️ 局限性**

主要局限包括：对大规模实际问题缺乏可扩展实现；复杂度极高，导致仅在理论分析上可行；未考虑噪声观测、序列化感知、概率/因果偏好等实际需求；并且对完整约束和矩形结构的适用性有限。

---

## 551. Frontier-OPD: Discovering the Edge of Agent Capability and Teaching Agents to Go Beyond

**arXiv ID:** 2610.32802 | [PDF](https://arxiv.org/pdf/2610.32802v1)

**作者:** Zheng Jiang `[一作]` (Tsinghua University), Lifeng Sun `[通讯]` (Tsinghua University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Frontier-OPD，基于学习前沿的多轮交互式监督获取框架

**💡 创新点**

创新点在于将学习前沿（学生尚未掌握但可通过短期教师桥接实现的决策场景）与连续验证相结合，利用共享状态图进行轨迹平衡的候选选择与窗口匹配验证

**🔧 技术方法**

使用共享状态图、轨迹平衡加权、教师-学生对齐的符号不一致度、匹配窗口同义度、OPD（on‑policy distillation）更新等技术

**📊 数据集**

在 ALFWorld、WebShop、ScienceWorld 三个交互式基准上进行实验，使用 Qwen3 系列大模型作为教师与不同规模学生

**📈 对比分析**

与 OPD、TCOD、Guided‑OPD、FutureBridge‑OPD 等方法对比，Frontier‑OPD 在三种环境中均实现了最高平均自主执行成功率，尤其在小学生规模下提升显著；在训练效率上也能更快达到设定成功阈值

**⚠️ 局限性**

局限包括：学习前沿估计依赖行为代理且不一定反映真正能力；只考虑共享状态，不利用图邻接；需要可重放环境与教师 token 概率；实验仅覆盖三种基准，缺乏统计显著性与更广泛泛化验证

---

## 552. CARES: Coverage-Aware Replay Exemplar Selection for Continual Learning

**arXiv ID:** 2610.32799 | [PDF](https://arxiv.org/pdf/2610.32799v1)

**作者:** Arundhati S. Shanbhag `[一作]` (University of Kaiserslautern-Landau), Andreas Dengel `[通讯]` (University of Kaiserslautern-Landau)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 CARES，一种基于冻结预训练特征空间的无优化连续学习缓冲区构建方法，利用类感知、密度加权覆盖在特征空间中选择样本，完全不依赖模型梯度或损失；

**💡 创新点**

创新点在于将 SubZeroCore 的密度感知覆盖方法改造为在线连续学习场景，加入类分配预算和暂存池，实现无优化、低计算成本的缓冲区更新，并在不平衡和标签噪声流下保持高性能；

**🔧 技术方法**

技术手段包括：冻结预训练 Inception‑V3（或其它）特征提取、类感知预算分配、密度加权 Facility‑Location 目标、贪心近似求解、投影到欧氏空间、使用 ρ‑近似邻域覆盖、Reservoir Sampling 维护暂存池；

**📊 数据集**

使用的公开图像分类基准为 Split‑CIFAR‑10、Split‑CIFAR‑100 和 Split‑Tiny‑ImageNet，分别在任务增量、类别增量、平衡、长尾不平衡和标签噪声的多种流配置下进行评估；

**📈 对比分析**

与 ER、GEM、GSS、iCaRL、MIR、DER、GCR、BCSR、OCS、lilac 等传统与最新共现缓冲方法对比，CARES 在类不平衡和标签噪声场景下平均准确率最高（比强基线提升 4–8.6 分），在大容量缓冲下保持或超过最强基线；同时其缓冲构建速度比 GCR、BCSR、OCS 快 2.3–8.8 倍，呈现优异的准确率–效率权衡；

**⚠️ 局限性**

局限性包括：在极小缓冲容量（仅保留少量样本）下，基于学习者梯度的选择方法仍可能更优；对冻结特征空间的依赖性较强，若预训练模型不合适会影响选择质量；暂存池增加额外内存占用；在极端小容量或高度噪声的流中仍需改进。

---

## 553. From Deliberately Constructed Systemic Representations of the World of Activity to Dialogue: A Five-Level Communication Architecture for Strong Language-Based ASI

**arXiv ID:** 2610.32869 | [PDF](https://arxiv.org/pdf/2610.32869v1)

**作者:** Maxim Osovsky `[一作]` `[通讯]` (Adam University), Maxim Osovsky (Adam University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并阐述了一个五层级的交流架构，强调在强语言基人工智能中，回应必须以系统化的活动表征为基础，而非仅靠上下文或模型权重；

**💡 创新点**

核心创新在于将“系统化的活动表征”拆分为五个互相关联的层级（系统化表征、情境概念模型、叙事描述、完整语义和文本），并引入可逆的从语义到系统表征的重构流程，公开并可检查回应的基础；

**🔧 技术方法**

技术层面主要基于系统论、结构差分、意义-文本理论的理论框架，并提出了对语言模型内部表示与系统表征对齐的概念性方法；

**📊 数据集**

未使用特定训练数据集，论文主要是理论与架构设计，但在后续研究与评估建议使用 Ambig‑SWE、Dialogue‑SWE‑Bench、QuestBench 等基准来检验情境缺失信息获取与语义重构能力；

**📈 对比分析**

比较方法：通过在上述基准任务上比较传统文本生成与五层级重构生成的结果，评估情境重构准确性、语义完整性、澄清与修正能力；由于论文尚未实现具体系统，未给出定量性能指标；

**⚠️ 局限性**

局限性：需要对大型系统化活动模型进行构建与重构，当前 LLM 并未具备完整的系统表征能力，架构尚在理论阶段，实际实现与评估仍需后续工作；

---

## 554. Interpretable sparse fine-tuning with TopKLoRA enables faithful circuit discovery

**arXiv ID:** 2610.32866 | [PDF](https://arxiv.org/pdf/2610.32866v1)

**作者:** Marek Masiak `[一作]` (University of Oxford), Constantin Venhoff `[通讯]` (University of Oxford)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出可解释的稀疏LoRA（TopKLoRA），通过在LoRA瓶颈加入ReLU+TopK机制，使模型在微调后能定位并消除基于触发器的后门；

**💡 创新点**

创新点在于构建可解释的稀疏适配器，能够将后门功能压缩到极小且可追踪的电路，并通过对比性潜变量电路发现（CLCD）实现高效定位；

**🔧 技术方法**

主要技术包括TopKLoRA稀疏适配器、ReLU+TopK激活筛选、对比性潜变量电路发现算法、梯度路由等；

**📊 数据集**

使用带有4.8%毒化样本的指令跟随数据集（如Alpaca、No-Robots等）进行实验；

**📈 对比分析**

与稠密LoRA对比，TopKLoRA产生的电路体积缩小约一秩，去除后门后模型保持99.9%后门消除率且指令遵循性能基本不降；

**⚠️ 局限性**

局限性包括仅在≤7B模型上验证、仅使用语法触发器、需要已知触发器、未对单个潜变量进行可解释性分析等。

---

## 555. A Foundation Model for Policy In-Context Longitudinal Causal Effect Estimation

**arXiv ID:** 2610.32863 | [PDF](https://arxiv.org/pdf/2610.32863v1)

**作者:** Wenxin Chen `[一作]` (Cornell University), Peniel Argaw `[通讯]` (Microsoft Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出一种基于in‑context学习的PICLE‑PFN基础模型，用于从观测数据估计纵向因果效应（CAPO），并在部署时无需重新训练即可处理静态和动态治疗策略。

**💡 创新点**

创新点在于：①将目标治疗策略用在观测历史上的决策演示（policy demonstrations）表示，从而实现通用的策略输入；②在合成的纵向结构因果模型（SCM）上预训练，直接学习多步CAPO，省去本地Nuisance回归；③通过单前向推理一次性输出不同时间窗口的CAPO和不确定性，显著提升效率。

**🔧 技术方法**

采用Prior‑Data Fitted Network（PFN）变压器架构，结合因果时间注意力和行注意力；在合成的纵向SCM和目标策略空间上进行任务采样和预训练；使用synthetic rollout生成多步潜在结果作为监督；在推理时使用in‑context学习获取CAPO预测和置信区间。

**📊 数据集**

实验使用七套合成/半合成纵向数据集（包含不同维度的治疗与协变量，静态/动态策略），以及真实MIMIC ICU数据（vasopressor/ventilation 作为治疗，收缩压预测为因果结果）作为事实结果的真实性检验。

**📈 对比分析**

与九个基线（DeepACE、DeepLTMLE、D3‑Net、CRN、RMSN、TE‑CDE、CT、G‑Net、IGC‑Net）在RMSE、覆盖率和推理时长进行对比。PICLE‑PFN在6/7个设置中获得最低RMSE，平均比最优基线低12–15%，推理速度比CT快约100倍、比IGC‑Net快5000倍；并在不确定性校准方面明显优于IGC‑Net。

**⚠️ 局限性**

局限性包括：①计算复杂度为O(N²T)（N为样本数，T为时间步长），限制大规模数据扩展；②仅支持二值治疗、离散时间、无删失；③对前置识别假设（如顺序可忽略性）高度依赖，前期分布偏差或未观测混杂可能导致性能下降；④对策略先验的敏感性尚未完全量化。

---

## 556. Regularised RL improves sim-to-real transfer

**arXiv ID:** 2610.32855 | [PDF](https://arxiv.org/pdf/2610.32855v1)

**作者:** Danilo Vucetic `[一作]` (Université de Montréal), Jonathan Tremblay `[通讯]` (NVIDIA)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发并评估了 Soft MellowMax Actor-Critic（SMM-AC）算法，在仿真与真实机器人上测试 sim-to-real 迁移与鲁棒性。

**💡 创新点**

创新点在于将 Soft MellowMax 价值函数推广到连续控制，并通过慢更新的参考网络实现可训练的 SMM-AC，从而扩展鲁棒集并提升迁移性能。

**🔧 技术方法**

技术包括基于 SAC 的 actor‑critic 框架、soft mellowmax 正则化、参考网络、Monte Carlo 目标估计，以及 FastSAC/FlashSAC 的加速训练。

**📊 数据集**

数据集涵盖 MuJoCo 连续控制基准（如 HalfCheetah、Ant 等）、Franka Panda 机器人的抓取与提升任务，以及 ANYmal‑D 四足机器人的平地与粗糙地面速度跟踪。

**📈 对比分析**

通过与 PPO、SAC、FastTD3 等基线比较，SMM-AC 在 MuJoCo 上与 SAC 相当，在真实机器人上在 sim-to-real 迁移及噪声、质量偏移、摩擦下降等扰动下保持最高回报并最小退化。

**⚠️ 局限性**

局限性包括需要手动调节温度参数、缺乏自适应调度；在某些 MuJoCo 任务（如 Humanoid）训练不稳定；参考网络更新速度与模型规模对性能的影响尚未充分探索。

---

## 557. Embodied Emissions of 2.5D and 3D Scaling in Chipletized GPUs for Low-Carbon AI

**arXiv ID:** 2610.32830 | [PDF](https://arxiv.org/pdf/2610.32830v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 558. Neural LoFi with Backward Coupling: A Spectral Theory of Cross-Layer Feature Learning

**arXiv ID:** 2610.32849 | [PDF](https://arxiv.org/pdf/2610.32849v1)

**作者:** Yatin Dandi `[一作]` (EPFL), Florent Krzakala `[通讯]` (EPFL)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种在深度网络训练中通过向下反馈学习信号来纠正低层特征的算法 BackLoFi。

**💡 创新点**

创新点在于将前向层次学习与后向耦合结合，利用跨层相关矩阵的奇异向量对低层特征进行选择和修正，并给出变分原理和连续时间动力学分析。

**🔧 技术方法**

主要技术包括谱分解、低阶特征滤波、逆向敏感性传播、核动态自洽方程和随机特征随机提升。

**📊 数据集**

实验使用了合成的深阶阶梯模型（高维高斯输入）和真实的二分类 CIFAR‑10 数据集。

**📈 对比分析**

与传统前向 Neural LoFi、随机特征网络和全批梯度下降进行对比，结果表明 BackLoFi 在相同网络宽度下在小样本情况下明显优于前向 LoFi，并在大样本时与梯度下降性能相近。

**⚠️ 局限性**

局限性包括只在局部近初始化小步长分析范围内得到理论保证，缺乏有限样本的严格泛化误差界定，且对多模式后向耦合的进一步提升仍未充分探索。

---

## 559. Communicating with the Entire Population: Machine Translation of Emergency Alerts and Warnings

**arXiv ID:** 2610.32842 | [PDF](https://arxiv.org/pdf/2610.32842v1)

**作者:** Jen Wilson `[一作]` (University of Washington), William D. Lewis `[通讯]` (University of Washington)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过对美国语言与无障碍警报工作组成员的问卷调查，了解紧急管理者在危机情境下使用机器翻译（MT）与大语言模型（LLM）的现状与挑战；随后在TICO-19疫情相关评测数据集上，对七个商业MT系统进行定量评估，并结合定性错误案例，提出安全、可行的翻译实践与工具选型建议。

**💡 创新点**

创新点在于：①将实地从业者问卷结果与语言资源评测相结合，形成针对危机翻译需求与工具表现的系统映射；②首次在TICO-19中引入短文本（News & Announcements子集）评估，揭示与普通文本不同的性能差异；③提供可操作的“翻译模板”“文本简化”“LLM提示”三大实践指南，为缺乏人力资源的局部机构提供实用策略。

**🔧 技术方法**

使用的技术包括：在线问卷平台（针对LAAWW成员）；机器翻译评测指标 CHRF++；对比实验（7款商业MT系统：Google Translate、Microsoft Translator、DeepL、ChatGPT、Claude、Gemini、GPT-5.4-mini）；定性错误分析；以及文本简化与提示工程方法。

**📊 数据集**

主要数据集是 TICO-19（36种语言的 COVID‑19 文章）以及其 News & Announcements 子集（近短信息）。问卷数据来自 25 名LAAWW成员，覆盖 42 州/领地/部落，涉及 24 种与 TICO-19 语言重叠的目标语言。

**📈 对比分析**

比较方法：对每个系统在整体数据集与 N+A 子集上分别计算 CHRF++，并按语言分为 Pivot、Important、Priority 三个资源级别。结果显示：① Pivot 语言整体得分最高，但在 N+A 子集中往往下降；② 重要/优先级语言在两组得分差异更大，部分语言甚至在子集中表现优于整体；③ 传统 MT 与 LLM 在不同语言、不同文本类型上的表现存在显著差异，说明单一指标不足以评判危机情境下的可靠性。

**⚠️ 局限性**

局限性包括：问卷样本量小且非随机，难以对全美各地区的翻译需求进行统计推断；评测仅基于 TICO-19 公开数据，未检验真实危机信息的可读性与行动导向性；对 LLM 的评估仅限于公开版本，未覆盖模型持续迭代对输出的影响；未对不同语种人类翻译与 MT 输出在实际危机响应中的影响进行纵向实地验证。

---

## 560. Compressed Boolean Matrix Multiplication using Bicliques

**arXiv ID:** 2610.32836 | [PDF](https://arxiv.org/pdf/2610.32836v1)

**作者:** Nicolás Araya `[一作]` (Universidad de Concepción), Gonzalo Navarro `[通讯]` (University of Chile)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出基于二分图压缩的布尔矩阵乘法方法，通过自适应阈值提取稠密二分子图并直接在压缩表示上进行乘法

**💡 创新点**

改进了二分图提取的自适应参数调节（基于度分布的百分位阈值）以及将Schoor乘法扩展到四种组合（矩阵-矩阵、矩阵-二分图、二分图-矩阵、二分图-二分图）

**🔧 技术方法**

使用CSR/CSC、二分图压缩、Schoor乘法、MinHash、集合哈希、分区搜索、堆合并等技术

**📊 数据集**

在网页、社交网络和学术引用网络等真实图数据集上实验，包含web-Stanford、cnr-2000-hc、wikipedia-link-lmo、web-Google、coPapersCiteseer、indochina-2011、arabic-2005

**📈 对比分析**

与CSR/CSC基线以及k²-tree进行对比，压缩比可达10×、空间可压缩至1/10，乘法速度可提升至290×，在大规模图上表现最佳；在小规模图上基线更快

**⚠️ 局限性**

对输出矩阵需要再次提取二分图导致耗时增加；在多次乘法后二分图结构可能退化；在极大图上内存限制导致无法完成更高阶乘法

---

## 561. Quantum Cadence: Post-Quantum Consensus with Multiple Concurrent Proposers

**arXiv ID:** 2610.32803 | [PDF](https://arxiv.org/pdf/2610.32803v1)

**作者:** Mohammad Mussadiq Jalalzai `[一作]` (Category Labs), Jovan Komatovic `[通讯]` (Category Labs)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种无证书、可并发多提议者的后量子共识协议，利用签名仅用于可追溯性，确保安全性、活性和可推测最终性。

**💡 创新点**

创新点在于彻底消除共识证书，改用“投票路径”计数法和签名-free Byzantine 协议（改进的 Forget‑IT），实现了与 HotStuff 相同的最优快速路径延迟并在提议者数量增大时具有更好扩展性。

**🔧 技术方法**

采用后量子签名（如 ML‑DSA、Falcon）、冲突抵抗哈希、已认证通道、Bracha可靠广播、视图变更机制、领导者无视图以及改进的 Forget‑IT 多提议者协议。

**📊 数据集**

通过在 200 名验证者、1 Gbps 链路、真实地理分布（与 2026‑10‑01 Monad 验证者集合相似）环境下的仿真，探测不同提议者数（5、20、100）和签名大小（0–5000 字节）的性能。

**📈 对比分析**

与 HotStuff 进行对比：在无限带宽情况下延迟相同，受限带宽下 PQC 延迟增幅 ≤ 16 ms；当签名尺寸增大或提议者增多时，PQC 的上传量和延迟均优于 HotStuff，展示了更好的可扩展性。

**⚠️ 局限性**

局限性：未给出完整的 ACS 调度与槽关闭细节；仅在无消息丢失、无故障的仿真环境下评估；未提供轻客户端/桥接端的最终性证明；签名-free 协议在极端网络不稳定或恶意节点大规模攻击下的安全性仍待进一步分析。

---

## 562. Every Edge Needs a Quote: Building a Quote-Grounded Knowledge Graph of Healthcare Software Adoption with Language-Model Agents

**arXiv ID:** 2610.32781 | [PDF](https://arxiv.org/pdf/2610.32781v1)

**作者:** Juan Felipe Duran `[一作]` (McGill University), Martin Vallières `[通讯]` (McGill University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并评估了一个端到端的知识图谱生成管道，用于从医疗软件供应商网站及第三方网页中提取客户采用（Adopts）和系统集成（Integrates）关系，并为每条边附上原句引用与来源URL。

**💡 创新点**

创新点包括：① 强制每条关系必须由原句证据支持并可验证；② 采用多层验证（引用修复、referent完整页面检查、确定性形状检查、五模版投票的语义面板）；③ 通过置信度评分（逻辑回归基于投票和元信息）提供可调的精度召回平衡；④ 设计应对长上下文退化和运行方差的限定式模型调用策略。

**🔧 技术方法**

使用技术：LangGraph orchestrator、OpenRouter LLM调用、Firecrawl 站点爬虫、ReAct 代理搜索、Graphiti+Neo4j持久化、五模版投票的语义面板、确定性字符串匹配、逻辑回归置信度评分、Pandas/Scikit‑learn 等。

**📊 数据集**

数据集：61 个医疗软件供应商的网页（含 vendor 自有页面与第三方页面）共 3,284 页；6,428 条候选关系；黄金参考（marketplace 自家手工整理的 749 客户 + 249 集成列表）；经过 GPT‑5.6+Claude Opus 5 验证的页面级别键；并使用了三种不同 LLM 版本进行评测。

**📈 对比分析**

评测方法：与页面级别键对比，计算 precision/recall/F1；与黄金参考对比；引入置信度阈值曲线展示在 0.90 精度下召回 0.90；结果显示在页面级键上 0.94 精度/0.71 召回；对黄金参考仅 0.34/0.41，说明参考不完整；置信度评分在 0.90/0.90 的操作点显著提升召回。

**⚠️ 局限性**

限制：① 仅覆盖提取到的候选，黄金参考缺失导致 recall 低估；② 依赖 LLM 的输出，仍可能出现误判或运行方差；③ 关系的时效性无法追踪，未建模时间维度；④ 实体解析不完全，可能导致同一医院出现多重节点；⑤ 仅在医疗软件领域测试，需进一步验证通用性；⑥ 需要人工审核以确保采购决策安全。

---

## 563. AI Alignment for Digital Democracy: Fairer Voters Representation against LLM Biases

**arXiv ID:** 2610.32780 | [PDF](https://arxiv.org/pdf/2610.32780v1)

**作者:** Yi Qi `[一作]` (University of Leeds), Evangelos Pournaras `[通讯]` (University of Leeds)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

通过让大型语言模型（Claude 与 Llama）根据选民的个人资料预测投票结果，构建并评估 AI 模拟的选民群体，分析其在群体层面的误差与偏差。

**💡 创新点**

首次量化 AI 选民模拟在不同人口群体中的误差分布，揭示误差系统性且与性别、教育、种族等属性相关，并验证常见的对齐干预对群体误差的无效甚至恶化。

**🔧 技术方法**

使用提示工程、拆分提示和激活层级对齐（对 Claude 采用 prompt，Llama 采用 CAA）以及多次抽样与多数投票与概率聚合等技术。

**📊 数据集**

采用美国全国选举研究（ANES）2012、2016、2020 年调查数据，包含完整的种族、性别、年龄、教育、意识形态等特征。

**📈 对比分析**

通过比较不同控制水平下的个体准确率、总体投票误差和最大群体误差，发现即使个体准确率≈85%，群体误差可达 10+ 个百分点；对齐措施几乎不降低准确率，却未能显著减小群体误差，甚至有时加剧偏差。

**⚠️ 局限性**

受限于只评估两党制选举、仅关注单一选举周期的模型与提示设置，且未探索更细粒度的多属性交互与更复杂的对齐机制，结果可能不易推广到多党制或跨国情境。

---

## 564. Predictive Long-Horizon Progress Modeling with Action-Aware Quantification

**arXiv ID:** 2610.32776 | [PDF](https://arxiv.org/pdf/2610.32776v1)

**作者:** Tianchi Xu `[一作]` (RIMBOT), Yuanxin Zhong `[通讯]` (RIMBOT)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了动作感知量化方法和NimCritic模型，用于对机器人长时序操作过程进行稠密进度评估和完成判断。

**💡 创新点**

创新点包括：① 将动作权重与信用回收相结合，生成绝对进度；② 通过VLM前置注释加人工校准构建大型训练集与LOOM基准；③ 采用双流扩散 Transformer 结合 Joint Video–Progress Learning、Historical Progress Perturbation 与 Video–Progress Alignment，实现视频预测与进度学习的联合训练。

**🔧 技术方法**

使用技术包括 VLM 子任务分解、双流扩散 Transformer、视频-进度对齐、历史进度扰动、冻结 VAE 视频编码器、对比学习等。

**📊 数据集**

使用的数据集包括 800+ 小时机器人与人类视频的动作感知训练集、LOOM 300 轨迹 19,618 帧基准、VPB（Video Progress Benchmark）以及 RoboArena。

**📈 对比分析**

与多种基准 Reward Model（如 GVL、RobbyDopamine、Robometer 等）进行比较。NimCritic 在 VPB 上 PRC‑F1 0.77、TSA 0.82；在 LOOM 上 PRC‑F1 0.86、TSA 0.92；在 RoboArena 成功回放检索 recall 96.72%、hit 98.09%，显著优于对比方法。

**⚠️ 局限性**

局限性包括：仍依赖 VLM 注释和人工校准，难以覆盖极端动作或未知对象；需要大量视频与动作标签，训练资源高；实时推理效率未充分评估；对多模态同步处理仍有改进空间。

---

## 565. Yalda-Embedding: Open Development of a Competitive Persian Retrieval Model

**arXiv ID:** 2610.32933 | [PDF](https://arxiv.org/pdf/2610.32933v1)

**作者:** Yalda Taheri `[一作]` (Azad University), Armon Rasooli `[通讯]` (Iran University of Science and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并训练了一个基于 Qwen3-Embedding‑0.6B 的 Persian dense retrieval 模型 Yalda‑Embedding，使用约一百万条问答对进行对比学习与关系式知识蒸馏；

**💡 创新点**

通过两阶段增量 ablation 证明：大批量（512）、梯度裁剪、学习率预热与余弦衰减、以及对 F2LLM‑v2‑8B 的关系式蒸馏，单轮训练即可在八项 FaMTEB 检索任务上突破同规模基线，且与 4B 参考模型仅相差 0.0075；

**🔧 技术方法**

使用 decoder‑only Transformer（Qwen3‑Embedding 结构）、InfoNCE 对比损失、关系式 KD、AdamW + warmup + cosine decay、混合 FP32/BF16 精度、FSDP 分布式训练；

**📊 数据集**

训练数据为 1,035,981 条 Persian query–positive‑document 对，来自 13 个检索资源（MSMARCO‑Fa、HotpotQA‑Fa、NFCorpus‑Fa、PersianWebDocumentRetrieval 等），并发布预计算的 F2LLM‑v2‑8B teacher 表示；

**📈 对比分析**

与同规模 Qwen3‑Embedding‑0.6B、F2LLM‑v2‑0.6B、Dibachain/Diba‑Embed、heydayAI/persian‑embeddings、Tooka‑SBERT‑V2‑Large 以及 4B 参考模型进行对比；在八项 FaMTEB 检索任务的宏平均上取得 0.5488，超越所有 0.6B 基线，接近 4B 模型（宏平均差 0.0075），并在硬负样本任务上表现尤为突出；

**⚠️ 局限性**

局限性包括：仅训练一次 epoch、未使用显式硬负样本挖掘、评估仅覆盖 Persian 任务、对未见任务的泛化未知、以及对 F2LLM‑v2‑8B 预计算成本的依赖。

---

## 566. SemantEq: Faithful Bidirectional Equivalence of Formal Statements

**arXiv ID:** 2610.32919 | [PDF](https://arxiv.org/pdf/2610.32919v1)

**作者:** Xinrui Li `[一作]` (EleutherAI), Stella Biderman `[通讯]` (EleutherAI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个基于允许步骤的语义等价关系SemantEq，用来判断两个 Lean 4 形式化语句是否在数学意义上等价。

**💡 创新点**

创新点在于将等价判定转化为对“允许步骤”链的检验，既限制了证明的形式，又通过审计证明词项而非脚本来保证正确性，从而在不依赖模型的情况下提高精度和召回率。

**🔧 技术方法**

采用了 Lean 4 元编程（Metaprogramming）实现搜索、审计；使用了桥（bridge）和结构弱化等四类步骤；利用 GPT‑5.6 Codex 生成等价证明，再通过自定义审计器检验。

**📊 数据集**

主要数据集为 ProofNetVerif 的 1,452 对参考/候选公式，经过人工重新标注后得到 1,303 对可评估样本；此外还构造了 100 个非等价对用于消融实验。

**📈 对比分析**

与先前的 EvalProof（^）相比，SemantEq 在裁定为等价时实现了 100% 的精度和 65.6% 的召回（裁定为非等价时 0% 的假阳性），并且在搜索加深后几乎不再提升召回；使用 GPT 生成的证明再通过审计后可验证 81.1% 的等价对，且仍保持 0% 的假阳性。

**⚠️ 局限性**

局限包括：仍依赖已可信的参考形式化；桥集合受库本身的覆盖限制，可能漏掉某些等价；审计器只检查已知的允许步骤，对未知证明结构缺乏通用性；当没有参考形式化时无法直接评估。

---

## 567. A Highly Efficient Depth-Averaged Material Point Method for Geophysical Mass Flows

**arXiv ID:** 2610.32907 | [PDF](https://arxiv.org/pdf/2610.32907v1)

**作者:** Johann Kammholz `[一作]` (ETH Zurich), Johan Gaume `[通讯]` (ETH Zurich)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `14d48e9d-0069-4ad9-996a-1d5968216998` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并实现了一种开源的深度平均材料点法（Depth‑averaged MPM）用于快速模拟山体地质大规模滑坡、碎屑流等地质质量流。

**💡 创新点**

创新点包括：① 在全局笛卡尔坐标系下推导深度平均方程，避免曲面坐标系的数值误差；② 将摩擦过程拆分为独立的隐式摩擦步，消除人工停止条件；③ 将摩擦律统一为库仑、Voellmy、μ(I)三种形式，支持速率相关摩擦和粒子级摩擦系数；④ 采用二次B‑spline插值和Poisson disk采样，实现低数值扩散的Lagrangian边界跟踪；⑤ 通过单线程实现的高效算法，使单次模拟在单核CPU上实时完成，且在5 m分辨率下完成1632场模拟仅耗时76秒。

**🔧 技术方法**

核心技术为材料点法（MPM）在深度平均框架下的实现，结合二维流体力学的保守方程、正交坐标系、B‑spline插值、PIC/FLIP混合更新、显式摩擦步和自适应时间步长控制；此外利用DEM与GIS数据生成地形、释放体积和释放高度，并输出GIS栅格和PLY点云。

**📊 数据集**

使用的数据集包括：① 2023年瑞士Brienz/Brinzauls岩石滑坡的无人机DSM差分数据（用于释放体积和地形校正）；② 实验室粒子柱坍塌实验数据（用于验证沉积形态）；③ 多种标准分析问题（滑块滑动、坝破裂）的理论解；④ 对比使用的RAMMS和com1DFA模型所需的公开DEM和实验参数。

**📈 对比分析**

与解析解、实验沉积、以及其他深度平均模型（RAMMS、com1DFA）进行比较。结果表明：① 模型能够精确再现滑块加速与稳态、坝破裂波前与余波；② 对粒子柱实验的沉积高度与形状无网格偏差；③ 在相同硬件与分辨率下，单场模拟速度比RAMMS快10–100倍，单线程实现的高分辨率模拟实时完成；④ 1632场概率实验仅耗时76秒，显示极高的并行吞吐量。

**⚠️ 局限性**

局限性包括：① 仅处理干燥单相流，未建模孔隙水与两相耦合；② 采用等向应力闭合（K=1），未考虑主动/被动应力、各向异性内部摩擦；③ 未包含曲率效应、侵入与颗粒尺寸分布对摩擦的影响；④ 采用均匀速度剖面（β=1），未考虑Bagnold等非均匀剖面导致的动量形状因子误差；⑤ 模型对大斜率的稳定性仍需改进（显式摩擦步导致的稳态误差），以及在极大体积释放时可能出现的数值扩散；⑥ 仅在单核CPU上验证，未探索GPU并行加速与大规模高分辨率模拟的进一步性能提升。

---

## 568. VTRL-Med: Progressive Visual Refinement with Topology-Guided Regularization for Medical Image Clustering

**arXiv ID:** 2610.32884 | [PDF](https://arxiv.org/pdf/2610.32884v1)

**作者:** Guangyu Meng `[一作]` (University of Notre Dame), Danny Z. Chen `[通讯]` (University of Notre Dame)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了一种基于视觉结构学习和拓扑正则化的无监督医疗图像聚类框架 VTRL-Med，能够在无标签数据上实现高精度聚类。

**💡 创新点**

创新点在于：①将 Segment Anything Model (SAM) 迁移至医疗领域，保留其基于掩码的结构先验；②设计两步伪标签筛选（原型选择+k‑NN一致性）提高伪标签可靠性；③在训练中引入持久同调 (PH) 的拓扑分支，作为辅助正则化以解决视觉模糊区的边界错误。

**🔧 技术方法**

核心技术包括：SAM 迁移学习、基于自监督学习的 DINOv3 风格的自蒸馏、两步伪标签筛选策略、持久同调特征提取、两分支（视觉+拓扑）逐步细化训练与互相正则化。

**📊 数据集**

在 13 个医疗图像基准（MedMNIST v2、ISIC2019、Kvasir 等）上进行评估，覆盖多种模态（CT、超声、眼底、皮肤病变等）。

**📈 对比分析**

与 10 种无监督/自监督聚类方法（SimCLR、MoCo‑v3、BYOL、DINOv3、SPICE、CDC、MedTrans、DeepMCAT）以及监督上界 MedMamba-B 对比，VTRL‑Med 在 13 个数据集上平均 ACC 81.51%，NMI 62.78%，ARI 63.34%，比第二名 SPICE 提升约 4.48% ACC，整体接近监督方法仅差 3.29% ACC，且计算成本低。

**⚠️ 局限性**

主要局限包括：①在某些数据集（RetinaMNIST、Kvasir、PneumoniaMNIST）仍落后于专门的自监督或监督方法；②拓扑分支训练依赖两相段的权重调度，若不当会影响收敛；③对 PH 计算的离线前处理仍占一定时间，且对图像尺寸和分辨率敏感。

---

## 569. ViT-H: Generative Video-To-Humanoid Control with Motion Tracking Feedback

**arXiv ID:** 2610.32966 | [PDF](https://arxiv.org/pdf/2610.32966v1)

**作者:** Zimo He `[一作]` (Peking University), Yixin Zhu `[通讯]` (Peking University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a4b10f5d-130b-4e77-9367-6469ec621899` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

该论文提出了一种直接从单目视频生成目标机器人（Unitree G1）动作的框架，使用扩散模型生成机器人参考运动，并通过跟踪反馈对生成器进行后训练（DPO），同时提供自回归版本实现实时视频遥操作。

**💡 创新点**

创新点在于：①打破传统的“重建‑重映射‑跟踪”三段管道，直接在机器人动作空间内生成可执行运动；②利用跟踪器提供的执行反馈进行无标注后训练，显著提升动作可执行性；③自回归扩散生成器实现在线遥操作，支持低延迟实机控制。

**🔧 技术方法**

核心技术包括：视频条件的扩散运动生成器、基于 6D 角度表示的动作编码、完整的根轨迹与关节运动预测、全身跟踪策略（SONIC 微调与特权版本）、基于 DPO 的后训练、以及自回归段落式生成。

**📊 数据集**

预训练使用 AMASS、BONES‑SEED、BEDLAM、BEDLAM2.0、Human3.6M；后训练与评估使用 BEDLAM、BEDLAM2.0、Human3.6M 以及无标注的视频‑仅数据；评估数据集包括 EMDB、RICH、以及 held‑out 视频‑仅集。

**📈 对比分析**

与传统的 GVHMR+GMR、GEM+GMR 等重建‑重映射‑跟踪管道相比，本文方法在动作质量（mpjpe、pampjpe、Jitter、Pen.）上均有显著提升，尤其在执行成功率上从 60‑70% 提升至 85‑90%（离线）和 63‑80%（在线），并且在无标注数据上的提升最为明显。

**⚠️ 局限性**

主要局限包括：缺乏对细粒度手部抓取与操作的建模；扩展性受限于可用训练数据和计算资源；后训练依赖特定跟踪器，尽管已证明一定泛化但仍可能对不同硬件或动态场景适应性不足；未对多模态（如语音、传感器融合）进行深入研究。

---

## 570. STROKE: Simple Sparse Structural Occlusions for Black-Box Adversarial Attacks

**arXiv ID:** 2610.32932 | [PDF](https://arxiv.org/pdf/2610.32932v1)

**作者:** Ankit Gupta `[一作]` (Michigan State University), Emily Dolson `[通讯]` (Michigan State University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了基于简单矩形笔划的黑盒对抗攻击 STROKE

**💡 创新点**

创新点在于使用结构化笔划参数化和贪婪加删改搜索，显著提升查询效率与运行速度

**🔧 技术方法**

使用了 score‑based 的贪婪加笔划与重采样搜索算法，并以旋转矩形遮挡来表示笔划

**📊 数据集**

在 ImageNet‑1k、CIFAR‑10/100 数据集上进行评估

**📈 对比分析**

与 IMPACT、Patch‑RS 等基线对比，STROKE 在 1–2% 面积、5k/10k 查询下，攻击成功率更高、平均查询量更低，且壁钟时间显著缩短

**⚠️ 局限性**

局限性包括仅适用于 score‑based 反馈，针对目标攻击效果不均衡，且假设可直接覆盖图像，未考虑物理可实现性

---

## 571. DexPort: A Hand-Held Skill Interface for Dexterous Hand Data Collection

**arXiv ID:** 2610.32969 | [PDF](https://arxiv.org/pdf/2610.32969v1)

**作者:** Amber Xie `[一作]` (Amazon Frontier Ai & Robotics), Haozhi Qi `[通讯]` (Amazon Frontier Ai & Robotics)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计并实现了一种可携带的数据采集界面，用同一只高自由度机械手和摄像头在实验室与户外环境下收集可执行演示，并通过行为克隆训练视觉-运动策略。

**💡 创新点**

提出利用同一硬件收集命令级动作数据并通过可参数化技能控制高自由度机械手，消除机器人与演示者之间的体积与动作差距，并在无臂或外部跟踪的情况下实现高质量演示收集。

**🔧 技术方法**

使用Meta Quest 2控制器、Wuji 20-DoF机械手、鱼眼相机、参数化技能接口、学习闭环转向技能、Diffusion Policy（带DINOv2视觉编码器）进行行为克隆。

**📊 数据集**

在实验室收集1605条演示（包含两种蔬菜抓取、螺丝刀旋转、抽屉存放、双手条形码扫描），以及1000条室内外随机场景的抓取演示，共计2605条。

**📈 对比分析**

与仅使用测量关节状态作为动作标签的state-as-action基线对比；在四项任务上，使用命令动作训练的策略成功率从11%提升至64%，在抓取任务中与野外数据共训练后成功率从36%提升至78%。

**⚠️ 局限性**

设备重量约0.96 kg，限制连续使用和旋转动作；技能集合覆盖面有限，缺少更多内在操作与抓取学习；高自由度手部硬件仍是主要瓶颈。

---

## 572. MiD-VLM: Detecting and Grounding Global Semantic Misinformation in Image-Caption Pairs

**arXiv ID:** 2610.32967 | [PDF](https://arxiv.org/pdf/2610.32967v1)

**作者:** Tarunya Prasad `[一作]` (Arizona State University), Bharatesh Chakravarthi `[通讯]` (Arizona State University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了MiD-VLM框架，联合检测图像与文本配对的真实性、操纵类型以及误导文本词汇；

**💡 创新点**

创新在于将SigLIP与DINOv2的双编码器特征通过投影MLP融合，再以LoRA仅更新少量参数微调Qwen2-VL-7B，实现全局语义不一致的跨模态检测与定位；

**🔧 技术方法**

使用的技术包括SigLIP（图像/文本嵌入）、DINOv2（视觉CLS token）、投影MLP特征融合、LoRA低秩适配、token‑level grounding头，以及PP‑MisDet挑战评测指标；

**📊 数据集**

使用了扩展版DGM^4+数据集（约5K图文对），包含真实、文本替换与文本属性两类操纵；

**📈 对比分析**

与HAMMER、UFAFormer等基线在DGM^4+验证集对比，MiD‑VLM在二分类F1达到0.968，整体score 0.928，显著优于传统方法（仅20%准确率）；

**⚠️ 局限性**

局限在于token‑level grounding召回率较低，模型难以准确定位导致语义不一致的具体误导词，需要进一步改进细粒度跨模态表示。

---

## 573. Measure Before You Train: Typed Decision Models vs. a Classical Baseline vs. a Generative LLM for Routing a Personal Multilingual Inbox

**arXiv ID:** 2610.32955 | [PDF](https://arxiv.org/pdf/2610.32955v1)

**作者:** Federico Lucca `[一作]` `[通讯]` (Independent researcher), Federico Lucca (Independent researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了个人邮箱代理的决策准确性，提出了基于人类标注的测量协议，并比较了规则、传统分类器、Typed Decision模型与生成式LLM在邮件类型、紧急度、动作等字段上的表现。

**💡 创新点**

创新点在于提供了可复现的、包含分层抽样、冻结快照、线程分组拆分和秘密脱敏的评测协议，并证明在单用户数据上，训练的logistic回归模型在准确率上优于所有零射击模型，同时展示了基于联系人图继承发件人关系的高精度。

**🔧 技术方法**

使用了TF‑IDF + 逻辑回归、MiniLM句子嵌入、Tev1、Nimble 9B 量化版、Bespoke Nimble v4‑max以及Gemma 4 26B/31B等生成式LLM，并在同一问句和约束下进行零射击推理。

**📊 数据集**

数据集为 231 封双语（意大利语/英语）个人邮箱邮件，经过三轮标注，覆盖 207 条线程；此外还构造了 722 封攻击性采样的“对抗”测试集。

**📈 对比分析**

通过线程分组 5‑折交叉验证和零射击全量评估，比较准确率、宏F1、推理延迟与成本；结果显示逻辑回归在类型、紧急度、回复/任务/噪声切换上最高精度，而生成式LLM在类型上最接近；但在误判噪声导致忽略重要邮件时，逻辑回归成本更低。

**⚠️ 局限性**

限制包括单用户单邮箱实验、缺乏独立第二标注者、样本受生产规则预测倾斜、对极少数商业类别评估不足，以及对大型生成式模型在多轮推理或更大规模数据时的鲁棒性未充分验证。

---

## 574. What Mean Error Misses: Device-Clock Diagnostics for Evaluating Eye Tracking across Three Virtual Reality Headsets

**arXiv ID:** 2610.32915 | [PDF](https://arxiv.org/pdf/2610.32915v1)

**作者:** Aleksandr Chernysh `[一作]` (Texas State University), Oleg Komogortsev `[通讯]` (Texas State University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对Meta Quest Pro、HTC VIVE Pro Eye和HTC VIVE Focus Vision三款VR头显的眼动追踪器进行配对实验，使用相同的随机扫视、阅读、眉部抬升和闭眼任务，收集30名参与者在每个头显上的眼动数据。

**💡 创新点**

首次系统评估VIVE Focus Vision的眼动追踪性能，并通过设备时钟诊断揭示重复时间戳、数据缺失组成以及闭眼期间有效标志等问题，提出三项评估检查以提升眼动评估的完整性。

**🔧 技术方法**

采用Unity采集应用统一接口调用，利用各SDK获取眼动样本和设备时钟，计算空间准确率、精度、误差随视场角的变化、线性再校准效果、时间延迟、数据丢失组成等指标。

**📊 数据集**

使用30名大学生在三种头显上完成相同的随机扫视、阅读、眉部抬升和闭眼任务，记录每秒90–120Hz的眼动轨迹和设备时钟信息，构成实验数据集。

**📈 对比分析**

通过配对秩和检验、Friedman检验和霍奇斯-莱曼估计进行统计比较；结果显示Quest Pro最准确（中位数1.04°），VIVE Focus Vision最不准（2.41°），Pro Eye介于两者；第二块任务中Pro Eye误差显著上升；精度方面FV RMS最低但STD/RMS比率极高，闭眼期间Quest Pro未正确标记无效样本。

**⚠️ 局限性**

实验受限于仅使用单个设备单元、固定头位姿限制、样本量相对较小、未测量显示亮度、无法区分闭眼误差来源，以及不同头显硬件版本的潜在差异。

---

## 575. Recommender regret for improving regularised RL estimation with bandits and MCTS

**arXiv ID:** 2610.32910 | [PDF](https://arxiv.org/pdf/2610.32910v1)

**作者:** Danilo Vucetic `[一作]` (Université de Montréal), Gauthier Gidel `[通讯]` (Université de Montréal)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了基于多臂采样的推荐者框架，定义了正则化 RL 的推荐者 regret，并在均匀采样下给出理论最优性；同时扩展到树搜索，提出 recommender MCTS，并在多任务上验证其优越性。

**💡 创新点**

将正则化 RL 目标与采样 Bandit 结合，引入正则化感知的推荐者 regret，证明均匀采样即可实现下界；并将其推广到 MCTS，实现更高效的正则化规划。

**🔧 技术方法**

多臂采样理论、正则化 MDP、Bregman 散度、均匀采样与 E3W 对比、正则化 MCTS (MENTS, RENTS, TENTS, SMM)、理论误差界与经验评估。

**📊 数据集**

合成树、OCTAX 19 种街机游戏、三种生物序列设计任务（AMP、GFP、UTR）使用蛋白/DNA序列数据。

**📈 对比分析**

与传统 MCTS、UCT、E3W、AlphaZero 以及 GFlowNet 基线在相同预算下对比；结果显示推荐者 MCTS（尤其是均匀采样）在多任务上收敛更快，获得更高的期望回报或多样性。

**⚠️ 局限性**

理论上界在动作维数 K 上仍不够紧凑，且对特定正则化缺乏专门算法，未来需设计正则化特定的采样策略。

---

## 576. When Human Corrections Travel: Safety in Federated VLA Adaptation

**arXiv ID:** 2610.32888 | [PDF](https://arxiv.org/pdf/2610.32888v1)

**作者:** Vigneshwaran Subbaraju `[一作]` (A*STAR), Monowar Bhuyan `[通讯]` (Umea University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a4b10f5d-130b-4e77-9367-6469ec621899` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文探讨了视觉-语言-动作（VLA）策略与联邦学习在共享多模态人类纠正（如口语、手势、物理引导等）时的安全与隐私挑战，并提出了分层共享、接收端安全门控以及隐私与安全联合评估等需求与评估议程。

**💡 创新点**

创新点在于将多模态纠正的语义与上下文细粒度考虑进联邦共享框架，提出基于行为范围的共享约束、接收端闭环安全评估、隐私与安全共同考量以及风险意识安全保证四个核心位置。

**🔧 技术方法**

主要技术包括概念框架分析、基于行为范围的共享标签化、动作条件世界模型仿真验证、联邦学习与安全聚合机制、以及隐私保护与攻击评估工具。

**📊 数据集**

文中未使用任何实际数据集，所有示例均为构造性案例与对话式说明。

**📈 对比分析**

提出的比较方法聚焦于任务成功率、物理安全违规率、人工干预频率、隐私泄漏程度等指标，但目前仅给出理论评估框架，未给出实验性能数据。

**⚠️ 局限性**

局限性在于缺乏实证验证，安全门控与隐私评估的具体实现细节尚未完善，且对多模态纠正的语义解释仍面临不确定性与泛化难题。

---

## 577. When Should I Trust This? Designing and Evaluating Factuality Indicators for LLM-based Chatbots

**arXiv ID:** 2610.32883 | [PDF](https://arxiv.org/pdf/2610.32883v1)

**作者:** Greta Warren `[一作]` (University of Copenhagen), Isabelle Augenstein `[通讯]` (University of Copenhagen)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a2602d71-93ab-4bad-974b-672788df8193` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

通过访谈、调查和大规模实验，研究人们如何评估LLM聊天机器人产生信息的真实性，并系统比较不同的真实性指示器（置信度标签、百分比、气泡、来源标签及来源+置信度语言）在不同主题（园艺、数学、医学）下的影响

**💡 创新点**

首次将置信度与来源两种真实性指示器结合、从用户多层视角（认知、情感、行为）考察其对信任与满意度的预期与后验差异，并揭示高风险情境下指示器效用低下且误配置信度导致信任骤降的现象

**🔧 技术方法**

设计并实现多种UI真实性指示器，利用现有LLM生成的回答与人工标注的正确性标签，进行对照实验；采用统计分析（ANOVA、效应量计算）评估用户主观评价

**📊 数据集**

基于美国YouGov面板的三组样本：8名访谈受访者、1000名调查受访者（代表性人口结构）以及5151名实验受访者，所有受访者均为月度以上使用生成式AI的用户

**📈 对比分析**

通过实验的前后测比较，发现置信度标签与来源标签在预测准确性、信息充分度、信任与情感上对预判影响小，效应量在0.002–0.007之间；但响应是否正确的效应显著，且在医学场景下效果更弱

**⚠️ 局限性**

研究仅覆盖美国受访者，样本年龄偏轻，实验采用静态截图而非实时交互，未考察实际决策或使用行为，且只呈现单一示例，导致生态效度和一般化受到限制

---

## 578. SceneBloom: Agentic Indoor Environment Generation with Realistic and Diverse Layouts for Embodied Learning

**arXiv ID:** 2610.32978 | [PDF](https://arxiv.org/pdf/2610.32978v1)

**作者:** Yuhan Liu `[一作]` (New York University), Shenji Wan `[通讯]` (New York University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种名为 SceneBloom 的“代理式”框架，能够生成物理合法、视觉逼真且符合真实室内布局分布的 3D 场景，并利用这些场景训练对象导航策略，提升其在真实扫描环境中的迁移性能。

**💡 创新点**

核心创新在于将视觉布局采样（基于扩散模型）与几何编译（通过提取“关系契约”指导 3D 位置校正）分离开来：先用视觉模型生成与目标资产一致的布局图像，再通过关系契约在保持功能关系的前提下统一修正几何冲突，避免了传统逐物体放置容易破坏关系的局限。

**🔧 技术方法**

关键技术包括：
- 视觉–语言模型（Qwen3‑VL‑8B‑Instruct）用于功能区域规划与契约提取；
- 资产特定 LoRA 的扩散模型（FLUX.1‑dev + FreeFuse）生成区域内的家具排列图；
- 3D 感知模型（OVMono3D‑LIFT）将 2D 检测盒上升到 3D；
- 层级布局拟合与关系约束驱动的几何修正算法；
- 基于 GPT‑4o 评估的视觉质量指标和 LDD 统计检验真实布局匹配度。

**📊 数据集**

使用的主要数据集包括：Matterport3D（M3DLayout）用于示例房间检索与真实室内布局对比；Scan2CAD 作为另一真实室内参考；HM3D（以及其子集 HM3D‑OVON）用于训练与评估对象导航策略；此外在生成阶段使用的资产来源为公开家具库，并通过 LoRA 训练得到资产特定的生成适配器。

**📈 对比分析**

与 Holodeck、SceneWeaver、SAGE 等三类基线相比，SceneBloom 在物理可行性（无网格碰撞、无越界）、视觉质量（GPT‑4o 实时性、功能性、连贯性、完整性评分最高）以及布局分布一致性（对 Matterport3D 与 Scan2CAD 的 LDD 低于所有基线）上均表现最佳。生成速度也更快（2.7 min/布局，约 SAGE 的 27×），并且在使用生成场景训练的对象导航策略上，在真实扫描环境中的成功率提升约 1.7 × 于仅用真实扫描训练的基线。

**⚠️ 局限性**

主要局限：
- 依赖预训练的 LoRA 资产适配器，需为每个资产单独训练，难以扩展到海量新类别；
- 目前仅支持固定类别的室内房间类型（卧室、客厅等），对大型多房间或开放式空间的处理尚未验证；
- 关系契约从视觉图像中自动抽取，若图像分辨率或姿态错误较大，契约质量可能下降，进而影响最终几何修正；
- 生成流程仍以单一回合完成，缺乏多轮交互式优化，可能在极端复杂布局下产生细微冲突。

---

## 579. FlexGlue: Sparse-to-Dense Feature Matching and Visual Disambiguation

**arXiv ID:** 2610.32976 | [PDF](https://arxiv.org/pdf/2610.32976v1)

**作者:** Philipp Lindenberger `[一作]` (ETH Zurich), Marc Pollefeys `[通讯]` (ETH Zurich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 FlexGlue，一种能把稀疏查询点映射到另一图像中任意像素的神经网络，既能在保持子像素精度的同时匹配高效，又能在同一模型中识别不匹配的图像对。

**💡 创新点**

创新点：①在稠密特征网格上聚合全局上下文，仅对稀疏查询点读取结果，实现稠密匹配的鲁棒性与稀疏匹配的速度并存；②联合训练匹配与可视化辨识，利用可视化头在早期层即决定是否继续匹配；③采用窗口化 soft‑argmax 与 1.5‑entmax 训练子像素位置，同时通过无标签的翻转增强产生负样本，提升对视觉对称/伪装的鲁棒性。

**🔧 技术方法**

技术：冻结的 ConvNeXt‑B (DINOv3) 特征提取 + 9 层 Transformer（自注意 + 交叉注意），匹配投影层、窗口化 soft‑argmax、MPL 子像素细化、可视化投票头、深度监督与联合损失。

**📊 数据集**

使用的数据集：MegaDepth、BlendedMVS、ScanNet++、TartanAir（MapAnything）、Hypersim、Map‑free、AerialMegaDepth、DL3DV；额外用于验证的 LaMAR‑DG（室内双胞胎）、HardMatch、Indoor Visual Localization InLoc 等。

**📈 对比分析**

与现有方法比较：在两视图位姿估计、ETH3D、IMC2025、InLoc 视觉定位、HardMatch、两视图评测等任务中，FlexGlue 在匹配精度、可视化辨识、吞吐量和内存占用上均优于 LightGlue、LoMa、RoMav2、MaSt3R 等。对室内双胞胎（LaMAR‑DG）和 ViSym‑F 的 AUROC 也高于 DG++，且速度提升 4‑倍以上。

**⚠️ 局限性**

局限：需要预训练的 DINOv3 backbone，冻结后对极端照明或遮挡的鲁棒性仍有限；对极高分辨率图像的处理仍受 token 数量限制；负样本生成依赖 flip/检索策略，对极端对称场景的误判仍可能出现；训练时相比纯稠密匹配仍有额外开销。

---

## 580. Flownomaly: Single-Step Conditional-Mean Recovery for Few-Shot Anomaly Detection

**arXiv ID:** 2610.32964 | [PDF](https://arxiv.org/pdf/2610.32964v1)

**作者:** Bernardo Perrone Ribeiro `[一作]` (Eindhoven University of Technology), Egor Bondarev `[通讯]` (Eindhoven University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于冻结DINOv2特征空间的正常对偶恢复方法，利用少量无缺陷图像在单步条件流匹配中学习速度场，实现极简的少样本工业缺陷检测与定位。

**💡 创新点**

核心创新在于将FSAD视为正向条件流恢复，利用直线路径的条件均值性质在单一步骤内得到正常对偶，避免轨迹积分并引入仅基于正常数据的Jacobian增益-噪声校准机制，完成无缺陷验证的时序参数选择。

**🔧 技术方法**

使用冻结的DINOv2 ViT-g/14-reg编码器提取多深度patch tokens，构建时间条件残差MLP作为速度场，训练时采用条件流匹配（CFM）对正态噪声到正常特征的路径进行拟合；在推理时通过一次噪声扰动和速度场评估实现单步恢复，残差平方作为异常分数。

**📊 数据集**

在MVTec-AD、VisA以及Real-IAD三个工业缺陷基准上进行实验，覆盖1/2/4 shot以及多视图样本-shot设置。

**📈 对比分析**

与PatchCore、AnomalyDINO、WinCLIP、PromptAD、SubspaceAD、DCP‑SFR、ANoCo等现有FSAD方法对比，Flownomaly在所有shot设置下均达成或接近最高的图像级和像素级AUROC、AUPR与F1最大值，尤其在VisA和Real-IAD上取得显著提升（如1-shot图像级AUROC提升至97.8%）。

**⚠️ 局限性**

局限性包括对全局结构缺陷（如部件计数错误）的敏感性不高，缺乏显式的全局约束建模；同时单步恢复对特征空间分布的依赖导致在极端噪声或不匹配情形下可能产生误判。

---

## 581. Beyond Discrete Tokens: Preserving Numerical Continuity in Medical Vision-Language Models

**arXiv ID:** 2610.32948 | [PDF](https://arxiv.org/pdf/2610.32948v1)

**作者:** Dianye Huang `[一作]` (University of Hong Kong), Zhongliang Jiang `[通讯]` (University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出HAD‑VLM模型，实现基于自然语言提示的超声连续测量，兼顾文本生成与数值预测。

**💡 创新点**

创新点在于将自回归语言模型与扩散模型结合，使用五元组测量表示与坐标‑长度一致性机制，解决离散标记化导致的数值误差。

**🔧 技术方法**

采用CLIP视觉编码器+视觉提示调优、Vicuna‑7B LLM+扩散预测、坐标‑长度一致性与集成后处理。

**📊 数据集**

主要使用EchoNet‑LVH超声视频数据进行四个心脏测量任务，并在X‑ray椎体端板长度测量数据集验证跨模态性能。

**📈 对比分析**

与基于Token的Vicuna、微调Qwen3‑VL及直接MLP回归等基线比较，HAD‑VLM在四个测量任务上MAE下降21–38%，并在问答任务和X‑ray测量上保持或提升性能。

**⚠️ 局限性**

局限在于仅针对线性长度测量，缺乏对复杂几何或面积测量的支持，以及尚未在多中心真实临床数据上验证。

---

## 582. Learning Positive Linear Combinations of ReLUs in Fully Polynomial Time

**arXiv ID:** 2610.32940 | [PDF](https://arxiv.org/pdf/2610.32940v1)

**作者:** Ilias Diakonikolas `[一作]` (University of Wisconsin), Nikos Zarifis `[通讯]` (MIT)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种基于Hermite张量的有效粗方向恢复和覆盖‑修剪算法，用于学习高维高斯ReLU模型。

**💡 创新点**

创新点在于将低阶过滤与隐式张量近似结合，得到多项式时间、样本复杂度的重构方法。

**🔧 技术方法**

主要技术包括Hermite多项式、矩阵本征分解、隐式张量估计与分层抽样。

**📊 数据集**

使用合成的高斯输入分布，实验基于随机生成的ReLU网络。

**📈 对比分析**

与之前的多项式时间方法相比，该算法在样本量和计算量上实现了显著降低，实验验证了在高维下仍能准确重构。

**⚠️ 局限性**

局限在于对高斯分布假设严格、在极大维度下仍需较大样本量，以及在非高斯环境下表现未知。

---

## 583. Runtime BiDi Structural Verification Catches Right-to-Left Layout Defects that Accessibility, Actionability, and Logical-Property Linting Miss

**arXiv ID:** 2610.32934 | [PDF](https://arxiv.org/pdf/2610.32934v1)

**作者:** Karan Yadav `[一作]` `[通讯]` (Google), Karan Yadav (Google)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出一种运行时双向结构验证器，能检测 RTL 布局中物理方向 CSS 引起的缺陷，且现有工具无法捕获；

**💡 创新点**

创新点在于两种互补的检测模式：零金金模式（单向渲染，检测装饰/样式不对称）和差分模式（双向渲染，检测块级 inset/margin 对齐）；

**🔧 技术方法**

技术手段包括基于 headless Chromium 的计算样式与几何检测、起始/终止装饰 heuristic、对称性差分对比以及 RTL‑脚本门控；

**📊 数据集**

数据集为六个主流 UI 设计系统（Bootstrap、Material Components Web、Foundation、Bulma、Primer 等）的真实 CSS，构造的 14 个从历史提交重建的 RTL 缺陷，及 43 个未改动的真实组件；

**📈 对比分析**

与 axe‑core、Playwright、Stylelint core 规则、伪本地化和像素镜像 diff 进行比较；在 14 个真实缺陷上检索率达 13/14，误报率为 0，差分模式在 11/14 中表现为 0 假警报、0 漏报；零金金模式虽召回率低，但补充了差分无法检测的装饰类缺陷；

**⚠️ 局限性**

局限包括仅支持阿拉伯/希伯来 RTL，无法检测仅靠文本重排导致的 inline‑margin 缺陷；差分模式需要两次渲染；对 RTL‑脚本门控导致空文本或无 RTL 内容的组件不被检测；依赖固定字体环境，可能在其他渲染环境中产生差异。

---

## 584. HoloBoxing: A Simple Recipe for Humanoid Boxing with 7 Minutes of Single-GPU Training

**arXiv ID:** 2610.32909 | [PDF](https://arxiv.org/pdf/2610.32909v1)

**作者:** Xihan Ma `[一作]`, Yucheng Wang `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

通过在预训练的全身控制器上训练一个小型反馈规划器，结合目标解码器和测量状态模仿，快速实现机器人拳击行为。

**💡 创新点**

将运动执行与任务决策分离，使用32维潜在动作空间和冻结的解码器/控制器，配合测量状态模仿初始化，使仅需7分钟额外训练即可产生可交互拳击。

**🔧 技术方法**

HoloMotion v2预训练控制器、潜在目标解码器、基于测量状态的监督模仿学习、PPO强化学习、动作空间压缩与观察掩码。

**📊 数据集**

InterHuman运动捕捉数据和RoboStriker轨迹组合共14段训练姿势，用于目标表示拟合与模仿学习。

**📈 对比分析**

在固定机器人、竞技场与裁判下与RoboStriker、GGSD以及消除组件版本对比，7分钟版本在15场比赛中排名第二（Elo 1541），长版（21.4分钟）排名第一，组件移除性能显著下降。

**⚠️ 局限性**

仅在仿真中验证，实验数据集有限，缺乏多种初始种子与硬件一致性，组件互作效应未完全分离，直接目标控制与解码器消除对学习速度的独立影响未明确。

---

## 585. EVSEFuzz : A Tool for Automated Testing of Protocol Implementations

**arXiv ID:** 2610.32902 | [PDF](https://arxiv.org/pdf/2610.32902v1)

**作者:** Soham Gumaste `[一作]` (Ohio State University), Rigel Gjomemo `[通讯]` (University of Illinois Chicago)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `3855fcda-48ef-4070-a15e-803cd5c84d83` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了 EVSEFuzz，一种基于协议规范自动生成模糊测试器的工具，用于评估电动汽车充电网络中的通信协议实现安全性。

**💡 创新点**

创新点在于利用 Python 的 Pydantic 数据校验库动态解析 JSON/XML 模式，自动构造符合或违背规范的消息，并结合状态规范和恶意载荷实现高效覆盖，减少手工编写模糊器的成本。

**🔧 技术方法**

采用了 Pydantic 进行类型推断与验证，Python 运行时元编程生成消息，结合自定义的 datatype、schema 与 sequence fuzzers，并通过黑盒方式发送 WebSocket 或 HTTP 请求。

**📊 数据集**

使用的数据集主要是 OCPP（1.6、2.0 等）、OCPI 2.2.1 及 OpenADR 等协议的公开或内部 JSON/XML 规范文件，并在实际系统（如 EVerest、CitrineOS 及签约公司）上收集响应日志。

**📈 对比分析**

通过在多版本 OCPP 实现和开源项目中运行 10,000 条随机消息（两三次/周），EVSEFuzz 发现多起零日漏洞并导致 CVE-2025-59398/59399，展示了工具在实际环境中的有效性；对比手工模糊测试，时间成本显著降低，覆盖面更广。

**⚠️ 局限性**

局限性包括：仍为黑盒测试，缺乏对内部状态的直接反馈；随机生成的消息对深层协议状态的覆盖有限，需依赖手工编写的状态规范；错误分析依赖人工审阅日志，缺乏自动化判定；对协议变更的即时适配需要重新提供更新后的模式文件。

---

## 586. Green Code or Costly Code? Evaluating the Energy Costs of Iterative LLM-Based Code Optimizations

**arXiv ID:** 2610.32891 | [PDF](https://arxiv.org/pdf/2610.32891v1)

**作者:** Sadra Emami `[一作]` (University of Augsburg), Bernhard Bauer `[通讯]` (University of Augsburg)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了使用大型语言模型（LLM）进行代码自我优化的能耗成本，系统测量了推理能耗与执行能耗，并与人类高效实现做对比；

**💡 创新点**

首次将推理能耗与执行能耗联合评估，量化不同模型与解码配置对整体能耗的影响，并给出“分摊点”指标，指出性能优化与绿色编码往往不对齐；

**🔧 技术方法**

采用EffiLearner自我优化管道、vLLM推理服务、Metrion硬件级能耗测量工具，结合CPU、GPU、内存RAPL计数；

**📊 数据集**

使用EffiBench 1,000道 LeetCode 效率挑战题目，包含官方测试用例和人工修复后的基准实现；

**📈 对比分析**

对比11款开源LLM（Dense与Mixture‑of‑Experts）在四种解码配置下的推理能耗、生成代码执行能耗、正确率及净能效；结果显示：推理能耗因模型规模差异近一顺序，MoE模型显著降低能耗；大多数配置生成的代码在执行时比人类实现更耗能，只有两种配置在百万级执行量后才实现盈亏平衡；

**⚠️ 局限性**

实验局限：仅考虑Python单次运行任务，未覆盖长时间/并发工作负载；基准人类实现为顶级竞赛提交，未衡量普通开发者水平；缺乏执行能耗的显式选择标准；仅评估开源模型，未涵盖商用API；未考虑模型训练与硬件生命周期等间接碳排放。

---

## 587. Closing the Modality Gap via Simplex-Constrained Representations

**arXiv ID:** 2610.32890 | [PDF](https://arxiv.org/pdf/2610.32890v1)

**作者:** Shubham Gupta `[一作]` (Mila -- Quebec AI Institute), Cem Subakan `[通讯]` (Mila -- Quebec AI Institute)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究如何通过将多模态模型的输出投影到概率单纯形来显著降低模态间的分离（模态间距）并保持或提升检索性能。

**💡 创新点**

创新点在于：① 仅通过改变输出几何（从球面到单纯形）实现模态间距大幅缩减；② 引入相对间距（RelGap）度量，使得不同几何空间下的间距可公平比较；③ 在多分辨率和跨模态设置中验证几何影响。

**🔧 技术方法**

技术手段包括：冻结大型预训练编码器（CLAP、FLAVA）；在其上加轻量级线性/MLP适配器；使用softmax投影将向量映射到单纯形；采用InfoNCE对比损失；使用Total Variation距离对单纯形嵌入进行相似度计算；比较欧氏空间的余弦相似度与单纯形空间的TV相似度；对比共享注意力机制（XAttn vs ReTreever）。

**📊 数据集**

数据集涵盖音频-文本检索（Clotho、AudioCaps、SoundDescs）和图像-文本检索（MS‑COCO、Flickr30k）。

**📈 对比分析**

与冻结模型、欧氏空间线性/MLP、共享注意力模型等基线对比，单纯形投影将中心点L2间距从0.36–0.81降至0.005–0.020（减少97–99%），RelGap亦大幅下降；检索指标NDCG@10保持不变甚至略有提升。

**⚠️ 局限性**

局限性包括：① 只在少量预训练模型上验证，未探讨不同网络架构的泛化；② 仍需研究单纯形投影对下游任务（如生成、分类）的影响；③ 对单纯形距离的选择（TV）与其他概率相似度（JS、KL）未做系统比较；④ 计算上对softmax投影的数值稳定性与可扩展性需进一步评估。

---

## 588. Reinforcement Learning with Multi-Step Transition Predictions: A Bellman Theory

**arXiv ID:** 2610.32887 | [PDF](https://arxiv.org/pdf/2610.32887v1)

**作者:** Chenbei Lu `[一作]` (Cornell University), Adam Wierman `[通讯]` (Caltech)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究在块反馈（block‑feedback）协议下，利用多步转移预测提升马尔可夫决策过程（MDP）的决策质量，提出低维贝尔曼方程与价值分析，并给出了样本复杂度上界；

**💡 创新点**

创新点包括：① 证明块反馈可通过贝叶斯贝尔曼算子将价值函数压缩回原始状态空间；② 通过贝尔曼‑贾森（Bellman–Jensen）效应定量化预测信息的价值；③ 提出 BF‑BOLA 算法，使用两源数据（未覆盖行的环境样本与预测样本）实现最优近似策略，并给出匹配的下界；

**🔧 技术方法**

主要技术手段有：动态规划、贝叶斯贝尔曼递推、价值迭代、概率集中不等式、样本复杂度分析与最优下界构造；

**📊 数据集**

实验使用人工构造的四状态生产 MDP 与真实的风能‑价格（CAISO）时间序列数据，后者构建了可调容量的风电储能 MDP；

**📈 对比分析**

与固定周期（FH）和滚动周期（RH）协议对比，BF 在多步预测下实现 12–16% 的成本/收益提升，实验表明其样本效率与理论上限相符；

**⚠️ 局限性**

局限性包括：仅针对块反馈协议；对滚动预测未给出低维贝尔曼表示；需满足预测一致性假设；计算复杂度随预测长度线性增长，难以直接扩展到连续或大规模状态空间。

---

## 589. LoCA: Enabling Multi-Modality in Geospatial Location Encoders

**arXiv ID:** 2610.32894 | [PDF](https://arxiv.org/pdf/2610.32894v1)

**作者:** Panagiotis Koromilas `[一作]` (University of Athens), Mihalis A. Nicolaou `[通讯]` (University of Cyprus)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文设计并验证了一种两阶段的多模态位置编码器 LoCA，用坐标函数将地理位置映射到高维特征，并直接用于多种地球观测下游任务。

**💡 创新点**

创新点在于：①证明位置编码器与对比学习存在固有维度限制；②提出多模态编码必须满足跨模态协同对齐与高有效秩两项要求；③通过 Stage1 联合训练 Sentinel‑1/2 编码器实现对齐，Stage2 用投影适配器和多视角对比学习学习位置与元数据编码器，满足两项要求并取得最优性能。

**🔧 技术方法**

使用技术包括 Lipschitz 维度理论分析、InfoNCE/DHEL/MV‑DHEL 对比学习、Spherical harmonic + Siren 坐标网络、投影适配器、线性/MLP 下游探针、LightGBM LambdaRank 等。

**📊 数据集**

使用数据集：扩展的 Sentinel‑1/2 S2‑100K 训练集（含 14 条元数据）以及 17 个多样化下游评测任务（土地覆盖、野火危险、长期气候排名、地理社会经济、物种分布等）。

**📈 对比分析**

与 GeoCLIP、SatCLIP、GT‑Loc、AlphaEarth、Copernicus‑FM 等基准在 10/3 试验种子下的线性/MLP 探针与排名评估中对比，LoCA 在 17 任务中获得 8 任务第一名，整体赢率 68.8%，在多任务平均上优于竞争模型。

**⚠️ 局限性**

局限性包括：缺乏时间维度，仅在 Sentinel‑1/2 进行 Stage1 训练，难以覆盖更多传感器；评估受空间自相关影响，缺少块/留区交叉验证；跨模态协同对齐与高秩需求在更大模态数下需进一步验证。

---

## 590. Separating Thinking from Spelling for Efficient Generative Recommendation

**arXiv ID:** 2610.32980 | [PDF](https://arxiv.org/pdf/2610.32980v1)

**作者:** Danil Gusak `[一作]`, Evgeny Frolov `[通讯]`

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种生成式推荐方法，将用户历史编码为紧凑概念，并让轻量解码器完成 beam 搜索，从而分离用户建模与候选 ID 拼写。

**💡 创新点**

创新点在于将完整 beam 搜索迁移到轻量解码器，仅保留一个固定尺寸概念作为接口，显著降低了候选生成的计算开销。

**🔧 技术方法**

使用了 Qwen2 风格的解码器、残差量化得到的 SID、KV 缓存、知识蒸馏和复制指针等技术。

**📊 数据集**

在七个公开数据集（Amazon Toys/Beauty/Sports/BPC-23/Books-23、MovieLens、Yambda）上进行实验。

**📈 对比分析**

与全骨干教师模型以及其他 SID 生成器（TIGER、LLaDA-Rec、RPG、NEZHA 等）对比，概念解码模型在准确率上相当或更优，同时在 50 ms p95 延迟下吞吐量提升 2.8–5.1×，单个模型可生成千级候选。

**⚠️ 局限性**

局限性在于对极端重复消费场景的概念表征可能不足，且需要在推理前预先计算概念，模型在非常长历史或极大 SID 深度时仍有一定瓶颈。

---

## 591. Robotic Optical Coherence Tomography-Guided Geometric Modeling of Soft Tissue: 3D Representations for Data-Driven Closed-Loop Shape Control

**arXiv ID:** 2610.32961 | [PDF](https://arxiv.org/pdf/2610.32961v1)

**作者:** Chaoyu Zhang `[一作]` (University of Michigan), Mark Draelos `[通讯]` (University of Michigan)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e15e3743-5ee0-4d5f-813d-d146868082fc` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种基于光学相干断层扫描（OCT）的闭环软组织形状控制框架，通过从用户演示学习，利用OCT反馈逐步调节机械工具姿态以逼近目标组织形态。

**💡 创新点**

创新点在于：①将三维OCT体素、稀疏体素和体素化点云等多种组织表示方式统一用于目标与当前状态的编码；②引入残差（当前-目标）条件来显式表征剩余变形；③同时评估确定性（Explicit）与生成式扩散（Diffusion）两种动作预测模型，系统探究表示、残差与动作模型的交互影响。

**🔧 技术方法**

技术包括：光学相干断层扫描、三维卷积/稀疏卷积/PointNet特征编码器、残差条件模块、基于Transformer的Explicit动作序列回归模型、条件扩散模型以及闭环实时执行循环。

**📊 数据集**

数据集为151条实验演示轨迹（40条深度、51条方向向量），共计1,714帧OCT观测；实验还包含60条猪皮外组织的控制试验（5深度×5方向×3次部署）。

**📈 对比分析**

对12种配置（3种表示×有/无残差×Explicit/ Diffusion）在模拟试验（深度和方向）和实际猪皮实验中进行对比。结果显示：①体素化点云+Explicit+残差获得最佳精度（深度RMSE≈0.53mm，方向RMSE≈0.44mm），并能在未见目标上保持类似性能；②残差显著提升全体素和稀疏体素的误差（例如全体素Diffusion从0.748→0.555mm）；③在猪皮实验中，策略误差与人工演示相近（≈1.1mm）。

**⚠️ 局限性**

局限性包括：①模型推理时延较高（全体素模型182–403ms，稀疏体素高达615ms），与OCT采集与机器人执行时延叠加限制了闭环更新率；②实验仅涉及短时局部形状调节，未覆盖长时序或多任务情境；③目前对组织的形变建模依赖于演示数据，缺乏对更复杂解剖结构或动态变化的适应性。

---

## 592. If at First You Don't Succeed, Backtrack: Lightweight Recovery for Frozen VLAs

**arXiv ID:** 2610.32941 | [PDF](https://arxiv.org/pdf/2610.32941v1)

**作者:** Thomas Landeg `[一作]` (Stanford University), Thierry Tambe `[通讯]` (Stanford University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了 Backtrack，一种轻量级的失败检测与恢复机制，能在冻结的视觉语言动作（VLA）模型上预测即将失败，并通过回溯机械臂运动来实现失败恢复；

**💡 创新点**

其创新点在于仅使用少量的失败预测头（不到1%参数量），无需收集恢复演示、重训练或额外推理模型，且通过简单的回溯操作即能提升成功率；

**🔧 技术方法**

技术实现包括多头注意力池化（MAP）读取 VLA 内部特征、三层前馈网络输出失败概率、阈值触发的控制器、记录实现位移并按逆序重放、可调阈值与回退长度等；

**📊 数据集**

使用的数据集为 LIBERO（四套任务：Spatial、Object、Goal、Long）进行主实验，MetaWorld 作为二次验证；

**📈 对比分析**

在五个随机种子、每任务 50 试验的基准下，Backtrack 在 LIBERO‑Long 上将失败率降低 19–48%（成功率提升最高 7.4pp），在其他三套任务提升幅度较小；在 MetaWorld 上，所有 SmolVLA 控制器配置均实现 +2.44–5.20pp 的成功率提升；

**⚠️ 局限性**

局限性包括：仅在仿真环境验证，未在真实机器人上测试；回溯仅恢复机械臂姿态，无法恢复物体位置，且对需要全新策略的失败无效。

---

## 593. On the Expressive Power of Hook-Indexed State Comprehending State in the React UI Library

**arXiv ID:** 2610.32925 | [PDF](https://arxiv.org/pdf/2610.32925v1)

**作者:** Toby Ueno `[一作]` (University of Edinburgh), Sam Lindley `[通讯]` (University of Edinburgh)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50`

**🎯 论文内容**

提出了一个核心演算（λhook）并为其定义了类型系统，旨在安全地捕捉React中的 hook‑索引状态，并给出从 λhook 到传统 ML 参考型状态的可证明翻译。

**💡 创新点**

创新点在于：①将 hook‑索引状态抽象为简洁的语法与语义；②引入轻量级效果系统和伪线性指针，消除 React 常见的错误；③证明 hook 状态可由第一类引用完全表达，搭建了从 hooks 到传统状态模型的正式桥梁。

**🔧 技术方法**

使用了小步语义、效果系统、线性/伪线性指针、标签化标识、类型推导与证明以及构造性翻译等技术。

**📊 数据集**

无使用数据集，所有工作均在形式化与证明层面完成。

**📈 对比分析**

通过语义对应与类型保持证明实现了翻译的可证性；未进行实测性能比较，理论上与引用实现等价。

**⚠️ 局限性**

局限性包括：①翻译导致代码复制与抽象泄露；②不支持宏局部性；③对 useEffect、custom hook 等扩展需要额外设计；④未给出实际实现或性能基准。

---

## 594. A Tutorial on Physics Constrained Machine Learning for Hybrid Modeling, Optimization, and Control

**arXiv ID:** 2610.32914 | [PDF](https://arxiv.org/pdf/2610.32914v1)

**作者:** Rahul Golder `[一作]` (Texas A&M University), MM Faruque Hasan `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了可模块化的 KKT‑HardNet 物理约束机器学习框架，并在混合建模、实时优化、模型预测控制及基于数据的灰盒优化等多种工程应用中进行演示。

**💡 创新点**

核心创新在于：①使用基于 KKT 条件的投影层严格满足非线性等式与不等式约束；②利用隐式函数定理实现投影层的高效反向传播；③提供多种投影策略（Fischer‑Burmeister 减少形式、原始‑对偶内部点法、显式线性等式投影）及自动前向/后向编译；④开源可扩展的 Python 包，支持自定义网络骨干、投影算法和训练流程。

**🔧 技术方法**

技术手段包括：神经网络（MLP、深层 NN）+投影层；KKT 投影求解（Newton、内部点、Fischer‑Burmeister 重构）；隐式微分（Implicit Function Theorem）用于梯度计算；基于 JAX 的高性能计算与 AOT 编译；批量化训练与多任务损失（数据拟合、目标优化、一致性约束）。

**📊 数据集**

使用的 benchmark 数据集包括：1）提取蒸馏（R‑410A）与池化问题（共计 9 个输出、7 个约束）；2）两相反应器-冷凝器系统（4000 条样本）；3）CSTR 的 RTO 与 MPC 训练集（2500 条和 5000 条不同参数组合）；4）合成气生产（Aspen Plus 4000 条样本）。

**📈 对比分析**

与传统 MLP、PINN、ENFORCE、DC3、IPOPT 等方法进行对比。KKT‑HardNet 在 MSE、约束残差、训练时间和推理时间上均显示出显著优势：约束违约率降至机器精度、预测误差下降 1–2 个数量级，训练时间仅为软约束方法的 10‑20%（例如 RTO 1.65 ms 推理时间 vs. IPOPT 1.83 ms），同时保持了可接受的推理延迟。

**⚠️ 局限性**

主要局限：①投影层的求解复杂度仍随约束规模增长；②对约束表达式的先验知识要求高，若缺失或错误会导致投影失效；③在极大规模（百万变量）工业系统中，投影求解与隐式梯度仍需进一步优化；④需要对投影参数（容差、迭代次数、平滑因子）进行细致调参；⑤在某些非凸或多模态问题中，投影可能陷入局部最优，需多起点或自适应投影策略。

---

## 595. Knowledge Distillation across Item and Code Spaces for Parallel Generative Recommendation

**arXiv ID:** 2610.32943 | [PDF](https://arxiv.org/pdf/2610.32943v1)

**作者:** Yixuan Xu `[一作]` (Hong Kong University of Science and Technology), Can Yang `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

对并行语义ID（SID）生成推荐器引入知识蒸馏，提出两种跨空间匹配目标，提升候选覆盖与排序质量。

**💡 创新点**

首次将商品级分布匹配（IDM）与码级边缘匹配（CMM）与固定SID映射结合，既保留了并行预测结构，又利用教师的软分布信息。

**🔧 技术方法**

使用并行SID预测框架AsymRec、预训练SASRec教师、soft cross‑entropy/KL蒸馏、梯度校准与温度调节等技术。

**📊 数据集**

在Amazon 2014的Beauty、Toys、Sports三类5-core子集上进行实验。

**📈 对比分析**

与多种基线（Cas, GRU, BERT, SASRec等序列推荐器以及TIGER、RPG等生成式推荐器）进行对比，IDM平均提升Recall@10 15.39%，CMM提升NDCG@10 8.28%，在所有数据集均显著优于基线。

**⚠️ 局限性**

局限在于仅考虑固定SID映射和单一教师；对稀疏数据效果有限；未探究可学习tokenizer或多模态联合蒸馏。

---

## 596. ZENO: Nonlinear filtering with Zakai Equation Neural Operators

**arXiv ID:** 2610.32926 | [PDF](https://arxiv.org/pdf/2610.32926v1)

**作者:** Aaditya Prasad `[一作]` (Massachusetts Institute of Technology), Brian DePasquale `[通讯]` (Boston University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `14d48e9d-0069-4ad9-996a-1d5968216998` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建并训练了ZENO，利用物理信息神经算子求解Zakai方程，实现对非线性滤波分布的全局近似，并在此基础上通过EM框架完成状态推断与参数学习。

**💡 创新点**

将物理信息神经算子应用于Zakai方程，实现了无需标签即可学习滤波分布的算子，并在EM框架中同时估计动力学与观测参数，突破了传统粒子滤波和PINN仅适用于单实例的限制。

**🔧 技术方法**

使用物理信息神经算子(PINO)、DeepONet深度算子网络、GRU上下文编码、MALA采样、EM优化；损失包括Fokker‑Planck残差、观测跳跃损失与初始条件损失。

**📊 数据集**

在三种仿真系统（双井势、Van Der Pol振荡器、Lorenz系统）以及真实的C. elegans全脑钙信号数据集上进行评估。

**📈 对比分析**

与SING、SDE‑matching、gpSLDS等基线比较，ZENO在状态推断上与最优数值滤波接近，参数学习与SING/SDE‑matching相当；在C. elegans数据中实现了与基线相当的重构R²、预测RMSE，并提取出行为相关的限周期。

**⚠️ 局限性**

局限包括未实现后向滤波（平滑）分布，受限于单一Stiefel观测矩阵导致重构R²略低；对多模态或多尺度动力学的捕捉仍有限，且在高维连续系统上训练成本较高。

---

## 597. The Language of Instruction: Genre and Generative AI

**arXiv ID:** 2610.32982 | [PDF](https://arxiv.org/pdf/2610.32982v1)

**作者:** James E. Dobson `[一作]` `[通讯]` (Dartmouth), James E. Dobson (Dartmouth)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文论证了经过指令微调的语言模型应被理解为实现一种体裁，称为指令对话体裁。这种体裁是生成模型与人类之间先前对话的语言的融合，通过指令微调、系统提示的包含以及使用特殊标记和角色分配的标记语言形式化输入。

**💡 创新点**

创新点在于将指令微调的语言模型视为一种特定的对话体裁，强调了这种体裁在现代生成AI聊天机器人中的重要性，并探讨了用户如何在这种体裁中被定位和参与。

**🔧 技术方法**

使用了指令微调技术，结合了生成模型的自回归特性和强化学习方法，特别是人类反馈的强化学习（RLHF）。

**📊 数据集**

使用了Anthropic的HH-RLHF数据集，该数据集包含用户与聊天机器人之间的配对提示和响应，旨在强化有用和无害的响应。

**📈 对比分析**

与传统的语言模型相比，经过指令微调的模型在响应用户指令时表现出更高的准确性和适应性。通过对比不同模型的输出，发现指令微调显著提高了生成文本的相关性和上下文适应性。

**⚠️ 局限性**

限制在于指令微调的过程可能导致模型输出的偏见和不一致性，且用户的输入可能受到预设的对话角色和结构的限制，难以完全突破体裁的约束。

---

## 598. RobustRLlib: A Unified Library and Benchmark for Robust Reinforcement Learning Algorithms

**arXiv ID:** 2610.32971 | [PDF](https://arxiv.org/pdf/2610.32971v1)

**作者:** Xi Chen `[一作]`, Laixi Shi `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 RobustRLlib 库与基准，整合 17 种鲁棒强化学习算法，并在 273 种扰动配置下统一评估。

**💡 创新点**

提出了算法中心统一接口、可扩展扰动工具箱（包括 MDP 元素、延迟与语义偏移）以及能力边界图谱，弥补了现有算法重用与性能界限不清的问题。

**🔧 技术方法**

采用了 17 种鲁棒 RL 方法（包括在线、离线与安全鲁棒算法）、系统化扰动生成框架、Isaac Lab 与 VLA 任务适配器以及大规模实验平台。

**📊 数据集**

使用了多种模拟控制任务（行走、操纵、导航、车辆）以及高级机器人任务（类人机器人、复杂操纵器、视觉‑语言‑动作任务）作为实验环境，没有使用真实世界数据集。

**📈 对比分析**

在 20 算法（6 基线 + 14 鲁棒方法）与 220 组扰动配置下对比；鲁棒方法整体优于基线，在线方法相对基线提升更显著，动态偏移导致性能下降最大，鲁棒安全方法在安全约束下表现最佳。

**⚠️ 局限性**

局限包括离线鲁棒性提升有限、对多重偏移的适应不足、离线方法对数据覆盖的依赖、在动态/复合偏移下仍出现显著性能损失，以及缺乏真实世界部署验证。

---

## 599. RynnUnify: Towards Unified Embodied Brain

**arXiv ID:** 2610.32965 | [PDF](https://arxiv.org/pdf/2610.32965v1)

**作者:** Minghao Zhu `[一作]` (DAMO Academy, Alibaba Group), Xin Li `[通讯]` (Hupan Lab)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `afceb026-1760-41ae-8d86-010831a37d97` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出RynnUnify模型，实现了从感知、视频预测到机器人动作的统一体化框架；

**💡 创新点**

创新点包括：①在Cosmos3基础上引入交叉块场景记忆与自强训练，实现长时序视频规划；②通过强化学习的“工具化”策略学习何时调用视频生成来增强推理；③将完整模型扩展为World‑Action Model（WAM），实现基于隐层规划的机器人控制；④提出专门评测长时序规划（Rynn‑VP）与生成辅助理解（Rynn‑GEU）的基准；

**🔧 技术方法**

采用Mixture‑of‑Transformers架构，包含8B自回归推理器与8B扩散生成器；使用自强训练（Self‑Forcing）、场景记忆、交叉块注意、流匹配损失、Branch‑Conditional Policy Optimization（BCPO）等技术；

**📊 数据集**

训练数据包括RynnBrain‑Planning、HiFi‑UMI、OpenAoE、HIW‑500、AgiBot‑Beta、RoboMIND、RoboCOIN、RoboTwin、RoboCasa365等多来源任务、视频与问答；生成数据涵盖I2V、FL2V、跨块延续；WAM训练使用RoboCasa365与Astribot S1真实机器人轨迹；

**📈 对比分析**

与其他统一多模态模型（BAGEL、RxBrain、Cosmos 3 Nano等）以及公开基准（VSI‑Bench、MMSI‑Bench、MindCube、ERQA、RoboSpatial等）进行对比。RynnUnify在Rynn‑VP总体得分从45.05升至54.80，Rynn‑GEU平均得分从44.0升至50.0；在RoboCasa365的Composite‑Unseen和Selected‑20任务中取得51.5%成功率，超过同类模型；在真实机器人Astribot S1上三项任务的平均成功率达46.7%，高于基线；

**⚠️ 局限性**

局限性包括：生成视频在细粒度几何上易产生误差，导致某些空间任务得分下降；视频生成的计算成本较高，限制闭环控制的实时性；目前评测多在模拟或无环境执行，需进一步验证在更广泛真实环境中的可执行性。

---

## 600. Acyclic 2-edge-coloured graphs: polynomial-time linkages and colouring hardness

**arXiv ID:** 2610.32954 | [PDF](https://arxiv.org/pdf/2610.32954v1)

**作者:** Jørgen Bang-Jensen `[一作]` (Shandong University), Amadeus Reinald `[通讯]` (University of Warsaw)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了二色无交错循环图（acyclic 2‑edge‑coloured graph）中的两类问题：一是针对给定终点对的交错 k‑linkage 问题，给出了当 k 为固定常数时的多项式时间算法；二是判定完整二色图的交错染色数是否为 2，证明其为 NP‑完整，并进一步指出等价于将图划分为两部分各诱导阈值图。

**💡 创新点**

创新点在于：① 将经典有向图中 DAG 上的 k‑linkage 多项式算法扩展到更一般的二色图，利用无交错图的特殊 acyclic 排序进行动态规划；② 通过巧妙的 gadget 构造和归约，首次证明完整二色图的交错染色数 2 的判定问题是 NP‑完整，等价于在图中划分为两阈值子图，填补了阈值图划分问题的空白。

**🔧 技术方法**

主要技术：
- 对无交错二色图利用 Grossman–Häggkvist 定理构造 acyclic 排序。
- 对 k‑linkage 采用基于该排序的动态规划与递归分块，处理路径中可能出现的顶点 v_i 的三种情况。
- 对染色数 2 的 NP‑难性证明使用了完整二色图的特殊 gadget（8 结点构造）并从 4‑uniform 超图 2‑着色问题归约。
- 通过构造阈值图与二色图的对应关系，将染色数问题转化为阈值图划分。

**📊 数据集**

该工作纯粹为理论算法研究，没有使用实验数据集；所有结论均来自构造性证明与多项式归约。

**📈 对比分析**

性能评估：
- 对于固定 k，k‑linkage 算法时间为 O(n^{O(k)})（具体实现中为 O(n^4) 或更低），显著优于一般图中的 NP‑完全情况。
- 对染色数 2 的判定问题被证明为 NP‑完整，意味着在一般情况下没有多项式时间算法（除非 P=NP）。

**⚠️ 局限性**

局限性：
- k‑linkage 算法仅在 k 为常数时多项式，k 作为输入参数时仍为 NP‑难。
- 仅证明了完整二色图的染色数 2 的 NP‑难性，未给出近似或特殊图结构（如稀疏图）下的多项式解。
- 对于非完整二色图的 acyclic alternating chromatic number 2 的判定仍未完全解决，尤其是稠密图或特定图类别的复杂性。

---

## 601. Where Rounding Happens Matters: Rounding-Location Alignment Stabilizes LLM Reinforcement Learning

**arXiv ID:** 2610.32951 | [PDF](https://arxiv.org/pdf/2610.32951v1)

**作者:** Yijing Ke `[一作]` (ByteDance Seed), Ruosong Wang `[通讯]` (Peking University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在BF16下通过对RoPE、SwiGLU和残差RMSNorm三个关键位置进行FP32中间值保留并对齐，显著降低了训练与推理间的数值差异，从而稳定了大语言模型的强化学习训练。

**💡 创新点**

提出仅对少数关键运算节点进行精度对齐，而非全栈对齐或额外的RL目标修正，即可防止BF16训练崩溃，展示了局部数值一致性即可实现稳定训练的可能。

**🔧 技术方法**

采用数值对齐（FP32保留）、RoPE/SwiGLU/RMSNorm路径分析、Kullback–Leibler误差评估、PPO-CLIPPED GRPO目标以及对比零不匹配后端等技术；实验使用vLLM推理与Hugging Face Transformers训练框架。

**📊 数据集**

使用Qwen系列大模型（DeepSeek-R1-Distill-Qwen-1.5B、Qwen3-1.7B、Qwen3-4B-Base、Qwen3-8B-Base）在Math子集（Sanity-Test-R1D-1.5B）与AIME 2025数学竞赛数据集上进行强化学习实验，并在Dapo-Math-17k上做进一步验证。

**📈 对比分析**

与BF16基线、FP32 RoPE路径、FP32三区间和零不匹配后端四种设置进行对比；实验显示BF16基线在训练后期崩溃，而仅对RoPE路径或三区间进行FP32保留均能保持训练稳定并保持较高AIME 2025准确率，性能下降不到3%。

**⚠️ 局限性**

实验仅覆盖数学任务与Qwen家族模型，缺乏对其他任务和架构的泛化验证；对齐策略需要对每个模型的计算图手动识别对应位置，难以自动化扩展。

---

## 602. Adaptive Frequency Threshold Byte Pair Encoding for Mixed Natural Language and Structured Symbolic Corpora

**arXiv ID:** 2610.32921 | [PDF](https://arxiv.org/pdf/2610.32921v1)

**作者:** Pietro Di Nicco `[一作]` (University of Calabria), Andrea Tagarelli `[通讯]` (University of Calabria)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种基于自适应频繁模式挖掘的分词方法（称为），通过动态阈值调节来学习跨空格的多词表达式和正式语言结构；

**💡 创新点**

创新点在于：①将分词阈值与模式长度和类型（一般语言vs.正式语言）耦合，使得对频繁短词与稀疏长词的处理更为合理；②通过频繁模式挖掘递归构造更长的“超词”，显著提升多词表达式与数学符号的压缩效果；

**🔧 技术方法**

技术核心为：BPE的改进版，结合频繁模式挖掘、动态阈值计算、正则表达式判定LaTeX/数学符号，随后逐步扩展词表；

**📊 数据集**

使用的评测数据集包括：latex‑formulas、Image2Struct（两者为数学/程序代码），以及通用语言的DROP、SQuAD、CSQA；

**📈 对比分析**

与BPE和SuperBPE比较：在latex与Image2Struct上AEL降低约7%，超词占比提升至≈2倍；在通用语料上保持相近甚至略优；在小型OLMo2模型的下游任务中，数学基准（MATH500、GSM8K）BPB下降，通用基准准确率无显著损失；

**⚠️ 局限性**

局限性：超参数（θ_general^min、θ_latex^min）对词表规模和压缩率影响显著，过度放宽阈值会导致词表过大；模型规模受限时下游表现受限；目前仅针对LaTeX，扩展到其他正式语言需进一步验证。

---

## 603. Polyhedral RIS Architecture with Quantum-Assisted Facet Selection for Uniform Coverage

**arXiv ID:** 2610.32898 | [PDF](https://arxiv.org/pdf/2610.32898v1)

**作者:** Noha Hassan `[一作]` (Carleton University), Halim Yanikomeroglu `[通讯]` (Carleton University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种以 chamfered 六角棱柱为几何的多面体可重构智能表面（RIS），并在多用户下行链路中将面片激活、相位、间距、预编码等问题联合优化，目标为最大化最小用户速率。

**💡 创新点**

创新点在于：①给出面片方向增益的闭式最坏情况保证，并扩展至随机安装误差；②将面片激活子问题精确建模为二元无约束二次优化（QUBO），可在每个信道保持时间内快速重解；③通过混合量子-经典方法（SA、QAOA）解决 QUBO，验证其优于贪婪和随机激活。

**🔧 技术方法**

主要技术包括：量子近似优化算法（QAOA）、模拟退火（SA）、Dinkelbach 最大最小化迭代、正则化零迫（RZF）预编码、相位量化模型、面片耦合矩阵近似、仿真中的自由空间 Rician 信道模型。

**📊 数据集**

使用基于 Monte Carlo 的仿真数据集：150 个随机 UE 布局场景，6 个 UE，64 天线基站，28 GHz 频段，Rician 5 dB，随机面片安装误差（σδ ≈ 1°）以及在 IBM Q 量子硬件上执行一次 QUBO 求解。

**📈 对比分析**

将多面体 RIS 与单面平面、双面 STAR 平面 RIS 进行对比；指标包括最小速率、死区比例、Jain 公平度、总速率。实验表明多面体 RIS 在死区比例从 77% 降到 0.3% 的同时，最小速率比 STAR 提升 34%，SA 在 95% 的实例中达到全局最优，QAOA 目前尚未实现最优解。

**⚠️ 局限性**

局限性包括：① QAOA 在实际噪声量子硬件上尚未能获得最优激活方案；② 随着棱柱阶数或面片数增大，QUBO 的变量规模快速膨胀，求解时间上升；③ 只考虑了面片安装误差的统计分布，未覆盖所有实际制造和部署误差；④ 需要精确的信道估计和面片激活的实时重解，可能对实时系统产生实现难度。

---

## 604. NgramRL: Accelerating RL Post-Training for Structured Generation via Rollout-Aware N-gram Speculative Decoding

**arXiv ID:** 2610.32881 | [PDF](https://arxiv.org/pdf/2610.32881v1)

**作者:** Nilesh Prasad Pandey `[一作]` (University of California San Diego), Tajana Rosing `[通讯]` (University of California San Diego)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种名为 NgramRL 的无模型、无额外训练的 RL 训练后加速方法，通过每个 prompt 的 n‑gram 缓存与 GPU 目标模型进行 speculatively decoding，并通过自适应的 fan‑out prefetcher 把 CPU 读取与 GPU 前向推理解耦。

**💡 创新点**

创新点在于：① 利用最近一次 rollout 本身产生的 n‑gram 作为 draft，天然跟随策略演进；② 采用 per‑prompt 有界历史窗口的 n‑gram LUT，避免全局缓存的稀疏与过时；③ 设计了基于接受长度与“bonus token”预测的自适应 fan‑out prefetcher，隐藏 CPU 访问导致的 GPU 空闲；④ 通过精确的 draft‑verify 机制保证采样分布不变，兼顾速度与精度。

**🔧 技术方法**

技术手段包括：speculative decoding + draft‑verify；per‑prompt n‑gram LUT（order 3–6）；bounded 轮次窗口；基于 causal 统计的预算分配的 adaptive fan‑out prefetcher；与 vLLM/GRPO 训练框架无缝集成；实验使用 GPU 与 CPU 协同调度。

**📊 数据集**

使用了三类数据集：GSM8K（自由形式算术推理），Spider（文本到 SQL 结构化生成），CodeContests（代码生成）。

**📈 对比分析**

对比基线 Prompt‑Lookup、REST、SuffixDecoding，使用 tokens‑per‑decoding‑step (TPS) 作为指标。NgramRL 在 Qwen3‑8B 上 GSM8K 3.60 TPS、Spider 7.77 TPS、CodeContests 3.60 TPS；在 DeepSeek‑7B 上对应 TPS 分别为 6.81、7.71、6.19。相较基线提升 1.2–1.99×（对 SuffixDecoding）及 7×（对 Prompt‑Lookup）。预取器进一步提升 1.38× 的 rollout 速度。

**⚠️ 局限性**

局限性包括：① 对 prompt 重复性依赖，若训练 prompt 多样化或策略快速漂移，缓存命中率下降；② 需要 CPU 侧 n‑gram 存储与预取计算，可能在大模型/大批量下受 CPU 限制；③ 对高熵自由生成任务（如 CodeContests）收益有限；④ 需要手动调节窗口大小、n‑gram 阈值和预取预算，调参成本存在。

---

## 605. UniAction: Unified Action Conditioning for World Models

**arXiv ID:** 2610.32977 | [PDF](https://arxiv.org/pdf/2610.32977v1)

**作者:** Bowen Xue `[一作]` (Stanford University), Jiajun Wu `[通讯]` (Stanford University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 UniAction 框架，将异构动作统一表示为稀疏动作元素（语义标识 + 时空坐标），并直接映射到预训练视频模型的本地位置几何中，实现单模型条件世界模型。

**💡 创新点**

创新点在于：1）通过动作语义与时空定位统一描述动作，摆脱原始形式差异；2）将动作坐标嵌入视频模型的 RoPE 位置空间，直接利用预训练时空建模；3）引入 cardinality‑calibrated attention，消除不同动作粒度导致的注意力偏差。

**🔧 技术方法**

技术手段包括：使用预训练的 Wan2.2 latent 视频 Transformer，3D RoPE 位置编码，动作语义词表与连续时空坐标，LoRA 轻量级微调，以及卡尔曼卡里得注意力校正。

**📊 数据集**

使用四类代表性动作数据集：GoalForce（力作用）、SolarWM（相机控制）、EgoDex（手部姿势）、DROID（机器人姿势），每个10k clip，总计40k clip。

**📈 对比分析**

通过与 VACE 视觉条件和 Cross‑Attention 令牌条件基线的对比实验，UniAction 在所有四种动作的 Action Following 分数最高、LPIPS 最低；单一联合训练模型在 Action Following 上均优于对应的动作专家模型。

**⚠️ 局限性**

局限性包括：① 受限于有限的动作标注数据，泛化至更稀疏或更复杂动作仍需验证；② 对极端时空偏移或多动作组合的处理尚不充分；③ 依赖预训练视频模型的 RoPE 结构，若预训练模型不匹配可能影响效果。

---

