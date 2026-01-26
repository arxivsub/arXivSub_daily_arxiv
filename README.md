# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-01-26 | 今日论文总数: 313

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. Analyzing Neural Network Information Flow Using Differential Geometry

**arXiv ID:** 2601.16366 | [PDF](https://arxiv.org/pdf/2601.16366v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 2. M3Kang: Evaluating Multilingual Multimodal Mathematical Reasoning in Vision-Language Models

**arXiv ID:** 2601.16218 | [PDF](https://arxiv.org/pdf/2601.16218v1)

**作者:** Aleix Torres-Camps `[一作]` (Qualcomm AI Research), Jordi Ros-Giralt `[通讯]` (Qualcomm AI Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个多语种、多模态的数学推理基准M3Kang，并对顶级视觉语言模型进行评估。

**💡 创新点**

首次在数学推理任务上提供108种语言和图像的基准，并设计自动翻译与质量评估流水线。

**🔧 技术方法**

使用自动化翻译管线、链式思考（CoT）、多语言翻译、CLSP投票、向量引导等多语言多模态技术。

**📊 数据集**

M3Kang（来自Kangaroo数学竞赛的1747道题，108语种，总计111k个实例），以及M2Kang英文子集；对比了68,000名学生的成绩。

**📈 对比分析**

将多种开源和闭源VLM在多语种和图像题目上进行对比，并与人类基准对照，结果表明模型在基本数学和图像推理上仍表现差劲，性能随语言出现频率和模型规模提升，但不随难度提升。

**⚠️ 局限性**

模型仍难以处理基本数学和图像推理；低资源语言仍有显著性能差距；难题难度对模型影响不大，且翻译质量可能影响评估。

---

## 3. Towards Latent Diffusion Suitable For Text

**arXiv ID:** 2601.16220 | [PDF](https://arxiv.org/pdf/2601.16220v1)

**作者:** Nesta Midavaine `[一作]` (University of Amsterdam), Grigory Bartosh `[通讯]` (University of Amsterdam)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种Neural Flow Diffusion Models（NFDM）框架，用于在离散词汇空间中进行语言生成。

**💡 创新点**

创新点在于学习多变量、数据条件化的前向扩散过程，替代固定噪声计划，实现连续扩散模型在离散状态空间的直接应用，并通过对每个 token 位置自适应噪声分配来降低 ELBO。

**🔧 技术方法**

采用NFDM、MuLAN 的多维线性 Gaussian 前向过程、DDIM 采样、BERT‑base 编码器与 Adaptive LayerNorm、重建损失与 ELBO 损失等技术。

**📊 数据集**

实验数据集为 ROCstories（约90k条常识生活故事）。

**📈 对比分析**

与 Diffusion‑LM、MuLAN、MuLAN‑Rescaled 以及自回归 GPT‑J 进行对比，NFDM 在 bits‑per‑character（ELBO）上达到 3.12，接近 GPT‑J 的 3.05；在样本质量上（PPL、MAUVE、多样性、记忆化）与 MuLAN‑Rescaled 接近，略低于 Diffusion‑LM 的 PPL 与 MAUVE。

**⚠️ 局限性**

局限性包括仅在无条件生成任务上验证，数据规模有限，且在条件生成和更大规模数据上的效果尚未验证。

---

## 4. Multi-User Content Diversity in Wireless Networks

**arXiv ID:** 2601.16323 | [PDF](https://arxiv.org/pdf/2601.16323v1)

**作者:** Belal Korany `[一作]` (Qualcomm Technologies Inc), Hemanth Sampath `[通讯]` (Qualcomm Technologies Inc)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

研究并提出了多用户内容多样性框架，利用内容复杂度信息实现基于内容感知的速率控制与资源分配，从而提升 XR、云游戏等沉浸式应用的用户体验。

**💡 创新点**

将用户内容复杂度纳入速率控制决策，首次在 6G 前景下设计网络中心与网络辅助的最优与轻量级内容感知算法，并通过内容感知调度与 OTT 率控制提升用户体验。

**🔧 技术方法**

采用 α‑公平资源分配、网络效用最大化（NUM）、梯度升降法、PF 调度改进、RTT‑PSNR 基准、SCONE、AQP、PDU‑set 等信令机制实现内容感知资源调度与速率控制。

**📊 数据集**

使用基于 3GPP InH/UMa 信道模型生成的 SINR 轨迹，模拟 60fps 云游戏视频场景切换，并在仿真平台评估算法性能。

**📈 对比分析**

与传统 RTT‑基准、Prague L4S、普通 PF 等方法对比，最优 MaxCap 在 5–6 UE 时提升 UX 容量 67%，MaxMin 提升最小 PSNR，轻量级 UX‑PF 亦显著提升满意率与公平度，同时网络利用率保持不升甚至下降。

**⚠️ 局限性**

依赖精确的 QB 曲线估计与应用反馈，算法对标识与同步要求高；网络辅助方案需 ECN/ECN 标记；轻量级方案缺乏全局协同，导致低 UE 时资源浪费；未考虑多类型流与实时估计误差。

---

## 5. Limits of n-gram Style Control for LLMs via Logit-Space Injection

**arXiv ID:** 2601.16224 | [PDF](https://arxiv.org/pdf/2601.16224v1)

**作者:** Sami-ul Ahmed `[一作]` `[通讯]` (University of Colorado Boulder), Sami-ul Ahmed (University of Colorado Boulder)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了一种在解码时利用n-gram风格先验注入到冻结LLM的logit空间，实现轻量化的风格控制。

**💡 创新点**

提出了无需训练额外模型即可通过n-gram概率调节logit的方式，实现可调节的风格偏置。

**🔧 技术方法**

使用1-3阶n-gram模型、logit注入公式、λ调节参数以及TinyLlama-1.1B进行实验。

**📊 数据集**

评估数据集包括《唐吉诃德》英文译本、CNN/DailyMail 头条新闻、arXiv 论文摘要。

**📈 对比分析**

与仅提示式控制和LoRA微调进行对比，结果显示在Don Quixote仅在λ=0.1时略有提升，提示式控制效果更好，LoRA效果最佳。

**⚠️ 局限性**

方法脆弱，仅在极小的λ区间可行；在多作者或高熵语料上表现更差，且高λ易导致模型崩溃或生成无意义文本。

---

## 6. Combining Tests and Proofs for Better Software Verification

**arXiv ID:** 2601.16239 | [PDF](https://arxiv.org/pdf/2601.16239v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 7. Coarse-to-Fine Non-rigid Multi-modal Image Registration for Historical Panel Paintings based on Crack Structures

**arXiv ID:** 2601.16348 | [PDF](https://arxiv.org/pdf/2601.16348v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 8. Regional Bias in Large Language Models

**arXiv ID:** 2601.16349 | [PDF](https://arxiv.org/pdf/2601.16349v1)

**作者:** M P V S Gopinadh `[一作]` (Vishnu Institute of Technology), Yesaswini Swarna `[通讯]` (Vishnu Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过构建FAZE评估框架，在100个中性地区对比提示上评估10种主流LLM的地域偏差。

**💡 创新点**

创新点在于提出针对地域偏差的行为化评估工具FAZE，并系统比较不同模型的偏差幅度。

**🔧 技术方法**

使用基于提示的评估方法，将问题转化为强制选择场景并进行二元分类统计。

**📊 数据集**

数据集为100个手工设计的中性地区对比提示，生成共1000个模型回答。

**📈 对比分析**

通过FAZE得分对模型进行对比，GPT‑3.5最高9.5分，Claude 3.5 Sonnet最低2.5分，展示显著的地域偏差差异。

**⚠️ 局限性**

局限在于仅采用单轮评估、提示数量有限、缺乏多语言覆盖以及对细粒度偏差缺乏定量解释。

---

## 9. DMAVA: Distributed Multi-Autonomous Vehicle Architecture Using Autoware

**arXiv ID:** 2601.16336 | [PDF](https://arxiv.org/pdf/2601.16336v1)

**作者:** Zubair Islam `[一作]` (Ontario Tech University), Mohamed El-Darieby `[通讯]` (Ontario Tech University)

**通讯引用:** 274 | [OpenAlex ID](https://openalex.org/A5047581908)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了 DMAVA 分布式多 AV 架构，实现了多台主机上独立 Autoware 堆栈的实时同步仿真。

**💡 创新点**

创新点在于基于 ROS 2 的命名空间隔离、Zenoh 低延迟数据中心通信层以及对 AWSIM Labs 的多车支持改造。

**🔧 技术方法**

使用 ROS 2 Humble、Autoware Universe、AWSIM Labs、Zenoh、Unity、OpenStreetMap、Blender 等技术栈。

**📊 数据集**

主要使用 OSM 制作地图与点云，利用 AWSIM Labs 生成 LiDAR、摄像头、IMU 等传感器数据；实验数据来自仿真场景。

**📈 对比分析**

通过两台和三台主机实验验证同步、定位稳定性与通信延迟；平均 RTT 低于 6.5 ms（两机），三机下仍保持可接受范围，未出现持续丢包，系统整体稳定。

**⚠️ 局限性**

局限在网络稳定性和时延波动导致三机长时间运行出现间歇性失稳；未能支持高带宽图像主题，ROS 2 Daemon 轻微崩溃，扩展至更大规模需改进容错和硬件一致性。

---

## 10. A Longitudinal, Multinational, and Multilingual Corpus of News Coverage of the Russo-Ukrainian War

**arXiv ID:** 2601.16309 | [PDF](https://arxiv.org/pdf/2601.16309v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 11. Algorithmic Identity Based on Metaparameters: A Path to Reliability, Auditability, and Traceability

**arXiv ID:** 2601.16234 | [PDF](https://arxiv.org/pdf/2601.16234v1)

**作者:** Juliao Braga `[一作]` (Federal University of ABC), Itana Stiubiener `[通讯]` (Federal University of ABC)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

提出将数字对象标识符（DOI）用于算法识别，提升算法治理、透明度和可追溯性。

**💡 创新点**

创新点在于构建三层算法识别税onomies并将DOI与元数据、加密认证协议结合，实现算法的机构化身份和可审计性。

**🔧 技术方法**

采用DOI注册、元数据模式、加密签名挑战-响应认证协议、版本管理与语义版本控制。

**📊 数据集**

未使用具体数据集，重点在理论框架和元数据设计。

**📈 对比分析**

未进行实验比较，主要通过与现有识别机制（Git哈希、数字签名、SWHID、专利）的结构比较说明优劣。

**⚠️ 局限性**

限制包括中心化注册成本、对独立开发者的门槛、DOI仅保证身份不保证质量，以及需要完善元数据验证等。

---

## 12. The Behavioral Fabric of LLM-Powered GUI Agents: Human Values and Interaction Outcomes

**arXiv ID:** 2601.16356 | [PDF](https://arxiv.org/pdf/2601.16356v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 13. FC-GUARD: Enabling Anonymous yet Compliant Fiat-to-Cryptocurrency Exchanges

**arXiv ID:** 2601.16298 | [PDF](https://arxiv.org/pdf/2601.16298v1)

**作者:** Shaoyu Li `[一作]` (Virginia Tech), Wenjing Lou `[通讯]` (Virginia Tech)

**通讯引用:** 32025 | [OpenAlex ID](https://openalex.org/A5001879281)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出了一种基于可验证凭证（VC）和零知识证明（ZKP）的匿名但合规的法币-加密货币兑换系统，能够在不泄露用户个人身份信息和银行账户的前提下完成交易，并支持税务审计时的合法去匿名化。

**💡 创新点**

创新点在于：①将VC与ZKP结合实现匿名身份验证与银行账户校验，完全拆除传统平台与用户身份的直接关联；②引入审计加密机制，既满足KYC与税务合规，又仅在违规时才允许审计机构解密；③兼容现有交换工作流，降低部署门槛。

**🔧 技术方法**

技术手段包括：可验证凭证（CL/​BBS+ 签名）、零知识证明（非交互式 NIZK）、非对称加密（ElGamal/Paillier）以及TLS 加密通信；实现上使用 Python+cryptography 库、W3C VC 标准、Bitcoin 测试网络。

**📊 数据集**

使用的数据集为：1）加密货币交易使用 Bitcoin 测试网络区块；2）用户身份信息仅为合成 PII 与银行账户数据；3）性能评测在 AWS EC2（Ubuntu 22.04）和 Google Pixel 8a（Android 14）上完成。

**📈 对比分析**

与传统未加密的基线平台比较，本文系统在注册阶段增加约 156 ms，交易阶段 VP/ZKP 生成和验证在 PC 上约 3 ms，移动端约 2.5 s；比基线仅 5 ms 与 0.5 ms；总体来看，隐私保护带来的开销在可接受范围内（约 2–3 倍时间，但在移动设备上仍保持在 3–5 秒以内）。

**⚠️ 局限性**

限制：①仅针对美国的银行转账模型，其他司法区需改动；②实现依赖 VC/​ZKP 基础设施，若缺失可扩展性受限；③审计加密仅覆盖 SSN，其他财务数据仍需按需求扩展；④性能在高并发交易场景下尚未验证。

---

## 14. My Parents Expectations Were Overwhelming: Online Dating Romance Scams Targeting Minors in Iran Through Exploitation of Parental Pressure

**arXiv ID:** 2601.16321 | [PDF](https://arxiv.org/pdf/2601.16321v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 15. SoundBreak: A Systematic Study of Audio-Only Adversarial Attacks on Trimodal Models

**arXiv ID:** 2601.16231 | [PDF](https://arxiv.org/pdf/2601.16231v1)

**作者:** Aafiya Hussain `[一作]` (Virginia Tech), Chris Thomas `[通讯]` (Virginia Tech)

**通讯引用:** 4417 | [OpenAlex ID](https://openalex.org/A5006675265)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `6215c339-3735-4be3-8a07-5bbb7004712d` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了在三模态音频-视频-语言模型上进行无目标、仅音频的对抗攻击，并系统评估了六种攻击目标的效果。

**💡 创新点**

创新点在于提出针对音频编码、跨模态注意力、隐藏层以及输出层的多维攻击目标，并展示了仅通过音频扰动即可显著破坏三模态推理的事实；同时对攻击成功率、转移性、感知失真及优化动态进行了全面的实验分析。

**🔧 技术方法**

使用梯度投影法（PGD）优化音频扰动，构造了六类损失函数（负语言建模、编码器余弦、视觉注意抑制、音频注意放大、注意随机化、隐藏状态余弦），并结合多模态模型的内部表示进行白盒攻击。

**📊 数据集**

实验数据集包括 AVQA、AVSD、Music‑AVQA（多模态问答与摘要）以及 LibriSpeech（语音识别）用于评估不同任务的攻击效果。

**📈 对比分析**

与基准模型 VideoLLAMA2、Qwen 2.5 Omni、Qwen 3 Omni 以及 Whisper 进行对比；结果显示编码器空间攻击可达 96% 的攻击成功率，其他目标低于 60%；攻击对不同模型和域的转移性差；在保持低 LPIPS/高 SI‑SNR 的情况下亦可实现高攻击成功率。

**⚠️ 局限性**

主要局限：仅在白盒场景下验证；扰动仅在数字域下施加，未考虑物理传输；实验仅覆盖三种模型和四个数据集，未评估防御措施。

---

## 16. Efficient Gaussian process learning via subspace projections

**arXiv ID:** 2601.16332 | [PDF](https://arxiv.org/pdf/2601.16332v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 17. Where is the multimodal goal post? On the Ability of Foundation Models to Recognize Contextually Important Moments

**arXiv ID:** 2601.16333 | [PDF](https://arxiv.org/pdf/2601.16333v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 18. DSGym: A Holistic Framework for Evaluating and Training Data Science Agents

**arXiv ID:** 2601.16344 | [PDF](https://arxiv.org/pdf/2601.16344v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 19. Machine-Assisted Grading of Nationwide School-Leaving Essay Exams with LLMs and Statistical NLP

**arXiv ID:** 2601.16314 | [PDF](https://arxiv.org/pdf/2601.16314v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 20. The CMU-AIST submission for the ICME 2025 Audio Encoder Challenge

**arXiv ID:** 2601.16273 | [PDF](https://arxiv.org/pdf/2601.16273v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876`

---

## 21. GameTalk: Training LLMs for Strategic Conversation

**arXiv ID:** 2601.16276 | [PDF](https://arxiv.org/pdf/2601.16276v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 22. SE Research is a Complex Ecosystem: Isolated Fixes Keep Failing -- and Systems Thinking Shows Why

**arXiv ID:** 2601.16363 | [PDF](https://arxiv.org/pdf/2601.16363v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 23. Teaching and Evaluating LLMs to Reason About Polymer Design Related Tasks

**arXiv ID:** 2601.16312 | [PDF](https://arxiv.org/pdf/2601.16312v1)

**作者:** Dikshya Mohanty `[一作]` (Stony Brook University), Niranjan Balasubramanian `[通讯]` (Stony Brook University)

**通讯引用:** 4479 | [OpenAlex ID](https://openalex.org/A5101768349)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并发布了一个面向高分子设计与合成的多任务基准数据集（约12.5万条任务），并通过知识增强的推理蒸馏生成结构化思路链（CoT），从而为大型语言模型（LLM）提供了高质量的训练和评估资源。

**💡 创新点**

创新点在于①把高分子化学专业知识、结构表示（如SMILES）和属性预测结合成六大任务类型；②通过在每个任务中注入完整的高分子资料并生成结构化CoT，显著提升模型的多步推理与组合能力；③使用知识增强的蒸馏策略，将教师模型在真实数据上的推理过程转化为可验证的步骤，降低幻觉风险。

**🔧 技术方法**

采用大规模知识蒸馏（knowledge‑augmented distillation）生成CoT，使用QLoRA微调技术对7B-14B模型进行任务特定训练，评估时采用多种指标（准确率、MAE、Kendall‑tau、结构合法性等）。

**📊 数据集**

数据集基于13个公开化学数据库（PolymerDB、OMG Polymers、Chembl等）和实验数据，统一构建高分子结构、属性与合成信息的知识库，并生成覆盖结构理解、概念知识、属性预测、比较/排序、高级推理和合成设计六大任务。

**📈 对比分析**

与现有开源及对齐化学LLM（如Mistral、InternLM、LLaMA）以及封闭源前沿模型进行对比。微调后的7B-14B模型在自建基准测试集上整体超越所有基线，并在外部高分子基准（Block Co‑Polymer、ChemD、Llm‑ML等）上保持竞争力，尤其在多目标设计和合成生成任务中显著提升。

**⚠️ 局限性**

局限性包括：①CoT生成仍可能包含教师模型的幻觉，尽管做了自动与人工校验；②数据主要为文本，缺乏图像或其它多模态信息；③未涵盖实验操作或工具调用的 agentic 任务；④任务范围受限于现有公开数据库，缺乏更深层的实验设计与优化问题。

---

## 24. Bringing order to network centrality measures

**arXiv ID:** 2601.16236 | [PDF](https://arxiv.org/pdf/2601.16236v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39`

---

## 25. AMBER: A Columnar Architecture for High-Performance Agent-Based Modeling in Python

**arXiv ID:** 2601.16292 | [PDF](https://arxiv.org/pdf/2601.16292v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99`

---

## 26. When Agents Fail to Act: A Diagnostic Framework for Tool Invocation Reliability in Multi-Agent LLM Systems

**arXiv ID:** 2601.16280 | [PDF](https://arxiv.org/pdf/2601.16280v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 27. SemanticALLI: Caching Reasoning, Not Just Responses, in Agentic Systems

**arXiv ID:** 2601.16286 | [PDF](https://arxiv.org/pdf/2601.16286v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 28. ChiEngMixBench: Evaluating Large Language Models on Spontaneous and Natural Chinese-English Code-Mixed Generation

**arXiv ID:** 2601.16217 | [PDF](https://arxiv.org/pdf/2601.16217v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 29. Topological transitivity of group cellular automata is decidable

**arXiv ID:** 2601.16243 | [PDF](https://arxiv.org/pdf/2601.16243v1)

**作者:** Niccolò Castronuovo `[一作]` (A. Einstein), Luciano Margara `[通讯]` (University of Bologna)

**通讯引用:** 1601 | [OpenAlex ID](https://openalex.org/A5061123241)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6`

**🎯 论文内容**

证明了有限群上的群胞自动机（GCA）的拓扑传递性可判定，并进一步推出与之等价的混合性、混沌性等性质亦可判定。

**💡 创新点**

提出了通过可完全不变子群分解把任意 GCA 转化为直接积同构简单群的 GCA，从而统一处理多维、非阿贝尔群的可判定性问题，填补了此前仅限于一维阿贝尔群的空白。

**🔧 技术方法**

使用了群论结构分类、可完全不变子群分解技术、莱文多项式矩阵表示与特征多项式判定、以及对周期性与循环结构的有效计算。

**📊 数据集**

无实际数据集；论文全部基于理论证明与符号计算。

**📈 对比分析**

未做实验对比；结论为理论可判定性证明，未给出具体算法复杂度或性能评估。

**⚠️ 局限性**

仅适用于全移位空间的 GCA；未考虑子移位、非阿贝尔群子移位、或者更广义的指数群/单子等情况，算法的可扩展性尚待研究。

---

## 30. FeTTL: Federated Template and Task Learning for Multi-Institutional Medical Imaging

**arXiv ID:** 2601.16302 | [PDF](https://arxiv.org/pdf/2601.16302v1)

**作者:** Abhijeet Parida `[一作]` (Sheikh Zayed Institute for Pediatric Surgical Innovation), Marius George Linguraru `[通讯]` (Sheikh Zayed Institute for Pediatric Surgical Innovation)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出一种联邦学习框架 FeTTL，联合学习全局图像模板和任务模型，用于多机构医学影像的分割和分类任务。

**💡 创新点**

创新点在于：① 在联邦环境下使用白化-彩色化（DataAlchemy）风格变换实现无原始像素共享的图像调和；② 将模板学习与下游任务学习放入同一优化循环，兼顾分布对齐和任务性能；③ 引入全局共享模板实现跨站点特征聚合。

**🔧 技术方法**

核心技术：联邦学习（FedAvg）、白化-彩色化风格变换、VGG‑19/ResNet‑34/U‑Net 编码器‑解码器、AdaIN、梯度聚合与同步更新。

**📊 数据集**

使用的数据集包括：① 视网膜视盘分割——Drishti‑GS1、RIGA‑BinRushed4、RIGA‑Magrabia、RIGA‑Messidor、REFUGE（共5站）；② 病理转移分类——CAMELYON16（两站 X、Y）。

**📈 对比分析**

对比方法：FedAvg、FedBN、FedProx、FedDG、FedHarmony 以及中心化训练；FeTTL 在视盘分割上平均 Dice 0.937（比最佳基线高 0.036，p≤0.002），在转移分类上 AUPR 0.681（高于 FedDG 0.660，p≤0.002）。

**⚠️ 局限性**

局限性：目前仅在 2D 图像上验证，3D MRI 适配尚缺乏合适特征提取器；模板学习的理论极限和对极端非 IID 情形的鲁棒性尚待进一步研究；初始化策略对少样本站点仍有一定影响。

---

## 31. Student Mental Health Screening via Fitbit Data Collected During the COVID-19 Pandemic

**arXiv ID:** 2601.16324 | [PDF](https://arxiv.org/pdf/2601.16324v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 32. GR3EN: Generative Relighting for 3D Environments

**arXiv ID:** 2601.16272 | [PDF](https://arxiv.org/pdf/2601.16272v1)

**作者:** Xiaoyan Xing `[一作]` (Google DeepMind), Dor Verbin `[通讯]` (Google DeepMind)

**通讯引用:** 2701 | [OpenAlex ID](https://openalex.org/A5066233418)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

对室内场景的三维重建进行光照重建，提供可控的三维场景重光照方法。

**💡 创新点**

通过把视频到视频扩散模型的输出蒸馏回三维重建，绕过逆渲染难题，实现对房间尺度场景的细粒度光源控制。

**🔧 技术方法**

使用视频扩散模型（Wan 2.2）进行光照条件微调，结合相机路径渲染、光源条件视频、旋转位置编码，并通过训练的噪声调度与3D蒸馏（Zip‑NeRF）实现。

**📊 数据集**

使用合成的 Infinigen 300+室内场景，30种灯光条件，共108k个视频；并在 Eyeful Tower 真实场景上进行评估。

**📈 对比分析**

与 LightLab、DiffusionRenderer 等基线相比，在用户研究中赢率达 79%（光照相似度 88%），在合成数据上能够保持高质量全局照明，显著优于对等方法。

**⚠️ 局限性**

受训练光源种类限制，对未见光源形状的泛化有限；需完整视频覆盖，缺失视角会产生伪影。

---

## 33. DMV-AVP: Distributed Multi-Vehicle Autonomous Valet Parking using Autoware

**arXiv ID:** 2601.16327 | [PDF](https://arxiv.org/pdf/2601.16327v1)

**作者:** Zubair Islam `[一作]` (Ontario Tech University), Mohamed El-Darieby `[通讯]` (Ontario Tech University)

**通讯引用:** 274 | [OpenAlex ID](https://openalex.org/A5047581908)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `e0540dec-d77f-42db-94ae-d039248f6393` `337e632d-5d88-4e08-b332-1e58d8df0f5e`

**🎯 论文内容**

开发了一个分布式多车自动代客泊车模拟系统（DMV‑AVP），整合了状态驱动的协同、排队与预留管理模块以及基于YOLOv5的实时泊车位检测模块。

**💡 创新点**

首次实现分布式多车代客泊车的完整模拟，创新性地将基于视觉的泊车位检测与分布式协同算法结合，并利用Zenoh实现跨主机低延迟同步。

**🔧 技术方法**

技术栈包括 ROS 2 Humble、Autoware Universe、AWSIM Labs、Unity、Zenoh、YOLOv5 以及自研的 U‑YOLO 与 Multi‑Vehicle AVP 节点。

**📊 数据集**

使用仿真生成的数据；泊车位检测采用预训练 YOLOv5 模型，无外部公开数据集。

**📈 对比分析**

通过在两台与三台主机上运行实验，验证了系统的确定性同步、无冲突泊车行为与可扩展性能；在三主机配置下虽出现内存瓶颈导致偶发不稳定，但整体仍能保持低延迟通信和闭环控制。

**⚠️ 局限性**

局限包括：集中式协同导致单点故障；U‑YOLO 对车辆颜色和外观的鲁棒性不足；Retrieval 阶段受限于 Freespace Planner；硬件资源受限时性能下降；未来需要去中心化/容错协同、提升检测泛化能力与真实平台验证。

---

## 34. Better as Generators Than Classifiers: Leveraging LLMs and Synthetic Data for Low-Resource Multilingual Classification

**arXiv ID:** 2601.16278 | [PDF](https://arxiv.org/pdf/2601.16278v1)

**作者:** Branislav Pecher `[一作]` (Kempelen Institute of Intelligent Technologies), Maria Bielikova `[通讯]` (Kempelen Institute of Intelligent Technologies)

**通讯引用:** 3358 | [OpenAlex ID](https://openalex.org/A5030414237)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `67630363-6be0-4f51-ab05-7198250671a5` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在多语种低资源分类任务中，利用大型语言模型生成合成样本并用以训练或微调小型模型，形成一种数据驱动的蒸馏流程。

**💡 创新点**

首次证明大型模型更适合作为数据生成器而非直接分类器，且仅需少量合成样本即可使小模型在低资源语言和不常见任务上超过生成模型本身。

**🔧 技术方法**

使用LLaMA‑3 70B生成样本，对XLM‑RoBERTa、LLaMA‑3.1‑8B、Gemma‑3‑4B、Qwen‑2.5‑7B等模型进行微调、指令调优、上下文学习，并采用LoRA、随机子采样等技术。

**📊 数据集**

涉及11种语言（阿塞拜疆语、罗马尼亚语、斯洛文尼亚语、泰卢固语、威尔士语、希伯来语、印尼语、斯瓦希里语、泰语、英语、德语）和4个分类任务（情感、主题、意图、讽刺检测），使用公开数据集如SIB‑200、MASSIVE、SemEval‑2021‑Task‑7等。

**📈 对比分析**

通过平均准确率和标准差比较，发现小模型在所有语言组上均能在仅50个样本时超过生成模型，低资源语言提升幅度最高（≈18%），但高资源语言和流行任务提升有限；相较于人工标注，小样本场景下性能相近，规模增大后人工数据更优。

**⚠️ 局限性**

实验受限于仅选用单一生成模型、单一提示模板、海量计算资源导致高方差和过拟合风险，合成样本多样性不足且对超参数极度敏感。

---

## 35. Contrastive Knowledge Distillation for Embedding Refinement in Personalized Speech Enhancement

**arXiv ID:** 2601.16235 | [PDF](https://arxiv.org/pdf/2601.16235v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876`

---

## 36. A New Paradigm for Trusted Respiratory Monitoring Via Consumer Electronics-grade Radar Signals

**arXiv ID:** 2601.16241 | [PDF](https://arxiv.org/pdf/2601.16241v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 37. Generating Literature-Driven Scientific Theories at Scale

**arXiv ID:** 2601.16282 | [PDF](https://arxiv.org/pdf/2601.16282v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 38. Computational Foundations for Strategic Coopetition: Formalizing Collective Action and Loyalty

**arXiv ID:** 2601.16237 | [PDF](https://arxiv.org/pdf/2601.16237v1)

**作者:** Vik Pant `[一作]` (University of Toronto), Eric Yu `[通讯]` (University of Toronto)

**通讯引用:** 43110 | [OpenAlex ID](https://openalex.org/A5100731451)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文扩展了战略共赢的计算基础，提出了团队层面的集体行动与忠诚机制模型，并通过理论推导和仿真验证了解决自由搭便车问题的有效性。

**💡 创新点**

创新点包括：
- 将忠诚拆解为忠诚效益（整合福利内化与暖流）与成本容忍两大机制；
- 通过 i* 结构依赖与团队凝聚度将团队内部动力与外部共赢关系相连；
- 在多代理与软件工程团队两大应用场景中统一建模，展示跨领域通用性；
- 提供完整的模型翻译框架与实证案例（Apache HTTP Server 1995‑2023）。

**🔧 技术方法**

技术手段：
- 基于均衡理论的团队生产函数与忠诚调制效用函数；
- 采用 i* 依赖网络估算团队内部依赖权重；
- 设计迭代求解算法求取团队生产均衡；
- 通过蒙特卡洛参数搜索与统计检验（p<0.001, Cohen’s d=0.71）验证模型；
- 对软件项目数据与多代理系统进行实验与案例验证。

**📊 数据集**

数据集：
- 软件工程领域：Apache HTTP Server 开源项目的贡献数据（1995‑2023 年）；
- 计算机仿真数据：3,125 种参数配置的实验组合（覆盖团队规模、生产函数、忠诚程度等）。

**📈 对比分析**

比较与性能：
- 与纯自利模型对比，模型能显著提升团队投入（最高可达 15× 的努力差异）并消除自由搭便车；
- 在 Apache 项目中，模型完全重现贡献分布，验证点 60/60；
- 模型在 3,125 组配置中均满足六大行为验证阈值，证明其鲁棒性和泛化能力。

**⚠️ 局限性**

局限性：
- 忠诚度估计仍依赖经验权重与人工测度，难以统一量化；
- 模型假设投入与产出共享方式为均分，实际项目可能有复杂的报酬机制；
- 对团队外部环境变化（如资源短缺、激励变动）的动态适配尚未完整建模；
- 多代理系统中将心理机制映射为计算奖励时需要更多的实验调参，易出现不稳定行为。

---

## 39. Domain Specific Specialization in Low-Resource Settings: The Efficacy of Offline Response-Based Knowledge Distillation in Large Language Models

**arXiv ID:** 2601.16219 | [PDF](https://arxiv.org/pdf/2601.16219v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 40. VibeTensor: System Software for Deep Learning, Fully Generated by AI Agents

**arXiv ID:** 2601.16238 | [PDF](https://arxiv.org/pdf/2601.16238v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 41. Identity, Cooperation and Framing Effects within Groups of Real and Simulated Humans

**arXiv ID:** 2601.16355 | [PDF](https://arxiv.org/pdf/2601.16355v1)

**作者:** Suhong Moon `[一作]` (University of California), John Canny `[通讯]` (Google DeepMind)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用预训练大型语言模型（LLM）结合丰富的叙事背景、时间绑定与一致性过滤技术，模拟人类在社会困境游戏（Dictator 和 Trust Game）中的决策行为，并考察身份与情境因素对偏好差异的影响。

**💡 创新点**

创新点在于：①通过深度绑定的叙事背书显著提升模型对身份驱动行为的再现；②引入时间上下文与一致性检查，增强模拟的真实性；③与传统的仅依赖身份问答或规则生成人物描述的方法相比，显著提高了对实验数据的匹配度。

**🔧 技术方法**

核心技术包括：预训练大模型（Mistral‑Small、Mixtral、Qwen‑2.5）、长文本提示与叙事生成、温度1.0的采样、基于指令调优的批评家过滤、时间绑定提示（QA 形式）、一致性过滤器；以及通过这些提示构建的多维身份属性。

**📊 数据集**

数据集：①来自2014、2015、2019年公开人类实验（Dictator 与 Trust Game）的行为记录；②从美国 Voices Project 采访得到的叙事背景样本；③从公开人类实验原始数据提取的受试者人口统计信息，用于匹配虚拟角色。

**📈 对比分析**

与三种基线提示（单一问答、规则式一人称/二人称自传）对比，在四种提示策略（包括本方法）下评估各自的党派差异 Δ。实验结果显示，本方法在所有模型与游戏中均取得最接近人类实验的 Δ，说明其在模拟身份偏好方面表现最佳。

**⚠️ 局限性**

局限性：①模型继承预训练数据中的社会与政治偏见，可能导致少数群体代表不足；②对时间跨度的模拟受训练语料覆盖范围限制；③实现需要大模型与长上下文窗口，计算成本高；④实验聚焦美国两党，泛化到其他文化或多语言场景尚未验证。

---

## 42. Scalable Screw-Theoretic Synthesis for PDE-Based Dynamic Modeling of Multibody Flexible Manipulators

**arXiv ID:** 2601.16242 | [PDF](https://arxiv.org/pdf/2601.16242v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 43. NOIR: Privacy-Preserving Generation of Code with Open-Source LLMs

**arXiv ID:** 2601.16354 | [PDF](https://arxiv.org/pdf/2601.16354v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 44. Ordering-based Causal Discovery via Generalized Score Matching

**arXiv ID:** 2601.16249 | [PDF](https://arxiv.org/pdf/2601.16249v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 45. Policy-Embedded Graph Expansion: Networked HIV Testing with Diffusion-Driven Network Samples

**arXiv ID:** 2601.16233 | [PDF](https://arxiv.org/pdf/2601.16233v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39`

---

## 46. Space Filling Curves is All You Need: Communication-Avoiding Matrix Multiplication Made Simple

**arXiv ID:** 2601.16294 | [PDF](https://arxiv.org/pdf/2601.16294v1)

**作者:** Evangelos Georganas `[一作]` (Intel Corporation), Pradeep Dubey `[通讯]` (Intel Corporation)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了一种基于空间填充曲线（SFC）的通信避免矩阵乘法算法SFC-CA，提供了平台和形状不敏感的GEMM实现。

**💡 创新点**

创新点在于将通用Hilbert曲线用于GEMM计算空间分块，实现天然的数据局部性，并通过复制输入张量和2.5D/3D通信避免技术实现理论最优的数据移动量；该方法不需要手工调优即可在多核CPU上获得接近roofline的性能。

**🔧 技术方法**

采用了Generalized Hilbert Curve、Tensor Processing Primitives（BRGEMM TPP）、2.5D/3D通信避免算法、SFC映射、JIT微核（LIBXSMM）以及OpenMP并行等技术。

**📊 数据集**

实验使用Bfloat16矩阵乘法，尺寸集合{512,1024,2048,4096,8192}的M×N×K三元组，共125种形状，测试平台包括Intel Emerald Rapids、Granite Rapids、AMD EPYC ZEN5和AWS Graviton4。

**📈 对比分析**

通过与厂商优化库oneDNN（x86）和ACL（Arm）在相同平台、相同数据集上对比，利用roofline模型评估，SFC-CA在所有平台上实现1.4×至2×的几何平均加速，且性能始终贴近紧致roofline；同时L2缓存命中率显著提升，进一步说明数据移动被有效降低。

**⚠️ 局限性**

局限性包括：目前仅验证了共享内存多核CPU，GPU或更大规模分布式环境尚未测试；复制因子引入的额外内存占用和归约开销在极大矩阵或高复制因子时可能成为瓶颈；对极小或非常不规则形状矩阵的加速效果相对有限。

---

## 47. Identifying Concurrency Bug Reports via Linguistic Patterns

**arXiv ID:** 2601.16338 | [PDF](https://arxiv.org/pdf/2601.16338v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 48. Mixing Expert Knowledge: Bring Human Thoughts Back To the Game of Go

**arXiv ID:** 2601.16447 | [PDF](https://arxiv.org/pdf/2601.16447v1)

**作者:** Yichuan Ma `[一作]` (Fudan University), Kai Chen `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**通讯引用:** 32630 | [OpenAlex ID](https://openalex.org/A5100437924)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本论文提出LoGos，一款能够在围棋领域达到职业水平同时保持优秀通用推理能力的大型语言模型。

**💡 创新点**

创新点在于通过先构造基于启发式规则的专家级围棋数据与通用长链式推理数据混合冷启动，然后利用GRPO强化学习实现自我探索，将通用推理与专业知识无缝融合。

**🔧 技术方法**

技术主要包括启发式规则生成的围棋下一步预测与解说数据、长链式推理数据混合微调、Group Relative Policy Optimization（GRPO）强化学习以及基于KataGo的奖励函数设计。

**📊 数据集**

使用的数据集包括约1千万条棋局状态的10M规模预测数据、10K规模解说数据、OpenThoughts-114K、NuminaMath-QwQ-CoT-5M、OpenCodeReasoning、Bespoke-Stratos-17k、AM-DeepSeek-R1-Distilled-1.4M等通用推理数据，以及自构建的KataGo-Bench-1K评测基准。

**📈 对比分析**

与现有通用LLM（如Claude3.7-Sonnet、DeepSeek-R1等）和专业围棋模型（KataGo-HumanSL系列）对比，LoGos在KataGo-Bench-1K上的预测准确率从基线的34.3%提升至88–89%，超过所有专业模型；在GPQA、AIME、MATH、LiveCodeBench等通用基准上也保持或略优于同规模LLM，证明模型兼顾专业与通用。

**⚠️ 局限性**

局限在于仍受限于启发式规则的质量、奖励函数设计的稀疏性，以及在更长棋局时出现的“上下文诅咒”问题，虽可通过2D棋盘渲染缓解，但整体仍需更高效的数据生成与更通用的策略学习机制。

---

## 49. Memory-V2V: Augmenting Video-to-Video Diffusion Models with Memory

**arXiv ID:** 2601.16296 | [PDF](https://arxiv.org/pdf/2601.16296v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 50. On the Expressive Power of Floating-Point Transformers

**arXiv ID:** 2601.16450 | [PDF](https://arxiv.org/pdf/2601.16450v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 51. Ringmaster: How to juggle high-throughput host OS system calls from TrustZone TEEs

**arXiv ID:** 2601.16448 | [PDF](https://arxiv.org/pdf/2601.16448v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 52. RENEW: Risk- and Energy-Aware Navigation in Dynamic Waterways

**arXiv ID:** 2601.16424 | [PDF](https://arxiv.org/pdf/2601.16424v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 53. VTFusion: A Vision-Text Multimodal Fusion Network for Few-Shot Anomaly Detection

**arXiv ID:** 2601.16381 | [PDF](https://arxiv.org/pdf/2601.16381v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 54. Learning Domain Knowledge in Multimodal Large Language Models through Reinforcement Fine-Tuning

**arXiv ID:** 2601.16419 | [PDF](https://arxiv.org/pdf/2601.16419v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 55. Jacobian Scopes: token-level causal attributions in LLMs

**arXiv ID:** 2601.16407 | [PDF](https://arxiv.org/pdf/2601.16407v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 56. Towards a Theoretical Understanding to the Generalization of RLHF

**arXiv ID:** 2601.16403 | [PDF](https://arxiv.org/pdf/2601.16403v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 57. PolyAgent: Large Language Model Agent for Polymer Design

**arXiv ID:** 2601.16376 | [PDF](https://arxiv.org/pdf/2601.16376v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 58. Improving the Accuracy of Community Detection on Signed Networks via Community Refinement and Contrastive Learning

**arXiv ID:** 2601.16372 | [PDF](https://arxiv.org/pdf/2601.16372v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39`

---

## 59. Game-to-Real Gap: Quantifying the Effect of Model Misspecification in Network Games

**arXiv ID:** 2601.16367 | [PDF](https://arxiv.org/pdf/2601.16367v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

---

## 60. Log-Likelihood Loss for Semantic Compression

**arXiv ID:** 2601.16461 | [PDF](https://arxiv.org/pdf/2601.16461v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

---

## 61. Segregation Before Polarization: How Recommendation Strategies Shape Echo Chamber Pathways

**arXiv ID:** 2601.16457 | [PDF](https://arxiv.org/pdf/2601.16457v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39`

---

## 62. Cramér-Rao Bound Minimization for Flexible Intelligent Metasurface-Enabled ISAC Systems

**arXiv ID:** 2601.16455 | [PDF](https://arxiv.org/pdf/2601.16455v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

---

## 63. Masked Face Recognition under Different Backbones

**arXiv ID:** 2601.16440 | [PDF](https://arxiv.org/pdf/2601.16440v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 64. VISTA-PATH: An interactive foundation model for pathology image segmentation and quantitative analysis in computational pathology

**arXiv ID:** 2601.16451 | [PDF](https://arxiv.org/pdf/2601.16451v1)

**作者:** Peixian Liang `[一作]` (University of Pennsylvania), Zhi Huang `[通讯]` (University of Pennsylvania)

**通讯引用:** 8274 | [OpenAlex ID](https://openalex.org/A5022500975)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发了一款交互式、文本提示驱动的病理图像分割基础模型VISTA‑PATH，能够统一多器官、多类组织的分割；

**💡 创新点**

创新点包括①构建1.6 百万像素‑掩码‑文本三元组的VISTA‑PATH Data数据集；②在视觉‑语言框架中同时融合语义文本提示和空间边框提示，实现多类别的语义‑空间联合推断；③设计实时人机交互流程，将少量补丁级专家修正通过边框提示传播至全切片像素级分割；④将高保真分割结果转化为可解释的Tumor Interaction Score（TIS），用于患者生存预测；

**🔧 技术方法**

主要技术为：基于PLIP预训练的视觉‑文本编码器、SAM提示编码器、跨注意力融合模块、Transformer解码器；补丁级分类器与主动学习框架实现人机交互；TIS计算与多层感知机融合的生存预测管道；

**📊 数据集**

使用了VISTA‑PATH Data（22公开资源，覆盖9个器官、93类、1.6 百万样本），外部验证集包括LungHP、OCDC、Visium HD、Xenium、TCGA‑COAD等；

**📈 对比分析**

与MedSAM、BiomedParse、Res2Net等基线比较，VISTA‑PATH在内部和外部数据上平均Dice提升0.1–0.4；在人机交互实验中4–5轮迭代即可将Dice提升至0.8以上；在TCGA‑COAD生存预测中TIS提升C‑index约17–21%，显著优于ABMIL和MedSAM；

**⚠️ 局限性**

局限包括：尚未支持更丰富的专家交互形式（如不确定性标注、区域描绘）；缺乏大规模多机构临床验证；对不同扫描仪、放大倍率和染色协议的鲁棒性仍待进一步评估。

---

## 65. Emotion-LLaMAv2 and MMEVerse: A New Framework and Benchmark for Multimodal Emotion Understanding

**arXiv ID:** 2601.16449 | [PDF](https://arxiv.org/pdf/2601.16449v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 66. Endless Terminals: Scaling RL Environments for Terminal Agents

**arXiv ID:** 2601.16443 | [PDF](https://arxiv.org/pdf/2601.16443v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 67. AlphaFace: High Fidelity and Real-time Face Swapper Robust to Facial Pose

**arXiv ID:** 2601.16429 | [PDF](https://arxiv.org/pdf/2601.16429v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 68. DCCS-Det: Directional Context and Cross-Scale-Aware Detector for Infrared Small Target

**arXiv ID:** 2601.16428 | [PDF](https://arxiv.org/pdf/2601.16428v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 69. Safe Multitask Molecular Graph Networks for Vapor Pressure and Odor Threshold Prediction

**arXiv ID:** 2601.16426 | [PDF](https://arxiv.org/pdf/2601.16426v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 70. Bayesian Experimental Design for Model Discrepancy Calibration: A Rivalry between Kullback--Leibler Divergence and Wasserstein Distance

**arXiv ID:** 2601.16425 | [PDF](https://arxiv.org/pdf/2601.16425v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 71. PyHealth 2.0: A Comprehensive Open-Source Toolkit for Accessible and Reproducible Clinical Deep Learning

**arXiv ID:** 2601.16414 | [PDF](https://arxiv.org/pdf/2601.16414v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 72. Toward Agentic Software Project Management: A Vision and Roadmap

**arXiv ID:** 2601.16392 | [PDF](https://arxiv.org/pdf/2601.16392v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 73. iPDB -- Optimizing SQL Queries with ML and LLM Predicates

**arXiv ID:** 2601.16432 | [PDF](https://arxiv.org/pdf/2601.16432v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7`

---

## 74. A Cosine Network for Image Super-Resolution

**arXiv ID:** 2601.16413 | [PDF](https://arxiv.org/pdf/2601.16413v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 75. Study of Switched Step-size Based Filtered-x NLMS Algorithm for Active Noise Cancellation

**arXiv ID:** 2601.16382 | [PDF](https://arxiv.org/pdf/2601.16382v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

---

## 76. Consensus In Asynchrony

**arXiv ID:** 2601.16460 | [PDF](https://arxiv.org/pdf/2601.16460v1)

**作者:** Ivan Klianev `[一作]` `[通讯]` (Transactum Pty Ltd), Ivan Klianev (Transactum Pty Ltd)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

**🎯 论文内容**

论文提出了一种基于事件同步的容错确定性共识算法，证明在完全异步环境下可实现安全、活性并容忍一次崩溃的向量共识。

**💡 创新点**

创新点在于把共识分为数据相关与数据无关两类，并揭示FLP不可能定理依赖的隐藏假设，从而证明向量共识（数据无关）在异步中是可行的；同时给出完整的三阶段（初始、提议、决策）算法与证明。

**🔧 技术方法**

使用事件驱动的可靠广播、Blend/Completion/Update 规则、原子广播模型以及基于向量的决策计算，实现了确定性并发执行。

**📊 数据集**

实验数据来源于在5个进程的完全同步系统中，对所有可能的16条正常链路与4条故障链路组合（总计23,474,025组）进行暴力搜索，验证向量共识可达而二值共识受“平局”影响而失效。

**📈 对比分析**

与传统二值共识算法相比，实验表明向量共识在同等条件下始终达成一致且不受“平局”导致的不终止；性能上虽然未给出时延指标，但证明了在任意消息延迟下均能终止，满足活性与安全性。

**⚠️ 局限性**

局限性包括：只能容忍单一进程崩溃；需要原子广播支持；算法适用于N≥5且假设消息不丢失；对更大规模系统的实验不可行，理论复杂度仍高。

---

## 77. Bridging Expert Reasoning and LLM Detection: A Knowledge-Driven Framework for Malicious Packages

**arXiv ID:** 2601.16458 | [PDF](https://arxiv.org/pdf/2601.16458v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 78. Brownian ReLU(Br-ReLU): A New Activation Function for a Long-Short Term Memory (LSTM) Network

**arXiv ID:** 2601.16446 | [PDF](https://arxiv.org/pdf/2601.16446v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 79. Two classes of LCD codes derived from $(\mathcal{L},\mathcal{P})$-TGRS codes

**arXiv ID:** 2601.16438 | [PDF](https://arxiv.org/pdf/2601.16438v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

---

## 80. Cite-While-You-Generate: Training-Free Evidence Attribution for Multimodal Clinical Summarization

**arXiv ID:** 2601.16397 | [PDF](https://arxiv.org/pdf/2601.16397v1)

**作者:** Qianqi Yan `[一作]` (University of California), Krishnaram Kenthapadi `[通讯]` (Oracle Health AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出一种训练无关、基于解码器注意力的即时源引用框架，用于临床文本与图像摘要的可解释生成。

**💡 创新点**

创新点包括：① 通过聚合多层多头注意力并采用多数投票的方式，将噪声较大的 token‑level 注意力转化为稳定的句子‑级引用；② 提供两种多模态引用策略（原图补丁注意力和图像生成字幕），兼顾高保真与轻量化部署；③ 在推理时即完成引用，避免了后处理或再训练。

**🔧 技术方法**

技术：Transformer 解码器跨模态注意力、层/头均值聚合、top‑k 采样、句子映射与阈值筛选；使用 Qwen2.5‑VL 与 LLaVA‑NEXT 等开源多模态 LLM。

**📊 数据集**

数据集：CliConSummation（医生‑患者对话，包含文本与图像）和 MIMIC‑CXR（放射报告 Findings→Impression 任务）。

**📈 对比分析**

对比基线为句子级相似度（Sentence‑BERT）、图像相似度（CLIP）与模型自回归引用；实验显示在两种模型上均实现约 15% F1 提升，文本引用宏 F1 超过 70%，并在多模态场景下保持竞争性。

**⚠️ 局限性**

局限性：依赖注意力与真实证据的对齐，未在更多医学语料上验证；引用质量评估仅基于 LLM 判别器，缺乏人工标注；对图像的直接关注可能受图像尺寸/分辨率影响。

---

## 81. MDAFNet: Multiscale Differential Edge and Adaptive Frequency Guided Network for Infrared Small Target Detection

**arXiv ID:** 2601.16434 | [PDF](https://arxiv.org/pdf/2601.16434v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 82. Tight Regret Bounds for Bilateral Trade under Semi Feedback

**arXiv ID:** 2601.16412 | [PDF](https://arxiv.org/pdf/2601.16412v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

---

## 83. Reasoning-Enhanced Rare-Event Prediction with Balanced Outcome Correction

**arXiv ID:** 2601.16406 | [PDF](https://arxiv.org/pdf/2601.16406v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 84. ResAgent: Entropy-based Prior Point Discovery and Visual Reasoning for Referring Expression Segmentation

**arXiv ID:** 2601.16394 | [PDF](https://arxiv.org/pdf/2601.16394v1)

**作者:** Yihao Wang `[一作]` (Sun Yat-sen University), Meng Yang `[通讯]` (Sun Yat-sen University)

**通讯引用:** 9415 | [OpenAlex ID](https://openalex.org/A5037873810)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种新的Referring Expression Segmentation框架ResAgent，解决了现有基于多模态大模型的RES方法在点提示选择和坐标推理上的两大瓶颈，能够在只使用少量点提示的前提下生成准确的像素级分割掩模。

**💡 创新点**

创新点包括：1）Entropy-Based Point Discovery（EBD）——利用边框内的空间不确定性信息，基于熵最大化的原则选择高价值点，取代传统的随机或均匀采样；2）Vision-Based Reasoning（VBR）——通过在图像上渲染可视化标记并利用多模态大模型的VQA能力对点的正负进行视觉语义验证，摆脱了把坐标编码为文本的缺陷；3）联合使用两者形成四阶段粗细化流程，并在SAM解码器上进行风格适配，提升最终分割质量。

**🔧 技术方法**

技术细节包括：多模态大模型（如LLM）用于生成粗略边框、执行VQA推理；信息熵公式与梯度引导的点采样；视觉标记渲染与颜色编码；概率聚合方法提升判断稳定性；LoRA微调和SAM风格适配；早停策略控制点数。

**📊 数据集**

在RefCOCO、RefCOCO+、RefCOCOg和ReasonSeg四大RES基准上进行评估。

**📈 对比分析**

与非LLM专用模型、LLM图像通用模型以及LLM视频通用模型等多类方法对比，ResAgent在所有四个基准上均达到了或超过了当前最优性能。例如，在RefCOCO+ 0.0/0.5/1.0子集上分别获得78.32%、78.07%和78.04%的mIoU，超越了Text4Seg 77.3%/77.0%/76.8%；在ReasonSeg上实现72.74% gIoU 和 68.38% cIoU，明显领先VideoLISA。

**⚠️ 局限性**

局限性包括：1）仍依赖粗略边框作为先验，若边框定位错误仍可能影响点发现；2）点采样与VBR的计算成本相对较高，尤其在多点情形下推理时间可达数秒；3）在极其复杂或高度遮挡的场景下，视觉标记与VQA的推理可能仍出现误判；4）评估指标与人类感知不完全一致，精细边缘差异可能被误判为错误。

---

## 85. Cross-Lingual Activation Steering for Multilingual Language Models

**arXiv ID:** 2601.16390 | [PDF](https://arxiv.org/pdf/2601.16390v1)

**作者:** Rhitabrat Pokharel `[一作]` (Portland State University), Tanay Nagar `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种训练无关的推理时激活引导方法（CLAS），通过调节多语言 LLM 中共享与语言特定神经元的比例，提升低资源语言的性能。

**💡 创新点**

创新点在于：1）仅在推理阶段对激活进行轻量级、比例化的调节，避免覆写并保持原始激活比例；2）通过功能性分化而非强制对齐实现跨语言迁移；3）揭示性能提升与语言聚类分离度的正相关。

**🔧 技术方法**

使用技术包括：构造并行多语言输入、统计神经元激活并按类别（dead、specific、partial-shared、all-shared）分组、在桥接层对共享/特定神经元应用 β、γ 调节、使用 α 混合原始与调整后的激活。

**📊 数据集**

采用的公开数据集：XNLI（15 语种的 NLI 分类）和 XQuAD（12 语种的多语言 QA 生成），并利用 100 条并行样本进行神经元统计。

**📈 对比分析**

与基线模型和 Int μ（平均值覆写）进行比较。对 Llama 3.1 8B：XNLI 平均 Acc 提升 1.93%，F1 0.45%；对 Qwen 2.5 7B：XNLI 平均 Acc 提升 0.45%，F1 0.45%。在 XQuAD 上，Llama 平均 F1 提升 0.94，Qwen 平均提升 1.10，但差异未显著。总体提升主要集中在低资源语言，同时保持英语性能不变。

**⚠️ 局限性**

局限性包括：1）仅以英语为锚点，可能不适用于其他锚语言或完全无锚设置；2）对某些语言或模型组合会出现性能退化；3）参数 α、β、γ 的选择需要手动调优，缺乏自适应机制；4）分析聚焦于神经元激活，未深入探究注意头或更细粒度的网络机制。

---

## 86. Graph-Anchored Knowledge Indexing for Retrieval-Augmented Generation

**arXiv ID:** 2601.16462 | [PDF](https://arxiv.org/pdf/2601.16462v1)

**作者:** Zhenghao Liu `[一作]` (Northeastern University), Maosong Sun `[通讯]` (Tsinghua University)

**通讯引用:** 36343 | [OpenAlex ID](https://openalex.org/A5046448314)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了一种基于可演化知识图的 GraphAnchor 方法，利用动态更新的图在迭代检索过程中对检索文档进行索引、判断知识是否充分，并指导 LLM 生成更准确答案。

**💡 创新点**

创新点在于将图结构从静态知识表示转变为动态索引；每次检索后递增更新图，通过图评估知识充足性并生成后续查询，使图从仅仅的知识表征变成主动的检索索引与解释工具。

**🔧 技术方法**

采用迭代检索 + 检索‑生成框架；利用 LLM（如 Qwen2.5‑7B、Qwen3‑32B、Llama3.1‑8B）完成图构建、更新与答案生成；使用 bge‑large‑en‑v1.5 进行检索；对图实体与三元组进行文本化并包裹特殊 token；并进行注意力可视化等技术。

**📊 数据集**

四个多跳问答基准：MuSiQue、HotpotQA、2WikiMultiHopQA、Bamboogle。

**📈 对比分析**

与 Vanilla RAG 以及四种多轮检索基线（IRCoT、Iter‑RetGen、DeepNote、Search‑R1）进行 Token‑F1 与 Exact Match 对比；GraphAnchor 在所有基准上平均提升约 3%+，相较于深度检索方法提升约 10%，证明其显著性能优势。

**⚠️ 局限性**

受限于 LLM 的抽取与结构化能力，图的质量受模型限制；当前仅采用文本化的图表示，缺乏更丰富的结构化表达，可能导致关键信息丢失。

---

## 87. RubberDuckBench: A Benchmark for AI Coding Assistants

**arXiv ID:** 2601.16456 | [PDF](https://arxiv.org/pdf/2601.16456v1)

**作者:** Ferida Mohammad `[一作]` (Bryn Mawr College), Elizabeth Dinella `[通讯]` (Bryn Mawr College)

**通讯引用:** 229 | [OpenAlex ID](https://openalex.org/A5063180853)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个15题的多语言、真实上下文化问题基准RubberDuckBench，用于评估 AI 编码助手回答代码相关问题的能力。

**💡 创新点**

创新点在于：① 从 GitHub PR 评论中抽取并重写成可提问的上下文化问题；② 为开放式答案设计细粒度扣分 rubrics；③ 系统评估了 20 个主流 LLM 的回答质量、成本与幻觉率。

**🔧 技术方法**

采用了大语言模型（Claude、Grok、GPT、Gemini、Qwen、Llama 等）进行推理与回答，并用人工评审结合 rubrics 进行打分。

**📊 数据集**

数据集来自公开的 CodeReview（GitHub PR 评论）并挑选了 13 个高星值开源项目（Java、Python、C++），再通过 LLM‑human 方式生成 15 个问题。

**📈 对比分析**

通过在 15 题上平均 3 次测试，Grok 4 得分 69.3%，Claude Opus 4 68.5%，GPT‑5 67.8%；所有模型平均分约 60%；但完全正确回答不到 3 题，幻觉率平均 58.3%；成本与模型大小与性能无显著相关。

**⚠️ 局限性**

局限性包括：① PR 评论经 LLM 过滤可能引入偏差；② 所选项目为流行开源，可能已被 LLM 训练；③ 基准规模仅 15 题，难以覆盖更广泛的编码情境。

---

## 88. Cutting the Gordian Knot: Detecting Malicious PyPI Packages via a Knowledge-Mining Framework

**arXiv ID:** 2601.16463 | [PDF](https://arxiv.org/pdf/2601.16463v1)

**作者:** Wenbo Guo `[一作]` (Nanyang Technological University), Yang Liu `[通讯]` (Nanyang Technological University)

**通讯引用:** 81937 | [OpenAlex ID](https://openalex.org/A5100355964)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一种基于知识驱动的恶意PyPI包检测框架，能够利用已知检测失败转化为可用行为模式知识。

**💡 创新点**

创新点在于将检测失败信息挖掘为层级行为模式，并结合大型语言模型进行语义抽象，构建可迁移的知识库，实现对恶意与合法代码的语义区分。

**🔧 技术方法**

使用了层级模式挖掘（PrefixSpan）、大型语言模型（GPT‑4.1/mini）生成语义描述、RAG检索机制、行为序列抽象以及向量检索（FAISS）等技术。

**📊 数据集**

采用了18,137个PyPI包（含9,552恶意与9,552合法）、4,757个测试集，以及通过Intensio生成的伪装包和1,762个NPM包进行跨生态验证。

**📈 对比分析**

与Bandit4Mal、GuardDog、OSSGadget、Cerebro等基线工具比较，框架在原始测试集上实现99.50%准确率，仅误报2例，远优于基线（误报率≈15‑30%）。在伪装包上保持98.28%准确率，在最新和NPM包上也保持高精度（≥98%）。

**⚠️ 局限性**

局限性包括：只能检测包内代码，无法识别依赖混淆、远程安装等攻击；对LLM的hallucination有一定风险；需要定期更新知识库以应对新型恶意技术。

---

## 89. Exploring the Effects of Alignment on Numerical Bias in Large Language Models

**arXiv ID:** 2601.16444 | [PDF](https://arxiv.org/pdf/2601.16444v1)

**作者:** Ayako Sato `[一作]` (Tokyo Metropolitan University), Mamoru Komachi `[通讯]` (Hitotsubashi University)

**通讯引用:** 3140 | [OpenAlex ID](https://openalex.org/A5061931124)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了对齐后大语言模型在作为评估者时出现的数值偏差，并尝试通过温度缩放、分布校准和分数范围调整来缓解此偏差。

**💡 创新点**

首次将模型对齐与数值偏差关联，提出将分数范围视为可调参数来提升评估准确性，并用峰度（kurtosis）与皮尔逊相关系数联合评估偏差与性能。

**🔧 技术方法**

使用温度缩放、概率分布校准、分数范围调整等技术，并通过对齐前后模型对比来验证其效果。

**📊 数据集**

在三大任务数据集上进行实验：MTQE（机器翻译质量评估）、GECQE（语法纠错质量评估）和LCP（词汇复杂度预测），涉及7个语言对和多种模型。

**📈 对比分析**

对比结果显示，对齐后模型偏差显著增大、峰度升高且相关性下降；温度缩放与校准能减少偏差但不一定提升相关性；分数范围调整在多数模型与任务中既降低偏差又提升相关性，表现最为突出。

**⚠️ 局限性**

局限性包括：对齐细节未知、仅研究数值评分而非文本标签/排名、实验仅覆盖可输出数值的模型、缓解方法为启发式且针对任务，缺乏通用性。

---

## 90. Reinforcement Learning-Based Energy-Aware Coverage Path Planning for Precision Agriculture

**arXiv ID:** 2601.16405 | [PDF](https://arxiv.org/pdf/2601.16405v1)

**作者:** Beining Wu `[一作]` (South Dakota State University), Jun Huang `[通讯]` (South Dakota State University)

**通讯引用:** 5054 | [OpenAlex ID](https://openalex.org/A5020146420)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于Soft Actor-Critic（SAC）强化学习的能量感知覆盖路径规划（CPP）框架，用于农业机器人在带障碍和充电站的网格环境中实现高效完整覆盖。

**💡 创新点**

创新点包括：①将能量安全约束和安全返回目标嵌入奖励函数，实现覆盖、能量消耗与安全性三目标协同优化；②设计了结合CNN、注意力机制和LSTM的层次化网络，能够同时提取空间全局信息、局部细节与时间依赖；③通过多重权重调节的奖励函数实现对不同任务需求的灵活平衡。

**🔧 技术方法**

使用的技术包括：Soft Actor-Critic（SAC）离散动作版本、卷积神经网络（CNN）提取网格特征、Multi‑Head Self‑Attention 捕获全局依赖、长短期记忆网络（LSTM）处理时间序列、经验回放与目标网络软更新等。

**📊 数据集**

采用了自构造的三种15×15网格数据集（包含不同障碍密度和充电站布局），每个数据集通过随机生成障碍和充电站位置来模拟真实农田环境。

**📈 对比分析**

与传统启发式算法RRT、PSO和ACO进行比较。实验显示该SAC方法在三种地图上均实现了>90%的覆盖率，平均约为96%/92%/91%（分别对应地图1/2/3），约比第二佳算法高7–9%，同时将能量约束违规次数降低了83–85%（约32/126/114次），证明了显著的性能提升。

**⚠️ 局限性**

局限性包括：①实验仅在离散网格仿真环境中验证，缺乏真实农田现场测试；②对大规模地图的可扩展性尚未完全评估，网络尺寸和训练时间随网格尺寸增长；③奖励函数中权重需手工调参，适应不同任务场景仍需进一步自动化。

---

## 91. A Regularized Actor-Critic Algorithm for Bi-Level Reinforcement Learning

**arXiv ID:** 2601.16399 | [PDF](https://arxiv.org/pdf/2601.16399v1)

**作者:** Sihan Zeng `[一作]` (JPMorgan AI Research), Alec Koppel `[通讯]` (Johns Hopkins University Applied Physics Laboratory)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种单循环、一阶 actor‑critic 算法，用熵正则化实现下层 PL 条件，证明其可收敛到原未正则化双层 RL 目标的驻点。

**💡 创新点**

创新点：①引入随时间衰减的熵正则化，使下层满足可随正则化权重降低的 PL 条件；②在罚项重构框架下设计单循环更新；③利用多时间尺度随机逼近分析，首次给出原始双层 RL 的理论收敛率（ε⁻¹/¹⁰）并对固定正则化给出更快率（ε⁻¹/³）。

**🔧 技术方法**

使用技术包括：罚项重构、熵正则化、三重时间尺度（x、策略、值函数）更新、actor‑critic 估计、PL 条件分析、TD 价值函数更新、随机逼近与收敛分析。

**📊 数据集**

实验数据集：1）10×10 GridWorld 目标放置（可完全计算目标函数）；2）情感生成实验，使用规则情感分析器和小型四层 Transformer（语言模型）。

**📈 对比分析**

方法对比：与部分 SGD（RLHF 近似）、有限差分估计、嵌套循环等基线对比。实验表明单循环算法收敛更快、最终性能优于基线；固定正则化的变体导致子优解。

**⚠️ 局限性**

局限性：收敛率相对慢（ε⁻¹/¹⁰）；理论仅在离散软最大化表格化、充分探索且转移不随 x 变化的环境下成立；对连续/高维状态动作空间的推广和对 x 影响转移的情况尚未覆盖。

---

## 92. Gen-DBA: Generative Database Agents (Towards a Move 37 for Databases)

**arXiv ID:** 2601.16409 | [PDF](https://arxiv.org/pdf/2601.16409v1)

**作者:** Yeasir Rayhan `[一作]` (Purdue University), Walid G. Aref `[通讯]` (Purdue University)

**通讯引用:** 9910 | [OpenAlex ID](https://openalex.org/A5000123743)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了Generative Database Agent（Gen-DBA）框架，将数据库学习任务统一为生成式优化问题，并在空间查询调度任务上进行实验验证。

**💡 创新点**

创新点在于将大型Transformer与硬件计数器令牌（DB-Tokens）结合，构建统一嵌入空间，采用两阶段目标条件下的下一令牌预测，实现跨数据库、跨硬件、跨任务的通用生成式推理；首次在数据库系统中实现Move 37式的创造性优化。

**🔧 技术方法**

技术包括Transformer主干、基于硬件PMU的DB-Tokens令牌化、两阶段（预训练+微调）目标条件下的自回归下一令牌预测、生成式推理过程。

**📊 数据集**

使用自构建的经验数据集，涵盖多种数据库（PostgreSQL等）、多硬件平台（Intel、AMD、NVIDIA云服务器）、多工作负载（YCSB）以及对应的硬件性能计数器和优化结果；实验中主要使用YCSB 50%读/50%写负载。

**📈 对比分析**

对比Linux OS基线（OS:D、OS:I、SE:N、SN:N）在YCSB负载下的空间查询调度任务，0th‑generation Gen-DBA在Intel Skylake‑X、AMD Milan、NVIDIA等服务器上分别提升2.5‑5.3倍；预训练跨服务器版本进一步提升约2%。

**⚠️ 局限性**

局限性包括：仅实现了0th‑generation模型，缺乏跨任务的泛化能力和对人类可解释性的支持；数据规模仍不足以训练大规模模型；训练与推理成本高，且尚未在多任务或真实生产环境中验证。

---

## 93. Clarify or Answer: Reinforcement Learning for Agentic VQA with Context Under-specification

**arXiv ID:** 2601.16400 | [PDF](https://arxiv.org/pdf/2601.16400v1)

**作者:** Zongwan Cao `[一作]` (University of Washington), Lucy Lu Wang `[通讯]` (Allen Institute for AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种名为CoA（Clarify-or-Answer）的框架，能够在视觉问答任务中自动判断是否需要澄清，并在需要时生成单一精准的澄清问题后再给出最终答案。

**💡 创新点**

创新点在于将“是否问问”与“怎么问”拆分为两步，并通过多信号强化学习GRPO‑CR优化澄清问题的质量，使模型在保持单步交互的同时显著提升了对不完整语境的处理能力。

**🔧 技术方法**

使用了强化学习（GRPO‑CR）来训练澄清策略、交叉熵损失训练控制器以及大规模视觉‑语言模型（InternVL3‑2B、Qwen2.5‑VL‑7B/3B）作为回答器。

**📊 数据集**

构建并公开了一个新的数据集CtxClarify，包含275个需要外部上下文的视觉问答对及其人工验证的澄清问题，以及相应的对照集，用于训练和评估判断与生成两个模块。

**📈 对比分析**

与零样本提示和监督微调的基线相比，CoA在三种视觉‑语言模型上在控制器精确度、澄清质量（自评/人工评）以及端到端VQA准确率上均提升了约20–30%（平均提升15.3点/83%），且在ClearVQA、VizWiz等OOV数据集上也保持了显著优势。

**⚠️ 局限性**

局限性包括：仅处理单一缺失因子的单步澄清，难以扩展到多因子或多轮对话；缺乏对交互成本的建模；训练过程未对答案生成器进行联合优化，且数据集覆盖的语境类型与语言仍有限。

---

## 94. A Refinement of Vapnik--Chervonenkis' Theorem

**arXiv ID:** 2601.16411 | [PDF](https://arxiv.org/pdf/2601.16411v1)

**作者:** A. Iosevich `[一作]`, E. Wyman `[通讯]` (Binghamton University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `4bf3b852-21ff-4736-b125-37e24f3c9a32`

**🎯 论文内容**

对Vapnik–Chervonenkis定理的概率估计部分进行改进，将Hoeffding不等式替换为带有Berry–Esseen误差控制的正态近似，得到更精细的中等偏差收敛速度。

**💡 创新点**

创新点在于在中等偏差区间内引入正态尾概率的(ε√n)⁻¹ 前因子，显著降低指数项的前导常数，并给出可取的Berry–Esseen常数上界。

**🔧 技术方法**

采用了正态逼近、Berry–Esseen不等式、联合上界、组合生长函数与Sauer–Shelah引理等技术手段。

**📊 数据集**

无实验数据集，纯理论分析。

**📈 对比分析**

与传统Hoeffding-VC界定量比较：在ε√n 较大时，新的界的指数前导因子更小，性能更优；但在ε√n 保持有界时，改进有限，仍不优于经典界。

**⚠️ 局限性**

局限性：改进仅在中等偏差范围内体现，无法在小偏差或大偏差极端情形下提升；且依赖Bernoulli独立同分布假设，未探讨更一般情形。

---

## 95. White-Box Sensitivity Auditing with Steering Vectors

**arXiv ID:** 2601.16398 | [PDF](https://arxiv.org/pdf/2601.16398v1)

**作者:** Hannah Cyberey `[一作]` (University of Virginia), David Evans `[通讯]` (University of Virginia)

**通讯引用:** 98345 | [OpenAlex ID](https://openalex.org/A5100768980)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出基于白盒激活调节的LLM敏感性审计框架，通过操纵模型内部表示来评估偏差。

**💡 创新点**

创新点在于将激活调节（steering）技术直接应用于审计，量化模型对抽象概念的依赖性，并与传统黑盒测试进行对比。

**🔧 技术方法**

使用激活向量调节、线性梯度敏感度指标、Sobol’灵敏度分析等技术。

**📊 数据集**

使用性别与种族激活向量数据集（性别语料、AAL/WME 对话、Racial Identity 等）以及四个模拟决策任务数据集（司法、信用、录取、医疗）。

**📈 对比分析**

与黑盒扰动方法对比，白盒方法能揭示更多隐藏偏差，结果更稳健，误差更小，显示出明显优于传统黑盒评估。

**⚠️ 局限性**

局限在于假设概念线性可表示、对调节向量来源的依赖、对任务特定变量干扰评估不完整。

---

## 96. Cognitively-Inspired Tokens Overcome Egocentric Bias in Multimodal Models

**arXiv ID:** 2601.16378 | [PDF](https://arxiv.org/pdf/2601.16378v1)

**作者:** Bridget Leonard `[一作]` (University of Washington), Scott O. Murray `[通讯]` (University of Washington)

**通讯引用:** 5232 | [OpenAlex ID](https://openalex.org/A5031619204)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并在多模态语言模型中加入两类认知驱动的视角标记（embodiment tokens 与 rotation tokens），以提升模型在二维空间的视角取向推理能力。

**💡 创新点**

创新点在于将姿态关键点与方向信息直接嵌入 token 空间，构建可在模型内部完成 allocentric 视角转换的结构化表征，而不依赖外部视觉模块或额外网络层。

**🔧 技术方法**

技术细节包括在 LLaVA‑1.5‑13B 上使用 LoRA 微调、扩展 tokenizer、层级学习曲线（token 生成→链式思考→直接回答），以及基于 ViTPose/OrientAnything 提取姿态与旋转标记。

**📊 数据集**

实验使用 COCO‑2017 单人图像、Isle Bricks V2、COCO 2017 验证集、3DSRBench 以及自建的 Blender 生成 Perspective‑Taking benchmark。

**📈 对比分析**

与基线 LLaVA 与 GPT‑4o 进行对比，评价指标为对齐/未对齐情景下的准确率；在视角取向基准上未对齐项从 0% 提升至 100%，在 Isle Bricks 及 COCO 验证集上分别提升约 50% 与 72%，在 3DSRBench 上提升约 21%；旋转标记在非人类参考上表现更好，整体提升显著。

**⚠️ 局限性**

局限性包括：仅处理二维方向，缺乏深度信息；embodiment token 只能描述人类身体姿态，难以推广至动物或无关物体；需要大量标注关键点/方向数据；对 3D 视角取向的支持仍不充分。

---

## 97. LOGICAL-COMMONSENSEQA: A Benchmark for Logical Commonsense Reasoning

**arXiv ID:** 2601.16504 | [PDF](https://arxiv.org/pdf/2601.16504v1)

**作者:** Obed Junias `[一作]` (University of Colorado Boulder), Maria Leonor Pacheco `[通讯]` (University of Colorado Boulder)

**通讯引用:** 1003 | [OpenAlex ID](https://openalex.org/A5005560875)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个名为Logical CSQA的基准，用逻辑组合（AND、OR、NEITHER/NOR）重新定义常识推理，将多答案情况转化为可组合的原子命题；

**💡 创新点**

创新点在于将常识推理从单一答案转变为多答案组合，并通过社会共识验证引入可组合可信度层级；

**🔧 技术方法**

技术包括使用GPT‑4o‑mini进行候选答案生成与精炼、符号程序实现逻辑组合、Cohen κ衡量人类共识、以及零样本、少样本和链式思考提示的评测；

**📊 数据集**

使用CommonsenseQA作为基础数据集，扩展为19,996个组合实例，按AND、OR、NEITHER/NOR和MIXED四种关系均匀划分；

**📈 对比分析**

在零样本/少样本条件下，LLM在AND上可达≈80%，OR约70%，但在NEITHER/NOR仅≈13%；在微调后可达≈90%；与CommonsenseQA对比，模型在单一答案上表现更好，但在组合逻辑上性能显著下降；

**⚠️ 局限性**

局限性包括：逻辑运算仅涵盖AND/OR/NEITHER/NOR，未覆盖蕴含、排他或时间因果结构；实验仅评估解码器LLM的提示方式，编码器模型仅在监督下训练；缺乏更广泛的模型和任务对比，文化偏见可能存在。

---

## 98. Expert Knowledge-Guided Decision Calibration for Accurate Fine-Grained Tree Species Classification

**arXiv ID:** 2601.16498 | [PDF](https://arxiv.org/pdf/2601.16498v1)

**作者:** Chen Long `[一作]` (Wuhan University), Bisheng Yang `[通讯]` (Wuhan University)

**通讯引用:** 7937 | [OpenAlex ID](https://openalex.org/A5100772177)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了EKDC-Net框架，通过外部领域专家与本地模型结合，实现对细粒度树种分类的决策校准。

**💡 创新点**

创新点包括：①局部先验引导知识提取模块（LPKEM），利用CAM热图精确筛选专家关注区域；②不确定性引导决策校准模块（UDCM），动态估计类别与实例不确定性并自适应加权；③将两模块集成为轻量化、可插拔的外部专家校准方案，显著提升长尾与高相似类别的识别。

**🔧 技术方法**

技术手段包括：使用Vision Foundation Model BioCLIP2作为领域专家；CAM热图生成与二值掩码过滤背景噪声；多尺度特征融合；不确定性估计（类别难度嵌入+熵/置信度）及基于分箱的动态加权。

**📊 数据集**

实验数据集包括：自建CU-Tree102（102类、9134张），以及RStree（极端长尾、8324张）和Jekyll（23类、4804张）用于泛化验证。

**📈 对比分析**

与ResNet‑50、ViT‑Base、Swin‑Base、MPSA、HERBS、CGL等基线对比，在CU-Tree102上平均提升6.42%准确率、11.46%精度；在RStree上宏观F1提升至49.63%；在Jekyll无训练情况下，准确率提升46%以上。

**⚠️ 局限性**

局限性在于仍需依赖预训练领域专家，面对极端噪声或完全无关背景时校准效果有限；模块参数需针对不同任务调优；未解决跨域标签不一致和完全缺失样本的挑战。

---

## 99. BoostFGL: Boosting Fairness in Federated Graph Learning

**arXiv ID:** 2601.16496 | [PDF](https://arxiv.org/pdf/2601.16496v1)

**作者:** Zekai Chen `[一作]` (Beijing Institute of Technology), Guoren Wang `[通讯]` (Beijing Institute of Technology)

**通讯引用:** 7036 | [OpenAlex ID](https://openalex.org/A5054991337)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文在联邦图学习框架中提出BoostFGL，通过节点加权、拓扑加权和模型加权三阶段协同提升机制，以解决少数类与异质结构的公平性问题。

**💡 创新点**

创新点在于将公平性提升拆解为客户端训练、消息传播和服务器聚合三个阶段，并分别设计轻量级的“加权”模块，形成可插拔且兼容现有子图FedAvg协议的完整公平提升体系。

**🔧 技术方法**

技术实现包括：节点难度指数移动平均 (EMA) 并通过线性权重提升训练信号；基于节点难度与邻居异质性构建的指数拓扑权重，用于调节注意力机制；以及基于更新幅度与公平差距的信任权重，对服务器聚合进行可靠度加权。

**📊 数据集**

实验涵盖九个公开图数据集（Cora、Citeseer、PubMed、Coauthor、Wiki、Ogbn-arxiv、Products 等）以及不同规模与异质性组合，验证方法的通用性。

**📈 对比分析**

与 FedAvg、MOON、FedSage+、FedTAD、AdaFGL、FedGTA、FedSpary 和 FairFGL 等基线比较，BoostFGL 在总体 F1 和准确率上均保持领先或第二名，并在公平指标 Hete-F1、Hete-min-F1 上提升超过 8%，同时保持低通信与计算开销。

**⚠️ 局限性**

局限在于对超参数（如 λ、τ 等）的选择仍有一定敏感性，且在极端异构或极大规模子图划分时的表现尚待进一步验证；此外，虽然已证明可与差分隐私等安全机制结合，但在强隐私预算下的性能衰减尚未全面评估。

---

## 100. Curate-Train-Refine: A Closed-Loop Agentic Framework for Zero Shot Classification

**arXiv ID:** 2601.16530 | [PDF](https://arxiv.org/pdf/2601.16530v1)

**作者:** Gaurav Maheshwari `[一作]` (Diabolocom), Kevin El Haddad `[通讯]` (Diabolocom and ISIA Lab - University of Mons)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于LLM的闭环生成‑评估‑改进框架，用以自动化生成和优化轻量文本分类器的训练数据，从而在不依赖大模型推理的情况下实现低延迟推断。

**💡 创新点**

创新点在于将LLM从传统的推理工具转变为数据策划者，通过动态分析模型错误并针对性生成难例，实现了零样本和少样本场景下的自适应数据增强，显著提升了小模型的性能。

**🔧 技术方法**

核心技术包括ReAct式LLM代理、循环式数据生成与诊断、SetFit与EuroBERT轻量化编码器、以及基于GPT‑5的多轮生成与评估工具；在训练循环中结合准确率、宏F1、混淆矩阵等指标进行错误模式分析。

**📊 数据集**

实验使用四大文本分类基准：SST‑5（情感细粒度）、Emotion（情感识别）、CR（商品评论情感）和AG News（新闻主题）进行评估。

**📈 对比分析**

与传统Baseline、Prompt、Anchored SetFit、ADAPET、GLiClass等方法比较，零样本时该框架在三项数据集上均优于更大参数量的GLiClass，在少样本（k=2,4,8）场景下也持续领先，整体提升幅度可达5–10个百分点。

**⚠️ 局限性**

局限性包括对LLM生成质量和调度的依赖、迭代循环的计算成本（离线阶段）、以及在极其稀疏标签或高度专业化任务中可能需要更细粒度的错误分析工具或人工干预。

---

## 101. Doc2AHP: Inferring Structured Multi-Criteria Decision Models via Semantic Trees with LLMs

**arXiv ID:** 2601.16479 | [PDF](https://arxiv.org/pdf/2601.16479v1)

**作者:** Hongjia Wu `[一作]` (Zhejiang University), Wei Chen `[通讯]` (Zhejiang University)

**通讯引用:** 66171 | [OpenAlex ID](https://openalex.org/A5100344384)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出Doc2AHP框架，通过LLM与AHP的结构约束实现从非结构化文本自动构建多准则决策模型。

**💡 创新点**

创新点在于将语义层次聚类作为结构先验，引入多智能体协同与一致性优化，兼顾可解释性与数值一致性。

**🔧 技术方法**

采用大型语言模型（GPT‑5.2/Llama‑3.1）与层次聚类、对数最小二乘优化、几何平均聚合等技术。

**📊 数据集**

使用自构造的DecisionBench（电影、酒店、啤酒三大领域的多情境评估数据），并在IMDb、HotelRec、Beer Advocate等公开数据集上扩充。

**📈 对比分析**

与标准AHP和Debate‑AHP对比，Doc2AHP在NDCG@5上均优于基线，且在不同模型规模下保持100%一致性通过率。

**⚠️ 局限性**

局限在于需要先验的层次聚类参数与多智能体协作的调参，且构建结构相对耗时，难以适用于极大规模文档集合。

---

## 102. GNSS-based Lunar Orbit and Clock Estimation With Stochastic Cloning UD Filter

**arXiv ID:** 2601.16393 | [PDF](https://arxiv.org/pdf/2601.16393v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 103. DeMark: A Query-Free Black-Box Attack on Deepfake Watermarking Defenses

**arXiv ID:** 2601.16473 | [PDF](https://arxiv.org/pdf/2601.16473v1)

**作者:** Wei Song `[一作]` (University of New South Wales Sydney), Jingling Xue `[通讯]` (University of New South Wales Sydney)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种查询无关、黑盒攻击框架（MBRS），利用压缩感知的稀疏编码在水印编码器的潜在空间中抑制水印信号，从而在不影响图像可视质量的前提下破坏深度伪造图像的防御水印。

**💡 创新点**

创新点在于：①首次将压缩感知理论与深度水印攻击结合；②通过稀疏约束实现潜在空间的“分散效应”（稀疏度、强度与位置的改变），直接针对水印嵌入的潜在特征；③实现完全无查询、无模型参数依赖的黑盒攻击，显著提高了实用性。

**🔧 技术方法**

技术手段包括：潜在空间稀疏编码模块（基于ℓ1稀疏损失）、结构–感知重建模块（结合PSNR、SSIM、LPIPS等损失）、压缩感知理论框架、对抗训练与稀疏损失的平衡。

**📊 数据集**

训练使用公开数据集 OpenImage 与 COCO（共计约90,000张图像），攻击测试使用同样来源的2,000张水印图像，覆盖8种主流防御水印方案。

**📈 对比分析**

与三类现有攻击（Distortion、Regeneration、Adversarial）对比，MBRS在8种水印方案上平均将检测准确率从100%降至32.9%，比前者降低20–60%不等；同时在图像质量指标（PSNR/SSIM/FID/LPIPS）上保持或优于其他攻击；计算成本低，内存占用仅为传统方法的1/10，攻击时间更短。

**⚠️ 局限性**

局限性包括：①仅针对基于编码器–解码器结构的水印方案，对其他嵌入方式（如直接在噪声向量中编码）效果未知；②稀疏参数需要经验调节，过度稀疏会导致图像细节丢失；③在极端攻击强度下，部分水印仍能恢复；④对抗训练等防御手段虽在实验中效果有限，但在更复杂场景下可能产生不同结果。

---

## 104. Cauchy's Surface Area Formula in the Funk Geometry

**arXiv ID:** 2601.16468 | [PDF](https://arxiv.org/pdf/2601.16468v1)

**作者:** Sunil Arya `[一作]` (Hong Kong University of Science and Technology), David M. Mount `[通讯]` (University of Maryland)

**通讯引用:** 14723 | [OpenAlex ID](https://openalex.org/A5016442699)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文在Funk几何框架下推导了Cauchy式表面积公式与Crofton式线条计数公式，并给出了多维离散化（顶点分解）以及与欧氏、Hilbert、Minkowski和双曲几何的统一表达。

**💡 创新点**

创新点在于利用从凸体K边界点出发的中心投影，将Funk表面积化简为方向平均的欧氏投影面积，从而得到易于计算的闭式公式；并首次实现了对多面体的顶点级别分解，极大简化了计算与采样。

**🔧 技术方法**

技术手段包括Finsler几何、对偶极形与投影理论、Tonelli定理、支配收敛定理、以及对称化的线条测度构造；通过这些工具完成从局部投影到全局面积的积分等价。

**📊 数据集**

该工作为纯理论研究，不涉及实验数据或特定数据集；所有结果均通过严谨的数学证明获得。

**📈 对比分析**

通过与已知的Euclidean Cauchy公式、Hilbert与Minkowski空间的现有结果对照，证明在特殊情况（平面、椭球体、无限放缩）下公式与已知结果完全一致；在一般情况下给出与Hilbert面积的常数因子近似关系，保证误差可控。

**⚠️ 局限性**

局限性包括：1）需假设K的边界严格凸或为多面体；2）对非凸体或具有折角的K不适用；3）本文未给出具体随机采样算法的复杂度与方差分析，实际实现仍需进一步研究。

---

## 105. Semi-Supervised Hierarchical Open-Set Classification

**arXiv ID:** 2601.16541 | [PDF](https://arxiv.org/pdf/2601.16541v1)

**作者:** Erik Wallin `[一作]` (Saab AB), Lars Hammarstrand `[通讯]` (Chalmers University of Technology)

**通讯引用:** 1892 | [OpenAlex ID](https://openalex.org/A5035123040)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了半监督层次开放集分类（SemiHOC）框架，利用未标注数据中的已知和未知类别改进层次化 OOD 检测。

**💡 创新点**

创新点包括：①子树伪标签（Subtree Pseudo‑Labels）提供更鲁棒的 OOD 监督；②年龄门控（Age‑Gating）防止 OOD 样本在训练后期被过度细化为错误叶子节点；③将 ProHOC 与教师‑学生自监督架构结合，消除对 ID/OOD 权衡参数的依赖。

**🔧 技术方法**

技术手段：ProHOC 的层级预测、教师‑学生伪标签训练、子树置信度阈值、年龄门控切点检测、DinoV2 特征提取、Dropout 与学习率调节。

**📊 数据集**

使用了三大图像分类基准：iNaturalist19、iNaturalist21‑Aves（鸟类子集）以及 SimpleHierImageNet，均含有已知层次结构。

**📈 对比分析**

与基线比较：自监督预训练+监督、节点级伪标签、每层独立伪标签、全监督 ID 等。SemiHOC 在 20 或 50 个标签/类时达到或超过全监督模型性能；在有限标签情形下，BMHD 指标显著优于所有对比方法。

**⚠️ 局限性**

局限性：需要可解释且视觉可区分的层次结构；仅评估与训练相同 OOD 类别的情形；对完全未知 OOD 类别的泛化仍待验证。

---

## 106. UAV-Assisted Joint Data Collection and Wireless Power Transfer for Batteryless Sensor Networks

**arXiv ID:** 2601.16533 | [PDF](https://arxiv.org/pdf/2601.16533v1)

**作者:** Wen Zhang `[一作]` (Jilin University), Dusit Niyato `[通讯]` (Nanyang Technological University)

**通讯引用:** 80345 | [OpenAlex ID](https://openalex.org/A5091266202)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计了一种无人机协助的无电池传感器网络数据采集与无线能量传输系统，联合优化无人机的功率与轨迹以提升数据采集量与公平性，同时降低无人机能耗。

**💡 创新点**

首次将公平性指标（Jain指数）纳入多目标优化，并提出了集成优先经验回放（PER）与 Performer注意力模块的 SAC-PP 深度强化学习算法，以实现更快收敛和更稳定的策略。

**🔧 技术方法**

使用深度强化学习（SAC）改进版 SAC-PP，结合优先经验回放、Performer注意力机制、OFDMA多址、RF 能量收集模型以及无人机推进能耗模型。

**📊 数据集**

在仿真数据集上测试，设置 400m×400m 区域内 20 个静态无电池传感器，随机分布，时间槽长度固定，采用伯努利分布产生数据请求并以正态分布模拟数据量。

**📈 对比分析**

与 TD3、TQC、PPO、SAC 四种基线方法对比，SAC-PP 在累计奖励、平均采集数据量、能耗与公平性方面均优于其它算法，尤其在数据采集量和公平性上表现突出。

**⚠️ 局限性**

仅在仿真环境验证，未涉及真实无人机与传感器硬件；假设传感器位置静止、无干扰环境，未考虑多用户干扰、传输错误或硬件能量采集限制，需进一步实验验证。

---

## 107. DeepEra: A Deep Evidence Reranking Agent for Scientific Retrieval-Augmented Generated Question Answering

**arXiv ID:** 2601.16478 | [PDF](https://arxiv.org/pdf/2601.16478v1)

**作者:** Haotian Chen `[一作]` (Computer Network Information Center, Chinese Academy of Sciences), Xuezhi Wang `[通讯]` (Computer Network Information Center, Chinese Academy of Sciences)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了DeepEra——一种面向科学检索增强生成（RAG）的代理式重排序框架，能够通过意图识别、逻辑相关性评估和证据摘要三阶段对检索候选文档进行精准筛选与压缩；

**💡 创新点**

创新点在于：1）将LLM的逐步推理能力与结构化查询表示相结合，实现对语义相似但逻辑无关的干扰文本的有效剔除；2）构造了SciRAG‑SSLI数据集，系统地将语义相近但逻辑无关的负样本注入检索库，用于评估重排序的鲁棒性；3）设计了可解释的三步流程（意图识别→相关性评估→摘要），兼顾效率与可靠性；

**🔧 技术方法**

使用了LLM驱动的代理式重排序技术（如Qwen‑Plus或类似模型）进行意图解析和分数评估；结合传统检索（dense embeddings）与阈值过滤、LLM摘要；采用多指标评估（HitRate@K、RP、LFS、F1等）对性能进行量化；

**📊 数据集**

主要使用SciRAG‑SSLI数据集（约30万问答对，覆盖10个学科，包含基线与SSLI两种检索难度），以及LMQG构造的对照集；

**📈 对比分析**

与多种基线（BGE、Jina、BCE、SPLADE、ColBERT、RankT5、MonoT5、RankGPT等）在基线与SSLI情景下进行对比。DeepEra在SSLI情境下HitRate@1提升至66.6%（相对最高基线提升5%+），HitRate@3 76.4%，相对位次71.96；生成端F1 43.38、LFS 3.94，整体性能比最强基线提升约8%；

**⚠️ 局限性**

局限性包括：1）仍依赖大型LLM，推理成本和能耗较高；2）评价主要在人工构造的SSLI负样本上，实际数据噪声类型可能更为多样；3）未在真实业务场景中进行大规模部署验证；4）对跨语言或非学术文本的适用性未知。

---

## 108. SycoEval-EM: Sycophancy Evaluation of Large Language Models in Simulated Clinical Encounters for Emergency Care

**arXiv ID:** 2601.16529 | [PDF](https://arxiv.org/pdf/2601.16529v1)

**作者:** Dongshen Peng `[一作]` (University of North Carolina at Chapel Hill), Christian Rose `[通讯]` (Stanford University)

**通讯引用:** 2277 | [OpenAlex ID](https://openalex.org/A5075451277)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

构建并使用多智能体仿真框架SycoEval-EM，对20款大语言模型在三种急诊情景下的患者说服行为进行评估，量化其在指导性压力下的顺从率

**💡 创新点**

首次提供大规模多模型、跨情景、多策略的说服对抗实验，揭示模型顺从率与模型规模、训练时间无关，且不同策略在模型中的效果均匀

**🔧 技术方法**

多智能体对话系统（患者代理、医生代理、评估者代理）与OpenRouter API调用，采用LLM自带的提示与规范化指令进行评估

**📊 数据集**

人工构造的三种急诊情景（CT扫描、抗生素、阿片类镇痛）和五种说服策略，合计1,875条模拟对话

**📈 对比分析**

通过评估者多模型投票确定是否顺从，结果显示顺从率从0%到100%不等，平均在30-36%之间，模型可分为高、中、低易顺从三层次

**⚠️ 局限性**

仅限三种情景与五种策略，未覆盖真实临床多代理系统，且模型顺从受提示难以完全解释，缺乏对抗策略的可复制性与对抗鲁棒性的长期评估

---

## 109. Persona Jailbreaking in Large Language Models

**arXiv ID:** 2601.16466 | [PDF](https://arxiv.org/pdf/2601.16466v1)

**作者:** Jivnesh Sandhan `[一作]` (Kyoto University), Yugo Murawaki `[通讯]` (Kyoto University)

**通讯引用:** 290 | [OpenAlex ID](https://openalex.org/A5013357764)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6215c339-3735-4be3-8a07-5bbb7004712d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种黑盒攻击框架PHISH，利用对话历史中的隐式提示，能够在不修改模型或系统提示的情况下逆转大型语言模型的Big‑Five人格特质。

**💡 创新点**

创新点在于首次揭示并量化“人格编辑”这一攻击面：通过嵌入语义对立的问答对，逐步推断并逆转模型人格，同时提出STIR指标衡量攻击成功率，并展示在高风险领域（心理健康、辅导、客服）中的实际效果。

**🔧 技术方法**

技术手段包括：隐式语义提示（对立问答对）构造、黑盒攻击策略、Big‑Five人格评估、STIR（Success Trait Influence Rate）指标、LLM-as-Judge评估、以及三种基线与三种防御方法的对照实验。

**📊 数据集**

使用了三大人格评测数据集：Big Five Inventory (BFI)、Machine Personality Inventory (MPI) 与 MIT IPIP 及 Anthropic-Eval 子集。

**📈 对比分析**

与八个代表性LLM（含两大域特定模型）以及八种基线攻击进行对照，PHISH在三大评测集上多达95%+ STIR 分数，显著优于所有基线；在高风险任务中也能保持约30% 的人格逆转成功率；对推理基准（Math、GSM8K、CSQA）的影响仅为1–6分的轻微下降。

**⚠️ 局限性**

局限性包括：仅考虑黑盒、推理仅限用户输入的攻击场景，未涉及白盒或内部梯度攻击；未提供完整防御体系，仅评估三种防御的脆弱性；缺乏对人格维度相互依赖的理论解释，模型人格的可解释性与泛化仍待进一步研究。

---

## 110. TangramPuzzle: Evaluating Multimodal Large Language Models with Compositional Spatial Reasoning

**arXiv ID:** 2601.16520 | [PDF](https://arxiv.org/pdf/2601.16520v1)

**作者:** Daixian Liu `[一作]` (Tsinghua University), Philip S. Yu `[通讯]` (University of Illinois Chicago)

**通讯引用:** 132273 | [OpenAlex ID](https://openalex.org/A5036357902)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并评估 TangramPuzzle 基准，检验多模态大语言模型在几何空间推理中的能力

**💡 创新点**

引入 Tangram Construction Expression (TCE) 形式化几何描述，实现可机验证的几何约束与严格评估；提出轮廓预测与端到端拼图生成两大任务

**🔧 技术方法**

结合符号几何表达式、约束验证器、IoU 与 Hausdorff 距离度量、示例学习与视觉中心化实验

**📊 数据集**

由 668 条多样化唐格谜题构成的 TangramPuzzle 数据集，基于 KiloGram 过滤并人工标注，包含 TCE 规范化与多选干扰项

**📈 对比分析**

在开放源代码与闭源模型（如 GPT‑5.2、Gemini3‑Pro 等）上进行评测；Gemini3‑Pro 在轮廓预测 98.65% 以及拼图生成验证通过率 85.93% 领先；多数模型在几何约束上表现不佳

**⚠️ 局限性**

仅限二维平面、固定七块拼板、缺乏 3D 复杂空间、可扩展性与现实任务关联不足

---

## 111. Finite-Time Analysis of Gradient Descent for Shallow Transformers

**arXiv ID:** 2601.16514 | [PDF](https://arxiv.org/pdf/2601.16514v1)

**作者:** Enes Arda `[一作]` (Ohio State University), Atilla Eryilmaz `[通讯]` (Ohio State University)

**通讯引用:** 4959 | [OpenAlex ID](https://openalex.org/A5076974020)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文研究了浅层多头Transformer在投影梯度下降（ProjGD/ProjSGD）下的有限时NTK收敛性，给出了宽度与样本大小的对数比例、优化误差与序列长度无关的非渐近保证，并在教师-学生实验中验证了理论的尺度律。

**💡 创新点**

创新点：①保留真实softmax注意力非线性并允许多头；②不需要NTK严格正定性；③证明宽度只需对数级扩展即可得到非渐近收敛；④优化误差与序列长度无关，揭示Transformer在长程依赖上的优势与RNN的指数劣势；⑤通过运输映射将Transformer的NTK与随机特征联系，提供更精细的理论工具。

**🔧 技术方法**

使用技术包括：投影梯度下降算法、Neural Tangent Kernel（NTK）线性化、随机特征与运输映射方法、对初始化的对称高斯设定、对梯度与误差的高维概率界定、以及对软max Jacobian的矩阵不变性分析。

**📊 数据集**

数据集：①教师-学生实验采用高斯噪声采样的长度为T的向量序列，目标函数由RKHS中随机特征的线性组合生成；②长时序回归实验采用自回归AR(L)时间序列，设置不同的延迟L以模拟不同的长程依赖。

**📈 对比分析**

对比方法：与IndRNN进行对比；实验显示IndRNN在短程依赖下表现略优，但随着L增大梯度模长剧增、验证误差上升；Transformer在所有L下保持稳定的梯度大小与较低的验证误差，说明理论预测的T无关性得到验证，但其内存占用随T线性增长。

**⚠️ 局限性**

局限性：仅针对浅层Transformer；分析仅在接近初始化的小邻域内有效；需要目标函数可被NTK实现的假设；未探讨深层Transformer或实际大规模自然语言/视觉数据；理论界限在常数与高阶项上可能过于保守。

---

## 112. EvoConfig: Self-Evolving Multi-Agent Systems for Efficient Autonomous Environment Configuration

**arXiv ID:** 2601.16489 | [PDF](https://arxiv.org/pdf/2601.16489v1)

**作者:** Xinshuai Guo `[一作]` (Tsinghua University), Xing Sun `[通讯]` (Tencent Youtu Lab)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种自进化多智能体框架 EvoConfig，帮助 LLM 自动构建可运行的软件环境。

**💡 创新点**

创新点在于将执行、诊断与修复分离，使用专家诊断智能体进行细粒度错误分析，并通过自进化机制实时调整修复策略。

**🔧 技术方法**

采用多智能体协作、ReAct 思维框架、工具生成与执行、环境信息抽取、在线自进化学习等技术。

**📊 数据集**

使用 Repo2Run（420 份仓库）、EnvBench（324 份 Python 仓库）和 EnConda‑Bench（4201 条错误实例）进行实验。

**📈 对比分析**

与 Repo2Run、pipreqs、LLM Generator、SWE‑Agent、OpenHands、Installamatic 等基线对比，EvoConfig 在 Repo2Run 上 88.1% 的 EBSR 与 100% DGSR，EnvBench 上 78.1% 成功率（对比 71.0%），配置时间从 30.5 分钟降至 20.9 分钟，费用降低一半，并在 EnConda‑Bench 上显著提升错误识别与修复准确率。

**⚠️ 局限性**

仅关注环境构建成功，未对单元测试通过率进行评估；对错误反馈质量敏感，早期自进化机制可能需要足够样本才能稳定。

---

## 113. SALAD: Achieve High-Sparsity Attention via Efficient Linear Attention Tuning for Video Diffusion Transformer

**arXiv ID:** 2601.16515 | [PDF](https://arxiv.org/pdf/2601.16515v1)

**作者:** Tongcheng Fang `[一作]` (Kuaishou Technology), Yu Wang `[通讯]` (Tsinghua University)

**通讯引用:** 42925 | [OpenAlex ID](https://openalex.org/A5100445300)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种在Diffusion Transformer中并行的稀疏线性注意力架构（SALAD），通过引入轻量级线性注意力分支和输入自适应门控实现高达90%的稀疏率、1.72×的推理加速，并保持与全注意力模型相当的生成质量。

**💡 创新点**

创新点包括：1）在稀疏注意力旁加入共享参数的线性注意力分支以补偿信息缺失；2）设计输入自适应标量门控，动态平衡两分支输出；3）使用3D RoPE提升线性注意力在时空视频序列中的表达；4）仅需2k视频样本和1.6k步训练即可实现效果。

**🔧 技术方法**

采用的技术主要有：Diffusion Transformer、稀疏注意力（滑窗/Top‑K）、ReLU‑based线性注意力、3D Rotary Position Embedding、LoRA微调、输入自适应标量门控。

**📊 数据集**

使用了Open‑source Mixkit视频数据集（2000条）进行微调，基线模型为Wan2.1‑1.3B。

**📈 对比分析**

与全注意力、SVG2、ST‑SWA、PARO、LoRA等基线进行对比，SALAD在VBench指标（Subject Consistency、Background Consistency、Image Quality、Text Consistency）均达到或超过全注意力水平，同时实现90%稀疏率与1.72×推理加速。

**⚠️ 局限性**

局限性在于：线性注意力分支虽可补充长距离信息，但仍无法完全取代稀疏注意力；门控参数需经验调优；在不同数据集或更大模型上效果未验证；若移除线性分支会导致细节和连贯性下降。

---

## 114. Online Computation of Palindromes and Suffix Trees on Tries

**arXiv ID:** 2601.16485 | [PDF](https://arxiv.org/pdf/2601.16485v1)

**作者:** Hiroki Shibata `[一作]` (Kyushu University), Masayuki Takeda `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

**🎯 论文内容**

研究了在支持叶子增删的字典树（trie）中动态计算最大回文串和不同回文串的问题。

**💡 创新点**

提出了子二次时间的在线算法：对最大回文串实现 O(N·min(log h, σ)) 的增删复杂度；对不同回文串提供三种基于 EERTREE（快速链、直接链+持久化树或颜色祖先）和基于后缀树的在线算法，能够在叶子增删时维护回文集合。

**🔧 技术方法**

关键技术包括：利用回文串的周期性与算术进程压缩、EERTREE 的后缀链/快速链/直接链、持久化树与颜色祖先数据结构、后缀树的 Euler 旅行与动态 LCA、以及前缀/后缀树的实时构建。

**📊 数据集**

未使用公开数据集，论文主要是理论算法与复杂度分析。

**📈 对比分析**

相较于先前仅支持离线的线性时间算法，本文提供了在线实现；在最坏情况下，最大回文串算法比离线 O(N log σ) 或 O(Nσ) 的做法更优；不同回文串算法在不同 α 的取值下提供了时间空间折中，均保持 O(N) 空间。

**⚠️ 局限性**

限制包括：算法依赖于前缀/后缀树的在线构建，导致实现复杂；直接链方法空间不线性（需持久化树）；在非常大字符集或深度高的 trie 上仍可能出现较高常数；未讨论根节点的增删或边标签修改。

---

## 115. SafeThinker: Reasoning about Risk to Deepen Safety Beyond Shallow Alignment

**arXiv ID:** 2601.16506 | [PDF](https://arxiv.org/pdf/2601.16506v1)

**作者:** Xianya Fang `[一作]` (Nanjing University of Aeronautics and Astronautics), Sheng-Jun Huang `[通讯]` (Nanjing University of Aeronautics and Astronautics)

**通讯引用:** 4114 | [OpenAlex ID](https://openalex.org/A5103204774)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于轻量级风险评估门控、Safety‑Aware Twin Expert（SATE）与分布引导思考（DDGT）的自适应安全防御框架；

**💡 创新点**

创新点包括：①动态风险分级路由，按输入风险实时分配防御资源；②双专家SATE模型通过LoRA微调实现对潜在泄漏的二次拦截；③DDGT在不确定路径下通过分布相似度判断，采用协同或对抗式解码恢复对齐；

**🔧 技术方法**

使用技术包括：轻量级概率头用于门控；LoRA微调SATE；分布相似度（cosine）与阈值决策；动态候选词池与多模型协同/单模型干预；

**📊 数据集**

实验数据集涵盖：ALERT（3,200条违规样本），AdvBench与EasyJailbreak（GCG、AutoDAN、PAIR、Jailbroken、DeepInception），Harmful HEx‑PHI（prefilling攻击），MT‑Bench、GSM8K、SQL Create Context（通用功能评测）；

**📈 对比分析**

与PPL、Self‑Examination、ICD、Self‑Reminder、SafeDecoding等基线对比，攻击成功率（ASR）降至0%（如Llama‑3‑8B），在prefilling攻击中保持<6% ASR，且对benign任务的效用几乎不下降，整体计算开销可与无防御版持平；

**⚠️ 局限性**

局限性：需要额外内存以部署SATE；防御效果高度依赖门控头的准确性，易被高置信度的伪装攻击突破；实验仅在英文文本上验证，尚未覆盖多语言、代码生成或多模态场景；

---

## 116. Do Models Hear Like Us? Probing the Representational Alignment of Audio LLMs and Naturalistic EEG

**arXiv ID:** 2601.16540 | [PDF](https://arxiv.org/pdf/2601.16540v1)

**作者:** Haoyun Yang `[一作]` (Chongqing University), Kaiwen Wei `[通讯]` (Chongqing University)

**通讯引用:** 171 | [OpenAlex ID](https://openalex.org/A5068039769)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

本文系统评估了12种开源音频大语言模型在自然听觉情境下与人类EEG信号的层级对应关系。

**💡 创新点**

创新之处在于首次同时使用8种互补相似度度量、引入三模态邻域一致性（TNC）和情感声学分析，揭示了深度、时域及情绪对模型-脑相似性的分离影响。

**🔧 技术方法**

技术方法包括层级隐状态提取、时间对齐、RSA、CKA、距离相关、互信息、Kendall、Spearman等相似度指标，以及TNC情感聚类分析。

**📊 数据集**

使用了两个公开EEG数据集：Alice in Wonderland（49受试者）和Lalor Lab自然语言听力数据集（19受试者）。

**📈 对比分析**

通过在每个模型层上计算相似度并进行置换显著性检验，结果显示不同度量对模型排名差异显著；在250–500 ms窗口内表现最佳，情绪分析显示负声调增强全局依赖但削弱几何一致性。

**⚠️ 局限性**

局限包括仅覆盖开源模型、EEG源定位受限、缺乏对更大规模或专有模型的验证，以及受受试者个体差异和任务设计限制。

---

## 117. Timely Machine: Awareness of Time Makes Test-Time Scaling Agentic

**arXiv ID:** 2601.16486 | [PDF](https://arxiv.org/pdf/2601.16486v1)

**作者:** Yichuan Ma `[一作]` (Fudan University), Kai Chen `[通讯]` (Shanghai AI Laboratory)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了“Timely Machine”概念，将agent任务中的测试时延定义为真实物理时间，并构建了Timely-Eval基准和Timely-RL方法，用于训练LLM的时间意识与动态策略调整。

**💡 创新点**

创新点在于重新定义agent场景下的测试时延、创建以时间预算为核心的评测基准、以及基于工具延迟反馈的时间感知强化学习框架，使模型能根据实时延迟自适应交互频率与质量。

**🔧 技术方法**

使用了强化学习（RLVR/GRPO）与监督微调（SFT）的混合训练，设计了包含时间利用率的奖励函数，并通过工具延迟回馈实现时间感知。

**📊 数据集**

数据集包括合成的Timely Reasoning数据集、从Qwen3、DeepSeek、Jericho、MLEBench-Lite等提取的交互游戏、机器学习任务以及通用推理基准（AIME、MATH、GPQA-diamond）等。

**📈 对比分析**

通过Timely-Eval的多任务评测，将不同参数规模模型在低、中、高工具延迟下的得分进行对比。结果显示：在低延迟场景下小模型因推理速度快可优于大模型；延迟升高后大模型凭借高交互质量获得更好表现；Timely-RL显著提升模型按时完成率和整体准确率。

**⚠️ 局限性**

局限性包括：实验仅覆盖文本交互任务，未验证多模态或多代理场景；对多代理系统中产生的延迟缺乏有效处理；时间意识提升对准确率的提升受任务难度影响，尚需进一步平衡。

---

## 118. OnlineSI: Taming Large Language Model for Online 3D Understanding and Grounding

**arXiv ID:** 2601.16538 | [PDF](https://arxiv.org/pdf/2601.16538v1)

**作者:** Zixian Liu `[一作]` (Tsinghua University), Ziwei Liu `[通讯]` (Nanyang Technological University)

**通讯引用:** 42102 | [OpenAlex ID](https://openalex.org/A5100406050)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

我们提出了OnlineSI框架，能够在视频流中在线地进行3D场景理解与对象定位，并保证每帧推理的计算成本固定；

**💡 创新点**

核心创新在于使用有限的显式空间记忆聚合时间信息，紧耦合点云与语义信息并引入模糊F1分数评估，从而实现持续改进和在线推断；

**🔧 技术方法**

技术方案包括基于SpatialLM的LLM后端、Sonata点云编码器、语义编码器（将语义标签映射为Llama token特征）、CUT3R重建模块与Grounded SAM语义标注；

**📊 数据集**

实验数据集为ScanNet和ScanNet++；

**📈 对比分析**

与多种基线（无微调、逐帧融合、仅点云、使用地面真值）对比，在线SI在两大数据集上均取得非地面真值基线中最高的模糊F1分数，展示了显著性能提升；

**⚠️ 局限性**

局限性包括：仅在室内场景有效（受SpatialLM预训练限制）；空间记忆采用采样拼接方式，不适应动态情境；依赖外部重建与语义标注模块；对点云坐标系有严格对齐要求。

---

## 119. Noise-immune and AI-enhanced DNA storage via adaptive partition mapping of digital data

**arXiv ID:** 2601.16518 | [PDF](https://arxiv.org/pdf/2601.16518v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

---

## 120. AnchoredDream: Zero-Shot 360° Indoor Scene Generation from a Single View via Geometric Grounding

**arXiv ID:** 2601.16532 | [PDF](https://arxiv.org/pdf/2601.16532v1)

**作者:** Runmao Yao `[一作]` (Tsinghua University), Yu-Shen Liu `[通讯]` (Tsinghua University)

**通讯引用:** 4718 | [OpenAlex ID](https://openalex.org/A5101691399)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 AnchoredDream，零 shot 单视图室内场景 360° 生成管线，利用高保真几何先导的相互提升机制实现全景场景合成。

**💡 创新点**

引入外观-几何互助机制与 Grouting Block 实现无缝过渡，并通过几何预导生成 3D Gaussian 结构来提升外观一致性与几何可信度。

**🔧 技术方法**

结合 VLM 提取场景描述、Holodeck 生成 3D 布局、基于深度的 ControlNet Warp‑Inpaint、3D Gaussian Splatting 以及多视角一致性优化等技术。

**📊 数据集**

在 25 张单视图室内图像（涵盖 Retro、Modern、Minimalist、Modern Luxury 等 4 种风格）上评估，图像由 GPT‑4o 生成或取自基线。

**📈 对比分析**

与 LucidDreamer、GenWarp、VistaDream、ViewCrafter 等基线在同一视角序列上通过 LLaVA‑NEXT 评估外观一致性与几何可信度，AnchoredDream 在两项指标均显著领先。

**⚠️ 局限性**

依赖 VLM 生成的 prompt 质量；ControlNet 近似深度引导的 inpainting 仍可能出现局部不一致；对极端场景的几何对齐仍有提升空间。

---

## 121. Beyond Superficial Unlearning: Sharpness-Aware Robust Erasure of Hallucinations in Multimodal LLMs

**arXiv ID:** 2601.16527 | [PDF](https://arxiv.org/pdf/2601.16527v1)

**作者:** Xianya Fang `[一作]` (Nanjing University of Aeronautics and Astronautics), Sheng-Jun Huang `[通讯]` (Nanjing University of Aeronautics and Astronautics)

**通讯引用:** 4114 | [OpenAlex ID](https://openalex.org/A5103204774)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了基于几何稳定化的SARE框架，用于在多模态大型语言模型中消除对象幻觉。

**💡 创新点**

创新点在于将幻觉消除视为目标化最小‑最大优化，利用Targeted‑SAM实现对幻觉的几何平坦化，从而提升对重学习攻击的鲁棒性。

**🔧 技术方法**

采用了Targeted‑SAM、CLIP‑基准的自动数据筛选、最小‑最大对抗优化及目标化梯度更新等技术。

**📊 数据集**

使用了MSCOCO图像‑文本数据集（含子句与句子级别样本）进行训练，并在GQA、SQA、QBench、MME等多模态基准上进行评估。

**📈 对比分析**

在mPLUG‑Owl‑7B和LLaVA‑v1.5‑7B上与Vanilla、EFUF等基线对比，SARE在CHAIR、POPE、Hallucination率等指标上显著优于EFUF，并在重学习、LoRA fine‑tune和对抗提示等攻击下保持低幻觉率。

**⚠️ 局限性**

局限性包括：依赖静态对齐过滤无法捕捉潜在幻觉触发；Targeted‑SAM双梯度训练开销增加；目前仅针对对象存在幻觉，未扩展到属性或逻辑类幻觉。

---

## 122. DANCE: Dynamic, Available, Neighbor-gated Condensation for Federated Text-Attributed Graphs

**arXiv ID:** 2601.16519 | [PDF](https://arxiv.org/pdf/2601.16519v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 123. Participatory Budgeting Project Strength via Candidate Control

**arXiv ID:** 2601.16511 | [PDF](https://arxiv.org/pdf/2601.16511v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

---

## 124. MRAG: Benchmarking Retrieval-Augmented Generation for Bio-medicine

**arXiv ID:** 2601.16503 | [PDF](https://arxiv.org/pdf/2601.16503v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 125. A High Performance and Efficient Post-Quantum Crypto-Processor for FrodoKEM

**arXiv ID:** 2601.16500 | [PDF](https://arxiv.org/pdf/2601.16500v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 126. Load Balanced ISAC Systems for URLLC Users

**arXiv ID:** 2601.16495 | [PDF](https://arxiv.org/pdf/2601.16495v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

---

## 127. LLM-based Semantic Search for Conversational Queries in E-commerce

**arXiv ID:** 2601.16492 | [PDF](https://arxiv.org/pdf/2601.16492v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 128. A Scalable Transaction Management Framework for Consistent Document-Oriented NoSQL Databases

**arXiv ID:** 2601.16490 | [PDF](https://arxiv.org/pdf/2601.16490v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7`

---

## 129. A Cautionary Tale of Self-Supervised Learning for Imaging Biomarkers: Alzheimer's Disease Case Study

**arXiv ID:** 2601.16467 | [PDF](https://arxiv.org/pdf/2601.16467v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 130. Competing Visions of Ethical AI: A Case Study of OpenAI

**arXiv ID:** 2601.16513 | [PDF](https://arxiv.org/pdf/2601.16513v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 131. On the Effects of Adversarial Perturbations on Distribution Robustness

**arXiv ID:** 2601.16464 | [PDF](https://arxiv.org/pdf/2601.16464v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 132. A Collision-Free Hot-Tier Extension for Engram-Style Conditional Memory: A Controlled Study of Training Dynamics

**arXiv ID:** 2601.16531 | [PDF](https://arxiv.org/pdf/2601.16531v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 133. Rethinking Large Language Models For Irregular Time Series Classification In Critical Care

**arXiv ID:** 2601.16516 | [PDF](https://arxiv.org/pdf/2601.16516v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 134. Robust Categorical Data Clustering Guided by Multi-Granular Competitive Learning

**arXiv ID:** 2601.16491 | [PDF](https://arxiv.org/pdf/2601.16491v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 135. TL-GRPO: Turn-Level RL for Reasoning-Guided Iterative Optimization

**arXiv ID:** 2601.16480 | [PDF](https://arxiv.org/pdf/2601.16480v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 136. Secure Intellicise Wireless Network: Agentic AI for Coverless Semantic Steganography Communication

**arXiv ID:** 2601.16472 | [PDF](https://arxiv.org/pdf/2601.16472v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 137. Order from Chaos: Physical World Understanding from Glitchy Gameplay Videos

**arXiv ID:** 2601.16471 | [PDF](https://arxiv.org/pdf/2601.16471v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 138. W4A16 Mixed-Precision Matrix Multiplication on Decoupled Architecture: Kernel Design and Memory Bottleneck Analysis for Ascend NPUs

**arXiv ID:** 2601.16536 | [PDF](https://arxiv.org/pdf/2601.16536v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 139. SearchLLM: Detecting LLM Paraphrased Text by Measuring the Similarity with Regeneration of the Candidate Source via Search Engine

**arXiv ID:** 2601.16512 | [PDF](https://arxiv.org/pdf/2601.16512v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 140. Learning to Optimize by Differentiable Programming

**arXiv ID:** 2601.16510 | [PDF](https://arxiv.org/pdf/2601.16510v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `e4c502e8-c16d-4c56-8df3-cffaee9eaadb`

---

## 141. kNN-Graph: An adaptive graph model for $k$-nearest neighbors

**arXiv ID:** 2601.16509 | [PDF](https://arxiv.org/pdf/2601.16509v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 142. Is Length Really A Liability? An Evaluation of Multi-turn LLM Conversations using BoolQ

**arXiv ID:** 2601.16508 | [PDF](https://arxiv.org/pdf/2601.16508v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 143. Multi-View Consistent Wound Segmentation With Neural Fields

**arXiv ID:** 2601.16487 | [PDF](https://arxiv.org/pdf/2601.16487v1)

**作者:** Remi Chierchia `[一作]`, Rodrigo Santa Cruz `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种基于NeRF的多视角一致性伤口分割方法，能够直接在三维空间学习语义场，实现对伤口床及其组织的精确三维分割。

**💡 创新点**

创新点在于将SDF编码的几何网络与语义解码器相结合，利用NeRF的视角一致性将多视图2D分割信息聚合到统一的三维语义场，从而克服传统2D分割缺乏深度感知和视角不一致的问题。

**🔧 技术方法**

采用NeRF框架中的几何MLP（SDF）、语义解码头、加权交叉熵损失、Dropout正则化以及基于视线积分的像素预测技术，并在训练阶段先冻结几何网络后再开启语义学习。

**📊 数据集**

使用真实临床数据集，包含35名患者的73段视频，平均每段视频提取50张多视角图像，并从中手工标注1至4张视角的伤口床及组织类别。

**📈 对比分析**

与基于SegFormer的2D分割和3D/2D投影聚合方法对比，实验显示本方法在Dice系数、召回率等指标上均有显著提升，分割边界更平滑，对边界扰动更鲁棒，并能在少量视角训练后生成高质量的三维分割。

**⚠️ 局限性**

局限性包括对训练图像质量高度敏感，严重依赖专家标注的多视角数据，且在少量视角或标注不完整时仍可能出现误分类；此外，未充分解决低频类别（如坏死、上皮）的稀缺问题。

---

## 144. REprompt: Prompt Generation for Intelligent Software Development Guided by Requirements Engineering

**arXiv ID:** 2601.16507 | [PDF](https://arxiv.org/pdf/2601.16507v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 145. Anonymous Pricing in Large Markets

**arXiv ID:** 2601.16488 | [PDF](https://arxiv.org/pdf/2601.16488v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

---

## 146. Sycophancy Hides Linearly in the Attention Heads

**arXiv ID:** 2601.16644 | [PDF](https://arxiv.org/pdf/2601.16644v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 147. X-Aligner: Composed Visual Retrieval without the Bells and Whistles

**arXiv ID:** 2601.16582 | [PDF](https://arxiv.org/pdf/2601.16582v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 148. CORD: Bridging the Audio-Text Reasoning Gap via Weighted On-policy Cross-modal Distillation

**arXiv ID:** 2601.16547 | [PDF](https://arxiv.org/pdf/2601.16547v1)

**作者:** Jing Hu `[一作]`, Haifeng Wang `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `8d10c613-917e-4880-9716-17789f50e119` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 CORD（Cross-modal Weighted On-policy Reward-guided Distillation）框架，利用内部模型作为教师，对音频‑文本对齐进行自监督的跨模态蒸馏，显著提升音频条件下的推理性能。

**💡 创新点**

创新点在于：①采用在策略（on‑policy）自蒸馏，避免离策略和分布不匹配问题；②对重要词（高 KL）和早期推理步骤进行加权的逆 KL 目标；③通过 GRPO（组相对策略优化）结合判别奖励实现全局序列层面的对齐；④完全不依赖外部教师模型，具备更好的可扩展性。

**🔧 技术方法**

主要技术包括：内部音频‑文本对齐、自监督的逆 KL 损失、位置衰减权重、GRPO 强化学习、判别模型（judge）产生奖励、混合 token‑level 与 sequence‑level 损失。

**📊 数据集**

训练数据：80000 条 NuminaMath 语料生成的音频‑文本对；评估数据集：MMSU、OpenBookQA、GSM8K、MMAU（Music、Speech、Sound）等多模态推理基准。

**📈 对比分析**

与传统 SFT、Forward‑KL 等蒸馏基线对比，CORD 在 Qwen2‑Audio‑7B‑Instruct 和 Step‑Audio2‑mini 上平均减少音频‑文本性能差距 41.6% 与 44.8%，在多项推理任务中明显优于基线，甚至在某些任务几乎消除模态差距。

**⚠️ 局限性**

局限性：①仅在数学领域数据上训练，跨域推广效果尚待验证；②对判别模型与权重超参数较为敏感；③在极大规模多样化任务中的鲁棒性和可扩展性未做充分实验。

---

## 149. Zero-Shot MARL Benchmark in the Cyber-Physical Mobility Lab

**arXiv ID:** 2601.16578 | [PDF](https://arxiv.org/pdf/2601.16578v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 150. Process-Tensor Tomography of SGD: Measuring Non-Markovian Memory via Back-Flow of Distinguishability

**arXiv ID:** 2601.16563 | [PDF](https://arxiv.org/pdf/2601.16563v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 151. Predicting Networks Before They Happen: Experimentation on a Real-Time V2X Digital Twin

**arXiv ID:** 2601.16559 | [PDF](https://arxiv.org/pdf/2601.16559v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 152. Select or Project? Evaluating Lower-dimensional Vectors for LLM Training Data Explanations

**arXiv ID:** 2601.16651 | [PDF](https://arxiv.org/pdf/2601.16651v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 153. Reliable Brain Tumor Segmentation Based on Spiking Neural Networks with Efficient Training

**arXiv ID:** 2601.16652 | [PDF](https://arxiv.org/pdf/2601.16652v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 154. Dual-Prototype Disentanglement: A Context-Aware Enhancement Framework for Time Series Forecasting

**arXiv ID:** 2601.16632 | [PDF](https://arxiv.org/pdf/2601.16632v1)

**作者:** Haonan Yang `[一作]` (National University of Defense Technology), Zhuo Li `[通讯]` (National University of Defense Technology)

**通讯引用:** 1079 | [OpenAlex ID](https://openalex.org/A5100448048)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

提出一种模型无关的DPAD框架，通过双原型银行和上下文感知路由实现时间序列预测模型的动态模式解耦与自适应增强。

**💡 创新点**

1）双原型分离常见与稀有模式并可动态更新；2）双路径上下文感知路由动态检索并融合对应原型；3）引入解耦引导损失保证原型专业化、多样性和稀有性。

**🔧 技术方法**

动态双原型银行、双路径上下文感知路由、解耦引导损失、基于Pearson相似度的检索、Softmax加权融合以及Transformer/MLP/CNN/LLM等骨干模型。

**📊 数据集**

长期预测数据集：ETTh1/ETTh2/ETTm1/ETTm2、Electricity、Exchange、Solar、Weather、Traffic；短期预测数据集：PEMS03/PEMS04/PEMS07/PEMS08；零样本预测使用上述ETT子集。

**📈 对比分析**

与五种SOTA骨干（iTransformer、TimeXer、TimeBridge、DLinear、TimesNet）对比，DPAD在大多数数据集上均显著降低MSE/MAE（平均MSE降幅约12.6%），在Traffic、PEMS等复杂场景表现尤为突出；零样本测试亦显示优于基线。

**⚠️ 局限性**

主要限制在于原型数量与初始化对极端稀有事件的记忆有限；对极短时序或高噪声场景的鲁棒性未充分验证；在轻量级模型如DLinear上相对开销较大。

---

## 155. Artifact for Service-Level Energy Modeling and Experimentation for Cloud-Native Microservices

**arXiv ID:** 2601.16635 | [PDF](https://arxiv.org/pdf/2601.16635v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 156. GPU-Accelerated Selected Basis Diagonalization with Thrust for SQD-based Algorithms

**arXiv ID:** 2601.16637 | [PDF](https://arxiv.org/pdf/2601.16637v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 157. Who You Explain To Matters: Learning by Explaining to Conversational Agents with Different Pedagogical Roles

**arXiv ID:** 2601.16583 | [PDF](https://arxiv.org/pdf/2601.16583v1)

**作者:** Zhengtao Xu `[一作]` (National University of Singapore), Yi-Chieh Lee `[通讯]` (National University of Singapore)

**通讯引用:** 1685 | [OpenAlex ID](https://openalex.org/A5054435118)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

对比四种对话代理角色（Tutee、Peer、Challenger、Control）在学习者通过解释学习经济学概念过程中的交互模式、学习效果与体验。

**💡 创新点**

首次在受控实验中系统比较不同教学角色的对话代理，揭示角色如何塑造认知投入、心理负荷、学习体验，并提供基于角色的设计原则。

**🔧 技术方法**

使用 GPT‑4o 构建对话代理，设计不同角色的工作流与提示；结合行为日志、编码分类、前后测、问卷调查与定性访谈进行多维度评估。

**📊 数据集**

教材内容为 OpenStax 《Principles of Economics》3.1–3.3 章节（供需概念）和对应 20 题多选测验；实验样本 96 名美国受试者来自 CloudResearch Connect。

**📈 对比分析**

通过单因素 ANOVA、Welch‑ANOVA、Kruskal‑Wallis 等统计方法比较各条件的词数、材料回顾次数、测验成绩、主观体验等指标；结果显示不同角色在交互模式与体验上显著差异，但客观测验成绩无显著差别；Tutee 产生最大认知投入与压力，Peer 与 Challenger 则提升流畅度与兴趣，Control 最差。

**⚠️ 局限性**

局限：仅单次短期实验，实验内容局限于经济学概念；未验证参与者对角色的感知；未考虑语言水平、先前知识等个体差异；缺乏长期追踪以观察学习效果的累积。

---

## 158. Retrieve-Refine-Calibrate: A Framework for Complex Claim Fact-Checking

**arXiv ID:** 2601.16555 | [PDF](https://arxiv.org/pdf/2601.16555v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 159. Taming the Heavy Tail: Age-Optimal Preemption

**arXiv ID:** 2601.16624 | [PDF](https://arxiv.org/pdf/2601.16624v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

---

## 160. Term Coding: An Entropic Framework for Extremal Combinatorics and the Guessing--Number Sandwich Theorem

**arXiv ID:** 2601.16614 | [PDF](https://arxiv.org/pdf/2601.16614v1)

**作者:** Søren Riis `[一作]` `[通讯]` (Queen Mary University of London), Søren Riis (Queen Mary University of London)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文提出一种“Term Coding”框架，研究给定有限项式恒等式在有限字母表上可解赋值集合的最大规模，并将其转化为依赖图的猜谜游戏，得到一套上下界定理。

**💡 创新点**

创新点在于把通用代数恒等式的可满足性问题提升为可量化的极值问题，并通过归一化、分化和依赖图构造，建立起Term Coding与图熵（猜谜数）之间的“猜谜数夹层定理”，从而让非整数指数的极限被熵和多项式不等式所决定。

**🔧 技术方法**

主要技术包括：术语的归一化与分化（消除子项、碰撞、符号重命名）、构造依赖有向图、使用标记化猜谜游戏、熵/多面体方法（Shannon/非Shannon不等式）以及整数规划（ILP）验证最优解。

**📊 数据集**

实验数据涵盖：对 Steiner 三重系统 n≤9 的最优代码大小、对自正交拉丁方 n≤6 的最优规模、对五环 C₅ 的熵上界、以及小规模网络编码实例（如 2×2×2 relay）的图像最大化结果。

**📈 对比分析**

通过 ILP 证明与熵下界对比，本文验证了在 n→∞ 时猜谜数趋于极限值；在有限 n 时，所得上限与实验结果在多项式比例下接近，且与传统信息流/网络编码极值理论一致。

**⚠️ 局限性**

局限性包括：归一化和分化虽保持指数不变，但在实际计算中引入大量辅助变量导致求解规模膨胀；对非单排序或包含不等式的情形尚未完整覆盖；算法复杂度高，精确计算大图的猜谜数仍是难题。

---

## 161. Emerging Threats and Countermeasures in Neuromorphic Systems: A Survey

**arXiv ID:** 2601.16589 | [PDF](https://arxiv.org/pdf/2601.16589v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 162. E2Former-V2: On-the-Fly Equivariant Attention with Linear Activation Memory

**arXiv ID:** 2601.16622 | [PDF](https://arxiv.org/pdf/2601.16622v1)

**作者:** Lin Huang `[一作]` (IQuest Research), Jia Zhang `[通讯]` (IQuest Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了可扩展的E2Former‑V2架构，利用节点级计算和稀疏注意力实现线性激活内存；

**💡 创新点**

核心创新在于：1）Equivariant Axis‑Aligned Sparsification（EAAS）将SO(3)卷积转化为SO(2)稀疏重排；2）On‑the‑Fly Equivariant Attention自定义Fused Triton kernel，消除边缘张量，实现20×TFLOPS提升；

**🔧 技术方法**

采用SO(3)→SO(2)基变换、Wigner‑6j重耦合、块对角线投影、隐式不变分数、在线softmax、GPU SRAM‑优化的融合核；

**📊 数据集**

在SPICE（蛋白‑配体、分子、溶剂等）和OMol25（大规模分子集合）等公开分子数据集上进行评估；

**📈 对比分析**

与MACE、eSEN、UMA、EquiformerV2、E2Former‑V1等方法对比，SPICE上能量/力MAE最低（相对MACE‑Large提升约48%），OMol25保持竞争力；推理速度提升约20×，可在单GPU上处理至10万原子而不OOM；

**⚠️ 局限性**

仍受限于SO(3)对称性，需在更高阶张量和更大规模任务中进一步验证；对参数调优与实现细节的依赖较高，易受GPU架构差异影响。

---

## 163. Edge-Aware Image Manipulation via Diffusion Models with a Novel Structure-Preservation Loss

**arXiv ID:** 2601.16645 | [PDF](https://arxiv.org/pdf/2601.16645v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 164. The Oval Strikes Back

**arXiv ID:** 2601.16628 | [PDF](https://arxiv.org/pdf/2601.16628v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

---

## 165. How Does Personalized Memory Shape LLM Behavior? Benchmarking Rational Preference Utilization in Personalized Assistants

**arXiv ID:** 2601.16621 | [PDF](https://arxiv.org/pdf/2601.16621v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 166. Boundary and Position Information Mining for Aerial Small Object Detection

**arXiv ID:** 2601.16617 | [PDF](https://arxiv.org/pdf/2601.16617v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 167. AuroraEdge-V-2B: A Faster And Stronger Edge Visual Large Language Model

**arXiv ID:** 2601.16615 | [PDF](https://arxiv.org/pdf/2601.16615v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 168. An Explicit Upper Bound of Generalized Quadratic Gauss Sums and Its Applications for Asymptotically Optimal Aperiodic Polyphase Sequence Design

**arXiv ID:** 2601.16599 | [PDF](https://arxiv.org/pdf/2601.16599v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

---

## 169. Generalized Forms of the Kraft Inequality for Finite-State Encoders

**arXiv ID:** 2601.16594 | [PDF](https://arxiv.org/pdf/2601.16594v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

---

## 170. Integrating Meteorological and Operational Data: A Novel Approach to Understanding Railway Delays in Finland

**arXiv ID:** 2601.16592 | [PDF](https://arxiv.org/pdf/2601.16592v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 171. On Best-of-Both-Worlds Fairness via Sum-of-Variances Minimization

**arXiv ID:** 2601.16579 | [PDF](https://arxiv.org/pdf/2601.16579v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

---

## 172. Predicting Startup Success Using Large Language Models: A Novel In-Context Learning Approach

**arXiv ID:** 2601.16568 | [PDF](https://arxiv.org/pdf/2601.16568v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 173. Eclipse Attacks on Ethereum's Peer-to-Peer Network

**arXiv ID:** 2601.16560 | [PDF](https://arxiv.org/pdf/2601.16560v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 174. PRISM: Purified Representation and Integrated Semantic Modeling for Generative Sequential Recommendation

**arXiv ID:** 2601.16556 | [PDF](https://arxiv.org/pdf/2601.16556v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 175. LUMINA: Long-horizon Understanding for Multi-turn Interactive Agents

**arXiv ID:** 2601.16649 | [PDF](https://arxiv.org/pdf/2601.16649v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 176. Typologically Informed Parameter Aggregation

**arXiv ID:** 2601.16629 | [PDF](https://arxiv.org/pdf/2601.16629v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 177. PROST-LLM: Progressively Enhancing the Speech-to-Speech Translation Capability in LLMs

**arXiv ID:** 2601.16618 | [PDF](https://arxiv.org/pdf/2601.16618v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 178. Omni-directional attention mechanism based on Mamba for speech separation

**arXiv ID:** 2601.16603 | [PDF](https://arxiv.org/pdf/2601.16603v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876`

---

## 179. SCHIGAND: A Synthetic Facial Generation Mode Pipeline

**arXiv ID:** 2601.16627 | [PDF](https://arxiv.org/pdf/2601.16627v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 180. HA2F: Dual-module Collaboration-Guided Hierarchical Adaptive Aggregation Framework for Remote Sensing Change Detection

**arXiv ID:** 2601.16573 | [PDF](https://arxiv.org/pdf/2601.16573v1)

**作者:** Shuying Li `[一作]` (Xi'an University of Posts and Telecommunications), Chuang Yang `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 429 | [OpenAlex ID](https://openalex.org/A5077254624)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种双模块协同引导的分层自适应聚合框架（HA2F）用于遥感图像变化检测。

**💡 创新点**

创新点包括：①动态层级特征校准模块（DHFCM）通过三层交叉注意力和双向感知选择实现多尺度差异特征的自适应融合；②噪声自适应特征细化模块（NAFRM）利用像素级空间偏移场和通道‑空间双重筛选进行精细对齐与噪声抑制。

**🔧 技术方法**

技术手段：CNN‑Transformer混合骨干（ResNet18 + ViT），交叉注意力机制，双向感知选择（HAFS），空间自适应变换（SAT），通道‑空间双重筛选（DFSM）等。

**📊 数据集**

使用数据集：LEVIR‑CD、WHU‑CD 和 SYSU‑CD 三大遥感变化检测基准。

**📈 对比分析**

与九种现有方法（FC‑EF、SNUNet、BIT、ChangeFormer、Paformer、RSMamba、ChangeMamba、CDMamba、STRobustNet）在精度、召回、总体准确率、F1 与 IoU 上进行对比，HA2F 在三数据集上均实现最高或次高分数，显著提升 F1 与 IoU。

**⚠️ 局限性**

局限性：模型规模相对较大，仍需大量标注数据，轻量化设计和半监督学习尚未深入探究。

---

## 181. HapticMatch: An Exploration for Generative Material Haptic Simulation and Interaction

**arXiv ID:** 2601.16639 | [PDF](https://arxiv.org/pdf/2601.16639v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 182. A Unified Calibration Framework for High-Accuracy Articulated Robot Kinematics

**arXiv ID:** 2601.16638 | [PDF](https://arxiv.org/pdf/2601.16638v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 183. Understanding and Improving UMAP with Geometric and Topological Priors: The JORC-UMAP Algorithm

**arXiv ID:** 2601.16552 | [PDF](https://arxiv.org/pdf/2601.16552v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 184. MultiLexNorm++: A Unified Benchmark and a Generative Model for Lexical Normalization for Asian Languages

**arXiv ID:** 2601.16623 | [PDF](https://arxiv.org/pdf/2601.16623v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 185. A Lightweight Medical Image Classification Framework via Self-Supervised Contrastive Learning and Quantum-Enhanced Feature Modeling

**arXiv ID:** 2601.16608 | [PDF](https://arxiv.org/pdf/2601.16608v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 186. Attention-MoA: Enhancing Mixture-of-Agents via Inter-Agent Semantic Attention and Deep Residual Synthesis

**arXiv ID:** 2601.16596 | [PDF](https://arxiv.org/pdf/2601.16596v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 187. LLM is Not All You Need: A Systematic Evaluation of ML vs. Foundation Models for text and image based Medical Classification

**arXiv ID:** 2601.16549 | [PDF](https://arxiv.org/pdf/2601.16549v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 188. Mitigating Bias in Automated Grading Systems for ESL Learners: A Contrastive Learning Approach

**arXiv ID:** 2601.16724 | [PDF](https://arxiv.org/pdf/2601.16724v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 189. Fine-grained quantum advantage beyond double-logarithmic space

**arXiv ID:** 2601.16695 | [PDF](https://arxiv.org/pdf/2601.16695v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

---

## 190. Affinity Contrastive Learning for Skeleton-based Human Activity Understanding

**arXiv ID:** 2601.16694 | [PDF](https://arxiv.org/pdf/2601.16694v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 191. EMemBench: Interactive Benchmarking of Episodic Memory for VLM Agents

**arXiv ID:** 2601.16690 | [PDF](https://arxiv.org/pdf/2601.16690v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 192. Adaptive Reinforcement and Model Predictive Control Switching for Safe Human-Robot Cooperative Navigation

**arXiv ID:** 2601.16686 | [PDF](https://arxiv.org/pdf/2601.16686v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 193. Stable Source Coding

**arXiv ID:** 2601.16680 | [PDF](https://arxiv.org/pdf/2601.16680v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

---

## 194. From Clicks to Consensus: Collective Consent Assemblies for Data Governance

**arXiv ID:** 2601.16752 | [PDF](https://arxiv.org/pdf/2601.16752v1)

**作者:** Lin Kyi `[一作]` (Max Planck Institute for Security and Privacy), Asia Biega `[通讯]`

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

**🎯 论文内容**

提出通过“同意大会”（consent assemblies）实现集体同意，以替代传统的通知与同意机制；

**💡 创新点**

创新点在于将集体决策民主机制（如公民大会）引入数据治理，利用未来回溯和投想设计构建可操作的同意框架；

**🔧 技术方法**

方法技术包括：投想式设计（speculative design）、未来回溯（future backcasting）、委托式小型公民议事（deliberative mini‑publics）以及案例情景（vignettes）来阐述实施细节；

**📊 数据集**

论文为理论性研究，无使用具体数据集；

**📈 对比分析**

未进行实验或性能评估，论文通过案例说明和理论论证展示同意大会的可行性与潜在优势；

**⚠️ 局限性**

局限性包括：可能削弱个体自主性、实施耗时、需中立监督机构、易受商业利益操控、缺乏实证验证和可操作性细节。

---

## 195. Evaluating Generative AI in the Lab: Methodological Challenges and Guidelines

**arXiv ID:** 2601.16740 | [PDF](https://arxiv.org/pdf/2601.16740v1)

**作者:** Hyerim Park `[一作]` (BMW Group), Michael Sedlmair `[通讯]` (University of Stuttgart)

**通讯引用:** 5341 | [OpenAlex ID](https://openalex.org/A5037110552)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对四个实验室用户研究进行多案例反思，识别出五大方法学挑战并给出十八条实践性建议，构建五条评估指南。

**💡 创新点**

首次系统性总结GenAI评估中的方法学困境，提出针对性指南，并通过跨案例反思揭示了生成性系统对实验设计、可信度、可解释性等维度的影响。

**🔧 技术方法**

采用多案例研究、反思式研究、主题分析与Affinity Mapping等质性方法，并在实验中使用了GPT‑4o、DALL·E系列、Flux.1 等主流生成模型。

**📊 数据集**

使用自建实验数据（实验日志、访谈记录、系统事件日志），未使用公开数据集。

**📈 对比分析**

没有传统意义上的性能比较；通过对四个案例的挑战和解决方案进行对比，展示了方法学改进后在实验可控性、信任感和解释性方面的提升。

**⚠️ 局限性**

样本量有限（四个短期实验），缺乏长期、真实环境和多领域的验证，生成模型更新快导致结论需随技术演进持续检验。

---

## 196. ReWeaver: Towards Simulation-Ready and Topology-Accurate Garment Reconstruction

**arXiv ID:** 2601.16672 | [PDF](https://arxiv.org/pdf/2601.16672v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 197. "What I Sign Is Not What I See": Towards Explainable and Trustworthy Cryptocurrency Wallet Signatures

**arXiv ID:** 2601.16751 | [PDF](https://arxiv.org/pdf/2601.16751v1)

**作者:** Yuyang Qin `[一作]` (Chinese University of Hong Kong), Haihan Duan `[通讯]` (Shenzhen MSU-BIT University)

**通讯引用:** 588 | [OpenAlex ID](https://openalex.org/A5068512767)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

针对加密钱包签名的语义不透明问题，本文提出并实现了Signature Semantic Decoder（SSD）中间件，并在模拟钱包界面上验证其对用户决策的影响。

**💡 创新点**

创新点：①将签名数据先行语义解析并构造“行动-主体-客体”框架；②生成简洁自然语言说明并配以颜色级风险提示；③通过实验展示语义透明度可提升用户风险识别、信任感和认知负荷。

**🔧 技术方法**

技术：结构化解析（JSON、ABI）、语义映射、风险评估模型、自然语言模板生成、前端可视化（色块、图标）以及基于浏览器的模拟钱包。

**📊 数据集**

数据集：采集自真实以太坊DApp（如Uniswap、OpenSea、DAO投票等）生成的交易/签名负载；受试者128人（经验型加密用户）完成六类签名任务。

**📈 对比分析**

比较方法：对照组使用原生MetaMask确认界面，实验组使用SSD增强界面；通过决策准确率、主观风险/清晰度/信心评分、NASA‑TLX负荷以及任务时长进行评估。结果显示：实验组决策准确率提升约26%（84.2% vs 67.9%），风险识别更准确，主观清晰度和信心提升，认知负荷下降，任务时长略增但仍可接受。

**⚠️ 局限性**

局限：实验仅在离线模拟环境下进行，缺乏真实交易压力；仅支持以太坊主网标准签名；界面信息密集部分用户反映负荷偏高；未来需在多链、移动端和长期使用情境下验证。

---

## 198. A Step to Decouple Optimization in 3DGS

**arXiv ID:** 2601.16736 | [PDF](https://arxiv.org/pdf/2601.16736v1)

**作者:** Renjie Ding `[一作]` (Hunan University), Xiang Che `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了AdamW-GS优化框架，解耦并重新组合3D高斯散射的优化过程

**💡 创新点**

在更新步长耦合与梯度耦合两大问题上进行分离，提出Sparse Adam、RSR和DAR三大组件并再耦合实现更高效、更可控的正则化

**🔧 技术方法**

采用AdamW式梯度分离、Sparse Adam异步更新、重置状态正则化（RSR）以及属性正则化再组合（DAR）等技术

**📊 数据集**

在MipNeRF‑360、Tanks & Temples、Deep Blending等公共数据集上进行实验

**📈 对比分析**

与原始3DGS、3DGS‑MCMC、MaskGaussian、RePR等方法对比，AdamW‑GS在PSNR/SSIM上提升0.2–0.6dB，显著降低冗余原语并提升优化速度

**⚠️ 局限性**

对超参数敏感，主要正则化仅针对不透明度和尺度，噪声探索在部分场景下不稳定，缺乏自适应调度与更广泛属性正则化

---

## 199. The Geometry of Coalition Power: Majorization, Lattices, and Displacement in Multiwinner Elections

**arXiv ID:** 2601.16723 | [PDF](https://arxiv.org/pdf/2601.16723v1)

**作者:** Qian Guo `[一作]` (University of Delaware), Rui Zhang `[通讯]`

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

**🎯 论文内容**

本文研究在多赢 Top‑k 投票规则下，协同投票团体可以用多少手段迫使现有前 k 名候选人被替换。

**💡 创新点**

创新点在于将投票力拆分为“提升强外来者”和“压制弱胜者”两条独立前缀主导约束，构建 Block–HLP 前缀多面体与共步算术级数格点的精确离散结构，从而得到完全可判定的最大位移问题。

**🔧 技术方法**

采用硬朗–利特尔伍德–帕里（HLP）主导理论、凸多面体与可交换子多面体技术，以及对共步算术级数的模约束推导，配合二分搜索实现多级判定与最大位移计算。

**📊 数据集**

实验使用合成的 Mallows 生成模型和真实的 PrefLib 投票数据集验证算法的准确性与效能。

**📈 对比分析**

与启发式基线相比，算法在预测位移阈值、发现可行区间以及计算最大位移上均达到完美匹配且表现出显著的性能提升，甚至可在 10⁹ 候选人规模下 28 秒内完成。

**⚠️ 局限性**

局限性包括仅在共步算术级数规则下保持多项式可解性；对非算术级数或异步步长的规则，整数可达性不再单靠前缀主导而足以判定，问题在此情形下仍为 NP‑hard。

---

## 200. SWE-Pruner: Self-Adaptive Context Pruning for Coding Agents

**arXiv ID:** 2601.16746 | [PDF](https://arxiv.org/pdf/2601.16746v1)

**作者:** Yuhang Wang `[一作]` (Shanghai Jiao Tong University), Xiaodong Gu `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 2317 | [OpenAlex ID](https://openalex.org/A5033286111)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个自适应线级上下文裁剪框架，作为编程代理与环境之间的轻量中间件，利用目标提示（Goal Hint）动态筛选代码上下文，显著减少 Token 消耗与交互轮次。

**💡 创新点**

创新点包括：① 任务感知的线级裁剪，保持代码结构完整；② 引入 0.6B 参数的轻量化神经筛选器；③ 通过 CRF 训练实现序列级保留决策；④ 结合 Goal Hint 与自监督生成式教师-学生数据生成机制，实现对多种编程任务的自适应裁剪。

**🔧 技术方法**

核心技术：Qwen3‑Reranker‑0.6B 作为编码器，Token‑to‑Line 评分与平均聚合；CRF‑NLL 损失训练线级裁剪头；与 MSE 损失的双重目标学习；自监督数据生成与 LLM‑as‑a‑Judge 过滤；在代理工具中加入可选参数实现裁剪接口。

**📊 数据集**

使用的数据集：61k 条合成的（查询、代码、行级掩码、文档级分数）训练集（从 GitHub 代码库采样）；SWE‑Bench Verified、SWE‑QA、Long Code Completion、Long Code QA 等公开基准用于评估；实验还使用了 Mini SWE Agent、OpenHands 等代理框架。

**📈 对比分析**

与 Full Context、No Context、LLMLingua‑2、Selective‑Context、RAG、LongCodeZip 等基线对比；在 SWE‑Bench Verified 任务上，Token 消耗下降 23–38% 且成功率保持 ≤1% 下降；在 SWE‑QA 上亦实现 29–54% 的 Token 减少；在单轮 Long Code Completion 与 Long Code QA 中，压缩比提升至 5.56–10.92 倍，Accuracy 与 ES 均保持在或优于基线。

**⚠️ 局限性**

局限性：① 目前仅在 Python 代码库验证；② 多语言支持与跨语言迁移尚未评估；③ 合成数据可能导致对真实开发场景的偏差；④ 轻量化筛选器虽低延迟，但仍存在微小额外开销，未来可通过蒸馏或提前退出进一步优化。

---

## 201. Talking about privacy always feels like opening a can of worms. How Intimate Partners Navigate Boundary-Setting in Mobile Phone Without Words

**arXiv ID:** 2601.16658 | [PDF](https://arxiv.org/pdf/2601.16658v1)

**作者:** Sima Amirkhani `[一作]` (University of Siegen), Gunnar Stevens `[通讯]` (Hochschule Bonn-Rhein-Sieg)

**通讯引用:** 3167 | [OpenAlex ID](https://openalex.org/A5001665434)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

通过对20名不同背景、不同关系阶段的伴侣进行半结构化访谈，研究者探索并描述了情侣在移动电话隐私管理中常见的非言语化边界设置方式，提出了“隐私沉默”概念，并构建了按内容敏感度分层的隐私优先级模型。

**💡 创新点**

创新点主要包括：1）首次将“隐私沉默”定义为一种主动的关系策略，填补了先前研究中对隐私规则明确协商的空白；2）从访谈中提炼出五种维持沉默的动机，揭示情感、信任、冲突规避与文化预期在隐私管理中的作用；3）构建了多层次的内容敏感度层级，展示不同信息类型在关系阶段中的共享与保留差异，提供比传统二元共享/隐藏更细粒度的视角。

**🔧 技术方法**

研究方法主要为定性内容分析（QCA），使用 MAXQDA 软件进行编码与主题提炼；访谈采用半结构化问卷，结合文本、语音、视频三种录音模式；此外，研究团队通过反思性讨论与共同编码确保编码的一致性与理论契合。

**📊 数据集**

数据集为20份匿名访谈文本，受访者年龄22-44岁，性别比例13女、6男、1跨性别，涉及多国背景（意大利、伊朗、印度、尼泊尔等），关系状态涵盖单身、情侣、已婚、离异等多样化情境，且覆盖同居与分居两种居住模式。

**📈 对比分析**

由于研究聚焦于质性发现，没有采用量化指标进行性能对比；作者通过主题饱和度评估研究完整性，并在结果中呈现不同主题在访谈中的出现频次与分布，说明数据覆盖的广度与深度。

**⚠️ 局限性**

主要局限包括：① 访谈为回顾性自述，易受记忆偏差与社会期望影响；② 采用多模态访谈，文本与语音在信息量与细节表现上存在差异，可能影响编码一致性；③ 样本规模有限，且集中在德国等西方环境，跨文化推广需要进一步验证；④ 研究聚焦非有害关系，无法直接说明沉默在暴力/控制情境中的风险。

---

## 202. Sim-to-Real Transfer via a Style-Identified Cycle Consistent Generative Adversarial Network: Zero-Shot Deployment on Robotic Manipulators through Visual Domain Adaptation

**arXiv ID:** 2601.16677 | [PDF](https://arxiv.org/pdf/2601.16677v1)

**作者:** Lucía Güitta-López `[一作]` (Comillas Pontifical University), Álvaro Jesús López-López `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于StyleID‑CycleGAN的域适配方法，使得在虚拟环境训练的深度强化学习机器人能够零样本直接在真实工况中完成物体逼近任务

**💡 创新点**

创新点在于结合批归一化替换为去调卷积和身份损失的CycleGAN架构，显著降低翻译伪影并提升样本效率，同时实现了可扩展的零样本部署

**🔧 技术方法**

采用StyleID‑CycleGAN、A3C深度强化学习、ResNet残差网络、PatchGAN判别器、Intel RealSense摄像头等技术

**📊 数据集**

使用自制的虚拟环境图像数据集（约1,300张224×224 RGB图像）与真实摄像头采集图像，分别用于IRB120与UR3e机器人

**📈 对比分析**

与原始CycleGAN和UVCGANv2比较，SICGAN在Wasserstein距离上更低，图像质量更好；代理在虚拟环境中达90–100%成功率，零样本真实环境准确率>95%

**⚠️ 局限性**

主要限制是对背景变化的鲁棒性不足、需额外训练SICGAN模型以及对蓝色目标识别偏差

---

## 203. CER-HV: A CER-Based Human-in-the-Loop Framework for Cleaning Datasets Applied to Arabic-Script HTR

**arXiv ID:** 2601.16713 | [PDF](https://arxiv.org/pdf/2601.16713v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 204. I Guess That's Why They Call it the Blues: Causal Analysis for Audio Classifiers

**arXiv ID:** 2601.16675 | [PDF](https://arxiv.org/pdf/2601.16675v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876`

---

## 205. Using Shadows in Circular Synthetic Aperture Sonar Imaging for Target Analysis

**arXiv ID:** 2601.16733 | [PDF](https://arxiv.org/pdf/2601.16733v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 206. A Categorical Approach to Semantic Interoperability across Building Lifecycle

**arXiv ID:** 2601.16663 | [PDF](https://arxiv.org/pdf/2601.16663v1)

**作者:** Zoltan Nagy `[一作]` (Eindhoven University of Technology), Gioele Zardini `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 133 | [OpenAlex ID](https://openalex.org/A5043524649)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7`

**🎯 论文内容**

本文通过构造理论扩展和使用范畴论中的“提升问题”，实现了建筑信息模型（IFC）与运营模型（BRICK）以及物业管理模型（RealEstateCore）之间的系统化、可验证的数据集成与双向迁移；

**💡 创新点**

创新点在于提出仅需 O(n) 条公式即可完成任意 n 个本体的集成，避免传统点对点映射的 O(n²) 复杂度，并通过范畴论的自洽性保证结构保持与数据一致性；

**🔧 技术方法**

采用了范畴论、第一阶逻辑理论、理论扩展（Theory Extension）和 Categorical Query Language（CQL）等技术；

**📊 数据集**

使用的示例数据集包括 IFC（建筑设计数据）、BRICK（运营监测数据）和 RealEstateCore（物业管理数据）；

**📈 对比分析**

作者与传统的点对点映射及参考本体对比，指出范畴论方法在映射规模、自动推导以及跨本体查询方面更具可扩展性和可维护性，但未给出量化性能指标；

**⚠️ 局限性**

局限性包括对第一阶逻辑和范畴论背景的需求、缺乏大规模基准实验、以及对更多建筑生命周期本体的支持尚未实现。

---

## 207. Flow Matching for Probabilistic Monocular 3D Human Pose Estimation

**arXiv ID:** 2601.16763 | [PDF](https://arxiv.org/pdf/2601.16763v1)

**作者:** Cuong Le `[一作]` (Linköping University), Mårten Wadenbäck `[通讯]` (Linköping University)

**通讯引用:** 330 | [OpenAlex ID](https://openalex.org/A5052501521)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `3f18e8e3-0266-457c-8567-9039b6d2394d` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于连续正则化流和最优传输的概率单目三维人体姿态估计方法 FMPose。

**💡 创新点**

首次将最优传输路径流匹配应用于 3D 姿态生成，并利用可学习的 GCN 直接从 2D 热图中提取多尺度约束。

**🔧 技术方法**

连续正则化流 (CNF)、最优传输 (OT) 路径流匹配、图卷积网络 (GCN)、数值 ODE 求解器 (RK2)。

**📊 数据集**

Human3.6M、MPI-INF-3DHP、3DPW 三大公开基准。

**📈 对比分析**

与 DiffPose、GFPose 等现有最优概率与扩散模型对比，在 MPJPE、P-MPJPE、PCK 等指标上均获得 SOTA 或显著提升。

**⚠️ 局限性**

仅利用 2D 热图，未考虑图像环境与人体交互，导致检测失误时性能下降。

---

## 208. A Feature Extraction Pipeline for Enhancing Lightweight Neural Networks in sEMG-based Joint Torque Estimation

**arXiv ID:** 2601.16712 | [PDF](https://arxiv.org/pdf/2601.16712v1)

**作者:** Kartik Chari `[一作]` (Robotics Innovation Center), Elsa Andrea Kirchner `[通讯]` (Robotics Innovation Center)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了一套针对表面肌电信号的特征提取流程，用于预测肘部和肩部关节扭矩。

**💡 创新点**

创新点在于通过组合时间域、频域与时频域特征，并采用条件特异性归一化与神经激活模型，使得即使是简单的MLP网络也能逼近时序模型的性能。

**🔧 技术方法**

采用的技术包括EMG预处理、第二阶差分神经激活、滑动窗口、RMS/SSC/频带功率、Morlet小波变换、PCA降维、one‑hot编码，以及MLP与TCN两种神经网络。

**📊 数据集**

数据集来自单名24岁男性志愿者，在三种负载(0kg,1.10kg,1.85kg)和两种动作(抓取与复杂运动)下采集的8通道sEMG与运动捕捉记录。

**📈 对比分析**

通过与仅使用原始时间点的TCN对比，本文评估了RMSE、R²和Pearson相关系数，结果显示MLP+特征提取在RMSE上约低10–15%且R²超过0.85，优于TCN。

**⚠️ 局限性**

局限性包括仅有单一受试者、使用静力学估计的参考扭矩、离线实验以及未实现在线实时估计。

---

## 209. Creating a biologically more accurate spider robot to study active vibration sensing

**arXiv ID:** 2601.16691 | [PDF](https://arxiv.org/pdf/2601.16691v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 210. ReLU Networks for Model Predictive Control: Network Complexity and Performance Guarantees

**arXiv ID:** 2601.16764 | [PDF](https://arxiv.org/pdf/2601.16764v1)

**作者:** Xingchen Li `[一作]` (Tsinghua University), Keyou You `[通讯]` (Tsinghua University)

**通讯引用:** 6835 | [OpenAlex ID](https://openalex.org/A5088962631)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究了使用ReLU神经网络逼近模型预测控制（MPC）策略时，网络宽度和深度与闭环性能之间的关系，并给出了满足约束与稳定性要求的网络复杂度上界。还提出了状态感知的非均匀误差缩放方法，以进一步降低网络复杂度。

**💡 创新点**

创新点：
1) 引入状态相关的Lipschitz连续性来精细化闭环收敛分析，取代传统的全局Lipschitz；
2) 通过集合收缩与投影技术保证逼近后的控制输入严格满足硬约束；
3) 设计非均匀误差框架及状态感知缩放函数，实现逼近误差随状态变化而调整，从而显著降低所需网络宽度/深度。

**🔧 技术方法**

采用的技术：ReLU网络逼近理论、集合投影与收缩、状态感知输入/输出缩放、MPC离线训练、数值实验与性能评估。

**📊 数据集**

使用的数据集：对12维状态、3维控制的振荡质量系统，随机采样5×10^5个训练样本，5×10^3个测试样本，全部来自系统状态空间。

**📈 对比分析**

与传统均匀误差方法相比，实验表明更深网络能够在相同参数量下获得更低的均方误差；非均匀误差方法在逼近误差可接受的范围内，能够实现更小的闭环收敛区间，展示了更好的稳定性与性能平衡。

**⚠️ 局限性**

局限性：
- 目前仅针对线性约束的离线MPC，无法直接推广到非线性或实时跟踪控制；
- 非均匀误差逼近的设计和训练相对复杂，且逼近误差上界依赖于难以获得的Lipschitz常数；
- 证明和实验仅在单一振荡质量系统上完成，缺乏更广泛的验证。

---

## 211. Adoption of Generative Artificial Intelligence in the German Software Engineering Industry: An Empirical Study

**arXiv ID:** 2601.16700 | [PDF](https://arxiv.org/pdf/2601.16700v1)

**作者:** Ludwig Felder `[一作]` (Technical University of Munich), Chunyang Chen `[通讯]` (Technical University of Munich)

**通讯引用:** 3922 | [OpenAlex ID](https://openalex.org/A5075639297)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对德国软件工程师采用生成式人工智能工具的使用模式、交互策略和组织影响进行混合方法研究，结合访谈和问卷数据，揭示经验水平与组织规模对工具效益的调节作用。

**💡 创新点**

首次系统梳理了德国市场下的“经验悖论”“上下文墙”“企业基础设施分裂”等关键模式，强调了上下文约束和经验差异在AI工具效益中的决定性作用。

**🔧 技术方法**

采用定性访谈（18份）与定量问卷（109份）相结合的研究设计；问卷包括工具使用频率、提示策略有效性、挑战与定制需求等量表。

**📊 数据集**

使用德国软件工程师自填问卷数据作为主要数据集；通过转录访谈内容生成编码主题。

**📈 对比分析**

通过与2025年Stack Overflow 开发者调查对比工具采纳率，使用交叉表、相关分析和k‑means聚类验证不同经验和组织规模的差异；未给出模型性能指标。

**⚠️ 局限性**

局限性包括便利抽样导致的代表性不足、全自报导致的主观偏差、研究时间点限定在2025年，难以推广至更广泛或未来的AI工具环境。

---

## 212. From Transactions to Exploits: Automated PoC Synthesis for Real-World DeFi Attacks

**arXiv ID:** 2601.16681 | [PDF](https://arxiv.org/pdf/2601.16681v1)

**作者:** Xing Su `[一作]` (Nanjing University), Fengyuan Xu `[通讯]` (Nanjing University)

**通讯引用:** 2264 | [OpenAlex ID](https://openalex.org/A5114549816)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一个从链上攻击交易的执行轨迹自动合成可验证 PoC 的完整框架，解决了人工 PoC 构造的高成本与低覆盖率问题。

**💡 创新点**

创新点包括：① 结合静态反汇编与动态轨迹的双重反汇编（dual‑decompiler）实现对单一执行路径的语义化重构；② 通过上下文提取与伪代码草图引导 LLM 合成，显著降低领域多样性带来的误差；③ 设计了基于资金流 oracle 的验证与迭代细化循环，确保生成 PoC 在语义上可重现原攻击。

**🔧 技术方法**

技术栈主要包含 EVM 反汇编器、执行轨迹分析器、LLM（如 GPT‑4 或等价模型）进行代码生成、Foundry 框架用于 PoC 编译与执行、以及自定义的资金流验证 oracle。

**📊 数据集**

使用了 321 起真实 DeFi 攻击案例（涵盖 20 个月内的事件），并在这些案例上进行大规模评估；同时也测试了 DeepSeek‑R1、Gemini‑2.5‑Flash 等不同 LLM，以验证方法的通用性。

**📈 对比分析**

与传统手工 PoC、仅基于静态分析或单纯 LLM 生成的对比，系统实现了 93% 的攻击重现率，其中 58.78% 直接可验证；平均运行时间 <5 分钟，成本约 0.07 美元/案；在社区 PoC 贡献上，短短两天内贡献 33 条 PoC，取得 38% 份额，获得 900 美元赏金。

**⚠️ 局限性**

局限性包括：① 对极长执行路径的静态分析会耗时或超时；② 需要完整的攻击事务序列，缺失的预处理事务或状态更新可能导致 PoC 失效；③ 对 LLM 生成的依赖性强，若基础模型性能下降会影响最终重现率。

---

## 213. LongCat-Flash-Thinking-2601 Technical Report

**arXiv ID:** 2601.16725 | [PDF](https://arxiv.org/pdf/2601.16725v1)

**作者:** Meituan LongCat Team `[一作]` (Meituan), Ziyuan Zhuang `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一款560B参数的Mixture‑of‑Experts（MoE）推理模型LongCat‑Flash‑Thinking‑2601，专门针对需要与外部环境交互的代理式推理任务进行训练与优化，目标是提升模型在复杂搜索、工具使用与多步推理场景下的表现。

**💡 创新点**

主要创新点包括：
1) 可扩展的环境构建与多域强化学习框架（DORA），能够在上万种异构环境上并行训练；
2) 将真实世界噪声（指令噪声、工具噪声）系统化注入训练，通过自适应噪声曲线提升鲁棒性；
3) Heavy Thinking测试时扩展模式，实现深度与宽度并行搜索，从而显著提升复杂推理任务的成功率；
4) Zigzag Attention设计，基于MLA+SSA的层间稀疏化，使原始全注意力模型在1M上下文长度下实现近1.5×推理速度。

**🔧 技术方法**

采用的核心技术包括：
- 大规模MoE模型与稀疏专家并行；
- 多阶段训练策略：预训练→中训练（扩展上下文+模拟代理轨迹）→后训练强化学习；
- DORA异步多域强化学习框架；
- 环境噪声注入与自适应噪声曲线；
- Heavy Thinking（并行推理+总结）与自监督验证；
- Zigzag Attention与YaRN位置编码；
- 数据合成与质量过滤（文本驱动与环境驱动混合策略）。

**📊 数据集**

使用的数据集与合成数据包括：
- 真实世界代理数据：BrowseComp、RWSearch、τ^2‑Bench、VitaBench、Random Complex Tasks、AMO‑Bench、AIME、HMMT、IMO、GPQA‑Diamond、HLE、LiveCodeBench、OJBench、OIBench、SWE‑bench；
- 通过文本提取与环境构造生成的中训练代理轨迹与工具调用序列；
- 通过自动化噪声注入生成的τ^2‑Noise、Vita‑Noise；
- 采用多域工具图（20+个域、60+工具/图）用于环境扩展。

**📈 对比分析**

与现有开源与闭源模型比较，LongCat‑Flash‑Thinking‑2601 在多项代理式基准上刷新了开源记录：
- BrowseComp（Pass@1）73.1%（无上下文管理）/77.7%（中文版）
- RWSearch 79.5%（仅略低于GPT‑5.2）
- τ^2‑Bench 88.2%（包含多工具情境）
- VitaBench 29.3%（在噪声环境下维持竞争力）
- 代码与数学推理等传统任务保持或超过同等规模模型。整体而言，模型在代理式任务上达到了接近闭源高端模型的表现，并在多任务、多域的鲁棒性与扩展性上表现优异。

**⚠️ 局限性**

限制与待改进之处：
- 训练成本极高，需数千个GPU与数万环境，资源门槛高；
- 依赖于复杂的数据合成与环境构建流水线，若工具或域演化，需重新构造；
- 虽注入噪声提升鲁棒性，但仍可能在极端真实环境下出现误差；
- Heavy Thinking与Zigzag Attention在特定硬件上实现较为复杂，迁移到其他框架时可能需要重构；
- 目前公开的模型版本尚未覆盖所有语言与工具生态，跨语言/跨工具迁移仍有待验证。

---

## 214. Provably Robust Bayesian Counterfactual Explanations under Model Changes

**arXiv ID:** 2601.16659 | [PDF](https://arxiv.org/pdf/2601.16659v1)

**作者:** Jamie Duell `[一作]` (Sheffield Hallam University), Xiuyi Fan `[通讯]` (Nanyang Technological University)

**通讯引用:** 970 | [OpenAlex ID](https://openalex.org/A5101917609)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出Probabilistically Safe Counterfactual Explanations（PSCE）方法，生成在模型更新时仍保持高置信度与低方差的对抗解释。

**💡 创新点**

引入δ‑safe与ϵ‑robust概念，给出对模型后验变化的理论下界保证；将ELBO与距离损失融入优化，兼顾可解释性与稳健性。

**🔧 技术方法**

采用贝叶斯神经网络与Monte Carlo Dropout做后验近似；使用变分自编码器估计数据分布；在优化中加入期望损失、KL下界、ELBO与距离项。

**📊 数据集**

在MNIST、German Credit、Wisconsin Breast Cancer、Spambase、PneumoniaMNIST等标准CE数据集上进行实验。

**📈 对比分析**

与BayesCF、Schut等现有贝叶斯对抗解释方法做基准，5次实验取平均与标准差，PSCE在IM1、Implausibility、Robustness Ratio、Validity等指标上普遍优于对手。

**⚠️ 局限性**

需要模型更新幅度较小（如低学习率）以满足KL界限；依赖贝叶斯近似的质量；在大幅模型变更或后验估计不充分时理论保证失效。

---

## 215. The Green Side of the Lua

**arXiv ID:** 2601.16670 | [PDF](https://arxiv.org/pdf/2601.16670v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 216. Generative Confidants: How do People Experience Trust in Emotional Support from Generative AI?

**arXiv ID:** 2601.16656 | [PDF](https://arxiv.org/pdf/2601.16656v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 217. Variability-Aware Detection and Repair of Compilation Errors Using Foundation Models in Configurable Systems

**arXiv ID:** 2601.16755 | [PDF](https://arxiv.org/pdf/2601.16755v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 218. Automated Road Crack Localization to Guide Highway Maintenance

**arXiv ID:** 2601.16737 | [PDF](https://arxiv.org/pdf/2601.16737v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 219. Watching AI Think: User Perceptions of Visible Thinking in Chatbots

**arXiv ID:** 2601.16720 | [PDF](https://arxiv.org/pdf/2601.16720v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 220. Better Generalizing to Unseen Concepts: An Evaluation Framework and An LLM-Based Auto-Labeled Pipeline for Biomedical Concept Recognition

**arXiv ID:** 2601.16711 | [PDF](https://arxiv.org/pdf/2601.16711v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 221. Revisiting the Role of Natural Language Code Comments in Code Translation

**arXiv ID:** 2601.16661 | [PDF](https://arxiv.org/pdf/2601.16661v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 222. Dynamic Expert-Guided Model Averaging for Causal Discovery

**arXiv ID:** 2601.16715 | [PDF](https://arxiv.org/pdf/2601.16715v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 223. Make the Unhearable Visible: Exploring Visualization for Musical Instrument Practice

**arXiv ID:** 2601.16708 | [PDF](https://arxiv.org/pdf/2601.16708v1)

**作者:** Frank Heyen `[一作]` (Visualization Research Center University of Stuttgart), Michael Sedlmair `[通讯]` (Visualization Research Center University of Stuttgart)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

通过设计和实现33种可视化原型，探讨如何让难以听出的乐器练习模式可视化，并通过定性访谈验证其在练习与教学中的价值。

**💡 创新点**

提出“让不可听见的模式可视化”的研究视角，系统性进行设计空间探索，融合音乐教育与可视化的多种编码与交互方案，并首次把这些设计与真实音乐家的反馈结合起来。

**🔧 技术方法**

采用实时MIDI数据流、交互式可视化（条形、饼图、线图、散点图等）、自动化评估阈值、Web应用原型与可视化脚本，配合音频/视觉叠加展示。

**📊 数据集**

收集并使用18名音乐家提供的练习数据（MIDI记录）和13名参与者的练习录音，构建多种练习场景（节奏、音高、力度、即兴等），以及相应的可视化实例。

**📈 对比分析**

没有基于数值指标的系统性能对比；通过定性访谈、观察和案例分析评估可视化的可用性与教学价值。研究结果显示参与者能在可视化中发现听不见的细节，认为其对练习和自评都有帮助。

**⚠️ 局限性**

受限于仅使用MIDI乐器（键盘、鼓、吉他），无法覆盖音频特征（音色、声学谱），样本量有限，缺乏对可视化界面的量化可用性评估，且未验证长期学习效果。

---

## 224. Developer Perspectives on REST API Usability: A Study of REST API Guidelines

**arXiv ID:** 2601.16705 | [PDF](https://arxiv.org/pdf/2601.16705v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 225. Supporting Stakeholder Requirements Expression with LLM Revisions: An Empirical Evaluation

**arXiv ID:** 2601.16699 | [PDF](https://arxiv.org/pdf/2601.16699v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 226. AgentsEval: Clinically Faithful Evaluation of Medical Imaging Reports via Multi-Agent Reasoning

**arXiv ID:** 2601.16685 | [PDF](https://arxiv.org/pdf/2601.16685v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 227. PLawBench: A Rubric-Based Benchmark for Evaluating LLMs in Real-World Legal Practice

**arXiv ID:** 2601.16669 | [PDF](https://arxiv.org/pdf/2601.16669v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 228. ReViP: Reducing False Completion in Vision-Language-Action Models with Vision-Proprioception Rebalance

**arXiv ID:** 2601.16667 | [PDF](https://arxiv.org/pdf/2601.16667v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 229. Curated endoscopic retrograde cholangiopancreatography images dataset

**arXiv ID:** 2601.16759 | [PDF](https://arxiv.org/pdf/2601.16759v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 230. Standardizing Longitudinal Radiology Report Evaluation via Large Language Model Annotation

**arXiv ID:** 2601.16753 | [PDF](https://arxiv.org/pdf/2601.16753v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 231. Explaining Group Recommendations via Counterfactuals

**arXiv ID:** 2601.16882 | [PDF](https://arxiv.org/pdf/2601.16882v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 232. Theory of Minimal Weight Perturbations in Deep Networks and its Applications for Low-Rank Activated Backdoor Attacks

**arXiv ID:** 2601.16880 | [PDF](https://arxiv.org/pdf/2601.16880v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 233. Orbitopal Fixing in SAT

**arXiv ID:** 2601.16855 | [PDF](https://arxiv.org/pdf/2601.16855v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

---

## 234. The Art of Being Difficult: Combining Human and AI Strengths to Find Adversarial Instances for Heuristics

**arXiv ID:** 2601.16849 | [PDF](https://arxiv.org/pdf/2601.16849v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 235. Information Contraction under $(\varepsilon,δ)$-Differentially Private Mechanisms

**arXiv ID:** 2601.16845 | [PDF](https://arxiv.org/pdf/2601.16845v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

---

## 236. Multi-Agent Non-Discriminatory Contracts

**arXiv ID:** 2601.16835 | [PDF](https://arxiv.org/pdf/2601.16835v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

---

## 237. AI builds, We Analyze: An Empirical Study of AI-Generated Build Code Quality

**arXiv ID:** 2601.16839 | [PDF](https://arxiv.org/pdf/2601.16839v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 238. Calibrated Probabilistic Interpolation for GEDI Biomass

**arXiv ID:** 2601.16834 | [PDF](https://arxiv.org/pdf/2601.16834v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 239. Privacy-Resolution Tradeoff for Adaptive Noisy Twenty Questions Estimation

**arXiv ID:** 2601.16825 | [PDF](https://arxiv.org/pdf/2601.16825v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

---

## 240. PI2I: A Personalized Item-Based Collaborative Filtering Retrieval Framework

**arXiv ID:** 2601.16815 | [PDF](https://arxiv.org/pdf/2601.16815v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 241. Sample-wise Constrained Learning via a Sequential Penalty Approach with Applications in Image Processing

**arXiv ID:** 2601.16812 | [PDF](https://arxiv.org/pdf/2601.16812v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 242. Large Language Models as Automatic Annotators and Annotation Adjudicators for Fine-Grained Opinion Analysis

**arXiv ID:** 2601.16800 | [PDF](https://arxiv.org/pdf/2601.16800v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 243. GTA: Generative Traffic Agents for Simulating Realistic Mobility Behavior

**arXiv ID:** 2601.16778 | [PDF](https://arxiv.org/pdf/2601.16778v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 244. Boosting Deep Reinforcement Learning with Semantic Knowledge for Robotic Manipulators

**arXiv ID:** 2601.16866 | [PDF](https://arxiv.org/pdf/2601.16866v1)

**作者:** Lucía Güitta-López `[一作]` (Comillas Pontifical University), Daniele Nardi `[通讯]` (Sapienza University of Rome)

**通讯引用:** 11018 | [OpenAlex ID](https://openalex.org/A5075651762)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在模拟环境中使用知识图嵌入(KGE)增强深度强化学习( A3C )以实现机器人抓取任务

**💡 创新点**

将KGE直接拼接至视觉特征后的全连接层，提供情境语义信息，显著提升样本效率和准确率

**🔧 技术方法**

深度强化学习（A3C）、卷积神经网络、长短时记忆网络、知识图嵌入（GloVe）

**📊 数据集**

TIAGo 与 IRB120 两机器人模拟实验，目标为杯子、瓶子、盒子，含颜色变化与否的两套数据集

**📈 对比分析**

与基线（无KGE）和部分KGE（仅目标类型）对比，完整KGE在无色域随机化时提升12%准确率，色域随机化时提升16%准确率，学习时间缩短约60%

**⚠️ 局限性**

依赖预先构建的知识图、仅在仿真环境验证、缺乏实地转移和动态场景的鲁棒性

---

## 245. LLM-powered Real-time Patent Citation Recommendation for Financial Technologies

**arXiv ID:** 2601.16775 | [PDF](https://arxiv.org/pdf/2601.16775v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 246. Mixture-of-Models: Unifying Heterogeneous Agents via N-Way Self-Evaluating Deliberation

**arXiv ID:** 2601.16863 | [PDF](https://arxiv.org/pdf/2601.16863v1)

**作者:** Tims Pecerskis `[一作]` (Peeramid Labs), Aivars Smirnovs `[通讯]` (Peeramid Labs)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一种名为 NSED 的运行时多模型（MoM）推理框架，通过动态专家代理与循环共识机制实现推理时计算可扩展性。

**💡 创新点**

创新点包括：将推理视为 RNN 循环而非 DAG；引入可信无中心化的二次投票与对角掩码；利用动态背包算法进行模型选择；设计了热力学疲劳模型与最佳停止阈值；实现了无梯度的自我校正与低内存消耗的交互。

**🔧 技术方法**

技术手段包括多模型协同推理、动态专家代理、二次投票（Quadratic Voting）、对角掩码、循环状态递归、存在惩罚与衰减因子 γ、热力学优化、强化学习或约束求解器实现的背包优化、日志量化传输、以及可视化的代理影响热图。

**📊 数据集**

使用了 AIME 2025 竞赛题集、LiveCodeBench v5 Hard 编码任务集以及 DarkBench 安全评测数据集进行实验评估。

**📈 对比分析**

与单模型和传统多数投票基线对比，消费级 20B 模型 NSED 集成在 AIME 上实现 84% Pass@1（相较 78% 基线提升 6%），在 LiveCodeBench Hard 上提升至 60%（相较 33% 基线提升 27%），且与 70B+ 参数的专有模型保持竞争力；同时在 DarkBench 上显示出更低的 sycophancy 和整体 RMS 分数。

**⚠️ 局限性**

局限性包括：推理延迟高（需多轮同步通信）；对代理多样性与验证精度高度依赖；未实现长时记忆或后期模型微调；仅支持文本推理，无法处理直接执行或动态检索；在低信噪比任务中可能无法突破 Condorcet 限界。

---

## 247. Assessing the Feasibility of Selective Instrumentation for Runtime Code Coverage in Large C++ Game Engines

**arXiv ID:** 2601.16881 | [PDF](https://arxiv.org/pdf/2601.16881v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 248. Optical Tag-Based Neuronavigation and Augmentation System for Non-Invasive Brain Stimulation

**arXiv ID:** 2601.16862 | [PDF](https://arxiv.org/pdf/2601.16862v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 249. Perfect Privacy and Strong Stationary Times for Markovian Sources

**arXiv ID:** 2601.16857 | [PDF](https://arxiv.org/pdf/2601.16857v1)

**作者:** Fangwei Ye `[一作]` (Nanjing University of Aeronautics and Astronautics), Salim El Rouayheb `[通讯]` (Rutgers University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出两种在完美隐私约束下共享马尔可夫链数据的红客（抹除）机制，最大化公开数据量并仅平均抹除常数个数据点；

**💡 创新点**

首次将完美隐私与强平稳时间（SST）关联，证明在可迁移不变链下SST机制可实现最优；提出通用的顺序马尔可夫抹除（SMR）机制，适用于任意有限马尔可夫链；

**🔧 技术方法**

使用信息理论（互信息与Hamming失真）、马尔可夫链理论、强平稳时间构造与谱分析等技术；

**📊 数据集**

无实测数据集，所有结果基于理论推导和对通用有限马尔可夫链的抽象模型；

**📈 对比分析**

与传统DP或信息瓶颈框架比较：该机制在完美隐私下实现最优失真，平均抹除点数不随序列长度增长；理论上给出失真上界与链的谱特征相关；

**⚠️ 局限性**

局限在于仅保护单个初始状态（可推广但未完全证明）、对更一般的部分隐私集合或非马尔可夫相关结构的适用性未完全覆盖，实际实现需进一步评估。

---

## 250. Navigating the Shift: A Comparative Analysis of Web Search and Generative AI Response Generation

**arXiv ID:** 2601.16858 | [PDF](https://arxiv.org/pdf/2601.16858v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 251. Privacy in Human-AI Romantic Relationships: Concerns, Boundaries, and Agency

**arXiv ID:** 2601.16824 | [PDF](https://arxiv.org/pdf/2601.16824v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 252. Incorporating Eye-Tracking Signals Into Multimodal Deep Visual Models For Predicting User Aesthetic Experience In Residential Interiors

**arXiv ID:** 2601.16811 | [PDF](https://arxiv.org/pdf/2601.16811v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 253. Will It Survive? Deciphering the Fate of AI-Generated Code in Open Source

**arXiv ID:** 2601.16809 | [PDF](https://arxiv.org/pdf/2601.16809v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 254. SoS: Analysis of Surface over Semantics in Multilingual Text-To-Image Generation

**arXiv ID:** 2601.16803 | [PDF](https://arxiv.org/pdf/2601.16803v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 255. SLD: Segmentation-Based Landmark Detection for Spinal Ligaments

**arXiv ID:** 2601.16782 | [PDF](https://arxiv.org/pdf/2601.16782v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 256. Tactile Rendering Using Three Basic Stimulus Components in Ultrasound Midair Haptics

**arXiv ID:** 2601.16767 | [PDF](https://arxiv.org/pdf/2601.16767v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 257. A Novel Transfer Learning Approach for Mental Stability Classification from Voice Signal

**arXiv ID:** 2601.16793 | [PDF](https://arxiv.org/pdf/2601.16793v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876`

---

## 258. Uncertainty propagation through trained multi-layer perceptrons: Exact analytical results

**arXiv ID:** 2601.16830 | [PDF](https://arxiv.org/pdf/2601.16830v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 259. Network Security under Heterogeneous Cyber-Risk Profiles and Contagion

**arXiv ID:** 2601.16805 | [PDF](https://arxiv.org/pdf/2601.16805v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 260. Building a Robust Risk-Based Access Control System to Combat Ransomware's Capability to Encrypt: A Machine Learning Approach

**arXiv ID:** 2601.16795 | [PDF](https://arxiv.org/pdf/2601.16795v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 261. E2E-AEC: Implementing an end-to-end neural network learning approach for acoustic echo cancellation

**arXiv ID:** 2601.16774 | [PDF](https://arxiv.org/pdf/2601.16774v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876`

---

## 262. Do LLM hallucination detectors suffer from low-resource effect?

**arXiv ID:** 2601.16766 | [PDF](https://arxiv.org/pdf/2601.16766v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 263. No Validation, No Problem: Predicting Model Performance from a Single Gradient

**arXiv ID:** 2601.16874 | [PDF](https://arxiv.org/pdf/2601.16874v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 264. Provably Learning Attention with Queries

**arXiv ID:** 2601.16873 | [PDF](https://arxiv.org/pdf/2601.16873v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 265. A Multimodal Data Collection Framework for Dialogue-Driven Assistive Robotics to Clarify Ambiguities: A Wizard-of-Oz Pilot Study

**arXiv ID:** 2601.16870 | [PDF](https://arxiv.org/pdf/2601.16870v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 266. Reasoning Promotes Robustness in Theory of Mind Tasks

**arXiv ID:** 2601.16853 | [PDF](https://arxiv.org/pdf/2601.16853v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 267. Stochastic Modeling and Resource Dimensioning of Multi-Cellular Edge Intelligent Systems

**arXiv ID:** 2601.16848 | [PDF](https://arxiv.org/pdf/2601.16848v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 268. ColorConceptBench: A Benchmark for Probabilistic Color-Concept Understanding in Text-to-Image Models

**arXiv ID:** 2601.16836 | [PDF](https://arxiv.org/pdf/2601.16836v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 269. An Efficient Insect-inspired Approach for Visual Point-goal Navigation

**arXiv ID:** 2601.16806 | [PDF](https://arxiv.org/pdf/2601.16806v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 270. REL-SF4PASS: Panoramic Semantic Segmentation with REL Depth Representation and Spherical Fusion

**arXiv ID:** 2601.16788 | [PDF](https://arxiv.org/pdf/2601.16788v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 271. Persuasion Tokens for Editing Factual Knowledge in LLMs

**arXiv ID:** 2601.16781 | [PDF](https://arxiv.org/pdf/2601.16781v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 272. CASP: Few-Shot Class-Incremental Learning with CLS Token Attention Steering Prompts

**arXiv ID:** 2601.16773 | [PDF](https://arxiv.org/pdf/2601.16773v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 273. AutoRegressive Generation with B-rep Holistic Token Sequence Representation

**arXiv ID:** 2601.16771 | [PDF](https://arxiv.org/pdf/2601.16771v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 274. From Atom to Community: Structured and Evolving Agent Memory for User Behavior Modeling

**arXiv ID:** 2601.16872 | [PDF](https://arxiv.org/pdf/2601.16872v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 275. Trapped in the past? Disentangling fluid and crystallized intelligence of large language models using chess

**arXiv ID:** 2601.16823 | [PDF](https://arxiv.org/pdf/2601.16823v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 276. Adaptive Beam Alignment using Noisy Twenty Questions Estimation with Trained Questioner

**arXiv ID:** 2601.16799 | [PDF](https://arxiv.org/pdf/2601.16799v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

---

## 277. Domain-invariant Mixed-domain Semi-supervised Medical Image Segmentation with Clustered Maximum Mean Discrepancy Alignment

**arXiv ID:** 2601.16954 | [PDF](https://arxiv.org/pdf/2601.16954v1)

**作者:** Ba-Thinh Lam `[一作]` (University of North Carolina at Charlotte), Min Xu `[通讯]` (Carnegie Mellon University)

**通讯引用:** 12223 | [OpenAlex ID](https://openalex.org/A5100413849)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了混域半监督医学图像分割框架，联合 Copy‑Paste 机制和 Cluster MMD 对齐，以提升多域鲁棒性。

**💡 创新点**

创新点在于同时通过输入层的 Copy‑Paste 生成跨域样本并在特征层使用 Cluster MMD 进行无标签聚类对齐标签，显式消除域偏差。

**🔧 技术方法**

采用 Teacher‑Student 互训、Copy‑Paste 数据增强、HDBSCAN 聚类、最大均值距离 (MMD) 对齐、Dice+CE 一致性损失等技术。

**📊 数据集**

使用 Fundus 眼底分割数据集和 M&Ms 多中心心血管图像数据集进行实验。

**📈 对比分析**

与多种 SSL 与 DA 方法（FixMatch、BCP、MiDSS 等）对比，在两套数据集上平均提升约 0.8%‑1% 的 Dice 分数，整体表现稳健领先。

**⚠️ 局限性**

局限性包括：假设教师模型输出足够稳定；在极少标注（≤10）情形下未验证；仅针对两类任务，需进一步推广到其他模态和更复杂场景。

---

## 278. The Trajectory Alignment Coefficient in Two Acts: From Reward Tuning to Reward Learning

**arXiv ID:** 2601.16906 | [PDF](https://arxiv.org/pdf/2601.16906v1)

**作者:** Calarina Muslimani `[一作]` (University of Alberta), Peter Wurman `[通讯]` (Sony AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过人类实验验证了Trajectory Alignment Coefficient（TAC）能够有效辅助RL从业者手动调参，并基于TAC提出可微分的Soft‑TAC损失，用于从人类偏好数据中学习奖励模型；随后在Lunar Lander和Gran Turismo 7两个复杂环境中证明Soft‑TAC在提升任务成功率、速度和驾驶风格多样性方面优于交叉熵损失和手工调参奖励；

**💡 创新点**

创新点在于①将TAC作为评估指标直接引入手工调参流程；②设计Soft‑TAC这一可微分近似，可直接用于梯度下降训练奖励模型；③在两个不同域（航天着陆与高保真赛车）中验证了该方法在多样化人类偏好上的有效性；

**🔧 技术方法**

技术手段包括TAC指标计算、Soft‑TAC损失构造、交叉熵（Cross‑Entropy）与Bradley‑Terry模型、SAC/QR‑SAC强化学习算法；

**📊 数据集**

数据集为：Lunar Lander中收集的148条人类偏好对；Gran Turismo 7中分别收集的1429条时间赛偏好与118条对战风格偏好；

**📈 对比分析**

与手工调参奖励、默认奖励、Cross‑Entropy基线进行对比；Soft‑TAC在Lunar Lander的成功率略高、失败类型更少；在GT7时间赛中BIAI比例和最短圈时显著优于其他方法；在对战任务中能实现明显的激进与谨慎驾驶行为；

**⚠️ 局限性**

局限性包括：仅使用线性奖励函数，依赖预定义特征；未验证黑盒奖励网络的可行性；实验规模相对有限，尤其GT7仅用3个随机种子；对TAC参数的敏感性和收敛性尚未深入探讨。

---

## 279. AERO: Adaptive and Efficient Runtime-Aware OTA Updates for Energy-Harvesting IoT

**arXiv ID:** 2601.16935 | [PDF](https://arxiv.org/pdf/2601.16935v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 280. Empowering Medical Equipment Sustainability in Low-Resource Settings: An AI-Powered Diagnostic and Support Platform for Biomedical Technicians

**arXiv ID:** 2601.16967 | [PDF](https://arxiv.org/pdf/2601.16967v1)

**作者:** Bernes Lorier Atabonfack `[一作]` (Carnegie Mellon University), Timothy X Brown `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发并验证了一款面向低资源环境医疗技术人员的 AI 支持平台，用于实时诊断和维修医疗诊断设备。

**💡 创新点**

采用分段向量库的 RAG 架构结合多语言 LLM、离线可用界面以及技术人员社群反馈循环，实现低带宽环境下的实时故障排查。

**🔧 技术方法**

使用 GPT‑3.5 Turbo 生成模型、FAISS 向量检索、React/Flask 前后端、RAG、嵌入、分段向量存储、离线部署和多语言对话技术。

**📊 数据集**

利用 15 份 Philips HDI 5000 设备技术手册、故障码目录，共 90 条错误码及 30 条自然语言查询，构建检索与评估数据集。

**📈 对比分析**

通过内部测试对比 90 条错误码查询和 30 条指导性查询的准确率，错误码检索 100% 正确匹配，指导性查询 80% 成功率，显示在结构化信息检索上表现卓越。

**⚠️ 局限性**

仅在 Philips HDI 5000 单机验证；缺乏真实现场外部评估；模型易产生幻觉；离线功能在多设备多语言场景尚未充分验证；数据集规模有限。

---

## 281. A Scalable Measure of Loss Landscape Curvature for Analyzing the Training Dynamics of LLMs

**arXiv ID:** 2601.16979 | [PDF](https://arxiv.org/pdf/2601.16979v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 282. AgentDrive: An Open Benchmark Dataset for Agentic AI Reasoning with LLM-Generated Scenarios in Autonomous Systems

**arXiv ID:** 2601.16964 | [PDF](https://arxiv.org/pdf/2601.16964v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 283. How Sequential Algorithm Portfolios can benefit Black Box Optimization

**arXiv ID:** 2601.16896 | [PDF](https://arxiv.org/pdf/2601.16896v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620`

---

## 284. Calibrated Similarity for Reliable Geometric Analysis of Embedding Spaces

**arXiv ID:** 2601.16907 | [PDF](https://arxiv.org/pdf/2601.16907v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 285. DataStates-LLM: Scalable Checkpointing for Transformer Models Using Composable State Providers

**arXiv ID:** 2601.16956 | [PDF](https://arxiv.org/pdf/2601.16956v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 286. VisGym: Diverse, Customizable, Scalable Environments for Multimodal Agents

**arXiv ID:** 2601.16973 | [PDF](https://arxiv.org/pdf/2601.16973v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 287. Group-realizable multi-group learning by minimizing empirical risk

**arXiv ID:** 2601.16922 | [PDF](https://arxiv.org/pdf/2601.16922v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 288. In Quest of an Extensible Multi-Level Harm Taxonomy for Adversarial AI: Heart of Security, Ethical Risk Scoring and Resilience Analytics

**arXiv ID:** 2601.16930 | [PDF](https://arxiv.org/pdf/2601.16930v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 289. Auto-Regressive Masked Diffusion Models

**arXiv ID:** 2601.16971 | [PDF](https://arxiv.org/pdf/2601.16971v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 290. Do We Know What They Know We Know? Calibrating Student Trust in AI and Human Responses Through Mutual Theory of Mind

**arXiv ID:** 2601.16960 | [PDF](https://arxiv.org/pdf/2601.16960v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 291. Nishpaksh: TEC Standard-Compliant Framework for Fairness Auditing and Certification of AI Models

**arXiv ID:** 2601.16926 | [PDF](https://arxiv.org/pdf/2601.16926v1)

**作者:** Shashank Prakash `[一作]` (Indraprastha Institute of Information Technology Delhi), Avinash Agarwal `[通讯]` (Department of Technology Government of India)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了Nishpaksh，一款基于印度电信工程中心标准的公平性评估与认证工具，集成问卷、阈值确定和量化评估于一体，并在COMPAS数据集上实现了可审计的公平性报告。

**💡 创新点**

实现了国家标准对齐的统一评估框架，提供风险量化问卷、上下文阈值校准及可证书的报告模板，并首次在工具中嵌入印度特定人口统计代理检测。

**🔧 技术方法**

基于Streamlit的Web仪表盘，使用JSON会话序列化、装饰器缓存、并行向量化计算、Bootstrap置信区间以及多线程交叉验证等技术。

**📊 数据集**

使用公开的COMPAS再犯率数据集进行实验验证。

**📈 对比分析**

与传统公平性工具（IBM AIF360、Fairlearn、What‑If Tool）比较，Nishpaksh在实现与标准对齐、报告可审计性方面表现更优；实验中基准模型表现出接近零偏差，而有偏模型在SPD、EOD等指标上显著恶化。

**⚠️ 局限性**

目前仅支持表格型机器学习模型，缺乏对图像、文本等多模态AI的评估，且需进一步验证跨领域的泛化能力。

---

## 292. Evaluating Wi-Fi Performance for VR Streaming: A Study on Realistic HEVC Video Traffic

**arXiv ID:** 2601.16950 | [PDF](https://arxiv.org/pdf/2601.16950v1)

**作者:** Ferran Maura `[一作]` (Universitat Pompeu Fabra), Boris Bellalta `[通讯]` (Universitat Pompeu Fabra)

**通讯引用:** 4209 | [OpenAlex ID](https://openalex.org/A5089762387)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

开发了一个模块化仿真框架，利用真实 HEVC 编码视频流在 802.11 仿真中评估云端 VR 流媒体的性能，分析不同码率、帧率、GOP/ir 编码、用户数和 AP–HMD 距离对 QoS 的影响。

**💡 创新点**

创新点：① 将真实编码视频流嵌入到 Wi‑Fi 仿真中，准确反映帧大小与时延；② 结合编码层与网络层动态共分析，揭示 ir 编码对延迟波动的优势；③ 通过仿真框架展示多用户情况下的通道饱和阈值（4 个并发用户）。

**🔧 技术方法**

使用技术包括：Rust + NexoSim 离散事件仿真、FFmpeg HEVC 编码、AlVR 原生协议模拟、802.11be（Wi‑Fi 7）PHY/MAC 模型、VMAF 质量评估、A‑MPDU 聚合、MCS 选取与距离建模。

**📊 数据集**

数据集：两段 4K 60FPS 视频样本（BigBuckBunny 与 snow），分别使用 HEVC 编码得到不同码率、GOP/ir 组合，作为仿真中真实的视频流来源。

**📈 对比分析**

比较方法与性能：通过 VF‑RTT、FLR、CU、VMAF 等指标，对码率（10–100 Mbps）、帧率（60–90 FPS）、GOP/ir、用户数（1–6）以及 AP–HMD 距离（1.5 m–11 m）进行系统实验。结果表明：ir 编码产生更小的延迟与帧大小方差；在同一网络下最多支持 4 个并发用户且 FLR<1%、VF‑RTT<33 ms；超过 4 用户后性能急剧下降；高码率提升 VMAF 但会增加延迟波动。

**⚠️ 局限性**

局限性：未包含 OFDMA、MLO、多 AP 协调、DL/UL 统一调度等先进 Wi‑Fi 功能；仅使用单用户 MIMO、HEVC fast preset；仿真时间短；噪声/干扰建模简化；未在真实多机房或大规模网络环境中验证。

---

## 293. LoL: Longer than Longer, Scaling Video Generation to Hour

**arXiv ID:** 2601.16914 | [PDF](https://arxiv.org/pdf/2601.16914v1)

**作者:** Justin Cui `[一作]` (UCLA), Cho-Jui Hsieh `[通讯]` (UCLA)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种训练无关的多头RoPE抖动方法，解决长视频生成中的sink‑collapse问题，并实现无限时长的流式视频生成。

**💡 创新点**

创新点在于通过对不同注意力头的RoPE频率进行微小偏移（多头抖动）来破坏头间同步，显著抑制sink‑collapse；并结合KV缓存、局部注意力与3D因果VAE，实现几小时甚至无限长的视频流式生成。

**🔧 技术方法**

使用RoPE、注意力sink、KV缓存、局部注意力、3D因果VAE解码，以及对RoPE进行多头频率抖动；评估采用No‑Repeat（Sink‑Collapse Max/Avg）和VBench等指标。

**📊 数据集**

实验主要在公开的MovieGen提示集（128个提示）上进行评估，并生成最长12小时的连续视频；模型基于1.3B参数的Wan‑2.1‑T2V进行推理，不进行额外训练。

**📈 对比分析**

与LongLive、Self‑Forcing++、LongLive++、Self‑Forcing、RIFLEx、PI、NTK、YARN等方法对比，在Sink‑Collapse指标上显著降低最大/平均跌幅，且保持与基线相近的生成质量；在75s/100s的视频生成任务中性能优于其他自回归模型。

**⚠️ 局限性**

局限性包括：仅在1.3B模型上验证，模型容量受限；对极长时序的长期记忆仍存在挑战；对控制信号和稀疏/线性注意力机制的支持不足；方法仅通过抖动减弱sink‑collapse，未彻底解决所有周期性位置编码带来的问题。

---

## 294. Preventing the Collapse of Peer Review Requires Verification-First AI

**arXiv ID:** 2601.16909 | [PDF](https://arxiv.org/pdf/2601.16909v1)

**作者:** Lei You `[一作]` (Technical University of Denmark), Iryna Gurevych `[通讯]` (Technical University Darmstadt)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

**🎯 论文内容**

本文提出了“验证优先”视角的同行评审框架，构建了验证压力、信号衰减两大动力的相位转移模型，并给出了真相耦合度与审核频率的闭式关系，进一步推导了科研者投入真相研究的激励崩溃条件；

**💡 创新点**

创新点在于把同行评审视作一个资源受限的“验证瓶颈”系统，用相位图阐释从真相主权到代理主权的转变，并将AI角色定位为可审计的验证工具而非评分或模仿器；

**🔧 技术方法**

技术主要包括形式化模型（混合评估模型、验证压力定义）、数理推导（真相耦合度闭式公式、激励崩溃阈值）、仿真与图示；

**📊 数据集**

论文未使用实际数据集，而是基于理论推导与假设的模拟场景进行分析；

**📈 对比分析**

由于缺乏实验数据，文中未给出与现有方法的性能比较；该工作主要提供理论洞察与设计原则；

**⚠️ 局限性**

局限性包括：模型假设简化了真实评审过程，忽略了人类行为的多样性与组织因素；验证压力与信号衰减的定量估计难以在实际会议中直接测量；

---

## 295. Embedding -based Crop Type Classification in the Groundnut Basin of Senegal

**arXiv ID:** 2601.16900 | [PDF](https://arxiv.org/pdf/2601.16900v1)

**作者:** Madeline C. Lisaius `[一作]` (University of Cambridge), Clement Atzberger `[通讯]` (dClimate Labs)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

针对塞内加尔油豆盆地的小规模农户地区，评估并对比多种基于遥感的作物类型与地表覆盖分类方法，重点验证基于TESSERA嵌入的模型在精度、可解释性、跨年迁移性及可访问性方面的表现。

**💡 创新点**

提出四项评估标准（性能、可解释性、迁移性、可访问性），系统性地将TESSERA、AlphaEarth、原始遥感VI/STM特征与传统随机森林等方法进行对比，并发现TESSERA在小尺度、复杂管理环境下显著优于其他方法。

**🔧 技术方法**

使用地表覆盖与作物类型嵌入（TESSERA 128维、AlphaEarth 64维）、原始多光谱/雷达VI（1106维）与时序统计特征（228维）作为输入；采用多分类器（RF、XGBoost、SVM、LR、MLP）集成，评估准确率、宏F1、加权F1，以及CPU占用。

**📊 数据集**

基于JECAM项目的标注多边形数据（2018年734多边形、2019年669多边形、2021年1870多边形）以及Sentinel-1/2年尺度影像，覆盖塞内加尔Fatick与Niakhar区。

**📈 对比分析**

与传统随机森林、SVM等方法比较，TESSERA在土地覆盖分类中实现最高精度（0.965±0.007），在作物类型分类中平均提高4–8%的准确率；迁移学习实验显示TESSERA在跨年预测中保持相对稳定，AlphaEarth性能波动更大，原始VI/STM方法在跨年时甚至低于随机猜测。

**⚠️ 局限性**

受限于标注数据质量不均、跨年采样策略差异导致准确率下降；AlphaEarth因融合多模态特征导致作物类型判别受损；嵌入方法无法直接区分互作物；此外，嵌入训练成本高、缺乏对混种耕作的显式建模。

---

## 296. Evaluating Large Vision-language Models for Surgical Tool Detection

**arXiv ID:** 2601.16895 | [PDF](https://arxiv.org/pdf/2601.16895v1)

**作者:** Nakul Poudel `[一作]` (Rochester Institute of Technology), Cristian A. Linte `[通讯]` (Rochester Institute of Technology)

**通讯引用:** 1150 | [OpenAlex ID](https://openalex.org/A5085736904)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

评估三款通用视觉语言模型（Qwen2.5、LLaVA1.5、InternVL3.5）在外科工具检测任务中的表现，比较零样本和LoRA微调两种部署方式；

**💡 创新点**

首次系统性比较VLM在手术图像中的零样本与微调检测能力，并结合TIDE框架进行误差级别解析，揭示模型在分类与定位上的互补优势；

**🔧 技术方法**

采用大规模视觉语言预训练模型，利用LoRA参数高效微调，使用Grounding DINO作为开放集检测基线，并通过TIDE误差分类评估模型输出；

**📊 数据集**

使用GraSP机器人手术数据集，提取13个机器人辅助手术视频的帧，标注7类手术工具，共计1125帧用于测试，2324帧用于微调；

**📈 对比分析**

通过对比零样本和微调两种设置下的误差计数，Qwen在分类准确率方面优于GDINO，GDINO在定位精度更高；微调后所有模型误差显著下降，表现出可观的性能提升；

**⚠️ 局限性**

局限性包括仅聚焦工具检测任务，未扩展到其他多模态手术任务；VLM缺乏置信度分数导致无法直接计算mAP；对背景误检率仍较高，影响整体实用性。

---

## 297. LLM-Based Adversarial Persuasion Attacks on Fact-Checking Systems

**arXiv ID:** 2601.16890 | [PDF](https://arxiv.org/pdf/2601.16890v1)

**作者:** João A. Leite `[一作]` (University of Sheffield), Carolina Scarton `[通讯]` (University of Sheffield)

**通讯引用:** 2474 | [OpenAlex ID](https://openalex.org/A5065368839)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6215c339-3735-4be3-8a07-5bbb7004712d` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了利用说服技巧对自动事实检查(AFC)系统进行攻击的框架——Persuasion Injection Attacks，展示了说服性变体如何让模型误判或检索失败。

**💡 创新点**

首创将说服/宣传技巧视为语义级攻击向量，构建了15种说服技术的攻击库，并系统评估其对验证与检索的双重破坏。

**🔧 技术方法**

使用大型语言模型（Qwen2.5-7B-Instruct/ Llama‑3‑8B）生成说服变体，基于RoBERTa-Base的分类器和BM25检索器进行评测；同时采用Blind/Oracle两种攻击者能力模型。

**📊 数据集**

在英语事实核查基准FEVER和FEVEROUS上进行实验，分别包含短句与长句、无结构与结构化证据的两类数据。

**📈 对比分析**

与同类基线攻击（同义词替换、字符扰动、词序交换、改写）对比，Persuasion攻击在Blind设置下的准确率下降超过15%，Oracle设置下几乎崩溃；检索Recall@5在Oracle下跌至0，证明双重失效。

**⚠️ 局限性**

实验仅限英文、基于Wikipedia的受控数据集，且使用规模中等的LLM生成攻击，可能对更强模型或不同语境下的说服效果产生偏差。

---

## 298. SyncLight: Controllable and Consistent Multi-View Relighting

**arXiv ID:** 2601.16981 | [PDF](https://arxiv.org/pdf/2601.16981v1)

**作者:** David Serrano-Lozano `[一作]` (Universitat Autònoma de Barcelona), Javier Vazquez-Corral `[通讯]` (Universitat Autònoma de Barcelona)

**通讯引用:** 887 | [OpenAlex ID](https://openalex.org/A5090215237)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于多视角扩散变压器的SyncLight方法，实现对未标定多视角场景的参数化一致重光照。

**💡 创新点**

创新点在于跨视角注意力实现零样本扩展到任意视角，并采用隐式桥匹配实现一次前向推理的高效重光照。

**🔧 技术方法**

使用多视角扩散变压器、隐式桥匹配（Latent Bridge Matching）以及Lab色彩光图控制。

**📊 数据集**

使用结合Infinigen、BlenderKit和真实拍摄的混合数据集，共约百万张多视角光照对。

**📈 对比分析**

与单视图方法ScribbleLight、Flux等对比，在PSNR、SSIM、ΔE、LPIPS上均显著优于基线，推理速度提升数十倍。

**⚠️ 局限性**

局限于稀有或复杂光源的识别、对极端遮挡或极宽基线时一致性下降，以及缺乏显式3D先验。

---

## 299. Latent Diffusion for Internet of Things Attack Data Generation in Intrusion Detection

**arXiv ID:** 2601.16976 | [PDF](https://arxiv.org/pdf/2601.16976v1)

**作者:** Estela Sánchez-Carballo `[一作]` (Universidad Rey Juan Carlos), José Luis Rojo-Álvarez `[通讯]` (Universidad Rey Juan Carlos)

**通讯引用:** 5202 | [OpenAlex ID](https://openalex.org/A5051286582)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

使用潜在扩散模型（LDM）对IoT攻击流量进行数据增强，以缓解机器学习IDS中的类别不平衡问题。

**💡 创新点**

创新点在于：①将扩散过程迁移到由自编码器学习的低维潜在空间，实现高保真、丰富且多样化的合成样本；②在潜在空间扩散显著降低了采样时间（比直接在数据空间的扩散降低约25%）；③通过对连续与离散特征的联合建模，保持了特征间的依赖关系。

**🔧 技术方法**

采用的技术包括：自编码器（AE）对混合型表格数据进行压缩与归一化；潜在扩散模型（LDM）在潜在空间学习噪声倒退过程；对比实验使用SMOTE、VAE、GAN、传统数据空间扩散（DM）。

**📊 数据集**

实验基于CICIoT2023数据集，聚焦三类攻击：DDoS-ICMP_Fragmentation、Mirai-greip_flood、MITM-ArpSpoofing。

**📈 对比分析**

与SMOTE、VAE、GAN、DM对比后，LDM在DDoS和Mirai攻击的F1分数均达到0.99，显著高于其他方法；在MITM攻击中，DM获得0.53分，LDM为0.44，均较基线（0.00）大幅提升。合成样本质量指标显示，LDM在保持分布相似性与特征相关性、并提供较高多样性的三者平衡方面优于传统方法；采样速度方面，LDM的效率介于VAE/ GAN与DM之间。

**⚠️ 局限性**

局限性包括：①需额外训练自编码器，增加一次性前期成本；②在大规模多类别或流式数据场景下的可扩展性尚未验证；③对非表格型IoT数据（如图像、音频）的适用性未知；④模型对不同网络环境与攻击模式的泛化能力仍待进一步评估。

---

## 300. Is BatchEnsemble a Single Model? On Calibration and Diversity of Efficient Ensembles

**arXiv ID:** 2601.16936 | [PDF](https://arxiv.org/pdf/2601.16936v1)

**作者:** Anton Zamyatin `[一作]` (TU Wien), Thomas Gärtner `[通讯]` (TU Wien)

**通讯引用:** 4068 | [OpenAlex ID](https://openalex.org/A5025793777)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

对比 BatchEnsemble 与 Deep Ensembles、MC Dropout 在 CIFAR‑10、CIFAR‑10C、SVHN 与 MNIST 上的准确率、校准与 OOD 检测性能，并通过理论与实验证明 BatchEnsemble 的多样性不足导致其表现几乎与单模型相同。

**💡 创新点**

通过证明 BatchEnsemble 的参数空间为度量零子集并量化其成员的功能与参数空间相似度，揭示低成本集成方法在多样性与不确定性评估方面的根本局限。

**🔧 技术方法**

使用 Rank‑1 乘法扰动的参数化、Jensen‑Shannon 散度、预测熵、误差度量、ResNet‑18 训练、向量化前向传播以及 MNIST MLP 进行多样性评估。

**📊 数据集**

采用 CIFAR‑10、CIFAR‑10C、SVHN 进行主实验，MNIST 用于控制实验验证多样性。

**📈 对比分析**

在相同架构（ResNet‑18）与 75 轮训练的设置下，BatchEnsemble 在准确率、NLL、ECE、JSD 等指标上与单模型几乎相同；Deep Ensemble 在所有指标上明显优于其余方法；MC Dropout 介于两者之间。

**⚠️ 局限性**

实验结果与先前报告存在差异，可能由于模型宽度、深度、训练时间不同；BatchEnsemble 在现有配置下缺乏多样性，无法在等参数预算下实现真正的 ensemble 效果。

---

## 301. Reward-Forcing: Autoregressive Video Generation with Reward Feedback

**arXiv ID:** 2601.16933 | [PDF](https://arxiv.org/pdf/2601.16933v1)

**作者:** Jingran Zhang `[一作]` (University of California), Justin Cui `[通讯]` (University of California)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a8e75ba4-7a2d-4153-b003-06c94533add0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

利用奖励信号引导自回归视频扩散模型，实现无须大规模教师蒸馏的高效视频生成；

**💡 创新点**

通过先用ODE轨迹预训练获得运动先验，再用奖励模型细化纹理，解耦教师与奖励优化，完成轻量化训练；

**🔧 技术方法**

结合视频扩散、ODE轨迹初始化、强化学习奖励（ImageReward等）、自回归采样、分布匹配蒸馏（DMD）以及EMA等技术；

**📊 数据集**

主要在VBench基准上评估，训练阶段不依赖任何外部数据，仅使用教师模型产生的轨迹；

**📈 对比分析**

与多类基线（双向、逐帧、自回归块）在VBench总分/质量/语义上对比，chunk‑wise方式下总分84.92，超越Self‑Forcing（84.31），接近教师Wan2.1（84.26）；

**⚠️ 局限性**

与教师对齐与奖励冲突导致难以兼顾；随机帧奖励会削弱运动质量；对某些视频仍存在一致性问题；需更强奖励模型提升性能。

---

## 302. Multigrade Neural Network Approximation

**arXiv ID:** 2601.16884 | [PDF](https://arxiv.org/pdf/2601.16884v1)

**作者:** Shijun Zhang `[一作]` (Hong Kong Polytechnic University), Yuesheng Xu `[通讯]` (Old Dominion University)

**通讯引用:** 5928 | [OpenAlex ID](https://openalex.org/A5026572582)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出多阶深度学习(MGDL)框架，并证明在固定宽度网络中通过逐级训练可使残差严格递减、最终趋于零。

**💡 创新点**

首次给出MGDL训练过程的严格误差收敛理论，证明任意连续目标函数均可构造固定宽度多阶网络，使残差在每一层严格递减并在所有L^p范数下收敛至0。

**🔧 技术方法**

采用算子理论构造正向收缩算子，并利用分层残差学习与ReLU激活实现梯度级联，网络结构为宽度5d的单隐藏层块。

**📊 数据集**

仅在人工合成的1维和2维正弦/余弦混合函数上进行实验，训练样本为均匀采样，测试样本同样随机采样。

**📈 对比分析**

与同等宽度、深度、激活和训练次数的全连接网络做端到端训练对比，MGDL在训练误差、测试误差和L^∞误差方面均显著优于传统方法。

**⚠️ 局限性**

证明仅适用于固定宽度、单隐藏层块和ReLU激活的网络，未涵盖更复杂激活、变宽网络、实际优化动态以及对真实数据的泛化性。

---

## 303. AnyView: Synthesizing Any Novel View in Dynamic Scenes

**arXiv ID:** 2601.16982 | [PDF](https://arxiv.org/pdf/2601.16982v1)

**作者:** Basile Van Hoorick `[一作]` (Toyota Research Institute), Vitor Campagnolo Guizilini `[通讯]` (Toyota Research Institute)

**通讯引用:** 262 | [OpenAlex ID](https://openalex.org/A5113933181)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种无显式几何重建、能在极端相机位姿变化下进行一致性动态视角合成的扩散框架；

**💡 创新点**

核心创新在于将相机参数编码为 Plücker 影射并与视频特征一起送入隐式视频扩散模型，完成多视角隐式 4D 理解，避免了昂贵的重投影与优化；

**🔧 技术方法**

技术实现基于 Cosmos 隐式视频扩散 Transformer，采用视频分词、Plücker 影射、旋转位置嵌入与多视角交叉注意力；

**📊 数据集**

使用了 12 个公开 3D/4D 数据集（包含 Kubric‑5D、Driving、Robotics、Human Activity 等多域），并构建了新 benchmark AnyViewBench；

**📈 对比分析**

与现有 DVS 方法（如 GCD、TrajAttn、CogNVS 等）进行对比，实验表明在狭窄与极端场景均能取得更高 PSNR/SSIM/LPIPS，尤其在极端视角缺失情况下表现明显优于基线；

**⚠️ 局限性**

局限性包括仍依赖大量多视角训练数据、对极端遮挡仍可能产生不确定性、未对非光滑动态场景（如高速运动）进行充分评估。

---

## 304. Spatial-Agent: Agentic Geo-spatial Reasoning with Scientific Core Concepts

**arXiv ID:** 2601.16965 | [PDF](https://arxiv.org/pdf/2601.16965v1)

**作者:** Riyang Bao `[一作]` (Emory University), Liang Zhao `[通讯]` (Emory University)

**通讯引用:** 6429 | [OpenAlex ID](https://openalex.org/A5061568038)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Spatial-Agent，构建 GeoFlow 图将地理分析问答转化为可执行的空间工作流，完成从概念提取到工具调用的完整推理流程。

**💡 创新点**

创新点在于将 GIScience 的核心概念与功能角色融合，形成可验证的概念转换图；利用宏模板实现结构化图生成，并通过两阶段 SFT + DPO 微调使模型内化空间约束。

**🔧 技术方法**

使用 LLM 代理、概念与角色识别、模板驱动图生成、GeoFlow 图因子化与工具映射、PostGIS/ArcGIS 等空间运算工具，辅以 SFT 与 DPO 的 fine‑tuning。

**📊 数据集**

实验基于 MapEval‑API（覆盖 180 城市、4 类任务）和 MapQA（3,154 个问题）两个大规模地理推理基准。

**📈 对比分析**

与 Direct LLM、ReAct、Reflexion、Plan‑and‑Solve 等基线比较，Spatial-Agent 在 MapEval‑API 和 MapQA 上实现 45–72% 的准确率提升（最高 71.9%），同时保持可接受的延迟与成本。

**⚠️ 局限性**

局限包括受外部地理 API 数据质量与可用性影响、模板库覆盖不全导致新问题难以处理、Fine‑tuning 需要大量标注、目前仅验证英语城市环境，专业领域性能未知。

---

## 305. 3D Molecule Generation from Rigid Motifs via SE(3) Flows

**arXiv ID:** 2601.16955 | [PDF](https://arxiv.org/pdf/2601.16955v1)

**作者:** Roman Poletukhin `[一作]` (Technical University of Munich), Stephan Günnemann `[通讯]` (Technical University of Munich)

**通讯引用:** 14160 | [OpenAlex ID](https://openalex.org/A5074504351)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `40105733-5154-44cd-8090-a8cab9e64b07` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

将分子生成从原子层面提升到刚体片段层面，在 SE(3) 流上联合离散与连续生成。

**💡 创新点**

提出基于刚体片段的 SE(3) 流匹配框架，兼顾离散片段类别和连续几何；并设计平面环为核心的分片词典。

**🔧 技术方法**

使用 SE(3) 流匹配、连续‑离散联合流、IPA 注意力、对称对齐、CTMC 离散流等技术。

**📊 数据集**

在 QM9、GEOM-Drugs、QMugs 等公开数据集上进行实验。

**📈 对比分析**

与原子级别基线（EDM、GeoLDM、EquiFM、END、GeoBFN 等）对比，在大型分子上实现更高原子稳定性、更少采样步骤、3.5× 更紧凑的表示；在条件生成任务中也优于基线。

**⚠️ 局限性**

依赖预定义词典导致大词典稀疏、罕见片段采样不足；未完整建模键网络；分片策略仍需改进。

---

## 306. Strategies for Span Labeling with Large Language Models

**arXiv ID:** 2601.16946 | [PDF](https://arxiv.org/pdf/2601.16946v1)

**作者:** Danil Semin `[一作]` (Charles University), Zdeněk Kasner `[通讯]` (Charles University)

**通讯引用:** 1059 | [OpenAlex ID](https://openalex.org/A5032644567)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文探讨了使用生成式大型语言模型进行 span 标注的多种策略，并提出了一种新的受限解码方法 LogitMatch，用于解决生成文本与输入文本对齐问题。

**💡 创新点**

创新点在于将三大类策略（标记、索引、匹配）系统归类，并设计了只在解码阶段修改 logits 的 LogitMatch 方法，以保证模型生成的 span 必须是输入文本的连续子串，从而显著降低内容匹配错误。

**🔧 技术方法**

技术上使用了不同的提示策略（XML‑style 标记、字符/单词索引、JSON 输出）以及受限解码、结构化输出、链式推理等手段，并在 Qwen3‑8B、Mistral‑Small‑24B、Llama‑3.3‑70B 等主流开源 LLM 上进行实验；对 GPT‑5‑mini 进行有限测试。

**📊 数据集**

使用了四个任务的数据集：多语言 NER（UniversalNER）、英语 GEC（MultiGEC Write&Improve）、机器翻译错误检测（ESA‑MT WMT24）和人工构造的多次出现模式查找（CPL）。

**📈 对比分析**

与其他方法比较时，以覆盖率校正的 F1 评估；标记（Tag）方法在所有任务上表现最稳健，LogitMatch 在需要区分多次出现的 CPL 任务中提升了 30–40% 的 F1，而在 NER、GEC 等任务中也能与基线相当；索引方法整体表现最差；结构化输出能减少解析错误但不总能提升整体性能。

**⚠️ 局限性**

局限性包括：只关注可解析文本输出的策略，未探索更重资源的编码器改造方案；对多重出现的 disambiguation 仍不完善；受限解码在处理 token 边界时会产生 tokenization 细节问题；并未针对模型本身进行微调或架构改造。

---

## 307. Information Representation Fairness in Long-Document Embeddings: The Peculiar Interaction of Positional and Language Bias

**arXiv ID:** 2601.16934 | [PDF](https://arxiv.org/pdf/2601.16934v1)

**作者:** Elias Schuhmacher `[一作]` (University of Zurich), Simon Clematide `[通讯]` (University of Zurich)

**通讯引用:** 1562 | [OpenAlex ID](https://openalex.org/A5073027507)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究多段长文档嵌入时的位置信息与语言公平性问题，探讨首位偏差与语言偏差对检索效果的影响。

**💡 创新点**

提出了基于置换的评估框架，用以量化文档位置信息和语言对嵌入表示的影响，并设计了推理时注意力校准方法，能显著减少首位偏差与语言偏差。

**🔧 技术方法**

采用Transformer编码器的pooling‑token（mGTE）和mean‑pooling（jina‑v3）嵌入模型，结合自注意力分析与注意力均衡校准技术。

**📊 数据集**

构建多语言可比语料库，以对齐长度的Wikipedia条目为基础，生成单语与多语混合的长文档（3–6段、≤8192 token）。

**📈 对比分析**

通过OLS回归与余弦相似度评估，与未校准模型相比，注意力校准后首位段落表示下降、后置段落表示提升，整体表示公平性显著改善。

**⚠️ 局限性**

仅针对Encoder‑based模型、仅使用Wikipedia语料、未评估下游检索任务、校准方法在单一模型上验证、未考虑主题/域等其他偏差来源。

---

## 308. Conditionally Tight Algorithms for Maximum k-Coverage and Partial k-Dominating Set via Arity-Reducing Hypercuts

**arXiv ID:** 2601.16923 | [PDF](https://arxiv.org/pdf/2601.16923v1)

**作者:** Nick Fischer `[一作]`, Mirza Redzic `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

针对图论中的部分k-支配集（Partial k-Dominating Set）和集合覆盖问题（Max k-Cover）提出了新的时间复杂度分析与算法，并给出了它们相对于k-支配集、3-均匀超图团检测和k-向量异或（k-OV）假设的下界。

**💡 创新点**

创新点在于：① 通过引入“arity-reducing hypercut”与“bundle”概念，构建了一个双赢的求解框架；② 通过“(k,h)-maxIP / minIP”的正则化技术，完成了对不同超图团假设的统一归约；③ 结合矩阵乘法和超图表示，给出了与ω/3指数相关的最优时间复杂度。

**🔧 技术方法**

主要技术包括：矩阵乘法（O(n^ω)）实现最大权三角搜索；超图表示与arity-reducing hypercut；正则化 (k,h)-IP 变换；归约到超图团检测与k-OV；递归分解与bundle猜测；以及极限情形下的特殊处理。

**📊 数据集**

由于研究对象为理论复杂度，论文未使用任何实际数据集；所有实验均基于合成实例与理论分析。

**📈 对比分析**

方法比较：对比传统的暴力枚举、动态规划与近似贪心算法，展示了在不同参数范围（如最大度Δ、最优值t）下实现了比n^k多项式下降到O(min{n,Δ^2}^ω/3 + Δ^3/2)的时间。性能在理论上优于已知最坏情况，但在实际实现上尚未给出可运行程序。

**⚠️ 局限性**

限制包括：① 结果依赖于矩阵乘法指数ω，若ω未进一步下降则算法无法进一步加速；② 对于极低度或高频域的实例，算法退化回O(n^k)；③ 归约过程中维度大幅扩张，导致实际空间与时间成本高；④ 仅提供理论证明，缺乏可扩展到大规模实际问题的实现细节。

---

## 309. Recovering Communities in Structured Random Graphs

**arXiv ID:** 2601.16910 | [PDF](https://arxiv.org/pdf/2601.16910v1)

**作者:** Michael Kapralov `[一作]` (EPFL), Weronika Wrzos-Kaminska `[通讯]` (EPFL)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了在随机图中恢复重叠社区结构的问题，特别是在高维超立方体图中，探讨了如何从边的随机样本中识别出多个重叠的稀疏切割。

**💡 创新点**

创新点在于证明了在超立方体图中，存在多个重叠的稀疏切割，并且可以在一定的采样率下准确恢复这些切割，尤其是当采样率满足特定条件时。

**🔧 技术方法**

使用了超立方体图的切割稀疏化界限，结合了Friedgut、Kalai和Naor关于布尔函数的定理，以及Karger's切割计数论证。

**📊 数据集**

使用了d维超立方体图作为数据集，特别是通过对边的随机采样来进行分析。

**📈 对比分析**

与现有方法相比，本文的方法在高概率下能够恢复所有d个切割，并且在适当的样本下可以实现精确恢复。性能表现出在特定条件下的最优性。

**⚠️ 局限性**

限制在于需要较高的采样率以确保恢复的准确性，且在处理更复杂的图结构时可能面临挑战。

---

## 310. GRIP: Algorithm-Agnostic Machine Unlearning for Mixture-of-Experts via Geometric Router Constraints

**arXiv ID:** 2601.16905 | [PDF](https://arxiv.org/pdf/2601.16905v1)

**作者:** Andy Zhu `[一作]` (Georgia Institute of Technology), Pan Li `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 10029 | [OpenAlex ID](https://openalex.org/A5100455171)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `afceb026-1760-41ae-8d86-010831a37d97`

**🎯 论文内容**

本文提出了GRIP框架，利用几何约束投影到保留集的null‑space，保证Mixture‑of‑Experts模型在机器无学习时路由保持稳定，真正实现知识遗忘；

**💡 创新点**

创新点在于将路由不变性与参数可塑性解耦，设计了专家特定的零空间投影约束，并提供训练时与后训练纠正两种高效实现方案；

**🔧 技术方法**

技术手段包括null‑space投影、投影梯度下降（PGD）、随机Kaczmarz迭代、后训练纠正（PTC）以及基于Jaccard相似度的路由稳定性度量；

**📊 数据集**

实验使用30B规模的Qwen3‑30B‑A3B MoE模型，评估数据集包括WMDP、MUSE、MMLU、TruthfulQA，并与SEUF、Dense基线及多种无学习算法比较；

**📈 对比分析**

在所有基线方法上，GRIP将路由稳定性提升至≥0.95，保留准确率提升59‑94%，遗忘准确率保持或提升，攻击恢复率从61%降至3%，计算时间仅提升约1.2‑1.7倍；

**⚠️ 局限性**

局限性包括需存储全部保留集表示导致O(LdN_r)内存开销、后训练纠正的O(d^3)复杂度、仅针对top‑k硬路由结构、以及验证无学习效果的可观测性不足。

---

## 311. FedSGM: A Unified Framework for Constraint Aware, Bidirectionally Compressed, Multi-Step Federated Optimization

**arXiv ID:** 2601.16897 | [PDF](https://arxiv.org/pdf/2601.16897v1)

**作者:** Antesh Upadhyay `[一作]` (Purdue University), Abolfazl Hashemi `[通讯]` (Purdue University)

**通讯引用:** 437 | [OpenAlex ID](https://openalex.org/A5036900440)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出FedSGM框架，统一解决联邦学习中的功能约束、通信压缩、多步本地更新和部分参与，并引入软硬切换机制；

**💡 创新点**

首次在联邦学习中同时处理功能约束、双向压缩、错误反馈、局部多步更新与部分参与，并实现投影无、双重无的单纯梯度更新，给出收敛理论；

**🔧 技术方法**

采用Switching Gradient Method与错误反馈（EF14/EF21）相结合的双向压缩策略（Top‑K/量化），实现软/硬切换、投影无、单纯梯度更新及高概率分析；

**📊 数据集**

实验使用乳腺癌数据集进行NP分类任务，以及Cartpole环境下的受限马尔科夫决策过程（CMDP）；

**📈 对比分析**

与FedAvg、压缩FedAvg及传统投影方法对比，在相同通信/参与设置下，FedSGM在目标函数和约束满足上收敛更快，软切换显著降低振荡；实验表明即使在Top‑K 0.1压缩和部分参与（m=10/20）下仍能满足约束并提升奖励；

**⚠️ 局限性**

理论仅适用于凸问题，非凸情况仅有经验验证；使用梯度下降而非随机梯度；未考虑异步更新、隐私保护等实际需求。

---

## 312. MAGE-KT: Multi-Agent Graph-Enhanced Knowledge Tracing with Subgraph Retrieval and Asymmetric Fusion

**arXiv ID:** 2601.16886 | [PDF](https://arxiv.org/pdf/2601.16886v1)

**作者:** Chi Yu `[一作]` (Inner Mongolia University), Zhiyi Duan `[通讯]` (Inner Mongolia University)

**通讯引用:** 120 | [OpenAlex ID](https://openalex.org/A5002768025)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种多代理多视角的知识追踪框架MAGE-KT，利用多代理协同抽取高质量的知识概念关系图，结合学生-问题交互图，通过学生条件子图检索和非对称交叉注意力融合，实现更精准的下一题预测。

**💡 创新点**

创新点包括：① 采用三代理（语义、评分、仲裁）循环式抽取多类型知识关系，提升关系质量；② 引入学生条件子图检索机制，聚焦高价值信息并抑制注意力扩散；③ 设计非对称交叉注意力融合模块，显式引导学生、问题与知识概念之间的双向信息流。

**🔧 技术方法**

核心技术涵盖：多代理关系抽取（基于LLM Qwen-Plus、DeepSeek-R1、GPT-4）；基于IRT的学生-问题交互图构建；子图检索与 k-hop 邻域聚合；非对称交叉注意力融合；GRU 时序编码；交叉熵损失训练。

**📊 数据集**

在三大公开数据集上进行实验：ASSIST09、Junyi 和 Statics2011。

**📈 对比分析**

与传统序列模型（DKT、DKT+、DKVMN）、Transformer模型（SAKT、AKT、SAINT）以及图模型（GKT、GIKT、MGEKT、TCL4KT、DyGKT、STHKT、DGEKT）进行对比。MAGE-KT 在所有数据集上均实现最高 AUC 与 ACC，尤其在预测准确性上显著优于全图传播方式，且计算开销更低。

**⚠️ 局限性**

局限性包括：依赖多代理LLM，可能带来推理成本和生成不确定性；子图检索与邻域阈值设定需经验调参；目前未针对持续学习或实时自适应场景进行扩展，且对极端稀疏数据的鲁棒性待进一步验证。

---

## 313. GPA-VGGT:Adapting VGGT to Large scale Localization by self-Supervised learning with Geometry and Physics Aware loss

**arXiv ID:** 2601.16885 | [PDF](https://arxiv.org/pdf/2601.16885v1)

**作者:** Yangfan Xu `[一作]`, Jun Mao `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `51c0528b-f690-4182-ae60-bb5f046c276c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种自监督的多序列三维空间一致性框架，利用VGGT Transformer实现大规模视觉定位与单视图深度预测；

**💡 创新点**

创新点在于通过物理感知的自监督损失，将监督范围从局部帧对提升至跨序列全局一致性，并引入鲁棒的硬视图选择与自动遮挡剔除；

**🔧 技术方法**

技术主要包括滑动窗口多视角投影、光度与几何一致性损失、基于Transformer的全局注意力、硬视图最小化策略及自遮挡掩码；

**📊 数据集**

在KITTI Odometry Benchmark上进行评估，使用320×1024尺寸的彩色视频帧；

**📈 对比分析**

与传统自监督方法（MonoDepth2、SC‑DepthV3）及监督式几何模型（VGGT、Streaming VGGT、VGGSfM、MapAnything、DUSt3R）进行对比，GPA‑VGGT在Seq07/09的ATE和RPE分别提升约1.5‑2倍，表现最佳；

**⚠️ 局限性**

局限性包括仍依赖全局视角一致性假设，动态场景与光照剧变下的鲁棒性待进一步提升，以及对极端大规模环境的推理效率尚需优化。

---

