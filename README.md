## tutorial

### 1.大模型推理业务简介

**大模型推理业务应用场景**

**大模型推理业务模型特征**

- MOE FFN

**大模型推理业务关键指标**

### 2.大模型推理关键技术

#### 1）大模型推理整体架构

- 大模型的系统级优化需要考虑多方因素，主要划分为充分利用带宽 & 内存，最大化硬件利用率；

- 大模型图例在prefill阶段通常是 compute-bound，在Decode阶段通常是 memory-bound。通过增加batch size等手段可以改变算数密度从而改变 roofline模型图上的位置；

- 大batch可以使decode阶段的MLP变成compute-bound，但对Attention计算没有收益（如果使用**投机推理**，可以通过增加验证长度提升MLP和Attention的算数密度*），这些优化手段可以逼近理论最优点；

- 量化作为一种优化手段也会改变roofline模型图，权重/KV 量化降低带宽需求，激活量化提供理论算力，W量化改变memory-bound区域的斜率，A量化改变峰值表现；

- 一个推理业务流（特别是多模）往往不是单个模型，需要结合业务场景大小模型对接做到部署最优，训练只要把单个模型训练出俩，推理要考虑的事情就很多了；

#### 2）模型部署

**模型切分（TP，PP，CP）**

- TP tensor并行

- PP 流水并行

- CP 长序列并行

**P-D分离**

**LLM投机推理**

这里涉及一个概念：LLM的全量推理和增量推理

#### 3）计算优化

**KV Cache**

【Flash Attention 为什么那么快？原理讲解】 https://www.bilibili.com/video/BV1UT421k7rA/?share_source=copy_web

- KV公共前缀及一查代算

- 长序列KV OFFLOAD及压缩

- KV压缩

**Flash Attention**

**通信计算融合（MC2）**

**MOE GMM算子**

#### 4）模型小型化

**量化**

- 量化类型

- 非量化/全量化/伪量化

- per-xx量化

**KV小型化**

### 3.大模型推理昇腾NPU和友商关键硬件差异


