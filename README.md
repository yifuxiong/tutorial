## tutorial

### 1.前言 Transformer

### 2.大模型推理整体架构



### 3.大模型推理关键技术

- 大模型额系统级优化需要考虑多方因素，主要划分为充分利用带宽 & 内存，最大化硬件利用率；

- 大模型图例在prefill阶段通常是 compute-bound，在Decode阶段通常是 memory-bound。通过增加batch size等手段可以改变算数密度从而改变 roofline模型图上的位置；

- 大batch可以使decode阶段的MLP变成compute-bound，但对Attention计算没有收益（如果使用**投机推理**，可以通过增加验证长度提升MLP和Attention的算数密度*），这些优化手段可以逼近理论最优点；

- 量化作为一种优化手段也会改变roofline模型图，权重/KV 量化降低带宽需求，激活量化提供理论算力，W量化改变memory-bound区域的斜率，A量化改变峰值表现；

- 一个推理业务流（特别是多模）往往不是单个模型，需要结合业务场景大小模型对接做到部署最优，训练只要把单个模型训练出俩，推理要考虑的事情就很多了；

#### 1）模型切分（TP，PP，CP）

- TP tensor并行

- PP 流水并行

- CP 长序列并行

#### 2）P-D分离

#### 3）LLM投机推理

这里涉及一个概念：LLM的全量推理和增量推理

### 4.


### 5.计算优化

#### 1）KV Cache

【Flash Attention 为什么那么快？原理讲解】 https://www.bilibili.com/video/BV1UT421k7rA/?share_source=copy_web

#### 2）Flash Attention

#### 3）通信计算融合

#### 4）MOE GMM算子

### 6.模型小型化

#### 1）量化

- 量化类型

- 非量化/全量化/伪量化

- per-xx量化

#### 2）KV小型化


