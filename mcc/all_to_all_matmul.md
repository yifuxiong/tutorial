## all_to_all_matmul

参考Ascend hccl高阶API：https://www.hiascend.com/document/detail/zh/canncommercial/900/API/ascendcopapi/atlasascendc_api_07_0875.html

### 理解

`mc2/allto_all_matmul/op_kernel/arch35/allto_all_kc_quant_matmul_arch35.h`

# ProcessTile 函数变量赋值的数学含义

## 数据维度约定

| 符号 | 含义 |
|------|------|
| `rankDim` | 分布式通信的 rank 数（专家/卡数） |
| `rankM` | 每个 rank 的 M 维行数 |
| `rankK` | 每个 rank 的 K 维列数 |
| `rankN` | 输出 N 维列数 |
| `tileM` | 主块每 tile 沿 M 轴切分的行数 |
| `tileCnt` | 主块 tile 迭代次数 |

**输入矩阵形状**：X1 `[rankDim·rankM, rankK]`，X2 `[rankDim·rankK, rankN]`
**输出矩阵形状**：Y `[rankM, rankN]`

## 复用基础量

| 变量 | 公式 | 数学含义 |
|------|------|----------|
| `tileMMultiRankK` | `tileM × rankK` | 一个 tile 中本地 X1 数据的元素总数：tileM 行 × rankK 列 |

## 通信上下文（AlltoAll）

整个计算的输入 X1 形状为 `[rankDim·rankM, rankK]`，通过 AlltoAll 将各 rank 的数据重分发，使每个 rank 汇聚属于自己的 rankM 行数据。

| 变量 | 公式 | 含义 |
|------|------|------|
| `taskCnt` | `tileCnt` | 主块流水迭代次数 |
| `sendBuffer` | `x1_` | 发送端基址：本地 X1 矩阵 |
| `recvBuffer` | `commOutGM_` | 接收端基址：workspace 中 AlltoAll 接收区 |
| `sendOffset` | `tileM·rankK·sizeof(X1)` | 相邻 tile 在发送缓冲中的字节间距：每 tile 发送 tileM·rankK 个元素 |
| `recvOffset` | `= sendOffset` | 相邻 tile 在接收缓冲中的字节间距（与发送对称） |
| `sendCount` | `tileM·rankK` | 每 tile 发送的元素总数 |
| `strideCount` | `rankM·rankK / rankDim` | AlltoAll 通信中，每个 rank 子块间的元素步长。X1 中每个 rank 拥有 rankM·rankK 个元素，跨 rankDim 个 rank 均分后步长为 rankM·rankK/rankDim |

## 量化上下文（动态量化 + 转置）

AlltoAll 收到的数据来自 rankDim 个 rank，每块 rankK 列，需先转置再按行做 per-token 动态量化，输出量化矩阵形状 `[tileM, rankDim·rankK]`（即 matmul 的左矩阵 A）。

| 变量 | 公式 | 含义 |
|------|------|------|
| `quantOutputScaleAddr` | `x1ScaleGM_` | 动态量化 scale 基址（每行一个 float） |
| `quantOutputScaleAddrOffset` | `tileM·sizeof(float)` | 相邻 tile 的 scale 字节步长：每 tile 有 tileM 行，每行 1 个 scale |
| `quantInputAddr` | `commOutGM_` | 量化输入基址 = AlltoAll 接收数据 |
| `quantInputAddrOffset` | `tileM·rankK·sizeof(X1)` | 相邻 tile 在量化输入的字节步长 |
| `quantOutputAddr` | `quantOutGM_` | 量化输出基址（量化后的 A 矩阵） |
| `quantOutputAddrOffset` | `tileM·rankK·rankDim·sizeof(X2)` | 相邻 tile 在量化输出的字节步长：每 tile 输出 tileM 行 × (rankDim·rankK) 列量化数据 |
| `rowNum` | `tileM` | 每 tile 量化的行数 |
| `colNumPerBlock` | `rankK` | 每个 rank 块的列数（AlltoAll 收到的数据按 rank 分块，每块 rankK 列） |
| `blockNum` | `rankDim` | 量化输入的 rank 块数（= rank 数） |
| `transposeDstAddr` | `transOutGM_` | 转置输出基址 |
| `transposeDstAddrOffset` | `tileM·rankK·rankDim·sizeof(X1)` | 相邻 tile 在转置输出的字节步长（与量化输出同大小，但 dtype 不同） |
| `nextBlockDataOffset` | `rankM·rankK/rankDim` | 量化输入中相邻 rank 块的元素偏移（= strideCount，数据在 AlltoAll 接收缓冲中的块间步长） |

## 计算上下文（QuantMatmul）

量化后的 A 矩阵 `[tileM, rankDim·rankK]` × X2 `[rankDim·rankK, rankN]` + bias → Y `[tileM, rankN]`

| 变量 | 公式 | 含义 |
|------|------|------|
| `aGM` | `quantOutGM_` | 左矩阵 A 基址 = 量化输出 |
| `bGM` | `x2_` | 右矩阵 B 基址 = 权重 X2 |
| `cGM` | `y_` | 输出 C 基址 = Y |
| `biasGM` | `bias_` | 偏置基址 |
| `aOffset` | `tileM·rankK·rankDim·sizeof(X2)` | 相邻 tile 在 A 矩阵的字节步长：A 每行有 rankDim·rankK 个量化元素 |
| `bOffset` | `0` | B 矩阵无偏移（X2 权重对所有 tile 相同，不随 M 轴 tile 移动） |
| `cOffset` | `tileM·rankN·sizeof(Y)` | 相邻 tile 在输出 Y 的字节步长：每 tile 输出 tileM·rankN 个元素 |
| `x1Scale` | `x1ScaleGM_` | A 矩阵的 per-row 动态量化 scale 基址 |
| `x1ScaleOffset` | `tileM·sizeof(float)` | 相邻 tile 的 x1Scale 字节步长 |
| `x2Scale` | `x2Scale_` | B 矩阵的静态量化 scale（per-channel/per-tensor） |
| `x2Offset` | `x2Offset_` | B 矩阵的量化零点 |
| `tilingDataPtr` | `mc2QuantMmTileTilingData` | 主块 matmul 的 tiling 参数指针 |

## 数学总结

整个流水线的数学过程为：

$$Y = \text{DequantA}(\text{AlltoAll}(X_1)) \times \text{DequantB}(X_2) + \text{bias}$$

- M 轴按 tileM 分块流水
- K 轴由 rankDim 个 rank 通过 AlltoAll 汇聚后拼接为 `rankDim·rankK`


---

## 

```cpp
template <typename quantInputDataType, typename quantOutputDataType, bool isSmallK>
__aicore__ inline void
Fp8DynamicQuantPertoken<quantInputDataType, quantOutputDataType, isSmallK>::ProcessOneTokenSmallK()
{
    // 空间大小和偏移计算，计算每个K块的实际字节长度（K维每块元素数 * 每个元素字节大小）
    uint32_t perBlockDataLength = static_cast<uint32_t>(context_.colNumPerBlock * sizeof(quantInputDataType));
    // 每块字节大小对齐到32字节，因为UB空间分配必须按照32字节对齐
    uint32_t inputSizePerBlock = Ceil(perBlockDataLength, UB_DATABLOCK) * UB_DATABLOCK;
    // 计算输入占用的总的UB空间
    uint32_t inputSize = inputSizePerBlock * context_.blockNum;

    constexpr uint8_t tempRShiftAmountIn = sizeof(quantInputDataType) / 2;
    // 每块在UB中元素偏移（字节对齐大小 >> 1 = fp16元素数），用于定位各块在UB内的起始位置
    // 这里perBlockDataOffsetInUB的单位是bit（elementwise），存储的是元素个数
    // 这里是要把bit转为byte，所以除以一个sizeof(fp16)，等于右移sizeof(quantInputDataType)/2
    uint64_t perBlockDataOffsetInUB = inputSizePerBlock >> tempRShiftAmountIn;

    // 计算输出fp8数据的UB空间，并做32字节对齐
    uint32_t outputSizePerBlock =
        Ceil(static_cast<uint32_t>(context_.colNumPerBlock * sizeof(quantOutputDataType)), UB_DATABLOCK) * UB_DATABLOCK;
    uint32_t outputSize = outputSizePerBlock * context_.blockNum;
    // 转换单位，从bit转为byte
    constexpr uint8_t tempRShiftAmountOut = sizeof(quantOutputDataType) / 2;
    uint64_t perBlockQuantDataOffsetInUB = outputSizePerBlock >> tempRShiftAmountOut;

    // 空间分配
    // 分配32字节（UB按照32字节对齐，这里最少分配32字节），用于存储每行最大绝对值（1个float）
    tPipe_->InitBuffer(maxValueBuf_, UB_DATABLOCK);
    // 分配量化系数（scale）的缓冲区buf，存放当前核所有行的scale，并对齐到32字节
    tPipe_->InitBuffer(quantScaleBuf_,
                       Ceil(static_cast<uint32_t>(this->rowsThisCore_ * sizeof(float)), UB_DATABLOCK) * UB_DATABLOCK);

    // 双缓冲：分配2组输入缓冲和2组输出缓冲（ping-pong），用于流水线重叠
    for (uint32_t i = 0; i < TWO_FACTOR; ++i) {
        tPipe_->InitBuffer(rawInputBuf_[i], inputSize);
        tPipe_->InitBuffer(quantOutputBuf_[i], outputSize);
    }

    // 从TBuf中获取LocalTensor，后续通过索引 0/1切换 ping-pong buffer
    LocalTensor<float> maxValueData = maxValueBuf_.Get<float>();
    LocalTensor<float> coreQuantScales = quantScaleBuf_.Get<float>();

    LocalTensor<quantInputDataType> rawInputTensor[2];
    LocalTensor<quantOutputDataType> quantOut[2];

    for (uint32_t i = 0; i < TWO_FACTOR; ++i) {
        rawInputTensor[i] = rawInputBuf_[i].Get<quantInputDataType>();
        quantOut[i] = quantOutputBuf_[i].Get<quantOutputDataType>();
    }

    /* 为双缓冲的搬入、搬出（量化搬出）、转置搬出各分配2个event，用于硬事件同步 */
    // 申请event同步数据搬入
    AscendC::TEventID copyInEvent[2];
    copyInEvent[0] = tPipe_->AllocEventID<AscendC::HardEvent::MTE2_S>();
    copyInEvent[1] = tPipe_->AllocEventID<AscendC::HardEvent::MTE2_S>();
    // 申请event同步量化数据搬出
    AscendC::TEventID quantOutEvent[2];
    quantOutEvent[0] = tPipe_->AllocEventID<AscendC::HardEvent::MTE3_S>();
    quantOutEvent[1] = tPipe_->AllocEventID<AscendC::HardEvent::MTE3_S>();
    // 申请event同步permute数据搬出
    AscendC::TEventID transOutEvent[2];
    transOutEvent[0] = tPipe_->AllocEventID<AscendC::HardEvent::MTE3_S>();
    transOutEvent[1] = tPipe_->AllocEventID<AscendC::HardEvent::MTE3_S>();

    // 数据搬入相关参数
    // GM中非连续块之间的字节间距，nextBlockDataOffset是下一个块在GM中的起始偏移，减去当前块长度即为stride
    uint32_t gmStrideBytes =
        static_cast<uint32_t>((context_.nextBlockDataOffset - context_.colNumPerBlock) * sizeof(quantInputDataType));
    // UB中块与块之间的padding块数（对齐导致的间距，以32字节为单位）
    uint32_t ubStrideBlocks = (inputSizePerBlock - perBlockDataLength) / UB_DATABLOCK;
    // 构造DataCopyPad参数：blockNum个块，每块长度，GM stride，UB stride，以及保留字段0。准备从GM搬入UB
    DataCopyExtParams inputCopyParams = {static_cast<uint16_t>(context_.blockNum), perBlockDataLength, gmStrideBytes,
                                         ubStrideBlocks, 0};
    // 首轮数据搬入
    // 当前核负责的第0行 * 一行非连续块的个数
    uint64_t globalInputIdx0 = this->startRowThisCore_ * context_.colNumPerBlock;
    // 即将第0行所有的非连续块一次性搬入 buffer_0，{true, 0, 0, 0}表示做padding
    DataCopyPad<quantInputDataType>(rawInputTensor[0], quantInputGM_[globalInputIdx0], inputCopyParams,
                                    {true, 0, 0, 0});
    // 设置搬入完成标志，后续WaitFlag等待此事件
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(copyInEvent[0]);
    // 保证循环中同步配对
    // 预先设置“假”的MTE3_S标志，保证循环首次WaitFlag时不会死锁（因为首轮还没有搬出操作）
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(transOutEvent[1]);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(quantOutEvent[0]);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(quantOutEvent[1]);

    for (uint64_t r = 0; r < this->rowsThisCore_; ++r) {
        // 第几行
        uint64_t globalRowIdx = this->startRowThisCore_ + r;
        // 每一行开始的索引
        uint64_t globalInputIdx = globalRowIdx * context_.colNumPerBlock;
        // 每一行结束的索引
        uint64_t globalOutIdx = globalRowIdx * context_.colNumPerBlock * context_.blockNum;
        // 交替使用buffer 0/1
        uint32_t bufIdx = r % 2;
        uint32_t nextBufIdx = (r + 1) % 2;

        // 提前启动下一轮搬入
        // 等待nextBuf上一次转置搬出完成，确保nextBuf可用（不会与新的搬入冲突）
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(transOutEvent[nextBufIdx]);

        if (r + 1 < this->rowsThisCore_) {
            // 下一行开始的索引
            uint64_t nextGlobalInputIdx = globalInputIdx + context_.colNumPerBlock;
            // 预取下一行：将下一行一次性搬入buffer
            DataCopyPad<quantInputDataType>(rawInputTensor[nextBufIdx], quantInputGM_[nextGlobalInputIdx],
                                            inputCopyParams, {true, 0, 0, 0});
            // 设置搬入完成标志，后续WaitFlag等待此事件
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(copyInEvent[nextBufIdx]);
        }
        // 启动本轮permute搬出
        // 等待当前行数据搬入完成
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(copyInEvent[bufIdx]);

        // 如果需要转置输出
        if (this->isTransOut_) {
            // 将原始fp16数据搬出到transOutGM（按非连续块组织），blockNum个块，每块长度，UB stride，GM stride，以及保留字段0
            DataCopyPad<quantInputDataType, PaddingMode::Normal>(
                transOutGM_[globalOutIdx], rawInputTensor[bufIdx],
                {static_cast<uint16_t>(context_.blockNum), perBlockDataLength, ubStrideBlocks, 0, 0});
        }
        // 设置搬出完成标志
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(transOutEvent[bufIdx]);

        // 求最大值
        // 输入数据当前这一行的首地址
        __local_mem__ quantInputDataType *xAddr =
            (__local_mem__ quantInputDataType *)rawInputTensor[bufIdx].GetPhyAddr();
        // 最大值存放地址
        __local_mem__ float *maxValueAddr = (__local_mem__ float *)maxValueData.GetPhyAddr();
        // K轴总长度 = block块数 * UB中每块的数据个数
        uint64_t totalKSize = context_.blockNum * perBlockDataOffsetInUB;
        // 微指令API，计算该行所有数据的最大值
        CalculateMaxRegBase(xAddr, totalKSize, maxValueAddr);
        // V_S同步确保向量计算完成，读取结果
        SyncFunc<AscendC::HardEvent::V_S>();
        // 获取偏移地址是0位置上的值
        float maxValue = maxValueData.GetValue(0);

        // 求量化系数
        // 最大值 * 倒数
        float scale = maxValue * this->recipFP8MaxLimit_;
        // 量化系数（scale）的缓冲区buf，从TBuf中获取到LocalTensor coreQuantScales，并写入r位置上的scale缓冲区
        coreQuantScales.SetValue(r, scale);
        // S_V同步确保scaler写入完成
        SyncFunc<AscendC::HardEvent::S_V>();

        // 做量化
        // 等待该buffer上一次量化搬出完成
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(quantOutEvent[bufIdx]);

        __local_mem__ quantOutputDataType *yAddr = (__local_mem__ quantOutputDataType *)quantOut[bufIdx].GetPhyAddr();
        // 逐块调用RegBase量化API
        for (uint32_t k = 0; k < context_.blockNum; ++k) {
            DoQuantRegBase(xAddr + k * perBlockDataOffsetInUB, yAddr + k * perBlockQuantDataOffsetInUB,
                           perBlockDataOffsetInUB, scale);
        }
        // V_S同步确保量化计算完成
        SyncFunc<AscendC::HardEvent::V_S>();

        // 量化结果搬出
        // 将量化后的fp8结果搬出到GM
        DataCopyPad<quantOutputDataType, PaddingMode::Normal>(
            quantOutputGM_[globalOutIdx], quantOut[bufIdx],
            {static_cast<uint16_t>(context_.blockNum),
             static_cast<uint32_t>(context_.colNumPerBlock * sizeof(quantOutputDataType)), 0, 0, 0});
        // 设置搬出完成标志
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(quantOutEvent[bufIdx]);
    }

    // 搬出scales
    DataCopyExtParams outScaleParams = {1, static_cast<uint32_t>(this->rowsThisCore_ * sizeof(float)), 0, 0, 0};
    // 将本核的所有行scale一次性从UB搬出到GM
    DataCopyPad<float, PaddingMode::Normal>(quantOutputScaleGM_[this->startRowThisCore_], coreQuantScales,
                                            outScaleParams);
    // MTE3_S同步，确保scale搬出完成
    SyncFunc<AscendC::HardEvent::MTE3_S>();

    // 保证循环中同步配对
    // 等待最后的搬出操作完成
    uint64_t lastRow = this->rowsThisCore_ - 1;
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(quantOutEvent[0]);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(quantOutEvent[1]);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(transOutEvent[lastRow % 2]);

    // 释放event
    tPipe_->ReleaseEventID<AscendC::HardEvent::MTE2_S>(copyInEvent[0]);
    tPipe_->ReleaseEventID<AscendC::HardEvent::MTE2_S>(copyInEvent[1]);
    tPipe_->ReleaseEventID<AscendC::HardEvent::MTE3_S>(quantOutEvent[0]);
    tPipe_->ReleaseEventID<AscendC::HardEvent::MTE3_S>(quantOutEvent[1]);
    tPipe_->ReleaseEventID<AscendC::HardEvent::MTE3_S>(transOutEvent[0]);
    tPipe_->ReleaseEventID<AscendC::HardEvent::MTE3_S>(transOutEvent[1]);
}

template <typename quantInputDataType, typename quantOutputDataType, bool isSmallK>
__aicore__ inline void
Fp8DynamicQuantPertoken<quantInputDataType, quantOutputDataType, isSmallK>::ProcessOneTokenLargeK()
{
    // 计算每个K块的实际字节长度（K维每块元素数 * 每个元素字节大小）
    uint32_t perBlockDataLength = static_cast<uint32_t>(context_.colNumPerBlock * sizeof(quantInputDataType));
    // 每个字节块大小对齐到32字节，UB空间是按照32B对齐的
    uint32_t inputSizePerBlock = Ceil(perBlockDataLength, UB_DATABLOCK) * UB_DATABLOCK;
    // 与SmallK相同的计算逻辑，但这里的inputSize是所有块的总UB空间（所有块放同一个buffer）
    uint32_t inputSize = inputSizePerBlock * context_.blockNum;
    constexpr uint8_t tempRShiftAmountIn = sizeof(quantInputDataType) / 2;
    // bit转byte，计算输入占用的总的UB空间
    uint64_t perBlockDataOffsetInUB = inputSizePerBlock >> tempRShiftAmountIn;

    // 注意：这里用的是perBlockDataOffsetInUB（元素数，elementwise），而非colNumPerBlock
    // 因为量化输出的对齐基于输入的元素偏移
    uint32_t outputSizePerBlock =
        Ceil(static_cast<uint32_t>(perBlockDataOffsetInUB * sizeof(quantOutputDataType)), UB_DATABLOCK) * UB_DATABLOCK;
    // outputSize也是用的元素数 * 块数
    uint32_t outputSize = perBlockDataOffsetInUB * context_.blockNum;
    constexpr uint8_t tempRShiftAmountOut = sizeof(quantOutputDataType) / 2;
    // bit转byte
    uint64_t perBlockQuantDataOffsetInUB = outputSizePerBlock >> tempRShiftAmountOut;

    tPipe_->InitBuffer(maxValueBuf_, UB_DATABLOCK);
    tPipe_->InitBuffer(quantScaleBuf_,
                       Ceil(static_cast<uint32_t>(this->rowsThisCore_ * sizeof(float)), UB_DATABLOCK) * UB_DATABLOCK);
    // 注意：这里给的TQue，非TBuf，是因为整行数据需要全部在UB中才能做全局max，不适合ping-pong，所以这里给的也是ONE_FACTOR
    tPipe_->InitBuffer(rawInputQue_, ONE_FACTOR, inputSize);
    tPipe_->InitBuffer(quantOutputQue_, ONE_FACTOR, outputSize);
    // 分配local tensor
    LocalTensor<float> maxValueData = maxValueBuf_.Get<float>();
    LocalTensor<float> coreQuantScales = quantScaleBuf_.Get<float>();

    // 分配local tensor
    LocalTensor<quantInputDataType> rawInputTensor = rawInputQue_.AllocTensor<quantInputDataType>();
    LocalTensor<quantOutputDataType> quantOut = quantOutputQue_.AllocTensor<quantOutputDataType>();
    // 然后入队，通过Que管理buffer周期
    transOutQue_.EnQue(rawInputTensor);
    quantOutputQue_.EnQue(quantOut);

    // 逐token处理
    float scale;
    float maxValue;
    float maxValuePerRank;
    // 申请两个event管理非连续block数据的并行搬入
    // 块K和块K+2交替使用event
    AscendC::TEventID rawInputEvent[2];
    rawInputEvent[0] = tPipe_->AllocEventID<AscendC::HardEvent::MTE2_S>();
    rawInputEvent[1] = tPipe_->AllocEventID<AscendC::HardEvent::MTE2_S>();
    // 
    for (uint64_t r = 0; r < this->rowsThisCore_; ++r) {
        uint64_t globalRowIdx = this->startRowThisCore_ + r;
        uint64_t globalInputIdx = globalRowIdx * context_.colNumPerBlock;
        // 遍历每行，从中取出rawInputTensor（确保buffer可用）
        transOutQue_.DeQue();
        // 提前启动数据搬运
        // 搬入第0块：连续搬入（1块，stride=0），不做padding({false, ...})
        DataCopyPad<quantInputDataType>(rawInputTensor, quantInputGM_[globalInputIdx], {1, perBlockDataLength, 0, 0, 0},
                                        {false, 0, 0, 0});
        // 设置MTE2到Scalar的标志
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(rawInputEvent[0]);
        // 搬入第1块：非连续地址，从GM的nextBlockDataOffset处搬入，放到UB的第1块偏移位置
        DataCopyPad<quantInputDataType>(rawInputTensor[perBlockDataOffsetInUB],
                                        quantInputGM_[globalInputIdx + context_.nextBlockDataOffset],
                                        {1, perBlockDataLength, 0, 0, 0}, {false, 0, 0, 0});
        // 设置MTE2到Scalar的标志
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(rawInputEvent[1]);

        // 非连续K循环处理
        __local_mem__ quantInputDataType *xAddr = (__local_mem__ quantInputDataType *)rawInputTensor.GetPhyAddr();
        __local_mem__ float *maxValueAddr = (__local_mem__ float *)maxValueData.GetPhyAddr();
        // 逐块计算最大绝对值，初始化maxValue为0
        maxValue = 0.0f;
        maxValuePerRank = 0.0f;
        // 非连续K块循环处理
        for (uint32_t k = 0; k < context_.blockNum; ++k) {
            // 等待第K块搬入完成（event0和event1交替使用）
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(rawInputEvent[k % 2]);
            // 提前启动数据搬运
            // 预取K+2块：在计算第K块的同时，提前搬入K+2块（利用同一个event交替，实现搬入与计算重叠
            uint32_t nextK = k + 2;
            if (nextK < context_.blockNum) {
                DataCopyPad<quantInputDataType, PaddingMode::Normal>(
                    rawInputTensor[nextK * perBlockDataOffsetInUB],
                    quantInputGM_[globalInputIdx + nextK * context_.nextBlockDataOffset],
                    {1, perBlockDataLength, 0, 0, 0}, {false, 0, 0, 0});
                // 注意重新SetFlag的是rawInputEvent[k % 2]，因为前面已经WaitFlag消耗掉了
                AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(rawInputEvent[k % 2]);
            }

            // 计算最大绝对值
            // 对第K块计算局部最大绝对值，取全局max（跨所有K块的最大值）
            CalculateMaxRegBase(xAddr + k * perBlockDataOffsetInUB, context_.colNumPerBlock, maxValueAddr);
            SyncFunc<AscendC::HardEvent::V_S>();
            maxValuePerRank = maxValueData.GetValue(0);
            maxValue = Max(maxValue, maxValuePerRank);
        }
        // 转置结果搬出
        // 所有块搬入完毕后，将整行原始数据搬出到转置输出GM
        uint64_t globalOutIdx = globalRowIdx * context_.colNumPerBlock * context_.blockNum;
        if (this->isTransOut_) {
            DataCopyPad<quantInputDataType, PaddingMode::Normal>(
                transOutGM_[globalOutIdx], rawInputTensor,
                {static_cast<uint16_t>(context_.blockNum), perBlockDataLength, 0, 0, 0});
        }
        // 然后EnQue释放rawInputTensor使用权
        transOutQue_.EnQue(rawInputTensor);

        // 计算量化系数
        scale = maxValue * this->recipFP8MaxLimit_;
        coreQuantScales.SetValue(r, scale);
        SyncFunc<AscendC::HardEvent::S_V>();

        // 量化
        // Deque取出quantOut Buffer，逐块做量化（此时 rawInputTensor 仍在UB中，数据完整可用）
        quantOutputQue_.DeQue();
        __local_mem__ quantOutputDataType *yAddr = (__local_mem__ quantOutputDataType *)quantOut.GetPhyAddr();
        for (uint32_t k = 0; k < context_.blockNum; ++k) {
            DoQuantRegBase(xAddr + k * perBlockDataOffsetInUB, yAddr + k * perBlockQuantDataOffsetInUB,
                           perBlockDataOffsetInUB, scale);
        }
        SyncFunc<AscendC::HardEvent::V_S>();
        // 搬出量化结果，将量化结果搬出到UB
        DataCopyPad<quantOutputDataType, PaddingMode::Normal>(
            quantOutputGM_[globalOutIdx], quantOut,
            {static_cast<uint16_t>(context_.blockNum),
             static_cast<uint32_t>(context_.colNumPerBlock * sizeof(quantOutputDataType)), 0, 0, 0});
        // EnQue释放quantOut buffer
        quantOutputQue_.EnQue(quantOut);
    }
    // 搬出量化系数scale
    DataCopyExtParams outScaleParams = {1, static_cast<uint32_t>(this->rowsThisCore_ * sizeof(float)), 0, 0, 0};
    DataCopyPad<float, PaddingMode::Normal>(quantOutputScaleGM_[this->startRowThisCore_], coreQuantScales,
                                            outScaleParams);
    SyncFunc<AscendC::HardEvent::MTE3_S>();
    // 释放event
    tPipe_->ReleaseEventID<AscendC::HardEvent::MTE2_S>(rawInputEvent[0]);
    tPipe_->ReleaseEventID<AscendC::HardEvent::MTE2_S>(rawInputEvent[1]);
    // Deque + FreeTensor清理Que资源
    transOutQue_.DeQue();
    quantOutputQue_.DeQue();
    rawInputQue_.FreeTensor(rawInputTensor);
    quantOutputQue_.FreeTensor(quantOut);
}
```