## all_to_all_matmul

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