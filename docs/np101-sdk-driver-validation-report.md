# NP101 SDK 与驱动验证报告

本文使用 `v105` 指代驱动及配套 SDK 安装包 `1.0.5`，使用 `v106` 指代 `1.0.6`。测试输出目录名和后文（包括表格）均使用这两个简称。`v106` 修复了此前的内存申请上限问题。环境安装见 [README.md](../README.md)，测试入口见 [tests/README.md](../tests/README.md)。

## 复现准备

配套 SDK 安装完成，且当前账号可读写 `/dev/galcore` 后，在仓库根目录执行：

```bash
conda activate SpecFerry
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -j 4
```

每次只运行一个板卡测试，并使用新的输出目录。示例目录名默认使用 `v105`；测试 `v106` 时将目录名中的 `v105` 改为 `v106`。P5 超过 1 GiB 的探测命令用于 `v106`；`v105` 使用不超过 1 GiB 的目标检查数据读回。

P2 的 FCL、P4 的三维缓存崩溃测试排在其他测试之后；其中一个崩溃后需恢复设备，再运行另一个。程序崩溃、超时或内核报告板卡卡死后，先恢复设备再继续；测试脚本会记录异常并阻止继续访问板卡。P3 复现前需冷重启主机和 NP101，热重启未能消除此前的卡死问题。

下文的建图、验图、执行，分别对应 `vsi_nn_SetupGraph`、`vsi_nn_VerifyGraph`、`vsi_nn_RunGraph`。验图发生在执行之前；验图失败时，不能认为图已经完成过计算。

P1–P9 的独立复现命令使用合成数据，不需要下载模型；P10 需要按主 README 下载并导出 OPT-350M。每条命令只运行指定测试，不会自动执行其他问题的复现。

## 测试环境

| 安装包 | 驱动模块 `galcore` 的 `srcversion` | SDK 与环境记录 |
|---|---|---|
| `v105` | `F53A3E6EFBB7D0FD7D4617C` | [环境及库校验值][rollback-environment] |
| `v106` | `8D46803809F15FA0FD98A92` | [环境及库校验值][v106-environment] |

SDK 动态库位于 `/usr/lib/ljmicro`。运行记录包含命令、系统启动编号（`boot ID`）、测试程序和 SDK 库的 `SHA256`，用于确认实际测试的版本。驱动模块输出的 SDK 版本字符串不等于安装包版本号。

两版本测试使用相同的图和显式参数，但并非全部使用同一可执行文件：`v105` 的程序整理过文件读写代码，并增加了计时。P1 中原厂程序和仓库卷积测试的结果分别列出，避免混用计时。单节点 `SELECT` 对比使用同一可执行文件，两版输入和输出文件逐字节一致。`v106` 冷启动测试的驱动模块及五个 SDK 库的校验值与此前记录一致，见[环境记录][cold106-environment]。

P5 两版本复测使用同一可执行文件，块大小、数据生成方式和扫描逻辑相同。`v105` 补测前，板卡完全断电，主机保持供电；系统 `boot ID` 已变化。驱动模块及五个 SDK 库的校验值与此前 `v105` 记录一致，见[补测环境][memory105-environment]。

## 需要反馈的问题

### 建图、验图崩溃或卡死

| 编号 | 问题 | `v106` | `v105` | 需要排查或说明的内容 |
|---|---|---|---|---|
| [P1](#p1卷积示例在新驱动下验图崩溃) | 压缩包中的卷积示例代码在新驱动下验图崩溃 | 验图时出现 `SIGFPE` | 原厂示例通过；仓库卷积测试验图耗时 42.478 ms | 驱动更新后出现新的验图错误，是否由新驱动或配套库的修改引入 |
| [P2](#p2带偏置的全连接层验图崩溃拆成矩阵乘法和加法后通过) | 带 `bias` 的全连接层 FCL 验图崩溃 | 验图时出现 `SIGSEGV`；拆成 `MatMul + Add` 完成同一计算则通过 | 结果与 `v106` 相同 | FCL 是否存在内部实现错误，或该配置是否缺少必要参数 |
| [P3](#p3单节点select图验图时驱动报告板卡卡死) | 单节点（`SELECT`）图验图时，驱动报告板卡卡死；该算子用于 LLM 词向量查表，选择 token 所属词表分块的结果 | 冷启动后，单节点 `SELECT` 验图 28.982 s，出现 `NP hang`，自动恢复后数值正确；历史 `GATHER` 也发生过 | 同一 `SELECT` 程序验图 9.228 ms，无 `NP hang`；selection 的 23 项检查也通过 | 冷启动后仍发生，且不局限于 `GATHER`；需排查验图触发的驱动等待及恢复 |
| [P4](#p4向三维-kv-缓存追加数据时建图崩溃) | 向预分配的三维 KV 缓存追加一个 token 的数据时，建图崩溃 | 三维追加与 attention 放在同一张图中时建图崩溃；二维追加及 attention 测试通过 | 结果与 `v106` 相同 | 需确认三维追加的参数是否受支持，以及建图失败时为何直接崩溃 |

### 数据读回或计算结果错误

| 编号 | 问题 | `v106` | `v105` | 需要排查或说明的内容 |
|---|---|---|---|---|
| [P5](#p5大载荷完整扫描中出现局部读回错误) | 接近申请上限时，完整读回发现局部数据错误 | 完整扫描 2840 MiB，只有三个局部区段出错，共 24 字节 | 完整扫描 1016 MiB，也在相同位置发现 24 字节错误 | 两版本均存在；需排查这些位置为何读回错误 |
| [P6](#p6fp16-随机采样返回越界索引) | FP16 随机采样返回越界索引 | 合法索引为 0–7，FP16 全部返回 8；FP32 和先转 FP32 再采样均通过 | 结果与 `v106` 相同 | FP16 采样实现是否存在问题 |
| [P7](#p7用加法合并查表结果时fp16-极小值变成-0) | 用 `ADD` 合并查表结果时，FP16 极小值变成 0 | 单节点 `ADD` 将 10 个非正规数全部置零，其余 6 个值正确；`SELECT` 数值正确但触发 P3 | 同一 `ADD` 程序的输入、输出逐字节相同；`SELECT` 数值正确且无卡死 | 需说明 `ADD` 的置零规则及是否有保留非正规数的设置 |

### 内存占用与释放

| 编号 | 问题 | `v106` | `v105` | 需要排查或说明的内容 |
|---|---|---|---|---|
| [P8](#p8申请-8-mib-张量sdk-内存计数增加约-16-mib) | 申请 8 MiB 张量，SDK 内存计数增加约 16 MiB | 增量为 `2 × 张量字节数 + 4672` | 结果与 `v106` 相同 | 为什么这样计数；是否实际占用双份板上内存，是否会影响可用容量或造成泄漏、越界写入 |
| [P9](#p9反复修改图的缓冲区绑定后主机内存持续增长) | 反复修改图的缓冲区绑定并验图，主机内存持续增长 | 运行 128 轮，主机内存约增加 16 MiB；不改绑定时稳定 | 结果与 `v106` 相同 | 每次重新绑定和验图是否留下未释放数据；若只是主机分配器保留，何时能够回收 |

### 验图耗时

| 编号 | 问题 | `v106` | `v105` | 需要排查或说明的内容 |
|---|---|---|---|---|
| [P10](#p10单层图验图需要两分多钟) | 单层图验图耗时高 | 612 个节点验图约 158 s | 验图 140.917 s；16 项输入和 KV 缓存检查通过 | 是否有不恰当的函数调用或图构造方式，导致验图需要两分多钟 |

## 反馈问题的复现方法

### 建图、验图崩溃或卡死

#### P1：卷积示例在新驱动下验图崩溃

**问题表现：** `v106` 下，原厂卷积示例的计时副本在 `vsi_nn_VerifyGraph` 中收到 `SIGFPE`。故障位于 `libNNArchPerf.so` 的 `NNTransposeCycleCount_V9` 整数除法指令。建图耗时 0.433 ms，验图未返回，还没有执行卷积。[计时日志][v106-conv]包含副本改动和程序校验值。

**测试计算什么：** [conv_relu_pool_test.cpp](../tests/native/np101/conv_relu_pool_test.cpp) 执行 FP16 卷积、ReLU、最大池化，并与 CPU 结果比较。输入 `[8,8,3,1]`、卷积核 `[3,3,3,4]`、`bias` 为 `[4]`、输出 `[4,4,4,1]`，共 3 个节点、6 个张量，随机种子为 42。中间张量的形状由 SDK 推导，不清空或覆盖 SDK 初始化的节点私有数据。

**复现命令：**

```bash
python scripts/check_np101.py conv --sdk-timing --timeout 60 --output .cache/runs/v105-conv
```

仓库测试与原厂示例使用相同尺寸和算子顺序，额外检查参数并更换输入再算一轮。两轮最大绝对误差都不超过 0.1，且资源释放成功，才判定通过。

**两版本区别：** 回滚到 `v105` 后，仓库测试验图耗时 42.478 ms，两次执行合计 0.185 ms，最大误差分别为 0.003234、0.003114。未改动的原厂程序也通过：64 个输出均在误差范围内，总耗时 0.167 s。原厂程序没有单独记录验图时间，42.478 ms 来自仓库测试。见[仓库测试结果][rollback-conv]和[原厂结果][rollback-vendor]。

冷启动后，仓库卷积测试再次通过，两轮误差不变，验图耗时 349.972 ms，测试进程总耗时 0.839 s，正常释放资源，见[冷启动结果][cold-conv]。

**需要排查：** 新驱动或配套 SDK 的哪些修改导致这张卷积图验图崩溃。

#### P2：带偏置的全连接层验图崩溃，拆成矩阵乘法和加法后通过

**问题表现：** 两版本的 FCL 都在验图时收到 `SIGSEGV`，还未进入计算。同样的输入、权重和 `bias` 拆成矩阵乘法和加法后，两轮结果均正确。

| 版本 | FCL | 矩阵乘法加偏置 |
|---|---|---|
| `v106` | 建图 2.174 ms，随后验图崩溃 | 验图 31.196 ms，两轮结果与预期完全一致 |
| `v105` | 建图 0.164 ms，随后验图崩溃 | 验图 9.355 ms，两轮结果与预期完全一致 |

**测试计算什么：** [projection_check.cpp](../tests/native/np101/projection_check.cpp) 计算 `Y = X × Wᵀ + bias`。FP16 输入为 `[8,1]`，权重为 `[8,8]` 单位矩阵，8 个 `bias` 值均为 0.25。FCL 参数为 `weights=8, axis=0`；拆开时使用 `MATRIXMUL`、调整 `bias` 形状的 `RESHAPE` 和 `ADD`。两轮输入分别全为 1、全为 -0.5，期望输出分别全为 1.25、全为 -0.25。输入在验图前写入，张量引用均有效，不覆盖 SDK 私有节点数据。

**复现命令：** 先确认拆开计算正常：

```bash
python scripts/check_np101.py projection --operator matmul-add --sdk-timing --timeout 60 \
  --output .cache/runs/v105-projection-control
```

完成其他测试后，再单独测试 FCL：

```bash
python scripts/check_np101.py projection --operator fcl --sdk-timing --timeout 60 \
  --output .cache/runs/v105-projection-fcl
```

**崩溃位置：** `v106` 的[调用栈][v106-fcl-stack]包含 `vxoGraphOptimization_getKernelType`、`vxoGraphOptimization_ConvertMaxPool2Conv` 和 `vxVerifyGraph`，见[完整结果][v106-projection]。`v105` 的内核日志记录 `libOpenVX.so.1` 读取地址 `0xb0` 时崩溃；结合库文件可定位到 `vxoGraphOptimization_getKernelType+0x5d2`。指令为 `mov 0xb0(%rax),%rax`，符合内部空指针加偏移后被读取的特征。`v105` 未获取完整调用栈。见[结果][rollback-fcl]、[故障位置][rollback-symbolication]和[内核摘要][rollback-kernel]。

**当前处理与待排查问题：** 使用 `MatMul + Add` 完成同一计算。需确认 FCL 是否有内部错误，或上述形状、精度、`bias` 配置是否还需要其他参数。现有结果不能证明是 `bias` 单独引起，也不能推断所有 FCL 配置都会失败。

#### P3：单节点（`SELECT`）图验图时，驱动报告板卡卡死

**使用场景：** 构建 LLM 的词向量查表负载时，词表分块存储，`SELECT` 根据 token ID 所属范围选择对应分块的查表结果，避免用 FP16 加法合并结果时将极小值置零（见 P7）。

**冷启动后的单节点复现：** `v106` 冷启动后运行的第一个测试只有一个 `SELECT` 节点，不含 `GATHER`，不加载模型。验图耗时 28.982 s，内核出现：

```text
09:58:12.701523 [galcore]: NP[0] hang, automatic recovery.
09:58:12.709248 [galcore]: recovery done
```

恢复后计算结果逐字节正确，程序正常释放并退出。测试进程总耗时 29.217 s，实际 `RunGraph` 仅 0.253 ms。见[SDK 计时][cold106-select-timing]、[内核日志][cold106-kernel-log]和[对比汇总][cold106-summary]。这次结果说明冷启动未消除故障，卡死也不局限于词表查询或 `GATHER`。

**这张图包含什么：** 使用 P7 的 [fp16_preservation_check.cpp](../tests/native/np101/fp16_preservation_check.cpp)。两个 FP16 输入、一个 FP16 输出、一个 BOOL8 条件张量，形状均为 `[16,1]`；4 个张量均为非 `const`，应用数据合计 112 字节。所有输入在验图前写入，条件张量交替选择左右输入中的非零值。

| 同一 `SELECT` 程序 | 验图 | 测试进程总耗时 | 数值与内核结果 |
|---|---|---|---|
| `v105` 冷启动后 | 9.228 ms | 0.367 s | 16 个值正确；无 `NP hang` |
| `v106` 冷启动后 | 28.982 s | 29.217 s | 16 个值正确；出现一次 `NP hang` 和自动恢复 |

两版可执行文件 `SHA256` 相同，输入数据相同。`v105` 的 `SELECT` 在卷积通过后运行，`v106` 的 `SELECT` 是冷启动后的首个测试，测试顺序并非完全相同。尚不能确定故障在驱动还是配套 SDK，也不能据此推断所有 `SELECT` 调用都会卡死。

**单节点复现命令：** 冷启动后单独运行，并同时检查内核日志：

```bash
python scripts/check_np101.py fp16-preservation --operator select --sdk-timing --timeout 30 \
  --output .cache/runs/v106-fp16-select
```

**此前的词表查询结果：** `v106` 热重启后，`selection` 测试仍出现 `NP hang/recovery`（板卡卡死及驱动自动恢复）。第一次验图耗时 29.429 s；自动恢复后，七组共 23 项数值检查全部通过，正常释放资源，总耗时 29.921 s。其他运行曾直接超时。见[程序结果][v106-selection]和[内核日志][v106-selection-kernel]。

**`v105` 冷启动复测：** 7 组共 23 项检查全部通过，正常释放。首张 `GATHER` 图验图 10.890 ms；六次验图分别为 10.890、8.791、8.695、30.397、6.705、140.460 ms。原生测试进程耗时 1.576 s，不包含 Python 准备参考数据的时间。对应内核日志没有 `NP hang` 或自动恢复记录，见[结果][cold-selection]和[日志摘要][cold-kernel]。上述 `GATHER` 对比仍是热重启与冷启动的差异；本轮 `v106` 在前置 `SELECT` 已发生卡死，因此未继续运行原 selection 测试。

**卡在哪张图：** `selection` 的第一张图只有一个 `GATHER(axis=1)` 节点。根据 INT32 `[1]` 的索引，从 FP16 `[8,8]` 表中取一行，输出 FP16 `[8,1]`。该功能用于根据 token ID 读取词向量。另一个小型取行测试曾约 39 ms 通过，因此不能认为所有取行图都必然卡住。

**复现命令：** 先冷重启主机和 NP101，再运行：

```bash
python scripts/check_np101_model.py selection --sdk-timing --timeout 60 \
  --output .cache/runs/v105-selection
```

测试后同时检查内核日志；输出正确也可能已经发生过卡死和自动恢复。

**其他尝试：** 历史 `selection` 测试在执行前上传索引；独立 `lookup` 测试的 `--operator gather`、`--operator embedding` 在验图前写入索引，两者的写入时机不同。换用 `EMBEDDING_LOOKUP` 曾在设备已经异常的启动周期中于验图时崩溃，尚不能认为这个算子可以绕开问题。

**需要排查：** 冷启动后的单节点图为什么在验图期间触发驱动卡死。最新用例不含索引查询且输入已在验图前写入，不能只从 `GATHER` 或索引写入时机解释。SDK 返回成功、输出正确和正常释放均不能替代内核日志检查。

#### P4：向三维 KV 缓存追加数据时，建图崩溃

**为什么使用三维缓存：** 自回归生成时，新 token 的 attention 要读取此前 token 的 K、V，并保存本次新增的 K、V，供后续 token 使用。对一层、一个序列，K 和 V 分别按 `[每个头的元素数, token 容量, 注意力头数]` 存储为三维张量；第三维区分注意力头，不是区分 K 和 V。这样可以用一次追加操作写入所有头的新 K，再用一次写入所有头的新 V，供后面的 attention 读取，减少逐头创建追加节点。

三维布局不是实现 KV Cache 的必要条件。每个头分别使用二维 `[每个头的元素数, token 容量]` 缓存，也能保留历史数据并追加新数据；无需每轮复制整个缓存，也无需双缓冲。当前实现采用这种方式，两版本均已有通过记录。三维方案的目的在于简化多头数据组织及追加节点，尚未证明其性能优于二维方案。

**测试目的：** 把新 token 的 16 个 FP16 元素写入已有缓存的下一个位置。缓存预先分配为 `[8,8,2]`，可存 8 个 token，每个 token 有 2 个注意力头、每个头 8 个元素，共 128 个元素。追加后增长的是有效数据长度，缓存容量始终不变。

**最小测试如何调用：** [cache_append_check.cpp](../tests/native/np101/cache_append_check.cpp) 只有一个 `TENSORSTACKCONCAT(axis=1)` 节点，没有 RESHAPE 节点，也不调用张量扩容接口。输入为待写入数据和位置索引，输出直接绑定预分配的缓存：

| 参数 | 二维对照 | 三维问题配置 |
|---|---|---|
| 待写入数据，FP16 | `[8,1]`，8 个元素 | `[8,1,2]`，16 个元素 |
| 位置索引，INT32 | `[1]`，依次为 0、1 | `[1]`，依次为 0、1 |
| 预分配缓存，FP16 | `[8,8]`，64 个元素 | `[8,8,2]`，128 个元素 |

所有张量均为 nonconst，输入、索引和缓存都在建图前初始化。预期连续写入两个位置，检查新数据、此前的数据及尚未写入的零值后缀；第二次写入后，三维缓存中有 32 个有效元素，总容量仍为 128。

**实际结果：** `v105` 的二维对照两次写入均正确，正常释放。三维测试在 `vsi_nn_SetupGraph` 内打印以下错误，随后收到 `SIGSEGV`，尚未进入验图和写入：

```text
Cannot calculate the reshape tensor 16 to 32.
Reshape tensor fail.
```

这是 SDK 内部处理时的报错，不是测试程序要求用 reshape 将 16 个元素扩成 32 个。单节点测试没有 RESHAPE 节点，可以直接排除这种应用侧调用。二维进程耗时 0.168 s，三维进程约 0.27 s；见[二维结果][append-2d-result]、[三维日志][append-3d-log]及[SDK 调用记录][append-3d-timing]。

**复现命令：** 先运行二维对照；其他测试完成后，再单独运行三维配置：

```bash
python scripts/check_np101.py cache-append --rank 2 --sdk-timing --timeout 30 \
  --output .cache/runs/v105-cache-append-2d
python scripts/check_np101.py cache-append --rank 3 --sdk-timing --timeout 30 \
  --output .cache/runs/v105-cache-append-3d
```

**三维测试的两版本记录：** 原测试把“生成测试 K、V → 写入三维缓存 → attention 读取缓存计算”放在同一张 SDK 图中。两版本使用同一程序及 `stack` 参数，均在建图阶段出现上述 reshape 报错并崩溃，尚未执行任何缓存写入或 attention 计算，见[`v106` 记录][cache-3d-result]和[`v105` 记录][cold-cache-3d]。为单独检查追加操作，新增测试去掉了数据生成和 attention，仅保留一个追加节点；该程序已在 `v105` 复现，尚未在 `v106` 运行。

**二维方案的两版本记录：** 每个注意力头使用独立的二维 K、V 缓存，在同一张图中追加数据并执行 attention。`v106` 和 `v105` 均完成 21 次执行，缓存数据和 attention 输出检查通过，运行期间无需重新验图，资源正常释放，见[`v106` 结果][v106-cache-2d]和[`v105` 结果][rollback-cache]。这项验证包含追加与 attention；另一个仅保留追加节点的二维测试目前只在 `v105` 运行，不能将两者混为同一测试。

**需要确认：** 芯片团队的算子指南列出了 `TENSORSTACKCONCAT`，安装头文件 `ops/vsi_nn_op_tensorstackconcat.h` 公开了 `axis`，但未说明上述三维输入、输出形状是否受支持。因此仍需确认正确调用方式；若配置不支持，SDK 应返回错误而不是进程崩溃。不能仅凭这条内部 reshape 报错判定调用参数一定正确或一定错误。

### 数据读回或计算结果错误

#### P5：大载荷完整扫描中出现局部读回错误

**问题表现：** `v106` 复测以 8 MiB 为单位申请并写入张量，目标上界为 4096 MiB；成功保留 2840 MiB 后，下一块被 SDK 拒绝。随后保留全部 355 个已写入张量，逐块读回并比较每个字节；发现错误后仍继续扫描后面的块。

四种组合（FP16／FP32 × const／nonconst）均完整扫描 2,977,955,840 字节，只有编号为 109 的张量（从 0 开始编号）内三个范围不同，共 24 字节；其余 2,977,955,816 字节全部一致。并非仅抽查这三个位置，也不是在首次错误处停止。

这里的四种配置分别是 FP16 const、FP16 nonconst、FP32 const、FP32 nonconst。精度由 `--dtype F16/F32` 指定；`--storage constant/mutable` 分别令张量的 `is_const` 为 true/false。const 数据随张量创建写入，nonconst 数据在创建后调用 `vsi_nn_CopyDataToTensor` 写入。

| 组合 | 成功写入并完整扫描 | 下一块 8 MiB | 不同字节数 | 进程耗时 |
|---|---|---|---|---|
| FP16 const | 2840 MiB，355 块 | 申请时拒绝 | 24 | 78.40 s |
| FP16 nonconst | 2840 MiB，355 块 | 写入时拒绝，SDK 状态 `-5` | 24 | 82.63 s |
| FP32 const | 2840 MiB，355 块 | 申请时拒绝 | 24 | 63.78 s |
| FP32 nonconst | 2840 MiB，355 块 | 写入时拒绝，SDK 状态 `-5` | 24 | 66.77 s |

四次均完整扫描并正常释放，退出码 1 表示字节校验失败；没有在错误处中止扫描。见[`v106` 复测汇总][memory-recheck-summary]及其中各次运行的 `readback-blocks.jsonl`。`v105` 补测也在相同块号和偏移处出错。两版此次观察到的块内位置相同，但相对历史 `v106` 扫描整体后移了 65,536 字节：

| 历史 `v106` 张量内起始偏移 | 此次两版本张量内起始偏移 | 每个范围长度 |
|---|---|---|
| 1,274,680 | 1,340,216 | 8 字节 |
| 1,274,744 | 1,340,280 | 8 字节 |
| 1,274,808 | 1,340,344 | 8 字节 |

第 109 块从累计载荷的 872 MiB 开始；此次第一处错误约在 873.278 MiB，历史记录约为 873.216 MiB。上述数字是张量内偏移或按申请顺序累计的逻辑位置，不是物理地址，不能称为始终不变的故障地址。

**相同 1024 MiB 目标的版本对照：** `v106` 四种组合均成功保留并完整扫描 128 块、1024 MiB；`v105` 四种组合均在保留 127 块、1016 MiB 后被 SDK 拒绝下一块，随后完整扫描全部成功写入空间。const 在申请时拒绝，nonconst 在写入时返回状态 `-5`。两版均覆盖第 109 块及其后的块，并正常释放退出。

| 组合 | `v106`：1024 MiB 完整扫描 | `v105`：1016 MiB 完整扫描 | `v106` 进程耗时 | `v105` 进程耗时 |
|---|---|---|---|---|
| FP16 const | 第 109 块三个范围、24 字节不同 | 相同范围、24 字节不同 | 27.62 s | 27.55 s |
| FP16 nonconst | 第 109 块三个范围、24 字节不同 | 相同范围、24 字节不同 | 27.81 s | 27.75 s |
| FP32 const | 第 109 块三个范围、24 字节不同 | 相同范围、24 字节不同 | 21.49 s | 22.44 s |
| FP32 nonconst | 两次扫描全部一致 | 第 109 块相同三个范围、24 字节不同 | 22.24 s、23.60 s | 22.04 s |

`v105` 每次扫描 1,065,353,216 字节，只有上述 24 字节不同，其余 1,065,353,192 字节一致，见[补测汇总][memory105-summary]。因此，P5 在 `v105` 已存在，不以超过 1 GiB 为前提，也不能归为 `v106` 独有问题。FP32 nonconst 在 `v106` 的 1024 MiB 测试中通过、在 2840 MiB 测试中失败，`v105` 的 1016 MiB 测试也失败；实际分配量和启动前设备状态不同，不能据此认定该组合在某一版本中始终正确或错误。

表中的“两次”专指 `v106` 下 FP32 nonconst、1024 MiB 目标的两次独立运行：首次结果与其他配置不同，因此保持参数不变再次完整扫描，确认两次均无错误。它不表示 FP32 在所有载荷下通过。

“完整扫描”指本次成功申请并写入的全部张量空间，不包括 SDK 隐藏开销、保留区或未能申请的物理内存，也不能据此推断所有负载、数据模式和分配顺序都只有这三处错误。

**测试做什么：** [capacity_check.cpp](../tests/native/np101/capacity_check.cpp) 使用[公共分配代码](../tests/native/np101/allocation_support.cpp)，向每个张量写入随块编号和元素位置变化的有限非零数值。达到目标量或下一块被 SDK 拒绝后，保留全部已写入张量并扫描每个字节，遇到数据差异仍继续，扫描结束后统一释放。不创建计算节点，不调用 `SetupGraph`、`VerifyGraph` 或 `RunGraph`。`capacity` 只检查申请、写入和释放；以下 `integrity` 命令才执行完整读回。

**复现命令：** 先运行 64 MiB，确认正常读回和释放：

```bash
python scripts/check_np101_memory.py integrity --mib 64 --storage constant --dtype F16 \
  --timeout 120 --output .cache/runs/v106-integrity-64
```

`v106` 再使用显式 `--probe-limit` 探测接近上限的载荷。4096 MiB 是停止申请的目标上界；若 SDK 提前拒绝下一块，仍完整扫描此前已成功写入的所有块。测试 FP32 时改为 `--dtype F32`，测试非 const 存储时改为 `--storage mutable`；每种组合使用新目录：

```bash
python scripts/check_np101_memory.py integrity --mib 4096 --probe-limit --storage constant --dtype F16 \
  --timeout 180 --output .cache/runs/v106-integrity-limit-f16-constant
```

比较两版本时，使用同样的 1024 MiB 目标、8 MiB 块大小、数据模式及存储类型。以下为 FP16 const 命令，另外三种组合按上述方法修改参数和目录名；切换版本时修改目录中的版本简称。SDK 若提前拒绝，则完整扫描实际保留量：

```bash
python scripts/check_np101_memory.py integrity --mib 1024 --storage constant --dtype F16 \
  --timeout 120 --output .cache/runs/v106-integrity-1024
```

`v105` 无需运行超过 1 GiB 的探测；上述 1024 MiB 目标已复现读回问题。只有 `v106` 的显式容量探测使用 `--probe-limit` 超过默认 1 GiB；模型加载和普通测试的分段预算不变。

**查看结果：** 终端及 `result.json` 分别给出实际保留量、下一块是否被拒绝、扫描字节数、`full_scan_completed`、错误块数、错误范围数及错误字节总数。只有全部保留块均已读回足量数据并比较，才能写“完整扫描”。`readback-blocks.jsonl` 每块一条记录，包括没有错误的块；`sdk.log` 打印错误的块内范围和累计逻辑偏移。首次出错另存预期和实际字节。读回失败与扫描未完成、申请拒绝是不同结果。见[`v106` 汇总][memory-recheck-summary]、[`v105` 汇总][memory105-summary]、[历史 2840 MiB 扫描][v106-memory-scan]和[历史 1024 MiB 测试][v106-memory-1024]。

**需要排查：** 为什么完整扫描仅发现三个局部区段出错，以及相同 SDK 下错误偏移为何发生变化。是否涉及保留区、地址映射或读写覆盖，需由驱动侧进一步定位；目前没有物理地址记录，尚不能确定原因。

#### P6：FP16 随机采样返回越界索引

**问题表现：** 输入 8 个类别，正确索引范围为 0–7；两版本的 FP16 采样均返回 8。七组测试，每组 4096 个样本，全部越界。即使 8 个输入分数全为 0，结果也相同。改为 FP32，或者先将 FP16 转为 FP32 再采样，检查均通过。见[`v106` 结果][v106-sampling]和[`v105` 汇总][rollback-summary]。

**测试计算什么：** [sampling_check.cpp](../tests/native/np101/sampling_check.cpp) 检查推理时按概率选择下一个 token 的功能。输入分数（`logits`）形状为 `[8,1]`，INT32 随机种子/计数器形状为 `[4]`，输出为 INT32 `[4096,1]`。直接使用 `RANDOM_MULTINOMIAL`；转换测试先增加 FP16→FP32 转换。[Python 校验代码](../python/specferry/validation/sampling.py) 检查索引有没有越界、各类别出现频率，以及随机种子和计数器是否生效。

当前源码在建图前初始化分数和随机种子。`v105` 使用该源码复测后，FP32 和转换后采样仍通过，直接 FP16 仍全部返回 8，见[复测结果][code-review]。

**复现命令：** 三条命令分别检查 FP32、转换后采样、直接 FP16 采样：

```bash
python scripts/check_np101.py sampling --dtype F32 --classes 8 --samples 4096 \
  --sdk-timing --timeout 60 --output .cache/runs/v105-sampling-f32
python scripts/check_np101.py sampling --dtype F16_TO_F32 --classes 8 --samples 4096 \
  --sdk-timing --timeout 60 --output .cache/runs/v105-sampling-converted
python scripts/check_np101.py sampling --dtype F16 --classes 8 --samples 4096 \
  --sdk-timing --timeout 60 --output .cache/runs/v105-sampling-f16
```

**查看结果：** `result.json` 记录越界数量、最小/最大索引、各类别出现频率。七个 `.bin` 文件保存原始采样结果。C++ 程序退出码为 0 只说明采样和释放执行结束；数值不正确时，Python 脚本返回非零。

**当前处理与待排查问题：** 推理时先把输入分数转为 FP32 再采样。需要确认 FP16 采样实现是否有错误；输入全零仍返回非法索引，不能简单归为舍入误差。

#### P7：用加法合并查表结果时，FP16 极小值变成 0

**问题表现：** `v106` 的查表组合曾将 FP16 非正规数（`subnormal`，一类极小的非零值）变为 0。改用 `SELECT` 后逐字节正确，见[`ADD` 失败记录][lookup-add-result] 和[`SELECT` 结果][lookup-select-result]。

**单节点测试：** [fp16_preservation_check.cpp](../tests/native/np101/fp16_preservation_check.cpp) 每个进程只建一个 `ADD` 或 `SELECT` 节点，不加载模型，也不使用 `GATHER`。FP16 输入、输出形状为 `[16,1]`，包含 10 个正负非正规数、最小正负正规数、0 和普通值。输入直接按位构造，避开主机和 SDK 的精度转换。每个位置的两个输入中，一个为待测值，一个为 0；`SELECT` 的布尔输入选择待测值。预期两种操作都保留该值。两版本使用同一可执行文件，`ADD` 的输入、预期值及实际输出文件逐字节相同。

| 操作 | `v105` 结果 | `v106` 结果 |
|---|---|---|
| `ADD` | 10 个非正规数全部变为 `0x0000`；其余 6 个值正确；正常释放后返回数值失败（退出码 1） | 与 `v105` 相同；验图 15.766 ms，进程 0.168 s，正常释放；没有新增 `NP hang` |
| `SELECT` | 16 个值逐字节正确；正常释放，退出码 0 | 相同 16 个值逐字节正确；出现内核卡死及自动恢复，随后正常释放，退出码 0 |

例如 `0x0001`、`0x03ff`、`0x8001`、`0x83ff` 经加法后均为 0，最小正规数 `0x0400`、`0x8400` 保持不变。执行前后输入字节均正确，CPU 对相同 FP16 数据加法的结果也逐字节正确。见[`v105 ADD` 日志][cold-add]、[`v106 ADD` 日志][recheck-add]、[`v105 SELECT` 日志][cold-select]及[本轮汇总][memory-recheck-summary]。

**复现命令：** 先单独运行 `ADD`；`SELECT` 属于 P3 的卡死用例，按 P3 要求冷启动后另行运行：

```bash
python scripts/check_np101.py fp16-preservation --operator add --sdk-timing --timeout 30 \
  --output .cache/runs/v105-fp16-add
python scripts/check_np101.py fp16-preservation --operator select --sdk-timing --timeout 30 \
  --output .cache/runs/v105-fp16-select
```

`device/sdk.log` 逐项打印预期位模式、实际位模式和差异数量；同目录下 `expected.bin`、`actual.bin`、`left.bin`、`right.bin` 保存原始数据。数值不一致时测试返回非零，不代表 SDK 崩溃。

**需要确认：** FP16 `ADD` 是否默认将非正规数置零，是否存在保留这些数值的设置。两版本均已用同一单节点用例复现。模型目前使用 `SELECT` 保留这些数值；其验图卡死另见 P3，完整推理受数值置零影响的程度尚未量化。

### 内存占用与释放

#### P8：申请 8 MiB 张量，SDK 内存计数增加约 16 MiB

**问题表现：** 两版本创建一个 FP16、8 MiB 的张量后，`gpu_memory.currentSize` 均按下表变化：

| 时刻 | SDK 计数 |
|---|---|
| 申请前 | 524,288 字节 |
| 申请后 | 17,306,176 字节 |
| 释放后 | 524,288 字节 |

计数增加 `2 × 8,388,608 + 4,672` 字节。数据读回正确，释放后计数恢复。见[`v106` 日志][v106-accounting]和[`v105` 日志][rollback-accounting]。

**测试做什么：** [memory_accounting_check.cpp](../tests/native/np101/memory_accounting_check.cpp) 只创建一个张量，用 `gcoOS_GetMemoryProfileInfo` 读取申请前、申请后、释放后的计数，并检查写入的数据能否正确读回。

**复现命令：**

```bash
python scripts/check_np101_memory.py accounting --mib 8 --storage mutable --dtype F16 \
  --timeout 60 --output .cache/runs/v105-accounting
```

脚本设置 `VIV_MEMORY_PROFILE=1`，`sdk.log` 列出张量字节数、三次计数和比例。

**需要说明：** 计数翻倍是实际保存了两份数据，还是包含预留、镜像等其他项目。若实际占用双份板上内存，会减少多少可用容量；是否存在泄漏或越界写入风险。这次测试中计数恢复、数据正确，尚未观察到泄漏或越界写入。

#### P9：反复修改图的缓冲区绑定后，主机内存持续增长

**问题表现：** 两版本均运行 128 轮复制。重新绑定时，主机驻留内存（RSS）约增加 16 MiB；保持绑定不变时没有增长。下表统一使用“第 127 轮复制后的 RSS 减去第 0 轮复制后的 RSS”，不包含初始化和最后释放后的变化：

| 绑定方式 | `v106` RSS 增量 | `v105` RSS 增量 |
|---|---|---|
| 固定绑定 | 0 MiB | 0 MiB |
| 每轮重新绑定到相同位置 | 16.04 MiB | 15.98 MiB |
| 每轮轮换绑定位置 | 16.04 MiB | 16.04 MiB |

数据读回和资源释放均通过。见[`v106` 相同位置日志][v106-host-rebind]、[轮换位置日志][v106-host-advance]、[固定绑定日志][v106-host-fixed]及[`v105` 汇总][rollback-summary]。逐轮增量并非严格相等，版本对比统一使用上表的全程差值。

当前源码已在日志末尾增加相同统计区间的汇总。`v105` 复测固定绑定无增长，两种重新绑定均约增加 16 MiB，读回和释放正常，见[复测结果][code-review]。

**测试做什么：** [memory_growth_check.cpp](../tests/native/np101/memory_growth_check.cpp) 提前创建一个复制图和 8 个目标张量视图，每轮复制 2048 字节。视图引用已有缓冲区，循环中不新增视图或扩大容器。三种模式分别为：

- `fixed`：不重新绑定，每轮写入相同位置。
- `same`：每轮把目标重新绑定到原位置。
- `advance`：每轮把目标绑定到下一个位置，8 个位置循环使用。

重新绑定后，只有 SDK 判断图需要重新验图时才调用 `vxVerifyGraph`。

**复现命令：**

```bash
python scripts/check_np101_memory.py growth --mode fixed --iterations 128 \
  --timeout 120 --output .cache/runs/v105-growth-fixed
```

分别改用 `--mode same`、`--mode advance` 并更换目录。`sdk.log` 列出每阶段 RSS 和耗时，末尾的 `Loop RSS` 直接给出首轮与末轮复制后的差值，单位同时列出字节和 MiB。

**尚不清楚的地方：** RSS 是主机进程占用的物理内存，不是板上内存。增长可能来自 SDK 未释放数据，也可能是主机分配器保留已释放空间。尚未测量主机与板卡传输了多少数据、传输耗时占比多少，不能用 RSS 推算这些指标。

**需要说明：** 重复绑定和验图会保留哪些主机数据、何时释放；固定绑定与重复绑定的内存差异是否符合预期。通信量和通信耗时需另行计时，不能从 RSS 增长得出结论。

### 验图耗时

#### P10：单层图验图需要两分多钟

**问题表现：** 两版本均需要两分多钟完成一次验图，而验图后的两次计算合计约 0.34 s。

| 版本 | 验图时间 | 两次执行合计 | 输出检查 |
|---|---|---|---|
| `v106` | 两次测试分别为 158.095 s、157.887 s | 后一次约 0.346 s | 后一次 14 项输入、2 项 KV 缓存检查通过，正常释放资源 |
| `v105` | 140.917 s | 0.335 s | 相同参考数据的 16 项检查通过，正常释放资源 |

见[`v106` 首次计时][v106-prefix-first]、[`v106` 复测][v106-prefix-repeat] 和[`v105` 结果][rollback-prefix]。上述两版本复测时，内核日志均未发现新的 `NP hang`。`v105` 验图期间一次采样发现进程正在运行，累计使用了 82.52 s CPU 时间，说明验图并非全程睡眠等待驱动。

**图包含什么：** [graph_pipeline_check.cpp](../tests/native/np101/graph_pipeline_check.cpp) 的 `prefix` 测试使用 OPT-350M 权重，图中有 token 和位置嵌入、一层 decoder、KV 缓存更新、输出投影、LM head 和 token 选择，共 612 个 SDK 节点。缓存最多容纳 16 个 token，测试逐次输入两个 token。当前只比较输入处理结果和已写入的 KV 数据，尚未检查 decoder 输出及 LM head 是否正确。

**复现命令：** 按主 README 下载并导出 OPT-350M，准备 CPU 参考数据后执行：

```bash
python scripts/check_np101_model.py prefix --layer-count 1 --steps 2 --capacity 16 \
  --prepare-only --output .cache/runs/prefix-reference
python scripts/check_np101_model.py prefix --layer-count 1 \
  --fixture .cache/runs/prefix-reference/fixture --sdk-timing --timeout 360 \
  --output .cache/runs/v105-prefix
```

比较两版本时使用相同参考目录。`sdk-calls.tsv` 分别记录建图、验图和执行的时间，`checks.tsv` 记录数值误差。

**需要排查：** 是否有不恰当的 API 调用、张量设置或图连接方式，导致验图明显变慢。目前不能仅凭节点数量判断原因，也不能根据一次计时差异认定哪个版本更快。

## 版本对比状态

P5 四种组合的版本对照已补齐：同一程序在 `v105` 完整扫描 1016 MiB，四种均复现局部读回错误；`v106` 的 1024 MiB 和 2840 MiB 结果一并保留。两版实际保留量不同，报告分别列出；没有将申请拒绝误记为读回通过。

P1、P2、P4、P6、P7、P8、P9、P10 已有两版本结果。P3 已有两版本的同一单节点 `SELECT` 对照；原 selection 在 `v106` 冷启动后的结果仍缺，但不影响现有卡死复现。P4 在两版本中均有“三维追加与 attention 在同一张图中时建图崩溃、二维追加与 attention 通过”的记录。新增单节点追加程序只在 `v105` 运行，二维通过、三维崩溃；`v106` 的单节点对照待补。

## 日志在哪里、怎样看

| 文件 | 计算测试的保存位置 | 内存测试的保存位置 | 主要内容 |
|---|---|---|---|
| `result.json` | 输出根目录 | 输出根目录 | 测试是否通过、数值检查结果 |
| `sdk.log` | `device/` | 输出根目录 | SDK 输出、测试错误信息 |
| `execution-evidence.json` | `device/` | 输出根目录 | 命令、文件校验值、耗时、退出状态 |
| `sdk-calls.tsv` | `device/`，需启用 `--sdk-timing` | 不生成 | 每次 SDK 调用的开始、结束及耗时 |

[历史测试][rollback-summary]及[内核日志][rollback-kernel-full]保留供对照。`v105` 冷启动后按“卷积、`SELECT`、`ADD`、词表取行、三维缓存”的顺序执行，共 5 个原生进程，无超时和残留进程；内核记录 5 次打开和 5 次释放，没有 `NP hang/recovery`。最后三维缓存测试崩溃后停止访问板卡并保留恢复标记，没有验证崩溃后能否继续计算。见[补测汇总][cold-summary]、[内核日志][cold-kernel-log]和[结束状态][cold-completion]。

切回 `v106` 并冷启动后，只执行了单节点 `SELECT`。内核记录一次打开、一次释放，以及一次 `NP hang` 和自动恢复；测试进程已退出，无残留进程。确认日志后停止 SDK 访问并写入恢复标记，见[结束状态][cold106-completion]。该次原始 `result.json` 的数值通过和进程退出记录先于内核日志检查；最终设备状态以[内核摘要][cold106-kernel]及恢复标记为准。

`v106` 的 P5 与单节点 `ADD` 补测在同一启动周期后续进行，并非新的冷启动：64 MiB 读回正确后，再执行大载荷及 1024 MiB 扫描，最后执行 `ADD`。对比测试前后内核日志，没有新增 `NP hang`、恢复或进程崩溃；全部进程已释放退出，结束时没有恢复标记。该结果不代表 P3 已修复。见[内核检查][memory-recheck-kernel]和[结束状态][memory-recheck-completion]。

`v105` 的 P5 补测先通过 64 MiB 读回，再执行四种 1024 MiB 目标扫描。测试前后均无恢复标记，本次启动的内核日志没有 NP101 hang/recovery；五个测试进程全部正常释放退出，内核记录五次打开和五次释放。见[内核检查][memory105-kernel]、[原始日志][memory105-kernel-log]及[结束状态][memory105-completion]。这轮只运行内存测试，不作为 P3 计算图验图的复测结果。

新增 P4 单节点测试的三维模式以 `SIGSEGV` 退出后，运行器保留恢复标记并停止后续设备测试；不能用此前内存测试的正常退出推断这次崩溃后的设备状态。

`Waiting for HDMA Transfer...` 只表示程序打印了等待传输的消息，没有等待时长；消息多不等于通信耗时高，需要计时才能判断。

本报告中的 `.cache/runs` 链接指向本地记录，不随 Git 分发。提供复现材料时，需同时附上所引用的结果目录和内核日志。

[v106-environment]: ../.cache/runs/p8-verified-20260925/environment.json
[v106-conv]: ../.cache/runs/demo-verify-20260925.log
[v106-memory-scan]: ../.cache/runs/capacity-map-20260923/scan-f16-constant/summary.json
[v106-memory-1024]: ../.cache/runs/memory-driver-20260922-environment/experiment-summary.json
[v106-prefix-first]: ../.cache/runs/f2-20260924/prefix-1/device/sdk-calls.tsv
[v106-prefix-repeat]: ../.cache/runs/f2-hotboot-20260925/prefix-1/result.json
[v106-accounting]: ../.cache/runs/memory-integrated-accounting/sdk.log
[v106-projection]: ../.cache/runs/p8-verified-20260925/summary.json
[v106-fcl-stack]: ../.cache/runs/p8-verified-20260925/fcl/sdk.log
[v106-sampling]: ../.cache/runs/driver-1.0.6-legacy-recheck-20260925/p9-summary.json
[v106-selection]: ../.cache/runs/driver-1.0.6-hotboot2-20260925/summary.json
[v106-selection-kernel]: ../.cache/runs/driver-1.0.6-hotboot2-20260925/kernel-test-window.log
[v106-host-rebind]: ../.cache/runs/memory-integrated-same/sdk.log
[v106-host-advance]: ../.cache/runs/memory-integrated-advance/sdk.log
[v106-host-fixed]: ../.cache/runs/memory-integrated-fixed/sdk.log
[cache-3d-result]: ../.cache/runs/o3-cache-stack-20260923/execution-evidence.json
[v106-cache-2d]: ../.cache/runs/o3-cache-column-20260924/cache.txt
[lookup-add-result]: ../.cache/runs/f2-20260924/real-lookup/result.json
[lookup-select-result]: ../.cache/runs/f2-20260924/real-lookup-select/result.json
[rollback-environment]: ../.cache/runs/driver-1.0.5-20260925/environment.json
[rollback-summary]: ../.cache/runs/driver-1.0.5-20260925/summary.json
[rollback-conv]: ../.cache/runs/driver-1.0.5-20260925/p1-conv/result.json
[rollback-vendor]: ../.cache/runs/driver-1.0.5-20260925/p1-vendor-original/result.json
[rollback-accounting]: ../.cache/runs/driver-1.0.5-20260925/p7-accounting/sdk.log
[rollback-cache]: ../.cache/runs/driver-1.0.5-20260925/p10-2d-control/result.json
[rollback-prefix]: ../.cache/runs/driver-1.0.5-20260925/p4-prefix/result.json
[rollback-fcl]: ../.cache/runs/driver-1.0.5-20260925/p8-fcl/result.json
[rollback-kernel]: ../.cache/runs/driver-1.0.5-20260925/kernel-review.json
[rollback-kernel-full]: ../.cache/runs/driver-1.0.5-20260925/kernel.log
[rollback-symbolication]: ../.cache/runs/driver-1.0.5-20260925/fcl-symbolication.txt
[cold-conv]: ../.cache/runs/v105-coldboot-20260928/control-conv/result.json
[cold-selection]: ../.cache/runs/v105-coldboot-20260928/p7-selection/result.json
[cold-cache-3d]: ../.cache/runs/v105-coldboot-20260928/p9-3d-archived/result.json
[cold-add]: ../.cache/runs/v105-coldboot-20260928/p10-add/device/sdk.log
[cold-select]: ../.cache/runs/v105-coldboot-20260928/p10-select/device/sdk.log
[cold-summary]: ../.cache/runs/v105-coldboot-20260928/summary.json
[cold-kernel]: ../.cache/runs/v105-coldboot-20260928/kernel-review.json
[cold-kernel-log]: ../.cache/runs/v105-coldboot-20260928/kernel-test-window.log
[cold-completion]: ../.cache/runs/v105-coldboot-20260928/completion.json
[cold106-environment]: ../.cache/runs/v106-coldboot-20260928/environment.json
[cold106-select-timing]: ../.cache/runs/v106-coldboot-20260928/p10-select/device/sdk-calls.tsv
[cold106-kernel-log]: ../.cache/runs/v106-coldboot-20260928/kernel-test-window.log
[cold106-summary]: ../.cache/runs/v106-coldboot-20260928/summary.json
[cold106-completion]: ../.cache/runs/v106-coldboot-20260928/completion.json
[cold106-kernel]: ../.cache/runs/v106-coldboot-20260928/kernel-review.json
[memory-recheck-summary]: ../.cache/runs/v106-p5-recheck-20260928/summary.json
[memory-recheck-kernel]: ../.cache/runs/v106-p5-recheck-20260928/kernel-review.json
[memory-recheck-completion]: ../.cache/runs/v106-p5-recheck-20260928/completion.json
[recheck-add]: ../.cache/runs/v106-p5-recheck-20260928/p7-add/device/sdk.log
[memory105-environment]: ../.cache/runs/v105-p5-recheck-20260928/environment.json
[memory105-summary]: ../.cache/runs/v105-p5-recheck-20260928/summary.json
[memory105-kernel]: ../.cache/runs/v105-p5-recheck-20260928/kernel-review.json
[memory105-kernel-log]: ../.cache/runs/v105-p5-recheck-20260928/kernel-after.log
[memory105-completion]: ../.cache/runs/v105-p5-recheck-20260928/completion.json
[append-2d-result]: ../.cache/runs/report-review-20260928/cache-append-2d/result.json
[append-3d-log]: ../.cache/runs/report-review-20260928/cache-append-3d/device/sdk.log
[append-3d-timing]: ../.cache/runs/report-review-20260928/cache-append-3d/device/sdk-calls.tsv
[code-review]: ../.cache/runs/report-review-20260928/review.json
