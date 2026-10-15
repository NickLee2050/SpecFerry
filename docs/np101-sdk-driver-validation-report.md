# NP101 SDK 与驱动问题

本页保留失败契约、解释边界和证据索引；性能/已验配置见[实现总结](np101-status.md)，
每次运行的约束及冷启动事故见[事件台账](np101-recovery-incidents.md)，命令见
[测试手册](../tests/README.md)。以下是历史结果，不是自动复现清单；模型成功不关闭底层问题。

10-14静态复核：v105 EDMA的SG分支会主动将timeout打印用dst字段设为全1，实际
提交SG表；因此`dst=0xffffffffffffffff`不能单独证明非法访问或DMA映射失败。
内部完成等待存在4ms/30s分支（HZ250），与应用900s期限不同；实际触发分支/根因未定。
本轮4字节EDMA的独立时间线和反汇编证据见台账，不据此放行故障设备。

`v105/v106` 指匹配的驱动与SDK安装包1.0.5/1.0.6，不是模块打印字符串。
库位于 `/usr/lib/ljmicro`；galcore srcversion分别为 `F53A3E6EFBB7D0FD7D4617C` /
`8D46803809F15FA0FD98A92`。每份证据固定boot、命令、binary/SDK哈希和内核窗口。
表内参数是历史binary的接口，不作为当前运行命令。自写卷积及原带bias FCL诊断均已退役；
保留原厂/旧改写身份与证据，不重建替代。

## 建图、验图与数据问题

| 操作与问题 | 预期 | 已记录实际结果 | CASE / ARGS；原始证据索引 |
|---|---|---|---|
| 卷积→ReLU→MaxPool 的 Verify | 原厂 demo 一轮输出误差 ≤0.1；旧改写控制历史两轮另存 | v106 `SIGFPE`，位于 `libNNArchPerf.so::NNTransposeCycleCount_V9` 整数除法，未执行卷积；v105 原厂及仓库版本通过，仓库 Verify 42.478 ms（冷启动 349.972 ms），最大误差 0.003234/0.003114 | `conv`；旧版验图对照、旧版驱动回归 的 `convolution-control/`、旧版冷启动 的 `control-conv/` |
| 带 bias 的 FCL Verify | 输入 `[8,1]`、单位矩阵 `[8,8]`、bias 0.25；输出 1.25/-0.25 | 两版 `SIGSEGV`；相同计算 MatMul+Add 两轮精确通过。v105 空指针偏移 `0xb0` 读取，v106 栈含 `vxoGraphOptimization_getKernelType`；未证明 bias 单独致错或所有 FCL 均失败 | `projection --operator matmul-add` 对照，`projection --operator fcl` 复现；旧版验图对照、旧版驱动回归 的 `fcl-verification/` 与 `fcl-symbolication.txt` |
| 单 SELECT Verify / 历史 selection GATHER | 完成验图，无驱动 hang；保留 FP16 位模式 | 同一 SELECT binary：v106 冷启动首测 Verify 28.982 s，NP hang/自动恢复后 16 值正确；v105 9.228 ms，无 hang。历史 GATHER 也曾 hang；无 GATHER 的单 SELECT 已复现，不能仅归因索引写入时机 | `fp16-preservation --operator select`，超时 30 s；旧版冷启动 的 `fp16-selection/`、新版冷启动。selection 为 `check_np101_model.py selection` |
| 三维 TENSORSTACKCONCAT(axis=1) Setup | `[8,1,2]` 追加到预分配 `[8,8,2]`，容量不变 | v105 单节点在 Setup 报 `Cannot calculate the reshape tensor 16 to 32` 后 SIGSEGV；没有应用 RESHAPE。两版追加+attention 三维组合也崩溃，二维组合通过；v106 单节点待补。三维参数是否支持仍需厂商确认 | `cache-append --rank 2` 对照，`cache-append --rank 3` 复现，超时 30 s；报告复核 的 `cache-append-{2d,3d}/` |
| 分块写入后的完整字节扫描 | 所有已成功写入张量的每个字节一致 | v106 上界 4096 MiB，保留/全扫 2840 MiB；F16/F32×const/nonconst 四种均在第 109 块三个范围损坏 24 B。v105 目标 1024、保留/全扫 1016 MiB，四种同位置损坏 24 B；不是超过 1 GiB 才出现 | `check_np101_memory.py integrity --mib 1024 --storage constant --dtype F16`；新版内存扫描、旧版内存扫描 的 `summary.json` / `readback-blocks.jsonl` |
| RANDOM_MULTINOMIAL FP16 | 8 类输出索引 0–7，频率和 seed/counter 有效 | 两版七组×4096 样本全返回 8，含全零 logits；F32 或 F16→F32 后采样通过。进程正常退出不代表数值通过 | `sampling --dtype F16 --classes 8 --samples 4096`；对照 `F32` / `F16_TO_F32`；旧版验图对照、旧版驱动回归、报告复核 |
| FP16 ADD 非正规数 | 与零相加保留输入位模式 | 两版同 binary 的 10 个正负非正规数全部变 `0x0000`，其余 6 值正确；SELECT 保留 16 值，但 v106 触发选择算子验图。输入字节及 CPU 对照正确，需确认置零规则/配置 | `fp16-preservation --operator add`，超时 30 s；旧版冷启动 的 `fp16-add/`、新版内存扫描 的 `fp16-add/` |
| SDK 内存计数 | 明确计数与实际板上分配关系，释放后恢复 | 两版 8 MiB tensor 使 `gpu_memory.currentSize` 增加 `2×8,388,608+4,672` B；524,288→17,306,176→524,288，读回正确。不能据此证明双份物理分配、泄漏或越界 | `check_np101_memory.py accounting --mib 8 --storage mutable --dtype F16`；旧版驱动回归 的 `memory-accounting/`、内存计数 |
| 固定/重绑 copy 图的主机 RSS | 固定与重绑成本可解释，数据正确且释放 | 两版 128 轮，fixed 增量 0，same/advance 约 16 MiB，读回/释放通过；统计为第 127 轮减第 0 轮。RSS 是主机占用，不能推断板上占用、通信成本或泄漏 | `check_np101_memory.py growth --mode fixed --iterations 128`；单独改 `same` / `advance`；旧版驱动回归、重绑内存 |


全窗口1×1 CONV的FFN候选也在首次Verify崩溃：固定OpenVX ELF `0x2db0c2`
（`vxoGraphOptimization_getKernelType+0x5d2`）与历史FCL同一指令；FULLY_CONNECTED_RELU
分支缺必需 `enable_relu` scalar parameter[7]，不是bias。参数丢失来源未知，入口已退役。
证据 `ffn-convolution-and-parallelism-20261004/ffn-linear-conv1x1/`。保留MatMul+Add，
不把卷积形式视为安全或快速NN路径保证。

分块完整性使用8 MiB块，拒绝下一块后仍全扫所有已写块，遇错继续扫描后统一释放。
第109块三个错误偏移为1,340,216/1,340,280/1,340,344，各8B；历史v106对应位置差
65,536 B，累计逻辑偏移约873.278/873.216 MiB，**不是物理地址**。v106的1024 MiB
FP32 nonconst两次通过、2840 MiB失败，不能认定该配置总是安全。先测64 MiB控制；
v105及模型/普通检查保持各段1 GiB上限。仅获授权的v106合成完整性探针可用
`integrity --mib 4096 --probe-limit`；不覆盖SDK隐藏空间或未申请的物理内存。

## 编译表缺项与共享槽位

逐头KV及const/mutable变体共1143个extension，前1024个operation非空、后119个缺失；
执行表随后清空、跳过Tiling/命令生成，Verify仍成功，第23/24 decoder全零。
反汇编发现user shader包装路径的1024槽，满后返回NULL、上层忽略失败码；尚未捕获
运行时首次满槽/NULL，不能称为已证明越界写。这是共享包装资源，不是每图1024节点，
也不是所有算子的统一额度，不能解释全部EDMA或分块内存损坏。

固定OpenVX SHA256 `2f7223e504b15886c1562889668a80af48b19d605461beac175d336d396ff108`：
`vxCreateContext@0x50795` 取进程级vxGlobalData，`0x507a5` 写Context+0xa8；
`vxnneGetUserShaderExecutable@0x114410` 经它访问槽表。单图释放未见归还，最后Context
释放才清零；并存Context共享预算。Verify可见锁在graph+0x20，槽函数内未见锁，
完整外层锁路径及厂商并发合同仍缺失，不宣称已证实竞争、不实施并行Verify。
多进程图移交/资源隔离也未验证。

合并KV降到423 extension、两套图846；同Context重建出现 `846+178=1024` 后缺668项，
编译保护在Run前拒绝。释放最后旧Context后两次各846完整、输出一致，采用此重建策略；
普通请求复用模型。详细地址与未闭环点在 `decode-static-audit-20261009/verify-and-incidents.md`。
图规模/Verify成本及已关闭结构候选的结果集中在实现总结。

## 原始证据索引

路径相对 `.cache/runs/`，别名映射为 `evidence-directory-aliases.json`。
需复查时同时取result、sdk.log、execution身份/退出记录及内核窗口；仅有SDK成功不足以
判定通过。`Waiting for HDMA Transfer...` 不提供等待时长。原始内容保持不变。

| 索引 | 目录 / 文件 |
|---|---|
| 旧版验图对照 | `operator-verification-20260925/`、`demo-verify-20260925.log`、`driver-1.0.6-legacy-recheck-20260925/` |
| 旧版驱动回归 | `driver-1.0.5-20260925/`：`summary.json`、`environment.json`、`kernel-review.json` |
| 旧版冷启动 | `v105-coldboot-20260928/`：`summary.json`、`kernel-review.json`，selection 23 项通过，无 hang |
| 新版冷启动 | `v106-coldboot-20260928/`：SELECT 计时、`kernel-test-window.log`、`completion.json`；首测 hang 后停测 |
| 报告复核 | `report-review-20260928/`：`review.json`、单节点追加及采样/重绑复测 |
| 新版内存扫描 | `v106-memory-integrity-20260928/`：完整四组合扫描、ADD 与前后内核观测，无新增 hang |
| 旧版内存扫描 | `v105-memory-integrity-20260928/`：环境/四组合/内核摘要；旧大载荷见 `capacity-map-20260923/` |
| 内存计数 | `memory-integrated-accounting/` |
| 重绑内存 | `memory-integrated-{fixed,same,advance}/` |
| 编译表完整性历史 | `incremental-model-checks-20260924/`、`incremental-model-warmboot-20260925/`、`incremental-cache-checks-v105-20260928/`、`weight-storage-and-cache-reshape-preparation-20260929/shader-limit/`、图规模/模型验收目录见状态文档 |


三维追加的v106单节点及v106冷启动原selection仍未补测，不自动安排危险复现。
退役结构需从历史归档独立恢复；当前固定图不能冒充旧实现。EDMA、控制hang与关机
page fault的精确SDK/内核时间线由事故台账独立维护，根因均未因恢复检查通过而关闭。
