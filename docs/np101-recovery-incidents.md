# NP101 冷启动事件与复发规避

本页是每次板卡运行必须读取的门禁及独立事故台账。稳定性能/验收结论见
[实现总结](np101-status.md)，算子失败契约见[SDK问题表](np101-sdk-driver-validation-report.md)。
记录区的停测/计划描述属于事故当时状态；后续执行以本页门禁和最新已存检查为准。
原始证据相对 `.cache/runs/`，不随Git分发；按boot、内核时间和证据去重。
2026-10-10压缩前全文及哈希在 `documentation-consolidation-20261010/before/`、
`before-manifest.json`；所有独立事故、原始日志和固定控制保持可追溯。

## 每次运行与恢复门禁

1. 运行前查boot ID、`np101-recovery-required.json`、`/dev/galcore` owner及全boot内核：
   NP hang、EDMA timeout、BUG/Oops/page fault/general protection fault/invalid opcode/
   soft lockup。`Waiting for HDMA Transfer...` 单独出现不算超时。串行、限时、运行中监控；
   任一故障停止板卡工作并保留标记，不能靠正确输出、owner清空或`recovery done`放行。
2. 恢复需用户确认**主机与NP101冷启动**，新boot清洁后归档原标记。热重启不作为恢复。
   串行跑固定仓库卷积，再跑冻结旧OPT component C16/prompt5/两输出 `[78,631]`；
   两者都须数值、执行、预算、释放和内核通过，且各在退出后检查至少45秒，禁止SDK插桩。
   历史hang曾在释放28–35秒后出现。只有两级通过才允许已检查范围内的后续工作。
3. `check_np101.py conv` 使用归档自写卷积，`recovery` 再运行冻结旧OPT；`vendor-conv`
   只是可选诊断。默认进程期限60/180秒，不代表性能目标。命令/身份清单见
   [测试手册](../tests/README.md#device-discipline-and-recovery)；缺失/哈希改变直接拒绝，
   不重编译替换固定控制。自写与原厂都有失败历史，两级成功不证明硬件无故障。
4. 相同binary/SDK控制在确认冷启动后再次失败时，停止相同条件重试，先分析已有证据；
   不重复消耗冷启动。故障复现需明确授权；有意复现selection-hang前先提醒冷启动。
5. 每次需冷启动的事件单独记：boot、binary/SDK身份、输入/结构、期望/实际结果，
   最后已返回/未返回SDK调用与内核微秒时间、退出/owner/标记、证据路径、已证实原因
   与假设、下一次规避。调用重叠只定位区间，不自动证明内部根因。

## 恢复成功记录

这些是历史放行窗口；下一次运行仍执行上述门禁，不把它们当作实时设备状态。

| 窗口 / boot | 结果与证据 |
|---|---|
| 10-08 15:04–15:07 / `83269f49-a97a-4713-a840-0a6224bd06e4` | 原厂卷积64/64、冻结旧OPT `[78,631]`、释放/预算及约70/57秒延迟门禁通过。`independent-controls-20261008-1502/`；随后16:31发生新EDMA，见独立事故 |
| 10-08 18:02–18:03 / `fe501a53-7dd6-4a7f-b99b-4b028238eb53` | 固定自写卷积/旧OPT及各45秒通过，18:02:50.486543/18:03:59.370937清洁；旧标记归档。`repository-recovery-20261008-1800/`；后续映射/KV/block4验收见总结 |
| 10-09 11:33–11:34 / `f214ee36-6fd8-42cd-922a-d5095f9980bd` | EVIS事故后确认冷启动；同身份两级控制及各45秒通过，11:33:27.158501/11:34:33.882395清洁。原标记在 `decode-working-set-20261009/recovery/previous-recovery-marker.json` |
| 最后已存检查：10-09 17:32:37.380997 +08:00 / 同上boot | 后续集合、模型、LN预处理及32/32生成门禁通过，无目标故障/owner/恢复标记。`layernorm-affine-preparation-20261009/summary.json`；不证明底层故障修复 |

## 历史故障指纹

较早事件的原始报告保留完整材料；以下是后续必须沿用的规避，不是自动复现清单。

| 时间 / 证据 | 已证实的现象及调用定位 | 下次运行的约束 |
|---|---|---|
| 09-09 `sdk-incident-20260909/report.md` | 动态 FP32 MatMul/GATHER 各有 150 秒期限，随后 `gckOS_UserSignal` 异常；1.071 GB 权重 `vsi_nn_CreateTensorFromData` 失败，线程在 `gckOS_AcquireMutex` D 态。调用间因果未证实。 | 模型应用 payload 各段保持 1 GiB 内；超时后停止，不叠加大分配探针。 |
| 09-22 `memory-driver-20260922-health-before/fault-location.json` | v106 三节点卷积验图 `vsi_nn_VerifyGraph` 中 `libNNArchPerf.so::NNTransposeCycleCount_V9` 整数除法 SIGFPE，尚未 Run。 | 用已验 v105；不在模型主线上试 v106 卷积验图。 |
| 09-25 `driver-1.0.6-hotboot-20260925/`、`recovery-retry-20260925/` | GATHER selection Verify 等待/超时；NP hang 可在释放约 32 秒后出现，热重启和重试仍有 hang。 | 避免该 selection 路径；故障后不靠热重启或即时无报错放行。 |
| 09-28 `v106-coldboot-20260928/summary.json` | 单 SELECT 在 v106 冷启动 Verify 28.982 秒且 NP hang；同 binary v105 Verify 9.228 ms。值正确。 | 不以 v106 SELECT 诊断成功替代设备健康；模型使用已验 greedy 路径。 |
| 09-29 `decoder-prefix-24-layer-20260929/`、`opt24-boundaries-resume-20260929/` | 旧图有 16 次 EDMA 超时；边界对照两次落在 `vsi_nn_ConvertTensorToData` 时间区间。旧图 1143 个 extension 超过 1024 包装槽，导致后段 operation 缺项；这**不解释** EDMA。 | 编译表逐项守门、重建用新 Context；减少诊断读回，不把 Q 零值一律归因 EDMA。 |
| 09-30 `block-prefill-cache-fault-20260930/full24-block4/driver-fault.json` | 第 20 份 KV（layer 9 V）`vsi_nn_ConvertTensorToData` 未返回时 EDMA 超时，接着 `hdma_dev_free_chan_resources` kernel BUG。 | KV 仅按验收所需选择粗粒度末层读回；一次故障后不在同 boot 重试。 |
| 10-04 `ffn-convolution-and-parallelism-20261004/ffn-linear-conv1x1/` | 1×1 CONV 候选首图 `vxVerifyGraph` 在 FCL_RELU 优化器 SIGSEGV；同日热重启控制 Verify/Run 有 NP hang，head-skip 故障状态 Verify 亦见 hang。 | 暂避 FCL 与可能折成 FCL 的全窗口卷积；保留 MatMul+Add；热重启不作为恢复。 |
| 10-05 `device-status-20261005/summary.json` | 热重启后三节点设备恢复检查 Verify 10 秒期限未完成；退出约 35 秒后 NP hang。 | 设备恢复检查加延迟日志门禁；控制失败不试模型。 |
| 10-08 `decode-attribution-20261008/device/` | 全模型初始化到人为 150.19 秒期限，未进入推理；该 boot 未见内核故障，既往同类初始化需 181.51 秒。不能称为 NP hang。 | 总期限按初始化与请求量设置；不拿进程期限测图吞吐。 |

## 10-09 EVIS 投影完成等待停滞及释放后 NP hang（独立事故）

事件 `evis-projection-completion-wait-hang-20261009-111234`，boot
`fe501a53-7dd6-4a7f-b99b-4b028238eb53`。用户授权 VX 候选有限运行/性能估测并接受
故障后冷重启；此前两级控制与 VX 集合及其 45 秒门禁清洁。新增故障发生在随后
补跑的匹配 EVIS 对照；不能由此证明 VX 或 EVIS 引擎切换是根因。

| 时间（UTC+8） | SDK / 进程 / 内核证据 |
|---|---|
| 首轮执行前 | 48 张真实 FP16 单 MatMul 图、shared mutable 矩阵绑定，按 24 层 fc1/fc2 排序；48 次 Setup/Verify 全部返回成功，编译为 48 EVIS transB GEMM，operation 表完整 |
| 11:06:46.394354 | 应用进入粗验首轮，调用 `vxProcessGraph`；不上传数据、不读回 logits/KV；尚未开始暖机/计时 |
| 两次只读观察，最后 11:10:32.605978 | 自有追踪库的 8 字节调用计数均为 7；主线程 `gckOS_WaitSignal`，无进展。未读取/修改 SDK 或板卡内存；没有每个调用的起止日志，不能确认层号或第 7 次调用的精确开始时刻 |
| 11:12:05.370794 附近 | 决定停止本测试进程组；总进程墙钟 328.286 s，未耗尽 600 s 期限，exit -15；无粗验输出/有效计时批次 |
| 11:12:05.402120 | 内核 `release galcore` |
| 11:12:07.948236 | 即时检查无目标故障、无 owner，不能立即放行 |
| 11:12:34.546130 | `[galcore]: NP[0] hang, automatic recovery`，距 release 29.144010 s |
| 11:12:34.550461 / .550634 | `hardware->core:0 is idle` / `recovery done`；不清除门禁 |
| 11:12:55.319192 | 45 秒延迟检查发现上述新 hang；恢复标记保留，全部板卡工作停止 |

**期望 / 实际**：48 图集合返回并粗验末层两个投影，随后取得重复集合时间；实际
完成信号等待停滞，主动停止后有延迟 NP hang。没有本轮 EDMA/FCL 验图错误证据，
没有 EVIS 数值或性能通过结论。不能把主动停止归因成根因，或把旧 FFN 成绩当作
此次匹配基线。前一 VX 集合通过，只能说明其当时负载通过，不能证明故障设备健康。

**未证实假设**：48 个独立单 MatMul 图与已验 24 个组合 FFN 图的边界/分配/IO 差异；
前序 NN/CONV 负载后的 EVIS/Context 状态切换；直接 `vxProcessGraph` 与已验 FFN
`vsi_nn_RunGraph` 的调用差异。静态 RunGraph 包装确有 RNN/换句柄检查；本例无 RNN
且不交换句柄，尚无证据说明它们解释此故障。先离线比较，不靠重复吸收态试验筛选。

**下次规则**：保留 `.cache/runs/np101-recovery-required.json`，确认主机与 NP101
冷启动后，固定仓库卷积再冻结旧 OPT，各数值/执行/释放及至少 45 秒延迟门禁通过；
不原样重复这份 48 图独立 EVIS 对照。恢复后的基线优先复用已验组合 FFN 的准确结构
与调用入口；模型保留 SDK 自动选择，不引入 VX 默认路径。

证据 `.cache/runs/vx-gemm-estimate-20261009/evis/`：独立 `incident.json`、SDK/binary
哈希与执行记录、编译快照、`stalled-observations.json`、`kernel-incident.log`、完整
即时/延迟内核日志；原始 VX 成功结果独立保留在上级 summary，不被本事故覆盖。

## 10-08 17:35 冷启动原厂卷积再次 NP hang（独立事故）

用户确认冷启动；新 boot `0c3e9352-5ad3-45cf-8345-5cc3cfa11db6`，
17:34:42.281755 全 boot 预检无目标故障、无 owner。原标记完整归档。
只跑 `check_np101.py conv --cold-restarted --sudo-read-only`，同一冻结 demo
SHA256 `d51a13abb7f82594bbe3bc889ff9bf958690d6e028c9185ad751bb9b79174a9a`，
五个 SDK 哈希未变。已加载/磁盘模块 srcversion 均匹配 v105，未换库或插桩。
期望一次三节点/六 tensor 卷积推理、64/64 数值通过，实际只有设备打开提示。

| 时间 | 已确认事件 |
|---|---|
| 17:34:42.504036 | galcore 被 PID 4321 打开 |
| 17:35:07.460039 / .464190 | 首次 NP hang / recovery done |
| 17:35:08.949419 | 五秒监控检测故障并停止子进程；exit -15，26.584 秒，60 秒期限未耗尽 |
| 17:35:09.012029 / .309708 | release galcore / owner 为空 |
| 17:35:38.180032 | 第二次 hang，距首次 30.719993 秒、距 release 29.168003 秒 |
| 17:35:54.610216 | 至少 45 秒延迟检查失败，无 owner；未见 EDMA/BUG/Oops |

未经插桩，不能确定未返回的是 Context/Setup/Verify/Run/读回中的哪一步。
三次原厂失败的复发间隔约 30.720 秒，提示恢复/看门狗节律，不证明其实现或首次 hang 根因。
新主机 boot 已确认；NP101 电源/固件复位状态没有独立测量，不能据此认定硬件损坏。
旧 OPT、CPU 归因、映射校准、KV/block4 均未运行。保留新恢复标记并停测；
不自动消耗另一次同条件冷启动，先分析已有原厂对照或取得可检验的恢复变更。

独立事故 `vendor-convolution-cold-restart-hang-20261008-173507` 保存于
`vendor-convolution-cold-restart-20261008-1734/vendor-conv/` 的
`incident.json`、`device/execution-evidence.json`、`kernel-event-window.log` 及前后置日志；
上级目录有 `previous-recovery-marker.json`、`preflight.json`、`result.json`。

## 10-08 17:20 原厂卷积同启动授权重试（独立事故）

Boot 仍为 `b2e528c3-0b20-4e24-b19f-8d41c6939d7a`，未再次冷启动。
用户明确要求再试一次；只对该诊断临时允许已有故障，不修改仓库默认门禁，
保留原恢复标记及独立副本。预检无 owner，仅有 16:52 两条历史 hang；监控按新增日志判断。
原 demo SHA256 与上次一致，五个 SDK 哈希匹配 v105，无 SDK 插桩、strace 或新读回。

| 时间 | 已确认事件 |
|---|---|
| 17:19:30.610226 | galcore 被子进程 PID 8228 打开 |
| 17:20:00.470164 | 第一次新增 NP hang，随后 recovery done / Waiting for HDMA Transfer |
| 17:20:03.442251 | 监控发现新故障并终止进程；exit -15，32.989 秒，60 秒期限未耗尽 |
| 17:20:03.530223 / .833196 | 内核 release galcore / 检查 owner 为空 |
| 17:20:31.190235 | 第二次新增 hang；距第一次 30.720071 秒、距 release 27.660012 秒 |
| 17:20:50.145501 | 退出后超过 45 秒检查完成，无 owner，未见 EDMA/BUG/Oops |

期望一次推理及 64/64 数值通过，实际无数值输出；stdout 仅设备打开提示。
精确 SDK 调用仍未知，不能断言 Verify 故障。`Waiting for HDMA Transfer` 不等于 EDMA 超时。
两次运行的复发间隔分别为 30.720014 / 30.720071 秒，提示恢复/看门狗周期，
尚不证明其实现或根因。已有故障状态可能持续，也可能存在间歇性问题；这次重试不能区分两者，
不能估计清洁冷启动的失败概率。已加载及磁盘 galcore 的 srcversion 均为
`F53A3E6EFBB7D0FD7D4617C`，对应 v105；启动日志的 `6.4.9.9.452545` 是底层版本字符串。

独立事故 `vendor-convolution-retry-hang-20261008-172000` 保存于
`vendor-convolution-authorized-retry-20261008-1718/` 的
`incident.json`、`result.json`、`device/execution-evidence.json`、`kernel-event-window.log`
及前后置原日志。原标记保留并追加重试证据引用；不自动再试或进入模型任务。

## 10-08 16:52 未修改原厂卷积 NP hang（新增独立事故）

用户确认冷启动，boot `b2e528c3-0b20-4e24-b19f-8d41c6939d7a`。16:51:58.133302
预检全 boot 目标故障为零、无 owner；旧 KV 标记已归档。运行的是
`demo/op_demo_conv_relu_pool`，SHA256
`d51a13abb7f82594bbe3bc889ff9bf958690d6e028c9185ad751bb9b79174a9a`，
五个 SDK 哈希与 15:04 原厂成功及 16:31 KV 故障相同。3 节点/6 tensor 的原厂
CONV2D→RELU→MAXPOOL；期望一次 Run、64/64 数值及至少 45 秒延迟门禁通过。
没有 SDK timing、LD_PRELOAD、Verify guard 或 strace，没有重编译/替换二进制。

| 时间 | 已确认事件 |
|---|---|
| 16:52:21.590218 | 首次 `[galcore]: NP[0] hang, automatic recovery` |
| 16:52:21.598363 | `recovery done`；不能据此放行 |
| 16:52:24.792848 | 五秒监控读到 hang；停止子进程，exit -15、运行 26.613 秒，60 秒期限未耗尽 |
| 16:52:25.190523 | 退出后首次检查确认 owner 清空 |
| 16:52:52.310232 | 第二次 NP hang，相隔 30.720014 秒；距确认 owner 清空至少 27.120 秒 |
| 16:53:11.491715 | 至少 45 秒延迟检查失败，无 owner；未见 EDMA/BUG/Oops |

**调用定位边界**：未经修改的二进制没有逐调用时间戳，stdout 只有打开 galcore 的
提示；不能确定 hang 位于 Context 创建、Setup、Verify、Run 还是读回。不能用已退役
控制的 Verify 时间线代替本次定位。没有数值结果或正常 SDK 释放完成证据。旧 OPT、
Decode CPU 请求、两路 fc1、映射读回校准和候选验收均未运行。

**结论与假设**：NP hang 早于监控终止，且退出后复发；原厂控制自身也会在另一个
boot 失败。与同 binary/SDK 成功记录对照，削弱“改写控制是 hang 必要条件”的解释，
不能证明改写实现没有问题、硬件损坏或某个 API 必定出错。30.720 秒节律与旧故障
相同，是恢复/看门狗线索；SDK、驱动、固件或设备启动状态的根因仍未定位。

**规避**：保留新标记、停止当前 boot 板卡工作，不把新 CPU 映射功能放进故障设备
校准，也不尝试旧模型。先分析这份原厂成功/失败对照、核实恢复条件或取得厂商可检验
的变更，再安排恢复检查；不自动重复相同条件控制消耗冷启动周期。映射实现仅离线
构建，仍为显式候选；没有板卡通过证据。

独立事故为 `vendor-convolution-hang-20261008-165221`，证据在
`model-recovery-before-mapped-readback-20261008/vendor-conv/` 的
`incident.json`、`device/execution-evidence.json`、`kernel-event-window.jsonl`、前后置
日志与原始 `sdk.log`。成功对照为 `independent-controls-20261008-1502/summary.json`。

## 10-08 16:31 第 24 层 K 诊断读回 EDMA（新增独立事故）

- Boot `83269f49-a97a-4713-a840-0a6224bd06e4`，与 15:04–15:07 原厂卷积/冻结旧
  OPT 成功控制相同。启动前全 boot 目标故障为零，无 owner/恢复标记。本次为 v105、
  全 24 层两路 fc1、block1/C16、prompt16/5/16、最多 8 输出，只读末两层 KV。
  binary SHA256 `cb0141ddde73ed074ffb8386b10e4a6e099f6160f71c775b8b344d13b879a190`；
  五个 SDK 哈希见独立 `incident.json` 与 `device/execution-evidence.json`，与原成功控制相同。
- 初始化 79.340 秒完成；26 图、1957 内部节点、447 个 extension 的 operation 完整。
  应用 const/nonconst 峰值 2,334 / 678,277,756 B，未超过各段 1 GiB。不是 Verify
  未返回或缺 operation 的事故；进程被监控终止，不能声称 model.close/live 归零已验收。
- 第一请求消费 16 token、返回 `[21]`、停止原因为 capacity，匹配既有 CPU oracle。
  第 23/24 层 K/V 四份 backing 都已保存且数值通过，最大误差依次为
  0.0078125 / 0.00537109375 / 0.0078125 / 0.005859375。满槽请求没有未写后缀；
  第二/第三请求、A/B/A、未写槽与关闭生命周期未完成，不能替代完整验收。

以下均为主机本地时间，SDK 使用 Unix 时间戳，内核原始微秒时间另存 JSON：

| 时间 | 调用或内核事件 |
|---|---|
| 16:31:39.699419–.703463 | 第 23 层 K 的 `vsi_nn_ConvertTensorToData` 返回 |
| 16:31:39.703669–.720407 | 第 23 层 V 的同一 API 返回 |
| **16:31:39.720563–.747944** | **第 24 层 K 的同一 API，27.371 ms；内核超时在调用区间内** |
| **16:31:39.737244** | `gckEDMA_StartTransfer---timeout`，src `0x5098df00`、dst `0xffffffffffffffff`、len `0x8000`（32768 B）、channel 0 |
| 16:31:39.737382 / .737477 | `hdma_terminate_all size=0` / retry status -15 |
| 16:31:39.748185–.754370 | 第 24 层 V 的读回仍返回 |
| **16:31:39.754700** | 第二请求首次 `vsi_nn_CopyDataToTensor` reset 开始，晚于超时报错 |
| 16:31:43.666140 | 五秒监控读到超时，终止子进程；不是 600 秒总期限耗尽 |
| 16:32:29.750727 | 退出后至少 45 秒门禁仍见原 EDMA，无 owner；未见新增 hang/BUG/Oops |

调用链为 `opt_graph_generate` 的 KV 诊断循环 → `PartitionedGraphModel::read_cache`
（零基 layer=23、values=false）→ `np101::read_tensor` → `vsi_nn_ConvertTensorToData`。
对象为普通 FP16 `[1024,16]` K backing，32 KiB；主机重排发生在 SDK 读回返回之后。
完整日志配对在故障时只有该 API 在执行；这比此前 logits 的长度/文件时间关联更明确。

**已确认**：EDMA 超时报错落在该诊断 API 区间内；该 API 随后仍返回，输出正确，
设备门禁失败。下一请求 reset、监控 SIGTERM 和进程期限不能解释已经发生的报错。
进程 exit -15、运行 111.476 秒、无残留进程；最后未返回的调用为第二请求中的
`vsi_nn_RunGraph`，不是最初报错所在调用。**根因未定**：不能从外层区间证明内部
哪次 DMA 提交触发、排除此前传输的延迟报错、把 dst=-1 当成应用非法指针，或判定
两路 fc1 算术错误。逐调用日志扰动未单独量化，本次不作为性能测量。

**后续规则**：当前 boot 停止所有板卡工作并保留标记；双设备冷启动后按两级恢复控制
和各至少 45 秒延迟检查放行。不能因读回从 24 层减少到两层就认为安全，不原样重跑
这条批量转换路径。先离线核实文档支持的 CPU 映射/拷贝及同步语义，必要的小范围校准
通过后再用于 KV/完整 logits 验收；事故时替代路径尚未校准；后续mapped路径的有限校准及验收见实现总结。仅选词读回的暖机/重复
CPU 归因优先于大诊断传输；任何新故障仍即时停测。

证据：`.cache/runs/split-fc1-validation-20261008/kv-lifecycle/` 下的 `incident.json`、
`partial-audit.json`、`kernel-event-window.jsonl`、`device/sdk-calls.tsv`、编译表与前后置日志。
block4 和 CPU 集合夹具仅离线准备，没有后续板卡重试；恢复标记关联独立事故
`kv-readback-edma-20261008-163139`。

## 10-08 11:44 候选数值与生命周期验收 EDMA（原因未定）

证据目录：`split-fc1-acceptance-20261008/logits-block1/`。

- Boot `51a30d13-c421-427c-96c2-0c15f8024c52`；v105、OPT-350m C16、24 层
  两路 fc1、block1，三请求长度 4/15/4，最终完整 logits 读回。26 图/1957 operation
  均有效，应用内存与释放合规，三条最终 logits 对既有设备基线逐字节一致，进程 0 退出。
  binary SHA256 为 `a453a4a5c07d94f2af4345f05c344245636b9e0b8983f15f2d32901f20da4bf9`；
  主要 SDK 哈希见同目录 `device/execution-evidence.json`。
- 11:44:40.237928 内核 `gckEDMA_StartTransfer---timeout`，
  `src=0x55c24e40 dst=0xffffffffffffffff len=0x8c0 chan_id=0`，随后
  `hdma_terminate_all size=0`、retry status -15。首条 logits 的 13 次 SDK 读回已返回，
  文件最后写入早 6.46 ms；下一请求首先重置 48 个 KV。2240 B 等于 LM head 尾块，
  单个 KV 重置张量为 32768 B。旧读回事故也见 `dst=-1`，因而延迟上报的尾块读回
  是较强假设，但驱动可以分段传输；没有逐调用时间戳，**不能证明故障 API**，
  也不能排除 reset 或更早的图内部传输。旧标记的 reason 写“during logits readback”
  比证据更具体，保留原文但以后按本条解释。
- 数值通过与进程正常退出只证明本次结果；设备门禁失败。原标记归档于
  `recovery-after-logits-transfer-20261008/previous-recovery-marker.json`。根因与避免 EDMA 的充分条件
  尚未确定；用较少读回降低暴露面是审慎措施，不是已证明的修复。若恢复后继续既有
  KV 验收，优先用已有逐 SDK 调用时间戳定位区间，取消 ptrace ioctl 追踪；这轮诊断
  不用于吞吐比较，也不额外增加板卡探针。

## 10-08 11:59 设备恢复检查控制失败（前次 boot）

证据目录：`recovery-after-logits-transfer-20261008/`。

- Boot `a1d6ff2d-88cd-4f40-8225-cdebab8a4892`，主机已重新启动且 galcore 重新
  加载；NP101 是否断电尚待确认。与先前通过的三节点控制同 binary SHA256
  `e77822ba009fe29bd0c1279a98ea3125a619a6b1cb5d318a5ec8bfcaec55862f`、
  同 v105 SDK 哈希（见 `control/device/execution-evidence.json`）。预期 Verify、
  两次 Run 和数值均完成，且后置日志无故障。
- `vsi_nn_VerifyGraph` 从 11:59:03.253203 到 11:59:30.744170，耗时 27.491 秒；
  同 binary/SDK 在上午通过的控制仅耗 0.273418 秒；
  NP hang 在 11:59:30.702072，**发生在 Verify 返回前**；11:59:30.738202 报
  `recovery done`。第一次 `vsi_nn_RunGraph` 11:59:30.748809–.751148 返回，首轮
  数值误差 0.00323391；第二次 11:59:30.753753 进入后未返回，30 秒进程总期限
  于约 11:59:33 发 SIGTERM，exit -15。11:59:33.218123 设备释放，无残留 owner。
- 12:00:01.422084 又报一次 NP hang/自动恢复，约在释放后 28 秒。30 秒总期限
  不足以容纳本次异常 Verify 与两轮 Run，因此第二轮未完成不能单独证明它永久卡住；
  **两次内核 hang 已独立证明设备门禁失败**。不能把 hang 归咎于 SIGTERM：第一次
  hang 早于它，第二次说明故障可延迟上报。10-04 同三节点控制也曾在 Verify
  27.042 秒时 hang；本次仅 3 节点、6 张量，不涉及候选数值与生命周期验收的 fc1 改动或 1024 槽
  上限。这指向跨工作负载的 SDK/驱动/NP 状态问题，但主因在哪层尚未定位。
- 该 boot 的标记现已归档于 `convolution-control-hang-20261008-1436/previous-recovery-marker.json`。
  后续确认双设备冷启动、把控制总期限加至 60 秒后仍失败，见下条。

## 10-08 14:40 设备恢复检查确认冷启动后复发（后续标记已归档）

- 用户确认主机与 NP101 冷启动，boot
  `9d35f26d-e6cf-41a4-89dd-f21b5c772e1f`；启动前设备无人占用、目标内核故障为零。
  同 11:59 故障及上午健康对照的三节点/6 张量 FP16 控制，binary SHA256
  `e77822ba009fe29bd0c1279a98ea3125a619a6b1cb5d318a5ec8bfcaec55862f`，
  五个 v105 SDK 库哈希相同，完整哈希见 `control/device/execution-evidence.json`。
  预期 Verify、两次 Run、两次数值以及退出后的延迟内核门禁均通过。
- `vsi_nn_VerifyGraph` 14:40:10.031038–14:40:31.971197，耗时 21.940 秒；
  14:40:31.968664 内核报首次 NP hang，比 API 返回早约 2.53 ms。首轮
  `vsi_nn_RunGraph` 14:40:31.971577–14:41:02.690773，耗时 30.719 秒；
  14:41:02.688686 报第二次 NP hang，比返回早约 2.09 ms。首轮输出最大绝对误差
  0.00323391。第二轮 Run 于 14:41:02.693526 进入，60 秒**进程总期限**耗尽时
  尚未返回；SIGTERM 后 14:41:10.024746 释放 `galcore`，无残留 owner。
- 14:41:33.408701 又报第三次 NP hang，在设备释放约 23.384 秒后。三次报错相隔
  30.720022/30.720015 秒；先前 11:59 和 10-04 的报错间隔也约 30.720 秒。
  这是强烈的固定看门狗/恢复节律线索，**不证明最初卡住的层次或调用内部原因**。
  60 秒期限只使第二轮 Run 的完整等待不可见，不能解释前两次 hang。`recovery done`
  与正确输出都不表示设备健康；运行及当次后置观察窗内无 EDMA 超时或 kernel BUG；随后关机有 page fault，见下文。
- 这次在确认冷启动后用最小三节点图复发，候选数值与生命周期验收的 fc1 改动、大图内存和 1024 槽
  都不是此种 hang 的必要条件。问题位于 SDK/驱动/固件/NP 哪一层仍未知。
  当时的 `.cache/runs/np101-recovery-required.json` 对应此 boot（现已归档），原始证据在
  `convolution-control-hang-20261008-1436/incident.json`、`postflight.json`、
  `control/device/sdk-calls.tsv`。停止 KV/block4 与所有设备试验；不再靠原样冷启动
  重复设备恢复检查。下一步向 NP101 团队提供三次内核时间、逐 SDK 调用记录、二进制/库哈希
  和上午健康对照，请其核查看门狗、固件命令队列与恢复方式；取得可检验的变化后才
  考虑新的设备恢复检查门禁。

后续恢复策略见本页门禁；原厂与旧OPT在新boot的成功不能解释此事故根因。

## 10-08 两次故障 boot 的关机 page fault（新增独立事故）

用户补充关机卡顿后，使用授权的只读 sudo journalctl 补查历史 boot。两次都是
**驱动内核非法地址读取**，不是仅有长 Verify，也不是用户程序的 SIGSEGV。

| 独立事件 | boot / 内核故障时间 | 已确认结果 / 原始证据 |
|---|---|---|
| shutdown-page-fault-20261008-143027 | `a1d6ff2d-88cd-4f40-8225-cdebab8a4892`；14:30:27.191754，此前 11:59 设备恢复检查两次 hang | `plymouthd` PID 10101；14:31:55.194252 Xorg 停止超时被杀。`retired-convolution-control-20261008/boot-2-{kernel,shutdown}.log`、`shutdown-page-fault-20261008-143027.json` |
| shutdown-page-fault-20261008-145252 | `9d35f26d-e6cf-41a4-89dd-f21b5c772e1f`；14:52:52.343614，此前 14:40 设备恢复检查三次 hang | `plymouthd` PID 7716；14:54:21.054145 Xorg 停止超时被杀。`retired-convolution-control-20261008/boot-1-{kernel,shutdown}.log`、`shutdown-page-fault-20261008-145252.json` |

两次 `#PF: supervisor read access in kernel mode` / `not-present page` 均指向
`gckEVENT_Submit+0x1ff/0x720 [galcore]`，`CR2=0x0000000100000008`，
`R13=0x0000000100000000`。调用链如下（无问号的栈帧为已捕获链路；带问号的
DestroyProcessDB 只作上下文，不作为必经调用）：

```text
plymouthd close(fd=11)
  → drm_release → drm_file_free → viv_drm_postclose
  → gckKERNEL_AttachProcessEx → gckEVENT_Submit
  → 读取 [R13 + 8] → 内核 page fault
```

安装 `galcore.ko` 与 SDK 目录驱动 SHA256 相同：
`733f8b5834ac87231d5c24d9a07b625a43ee8136ca276de7e3eb63bf853a58fe`。
其反汇编在同一 `+0x1ff` 有相同指令字节，准确为
`mov 0x8(%r13),%ecx`；前面 `mov 0x5e0(%r15),%r13` 从 Event 对象取内部指针。
按已安装 `gc_hal_kernel.h` 的 x86-64 布局，`+0x5e0` 与 `queueHead`、队列内 `+8`
与 `source` 相符，因此**待提交事件队列头指针无效**是有结构依据的定位；这不是
应用 tensor backing 的地址。关机触发该 close 时，原设备恢复检查已退出，不存在对应的即时
推理 SDK 调用。两个 boot 的 Power Off Screen 都失败，随后会话停止耗尽约 90 秒。
日志证明这两个异常与关机等待共现，不能证明 Xorg 的全部等待都由该 Oops 引起。

**边界与假设**：页故障的直接原因已确认是驱动解引用无效内部指针；谁写坏它、何时
写坏、是否由首次 NP hang/自动恢复、进程超时终止或独立 close 路径竞态引起，仍未知。
未捕获内核写入过程，不能声称自写卷积控制直接越界写内核，也不能把 SIGTERM 当作
首个 hang 的原因。对照 boot `d6c7afa2-4734-42dd-bcdb-cd4948692eba`（旧卷积控制两轮通过）
和 `51a30d13-c421-427c-96c2-0c15f8024c52`（候选数值与生命周期验收 EDMA）均无同类关机 page fault；
这加强“NP hang 后残留驱动事件状态”的关联线索，不构成单变量因果证明。

**规避**：当时退役自写卷积控制，后按恢复判定约定复用其固定归档；不做故障复现。
故障后即使 owner 清空也保持门禁，因为共享
DRM 驱动状态可在另一进程关闭时再次崩溃。后续检查必须识别 `BUG:`/`Oops:`、page
fault/general protection fault，而不只匹配 hang/EDMA。不卸载驱动、不改 DRM/图形
配置，不用关机重现验证假设。事故后的新boot控制成功不覆盖这两个独立事故；原标记继续归档保留。

## 卷积与关机故障的离线结论

证据：`graph-construction-comparison-20261008/`、`retired-convolution-control-20261008/`、
`convolution-event-offline-20261009/{control-comparison.json,provenance.json}`，包含源码、
SDK/驱动/头文件哈希、反汇编与原始时间线。未找到事件实现源码，模块无DWARF/BTF。

### 建图差异与排除范围

原厂卷积先分配IO槽/节点再建tensor；改写控制先建tensor/节点再设IO；冻结旧OPT按
依赖交错创建，IO最后设置。两种卷积都是3节点/6tensor、相同FP16/const/virtual有效
契约；旧OPT同样使用RAII、显式shape及IO setter，所以这些机制本身不足以解释失败。

| 因素 | 已核实内容与解释边界 |
|---|---|
| AUTO/virtual | 两种卷积中间tensor均为AUTO/virtual，旧OPT通常显式普通tensor。AUTO在 `_create_tensor` 保留空句柄，Setup推导shape后reinit/compute；不是按零shape直接分配 |
| bias尾部 | 原厂rank1 `[4]` 的未用size尾部为3/3/4，改写版为零；`vxCreateTensor2` 普通路径只传dim_num个维度，卷积消费OpenVX句柄。削弱“尾部零初始化致错”解释，但未证明所有优化路径都不读尾部 |
| IO与node参数 | setter分配并复制ID，不保留局部ID地址；未发现pool.local被覆盖。分配顺序/堆布局/SDK状态仍可能影响特定路径，不能凭差异判根因 |
| Verify | 内部包含分配、优化、初始化、命令生成及可能设备活动，不能全算CPU编译。自写首次hang已在Verify内，后续上传/Run/读回/终止不能解释首次触发 |
| 并行与同身份结果 | 应用均串行；自写binary `e77822ba…55862f`、原厂 `d51a13ab…74a9a` 各有成功/失败，九次记录五库哈希一致。17:20为故障boot授权重试，不能当独立冷启动样本或估计概率 |

### 事件队列与关机崩溃

```text
Event.repoList[3] → freeList → AddListEx → queueHead/queueTail
  → Submit/GetEvent → queues[28或29个硬件事件槽] → Notify处理信号/释放
进程close → AttachProcessEx → Submit → 读取queueHead+8 → page fault
```

- `_gckEVENT`布局与指令一致；队列头在Event+0x5e0，source在head+8，两次Oops均
  解引用无效head=0x100000000。这是驱动内部事件指针，不是tensor物理地址。
- Construct分配0x6a8 B对象并初始化repoList；AddListEx和Submit在eventListMutex下
  连接/回收队列。Submit/GetEvent还取eventQueueMutex，Notify取走记录后释放锁再
  处理信号；Event_Destroy停止timer并释放对象。这些锁不证明所有调用者正确同步。
- Recovery调用Hardware_Reset/Notify；已查函数体未见直接重建head/tail/freeList，
  不能外推整个间接调用树。未捕获第一次异常写入，也未证明Event已释放或具体竞态。
- Monitor按进度与timeOut触发Recovery/重排timer，支持约30.72秒看门狗节律线索；
  不证明首次卡住原因，更不能用该节律推导Verify算法复杂度。

**结论与后续**：首次Verify hang和关机close非法指针读取是两个已定位边界，其间
因果仍未闭环。优先用已有证据向厂商核实head/next/freeList写入者、Reset/Notify/
Destroy/close同步、失败Verify返回成功及已知修复；外发需授权。避免再次用创建顺序、
bias尾部或关机进行吸收态试验，不卸载驱动或改DRM配置；遵守固定恢复门禁。
