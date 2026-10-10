# NP101 冷启动事件与复发规避

本页单独追踪使板卡测试停下、需要确认恢复的事件。它记录**能定位到的调用**，不把
时间相邻或传输长度相同当成根因证明。原始运行文件位于 `.cache/runs/`，不随 Git 分发；
SDK 问题的操作说明仍在 [驱动报告](np101-sdk-driver-validation-report.md)。同一故障的
复制标记按 boot ID、内核时间和运行目录去重，不按文件数计数。

## 每次板卡运行前的门禁

1. 查当前 boot ID、`np101-recovery-required.json`、`/dev/galcore` owner，以及当前 boot
   的 NP hang、`gckEDMA_StartTransfer---timeout`、`BUG:`/`Oops:`/page fault/general protection fault/invalid opcode/soft lockup。
   `Waiting for HDMA Transfer...` 单独出现不算 EDMA 超时。已有故障时停止；不能因进程
   退出码为 0、数值正确或内核写了 `recovery done` 而重试。
2. 如需冷启动，保留并归档原标记及证据。确认**主机与 NP101**恢复后，在清洁新 boot
   先跑固定的仓库自写卷积控制，再跑冻结旧 OPT component C16 prompt5/两个输出
   （180 秒、0 暖机/1 次）；两项通过后不强制运行原厂 demo，详见下方最新约定。
   两级均不开 SDK 插桩；期限不表示可接受性能。每级结束后至少
   再看 45 秒内核日志和 owner；历史 hang 曾在释放 28–35 秒后出现。控制或延迟门禁
   失败就停止，不进入模型测试。同 binary/SDK 的控制若在确认冷启动后再次失败，
   暂停相同条件的板卡重试，先用已有证据分析或提交厂商；不连续消耗冷启动周期。
3. 新事故立即保存独立记录：时间与 boot ID、二进制/SDK 哈希、输入与图结构、最后
   已返回/未返回的 SDK API、内核微秒时间、期望/实际输出、进程与 owner 状态、恢复
   标记、可证实原因与未证实假设、下次规避动作。先查下表的相同指纹，再决定候选。

## 恢复判定约定（10-08 用户更新）

- 日常放行条件改为**固定仓库自写卷积 + 冻结旧 OPT 短请求**串行通过；两者均通过后，
  不要求再跑原厂 demo。原厂保留为可选排查/厂商复现负载，不删除其失败证据。
- “通过”同时要求设备计算数值正确、正常退出/释放、预算合规、无残留 owner；
  两项各自运行中及退出后至少 45 秒无 NP hang、EDMA 超时或其他目标内核故障。
  任意一级失败即停测，不以另一级成功覆盖；设备选择和冻结 binary/SDK/输入身份必须记录。
- 这是“本次已测负载下可继续开发验证”的工作判据，不证明所有硬件、算子、负载均正常，
  也不证明原厂故障或底层驱动问题已修复。环境身份相同不能排除设备运行状态差异。
- 自写卷积指归档的 `retired-convolution-control-20261008/` 控制，源码和二进制保留。
  它有历史成功，也有 Verify hang/关机异常的关联记录；本轮 18:02 自写卷积、18:03
  冻结旧 OPT 及各至少 45 秒延迟检查已通过，见下节，不改写旧事故的结论。
- 调度入口已调整：`conv` 直接复用固定哈希的归档自写卷积，不重编译；`recovery` 再运行
  冻结旧 OPT；`vendor-conv` 为可选原厂诊断。自写源码/二进制与 SDK 身份在恢复清单中固定，
  缺失或改变则运行前拒绝。恢复标记按冷启动及两级检查处理，不因策略切换清除。
  下文退役/原厂必跑描述均是此前处理记录，后续恢复策略以本节为准。

## 10-08 18:02–18:03 自写卷积与冻结旧 OPT 恢复通过

用户确认冷启动，新 boot `fe501a53-7dd6-4a7f-b99b-4b028238eb53`，预检清洁、无 owner。
归档自写卷积 binary/source 与 v105 SDK 固定哈希匹配，未重编译/插桩；两轮误差
0.00323391 / 0.00311446，显式释放、exit 0，进程 0.216765 秒。
18:02:50.486543 延迟检查清洁。随后冻结旧 OPT 全 24 层 C16 prompt5/两个输出
`[78,631]`，消费 6 token、max_new_tokens；进程 21.984068 秒，初始化 5.81062 秒、
请求 8.884368 秒、释放 1.44816 秒（这些量不能直接相加为进程墙钟）。应用 const/nonconst
峰值 610,916,276 / 61,080,856 B、关闭后 live 归零；18:03:59.370937 延迟检查清洁。
两级各退出后超过 45 秒，无 EDMA/hang/BUG/Oops/owner；原标记完整归档。

证据 `.cache/runs/repository-recovery-20261008-1800/`：`result.json`、各级 execution/
前后置日志、`previous-recovery-marker.json`。依据新约定允许继续已准备的 CPU 归因和
候选验证，不跑原厂 demo；这不是底层故障修复或跨启动稳定性证明。
后续 CPU 52 项、映射 16 次精确输出、KV/生命周期 58 项、block4 72 项均通过，
每轮各至少 45 秒延迟门禁清洁，最后 18:13:31.828531 无故障/owner/恢复标记。
诊断 Map/Unmap 已覆盖 KV 12 次、logits 39 次，仅选词仍用 SDK Convert；
原 EDMA 原因未定位，不把有限次数通过解释为永久规避。证据见
`.cache/runs/recovery-followup-20261008-1802/summary.json` 及各轮原始文件。

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

**后续状态（10-09）**：用户确认主机与 NP101 冷启动，当前主机 boot 为
`f214ee36-6fd8-42cd-922a-d5095f9980bd`；固定两级控制尚未执行，恢复标记保留。
用户决定关闭该后端探索，已删除 VX 创建探针、集合测试及匹配 EVIS 对照的临时
源码、脚本与独立二进制。原件仅保存在上级 `retired-diagnostic-artifacts.tar.gz`，
路径与 SHA-256 见 `retirement.json`；原始测量、事故和内核日志未修改。
不再安排该对照复测；本轮清理没有执行板卡操作。

**恢复检查（10-09 11:33–11:34）**：同一新 boot，固定哈希的仓库卷积两轮误差
0.00323391 / 0.00311446、正常释放，进程 0.217432 秒；冻结旧 OPT 输出 `[78,631]`、
预算/释放通过，进程 20.956351 秒。两级均无插桩，各退出后至少 45 秒延迟日志清洁，
11:33:27.158501 / 11:34:33.882395 无目标故障及 owner。前次标记完整归档于
`decode-working-set-20261009/recovery/previous-recovery-marker.json`；同目录 `result.json`
及两级执行/内核记录保留。允许继续已准备的 FFN 权重工作集对照，不证明故障修复。
后续不同权重→共享一套→不同权重的三组集合及各 45 秒延迟检查均通过，最终
11:40:32.381516 无目标故障/owner/恢复标记；证据在该目录的三组结果及 `summary.json`。
随后单输入→四输入→单输入 FFN 集合的执行/数值/释放及各 45 秒门禁亦通过，
最终 11:55:44.434123 日志清洁、无 owner/标记；证据在 `ffn-input-width-20261009/`。
CPU 参考导出形状错误仅离线纠正，复用已完成的单输入执行，不算板卡恢复事故。

同 boot 后续两路→四路→两路的 FFN 集合及整模型短请求对照，各三组执行/释放和
各至少 45 秒门禁通过，未新增内核故障；四路模型因收益有限、槽位余量减少而未采纳。
随后两路 C64 最终 logits/容量 70 项、末两层 KV/A/B/A 99 项通过；39/12 次映射
诊断无 SDK-copy，两组各至少 45 秒门禁清洁，最后 12:33:09.353280 无 owner/标记。
证据 `ffn-channel-shape-20261009/`、`fc1-shape-decode-20261009/`、
`split-fc1-context-20261009/`。这些是本次所测负载成功记录，不改变任何历史事故根因
或永久规避结论；本轮没有新的需冷启动事件。

同 boot 的更大 FFN 批量 M4→M8→M4→M16→M4 五组集合，以及完整模型
block4→block16→block4、block16 最终 logits/末两层 KV 两组粗验均通过。
各组正常执行/释放、预算合规并各至少 45 秒延迟门禁清洁；最后
14:53:51.389226 无目标故障/owner/恢复标记。本轮没有新的需冷启动事件；未扩至
M32、跨启动或独立请求并行。证据 `ffn-batch-scaling-20261009/`、
`prefill-batch-scaling-20261009/summary.json`；历史事故根因及每次运行规避规则不变。

## 历史指纹与规避

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
| **10-08 11:44 候选数值与生命周期验收** `split-fc1-acceptance-20261008/logits-block1/` | `gckEDMA_StartTransfer` 超时，详见下文。 | 本启动停测；后续只在清洁设备恢复检查后做已准备的有限 KV/block4 验收。 |
| **10-08 11:59 设备恢复检查** `recovery-after-logits-transfer-20261008/control/device/` | 三节点控制 Verify 期间 NP hang，释放后延迟复发，详见下文。 | 该 boot 已停测、标记已归档；随后确认双设备冷启动仍复发。 |
| **10-08 14:40 设备恢复检查** `convolution-control-hang-20261008-1436/` | 确认主机与 NP101 冷启动后，同 binary/SDK 三节点控制又有三次 NP hang，详见下文。 | 该故障 boot 停测；暂停同条件重试，先将证据提交 NP101 团队核查。 |
| **10-08 16:31 KV 诊断读回** `split-fc1-validation-20261008/kv-lifecycle/` | 第 24 个 decoder 的 K backing `vsi_nn_ConvertTensorToData` 调用内发生 32 KiB EDMA 超时；随后 SDK 返回且数值正确，监控停测，详见下文。 | 保留标记；不原样重复多 tensor KV/完整 logits 读回。先离线审查 CPU 映射/拷贝路径；恢复控制通过后，优先仅选词读回的 CPU 集合归因。 |
| **10-08 16:52 原厂卷积恢复控制** `model-recovery-before-mapped-readback-20261008/vendor-conv/` | 确认冷启动后，哈希匹配的原厂 demo 报两次 NP hang；无插桩、未运行旧模型或新读回，详见下文。 | 本 boot 停测；原厂路径也可能失败。先分析成功/失败原厂证据并核实恢复条件，不自动重复冷启动控制或归因于改写控制。 |
| **10-08 17:20 原厂卷积授权重试** `vendor-convolution-authorized-retry-20261008-1718/` | 用户明确授权同故障 boot 再试一次；原二进制/SDK 新增两次 hang，32.989 秒被监控停止，退出后仍复发。 | 原恢复标记保留；此次仅为授权诊断，不放行模型、不自动扩大重试。不能用同故障 boot 的结果估计冷启动故障概率。 |
| **10-08 17:35 冷启动原厂卷积** `vendor-convolution-cold-restart-20261008-1734/vendor-conv/` | 用户确认冷启动、新 boot 清洁预检后，原厂卷积再次两次 NP hang；26.584 秒被监控停止，无数值输出。 | 保留新标记，停止本 boot 板卡工作；不自动重复同条件控制，不进入旧 OPT/新候选。 |

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

独立事故 `vendor-convolution-cold-restart-hang-20261008-173507` 保存于上表目录的
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

独立事故 `vendor-convolution-retry-hang-20261008-172000` 保存于上表目录的
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

独立事故为 `vendor-convolution-hang-20261008-165221`，证据在上表目录的
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
通过后再用于 KV/完整 logits 验收；事故发生时替代路径尚未实现，现已离线实现但未做板卡校准。仅选词读回的暖机/重复
CPU 归因优先于大诊断传输；任何新故障仍即时停测。

证据：`.cache/runs/split-fc1-validation-20261008/kv-lifecycle/` 下的 `incident.json`、
`partial-audit.json`、`kernel-event-window.jsonl`、`device/sdk-calls.tsv`、编译表与前后置日志。
block4 和 CPU 集合夹具仅离线准备，没有后续板卡重试；恢复标记关联独立事故
`kv-readback-edma-20261008-163139`。

## 10-08 11:44 候选数值与生命周期验收 EDMA（原因未定）

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

用户提出用原始成功负载区分平台故障与设备恢复检查实现差异。已确认 `demo/op_demo_conv_relu_pool`
未改动的二进制在同 v105 SDK 下曾通过，2026-10-01 冻结的旧 OPT component 也有
完整成功记录。下次恢复后先单独运行原始 demo，延迟门禁清洁才运行旧 component
短请求；任一项异常立即停测，不在故障 boot 上试。具体命令、哈希与验收顺序见
[TODO](../TODO.md)。两项通过只能证明这些路径当次可工作，不能证明所有 NP 单元完好。

## 10-08 15:04–15:07 独立成功负载恢复对照

用户确认双设备冷启动，boot `83269f49-a97a-4713-a840-0a6224bd06e4`；预检无故障
和 owner，14:40 标记归档于 `independent-controls-20261008-1502/previous-recovery-marker.json`。
未修改的原厂 demo（上文 SHA）64/64 输出匹配，最大误差 0.003234、进程 0.924 秒；
释放后约 70 秒门禁清洁。随后冻结旧 OPT component（TODO 所列 SHA）C16 prompt5、
两输出、0 暖机/1 次通过，全 24 层输出 `[78,631]`，应用预算及释放合规；进程
80.126 秒，释放后约 57 秒门禁清洁。五个 SDK 哈希均与旧成功及改写卷积控制故障版本相同。
没有 hang、EDMA 超时、BUG、invalid opcode、soft lockup、MCE，无 owner 或新恢复标记。
原始日志、执行证据和检查在 `independent-controls-20261008-1502/`。

这证明两条已知负载本次可工作，削弱“板卡完全不能运行”的判断；不能证明所有硬件
完好、改写控制实现导致故障或驱动已修复。没有重跑改写控制，也未继续两路 fc1 验收。
随后离线核查三方建图顺序、tensor 属性与 IO 绑定，结果见下文。

## 原厂卷积、改写控制与旧 OPT 的建图比较（10-08，离线）

对照对象为未修改的 `demo/main.c`、已退役改写卷积控制的归档源码，以及冻结旧 OPT
component 的模型、decoder 与公共建图代码。冻结旧 OPT 对应 10-01 性能基线、10-08
恢复对照通过的二进制；不能把它直接称为更早的第一份完整单图实现。归档的模型和
decoder 源码哈希与冻结记录匹配，Context/tensor/cache 实现与当前源码相同；建图
helper 的核心路径已逐项核对，但冻结记录没有它的独立源码哈希。

源码与比较证据在 `.cache/runs/graph-construction-comparison-20261008/`；改写控制源码、
二进制、setter/Verify 反汇编在 `.cache/runs/retired-convolution-control-20261008/`。
本轮不运行 NP101，也不通过再次 hang 逐项筛选原因。

### 创建顺序

```text
原厂卷积
  Context → Graph(6 tensors, 3 nodes)
  → 分配图 IO 槽 → 3 个节点 → 6 个 tensor → 连接节点/填写 IO ID
  → 上传输入 → Setup(FALSE) → Verify → Run/读回一次 → Graph/Context 释放

改写卷积控制（已退役）
  Context → Graph(6 tensors, 3 nodes)
  → 6 个 tensor → 3 个节点 → 连接节点 → 设置图 IO
  → 上传输入 → Setup(FALSE) → Verify
  → 再上传/Run/读回 → 换输入/Run/读回 → Graph/Context 释放

冻结旧 OPT component（逐层小图）
  Context → embedding / projection / cache-copy / attention+FFN / 输出小图
  → tensor 与节点按依赖交错创建 → 最后设置各图 IO → Setup(FALSE)/Verify
  → 逐 token 串行执行并更新 KV → 复用图 → Graph/Context 释放
  注：cache-copy 使用直接 OpenVX copy/view/Verify，不能套用普通 VSI 图流程。
```

### 三方对照

| 因素 | 原厂卷积 | 已退役改写卷积 | 冻结旧 OPT component | 判断 |
|---|---|---|---|---|
| 图规模/结构 | 3 节点、6 tensor；CONV2D→RELU→POOL | 同样 3 节点、6 tensor、shape、参数和 seed | 多张 projection/cache/tail 小图，全 24 层；不含该卷积链 | 大图规模、1024 包装槽位不是卷积故障的必要条件 |
| 创建及 IO 时序 | 先 IO/节点，后 tensor；早分配槽再写 ID | 先 tensor/节点，后 IO setter | 多数先创建 tensor，再创建使用它的节点；IO setter 最后调用 | 先 tensor/后 IO 并非普遍不可用；与虚拟 tensor/卷积优化组合的影响仍未知 |
| 输入/权重/输出 | FP16、NCHW；输入/输出普通 tensor，权重/bias 为 const | 相同有效 shape、dtype、格式和 const/virtual 属性 | 普通显式 shape 的中间 tensor；默认图内 const 权重；FP16 主体与 FP32 敏感操作 | 卷积两者的公开有效契约相同；旧 OPT 与卷积的算子优化路径不同 |
| 中间 tensor | 两个 `vtl=1`、`dim_num=VSI_NN_DIM_AUTO=0`、size 全零 | 同原厂 | 普通中间 tensor 通常 `vtl=0`，shape 显式；另有 KV view | 自动 shape/虚拟分配路径值得查，但不是两种卷积间的属性差异 |
| bias 元数据 | rank=1，有效 size 为 `[4]`；复用 attr，未使用尾部为 `[3,3,4,0,0,0,0]` | rank=1，有效 size 为 `[4]`；独立零初始化，尾部全零 | 独立零初始化显式 shape；MatMul 后 ADD 实现 bias | 找到了原厂/改写卷积的具体字节差异；尚无证据 SDK 读取 rank 之外的 size，不能宣称零初始化错误 |
| node 初始化 | 显式 arity，逐字段设 conv/pool 公共参数 | 相同 arity/参数；不清零 nn_param、不覆盖 pool.local | 通常 arity=0 由 SDK 推导，逐字段设置公共参数 | 未发现此前 pool.local 被误写的同类确定错误 |
| IO ID 所有权 | setter(NULL) 分配后直接写槽 | setter(&id) | setter(vector IDs) | 安装库 setter 分配并 memcpy ID，不保留调用者地址；排除局部 ID 悬空解释 |
| 封装/分配/寿命 | C 直接调用；无应用预算 map | C++ RAII、预算 map、API 日志 flush；先释放 Graph 再 Context | 同类 Context/Graph/tensor 封装与预算；跨图 retain、KV view/copy | 封装并非故障路径独有；分配时序、堆布局和 SDK 状态敏感性尚未排除 |
| Verify 后操作 | 一次 Run/读回 | 两次上传/Run/读回 | 重复执行、运行期 KV copy 绑定/验图 | 首次 hang 已在 Verify 内，后续操作不能解释首次触发 |
| 并行/同步 | 应用串行 | 应用串行 | 应用逐层串行 | 没有应用多图并行这一特有差异；SDK 内部提交/同步/恢复机制仍未知 |

两种卷积的 shape 均为输入 `[8,8,3,1]`、权重 `[3,3,3,4]`、bias `[4]`、输出
`[4,4,4,1]`。原厂 bias 的完整八维存储是 `[4,3,3,4,0,0,0,0]`，改写版是
`[4,0,0,0,0,0,0,0]`；这只是未使用字段差异，不改变声明的一维契约。

### 故障边界与解释强度

14:40 首次 hang 在 `vsi_nn_VerifyGraph` 返回前 2.53 ms。该 API 在当前库仅取
`graph->g` 后跳转 `vxVerifyGraph`；内部 `vxoGraph_VerifyGraph` 包括内存分配、
图优化、kernel 初始化、operation 表与命令生成。没有内部运行 trace，不能指定
某个子函数卡住，也不能把 Verify 理解为纯 CPU 参数检查。

- **已确认**：首次故障早于 Verify 后上传/重复 Run；两种卷积有效契约一致，存在创建
  顺序、IO 分配时机、bias 未使用字段和主机分配/日志时序差异。
- **仍有价值的假设**：虚拟/AUTO tensor 与卷积优化的组合受创建/分配时序影响；驱动或
  SDK 的未初始化状态、堆布局或恢复状态敏感。现有证据不能对这几项排序或给出概率。
- **被削弱的解释**：单凭先 tensor 后 node、后设置 IO、C++ RAII、零初始化显式
  tensor 就认定不支持；成功旧 OPT 也采用这些机制。不是单变量对照，仍不能证明
  它们与特定卷积路径组合无关。
- **不能作为根因的后续操作**：重复上传、第二次 Run、读回和进程超时终止均晚于
  首次 Verify hang；它们可能加重故障后的状态，但没有解释最初的触发。

同一改写控制二进制 SHA 与五个 SDK 哈希在上午曾成功：Verify 0.273 秒，两次 Run
和约 75 秒延迟检查均通过。下午故障 Verify 为 27.491/21.940 秒；第二次故障 boot
后续 hang 间隔约 30.720 秒。这支持状态/看门狗节律线索，不支持已经发现每次必现的
固定参数错误。原厂与旧 OPT 是新 boot 的成功负载，不是故障 boot 的单变量因果实验。
安装库与 include 路径未发现不匹配，但没有据此证明完整 ABI 无误。

关机日志另确认驱动事件队列内部无效指针读取，见下文；不能反推应用哪项建图操作写坏
了它。后续先查虚拟 tensor 的 Setup/优化数据流、公开字段的读取范围和 Event 恢复/
释放链，以现有源码、反汇编和日志为证据，不重复吸收态故障试验。

**当时的调整（后被本页恢复判定约定替代）**：恢复门禁直接使用成功原始负载。改写控制的 CMake target/源码及
常规 build 二进制已移除，归档证据保留。`check_np101.py conv` 使用原厂 demo；
`recovery` 再串行运行冻结旧 OPT 短请求。每级检查数值与至少 45 秒延迟内核门禁；
长进程每五秒查内核，异常终止子进程并留恢复标记。清单固定在
`tests/fixtures/np101-recovery-controls.json`，身份不符提前拒绝，不重编译替换。
对已成功的两条负载本轮不重复板卡验证。

### 调查方式与命名约定（10-08 用户确认）

- 会使驱动持续故障、需要冷启动的 hang 探针不能作为常规逐项排除手段。当前卷积
  故障优先通过已有日志、成功负载的建图差异和离线 SDK/驱动分析调查；本次不新增
  板卡复现。创建顺序与 bias 未使用字段只是差异，尚未证明其导致首次 Verify hang。
- 源码、模块、夹具、命令、输出名称和对外文档统一使用功能名称。设备恢复模块为
  `python/specferry/validation/recovery_controls.py`，身份清单为
  `tests/fixtures/np101-recovery-controls.json`，恢复命令为 `check_np101.py recovery`；
  导入、测试和使用说明同步更新，不保留任务代号兼容入口。
- 原始证据及源码快照保持不变，不随 Git 分发；当前文档只使用描述性证据别名。
  后续新增记录沿用功能名称，避免将对话中的任务编号固化成接口。

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
配置，不用关机重现验证假设。当前新 boot 的原厂/旧模型控制通过，不给它补旧 boot
恢复标记；两个独立事故及旧标记继续归档保留。

## 卷积 hang 与关机故障的离线追踪（10-09）

本轮只读取已保存的执行/内核证据、公开头文件和安装库/驱动反汇编，没有运行板卡、
重建固定控制或修改模型。新增证据在 `.cache/runs/convolution-event-offline-20261009/`：
`control-comparison.json` 保存九次控制身份与可用时间线，`provenance.json` 固定原证据、
库、头文件和定向反汇编哈希。安装驱动与两次关机 Oops 的指令相符；安装/SDK 包内
查到内核头文件及预编译对象，未找到事件实现源码，模块无 DWARF/BTF。

### 同身份成功与失败

下表 boot 使用唯一前缀，完整值及证据路径在比较 JSON。自写卷积二进制 SHA256
始终为 `e77822ba…55862f`，原厂为 `d51a13ab…74a9a`；九次的五个 SDK 库哈希
完全一致，并与当前安装文件匹配。时间为 UTC+8；进程墙钟包含准备/释放，不是 Verify
或推理性能。无逐 API trace 的原厂运行不能指定 hang 落在 Setup、Verify 还是 Run。

| 控制 / boot | 既有结果与时间 | 解释边界 |
|---|---|---|
| 10-08 09:14 自写 / `d6c7afa2` | Verify 0.273418 s，两次 Run/数值及约 75 s 延迟检查通过 | 与两次失败均启用 SDK 调用计时；计时存在本身不充分解释故障 |
| 11:59 自写 / `a1d6ff2d` | Verify 27.491 s，11:59:30.702072 首次 hang 比 Verify 返回早 42.098 ms；首次 Run 返回，第二次未返回 | release 11:59:33.218123 后仍有 12:00:01.422084 hang；关机另有 page fault |
| 14:40 自写 / `9d35f26d` | Verify 21.9402 s；首次 hang 比返回早 2.533 ms；首次 Run 30.7192 s，第二次未返回 | 三次 hang 间隔 30.720022/30.720015 s；release 后 23.383955 s 再 hang，关机另有 page fault |
| 15:04 原厂 / `83269f49` | 进程 0.923541 s，64/64 数值及约 70 s 延迟检查通过 | 后续冻结旧 OPT 亦通过 |
| 16:52 原厂 / `b2e528c3` | 确认冷启动、前置日志清洁且无 owner；进程 26.613422 s，监控发现 hang 后终止 | 两次 hang 间隔 30.720014 s；未追踪 SDK 内部调用 |
| 17:35 原厂 / `0c3e9352` | 再次确认冷启动、前置日志清洁且无 owner；进程 26.584102 s，hang 后终止 | 两次 hang 间隔 30.719993 s；不能归因成自写建图独有错误 |
| 18:02 自写 / `fe501a53` | 进程 0.216765 s，两轮数值及延迟门禁通过，随后旧 OPT 通过 | SDK 计时关闭，不与上午单个 API 时间直接比较 |
| 10-09 自写 / `f214ee36` | 进程 0.217432 s，两轮数值及延迟门禁通过，随后旧 OPT 与已记录候选通过 | 仅放行已检查负载范围，不证明卷积故障修复 |

17:20 原厂重试属于用户授权的同故障 boot 诊断，单独保留在 JSON，不当作独立冷启动
样本。成功与失败前置检查均有清洁记录；这不等于捕获了启动以来全部隐藏状态或
证明无其他 DRM 使用者。现有样本不能估计故障概率，也未找到足以区分结果的环境因素。

### tensor 与 Setup/Verify 的实际机制

| 检查项 | 已确认的静态路径 | 对原假设的影响 |
|---|---|---|
| AUTO 中间 tensor | `libovxlib` `_create_tensor` 的 `0x41f86–0x41fa9` 在 dim_num=0 时保留元数据/空 OpenVX 句柄，不调用 `_init_tensor`。Setup 调用算子 shape 回调、优化，再 `TensorReinit`/`OpCompute`；CONV2D setup 在 `0x5925e` 检查 AUTO 并推导输出 | AUTO 不是直接把零 shape 当成实际 backing 分配；两种卷积均使用 SDK 的延迟创建机制 |
| virtual 创建 | `_init_tensor` 检查 attr.vtl（`+0x24`），虚拟分支 `0x41b88–0x41b96` 调用 `vxCreateVirtualTensor2`；普通分支调用 `vxCreateTensor2` | 与旧 OPT 显式普通中间 tensor 有真实路径差异，但没有其导致 hang 的证据 |
| bias 尾部 | `_init_tensor` `0x41928–0x41936` 确实复制全部八个 size；OpenVX `vxCreateTensor2` 的 `0xb2512–0xb252f` 却只转换/传递 dim_num 个元素，其余内部维度先清零 | 普通创建入口不会把原厂 rank=1 的 `[3,3,4,…]` 尾部传给内部 tensor；“尾部零初始化导致创建不同”在这条路径上被削弱，不证明整个 SDK 从不读尾部 |
| 卷积消费 bias | 普通 VX 卷积 setup `0x13e818–0x13e847` 取 bias 的 OpenVX 句柄传给 `vxConvolutionLayer`，不是传完整 VSI attr | 进一步削弱尾部经普通卷积入口影响计算的解释；未捕获故障运行实际选择的优化/融合分支 |
| IO 初始化 | 既有 setter 反汇编确认分配/复制 IDs；Setup 的排序、shape 回调、TensorReinit 和 Compute 消费已绑定的张量 | 没有局部 ID 悬空的证据；早/晚分配可能改变布局与时序，不能据此认定非法 |
| Verify 边界 | 包装跳转 `vxVerifyGraph`，内部有分配、优化、初始化与命令生成；首次 hang 在该调用返回前 | 不能把长时间全部算作 CPU 编译，也不能仅由返回成功推断设备正常 |

这是静态数据流审查，不是故障现场内部 trace。Setup 本身含图优化，实际是否融合及
哪一条 kernel 分支执行仍未知；因此不据此修改已验模型或断言所有下游路径等价。

### 驱动队列所有权与回收

公开 `_gckEVENT` 布局与模块指令一致。Construct `0x2c194–0x2c1c1` 分配/清零
整个 0x6a8 B 对象，`0x2c35c–0x2c397` 将对象内三个 `repoList` 元素链接为 freeList；
不是每张图各分配一个这种队列。三个待提交队列用于 COMMAND/PIXEL 源顺序；
硬件事件槽数组容量为 29，有效计数按命令模式取 28/29。这些数量都不是 SDK 节点
限制或先前的 1024 包装槽位。

```text
Event 对象内 repoList[3] → freeList
          │ AddListEx：eventListMutex，取空闲队列/追加事件记录
          ▼
    queueHead → 待提交队列 → queueTail
          │ Submit：读取 source；GetEvent 预留硬件事件 ID
          ├── 事件记录移到 queues[id]；命令携带该事件
          └── 待提交队列本体归还 freeList
                          │
    queues[29] ── 中断 / Recovery→Notify
                          └── 取出事件记录、释放硬件槽、处理信号/资源释放

进程 close → AttachProcessEx → Submit → 仍会访问共享 queueHead
故障现场 queueHead=0x100000000 → 读取 source(+8) → page fault
```

- **队列写入范围**：AddListEx 在 `0x2c9bf–0x2c9f2` 从 freeList 取队列并连接 tail；
  首队列写 head 在 `0x2cb9b`。Submit 在 `0x2cd7a–0x2cda0` 前移 head、回收旧队列，
  最后一个队列在 `0x2cf4f/0x2cf5a` 清 head/tail。以上路径使用 eventListMutex。
- **整体销毁**：Event_Destroy 先停止/销毁提交 timer，清理记录与待提交链，删除锁和
  atom 后释放整个 Event（`0x2c4a1–0x2c5c7`）；它也写 head/freeList。这条销毁路径
  不等于单张图 Release，未捕获它在故障前被错误调用的证据，不能排除对象寿命问题。
- **锁与完成回调**：Submit 可先进入 command commit；在持有 eventListMutex 时，
  GetEvent 再获取 eventQueueMutex。Notify 用 eventQueueMutex 取走硬件槽的记录并
  释放锁后处理信号/释放回调（`0x2d927–0x2d99b`）。这些锁存在不证明所有调用者
  正确串行，也没有捕获能确证竞态或死锁的运行证据。
- **恢复不等于重新构造**：Recovery 调用 Hardware_Reset、调整 pending/interrupt
  状态后调用 Notify（`0x20bda`）。所查 Recovery/Notify/FreeProcess 函数体未见直接
  重建 queueHead/queueTail/freeList；FreeProcess 处理硬件槽内按 PID 归属的记录。
  不由此声称整个间接调用树没有修改它们。输出正确或 recovery done 都不能证明
  待提交链恢复了有效指针。
- **确定的崩溃点**：Submit 获得 eventListMutex 后，从 `Event+0x5e0` 取 head，
  在 `0x2ceef` 读取 `head+8`；两次 Oops 的无效值相同。它不是 tensor 地址，
  也未证明 Event 对象本身已被释放。没有捕获 head/next/freeList 的第一次异常写入。
- **约 30.72 s 节律**：Monitor 比较事件/提交进度，停滞达 timeOut 后调用 Recovery，
  并用该间隔重排 timer；StartTimer 使用 `__msecs_to_jiffies`/`mod_delayed_work_on`。
  这支持看门狗反复恢复的解释；没有故障时内存快照/调度 trace，不能解释精确
  30.720 s 的全部来源，更不能由这个节律反推 Verify 的算法复杂度。

### 调查结论与下一次规则

已确认两个不同故障边界：首次卷积 hang 位于自写控制的 Verify 内；关机 close 在
共享驱动待提交队列头解引用时 page fault。两者在同 boot 关联，但尚未证明第一次
hang、恢复、应用退出中哪一步破坏了指针，也未排除独立驱动问题。无需为了排查再次
更换 bias 尾部、创建顺序或逐步复现吸收态故障；保持固定二级控制及至少 45 秒门禁，
任何内核异常后停测并保留标记，owner 清空和正确数值不放行，不复用故障 Context。

后续若向厂商核实，优先提供两次 Oops、原始时间线和匹配模块哈希，要求确认：
(1) queueHead 是否应始终为空或指向这三个 repoList 元素，谁能写 head/next/freeList；
(2) Hardware_Reset/Notify、Event_Destroy 与进程 close 的同步及待提交链恢复契约；
(3) Verify 内设备提交失败为何仍可返回成功；(4) 是否有针对同模块事件链损坏的已知修复。
这些是待确认项，本轮未向外发送消息，未声称根因已经解决。
