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
2. 如需冷启动，保留并归档原标记及证据。确认**主机与 NP101**恢复后，在新 boot 先跑
   未修改的原厂三节点卷积 demo（60 秒进程护栏、一次 Run）；需要进入模型验证时，
   再跑冻结旧 OPT component C16 prompt5/两个输出（180 秒、0 暖机/1 次）。不再使用
   已退役的自写卷积控制。两级均不开 SDK 插桩；期限不表示可接受性能。每级结束后至少
   再看 45 秒内核日志和 owner；历史 hang 曾在释放 28–35 秒后出现。控制或延迟门禁
   失败就停止，不进入模型测试。同 binary/SDK 的控制若在确认冷启动后再次失败，
   暂停相同条件的板卡重试，先用已有证据分析或提交厂商；不连续消耗冷启动周期。
3. 新事故立即保存独立记录：时间与 boot ID、二进制/SDK 哈希、输入与图结构、最后
   已返回/未返回的 SDK API、内核微秒时间、期望/实际输出、进程与 owner 状态、恢复
   标记、可证实原因与未证实假设、下次规避动作。先查下表的相同指纹，再决定候选。

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

**已实施的调整**：恢复门禁直接使用成功原始负载。改写控制的 CMake target/源码及
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

**规避**：退役自写卷积控制、不做故障复现；故障后即使 owner 清空也保持门禁，因为共享
DRM 驱动状态可在另一进程关闭时再次崩溃。后续检查必须识别 `BUG:`/`Oops:`、page
fault/general protection fault，而不只匹配 hang/EDMA。不卸载驱动、不改 DRM/图形
配置，不用关机重现验证假设。当前新 boot 的原厂/旧模型控制通过，不给它补旧 boot
恢复标记；两个独立事故及旧标记继续归档保留。
