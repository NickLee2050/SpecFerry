# NP101 SDK 与驱动问题

`v105` / `v106` 指匹配的驱动与 SDK 安装包 1.0.5 / 1.0.6，不是模块打印的版本字符串。
库位于 `/usr/lib/ljmicro`；记录包含 boot ID、命令、二进制/SDK SHA256 和内核窗口。
驱动 `galcore` srcversion 为 v105 `F53A3E6EFBB7D0FD7D4617C`、
v106 `8D46803809F15FA0FD98A92`。当前模型结果见 [状态文档](np101-status.md)。

## 复现规则

按 [README](../README.md) 配置并构建。每次一个限时进程、一个新输出目录；合成算子与内存检查不加载模型。Setup/Verify/Run 分别为建图/验图/执行，Verify 未返回的用例没有推理结果。
带 bias 的 FCL 和三维缓存追加是已知崩溃配置，单独运行并排在正常对照之后；崩溃、超时、hang、
kernel BUG 或残留占用立即停测、保留恢复标记，不自动重试。**下一次有意复现 selection 验图 hang 前，先提醒用户冷重启主机与 NP101**；热重启未消除故障。数值通过也须核查内核日志。

命令模板（下表给出 CASE / ARGS，每次自行选择一个，不能当作自动 sweep）。
原厂卷积恢复检查不开 SDK 插桩，使用单列的 conv 命令；sudo 只读参数仍需授权：

```text
python scripts/check_np101.py CASE ARGS --sdk-timing --timeout 60 --output NEW_OUTPUT
python scripts/check_np101.py conv --sudo-read-only --output NEW_CONTROL_OUTPUT
python scripts/check_np101_memory.py CASE ARGS --timeout 120 --output NEW_OUTPUT
```

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

全窗口 1×1 CONV 的 FFN 候选首图 Verify 在固定 OpenVX ELF `0x2db0c2`
（`vxoGraphOptimization_getKernelType+0x5d2`）SIGSEGV，与历史 FCL 同一指令。
分支为 FULLY_CONNECTED_RELU，空的是必需 `enable_relu` scalar 的 parameter[7]，
不是 bias。具体转换链及参数失效来源未知；该诊断入口已退役，不用于自动 sweep。
原始证据在 `.cache/runs/ffn-convolution-and-parallelism-20261004/ffn-linear-conv1x1/`。

两路 fc1 最终 logits 数值通过但同轮 EDMA 超时；改写卷积控制在确认冷启动后仍有
Verify/Run hang，并在两次关机的 `gckEVENT_Submit+0x1ff` 中读取无效地址
`0x100000008`。改写控制已退役，恢复改用原厂卷积与冻结旧 OPT。最近两项通过
不表示驱动修复。每项 boot、精确 SDK/内核时间、已证实原因与假设、证据及规避统一
维护在 [独立事故台账](np101-recovery-incidents.md)，不在本页重复时间线。

分块内存完整性使用 8 MiB 块，拒绝下一块后仍保留并完整扫描所有已写块，错误后继续扫描，再统一释放。
扫描不覆盖 SDK 隐藏空间或未申请的物理内存；申请拒绝、读回失败、未完成扫描分别记录。
此次第 109 块错误起始偏移为 1,340,216 / 1,340,280 / 1,340,344，各 8 B；历史 v106
分别为 1,274,680 / 1,274,744 / 1,274,808，整体差 65,536 B。累计逻辑位置约
873.278 / 873.216 MiB，**不是物理地址**，未定位原因。v106 1024 MiB 的 FP32 nonconst
两次全扫通过，但 2840 MiB 失败，不能据此认定该配置始终正确。
先测 64 MiB 控制；v105 不超过 1024 MiB。仅获授权的 v106 分块内存完整性合成探针可用
`integrity --mib 4096 --probe-limit --storage constant --dtype F16 --timeout 180`；
这不改变模型和普通检查的 const/nonconst 各 1 GiB 上限。

## Verify 成本与编译表缺项

预期：Verify 时间可解释，返回成功时有完整可执行结果。历史单层 612 SDK 节点合图：
v106 Verify 158.095/157.887 s，v105 140.917 s、增加诊断后 168.742 s，运行及数值通过。
两层 993 节点到 758.233 s 截止仍未 Verify 完；批量 16 头单层 416 节点为 49.846 s、
54 项通过；两层批量到 180 s 未完成。未进入 RunGraph 的测试不能判断数值。
两层四张分图累计 Verify 11.026 s、88 项通过；合并 KV 24 层 26 图累计 14.221 s、244 项通过。
缩小预留量没有显著收益；实际结构/边界非线性，O(n³) 仅经验猜想，不证明内部算法。

历史逐头 KV 追加及其图内 const/mutable 权重变体：1143 个 extension 的前 1024 个 operation 非空、后 119 个缺失，
执行表随后清空，跳过 Tiling/命令生成，Verify 仍成功，第 23/24 decoder 全零。
只读反汇编找到用户 shader 包装路径的 1024 槽，满后 NULL、上层忽略失败码；
这是共享包装资源，**不是每图最多 1024 个 SDK 节点**，也不是所有算子统一额度。
尚未直接捕获运行时满槽/首次 NULL；不能称为已证明越界写，也不解释全部 EDMA/分块内存完整性故障。
静态释放链未见单图归还，最后 Context 完整释放才清零；并发 Context 隔离未验证。
分析对应 OpenVX SHA256：`2f7223e504b15886c1562889668a80af48b19d605461beac175d336d396ff108`。

合并 KV 图的 extension 降为 423，完整表与全层数值/生命周期通过；两套图为 846。
完整模型验收保留 Context 重建出现 `846+178=1024`、缺 668 项，保护在运行前拒绝；新 Context
两次各 846 完整且输出一致。已采用新 Context 重建策略，普通请求复用模型。
详见 [状态与证据](np101-status.md)，编译表完整性尚未得到厂商解释，不与模型已测验收混淆。

历史整图、图规模探针、逐头 KV、图内权重与去缓存 reshape 入口已退役。用 `cleanup-20261001/before-source.tar.gz` 中的
嵌套旧记录和 `code-reduction-20260930/before-source.tar.gz` 恢复到独立目录才能复现，
当前默认合并 KV 图 不能冒充旧结构。当前检查可用两层粗输出：

```bash
python scripts/check_np101_model.py prefix-split --layer-count 2 --steps 2 --capacity 16 \
  --prepare-only --output .cache/runs/prefix-reference
python scripts/check_np101_model.py prefix-split --layer-count 2 --readback outputs \
  --fixture .cache/runs/prefix-reference/fixture --sdk-timing --timeout 180 \
  --output .cache/runs/prefix-merged-kv
```

## 原始证据索引

路径均相对本机 `.cache/runs/`，使用描述功能的别名，不随 Git 分发；
原始记录内容与目录保留不变，别名映射在本机 `evidence-directory-aliases.json`。
复现材料需附对应 `result.json`、`sdk.log`、execution 哈希/退出记录及内核窗口；
`sdk-calls.tsv` 是可选详细计时，`Waiting for HDMA Transfer...` 文本不提供等待时长。

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

卷积、FCL、追加、采样、FP16 保真、内存计数、重绑与编译表问题已有两版本结果；选择算子验图有同 binary SELECT 对照，v106 冷启动原 selection
未补测；三维缓存追加 v106 单节点追加未补测。分块内存完整性两版实际保留量不同，申请拒绝没有记为读回通过。
这些问题未因后续模型通过而自动关闭；分块 prompt 处理 EDMA/kernel BUG 与第 23 层编译缺项分别记录。
