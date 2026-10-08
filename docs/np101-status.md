# NP101 当前实现、验收与性能

截至 2026-10-08，目标是在旧 component 正确性的基础上降低 OPT-350m decode 延迟。
默认仍为 v105、merged-kv/group1；已验收 block1/block4、C16/C64 范围。
两路 fc1 是显式候选：短请求图时间有收益，最终 logits 通过，但同轮 EDMA 超时，
末两层 KV/block4 与设备验收未完成。后续步骤见 [TODO](../TODO.md)。

最近恢复记录的 boot 为 `83269f49-a97a-4713-a840-0a6224bd06e4`：未改动原厂卷积
及冻结旧 OPT component 短请求通过，退出后约 70/57 秒日志清洁、无 owner/恢复标记。
原厂最大误差 0.003234；旧 OPT C16 prompt5/两输出为 `[78,631]`，全 24 层、预算与
释放合规。旧 OPT 进程 80.126 秒、初始化 55.834 秒、请求 16.176 秒、释放 2.124 秒，
仅一次且无暖机，不能当图吞吐。binary/SDK 身份匹配，证据为
`.cache/runs/independent-controls-20261008-1502/summary.json`。

改写卷积控制已退役，恢复入口改用这两条原始负载。其 Verify hang 与关机 page fault
根因未定；三方建图差异、SDK/内核时间线及规避规则统一见
[事件台账](np101-recovery-incidents.md)，算子问题见
[SDK 问题报告](np101-sdk-driver-validation-report.md)。成功控制不证明驱动已修复。

## 已验证契约

- 配置：v105、真实 OPT-350m 全 24 层、checkpoint 默认 greedy、合并 KV、
  每图一个 decoder、block1/block4、C16/C64。其他配置需对应验证，不声明通用支持。
- embedding、decoder、KV 更新、最终 normalization、LM head 与 greedy 均在图内；
  主机负责分词、调度和必要 INT32 token 读回。固定 SDK 的完整模型验收重放/提交/完成链
  支持已测图走编译设备命令路径，属于图级证据，未逐指令跟踪或验证 DLM/TLM 联合运行。
- block4 处理同一请求的四个已知连续 token；尾部与 decode 使用 block1，不等待凑块。
  两套执行图共用一份权重和每层 K/V backing，整个模型生命周期常驻；请求间只 reset，
  不重新 Setup/Verify。多请求 batch、分次追加 prompt 的增量输入接口尚未实现。
- K/V 每层各用 `[1024,C]` backing，二维追加后 reshape/permute 供 16 头批量 attention。
  block4 固定写视图使用已校准的 `vxReshapeTensor` alias：指南的普通 RESHAPE 可以复制，
  不保证写回共享 backing。这是因所需写别名语义缺失而保留的 API 例外，仿真成功不够；
  真实头维、混合写入、覆盖、末块与 causal 不变性均已板卡校准，不交换运行期句柄。
- 模型重建同时销毁并重建 Context。同 Context 第二模型的非空 extension 为
  `846+178=1024`，后者缺 668 项；保护在运行前拒绝。新 Context 两模型各 846 项完整、
  输出 bit-exact、89 项通过。普通请求继续复用模型；并存 Context 的额度隔离未验证。
- const/nonconst 应用 payload 分别不超过 1 GiB，共享 backing 计一次，含临时分配峰值。
  C64 实测为 13,884 / 721,563,588 B；关闭后 live 归零。SDK 隐藏副本/workspace 与
  物理地址未知，预算合规不保证分块内存数据正确。合成容量探针特例不扩大模型预算。

## 数值与生命周期覆盖

| 工作 | 已完成范围 | 明确限制 |
|---|---|---|
| 固定图逐 token 推理 | merged-kv/C16 两步全层 logits/选词/KV 244 项；连续生成 A/B/A、末槽、越界/close、释放 1084 项 | 两步 teacher 与生成分别验收 |
| 分块 prompt 处理 | merged-kv/block4/C16 完整 logits、全部 24 层 KV、prompt 16/5/16、尾部/decode、A/B/A、最终 block1 回归 | 每请求最多读 4 层 KV，六批全覆盖；1 次 EDMA 重试后完成 |
| 完整模型验收 | 新 C16 长度 1/3/15；C64 的 15/16/17/49/63/64 边界、长短请求、完整最终 logits、末四层 KV | C64 未重复所有层 KV；一项已分析的选词差异获准接受 |
| 完整模型验收后端 | C64：52 图/936 调用；C16 新 Context 两实例：104 图/104 调用 | 固定库哈希、编译表完整、Replay→Submit→HAL COMMIT→完成 ioctl；不外推其他库 |

表内检查数包含文件、配置和生命周期审计，不是独立算子测试数。数值、正常退出及硬件
路径证明分别判定；SDK 成功或非空 operation 不能代替它们。

**完整模型验收精度例外**：C64 的 16-token prompt 在第 15 个新词首次分歧，消费历史同为 30 token。
CPU FP16 增量选 164（` going`），NP101 block1/block4 均选 45（` not`），设备完整
logits 逐字节一致；两候选 logit 差为 CPU `+0.000976563`、设备 `-0.000976563`。
CPU FP32 整段/增量均选 45，差约 `-0.0004367`。同历史完整向量误差最大 0.03125，
通过既定 `atol=0.08, rtol=0.02`。按设备实际 31-token 历史补查，CPU/设备下一词均为
686，完整 logits 最大误差 0.0234375。用户接受仅此已量化的 DLM 近似并列例外；
原严格失败与判定文件保留，默认精度、生成策略及严格检查未变，不豁免未来任意分歧。

## 图推理性能基线

四个串行进程、新 Context，各输入 1 次暖机与 3 次测量，复用分块 prompt 处理/完整模型验收 CPU oracle；
不重读诊断 logits/KV，不启用 calls/strace/Verify 详细追踪。142 项审计通过，进程释放、
应用 live 归零，无 owner；零 EDMA/hang/kernel BUG/invalid opcode/soft lockup，24 条 MCE。

主指标为外层 `vsi_nn_RunGraph` 加独立 `vxProcessGraph` 的时间；内部 Process/Replay/Wait
不重复相加。包含 SDK 编排与完成等待，不是纯 NPU 算术耗时。初始化、Setup/Verify、reset、
显式上传/选词读回、CPU oracle、落盘和释放排除；旧路径运行期 `vxVerifyGraph` 独立剔除。
首词预测属于 prefill，decode 步数为 `consumed - prompt_length`，全部样本无嵌套计时。

| 实现 / 容量 | prompt / 输出 / decode 步 | Prefill 秒，中位数（范围） | Decode ms/token，中位数（范围） | Decode tokens/s |
|---|---|---:|---:|---:|
| 旧 component / C16 | 5 / 8 / 7 | 7.757（7.712–7.773） | 1650.24（1646.61–1660.86） | 0.606 |
| merged-kv/block1 / C16 | 5 / 8 / 7 | 7.746（7.710–7.749） | 1548.84（1534.19–1550.63） | 0.646 |
| merged-kv/block4 / C16 | 5 / 8 / 7 | 3.234（3.225–3.235） | 1523.43（1506.57–1528.70） | 0.656 |
| merged-kv/block1 / C16 | 15 / 2 / 1 | 23.093（23.075–23.249） | 1537.60（1535.80–1548.16） | 0.650 |
| merged-kv/block4 / C16 | 15 / 2 / 1 | 9.720（9.696–9.790） | 1505.93（1473.05–1559.39） | 0.664 |
| merged-kv/block4 / C64 | 49 / 16 / 15 | 22.226（22.043–22.319） | 1561.37（1561.20–1567.37） | 0.640 |

同一 15-token 输入 block4 prefill 快 2.376 倍、时间减少 57.91%；5-token 输入的 merged-kv/block4 decode 比旧路径
减少 7.68%。两种 block 的 decode 都运行单 token 图，1–2% 差异不能宣称结构性加速。
C16/C64 同时改变上下文和块尾比例，不能从该矩阵单独推断容量影响。每组仅三次，
无跨启动稳定性/p95 结论；旧历史单次墙钟吞吐不得混入本表。

图 decode 约 93% 在 decoder、7% 在 normalization/head/选词、0.3% 在 embedding。
旧路径 attention/FFN 占 69.50%、QKV 21.80%、KV copy 1.79%，只能用于旧边界分解。
5-token 输入的旧路径额外 decode Verify 1.087 秒/7 步、显式 IO 0.318 秒；新 merged-kv/block4 为
0 / 0.0273 秒。新路径图外 CPU 余量约 0.003 秒/请求，优先调查 decoder 内部与权重放置。

准备时间单列：旧 component、merged-kv/block1 C16、merged-kv/block4 C16/C64 初始化为
13.408 / 20.910 / 39.235 / 40.048 秒；Verify 为 3.580 / 6.220 / 14.893 / 14.779 秒。
各配置仅观察一次，Setup/Verify 已包含初始化，不再相加。计时器 CPU 微基准约
0.27–0.45 µs/次，未做板卡开关对照，不能给出实机扰动百分比。
完整模型验收旧 trace 的 galcore ioctl 约占图范围 92–94%，含驱动处理/调度/等待且有追踪扰动，
不混入本基准或解释为纯 WaitSignal/NPU 时间。公开图/节点计数器仅核实接口，尚未启用。

## 性能候选与决策

### 结构初筛

全 24 层、block1/C16、prompt5/8输出，各新 Context、1 暖机+1 测量。
四组各 1885 个内部节点、编译表完整，token、预算与释放通过，无目标内核故障。
仅初筛，不是完整生命周期验收。所有组启用安静编译保护；其扰动未单独量化，
不与未启用保护的性能基线样本直接归因比较。

| 配置 | 图数/步 | Prefill 秒 | Decode ms/token | 初始化秒 |
|---|---:|---:|---:|---:|
| merged-kv/group1 | 26 | 7.567 | 1512.7 | 17.717 |
| merged-kv/group2 | 14 | 7.658 | 1529.4 | 29.557 |
| 图内 const | 26 | 7.818 | 1564.9 | 14.078 |
| attention/FFN 分图 | 50 | 8.084 | 1600.8 | 14.831 |

没有收益，保留 group1；attention/FFN 分图入口已退役。分图样本 FFN 为 decode
图时间的 63.7%、attention 侧 29.4%，不直接外推合图，也不证明算力或带宽瓶颈。
group2 不能代替 decoder 内部边界调查；图内 const 双图权重约 1.22 GB，不能绕过预算。

### FFN 集合与静态路径

24 层真实权重/输入、shared mutable bank，矩阵 payload 384 MiB；每组独立 Context，
4 暖机 sweep、3 批×8 sweep×24图，整批计时，无循环内显式 IO/逐调用日志/query。
只读末层输出，编译表、容差、预算、释放与设备日志通过。输入重放不含 attention/KV
间隔，图范围含 SDK 编排/隐藏搬运/等待，不能当整模型吞吐或纯 GEMM 时间。

| 集合对照 | 基线秒/24层轮 | 候选秒/24层轮 | 结论 |
|---|---:|---:|---|
| 完整 FFN / 线性主体 | 1.0106 | 0.9781 | 主体相当于 96.8%；差值不能逐算子归因 |
| 原布局 / 预转置 | 1.030850 | 1.047386 | 慢 1.60%，不接入模型；专用入口已退役 |
| 原线性 / 两路 fc1 | 1.030926 | 0.935383 | 快 9.268%，进入完整模型候选 |

两路 fc1 按连续输出通道切分权重，两次 MatMul 后 Concat，再用原 bias/ReLU/fc2；
内部节点 120→192、nonconst 峰值 412,018,688→412,215,296 B。末层与基线逐字节一致，
对 CPU 最大误差 0.00048828125；测量各 576 次 RunGraph，退出后超过 85 秒无故障。
当前集合测试复用模型线性 builder，保留完整/线性基线与两路 fc1 三种模式。
历史集合测量使用独立 helper，源码已归档；本次复用后未重跑板卡，历史数字只归于原样本。

安装 SDK 20 个路径去重为 15 个 ELF，只反汇编主机可执行节；MATRIXMUL 静态
优先分数为 EVIS 4、VX 3、CL 1、CPU 0，`fps` 不是实测帧率。当前 transB 表项为
`gemm_transb_F16F16toF16`，预转置为 `gemm_F16F16toF16`，都用 4×4 输出 tile；
注册/分派不能证明物理核利用率。公开 AsyncRun/Wait 在同 Context 使用单 worker，
未找到文档完整的 NN 核数分配接口。跨 Context 并发/公开计数器未验证，profiler 暂不使用。

全窗口 1×1 CONV 候选首图 Verify 在 FCL_RELU 优化器 SIGSEGV，没有推理/性能结果；
该入口已退役，保留 MatMul+Add。空的必需 `enable_relu` 参数和故障规避见 SDK 报告。
退役诊断源码及本次压缩前文档在 `.cache/runs/diagnostic-cleanup-20261008/`。

### 两路 fc1 完整模型

C16 prompt5/8输出、block1/group1，先基线、候选、反向基线，每组新 Context、
1 暖机+3 测量。外层图调用及完成等待计时，初始化/reset/显式 IO/释放排除；
三组 37/37 benchmark 检查通过，输出均为 `[78,631,38,5324,21,14,5,50118]`。

| 配置 | Decode ms/token，三次测量 | 中位数 | Prefill ms/输入 token，中位数 | 内部节点 |
|---|---|---:|---:|---:|
| 基线 | 1600.622 / 1590.990 / 1571.108 | 1590.990 | 1587.450 | 1885 |
| 两路 fc1，预装 fc2 | 1511.893 / 1508.881 / 1502.976 | 1508.881 | 1503.988 | 1957 |
| 反向基线，同候选二进制 | 1603.004 / 1608.923 / 1613.658 | 1608.923 | 1607.398 | 1885 |

候选 decode 中位数比前后基线少 5.16%/6.22%，逐次差异同向；447 个 extension
完整，26 图、无运行期 Verify。权重 payload 662,392,832 B，nonconst 峰值
678,277,756 B，比基线多 196,608 B；关闭后 live 归零，退出后 141/134 秒日志清洁。
这些是同 boot、同短请求样本，尚无跨启动稳定性或 p95 结论。

原候选在 decoder 23 的 `vsi_nn_AddTensor` 加载 8 MiB `fc2.weight` 失败，35.13 秒
代码 1 退出，无 hang/后置故障。库分支位于内部 CopyDataToTensor 失败后，不能区分
创建与复制失败。先装入共享 bank 的全部 fc2 后成功；改变顺序未改字节、图计算或
应用预算，不能证明底层原因或永久规避。生成期限本轮 360 秒，默认 600 秒。

候选最终 logits 验收复用 C16 prompt4/15/4、最多两输出，三条完整向量与既有设备
基线逐字节相同，对 CPU 最大误差 0.03125，两个 A 相同。21/21 数值/生命周期检查、
满槽/关闭拒绝、编译表、预算与释放通过；但同轮 EDMA 超时，设备验收未通过。
未覆盖每步 logits 或新候选全层 KV；末两层 KV 和 block4 仅离线准备，详情见事件台账。

### 跳过非末 prompt 块的输出头

block4 使用已存在的 LastToken 调度，只在最后 prompt chunk 执行 head，decode
与公开 step 仍每步执行。C16 prompt15/2输出，各新 Context、1 暖机+3 测量，
默认仅 INT32 token 读回、安静编译保护，计时口径与上述相同。

| 策略 | Prefill 秒，中位数 | 请求总图秒，中位数 | Decode ms/token，中位数 | head/请求 |
|---|---:|---:|---:|---:|
| every-token | 9.775 | 11.298 | 1522.82 | 7 |
| last-token | 9.179 | 10.770 | 1526.63 | 2 |

Prefill/总图时间减少 6.10%/4.68%，不宣称 decode 加速。冷启动后 prompt4/15/4
完整 logits/A/B/A、满块/尾块/reset、容量/close、预算与释放通过，每请求 head 两次；
52 图/4107 节点完整，nonconst 峰值 695,683,908 B，退出后约 65 秒日志清洁。
未重做已过全层 KV，后端链证据继承完整模型验收。该范围验收完成，默认仍 every-token。

## 历史结论与证据入口

- 旧 component→逐头 KV 的 AddNode 为 1572→4741，96.18% 增量来自逐头 KV/控制；
  合并 KV 后为 1861，内部节点 5485→1885、extension 1143→423，每层 KV 节点 126→6。
  逐头 KV 的合层/权重变体未修复末层缺项；槽表解释和新 Context 规则见 SDK 报告。
- 缩小节点预留量无明显收益，实际图组合非线性；两层分图累计 Verify 11.026 秒。
  O(n³) 只是经验假设，不证明内部算法或可并发条件。
- 当前合并 KV 最长已测 prompt64；旧 component 的 128/512/2033 输入仅有 SDK
  数值证据、没有完整硬件证明。2048 输入测量中断，不算通过或据此认定 hang。
- Decode CPU 归因旧尝试在初始化 150.19 秒期限中断，未进入推理；同类初始化曾需
  181.51 秒。计时已实现，留给下次必要的代表性负载，初始化超时不是性能样本。

下列路径相对本机 `.cache/runs/`，不随 Git 分发；故障证据索引归事件台账，
原始输入/源码/二进制/SDK 哈希保留。历史入口应独立解包，不能覆盖当前源码。

| 范围 | 目录 / 关键文件 |
|---|---|
| 合并 KV | `merged-kv-optimization-20260929/`、`merged-kv-preparation-20260929/` |
| block4 验收 | `block-prefill-acceptance-20260930/summary.json`、`block-prefill-reduced-readback-20260930/` |
| 模型范围/精度例外 | `model-acceptance-20260930/summary.json`、`acceptance-decision-20261001.json`、`token-logit-precision-20261001/` |
| 图推理基线 | `validated-opt-component-baseline-20261001/performance-summary.json`、`inherited-wait-attribution.json` |
| 结构初筛 | `decoder-structure-screening-20261001/summary.json`；退役分图 patch/source 同目录 |
| FFN/SDK 静态分析 | `ffn-workload-analysis-20261004/measurement-summary.json`、`library-scan/`；`decode-offline-20261005/` |
| FFN 布局/两路 fc1 | `ffn-layout-comparison-20261008/summary.json`、`split-fc1-performance-20261008/` |
| 候选待收尾 | `split-fc1-acceptance-20261008/`：logits 已测，`kv-block1/`、`logits-block4/` 仅准备 |
| 末块 head / 并行审计 | `head-skip-acceptance-20261008/`、`ffn-convolution-and-parallelism-20261004/parallel-api-audit.md` |
| CPU 归因旧尝试 | `decode-attribution-20261008/device/` |
| 旧长请求/图规模 | `acceptance-20260923-context-final/`、`graph-reservation-analysis-20260928/`、`attention-layout-analysis-20260928/`、`verify-boundary-analysis-20260928/` |
| 旧源码/文档恢复 | `cleanup-20261001/before-source.tar.gz`、`code-reduction-20260930/`；本次压缩前副本 `diagnostic-cleanup-20261008/` |

运行命令与保留诊断见 [测试手册](../tests/README.md)。
