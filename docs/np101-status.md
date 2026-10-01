# NP101 当前实现、验收与性能

截至 2026-10-01，目标是在旧 component 的正确性基础上优化 OPT-350m 图推理。
F2/F3 原定范围与 F4 已测范围验收完成，F5.P1 完成，下一步为 [F5.P2](../TODO.md)。
SDK/驱动未认定修复；具体问题使用 [P1–P10](np101-sdk-driver-validation-report.md)。

## 已验证契约

- 配置：v105、真实 OPT-350m 全 24 层、checkpoint 默认 greedy、K 合并 KV、
  每图一个 decoder、block1/block4、C16/C64。其他配置需对应验证，不声明通用支持。
- embedding、decoder、KV 更新、最终 normalization、LM head 与 greedy 均在图内；
  主机负责分词、调度和必要 INT32 token 读回。固定 SDK 的 F4 重放/提交/完成链
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
  物理地址未知，预算合规不保证 P5 内存数据正确。P5 特例不扩大模型预算。

## 数值与生命周期覆盖

| 工作 | 已完成范围 | 明确限制 |
|---|---|---|
| F2 | K/C16 两步全层 logits/选词/KV 244 项；连续生成 A/B/A、末槽、越界/close、释放 1084 项 | 两步 teacher 与生成分别验收 |
| F3 | K/block4/C16 完整 logits、全部 24 层 KV、prompt 16/5/16、尾部/decode、A/B/A、最终 block1 回归 | 每请求最多读 4 层 KV，六批全覆盖；1 次 EDMA 重试后完成 |
| F4 | 新 C16 长度 1/3/15；C64 的 15/16/17/49/63/64 边界、长短请求、完整最终 logits、末四层 KV | C64 未重复所有层 KV；一项已分析的选词差异获准接受 |
| F4 后端 | C64：52 图/936 调用；C16 新 Context 两实例：104 图/104 调用 | 固定库哈希、编译表完整、Replay→Submit→HAL COMMIT→完成 ioctl；不外推其他库 |

表内检查数包含文件、配置和生命周期审计，不是独立算子测试数。数值、正常退出及硬件
路径证明分别判定；SDK 成功或非空 operation 不能代替它们。

**F4 精度例外**：C64 的 16-token prompt 在第 15 个新词首次分歧，消费历史同为 30 token。
CPU FP16 增量选 164（` going`），NP101 block1/block4 均选 45（` not`），设备完整
logits 逐字节一致；两候选 logit 差为 CPU `+0.000976563`、设备 `-0.000976563`。
CPU FP32 整段/增量均选 45，差约 `-0.0004367`。同历史完整向量误差最大 0.03125，
通过既定 `atol=0.08, rtol=0.02`。按设备实际 31-token 历史补查，CPU/设备下一词均为
686，完整 logits 最大误差 0.0234375。用户接受仅此已量化的 DLM 近似并列例外；
原严格失败与判定文件保留，默认精度、生成策略及严格检查未变，不豁免未来任意分歧。

**驱动故障边界**：F3 首次全读回时全部 846 个 extension 完整、首块 logits 正确，
第 20 份 KV（零基 layer 9 的 V）读回触发 EDMA 及 `hdma_dev_free_chan_resources`
kernel BUG，仅保存 19/48 份 KV。这不同于历史第 23 层空执行表。用户报告热重启后，
少读回及分批完整验收通过；重启和读回策略同时变化，未证明驱动修复。F4 九进程
有 6 次 EDMA 重试、54 条 MCE，零 hang/kernel BUG；异常后必须停测并保留恢复标记。

## F5.P1 图推理性能

四个串行进程、新 Context，各输入 1 次暖机与 3 次测量，复用 F3/F4 CPU oracle；
不重读诊断 logits/KV，不启用 calls/strace/Verify 详细追踪。142 项审计通过，进程释放、
应用 live 归零，无 owner；零 EDMA/hang/kernel BUG/invalid opcode/soft lockup，24 条 MCE。

主指标为外层 `vsi_nn_RunGraph` 加独立 `vxProcessGraph` 的时间；内部 Process/Replay/Wait
不重复相加。包含 SDK 编排与完成等待，不是纯 NPU 算术耗时。初始化、Setup/Verify、reset、
显式上传/选词读回、CPU oracle、落盘和释放排除；旧路径运行期 `vxVerifyGraph` 独立剔除。
首词预测属于 prefill，decode 步数为 `consumed - prompt_length`，全部样本无嵌套计时。

| 实现 / 容量 | prompt / 输出 / decode 步 | Prefill 秒，中位数（范围） | Decode ms/token，中位数（范围） | Decode tokens/s |
|---|---|---:|---:|---:|
| 旧 component / C16 | 5 / 8 / 7 | 7.757（7.712–7.773） | 1650.24（1646.61–1660.86） | 0.606 |
| K/block1 / C16 | 5 / 8 / 7 | 7.746（7.710–7.749） | 1548.84（1534.19–1550.63） | 0.646 |
| K/block4 / C16 | 5 / 8 / 7 | 3.234（3.225–3.235） | 1523.43（1506.57–1528.70） | 0.656 |
| K/block1 / C16 | 15 / 2 / 1 | 23.093（23.075–23.249） | 1537.60（1535.80–1548.16） | 0.650 |
| K/block4 / C16 | 15 / 2 / 1 | 9.720（9.696–9.790） | 1505.93（1473.05–1559.39） | 0.664 |
| K/block4 / C64 | 49 / 16 / 15 | 22.226（22.043–22.319） | 1561.37（1561.20–1567.37） | 0.640 |

同输入 P15 block4 prefill 快 2.376 倍、时间减少 57.91%；P5 K/block4 decode 比旧路径
减少 7.68%。两种 block 的 decode 都运行单 token 图，1–2% 差异不能宣称结构性加速。
C16/C64 同时改变上下文和块尾比例，不能从该矩阵单独推断容量影响。每组仅三次，
无跨启动稳定性/p95 结论；旧历史单次墙钟吞吐不得混入本表。

图 decode 约 93% 在 decoder、7% 在 normalization/head/选词、0.3% 在 embedding。
旧路径 attention/FFN 占 69.50%、QKV 21.80%、KV copy 1.79%，只能用于旧边界分解。
P5 的旧路径额外 decode Verify 1.087 秒/7 步、显式 IO 0.318 秒；新 K/block4 为
0 / 0.0273 秒。新路径图外 CPU 余量约 0.003 秒/请求，优先调查 decoder 内部与权重放置。

准备时间单列：旧 component、K/block1 C16、K/block4 C16/C64 初始化为
13.408 / 20.910 / 39.235 / 40.048 秒；Verify 为 3.580 / 6.220 / 14.893 / 14.779 秒。
各配置仅观察一次，Setup/Verify 已包含初始化，不再相加。计时器 CPU 微基准约
0.27–0.45 µs/次，未做板卡开关对照，不能给出实机扰动百分比。
F4 旧 trace 的 galcore ioctl 约占图范围 92–94%，含驱动处理/调度/等待且有追踪扰动，
不混入本基准或解释为纯 WaitSignal/NPU 时间。公开图/节点计数器仅核实接口，尚未启用。

## 历史结论与证据入口

V1–V4 已完成，V5 的数值/生命周期归 F2/F4、性能归 F5；旧 Phase 1/2 归 F2，
扩展验收归 F4，并行 Verify 归 F5。当前只有 F5.P1–P4 参与排期。
缩小节点预留量收益不明显；实际图组合开销非线性，两层分图累计 Verify 11.026 秒。
O(n³) 仅是经验假设，未证明 SDK 算法、三层循环或可并发条件。
旧 component→A 的 AddNode 为 1572→4741（3.016 倍），96.18% 增量来自逐头 KV/控制。
A→K 为 4741→1861，内部节点 5485→1885，extension 1143→423；每层 KV 节点 126→6。
A 的 group2/W/Wc 均未修复末层失败；R 全深度截止前未推理，不能记为通过。
K/group2 与 WK 两步全层数值通过，完整生命周期组合仍需 F5；确定结构前应提醒用户
group2 对照不能替代 decoder 内部图边界调查。旧 component 与 Qwen3.5 回归继续保留。

下列目录均在本机 `.cache/runs/`，不随 Git 分发，包含输入/源码/二进制/SDK 哈希和原始输出：

| 范围 | 目录 / 关键文件 |
|---|---|
| F2/K、早期候选 | `k-optimization-20260929/`、`k-optimization-offline-20260929/` |
| F3 故障、续测验收 | `f3-block4-20260930/`、`f3-block4-offline-20260930/`、`f3-low-readback-20260930/`、`f3-acceptance-20260930/summary.json` |
| F4 完整范围与判定 | `f4-acceptance-20260930/summary.json`、`acceptance-decision-20261001.json`，各 case 的 `backend-audit.json` / `rebuild-audit.json` |
| 精度对照 | `f4-token-precision-20261001/comparison.json`、`post-divergence-check.json`、脚本与完整向量 |
| F5.P1 | `f5-p1-20261001/performance-summary.json`、`inherited-wait-attribution.json`、每 case 的 `audit.json` / `postflight.json` / `device/` |
| 旧 A/W/Wc/R 与槽表 | `w-wc-r-20260929/`、`w-wc-r-offline-20260929/shader-limit/`、`opt24-boundaries-resume-20260929/` |
| V 系列 | `v2-reservation-20260928/`、`v3-20260928/`、`v4-20260928/`、`v5-20260928/`、`split-prefix-{one,two,repeat}-20260929/` |
| 文档/临时入口恢复 | `cleanup-20261001/before-source.tar.gz`、`before.json`；包括旧 `docs/archive` 及未提交源码。更早快照在 `code-reduction-20260930/` |

本轮删去重复历史正文和 Git 内归档副本；原始证据未删除。恢复历史入口应在独立目录
解包快照，不能覆盖当前源码，也不会恢复当时 SDK 或板卡状态。复测命令见
[测试手册](../tests/README.md)，本轮变更统计见 [清理报告](cleanup-20261001.md)。
