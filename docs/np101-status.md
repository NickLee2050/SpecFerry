# NP101 推理实现与优化结论

更新：2026-10-16。活跃模型为真实 `facebook/opt-350m`；目标是保持旧实现的正确性，
降低单请求decode延迟。未完成项见[TODO](../TODO.md)，每次上板先读
[事故台账](np101-recovery-incidents.md)，接口见[测试手册](../tests/README.md)，
SDK失败契约见[问题表](np101-sdk-driver-validation-report.md)。

原始证据相对 `.cache/runs/`，不随Git分发；本次精简前全部源码/文档及HEAD身份在
`repository-compaction-20261016/before/`、`before-inventory.json`；退役文件列于
`retired-paths.json`。早期归档及原始硬件记录保持不变。

## EP阶段封版

本轮EP功能验收和性能路径探索收尾；后续优化按新需求/证据重开，不要求完成所有
可选TODO才结项。推荐配置固定如下，历史实现保留作回归及诊断，不自动重试故障条件：

| 模型 | 推荐运行配置 | 已验范围 / 当前性能口径 |
|---|---|---|
| OPT-350m当前NBG组合 | QKV/out、M1融合FFN/末LN及单head用NBG；可选M16 prefill，保留手写attention | C64；同轮17输入/16输出，M1→M16为0.65109→1.08032输出token/s，首词后decode为1.24558→1.21002 token/s |
| OPT历史最快组合 | FCL attention投影、融合NBG FFN/末LN和NBG head；本轮已停止直接FCL验证 | C64/C128历史范围保留；C64历史最高1.96582输出token/s、3.63403首词后decode token/s，不与当前时段混算 |
| Qwen3.5-0.8B缓存模型 | 保留GATHER_ND、layer9 MatMul FFN、18 DeltaNet＋23 FFN NBG、mapped读回；最快显式组合再用完整共享FCL head、广播外积、融合input RMSNorm、合并状态提交 | C64；23输入/16输出，组合两请求为0.287186输出token/s、0.610743首词后decode token/s；保守默认仍保留 |

模型、输入长度与请求数不同，上表不作跨模型速度对照；准备/reset/粗读回不计入生成。
对应包保持原始内容，不移动或重导出；各自manifest绑定NBG、源权重、端口及五库身份：

| 本机已验包（相对`.cache/runs/`） | manifest.json SHA256 |
|---|---|
| `opt-entry-integration-20261014/ffn-head-package` | `ec99e136f8458c301ed4165a582ade84751c468b596fcd3ef5431d0d96cf64fd` |
| `opt-fast-prefill-20261016/nbg-projections-prefill-package` | `910f4918102fc6569280e126aac3dc4d93d7f14320fb9fc409dade8e5438c450` |
| `opt-qwen-integration-20261014/qwen-compiled-blocks-package` | `9ccdf15e4c8e783b5e875af8ae03960a918563326a9529c2a57e2cd0d2e7f0be` |

这些包均绑定v105 `srcversion=F53A3E6EFBB7D0FD7D4617C`。原始checkpoint/NBG及固定
已验binary继续位于忽略缓存，源码仓库不分发这些产物；manifest/源权重不匹配应拒绝。
Qwen Python/native默认仍为原保守组合；共享FCL head及最新递归优化需显式选择。
OPT默认component回归路径不变，推荐混合路径由README命令显式选择。默认策略的收尾
中，含完整QKV/out产物的包现在默认选择NBG，旧head-only包保留原FCL路径；默认选择
已做CPU契约检查，硬件证据使用相同NBG配置的显式参数，不重复数学验证。

最快Qwen另用`qwen-deltanet-optimization-20261015/outer-normalized-package`，manifest SHA256
`4c9d1100efd57e751318f0e74a5dee53bd01fca6f678db058ec6b21abc6bbd35`。

当前证据覆盖有界生成、状态/隔离、预算/释放和延迟内核检查；长期稳定、Qwen C128、
内部硬件利用率及间歇性hang/EDMA根因未解决。恢复遵守板卡断电确认及完整门禁，
不再强制主机冷启动或boot ID改变。更长容量/全图prefill、SoC对照和TLM协议见TODO。
收尾前文档及身份留存`ep-baseline-finalization-20261015/before/`、`before-manifest.json`。

`M`表示GEMM一次处理的token行数：`Y[M,N] = X[M,K] × W[K,N]`。
M1用于逐token decode；M16可同时处理16个已知prompt token，不表示一次生成16个
未知后续token，也不表示16个用户。两种固定形状可用不同图；常驻两套图是否复制权重
取决于实现和NBG产物，不是M本身的要求。C64/C128另指上下文容量，不是M。

## 当前实现与验收范围

- 图内完成embedding、24层decoder、KV更新、最终归一化、LM head和greedy；主机
  负责分词、初始化、调度及token读回。Qwen3.5实现/checkpoint/回归证据保留。
- Python默认仍为component回归后端；手写graph默认`merged-kv`、每图一层decoder。
  两路fc1、block4/16及last-token head按显式配置使用。每请求复用图，无运行期Verify。
- 手写prefill/decode共享一份权重/KV；KV为FP16 `[1024,C]`。分块写采用已校准的
  `vxReshapeTensor`别名并保留写后读依赖；普通RESHAPE可能复制，不能替代共享写保证。
- 编译FFN候选每层attention→NBG FFN→残差/LN三图。M1及block16已验；M16包与
  M1包分别常驻，预算含额外FFN权重。固定48-copy图清零KV，首/末层粗验及重复请求
  通过；v105外层审计149图、3579 operation、750 extension。
- 新投影候选在M1编译FFN链中，将QKV/out_proj及project_out/词表投影改为直接FCL，
  bias保留独立FP16 Add，greedy使用INT32全局ArgMax；沿用三图边界，加固定KV清零图。
  v105/C64、17输入/16输出的1暖机+1正式请求通过，不切默认CLI、不外推M16或v106。
- 完整decoder候选每层一份NBG，普通rank3 FP32 KV，加独立copy图提交；
  正式C128入口新增固定KV清零图，97/3/97输入及96输出通过。未接block16或NBG原地别名。
- 全图NBG包含embedding、所有decoder、project_out、LM head及greedy，独立48-copy
  KV提交；C64共2图，正式C128另加固定reset共3图。仅M1已验，主机逐token调度；
  48个FFN投影TTGT无明显收益。
- NBG固定端口/图版本/权重来源；builder持有路径，图计入嵌入常量预算。const/nonconst
  各≤1GiB，共享backing只计一次；SDK副本、workspace和物理地址未知。
- 重建前释放所有旧图和最后一个Context；v105 Context共享进程1024 shader槽表，
  并存Context不隔离预算，平行Verify未验。AddNode仅逐项设置字段，不覆盖nn_param。
- 归一化及FP32 KV保留敏感精度契约。普通生成只读token，logits/KV按需粗读；
  普通非量化rank1/2/3 tensor可显式MappedHost读回，无SDK-copy回退。

| 验收对象 | 已完成范围 | 边界 |
|---|---|---|
| 合并 KV / C16 | 两步全层 logits/选词/KV；连续生成、A/B/A 隔离、未写槽、末槽、越界/close、释放 | teacher 与连续生成分别验收 |
| block4 / C16 | 全 24 层 KV、最终 logits、prompt16/5/16、尾部与 decode、block1 回归 | KV 分批读回；故障后恢复再完成，不把失败轮算通过 |
| 完整模型 / C64 | prompt15/16/17/49/63/64 边界、最终 logits、末四层 KV；Context 重建 | 未重做全部层 KV；精度例外见下文 |
| 两路 fc1 | C16 缓存/生命周期及 block4 logits；C64 prompt17/63/64 logits、64/3/64 末两层 KV/A/B/A | 不外推所有 C64 输入、全层 KV 或长期稳定性 |
| block16 / 两路 fc1 / C64 | last-token head，17/63/64 logits，64/3/64 末两层 KV、隔离/容量/释放；64-token prefill 对照 | 模型 block8 未测，FFN 集合通过不等于模型通过 |
| 固定 LN 参数预处理 | 同配置17输入/16输出前→候选→后及最终 logits；32输入/32输出重复生成 | 50,272 个最终 logits 与旧版本逐字节相同；未重做全部配置 |
| 有界常驻复用 | 同 boot、17输入/16输出，1暖机+10正式请求，160输出一致、无运行期编译、释放归零 | 约五分钟证据，不构成跨启动或长期验收 |
| 精简后 / C64 | 17输入/16输出，1暖机+1正式请求均与原CPU oracle一致；39项检查、编译完整性、预算/释放及延迟门禁通过 | 复用已验配置，无新增全层KV矩阵 |
| 精简后 / C128 | 96/97/96输入，各32输出，96个输出均与CPU一致；最终50,272维logits全量比较、A/B/A重复logits逐字节一致、末槽/越界拒绝/释放及延迟门禁通过 | 58项检查；仅这些输入与粗验范围，无全层KV或长期稳定性结论 |
| 编译FFN / C64 | 24层三图设备连接，block1、17输入/16输出，1暖机+1正式请求；选词/最终logits、v105的74图编译完整性、预算/释放及延迟门禁通过 | 单层两步KV另有粗验；block16见下文v106验收，未重验全层KV或长期稳定性；未切默认CLI |
| 完整decoder NBG / C64、C128 | C64原验收保留；正式C128入口97/3/97输入、各32输出，末层FP32 KV/未写槽、重复long逐字节一致、末槽/拒绝、编译审计和释放/45秒通过 | C128约935ms/token，使用旧分块MatMul head；不是已验NBG head的速度。未验block16、全层KV或长期稳定性 |
| 单个全图NBG / C64 | M1、17输入/16输出，1暖机+1正式请求；全部选词、最终logits、末层KV/未写槽、2图外层operation、预算/释放及45秒门禁通过 | 含GatherND及INT32选词修正；KV回写独立，不是整段生成循环编进NBG；未验block16、C128或长期稳定性 |
| 正式全图NBG / C128 | M1、97/3/97输入，各32输出、无暖机；96选词、末层FP32 KV/未写槽、重复long逐字节一致、末槽/零IO拒绝、预算/释放及45秒通过 | 31项粗验，3张外层图/97 operation完整；末层KV共6次mapped读回，未验全层KV、block16或长期稳定性 |

近并列精度例外仅有一例：C64、16-token prompt、第15个新词、同一30-token历史，
FP16参考token164（` going`）与设备45（` not`），top2间隔
约0.00098，整向量最大误差0.03125，容差`atol=0.08, rtol=0.02`通过；CPU FP32也选45，
沿设备历史下一词与CPU一致。此授权不豁免其他选词分歧。证据
`token-logit-precision-20261001/`、`acceptance-decision-20261001.json`。

C128的96/97输入、32输出及末槽已验，设备与CPU整段续写一致；答案有重复和事实错误，
说明计算一致性不等于回答质量。两轮96输入共64输出/113.310s，97输入32输出/57.921s，
未剔除暖机，不作持续吞吐基线。证据 `post-cleanup-context-20261010/`。

## 保留的优化与关闭的路线

| 修改 | 证据与决策 |
|---|---|
| 合并逐头KV | 逐头实现4741 AddNode→合并1861；每层KV节点126→6，extension1143→423。保留，减轻真实展开与共享槽表压力 |
| 固定图和共享存储 | 权重/KV复用、消除运行期Verify；应用共享不证明物理零复制 |
| block4/16与last-token head | 15-token图prefill约23.09→9.72s；64输入block4→16约22.7→8.233s。收益主要在首词，未知未来decode token不能直接合批 |
| 两路fc1 | 沿输出通道拆两次MatMul再Concat，保留完整归约/FP16边界；整模decode约快5%–6%。预装fc2曾绕过AddTensor失败，根因未定 |
| 固定LayerNorm affine | 96份gamma/beta初始化扩为共享FP32，4227→3843编译节点；Verify约13.2→10.7s，decode无收益。权重662,589,440 B |
| 编译FFN→完整decoder→全图 | 局部FFN集合约快11.7倍，但串行整模受其余阶段限制；最新生成性能见下表 |
| KV读回/清零 | 粗验和Map/Unmap减少大SDK-copy暴露面；48次清零上传改固定copy图，重复请求通过，不证明EDMA永久修复 |
| 更大/更小图、const MatMul、四路fc1、权重预转置、小工作集 | 未获采用收益或预算/槽位接近上限，专用分支/入口退役，原始证据保留 |
| 原始带bias FCL、1×1卷积、强制VX投影 | 失败/无有效匹配收益；原诊断分支退役。新的独立bias FCL契约另验，不能认定历史构造故障修复 |
| 两路fc2、INT8 | fc2拆分整模decode少5.28%–5.54%，与block16组合暂缓；INT8经FP16输出明显更慢，无纯I8→I32已验通路，整数探索已停止 |

单层612 SDK节点Verify约141–169s；两层993节点到758s未完成；两层四张小图累计
11.026s且正确。缩小预留量无明显收益。非线性已观察，O(n³)仅经验假设；AddNode、
编译operation、extension与shader槽位是不同计数，不能混用。证据见
`graph-reservation-analysis-20260928/`、`verify-boundary-analysis-20260928/`、
`merged-kv-optimization-20260929/`、`int8-and-ffn-partition-20261010/`。

## 性能口径与代表性结果

- 请求输出吞吐：`Σ实际输出数 / Σgeneration.total_seconds`，包含prefill/首词、decode、
  必要IO与调度；排除初始化、独立Setup/Verify、reset、CPU参考、落盘和释放。
  旧实现运行期重复Verify计入墙钟，不能扣除。
- 首词后decode速率：`Σ(输出数−1) / Σ(total_seconds−first_token_seconds)`；
  单输出无此指标。暖机剔除，不合并不同长度/配置，首词单列。
- 图/算子时间包含执行、内部搬运和完成等待；排除准备、显式IO、日志，不重加嵌套区间。
  输入处理率、算子回放、有效GFLOP/s均不是生成输出吞吐。
- 重复真实负载集合，前→候选→后；新追踪按关闭→开启→关闭量化扰动。

| 配置 | 输入→每次输出 | 请求输出 token/s | 首词后decode token/s | TTFT中位秒 |
|---|---:|---:|---:|---:|
| 旧component / C16 | 5→8 | 0.368 | 0.540 | 8.812 |
| 合并KV / block1 / C16 | 5→8 | 0.430 | 0.646 | 7.760 |
| 合并KV / block4 / C16 | 5→8 | 0.575 | 0.657 | 3.242 |
| 两路fc1 / block16 / C64，常驻复用10次 | 17→16 | 0.660356 | 0.720658 | 3.413 |
| 同配置加LN预处理 / C64 | 32→32 | 0.672882 | 0.714363 | 4.159 |

历史39组/103次正式请求已按上述口径重算。手写block16的64输入/1输出实际约
0.1215token/s，7.773是输入处理率。证据 `output-throughput-review-20261009/`、
`layernorm-affine-preparation-20261009/`。

以下均C64、17输入/16输出；v105前四行早先各1暖机+1正式，后四行最新匹配测试，
普通全图取前后两进程、block16普通取前后各3正式，TTGT为夹在其中的1/3正式。

| 实现 | v105 decode ms/token | v106 decode ms/token | 实际输出 token/s，v105 / v106 |
|---|---:|---:|---:|
| 两路fc1手写 / M1，历史 | 1407.787 | 1597.820 | 0.369 / 0.324 |
| 编译FFN三图 / M1，历史 | 666.124 | 1195.969 | 0.816 / 0.443 |
| 逐decoder NBG / M1，历史 | 621.903 | 839.248 | 0.875 / 0.640 |
| 普通全图 / M1，最新 | 542.949 | 602.512 | 0.921 / 0.834 |
| 全图FFN TTGT / M1，最新 | 542.225 | 603.427 | 0.922 / 0.839 |
| 混合编译 / block16，最新 | 650.103 | 1188.585 | 1.407 / 0.780 |
| 混合编译＋prefill TTGT / block16，最新 | 651.436 | 1167.116 | 1.425 / 0.802 |

最新v105普通/TTGT全图首词9.228/9.219s、生成17.373/17.352s；block16普通/TTGT
首词1.618/1.460s、生成11.369/11.231s。全图TTGT输出率仅增0.12%、decode少0.13%，
无明显收益；block16首词少9.76%、输出率增1.23%，M1decode未改且反增0.21%。
v105普通前后decode漂移全图−0.037%、block16+0.008%；v106分别−0.90%/−2.19%。

v106的9个模型进程/15正式/240输出、重装v105后的6进程/12正式/192输出均通过
选词/logits、所列KV/清零、预算/释放及各45秒延迟门禁。没有新hang或EDMA。
初始化v105全图0.743–0.775s、block16约12.5s；v106约10–12s/86–92s，独立记录。
回退速度恢复，支持与软件栈变化有关；驱动/SDK同时变化且boot不同，不能单独归因驱动。

证据：`ffn-compiled-integration-20261012/summary.json`、
`compiled-decoder-integration-20261012/summary.json`、`compiled-opt-fullgraph-20261013/`、
`compiled-opt-layout-v106-20261013/summary.json`、`compiled-opt-layout-v105-20261013/summary.json`。
首次v106单样本585.566ms/0.857token/s另存 `compiled-opt-driver-recheck-20261013/`，
不覆盖或与后续匹配平均混算。正式导出及规范入口见测试手册。

## OPT算子替换与正式入口

直接FCL不融合bias，FP16 Add单列；共享普通权重必须显式承诺图存活期只读。
以下为相同真实输入的重复集合局部结果，不是整模吞吐：

| 计算段 | MatMul ms | 直接FCL ms | NBG ms | FCL局部加速 |
|---|---:|---:|---:|---:|
| 每层三路QKV，含Q缩放 | 12.028 | 2.156 | 2.154 | 5.58倍 |
| 每层out_proj | 4.055 | 0.751 | 0.736 | 5.40倍 |
| head＋greedy | 103.556 | 40.502 | 40.806 | 2.56倍 |

| 计算段 | 手写前后均值 ms | NBG ms | 延迟变化 | 基线漂移 | 决策 |
|---|---:|---:|---:|---:|---|
| M1 attention核心 | 0.653 | 0.687 | +5.11% | +1.96% | 保留手写batched核心 |
| 每层两种残差/LN | 1.743 | 1.586 | −9.04% | +1.12% | 不单独增加图 |
| FFN＋手写末残差/LN → 融合NBG | 3.855 | 3.733 | −3.17% | +0.74% | 保留融合 |
| M16完整FFN | 56.033 | 10.893 | −80.56%（5.14倍） | −0.23% | 核实已有编译路径 |

保留手写batched attention核心、融合FFN/末残差/LN。直接FCL曾将整模decode
654.637降至284.357ms，但Verify由2.913增至343.751s；准备成本另计。
融合后单请求曾达3.708303 decode token/s，不与下表正式入口的三请求汇总混算。
对应逐轮证据入口保存在本次精简前全文及下述正式入口归档；不由局部结果外推M16。

| QKV / out_proj | 初始化 s | 实际输出 token/s | 首词后decode ms/token | TTFT s |
|---|---:|---:|---:|---:|
| FCL / FCL | 336.45 | 1.915 | 280.53 | 4.148 |
| NBG / FCL | 76.74 | 1.870 | 288.09 | 4.234 |
| FCL / NBG | 255.33 | 1.725 | 311.55 | 4.604 |
| NBG / NBG | 3.24 | 1.731 | 309.85 | 4.595 |

投影表固定融合FFN/NBG head，v105、17输入/16输出、每组合3请求；各自数值/KV、
重复、预算/释放及45秒通过。全NBG准备更短，decode约慢10.5%；图数52/76/100/124
同时变化，不能全归因提交。QKV节点内部V/K/Q与公开Q/K/V必须分别记录；错误映射
曾致数值失败，修正后通过，未构成设备故障。此表是历史记录；当前完整投影包默认NBG，
直接FCL已按用户要求停止本轮验证。

| C64正式入口，均仅末词head | 实际输出 token/s | 首词后decode token/s | decode ms/token | TTFT s | 初始化墙钟 s |
|---|---:|---:|---:|---:|---:|
| FCL投影/head＋融合FFN | 1.940457 | 3.608712 | 277.108 | 4.089 | 388.965 |
| FCL attention投影＋NBG head/融合FFN | **1.965820** | 3.634035 | 275.176 | 4.011 | 285.290 |

C64均17输入/16输出、各串行3请求、无独立暖机；选词/最终logits、释放/预算、45秒
通过。后一组48输出/24.417s为OPT最高请求输出率；初始化存在其他离线任务，不作纯
Verify归因。正式导出核对checkpoint、24层CPU/生成C/端口/版本，已验NBG逐字节一致。

最新混合C128：97输入/32输出、1暖机+1正式，32输出/34.779s=0.920104 token/s，
TTFT25.452s、decode3.323726 token/s。52图/1243 operation/238 extension完整；
末层FP16 KV、末槽、零IO拒绝、释放/45秒通过，峰值657,083,173/98,098,256B。
正式CLI同Context的97/3/97输入各32输出再通过粗KV与隔离；96输出/81.142s是混合
长度汇总，不替代固定负载吞吐。主机全线程user+system占C64 decode墙钟约2.36%–2.47%，
不说明板载CPU或NPU利用率。证据`opt-entry-integration-20261014/`。

### NBG投影链接M16 prefill（C64已验）

10-16接入已有M16产物，绑定checkpoint、M1包、端口和版本。M1使用QKV/out NBG、
融合FFN/末LN和一份NBG head；M16 attention用共享普通权重MatMul，前4层FFN为
两路fc1/普通fc2/FP32末LN，其余20层为MLP NBG及独立残差/FP32末LN。完整24份M16
NBG会超const预算，因此保留4层普通FFN，未改变is_const分类。大fc2先预载再共享。

```text
M16 embedding → 24层M16 → 取最后一列 → 固定copy ─┐
                                                  ↓
M1 embedding  → 24层M1   → 固定输入buffer → 唯一NBG head → greedy
                      两套执行图共享FP16 KV；先释放M16消费者，再释放M1 owner
```

固定桥接只在需要head时拷贝最后一列到M1固定输入；没有NBG原地alias、端口重绑或
第二份head。实际const/nonconst峰值1,035,583,855 / 410,013,944B，即987.61/391.02MiB，
分别低于1GiB且释放归零；const余36.39MiB，SDK内部副本/workspace仍未知。

| 同轮NBG组合，17输入/16输出、各3次 | 输出token/s | 首词s | 首词后decode token/s | 每请求s | 建图初始化s |
|---|---:|---:|---:|---:|---:|
| M1 | 0.65109 | 12.532 | 1.24558 | 24.574 | 15.916 |
| M1＋M16 prefill | 1.08032 | 2.414 | 1.21002 | 14.810 | 53.096–54.724 |

输出率提高65.92%，首词缩短80.74%，生成wall缩短39.73%；decode率反降2.86%，收益
来自prefill。准备/reset/CPU参考/粗读回另计；M16初始化增加，不能将此收益解释为
冷启动整体加速。本轮M1基线也慢于历史NBG记录，不据此宣布刷新历史最高吞吐，
整体慢速差异尚未归因。没有新增高扰动追踪或内部搬运实验。

原M16同Context的17/64/17输入、33个输出、末层KV/未写槽、重复状态逐字节一致、
末槽/零IO拒绝和释放全部通过；64输入的单输出仅验整块末词head及容量，不入性能表。
原生程序已完成model/Context.close，但外层监督被对话中断，原OS退出码没有保存。
没有补造退出码；随后M1三请求及M16单请求均正常退出0并通过各45秒门禁，补齐同配置
完整退出检查。M16性能汇总前后3个17→16请求，不重复原容量矩阵。
M1/M16审计为124/194图、1075/2918 outer operation、238/583 extension，均完整；
未新增并行Verify。最后11:48:53.351479 +08:00延迟检查清洁，无owner或恢复标记。
仅接受该C64有界负载，不外推新NBG组合C128、长期稳定性或TTGT组合。

直接FCL准备异常已停止验证，CPU权重压缩抓栈与中止记录见事故台账；当前所有板卡
实验子进程均限300秒。固定binary SHA256
`45b2683b1e5074c10b241e3a3007901e88e4da5e2e6fef5f28da579f995e9dda`；证据
`opt-fast-prefill-20261016/summary.json`，含包身份、原监督中断记录、逐项CPU/编译审计、
完整内核和补验结果。后续只改默认NBG选择和如实报告backend，已做CPU入口检查，
没有改动已测显式NBG模型计算。

### NBG速度差异的有界定位

10-16固定C64、M1、17输入/16输出、各三请求，QKV/out/FFN/head均为NBG。
历史与当前的M1 NBG、权重来源和已记录五库身份相同，124张常驻图及应用预算相同；
历史与当前实际输入token不同，不能只按长度相同认定为相同请求。

| 对照 | 实际输出token/s | 首词后decode token/s | 平均TTFT s |
|---|---:|---:|---:|
| 历史全NBG，原文本 | 1.73116 | 3.22732 | 4.595 |
| 今天冻结程序，关闭Verify审计与SDK计时 | 0.65933 | 1.27225 | 12.477 |
| 本轮修改前提交源码重新构建，同当前文本/NBG | 0.61912 | 1.19627 | 13.304 |
| 今天冻结程序，开启SDK内存汇总，仅作归因 | 0.57492 | 1.10615 | 14.270 |

三个新进程均各生成48 token，选词/末层KV/未写槽、预算/释放及45秒检查通过；
只在用户显式诊断授权下豁免16:48:18.926707这一次旧EDMA，未清恢复标记。
源码对照固定提交`eec8af1`、相同O2配置，显式选择NBG；不含本轮M16及对话改动，
不是早期全手写模型或历史快binary的重放。关闭审计后仍慢，降低了审计、M16和
本轮会话代码作为主要原因的可能性，尚未证明具体运行时或设备根因。

汇总轮45个首词后decode步骤中，`vsi_nn_RunGraph`为894.32ms/token，约占墙钟98.93%；
显式`CopyDataToTensor`/`ConvertTensorToData`合计8.13ms/token，约0.90%。
24层合计QKV153.25、attention160.07、out_proj152.82、attention收尾156.29、融合FFN
214.66ms/token；未见集中在第23层的耗时异常。图时间包含执行、内部搬运及同步等待，不能
把98.93%称为IO占比，也不能由此断言主机CPU回退。历史首请求主机user+system约
0.386s，今天关闭审计约4.288s，但两者文本不同，CPU增量不自动解释全部延迟。

后置关闭计时的同binary/C64/M1复测发生**新的4字节EDMA**，已终止并停测，详见
事故台账。关闭→开启→关闭未完成，不能精确量化本轮采集扰动；分段比例仅适用于
开启轮。历史原始token的复测已离线准备但未上板，因此输入内容和更早实现差异尚未
完全排除。没有改变生产路径、驱动、时钟或内存分类，未宣布修复性能回退。
证据`opt-decode-regression-20261016/{summary,artifact-comparison,sdk-attribution}.json`，
含冻结源码/编译配置、实际请求、SDK身份、逐层集合、完整内核及恢复标记链。

### C256三轮对话：保留KV与完整历史重放

10-16新增有界对话入口：完整前缀必须匹配上一轮输入和全部输出，末预测尚未进入KV，
下轮只补消费一次；已缓存历史不重复prefill。M16按绝对位置对齐，前后不足整块时用M1。
OPT使用原checkpoint greedy策略；三轮新增36/26/27 token，累计prompt36/78/121，
各输出16，最终消费136槽。英文材料/追问/总结的原始token轨迹由CPU独立生成，两侧共
96输出与CPU一致，末层KV/未写槽及两侧全缓存比较通过；没有加入额外BOS或重编码历史。

| 同binary/NBG/SDK，独立常驻进程、各三轮 | 平均TTFT s | 实际输出token/s | 48输出的响应总s | 首词后decode token/s |
|---|---:|---:|---:|---:|
| 保留KV | 13.64094 | 0.602979 | 79.60470 | 1.163335 |
| 清空KV后完整重放 | 13.74786 | 0.606478 | 79.14548 | 1.187276 |

响应口径为逐轮`state_preparation_seconds + generation`，包含reset及必要token IO，
不含初始化、tokenization、CPU参考、粗诊断、文件输出或用户思考时间。初始化分别
56.541/51.223s；SDK子进程149.790/143.408s，各低于300秒。旧表排除reset，不混算。
平均TTFT仅少0.78%，输出率反降0.58%；这是一次三轮对照，未观察到明确的整体收益。

| 轮次 | 新输入 / 累计prompt | 保留KV TTFT s | 重放TTFT s | 保留KV的M16块 / M1 prompt步 | 重放M16块 / M1 prompt步 |
|---|---|---:|---:|---|---|
| 1 | 36 / 36 | 6.62349 | 6.28712 | 2 / 4 | 2 / 4 |
| 2 | 26 / 78 | 23.13141 | 16.93983 | 0 / 27 | 4 / 14 |
| 3 | 27 / 121 | 11.16791 | 18.01664 | 1 / 12 | 7 / 9 |

第二轮从已消费位置51追加末预测及26新输入：13个M1补到64后只剩14个，不能构成
M16整块，所以27个全用M1；重放能用4个M16块。这是短追加损失的重要线索，不能仅
凭处理token数减少推断延迟降低。后续可另评估有限历史重算等对齐策略，当前未实现。

两侧const/nonconst峰值均1,035,620,719 / 535,651,064B，释放归零；194图、2918 outer
operation、583 extension完整。原NBG与native包字节不变，派生manifest仅声明C256；
未扩展普通生成CLI默认范围，也未验C256末槽/溢出、其他对话、长期稳定或更长容量。

**故障范围**：最初单进程串接六轮在重放第二轮发生4字节EDMA，结果不入上述表。
用户明确授权忽略这一已记录事件后，拆为两个三轮进程，执行/数值/预算/释放及各
45秒均通过；16:58:25.697036、17:02:40.443150完整日志无新故障/owner。没有断电，
只在本次诊断中豁免16:48:18.926707这一旧事件，恢复标记与事故证据保留；不能据此
宣称EDMA已修复。证据`opt-conversation-20261016/comparison.json`及事故台账。

### 全图／逐decoder C128

首版全图C128仅在模拟器Verify后重存，导入报`VX_ERROR_NO_RESOURCES`，未进入
Setup/Verify/Run；正常释放且45秒无故障。按生成程序初始化并运行一次模拟器再保存，
新636,700,160B NBG最小导入及整模通过；输入初始化等也变化，不能将Run认定唯一根因。
新NBG公开IO成为kernel顺序，必须修正排列；INT32 ArgMax修正与旧C64契约保留。

| 实现 | 初始化 s | 97输入的两次请求 wall s | 长请求实际输出 token/s | 3输入的请求 wall s / 输出 token/s | 首词后decode ms/token |
|---|---:|---|---:|---|---:|
| 逐decoder NBG＋旧MatMul head | 3.654 | 119.767 / 119.711 | 0.2672 | 31.842 / 1.0050 | 935.68 |
| 新全图NBG，含head/选词 | 1.181 | 110.355 / 110.386 | 0.2899 | 29.326 / 1.0912 | 862.09 |

v105同boot、97/3/97输入各32输出，93个首词后间隔；末层FP32 KV/未写槽、
重复long逐字节一致、容量拒绝、审计/释放/45秒通过，未做全层KV或block16。
证据`opt-qwen-integration-20261014/corrected-fullgraph128-{import,acceptance}/`。

## Qwen完整head与递归状态优化

完整共享FCL head复用embedding的61个普通只读块：60×4096＋2560=248,320行。
不新增权重、不改const；SDK内部副本/打包未知。动态权重NBG只生成MatMul，故采用
公开FCL接口；真实4096/2560行模拟器与整模通过，原默认MatMul保留。

| 完整模型组合 | 平均生成wall s | 实际输出token/s | 首词后decode ms/token | 首词后decode token/s | TTFT s |
|---|---:|---:|---:|---:|---:|
| 原共享MatMul head，前后两轮 | 81.656 | 0.195944 | 2927.05 | 0.341641 | 37.750 |
| 完整共享FCL head | 64.212 | 0.249173 | 1863.20 | 0.536712 | 36.265 |

同binary/SDK、C64、23输入/16输出、前→候选→后各一请求；CPU选词/粗状态、
未写槽、预算/释放和每轮45秒均通过。完整共享FCL生成耗时少21.36%，decode间隔
少36.35%；峰值与原实现相同887,143,543/669,684,140B。仅覆盖该有界请求。
前65,536行NBG实验仅约5.8%输出收益并增加134,219,128B const，已被全head替代，
其源码入口退役，原始端口修复与性能证据仍在`qwen-head-prefix-20261015/`。
完整head证据`qwen-shared-head-20261015/`。

共享FCL head、尚未应用递归优化时的旧归因保存在
`qwen-shared-head-attribution-20261015/`；最新组合归因见下节，不混用二者的占比。

### DeltaNet外积、归一化与状态提交优化

三个独立候选及其组合均已实施并通过C64/v105整模验收：

- 导出时用`--outer-product multiply`将收缩长度为1的FP32外积改成广播Mul。
  生成C中的MatMul由3个减为2个，5个FCL投影保留；没有缩小状态或改变精度契约。
- `--fuse-input-norm`把18层输入RMSNorm放入各自NBG，保留FP32归约、权重加1和
  投影前FP16边界。版本2原生manifest绑定额外norm权重及原始输入语义；版本1沿用独立norm。
- `--state-commit combined`使FFN固定绑定NBG激活输出，在24层完成后用一张固定图
  提交18组递归/卷积状态。新旧状态仍分离，下一token前完成提交；pending时拒绝再次
  step/reset/读状态，执行或提交失败使整模失效。消费者先于状态owner释放。

按执行链，每次consume少18次norm和17次commit RunGraph，38次consume少1,330次外层调用。
提交仍写约18.84MiB状态/token，仅省去约36KiB激活中转；不是以缩小状态容量获得收益。
初始化外层图数165→130；同进程两请求的Verify审计615→580、outer operation
2,322→2,106，extension均248且完整。未新增并行Verify或改变普通权重分类。

真实layer0的8步ONNX对照中，外积、融合norm及两者组合的72项输出/状态比较逐字节一致；
全部18层分别隔离编译、绑定权重/端口并审计FP32状态和norm。板卡沿用同一固定请求、
CPU参考、完整共享FCL head、GATHER_ND、layer9 MatMul FFN和mapped选词。
SDK calls关闭、Verify审计开启；每请求23输入/16输出，生成包含现有推理内KV重验，
初始Setup/Verify、reset和粗读回另计。下面wall为每请求均值，速率按总输出/总生成时间计算：

| 组合 | 请求数 | 生成wall s | 实际输出token/s | 首词后ms/token | 首词后token/s |
|---|---:|---:|---:|---:|---:|
| 原配置前置 | 1 | 65.884 | 0.242852 | 1886.38 | 0.530115 |
| 仅广播外积 | 1 | 64.655 | 0.247468 | 1858.15 | 0.538169 |
| 仅融合input RMSNorm | 1 | 63.100 | 0.253567 | 1842.67 | 0.542690 |
| 仅合并提交及激活直绑 | 1 | 61.501 | 0.260160 | 1791.02 | 0.558340 |
| 三项组合 | 2 | 55.713 | 0.287186 | 1637.35 | 0.610743 |
| 原配置后置 | 2 | 63.097 | 0.253577 | 1825.75 | 0.547720 |

相对后置原配置，组合的生成延迟降低11.70%、实际输出率提高13.25%、decode速率提高
11.51%（间隔缩短10.32%）。前置65.884s到后置两轮63.475/62.719s有可见漂移，
前两项独立变化没有超出该范围，不能认定稳定单项收益或将收益相加。组合两轮分别
55.901/55.525s；未遍历所有组合或分离编译优化交互。

6进程/8请求/128输出ID均与CPU一致，粗末层hidden、layer22递归状态、末层KV和未写槽
通过原容差；设备改写后的数组不要求与原配置按位相同。组合两请求及后置原配置两请求
各自的粗状态reset逐字节一致。组合const/nonconst峰值887,221,483 / 669,094,100B，
相对原配置const仅增77,940B、nonconst减少590,040B，各段≤1GiB且释放归零。
组合初始化130.703s，后置原配置124.592s，另计且不宣称初始化收益。
每进程完整内核/owner及45秒延迟门禁通过，无新设备故障或恢复标记。
仅覆盖上述C64请求，不外推C128或长期稳定性；保留原默认和各独立开关。

已验组合包`qwen-deltanet-optimization-20261015/outer-normalized-package`，manifest SHA256
`4c9d1100efd57e751318f0e74a5dee53bd01fca6f678db058ec6b21abc6bbd35`；配合
`--head-backend shared-fcl --state-commit combined`使用。测量binary SHA256
`c67e859d1e4650a56ac7659738582fdcd2df07facb6c772e25978ffbc9d7e332`；最终构建仅补显式
实现依赖，移除debug/build-id后与测量文件逐字节一致，证明在`final-build-identity.json`。
同目录`summary.json`、`onnx-combined-comparison.json`、逐轮原始日志/审计/数值结果保留身份
及计时证据；新增FP32归约拒绝回归，14项相关CPU测试、构建与格式检查通过。

### 最新Qwen组合的decode归因

10-16以精简后同一binary、同一包和23输入/16输出请求做SDK calls关闭→开启→关闭，
三轮48个输出与固定CPU参考一致，粗hidden/递归状态/KV、未写槽、预算/释放、编译
完整性及各45秒内核检查通过。完整共享FCL head、广播外积、融合input RMSNorm、
合并提交、layer9 MatMul回退和mapped选词全部开启，没有新增模型优化。

| 段 | ms/首词后decode间隔 | decode墙钟占比 |
|---|---:|---:|
| 18层DeltaNet NBG，包含融合input RMSNorm | 651.58 | 34.98% |
| 23层编译FFN | 318.93 | 17.12% |
| 完整共享FCL head＋greedy | 299.17 | 16.06% |
| 6层attention投影/RoPE/reader/out_proj | 267.24 | 14.35% |
| attention控制及显式IO | 75.28 | 4.04% |
| KV重新Verify / KV copy | 62.93 / 27.09 | 3.38% / 1.45% |
| layer9 MatMul FFN | 61.54 | 3.30% |
| 其余6层独立input RMSNorm | 48.92 | 2.63% |
| 单张合并状态提交图 | 13.15 | 0.71% |

开启记录的15个decode间隔平均1862.51ms；固定调用序列逐项匹配全部2388次RunGraph，
38次consume确实比旧结构少1330次调用。DeltaNet仍是最大单段，合并提交已不占主要
时间；FFN合计380.47ms、20.43%。融合后的DeltaNet包含输入norm，不能将651.58ms
直接与旧表不含norm的613.47ms相比。SDK区间包含计算、内部搬运与完成等待，
不是硬件kernel时间或利用率证据。

关闭前/开启/关闭后的generation为58.867/63.833/62.548s；两轮关闭合计32输出/
121.416s，实际输出率0.263557 token/s，首词后decode率0.562295 token/s，平均
TTFT34.031s。记录相对两轮关闭均值多5.15% generation wall、4.73% decode wall，
但前后关闭本身漂移6.25%/6.89%，因此不能精确分离探针成本，也不能用开启记录的
速率替代未记录吞吐。此轮低于10-15历史最好成绩，不覆盖或更新其最佳记录。
准备、reset、CPU参考和粗读回均另计。证据
`qwen-optimized-attribution-20261016/{summary,decode-attribution}.json`，包含固定binary/
包/参考身份、原始逐API与完整内核日志；仅覆盖本次C64有界负载。

## GEMM、布局与设备有效算力

MAC=2 FLOPs，只计GEMM/卷积算术量，分母为真实重复图集合。手写权重为运行时tensor，
NBG将其嵌入并允许打包，比较不只是编译器变量。每集合8份不同矩阵、2暖机、3重复批次，
2048方阵为2图各4组；表中延迟为集合均摊，每格 **ms / GFLOP/s**。
TTGT为 `C=A·B → Cᵀ=Bᵀ·Aᵀ → C`，显式输入/权重/输出转置均计时，主机布局准备除外。

| M / K / N | 手写 GEMM ms / GFLOP/s | 手写完整 TTGT ms / GFLOP/s | NBG 固定权重 GEMM ms / GFLOP/s | NBG 固定权重 TTGT ms / GFLOP/s |
|---|---:|---:|---:|---:|
| 1 / 1024 / 4096 | 20.488 / 0.409 | 282.677 / 0.030 | 2.771 / 3.027 | 2.784 / 3.013 |
| 16 / 1024 / 4096 | 29.689 / 4.521 | 285.551 / 0.470 | 8.741 / 15.355 | 1.911 / 70.233 |
| 16 / 4096 / 1024 | 28.679 / 4.680 | 272.540 / 0.492 | 2.052 / 65.416 | 1.967 / 68.242 |
| 64 / 1024 / 4096 | 126.035 / 4.260 | 348.383 / 1.541 | 33.258 / 16.143 | 4.913 / 109.276 |
| 256 / 1024 / 4096 | 510.485 / 4.207 | 618.060 / 3.475 | 161.880 / 13.266 | 58.293 / 36.839 |
| 1024 / 1024 / 1024 | 514.145 / 4.177 | 494.418 / 4.343 | 156.208 / 13.748 | 135.888 / 15.803 |
| 1024 / 1024 / 4096 | 2045.272 / 4.200 | 1796.324 / 4.782 | 780.590 / 11.004 | 363.978 / 23.600 |
| 2048 / 2048 / 2048 | 4101.422 / 4.189 | 3235.913 / 5.309 | 1633.039 / 10.520 | 605.707 / 28.363 |

固定FP16权重的扩展集合：前两行8乘积，后三行2图各1乘积，非仅尺寸变量；最大
源payload界384MiB，显式IO峰值160MiB，SDK隐藏workspace未知。

| M / K / N | NBG普通布局 | NBG完整TTGT |
|---|---:|---:|
| 64 / 1024 / 8192 | 66.537 / 16.137 | 9.813 / 109.418 |
| 64 / 8192 / 1024 | 10.879 / 98.698 | 12.762 / 84.136 |
| 4096 / 1024 / 4096 | 3144.465 / 10.927 | 1494.615 / 22.989 |
| 4096 / 4096 / 4096 | 997.590 / 137.771 | 2857.784 / 48.093 |
| 8192 / 1024 / 4096 | 数值失败，不计有效算力 | 2960.221 / 23.214 |

最高实测137.771GFLOP/s；N增大后仍约109，M增大不单调。TTGT不是通用最优。
M8192普通前4096行正确、后半错误（最大1.9194336），同形TTGT通过；正常释放/延迟
日志清洁，此数值失败不记为有效性能或冷启动事故。双运行时输入M16 NBG数值失败，
最大误差4.612/3.361，输入前后逐位一致，不能外推为所有动态MatMul不支持。

8图GEMM→bias→ReLU→fc2→bias→残差→LN消融表明低有效算力已在纯GEMM出现；
LN最大增量约2.26ms/3.85%。真实8层M1 FFN：手写前后34.426/34.972、常量MatMul35.142、
NBG约2.964ms/FFN，不能据FCL融合名判定物理引擎。M16 FFN普通/TTGT约10.912/4.067ms，
局部2.683倍收益不能直接外推串行整模。有效算力不是物理利用率。
证据 `gemm-ffn-ablation-20261010/`、`gemm-layout-collection-20261012/`、
`compiled-gemm-layout-20261012/`、`compiled-opt-exploration-20261012/`、
`compiled-prefill-ffn-layout-20261013/`。

## 官方转换与NBG契约

联合阅读 `~/NP101_Official`、`~/NP101_API` 的6份PDF（39页）及56份Markdown，
英文改名映射、阅读清单/hash与原始评估在 `official-knowledge-consolidation-20261012/`。
不把第三方镜像、源码、许可证或逐页记录复制进Git。

```text
ONNX → 导入/精度配置/离线编译
     → network_binary.nb + metadata + 生成C/C++
     → 主机ELF/动态库加载NBG → Setup → Verify → 重复RunGraph → 释放
```

`.nb`不是ELF。Setup/Verify/Run链直接来自YOLO官方源码，不是仅由反汇编猜测：
`tools/compile/ModelTest/apps/yolov5s_uint8_nbg_unify/`中`vnn_yolov5suint8.cpp:231–233,352`、
`main.cpp:91,143,235–255`。这些调用不证明运行时重做全部离线编译。

- 工具Pegasus 6.6.1，目标`VIP9000LD_PLUS_PID0X1000001D`，ONNX opset11/IR7。
  非特权bubblewrap隔离编译、不暴露galcore。tensor meta无图像预处理；FP16配置映射
  `--dtype float`，输出强制FP32关闭，实际精度/端口必须查产物。
- NBG/metadata/生成源码须成套且匹配SDK；常量Gemm可能变fullconnect，算子名不保证
  物理引擎或绕过历史FCL失败。完整动态shape、自动KV和跨NBG共享权重语义未确认。
- OPT单层导出初始FP16 KV的次正规值被文本转换及算术清零；改普通rank3 FP32 KV。
  reshape输入绑定与混图NBG曾Verify正常返回−1，独立NBG/普通KV通过，无设备事故；
  没有依据时不重试失败构造。独立copy提交，不换运行期tensor句柄。
- 全图原Gather要求runtime1.1.39，本机头文件/模拟器1.1.37；等价GatherND导出1.1.35。
  原ONNX1450节点/52输入/50输出，logits/48状态逐位等价。
- 生成C的ArgMax中间INT16使50118饱和到32767，外部INT32不足以修复。源IR指定int32
  仍被忽略；仅把该virtual tensor改INT32再用同目标重建，版本保护不改。
  手工重建需`VIVANTE_SDK_DIR`找到shader头、`VSIMULATOR_CONFIG`为完整.config路径。
- 全图TTGT只换24层FFN的48个投影；完整ONNX50输出逐位等价。导入IR fullconnect
  147→99、matmul48→96，不能当物理核分配。102板端端口匹配，NBG636,229,888 B。
  两布局const/nonconst峰值662,593,528/25,528,908 B，释放归零。
- 同时常驻两套整模权重超过1GiB；全图M1加24层M16 FFN约1065MiB也超限。
  不假定NBG能共享权重、不借改is_const绕过；完整prefill方案见TODO。

板卡手册声称12 NN单元/48 CPU单元、4GB SeDRAM、2TB/s、INT8 20TOPS、FP16/BF16
10TFLOPS、FP32 512GFLOPS、典型10W；均为厂商规格，不是本机实测。匹配文件
`np101_chip_overview_rev0.6.pdf`、`np101-pe-22_board_manual_rev0.6.pdf`。
官方YOLOv5s UINT8为69.525ms、236.368GOP/s（64正式），ResNet34目录模型为
7.405ms、989.599GOP/s（192正式）；按实际ONNX Conv/Gemm计MAC=2 ops，其余工作
只入分母。前者图片检测无独立准确率oracle，后者随机输入仅验logits稳定；无分类精度结论。

## 归因、并行与profiler

v105静态对照已核实SDK hash，并检查生成C、导入IR、实际NBG字节和SDK反汇编：

| 对照 | 已确认差异 |
|---|---|
| 公共执行入口 | 两种图均走`vsi_nn_RunGraph → vxProcessGraph → vxoGraph_Process`；NBG含`gcoVX_Replay`路径，普通图也有replay，不能把缓存回放视为NBG独有 |
| M1 FFN | 手写选中`gemm_transb_F16F16toF16`；对应SDK内嵌EVIS程序是4×4输出tile、K每轮8元素、FP32累加。导出IR将Gemm转FCL；单层NBG的fc1/fc2分别8/4个TP命令，bias仍为SH Add，ReLU仍单独存在，无NN命令 |
| 常量及提交数对照 | 8层FFN各版本均每集合一次RunGraph：共享/const MatMul约34.699/35.142ms，NBG2.964ms。const标记未消除差距；整模手写26图→编译FFN74图仍提速，减少主机图提交不能解释全部收益 |
| M16 FFN | 8层普通布局NBG为16 NN、32 TP、16 SH命令；投影被改为`ConvolutionReluPoolingLayer2`的NN路径，与M1的TP FCL不同 |
| 全图M1 | 3382条序列项含1664 SH、900 TP、817 NN及1 START；attention的48个MatMul展开为768个逐头NN命令并配768 SH，FFN投影288 TP，LM head100 TP。内部命令数不等于SDK节点数、shader槽位或物理核数量 |

NN/TP/SH类型由固定版本NBG reader/serializer及操作类型表交叉推断，属于编译选择的
静态证据，尚无逐命令硬件耗时/利用率。已确认算法展开和执行路径不同；权重打包、
内部搬运、各路径的收益份额仍未知，不把导出器的`Packing data`日志当物理权重布局证明。
例如fc1首条记录类型为3；SaveTPNNOperation的3分支取TP command size，2分支取NN
command size，并原样写类型，PrintOperationInfo的类型表也对应3=TP、2=NN、1=SH。
文档接口仍为FCL/MatMul等算子，未找到有完整语义的直接TP选择接口；库虽导出
`gcoVX_ForceTpCoreCount`，安装头文件和算子指南没有其契约，未采用。M16的卷积映射
尚缺控制变量后的独立收益归因，不能由NN类型直接推出它更快。
证据`implementation-and-profiler-audit-20261013/{summary,nbg-operation-audit}.json`及同目录
函数反汇编；仅分析，未上板或新增测试入口。

应用权重在首次Run前已有固定消费绑定；424分块与10矩阵首/中/尾960B抽样通过，
执行期应用无AddTensor/Setup/Verify。抽样不证明全部权重、物理驻留或内部零搬运。
层间hidden和KV没有应用逐层下载/上传，逻辑tensorcopy节点数不是物理通信量。

早期strace图区间得到ioctl约92%–94%，有追踪扰动。内存汇总关闭/开启/关闭后，
提交后Wait占外层图99.536%、Submit0.255%；Wait包括未完成计算、内部搬运和驱动，
不能作为IO占比。手写decode应用CPU约0.412%，编译FFN约1.72%、逐decoder约0.723%，
未测后台线程，CPU不能与墙钟相加；显式控制/token IO约0.421–0.551ms/token。

旧编译FFN链归因：attention相关438.283ms（66.345%），FFN85.029ms（12.871%），
末残差/LN30.368ms（4.597%），head103.018ms（15.594%）。逐decoder NBG为510.445ms
（82.081%），head103.103ms（16.579%），外部KV提交5.014ms（0.806%）。
SDK汇总关闭/开启/关闭扰动分别−0.061%/+0.058%；不能将旧占比搬到最新模型。
局部FFN快11.7倍不等于整模同比加速，按当时归因完全消掉FFN也仅约1.148倍。
证据`decode-wait-attribution-20261009/`、`profiler-decode-survey-20261012/`。

Async由官方样例源码及SDK反汇编确认：同Context一个worker/任务队列，再进入普通
同步ProcessGraph；不意味着设备多核重叠。官方多图多句柄用`gcoHAL_SetMnpuAffinity`
分配独立帧，`--cores`不是单图自动切分，可能复制权重/增加槽位；本项目未采用。
静态EVIS/NN/FCL名称、库名、核编号都不足以证明物理执行或利用率。

匹配转换镜像的VivanteIDE5.6.0文档说明vProfiler采集、VPD Analyzer解析；板端环境：

| 设置 | 意义 |
|---|---|
| `VIV_VX_PROFILE=1/0` | 开启/关闭OpenVX采集；OpenGL的VIV_PROFILE不等价 |
| `VP_OUTPUT=/绝对路径/request.vpd` | 普通用户可写目录、每轮独立输出 |
| `VP_ENABLE_PRINT=0` | 请求关闭打印，不等于停止采集 |

`run_device`默认关闭，显式profiler_output开启；不与恢复控制、SDK/ioctl计时同用。
8个真实FFN集合（2暖机+3×16正式）关闭/开启/关闭为2.9563/10.1490/2.9747ms，
+242.24%扰动，输出逐位一致；SDK未遵守关闭打印，并改名`capture_<pid>_1.vpd`。
71,970B文件含VP2000头、50记录×164字段，chip/product 0x9000/0x1000001d；HI总/idle
与stdout匹配，但没有NN(0x310000)/TP(0x320000)分组。硬件counter单位、覆盖、时钟域及
Analyzer兼容性未确认，不能推断CPU执行、NN闲置、核利用率或PCIe瓶颈。

已逐字节解析至EOF：frame ID为0–49，只有`VPC_ELAPSETIME=0x30001`，无CPU时间字段。
`gcoOS_GetTime`反汇编为gettimeofday秒×10⁶+微秒；`vxoProfiler_End`只写入低32位
`end−start`，不保存绝对起止时间。正式48轮耗时字段合计3.870116s，应用计时3.879141s；
每条对应包含8个FFN的一次整图执行，不是算子事件，不能还原间隙、NN/TP重叠或流水时间线。
Begin先Flush/Commit/Stall，End收集counter后才取结束时间；采集包含额外同步，所见时间线
也会受扰动。墙钟非单调，32位微秒约4295s回绕；本次毫秒区间未触及宽度限制。
证据`implementation-and-profiler-audit-20261013/profiler-time-audit.json`及Begin/End/GetTime反汇编。

使用同绑定/重复数关闭→开启→关闭，排除准备/暖机，报告扰动和漂移，生产延迟取关闭
结果。IDE `Tools → VPD Analyzer → Load VPD File`先核对System Info；NP101文件解析
尚未验。NN busy/MAC、DDR/SRAM、每核idle和TP是候选字段，需确认单位/回绕/覆盖，
不能直接相加或把DDR等同PCIe。没有新依据不扩大高扰动采集或改驱动/电源。
原文、反汇编在 `compiled-opt-exploration-20261012/profiler/`，实测在
`profiler-decode-survey-20261012/{profiler-summary,capture-inspection}.json`。

后续离线复核：安装HAL头有NN/TP busy、MAC、DDR/SRAM、idle及overflow字段，但
`gceCOUNTER_OPTYPE`是DRAW/BLT/COMPUTE等操作类别，不是NN/TP选择器；头中声明的
`gcoPROFILER_Start`不在固定libGAL导出表。镜像内有20220310版Analyzer插件，抽样类
常量池有elapsed标签，未见NN_/TP_/绝对起止标签；未运行其解析器，不据此断定兼容性。
主机未设VP_DISABLE_PROBE/VP_SUPPORT_V62；未盲改采集或重跑高扰动VPD。
通用每核字段不证明NP101实际核心数。证据`opt-entry-integration-20261014/profiler-*`。

## 方法来源与证据入口

| 来源 | 可借鉴方向与限制 |
|---|---|
| [OpenVINO NPU](https://docs.openvino.ai/2025/openvino-workflow-generative/inference-with-genai/inference-with-genai-on-npu.html) | 固定形状、分块prompt、两阶段配置与编译缓存；准备收益与推理收益分开 |
| [ExecuTorch/QNN](https://github.com/pytorch/executorch/blob/main/examples/qualcomm/README.md)、[ORT/QNN](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html) | 共享buffer/allocator、权重/context复用；本项目已共享应用tensor，物理搬运仍需测量 |
| [LiteRT NPU](https://developers.google.com/edge/litert/next/npu)、[Core ML/ANE](https://machinelearning.apple.com/research/neural-engine-transformers) | 分别借鉴设备buffer交接、布局/对齐与减少中间复制；平台内存契约和ANE特定卷积方法不能直接套用 |
| [vLLM Ascend ACLGraph](https://docs.vllm.ai/projects/ascend/en/main/developer_guide/Design_Documents/ACL_Graph.html) | 捕获/重放与shape分桶；当前已按图提交，先量化设备空隙，不由此推定合大图收益 |
| [llm.npu](https://arxiv.org/html/2407.05858v2)、[HeteroInfer](https://arxiv.org/html/2501.14794v2) | prompt分块、真实矩阵形状与异构同步；量化、CPU decode及移动统一内存前提与当前里程碑不同 |
| [FlashAttention](https://arxiv.org/abs/2205.14135)、[Flash-Decoding](https://pytorch.org/blog/flash-decoding/)、[NeuPIMs](https://arxiv.org/html/2403.00579v2) | 长KV并行/减少访存及专用NPU+PIM重叠；当前C64、SDK接口和硬件条件不支持直接移植，不新增自定义kernel |

单家庭端优先单请求标准自回归；不在初版DLM内猜token，多端batch和DLM/TLM联合
推理待TLM服务与token契约就绪。对NP101方法的适用性以本项目证据为准，不搬用外部加速比。

设备状态仅由事故台账和实时恢复标记判定；历史成功不放行新运行。

本轮退役内存压力/完整性/计数/重绑实验、packed-cache矩阵、Qwen局部decoder及
手写递归ping-pong、部分head NBG、vendor诊断入口；已验完整模型与必要保护保留。
本轮仅做构建和CPU契约检查，未重测硬件或声明新配置通过。旧命令不是当前操作说明。
