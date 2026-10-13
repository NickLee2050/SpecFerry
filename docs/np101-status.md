# NP101 推理实现与优化结论

更新：2026-10-13。活跃模型为真实 `facebook/opt-350m`；目标是保持旧实现的正确性，
降低单请求decode延迟。未完成项见[TODO](../TODO.md)，每次上板先读
[事故台账](np101-recovery-incidents.md)，接口见[测试手册](../tests/README.md)，
SDK失败契约见[问题表](np101-sdk-driver-validation-report.md)。

原始证据相对 `.cache/runs/`，不随Git分发；压缩前全文、代码与hash在
`commit-cleanup-20261013/before/`、`before-manifest.json`。早期归档见
`documentation-consolidation-20261010/`；退役脚本保存在压缩证据归档中。

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
- 完整decoder候选每层一份NBG，M1/C64、普通rank3 FP32 KV，加独立copy图提交；
  共27图。未接block16，不作NBG输入/输出原地别名。
- 全图NBG包含embedding、所有decoder、project_out、LM head及greedy；独立48-copy
  KV提交图，共2图。仅M1/C64已验，主机逐token调度；48个FFN投影TTGT无明显收益。
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
| 完整decoder NBG / C64 | 第0层端口/接续/末槽及单份循环KV通过；24层block1、17输入/16输出，三进程各1暖机+1正式请求；选词、最终logits、末层KV/未写槽、27图外层operation、释放及各45秒门禁通过 | 普通FP32三维KV；未重验全层KV、block16、C128或长期稳定性；正式CLI未接入 |
| 单个全图NBG / C64 | M1、17输入/16输出，1暖机+1正式请求；全部选词、最终logits、末层KV/未写槽、2图外层operation、预算/释放及45秒门禁通过 | 含GatherND及INT32选词修正；KV回写独立，不是整段生成循环编进NBG；未验block16、C128或长期稳定性 |

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
| 原始FCL、1×1卷积、强制VX投影 | 失败/无有效匹配收益；诊断分支退役。NBG可用不证明手写FCL故障修复 |
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
不覆盖或与后续匹配平均混算。编译模型规范入口见测试手册，正式导出仍待完善。

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

应用权重在首次Run前已有固定消费绑定；424分块与10矩阵首/中/尾960B抽样通过，
执行期应用无AddTensor/Setup/Verify。抽样不证明全部权重、物理驻留或内部零搬运。
层间hidden和KV没有应用逐层下载/上传，逻辑tensorcopy节点数不是物理通信量。

早期strace图区间得到ioctl约92%–94%，有追踪扰动。内存汇总关闭/开启/关闭后，
提交后Wait占外层图99.536%、Submit0.255%；Wait包括未完成计算、内部搬运和驱动，
不能作为IO占比。手写decode应用CPU约0.412%，编译FFN约1.72%、逐decoder约0.723%，
未测后台线程，CPU不能与墙钟相加；显式控制/token IO约0.421–0.551ms/token。

| 已验版本的同步SDK阶段 | ms/输出token | decode占比 |
|---|---:|---:|
| 编译FFN版：QKV/KV/attention/out_proj/首残差/LN | 438.283 | 66.345% |
| 编译FFN版：NBG FFN执行 | 85.029 | 12.871% |
| 编译FFN版：末残差/LN | 30.368 | 4.597% |
| 编译FFN版：LM head | 103.018 | 15.594% |
| 逐decoder版：24层NBG | 510.445 | 82.081% |
| 逐decoder版：LM head | 103.103 | 16.579% |
| 逐decoder版：外部KV提交copy | 5.014 | 0.806% |

编译FFN的SDK汇总扰动−0.061%，逐decoder+0.058%（关闭/开启/关闭）；数据只涵盖
对应旧模型，不把比例搬到最新全图。FFN局部11.7倍受Amdahl限制；当时完全去掉
85ms也最多约1.148倍。外部KV提交不是该版本大头；全图内部仍缺可靠分项。
证据 `decode-wait-attribution-20261009/`、`weight-preload-and-vx-review-20261009/`、
`profiler-decode-survey-20261012/`、`compiled-decoder-integration-20261012/`。

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
与stdout匹配，但没有NN(0x310000)/TP(0x320000)分组。单位、覆盖、时钟域及Analyzer
兼容性未确认，不能推断CPU执行、NN闲置、核利用率或PCIe瓶颈。

使用同绑定/重复数关闭→开启→关闭，排除准备/暖机，报告扰动和漂移，生产延迟取关闭
结果。IDE `Tools → VPD Analyzer → Load VPD File`先核对System Info；NP101文件解析
尚未验。NN busy/MAC、DDR/SRAM、每核idle和TP是候选字段，需确认单位/回绕/覆盖，
不能直接相加或把DDR等同PCIe。没有新依据不扩大高扰动采集或改驱动/电源。
原文、反汇编在 `compiled-opt-exploration-20261012/profiler/`，实测在
`profiler-decode-survey-20261012/{profiler-summary,capture-inspection}.json`。

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
推理暂缓。对NP101方法的适用性以本项目证据为准，不搬用外部加速比。

最新已存健康检查：2026-10-13 15:06:01 +08:00，boot
`2e8953e2-a612-45e9-8a97-17f94fbe0cc0`，v105两组对照通过、无故障/owner/恢复标记。
这是历史窗口；每次新运行仍遵守事故门禁。第23层附近的operation缺项、AddTensor
失败和EDMA是不同API事件，不共用一个层号解释。关机page fault、卷积/旧OPT失败
原因仍未闭环，逐次证据与规避保存在独立事故台账。
