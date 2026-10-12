# NP101 推理实现与优化结论

整理日期：2026-10-12；设备与性能证据截至 2026-10-12。目标是在旧 component 正确性的
基础上降低 OPT-350m 单请求 decode 延迟。完整模型在已测范围内通过，prefill 明显加速，
持续 decode 仍约 **0.71–0.72 输出 token/s**，主要性能目标尚未完成。

本页集中维护实现、验收、性能和技术结论；[TODO](../TODO.md) 只维护未完成工作。
[SDK 问题表](np101-sdk-driver-validation-report.md) 维护失败契约，
[冷启动事件台账](np101-recovery-incidents.md) 维护运行门禁、独立事故及规避，
[测试手册](../tests/README.md) 维护命令与诊断接口。

## 实现与验收范围

- 活跃 DLM 为真实 `facebook/opt-350m`，24 层，checkpoint 默认 greedy。Qwen3.5
  既有实现、checkpoint 和回归证据保留；DLM/TLM 联合推理尚未验证。
- 已测计算链在 NP101 图内完成 embedding、decoder、KV 更新、最终归一化、LM head
  与 greedy；主机负责分词、初始化、调度及 token 读回/文字展示。Replay→Submit→HAL
  COMMIT→完成等待支持编译设备命令路径，不代表逐指令核利用率证明。
- Python 入口默认仍为 component 回归后端；graph 默认 `merged-kv`，固定每图一层 decoder。
  两路 fc1、block16、last-token head 和 mapped 诊断均按明确选项使用。
- 固定图、固定绑定与常驻权重/KV；prefill/decode 两套图共享 backing。每层 K/V
  分别用 `[1024,C]` 存储，供 16 头 attention；普通请求 reset 后复用，不重新编译。
  block4/16 处理同一请求的已知连续输入，尾部和 decode 使用 block1，不等待凑块。
- 分块 KV 写使用已校准的 `vxReshapeTensor` 写别名：普通 RESHAPE 可能复制，缺少
  共享写回保证，这是有语义依据的 API 例外。已校准真实头维、混合写入、覆盖、末块
  和 causal 不变性；不交换运行期句柄。分次追加 prompt 的增量输入接口尚未实现。
- 模型重建先释放全部旧图及最后一个旧 Context，再创建新 Context；并存 Context
  共享进程 shader 槽表。普通请求继续复用模型，不能用第二 Context 隔离槽位预算。
- const/nonconst 应用 tensor payload 分别限 1 GiB，共享 backing 只计一次并计临时
  峰值；SDK 隐藏副本、workspace 和物理位置未知。预算合规不等于数据完整性保证。

当前维护入口集中在完整生成、logits/KV、生命周期及性能集合。旧前缀诊断、局部权重/
双层合图、预加载抽样、四路fc1及单层权重复用实验已退役，源码快照见
`.cache/runs/code-consolidation-20261010/before/`。精简后已完成主机编译、离线回归及下表
最后两项板卡验收；此前各项保留原版本证据，不宣称已对当前binary全量重验。

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

**已接受的精度例外**：C64 的16-token prompt，第15个新词在同一30-token历史下，
CPU FP16 增量选164（` going`），设备 block1/block4 均选45（` not`）。两者 logit
差在 CPU/设备分别为 +0.000976563/-0.000976563；CPU FP32 也选45。完整向量最大
误差0.03125，通过 `atol=0.08, rtol=0.02`。沿设备实际历史下一词均为686，最大误差
0.0234375。仅此近并列差异获准接受，原失败证据及严格检查保留，不豁免未来任意分歧。

## 做过的修改与决策

| 工作 | 观察与收益 | 决策 |
|---|---|---|
| 实际节点规模 / 图边界 | 单层612 SDK节点 Verify约141–169秒；两层993节点到758秒仍未完成；两层四张分图累计11.026秒且数值通过。缩小预留量无明显收益 | 采用可控固定小图；非线性已观察，O(n³)仍只是经验假设 |
| 合并逐头 KV | AddNode 从旧component的1572增到逐头KV的4741，约96%增量来自KV/控制；合并后1861。每层KV节点126→6，extension1143→423 | 保留合并KV，降低真实展开和共享槽位压力 |
| 固定图与共享存储 | 消除新路径运行期Verify，减少控制IO；两套图共享一份权重/KV | 保留；应用共享不证明物理零复制 |
| block4 prefill | 15-token输入图prefill约23.09→9.72秒，减少58% | 保留；首词改善为主 |
| 仅末prompt块运行head | 对应C16实验prefill/总图时间减少6.10%/4.68% | 保留显式last-token策略，decode仍每步运行head |
| 两路fc1 | 沿输出通道拆两次MatMul后Concat，保留完整归约与FP16边界；FFN集合快9.27%，完整模型decode图时间少5.16%–6.22% | 保留显式候选；预装fc2后曾绕过AddTensor失败，根因未定 |
| 更大已知输入块 | FFN M1→M4每输入成本低约72%；M4/M8/M16约0.258/0.146/0.088秒每输入每24层轮。模型64-token prefill由block4约22.7秒降到block16的8.233秒 | 接入block16；算子回放及输入处理率不作输出吞吐 |
| 固定LayerNorm参数 | 初始化将96份FP16 gamma/beta扩为共享FP32 `[1024,1]`；动态归一化精度不变；52图编译节点4227→3843，只有tensorcopy计数少384 | 保留图简化；Verify约13.2–13.5→10.7秒，decode无提速 |
| 更大/更小图、存储变体 | 两decoder合图、attention/FFN再拆图、图内const初筛均未获益 | 不采用；不据此判断所有图边界 |
| 四路fc1 | 整模型额外收益仅约1%–2%，两套图槽位估算894→990/1024 | 模型及集合入口均退役，保留证据 |
| 权重预转置 / 缩小工作集 | 预转置慢1.60%；FFN矩阵工作集384→16 MiB，集合慢约4%–5% | 不接入模型；不能排除带宽/隐藏搬运瓶颈 |
| 1×1卷积 / VX MatMul候选 | 卷积Verify进入FCL优化器崩溃；VX集合未建立采用收益，匹配EVIS对照hang，无可比时间 | 关闭探索、退役专用入口，保留SDK自动选择；不能声称后端性能等价 |
| 诊断读回 | 大量SDK-copy曾触发EDMA；减少读回、显式Map/Unmap后所列粗验通过 | 常规生成只读token，诊断按需要粗读；未证明永久规避EDMA |

AddNode、编译节点、extension及shader槽位是不同计数。不同日期/boot/负载的收益
不能相乘或相加；增加节点的两路fc1能提速，减少节点的LN预处理没有decode收益。
LN预处理使共享权重净增0.1875 MiB至662,589,440 B，nonconst峰值反降约1.297 MiB
至766,517,812 B，const为13,884 B；关闭后均归零，保留了初始化收益。

## 性能口径与代表性结果

- **驻留请求输出吞吐**：`Σ实际输出数 / Σgeneration.total_seconds`，含prefill/首词、
  decode、必要IO及调度；排除初始化、启动前Setup/Verify、reset、CPU参考、落盘、释放。
  旧实现运行期重复Verify仍计入实际请求墙钟，不能为美化输出吞吐而扣除。
- **首词后decode速率**：`Σ(输出数−1) / Σ(total_seconds−first_token_seconds)`；
  单输出请求无此指标。暖机剔除，不合并不同长度/配置；TTFT单列。
- **图时间**：外层RunGraph及独立ProcessGraph，包含完成等待，剔除独立Verify、
  显式IO等，不重复相加内层区间；不是纯NPU算术时间。输入处理与算子回放另列。
- 使用代表性重复集合、前→候选→后和内存汇总；新增追踪做关闭→开启→关闭对照。
  已重算历史39组/103次正式请求；旧版长输入的单次数值记录不作为持续硬件吞吐基线。

下表输出速率均按正式请求的输出数/时间累计计算。前三行为历史匹配对照；后两行
说明当前水平，不能据跨boot的差值归因某项优化。除常驻复用10次外，每组3次正式请求。

| 配置 | 输入→每次输出 | 请求输出 token/s | 首词后decode token/s | TTFT中位秒 |
|---|---:|---:|---:|---:|
| 旧component / C16 | 5→8 | 0.368 | 0.540 | 8.812 |
| 合并KV / block1 / C16 | 5→8 | 0.430 | 0.646 | 7.760 |
| 合并KV / block4 / C16 | 5→8 | 0.575 | 0.657 | 3.242 |
| 两路fc1 / block16 / C64，常驻复用10次 | 17→16 | 0.660356 | 0.720658 | 3.413 |
| 同配置加LN预处理 / C64 | 32→32 | 0.672882 | 0.714363 | 4.159 |

- 合并KV相对旧component的请求/decode输出速率高16.93%/19.71%，含旧运行期Verify/IO
  减少的收益。block4相对block1的请求高33.68%，decode变化很小，主要缩短首词。
- 两路fc1匹配对照的decode图时间1590.990→1508.881→1608.923 ms/token；按实际输出
  墙钟重算，decode速率相对两侧基线高5.40%/6.70%。
- block16的64输入/1输出：图prefill约22.7→8.233秒，实际请求输出约0.044→0.1215
  token/s；7.773是**输入处理率**。63-token输入有15个单token尾部，粗验未观察到收益。
- LN预处理前→候选→后，17输入/16输出，图decode中位1388.886→1392.695→1385.400
  ms/token，候选慢0.27%/0.53%，没有推理提速。Verify总计13.455/10.722/13.166秒。
- 10-09的32输入/32输出三次均与CPU参考逐token一致，共96输出/142.669892秒；首词后
  93输出/130.186002秒。请求中位47.565700秒。英文续写有重复，CPU参考同样重复；
  准确文字及IDs在 `layernorm-affine-preparation-20261009/generated-text.json`。

10-10精简后短请求正式16输出/24.434267秒，输出/decode速率0.654818/0.716318，
首词3.493851秒；初始化20.982537秒另计。C128为有界验收请求，未剔除暖机，
保留原有SDK内存汇总/编译守门，不作为新的持续性能基线：

| C128输入→每次输出 | 次数 | 累计输出/生成秒 | 请求输出 token/s | 首词后decode token/s | 首词均值秒 |
|---|---:|---:|---:|---:|---:|
| 96→32 | 2 | 64 / 113.310060 | 0.564822 | 0.702754 | 12.542853 |
| 97→32 | 1 | 32 / 57.920707 | 0.552479 | 0.703752 | 13.871129 |

C128初始化20.918025秒另计；52图/3843编译节点，权重仍为662,589,440 B，
const/nonconst峰值26,172/808,396,852 B。最终logits最大误差0.025634766/0.0234375，
均通过既有容差。英文阅读题中，钥匙明在厨房桌上，模型却答“包里”并重复天气问答；
闭馆日答对“周日”，后续续写虚构地址。CPU与设备的完整32-token续写完全一致，
这两例说明计算一致性通过不等于回答质量通过；不是新增设备精度例外。完整原文、
未修饰续写、IDs及统计在 `post-cleanup-context-20261010/`。

## FFN 拆分与精度对照

10-10临时实验复用真实24层OPT、C64/block1、17输入→16输出，各1暖机+1正式请求，
顺序为FP16前基线→fc2拆分→INT8→后基线。准备/Verify另计，运行期无编译；最终
logits在计时外映射读回。block1首词/请求吞吐不能直接与block16成绩比较。

| 配置 | 24层FFN集合秒/轮 | 整模decode秒/token | 请求输出token/s | 首词后decode token/s |
|---|---:|---:|---:|---:|
| FP16前/后基线，fc1两路 | 0.833293 / 0.838159 | 1.403291 / 1.399385 | 0.370544 / 0.371218 | 0.712611 / 0.714600 |
| FP16，fc1/fc2各两路 | 0.771849 | 1.325552 | 0.393179 | 0.754403 |
| W8A8线性层，fc1两路 | 1.532214 | 2.435580 | 0.210961 | 0.410580 |

- **两路fc2仍是候选**：沿输出通道拆分，每路保留完整归约，拼接后加bias，不复制
  权重。FFN时间减少7.37%–7.91%，整模decode延迟减少5.28%–5.54%；16个输出及
  最终50,272维logits与基线逐字节一致。单套图extension447→471，权重仍
  662,589,440 B；block16组合与长期复用未验，正式代码未接入。
- **INT8路线已关闭**：线性层用DFP W8A8，权重/激活步长1/256、1/8，MatMul输出
  FP16，其余敏感状态及共享head保留原精度。170个GEMM选中`evis.gemm_I8I8toF16`，
  权重payload降至359,550,976 B，但FFN慢约83%、整模decode慢约74%。整模仅验
  logits有限与重复一致性，输出16个换行，无整模INT8精度/质量验收；dtype、布局及
  转换同时变化，不能把减速归因于输出FP16或据此否定整数峰值能力。
- **输出改INT32无已证实通路**：同库内嵌shader的I8→F16/I8变体包含FP32归约；
  MATRIXMUL的28项类型表不含I8×I8→I32，公开参数无累加类型开关。源码/分派名称
  不足以确定物理乘法器或解释减速，用户已停止此路线。

FFN使用24份不同真实权重，4轮暖机后3×8轮计时；该集合不是输出吞吐。8次板卡
进程均释放/预算归零，各至少45秒延迟门禁通过。精确参数、前后漂移及独立参考见
`int8-and-ffn-partition-20261010/`；整数类型审计见`int8-output-type-audit-20261010/`。

## FP16 计算压力与部件消融

以下均按MAC=2 FLOPs，分子只计GEMM/卷积，分母为固定绑定的重复图集合，含附加
算子、内部搬运和完成等待；准备/Verify、显式IO、日志及释放在计时外。有效GFLOP/s
不是输出吞吐、净kernel时间或物理利用率。完整证据分别在
`fp16-compute-scaling-20261010/`、`gemm-ffn-ablation-20261010/`。

| 集合 | 每轮图数 | 每轮秒（批次中位） | 有效GFLOP/s |
|---|---:|---:|---:|
| FFN M=1 前/后 | 24 | 0.906432 / 0.904791 | 0.444218 / 0.445024 |
| FFN M=16 | 24 | 1.410359 | 4.567950 |
| FFN M=64 | 24 | 6.101793 | 4.223316 |
| FFN M=256 | 24 | 24.852782 | 4.147593 |
| 3×3卷积集合 | 8 | 1.197557 | 8.069493 |

FFN使用24份OPT真实权重、fc1/fc2各一路；每轮GEMM算术量为
`24×4×M×1024×4096`。卷积按demo契约，以8份不同随机输入/权重执行
64×64、128入/出通道、3×3 CONV→RELU→2×2 MAXPOOL，每轮卷积算术量为
`8×2×64×64×128²×3²`。测三个批次，大M缩减重复轮数；末图完整输出粗验通过，
M1前后漂移−0.181%，最大nonconst峰值571.232 MiB；6进程释放及延迟门禁通过。

随后以8份不同真实权重逐步增加FFN部件；纯GEMM每图只有一个
`evis.gemm_transb_F16F16toF16`。M1、M16、M64、1024方阵、M16第二投影分别为
0.413、4.467、4.237、4.156、4.624 GFLOP/s。固定M16的消融如下：

| 累计结构 | 平均ms/图 | 较上一步增加ms | 有效GFLOP/s |
|---|---:|---:|---:|
| GEMM1 | 30.046773 | — | 4.466960 |
| +bias1 | 30.048160 | 0.001387 | 4.466754 |
| +ReLU | 30.233394 | 0.185234 | 4.439387 |
| +GEMM2 | 58.553615 | 28.320221 | 4.584439 |
| +bias2，核心FFN完整 | 58.933232 | 0.379616 | 4.554908 |
| +残差 | 58.617851 | −0.315380 | 4.579415 |
| +LayerNorm | 60.876714 | 2.258863 | 4.409493 |

**低有效算力已在纯GEMM出现**；最大非GEMM增量为LayerNorm约2.26ms/3.85%。
增量包含整图优化/状态差异，负增量不认定为加速。12进程末图完整输出粗验最大误差
≤0.00390625，前后M16漂移−0.00458%；释放/预算/延迟门禁通过。不能分离GEMM
内部计算、搬运与等待；后续应核实编译路径及厂商预期，而非继续盲目扩大M。

### 矩阵尺寸与完整TTGT

10-12确认冷启动并通过固定控制后，完成全部8种尺寸配对。
`C=A·B → Cᵀ=Bᵀ·Aᵀ → C`显式转置两个输入及输出，三次转置全部计时。
原始路径存储Bᵀ，TTGT存储数学B，主机布局准备在计时外；数学输入相同。
除2048方阵用确定性随机FP16外均取OPT权重，每集合8份不同矩阵。

一图8组独立GEMM，2048方阵两图各4组以控制单次调用长度；每布局独立进程，
2轮暖机后3批、每批至少2轮。表中为集合中位时间÷8，非孤立调用延迟；分子为
8×2MNK，计时范围同上，未将合图的成功解释为历史hang根因已修复。

| M / K / N | 原始均摊ms | 原始GFLOP/s | 完整TTGT均摊ms | TTGT GFLOP/s |
|---|---:|---:|---:|---:|
| 1 / 1024 / 4096 | 20.488 | 0.409 | 282.677 | 0.030 |
| 16 / 1024 / 4096 | 29.689 | 4.521 | 285.551 | 0.470 |
| 16 / 4096 / 1024 | 28.679 | 4.680 | 272.540 | 0.492 |
| 64 / 1024 / 4096 | 126.035 | 4.260 | 348.383 | 1.541 |
| 256 / 1024 / 4096 | 510.485 | 4.207 | 618.060 | 3.475 |
| 1024 / 1024 / 1024 | 514.145 | 4.177 | 494.418 | 4.343 |
| 1024 / 1024 / 4096 | 2045.272 | 4.200 | 1796.324 | 4.782 |
| 2048 / 2048 / 2048 | 4101.422 | 4.189 | 3235.913 | 5.309 |

完整TTGT对小M明显更慢；三种最大形状分别减少约3.8%、12.2%、21.1%的时间，
仍无量级改善。M16另测预转置输入、仅交换乘法得到21.058ms/6.374GFLOP/s，
其输入准备和CPU最终转置不计时，不能作为完整TTGT或模型收益。

25进程检查全部编译GEMM节点及最后一组完整输出，最大误差0.00390625，前后
输出逐字节一致、最大漂移0.559%；nonconst峰值384 MiB、const峰值0，释放归零，
各45秒延迟门禁通过。证据`gemm-layout-collection-20261012/summary.json`；10-11
失败轮无有效性能结论，调用定位及关机page fault独立保留于事故台账。

## 官方资料、离线模型与性能对照

2026-10-12联合阅读本机 `~/NP101_Official`、`~/NP101_API` 已展开的6份PDF
（39页）及56份Markdown，补查示例源码、驱动归档及SDK符号。系统/容器镜像内部
依赖文档未穷举。以下路径相对这两个目录；原始资料不复制进Git。

| 资料 | 已确认用途 / 边界 |
|---|---|
| Official `docs/np101-pe-22_board_manual_rev0.6.pdf` 第3–4、9–11页；`np101_chip_overview_rev0.6.pdf` 第4页 | 厂商规格：12个NN处理单元、48个CPU单元、4GB SeDRAM、2TB/s；INT8 20TOPS、FP16/BF16 10TFLOPS、FP32 512GFLOPS、FP64 256GFLOPS、典型10W。不是本机实测峰值/功耗；软件栈图区分CPU/RVV与NPU后端，虚线模块仍在开发 |
| Official `docs/np101_ep_model_deployment.pdf`；`np101_soc_model_deployment.pdf` | 相同模型转换流程，EP构建主机程序，SoC交叉编译RISC-V程序；支持FP16配置，不必先开展INT8量化 |
| Official `demo/ModelZoo/README.md`；`tools/compile/ModelTest/readme.md` | 网络二进制与运行库需匹配；NBG、metadata、生成源码应成套更新，再构建应用 |
| Official 串口/网口/SD说明、SoC CPU多核示例 | 系统维护、串口日志及CPU应用部署参考。UART3、115200/8N1、UTF-8；串口工具和libgomp不证明NPU profiling或单图多核能力 |
| API `ref_op_api_guide.md`、`op_demo_conv_relu_pool/README.md` | 与仓库demo对应文件逐字节相同，无需重复导入；算子可调用性不等于全部参数/物理引擎已验证 |
| API `ljmnp_drivers_1.0.5-ubuntu20.04_x86_64/updatelist.txt` | 声明修复Transformer导出NB、显存写入正确性及ARM拷贝bus error；未证明对应本项目事故。README残留1.0.1例子，不能据文件夹名确定当前内核版本 |

### 编译产物与执行证据

```text
ONNX → 导入 / 精度配置 / 离线编译
     → network_binary.nb + nbg_meta.json + vnn_* / main 等源码
     → C/C++构建 infer.out 或 libinfer.so
     → SDK加载独立.nb → Setup → Verify → 重复RunGraph → 释放
```

`.nb`是NBG网络二进制，不是可直接启动的ELF；动态库也不替代它。
`convert_model.py`包装已有生成源码并尝试构建，真正模型编译由转换工具的Pegasus完成。
离线编译仍有运行期准备，但不能由这些调用推出内部重新完整编译或准备成本相同。

上述调用链**直接来自官方源码**：以Official
`tools/compile/ModelTest/apps/yolov5s_uint8_nbg_unify/`为例，
`vnn_yolov5suint8.cpp:231–233`创建NBG文件节点，`:352`调用`vsi_nn_SetupGraph`；
`main.cpp:91`调用`vsi_nn_VerifyGraph`，`:143`同步调用`vsi_nn_RunGraph`，
`:235–255`为创建/预处理/Verify/执行/后处理/释放主流程。反汇编只补充确认实际ELF
的RunGraph调用，不是Setup/Verify结论的唯一依据，也未由此推断硬件引擎。

### 转换工具的已知契约

本机Official `tools/model_conversion/quantize_toolkit`的OCI包内工具为6.6.1，
导出目标`VIP9000LD_PLUS_PID0X1000001D`，调用Pegasus的
`export ovxlib --dtype float --pack-nbg-unify`。脚本把`fp16`映射成`float`并跳过
整数量化；此命名不保证每个节点/累加均为FP16，目标与当前SDK兼容性仍需验证。

- 导入脚本生成`*_postprocess_file.yml`，导出脚本读取`*_postprocessmeta.yml`；
  需在工作副本核对。脚本执行后仍有echo，不能仅用退出码判断成功，须检查产物/日志。
- ONNX常量权重二维MatMul/Gemm可能融合为`fullconnect`；也有通用matmul规则。
  必须看导入JSON/编译报告，不能用ONNX算子名保证避开历史FCL优化器风险。
- 已见样例为固定图像shape；OPT需要张量meta、无图像预处理。动态shape、ONNX
  opset上限、自动KV管理/原地别名及NBG之间权重共享尚未确认。可见规则无显式
  LayerNormalization，必要时按原语导出并保持敏感精度，不以规则表代替执行验收。
- 单NBG外层一个节点不表示内部一个算子，也不证明绕过共享shader槽位限制。
  两份现有完整权重为1,325,178,880 B，超过单段1GiB；独立prefill/decode NBG
  不能沿用手写图共享一份权重的结论，优先单套M1并复用其处理prompt。

### 官方模型的有界实测

10-12同boot、当前 `/usr/lib/ljmicro`、单实例、默认affinity；以下为所附ONNX的
Conv/Gemm算术量（MAC=2 ops）除以NBG图时间，按实际文件逐节点计数并独立复核，
不按模型名称套标准FLOPs。除这两类算子外的工作只计入分母。

| 提供的模型 / 精度 | 正式次数 | 均摊ms/次 | 源模型GOP/次 | 等效GOP/s |
|---|---:|---:|---:|---:|
| YOLOv5s / UINT8 NBG | 64 | 69.5253 | 16.4336 | 236.368 |
| ResNet34目录中的模型 / UINT8 NBG | 192 | 7.40454 | 7.32752 | 989.599 |

各先暖机2次；YOLO累加官方逐RunGraph计时，打印在调用间；ResNet为原生循环3×64
次集合。准备、显式IO、后处理和释放不计入。YOLO真实图片得到bus/person检测，
无独立准确率oracle；ResNet用固定随机输入，1000 logits有限、非恒定、前后逐字节
一致，不是分类准确率验收。两进程正常退出/释放，各45秒延迟内核门禁通过。
ONNX与NBG为随附配套且IO匹配，但无编译器源文件哈希映射；不能称为真实指令计数。
**这些是量化网络GOP/s，不是FP16 GFLOP/s**，不能据此许诺OPT提速或推定CPU回退。

### 执行引擎、并行与profiler线索

- API指南将EVIS/CL归为SH shader，与CPU回退分开；其中105/50/7/2是算子分类
  数量，不是核数。VX入口也可能再选EVIS GEMM，软件后端名称不能证明物理利用率。
- API 1.0.5包中的ovxlib/OpenVX/GAL与当前用户态库逐字节相同；这不鉴定运行中
  内核版本，也不证明离线/在线采用相同融合、布局、精度或执行引擎。
- 官方 `convert_model.py` 和ResNet的 `fps_test.py` 用多句柄/多图、线程及
  `gcoHAL_SetMnpuAffinity`分配独立帧。`--cores`不是单图自动切分开关，`--chunk`
  是领取任务批次；不能直接加速有自回归依赖的单请求。多实例可能复制权重并增加槽位；
  SDK语义和并发尚未校准，本项目未启用。安装脚本的12个EnableCore编号不证明核利用率。
- 板卡手册第10页§3.3宣称算子性能分析能力，SDK含`gc_hal_profiler.h`和部分
  `gcoPROFILER_*`导出符号；未找到独立采集程序、获取/安装/解码说明。驱动包的
  `/usr/bin`为空。FPS脚本测重放吞吐，串口软件采日志，都不是硬件计数器profiler。
  需要厂商提供匹配版本的采集器、解析器、最小命令及NN/SH/内存计数器语义；不直接
  试调用未验证HAL接口，也不由`libArchModelSw`等库名判定仿真执行。

资料清单/hash、英文改名映射、源码依据及实测原始记录已归档到
`official-knowledge-consolidation-20261012/`；主文保留结论，不导入第三方源码/
许可证、镜像或逐页日记。官方OPT编译尚未开始，步骤与通过条件见[TODO](../TODO.md)。

## 瓶颈归因与认识

| 问题 | 已有证据 | 仍不能推出的结论 |
|---|---|---|
| CPU是否占大头 | 7步decode应用CPU中位40.046ms，约墙钟0.412%；整进程CPU含初始化/编译/释放 | 未测驱动后台线程，CPU时间不能与墙钟相加 |
| 时间消耗在哪里 | 新增内层计时关闭/开启/关闭为1390.025/1389.392/1402.172ms/token；21步decode中，提交后Wait占外层图99.536%，Submit0.255%，显式token IO仅相当于0.019%–0.038% | Wait涵盖未完成计算、内部搬运、驱动和调度，不等于通信占比；三次对照不证明零计时开销 |
| 旧ioctl结论 | 历史strace按图区间配对得到92%–94%，有逐调用追踪扰动；新测量用内存汇总 | 新HAL/API占比不是重新测得的原始ioctl占比；嵌套区间不能相加 |
| 权重是否提前加载 | 首次Run前424个分块都有共享消费绑定；10个矩阵分块首/中/尾共960B与checkpoint一致，执行期应用无AddTensor/Setup/Verify | 抽样不证明全部权重、物理驻留、片上缓存或SDK内部零复制 |
| 内部数据流 | 层间hidden、权重和KV共享；应用无逐层hidden下载/上传。预处理前M1的690个tensorcopy包含转换/reshape等；C64的48次KV transpose逻辑输出共6MiB | 节点名称和逻辑字节数不是实测传输次数/流量，不应直接删除写后读的依赖节点 |
| 为什么batch有效 | M4工作量四倍而FFN轮时间只多约12%–14%；更大M继续降单位成本 | 还未分开向量利用、权重复用及提交摊销；不能直接批量计算未知的未来decode token |
| 并行与流水 | 每token的26图有依赖，Q/K/V和两路fc1独立分支已在图内；同Context Async为单worker | 换Async不自动建立设备重叠；没有文档完整的NN核数控制接口或物理利用率证据 |
| 第23层附近故障 | 分别出现过编译operation缺项、AddTensor加载失败、末层诊断EDMA | 层号不能代替准确失败API；1024槽解释不涵盖所有EDMA/内存损坏 |

已扫描15个去重SDK ELF的主机可执行节；当前模型GEMM注册为EVIS路径，静态分派
不能证明物理核占用。家庭端保持单请求标准自回归，初版不在DLM内部再增加猜token；
跨端请求合批后置到家庭中心。重点仍为**单步内部计算、数据搬运和同步的硬件时间线**。

## 外部方法调研的适用范围

10-09已核对官方/作者资料；保留方法来源，不将外部加速比当作NP101收益。

| 来源 | 可借鉴方向与限制 |
|---|---|
| [OpenVINO NPU](https://docs.openvino.ai/2025/openvino-workflow-generative/inference-with-genai/inference-with-genai-on-npu.html) | 固定形状、分块prompt、两阶段配置与编译缓存；准备收益与推理收益分开 |
| [ExecuTorch/QNN](https://github.com/pytorch/executorch/blob/main/examples/qualcomm/README.md)、[ORT/QNN](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html) | 共享buffer/allocator、权重/context复用；本项目已共享应用tensor，物理搬运仍需测量 |
| [LiteRT NPU](https://developers.google.com/edge/litert/next/npu)、[Core ML/ANE](https://machinelearning.apple.com/research/neural-engine-transformers) | 分别借鉴设备buffer交接、布局/对齐与减少中间复制；平台内存契约和ANE特定卷积方法不能直接套用 |
| [vLLM Ascend ACLGraph](https://docs.vllm.ai/projects/ascend/en/main/developer_guide/Design_Documents/ACL_Graph.html) | 捕获/重放与shape分桶；当前已按图提交，先量化设备空隙，不由此推定合大图收益 |
| [llm.npu](https://arxiv.org/html/2407.05858v2)、[HeteroInfer](https://arxiv.org/html/2501.14794v2) | prompt分块、真实矩阵形状与异构同步；量化、CPU decode及移动统一内存前提与当前里程碑不同 |
| [FlashAttention](https://arxiv.org/abs/2205.14135)、[Flash-Decoding](https://pytorch.org/blog/flash-decoding/)、[NeuPIMs](https://arxiv.org/html/2403.00579v2) | 长KV并行/减少访存及专用NPU+PIM重叠；当前C64、SDK接口和硬件条件不支持直接移植，不新增自定义kernel |

## 设备状态与证据入口

最后已存检查**2026-10-12T10:35:36.070813+08:00**，boot
`7d7a10f6-a1b8-4e20-84df-2b8805bdefbd`，官方两个NBG示例及各45秒延迟门禁通过，
无目标内核故障、owner或恢复标记。同boot此前已通过固定两级恢复控制及GEMM/TTGT集合。
这只覆盖所测负载；10-11 RunGraph停滞与10:32关机page fault独立保留，根因未闭环。
普通运行不再因boot改变自动删除恢复标记，必须两级恢复完整通过。

以下相对 `.cache/runs/`，原始证据不随Git分发；归档仅供取证，不能覆盖当前代码或
把当前实现冒充旧实验。首次清理前全文和SHA256在
`documentation-consolidation-20261010/before/`、`before-manifest.json`。

| 范围 | 证据目录／文件 |
|---|---|
| 图规模、编译表、合并KV | `graph-reservation-analysis-20260928/`、`attention-layout-analysis-20260928/`、`verify-boundary-analysis-20260928/`、`merged-kv-optimization-20260929/` |
| block4、完整模型、精度例外 | `block-prefill-acceptance-20260930/`、`model-acceptance-20260930/`、`token-logit-precision-20261001/`、`acceptance-decision-20261001.json` |
| 旧基线、吞吐重算 | `validated-opt-component-baseline-20261001/`、`output-throughput-review-20261009/summary.json` |
| 两路fc1与映射验收 | `ffn-layout-comparison-20261008/`、`split-fc1-performance-20261008/`、`recovery-followup-20261008-1802/summary.json`、`split-fc1-context-20261009/` |
| CPU/等待与预加载归因 | `recovery-followup-20261008-1802/decode-cpu/attribution-summary.json`、`decode-wait-attribution-20261009/`、`weight-preload-and-vx-review-20261009/` |
| 工作集、输入宽度、块接入 | `decode-working-set-20261009/`、`ffn-input-width-20261009/`、`ffn-batch-scaling-20261009/`、`prefill-batch-scaling-20261009/summary.json` |
| LN预处理与32/32文本 | `layernorm-affine-preparation-20261009/summary.json`、同目录源码前后快照、固定binary与逐次结果 |
| 官方资料 / NBG实测 | `official-knowledge-consolidation-20261012/`：全文阅读清单、原始评估/门禁、MAC审计及清理前文档/hash |
| 矩阵布局与TTGT | `gemm-layout-collection-20261012/summary.json`及各布局源码/identity/结果；失败轮见事故台账 |
| 临时INT8 / 两路fc2 | `int8-and-ffn-partition-20261010/summary.json`、`source/`、`experiment-model.patch`、`run.py`及逐项门禁 |
| FP16压力 / 纯GEMM与FFN消融 | `fp16-compute-scaling-20261010/summary.json`、`gemm-ffn-ablation-20261010/summary.json`及同目录临时源码、夹具与逐项门禁 |
| 精简后短请求 / C128问答 | `post-cleanup-context-20261010/summary.json`、`generated-text.json`、`short-request/`、`context-questions/` |
| 常驻复用、静态审计、SDK分析 | `resident-request-reuse-20261009/`、`decode-static-audit-20261009/`、`ffn-workload-analysis-20261004/`、`decode-offline-20261005/` |
| 退役候选与历史源码 | `decoder-structure-screening-20261001/`、`fc1-shape-decode-20261009/`、`vx-gemm-estimate-20261009/`、`cleanup-20261001/before-source.tar.gz`、`diagnostic-cleanup-20261008/` |
