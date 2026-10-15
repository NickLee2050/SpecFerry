# NP101 推理实现与优化结论

更新：2026-10-15。活跃模型为真实 `facebook/opt-350m`；目标是保持旧实现的正确性，
降低单请求decode延迟。未完成项见[TODO](../TODO.md)，每次上板先读
[事故台账](np101-recovery-incidents.md)，接口见[测试手册](../tests/README.md)，
SDK失败契约见[问题表](np101-sdk-driver-validation-report.md)。

原始证据相对 `.cache/runs/`，不随Git分发；压缩前全文、代码与hash在
`commit-cleanup-20261013/before/`、`before-manifest.json`。早期归档见
`documentation-consolidation-20261010/`；退役脚本保存在压缩证据归档中。

## EP阶段封版

本轮EP功能验收和性能路径探索收尾；后续优化按新需求/证据重开，不要求完成所有
可选TODO才结项。推荐配置固定如下，历史实现保留作回归及诊断，不自动重试故障条件：

| 模型 | 推荐运行配置 | 已验范围 / 当前性能口径 |
|---|---|---|
| OPT-350m | FCL attention投影、融合NBG FFN/末LN、NBG head，M1、prefill仅末词head | C64/C128；C64的3次17输入/16输出为1.96582输出token/s、3.63403首词后decode token/s |
| Qwen3.5-0.8B缓存模型 | GATHER_ND、零基layer9设备MatMul FFN、18 DeltaNet＋23 FFN NBG、mapped选词读回 | C64；23输入/16输出自然句为0.19347输出token/s、0.33876首词后decode token/s |

模型、输入长度与请求数不同，上表不作跨模型速度对照；准备/reset/粗读回不计入生成。
对应包保持原始内容，不移动或重导出；各自manifest绑定NBG、源权重、端口及五库身份：

| 本机已验包（相对`.cache/runs/`） | manifest.json SHA256 |
|---|---|
| `opt-entry-integration-20261014/ffn-head-package` | `ec99e136f8458c301ed4165a582ade84751c468b596fcd3ef5431d0d96cf64fd` |
| `opt-qwen-integration-20261014/qwen-compiled-blocks-package` | `9ccdf15e4c8e783b5e875af8ae03960a918563326a9529c2a57e2cd0d2e7f0be` |

两包均绑定v105 `srcversion=F53A3E6EFBB7D0FD7D4617C`。原始checkpoint/NBG及固定
已验binary继续位于忽略缓存，源码仓库不分发这些产物；manifest/源权重不匹配应拒绝。
本轮将Qwen Python/native入口省略参数时的默认值对齐上述已验组合；显式覆盖仅供诊断。
OPT默认component回归路径不变，推荐混合路径由README命令显式选择。默认策略的收尾
仅做离线请求契约/构建检查，硬件验收和性能沿用固定显式配置的既有记录，没有重新上板。

当前证据覆盖有界生成、状态/隔离、预算/释放和延迟内核检查；长期稳定、Qwen C128、
内部硬件利用率及间歇性hang/EDMA根因未解决。恢复遵守板卡断电确认及完整门禁，
不再强制主机冷启动或boot ID改变。可选head/分块prefill、SoC对照和TLM协议见TODO。
收尾前文档及身份留存`ep-baseline-finalization-20261015/before/`、`before-manifest.json`。

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

## 投影组合、权重复用与Qwen候选

v105同boot，三个相同17-token请求，各生成16个输出，无独立暖机，粗读末层KV；
每个组合均通过数值、重复请求、预算/释放和45秒日志检查。FFN＋末残差/LN及head
固定为NBG，表中仅改变QKV/out_proj。初始化不计入吞吐；这些是本轮匹配结果，
不替换此前不同配置/负载的最高记录。初始化期间存在离线编译/构建活动，数值不作纯Verify归因。

| QKV / out_proj | 初始化 s | 实际输出 token/s | 首词后decode ms/token | TTFT s |
|---|---:|---:|---:|---:|
| FCL / FCL | 336.45 | 1.915 | 280.53 | 4.148 |
| NBG / FCL | 76.74 | 1.870 | 288.09 | 4.234 |
| FCL / NBG | 255.33 | 1.725 | 311.55 | 4.604 |
| NBG / NBG | 3.24 | 1.731 | 309.85 | 4.595 |

**选择**：混合入口仍以FCL投影为默认；NBG组合是显式选项，适合探索准备时间的取舍。
全部NBG大幅缩短准备，decode约慢10.5%。静态图数分别为52/76/100/124（含reset），
图边界与投影实现同时变化，不能将全部差异归因于提交数。应用热路径没有重载权重；
共享句柄和固定绑定不证明SDK内部没有搬运、打包或副本。

多输出导入有两个顺序：NBG节点的SDK端口顺序和公开图的语义顺序。QKV实例前者
为V/K/Q，后者为Q/K/V。最初公开输出直接使用SDK顺序产生整模选词错误；单段QKV
集合正确，保留输入/残差寿命未解决，修正公开顺序后整模48输出与KV通过。
产物记录完整端口排列，原始数值失败及修复证据保留；未产生设备故障或冷启动事件。

Qwen使用已有固定checkpoint，无新下载。4层（3个DeltaNet＋1个GQA）真实参考输入，
四条8步轨迹，计时覆盖完整step集合及边界IO，排除初始化/reset/读回/日志；这些不是生成吞吐。

| 4层组合 | 8步平均 s | const / nonconst峰值 MiB | 已验范围 |
|---|---:|---:|---|
| 原MatMul FFN、两份递归图权重 | 3.460 | 218.8 / 24.6 | 粗读末步各层输出/状态；reset/fresh及45秒通过 |
| FCL FFN、共享不可变应用权重 | 2.241 | 84.1 / 102.5 | 同上；约减少35%序列时间，准备明显更慢 |
| 再将第0层DeltaNet替换为NBG＋固定copy | 1.957 | 104.3 / 77.0 | 同上；相比上一组合再减少约12.7%；其余两个DeltaNet仍手写 |

NBG DeltaNet保留FP32递归/归一化、FP16投影/短卷积边界，无NBG输入输出别名。
先在CPU对照8步旧参考并审计生成C，再做设备轨迹；单份NBG加固定回写减少重复权重
和公开中间缓冲。以上优化复用共用FCL、端口排列、编译隔离和固定存储机制，不复制OPT模型组合。

**完整Qwen的保守组合已通过C64/v105有界验收。** 配置为显式
`--embedding-lookup gather-nd --matmul-feed-forward-layer 9`：共享tied词表、24层、
18份DeltaNet＋23份FFN NBG，零基layer9使用已有设备MatMul FFN，最终RMSNorm及
device head/greedy。保留FP32递归/归一化与FP16投影/SwiGLU边界，无CPU FFN或
is_const改动。该层少计NBG22,024,843B、新增共享普通权重22,022,144B；
实际const/nonconst峰值887,143,543 / 669,684,140B，释放归零。

2026-10-14冷boot `4a5c4f7f-37c9-460c-a973-e14ca18cd81a`：Hi 13输入/1输出，
以及算术26输入/两次各4输出，均与固定CPU oracle一致；粗读末层hidden/KV、
layer22 FP32循环状态、未写槽按位零、两请求状态逐字节一致、预算/释放和各45秒
延迟门禁全部通过。算术问题为“What is 2 + 2? Answer with only the number.”，
两次显示`4\n`，均消费29个位置、遇EOS停止。原恢复标记归档，无新内核故障。

| 已验诊断负载 | 输出文字 | 初始化 s | 每请求推理 wall s | 实际输出 token/s | 首词后decode |
|---|---|---:|---:|---:|---|
| Hi，13输入/1输出，1请求 | `Hello` | 117.506 | 23.076 | 0.04334 | 无间隔，不计算 |
| 算术，26输入/各4输出，2请求 | `4\n` / `4\n` | 122.308 | 57.652 / 57.612 | 0.06941 | 6个间隔，3175.08ms/token，0.31495 token/s |

输出计数包含特殊token ID，算术共8个输出ID；不将输入处理或可见文字字符计为生成
吞吐。生成wall排除初始化、reset和粗读回；算术平均TTFT48.106s，prefill46.725s。
以上均开启逐SDK调用及进度记录，是有界正确性诊断耗时；未量化扰动，不能替代普通
模式的持续性能基线。两进程完成1,263 / 5,642次RunGraph，审计309项extension完整；
各237 / 507次Verify含运行期KV改绑重验，未复现已知1024槽缺项。

**此前失败边界仍未解释**：直接FCL FFN曾在layer18创建7MiB常量时正常拒绝；
24份编译FFN版本曾在prefill粗窗口hang。后续独立冷boot分别遇首个embedding
GATHER Verify hang，以及首token零基layer9编译FFN第41次RunGraph hang；后一事件
首次故障前没有运行期KV重验。该FFN生成C含三FCL及norm/Swish/残差，不能指定
内部致错算子或与旧FCL Verify主机崩溃合并。真实输入的源ONNX误差0.0001221通过，
原NBG在隔离模拟器Setup拒绝，不算NBG数值验收。独立事故、原证据及各次冷boot
边界保留在台账；新组合和boot同时改变，本次通过只证明绕行组合的已验范围。

GATHER_ND保持共享FP16词表和INT32 `[1,1]`索引，离线逐位语义及本次硬件请求
均通过；同槽KV写避免重复改绑，非首槽仍有重验，不切默认或外推长期稳定/C128。
此前165次初始化Verify累计78.358s，其中单个547-operation head为76.927s
（98.17%）；即使消去其他Verify也仅省1.431s，仍不优先并行Verify。
10-14完整证据及固定binary/source身份在`qwen-matmul-fallback-coldboot-20261014/summary.json`；
历史失败证据分别在`opt-qwen-integration-20261014/`、`qwen-coldboot-diagnostic-20261014/`
及`qwen-gather-nd-coldboot-20261014/`。

10-14后续关闭诊断对照曾发生**4字节EDMA超时**，当时停测并保留标记；
正确的部分输出不算该轮性能/健康验收。驱动反汇编确认SG分支可主动将日志dst字段
设为全1、实际提交另一张地址表，不能仅凭该值断言非法地址或DMA映射失败；
活动分支/回调状态未采到，超时根因仍未确定，独立时间线见台账。

**10-15映射读回组合通过新一轮完整验收。** 显式`--token-readback mapped`仅替换
选中INT32 token的读取，复用Map/Unmap，不回退SDK-copy；本轮硬件验收时默认仍为sdk，封版入口现已对齐mapped。
其余GATHER_ND、零基layer9设备MatMul、41份NBG、权重/精度/预算均保持上述保守组合。
Hi、三轮各两次算术、单次自然句共5进程/8请求/41输出ID，tokens/粗hidden/recurrent/KV、
未写槽、适用的重复状态逐字节一致、预算/释放及各45秒门禁全部通过；无新设备故障，
旧标记归档。仅证明该C64/v105有界负载；不能认定间歇性EDMA已修复或外推C128。
用户更新恢复规则：板卡断电确认即可，不强制服务器冷启动或boot改变；完整日志保留，
同boot仅豁免上次已记录的带时间戳故障，新故障仍停测。本次实际host boot发生改变，
不把本次成功当作同host boot恢复的实验结论。

| 10-15负载 / SDK逐调用记录 | 输入/每请求输出ID | 请求数 | 平均生成 wall s | 实际输出 token/s | 首词后decode ms/token |
|---|---:|---:|---:|---:|---:|
| 算术，关闭前置 | 26/4 | 2 | 51.977 | 0.07696 | 2953.32 |
| 算术，开启 | 26/4 | 2 | 54.096 | 0.07394 | 3044.55 |
| 算术，关闭后置 | 26/4 | 2 | 51.830 | 0.07718 | 2930.95 |
| 自然邀请句，关闭 | 23/16 | 1 | 82.701 | 0.19347 | 2951.97 |

同负载关闭轮合并4请求/16输出ID，平均wall51.904s，实际输出0.07707 token/s；
12个首词后间隔平均2942.13ms，即0.33989 token/s。开启记录比关闭均值增加生成wall
4.22%、decode3.48%；后置关闭比前置生成−0.28%、decode−0.76%。各组仅两请求，
是有界扰动估计；Verify审计在三轮均开启，不代表完全无诊断的生产性能。
初始化、reset、粗读及文本展示不计入生成wall；token数包含生成的特殊ID，不均摊输入。
与10-14的跨boot差异不能全部归因mapped。Hi生成23.093s，初始化123.686s另计。

新开启轮按固定API语法覆盖全部5,642次RunGraph、6个首词后间隔，无额外设备探针。
下表是SDK外层调用墙钟，含执行/内部搬运/完成等待，不是硬件kernel利用率：

| Qwen decode段 | ms/token | decode墙钟占比 |
|---|---:|---:|
| 共享分块MatMul LM head＋greedy | 1356.30 | 44.55% |
| 18层编译DeltaNet | 621.93 | 20.43% |
| 23层编译FFN | 296.13 | 9.73% |
| 6层attention投影/RoPE | 183.51 | 6.03% |
| 24层input RMSNorm | 171.34 | 5.63% |
| 循环状态固定回写 | 109.05 | 3.58% |
| KV重Verify | 56.56 | 1.86% |
| 单层MatMul FFN回退 | 56.28 | 1.85% |
| selected-token显式Map＋Unmap | 0.0061 | 0.0002% |

优先检查head及DeltaNet；仅消除KV重验的理论上限约1.02倍。每槽固定copy图会增加
378张常驻图/756个copy节点，资源和workspace未验，因此未施工。可复用OPT rank2
indexed append，但需将GQA backing改为`[512,64]`，显式reshape/permute恢复reader
`[256,64,2]`及写后依赖；旧rank3 append失败契约继续适用，不自动切换。
自然句输出与固定CPU参考的16个ID一致：
`Happy Saturday, and I'd love to host a nice dinner with you tomorrow!`
输入为“Write one friendly sentence inviting a neighbor to dinner tomorrow.”；
Saturday非输入指定，CPU也如此，不把回答质量当设备正确性。TTFT38.421s，
prefill37.077s，15个decode间隔、0.33876 token/s；消费38个位置，max_new_tokens停止。

本轮原始身份/确认、每请求输出/性能、完整日志和归因集中在
`qwen-mapped-recovery-20261015/{summary,performance-comparison,decode-attribution}.json`。
此前失败、复用归因和离线KV/OPT驻留预算证据保留`qwen-decode-night-20261014/`；
未执行新的profiler、容量矩阵或同boot故障筛选。

全图/逐decoder的C64/C128源绑定导出、包和显式CLI已实现。C128全图首版ArgMax
修正版仅在模拟器Verify后重存，v105导入返回`VX_ERROR_NO_RESOURCES`；未Setup/Verify/Run。
正常释放及45秒无设备故障。改为官方生成程序的准备及一次模拟器Run后，
文件由578,102,016增至636,700,160B；用户授权的最小设备导入正常返回，进程0.42秒、
102端口及45秒无故障。重存后新NBG采用生成图的公开IO顺序，不能沿用原NBG节点
的端口排列；修正导出契约后复核原查询证据，不重跑导入。
随后C128完整数值/执行/释放/45秒验收通过，旧导入标记归档；Run、输入初始化和其他生成行为同时变化，
不把Run单独当作此前拒绝的根因。旧已验C64全图保持原有验收范围。

C128性能对照（v105同boot、97/3/97输入、每请求32输出；初始化、reset、诊断读回
和CPU参考均不计入生成时间；M1逐token处理prefill）：

| 实现 | 初始化 s | 97输入的两次请求 wall s | 长请求实际输出 token/s | 3输入的请求 wall s / 输出 token/s | 首词后decode ms/token |
|---|---:|---|---:|---|---:|
| 逐decoder NBG＋旧MatMul head | 3.654 | 119.767 / 119.711 | 0.2672 | 31.842 / 1.0050 | 935.68 |
| 新全图NBG，含head/选词 | 1.181 | 110.355 / 110.386 | 0.2899 | 29.326 / 1.0912 | 862.09 |

decode汇总各93个首词后间隔；新全图约减少7.9%间隔时间，长请求输出吞吐增加约8.5%。
不把长短混合汇总的0.3839 token/s当成同一长度负载吞吐；不与不同C64/混合block16
负载直接比较或替换历史最高值。两段payload峰值分别713,919,130 / 51,481,420 B，
释放归零。证据`corrected-fullgraph128-{import,acceptance}/`；NBG内部仍是不透明边界，
外层审计和SDK运行不提供所有物理核的利用率或逐命令硬件时间。

本轮证据：`opt-qwen-integration-20261014/`。Qwen轨迹报告器曾传错evidence类型，
SDK已正常完成；按原始输出补全数值/释放/编译/45秒及额外日志检查，没有重跑或要求冷启动。
千字中文样例的实际分词：OPT 2174、Qwen 569 token；本轮258字符请求正文为442/140，
Qwen加入聊天模板为152。不是整段聊天计数；长历史需累加所有保留轮次，C128仍不足以覆盖日常长对话。

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

### 直接FCL投影的匹配验收

真实教师输入、相同FP16权重/边界；QKV/out各8层不同权重，head含project_out、
完整词表及greedy。2暖机＋3×16集合，排除准备/读回，NBG选词图计入。
MatMul取前后均值，漂移分别−0.25%/+0.44%/+0.29%；均为局部回放时间。

| 计算段 | MatMul ms | 直接FCL ms | NBG ms | FCL局部加速 |
|---|---:|---:|---:|---:|
| 每层三路QKV，含Q缩放 | 12.028 | 2.156 | 2.154 | 5.58倍 |
| 每层out_proj | 4.055 | 0.751 | 0.736 | 5.40倍 |
| head＋greedy | 103.556 | 40.502 | 40.806 | 2.56倍 |

FCL只绑定输入/静态FP16 `[K,N]` 权重，第三bias槽NA；保持SDK参数/local初始化，
SATURATE/NEAREST_EVEN/FLOOR，bias独立Add保留FP16边界。原三输入FCL故障未解释；
kernel名不证明ReLU融合或物理利用率。head还取消分块/简化归约，不能单归因于FCL。
词号50118及完整logits通过；NBG公开输出需按端口名校准，不沿用SDK节点顺序。

整模v105/C64/M1、17输入/16输出、1暖机＋1正式；同prompt复用、logits/末层KV、
未写槽、预算/释放/45秒通过，两版均75图，固定copy替代48次清零上传。

| 匹配整模 | 请求输出 token/s | 首词后decode token/s | decode ms/token | TTFT s | Verify合计 s |
|---|---:|---:|---:|---:|---:|
| MatMul投影＋NBG FFN | 0.764818 | 1.527563 | 654.637 | 11.100 | 2.913 |
| FCL投影＋NBG FFN | 1.751660 | 3.516706 | 284.357 | 4.869 | 343.751 |

生成wall20.920→9.134s，约2.29倍；初始化Verify未计入。operation1645→1532，
extension375→238，const/nonconst峰值656,759,530/79,256,528B，释放归零。
基线JSON漏括号按原输出离线修正，未重跑。证据
`opt-projection-replacement-20261013/{collection-summary,summary,model-plan}.json`。

### attention与归一化的匹配替换

真实前8层教师输入，手写→NBG→手写；2暖机＋3×16集合、各45秒门禁通过。
准备/IO/读回不计时；M16是已知输入块，局部回放不是生成吞吐。

| 计算段 | 手写前后均值 ms | NBG ms | 延迟变化 | 基线漂移 | 决策 |
|---|---:|---:|---:|---:|---|
| M1 attention核心 | 0.653 | 0.687 | +5.11% | +1.96% | 保留手写batched核心 |
| 每层两种残差/LN | 1.743 | 1.586 | −9.04% | +1.12% | 不单独增加图 |
| FFN＋手写末残差/LN → 融合NBG | 3.855 | 3.733 | −3.17% | +0.74% | 保留融合 |
| M16完整FFN | 56.033 | 10.893 | −80.56%（5.14倍） | −0.23% | 核实已有编译路径 |

最大误差attention0.000977、其余0.007813。attention保留FP32 softmax、FP16 Q/K/V、
INT32长度，长度1/17/64及非零未写槽屏蔽另有软件检查。残差/LN保留FP16 Add、
FP32归一化/affine和epsilon1e-5。FFN融合的TP命令数未减少；边界/交接收益尚无硬件归因。

融合整模75→51图、operation1532→1244、extension238；契约增加两份final norm源hash。
C64/M1、17输入/16输出、1暖机＋1正式的选词/logits/末层KV/未写槽/释放/45秒通过。
前置是同boot此前验收，后置使用相同binary/SDK重新采样，不称紧邻新前置。

| 整模 | 请求输出 token/s | 首词后decode token/s | decode ms/token | TTFT s | Verify合计 s |
|---|---:|---:|---:|---:|---:|
| 原FCL投影＋NBG FFN，前置 | 1.751660 | 3.516706 | 284.357 | 4.869 | 343.751 |
| FCL投影＋融合FFN/末LN | 1.862587 | 3.708303 | 269.665 | 4.545 | 316.442 |
| 原实现，后置 | 1.750127 | 3.478811 | 287.455 | 4.830 | 338.526 |

输出率增6.38%、decode延迟少5.68%；基线输出率/decode漂移−0.088%/+1.09%。
const/nonconst峰值657,076,642/78,371,408B；logits误差0.03125、末层K/V
0.0078125/0.0087891，均释放归零。单条正式请求不外推长期或其他形状。

M16生成代码观察到8次正向及8次反向通道的1×1卷积映射，FP16、空间16×1、
stride1、padding/dilation0、ReLU/pooling关闭；未读取权重物理布局。匹配集合按两次
GEMM运算/完整FFN时间计为4.79→24.64 GFLOPS；布局/打包/调度同时改变，不能将
5.14倍独立归因卷积，不恢复退役卷积测试，也不混入另一TTGT集合的4.067ms。
`float`策略会覆盖敏感FP32运算，采用float32并核对生成C；原始及sealed wrapper
声明优先于误报的metadata。节点C用network.export.data，不用model.data；模拟器NBG
Setup拒绝不等于硬件结论。证据`opt-core-replacement-20261013/`的collection/model摘要、
静态命令审计及geometry-only记录；临时程序仅保留归档。

### 正式产物入口与较长上下文

融合FFN已有checkpoint→ONNX→隔离编译器→NBG→正式包的流程，24层各做CPU精度及
生成C/端口/版本审计，新NBG与此前硬件验收的文件逐字节一致。运行包仅含NBG和两份
manifest，绑定源权重、checkpoint/index、SDK及图版本；Python和native均在创建Context前
拒绝损坏产物。`generate_opt.py --backend graph --compiled-ffn`显式加载，默认实现不变；
固定M1，接受已验C64/C128。可选head包复用已验project_out/词表NBG，另接INT32
ArgMax图，不暴露中间projected输出。prefill仅末词head仍更新每个输入的全部decoder/KV。

v105同boot的正式入口各串行3次相同17输入/16输出，选词（包含50118）、最终logits、
释放/预算及45秒检查通过；共6正式请求96输出。没有单独暖机，均计入初次请求；
首组初始化期间有并行主机离线编译，不能把准备差值全部归因于head。

| C64正式入口，均仅末词head | 实际输出 token/s | 首词后decode token/s | decode ms/token | TTFT s | 初始化墙钟 s |
|---|---:|---:|---:|---:|---:|
| FCL投影/head＋融合FFN | 1.940457 | 3.608712 | 277.108 | 4.089 | 388.965 |
| FCL attention投影＋NBG head/融合FFN | **1.965820** | 3.634035 | 275.176 | 4.011 | 285.290 |

后一组48输出/24.417s为当前最高实测请求输出率；历史单请求3.708 token/s仍是较高的
首词后decode记录，两者口径不能互换。与此前每词head、1暖机/1正式的1.863结果不是
严格匹配的改进百分比。图准备和诊断读回未计入生成；NBG head减少一份直接FCL构造，
未证明其内部执行更快。原局部集合的Verify API累计：QKV八层FCL/NBG为63.413s/4.761ms，
out八层21.511s/2.693ms，head含选词72.884s/3.989ms。这些是集合准备证据，不是整模预测。

新混合候选C128用已验97输入/32输出，1暖机＋1正式：全部选词/最终logits、末层
FP16 KV、完整末槽、越界step零IO拒绝、52图/1243 operation/238 extension、预算/释放
和45秒门禁通过。32输出/34.779s＝0.920104 token/s；TTFT25.452s，首词后decode
3.323726 token/s（300.867ms）。const/nonconst峰值657,083,173/98,098,256B，均释放归零；
末层K/V最大误差0.011719/0.010986。随后正式CLI同Context做97/3/97输入，各32输出，
28项选词/KV/释放检查通过；short未写KV的385,024B按位零，两次long末层KV逐字节一致。
96输出/81.142s，属于长短混合请求，不与固定17输入的最高输出率混算。
未重做全层KV或跨boot；
不把该证据外推全图/逐decoder的FP32 KV路径。长prompt逐词prefill仍是明显开销。

两组C64 decode主机进程全部线程user＋system占对应墙钟2.469%/2.356%，不含初始化；
只排查主机CPU，不能说明板载CPU是否参与或NPU利用率。相同请求的有界复用已通过，
跨启动及长期常驻仍待验。完整全图清理后的固定粗验亦通过，16输出/17.369s＝0.921206，
decode542.877ms，预算/释放和延迟门禁正常，不重复此前布局矩阵。

M1融合FFN＋24层M16 FFN＋FCL QKV/out＋一份head的已知const下界为1,111,016,984B，
超过1GiB约35.548MiB；现有构造若各建一份head为1,163,544,088B，超约85.642MiB。
TTGT M16额外7,680B。下界不含小常量/未知SDK workspace；不假设两份NBG内部共享。
因此尚未接block16，需解决实际驻留/复用，不能直接放行超预算模型。
证据统一为`opt-entry-integration-20261014/`的entry/rebuild/budget/host-cpu摘要、
fullgraph/entry/head/context128/reuse验收及固定source/SDK身份；本轮5板卡进程无新设备故障。

后续离线按当前M1＋一份NBG head及真实M16逐层manifest重新核算：全部M16 NBG仍超
const约35.55MiB。前2/3/4层M16改用已有普通权重FFN，已知const余量分别约
0.53/18.57/36.61MiB（TTGT近似相同）；2层仅满足数学下界，不是安全峰值。
选择前4层的保守方案，保留全部M1 FCL及20层M16 NBG，单份head经固定输入桥接复用，
不改is_const。已知nonconst下界约330.69MiB，不含新增激活/临时alias，SDK workspace未知。
32MiB规划余量是工程选择，不是SDK空间保证；尚未修改运行时、创建Context或上板。
后续需拆开当前FCL只允许M1的门禁、支持显式部分M16 FFN、落实单head所有权及分段
峰值守门，再做最小验收。证据`qwen-decode-night-20261014/offline-opt-residency.json`。

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
推理暂缓。对NP101方法的适用性以本项目证据为准，不搬用外部加速比。

设备当前状态只由[事故台账](np101-recovery-incidents.md)及实时恢复标记判定；
历史成功不放行新运行。operation缺项、AddTensor拒绝、EDMA和关机page fault是不同
API事件，不能用同一层号解释；独立原始证据和复发规避继续保留。整理前全文/源码
身份在`commit-preparation-20261014/before-manifest.json`，完成实验的专用入口已退役。
