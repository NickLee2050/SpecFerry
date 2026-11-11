# NP101实现与验收状态

更新：2026-11-10。这里仅保留运行契约、代表性结果和影响下一步决策的结论。
使用步骤见[README](../README.md)，代码边界见[开发指南](development.md)，
未完成项见[TODO](../TODO.md)，上板前遵守[事故台账](np101-recovery-incidents.md)。

历史日志、权重和编译产物位于忽略目录`.cache/runs/`，下文证据路径均相对于该目录。
这些文件不随源码分发，不能仅凭路径在另一台机器复现实验。精简前完整技术记录可从
提交`23b7530b0c66b95203feb97424cf4930aa14a71b`读取，本机快照位于
`.cache/maintenance/pr-cleanup-20261110/`。故障台账保留原始时间线，不因文档整理删改。

## 性能口径

- 实际输出吞吐＝实际生成token数／生成墙钟；包含prefill与必要token IO。
- TTFT＝请求开始到第一个输出；首词后decode率使用其余输出数及对应时间。
- 准备、Setup/Verify、reset、粗粒度快照和日志独立报告，不能计入算子时间或冒充生成收益。
- `M`是一次矩阵运算的token行数，`C`是上下文容量。M16处理16个已知输入，不同时生成
  16个未知token。prompt处理率不是输出吞吐。
- 图调用的完成等待包含未完成计算；不能仅凭ioctl占比认定IO瓶颈。新追踪用同负载
  关闭→开启→关闭量化扰动。历史、不同请求或不同启动的对照必须标明，不能混算。

## EP阶段封版

本轮EP功能验收和性能路径探索收尾；后续优化按新需求/证据重开，不要求完成所有
可选TODO才结项。推荐配置固定如下，历史实现保留作回归及诊断，不自动重试故障条件：

| 模型 | 推荐运行配置 | 已验范围 / 当前性能口径 |
|---|---|---|
| OPT-350m当前NBG组合 | QKV/out、M1融合FFN/末LN及单head用NBG；可选M16 prefill，保留手写attention；选词映射读回 | C64及有界C256；C64的17输入/16输出，M1/M16为1.21641/1.95893输出token/s，首词后decode为2.35418/2.34884 token/s；C256及计数范围见正文 |
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

## SoC部署与内存

当前Qwen板端为768 MiB配置Linux、2048 MiB SYSTEM池：Linux System RAM
`0x40200000–0x6fffffff`，驱动`contiguousBase=0x70000000`、`contiguousSize=0x80000000`，
池止于`0xefffffff`。Linux实际MemTotal约672.4 MiB，不等于配置的完整物理窗口。
板端`/np101_driver/go.sh`需显式加载；当前不自动加载，不应把重启当作已执行该脚本。

| 区域/限制 | 含义 |
|---|---|
| SYSTEM | 驱动预留的连续NPU池，容纳应用tensor及SDK空间 |
| VIRTUAL | 来自Linux内存，不能与预留SYSTEM容量重复相加 |
| EXTERNAL/EXCLUSIVE | 未取得可用于当前模型的已验配置，不按名称等同const/nonconst |
| 应用const/nonconst | 各限1 GiB，按共享backing去重；两类共同占物理池，不是两个独立物理区 |
| 保留区/未登记地址 | 未确认归属就不能并入NPU池；逻辑错误偏移不是物理地址 |

Qwen新部署匹配以上布局；OPT打包默认仍固定历史512 MiB Linux/2304 MiB池，不能直接
用于当前Qwen布局。旧已验bundle保持原参，不修改manifest假装兼容。缩至512 MiB后，
C512长请求在2304 MiB池连续hang，恢复1792 MiB池仍失败，尚未证明Linux容量或地址是根因。

部署校验载荷哈希；每次启动核验receipt、文件元数据、端口、源权重、预算和SDK身份。
准备≤240 s，OPT正式推理≤300 s，Qwen默认300 s、可显式选600 s；不能延长运行中期限。
SoC退出观察10 s，下次再读完整内核。成功输出不代替资源释放和内核检查。
仅孤立SoC NP hang可按台账有界重试，连续三次同实验失败即停；SDHCI/EXT4/Oops不在例外内。

## 保留的SoC模型路径

| 模型/路径 | 内容与边界 |
|---|---|
| OPT保守路径 | FP16 KV、无Verify缓存，M16前四层现场卷积FFN，其余共享FCL/编译decoder；C256长短长和原布局C512短长曾通过，不表示当前布局长请求或长期稳定已验 |
| Qwen M1 | 18段DeltaNet NBG，六层手写attention，固定权重FFN；共享FCL QKV/out及第10层FFN、共享完整词表head，保留MatMul回退 |
| Qwen批量FFN | 全24层一套M8或M16 NBG，decode填零复用；映射KV追加、attention输入norm合图、FFN内状态提交可显式选择；不新增M1权重副本 |
| 两层M8 Delta候选 | 共享M1 next-state输出，单请求已验；尚未进入正式入口、尚未完成该候选的两轮/reset验收 |

Qwen完整C128的96输入/16输出：共享FCL原路径TTFT100.293 s、实际0.1328 token/s；
映射KV后同负载对照TTFT100.201→90.281 s、实际0.132680→0.146514 token/s，运行期
KV重验为零。后续合并Delta状态读取、批量FFN及归一化合图的收益见下表。

| Qwen配置 | 请求 | TTFT | 实际输出吞吐 | 范围 |
|---|---|---:|---:|---|
| 合并Delta状态读取，M1 FFN | 96→16 | 82.749 s | 0.159380 token/s | 对历史映射KV基线，非同boot A/B |
| M8 FFN | 96→16 | 77.413 s | 0.168200 token/s | 单套权重，decode基本不变 |
| M16 FFN | 96→16 | 76.565 s | 0.169778 token/s | 比M8约0.9%，不认定稳健优势 |
| M8＋attention归一化合图 | 96→16 | 71.993 s | 0.180081 token/s | 同boot基线77.557 s、0.167861 token/s |
| 再合并FFN状态提交 | 96→16 | 70.069 s | 0.184747 token/s | 同boot原提交73.081 s、0.177784 token/s |
| 上述M8组合，两轮清KV重放 | 28→20、66→49 | 均值34.661 s | 合计0.480163 token/s | 69输出/143.701 s；两轮EOS，覆盖尾块/reset |
| 额外两层M8 Delta，独立候选 | 96→16 | 66.340 s | 0.192707 token/s | 首词后0.898862 token/s，未验整模多轮 |

C512沿用全24层M8 FFN、attention norm合图及FFN状态提交，不重编NBG或增加权重副本。
同boot的两次有界回归均通过原数值阈值、未写KV零检查、释放及完整内核/10秒观察：

| Qwen容量与输入→输出 | 准备 | TTFT | 实际输出吞吐 | 首词后decode | SYSTEM采样峰值 |
|---|---:|---:|---:|---:|---:|
| C512，96→16 | 148.127 s | 70.261 s | 0.184288 token/s | 0.905794 token/s | 1612.617 MiB |
| C512，241→16 | 146.059 s | 175.641 s | 0.083262 token/s | 0.907773 token/s | 1612.617 MiB |

对本页末尾同配置C128的96→16，扩大容量使TTFT增加约0.74%、实际吞吐降低约0.52%，
单次差异不判为稳健退化。实际池峰值增加7.781 MiB；KV本身由1.5增至6 MiB，
长请求Linux最低可用282.012 MiB。输入96→241时TTFT约2.50倍，首词后decode基本不变；
当前长请求主要受逐token prefill限制。241→16消费256个位置，不代表512满槽或长期稳定性
通过。证据`qwen-context-512-20261110/comparison.json`，长请求CPU参考与短请求上板并行。

两轮输出与CPU一致，但第二轮漏答米饭问题属于模型质量限制；不把数值验收当回答质量或
推测接收率。以上使用原pipeline容差，M4/M8 Delta严格FP32状态差异留证，不放宽阈值。
M8整模SYSTEM峰值约1604.836 MiB；两层Delta候选运行期增加约55.8 MiB。直接扩18层
M8 Delta预计const1347.632 MiB，超过1 GiB应用限制；物理运行期估算2043 MiB也过于紧张，
未做全18层容量测试。共享状态输出不是共享权重。

**图边界经验**：将状态复制源/目的追加为FFN NBG包装图外部端口曾导致hidden含Inf和
选词全零；保持原两输入/一输出、内部固定绑定复制后通过。图顺序、生命周期、端口排列
均是执行契约，不能为了“清理”而重建图或改端口。原始失败和修复见
`qwen-ffn-state-diagnosis-20261110/`，两轮见`qwen-block-progress-20261110/`，两层批量
候选见`qwen-shared-state-output-20261110/`。

## 当前瓶颈与候选结论

**当前整模大头与容量归因（11-10）**：复用上述两层M8 Delta＋全层M8 FFN候选，
96输入/16输出，低开销内存聚合计时，关闭→开启→关闭；准备、输出快照和日志不计入推理。
关闭端生成83.028/83.262 s，开启83.309 s，相对两端均值增加0.198%，小于关闭端
0.282%的跨度；token及四份hidden/状态/KV快照逐位一致，执行、释放、全内核及10秒检查通过。
计时覆盖SDK图调用及外层IO，不是设备流水计数器，不能把RunGraph时间全解释为搬运或等待。

| 区段 | 首词前66.465 s | 首词后15步16.844 s |
|---|---:|---:|
| 剩余16层逐token Delta | 33.434 s（50.30%，1536次） | 全18层6.023 s（35.76%，270次） |
| 独立状态复制 | 14.873 s（22.38%，1344次） | 已包含于FFN包装图 |
| 两层M8 Delta | 2.630 s（3.96%，24次） | 不使用 |
| attention输入投影/RoPE | 5.478 s（8.24%） | 0.910 s（5.40%） |
| attention读取/输出投影 | 5.461 s（8.22%） | 0.837 s（4.97%） |
| FFN包装图，含块末/单步状态提交 | 2.795 s（4.21%） | 3.760 s（22.32%） |
| LM head | 0.320 s（0.48%） | 4.811 s（28.56%） |
| 显式decoder IO＋mapped KV | 0.370 s（0.56%） | 0.163 s（0.97%） |

其余为embedding、最终norm及未覆盖的小段；未覆盖量为0.155/0.037 s。
不能把FFN包装图的22.32%全归于矩阵计算。独立状态提交每次逻辑复制1.047 MiB，
平均11.07 ms；累计约1407 MiB，但内部实际访存量未知。首词前72.68%集中在尚未批量化
的Delta和独立状态提交；继续优化仅占4.21%的FFN不是当前主线。仅消除独立提交且其它
不变的数学上限也只是TTFT减少22.38%，不是端到端数倍提升。

初始化所有权快照：应用驻留const976.283 MiB、nonconst611.059 MiB，共1587.342 MiB。
权重/NBG常量账本约1545.288 MiB：共享embedding/head485、attention原权重84.006、
18段M1 Delta NBG362.921、24段M8 FFN NBG566.753、两段额外M8 Delta46.419，
其它const0.191 MiB。递归双缓冲36、卷积历史双缓冲1.688、C128 attention KV1.5、
其它ordinary2.865 MiB。NBG常量记账保守覆盖文件/来源权重，不是内部逐字节物理归因；
SDK workspace、内部副本另由SYSTEM采样约束。KV容量目前不是驻留大头，重复Delta权重
才直接限制批量图扩层；共享head没有第二份485 MiB词表权重。

**共享投影与decode候选（11-10）**：恢复控制后串行测量，所有最终验收均释放资源，
全内核及10秒检查无新故障。正式默认不变；初期两次Verify拒绝和head末尾控制超时的
原始失败保留，后续修订/补测不改写原记录。

| 单层候选 | prefill集合变化 | M1接续集合变化 | SYSTEM驻留 | 决定 |
|---|---:|---:|---:|---|
| 共享输入/输出投影，拆出M1/M8递归图 | 1.497→1.662 s | 0.524→0.712 s | 51.701→34.273 MiB | 省容量，性能回退 |
| 再将输出投影与状态提交合图 | 1.496→1.579 s | 0.523→0.594 s | 51.701→34.277 MiB | 收回部分提交成本，尚不整合 |
| CPU映射复制状态 | 3.136→4.399 s | 0.531→0.754 s | 均51.576 MiB | 不采用 |

以上为96行prefill接16次M1的集合；前两项基线prefill用M8，CPU复制基线全用M1，
不能混用基线或称作输出吞吐。候选均通过原pipeline容差；共享投影接M1后严格状态检查
有2个元素超差，最大0.0004578，未放宽阈值；CPU复制也过严格阈值。
共享后单层常量43.355→23.392 MiB，全18层整模const约987.993 MiB仅为外推，
实际峰值及完整模型未测。投影/状态合图保持原NBG对外端口，仅内部绑定DATACONVERT；
相对独立提交拆图M1快约16.5%，但仍比原M1慢13.7%，不能据此证明设备并行。

**FFN边际成本**：相同集合重复32遍，上传、读回和日志不计入图时间。隐藏输出逐位一致，
内提交与独立提交的最终状态逐位一致；仅FFN不更新状态，只能用于归因。

| 路径 | 384次prefill图时间 | 512次decode图时间 | SYSTEM驻留 |
|---|---:|---:|---:|
| FFN内含状态提交，前后均值 | 3.747 s | 5.068 s | 28.514 MiB |
| 仅FFN | 3.721 s | 4.924 s | 26.418 MiB |
| FFN＋独立状态提交 | 7.590 s | 10.603 s | 26.418 MiB |

去掉内提交的decode差异2.85%小于控制跨度4.26%；独立copy约10 ms/次不能简单加到
合图时间。当前FFN已摊薄提交成本，优先处理prefill尚独立提交的部分。

**head分组**：每图词表块8→16，不复制权重。16次选词集合5.103→4.787 s，完整logits
逐位一致。随后同boot、96输入/16输出整模通过原数值/状态/KV验收：

| 指标 | 每图8块 | 每图16块 |
|---|---:|---:|
| TTFT / 完整生成 | 66.948 / 83.768 s | 66.582 / 82.882 s |
| 实际输出 / 首词后decode | 0.1910 / 0.8918 token/s | 0.1930 / 0.9202 token/s |
| 准备时间 | 149.744 s | 213.177 s |
| SYSTEM全程采样峰值 / 就绪后峰值 | 1604.836 / 1596.727 MiB | 1652.623 / 1596.719 MiB |

仅一对整模样本，实际吞吐约增1.07%，不宣布稳定优势；额外准备约63 s、初始化峰值
约48 MiB，暂不采用。后续集中于共享Delta投影与FFN联合组织及扩大批量覆盖，暂停零碎
decode候选。证据在`qwen-shared-projection-resume-20261110/`：`comparison.json`、
`ffn-cost-comparison.json`、`head-model-comparison.json`、原始运行及源码身份；首次归因和
离线产物在`qwen-performance-capacity-20261110/`。临时工具退役，测量源码归档留证。

## 不再推进的路线与接口限制

| 路线 | 结论/重开条件 |
|---|---|
| FFN Verify缓存 | 只减少准备；长短请求及历史成功复测不稳定，已退役。旧包只能用原runner/binary |
| 共享残差块、省中间发布 | 约0.17%整模变化，无明确收益；释放曾报SDHCI，未完成健康验收。本次从源码/CLI退役 |
| TTGT | 收益依尺寸/选路；Qwen手写MatMul明显慢于共享FCL，编译器可消去等价转置，不继续泛化尺寸矩阵 |
| 动态外置FFN权重 | Conv/FCL端口不等于真实动态系数语义；出现NaN，MatMul绕过正确但慢。等待新契约/IR |
| CPU复制、固定行复制、更多小图 | 无明确收益，部分有数值差异；已留证，不保留正式实验模式 |
| 并行Verify | 未验证线程安全；多个Context仍共享v105进程shader槽表，不能靠Context隔离1024槽限制 |
| 私有Verify插桩 | 绑定x86特定库内存偏移，现已退役；保留公共API计时与独立生命周期/数值验收 |
| profiler/VPD | SoC可采集，但旧整模扰动约127–148倍，缺少可信流水时序。计数器单位/含义待确认 |

导入NBG需要SDK的public/内部端口排列一致；不能只按ONNX顺序绑定。
FCL和MatMul可能选择不同硬件路径，数学等价不代表性能/精度等价；SoC和EP的最优组合
也不相同。`flush`/完成信号包含设备执行，静态反汇编只能给调用边界，不能证明等待根因。
FP32递归状态必须保留；FP16 ADD会清零非正规数，不能用算术伪装位复制。
底层IR未就绪，不提前引入MLIR、自定义ISA或全新kernel框架。

## 历史证据入口

以下保留来源索引；详细过程在原始数据及精简前Git版本，不是可用当前代码重放的命令。

| 主题 | `.cache/runs/`证据目录 |
|---|---|
| EP结项、输出吞吐口径 | `ep-baseline-finalization-20261015/`、`output-throughput-review-20261009/` |
| OPT NBG与M16 | `opt-fast-prefill-20261016/`、`opt-entry-integration-20261014/` |
| Qwen合并状态读取、两轮 | `qwen-delta-joint-model-20261109/`、`qwen-prefill-resume-20261109/` |
| Qwen全层批量FFN | `qwen-ffn-block-model-20261109/` |
| 图组织、状态提交 | `qwen-ffn-state-commit-20261110/`、`qwen-ffn-state-diagnosis-20261110/` |
| 批量状态共享 | `qwen-block-progress-20261110/`、`qwen-shared-state-output-20261110/` |
| 主要耗时、容量 | `qwen-performance-capacity-20261110/` |
| 共享投影、CPU复制、head和FFN归因 | `qwen-shared-projection-resume-20261110/`，含`diagnostic-sources.tar.gz` |

## 当前交付边界

11-10源码整理后，两套重建程序均通过一次SoC整模回归，复用原NBG、请求和CPU参考。
两者数值、应用/池释放、完整内核和10秒观察通过，无新增设备故障；历史恢复标记保留。

| 配置与输入→输出 | 准备 | TTFT | 实际输出吞吐 | 首词后decode | SYSTEM采样峰值 |
|---|---:|---:|---:|---:|---:|
| Qwen C128，M8 FFN＋norm合图＋FFN状态提交，96→16 | 145.394 s | 69.745 s | 0.185252 token/s | 0.902308 token/s | 1604.836 MiB |
| OPT C256，原四层无缓存M16 prefill，17→16 | 149.937 s | 1.687 s | 2.970294 token/s | 4.054028 token/s | 1447.515 MiB |

性能与对应历史配置接近，本轮不主张新增性能收益。EP/SoC构建、142项Python和4项CPU
测试通过；合并后的NBG清单生成与原EP/SoC清单逐字一致。证据
`pr-cleanup-validation-20261110/acceptance.json`。硬件结果仅对应本次binary、SDK和负载。
未完成项包括批量Delta正式集成、
Qwen保留KV多轮/满槽、拒绝纠正、真实DLM–TLM接收率及长期可靠性。EP已验配置保持封版。
