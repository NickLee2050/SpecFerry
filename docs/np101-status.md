# NP101 推理实现与优化结论

更新：2026-10-27。活跃模型为真实 `facebook/opt-350m`；目标是保持旧实现的正确性，
降低单请求decode延迟。未完成项见[TODO](../TODO.md)，每次上板先读
[事故台账](np101-recovery-incidents.md)，接口见[测试手册](../tests/README.md)，
SDK失败契约见[问题表](np101-sdk-driver-validation-report.md)。
10-23精简前文件和diff保存在`.cache/runs/commit-preparation-20261023/`；原始实验数据不删改。

原始证据相对 `.cache/runs/`，不随Git分发；本次精简前全部源码/文档及HEAD身份在
`repository-compaction-20261016/before/`、`before-inventory.json`；退役文件列于
`retired-paths.json`。早期归档及原始硬件记录保持不变。

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

当前证据覆盖有界生成、状态/隔离、预算/释放和延迟内核检查；长期稳定、Qwen C128、
内部硬件利用率及间歇性hang/EDMA根因未解决。恢复遵守板卡断电确认及完整门禁，
不再强制主机冷启动或boot ID改变。更长容量/全图prefill、SoC对照和TLM协议见TODO。
收尾前文档及身份留存`ep-baseline-finalization-20261015/before/`、`before-manifest.json`。

`M`表示GEMM一次处理的token行数：`Y[M,N] = X[M,K] × W[K,N]`。
M1用于逐token decode；M16可同时处理16个已知prompt token，不表示一次生成16个
未知后续token，也不表示16个用户。两种固定形状可用不同图；常驻两套图是否复制权重
取决于实现和NBG产物，不是M本身的要求。C64/C128另指上下文容量，不是M。

## SoC部署与内存

### 当前配置

- 通过`ssh np101`访问eth1的`192.168.35.21/22`，网关`192.168.33.100`；eth0关闭。
  仅用IPv4，已验链路为100Mbps全双工；灯速不能作为接口编号。
- SSH服务是Dropbear，`/etc/default/dropbear`配置`-p 5062`，不是OpenSSH。
  主机使用专用`~/.ssh/id_ed25519_np101`；开机诊断脚本已移除，rcS恢复原厂版本。
- 主机EP驱动仍masked/未加载；板端`S90test`已移出启动目录，NPU不自动加载。
  10-23重启后沿用go.sh参数手动加载，另加`stuckDump=1`，当前SYSTEM池为1792MiB。
  故障转储选项仅用于本次加载，未写入go.sh。
  脚本备份为同目录`go.sh.bak`。板卡重启会卸载驱动，不会自动重新执行脚本。
- 最近已记录板卡boot为`7013aa1c-2105-4d00-8de1-08aa32569a22`。软件reboot曾导致失联，
  原因未定位；后续上电能正常启动，不通过反复reboot试探恢复。

### SDK与部署流程

板端是四核riscv64、Linux5.10.4/Buildroot、glibc2.33，缺少本地编译工具。
厂商SDK位于`/np101_driver/build/sdk/`；核对的36个导出接口均存在，7份核心头文件
与主机相同。沿用现有API源码，使用厂商RISC-V GCC V2.6.1交叉编译，方法见
[README](../README.md#soc-cross-build)。两个OPT入口已构建、动态加载通过；小NBG及
单层计算结果见下表。x86可执行文件/动态库不能直接搬到板上。

主机负责下载、导出、CPU参考和打包，通过SSH传入，生成循环留板上。无需板卡联网
安装工具；其DNS未配置完整、系统时钟未校准，不把一次公网连接失败认定为全面断网。
运行时使用板端`LD_LIBRARY_PATH`和`VIV_MGPU_AFFINITY=1:0`。手写EVIS图编译还需
在私有运行目录提供SDK的CL头文件，设置`LJMICRO_SDK_DIR`；缺少`cl_viv_vx_ext.h`
曾使shader编译失败，必须在初始化阶段拒绝，不带错继续建图。

原生OPT包加载器支持显式`np101-soc-riscv64`配置，核对板端模块文件及五库哈希；
SoC模块没有sysfs srcversion，活跃模块身份还需运行器记录boot和加载过程。
本地SoC派生包只更新运行库身份，保持权重/NBG和端口契约；EP包不改动。
Python导出仍在主机完成；SoC使用独立的`np101_soc.py`部署/监督入口，复用OPT数值验收。
Qwen尚未在SoC验收。

构建显式选择`SPECFERRY_TARGET=ep|soc`，产物分置`build/ep`和`build/soc`。共用模型/
tensor逻辑调用`native/platform/runtime_profile.hpp`，CMake仅链接对应平台实现，
无模式宏；驱动身份与库路径按平台归属，包校验共用。正式生成记录`platform.json`。
目录职责与命令见README；构建目标不自动选择图/KV策略。

驱动身份：10-24测试前核验的SoC模块SHA256为`4c9a6ca2…78449bc`，内部版本
`6.4.9.9.452545`，目标为RISC-V/Linux5.10.4，无srcversion。fpga02安装包为
`ljmnp 1.0.5`，源码在`/usr/src/ljmnp-1.0.5-ubuntu20.04/`，模块在
`/usr/lib/ljmicro/driver/galcore.ko`；内部版本相同，但目标为x86-64/Linux5.15，
srcversion为`F53A3E6EFBB7D0FD7D4617C`。同一内部版本不证明补丁/配置相同，不能
把SoC直接标作v105/v106或互换模块。官方ModelTestSoc包也有同号但不同哈希的模块。
证据为`np101-soc-driver-comparison-20261024/summary.json`，使用当日已核验身份及匹配副本。

### 内存布局与分配结论

| 区域／API池 | 当前范围与含义 |
|---|---|
| Linux系统内存 | `0x40200000–0x7fffffff`；设备树声明1GiB、前2MiB留给固件，实际MemTotal约925MiB |
| SYSTEM连续池 | 按用户转述的厂商指引，从`0xe0000000/0x10000000`改为base=`0x80000000`、size=`0x70000000`；当前1792MiB，止于`0xefffffff` |
| DLA保留声明 | 设备树仍声明`0xd0000000–0xdfffffff`的256MiB `dla_dma`，位于新池内；未发现memory-region消费者，但小容量通过不证明此区无其他访问者 |
| VPU区域 | `0xf0000000–0xf07fffff`的8MiB缓冲区，不能向上并入SYSTEM |
| 未说明尾部 | `0xf0800000–0xffffffff`的248MiB归属未确认，不因iomem未列出就当作空闲 |
| VIRTUAL | 来自Linux页分配器，与CPU共享系统内存，不是另有一块2.7GiB |
| external／exclusive／SRAM | 当前大小为0；显式申请返回OUT_OF_MEMORY，未在这些池配置其他地址 |

const/nonconst是张量语义，不与物理池一一对应；当前两类均可进入SYSTEM，应用各1GiB
预算保持。普通tensor没有pool选择字段，驱动可从SYSTEM回退到VIRTUAL。诊断用HAL
接口能指定池并返回实际池及句柄；DEFAULT/LOCAL/UNIFIED均返回SYSTEM，不增加容量。
HAL句柄导入tensor的所有权契约尚未验证，不把句柄直接当作主机指针。

旧256MiB配置下，11个池选择项全部检查：只有SYSTEM、VIRTUAL及SYSTEM别名成功。
SYSTEM保留载荷255.3125MiB后再申请64KiB被拒；VIRTUAL单独保留224MiB后因Linux
余量保护主动停止。另一次普通tensor检查同时保留SYSTEM248MiB＋VIRTUAL176MiB，
合计424MiB，完整性及释放通过。VIRTUAL的停止值不是容量上限，两次独立容量结果也
不能相加成同时常驻验收。

EP反汇编及历史日志追踪到external=`0x44100000–0xbbbfffff`（1915MiB）、
exclusive=`0xc0000000–0xffffffff`（1024MiB）。这些范围与SoC Linux/保留区重叠，
不能直接照搬EP配置。历史2840MiB累计载荷还出现过24字节损坏，不能当作可靠池容量。
SoC固定池可以跳过资源占用检查；驱动加载成功本身不能证明地址安全。

### 已验范围与实现边界

完整OPT已在SoC完成C64/M1短请求生成，计算包含设备embedding、24层decoder、
最终归一化、head和选词。逐decoder保守链及单个整图NBG均使用固定独立输入/输出，
CPU衔接状态；最新整图只回写当前token的KV位置。**正式OPT入口已完成双平台构建、
可迁移部署及有界请求隔离验收。**同Context依次处理17/6/17个输入、各16个输出，
48个选词、末层KV/未写槽、重复长请求KV逐字节一致及传输计数全部通过。
预算/释放和45.022秒延迟检查通过，未新增hang或EDMA故障。

正式入口显式使用`kv_transfer mapped-slot`，仅支持整图C64/M1及请求间reset。
固定独立输入/输出保留在同一图内，复用4KiB主机缓冲回写当前KV槽，不建额外复制或
清零图；每步KV读写共384KiB。初始化、请求清零与粗粒度诊断单独计时，传输次数及
字节数纳入验收。默认`device-copy`保持已有EP路径，无模式宏或隐式平台切换。
实现/上板证据：`opt-mapped-slot-integration-20261024/summary.json`、
`soc-formal-deployment-20261024/{implementation.json,formal-run/summary.json}`及
`soc-formal-finalization-20261024/summary.json`。正式收尾完成37项检查；边界位置及全层KV
仍由前述实验入口证据覆盖，未重复上板；不外推长期常驻或不同容量。

部署包约1.212GiB，包含binary、权重、NBG、请求/参考和CL头文件；逐文件核验哈希、
RISC-V架构、执行位和板端SDK，原EP包不改动。板端监督独立于SSH连接，持有进程组
期限、内核监测及Linux余量保护；主机断线只重连同一run并取回证据，不重提模型。
普通运行仍拒绝恢复标记；本轮按用户指示限定同boot旧故障诊断，旧标记逐字节保留。

| 范围 | 结论 |
|---|---|
| 原256MiB池 | 小out_proj及完整layer0/M1/C64通过，纯图均值11.322/26.265ms；完整模型未进入推理，受Linux余量与初始化期限限制 |
| VIRTUAL计算 | 三个8MiB张量的ADD通过；单decoder进入VIRTUAL后数值通过，纯图26.459ms。不是大容量整模验收 |
| 新1792MiB池 | 42MiB存量写读及小out_proj通过；单层重复执行和共享八层链曾hang，详见事故台账 |
| 独立固定绑定 | 单图36次轮换输入通过；CPU衔接的8/16/24层各4个连续输入通过。24层图执行617.613ms/输入，含CPU衔接906.952ms；未接head，不是输出吞吐 |
| 完整生成 | 逐decoder两请求共32个输出与CPU参考一致；整图科学博物馆样例16个输出一致；logits、首末层KV及未写槽通过 |
| 最新局部KV回写 | 同请求五轮共80个输出一致，160次RunGraph返回；位置63局部写及全部48个输入KV与完整输出逐字节核对通过 |
| 正式入口及部署监督 | 单请求及长/短/重复长验收通过；最新48输出、85次RunGraph，161.841秒正常退出，Linux余量最低813.172MiB，池/GFP恢复基线 |

成功轮均检查预算、释放、完整内核和至少45秒延迟日志。当前整图const/nonconst峰值
662,593,528/25,266,764字节，SYSTEM采样峰值632.502MiB，Linux余量最低约813MiB，
VIRTUAL为0；各段1GiB预算保持。未覆盖全池/DLA区域、新上下文容量或长期稳定。
用户曾明确授权同boot故障后的诊断；这些成功不证明旧hang修复，原恢复标记保留。

### SoC完整生成性能

以下均为相同科学博物馆样例、17输入/16输出，C64；注明M16的配置仅prefill使用M16。
实际吞吐为16/生成wall，首词后速率为15/后续decode时间。初始化、reset、最终校验和
释放另计；早期SoC两行含逐调用记录及每步logits诊断，不能作纯粹的实现单变量比较。

| 配置 | TTFT秒 | 实际输出token/s | 首词后decode token/s |
|---|---:|---:|---:|
| SoC逐decoder，完整KV搬运 | 16.180 | 0.514 | 1.003 |
| SoC单个整图NBG，完整KV搬运 | 13.939 | 0.609 | 1.216 |
| SoC整图轻量完整KV，同轮候选前后基线 | 13.665 | 0.622 | 1.243 |
| SoC整图局部KV回写 | 9.467 | 0.898 | 1.796 |
| SoC整图局部回写＋图内FP32 Select更新 | 5.147 | 1.651 | 3.300 |
| **SoC上述M1＋编译attention/共享FCL的M16 prefill** | **2.230** | **2.360** | **3.297** |
| SoC正式入口局部KV，两次同长请求，逐调用诊断开启 | 9.661 | 0.880 | 1.759 |
| EP已验混合M1 | 6.782 | 1.216 | 2.354 |
| EP已验混合M1＋M16 prefill | 1.782 | 1.959 | 2.349 |
| EP早期最快FCL投影组合，历史记录 | 4.011 | 1.966 | 3.634 |

正式入口性能行只取两次17输入请求：合计32输出/36.375秒，首词后30个输出间隔。
6输入短请求只用于隔离验收，不混入同长对照。三请求进程161.841秒包含文件读取/
身份核验、初始化及释放，生成wall合计48.307秒；其余不计入输出吞吐。
首次单请求模型初始化6.866秒，其中Setup/Verify为5.834/0.074秒。正式轮含SDK逐调用
记录，未做匹配off/on/off，不把约2%差异归因于插桩或代码重构，也不宣称额外加速。

各路径输出相同续文：` the history of computing from the early days of the computer to today.\n\n`。
逐decoder另验6输入/16输出，实际0.775、首词后1.007token/s。旧EP单个整图的0.920/
1.839token/s使用不同实际输入，仅作历史参照；不能用旧request.json的文本替代实际
requests.txt。SoC与EP的SDK、图组织、连接和监测也不同，尚不能归因为硬件模式本身。

10-25的Select更新行为三次相同17输入/16输出：48输出/29.077秒，首词后45个间隔。
相对本轮替换前整图，实际输出/decode提高83.87%/83.81%，TTFT由9.466降至5.147秒。
decode超过最近已验EP的M1 NBG组合约40.2%，尚低于早期FCL峰值3.634token/s；
该M1配置的实际输出仍低于EP的M16组合；新增M16的结果见下节。
三次选词、末层FP32 KV/未写槽、reset后重复状态及释放/45秒检查通过，无新故障。

### 图内KV选择更新

`export_np101_opt_model.py --kv-update select`将ScatterND替换为Slice/Equal生成槽位
掩码，再由Where选择当前新K/V或原缓存；编译为FP32 Select。索引契约仍是按head
顺序传入`[head, position]`，不需要CPU预知当前K/V。掩码在24层间复用，52入50出、
权重和其余ONNX节点保持不变。仍产生完整KV输出，由原局部映射回写提交当前槽位；
这是更新算子的替换，不是原地更新或进一步缩小主机IO。

同工具链第12层完整decoder对照各18次计时：26.992→17.624ms，减少34.70%；
数值最大绝对差0.0078125，未更新槽位逐字节一致。整模48条ScatterND均替换，
原INT32 ArgMax修正保留；常量/非常量峰值预算713,919,739/25,266,764字节。
当前仅接受SoC C64/M1与固定请求范围；EP默认导出仍用ScatterND，C128未新增验收。

整模复用上一轮采集binary及关闭/开启/关闭请求。首词后RunGraph约297.101ms，
计数关闭前后297.885/297.337ms，计数开启相对均值−0.17%，未见可分辨的增量。
事件提交/命令提交/完成等待仍为1/1/8次，分别4.788/39.709/244.959ms；主要减少
发生在完成等待区间，不能据此进一步区分设备计算与内部搬运。初始化6.864秒及文件
加载、核验、诊断读回均排除在推理指标外。证据`soc-cache-selection-20261025/`
的`summary.json`、`implementation.json`、`wait-summary.json`及两份run目录。

### 手写attention常驻M16：初始化未通过

10-25新增独立M16链，使用18层已验MLP NBG及6层普通FFN，保留原M1整图。
两者NBG常量预算合计1,054,359,931B，低于1GiB；普通权重保持既有分类，没有共享
NBG内部权重或修改is_const。M16有效前缀KV由FP16精确扩为FP32并转换布局，再交给
M1处理剩余1–16个prompt token与后续decode。交叉构建、CPU转换及计数契约检查通过。

三次最小17输入/16输出尝试均在初始化触发192MiB Linux余量保护，尚无RunGraph，
因此没有M16数值或吞吐验收结果。SYSTEM池约1.43GiB峰值未耗尽；固定malloc阈值与
逐图Verify后malloc_trim均未解决Linux占用。最后62次成功compile后，glibc活堆约
607.13MiB；22个attention构建区间累计增加492.48MiB，占81.1%，首层后约21.79MiB/层。
区间包含建图、AddTensor、Setup及Verify，不能单独归因Verify，更不能归因未执行的Run。

Linux CPU堆与go.sh的SYSTEM设备池是不同分配路径。扩大Linux需协调设备树及NPU池
边界并重启；只改go.sh不会增加Linux内存，固件是否覆写设备树仍待确认。用户因不能
重启要求停止，板端配置未改。三次均完成退出、池/GFP释放及45秒检查，无新增hang；
旧恢复标记保留。原已验M1性能不变，本候选不得记为已验优化。
证据：`soc-prefill-integration-20261025/`的三份run、`heap-stage-attribution.json`及
预算/CPU契约记录；只读启动配置快照为`soc-memory-repartition-20261025/`。

### 编译M16 decoder与共享权重FFN

10-25将完整decoder导出为M16 NBG，包含QKV、因果attention、FP32块选择KV更新、
out_proj、FFN及归一化；CPU只负责输入和调用。对照同源Select更新的M1 NBG，以真实
layer0/11/23权重、同一48词prompt的首块及第33–48词块测试。每组一轮暖机、两轮测量，
M1用16个逐词快照调用，M16用一个16词快照调用；图外IO、日志、准备和读回均不计。
这是已知输入的decoder集合回放，不是连续24层生成或实际输出吞吐。

| 零基层号 | M1处理16词，ms | M16处理16词，ms | 加速比 |
|---|---:|---:|---:|
| 0 | 217.217 | 18.835 | 11.53倍 |
| 11 | 224.662 | 21.926 | 10.25倍 |
| 23 | 233.653 | 21.022 | 11.11倍 |

集合合计加速10.934倍；输出和KV通过原容差，最大绝对误差0.015625，未改写缓存按位
保留。六图初始化1.700s，306次RunGraph的整个子进程9.265s；应用const/nonconst峰值
154.03/6.21MiB，SYSTEM采样峰值153.19MiB，Linux最低余量757.20MiB，VIRTUAL为0。
退出、池/GFP释放及45.062秒延迟检查通过，无新故障，原恢复标记保留。此前无效的
malloc_trim诊断钩子已移除，历史证据保留；没有更改驱动、内存布局或默认推理配置。

按三个产物平均大小估算，24层M16 NBG加现有M1整图常量约1.305GiB，尚不含M16
embedding等入口，超过既定const≤1GiB；不能直接复制两套整模。
未与手写M16进行硬件性能对照，也未验C128、尾部不足16词或长期常驻。
证据：`soc-compiled-prefill-20261025/summary.json`、源权重/端口绑定、CPU检查、编译产物
及`run/`；临时诊断仅在忽略的缓存目录，复用导出函数为`prefill_decoder_model`。

随后接入预算内混合整模：24层attention均用NBG，其中末5层包含FFN，前19层沿用
普通权重FFN；保留原M1整图，使用独立FP32 KV及一次前缀交接。初始化116.933秒通过，
const/nonconst峰值993.10/431.83MiB，SYSTEM采样峰值1513.10MiB，Linux最低余量
660.68MiB；解决了本候选的常驻预算与Linux余量问题，未修改权重分类、驱动或内存布局。

首个17输入/16输出请求完成，但16个输出均为1437，且前16个已知输入对应的末层KV
已明显偏离参考（K/V最大绝对误差12.543/4.457），非仅选词临界差异。首词10.852秒、
生成总15.356秒仅保留为失败诊断耗时，**不计为有效吞吐或收益**；后续49/17输入请求
按数值门禁跳过。子进程268.906秒退出1，无新增内核故障，释放及45.004秒检查正常。
随后独立运行同一M16链，首个偏差已在embedding出现，48个KV输出到提交后的前缀
均逐字节一致；M1常驻不是错误的必要条件。原失败性能不作为有效基线。
证据：`soc-resident-compiled-prefill-20261025/{failure-analysis.json,run/}`，包含NBG/权重
哈希、公开端口、实际计数及完整失败结果；恢复标记保留。

联合组件对照进一步定位到M16输入投影MatMul：token查表、位置编码正确；第2/4/6/8行
的误差与漏算K=504–511的8个输入维度相符，替换CPU计算后误差降至6.1e-5。
这是数值模式证据，尚不能断言SDK内部漏算的根因。显式设置图版本1.1.35仍失败。
19层普通FFN占诊断链RunGraph约96%，完整layer0 FFN对照自身也有数值错误。

将输入投影及19层FFN的fc1/fc2改为共享普通权重FCL，保留独立bias、ReLU、FP32
归一化与原有24层attention NBG/末5层完整decoder NBG。WeightBank持有只读普通
backing，不修改is_const、不新增常量副本；M1整图及默认配置不变。两个完整组件通过，
embedding/FFN最大绝对误差分别6.1e-5/0.0078125。每组三次调用及前后普通路径对照，
embedding约32.4→7.84ms、FFN约474→71.95ms；准备、IO和读回不计。
普通路径数值错误，以上仅解释成本变化，不宣称是正确模型的加速比。
证据：`soc-prefill-boundaries-20261025/`、`soc-prefill-components-20261025/`。

部署包和权重文件的完整SHA256核验改为最多4线程，保留全部身份、大小与校验门禁；
每个权重核验线程独立文件流和摘要上下文、最多8MiB缓冲。这里只并行文件核验，
SDK Setup/Verify仍串行，未引入并行Verify。

修复后的常驻组合已通过正式入口C64验收：依次reset，17/49/17输入各生成16词，
48个选词、末层KV/未写槽、重复短请求状态、容量拒绝、关闭后拒绝与传输计数均通过。
M16分别执行1/3/1块，剩余prompt和输出沿用原M1整图；长请求消费到第64槽。

| 输入→输出 | 请求数 | 平均TTFT秒 | 实际输出token/s | 首词后decode token/s |
|---|---:|---:|---:|---:|
| 17→16，匹配既有M1样例 | 2 | 2.230 | 2.360 | 3.297 |
| 49→16 | 1 | 6.069 | 1.511 | 3.319 |

短请求合计32输出/13.561秒，首词后30个间隔/9.100秒；对既有M1基线，TTFT降低
56.67%、实际输出吞吐提高42.94%，decode变化−0.11%。长请求16输出/10.589秒，
没有匹配的M1长输入对照，不外推加速比。相对历史EP M16组合，短请求实际吞吐高
20.46%，但TTFT仍慢于其1.782秒；不是控制其他条件相同的双平台对照。

模型初始化114.564秒，整子进程232.652秒，均不混入生成指标；文件核验、请求reset、
诊断读回及释放另计。应用const/nonconst峰值993.10/429.45MiB，未改权重分类。
SYSTEM采样峰值1353.67MiB，Linux最低余量676.42MiB，VIRTUAL为0。
释放及45.059秒延迟检查通过，无新内核故障，旧恢复标记保留；仅接受上述有界C64
请求，不外推长期稳定、分次追加或新容量。证据：
`soc-shared-prefill-20261025/{performance.json,run/summary.json}`。

### C128与共享输入KV

10-26将已验分图拓扑扩展到C128：M1整图仍用Select更新和INT32 ArgMax，M16保留
24层编译attention、19层共享FCL FFN及末5层完整decoder。attention NBG与普通FFN
合图的候选被Verify拒绝、尚未推理，已撤回；无新增设备故障，详见事故台账。

运行时改为M16固定引用M1的48份FP32输入KV；各NBG输出仍独立，每块同步完成后
按原映射路径提交16槽。由此省掉最终完整前缀交接及重复清零，不改算术、权重分类、
图划分或每块44次RunGraph。新结果显式记录`prefill_cache_policy=shared-input`；
旧记录缺字段按独立缓存核验，传输计数不能混用。M16消费者先于M1 owner释放。

同一C128包和SDK各运行一次97→16、reset后17→16集合；两配置共64个输出、末层KV、
未写槽、reset、传输计数和释放均通过。每个长请求使用6个M16块、消费到112槽；
没有满128槽、分次追加或长期稳定验收，也没有新增C64共享缓存硬件结果。

| 输入→输出 | KV策略 | TTFT秒 | 实际输出token/s | 首词后decode token/s |
|---|---|---:|---:|---:|
| 97→16 | 独立缓存＋前缀交接 | 12.356 | 0.898 | 2.747 |
| 97→16 | 共享输入缓存 | 11.843 | 0.925 | 2.747 |
| 17→16 | 独立缓存＋前缀交接 | 2.365 | 2.043 | 2.743 |
| 17→16 | 共享输入缓存 | 2.277 | 2.066 | 2.743 |

单轮对照长/短TTFT分别下降4.15%/3.72%，实际输出吞吐提高2.97%/1.14%；不外推
统计稳定收益。长请求为16输出/17.303秒，首词后15个间隔/5.460秒；短请求单列。
97词航空展览样例续文与CPU一致：` people understand the history of flight.\n\nThe Science Museum is located at the`。
C128短请求decode也低于历史C64约3.30token/s，记录扩容后的成本，不据此确定内部根因。

长请求确实减少48次读、48次写，合计36MiB；后续reset少写24MiB，耗时0.201→0.119秒，
reset不计入上述生成吞吐。nonconst峰值474.96→450.96MiB，const为993.85MiB；
共享候选SYSTEM峰值1385.86MiB、Linux最低余量672.12MiB、VIRTUAL为0。
初始化119.594秒、子进程234.977秒单列，释放及45.034秒检查通过，无新hang/EDMA。
此轮尚未实施初始化优化；旧恢复标记保留。证据：`soc-context-prefill-20261026/`及
`soc-prefill-followup-20261026/{comparison.json,shared-kv-c128/run/}`。

### 初始化归因与校验优化

10-26复用上述C128包和请求，仅初始化期间采集内存汇总，进入生成前关闭采集。
初始化完整返回，但首请求随后发生新NP hang；没有有效输出吞吐，不能作为数值验收。
用户随后授权同boot重试，原版本关闭采集的构造为116.618秒，长请求通过后在短请求
再次hang。初始化off/on/off为119.594/118.763/116.618秒；采集值落在两次off之间，
不能宣称零扰动或据此建立hang因果。两个失败轮均完成初始化，仅用于该阶段归因。

| 阶段 | 优化前归因轮秒 | 优化后单请求秒 | 边界与主要部分 |
|---|---:|---:|---|
| 原生前置校验 | 88.627 | 79.748 | 原权重21.413、M1包及SDK44.826、M16包22.343；均在构造计时之前 |
| Context及模型构造 | 118.763 | 118.054 | 原Setup43.564、AddTensor30.212、Verify31.235、权重准备12.793；其中读取12.785秒，不重复相加 |
| 前置校验至模型就绪 | 207.391 | 197.802 | 不含Python监督器的部署包校验、推理及释放检查 |

M16 embedding单次Verify29.879秒，占全部Verify的95.7%；其余Verify仅1.355秒。
M1整图Setup另占27.517秒。没有证据把Setup全部认定为SD读取或设备搬运；
并行Verify暂不优先，也不重复已有“缩小预留量无明显收益”的试验。

先实现不改变SDK执行的优化：M16包完成语义解析后，用最多4个CPU线程完整核验24份
NBG，复用权重校验的有界调度。保留路径/大小/版本/来源/摘要/预算和SDK身份检查，
不缓存跨启动校验结果。不加载SDK的板端CPU对照对同一24份文件各完整读取
328,203,008B：串行19.193秒→并行5.030秒→串行18.649秒，全部摘要通过。
相对两次串行均值减少13.891秒（73.4%，3.76倍）；这是文件校验集合的收益，
不是整模启动或推理加速，不能把原22.343秒直接记作节省。
原生入口新增`preparation_seconds`与`ready_seconds`，后者为前置准备加当前一次构造；
与`initialize_seconds`、生成耗时分开。4项CPU契约测试和SoC正式入口交叉构建通过。
按本轮授权缩为已有C128的17→16单请求，45项输出/末层KV/未写槽/传输/释放检查通过，
45.038秒无新故障；子进程208.936秒。Linux最低余量680.50MiB、SYSTEM峰值1385.86MiB，
const/nonconst峰值993.85/450.96MiB，VIRTUAL为0；恢复标记保留，没有并行SDK Verify。

同模型/NBG/权重的本次就绪时间比归因轮减少9.589秒（4.62%），构造成本基本不变。
这是单次初始化观察：此前请求集合失败、候选缩为单请求，不能比较两轮整个进程吞吐，
也不将故障前的初始化测量当完整验收。实际16输出/7.775秒=2.058token/s，
TTFT2.303秒；首词后15个间隔/5.472秒=2.741token/s，与此前短请求2.743接近。
仅接受本次最小完整链，未新增长/短reset或长期稳定验收。

embedding仍保持已验实现。可进一步评估将512列词表块从4096行扩大到8192行，
把13组Gather/边界/Select缩为7组，单块仍不超过8MiB；尚未实施/上板，不能预报收益。
当前无内存耗尽证据，故障详情及恢复门禁见事故台账。
证据：`soc-initialization-20261026/{analysis.json,trace/run/,cpu-hash/,retry/,retry-single/}`，
当前验收/性能在`retry-single/{parallel-checks/run/summary.json,performance.json}`。

### 合并投影、私有存储与归一化候选

10-26以C128第12层完整decoder、三组既有真实状态做QKV合并对照：三次固定FP16
Gemm改为一次宽Gemm后切片，保留各自bias Add、Q缩放和FP16边界。CPU九份输出按位
一致；硬件36次计时及12次预热通过hidden、FP32 KV/未改槽、释放和45秒检查。
NBG的TP由33降至29，SH72/NN33不变；完整decoder均值19.863→19.407ms，减少2.30%，
但原图/候选前后漂移分别−8.50%/+4.56%，未建立超出漂移的收益，不接整模默认。

同轮真实layer0/M16完整FFN包含共享普通FCL权重、残差和FP32归一化，同binary各做
普通/virtual及普通/virtual＋SDK LayerNorm的前后夹持集合，共48次计时、6次预热。
匹配对照为72.736→72.886ms（+0.21%）及73.299→72.297ms（−1.37%），未见明显加速；
八份输出通过原容差，最大绝对误差0.0078125，公开IO保持普通张量，释放/45秒检查通过。
只接受上述完整功能段回放，不是输出吞吐或整模数值验收；正式配置保持原策略。

SDK的FP32 LAYER_NORM按width归约，data/bias/scale顺序，使用E[x²]−E[x]²＋epsilon；
不同于原centered-square算法，不能无验收扩用。TENSOR_ADD_MEAN_STDDEV_NORM的内嵌
F32 kernel非零方差时不加epsilon，已排除OPT替换。虚拟张量语义与VIRTUAL池无关。

10-27退役QKV合并、私有virtual及融合LayerNorm的可选实现和专用测试；正式默认推理
未采用这些候选。源码、补丁和SHA256归档于`optimization-candidate-retirement-20261027/`，
保留数值、性能与SDK语义证据。监督器的启动失败记录修复及回归测试继续保留。

证据：`soc-packed-qkv-20261026/attempt-3/run/`、
`soc-graph-storage-normalization-20261026/attempt-1/run/`及`soc-dataflow-20261026/summary.json`。
均无新增设备故障，全部历史恢复标记保留；无效的pre-exec打包失败单独归档，未调用SDK。

### M16 prefill功能段归因

10-26复用正式C128/shared-input配置，在同一常驻模型执行三次相同17输入/16输出请求，
SDK计时关闭→内存汇总→关闭，VPD始终关闭；只在缓存诊断binary加入完整块与KV提交
的外层薄计时，初始化不采集。中间请求的非重叠M16块时间为：

| 功能段 | 次数 | 合计ms | M16块占比 |
|---|---:|---:|---:|
| 共享普通权重FFN | 19 | 1372.736 | 72.23% |
| attention NBG | 19 | 341.329 | 17.96% |
| 完整decoder NBG | 5 | 107.758 | 5.67% |
| KV映射提交，含拷贝 | 1 | 62.687 | 3.30% |
| embedding图 | 1 | 13.865 | 0.73% |
| 控制及其它残余 | — | 2.038 | 0.11% |

M16块合计1900.413ms、44次RunGraph；随后一个M1尾图357.068ms，另列、不混入块占比。
以上是完整提交图/提交过程的时间，仍包含内部计算、搬运和等待，不是单算子或物理
引擎利用率。普通FFN每层约72.25ms，与独立完整FFN对照一致；末5层完整decoder
每层约21.55ms，但层号、权重和边界不同，不能把比值当同层替换加速比。

开启汇总对TTFT/生成wall相对前后关闭均值分别−0.08%/+0.30%；关闭轮TTFT漂移−0.91%，
未见显著采集干扰。两次关闭轮32输出/15.476秒，平均TTFT2.268秒，实际2.068输出
token/s，首词后2.742token/s；准备、reset、粗读回与释放不计入生成。
三请求共48输出、末层KV/未写槽、重复状态和传输共71项检查通过；child233.022秒，
释放及45.091秒检查通过，无新增故障。Linux最低672.95MiB、SYSTEM峰值1385.86MiB，
VIRTUAL为0；未改正式默认或清除历史门禁。优先追查共享普通FFN的执行/权重路径，
不优先继续embedding、图外KV提交或仅减少小操作。
证据：`soc-prefill-attribution-20261026/{analysis.json,attempt-1/run/}`。

### 当前权重常驻量与后续模型预算

10-27按缓存部署清单核算，OPT固定checkpoint有331,196,416个独立参数，全部FP16原始
权重662,392,832B（631.71MiB）。当前M1完整NBG与M16分段链没有建立跨NBG权重共享；
M16内部WeightBank共享不能理解为与M1只保留一套参数。其权重相关应用记账为：

| 部分 | MiB | 含义 |
|---|---:|---|
| M1整图NBG | 680.85 | 一套完整模型的编译产物保守常量界 |
| M16的19个attention NBG | 176.31 | 对应投影/归一化参数及编译布局 |
| M16末5个完整decoder NBG | 136.69 | 另含这5层FFN |
| M16普通embedding权重 | 54.10 | token/位置表及输入投影 |
| M16的19层普通FFN权重 | 304.33 | 含FP32 norm affine；WeightBank内部复用 |
| 权重相关记账合计 | 1352.28 | M16整体又覆盖近乎一套模型参数 |

4层普通包另有约93.77MiB显式KV/激活/控制等应用payload，SYSTEM采样峰值1491.56MiB；
12层受控缓存包的SDK副本和池峰值另见下文，不包含在此权重表中。
NBG数字含保守界，不能当成实际物理分区，也不能直接相减推断SDK workspace；内部
布局、打包副本和临时缓冲尚未逐项测清。磁盘weights.bin不是第三套完整设备常驻权重。
逐头attention展开增加命令和中间数据，不代表16份全模型参数。本轮FCL→卷积没有
新增大矩阵副本，当前冗余主要来自此前选择M1/M16两条常驻性能路径。

缓存Qwen3.5文本部署清单有752,393,024个参数（含少量FP32权重），原始payload
1,504,791,232B，即1.401GiB/1435.08MiB。当前SoC池1792MiB扣掉单份原始权重后仅余
356.92MiB，尚需状态、激活和SDK工作区，并满足const/nonconst各1GiB预算。不能照搬
OPT的双套常驻组合；后续优先基于现有Qwen共享WeightBank设计一份参数常驻，再独立
核算SoC开销。未证明单份方案放不下，也不能把EP部署通过外推为SoC容量通过。
证据：`soc-ffn-verify-20261027/memory-audit.json`，由固定部署manifest离线核算，未下载
模型、未进行Qwen设备分配。

### 共享普通权重FFN的batch卷积路径

10-27固定真实第1层、M16输入、完整FFN/残差/FP32归一化做夹持集合对照。
普通FCL与编译器生成C的参数一致，但SoC普通FCL默认dispatch的NN/TP入口不支持；
离线编译器另有batch FCL转卷积及转置优化。不能仅凭算子名推断执行路径或权重搬运。

| 完整FFN实现 | 匹配普通FCL ms | 候选ms | 数值 |
|---|---:|---:|---|
| 独立真正const FCL | 74.025 | 15.418 | 通过；额外常量副本不适合扩满19层 |
| 固定权重编译NBG | 72.645 | 15.297 | 通过；仍需独立常量预算 |
| 共享普通权重batch 1×1卷积 | 72.585 | 16.910 | 通过，约4.29倍，最大误差0.0078125 |

已保留的API路径将同一请求的16个prefill token置于卷积batch维：
`[K,16]→[1,1,K,16] --普通[1,1,K,N]权重→ [1,1,N,16]→[N,16]`。
每个token的1×1卷积计算`y[t,n]=Σ_k x[t,k]w[n,k]`，与线性投影相同；FFN在token间
独立，故可这样重写。两个投影仍为1024→4096→1024，未减少算术量或改变权重精度。

可复用经验：等价数学表达、API算子名、实际后端路径应分开核实。形状、布局及权重
属性可能决定SDK能否采用特定优化；先检查编译器生成代码/命令，再用公开接口复现
候选，固定原权重分类、精度边界及同层完整功能段做数值/夹持计时验证。本次证据支持
“路径选择显著影响成本”，尚未把4.29倍收益拆成算术、打包、搬运和等待各自的贡献，
也未证明普通FCL本次实际采用哪个shader。仅对当前SoC SDK及M16验收，不外推M1/EP。

权重固定常驻，bias、ReLU和FP32归一化保持原边界。空间M维卷积、2D/4D/FP32转置
及1024维分块均有确定数值错误，不接入；原始卷积输出已错，不能归因应用转置。
通用公开权重NBG虽导出成功，但实测NaN，fc2清零/恢复后输出仍逐位相同；动态filter
契约未成立。其大文件内部数据用途未知，不能据生成C只有4B常量就减少const记账。
原生4D公开filter版本仅离线导出，尚未硬件验证，也未解决保守预算。

SoC先仅替换前4层普通FFN，其余15层保持FCL、末5层保持完整decoder NBG；M1整图
和EP策略不变。每层batch卷积构造约16.7秒，限制层数是为遵守300秒实验期限。
复用原C128/shared-input请求，两次reset后17输入→16输出，32输出及58项契约/末层KV、
未写槽、重复状态通过；结果与此前相同请求的两次无采集基线对照如下：

| 指标 | 原实现 | 前4层batch卷积 | 变化 |
|---|---:|---:|---:|
| 平均TTFT秒 | 2.267911 | 2.070370 | −8.71% |
| 实际输出token/s | 2.067701 | 2.121369 | +2.60% |
| 首词后decode token/s | 2.742155 | 2.741265 | −0.03% |

新结果为32输出/15.084599秒、首词后30间隔/10.943852秒；准备、reset、读回与释放
不计入生成。基线为前日同boot/SDK/容量/请求的关闭采集结果，非新一轮整模ABBA。
模型构造180.320秒、原生校验至就绪262.944秒，比归因轮构造约增加60秒；child281.793秒。
后续同层拆图记录显示batch卷积FFN的Verify15.789秒、Setup仅9.39ms，优先定位Verify。
const峰值993.85MiB不变、nonconst452.21MiB，权重payload不变；SYSTEM采样峰值
1491.56MiB、Linux最低668.68MiB，退出及45.009秒检查通过，无新故障。
以下是接入缓存前的扩展约束；4层是当时实验策略，不是硬件层数上限：

- 已测初始化阻塞：单层batch卷积FFN的Verify约15.789秒，Setup仅0.00939秒；原FCL
  同层Verify约0.257秒。当时child281.793秒，仅余18.207秒到300秒期限；再加一层就
  接近期限，缺乏波动余量。按每层额外约15秒粗估，余15层会再增加约225秒，不能原样
  全扩。后续CPU采样定位了压缩/打包热点，见下节；不能把整个Verify归为搬运。
- 应用权重共享已保持：4层替换后权重payload与const峰值不变；新的普通权重4D布局与
  预载键一致，没有另建一份2D矩阵。const/nonconst逻辑预算分别约993.85/452.21MiB。
- 设备池是待核风险：SYSTEM采样峰值约1385.86→1491.56MiB，池总量1792MiB，当前
  峰值余量约300.44MiB。增长约105.70MiB尚未拆清驻留/瞬时缓冲及SDK副本；不能当成
  每层固定开销线性外推，也尚未证明全部19层必定超池。Linux最低余量668.68MiB，
  当前没有Linux内存耗尽证据。
- 两条离线编译捷径未闭环：固定权重FFN NBG需要额外常量副本，而当前const只余约
  30MiB，无法直接增加19层。公开权重通用NBG实测NaN且改变fc2无效，不能当成可用
  共享模板；其18.6MB内部数据也不能擅自改按4B常量计账。原生4D模板仍仅离线。

因此后续针对batch卷积FFN的压缩/打包成本，并核算增加实例时的设备池峰值，再逐步
扩展；不扩大正在运行的期限。新组合的长prompt/长期运行尚未验收。首轮独立hang及
重试详见事故台账，所有旧标记保留。
证据：`soc-prefill-ffn-20261027/{summary.json,bounded-compare/,full-model/}`；
数值失败候选及离线模板仅保留在该忽略目录，不增加正式测试入口。

同日再用一个完整C128/M16 decoder比较两图与attention NBG＋batch卷积FFN合图。
拆开的attention/FFN均完成Setup/Verify；合图在`vsi_nn_VerifyGraph`返回−1，未到
RunGraph；Verify1.552秒返回，退出后池/GFP回归、45.042秒无新故障，故没有合图
推理或性能结论。共享普通权重、端口和版本已固定，不外推所有混合图都不支持。
后续两次原因诊断将合图提前，复用原参考、公开日志及匹配SDK的透明阶段返回码记录；
两次child均约4.312秒，不再先等待拆图FFN冷编译。公开日志无详细错误，22个节点状态
均为0；阶段跟踪确认参数/写依赖/拓扑/分配/图优化/环路/可达性/目标校验全部返回0，
随后`vxoGraph_InitializeAllNodeKernels`首先返回−1，未进入后续成本/命令生成阶段。
匹配反汇编`0x6cd3e–0x6cd4e`调用kernel initializer(node,parameters,count)并传播错误，
但尚未确定哪个kernel及其失败条件；不能把公开节点状态0当作初始化已通过，也不能
据此归因某个FCL/CONV/NBG参数。后一次SYSTEM采样峰值53.215MiB，未见容量耗尽证据。
退出后资源回归，两次45秒延迟检查无新故障。按用户快速排查要求止步，不盲试私有
优化开关或建图变体，正式保留已验拆图配置。新证据：
`soc-merged-verify-stage-20261027/{analysis.json,verify-disassembly.txt,attempt-1/run/}`；
公开日志对照在`soc-merged-verify-reason-20261027/attempt-1/run/`。原始证据：
`soc-prefill-merged-20261027/attempt-2/run/`。夹具SDK前的目录创建错误另存attempt-1，
已修复，不能与该Verify拒绝或NP hang混为一谈。

### batch卷积FFN的Verify热点与拆图边界

同一真实M16完整FFN采用原权重/精度/输入，串行比较整图、输出通道二分、三图拆分；
准备和显式IO不计入功能段运行集合。结果如下，Setup包含在初始化计时内且仅约10ms：

| 构造方式 | 累计Setup＋Verify秒 | 匹配功能段运行ms | 结论 |
|---|---:|---:|---|
| 整个FFN一图 | 15.70/15.39，前后控制 | 约17 | 当前对照 |
| 两个投影各按输出通道二分，仍一图 | 14.88 | 17.42→15.36 | 初始化仅小幅减少，运行局部受益，未接整模 |
| fc1/激活、fc2、残差/归一化三图 | 15.53 | 16.62→34.05 | 总初始化不变，运行约慢一倍，未采用 |

全部数值最大误差0.0078125并通过原容差、释放及45秒检查，无新设备故障。Verify时
进程CPU时间约等于墙钟；这不是原先数百SDK节点图的同一种复杂度证据。历史612节点
约141–169秒、两decoder四图累计11.026秒的成功依然成立，但不能据此外推当前小FFN；
也不把经验非线性认作已证明的三重循环。SDK并行Verify仍未实现/验证。

板端已有perf可对本进程做userspace CPU采样，无需VPD或内核改动。固定同层FFN，
仅在Setup＋Verify期间49Hz采样，关闭→开启→关闭分别15.569/15.440/15.417秒；相对
两端均值−0.34%，端点漂移约0.97%，本次未检出额外减速。751个样本、无丢失，按相同
SDK ELF的函数范围解析后，直接Huffman函数及writeBits合计76.03%，
2D transpose为9.72%；主要符号为fillinKernelBufferV8Huffman、calcKernelSizeV8Huffman、
reorderKernelBufferV8HuffmanFP16BF16和writeBits。它们是CPU初始化热点，不是推理时
CPU占比、含子调用墙钟比例或NPU利用率。下一步优先权重压缩/打包复用，不能仅减图数。

根据匹配SDK反汇编，`VIV_VX_ENABLE_HUFFMAN`在Context创建时默认置1、读取环境
覆盖，并参与软件feature判定；这不是完整的压缩控制公开契约。独立进程默认→置0→默认
对照的初始化为15.603/15.576/15.568秒，置0相对前后均值仅−0.064%，运行分别
17.27/18.16/16.83ms，编译后SYSTEM均为46,288,576B；没有提速或省内存证据，不采用。
这不能证明相关V8打包调用都遵守该开关，也不能据此否定采样热点。未改生产环境、驱动
或权重分类。三个worker的数值、预算/释放和45.027秒日志复核正常，无新增设备故障；
外层诊断摘要漏写预算峰值，自动验收仍保留失败原判定，另存补充复核，不当作整模验收。
独立权重对象的缓存契约未闭环；后续SDK图二进制缓存已有局部结果，见下节。
不继续盲试环境变量，也不先引入SDK并行Verify。

证据：`soc-ffn-verify-20261027/{attempt-1/,api-totals.json,cpu-sampling/,compression-control/}`。
诊断夹具仅在忽略缓存中保留，不增加正式CLI或改变EP/SoC默认选项。

### batch卷积FFN的SDK二进制缓存

10-27匹配SoC SDK的反汇编确认，Verify调用vxoBinaryGraph_CacheOrImport，读取
`VIV_VX_ENABLE_CACHE_GRAPH_BINARY`，可通过`VIV_VX_CACHE_BINARY_GRAPH_DIR`限定目录。
它缓存并导入完整图产物，不是可移植的独立压缩权重格式；opaque权重对象虽有创建/
释放接口，未找到直接接回当前CONV2D的完整序列化契约。通用图导入/导出扩展仅有头文件
声明，匹配板端libOpenVX未提供对应符号，未调用。

在已测约76%初始化CPU样本属于压缩/打包的前提下，限定独立进程诊断该未公开缓存选项，
未写私有字段、改is_const或正式默认。相同真实第1层完整M16 FFN、原容差、串行对照：

| 阶段 | Setup＋Verify秒 | 完整FFN集合均值ms | 编译后SYSTEM MiB |
|---|---:|---:|---:|
| 关闭，前控制 | 15.604 | 16.573 | 44.144 |
| 开启，首次生成 | 16.766 | 16.687 | 44.145 |
| 开启，缓存命中 | 0.405 | 17.237 | 53.170 |
| 关闭，后控制 | 15.597 | 17.503 | 44.144 |

命中相对前后控制均值快38.51倍、初始化减少97.40%；运行处于控制漂移内，不宣称推理
提速。单个缓存20,208,896B（19.27MiB），命中后设备池增加9,464,480B（9.03MiB），
文件大小不等于新增设备内存。首次生成仍需压缩并写盘，只改善后续构造，不免除冷构造。
四个进程共8份数值输出、预算/释放及45秒检查通过，无新增设备故障；旧标记保留。

缓存键可见图结构与部分权重摘要，但未证明覆盖所有矩阵/bias/norm；正式接入前须在
外层绑定全部相关权重摘要、图契约、程序/SDK身份并验证缓存文件完整性，不能仅信任
SDK内部键。不同层诊断分别隔离目录，不把相同shape当作相同权重。共享Context中，
该开关会保留内部状态，且相同字段还被Node_SetParameter、SwapTensor等路径读取；
非目标阶段设环境0也不能证明内部字段已即时刷新，混合NBG/API的绑定和释放须整体验证。
不能仅unset或直接全局开启。SDK并行Verify仍未启用。
缓存会改变SDK常驻和瞬时开销，扩层须先核算；若另行显式导入NBG，仍须计入应用常量预算，
不得重分类原权重规避1GiB限制。普通包保留前4层，受控缓存包的12/19层整模验收见下文；
未向任意图或未匹配SDK全局启用该开关。

证据：`soc-ffn-cache-20261027/{api-audit.json,summary.json,attempt-1/run/}`。

后续复用已验输入、固定真实layer4–11的8层参数，各FFN独立运行；FCL基准已完成，
缓存填充/命中分别用新进程，8图同时常驻，未加载整模。累计Setup＋Verify为
131.994→8.823秒，16份输出最大误差0.0078125；退出、预算及45.076秒检查通过，无新故障。
首次填充/命中后的SYSTEM为347.792/292.600MiB，对照8层FCL为136.413MiB；
显式nonconst live仅135.393→137.892MiB，原权重分类和大小不变，主要增量属于SDK。
命中时逐层计数存在回落，说明有批量回收/分配复用现象，内部具体原因未定；不能把单层
额外9.03MiB或8层均值直接线性外推全19层，也不能把缓存文件大小当作设备常驻量。

八层FCL来源为已完成数值/常驻测量、阶段间释放检查退出的首轮；进程退出后的池/GFP/
owner及45.097秒日志正常。首轮未保存即时不匹配计数；后续独立进程记录到Context
关闭后仍约0.654MiB，
进程退出才回到约0.117MiB基线，支持存在进程级持有，内部对象种类未定。原退出判定
和标记保留，补充复核后仅复用其已完成基准，不重复整个FCL测试。
证据：`soc-ffn-residency-20261027/partial-baseline-review.json`、
`soc-ffn-residency-isolated-20261027/{summary.json,attempt-1/run/}`。

复用8层已验缓存后，再补齐其余11层；19个FFN同时常驻并逐图预热，19份输出通过，
子进程269.546秒、45.071秒检查无新故障。每图的一次预热移到其Verify之后，调用总数
未增加，以便及时提交和回收临时分配。补齐轮混合了新生成与命中，196.031秒累计初始化
不当作19层全冷基线；未为补齐表格另跑整批冷构造。

随后直接复用板端运行缓存，19个FFN全命中Setup＋Verify累计21.807秒；19次逐图预热
及两次完整19图集合执行，集合均值336.051ms，仅表示FFN功能段，不是输出吞吐。
19份输出最大误差0.0078125，应用const/nonconst峰值分别366.129/334.829MiB，
其中const还保守覆盖了19份缓存文件；未改变原权重is_const。子进程83.832秒，
Linux最低764.42MiB，池/GFP/owner及45.007秒检查通过，无新增设备故障，所有旧标记保留。
19个FFN最终SYSTEM为694.020MiB，采样峰值710.007MiB；该范围不含M1、attention、
embedding、KV及LM head，不能当作整模内存或整模验收。

结合已验C128整模4层卷积的稳态1490.897MiB、峰值1491.562MiB，扣除其约105.686MiB
卷积增量，再用8层FCL实测外推19层FCL对照，得到以下**容量估算**：

| 组合 | 稳态估算MiB | 峰值估算MiB | 相对1792MiB池 |
|---|---:|---:|---|
| 19层缓存卷积FFN接整模 | 1756.34 | 1772.99 | 峰值仅约19MiB余量 |
| 12层缓存卷积FFN接整模 | 1619.57 | 1636.22 | 当时估算；后续实测见下文 |

上述估算依赖FCL外推、相同共享关系及当前分配顺序，采样还可能漏掉短暂峰值；不能
据此保证全19层放得下或必然放不下。后续直接验收12层受控缓存候选，同时覆盖缓存
绑定与扩层；不重复4层验收，不另堆显式FFN NBG常量或改变1GiB应用分段预算。

有一轮父进程内复制约366MiB缓存加上权重核验，占去首个Context前70.155秒，导致
120秒期限在layer17构造中触发，未出现设备故障。已保留未完成运行并改为直接复用缓存，
未延长原期限；具体时间线见事故台账。原始缓存及哈希在主机归档，SDK只更新板端运行
副本的访问时间。证据：`soc-ffn-residency-all-20261027/summary.json`、
`soc-ffn-residency-warm-20261027/timeout-analysis.json`及
`soc-ffn-residency-direct-20261027/{summary.json,attempt-1/run/}`。

### 受控缓存与FFN整模验收

10-27先完成前12个、再完成全部19个普通M16 FFN的缓存batch卷积接入；末5层仍为
完整decoder NBG，M1整图、embedding、attention、KV及head不变，实际执行完整24层。
复用同一C128/shared-input包和CPU参考，两次reset后各17输入→16输出：

| 指标 | 4层普通包 | 12层缓存包 | 19层缓存包 |
|---|---:|---:|---:|
| 平均TTFT秒 | 2.070370 | 1.619694 | 1.232234 |
| 实际输出token/s | 2.121369 | 2.257436 | 2.382991 |
| 首词后decode token/s | 2.741265 | 2.743238 | 2.736221 |
| 模型构造秒 | 180.320 | 145.263 | 161.604 |
| 原生准备/校验秒 | 82.625 | 103.665 | 110.879 |
| 原生启动至就绪秒 | 262.944 | 248.928 | 272.482 |
| SYSTEM峰值MiB | 1491.562 | 1619.687 | 1756.365 |

19层组合32输出/13.428501秒，首词后30间隔/10.964029秒；初始化、reset、粗读回和
释放不计入生成，19次初始化FFN预热单独记录。相对12层，TTFT再降23.92%、实际输出
提高5.56%；相对4层分别−40.48%/＋12.33%，decode约0.26%的变化不作为收益。
收益主要来自更多FFN走卷积路径，缓存使这些图能在期限内构造；以上不是同19层缓存
开/关隔离实验。12/19份磁盘缓存为231.25/366.13MiB，首次生成成本和磁盘大小另列。

19层const峰值993.847MiB、nonconst456.737MiB，分别低于1GiB，退出live归零；
原始矩阵共享及is_const不变。SYSTEM驱动高水位从1619.687增到1756.365MiB，与本轮
采样峰值一致，最低余35.635MiB；VIRTUAL为0，Linux最低可用711.30MiB、进程峰值
RSS100.53MiB。58项原数值/末层KV/未写槽/reset/传输契约、释放及45.093秒检查通过，
child288.337秒，无新内核故障，所有旧标记保留。它只证明本次短请求配置：更长上下文、
长期常驻尚未验收，剩余池和300秒期限余量均有限，不继续盲目扩展。

可选包清单绑定全部六项FFN权重、recipe/形状、生产者与消费者binary及完整SDK身份，
每层独立目录核验大小/SHA256，命中后再次核验且要求外层仅一个binary节点；缺失/
变化直接失败，不隐式冷编译。打包器默认保留12层，`--layers 19`选择完整前缀；普通包
仍用4层，EP不变。运行缓存与主机原始归档分离，旧板端缓存不能直接视为可信来源。
本轮完整核验曾发现其中三份旧源内容与归档不符，未提交NPU；重传校验后才继续。

**混合缓存生命周期经验**：SDK会保留Context缓存状态，unset环境变量不足以恢复。
所有Setup/绑定及普通图Verify先完成，缓存FFN仅延后Verify；Setup不能推迟到下游
绑定之后。首轮生成数值通过但释放SIGABRT；最小连接复现于下游NBG的ReleaseGraph。
修复是在缓存前Setup微小普通转换图，缓存结束后以环境0做其首次Verify，刷新SDK状态
再释放该图，随后正式推理/释放；已验证图的重复Verify会提前返回，不能替代这个边界。
作用域对象在异常退出时也尝试恢复，不写私有字段、不引入SDK并行。12/19层整模通过
仅证明这些配置的生命周期；具体SDK重复free对象未定位，事故和工具问题见台账。

证据：`soc-controlled-prefill-cache-restored-20261027/{summary.json,attempt-2/run/}`及
`soc-all-prefill-cache-20261027/{summary.json,capacity-review.json,cache-repair.json,attempt-2/run/}`。
首次失败与原始坏文件独立保留，不以成功覆盖历史。

### 常驻内存为何超过原始权重

原始OPT有331,196,416个独立参数，FP16共631.707MiB；19层组合的SYSTEM峰值是其
2.78倍。源码和清单确定的主要来源是：

| 来源 | 已有证据与数量 |
|---|---|
| M1与M16独立参数表示 | M1完整NBG自带权重，M16又用19个attention NBG、5个完整decoder NBG及自己的WeightBank覆盖近乎一套参数；跨两套执行链未建立共享。权重相关保守应用记账共1352.28MiB |
| 原始FFN与SDK派生表示同时保留 | 19个独立FFN的原始权重304.33MiB，显式live张量合计327.49MiB，设备池694.02MiB；池与显式张量差额366.52MiB接近缓存文件366.13MiB，支持有明显导入/打包副本，但不是逐对象物理分区证明 |
| KV与激活/控制 | canonical FP32 KV输入24MiB由M1/M16共用，两套输出各24MiB，显式KV共72MiB。整模非权重应用峰值记账约98.31MiB，其余含hidden、控制和临时张量，不含不透明NBG内部张量 |

WeightBank按源权重/offset/dtype/shape只保留一份，图间别名使用引用计数；本次4D
预载与消费形状一致，没有额外2D/4D大矩阵副本。参数总数未增加，FP32敏感操作也
不意味着全部权重翻倍。磁盘weights.bin或缓存文件不能再作为完整设备副本直接相加，
Linux文件缓存/RSS和NPU SYSTEM也不是同一测量口径。

应用记账不是精确物理占用：NBG常量界取编译数据、源常量和NBG文件大小的最大值，
例如M1记680.85MiB来自model.data，而实际加载NBG文件607.20MiB、源常量631.90MiB。
因此SYSTEM1756.37MiB减应用各段峰值之和1450.58MiB得到约305.78MiB，只是记账
差额，不能全部称为workspace；各段峰值也未必同时发生。压缩系数、命令、内部激活、
对齐/碎片及临时副本还没有逐项测清。

后续内存优化优先查跨M1/M16参数共享及原始/派生FFN的所有权，潜在空间大于小控制
张量；当前原始权重仍作为图输入被引用，没有安全释放/重绑契约，不能直接删除。
只做了离线所有权核算并复用已有池数据，没有新增上板探针或改精度/权重分类。
证据：`soc-all-prefill-cache-20261027/memory-analysis.json`；原始参数清单及19个独立
FFN实测见前述证据，源码入口为`WeightBank::ensure_loaded`、`build_full_execution`和
`BlockPrefill::build_compiled`。

### 内部KV与attention布局的离线候选

10-27已查明现M1内部先FP32完整Select更新、再完整Cast FP16；host本就仅提交当前槽。
候选改为先缩窄旧缓存、FP16 Select供attention，只输出FP32当前槽，canonical状态保持
FP32。三组真实状态及重建KV的CPU输出按位一致；逻辑数组流量预计少24MiB/token，
不等于实测总线流量或延迟，FP16 Select的次正规值行为及提交源/目标偏移仍须硬件验收。

另一候选用`K×Qᵀ`再reshape替代`Q×Kᵀ`：现NBG确有完整K转置，C128逻辑转置数据
可由256KiB降至2KiB，三组完整decoder共9份CPU输出按位一致。现有9个reshape未有
独立命令层，不优先改view；逐头展开命令数也不能证明物理并行或利用率。
两候选均未转换/上板、未改正式decode；证据：`soc-decode-dataflow-20261027/{kv,attention}/`。

### 仅当前token的KV输出：完整decoder未见收益

10-26保持C128完整decoder的FP32 Select与attention不变，只把公开完整KV替换为
FP32当前槽；输入/默认导出和正式整模链不变。复用第12层及位置17/24/31三组旧参考，
扩容后的CPU hidden/KV逐元素一致，连续提交测试也覆盖首尾槽、覆盖写和FP32位模式。
显式KV输出缓冲由1MiB缩为8KiB，但图内仍有完整KV中间结果，不能视为实际设备内存
或搬运减少同样大小。两版NBG均SH72/TP33/NN33、生成C均6个FCL/2个Select/10个Cast；
现有host本来就只读写当前槽，输出缩小不减少这些调用。

首轮第56次RunGraph出现hang，故障前21份落盘输出正确但完整对照未完成；该轮不计
有效性能。按SoC常设约定，同boot释放/观察后重试，复用两份NBG与三组参考，A/B/B/A
每阶段9次计时，共36次计时、12次预热；每组结束在RunGraph计时之外保存数据。

| 完整decoder RunGraph | 前段均值ms | 后段均值ms | 总均值ms |
|---|---:|---:|---:|
| 完整KV输出 | 18.522 | 18.553 | 18.537 |
| 仅当前K/V输出 | 18.532 | 19.412 | 18.972 |

候选均值慢2.35%，其前后段自身波动4.75%；**未观察到收益，不能据此认定稳定退化**。
只比较同一完整decoder，不是生成吞吐，也不外推24层收益。36份输出通过，所有K/V
与CPU完全一致，hidden最大绝对差0.0078125；子进程2.267秒、45.008秒延迟检查通过，
无新hang、无残留owner，池/GFP释放正常。SYSTEM峰值46.337MiB、Linux最低余量
798.504MiB；两版图同时常驻，不能用这个共同峰值分别计算各版内存节省。
不接整模，临时候选导出实现/测试移入缓存归档，正式入口保持已验配置。
证据：`soc-current-kv-20261026/{retry-2/performance.json,retry-2/run/,static-command-audit.json,export-candidate-source/}`。

### 局部KV回写的收益与限制

复用同一636,651,008字节整图NBG（SHA256 `be581668…8a7a`），固定52入50出、FP32 KV。
每请求完整清零一次；同步RunGraph返回后，只读回当前位置，经
`vxMapTensorPatch(VX_WRITE_ONLY)`写入输入相同位置，Unmap后再Run。不重绑端口、
不使用输入输出别名或独立设备提交图。通用接口为`read_tensor_region_into`和
`upload_tensor_region`，复用缓冲，检查区域/字节数/stride/可写属性，无SDK-copy回退。

| 同轮测量，每请求 | 轻量完整KV，候选前后均值 | 局部KV回写 |
|---|---:|---:|
| 生成wall | 25.729秒 | 17.819秒 |
| 显式上传＋读回 | 8.042秒 | 0.163秒 |
| 每步逻辑KV往返量 | 24MiB | 384KiB |

以候选紧邻的前后完整KV请求均值为基线，实际输出吞吐提升44.39%，首词后decode
提升44.45%；TTFT及生成wall分别减少30.72%和30.74%，显式IO时间下降97.97%。
每步KV往返量缩小64倍。复算见`soc-formal-finalization-20261024/optimization-benefits.json`。
读/写辅助函数次数未减，
且上传API也换为映射写入，不能把收益全部归因字节数。完整KV逐API日志关闭/开启/
关闭对照测得生成wall增加1.56%、decode延迟增加1.51%；仅量化该项日志开销。
普通生成只读选词，最终logits/KV在计时外校验；稀疏RunGraph起止及内核监控仍保留。

候选首词后整步556.798ms，其中RunGraph551.387ms，约99%已在图执行区间；该区间
包含内部搬运及完成等待，不代表纯算术时间。外部IO已低于1%，暂缓设备复制候选，
图内功能分段结果见下节。临时局部KV对照入口已由正式OPT入口替代并退出构建；源码及旧binary
归档于`soc-formal-finalization-20261024/retired-transfer-diagnostic/`，原始端口契约、
数值和性能结果仍保存在`np101-soc-kv-transfer-20261024/`。小内存、单层及通用NBG诊断
继续维护，不以归档为由删除必要的故障定位入口。

### SoC功能分段与驱动等待

10-25以真实OPT权重和decode状态测C64/M1，选第1/12/24层、有效KV长度18/25/32。
每个分段18次计时，跨三组输入轮转；预热、Setup/Verify、显式IO和日志落盘不计入
RunGraph。数值对照通过（atol=0.08、rtol=0.02，最大绝对差0.015625），未用KV保持零。
以下为**分别编译的诊断图延迟**，不是完整模型内部的可加占比，也不是输出吞吐。

| 功能，平均毫秒 | 第1层 | 第12层 | 第24层 |
|---|---:|---:|---:|
| 完整decoder | 30.56 | 24.79 | 25.01 |
| 完整attention，含KV更新/out_proj/残差/LN | 20.56 | 26.11 | 21.01 |
| FFN，含残差/LN | 10.09 | 14.93 | 9.40 |
| QKV投影及query缩放 | 9.39 | 8.74 | 8.73 |
| K/V的ScatterND更新 | 16.51 | 17.20 | 17.21 |
| QK、mask、softmax及PV | 9.16 | 12.02 | 12.47 |
| out_proj、残差及attention LN | 9.36 | 10.48 | 10.92 |

attention四段之和44.41–49.33ms，明显超过完整attention；第12层单独attention甚至
略慢于完整decoder。拆分改变端口、调度和融合，FFN残差ADD还由NN变为SH；部分短段
前后轮均值漂移约24%，不能按这些数字推算整模优化收益。静态NBG中KV更新只有2条SH，
attention核心仍有32条NN、38条SH；命令数量不等于执行时间。KV更新及边界开销值得
对照，不能据此认定单一算子已占满整模时间。旧逐decoder生成记录中24层合计占图时间
约88.8%，仅作定位顺序参考，不套用到当前整图551ms。

另对三个完整decoder做108次短集合，其中27次开启执行线程内存计数，27次前置关闭、
27次后置关闭，其余为预热。采集开启相对前后关闭均值增加1.10%（逐层0.33%–2.01%）；
包装转发和稀疏RunGraph记录各轮均存在，未测其总开销。已核对SDK头文件枚举、结构布局
及对应反汇编，`gcoOS_DeviceControl`下接ioctl：

| 27次采集调用，平均每个decoder | 毫秒 | RunGraph占比 |
|---|---:|---:|
| `gcvHAL_EVENT_COMMIT`，事件提交 | 4.87 | 18.20% |
| `gcvHAL_COMMIT`，命令提交 | 8.29 | 30.98% |
| `gcvHAL_USER_SIGNAL / WAIT`，完成信号等待 | 13.36 | 49.94% |

三者合计99.12%；提交也可能等待锁、设备或内部同步，不能把它算作纯CPU提交或IO。
粗分段中进程CPU只占RunGraph墙钟1.4%–2.1%，未覆盖独立驱动线程或设备执行。
短集合与旧整图记录的完成时间均显著对齐4ms；粗/细分段共504次返回中502次落在
每个4ms周期开头0.5ms内。内核为HZ=250、PREEMPT_NONE、未启用HIGH_RES_TIMERS，
isrPoll=0；FTRACE/KPROBES也未启用。用户态fence轮询分支存在Delay(1)→usleep(1000)，
但本次执行线程采集未命中Delay/usleep，不能据静态分支认定根因。需要区分设备完成、
驱动内部等待与调度唤醒；4ms相位本身不证明整图每条命令都多等4ms。

三轮短集合合计612次RunGraph、数值/释放及各45秒延迟检查通过，未新增故障；更早
两次hang和一次端口错误导致的用户态崩溃独立记录在事故台账。未改正式推理链、驱动
或等待行为，不宣称新增整模加速。诊断源码/构建/NBG身份、数值/时间和反汇编保存在
`soc-functional-attribution-20261025/`，入口为`summary.json`与`wait-summary.json`；
临时实验不加入正式CLI。

后续完整decoder对照使用第12层及相同三组状态，每图18次计时。原图经同工具链
重编后24.764ms，绕过两处ScatterND后12.106ms，减少51.11%；数值最大绝对差
0.0078125，72次调用及释放/45秒检查通过。两图均为版本1.1.35，保持6个FCL、
TP33/NN33，SH由70降至68。候选输入由CPU预先准备更新后KV，同时输出当前K/V以
保留投影计算；输入语义和状态输出形状改变，属于诊断，不能直接用于生成或据此宣称
整模翻倍。冻结旧图在本组为30.758ms，图版本及放置次序不同，不据此宣称重编收益。
证据`soc-fullgraph-kv-attribution-20261025/`的`summary.json`、`kv-control-provenance.json`
及`kv-command-audit.json`。后续可执行的Select替换及整模结果见上文。

正式24层整图随后完成三次相同请求（各17输入、16输出，共96次RunGraph），采用
内存计数关闭/开启/关闭；模型、权重和执行逻辑未变，仅临时链接采集包装。每个消费的
token仍一次RunGraph，未改成逐decoder提交。中间请求首token后15步的实测如下：

| 每个decode步骤的SDK调用 | 次数 | 毫秒 | RunGraph占比 |
|---|---:|---:|---:|
| 事件提交 | 1 | 4.496 | 0.82% |
| 命令提交 | 1 | 41.488 | 7.53% |
| 完成信号等待 | 8 | 496.973 | 90.20% |

RunGraph均值550.997ms；前后关闭计数为551.749/551.495ms，计数开启相对两者均值
为−0.11%，未见可分辨的采集增量。执行线程CPU时间占1.55%，不包含其他驱动线程。
等待包含尚未完成的设备工作，不能解释为纯同步损耗或IO；提交调用也可能阻塞。
整图提交占比远低于独立decoder，不能将短图开销乘24。三次请求平均TTFT9.466s、
实际输出吞吐0.898token/s、首token后decode1.795token/s，与已验基线一致；准备、
Verify和事后KV读回不计入生成时间。数值、末层KV/reset、释放和45秒检查通过，
无新增设备故障。证据同目录`fullgraph-linked-run/`及`summary.json`。

内核反汇编另确认`gckCOMMAND_Commit`申请异步槽位的等待分支可调用
`gckOS_Delay(1)`，后者调用`schedule_hrtimeout`；区别于未命中的用户态Delay。
当前没有进程内核栈接口，未证明本次实际走该分支，4ms节拍根因仍未确定。
首次LD_PRELOAD采集的用户态崩溃及替代方式见事故台账。

### 单层加载与离线命令审计

官方生成加载器和仓库包装均用一个NBG FILE节点、5入3出、graph版本1.1.34，执行
Setup(false)/Verify/同步Run；实际tensor均为8个。`.nb`分支端口为hidden/K/V，
另一分支顺序不同，必须按部署契约绑定。图先于Context释放，路径所有权已核对。
RunGraph下接vxProcessGraph，NBG执行含gcoVX_Replay→gcoHARDWAREVX_ReplayCmd，
仍有运行时提交/完成等待；这是静态反汇编证据，不是逐内部函数动态追踪。

已验单层NBG `653469ac…c7259`含6个FCL、2个MATRIXMUL；容器记录为
START1/TP33/SH70/NN33。attention的两次MATRIXMUL各展开16组SH+NN，FCL共28个TP。
这揭示与纯手写MatMul的路径差异，但不能证明利用率或定位故障算子；EP最终混合方案
已采用NBG投影/FFN。证据见`np101-soc-decoder-artifact-audit-20261024/`。

### SD完整性与恢复边界

- 网络现状见本节配置。原SD曾有至少70个inode异常，已修复并从匹配原件恢复关键文件；
  demo可能缺失，早期partial镜像不是完整备份。
- 根分区已从2GiB扩至29.118GiB，可用约25.353GiB；保留p2起点6144及UUID，修正备GPT，
  离线fsck通过且扩容后已启动。容量和文件系统检查通过不等于介质可靠。
- 六份异常NBG已恢复，待用SDK/模型身份逐项校验。USB读卡器和板端原生SD控制器均
  复现同一4KiB块O_DIRECT三读不同，期间无写入；不能仅归咎读卡器。SD健康未确认，
  对hang的贡献也未证实，逐次偏移/哈希留原始证据，不继续无目的扫描。
- 换卡不能保证缺失数据恢复，旧备份不能直接当作可信原件。主机OPT checkpoint、
  weights/index、24份decoder及93份混合M1/M16产物约2.81GB已全量验哈希，可重部署。
  现有官方镜像完整性通过，但为4月内核，与当前7月内核及关键SDK不同；等价重装仍需
  匹配版本的厂商镜像。重建后迁移已核对配置、校验部署文件，并做有界启动/数值验收。

### 原始证据入口

路径均相对`.cache/runs/`；目录内保留summary、逐次日志、脚本、文件身份和完整数值。
设备事故的精确时序、boot ID、原因边界及后续规避只在[事故台账](np101-recovery-incidents.md)维护。

| 内容 | 目录 |
|---|---|
| 网络/启动与SDK准备 | `np101-soc-network-config-20261021/`、`np101-network-finalization-20261021/`、`np101-soc-platform-inspection-20261022/`、`np101-soc-build-memory-20261022/` |
| 内存布局与容量 | `np101-soc-memory-layout-20261022/`、`np101-memory-map-audit-20261023/`、`np101-soc-capacity-20261022/`、`np101-soc-pool-capacity-20261023/` |
| 旧池计算及整模初始化 | `np101-soc-graph-validation-20261022/`、`np101-soc-virtual-compute-20261023/`、`np101-soc-full-model-20261023/` |
| 新池配置及写读 | `np101-soc-contiguous-config-20261023/`、`np101-soc-expanded-pool-20261023/` |
| 独立绑定递进及完整生成 | `np101-soc-independent-decoder-chain-20261024/`、`np101-soc-opt-generation-20261024/`、`np101-soc-fullgraph-generation-20261024/` |
| 轻量基线及局部KV回写 | `np101-soc-kv-transfer-20261024/` |
| SD扩容、启动及双路径完整性 | `np101-soc-sd-expand-20261023/`、`np101-soc-storage-boot-20261024/`、`np101-soc-sd-integrity-20261024/` |

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
  OPT partitioned及Qwen的INT32选词固定走Map/Unmap，映射失败直接报错；
  OPT另用`token_readback_transport`单列报告，不与粗验的`readback_transport`混淆。

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

### NBG投影链接M16 prefill（C64及有界C256已验）

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
10-16只接受该C64有界负载；10-20新增C256边界见下文，不外推新NBG组合C128、
长期稳定性或TTGT组合。

直接FCL准备异常已停止验证，CPU权重压缩抓栈与中止记录见事故台账；当前所有板卡
实验子进程均限300秒。固定binary SHA256
`45b2683b1e5074c10b241e3a3007901e88e4da5e2e6fef5f28da579f995e9dda`；证据
`opt-fast-prefill-20261016/summary.json`，含包身份、原监督中断记录、逐项CPU/编译审计、
完整内核和补验结果。后续只改默认NBG选择和如实报告backend，已做CPU入口检查，
没有改动已测显式NBG模型计算。

### NBG速度差异与选词读回

10-16固定C64/M1、17输入/16输出、各三请求，历史与当前的M1 NBG、权重和五库身份、
124张常驻图及应用预算相同，但实际输入不同。关闭审计及提交`eec8af1`源码对照仍慢，
后者不含M16和对话改动，降低了这些改动作为主要原因的可能性。采集轮45个decode
步骤中RunGraph为894.32ms/token、占墙钟98.93%，显式IO为8.13ms/token、占0.90%；
这不是IO瓶颈的证明。后置关闭轮发生4字节EDMA，未完成当次采集扰动对照；独立事故
见台账，原证据在`opt-decode-regression-20261016/`。

10-20用户确认板卡断电恢复后，固定v105及相同NBG，补齐历史原始token重放。
SDK-copy使用冻结修改前binary；候选只把`PartitionedGraphModel::Impl::predict`的
单INT32读取改为已有`MappedHost`路径，按stride拷贝后Unmap，不增加图、权重或端口重绑。
恢复检查通过后采用映射读取；不再调用该处的`vsi_nn_ConvertTensorToData`，
但映射内部仍可能用EDMA，不能声称历史故障已修复。

| 固定17输入/16输出 | 请求数/输出数 | 实际输出token/s | 首词后decode token/s | 平均TTFT s |
|---|---:|---:|---:|---:|
| 10-14历史全NBG，原文本 | 3/48 | 1.73116 | 3.22732 | 4.595 |
| 10-16冻结程序，无审计/计时，不同文本 | 3/48 | 0.65933 | 1.27225 | 12.477 |
| 10-16修改前源码，同10-16文本 | 3/48 | 0.61912 | 1.19627 | 13.304 |
| 10-20 SDK-copy，重放历史原token，无审计/计时 | 3/48 | 1.15193 | 2.18849 | 7.036 |
| 10-20映射M1，采集前后关闭轮合计 | 2/32 | 1.21641 | 2.35418 | 6.782 |
| 10-20映射M1＋M16，编译完整性审计 | 1/16 | 1.95893 | 2.34884 | 1.782 |

10-20吞吐为输出数/生成wall，排除初始化、粗KV检查和日志；不是输入处理率。
M16未采执行区间，仅有Verify审计及执行完整性门禁；SDK计时关闭。相对同日映射M1，
输出率提高61.04%、TTFT缩短73.73%，decode率基本不变；样本少且存在运行顺序漂移，
不作为统计估计或映射相对SDK-copy的确定提速。映射M1两请求共32输出/26.30690s；
M16为16输出/8.16771s，初始化分别约4.04–4.37/13.06s，未混入吞吐。

历史相同输入复测仍低于3.22732decode token/s，输入内容不能单独解释差异；历史快
binary没有保存，当前不能完全排除更早实现及运行状态差异。相同输入的SDK-copy首请求
主机user+system约1.038s，历史约0.386s；并非全部墙钟，更不代表板载CPU/NPU利用率。

映射候选同binary/原token完成关闭→内存汇总→关闭，每轮一个请求。
生成wall为13.54640/13.57371/12.76051s，decode为6.55325/6.60233/6.19004s。
开启轮相对关闭均值分别+3.20%/+3.62%，但关闭轮自身漂移−5.80%/−5.54%，
不能精确分离采集开销。开启轮共3904图调用；其中15个首词后decode步骤的1845次
`vxProcessGraph`占墙钟99.59%，`gcoOS_WaitSignal`占图时间93.76%、
`vxoGraph_Submit`占2.31%。`post_submit_wait`仅覆盖走Submit分支的调用，整体为32.07%；
不把它当所有NBG的完成等待。各区间嵌套，不能相加，也不能由等待推断物理IO或根因。
这与早期另一执行链的99.536%/0.255%不是同一负载，不能替换使用。

该轮每decode token的24层QKV/attention/out_proj/attention收尾/融合FFN合计为
83.57/67.27/54.60/58.52/128.43ms；head/selection为18.26/24.24ms。
单层合计14.22–18.06ms，零基layer23为15.45ms，未见固定第23层异常。
本轮六进程/八请求/128输出均通过适用的选词、末层KV/未写槽、编译完整性、预算、
释放和各45秒门禁，无新EDMA/hang。M1/M16分别124/194图及238/583 extension完整，
const/nonconst峰值仍为657,302,149/77,928,016B和1,035,583,855/410,013,944B。
该轮只补验C64范围；随后C256工作见下节。未重跑全图或逐decoder C128，也未验长期稳定性。
汇总、实际请求、冻结binary/源码、完整内核、采集对照及成本模型在
`opt-ep-completion-20261020/{summary,tracing-comparison,execution-attribution,prefill-alignment-cost}.json`。

同日后续核对`eec8af1`至当前源码，Context、tensor、GraphBuilder及mixers实现逐字节
相同；M1仍使用固定QKV/core/out/残差/FFN及单head/selection。此前已测的提交源码对照
也慢，不能把M16或会话改动当作必要触发条件；历史快binary及完整源码状态缺失，仍不能
完成严格二分或排除更早实现差异。
PCIe设备`0000:af:00.0`位于NUMA1，IRQ284有效CPU为36。固定已验映射binary、
历史17输入/16输出、NBG及五库，每侧三请求，无采集，仅改测试进程CPU affinity：

| 运行顺序 | 输出token/s | 首词后decode token/s | 平均TTFT s |
|---|---:|---:|---:|
| 正常调度，前置 | 1.168253 | 2.225717 | 6.95626 |
| 固定NUMA1的CPU20 | 1.232242 | 2.324062 | 6.53025 |
| 正常调度，后置 | 1.149025 | 2.207234 | 7.12901 |

两个正常轮合并decode为2.216437token/s；固定CPU观察到+4.86%，仍比历史3.227321
低约28%。每侧45个decode步骤的进程CPU user＋system为1.336/1.109/1.443s，
不是全部墙钟。主机放置有小幅影响的线索，未证明NUMA、迁移、频率或IRQ中的具体原因，
也不能解释主要回退。没有修改系统affinity、IRQ、governor、时钟、驱动或SDK；仅有这一组
有界对照，不新增自动绑核策略。根因继续待匹配的底层事件或厂商信息，不盲目重跑FCL。
源码审计、进程调度范围及完整结果见`opt-context-alignment-20261020/summary.json`、
`source-and-host-audit.json`及各`*-scheduling.json`。

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
凭处理token数减少推断延迟降低。

10-20先用旧记录拟合成本，随后实现基于本进程实测的历史重算，不采用旧时段的固定阈值。
完整前缀及token范围校验通过、历史临时标为无效后，才允许回退少于16个已消费token；
原buffer不变，已知ID重新prefill，因果mask隐藏旧未来槽，末预测只消费一次。
比较无head prompt步骤的最小M1/最大M16成本，仅在增加整块且节省成本时回退；
没有足够样本时保持原路径。仅为M16＋NBG QKV/out的编译FFN组合启用；
调度成本含必要控制IO，不作独立算子计时；不加权重、图或CLI参数。
任一执行失败使历史失效；CPU契约覆盖错误前缀零状态改动、重算内容和失败后拒绝续接。

| 10-20同三轮/每侧48输出，含reset的响应口径 | 平均TTFT s | 输出token/s | 首词后decode token/s |
|---|---:|---:|---:|
| 冻结映射版本，原保留KV策略 | 8.50476 | 0.966405 | 1.863021 |
| 自适应历史重算＋保留KV | 6.70001 | 1.089403 | 1.878067 |
| 候选版本，清空后完整重放 | 10.37047 | 0.875394 | 1.897048 |

第二轮重算位置48–50的3个已知token，将27个M1改为1个M16＋14个M1，TTFT由
13.58737降至8.07479s；第三轮按保守实测成本保持原路径。相对原保留KV策略，
平均TTFT减少21.22%、输出率提高12.73%；相对清空重放，分别减少35.39%/提高24.45%。
decode基本不变，不把不同boot的旧时段性能差异归因于本次调度。
三侧选词/末层KV/未写槽、预算/释放及各45秒门禁通过；原始CPU轨迹复用10-16记录。

两侧const/nonconst峰值均1,035,620,719 / 535,651,064B，释放归零；194图、2918 outer
operation、583 extension完整。原NBG与native包字节不变，派生manifest仅声明C256；
10-16未扩展普通生成CLI或验C256末槽；10-20补验范围如下，仍未验其他对话、长期稳定或更长容量。

**故障范围**：最初单进程串接六轮在重放第二轮发生4字节EDMA，结果不入上述表。
用户明确授权忽略这一已记录事件后，拆为两个三轮进程，执行/数值/预算/释放及各
45秒均通过；16:58:25.697036、17:02:40.443150完整日志无新故障/owner。没有断电，
只在本次诊断中豁免16:48:18.926707这一旧事件，恢复标记与事故证据保留；不能据此
宣称EDMA已修复。证据`opt-conversation-20261016/comparison.json`及事故台账。

### C256长输入、隔离与容量边界

10-20使用相同NBG/M16组合，241/3/241输入各生成16 token，长请求消费到256槽。
48个选词及末层KV与独立CPU参考一致，重复长请求KV逐字节一致，短请求未写槽按位零，
两次满容量step均在零IO/零图执行下拒绝；close拒绝、释放归零及45秒门禁通过。
194图/2918 operation/583 extension完整，const/nonconst峰值
1,035,620,719/535,651,064B，各低于1GiB；没有改变NBG、权重分类或绑定。

两个241输入请求共32输出/57.80994s=0.553538token/s，平均TTFT21.02312s，
首词后decode1.903106token/s；不把中间短请求混入长请求吞吐。
长请求输出为`Alice lives in Paris. Bob lives in Rome. Alice owns a red bicycle.`，
与本次输入事实及CPU轨迹一致；不据此泛化模型问答能力。
普通生成入口开放显式声明C256的NBG包及block16，默认仍为C64；未开放C256直接FCL、
全图或逐decoder候选，也未外推原生C128 M16配置。
调度/长输入的冻结binary、CPU参考、各请求计数及内核证据在
`opt-context-alignment-20261020/`；后续decode主机放置对照也归档于此。

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

### SoC的VPD采集

10-26确认板端匹配的RISC-V `libOpenVX/libGAL`包含采集实现；x86 VivanteIDE仅用于
离线查看，并不要求板端运行x86程序。用已验C128第12层完整decoder、位置17/24/31
三组状态做关闭→开启→关闭，每轮3次预热、18次计时，显式IO/准备/应用日志在计时外；
SDK自身的采集、同步和打印成本计入开启轮。
首轮关闭采集出现hang，释放/45秒观察后同配置重试通过；失败轮不纳入性能对照。

| 完整decoder RunGraph | 关闭前 | 开启 | 关闭后 |
|---|---:|---:|---:|
| 平均延迟ms | 18.551 | 2338.022 | 18.328 |
| 平均进程CPU时间ms | 0.285 | 43.920 | 0.281 |

开启延迟为关闭均值的126.79倍，关闭前后漂移-1.20%；不是生产吞吐回退。
输出/未写KV槽、资源释放及45秒检查通过，原恢复标记保留。30,326B VPD完整解析为
21条整图记录×164字段，与21次RunGraph对应；没有NN/TP分组，仅有整图elapsed，
没有算子或引擎的绝对起止时间。18条正式elapsed合计41.735秒，应用RunGraph合计
42.084秒；VPD不覆盖全部调用外围成本，也不能还原设备内部流水。

HI读/写按匹配头文件的8字节单位与SDK打印核对，约18.90/1.01MiB每次decoder；
尚未校准覆盖范围，不能拆成权重、KV、工作区或host IO。其它非零字段保留原始ID和
通用头文件名称，不把图形字段名直接等同NP101内部单元。HI idle/total约98.69%，
但这来自已被放大127倍的采集过程，**不能解释为正常推理时NPU闲置98.69%**。

RISC-V反汇编确认Begin包含Flush，Enable/End路径包含额外Commit、counter收集和
fence等待；End在收集后才取结束时间。CPU时间增长远小于墙钟增长，支持采集引入
大量等待的判断，未动态定位到其中某个内部函数。`VP_ENABLE_PRINT=0`仍未关闭
SDK打印。当前采集适合有限的图级对照，不能代替未采集运行的性能，也未解决内部
搬运/核利用率归因。

整模沿用同一C128、24层M1整图NBG和正式binary。鉴于上述开销，缩为1输入/2输出，
不构建或执行M16；三轮相同请求，不能外推长prompt或持续吞吐。初次短请求因遗漏
将head策略改为`every-token`在SDK加载前被拒，无设备故障；原参数与修正记录保留。

| 整模短生成，不含初始化 | 关闭前 | 开启 | 关闭后 |
|---|---:|---:|---:|
| 2个输出的生成wall秒 | 0.77997 | 112.18685 | 0.73958 |
| TTFT秒 | 0.41609 | 56.00690 | 0.37437 |
| 实际输出token/s | 2.5642 | 0.01783 | 2.7042 |
| 首词后decode token/s，仅1个间隔 | 2.7482 | 0.01780 | 2.7382 |

生成耗时放大147.66倍；关闭前后wall漂移-5.18%，主要差异在首词，首词后间隔仅差
约0.37%。三轮选词均为78/86、末层KV逐字节相同，数值/未写槽/传输/释放及45秒检查
通过，无新增故障。模型就绪分别97.961/89.746/86.023秒，包含前置校验和Setup/Verify，
不计入上表；Python部署包校验另计。最大native子进程203.008秒，未延长期限。

整模VPD仅3,042B、2条整图记录×164字段，与decoder字段完全相同，没有新增内部
事件；elapsed合计112.114秒。报告读/写约476.98/24.29MiB每次整图，不能拆分来源。
整模平均elapsed约为单层的24.17倍，说明此次采集成本随图内容显著增加，不能按每次
RunGraph的固定开销估算。与此同时HI total均值与单层接近，32位计数的时钟域/回绕
未校准，进一步限制idle比例和带宽推导；不根据这些数字判断生产时的NN利用率。
本轮不改正式推理链或默认关闭采集策略，不扩大原样VPD试验。

证据：`soc-profiler-20261026/{decoder-summary.json,model-summary.json,capture-comparison.json,
decoder/,model/,disassembly/,inspect_vpd.py}`。

随后复用相同C128完整decoder和三组状态，对profiler自身做链接包装计时。VPD始终
开启，新增内存计数关闭/开启/关闭，每段6次正式调用及3次预热，共27次；没有新增
SDK调用、同步或计时内文件输出。三段RunGraph为2331.900/2351.278/2348.582ms；
计数增量+0.47%，关闭轮漂移+0.72%。这是同包装下的增量，不是包装总开销测量。

| 开启计数轮，每次完整decoder；互不重叠 | 毫秒 |
|---|---:|
| profiler Begin，含1次Flush | 9.839 |
| 图执行期间140次`gcoVX_Flush` | 2339.103 |
| profiler End，含读计数、输出和Disable | 0.841 |
| 其余 | 1.489 |

执行期间Flush占99.48%；其嵌套命令提交/事件提交/USER_SIGNAL类ioctl分别约
1500.876/734.222/100.918ms，不能再与父函数相加，也不能把提交视为纯CPU工作。
全部计时函数exclusive之和与RunGraph对应；未拦截的内部调用留在父函数自身时间。
未命中`gcoBUFOBJ_WaitFence`或用户态Delay，只覆盖当前执行线程及已包装入口。

匹配库的`vxoBinaryGraph_Run`在操作循环内按context两个标志决定执行`Flush(1)`
（0xdd880–0xdd8d0），与高频同步观测吻合。此次未采返回地址及标志实际值，不能把
140次都指定为该调用点，也不证明正常关闭VPD时的提交数或驱动内部等待根因。
优先查清这些标志与公开控制方式，再考虑采集专用的批量提交；仅减少落盘或把End
移到集合末尾无法解决已测大头，不直接删除必要Fence/完成同步。

EP与SoC的`gcoVX_Flush`均为FlushPipe→Commit→按参数Stall；Stall提交完成事件并
等待信号，Flush墙钟包含未完成设备工作，不能全算同步开销。两端NBG操作循环均按
context的0x2df1ac/0x2df118条件调用`Flush(1)`，EP并非没有此路径。缓存SoC驱动的
`gckCOMMAND_Commit`及其WaitLink分支包含锁、WaitSignal、WaitFence和Delay候选；
静态存在不证明本次命中。采集轮进程CPU约45ms、RunGraph约2351ms，不能解释为
提交线程做了2.3秒CPU计算，也未覆盖独立驱动线程。后续需比较同负载VPD开关的调用
数量、CPU/墙钟及等待子类型；已有EP FFN约3.42倍与SoC decoder约127倍来自不同负载。
离线调用路径证据在`profiler-flush-semantics-20261026/`，本次未新增板卡试验。

27份hidden/KV输出跨三段逐字节相同，CPU参考最大误差0.0078125；子进程64.358秒，
释放及45.023秒观察通过，无新增内核故障，原恢复标记保留。正式推理链未改。
源码、构建、反汇编、原始采集和分层账目在`soc-profiler-overhead-20261026/`，入口
`summary.json`；这些毫秒值仅归因VPD开启时的失真，不作为正常模型耗时或输出吞吐。

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
代码精简采用离线构建和CPU检查，硬件验收按各节日期与范围区分；旧命令不是当前操作说明。
