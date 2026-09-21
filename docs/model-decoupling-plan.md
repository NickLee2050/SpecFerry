# 已有实现全面解耦施工清单

更新日期：2026-09-21。状态：R1–R8 已实施，host 检查与约定的设备数值回归通过。
审查和改造范围为 **S0–S8 已完成的全部内容**，包括外围工具、CPU 基线、SDK、
权重准备、算子与状态验证、DeltaNet、Attention、MLP 和 decoder 切片。

## 范围与验收边界

本次拆开“可复用机制”和“Qwen 模型契约”，保留已有计算语义及验证能力。
已有通用模块直接保留；模型专属逻辑集中到明确的 Qwen 模块；只有两者混在一起的
部分才拆分。测试样例的固定输入、硬件限制和模型配置值不作同等处理。

设备验收上限仍是已经实现的 mixer、完整单层和四层 decoder 切片。
既有完整 CPU 参考属于本次维护范围，但不新增完整设备推理。
OPT 下载/CPU 导入结果保留，不要求新增 OPT decoder、PyTorch bin 导出或完整生成。
S9 及之后按原施工计划继续，不在本清单中施工。

## 全面检查结果与处置

下表记录改造前的全面审查结论及已执行的处置；“保留”表示未新增抽象层。
实施后的模块边界、fixture 格式和复现入口见 [组件验证说明](../tests/np101-components.md)。

| 已实现内容与文件 | 检查结果 | 本次处置 |
|---|---|---|
| Conda、requirements、格式脚本、`.clang-format`、`ruff.toml`、环境采集 | 依赖与工具规则独立于模型；NP101 库名、设备路径属于平台配置 | **保留**。不改 Python 3.12、格式规范、系统配置或采集机制 |
| 根/测试 `CMakeLists.txt` | 已分离不含 SDK 的 `np101_data` 与 SDK context；后续计算库按具体组件链接 | **小改**。随职责提取调整 target 和依赖，保留编译数据库、SDK 路径配置和 host-only CTest |
| `scripts/download_models.py`、`range_download.py` | 已有 ModelSpec、独立传输、格式选择、锁定 revision、摘要验证；启用列表是项目策略 | **保留**。继续只启用 OPT-350M，不将下载配置扩建成模型框架 |
| `reference/checkpoint.py` | 文件校验/safetensors 解析与固定 Qwen repo、revision、text/vision/mtp 分类混合 | **拆分**。通用文件检查与张量盘点独立；模型分类、名称映射和 revision 检查留给 Qwen，见 R2 |
| `reference/model.py`、`runner.py`、`trace.py`、`tolerances.json` | 官方类、chat template、精度补丁、层 0/3 hook 和参考场景明确属于 Qwen；数组保存、比较与报告部分可复用 | **局部拆分**。保留 Qwen 参考算法/场景，提取公共工具和调用边界，不建设通用生成器，见 R2/R8 |
| `native/np101/context.*`、`tensor_spec.*` | SDK 生命周期、dtype、shape 和溢出检查无模型绑定 | **保留**。不增加设备后端接口或改写所有权机制 |
| `tensor.*` | tensor 创建/传输/retain 通用；只有 `bind_hidden` 固定 FP16 `[1024,1]` | **局部参数化**。修改绑定契约，保留其余机制和 SDK 例外说明，见 R5 |
| 厂商 `demo/`、卷积/ReLU/Pool 回归及脚本 | 独立 SDK 基线，固定形状和误差预算是用例定义 | **保留**。不模型化、不为减少重复而替换成待测 DLM 算子实现 |
| `validation/fixtures.py`、native `case_file.*`、`operators.*`、`op_check.cpp` | 已按文件描述任意测试 tensor/node；属于 SDK 能力探针，不是模型执行器 | **保留协议与职责**。不升格为生产推理 IR；不强迫原始 SDK 探针依赖生产算子 |
| `operator_cases.py`、`extended_cases.py`、`reference_cases.py`、算子检查脚本 | 基础生成函数多已参数化；模型尺寸目录、`layers.0/3` 和 trace 名称映射仍是 Qwen | **局部拆分**。通用小图和 Qwen 测试配置分离，见 R8；已有 Gather/head/Argmax 探针仍仅是探针 |
| `export/schema.py` | 严格规定完整 Qwen 权重集合、24 层结构和 dtype | **归入 Qwen 契约**。保留已验证 checkpoint 的严格检查；不把限制复制到通用打包器，见 R1/R3 |
| `export/weights.py`、`verification.py`、导出/校验脚本 | 对齐、分块、摘要已通用；dtype 策略、固定模型身份和 embedding/head alias 混入打包/校验 | **拆分**。物理格式、模型契约和精度选择分开，见 R3 |
| `native/np101/weights.*` | 有界读取/完整性检查通用；名称必须以 `model.` 开头，`find` 特判 LM head | **局部解耦**。移出名称语义和 alias 特判，保留安全边界，见 R3 |
| `export/memory.py`、分配诊断及脚本 | 权重求和通用；18/6 层、状态副本数、KV/词表尺寸和原生诊断状态表写死 | **拆分数据与机制**。模型提供资源条目，通用代码计数/执行；旧复现配置保留，见 R3 |
| `delta_net.*` | 基础构图与 Qwen 投影、维度、权重路径、递推及双图调度混合 | **拆分并参数化**，见 R4/R6；保留 FP32 修正和当前两图策略 |
| `kv_cache.*` | 单父缓冲/slot view 可复用；KV 头数、head_dim、字节数和容量常量绑定当前模型 | **参数化**，见 R5；不改存储方案 |
| `attention.*` | 基础构图、Attention 核心与第 3 层权重、Q/K norm、RoPE、gate 混合 | **拆分并参数化**，见 R4/R6 |
| `decoder.*` | 重复构图、RMSNorm/SwiGLU、模型层排列和组执行管理混合 | **拆分并参数化**，见 R4/R7 |
| Python mixer/decoder 验证与 native 对应测试 | 固定状态形状/层号/字节数；Attention/Decoder 为比较文件而依赖 DeltaNet 模块 | **局部拆分**。公共比较工具独立，模型参考和测试配置显式化，见 R8 |
| `validation/device.py`、`capabilities.py`、设备检查 CLI | 设备锁、超时、恢复标记、日志和硬件证据判定已独立；CLI 默认模型与容差选择需要明确归属 | **保留调度机制**。只改参数/配置接线，不引入测试框架或统一万能 runner，见 R8 |
| 共享诊断、历史内存/状态报告、README/TODO/测试说明 | 包含仍有效的限制和复现证据，不是应删除的模型耦合 | **保留证据并同步说明**。不重启已退役实验，不因重构关闭硬件验收代办 |

## 目标依赖边界

- Python 公共文件/数据工具不导入 Qwen 官方类或 Qwen 校验策略；
  `python/specferry/models/qwen3_5/` 持有模型配置、权重映射、精度策略和专属参考逻辑。
  现有脚本保持薄入口，公共参考/校验工具由调用方传入策略。
- `native/np101/` 保留 SDK 与存储；`native/np101/ops/` 提供实际复用的构图函数和
  计算片段；`native/models/qwen3_5/` 负责模型组合。CMake 依赖只从模型层指向公共层。
- 普通参数结构、TensorSpec 和权重引用足够表达当前需求；不建设统一模型描述语言、
  后端注册系统或一算子一类的包装体系。目录移动随职责拆分完成。
- 通用层依然允许存在有依据的 SDK/实现限制，例如读取块上限和已验证 shape 范围；
  模型尺寸和层顺序由适配代码提供，测试固定值留在具名用例中。

以下 R1–R8 是本次重构顺序，不替代原 S 阶段编号。

## R1. 固定模型边界、组件参数和兼容要求

依赖：无。

- [x] 将 Qwen 身份/revision、配置解析、权重命名、tensor 分类、层类型列表及
  `export/schema.py` 的严格 checkpoint 契约集中到 Qwen 模块。
- [x] 区分“受验证的 Qwen checkpoint 配置”和“公共组件接受的参数”：前者继续严格
  限定原 checkpoint，后者允许独立的小尺寸合成用例，不以放宽模型校验实现复用。
- [x] 定义小型组件参数：hidden/intermediate、Q/KV 头数/head_dim、rotary_dim/theta、
  DeltaNet key/value 维度/卷积宽度、epsilon、dtype/舍入边界、容量和层列表。
  从配置/权重形状推导 shape、slice 偏移及字节数。
- [x] 确定 Python → native 的组件参数传递：利用现有部署/测试元数据，只增加实际
  使用的字段和必要读取代码；缺失或不支持的配置明确报错，不默认为 Qwen 尺寸。
- [x] 保持旧权重包和历史 fixture 可解释，保留 Qwen 旧 CLI 默认路径与回归入口。
  若必须修改 fixture 字段语义，同步版本检查；不静默改变旧文件含义。

交付/验收：Qwen 配置入口、组件参数结构、旧入口兼容说明；原配置得到相同的
shape/字节数，非法形状、头分组和权重错配在对应组件分配/建图前失败。

## R2. 解耦 checkpoint 检查与既有 CPU 参考

依赖：R1。无需设备。

- [x] 从 `checkpoint.py` 提取路径约束、摘要/大小验证、safetensors header/index
  解析和范围检查；公共盘点接收身份契约，模型包装函数补充分类，不内置 Qwen repo/revision。
- [x] 保留所有重复 tensor、缺失 shard、重叠/越界 payload 和未分类权重检查。
  当前仅维护已实现的 safetensors 盘点路径，不新增 OPT bin 导入到这条流水线。
- [x] Qwen 官方模型构造、text-only 筛选、tied weight、RoPE 恢复、chat template、
  `precision()` 和 functional hook 留在 Qwen 参考模块，继续使用官方计算实现。
- [x] 数组序列化、有限值检查和比较等工具独立；通用文件/导出工具不能为获取
  摘要或容差而导入整个 Qwen 模型运行器。
- [x] trace 的层选择/名称映射由 Qwen 测试配置提供，原层 0/3 场景保留默认值。
  现有 CPU prefill、逐 token、teacher forcing 和短生成流程维持行为，不改写为
  支持所有模型的生成框架。
- [x] 保留参考来源、源码 hash、精度说明及冻结阈值；Qwen 的两种精度模式不作为
  其他模型的默认策略。

交付/验收：通用盘点/比较工具与明确的 Qwen 参考入口。小型伪 checkpoint 能独立
校验；原 Qwen 身份拒绝测试、固定输入参考与关键 trace 对齐仍通过。

## R3. 解耦权重包、内存预算及分配诊断

依赖：R1–R2。格式与预算检查无需设备。

- [x] 将导出拆为“模型选择 tensor/目标 dtype/alias”与“按清单有界读取、转换、
  对齐写出及记录摘要”。保留现有 64 字节对齐、8 MiB 有界读取和逐 tensor 校验。
- [x] `convert_bytes` 显式接受目标精度；保留 Qwen BF16→F16、原生 F32 保留策略，
  补齐公共 F16 原样写入。只支持本轮明确验证的转换，不引入新的精度优化。
- [x] 将 `verify_weight_pack` 的物理格式检查与 `verify_export` 的模型契约检查分开。
  generic 校验不强制存在 Qwen embedding；alias 仅允许直接指向已有物理记录，
  拒绝悬空、覆盖物理名称和链式/循环 alias。
- [x] C++ `WeightStore` 按物理名称精确查找，移除 `model.` 前缀要求和 LM head 特判；
  名称仍需满足索引语法。alias 由模型适配器解析，生产算子接收显式 WeightRecord/
  tensor 引用。优先保持 weights.index v1 不变，旧 Qwen alias 由适配器兼容。
- [x] 独立导出验证器继续以 PyTorch 转换结果检查字节，不调用待测转换函数计算
  expected；模型身份、选择规则和目标 dtype 由 Qwen 契约提供。
- [x] `memory_budget` 汇总物理权重和具名资源字节数；模型侧计算副本数。Qwen 的
  18/6 层、双图副本与既有 head 预算移到 Qwen 资源描述，保留原估算场景。
  区分历史全模型预算、所选切片预算和实测分配；未知 SDK/workspace 继续标为未知。
- [x] 分配诊断的固定状态列表移为显式具名测试配置；常量/可写权重加载与回读流程
  保留。原 Qwen 故障复现配置的 shape、数量、顺序原样保存，不冒充当前 KV 执行布局。
  新切片诊断使用实际组件规格，不再在分配器中写死模型状态。

交付/验收：模型无关的权重读写/完整性检查和预算汇总；旧 Qwen 包可读且导出 payload
保持一致，小型非 `model.*` 名称/无 embedding 包及 F16 原样字节测试通过。
常量/可写分配只作有界小样本回归，不重新触发已知约 1 GiB 的全量失败实验。

## R4. 提取公共 SDK 构图和基础算子

依赖：R1/R3。

- [x] 从 `AttentionGraph`、`StepGraph`、`DecoderGraph` 合并重复的 tensor/常量/node、
  reshape/slice、转换、图 IO 声明和参数数组持有机制；保留 Context/Graph 实现。
- [x] 提取现有 MatMul/无 bias 投影、逐元素、归约、激活与归一化函数；参数化 shape、
  轴、转置和精度。集中核对逻辑维度与 SDK 轴序，不增加当前未用的算子路径。
- [x] 明确区分 RMSNorm 和基于平方和的 L2 归一化。`1 + weight` 由调用方显式指定，
  保留原 FP32 加法及舍入位置，不在 FP16 权重上预折叠。
- [x] 保留现有投影分块和有界读取策略；实现限制与模型参数分开表达，拒绝尚未支持
  的组合。节点参数数组活到图释放之后，不覆盖 `vsi_nn_AddNode` 初始化的私有状态。
- [x] 构图函数向调用方已有图添加节点；代码拆分不增加执行图、host 中转、复制或
  权重加载。公共库不依赖任何模型模块或测试 case 协议。

交付/验收：现有三个计算模块实际调用公共算子；小尺寸投影、归一化和轴/布局对照通过。
原始 SDK 能力探针独立保留，公共函数另由实际调用路径验证，避免只测试未被调用的旧代码。

## R5. 参数化绑定、KV 与状态规格

依赖：R1/R4。

- [x] `bind_hidden` 改为按调用方 TensorSpec 检查，保留同 context、普通可写且已物化
  tensor 的约束与消费者先释放顺序；`add/upload/read/retain` 无须改写。
- [x] `KvCache` 按 KV 头数、head_dim 和实例容量生成父 tensor、slot view 及字节统计；
  首版仍支持既有 FP16 布局，当前验收容量上限保持 512。
- [x] 保留单套 K/V、只追加当前槽、读取图直接引用父缓冲、有效前缀 reset/truncate；
  保留 copy 图重验证与耗时计数，不恢复句柄交换、整段复制或 KV 双缓冲。
- [x] DeltaNet recurrent/卷积状态大小由已检查的组件配置推导；精度、双 bank 与
  初始化行为保持现状。避免在 native、Python reference、预算中分别写同一组常数。
- [x] 保留容量/输入先检查再执行，部分失败使实例失效；不给递推状态增加按长度回滚。

交付/验收：原布局和一组不同头数/维度的小配置通过绑定、槽位写入、历史内容保持、
边界与生命周期检查；原状态规格与资源计数保持一致。

## R6. 分离 mixer/MLP 计算与 Qwen 组合

依赖：R4–R5。

- [x] Attention 核心只接收处理后的 Q、缓存 K/V、有效长度和缩放参数，构建 score、
  mask、Softmax、V 加权；投影、Q/K norm、RoPE、输出 gate/投影由 Qwen 组件组合。
- [x] 头分组、旋转维度和 slice 范围从参数推导；保留缩放顺序、FP32 masked Softmax
  及已验证二维布局，不恢复有问题的三维 Softmax 或展开整份 KV 的路径。
- [x] DeltaNet 拆出无 checkpoint 路径依赖的递推和卷积/归一化片段；投影拆分、
  decay/beta 构造、gate 和权重映射归 Qwen 组装。保留 FP32 recurrent/core 修正。
- [x] 保留固定 A→B/B→A 调度和图间引用；不处理暂缓的双图权重去重，不增加副本。
- [x] 提取已有 `down(SiLU(gate(x)) * up(x))` 片段，权重与维度显式传入；
  外部 norm/residual 放置留给模型层，不将 SwiGLU 作为所有 FFN 的默认结构。

交付/验收：公共计算片段无模型名、权重前缀或绝对层号；原 Attention/DeltaNet
参考轨迹、精度阈值及少量不同尺寸小图检查通过。此步不新增 OPT 专属算子。

## R7. 解耦 decoder 层排列与执行管理

依赖：R6。

- [x] Qwen 层组装持有 pre-norm、mixer、residual、post-norm、SwiGLU、residual 顺序。
- [x] mixer 类型由层描述提供，移除计算代码中的 `layer == 3`、`layer > 2`；
  层组按显式列表连接，去除 `first=0/3` 和 `layer<4` 的执行器限制。
- [x] 保留既有“完整第 3 层”和“第 0–3 层组合”默认验证配置；新增一个不同层号/
  长度的短切片验证列表执行，不扩展到全模型。
- [x] 保留固定层间 tensor 绑定、全组执行前检查、全部层成功后提交长度，以及中途
  失败使整组失效；指标按所选层和配置求和。

交付/验收：Qwen 专属层组合调用公共算子；旧单层/四层组及短切片通过，
未验证模型或结构仍明确拒绝。

## R8. 整理验证依赖并完成回归

依赖：随 R2–R7 迁移同步修改，最后统一验收；不等待此步才修复测试调用方。

- [x] 将 `compare_file`、数组保存等公共工具移出 `validation/delta_net.py`，
  Attention/Decoder 不再因文件比较而导入 DeltaNet 官方参考。
- [x] 保留官方模型参考与设备实现的独立性；共享配置/文件格式，不共享待测算术来
  生成 expected。Qwen 精度补丁、cache 协议和 hook 放在模型参考模块。
- [x] 将模型尺寸 operator catalog 和真实 trace 投影映射归 Qwen 测试配置；
  通用小图生成函数接收参数。固定随机种子、诊断 shape 和边界常数可留在用例中。
- [x] 通用比较器接收明确容差；模型误差预算由 fixture/参考策略携带。维持当前数值、
  生命周期、执行后端和驻留证据的独立判定，不合并具有不同语义的比较函数。
- [x] CLI 只负责选择已有配置、参数校验和调度；保留设备锁、超时/恢复标记、
  `--prepare-only` 与显式设备运行方式，不修改 `validation/device.py` 的工作机制。
- [x] 更新 CMake 依赖、公开头文件和所有受影响 CLI/import；host 数据测试不链接 SDK，
  普通 CTest 不打开设备。沿用 unittest/CTest，不引入 lit 或统一测试框架改造。

最小验收矩阵：

| 范围 | 本次需要的验证 |
|---|---|
| 保留的环境/下载 | 原 host 测试与 CLI 导入检查；不重新下载或改系统环境 |
| checkpoint/CPU 参考 | 伪 checkpoint 的格式/身份拒绝；固定 Qwen 8-token prefill/逐步与关键 trace 回归，保留原阈值 |
| 导出/读取/预算 | 小包完整性、损坏/alias/非 Qwen 名称、F16 原样及原 BF16/F32 策略；旧 Qwen 包/资源计数兼容 |
| SDK 与算子 | 受影响的既有探针和生产公共函数小图；Context/tensor 公共路径变动时补一次卷积基线 |
| 状态与模型切片 | 不同尺寸 KV/递推小例、独立 mixer 短轨迹、完整 Attention 层、四层组 32 步及短切片 |
| 长度与传输 | 一次四层组 512 容量检查及第 513 次拒绝；沿用 reset/fresh/final-only；旧组每步 2,056 字节显式上传、零中间读回 |
| 工程质量 | 必要 host 单测、构建/CTest、公共头自包含、C++ 格式/include、Ruff `--check`、文档命令一致性 |

交付/验收：全部迁移路径确实使用公共模块；原行为保留；不同名称/尺寸的小用例证明
解耦有效。测试只覆盖迁移风险，不建立每个包装函数的镜像测试或重复长轨迹矩阵。

## 执行顺序与停止条件

- 顺序：R1 → R2 → R3 → R4 → R5 → R6 → R7；R8 随每步同步，最后收口。
  每步交付可构建/可验证结果，再迁移下一层，避免一次改完整条链后才定位误差。
- 全部源代码提取及 host 检查不依赖驱动扩容；设备检查仅加载有界小图或原有切片。
  约 1 GiB 分配限制不阻止这次解耦，但实际分配失败仍需停止对应设备验证。
- `NP101-MEM-002` 权重去重继续暂缓；`NP101-OBS-001` 和 `NP101-STATE-001`
  不因数值回归通过而关闭。报告不把 SDK 成功当作完整 NPU/驻留证明。
- 不新增未经论证的 SDK 接口、不改变计算顺序或放宽误差阈值来完成重构；
  新形状触及 SDK 限制时明确记录支持范围，不扩展成性能优化或新模型适配项目。
- 完成后更新 README、TODO 和测试说明，说明本次已解耦能力及保留的模型专属部分，
  然后回到原施工计划。未新增的模型适配与完整设备生成不计入本次验收。

依据：[芯片团队 API 文档](../demo/ref_op_api_guide.md)、[demo](../demo/main.c)、
[验证目录](../tests/README.md)、[算子](../tests/np101-operator-acceptance.md)、
[DeltaNet](../tests/np101-delta-net.md)、[Attention/KV](../tests/np101-attention.md)、
[Decoder](../tests/np101-decoder.md)、[工程代办](../TODO.md)及当前 SDK 头文件。

## 本次验收结果

- Host：55 项 Python 单测、C++ 全量构建与 CTest 通过；公共头自包含、同名文件
  include 检查、LLVM 100 列格式、Ruff lint/import/format 和 diff 空白检查通过。
- CPU：original-fp32 / deployment-fp16 的 1/2/4/8-token prefill/逐步对齐通过；
  两种模式各 718 项历史 trace 比较通过，冻结容差保持不变。
- 权重：原 Qwen 320 张量逐字节通过独立 PyTorch 转换验证，旧 1,504,791,808 字节
  权重包通过格式/摘要检查；新增非 Qwen 名称、无 embedding、F16 原样和 alias 拒绝检查。
- 设备数值回归：独立 DeltaNet 4 步、Attention 2 步、完整 Attention 层、两层 DeltaNet
  切片、四层组 32/512 步、不同尺寸合成 decoder/KV、卷积基线及 3 个受影响原始探针通过。
- 四层 512 步组包含 reset/final-only/fresh，共执行 560 步、801 项比较通过；第 513 次
  提交拒绝。每步显式上传 2,056 字节，零中间读回，原状态复用方案与重复权重开销保留。
- 分配：同一 118,156 字节合成权重包在 constant/mutable 两种模式下，与显式状态
  同时分配并完成逐字节回读。未重跑已知约 1 GiB 的全量失败实验。

详细结果与路径见 [组件验收记录](../tests/np101-components.md#decoupling-acceptance-record)。
SDK 数值通过仍未解决 NPU 后端证明、物理驻留、全模型内存或完整生成验收；
`NP101-MEM-001/002`、`NP101-OBS-001`、`NP101-STATE-001` 保持原状态。
