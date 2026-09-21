# 模型解耦与 OPT-350M 适配施工清单

更新日期：2026-09-21。状态：步骤 0 已完成；步骤 1–7 待审阅，尚未实施。

## 目标与边界

- 当前 DLM 改为 `facebook/opt-350m`，只启用这一模型的下载。
- 将模型结构、权重映射与 NP101 通用算子/存储分开；以 OPT 和已有 Qwen3.5
  两个实际使用方验证解耦，不以“类已经参数化”代替复用验收。
- 保留 Qwen3.5 已有实现、缓存和验证证据；其支持范围仍是已经验证的部分。
- 首次 OPT 部署仍要求完整文本计算在 NP101 上执行。CPU 只承担初始化、
  tokenizer、调度和显示；CPU 导入通过不等于 NP101 部署通过。
- 本轮不实施下述推理重构，不引入新测试框架、MLIR、量化、训练或动态 KV 分配。
  双图 DeltaNet 权重去重继续暂缓；TLM 下载和两机联动保留到后续阶段。

## 必须固定的 OPT-350M 计算契约

以下载的固定 revision、实际 tensor 和当前 Conda 环境中的官方 Transformers 实现为准。

| 项目 | 适配要求 |
|---|---|
| 层数与维度 | 24 层；hidden=1024，FFN=4096，词嵌入维度=512 |
| Attention | 16 个 Q/K/V 头，head_dim=64；标准 MHA，保留投影 bias |
| 位置编码 | 学习式位置嵌入；按 attention mask 计算位置，表索引有 +2 偏移；无 RoPE |
| 归一化 | `do_layer_norm_before=false`；两个残差相加后分别做 LayerNorm |
| LayerNorm | 包含均值消去、方差、gamma 和 beta；核验 epsilon；不能沿用 Qwen 的 RMSNorm 或 `1 + weight` |
| FFN | 带 bias 的 Linear → ReLU → Linear；不是 SwiGLU |
| 输入 | token embedding[512] → project_in[1024] → 加位置嵌入 |
| 输出 | 24 层后 project_out[512] → 与 embedding 绑定的 LM head；该配置无额外 decoder final LayerNorm |
| 词表与特殊 token | 模型输出 50272 行，tokenizer 实际 50265 项；BOS=EOS=2、PAD=1，默认添加 BOS，无 chat template |
| 推理模式 | eval、关闭 dropout；batch=1；首次上下文容量为 512，之后按能力验证扩展 |
| 精度 | 原始 FP16 权重按位保留；Softmax、归一化等累加精度单独规定和验证 |

## 现有耦合点与拟定归属

| 当前文件 | 需要调整的内容 | 归属 |
|---|---|---|
| `python/specferry/reference/checkpoint.py` | 固定模型/revision、safetensors index、Qwen tensor 分组 | 通用读取/完整性检查 + 模型适配器 |
| `python/specferry/reference/model.py`、`trace.py` | 固定 Qwen 类、聊天模板、0/3 层 hook | 每个模型独立的官方参考适配器 |
| `python/specferry/export/schema.py`、`weights.py`、`memory.py` | 固定形状、只接受 BF16/F32、固定 alias 与 Qwen 内存公式 | 通用打包器 + 模型提供的契约/预算 |
| `native/np101/weights.cpp` | 固定名称前缀和 embedding/head alias | 由部署包显式描述 |
| `native/np101/tensor.cpp` | `bind_hidden` 写死 `[1024,1]` | 按调用方 TensorSpec 检查的通用绑定 |
| `native/np101/kv_cache.*` | 固定 2 个 KV 头、head_dim=256、token_bytes | 参数化单缓冲 KV 存储 |
| `native/np101/attention.cpp` | 固定第 3 层、Qwen Q/K norm、RoPE、gate | 通用 attention 核心 + Qwen 模型组合 |
| `native/np101/decoder.cpp` | 固定四层排列、RMSNorm、SwiGLU | 独立 OPT/Qwen decoder 组合 |
| `native/np101/delta_net.*` | Qwen DeltaNet 形状与双图调度 | 保留为独立模型组件，复用必要的底层算子 |

拟定目录边界如下；只在对应步骤落地时迁移，不单独进行全仓库目录改名。

```text
native/model/                 模型描述、权重角色等普通 C++ 数据契约
native/np101/                 Context、Tensor、Graph、权重读取、KV 存储
native/np101/ops/             SDK 算子构图及可复用计算片段
native/models/opt/            OPT 输入、decoder、输出组合
native/models/qwen3_5/        已有 Qwen 组合及专有组件
python/specferry/models/      模型配置/权重映射/官方参考适配器
python/specferry/export/      通用部署包写入与校验
tests/native/np101/           继续使用现有 CTest/诊断目录
tests/python/                继续使用现有 unittest 目录
```

通用算子接收 tensor 和显式参数，不读取 Hugging Face 名称，不判断模型品牌。
模型组合调用这些算子。避免一个 Decoder 类积累大量互斥开关，也不为每个 SDK
函数增加一层无实际职责的类。暂不增加第二套后端抽象。

## 0. 下载与 CPU 导入验证（已完成）

- [x] 锁定 OPT-350M 官方 revision，下载 config、tokenizer 和单套官方权重；
  记录文件大小、校验和、完整下载 manifest；保留 Qwen 缓存。
- [x] 在独立导入验证中安全读取官方 PyTorch bin；不依赖 safetensors index，
  不启用远程代码或不受限 pickle 加载。
- [x] 核验实际 dtype、tensor 名称/形状、独占存储字节和 embedding/head 别名。
- [x] 在 SpecFerry Conda 环境中以官方 OPT 类进行离线 CPU 导入和固定输入前向/
  短文本生成；记录 token IDs、非有限值、版本和结构参数。
- [x] 明确区分 CPU 导入已验证与 NP101 exporter/runtime 尚未适配。

交付：下载器必要修改、最小回归测试、下载 manifest、`.cache/runs/` 下的导入报告。
验收：校验通过且官方 CPU 模型可加载和运行；不得因此标记设备部署完成。

已完成记录：

- revision：`08ab08cc4b72ff5593870b5d527cf4230323703c`；9 个下载文件校验通过。
- 模型目录：`.cache/models/facebook/opt-350m/`。
- 388 个 tensor 全部为 FP16；331,196,416 个独立参数，payload 为 662,392,832 字节
  （631.707 MiB）。checkpoint 无单独 `lm_head.weight`，官方加载后与 embedding 共享存储。
- 官方 CPU FP16 模型无缺失/多余/形状不匹配权重，3 组提示各生成 20 token，logits 有限。
  8-token 测例的缓存增量与整段前向最终 logits 最大绝对差为 0，cache 覆盖 24 层。
- 下载相关 8 项测试及 Ruff 检查通过。报告、tensor 清单和可重跑验证脚本位于
  `.cache/runs/opt-350m-import-20260921/`；运行 `verify_import.py` 使用 SpecFerry Conda。
- 验证报告的 scope 仅为 CPU 导入/推理；现有 Qwen exporter 和 NP101 runtime 尚未适配。

## 1. 固定模型描述与适配器边界

依赖：步骤 0。无设备依赖。

- [ ] 定义版本化模型描述，包含架构、实际维度、层配置、词表、位置编码、精度策略、
  权重角色/别名及 checkpoint 标识。embedding_dim 与 hidden_size 必须分开。
- [ ] OPT/Qwen 分别解析配置和映射权重；移出通用层中的固定 repo、revision、
  tensor 前缀、层号、聊天模板及模型尺寸。
- [ ] 通用入口从模型描述分派适配器；支持范围外的结构在建图前明确报错。
- [ ] Python 和 C++ 使用同一份序列化契约；如修改包格式，同步更新两端版本校验，
  明确旧 Qwen 包的兼容/再导出方式，禁止静默按新格式解释旧包。

交付：模型描述 schema、两个适配器骨架、迁移入口。
验收：OPT 与 Qwen 的真实配置，以及一个不同尺寸的小型合成配置均能正确解析；
缺失权重、错误 bias/维度、未知版本在 SDK 初始化前失败。

## 2. 通用权重导出与独立 CPU 参考

依赖：步骤 1。无设备依赖。

- [ ] 将 checkpoint 读取与物理打包分开，支持单文件/分片 safetensors 及安全读取
  官方 PyTorch bin；保留流式写出、完整性校验和有界 host 内存使用。
- [ ] 增加 F16 原样导出，逐 tensor 验证数值字节不变；Qwen BF16/F32 策略保持独立。
- [ ] alias 由模型描述提供；记录唯一物理数据及逻辑引用，检测悬空/循环 alias，
  去掉 `WeightStore::find` 中对 LM head 名称的特判。
- [ ] 将权重大小、每层 KV、激活和已知复制成本按真实配置计算；SDK 私有开销保留未知，
  不将导出包大小当作设备占用。超过 8 MiB 的 embedding 不能绕过现有有界读取检查。
- [ ] OPT 使用原始文本续写输入，不套用 Qwen chat template；固定 BOS/EOS/pad 行为。
  分别记录 tokenizer token 集合与模型输出行数，不自行裁剪 head 的额外行。
- [ ] 用官方 eager 实现生成 embedding、Q/K/V、Attention、两次残差/LayerNorm、
  FFN、project_out 和 logits 参考；捕获节点可配置，不再固定层 0/3。
- [ ] 固定精度边界及验收阈值。先核验 CPU 整段前向与逐 token 缓存前向，
  再制作设备测试数据；参考计算不能复用待测设备实现。

交付：OPT 部署包、内存预算、独立 CPU fixtures；Qwen 旧参考仍可运行。
验收：全部 OPT 权重映射闭合、FP16 按位一致、alias 正确；参考前向/缓存路径一致，
损坏包和模型/权重错配被拒绝。

## 3. 提取通用算子与图构建设施

依赖：步骤 1–2；硬件数值检查使用现有串行设备 runner。

- [ ] 提取重复的 tensor/node/参数生命周期管理及 shape/layout 检查；保留
  `vsi_nn_AddNode` 初始化的内部状态，错误时按所有权顺序清理。
- [ ] 复用 Linear/MatMul、可选 bias、Add、ReLU/SiLU、reshape/slice、Gather、
  masked Softmax、dtype conversion；逻辑维度与 SDK 轴序的转换集中处理。
- [ ] LayerNorm 和 RMSNorm 使用不同的显式契约；优先核验文档中的 `LAYER_NORM`
  对实际形状/精度的支持，必要时用文档中的基础算子组合并说明原因。
- [ ] 参数化归一化 epsilon、gamma 约定、Softmax 轴与累加/输出精度；
  不因权重是 FP16 就强制所有中间结果为 FP16。
- [ ] 抽取无位置编码、无 Q/K norm、无输出 gate 的 Attention 核心：接收已投影
  Q/K/V，处理缩放、因果 mask、Softmax 和 V 加权；分别验证 MHA/GQA 头映射。
- [ ] 将 Qwen 专有 Q/K norm、RoPE、gate 与 DeltaNet 算法留在模型组合中。
  迁移时保留 FP32 recurrent/core 的现有修正，不处理双图权重去重。

交付：由 OPT 与 Qwen 实际调用的通用算子模块。
验收：小型非 1024 维样例和真实 OPT 形状通过；覆盖 bias、非零均值 LayerNorm、
Softmax 轴/无效位置。被迁移的 Qwen 路径以既有阈值回归，不扩大临时诊断矩阵。

## 4. 参数化 KV 与推理状态

依赖：步骤 3。

- [ ] `KvCache` 接收 KV 头数、head_dim、dtype、capacity 和明确布局；
  `bind_hidden` 改为验证调用方提供的 TensorSpec。
- [ ] 保持每层一套预分配 K/V，通过 slot view 写入当前 token，后续计算直接读取
  同一存储；不复制历史缓存，不恢复句柄交换或 KV 双缓冲。
- [ ] OPT 使用 16×64 的 K/V 头布局；512 容量时每层约 2 MiB，24 层合计 48 MiB。
  每层每次只写新 token 的 K/V，共 4 KiB；将计数按描述计算。
- [ ] 序列对象统一管理已消费 token 数；执行前检查全模型容量/位置边界，全部层成功后
  才提交长度。部分层执行失败时实例失效并要求重建，不宣称自动回滚。
- [ ] reset/truncate 仅改变有效前缀；覆盖无效槽、truncate 后重新追加及 fresh/reset
  等价性。Qwen recurrent state 不提供虚假的长度回滚能力。
- [ ] 保留 slot-copy 图重验证的显式统计及跨图引用的所有权/释放顺序。

交付：通用单缓冲 KV、参数化绑定与序列状态契约。
验收：OPT 与原 Qwen 两种真实布局均通过追加/边界/生命周期检查；
无历史 KV host 回传或整段复制，应用计数与设备驻留证据分开报告。

## 5. 组合 OPT Decoder 并验收层间连接

依赖：步骤 2–4。

- [ ] 实现带 bias 的 Q/K/V 与输出投影；保留官方实现先缩放 Q 的运算顺序，
  不以实数等价为由任意移动 FP16 舍入位置。
- [ ] 单层严格按以下顺序组合：

  ```text
  a = LayerNorm(x + Attention(x))
  y = LayerNorm(a + Linear2(ReLU(Linear1(a))))
  ```

- [ ] 层号、权重角色、输入输出绑定均显式传入；用同一实现构建任意已验证层区间，
  不保留“只有第 3 层是 Attention”或“只支持 0–3 层”的通用限制。
- [ ] 依次验证独立真实层、连续四层、24 层 decoder；层间保持固定 tensor 引用，
  不以 host 读回再上传连接各层。
- [ ] 在 2/32 token 轨迹中检查关键中间值及 KV，另用一次完整容量检查覆盖边界；
  保留 reset/fresh/final-only 和失败失效检查，避免每层重复全部长轨迹。

交付：OPT decoder 组合、独立参考对照及实际权重/状态/传输计数。
验收：逐层和层间数值检查均通过；原 Qwen 已支持的组合调用共享设施后保持通过。

## 6. 接通 OPT 完整文本生成

依赖：步骤 5；实际常驻内存和新增算子验收。

- [ ] 输入侧在设备完成 token Gather、512→1024 投影、学习式位置 Gather 和相加；
  测试位置 0/1/容量末尾及 +2 偏移，声明首版不支持的 padding/batch 情况。
- [ ] 输出侧完成 1024→512 投影、全词表 LM head 和设备 greedy 选择；
  不添加该 OPT 配置不存在的最终 LayerNorm。
- [ ] embedding/head 的逻辑绑定与设备物理共享分别验收。优先尝试支持的单份只读
  存储/view；无法共享时记录必要复制及预算，禁止把 manifest alias 当作 SDK 共享证明。
  本项不扩大到暂缓的 DeltaNet 双图去重。
- [ ] 若 LM head 分块，验证完整覆盖、尾块、全局 token ID、最大值和相同值时的
  选择规则；不得遗漏词表行或在 host 计算 head/argmax。
- [ ] 首版 prompt processing 可逐 token 复用 decode 路径；批量 prefill 优化另列后续。
  明确已消费长度、待消费新 token、EOS、最大输出数及上下文耗尽语义。
- [ ] 在建图前预算唯一权重、SDK 复制、KV、工作区；按单层→多层→完整模型逐级验证，
  不重跑之前已知失败的 Qwen 容量压力实验。

交付：完整 OPT-350M C++ 生成入口与 host tokenizer/显示脚本。
验收：embedding、24 层、状态更新、输出投影、head、greedy 均纳入执行证据；
固定输入 teacher forcing logits 对齐，固定提示生成可复现，reset/重复生成正常。
出现首个 token 分歧时定位 logits/中间值，不靠放宽阈值宣布通过。

## 7. 解耦与交付验收

依赖：步骤 1–6。

- [ ] OPT 与保留的 Qwen 测试通过同一组通用算子/存储；通用层不包含 checkpoint
  名称、固定层号或模型规模常量。使用至少一个不同隐藏维度的小配置防止假参数化。
- [ ] 测试继续放现有 Python/native 目录，按真实功能命名；不引入 lit 等框架，
  不恢复已经删除的句柄交换或临时探针。
- [ ] 通过必要的单元测试、目标 CTest、对应设备数值回归；执行 Ruff 与 C++ 格式检查，
  保持 100 列、控制语句花括号及既有 include 规则。
- [ ] 报告分别列出下载完整性、CPU 导入、导出完整性、SDK 数值、执行后端、设备驻留、
  实际内存和端到端生成；缺少硬件证据时保留阻塞，不把数值通过写成纯 NPU 部署通过。
- [ ] 更新 README、TODO 和模型支持表，给出可复现命令；按上述步骤分次提交，
  避免将目录迁移、语义变化和完整模型接线压进一个不可审阅的大提交。

## 依赖与停止条件

- 下载/CPU 导入可立即执行；步骤 1–2 不依赖设备或驱动内存扩容。
- 步骤 3–5 可先用小图/模型切片实施和验证；新的真实形状仍需算子能力验收。
- OPT 权重预计低于当前约 1 GiB 限制，但不保证完整图可常驻。步骤 6 应依据 OPT
  自身的权重、复制、KV 和工作区实测决定是否被 `NP101-MEM-001` 阻塞。
- `NP101-MEM-002` 只涉及暂缓的 DeltaNet 双图权重去重，不是 OPT 主线前置条件。
- `NP101-OBS-001` 以及 KV 路径相关的 `NP101-STATE-001` 继续约束硬件/驻留验收。
  模型切换不能自动关闭这些代办。

## 核对依据

- [芯片团队算子/API 文档](../demo/ref_op_api_guide.md)、[卷积 demo](../demo/main.c)
  及本机 SDK 头文件；文档列出算子不等于特定参数已在硬件上验证。
- [OPT-350M 官方配置](https://huggingface.co/facebook/opt-350m/blob/main/config.json)；
  下载报告记录实际固定 revision。
- 当前 SpecFerry Conda 环境的 `transformers/models/opt/modeling_opt.py`，重点核对
  `OPTLearnedPositionalEmbedding`、`OPTAttention`、`OPTDecoderLayer` 和 `OPTDecoder`；
  制作参考时记录版本及源码 hash。
- [既有 Qwen decoder 验证](../tests/np101-decoder.md)、
  [Attention/KV 验证](../tests/np101-attention.md)、[工程代办](../TODO.md)。
