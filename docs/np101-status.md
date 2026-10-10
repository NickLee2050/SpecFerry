# NP101 推理实现与优化结论

整理日期：2026-10-10；设备与性能证据截至 2026-10-09。目标是在旧 component 正确性的
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
`.cache/runs/code-consolidation-20261010/before/`。2026-10-10代码精简仅做主机编译和离线回归；
下表板卡证据对应精简前版本，不能表述为当前binary的新验收。

| 验收对象 | 已完成范围 | 边界 |
|---|---|---|
| 合并 KV / C16 | 两步全层 logits/选词/KV；连续生成、A/B/A 隔离、未写槽、末槽、越界/close、释放 | teacher 与连续生成分别验收 |
| block4 / C16 | 全 24 层 KV、最终 logits、prompt16/5/16、尾部与 decode、block1 回归 | KV 分批读回；故障后恢复再完成，不把失败轮算通过 |
| 完整模型 / C64 | prompt15/16/17/49/63/64 边界、最终 logits、末四层 KV；Context 重建 | 未重做全部层 KV；精度例外见下文 |
| 两路 fc1 | C16 缓存/生命周期及 block4 logits；C64 prompt17/63/64 logits、64/3/64 末两层 KV/A/B/A | 不外推所有 C64 输入、全层 KV 或长期稳定性 |
| block16 / 两路 fc1 / C64 | last-token head，17/63/64 logits，64/3/64 末两层 KV、隔离/容量/释放；64-token prefill 对照 | 模型 block8 未测，FFN 集合通过不等于模型通过 |
| 固定 LN 参数预处理 | 同配置17输入/16输出前→候选→后及最终 logits；32输入/32输出重复生成 | 50,272 个最终 logits 与旧版本逐字节相同；未重做全部配置 |
| 有界常驻复用 | 同 boot、17输入/16输出，1暖机+10正式请求，160输出一致、无运行期编译、释放归零 | 约五分钟证据，不构成跨启动或长期验收 |

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
- 最新32输入/32输出三次均与CPU参考逐token一致，共96输出/142.669892秒；首词后
  93输出/130.186002秒。请求中位47.565700秒。英文续写有重复，CPU参考同样重复；
  准确文字及IDs在 `layernorm-affine-preparation-20261009/generated-text.json`。

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

最后已存门禁为 **2026-10-09 17:32:37.380997 +08:00**，boot
`f214ee36-6fd8-42cd-922a-d5095f9980bd`。固定两级恢复控制及后续所列负载通过，
各有至少45秒延迟检查，无目标故障/owner/恢复标记。这是历史检查，运行前仍须重查；
原厂/自写卷积hang、EDMA和关机队列指针损坏的根因未定，详见事故台账。

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
| 常驻复用、静态审计、SDK分析 | `resident-request-reuse-20261009/`、`decode-static-audit-20261009/`、`ffn-workload-analysis-20261004/`、`decode-offline-20261005/` |
| 退役候选与历史源码 | `decoder-structure-screening-20261001/`、`fc1-shape-decode-20261009/`、`vx-gemm-estimate-20261009/`、`cleanup-20261001/before-source.tar.gz`、`diagnostic-cleanup-20261008/` |
