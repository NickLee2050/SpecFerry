# SDK categorical sampling

OPT generation accepts `--sample --seed 42` as an explicit override of the
checkpoint's default policy, which resolves to greedy for the retained OPT model. Sampling
draws from softmax of all 50,272 logits at temperature 1. There is no top-k/top-p
filter or host sampling fallback. This is a generation policy, not yet a
speculative-decoding acceptance/rejection implementation.

```bash
conda activate SpecFerry
cmake --build build -j 4
python scripts/check_np101_sampling.py --output .cache/runs/sampling-check
python scripts/generate_opt.py --sample --seed 42 \
  --prompt 'The capital of France is' --max-new-tokens 32 \
  --output .cache/runs/opt-sampling
```

Use new output directories. The checker runs 4,096 samples per case, validates
output size/range, uniform and weighted frequencies, the final vocabulary token,
same-seed replay, graph recreation, and changes to either seed or draw counter.
Frequency checks use six binomial standard deviations plus two samples of margin;
they are regression checks, not certification of a random-number generator.
The checker returns 0 for numerical success and 1 for a failure. Device calls are
serialized with the existing lock, bounded timeout and recovery-marker handling.

## Interface and implementation

The chip team's `demo/ref_op_api_guide.md` lists RANDOM_MULTINOMIAL. Installed
`ops/vsi_nn_op_random_multinomial.h` exposes `sample_num`, but does not explain
seed shape. The upstream VeriSilicon [operator implementation](https://github.com/VeriSilicon/TIM-VX/blob/main/src/tim/vx/internal/src/ops/vsi_nn_op_random_multinomial.c)
and [OpenCL implementation](https://github.com/VeriSilicon/TIM-VX/blob/main/src/tim/vx/internal/src/kernel/cl/random_multinomial_cl.c)
provided starting points; installed-SDK execution is the acceptance evidence.
Retrieved source snapshots are under `.cache/runs/sampling-api-20260922/`.

The tested SDK tensor dimensions are logits `[classes, 1]`, four initialized
INT32 seed words, and INT32 output `[sample_num, 1]`. Generation uses one sample
and seed words `{user_seed, draw_counter, 0, 0}`. The counter advances after each
successful prediction and resets with the request. Seed replay applies to the
same SDK, inputs and configuration; equality with a CPU library's seeded output
or a future SDK version is not promised. `--compare-cpu` checks greedy equality
and is deliberately rejected with `--sample`.

`VocabularyHead` concatenates its valid FP16 head blocks, converts logits exactly
to FP32, and invokes RANDOM_MULTINOMIAL. Model weights and head products remain
FP16. The host uploads 16 seed bytes and reads one four-byte token per prediction;
it never downloads logits or copies KV for sampling. Conversion and concatenation
add graph buffers/work; their device memory cost and backend require later profiling.

## Direct FP16 failure and workaround

On 2026-09-22, the installed SDK returned success for direct FP16 sampling but
produced index 8 for every eight-class sample (out of range). With 50,272 classes
it returned 8216 for every sample, even with all probability concentrated on
the final token 50271. Initialization, graph execution and release all succeeded.
These observations establish incorrect results for the tested configurations,
not the internal cause or a failure of every possible FP16 configuration.

Reproduce the rejected configuration with:

```bash
python scripts/check_np101_sampling.py --dtype F16 --classes 8 \
  --output .cache/runs/sampling-f16-repro
```

The checker must fail if the SDK still exhibits this behavior. Direct FP32 logits
passed the eight-class and full-vocabulary cases. FP16-to-FP32 graph conversion
also passed; it cannot lose information already represented in FP16. Only this
promoted path is enabled in OPT's sampling head.

Evidence under `.cache/runs/`:

| Run | Result |
|---|---|
| `sampling-f16-small-20260922` | Incorrect: every index was 8 |
| `sampling-f16-repro-20260922` | Permanent checker correctly rejected the eight-class FP16 output; lifecycle passed |
| `sampling-f16-vocabulary-20260922` | Incorrect: every index was 8216 |
| `sampling-f32-small-20260922` | Frequencies, seed replay and final token passed |
| `sampling-f32-vocabulary-20260922` | Full-vocabulary checks passed |
| `sampling-f32-single-20260922` | One-sample shape, seed replay and final token passed |
| `sampling-acceptance-20260922` | Promoted full-vocabulary path passed all report checks |
| `opt-sampling-seed42-20260922` | Full 24-layer model returned 32 sampled tokens |
| `opt-sampling-seed42-repeat-20260922` | All 32 IDs exactly reproduced after recreating the complete model |
| `opt-sampling-seed0-20260922` | Zero seed supported; 16-token trajectory differed from seed 42 |
| `opt-sampling-greedy-regression-20260922` | Default greedy path still matched all eight official CPU tokens |

The full-model seed-42 run consumed 37 tokens, appended KV 888 times, uploaded
4,360 bytes and read 128 bytes. Successful SDK execution and generic driver
activity do not prove that every kernel ran on the physical NPU. Per-kernel
backend and physical residency evidence remain open, as in greedy generation.
`sampling-acceptance-20260922/generation-comparison.json` records cross-run checks.
All four complete-model runs released resources and exited normally. The 64 Python
unit tests, both native CTests, header checks and C++/Python formatting checks passed.
Negative/out-of-range seeds, a seed without sampling and sampling combined with
greedy CPU comparison were rejected before device execution.
