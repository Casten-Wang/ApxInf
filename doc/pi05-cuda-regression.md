# PI0.5 CUDA Regression Test for Every Pull Request

> Status: This document defines the canonical benchmark workload, tuning rules, accuracy protocol, and result format.
> As part of the manual CI/CD process, rerun the applicable tests after every code change and attach the results to the pull request.

## Paired regression for the GR00T optimization PR

This campaign compares upstream `9bbf948209b16e6344c2556d84cc42f7ba82df92`
with compute revision `2b90781f7d8252628835069c63660d9183047df9` on the same
device. Both benchmark binaries include the same reporting-only change for
complete action outputs and ordered latency samples. Historical best timings
below are context, not a substitute for this contemporaneous baseline.

The performance scope is **eight cells**: four device/precision paths, two
real cameras, and T=10/T=21. Each cell uses H=10 and three interleaved A/B rounds
of 10 warmups and 30 measured samples per arm. Pool 90 samples per arm and
report both timing boundaries, P50/P95, complete outputs, tactic identities
and the observed power/clock/thermal conditions. Each arm tunes a fresh local
database; no shared tactic file is changed. Three-view tests are optional
deployment-shape extensions and are outside this campaign.

The requested closed-loop validation first runs the PR candidate for **100
episodes on each of the four paths: all ten LIBERO-10 tasks, ten fixed trials
each**. Run additional upstream accuracy comparisons when a suspected
regression needs attribution; a second complete 100-episode arm is not an
unconditional requirement. Its checkpoint and tokenizer hashes
are pinned in section 3.1. Use the original checkpoint's OpenPI LIBERO
deployment configuration: H=10, ten flow steps starting at 1.0, no warm-start,
no discrete state tokens, two real cameras, replan every five actions and at
most 520 action steps. The original OpenPI configuration at
`175f89c31d1b2631a8ff3b678768f17489c5ead4` explicitly sets
`action_horizon=10, discrete_state_input=False` for `pi05_libero`.

Use each task's complete language instruction; the ten actual token lengths
are `14,14,14,16,21,16,20,15,10,14`. A T=10 performance fixture does not stand in
for all task prompts. Pin the original `norm_stats.json`, select the `actions`
statistics and quantile normalization with float32 arithmetic and epsilon
`1e-6`. Rotate both simulator images by 180 degrees and apply the same
PIL/BILINEAR 224-pixel letterbox preprocessing in both arms.

For every episode, seed the environment with 7, select its bundled initial
state by trial index, and take ten settling steps with gripper action -1.
Generate float32 `[10,32]` noise from an episode-local NumPy PCG64 stream seeded
with `SeedSequence([7, task_id, trial_id])`. Keep failures and allow only one
scored attempt per case. This noise schedule is a paired code-regression
protocol, not a reconstruction of historical Torch noise streams.

Prepare the candidate database using all ten real prompts and freeze its
original bytes before scoring. If results need attribution, start with saved
input/output comparisons. Use that unchanged database on the upstream
revision after verifying provider, tactic, SM and library compatibility, and
record any missing-key fallback. Do not rewrite database headers or enable
tuning during scored runs. Record database hashes before and after each run
and use the same calibration for paired FP8 comparisons. This diagnostic
holds available tactics fixed; the performance experiment retains independent
tuning. Escalate to full trajectory replay or a complete upstream 100-episode
arm only when the remaining uncertainty requires it.
Save every native input, complete `[10,32]` output and decoded action, including
failed episodes. Equal complete outputs establish numerical equality on the
tested inputs and tactics; they do not alone prove equal task success or
equivalence on untested inputs.

Report all four candidate accuracy results and all eight performance cells
explicitly, along with any additional upstream comparison actually performed.
In-progress runs and preparation calls are not completed accuracy results.
This 100-episode campaign must not be reported as the historical 500-episode
standard described below.

## 1. Goals

This benchmark answers two questions:

1. What is the actual PI0.5 inference latency of ApxInf on Thor SM110 and Orin SM87?
2. Which tactic should be selected for each operator shape at real LIBERO language lengths and reserved lengths?

The primary result uses an official LIBERO instruction with 10 tokens. The extended real-world result uses an official longest LIBERO instruction with 21 tokens.

## 2. Fixed workload

| Parameter | Fixed value |
|---|---:|
| Batch size | 1 |
| Camera views | 2 / 3 views |
| Images | 224 x 224 RGB, NHWC `uint8` |
| Action horizon `H` | 10 for the pinned OpenPI LIBERO deployment |
| Action dimension | 32 |
| Flow-matching steps | 10 |
| Token execution mode | Exact length; do not pad to 200 |
| Real benchmark token counts `T` | 10 / 21 |
| Autotune-only token shapes | 50 / 200 |
| Warm-up | 10 iterations |
| Measured samples | 30 iterations |
| Timing statistics | P50 / P95 / min / max / mean / standard deviation |

`H` denotes the action horizon and `T` the language token count. The paired
campaign above uses H=10 for both performance and accuracy. Historical H=50
runs are a different workload and must retain that label.

## 3. Token dataset

| `T` | Source | Role |
|---:|---|---|
| 10 | Official LIBERO; 10 instructions have a PaliGemma token length of exactly 10 | **Primary LIBERO** |
| 21 | Official LIBERO; 2 instructions share the maximum length | LIBERO worst-case language |

The primary 10-token result may use this official LIBERO instruction:

```text
put the bowl on top of the cabinet
```

The extended 21-token result may use:

```text
pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate
```

Before a formal T=10 or T=21 run, pin the text, token IDs, tokenizer hash, and simulation fixture hash together.

### 3.1 Fixed baseline fixtures

The baseline uses the real LIBERO first-replan fixtures already stored in the repository. The prompts and token IDs below are the actual baseline inputs. They replace the optional example prompts above and must not be interchanged when reproducing the baseline.

| `T` | Fixture | Prompt | PaliGemma token IDs |
|---:|---|---|---|
| 10 | `task_08_first_replan.npz` | `put both moka pots on the stove` | `2,1065,2145,705,1161,37801,611,573,37932,108` |
| 21 | `task_04_first_replan.npz` | `put the white mug on the left plate and put the yellow and white mug on the right plate` | `2,1065,573,2674,24464,611,573,2731,8811,578,2507,573,8123,578,2674,24464,611,573,1833,8811,108` |

Pinned artifact SHA256 values:

| Artifact | SHA256 |
|---|---|
| PaliGemma `tokenizer.model` | `8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6` |
| PI0.5 checkpoint `model.safetensors` | `21b8711787c4a75861b02cff6aa81675a3a943d32b435a68262ac4461e476ba4` |
| Raw T=10 NPZ fixture | `97f9d8b112605a67277cca65e4cadc06f7fd4ccd5e21f339a215670ea9e56473` |
| Raw T=21 NPZ fixture | `2663c33a3b801a7bf67bdefdea1526fdd9acad8564a0ede5ec98ee10f03381d6` |

Deterministically reconstruct 224 x 224 NHWC `uint8` images from the normalized patches in these fixtures, then pass them through ApxInf CUDA preprocessing. For the third view, reuse the wrist image. Label every three-view result as **duplicated wrist fixture**; it is not a real third LIBERO camera.

## 4. Meaning of views

| Views | Meaning |
|---:|---|
| 2 | Real LIBERO workload: base camera + wrist camera |
| 3 | Three-camera production-shape workload; not an official LIBERO camera configuration |

Run three-view performance tests only. Do not use three views for LIBERO task-suite accuracy evaluation.

## 5. Execution paths

| Device | Precision path | Purpose |
|---|---|---|
| Thor SM110 | BF16 | Thor high-precision baseline |
| Thor SM110 | FP8 native | Thor native FP8 quantized path |
| Orin SM87 | BF16 | Orin high-precision baseline |
| Orin SM87 | INT8 (W8A8) | Orin native INT8 quantized path |

The full matrix including optional three-view deployment shapes contains:

```text
4 device/precision paths x 2 view counts x 2 real token lengths = 16 cells
```

There are also 16 T=50/200 view/device/precision autotune-only profiles. They do not count as end-to-end benchmark results.

The current GR00T PR regression uses only the eight two-real-view cells, as
specified above; it does not require the optional third-view or T=50/200 cases.

NVFP4 is outside the scope of the current ApxInf benchmark. ApxInf currently has no PI0.5 NVFP4 executor, calibration, tactic, or validated result.

## 6. Historical performance reference results

These recorded results are useful reference points. For a code-regression
claim, run the declared upstream baseline and candidate under the same
current conditions; do not compare a new candidate only against an older
best run. Investigate and report observed regressions before acceptance.

All baseline cells below used 10 warm-up iterations and 30 measured samples. Each latency cell is **P50 / P95** in milliseconds.

### 6.1 Thor SM110 baseline

| Path | Views | `T` | Graph replay P50 / P95 | Input update + graph P50 / P95 |
|---|---:|---:|---:|---:|
| Thor SM110 BF16 | 2 | 10 | **72.454 / 72.673** | 73.392 / 73.773 |
| Thor SM110 BF16 | 2 | 21 | **79.654 / 80.036** | 79.673 / 79.969 |
| Thor SM110 BF16 | 3 | 10 | **89.757 / 90.038** | 89.769 / 90.043 |
| Thor SM110 BF16 | 3 | 21 | **93.649 / 93.908** | 93.206 / 93.502 |
| Thor SM110 FP8 native | 2 | 10 | **41.159 / 41.312** | 41.193 / 41.288 |
| Thor SM110 FP8 native | 2 | 21 | **42.079 / 42.204** | 42.015 / 42.105 |
| Thor SM110 FP8 native | 3 | 10 | **53.636 / 53.903** | 53.588 / 53.763 |
| Thor SM110 FP8 native | 3 | 21 | **55.506 / 55.616** | 55.276 / 55.403 |

### 6.2 Orin SM87 baseline

Orin uses the same fixed LIBERO fixtures, token IDs, NHWC `uint8` images, and BF16 noise as Thor.

| Path | Views | `T` | Graph replay P50 / P95 | Input update + graph P50 / P95 |
|---|---:|---:|---:|---:|
| Orin SM87 BF16 | 2 | 10 | **165.665 / 165.845** | 165.747 / 165.870 |
| Orin SM87 BF16 | 2 | 21 | **166.606 / 166.776** | 166.728 / 166.882 |
| Orin SM87 BF16 | 3 | 10 | **205.269 / 205.739** | 205.565 / 206.132 |
| Orin SM87 BF16 | 3 | 21 | **204.319 / 205.121** | 204.425 / 205.273 |
| Orin SM87 INT8 W8A8 | 2 | 10 | **124.250 / 124.293** | 124.306 / 124.352 |
| Orin SM87 INT8 W8A8 | 2 | 21 | **124.808 / 124.901** | 124.888 / 124.933 |
| Orin SM87 INT8 W8A8 | 3 | 10 | **165.693 / 165.753** | 165.755 / 165.792 |
| Orin SM87 INT8 W8A8 | 3 | 21 | **166.285 / 166.358** | 166.335 / 166.405 |

#### Orin INT8 accuracy limitation

The Orin INT8 CUDA Graph and eager outputs match element by element. Replacing the SM87 CUTLASS W8A8 GEMM with cuBLAS also produces elementwise-identical final outputs across all eight mixed-precision combinations (`max_abs=0`). The accuracy issue comes from the current naive PTQ W8A8 quantization algorithm: weights use per-output-channel absmax scales, activations use dynamic per-token-row absmax scales, and the algorithm has no calibration, SmoothQuant, outlier handling, or QAT. Improving the quantization algorithm remains a TODO.

## 7. Historical accuracy standard and reference results

The previously documented broader accuracy standard was:

| Parameter | Required value |
|---|---:|
| Action horizon `H` | 50 |
| Episodes | 500 |
| Replan interval | 5 |
| Views | 2 |
| Token length | T=10 |

A material task-success-rate regression requires investigation and review.
A run using this 500-episode protocol must report its completed count out of
500 and the corresponding percentage. The current paired campaign uses the
explicit H=10, complete-prompt, 10-task-by-10-trial protocol above instead.

The tables below are historical 100-episode reference runs. They are useful comparison points, but they do **not** satisfy the current 500-episode pull-request accuracy standard and must not be reported as new formal PR accuracy results.

### 7.1 Thor T=10, 2 views: historical 100-episode reference

| Platform | Precision | Input | LIBERO-10 task success | Official reference |
|---|---|---|---:|---:|
| Thor | BF16 | 2 views / T=10 | **93/100 (93%)** | 92.4% |
| Thor | FP8 | 2 views / T=10 | **94/100 (94%)** | 92.4% |

### 7.2 Orin T=10, 2 views: historical 100-episode reference

The naive W8A8 PTQ scaling strategy used by language QKV causes accuracy loss on the INT8 path.

| Platform | Precision | Input | LIBERO-10 task success | Official reference |
|---|---|---|---:|---:|
| Orin | BF16 | 2 views / T=10 | **93/100 (93%)** | 92.4% |
| Orin | INT8 W8A8 | 2 views / T=10 | **91/100 (91%)** | 92.4% |

## 8. Timing boundaries

Every result cell must report both timing boundaries:

```text
Graph replay
  = steady-state CUDA Graph launch + synchronize

Input update + graph
  = update already-resized uint8 images, tokens, and noise
    + CUDA preprocessing
    + graph replay
    + synchronize
```

Image decoding, camera rotation, and CPU resize are outside both boundaries by default. If Python/client end-to-end latency is also measured, report it in a separate table.

Use Nsight Systems and Nsight Compute only to locate bottlenecks. Profilers change timing behavior, so profiler-instrumented measurements must not replace uninstrumented formal benchmark results.
