# GR00T optimization and local tuning

GR00T owns the precision and device choices for these optimizations: Thor
(SM110) BF16/FP8 and Orin (SM87) BF16/W8A8. The CUDA layer exposes explicit
operations for projection epilogues, packed quantization, normalization, RoPE,
and attention. Existing generic entries remain the fallback; other models do
not opt into the new operations through a matching shape alone.

The optimization work does not install or replace a shared `tactics.json`.
Tuning results and FP8 calibration remain local deployment artifacts, following
[the existing GR00T workflow](gr00t-n1.7.md#fixed-input-benchmark).

## What tuning reproduces

Online GEMM tuning searches implemented provider candidates for the actual
operands. It does not invent kernels or discover model-level fusion, layout
reuse, or quantization boundaries. Those choices are explicit in GR00T code.
The ordinary BF16/FP8 candidate sets already include the heuristic ranks and
Thor BF16 custom configurations present in the historical winning databases.
A fresh run can evaluate these candidates, but need not choose identical winners.

The historical Orin W8A8 database contained CUTLASS IDs 1 and 6 that the ordinary
online candidate list does not enumerate. Its QKV ID 1 was bypassed by explicit
bias fusion. Its FC1 ID 6 was used to qualify a separate fused implementation;
it was not itself the fused kernel. The integrated explicit FC1 path removes
that dependency on a saved ordinary-GEMM plan. The public W8A8 provider retains
its original candidates and dispatch. A saved historical ID is not evidence
that the current online search can generate it.

Autotune evaluates operator latency and numerical tolerance on its current
inputs. That does not establish full-model latency, bitwise identity, or
LIBERO success rate. In particular, a faster GEMM that passes its local tolerance
can still change the outputs and trajectories of a closed-loop policy. Validate
the complete model after tuning.

## Generate a local database

Use the same checkpoint, official processor fixtures, calibration, target SM,
and CUDA/cuBLAS versions when comparing revisions. Provide both one-view and
two-view fixtures so both token counts are exercised. Run only one GPU job per
device; record power, clock, fan and temperature settings for comparisons.

The following Bash example starts a separate database for each device and
precision. Supply absolute asset and binary paths. Use `bf16` or `fp8` on Thor,
and `bf16` or `int8` on Orin. FP8 calibration is a separate numerical input,
required even without a tactic database; see
[FP8 calibration](gr00t-n1.7.md#fp8-calibration).

```bash
set -euo pipefail
: "${GR00T_BENCH:?prebuilt gr00t_bench executable}"
: "${GR00T_CHECKPOINT:?}" "${GR00T_BACKBONE:?}"
: "${GR00T_FIXTURE_1V:?}" "${GR00T_FIXTURE_2V:?}"
: "${GR00T_DEVICE_LABEL:?thor or orin}" "${GR00T_PRECISION:?bf16, fp8 or int8}"
gr00t_calibration=-
if [[ $GR00T_PRECISION == fp8 ]]; then
  : "${GR00T_CALIBRATION:?validated calibration for this checkpoint}"
  gr00t_calibration=$GR00T_CALIBRATION
fi
gr00t_parent="$PWD/devlocal/gr00t-local-tuning"
mkdir -p "$gr00t_parent"
gr00t_run=$(mktemp -d "$gr00t_parent/$GR00T_DEVICE_LABEL-$GR00T_PRECISION.XXXXXX")
gr00t_tactics="$gr00t_run/tactics.json"
for gr00t_view in 1v 2v; do
  gr00t_fixture=$GR00T_FIXTURE_1V
  [[ $gr00t_view == 1v ]] || gr00t_fixture=$GR00T_FIXTURE_2V
  "$GR00T_BENCH" "$GR00T_CHECKPOINT" "$GR00T_BACKBONE" "$gr00t_fixture" \
    "$GR00T_PRECISION" 0 1 1 "$gr00t_calibration" "$gr00t_tactics" \
    "$gr00t_run/tune-$gr00t_view.json" --autotune \
    >"$gr00t_run/tune-$gr00t_view.stdout" 2>"$gr00t_run/tune-$gr00t_view.stderr"
done
```

`--autotune` fills missing keys; it does not retune existing exact entries.
Use a fresh run directory for an independent tuning experiment. The file is
explicitly selected, so the benchmark does not write a shared hardware database.
After tuning, run numerical checks and performance rounds without `--autotune`,
passing this same local database. Keep raw samples, outputs, warnings, effective
routes, source revision, executable hash, and library identities together.

## Historical source measurements

The following September 28, 2026 measurements reused the September 25 accepted
source binaries. They **do not validate a new combined build**. Each case pooled
600 samples from three rounds of 30 warmups and 200 iterations. Timing covers
preprocessed host tensors through synchronized CUDA Graph model-core execution
and action D2H; it excludes loading, preprocessing, and simulation.

| Source stack | Source commit | One-view P50 / P95 (ms) | Two-view P50 / P95 (ms) |
|---|---|---:|---:|
| Thor BF16 | `b321a421fd428b05440f31c316b81ae158f1e468` | 52.277 / 52.862 | 54.573 / 55.148 |
| Thor FP8 | `b6ae3818942c0c50c59c32952ae5f89cadba985b` | 32.754 / 33.074 | 36.353 / 36.526 |
| Orin BF16 | `36d125854b340acfe8009615b1c9fabb4588a8a4` | 75.399 / 75.615 | 84.498 / 84.732 |
| Orin W8A8 | `36d125854b340acfe8009615b1c9fabb4588a8a4` | 56.802 / 56.952 | 65.086 / 65.277 |

Thor used CUDA/cuBLAS 13.0 and driver 580.00. Orin used CUDA 13.2, cuBLAS 13.4
and driver 595.78. Both used their existing MAXN settings, with no clock/fan
changes. Orin's dynamic clocks differ from the locked September 25 environment;
these are not contemporaneous performance A/B results. Fixed outputs matched
the corresponding accepted outputs, not a newly run closed-loop evaluation.

Historical September 25 LIBERO-10 success counts were Thor BF16 88/100,
Thor FP8 87/100, Orin BF16 93/100, and Orin W8A8 91/100. All episodes completed
without technical errors. These counts do not prove no regression relative to
older results, and do not validate the integrated revision. A new release gate
must compare a declared baseline with matching protocol and episode-level noise;
a reused stream seed alone does not keep later episodes paired after an earlier
episode changes its number of inference calls.
