# GR00T performance and tuning

- Optimize Thor SM110 BF16/FP8 and Orin SM87 BF16/W8A8 inference.
- Fuse quantization, normalization, RoPE, attention, and projection epilogues.
- GR00T explicitly selects optimized operations; shared default paths and interfaces remain compatible.

## Performance

| Device | Precision | 1-view P50 | 2-view P50 |
| --- | --- | ---: | ---: |
| Thor | BF16 | | |
| Thor | FP8 | | |
| Orin | BF16 | | |
| Orin | W8A8 | | |

## LIBERO-10 task accuracy

| Device | Precision | Episodes | Successes | Success rate |
| --- | --- | ---: | ---: | ---: |
| Thor | BF16 | 100 | 94 | 94.0% |
| Thor | FP8 | 100 | 90 | 90.0% |
| Orin | BF16 | 100 | 93 | 93.0% |
| Orin | W8A8 | 100 | 93 | 93.0% |

Two views, 10 episodes for each of the 10 tasks, a 720-step limit, and eight
executed actions per chunk.

## Local tuning

Prepare the executable and fixtures using the [benchmark setup](gr00t-n1.7.md#fixed-input-benchmark).
Set the asset paths below. Use `bf16` or `fp8` on Thor, and `bf16` or `int8` on Orin.
FP8 additionally requires the checkpoint's [calibration file](gr00t-n1.7.md#fp8-calibration).

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

`--autotune` fills missing entries. Reuse the resulting database without
`--autotune` for subsequent benchmarks. Use a new run directory to retune.
