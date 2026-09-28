#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model_root="$repo_root/crates/apxinf-model/src"

# These directories are documented top-level infrastructure rather than model
# families. Add a directory only after its shared ownership is reviewed.
shared_dirs=(profiling vla)

is_shared_dir() {
    local candidate="$1"
    local shared
    for shared in "${shared_dirs[@]}"; do
        if [[ "$candidate" == "$shared" ]]; then
            return 0
        fi
    done
    return 1
}

families=()
for directory in "$model_root"/*; do
    [[ -d "$directory" && -f "$directory/mod.rs" ]] || continue
    family="$(basename "$directory")"
    is_shared_dir "$family" && continue
    families+=("$family")
done

violations=0
for family in "${families[@]}"; do
    family_dir="$model_root/$family"
    for other in "${families[@]}"; do
        [[ "$family" == "$other" ]] && continue
        pattern="(crate::${other}|super::super::${other})([^[:alnum:]_]|$)|^[[:space:]]*use[[:space:]]+crate::\\{[^;]*${other}::"
        if command -v rg >/dev/null 2>&1; then
            if rg -n -g '*.rs' "$pattern" "$family_dir"; then
                violations=1
            fi
        elif grep -R -n -E --include='*.rs' "$pattern" "$family_dir"; then
            violations=1
        fi
    done
done

if ((violations)); then
    echo 'model-family boundary violation: a family references another family directory' >&2
    echo 'copy architecture code locally or extract an explicitly reviewed shared module' >&2
    exit 1
fi

python3 "$repo_root/scripts/check_pi05_module_boundaries.py"

# GR00T selects its measured shapes and launch overrides in the model layer.
# Shared Rust operators must not acquire GR00T-named APIs or environment knobs.
cuda_policy_roots=(
    "$repo_root/crates/apxinf-cuda/src/kernels"
    "$repo_root/crates/apxinf-cuda/adapters"
    "$repo_root/crates/apxinf-cuda/kernels/custom"
)
gr00t_policy_pattern='APXINF_GR00T_|pub(\([^)]*\))?[[:space:]]+(unsafe[[:space:]]+)?(const[[:space:]]+)?fn[[:space:]]+(try_)?gr00t_'
if command -v rg >/dev/null 2>&1; then
    if rg -n -g '*.rs' -g '*.cu' -g '*.cuh' "$gr00t_policy_pattern" "${cuda_policy_roots[@]}"; then
        violations=1
    fi
elif grep -R -n -E --include='*.rs' --include='*.cu' --include='*.cuh' "$gr00t_policy_pattern" "${cuda_policy_roots[@]}"; then
    violations=1
fi
if ((violations)); then
    echo 'model/backend boundary violation: move GR00T API and launch policy to its model seam' >&2
    exit 1
fi

echo 'model-family boundary checks passed'
