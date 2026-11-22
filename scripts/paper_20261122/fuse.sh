#!/usr/bin/env bash
# Point-wise late fusion of two checkpoints from the probability dumps written by score.sh.
# Equal weights, fixed before scoring. CPU only.
set -euo pipefail
source "$(dirname "$0")/common.sh"
check_commit; check_python
cd "$ROOT"
probs() {  # <key> <dir> [fallback dir] -> existing probs dir or nothing
    local run; run="$(run_dir_for "$1")"; [ -n "$run" ] || return 0
    for d in "$2" "${3:-}"; do [ -n "$d" ] && [ -d "$run/$d" ] && { echo "$run/$d"; return; }; done
}
fuse() {  # <out> <probsA> <probsB>
    local out="$1" a="$2" b="$3"
    [ -n "$a" ] && [ -n "$b" ] || { echo "[skip] $out: a member's probability dump is missing (score.sh)"; return; }
    if [ -e "$out" ] && [ "${FORCE:-0}" != 1 ]; then log "keep  $out"; return; fi
    local args=(--probs "$a" "$b" --processed_data_path bridge/processed --test_area test
                --config configs/model_params_semanticbridge.json --out "$out")
    [ "${FORCE:-0}" = 1 ] && args+=(--overwrite)
    log "fuse  $out"
    run "$PYTHON" tools/fuse_votes.py "${args[@]}"
}
fuse logs/fusion_w6_w12_single_final.json "$(probs w6 probs_single_probs)" "$(probs w12 probs_single_probs)"
fuse logs/fusion_w6_w24_single_final.json "$(probs w6 probs_single_probs)" "$(probs w24 probs_single_probs)"
fuse logs/fusion_w6all_w12_om.json "$(probs w6_all_s42 probs_om_final)" "$(probs w12 probs_om_final probs_s6_tta2)"
