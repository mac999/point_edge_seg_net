#!/usr/bin/env bash
# Score the checkpoints into the result files the reported numbers come from.
#
#   score.sh <key|all>
#
# Each output keeps its original file name next to the checkpoint (or under reproduce/ for
# the two ablation files that live there). Existing files are kept unless FORCE=1. The
# probability dumps (--save_probs, gigabytes) are what fuse.sh needs.
#
# Geometry rule: a checkpoint is scored with --domain <score_domain> from runs.tsv, i.e. on
# the window and voxel it was trained with; the all-blocks runs use their base recipe. The
# two test_full_final_model.json files of the 6 m and 12 m runs are the deliberate
# exception: scored at the default 2 m window at full resolution to show the mismatch.
#
# Cost on one GPU: single about 6 min per pass, overlap_mirror about 25 min (12 m / 4 cm).
set -euo pipefail
source "$(dirname "$0")/common.sh"
[ $# -ge 1 ] || { echo "usage: $0 <key|all>"; echo "keys:"; all_keys | sed 's/^/  /'; exit 1; }
sel="$1"
check_commit; check_python
cd "$ROOT"; mkdir -p logs reproduce

score() {  # <key> <protocol> <out> [probs dir|-] [extra evaluate_full.py flags...]
    local key="$1" proto="$2" out="$3" probs="${4:--}"; shift 4 || shift $#
    local run domain outp
    run="$(run_dir_for "$key")"
    [ -n "$run" ] || { echo "[skip] $key: no run directory (train.sh $key, or RUN_${key^^}=...)"; return; }
    [ -f "$run/final_model.pth" ] || { echo "[skip] $key: $run/final_model.pth missing"; return; }
    domain="$(tsv_field "$key" score_domain)"
    [ -f "domains/$domain.json" ] || { echo "[skip] $key: domains/$domain.json is not in this checkout"; return; }
    case "$out" in reproduce/*) outp="$out" ;; *) outp="$run/$out" ;; esac
    if [ -e "$outp" ] && [ "${FORCE:-0}" != 1 ]; then log "keep  $outp"; return; fi
    local args=(--domain "$domain" --protocol "$proto" --model_weights "$run/final_model.pth" --out "$outp")
    [ "${FORCE:-0}" = 1 ] && args+=(--overwrite)
    [ "$probs" != - ] && args+=(--save_probs "$run/$probs")
    log "score $key $proto -> $outp"
    run "$PYTHON" evaluate_full.py "${args[@]}" "$@" 2>&1 | tee -a "logs/score_${key}.log"
}
MISMATCH=(--window 2.0 --stride 2.0 --column_grid 0.0)   # default geometry, not the trained one

keys="$([ "$sel" = all ] && all_keys || echo "$sel")"
for key in $keys; do
    case "$key" in
    w2)  score w2  single test_full_final.json probs_single_probs ;;
    w6)  score w6  single         matched_final_model.json         probs_single_probs
         score w6  overlap_mirror matched_s3_tta2_final_model.json probs_s3_tta2
         score w6  single         test_full_final_model.json - "${MISMATCH[@]}" ;;
    w12) score w12 single         matched_final_model.json         probs_single_probs
         score w12 overlap_mirror matched_s6_tta2_final_model.json probs_s6_tta2
         score w12 single         test_full_final_model.json - "${MISMATCH[@]}" ;;
    w24) score w24 single         reproduce/ablate_w24_single.json probs_single_probs
         score w24 overlap_mirror matched_s12_tta2_final_model.json probs_s12_tta2 ;;
    w6_gpos) score w6_gpos single matched_final_model.json ;;
    w6_sol)  score w6_sol  single reproduce/ablate_sol_single.json ;;
    w6_v8)   score w6_v8   single matched_single_final.json probs_single_probs ;;
    w12_v4)  score w12_v4  single         matched_single_final.json probs_single_probs
             score w12_v4  overlap_mirror matched_om_final.json     probs_om_final ;;
    w6_all_s42) score w6_all_s42 single         matched_single_final.json         probs_single_final
                score w6_all_s42 overlap_mirror matched_overlap_mirror_final.json probs_om_final ;;
    w6_all_s1|w6_all_s2) score "$key" single matched_single_final.json ;;
    w12_v4_all_s42)
        score w12_v4_all_s42 single         matched_single_final.json         probs_single_probs
        score w12_v4_all_s42 mirror         matched_mirror_final.json
        score w12_v4_all_s42 overlap        matched_overlap_final.json
        score w12_v4_all_s42 overlap_mirror matched_overlap_mirror_final.json probs_om_final
        score w12_v4_all_s42 single matched_tls3_single_final.json - --processed_data_path bridge/processed_tls3
        score w12_v4_all_s42 single matched_mls3_single_final.json - --processed_data_path bridge/processed_mls ;;
    w12_v4_all_s1)
        score w12_v4_all_s1 single         matched_single_final.json probs_single_probs
        score w12_v4_all_s1 overlap_mirror matched_om_final.json     probs_om_final ;;
    *) die "unknown key '$key'" ;;
    esac
done
