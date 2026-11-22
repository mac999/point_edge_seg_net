#!/usr/bin/env bash
# Retrain one or all of the runs in runs.tsv.
#
#   train.sh <key|all> [extra train_model.py flags]
#
# Recipe, seed and the one per-run flag (--learning_rate 0.0015 on the 2 m run) come from
# runs.tsv; the recipe is domains/<train_domain>.json. Each run lands in
# logs/<timestamp>_<domain>/ with a `paper_key` marker so score.sh finds it later.
#
# Budgets on one GPU: 2 m 1 day; 6 m held-out 38 h; 12 m / 8 cm 25 h; 24 m 10 h; 6 m / 8 cm 15 h;
# 12 m / 4 cm held-out 71 h; 6 m all-blocks 55 h; 12 m / 4 cm all-blocks 96 h. Runs are
# independent; use CUDA_VISIBLE_DEVICES to spread them over GPUs.
#   setsid nohup scripts/paper_runs/train.sh w12_v4_all_s42 > /dev/null 2>&1 &
#
# Expected: held-out runs stop at epoch 346 (patience 10 after the cosine anneal), the
# all-blocks runs complete 400, the 24 m run 800, the 2 m run about 144. Results land
# within the seed spread (0.35 mIoU at 6 m over three seeds, 1.97 at 12 m over two); they
# are not bit-exact across GPUs or library versions.
set -euo pipefail
source "$(dirname "$0")/common.sh"
[ $# -ge 1 ] || { echo "usage: $0 <key|all> [extra flags]"; echo "keys:"; all_keys | sed 's/^/  /'; exit 1; }
sel="$1"; shift
check_commit; check_python
cd "$ROOT"; mkdir -p logs
keys="$([ "$sel" = all ] && all_keys || echo "$sel")"
for key in $keys; do
    domain="$(tsv_field "$key" train_domain)"; seed="$(tsv_field "$key" seed)"; extra="$(tsv_field "$key" extra_flags)"
    [ -n "$domain" ] || die "unknown key '$key' (see runs.tsv)"
    [ -f "domains/$domain.json" ] || { echo "[skip] $key: domains/$domain.json is not in this checkout"; continue; }
    if [ -n "$(run_dir_for "$key")" ] && [ "${RETRAIN:-0}" != 1 ]; then
        log "$key: run directory exists ($(run_dir_for "$key")); RETRAIN=1 to train another"; continue
    fi
    before="$(ls -d logs/*_"$domain" logs/*_"$domain"-[0-9]* 2>/dev/null || true)"
    log "$key: train_model.py --domain $domain --seed $seed $extra $* (GPU $CUDA_VISIBLE_DEVICES)"
    # shellcheck disable=SC2086
    run "$PYTHON" train_model.py --domain "$domain" --seed "$seed" $extra "$@" 2>&1 | tee -a "logs/train_${key}.log"
    [ "${DRY_RUN:-0}" = 1 ] && continue
    after="$(ls -d logs/*_"$domain" logs/*_"$domain"-[0-9]* 2>/dev/null || true)"
    new="$(comm -13 <(echo "$before") <(echo "$after") | tail -1)"
    [ -n "$new" ] || die "$key: no new run directory under logs/"
    echo "$key" > "$new/paper_key"
    log "$key: done -> $new"
done
