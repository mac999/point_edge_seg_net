#!/usr/bin/env bash
# Shared by the scripts in this directory; source it, do not run it.
#
# Pins the code revision the reported runs were trained and scored with, resolves the
# interpreter and the run directory of each key in runs.tsv.
#
# Environment:
#   PESN_ROOT            repository root (default: two levels above this directory)
#   PYTHON               interpreter with torch + torch_geometric (default: conda env pesn)
#   CUDA_VISIBLE_DEVICES GPU to use (default 0)
#   ALLOW_DRIFT=1        run although the code differs from the pinned commit
#   DRY_RUN=1            print the commands instead of running them
#   FORCE=1              overwrite existing score files (default: keep them)
#   RUN_<KEY>=<dir>      run directory of one key by hand (e.g. RUN_W6=logs/x)
ROOT_COMMIT=9a74a9ef2ff1fe484c3bce72f1de9c912a265be8   # code the runs were made with
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${PESN_ROOT:-$(cd "$HERE/../.." && pwd)}"
PYTHON="${PYTHON:-$HOME/miniconda3/envs/pesn/bin/python}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
RUNS_TSV="$HERE/runs.tsv"

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }
die() { echo "[ERROR] $*" >&2; exit 1; }

run() {  # run or print a command
    if [ "${DRY_RUN:-0}" = 1 ]; then
        printf '  (dry) %q' "$1"; shift; printf ' %q' "$@"; echo
    else
        "$@"
    fi
}

code_matches_pin() {  # tracked files outside this directory equal the pinned commit, no local edits
    git -C "$ROOT" diff --quiet "$ROOT_COMMIT" HEAD -- . ':(exclude)scripts/paper_20261122' ':(exclude)README.md' ':(exclude)docs' 2>/dev/null \
        && git -C "$ROOT" diff --quiet HEAD -- . ':(exclude)scripts/paper_20261122' ':(exclude)README.md' ':(exclude)docs' 2>/dev/null
}

check_commit() {
    code_matches_pin && return
    echo "[warn] the code differs from the pinned commit ${ROOT_COMMIT:0:7}." >&2
    echo "       Pinned checkout:  git -C $ROOT worktree add ../pesn_paper $ROOT_COMMIT" >&2
    echo "       then PESN_ROOT=<that path> for these scripts." >&2
    [ "${ALLOW_DRIFT:-0}" = 1 ] || die "set ALLOW_DRIFT=1 to run on the current code anyway."
}

check_python() {
    "$PYTHON" -c 'import torch, torch_geometric, torch_cluster' 2>/dev/null \
        || die "'$PYTHON' lacks torch / torch_geometric / torch_cluster; set PYTHON=... (requirements_frozen.txt)."
}

tsv_field() {  # <key> <column> -> value
    awk -F'\t' -v k="$1" -v c="$2" 'NR==1 {for (i=1;i<=NF;i++) if ($i==c) col=i; next} $1==k {print $col}' "$RUNS_TSV"
}
all_keys() { awk -F'\t' 'NR>1 {print $1}' "$RUNS_TSV"; }

run_dir_for() {  # <key> -> run directory (absolute), or nothing
    local key="$1" var="RUN_${1^^}" d orig domain
    if [ -n "${!var:-}" ]; then echo "${!var}"; return; fi
    domain="$(tsv_field "$key" train_domain)"
    for d in "$ROOT"/logs/*_"$domain" "$ROOT"/logs/*_"$domain"-[0-9]*; do   # trained by train.sh
        [ -f "$d/paper_key" ] && [ "$(cat "$d/paper_key")" = "$key" ] && { echo "$d"; return; }
    done
    orig="$(tsv_field "$key" orig_dir)"
    [ -n "$orig" ] && [ -d "$ROOT/$orig" ] && echo "$ROOT/$orig"
}
