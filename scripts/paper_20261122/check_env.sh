#!/usr/bin/env bash
# Check the code revision, the Python stack, the GPU, the data and the checkpoints.
# Read-only. Exit status 0 when everything needed for scoring is present.
set -uo pipefail
source "$(dirname "$0")/common.sh"
ok=0
say() { printf '  %-8s %s\n' "$1" "$2"; }

echo "== code ($ROOT)"
if code_matches_pin; then say ok "code equals the pinned commit ${ROOT_COMMIT:0:7}"
else say WARN "code differs from the pinned commit ${ROOT_COMMIT:0:7}; results may differ"; fi
for f in domains/bridge_w24.json domains/bridge_w6_gpos.json domains/bridge_w6_sol.json \
         configs/model_params_semanticbridge_gpos.json configs/structure_presets.local.json; do
    [ -f "$ROOT/$f" ] && say ok "$f" || say absent "$f (recipe not in the repository; its run cannot be retrained)"
done

echo "== python ($PYTHON)"
if "$PYTHON" - <<'PY' 2>/dev/null
import torch, torch_geometric, torch_cluster, torch_scatter, numpy
print(f"  ok       torch {torch.__version__}  pyg {torch_geometric.__version__}  numpy {numpy.__version__}")
print(f"  {'ok' if torch.cuda.is_available() else 'WARN':<8} cuda {'available' if torch.cuda.is_available() else 'NOT available'}")
PY
then :; else say FAIL "torch / torch_geometric / torch_cluster import failed (see requirements_frozen.txt)"; ok=1; fi

echo "== data ($ROOT/bridge)"
n() { ls "$ROOT/$1"/*.pt 2>/dev/null | wc -l; }
[ "$(n bridge/processed/train)" = 15 ] && say ok "processed/train: 15 scenes" || { say MISSING "processed/train has $(n bridge/processed/train) scenes, expected 15 (scripts/get_bridge_data.sh)"; ok=1; }
[ "$(n bridge/processed/test)"  = 5  ] && say ok "processed/test: 5 scenes"   || { say MISSING "processed/test has $(n bridge/processed/test) scenes, expected 5 (scripts/get_bridge_data.sh)"; ok=1; }
[ "$(n bridge/processed_tls3/test)" = 3 ] && say ok "processed_tls3/test: 3 scenes" || say absent "processed_tls3/test (cross-sensor TLS pair)"
[ "$(n bridge/processed_mls/test)"  = 3 ] && say ok "processed_mls/test: 3 scenes"  || say absent "processed_mls/test (cross-sensor MLS pair)"

echo "== checkpoints"
while read -r sha size path; do
    if [ -f "$ROOT/$path" ]; then
        if [ "$(sha256sum "$ROOT/$path" | cut -c1-64)" = "$sha" ]; then say ok "$path"; else say CHANGED "$path (sha256 differs)"; fi
    else say absent "$path (train.sh, or restore the weights)"; fi
done < "$HERE/checkpoints.sha256"
exit $ok
