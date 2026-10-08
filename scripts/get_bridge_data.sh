#!/usr/bin/env bash
# ============================================================
#  Fetch the SemanticBridge scans and convert them to the
#  format the pipeline reads. ~8 GB download, ~14 GB after
#  conversion; nothing here is redistributed by this repo.
#
#  Usage (from anywhere):  ./scripts/get_bridge_data.sh
#  Re-running is safe: the download resumes and conversion
#  skips scenes that are already present.
# ============================================================
set -euo pipefail
cd "$(dirname "$0")/.."            # repo root, whatever it is called
ROOT="$PWD"
RAW="bridge/raw"
RECORD="https://zenodo.org/api/records/18738225/files"

mkdir -p "$RAW"
need() { command -v "$1" >/dev/null || { echo "[ERROR] '$1' not found" >&2; exit 1; }; }
need curl; need unzip

for f in tls_dataset mls_data; do
    if [ -d "$RAW/${f%%_*}" ] && [ -n "$(ls -A "$RAW/${f%%_*}" 2>/dev/null)" ]; then
        echo "== $f already unpacked, skipping"
        continue
    fi
    echo "== downloading $f.zip (resumable)"
    curl -L -C - --retry 5 --retry-delay 10 -o "$RAW/$f.zip" "$RECORD/$f.zip/content"
done
[ -f "$RAW/tls_dataset.zip" ] && unzip -q -o "$RAW/tls_dataset.zip" -d "$RAW/tls"
[ -f "$RAW/mls_data.zip" ]    && unzip -q -o "$RAW/mls_data.zip"    -d "$RAW/mls"

PYTHON="${PYTHON:-python}"
echo "== converting TLS scans -> bridge/processed (train/ and test/ by the official split)"
"$PYTHON" convert_dataset.py --dataset semanticbridge \
    --input_dir "$RAW/tls" --output_dir bridge/processed

# The three MLS scans of test bridges are the cross-sensor set; they are scored, never
# trained on, so they go to their own tree and are all routed to test/.
if [ -d "$RAW/mls/val" ]; then
    echo "== converting MLS scans -> bridge/processed_mls"
    "$PYTHON" convert_dataset.py --dataset semanticbridge \
        --input_dir "$RAW/mls/val" --output_dir bridge/processed_mls --split test
    # The same three bridges on the other sensor, for the paired comparison.
    mkdir -p bridge/processed_tls3/test
    for b in 13 17 19; do
        ln -sf "../../processed/test/bridge_${b}_fr_rtc.pt" bridge/processed_tls3/test/ 2>/dev/null \
            || cp "bridge/processed/test/bridge_${b}_fr_rtc.pt" bridge/processed_tls3/test/
    done
fi

echo
echo "done. Next:  ./run_domain_eval.sh bridge_w6 logs/20260929_150153_bridge_w6/final_model.pth single"
