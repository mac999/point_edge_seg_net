# Worked example: bridges (SemanticBridge)

Outdoor bridge structures from terrestrial and mobile laser scans, 9 classes, the dataset's
own 15/5 split. The building counterpart is [building_example.md](./building_example.md);
both run the same code, differing in the dataset profile, the run recipe and the block
geometry.

This page is the procedure for getting from the public scans to the numbers below on a
clean machine. It documents the **released configuration only** — the study of why that
configuration was chosen (context/resolution behaviour, the preprocessing and scoring
defects found along the way, and the ablations) is being written up separately and is
deliberately not repeated here.

The dataset is not redistributed by this repository. It is published by its authors; the
scripts below fetch it from the authors' Zenodo record.

---

## Result

`logs/20260929_150153_bridge_w6/final_model.pth`, 3.06 M parameters (11.8 MiB), one GPU, 37.9 h of
training. Scored over **all 84,153,822 labelled test points** of the official split:

| | mIoU | OA |
|---|---|---|
| **This model**, single view | **69.81** | 91.61 |
| UNet3D | 70.7 | — |
| KPConv | 70.5 | — |
| PointTransformer v2 | 63.5 | — |

Baselines are as published by the dataset authors, at roughly 4.6x the parameters. All four
are single-view; see [protocols](#3-score-the-released-model) for the voted figures.

<p align="center">
<img src="./imgs/bridge_segmentation_example.png" width="820"
     alt="Elevation view of a held-out bridge: ground truth, prediction, and misclassified points"><br>
<sub>A held-out test bridge seen along its deck axis: labels, prediction, and where the two
disagree. Abutments close the deck at both ends, a pillar carries its middle.</sub>
</p>

---

## 1. Requirements

A CUDA GPU with about 6 GB free for scoring (training the released recipe peaks near 34 GB),
roughly 25 GB of disk for the scans and their converted form, and the environment described
in the main [README](./README.md#installation). Versions are pinned in `requirements.txt`;
`torch 2.7.1 / cu128` is what the released checkpoint was produced with.

## 2. Get the data

```bash
./scripts/get_bridge_data.sh          # Linux / macOS
scripts\get_bridge_data.bat           # Windows
```

Roughly 8 GB of download and 15 minutes of conversion. Re-running is safe: the download
resumes and conversion skips what is already there. Afterwards:

```
bridge/raw/                 the downloaded scans, untouched
bridge/processed/           TLS scans, split train/ (15 bridges) and test/ (5) as published
bridge/processed_mls/       the three MLS scans of test bridges (scored, never trained on)
bridge/processed_tls3/      the same three bridges on TLS, for the paired comparison
```

The split is the dataset's own: `test_stems` in `dataset_profiles.json` lists the five test
bridges, and the converter routes files by that list. Nothing is held out beyond it — the
validation set used during training is carved spatially out of the 15 training bridges
(`split_train_val`, 20 % of super-cells), so no test bridge is seen in any form.

## 3. Score the released model

```bash
./scripts/run_domain_eval.sh bridge_w6 logs/20260929_150153_bridge_w6/final_model.pth single
```

`--domain bridge_w6` reads the block geometry, voxel lattice and architecture out of
`domains/bridge_w6.json`, so none of it has to be retyped; `single` is the protocol to use
when comparing against published baselines, which vote once.

Expected, over all 84,153,822 labelled test points:

| protocol | stride | views | mIoU | OA |
|---|---|---|---|---|
| `single` | window | 1 | **69.81** | 91.61 |
| `mirror` | window | 2 | 70.02 | 91.68 |
| `overlap` | window/2 | 1 | 70.57 | 91.92 |
| `overlap_mirror` | window/2 | 2 | 70.64 | 91.94 |

Cross-sensor, on the three bridges captured with both scanners:

```bash
# the wrapper is for the plain case; point the scorer at another tree directly
python evaluate_full.py --domain bridge_w6 --protocol single \
    --model_weights logs/20260929_150153_bridge_w6/final_model.pth \
    --processed_data_path bridge/processed_tls3 --out tls3.json   # expect mIoU 74.82
python evaluate_full.py --domain bridge_w6 --protocol single \
    --model_weights logs/20260929_150153_bridge_w6/final_model.pth \
    --processed_data_path bridge/processed_mls  --out mls3.json   # expect mIoU 61.97
```

All four protocols plus the cross-sensor pair in one go:

```bash
./scripts/run_bridge_reproduce.sh logs/20260929_150153_bridge_w6/final_model.pth
```

### What counts as a match

Scoring is not bit-exact across runs. Measured here over four repeats at identical settings,
the spread is **0.01 mIoU** (sd 0.005): a few thousand of 29 million points sit on a near-tie
between overlapping votes, and GPU reduction order decides them. Changing the **scoring batch
size** moves the result by more than that — about 0.08 mIoU — which is why `--domain` takes
the batch width from the recipe too. Treat anything under ~0.05 mIoU as noise; a different
GPU or driver may land anywhere in that band.

Released runs keep the same layout as every other run in `logs/`: weights, the epoch log, the
curves and the scores they produced.

## 4. Retrain, if you want to

```bash
./scripts/run_domain_train.sh bridge_w6         # ~38 h on one GPU, peak ~34 GB
```

The first run builds a block cache under `bridge/chunks_w6/` (about 4 GB, 4,097 blocks) and
reuses it afterwards. The build is deterministic: rebuilding from the same `bridge/processed`
tree reproduces the same 4,097 blocks with identical contents and the same 2,357 / 597 /
1,143 train / val / test split, so a fresh machine trains on the same inputs.

Training itself is **not** bit-reproducible — data order, augmentation sampling and CUDA
accumulation all vary — and no multi-seed study was run, in line with how this benchmark's
published baselines report single runs. Logs go to a fresh `logs/<timestamp>_<domain>`
directory; an existing run is never overwritten.

Two recipes ship: `bridge_w6`, the released configuration, and `bridge`, the plain 2 m
baseline it is compared against. `./scripts/run_domain_train.sh` with no argument lists whatever is
present. The recipes used for the ablations are released with the write-up.

## 5. If a number does not match

| symptom | likely cause |
|---|---|
| far below the table, all classes | geometry mismatch — score with `--domain`, not hand-typed flags |
| one class at 0.00 | the config's `class_weights` are uniform; regenerate with `compute_class_weights.py` |
| out of memory while scoring | the scoring batch default assumes dense blocks; `--domain` sets the right width |
| off by < 0.05 mIoU | expected run-to-run variation, see above |
| a cache error about feature width | a stale block cache from a different feature configuration; point `--block_data_path` somewhere new |
