# Runs behind the reported SemanticBridge numbers

A fixed record of the thirteen training runs and their scoring commands, so the figures in
the release notes and the accompanying study can be regenerated after the code moves on.

| file | content |
|---|---|
| `runs.tsv` | one row per run: key, training recipe, scoring recipe, seed, extra flags, original run directory, epochs, single-view mIoU |
| `checkpoints.sha256` | hash and size of each run's `final_model.pth` |
| `requirements_frozen.txt` | `pip freeze` of the environment the runs were made in (Python 3.12, torch 2.7.1+cu128, PyG 2.8.0) |
| `common.sh` | pinned commit, interpreter, run-directory lookup |
| `check_env.sh` | read-only check of code, Python stack, data and checkpoints |
| `train.sh` | retrain one run or all |
| `score.sh` | score the checkpoints into the result JSONs |
| `fuse.sh` | the three two-model fusion rows |

The code is pinned to commit `9a74a9e`. The scripts refuse to run on a checkout whose
tracked files differ from it (`ALLOW_DRIFT=1` overrides). On a later revision, use a
worktree of the pinned commit instead:

```bash
git worktree add ../pesn_paper 9a74a9ef2ff1fe484c3bce72f1de9c912a265be8
export PESN_ROOT=$(realpath ../pesn_paper)
```

## Use

```bash
scripts/get_bridge_data.sh                 # SemanticBridge, 15 train / 5 test scenes
scripts/paper_runs/check_env.sh            # code, stack, data, checkpoints
scripts/paper_runs/score.sh all            # result JSONs from the checkpoints (GPU, hours)
scripts/paper_runs/fuse.sh                 # fusion rows from the probability dumps (CPU)
scripts/paper_runs/train.sh <key>          # retrain one run, then score.sh <key>
```

`PYTHON` selects the interpreter (default: conda env `pesn`), `CUDA_VISIBLE_DEVICES` the GPU,
`DRY_RUN=1` prints the commands, `FORCE=1` rewrites existing result files.

## Rules the scripts encode

- A checkpoint is scored on the window and voxel it was trained with (`score_domain` in
  `runs.tsv`). The all-blocks runs are scored with their base recipe (`bridge_w6`,
  `bridge_w12_v4`). The two `test_full_final_model.json` files of the 6 m and 12 m runs are
  scored at the default 2 m window on purpose; they measure the geometry mismatch.
- `final_model.pth` is the reported checkpoint of every run; `best_model.pth` is never scored.
- The 2 m run was trained with `--learning_rate 0.0015`; the recipe file says 0.003.
- Seed repeats of a recipe are scored single view only, with the same file name.
- Retrained runs land in `logs/<timestamp>_<domain>/` with a `paper_key` marker that
  `score.sh` and `fuse.sh` use to find them. `RUN_<KEY>=<dir>` points a key at any directory.

Three recipes in `runs.tsv` are not in the repository: `bridge_w24`, `bridge_w6_gpos` and
`bridge_w6_sol` (with `model_params_semanticbridge_gpos.json` and
`structure_presets.local.json`). The scripts skip those keys when the files are absent.

Retraining reproduces each number within the seed spread, 0.35 mIoU over three seeds for
the 6 m recipe and 1.97 over two for the 12 m recipe, not bit-exactly. Rescoring an existing
checkpoint repeats to about 0.01 mIoU.
