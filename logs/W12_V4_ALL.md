# All-blocks retraining of the 12 m / 4 cm recipe

Written 2026-11-11, before either run started and before any test score exists.

## Why

The 2x2 control separated the window from the voxel, which the density-preserving sweep
changes together. Single view, final checkpoints, all bridges trained with 20% of blocks held
out for validation:

| window \ voxel | 4 cm  | 8 cm  |
|----------------|-------|-------|
| 6 m            | 69.81 | 68.65 |
| 12 m           | 70.82 | 68.80 |

The sweep's 6 m optimum was a resolution effect. At 4 cm the 12 m window is 1.01 higher,
mostly on abutment (60.2 against 53.7).

## Runs

`bridge_w12_v4_all`: bridge_w12_v4 with the held-out blocks folded back into training,
400 epochs, early stopping disabled, matching `bridge_w6_all`.

    seed 42 -> logs/<timestamp>_bridge_w12_v4_all  (GPU 2)
    seed 1  -> logs/<timestamp>_bridge_w12_v4_all  (GPU 1)

## The rule, fixed in advance

1. Each run is scored once with `final_model.pth`, `--domain bridge_w12_v4`, protocols
   `single` and `overlap_mirror`. `best_model.pth` is not opened.
2. Both seeds are reported. Seed 42 is the headline and released checkpoint whatever the
   ordering of the two, as for `bridge_w6_all`.
3. No setting is changed after a test score is seen.

## Result (2026-11-16, both seeds scored once as fixed above)

| seed | run                                     | single mIoU / OA | voted mIoU / OA | abut. | pillar |
|------|-----------------------------------------|------------------|-----------------|-------|--------|
| 42   | logs/20261111_215452_bridge_w12_v4_all  | 70.80 / 92.44    | 71.87 / 92.75   | 61.4  | 45.8   |
| 1    | logs/20261112_083228_bridge_w12_v4_all  | 68.83 / 91.79    | 70.49 / 92.22   | 56.0  | 41.1   |

Single-view range 1.97 (mean 69.82), against 0.35 over three seeds of `bridge_w6_all`
(mean 70.24). Reading, under rule 2: seed 42 stays the released checkpoint. The +1.01 of
the window control is inside this spread and is no longer read as a mean gain. Both seeds
sit above all three 6 m seeds on ground, vegetation and overall accuracy and below them on
unlabelled and pillar, so the class shift is what the window change is reported to do.
