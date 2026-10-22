# Seed repeats of the released recipe

Written 2026-10-22, before either run started and before any test score exists.

## What is being measured

The paper reports one training run per configuration, as the published baselines on this
benchmark do, and says so. That leaves the headline figure's training variance unmeasured:
the 0.01 mIoU spread documented elsewhere is *scoring* variance, measured by rescoring one
checkpoint, and says nothing about how much of 70.04 is the seed.

These two runs repeat the released recipe `bridge_w6_all` with the initialisation,
augmentation sampling and batch order reseeded. Nothing else changes: the same 2954 blocks,
the same 400-epoch budget, the same disabled early stopping.

    seed 1  -> logs/<timestamp>_bridge_w6_all  (GPU 1)
    seed 2  -> logs/<timestamp>_bridge_w6_all  (GPU 2)

The already-released run is seed 42.

## The rule, fixed in advance

1. Each run is scored once, `final_model.pth`, protocol `single`, `--domain bridge_w6`.
   `best_model.pth` is not opened: with `--val_in_train` its selection metric is in-sample.
2. All three seeds are reported. The released checkpoint stays the released checkpoint
   whatever the repeats say; it is not swapped for a luckier seed, because choosing a seed
   on the test set is the practice this paper criticises.
3. What the repeats may change is the *uncertainty* attached to reported differences. If
   the seed spread turns out to be larger than a difference the paper interprets, that
   interpretation is weakened and the text has to say so.
4. If a run crashes or is interrupted it is restarted from scratch, not resumed from a
   checkpoint chosen after the fact.

## What would count as a problem

The paper interprets differences of 0.23 mIoU (returning the withheld blocks) and larger.
A seed standard deviation above roughly 0.2 mIoU would mean the 0.23 cannot be read as an
effect from single runs, and the all-blocks claim would have to be restated as consistent
with but not demonstrated by the data. That outcome gets reported.
