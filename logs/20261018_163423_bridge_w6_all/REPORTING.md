# What gets reported from this run

Recipe `bridge_w6_all`: the released 6 m / 4 cm recipe with the held-out validation blocks
folded back into training (`--val_in_train`), so that the training set matches what the
benchmark baselines used. 2954 blocks instead of 2357.

**Decided before any test score was computed (2026-10-19, at epoch 110 of 400):**

1. The reported figure is `final_model.pth` at the fixed 400 epochs, scored single-view
   with `--domain bridge_w6`. Nothing else.
2. `best_model.pth` is not reported and not used. Its selection metric is in-sample here,
   because the validation blocks are in the training set.
3. No intermediate checkpoint is scored on the test split to decide anything. The run is
   reported whatever the number turns out to be, above or below the 69.81 of the
   held-out-validation run.

The point of the run is to remove one protocol difference against the published baselines,
not to produce a higher number. Selecting a checkpoint on the test split is the practice
the manuscript documents against in Table 5.
