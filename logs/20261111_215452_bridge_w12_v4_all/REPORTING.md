# What gets reported from this run

Recipe `bridge_w12_v4_all`, seed 42: the 12 m / 4 cm configuration of the 2x2 control with
the held-out validation blocks folded back into training (`--val_in_train`), 400 epochs,
early stopping disabled. This is the released checkpoint.

The rule was fixed in `logs/W12_V4_ALL.md` before either seed started: `final_model.pth`
only, scored with `--domain bridge_w12_v4`, protocols `single` and `overlap_mirror`;
`best_model.pth` is not opened; seed 42 is the headline whatever the ordering of the seeds.
