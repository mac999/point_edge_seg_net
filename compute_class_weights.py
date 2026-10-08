"""Derive `class_weights` for a dataset config from the blocks training actually sees.

Why this exists: `class_weights` drives two separate mechanisms in train_model.py --
the focal loss's per-class weighting, and the rare-class block oversampler, which
takes `max(class_weights[c])` over the classes present in a block. Leave the weights
uniform and BOTH silently turn off: the oversampler reports "0/N blocks boosted" and
the loss treats a 0.05%-of-points class the same as a 51% one.

That is easy to miss because a uniform list looks like a valid config rather than a
disabled feature, so this script measures the real label distribution and writes
weights back into the config instead of anyone hand-guessing them.

    python compute_class_weights.py --blocks bridge/chunks --pattern '*__train__*.pt' \
        --config model_params_semanticbridge.json --write

Weights are (1/frequency)**power, rescaled to mean 1 and clipped. `--power 0.5`
(inverse square root, the default) reproduces the shape of the existing S3DIS
weights; 1.0 is full inverse frequency and is usually too aggressive.
"""

import argparse
import glob
import json
import os

import numpy as np
import torch


def label_counts(paths, num_classes):
	"""Point count per class across blocks, ignoring padding (label < 0)."""
	counts = np.zeros(num_classes, dtype=np.int64)
	for path in paths:
		y = torch.load(path, weights_only=False).y.reshape(-1).numpy()
		counts += np.bincount(y[y >= 0], minlength=num_classes)[:num_classes]
	return counts


def block_presence(paths, num_classes, min_share):
	"""How many blocks contain each class at >= `min_share` of their valid points.

	The oversampler only boosts a block when a rare class clears this share, so a class
	can be starved by block scarcity even after its loss weight is raised.
	"""
	present = np.zeros(num_classes, dtype=np.int64)
	for path in paths:
		y = torch.load(path, weights_only=False).y.reshape(-1).numpy()
		y = y[y >= 0]
		if not len(y):
			continue
		present += (np.bincount(y, minlength=num_classes)[:num_classes] >= min_share * len(y))
	return present


def derive_weights(counts, power, clip_min, clip_max):
	"""(1/frequency)**power, rescaled to mean 1 and clipped.

	Absent classes get the maximum weight rather than infinity, so a class missing from
	the sampled split does not blow up the whole vector.
	"""
	freq = counts / max(counts.sum(), 1)
	with np.errstate(divide='ignore'):
		weights = np.where(freq > 0, np.power(np.maximum(freq, 1e-12), -power), np.inf)
	finite = weights[np.isfinite(weights)]
	weights = np.minimum(weights, finite.max() if len(finite) else 1.0)
	weights = weights / weights.mean()
	return np.clip(weights, clip_min, clip_max)


def main():
	ap = argparse.ArgumentParser(description=__doc__,
								 formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument('--blocks', required=True, help='Cached block directory (e.g. bridge/chunks)')
	ap.add_argument('--pattern', default='*__train__*.pt', help='Which blocks to measure')
	ap.add_argument('--config', required=True, help='Dataset config JSON to read/update')
	ap.add_argument('--power', type=float, default=0.5,
					help='0.5 = inverse sqrt frequency (default), 1.0 = full inverse frequency')
	ap.add_argument('--clip_min', type=float, default=0.2, help='Lower bound on a weight')
	ap.add_argument('--clip_max', type=float, default=6.0, help='Upper bound on a weight')
	ap.add_argument('--min_share', type=float, default=0.01,
					help="Block share a class needs for the oversampler to count it present")
	ap.add_argument('--stride', type=int, default=1, help='Measure every Nth block (sampling)')
	ap.add_argument('--write', action='store_true', help='Write class_weights back into --config')
	args = ap.parse_args()

	with open(args.config, 'r', encoding='utf-8') as fh:
		config = json.load(fh)
	num_classes = config['num_classes']
	names = config.get('class_names') or [str(i) for i in range(num_classes)]

	paths = sorted(glob.glob(os.path.join(args.blocks, args.pattern)))[::args.stride]
	if not paths:
		raise SystemExit(f"no blocks matched {os.path.join(args.blocks, args.pattern)}")
	print(f"Measuring {len(paths)} blocks from {args.blocks}")

	counts = label_counts(paths, num_classes)
	present = block_presence(paths, num_classes, args.min_share)
	weights = derive_weights(counts, args.power, args.clip_min, args.clip_max)
	current = config.get('class_weights') or [1.0] * num_classes

	print(f"\n{'class':<18}{'share':>9}{'blocks':>12}{'current':>9}{'new':>8}")
	for c in range(num_classes):
		share = 100.0 * counts[c] / max(counts.sum(), 1)
		print(f"{names[c]:<18}{share:8.3f}%{present[c]:>7}/{len(paths):<5}"
			  f"{current[c]:9.2f}{weights[c]:8.2f}")

	starved = [names[c] for c in range(num_classes) if present[c] == 0]
	if starved:
		print(f"\nNOTE: {', '.join(starved)} never reach {args.min_share:.0%} of a block, so the "
			  f"oversampler cannot boost them -- raising their loss weight is the only lever.")

	if not args.write:
		print("\n(dry run -- pass --write to update the config)")
		return
	config['class_weights'] = [round(float(w), 3) for w in weights]
	with open(args.config, 'w', encoding='utf-8') as fh:
		json.dump(config, fh, indent=2)
		fh.write('\n')
	print(f"\nWrote class_weights to {args.config}")


if __name__ == '__main__':
	main()
