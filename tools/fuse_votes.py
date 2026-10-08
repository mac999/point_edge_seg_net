# fuse_votes.py
# Point-wise late fusion of per-point softmax dumps written by `evaluate_full.py --save_probs`.
#
# Lets models trained on different voxel grids / window sizes (e.g. a 6 m / 4 cm model and a
# 24 m / 16 cm model) be combined without a shared lattice: each dump is already carried back
# to the ORIGINAL points of each room, so fusion is an element-wise weighted mean.
#
#   python fuse_votes.py --probs bridge/logs/A/probs bridge/logs/B/probs \
#       --processed_data_path bridge/processed --test_area test --out fused.json
#
# Weights default to equal. Keep it that way for a reported number unless they were fixed
# before the test set was scored -- tuning them on the test set is test-set training.

import os as _os, sys as _sys
# runnable from anywhere: the modules it imports live at the repository root
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os, json, glob, argparse
import numpy as np
import torch

from data_processing import load_model_config
from evaluate_full import metrics_from_confusion


def main():
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument('--probs', nargs='+', required=True, help='Directories written by --save_probs')
	ap.add_argument('--weights', nargs='+', type=float, default=None,
					help='One weight per --probs dir (default: equal)')
	ap.add_argument('--processed_data_path', required=True)
	ap.add_argument('--test_area', default='test')
	ap.add_argument('--config', default='model_params.json', help='Dataset config (class names)')
	ap.add_argument('--out', required=True)
	ap.add_argument('--overwrite', action='store_true')
	args = ap.parse_args()

	if os.path.exists(args.out) and not args.overwrite:
		raise SystemExit(f'{args.out} exists; pass --overwrite to replace it')
	w = args.weights or [1.0] * len(args.probs)
	if len(w) != len(args.probs):
		raise SystemExit('--weights must have one entry per --probs dir')
	w = np.asarray(w, dtype=np.float64) / np.sum(w)

	config = load_model_config(args.config)
	class_names = config['class_names']
	C = int(config['num_classes'])

	rooms = sorted(glob.glob(os.path.join(args.processed_data_path, args.test_area, '*.pt')))
	if not rooms:
		raise SystemExit(f'no rooms under {args.processed_data_path}/{args.test_area}')
	conf = np.zeros((C, C), dtype=np.int64)
	for room_pt in rooms:
		stem = os.path.basename(room_pt)[:-3]
		labels = torch.load(room_pt, weights_only=False).y.numpy()
		fused = None
		for d, wi in zip(args.probs, w):
			p = np.load(os.path.join(d, stem + '.npy')).astype(np.float32)
			if p.shape != (len(labels), C):
				raise SystemExit(f'{d}/{stem}.npy has shape {p.shape}, expected {(len(labels), C)}')
			fused = p * wi if fused is None else fused + p * wi
		pred = fused.argmax(axis=1)
		valid = (labels >= 0) & (labels < C)
		conf += np.bincount(labels[valid] * C + pred[valid], minlength=C * C).reshape(C, C)
		print(f'{stem}: {int(valid.sum()):,} points')

	m = metrics_from_confusion(conf)
	print(f"\n=== {args.test_area} FUSED RESULTS ({int(conf.sum()):,} points, "
		  f"{len(args.probs)} models, weights {np.round(w, 3).tolist()}) ===")
	print(f"OA {m['accuracy']*100:.2f} | mAcc {m['mAcc']*100:.2f} | mIoU {m['mIoU']*100:.2f}")
	print(f"{'class':10s} {'acc':>7s} {'iou':>7s} {'gt_points':>12s}")
	for i, name in enumerate(class_names):
		print(f"{name:10s} {m['per_class_acc'][i]*100:7.2f} {m['per_class_iou'][i]*100:7.2f} "
			  f"{int(m['gt'][i]):>12,}")

	result = {
		'protocol': 'pointwise_late_fusion',
		'test_area': args.test_area,
		'members': [{'probs': d, 'weight': float(wi)} for d, wi in zip(args.probs, w)],
		'overall_metrics': {'accuracy': m['accuracy'], 'mAcc': m['mAcc'], 'mIoU': m['mIoU'],
							'total_points': int(conf.sum())},
		'per_class_results': {
			name: {'accuracy': float(m['per_class_acc'][i]), 'iou': float(m['per_class_iou'][i]),
				   'gt_points': int(m['gt'][i])}
			for i, name in enumerate(class_names)},
		'confusion_matrix': {'row_is_ground_truth': True, 'class_order': class_names,
							 'counts': conf.tolist()},
	}
	with open(args.out, 'w') as f:
		json.dump(result, f, indent=2)
	print(f'\nSaved: {args.out}')


if __name__ == '__main__':
	main()
