# evaluate_full.py
# Standard-protocol S3DIS evaluation: score EVERY point of the held-out area.
#
# Why this exists: train_model.py's test() evaluates only the cached test blocks, i.e.
# the ~7-9% subsample that partition_columns() kept per column, with no voting. Published
# S3DIS Area 5 numbers (PTv3, KPConvX, DeLA, SPT, ...) score all points of the area, so
# the two are not comparable. This script aligns the protocol:
#   - partition_columns_cover(): every point lands in >= 1 block (dense columns are tiled,
#     nothing subsampled away),
#   - overlapping windows (stride < window) give multiple predictions per point,
#   - per-point softmax votes are accumulated and argmax'd -> one label per point,
#   - metrics over ALL labelled points: OA, mAcc, mIoU, per-class accuracy AND IoU
#     (papers publish per-class IoU only), plus the full 13x13 confusion matrix.
#
# The feature pipeline mirrors training block build exactly: room .pt base features with
# the curvature channel refreshed, and (optionally) the block-context descriptor appended
# per block with --block_context for context-trained (22D) checkpoints.
#
# Usage:
#   python evaluate_full.py --model_weights logs/<run>/best_model.pth
#   python evaluate_full.py --model_weights ... --stride 1.0        # 2x-overlap voting
#   python evaluate_full.py --model_weights ... --block_context     # 22D context model
#
# Output: <model_dir>/test_full_summary.json (+ console table). Existing
# test_summary.json files are left untouched for comparison.

import os, json, argparse, time
import numpy as np
import torch
from glob import glob
from tqdm import tqdm

from torch_geometric.data import Data, Batch
from model import PointEdgeSegNet
from data_processing import (
	load_model_config,
	resolve_feature_config,
	partition_columns_cover,
	compute_surface_variation,
	make_block_context_extractor,
	append_block_context,
)

def refresh_curvature(features, points, spec):
	"""Mirror train_model.refresh_curvature_inplace: recompute the stored curvature
	channel (stale in some processed_s3dis versions) so eval features == training features."""
	normals_count = 3 if spec['use_normals'] else 0
	if spec['geo_dim'] <= normals_count:  # geo group is normals-only -> no curvature slot
		return features
	features[:, normals_count] = compute_surface_variation(points, knn=spec['neighbor_knn'])
	return features

def tta_views(n_scale=5, flip=True):
	"""Standard point-cloud TTA view list: scale x mirror, as used by every SOTA recipe.

	Pointcept (PTv3/PTv2/Sonata) tests 10 views = scale {0.9,0.95,1.0,1.05,1.1} x {no flip,
	flip}, accumulating softmax per point; KPConvX votes 10x, Sonata 13x, DeLA 12x. The
	transform must be one the model is invariant to by training augmentation -- scaling and
	mirroring are, so no un-transform of the prediction is needed (predictions are per point
	and the point ORDER is preserved, only coordinates change).

	Returns a list of (scale, flip_x) tuples; the first is always the identity view.
	"""
	scales = [1.0] if n_scale <= 1 else list(np.linspace(0.9, 1.1, n_scale))
	views = []
	for f in ([False, True] if flip else [False]):
		for s in scales:
			views.append((float(s), f))
	views.sort(key=lambda v: (abs(v[0] - 1.0) + (1.0 if v[1] else 0.0)))  # identity first
	return views

def evaluate_room(model, room_pt, spec, num_classes, device, block_size, window, stride,
				  batch_size, use_amp=False, views=((1.0, False),)):
	"""Return (13x13 confusion-matrix counts, points scored, blocks used) for one room.

	`views` is a list of (scale, flip_x) TTA transforms; per-point softmax is summed over
	all of them (and over overlapping blocks when stride < window) before the argmax.
	"""
	d = torch.load(room_pt, weights_only=False)
	points = d.pos.numpy().astype(np.float32)
	features = d.x.numpy().astype(np.float32).copy()
	labels = d.y.numpy()
	n = len(points)

	features = refresh_curvature(features, points, spec)
	ctx = make_block_context_extractor(points, features, spec)

	blocks = partition_columns_cover(points, block_size=block_size,
									 window=window, stride=stride, seed=0)

	votes = np.zeros((n, num_classes), dtype=np.float32)
	for scale, flip_x in views:
		# Apply the view transform to coordinates only. Blocks were computed on the
		# original coordinates, so point membership (and therefore coverage) is identical
		# across views -- only the geometry the network sees changes.
		vpoints = points * np.float32(scale)
		if flip_x:
			vpoints = vpoints.copy()
			vpoints[:, 0] = -vpoints[:, 0]
		vfeatures = features
		if flip_x and spec['use_normals']:
			vfeatures = features.copy()
			vfeatures[:, 0] = -vfeatures[:, 0]   # mirror the normal x-component too
		for start in range(0, len(blocks), batch_size):
			chunk = blocks[start:start + batch_size]
			data_list = []
			for idx, num_real in chunk:
				feats = vfeatures[idx]
				if ctx is not None:
					feats = append_block_context(feats, ctx, idx[:num_real])
				data_list.append(Data(x=torch.from_numpy(np.ascontiguousarray(feats)),
									  pos=torch.from_numpy(np.ascontiguousarray(vpoints[idx]))))
			batch = Batch.from_data_list(data_list).to(device)
			with torch.no_grad():
				out = model(batch)
				probs = torch.softmax(out.float(), dim=-1).cpu().numpy()
			for j, (idx, num_real) in enumerate(chunk):
				p = probs[j * block_size:(j + 1) * block_size][:num_real]
				np.add.at(votes, idx[:num_real], p)  # scatter-add: padded rows excluded

	assert (votes.sum(axis=1) > 0).all(), f"uncovered points in {room_pt}"
	pred = votes.argmax(axis=1)
	valid = (labels >= 0) & (labels < num_classes)
	conf = np.bincount(labels[valid] * num_classes + pred[valid],
					   minlength=num_classes * num_classes).reshape(num_classes, num_classes)
	return conf, int(valid.sum()), len(blocks)

def metrics_from_confusion(conf):
	tp = np.diag(conf).astype(np.float64)
	gt = conf.sum(axis=1).astype(np.float64)    # per-class ground-truth count
	pr = conf.sum(axis=0).astype(np.float64)    # per-class predicted count
	union = gt + pr - tp
	acc = np.divide(tp, gt, out=np.zeros_like(tp), where=gt > 0)
	iou = np.divide(tp, union, out=np.zeros_like(tp), where=union > 0)
	present = gt > 0  # standard S3DIS: average over classes that exist in the GT
	return {
		'accuracy': tp.sum() / max(gt.sum(), 1),
		'mAcc': acc[present].mean(),
		'mIoU': iou[present].mean(),
		'per_class_acc': acc, 'per_class_iou': iou,
		'gt': gt, 'pred': pr, 'tp': tp,
	}

def main():
	ap = argparse.ArgumentParser(description='Full-coverage (standard-protocol) S3DIS evaluation')
	ap.add_argument('--config', default='model_params.json')
	ap.add_argument('--model_weights', required=True)
	ap.add_argument('--processed_data_path', default='./processed_s3dis')
	ap.add_argument('--test_area', default='Area_5')
	ap.add_argument('--block_size', type=int, default=8192,
					help='MUST match the block_size the checkpoint was trained with')
	ap.add_argument('--window', type=float, default=2.0,
					help='MUST match the training column window')
	ap.add_argument('--stride', type=float, default=2.0,
					help='< window enables multi-view voting (e.g. 1.0 = 2x overlap, slower)')
	ap.add_argument('--batch_size', type=int, default=18)
	ap.add_argument('--block_context', action='store_true',
					help='Append the wide-area context descriptor (22D context-trained models)')
	ap.add_argument('--context_mode', type=str, default='bottleneck', choices=['input', 'bottleneck'],
					help="MUST match training: 'bottleneck' (new default) or 'input' (legacy, e.g. logs/20260722_104628)")
	ap.add_argument('--width_mult', type=float, default=1.0, help='MUST match training --width_mult')
	ap.add_argument('--mid_transformer', action='store_true', help='MUST match training --mid_transformer')
	ap.add_argument('--enc_channels', type=str, default=None, help='MUST match training --enc_channels')
	ap.add_argument('--bottleneck_dim', type=int, default=None, help='MUST match training --bottleneck_dim')
	ap.add_argument('--sampler', type=str, default='fps', choices=['fps', 'grid'],
					help='MUST match training --sampler (sampling decides which points survive each stage)')
	ap.add_argument('--mode', type=str, default='block', choices=['block', 'room', 'chunk'],
					help="'block' = 2 m column blocks; 'room' = voxelize and predict the whole room; "
						 "'chunk' = voxelize + KD-median chunks with halo, score cores only, then "
						 "propagate by nearest neighbour (must match a chunk-trained checkpoint).")
	ap.add_argument('--room_grid', type=float, default=0.04, help='room mode: voxel size, MUST match training')
	ap.add_argument('--room_max_points', type=int, default=200000,
					help='room mode: voxels per forward pass; larger rooms are split into overlapping chunks')
	ap.add_argument('--core_max', type=int, default=12288, help='chunk mode: KD-split target, MUST match training')
	ap.add_argument('--halo', type=float, default=1.0, help='chunk mode: halo width in m, MUST match training')
	ap.add_argument('--invariant_geo', action='store_true',
					help='chunk mode: recompute linearity/planarity/verticality into the last 3 feature '
						 'columns. MUST match how the training cache was built -- a mismatch silently feeds '
						 'different quantities in the same columns and collapses the score.')
	ap.add_argument('--tta', type=int, default=1, metavar='N_SCALE',
					help='Test-time augmentation: number of scale views in [0.9,1.1] (1 = off). '
						 'Combined with --tta_flip this gives N_SCALE (x2) views whose softmax is summed '
						 'per point, as in Pointcept (10 views) / DeLA (12 votes). Cost scales linearly.')
	ap.add_argument('--tta_flip', action='store_true', help='Add mirrored views to the TTA set (doubles views)')
	ap.add_argument('--tta_d4', type=int, default=1, choices=[1, 4, 8],
					help='CHUNK mode TTA: grid-preserving D4 views (1=off, 4=rot90s, 8=rot90s x flip). '
						 'Scale-TTA is wrong for stencil models (breaks the lattice); this is the safe family.')
	ap.add_argument('--out', default=None, help='Output JSON (default: <model_dir>/test_full_summary.json)')
	ap.add_argument('--arch', type=str, default='v1', choices=['v1', 'v2'],
					help="Architecture the checkpoint was trained with ('v2' = model_v2.py serialized meta)")
	ap.add_argument('--v2_knn', type=int, default=32, help='v2: window size, MUST match training')
	ap.add_argument('--v2_curves', type=int, default=1, help='v2: curves per stage, MUST match training')
	ap.add_argument('--v2_neighbors', type=str, default='serial', choices=['serial', 'stencil'],
					help='v2: neighbour source, MUST match training')
	ap.add_argument('--v2_stencil', type=int, default=1, help='v2: stencil radius, MUST match training')
	ap.add_argument('--v2_diff', action='store_true', help='v2: feature-diff term, MUST match training')
	ap.add_argument('--v2_base_grid', type=float, default=0.04, help='v2: input voxel size, MUST match training')
	ap.add_argument('--v2_pool_grids', type=str, default='0.08,0.16,0.32', help='v2: pool grids, MUST match training')
	ap.add_argument('--v2_directional', action='store_true', help='v2: anisotropic aggregation, MUST match training')
	ap.add_argument('--v2_stencil_z', type=int, default=0, help='v2: vertical stencil reach, MUST match training')
	args = ap.parse_args()

	config = load_model_config(args.config)
	if args.block_context:
		config.setdefault('features', {})['use_block_context'] = True
	spec = resolve_feature_config(config)
	num_classes = int(config['num_classes'])
	class_names = config['class_names']
	feature_dims = (spec['geo_dim'], spec['rgb_dim'], spec['spatial_dim'], spec['context_dim'])

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	enc = tuple(int(c) for c in args.enc_channels.split(',')) if args.enc_channels else None
	if args.arch == 'v2':
		from model_v2 import PointEdgeSegNetV2
		model = PointEdgeSegNetV2(num_features=spec['num_features'], num_classes=num_classes,
								  feature_dims=feature_dims, enc_channels=enc or (64, 192, 320, 448),
								  bottleneck_dim=args.bottleneck_dim,
								  knn=args.v2_knn, curves=args.v2_curves,
								  neighbor_mode=args.v2_neighbors, stencil_radius=args.v2_stencil,
								  feature_diff=args.v2_diff, base_grid=args.v2_base_grid,
								  pool_grids=tuple(float(g) for g in args.v2_pool_grids.split(',')),
								  directional=args.v2_directional,
								  stencil_z=args.v2_stencil_z or None).to(device)
	else:
		model = PointEdgeSegNet(num_features=spec['num_features'], num_classes=num_classes,
							feature_dims=feature_dims, context_mode=args.context_mode,
							width_mult=args.width_mult, mid_transformer=args.mid_transformer,
							sampler=args.sampler,
							enc_channels=enc,
							bottleneck_dim=args.bottleneck_dim).to(device)
	state = torch.load(args.model_weights, map_location=device, weights_only=False)
	if isinstance(state, dict) and 'model_state_dict' in state:
		state = state['model_state_dict']
	model.load_state_dict(state)
	model.eval()
	print(f"Model: {args.model_weights}  ({spec['num_features']}D, dims={feature_dims})")
	views = tta_views(n_scale=args.tta, flip=args.tta_flip)
	if args.mode == 'room':
		print(f"Protocol: ALL points of {args.test_area} | ROOM mode, grid={args.room_grid} m, "
			  f"chunk cap {args.room_max_points:,} voxels, TTA={len(views)} view(s). "
			  f"Voxel predictions are propagated to every original point by nearest neighbour.")
	else:
		print(f"Protocol: ALL points of {args.test_area}, window={args.window}, stride={args.stride}, "
			  f"block_size={args.block_size}, voting={'on (overlap)' if args.stride < args.window else 'coverage-only'}, "
			  f"TTA={len(views)} view(s)")

	rooms = sorted(glob(os.path.join(args.processed_data_path, args.test_area, '*.pt')))
	if not rooms:
		raise SystemExit(f"No rooms found under {args.processed_data_path}/{args.test_area}")

	conf = np.zeros((num_classes, num_classes), dtype=np.int64)
	total_blocks = 0
	t0 = time.time()
	if args.mode == 'chunk':
		from voxel_chunk import predict_room_chunks, d4_views
		chunk_views = d4_views(args.tta_d4)
		if len(chunk_views) > 1:
			print(f"Chunk-mode TTA: {len(chunk_views)} grid-preserving D4 views (rot90 x flip)")
		for room_pt in tqdm(rooms, desc=f'[Full eval {args.test_area} / chunk]'):
			c, npts, nch = predict_room_chunks(model, room_pt, device, num_classes,
											   grid=args.room_grid, core_max=args.core_max,
											   halo=args.halo, block_size=args.block_size,
											   feature_dim=spec['num_features'],
											   neighbor_knn=spec['neighbor_knn'],
											   invariant_geo=args.invariant_geo,
											   views=chunk_views)
			conf += c
			total_blocks += nch
	elif args.mode == 'room':
		from room_pipeline import predict_room_full
		for room_pt in tqdm(rooms, desc=f'[Full eval {args.test_area} / room]'):
			c, npts, nvox, nchunk = predict_room_full(
				model, room_pt, device, spec, num_classes, grid=args.room_grid,
				max_points=args.room_max_points, neighbor_knn=spec['neighbor_knn'],
				views=views, feature_dim=spec['num_features'])
			conf += c
			total_blocks += nchunk
	else:
		for room_pt in tqdm(rooms, desc=f'[Full eval {args.test_area}]'):
			c, npts, nblk = evaluate_room(model, room_pt, spec, num_classes, device,
										  args.block_size, args.window, args.stride, args.batch_size,
										  views=views)
			conf += c
			total_blocks += nblk
	elapsed = time.time() - t0

	m = metrics_from_confusion(conf)
	print(f"\n=== {args.test_area} FULL-COVERAGE RESULTS "
		  f"({int(conf.sum()):,} points, {total_blocks} blocks, {elapsed/60:.1f} min) ===")
	print(f"OA {m['accuracy']*100:.2f} | mAcc {m['mAcc']*100:.2f} | mIoU {m['mIoU']*100:.2f}")
	print(f"{'class':10s} {'acc':>7s} {'iou':>7s} {'gt_points':>12s}")
	for i, name in enumerate(class_names):
		print(f"{name:10s} {m['per_class_acc'][i]*100:7.2f} {m['per_class_iou'][i]*100:7.2f} "
			  f"{int(m['gt'][i]):>12,}")

	out_path = args.out or os.path.join(os.path.dirname(args.model_weights), 'test_full_summary.json')
	result = {
		'protocol': f'full_coverage_voting_{args.mode}',
		'test_area': args.test_area,
		'model_path': args.model_weights,
		'eval_config': {'mode': args.mode, 'room_grid': args.room_grid,
						'block_size': args.block_size, 'window': args.window, 'stride': args.stride,
						'block_context': bool(args.block_context), 'num_blocks': total_blocks,
						'num_rooms': len(rooms), 'tta_views': len(views),
						'tta_view_list': [list(v) for v in views]},
		'overall_metrics': {'accuracy': m['accuracy'], 'mAcc': m['mAcc'], 'mIoU': m['mIoU'],
							'total_points': int(conf.sum())},
		'per_class_results': {
			name: {'accuracy': float(m['per_class_acc'][i]), 'iou': float(m['per_class_iou'][i]),
				   'gt_points': int(m['gt'][i]), 'predicted_points': int(m['pred'][i]),
				   'correct': int(m['tp'][i])}
			for i, name in enumerate(class_names)
		},
		'confusion_matrix': {'row_is_ground_truth': True, 'class_order': class_names,
							 'counts': conf.tolist()},
	}
	with open(out_path, 'w') as f:
		json.dump(result, f, indent=2)
	print(f"\nSaved: {out_path}")

if __name__ == '__main__':
	main()
