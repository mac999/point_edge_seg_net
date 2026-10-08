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

import os, sys, json, argparse, time
import numpy as np
import torch
from glob import glob
from tqdm import tqdm

from torch_geometric.data import Data, Batch
from models.builder import (spec_from_args, describe_spec, load_ensemble_members,
                           add_ensemble_arguments)
from data_processing import (
	load_model_config,
	resolve_feature_config,
	partition_columns_cover,
	extract_global_position_features,
	compute_surface_variation,
	make_block_context_extractor,
	append_block_context,
	feature_dims_from_spec,
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
				  batch_size, use_amp=False, views=((1.0, False),), column_grid=0.0,
				  save_probs=None):
	"""Return (13x13 confusion-matrix counts, points scored, blocks used) for one room.

	`views` is a list of (scale, flip_x) TTA transforms; per-point softmax is summed over
	all of them (and over overlapping blocks when stride < window) before the argmax.

	`save_probs`: optional path; the per-point mean softmax (over views and overlapping
	blocks), carried back to every ORIGINAL point, is written there as float16 so models
	scored on different voxel grids can be fused point-for-point afterwards (tools/fuse_votes.py).
	"""
	d = torch.load(room_pt, weights_only=False)
	all_points = d.pos.numpy().astype(np.float32)
	features = d.x.numpy().astype(np.float32).copy()
	labels = d.y.numpy()

	# Score where the model was trained. A checkpoint trained on a voxelised cloud sees a
	# different point density than the raw room, and scoring it raw measures the wrong
	# thing -- so voxelise first and propagate the result back, exactly as chunk mode does.
	# Coverage is unaffected: every original point still receives a prediction.
	if column_grid and column_grid > 0:
		from voxel_chunk import voxelize_and_featurize
		base_dim = spec['num_features'] - spec.get('global_position_dim', 0)
		_, points, features = voxelize_and_featurize(all_points, features, column_grid,
													 neighbor_knn=spec['neighbor_knn'],
													 feature_dim=base_dim)
	else:
		points = all_points
		features = refresh_curvature(features, points, spec)
	if spec.get('global_position_dim'):
		# after voxelisation, matching how the training cache builds it
		features = np.concatenate([features, extract_global_position_features(points)], axis=1)
	n = len(points)
	ctx = make_block_context_extractor(points, features, spec)

	blocks = partition_columns_cover(points, block_size=block_size,
									 window=window, stride=stride, seed=0)

	votes = np.zeros((n, num_classes), dtype=np.float32)
	nvotes = np.zeros(n, dtype=np.int32)      # how many softmax rows each point received
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
				np.add.at(nvotes, idx[:num_real], 1)

	assert (votes.sum(axis=1) > 0).all(), f"uncovered points in {room_pt}"
	pred = votes.argmax(axis=1)
	back = None
	if len(points) != len(all_points):
		# Voxels were scored; carry each prediction to the original points nearest it, so
		# the confusion matrix still covers every labelled point in the room.
		from scipy.spatial import cKDTree
		back = cKDTree(points).query(all_points, k=1)[1]
		pred = pred[back]
	if save_probs:
		mean = votes / nvotes[:, None]
		if back is not None:
			mean = mean[back]
		np.save(save_probs, mean.astype(np.float16))
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
	ap.add_argument('--domain', default=None, metavar='NAME|PATH',
					help="Score a checkpoint on the geometry its domain trained it on "
						 "(domains/NAME.json): block size, window/stride, voxel lattice and "
						 "architecture are all read from the run's own recipe. Explicit flags "
						 "still win. Without this they must be retyped by hand, which is how "
						 "w6/w12 were once scored 3.9 mIoU low.")
	ap.add_argument('--protocol', default=None,
					choices=['single', 'overlap', 'mirror', 'overlap_mirror'],
					help="Named inference protocol, applied after --domain: 'single' = one view, "
						 "stride = window (coverage only); 'overlap' = stride window/2; "
						 "'mirror' = 2 views (identity + mirrored); 'overlap_mirror' = both. "
						 "Overrides --stride/--tta_flip; report which one was used.")
	ap.add_argument('--config', default='model_params.json')
	ap.add_argument('--model_weights', default=None,
					help='Checkpoint to score. Required unless --ensemble_config is given.')
	ap.add_argument('--processed_data_path', default='./processed_s3dis')
	ap.add_argument('--test_area', default='Area_5')
	ap.add_argument('--block_size', type=int, default=8192,
					help='MUST match the block_size the checkpoint was trained with')
	ap.add_argument('--column_grid', type=float, default=0.0, metavar='M',
					help='block mode: voxel size the checkpoint was TRAINED with (0 = full resolution).\n'
						 'Scoring a voxel-trained model on the raw cloud measures a density it never\n'
						 'saw; predictions are propagated back to every original point either way.')
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
	ap.add_argument('--core_max', type=int, default=12288,
					help='chunk mode: scored voxels per chunk. NOT required to match training: '
						 'block_size - core_max is the halo budget, and giving the network more '
						 'context than it saw in training measurably helps (12288 -> 8192 is '
						 '+0.22 mIoU on Area 5; 6144 with --halo 1.5 is +0.40, at 1.5-2x the '
						 'scoring time).')
	ap.add_argument('--halo', type=float, default=1.0,
					help='chunk mode: width (m) of the unscored context ring. 1.5 is the measured '
						 'optimum for this data, but only once core_max leaves budget to retain it '
						 '-- at the default core_max a wider ring is subsampled away and scores WORSE.')
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
	ap.add_argument('--overwrite', action='store_true',
					help='Allow replacing an existing --out file. Refused by default: a scored run is \n'
						 'the evidence for a comparison, and silently rewriting it loses the baseline \n'
						 'you were comparing against.')
	ap.add_argument('--save_probs', default=None, metavar='DIR',
					help='Write each room\'s per-point mean softmax (float16 .npy, original points) '
						 'under DIR for later point-wise fusion with tools/fuse_votes.py.')
	add_ensemble_arguments(ap)
	ap.add_argument('--arch', type=str, default='edgeconv',
					choices=['edgeconv', 'stencil', 'v1', 'v2'],
					help="Architecture the checkpoint was trained with ('v1'/'v2' aliases accepted)")
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
	domain_applied = None
	if args.domain:
		from domain_config import load_domain, apply_domain_eval, describe
		domain = load_domain(args.domain)
		domain_applied = apply_domain_eval(args, domain, sys.argv[1:], ap)
		print(describe(domain, domain_applied))
	if args.protocol:
		# Named protocols are resolved after the domain so they compose: the domain fixes the
		# window, the protocol decides how densely it is swept and how many views are voted.
		if '--stride' not in sys.argv:
			args.stride = args.window / 2 if args.protocol in ('overlap', 'overlap_mirror') else args.window
		if '--tta_flip' not in sys.argv:
			args.tta_flip = args.protocol in ('mirror', 'overlap_mirror')
		print(f"Protocol '{args.protocol}': stride {args.stride} (window {args.window}), "
			  f"mirror TTA {'on' if args.tta_flip else 'off'}")
	if not args.model_weights and not args.ensemble_config:
		ap.error('give --model_weights (a checkpoint) or --ensemble_config (an ensemble spec)')
	if args.ensemble and args.ensemble_config:
		ap.error('--ensemble and --ensemble_config are alternatives: --ensemble lists checkpoints '
				 'that share this command line\'s architecture, --ensemble_config lets each member '
				 'declare its own')

	# Resolve and guard the output BEFORE scoring: this run takes minutes, and discovering
	# at the end that it would clobber a previous result wastes all of it.
	default_near = args.model_weights or args.ensemble_config
	out_path = args.out or os.path.join(os.path.dirname(default_near), 'test_full_summary.json')
	if os.path.exists(out_path) and not args.overwrite:
		ap.error(f"{out_path} already exists. That file is the evidence for an earlier "
				 f"comparison; pass --overwrite to replace it, or --out with a new path.")

	config = load_model_config(args.config)
	if args.block_context:
		config.setdefault('features', {})['use_block_context'] = True
	spec = resolve_feature_config(config)
	num_classes = int(config['num_classes'])
	class_names = config['class_names']
	feature_dims = feature_dims_from_spec(spec)

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model_spec = spec_from_args(args)
	source = args.ensemble_config or args.model_weights
	print(f"Model(s): {source}  ({spec['num_features']}D, dims={feature_dims})")
	model, members = load_ensemble_members(model_spec, args.model_weights, args.ensemble,
										   args.ensemble_config, spec['num_features'],
										   num_classes, feature_dims, device)
	if members is None:
		print(f"  {describe_spec(model_spec)}")
	views = tta_views(n_scale=args.tta, flip=args.tta_flip)
	# Two independent TTA families live here: chunk mode votes over the grid-preserving D4
	# set (--tta_d4), every other mode over the scale x mirror set (--tta/--tta_flip).
	# Resolve which one is actually in play BEFORE the banner, so the printout and the result
	# JSON report the views that really ran instead of the unused default set.
	if args.mode == 'chunk':
		from voxel_chunk import d4_views
		active_views, tta_family = d4_views(args.tta_d4), 'd4_rot90_flip'
		if len(views) > 1:
			print(f"WARNING: --tta/--tta_flip ({len(views)} scale views) is ignored in chunk mode -- "
				  f"scale TTA breaks the voxel lattice. Use --tta_d4 instead (currently {args.tta_d4}).")
	else:
		active_views, tta_family = views, 'scale_flip'
	if args.mode == 'room':
		print(f"Protocol: ALL points of {args.test_area} | ROOM mode, grid={args.room_grid} m, "
			  f"chunk cap {args.room_max_points:,} voxels, TTA={len(active_views)} view(s). "
			  f"Voxel predictions are propagated to every original point by nearest neighbour.")
	else:
		print(f"Protocol: ALL points of {args.test_area}, window={args.window}, stride={args.stride}, "
			  f"block_size={args.block_size}, voting={'on (overlap)' if args.stride < args.window else 'coverage-only'}, "
			  f"TTA={len(active_views)} view(s) [{tta_family}]")

	rooms = sorted(glob(os.path.join(args.processed_data_path, args.test_area, '*.pt')))
	if not rooms:
		raise SystemExit(f"No rooms found under {args.processed_data_path}/{args.test_area}")

	conf = np.zeros((num_classes, num_classes), dtype=np.int64)
	total_blocks = 0
	t0 = time.time()
	if args.mode == 'chunk':
		from voxel_chunk import predict_room_chunks
		chunk_views = active_views
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
		if args.save_probs:
			os.makedirs(args.save_probs, exist_ok=True)
		for room_pt in tqdm(rooms, desc=f'[Full eval {args.test_area}]'):
			probs_path = (os.path.join(args.save_probs, os.path.basename(room_pt)[:-3] + '.npy')
						  if args.save_probs else None)
			c, npts, nblk = evaluate_room(model, room_pt, spec, num_classes, device,
										  args.block_size, args.window, args.stride, args.batch_size,
										  views=views, column_grid=args.column_grid,
										  save_probs=probs_path)
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

	result = {
		'protocol': f'full_coverage_voting_{args.mode}',
		'test_area': args.test_area,
		'model_path': args.model_weights if members is None else args.ensemble_config,
		'ensemble_members': members,
		'eval_config': {'mode': args.mode, 'room_grid': args.room_grid,
						'column_grid': args.column_grid,
						'block_size': args.block_size, 'window': args.window, 'stride': args.stride,
						'block_context': bool(args.block_context), 'num_blocks': total_blocks,
						'num_rooms': len(rooms), 'tta_views': len(active_views),
						'tta_family': tta_family, 'tta_d4': args.tta_d4,
						'tta_view_list': [list(v) for v in active_views],
						'domain': args.domain, 'protocol': args.protocol,
						'domain_applied': domain_applied},
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
