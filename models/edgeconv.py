# Title: PointEdgeSegNet (version 0.4)
# Author: taewook kang (laputa9999@gmail.com)
# Date: 2025-09-21
# Purpose: Further optimized PointEdgeSegNet model architecture for memory efficiency.
# Dependencies: torch, torch_geometric

import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.utils import scatter
from torch.cuda.amp import autocast, GradScaler

# --- kNN / FPS backend ----------------------------------------------------------------
# PyG >= 2.8 routes fps / knn / knn_graph through pyg-lib ONLY: without 'pyg-lib>=0.6.0'
# they raise ImportError at the first EdgeConv ("'knn_graph' requires 'pyg-lib>=0.6.0'").
# torch-cluster ships the same CUDA kernels with identical call semantics and index
# conventions, so we fall back to it when pyg-lib is unavailable -- numerics are unchanged
# either way, and no checkpoint is affected.
try:
	import torch_geometric.typing as _pyg_typing
	_WITH_PYG_OPS = bool(getattr(_pyg_typing, 'WITH_KNN', True) and getattr(_pyg_typing, 'WITH_FPS', True))
except Exception:  # torch_geometric.typing unavailable (e.g. stubbed in tests)
	_WITH_PYG_OPS = True

if _WITH_PYG_OPS:
	from torch_geometric.nn import fps, knn_interpolate
	from torch_geometric.nn.pool import knn_graph
else:
	from torch_cluster import fps, knn, knn_graph

	def knn_interpolate(x, pos_x, pos_y, batch_x=None, batch_y=None, k=3, num_workers=1):
		"""torch-cluster backed copy of torch_geometric.nn.unpool.knn_interpolate."""
		with torch.no_grad():
			assign_index = knn(pos_x, pos_y, k, batch_x=batch_x, batch_y=batch_y,
							   num_workers=num_workers)
			y_idx, x_idx = assign_index[0], assign_index[1]
			diff = pos_x[x_idx] - pos_y[y_idx]
			squared_distance = (diff * diff).sum(dim=-1, keepdim=True)
			weights = 1.0 / torch.clamp(squared_distance, min=1e-16)

		y = scatter(x[x_idx] * weights, y_idx, 0, pos_y.size(0), reduce='sum')
		# Guard the division: a target point that received no neighbour would otherwise
		# yield 0/0 = NaN and poison the whole batch. That can only happen if the kNN query
		# under-covers (e.g. an unsorted batch vector), but silently returning zeros for a
		# handful of points is far better than a NaN loss, and the assert below surfaces it.
		denom = scatter(weights, y_idx, 0, pos_y.size(0), reduce='sum')
		return y / denom.clamp(min=1e-16)

def grid_subsample(pos, batch, ratio):
	"""Grid (voxel) subsampling with the same signature/contract as `fps(pos, batch, ratio)`.

	Returns indices of a spatially-uniform subset keeping ~`ratio` of the points: one
	representative (the lowest original index) per occupied voxel, with the voxel size
	solved from the current point density so the kept fraction matches `ratio`.

	Why this exists: FPS is quadratic and dominates cost as soon as the input grows beyond
	a small block. Measured on this box (torch_cluster fps, ratio 0.35, vs this function):

	     N        FPS        grid     speedup
	     8,192     93 ms     14 ms       6.7x
	    30,000    597 ms    1.5 ms       405x
	   130,000  11,559 ms    7.0 ms     1,663x
	   320,000  81,583 ms   20.0 ms     4,083x

	This matches PTv2's published table (Appendix B.3: 320K points, FPS-kNN 7207 ms vs
	grid pooling 1.33 ms) and is the stated reason every recent method dropped FPS:
	  DeLA 3.3.1: "FPS possesses a quadratic complexity while grid subsampling performs in
	  linear time."
	  PTv2 1: sampling+neighbour-query pooling is "time-consuming and not spatially
	  well-aligned."
	Grid sampling is also spatially uniform, whereas FPS density follows input density.
	"""
	n = pos.size(0)
	k = max(int(round(n * float(ratio))), 1)
	if k >= n:
		return torch.arange(n, device=pos.device)

	# Voxel edge length: per-point volume scaled so that ~k voxels are occupied. Points sit
	# on 2D surfaces in 3D, so the exponent is fitted by a short search rather than assumed.
	span = (pos.max(dim=0).values - pos.min(dim=0).values).clamp(min=1e-6)
	# NOTE: use an explicit product, not span.prod() -- the reduction lowers to an nvrtc
	# JIT kernel that fails on this GB10/sm_121 build ("invalid value for --gpu-architecture").
	vol = float(span[0]) * float(span[1]) * float(span[2])
	base = (vol / max(k, 1)) ** (1.0 / 3.0)
	lo, hi = base * 0.15, base * 4.0
	idx = None
	for _ in range(12):  # bisect on voxel size until the kept count is within 5% of k
		g = (lo * hi) ** 0.5
		coords = torch.floor(pos / g).long()
		key = (coords[:, 0] * 73856093) ^ (coords[:, 1] * 19349663) ^ (coords[:, 2] * 83492791)
		key = key * (int(batch.max().item()) + 1) + batch.long()
		_, inv = torch.unique(key, return_inverse=True)
		cnt = int(inv.max().item()) + 1
		if abs(cnt - k) <= max(1, k // 20):
			idx = (inv, cnt)
			break
		if cnt > k:      # too many voxels kept -> coarser grid
			lo = g
		else:
			hi = g
		idx = (inv, cnt)
	inv, cnt = idx
	# One representative per voxel: the smallest original index (deterministic).
	out = torch.full((cnt,), n, dtype=torch.long, device=pos.device)
	out.scatter_reduce_(0, inv, torch.arange(n, device=pos.device), reduce='amin')
	# MUST be sorted. scatter_reduce_ emits indices in voxel-hash order, so pos[out] would
	# carry an UNSORTED batch vector -- and torch_cluster's knn/knn_graph silently return
	# incomplete results for an unsorted batch (measured: one sample of a 4-room batch got
	# zero neighbours for all 15,430 of its points, so knn_interpolate divided by zero and
	# the whole forward pass went NaN; ~75% of training batches were being discarded).
	# The input batch vector is sorted, so sorting the indices restores it.
	return torch.sort(out).values

from .common import FeatureGate, AttentionModule, LightweightTransformer

class EdgeConv(nn.Module):
	def __init__(self, in_channels, out_channels, residual=False, k=20):
		super(EdgeConv, self).__init__()
		self.residual = residual and (in_channels == out_channels)
		self.k = k  # kNN neighbourhood size for the edge graph (was hard-coded 20)

		self.mlp = nn.Sequential(
			nn.Linear(2 * in_channels, out_channels),
			nn.BatchNorm1d(out_channels),
			nn.ReLU(),
			nn.Linear(out_channels, out_channels),
			nn.BatchNorm1d(out_channels)
		)
		self.attention = AttentionModule(out_channels)
		self.final_activation = nn.ReLU()

	def forward(self, x, pos, batch):
		edge_index = knn_graph(pos, k=self.k, batch=batch, loop=False)
		row, col = edge_index
		
		edge_features = torch.cat([x[row], x[col] - x[row]], dim=1)
		out = self.mlp(edge_features)
		
		aggr_out = scatter(out, row, dim=0, dim_size=x.size(0), reduce='max')
		aggr_out = self.attention(aggr_out)
		
		if self.residual:
			aggr_out = aggr_out + x
			
		return self.final_activation(aggr_out)

class PointEdgeSegNet(nn.Module):
	def __init__(self, num_features, num_classes, feature_dims=(4, 3, 3), knn=32, transformer_layers=2,
				 context_mode='input', width_mult=1.0, mid_transformer=False, sampler='fps',
				 enc_channels=None, bottleneck_dim=None):
		super(PointEdgeSegNet, self).__init__()
		# knn: EdgeConv neighbourhood size (was 20; 32 gives a richer local graph).
		# transformer_layers: bottleneck attention depth (was 1). All constructor args are
		# architecture-defining, so train and inference must use the SAME values -> they
		# default here to the historical architecture and every existing checkpoint loads
		# unchanged with the defaults. Changing any of them requires retraining.
		#
		# context_mode ('input' | 'bottleneck'): where the block-context descriptor
		#   (context_dim > 0) enters the network.
		#   'input'      - legacy: appended as per-point constant channels through the
		#                  FeatureGate/EdgeConv path. Measured to FAIL on S3DIS
		#                  (logs/20260722_104628, mIoU -6.4 vs baseline): the gate learned to
		#                  weight the constant group highest (0.727) although a within-block
		#                  constant cannot separate points of the same block, acting as an
		#                  over-trusted block prior and dragging optimization (2x train loss).
		#   'bottleneck' - the descriptor is kept OUT of the per-point feature path and
		#                  injected once per block at the bottleneck as a ZERO-INITIALIZED
		#                  residual (LayerNorm -> MLP -> +x4). At init the network is exactly
		#                  the no-context baseline, so the worst case is "context ignored" --
		#                  it can only be used where it reduces the loss (block-level prior,
		#                  which is all a per-block constant can express).
		# width_mult: channel multiplier for encoder/decoder (1.0 = historical 64/128/256/512).
		# mid_transformer: adds a 1-layer attention block at the mid level (~860 pts/block,
		#   c3 dim) so long-range within-block context is modeled before the bottleneck.
		# sampler ('fps' | 'grid'): how each encoder stage picks its coarser point set.
		#   'fps'  - farthest point sampling (historical; quadratic, dominates cost above
		#            ~30k points, see grid_subsample docstring for measurements).
		#   'grid' - voxel subsampling: linear, spatially uniform, and what every recent
		#            method uses. Only matters for LARGE inputs -- at 8192 points/block FPS
		#            is just 4.4% of the forward pass, so this is the enabler for
		#            whole-room training, not a speedup in the block-based setup.
		#   Sampling changes which points survive, so a checkpoint must be evaluated with
		#   the same sampler it was trained with.
		assert context_mode in ('input', 'bottleneck')
		assert sampler in ('fps', 'grid')
		self.sampler = sampler

		# Feature layout is configurable (geo, rgb, spatial[, block-context]). A 3-tuple is
		# accepted for backward compatibility (context_dim=0); (4,3,3)=10D reproduces the
		# original S3DIS model exactly, so existing weights load unchanged. Other domains
		# (colorless clouds, no-spatial, block-context, ...) pass dims from model_params.json.
		dims = tuple(feature_dims) + (0,) * (4 - len(feature_dims))
		geo_dim, rgb_dim, spatial_dim, context_dim = dims
		assert sum(dims) == num_features, (
			f"feature_dims {feature_dims} sum ({sum(dims)}) != num_features ({num_features})")
		self.geo_dim = geo_dim
		self.rgb_dim = rgb_dim
		self.spatial_dim = spatial_dim
		self.context_dim = context_dim
		self.spatial_start = geo_dim + rgb_dim
		self.context_mode = context_mode if context_dim > 0 else 'input'
		# Per-point input width: in bottleneck mode the context columns are split off before
		# the gate/EdgeConv path, so the point pathway is the plain base layout.
		ctx_in_input = context_dim if self.context_mode == 'input' else 0
		self.point_in_dim = geo_dim + rgb_dim + spatial_dim + ctx_in_input

		# Channel plan. width_mult scales the historical (64,128,256,512) uniformly -- which
		# was measured NOT to help (5.7M -> 14.1M params, mIoU 53.70 vs 54.35), because it
		# also inflates a bottleneck that is already oversized. enc_channels overrides the
		# plan outright so capacity can be moved to where the decisions actually happen.
		#
		# Measured motivation: 74.6% of the parameters sit in the bottleneck transformer,
		# which sees 215 points (2.6% of the input), while conv1/conv1_2 -- the only stage
		# that sees all 8,192 points, and therefore the only place a window can be told from
		# the wall around it -- holds 0.4%. The per-class numbers agree: window recall 51.6%
		# at 91.3% precision, column 33.3% at 61.0% -- the model is UNDER-predicting the
		# fine-boundary classes, i.e. it cannot resolve them, not that it is being cautious.
		if enc_channels is not None:
			c1, c2, c3, c4 = [int(c) for c in enc_channels]
		else:
			c1, c2, c3, c4 = (int(64 * width_mult), int(128 * width_mult),
							  int(256 * width_mult), int(512 * width_mult))
		bdim = int(bottleneck_dim) if bottleneck_dim else c4

		# Add Feature Gate at input (minimal memory: ~2MB for 20 batches)
		self.feature_gate = FeatureGate(geo_dim=geo_dim, rgb_dim=rgb_dim, spatial_dim=spatial_dim,
										context_dim=ctx_in_input)

		# Optimized Encoder - reduced channels and removed deepest layer
		self.conv1 = EdgeConv(self.point_in_dim, c1, k=knn)
		self.conv1_2 = EdgeConv(c1, c1, residual=True, k=knn)
		self.conv2 = EdgeConv(c1, c2, k=knn)
		self.conv2_2 = EdgeConv(c2, c2, residual=True, k=knn)
		self.conv3 = EdgeConv(c2, c3, k=knn)
		self.conv3_2 = EdgeConv(c3, c3, residual=True, k=knn)
		self.conv4 = EdgeConv(c3, c4, k=knn)
		# Optional mid-level attention (~860 pts at c3): long-range context where it is cheap
		self.mid_transformer = (LightweightTransformer(dim=c3, num_heads=4, dropout=0.1,
													   spatial_dim=spatial_dim, num_layers=1,
													   zero_init_residual=True)
								if mid_transformer else None)
		# Bottleneck Transformer with (now centered) Position Encoding (spatial-dim aware)
		# The transformer may be narrower than the encoder's last stage; project if so.
		self.bottleneck_proj = nn.Linear(c4, bdim) if bdim != c4 else nn.Identity()
		self.bottleneck_unproj = nn.Linear(bdim, c4) if bdim != c4 else nn.Identity()
		self.bottleneck_transformer = LightweightTransformer(dim=bdim, num_heads=max(1, bdim // 128),
															  dropout=0.1, spatial_dim=spatial_dim,
															  num_layers=transformer_layers)

		# Block-context bottleneck injection (see context_mode docstring above)
		if self.context_mode == 'bottleneck':
			self.ctx_norm = nn.LayerNorm(context_dim)
			self.ctx_mlp = nn.Sequential(
				nn.Linear(context_dim, 128),
				nn.ReLU(),
				nn.Linear(128, c4),
			)
			# Zero-init the projection: at initialization the context contributes exactly
			# nothing, so training starts from the proven baseline behaviour.
			nn.init.zeros_(self.ctx_mlp[-1].weight)
			nn.init.zeros_(self.ctx_mlp[-1].bias)

		# Simplified Decoder
		self.deconv1_mlp = nn.Sequential(
			nn.Linear(c4 + c3, c3),  # Updated to match bottleneck channels
			nn.BatchNorm1d(c3),
			nn.ReLU()
		)
		self.deconv2_mlp = nn.Sequential(
			nn.Linear(c3 + c2, c2),
			nn.BatchNorm1d(c2),
			nn.ReLU()
		)
		self.deconv3_mlp = nn.Sequential(
			nn.Linear(c2 + c1, c1),
			nn.BatchNorm1d(c1),
			nn.ReLU()
		)

		# Optimized prediction head
		h1, h2 = int(96 * width_mult), int(48 * width_mult)
		self.head_attention = AttentionModule(c1 + self.point_in_dim)
		self.head = nn.Sequential(
			nn.Linear(c1 + self.point_in_dim, h1),  # Reduced from 128
			nn.BatchNorm1d(h1),
			nn.ReLU(),
			nn.Dropout(0.5),
			nn.Linear(h1, h2),  # Reduced from 64
			nn.BatchNorm1d(h2),
			nn.ReLU(),
			nn.Dropout(0.4),
			nn.Linear(h2, num_classes)
		)

	def forward(self, data):
		x, pos, batch = data.x, data.pos, data.batch

		# Per-sample coordinate centering (translation invariance). knn_graph / FPS /
		# knn_interpolate are already translation-invariant, but the bottleneck transformer's
		# position encoding consumed ABSOLUTE world coords, so it overfit to each training
		# area's coordinate range and generalized worse to unseen areas (part of the val->test
		# gap). Centering every block to its own centroid feeds small local coords in BOTH
		# training and inference. Output is per-point (order preserved), so the caller keeps its
		# original coordinates unchanged -- no un-normalization needed downstream.
		centroid = scatter(pos, batch, dim=0, reduce='mean')  # [num_samples, 3]
		pos = pos - centroid[batch]

		# Stage sampler: fps (historical) or grid_subsample (linear, needed at scale)
		sample = grid_subsample if self.sampler == 'grid' else fps

		# Bottleneck context mode: split the (per-block constant) context columns off BEFORE
		# the gate so the per-point pathway never sees them (see __init__ docstring).
		ctx_block = None
		if self.context_mode == 'bottleneck' and self.context_dim > 0:
			ctx = x[:, self.point_in_dim:self.point_in_dim + self.context_dim]
			ctx_block = scatter(ctx, batch, dim=0, reduce='mean')  # [num_samples, context_dim]
			x = x[:, :self.point_in_dim]

		# Apply feature gating at input
		x_gated, gates = self.feature_gate(x)
		x0, pos0, batch0 = x_gated, pos, batch

		# Store gates for analysis (keep on GPU)
		self.last_gates = gates.detach()

		# Optimized encoder with fewer layers
		x1 = self.conv1(x0, pos0, batch0)
		x1 = self.conv1_2(x1, pos0, batch0)
		idx1 = sample(pos0, batch0, ratio=0.35)
		pos1, x1_sampled, batch1 = pos0[idx1], x1[idx1], batch0[idx1]

		x2 = self.conv2(x1_sampled, pos1, batch1)
		x2 = self.conv2_2(x2, pos1, batch1)
		idx2 = sample(pos1, batch1, ratio=0.3)
		pos2, x2_sampled, batch2 = pos1[idx2], x2[idx2], batch1[idx2]

		x3 = self.conv3(x2_sampled, pos2, batch2)
		x3 = self.conv3_2(x3, pos2, batch2)

		# Extract spatial features from original gated features (configured slice)
		if self.spatial_dim > 0:
			spatial0 = x_gated[:, self.spatial_start:self.spatial_start + self.spatial_dim]
			spatial2 = spatial0[idx1][idx2]  # follows the same FPS indices
		else:
			spatial0 = spatial2 = None  # transformer falls back to XYZ-only position encoding

		# Optional mid-level attention: long-range context over the ~860-pt level, where
		# quadratic attention is still cheap, before the final downsample.
		if self.mid_transformer is not None:
			x3 = self.mid_transformer(x3, pos2, spatial2, batch2)

		idx3 = sample(pos2, batch2, ratio=0.25)
		pos3, x3_sampled, batch3 = pos2[idx3], x3[idx3], batch2[idx3]
		spatial3 = spatial2[idx3] if self.spatial_dim > 0 else None

		# Final encoder layer (bottleneck)
		x4_bottleneck = self.conv4(x3_sampled, pos3, batch3)
		# Apply Transformer with XYZ (and Spatial, if present) position encoding
		x4_bottleneck = self.bottleneck_unproj(
			self.bottleneck_transformer(self.bottleneck_proj(x4_bottleneck), pos3, spatial3, batch3))

		# Bottleneck block-context injection (zero-init residual; exact baseline at init)
		if ctx_block is not None:
			x4_bottleneck = x4_bottleneck + self.ctx_mlp(self.ctx_norm(ctx_block))[batch3]

		# Simplified decoder
		up_x2 = knn_interpolate(x4_bottleneck, pos3, pos2, batch3, batch2, k=3)
		dec_x2 = self.deconv1_mlp(torch.cat([up_x2, x3], dim=1))

		up_x1 = knn_interpolate(dec_x2, pos2, pos1, batch2, batch1, k=3)
		dec_x1 = self.deconv2_mlp(torch.cat([up_x1, x2], dim=1))
		
		up_x0 = knn_interpolate(dec_x1, pos1, pos0, batch1, batch0, k=3)
		dec_x0 = self.deconv3_mlp(torch.cat([up_x0, x1], dim=1))
		
		# Final prediction with attention
		final_x = torch.cat([dec_x0, x0], dim=1)
		final_x = self.head_attention(final_x)
		out = self.head(final_x)
		
		return out
