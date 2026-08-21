# models/stencil.py
# Voxel-stencil backbone of PointEdgeSegNet (the current default architecture).
#
# The input cloud is voxelized and every pooling stage stays on a voxel lattice, so
# spatial neighbours can be looked up at fixed lattice offsets (sorted integer keys +
# binary search) instead of searched. Local aggregation is a point-wise MLP with a
# relative-position encoding and a feature-difference term; down/upsampling use grid
# pooling with an exact inverse map. U-Net layout, feature gate and the bottleneck
# Transformer are shared with v1 (model.py), but checkpoints are not compatible.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter

from .common import FeatureGate, AttentionModule, LightweightTransformer


# --- serialization -------------------------------------------------------------------

def _part1by2(v):
	"""Spread the low 21 bits of v so consecutive bits land 3 apart (Morton interleave)."""
	v = v & 0x1FFFFF
	v = (v | (v << 32)) & 0x1F00000000FFFF
	v = (v | (v << 16)) & 0x1F0000FF0000FF
	v = (v | (v << 8))  & 0x100F00F00F00F00F
	v = (v | (v << 4))  & 0x10C30C30C30C30C3
	v = (v | (v << 2))  & 0x1249249249249249
	return v

_AXIS_PERMS = ((0, 1, 2), (1, 2, 0), (2, 0, 1))  # rotate the curve orientation per stage

def serialize(pos, batch, grid, perm=0):
	"""Sort points along a Morton curve; samples stay contiguous. `perm` rotates the
	curve axis order per stage so seams do not repeat across stages."""
	a, b, c = _AXIS_PERMS[perm % 3]
	g = ((pos - pos.min(dim=0).values) / grid).long()
	key = (_part1by2(g[:, a]) << 2) | (_part1by2(g[:, b]) << 1) | _part1by2(g[:, c])
	# batch id in the top bits keeps samples contiguous after one global sort
	key = (batch.long() << 44) | (key & ((1 << 44) - 1))
	return torch.argsort(key)

def stencil_neighbors(pos, batch, grid, radius=1, z_radius=None):
	"""Neighbour lookup on the voxel lattice: sorted keys + one batched searchsorted.

	Returns (col, mask) of shape (N, K) with K = (2r+1)^2 * (2rz+1); mask marks empty
	cells. Missing cells use the point's own index as sentinel so the gather backward
	spreads its atomic adds (a shared sentinel row serializes them)."""
	R = int(radius)
	# z_radius extends the stencil vertically (columns/pipes span many z-levels).
	Rz = int(z_radius) if z_radius else R
	dev = pos.device
	shift = torch.tensor([R, R, Rz], device=dev)
	g = ((pos - pos.min(dim=0).values) / grid).long() + shift  # keeps g+offset >= 0
	def key_of(q, bb):
		return (bb.long() << 44) | ((_part1by2(q[..., 0]) << 2 |
		                             _part1by2(q[..., 1]) << 1 |
		                             _part1by2(q[..., 2])) & ((1 << 44) - 1))
	keys = key_of(g, batch)
	S, perm = torch.sort(keys)
	n = pos.size(0)
	r = torch.arange(-R, R + 1, device=dev)
	rz = torch.arange(-Rz, Rz + 1, device=dev)
	offs = torch.stack(torch.meshgrid(r, r, rz, indexing='ij'), dim=-1).reshape(-1, 3)
	K = offs.size(0)
	q = g.unsqueeze(1) + offs.unsqueeze(0)                     # N x K x 3
	nk = key_of(q, batch.unsqueeze(1).expand(-1, K))           # N x K
	p = torch.searchsorted(S, nk.reshape(-1)).clamp(max=n - 1)
	found = S[p] == nk.reshape(-1)
	own = torch.arange(n, device=dev).unsqueeze(1).expand(n, K).reshape(-1)
	col = torch.where(found, perm[p], own).view(n, K)
	return col, (~found).view(n, K)


def stencil_groups(radius, device):
	"""Cylindrical direction-group id per stencil offset: (z level) x (centre | ring).
	Yaw-invariant, so it is compatible with full-rotation augmentation. Order matches
	stencil_neighbors."""
	r = torch.arange(-radius, radius + 1, device=device)
	offs = torch.stack(torch.meshgrid(r, r, r, indexing='ij'), dim=-1).reshape(-1, 3)
	ring = ((offs[:, 0] != 0) | (offs[:, 1] != 0)).long()      # 0 = on the z-axis
	return (offs[:, 2] + radius) * 2 + ring                     # (K,) int64


def window_neighbors(n, k, sample_start, sample_end):
	"""Serialized-window neighbours: +-k/2 positions in sorted order, clamped to each
	sample's range."""
	device = sample_start.device
	offs = torch.arange(k, device=device) - k // 2
	offs = offs + (offs >= 0).long()          # -k/2..-1, 1..k/2 (skip self; self is added by the residual path)
	col = torch.arange(n, device=device).unsqueeze(1) + offs.unsqueeze(0)
	return col.clamp(sample_start.unsqueeze(1), (sample_end - 1).unsqueeze(1))


# --- building blocks -----------------------------------------------------------------

class MetaBlock(nn.Module):
	"""Local aggregation block: point-wise MLP -> neighbour gather (+ relative-position
	encoding, optional feature-difference term and directional gating) -> max -> MLP."""
	def __init__(self, in_channels, out_channels, k=32, feature_diff=False, dir_groups=0):
		super().__init__()
		self.k = k
		self.residual = in_channels == out_channels
		# dir_groups > 0: per-direction channel scale/bias on neighbour features (zero-init).
		if dir_groups > 0:
			self.dir_scale = nn.Parameter(torch.zeros(dir_groups, out_channels))
			self.dir_bias = nn.Parameter(torch.zeros(dir_groups, out_channels))
		else:
			self.dir_scale = self.dir_bias = None
		self.pre = nn.Sequential(
			nn.Linear(in_channels, out_channels),
			nn.BatchNorm1d(out_channels),
			nn.ReLU(),
		)
		self.pose = nn.Linear(3, out_channels)
		# feature_diff: W2*(h_j - h_i) computed per point and gathered (no per-edge MLP).
		self.diff = nn.Linear(out_channels, out_channels, bias=False) if feature_diff else None
		self.post = nn.Sequential(
			nn.Linear(out_channels, out_channels),
			nn.BatchNorm1d(out_channels),
		)
		self.act = nn.ReLU()

	def forward(self, x, pos, col, mask=None, gvec=None):
		h = self.pre(x)                                        # N x C  (per point)
		if gvec is not None and self.dir_scale is not None:
			# Gating applied per group before the gather to keep backward memory low.
			n = h.size(0)
			hG = h.unsqueeze(1) * (1.0 + self.dir_scale)       # N x G x C
			base = hG.reshape(-1, hG.size(-1))[col * self.dir_scale.size(0) + gvec]
			e = base + self.pose(pos[col] - pos.unsqueeze(1)) + self.dir_bias[gvec]
		else:
			e = h[col] + self.pose(pos[col] - pos.unsqueeze(1))    # N x k x C (gather only)
		if self.diff is not None:
			b = self.diff(h)                                   # N x C  (per point)
			e = e + (b[col] - b.unsqueeze(1))                  # decomposed W2*(h_j - h_i)
		if mask is not None:                                   # empty stencil cells
			e = e.masked_fill(mask.unsqueeze(-1), float('-inf'))
		agg = e.max(dim=1).values
		out = self.post(agg)
		if self.residual:
			out = out + x
		return self.act(out)


def grid_pool(x, pos, batch, grid, reduce='max'):
	"""Voxel-grid pooling. Returns (x_p, pos_p, batch_p, cluster); unpooling is
	x_p[cluster]."""
	g = ((pos - pos.min(dim=0).values) / grid).long()
	key = (batch.long() << 44) | ((_part1by2(g[:, 0]) << 2 |
	                               _part1by2(g[:, 1]) << 1 |
	                               _part1by2(g[:, 2])) & ((1 << 44) - 1))
	uniq, cluster = torch.unique(key, return_inverse=True)
	m = uniq.numel()
	x_p = scatter(x, cluster, dim=0, dim_size=m, reduce=reduce)
	pos_p = scatter(pos, cluster, dim=0, dim_size=m, reduce='mean')
	batch_p = scatter(batch, cluster, dim=0, dim_size=m, reduce='max')
	return x_p, pos_p, batch_p, cluster


def _sample_ranges(batch):
	"""Per-point [start, end) of its own sample in a batch-sorted array."""
	counts = torch.bincount(batch)
	ends = counts.cumsum(0)
	starts = ends - counts
	return starts[batch], ends[batch]


class PointEdgeSegNet(nn.Module):
	"""Stencil backbone. Interface-compatible with the EdgeConv architecture: forward(data) with data.x/pos/batch,
	per-point logits out. Stage downsampling is by voxel grid (pool_grids)."""
	def __init__(self, num_features, num_classes, feature_dims=(4, 3, 0, 0), knn=32,
				 transformer_layers=2, enc_channels=(64, 192, 320, 448),
				 bottleneck_dim=256, base_grid=0.04, pool_grids=(0.08, 0.16, 0.32),
				 curves=1, neighbor_mode='serial', stencil_radius=1, feature_diff=False,
				 directional=False, stencil_z=None):
		super().__init__()
		dims = tuple(feature_dims) + (0,) * (4 - len(feature_dims))
		geo_dim, rgb_dim, spatial_dim, context_dim = dims
		assert sum(dims) == num_features
		assert context_dim == 0, "v2 does not carry the (twice-failed) block-context path"
		self.k = knn
		self.curves = curves        # Morton curves unioned per stage (independent seams)
		self.base_grid = base_grid
		self.pool_grids = pool_grids
		assert neighbor_mode in ('serial', 'stencil')
		self.neighbor_mode = neighbor_mode   # 'serial' | 'stencil'
		self.stencil_radius = stencil_radius # lattice-offset lookup radius (K=(2r+1)^3)
		self.stencil_z = stencil_z           # optional taller vertical reach
		c1, c2, c3, c4 = [int(c) for c in enc_channels]
		bdim = int(bottleneck_dim) if bottleneck_dim else c4
		self.point_in_dim = num_features

		self.feature_gate = FeatureGate(geo_dim=geo_dim, rgb_dim=rgb_dim,
										spatial_dim=spatial_dim, context_dim=0)

		# Encoder: two meta blocks per stage (mirrors v1's conv_n/conv_n_2 depth)
		self.directional = directional and neighbor_mode == 'stencil'
		G = (2 * stencil_radius + 1) * 2 if self.directional else 0
		def MB(ci, co):
			return MetaBlock(ci, co, k=knn, feature_diff=feature_diff, dir_groups=G)
		self.enc = nn.ModuleList([
			nn.ModuleList([MB(self.point_in_dim, c1), MB(c1, c1)]),
			nn.ModuleList([MB(c1, c2), MB(c2, c2)]),
			nn.ModuleList([MB(c2, c3), MB(c3, c3)]),
			nn.ModuleList([MB(c3, c4)]),
		])

		self.bottleneck_proj = nn.Linear(c4, bdim) if bdim != c4 else nn.Identity()
		self.bottleneck_unproj = nn.Linear(bdim, c4) if bdim != c4 else nn.Identity()
		self.bottleneck_transformer = LightweightTransformer(
			dim=bdim, num_heads=max(1, bdim // 128), dropout=0.1,
			spatial_dim=spatial_dim, num_layers=transformer_layers)

		# Decoder: unpool (free index_select) + skip concat + point MLP, as in v1
		def dec_mlp(cin, cout):
			return nn.Sequential(nn.Linear(cin, cout), nn.BatchNorm1d(cout), nn.ReLU())
		self.dec3 = dec_mlp(c4 + c3, c3)
		self.dec2 = dec_mlp(c3 + c2, c2)
		self.dec1 = dec_mlp(c2 + c1, c1)

		h1, h2 = 96, 48
		self.head_attention = AttentionModule(c1 + self.point_in_dim)
		self.head = nn.Sequential(
			nn.Linear(c1 + self.point_in_dim, h1), nn.BatchNorm1d(h1), nn.ReLU(),
			nn.Dropout(0.5),
			nn.Linear(h1, h2), nn.BatchNorm1d(h2), nn.ReLU(),
			nn.Dropout(0.4),
			nn.Linear(h2, num_classes),
		)

	def _stage_prep(self, pos, batch, perm):
		"""Serialize one resolution level; with curves > 1 the neighbour set is the union
		of windows on differently oriented curves, expressed in the first curve's order."""
		n = pos.size(0)
		order = serialize(pos, batch, self.base_grid, perm=perm)
		inv = torch.empty_like(order)
		inv[order] = torch.arange(n, device=order.device)
		start, end = _sample_ranges(batch[order])
		kc = self.k // self.curves
		col = window_neighbors(n, kc, start, end)
		for c in range(1, self.curves):
			o2 = serialize(pos, batch, self.base_grid, perm=perm + c)
			inv2 = torch.empty_like(o2)
			inv2[o2] = torch.arange(n, device=o2.device)
			s2, e2 = _sample_ranges(batch[o2])
			col2 = window_neighbors(n, kc, s2, e2)      # windows in curve-2 order
			# remap: curve-2 window of point p, expressed in curve-1 positions
			col = torch.cat([col, inv[o2[col2]][inv2[order]]], dim=1)
		return order, inv, col

	def forward(self, data):
		x, pos, batch = data.x, data.pos, data.batch
		centroid = scatter(pos, batch, dim=0, reduce='mean')
		pos = pos - centroid[batch]

		x_gated, gates = self.feature_gate(x)
		self.last_gates = gates.detach()

		# ---- encoder ----
		feats, clusters = [], []
		h, p, b = x_gated, pos, batch
		for s, blocks in enumerate(self.enc):
			if self.neighbor_mode == 'stencil':
				stage_grid = self.base_grid if s == 0 else self.pool_grids[s - 1]
				col, mask = stencil_neighbors(p, b, stage_grid, self.stencil_radius,
				                              z_radius=self.stencil_z)
				# Directional gating from stage 1 onward (stage 0 dominates cost).
				gvec = (stencil_groups(self.stencil_radius, p.device)
						if self.directional and s >= 1 else None)
				for blk in blocks:
					h = blk(h, p, col, mask, gvec)
			else:
				order, inv, col = self._stage_prep(p, b, perm=s)
				hs, ps = h[order], p[order]
				for blk in blocks:
					hs = blk(hs, ps, col)
				h = hs[inv]                               # back to canonical order
			if s < len(self.enc) - 1:
				feats.append((h, p, b))
				h, p, b, cluster = grid_pool(h, p, b, self.pool_grids[s])
				clusters.append(cluster)

		# ---- bottleneck ----
		h = self.bottleneck_unproj(
			self.bottleneck_transformer(self.bottleneck_proj(h), p, None, b))

		# ---- decoder (unpool = index_select through the stored cluster maps) ----
		for dec, cluster, (skip, _, _) in zip((self.dec3, self.dec2, self.dec1),
											  reversed(clusters), reversed(feats)):
			h = dec(torch.cat([h[cluster], skip], dim=1))

		final = self.head_attention(torch.cat([h, x_gated], dim=1))
		return self.head(final)
