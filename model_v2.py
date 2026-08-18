# model_v2.py
# PointEdgeSegNet v2: serialization + meta-aggregation rework of the same model.
# The public name stays PointEdgeSegNet — this is an internal architecture revision;
# once validated it becomes THE PointEdgeSegNet of the next GitHub release, and the
# EdgeConv+kNN implementation in model.py remains as the legacy/rollback path.
#
# WHY (all measured on this box, N=98,304, RTX PRO 6000 Blackwell — see VERSIONS.md):
#   1. torch_cluster knn_graph costs 451 ms/call and v1 calls it 7x per forward: 99% of
#      forward time at room scale. PTv2 rebuilt its architecture over a 28% share; ours
#      is 99%. Fix: sort points ONCE per stage along a Morton (z-order) curve (16 ms) and
#      take each point's neighbours as a +-k/2 window in sorted order. No neighbour
#      search anywhere in the network (PTv3's "serialized neighbor mapping").
#   2. v1's EdgeConv runs a 2-layer MLP on every EDGE (N*k x 2C tensor): 22.9 ms /
#      3.85 GB per stage-1 block. Moving the MLP to POINTS and aggregating with max
#      (PointMetaBase "MLP-before-Group", DeLA "decoupled aggregation") measures
#      7.5 ms / 1.28 GB — 3x cheaper — with relative-position encoding preserving the
#      geometry cue. Quality upper bound is established by DeLA (74.1 mIoU on S3DIS).
#   3. v1's decoder knn_interpolate needs 3 more kNN queries. Serialized grid pooling
#      keeps the point->voxel cluster map, so unpooling is a free index_select
#      (PTv3's serialized unpooling) — zero search, exact inverse of pooling.
#
# KEPT from v1 (measured to be fine): U-Net topology + channel plan, FeatureGate,
# bottleneck LightweightTransformer (runs on the coarsest set, cheap), prediction head
# with input skip, per-sample coordinate centering.
#
# NOT kept: any weight compatibility. This is a from-scratch architecture (v0.8.x).

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter

from model import FeatureGate, AttentionModule, LightweightTransformer


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
	"""Sort points along a Morton curve, samples kept contiguous.

	Returns `order` (int64 indices into pos). The curve axis order rotates with `perm`
	(PTv3 alternates curve patterns per layer so thin structures are not always cut at
	the same seam). Quantization grid sets curve resolution; anything at or below the
	data's voxel size is fine.
	"""
	a, b, c = _AXIS_PERMS[perm % 3]
	g = ((pos - pos.min(dim=0).values) / grid).long()
	key = (_part1by2(g[:, a]) << 2) | (_part1by2(g[:, b]) << 1) | _part1by2(g[:, c])
	# batch id in the top bits keeps samples contiguous after one global sort
	key = (batch.long() << 44) | (key & ((1 << 44) - 1))
	return torch.argsort(key)

def stencil_neighbors(pos, batch, grid, radius=1, z_radius=None):
	"""Exact metric neighbours by voxel-stencil hash lookup — no search, no approximation.

	Every stage's points sit on a voxel grid (the input is grid-voxelized and every
	pooling step is a grid_pool), so a point's spatial neighbours are exactly the
	occupied voxels at fixed offsets: sort the voxel keys once, then one searchsorted
	per stencil offset. This is sparse convolution's kernel map (MinkowskiNet et al.)
	— it combines serialization's zero-search cost with kNN's true neighbourhoods,
	which the E1/E2 ablations showed is what the serialized windows lack (door/board/
	chair never recovered with wider windows; see VERSIONS.md v0.8.x log).

	Returns (col, mask): (N, K) neighbour indices with K=(2*radius+1)^3 (self included
	at the centre offset) and a bool mask that is True where the offset voxel is empty.
	Duplicate keys (stage-0 padding repeats points) resolve to one representative,
	which is harmless because duplicates carry identical features.
	"""
	R = int(radius)
	# z_radius > radius extends the stencil VERTICALLY only (E10): columns/pipes span
	# many z-levels while their xy footprint stays tiny, and E8 showed that per-level
	# gating alone loses column IoU — the receptive field itself must reach further
	# along gravity. K = (2R+1)^2 * (2Rz+1).
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
	# All K offsets in one batched morton + ONE searchsorted: a per-offset python loop
	# measured 136 ms here; this vectorized form is 1.3 ms for identical output.
	q = g.unsqueeze(1) + offs.unsqueeze(0)                     # N x K x 3
	nk = key_of(q, batch.unsqueeze(1).expand(-1, K))           # N x K
	p = torch.searchsorted(S, nk.reshape(-1)).clamp(max=n - 1)
	found = S[p] == nk.reshape(-1)
	# Missing-cell sentinel is each point's OWN index, not 0: gather backward is a
	# scatter_add, and a shared sentinel row serializes millions of atomic adds onto one
	# address (measured 6.5 s/step); self-indices spread them evenly (65 ms/step). The
	# mask still removes these entries from the max, so numerics are unchanged.
	own = torch.arange(n, device=dev).unsqueeze(1).expand(n, K).reshape(-1)
	col = torch.where(found, perm[p], own).view(n, K)
	return col, (~found).view(n, K)


def stencil_groups(radius, device):
	"""Direction-group id per stencil offset, for anisotropic aggregation (E8).

	Full per-offset weight matrices (true sparse-conv kernels) are pointless here:
	training uses full 2*pi yaw augmentation, so x/y directions are not canonical and
	xy-anisotropic weights would just be averaged out. Gravity (z) IS canonical, so
	offsets are grouped CYLINDRICALLY: (vertical level dz) x (centre vs horizontal ring).
	G = (2r+1)*2 groups (r1: 6, r2: 10). This is what vertical structure needs --
	columns are vertical stacks, windows/boards live on vertical walls, beams are
	horizontal at ceiling height -- while remaining invariant to yaw augmentation.
	Order matches the meshgrid order used by stencil_neighbors.
	"""
	r = torch.arange(-radius, radius + 1, device=device)
	offs = torch.stack(torch.meshgrid(r, r, r, indexing='ij'), dim=-1).reshape(-1, 3)
	ring = ((offs[:, 0] != 0) | (offs[:, 1] != 0)).long()      # 0 = on the z-axis
	return (offs[:, 2] + radius) * 2 + ring                     # (K,) int64


def window_neighbors(n, k, sample_start, sample_end):
	"""Neighbour indices for serialized points: the +-k/2 window in sorted order.

	sample_start/sample_end give each point's own sample's [start, end) range so windows
	never cross sample boundaries (clamping duplicates edge neighbours, which is
	harmless under max aggregation). Returns col of shape (n, k); row is implicit
	(each point i owns row i).
	"""
	device = sample_start.device
	offs = torch.arange(k, device=device) - k // 2
	offs = offs + (offs >= 0).long()          # -k/2..-1, 1..k/2 (skip self; self is added by the residual path)
	col = torch.arange(n, device=device).unsqueeze(1) + offs.unsqueeze(0)
	return col.clamp(sample_start.unsqueeze(1), (sample_end - 1).unsqueeze(1))


# --- building blocks -----------------------------------------------------------------

class MetaBlock(nn.Module):
	"""PointMetaBase/DeLA-style aggregation over a serialized window.

	point MLP -> gather window -> +relative-position encoding -> max -> point MLP.
	The only per-neighbour work is a gather and one linear on the 3D offset: no per-edge
	feature MLP (v1's cost center). Residual when shapes allow.
	"""
	def __init__(self, in_channels, out_channels, k=32, feature_diff=False, dir_groups=0):
		super().__init__()
		self.k = k
		self.residual = in_channels == out_channels
		# dir_groups > 0: anisotropic aggregation (E8). Each direction group g gets a
		# learned channel-wise scale and bias applied to the neighbour features BEFORE
		# the max — a cheap diagonal approximation of sparse conv's per-offset weight
		# matrices (full matrices would add ~2M params/block at c4=448; this adds 2*G*C).
		# Zero-initialised, so at init the block is EXACTLY the isotropic (E5) block.
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
		# feature_diff: restore EdgeConv's geometric-gradient cue W2*(h_j - h_i) WITHOUT a
		# per-edge MLP: W2 is applied per POINT once, the difference is a gather + subtract.
		# (v1's edge feature was cat(h_i, h_j - h_i) through a 2-layer edge MLP; this is
		# the decomposed, k-times-cheaper equivalent of its first layer.)
		self.diff = nn.Linear(out_channels, out_channels, bias=False) if feature_diff else None
		self.post = nn.Sequential(
			nn.Linear(out_channels, out_channels),
			nn.BatchNorm1d(out_channels),
		)
		self.act = nn.ReLU()

	def forward(self, x, pos, col, mask=None, gvec=None):
		h = self.pre(x)                                        # N x C  (per point)
		if gvec is not None and self.dir_scale is not None:
			# Anisotropic gating (E8), applied PRE-gather: scaling the gathered N x K x C
			# tensor directly makes autograd save an extra N*K*C operand (measured 3.3x
			# memory, +89% time). Scaling h per GROUP first costs only N x G x C (G=K/12)
			# and the gather's backward stays index-only.
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
	"""Voxel-grid pooling that keeps the inverse map for free unpooling.

	Returns (x_p, pos_p, batch_p, cluster) where cluster maps every fine point to its
	pooled row: unpooling is x_p[cluster]. Feature reduce is max (matches v1's
	aggregation character); positions average so pooled points sit at voxel centroids.
	"""
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


class PointEdgeSegNetV2(nn.Module):
	"""Serialized, meta-aggregation PointEdgeSegNet (v0.8.x architecture).

	Interface-compatible with v1: forward(data) with data.x/pos/batch, per-point logits
	out. Constructor args mirror v1 where they still apply; stage downsampling is by
	voxel grid (pool_grids) instead of ratios, because grid pooling is what gives the
	free unpooling map.
	"""
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
		self.curves = curves        # Morton curves whose windows are unioned per stage:
		                            # 1 = a single +-k/2 window (curve axis-order rotates per
		                            # stage); 2 = two curves x k/2 window each — same k budget
		                            # but two independent seams, so a neighbourhood cut by one
		                            # curve is usually intact on the other (PTv3 uses 4 curve
		                            # patterns for the same reason).
		self.base_grid = base_grid
		self.pool_grids = pool_grids
		assert neighbor_mode in ('serial', 'stencil')
		self.neighbor_mode = neighbor_mode   # 'serial': +-k/2 window(s) on Morton curve(s)
		self.stencil_radius = stencil_radius #  'stencil': exact voxel-offset neighbours,
		                                     #  K=(2r+1)^3 (sparse-conv kernel map)
		self.stencil_z = stencil_z           #  optional taller z-reach (E10)
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
		"""Serialize one resolution level: returns (order, inverse, col).

		With curves > 1 the neighbour set is the union of +-(k/curves)/2 windows taken on
		`curves` differently-oriented Morton curves. col indices are expressed in the
		FIRST curve's order (the one features are laid out in), so extra curves only add
		an index remap, not extra sorts of the feature tensor.
		"""
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
				# Directional gating from stage 1 onward: stage 0 holds ~70% of the points
				# (cost) while object-scale verticality lives at pooled resolutions (value).
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
