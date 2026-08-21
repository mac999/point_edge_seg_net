# models/common.py
# Building blocks shared by every PointEdgeSegNet architecture.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter


class FeatureGate(nn.Module):
	"""Lightweight feature-wise gating for Geo, RGB, Spatial and Block-Context groups.

	Adapts to the configured feature layout: any group whose dim is 0 (e.g. a colorless
	cloud with rgb_dim=0, or context_dim=0 when block context is off) is skipped, and a
	gate is learned only for the present groups. With the default (geo=4, rgb=3,
	spatial=3, context=0) this is byte-identical to the original 10D gate, so existing
	weights load unchanged.
	"""
	def __init__(self, geo_dim=4, rgb_dim=3, spatial_dim=3, context_dim=0):
		super(FeatureGate, self).__init__()
		self.geo_dim = geo_dim
		self.rgb_dim = rgb_dim
		self.spatial_dim = spatial_dim
		self.context_dim = context_dim
		self.dims = [geo_dim, rgb_dim, spatial_dim, context_dim]
		self.offsets = [0, geo_dim, geo_dim + rgb_dim, geo_dim + rgb_dim + spatial_dim]
		self.present = [i for i, d in enumerate(self.dims) if d > 0]  # which groups exist
		total_dim = sum(self.dims)
		num_gates = len(self.present)

		# Shared feature encoder (minimal overhead); one gate per present group
		self.encoder = nn.Sequential(
			nn.Linear(total_dim, 16),
			nn.ReLU(),
			nn.Linear(16, num_gates)
		)
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		# x: [N, total_dim] laid out as [geo | rgb | spatial | context] (absent groups omitted)
		gates = self.sigmoid(self.encoder(x))  # [N, num_gates]

		out = []
		# full_gates keeps a stable [N, 4] (geo, rgb, spatial, context) view for logging;
		# absent groups report gate 0 (indices 0-2 keep their original meaning).
		full_gates = x.new_zeros(x.size(0), 4)
		gi = 0
		for group_idx, d in enumerate(self.dims):
			if d == 0:
				continue
			off = self.offsets[group_idx]
			seg = x[:, off:off + d]
			g = gates[:, gi:gi + 1]
			out.append(seg * g)
			full_gates[:, group_idx] = gates[:, gi]
			gi += 1

		return torch.cat(out, dim=1), full_gates

class AttentionModule(nn.Module):
	def __init__(self, channels):
		super(AttentionModule, self).__init__()
		self.attention = nn.Sequential(
			nn.Linear(channels, channels // 4),
			nn.ReLU(),
			nn.Linear(channels // 4, channels),
			nn.Sigmoid()
		)
	
	def forward(self, x):
		att_weights = self.attention(x)
		return x * att_weights

class LightweightTransformer(nn.Module):
	"""Lightweight Transformer for Bottleneck (XYZ + Spatial Position Encoding).

	Now stacks num_layers pre-norm attention+FFN blocks (was a single block). The bottleneck
	operates on the FPS-downsampled set (~200 points), so extra layers cost almost no memory
	but add global-context depth where it is cheapest.
	"""
	def __init__(self, dim=512, num_heads=4, dropout=0.1, spatial_dim=3, num_layers=2,
				 zero_init_residual=False):
		super().__init__()
		# zero_init_residual: start every residual branch (attention out-proj + FFN last
		# linear) at zero so the block is an EXACT identity at init. Required when the
		# transformer is inserted into the MIDDLE of a trained-from-scratch encoder: with
		# default init its output perturbs all downstream features and stalls optimization
		# (measured: logs/20260723_091040 -- mid-level insertion without this collapsed
		# training to val_acc 0.79 vs 0.93). The historical bottleneck placement keeps the
		# default (False) so existing checkpoints/behaviour are unchanged.
		self.spatial_dim = spatial_dim
		self.num_layers = num_layers
		self.layers = nn.ModuleList([
			nn.ModuleDict({
				'norm1': nn.LayerNorm(dim),
				'attn': nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True),
				'norm2': nn.LayerNorm(dim),
				'ffn': nn.Sequential(
					nn.Linear(dim, dim * 2),
					nn.GELU(),
					nn.Dropout(dropout),
					nn.Linear(dim * 2, dim),
					nn.Dropout(dropout)
				),
			}) for _ in range(num_layers)
		])
		self._zero_init_residual = zero_init_residual
		if zero_init_residual:
			for layer in self.layers:
				nn.init.zeros_(layer['attn'].out_proj.weight)
				nn.init.zeros_(layer['attn'].out_proj.bias)
				nn.init.zeros_(layer['ffn'][3].weight)
				nn.init.zeros_(layer['ffn'][3].bias)

		# Position encoding over XYZ(3) + Spatial(spatial_dim). Default spatial_dim=3 -> 6D,
		# identical to the original; when spatial features are disabled it falls back to 3D (XYZ).
		self.pos_enc = nn.Sequential(
			nn.Linear(3 + spatial_dim, dim // 4),
			nn.ReLU(),
			nn.Linear(dim // 4, dim)
		)
		if zero_init_residual:
			# pos_enc is added straight to the trunk (x + pos_emb), so it must start at
			# zero as well for the block to be an exact identity (see docstring).
			nn.init.zeros_(self.pos_enc[2].weight)
			nn.init.zeros_(self.pos_enc[2].bias)
	
	def forward(self, x, pos, spatial, batch):
		"""
		Args:
			x: [Total_Points, 512] (features)
			pos: [Total_Points, 3] (xyz coordinates)
			spatial: [Total_Points, spatial_dim] or None (density, anisotropy, structure)
			batch: [Total_Points] (batch assignment)
		"""
		use_spatial = self.spatial_dim > 0 and spatial is not None
		# Force float32 for indexing operations (FP16 not supported)
		original_dtype = x.dtype
		if x.dtype == torch.float16:
			x = x.float()
			pos = pos.float()
			if use_spatial:
				spatial = spatial.float()

		batch_size = batch.max().item() + 1
		max_points = scatter(torch.ones_like(batch), batch, reduce='sum').max().item()

		# Use original dtype for tensor creation
		x_batched = torch.zeros(batch_size, max_points, x.size(1), device=x.device, dtype=x.dtype)
		pos_batched = torch.zeros(batch_size, max_points, 3, device=x.device, dtype=x.dtype)
		batch_mask = torch.zeros(batch_size, max_points, dtype=torch.bool, device=x.device)
		if use_spatial:
			spatial_batched = torch.zeros(batch_size, max_points, self.spatial_dim, device=x.device, dtype=x.dtype)

		for b in range(batch_size):
			mask = (batch == b)
			num_points = mask.sum().item()
			x_batched[b, :num_points] = x[mask]
			pos_batched[b, :num_points] = pos[mask]
			if use_spatial:
				spatial_batched[b, :num_points] = spatial[mask]
			batch_mask[b, :num_points] = True

		# Combine XYZ (+ Spatial) for position encoding
		combined_pos = torch.cat([pos_batched, spatial_batched], dim=2) if use_spatial else pos_batched
		pos_emb = self.pos_enc(combined_pos)
		x_batched = x_batched + pos_emb

		# Stacked pre-norm attention + FFN blocks
		for layer in self.layers:
			x_norm = layer['norm1'](x_batched)
			attn_out, _ = layer['attn'](x_norm, x_norm, x_norm, key_padding_mask=~batch_mask)
			x_batched = x_batched + attn_out
			x_batched = x_batched + layer['ffn'](layer['norm2'](x_batched))

		# [Batch, Max_Points, C] to [Total_Points, C]
		x_out = torch.zeros_like(x)
		for b in range(batch_size):
			mask = (batch == b)
			num_points = mask.sum().item()
			x_out[mask] = x_batched[b, :num_points]
		
		# Convert back to original dtype if needed
		if original_dtype == torch.float16:
			x_out = x_out.half()
		
		return x_out

