# Title: PointEdgeSegNet (version 0.4)
# Author: taewook kang (laputa9999@gmail.com)
# Date: 2025-09-21
# Purpose: Further optimized PointEdgeSegNet model architecture for memory efficiency.
# Dependencies: torch, torch_geometric

import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import fps, knn_interpolate
from torch_geometric.nn.pool import knn_graph
from torch_geometric.utils import scatter
from torch.cuda.amp import autocast, GradScaler

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

class LightweightTransformer(nn.Module):
	"""Lightweight Transformer for Bottleneck (XYZ + Spatial Position Encoding).

	Now stacks num_layers pre-norm attention+FFN blocks (was a single block). The bottleneck
	operates on the FPS-downsampled set (~200 points), so extra layers cost almost no memory
	but add global-context depth where it is cheapest.
	"""
	def __init__(self, dim=512, num_heads=4, dropout=0.1, spatial_dim=3, num_layers=2):
		super().__init__()
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

		# Position encoding over XYZ(3) + Spatial(spatial_dim). Default spatial_dim=3 -> 6D,
		# identical to the original; when spatial features are disabled it falls back to 3D (XYZ).
		self.pos_enc = nn.Sequential(
			nn.Linear(3 + spatial_dim, dim // 4),
			nn.ReLU(),
			nn.Linear(dim // 4, dim)
		)
	
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

class PointEdgeSegNet(nn.Module):
	def __init__(self, num_features, num_classes, feature_dims=(4, 3, 3), knn=32, transformer_layers=2):
		super(PointEdgeSegNet, self).__init__()
		# knn: EdgeConv neighbourhood size (was 20; 32 gives a richer local graph).
		# transformer_layers: bottleneck attention depth (was 1). Both are architecture-defining,
		# so train and inference must use the SAME values -> they default here and every caller
		# constructs with the defaults. Changing them requires retraining (checkpoint changes).

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

		# Add Feature Gate at input (minimal memory: ~2MB for 20 batches)
		self.feature_gate = FeatureGate(geo_dim=geo_dim, rgb_dim=rgb_dim, spatial_dim=spatial_dim,
										context_dim=context_dim)

		# Optimized Encoder - reduced channels and removed deepest layer
		self.conv1 = EdgeConv(num_features, 64, k=knn)
		self.conv1_2 = EdgeConv(64, 64, residual=True, k=knn)
		self.conv2 = EdgeConv(64, 128, k=knn)
		self.conv2_2 = EdgeConv(128, 128, residual=True, k=knn)
		self.conv3 = EdgeConv(128, 256, k=knn)
		self.conv3_2 = EdgeConv(256, 256, residual=True, k=knn)
		self.conv4 = EdgeConv(256, 512, k=knn)
		# Bottleneck Transformer with (now centered) Position Encoding (spatial-dim aware)
		self.bottleneck_transformer = LightweightTransformer(dim=512, num_heads=4, dropout=0.1,
															  spatial_dim=spatial_dim, num_layers=transformer_layers)

		# Simplified Decoder
		self.deconv1_mlp = nn.Sequential(
			nn.Linear(512 + 256, 256),  # Updated to match 512 channels
			nn.BatchNorm1d(256), 
			nn.ReLU()
		)
		self.deconv2_mlp = nn.Sequential(
			nn.Linear(256 + 128, 128), 
			nn.BatchNorm1d(128), 
			nn.ReLU()
		)
		self.deconv3_mlp = nn.Sequential(
			nn.Linear(128 + 64, 64), 
			nn.BatchNorm1d(64), 
			nn.ReLU()
		)

		# Optimized prediction head
		self.head_attention = AttentionModule(64 + num_features)
		self.head = nn.Sequential(
			nn.Linear(64 + num_features, 96),  # Reduced from 128
			nn.BatchNorm1d(96), 
			nn.ReLU(),
			nn.Dropout(0.5),
			nn.Linear(96, 48),  # Reduced from 64
			nn.BatchNorm1d(48),
			nn.ReLU(),
			nn.Dropout(0.4),
			nn.Linear(48, num_classes)
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

		# Apply feature gating at input
		x_gated, gates = self.feature_gate(x)
		x0, pos0, batch0 = x_gated, pos, batch
		
		# Store gates for analysis (keep on GPU)
		self.last_gates = gates.detach()

		# Optimized encoder with fewer layers
		x1 = self.conv1(x0, pos0, batch0)
		x1 = self.conv1_2(x1, pos0, batch0)
		idx1 = fps(pos0, batch0, ratio=0.35)
		pos1, x1_sampled, batch1 = pos0[idx1], x1[idx1], batch0[idx1]

		x2 = self.conv2(x1_sampled, pos1, batch1)
		x2 = self.conv2_2(x2, pos1, batch1)
		idx2 = fps(pos1, batch1, ratio=0.3)
		pos2, x2_sampled, batch2 = pos1[idx2], x2[idx2], batch1[idx2]

		x3 = self.conv3(x2_sampled, pos2, batch2)
		x3 = self.conv3_2(x3, pos2, batch2)
		idx3 = fps(pos2, batch2, ratio=0.25)
		pos3, x3_sampled, batch3 = pos2[idx3], x3[idx3], batch2[idx3]
		
		# Extract spatial features from original gated features (configured slice)
		if self.spatial_dim > 0:
			spatial0 = x_gated[:, self.spatial_start:self.spatial_start + self.spatial_dim]
			# Downsample spatial features following the same FPS indices
			spatial3 = spatial0[idx1][idx2][idx3]
		else:
			spatial3 = None  # transformer falls back to XYZ-only position encoding

		# Final encoder layer (bottleneck)
		x4_bottleneck = self.conv4(x3_sampled, pos3, batch3)
		# Apply Transformer with XYZ (and Spatial, if present) position encoding
		x4_bottleneck = self.bottleneck_transformer(x4_bottleneck, pos3, spatial3, batch3)

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