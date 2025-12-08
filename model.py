# Title: PointEdgeSegNet (version 0.3)
# Author: taewook kang (laputa9999@gmail.com)
# Date: 2025-09-21
# Purpose: Further optimized PointEdgeSegNet model architecture for memory efficiency.
# Dependencies: torch, torch_geometric

import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import fps, knn_interpolate
from torch_geometric.nn.pool import knn_graph
from torch_geometric.utils import scatter

class FeatureGate(nn.Module):
	"""Lightweight feature-wise gating for Geo, RGB and Spatial features."""
	def __init__(self, geo_dim=4, rgb_dim=3, spatial_dim=3):
		super(FeatureGate, self).__init__()
		self.geo_dim = geo_dim
		self.rgb_dim = rgb_dim
		self.spatial_dim = spatial_dim
		total_dim = geo_dim + rgb_dim + spatial_dim
		
		# Shared feature encoder (minimal overhead: ~200 params)
		self.encoder = nn.Sequential(
			nn.Linear(total_dim, 16),
			nn.ReLU(),
			nn.Linear(16, 3)  # 3 gates: geo, rgb, spatial
		)
		self.sigmoid = nn.Sigmoid()
	
	def forward(self, x):
		# x: [N, 10] = [geo(4) + rgb(3) + spatial(3)]
		gates = self.sigmoid(self.encoder(x))  # [N, 3]
		
		# Split features
		geo = x[:, :self.geo_dim]
		rgb = x[:, self.geo_dim:self.geo_dim+self.rgb_dim]
		spatial = x[:, self.geo_dim+self.rgb_dim:]
		
		# Apply gates
		geo_gated = geo * gates[:, 0:1]
		rgb_gated = rgb * gates[:, 1:2]
		spatial_gated = spatial * gates[:, 2:3]
		
		return torch.cat([geo_gated, rgb_gated, spatial_gated], dim=1), gates

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
	def __init__(self, in_channels, out_channels, residual=False):
		super(EdgeConv, self).__init__()
		self.residual = residual and (in_channels == out_channels)
		
		self.mlp = nn.Sequential(
			nn.Linear(2 * in_channels, out_channels), 
			nn.BatchNorm1d(out_channels), 
			nn.ReLU(),
			nn.Linear(out_channels, out_channels), 
			nn.BatchNorm1d(out_channels)
		)
		self.attention = AttentionModule(out_channels)
		self.final_activation = nn.ReLU()

	def forward(self, x, pos, batch, k=20):  # Reduced k for memory efficiency
		edge_index = knn_graph(pos, k=k, batch=batch, loop=False)
		row, col = edge_index
		
		edge_features = torch.cat([x[row], x[col] - x[row]], dim=1)
		out = self.mlp(edge_features)
		
		aggr_out = scatter(out, row, dim=0, dim_size=x.size(0), reduce='max')
		aggr_out = self.attention(aggr_out)
		
		if self.residual:
			aggr_out = aggr_out + x
			
		return self.final_activation(aggr_out)

class PointEdgeSegNet(nn.Module):
	def __init__(self, num_features, num_classes):
		super(PointEdgeSegNet, self).__init__()
		
		# Add Feature Gate at input (minimal memory: ~2MB for 20 batches)
		self.feature_gate = FeatureGate(geo_dim=4, rgb_dim=3, spatial_dim=3)
		
		# Optimized Encoder - reduced channels and removed deepest layer
		self.conv1 = EdgeConv(num_features, 64)
		self.conv1_2 = EdgeConv(64, 64, residual=True)
		self.conv2 = EdgeConv(64, 128)
		self.conv2_2 = EdgeConv(128, 128, residual=True)
		self.conv3 = EdgeConv(128, 256)
		self.conv3_2 = EdgeConv(256, 256, residual=True)
		self.conv4 = EdgeConv(256, 512)  # Changed back from 384 to 512
		# Removed conv4_2 and conv5 for memory efficiency

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
		
		# Apply feature gating at input
		x_gated, gates = self.feature_gate(x)
		x0, pos0, batch0 = x_gated, pos, batch
		
		# Store gates for analysis (keep on GPU)
		self.last_gates = gates.detach()

		# Optimized encoder with fewer layers
		x1 = self.conv1(x0, pos0, batch0)
		x1 = self.conv1_2(x1, pos0, batch0)
		idx1 = fps(pos0, batch0, ratio=0.25)  # More aggressive sampling
		pos1, x1_sampled, batch1 = pos0[idx1], x1[idx1], batch0[idx1]

		x2 = self.conv2(x1_sampled, pos1, batch1)
		x2 = self.conv2_2(x2, pos1, batch1)
		idx2 = fps(pos1, batch1, ratio=0.25)
		pos2, x2_sampled, batch2 = pos1[idx2], x2[idx2], batch1[idx2]

		x3 = self.conv3(x2_sampled, pos2, batch2)
		x3 = self.conv3_2(x3, pos2, batch2)
		idx3 = fps(pos2, batch2, ratio=0.25)
		pos3, x3_sampled, batch3 = pos2[idx3], x3[idx3], batch2[idx3]
		
		# Final encoder layer (bottleneck)
		x4_bottleneck = self.conv4(x3_sampled, pos3, batch3)

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