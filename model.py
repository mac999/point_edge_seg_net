# Title: PointEdgeSegNet (version 0.1)
# Author: taewook kang (laputa9999@gmail.com)
# Date: 2025-09-21
# Purpose: Defines the PointEdgeSegNet model architecture.
# Dependencies: torch, torch_geometric

import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import fps, knn_interpolate
from torch_geometric.nn.pool import knn_graph
from torch_geometric.utils import scatter

class EdgeConv(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(EdgeConv, self).__init__()
		self.mlp = nn.Sequential(
			nn.Linear(2 * in_channels, out_channels), 
			nn.BatchNorm1d(out_channels), 
			nn.ReLU(),
			nn.Linear(out_channels, out_channels), 
			nn.BatchNorm1d(out_channels), 
			nn.ReLU()
		)

	def forward(self, x, pos, batch, k=20):
		edge_index = knn_graph(pos, k=k, batch=batch, loop=False)
		row, col = edge_index
		
		edge_features = torch.cat([x[row], x[col] - x[row]], dim=1)
		out = self.mlp(edge_features)
		
		aggr_out = scatter(out, row, dim=0, dim_size=x.size(0), reduce='max')
		
		return aggr_out

class PointEdgeSegNet(nn.Module):
	def __init__(self, num_features, num_classes):
		super(PointEdgeSegNet, self).__init__()
		
		# Encoder
		self.conv1 = EdgeConv(num_features, 64)
		self.conv2 = EdgeConv(64, 128)
		self.conv3 = EdgeConv(128, 256)
		self.conv4 = EdgeConv(256, 512)

		# Decoder
		self.deconv1_mlp = nn.Sequential(nn.Linear(512 + 256, 256), nn.BatchNorm1d(256), nn.ReLU())
		self.deconv2_mlp = nn.Sequential(nn.Linear(256 + 128, 128), nn.BatchNorm1d(128), nn.ReLU())
		self.deconv3_mlp = nn.Sequential(nn.Linear(128 + 64, 64), nn.BatchNorm1d(64), nn.ReLU())

		# Prediction Head
		self.head = nn.Sequential(
			nn.Linear(64 + num_features, 64), 
			nn.BatchNorm1d(64), 
			nn.ReLU(),
			nn.Dropout(0.5), 
			nn.Linear(64, num_classes)
		)

	def forward(self, data):
		x, pos, batch = data.x, data.pos, data.batch
		
		x0, pos0, batch0 = x, pos, batch

		x1 = self.conv1(x0, pos0, batch0)
		idx1 = fps(pos0, batch0, ratio=0.25)
		pos1, x1_sampled, batch1 = pos0[idx1], x1[idx1], batch0[idx1]

		x2 = self.conv2(x1_sampled, pos1, batch1)
		idx2 = fps(pos1, batch1, ratio=0.25)
		pos2, x2_sampled, batch2 = pos1[idx2], x2[idx2], batch1[idx2]

		x3 = self.conv3(x2_sampled, pos2, batch2)
		idx3 = fps(pos2, batch2, ratio=0.25)
		pos3, x3_sampled, batch3 = pos2[idx3], x3[idx3], batch2[idx3]
		
		x4_bottleneck = self.conv4(x3_sampled, pos3, batch3)

		up_x2 = knn_interpolate(x4_bottleneck, pos3, pos2, batch3, batch2, k=3)
		dec_x2 = self.deconv1_mlp(torch.cat([up_x2, x3], dim=1))

		up_x1 = knn_interpolate(dec_x2, pos2, pos1, batch2, batch1, k=3)
		dec_x1 = self.deconv2_mlp(torch.cat([up_x1, x2], dim=1))
		
		up_x0 = knn_interpolate(dec_x1, pos1, pos0, batch1, batch0, k=3)
		dec_x0 = self.deconv3_mlp(torch.cat([up_x0, x1], dim=1))
		
		final_x = torch.cat([dec_x0, x0], dim=1)
		out = self.head(final_x)
		
		return F.log_softmax(out, dim=-1)