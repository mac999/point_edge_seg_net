import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

class PointCloudAnalyzer:
	def __init__(self, block_dir, area_name=None):
		self.block_dir = block_dir
		self.area_name = area_name
		if not os.path.exists(block_dir):
			raise FileNotFoundError(f"Directory not found: {block_dir}")
		
		# Get all .pt files
		all_pt_files = [f for f in os.listdir(block_dir) if f.endswith('.pt')]
		
		# Filter by area_name if specified
		self.files = [os.path.join(block_dir, f) for f in all_pt_files if area_name in f]
		print(f"Area '{area_name}' filtering applied: {len(self.files)} files found")

		if not self.files:
			print(f"Warning: No .pt files containing '{area_name}' found in {block_dir}")

	def load_blocks(self):
		blocks = []
		for f in self.files:
			try:
				# Set weights_only=False to allow loading point cloud data objects
				data = torch.load(f, weights_only=False)
				blocks.append(data)
			except Exception as e:
				print(f"Failed to load file: {f}, error: {e}")
				continue
		
		if not blocks:
			print("Warning: No blocks loaded.")
		else:
			print(f"Successfully loaded blocks: {len(blocks)}")
		
		return blocks

	def analyze(self):
		blocks = self.load_blocks()
		coords = []
		colors = []
		normal_vectors = []
		geometric_features = []  # verticality, planarity, curvature
		labels = []
		
		for block in blocks:
			if 'pos' in block:
				coords.append(block['pos'].cpu().numpy())
			if 'x' in block:
				# x structure: [normals(3), verticality(1), planarity(1), curvature(1), colors(3)]
				x_data = block['x'].cpu().numpy()
				
				# Normal vectors (0:3)
				normals = x_data[:, 0:3]
				normal_vectors.append(normals)
				
				# Geometric features (3:6) - verticality, planarity, curvature
				geo_feat = x_data[:, 3:6]
				geometric_features.append(geo_feat)
				
				# Colors (6:9) - normalized to 0~1
				color_data = x_data[:, 6:9]
				colors.append(color_data)
				
			if 'y' in block:
				labels.append(block['y'].cpu().numpy())
				
		coords = np.concatenate(coords, axis=0) if coords else None
		colors = np.concatenate(colors, axis=0) if colors else None
		normal_vectors = np.concatenate(normal_vectors, axis=0) if normal_vectors else None
		geometric_features = np.concatenate(geometric_features, axis=0) if geometric_features else None
		labels = np.concatenate(labels, axis=0) if labels else None
		
		return coords, colors, labels, normal_vectors, geometric_features

	def plot_stats(self, coords, colors, labels, normals=None, geo_features=None):
		# Dynamically adjust layout based on the number of existing charts
		available_data = [coords, colors, labels, normals, geo_features]
		chart_count = sum([data is not None for data in available_data])
		
		if chart_count == 0:
			print("No data to display.")
			return
			
		# Adjust figure size and subplot layout based on chart count
		if chart_count <= 2:
			fig_width, subplot_rows, subplot_cols = 12, 1, 2
		elif chart_count <= 4:
			fig_width, subplot_rows, subplot_cols = 15, 2, 2
		else:
			fig_width, subplot_rows, subplot_cols = 18, 2, 3
			
		plt.figure(figsize=(fig_width, 5 * subplot_rows))
		
		chart_idx = 1
		
		if coords is not None:
			plt.subplot(subplot_rows, subplot_cols, chart_idx)
			plt.title('XYZ Histogram')
			for i, axis in enumerate(['X', 'Y', 'Z']):
				plt.hist(coords[:, i], bins=50, alpha=0.5, label=axis)
			plt.legend()
			chart_idx += 1
			
		if colors is not None:
			plt.subplot(subplot_rows, subplot_cols, chart_idx)
			plt.title('RGB Histogram (0~1 normalized)')
			for i, color in enumerate(['R', 'G', 'B']):
				plt.hist(colors[:, i], bins=50, alpha=0.5, label=color, range=(0, 1))
			plt.legend()
			plt.xlim(0, 1)
			chart_idx += 1
			
		if normals is not None:
			plt.subplot(subplot_rows, subplot_cols, chart_idx)
			plt.title('Normal Vectors')
			for i, axis in enumerate(['Nx', 'Ny', 'Nz']):
				plt.hist(normals[:, i], bins=50, alpha=0.5, label=axis, range=(-1, 1))
			plt.legend()
			plt.xlim(-1, 1)
			chart_idx += 1
			
		if geo_features is not None:
			plt.subplot(subplot_rows, subplot_cols, chart_idx)
			plt.title('Geometric Features')
			feature_names = ['Verticality', 'Planarity', 'Curvature']
			for i, feat in enumerate(feature_names):
				plt.hist(geo_features[:, i], bins=50, alpha=0.5, label=feat, range=(0, 1))
			plt.legend()
			plt.xlim(0, 1)
			chart_idx += 1
			
		if labels is not None:
			plt.subplot(subplot_rows, subplot_cols, chart_idx)
			plt.title('Label Distribution')
			label_counts = Counter(labels)
			plt.bar(label_counts.keys(), label_counts.values())
			plt.xlabel('Label')
			plt.ylabel('Count')
			
		plt.tight_layout()
		plt.show()

	def run_full_analysis(self):
		coords, colors, labels, normals, geo_features = self.analyze()
		self.plot_stats(coords, colors, labels, normals, geo_features)
		
		print('--- Point Cloud Analysis Stats ---')
		print(f'Analyzed files: {len(self.files)}')
		
		if coords is not None:
			print('\n[Coordinate Information]')
			print('XYZ mean:', np.mean(coords, axis=0))
			print('XYZ std:', np.std(coords, axis=0))
			print('Total points:', len(coords))
			
		if colors is not None:
			print('\n[Color Information] (0~1 normalized)')
			print('RGB mean:', np.mean(colors, axis=0))
			print('RGB std:', np.std(colors, axis=0))
			
		if normals is not None:
			print('\n[Normal Vectors]')
			print('Normal mean:', np.mean(normals, axis=0))
			print('Normal std:', np.std(normals, axis=0))
			
		if geo_features is not None:
			print('\n[Geometric Features] (Verticality, Planarity, Curvature)')
			print('Geo features mean:', np.mean(geo_features, axis=0))
			print('Geo features std:', np.std(geo_features, axis=0))
			
		if labels is not None:
			print('\n[Label Distribution]')
			label_counts = Counter(labels)
			print('Label counts:', label_counts)
			print('Total labels:', len(set(labels)))

		input("Press Enter to continue...")

if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser(description='Point Cloud Block Analysis')
	parser.add_argument('--block_dir', type=str, default='./block_s3dis', help='Directory containing block .pt files')
	parser.add_argument('--area_name', type=str, default='Area_1', help='Specific area to analyze (e.g., Area_1, Area_5)')
	args = parser.parse_args()
	
	try:
		analyzer = PointCloudAnalyzer(args.block_dir, area_name=args.area_name)
		analyzer.run_full_analysis()
	except FileNotFoundError as e:
		print(f"Error: {e}")
