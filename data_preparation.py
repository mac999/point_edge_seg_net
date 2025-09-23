# Title: data_preparation (version 0.1)
# Author: taewook kang (laputa9999@gmail.com)
# Date: 2025-09-21
# Purpose: Pre-processes the S3DIS dataset based on the specified Annotation structure.
# Dependencies: numpy, torch, torch_geometric, open3d, tqdm
import os, numpy as np, torch, open3d as o3d, glob, warnings
from torch_geometric.data import Data
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

# Configuration
# S3DIS original dataset path
S3DIS_PATH = './s3dis_v1.2_aligned'
# Preprocessed data storage path
SAVE_PATH = './processed_s3dis'
# Specify areas to process
AREAS_TO_PROCESS = ['Area_1'] 
# Number of points to sample per block
NUM_POINTS_PER_BLOCK = 8192

# Class name and integer label mapping
class_names = [
	'ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door',
	'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter'
]

def calculate_features_with_open3d(points_xyz):
	"""
	Efficiently calculate normal vectors and geometric features using Open3D.
	Open3D uses KD-Tree internally to accelerate neighbor search.
	"""
	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(points_xyz)
	
	# Calculate normal vectors (using k=20 neighbors)
	pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=20))
	
	# Maintain normal direction consistency (towards positive z-axis)
	pcd.orient_normals_consistent_tangent_plane(20)
	normals = np.asarray(pcd.normals)

	# Create KD-Tree for neighbor search
	pcd_tree = o3d.geometry.KDTreeFlann(pcd)
	
	curvatures = np.zeros(len(points_xyz))
	
	# Calculate covariance matrix for each point to estimate curvature
	for i in tqdm(range(len(points_xyz)), desc="Computing curvatures", leave=False):
		[k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[i], 20)
		neighbors = np.asarray(pcd.points)[idx, :]
		
		# Curvature calculation through PCA
		centered_neighbors = neighbors - np.mean(neighbors, axis=0)
		cov_matrix = np.cov(centered_neighbors, rowvar=False)
		eigenvalues, _ = np.linalg.eigh(cov_matrix)
		
		# Define curvature by dividing the smallest eigenvalue by the total sum
		curvatures[i] = eigenvalues[0] / (np.sum(eigenvalues) + 1e-9)

	# Verticality and planarity
	verticality = 1 - np.abs(normals[:, 2]) # 0: horizontal, 1: close to vertical
	planarity = np.zeros(len(points_xyz)) # Initialize to 0 in this example

	# Feature combination: [normals(3), verticality(1), planarity(1), curvature(1)]
	geometric_features = np.concatenate([
		normals,
		verticality[:, np.newaxis],
		planarity[:, np.newaxis],
		curvatures[:, np.newaxis]
	], axis=1)
	
	return geometric_features

def process_area(area_path, save_path):
	print(f"Processing {os.path.basename(area_path)}...")
	area_save_path = os.path.join(save_path, os.path.basename(area_path))
	os.makedirs(area_save_path, exist_ok=True)
	
	room_folders = [d for d in glob.glob(os.path.join(area_path, '*')) if os.path.isdir(d)]
	
	for room_folder in tqdm(room_folders, desc=f'Processing rooms in {os.path.basename(area_path)}'):
		annotation_folder = os.path.join(room_folder, 'Annotations')
		if not os.path.exists(annotation_folder):
			continue

		points_list, labels_list = [], []
		
		annotation_files = glob.glob(os.path.join(annotation_folder, '*.txt'))
		for annotation_file in tqdm(annotation_files, desc=f'Processing annotations in {os.path.basename(room_folder)}', leave=False):
			try:
				# Extract class name from filename (e.g., 'beam_1.txt' -> 'beam')
				class_name = os.path.basename(annotation_file).split('_')[0]
				if class_name not in class_names:
					continue
				
				label = class_names.index(class_name)
				
				obj_data = np.loadtxt(annotation_file, dtype=np.float32)
				# Skip if data is empty or format is incorrect
				if obj_data.ndim != 2 or obj_data.shape[1] != 6:
					continue
				
				points_list.append(obj_data)
				labels = np.full((obj_data.shape[0],), label, dtype=np.int64)
				labels_list.append(labels)
			except Exception as e:
				print(f"Could not process file {annotation_file}: {e}")

		if not points_list:
			continue

		# Combine all points and labels for the room
		points_room = np.concatenate(points_list, axis=0)
		labels_room = np.concatenate(labels_list, axis=0)
		
		coords_room = points_room[:, :3]
		colors_room = points_room[:, 3:6] / 255.0

		# Calculate geometric features using Open3D
		print(f"Computing geometric features for {os.path.basename(room_folder)} ({len(coords_room)} points)...")
		geometric_features = calculate_features_with_open3d(coords_room)
		
		# Final feature vector combination: [normals(3), verticality(1), planarity(1), curvature(1), colors(3)] = 9 dimensions
		features_room = np.concatenate([geometric_features, colors_room], axis=1)

		# Block sampling from entire room point cloud
		num_room_points = len(points_room)
		if num_room_points > 0:
			# Use original data as is
			data = Data(
				x=torch.tensor(features_room, dtype=torch.float),
				pos=torch.tensor(coords_room, dtype=torch.float),
				y=torch.tensor(labels_room, dtype=torch.long)
			)
			
			room_name = os.path.basename(room_folder)
			torch.save(data, os.path.join(area_save_path, f"{room_name}.pt"))

if __name__ == '__main__':
	print(f"Starting data preparation for {len(AREAS_TO_PROCESS)} areas...")
	for area in tqdm(AREAS_TO_PROCESS, desc="Processing areas"):
		area_path = os.path.join(S3DIS_PATH, area)
		process_area(area_path, SAVE_PATH)
	print("\nData preparation finished.")