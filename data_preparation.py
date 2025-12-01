# Title: data_preparation (version 0.1)
# Author: taewook kang (laputa9999@gmail.com)
# Date: 2025-09-21
# Purpose: Pre-processes the S3DIS dataset based on the specified Annotation structure.
# Dependencies: numpy, torch, torch_geometric, open3d, tqdm
import os, numpy as np, torch, open3d as o3d, glob, warnings, argparse, shutil
from torch_geometric.data import Data
from tqdm import tqdm
from data_processing import (
    extract_features_from_room_data, 
    get_class_label, 
    CLASS_NAMES,
    create_point_cloud_visualization,
    convert_numpy_to_torch,
    validate_point_cloud_data
)

warnings.filterwarnings("ignore", category=UserWarning)

# Configuration
S3DIS_PATH = 'G:/data/Stanford3dDataset_v1.2_Aligned_Version' # S3DIS original dataset path
SAVE_PATH = './processed_s3dis' 	# Preprocessed data storage path
AREAS_TO_PROCESS = ['Area_1', 'Area_2', 'Area_3', 'Area_4', 'Area_5', 'Area_6'] 		# Specify areas to process
NUM_POINTS_PER_BLOCK = 8192 		# Number of points to sample per block

# Use shared class names from data_processing module
class_names = CLASS_NAMES

def process_area(area_path, args):
	save_path = args.save_path
	print(f"Processing {os.path.basename(area_path)}...")
	area_save_path = os.path.join(save_path, os.path.basename(area_path))
	if os.path.exists(area_save_path):
		shutil.rmtree(area_save_path, ignore_errors=True)
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
				
				# Use shared function for label lookup
				label = get_class_label(class_name)
				if label is None:
					continue
				
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

		# Use shared feature extraction function
		print(f"Computing features for {os.path.basename(room_folder)} ({len(coords_room)} points)...")
		
		features_room = extract_features_from_room_data(
			points_room, 
			normalize_colors=True
		)
		
		# Validate data consistency
		if not validate_point_cloud_data(coords_room, features_room, labels_room):
			print(f"Warning: Invalid data in {os.path.basename(room_folder)}, skipping...")
			continue

		# Block sampling from entire room point cloud
		num_room_points = len(points_room)
		if num_room_points > 0:
			# Use original data as is
			data = Data(
				x=convert_numpy_to_torch(features_room, torch.float),
				pos=convert_numpy_to_torch(coords_room, torch.float),
				y=convert_numpy_to_torch(labels_room, torch.long)
			)
			
			room_name = os.path.basename(room_folder)
			torch.save(data, os.path.join(area_save_path, f"{room_name}.pt"))

		if args.visualize:  # Visualization for verification with normal vectors
			# Keyboard controls. H=help, Q=quit, F=fullscreen, N=toggle normals, W=toggle wireframe, R=reset view, +-=normal vector size, []=view projection scale, L=turn on/off lighting, 0..9=change color
			pcd = create_point_cloud_visualization(
				coords_room, colors_room, features_room[:, :3], show_normals=True
			)
			o3d.visualization.draw_geometries([pcd], point_show_normal=True, window_name=os.path.basename(room_folder))
			print("Visualization done.")

		if args.save_3d_model:  # Save as ply for 3D model verification
			pcd = create_point_cloud_visualization(coords_room, colors_room)
			ply_save_path = os.path.join(area_save_path, f"{os.path.basename(room_folder)}.ply")
			o3d.io.write_point_cloud(ply_save_path, pcd)
			print(f"Saved 3D model to {ply_save_path}")

def main():
	parser = argparse.ArgumentParser(description='S3DIS Dataset Preprocessing')
	parser.add_argument('--s3dis_path', default=S3DIS_PATH, help='S3DIS dataset path')
	parser.add_argument('--save_path', default=SAVE_PATH, help='Output directory')
	parser.add_argument('--areas', nargs='+', default=AREAS_TO_PROCESS, help='Areas to process')
	parser.add_argument('--num_points', type=int, default=NUM_POINTS_PER_BLOCK, help='Number of points per block')
	parser.add_argument('--visualize', type=bool, default=False, help='Visualize point clouds for verification')
	parser.add_argument('--save_3d_model', type=bool, default=True, help='Output flag to the ply files')
	args = parser.parse_args()
	
	print(f"Starting data preparation for {len(args.areas)} areas...")
	for area in tqdm(args.areas, desc="Processing areas"):
		area_path = os.path.join(args.s3dis_path, area)
		process_area(area_path, args)
	print("\nData preparation finished.")

if __name__ == '__main__':
	main()