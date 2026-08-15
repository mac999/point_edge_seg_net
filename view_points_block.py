import os, torch, torch.optim as optim, json, csv, argparse, time, numpy as np, random
import open3d as o3d
from glob import glob
from tqdm import tqdm
from data_processing import hsv_to_rgb  # Import HSV to RGB conversion

BLOCK_DATA_PATH = './block_s3dis'  # Block data storage path

print("Loading and visualizing block files...")
sample_blocks = glob(os.path.join(BLOCK_DATA_PATH, '*.pt'))[:5]

for block_file in tqdm(sample_blocks, desc="Processing blocks"):
    try:
        data = torch.load(block_file, weights_only=False)
        points = data.pos.numpy()
        
        feature_dim = data.x.shape[1]
        print(f"Block: {os.path.basename(block_file)}, Feature dimensions: {feature_dim}")
        
        if feature_dim >= 12:  # 12D format: [normals(3) + curvature(1) + RGB(3) + HSV(2) + spatial(3)]
            colors = data.x[:, 4:7].numpy()  # RGB at indices 4:7
        elif feature_dim >= 9:  # 9D format: [normals(3) + curvature(1) + RGB(3) + HSV(2)]
            colors = data.x[:, 4:7].numpy()  # RGB at indices 4:7
        elif feature_dim >= 7:  # 7D format: [normals(3) + curvature(1) + RGB(3)]
            colors = data.x[:, 4:7].numpy()  # RGB at indices 4:7
        elif feature_dim >= 6:  # 6D format: [normals(3) + curvature(1) + HSV(2)]
            hsv_features = data.x[:, 4:6].numpy()  # HSV at indices 4:6
            # Add default value component for HSV->RGB conversion
            hsv_full = np.column_stack([hsv_features, np.full(len(hsv_features), 0.8)])
            colors = hsv_to_rgb(hsv_full)
        else:  # Fallback for older formats
            colors = data.x[:, -3:].numpy()
        
        colors = np.clip(colors, 0.0, 1.0)
        
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Save as PLY file for visualization
        ply_save_path = os.path.join(BLOCK_DATA_PATH, f"{os.path.basename(block_file)}_visualization.ply")
        o3d.io.write_point_cloud(ply_save_path, pcd)
        print(f"  Saved visualization to {ply_save_path}")
        
        # Print color statistics for verification
        print(f"  Color statistics: RGB mean={colors.mean(axis=0)}, RGB std={colors.std(axis=0)}")
        
    except Exception as e:
        print(f"Error processing {block_file}: {e}")

print(f"\nProcessed {len(sample_blocks)} block files")
print("Check the generated .ply files for color visualization quality")

