import os, torch, torch.optim as optim, json, csv, argparse, time, numpy as np, random
import open3d as o3d
from glob import glob
from tqdm import tqdm

BLOCK_DATA_PATH = './block_s3dis'  # Block data storage path

sample_blocks = glob(os.path.join(BLOCK_DATA_PATH, '*.pt'))[:5]
for block_file in sample_blocks:
    data = torch.load(block_file, weights_only=False)
    points = data.pos.numpy()
    colors = data.x[:, -3:].numpy()  # Assuming last 3 features are RGB colors
    # colors = np.clip(colors, 0.0, 1.0)
    # colors = (colors * 255).astype(np.float32)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    ply_save_path = os.path.join(BLOCK_DATA_PATH, f"{os.path.basename(block_file)}.ply")
    o3d.io.write_point_cloud(ply_save_path, pcd)
    print(f"Saved 3D model to {ply_save_path}")

