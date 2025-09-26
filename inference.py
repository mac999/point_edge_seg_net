# Title: inference (version 0.1)
# Author: taewook kang (laputa9999@gmail.com)
# Date: 2025-09-21
# Purpose: Loads a trained model to perform segmentation on a new point cloud file.
# Dependencies: torch, torch_geometric, numpy, open3d

import torch, numpy as np, open3d as o3d, os, argparse
from model import PointEdgeSegNet
from data_preparation import calculate_features_with_open3d
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm

# Configuration Constants
DEFAULT_MODEL_WEIGHTS_PATH = './logs/20250924_053221/best_model.pth'
DEFAULT_TEST_POINT_CLOUD_PATH = './sample/area_6_conferenceRoom_1.txt'
INFERENCE_BLOCK_PATH = './inference_blocks'  # Directory to save block files
BLOCK_SIZE = 8192  # Same block size as training
NUM_FEATURES = 9
NUM_CLASSES = 13
class_colors = np.array([
	[233, 229, 107],[ 95, 156, 196],[179, 116,  81],[241, 149, 131],
	[ 81, 163, 163],[223, 160, 168],[142,  86, 114],[153, 223, 138],
	[149, 149, 241],[107, 229, 233],[233, 107, 229],[107, 233, 107],
	[160, 160, 160],
]) / 255.0

def create_inference_blocks(point_cloud_path, block_output_dir, block_size=8192):
    """Load point cloud, split into blocks, and save to files"""
    print(f"Loading point cloud: {point_cloud_path}")
    points = np.loadtxt(point_cloud_path)
    coords = points[:, :3]
    
    print("Calculating geometric features...")
    geometric_features = calculate_features_with_open3d(coords)
    colors = points[:, 3:6] / 255.0  # RGB color normalization
    features = np.concatenate([geometric_features, colors], axis=1)  # 9-dimensional features
    
    # Create output directory
    os.makedirs(block_output_dir, exist_ok=True)
    
    # Clear existing block files
    for file in os.listdir(block_output_dir):
        if file.startswith('block_') and file.endswith('.pt'):
            os.remove(os.path.join(block_output_dir, file))
    
    num_points = len(coords)
    block_files = []
    
    print(f"Splitting {num_points} points into blocks of {block_size}...")
    
    for start_idx in tqdm(range(0, num_points, block_size), desc="Creating blocks"):
        end_idx = min(start_idx + block_size, num_points)
        current_block_size = end_idx - start_idx
        
        # Extract block
        block_coords = coords[start_idx:end_idx]
        block_features = features[start_idx:end_idx]
        
        # Apply padding if necessary
        if current_block_size < block_size:
            padding_size = block_size - current_block_size
            
            # Feature padding (zeros)
            pad_features = np.zeros((padding_size, features.shape[1]), dtype=features.dtype)
            block_features = np.concatenate([block_features, pad_features], axis=0)
            
            # Position padding (copy last point)
            last_pos = block_coords[-1:].repeat(padding_size, axis=0)
            block_coords = np.concatenate([block_coords, last_pos], axis=0)
        
        # Create Data object with valid mask
        valid_mask = np.ones(block_size, dtype=bool)
        if current_block_size < block_size:
            valid_mask[current_block_size:] = False
        
        data = Data(
            x=torch.tensor(block_features, dtype=torch.float),
            pos=torch.tensor(block_coords, dtype=torch.float),
            valid_mask=torch.tensor(valid_mask, dtype=torch.bool),
            num_valid_points=current_block_size,
            start_idx=start_idx,
            end_idx=end_idx
        )
        
        # Save block file
        block_filename = f"block_{start_idx:06d}_{end_idx:06d}.pt"
        block_filepath = os.path.join(block_output_dir, block_filename)
        torch.save(data, block_filepath)
        block_files.append(block_filepath)
    
    print(f"Created {len(block_files)} block files in {block_output_dir}")
    return block_files, coords

class InferenceBlockDataset(Dataset):
    """Dataset for loading inference block files"""
    def __init__(self, block_files):
        self.block_files = block_files
    
    def __len__(self):
        return len(self.block_files)
    
    def __getitem__(self, idx):
        return torch.load(self.block_files[idx], weights_only=False)

def run_inference_on_blocks(model, block_files, device):
    """Run inference on block files and return predictions"""
    dataset = InferenceBlockDataset(block_files)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    
    all_predictions = []
    all_indices = []
    
    print("Running inference on blocks...")
    model.eval()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference"):
            batch = batch.to(device)
            
            # Run inference
            out = model(batch)
            block_predictions = out.argmax(dim=-1).cpu().numpy()
            
            # Extract valid predictions (excluding padding)
            valid_mask = batch.valid_mask.cpu().numpy()[0]  # Remove batch dimension
            num_valid = batch.num_valid_points[0].item()  # Remove batch dimension
            start_idx = batch.start_idx[0].item()  # Remove batch dimension
            
            # Only take the first num_valid predictions (valid mask might include more)
            valid_predictions = block_predictions[:num_valid]
            
            all_predictions.append(valid_predictions)
            all_indices.append((start_idx, start_idx + num_valid))
    
    # Merge predictions back to original order
    total_points = max(end for _, end in all_indices)
    merged_predictions = np.zeros(total_points, dtype=np.int32)
    
    for predictions, (start_idx, end_idx) in zip(all_predictions, all_indices):
        merged_predictions[start_idx:end_idx] = predictions
    
    return merged_predictions

def parse_arguments():
    parser = argparse.ArgumentParser(description='PointEdgeSegNet Inference')
    
    parser.add_argument('--model_weights', '-m', default=DEFAULT_MODEL_WEIGHTS_PATH,
                       help='Path to model weights (.pth)')
    parser.add_argument('--input_cloud', '-i', default=DEFAULT_TEST_POINT_CLOUD_PATH,
                       help='Path to input point cloud (.txt)')
    parser.add_argument('--block_path', '-b', default=INFERENCE_BLOCK_PATH,
                       help='Directory for inference blocks')
    parser.add_argument('--no_visualization', '--no_vis', action='store_true',
                       help='Skip visualization')
    
    return parser.parse_args()

def main():
    """Main inference function"""
    # Parse command line arguments
    args = parse_arguments()
    
    # Print configuration
    print("PointEdgeSegNet Inference")
    print(f"Model weights: {args.model_weights}")
    print(f"Input point cloud: {args.input_cloud}")
    print(f"Block directory: {args.block_path}")
    print(f"Visualization: {'Disabled' if args.no_visualization else 'Enabled'}")
    
    # Model loading
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = PointEdgeSegNet(num_features=NUM_FEATURES, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(args.model_weights, map_location=device))
    model = model.to(device)
    model.eval()
    print("Model loaded successfully!")
    
    # Step 1: Create inference blocks
    block_files, original_coords = create_inference_blocks(
        args.input_cloud, 
        args.block_path, 
        BLOCK_SIZE
    )
    
    # Step 2: Run inference on blocks
    pred_labels = run_inference_on_blocks(model, block_files, device)
    
    # Load original points for visualization
    print("Loading original point cloud for visualization...")
    original_points = np.loadtxt(args.input_cloud)
    
    # Save results (optional)
    output_file = args.input_cloud.replace('.txt', '_segmented.txt')
    segmented_points = np.column_stack([original_points, pred_labels])
    np.savetxt(output_file, segmented_points, fmt='%.6f %.6f %.6f %d %d %d %d')
    print(f"Segmentation results saved to: {output_file}")
    
    # 3D visualization (if not disabled)
    if not args.no_visualization:
        print("Visualizing results...")
        
        # Use the original coordinates and points for visualization
        predict_pcd = o3d.geometry.PointCloud()
        predict_pcd.points = o3d.utility.Vector3dVector(original_coords)
        pred_colors = class_colors[pred_labels]
        predict_pcd.colors = o3d.utility.Vector3dVector(pred_colors)
        o3d.visualization.draw_geometries([predict_pcd], window_name="Predicted Segmentation (Press Q to close)")

        original_pcd = o3d.geometry.PointCloud()
        original_pcd.points = o3d.utility.Vector3dVector(original_coords)
        original_pcd.colors = o3d.utility.Vector3dVector(original_points[:, 3:6] / 255.0)
        
        o3d.visualization.draw_geometries([original_pcd], window_name="Original Colors (Press Q to close)")
    
    print("Inference completed successfully!")

if __name__ == '__main__':
    main()