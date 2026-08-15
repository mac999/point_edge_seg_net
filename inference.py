# Title: inference (version 0.1)
# Author: taewook kang (laputa9999@gmail.com)
# Date: 2025-09-21
# Purpose: Loads a trained model to perform segmentation on a new point cloud file.
# Dependencies: torch, torch_geometric, numpy, open3d

import torch, numpy as np, open3d as o3d, os, argparse, math
from model import PointEdgeSegNet
from data_processing import (
    calculate_features_with_open3d,
    CLASS_NAMES,
    extract_features_from_room_data,
    load_model_config,
    get_class_colors,
    partition_columns_cover,
    merge_block_votes,
    resolve_feature_config,
    make_block_context_extractor,
    append_block_context
)
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm

# Configuration Constants
DEFAULT_MODEL_WEIGHTS_PATH = './logs/20260715_204942/best_model.pth'  # confirmed model (v1.1, OA 86.60 / mIoU 59.99, Area_5)
DEFAULT_TEST_POINT_CLOUD_PATH = './sample/area_6_conferenceRoom_1.txt'
DEFAULT_CONFIG_PATH = 'model_params.json'
INFERENCE_BLOCK_PATH = './inference_blocks'
BLOCK_SIZE = 8192
NUM_FEATURES = 10  # geo(4) + RGB(3) + spatial(3) = 10D

# Column-mode inference MUST match the training window so the model sees the same spatial
# extent / point density it was trained on (train_model.py COLUMN_WINDOW=2.0). The previous
# default (1.5) was a distribution mismatch. Stride < window adds overlap => multiple votes
# per point => smoother boundaries via majority voting.
INFERENCE_COLUMN_WINDOW = 2.0   # == training COLUMN_WINDOW
INFERENCE_COLUMN_STRIDE = 2.0   # == window => no overlap (matches training); dense columns are
                                # tiled to cover ALL points. Set < window for extra multi-view votes.

def create_inference_blocks(point_cloud_path, block_output_dir, block_size=8192, num_features=NUM_FEATURES,
                            feature_config=None):
    """Load point cloud, split into blocks, and save to files"""
    print(f"Loading point cloud: {point_cloud_path}")
    points = np.loadtxt(point_cloud_path)

    spec = feature_config or resolve_feature_config()
    num_features = spec['num_features']
    base_features = num_features - spec.get('context_dim', 0)  # block context is appended per block
    print(f"Calculating {num_features}D features "
          f"(normals={spec['use_normals']}, curv={spec['use_curvature']}, rgb={spec['use_rgb']}, "
          f"spatial={spec['use_spatial']}, block_context={spec.get('use_block_context', False)})...")
    features = extract_features_from_room_data(points, normalize_colors=True, feature_config=spec)
    coords = points[:, spec['xyz_cols']]

    if features.shape[1] != base_features:
        raise ValueError(f"Feature dimension mismatch! Expected {base_features}D base features, got {features.shape[1]}D")

    # Optional per-block buffered-context descriptor (must match training). Note: the
    # sequential chunks of this legacy path are not spatially coherent, so the column
    # voting path (--voting, default) is the recommended pairing for block context.
    ctx_extractor = make_block_context_extractor(coords, features, spec)

    print(f"Feature validation passed: {features.shape[1]}D base features"
          + (f" + {spec['context_dim']}D block context" if ctx_extractor is not None else ""))
    
    # Create output directory
    os.makedirs(block_output_dir, exist_ok=True)
    
    # Clear existing block files
    for file in os.listdir(block_output_dir):
        if file.startswith('block_') and file.endswith('.pt'):
            os.remove(os.path.join(block_output_dir, file))
    
    num_points = len(coords)
    block_files = []
    
    print(f"Splitting {len(coords)} points into blocks of {block_size}...")
    
    for start_idx in tqdm(range(0, num_points, block_size), desc="Creating blocks"):
        end_idx = min(start_idx + block_size, num_points)
        current_block_size = end_idx - start_idx
        
        # Extract block
        block_coords = coords[start_idx:end_idx]
        block_features = features[start_idx:end_idx]
        if ctx_extractor is not None:
            block_features = append_block_context(block_features, ctx_extractor,
                                                  np.arange(start_idx, end_idx))

        # Apply padding if necessary
        if current_block_size < block_size:
            padding_size = block_size - current_block_size

            # Feature padding (zeros)
            pad_features = np.zeros((padding_size, block_features.shape[1]), dtype=block_features.dtype)
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

def create_inference_blocks_columns(point_cloud_path, block_output_dir, block_size=8192,
                                    window=1.5, stride=0.75, num_features=NUM_FEATURES,
                                    feature_config=None):
    """
    Inference blocking that MATCHES training's column mode (--block_mode column).

    The default create_inference_blocks() slices the file into arbitrary sequential
    chunks - a distribution the model never saw in training (which uses spatially
    coherent blocks), so EdgeConv's k-NN graph and FPS behave differently at test
    time. Here we build the SAME overlapping full-height columns used for training and
    remember each block's original point indices so predictions can be voted together.
    """
    print(f"Loading point cloud: {point_cloud_path}")
    points = np.loadtxt(point_cloud_path)
    spec = feature_config or resolve_feature_config()
    num_features = spec['num_features']
    base_features = num_features - spec.get('context_dim', 0)  # block context is appended per block
    features = extract_features_from_room_data(points, normalize_colors=True, feature_config=spec)
    coords = points[:, spec['xyz_cols']]
    if features.shape[1] != base_features:
        raise ValueError(f"Feature dimension mismatch! Expected {base_features}D base features, got {features.shape[1]}D")

    # Optional per-block buffered-context descriptor; MUST match how the model was
    # trained (features.use_block_context / context_buffer / context_bins in the config).
    ctx_extractor = make_block_context_extractor(coords, features, spec)
    if ctx_extractor is not None:
        print(f"Block-context enabled: +{spec['context_dim']}D per block "
              f"(buffer={spec['context_buffer']}m, bins={spec['context_bins']})")

    os.makedirs(block_output_dir, exist_ok=True)
    for file in os.listdir(block_output_dir):
        if file.startswith('block_') and file.endswith('.pt'):
            os.remove(os.path.join(block_output_dir, file))

    # Coverage-guaranteeing blocker: every point is predicted at least once (dense columns
    # are tiled into multiple block_size blocks). partition_columns() would drop most points
    # of a dense column, which merge_block_votes then mislabels as class 0 (ceiling).
    blocks = partition_columns_cover(coords, block_size=block_size, window=window, stride=stride, seed=0)
    print(f"Split {len(coords)} points into {len(blocks)} column blocks (full coverage)...")

    block_files = []
    for bi, (idx, num_real) in enumerate(tqdm(blocks, desc="Creating column blocks")):
        block_feats = features[idx]
        if ctx_extractor is not None:
            block_feats = append_block_context(block_feats, ctx_extractor, idx[:num_real])
        data = Data(
            x=torch.tensor(block_feats, dtype=torch.float),
            pos=torch.tensor(coords[idx], dtype=torch.float),
            valid_mask=torch.tensor(np.arange(block_size) < num_real, dtype=torch.bool),
            num_valid_points=num_real,
            point_indices=torch.tensor(idx, dtype=torch.long),
        )
        fp = os.path.join(block_output_dir, f"block_{bi:06d}.pt")
        torch.save(data, fp)
        block_files.append(fp)
    return block_files, coords, len(coords)

def run_inference_with_voting(models, block_files, device, total_points, num_classes,
                              coords=None, tta_rotations=(0.0,)):
    """Run inference on column blocks and majority-vote per point.

    Supports two multi-view boosts that cost no extra VRAM (blocks stay batch_size=1):
      - Model ensemble: `models` may be a list; per block their softmax probabilities are
        averaged before argmax (complementary models cancel each other's errors).
      - Test-time augmentation (TTA): each block is additionally predicted under Z-axis
        rotations in `tta_rotations` (degrees). Coordinates AND the normal channels (x[:,0:3])
        are rotated together so geometry stays consistent; every rotated prediction casts a
        vote. Labels are rotation-invariant, so votes accumulate correctly.

    coords (full-cloud XYZ) is passed to merge_block_votes so any residual uncovered point is
    filled from its nearest voted neighbour rather than silently labelled class 0.
    """
    if not isinstance(models, (list, tuple)):
        models = [models]
    for m in models:
        m.eval()
    dataset = InferenceBlockDataset(block_files)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    per_block = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference (voting)"):
            batch = batch.to(device)
            num_valid = batch.num_valid_points[0].item()
            idx = batch.point_indices.cpu().numpy()[:num_valid]
            base_pos = batch.pos.clone()
            base_nrm = batch.x[:, 0:3].clone()  # normals rotate rigidly with the geometry
            for deg in tta_rotations:
                if deg % 360 != 0.0:
                    a = math.radians(deg)
                    c, s = math.cos(a), math.sin(a)
                    Rz_t = torch.tensor([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]],
                                        device=device, dtype=base_pos.dtype)  # == Rz^T for row vectors
                    batch.pos = base_pos @ Rz_t
                    batch.x[:, 0:3] = base_nrm @ Rz_t
                else:
                    batch.pos = base_pos
                    batch.x[:, 0:3] = base_nrm
                # Ensemble: average softmax over models
                probs = None
                for m in models:
                    p = torch.softmax(m(batch), dim=-1)
                    probs = p if probs is None else probs + p
                preds = probs.argmax(dim=-1).cpu().numpy()
                per_block.append((preds[:num_valid], idx))
    final_labels, vote_counts = merge_block_votes(total_points, num_classes, per_block, coords=coords)
    return final_labels

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
    
    parser.add_argument('--config', '-c', default=DEFAULT_CONFIG_PATH,
                       help='Path to model configuration JSON file')
    parser.add_argument('--model_weights', '-m', default=DEFAULT_MODEL_WEIGHTS_PATH,
                       help='Path to model weights (.pth)')
    parser.add_argument('--input_cloud', '-i', default=DEFAULT_TEST_POINT_CLOUD_PATH,
                       help='Path to input point cloud (.txt)')
    parser.add_argument('--block_path', '-b', default=INFERENCE_BLOCK_PATH,
                       help='Directory for inference blocks')
    parser.add_argument('--no_visualization', '--no_vis', action='store_true',
                       help='Skip visualization')
    # Voting is ON by default: the shipped model is trained with --block_mode column, so
    # overlapping full-height column blocks + majority voting match the training-time point
    # distribution. --no_voting falls back to legacy sequential chunks (not recommended).
    parser.add_argument('--voting', dest='voting', action='store_true', default=True,
                       help='Use overlapping column blocks + majority voting (default; matches column training)')
    parser.add_argument('--no_voting', dest='voting', action='store_false',
                       help='Disable voting; use legacy sequential chunks (NOT recommended for column-trained models)')
    parser.add_argument('--column_window', type=float, default=INFERENCE_COLUMN_WINDOW,
                       help='Column-mode window (m). Should match training COLUMN_WINDOW (2.0).')
    parser.add_argument('--column_stride', type=float, default=INFERENCE_COLUMN_STRIDE,
                       help='Column-mode stride (m). < window => overlap => more votes per point.')
    parser.add_argument('--tta', action='store_true',
                       help='Test-time augmentation: also predict under Z-rotations 90/180/270 and vote (no extra VRAM).')
    parser.add_argument('--ensemble', nargs='*', default=None, metavar='WEIGHTS.pth',
                       help='Extra model weight paths to ensemble (softmax-averaged) with --model_weights.')
    # Block-context overrides. MUST match how --model_weights was trained (same
    # on/off, buffer, bins); normally leave unset and let the config decide.
    parser.add_argument('--block_context', dest='block_context', action='store_true', default=None,
                       help='Force-enable the per-block buffered-context descriptor (overrides config).')
    parser.add_argument('--no_block_context', dest='block_context', action='store_false',
                       help='Force-disable the per-block buffered-context descriptor (overrides config).')
    parser.add_argument('--context_buffer', type=float, default=None,
                       help='Block-context buffer distance (m); must match training.')
    parser.add_argument('--context_mode', type=str, default='bottleneck', choices=['input', 'bottleneck'],
                        help="MUST match training: where the block-context descriptor enters the model. "
                             "'bottleneck' (new default) or 'input' (legacy, e.g. logs/20260722_104628).")
    parser.add_argument('--width_mult', type=float, default=1.0,
                        help='MUST match training --width_mult (1.0 = historical architecture).')
    parser.add_argument('--mid_transformer', action='store_true',
                        help='MUST match training --mid_transformer.')
    parser.add_argument('--context_bins', type=int, default=None,
                       help='Block-context z-histogram bins; must match training.')

    return parser.parse_args()

def save_segmented_las(output_path, coords, pred_labels, class_colors):
    """Save the segmented cloud as a LAS file for immediate viewing (CloudCompare, etc.).

    - XYZ from the original coordinates
    - RGB painted with each point's predicted-class color, so the segmentation is visible
      the moment the file is opened
    - the integer class id is also stored in the standard LAS 'classification' field, so
      downstream tools can filter/relabel by class without needing the color LUT
    """
    import laspy
    coords = np.asarray(coords, dtype=np.float64)
    pred_labels = np.asarray(pred_labels)

    header = laspy.LasHeader(point_format=3, version="1.2")  # fmt 3 = XYZ + RGB
    header.offsets = coords.min(axis=0)
    header.scales = np.array([0.001, 0.001, 0.001])          # 1 mm precision
    las = laspy.LasData(header)

    las.x, las.y, las.z = coords[:, 0], coords[:, 1], coords[:, 2]
    # class_colors is 0-1 float; LAS color channels are 16-bit (0-65535)
    col16 = (class_colors[pred_labels] * 65535.0).astype(np.uint16)
    las.red, las.green, las.blue = col16[:, 0], col16[:, 1], col16[:, 2]
    las.classification = pred_labels.astype(np.uint8)        # 0-31 range; 13 classes fit
    las.write(output_path)
    return output_path

def main():
    """Main inference function"""
    # Parse command line arguments
    args = parse_arguments()
    
    # Load model configuration
    print(f"Loading configuration from: {args.config}")
    config = load_model_config(args.config)
    NUM_CLASSES = config['num_classes']
    class_colors = get_class_colors(as_numpy=True) / 255.0

    # Apply block-context CLI overrides before resolving the spec (mirrors train_model.py).
    feats_cfg = config.get('features') or {}
    config['features'] = feats_cfg
    if args.block_context is not None:
        feats_cfg['use_block_context'] = args.block_context
    if args.context_buffer is not None:
        feats_cfg['context_buffer'] = args.context_buffer
    if args.context_bins is not None:
        feats_cfg['context_bins'] = args.context_bins

    # Resolve the domain-agnostic feature spec (must match how the model was trained).
    spec = resolve_feature_config(config)
    num_features = spec['num_features']
    feature_dims = (spec['geo_dim'], spec['rgb_dim'], spec['spatial_dim'], spec['context_dim'])

    print(f"Dataset: {config.get('dataset_name', 'Custom')}")
    print(f"Number of classes: {NUM_CLASSES}")
    print(f"Features: {num_features}D {feature_dims} (normals={spec['use_normals']}, curv={spec['use_curvature']}, "
          f"rgb={spec['use_rgb']}, spatial={spec['use_spatial']}, block_context={spec['use_block_context']})")
    
    # Print configuration
    print("PointEdgeSegNet Inference")
    print(f"Model weights: {args.model_weights}")
    print(f"Input point cloud: {args.input_cloud}")
    print(f"Block directory: {args.block_path}")
    print(f"Visualization: {'Disabled' if args.no_visualization else 'Enabled'}")
    
    # Model loading
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    def _load(wpath):
        m = PointEdgeSegNet(num_features=num_features, num_classes=NUM_CLASSES, feature_dims=feature_dims,
                            context_mode=args.context_mode, width_mult=args.width_mult,
                            mid_transformer=args.mid_transformer)
        try:
            m.load_state_dict(torch.load(wpath, map_location=device, weights_only=False))
        except RuntimeError as e:
            raise RuntimeError(
                f"Checkpoint/architecture mismatch for '{wpath}'. The constructor flags must match "
                f"training: check --context_mode (input|bottleneck), --width_mult and --mid_transformer. "
                f"Original error: {e}") from e
        return m.to(device).eval()

    models = [_load(args.model_weights)]
    for wpath in (args.ensemble or []):
        models.append(_load(wpath))
        print(f"Ensemble model added: {wpath}")
    model = models[0]  # primary (used by the non-voting path)
    print(f"Model(s) loaded successfully! ({len(models)} model{'s' if len(models) > 1 else ''})")
    tta_rotations = (0.0, 90.0, 180.0, 270.0) if args.tta else (0.0,)
    if args.tta:
        print("TTA enabled: Z-rotations 0/90/180/270")

    # Step 1 + 2: Create inference blocks and run inference
    if args.voting:
        print(f"Column voting: window={args.column_window}m, stride={args.column_stride}m "
              f"(overlap={'yes' if args.column_stride < args.column_window else 'no'})")
        block_files, original_coords, total_points = create_inference_blocks_columns(
            args.input_cloud, args.block_path, BLOCK_SIZE,
            window=args.column_window, stride=args.column_stride, feature_config=spec
        )
        pred_labels = run_inference_with_voting(models, block_files, device, total_points, NUM_CLASSES,
                                                coords=original_coords, tta_rotations=tta_rotations)
    else:
        block_files, original_coords = create_inference_blocks(
            args.input_cloud,
            args.block_path,
            BLOCK_SIZE,
            feature_config=spec
        )
        pred_labels = run_inference_on_blocks(model, block_files, device)
    
    # Load original points for visualization
    print("Loading original point cloud for visualization...")
    original_points = np.loadtxt(args.input_cloud)
    
    # Save results (optional). Build the column format from the actual input width so
    # colorless (XYZ) or attribute-carrying clouds are written correctly, not just XYZRGB.
    output_file = args.input_cloud.replace('.txt', '_segmented.txt')
    if original_points.ndim == 1:
        original_points = original_points.reshape(1, -1)
    segmented_points = np.column_stack([original_points, pred_labels])
    n_in = original_points.shape[1]
    rgb_cols = spec.get('rgb_cols') or []
    # integer format for color/label columns, float for coordinates/attributes
    int_cols = set(rgb_cols) | {n_in}  # rgb columns + the appended label column
    fmt = ' '.join('%d' if c in int_cols else '%.6f' for c in range(n_in + 1))
    np.savetxt(output_file, segmented_points, fmt=fmt)
    print(f"Segmentation results saved to: {output_file}")

    # Also save a colored LAS for immediate viewing (RGB = predicted-class color, and the
    # class id in the standard 'classification' field). This is the file you open directly.
    las_output_file = args.input_cloud.replace('.txt', '_segmented.las')
    try:
        save_segmented_las(las_output_file, original_coords, pred_labels, class_colors)
        print(f"Segmentation LAS saved to: {las_output_file}")
    except Exception as e:
        print(f"Warning: could not write LAS ({e}); the .txt result above is still available.")
    
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
        rgb_cols = spec.get('rgb_cols')
        if rgb_cols is not None and original_points.shape[1] > max(rgb_cols):
            original_pcd.colors = o3d.utility.Vector3dVector(original_points[:, rgb_cols] / spec.get('rgb_max', 255.0))
        else:
            original_pcd.paint_uniform_color([0.5, 0.5, 0.5])

        o3d.visualization.draw_geometries([original_pcd], window_name="Original Colors (Press Q to close)")
    
    print("Inference completed successfully!")

if __name__ == '__main__':
    main()