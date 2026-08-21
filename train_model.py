# Title: train_model (version 0.2)
# Author: taewook kang (laputa9999@gmail.com)
# Date: 2025-09-21
# Purpose: Trains and validates the PointEdgeSegNet model.
# Dependencies: torch, torch_geometric, matplotlib, scikit-learn, tqdm

import os, torch, torch.optim as optim, json, csv, argparse, time, numpy as np, random
import torch.nn as nn, torch.nn.functional as F, matplotlib.pyplot as plt, gc
from torch_geometric.loader import DataLoader
from glob import glob
from models import get_arch, resolve_arch
from models.edgeconv import PointEdgeSegNet
from models.stencil import PointEdgeSegNet as StencilSegNet


def build_model(num_features, num_classes, feature_dims, context_mode, width_mult,
				mid_transformer, sampler, enc_channels, bottleneck_dim, arch='v1'):
	"""Construct the requested architecture. 'v1' is the historical EdgeConv+kNN model
	(every existing checkpoint); 'v2' is the serialized meta-aggregation rework
	(model_v2.py) — measured 15-55x faster and 6-7x lighter at identical channel plan.
	The two are NOT weight-compatible; a checkpoint must be evaluated with the arch
	that trained it."""
	if resolve_arch(arch) == 'stencil':
		return StencilSegNet(num_features=num_features, num_classes=num_classes,
								 feature_dims=feature_dims, enc_channels=enc_channels or (64, 192, 320, 448),
								 bottleneck_dim=bottleneck_dim, knn=V2_KNN, curves=V2_CURVES,
								 neighbor_mode=V2_NEIGHBORS, stencil_radius=V2_STENCIL,
								 feature_diff=V2_DIFF, base_grid=V2_BASE_GRID,
								 pool_grids=tuple(float(g) for g in V2_POOL_GRIDS.split(',')),
								 directional=V2_DIRECTIONAL,
								 stencil_z=V2_STENCIL_Z or None)
	return PointEdgeSegNet(num_features=num_features, num_classes=num_classes, feature_dims=feature_dims,
						   context_mode=context_mode, width_mult=width_mult,
						   mid_transformer=mid_transformer, sampler=sampler,
						   enc_channels=enc_channels, bottleneck_dim=bottleneck_dim)
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from datetime import datetime
from diagnose_kpi_grad import monitor_kpi
from data_processing import (
    apply_torch_color_augmentation,
    apply_torch_enhanced_color_augmentation,
    augment_training_block,
    partition_columns,
    spatial_split_is_val,
    resolve_feature_config,
    make_block_context_extractor,
    append_block_context,
    CLASS_NAMES,
    extract_features_from_room_data,
    load_model_config,
    get_model_config,
    get_class_colors,
    get_class_weights
)
from torch.cuda.amp import autocast, GradScaler

# wandb is optional. Training must not hard-fail when it is not installed or the
# user is offline / not logged in. Gated by USE_WANDB (set from --no_wandb in main()).
try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    _WANDB_AVAILABLE = False
USE_WANDB = False  # resolved in main(): _WANDB_AVAILABLE and not args.no_wandb

# Monotonic wandb step counter, shared by batch- and epoch-level logs. Previously batch
# logs used step=epoch*len(loader)+batch_idx while epoch logs used a separate global_step;
# the two step sequences interleaved out of order so wandb silently dropped the epoch
# points ("Tried to log to step X that is less than current step Y"). A single strictly
# increasing counter guarantees valid steps for every wandb.log() call.
WANDB_STEP = 0

# .env loading is optional (only needed for wandb API key)
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join('logs', '.env'))
except ImportError:
    pass

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# GPU optimization settings (optional)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
	print(f"GPU available: {torch.cuda.get_device_name(0)}")
	print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
	# Stable memory settings for safety
	torch.backends.cudnn.deterministic = True  # Ensure reproducible results
	torch.backends.cudnn.benchmark = False     # Disable for memory stability
	torch.cuda.empty_cache()  # Clear GPU memory
	print("GPU safety settings applied: deterministic=True, benchmark=False")
else:
	print("CUDA not available, using CPU")

# Configuration (optimized for faster convergence)
PROCESSED_DATA_PATH = './processed_s3dis'
BLOCK_DATA_PATH = './block_s3dis'  # Block data storage path
TRAIN_AREAS = ['Area_1', 'Area_2', 'Area_3', 'Area_4', 'Area_6']  # Fixed: proper training areas
TEST_AREA = 'Area_5'  # Fixed: proper test area
NUM_EPOCHS = 60  # Extended for full convergence
BATCH_SIZE = 10  # Lowered 14->10 to offset the larger EdgeConv k (20->32) VRAM; effective batch kept at 60 via ACCUMULATION_STEPS=6
VAL_BATCH_SIZE = 18
LEARNING_RATE = 0.003  # Increased from 0.002 for faster learning
NUM_FEATURES = 10  # Training: normals(3) + curvature(1) + RGB(3) + spatial(3)
NUM_CLASSES = 13  # Default for S3DIS, will be overridden by config in main()
BLOCK_SIZE = 8192  # Number of points per block. TBD

# Settings for validation loss stabilization
GRADIENT_CLIP_VALUE = 8.0  # Gradient clipping (increased for better learning)
WARMUP_EPOCHS = 3  # Reduced for faster ramp-up

# Cosine-annealing floor. Named (was inline in the scheduler) because early stopping now
# has to know when the schedule is effectively done -- see EARLY_STOP_PATIENCE below.
LR_ETA_MIN = 5e-4

# Early-stopping patience, in epochs without a val_acc improvement.
#
# CAUTION: patience interacts with --num_epochs, because the LR schedule is
# CosineAnnealingLR(T_max=NUM_EPOCHS): raising --num_epochs stretches the cosine but does
# NOT make val_acc any less noisy, so a fixed patience makes long schedules structurally
# unable to finish. Measured (logs/20260721_233224): --num_epochs 100 stopped at epoch 72
# with LR still 1.05e-3 -- 2x LR_ETA_MIN -- so the low-LR refinement phase never ran, and
# that run scored below a 60-epoch run that annealed all the way down (mIoU 52.10 vs
# 53.44). EARLY_STOP_REQUIRE_ANNEAL below closes that trap.
EARLY_STOP_PATIENCE = 10

# Refuse to early-stop while the cosine still has a meaningful annealing phase left, i.e.
# until the LR has come within EARLY_STOP_LR_TOL x LR_ETA_MIN. Patience then only decides
# how much of the *tail* to skip, never how much of the anneal to skip. Set
# EARLY_STOP_REQUIRE_ANNEAL = False for the old (pure-patience) behaviour.
EARLY_STOP_REQUIRE_ANNEAL = True
EARLY_STOP_LR_TOL = 1.25

# GPU safety settings
GPU_MEMORY_THRESHOLD = 0.85  # 85% GPU memory usage threshold
MAX_GRADIENT_NORM = 20.0     # Maximum gradient norm threshold (increased)

# Gradient Accumulation settings (virtual batch size increase)
ACCUMULATION_STEPS = 6  # Effective batch size: 10 × 6 = 60

# Thermal cool-down for laptops/small GPUs. 0.0 disables all the throughput-killing
# time.sleep() pauses that were previously hard-coded (10s every 200 batches, etc.).
# Set >0 (e.g. 5.0) only if you actually need to protect a thermally-limited laptop.
COOLDOWN_SEC = 0.0

# Focal Loss focusing parameter. Previously train used 1.5 while test used 2.5, which
# made the reported test loss incomparable to the training/val loss. Now shared.
FOCAL_GAMMA = 2.0

# Probability of dropping (zeroing) RGB per training block. >0 trains a model that
# still works when the input point cloud has no color (the configurable-feature goal).
# NOTE: this doubles as the standard "colour drop" augmentation -- every SOTA recipe
# surveyed uses it (DeLA p=0.2, KPConvX p=0.2) because it forces geometric reasoning.
RGB_DROPOUT_PROB = 0.0

# --- Augmentation preset -------------------------------------------------------------
# 'legacy' reproduces the historical (measured-weak) setup; 'strong' follows the recipes
# that every S3DIS SOTA model uses. Measured motivation: the legacy schedule DECAYS to
# prob 0.1 / strength 0.2 for the last half of training, giving a mean (prob x strength)
# of 0.046 over 60 epochs -- roughly an order of magnitude below standard practice --
# while the val->test gap sat at a constant 8.4 points across every architecture we tried
# (baseline, +context, width 1.5), i.e. generalization, not capacity, is the binding
# constraint. 'strong' stops the decay and adds full 360 deg yaw, isotropic scaling,
# mirroring, small tilts, colour drop and chromatic auto-contrast.
AUG_PRESET = 'legacy'
AUG_FULL_ROTATE = False
AUG_ISOTROPIC_SCALE = False
AUG_FLIP_PROB = 0.0
AUG_TILT_STD = 0.0
AUG_AUTOCONTRAST_PROB = 0.0
AUG_SCALE_RANGE = (0.9, 1.1)

def apply_aug_preset(preset):
	"""Set the module-level augmentation globals for the chosen preset."""
	global AUG_PRESET, AUG_FULL_ROTATE, AUG_ISOTROPIC_SCALE, AUG_FLIP_PROB
	global AUG_TILT_STD, AUG_AUTOCONTRAST_PROB, AUG_SCALE_RANGE, RGB_DROPOUT_PROB
	AUG_PRESET = preset
	if preset == 'strong':
		AUG_FULL_ROTATE = True             # DeLA: uniform(0, 2pi); Pointcept: [-1,1]*pi
		AUG_ISOTROPIC_SCALE = True         # DeLA: single factor, preserves proportions
		AUG_SCALE_RANGE = (0.8, 1.2)       # DeLA range (legacy was 0.9-1.1)
		AUG_FLIP_PROB = 0.5                # Pointcept RandomFlip(p=0.5)
		AUG_TILT_STD = np.pi / 64          # Pointcept x/y rotate +-pi/64
		AUG_AUTOCONTRAST_PROB = 0.2        # Pointcept/DeLA/KPConvX p=0.2
		if RGB_DROPOUT_PROB == 0.0:
			RGB_DROPOUT_PROB = 0.2         # colour drop (DeLA/KPConvX p=0.2)
	return preset

# Rare-class block oversampling exponent (0 = off, uniform sampling as before).
# Motivation (measured, logs/20260721_132243 analysis): sofa appears in only 61/1400
# training blocks (4.4%), column in 12.4%, board in 12.9% -- their gradient signal is
# starved regardless of loss weighting. When > 0, each training block is drawn by a
# WeightedRandomSampler with weight max(1, max(class_weights[c]))**power over the classes
# holding >= 1% of the block's valid points, so e.g. sofa blocks (weight 2.87) are drawn
# ~2.9x more often at power 1.0. Epoch length is unchanged (num_samples = len(dataset)).
OVERSAMPLE_RARE = 0.0

# Where the block-context descriptor enters the model when --block_context is on.
# 'bottleneck' (default): zero-init residual at the bottleneck -- worst case it is ignored.
# 'input': legacy constant-channel injection; measured to LOSE 6.4 mIoU (logs/20260722_104628).
CONTEXT_MODE = 'bottleneck'

# Architecture scaling (defaults reproduce the historical checkpoints exactly).
WIDTH_MULT = 1.0        # channel multiplier for encoder/decoder (needs VRAM; DGX ok)
MID_TRANSFORMER = False # extra 1-layer attention at the ~860-pt mid level
SAMPLER = 'fps'         # stage sampler: 'fps' (historical) or 'grid' (linear; needed at scale)
ARCH = 'edgeconv'       # 'edgeconv' (legacy kNN) | 'stencil' (current); 'v1'/'v2' accepted as aliases
V2_KNN = 32             # v2 only: serialized-window neighbour count (total across curves)
V2_CURVES = 1           # v2 only: Morton curves unioned per stage (1 or 2)
V2_NEIGHBORS = 'serial' # v2 only: 'serial' (Morton window) | 'stencil' (exact voxel offsets)
V2_STENCIL = 1          # v2 only: stencil radius (1 -> K=27, 2 -> K=125)
V2_DIFF = False         # v2 only: add decomposed W2*(h_j - h_i) feature-difference term
V2_BASE_GRID = 0.04     # v2 only: voxel size of the INPUT cloud (stencil/serialize quantization)
V2_POOL_GRIDS = '0.08,0.16,0.32'  # v2 only: per-stage pooling voxel sizes
V2_DIRECTIONAL = False  # v2 only: cylindrical anisotropic aggregation (E8)
V2_STENCIL_Z = 0        # v2 only: taller vertical stencil reach (0 = same as radius)
INIT_WEIGHTS = ''       # optional checkpoint to warm-restart from (see run_training)
RESUME = ''             # optional checkpoint.pth to resume the SAME schedule from
LAST_VAL_MIOU = 0.0     # mIoU of the most recent validate() call (see validate)
SELECT_METRIC = 'val_miou'  # 'val_miou' (default, evidence-based) or 'val_acc' (legacy)
ENC_CHANNELS = None     # e.g. '128,256,320,384' -- overrides the (64,128,256,512) plan
BOTTLENECK_DIM = None   # transformer width; defaults to the last encoder channel

# --- Room mode (Stage E) -------------------------------------------------------------
# 'room' block_mode trains on whole voxel-subsampled rooms instead of 2 m columns, which
# is what every S3DIS SOTA model does (see room_pipeline.py for the measurements). Needs
# --sampler grid: FPS is quadratic and costs 24.1 s at 131k points vs 54 ms for grid.
ROOM_DATA_PATH = './room_s3dis'
ROOM_GRID = 0.04          # voxel size (DeLA/KPConvX/PTv2 all use 0.04 m for S3DIS)
ROOM_MAX_POINTS = 30000   # training point budget per room crop (DeLA's value)
ROOM_LOOP = 4             # repeats of the room list per epoch (only ~170 train rooms exist,
                          # so one pass is ~40 steps; references use loop=30 x 100 epochs)

# Block partitioning strategy:
#   'grid'   - legacy 0.5 m cubic-cell hashing (loses vertical context, drops sparse cells)
#   'column' - overlapping full-height columns (preserves up/down/left/right context)
# 'column' is recommended; kept non-default so existing block_s3dis/weights still work.
BLOCK_MODE = 'column'   # default: context-preserving columns (was 'grid'); override with --block_mode
COLUMN_WINDOW = 2.0     # XY column side length (m)
COLUMN_STRIDE = 2.0     # XY step between columns (m); == window => no overlap (block count ~ grid)

# Feature layout (geo, rgb, spatial, block-context), resolved from model_params.json in
# main(). Default (4,3,3,0)=10D reproduces the original S3DIS model; other domains (or
# --block_context) override it. context_dim > 0 appends a per-block buffered-context
# descriptor (see data_processing.BlockContextExtractor) at block-build time.
FEATURE_DIMS = (4, 3, 3, 0)
NORMALS_PRESENT = True

class FocalLoss(nn.Module):
	"""Focal Loss with per-class weights for addressing class imbalance."""
	def __init__(self, class_weights=None, gamma=2.5, ignore_index=-1):
		super(FocalLoss, self).__init__()
		self.class_weights = class_weights
		self.gamma = gamma
		self.ignore_index = ignore_index
	
	def forward(self, inputs, targets):
		# Filter out ignored indices
		valid_mask = (targets != self.ignore_index)
		inputs = inputs[valid_mask]
		targets = targets[valid_mask]
		
		# Calculate cross entropy with class weights
		ce_loss = F.cross_entropy(inputs, targets, weight=self.class_weights, reduction='none')
		
		# Calculate pt (probability of true class)
		pt = torch.exp(-ce_loss)
		
		# Focal Loss formula: (1-pt)^gamma * log(pt)
		focal_loss = (1 - pt) ** self.gamma * ce_loss

		return focal_loss.mean()

def _lovasz_grad(gt_sorted):
	"""Gradient of the Lovász extension of the Jaccard loss (Berman et al., 2018)."""
	p = len(gt_sorted)
	gts = gt_sorted.sum()
	intersection = gts - gt_sorted.float().cumsum(0)
	union = gts + (1 - gt_sorted).float().cumsum(0)
	jaccard = 1. - intersection / union
	if p > 1:
		jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
	return jaccard

def lovasz_softmax_flat(probas, labels, num_classes):
	"""Multi-class Lovász-Softmax over flat (already valid-masked) points; present classes only."""
	if probas.numel() == 0:
		return probas.sum() * 0.
	losses = []
	for c in range(num_classes):
		fg = (labels == c).float()
		if fg.sum() == 0:
			continue  # class absent in this batch -> skip (present-class averaging)
		errors = (fg - probas[:, c]).abs()
		errors_sorted, perm = torch.sort(errors, 0, descending=True)
		losses.append(torch.dot(errors_sorted, _lovasz_grad(fg[perm])))
	if not losses:
		return probas.sum() * 0.
	return torch.stack(losses).mean()

class FocalLovaszLoss(nn.Module):
	"""Focal (per-point) + Lovász-Softmax (per-class IoU surrogate).

	Focal optimizes per-point accuracy/recall; Lovász directly optimizes the mean IoU we
	actually report and on which the rare classes (sofa/column/board/door) are graded. They
	are complementary: loss = focal_weight*focal + lovasz_weight*lovasz. This targets mIoU,
	not just OA, which is where the previous recall-only weighting left gains on the table.
	"""
	def __init__(self, class_weights=None, gamma=2.0, ignore_index=-1, focal_weight=1.0, lovasz_weight=1.0):
		super(FocalLovaszLoss, self).__init__()
		self.focal = FocalLoss(class_weights=class_weights, gamma=gamma, ignore_index=ignore_index)
		self.ignore_index = ignore_index
		self.focal_weight = focal_weight
		self.lovasz_weight = lovasz_weight

	def forward(self, inputs, targets):
		loss = self.focal_weight * self.focal(inputs, targets)
		valid = (targets != self.ignore_index)
		if valid.any():
			# Lovász needs probabilities in FP32 (sort/cumsum); cast for AMP safety.
			probas = F.softmax(inputs[valid].float(), dim=1)
			loss = loss + self.lovasz_weight * lovasz_softmax_flat(probas, targets[valid], inputs.size(1))
		return loss

def safe_gpu_operation():
	"""GPU memory safety check and cleanup with detailed logging"""
	if not torch.cuda.is_available():
		return True
	
	try:
		memory_allocated = torch.cuda.memory_allocated() / 1024**3
		memory_cached = torch.cuda.memory_reserved() / 1024**3  # Track cached memory
		memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
		memory_usage_ratio = memory_allocated / memory_total
		
		# Log memory usage every 50 calls for debugging
		if not hasattr(safe_gpu_operation, 'call_count'):
			safe_gpu_operation.call_count = 0
		safe_gpu_operation.call_count += 1
		
		# if safe_gpu_operation.call_count % 50 == 0:
			# print(f"Memory: {memory_allocated:.1f}GB allocated, {memory_cached:.1f}GB cached, {memory_total:.1f}GB total")
		
		if memory_usage_ratio > GPU_MEMORY_THRESHOLD:
			torch.cuda.empty_cache()
			gc.collect()  # Also clear Python object memory
			memory_after_cleanup = torch.cuda.memory_allocated() / 1024**3
			print(f"GPU memory cleanup: {memory_allocated:.1f}GB -> {memory_after_cleanup:.1f}GB")
			
			# Return False if still above critical threshold after cleanup
			return (memory_after_cleanup / memory_total) < 0.95
			
		return True
	except Exception as e:
		print(f"GPU memory check failed: {e}")
		torch.cuda.empty_cache()
		return True

def check_numerical_stability(tensor, name="tensor"):
	"""Check for NaN/Inf values in tensors"""
	if torch.isnan(tensor).any():
		print(f"NaN detected in {name}")
		return False
	if torch.isinf(tensor).any():
		print(f"Inf detected in {name}")
		return False
	return True

def setup_logging():
	"""Log file setup and initialization"""
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	log_dir = os.path.join("logs", timestamp)
	os.makedirs(log_dir, exist_ok=True)
	
	# CSV log file path
	csv_log_path = os.path.join(log_dir, "training_log.csv")
	
	# Write CSV header
	with open(csv_log_path, 'w', newline='') as f:
		writer = csv.writer(f)
		writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc',
						'loss_diff', 'acc_diff', 'learning_rate', 'val_miou'])
	
	return log_dir, csv_log_path

def log_epoch_metrics(csv_path, epoch, train_loss, train_acc, val_loss, val_acc, lr, val_miou=None):
	"""Save epoch metrics to CSV file. val_miou is the metric to judge by -- see validate()."""
	loss_diff = val_loss - train_loss
	acc_diff = train_acc - val_acc

	with open(csv_path, 'a', newline='') as f:
		writer = csv.writer(f)
		writer.writerow([epoch, f"{train_loss:.6f}", f"{train_acc:.6f}",
						f"{val_loss:.6f}", f"{val_acc:.6f}",
						f"{loss_diff:.6f}", f"{acc_diff:.6f}", f"{lr:.8f}",
						f"{(val_miou if val_miou is not None else 0.0):.6f}"])

def save_training_summary(log_dir, history, best_epoch, best_val_acc):
	"""Save complete summary after training"""
	summary = {
		"training_config": {
			"num_epochs": NUM_EPOCHS,
			"batch_size": BATCH_SIZE,
			"learning_rate": LEARNING_RATE,
			"block_size": BLOCK_SIZE,
			"num_features": NUM_FEATURES,
			"num_classes": NUM_CLASSES,
			"gradient_clip_value": GRADIENT_CLIP_VALUE
		},
		"best_performance": {
			"best_epoch": best_epoch,
			"best_val_accuracy": best_val_acc,
			"final_train_loss": history['train_loss'][-1] if history['train_loss'] else 0.0,
			"final_val_loss": history['val_loss'][-1] if history['val_loss'] else 0.0
		},
		"training_history": history
	}
	
	summary_path = os.path.join(log_dir, "training_summary.json")
	with open(summary_path, 'w') as f:
		json.dump(summary, f, indent=2)
	
	print(f"Training summary saved to: {summary_path}")
	
	# Generate training plots with 4 subplots
	fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
	
	# Training and Validation Loss
	ax1.plot(history['train_loss'], label='Train Loss', color='blue')
	ax1.plot(history['val_loss'], label='Validation Loss', color='red')
	ax1.set_xlabel('Epoch')
	ax1.set_ylabel('Loss')
	ax1.set_title('Training and Validation Loss')
	ax1.legend()
	ax1.grid(True)
	
	# Training and Validation Accuracy  
	ax2.plot(history['train_acc'], label='Train Accuracy', color='blue')
	ax2.plot(history['val_acc'], label='Validation Accuracy', color='red')
	ax2.set_xlabel('Epoch')
	ax2.set_ylabel('Accuracy')
	ax2.set_title('Training and Validation Accuracy')
	ax2.legend()
	ax2.grid(True)
	
	# Loss difference (Overfitting detection)
	loss_diff = [val - train for val, train in zip(history['val_loss'], history['train_loss'])]
	ax3.plot(loss_diff, label='Val Loss - Train Loss', color='orange')
	ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
	ax3.set_xlabel('Epoch')
	ax3.set_ylabel('Loss Difference')
	ax3.set_title('Overfitting Detection (Val - Train Loss)')
	ax3.legend()
	ax3.grid(True)
	
	# Accuracy difference (Generalization gap)
	acc_diff = [train - val for val, train in zip(history['val_acc'], history['train_acc'])]
	ax4.plot(acc_diff, label='Train Acc - Val Acc', color='green')
	ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
	ax4.set_xlabel('Epoch')
	ax4.set_ylabel('Accuracy Difference')
	ax4.set_title('Generalization Gap (Train - Val Acc)')
	ax4.legend()
	ax4.grid(True)
	
	plt.tight_layout()
	plot_path = os.path.join(log_dir, 'training_plots.png')
	plt.savefig(plot_path)
	plt.close()
	
	print(f"Training plots saved to: {plot_path}")

def refresh_curvature_inplace(features, points):
	"""Recompute the curvature channel from coordinates at block-build time.

	The stored processed_s3dis/*.pt carry a DEPRECATED curvature (distance of each normal
	from the room's global mean normal, ~constant, "almost no usable signal"), while the live
	code and inference use local surface variation. Rather than mutate the source .pt with a
	separate script, we refresh channel once per room here (on the FULL cloud, matching how
	inference computes features) so the generated blocks - the actual training data - carry the
	correct edge cue. No-op if the configured feature layout has no curvature channel.
	"""
	normals_count = 3 if NORMALS_PRESENT else 0
	if FEATURE_DIMS[0] <= normals_count:   # geo group is normals-only -> no curvature slot
		return features
	from data_processing import compute_surface_variation
	knn = int(get_model_config().get('features', {}).get('neighbor_knn', 15))
	features[:, normals_count] = compute_surface_variation(points, knn=knn)
	return features

def preprocess_dataset():
	"""
	Split existing processed_s3dis pt files into 8192 point blocks
	and save them to the block_s3dis folder.
	"""
	print("Starting dataset preprocessing...")
	if os.path.exists(BLOCK_DATA_PATH):
		return
	os.makedirs(BLOCK_DATA_PATH, exist_ok=True)

	start_time = time.time()

	# Resolve the feature spec ONCE (it is config-global); resolving per file re-printed
	# its consistency warning for every room.
	feat_spec = resolve_feature_config(get_model_config())

	# Process all area files
	all_areas = TRAIN_AREAS + [TEST_AREA]
	block_counter = 0
	
	for area in tqdm(all_areas, desc="Processing areas"):
		area_path = os.path.join(PROCESSED_DATA_PATH, area)
		if not os.path.exists(area_path):
			continue
			
		print(f"Processing {area}...")
		pt_files = glob(os.path.join(area_path, '*.pt'))
		
		for pt_file in tqdm(pt_files, desc=f"Processing {area} files", leave=False):
			try:
				# Load original data (set weights_only=False for PyTorch 2.6+ compatibility)
				data = torch.load(pt_file, weights_only=False)
				
				# Convert to numpy for spatial processing
				points = data.pos.numpy()
				features = data.x.numpy()
				labels = data.y.numpy()
				features = refresh_curvature_inplace(features, points)  # fix stale curvature channel
				# Optional wide-area block-context descriptor (None when disabled in config)
				ctx_extractor = make_block_context_extractor(points, features, feat_spec)

				# Load preprocessing parameters from config
				config = get_model_config()
				preprocessing = config.get('preprocessing', {})
				grid_min_coords = np.array(preprocessing.get('grid_min_coords', [0.0, 0.0, 0.0]))
				grid_resolution = preprocessing.get('grid_resolution', 1.0)
				
				# Normalize coordinates to handle negative values and optimize grid distribution
				normalized_points = points - grid_min_coords
				grid_coords = np.floor(normalized_points / grid_resolution).astype(np.int32)
				grid_hashes = (grid_coords[:, 0] * 73856093) ^ (grid_coords[:, 1] * 19349663) ^ (grid_coords[:, 2] * 83492791)  # Hash function using large prime numbers for 3D spatial hashing
				unique_hashes, inverse_indices = np.unique(grid_hashes, return_inverse=True)
				
				# Create spatial blocks with guaranteed BLOCK_SIZE
				for hash_idx, grid_hash in enumerate(unique_hashes):
					point_indices = np.where(inverse_indices == hash_idx)[0]
					if len(point_indices) < BLOCK_SIZE * 0.1:
						continue
					
					for start_idx in range(0, len(point_indices), BLOCK_SIZE):
						end_idx = min(start_idx + BLOCK_SIZE, len(point_indices))
						raw_indices = point_indices[start_idx:end_idx]
						original_size = len(raw_indices)
						
						# Always create EXACTLY BLOCK_SIZE indices
						if original_size < BLOCK_SIZE:
							padding_needed = BLOCK_SIZE - original_size
							if original_size > 0:
								padding_indices = np.random.choice(raw_indices, padding_needed, replace=True)
								final_indices = np.concatenate([raw_indices, padding_indices])
							else:
								continue
						else:
							final_indices = raw_indices[:BLOCK_SIZE]
							original_size = min(original_size, BLOCK_SIZE)
						
						# Verify BLOCK_SIZE constraint
						assert len(final_indices) == BLOCK_SIZE, f"Block size mismatch: {len(final_indices)} != {BLOCK_SIZE}"
						
						# Extract features with EXACT BLOCK_SIZE
						block_feats = features[final_indices]
						if ctx_extractor is not None:
							block_feats = append_block_context(block_feats, ctx_extractor,
															   final_indices[:original_size])
						block_x = torch.FloatTensor(block_feats)
						block_pos = torch.FloatTensor(points[final_indices])
						block_y = torch.LongTensor(labels[final_indices])
						
						# Handle padding labels (set padded points to ignore_index=-1)
						if original_size < BLOCK_SIZE:
							block_y[original_size:] = -1  # ignore_index for loss
						
						# Generate valid point mask
						valid_mask = torch.ones(BLOCK_SIZE, dtype=torch.bool)
						if original_size < BLOCK_SIZE:
							valid_mask[original_size:] = False
						
						# Final size verification
						assert block_x.shape[0] == BLOCK_SIZE and block_pos.shape[0] == BLOCK_SIZE and block_y.shape[0] == BLOCK_SIZE
					
						# Create torch_geometric Data object
						from torch_geometric.data import Data
						block_data = Data(
							x=block_x,
							pos=block_pos,
							y=block_y,
							valid_mask=valid_mask,
							num_valid_points=original_size,
							area=area  # Store which area this block came from
						)
						
						# Save block
						block_filename = f"block_{block_counter:012d}_{area}.pt"
						torch.save(block_data, os.path.join(BLOCK_DATA_PATH, block_filename))
						block_counter += 1

						if COOLDOWN_SEC > 0 and block_counter % 100 == 0: # Optional pause for GPU cooling
							time.sleep(COOLDOWN_SEC)
		
			except Exception as e:
				print(f"Error processing file {pt_file}: {e}")
				continue
	
	end_time = time.time()
	preprocessing_time = end_time - start_time
	print(f"Preprocessing completed in {preprocessing_time:.2f} seconds")
	print(f"Total blocks created: {block_counter}")
	print(f"Average time per block: {preprocessing_time/block_counter:.4f} seconds")
	return block_counter

def preprocess_dataset_columns():
	"""
	Spatial-context-preserving alternative to preprocess_dataset().

	Splits each processed room into OVERLAPPING full-height columns via
	partition_columns() instead of tiny cubic grid cells. Benefits:
	  - vertical/side context preserved (walls, columns, doors seen whole),
	  - no silent dropping of sparse regions (every point is covered),
	  - room encoded in the filename ('block_<n>_<room>__<area>.pt') so the
	    train/val split can be room-disjoint (see split_train_val()).

	Enable with --block_mode column. Blocks are written to BLOCK_DATA_PATH exactly
	like the grid path, so the rest of the pipeline is unchanged.
	"""
	from torch_geometric.data import Data
	print("Starting COLUMN-mode dataset preprocessing (overlapping full-height columns)...")
	if os.path.exists(BLOCK_DATA_PATH) and len(glob(os.path.join(BLOCK_DATA_PATH, '*.pt'))) > 0:
		return
	os.makedirs(BLOCK_DATA_PATH, exist_ok=True)

	# Resolve the feature spec ONCE (it is config-global); resolving per file re-printed
	# its consistency warning for every room.
	feat_spec = resolve_feature_config(get_model_config())

	all_areas = TRAIN_AREAS + [TEST_AREA]
	block_counter = 0
	for area in tqdm(all_areas, desc="Processing areas"):
		area_path = os.path.join(PROCESSED_DATA_PATH, area)
		if not os.path.exists(area_path):
			continue
		pt_files = glob(os.path.join(area_path, '*.pt'))
		for pt_file in tqdm(pt_files, desc=f"Processing {area} files", leave=False):
			try:
				data = torch.load(pt_file, weights_only=False)
				points = data.pos.numpy()
				features = data.x.numpy()
				labels = data.y.numpy()
				features = refresh_curvature_inplace(features, points)  # fix stale curvature channel
				# Optional wide-area block-context descriptor (None when disabled in config)
				ctx_extractor = make_block_context_extractor(points, features, feat_spec)
				# sanitize room name for the filename (area token must stay last, after '__')
				room = os.path.splitext(os.path.basename(pt_file))[0]
				room = ''.join(c if (c.isalnum() or c in '-') else '-' for c in room)

				blocks = partition_columns(points, block_size=BLOCK_SIZE,
										   window=COLUMN_WINDOW, stride=COLUMN_STRIDE, seed=block_counter)
				for idx, num_real in blocks:
					block_feats = features[idx]
					if ctx_extractor is not None:
						block_feats = append_block_context(block_feats, ctx_extractor, idx[:num_real])
					block_x = torch.FloatTensor(block_feats)
					block_pos = torch.FloatTensor(points[idx])
					block_y = torch.LongTensor(labels[idx])
					valid_mask = torch.ones(BLOCK_SIZE, dtype=torch.bool)
					if num_real < BLOCK_SIZE:
						block_y[num_real:] = -1        # ignore padded points in the loss
						valid_mask[num_real:] = False
					# Deterministic spatial train/val tag (no room labels needed). Blocks in
					# the same coarse super-cell of the same source share a tag, so adjacent
					# / overlapping blocks never split across train and val.
					centroid_xy = points[idx[:num_real], :2].mean(axis=0)
					is_val = spatial_split_is_val(room, centroid_xy, super_size=4.0, val_ratio=0.2, seed=42)
					tag = 'val' if is_val else 'train'
					block_data = Data(x=block_x, pos=block_pos, y=block_y,
									  valid_mask=valid_mask, num_valid_points=num_real, area=area)
					block_filename = f"block_{block_counter:012d}_{room}__{tag}__{area}.pt"
					torch.save(block_data, os.path.join(BLOCK_DATA_PATH, block_filename))
					block_counter += 1
			except Exception as e:
				print(f"Error processing file {pt_file}: {e}")
				continue
	print(f"Total column blocks created: {block_counter}")
	return block_counter

def validate_block_files(file_list):
	"""Validate block files and remove corrupted ones"""
	valid_files = []
	corrupted_files = []
	
	print("Validating block files...")
	for file_path in tqdm(file_list, desc="Checking files"):
		try:
			data = torch.load(file_path, weights_only=False)
			# Basic validation checks
			if (hasattr(data, 'x') and hasattr(data, 'pos') and 
				hasattr(data, 'y') and hasattr(data, 'valid_mask')):
				valid_files.append(file_path)
			else:
				corrupted_files.append(file_path)
		except Exception as e:
			print(f"Corrupted file found: {file_path} - {e}")
			corrupted_files.append(file_path)
	
	# Remove corrupted files
	for corrupted_file in corrupted_files:
		os.remove(corrupted_file)
		print(f"Removed corrupted file: {corrupted_file}")
	
	print(f"Valid files: {len(valid_files)}, Removed corrupted files: {len(corrupted_files)}")
	return valid_files

# Further reduced augmentation for maximum learning speed
def get_augment_prob_and_strength(epoch, max_epochs):
	"""Per-epoch (probability, strength) for block augmentation.

	'strong' preset: CONSTANT full-strength augmentation, as in every SOTA recipe (the
	per-op probabilities inside augment_training_block do the modulation). The legacy
	schedule below decayed instead -- mean (prob x strength) 0.046 over 60 epochs, ~10x
	below standard practice -- which showed up as a constant 8.4-point val->test gap that
	no architecture change could move. Constant augmentation is the fix being tested.

	'legacy' preset: the original decaying schedule, kept so old runs stay reproducible.
	"""
	if AUG_PRESET == 'strong':
		return 1.0, 1.0
	if epoch <= max_epochs * 0.1:
		return 0.3, 0.4  # Very low augmentation
	elif epoch <= max_epochs * 0.5:
		return 0.2, 0.3  # Minimal augmentation
	return 0.1, 0.2  # Almost no augmentation

class BlockDataset(torch.utils.data.Dataset):
	def __init__(self, file_list, augment=False, augment_prob=0.0, augment_strength=1.0, rgb_dropout_prob=0.0,
				 feature_dims=(4, 3, 3), normals_present=True):
		self.file_list = file_list
		self.augment = augment
		self.augment_prob = augment_prob  # Probability of applying augmentation
		self.augment_strength = augment_strength  # Strength of augmentation
		self.rgb_dropout_prob = rgb_dropout_prob  # Prob. of zeroing RGB (RGB-free robustness)
		self.feature_dims = feature_dims          # (geo, rgb, spatial) for augmentation offsets
		self.normals_present = normals_present    # whether x[:,0:3] are normals (co-rotate)
		self.error_count = 0
		self.max_error_rate = 0.1  # Allow up to 10% error rate
	
	def __len__(self):
		return len(self.file_list)
	
	def __getitem__(self, idx):
		max_retries = 3
		for retry in range(max_retries):
			try:
				data = torch.load(self.file_list[idx], weights_only=False)
				if hasattr(data, 'x') and hasattr(data, 'pos') and hasattr(data, 'y'):
					# Format: geo(4) + RGB(3) + spatial(3) = 10D
					# On-the-fly geometric + color augmentation (was previously a no-op).
					# Applied per-block with probability augment_prob; rotation about Z,
					# jitter, scaling and RGB jitter/dropout. Rotation also rotates the
					# stored normal vectors so geometry + features stay consistent.
					if self.augment and self.augment_prob > 0 and random.random() < self.augment_prob:
						data = augment_training_block(
							data,
							strength=self.augment_strength,
							geo_dim=self.feature_dims[0],
							rgb_dim=self.feature_dims[1],
							normals_present=self.normals_present,
							rgb_dropout_prob=self.rgb_dropout_prob,
							full_rotate=AUG_FULL_ROTATE,
							isotropic_scale=AUG_ISOTROPIC_SCALE,
							flip_prob=AUG_FLIP_PROB,
							tilt_std=AUG_TILT_STD,
							autocontrast_prob=AUG_AUTOCONTRAST_PROB,
							scale_range=AUG_SCALE_RANGE,
						)
					return data
				print(f"Invalid data structure in file {self.file_list[idx]}")
				return None
			except Exception as e:
				self.error_count += 1
				if retry == max_retries - 1:  # Last retry
					print(f"Error loading file {self.file_list[idx]} after {max_retries} retries: {e}")
					# Check if error rate is too high
					if self.error_count / len(self.file_list) > self.max_error_rate:
						print(f"Warning: High error rate detected ({self.error_count}/{len(self.file_list)})")
					return None
				# Wait a bit before retry
				torch.cuda.empty_cache() if torch.cuda.is_available() else None
		return None

def collate_fn(batch):
	# Filter out None objects that caused errors
	valid_batch = [d for d in batch if d is not None]
	
	# Require at least half of the batch to be valid
	min_batch_size = max(1, len(batch) // 2)
	if len(valid_batch) < min_batch_size:
		print(f"Warning: Only {len(valid_batch)}/{len(batch)} valid samples in batch. Skipping batch.")
		return None
	
	from torch_geometric.data import Batch
	return Batch.from_data_list(valid_batch)

# Globals populated by build_dataloaders() and run_training().
# IMPORTANT (v0.3 bug fix): dataset/loader construction used to run at *import time*,
# i.e. BEFORE main() parsed --config / --block_data_path / --train_areas / --block_size
# and overrode the globals. As a result those CLI/config options were silently ignored
# (loaders were always built from the module defaults). Construction is now deferred to
# build_dataloaders(), invoked from run_training() after main() has applied the config.
train_loader = None
val_loader = None
train_dataset = None
val_dataset = None
test_block_files = []
model = None
optimizer = None
criterion = None
scaler = None

def _split_tag(block_path):
	"""Return 'train'/'val' if the block name carries an explicit spatial split tag.

	Column-mode blocks are named 'block_<n>_<source>__<tag>__<area>.pt' where <tag> is a
	deterministic *spatial* assignment (see data_processing.spatial_split_is_val). This
	needs no room labelling and works even when a whole area is a single source file.
	"""
	name = os.path.splitext(os.path.basename(block_path))[0]
	if '__val__' in name:
		return 'val'
	if '__train__' in name:
		return 'train'
	return None

def _source_group_key(block_path):
	"""Best-effort *source-file* group id (NOT a semantic room label).

	This is simply the preprocessing source (an S3DIS room happens to map to one source
	.pt). Used only as a fallback grouping when no spatial tag is present. Legacy
	grid-mode blocks ('block_<n>_<area>.pt') carry no source id -> returns None.
	"""
	name = os.path.splitext(os.path.basename(block_path))[0]
	if '__' in name:
		parts = name.split('__', 1)[0].split('_', 2)   # block_<n>_<source>
		if len(parts) == 3:
			return parts[2]
	return None

def split_train_val(train_block_files, val_ratio=0.2, seed=42):
	"""Leakage-controlled train/val split with graceful degradation.

	Precedence:
	  1. Explicit spatial tag in the filename ('__train__'/'__val__', written by the
	     column preprocessor). This is coordinate-based, needs no room labels, and keeps
	     adjacent/overlapping blocks together -> the correct general solution.
	  2. Source-file grouping (hold out whole source clouds). Only meaningful when there
	     are several sources; requires the input to actually be split into multiple files
	     (e.g. S3DIS rooms) - which is NOT guaranteed for arbitrary datasets.
	  3. Plain random block split (last resort; may leak context across train/val).

	Note: this pipeline has no room *labels*. Random block-level splitting leaks context
	because adjacent (and padded-duplicate) blocks land in both train and val, making val
	accuracy optimistic (part of the val 90% vs Area5 test 80% gap).
	"""
	# 1) explicit spatial tags
	tags = [_split_tag(f) for f in train_block_files]
	if all(t is not None for t in tags) and any(t == 'val' for t in tags):
		train_files = [f for f, t in zip(train_block_files, tags) if t == 'train']
		val_files = [f for f, t in zip(train_block_files, tags) if t == 'val']
		if train_files and val_files:
			print(f"Spatial split: {len(train_files)} train / {len(val_files)} val blocks (super-cell holdout)")
			return train_files, val_files

	# 2) source-file group holdout
	groups = {}
	for f in train_block_files:
		groups.setdefault(_source_group_key(f), []).append(f)
	if None not in groups and len(groups) >= 5:
		names = sorted(groups.keys())
		rng = random.Random(seed)
		rng.shuffle(names)
		n_val = max(1, int(len(names) * val_ratio))
		val_names = set(names[:n_val])
		train_files, val_files = [], []
		for g, files in groups.items():
			(val_files if g in val_names else train_files).extend(files)
		print(f"Source-group split: {len(names)-n_val} train / {n_val} val source clouds")
		return train_files, val_files

	# 3) random fallback
	print("No spatial tag / insufficient source groups -> random block split (context leakage possible).")
	return train_test_split(train_block_files, test_size=val_ratio, random_state=seed)

def build_dataloaders(rgb_dropout_prob=0.0):
	"""Build train/val datasets + loaders using the *current* global config.

	Must be called after main() has applied CLI args / config, not at import time.
	"""
	global train_loader, val_loader, train_dataset, val_dataset, test_block_files

	# ROOM MODE (Stage E): whole voxel-subsampled rooms instead of 2 m columns. Separate
	# cache, dataset and split; the block path below is left completely untouched.
	if BLOCK_MODE == 'room':
		from room_pipeline import preprocess_rooms, RoomDataset, split_rooms_by_area
		preprocess_rooms(PROCESSED_DATA_PATH, ROOM_DATA_PATH, TRAIN_AREAS + [TEST_AREA],
						 TEST_AREA, grid=ROOM_GRID,
						 neighbor_knn=int(get_model_config().get('features', {}).get('neighbor_knn', 15)))
		room_files = sorted(glob(os.path.join(ROOM_DATA_PATH, '*.pt')))
		train_files, val_files, test_block_files = split_rooms_by_area(room_files, TRAIN_AREAS, TEST_AREA)
		aug_kwargs = dict(full_rotate=AUG_FULL_ROTATE, isotropic_scale=AUG_ISOTROPIC_SCALE,
						  flip_prob=AUG_FLIP_PROB, tilt_std=AUG_TILT_STD,
						  autocontrast_prob=AUG_AUTOCONTRAST_PROB, scale_range=AUG_SCALE_RANGE)
		train_dataset = RoomDataset(train_files, max_points=ROOM_MAX_POINTS, augment=True,
									augment_prob=1.0, augment_strength=1.0,
									feature_dims=FEATURE_DIMS, normals_present=NORMALS_PRESENT,
									rgb_dropout_prob=rgb_dropout_prob, aug_kwargs=aug_kwargs,
									loop=ROOM_LOOP, feature_dim=NUM_FEATURES)
		val_dataset = RoomDataset(val_files, max_points=ROOM_MAX_POINTS, augment=False,
								  feature_dims=FEATURE_DIMS, normals_present=NORMALS_PRESENT,
								  feature_dim=NUM_FEATURES)
		train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
								  num_workers=0, collate_fn=collate_fn)
		val_loader = DataLoader(val_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False,
								num_workers=0, collate_fn=collate_fn)
		print(f"ROOM MODE: {len(train_files)} train / {len(val_files)} val / {len(test_block_files)} test rooms")
		print(f"  grid={ROOM_GRID} m, max_points={ROOM_MAX_POINTS}, loop={ROOM_LOOP}, sampler={SAMPLER}")
		if SAMPLER != 'grid':
			print("  WARNING: room mode with --sampler fps is ~450x slower at this scale; use --sampler grid.")
		print(f"Training batches: {len(train_loader)} | Validation batches: {len(val_loader)}")
		if len(train_loader) == 0 or len(val_loader) == 0:
			raise ValueError("Empty room loader! Check processed_s3dis and the room cache.")
		return

	# Run preprocessing if block data doesn't exist
	if not os.path.exists(BLOCK_DATA_PATH) or len(glob(os.path.join(BLOCK_DATA_PATH, '*.pt'))) == 0:
		print(f"Block data not found. Running preprocessing (mode={BLOCK_MODE})...")
		if BLOCK_MODE == 'column':
			preprocess_dataset_columns()
		else:
			preprocess_dataset()

	# Separate block files by area
	all_block_files = glob(os.path.join(BLOCK_DATA_PATH, '*.pt'))

	# Guard against stale caches: blocks are cached on disk, so toggling --block_context
	# (or any feature group) without changing --block_data_path would silently train on
	# mismatched features. Fail fast with an actionable message instead.
	if all_block_files:
		sample = torch.load(all_block_files[0], weights_only=False)
		if sample.x.shape[1] != NUM_FEATURES:
			raise ValueError(
				f"Existing blocks in '{BLOCK_DATA_PATH}' carry {sample.x.shape[1]}D features but the "
				f"current config expects {NUM_FEATURES}D (block_context on/off or feature toggle "
				f"mismatch?). Point --block_data_path to a fresh folder or delete the stale blocks.")

	train_block_files = []
	test_block_files = []

	print("Categorizing block files by area...")
	for block_file in tqdm(all_block_files, desc="Categorizing blocks"):
		filename = os.path.basename(block_file)
		if TEST_AREA in filename:
			test_block_files.append(block_file)
		else:
			for area in TRAIN_AREAS:
				if area in filename:
					train_block_files.append(block_file)
					break

	# Room-disjoint (preferred) or random 8:2 split
	train_files, val_files = split_train_val(train_block_files, val_ratio=0.2, seed=42)

	print(f"Total training blocks: {len(train_files)}")
	print(f"Total validation blocks: {len(val_files)}")
	print(f"Total test blocks: {len(test_block_files)}")
	print(f"Block size: {BLOCK_SIZE} points per block")

	train_dataset = BlockDataset(train_files, augment=True, augment_prob=0.5, augment_strength=0.5,
								 rgb_dropout_prob=rgb_dropout_prob,
								 feature_dims=FEATURE_DIMS, normals_present=NORMALS_PRESENT)
	val_dataset = BlockDataset(val_files, augment=False, augment_prob=0.0, augment_strength=0.0,
								 feature_dims=FEATURE_DIMS, normals_present=NORMALS_PRESENT)

	# Rare-class block oversampling (see OVERSAMPLE_RARE). Weights are computed once from
	# the block label content; epoch length stays len(train_files) so runtimes are unchanged.
	train_sampler = None
	if OVERSAMPLE_RARE > 0:
		from torch.utils.data import WeightedRandomSampler
		cw = get_class_weights()
		weights = []
		for f in tqdm(train_files, desc='Computing sampling weights'):
			d = torch.load(f, weights_only=False)
			y = d.y[d.valid_mask].numpy()
			y = y[y >= 0]
			counts = np.bincount(y, minlength=NUM_CLASSES)[:NUM_CLASSES]
			present = np.nonzero(counts >= 0.01 * max(len(y), 1))[0]  # >=1% of the block
			w = max([cw[c] for c in present], default=1.0)
			weights.append(max(1.0, w) ** OVERSAMPLE_RARE)
		weights = np.asarray(weights, dtype=np.float64)
		train_sampler = WeightedRandomSampler(torch.from_numpy(weights),
											  num_samples=len(train_files), replacement=True)
		print(f"Rare-class oversampling ON (power={OVERSAMPLE_RARE}): "
			  f"mean weight {weights.mean():.2f}, max {weights.max():.2f} "
			  f"({int((weights > 1.0).sum())}/{len(weights)} blocks boosted)")

	# num_workers=0 to prevent Windows multiprocessing issues
	train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
							  shuffle=(train_sampler is None), sampler=train_sampler,
							  num_workers=0, collate_fn=collate_fn)
	val_loader = DataLoader(val_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_fn)

	print(f"Training dataset size: {len(train_dataset)}")
	print(f"Validation dataset size: {len(val_dataset)}")
	print(f"Training batches: {len(train_loader)}")
	print(f"Validation batches: {len(val_loader)}")

	if len(train_loader) == 0:
		raise ValueError("Training loader is empty! Check your data files.")
	if len(val_loader) == 0:
		raise ValueError("Validation loader is empty! Check your data files.")

def train(epoch):
	global WANDB_STEP
	model.train()
	
	# Learning Rate Warm-up implementation (Linear warm-up for stability)
	if epoch <= WARMUP_EPOCHS:
		# Gentler warmup progression
		warmup_factor = epoch / WARMUP_EPOCHS  # Removed square root for faster ramp
		min_lr_factor = 0.1  # Increased from 0.05
		warmup_lr = LEARNING_RATE * (min_lr_factor + (1.0 - min_lr_factor) * warmup_factor)
		for param_group in optimizer.param_groups:
			param_group['lr'] = warmup_lr
		print(f"Warmup Epoch {epoch}/{WARMUP_EPOCHS}: LR = {warmup_lr:.6f} (factor: {warmup_factor:.3f})")
	
	if len(train_loader) == 0:
		print("Warning: Training loader is empty!")
		return 0.0, 0.0
	
	pbar = tqdm(train_loader, desc=f'Epoch {epoch:02d}/{NUM_EPOCHS} [Training]')
	total_loss, correct_nodes, total_nodes = 0, 0, 0
	valid_batches = 0
	
	# Per-class accuracy tracking
	class_correct = np.zeros(NUM_CLASSES)
	class_total = np.zeros(NUM_CLASSES)

	debug_break_flag = False
	
	# Initialize gradient accumulation
	optimizer.zero_grad()
	
	for batch_idx, data in enumerate(pbar):
		# GPU memory safety check
		if not safe_gpu_operation():
			print(f"GPU memory critical at epoch {epoch}, batch {batch_idx}. Stopping training.")
			break
			
		if debug_break_flag:
			print("Debug break flag activated. Stopping training loop.")
			break

		if data is None: 
			print(f"Warning: Skipping corrupted batch {batch_idx} in training")
			continue
			
		valid_batches += 1
		data = data.to(device)
		
		try:
			# Run model in FP32 (Transformer indexing doesn't support FP16)
			out = model(data)
			
			# Only use autocast for loss calculation
			with autocast():
				valid_mask = data.valid_mask.view(-1)
				valid_out = out[valid_mask]
				valid_y = data.y[valid_mask]
				
				# Apply Focal Loss on all samples (OHEM removed for stability)
				loss = criterion(valid_out, valid_y)
		
			if not check_numerical_stability(loss, "loss"):
				print(f"Numerical instability at epoch {epoch}, batch {batch_idx}\n")
				torch.cuda.empty_cache()
				continue
			
			# Gradient Accumulation: Scale loss and backward
			loss = loss / ACCUMULATION_STEPS
			scaler.scale(loss).backward()
			
			# Update weights every ACCUMULATION_STEPS
			if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
				scaler.unscale_(optimizer)
				grad_norm_before = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
				
				if grad_norm_before > MAX_GRADIENT_NORM:
					print(f"Very large gradient detected: {grad_norm_before:.2f}, skipping update")
					# unscale_() was already called for this optimizer, so the scaler must be
					# cycled with update() before the next unscale_() or it raises
					# "unscale_() has already been called ..." on every following boundary,
					# silently freezing all weight updates. update() also backs off the scale
					# if the oversized gradient contained inf/nan.
					scaler.update()
					optimizer.zero_grad()
					torch.cuda.empty_cache()
					continue
				elif grad_norm_before > GRADIENT_CLIP_VALUE:
					torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_VALUE)
					if epoch % 5 == 0 and batch_idx % 50 == 0:
						print(f"Gradient clipped: {grad_norm_before:.2f} -> {GRADIENT_CLIP_VALUE}")
				
				scaler.step(optimizer)
				scaler.update()
				optimizer.zero_grad()
			
			if COOLDOWN_SEC > 0 and batch_idx % 200 == 0:
				time.sleep(COOLDOWN_SEC)

			total_loss += loss.item() * ACCUMULATION_STEPS  # Unscale for logging
			
			# Calculate accuracy (valid points only)
			pred = valid_out.argmax(dim=-1)
			correct_nodes += pred.eq(valid_y).sum().item()
			total_nodes += valid_mask.sum().item()
			
			# Track per-class accuracy
			for cls in range(NUM_CLASSES):
				cls_mask = (valid_y == cls)
				if cls_mask.sum() > 0:
					class_total[cls] += cls_mask.sum().item()
					class_correct[cls] += (pred[cls_mask] == cls).sum().item()
			
			# Log feature gates to wandb (batch-level)
			if USE_WANDB and hasattr(model, 'last_gates'):
				gates = model.last_gates
				current_batch_acc = pred.eq(valid_y).sum().item() / valid_mask.sum().item()
				WANDB_STEP += 1
				wandb.log({
					"batch/gate_geo": gates[:, 0].mean().item(),
					"batch/gate_rgb": gates[:, 1].mean().item(),
					"batch/gate_spatial": gates[:, 2].mean().item(),
					"batch/gate_context": gates[:, 3].mean().item(),  # 0 when block_context off
					"batch/loss": loss.item(),
					"batch/accuracy": current_batch_acc,
				}, step=WANDB_STEP)

			# Periodic memory cleanup. empty_cache()+gc.collect() force a CUDA sync and
			# are expensive; doing it every 10 batches throttled throughput badly. Every
			# 100 batches is plenty to keep the allocator from fragmenting.
			if batch_idx % 100 == 0:
				torch.cuda.empty_cache()
			
		except RuntimeError as e:
			print(f"Runtime error at epoch {epoch}, batch {batch_idx}: {str(e)}")
			torch.cuda.empty_cache()
			continue
		
		# More detailed progress bar information
		current_acc = correct_nodes / total_nodes if total_nodes > 0 else 0
		current_loss = total_loss / valid_batches if valid_batches > 0 else 0
		pbar.set_postfix({
			'Loss': f'{current_loss:.4f}',
			'Acc': f'{current_acc:.4f}',
			'Batch': f'{batch_idx+1}/{len(train_loader)}'
		})
	
	# Remove manual gate tracking - wandb handles it
	# geo_gates, rgb_gates, spatial_gates = [], [], # REMOVED
	
	# Handle case where no valid batches were processed
	if valid_batches == 0:
		print("Warning: No valid batches were processed in training!")
		return 0.0, 0.0
	
	# Print per-class accuracy for monitoring
	print(f"\n--- Per-Class Training Accuracy (Epoch {epoch}) ---")
	for cls in range(NUM_CLASSES):
		if class_total[cls] > 0:
			cls_acc = class_correct[cls] / class_total[cls]
			cls_name = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else f"Class_{cls}"
			print(f"{cls_name:12s}: {cls_acc*100:5.1f}% ({int(class_correct[cls])}/{int(class_total[cls])})")
	print("-" * 50)
	
	# Force memory cleanup at end of epoch
	torch.cuda.empty_cache()
	import gc
	gc.collect()
		
	return total_loss / valid_batches, correct_nodes / total_nodes

def validate(loader):
	model.eval()
	loader_name = 'Validation' if loader == val_loader else 'Test'
	
	if len(loader) == 0:
		print(f"Warning: {loader_name} loader is empty!")
		return 0.0, 0.0
	
	pbar = tqdm(loader, desc=f'[{loader_name}]')
	correct_nodes, total_nodes, total_loss = 0, 0, 0.0
	valid_batches = 0
	total_valid_points = 0
	total_points = 0
	# Per-class intersection/union for mIoU. Accuracy alone is a poor guide here: it is
	# dominated by ceiling/floor/wall (65% of the points), whereas mIoU averages all 13
	# classes equally. Measured across this project's runs, val_acc correlates +0.92 with
	# test OA but only +0.43 with test mIoU -- the run with the HIGHEST val_acc (0.9358)
	# scored 53.67 mIoU while the run with a much lower val_acc (0.9076) scored 56.45.
	# Selecting or judging on accuracy therefore picks the wrong model.
	cls_inter = np.zeros(NUM_CLASSES, dtype=np.int64)
	cls_union = np.zeros(NUM_CLASSES, dtype=np.int64)
	
	with torch.no_grad():
		for batch_idx, data in enumerate(pbar):
			# GPU memory safety check during validation
			if not safe_gpu_operation():
				print(f"GPU memory critical during {loader_name.lower()} at batch {batch_idx}. Stopping validation.")
				break
				
			if data is None: 
				print(f"Warning: Skipping corrupted batch {batch_idx} in {loader_name.lower()}")
				continue
				
			valid_batches += 1
			
			try:
				data = data.to(device)
				out = model(data)
				
				valid_mask = data.valid_mask.view(-1)
				valid_out = out[valid_mask]
				valid_y = data.y[valid_mask]
				
				# Track padding efficiency
				batch_valid_points = valid_mask.sum().item()
				batch_total_points = valid_mask.numel()
				total_valid_points += batch_valid_points
				total_points += batch_total_points
				
				# Warn if padding ratio is too high
				if batch_idx % 100 == 0:
					padding_ratio = 1.0 - (batch_valid_points / batch_total_points)
					if padding_ratio > 0.3:  # More than 30% padding
						print(f"High padding ratio in {loader_name} batch {batch_idx}: {padding_ratio:.1%}")
				
				# Add validation loss calculation with numerical stability check
				loss = criterion(valid_out, valid_y)
				if check_numerical_stability(loss, "validation_loss"):
					total_loss += loss.item()
					
					pred = valid_out.argmax(dim=-1)
					correct_nodes += pred.eq(valid_y).sum().item()
					total_nodes += valid_mask.sum().item()
					for cls in range(NUM_CLASSES):
						pm = (pred == cls); gm = (valid_y == cls)
						cls_inter[cls] += int((pm & gm).sum().item())
						cls_union[cls] += int((pm | gm).sum().item())
				else:
					print(f"Numerical instability in {loader_name.lower()} at batch {batch_idx}")
					torch.cuda.empty_cache()
					continue
					
			except RuntimeError as e:
				print(f"Runtime error in {loader_name.lower()} at batch {batch_idx}: {str(e)}")
				torch.cuda.empty_cache()
				continue
			
			# More detailed progress bar information
			current_acc = correct_nodes / total_nodes if total_nodes > 0 else 0
			current_loss = total_loss / valid_batches if valid_batches > 0 else 0
			pbar.set_postfix({
				'Loss': f'{current_loss:.4f}',
				'Acc': f'{current_acc:.4f}',
				'Batch': f'{batch_idx+1}/{len(loader)}'
			})
	
	# Handle case where no valid batches were processed
	if valid_batches == 0:
		print(f"Warning: No valid batches were processed in {loader_name.lower()}!")
		return 0.0, 0.0
	
	avg_loss = total_loss / valid_batches
	accuracy = correct_nodes / total_nodes if total_nodes > 0 else 0
	present = cls_union > 0                      # average over classes present in this split
	global LAST_VAL_MIOU
	LAST_VAL_MIOU = float((cls_inter[present] / cls_union[present]).mean()) if present.any() else 0.0
	print(f"  {loader_name} mIoU: {LAST_VAL_MIOU*100:.2f}  (acc {accuracy*100:.2f})")
	return avg_loss, accuracy

def run_training(args=None):
	global model, optimizer, criterion, scaler, WANDB_STEP  # Access global variables
	
	# Add freeze_support for Windows multiprocessing support
	import multiprocessing
	multiprocessing.freeze_support()
	
	# Log setup
	log_dir, csv_log_path = setup_logging()
	print(f"Logging to directory: {log_dir}")

	# Build datasets/loaders now that main() has applied CLI args + config (deferred
	# from import time so --config / --block_data_path / --train_areas take effect).
	build_dataloaders(rgb_dropout_prob=RGB_DROPOUT_PROB)

	# Load configuration and create model/optimizer/criterion (after config is loaded in main)
	config = get_model_config()
	print(f"\nInitializing model with {NUM_CLASSES} classes, {NUM_FEATURES} features...")

	# Create model
	model = build_model(NUM_FEATURES, NUM_CLASSES, FEATURE_DIMS, CONTEXT_MODE, WIDTH_MULT,
	                    MID_TRANSFORMER, SAMPLER, ENC_CHANNELS, BOTTLENECK_DIM, arch=ARCH).to(device)

	# Warm restart: continue from an existing checkpoint instead of random init.
	# Needed because a run can still be improving when its cosine schedule ends -- the
	# 60-epoch chunk run was climbing at +0.113 val-acc points/epoch at its last epoch.
	# Restarting the schedule from the reached weights (SGDR-style) keeps that progress
	# instead of throwing it away and retraining from scratch at a higher epoch count.
	# Optimizer state is deliberately NOT restored: a fresh cosine cycle wants a fresh
	# moment estimate, and the checkpoint only stores model weights anyway.
	if INIT_WEIGHTS:
		sd = torch.load(INIT_WEIGHTS, map_location=device, weights_only=False)
		if isinstance(sd, dict) and 'model_state_dict' in sd:
			sd = sd['model_state_dict']
		model.load_state_dict(sd)
		print(f"Warm restart: initialized from {INIT_WEIGHTS}")

	optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

	# Load class weights from configuration
	class_weights = get_class_weights(as_tensor=True, device=device)
	criterion = FocalLovaszLoss(class_weights=class_weights, gamma=FOCAL_GAMMA, ignore_index=-1,
								focal_weight=1.0, lovasz_weight=1.0)

	# Initialize GradScaler for mixed precision
	scaler = GradScaler()

	# Learning rate scheduler
	scheduler = optim.lr_scheduler.CosineAnnealingLR(
	    optimizer, T_max=NUM_EPOCHS, eta_min=LR_ETA_MIN
	)

	# Initialize wandb (online mode for cloud sync) - optional
	if USE_WANDB:
		dataset_name = config.get('dataset_name', 'custom').lower().replace('_', '-')
		wandb.init(
			project=f"pointedge-{dataset_name}",
			name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
			config={
				"learning_rate": LEARNING_RATE,
				"batch_size": BATCH_SIZE,
				"num_epochs": NUM_EPOCHS,
				"num_features": NUM_FEATURES,
				"block_size": BLOCK_SIZE,
			}
		)
		wandb.watch(model, log="all", log_freq=100)
	
	# Initial GPU memory check
	if torch.cuda.is_available():
		initial_memory = torch.cuda.memory_allocated() / 1024**3
		total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
		print(f"Initial GPU memory: {initial_memory:.1f}GB / {total_memory:.1f}GB")
	
	history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
	best_val_acc = 0.0
	best_epoch = 0
	consecutive_failures = 0
	max_consecutive_failures = 3
	
	# Early stopping parameters. Keyed off val_acc (the SAME metric that selects the saved
	# best_model.pth), so training is not cut short while val_acc is still climbing. Patience
	# raised 7->10 to ride out the noisy mid-training val plateau before the LR anneals, and
	# gated on the cosine having annealed (see EARLY_STOP_PATIENCE / EARLY_STOP_REQUIRE_ANNEAL).
	early_stop_patience = EARLY_STOP_PATIENCE
	early_stop_counter = 0
	best_select = -1.0   # best value of whichever metric SELECT_METRIC names
	best_val_loss = float('inf')
	
	start_epoch = 1
	if RESUME:
		ck = torch.load(RESUME, map_location=device, weights_only=False)
		model.load_state_dict(ck['model_state_dict'])
		optimizer.load_state_dict(ck['optimizer_state_dict'])
		scheduler.load_state_dict(ck['scheduler_state_dict'])
		scaler.load_state_dict(ck['scaler_state_dict'])
		history = ck['history']
		best_val_acc = ck['best_val_acc']; best_val_loss = ck['best_val_loss']
		best_select = ck.get('best_select', best_val_acc)
		best_epoch = ck['best_epoch']; early_stop_counter = ck['early_stop_counter']
		start_epoch = ck['epoch'] + 1
		if ck.get('num_epochs') != NUM_EPOCHS:
			print(f"  WARNING: checkpoint was scheduled for {ck.get('num_epochs')} epochs but "
				  f"--num_epochs is {NUM_EPOCHS}; the cosine will not line up.")
		print(f"Resumed from {RESUME}: continuing at epoch {start_epoch}/{NUM_EPOCHS} "
			  f"(best val_acc {best_val_acc:.4f} @ep{best_epoch})")

	print(f"Starting training on {device} with safety measures enabled...")

	# Add progress bar showing overall training progress
	model_best_fname = ''
	debug_break_flag = False
	epoch_pbar = tqdm(range(start_epoch, NUM_EPOCHS + 1), desc="Training Progress")
	for epoch in epoch_pbar:
		try:
			if debug_break_flag:
				print("Debug break flag activated. Stopping training loop.")
				break
			
			# Dynamic augmentation for training optimization
			augment_prob, augment_strength = get_augment_prob_and_strength(epoch, NUM_EPOCHS)
			train_dataset.augment_prob = augment_prob
			train_dataset.augment_strength = augment_strength
			
			train_loss, train_acc = train(epoch)

			val_loss, val_acc = validate(val_loader)
			
			# KPI monitoring (if enabled)
			if args.diagnose:
				monitor_kpi(model, epoch, save_dir=log_dir, loss=train_loss, accuracy=train_acc)

			# Check for numerical stability in results
			if not all(check_numerical_stability(torch.tensor([x]), f"metric_{i}") for i, x in enumerate([train_loss, train_acc, val_loss, val_acc])):
				print(f"Numerical instability in epoch {epoch} results. Skipping epoch.")
				torch.cuda.empty_cache()
				continue
			
			# Get current learning rate
			current_lr = optimizer.param_groups[0]['lr']
			
			# Update scheduler (Cosine Annealing - epoch based) - Skip during warmup
			old_lr = current_lr
			if epoch > WARMUP_EPOCHS:
				scheduler.step()  
			new_lr = optimizer.param_groups[0]['lr']
			log_epoch_metrics(csv_log_path, epoch, train_loss, train_acc, val_loss, val_acc, new_lr,
							  val_miou=LAST_VAL_MIOU)
			
			# Detect learning rate changes and log output
			if new_lr != old_lr:
				print(f"Learning rate changed from {old_lr:.2e} to {new_lr:.2e} (Cosine Annealing)")
			
			# Update history
			history['train_loss'].append(train_loss)
			history['train_acc'].append(train_acc)
			history['val_loss'].append(val_loss)
			history['val_acc'].append(val_acc)

			# Resumable checkpoint, rewritten every epoch. best_model.pth only holds weights
			# at the best val_acc, so killing a run mid-schedule used to cost the optimizer
			# moments and the cosine position -- a restart could only warm-start from the
			# best weights and had to begin a fresh LR cycle. This snapshot lets --resume
			# continue the SAME schedule exactly, which matters for runs measured in days.
			# Written to a temp file then renamed so a crash mid-write cannot corrupt it.
			ckpt_path = os.path.join(log_dir, 'checkpoint.pth')
			torch.save({'epoch': epoch,
						'model_state_dict': model.state_dict(),
						'optimizer_state_dict': optimizer.state_dict(),
						'scheduler_state_dict': scheduler.state_dict(),
						'scaler_state_dict': scaler.state_dict(),
						'history': history,
						'best_val_acc': best_val_acc,
						'best_select': best_select,
						'best_val_loss': best_val_loss,
						'best_epoch': best_epoch,
						'early_stop_counter': early_stop_counter,
						'num_epochs': NUM_EPOCHS}, ckpt_path + '.tmp')
			os.replace(ckpt_path + '.tmp', ckpt_path)
			
			# Track best model + early stopping on the SAME metric (val_acc). Previously the
			# counter keyed off val_loss while the checkpoint saved on val_acc, so a noisy
			# val_loss could stop training before val_acc peaked (a run stopped at epoch 40
			# though the best-acc epoch was ~47). best_val_loss is kept for logging only.
			if val_loss < best_val_loss:
				best_val_loss = val_loss

			# Which metric selects the saved checkpoint (and drives early stopping -- the two
			# must agree, see the note above). Default is val_miou: measured across this
			# project's runs val_acc correlates only +0.43 with the test mIoU we actually
			# report (+0.92 with test OA), and the run with the highest val_acc scored a
			# mediocre 53.67 mIoU while a much lower-val_acc run scored the best 56.45.
			# Accuracy is dominated by ceiling/floor/wall (65% of points); mIoU weights all
			# 13 classes equally, so selecting on accuracy saves the wrong epoch.
			select_now = LAST_VAL_MIOU if SELECT_METRIC == 'val_miou' else val_acc
			if select_now > best_select:
				best_select = select_now
				best_val_acc = val_acc
				best_epoch = epoch
				early_stop_counter = 0  # new best model found -> reset patience

				# Force memory cleanup before model saving
				torch.cuda.synchronize();
				torch.cuda.empty_cache()
				gc.collect()
				model_best_fname = os.path.join(log_dir, 'best_model.pth')
				torch.save(model.state_dict(), model_best_fname)
				torch.cuda.empty_cache()
				if COOLDOWN_SEC > 0:
					time.sleep(COOLDOWN_SEC)
			else:
				early_stop_counter += 1

			# Early stopping check. Patience alone is not enough: stopping while the cosine
			# is still high throws away the low-LR refinement phase that produces the final
			# gains, so the LR must have annealed to within EARLY_STOP_LR_TOL x LR_ETA_MIN
			# first. Until then a stalled val_acc only reports, it does not stop.
			if early_stop_counter >= early_stop_patience:
				anneal_done = (not EARLY_STOP_REQUIRE_ANNEAL) or (new_lr <= LR_ETA_MIN * EARLY_STOP_LR_TOL)
				if anneal_done:
					print(f"Early stopping triggered! No val_acc improvement for {early_stop_patience} epochs "
						  f"(LR {new_lr:.2e} has annealed to the {LR_ETA_MIN:.0e} floor).")
					break
				if early_stop_counter == early_stop_patience:
					print(f"val_acc stalled for {early_stop_patience} epochs, but LR is still {new_lr:.2e} "
						  f"(> {LR_ETA_MIN * EARLY_STOP_LR_TOL:.2e}); continuing so the cosine can anneal. "
						  f"Best so far: {best_val_acc:.4f} @ epoch {best_epoch}.")
			
			if USE_WANDB:
				WANDB_STEP += 1
				wandb.log({
					"epoch/train_loss": train_loss,
					"epoch/train_acc": train_acc,
					"epoch/val_loss": val_loss,
					"epoch/val_acc": val_acc,
					"epoch/loss_diff": val_loss - train_loss,
					"epoch/acc_diff": train_acc - val_acc,
					"epoch/lr": new_lr,
				}, step=WANDB_STEP)
			
			# Update epoch progress
			epoch_pbar.set_postfix({
				'Loss': f'{val_loss:.4f}',
				'Acc': f'{val_acc:.4f}',
				'Epoch': f'{epoch}/{NUM_EPOCHS}'
			})
			
		except Exception as e:
			print(f"Error in training loop at epoch {epoch}: {e}")
			torch.cuda.empty_cache()
			continue
	
	# Final model saving - save the best model at the end of training
	model_fname = os.path.join(log_dir, 'final_model.pth')
	torch.save(model.state_dict(), model_fname)
	print(f"Final model saved to: {model_fname}")
	
	# Save training summary
	save_training_summary(log_dir, history, best_epoch, best_val_acc)
	
	# Final GPU memory check
	if torch.cuda.is_available():
		final_memory = torch.cuda.memory_allocated() / 1024**3
		print(f"Final GPU memory: {final_memory:.1f}GB / {total_memory:.1f}GB")
		print(f"Memory used during training: {final_memory - initial_memory:.1f}GB")
	
	# Finish wandb run
	if USE_WANDB:
		wandb.finish()

	return model_best_fname

def test_model(model_path):
	"""Test trained model on TEST_AREA dataset"""
	print(f"\nTesting Model on {TEST_AREA}")
	print(f"Loading model from: {model_path}")
	
	# Load trained model (create model using current NUM_CLASSES)
	try:
		model = build_model(NUM_FEATURES, NUM_CLASSES, FEATURE_DIMS, CONTEXT_MODE, WIDTH_MULT,
		                    MID_TRANSFORMER, SAMPLER, ENC_CHANNELS, BOTTLENECK_DIM, arch=ARCH).to(device)
		model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
		model.eval()
		print("Model loaded successfully")
	except Exception as e:
		print(f"Error loading model: {e}")
		return
	
	# Load test dataset (TEST_AREA only)
	test_block_files = glob(os.path.join(BLOCK_DATA_PATH, f'*_{TEST_AREA}.pt'))
	if len(test_block_files) == 0:
		print(f"No test blocks found for {TEST_AREA}!")
		return
	
	print(f"Total test blocks: {len(test_block_files)}")
	
	# Create test dataset and loader
	test_dataset = BlockDataset(test_block_files, augment=False)
	test_loader = DataLoader(test_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_fn)
	
	# Test metrics - load class weights from configuration
	class_weights = get_class_weights(as_tensor=True, device=device)
	criterion_test = FocalLovaszLoss(class_weights=class_weights, gamma=FOCAL_GAMMA, ignore_index=-1,
									 focal_weight=1.0, lovasz_weight=1.0)
	correct_nodes, total_nodes, total_loss = 0, 0, 0.0
	valid_batches = 0
	
	# Per-class metrics. class_correct/class_total give accuracy (recall); class_union adds
	# the denominator for IoU. mIoU is the standard S3DIS metric (accuracy/mAcc alone are not
	# comparable to the literature). intersection == class_correct (TP); IoU = TP/(TP+FP+FN).
	class_correct = np.zeros(NUM_CLASSES)
	class_total = np.zeros(NUM_CLASSES)
	class_union = np.zeros(NUM_CLASSES)

	print(f"Starting evaluation on {len(test_loader)} batches...")
	pbar = tqdm(test_loader, desc=f'[Test {TEST_AREA}]')
	
	with torch.no_grad():
		for batch_idx, data in enumerate(pbar):
			if data is None:
				continue
			
			valid_batches += 1
			
			try:
				data = data.to(device)
				out = model(data)
				
				valid_mask = data.valid_mask.view(-1)
				valid_out = out[valid_mask]
				valid_y = data.y[valid_mask]
				
				loss = criterion_test(valid_out, valid_y)
				total_loss += loss.item()
				
				pred = valid_out.argmax(dim=-1)
				correct_nodes += pred.eq(valid_y).sum().item()
				total_nodes += valid_mask.sum().item()
				
				# Per-class accuracy (recall) + union for IoU
				for cls in range(NUM_CLASSES):
					cls_mask = (valid_y == cls)
					pred_mask = (pred == cls)
					if cls_mask.sum() > 0:
						class_total[cls] += cls_mask.sum().item()
						class_correct[cls] += (pred[cls_mask] == cls).sum().item()
					class_union[cls] += (cls_mask | pred_mask).sum().item()
				
				# Update progress
				current_acc = correct_nodes / total_nodes if total_nodes > 0 else 0
				current_loss = total_loss / valid_batches
				pbar.set_postfix({
					'Loss': f'{current_loss:.4f}',
					'Acc': f'{current_acc:.4f}',
					'Batch': f'{batch_idx+1}/{len(test_loader)}'
				})
				
			except Exception as e:
				print(f"Error in test batch {batch_idx}: {e}")
				continue
	
	if valid_batches == 0:
		print("No valid batches processed!")
		return
	
	# Calculate final metrics
	final_loss = total_loss / valid_batches
	final_acc = correct_nodes / total_nodes if total_nodes > 0 else 0
	
	print(f"Test Results on {TEST_AREA}:")
	print(f"Overall Loss: {final_loss:.4f}")
	print(f"Overall Accuracy: {final_acc:.4f} ({final_acc*100:.2f}%)")
	print(f"Total Points: {total_nodes}")
	print(f"Correct Predictions: {correct_nodes}")
	print(f"\nPer-Class Accuracy (recall) and IoU:")
	print(f"{'class':15s} {'acc(recall)':>12s} {'IoU':>9s}")

	# Prepare per-class results for JSON; collect present-class acc/IoU for the means.
	per_class_results = {}
	acc_list, iou_list = [], []
	for cls in range(NUM_CLASSES):
		cls_name = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else f"Class_{cls}"
		if class_total[cls] > 0:
			cls_acc = class_correct[cls] / class_total[cls]
			cls_iou = class_correct[cls] / class_union[cls] if class_union[cls] > 0 else 0.0
			acc_list.append(cls_acc)
			iou_list.append(cls_iou)
			print(f"{cls_name:15s} {cls_acc*100:11.2f}% {cls_iou*100:8.2f}%")
			per_class_results[cls_name] = {
				"accuracy": float(cls_acc),
				"iou": float(cls_iou),
				"correct": int(class_correct[cls]),
				"total": int(class_total[cls]),
				"union": int(class_union[cls])
			}
		else:
			print(f"{cls_name:15s} {'N/A':>12s} {'N/A':>9s}")
			per_class_results[cls_name] = {
				"accuracy": None,
				"iou": None,
				"correct": 0,
				"total": 0,
				"union": 0
			}

	# Standard segmentation summary metrics
	mAcc = float(np.mean(acc_list)) if acc_list else 0.0
	mIoU = float(np.mean(iou_list)) if iou_list else 0.0
	print("-" * 40)
	print(f"Overall Accuracy (OA)      : {final_acc*100:6.2f}%")
	print(f"Mean Class Accuracy (mAcc) : {mAcc*100:6.2f}%")
	print(f"Mean IoU (mIoU)            : {mIoU*100:6.2f}%")
	
	# Save test summary to JSON
	log_dir = os.path.dirname(model_path)
	test_summary = {
		"test_area": TEST_AREA,
		"model_path": model_path,
		"overall_metrics": {
			"loss": float(final_loss),
			"accuracy": float(final_acc),
			"mAcc": mAcc,
			"mIoU": mIoU,
			"total_points": int(total_nodes),
			"correct_predictions": int(correct_nodes)
		},
		"per_class_results": per_class_results,
		"test_config": {
			"num_classes": NUM_CLASSES,
			"batch_size": VAL_BATCH_SIZE,
			"total_blocks": len(test_block_files),
			"valid_batches": valid_batches
		}
	}
	
	test_summary_path = os.path.join(log_dir, "test_summary.json")
	with open(test_summary_path, 'w') as f:
		json.dump(test_summary, f, indent=2)
	
	print(f"\nTest summary saved to: {test_summary_path}")
	return final_loss, final_acc

def main():
	global PROCESSED_DATA_PATH, BLOCK_DATA_PATH, TRAIN_AREAS, TEST_AREA
	global NUM_EPOCHS, BATCH_SIZE, VAL_BATCH_SIZE, LEARNING_RATE, NUM_FEATURES, NUM_CLASSES, BLOCK_SIZE
	global USE_WANDB, COOLDOWN_SEC, FOCAL_GAMMA, RGB_DROPOUT_PROB, BLOCK_MODE
	global FEATURE_DIMS, NORMALS_PRESENT, COLUMN_WINDOW, COLUMN_STRIDE
	global EARLY_STOP_PATIENCE, EARLY_STOP_REQUIRE_ANNEAL
	global OVERSAMPLE_RARE, CONTEXT_MODE, WIDTH_MULT, MID_TRANSFORMER, SAMPLER
	global ROOM_DATA_PATH, ROOM_GRID, ROOM_MAX_POINTS, ROOM_LOOP, INIT_WEIGHTS, RESUME, SELECT_METRIC
	global ENC_CHANNELS, BOTTLENECK_DIM, ARCH, V2_KNN, V2_CURVES, V2_NEIGHBORS, V2_STENCIL, V2_DIFF, V2_BASE_GRID, V2_POOL_GRIDS, V2_DIRECTIONAL, V2_STENCIL_Z

	parser = argparse.ArgumentParser(description='PointEdgeSegNet Training')
	parser.add_argument('--config', type=str, default='model_params.json', help='Path to model configuration JSON file')
	parser.add_argument('--processed_data_path', default=PROCESSED_DATA_PATH, help='Processed data path')
	parser.add_argument('--block_data_path', default=BLOCK_DATA_PATH, help='Block data storage path')
	parser.add_argument('--train_areas', nargs='+', default=TRAIN_AREAS, help='Training areas')
	parser.add_argument('--test_area', default=TEST_AREA, help='Test area')
	parser.add_argument('--num_epochs', type=int, default=NUM_EPOCHS, help='Number of epochs')
	parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size')
	parser.add_argument('--val_batch_size', type=int, default=VAL_BATCH_SIZE, help='Validation batch size')
	parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE, help='Learning rate')
	parser.add_argument('--num_features', type=int, default=NUM_FEATURES, help='Number of features')
	parser.add_argument('--num_classes', type=int, default=13, help='Number of classes (default: 13 for S3DIS)')
	parser.add_argument('--block_size', type=int, default=BLOCK_SIZE, help='Block size')
	parser.add_argument('--diagnose', type=bool, default=False, help='Enable KPI monitoring and diagnostics')
	parser.add_argument('--test_model_path', type=str, default='', help='Path to trained model for testing') # D:\\projects\\point_edge_seg_net\\logs\\20260107_080233\\best_model.pth
	parser.add_argument('--no_wandb', action='store_true', help='Disable Weights & Biases logging')
	parser.add_argument('--cooldown_sec', type=float, default=COOLDOWN_SEC, help='Thermal cool-down seconds (0 disables all sleeps)')
	parser.add_argument('--focal_gamma', type=float, default=FOCAL_GAMMA, help='Focal Loss focusing parameter (shared by train/test)')
	parser.add_argument('--early_stop_patience', type=int, default=EARLY_STOP_PATIENCE,
						help='Epochs without a val_acc improvement before stopping. Only takes effect once the '
							 'cosine LR has annealed to the floor (see --allow_early_stop_before_anneal).')
	parser.add_argument('--allow_early_stop_before_anneal', action='store_true',
						help='Legacy behaviour: let patience stop training even while the LR is still high. Off by '
							 'default because it silently truncates the cosine schedule when --num_epochs is raised.')
	parser.add_argument('--aug_preset', type=str, default=AUG_PRESET, choices=['legacy', 'strong'],
						help="Augmentation preset. 'strong' = constant full-strength + 360deg yaw, isotropic "
							 "scale 0.8-1.2, mirroring, small tilts, colour drop 0.2, auto-contrast 0.2 (SOTA "
							 "practice). 'legacy' = the original decaying schedule (mean prob*strength 0.046).")
	parser.add_argument('--oversample_rare', type=float, default=OVERSAMPLE_RARE,
						help='Rare-class block oversampling exponent (0 = off). Blocks are drawn with weight '
							 'max(1, max class_weight present)**power, so sofa/board/column blocks appear more often. '
							 'Epoch length is unchanged.')
	parser.add_argument('--context_mode', type=str, default=CONTEXT_MODE, choices=['input', 'bottleneck'],
						help="Injection point for --block_context: 'bottleneck' (zero-init residual, safe default) "
							 "or 'input' (legacy constant channels; measured -6.4 mIoU on S3DIS).")
	parser.add_argument('--width_mult', type=float, default=WIDTH_MULT,
						help='Channel multiplier for the encoder/decoder (1.0 = historical architecture; '
							 '1.5 needs ~30GB VRAM at batch 10).')
	parser.add_argument('--room_data_path', type=str, default=ROOM_DATA_PATH,
						help='Cache dir for voxel-subsampled rooms (block_mode=room)')
	parser.add_argument('--room_grid', type=float, default=ROOM_GRID,
						help='Voxel size for room mode (0.04 m is the S3DIS standard)')
	parser.add_argument('--room_max_points', type=int, default=ROOM_MAX_POINTS,
						help='Training point budget per room crop (DeLA uses 30000)')
	parser.add_argument('--room_loop', type=int, default=ROOM_LOOP,
						help='Room-list repeats per epoch (each repeat = a different random crop). '
							 'S3DIS has ~170 train rooms, so loop=1 gives far too few optimizer steps.')
	parser.add_argument('--enc_channels', type=str, default=None,
						help='Comma-separated encoder channels, e.g. 128,256,320,384. Overrides the historical '
							 '64,128,256,512. Use to move capacity to the full-resolution stages: 74.6% of the '
							 'parameters currently sit in a bottleneck that sees 215 points while the stage that '
							 'sees all 8192 holds 0.4%.')
	parser.add_argument('--bottleneck_dim', type=int, default=None,
						help='Bottleneck transformer width (default: last encoder channel). Narrowing it is how '
							 'capacity is freed for the earlier stages.')
	parser.add_argument('--select_metric', type=str, default=SELECT_METRIC, choices=['val_miou', 'val_acc'],
						help="Metric that picks best_model.pth and drives early stopping. 'val_miou' is the "
							 "default because val_acc correlates only +0.43 with the test mIoU we report.")
	parser.add_argument('--resume', type=str, default=RESUME,
						help='Resume the SAME run from a logs/<run>/checkpoint.pth: restores weights, '
							 'optimizer, LR schedule, AMP scaler, history and epoch counter, then continues '
							 'to --num_epochs. Use this (not --init_weights) to continue an interrupted run.')
	parser.add_argument('--init_weights', type=str, default=INIT_WEIGHTS,
						help='Warm restart: load these weights before training instead of random init. '
							 'Use to extend a run that was still improving when its cosine schedule ended '
							 '(the architecture flags must match the checkpoint).')
	parser.add_argument('--sampler', type=str, default=SAMPLER, choices=['fps', 'grid'],
						help="Encoder stage sampler. 'grid' (voxel) is linear and spatially uniform; FPS is "
							 "quadratic and dominates cost above ~30k points (446x slower at 131k, measured). "
							 "Must match between training and evaluation.")
	parser.add_argument('--mid_transformer', action='store_true',
						help='Add a 1-layer attention block at the ~860-pt mid level (long-range within-block '
							 'context before the bottleneck). Architecture change: requires training from scratch.')
	parser.add_argument('--arch', type=str, default=ARCH,
						choices=['edgeconv', 'stencil', 'v1', 'v2'],
						help="Architecture: 'edgeconv' (legacy kNN graph) or 'stencil' (current voxel-"
							 "stencil backbone). 'v1'/'v2' are accepted as aliases. Checkpoints are not "
							 "interchangeable; must match between training and evaluation.")
	parser.add_argument('--v2_knn', type=int, default=V2_KNN,
						help='v2 only: neighbour window size, total across curves (architecture-defining)')
	parser.add_argument('--v2_curves', type=int, default=V2_CURVES, choices=[1, 2],
						help='v2 only: Morton curves unioned per stage; 2 gives two independent seams')
	parser.add_argument('--v2_neighbors', type=str, default=V2_NEIGHBORS, choices=['serial', 'stencil'],
						help="v2 only: neighbour source — 'serial' Morton window (approx) or 'stencil' "
							 "exact voxel-offset lookup (sparse-conv kernel map; architecture-defining)")
	parser.add_argument('--v2_stencil', type=int, default=V2_STENCIL, choices=[1, 2],
						help='v2 stencil radius: 1 -> K=27 (~9 surface nbrs), 2 -> K=125 (~25, ~= kNN-32)')
	parser.add_argument('--v2_diff', action='store_true',
						help='v2: add decomposed feature-difference term (EdgeConv gradient cue, k-x cheaper)')
	parser.add_argument('--v2_base_grid', type=float, default=V2_BASE_GRID,
						help='v2: input voxel size — MUST equal the cache voxel grid (0.04 default, 0.02 for fine)')
	parser.add_argument('--v2_pool_grids', type=str, default=V2_POOL_GRIDS,
						help='v2: comma list of per-stage pooling voxel sizes')
	parser.add_argument('--v2_directional', action='store_true',
						help='v2 stencil: cylindrical (z-level x ring) anisotropic aggregation — yaw-'
							 'augmentation-safe approximation of sparse-conv per-offset kernels')
	parser.add_argument('--v2_stencil_z', type=int, default=V2_STENCIL_Z,
						help='v2 stencil: vertical reach in voxels (0 = same as --v2_stencil). '
							 'Columns/pipes span many z-levels; E10 tests a taller receptive field.')
	parser.add_argument('--rgb_dropout', type=float, default=RGB_DROPOUT_PROB, help='Prob. of zeroing RGB per block for RGB-free robustness')
	parser.add_argument('--block_mode', type=str, default=BLOCK_MODE, choices=['grid', 'column', 'room'], help="Block partitioning: 'grid' (legacy) or 'column' (overlapping full-height, preserves context)")
	parser.add_argument('--column_window', type=float, default=COLUMN_WINDOW, help='Column mode only: XY window side length (m). Larger => fewer blocks. VRAM is unaffected (blocks stay block_size points).')
	parser.add_argument('--column_stride', type=float, default=COLUMN_STRIDE, help='Column mode only: XY step between columns (m). Set == column_window for no overlap (block count ~ grid); < window for overlap.')
	parser.add_argument('--block_context', dest='block_context', action='store_true', default=None,
						help='Append a per-block buffered-context descriptor (verticality/horizontality/curvature/density + z-histogram aggregated over neighbouring blocks within context_buffer m). Overrides features.use_block_context in the config. Requires fresh blocks (block cache path is auto-suffixed with _ctx).')
	parser.add_argument('--no_block_context', dest='block_context', action='store_false',
						help='Force-disable the block-context descriptor (overrides config), e.g. for the A/B baseline run.')
	parser.add_argument('--context_buffer', type=float, default=None,
						help='Block-context only: buffer distance (m) added around each block\'s XY bounds (default from config: 2.0).')
	parser.add_argument('--context_bins', type=int, default=None,
						help='Block-context only: number of z-histogram bins in the descriptor (default from config: 4).')
	args = parser.parse_args()

	# Resolve runtime toggles
	USE_WANDB = _WANDB_AVAILABLE and (not args.no_wandb)
	if args.no_wandb:
		print("wandb disabled by --no_wandb")
	elif not _WANDB_AVAILABLE:
		print("wandb not installed -> logging disabled (pip install wandb to enable)")
	COOLDOWN_SEC = args.cooldown_sec
	FOCAL_GAMMA = args.focal_gamma
	RGB_DROPOUT_PROB = args.rgb_dropout
	apply_aug_preset(args.aug_preset)   # may raise RGB_DROPOUT_PROB to 0.2 (colour drop)
	BLOCK_MODE = args.block_mode
	COLUMN_WINDOW = args.column_window
	COLUMN_STRIDE = args.column_stride
	EARLY_STOP_PATIENCE = args.early_stop_patience
	EARLY_STOP_REQUIRE_ANNEAL = not args.allow_early_stop_before_anneal
	OVERSAMPLE_RARE = args.oversample_rare
	CONTEXT_MODE = args.context_mode
	WIDTH_MULT = args.width_mult
	MID_TRANSFORMER = args.mid_transformer
	SAMPLER = args.sampler
	ARCH = args.arch
	V2_KNN = args.v2_knn
	V2_CURVES = args.v2_curves
	V2_NEIGHBORS = args.v2_neighbors
	V2_STENCIL = args.v2_stencil
	V2_DIFF = args.v2_diff
	V2_BASE_GRID = args.v2_base_grid
	V2_POOL_GRIDS = args.v2_pool_grids
	V2_DIRECTIONAL = args.v2_directional
	V2_STENCIL_Z = args.v2_stencil_z
	INIT_WEIGHTS = args.init_weights
	RESUME = args.resume
	SELECT_METRIC = args.select_metric
	ENC_CHANNELS = tuple(int(c) for c in args.enc_channels.split(',')) if args.enc_channels else None
	BOTTLENECK_DIM = args.bottleneck_dim
	ROOM_DATA_PATH = args.room_data_path
	ROOM_GRID = args.room_grid
	ROOM_MAX_POINTS = args.room_max_points
	ROOM_LOOP = args.room_loop

	# Load model configuration from JSON file
	try:
		print(f"Loading model configuration from: {args.config}")
		config = load_model_config(args.config)
		
		# Override with config file values if not explicitly provided via command line
		# Check if user provided explicit command line values
		import sys
		if '--num_classes' not in sys.argv and 'num_classes' in config:
			args.num_classes = config['num_classes']
		if '--num_features' not in sys.argv and 'num_features' in config:
			args.num_features = config['num_features']
		if '--block_size' not in sys.argv and 'block_size' in config:
			args.block_size = config['block_size']
			
		print(f"Dataset: {config.get('dataset_name', 'Custom')}")
		print(f"Classes: {config['class_names']}")
		
	except Exception as e:
		print(f"Warning: Could not load config file: {e}")
		print("Using default configuration...")

	# Apply block-context CLI overrides into the active config BEFORE resolving the
	# feature spec, so config file and command line stay interchangeable (A/B testing:
	# run once with --no_block_context, once with --block_context, compare test_summary).
	cfg = get_model_config()
	feats_cfg = cfg.get('features') or {}
	cfg['features'] = feats_cfg
	if args.block_context is not None:
		feats_cfg['use_block_context'] = args.block_context
	if args.context_buffer is not None:
		feats_cfg['context_buffer'] = args.context_buffer
	if args.context_bins is not None:
		feats_cfg['context_bins'] = args.context_bins

	# Update global variables
	PROCESSED_DATA_PATH = args.processed_data_path
	BLOCK_DATA_PATH = args.block_data_path
	TRAIN_AREAS = args.train_areas
	TEST_AREA = args.test_area
	NUM_EPOCHS = args.num_epochs
	BATCH_SIZE = args.batch_size
	VAL_BATCH_SIZE = args.val_batch_size
	LEARNING_RATE = args.learning_rate
	NUM_FEATURES = args.num_features
	NUM_CLASSES = args.num_classes
	BLOCK_SIZE = args.block_size

	# Resolve the configurable feature layout (domain-agnostic). This determines the
	# actual feature dimensions and the model's input group split; NUM_FEATURES is taken
	# from the enabled feature groups so it always matches what the pipeline produces.
	try:
		spec = resolve_feature_config(get_model_config())
		FEATURE_DIMS = (spec['geo_dim'], spec['rgb_dim'], spec['spatial_dim'], spec['context_dim'])
		NORMALS_PRESENT = spec['use_normals']
		NUM_FEATURES = spec['num_features']
		print(f"  Feature layout: {FEATURE_DIMS} (normals={spec['use_normals']}, curv={spec['use_curvature']}, "
			  f"rgb={spec['use_rgb']}, spatial={spec['use_spatial']}, scale={spec['spatial_scale']}, "
			  f"block_context={spec['use_block_context']}"
			  + (f", buffer={spec['context_buffer']}m, bins={spec['context_bins']}" if spec['use_block_context'] else "")
			  + ")")
		# Keep context and non-context block caches separate so A/B runs never mix: when
		# block context is on and the user did not pick a custom path, suffix the default.
		import sys
		if spec['use_block_context'] and '--block_data_path' not in sys.argv:
			BLOCK_DATA_PATH = BLOCK_DATA_PATH.rstrip('/\\') + '_ctx'
			print(f"  block_context on -> block cache path: {BLOCK_DATA_PATH}")
	except Exception as e:
		print(f"Warning: feature spec resolution failed ({e}); using default {FEATURE_DIMS}.")

	print(f"Training configuration:")
	print(f"  Epochs: {NUM_EPOCHS}, Batch size: {BATCH_SIZE}, Learning rate: {LEARNING_RATE}")
	print(f"  Train areas: {TRAIN_AREAS}, Test area: {TEST_AREA}")
	print(f"  Block size: {BLOCK_SIZE}, Validation Batch Size: {VAL_BATCH_SIZE}, Features: {NUM_FEATURES}, Classes: {NUM_CLASSES}")
	
	# Start training
	if len(args.test_model_path) > 0:
		test_model(args.test_model_path)
	else:
		final_model_fname = run_training(args)
		test_model(final_model_fname)

if __name__ == '__main__':
	main()