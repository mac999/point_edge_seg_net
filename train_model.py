# Title: train_model (version 0.2)
# Author: taewook kang (laputa9999@gmail.com)
# Date: 2025-09-21
# Purpose: Trains and validates the PointEdgeSegNet model.
# Dependencies: torch, torch_geometric, matplotlib, scikit-learn, tqdm

import os, torch, torch.optim as optim, json, csv, argparse, time, numpy as np, random, wandb
import torch.nn as nn, torch.nn.functional as F, matplotlib.pyplot as plt, gc
from torch_geometric.loader import DataLoader
from dotenv import load_dotenv
from glob import glob
from model import PointEdgeSegNet
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from datetime import datetime
from diagnose_kpi_grad import monitor_kpi
from data_processing import (
    apply_torch_color_augmentation, 
    apply_torch_enhanced_color_augmentation,
    CLASS_NAMES,
    extract_features_from_room_data
)
from torch.cuda.amp import autocast, GradScaler

# Load wandb API key from .env file (single line)
load_dotenv(os.path.join('logs', '.env'))

random.seed(42)  

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
NUM_EPOCHS = 80  # Extended for full convergence
BATCH_SIZE = 14  # Increased from 16 for better gradient estimation
VAL_BATCH_SIZE = 18
LEARNING_RATE = 0.003  # Increased from 0.002 for faster learning
NUM_FEATURES = 10  # Training: normals(3) + curvature(1) + RGB(3) + spatial(3)
NUM_CLASSES = 13
BLOCK_SIZE = 8192  # Number of points per block. TBD
block_grid_min_coords = np.array([-50.0, 0.0, 0.0])  # base point for normalization

# Settings for validation loss stabilization
GRADIENT_CLIP_VALUE = 8.0  # Gradient clipping (increased for better learning)
WARMUP_EPOCHS = 3  # Reduced for faster ramp-up

# GPU safety settings
GPU_MEMORY_THRESHOLD = 0.85  # 85% GPU memory usage threshold
MAX_GRADIENT_NORM = 20.0     # Maximum gradient norm threshold (increased)

# Gradient Accumulation settings (virtual batch size increase)
ACCUMULATION_STEPS = 4  # Effective batch size: 14 × 4 = 56

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
						'loss_diff', 'acc_diff', 'learning_rate'])
	
	return log_dir, csv_log_path

def log_epoch_metrics(csv_path, epoch, train_loss, train_acc, val_loss, val_acc, lr):
	"""Save epoch metrics to CSV file"""
	loss_diff = val_loss - train_loss
	acc_diff = train_acc - val_acc
	
	with open(csv_path, 'a', newline='') as f:
		writer = csv.writer(f)
		writer.writerow([epoch, f"{train_loss:.6f}", f"{train_acc:.6f}", 
						f"{val_loss:.6f}", f"{val_acc:.6f}", 
						f"{loss_diff:.6f}", f"{acc_diff:.6f}", f"{lr:.8f}"])

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
				
				# Grid Hash spatial partitioning with coordinate range adjustment
				# S3DIS coordinates: X: -15.609~-15.331, Y: 37.738~39.569, Z: 0.299~2.657
				grid_resolution = 0.5  # Resolution for narrow coordinate ranges
				
				# Normalize coordinates to handle negative values and optimize grid distribution
				normalized_points = points - block_grid_min_coords
				grid_coords = np.floor(normalized_points / grid_resolution).astype(np.int32)
				
				# Ensure all grid coordinates are non-negative
				grid_coords = np.maximum(grid_coords, 0)
				
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
						block_x = torch.FloatTensor(features[final_indices])
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

						if block_counter % 100 == 0: # Brief pause for GPU cooling
							time.sleep(10.0)  
		
			except Exception as e:
				print(f"Error processing file {pt_file}: {e}")
				continue
	
	end_time = time.time()
	preprocessing_time = end_time - start_time
	print(f"Preprocessing completed in {preprocessing_time:.2f} seconds")
	print(f"Total blocks created: {block_counter}")
	print(f"Average time per block: {preprocessing_time/block_counter:.4f} seconds")
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
	"""Minimal augmentation for fastest convergence"""
	if epoch <= max_epochs * 0.1:  
		return 0.3, 0.4  # Very low augmentation
	elif epoch <= max_epochs * 0.5:
		return 0.2, 0.3  # Minimal augmentation
	return 0.1, 0.2  # Almost no augmentation

class BlockDataset(torch.utils.data.Dataset):
	def __init__(self, file_list, augment=False, augment_prob=0.0, augment_strength=1.0):
		self.file_list = file_list
		self.augment = augment
		self.augment_prob = augment_prob  # Probability of applying augmentation
		self.augment_strength = augment_strength  # Strength of augmentation
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
					# Convert 10D storage format to 10D training format
					# No conversion needed - use all features directly
					# Format: geo(4) + RGB(3) + spatial(3) = 10D
					
					# No color augmentation for pure geometric+color learning
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

# Data loader preparation
# Run preprocessing if block data doesn't exist
if not os.path.exists(BLOCK_DATA_PATH) or len(glob(os.path.join(BLOCK_DATA_PATH, '*.pt'))) == 0:
	print("Block data not found. Running preprocessing...")
	preprocess_dataset()

# Separate block files by area
all_block_files = glob(os.path.join(BLOCK_DATA_PATH, '*.pt'))
train_block_files = []
test_block_files = []

print("Categorizing block files by area...")
for block_file in tqdm(all_block_files, desc="Categorizing blocks"):
	filename = os.path.basename(block_file)
	if TEST_AREA in filename:
		test_block_files.append(block_file)
	else:
		# Blocks belonging to TRAIN_AREAS
		for area in TRAIN_AREAS:
			if area in filename:
				train_block_files.append(block_file)
				break

# Split training blocks into 8:2 using train_test_split
# Validate block files before using them
# print("Validating training block files...")
# train_block_files = validate_block_files(train_block_files)
# print("Validating test block files...")
# test_block_files = validate_block_files(test_block_files)

# random.shuffle(train_block_files)
# random.shuffle(test_block_files)
# train_files = train_block_files
# val_files = test_block_files 
train_files, val_files = train_test_split(train_block_files, test_size=0.2, random_state=42)

print(f"Total training blocks: {len(train_files)}")
print(f"Total validation blocks: {len(val_files)}")
print(f"Total test blocks: {len(test_block_files)}")
print(f"Block size: {BLOCK_SIZE} points per block")

train_dataset = BlockDataset(train_files, augment=True, augment_prob=0.5, augment_strength=0.5)  # Reduced
val_dataset = BlockDataset(val_files, augment=False, augment_prob=0.0, augment_strength=0.0)

# Apply collate_fn to DataLoader (set num_workers=0 to prevent Windows multiprocessing issues)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_fn)

# DataLoader
print(f"Training dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
print(f"Training batches: {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")

if len(train_loader) == 0:
	raise ValueError("Training loader is empty! Check your data files.")
if len(val_loader) == 0:
	raise ValueError("Validation loader is empty! Check your data files.")

model = PointEdgeSegNet(num_features=NUM_FEATURES, num_classes=NUM_CLASSES).to(device)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)  # ✅ 0.005 → 0.01

# Per-class weights for Focal Loss (near-uniform with slight rare class boost)
# Strategy: Minimal intervention, let Focal Loss handle imbalance naturally
class_weights = torch.tensor([
	0.7,  # 0: ceiling (important structural element)
	0.7,  # 1: floor (important structural element)
	0.7,  # 2: wall (important structural element)
	0.8,  # 3: beam
	0.9,  # 4: column (rare - 2.39%, needs boost)
	0.85, # 5: window
	0.9,  # 6: door (rare - 11.81%, needs boost)
	0.75, # 7: table
	0.75, # 8: chair
	0.9,  # 9: sofa (rare - 36.13%, needs boost)
	0.8,  # 10: bookcase
	0.8,  # 11: board
	0.75  # 12: clutter
], device=device)

# Use Focal Loss with minimal gamma (let class weights do the work)
criterion = FocalLoss(class_weights=class_weights, gamma=1.5, ignore_index=-1)  # Reduced gamma 2.0 -> 1.5

# More conservative learning rate scheduler
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=NUM_EPOCHS*2, eta_min=5e-4  # Slower decay, higher minimum LR
)

scaler = GradScaler()

def train(epoch):
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
			
			if batch_idx % 200 == 0:
				time.sleep(10.0)
			
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
			if hasattr(model, 'last_gates'):
				gates = model.last_gates
				current_batch_acc = pred.eq(valid_y).sum().item() / valid_mask.sum().item()
				wandb.log({
					"batch/gate_geo": gates[:, 0].mean().item(),
					"batch/gate_rgb": gates[:, 1].mean().item(),
					"batch/gate_spatial": gates[:, 2].mean().item(),
					"batch/loss": loss.item(),
					"batch/accuracy": current_batch_acc,
				}, step=epoch * len(train_loader) + batch_idx)

			# Memory cleanup every 10 batches to prevent accumulation
			if batch_idx % 10 == 0:
				torch.cuda.empty_cache()
				# Force garbage collection for Python objects
				import gc
				gc.collect()
			
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
	return avg_loss, accuracy

def run_training(args=None):
	# Add freeze_support for Windows multiprocessing support
	import multiprocessing
	multiprocessing.freeze_support()
	
	# Log setup
	log_dir, csv_log_path = setup_logging()
	print(f"Logging to directory: {log_dir}")
	
	# Initialize wandb (online mode for cloud sync)
	wandb.init(
		project="pointedge-s3dis",
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
	
	# Early stopping parameters - more aggressive
	early_stop_patience = 7   
	early_stop_counter = 0
	best_val_loss = float('inf')
	
	print(f"Starting training on {device} with safety measures enabled...")

	# Add progress bar showing overall training progress
	model_best_fname = ''
	debug_break_flag = False
	epoch_pbar = tqdm(range(1, NUM_EPOCHS + 1), desc="Training Progress")
	global_step = 0  
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
			global_step += len(train_loader)  
			
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
			log_epoch_metrics(csv_log_path, epoch, train_loss, train_acc, val_loss, val_acc, new_lr)
			
			# Detect learning rate changes and log output
			if new_lr != old_lr:
				print(f"Learning rate changed from {old_lr:.2e} to {new_lr:.2e} (Cosine Annealing)")
			
			# Update history
			history['train_loss'].append(train_loss)
			history['train_acc'].append(train_acc)
			history['val_loss'].append(val_loss)
			history['val_acc'].append(val_acc)
			
			# Track best performance and early stopping
			if val_loss < best_val_loss:
				best_val_loss = val_loss
				early_stop_counter = 0
			else:
				early_stop_counter += 1
				
			if val_acc > best_val_acc:
				best_val_acc = val_acc
				best_epoch = epoch

				# Force memory cleanup before model saving
				torch.cuda.synchronize(); 
				torch.cuda.empty_cache()
				gc.collect()
				model_best_fname = os.path.join(log_dir, 'best_model.pth')
				torch.save(model.state_dict(), model_best_fname)
				torch.cuda.empty_cache()
				time.sleep(2.0)  
			
			# Early stopping check
			if early_stop_counter >= early_stop_patience:
				print(f"Early stopping triggered! No improvement for {early_stop_patience} epochs.")
				break
			
			wandb.log({
				"epoch/train_loss": train_loss,
				"epoch/train_acc": train_acc,
				"epoch/val_loss": val_loss,
				"epoch/val_acc": val_acc,
				"epoch/loss_diff": val_loss - train_loss,
				"epoch/acc_diff": train_acc - val_acc,
				"epoch/lr": new_lr,
			}, step=global_step)  # global_step
			
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
	wandb.finish()

	return model_best_fname

def test_model(model_path):
	"""Test trained model on TEST_AREA dataset"""
	print(f"\nTesting Model on {TEST_AREA}")
	print(f"Loading model from: {model_path}")
	
	# Load trained model
	try:
		model = PointEdgeSegNet(num_features=NUM_FEATURES, num_classes=NUM_CLASSES).to(device)
		model.load_state_dict(torch.load(model_path, map_location=device))
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
	
	# Test metrics - use same class weights as training
	class_weights = torch.tensor([
		0.1, 0.1, 0.1, 0.5, 0.9, 0.7, 0.8, 0.4, 0.4, 0.8, 0.5, 0.6, 0.5
	], device=device)
	criterion_test = FocalLoss(class_weights=class_weights, gamma=2.5, ignore_index=-1)
	correct_nodes, total_nodes, total_loss = 0, 0, 0.0
	valid_batches = 0
	
	# Per-class metrics
	class_correct = np.zeros(NUM_CLASSES)
	class_total = np.zeros(NUM_CLASSES)
	
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
				
				# Per-class accuracy
				for cls in range(NUM_CLASSES):
					cls_mask = (valid_y == cls)
					if cls_mask.sum() > 0:
						class_total[cls] += cls_mask.sum().item()
						class_correct[cls] += (pred[cls_mask] == cls).sum().item()
				
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
	print(f"\nPer-Class Accuracy:")
	
	# Prepare per-class results for JSON
	per_class_results = {}
	for cls in range(NUM_CLASSES):
		cls_name = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else f"Class_{cls}"
		if class_total[cls] > 0:
			cls_acc = class_correct[cls] / class_total[cls]
			print(f"{cls_name:15s}: {cls_acc*100:6.2f}% ({int(class_correct[cls])}/{int(class_total[cls])})")
			per_class_results[cls_name] = {
				"accuracy": float(cls_acc),
				"correct": int(class_correct[cls]),
				"total": int(class_total[cls])
			}
		else:
			print(f"{cls_name:15s}: N/A (no samples)")
			per_class_results[cls_name] = {
				"accuracy": None,
				"correct": 0,
				"total": 0
			}
	
	# Save test summary to JSON
	log_dir = os.path.dirname(model_path)
	test_summary = {
		"test_area": TEST_AREA,
		"model_path": model_path,
		"overall_metrics": {
			"loss": float(final_loss),
			"accuracy": float(final_acc),
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
	
	parser = argparse.ArgumentParser(description='PointEdgeSegNet Training')
	parser.add_argument('--processed_data_path', default=PROCESSED_DATA_PATH, help='Processed data path')
	parser.add_argument('--block_data_path', default=BLOCK_DATA_PATH, help='Block data storage path')
	parser.add_argument('--train_areas', nargs='+', default=TRAIN_AREAS, help='Training areas')
	parser.add_argument('--test_area', default=TEST_AREA, help='Test area')
	parser.add_argument('--num_epochs', type=int, default=NUM_EPOCHS, help='Number of epochs')
	parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size')
	parser.add_argument('--val_batch_size', type=int, default=VAL_BATCH_SIZE, help='Validation batch size')
	parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE, help='Learning rate')
	parser.add_argument('--num_features', type=int, default=NUM_FEATURES, help='Number of features')
	parser.add_argument('--num_classes', type=int, default=NUM_CLASSES, help='Number of classes')
	parser.add_argument('--block_size', type=int, default=BLOCK_SIZE, help='Block size')
	parser.add_argument('--diagnose', type=bool, default=False, help='Enable KPI monitoring and diagnostics')
	parser.add_argument('--test_model_path', type=str, default='', help='Path to trained model for testing') # D:\\projects\\point_edge_seg_net\\logs\\20260107_080233\\best_model.pth
	args = parser.parse_args()
	
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