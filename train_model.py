# Title: train_model (version 0.1)
# Author: taewook kang (laputa9999@gmail.com)
# Date: 2025-09-21
# Purpose: Trains and validates the PointEdgeSegNet model.
# Dependencies: torch, torch_geometric, matplotlib, scikit-learn, tqdm

import os, torch, torch.optim as optim, json, csv
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from glob import glob
from model import PointEdgeSegNet
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from datetime import datetime

# GPU optimization settings (optional)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
	print(f"GPU available: {torch.cuda.get_device_name(0)}")
	print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
	# Memory efficiency settings
	torch.backends.cudnn.benchmark = True  # Optimized for fixed size input
	torch.cuda.empty_cache()  # Clear GPU memory
else:
	print("CUDA not available, using CPU")

# Configuration (adjusted for 8GB VRAM environment)
PROCESSED_DATA_PATH = './processed_s3dis'
BLOCK_DATA_PATH = './block_s3dis'  # Block data storage path
TRAIN_AREAS = ['Area_1', 'Area_2', 'Area_3', 'Area_4', 'Area_6']
TEST_AREA = 'Area_5'
NUM_EPOCHS = 30
BATCH_SIZE = 4  # Increased from 2 to 4 (improved validation stability)
LEARNING_RATE = 0.001
NUM_FEATURES = 9 
NUM_CLASSES = 13
BLOCK_SIZE = 8192  # Number of points per block

# Settings for validation loss stabilization
GRADIENT_CLIP_VALUE = 1.0  # Gradient clipping
WARMUP_EPOCHS = 3  # Learning rate warmup

def setup_logging():
	"""Log file setup and initialization"""
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	log_dir = f"logs_{timestamp}"
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
			"final_train_loss": history['train_loss'][-1],
			"final_val_loss": history['val_loss'][-1]
		},
		"training_history": history
	}
	
	summary_path = os.path.join(log_dir, "training_summary.json")
	with open(summary_path, 'w') as f:
		json.dump(summary, f, indent=2)
	
	print(f"Training summary saved to: {summary_path}")

def preprocess_dataset():
	"""
	Split existing processed_s3dis pt files into 8192 point blocks
	and save them to the block_s3dis folder.
	"""
	import time
	start_time = time.time()
	
	print("Starting dataset preprocessing...")
	os.makedirs(BLOCK_DATA_PATH, exist_ok=True)
	
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
				num_points = data.x.shape[0]
				num_blocks = (num_points + BLOCK_SIZE - 1) // BLOCK_SIZE
				
				# Split into 8192 point blocks
				for i in tqdm(range(0, num_points, BLOCK_SIZE), 
							 desc=f"Creating blocks from {os.path.basename(pt_file)}", 
							 leave=False, 
							 total=num_blocks):
					end_idx = min(i + BLOCK_SIZE, num_points)
					
					# Extract current block
					block_x = data.x[i:end_idx]
					block_pos = data.pos[i:end_idx]
					block_y = data.y[i:end_idx]
					
					current_block_size = block_x.shape[0]
					
					# Apply padding if needed
					if current_block_size < BLOCK_SIZE:
						padding_size = BLOCK_SIZE - current_block_size
						
						# Feature vector padding (fill with zeros)
						padding_x = torch.zeros(padding_size, block_x.shape[1], dtype=block_x.dtype)
						block_x = torch.cat([block_x, padding_x], dim=0)
						
						# Position padding (fill with last point position)
						if current_block_size > 0:
							last_pos = block_pos[-1:].repeat(padding_size, 1)
						else:
							last_pos = torch.zeros(padding_size, 3, dtype=block_pos.dtype)
						block_pos = torch.cat([block_pos, last_pos], dim=0)
						
						# Label padding (fill with ignore_index value -1)
						padding_y = torch.full((padding_size,), -1, dtype=block_y.dtype)
						block_y = torch.cat([block_y, padding_y], dim=0)
					
					# Generate valid point mask
					valid_mask = torch.ones(BLOCK_SIZE, dtype=torch.bool)
					if current_block_size < BLOCK_SIZE:
						valid_mask[current_block_size:] = False
					
					# Create torch_geometric Data object
					from torch_geometric.data import Data
					block_data = Data(
						x=block_x,
						pos=block_pos,
						y=block_y,
						valid_mask=valid_mask,
						num_valid_points=current_block_size,
						area=area  # Store which area this block came from
					)
					
					# Save block
					block_filename = f"block_{block_counter:06d}_{area}.pt"
					torch.save(block_data, os.path.join(BLOCK_DATA_PATH, block_filename))
					block_counter += 1
					
			except Exception as e:
				print(f"Error processing file {pt_file}: {e}")
				continue
	
	end_time = time.time()
	preprocessing_time = end_time - start_time
	print(f"Preprocessing completed in {preprocessing_time:.2f} seconds")
	print(f"Total blocks created: {block_counter}")
	print(f"Average time per block: {preprocessing_time/block_counter:.4f} seconds")
	return block_counter

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
		try:
			os.remove(corrupted_file)
			print(f"Removed corrupted file: {corrupted_file}")
		except Exception as e:
			print(f"Failed to remove {corrupted_file}: {e}")
	
	print(f"Valid files: {len(valid_files)}, Removed corrupted files: {len(corrupted_files)}")
	return valid_files

# Split training blocks into 8:2 using train_test_split
# Validate block files before using them
print("Validating training block files...")
train_block_files = validate_block_files(train_block_files)
print("Validating test block files...")
test_block_files = validate_block_files(test_block_files)

train_files, val_files = train_test_split(train_block_files, test_size=0.2, random_state=42)

print(f"Total training blocks: {len(train_files)}")
print(f"Total validation blocks: {len(val_files)}")
print(f"Total test blocks: {len(test_block_files)}")
print(f"Block size: {BLOCK_SIZE} points per block")

# Simplified Dataset class (for block data)
class BlockDataset(torch.utils.data.Dataset):
	def __init__(self, file_list):
		self.file_list = file_list
	
	def __len__(self):
		return len(self.file_list)
	
	def __getitem__(self, idx):
		try:
			return torch.load(self.file_list[idx], weights_only=False)
		except Exception as e:
			print(f"Error loading file {self.file_list[idx]}: {e}") # TBD. error handling
			return None

def collate_fn(batch):
	# Filter out None objects that caused errors
	batch = [d for d in batch if d is not None]
	if not batch:
		return None
	
	from torch_geometric.data import Batch
	return Batch.from_data_list(batch)


train_dataset = BlockDataset(train_files)
val_dataset = BlockDataset(val_files)

# Apply collate_fn to DataLoader (set num_workers=0 to prevent Windows multiprocessing issues)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_fn)

# DataLoader 검증
print(f"Training dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
print(f"Training batches: {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")

if len(train_loader) == 0:
	raise ValueError("Training loader is empty! Check your data files.")
if len(val_loader) == 0:
	raise ValueError("Validation loader is empty! Check your data files.")

model = PointEdgeSegNet(num_features=NUM_FEATURES, num_classes=NUM_CLASSES).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
# Set ignore_index=-1 to ignore labels (-1) of padded points
criterion = torch.nn.NLLLoss(ignore_index=-1)

# Add learning rate scheduler (validation loss stabilization)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
	optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
)

def train(epoch):
	model.train()
	
	# DataLoader 유효성 검사
	if len(train_loader) == 0:
		print("Warning: Training loader is empty!")
		return 0.0, 0.0
	
	pbar = tqdm(train_loader, desc=f'Epoch {epoch:02d}/{NUM_EPOCHS} [Training]')
	total_loss, correct_nodes, total_nodes = 0, 0, 0
	valid_batches = 0
	
	for batch_idx, data in enumerate(pbar):
		if data is None: 
			print(f"Warning: Skipping corrupted batch {batch_idx} in training")
			continue
			
		valid_batches += 1
		data = data.to(device)
		optimizer.zero_grad()
		out = model(data)
		
		# Include only valid points in loss calculation
		valid_mask = data.valid_mask.view(-1)  # Flatten batch dimension
		valid_out = out[valid_mask]
		valid_y = data.y[valid_mask]
		
		loss = criterion(valid_out, valid_y)
		loss.backward()
		
		# Stabilize training with gradient clipping
		torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_VALUE)
		
		optimizer.step()
		total_loss += loss.item()
		
		# Calculate accuracy (valid points only)
		pred = valid_out.argmax(dim=-1)
		correct_nodes += pred.eq(valid_y).sum().item()
		total_nodes += valid_mask.sum().item()
		
		# More detailed progress bar information
		current_acc = correct_nodes / total_nodes if total_nodes > 0 else 0
		current_loss = total_loss / valid_batches if valid_batches > 0 else 0
		pbar.set_postfix({
			'Loss': f'{current_loss:.4f}',
			'Acc': f'{current_acc:.4f}',
			'Batch': f'{batch_idx+1}/{len(train_loader)}'
		})
	
	# Handle case where no valid batches were processed
	if valid_batches == 0:
		print("Warning: No valid batches were processed in training!")
		return 0.0, 0.0
		
	return total_loss / valid_batches, correct_nodes / total_nodes

def validate(loader):
	model.eval()
	loader_name = 'Validation' if loader == val_loader else 'Test'
	
	# DataLoader 유효성 검사
	if len(loader) == 0:
		print(f"Warning: {loader_name} loader is empty!")
		return 0.0, 0.0
	
	pbar = tqdm(loader, desc=f'[{loader_name}]')
	correct_nodes, total_nodes, total_loss = 0, 0, 0.0
	valid_batches = 0
	
	with torch.no_grad():
		for batch_idx, data in enumerate(pbar):
			if data is None: 
				print(f"Warning: Skipping corrupted batch {batch_idx} in {loader_name.lower()}")
				continue
				
			valid_batches += 1
			data = data.to(device)
			out = model(data)
			
			# Include only valid points in loss and accuracy calculation
			valid_mask = data.valid_mask.view(-1)  # Flatten batch dimension
			valid_out = out[valid_mask]
			valid_y = data.y[valid_mask]
			
			# Add validation loss calculation
			loss = criterion(valid_out, valid_y)
			total_loss += loss.item()
			
			pred = valid_out.argmax(dim=-1)
			correct_nodes += pred.eq(valid_y).sum().item()
			total_nodes += valid_mask.sum().item()
			
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

if __name__ == '__main__':
	# Add freeze_support for Windows multiprocessing support
	import multiprocessing
	multiprocessing.freeze_support()
	
	# Log setup
	log_dir, csv_log_path = setup_logging()
	print(f"Logging to directory: {log_dir}")
	
	history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
	best_val_acc = 0.0
	best_epoch = 0
	
	print(f"Starting training on {device}...")

	# Add progress bar showing overall training progress
	epoch_pbar = tqdm(range(1, NUM_EPOCHS + 1), desc="Training Progress")
	for epoch in epoch_pbar:
		train_loss, train_acc = train(epoch)
		val_loss, val_acc = validate(val_loader)
		
		# Get current learning rate
		current_lr = optimizer.param_groups[0]['lr']
		
		# Update scheduler (based on validation loss)
		old_lr = current_lr
		scheduler.step(val_loss)
		new_lr = optimizer.param_groups[0]['lr']
		
		# Detect learning rate changes and log output
		if new_lr != old_lr:
			print(f"Learning rate reduced from {old_lr:.2e} to {new_lr:.2e}")
		
		# Update history
		history['train_loss'].append(train_loss)
		history['train_acc'].append(train_acc)
		history['val_loss'].append(val_loss)
		history['val_acc'].append(val_acc)
		
		# Track best performance
		if val_acc > best_val_acc:
			best_val_acc = val_acc
			best_epoch = epoch
			# Save best performance model
			torch.save(model.state_dict(), os.path.join(log_dir, 'best_model.pth'))
		
		# Save CSV log
		log_epoch_metrics(csv_log_path, epoch, train_loss, train_acc, val_loss, val_acc, new_lr)
		
		# Update epoch progress bar
		epoch_pbar.set_postfix({
			'Train Loss': f'{train_loss:.4f}',
			'Train Acc': f'{train_acc:.4f}',
			'Val Loss': f'{val_loss:.4f}',
			'Val Acc': f'{val_acc:.4f}',
			'LR': f'{new_lr:.2e}'
		})
		
		print(f'\nEpoch: {epoch:02d}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, LR: {new_lr:.2e}')

	# Save final model and logs
	torch.save(model.state_dict(), os.path.join(log_dir, 'final_model.pth'))
	save_training_summary(log_dir, history, best_epoch, best_val_acc)
	
	print(f"\nBest validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")

	# Display all train/val loss and accuracy with 4 subplots
	fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
	
	# Training and Validation Loss
	ax1.plot(history['train_loss'], label='Train Loss', color='blue')
	ax1.plot(history['val_loss'], label='Validation Loss', color='red')
	ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.set_title('Training and Validation Loss'); ax1.legend()
	ax1.grid(True)
	
	# Training and Validation Accuracy  
	ax2.plot(history['train_acc'], label='Train Accuracy', color='blue')
	ax2.plot(history['val_acc'], label='Validation Accuracy', color='red')
	ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy'); ax2.set_title('Training and Validation Accuracy'); ax2.legend()
	ax2.grid(True)
	
	# Loss difference (Overfitting detection)
	loss_diff = [val - train for val, train in zip(history['val_loss'], history['train_loss'])]
	ax3.plot(loss_diff, label='Val Loss - Train Loss', color='orange')
	ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
	ax3.set_xlabel('Epoch'); ax3.set_ylabel('Loss Difference'); ax3.set_title('Overfitting Detection (Val - Train Loss)'); ax3.legend()
	ax3.grid(True)
	
	# Accuracy difference
	acc_diff = [train - val for val, train in zip(history['val_acc'], history['train_acc'])]
	ax4.plot(acc_diff, label='Train Acc - Val Acc', color='green')
	ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
	ax4.set_xlabel('Epoch'); ax4.set_ylabel('Accuracy Difference'); ax4.set_title('Generalization Gap (Train - Val Acc)'); ax4.legend()
	ax4.grid(True)
	
	plt.tight_layout()
	plt.savefig(os.path.join(log_dir, 'training_plots.png'))
	plt.show()

	print("\n--- Final Test Performance ---")
	test_dataset = BlockDataset(test_block_files)
	test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_fn)
	test_acc = validate(test_loader)
	print(f"Final Test Accuracy on {TEST_AREA}: {test_acc:.4f}")