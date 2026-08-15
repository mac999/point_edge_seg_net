# room_pipeline.py
# Whole-room ("room mode") data pipeline: the Stage-E change.
#
# WHY THIS EXISTS
# The block pipeline feeds the network 2 m x 2 m columns of 8192 full-density points.
# Measured consequences on this repo:
#   - the receptive field is hard-capped at 2.0 m (kNN radii 0.15/0.26/0.47/0.96 m over the
#     four stages sum past the block, so the block, not the network, is the limit);
#   - the classes that need wider context collapse into `wall`: column -> wall 77.7%,
#     door -> wall 42.4%, board -> wall 28.0%;
#   - the val->test gap sat at a constant 8.4 points across every architecture tried
#     (baseline, +block-context, width 1.5), i.e. neither capacity nor features move it.
#
# Every S3DIS SOTA model instead trains on WHOLE ROOMS that were first voxel-subsampled,
# and tests on the whole room. DeLA (arXiv 2308.16532) is the closest to our architecture
# -- 7.0M params, plain kNN k=24, no fancy attention -- and reaches 74.1 mIoU with exactly
# this recipe: grid-subsample the room at 0.04 m, center-crop to the 30,000 nearest points
# around a random seed for training, process the whole room at test time.
#
# The enabling measurement: voxelizing at 4 cm costs 4.1x more points for 9x more area
# (2 m crop 8,192 pts -> 6 m crop 34,107 pts), and a whole room is only 47-89k points. What
# made this impossible before was FPS, which is quadratic (24.1 s at 131k points); with
# model.grid_subsample it is 54 ms, i.e. 446x, and flat in N.
#
# Prediction is made at voxel resolution and propagated back to every original point by
# nearest neighbour, which is what Pointcept and DeLA both do, so the reported mIoU stays
# comparable to published Area-5 numbers.

import os
import hashlib
import numpy as np
import torch
from glob import glob
from tqdm import tqdm
from torch_geometric.data import Data

from data_processing import compute_surface_variation, augment_training_block

def voxel_first_index(points: np.ndarray, grid: float) -> np.ndarray:
    """Indices of one representative point per occupied voxel (lowest index wins).

    Deterministic and O(N log N) (a sort inside np.unique). Returns sorted indices so the
    voxelized cloud keeps the original point ordering.
    """
    keys = np.floor(np.asarray(points, dtype=np.float64) / grid).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return np.sort(idx)

def room_is_val(room_key: str, val_ratio: float = 0.2, seed: int = 42) -> bool:
    """Deterministic room-level train/val split.

    Room mode makes the whole room one sample, so the block-level spatial split no longer
    applies; splitting by room is the natural leak-free choice (a room is never partly in
    train and partly in val).
    """
    h = hashlib.md5(f"{seed}|{room_key}".encode()).hexdigest()
    return (int(h[:8], 16) % 10000) < int(val_ratio * 10000)

def preprocess_rooms(processed_data_path: str, room_data_path: str, areas, test_area: str,
                     grid: float = 0.04, neighbor_knn: int = 15, recompute_curvature: bool = True):
    """Voxel-subsample every processed room once and cache it for room-mode training.

    Curvature is RECOMPUTED after subsampling: it is a local surface-variation statistic, so
    a value estimated at the original ~2 cm spacing does not describe the 4 cm cloud the
    network actually sees. Everything else (normals, RGB, spatial) is a per-point attribute
    and survives subsampling unchanged.

    Cached file name: `room_<room>__<train|val>__<area>.pt` (test-area rooms are tagged
    `test`), matching the block pipeline's convention so area filtering works the same way.
    """
    if os.path.exists(room_data_path) and len(glob(os.path.join(room_data_path, '*.pt'))) > 0:
        print(f"Room cache already present at {room_data_path}; skipping preprocessing.")
        return
    os.makedirs(room_data_path, exist_ok=True)

    kept_total = orig_total = 0
    for area in tqdm(areas, desc="Voxelizing areas"):
        area_dir = os.path.join(processed_data_path, area)
        if not os.path.isdir(area_dir):
            continue
        for pt_file in sorted(glob(os.path.join(area_dir, '*.pt'))):
            try:
                d = torch.load(pt_file, weights_only=False)
                points = d.pos.numpy().astype(np.float32)
                features = d.x.numpy().astype(np.float32).copy()
                labels = d.y.numpy()

                idx = voxel_first_index(points, grid)
                orig_total += len(points)
                kept_total += len(idx)
                vp, vf, vy = points[idx], features[idx], labels[idx]

                if recompute_curvature and vf.shape[1] > 3:
                    vf[:, 3] = compute_surface_variation(vp, knn=neighbor_knn)

                room = os.path.splitext(os.path.basename(pt_file))[0]
                room = ''.join(c if (c.isalnum() or c in '-') else '-' for c in room)
                tag = 'test' if area == test_area else ('val' if room_is_val(f"{area}/{room}") else 'train')
                torch.save(Data(x=torch.from_numpy(vf), pos=torch.from_numpy(vp),
                                y=torch.from_numpy(vy).long()),
                           os.path.join(room_data_path, f"room_{room}__{tag}__{area}.pt"))
            except Exception as e:
                print(f"Error voxelizing {pt_file}: {e}")
                continue
    if orig_total:
        print(f"Room cache built at {room_data_path}: {orig_total:,} -> {kept_total:,} points "
              f"({kept_total / orig_total * 100:.1f}% at {grid} m grid)")

def center_crop_indices(points: np.ndarray, max_points: int, rng) -> np.ndarray:
    """DeLA's training crop: the `max_points` points nearest a random seed point.

    A contiguous spatial neighbourhood (not a random subset), so the crop looks like a real
    partial scan and the network still sees a metres-wide context.
    """
    n = len(points)
    if n <= max_points:
        return np.arange(n)
    seed = int(rng.integers(n))
    d = ((points - points[seed]) ** 2).sum(axis=1)
    idx = np.argpartition(d, max_points - 1)[:max_points]
    return np.sort(idx)

class RoomDataset(torch.utils.data.Dataset):
    """One voxelized room per item, randomly center-cropped to a point budget.

    Args:
        file_list: cached room .pt paths
        max_points: training point budget per sample (DeLA uses 30,000 for S3DIS)
        loop: how many times the room list is repeated per epoch. S3DIS has only ~170
              training rooms, so one pass is ~40 optimizer steps -- far too few. Every
              reference recipe repeats the scene list instead (Pointcept and DeLA both use
              loop=30 with 100 epochs); each repeat yields a DIFFERENT random crop and
              augmentation, so it is genuine extra data, not duplicated batches.
        augment/aug_kwargs: passed through to data_processing.augment_training_block
    """
    def __init__(self, file_list, max_points=30000, augment=False, augment_prob=1.0,
                 augment_strength=1.0, feature_dims=(4, 3, 3, 0), normals_present=True,
                 rgb_dropout_prob=0.0, aug_kwargs=None, seed=0, loop=1, feature_dim=None):
        # feature_dim: keep only the first N feature columns. The stored layout is
        # [normals(3) | curvature(1) | rgb(3) | spatial(3)], so feature_dim=7 drops the
        # SPATIAL group -- which is the only group that is NOT invariant to the geometric
        # augmentations. Measured on a real room: rotating the cloud changes the stored
        # density/anisotropy/structure channels by 0.077/0.162/0.145 MAE against their own
        # std of 0.240/0.242/0.300, i.e. up to 67% of the feature's dynamic range, because
        # they are computed once at preprocessing in the original orientation and never
        # recomputed. With the legacy schedule only ~10% of samples were rotated so the
        # corruption was diluted; at full-strength augmentation it hits every sample.
        # Normals co-rotate, curvature is eigenvalue-based (rotation invariant) and colour
        # is unaffected, so dropping the spatial group makes strong augmentation consistent.
        self.feature_dim = feature_dim
        self.loop = max(1, int(loop))
        self.file_list = list(file_list)
        self.max_points = int(max_points)
        self.augment = augment
        self.augment_prob = augment_prob
        self.augment_strength = augment_strength
        self.feature_dims = feature_dims
        self.normals_present = normals_present
        self.rgb_dropout_prob = rgb_dropout_prob
        self.aug_kwargs = aug_kwargs or {}
        self._rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.file_list) * self.loop

    def __getitem__(self, i):
        i = i % len(self.file_list)   # `loop` repeats give different crops/augmentations
        try:
            d = torch.load(self.file_list[i], weights_only=False)
            pos = d.pos.numpy()
            idx = center_crop_indices(pos, self.max_points, self._rng)
            fx = d.x[idx]
            if self.feature_dim is not None and fx.shape[1] > self.feature_dim:
                fx = fx[:, :self.feature_dim]
            data = Data(x=fx.clone(), pos=d.pos[idx].clone(), y=d.y[idx].clone())
            # Room mode has no padding: every point is real, so valid_mask is all-True.
            data.valid_mask = torch.ones(len(idx), dtype=torch.bool)
            data.num_valid_points = len(idx)
            if self.augment and self.augment_prob > 0 and self._rng.random() < self.augment_prob:
                data = augment_training_block(
                    data, strength=self.augment_strength,
                    geo_dim=self.feature_dims[0], rgb_dim=self.feature_dims[1],
                    normals_present=self.normals_present,
                    rgb_dropout_prob=self.rgb_dropout_prob, **self.aug_kwargs)
            return data
        except Exception as e:
            print(f"Error loading room {self.file_list[i]}: {e}")
            return None

def split_rooms_by_area(room_files, train_areas, test_area):
    """Split cached room files into (train, val, test) by the tags in their file names."""
    train, val, test = [], [], []
    for f in room_files:
        name = os.path.basename(f)
        if test_area in name:
            test.append(f)
        elif any(a in name for a in train_areas):
            (val if '__val__' in name else train).append(f)
    return train, val, test

def predict_room_full(model, room_pt_original, device, spec, num_classes,
                      grid=0.04, max_points=200000, neighbor_knn=15,
                      recompute_curvature=True, views=((1.0, False),), batch_rooms=1,
                      feature_dim=None):
    """Predict every ORIGINAL point of a room (voxelize -> model -> nearest-neighbour propagate).

    This mirrors the standard test protocol: the network runs on the voxelized cloud (the
    same representation it was trained on), then each original point takes the prediction of
    its nearest voxel representative, so metrics are computed over the full-resolution cloud
    and stay comparable to published Area-5 numbers.

    Rooms larger than `max_points` voxels are processed in overlapping spatial chunks and the
    per-voxel softmax is summed, so no point is ever left unpredicted.
    """
    from scipy.spatial import cKDTree
    from torch_geometric.data import Batch

    d = torch.load(room_pt_original, weights_only=False)
    points = d.pos.numpy().astype(np.float32)
    features = d.x.numpy().astype(np.float32).copy()
    labels = d.y.numpy()

    vidx = voxel_first_index(points, grid)
    vp, vf = points[vidx], features[vidx]
    if recompute_curvature and vf.shape[1] > 3:
        vf[:, 3] = compute_surface_variation(vp, knn=neighbor_knn)
    if feature_dim is not None and vf.shape[1] > feature_dim:
        vf = vf[:, :feature_dim]   # must match the training feature layout

    nv = len(vp)
    votes = np.zeros((nv, num_classes), dtype=np.float32)

    # Chunking: split along the longest XY axis into overlapping slabs when the room is
    # larger than the memory budget. 50% overlap so points near a seam still get a
    # prediction made with full surrounding context from at least one chunk.
    if nv <= max_points:
        chunks = [np.arange(nv)]
    else:
        axis = int(np.argmax(vp[:, :2].max(axis=0) - vp[:, :2].min(axis=0)))
        order = np.argsort(vp[:, axis])
        n_chunk = int(np.ceil(nv / max_points)) * 2 - 1     # 50% overlap
        step = max(1, nv // max(n_chunk, 1))
        chunks = []
        for s in range(0, nv, step):
            c = order[s:s + max_points]
            if len(c):
                chunks.append(c)
            if s + max_points >= nv:
                break

    for scale, flip_x in views:
        for c in chunks:
            cp = vp[c] * np.float32(scale)
            cf = vf[c]
            if flip_x:
                cp = cp.copy(); cp[:, 0] = -cp[:, 0]
                if spec.get('use_normals'):
                    cf = cf.copy(); cf[:, 0] = -cf[:, 0]
            data = Data(x=torch.from_numpy(np.ascontiguousarray(cf)),
                        pos=torch.from_numpy(np.ascontiguousarray(cp)))
            batch = Batch.from_data_list([data]).to(device)
            with torch.no_grad():
                probs = torch.softmax(model(batch).float(), dim=-1).cpu().numpy()
            np.add.at(votes, c, probs)

    assert (votes.sum(axis=1) > 0).all(), f"unpredicted voxels in {room_pt_original}"
    vpred = votes.argmax(axis=1)

    # Nearest-neighbour propagation from voxel representatives to all original points.
    tree = cKDTree(vp)
    _, nn = tree.query(points, k=1, workers=-1)
    pred = vpred[nn]

    valid = (labels >= 0) & (labels < num_classes)
    conf = np.bincount(labels[valid] * num_classes + pred[valid],
                       minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    return conf, int(valid.sum()), nv, len(chunks)
