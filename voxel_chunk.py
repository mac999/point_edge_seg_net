# voxel_chunk.py
# Large-cloud chunking pipeline: voxelize -> KD median split -> halo -> fixed-size blocks.
#
# This replaces "cut a 2 m column and keep 8192 full-density points" with the recipe every
# scalable method uses. Four ideas, each measured or cited:
#
# 1. VOXELIZE FIRST. Our blocks spend the point budget on redundant density instead of
#    area: 8192 points over 4 m^2 = 2048 pts/m^2, while DeLA (7.0M params, 74.1 mIoU) uses
#    998 pts/m^2 over 30 m^2. Measured here: at a 4 cm grid a 6 m crop costs only 4.1x the
#    points of a 2 m crop while covering 9x the area. Indoor objects do not need 2 cm
#    resolution, but doors/columns/sofas DO need context past 2 m.
#
# 2. KD MEDIAN SPLIT, NOT OCTREE. Both bound the points per chunk, but an octree splits
#    8-ways all-or-nothing, so a node just over budget becomes 8 sparse children. Measured
#    on a 21.3M-point bridge (4.27M voxels, 30k budget):
#        uniform 8 m tiles : 40 chunks, std/mean 1.12, 29 chunks OVER budget (OOM)
#        octree           : 619 chunks, std/mean 1.06, budget utilisation 23%
#        KD median split  : 256 chunks, std/mean 0.00, budget utilisation 56%
#    KD dominates on every axis. Chunks have variable physical size, which the literature
#    says is FINE (SensatUrban: constant-point-count beats constant-volume by 7-9.5 mIoU)
#    as long as you never RESCALE a chunk -- we only translate (pos - centroid).
#
# 3. HALO OF REAL POINTS. Each chunk is fed the points within `halo` metres of its bounds,
#    but only its own (core) points are supervised and scored. A point at the chunk edge
#    therefore still sees real neighbours, instead of a truncated neighbourhood. This is
#    KPConv's `test_radius_ratio=0.7` (discard the outer 30% of each sphere) and
#    TreeLearn's 35 m input / 8 m prediction window, expressed as a mask.
#    NOTE this is the OPPOSITE of the block-context descriptor we tried twice and which
#    lost 6.4 / 4.7 mIoU: that broadcast a per-chunk CONSTANT summary, which cannot
#    separate points inside its own chunk. A halo adds real per-point neighbours.
#
# 4. FIXED POINT COUNT. The KD split targets a point count, not a volume, so every chunk
#    fills the same memory. Core points are never dropped; only the halo is subsampled if
#    the total would exceed block_size.
#
# Output is the SAME format the existing block pipeline produces
# (Data with x/pos/y/valid_mask/num_valid_points/area, `block_<n>_<room>__<tag>__<area>.pt`),
# so train_model.py consumes it with no changes -- just point --block_data_path at it.

import os
import numpy as np
import torch
from glob import glob
from tqdm import tqdm
from torch_geometric.data import Data

from data_processing import compute_surface_variation, spatial_split_is_val, EPSILON

def invariant_geo_features(points: np.ndarray, normals: np.ndarray, knn: int = 15) -> np.ndarray:
    """linearity, planarity, verticality -- a rotation- and scale-invariant replacement for
    the stored `spatial` group (density / angular-anisotropy / local-structure).

    WHY: the stored spatial channels are computed once at preprocessing in the cloud's
    original orientation off an axis-aligned grid, so they do NOT survive augmentation --
    measured MAE 0.077 / 0.162 / 0.145 against their own std of 0.240 / 0.242 / 0.300, i.e.
    up to 67% of the feature's range under a rotation. Both experiments that disturbed them
    hurt the same classes: corrupting them (Stage A) cost board -13.3 / window -6.9, and
    removing them entirely (voxel-chunk, 7D) cost board -7.9 / window -5.8 / bookcase -9.6.
    Those are exactly the planar, structured classes such descriptors are supposed to serve.

    The eigenvalue ratios below carry the same kind of information but are invariant by
    construction (l0 <= l1 <= l2 are the local covariance eigenvalues):
        linearity   = (l2 - l1) / l2   edges, wires, chair legs
        planarity   = (l1 - l0) / l2   boards, walls, shelves
        verticality = 1 - |n_z|        gravity cue; invariant under the Z-rotation we apply
    Ratios of eigenvalues are unchanged by rotation AND by isotropic scaling, so the strong
    augmentation preset stays valid. These are the standard Weinmann/Hackel eigenfeatures
    and the same ones Superpoint Transformer feeds its encoder.

    Note density is deliberately dropped: after voxelization every occupied voxel holds
    exactly one point, so a density channel carries almost no signal.
    """
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    pcd.estimate_covariances(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn))
    ev = np.clip(np.linalg.eigvalsh(np.asarray(pcd.covariances)), 0.0, None)  # ascending
    l2 = ev[:, 2] + EPSILON
    lin = np.clip((ev[:, 2] - ev[:, 1]) / l2, 0.0, 1.0)
    pla = np.clip((ev[:, 1] - ev[:, 0]) / l2, 0.0, 1.0)
    ver = np.clip(1.0 - np.abs(np.asarray(normals)[:, 2]), 0.0, 1.0)
    return np.stack([lin, pla, ver], axis=1).astype(np.float32)

def voxel_first_index(points: np.ndarray, grid: float) -> np.ndarray:
    """One representative point index per occupied voxel (lowest index), sorted."""
    keys = np.floor(np.asarray(points, dtype=np.float64) / grid).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return np.sort(idx)

def kd_median_chunks(points: np.ndarray, max_points: int, split_axes=(0, 1)):
    """Recursively split at the MEDIAN of the longest axis until every leaf <= max_points.

    Median splitting is what makes the chunks balanced: each split puts exactly half the
    points on each side, so leaf sizes end up within a factor of two of each other
    (measured std/mean 0.00 on real data) instead of the octree's 1.06.

    split_axes defaults to XY ONLY, so every chunk keeps the full vertical extent. Allowing
    a Z split would cut floors off ceilings and slice walls/doors/columns in half -- exactly
    the failure the original cubic-cell blocker had and that full-height columns fixed.
    Measured: with Z splitting enabled a chunk of 11,160 points collapsed to a 2.1 x 1.3 m
    footprint (a thin slab) instead of spanning a real floor area.
    """
    leaves, stack = [], [np.arange(len(points))]
    while stack:
        idx = stack.pop()
        if len(idx) <= max_points:
            if len(idx):
                leaves.append(idx)
            continue
        sub = points[idx]
        spans = sub.max(axis=0) - sub.min(axis=0)
        axis = int(max(split_axes, key=lambda a: spans[a]))
        med = np.median(sub[:, axis])
        left = sub[:, axis] <= med
        if left.all() or (~left).all():      # degenerate (all coords equal on this axis)
            leaves.append(idx)
            continue
        stack.append(idx[left])
        stack.append(idx[~left])
    return leaves

def chunk_with_halo(points: np.ndarray, core_idx: np.ndarray, halo: float,
                    block_size: int, rng) -> tuple:
    """Return (indices, num_core) for one chunk: core points first, then halo context.

    `indices[:num_core]` are the supervised points; the rest are context the network sees
    but is never scored on. If core + halo exceeds block_size the HALO is subsampled --
    core points are never dropped, so coverage stays exact.
    """
    # Halo is an XY ring only: chunks already span the full height, so there is nothing
    # above or below to add, and a Z buffer would just duplicate the chunk's own points.
    lo = points[core_idx][:, :2].min(axis=0) - halo
    hi = points[core_idx][:, :2].max(axis=0) + halo
    inside = np.all((points[:, :2] >= lo) & (points[:, :2] <= hi), axis=1)
    inside[core_idx] = False                       # halo = neighbourhood minus core
    halo_idx = np.nonzero(inside)[0]

    n_core = len(core_idx)
    room = max(block_size - n_core, 0)
    if len(halo_idx) > room:
        halo_idx = rng.choice(halo_idx, room, replace=False) if room else halo_idx[:0]

    idx = np.concatenate([core_idx, halo_idx])
    if len(idx) < block_size:                      # pad by repeating (never supervised)
        pad = rng.choice(idx, block_size - len(idx), replace=True)
        idx = np.concatenate([idx, pad])
    return idx.astype(np.int64), int(n_core)


def voxelize_and_featurize(points, features, grid, neighbor_knn=15,
                           invariant_geo=False, feature_dim=None):
    """THE voxel-level feature pipeline. Both the training cache and the evaluator must
    call this -- nothing else may compute chunk features.

    It exists because they used to duplicate the logic, and the duplicate drifted: the
    cache builder gained an `invariant_geo` branch while the evaluator kept reading the
    stored density/anisotropy/structure channels. The same three columns then carried
    different quantities at train and test time, and the model scored 2.47 mIoU -- near
    chance -- with nothing in the logs to suggest a mismatch rather than a bad model.

    Returns (voxel_indices, voxel_points, voxel_features).
    """
    vidx = voxel_first_index(points, grid)
    vp = points[vidx]
    vf = features[vidx].copy()
    if feature_dim is not None and vf.shape[1] > feature_dim:
        vf = vf[:, :feature_dim]
    if vf.shape[1] > 3:                    # curvature is scale-dependent: redo at voxel scale
        vf[:, 3] = compute_surface_variation(vp, knn=neighbor_knn)
    if invariant_geo:
        if vf.shape[1] < 10:               # widen 7D -> 10D if needed
            vf = np.concatenate([vf, np.zeros((len(vf), 10 - vf.shape[1]), np.float32)], axis=1)
        vf[:, 7:10] = invariant_geo_features(vp, vf[:, 0:3], knn=neighbor_knn)
    return vidx, vp, vf

def chunk_layout(vp, core_max, halo, block_size, seed=0):
    """THE chunk layout. Same reason as above: train and test must partition identically,
    down to the halo subsampling draw, or the evaluated chunks are not the trained ones.
    Returns a list of (indices, num_core).
    """
    rng = np.random.default_rng(seed)
    return [chunk_with_halo(vp, core, halo, block_size, rng)
            for core in kd_median_chunks(vp, core_max)]

def build_room_chunks(points, features, labels, grid, core_max, halo, block_size, rng,
                      invariant_geo=False, feature_dim=None, neighbor_knn=15):
    """Voxelize a room, KD-split it, and emit fixed-size chunks with halo context.

    Thin wrapper over voxelize_and_featurize + chunk_layout so the cache and the evaluator
    cannot diverge. `rng` is accepted for backwards compatibility but the layout uses its
    own seeded generator, which is what makes train and test chunks identical.
    """
    _, vp, vf = voxelize_and_featurize(points, features, grid, neighbor_knn=neighbor_knn,
                                       invariant_geo=invariant_geo, feature_dim=feature_dim)
    vy = labels[voxel_first_index(points, grid)]
    return [(vp[idx], vf[idx], vy[idx], n_core)
            for idx, n_core in chunk_layout(vp, core_max, halo, block_size)]

def prepare_chunk_cache(processed_data_path, out_path, areas, test_area,
                        grid=0.04, core_max=8192, halo=1.0, block_size=16384,
                        feature_dim=None, seed=0, invariant_geo=False):
    """Write a block cache in the standard format so train_model.py can use it unchanged."""
    if os.path.exists(out_path) and len(glob(os.path.join(out_path, '*.pt'))) > 0:
        print(f"Chunk cache already present at {out_path}; skipping.")
        return
    os.makedirs(out_path, exist_ok=True)
    rng = np.random.default_rng(seed)
    n_chunk = 0
    stats = []
    for area in tqdm(areas, desc="Chunking areas"):
        for pt_file in sorted(glob(os.path.join(processed_data_path, area, '*.pt'))):
            try:
                d = torch.load(pt_file, weights_only=False)
                p = d.pos.numpy().astype(np.float32)
                f = d.x.numpy().astype(np.float32)
                y = d.y.numpy()
                if feature_dim is not None and f.shape[1] > feature_dim:
                    f = f[:, :feature_dim]
                room = os.path.splitext(os.path.basename(pt_file))[0]
                room = ''.join(c if (c.isalnum() or c in '-') else '-' for c in room)
                for cp, cf, cy, n_core in build_room_chunks(p, f, y, grid, core_max,
                                                            halo, block_size, rng,
                                                            invariant_geo=invariant_geo):
                    vm = torch.zeros(len(cp), dtype=torch.bool)
                    vm[:n_core] = True             # only core points are supervised
                    ly = torch.from_numpy(cy).long()
                    ly[n_core:] = -1               # halo + padding excluded from the loss
                    centroid = cp[:n_core, :2].mean(axis=0)
                    tag = ('test' if area == test_area else
                           ('val' if spatial_split_is_val(room, centroid, super_size=4.0,
                                                          val_ratio=0.2, seed=42) else 'train'))
                    torch.save(Data(x=torch.from_numpy(cf), pos=torch.from_numpy(cp), y=ly,
                                    valid_mask=vm, num_valid_points=n_core, area=area),
                               os.path.join(out_path, f"block_{n_chunk:012d}_{room}__{tag}__{area}.pt"))
                    stats.append((n_core, len(cp)))
                    n_chunk += 1
            except Exception as e:
                print(f"Error chunking {pt_file}: {e}")
    if stats:
        c = np.array([s[0] for s in stats])
        print(f"Chunk cache at {out_path}: {n_chunk} chunks | core pts mean {c.mean():.0f} "
              f"std/mean {c.std()/c.mean():.2f} min {c.min()} max {c.max()} | block_size {block_size}")

def predict_room_chunks(model, room_pt, device, num_classes, grid=0.04, core_max=12288,
                        halo=1.0, block_size=20480, feature_dim=None, batch_chunks=4,
                        neighbor_knn=15, seed=0, invariant_geo=False):
    """Score every ORIGINAL point of a room using the SAME chunking as training.

    Only CORE predictions are kept (the halo is context, never scored -- this is
    KPConv's `test_radius_ratio=0.7` idea expressed exactly). Cores tile the voxel cloud
    without gaps or overlap, so each voxel gets exactly one prediction, which is then
    propagated to full resolution by a single nearest-neighbour query.
    """
    from scipy.spatial import cKDTree
    from torch_geometric.data import Batch

    d = torch.load(room_pt, weights_only=False)
    points = d.pos.numpy().astype(np.float32)
    feats = d.x.numpy().astype(np.float32)
    labels = d.y.numpy()

    _, vp, vf = voxelize_and_featurize(points, feats, grid, neighbor_knn=neighbor_knn,
                                       invariant_geo=invariant_geo, feature_dim=feature_dim)
    chunks = chunk_layout(vp, core_max, halo, block_size, seed=seed)

    vpred = np.full(len(vp), -1, dtype=np.int64)
    for s in range(0, len(chunks), batch_chunks):
        part = chunks[s:s + batch_chunks]
        data_list = [Data(x=torch.from_numpy(np.ascontiguousarray(vf[i])),
                          pos=torch.from_numpy(np.ascontiguousarray(vp[i])))
                     for i, _ in part]
        batch = Batch.from_data_list(data_list).to(device)
        with torch.no_grad():
            pred = model(batch).argmax(dim=-1).cpu().numpy()
        off = 0
        for i, n_core in part:
            vpred[i[:n_core]] = pred[off:off + n_core]
            off += len(i)
    assert (vpred >= 0).all(), f"unscored voxels in {room_pt}"

    pred = vpred[cKDTree(vp).query(points, k=1, workers=-1)[1]]
    valid = (labels >= 0) & (labels < num_classes)
    conf = np.bincount(labels[valid] * num_classes + pred[valid],
                       minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    return conf, int(valid.sum()), len(chunks)
