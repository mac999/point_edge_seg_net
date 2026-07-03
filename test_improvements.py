# Title: test_improvements
# Purpose: Standalone smoke tests for the PointEdgeSegNet improvements.
#          Verifies the numpy/torch logic WITHOUT training and WITHOUT torch_geometric.
#   1. compute_surface_variation  (curvature fix)   -> discriminates flat vs edge
#   2. augment_training_block      (aug + normal-rotation consistency + RGB dropout)
#   3. partition_columns           (full coverage, exact block size, overlap)
#   4. merge_block_votes           (majority-voting correctness)
# Run:  conda run -n venv_lmm python test_improvements.py
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import data_processing as dp

PASS, FAIL = 0, 0
def check(name, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1; print(f"  [PASS] {name} {extra}")
    else:
        FAIL += 1; print(f"  [FAIL] {name} {extra}")

class Data:  # minimal stand-in so we don't need torch_geometric here
    def __init__(self, pos, x):
        self.pos = pos; self.x = x

print("\n[1] compute_surface_variation: flat plane ~0, sharp edge >> plane")
rng = np.random.default_rng(0)
plane = np.column_stack([rng.uniform(0, 1, 4000), rng.uniform(0, 1, 4000), np.zeros(4000)])
sv_plane = dp.compute_surface_variation(plane, knn=15)
half = 2000
floor = np.column_stack([rng.uniform(0, 1, half), rng.uniform(0, 1, half), np.zeros(half)])
wall = np.column_stack([np.zeros(half), rng.uniform(0, 1, half), rng.uniform(0, 1, half)])
edge = np.vstack([floor, wall])
sv_edge = dp.compute_surface_variation(edge, knn=15)
crease = sv_edge[np.abs(edge[:, 0]) < 0.05]
check("flat plane surface variation is small", sv_plane.mean() < 0.05, f"(mean={sv_plane.mean():.4f})")
check("crease variation > flat plane variation", crease.mean() > sv_plane.mean() * 3,
      f"(crease={crease.mean():.4f} vs plane={sv_plane.mean():.4f})")
check("curvature bounded [0,1]", sv_edge.min() >= 0 and sv_edge.max() <= 1.0)

print("\n[2] augment_training_block: rotation is isometric & rotates normals consistently")
N = 500
pos = torch.rand(N, 3)
normals = torch.nn.functional.normalize(torch.randn(N, 3), dim=1)
x = torch.cat([normals, torch.rand(N, 1), torch.rand(N, 3), torch.rand(N, 3)], dim=1)
d = Data(pos.clone(), x.clone())
def pdist_sample(p, k=50):
    a = p[:k]
    return torch.cdist(a, a)
before = pdist_sample(d.pos)
d2 = dp.augment_training_block(d, strength=1.0, jitter_std=0.0, scale_range=(1.0, 1.0),
                               rgb_jitter=0.0, rgb_dropout_prob=0.0)
after = pdist_sample(d2.pos)
check("pure rotation preserves distances", torch.allclose(before, after, atol=2e-3),
      f"(max diff={(before-after).abs().max():.2e}, float32 rounding)")
nrm = d2.x[:, 0:3].norm(dim=1)
check("rotated normals remain unit-length", torch.allclose(nrm, torch.ones(N), atol=1e-4),
      f"(mean norm={nrm.mean():.4f})")
d3 = Data(torch.rand(N, 3), torch.cat([normals, torch.rand(N, 1), torch.rand(N, 3) + 0.1, torch.rand(N, 3)], dim=1))
d3 = dp.augment_training_block(d3, strength=1.0, rgb_dropout_prob=1.0)
check("RGB channels zeroed on dropout", float(d3.x[:, 4:7].abs().sum()) == 0.0)

print("\n[3] partition_columns: full coverage, exact block size, overlap present")
pts = rng.uniform(0, 6, (50000, 3))
pts[:, 2] = rng.uniform(0, 3, 50000)
bs = 4096
blocks = dp.partition_columns(pts, block_size=bs, window=1.5, stride=0.75, seed=1)
check("every block has exactly block_size points", all(len(idx) == bs for idx, _ in blocks), f"({len(blocks)} blocks)")
covered = set()
for idx, num_real in blocks:
    covered.update(idx[:num_real].tolist())
check("all points covered (no silent drops)", len(covered) == len(pts), f"({len(covered)}/{len(pts)})")
total_real = sum(nr for _, nr in blocks)
check("overlap present (sum real > N)", total_real > len(pts), f"(sum={total_real} vs N={len(pts)})")

print("\n[4] merge_block_votes: majority vote resolves disagreement")
total, ncls = 5, 3
per_block = [
    (np.array([1, 0, 2, 2, 1]), np.array([0, 1, 2, 3, 4])),
    (np.array([1, 0, 0]), np.array([0, 1, 2])),
    (np.array([2]), np.array([0])),
]
labels, counts = dp.merge_block_votes(total, ncls, per_block)
check("voted label for point0 is class1 (2 vs 1)", labels[0] == 1, f"(got {labels[0]})")
check("voted label for point2 is class0 (tie->argmax0)", labels[2] == 0, f"(got {labels[2]})")
check("vote count recorded", counts[0] == 2, f"(got {counts[0]})")

print("\n[5] spatial_split_is_val: deterministic, adjacency-consistent, ~val_ratio")
# determinism
r1 = dp.spatial_split_is_val("roomA", (2.1, 3.4), super_size=4.0, val_ratio=0.2, seed=42)
r2 = dp.spatial_split_is_val("roomA", (2.1, 3.4), super_size=4.0, val_ratio=0.2, seed=42)
check("same inputs -> same split (deterministic)", r1 == r2)
# adjacency: two centroids in the same 4m super-cell get the same tag
a = dp.spatial_split_is_val("roomA", (0.5, 0.5), super_size=4.0, seed=42)
b = dp.spatial_split_is_val("roomA", (3.9, 3.9), super_size=4.0, seed=42)  # same cell [0,0]
check("same super-cell -> same split (no boundary leak)", a == b)
# ratio ~ val_ratio across many distinct super-cells
cells = [dp.spatial_split_is_val("roomA", (i * 4.0 + 1, j * 4.0 + 1), super_size=4.0, val_ratio=0.2, seed=42)
         for i in range(40) for j in range(40)]
frac = sum(cells) / len(cells)
check("val fraction close to 0.2", 0.13 < frac < 0.27, f"(got {frac:.3f})")
# different source keys are independent (not identical assignment everywhere)
src_diff = [dp.spatial_split_is_val(f"room{k}", (1.0, 1.0), super_size=4.0, seed=42) for k in range(20)]
check("different sources -> mixed assignment", 0 < sum(src_diff) < len(src_diff))

print("\n[6] resolve_feature_config + extract_features_from_room_data: configurable composition")
spec_def = dp.resolve_feature_config({'num_features': 10})
check("default spec -> 10D (4,3,3)", spec_def['num_features'] == 10 and
      (spec_def['geo_dim'], spec_def['rgb_dim'], spec_def['spatial_dim']) == (4, 3, 3))
pts_xyzrgb = np.random.rand(1500, 6); pts_xyzrgb[:, 3:6] *= 255
f_def = dp.extract_features_from_room_data(pts_xyzrgb, feature_config=spec_def)
check("default extraction -> 10 columns", f_def.shape[1] == 10, f"(got {f_def.shape[1]})")

spec_norgb = dp.resolve_feature_config({'features': {'use_rgb': False}})
check("no-RGB spec -> 7D (4,0,3)", spec_norgb['num_features'] == 7 and spec_norgb['rgb_dim'] == 0)
pts_xyz = np.random.rand(1500, 3)  # colorless input (3 columns only)
f_norgb = dp.extract_features_from_room_data(pts_xyz, feature_config=spec_norgb)
check("colorless 3-col input extracts without crash -> 7D", f_norgb.shape[1] == 7, f"(got {f_norgb.shape[1]})")

spec_terrain = dp.resolve_feature_config({'features': {'use_rgb': False, 'spatial_scale': 5.0, 'use_curvature': False}})
check("terrain spec: scale=5, no rgb/curv -> 6D (3,0,3)", spec_terrain['spatial_scale'] == 5.0 and
      spec_terrain['num_features'] == 6 and spec_terrain['geo_dim'] == 3, f"(got {spec_terrain['num_features']}D)")

print("\n[7] model.FeatureGate / PointEdgeSegNet: adapt to feature layout (real model.py via PyG stub)")
import types
for name, attrs in [('torch_geometric', {}),
                    ('torch_geometric.nn', {'fps': lambda *a, **k: None, 'knn_interpolate': lambda *a, **k: None}),
                    ('torch_geometric.nn.pool', {'knn_graph': lambda *a, **k: None}),
                    ('torch_geometric.utils', {'scatter': lambda *a, **k: None})]:
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
import importlib
M = importlib.import_module('model')

fg = M.FeatureGate(geo_dim=4, rgb_dim=0, spatial_dim=3)  # colorless -> 7D, 2 gates
xg = torch.rand(40, 7)
out, gates = fg(xg)
check("FeatureGate(no rgb) output keeps 7 dims", out.shape == (40, 7), f"(got {tuple(out.shape)})")
check("FeatureGate reports [N,3] gate view", gates.shape == (40, 3))
check("absent RGB group gate is exactly 0", float(gates[:, 1].abs().sum()) == 0.0)
check("present groups gate nonzero", float(gates[:, 0].abs().sum()) > 0 and float(gates[:, 2].abs().sum()) > 0)

# default gate unchanged (3 gates, 10D)
fg_def = M.FeatureGate(4, 3, 3)
o2, g2 = fg_def(torch.rand(20, 10))
check("default FeatureGate -> 10D out, 3 gates", o2.shape == (20, 10) and g2.shape == (20, 3))

# model builds for matching dims and rejects mismatched dims
ok_build = True
try:
    M.PointEdgeSegNet(num_features=7, num_classes=5, feature_dims=(4, 0, 3))
except Exception as e:
    ok_build = False
check("PointEdgeSegNet builds for (4,0,3)=7D", ok_build)
raised = False
try:
    M.PointEdgeSegNet(num_features=7, num_classes=5, feature_dims=(4, 3, 3))  # sums to 10 != 7
except AssertionError:
    raised = True
check("mismatched feature_dims vs num_features raises", raised)

print("\n[8] convert_dataset: label remap + build_arrays + config emit (no torch_geometric)")
import convert_dataset as cv

# label remap: source ids 1..3 -> 0..2, 0 ignored
lab = np.array([0, 1, 2, 3, 1, 0], dtype=np.int64)
r = cv.remap_labels(lab, label_map={1: 0, 2: 1, 3: 2}, ignore={0})
check("remap maps ids and ignores 0 -> -1", r.tolist() == [-1, 0, 1, 2, 0, -1], f"(got {r.tolist()})")
r2 = cv.remap_labels(np.array([255, 0, 5], dtype=np.int64), label_map=None, ignore={255})
check("identity remap with ignore sentinel", r2.tolist() == [-1, 0, 5], f"(got {r2.tolist()})")

# build_arrays with RGB (toronto3d-like) -> 10D; recenter shifts min to 0
xyz = np.random.rand(1200, 3) * 3 + np.array([627285.0, 4841948.0, 10.0])  # UTM-like offset
rgb = (np.random.rand(1200, 3) * 255)
labels = np.random.randint(0, 8, 1200)
spec_rgb = cv.build_spec(cv.PROFILES["toronto3d"])
feats, pos, y = cv.build_arrays(xyz, rgb, labels, spec_rgb, recenter=True)
check("build_arrays(rgb) -> 10D features", feats.shape == (1200, 10), f"(got {feats.shape})")
check("pos and y aligned to points", pos.shape == (1200, 3) and y.shape == (1200,))
check("recenter translates min to ~0", float(np.abs(pos.min(axis=0)).max()) < 1e-3)

# build_arrays without RGB (dales-like) -> 7D, colorless input (3 cols)
spec_norgb = cv.build_spec(cv.PROFILES["dales"])
feats2, pos2, y2 = cv.build_arrays(np.random.rand(1000, 3) * 50, None, np.random.randint(0, 8, 1000), spec_norgb)
check("build_arrays(no rgb) -> 7D features", feats2.shape == (1000, 7), f"(got {feats2.shape})")

# emit config -> valid json with matching dims
import json as _json, tempfile
cfg_path = os.path.join(os.environ.get("TEMP", "."), "model_params_opentrench3d_test.json")
cv.emit_model_params("opentrench3d", cv.PROFILES["opentrench3d"], cv.build_spec(cv.PROFILES["opentrench3d"]), cfg_path)
cfg = _json.load(open(cfg_path))
check("emitted config: 5 classes, 10D (rgb)", cfg["num_classes"] == 5 and cfg["num_features"] == 10)
check("emitted config: rgb_cols set", cfg["input"]["rgb_cols"] == [3, 4, 5])
cfg_d = os.path.join(os.environ.get("TEMP", "."), "model_params_dales_test.json")
cv.emit_model_params("dales", cv.PROFILES["dales"], cv.build_spec(cv.PROFILES["dales"]), cfg_d)
cfgd = _json.load(open(cfg_d))
check("emitted DALES config: 8 classes, 7D, no rgb", cfgd["num_classes"] == 8 and cfgd["num_features"] == 7 and cfgd["input"]["rgb_cols"] is None)

print(f"\n==== SMOKE TEST RESULT: {PASS} passed, {FAIL} failed ====")
sys.exit(1 if FAIL else 0)
