# Title: convert_dataset (version 0.1)
# Author: taewook kang (laputa9999@gmail.com)
# Purpose: Convert public infrastructure / urban point-cloud datasets into the
#          PointEdgeSegNet processed format (torch_geometric Data .pt with pos/x/y),
#          so they can be trained/evaluated with this repo's pipeline directly.
# Dependencies: numpy, torch, open3d (features); optional: plyfile (.ply), laspy (.las/.laz)
#
# Downstream flow after conversion:
#   convert_dataset.py  -> processed_<name>/train/<scene>.pt , processed_<name>/test/<scene>.pt
#   train_model.py --processed_data_path ./processed_<name> --train_areas train --test_area test \
#                  --config model_params_<name>.json --block_mode column --no_wandb
#
# The 'train'/'test' subfolders act as the pipeline's "areas". The .pt files are the same
# Data(pos, x, y) objects produced by data_preparation.py for S3DIS, so grid/column block
# splitting, augmentation, training and inference all work unchanged.
#
# ---------------------------------------------------------------------------------------
# Datasets covered (verified public benchmarks, good for paper comparison). Field names
# follow each dataset's documentation but CAN VARY between download versions/exports, so
# every profile is overridable from the CLI (--label_field / --rgb / --no_rgb /
# --spatial_scale / --split). Always check the dataset's own license before use.
#
#   name           domain            sensor  rgb  format        download
#   toronto3d      urban roadway     MLS     yes  .ply          github.com/WeikaiTan/Toronto-3D  (CC BY-NC 4.0)
#   sensaturban    urban             UAV     yes  .ply          github.com/QingyongHu/SensatUrban  (form; MIT code)
#   dales          aerial/terrain    ALS     no   .ply/.las     udayton DALES page (form)
#   stpls3d        aerial            photo   yes  .ply          stpls3d.com/data (form)
#   opentrench3d   utilities/trench  TLS     yes  .ply          kaggle.com/datasets/hestogpony/opentrench3d
#   whu_railway3d  railway           MLS     no   .npy pair     github.com/WHU-USI3DV/WHU-Railway3D (form)
#   semanticbridge bridge            TLS/MLS yes  .ply/.las     github.com/mvg-inatech/3d_bridge_segmentation
# ---------------------------------------------------------------------------------------

import os, sys, glob, json, argparse
import numpy as np


# =====================================================================================
# Dataset profiles
# =====================================================================================
# label_map: {source_id: target_id}. If None, source ids are assumed already 0..K-1.
# ignore:    source ids mapped to -1 (ignored by the loss, like S3DIS padding).
# test_stems: filename stems routed to the test/ folder; others go to train/. If a
#             dataset ships its own train/ and test/ folders, use --split to force one.
DEFAULT_PROFILES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "configs", "dataset_profiles.json")


def load_profiles(path=None):
    """Read the dataset profiles from JSON.

    What differs between datasets -- extension, label field, label-id order, split -- is data,
    not code, so it lives in `dataset_profiles.json`. Supporting a new dataset is one entry in
    that file; pass --profiles to use a different file (e.g. a private dataset kept out of the
    repo). JSON has no set and no integer keys, so `ignore` and `label_map` are converted back
    to the shapes the rest of this script expects.
    """
    path = path or DEFAULT_PROFILES_PATH
    with open(path, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    raw = doc.get("profiles", doc)          # also accept a bare {name: profile} mapping
    profiles = {}
    for name, entry in raw.items():
        if name.startswith("_"):            # _comment and friends
            continue
        p = dict(entry)
        missing = {"ext", "class_names"} - set(p)
        if missing:
            raise ValueError(f"{path}: profile '{name}' is missing {sorted(missing)}")
        p["ignore"] = set(p.get("ignore") or [])
        p["label_map"] = ({int(k): v for k, v in p["label_map"].items()}
                          if p.get("label_map") else None)
        p.setdefault("has_rgb", False)
        p.setdefault("rgb_max", 255.0)
        p.setdefault("spatial_scale", 0.10)
        # no label_field default: .npy_pair layouts have none, and the use site uses .get()
        p.setdefault("test_stems", [])
        profiles[name] = p
    if not profiles:
        raise ValueError(f"{path}: no profiles found")
    return profiles


# =====================================================================================
# Readers  (lazy-import optional deps so the core stays testable without them)
# =====================================================================================
def _first_present(mapping, names):
    for n in names:
        if n in mapping:
            return mapping[n]
    return None


def _ply_columns(ply, path):
    """Flatten a PLY into one {property: array} mapping, whatever element layout it uses.

    Most exports put everything on a single `vertex` element. Some -- SemanticBridge among
    them -- split the same N points across parallel elements instead:

        element points N   property float x/y/z
        element color  N    property uchar red/green/blue
        element label  N    property uchar label

    Both are the same table stored differently, so every element whose length matches the
    point count is merged into one column mapping. Elements of any other length (faces, say)
    are skipped, and a property name appearing twice keeps its first occurrence.
    """
    elements = list(ply.elements)
    if not elements:
        raise ValueError(f"{path}: PLY contains no elements")
    # The point count is the length of the element carrying the coordinates.
    n_points = next((len(e.data) for e in elements
                     if {"x", "y", "z"} <= set(e.data.dtype.names or ())), None)
    if n_points is None:
        names = {e.name: (e.data.dtype.names or ()) for e in elements}
        raise KeyError(f"{path}: no element carries x/y/z. Elements: {names}")
    cols = {}
    for e in elements:
        if len(e.data) != n_points:
            continue
        for name in (e.data.dtype.names or ()):
            cols.setdefault(name, e.data[name])
    return cols


def read_ply(path, label_field, has_rgb, rgb_max):
    """Read x/y/z (+optional rgb) and an integer label field from a .ply file."""
    try:
        from plyfile import PlyData
    except ImportError:
        raise ImportError("Reading .ply requires plyfile: pip install plyfile")
    ply = PlyData.read(path)
    cols = _ply_columns(ply, path)
    xyz = np.stack([cols["x"], cols["y"], cols["z"]], axis=1).astype(np.float64)

    rgb = None
    if has_rgb:
        r = _first_present(cols, ["red", "r", "Red", "diffuse_red"])
        g = _first_present(cols, ["green", "g", "Green", "diffuse_green"])
        b = _first_present(cols, ["blue", "b", "Blue", "diffuse_blue"])
        if r is not None and g is not None and b is not None:
            rgb = np.stack([r, g, b], axis=1).astype(np.float64)
        else:
            print(f"  [warn] {os.path.basename(path)}: rgb requested but not found; using zeros")
            rgb = np.zeros((len(xyz), 3), dtype=np.float64)

    lab = _first_present(cols, [label_field, "scalar_Label", "label", "class", "sem_class", "C", "scalar_label"])
    if lab is None:
        raise KeyError(f"Label field '{label_field}' not found in {path}. "
                       f"Available: {list(cols.keys())} (override with --label_field)")
    labels = np.asarray(lab).astype(np.int64)
    return xyz, rgb, labels


def read_las(path, has_rgb, rgb_max):
    """Read x/y/z (+optional rgb) and 'classification' from a .las/.laz file."""
    try:
        import laspy
    except ImportError:
        raise ImportError("Reading .las/.laz requires laspy: pip install laspy")
    las = laspy.read(path)
    xyz = np.stack([np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)], axis=1).astype(np.float64)
    labels = np.asarray(las.classification).astype(np.int64)
    rgb = None
    if has_rgb and hasattr(las, "red"):
        # LAS colors are 16-bit; scale down to 0-255 range for consistency
        rgb = np.stack([np.asarray(las.red), np.asarray(las.green), np.asarray(las.blue)], axis=1).astype(np.float64)
        if rgb.max() > 255:
            rgb = rgb / 256.0
    elif has_rgb:
        rgb = np.zeros((len(xyz), 3), dtype=np.float64)
    return xyz, rgb, labels


def read_txt(path, has_rgb, label_col):
    """Read a whitespace/csv text cloud: cols 0-2 = xyz, [3-5 = rgb], label_col = label."""
    arr = np.loadtxt(path)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    xyz = arr[:, 0:3].astype(np.float64)
    rgb = arr[:, 3:6].astype(np.float64) if has_rgb and arr.shape[1] >= 6 else None
    labels = arr[:, label_col].astype(np.int64)
    return xyz, rgb, labels


def read_npy_pair(points_path, labels_path):
    """Read coords .npy (N,3+) and labels .npy (N,)."""
    pts = np.load(points_path)
    labels = np.load(labels_path).astype(np.int64).reshape(-1)
    xyz = pts[:, 0:3].astype(np.float64)
    return xyz, None, labels


# =====================================================================================
# Core (dependency-light, unit-tested): remap labels + build repo feature arrays
# =====================================================================================
def remap_labels(labels, label_map=None, ignore=None):
    """Map source label ids to contiguous target ids; ignored ids -> -1."""
    ignore = ignore or set()
    out = np.full(labels.shape, -1, dtype=np.int64)
    if label_map is None:
        # identity, except ignored ids
        keep = ~np.isin(labels, list(ignore)) if ignore else np.ones(len(labels), bool)
        out[keep] = labels[keep]
    else:
        for src, tgt in label_map.items():
            out[labels == src] = tgt
    if ignore:
        out[np.isin(labels, list(ignore))] = -1
    return out


def build_arrays(xyz, rgb, labels, spec, recenter=True):
    """Build (features, pos, y) numpy arrays in the repo's processed format.

    Uses the same domain-configurable feature extractor as the S3DIS pipeline
    (data_processing.extract_features_from_room_data), so the produced .pt is identical
    in structure to processed_s3dis/*.pt. Returns numpy arrays (no torch_geometric), so
    this function is testable without the heavy training deps installed.
    """
    from data_processing import extract_features_from_room_data
    xyz = np.asarray(xyz, dtype=np.float64)
    if recenter:
        # Translate to a local origin (critical for UTM/large-coordinate datasets to
        # avoid float32 precision loss in kNN/FPS). Relative geometry is preserved.
        xyz = xyz - xyz.min(axis=0, keepdims=True)

    if spec["use_rgb"] and rgb is not None:
        points = np.concatenate([xyz, np.asarray(rgb, dtype=np.float64)], axis=1)
    else:
        points = xyz

    feats = extract_features_from_room_data(points, normalize_colors=True, feature_config=spec)
    return feats.astype(np.float32), xyz.astype(np.float32), np.asarray(labels, dtype=np.int64)


def save_scene(feats, pos, y, out_path):
    """Wrap arrays into a torch_geometric Data and save (needs torch + torch_geometric)."""
    import torch
    from torch_geometric.data import Data
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    data = Data(x=torch.from_numpy(feats), pos=torch.from_numpy(pos), y=torch.from_numpy(y))
    torch.save(data, out_path)


# =====================================================================================
# Config emission
# =====================================================================================
def _distinct_colors(n):
    """Generate n visually-distinct RGB colors (0-255) by spreading hue."""
    colors = []
    for i in range(n):
        h = (i / max(n, 1)) % 1.0
        # simple HSV(h,0.65,0.95) -> RGB
        import colorsys
        r, g, b = colorsys.hsv_to_rgb(h, 0.65, 0.95)
        colors.append([int(r * 255), int(g * 255), int(b * 255)])
    return colors


def emit_model_params(name, profile, spec, out_path):
    """Write a ready-to-use model_params_<name>.json for the converted dataset."""
    class_names = profile["class_names"]
    cfg = {
        "dataset_name": name,
        "num_classes": len(class_names),
        "class_names": class_names,
        "class_colors": _distinct_colors(len(class_names)),
        "class_weights": [1.0] * len(class_names),
        "num_features": spec["num_features"],
        "block_size": 8192,
        "input": {
            "xyz_cols": [0, 1, 2],
            "rgb_cols": [3, 4, 5] if profile["has_rgb"] else None,
            "rgb_max": profile["rgb_max"],
        },
        "features": {
            "use_normals": True,
            "use_curvature": True,
            "use_rgb": bool(profile["has_rgb"]),
            "use_spatial": True,
            "spatial_scale": profile["spatial_scale"],
            "neighbor_knn": 15,
        },
        "preprocessing": {"grid_min_coords": [0.0, 0.0, 0.0], "grid_resolution": profile["spatial_scale"]},
        "description": f"Auto-generated config for the {name} dataset (convert_dataset.py).",
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print(f"Wrote config -> {out_path}")


# =====================================================================================
# Driver
# =====================================================================================
def build_spec(profile, spatial_scale=None, force_rgb=None):
    """Resolve a feature spec (data_processing.resolve_feature_config) from the profile."""
    from data_processing import resolve_feature_config
    has_rgb = profile["has_rgb"] if force_rgb is None else force_rgb
    cfg = {
        "input": {"xyz_cols": [0, 1, 2], "rgb_cols": [3, 4, 5] if has_rgb else None, "rgb_max": profile["rgb_max"]},
        "features": {
            "use_normals": True, "use_curvature": True, "use_rgb": has_rgb, "use_spatial": True,
            "spatial_scale": spatial_scale if spatial_scale is not None else profile["spatial_scale"],
            "neighbor_knn": 15,
        },
    }
    return resolve_feature_config(cfg)


def load_one(path, profile, label_field, has_rgb, label_col):
    ext = profile["ext"]
    if ext == ".ply":
        return read_ply(path, label_field, has_rgb, profile["rgb_max"])
    if ext in (".las", ".laz"):
        return read_las(path, has_rgb, profile["rgb_max"])
    if ext in (".txt", ".csv"):
        return read_txt(path, has_rgb, label_col)
    if ext == ".npy_pair":
        labels_path = path.replace(profile["points_suffix"], profile["labels_suffix"])
        return read_npy_pair(path, labels_path)
    raise ValueError(f"Unsupported ext {ext}")


def list_scenes(input_dir, profile):
    if profile["ext"] == ".npy_pair":
        return sorted(glob.glob(os.path.join(input_dir, "*" + profile["points_suffix"])))
    return sorted(glob.glob(os.path.join(input_dir, "*" + profile["ext"])))


def main():
    ap = argparse.ArgumentParser(description="Convert public datasets to PointEdgeSegNet processed .pt format")
    ap.add_argument("--dataset", required=True,
                    help="Profile name from the profile file (see dataset_profiles.json)")
    ap.add_argument("--profiles", default=None, metavar="PROFILES.json",
                    help="Dataset profile definitions (default: dataset_profiles.json next to "
                         "this script). Point it at your own file to convert a dataset that is "
                         "not in the repo, without touching any code.")
    ap.add_argument("--input_dir", required=True, help="Folder with the dataset scene files")
    ap.add_argument("--output_dir", default=None, help="Output processed dir (default: ./processed_<dataset>)")
    ap.add_argument("--split", choices=["auto", "train", "test"], default="auto",
                    help="Force all scenes to train/ or test/ (auto uses profile test_stems)")
    ap.add_argument("--label_field", default=None, help="Override the .ply label property name")
    ap.add_argument("--label_col", type=int, default=-1, help="Label column index for .txt/.csv inputs")
    ap.add_argument("--rgb", dest="rgb", action="store_true", help="Force-enable RGB")
    ap.add_argument("--no_rgb", dest="rgb", action="store_false", help="Force-disable RGB")
    ap.set_defaults(rgb=None)
    ap.add_argument("--spatial_scale", type=float, default=None, help="Override spatial-context scale (m)")
    ap.add_argument("--no_recenter", action="store_true", help="Do NOT translate each scene to a local origin")
    ap.add_argument("--limit", type=int, default=0, help="Convert at most N scenes (0 = all)")
    ap.add_argument("--emit_config", action="store_true", help="Also write model_params_<dataset>.json")
    ap.add_argument("--dry_run", action="store_true", help="Parse + report shapes, do not write .pt")
    args = ap.parse_args()

    profiles = load_profiles(args.profiles)
    if args.dataset not in profiles:
        ap.error(f"unknown dataset '{args.dataset}'. "
                 f"{args.profiles or DEFAULT_PROFILES_PATH} defines: "
                 f"{', '.join(sorted(profiles))}")
    profile = profiles[args.dataset]
    has_rgb = profile["has_rgb"] if args.rgb is None else args.rgb
    label_field = args.label_field or profile.get("label_field")
    out_dir = args.output_dir or f"./processed_{args.dataset}"
    spec = build_spec(profile, spatial_scale=args.spatial_scale, force_rgb=has_rgb)
    print(f"Dataset={args.dataset}  rgb={has_rgb}  features={spec['num_features']}D  "
          f"spatial_scale={spec['spatial_scale']}  -> {out_dir}")

    if args.emit_config:
        emit_model_params(args.dataset, profile, spec, f"model_params_{args.dataset}.json")

    scenes = list_scenes(args.input_dir, profile)
    if args.limit:
        scenes = scenes[: args.limit]
    if not scenes:
        print(f"No scene files found in {args.input_dir} (ext {profile['ext']}).")
        return
    print(f"Found {len(scenes)} scene(s).")

    test_stems = set(profile.get("test_stems", []))
    n_ok = 0
    for path in scenes:
        stem = os.path.basename(path)
        for e in (profile["ext"], profile.get("points_suffix", "")):
            if e:
                stem = stem.replace(e, "")
        if args.split == "auto":
            split = "test" if any(t in stem for t in test_stems) else "train"
        else:
            split = args.split
        try:
            xyz, rgb, raw_labels = load_one(path, profile, label_field, has_rgb, args.label_col)
            labels = remap_labels(raw_labels, profile.get("label_map"), profile.get("ignore"))
            n_valid = int((labels >= 0).sum())
            print(f"  {stem}: {len(xyz):,} pts, {n_valid:,} labeled, split={split}")
            if args.dry_run:
                n_ok += 1
                continue
            feats, pos, y = build_arrays(xyz, rgb, labels, spec, recenter=not args.no_recenter)
            save_scene(feats, pos, y, os.path.join(out_dir, split, f"{stem}.pt"))
            n_ok += 1
        except Exception as e:
            print(f"  [error] {stem}: {e}")

    print(f"\nDone: {n_ok}/{len(scenes)} scenes converted -> {out_dir}/(train|test)")
    if not args.dry_run:
        print("Next: python train_model.py "
              f"--config model_params_{args.dataset}.json "
              f"--processed_data_path {out_dir} --train_areas train --test_area test "
              "--block_mode column --no_wandb --cooldown_sec 0")


if __name__ == "__main__":
    main()
