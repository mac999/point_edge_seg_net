import os
import glob
import json
import argparse
import numpy as np
import ifcopenshell
import ifcopenshell.geom
import laspy
from pathlib import Path
from tqdm import tqdm

def load_category_mapping(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    mapping = {}
    colors = {}
    for category, data in config.items():
        if isinstance(data, list):
            ifc_classes = data
            color = [255, 255, 255] 
        else:
            ifc_classes = data.get("classes", [])
            color = data.get("color", [255, 255, 255]) 
            
        colors[category] = color
        
        for ifc_class in ifc_classes:
            mapping[ifc_class] = category
            
    return mapping, colors

def sample_points_from_faces(verts, faces, spacing=0.03):
    points_list = []
    points_per_area = 1.0 / (spacing ** 2)
    
    for i in range(0, len(faces), 3):
        idxA, idxB, idxC = faces[i], faces[i+1], faces[i+2]
        A, B, C = verts[idxA], verts[idxB], verts[idxC]
        
        cross_prod = np.cross(B - A, C - A)
        area = 0.5 * np.linalg.norm(cross_prod)
        
        if area <= 0:
            continue
            
        num_points = np.random.poisson(area * points_per_area)
        
        if num_points > 0:
            r1 = np.random.rand(num_points, 1)
            r2 = np.random.rand(num_points, 1)
            
            sqrt_r1 = np.sqrt(r1)
            u = 1.0 - sqrt_r1
            v = r2 * sqrt_r1
            w = 1.0 - u - v
            
            sampled = u * A + v * B + w * C
            points_list.append(sampled)
            
    if points_list:
        return np.vstack(points_list)
    else:
        return np.empty((0, 3))

def save_to_las(points, out_filepath, rgb_color):
    header = laspy.LasHeader(point_format=3, version="1.2")
    las = laspy.LasData(header)
    
    # 좌표 할당
    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]
    
    # LAS 파일의 16-bit 색상 규격 변환 (8-bit → 16-bit: 0-255 → 0-65535)
    # 올바른 스케일링: value * 257 (255 * 257 = 65535)
    r16 = int(rgb_color[0] * 257)
    g16 = int(rgb_color[1] * 257)
    b16 = int(rgb_color[2] * 257)
    
    num_points = points.shape[0]
    las.red = np.full(num_points, r16, dtype=np.uint16)
    las.green = np.full(num_points, g16, dtype=np.uint16)
    las.blue = np.full(num_points, b16, dtype=np.uint16)
    
    las.write(str(out_filepath))

def process_ifc_to_s3dis_format(input_dir, output_dir, config_path, spacing=0.03, deflection_tol=0.001):
    mapping, colors = load_category_mapping(config_path)
    supported_classes = list(mapping.keys())
    
    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True)
    
    # Deflection Tolerance 강제 적용 로직
    try:
        settings.set("deflection-tolerance", deflection_tol)
    except Exception:
        try:
            if hasattr(settings, '_settings'):
                settings._settings.set_deflection_tolerance(deflection_tol)
            else:
                settings.set_deflection_tolerance(deflection_tol)
        except Exception:
            pass
    
    ifc_files = glob.glob(os.path.join(input_dir, "*.ifc"))
    if not ifc_files:
        print(f"No IFC files found in '{input_dir}'.")
        return
        
    for ifc_file in ifc_files:
        filename = Path(ifc_file).stem
        print(f"\nProcessing: {filename}.ifc")
        
        model = ifcopenshell.open(ifc_file)
        
        model_out_dir = Path(output_dir) / filename
        model_out_dir.mkdir(parents=True, exist_ok=True)
        
        counter = {cat: 1 for cat in set(mapping.values())}
        extracted_count = 0
        
        pbar = tqdm(supported_classes, desc="Extracting classes", leave=True)
        
        for ifc_class in pbar:
            entities = model.by_type(ifc_class)
            if not entities:
                continue
                
            category_name = mapping[ifc_class]
            category_color = colors[category_name]
            pbar.set_postfix({"Category": category_name, "Entities": len(entities)})
            
            for entity in entities:
                try:
                    shape = ifcopenshell.geom.create_shape(settings, entity)
                    verts = np.array(shape.geometry.verts).reshape(-1, 3)
                    faces = shape.geometry.faces 
                    
                    points = sample_points_from_faces(verts, faces, spacing=spacing)
                    
                    if points.shape[0] > 0:
                        out_filename = f"{category_name}#{counter[category_name]}.las"
                        out_filepath = model_out_dir / out_filename
                        
                        save_to_las(points, out_filepath, rgb_color=category_color)
                        counter[category_name] += 1
                        extracted_count += 1
                        
                except Exception:
                    continue
                    
        print(f"Total extracted component point clouds: {extracted_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert IFC files to LAS point cloud format with semantic labels")
    
    parser.add_argument("--input", "-i", type=str, default="./input", help="Input folder containing IFC files (default: ./input)")
    parser.add_argument("--output", "-o", type=str, default="./output", help="Output folder for LAS files (default: ./output)")
    parser.add_argument("--config", "-c", type=str, default="./config.json", help="Path to category mapping config file (default: ./config.json)")
    parser.add_argument("--spacing", "-s", type=float, default=0.03, help="Point sampling spacing in meters (default: 0.03)")
    parser.add_argument("--tolerance", "-t", type=float, default=0.001, help="Mesh deflection tolerance in meters (default: 0.001)")
    
    args = parser.parse_args()
    
    INPUT_FOLDER = args.input
    OUTPUT_FOLDER = args.output
    CONFIG_FILE = args.config
    POINT_SPACING = args.spacing
    DEFLECTION_TOL = args.tolerance
    
    os.makedirs(INPUT_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    if not os.path.exists(CONFIG_FILE):
        sample_config = {
            "wall": {
                "classes": ["IfcWall", "IfcWallStandardCase", "IfcCurtainWall"],
                "color": [150, 150, 150]
            },
            "column": {
                "classes": ["IfcColumn"],
                "color": [255, 100, 100]
            },
            "beam": {
                "classes": ["IfcBeam"],
                "color": [100, 255, 100]
            },
            "slab": {
                "classes": ["IfcSlab", "IfcRoof"],
                "color": [100, 100, 255]
            },
            "door": {
                "classes": ["IfcDoor"],
                "color": [255, 200, 100]
            },
            "window": {
                "classes": ["IfcWindow"],
                "color": [100, 200, 255]
            }
        }
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(sample_config, f, indent=4)
            
    print(f"Settings -> Tolerance: {DEFLECTION_TOL}m | Spacing: {POINT_SPACING}m")
    
    process_ifc_to_s3dis_format(INPUT_FOLDER, OUTPUT_FOLDER, CONFIG_FILE, 
                                spacing=POINT_SPACING, deflection_tol=DEFLECTION_TOL)