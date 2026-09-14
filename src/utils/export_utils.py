import numpy as np
import cv2
from PIL import Image
from pathlib import Path

def save_component(component_data, output_dir, name_prefix, number, padding, input_size=None):
    """Save a single component with proper naming and formatting"""
    try:
        if component_data is None:
            return None
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if input_size is not None and isinstance(component_data, np.ndarray) and len(component_data.shape) >= 2:
            current_height, current_width = component_data.shape[:2]
            target_height, target_width = input_size
            
            if (current_height, current_width) != (target_height, target_width):
                component_data = cv2.resize(component_data, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        
        filename = f"{name_prefix}_{str(number).zfill(padding)}.png"
        filepath = output_path / filename
        
        if component_data.dtype != np.uint8:
            if component_data.max() <= 1.0:
                img_data = (component_data * 255).astype(np.uint8)
            else:
                img_data = ((component_data - component_data.min()) / 
                          (component_data.max() - component_data.min()) * 255).astype(np.uint8)
        else:
            img_data = component_data
        
        if len(img_data.shape) == 3:
            img = Image.fromarray(img_data, 'RGB')
        elif len(img_data.shape) == 2:
            img = Image.fromarray(img_data, 'L')
        else:
            return None
        
        img.save(filepath)
        return str(filepath)
        
    except Exception as e:
        return None

def export_all_components(result, output_dir, start_number=0, padding=6, input_size=None):
    """Export processed components: albedo, specular, depth, and normal"""
    try:
        main_output = Path(output_dir)
        main_output.mkdir(parents=True, exist_ok=True)
        
        albedo_dir = main_output / "albedo"
        specular_dir = main_output / "specular"
        depth_dir = main_output / "depth"
        normal_dir = main_output / "normal"
        
        albedo_dir.mkdir(parents=True, exist_ok=True)
        specular_dir.mkdir(parents=True, exist_ok=True)
        depth_dir.mkdir(parents=True, exist_ok=True)
        normal_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = []
        
        if 'albedo' in result and result['albedo'] is not None:
            albedo_path = save_component(result['albedo'], albedo_dir, "albedo", start_number, padding, input_size)
            if albedo_path:
                saved_files.append(f"Albedo: {albedo_path}")
        
        if 'specular' in result and result['specular'] is not None:
            specular_path = save_component(result['specular'], specular_dir, "specular", start_number, padding, input_size)
            if specular_path:
                saved_files.append(f"Specular: {specular_path}")
        
        if 'depth' in result and result['depth'] is not None:
            depth_path = save_component(result['depth'], depth_dir, "depth", start_number, padding, input_size)
            if depth_path:
                saved_files.append(f"Depth: {depth_path}")
        
        if 'normal' in result and result['normal'] is not None:
            normal_path = save_component(result['normal'], normal_dir, "normal", start_number, padding, input_size)
            if normal_path:
                saved_files.append(f"Normal: {normal_path}")
        
        return saved_files
        
    except Exception as e:
        return []