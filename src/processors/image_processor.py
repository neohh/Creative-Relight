import os
# Enable OpenCV OpenEXR support for 32-bit float export BEFORE cv2 is loaded
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import numpy as np
from PIL import Image
from pathlib import Path
import cv2
import sys
import re

# Handle imports for both development and frozen (PyInstaller) environments
try:
    from lighting_model import LightingModel
    from geometry_model import GeometryModel
except ModuleNotFoundError:
    try:
        from models.lighting_model import LightingModel
        from models.geometry_model import GeometryModel
    except ModuleNotFoundError:
        from src.models.lighting_model import LightingModel
        from src.models.geometry_model import GeometryModel

try:
    from image_utils import resize_to_original, ensure_uint8
except ModuleNotFoundError:
    try:
        from utils.image_utils import resize_to_original, ensure_uint8
    except ModuleNotFoundError:
        from src.utils.image_utils import resize_to_original, ensure_uint8


def _safe_write_exr(filepath, data):
    """Safely write OpenEXR file without raising uncaught exceptions"""
    try:
        cv2.imwrite(str(filepath), data)
    except Exception as e:
        print(f"[WARN] Failed to write EXR {filepath}: {e}")


class ImageProcessor:
    """Process single images to generate all 5 passes in 32-bit OpenEXR and 16-bit PNG"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.lighting_model = LightingModel(device)
        self.geometry_model = GeometryModel(device)
        self.should_stop = False
    
    def stop(self):
        """Stop processing"""
        self.should_stop = True
    
    def process(self, image_input, output_dir, export_config=None, start_number=0, padding=6):
        """
        Process a single image and save selected passes in 32-bit OpenEXR and 16-bit PNG.
        """
        self.should_stop = False
        
        if export_config is None:
            export_config = {
                'albedo': True,
                'specular': True,
                'depth': True,
                'normal': True
            }
        
        if self.should_stop:
            return {'saved_files': {}, 'previews': {}, 'message': 'Processing stopped'}
        
        if isinstance(image_input, str):
            image = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, np.ndarray):
            image = Image.fromarray(image_input)
        else:
            image = image_input
        
        original_size = image.size
        image_array = np.array(image).astype(np.float32) / 255.0
        
        output_path = Path(output_dir)
        dirs = {}
        for component in ['albedo', 'specular', 'depth', 'normal']:
            if export_config.get(component, False):
                dirs[component] = output_path / component
                dirs[component].mkdir(parents=True, exist_ok=True)
        
        if self.should_stop:
            return {'saved_files': {}, 'previews': {}, 'message': 'Processing stopped'}
        
        lighting_results = {}
        if any(export_config.get(comp, False) for comp in ['albedo', 'specular']):
            lighting_results = self.lighting_model.process(image_array)
        
        if self.should_stop:
            return {'saved_files': {}, 'previews': {}, 'message': 'Processing stopped'}
        
        geometry_results = {}
        if any(export_config.get(comp, False) for comp in ['depth', 'normal']):
            geometry_results = self.geometry_model.process(image_array)
        
        saved_files = {}
        previews = {}
        
        # Auto-detect next free frame number by checking existing files (.exr and .png)
        max_existing = -1
        for comp, comp_dir in dirs.items():
            if comp_dir.exists():
                for item in comp_dir.iterdir():
                    m = re.match(rf"^{comp}_(\d+)\.(png|exr)$", item.name)
                    if m:
                        max_existing = max(max_existing, int(m.group(1)))
        
        if max_existing >= 0:
            assigned_num = max_existing + 1
        else:
            assigned_num = int(start_number) if start_number is not None else 0
            
        frame_number = f"{assigned_num:0{padding}d}"
        
        # Save albedo (32-bit float EXR + 16-bit PNG)
        if export_config.get('albedo', False) and 'albedo' in lighting_results and lighting_results['albedo'] is not None:
            albedo = resize_to_original(lighting_results['albedo'], original_size)
            albedo_f32 = np.clip(albedo, 0.0, 1.0).astype(np.float32)
            albedo_bgr_f32 = cv2.cvtColor(albedo_f32, cv2.COLOR_RGB2BGR)
            
            albedo_exr_path = dirs['albedo'] / f"albedo_{frame_number}.exr"
            albedo_png_path = dirs['albedo'] / f"albedo_{frame_number}.png"
            
            # Always save 16-bit PNG first (safe, guaranteed to succeed)
            cv2.imwrite(str(albedo_png_path), (albedo_bgr_f32 * 65535.0).astype(np.uint16))
            # Save 32-bit float OpenEXR
            _safe_write_exr(albedo_exr_path, albedo_bgr_f32)
            
            saved_files['albedo'] = str(albedo_exr_path if albedo_exr_path.exists() else albedo_png_path)
            previews['albedo'] = ensure_uint8(albedo)
        
        # Save specular (32-bit float EXR + 16-bit PNG)
        if export_config.get('specular', False) and 'specular' in lighting_results and lighting_results['specular'] is not None:
            specular = resize_to_original(lighting_results['specular'], original_size)
            specular_f32 = np.clip(specular, 0.0, 1.0).astype(np.float32)
            
            specular_exr_path = dirs['specular'] / f"specular_{frame_number}.exr"
            specular_png_path = dirs['specular'] / f"specular_{frame_number}.png"
            
            if len(specular_f32.shape) == 2:
                cv2.imwrite(str(specular_png_path), (specular_f32 * 65535.0).astype(np.uint16))
                _safe_write_exr(specular_exr_path, specular_f32)
            else:
                spec_bgr = cv2.cvtColor(specular_f32, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(specular_png_path), (spec_bgr * 65535.0).astype(np.uint16))
                _safe_write_exr(specular_exr_path, spec_bgr)
                
            saved_files['specular'] = str(specular_exr_path if specular_exr_path.exists() else specular_png_path)
            previews['specular'] = ensure_uint8(specular)
        
        # Save depth (32-bit float EXR + 16-bit PNG)
        if export_config.get('depth', False) and 'depth' in geometry_results and geometry_results['depth'] is not None:
            if 'depth_float' in geometry_results and geometry_results['depth_float'] is not None:
                depth_src = geometry_results['depth_float']
            else:
                depth_src = geometry_results['depth'].astype(np.float32) / 255.0
                
            depth_f32 = cv2.resize(depth_src, original_size, interpolation=cv2.INTER_LINEAR).astype(np.float32)
            depth_f32 = np.clip(depth_f32, 0.0, 1.0)
            
            depth_exr_path = dirs['depth'] / f"depth_{frame_number}.exr"
            depth_png_path = dirs['depth'] / f"depth_{frame_number}.png"
            
            # 1. 16-bit PNG (.png) - 65,536 levels, guaranteed zero banding
            cv2.imwrite(str(depth_png_path), (depth_f32 * 65535.0).astype(np.uint16))
            
            # 2. 32-bit float OpenEXR (.exr)
            _safe_write_exr(depth_exr_path, depth_f32)
            
            # 3. Optional: Metric depth EXR if available
            if 'depth_raw' in geometry_results and geometry_results['depth_raw'] is not None:
                depth_raw_resized = cv2.resize(geometry_results['depth_raw'], original_size, interpolation=cv2.INTER_LINEAR).astype(np.float32)
                depth_metric_exr_path = dirs['depth'] / f"depth_metric_{frame_number}.exr"
                _safe_write_exr(depth_metric_exr_path, depth_raw_resized)
            
            saved_files['depth'] = str(depth_exr_path if depth_exr_path.exists() else depth_png_path)
            previews['depth'] = (depth_f32 * 255.0).astype(np.uint8)
        
        # Save normal (32-bit float EXR + 16-bit PNG)
        if export_config.get('normal', False) and 'normal' in geometry_results and geometry_results['normal'] is not None:
            if 'normal_float' in geometry_results and geometry_results['normal_float'] is not None:
                normal_src = geometry_results['normal_float']
            else:
                normal_src = geometry_results['normal'].astype(np.float32) / 255.0
                
            normal_f32 = cv2.resize(normal_src, original_size, interpolation=cv2.INTER_LINEAR).astype(np.float32)
            normal_f32 = np.clip(normal_f32, 0.0, 1.0)
            normal_bgr = cv2.cvtColor(normal_f32, cv2.COLOR_RGB2BGR)
            
            normal_exr_path = dirs['normal'] / f"normal_{frame_number}.exr"
            normal_png_path = dirs['normal'] / f"normal_{frame_number}.png"
            
            # 1. 16-bit PNG (.png)
            cv2.imwrite(str(normal_png_path), (normal_bgr * 65535.0).astype(np.uint16))
            # 2. 32-bit float OpenEXR (.exr)
            _safe_write_exr(normal_exr_path, normal_bgr)
            
            saved_files['normal'] = str(normal_exr_path if normal_exr_path.exists() else normal_png_path)
            previews['normal'] = (normal_f32 * 255.0).astype(np.uint8)
        
        return {
            'saved_files': saved_files,
            'previews': previews
        }
