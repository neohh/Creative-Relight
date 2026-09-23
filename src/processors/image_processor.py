import os
# Enable OpenCV OpenEXR support BEFORE cv2 is loaded
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
    """Safely write OpenEXR file without throwing unhandled exceptions"""
    try:
        res = cv2.imwrite(str(filepath), data)
        if not res:
            print(f"[WARN] cv2.imwrite returned False for {filepath}")
    except Exception as e:
        print(f"[ERROR] Failed to save EXR {filepath}: {e}")


class ImageProcessor:
    """Process single images to generate passes in optimal formats:
       - Depth: 32-bit float OpenEXR (.exr)
       - Normal: 32-bit float OpenEXR (.exr)
       - Albedo: 8-bit PNG (.png)
       - Specular: 8-bit PNG (.png)
       - UI previews: In-memory uint8 (no disk waste)
    """
    
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
        Process a single image and save passes in optimal formats.
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
        
        # 1. Albedo: 8-bit PNG (fast, lightweight, standard)
        if export_config.get('albedo', False) and 'albedo' in lighting_results and lighting_results['albedo'] is not None:
            albedo = resize_to_original(lighting_results['albedo'], original_size)
            albedo_uint8 = ensure_uint8(albedo)
            albedo_png_path = dirs['albedo'] / f"albedo_{frame_number}.png"
            Image.fromarray(albedo_uint8).save(albedo_png_path)
            
            saved_files['albedo'] = str(albedo_png_path)
            previews['albedo'] = albedo_uint8
        
        # 2. Specular: 8-bit PNG (fast, lightweight mask)
        if export_config.get('specular', False) and 'specular' in lighting_results and lighting_results['specular'] is not None:
            specular = resize_to_original(lighting_results['specular'], original_size)
            specular_uint8 = ensure_uint8(specular)
            specular_png_path = dirs['specular'] / f"specular_{frame_number}.png"
            
            if len(specular_uint8.shape) == 2:
                Image.fromarray(specular_uint8, mode='L').save(specular_png_path)
            else:
                Image.fromarray(specular_uint8).save(specular_png_path)
                
            saved_files['specular'] = str(specular_png_path)
            previews['specular'] = specular_uint8
        
        # 3. Depth: 32-bit float OpenEXR (.exr) - Smooth, zero banding
        if export_config.get('depth', False) and 'depth' in geometry_results and geometry_results['depth'] is not None:
            if 'depth_float' in geometry_results and geometry_results['depth_float'] is not None:
                depth_src = geometry_results['depth_float']
            else:
                depth_src = geometry_results['depth'].astype(np.float32) / 255.0
                
            depth_f32 = cv2.resize(depth_src, original_size, interpolation=cv2.INTER_LINEAR).astype(np.float32)
            depth_f32 = np.clip(depth_f32, 0.0, 1.0)
            
            depth_exr_path = dirs['depth'] / f"depth_{frame_number}.exr"
            _safe_write_exr(depth_exr_path, depth_f32)
            
            saved_files['depth'] = str(depth_exr_path)
            # Monotonic 8-bit preview for UI display directly in memory
            previews['depth'] = (depth_f32 * 255.0).astype(np.uint8)
        
        # 4. Normal: 32-bit float OpenEXR (.exr) - Smooth vectors for high-end relighting
        if export_config.get('normal', False) and 'normal' in geometry_results and geometry_results['normal'] is not None:
            if 'normal_float' in geometry_results and geometry_results['normal_float'] is not None:
                normal_src = geometry_results['normal_float']
            else:
                normal_src = geometry_results['normal'].astype(np.float32) / 255.0
                
            normal_f32 = cv2.resize(normal_src, original_size, interpolation=cv2.INTER_LINEAR).astype(np.float32)
            normal_f32 = np.clip(normal_f32, 0.0, 1.0)
            normal_bgr = cv2.cvtColor(normal_f32, cv2.COLOR_RGB2BGR)
            
            normal_exr_path = dirs['normal'] / f"normal_{frame_number}.exr"
            _safe_write_exr(normal_exr_path, normal_bgr)
            
            saved_files['normal'] = str(normal_exr_path)
            # RGB 8-bit preview for UI display directly in memory
            previews['normal'] = (normal_f32 * 255.0).astype(np.uint8)
        
        return {
            'saved_files': saved_files,
            'previews': previews
        }
