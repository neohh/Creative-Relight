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
    from models.lighting_model import LightingModel
    from models.geometry_model import GeometryModel

try:
    from image_utils import resize_to_original, ensure_uint8
except ModuleNotFoundError:
    from utils.image_utils import resize_to_original, ensure_uint8

class ImageProcessor:
    """Process single images to generate all 5 passes"""
    
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
        Process a single image and save selected passes
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
        
        # Auto-detect next free frame number by checking existing files
        max_existing = -1
        for comp, comp_dir in dirs.items():
            if comp_dir.exists():
                for item in comp_dir.iterdir():
                    m = re.match(rf"^{comp}_(\d+)\.png$", item.name)
                    if m:
                        max_existing = max(max_existing, int(m.group(1)))
        
        if max_existing >= 0:
            assigned_num = max_existing + 1
        else:
            assigned_num = int(start_number) if start_number is not None else 0
            
        frame_number = f"{assigned_num:0{padding}d}"
        
        # Save albedo
        if export_config.get('albedo', False) and 'albedo' in lighting_results and lighting_results['albedo'] is not None:
            albedo = resize_to_original(lighting_results['albedo'], original_size)
            albedo_uint8 = ensure_uint8(albedo)
            albedo_path = dirs['albedo'] / f"albedo_{frame_number}.png"
            Image.fromarray(albedo_uint8).save(albedo_path)
            saved_files['albedo'] = str(albedo_path)
            previews['albedo'] = albedo_uint8
        
        # Save specular
        if export_config.get('specular', False) and 'specular' in lighting_results and lighting_results['specular'] is not None:
            specular = resize_to_original(lighting_results['specular'], original_size)
            specular_uint8 = ensure_uint8(specular)
            specular_path = dirs['specular'] / f"specular_{frame_number}.png"
            if len(specular_uint8.shape) == 2:
                Image.fromarray(specular_uint8, mode='L').save(specular_path)
            else:
                Image.fromarray(specular_uint8).save(specular_path)
            saved_files['specular'] = str(specular_path)
            previews['specular'] = specular_uint8
        
        # Save depth
        if export_config.get('depth', False) and 'depth' in geometry_results and geometry_results['depth'] is not None:
            depth = cv2.resize(geometry_results['depth'], original_size, interpolation=cv2.INTER_LINEAR)
            depth_path = dirs['depth'] / f"depth_{frame_number}.png"
            cv2.imwrite(str(depth_path), depth)
            saved_files['depth'] = str(depth_path)
            previews['depth'] = depth
        
        # Save normal
        if export_config.get('normal', False) and 'normal' in geometry_results and geometry_results['normal'] is not None:
            normal = cv2.resize(geometry_results['normal'], original_size, interpolation=cv2.INTER_LINEAR)
            normal_path = dirs['normal'] / f"normal_{frame_number}.png"
            cv2.imwrite(str(normal_path), cv2.cvtColor(normal, cv2.COLOR_RGB2BGR))
            saved_files['normal'] = str(normal_path)
            previews['normal'] = normal
        
        return {
            'saved_files': saved_files,
            'previews': previews
        }
