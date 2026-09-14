import numpy as np
from PIL import Image
from pathlib import Path
import cv2
import re
import sys

# Handle imports for both development and frozen (PyInstaller) environments
try:
    # Try direct import first (dev mode - models/ is in sys.path)
    from lighting_model import LightingModel
    from geometry_model import GeometryModel
except ModuleNotFoundError:
    # Fallback to package import (frozen/built mode)
    from models.lighting_model import LightingModel
    from models.geometry_model import GeometryModel

try:
    # Try direct import first (dev mode - utils/ is in sys.path)
    from image_utils import resize_to_original, ensure_uint8
except ModuleNotFoundError:
    # Fallback to package import (frozen mode)
    from utils.image_utils import resize_to_original, ensure_uint8

class ImageProcessor:
    """Process single images to generate all 5 passes"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.lighting_model = LightingModel(device)
        self.geometry_model = GeometryModel(device)
        self.should_stop = False
        self._auto_index = 0  # fallback counter when no number in filename

    @staticmethod
    def _derive_frame_number(image_input):
        """Extract frame number from the source filename (e.g. '0001.png' -> '0001')
        so batch runs don't overwrite a single *_000000.png. Falls back to None."""
        try:
            if isinstance(image_input, (str, Path)):
                groups = re.findall(r"\d+", Path(str(image_input)).stem)
                if groups:
                    # prefer the last digit group of length >= 2 (ignores things like 'v2')
                    chosen = next((g for g in reversed(groups) if len(g) >= 2), groups[-1])
                    return f"{int(chosen):0{len(chosen)}d}"
        except Exception:
            pass
        return None
    
    def stop(self):
        """Stop processing"""
        self.should_stop = True
    
    def process(self, image_input, output_dir, export_config=None):
        """
        Process a single image and save selected passes
        
        Args:
            image_input: PIL Image, numpy array, or file path
            output_dir: Output directory path
            export_config: Dictionary specifying which passes to export
            start_number: Starting number for file naming
            padding: Number padding for filenames
            
        Returns:
            dict with paths to saved files and preview images
        """
        # Reset stop flag
        self.should_stop = False
        
        # Default export all if not specified
        if export_config is None:
            export_config = {
                'albedo': True,
                'specular': True,
                'depth': True,
                'normal': True
            }
        
        # Check if we should stop before starting
        if self.should_stop:
            return {'saved_files': {}, 'previews': {}, 'message': 'Processing stopped'}
        
        # Load and prepare image
        if isinstance(image_input, str):
            image = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, np.ndarray):
            image = Image.fromarray(image_input)
        else:
            image = image_input
        
        # Get original dimensions
        original_size = image.size
        image_array = np.array(image).astype(np.float32) / 255.0
        
        # Create output directories only for selected passes
        output_path = Path(output_dir)
        dirs = {}
        for component in ['albedo', 'specular', 'depth', 'normal']:
            if export_config.get(component, False):
                dirs[component] = output_path / component
                dirs[component].mkdir(parents=True, exist_ok=True)
        
        # Check stop flag before processing
        if self.should_stop:
            return {'saved_files': {}, 'previews': {}, 'message': 'Processing stopped'}
        
        # Process with lighting model if any lighting passes are needed
        lighting_results = {}
        if any(export_config.get(comp, False) for comp in ['albedo', 'specular']):
            lighting_results = self.lighting_model.process(image_array)
        
        # Check stop flag after lighting processing
        if self.should_stop:
            return {'saved_files': {}, 'previews': {}, 'message': 'Processing stopped'}
        
        # Process with geometry model if any geometry passes are needed
        geometry_results = {}
        if any(export_config.get(comp, False) for comp in ['depth', 'normal']):
            geometry_results = self.geometry_model.process(image_array)
        
        # Save selected components
        saved_files = {}
        previews = {}
        # Use the number from the source filename when available (keeps sequence
        # numbering and prevents every frame overwriting *_000000.png)
        frame_number = self._derive_frame_number(image_input)
        if frame_number is None:
            frame_number = f"{self._auto_index:06d}"
            self._auto_index += 1
        print(f"[IMAGE_PROC] Saving frame number: {frame_number} (from: {image_input})")
        
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
            # Save as grayscale if it's a 2D array
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