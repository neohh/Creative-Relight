import torch
import numpy as np
from pathlib import Path
import sys
import os

# Setup paths for bundled external code and model weights
def _setup_cr_lighting_paths():
    """Setup paths for Creative Relight lighting module (bundled Intrinsic v2.1)"""
    if getattr(sys, 'frozen', False):
        # Running in PyInstaller bundle
        base_path = Path(sys._MEIPASS)
        external_path = base_path / "src" / "external"
        models_path = base_path / "models" / "cr_lighting_v21"
    else:
        # Running in development
        base_path = Path(__file__).parent.parent.parent
        external_path = base_path / "src" / "external"
        models_path = base_path / "models" / "cr_lighting_v21"
    
    # Add external path (not cr_lighting itself) to avoid conflicts with src/utils
    if str(external_path) not in sys.path:
        sys.path.append(str(external_path))
    
    return external_path, models_path

_external_path, _cr_models_path = _setup_cr_lighting_paths()

# Import from bundled Creative Relight lighting module  
from cr_lighting.pipeline import load_models, run_pipeline
from cr_lighting.utils.general import view, invert

class LightingModel:
    """Wrapper for the lighting decomposition model to generate albedo, shading, and specular passes"""
    
    # Class variable to store the model (shared across instances)
    _models = None
    _loaded = False
    
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self._load_model()
    
    def _load_model(self):
        """Load the lighting decomposition v2.1 model (only once) from bundled weights"""
        if not LightingModel._loaded:
            print("🎨 Loading Creative Relight lighting model (v2.1)...")
            print(f"   Model path: {_cr_models_path}")
            
            # Set environment variable to use bundled model weights
            os.environ['TORCH_HOME'] = str(_cr_models_path.parent.parent / "models")
            
            # Load models from bundled weights
            # The load_models function will look for checkpoints in TORCH_HOME
            LightingModel._models = load_models('v2.1', device=self.device, checkpoint_dir=str(_cr_models_path))
            LightingModel._loaded = True
            print("✅ Lighting model loaded successfully!")
        self.models = LightingModel._models
    
    def process(self, image_array):
        """
        Process an image to extract albedo, shading, and specular components
        
        Args:
            image_array: numpy array of shape (H, W, 3) with values in [0, 1]
            
        Returns:
            dict with keys: 'albedo', 'shading', 'specular'
        """
        # Run the pipeline
        result = run_pipeline(self.models, image_array, device=self.device, resize_conf=1024)
        
        # Debug: Print available components
        print(f"Available components in result: {list(result.keys())}")
        
        # Extract albedo (prefer high-res version)
        albedo = None
        if 'hr_alb' in result and result['hr_alb'] is not None:
            albedo = view(result['hr_alb'])
        elif 'alb' in result and result['alb'] is not None:
            albedo = view(result['alb'])
        
        # Extract shading (prefer diffuse shading)
        shading = None
        if 'dif_shd' in result and result['dif_shd'] is not None:
            shading = 1 - invert(result['dif_shd'])
        elif 'shd' in result and result['shd'] is not None:
            shading = 1 - invert(result['shd'])
        elif 'hr_shd' in result and result['hr_shd'] is not None:
            shading = 1 - invert(result['hr_shd'])
        
        # Extract specular using the same logic as export_app.py
        specular = None
        specular_key_used = None
        
        # Extended specular detection - check all possible specular-related keys
        specular_keys = [
            'spec', 'spec_res', 'specular', 'specular_residual',
            'pos_res', 'residual', 'diff_res', 'chr'
        ]
        
        for key in specular_keys:
            if key in result and result[key] is not None:
                candidate = result[key]
                if isinstance(candidate, np.ndarray) and len(candidate.shape) >= 2:
                    specular = candidate
                    specular_key_used = key
                    print(f"Found specular component: {key}")
                    break
        
        # If no direct specular found, try to derive from pos_res
        if specular is None and 'pos_res' in result and result['pos_res'] is not None:
            pos_res_data = result['pos_res']
            if len(pos_res_data.shape) == 3:
                specular = np.mean(pos_res_data, axis=2)  # Convert to grayscale
            else:
                specular = pos_res_data
            specular_key_used = "pos_res (derived)"
            print("Deriving specular from pos_res")
        
        # If still no specular, create a default one
        if specular is None and albedo is not None:
            print("Warning: No specular component found, creating default")
            specular = np.zeros_like(albedo)
            if len(specular.shape) == 3:
                specular = np.mean(specular, axis=2)  # Convert to grayscale
        
        # Ensure specular is properly normalized
        if specular is not None:
            if specular.max() > 1.0:
                specular = (specular - specular.min()) / (specular.max() - specular.min())
            
            # Ensure it's grayscale for specular maps
            if len(specular.shape) == 3:
                specular = np.mean(specular, axis=2)
        
        return {
            'albedo': albedo,
            'shading': shading,
            'specular': specular
        }