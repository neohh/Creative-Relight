import torch
import numpy as np
from pathlib import Path
import sys

# Add the Intrinsic path to system path
INTRINSIC_PATH = Path("C:/Users/Creative Twins/Videos/Projects/Intrinsic V3/Intrinsic")
if str(INTRINSIC_PATH) not in sys.path:
    sys.path.insert(0, str(INTRINSIC_PATH))

# Import only what we need for albedo and specular
from intrinsic.pipeline import load_models, run_pipeline
from chrislib.general import view, invert

class LightingModel:
    """Optimized lighting model for albedo and specular only"""
    
    # Class variable to store the model (shared across instances)
    _models = None
    _loaded = False
    
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self._load_model()
    
    def _load_model(self):
        """Load the v2.1 models for albedo and specular"""
        if not LightingModel._loaded:
            print("Loading lighting model...")
            
            try:
                # Load v2.1 model for albedo and specular extraction
                LightingModel._models = load_models('v2.1', device=self.device)
                LightingModel._loaded = True
                print("✅ Lighting model loaded successfully!")
            except Exception as e:
                print(f"❌ Error loading lighting model: {e}")
                LightingModel._models = None
                LightingModel._loaded = False
                raise e
        
        self.models = LightingModel._models
    
    def process(self, image_array):
        """
        Process image to extract albedo and specular, maintaining original dimensions
        
        Args:
            image_array: numpy array of shape (H, W, 3) with values in [0, 1]
            
        Returns:
            dict with keys: 'albedo', 'specular'
        """
        if self.models is None:
            raise RuntimeError("Models not loaded. Please check model loading.")
        
        # Run pipeline for albedo and specular extraction
        result = run_pipeline(
            self.models, 
            image_array, 
            device=self.device, 
            resize_conf=1024  # Use same config as working version!
        )
        
        # Extract albedo (prefer high-res version)
        albedo = None
        if 'hr_alb' in result and result['hr_alb'] is not None:
            albedo = view(result['hr_alb'])
        elif 'alb' in result and result['alb'] is not None:
            albedo = view(result['alb'])
        
        # Extract specular using the same logic as working version
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
        
        # Ensure specular is properly normalized (keep in [0,1] range)
        if specular is not None:
            if specular.max() > 1.0:
                specular = (specular - specular.min()) / (specular.max() - specular.min())
            
            # Ensure it's grayscale for specular maps
            if len(specular.shape) == 3:
                specular = np.mean(specular, axis=2)
        
        # Return raw values (no uint8 conversion - let processors handle that)
        return {
            'albedo': albedo,
            'specular': specular
        }
