import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Setup paths for bundled external code and model weights
def _setup_cr_geometry_paths():
    """Setup paths for Creative Relight geometry module (bundled MoGe)"""
    if getattr(sys, 'frozen', False):
        # Running in PyInstaller bundle
        base_path = Path(sys._MEIPASS)
        external_path = base_path / "src" / "external"
        models_path = base_path / "models" / "cr_geometry_vitl"
    else:
        # Running in development
        base_path = Path(__file__).parent.parent.parent
        external_path = base_path / "src" / "external"
        models_path = base_path / "models" / "cr_geometry_vitl"
    
    # Add external path to Python path (not cr_geometry itself, but its parent)
    # Important: append instead of insert(0) so src/utils takes precedence over cr_geometry/utils
    if str(external_path) not in sys.path:
        sys.path.append(str(external_path))
    
    return external_path, models_path

_external_path, _cr_geometry_models_path = _setup_cr_geometry_paths()

# Import from bundled Creative Relight geometry module
from cr_geometry.model.v2 import MoGeModel

class GeometryModel:
    """Wrapper for the geometry model to generate depth and normal passes"""
    
    # Class variable to store the model (shared across instances)
    _model = None
    _loaded = False
    
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self._load_model()
    
    def _load_model(self):
        """Load the geometry model (only once) from bundled weights"""
        if not GeometryModel._loaded:
            print("🌍 Loading Creative Relight geometry model (MoGe ViT-L)...")
            print(f"   Model path: {_cr_geometry_models_path}")
            
            # Find the actual model.pt file in the HuggingFace cache structure
            import glob
            model_files = glob.glob(str(_cr_geometry_models_path / "snapshots" / "*" / "model.pt"))
            if not model_files:
                # Fallback: try direct path
                model_file = _cr_geometry_models_path / "model.pt"
            else:
                model_file = Path(model_files[0])
            
            print(f"   Loading from: {model_file}")
            
            # Load from bundled local weights instead of downloading
            GeometryModel._model = MoGeModel.from_pretrained(
                str(model_file),
                local_files_only=True
            ).to(self.device)
            
            GeometryModel._model.eval()
            GeometryModel._loaded = True
            print("✅ Geometry model loaded successfully!")
        self.model = GeometryModel._model
    
    def process(self, image_array):
        """
        Process an image to extract depth and normal maps
        
        Args:
            image_array: numpy array of shape (H, W, 3) with values in [0, 1]
            
        Returns:
            dict with keys: 'depth', 'normal'
        """
        # Convert to tensor and prepare for model
        input_tensor = torch.tensor(image_array, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            output = self.model.infer(input_tensor[0])
        
        # Process depth map
        depth_map = output["depth"].cpu().numpy()
        depth_map = np.nan_to_num(depth_map, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Convert to disparity and colorize
        disp = 1 / np.where(depth_map > 0, depth_map, np.nan)
        min_disp, max_disp = np.nanquantile(disp, 0.001), np.nanquantile(disp, 0.99)
        disp = (disp - min_disp) / (max_disp - min_disp)
        colored_depth = np.nan_to_num(plt.cm.Spectral(1.0 - disp)[..., :3], 0)
        colored_depth = np.ascontiguousarray((colored_depth.clip(0, 1) * 255).astype(np.uint8))
        
        # Convert to grayscale
        gray_depth = cv2.cvtColor(colored_depth, cv2.COLOR_RGB2GRAY)
        
        # Process normal map
        normal_map = output["normal"].cpu().numpy()
        normal_map = normal_map * [0.5, -0.5, -0.5] + 0.5
        normal_map = (normal_map.clip(0, 1) * 255).astype(np.uint8)
        
        return {
            'depth': gray_depth,
            'normal': normal_map
        }