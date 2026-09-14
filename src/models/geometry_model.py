import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import warnings

# Suppress library warnings for cleaner user experience
warnings.filterwarnings('ignore', message='.*symlinks are not supported on Windows.*')
warnings.filterwarnings('ignore', category=UserWarning, module='huggingface_hub')
warnings.filterwarnings('ignore', category=UserWarning, module='torch.hub')

# Setup paths for bundled external code and model weights
def _setup_cr_geometry_paths():
    """Setup paths for Creative Relight geometry module (bundled MoGe)"""
    # Get models directory (persistent location for both dev and production)
    try:
        # Try direct import first (dev mode - utils/ is in sys.path)
        from paths import get_models_directory
    except ImportError:
        try:
            # Fallback to package import (frozen build)
            from utils.paths import get_models_directory
        except ImportError:
            # Fallback: manually implement the function
            import os
            if getattr(sys, 'frozen', False):
                appdata = Path(os.getenv('LOCALAPPDATA', os.path.expanduser('~')))
                models_dir = appdata / 'CreativeRelight' / 'models'
            else:
                project_root = Path(__file__).parent.parent.parent
                models_dir = project_root / 'models'
            models_dir.mkdir(parents=True, exist_ok=True)
            def get_models_directory():
                return models_dir
    
    models_path = get_models_directory() / "cr_geometry_vitl"
    
    # Get external code path
    if getattr(sys, 'frozen', False):
        # Running in PyInstaller bundle - code is in temp folder
        base_path = Path(sys._MEIPASS)
        external_path = base_path / "src" / "external"
    else:
        # Running in development
        base_path = Path(__file__).parent.parent.parent
        external_path = base_path / "src" / "external"
    
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
        """Load the geometry model (only once) from bundled weights or HuggingFace"""
        if not GeometryModel._loaded:
            print("🌍 Loading Creative Relight geometry model (MoGe ViT-L)...")
            
            # Try to load from local path first
            model_file = _cr_geometry_models_path / "model.pt"
            
            if model_file.exists():
                # Load from local bundled weights
                print(f"   Model path: {_cr_geometry_models_path}")
                print(f"   Loading from: {model_file}")
                GeometryModel._model = MoGeModel.from_pretrained(
                    str(model_file),
                    local_files_only=True
                ).to(self.device)
            else:
                # Download from YOUR HuggingFace repo (cloud edition)
                print(f"   Local model not found, downloading from HuggingFace...")
                
                # Load config to get the correct repo
                import yaml
                config_path = Path(__file__).parent.parent.parent / "config.yaml"
                if config_path.exists():
                    with open(config_path, 'r') as f:
                        config = yaml.safe_load(f)
                    repo_id = config['models']['download']['repository']
                    print(f"   Repository: {repo_id}")
                else:
                    raise FileNotFoundError(
                        "Config file not found. Cannot determine HuggingFace repository.\n"
                        "Please ensure config.yaml exists in the project root."
                    )
                
                # Download from HuggingFace
                from huggingface_hub import hf_hub_download
                
                print(f"   Downloading cr_geometry_vitl/model.pt from {repo_id}...")
                cached_file = hf_hub_download(
                    repo_id=repo_id,
                    filename="cr_geometry_vitl/model.pt",
                    cache_dir=str(_cr_geometry_models_path.parent)
                )
                
                print(f"   Downloaded to: {cached_file}")
                GeometryModel._model = MoGeModel.from_pretrained(
                    cached_file,
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