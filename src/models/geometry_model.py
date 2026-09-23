import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
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
    try:
        from paths import get_models_directory
    except ImportError:
        try:
            from utils.paths import get_models_directory
        except ImportError:
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
    
    if getattr(sys, 'frozen', False):
        base_path = Path(sys._MEIPASS)
        external_path = base_path / "src" / "external"
    else:
        base_path = Path(__file__).parent.parent.parent
        external_path = base_path / "src" / "external"
    
    if str(external_path) not in sys.path:
        sys.path.append(str(external_path))
    
    return external_path, models_path

_external_path, _cr_geometry_models_path = _setup_cr_geometry_paths()

from cr_geometry.model.v2 import MoGeModel

class GeometryModel:
    """Wrapper for the geometry model to generate depth and normal passes"""
    
    _model = None
    _loaded = False
    
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self._load_model()
    
    def _load_model(self):
        """Load the geometry model (only once) from bundled weights or HuggingFace"""
        if not GeometryModel._loaded:
            print("🌍 Loading Creative Relight geometry model (MoGe ViT-L)...")
            model_file = _cr_geometry_models_path / "model.pt"
            
            if model_file.exists():
                print(f"   Model path: {_cr_geometry_models_path}")
                print(f"   Loading from: {model_file}")
                GeometryModel._model = MoGeModel.from_pretrained(
                    str(model_file),
                    local_files_only=True
                ).to(self.device)
            else:
                print(f"   Local model not found, downloading from HuggingFace...")
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
        Process an image to extract depth and normal maps (both 32-bit float and 8-bit preview)
        
        Args:
            image_array: numpy array of shape (H, W, 3) with values in [0, 1]
            
        Returns:
            dict with keys: 'depth' (uint8), 'depth_float' (float32), 'depth_raw' (float32),
                            'normal' (uint8), 'normal_float' (float32)
        """
        input_tensor = torch.tensor(image_array, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model.infer(input_tensor[0])
        
        # 1. Metric raw depth (float32, in meters)
        depth_map = output["depth"].cpu().numpy().astype(np.float32)
        depth_map = np.nan_to_num(depth_map, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 2. Smooth normalized disparity (float32, [0, 1]) - NO color quantization!
        # Near = 1.0 (bright), Far = 0.0 (dark)
        disp = 1.0 / np.where(depth_map > 0, depth_map, np.nan)
        min_disp, max_disp = np.nanquantile(disp, 0.001), np.nanquantile(disp, 0.99)
        if max_disp > min_disp:
            disp_norm = (disp - min_disp) / (max_disp - min_disp)
        else:
            disp_norm = np.zeros_like(disp)
        disp_norm = np.nan_to_num(disp_norm, nan=0.0, posinf=1.0, neginf=0.0)
        depth_float = np.clip(disp_norm, 0.0, 1.0).astype(np.float32)
        
        # 8-bit preview for UI (monotonically mapped from smooth float)
        gray_depth = (depth_float * 255.0).astype(np.uint8)
        
        # 3. Normal map (float32 [0, 1] and uint8)
        normal_map = output["normal"].cpu().numpy().astype(np.float32)
        normal_float = normal_map * [0.5, -0.5, -0.5] + 0.5
        normal_float = np.clip(normal_float, 0.0, 1.0).astype(np.float32)
        normal_uint8 = (normal_float * 255.0).astype(np.uint8)
        
        return {
            'depth': gray_depth,
            'depth_float': depth_float,
            'depth_raw': depth_map,
            'normal': normal_uint8,
            'normal_float': normal_float
        }
