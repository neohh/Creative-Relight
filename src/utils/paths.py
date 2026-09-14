"""
Helper module to determine correct paths for models in both dev and production
"""
import sys
import os
from pathlib import Path

def get_models_directory():
    """
    Get the correct models directory for both development and production.
    
    In development: creative-relight-cloud/models/
    In production (frozen exe): User's AppData/Local/CreativeRelight/models/
    
    Returns:
        Path: Absolute path to models directory
    """
    if getattr(sys, 'frozen', False):
        # Running as compiled exe - use user's AppData
        appdata = Path(os.getenv('LOCALAPPDATA', os.path.expanduser('~')))
        models_dir = appdata / 'CreativeRelight' / 'models'
    else:
        # Running in development - use project models folder
        # This file is in: creative-relight-cloud/src/utils/paths.py
        project_root = Path(__file__).parent.parent.parent
        models_dir = project_root / 'models'
    
    # Ensure directory exists
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir

def get_project_root():
    """
    Get the project root directory.
    
    In development: creative-relight-cloud/
    In production: Directory containing the exe
    
    Returns:
        Path: Absolute path to project root
    """
    if getattr(sys, 'frozen', False):
        # Running as compiled exe - use exe directory
        return Path(sys.executable).parent
    else:
        # Running in development - use project root
        return Path(__file__).parent.parent.parent

def get_config_path():
    """
    Get the path to config.yaml
    
    Returns:
        Path: Absolute path to config.yaml
    """
    if getattr(sys, 'frozen', False):
        # Running as frozen exe - config is in _internal folder (PyInstaller bundle)
        config_path = Path(sys._MEIPASS) / 'config.yaml'
        print(f"[DEBUG PATHS] Frozen: sys._MEIPASS={sys._MEIPASS}, config_path={config_path}")
    else:
        # Running in development
        config_path = get_project_root() / 'config.yaml'
        print(f"[DEBUG PATHS] Development: config_path={config_path}")
    
    return config_path

def configure_pytorch_hub_cache():
    """
    Configure PyTorch Hub to use our models directory instead of default cache.
    This prevents PyTorch Hub from downloading to user's home directory.
    
    Should be called early in application startup, before any torch.hub.load() calls.
    """
    models_dir = get_models_directory()
    
    # PyTorch Hub uses TORCH_HOME environment variable
    # If not set, it defaults to ~/.cache/torch (Linux) or %USERPROFILE%\.cache\torch (Windows)
    # We want it to use our models directory instead
    os.environ['TORCH_HOME'] = str(models_dir)
    
    # Also ensure hub subdirectory exists
    hub_dir = models_dir / 'hub'
    hub_dir.mkdir(exist_ok=True)
    
    print(f"[CONFIG] PyTorch Hub cache configured: {hub_dir}")
    return hub_dir
