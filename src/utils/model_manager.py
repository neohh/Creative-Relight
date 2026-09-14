"""
Model Manager for Creative Relight
Manages AI model downloads, caching, and initialization
"""

import yaml
from pathlib import Path
from typing import Dict, List, Optional, Callable
from .model_downloader import ModelDownloader


class ModelManager:
    """Manage AI models for Creative Relight"""
    
    # Default model configuration
    DEFAULT_CONFIG = {
        "huggingface_repo": "creative-twins/creative-relight-models",
        "models": {
            "lighting_v21": {
                "files": [
                    "stage_0_v21.pt",
                    "stage_1_v21.pt", 
                    "stage_2_v21.pt",
                    "stage_3_v21.pt",
                    "stage_4_v21.pt"
                ],
                "subfolder": "cr_lighting_v21",
                "required_for": ["albedo", "specular"],
                "size_mb": 1693
            },
            "geometry_vitl": {
                "files": ["model.pt"],
                "subfolder": "cr_geometry_vitl/snapshots/b135031bae30b5ac2ae141a0e68717795ce38340",
                "required_for": ["depth", "normal"],
                "size_mb": 1262
            },
            "resnext": {
                "files": ["ig_resnext101_32x8-c38310e5.pth"],
                "subfolder": "hub/checkpoints",
                "required_for": ["albedo", "specular"],
                "size_mb": 340
            }
        }
    }
    
    def __init__(self, config_path: Optional[str] = None, cache_dir: str = "models"):
        """
        Initialize model manager
        
        Args:
            config_path: Path to model configuration YAML file
            cache_dir: Directory to cache downloaded models
        """
        self.cache_dir = Path(cache_dir)
        self.downloader = ModelDownloader(cache_dir=str(self.cache_dir))
        
        # Load configuration
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                self.config = config.get('models', self.DEFAULT_CONFIG)
        else:
            self.config = self.DEFAULT_CONFIG
        
        self.repo_id = self.config.get('huggingface_repo', 'creative-twins/creative-relight-models')
    
    def check_models_available(self, model_names: Optional[List[str]] = None) -> Dict[str, bool]:
        """
        Check which models are available locally
        
        Args:
            model_names: List of model names to check, or None for all models
            
        Returns:
            Dict mapping model name to availability status
        """
        if model_names is None:
            model_names = list(self.config['models'].keys())
        
        availability = {}
        
        for model_name in model_names:
            if model_name not in self.config['models']:
                availability[model_name] = False
                continue
            
            model_config = self.config['models'][model_name]
            subfolder = model_config.get('subfolder', '')
            files = model_config.get('files', [])
            
            # Check if all files exist
            all_exist = True
            for filename in files:
                file_path = self.cache_dir / subfolder / filename
                if not file_path.exists():
                    all_exist = False
                    break
            
            availability[model_name] = all_exist
        
        return availability
    
    def get_missing_models(self, required_passes: Optional[List[str]] = None) -> List[str]:
        """
        Get list of models that need to be downloaded
        
        Args:
            required_passes: List of pass types needed (e.g., ['albedo', 'depth'])
                           If None, checks all models
        
        Returns:
            List of model names that are missing
        """
        missing = []
        models_to_check = []
        
        if required_passes:
            # Find models required for these passes
            for model_name, model_config in self.config['models'].items():
                required_for = model_config.get('required_for', [])
                if any(pass_type in required_for for pass_type in required_passes):
                    models_to_check.append(model_name)
        else:
            models_to_check = list(self.config['models'].keys())
        
        availability = self.check_models_available(models_to_check)
        
        for model_name, available in availability.items():
            if not available:
                missing.append(model_name)
        
        return missing
    
    def calculate_download_size(self, model_names: List[str]) -> int:
        """
        Calculate total download size for given models
        
        Args:
            model_names: List of model names
            
        Returns:
            Total size in MB
        """
        total_mb = 0
        
        for model_name in model_names:
            if model_name in self.config['models']:
                size_mb = self.config['models'][model_name].get('size_mb', 0)
                total_mb += size_mb
        
        return total_mb
    
    def download_models(
        self,
        model_names: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
        force_redownload: bool = False
    ) -> Dict[str, bool]:
        """
        Download models from Hugging Face
        
        Args:
            model_names: List of models to download, or None for all
            progress_callback: Optional callback(model_name, downloaded, total)
            force_redownload: If True, redownload even if files exist
            
        Returns:
            Dict mapping model name to success status
        """
        if model_names is None:
            model_names = list(self.config['models'].keys())
        
        results = {}
        
        for model_name in model_names:
            if model_name not in self.config['models']:
                print(f"⚠ Unknown model: {model_name}")
                results[model_name] = False
                continue
            
            model_config = self.config['models'][model_name]
            subfolder = model_config.get('subfolder', '')
            files = model_config.get('files', [])
            
            print(f"\n📦 Downloading model: {model_name}")
            print(f"   Size: ~{model_config.get('size_mb', 0)} MB")
            print(f"   Files: {len(files)}")
            
            all_success = True
            
            for filename in files:
                # Check if already exists
                output_path = self.cache_dir / subfolder / filename
                
                if output_path.exists() and not force_redownload:
                    print(f"   ✓ Already cached: {filename}")
                    continue
                
                # Download file
                def file_progress(downloaded, total):
                    if progress_callback:
                        progress_callback(f"{model_name}/{filename}", downloaded, total)
                
                downloaded_path = self.downloader.download_from_huggingface(
                    repo_id=self.repo_id,
                    filename=filename,
                    subfolder=subfolder,
                    progress_callback=file_progress
                )
                
                if not downloaded_path:
                    print(f"   ✗ Failed to download: {filename}")
                    all_success = False
                    break
            
            results[model_name] = all_success
            
            if all_success:
                print(f"✓ Model ready: {model_name}")
            else:
                print(f"✗ Model download failed: {model_name}")
        
        return results
    
    def get_model_paths(self, model_name: str) -> Dict[str, Path]:
        """
        Get paths to model files
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dict mapping filename to full path
        """
        if model_name not in self.config['models']:
            return {}
        
        model_config = self.config['models'][model_name]
        subfolder = model_config.get('subfolder', '')
        files = model_config.get('files', [])
        
        paths = {}
        for filename in files:
            paths[filename] = self.cache_dir / subfolder / filename
        
        return paths
    
    def ensure_models_for_passes(
        self,
        pass_types: List[str],
        progress_callback: Optional[Callable[[str, int, int], None]] = None
    ) -> bool:
        """
        Ensure all models required for given pass types are available
        
        Args:
            pass_types: List of pass types (e.g., ['albedo', 'depth'])
            progress_callback: Optional progress callback
            
        Returns:
            True if all models are ready, False otherwise
        """
        missing = self.get_missing_models(pass_types)
        
        if not missing:
            print("✓ All required models are already available")
            return True
        
        print(f"\n⬇ Need to download {len(missing)} model(s)")
        total_size = self.calculate_download_size(missing)
        print(f"   Total download size: ~{total_size} MB ({total_size/1024:.2f} GB)")
        
        results = self.download_models(missing, progress_callback)
        
        return all(results.values())


# Example usage
if __name__ == "__main__":
    # Test model manager
    manager = ModelManager(cache_dir="models")
    
    print("=" * 60)
    print("Creative Relight - Model Manager")
    print("=" * 60)
    
    # Check what's available
    print("\n📊 Model Status:")
    availability = manager.check_models_available()
    for model_name, available in availability.items():
        status = "✓ Available" if available else "✗ Missing"
        size = manager.config['models'][model_name].get('size_mb', 0)
        print(f"   {model_name:20s} {status:15s} ({size} MB)")
    
    # Check what's needed for specific passes
    print("\n🎨 Required for Albedo + Specular:")
    missing = manager.get_missing_models(['albedo', 'specular'])
    if missing:
        total_size = manager.calculate_download_size(missing)
        print(f"   Need to download: {', '.join(missing)}")
        print(f"   Total size: ~{total_size} MB")
    else:
        print("   ✓ All models available")
    
    print("\n🌊 Required for Depth + Normal:")
    missing = manager.get_missing_models(['depth', 'normal'])
    if missing:
        total_size = manager.calculate_download_size(missing)
        print(f"   Need to download: {', '.join(missing)}")
        print(f"   Total size: ~{total_size} MB")
    else:
        print("   ✓ All models available")
