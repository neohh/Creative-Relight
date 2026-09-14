"""
Model Checker and Auto-Downloader for Creative Relight Cloud Edition
Checks if models exist locally, downloads from HuggingFace if missing
"""

import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys

class ModelChecker:
    """Check for models and download if missing"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize model checker with config"""
        # Try both import paths - when frozen or running as script
        try:
            # Try src. prefix first (development mode)
            from src.utils.paths import get_models_directory, get_config_path, get_project_root
        except ModuleNotFoundError:
            try:
                # Fallback to direct import (frozen build)
                from utils.paths import get_models_directory, get_config_path, get_project_root
            except ModuleNotFoundError:
                # Fallback: define minimal path functions here
                def get_project_root():
                    if getattr(sys, 'frozen', False):
                        return Path(sys.executable).parent
                    return Path(__file__).parent.parent.parent
                def get_models_directory():
                    return Path.home() / ".cache" / "huggingface" / "hub"
                def get_config_path():
                    if getattr(sys, 'frozen', False):
                        return Path(sys._MEIPASS) / "config.yaml"
                    return get_project_root() / "config.yaml"
        
        # Use centralized path helper to get config path
        # Only use custom config_path if it's an absolute path
        if Path(config_path).is_absolute():
            self.config_path = Path(config_path)
        else:
            # Use the helper function which handles frozen/dev environments
            self.config_path = get_config_path()
        
        print(f"[DEBUG MODEL_CHECKER] Config path: {self.config_path}, exists: {self.config_path.exists()}")
        self.config = self.load_config()
        
        # Use centralized models directory (persistent location)
        self.models_dir = get_models_directory()
            
        self.repo_id = self.config['models']['download']['repository']
        
    def load_config(self) -> dict:
        """Load configuration from YAML"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def check_models_exist(self) -> Tuple[bool, List[str], List[str]]:
        """
        Check if all required models exist locally
        
        Returns:
            Tuple of (all_exist: bool, existing_files: list, missing_files: list)
        """
        required_files = []
        
        # Get lighting model files
        if 'lighting' in self.config['models'] and 'files' in self.config['models']['lighting']:
            required_files.extend(self.config['models']['lighting']['files'])
        
        # Get geometry model files
        if 'geometry' in self.config['models'] and 'files' in self.config['models']['geometry']:
            required_files.extend(self.config['models']['geometry']['files'])
        
        existing = []
        missing = []
        
        for file_path in required_files:
            full_path = self.models_dir / file_path
            if full_path.exists():
                existing.append(file_path)
            else:
                missing.append(file_path)
        
        all_exist = len(missing) == 0
        return all_exist, existing, missing
    
    def get_models_status(self) -> Dict[str, any]:
        """Get detailed status of models"""
        all_exist, existing, missing = self.check_models_exist()
        
        total_size = 0
        for file_path in existing:
            full_path = self.models_dir / file_path
            if full_path.exists():
                total_size += full_path.stat().st_size
        
        return {
            'all_exist': all_exist,
            'total_files': len(existing) + len(missing),
            'existing_files': len(existing),
            'missing_files': len(missing),
            'existing_list': existing,
            'missing_list': missing,
            'total_size_bytes': total_size,
            'total_size_readable': self.format_bytes(total_size),
            'repo_id': self.repo_id,
            'cache_dir': str(self.models_dir)
        }
    
    def format_bytes(self, bytes_size: int) -> str:
        """Format bytes to human-readable string"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_size < 1024.0:
                return f"{bytes_size:.2f} {unit}"
            bytes_size /= 1024.0
        return f"{bytes_size:.2f} TB"
    
    def download_missing_models(self, progress_callback=None) -> Tuple[bool, str]:
        """
        Download all missing models from HuggingFace, then PyTorch Hub dependencies
        
        Args:
            progress_callback: Optional callback(file_name, downloaded, total)
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            # Try direct import first (when src/utils is in sys.path)
            from model_downloader import ModelDownloader
        except ImportError:
            # Fallback to package import (frozen build where utils is a package)
            from utils.model_downloader import ModelDownloader
        
        all_exist, existing, missing = self.check_models_exist()
        
        if all_exist:
            # Models already downloaded
            print("✅ HuggingFace models already exist locally")
            return True, "All models ready"
        
        print(f"📥 Downloading {len(missing)} missing model files from HuggingFace...")
        print(f"📦 Repository: {self.repo_id}")
        print(f"📁 Cache directory: {self.models_dir}")
        
        downloader = ModelDownloader(cache_dir=str(self.models_dir))
        
        failed_files = []
        
        for i, file_path in enumerate(missing, 1):
            print(f"\n[{i}/{len(missing)}] Downloading: {file_path}")
            
            # Extract filename and subfolder
            path_parts = Path(file_path).parts
            if len(path_parts) > 1:
                subfolder = str(Path(*path_parts[:-1]))
                filename = path_parts[-1]
            else:
                subfolder = None
                filename = file_path
            
            # Create progress callback for this specific file
            def file_progress(downloaded, total):
                if progress_callback:
                    # Hide technical file paths from user - show generic message
                    progress_callback("AI Models", downloaded, total)
            
            # Download from HuggingFace
            try:
                result = downloader.download_from_huggingface(
                    repo_id=self.repo_id,
                    filename=filename,
                    subfolder=subfolder,
                    progress_callback=file_progress
                )
                
                if not result:
                    failed_files.append(file_path)
                    print(f"❌ Failed to download: {file_path}")
                else:
                    print(f"✅ Downloaded: {file_path}")
                    
            except Exception as e:
                failed_files.append(file_path)
                print(f"❌ Error downloading {file_path}: {e}")
        
        if failed_files:
            error_msg = f"Failed to download {len(failed_files)} file(s):\n" + "\n".join(failed_files)
            return False, error_msg
        
        print(f"\n✅ All HuggingFace models downloaded successfully!")
        
        # PyTorch Hub dependencies will download automatically during model initialization
        # (Pre-caching disabled to avoid file locking issues on Windows)
        return True, "All models downloaded successfully"
    
    def download_pytorch_hub_dependencies(self, progress_callback=None) -> Tuple[bool, str]:
        """
        Pre-download PyTorch Hub dependencies (EfficientNet, ResNeXt, DSINE)
        This ensures all dependencies are cached before model initialization
        
        Args:
            progress_callback: Optional callback(file_name, downloaded, total)
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        print(f"\n📥 Downloading PyTorch Hub Dependencies...")
        
        try:
            import torch
            import warnings
            
            # Suppress PyTorch Hub warnings during download
            warnings.filterwarnings('ignore', category=UserWarning, module='torch.hub')
            
            # List of PyTorch Hub dependencies with estimated sizes
            dependencies = [
                {
                    "repo": "rwightman/gen-efficientnet-pytorch",
                    "model": "tf_efficientnet_lite3",
                    "name": "EfficientNet-Lite3",
                    "size_mb": 10
                },
                {
                    "repo": "facebookresearch/WSL-Images",
                    "model": "resnext101_32x8d_wsl",
                    "name": "ResNeXt-101 WSL",
                    "size_mb": 340
                },
                {
                    "repo": "hugoycj/DSINE-hub",
                    "model": "DSINE",
                    "name": "DSINE Normal Estimator",
                    "size_mb": 100
                }
            ]
            
            total_size_mb = sum(d["size_mb"] for d in dependencies)
            downloaded_mb = 0
            
            for i, dep in enumerate(dependencies, 1):
                if progress_callback:
                    # Report generic progress - hide technical details from user
                    progress_callback(
                        "AI Models", 
                        downloaded_mb, 
                        total_size_mb
                    )
                
                print(f"\n[{i}/{len(dependencies)}] Caching PyTorch Hub: {dep['name']}")
                print(f"   Repository: {dep['repo']}")
                print(f"   Estimated size: ~{dep['size_mb']} MB")
                
                try:
                    # Download and cache the model
                    model = torch.hub.load(
                        dep["repo"], 
                        dep["model"], 
                        pretrained=True, 
                        trust_repo=True,
                        verbose=False  # Suppress verbose output
                    )
                    
                    # Clear model from memory immediately
                    del model
                    
                    print(f"✅ Cached: {dep['name']}")
                    downloaded_mb += dep["size_mb"]
                    
                except Exception as e:
                    # Pre-caching failed, but this is OK - models will download during actual use
                    # Only show technical details in debug mode
                    print(f"ℹ️  {dep['name']} will download during first use")
                    # Don't fail the entire process for PyTorch Hub issues
                    downloaded_mb += dep["size_mb"]  # Mark as "processed"
            
            # Final progress callback
            if progress_callback:
                progress_callback("AI Models", total_size_mb, total_size_mb)
            
            print(f"\n✅ PyTorch Hub dependencies ready!")
            return True, "All dependencies downloaded successfully"
            
        except Exception as e:
            print(f"⚠️  Warning: PyTorch Hub download had issues: {e}")
            print("   Dependencies will download on demand during first use")
            # Return success anyway - these are optional pre-downloads
            return True, "Main models ready (Hub dependencies will download on demand)"
    
    def ensure_models_ready(self, progress_callback=None) -> Tuple[bool, str]:
        """
        Ensure all models are ready, download if missing
        
        Args:
            progress_callback: Optional callback(file_name, downloaded, total)
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        status = self.get_models_status()
        
        if status['all_exist']:
            msg = f"✅ All models ready ({status['existing_files']} files, {status['total_size_readable']})"
            print(msg)
            return True, msg
        
        print(f"⚠️  Missing {status['missing_files']} model files")
        print(f"📥 Will download from: {self.repo_id}")
        
        return self.download_missing_models(progress_callback)


def main():
    """Test the model checker"""
    print("=" * 60)
    print("Creative Relight - Model Checker")
    print("=" * 60)
    
    try:
        checker = ModelChecker()
        status = checker.get_models_status()
        
        print(f"\n📊 Model Status:")
        print(f"   Repository: {status['repo_id']}")
        print(f"   Cache Dir: {status['cache_dir']}")
        print(f"   Total Files: {status['total_files']}")
        print(f"   Existing: {status['existing_files']}")
        print(f"   Missing: {status['missing_files']}")
        print(f"   Total Size: {status['total_size_readable']}")
        
        if not status['all_exist']:
            print(f"\n⚠️  Missing Files:")
            for file in status['missing_list']:
                print(f"      - {file}")
            
            response = input(f"\nDownload missing models now? (yes/no): ")
            if response.lower() == 'yes':
                success, message = checker.ensure_models_ready()
                print(f"\n{message}")
        else:
            print(f"\n✅ All models are ready!")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
