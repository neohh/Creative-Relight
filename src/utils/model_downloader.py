"""
Model Downloader for Creative Relight
Downloads AI models from Hugging Face Hub with progress tracking
"""

import os
import hashlib
from pathlib import Path
from typing import Optional, Callable
import requests
from tqdm import tqdm


class ModelDownloader:
    """Download and manage AI models from online storage"""
    
    def __init__(self, cache_dir: str = "models"):
        """
        Initialize model downloader
        
        Args:
            cache_dir: Local directory to cache downloaded models
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def download_from_url(
        self, 
        url: str, 
        output_path: Path,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        chunk_size: int = 8192
    ) -> bool:
        """
        Download a file from URL with progress tracking
        
        Args:
            url: URL to download from
            output_path: Where to save the file
            progress_callback: Optional callback(downloaded_bytes, total_bytes)
            chunk_size: Download chunk size in bytes
            
        Returns:
            True if download successful, False otherwise
        """
        try:
            # Create output directory
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Check if file already exists
            if output_path.exists():
                print(f"✓ File already exists: {output_path.name}")
                return True
            
            # Stream download
            print(f"⬇ Downloading: {output_path.name}")
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            # Get file size
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            # Download with progress
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if progress_callback:
                            progress_callback(downloaded, total_size)
            
            print(f"✓ Downloaded: {output_path.name} ({total_size / 1024 / 1024:.2f} MB)")
            return True
            
        except Exception as e:
            print(f"✗ Download failed: {e}")
            # Clean up partial download
            if output_path.exists():
                output_path.unlink()
            return False
    
    def download_from_huggingface(
        self,
        repo_id: str,
        filename: str,
        subfolder: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Optional[Path]:
        """
        Download a file from Hugging Face Hub with real-time progress
        
        Args:
            repo_id: Repository ID (e.g., "username/model-name")
            filename: File to download
            subfolder: Optional subfolder in the repo
            progress_callback: Optional callback(downloaded_bytes, total_bytes)
            
        Returns:
            Path to downloaded file, or None if failed
        """
        try:
            from huggingface_hub import hf_hub_url
            import requests
            
            # Determine cache path
            if subfolder:
                cache_path = self.cache_dir / subfolder
            else:
                cache_path = self.cache_dir
            
            cache_path.mkdir(parents=True, exist_ok=True)
            output_path = cache_path / filename
            
            # Check if already downloaded
            if output_path.exists():
                print(f"✓ Model already cached: {filename}")
                if progress_callback:
                    # Report 100% if already cached
                    file_size = output_path.stat().st_size
                    progress_callback(file_size, file_size)
                return output_path
            
            print(f"⬇ Downloading from Hugging Face: {filename}")
            
            # Get download URL
            file_path = filename if not subfolder else f"{subfolder}/{filename}"
            url = hf_hub_url(repo_id=repo_id, filename=file_path)
            
            # Download with progress tracking
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            
            # Get total file size
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            # Create temporary file
            temp_path = output_path.with_suffix('.tmp')
            
            # Download with real-time progress
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Call progress callback in real-time
                        if progress_callback and total_size > 0:
                            progress_callback(downloaded, total_size)
            
            # Rename temp file to final name
            temp_path.rename(output_path)
            
            print(f"✓ Downloaded: {filename}")
            return output_path
            
        except ImportError:
            print("⚠ huggingface_hub not installed. Install with: pip install huggingface_hub")
            return None
        except Exception as e:
            print(f"✗ Download failed: {e}")
            # Clean up temp file
            temp_path = output_path.with_suffix('.tmp')
            if temp_path.exists():
                temp_path.unlink()
            return None
    
    def verify_checksum(self, file_path: Path, expected_hash: str, algorithm: str = "sha256") -> bool:
        """
        Verify file integrity using checksum
        
        Args:
            file_path: Path to file to verify
            expected_hash: Expected hash value
            algorithm: Hash algorithm (sha256, md5, etc.)
            
        Returns:
            True if checksum matches, False otherwise
        """
        if not file_path.exists():
            return False
        
        try:
            hash_func = hashlib.new(algorithm)
            
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    hash_func.update(chunk)
            
            calculated_hash = hash_func.hexdigest()
            matches = calculated_hash == expected_hash
            
            if matches:
                print(f"✓ Checksum verified: {file_path.name}")
            else:
                print(f"✗ Checksum mismatch: {file_path.name}")
                print(f"  Expected: {expected_hash}")
                print(f"  Got: {calculated_hash}")
            
            return matches
            
        except Exception as e:
            print(f"✗ Checksum verification failed: {e}")
            return False
    
    def get_file_size(self, file_path: Path) -> int:
        """Get file size in bytes"""
        if file_path.exists():
            return file_path.stat().st_size
        return 0
    
    def format_bytes(self, bytes_size: int) -> str:
        """Format bytes to human-readable string"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_size < 1024.0:
                return f"{bytes_size:.2f} {unit}"
            bytes_size /= 1024.0
        return f"{bytes_size:.2f} TB"


# Example usage
if __name__ == "__main__":
    # Test download
    downloader = ModelDownloader(cache_dir="models")
    
    def progress(downloaded, total):
        if total > 0:
            percent = (downloaded / total) * 100
            print(f"\rProgress: {percent:.1f}% ({downloaded}/{total} bytes)", end='')
    
    # Test with a small file
    print("Testing model downloader...")
    # downloader.download_from_huggingface(
    #     repo_id="username/creative-relight-models",
    #     filename="test.txt",
    #     progress_callback=progress
    # )
