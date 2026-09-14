from pathlib import Path
from typing import List
from PIL import Image
import torch

def ensure_dir(path):
    """Create directory if it doesn't exist"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_image_files(directory):
    """Get all image files from a directory (including EXR/HDR)"""
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.exr', '.hdr'}
    directory = Path(directory)
    
    if not directory.exists():
        raise ValueError(f"Directory does not exist: {directory}")
    
    image_files = []
    for ext in valid_extensions:
        image_files.extend(directory.glob(f"*{ext}"))
        image_files.extend(directory.glob(f"*{ext.upper()}"))
    
    return sorted(image_files)

def create_output_dir(output_path):
    """Create output directory if it doesn't exist"""
    try:
        Path(output_path).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        print(f"Error creating directory: {e}")
        return False

def save_image(image_data, output_path):
    """Save an image to the specified path"""
    try:
        image = Image.fromarray(image_data)
        image.save(output_path)
        return str(output_path)
    except Exception as e:
        print(f"Error saving image: {e}")
        return None

def load_image(image_path):
    """Load an image from the specified path"""
    try:
        return Image.open(image_path)
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def save_model_weights(model, path):
    """Save model weights to the specified path"""
    try:
        torch.save(model.state_dict(), path)
        return str(path)
    except Exception as e:
        print(f"Error saving model weights: {e}")
        return None

def load_model_weights(model, path):
    """Load model weights from the specified path"""
    try:
        model.load_state_dict(torch.load(path))
        return True
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return False