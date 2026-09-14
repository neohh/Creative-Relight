import numpy as np
import cv2
from PIL import Image

def resize_image(image, target_size):
    """Resize the input image to the target size."""
    return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

def convert_color_space(image, color_space):
    """Convert the input image to the specified color space."""
    if color_space == 'RGB':
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif color_space == 'GRAY':
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Unsupported color space: {color_space}")

def normalize_image(image):
    """Normalize the pixel values of the input image to the range [0, 1]."""
    return image.astype(np.float32) / 255.0

def denormalize_image(image):
    """Denormalize the pixel values of the input image to the range [0, 255]."""
    return (image * 255).astype(np.uint8)

def resize_to_original(image_array, original_size):
    """
    Resize image array to original dimensions
    
    Args:
        image_array: numpy array of image
        original_size: tuple of (width, height)
    
    Returns:
        Resized numpy array
    """
    if image_array is None:
        return None
    
    # Handle different input shapes
    if len(image_array.shape) == 3:
        # RGB image
        return cv2.resize(image_array, original_size, interpolation=cv2.INTER_LINEAR)
    elif len(image_array.shape) == 2:
        # Grayscale image
        return cv2.resize(image_array, original_size, interpolation=cv2.INTER_LINEAR)
    else:
        return image_array

def ensure_uint8(image_array):
    """
    Convert image array to uint8 format
    
    Args:
        image_array: numpy array
        
    Returns:
        uint8 numpy array
    """
    if image_array is None:
        return None
    
    if image_array.dtype == np.uint8:
        return image_array
    
    # If float, assume range [0, 1]
    if image_array.dtype in [np.float32, np.float64]:
        if image_array.max() <= 1.0:
            return (image_array * 255).astype(np.uint8)
        else:
            return image_array.clip(0, 255).astype(np.uint8)
    
    # For other types, clip to valid range
    return image_array.clip(0, 255).astype(np.uint8)