import os

# OpenCV (>=4.x pip builds) compiles the OpenEXR codec in but disables it by
# default for security reasons; it must be enabled via env var BEFORE cv2 is
# imported (the codec's enable/disable decision is cached at import time).
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')

import numpy as np
import cv2
from PIL import Image

# High dynamic range image formats (loaded via OpenCV's OpenEXR codec)
HDR_EXTENSIONS = {'.exr', '.hdr', '.pic'}

# All image extensions accepted for processing
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp', '.gif'} | HDR_EXTENSIONS


def is_hdr_path(image_path):
    """Return True if the path points to an HDR image (.exr/.hdr/.pic)"""
    return str(image_path).lower().endswith(tuple(HDR_EXTENSIONS))


def _linear_to_srgb(x):
    """Convert linear [0..1] values to sRGB gamma (piecewise, vectorized)."""
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(np.maximum(x, 1e-10), 1.0 / 2.4) - 0.055)


def _fill_transparent_pixels(rgb, hole_mask):
    """Fill transparent (hole_mask==True) pixels by diffusing surrounding
    opaque colors inward, so models don't see black/garbage at sprite edges.
    Only currently-known (opaque) pixels are weighted in the blur, so filled
    values stay within the [0..1] range of the source data."""
    img = rgb.copy()
    hole_mask = hole_mask.copy()
    max_iters = max(8, int(np.ceil(np.log2(max(img.shape[0], img.shape[1]) + 2))) * 4)
    for _ in range(max_iters):
        if not hole_mask.any():
            break
        known = (~hole_mask).astype(np.float32)
        # blur ONLY known pixels: zero-out holes before averaging, then
        # normalize by the known-pixel fraction of each window
        blurred = cv2.blur(img * known[:, :, None], (5, 5))
        bw = cv2.blur(known, (5, 5))
        fill_now = hole_mask & (bw > 1e-4)
        if fill_now.any():
            img[fill_now] = blurred[fill_now] / bw[fill_now][:, None]
            hole_mask[fill_now] = False
        else:
            break
    if hole_mask.any():
        # Nothing opaque to diffuse from (fully transparent image) - neutral gray
        img[hole_mask] = 0.5
    return img


def load_hdr_image(image_path):
    """
    Load an EXR/HDR image and prepare it for the models.

    Returns:
        image_array: float32 RGB in [0..1] (sRGB-tonemapped from linear data,
                     transparent regions filled from surrounding opaque pixels)
        alpha:       uint8 alpha mask (H, W) in [0..255], or None if opaque
    """
    # IMREAD_UNCHANGED|IMREAD_ANYCOLOR|IMREAD_ANYDEPTH keeps float precision + alpha
    data = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if data is None:
        raise ValueError(f"Could not read HDR image: {image_path}")
    data = data.astype(np.float32)

    alpha = None
    if data.ndim == 2:
        rgb = np.dstack([data, data, data])
    elif data.shape[2] == 4:
        rgb = data[:, :, :3]
        alpha = data[:, :, 3]
        if alpha.max() <= 1.0 and alpha.min() < 1.0:
            alpha = alpha * 255.0
    elif data.shape[2] == 3:
        rgb = data
    else:
        rgb = data[:, :, :3]

    # BGR -> RGB
    rgb = rgb[:, :, ::-1].copy()

    if alpha is not None:
        alpha8 = np.clip(alpha, 0, 255).astype(np.uint8)
        hole_mask = alpha8 < 8  # effectively fully transparent
        if hole_mask.any() and not hole_mask.all():
            # tonemap first, then fill holes with neighboring opaque colors
            rgb01 = _linear_to_srgb(rgb)
            rgb01 = _fill_transparent_pixels(rgb01, hole_mask)
        else:
            rgb01 = _linear_to_srgb(rgb)
    else:
        alpha8 = None
        rgb01 = _linear_to_srgb(rgb)

    return np.ascontiguousarray(rgb01, dtype=np.float32), alpha8


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