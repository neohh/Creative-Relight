"""Models for Creative Relight"""

import os
# Enable OpenCV's OpenEXR codec before the first cv2 import of the process
# (pip builds ship it compiled-in but disabled by default)
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')

from .lighting_model import LightingModel
from .geometry_model import GeometryModel

__all__ = ['LightingModel', 'GeometryModel']