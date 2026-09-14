"""
Base Model Interface for Creative Relight
Provides abstraction layer for local and online model implementations
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np


class BaseModel(ABC):
    """
    Abstract base class for all Creative Relight models.
    
    This interface allows seamless switching between local (on-device)
    and online (API-based) model implementations.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the model.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.is_loaded = False
    
    @abstractmethod
    def load(self) -> None:
        """
        Load the model and prepare it for inference.
        This might involve loading weights from disk (local) or
        initializing an API client (online).
        
        Raises:
            ModelLoadError: If model loading fails
        """
        pass
    
    @abstractmethod
    def process(self, image_array: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        """
        Process an image and return results.
        
        Args:
            image_array: Input image as numpy array of shape (H, W, 3) with values in [0, 1]
            **kwargs: Additional model-specific parameters
            
        Returns:
            Dictionary containing output components (e.g., 'albedo', 'shading', 'specular')
            
        Raises:
            ModelNotLoadedError: If model hasn't been loaded
            InferenceError: If inference fails
        """
        pass
    
    @abstractmethod
    def unload(self) -> None:
        """
        Unload the model and free resources.
        For local models, this clears memory. For online models, this might
        close connections or cleanup temporary files.
        """
        pass
    
    @property
    @abstractmethod
    def model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dictionary with model metadata (name, version, size, etc.)
        """
        pass
    
    def is_available(self) -> bool:
        """
        Check if the model is available for inference.
        For local models, checks if weights exist. For online models,
        checks if API is reachable.
        
        Returns:
            True if model is available, False otherwise
        """
        return self.is_loaded


class BaseLightingModel(BaseModel):
    """
    Base class for lighting decomposition models.
    Extracts albedo, shading, and specular components from images.
    """
    
    @abstractmethod
    def process(self, image_array: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        """
        Process an image to extract lighting components.
        
        Args:
            image_array: Input image as numpy array of shape (H, W, 3) with values in [0, 1]
            **kwargs: Additional parameters (e.g., resize_conf)
            
        Returns:
            Dictionary with keys: 'albedo', 'shading', 'specular'
            Each value is a numpy array of shape (H, W, 3) with values in [0, 1]
        """
        pass


class BaseGeometryModel(BaseModel):
    """
    Base class for geometry estimation models.
    Extracts depth and normal maps from images.
    """
    
    @abstractmethod
    def process(self, image_array: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        """
        Process an image to extract geometry information.
        
        Args:
            image_array: Input image as numpy array of shape (H, W, 3) with values in [0, 1]
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with keys: 'depth', 'normal'
            - 'depth': numpy array of shape (H, W) with depth values
            - 'normal': numpy array of shape (H, W, 3) with normal vectors
        """
        pass


class ModelMode:
    """Enumeration of model modes"""
    LOCAL = "local"
    ONLINE = "online"


class ModelLoadError(Exception):
    """Raised when model loading fails"""
    pass


class ModelNotLoadedError(Exception):
    """Raised when attempting to use a model that hasn't been loaded"""
    pass


class InferenceError(Exception):
    """Raised when inference fails"""
    pass


class APIError(Exception):
    """Raised when API communication fails (for online models)"""
    pass
