"""
Model Factory for Creative Relight
Creates appropriate model instances based on configuration
"""

from typing import Dict, Any, Optional
import yaml
from pathlib import Path

from .base_model import BaseLightingModel, BaseGeometryModel, ModelMode


class ModelFactory:
    """
    Factory class for creating model instances.
    
    Handles creation of either local or online models based on configuration.
    """
    
    @staticmethod
    def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to config file. If None, uses default config.yaml
            
        Returns:
            Configuration dictionary
        """
        if config_path is None:
            # Default to config.yaml in project root
            config_path = Path(__file__).parent.parent.parent / "config.yaml"
        
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    @staticmethod
    def create_lighting_model(
        mode: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> BaseLightingModel:
        """
        Create a lighting decomposition model.
        
        Args:
            mode: 'local' or 'online'. If None, reads from config.
            config: Configuration dictionary. If None, loads from config.yaml.
            **kwargs: Additional parameters passed to model constructor.
            
        Returns:
            Instance of BaseLightingModel (either local or online implementation)
            
        Raises:
            ValueError: If mode is invalid
        """
        if config is None:
            config = ModelFactory.load_config()
        
        if mode is None:
            mode = config.get('models', {}).get('mode', ModelMode.LOCAL)
        
        if mode == ModelMode.LOCAL:
            # Import here to avoid circular imports
            from .lighting_model import LightingModel
            
            # Get local lighting config
            lighting_config = config.get('models', {}).get('local', {}).get('lighting', {})
            device = kwargs.get('device', 'cuda')
            
            return LightingModel(device=device)
        
        elif mode == ModelMode.ONLINE:
            # Import here to avoid circular imports
            from .lighting_model_online import OnlineLightingModel
            
            # Get online config
            online_config = config.get('models', {}).get('online', {})
            
            return OnlineLightingModel(config=online_config, **kwargs)
        
        else:
            raise ValueError(f"Invalid model mode: {mode}. Must be 'local' or 'online'.")
    
    @staticmethod
    def create_geometry_model(
        mode: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> BaseGeometryModel:
        """
        Create a geometry estimation model.
        
        Args:
            mode: 'local' or 'online'. If None, reads from config.
            config: Configuration dictionary. If None, loads from config.yaml.
            **kwargs: Additional parameters passed to model constructor.
            
        Returns:
            Instance of BaseGeometryModel (either local or online implementation)
            
        Raises:
            ValueError: If mode is invalid
        """
        if config is None:
            config = ModelFactory.load_config()
        
        if mode is None:
            mode = config.get('models', {}).get('mode', ModelMode.LOCAL)
        
        if mode == ModelMode.LOCAL:
            # Import here to avoid circular imports
            from .geometry_model import GeometryModel
            
            # Get local geometry config
            geometry_config = config.get('models', {}).get('local', {}).get('geometry', {})
            device = kwargs.get('device', 'cuda')
            
            return GeometryModel(device=device)
        
        elif mode == ModelMode.ONLINE:
            # Import here to avoid circular imports
            from .geometry_model_online import OnlineGeometryModel
            
            # Get online config
            online_config = config.get('models', {}).get('online', {})
            
            return OnlineGeometryModel(config=online_config, **kwargs)
        
        else:
            raise ValueError(f"Invalid model mode: {mode}. Must be 'local' or 'online'.")
    
    @staticmethod
    def get_model_mode(config: Optional[Dict[str, Any]] = None) -> str:
        """
        Get the current model mode from configuration.
        
        Args:
            config: Configuration dictionary. If None, loads from config.yaml.
            
        Returns:
            Current mode ('local' or 'online')
        """
        if config is None:
            config = ModelFactory.load_config()
        
        return config.get('models', {}).get('mode', ModelMode.LOCAL)
    
    @staticmethod
    def set_model_mode(mode: str, config_path: Optional[str] = None) -> None:
        """
        Update the model mode in configuration file.
        
        Args:
            mode: 'local' or 'online'
            config_path: Path to config file. If None, uses default config.yaml
            
        Raises:
            ValueError: If mode is invalid
        """
        if mode not in [ModelMode.LOCAL, ModelMode.ONLINE]:
            raise ValueError(f"Invalid model mode: {mode}. Must be 'local' or 'online'.")
        
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config.yaml"
        
        # Load current config
        config = ModelFactory.load_config(str(config_path))
        
        # Update mode
        if 'models' not in config:
            config['models'] = {}
        config['models']['mode'] = mode
        
        # Save config
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)


# Convenience functions for backward compatibility
def create_lighting_model(device='cuda', mode: Optional[str] = None) -> BaseLightingModel:
    """
    Convenience function to create a lighting model.
    
    Args:
        device: Device to use for local models ('cuda' or 'cpu')
        mode: 'local' or 'online'. If None, reads from config.
        
    Returns:
        Lighting model instance
    """
    return ModelFactory.create_lighting_model(mode=mode, device=device)


def create_geometry_model(device='cuda', mode: Optional[str] = None) -> BaseGeometryModel:
    """
    Convenience function to create a geometry model.
    
    Args:
        device: Device to use for local models ('cuda' or 'cpu')
        mode: 'local' or 'online'. If None, reads from config.
        
    Returns:
        Geometry model instance
    """
    return ModelFactory.create_geometry_model(mode=mode, device=device)
