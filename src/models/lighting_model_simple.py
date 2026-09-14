import torch
import numpy as np
import cv2
from PIL import Image

class LightingModel:
    """Simplified lighting decomposition model that doesn't require external dependencies"""
    
    # Class variable to store the model (shared across instances)
    _loaded = False
    
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self._load_model()
    
    def _load_model(self):
        """Initialize the simplified lighting model"""
        if not LightingModel._loaded:
            print("Loading simplified lighting decomposition model...")
            LightingModel._loaded = True
            print("✅ Lighting model loaded successfully!")
    
    def process(self, image_array):
        """
        Process image to extract albedo and specular using simplified computer vision methods
        
        Args:
            image_array: Input image as numpy array (H, W, 3) in range [0, 1]
            
        Returns:
            dict with 'albedo' and 'specular' keys
        """
        try:
            # Convert to uint8 for OpenCV processing
            img_uint8 = (image_array * 255).astype(np.uint8)
            
            # Extract albedo (base color without highlights)
            albedo = self._extract_albedo(img_uint8)
            
            # Extract specular highlights
            specular = self._extract_specular(img_uint8)
            
            return {
                'albedo': albedo.astype(np.float32) / 255.0,
                'specular': specular.astype(np.float32) / 255.0
            }
            
        except Exception as e:
            print(f"❌ Error in lighting processing: {e}")
            # Return original image as albedo, empty specular
            h, w = image_array.shape[:2]
            return {
                'albedo': image_array,
                'specular': np.zeros((h, w), dtype=np.float32)
            }
    
    def _extract_albedo(self, image):
        """Extract base albedo by removing highlights and normalizing lighting"""
        # Convert to LAB color space for better separation
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply bilateral filter to smooth lighting while preserving edges
        l_filtered = cv2.bilateralFilter(l, 9, 75, 75)
        
        # Normalize the L channel to reduce lighting variations
        l_normalized = cv2.equalizeHist(l_filtered)
        
        # Merge back and convert to RGB
        lab_result = cv2.merge([l_normalized, a, b])
        albedo = cv2.cvtColor(lab_result, cv2.COLOR_LAB2RGB)
        
        return albedo
    
    def _extract_specular(self, image):
        """Extract specular highlights"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Find very bright regions (potential specular highlights)
        _, bright_mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
        
        # Apply Gaussian blur to soften the highlights
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Create specular map by combining bright regions with blurred image
        specular = np.where(bright_mask > 0, blurred, 0).astype(np.uint8)
        
        # Apply morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        specular = cv2.morphologyEx(specular, cv2.MORPH_CLOSE, kernel)
        
        # Normalize to full range
        if specular.max() > 0:
            specular = (specular.astype(np.float32) / specular.max() * 255).astype(np.uint8)
        
        return specular
