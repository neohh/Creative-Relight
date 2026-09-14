import torch
import numpy as np
import cv2
from transformers import pipeline
from PIL import Image

class GeometryModel:
    """Simplified geometry model for depth and normal estimation"""
    
    # Class variable to store the model (shared across instances)
    _model = None
    _loaded = False
    
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self._load_model()
    
    def _load_model(self):
        """Load depth estimation model"""
        if not GeometryModel._loaded:
            print("Loading geometry model...")
            try:
                # Use Intel's MiDaS for depth estimation - more portable
                GeometryModel._model = pipeline(
                    "depth-estimation",
                    model="Intel/dpt-large",
                    device=0 if self.device.type == 'cuda' else -1
                )
                GeometryModel._loaded = True
                print("✅ Geometry model loaded successfully!")
            except Exception as e:
                print(f"⚠️ Could not load depth model, using simplified processing: {e}")
                GeometryModel._model = "simplified"
                GeometryModel._loaded = True
    
    def process(self, image_array):
        """
        Process image to extract depth and normal maps
        
        Args:
            image_array: Input image as numpy array (H, W, 3) in range [0, 1]
            
        Returns:
            dict with 'depth' and 'normal' keys
        """
        try:
            if GeometryModel._model == "simplified":
                return self._simplified_processing(image_array)
            else:
                return self._advanced_processing(image_array)
        except Exception as e:
            print(f"⚠️ Geometry processing failed, using simplified method: {e}")
            return self._simplified_processing(image_array)
    
    def _advanced_processing(self, image_array):
        """Use AI model for depth estimation"""
        try:
            # Convert to PIL Image for the pipeline
            img_uint8 = (image_array * 255).astype(np.uint8)
            pil_image = Image.fromarray(img_uint8)
            
            # Get depth from the model
            depth_result = GeometryModel._model(pil_image)
            depth_map = np.array(depth_result["depth"])
            
            # Normalize depth to 0-255 range
            depth_normalized = ((depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255).astype(np.uint8)
            
            # Generate normals from depth
            normal_map = self._depth_to_normals(depth_normalized)
            
            return {
                'depth': depth_normalized,
                'normal': normal_map
            }
        except Exception as e:
            print(f"Advanced processing failed: {e}")
            return self._simplified_processing(image_array)
    
    def _simplified_processing(self, image_array):
        """Simplified depth and normal estimation using computer vision"""
        # Convert to uint8 for OpenCV
        img_uint8 = (image_array * 255).astype(np.uint8)
        
        # Simple depth estimation based on luminance and edges
        depth = self._estimate_depth_simple(img_uint8)
        
        # Generate normals from depth
        normal = self._depth_to_normals(depth)
        
        return {
            'depth': depth,
            'normal': normal
        }
    
    def _estimate_depth_simple(self, image):
        """Simple depth estimation based on luminance and gradients"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Calculate gradients
        grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        
        # Combine gradients to estimate depth (edges are closer)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Invert so edges (high gradients) are closer (darker)
        depth_estimate = 255 - gradient_magnitude
        
        # Normalize
        depth_estimate = (depth_estimate - depth_estimate.min()) / (depth_estimate.max() - depth_estimate.min()) * 255
        
        return depth_estimate.astype(np.uint8)
    
    def _depth_to_normals(self, depth_map):
        """Convert depth map to normal map"""
        # Calculate gradients of the depth map
        grad_x = cv2.Sobel(depth_map.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth_map.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        
        # Calculate normal vectors
        # Normal = (-dz/dx, -dz/dy, 1) normalized
        normal_x = -grad_x
        normal_y = -grad_y
        normal_z = np.ones_like(grad_x) * 255  # Pointing towards camera
        
        # Normalize the vectors
        length = np.sqrt(normal_x**2 + normal_y**2 + normal_z**2)
        length[length == 0] = 1  # Avoid division by zero
        
        normal_x = normal_x / length
        normal_y = normal_y / length  
        normal_z = normal_z / length
        
        # Convert to 0-255 range (from -1,1 to 0,255)
        normal_x = ((normal_x + 1) * 127.5).astype(np.uint8)
        normal_y = ((normal_y + 1) * 127.5).astype(np.uint8)
        normal_z = ((normal_z + 1) * 127.5).astype(np.uint8)
        
        # Combine into RGB normal map
        normal_map = np.stack([normal_x, normal_y, normal_z], axis=2)
        
        return normal_map
