"""
Inference API Client for Creative Relight
Handles image inference requests to online model endpoints
"""

from typing import Dict, Any, Optional, Union
import io
import base64
import time
import logging
import numpy as np
from PIL import Image

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logging.warning("requests library not available. Install with: pip install requests")

from .base_client import (
    BaseAPIClient,
    APIException,
    APITimeoutError,
    APIAuthenticationError,
    APIRateLimitError,
    APIServerError,
    APIInvalidResponseError,
    APIProvider
)

logger = logging.getLogger(__name__)


class InferenceClient(BaseAPIClient):
    """
    Client for model inference API requests.
    Supports HuggingFace Inference API and custom endpoints.
    """
    
    def __init__(
        self,
        api_url: str,
        api_key: Optional[str] = None,
        provider: APIProvider = APIProvider.HUGGINGFACE,
        **kwargs
    ):
        """
        Initialize inference client.
        
        Args:
            api_url: Base URL for inference API
            api_key: Authentication key
            provider: API provider (HuggingFace or custom)
            **kwargs: Additional arguments for BaseAPIClient
        """
        super().__init__(api_url, api_key, **kwargs)
        self.provider = provider
        
        if not REQUESTS_AVAILABLE:
            raise ImportError(
                "requests library is required for online inference. "
                "Install with: pip install requests"
            )
    
    def _encode_image(self, image_array: np.ndarray, format: str = 'PNG') -> str:
        """
        Encode numpy array as base64 string.
        
        Args:
            image_array: Image as numpy array (H, W, 3) with values in [0, 1]
            format: Image format (PNG, JPEG)
            
        Returns:
            Base64 encoded image string
        """
        # Convert to uint8
        if image_array.dtype != np.uint8:
            image_array = (image_array * 255).astype(np.uint8)
        
        # Create PIL Image
        image = Image.fromarray(image_array)
        
        # Encode to bytes
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        buffer.seek(0)
        
        # Encode to base64
        encoded = base64.b64encode(buffer.read()).decode('utf-8')
        return encoded
    
    def _decode_image(self, encoded_str: str) -> np.ndarray:
        """
        Decode base64 string to numpy array.
        
        Args:
            encoded_str: Base64 encoded image string
            
        Returns:
            Image as numpy array (H, W, 3) with values in [0, 1]
        """
        # Decode base64
        decoded = base64.b64decode(encoded_str)
        
        # Load as PIL Image
        buffer = io.BytesIO(decoded)
        image = Image.open(buffer)
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Normalize to [0, 1]
        if image_array.dtype == np.uint8:
            image_array = image_array.astype(np.float32) / 255.0
        
        return image_array
    
    def _prepare_request_data(
        self,
        image_array: np.ndarray,
        model_name: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Prepare request data based on provider.
        
        Args:
            image_array: Input image
            model_name: Name of the model to use
            parameters: Additional model parameters
            
        Returns:
            Request data dictionary
        """
        if self.provider == APIProvider.HUGGINGFACE:
            # HuggingFace format
            return {
                "inputs": self._encode_image(image_array),
                "parameters": parameters or {}
            }
        else:
            # Custom format
            return {
                "model": model_name,
                "image": self._encode_image(image_array),
                "parameters": parameters or {}
            }
    
    def _parse_response(
        self,
        response_data: Union[Dict, list],
        expected_outputs: list
    ) -> Dict[str, np.ndarray]:
        """
        Parse API response and extract output images.
        
        Args:
            response_data: Raw response data
            expected_outputs: List of expected output keys (e.g., ['albedo', 'shading'])
            
        Returns:
            Dictionary mapping output names to numpy arrays
        """
        results = {}
        
        if self.provider == APIProvider.HUGGINGFACE:
            # HuggingFace returns a list of outputs or a dict
            if isinstance(response_data, list):
                # Assume order matches expected_outputs
                for i, output_name in enumerate(expected_outputs):
                    if i < len(response_data):
                        results[output_name] = self._decode_image(response_data[i])
            elif isinstance(response_data, dict):
                # Extract from dict
                for output_name in expected_outputs:
                    if output_name in response_data:
                        results[output_name] = self._decode_image(response_data[output_name])
        else:
            # Custom format - expect dict with output names
            for output_name in expected_outputs:
                if output_name in response_data:
                    results[output_name] = self._decode_image(response_data[output_name])
        
        return results
    
    def infer(
        self,
        image_array: np.ndarray,
        model_name: str,
        expected_outputs: list,
        parameters: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, np.ndarray]:
        """
        Perform inference on an image.
        
        Args:
            image_array: Input image as numpy array (H, W, 3) with values in [0, 1]
            model_name: Name/ID of the model to use
            expected_outputs: List of expected output keys
            parameters: Additional model parameters
            progress_callback: Optional callback(status: str, progress: float)
            
        Returns:
            Dictionary mapping output names to numpy arrays
            
        Raises:
            APIException: If inference fails
        """
        endpoint = f"{self.api_url}/{model_name}" if self.provider == APIProvider.HUGGINGFACE else self.api_url
        
        # Prepare request
        request_data = self._prepare_request_data(image_array, model_name, parameters)
        headers = self.get_headers()
        
        if progress_callback:
            progress_callback("Uploading image...", 0.1)
        
        # Make request with retries
        for attempt in range(self.max_retries + 1):
            try:
                self.log_request("POST", endpoint)
                
                start_time = time.time()
                response = requests.post(
                    endpoint,
                    json=request_data,
                    headers=headers,
                    timeout=self.timeout,
                    verify=self.verify_ssl
                )
                elapsed = time.time() - start_time
                
                logger.debug(f"Request completed in {elapsed:.2f}s")
                
                # Check status code
                if response.status_code == 200:
                    if progress_callback:
                        progress_callback("Processing complete", 0.9)
                    
                    # Parse response
                    try:
                        response_data = response.json()
                        results = self._parse_response(response_data, expected_outputs)
                        
                        if not results:
                            raise APIInvalidResponseError(
                                f"No valid outputs found in response. Expected: {expected_outputs}"
                            )
                        
                        if progress_callback:
                            progress_callback("Done", 1.0)
                        
                        return results
                    
                    except (ValueError, KeyError) as e:
                        raise APIInvalidResponseError(f"Invalid response format: {e}")
                
                elif response.status_code == 401:
                    raise APIAuthenticationError("Authentication failed. Check your API key.")
                
                elif response.status_code == 429:
                    raise APIRateLimitError("Rate limit exceeded. Please try again later.")
                
                elif response.status_code >= 500:
                    raise APIServerError(f"Server error: {response.status_code} - {response.text}")
                
                else:
                    raise APIException(f"Request failed: {response.status_code} - {response.text}")
            
            except requests.exceptions.Timeout as e:
                error = APITimeoutError(f"Request timed out after {self.timeout}s")
                if not self.should_retry(error, attempt):
                    self.log_error(error, "POST", endpoint)
                    raise error
            
            except requests.exceptions.RequestException as e:
                error = APIException(f"Request failed: {e}")
                if not self.should_retry(error, attempt):
                    self.log_error(error, "POST", endpoint)
                    raise error
            
            # Retry logic
            if attempt < self.max_retries:
                delay = self.calculate_retry_delay(attempt)
                self.log_retry(attempt, delay, error)
                
                if progress_callback:
                    progress_callback(f"Retrying in {delay:.0f}s...", 0.3)
                
                time.sleep(delay)
        
        # Should never reach here, but just in case
        raise APIException("All retry attempts failed")
    
    def health_check(self) -> bool:
        """
        Check if API is reachable and healthy.
        
        Returns:
            True if API is healthy, False otherwise
        """
        try:
            # Try a simple GET request to check connectivity
            response = requests.get(
                self.api_url,
                headers=self.get_headers(),
                timeout=10,
                verify=self.verify_ssl
            )
            return response.status_code < 500
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
