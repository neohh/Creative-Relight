"""
API Client Infrastructure for Creative Relight Online Models
"""

from typing import Dict, Any, Optional, Union
import time
import logging
from enum import Enum

# Configure logging
logger = logging.getLogger(__name__)


class APIProvider(Enum):
    """Supported API providers"""
    HUGGINGFACE = "huggingface"
    CUSTOM = "custom"


class RetryStrategy(Enum):
    """Retry strategies for failed requests"""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR = "linear"
    NONE = "none"


class BaseAPIClient:
    """
    Base API client with common functionality for retries, error handling, and logging.
    """
    
    def __init__(
        self,
        api_url: str,
        api_key: Optional[str] = None,
        timeout: int = 300,
        max_retries: int = 3,
        retry_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF,
        verify_ssl: bool = True
    ):
        """
        Initialize the API client.
        
        Args:
            api_url: Base URL for the API
            api_key: Authentication key (if required)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            retry_strategy: Strategy for retry delays
            verify_ssl: Whether to verify SSL certificates
        """
        self.api_url = api_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_strategy = retry_strategy
        self.verify_ssl = verify_ssl
        
        # Statistics
        self.request_count = 0
        self.error_count = 0
        self.total_retry_count = 0
        
        logger.info(f"Initialized API client for {api_url}")
    
    def get_headers(self) -> Dict[str, str]:
        """
        Get HTTP headers for requests.
        
        Returns:
            Dictionary of headers
        """
        headers = {
            "User-Agent": "CreativeRelight/1.0",
            "Content-Type": "application/json"
        }
        
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        return headers
    
    def calculate_retry_delay(self, attempt: int) -> float:
        """
        Calculate delay before retry based on strategy.
        
        Args:
            attempt: Current retry attempt number (0-indexed)
            
        Returns:
            Delay in seconds
        """
        if self.retry_strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            # Exponential backoff: 1s, 2s, 4s, 8s, ...
            return min(2 ** attempt, 60)  # Cap at 60 seconds
        
        elif self.retry_strategy == RetryStrategy.LINEAR:
            # Linear: 2s, 4s, 6s, ...
            return min((attempt + 1) * 2, 30)  # Cap at 30 seconds
        
        else:  # NONE
            return 0
    
    def should_retry(self, error: Exception, attempt: int) -> bool:
        """
        Determine if request should be retried.
        
        Args:
            error: The exception that occurred
            attempt: Current retry attempt number (0-indexed)
            
        Returns:
            True if should retry, False otherwise
        """
        if attempt >= self.max_retries:
            return False
        
        # Check if error is retryable
        error_str = str(error).lower()
        
        # Retryable errors
        retryable_errors = [
            'timeout',
            'connection',
            'temporary',
            '429',  # Rate limit
            '500',  # Server error
            '502',  # Bad gateway
            '503',  # Service unavailable
            '504',  # Gateway timeout
        ]
        
        return any(err in error_str for err in retryable_errors)
    
    def log_request(self, method: str, endpoint: str, **kwargs):
        """Log API request"""
        self.request_count += 1
        logger.debug(f"API Request #{self.request_count}: {method} {endpoint}")
    
    def log_error(self, error: Exception, method: str, endpoint: str):
        """Log API error"""
        self.error_count += 1
        logger.error(f"API Error in {method} {endpoint}: {error}")
    
    def log_retry(self, attempt: int, delay: float, error: Exception):
        """Log retry attempt"""
        self.total_retry_count += 1
        logger.warning(
            f"Retry attempt {attempt + 1}/{self.max_retries} "
            f"after {delay:.1f}s delay. Error: {error}"
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get API usage statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            'request_count': self.request_count,
            'error_count': self.error_count,
            'total_retry_count': self.total_retry_count,
            'error_rate': self.error_count / max(self.request_count, 1)
        }
    
    def reset_statistics(self):
        """Reset usage statistics"""
        self.request_count = 0
        self.error_count = 0
        self.total_retry_count = 0


class APIException(Exception):
    """Base exception for API errors"""
    pass


class APITimeoutError(APIException):
    """Raised when API request times out"""
    pass


class APIAuthenticationError(APIException):
    """Raised when authentication fails"""
    pass


class APIRateLimitError(APIException):
    """Raised when rate limit is exceeded"""
    pass


class APIServerError(APIException):
    """Raised when server returns 5xx error"""
    pass


class APIInvalidResponseError(APIException):
    """Raised when response format is invalid"""
    pass
