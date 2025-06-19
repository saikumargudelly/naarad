class NaaradException(Exception):
    """Base exception for Naarad agent errors."""
    pass

class RateLimitExceeded(NaaradException):
    """Raised when rate limit is exceeded."""
    pass

class ServiceUnavailable(NaaradException):
    """Raised when an external service is unavailable."""
    pass

class ValidationError(NaaradException):
    """Raised when input validation fails."""
    pass
