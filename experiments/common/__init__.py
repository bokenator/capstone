"""
Common Types and Utilities
==========================

Shared types and functions for verification across all conditions.
"""

from .types import VerificationResult
from .utils import load_module_from_path, get_source_code

__all__ = [
    "VerificationResult",
    "load_module_from_path",
    "get_source_code",
]
