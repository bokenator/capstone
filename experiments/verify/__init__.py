"""
Verification Functions for All Conditions
==========================================

Each verify_cN.py file implements verification functions for condition N.

Structure:
- verify/shared/     - Shared validation logic (schema, VAS, tests)
- verify/verify_cN.py - Condition-specific verification modules
"""

from . import verify_c0
from . import verify_c1
from . import verify_c2
from . import verify_c3
from . import verify_c4
from . import verify_c5
from . import verify_c6
from . import verify_c7
from . import shared

# Mapping from condition ID to verification module
CONDITION_MODULES = {
    "C0": verify_c0,
    "C1": verify_c1,
    "C2": verify_c2,
    "C3": verify_c3,
    "C4": verify_c4,
    "C5": verify_c5,
    "C6": verify_c6,
    "C7": verify_c7,
}

__all__ = [
    "verify_c0",
    "verify_c1",
    "verify_c2",
    "verify_c3",
    "verify_c4",
    "verify_c5",
    "verify_c6",
    "verify_c7",
    "shared",
    "CONDITION_MODULES",
]
