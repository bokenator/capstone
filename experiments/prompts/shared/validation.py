"""
Validation Requirements
=======================

Validation requirement sections for different condition combinations.
"""

# Schema validation requirements
VALIDATION_SCHEMA = """
## Validation Requirements

- Code must only access columns declared in DATA_SCHEMA
- Code must only use parameters declared in PARAM_SCHEMA
"""

# VAS validation requirements
VALIDATION_VAS = """
## Validation Requirements

- Code must only use APIs listed in VAS
"""

# Test validation requirements
VALIDATION_TESTS = """
## Validation Requirements

- Code must pass all invariant and strategy-specific tests
"""

# Combined validation requirements
VALIDATION_SCHEMA_VAS = """
## Validation Requirements

- Code must only access columns declared in DATA_SCHEMA
- Code must only use parameters declared in PARAM_SCHEMA
- Code must only use APIs listed in VAS
"""

VALIDATION_SCHEMA_TESTS = """
## Validation Requirements

- Code must only access columns declared in DATA_SCHEMA
- Code must only use parameters declared in PARAM_SCHEMA
- Code must pass all invariant and strategy-specific tests
"""

VALIDATION_VAS_TESTS = """
## Validation Requirements

- Code must only use APIs listed in VAS
- Code must pass all invariant and strategy-specific tests
"""

VALIDATION_ALL = """
## Validation Requirements

- Code must only access columns declared in DATA_SCHEMA
- Code must only use parameters declared in PARAM_SCHEMA
- Code must only use APIs listed in VAS
- Code must pass all invariant and strategy-specific tests
"""
