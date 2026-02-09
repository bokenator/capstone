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

# Documentation (RAG) validation requirements
VALIDATION_VAS = """
## Validation Requirements

- Search the documentation index to verify each third-party API before using it
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
- Search the documentation index to verify each third-party API before using it
"""

VALIDATION_SCHEMA_TESTS = """
## Validation Requirements

- Code must only access columns declared in DATA_SCHEMA
- Code must only use parameters declared in PARAM_SCHEMA
- Code must pass all invariant and strategy-specific tests
"""

VALIDATION_VAS_TESTS = """
## Validation Requirements

- Search the documentation index to verify each third-party API before using it
- Code must pass all invariant and strategy-specific tests
"""

VALIDATION_ALL = """
## Validation Requirements

- Code must only access columns declared in DATA_SCHEMA
- Code must only use parameters declared in PARAM_SCHEMA
- Search the documentation index to verify each third-party API before using it
- Code must pass all invariant and strategy-specific tests
"""
