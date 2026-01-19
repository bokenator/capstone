"""
Experimental Condition Prompts
==============================

8 conditions for 2³ factorial design testing Schema (S), Documentation (D), and TDD (T) treatments.

Usage:
    from prompts import c0_control, c1_schema, ...

    # Get prompt for a specific condition and strategy
    prompt = c0_control.PROMPTS["simple"]["prompt"]

    # Get condition metadata
    condition_info = c0_control.CONDITION
"""

from . import c0_control
from . import c1_schema
from . import c2_docs
from . import c3_tdd
from . import c4_schema_docs
from . import c5_schema_tdd
from . import c6_docs_tdd
from . import c7_all

# Quick access to all conditions
CONDITIONS = {
    "C0": c0_control,
    "C1": c1_schema,
    "C2": c2_docs,
    "C3": c3_tdd,
    "C4": c4_schema_docs,
    "C5": c5_schema_tdd,
    "C6": c6_docs_tdd,
    "C7": c7_all,
}

__all__ = [
    "c0_control",
    "c1_schema",
    "c2_docs",
    "c3_tdd",
    "c4_schema_docs",
    "c5_schema_tdd",
    "c6_docs_tdd",
    "c7_all",
    "CONDITIONS",
]
