"""
Common Types for Verification
=============================
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class VerificationResult:
    """Standardized result from verification functions."""
    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.passed

    def add_error(self, msg: str) -> None:
        """Add an error and mark result as failed."""
        self.errors.append(msg)
        self.passed = False

    def add_warning(self, msg: str) -> None:
        """Add a warning (does not affect pass/fail status)."""
        self.warnings.append(msg)

    def merge(self, other: "VerificationResult") -> "VerificationResult":
        """Merge another result into this one."""
        self.passed = self.passed and other.passed
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.details.update(other.details)
        return self

    def summary(self) -> str:
        """Return a human-readable summary."""
        status = "PASSED" if self.passed else "FAILED"
        lines = [f"Status: {status}"]
        if self.errors:
            lines.append(f"Errors ({len(self.errors)}):")
            for err in self.errors:
                lines.append(f"  - {err}")
        if self.warnings:
            lines.append(f"Warnings ({len(self.warnings)}):")
            for warn in self.warnings:
                lines.append(f"  - {warn}")
        return "\n".join(lines)
