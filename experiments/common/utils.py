"""
Common Utilities for Verification
=================================
"""

import inspect
import importlib.util
from pathlib import Path
from typing import Any


def load_module_from_path(file_path: str | Path) -> Any:
    """
    Load a Python module from file path.

    Args:
        file_path: Path to the .py file to load

    Returns:
        The loaded module object

    Raises:
        ImportError: If the module cannot be loaded
    """
    file_path = Path(file_path)
    spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_source_code(module_or_path: Any) -> str:
    """
    Extract source code from module or file path.

    Args:
        module_or_path: Either a loaded module or a path to a .py file

    Returns:
        The source code as a string
    """
    if isinstance(module_or_path, (str, Path)):
        return Path(module_or_path).read_text()
    else:
        return inspect.getsource(module_or_path)
