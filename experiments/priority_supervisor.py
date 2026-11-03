"""Compatibility entry point for experiments.legacy.priority_supervisor.

The implementation now lives in the legacy package. Existing imports and
``python -m experiments.priority_supervisor`` commands remain supported.
"""
import sys
from .legacy import priority_supervisor as _implementation

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
