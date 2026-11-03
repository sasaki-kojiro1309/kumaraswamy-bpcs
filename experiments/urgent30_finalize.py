"""Compatibility entry point for experiments.legacy.urgent30_finalize.

The implementation now lives in the legacy package. Existing imports and
``python -m experiments.urgent30_finalize`` commands remain supported.
"""
import sys
from .legacy import urgent30_finalize as _implementation

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
