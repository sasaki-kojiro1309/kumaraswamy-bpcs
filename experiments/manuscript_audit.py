"""Compatibility entry point for experiments.validation.manuscript_audit.

The implementation now lives in the validation package. Existing imports and
``python -m experiments.manuscript_audit`` commands remain supported.
"""
import sys
from .validation import manuscript_audit as _implementation

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
