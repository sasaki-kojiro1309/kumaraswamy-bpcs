"""Compatibility entry point for experiments.validation.within_block_validation.

The implementation now lives in the validation package. Existing imports and
``python -m experiments.within_block_validation`` commands remain supported.
"""
import sys
from .validation import within_block_validation as _implementation

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
