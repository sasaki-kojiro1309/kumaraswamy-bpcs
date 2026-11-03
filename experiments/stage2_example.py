"""Compatibility entry point for experiments.examples.stage2_example.

The implementation now lives in the examples package. Existing imports and
``python -m experiments.stage2_example`` commands remain supported.
"""
import sys
from .examples import stage2_example as _implementation

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
