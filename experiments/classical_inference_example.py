"""Compatibility entry point for experiments.examples.classical_inference_example.

The implementation now lives in the examples package. Existing imports and
``python -m experiments.classical_inference_example`` commands remain supported.
"""
import sys
from .examples import classical_inference_example as _implementation

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
