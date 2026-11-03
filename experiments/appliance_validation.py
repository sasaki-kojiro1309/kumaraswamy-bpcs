"""Compatibility entry point for experiments.real_data.appliance_validation.

The implementation now lives in the real_data package. Existing imports and
``python -m experiments.appliance_validation`` commands remain supported.
"""
import sys
from .real_data import appliance_validation as _implementation

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
