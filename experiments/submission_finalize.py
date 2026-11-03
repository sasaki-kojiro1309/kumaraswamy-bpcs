"""Compatibility entry point for experiments.execution.submission_finalize.

The implementation now lives in the execution package. Existing imports and
``python -m experiments.submission_finalize`` commands remain supported.
"""
import sys
from .execution import submission_finalize as _implementation

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
