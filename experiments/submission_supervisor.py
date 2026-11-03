"""Compatibility entry point for experiments.execution.submission_supervisor.

The implementation now lives in the execution package. Existing imports and
``python -m experiments.submission_supervisor`` commands remain supported.
"""
import sys
from .execution import submission_supervisor as _implementation

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
