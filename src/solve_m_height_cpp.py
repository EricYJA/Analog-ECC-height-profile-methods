"""Compatibility import for the native module in analog-ecc-heights.

The native-enabled build is required, just as for the original extension.
New code should use analog_ecc_heights with an explicit backend argument.
"""

from analog_ecc_heights.cpp_backend.adapter import load_native as _load_native

_native = _load_native()
__all__ = [name for name in dir(_native) if not name.startswith("_")]
globals().update({name: getattr(_native, name) for name in __all__})
