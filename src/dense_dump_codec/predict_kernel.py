"""Explicit opt-in single-pass prediction with separate binary32 operations."""

import ctypes
import hashlib
from pathlib import Path

import numpy as np


class FusedPredictor:
    """Load an explicitly built local kernel; never compile or discover binaries implicitly."""

    def __init__(self, library):
        self.path = Path(library).resolve(strict=True)
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.handle = ctypes.CDLL(str(self.path))
        self.handle.ddc_predict_abi.argtypes = []
        self.handle.ddc_predict_abi.restype = ctypes.c_int
        if self.handle.ddc_predict_abi() != 1:
            raise ValueError("Unsupported DDC prediction kernel ABI")
        self.function = self.handle.ddc_predict_add
        self.function.argtypes = [ctypes.c_size_t, ctypes.c_void_p, ctypes.c_void_p,
                                  ctypes.c_void_p, ctypes.c_float, ctypes.c_float,
                                  ctypes.c_void_p]
        self.function.restype = None

    def __call__(self, start, end, residual, tau, out):
        arrays = start, end, residual, out
        if any(array.dtype != np.dtype("float32") or not array.flags.c_contiguous
               or array.shape != out.shape for array in arrays):
            raise ValueError("Prediction kernel requires matched contiguous float32 arrays")
        if not out.flags.writeable or any(np.shares_memory(out, array) for array in arrays[:-1]):
            raise ValueError("Prediction output must be writable and independently owned")
        if not np.isfinite(tau):
            raise ValueError("Prediction weight must be finite")
        self.function(out.size, start.ctypes.data, end.ctypes.data, residual.ctypes.data,
                      float(np.float32(1.0 - tau)), float(np.float32(tau)), out.ctypes.data)
        return out
