"""core algorithms and helpers."""

from __future__ import annotations

import numpy as np
from typing import Union

Array = np.ndarray


def dtype2real(dtype: np.dtype) -> np.dtype:
  return np.array(0, dtype=dtype).real.dtype


def dtype2complex(dtype: np.dtype) -> np.dtype:
  return np.promote_types(dtype, np.complex64)


def realsign(x: Array) -> Array:
  """Scalar sign applied to the real part, as in Diniz Eq. (4.5)."""
  xr = np.real(x)
  out = np.zeros_like(xr, dtype=np.float32)
  out[xr > 0] = 1.0
  out[xr < 0] = -1.0
  return out


def csignvec(x: Array, eps: float = 1e-12) -> Array:
  """Elementwise complex sign for a vector."""
  x = np.asarray(x, dtype=np.complex64)
  mag = np.abs(x)
  out = np.zeros_like(x)
  nz = mag > eps
  out[nz] = x[nz] / mag[nz]
  return out


def csignscalar(e: Union[complex, Array], eps: float = 1e-12) -> np.complex64:
  """Complex sign scalar estimation."""
  e = np.asarray(e, dtype=np.complex64)
  mag = np.abs(e)
  if np.ndim(mag) != 0:
    raise ValueError("csignscalar expects scalar")
  return np.conj(e) / (mag + eps)


def _prepad(x: Array, n_coef: int) -> Array:
  x_flat = np.asarray(x).ravel()
  return np.concatenate([np.zeros(n_coef - 1, dtype=x_flat.dtype), x_flat])


__all__ = ['dtype2real', 'dtype2complex', 'realsign', 'csignvec', 'csignscalar']
