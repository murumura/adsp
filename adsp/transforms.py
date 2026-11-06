"""transforms algorithms and helpers."""

from __future__ import annotations

import numpy as np
from adsp.core import Array


def dctOrthoMatrix(M: int, dtype=np.float32) -> Array:
  """
  Orthonormal DCT-II matrix T (M x M):
    s = T x
  """
  n = np.arange(M, dtype=dtype)[:, None]  # (M,1)
  k = np.arange(M, dtype=dtype)[None, :]  # (1,M)
  T = np.cos(np.pi / M * (n + 0.5) * k).astype(dtype)  # (M,M)

  T[:, 0] *= (1.0 / np.sqrt(M))
  if M > 1:
    T[:, 1:] *= np.sqrt(2.0 / M)
  return T


def dftUnitaryMatrix(M: int, dtype=np.complex64) -> Array:
  """
  Unitary DFT matrix T (M x M):
    s = T x
  with 1/sqrt(M) normalization.
  """
  n = np.arange(M)[:, None]
  k = np.arange(M)[None, :]
  W = np.exp(-1j * 2.0 * np.pi * n * k / M).astype(dtype)
  return W / np.sqrt(M)


__all__ = ['dctOrthoMatrix', 'dftUnitaryMatrix']
