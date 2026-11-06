"""filters / lattice / base algorithms and helpers."""

from __future__ import annotations

import numpy as np
import math
from dataclasses import dataclass
from typing import Tuple
from adsp.core import Array
from adsp.filters.base import BaseLMS


@dataclass
class BaseLatticeRLS(BaseLMS):
  """Shared base for Diniz Algorithm 7.1 and 7.2."""

  lam: float = 0.99
  epsilon: float = 1e-3
  eps: float = 1e-12
  dtype: object = np.float64

  def __post_init__(self):
    self.mu = None
    super().__post_init__()

    if not (0.0 < self.lam <= 1.0):
      raise ValueError(f"lam must satisfy 0 < lam <= 1, got {self.lam}.")
    if self.epsilon <= 0.0:
      raise ValueError(f"epsilon must be > 0, got {self.epsilon}.")
    if self.eps <= 0.0:
      raise ValueError(f"eps must be > 0, got {self.eps}.")

    self.resetState()

  def checkRealInput(self, x: Array, d: Array) -> Tuple[Array, Array]:
    x = np.asarray(x)
    d = np.asarray(d)

    if np.iscomplexobj(x) and np.any(np.abs(np.imag(x)) > self.eps):
      raise ValueError("Diniz Algorithm 7.1/7.2 here implements the real-valued textbook form.")
    if np.iscomplexobj(d) and np.any(np.abs(np.imag(d)) > self.eps):
      raise ValueError("Diniz Algorithm 7.1/7.2 here implements the real-valued textbook form.")

    x = np.asarray(np.real(x), dtype=self.dtype)
    d = np.asarray(np.real(d), dtype=self.dtype)

    if x.ndim != 1 or d.ndim != 1:
      raise ValueError(f"x and d must be 1-D, got {x.shape} and {d.shape}.")
    if len(x) != len(d):
      raise ValueError(f"x and d must have same length, got {len(x)} and {len(d)}.")

    return x, d

  def posGuard(self, x: float) -> float:
    return max(float(x), self.eps)

  def sqrtOneMinusSquare(self, x: float) -> float:
    return math.sqrt(max(1.0 - float(x) * float(x), self.eps))


__all__ = ['BaseLatticeRLS']
