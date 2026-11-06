"""filters / base algorithms and helpers."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from adsp.core import Array


@dataclass
class BaseLMS:
  filter_order: int
  mu: Optional[float] = None  # Default to None for non-LMS types
  init_coef: Optional[Array] = None
  w: Array = field(init=False)

  def __post_init__(self):
    self.n_coef = self.filter_order + 1
    if self.init_coef is None:
      self.w = np.zeros((self.n_coef, ), dtype=np.complex64)
    else:
      self.w = np.asarray(self.init_coef, dtype=np.complex64).copy()
      if self.w.shape != (self.n_coef, ):
        raise ValueError(f"init_coef must have shape {(self.n_coef,)}, got {self.w.shape}.")


__all__ = ['BaseLMS']
