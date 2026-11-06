"""whitening / fir algorithms and helpers."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from adsp.whitening.base import BaseWhitener


@dataclass
class FirWhitener(BaseWhitener):
  """
  Fixed-coefficient FIR whitener.

  Use this when whitening coefficients are already known from:
    - noise PSD estimation,
    - spectral factorization,
    - offline channel/noise characterization,
    - a previous adaptation phase.

  Example:
    W(z) = 1 - 0.92 z^-1

    init_coef = [1.0, -0.92]
  """

  init_coef: np.ndarray = field(default_factory=lambda: np.array([1.0], dtype=np.complex64))

  def __post_init__(self):
    self.setCoef(self.init_coef)


__all__ = ['FirWhitener']
