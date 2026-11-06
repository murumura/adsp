"""whitening / ar algorithms and helpers."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple
from adsp.whitening.base import BaseWhitener


@dataclass
class ArWhitener(BaseWhitener):
  """
  AR prediction-error whitener using Yule-Walker coefficient estimation.

  Signal model:

    x[n] = a1*x[n-1] + ... + ap*x[n-p] + e[n]

  Whitening filter:

    W(z) = 1 - a1*z^-1 - ... - ap*z^-p
  """

  ar_order: int = 1
  init_ar_coef: Optional[np.ndarray] = None
  ridge: float = 1e-8

  # Estimated AR model
  ar_coef: np.ndarray = field(init=False)

  def __post_init__(self):
    if self.ar_order <= 0:
      raise ValueError("ar_order must be greater than zero.")

    if self.init_ar_coef is None:
      self.ar_coef = np.zeros(self.ar_order, dtype=np.complex64)
    else:
      ar_coef = np.asarray(self.init_ar_coef, dtype=np.complex64).reshape(-1)

      if len(ar_coef) != self.ar_order:
        raise ValueError("init_ar_coef length must equal ar_order. "
                         f"Got {len(ar_coef)} coefficients for ar_order={self.ar_order}.")

      self.ar_coef = ar_coef.copy()

    self.setArCoef(self.ar_coef)

  def setArCoef(self, ar_coef: np.ndarray):
    """
    Set AR coefficients and construct the prediction-error filter.

    Given:
      x[n] = a1*x[n-1] + ... + ap*x[n-p] + e[n]

    construct:
      e[n] = x[n] - a1*x[n-1] - ... - ap*x[n-p]
    """
    ar_coef = np.asarray(ar_coef, dtype=np.complex64).reshape(-1)

    if len(ar_coef) != self.ar_order:
      raise ValueError("ar_coef length must equal ar_order. "
                       f"Got {len(ar_coef)} coefficients for ar_order={self.ar_order}.")

    self.ar_coef = ar_coef.copy()

    whitening_coef = np.concatenate([np.array([1.0], dtype=np.complex64), -self.ar_coef])

    self.setCoef(whitening_coef)

  def estimateArCoef(self, x: np.ndarray) -> np.ndarray:
    """
    Estimate AR coefficients with complex-safe Yule-Walker equations.

    Defines:
      r[l] = E{x[n] * conj(x[n-l])}

    and solves:
      R * a = r_vec
    """
    x = np.asarray(x, dtype=np.complex64).reshape(-1)

    if len(x) <= self.ar_order:
      raise ValueError("Input length must be greater than ar_order. "
                       f"len(x)={len(x)}, ar_order={self.ar_order}.")

    x0 = x - np.mean(x)
    n = len(x0)

    r = np.zeros(self.ar_order + 1, dtype=np.complex128)

    for lag in range(self.ar_order + 1):
      if lag == 0:
        r[lag] = np.vdot(x0, x0) / n
      else:
        r[lag] = np.vdot(x0[:-lag], x0[lag:]) / n

    # r[0] is theoretically real and nonnegative.
    r[0] = np.real(r[0])

    R = np.zeros((self.ar_order, self.ar_order), dtype=np.complex128)

    for row in range(self.ar_order):
      for col in range(self.ar_order):
        lag = row - col

        if lag >= 0:
          R[row, col] = r[lag]
        else:
          R[row, col] = np.conj(r[-lag])

    regularization = self.ridge * max(float(np.real(r[0])), self.eps)
    R += regularization * np.eye(self.ar_order, dtype=np.complex128)

    rhs = r[1:self.ar_order + 1]

    ar_coef = np.linalg.solve(R, rhs)

    return ar_coef.astype(np.complex64)

  def fit(self, x: np.ndarray) -> np.ndarray:
    """
    Estimate AR coefficients from x and update the whitening filter.

    Returns:
      Estimated AR coefficient vector [a1, ..., ap].
    """
    ar_coef = self.estimateArCoef(x)
    self.setArCoef(ar_coef)

    return self.ar_coef.copy()

  def fitTransform(self, x: np.ndarray, normalize: bool = False) -> Tuple[np.ndarray, float]:
    """
    Estimate AR coefficients from x, then whiten x.

    Returns:
      x_white
      scale

    If normalize=False, scale is returned but not applied.
    """
    self.fit(x)

    x_white = self(x, reset=True)

    scale = float(np.sqrt(np.mean(np.abs(x_white)**2)))

    if normalize:
      scale = max(scale, self.eps)
      x_white = x_white / scale

    return x_white, scale

  def fitWhitenPair(self, x: np.ndarray, d: np.ndarray, normalize: bool = True) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Fit the AR model using x, then whiten x and d with the same filter.

    Use this for system identification:

      d[n] = h[n] * x[n]

    where preserving the same target system h[n] matters.
    """
    self.fit(x)

    return self.whitenPair(x=x, d=d, normalize=normalize)


__all__ = ['ArWhitener']
