"""filters / newton algorithms and helpers."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple
from adsp.core import Array
from adsp.filters.base import BaseLMS
from adsp.core import _prepad


@dataclass
class LMSNewton(BaseLMS):
  """
  Complex LMS-Newton adaptive filter.
  Implements the LMS-Newton algorithm as described in:
  Diniz, Adaptive Filtering: Algorithms and Practical Implementation.
  Update equations:
      y(k) = w^H(k) x(k)
      e(k) = d(k) - y(k)
      p(k) = P(k) x(k)
      phi(k) = x^H(k) p(k)
      k(k) = p(k) / (alpha + phi(k))
      w(k+1) = w(k) + 2 mu e*(k) p(k)
      P(k+1) = ( P(k) - k(k) x^H(k) P(k) ) / alpha
  where:
      alpha ∈ (0, 1] is the forgetting factor
      P(0) = (1/delta) I
  """
  alpha: float = 0.99  # forgetting factor (≈ 1)
  delta: float = 1e-2  # P(0) = (1/delta) I
  eps: float = 1e-12  # small numerical guard
  R_hat_inv: Array = field(init=False)  # P(k)

  def __post_init__(self):
    super().__post_init__()
    if not (0.0 < self.alpha <= 1.0):
      raise ValueError("alpha must satisfy 0 < alpha <= 1.")
    # P(0) = (1/delta) I
    self.R_hat_inv = (1.0 / self.delta) * np.eye(self.n_coef, dtype=np.complex64)

  def __call__(self, x: Array, d: Array, train: bool = True) -> Tuple[Array, Array, Array]:
    x = np.asarray(x)
    d = np.asarray(d)
    N = x.shape[0]
    x_pad = _prepad(x, self.n_coef)
    w = self.w.copy()
    P = self.R_hat_inv.copy()
    y_hist = np.empty((N, ), dtype=np.complex64)
    e_hist = np.empty((N, ), dtype=np.complex64)
    w_hist = np.empty((N + 1, self.n_coef), dtype=np.complex64)
    w_hist[0] = w

    for k in range(N):
      reg = x_pad[k:k + self.n_coef][::-1].astype(np.complex64)

      y = np.vdot(w, reg)  # w^H x
      e = np.complex64(d[k]) - y

      if train:
        x_col = reg.reshape(-1, 1)
        # p = P x
        p = P @ x_col
        phi = (x_col.conj().T @ p).item()
        denom = self.alpha + phi
        if abs(denom) < self.eps:
          denom = denom + (self.eps + 0.0j)

        w = (w + 2.0 * self.mu * np.conj(e) * p.ravel()).astype(np.complex64)
        P = (P - (p @ p.conj().T) / denom) / self.alpha
        P = 0.5 * (P + P.conj().T)

      y_hist[k] = y
      e_hist[k] = e
      w_hist[k + 1] = w

    if train:
      self.w = w
      self.R_hat_inv = P

    return y_hist, e_hist, w_hist


__all__ = ['LMSNewton']
