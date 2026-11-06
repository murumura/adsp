"""filters / lattice / normalized algorithms and helpers."""

from __future__ import annotations

import numpy as np
import math
from dataclasses import dataclass, field
from typing import Dict
from adsp.core import Array
from adsp.filters.lattice.base import BaseLatticeRLS


@dataclass
class NormalizedLatticeRLS(BaseLatticeRLS):
  """
  Diniz Algorithm 7.2 - Normalized lattice RLS based on a posteriori errors.

  The algorithm directly produces normalized errors and normalized correlations:
      delta_bar, delta_d_bar, ef_bar, eb_bar, e_bar.

  There is no physical direct-form FIR coefficient vector in Algorithm 7.2, so inherited self.w is not used.
  """

  delta_bar: Array = field(init=False, repr=False)
  delta_d_bar: Array = field(init=False, repr=False)
  eb_bar_prev: Array = field(init=False, repr=False)

  sigma_x2: float = field(init=False)
  sigma_d2: float = field(init=False)

  def resetState(self) -> None:
    M = self.n_coef

    self.delta_bar = np.zeros(M, dtype=self.dtype)
    self.delta_d_bar = np.zeros(M, dtype=self.dtype)
    self.eb_bar_prev = np.zeros(M + 1, dtype=self.dtype)

    self.sigma_x2 = float(self.epsilon)
    self.sigma_d2 = float(self.epsilon)

    self.w = np.zeros(M, dtype=np.complex64)

  def step(self, xk: float, dk: float) -> Dict[str, Array]:
    M = self.n_coef

    delta_prev = self.delta_bar.copy()
    delta_d_prev = self.delta_d_bar.copy()
    eb_prev = self.eb_bar_prev.copy()

    # Algorithm 7.2 input/reference power recursions.
    self.sigma_x2 = self.lam * self.sigma_x2 + xk * xk
    self.sigma_d2 = self.lam * self.sigma_d2 + dk * dk

    sigma_x = math.sqrt(self.posGuard(self.sigma_x2))
    sigma_d = math.sqrt(self.posGuard(self.sigma_d2))

    ef = np.zeros(M + 1, dtype=self.dtype)
    eb = np.zeros(M + 1, dtype=self.dtype)
    error = np.zeros(M + 1, dtype=self.dtype)

    delta = np.empty(M, dtype=self.dtype)
    delta_d = np.empty(M, dtype=self.dtype)
    eta_f = np.empty(M, dtype=self.dtype)
    eta_b = np.empty(M, dtype=self.dtype)
    eta_d = np.empty(M, dtype=self.dtype)

    # Order-0 normalized innovations.
    eb[0] = ef[0] = xk / sigma_x
    error[0] = dk / sigma_d

    for i in range(M):
      sqrt_eb_prev = self.sqrtOneMinusSquare(eb_prev[i])
      sqrt_ef = self.sqrtOneMinusSquare(ef[i])

      # Eq. (7.81): normalized forward/backward correlation.
      delta[i] = delta_prev[i] * sqrt_eb_prev * sqrt_ef + eb_prev[i] * ef[i]

      sqrt_delta = self.sqrtOneMinusSquare(delta[i])

      # Eqs. (7.91)-(7.92).
      eta_f[i] = 1.0 / (sqrt_delta * sqrt_eb_prev)
      eta_b[i] = 1.0 / (sqrt_delta * sqrt_ef)

      # Eqs. (7.84), (7.83).
      eb[i + 1] = eta_b[i] * (eb_prev[i] - delta[i] * ef[i])
      ef[i + 1] = eta_f[i] * (ef[i] - delta[i] * eb_prev[i])

      sqrt_eb = self.sqrtOneMinusSquare(eb[i])
      sqrt_error = self.sqrtOneMinusSquare(error[i])

      # Eq. (7.89): normalized joint-process correlation.
      delta_d[i] = delta_d_prev[i] * sqrt_eb * sqrt_error + error[i] * eb[i]

      # Eq. (7.93) and Eq. (7.88).
      eta_d[i] = 1.0 / (sqrt_eb * self.sqrtOneMinusSquare(delta_d[i]))
      error[i + 1] = eta_d[i] * (error[i] - delta_d[i] * eb[i])

    self.delta_bar = delta
    self.delta_d_bar = delta_d
    self.eb_bar_prev = eb

    return {
      "e_bar": np.asarray(error[-1], dtype=self.dtype)[()],
      "e_bar_order": error,
      "ef_bar": ef,
      "eb_bar": eb,
      "delta_bar": delta,
      "delta_d_bar": delta_d,
      "eta_f": eta_f,
      "eta_b": eta_b,
      "eta_d": eta_d,
      "sigma_x": np.asarray(sigma_x, dtype=self.dtype)[()],
      "sigma_d": np.asarray(sigma_d, dtype=self.dtype)[()],
    }

  def __call__(self, x: Array, d: Array, train: bool = True) -> Dict[str, Array]:
    x, d = self.checkRealInput(x, d)

    N = len(x)
    M = self.n_coef

    saved_state = None
    if not train:
      saved_state = (
        self.delta_bar.copy(),
        self.delta_d_bar.copy(),
        self.eb_bar_prev.copy(),
        self.sigma_x2,
        self.sigma_d2,
      )

    e_bar_hist = np.empty(N, dtype=self.dtype)
    ef_bar_hist = np.empty((N, M + 1), dtype=self.dtype)
    eb_bar_hist = np.empty((N, M + 1), dtype=self.dtype)
    delta_bar_hist = np.empty((N, M), dtype=self.dtype)
    delta_d_bar_hist = np.empty((N, M), dtype=self.dtype)
    eta_f_hist = np.empty((N, M), dtype=self.dtype)
    eta_b_hist = np.empty((N, M), dtype=self.dtype)
    eta_d_hist = np.empty((N, M), dtype=self.dtype)

    for k in range(N):
      result = self.step(x[k], d[k])

      e_bar_hist[k] = result["e_bar"]
      ef_bar_hist[k] = result["ef_bar"]
      eb_bar_hist[k] = result["eb_bar"]
      delta_bar_hist[k] = result["delta_bar"]
      delta_d_bar_hist[k] = result["delta_d_bar"]
      eta_f_hist[k] = result["eta_f"]
      eta_b_hist[k] = result["eta_b"]
      eta_d_hist[k] = result["eta_d"]

    if not train:
      self.delta_bar, self.delta_d_bar, self.eb_bar_prev, self.sigma_x2, self.sigma_d2 = saved_state

    return {
      "e_bar": e_bar_hist,
      "ef_bar_hist": ef_bar_hist,
      "eb_bar_hist": eb_bar_hist,
      "delta_bar_hist": delta_bar_hist,
      "delta_d_bar_hist": delta_d_bar_hist,
      "eta_f_hist": eta_f_hist,
      "eta_b_hist": eta_b_hist,
      "eta_d_hist": eta_d_hist,
    }


__all__ = ['NormalizedLatticeRLS']
