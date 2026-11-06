"""filters / lattice / aposteriori algorithms and helpers."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple
from adsp.core import Array
from adsp.filters.lattice.base import BaseLatticeRLS


@dataclass
class LatticeRLS(BaseLatticeRLS):
  """
  Diniz Algorithm 7.1 - Lattice RLS based on a posteriori errors.

  self.w stores the lattice feedforward coefficients:
      self.w[i] = v_i(k)

  These are lattice-basis coefficients, not direct-form transversal FIR taps.
  """

  delta: Array = field(init=False, repr=False)
  delta_d: Array = field(init=False, repr=False)
  xi_b: Array = field(init=False, repr=False)
  xi_f: Array = field(init=False, repr=False)
  gamma: Array = field(init=False, repr=False)
  eb_prev: Array = field(init=False, repr=False)

  def resetState(self) -> None:
    M = self.n_coef

    self.delta = np.zeros(M, dtype=self.dtype)
    self.delta_d = np.zeros(M, dtype=self.dtype)

    self.xi_b = np.full(M + 1, self.epsilon, dtype=self.dtype)
    self.xi_f = np.full(M + 1, self.epsilon, dtype=self.dtype)

    self.gamma = np.ones(M + 1, dtype=self.dtype)
    self.eb_prev = np.zeros(M + 1, dtype=self.dtype)

    self.w = np.zeros(M, dtype=np.result_type(self.dtype, np.complex64))

  def step(self, xk: float, dk: float) -> Dict[str, Array]:
    M = self.n_coef

    delta_prev = self.delta.copy()
    delta_d_prev = self.delta_d.copy()
    xi_b_prev = self.xi_b.copy()
    xi_f_prev = self.xi_f.copy()
    gamma_prev = self.gamma.copy()
    eb_prev = self.eb_prev.copy()

    ef = np.zeros(M + 1, dtype=self.dtype)
    eb = np.zeros(M + 1, dtype=self.dtype)
    error = np.zeros(M + 1, dtype=self.dtype)
    gamma = np.ones(M + 1, dtype=self.dtype)
    xi_b = np.zeros(M + 1, dtype=self.dtype)
    xi_f = np.zeros(M + 1, dtype=self.dtype)

    delta = np.empty(M, dtype=self.dtype)
    delta_d = np.empty(M, dtype=self.dtype)
    kappa_b = np.empty(M, dtype=self.dtype)
    kappa_f = np.empty(M, dtype=self.dtype)
    v = np.empty(M, dtype=self.dtype)

    # Algorithm 7.1 order-0 initialization, Eqs. (7.35)-(7.36).
    gamma[0] = 1.0
    eb[0] = ef[0] = xk
    xi_b[0] = xi_f[0] = xk * xk + self.lam * xi_f_prev[0]
    error[0] = dk

    for i in range(M):
      # Eq. (7.51): time update of prediction cross-correlation.
      delta[i] = self.lam * delta_prev[i] + eb_prev[i] * ef[i] / self.posGuard(gamma_prev[i])

      # Eq. (7.60): order update of conversion factor.
      gamma[i + 1] = gamma[i] - eb[i] * eb[i] / self.posGuard(xi_b[i])

      # Reflection coefficients.
      kappa_b[i] = delta[i] / self.posGuard(xi_f[i])
      kappa_f[i] = delta[i] / self.posGuard(xi_b_prev[i])

      # Eqs. (7.34), (7.33): backward/forward a posteriori errors.
      eb[i + 1] = eb_prev[i] - kappa_b[i] * ef[i]
      ef[i + 1] = ef[i] - kappa_f[i] * eb_prev[i]

      # Eqs. (7.27), (7.31): minimum LS error order updates.
      xi_b[i + 1] = xi_b_prev[i] - delta[i] * kappa_b[i]
      xi_f[i + 1] = xi_f[i] - delta[i] * kappa_f[i]

      # Eqs. (7.64), (7.67), (7.68): joint-process feedforward section.
      delta_d[i] = self.lam * delta_d_prev[i] + eb[i] * error[i] / self.posGuard(gamma[i])
      v[i] = delta_d[i] / self.posGuard(xi_b[i])
      error[i + 1] = error[i] - v[i] * eb[i]

    self.delta = delta
    self.delta_d = delta_d
    self.xi_b = xi_b
    self.xi_f = xi_f
    self.gamma = gamma
    self.eb_prev = eb
    self.w = v.astype(np.result_type(self.dtype, np.complex64))

    return {
      "y": np.asarray(dk - error[-1], dtype=self.dtype)[()],
      "e": np.asarray(error[-1], dtype=self.dtype)[()],
      "e_order": error,
      "ef": ef,
      "eb": eb,
      "delta": delta,
      "delta_d": delta_d,
      "gamma": gamma,
      "xi_f": xi_f,
      "xi_b": xi_b,
      "kappa_f": kappa_f,
      "kappa_b": kappa_b,
      "v": v,
    }

  def __call__(self, x: Array, d: Array, train: bool = True) -> Tuple[Array, Array, Array]:
    x, d = self.checkRealInput(x, d)

    N = len(x)
    M = self.n_coef

    saved_state = None
    if not train:
      saved_state = (
        self.delta.copy(),
        self.delta_d.copy(),
        self.xi_b.copy(),
        self.xi_f.copy(),
        self.gamma.copy(),
        self.eb_prev.copy(),
        self.w.copy(),
      )

    y_hist = np.empty(N, dtype=self.dtype)
    e_hist = np.empty(N, dtype=self.dtype)
    w_hist = np.empty((N + 1, M), dtype=np.result_type(self.dtype, np.complex64))
    w_hist[0] = self.w

    for k in range(N):
      result = self.step(x[k], d[k])
      y_hist[k] = result["y"]
      e_hist[k] = result["e"]
      w_hist[k + 1] = result["v"]

    if not train:
      self.delta, self.delta_d, self.xi_b, self.xi_f, self.gamma, self.eb_prev, self.w = saved_state

    return y_hist, e_hist, w_hist


__all__ = ['LatticeRLS']
