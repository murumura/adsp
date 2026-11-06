"""filters / lattice / error feedback algorithms and helpers."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple
from adsp.core import Array
from adsp.filters.lattice.aposteriori import LatticeRLS


@dataclass
class ErrorFeedbackLatticeRLS(LatticeRLS):
  """
  Diniz Algorithm 7.3 - Error-feedback LRLS based on a posteriori errors.

  Differences from Algorithm 7.1:
    kappa_f(k,i) is recursively updated by Eq. (7.94).
    kappa_b(k,i) is recursively updated by Eq. (7.95).
    v_i(k)       is recursively updated by Eq. (7.96).

  self.w stores the feedforward lattice coefficients v_i(k).
  """

  kappa_f: Array = field(init=False, repr=False)
  kappa_b: Array = field(init=False, repr=False)
  xi_b_prev2: Array = field(init=False, repr=False)

  def resetState(self) -> None:
    # Reuse Algorithm 7.1 initialization for delta, xi_f, xi_b, gamma, eb_prev, and v.
    super().resetState()

    M = self.n_coef
    self.kappa_f = np.zeros(M, dtype=self.dtype)
    self.kappa_b = np.zeros(M, dtype=self.dtype)

    # Algorithm 7.3 additionally requires xi_b(-2,i) because Eq. (7.94) uses xi_b(k-2,i).
    self.xi_b_prev2 = np.full(M + 1, self.epsilon, dtype=self.dtype)

  def step(self, xk: float, dk: float) -> Dict[str, Array]:
    M = self.n_coef

    # Saved states correspond to time k-1, except xi_b_prev2 = xi_b(k-2,i).
    delta_prev = self.delta.copy()
    kappa_f_prev = self.kappa_f.copy()
    kappa_b_prev = self.kappa_b.copy()
    v_prev = np.asarray(self.w.real, dtype=self.dtype)

    gamma_prev = self.gamma.copy()
    eb_prev = self.eb_prev.copy()

    xi_f_prev = self.xi_f.copy()
    xi_b_prev = self.xi_b.copy()
    xi_b_prev2 = self.xi_b_prev2.copy()

    ef = np.zeros(M + 1, dtype=self.dtype)
    eb = np.zeros(M + 1, dtype=self.dtype)
    error = np.zeros(M + 1, dtype=self.dtype)
    gamma = np.ones(M + 1, dtype=self.dtype)

    xi_f = np.zeros(M + 1, dtype=self.dtype)
    xi_b = np.zeros(M + 1, dtype=self.dtype)

    delta = np.empty(M, dtype=self.dtype)
    kappa_f = np.empty(M, dtype=self.dtype)
    kappa_b = np.empty(M, dtype=self.dtype)
    v = np.empty(M, dtype=self.dtype)

    # Algorithm 7.3 order-0 initialization, Eqs. (7.35)-(7.36).
    gamma[0] = 1.0
    eb[0] = ef[0] = xk
    xi_f[0] = xi_b[0] = xk * xk + self.lam * xi_f_prev[0]
    error[0] = dk

    for i in range(M):
      gamma_prev_i = self.posGuard(gamma_prev[i])
      gamma_i = self.posGuard(gamma[i])

      xi_f_i = self.posGuard(xi_f[i])
      xi_f_prev_i = self.posGuard(xi_f_prev[i])

      xi_b_i = self.posGuard(xi_b[i])
      xi_b_prev_i = self.posGuard(xi_b_prev[i])
      xi_b_prev2_i = self.posGuard(xi_b_prev2[i])

      # Eq. (7.51): prediction cross-correlation time update.
      pred_corr = eb_prev[i] * ef[i] / gamma_prev_i
      delta[i] = self.lam * delta_prev[i] + pred_corr

      # Eq. (7.60): conversion-factor order update.
      gamma[i + 1] = gamma[i] - eb[i] * eb[i] / xi_b_i

      # Eq. (7.94): direct recursive forward reflection-coefficient update.
      kappa_f[i] = gamma_prev[i + 1] / gamma_prev_i * (kappa_f_prev[i] + pred_corr / (self.lam * xi_b_prev2_i))

      # Eq. (7.95): direct recursive backward reflection-coefficient update.
      kappa_b[i] = gamma[i + 1] / gamma_prev_i * (kappa_b_prev[i] + pred_corr / (self.lam * xi_f_prev_i))

      # Eqs. (7.34), (7.33): prediction-error order updates.
      eb[i + 1] = eb_prev[i] - kappa_b[i] * ef[i]
      ef[i + 1] = ef[i] - kappa_f[i] * eb_prev[i]

      # Eqs. (7.31), (7.27): minimum LS prediction-error updates.
      xi_f[i + 1] = xi_f[i] - delta[i] * delta[i] / xi_b_prev_i
      xi_b[i + 1] = xi_b_prev[i] - delta[i] * delta[i] / xi_f_i

      # Eq. (7.96): direct recursive feedforward-coefficient update.
      ff_corr = error[i] * eb[i] / gamma_i
      v[i] = gamma[i + 1] / gamma_i * (v_prev[i] + ff_corr / (self.lam * xi_b_prev_i))

      # Eq. (7.68): joint-process a posteriori output-error update.
      error[i + 1] = error[i] - v[i] * eb[i]

    # Shift time states: k-2 <- k-1, k-1 <- k.
    self.xi_b_prev2 = xi_b_prev
    self.xi_b = xi_b
    self.xi_f = xi_f

    self.delta = delta
    self.gamma = gamma
    self.eb_prev = eb

    self.kappa_f = kappa_f
    self.kappa_b = kappa_b
    self.w = v.astype(np.result_type(self.dtype, np.complex64))

    return {
      "y": np.asarray(dk - error[-1], dtype=self.dtype)[()],
      "e": np.asarray(error[-1], dtype=self.dtype)[()],
      "e_order": error,
      "ef": ef,
      "eb": eb,
      "delta": delta,
      "gamma": gamma,
      "xi_f": xi_f,
      "xi_b": xi_b,
      "kappa_f": kappa_f,
      "kappa_b": kappa_b,
      "v": v,
    }

  def __call__(self, x: Array, d: Array, train: bool = True) -> Tuple[Array, Array, Array]:
    # LatticeRLS.__call__ already provides y/e/v histories. Only preserve the extra Algorithm 7.3 states for
    # train=False.
    if train:
      return super().__call__(x, d, train=True)

    extra_state = (self.kappa_f.copy(), self.kappa_b.copy(), self.xi_b_prev2.copy())
    result = super().__call__(x, d, train=False)
    self.kappa_f, self.kappa_b, self.xi_b_prev2 = extra_state

    return result


__all__ = ['ErrorFeedbackLatticeRLS']
