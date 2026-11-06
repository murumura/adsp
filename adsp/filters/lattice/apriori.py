"""filters / lattice / apriori algorithms and helpers."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple
from adsp.core import Array
from adsp.filters.lattice.aposteriori import LatticeRLS


@dataclass
class APrioriLatticeRLS(LatticeRLS):
  """
  Diniz Algorithm 7.4 - LRLS based on a priori errors.

  Main difference from Algorithm 7.1:
    - Current prediction errors use kappa_f(k-1,i), kappa_b(k-1,i).
    - Current joint-process error uses v_i(k-1).
    - Current a priori errors are then used to update delta, delta_D, kappa, and v.

  self.w stores the current feedforward lattice coefficients v_i(k).
  """

  kappa_f: Array = field(init=False, repr=False)
  kappa_b: Array = field(init=False, repr=False)

  def resetState(self) -> None:
    super().resetState()

    M = self.n_coef

    # Algorithm 7.4:
    #   kappa_f(-1,i) = kappa_b(-1,i) = 0
    self.kappa_f = np.zeros(M, dtype=self.dtype)
    self.kappa_b = np.zeros(M, dtype=self.dtype)

    # Eq. (7.103) with delta_D(-1,i)=0 implies v_i(-1)=0.
    self.w = np.zeros(M, dtype=np.result_type(self.dtype, np.complex64))

  def step(self, xk: float, dk: float) -> Dict[str, Array]:
    M = self.n_coef

    # States available at the beginning of time k.
    delta_prev = self.delta.copy()
    delta_d_prev = self.delta_d.copy()

    xi_b_prev = self.xi_b.copy()
    xi_f_prev = self.xi_f.copy()
    gamma_prev = self.gamma.copy()

    eb_prev = self.eb_prev.copy()

    kappa_f_prev = self.kappa_f.copy()
    kappa_b_prev = self.kappa_b.copy()
    v_prev = np.asarray(self.w.real, dtype=self.dtype)

    # Current-time order variables, i = 0 ... N+1.
    ef = np.zeros(M + 1, dtype=self.dtype)
    eb = np.zeros(M + 1, dtype=self.dtype)
    error = np.zeros(M + 1, dtype=self.dtype)

    gamma = np.ones(M + 1, dtype=self.dtype)
    xi_b = np.zeros(M + 1, dtype=self.dtype)
    xi_f = np.zeros(M + 1, dtype=self.dtype)

    # Current-time per-stage variables, i = 0 ... N.
    delta = np.empty(M, dtype=self.dtype)
    delta_d = np.empty(M, dtype=self.dtype)
    kappa_f = np.empty(M, dtype=self.dtype)
    kappa_b = np.empty(M, dtype=self.dtype)
    v = np.empty(M, dtype=self.dtype)

    # -------------------------------------------------------------------------
    # Algorithm 7.4 order-0 initialization.
    # -------------------------------------------------------------------------

    gamma[0] = 1.0
    eb[0] = ef[0] = xk
    xi_b[0] = xi_f[0] = xk * xk + self.lam * xi_f_prev[0]
    error[0] = dk

    for i in range(M):
      gamma_prev_i = self.posGuard(gamma_prev[i])
      gamma_i = self.posGuard(gamma[i])

      xi_b_i = self.posGuard(xi_b[i])
      xi_b_prev_i = self.posGuard(xi_b_prev[i])

      xi_f_i = self.posGuard(xi_f[i])

      # -----------------------------------------------------------------------
      # Eq. (7.97)
      #
      # delta(k,i) = lambda delta(k-1,i)
      #            + gamma(k-1,i) e_b(k-1,i) e_f(k,i)
      # -----------------------------------------------------------------------

      delta[i] = (self.lam * delta_prev[i] + gamma_prev_i * eb_prev[i] * ef[i])

      # -----------------------------------------------------------------------
      # Eq. (7.100), shifted to current k.
      #
      # gamma(k,i+1) = gamma(k,i)
      #              - gamma^2(k,i) e_b^2(k,i) / xi_b(k,i)
      # -----------------------------------------------------------------------

      gamma[i + 1] = (gamma[i] - gamma_i * gamma_i * eb[i] * eb[i] / xi_b_i)

      # -----------------------------------------------------------------------
      # Eqs. (7.99), (7.98)
      #
      # Important: current a priori errors use PREVIOUS-TIME coefficients.
      # -----------------------------------------------------------------------

      eb[i + 1] = eb_prev[i] - kappa_b_prev[i] * ef[i]
      ef[i + 1] = ef[i] - kappa_f_prev[i] * eb_prev[i]

      # -----------------------------------------------------------------------
      # Current reflection coefficients.
      #
      # These are calculated AFTER the current a priori errors and are stored
      # for use at time k+1.
      # -----------------------------------------------------------------------

      kappa_f[i] = delta[i] / xi_b_prev_i
      kappa_b[i] = delta[i] / xi_f_i

      # -----------------------------------------------------------------------
      # Eqs. (7.31), (7.27)
      # -----------------------------------------------------------------------

      xi_f[i + 1] = (xi_f[i] - delta[i] * kappa_f[i])

      xi_b[i + 1] = (xi_b_prev[i] - delta[i] * kappa_b[i])

      # -----------------------------------------------------------------------
      # Feedforward filtering.
      #
      # Eq. (7.101):
      #   delta_D(k,i) = lambda delta_D(k-1,i)
      #                + gamma(k,i) e_b(k,i) e(k,i)
      # -----------------------------------------------------------------------

      delta_d[i] = (self.lam * delta_d_prev[i] + gamma_i * eb[i] * error[i])

      # -----------------------------------------------------------------------
      # Eq. (7.102):
      #
      #   e(k,i+1) = e(k,i) - v_i(k-1) e_b(k,i)
      #
      # Important: use v_prev here, NOT the newly calculated v_i(k).
      # -----------------------------------------------------------------------

      error[i + 1] = (error[i] - v_prev[i] * eb[i])

      # -----------------------------------------------------------------------
      # Eq. (7.103), current-time form used in Algorithm 7.4:
      #
      #   v_i(k) = delta_D(k,i) / xi_b(k,i)
      #
      # It is stored for use as v_i(k-1) at the next sample.
      # -----------------------------------------------------------------------

      v[i] = delta_d[i] / xi_b_i

    # Commit k states. At the next call they become the k-1 states.
    self.delta = delta
    self.delta_d = delta_d

    self.xi_b = xi_b
    self.xi_f = xi_f
    self.gamma = gamma

    self.eb_prev = eb

    self.kappa_f = kappa_f
    self.kappa_b = kappa_b

    self.w = v.astype(np.result_type(self.dtype, np.complex64))

    # Algorithm 7.4 output error is already the a priori error.
    e_final = error[-1]
    y_final = dk - e_final

    return {
      "y": np.asarray(y_final, dtype=self.dtype)[()],
      "e": np.asarray(e_final, dtype=self.dtype)[()],
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

      # Useful for debugging Algorithm 7.4 causality.
      "kappa_f_prev": kappa_f_prev,
      "kappa_b_prev": kappa_b_prev,
      "v_prev": v_prev,
    }

  def __call__(self, x: Array, d: Array, train: bool = True) -> Tuple[Array, Array, Array]:
    # LatticeRLS.__call__ already provides y/e/v histories and calls self.step().
    if train:
      return super().__call__(x, d, train=True)

    # Parent train=False restores the common LatticeRLS states. Preserve the
    # Algorithm 7.4-specific previous reflection-coefficient states as well.
    extra_state = (
      self.kappa_f.copy(),
      self.kappa_b.copy(),
    )

    result = super().__call__(x, d, train=False)

    self.kappa_f, self.kappa_b = extra_state

    return result


__all__ = ['APrioriLatticeRLS']
