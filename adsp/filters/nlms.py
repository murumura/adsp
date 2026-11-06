"""filters / nlms algorithms and helpers."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, Tuple
from adsp.core import Array
from adsp.filters.base import BaseLMS
from adsp.core import _prepad


@dataclass
class NLMSController:
  """
  Variable Step-Size Controller for NLMS
  ---------------------------------------
  Diniz, Adaptive Filtering: Algorithms and Practical Implementation
      Sec. 4.3 (Normalized LMS)
      Variable step-size discussion

  Theory
  ------
  Under Assumptions 1-3 (independence, small step-size,
  and flat spectral input assumption), the optimal
  normalized step size is:

      u_opt(n) = E{|w_delta_u(n)|^2} / E{|e(n)|^2}

  where
      w_delta_u(n) = ε^H(n) u(n)  (noise-free a priori error)
      e(n)   = w_delta_u(n) + v(n)

  Since:
      E{|e(n)|^2} = E{|w_delta_u(n)|^2} + sigma_n^2

  We estimate:

      w_delta_u^2 ≈ e_power - sigma_n^2

  Therefore:

      u_opt ≈ (e_power - sigma_n^2) / e_power

  This matches Haykin Eq. (7.19) after applying
  Assumptions 1-3.

  Practical Notes
  ---------------
  • u_opt → 1 during initial convergence
  • u_opt → 0 near steady state
  • u_opt automatically adapts to SNR
  • Must satisfy stability condition: 0 < u < 2
  """

  # Short-term power smoothing factors
  # First-order recursive estimators:
  #   u_power(n) = γ_u u_power(n-1) + (1-γ_u) ||u(n)||^2
  #   e_power(n) = γ_e e_power(n-1) + (1-γ_e) |e(n)|^2
  #
  # (Haykin Sec. 7.4 practical implementation)
  gamma_u: float = 0.95
  gamma_e: float = 0.95

  # Noise variance estimate sigma_n^2
  # Can be obtained during silence periods (echo canceller practice)
  sigma_v2: float = 0.0

  # Upper bound to enforce NLMS stability (Haykin Eq. 7.18)
  mu_max: float = 1.0

  # Numerical safeguard
  eps: float = 1e-12

  # Internal state (running power estimates)
  u_power: float = 0.0
  e_power: float = 0.0

  def update(self, reg: np.ndarray, e: complex) -> float:
    """
    Update short-term power estimates and compute adaptive u.

    Parameters
    ----------
    reg : ndarray
        Regressor vector u(n)
    e : complex
        A priori error e(n)

    Returns
    -------
    mu_opt : float
        Adaptive normalized step size μ_n
    """

    # 1) Instantaneous power measurements
    # ||u(n)||^2
    u_inst = float(np.real(np.vdot(reg, reg)))

    # |e(n)|^2
    e_inst = float(np.abs(e)**2)

    # 2) First-order exponential smoothing (MSD estimation approach)
    self.u_power = (self.gamma_u * self.u_power + (1 - self.gamma_u) * u_inst)
    self.e_power = (self.gamma_e * self.e_power + (1 - self.gamma_e) * e_inst)

    # 3) Estimate noise-free error power
    #    w_delta_u^2 ≈ e_power - sigma_n^2
    #
    # From:
    #   E{|e|^2} = E{|w_delta_u|^2} + sigma_n^2
    xi_u2 = max(self.e_power - self.sigma_v2, 0.0)

    # 4) Haykin optimal u (Eq. 7.19 under assumptions)
    #   u_opt = E{|w_delta_u|^2} / E{|e|^2}
    if self.e_power > self.eps:
      mu_opt = xi_u2 / self.e_power
    else:
      mu_opt = 0.0

    # 5) Enforce stability bound
    #    0 < u < 2  (Haykin Eq. 7.18)
    mu_opt = min(max(mu_opt, 0.0), self.mu_max)

    return mu_opt


@dataclass
class NLMS(BaseLMS):
  """
  Complex NLMS (Diniz Sec. 4.3)
    mu_k = mu_n / (tau + ||x||^2)
    w(k+1) = w(k) + mu_k e*(k) x(k)
  Here mu means mu_n (should satisfy 0 < mu_n < 2 for stability in the standard case).
  """
  tau: float = 1e-3
  controller: NLMSController | None = None

  def __call__(self, x: Array, d: Array, train: bool = True) -> Tuple[Array, Array, Array]:
    x = np.asarray(x)
    d = np.asarray(d)
    N = x.shape[0]

    x_pad = _prepad(x, self.n_coef)
    w = self.w.copy()

    y_hist = np.empty((N, ), dtype=np.complex64)
    e_hist = np.empty((N, ), dtype=np.complex64)
    w_hist = np.empty((N + 1, self.n_coef), dtype=np.complex64)
    w_hist[0] = w

    for k in range(N):
      reg = x_pad[k:k + self.n_coef][::-1].astype(np.complex64)
      y = np.vdot(w, reg)
      e = np.complex64(d[k]) - y

      if train:
        # step size control
        if self.controller is not None:
          mu_n = self.controller.update(reg, e)
        else:
          mu_n = self.mu
        norm = float(np.real(np.vdot(reg, reg)))
        mu_k = mu_n / (self.tau + norm)

        w = (w + mu_k * np.conj(e) * reg).astype(np.complex64)

      y_hist[k] = y
      e_hist[k] = e
      w_hist[k + 1] = w

    if train:
      self.w = w
    return y_hist, e_hist, w_hist

  def analyze(self, R: Array, verbose: bool = False):

    tr = float(np.trace(R).real)

    # If controller exists, u is time-varying
    if self.controller is not None:
      mu_current = None
      mu_bound = self.controller.mu_max
    else:
      mu_current = float(self.mu)
      mu_bound = 2.0

    is_stable = True if mu_bound <= 2.0 else False

    misadj_small_mu = None
    if mu_current is not None:
      misadj_small_mu = mu_current / 2.0

    res = dict(
      tau=self.tau,
      trace_R=tr,
      controller_enabled=self.controller is not None,
      stability_bound=2.0,
      mu_current=mu_current,
      mu_max=mu_bound,
      theoretical_misadjustment=misadj_small_mu,
    )

    if verbose:
      self.printMuAnalysisNLMS(res)

    return res

  def printMuAnalysisNLMS(self, a: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("NLMS STEP SIZE ANALYSIS (Haykin Ch.7 / Diniz Sec.4.3)")
    print("=" * 60)

    print(f"tau: {a['tau']:.6e}")
    print(f"trace[R]: {a['trace_R']:.6e}")

    print("\nStability Condition (Haykin Eq. 7.18):")
    print("  0 < u(n) < 2")

    if a["controller_enabled"]:
      print("\nStep-size Mode: VARIABLE (controller enabled)")
      print(f"  u_max enforced: {a['mu_max']:.6f}")
      print("  Stability guaranteed if mu_max < 2")

      if a["mu_current"] is not None:
        print(f"  Current μ̃: {a['mu_current']:.6f}")

    else:
      print("\nStep-size Mode: CONSTANT")

      print(f"  u_n(mu_n): {a['mu_current']:.6f}")
      print(f"  Stability: {'O STABLE' if a['mu_current'] < 2.0 else 'X UNSTABLE'}")

      print("\nEffective LMS-equivalent step size:")
      mu_eff = a['mu_current'] / (2.0 * a['trace_R'])
      print(f"  u_eff ≈ u_opt/ (2 tr[R]) = {mu_eff:.6e}")

      if a["theoretical_misadjustment"] is not None:
        print("\nTheoretical steady-state misadjustment (small u):")
        print(f"  M ≈ u_n / 2 = {a['theoretical_misadjustment']:.6f}")

      print("\nRecommended u_n ranges (practical):")
      print("  Conservative: 0.3-0.5")
      print("  Aggressive:   0.7-1.0")

    print("=" * 60)


__all__ = ['NLMSController', 'NLMS']
