"""filters / lms algorithms and helpers."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, Tuple
from adsp.core import Array
from adsp.filters.base import BaseLMS
from adsp.core import _prepad


@dataclass
class LMS(BaseLMS):
  """
  Complex LMS (Diniz Algorithm 3.2)
    w(k+1) = w(k) + 2 mu e*(k) x(k)
  """

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
      y = np.vdot(w, reg)  # w^H reg
      e = np.complex64(d[k]) - y
      if train:
        w = (w + (2.0 * self.mu) * np.conj(e) * reg).astype(np.complex64)

      y_hist[k] = y
      e_hist[k] = e
      w_hist[k + 1] = w

    if train:
      self.w = w

    return y_hist, e_hist, w_hist

  def excessMSE(self, R: Array, delta_w_cov: Array) -> float:
    """Excess MSE = tr[R * cov(Δw)]."""
    return float(np.trace(R @ delta_w_cov).real)

  def isStableR(self, R: Array) -> bool:
    """Check stability condition by Equation 3.19 and Equation 3.30"""
    eig = np.linalg.eigvals(R)
    lam_max = np.max(np.real(eig))
    tr = np.trace(R).real
    return (0.0 < self.mu < 1.0 / lam_max) and (0.0 < self.mu < 1.0 / tr)

  def analyze(self, R: Array, verbose: bool = False) -> Dict[str, Any]:
    """Analyze step size and suggest improvements"""
    eig = np.linalg.eigvals(R)
    lam_max = float(np.max(np.real(eig)))
    lam_min = float(np.min(np.real(eig)))
    tr = float(np.trace(R).real)

    bound_eig = 1.0 / lam_max
    bound_tr = 1.0 / tr
    is_stable = self.isStableR(R)

    safe_mu = 0.1 * bound_tr
    aggressive_mu = 0.5 * bound_tr

    spread = (lam_max / lam_min) if lam_min > 0 else float("inf")
    misadj = (self.mu * tr) / (1.0 - self.mu * tr) if (1.0 - self.mu * tr) != 0 else float("inf")

    res = dict(
      is_stable_R=is_stable,
      current_mu=float(self.mu),
      max_stable_mu_eigenvalue=float(bound_eig),
      max_stable_mu_trace=float(bound_tr),
      suggested_conservative_mu=float(safe_mu),
      suggested_aggressive_mu=float(aggressive_mu),
      eigenvalue_spread=float(spread),
      convergence_warning=bool(spread > 100.0),
      theoretical_misadjustment=float(misadj),
      lambda_max=float(lam_max),
      lambda_min=float(lam_min),
      trace_R=float(tr),
    )

    if verbose:
      self.printMuAnalysis(res)
    return res

  def printMuAnalysis(self, a: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("LMS STEP SIZE ANALYSIS")
    print("=" * 60)
    print(f"Current mu: {a['current_mu']:.6f}")
    print(f"Stability: {'O STABLE' if a['is_stable_R'] else 'X UNSTABLE'}")

    print("\nEIGENVALUE ANALYSIS:")
    print(f"  Max eigenvalue (lambda_max): {a['lambda_max']:.6f}")
    print(f"  Min eigenvalue (lambda_min): {a['lambda_min']:.6f}")
    print(f"  Eigenvalue spread: {a['eigenvalue_spread']:.2f}")
    if a["convergence_warning"]:
      print("  WARNING: High eigenvalue spread will slow convergence")

    print("\nSTABILITY BOUNDS:")
    print(f"  From eigenvalues (1/lambda_max): {a['max_stable_mu_eigenvalue']:.6f}")
    print(f"  From trace (1/tr[R]): {a['max_stable_mu_trace']:.6f}")

    print("\nPERFORMANCE PREDICTIONS:")
    print(f"  Theoretical misadj: {a['theoretical_misadjustment']:.4f}")

    if a["lambda_min"] > 0:
      slow_tc = 1.0 / (2.0 * a["current_mu"] * a["lambda_min"])
      approx_iters = 4.6 * slow_tc
      print(f"  Approx. convergence iterations: {approx_iters:.0f}")

    print("\nRECOMMENDED mu RANGES:")
    print(f"  Conservative: {a['suggested_conservative_mu']:.6f}")
    print(f"  Aggressive:   {a['suggested_aggressive_mu']:.6f}")

    m1 = a["max_stable_mu_eigenvalue"] / a["current_mu"] if a["current_mu"] > 0 else float("inf")
    m2 = a["max_stable_mu_trace"] / a["current_mu"] if a["current_mu"] > 0 else float("inf")
    print("\nSAFETY MARGINS:")
    print(f"  Eigenvalue bound margin: {m1:.2f}x")
    print(f"  Trace bound margin:      {m2:.2f}x")
    if m1 < 2:
      print("  WARNING: Small stability margin")
    print("=" * 60)


__all__ = ['LMS']
