"""filters / sign lms algorithms and helpers."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple, Union
from adsp.core import Array
from adsp.filters.base import BaseLMS
from adsp.core import _prepad, csignscalar, csignvec


@dataclass
class QuantizedSharedLMS(BaseLMS):
  """
  Shared analysis for all sign-* / quantized-error LMS algorithms.
  Implements Chapter 4 analysis (Eq 4.13, 4.14, 4.21, 4.28).
  """

  xi_ema_beta: float = 0.99
  xi_hat: float = field(default=1.0, init=False)  # E|e|^2 estimate

  # xi tracking (shared)
  def update_xi_hat(self, e: np.complex64) -> None:
    p = float(np.abs(e)**2)
    self.xi_hat = (self.xi_ema_beta * self.xi_hat + (1.0 - self.xi_ema_beta) * p)

  def excessMSE(self, R: Array, delta_w_cov: Array) -> float:
    return float(np.trace(R @ delta_w_cov).real)

  def analyze(self, R: Array, verbose: bool = False) -> Dict[str, Any]:
    """
    Sign-Error LMS analysis based strictly on Diniz Chapter 4.
    Uses Eq. (4.13), (4.14), (4.21), and (4.28).
    """
    eig = np.linalg.eigvals(R)
    lam = np.real(eig)
    lam_max = float(np.max(lam))
    lam_min = float(np.min(lam))
    tr = float(np.trace(R).real)

    xi = float(self.xi_hat)

    # ---- Bounds from textbook ----
    # Eq (4.13) – mean convergence
    mu_max_mean = (1.0 / lam_max) * np.sqrt(np.pi * xi / 2.0)

    # Eq (4.14) – practical mean bound
    mu_max_trace = (1.0 / tr) * np.sqrt(np.pi * xi / 2.0)

    # Eq (4.21) – MSE (second-order) bound
    mu_max_mse = (1.0 / (2.0 * tr)) * np.sqrt(np.pi * xi / 2.0)

    is_stable_mean = 0.0 < self.mu < mu_max_mean
    is_stable_mse = 0.0 < self.mu < mu_max_mse

    # Eq (4.28)
    misadj = (self.mu * np.sqrt(np.pi / (2.0 * max(xi, 1e-20))) * tr)
    spread = lam_max / lam_min if lam_min > 0 else float("inf")

    res = dict(
      current_mu=float(self.mu),
      xi_hat=float(xi),
      mu_max_mean=float(mu_max_mean),
      mu_max_trace=float(mu_max_trace),
      mu_max_mse=float(mu_max_mse),
      is_stable_mean=is_stable_mean,
      is_stable_mse=is_stable_mse,
      theoretical_misadjustment=float(misadj),
      lambda_max=float(lam_max),
      lambda_min=float(lam_min),
      trace_R=float(tr),
      eigenvalue_spread=float(spread),
    )

    if verbose:
      self.printMuAnalysisSign(res)

    return res

  def printMuAnalysisSign(self, a: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("QUANTIZED / SIGN LMS ANALYSIS (Ch.4)")
    print("=" * 60)
    print(f"mu: {a['current_mu']:.6e}")
    print(f"xi_hat: {a['xi_hat']:.6e}")
    print("\nBounds:")
    print(f"Mean bound (4.13): {a['mu_max_mean']:.6e}")
    print(f"MSE bound  (4.21): {a['mu_max_mse']:.6e}")
    print(f"Mean stable: {'O' if a['is_stable_mean'] else 'X'}")
    print(f"MSE stable:  {'O' if a['is_stable_mse'] else 'X'}")
    print("\nMisadjustment (4.28):")
    print(f"M = {a['theoretical_misadjustment']:.6e}")
    print("=" * 60)


@dataclass
class SignErrorLMS(QuantizedSharedLMS):
  """
  Complex Sign-Error LMS (Diniz Algorithm 4.1, complex extension)
    w(k+1) = w(k) + 2 mu sgn[e(k)] x(k)
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
      y = np.vdot(w, reg)
      e = np.complex64(d[k]) - y
      if train:
        # track xi_hat = E|e|^2 (for effective-mu analysis)
        self.update_xi_hat(e)
        sigma_e = csignscalar(e)
        w = (w + (2.0 * self.mu) * sigma_e * reg).astype(np.complex64)
      y_hist[k] = y
      e_hist[k] = e
      w_hist[k + 1] = w

    if train:
      self.w = w

    return y_hist, e_hist, w_hist


@dataclass
class SignDataLMS(BaseLMS):
  """
  Complex Sign-Data LMS (Diniz Algorithm 4.2, complex extension)
    w(k+1) = w(k) + 2 mu e*(k) sgn[x(k)]
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
      y = np.vdot(w, reg)
      e = np.complex64(d[k]) - y
      if train:
        sreg = csignvec(reg)
        w = (w + (2.0 * self.mu) * np.conj(e) * sreg).astype(np.complex64)
      y_hist[k] = y
      e_hist[k] = e
      w_hist[k + 1] = w

    if train:
      self.w = w
    return y_hist, e_hist, w_hist


@dataclass
class SignSignLMS(QuantizedSharedLMS):
  """
  Complex Sign-Sign LMS (Diniz Sec. 4.2.4 extended)
    w(k+1) = w(k) + 2 mu sgn[e(k)] sgn[x(k)]
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
      y = np.vdot(w, reg)
      e = np.complex64(d[k]) - y
      if train:
        # track xi_hat = E|e|^2 (for effective-mu analysis)
        self.update_xi_hat(e)
        sigma_e = csignscalar(e)
        sreg = csignvec(reg)
        w = (w + (2.0 * self.mu) * sigma_e * sreg).astype(np.complex64)

      y_hist[k] = y
      e_hist[k] = e
      w_hist[k + 1] = w

    if train:
      self.w = w
    return y_hist, e_hist, w_hist


@dataclass
class DualSignLMS(QuantizedSharedLMS):
  """
  Complex Dual-Sign LMS (Diniz Sec. 4.2.3 style)
  if |e(k)| > rho:
      w(k+1) = w(k) + 2 mu ε sgn[e(k)] x(k)
  else:
      w(k+1) = w(k) + 2 mu sgn[e(k)] x(k)
  """
  epsilon: float = 2.0
  rho: float = 1.0

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
        # track xi_hat = E|e|^2 (for effective-mu analysis)
        self.update_xi_hat(e)
        sigma_e = csignscalar(e)
        gain = self.epsilon if (np.abs(e) > self.rho) else 1.0
        w = (w + (2.0 * self.mu) * gain * sigma_e * reg).astype(np.complex64)

      y_hist[k] = y
      e_hist[k] = e
      w_hist[k + 1] = w

    if train:
      self.w = w
    return y_hist, e_hist, w_hist


@dataclass
class PowerOfTwoErrorLMS(QuantizedSharedLMS):
  """
  Power-of-Two Error LMS (Diniz Sec. 4.2.3)
    pe[e] as in Eq. (4.40)
    w(k+1) = w(k) + 2 mu pe[e(k)] x(k)
  """
  bd: int = 8
  tau: float = 0.0

  def p2e(self, e: Union[complex, Array], eps: float = 1e-12) -> np.complex64:
    e = np.asarray(e, dtype=np.complex64)
    abs_e = np.abs(e)
    s = csignscalar(e, eps=eps)  # already conj(e)/(abs+eps)
    thresh = 2.0**(-(self.bd - 1))

    if abs_e >= 1.0:
      return s
    elif abs_e >= thresh:
      pow_mag = np.exp2(np.floor(np.log2(abs_e + eps))).astype(np.float32)
      return (pow_mag * s).astype(np.complex64)
    else:
      return np.complex64(self.tau) * s

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
        # track xi_hat = E|e|^2 (for effective-mu analysis)
        self.update_xi_hat(e)
        pe = self.p2e(e).astype(np.complex64)
        w = (w + (2.0 * self.mu) * pe * reg).astype(np.complex64)

      y_hist[k] = y
      e_hist[k] = e
      w_hist[k + 1] = w

    if train:
      self.w = w
    return y_hist, e_hist, w_hist


__all__ = ['QuantizedSharedLMS', 'SignErrorLMS', 'SignDataLMS', 'SignSignLMS', 'DualSignLMS', 'PowerOfTwoErrorLMS']
