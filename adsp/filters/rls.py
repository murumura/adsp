"""filters / rls algorithms and helpers."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from adsp.core import Array
from adsp.filters.base import BaseLMS
from adsp.core import _prepad


def rlsMisadj2Lambda(
  M_desired: float,
  filter_order: int,
  K: float = 2.0,
) -> float:
  """
  Solve for λ (forgetting factor) from desired RLS misadjustment using Diniz Example 5.3.

  Uses:
    a = (1 - λ) / (1 + λ)
    M = (N+1) a (1 + 2 a K)

  Args:
    M_desired: desired misadjustment M
    filter_order: adaptive filter order N
    K: ratio term used in the textbook expression

  Returns:
    λ in (0, 1]
  """

  if M_desired <= 0:
    raise ValueError("M_desired must be > 0")
  if filter_order < 0:
    raise ValueError("filter_order must be >= 0")
  if K < 0:
    raise ValueError("K must be >= 0")

  L = filter_order + 1  # number of taps

  # 2K a^2 + a - M/L = 0
  c2 = 2.0 * K
  c1 = 1.0
  c0 = -M_desired / L

  if K == 0.0:
    a = M_desired / L
    if a >= 1.0:
      raise ValueError("Requested misadjustment gives a nonpositive forgetting factor.")
    return float((1.0 - a) / (1.0 + a))

  disc = c1 * c1 - 4.0 * c2 * c0
  if disc < 0:
    raise ValueError("No real solution for a")

  sqrt_disc = np.sqrt(disc)

  # physical valid root: a > 0
  a1 = (-c1 + sqrt_disc) / (2.0 * c2)
  a2 = (-c1 - sqrt_disc) / (2.0 * c2)

  a = a1 if a1 > 0 else a2
  if a <= 0:
    raise ValueError("No positive solution for a")

  lam = (1.0 - a) / (1.0 + a)

  if not (0.0 < lam <= 1.0):
    raise ValueError("Computed λ is out of valid range")

  return float(lam)


def rlsMisadj(
  lam: float,
  filter_order: int,
  K: float = 2.0,
) -> float:
  """
  Compute RLS misadjustment:
    M = (N+1) a (1 + 2 a K),
    a = (1-λ)/(1+λ)
  """
  if not (0.0 < lam <= 1.0):
    raise ValueError("lam must satisfy 0 < lam <= 1")
  if filter_order < 0:
    raise ValueError("filter_order must be >= 0")
  if K < 0:
    raise ValueError("K must be >= 0")

  a = (1.0 - lam) / (1.0 + lam)
  L = filter_order + 1
  return float(L * a * (1.0 + 2.0 * a * K))


def rlsMisadj2Ord(
  M_desired: float,
  lam: float,
  K: float = 2.0,
) -> int:
  """
  Solve for filter order N from desired misadjustment:
    M = (N+1) a (1 + 2 a K)
  """
  if M_desired <= 0:
    raise ValueError("M_desired must be > 0")
  if not (0.0 < lam <= 1.0):
    raise ValueError("lam must satisfy 0 < lam <= 1")
  if K < 0:
    raise ValueError("K must be >= 0")

  a = (1.0 - lam) / (1.0 + lam)
  denom = a * (1.0 + 2.0 * a * K)
  if denom <= 0:
    raise ValueError("Invalid denominator")

  N = M_desired / denom - 1.0
  return int(np.round(N))


def rlsSnrInitParams(
  snr_db: float,
  n_coef: int,
  sigma_u2: float = 1.0,
  lam: float = 0.99,
  delta_high_snr: float = 100.0,
  delta_mid_snr: float = 1.0,
  delta_low_snr: float = 0.01,
) -> dict:
  """
  Intuitive SNR-based initialization for a Diniz-style RLS class.

  Your class uses:
      S_D(-1) = delta * I

  Intuition:
    high SNR -> trust data early -> weaker prior protection
             -> larger Diniz delta

    low SNR  -> trust data less early -> stronger prior protection
             -> smaller Diniz delta

  lambda is kept fixed by default because Haykin's SNR discussion is mainly
  about initialization / regularization, not about changing lambda a lot.
  """
  if n_coef <= 0:
    raise ValueError("n_coef must be > 0")
  if sigma_u2 <= 0.0:
    raise ValueError("sigma_u2 must be > 0")
  if not (0.0 < lam <= 1.0):
    raise ValueError("lam must satisfy 0 < lam <= 1")
  if not (delta_high_snr > 0.0 and delta_mid_snr > 0.0 and delta_low_snr > 0.0):
    raise ValueError("all deltas must be > 0")

  # piecewise-linear interpolation in SNR
  if snr_db >= 30.0:
    delta = delta_high_snr
    mode = "high_snr"
  elif snr_db <= -10.0:
    delta = delta_low_snr
    mode = "low_snr"
  elif snr_db >= 10.0:
    t = (snr_db - 10.0) / 20.0
    delta = delta_mid_snr + t * (delta_high_snr - delta_mid_snr)
    mode = "mid_to_high_snr"
  else:
    t = (snr_db + 10.0) / 20.0
    delta = delta_low_snr + t * (delta_mid_snr - delta_low_snr)
    mode = "low_to_mid_snr"

  effective_memory = 1.0 / (1.0 - lam) if lam < 1.0 else float("inf")

  return dict(
    snr_db=float(snr_db),
    n_coef=int(n_coef),
    sigma_u2=float(sigma_u2),
    lambda_value=float(lam),
    delta=float(delta),
    effective_memory=float(effective_memory),
    mode=mode,
  )


@dataclass
class RLS(BaseLMS):
  """
  Conventional Complex RLS (Diniz Algorithm 5.3)

    S_D(k) = R_D^{-1}(k)

    k(k) = S_D(k-1)x(k) / (λ + x^H(k)S_D(k-1)x(k))

    w(k) = w(k-1) + k(k)e*(k)

    S_D(k) = (1/λ)[ S_D(k-1) - k(k)x^H(k)S_D(k-1) ]

  Initialization:
    S_D(-1) = δ I
    w(-1)   = 0
  """
  mu = None
  lam: float = 0.99
  delta: float = 1e-2
  eps: float = 1e-12
  dtype: object = np.complex128
  S_D: Array = field(init=False)

  def __post_init__(self):
    self.mu = None
    super().__post_init__()
    self.S_D = self.delta * np.eye(self.n_coef, dtype=self.dtype)

  def __call__(self, x: Array, d: Array, train: bool = True):
    x = np.asarray(x, dtype=self.dtype)
    d = np.asarray(d, dtype=self.dtype)

    N = x.shape[0]
    x_pad = _prepad(x, self.n_coef)

    w = self.w.copy()
    S_D = self.S_D.copy()

    y_hist = np.empty((N, ), dtype=self.dtype)
    e_hist = np.empty((N, ), dtype=self.dtype)
    w_hist = np.empty((N + 1, self.n_coef), dtype=self.dtype)
    w_hist[0] = w

    for k_idx in range(N):
      xk = x_pad[k_idx:k_idx + self.n_coef][::-1].reshape(-1, 1)

      y = np.vdot(w, xk.ravel())
      e = d[k_idx] - y

      if train:
        psi = S_D @ xk
        denom = self.lam + np.real((xk.conj().T @ psi).item())
        if abs(denom) < self.eps:
          denom += self.eps

        k_vec = psi / denom
        w = w + k_vec.ravel() * np.conj(e)

        S_D = (S_D - k_vec @ (xk.conj().T @ S_D)) / self.lam

        # numerical hygiene
        S_D = 0.5 * (S_D + S_D.conj().T)

      y_hist[k_idx] = y
      e_hist[k_idx] = e
      w_hist[k_idx + 1] = w

    if train:
      self.w = w
      self.S_D = S_D

    return y_hist, e_hist, w_hist

  def analyze(self, R: Optional[Array] = None, verbose: bool = False):
    """
    RLS analysis based on Diniz Chapter 5.

    Structural points:
      Stability condition: 0 < λ ≤ 1
      Effective memory:    N_eff ≈ 1 / (1 - λ)

    Common first-order approximation for λ ≈ 1:
      M ≈ ((1 - λ) / 2) * tr[R]

    If R is not provided, only structural info is returned.
    """

    lam = float(self.lam)
    is_stable = (0.0 < lam <= 1.0)

    if lam < 1.0:
      N_eff = 1.0 / (1.0 - lam)
    else:
      N_eff = float("inf")

    if R is not None:
      tr = float(np.trace(R).real)
      misadj = (1.0 - lam) / 2.0 * tr
    else:
      tr = None
      misadj = None

    eigvals = np.linalg.eigvals(self.S_D)
    abs_eigs = np.abs(eigvals)
    min_abs = float(np.min(abs_eigs))
    max_abs = float(np.max(abs_eigs))
    cond_P = float(max_abs / min_abs) if min_abs > 0.0 else float("inf")

    res = dict(
      lambda_value=lam,
      is_stable=is_stable,
      effective_memory=float(N_eff),
      trace_R=tr,
      theoretical_misadjustment=misadj,
      P_condition_number=cond_P,
    )

    if verbose:
      self.printAnalyzeRLS(res)

    return res

  def printAnalyzeRLS(self, a: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("RLS ANALYSIS (Diniz Chapter 5)")
    print("=" * 60)

    print(f"Lambda (λ): {a['lambda_value']:.6f}")
    print(f"Stability: {'O STABLE' if a['is_stable'] else 'X UNSTABLE'}")

    print("\nMEMORY / TRACKING:")
    print(f"  Effective memory length ≈ {a['effective_memory']:.2f} samples")

    if a["effective_memory"] < 20:
      print("  WARNING: Very short memory -> noisy estimate")
    elif a["effective_memory"] > 1000:
      print("  WARNING: Very long memory -> slow tracking")

    print("\nSTEADY-STATE PERFORMANCE:")
    if a["trace_R"] is not None:
      print(f"  trace[R]: {a['trace_R']:.6f}")
      print(f"  Misadjustment ≈ {a['theoretical_misadjustment']:.6e}")
    else:
      print("  (R not provided -> misadjustment unavailable)")

    print("\nNUMERICAL CONDITIONING:")
    print(f"  cond(S_D): {a['P_condition_number']:.2e}")
    if a["P_condition_number"] > 1e6:
      print("  WARNING: S_D is ill-conditioned -> numerical instability risk")

    print("\nDESIGN INSIGHT:")
    print("  • λ -> 1     -> lower misadjustment, slower tracking")
    print("  • λ smaller  -> faster tracking, noisier estimate")
    print("  • Effective memory ≈ 1 / (1 - λ)")
    print("=" * 60)


@dataclass
class RLSAlt(BaseLMS):
  """
  Alternative Complex RLS (Diniz Algorithm 5.4)

    e(k) = d(k) - w^H(k-1)x(k)
    ψ(k) = S_D(k-1)x(k)
    S_D(k) = (1/λ)[ S_D(k-1) - ψ(k)ψ^H(k)/(λ + ψ^H(k)x(k)) ]
    w(k) = w(k-1) + e*(k) S_D(k)x(k)
  """
  mu = None
  lam: float = 0.99
  delta: float = 1e-2
  eps: float = 1e-12
  dtype: object = np.complex128

  S_D: Array = field(init=False)

  def __post_init__(self):
    self.mu = None
    super().__post_init__()
    self.S_D = self.delta * np.eye(self.n_coef, dtype=self.dtype)

  def __call__(self, x: Array, d: Array, train: bool = True):
    x = np.asarray(x, dtype=self.dtype)
    d = np.asarray(d, dtype=self.dtype)

    N = x.shape[0]
    x_pad = _prepad(x, self.n_coef)

    w = self.w.copy()
    S_D = self.S_D.copy()

    y_hist = np.empty((N, ), dtype=self.dtype)
    e_hist = np.empty((N, ), dtype=self.dtype)
    w_hist = np.empty((N + 1, self.n_coef), dtype=self.dtype)
    w_hist[0] = w

    for k_idx in range(N):
      xk = x_pad[k_idx:k_idx + self.n_coef][::-1].reshape(-1, 1)

      y = np.vdot(w, xk.ravel())
      e = d[k_idx] - y

      if train:
        psi = S_D @ xk
        denom = self.lam + np.real((psi.conj().T @ xk).item())
        if abs(denom) < self.eps:
          denom += self.eps

        S_D = (S_D - (psi @ psi.conj().T) / denom) / self.lam
        S_D = 0.5 * (S_D + S_D.conj().T)

        w = w + np.conj(e) * (S_D @ xk).ravel()

      y_hist[k_idx] = y
      e_hist[k_idx] = e
      w_hist[k_idx + 1] = w

    if train:
      self.w = w
      self.S_D = S_D

    return y_hist, e_hist, w_hist


__all__ = ['rlsMisadj2Lambda', 'rlsMisadj', 'rlsMisadj2Ord', 'rlsSnrInitParams', 'RLS', 'RLSAlt']
