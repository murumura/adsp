"""filters / set membership / binormalized algorithms and helpers."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
from adsp.core import Array
from adsp.filters.base import BaseLMS
from adsp.core import _prepad


@dataclass
class SMBNLMS(BaseLMS):
  """
  Set-Membership Binormalized LMS.

  algorithm = 1:
    Two-stage SM-BNLMS:
      1. SM-NLMS update for H(k)
      2. If H(k-1) is violated, correct along x(k-1) orthogonal to x(k)

  algorithm = 2:
    One-shot SM-BNLMS:
      update direction preserves previous a posteriori error.
  """

  gamma_bar: float = 0.0
  eps: float = 1e-12
  delta: float = 1e-2
  algorithm: int = 2
  init_coef: Optional[Array] = None
  mu: Optional[float] = None

  update_flags: Array = field(init=False, repr=False)
  correction_flags: Array = field(init=False, repr=False)
  singular_flags: Array = field(init=False, repr=False)
  mu_hist: Array = field(init=False, repr=False)
  mu2_hist: Array = field(init=False, repr=False)

  def __post_init__(self):
    super().__post_init__()
    if self.mu is None:
      self.mu = 1.0
    if self.gamma_bar < 0.0:
      raise ValueError(f"gamma_bar must be non-negative, got {self.gamma_bar}.")

    if self.eps <= 0.0:
      raise ValueError(f"eps must be positive, got {self.eps}.")

    if self.delta < 0.0:
      raise ValueError(f"delta must be non-negative, got {self.delta}.")

    if self.algorithm not in (1, 2):
      raise ValueError(f"algorithm must be 1 or 2, got {self.algorithm}.")

  def __call__(self, x: Array, d: Array, train: bool = True) -> Tuple[Array, Array, Array]:
    x = np.asarray(x, dtype=np.complex64)
    d = np.asarray(d, dtype=np.complex64)

    if x.ndim != 1:
      raise ValueError(f"x must be 1-D, got shape {x.shape}.")

    if d.ndim != 1:
      raise ValueError(f"d must be 1-D, got shape {d.shape}.")

    if x.shape[0] != d.shape[0]:
      raise ValueError(f"x and d must have same length, got {x.shape[0]} and {d.shape[0]}.")

    N = x.shape[0]
    M = self.n_coef

    x_pad = _prepad(x, M)
    w = self.w.copy()

    y_hist = np.empty((N, ), dtype=np.complex64)
    e_hist = np.empty((N, ), dtype=np.complex64)
    w_hist = np.empty((N + 1, M), dtype=np.complex64)

    update_flags = np.zeros((N, ), dtype=bool)
    correction_flags = np.zeros((N, ), dtype=bool)
    singular_flags = np.zeros((N, ), dtype=bool)

    mu_hist = np.zeros((N, ), dtype=np.float32)
    mu2_hist = np.zeros((N, ), dtype=np.float32)

    w_hist[0] = w

    for k in range(N):
      x_cur = x_pad[k:k + M][::-1].astype(np.complex64)

      y = np.vdot(w, x_cur)
      e = d[k] - y
      abs_e = float(np.abs(e))

      if train and abs_e > self.gamma_bar:
        if self.algorithm == 1:
          w, did_update, did_correct, is_singular, mu1, mu2 = self.updateAlg1(
            w=w,
            x_pad=x_pad,
            d=d,
            k=k,
            x_cur=x_cur,
            e=e,
          )
        else:
          w, did_update, did_correct, is_singular, mu1, mu2 = self.updateAlg2(
            w=w,
            x_pad=x_pad,
            k=k,
            x_cur=x_cur,
            e=e,
          )

        update_flags[k] = did_update
        correction_flags[k] = did_correct
        singular_flags[k] = is_singular
        mu_hist[k] = np.float32(mu1)
        mu2_hist[k] = np.float32(mu2)

      y_hist[k] = y
      e_hist[k] = e
      w_hist[k + 1] = w

    if train:
      self.w = w

    self.update_flags = update_flags
    self.correction_flags = correction_flags
    self.singular_flags = singular_flags
    self.mu_hist = mu_hist
    self.mu2_hist = mu2_hist

    return y_hist, e_hist, w_hist

  def updateAlg1(self, w, x_pad, d, k, x_cur, e):
    abs_e = float(np.abs(e))
    norm_cur = float(np.vdot(x_cur, x_cur).real)

    if norm_cur <= self.eps:
      return w, False, False, True, 0.0, 0.0

    mu1 = self.mu * (1.0 - self.gamma_bar / max(abs_e, self.eps))

    w_hat = (w + mu1 * np.conj(e) * x_cur / (norm_cur + self.delta)).astype(np.complex64)

    did_update = True
    did_correct = False
    is_singular = False
    mu2 = 0.0

    if k == 0:
      return w_hat, did_update, did_correct, is_singular, mu1, mu2

    M = self.n_coef
    x_prev = x_pad[k - 1:k - 1 + M][::-1].astype(np.complex64)

    eps_prev = d[k - 1] - np.vdot(w_hat, x_prev)
    abs_eps_prev = float(np.abs(eps_prev))

    if abs_eps_prev <= self.gamma_bar:
      return w_hat, did_update, did_correct, is_singular, mu1, mu2

    c_cur_prev = np.vdot(x_cur, x_prev)
    x_prev_perp = x_prev - (c_cur_prev / max(norm_cur, self.eps)) * x_cur

    norm_perp = float(np.vdot(x_prev_perp, x_prev_perp).real)

    if norm_perp <= self.eps:
      return w_hat, did_update, did_correct, True, mu1, mu2

    mu2 = self.mu * (1.0 - self.gamma_bar / max(abs_eps_prev, self.eps))

    w_new = (w_hat + mu2 * np.conj(eps_prev) * x_prev_perp / (norm_perp + self.delta)).astype(np.complex64)
    did_correct = True

    return w_new, did_update, did_correct, is_singular, mu1, mu2

  def updateAlg2(self, w, x_pad, k, x_cur, e):
    abs_e = float(np.abs(e))

    mu1 = self.mu * (1.0 - self.gamma_bar / max(abs_e, self.eps))

    if k == 0:
      return self.updateSmnlmsFallback(
        w=w,
        x_cur=x_cur,
        e=e,
        mu1=mu1,
      )

    M = self.n_coef
    x_prev = x_pad[k - 1:k - 1 + M][::-1].astype(np.complex64)

    A = float(np.vdot(x_cur, x_cur).real)
    B = float(np.vdot(x_prev, x_prev).real)
    c_cur_prev = np.vdot(x_cur, x_prev)

    D = A * B - float(np.abs(c_cur_prev)**2)

    if A <= self.eps or B <= self.eps or D <= self.eps:
      return self.updateSmnlmsFallback(
        w=w,
        x_cur=x_cur,
        e=e,
        mu1=mu1,
        is_singular=True,
      )

    q = (B * x_cur - np.conj(c_cur_prev) * x_prev) / (D + self.delta)
    w_new = (w + mu1 * np.conj(e) * q).astype(np.complex64)

    return w_new, True, False, False, mu1, 0.0

  def updateSmnlmsFallback(
    self,
    w,
    x_cur,
    e,
    mu1,
    did_correct: bool = False,
    is_singular: bool = False,
  ):
    """
    Fallback to ordinary SM-NLMS when the SM-BNLMS update is singular.

    Complex convention:
      y(k) = w^H x(k)
      e(k) = d(k) - y(k)

    SM-NLMS fallback:
      w(k+1) = w(k) + mu1 * e*(k) x(k) / ||x(k)||^2

    If self.delta > 0, the boundary equality becomes approximate.
    """

    norm_cur = float(np.vdot(x_cur, x_cur).real)

    if norm_cur <= self.eps:
      return w, False, did_correct, True, 0.0, 0.0

    denom = norm_cur + self.delta

    w_new = (w + mu1 * np.conj(e) * x_cur / denom).astype(np.complex64)

    return w_new, True, did_correct, is_singular, mu1, 0.0

  def getUpdateRate(self) -> float:
    if not hasattr(self, "update_flags") or self.update_flags.size == 0:
      return 0.0
    return float(np.mean(self.update_flags))

  def updateRate(self) -> float:
    return self.getUpdateRate()

  def getCorrectionRate(self) -> float:
    if not hasattr(self, "correction_flags") or self.correction_flags.size == 0:
      return 0.0
    return float(np.mean(self.correction_flags))

  def getSingularRate(self) -> float:
    if not hasattr(self, "singular_flags") or self.singular_flags.size == 0:
      return 0.0
    return float(np.mean(self.singular_flags))

  def analyzeUpdates(self) -> Dict[str, Any]:
    if not hasattr(self, "update_flags"):
      return dict(
        gamma_bar=float(self.gamma_bar),
        algorithm=int(self.algorithm),
        n_updates=0,
        update_rate=0.0,
        correction_rate=0.0,
        singular_rate=0.0,
        mean_mu_sm=0.0,
        min_mu_sm=0.0,
        max_mu_sm=0.0,
        mean_mu2_sm=0.0,
      )

    n_updates = int(np.sum(self.update_flags))
    n_corrections = int(np.sum(self.correction_flags))

    update_rate = float(np.mean(self.update_flags)) if self.update_flags.size else 0.0
    correction_rate = float(np.mean(self.correction_flags)) if self.correction_flags.size else 0.0
    singular_rate = float(np.mean(self.singular_flags)) if self.singular_flags.size else 0.0

    if n_updates > 0:
      active_mu = self.mu_hist[self.update_flags]
      mean_mu = float(np.mean(active_mu))
      min_mu = float(np.min(active_mu))
      max_mu = float(np.max(active_mu))
    else:
      mean_mu = 0.0
      min_mu = 0.0
      max_mu = 0.0

    if n_corrections > 0:
      active_mu2 = self.mu2_hist[self.correction_flags]
      mean_mu2 = float(np.mean(active_mu2))
    else:
      mean_mu2 = 0.0

    return dict(
      gamma_bar=float(self.gamma_bar),
      algorithm=int(self.algorithm),
      delta=float(self.delta),
      n_updates=n_updates,
      n_corrections=n_corrections,
      update_rate=update_rate,
      correction_rate=correction_rate,
      singular_rate=singular_rate,
      mean_mu_sm=mean_mu,
      min_mu_sm=min_mu,
      max_mu_sm=max_mu,
      mean_mu2_sm=mean_mu2,
    )

  def printUpdateAnalysis(self) -> None:
    a = self.analyzeUpdates()

    print("\n" + "=" * 60)
    print("SM-BNLMS UPDATE ANALYSIS")
    print("=" * 60)
    print(f"Algorithm:                 {a['algorithm']}")
    print(f"gamma_bar:                 {a['gamma_bar']:.6g}")
    print(f"delta:                     {a['delta']:.6g}")
    print(f"Number of updates:         {a['n_updates']}")
    print(f"Number of corrections:     {a['n_corrections']}")
    print(f"Update rate:               {a['update_rate']:.4f}")
    print(f"Correction rate:           {a['correction_rate']:.4f}")
    print(f"Singular rate:             {a['singular_rate']:.4f}")
    print(f"Mean mu_sm:                {a['mean_mu_sm']:.6f}")
    print(f"Min mu_sm:                 {a['min_mu_sm']:.6f}")
    print(f"Max mu_sm:                 {a['max_mu_sm']:.6f}")
    print(f"Mean mu2_sm:               {a['mean_mu2_sm']:.6f}")
    print("=" * 60)


__all__ = ['SMBNLMS']
