"""filters / set membership / affine projection algorithms and helpers."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
from adsp.core import Array
from adsp.filters.base import BaseLMS
from adsp.core import _prepad


@dataclass
class SMAffineProjection(BaseLMS):
  """
  Simplified Set-Membership Affine Projection.

  P = projection order.
  This implementation uses P+1 reused vectors.

  Simplified SM-AP RHS:
    [mu_sm e(k), 0, ..., 0]^T
  """

  P: int = 3
  gamma_bar: float = 0.0
  delta: float = 1e-3
  eps: float = 1e-12
  mu: Optional[float] = None

  update_flags: Array = field(init=False, repr=False)
  mu_sm_hist: Array = field(init=False, repr=False)
  post_e_hist: Array = field(init=False, repr=False)

  def __post_init__(self):
    super().__post_init__()

    if self.P < 0:
      raise ValueError(f"P must be non-negative, got {self.P}.")

    if self.gamma_bar < 0.0:
      raise ValueError(f"gamma_bar must be non-negative, got {self.gamma_bar}.")

    if self.delta < 0.0:
      raise ValueError(f"delta must be non-negative, got {self.delta}.")

    if self.eps <= 0.0:
      raise ValueError(f"eps must be positive, got {self.eps}.")

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
    Pp = self.P + 1

    if N < Pp:
      raise ValueError(f"Need at least P+1={Pp} samples, got N={N}.")

    x_pad = _prepad(x, M)
    w = self.w.copy()

    n_out = N - self.P

    y_hist = np.empty((n_out, ), dtype=np.complex64)
    e_hist = np.empty((n_out, ), dtype=np.complex64)
    w_hist = np.empty((n_out + 1, M), dtype=np.complex64)

    update_flags = np.zeros((n_out, ), dtype=bool)
    mu_sm_hist = np.zeros((n_out, ), dtype=np.float32)
    post_e_hist = np.empty((n_out, Pp), dtype=np.complex64)

    eye = np.eye(Pp, dtype=np.complex64)

    w_hist[0] = w

    out_idx = 0

    for k in range(self.P, N):
      cols = []

      for i in range(Pp):
        kk = k - i
        reg = x_pad[kk:kk + M][::-1].astype(np.complex64)
        cols.append(reg)

      Xap = np.stack(cols, axis=1)
      d_ap = np.array([d[k - i] for i in range(Pp)], dtype=np.complex64)

      y_ap = Xap.T @ np.conj(w)
      e_ap = d_ap - y_ap

      e0 = e_ap[0]
      abs_e0 = float(np.abs(e0))

      if train and abs_e0 > self.gamma_bar:
        mu_sm = 1.0 - self.gamma_bar / max(abs_e0, self.eps)

        rhs = np.zeros((Pp, ), dtype=np.complex64)
        rhs[0] = np.complex64(mu_sm) * e0

        R = Xap.conj().T @ Xap
        R_reg = R + self.delta * eye

        try:
          g = np.linalg.solve(R_reg, np.conj(rhs))
        except np.linalg.LinAlgError:
          g = (np.linalg.pinv(R_reg) @ np.conj(rhs)).astype(np.complex64)

        upd = (Xap @ g).astype(np.complex64)
        w = (w + upd).astype(np.complex64)

        update_flags[out_idx] = True
        mu_sm_hist[out_idx] = np.float32(mu_sm)

      y_post_ap = Xap.T @ np.conj(w)
      post_e_ap = d_ap - y_post_ap

      y_hist[out_idx] = y_ap[0]
      e_hist[out_idx] = e_ap[0]
      post_e_hist[out_idx] = post_e_ap
      w_hist[out_idx + 1] = w

      out_idx += 1

    if train:
      self.w = w

    self.update_flags = update_flags
    self.mu_sm_hist = mu_sm_hist
    self.post_e_hist = post_e_hist

    return y_hist, e_hist, w_hist

  def getUpdateRate(self) -> float:
    if not hasattr(self, "update_flags") or self.update_flags.size == 0:
      return 0.0
    return float(np.mean(self.update_flags))

  def updateRate(self) -> float:
    return self.getUpdateRate()

  def analyzeUpdates(self) -> Dict[str, Any]:
    if not hasattr(self, "update_flags"):
      return dict(
        gamma_bar=float(self.gamma_bar),
        n_updates=0,
        update_rate=0.0,
        mean_mu_sm=0.0,
        min_mu_sm=0.0,
        max_mu_sm=0.0,
      )

    n_updates = int(np.sum(self.update_flags))
    update_rate = float(np.mean(self.update_flags)) if self.update_flags.size else 0.0

    if n_updates > 0:
      active_mu = self.mu_sm_hist[self.update_flags]
      mean_mu = float(np.mean(active_mu))
      min_mu = float(np.min(active_mu))
      max_mu = float(np.max(active_mu))
    else:
      mean_mu = 0.0
      min_mu = 0.0
      max_mu = 0.0

    return dict(
      gamma_bar=float(self.gamma_bar),
      projection_order=int(self.P),
      projection_dimension=int(self.P + 1),
      delta=float(self.delta),
      n_updates=n_updates,
      update_rate=update_rate,
      mean_mu_sm=mean_mu,
      min_mu_sm=min_mu,
      max_mu_sm=max_mu,
    )

  def printUpdateAnalysis(self) -> None:
    a = self.analyzeUpdates()

    print("\n" + "=" * 60)
    print("SIMPLIFIED SM-AP UPDATE ANALYSIS")
    print("=" * 60)
    print(f"Projection order P:        {a['projection_order']}")
    print(f"Projection dimension P+1:  {a['projection_dimension']}")
    print(f"gamma_bar:                 {a['gamma_bar']:.6g}")
    print(f"delta:                     {a['delta']:.6g}")
    print(f"Number of updates:         {a['n_updates']}")
    print(f"Update rate:               {a['update_rate']:.4f}")
    print(f"Mean mu_sm:                {a['mean_mu_sm']:.6f}")
    print(f"Min mu_sm:                 {a['min_mu_sm']:.6f}")
    print(f"Max mu_sm:                 {a['max_mu_sm']:.6f}")
    print("=" * 60)


__all__ = ['SMAffineProjection']
