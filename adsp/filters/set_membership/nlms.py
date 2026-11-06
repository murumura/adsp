"""filters / set membership / nlms algorithms and helpers."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
from adsp.core import Array
from adsp.filters.base import BaseLMS
from adsp.core import _prepad


@dataclass
class SMPNLMS(BaseLMS):
  """
  Set-Membership NLMS.

  Complex convention:
    y(k) = w^H(k) x(k)
    e(k) = d(k) - y(k)

  Update:
    if |e(k)| <= gamma_bar:
      w(k+1) = w(k)
    else:
      w(k+1) = w(k) + mu_sm e*(k) x(k) / ||x(k)||^2

  where:
    mu_sm = 1 - gamma_bar / |e(k)|
  """

  gamma_bar: float = 0.0
  eps: float = 1e-12
  init_coef: Optional[Array] = None
  mu: Optional[float] = None

  update_flags: Array = field(init=False, repr=False)
  mu_hist: Array = field(init=False, repr=False)
  delta: float = 1e-2

  def __post_init__(self):
    super().__post_init__()

    if self.gamma_bar < 0.0:
      raise ValueError(f"gamma_bar must be non-negative, got {self.gamma_bar}.")

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
    x_pad = _prepad(x, self.n_coef)
    w = self.w.copy()

    y_hist = np.empty((N, ), dtype=np.complex64)
    e_hist = np.empty((N, ), dtype=np.complex64)
    w_hist = np.empty((N + 1, self.n_coef), dtype=np.complex64)

    update_flags = np.zeros((N, ), dtype=bool)
    mu_hist = np.zeros((N, ), dtype=np.float32)

    w_hist[0] = w

    for k in range(N):
      reg = x_pad[k:k + self.n_coef][::-1].astype(np.complex64)

      y = np.vdot(w, reg)
      e = d[k] - y
      abs_e = float(np.abs(e))

      if train and abs_e > self.gamma_bar:
        norm_x = float(np.vdot(reg, reg).real)

        if norm_x > self.eps:
          mu_sm = 1.0 - self.gamma_bar / max(abs_e, self.eps)
          w = (w + mu_sm * np.conj(e) * reg / (norm_x + self.delta)).astype(np.complex64)

          update_flags[k] = True
          mu_hist[k] = np.float32(mu_sm)

      y_hist[k] = y
      e_hist[k] = e
      w_hist[k + 1] = w

    if train:
      self.w = w

    self.update_flags = update_flags
    self.mu_hist = mu_hist

    return y_hist, e_hist, w_hist

  def getUpdateRate(self) -> float:
    if not hasattr(self, "update_flags") or self.update_flags.size == 0:
      return 0.0
    return float(np.mean(self.update_flags))

  def updateRate(self) -> float:
    return self.getUpdateRate()

  def analyzeUpdates(self) -> Dict[str, Any]:
    if not hasattr(self, "update_flags"):
      return dict(update_rate=0.0, n_updates=0, mean_mu_sm=0.0)

    n_updates = int(np.sum(self.update_flags))
    update_rate = float(np.mean(self.update_flags)) if self.update_flags.size else 0.0

    if n_updates > 0:
      active_mu = self.mu_hist[self.update_flags]
      mean_mu = float(np.mean(active_mu))
      max_mu = float(np.max(active_mu))
      min_mu = float(np.min(active_mu))
    else:
      mean_mu = 0.0
      max_mu = 0.0
      min_mu = 0.0

    return dict(
      gamma_bar=float(self.gamma_bar),
      n_updates=n_updates,
      update_rate=update_rate,
      mean_mu_sm=mean_mu,
      max_mu_sm=max_mu,
      min_mu_sm=min_mu,
    )


__all__ = ['SMPNLMS']
