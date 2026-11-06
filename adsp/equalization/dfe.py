"""equalization / dfe algorithms and helpers."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from adsp.core import Array
from adsp.equalization.base import BaseEqualizer


@dataclass
class BaseDfe(BaseEqualizer):
  """
  Base class for decision-feedback equalizers.

  The regressor follows the adaptive-filter convention

    x(k) = [u(k), ..., u(k-Nff+1), d_hat(k-1), ..., d_hat(k-Nfb)]

  and the equalizer output is

    y(k) = w^H(k) x(k).

  This coefficient convention intentionally differs from BaseLinearEq, whose
  self.w contains the actual LTI coefficients of W(D) = sum_k w(k) D^k.
  """

  n_ff: int = 1
  n_fb: int = 0
  constellation: Array = field(default_factory=lambda: np.array([-1.0, 1.0]))
  init_coef: Optional[Array] = None
  dtype: object = np.complex64

  n_taps: int = field(init=False)
  w: Array = field(init=False, repr=False)
  ff_buf: Array = field(init=False, repr=False)
  fb_buf: Array = field(init=False, repr=False)

  def __post_init__(self) -> None:
    super().__post_init__()

    if self.n_ff <= 0:
      raise ValueError(f"n_ff must be > 0, got {self.n_ff}.")
    if self.n_fb < 0:
      raise ValueError(f"n_fb must be >= 0, got {self.n_fb}.")

    self.n_taps = self.n_ff + self.n_fb
    self.constellation = np.asarray(self.constellation, dtype=self.dtype).reshape(-1)

    if self.constellation.size == 0:
      raise ValueError("constellation must contain at least one symbol.")

    self.ff_buf = np.zeros(self.n_ff, dtype=self.dtype)
    self.fb_buf = np.zeros(self.n_fb, dtype=self.dtype)

    if self.init_coef is None:
      self.w = np.zeros(self.n_taps, dtype=self.dtype)
    else:
      coef = np.asarray(self.init_coef, dtype=self.dtype).reshape(-1)
      if coef.size != self.n_taps:
        raise ValueError(f"init_coef must have {self.n_taps} coefficients, got {coef.size}.")
      self.w = coef.copy()

  def decision(self, y: complex) -> complex:
    """Nearest-neighbor constellation decision."""

    idx = int(np.abs(self.constellation - y).argmin())
    return self.constellation[idx].item()

  def makeRegressor(self, new_u: complex) -> Array:
    """Update the delay lines and return [feedforward; feedback] samples."""

    if self.n_ff > 1:
      self.ff_buf[1:] = self.ff_buf[:-1]
    self.ff_buf[0] = new_u
    return np.concatenate([self.ff_buf, self.fb_buf]).astype(self.dtype, copy=False)

  def output(self, reg: Array) -> complex:
    """Evaluate y = w^H x for an externally prepared DFE regressor."""

    reg_arr = np.asarray(reg, dtype=self.dtype).reshape(-1)
    if reg_arr.size != self.n_taps:
      raise ValueError(f"reg must have length {self.n_taps}, got {reg_arr.size}.")
    return np.vdot(self.w, reg_arr).item()

  def updateFeedback(self, y: complex, d_true: complex, training_mode: bool) -> None:
    """Shift the feedback delay line using training or decision-directed data."""

    if self.n_fb == 0:
      return

    if self.n_fb > 1:
      self.fb_buf[1:] = self.fb_buf[:-1]
    self.fb_buf[0] = d_true if training_mode else self.decision(y)

  def resetBuffers(self) -> None:
    """Zero the feedforward and feedback delay lines without changing coefficients."""

    self.ff_buf.fill(0)
    self.fb_buf.fill(0)

  def resetState(self, reset_coef: bool = True) -> None:
    """Reset delay lines and optionally restore the initial coefficient vector."""

    self.resetBuffers()
    if not reset_coef:
      return

    if self.init_coef is None:
      self.w.fill(0)
    else:
      self.w[:] = np.asarray(self.init_coef, dtype=self.dtype).reshape(-1)


@dataclass
class LmsDfe(BaseDfe):
  """LMS DFE with teacher forcing during training and nearest-symbol feedback afterwards."""

  mu: float = 0.01

  def __call__(self, u, d, train=True, *, training_samples=None, reset_buffers=True):
    u, d = np.asarray(u, dtype=self.dtype), np.asarray(d, dtype=self.dtype)
    if u.ndim != 1 or u.shape != d.shape:
      raise ValueError("u and d must be matching vectors.")
    if reset_buffers:
      self.resetBuffers()
    training_samples = len(u) if training_samples is None else training_samples
    if not 0 <= training_samples <= len(u):
      raise ValueError("training_samples must lie between zero and len(u).")
    y, error = np.empty_like(d), np.empty_like(d)
    self.w_hist = np.empty((len(u) + 1, self.n_taps), dtype=self.dtype)
    self.w_hist[0] = self.w
    for k in range(len(u)):
      reg = self.makeRegressor(u[k])
      y[k] = np.vdot(self.w, reg)
      error[k] = d[k] - y[k]
      teacher = train and k < training_samples
      desired = d[k] if teacher else self.decision(y[k])
      if train:
        self.w += 2 * self.mu * np.conj(desired - y[k]) * reg
      self.updateFeedback(y[k], d[k], teacher)
      self.w_hist[k + 1] = self.w
    return y, error, self.w_hist

  def resetStates(self):
    self.resetState()


@dataclass
class RlsDfe(BaseDfe):
  """RLS DFE; delta initializes the inverse correlation matrix as P0=delta*I."""

  lam: float = 0.99
  delta: float = 0.01
  dtype: object = np.complex128
  S_D: Array = field(init=False, repr=False)

  def __post_init__(self):
    super().__post_init__()
    if not 0 < self.lam <= 1 or self.delta <= 0:
      raise ValueError("Require 0 < lam <= 1 and positive delta.")
    self.S_D = self.delta * np.eye(self.n_taps, dtype=self.dtype)

  def resetStates(self):
    super().resetState()
    self.S_D = self.delta * np.eye(self.n_taps, dtype=self.dtype)

  def __call__(self, u, d, train=True, *, training_samples=None, reset_buffers=True):
    u, d = np.asarray(u, dtype=self.dtype), np.asarray(d, dtype=self.dtype)
    if u.ndim != 1 or u.shape != d.shape:
      raise ValueError("u and d must be matching vectors.")
    if reset_buffers:
      self.resetBuffers()
    training_samples = len(u) if training_samples is None else training_samples
    if not 0 <= training_samples <= len(u):
      raise ValueError("training_samples must lie between zero and len(u).")
    y, error = np.empty_like(d), np.empty_like(d)
    self.w_hist = np.empty((len(u) + 1, self.n_taps), dtype=self.dtype)
    self.w_hist[0] = self.w
    for k in range(len(u)):
      reg = self.makeRegressor(u[k])
      y[k] = np.vdot(self.w, reg)
      error[k] = d[k] - y[k]
      teacher = train and k < training_samples
      desired = d[k] if teacher else self.decision(y[k])
      if train:
        projected = self.S_D @ reg
        gain = projected / max(self.lam + float(np.vdot(reg, projected).real), self.eps)
        self.w += gain * np.conj(desired - y[k])
        self.S_D = (self.S_D - np.outer(gain, reg.conj() @ self.S_D)) / self.lam
        self.S_D = (self.S_D + self.S_D.conj().T) / 2
      self.updateFeedback(y[k], d[k], teacher)
      self.w_hist[k + 1] = self.w
    return y, error, self.w_hist


__all__ = ['BaseDfe', 'LmsDfe', 'RlsDfe']

# Known-channel analytical DFE is implemented separately to preserve the
# adaptive BaseDfe/LmsDfe/RlsDfe coefficient and streaming conventions.
from adsp.equalization.canonical_dfe import DfeOutput, BaseCanonicalDfe, MMSEDFE, ZFDFE

__all__ += ['DfeOutput', 'BaseCanonicalDfe', 'MMSEDFE', 'ZFDFE']
