"""filters / transform algorithms and helpers."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Tuple
from adsp.core import Array
from adsp.filters.base import BaseLMS
from adsp.core import _prepad
from adsp.transforms import dctOrthoMatrix, dftUnitaryMatrix


@dataclass
class TransformDomain(BaseLMS):
  """
  Transform-Domain (Power-Normalized) NLMS / TD-LMS (Diniz Sec. 4.5)

    s(k) = T x_reg(k)
    p_i(k) = alpha |s_i(k)|^2 + (1-alpha) p_i(k-1)
    w_hat(k+1) = w_hat(k) + 2*mu * e*(k) * ( s(k) / (gamma + p(k)) )

  self.w stores transform-domain coefficients w_hat.
  """
  matrix: str = "dct"  # "dct" or "dft"
  alpha: float = 0.01
  gamma: float = 1e-8
  init_power: float = 1.0

  def __post_init__(self):
    super().__post_init__()

    M = self.n_coef
    mat = self.matrix.lower()
    if mat == "dct":
      T = dctOrthoMatrix(M, dtype=np.float32).astype(np.complex64)
    elif mat == "dft":
      T = dftUnitaryMatrix(M, dtype=np.complex64)
    else:
      raise ValueError(f"Unknown transform '{self.matrix}', use 'dct' or 'dft'.")

    self.T = T
    self.TH = np.conj(T).T
    self.power = (self.init_power * np.ones((M, ), dtype=np.float32))

  def __call__(self, x: Array, d: Array, train: bool = True) -> Tuple[Array, Array, Array]:
    x = np.asarray(x)
    d = np.asarray(d)
    N = x.shape[0]
    M = self.n_coef

    x_pad = _prepad(x, M)
    w_hat = self.w.copy()  # transform-domain coeffs
    p = self.power.copy()  # real power vector

    y_hist = np.empty((N, ), dtype=np.complex64)
    e_hist = np.empty((N, ), dtype=np.complex64)
    w_time_hist = np.empty((N + 1, M), dtype=np.complex64)

    w_time0 = self.TH @ w_hat
    w_time_hist[0] = w_time0

    for k in range(N):
      reg = x_pad[k:k + M][::-1].astype(np.complex64)
      s = (self.T @ reg).astype(np.complex64)

      p_new = (self.alpha * (np.abs(s)**2).astype(np.float32) + (1.0 - self.alpha) * p).astype(np.float32)

      y = np.vdot(w_hat, s)
      e = np.complex64(d[k]) - y

      if train:
        denom = (self.gamma + p_new).astype(np.float32)
        w_hat = (w_hat + (2.0 * self.mu) * (np.conj(e) * s / denom.astype(np.complex64))).astype(np.complex64)
        p = p_new
      # if not training, still advance y/e histories; keep p unchanged

      y_hist[k] = y
      e_hist[k] = e
      w_time_hist[k + 1] = (self.TH @ w_hat)

    if train:
      self.w = w_hat
      self.power = p

    return y_hist, e_hist, w_time_hist


@dataclass
class SlidingDctLMS(BaseLMS):
  beta: float = 0.99
  gamma: float = 0.98
  eps: float = 1e-8

  def __post_init__(self):
    super().__post_init__()

    M = self.n_coef
    self.m = np.arange(M, dtype=np.float64)

    # Table 8.2: alpha = 1/(2M)
    self.alpha = 1.0 / (2.0 * M)

    # W_{2M} = exp(-j2π/(2M))
    self.W2M = np.exp(-1j * 2.0 * np.pi / (2.0 * M))

    # W_{2M}^m and W_{2M}^{-m}
    self.Wm = self.W2M**self.m
    self.Wm_inv = np.conj(self.Wm)

    # (-1)^m and k_m
    self.sign = (-1.0)**self.m
    self.km = np.ones(M, dtype=np.float64)
    self.km[0] = 1.0 / np.sqrt(2.0)

    # States (Table 8.2 initialization)
    self.A1 = np.zeros(M, dtype=np.complex128)
    self.A2 = np.zeros(M, dtype=np.complex128)
    self.lambda_hat = np.zeros(M, dtype=np.float64)

    # delay line holds u(n-1)...u(n-M)
    self.delay_line = np.zeros(M, dtype=np.float64)

    # IMPORTANT: Table 8.2 uses real w_hat (DCT-domain weights)
    self.w = np.zeros(M, dtype=np.float64)

    self.n_iter = 1

  def estR(self, x):
    """Sample correlation"""
    x = np.asarray(x)
    return np.outer(x, x.conj())

  def step(self, u: float, d: float | None = None, train: bool = True):
    """
    One-sample update of Table 8.2 DCT-LMS.
    Returns: y, e, C  (C is the DCT-domain input vector)
    """

    M = self.n_coef
    u = np.float64(u)

    # sliding window delay
    u_old = self.delay_line[-1]
    self.delay_line[1:] = self.delay_line[:-1]
    self.delay_line[0] = u

    # sliding DCT innovation term: u(n) - beta^M (-1)^m u(n-M)
    innovation = u - (self.beta**M) * self.sign * u_old  # vector over m

    # Fig 8.6 / Table 8.2 recursions
    self.A1 = self.beta * self.Wm * self.A1 + innovation
    self.A2 = self.beta * self.Wm_inv * self.A2 + self.Wm_inv * innovation

    A = self.A1 + self.A2

    # C_m(n)
    C = 0.5 * self.km * self.sign * (self.W2M**(self.m / 2.0)) * A
    C = C.real

    # filter output
    y = float(np.dot(C, self.w))

    if d is None:
      # no training signal
      self.n_iter += 1
      return y, 0.0, C

    d = np.float64(d)
    e = float(d - y)

    # eigenvalue estimation Eq (8.74)
    self.lambda_hat = (self.gamma * self.lambda_hat + (1.0 / self.n_iter) * (C * C - self.gamma * self.lambda_hat))
    self.lambda_hat = np.maximum(self.lambda_hat, self.eps)

    # ---- weight update Table 8.2 ----
    if train:
      step_size = self.mu if self.mu is not None else self.alpha
      self.w += (step_size / self.lambda_hat) * C * e

    self.n_iter += 1
    return y, e, C

  def __call__(self, x, d=None, train=True):
    """Run DCT-LMS over a sequence."""

    x = np.asarray(x, dtype=np.float64)
    N = len(x)

    if d is not None:
      d = np.asarray(d, dtype=np.float64)
      if len(d) != N:
        raise ValueError("x and d must have same length")

    y_hist = np.zeros(N, dtype=np.float64)
    e_hist = np.zeros(N, dtype=np.float64)
    C_hist = np.zeros((N, self.n_coef), dtype=np.float64)

    for n in range(N):
      if d is None:
        y, e, C = self.step(x[n], None, train=False)
      else:
        y, e, C = self.step(x[n], d[n], train=train)

      y_hist[n] = y
      e_hist[n] = e
      C_hist[n] = C

    return y_hist, e_hist, C_hist


__all__ = ['TransformDomain', 'SlidingDctLMS']
