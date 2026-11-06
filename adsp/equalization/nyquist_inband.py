"""equalization / nyquist inband algorithms and helpers."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from adsp.core import Array
from scipy.signal import fftconvolve
from adsp.equalization.pulses import _requireScipy


@dataclass
class NyquistInband:
  """Four independent real FIR paths for Nyquist-inband equalization.

    [z_I]   [W_II  W_IQ] [I]
    [z_Q] = [W_QI  W_QQ] [Q]

  ``w[out, inp, tap]`` is sampled on the 2/T stream grid.  The textbook
  specifies the four-independent-filter architecture, but not a particular
  LMS/NLMS update.  ``fitNlms`` and ``fitWiener`` are therefore explicit
  executable modeling choices for illustration and verification.
  """

  n_taps: int = 31
  step_size: float = 0.45
  eps: float = 1e-5
  w: Array = field(init=False, repr=False)
  w_index: Array = field(init=False, repr=False)
  trace: Array = field(init=False, repr=False)

  def __post_init__(self) -> None:
    if self.n_taps < 1 or self.n_taps % 2 != 1:
      raise ValueError("n_taps must be a positive odd integer")
    if not 0.0 < self.step_size < 1.0 or self.eps <= 0.0:
      raise ValueError("require 0 < step_size < 1 and eps > 0")
    self.w_index = np.arange(-(self.n_taps // 2), self.n_taps // 2 + 1, dtype=int)
    self.w = np.zeros((2, 2, self.n_taps), dtype=np.float64)
    self.trace = np.zeros(0, dtype=np.float64)

  @staticmethod
  def _validateInputs(i_stream: Array, q_stream: Array) -> tuple[Array, Array]:
    i = np.asarray(i_stream, dtype=np.float64).reshape(-1)
    q = np.asarray(q_stream, dtype=np.float64).reshape(-1)
    if len(i) == 0 or len(i) != len(q):
      raise ValueError("I and Q streams must have the same nonzero length")
    return i, q

  def makeRegressor(self, i_stream: Array, q_stream: Array, symbol_index: int) -> Array:
    """Return the 2*n_taps real observations used for one symbol output."""
    i, q = self._validateInputs(i_stream, q_stream)
    j = 2 * symbol_index - self.w_index
    if j.min() < 0 or j.max() >= len(i):
      raise IndexError("need more input/guard samples for centered taps")
    return np.concatenate((i[j], q[j]))

  def fitNlms(
    self,
    i_stream: Array,
    q_stream: Array,
    desired_passband: Array,
    start: int,
    stop: int,
    epochs: int = 2,
  ) -> Array:
    """Supervised NLMS for the four-real-FIR demonstration."""
    i, q = self._validateInputs(i_stream, q_stream)
    target = np.asarray(desired_passband, dtype=np.complex128).reshape(-1)
    if start < 0 or stop > len(target) or start >= stop or epochs < 1:
      raise ValueError("invalid training range or epochs")
    if 2 * start - self.w_index.max() < 0 or 2 * (stop - 1) - self.w_index.min() >= len(i):
      raise ValueError("training range exceeds available centered FIR inputs")

    coef = self.w.reshape(2, -1)
    trace: list[float] = []
    for _ in range(epochs):
      for k in range(start, stop):
        taps = 2 * k - self.w_index
        u = np.concatenate((i[taps], q[taps]))
        desired = np.array([target[k].real, target[k].imag])
        err = desired - coef @ u
        coef += self.step_size * np.outer(err, u) / (self.eps + np.dot(u, u))
        if k % 128 == 0:
          trace.append(float(np.dot(err, err)))
    self.trace = np.asarray(trace)
    return self.w.copy()

  def fitWiener(
    self,
    i_stream: Array,
    q_stream: Array,
    desired_passband: Array,
    start: int,
    stop: int,
    ridge: float = 1e-4,
  ) -> Array:
    """Finite-FIR regularized least-squares baseline for the inband model."""
    i, q = self._validateInputs(i_stream, q_stream)
    target = np.asarray(desired_passband, dtype=np.complex128).reshape(-1)
    if not (0 <= start < stop <= len(target)) or ridge <= 0.0:
      raise ValueError("invalid range or ridge")

    idx = 2 * np.arange(start, stop)[:, None] - self.w_index[None, :]
    if idx.min() < 0 or idx.max() >= len(i):
      raise ValueError("not enough I/Q samples for design")

    u = np.hstack((i[idx], q[idx]))
    d = np.stack((target[start:stop].real, target[start:stop].imag), axis=1)
    scale = len(u)
    normal = (u.T @ u) / scale + ridge * np.eye(u.shape[1])
    self.w[:] = np.linalg.solve(normal, (u.T @ d) / scale).T.reshape(2, 2, self.n_taps)
    return self.w.copy()

  def apply(self, i_stream: Array, q_stream: Array, n_symbols: int) -> Array:
    """Compute only symbol-time outputs while consuming both 2/T input streams."""
    _requireScipy()
    i, q = self._validateInputs(i_stream, q_stream)
    input_streams = (i, q)
    out = np.zeros((2, n_symbols), dtype=np.float64)
    p = 2 * np.arange(n_symbols) - self.w_index[0]

    for output in range(2):
      for inp in range(2):
        conv = fftconvolve(input_streams[inp], self.w[output, inp], mode="full")
        valid = (p >= 0) & (p < len(conv))
        out[output, valid] += conv[p[valid]]
    return out[0] + 1j * out[1]

  def symmetryMismatch(self) -> float:
    """Zero iff the 2x2 real FIR obeys ordinary complex-FIR symmetry."""
    return float(np.sqrt(np.sum((self.w[0, 0] - self.w[1, 1])**2) + np.sum((self.w[0, 1] + self.w[1, 0])**2)))


__all__ = ['NyquistInband']
