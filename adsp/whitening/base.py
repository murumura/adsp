"""whitening / base algorithms and helpers."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class BaseWhitener:
  """
  Base class for causal FIR whitening filters.

  The filter convention is:

    y[k] = sum_i coef[i] * x[k - i]

  For an AR(p) prediction-error whitener:

    W(z) = 1 - a1*z^-1 - a2*z^-2 - ... - ap*z^-p

  therefore:

    coef = [1, -a1, -a2, ..., -ap]
  """

  eps: float = 1e-12

  # Structural state
  n_taps: int = field(init=False)
  coef: np.ndarray = field(init=False)
  buf: np.ndarray = field(init=False)

  def setCoef(self, coef: np.ndarray):
    """Configure causal FIR whitening coefficients and reset state."""
    coef = np.asarray(coef, dtype=np.complex64).reshape(-1)

    if coef.size == 0:
      raise ValueError("Whitening filter must have at least one coefficient.")

    self.coef = coef.copy()
    self.n_taps = int(coef.size)
    self.buf = np.zeros(max(self.n_taps - 1, 0), dtype=np.complex64)

    self.resetStates()

  def resetStates(self):
    """Zero out the input delay line."""
    self.buf.fill(0)

  def step(self, new_x: complex) -> complex:
    """
    Process one input sample.

    Buffer convention:
      buf[0] = x[k - 1]
      buf[1] = x[k - 2]
      ...
    """
    y = self.coef[0] * new_x

    if self.n_taps > 1:
      y += np.dot(self.coef[1:], self.buf)

      self.buf[1:] = self.buf[:-1]
      self.buf[0] = new_x

    return np.complex64(y)

  def __call__(self, x: np.ndarray, reset: bool = True) -> np.ndarray:
    """Batch causal FIR filtering."""
    x = np.asarray(x, dtype=np.complex64).reshape(-1)

    if reset:
      self.resetStates()

    y = np.zeros(len(x), dtype=np.complex64)

    for k in range(len(x)):
      y[k] = self.step(x[k])

    return y

  def whitenPair(self, x: np.ndarray, d: np.ndarray, normalize: bool = True) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Apply the same whitening filter independently to x and d.

    This is appropriate for system identification when:

      d[n] = h[n] * x[n]

    because:

      W(z)d[n] = h[n] * W(z)x[n]

    Thus the identified system h[n] is unchanged.

    Returns:
      x_white
      d_white
      scale

    If normalize=True, both outputs are divided by the RMS of x_white.
    """
    x = np.asarray(x, dtype=np.complex64).reshape(-1)
    d = np.asarray(d, dtype=np.complex64).reshape(-1)

    if len(x) != len(d):
      raise ValueError("x and d must have the same length.")

    # Each stream needs identical zero initial conditions.
    self.resetStates()
    x_white = self(x, reset=False)

    self.resetStates()
    d_white = self(d, reset=False)

    self.resetStates()

    scale = float(np.sqrt(np.mean(np.abs(x_white)**2)))

    if normalize:
      scale = max(scale, self.eps)
      x_white = x_white / scale
      d_white = d_white / scale

    return x_white, d_white, scale


__all__ = ['BaseWhitener']
