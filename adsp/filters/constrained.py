"""Monic prediction-error filtering for the original sign-LMS sinusoid example."""

from dataclasses import dataclass

import numpy as np


@dataclass
class MonicSignLMS:
  """Adapt a residual FIR while holding its leading coefficient equal to one."""

  n_taps: int = 3
  mu: float = 0.01

  def __post_init__(self):
    if self.n_taps < 2 or self.mu <= 0:
      raise ValueError("Use at least two taps and a positive mu.")
    self.w = np.r_[1.0, np.zeros(self.n_taps - 1)]

  def __call__(self, x, d, train=True):
    x, d = np.asarray(x, dtype=float), np.asarray(d, dtype=float)
    padded = np.pad(x, (self.n_taps - 1, 0))
    output = np.empty(len(d))
    history = np.empty((len(d) + 1, self.n_taps))
    history[0] = self.w
    for k in range(len(d)):
      reg = padded[k:k + self.n_taps][::-1]
      output[k] = self.w @ reg
      if train:
        self.w -= 2 * self.mu * np.sign(output[k]) * reg
        self.w[0] = 1.0
      history[k + 1] = self.w
    return output, d - output, history
