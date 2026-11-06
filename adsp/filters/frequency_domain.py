"""Constrained overlap-save FDAF with complex samples and complete block tails."""

from dataclasses import dataclass

import numpy as np

from .base import BaseLMS


@dataclass
class OverLapSaveFDAF(BaseLMS):
  """Block FIR adaptation. w contains physical FIR taps, so y=convolve(x,w).

  FFT size is twice the FIR length. Each update uses at most M fresh samples.
  A partial final block is zero-padded and only real samples contribute error.
  w_hist contains the initial state and the weights available after each sample;
  coefficient changes occur only at block boundaries.
  """

  alpha: float = 0.9
  eps: float = 1e-8

  def __post_init__(self):
    self.fft_size = 2 * (self.filter_order + 1)
    super().__post_init__()
    if self.init_coef is not None:
      # BaseLMS defaults to complex64; retain the precision of supplied physical FIR taps.
      self.w = np.asarray(self.init_coef, dtype=np.complex128)
    if not 0 <= self.alpha < 1 or self.eps <= 0:
      raise ValueError("Require 0 <= alpha < 1 and positive eps.")
    self.pow_est = np.full(self.fft_size, self.eps, dtype=np.float64)

  @property
  def w(self):
    return np.fft.ifft(self.Wf)[:self.n_coef].copy()

  @w.setter
  def w(self, value):
    value = np.asarray(value, dtype=np.complex128)
    if value.shape != (self.n_coef, ):
      raise ValueError("w must contain exactly n_coef physical FIR taps.")
    self.Wf = np.fft.fft(np.pad(value, (0, self.fft_size - self.n_coef)))

  def __call__(self, x, d, train=True):
    x, d = np.asarray(x, dtype=np.complex128), np.asarray(d, dtype=np.complex128)
    if x.ndim != 1 or x.shape != d.shape or not len(x):
      raise ValueError("x and d must be matching nonempty vectors.")
    size, block_size = len(x), self.n_coef
    padded = np.pad(x, (block_size, block_size))
    output, error = np.empty(size, dtype=np.complex128), np.empty(size, dtype=np.complex128)
    history = np.empty((size + 1, block_size), dtype=np.complex128)
    history[0] = self.w
    frequency_weights = self.Wf.copy()
    power = self.pow_est.copy()
    for start in range(0, size, block_size):
      stop = min(start + block_size, size)
      count = stop - start
      spectrum = np.fft.fft(padded[start:start + self.fft_size])
      filtered = np.fft.ifft(spectrum * frequency_weights)[block_size:block_size + count]
      output[start:stop] = filtered
      error[start:stop] = d[start:stop] - filtered
      before = np.fft.ifft(frequency_weights)[:block_size]
      history[start + 1:stop + 1] = before
      if train:
        error_pad = np.zeros(self.fft_size, dtype=np.complex128)
        error_pad[block_size:block_size + count] = error[start:stop]
        power = self.alpha * power + (1 - self.alpha) * np.abs(spectrum)**2
        gradient = np.fft.ifft(spectrum.conj() * np.fft.fft(error_pad) / (power + self.eps))
        gradient[block_size:] = 0
        frequency_weights += self.mu * np.fft.fft(gradient)
        history[stop] = np.fft.ifft(frequency_weights)[:block_size]
    if train:
      self.Wf = frequency_weights
      self.pow_est = power
    return output, error, history


__all__ = ["OverLapSaveFDAF"]
