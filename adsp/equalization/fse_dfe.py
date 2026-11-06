"""Cioffi Ch. 3.6 polyphase fractionally spaced MMSE-DFE and ZF-DFE.

The input is the *high-rate* anti-alias-filtered receive stream.  Feedforward
processing is polyphase, while decision feedback remains symbol-rate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
from scipy.signal import fftconvolve
from adsp.core import Array
from adsp.equalization.canonical_dfe import BaseCanonicalDfe
from adsp.equalization.fse import BaseFse


@dataclass
class BaseFractionallySpacedDfe(BaseCanonicalDfe):
  """Ideal polyphase DFE plus finite per-branch FIR and symbol-rate feedback.

  The polyphase convention is shared with BaseFse:
    Y_i[k] = r[ell*k-i], H_i(D) = sum_m g[ell*m-i] D**m.
    z[k] = sum_i sum_m w_branch[i,m] Y_i[k-m].

  The branch FF frequency response is
    W_i(D) = H_i*(D**(-*)) / (ell*||h||**2*factor_scale*G*(D**(-*)))
  and the feedback response is 1-G(D), or 1-G_U(D) when unbiased.

  Receiver noise normalization matches BaseFse:
    R_{N_i N_j}(D) = ell*noise_var*delta_ij.
    SNR_MFB = E_x*||h||**2/noise_var.

  ``n_taps`` is the number of finite FIR taps PER polyphase branch.
  This implementation is a centered truncation of the ideal polyphase FF;
  it is not the optimal finite-length jointly designed DFE.
  """

  g: Array = field(default_factory=lambda: np.array([1.0 + 0j]))
  g_index: Array = field(default_factory=lambda: np.array([0]))
  oversampling: int = 2
  symbol_energy: float = 1.0
  noise_var: float = 0.1
  n_taps: int = 129

  h_branch: Array = field(init=False, repr=False)
  h_index: Array = field(init=False, repr=False)
  h_bins: Array = field(init=False, repr=False)
  w_branch: Array = field(init=False, repr=False)
  w_branch_index: Array = field(init=False, repr=False)
  w_branch_bins: Array = field(init=False, repr=False)

  def __post_init__(self) -> None:
    if self.n_taps <= 0 or self.n_taps % 2 != 1:
      raise ValueError('polyphase n_taps must be a positive odd integer')
    if self.symbol_energy <= 0 or self.noise_var <= 0:
      raise ValueError('symbol_energy and noise_var must be positive')

    # Preserve the existing FSE pulse indexing, matched energy and Q(D).
    # The LE baseline is used only as a polyphase geometry/reference object.
    reference = BaseFse(
      g=self.g, g_index=self.g_index, oversampling=self.oversampling,
      n_taps=self.n_taps, n_fft=self.n_fft, criterion=self.criterion,
      symbol_energy=self.symbol_energy, noise_var=self.noise_var, eps=self.eps,
    )
    self.q = reference.q.copy()
    self.q_zero_index = int(np.flatnonzero(reference.q_index == 0)[0])
    self.h_norm = reference.h_norm
    self.snr_mfb = float(self.symbol_energy * self.h_norm**2 / self.noise_var)
    self.h_branch = reference.h_branch.copy()
    self.h_index = reference.h_index.copy()
    self.h_bins = reference.h_bins.copy()
    super().__post_init__()

  @classmethod
  def fromPulse(cls, g_index: Array, g: Array, **kwargs):
    return cls(g_index=g_index, g=g, **kwargs)

  def design(self) -> Array:
    """Design canonical feedback and per-phase fractionally spaced feedforward."""
    super().design()
    # A matched-filter observation formed from polyphase samples is
    # Y_MF(D) = sum_i H_i*(D**(-*)) Y_i(D) / (ell*||h||).
    # Cascading canonical W_DFE with this observation gives each W_i.
    self.w_branch_bins = (
      np.conj(self.h_bins) * self._w_bins[np.newaxis, :] / (self.oversampling * self.h_norm)
    )
    half = self.n_taps // 2
    self.w_branch_index = np.arange(-half, half + 1, dtype=int)
    periodic = np.fft.ifft(self.w_branch_bins, axis=1)
    self.w_branch = periodic[:, np.mod(self.w_branch_index, self.n_fft)].astype(self.dtype)
    return self.w_branch.copy()

  def polyphaseQBins(self) -> Array:
    """Q_FS(D)=sum_i |H_i(D)|^2/(ell*||h||^2), on the native FFT grid."""
    return np.sum(np.abs(self.h_bins)**2, axis=0) / (self.oversampling * self.h_norm**2)

  def branchFrequencyResponse(self, centered: bool = True) -> tuple[Array, Array]:
    """Return the ideal complex frequency response of every symbol-rate branch."""
    if centered:
      return np.fft.fftshift(self._omega_bins), np.fft.fftshift(self.w_branch_bins, axes=1)
    return self._omega_bins.copy(), self.w_branch_bins.copy()

  def compositeTaps(self) -> tuple[Array, Array]:
    """Interleave w_branch[i,m] as high-rate lag ell*m+i (as in BaseFse)."""
    lag = (
      self.oversampling * self.w_branch_index[np.newaxis, :]
      + np.arange(self.oversampling)[:, np.newaxis]
    ).ravel()
    values = self.w_branch.ravel()
    index = np.arange(lag.min(), lag.max() + 1, dtype=int)
    taps = np.zeros(index.size, dtype=np.complex128)
    taps[lag - index[0]] = values
    return index, taps.astype(self.dtype)

  def feedforwardSamples(self, received: Array, n_symbols: Optional[int] = None) -> Array:
    """Calculate symbol-rate feedforward output from high-rate receive samples."""
    received = np.asarray(received, dtype=self.dtype).reshape(-1)
    if n_symbols is None:
      n_symbols = len(received) // self.oversampling
    if n_symbols < 0:
      raise ValueError('n_symbols must be nonnegative')
    index, taps = self.compositeTaps()
    filtered = fftconvolve(received, taps, mode='full')
    indices = self.oversampling * np.arange(n_symbols) - index[0]
    output = np.zeros(n_symbols, dtype=self.dtype)
    valid = (indices >= 0) & (indices < len(filtered))
    output[valid] = filtered[indices[valid]]
    return output

  def finiteCascade(self) -> tuple[Array, Array]:
    """Return symbol-domain equivalent channel of the actual finite FS feedforward."""
    cascade = sum(
      np.convolve(self.w_branch[i], self.h_branch[i])
      for i in range(self.oversampling)
    )
    start = int(self.w_branch_index[0] + self.h_index[0])
    return np.arange(start, start + cascade.size), cascade


@dataclass
class MMSEFSDFE(BaseFractionallySpacedDfe):
  """Polyphase MMSE-DFE with optional unbiased/biased decision processing."""
  criterion: Literal['mmse'] = 'mmse'


@dataclass
class ZFFSDFE(BaseFractionallySpacedDfe):
  """Polyphase ZF-DFE with postcursor decision feedback."""
  criterion: Literal['zf'] = 'zf'


__all__ = ['BaseFractionallySpacedDfe', 'MMSEFSDFE', 'ZFFSDFE']
