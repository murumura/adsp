"""equalization / fse algorithms and helpers."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Literal
from adsp.core import Array
from scipy.signal import fftconvolve
from adsp.equalization.base import BaseEqualizer
from adsp.equalization.pulses import _requireScipy


@dataclass
class BaseFse(BaseEqualizer):
  """Polyphase fractionally-spaced equalizer for Ch. 3.5.4.

  The high-rate receive pulse ``g[n]`` is split into ``ell=oversampling``
  symbol-rate branches

    h_i[m] = g[ell*m - i],
    Y_i(D) = H_i(D) X(D) + N_i(D).

  The infinite-length single-stream settings are

    W_ZF(D)   = H^dagger(D) / (H^dagger(D) H(D)),
    W_MMSE(D) = H^dagger(D) / (H^dagger(D) H(D) + ell*noise_var/E_x).

  ``w_branch`` is a centered finite FIR truncation of that ideal two-sided
  response; it is not a separately optimized finite-length Wiener FIR.
  """

  g: Array = field(default_factory=lambda: np.array([1.0 + 0j]))
  g_index: Array = field(default_factory=lambda: np.array([0]))
  oversampling: int = 2
  symbol_energy: float = 1.0
  noise_var: float = 0.1
  n_taps: int = 81
  n_fft: int = 8192
  criterion: Literal["mmse", "zf"] = "mmse"

  h_index: Array = field(init=False, repr=False)
  h_branch: Array = field(init=False, repr=False)
  w_index: Array = field(init=False, repr=False)
  w_branch: Array = field(init=False, repr=False)
  h_bins: Array = field(init=False, repr=False)
  w_bins: Array = field(init=False, repr=False)
  s_bins: Array = field(init=False, repr=False)
  q_index: Array = field(init=False, repr=False)
  q: Array = field(init=False, repr=False)
  h_norm: float = field(init=False)

  def __post_init__(self) -> None:
    super().__post_init__()
    self.g = np.asarray(self.g, dtype=np.complex128).reshape(-1)
    self.g_index = np.asarray(self.g_index, dtype=int).reshape(-1)

    if self.g.size == 0 or self.g.size != self.g_index.size:
      raise ValueError("g_index and g must have the same nonzero length")
    if self.g.size > 1 and np.any(np.diff(self.g_index) != 1):
      raise ValueError("g_index must be consecutive")
    if self.oversampling < 2:
      raise ValueError("oversampling must be >= 2")
    if self.n_taps < 1 or self.n_taps % 2 != 1:
      raise ValueError("n_taps must be a positive odd integer")
    if self.n_fft < self.n_taps:
      raise ValueError("n_fft must be >= n_taps")
    if self.symbol_energy <= 0.0 or self.noise_var <= 0.0:
      raise ValueError("symbol_energy and noise_var must be positive")
    if self.criterion not in ("zf", "mmse"):
      raise ValueError("criterion must be 'zf' or 'mmse'")

    # With T=1 and samples spaced 1/ell, this approximates continuous-time energy.
    self.h_norm = float(np.sqrt(np.vdot(self.g, self.g).real / self.oversampling))
    self.design()

  @classmethod
  def fromPulse(cls, g_index: Array, g: Array, **kwargs):
    return cls(g_index=g_index, g=g, **kwargs)

  def design(self) -> Array:
    """Evaluate the ideal polyphase response and extract centered FIR taps."""
    ell = self.oversampling

    self.h_index = np.arange(
      int(np.floor(self.g_index[0] / ell)) - 1,
      int(np.ceil(self.g_index[-1] / ell)) + 2,
      dtype=int,
    )
    self.h_branch = np.zeros((ell, self.h_index.size), dtype=np.complex128)
    for i in range(ell):
      g_positions = ell * self.h_index - i
      valid = (g_positions >= self.g_index[0]) & (g_positions <= self.g_index[-1])
      self.h_branch[i, valid] = self.g[g_positions[valid] - self.g_index[0]]

    if len(self.h_index) > self.n_fft:
      raise ValueError("n_fft must exceed the polyphase channel length")

    # Hi(D) sampled on the unit circle.  FFT of a periodic sequence indexed by
    # physical lag gives sum_m h_i[m] exp(-j Omega m).
    self.h_bins = np.empty((ell, self.n_fft), dtype=np.complex128)
    for i in range(ell):
      h_periodic = np.zeros(self.n_fft, dtype=np.complex128)
      h_periodic[np.mod(self.h_index, self.n_fft)] = self.h_branch[i]
      self.h_bins[i] = np.fft.fft(h_periodic)

    self.s_bins = np.sum(np.abs(self.h_bins)**2, axis=0)
    reg = ell * self.noise_var / self.symbol_energy if self.criterion == "mmse" else 0.0
    if self.criterion == "zf" and np.min(self.s_bins) <= self.eps:
      raise ValueError("ZF-FSE is undefined at a channel spectral zero")

    self.w_bins = np.conj(self.h_bins) / (self.s_bins[None, :] + reg)

    half = self.n_taps // 2
    self.w_index = np.arange(-half, half + 1, dtype=int)
    w_time = np.fft.ifft(self.w_bins, axis=1)
    self.w_branch = w_time[:, np.mod(self.w_index, self.n_fft)].astype(self.dtype)

    # Build Q(D) from the polyphase autocorrelations.  With the normalization
    # used here: sum_i Hi^dagger Hi = ell ||h||^2 Q(D).
    self.q_index = np.arange(-len(self.h_index) + 1, len(self.h_index), dtype=int)
    correlation = sum(np.correlate(hi, hi, mode="full") for hi in self.h_branch)
    self.q = correlation / (ell * self.h_norm**2)
    if not np.isclose(self.q[self.q_index == 0][0], 1.0, atol=1e-10):
      raise AssertionError("polyphase energy normalization is inconsistent")

    return self.w_branch.copy()

  def qBins(self) -> Array:
    """Q(D)=sum_i Hi^dagger(D)Hi(D)/(ell*||h||^2), on the unit circle."""
    return self.s_bins / (self.oversampling * self.h_norm**2)

  def idealMse(self) -> float:
    """Infinite-length MMSE-FSE MSE, averaged over the frequency grid."""
    if self.criterion != "mmse":
      raise ValueError("idealMse is only defined for MMSE-FSE")
    ell = self.oversampling
    c = ell * self.noise_var
    return float(np.mean(self.symbol_energy * c / (self.symbol_energy * self.s_bins + c)))

  def matchedReferenceMse(self) -> float:
    """Ideal matched-filter + MMSE-LE MSE for the same sampled pulse."""
    snr_mfb = self.symbol_energy * self.h_norm**2 / self.noise_var
    return float(np.mean(self.symbol_energy / (1.0 + snr_mfb * self.qBins())))

  def zfNoiseVariance(self) -> float:
    """Infinite-length ZF-FSE output-noise variance under this noise convention."""
    if self.criterion != "zf":
      raise ValueError("zfNoiseVariance is only defined for ZF-FSE")
    return float(self.oversampling * self.noise_var * np.mean(1.0 / self.s_bins))

  def compositeTaps(self) -> tuple[Array, Array]:
    """Interleave polyphase coefficients into one high-rate FIR representation."""
    lag = (self.oversampling * self.w_index[None, :] + np.arange(self.oversampling)[:, None]).ravel()
    vals = self.w_branch.ravel()
    idx = np.arange(np.min(lag), np.max(lag) + 1, dtype=int)
    taps = np.zeros(idx.size, dtype=np.complex128)
    taps[lag - idx[0]] = vals
    return idx, taps

  def apply(self, r: Array, n_symbols: Optional[int] = None) -> Array:
    """Use all high-rate inputs but compute only symbol-rate decision outputs."""
    _requireScipy()
    r_arr = np.asarray(r, dtype=np.complex128).reshape(-1)
    if n_symbols is None:
      n_symbols = len(r_arr) // self.oversampling
    if n_symbols < 0:
      raise ValueError("n_symbols must be nonnegative")

    idx, taps = self.compositeTaps()
    conv = fftconvolve(r_arr, taps, mode="full")
    p = self.oversampling * np.arange(n_symbols, dtype=int) - idx[0]
    output = np.zeros(n_symbols, dtype=np.complex128)
    valid = (p >= 0) & (p < len(conv))
    output[valid] = conv[p[valid]]
    return output.astype(self.dtype, copy=False)

  def __call__(self, r: Array, n_symbols: Optional[int] = None) -> Array:
    return self.apply(r, n_symbols)

  def finiteError(self) -> dict[str, float]:
    """MSE decomposition of the implemented centered FIR truncation."""
    cascade = sum(np.convolve(self.w_branch[i], self.h_branch[i]) for i in range(self.oversampling))
    lag0 = int(self.w_index[0] + self.h_index[0])
    center_pos = -lag0
    if not (0 <= center_pos < cascade.size):
      raise RuntimeError("implemented cascade does not contain lag zero")

    center = cascade[center_pos]
    isi = self.symbol_energy * (np.sum(np.abs(cascade)**2) - abs(center)**2)
    bias = self.symbol_energy * abs(1.0 - center)**2
    filtered = self.oversampling * self.noise_var * np.sum(np.abs(self.w_branch)**2)
    isi = float(max(float(np.real(isi)), 0.0))
    return {
      "alpha_real": float(np.real(center)),
      "alpha_imag": float(np.imag(center)),
      "bias": float(np.real(bias)),
      "residual_isi": isi,
      "filtered_noise": float(np.real(filtered)),
      "mse": float(np.real(bias) + isi + np.real(filtered)),
    }


@dataclass
class MMSEFSE(BaseFse):
  """Infinite-length MMSE-FSE setting plus centered FIR implementation."""
  criterion: Literal["mmse"] = "mmse"


@dataclass
class ZFFSE(BaseFse):
  """Infinite-length ZF-FSE setting plus centered FIR implementation."""
  criterion: Literal["zf"] = "zf"


__all__ = ['BaseFse', 'MMSEFSE', 'ZFFSE']
