"""equalization / linear algorithms and helpers."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from adsp.core import Array
from adsp.equalization.base import BaseEqualizer


@dataclass
class BaseLinearEq(BaseEqualizer):
  """
  Base class for symbol-spaced linear equalizers used in Chapter 3.

  The matched-filter sampled channel is

    y(k) = ||h|| sum_m q(m) x(k-m) + n(k)

  with

    q(0) = 1
    R_nn(D) = noise_var Q(D)

  where noise_var = N0 / 2 per real dimension in the textbook notation.

  q is stored in ascending physical-lag order. q_zero_index identifies the
  array element corresponding to lag k = 0.

  self.w stores the actual LTI impulse response coefficients of W(D):

    W(D) = sum_k w(k) D^k

  Therefore filtering uses convolution, not y = w^H x as in BaseDfe and
  adaptive-filter classes.
  """

  q: Array = field(default_factory=lambda: np.array([1.0], dtype=np.complex128))
  h_norm: float = 1.0
  q_zero_index: Optional[int] = None

  q_index: Array = field(init=False, repr=False)
  w: Array = field(init=False, repr=False)
  w_index: Array = field(init=False, repr=False)

  def __post_init__(self) -> None:
    super().__post_init__()

    self.q = np.asarray(self.q, dtype=self.dtype).reshape(-1)
    if self.q.size == 0:
      raise ValueError("q must contain at least one coefficient.")

    if self.h_norm <= 0.0:
      raise ValueError(f"h_norm must be > 0, got {self.h_norm}.")

    if self.q_zero_index is None:
      if self.q.size % 2 == 0:
        raise ValueError("q_zero_index is required when q has even length.")
      self.q_zero_index = self.q.size // 2

    if not (0 <= self.q_zero_index < self.q.size):
      raise ValueError(f"q_zero_index must be in [0, {self.q.size - 1}], got {self.q_zero_index}.")

    self.q_index = np.arange(self.q.size, dtype=int) - int(self.q_zero_index)

    q0 = self.q[self.q_zero_index]
    if not np.isclose(q0, 1.0, rtol=1e-7, atol=1e-9):
      raise ValueError(f"q must be normalized to q[0] = 1, got {q0}.")

    self.w = np.zeros(0, dtype=self.dtype)
    self.w_index = np.zeros(0, dtype=int)

  @classmethod
  def fromChannel(cls, h: Array, **kwargs):
    """
    Construct from symbol-spaced channel/basis coefficients h(k).

    This is exact for the textbook examples where h(t) is expanded in an
    orthonormal shifted-sinc basis. For a general continuous-time channel,
    compute q(kT) separately and construct the class with q and h_norm.
    """

    h_arr = np.asarray(h, dtype=np.complex128).reshape(-1)
    if h_arr.size == 0:
      raise ValueError("h must contain at least one coefficient.")

    h_energy = float(np.vdot(h_arr, h_arr).real)
    if h_energy <= 0.0:
      raise ValueError("h must have nonzero energy.")

    q = np.correlate(h_arr, h_arr, mode="full") / h_energy
    return cls(q=q, h_norm=np.sqrt(h_energy), q_zero_index=h_arr.size - 1, **kwargs)

  @staticmethod
  def _filterIndexed(x: Array, taps: Array, tap_index: Array, dtype: object) -> Array:
    """Apply y(k) = sum_m taps(m) x(k-m) with zero extension outside x."""

    x_arr = np.asarray(x, dtype=dtype).reshape(-1)
    taps_arr = np.asarray(taps, dtype=dtype).reshape(-1)
    idx_arr = np.asarray(tap_index, dtype=int).reshape(-1)

    if taps_arr.size != idx_arr.size:
      raise ValueError("taps and tap_index must have the same length.")

    if taps_arr.size == 0:
      return np.zeros_like(x_arr)

    full = np.convolve(x_arr, taps_arr, mode="full")
    start = -int(idx_arr[0])
    stop = start + x_arr.size

    out = np.zeros(x_arr.size, dtype=np.result_type(x_arr.dtype, taps_arr.dtype))
    src_start = max(start, 0)
    src_stop = min(stop, full.size)
    dst_start = src_start - start
    dst_stop = dst_start + max(src_stop - src_start, 0)

    if src_stop > src_start:
      out[dst_start:dst_stop] = full[src_start:src_stop]

    return out.astype(dtype, copy=False)

  def qResponse(self, omega: Array) -> Array:
    """Evaluate Q(exp(-j*omega)) = sum_k q(k) exp(-j*omega*k), omega in rad/sample."""

    omega_arr = np.asarray(omega, dtype=np.float64).reshape(-1)
    phase = np.exp(-1j * omega_arr[:, None] * self.q_index[None, :])
    return (phase @ self.q).astype(self.dtype)

  def qSpectrum(self, n_fft: int = 4096, centered: bool = True) -> tuple[Array, Array]:
    """Return a frequency grid and Q(exp(-j*omega))."""

    if n_fft <= 0:
      raise ValueError(f"n_fft must be positive, got {n_fft}.")

    omega = 2.0 * np.pi * np.fft.fftfreq(n_fft)
    q_spec = self.qResponse(omega)

    if centered:
      omega = np.fft.fftshift(omega)
      q_spec = np.fft.fftshift(q_spec)

    return omega, q_spec

  def matchedFilterSignal(self, x: Array) -> Array:
    """Noiseless matched-filter samples: ||h|| q(k) * x(k)."""

    return self.h_norm * self._filterIndexed(x, self.q, self.q_index, self.dtype)

  def matchedFilterOutput(self, x: Array, noise: Optional[Array] = None) -> Array:
    """Matched-filter sampled channel output."""

    y = self.matchedFilterSignal(x)
    if noise is None:
      return y

    noise_arr = np.asarray(noise, dtype=self.dtype).reshape(-1)
    if noise_arr.shape != y.shape:
      raise ValueError(f"noise must have shape {y.shape}, got {noise_arr.shape}.")
    return y + noise_arr

  def matchedNoiseCorrelation(self, noise_var: float) -> tuple[Array, Array]:
    """Matched-filter output-noise autocorrelation r_nn(k) = noise_var q(k)."""

    if noise_var < 0.0:
      raise ValueError(f"noise_var must be >= 0, got {noise_var}.")
    return self.q_index.copy(), (noise_var * self.q).copy()

  def matchedNoisePsd(
    self,
    noise_var: float,
    n_fft: int = 4096,
    centered: bool = True,
  ) -> tuple[Array, Array]:
    """Matched-filter output-noise PSD: R_nn = noise_var Q."""

    omega, q_spec = self.qSpectrum(n_fft=n_fft, centered=centered)
    return omega, noise_var * q_spec

  def isNyquist(self, atol: float = 1e-9) -> bool:
    """True when q(k) = delta(k), i.e. no symbol-spaced ISI."""

    target = np.zeros_like(self.q)
    target[self.q_zero_index] = 1.0
    return bool(np.allclose(self.q, target, rtol=0.0, atol=atol))

  def peakIsi(self, x_peak: float) -> float:
    """Chapter 3 peak-ISI bound: D_p = |x|max ||h|| sum_{k != 0} |q(k)|."""

    if x_peak < 0.0:
      raise ValueError(f"x_peak must be >= 0, got {x_peak}.")

    mask = self.q_index != 0
    return float(x_peak * self.h_norm * np.sum(np.abs(self.q[mask])))

  def meanSquareIsi(self, symbol_energy: float) -> float:
    """Chapter 3 mean-square ISI: E_x ||h||^2 sum_{k != 0} |q(k)|^2."""

    if symbol_energy < 0.0:
      raise ValueError(f"symbol_energy must be >= 0, got {symbol_energy}.")

    mask = self.q_index != 0
    return float(symbol_energy * self.h_norm**2 * np.sum(np.abs(self.q[mask])**2))

  def mfbSnr(self, symbol_energy: float, noise_var: float) -> float:
    """Matched-filter bound SNR = E_x ||h||^2 / noise_var."""

    if symbol_energy < 0.0:
      raise ValueError(f"symbol_energy must be >= 0, got {symbol_energy}.")
    if noise_var <= 0.0:
      raise ValueError(f"noise_var must be > 0, got {noise_var}.")
    return float(symbol_energy * self.h_norm**2 / noise_var)

  def __call__(self, y: Array) -> Array:
    """Apply the currently designed linear equalizer to matched-filter samples y(k)."""

    if self.w.size == 0:
      raise RuntimeError("Equalizer coefficients are not designed yet.")
    return self._filterIndexed(y, self.w, self.w_index, self.dtype)

  def cascadeResponse(self) -> tuple[Array, Array]:
    """Return the finite implemented cascade ||h|| W(D) Q(D)."""

    if self.w.size == 0:
      raise RuntimeError("Equalizer coefficients are not designed yet.")

    cascade = self.h_norm * np.convolve(self.w, self.q, mode="full")
    idx0 = int(self.w_index[0] + self.q_index[0])
    cascade_index = np.arange(cascade.size, dtype=int) + idx0
    return cascade_index, cascade.astype(self.dtype)

  def residualIsi(self) -> dict[str, float | complex]:
    """Measure center-tap error and residual ISI of the finite implementation."""

    idx, cascade = self.cascadeResponse()
    center_pos = np.flatnonzero(idx == 0)
    if center_pos.size != 1:
      raise RuntimeError("Cascade response does not contain a unique k = 0 sample.")

    center = cascade[center_pos[0]]
    mask = idx != 0
    return {
      "center": complex(center),
      "center_error": float(abs(center - 1.0)),
      "max_offcenter": float(np.max(np.abs(cascade[mask]))) if np.any(mask) else 0.0,
      "isi_energy": float(np.sum(np.abs(cascade[mask])**2)),
    }


@dataclass
class ZFE(BaseLinearEq):
  """
  Ideal symbol-spaced zero-forcing linear equalizer, Chapter 3.4.

    W_ZFE(D) = 1 / [||h|| Q(D)]

  The ideal stable inverse is generally infinite and two-sided. The class
  evaluates the Chapter 3.4 performance quantities from the ideal frequency
  response, then builds a finite centered FIR approximation with an IFFT.
  """

  n_taps: int = 129
  n_fft: int = 65536

  _omega_bins: Array = field(init=False, repr=False)
  _q_bins: Array = field(init=False, repr=False)
  _w_bins: Array = field(init=False, repr=False)

  def __post_init__(self) -> None:
    super().__post_init__()

    if self.n_taps <= 0 or self.n_taps % 2 == 0:
      raise ValueError(f"n_taps must be a positive odd integer, got {self.n_taps}.")
    if self.n_fft < self.n_taps:
      raise ValueError(f"n_fft must be >= n_taps, got {self.n_fft} < {self.n_taps}.")

    self.design()

  def _buildIdealResponse(self) -> None:
    self._omega_bins = 2.0 * np.pi * np.fft.fftfreq(self.n_fft)
    q_complex = self.qResponse(self._omega_bins)

    imag_peak = float(np.max(np.abs(np.imag(q_complex))))
    scale = max(1.0, float(np.max(np.abs(q_complex))))
    if imag_peak > 1e-9 * scale:
      raise ValueError("Q(exp(-j*omega)) must be real for an autocorrelation sequence; "
                       f"maximum imaginary residue is {imag_peak:.3e}.")

    self._q_bins = np.real(q_complex).astype(np.float64)
    q_min = float(np.min(self._q_bins))
    if q_min <= self.eps:
      raise ValueError("Ideal ZFE does not have finite noise enhancement because Q has a zero "
                       f"or nonpositive value on the frequency grid; min(Q) = {q_min:.6e}.")

    self._w_bins = (1.0 / (self.h_norm * self._q_bins)).astype(self.dtype)

  def design(self) -> Array:
    """Build a centered finite FIR approximation of the ideal two-sided ZFE."""

    self._buildIdealResponse()

    w_periodic = np.fft.ifft(self._w_bins)
    half = self.n_taps // 2
    self.w_index = np.arange(-half, half + 1, dtype=int)
    self.w = w_periodic[np.mod(self.w_index, self.n_fft)].astype(self.dtype)
    return self.w.copy()

  def frequencyResponse(self, centered: bool = True) -> tuple[Array, Array]:
    """Return the ideal ZFE W(exp(-j*omega)) on the design grid."""

    omega = self._omega_bins.copy()
    response = self._w_bins.copy()

    if centered:
      omega = np.fft.fftshift(omega)
      response = np.fft.fftshift(response)

    return omega, response

  def gammaInverse(self) -> float:
    """Chapter 3.4 noise-enhancement/SNR-loss factor gamma_ZFE^{-1}."""

    return float(np.mean(1.0 / self._q_bins))

  def gamma(self) -> float:
    """gamma_ZFE, satisfying SNR_ZFE = gamma_ZFE SNR_MFB."""

    return float(1.0 / self.gammaInverse())

  def idealCenterTap(self) -> float:
    """Ideal center tap w(0) = gamma_ZFE^{-1} / ||h||."""

    return float(self.gammaInverse() / self.h_norm)

  def outputNoiseVariance(self, noise_var: float) -> float:
    """Ideal ZFE output-noise variance."""

    if noise_var < 0.0:
      raise ValueError(f"noise_var must be >= 0, got {noise_var}.")
    return float(noise_var * self.gammaInverse() / self.h_norm**2)

  def outputNoiseGain(self) -> float:
    """sigma_ZFE^2 / noise_var for the matched-filter noise_var convention."""

    return float(self.gammaInverse() / self.h_norm**2)

  def snr(self, symbol_energy: float, noise_var: float) -> float:
    """Ideal ZFE SNR = gamma_ZFE * SNR_MFB."""

    return float(self.gamma() * self.mfbSnr(symbol_energy, noise_var))

  def lossDb(self) -> float:
    """SNR loss from the matched-filter bound in dB."""

    return float(10.0 * np.log10(self.gammaInverse()))

  def analyze(self, symbol_energy: float = 1.0, noise_var: float = 1.0) -> dict[str, float | bool]:
    """Return the main Chapter 3.1-3.4 channel/ZFE quantities."""

    mfb_snr = self.mfbSnr(symbol_energy, noise_var)
    zfe_snr = self.snr(symbol_energy, noise_var)

    return {
      "h_norm": float(self.h_norm),
      "q_min": float(np.min(self._q_bins)),
      "q_max": float(np.max(self._q_bins)),
      "is_nyquist": self.isNyquist(),
      "gamma_inverse": self.gammaInverse(),
      "gamma": self.gamma(),
      "ideal_center_tap": self.idealCenterTap(),
      "output_noise_gain": self.outputNoiseGain(),
      "output_noise_variance": self.outputNoiseVariance(noise_var),
      "mfb_snr": mfb_snr,
      "zfe_snr": zfe_snr,
      "zfe_loss_db": self.lossDb(),
    }


@dataclass
class MMSELE(BaseLinearEq):
  """MMSE-LE for the matched-filter sampled model.

  y[k] = ||h|| (q*x)[k] + n[k], with R_nn(D) = noise_var Q(D).
  W(D) = 1 / (||h|| * (Q(D) + 1 / snr_mfb)).

  ``snr_mfb`` is a *linear* SNR, not dB. Ideal metrics use the full frequency
  response. ``self.w`` is a truncated two-sided impulse response used only for
  optional finite-filter simulation; it is NOT a new finite-FIR MMSE optimum.
  """

  snr_mfb: float = 10.0
  n_taps: int = 257
  n_fft: int = 65536

  _omega_bins: np.ndarray = field(init=False, repr=False)
  _q_bins: np.ndarray = field(init=False, repr=False)
  _w_bins: np.ndarray = field(init=False, repr=False)

  def __post_init__(self) -> None:
    super().__post_init__()
    if self.snr_mfb <= 0.0:
      raise ValueError("snr_mfb must be positive (linear SNR).")
    if self.n_taps <= 0 or self.n_taps % 2 != 1 or self.n_fft < self.n_taps:
      raise ValueError("n_taps must be positive and odd; n_fft >= n_taps.")
    self.design()

  def design(self) -> np.ndarray:
    """Evaluate the ideal response and extract a centered FIR approximation."""
    self._omega_bins = 2.0 * np.pi * np.fft.fftfreq(self.n_fft)
    q_complex = self.qResponse(self._omega_bins)
    if np.max(np.abs(q_complex.imag)) > 1e-9:
      raise ValueError("Autocorrelation Q(exp(-j omega)) must be real.")
    self._q_bins = q_complex.real
    if self._q_bins.min() < -1e-10:
      raise ValueError("Autocorrelation power spectrum must be nonnegative.")

    self._w_bins = 1.0 / (self.h_norm * (self._q_bins + 1.0 / self.snr_mfb))
    w_periodic = np.fft.ifft(self._w_bins)
    half = self.n_taps // 2
    self.w_index = np.arange(-half, half + 1, dtype=int)
    self.w = w_periodic[np.mod(self.w_index, self.n_fft)].astype(self.dtype)
    return self.w.copy()

  def frequencyResponse(self, centered: bool = True) -> tuple[np.ndarray, np.ndarray]:
    omega = self._omega_bins.copy()
    response = self._w_bins.copy()
    if centered:
      return np.fft.fftshift(omega), np.fft.fftshift(response)
    return omega, response

  def idealCenterTap(self) -> float:
    """w[0] = (1/2pi) integral W(exp(-j omega)) d omega."""
    return float(np.mean(self._w_bins))

  def mse(self, symbol_energy: float, noise_var: float) -> float:
    """Total MMSE: symbol bias + residual ISI + filtered noise."""
    if symbol_energy <= 0.0 or noise_var <= 0.0:
      raise ValueError("symbol_energy and noise_var must be positive.")
    if not np.isclose(self.mfbSnr(symbol_energy, noise_var), self.snr_mfb, rtol=1e-9):
      raise ValueError("symbol_energy/noise_var is inconsistent with the design snr_mfb.")
    return float(symbol_energy * np.mean(1.0 / (1.0 + self.snr_mfb * self._q_bins)))

  def mseFromCenterTap(self, noise_var: float) -> float:
    return float(self.idealCenterTap() * noise_var / self.h_norm)

  def alpha(self) -> float:
    """Desired-symbol coefficient of the ideal equalized channel."""
    return float(np.mean(self.h_norm * self._w_bins * self._q_bins))

  def biasedSnr(self, symbol_energy: float, noise_var: float) -> float:
    return float(symbol_energy / self.mse(symbol_energy, noise_var))

  def unbiasedSnr(self, symbol_energy: float, noise_var: float) -> float:
    return float(self.biasedSnr(symbol_energy, noise_var) - 1.0)

  def lossDb(self, symbol_energy: float, noise_var: float) -> float:
    mfb_snr = self.mfbSnr(symbol_energy, noise_var)
    return float(10.0 * np.log10(mfb_snr / self.unbiasedSnr(symbol_energy, noise_var)))

  def errorComponents(self, symbol_energy: float, noise_var: float) -> dict[str, float]:
    """Parseval decomposition of total error, using ideal frequency responses."""
    if not np.isclose(self.mfbSnr(symbol_energy, noise_var), self.snr_mfb, rtol=1e-9):
      raise ValueError("symbol_energy/noise_var is inconsistent with the design snr_mfb.")
    g = self.h_norm * self._w_bins * self._q_bins
    alpha = float(np.mean(g))
    bias = float(symbol_energy * (1.0 - alpha)**2)
    isi = float(symbol_energy * np.mean(np.abs(g - alpha)**2))
    filtered_noise = float(noise_var * np.mean(self._q_bins * abs(self._w_bins)**2))
    return {
      "alpha": alpha,
      "bias_power": bias,
      "residual_isi_power": isi,
      "filtered_noise_power": filtered_noise,
      "sum": bias + isi + filtered_noise,
    }


__all__ = ['BaseLinearEq', 'ZFE', 'MMSELE']
