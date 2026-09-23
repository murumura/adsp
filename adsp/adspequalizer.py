from dataclasses import dataclass, field
import numpy as np
from typing import Optional, TypeAlias, Tuple
from . import adspfilter


Array: TypeAlias = np.ndarray

@dataclass
class BaseEqualizer:
  """Common state and utilities shared by Chapter 3 equalizers."""

  eps: float = 1e-12
  dtype: object = np.complex128

  def __post_init__(self) -> None:
    if self.eps <= 0.0:
      raise ValueError(f"eps must be > 0, got {self.eps}.")

  @staticmethod
  def responseFromTaps(taps: Array, tap_index: Array, normalized_frequency: Array) -> Array:
    """
    Evaluate A(D) = sum_k a(k) D^k on D = exp(-j*2*pi*f).

    normalized_frequency is in cycles/sample, so the Nyquist interval is
    -0.5 <= f <= 0.5.
    """

    taps_arr = np.asarray(taps).reshape(-1)
    idx_arr = np.asarray(tap_index, dtype=int).reshape(-1)
    freq_arr = np.asarray(normalized_frequency, dtype=np.float64).reshape(-1)

    if taps_arr.size != idx_arr.size:
      raise ValueError("taps and tap_index must have the same length.")

    phase = np.exp(-1j * 2.0 * np.pi * freq_arr[:, None] * idx_arr[None, :])
    return phase @ taps_arr


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
        raise ValueError(
          f"init_coef must have {self.n_taps} coefficients, got {coef.size}."
        )
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
      raise ValueError(
        f"q_zero_index must be in [0, {self.q.size - 1}], got {self.q_zero_index}."
      )

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
      raise ValueError(
        "Q(exp(-j*omega)) must be real for an autocorrelation sequence; "
        f"maximum imaginary residue is {imag_peak:.3e}."
      )

    self._q_bins = np.real(q_complex).astype(np.float64)
    q_min = float(np.min(self._q_bins))
    if q_min <= self.eps:
      raise ValueError(
        "Ideal ZFE does not have finite noise enhancement because Q has a zero "
        f"or nonpositive value on the frequency grid; min(Q) = {q_min:.6e}."
      )

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

@dataclass
class ErrorFeedbackLatticeRlsEq(BaseDfe):
  """
  Diniz Example 7.2 linear equalizer using Algorithm 7.3.

  Example 7.2:
    channel          : h(k) = 0.1 * 0.5^k, k = 0,...,8
    equalizer order  : 25
    equalizer taps   : 26
    lambda           : 0.99
    epsilon          : 0.1
    equalizer delay L: 18

  This is a linear equalizer, not a decision-feedback equalizer.
  Therefore n_fb must be zero.

  Notes:
    lrls_engine.w : lattice feedforward coefficients v_i(k)
    self.w        : equivalent direct-form FIR equalizer coefficients
  """

  n_ff: int = 26
  n_fb: int = 0
  lam: float = 0.99
  epsilon: float = 0.1
  eps: float = 1e-12
  verify_direct: bool = False

  lrls_engine: adspfilter.ErrorFeedbackLatticeRLS = field(init=False, repr=False)
  wb_prev: list[np.ndarray] = field(init=False, repr=False)

  post_y_hist: np.ndarray = field(init=False, repr=False)
  post_e_hist: np.ndarray = field(init=False, repr=False)
  w_hist: np.ndarray = field(init=False, repr=False)

  def __post_init__(self):
    super().__post_init__()

    if self.n_fb != 0:
      raise ValueError("Diniz Example 7.2 is a linear equalizer; n_fb must be 0.")

    if self.n_ff <= 0:
      raise ValueError("n_ff must be positive.")

    if self.init_coef is not None and np.any(np.abs(self.w) > self.eps):
      raise ValueError("Nonzero direct-form init_coef is not supported for the LRLS initialization.")

    self.lrls_engine = adspfilter.ErrorFeedbackLatticeRLS(
      filter_order=self.n_ff - 1,
      lam=self.lam,
      epsilon=self.epsilon,
      eps=self.eps,
    )

    self.resetStates()

  @property
  def v(self) -> np.ndarray:
    """Lattice feedforward coefficients v_i(k), Eq. (7.96)."""
    return np.asarray(np.real(self.lrls_engine.w), dtype=np.float64)

  def resetStates(self):
    """Reset the equalizer buffers, LRLS states, and direct-form reconstruction states."""
    self.resetBuffers()
    self.lrls_engine.resetState()

    self.w = np.zeros(self.n_ff, dtype=np.complex64)

    # wb_prev[i] = w_b(k-1, i), a vector with i coefficients.
    self.wb_prev = [np.zeros(i, dtype=np.float64) for i in range(self.n_ff + 1)]

  def latticeToDirect(self):
    """
    Convert the current lattice representation to the direct FIR coefficient vector.

    Predictor order updates:
      w_f(k,i+1) = [w_f(k,i); 0] - kappa_f(k,i) [w_b(k-1,i); -1]       (7.32)

      w_b(k,i+1) = [0; w_b(k-1,i)] - kappa_b(k,i) [-1; w_f(k,i)]       (7.28)

    Joint-process direct-form order update:
      w(k,i+1) = [w(k,i); 0] + v_i(k) [-w_b(k,i); 1]                    (7.63)
    """

    kappa_f = np.asarray(self.lrls_engine.kappa_f, dtype=np.float64)
    kappa_b = np.asarray(self.lrls_engine.kappa_b, dtype=np.float64)
    v = self.v

    wf_cur = [np.zeros(0, dtype=np.float64)]
    wb_cur = [np.zeros(0, dtype=np.float64)]
    w_direct = np.zeros(0, dtype=np.float64)

    for i in range(self.n_ff):
      # Eq. (7.63): direct joint-process coefficient vector of order i+1.
      basis_i = np.concatenate([-wb_cur[i], [1.0]])
      w_direct = np.concatenate([w_direct, [0.0]]) + v[i] * basis_i

      # Eqs. (7.32), (7.28): predictor coefficients required by the next section.
      wf_aug = np.concatenate([wf_cur[i], [0.0]])
      wb_aug = np.concatenate([self.wb_prev[i], [-1.0]])
      wf_next = wf_aug - kappa_f[i] * wb_aug

      wb_time_aug = np.concatenate([[0.0], self.wb_prev[i]])
      wf_aug_b = np.concatenate([[-1.0], wf_cur[i]])
      wb_next = wb_time_aug - kappa_b[i] * wf_aug_b

      wf_cur.append(wf_next)
      wb_cur.append(wb_next)

    self.wb_prev = wb_cur
    self.w = w_direct.astype(np.complex64)

  def __call__(
    self,
    u: np.ndarray,
    d: np.ndarray,
    train: bool = True,
    reset: bool = True,
  ):
    """
    Process a received sequence.

    Returned y/e follow the normal adaptive-filter convention:
      y(k) = output produced by coefficients available before sample k
      e(k) = d(k) - y(k)

    During training, post_y_hist/post_e_hist contain the a posteriori LRLS results.
    """

    u = np.asarray(u, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)

    if u.ndim != 1 or d.ndim != 1:
      raise ValueError("u and d must be one-dimensional.")

    if len(u) != len(d):
      raise ValueError(f"u and d must have equal length, got {len(u)} and {len(d)}.")

    if reset:
      self.resetStates()

    n_samples = len(u)

    y_hist = np.zeros(n_samples, dtype=np.float64)
    e_hist = np.zeros(n_samples, dtype=np.float64)
    post_y_hist = np.zeros(n_samples, dtype=np.float64)
    post_e_hist = np.zeros(n_samples, dtype=np.float64)
    w_hist = np.zeros((n_samples + 1, self.n_ff), dtype=np.float64)

    w_hist[0] = np.real(self.w)

    for k in range(n_samples):
      # BaseDfe supplies the direct-form delay-line regressor.
      xk = self.makeRegressor(u[k])

      # A priori output/error: coefficients learned through k-1.
      y_prior = float(np.real(np.vdot(self.w, xk)))
      e_prior = float(d[k] - y_prior)

      y_hist[k] = y_prior
      e_hist[k] = e_prior

      if train:
        # LRLS itself accepts the scalar process u(k), not a prebuilt regressor.
        result = self.lrls_engine.step(float(u[k]), float(d[k]))

        # Recover the equivalent direct-form FIR weights.
        self.latticeToDirect()

        y_post = float(np.real(np.vdot(self.w, xk)))
        e_post = float(d[k] - y_post)

        post_y_hist[k] = y_post
        post_e_hist[k] = e_post

        if self.verify_direct:
          lattice_y = float(np.real(result["y"]))
          lattice_e = float(np.real(result["e"]))

          if not np.isclose(y_post, lattice_y, rtol=1e-5, atol=1e-9):
            raise RuntimeError(f"Direct/lattice output mismatch at k={k}: {y_post} vs {lattice_y}")

          if not np.isclose(e_post, lattice_e, rtol=1e-5, atol=1e-9):
            raise RuntimeError(f"Direct/lattice error mismatch at k={k}: {e_post} vs {lattice_e}")

      else:
        post_y_hist[k] = y_prior
        post_e_hist[k] = e_prior

      w_hist[k + 1] = np.real(self.w)

    self.post_y_hist = post_y_hist
    self.post_e_hist = post_e_hist
    self.w_hist = w_hist

    return y_hist, e_hist