from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


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

  def whitenPair(
    self,
    x: np.ndarray,
    d: np.ndarray,
    normalize: bool = True
  ) -> Tuple[np.ndarray, np.ndarray, float]:
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

    scale = float(np.sqrt(np.mean(np.abs(x_white) ** 2)))

    if normalize:
      scale = max(scale, self.eps)
      x_white = x_white / scale
      d_white = d_white / scale

    return x_white, d_white, scale


@dataclass
class FirWhitener(BaseWhitener):
  """
  Fixed-coefficient FIR whitener.

  Use this when whitening coefficients are already known from:
    - noise PSD estimation,
    - spectral factorization,
    - offline channel/noise characterization,
    - a previous adaptation phase.

  Example:
    W(z) = 1 - 0.92 z^-1

    init_coef = [1.0, -0.92]
  """

  init_coef: np.ndarray = field(
    default_factory=lambda: np.array([1.0], dtype=np.complex64)
  )

  def __post_init__(self):
    self.setCoef(self.init_coef)


@dataclass
class ArWhitener(BaseWhitener):
  """
  AR prediction-error whitener using Yule-Walker coefficient estimation.

  Signal model:

    x[n] = a1*x[n-1] + ... + ap*x[n-p] + e[n]

  Whitening filter:

    W(z) = 1 - a1*z^-1 - ... - ap*z^-p
  """

  ar_order: int = 1
  init_ar_coef: Optional[np.ndarray] = None
  ridge: float = 1e-8

  # Estimated AR model
  ar_coef: np.ndarray = field(init=False)

  def __post_init__(self):
    if self.ar_order <= 0:
      raise ValueError("ar_order must be greater than zero.")

    if self.init_ar_coef is None:
      self.ar_coef = np.zeros(self.ar_order, dtype=np.complex64)
    else:
      ar_coef = np.asarray(self.init_ar_coef, dtype=np.complex64).reshape(-1)

      if len(ar_coef) != self.ar_order:
        raise ValueError(
          "init_ar_coef length must equal ar_order. "
          f"Got {len(ar_coef)} coefficients for ar_order={self.ar_order}."
        )

      self.ar_coef = ar_coef.copy()

    self.setArCoef(self.ar_coef)

  def setArCoef(self, ar_coef: np.ndarray):
    """
    Set AR coefficients and construct the prediction-error filter.

    Given:
      x[n] = a1*x[n-1] + ... + ap*x[n-p] + e[n]

    construct:
      e[n] = x[n] - a1*x[n-1] - ... - ap*x[n-p]
    """
    ar_coef = np.asarray(ar_coef, dtype=np.complex64).reshape(-1)

    if len(ar_coef) != self.ar_order:
      raise ValueError(
        "ar_coef length must equal ar_order. "
        f"Got {len(ar_coef)} coefficients for ar_order={self.ar_order}."
      )

    self.ar_coef = ar_coef.copy()

    whitening_coef = np.concatenate([np.array([1.0], dtype=np.complex64), -self.ar_coef])

    self.setCoef(whitening_coef)

  def estimateArCoef(self, x: np.ndarray) -> np.ndarray:
    """
    Estimate AR coefficients with complex-safe Yule-Walker equations.

    Defines:
      r[l] = E{x[n] * conj(x[n-l])}

    and solves:
      R * a = r_vec
    """
    x = np.asarray(x, dtype=np.complex64).reshape(-1)

    if len(x) <= self.ar_order:
      raise ValueError(
        "Input length must be greater than ar_order. "
        f"len(x)={len(x)}, ar_order={self.ar_order}."
      )

    x0 = x - np.mean(x)
    n = len(x0)

    r = np.zeros(self.ar_order + 1, dtype=np.complex128)

    for lag in range(self.ar_order + 1):
      if lag == 0:
        r[lag] = np.vdot(x0, x0) / n
      else:
        r[lag] = np.vdot(x0[:-lag], x0[lag:]) / n

    # r[0] is theoretically real and nonnegative.
    r[0] = np.real(r[0])

    R = np.zeros((self.ar_order, self.ar_order), dtype=np.complex128)

    for row in range(self.ar_order):
      for col in range(self.ar_order):
        lag = row - col

        if lag >= 0:
          R[row, col] = r[lag]
        else:
          R[row, col] = np.conj(r[-lag])

    regularization = self.ridge * max(float(np.real(r[0])), self.eps)
    R += regularization * np.eye(self.ar_order, dtype=np.complex128)

    rhs = r[1:self.ar_order + 1]

    ar_coef = np.linalg.solve(R, rhs)

    return ar_coef.astype(np.complex64)

  def fit(self, x: np.ndarray) -> np.ndarray:
    """
    Estimate AR coefficients from x and update the whitening filter.

    Returns:
      Estimated AR coefficient vector [a1, ..., ap].
    """
    ar_coef = self.estimateArCoef(x)
    self.setArCoef(ar_coef)

    return self.ar_coef.copy()

  def fitTransform(
    self,
    x: np.ndarray,
    normalize: bool = False
  ) -> Tuple[np.ndarray, float]:
    """
    Estimate AR coefficients from x, then whiten x.

    Returns:
      x_white
      scale

    If normalize=False, scale is returned but not applied.
    """
    self.fit(x)

    x_white = self(x, reset=True)

    scale = float(np.sqrt(np.mean(np.abs(x_white) ** 2)))

    if normalize:
      scale = max(scale, self.eps)
      x_white = x_white / scale

    return x_white, scale

  def fitWhitenPair(
    self,
    x: np.ndarray,
    d: np.ndarray,
    normalize: bool = True
  ) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Fit the AR model using x, then whiten x and d with the same filter.

    Use this for system identification:

      d[n] = h[n] * x[n]

    where preserving the same target system h[n] matters.
    """
    self.fit(x)

    return self.whitenPair(
      x=x,
      d=d,
      normalize=normalize
    )

@dataclass
class SpectralWhitener:
  """
  Block-domain spectral whitener.

  For each frame:

    X[k] = FFT{x[n]}
    Y[k] = G[k] * X[k]
    y[n] = IFFT{Y[k]}

  In PSD mode:

    G[k] = 1 / sqrt(S_hat[k] + eps)

  In magnitude mode, which reproduces the behavior of the
  original spectral_whiten() function:

    G[k] = 1 / (A_hat[k] + eps)
  """

  fft_len: Optional[int] = None
  smooth_len: int = 17
  eps: float = 1e-6
  max_gain: Optional[float] = 32.0

  remove_mean: bool = True
  normalize_output: bool = False

  # True  : use smoothed |X[k]|^2 and 1 / sqrt(PSD).
  # False : use smoothed |X[k]| and 1 / magnitude.
  #         This matches the original function more closely.
  smooth_psd: bool = True

  # "edge" avoids artificial low envelope values at DC/Nyquist.
  # "zero" reproduces np.convolve(..., mode="same") behavior.
  smooth_boundary: str = "edge"

  # Learned / configured spectral model
  is_complex: Optional[bool] = field(default=None, init=False)
  spectral_model: Optional[np.ndarray] = field(default=None, init=False)
  gain: Optional[np.ndarray] = field(default=None, init=False)

  # Debug / analysis state
  last_input_mean: complex = field(default=0.0 + 0.0j, init=False)
  last_output_rms: float = field(default=1.0, init=False)

  def __post_init__(self):
    if self.smooth_len <= 0:
      raise ValueError("smooth_len must be greater than zero.")

    if self.smooth_len % 2 == 0:
      raise ValueError("smooth_len must be odd.")

    if self.eps <= 0.0:
      raise ValueError("eps must be greater than zero.")

    if self.max_gain is not None and self.max_gain <= 0.0:
      raise ValueError("max_gain must be positive or None.")

    if self.smooth_boundary not in ("edge", "zero"):
      raise ValueError(
        "smooth_boundary must be either 'edge' or 'zero'."
      )

  def resetModel(self):
    """Forget the fitted PSD/magnitude model and whitening gain."""
    self.spectral_model = None
    self.gain = None

  def validateFrame(self, x: np.ndarray):
    """Validate frame length and real/complex mode."""
    x = np.asarray(x)

    if x.ndim != 1:
      raise ValueError("Input x must be a one-dimensional array.")

    if len(x) == 0:
      raise ValueError("Input x must not be empty.")

    if self.fft_len is None:
      self.fft_len = len(x)
    elif len(x) != self.fft_len:
      raise ValueError(
        "Input frame length must equal fft_len. "
        f"Got len(x)={len(x)}, fft_len={self.fft_len}."
      )

    frame_is_complex = bool(np.iscomplexobj(x))

    if self.is_complex is None:
      self.is_complex = frame_is_complex
    elif self.is_complex != frame_is_complex:
      raise ValueError(
        "Cannot mix real and complex frames in one SpectralWhitener "
        "instance. Create a separate instance for each mode."
      )

  def prepareInput(self, x: np.ndarray) -> np.ndarray:
    """Convert to a suitable dtype and optionally remove DC."""
    self.validateFrame(x)

    if self.is_complex:
      x = np.asarray(x, dtype=np.complex64)
    else:
      x = np.asarray(x, dtype=np.float32)

    if self.remove_mean:
      self.last_input_mean = np.mean(x)
      x = x - self.last_input_mean
    else:
      self.last_input_mean = 0.0 + 0.0j

    return x

  def forwardFFT(self, x: np.ndarray) -> np.ndarray:
    """FFT for complex input, RFFT for real input."""
    if self.is_complex:
      return np.fft.fft(x, n=self.fft_len)

    return np.fft.rfft(x, n=self.fft_len)

  def inverseFFT(self, spectrum: np.ndarray) -> np.ndarray:
    """IFFT for complex input, IRFFT for real input."""
    if self.is_complex:
      return np.fft.ifft(spectrum, n=self.fft_len)

    return np.fft.irfft(spectrum, n=self.fft_len)

  def smoothSpectrum(self, value: np.ndarray) -> np.ndarray:
    """Smooth a nonnegative spectral estimate across frequency bins."""
    value = np.asarray(value, dtype=np.float64)

    if self.smooth_len == 1:
      return value.copy()

    kernel = np.ones(self.smooth_len, dtype=np.float64)
    kernel /= self.smooth_len

    # smooth by mean-filter
    if self.smooth_boundary == "zero":
      return np.convolve(value, kernel, mode="same")

    pad_len = self.smooth_len // 2
    padded = np.pad(
      value,
      pad_width=(pad_len, pad_len),
      mode="edge"
    )

    return np.convolve(padded, kernel, mode="valid")

  def estimateSpectralModel(
    self,
    spectrum: np.ndarray
  ) -> np.ndarray:
    """
    Estimate either a smoothed PSD or a smoothed magnitude envelope.
    """
    mag = np.abs(spectrum)

    if self.smooth_psd:
      raw_model = mag ** 2
    else:
      raw_model = mag

    return self.smoothSpectrum(raw_model)

  def estimateGain(
    self,
    spectral_model: np.ndarray
  ) -> np.ndarray:
    """Convert a PSD/magnitude estimate into whitening gain bins."""
    spectral_model = np.asarray(
      spectral_model,
      dtype=np.float64
    )

    if self.smooth_psd:
      gain = 1.0 / np.sqrt(
        np.maximum(spectral_model, 0.0) + self.eps
      )
    else:
      gain = 1.0 / (
        np.maximum(spectral_model, 0.0) + self.eps
      )

    if self.max_gain is not None:
      gain = np.minimum(gain, self.max_gain)

    return gain.astype(np.float32)

  def fit(self, x_ref: np.ndarray):
    """
    Estimate a fixed whitening spectrum from a reference frame.

    In a receiver, x_ref should preferably be:
      - a noise-only calibration frame,
      - an idle / training period,
      - or a long stationary input segment.

    Do not normally re-fit independently on every data frame when
    preserving a fixed receiver transfer function matters.
    """
    x0 = self.prepareInput(x_ref)
    spectrum = self.forwardFFT(x0)

    self.spectral_model = self.estimateSpectralModel(spectrum)

    self.gain = self.estimateGain(self.spectral_model)

    return self

  def updateModel(
    self,
    x_ref: np.ndarray,
    alpha: float = 0.98
  ):
    """
    Exponentially update the spectral model.

    model_new = alpha * model_old + (1 - alpha) * model_current

    alpha close to 1.0 means slow, stable adaptation.
    """
    if not 0.0 <= alpha < 1.0:
      raise ValueError("alpha must satisfy 0 <= alpha < 1.")

    x0 = self.prepareInput(x_ref)
    spectrum = self.forwardFFT(x0)

    current_model = self.estimateSpectralModel(
      spectrum
    )

    if self.spectral_model is None:
      self.spectral_model = current_model
    else:
      self.spectral_model = (
        alpha * self.spectral_model
        + (1.0 - alpha) * current_model
      )

    self.gain = self.estimateGain(self.spectral_model)

    return self

  def getGain(self) -> np.ndarray:
    """Return a copy of the current per-frequency-bin gain."""
    if self.gain is None:
      raise RuntimeError(
        "No whitening gain is available. Call fit() or updateModel() first."
      )

    return self.gain.copy()

  def transform(
    self,
    x: np.ndarray,
    use_fitted_gain: bool = True
  ) -> np.ndarray:
    """
    Whiten one frame.

    use_fitted_gain=True:
      Use a previously estimated fixed gain.

    use_fitted_gain=False:
      Estimate a gain from this same frame and whiten it immediately.
      This is useful for offline visualization, but not normally the
      preferred receiver implementation.
    """
    x0 = self.prepareInput(x)
    spectrum = self.forwardFFT(x0)

    if use_fitted_gain:
      if self.gain is None:
        raise RuntimeError(
          "No fitted gain exists. Call fit(), updateModel(), or set "
          "use_fitted_gain=False."
        )

      gain = self.gain
    else:
      local_model = self.estimateSpectralModel(spectrum)
      gain = self.estimateGain(local_model)

    white_spectrum = spectrum * gain
    y = self.inverseFFT(white_spectrum)

    self.last_output_rms = float(
      np.sqrt(np.mean(np.abs(y) ** 2))
    )

    if self.normalize_output:
      y = y / max(self.last_output_rms, self.eps)

    if self.is_complex:
      return y.astype(np.complex64)

    return np.real(y).astype(np.float32)

  def fitTransform(
    self,
    x: np.ndarray
  ) -> np.ndarray:
    """Fit from x, then whiten x using the fitted gain."""
    self.fit(x)

    return self.transform(x, use_fitted_gain=True)

  def __call__(self, x: np.ndarray, use_fitted_gain: bool = True) -> np.ndarray:
    """Convenience wrapper for transform()."""
    return self.transform(x, use_fitted_gain=use_fitted_gain)