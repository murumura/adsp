"""Cioffi Ch. 3.6 analytical MMSE-DFE and ZF-DFE.

These classes model *known-channel* DFE design, not LMS/RLS adaptation.
All ``w`` arrays contain physical LTI taps W(D) = sum_m w[m] D**m, unlike
BaseDfe's adaptive inner-product coefficients (which are conjugated).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
from adsp.core import Array
from adsp.equalization.linear import BaseLinearEq


@dataclass
class DfeOutput:
  """One receiver run; arrays are aligned to the requested symbol indices.

  feedforward: Z[k] before decision feedback and bias removal.
  decision_input: biased Z'[k], or unbiased Z'_U[k], per receiver setting.
  slicer_input: always constellation-scaled; equals decision_input if unbiased.
  decisions: hard nearest-neighbor decisions (even during teacher forcing).
  feedback_used: symbols actually inserted in the past-decision delay line.
  """

  feedforward: Array
  decision_input: Array
  slicer_input: Array
  decisions: Array
  feedback_used: Array


@dataclass
class BaseCanonicalDfe(BaseLinearEq):
  """Ideal infinite-length MMSE/ZF-DFE + explicit finite filter execution.

  Matched-filter sampled input model:
    Y(D) = ||h|| Q(D) X(D) + N(D)
    R_nn(D) = noise_var Q(D)
    SNR_MFB = E_x ||h||**2 / noise_var (linear, not dB).

  Canonical factorization:
    Q(D) + regularization = factor_scale * G(D) G*(D**(-*))

  The scalar ``factor_scale`` is gamma_0 (MMSE) or eta_0 (ZF).
  G is monic and minimum-phase; B(D) = G(D) and the actual feedback
  transfer function is 1-B(D), with no zero-delay feedback term.

  Spectral factorization uses the causal complex cepstrum.  This avoids
  ill-conditioned high-order polynomial root finding, and its spectral
  reconstruction is exact on the sampled FFT grid.  Finite feedback taps
  and finite feedforward taps are only approximations to the ideal filters.
  """

  snr_mfb: float = 10.0
  n_taps: int = 129
  n_fft: int = 65536
  n_fb: Optional[int] = None
  constellation: Array = field(default_factory=lambda: np.array([-1.0, 1.0]))
  unbiased: bool = True
  criterion: Literal['mmse', 'zf'] = 'mmse'

  factor_scale: float = field(init=False)
  canonical_factor: Array = field(init=False, repr=False)
  feedback_polynomial: Array = field(init=False, repr=False)
  _omega_bins: Array = field(init=False, repr=False)
  _q_bins: Array = field(init=False, repr=False)
  _g_bins: Array = field(init=False, repr=False)
  _w_bins: Array = field(init=False, repr=False)
  _fb_order: int = field(init=False, repr=False)

  def __post_init__(self) -> None:
    super().__post_init__()
    if self.snr_mfb <= 0.0 or not np.isfinite(self.snr_mfb):
      raise ValueError('snr_mfb must be a positive, finite linear SNR')
    if self.n_taps <= 0 or self.n_fft < 2 * self.n_taps or self.n_fft % 2:
      raise ValueError('n_taps must be positive; n_fft must be even and >= 2*n_taps')
    if self.criterion not in ('mmse', 'zf'):
      raise ValueError('criterion must be mmse or zf')
    if self.n_fb is not None and self.n_fb < 0:
      raise ValueError('n_fb must be nonnegative or None')
    self.constellation = np.asarray(self.constellation, dtype=self.dtype).reshape(-1)
    if not self.constellation.size or not np.all(np.isfinite(self.constellation)):
      raise ValueError('constellation must contain finite complex symbols')
    self.design()

  def design(self) -> Array:
    """Design the ideal spectral response, finite FF, and causal feedback taps."""
    self._omega_bins = 2 * np.pi * np.fft.fftfreq(self.n_fft)
    # FFT the indexed Laurent coefficients instead of allocating an n_fft x len(q)
    # Vandermonde matrix; this is important for long RRC/polyphase pulses.
    if np.max(np.abs(self.q_index)) >= self.n_fft // 2:
      raise ValueError('n_fft is too small for the supplied autocorrelation lags')
    q_periodic = np.zeros(self.n_fft, dtype=np.complex128)
    np.add.at(q_periodic, np.mod(self.q_index, self.n_fft), self.q)
    q_complex = np.fft.fft(q_periodic)
    if np.max(np.abs(q_complex.imag)) > 1e-9 * max(1.0, np.max(np.abs(q_complex))):
      raise ValueError('Q(D) must be Hermitian: Q(exp(-j*omega)) must be real')
    self._q_bins = np.asarray(q_complex.real, dtype=np.float64)
    if self._q_bins.min() < -1e-9:
      raise ValueError('Q(D) is not a nonnegative power spectrum')
    if self.criterion == 'zf' and self._q_bins.min() <= self.eps:
      raise ValueError('ZF-DFE requires a positive Q on the design grid; spectral null detected')

    # An exact unit-circle zero may fall between FFT bins.  For low-degree
    # autocorrelations, inspect the Laurent-polynomial roots as an additional
    # guard against an unstable, non-BIBO zero-forcing inverse.
    if self.criterion == 'zf':
      active = np.flatnonzero(np.abs(self.q) > 1e-12 * max(1.0, np.max(np.abs(self.q))))
      degree = int(np.max(np.abs(self.q_index[active]))) if active.size else 0
      if 0 < degree <= 8:
        polynomial = np.zeros(2 * degree + 1, dtype=np.complex128)
        supported = np.abs(self.q_index) <= degree
        np.add.at(polynomial, self.q_index[supported] + degree, self.q[supported])
        zeros_in_d = np.roots(polynomial[::-1])
        if np.any(np.abs(np.abs(zeros_in_d) - 1.0) < 1e-7):
          raise ValueError('ZF-DFE channel has a unit-circle spectral null; stable inverse unavailable')

    # Only the MMSE design adds noise regularization to the channel spectrum.
    regularization = 1.0 / self.snr_mfb if self.criterion == 'mmse' else 0.0
    spectrum = self._q_bins + regularization
    if np.any(spectrum <= 0):
      raise ValueError('canonical spectral factorization requires a positive spectrum')

    # For D = exp(-j*omega), log(G(D)) has only strictly positive delay powers.
    # log(spectrum) = log(factor_scale) + log G(D) + log G*(D**(-*)).
    log_spectrum = np.log(spectrum)
    self.factor_scale = float(np.exp(np.mean(log_spectrum)))
    complex_cepstrum = np.fft.ifft(log_spectrum)
    causal_cepstrum = np.zeros(self.n_fft, dtype=np.complex128)
    causal_cepstrum[1:self.n_fft // 2] = complex_cepstrum[1:self.n_fft // 2]
    causal_cepstrum[self.n_fft // 2] = 0.5 * complex_cepstrum[self.n_fft // 2]
    self._g_bins = np.exp(np.fft.fft(causal_cepstrum))

    # The degree of a finite polynomial Q equals the degree of its FIR factor.
    # Ignore padded/near-zero autocorrelation coefficients at the boundaries.
    active = np.flatnonzero(np.abs(self.q) > 1e-11 * max(1.0, np.max(abs(self.q))))
    self._fb_order = int(np.max(np.abs(self.q_index[active]))) if active.size else 0
    if self._fb_order >= self.n_fft // 2:
      raise ValueError('n_fft is too small for the channel autocorrelation order')
    full_factor = np.fft.ifft(self._g_bins)[:self._fb_order + 1]
    if abs(full_factor[0] - 1.0) > 1e-6:
      raise ValueError('failed to obtain monic canonical factor: enlarge n_fft')
    full_factor[0] = 1.0
    self.canonical_factor = np.asarray(full_factor, dtype=self.dtype)
    n_feedback = self._fb_order if self.n_fb is None else self.n_fb
    if n_feedback >= self.n_fft // 2:
      raise ValueError('n_fb must be smaller than n_fft/2')
    self.feedback_polynomial = np.zeros(n_feedback + 1, dtype=self.dtype)
    overlap = min(n_feedback, self._fb_order) + 1
    self.feedback_polynomial[:overlap] = self.canonical_factor[:overlap]

    # W(D) = 1/(||h|| * factor_scale * G*(D**(-*))) is anticausal.
    # Thus the finite approximation uses nonpositive *physical* FF lags.
    self._w_bins = 1.0 / (self.h_norm * self.factor_scale * np.conj(self._g_bins))
    self.w_index = np.arange(1 - self.n_taps, 1, dtype=int)
    periodic = np.fft.ifft(self._w_bins)
    self.w = periodic[np.mod(self.w_index, self.n_fft)].astype(self.dtype)
    return self.w.copy()

  @property
  def gamma0(self) -> float:
    """MMSE canonical spectral-factorization constant gamma_0."""
    if self.criterion != 'mmse':
      raise ValueError('gamma0 is defined only for MMSE-DFE; use eta0 for ZF-DFE')
    return self.factor_scale

  @property
  def eta0(self) -> float:
    """ZF canonical spectral-factorization constant eta_0."""
    if self.criterion != 'zf':
      raise ValueError('eta0 is defined only for ZF-DFE; use gamma0 for MMSE-DFE')
    return self.factor_scale

  def alpha(self) -> float:
    """Ideal desired-symbol gain at the *biased* DFE decision input."""
    if self.criterion == 'zf':
      return 1.0
    mmse_snr = self.factor_scale * self.snr_mfb
    return float(1.0 - 1.0 / mmse_snr)

  def unbiasedFeedbackPolynomial(self) -> Array:
    """G_U(D)=1+[G(D)-1]/alpha; applies to the actual fitted feedback taps."""
    value = self.feedback_polynomial.copy()
    if self.alpha() <= 0.0:
      raise ValueError('unbiased MMSE-DFE has a nonpositive desired-symbol gain')
    value[1:] /= self.alpha()
    return value

  def actualFeedbackTaps(self, unbiased: Optional[bool] = None) -> Array:
    """Coefficients of 1-G(D), or 1-G_U(D), excluding the zero zero-lag tap."""
    use_unbiased = self.unbiased if unbiased is None else bool(unbiased)
    polynomial = self.unbiasedFeedbackPolynomial() if use_unbiased else self.feedback_polynomial
    return -polynomial[1:].copy()

  def idealMse(self, symbol_energy: float = 1.0) -> float:
    """Correct-decision optimum MSE E_x/(factor_scale*SNR_MFB)."""
    if symbol_energy <= 0:
      raise ValueError('symbol_energy must be positive')
    return float(symbol_energy / (self.factor_scale * self.snr_mfb))

  def biasedSnr(self) -> float:
    """MSE-based SNR, not the unbiased detector SNR for MMSE-DFE."""
    return float(self.factor_scale * self.snr_mfb)

  def unbiasedSnr(self) -> float:
    """Unbiased detection SNR: gamma_0*SNR_MFB-1 or eta_0*SNR_MFB."""
    return float(self.biasedSnr() - 1.0 if self.criterion == 'mmse' else self.biasedSnr())

  def idealOutputNoiseVariance(self, symbol_energy: float = 1.0) -> float:
    """Only the ZF output-error variance is identical to filtered-noise power."""
    if self.criterion != 'zf':
      raise ValueError('MMSE-DFE error contains symbol-dependent terms; use idealMse()')
    return self.idealMse(symbol_energy)

  def frequencyResponse(self, centered: bool = True) -> tuple[Array, Array]:
    """Return ideal infinite-length feedforward W(D) on the sampled grid."""
    if centered:
      return np.fft.fftshift(self._omega_bins.copy()), np.fft.fftshift(self._w_bins.copy())
    return self._omega_bins.copy(), self._w_bins.copy()

  def feedforwardSamples(self, received: Array, n_symbols: Optional[int] = None) -> Array:
    """Filter matched-filter output Y[k] using finite indexed feedforward taps."""
    received = np.asarray(received, dtype=self.dtype).reshape(-1)
    if n_symbols is not None and n_symbols != len(received):
      raise ValueError('symbol-spaced input length must equal n_symbols')
    return self._filterIndexed(received, self.w, self.w_index, self.dtype)

  def apply(
    self,
    received: Array,
    *,
    n_symbols: Optional[int] = None,
    known_symbols: Optional[Array] = None,
    training_samples: int = 0,
    ideal_feedback: bool = False,
    unbiased: Optional[bool] = None,
  ) -> DfeOutput:
    """Run the finite DFE with chronological, decision-directed feedback.

    known_symbols: aligned desired symbols; used only for feedback when
      k < training_samples, or at every k if ideal_feedback=True.
    training_samples: number of initial symbols using teacher-forced feedback.
    ideal_feedback: insert known past decisions for the entire record.
    unbiased: place 1/alpha before the adder and use 1-G_U(D); when False,
      leave the adder biased and scale only the slicer decision variable.

    No coefficient adaptation occurs.  The feedforward part may be noncausal;
    accessing future samples requires practical delay/lookahead and guarding.
    """
    z = np.asarray(self.feedforwardSamples(received, n_symbols=n_symbols), dtype=self.dtype)
    n_symbols = len(z)
    if not 0 <= training_samples <= n_symbols:
      raise ValueError('training_samples must lie in [0, n_symbols]')
    if (training_samples or ideal_feedback) and known_symbols is None:
      raise ValueError('known_symbols required for training or ideal_feedback')
    known = None if known_symbols is None else np.asarray(known_symbols, dtype=self.dtype).reshape(-1)
    if known is not None and len(known) != n_symbols:
      raise ValueError('known_symbols must have one element per output symbol')

    use_unbiased = self.unbiased if unbiased is None else bool(unbiased)
    scale = self.alpha()
    feedback = self.actualFeedbackTaps(unbiased=use_unbiased)
    history = np.zeros(len(feedback), dtype=self.dtype)
    decision_input = np.empty(n_symbols, dtype=self.dtype)
    slicer_input = np.empty(n_symbols, dtype=self.dtype)
    decisions = np.empty(n_symbols, dtype=self.dtype)
    feedback_used = np.empty(n_symbols, dtype=self.dtype)

    for k in range(n_symbols):
      # In the unbiased realization, the 1/alpha multiplier is BEFORE feedback.
      feedforward_value = z[k] / scale if use_unbiased else z[k]
      decision_input[k] = feedforward_value + np.dot(feedback, history)
      slicer_input[k] = decision_input[k] if use_unbiased else decision_input[k] / scale
      decisions[k] = self.constellation[np.argmin(abs(self.constellation - slicer_input[k]))]

      # Teacher forcing and ideal feedback modify the *past decisions*, not W.
      use_known = known is not None and (ideal_feedback or k < training_samples)
      feedback_used[k] = known[k] if use_known else decisions[k]
      if history.size:
        history[1:] = history[:-1]
        history[0] = feedback_used[k]

    return DfeOutput(z, decision_input, slicer_input, decisions, feedback_used)

  def __call__(self, received: Array, **kwargs) -> DfeOutput:
    return self.apply(received, **kwargs)


@dataclass
class MMSEDFE(BaseCanonicalDfe):
  """Cioffi 3.6.1–3.6.2, 3.6.4: canonical MMSE-DFE, biased/unbiased output."""
  criterion: Literal['mmse'] = 'mmse'


@dataclass
class ZFDFE(BaseCanonicalDfe):
  """Cioffi 3.6.3–3.6.4: canonical ZF-DFE with unit desired gain."""
  criterion: Literal['zf'] = 'zf'


__all__ = ['DfeOutput', 'BaseCanonicalDfe', 'MMSEDFE', 'ZFDFE']
