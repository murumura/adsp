"""filters / quantized rls algorithms and helpers."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Literal
from adsp.core import Array
from adsp import quantization as adspquant
from adsp.filters.base import BaseLMS
from adsp.core import _prepad


@dataclass
class QuantizedRLSAlt(BaseLMS):
  """
  Quantized alternative complex RLS.

  Exact-form backbone:
    e(k)   = d(k) - w^H(k-1)x(k)
    psi(k) = S_D(k-1)x(k)
    S_D(k) = (1/lambda)[S_D(k-1) - psi(k)psi^H(k)/(lambda + psi^H(k)x(k))]
    w(k)   = w(k-1) + conj(e(k)) S_D(k)x(k)

  Quantized form:
    quantizes y, e, psi, denominator, S_D, gain, update, and w.
  """

  mu: Optional[float] = None
  lam: float = 0.99
  delta: float = 0.5
  eps: float = 1e-12

  q_total_bits: int = 16
  q_frac_bits: int = 12
  q_inner_mode: Literal["after_add", "after_product"] = "after_add"
  q_store_inputs: bool = False
  enforce_hermitian: bool = True
  gain_clip: Optional[float] = None
  update_clip: Optional[float] = None
  diagonal_loading: float = 0.0

  s_d: Array = field(init=False)
  q: adspquant.FixedQuantizer = field(init=False)

  def __post_init__(self) -> None:
    super().__post_init__()

    self.q = adspquant.FixedQuantizer(
      total_bits=self.q_total_bits,
      frac_bits=self.q_frac_bits,
      signed=True,
    )

    init_val = 1.0 / self.delta
    self.w = self.q(np.zeros((self.n_coef, ), dtype=np.complex64))
    self.s_d = self.q(init_val * np.eye(self.n_coef, dtype=np.complex64))

  def resetState(self) -> None:
    init_val = 1.0 / self.delta
    self.w = self.q(np.zeros((self.n_coef, ), dtype=np.complex64))
    self.s_d = self.q(init_val * np.eye(self.n_coef, dtype=np.complex64))

  def clip(self, x: Array, clip_value: Optional[float]) -> Array:
    if clip_value is None:
      return x
    x_arr = np.asarray(x)
    x_real = np.clip(np.real(x_arr), -clip_value, clip_value)
    x_imag = np.clip(np.imag(x_arr), -clip_value, clip_value)
    return (x_real + 1j * x_imag).astype(np.complex64)

  def makeRegressor(self, x_pad: Array, k_idx: int) -> Array:
    x_k = x_pad[k_idx:k_idx + self.n_coef][::-1].astype(np.complex64)
    if self.q_store_inputs:
      x_k = self.q(x_k)
    return x_k

  def __call__(self, x: Array, d: Array, train: bool = True) -> Dict[str, Array]:
    x_arr = np.asarray(x)
    d_arr = np.asarray(d)

    n_samples = x_arr.shape[0]
    x_pad = _prepad(x_arr, self.n_coef)

    w = self.q(self.w.copy())
    s_d = self.q(self.s_d.copy())

    y_hist = np.empty((n_samples, ), dtype=np.complex64)
    e_hist = np.empty((n_samples, ), dtype=np.complex64)
    eps_hist = np.empty((n_samples, ), dtype=np.complex64)
    psi_hist = np.empty((n_samples, self.n_coef), dtype=np.complex64)
    denom_hist = np.empty((n_samples, ), dtype=np.float32)
    w_hist = np.empty((n_samples + 1, self.n_coef), dtype=np.complex64)
    w_hist[0] = w

    for k_idx in range(n_samples):
      x_k = self.makeRegressor(x_pad, k_idx)

      y_q = adspquant.qdot(w, x_k, self.q, conj_a=True, mode=self.q_inner_mode)
      d_q = np.complex64(self.q(np.array(d_arr[k_idx], dtype=np.complex64))[()])
      e_q = np.complex64(self.q(d_q - y_q))

      eps_q = e_q
      psi_q = np.zeros((self.n_coef, ), dtype=np.complex64)
      denom_q = np.float32(0.0)

      if train:
        psi_q = adspquant.qmatVec(s_d, x_k, self.q, mode=self.q_inner_mode)
        psi_q = self.q(psi_q)

        psi_h_x_q = adspquant.qdot(psi_q, x_k, self.q, conj_a=True, mode=self.q_inner_mode)
        denom_val = float(np.real(psi_h_x_q) + self.lam)
        denom_val = max(denom_val, self.eps)
        denom_q = np.float32(self.q.quantizeReal(np.array(denom_val))[()])

        if abs(denom_q) < self.eps:
          denom_q = np.float32(self.eps)

        outer_q = adspquant.qouter(psi_q, np.conj(psi_q), self.q)
        corr_q = self.q(outer_q / denom_q)

        s_d = self.q((s_d - corr_q) / self.lam)

        if self.diagonal_loading > 0.0:
          s_d = self.q(s_d + self.diagonal_loading * np.eye(self.n_coef, dtype=np.complex64))

        if self.enforce_hermitian:
          s_d = 0.5 * (s_d + np.conj(s_d.T))
          s_d = self.q(s_d)

        gain_q = adspquant.qmatVec(s_d, x_k, self.q, mode=self.q_inner_mode)
        gain_q = self.q(gain_q)
        gain_q = self.clip(gain_q, self.gain_clip)
        gain_q = self.q(gain_q)

        update_q = self.q(np.conj(e_q) * gain_q)
        update_q = self.clip(update_q, self.update_clip)
        update_q = self.q(update_q)

        w = self.q(w + update_q)

        y_post_q = adspquant.qdot(w, x_k, self.q, conj_a=True, mode=self.q_inner_mode)
        eps_q = np.complex64(self.q(d_q - y_post_q))

      y_hist[k_idx] = y_q
      e_hist[k_idx] = e_q
      eps_hist[k_idx] = eps_q
      psi_hist[k_idx] = psi_q
      denom_hist[k_idx] = denom_q
      w_hist[k_idx + 1] = w

    if train:
      self.w = w
      self.s_d = s_d

    return {
      "y": y_hist,
      "e": e_hist,  # a priori error
      "eps": eps_hist,  # a posteriori error
      "psi_hist": psi_hist,
      "denom_hist": denom_hist,
      "w_hist": w_hist,
      "w_final": w,
      "s_d_final": s_d,
    }

  def analyze(self, R: Optional[Array] = None, verbose: bool = False):
    """
    Quantized RLS analysis based strictly on Diniz Chapter 5.

    Key relations:
      Stability:
        always stable if 0 < λ ≤ 1

      Effective memory:
        N_eff ≈ 1 / (1 - λ)

      Misadjustment:
        M ≈ (1 - λ)/2 * tr[R]

    Quantization effects:
      finite precision causes P-conditioning degradation
      and additional steady-state error
    """

    lam = float(self.lam)

    is_stable = (0.0 < lam <= 1.0)

    # ---- memory ----
    if lam < 1.0:
      N_eff = 1.0 / (1.0 - lam)
    else:
      N_eff = float("inf")

    # ---- misadjustment ----
    if R is not None:
      tr = float(np.trace(R).real)
      misadj = (1.0 - lam) / 2.0 * tr
    else:
      tr = None
      misadj = None

    # ---- conditioning of P ----
    eigvals = np.linalg.eigvals(self.S_D)
    cond_P = float(np.max(np.abs(eigvals)) / np.min(np.abs(eigvals)))

    # ---- quantization ----
    q_step = 2.0**(-self.q_frac_bits)
    q_noise_power = q_step**2 / 12.0

    # sensitivity: RLS more sensitive than LMS
    quant_sensitivity = cond_P * q_noise_power

    res = dict(
      lambda_value=lam,
      is_stable=is_stable,
      effective_memory=float(N_eff),
      trace_R=tr,
      theoretical_misadjustment=misadj,
      P_condition_number=cond_P,
      quantization_step=q_step,
      quantization_noise_power=q_noise_power,
      quantization_sensitivity=quant_sensitivity,
    )

    if verbose:
      self.printAnalyzeRLSQuant(res)

    return res

  def printAnalyzeRLSQuant(self, a: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("QUANTIZED RLS ANALYSIS (Diniz Chapter 5)")
    print("=" * 60)

    print(f"Lambda (λ): {a['lambda_value']:.6f}")
    print(f"Stability: {'O STABLE' if a['is_stable'] else 'X UNSTABLE'}")

    print("\nMEMORY / TRACKING:")
    print(f"  Effective memory ≈ {a['effective_memory']:.2f}")

    if a["effective_memory"] < 20:
      print("  WARNING: Very short memory → noisy")
    elif a["effective_memory"] > 1000:
      print("  WARNING: Very long memory → slow tracking")

    if a["trace_R"] is not None:
      print("\nSTEADY-STATE PERFORMANCE:")
      print(f"  trace[R]: {a['trace_R']:.6f}")
      print(f"  Misadjustment ≈ {a['theoretical_misadjustment']:.6e}")

    print("\nNUMERICAL CONDITIONING:")
    print(f"  cond(P): {a['P_condition_number']:.2e}")

    if a["P_condition_number"] > 1e6:
      print("  WARNING: Ill-conditioned → high quantization sensitivity")

    print("\nQUANTIZATION EFFECT:")
    print(f"  Δ (LSB): {a['quantization_step']:.3e}")
    print(f"  Noise power ≈ {a['quantization_noise_power']:.3e}")
    print(f"  Sensitivity ≈ {a['quantization_sensitivity']:.3e}")

    print("\nDESIGN INSIGHT:")
    print("  • λ → 1 → low misadjustment, high sensitivity")
    print("  • Smaller λ → robust but noisier")
    print("  • RLS highly sensitive to quantization of P")

    print("=" * 60)


__all__ = ['QuantizedRLSAlt']
