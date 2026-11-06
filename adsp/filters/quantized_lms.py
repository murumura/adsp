"""filters / quantized lms algorithms and helpers."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Literal
from adsp.core import Array
from adsp import quantization as adspquant
from adsp.filters.base import BaseLMS
from adsp.core import _prepad


@dataclass
class QuantizedLMS(BaseLMS):
  """
  Quantized complex LMS.

  Update:
    y_Q(k) = Q{ w_Q^H(k) x(k) }
    e_Q(k) = Q{ d(k) - y_Q(k) }
    w_Q(k+1) = Q{ w_Q(k) + mu * conj(e_Q(k)) * x_Q(k) }
  """

  q_total_bits: int = 16
  q_frac_bits: int = 12
  q_store_inputs: bool = False
  q_inner_mode: Literal["after_add", "after_product"] = "after_add"

  q: adspquant.FixedQuantizer = field(init=False)

  def __post_init__(self) -> None:
    super().__post_init__()
    self.q = adspquant.FixedQuantizer(
      total_bits=self.q_total_bits,
      frac_bits=self.q_frac_bits,
      signed=True,
    )
    self.w = self.q(np.asarray(self.w, dtype=np.complex64))

  def resetState(self) -> None:
    self.w = self.q(np.zeros((self.n_coef, ), dtype=np.complex64))

  def __call__(self, x: Array, d: Array, train: bool = True) -> Dict[str, Array]:
    x_arr = np.asarray(x)
    d_arr = np.asarray(d)

    n_samples = x_arr.shape[0]
    x_pad = _prepad(x_arr, self.n_coef)

    w = self.q(self.w.copy())

    y_hist = np.empty((n_samples, ), dtype=np.complex64)
    e_hist = np.empty((n_samples, ), dtype=np.complex64)
    w_hist = np.empty((n_samples + 1, self.n_coef), dtype=np.complex64)
    w_hist[0] = w

    for k_idx in range(n_samples):
      x_k = x_pad[k_idx:k_idx + self.n_coef][::-1].astype(np.complex64)
      if self.q_store_inputs:
        x_k = self.q(x_k)

      y_q = adspquant.qdot(w, x_k, self.q, conj_a=True, mode=self.q_inner_mode)
      d_q = np.complex64(self.q(np.array(d_arr[k_idx], dtype=np.complex64))[()])
      e_q = np.complex64(self.q(d_q - y_q))

      if train:
        update_q = self.q(self.mu * np.conj(e_q) * x_k)
        w = self.q(w + update_q)

      y_hist[k_idx] = y_q
      e_hist[k_idx] = e_q
      w_hist[k_idx + 1] = w

    if train:
      self.w = w

    return {
      "y": y_hist,
      "e": e_hist,
      "w_hist": w_hist,
      "w_final": w,
    }

  def analyze(self, R: Optional[Array] = None, verbose: bool = False):
    """
    Quantized LMS analysis based on Diniz Chapter 3 & 4.

    Key relations:
      Stability:
        0 < μ < 1 / λ_max

      Misadjustment:
        M ≈ μ tr[R] / (1 - μ tr[R])

      Quantization:
        introduces additional steady-state error floor
    """

    mu = float(self.mu)

    if R is not None:
      eig = np.linalg.eigvals(R)
      lam_max = float(np.max(np.real(eig)))
      lam_min = float(np.min(np.real(eig)))
      tr = float(np.trace(R).real)

      is_stable = (0.0 < mu < 1.0 / lam_max)

      # misadjustment (Diniz Ch.3)
      if (1.0 - mu * tr) > 0:
        misadj = (mu * tr) / (1.0 - mu * tr)
      else:
        misadj = float("inf")

      spread = lam_max / lam_min if lam_min > 0 else float("inf")

    else:
      lam_max = None
      lam_min = None
      tr = None
      is_stable = None
      misadj = None
      spread = None

    # quantization noise proxy (very useful in practice)
    q_step = 2.0**(-self.q_frac_bits)
    q_noise_power = q_step**2 / 12.0

    res = dict(
      mu_value=mu,
      is_stable=is_stable,
      lambda_max=lam_max,
      lambda_min=lam_min,
      eigenvalue_spread=spread,
      trace_R=tr,
      theoretical_misadjustment=misadj,
      quantization_step=q_step,
      quantization_noise_power=q_noise_power,
    )

    if verbose:
      self.printAnalyzeLMSQuant(res)

    return res

  def printAnalyzeLMSQuant(self, a: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("QUANTIZED LMS ANALYSIS (Diniz Ch.3 / Ch.4)")
    print("=" * 60)

    print(f"Step size (μ): {a['mu_value']:.6f}")

    if a["is_stable"] is not None:
      print(f"Stability: {'O STABLE' if a['is_stable'] else 'X UNSTABLE'}")

    if a["lambda_max"] is not None:
      print("\nEIGENVALUE PROPERTIES:")
      print(f"  λ_max: {a['lambda_max']:.6f}")
      print(f"  λ_min: {a['lambda_min']:.6f}")
      print(f"  Spread: {a['eigenvalue_spread']:.2f}")

    if a["trace_R"] is not None:
      print("\nSTEADY-STATE PERFORMANCE:")
      print(f"  trace[R]: {a['trace_R']:.6f}")
      print(f"  Misadjustment ≈ {a['theoretical_misadjustment']:.6e}")

    print("\nQUANTIZATION EFFECT:")
    print(f"  Δ (LSB): {a['quantization_step']:.3e}")
    print(f"  Noise power ≈ {a['quantization_noise_power']:.3e}")

    print("\nDESIGN INSIGHT:")
    print("  • Larger μ → faster convergence, higher misadjustment")
    print("  • Quantization adds noise floor")
    print("  • High eigenvalue spread → slow LMS")

    print("=" * 60)


__all__ = ['QuantizedLMS']
