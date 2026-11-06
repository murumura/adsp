"""filters / lattice / quantized algorithms and helpers."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict
from adsp.core import Array
from adsp import quantization as adspquant
from adsp.filters.lattice.aposteriori import LatticeRLS


@dataclass
class QuantizedLatticeRLS(LatticeRLS):
  """
  Fixed-point Diniz Algorithm 7.1.

  Quantizer groups:
    data_q  : x, d, ef, eb, output error
    coef_q  : kappa_f, kappa_b, v, gamma, lambda
    state_q : xi_f, xi_b, delta, delta_D

  All arithmetic results are requantized so quantization error propagates through
  the lattice recursion.
  """

  data_q: adspquant.FixedQuantizer = field(
    default_factory=lambda: adspquant.FixedQuantizer(total_bits=16, frac_bits=12),
    repr=False,
  )

  coef_q: adspquant.FixedQuantizer = field(
    default_factory=lambda: adspquant.FixedQuantizer(total_bits=16, frac_bits=13),
    repr=False,
  )

  state_q: adspquant.FixedQuantizer = field(
    default_factory=lambda: adspquant.FixedQuantizer(total_bits=16, frac_bits=9),
    repr=False,
  )

  def __post_init__(self) -> None:
    super().__post_init__()

  @staticmethod
  def _scalar(x) -> float:
    return float(np.asarray(x).item())

  def qData(self, x):
    return self.data_q(x)

  def qCoef(self, x):
    return self.coef_q(x)

  def qState(self, x):
    return self.state_q(x)

  def qDataScalar(self, x: float) -> float:
    return self._scalar(self.qData(x))

  def qCoefScalar(self, x: float) -> float:
    return self._scalar(self.qCoef(x))

  def qStateScalar(self, x: float) -> float:
    return self._scalar(self.qState(x))

  def guardDenom(self, x: float, q: adspquant.FixedQuantizer) -> float:
    """
    Replace a denominator quantized to zero by one representable LSB.
    """
    x_q = self._scalar(q(x))
    lsb = 1.0 / q.scale

    if abs(x_q) >= lsb:
      return x_q

    return lsb if x_q >= 0.0 else -lsb

  def resetState(self) -> None:
    super().resetState()

    self.delta = self.qState(self.delta).astype(self.dtype)
    self.delta_d = self.qState(self.delta_d).astype(self.dtype)

    self.xi_b = self.qState(self.xi_b).astype(self.dtype)
    self.xi_f = self.qState(self.xi_f).astype(self.dtype)

    self.gamma = self.qCoef(self.gamma).astype(self.dtype)
    self.eb_prev = self.qData(self.eb_prev).astype(self.dtype)

    self.w = self.qCoef(np.real(self.w)).astype(self.dtype)

  def step(self, xk: float, dk: float) -> Dict[str, Array]:
    M = self.n_coef

    xk = self.qDataScalar(xk)
    dk = self.qDataScalar(dk)

    lam_q = self.qCoefScalar(self.lam)

    delta_prev = self.delta.copy()
    delta_d_prev = self.delta_d.copy()

    xi_b_prev = self.xi_b.copy()
    xi_f_prev = self.xi_f.copy()

    gamma_prev = self.gamma.copy()
    eb_prev = self.eb_prev.copy()

    ef = np.zeros(M + 1, dtype=self.dtype)
    eb = np.zeros(M + 1, dtype=self.dtype)
    error = np.zeros(M + 1, dtype=self.dtype)

    gamma = np.ones(M + 1, dtype=self.dtype)

    xi_b = np.zeros(M + 1, dtype=self.dtype)
    xi_f = np.zeros(M + 1, dtype=self.dtype)

    delta = np.empty(M, dtype=self.dtype)
    delta_d = np.empty(M, dtype=self.dtype)

    kappa_b = np.empty(M, dtype=self.dtype)
    kappa_f = np.empty(M, dtype=self.dtype)
    v = np.empty(M, dtype=self.dtype)

    # Order-0 initialization.
    gamma[0] = self.qCoefScalar(1.0)

    eb[0] = self.qDataScalar(xk)
    ef[0] = self.qDataScalar(xk)
    error[0] = self.qDataScalar(dk)

    x2_q = self.qStateScalar(xk * xk)
    lam_xi_q = self.qStateScalar(lam_q * xi_f_prev[0])

    xi0_q = self.qStateScalar(x2_q + lam_xi_q)

    xi_b[0] = xi0_q
    xi_f[0] = xi0_q

    for i in range(M):
      gamma_prev_i = self.guardDenom(gamma_prev[i], self.coef_q)
      gamma_i = self.guardDenom(gamma[i], self.coef_q)

      xi_b_prev_i = self.guardDenom(xi_b_prev[i], self.state_q)
      xi_b_i = self.guardDenom(xi_b[i], self.state_q)
      xi_f_i = self.guardDenom(xi_f[i], self.state_q)

      # Eq. (7.51)
      lam_delta = self.qStateScalar(lam_q * delta_prev[i])

      ebef = self.qStateScalar(eb_prev[i] * ef[i])
      corr = self.qStateScalar(ebef / gamma_prev_i)

      delta[i] = self.qStateScalar(lam_delta + corr)

      # Eq. (7.60)
      eb2 = self.qStateScalar(eb[i] * eb[i])
      gamma_corr = self.qCoefScalar(eb2 / xi_b_i)

      gamma[i + 1] = self.qCoefScalar(gamma[i] - gamma_corr)

      # Reflection coefficients.
      kappa_b[i] = self.qCoefScalar(delta[i] / xi_f_i)
      kappa_f[i] = self.qCoefScalar(delta[i] / xi_b_prev_i)

      # Eqs. (7.34), (7.33)
      kb_ef = self.qDataScalar(kappa_b[i] * ef[i])
      kf_eb = self.qDataScalar(kappa_f[i] * eb_prev[i])

      eb[i + 1] = self.qDataScalar(eb_prev[i] - kb_ef)
      ef[i + 1] = self.qDataScalar(ef[i] - kf_eb)

      # Eqs. (7.27), (7.31)
      delta_kb = self.qStateScalar(delta[i] * kappa_b[i])
      delta_kf = self.qStateScalar(delta[i] * kappa_f[i])

      xi_b[i + 1] = self.qStateScalar(xi_b_prev[i] - delta_kb)
      xi_f[i + 1] = self.qStateScalar(xi_f[i] - delta_kf)

      # Eq. (7.64)
      lam_delta_d = self.qStateScalar(lam_q * delta_d_prev[i])

      eb_error = self.qStateScalar(eb[i] * error[i])
      corr_d = self.qStateScalar(eb_error / gamma_i)

      delta_d[i] = self.qStateScalar(lam_delta_d + corr_d)

      # Eq. (7.67)
      v[i] = self.qCoefScalar(delta_d[i] / xi_b_i)

      # Eq. (7.68)
      veb = self.qDataScalar(v[i] * eb[i])
      error[i + 1] = self.qDataScalar(error[i] - veb)

    self.delta = delta
    self.delta_d = delta_d

    self.xi_b = xi_b
    self.xi_f = xi_f

    self.gamma = gamma
    self.eb_prev = eb

    self.w = v.copy()

    return {
      "y": self.qDataScalar(dk - error[-1]),
      "e": self.qDataScalar(error[-1]),
      "e_order": error,
      "ef": ef,
      "eb": eb,
      "delta": delta,
      "delta_d": delta_d,
      "gamma": gamma,
      "xi_f": xi_f,
      "xi_b": xi_b,
      "kappa_f": kappa_f,
      "kappa_b": kappa_b,
      "v": v,
    }


__all__ = ['QuantizedLatticeRLS']
