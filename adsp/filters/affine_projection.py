"""filters / affine projection algorithms and helpers."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Tuple
from adsp.core import Array
from adsp.filters.base import BaseLMS
from adsp.core import _prepad


@dataclass
class AffineProjection(BaseLMS):
  """
  Complex Affine Projection Algorithm

  P = projection order in Diniz problem notation L
  This implementation uses P+1 reused data vectors.
  """
  P: int = 3
  delta: float = 1e-3

  def __call__(self, x: Array, d: Array, train: bool = True) -> Tuple[Array, Array, Array]:
    x = np.asarray(x, dtype=np.complex64)
    d = np.asarray(d, dtype=np.complex64)

    N = x.shape[0]
    M = self.n_coef
    Pp = self.P + 1

    x_pad = _prepad(x, M)
    w = self.w.copy()

    n_out = N - self.P
    y_hist = np.empty((n_out, ), dtype=np.complex64)
    e_hist = np.empty((n_out, ), dtype=np.complex64)
    w_hist = np.empty((n_out + 1, M), dtype=np.complex64)
    w_hist[0] = w

    out_idx = 0
    for k in range(self.P, N):
      # Xap(k): shape (M, P+1)
      cols = []
      for i in range(Pp):
        kk = k - i
        # reverse to get [x(k-i), x(k-i-1), ..., x(k-i-M+1)]
        if kk < 0:
          # pad zero for current time < 0 for aligning
          reg = np.zeros((M, ), dtype=np.complex64)
        else:
          reg = x_pad[kk:kk + M][::-1].astype(np.complex64)
        cols.append(reg)

      Xap = np.stack(cols, axis=1)  # (M, P+1)

      d_ap = np.array([d[k - i] if (k - i) >= 0 else 0.0 for i in range(Pp)], dtype=np.complex64)
      # Diniz complex notation: y_ap = X_ap^T w*
      y_ap = Xap.T @ np.conj(w)  # (P+1,)
      e_ap = d_ap - y_ap  # (P+1,)

      R = Xap.conj().T @ Xap
      R_reg = R + self.delta * np.eye(Pp, dtype=np.complex64)

      try:
        g = np.linalg.solve(R_reg, np.conj(e_ap))
      except np.linalg.LinAlgError:
        g = (np.linalg.pinv(R_reg) @ np.conj(e_ap)).astype(np.complex64)

      upd = (self.mu * (Xap @ g)).astype(np.complex64)

      if train:
        w = (w + upd).astype(np.complex64)

      y_hist[out_idx] = y_ap[0]
      e_hist[out_idx] = e_ap[0]
      w_hist[out_idx + 1] = w
      out_idx += 1

    if train:
      self.w = w

    return y_hist, e_hist, w_hist

  def analyze(self, verbose=False):
    mu = float(self.mu)
    Lp = self.P + 1

    is_stable = (0.0 < mu < 2.0)

    num1 = (Lp * mu) / (2.0 - mu)
    num2 = 1.0 - (1.0 - mu)**2
    den2 = 1.0 - (1.0 - mu)**(2 * Lp)

    if abs(den2) < 1e-12:
      M_exact = np.nan
    else:
      M_exact = num1 * (num2 / den2)

    M_approx = (Lp * mu) / (2.0 - mu)

    res = dict(
      mu=mu,
      projection_order=self.P,
      projection_dimension=Lp,
      is_stable=is_stable,
      misadjustment_exact=M_exact,
      misadjustment_approx=M_approx,
      stability_bound_upper=2.0,
    )

    if verbose:
      self.printAnalyzeAPA(res)

    return res

  def printAnalyzeAPA(self, a):
    print("\n" + "=" * 60)
    print("AFFINE PROJECTION ANALYSIS")
    print("=" * 60)
    print(f"Projection order P: {a['projection_order']}")
    print(f"Projection dimension P+1: {a['projection_dimension']}")
    print(f"mu: {a['mu']:.6f}")
    print(f"Stability: {'O STABLE' if a['is_stable'] else 'X UNSTABLE'}")
    print(f"Exact misadjustment: {a['misadjustment_exact']:.6f}")
    print(f"Approx misadjustment: {a['misadjustment_approx']:.6f}")
    print("=" * 60)


__all__ = ['AffineProjection']
