"""correlation algorithms and helpers."""

from __future__ import annotations

import numpy as np
from adsp.core import Array


def toeplitzFromFirstRow(first_row: Array) -> Array:
  """
  Toeplitz with first row = [r0, r1, ..., r_{M-1}] 
  and Hermitian-symmetric for real AR(1).
  """
  first_row = np.asarray(first_row)
  M = first_row.size
  T = np.empty((M, M), dtype=first_row.dtype)
  for i in range(M):
    for j in range(M):
      k = abs(i - j)
      T[i, j] = first_row[k]
  return T


def ar1theoR(a: float, order: int, sigma_v2: float = 1.0) -> Array:
  """
  Compute theoretical autocorrelation matrix for AR(1) process.
  Assume the input x(n) is WSS, this is an implmentation of eq 2.83
  From equation (2.83): R = sigma_v²/(1-a²) * Toeplitz([1, a, a², ..., a^(order-1)])
  
  This matches the definition R = E[x(k)x^T(k)] where x(k) = [x(k), x(k-1), ..., x(k-order+1)]^T
  """
  first_row = np.array([a**k for k in range(order)], dtype=np.float64)
  scale = sigma_v2 / (1.0 - a**2)
  R = scale * toeplitzFromFirstRow(first_row)
  return R


def ar1poleSpreadSearch(
  target_spread: float,
  order: int,
  sigma_v2: float = 1.0,
  tol: float = 0.01,
  ngrid: int = 1000,
) -> float:
  """Find AR(1) pole a that produces target eigenvalue spread (max/min) using theoretical formula."""
  a_grid = np.linspace(0.0, 0.999, ngrid)
  best_diff = np.inf
  best_a = 0.0

  for a in a_grid:
    R = ar1theoR(float(a), order, sigma_v2)
    eigs = np.linalg.eigvalsh(R)
    spread = float(np.max(eigs) / np.min(eigs))
    diff = abs(spread - target_spread)

    if diff < best_diff:
      best_diff = diff
      best_a = float(a)

    if best_diff <= tol:
      break

  return best_a


def arCorrMatrix(a, u_var, order=1):
  """
  Correlation matrix R for AR(1):
    x(n) = -a x(n-1) + u(n)
  u_var = var{u(n)}
  order = filter length M
  """

  if abs(a) >= 1:
    raise ValueError("AR(1) is not stationary unless |a| < 1")

  r0 = u_var / (1 - a**2)

  # autocorrelation sequence r(0)...r(M-1)
  r = np.array([r0 * (-a)**k for k in range(order)])

  # Toeplitz matrix
  R = np.empty((order, order))
  for i in range(order):
    for j in range(order):
      R[i, j] = r[abs(i - j)]

  return R


__all__ = ['toeplitzFromFirstRow', 'ar1theoR', 'ar1poleSpreadSearch', 'arCorrMatrix']
