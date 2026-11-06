"""equalization / alignment algorithms and helpers."""

from __future__ import annotations

import numpy as np


def eqfirDesired(s, n_samples, delay, dtype=np.complex64):
  """
  Desired signal for causal FIR equalizer.

  Equalizer regressor:
      reg[k] = [r[k], r[k-1], ..., r[k-M+1]]

  Therefore:
      y[k] ≈ s[k-delay]

  So:
      d[k] = s[k-delay]

  This means the first `delay` desired samples are invalid and set to zero.
  Ignore them during final constellation / MSE judgment.
  """

  if delay < 0:
    raise ValueError("delay must be >= 0")

  s = np.asarray(s, dtype=dtype)
  d = np.zeros(n_samples, dtype=dtype)

  if delay >= n_samples:
    return d

  n_valid = min(n_samples - delay, len(s))
  d[delay:delay + n_valid] = s[:n_valid]

  return d


def eqResultAlign(r, d, y, e):
  """
  Align r, d, y, e for plotting.

  NLMS:
      len(y) == len(r), so start = 0

  APA:
      your class outputs from k = P, so len(y) == len(r) - P,
      therefore start = len(r) - len(y)
  """

  r = np.asarray(r)
  d = np.asarray(d)
  y = np.asarray(y)
  e = np.asarray(e)

  start = len(r) - len(y)
  stop = start + len(y)

  if start < 0:
    raise ValueError("len(y) cannot be longer than len(r)")

  r_aligned = r[start:stop]
  d_aligned = d[start:stop]

  if len(e) != len(y):
    raise ValueError("e and y should have same length")

  return r_aligned, d_aligned, y, e, start


def makeEqW0(h, n_taps, delay):
  """
  Build a reference equalizer w0 for comparison.

  We solve for physical FIR equalizer q0 such that:
      h * q0 ≈ delayed impulse

  Since your adaptive filters store coefficients w such that:
      y = vdot(w, reg) = conj(w)^T reg

  the physical FIR taps are:
      q = conj(w)

  Therefore:
      w0 = conj(q0)
  """
  h = np.asarray(h, dtype=np.complex64)

  Lh = len(h)
  M = n_taps

  # Convolution matrix A so that:
  #   A @ q0 = h * q0
  A = np.zeros((Lh + M - 1, M), dtype=np.complex64)
  for i in range(M):
    A[i:i + Lh, i] = h

  # Desired delayed impulse
  desired = np.zeros(Lh + M - 1, dtype=np.complex64)
  desired[delay] = 1.0 + 0j

  # Least-squares solution for physical equalizer q0
  q0, _, _, _ = np.linalg.lstsq(A, desired, rcond=None)

  # Convert to your adaptive-filter stored coefficient convention
  w0 = np.conj(q0)

  return w0, q0, desired, A


__all__ = ['eqfirDesired', 'eqResultAlign', 'makeEqW0']
