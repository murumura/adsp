"""equalization / pulses algorithms and helpers."""

from __future__ import annotations

import numpy as np
from adsp.core import Array
from scipy.signal import fftconvolve, hilbert
from scipy.signal import fftconvolve, hilbert

_scipy_import_error = None


def _requireScipy() -> None:
  if fftconvolve is None or hilbert is None:
    raise ImportError("SciPy is required for the FSE / Nyquist-inband illustration helpers.") from _scipy_import_error


def rootRaisedCosine(t: Array, beta: float = 0.35, symbol_period: float = 1.0) -> Array:
  """Unit-energy root-raised-cosine pulse used by the FSE illustrations.

  This is a numerical pulse generator for the executable examples; it is not a
  new equalizer assumption.  ``beta=0`` reduces to the sinc pulse used by the
  textbook's orthonormal shifted-sinc examples.
  """
  if not 0.0 <= beta <= 1.0 or symbol_period <= 0.0:
    raise ValueError("beta must be in [0, 1] and symbol_period > 0")
  u = np.asarray(t, dtype=np.float64) / symbol_period
  if beta == 0.0:
    return np.sinc(u) / np.sqrt(symbol_period)

  result = np.empty_like(u)
  at_zero = np.isclose(u, 0.0, atol=1e-12)
  at_special = np.isclose(np.abs(u), 1.0 / (4.0 * beta), atol=1e-12)
  other = ~(at_zero | at_special)

  result[at_zero] = (1.0 + beta * (4.0 / np.pi - 1.0)) / np.sqrt(symbol_period)
  result[at_special] = beta / np.sqrt(2.0 * symbol_period) * ((1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * beta)) +
                                                              (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * beta)))
  v = u[other]
  result[other] = (np.sin(np.pi * (1.0 - beta) * v) +
                   4.0 * beta * v * np.cos(np.pi * (1.0 + beta) * v)) / (np.pi * v *
                                                                         (1.0 -
                                                                          (4.0 * beta * v)**2) * np.sqrt(symbol_period))
  return result


def sampledPulse(
  channel: Array,
  oversampling: int,
  span_symbols: int = 12,
  timing_offset: float = 0.0,
  rolloff: float = 0.35,
) -> tuple[Array, Array]:
  """Build the high-rate receive pulse ``g[n]`` used by the FSE examples.

  With T=1,

    g[n] = sum_j h[j] p(n/ell + timing_offset - j),

  where ``p`` is an RRC pulse (sinc when ``rolloff=0``).  The textbook's
  anti-alias gain sqrt(T) is one under this T=1 normalization.  The finite pulse
  span is an explicit simulation approximation.
  """
  if oversampling < 2 or span_symbols < 2:
    raise ValueError("oversampling >= 2 and span_symbols >= 2 required")
  c = np.asarray(channel, dtype=np.complex128).reshape(-1)
  if c.size == 0 or not np.any(c):
    raise ValueError("channel must contain nonzero taps")

  idx = np.arange(
    -span_symbols * oversampling,
    (span_symbols + c.size - 1) * oversampling + 1,
    dtype=int,
  )
  t = idx / oversampling + float(timing_offset)
  g = sum(coef * rootRaisedCosine(t - j, rolloff) for j, coef in enumerate(c))
  return idx, np.asarray(g, dtype=np.complex128)


def sampleChannelOutput(x: Array, g_index: Array, g: Array, oversampling: int) -> Array:
  """Generate high-rate FSE input samples ``r[n] = sum_k x[k] g[n-ell*k]``."""
  _requireScipy()
  x_arr = np.asarray(x, dtype=np.complex128).reshape(-1)
  idx = np.asarray(g_index, dtype=int).reshape(-1)
  pulse = np.asarray(g, dtype=np.complex128).reshape(-1)
  if x_arr.size == 0 or idx.size != pulse.size or np.any(np.diff(idx) != 1):
    raise ValueError("x must be nonempty; g_index consecutive and same length as g")
  if oversampling < 2:
    raise ValueError("oversampling must be >= 2")

  up = np.zeros(oversampling * len(x_arr), dtype=np.complex128)
  up[::oversampling] = x_arr
  full = fftconvolve(up, pulse, mode="full")
  n = np.arange(up.size, dtype=int) - idx[0]
  return full[n]


__all__ = ['rootRaisedCosine', 'sampledPulse', 'sampleChannelOutput']
