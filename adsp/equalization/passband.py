"""equalization / passband algorithms and helpers."""

from __future__ import annotations

import numpy as np
from adsp.core import Array
from scipy.signal import fftconvolve, hilbert
from adsp.equalization.fse import BaseFse
from adsp.equalization.pulses import _requireScipy


def analyticFromReal(passband: Array) -> Array:
  """Ideal finite-record Hilbert phase splitter used for illustration."""
  _requireScipy()
  real = np.asarray(passband)
  if np.iscomplexobj(real) and np.max(np.abs(real.imag)) > 1e-12:
    raise ValueError("passband input to the Hilbert splitter must be real")
  return hilbert(np.asarray(real.real, dtype=np.float64).reshape(-1))


def complexToPassband(baseband: Array, carrier_cycles_per_sample: float) -> tuple[Array, Array]:
  """Return real passband and its ideal analytic representation."""
  u = np.asarray(baseband, dtype=np.complex128).reshape(-1)
  if not 0.0 < carrier_cycles_per_sample < 0.5:
    raise ValueError("carrier must be between 0 and 0.5 cycles/high-rate sample")
  n = np.arange(len(u))
  analytic = u * np.exp(2j * np.pi * carrier_cycles_per_sample * n)
  return analytic.real.copy(), analytic


def passbandEquivalentOutput(
  fse: BaseFse,
  analytic_passband: Array,
  carrier_cycles_per_sample: float,
  n_symbols: int,
) -> Array:
  """Apply carrier-shifted FSE taps to an analytic passband signal
  and return the demodulated symbol-rate equivalent output."""
  _requireScipy()
  lag, coeff = fse.compositeTaps()
  shifted = coeff * np.exp(2j * np.pi * carrier_cycles_per_sample * lag)
  full = fftconvolve(np.asarray(analytic_passband, dtype=np.complex128), shifted, mode="full")
  sample_times = fse.oversampling * np.arange(n_symbols, dtype=int)
  loc = sample_times - lag[0]
  out = np.zeros(n_symbols, dtype=np.complex128)
  valid = (loc >= 0) & (loc < len(full))
  out[valid] = full[loc[valid]]
  out *= np.exp(-2j * np.pi * carrier_cycles_per_sample * sample_times)
  return out


def splitStaggeredStreams(
  analytic_rf: Array,
  samples_per_symbol: int,
  i_gain: float = 1.0,
  q_gain: float = 1.0,
  i_leak: float = 0.0,
  q_leak: float = 0.0,
) -> tuple[Array, Array]:
  """Create two 2/T real streams with a relative T/4 sampling offset.

  Gain/leakage parameters are an illustrative nonideal phase-splitter model,
  not a textbook-prescribed impairment model.
  """
  if samples_per_symbol < 4 or samples_per_symbol % 4:
    raise ValueError("samples_per_symbol must be a multiple of 4")
  rf = np.asarray(analytic_rf, dtype=np.complex128).reshape(-1)
  stride = samples_per_symbol // 2
  stagger = samples_per_symbol // 4
  ni = rf[::stride]
  nq = rf[stagger::stride]
  length = min(len(ni), len(nq))
  i0 = ni[:length].real
  q0 = nq[:length].imag
  return i_gain * i0 + i_leak * q0, q_gain * q0 + q_leak * i0


def demodulateSymbols(
  z_passband: Array,
  carrier_cycles_per_sample: float,
  samples_per_symbol: int,
) -> Array:
  """Remove the carrier at symbol-rate decision instants."""
  k = np.arange(len(z_passband))
  return np.asarray(z_passband) * np.exp(-2j * np.pi * carrier_cycles_per_sample * samples_per_symbol * k)


__all__ = [
  'analyticFromReal', 'complexToPassband', 'passbandEquivalentOutput', 'splitStaggeredStreams', 'demodulateSymbols'
]
