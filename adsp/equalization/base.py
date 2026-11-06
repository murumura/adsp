"""equalization / base algorithms and helpers."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from adsp.core import Array


@dataclass
class BaseEqualizer:
  """Common state and utilities shared by Chapter 3 equalizers."""

  eps: float = 1e-12
  dtype: object = np.complex128

  def __post_init__(self) -> None:
    if self.eps <= 0.0:
      raise ValueError(f"eps must be > 0, got {self.eps}.")

  @staticmethod
  def responseFromTaps(taps: Array, tap_index: Array, normalized_frequency: Array) -> Array:
    """
    Evaluate A(D) = sum_k a(k) D^k on D = exp(-j*2*pi*f).

    normalized_frequency is in cycles/sample, so the Nyquist interval is
    -0.5 <= f <= 0.5.
    """

    taps_arr = np.asarray(taps).reshape(-1)
    idx_arr = np.asarray(tap_index, dtype=int).reshape(-1)
    freq_arr = np.asarray(normalized_frequency, dtype=np.float64).reshape(-1)

    if taps_arr.size != idx_arr.size:
      raise ValueError("taps and tap_index must have the same length.")

    phase = np.exp(-1j * 2.0 * np.pi * freq_arr[:, None] * idx_arr[None, :])
    return phase @ taps_arr


__all__ = ['BaseEqualizer']
