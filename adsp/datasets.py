"""In-memory experiment data and explicit coefficient conventions."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SampleBatch:
  """Samples with one desired value per row; all arrays stay in memory.

  x can be a scalar stream (N,) or a snapshot matrix (N,M).
  input_rate > 1 represents a high-rate stream with input_rate samples per target.
  clean is an optional evaluation reference, not necessarily the training target.
  true_weights are stored adaptive weights for y = w^H x, not physical FIR taps.
  """

  x: np.ndarray
  d: np.ndarray
  clean: np.ndarray | None = None
  true_weights: np.ndarray | None = None
  valid: np.ndarray | None = None
  info: dict[str, Any] = field(default_factory=dict)
  input_rate: int = 1

  def __post_init__(self):
    self.x = np.asarray(self.x)
    self.d = np.asarray(self.d)
    if not isinstance(self.input_rate, int) or self.input_rate < 1:
      raise ValueError("input_rate must be a positive integer.")
    if self.x.ndim not in (1, 2) or self.d.ndim != 1 or len(self.x) != self.input_rate * len(self.d):
      raise ValueError("x must have input_rate rows per target; d must be a vector.")
    if self.x.ndim == 2 and self.input_rate != 1:
      raise ValueError("Snapshot matrices require input_rate=1.")
    if not len(self.d):
      raise ValueError("A batch must contain at least one sample.")
    if not np.all(np.isfinite(self.x)) or not np.all(np.isfinite(self.d)):
      raise ValueError("Data must be finite.")
    if self.clean is not None:
      self.clean = np.asarray(self.clean)
      if self.clean.shape != self.d.shape:
        raise ValueError("clean must have the same shape as d.")
    if self.valid is None:
      self.valid = np.ones(len(self.d), dtype=bool)
    else:
      self.valid = np.asarray(self.valid, dtype=bool)
      if self.valid.shape != self.d.shape:
        raise ValueError("valid must have the same shape as d.")
    if self.true_weights is not None:
      self.true_weights = np.asarray(self.true_weights)
