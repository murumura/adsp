"""Array geometry and constrained/GSC LMS with the shared training interface."""

from dataclasses import dataclass

import numpy as np

from .data import gaussian
from .datasets import SampleBatch


def steering(n_sensors, direction, *, spacing_wavelengths=0.5, angle_convention="sin"):
  """direction is sin(theta), or degrees measured from the array axis for convention='axis'."""
  if n_sensors < 2 or spacing_wavelengths <= 0:
    raise ValueError("At least two sensors and positive spacing are required.")
  if angle_convention == "axis":
    direction = np.cos(np.deg2rad(direction))
  elif angle_convention != "sin" or not -1 <= direction <= 1:
    raise ValueError("Use axis degrees or a sin(theta) direction in [-1,1].")
  return np.exp(-2j * np.pi * spacing_wavelengths * np.arange(n_sensors) * direction)


def snapshots(n_samples, *, rng, target_vector, interference_vector, target_power=10.0, interference_power=10000.0,
              noise_var=1.0, target_frequency=None, interference_frequency=None):
  """Return a snapshot matrix (N,M) and the desired target waveform (N,)."""
  target_vector, interference_vector = np.asarray(target_vector), np.asarray(interference_vector)
  if target_vector.shape != interference_vector.shape:
    raise ValueError("Steering vectors must have the same shape.")
  n_sensors = len(target_vector)
  time = np.arange(n_samples)
  target = np.full(n_samples, np.sqrt(target_power), dtype=np.complex128)
  if target_frequency is not None:
    target *= np.exp(2j * np.pi * target_frequency * time)
  interference = np.sqrt(interference_power) * np.exp(2j * np.pi * rng.random(n_samples))
  if interference_frequency is not None:
    interference = np.sqrt(interference_power) * np.exp(2j * np.pi * interference_frequency * time)
  noise = gaussian(n_samples * n_sensors, rng=rng, variance=noise_var, complex_data=True)
  matrix = target[:, None] * target_vector + interference[:, None] * interference_vector
  matrix += noise.reshape(n_samples, n_sensors)
  return SampleBatch(matrix, target, info={"target_vector": target_vector, "interference_vector": interference_vector})


def gsc_batch(batch, *, gain=1.0):
  """First-sensor blocking without allocating an N-by-N diagonal matrix."""
  target = batch.info["target_vector"]
  blocked = batch.x[:, :1] * target[None, 1:] - batch.x[:, 1:]
  return SampleBatch(blocked, gain * batch.x[:, 0], info=batch.info.copy())


def gsc_weights(adaptive_weights, target_vector, *, gain=1.0):
  adaptive_weights = np.asarray(adaptive_weights)
  first = np.conj(gain) - adaptive_weights @ np.conj(np.asarray(target_vector)[1:])
  return np.concatenate([np.asarray(first)[..., None], adaptive_weights], axis=-1)


def response(weights, directions, *, spacing_wavelengths=0.5):
  weights, directions = np.asarray(weights), np.asarray(directions)
  matrix = np.exp(-2j * np.pi * spacing_wavelengths * np.outer(np.arange(len(weights)), directions))
  return weights.conj() @ matrix


@dataclass
class SnapshotLMS:
  """Complex LMS on externally constructed regressors; update uses mu (not 2*mu)."""

  n_features: int
  mu: float = 1e-6

  def __post_init__(self):
    self.w = np.zeros(self.n_features, dtype=np.complex128)

  def __call__(self, x, d, train=True):
    x, d = np.asarray(x), np.asarray(d)
    if x.shape != (len(d), self.n_features):
      raise ValueError("x must have one row per target and n_features columns.")
    output, error = np.empty(len(d), dtype=complex), np.empty(len(d), dtype=complex)
    history = np.empty((len(d) + 1, self.n_features), dtype=complex)
    history[0] = self.w
    for k in range(len(d)):
      output[k] = np.vdot(self.w, x[k])
      error[k] = d[k] - output[k]
      if train:
        self.w += self.mu * np.conj(error[k]) * x[k]
      history[k + 1] = self.w
    return output, error, history


@dataclass
class ConstrainedLMS:
  """Minimize output power subject to C^H w=f; complex gradient uses conj(y)."""

  constraint: np.ndarray
  desired_gain: np.ndarray
  mu: float = 0.01

  def __post_init__(self):
    matrix = np.asarray(self.constraint, dtype=complex)
    if matrix.ndim == 1:
      matrix = matrix[:, None]
    gain = np.asarray(self.desired_gain, dtype=complex).reshape(-1)
    if len(gain) != matrix.shape[1] or np.linalg.matrix_rank(matrix) != matrix.shape[1]:
      raise ValueError("Use independent constraint columns and one gain per column.")
    inverse = np.linalg.inv(matrix.conj().T @ matrix)
    self.fixed = matrix @ inverse @ gain
    self.projector = np.eye(matrix.shape[0]) - matrix @ inverse @ matrix.conj().T
    self.w = self.fixed.copy()

  def __call__(self, x, d, train=True):
    output = np.empty(len(d), dtype=complex)
    history = np.empty((len(d) + 1, len(self.w)), dtype=complex)
    history[0] = self.w
    for k in range(len(d)):
      output[k] = np.vdot(self.w, x[k])
      if train:
        self.w = self.projector @ (self.w - 2 * self.mu * np.conj(output[k]) * x[k]) + self.fixed
      history[k + 1] = self.w
    return {"y": output, "e": d - output, "w_hist": history, "output_power": np.abs(output)**2}
