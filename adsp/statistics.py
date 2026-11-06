"""Shared numerical diagnostics and stable display conversions."""

import numpy as np
from scipy.signal import welch


def power_db(x, *, floor_db=-180.0):
  return 10 * np.log10(np.maximum(np.real(np.asarray(x)), 10**(floor_db / 10)))


def amplitude_db(x, *, floor_db=-180.0):
  return 20 * np.log10(np.maximum(np.abs(x), 10**(floor_db / 20)))


def smooth(x, *, window=31):
  """Centered moving average with edge padding; output length equals input length."""
  x = np.asarray(x)
  if x.ndim != 1 or window < 1 or not len(x):
    raise ValueError("A nonempty vector and positive window are required.")
  window = min(int(window), len(x))
  left = window // 2
  padded = np.pad(x, (left, window - 1 - left), mode="edge")
  return np.convolve(padded, np.ones(window) / window, mode="valid")


def rms_normalize(x, *, eps=1e-12):
  x = np.asarray(x)
  return x / max(float(np.sqrt(np.mean(np.abs(x)**2))), eps)


def tail_mean(x, *, tail=200):
  if tail < 1 or not len(x):
    raise ValueError("tail and input length must be positive.")
  return float(np.mean(np.asarray(x)[-tail:]))


def autocorrelation(x, *, max_lag=30, normalize=True):
  x = np.asarray(x).reshape(-1)
  if not len(x) or not 0 <= max_lag < len(x):
    raise ValueError("max_lag must be nonnegative and shorter than x.")
  x = x - np.mean(x)
  r = np.array([np.vdot(x[:len(x) - lag], x[lag:]) / len(x) for lag in range(max_lag + 1)])
  if normalize:
    r /= max(float(r[0].real), 1e-15)
  return r


def covariance(x, *, n_taps):
  """R=E[reg*reg^H], reg=[x[k],x[k-1],...]; no zero-padded rows."""
  x = np.asarray(x).reshape(-1)
  if not 1 <= n_taps <= len(x):
    raise ValueError("n_taps must lie between 1 and len(x).")
  regressors = np.lib.stride_tricks.sliding_window_view(x, n_taps)[:, ::-1]
  return regressors.T @ regressors.conj() / len(regressors)


def eigenvalue_spread(matrix):
  eig = np.linalg.eigvalsh(np.asarray(matrix))
  return float(eig[-1] / eig[0]) if eig[0] > 0 else float("inf")


def psd(x, *, fs=1.0, n_fft=1024, segment_length=256):
  """Welch density, one-sided for real data and centered two-sided for complex data."""
  x = np.real_if_close(np.asarray(x))
  segment_length = min(segment_length, len(x), n_fft)
  freq, power = welch(x, fs=fs, nperseg=segment_length, nfft=n_fft, return_onesided=not np.iscomplexobj(x))
  if np.iscomplexobj(x):
    freq, power = np.fft.fftshift(freq), np.fft.fftshift(power)
  return freq, power


def prediction_error_taps(weights, *, horizon=1):
  """The residual/notch filter is 1-z^-horizon*conj(W), not W alone."""
  if horizon < 1:
    raise ValueError("horizon must be positive.")
  return np.r_[1.0, np.zeros(horizon - 1), -np.conj(np.asarray(weights))]


def normalized_lattice_error(state, *, eps=1e-12):
  """Recover physical a-priori error from normalized Algorithm 7.2 state."""
  eb = np.asarray(state["eb_bar"][:-1])
  delta_d = np.asarray(state["delta_d_bar"])
  gamma = np.prod(np.maximum(1 - eb**2, eps))
  energy = float(state["sigma_d"])**2 * np.prod(np.maximum(1 - delta_d**2, eps))
  return float(state["e_bar"]) * np.sqrt(energy / max(gamma, eps))
