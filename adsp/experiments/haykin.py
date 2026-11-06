"""Channel family and approximate theory used in the uploaded Haykin experiments."""

import numpy as np
from scipy.linalg import toeplitz


def channel(width):
  n = np.arange(1, 4)
  return 0.5 * (1 + np.cos(2 * np.pi / width * (n - 2)))


def channel_covariance(h, *, n_taps=11, noise_var=0.001):
  h = np.asarray(h)
  positive_lags = np.correlate(h, h, mode="full")[len(h) - 1:]
  first_column = np.zeros(n_taps, dtype=np.result_type(h, complex))
  first_column[:min(n_taps, len(h))] = positive_lags[:n_taps]
  first_column[0] += noise_var
  return toeplitz(first_column)


def scalar_lms_theory(n_samples, *, mu, process_variance=0.936, pole=0.99):
  """Small-step independence approximation using the Haykin mu convention."""
  floor = process_variance * (1 - pole**2)
  steady = mu * floor / (2 - mu * process_variance)
  return floor + process_variance * (steady + (1 - mu * process_variance)**(2 * np.arange(n_samples)) *
                                     (pole**2 - steady))
