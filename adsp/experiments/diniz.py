"""Shared plants and theory for the uploaded Diniz examples."""

import numpy as np
from scipy.optimize import brentq

from adsp import filters as af
from adsp.correlation import ar1theoR
from adsp.statistics import eigenvalue_spread

SYSTEM_TAPS = np.array([0.1, 0.3, 0.0, -0.2, -0.4, -0.7, -0.4, -0.2])
QAM_CHANNEL = np.array([0.34 - 0.27j, 0.87 + 0.43j, 0.34 - 0.21j])
QAM_APA_WEIGHTS = np.array([0.32 + 0.21j, -0.30 + 0.70j, 0.50 - 0.80j, 0.20 + 0.50j])
QAM_RLS_WEIGHTS = np.array([0.5 + 0.2j, -0.1 + 0.4j, 0.2 - 0.4j, 0.2 + 0.7j])
SEVERE_CHANNEL = np.array([
  -0.001,
  -0.002,
  0.002,
  0.2,
  0.6,
  0.76,
  0.9,
  0.78,
  0.67,
  0.58,
  0.45,
  0.3,
  0.2,
  0.12,
  0.06,
  0.0,
  -0.2,
  -1.0,
  -2.0,
  -10.0,
  0.1,
])
TRACKING_WEIGHTS = np.array([0.03490, -0.01100, -0.06864, 0.22391, 0.55686, 0.35798, -0.02390, -0.07594])
FIXED_LATTICE_WEIGHTS = np.array([
  0.0219360,
  0.0015786,
  -0.0602449,
  -0.0118907,
  0.1375379,
  0.0574545,
  -0.3216703,
  -0.5287203,
  -0.2957797,
  0.0002043,
  0.2906700,
  -0.0353349,
  -0.0068210,
  0.0026067,
  0.0010333,
  -0.0143593,
])


def pole_for_spread(spread, *, n_taps=8):
  """Solve for stationary AR(1) eigenvalue spread rather than a coarse grid search."""
  if spread < 1 or n_taps < 2:
    raise ValueError("spread >= 1 and n_taps >= 2 are required.")
  if spread == 1:
    return 0.0
  return brentq(lambda pole: eigenvalue_spread(ar1theoR(pole, n_taps)) - spread, 0.0, 0.99999)


def membership_models(*, n_taps=8, noise_var=1e-4, spread=20):
  from functools import partial
  models = {}
  for order in (0, 1, 4):
    models[f"AP P={order}"] = partial(af.AffineProjection, filter_order=n_taps - 1, mu=0.18, P=order, delta=1e-6)
    models[f"SM-AP P={order}"] = partial(af.SMAffineProjection, filter_order=n_taps - 1,
                                         gamma_bar=np.sqrt(5 * noise_var), P=order, delta=1e-6)
  if spread == 20:
    for order, factor in enumerate((5, 7, 17)):
      models[f"SM-PUAP P={order}"] = partial(af.SMPartialUpdateAffineProjection, filter_order=n_taps - 1,
                                             gamma_bar=np.sqrt(factor * noise_var), P=order, max_update=5, delta=1e-3,
                                             selector_mode="topEnergy")
  return models


def tracking_theory(lam, *, coefficient_memory=0.999, coefficient_noise_var=0.033, input_variance=1.0, noise_var=0.01,
                    n_taps=8):
  denominator = coefficient_memory * (1 + lam**2) - lam * (1 + coefficient_memory**2)
  if abs(denominator) < 1e-15:
    raise ValueError("Tracking expression is singular for these memory factors.")
  ratio = (1 - lam) / (1 + lam) - (1 - coefficient_memory) / (1 + coefficient_memory)
  lag = n_taps * coefficient_noise_var * input_variance * ratio / denominator
  memory = n_taps * (1 - lam) / (1 + lam) * noise_var
  return {"lag": lag, "memory": memory, "excess_mse": lag + memory}
