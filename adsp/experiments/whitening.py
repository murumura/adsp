"""Reusable whitening preparation; shared training and plotting handle the experiment."""

from dataclasses import replace

import numpy as np

from adsp import data, statistics as stats
from adsp.whitening import FirWhitener, ArWhitener, SpectralWhitener

PLANT_TAPS = np.array([
  0.45,
  -0.25,
  0.18,
  0.12,
  -0.08,
  0.05,
  0.03,
  -0.02,
  0.015,
  -0.01,
  0.008,
  0.0,
  -0.006,
  0.004,
  0.002,
  -0.001,
])


def whiten_pair(batch, *, order=1):
  whitener = ArWhitener(ar_order=order)
  x, desired, _ = whitener.fitWhitenPair(batch.x, batch.d, normalize=True)
  return replace(batch, x=x, d=desired, clean=None)


def prepare(n_samples=4096, *, rng, pole=0.92):
  colored = data.ar_process(n_samples, rng=rng, coefficients=[pole], burn_in=512)
  difference = FirWhitener(init_coef=np.array([1.0, -1.0]))(colored)
  ar = ArWhitener(ar_order=1)
  ar_white, _ = ar.fitTransform(colored, normalize=True)
  spectral = SpectralWhitener(fft_len=n_samples, smooth_len=17, max_gain=32.0, normalize_output=True)
  spectral_white = spectral.fitTransform(colored)
  signals = {
    "Colored input": stats.rms_normalize(colored),
    "First difference": stats.rms_normalize(difference),
    "AR whitener": ar_white.real,
    "Spectral whitener": spectral_white.real
  }
  return signals, ar, spectral
