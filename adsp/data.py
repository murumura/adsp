"""Reproducible signal generators. Supply rng explicitly; no function saves data."""

import numpy as np
from scipy.signal import lfilter

from .datasets import SampleBatch


def gaussian(n_samples, *, rng, variance=1.0, complex_data=False):
  """Generate zero-mean noise; variance always means E|n|^2."""
  if n_samples < 1 or variance < 0:
    raise ValueError("n_samples must be positive and variance must be nonnegative.")
  if complex_data:
    return np.sqrt(variance / 2) * (rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples))
  return np.sqrt(variance) * rng.standard_normal(n_samples)


def symbols(n_samples, *, rng, kind="qpsk", energy=1.0):
  """BPSK, 4-QAM/QPSK, or 16-QAM symbols with a specified ensemble mean energy."""
  if n_samples < 1 or energy <= 0:
    raise ValueError("n_samples and energy must be positive.")
  if kind == "bpsk":
    return np.sqrt(energy) * rng.choice([-1.0, 1.0], n_samples)
  if kind in ("qpsk", "qam4"):
    return np.sqrt(energy / 2) * (rng.choice([-1.0, 1.0], n_samples) + 1j * rng.choice([-1.0, 1.0], n_samples))
  if kind == "qam16":
    levels = [-3.0, -1.0, 1.0, 3.0]
    return np.sqrt(energy / 10) * (rng.choice(levels, n_samples) + 1j * rng.choice(levels, n_samples))
  raise ValueError("kind must be bpsk, qpsk, qam4, or qam16.")


def add_noise(x, *, rng, noise_var=None, snr_db=None):
  """Add proper complex noise to a complex stream, or real noise to a real stream."""
  x = np.asarray(x)
  if (noise_var is None) == (snr_db is None):
    raise ValueError("Specify exactly one of noise_var or snr_db.")
  if snr_db is not None:
    noise_var = float(np.mean(np.abs(x)**2)) / 10**(snr_db / 10)
  noise = gaussian(len(x), rng=rng, variance=noise_var, complex_data=np.iscomplexobj(x))
  return x + noise, float(noise_var)


def ar_process(n_samples, *, rng, coefficients, variance=1.0, burn_in=0):
  """x[k] = sum_i coefficients[i] x[k-i-1] + innovation[k].

  variance is the stationary process variance, not the innovation variance.
  Zero initial state and optional burn-in are explicit simulation choices.
  """
  coefficients = np.asarray(coefficients, dtype=np.float64)
  if coefficients.ndim != 1 or not len(coefficients) or variance <= 0 or burn_in < 0:
    raise ValueError("Provide nonempty coefficients, positive variance, and nonnegative burn_in.")
  if np.any(np.abs(np.roots(np.r_[1.0, -coefficients])) >= 1):
    raise ValueError("AR poles must lie strictly inside the unit circle.")
  from scipy.linalg import solve_discrete_lyapunov
  size = len(coefficients)
  companion = np.zeros((size, size))
  companion[0] = coefficients
  if size > 1:
    companion[1:, :-1] = np.eye(size - 1)
  drive = np.zeros((size, size))
  drive[0, 0] = 1.0
  unit_cov = solve_discrete_lyapunov(companion, drive)
  innovation_var = variance / unit_cov[0, 0]
  noise = gaussian(n_samples + burn_in, rng=rng, variance=innovation_var)
  return lfilter([1.0], np.r_[1.0, -coefficients], noise)[burn_in:]


def ar2_innovation_variance(a1=0.1, a2=-0.8, *, variance=1.0):
  """Haykin sign convention: x[k] = -a1 x[k-1] - a2 x[k-2] + v[k]."""
  phi1, phi2 = -a1, -a2
  if np.any(np.abs(np.roots([1, a1, a2])) >= 1) or variance <= 0:
    raise ValueError("A stable AR(2) model and positive variance are required.")
  r1 = phi1 * variance / (1 - phi2)
  r2 = phi1 * r1 + phi2 * variance
  return float(variance - phi1 * r1 - phi2 * r2)


def sinusoids(n_samples, *, frequencies=(0.1, ), amplitudes=(1.0, ), phases=None):
  """Frequencies are cycles/sample, phases are radians."""
  if len(frequencies) != len(amplitudes):
    raise ValueError("frequencies and amplitudes must have matching lengths.")
  if phases is None:
    phases = np.zeros(len(frequencies))
  if len(phases) != len(frequencies):
    raise ValueError("phases and frequencies must have matching lengths.")
  time = np.arange(n_samples)
  return sum(a * np.sin(2 * np.pi * f * time + p) for f, a, p in zip(frequencies, amplitudes, phases))


def delay(x, n_samples):
  """A causal zero-filled delay; never wraps samples around the record."""
  x = np.asarray(x)
  if n_samples < 0:
    raise ValueError("Delay must be nonnegative.")
  y = np.zeros_like(x)
  if n_samples == 0:
    return x.copy()
  if n_samples < len(x):
    y[n_samples:] = x[:-n_samples]
  return y


def system_identification(n_samples, *, rng, h, pole=0.0, input_kind="gaussian", input_variance=1.0, noise_var=1e-4,
                          burn_in=0, coefficient_convention="physical"):
  """FIR identification with physical h or stored weights for y = w^H x."""
  h = np.asarray(h)
  if h.ndim != 1 or not len(h):
    raise ValueError("h must be a nonempty vector.")
  if input_kind == "gaussian":
    x = ar_process(n_samples, rng=rng, coefficients=[pole], variance=input_variance, burn_in=burn_in)
  elif input_kind == "complex_gaussian":
    if pole != 0:
      raise ValueError("Colored complex Gaussian input is not configured by this generator.")
    x = gaussian(n_samples, rng=rng, variance=input_variance, complex_data=True)
  else:
    x = symbols(n_samples, rng=rng, kind=input_kind, energy=input_variance)
  if coefficient_convention not in ("physical", "stored"):
    raise ValueError("coefficient_convention must be physical or stored.")
  physical = h if coefficient_convention == "physical" else np.conj(h)
  clean = lfilter(physical, [1.0], x)
  desired, _ = add_noise(clean, rng=rng, noise_var=noise_var)
  return SampleBatch(x, desired, clean=clean, true_weights=np.conj(physical), info={
    "h": physical,
    "pole": pole,
    "noise_var": noise_var,
  })


def prediction(process, *, horizon=1, true_weights=None, info=None):
  """Predict process[k+horizon] from process[k] and its causal history."""
  process = np.asarray(process)
  if horizon < 1 or len(process) <= horizon:
    raise ValueError("horizon must be positive and shorter than the process.")
  return SampleBatch(process[:-horizon], process[horizon:], true_weights=true_weights, info=info or {})


def ar_prediction(n_samples, *, rng, coefficients, variance=1.0, burn_in=0, observation_snr_db=None):
  process = ar_process(n_samples + 1, rng=rng, coefficients=coefficients, variance=variance, burn_in=burn_in)
  noise_var = 0.0
  if observation_snr_db is not None:
    process, noise_var = add_noise(process, rng=rng, snr_db=observation_snr_db)
  return prediction(
    process, true_weights=np.asarray(coefficients) if observation_snr_db is None else None, info={
      "observation_noise_var": noise_var,
      "coefficients": np.asarray(coefficients),
      "reference_note": "Clean AR parameters need not be the optimal noisy-observation predictor.",
    })


def sinusoid_prediction(n_samples, *, rng, noise_var=1.0, horizon=1):
  clean = sinusoids(n_samples + horizon, frequencies=(0.1, 0.025), amplitudes=(-np.sqrt(2), np.sqrt(2)))
  observed, _ = add_noise(clean, rng=rng, noise_var=noise_var)
  batch = prediction(observed, horizon=horizon)
  batch.info.update({"clean_process": clean, "noise_var": noise_var, "horizon": horizon})
  return batch


def denoising(n_samples, *, rng, frequency=0.1, snr_db=-10.0):
  clean = sinusoids(n_samples, frequencies=(frequency, ))
  observed, noise_var = add_noise(clean, rng=rng, snr_db=snr_db)
  return SampleBatch(observed, clean, clean=clean, info={"noise_var": noise_var, "frequency": frequency})


def equalization(n_samples, *, rng, h, decision_delay=5, symbol_kind="qpsk", symbol_energy=1.0, noise_var=None,
                 snr_linear=None, prefix=0, noise_exponent=None):
  """Causal channel model r=h*s+n with d[k]=s[k-decision_delay].

  snr_linear is a linear power ratio (20 means about 13 dB).
  For spectral noise shaping, noise_var is the PSD multiplier, as in Problem 20.
  """
  if decision_delay < 0 or prefix < 0:
    raise ValueError("decision_delay and prefix must be nonnegative.")
  if (noise_var is None) == (snr_linear is None):
    raise ValueError("Specify exactly one of noise_var or snr_linear.")
  h = np.asarray(h)
  tx = symbols(n_samples + prefix, rng=rng, kind=symbol_kind, energy=symbol_energy)
  clean = lfilter(h, [1.0], tx)
  if snr_linear is not None:
    if snr_linear <= 0:
      raise ValueError("snr_linear must be positive.")
    noise_var = float(np.mean(np.abs(clean[prefix:])**2)) / snr_linear
  if noise_exponent is None:
    received, _ = add_noise(clean, rng=rng, noise_var=noise_var)
  else:
    if np.iscomplexobj(clean):
      raise ValueError("This spectral-noise example expects a real channel.")
    fft_size = 1 << (len(clean) - 1).bit_length()
    omega = 2 * np.pi * np.fft.fftfreq(fft_size)
    gain = np.sqrt(noise_var * np.abs(omega)**noise_exponent)
    white = rng.standard_normal(fft_size)
    received = clean + np.fft.ifft(np.fft.fft(white) * gain).real[:len(clean)]
  desired = delay(tx, decision_delay)
  valid = np.arange(prefix, prefix + n_samples) >= decision_delay
  return SampleBatch(
    received[prefix:], desired[prefix:], valid=valid, info={
      "h": h,
      "tx": tx,
      "noise_var": noise_var,
      "decision_delay": decision_delay,
      "symbol_energy": symbol_energy,
    })


def enhancement(n_samples, *, rng, reference_delay=10, noise_var=10.0, filter_b=(0.4, ), filter_a=(1.0, -1.36, 0.79),
                frequency=0.1):
  """Primary d=clean+H*noise, reference x=delay(clean+noise)."""
  clean = sinusoids(n_samples, frequencies=(frequency, ))
  noise = gaussian(n_samples, rng=rng, variance=noise_var)
  primary = clean + lfilter(filter_b, filter_a, noise)
  reference = delay(clean + noise, reference_delay)
  return SampleBatch(reference, primary, clean=clean, info={"reference_delay": reference_delay})


def tracking(n_samples, *, rng, initial_weights, coefficient_memory=0.999, coefficient_noise_var=0.033,
             input_variance=1.0, noise_var=0.01):
  """Time-varying stored weights w[k+1]=memory*w[k]+innovation[k]."""
  weights = np.asarray(initial_weights, dtype=np.float64).copy()
  x = gaussian(n_samples, rng=rng, variance=input_variance)
  noise = gaussian(n_samples, rng=rng, variance=noise_var)
  padded = np.pad(x, (len(weights) - 1, 0))
  desired = np.empty(n_samples)
  history = np.empty((n_samples, len(weights)))
  for k in range(n_samples):
    history[k] = weights
    desired[k] = weights @ padded[k:k + len(weights)][::-1] + noise[k]
    weights = coefficient_memory * weights + rng.normal(0, np.sqrt(coefficient_noise_var), len(weights))
  return SampleBatch(x, desired, true_weights=history, info={"noise_var": noise_var})


def matched_filter(n_samples, *, rng, equalizer, h, symbol_energy=1.0, noise_var=0.1, symbol_kind="qam16"):
  """Matched-filter model with colored noise R_nn=noise_var*Q, including two-sided signal taps."""
  desired = symbols(n_samples, rng=rng, kind=symbol_kind, energy=symbol_energy)
  clean = equalizer.matchedFilterSignal(desired)
  white = gaussian(n_samples, rng=rng, variance=noise_var, complex_data=True)
  colored = lfilter(np.asarray(h) / equalizer.h_norm, [1.0], white)
  return SampleBatch(clean + colored, desired, clean=clean, info={"noise_var": noise_var})


def pulse_channel(n_samples, *, rng, g_index, g, oversampling=2, symbol_energy=1.0, noise_var=0.1, symbol_kind="qam16"):
  """Pulse-shaped high-rate reception; each target has oversampling input samples.

  noise_var follows the continuous-time N0/2 convention of BaseFse. The variance
  of each independent high-rate noise sample is oversampling*noise_var.
  """
  from .equalization.pulses import sampleChannelOutput
  desired = symbols(n_samples, rng=rng, kind=symbol_kind, energy=symbol_energy)
  clean = sampleChannelOutput(desired, g_index, g, oversampling)
  received, _ = add_noise(clean, rng=rng, noise_var=oversampling * noise_var)
  return SampleBatch(received, desired, input_rate=oversampling, info={"noise_var": noise_var})
