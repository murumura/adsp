#!/usr/bin/env python3
"""
demo_whitening.py

Illustrate the effect of whitening on:
  1. Autocorrelation
  2. Power spectral density
  3. Input covariance eigenvalue spread / condition number
  4. LMS system-identification convergence
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np

try:
  from adsp import adspwhitener
except ImportError:
  import adspwhitener


@dataclass
class WhiteningDemoConfig:
  n_samples: int = 4096
  ar_pole: float = 0.92
  ar_order: int = 1
  max_lag: int = 30
  lms_trials: int = 24
  lms_samples: int = 1800
  lms_mu: float = 0.004
  lms_taps: int = 16
  psd_nfft: int = 1024
  psd_seg_len: int = 256
  spectral_smooth_len: int = 17
  seed: int = 7
  out_dir: Path = Path("whitening_assets")


def styleAxes(ax, title: str, xlabel: str = "Sample index", ylabel: str = "Amplitude"):
  ax.set_title(title, fontsize=11, weight="bold")
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  ax.grid(True, alpha=0.25)


def saveFigure(fig, path: Path):
  fig.tight_layout()
  fig.savefig(path, dpi=180, bbox_inches="tight")
  plt.close(fig)


def rmsNorm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
  x = np.asarray(x)
  return x / max(float(np.sqrt(np.mean(np.abs(x) ** 2))), eps)


def smoothCurve(x: np.ndarray, win: int = 31) -> np.ndarray:
  x = np.asarray(x, dtype=np.float64)
  if win <= 1:
    return x.copy()
  return np.convolve(x, np.ones(win, dtype=np.float64) / win, mode="same")


def powerToDb(x: np.ndarray, floor: float = 1e-18) -> np.ndarray:
  return 10.0 * np.log10(np.maximum(np.asarray(x, dtype=np.float64), floor))


def generateAr1(n_samples: int, pole: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
  if not 0.0 <= abs(pole) < 1.0:
    raise ValueError("AR(1) pole must satisfy abs(pole) < 1.")
  rng = np.random.default_rng(seed)
  u = rng.standard_normal(n_samples).astype(np.float32)
  x = np.zeros(n_samples, dtype=np.float32)
  for k in range(1, n_samples):
    x[k] = pole * x[k - 1] + u[k]
  return x, u


def normalizedAutocorr(x: np.ndarray, max_lag: int) -> np.ndarray:
  x = np.asarray(x, dtype=np.complex64).reshape(-1)
  x = x - np.mean(x)
  denom = float(np.real(np.vdot(x, x))) + 1e-12
  r = np.zeros(max_lag + 1, dtype=np.complex64)
  r[0] = 1.0
  for lag in range(1, max_lag + 1):
    r[lag] = np.vdot(x[:-lag], x[lag:]) / denom
  return r


def averagePsd(x: np.ndarray, nfft: int, seg_len: int) -> Tuple[np.ndarray, np.ndarray]:
  x = np.asarray(x, dtype=np.float64).reshape(-1) - np.mean(x)
  if seg_len > len(x):
    seg_len = len(x)
  if seg_len < 8:
    raise ValueError("Need at least eight samples for PSD estimation.")
  step = max(seg_len // 2, 1)
  window = np.hanning(seg_len)
  window_energy = np.sum(window ** 2)
  psd = np.zeros(nfft // 2 + 1, dtype=np.float64)
  count = 0
  for start in range(0, len(x) - seg_len + 1, step):
    seg = x[start:start + seg_len] * window
    psd += (np.abs(np.fft.rfft(seg, n=nfft)) ** 2) / max(window_energy, 1e-12)
    count += 1
  psd = (psd / max(count, 1)) / (np.max(psd) + 1e-12)
  return np.linspace(0.0, 0.5, len(psd)), powerToDb(psd)


def estimateCovariance(x: np.ndarray, n_taps: int) -> np.ndarray:
  x = np.asarray(x, dtype=np.complex64).reshape(-1)
  if len(x) < n_taps:
    raise ValueError("Input must be longer than n_taps.")
  n_rows = len(x) - n_taps + 1
  regressors = np.zeros((n_rows, n_taps), dtype=np.complex64)
  for row in range(n_rows):
    regressors[row] = x[row:row + n_taps][::-1]
  return (regressors.conj().T @ regressors) / n_rows


def condnumFromCovar(covariance: np.ndarray) -> Tuple[np.ndarray, float]:
  eig = np.sort(np.maximum(np.linalg.eigvalsh(covariance).real, 1e-12))[::-1]
  return eig, float(eig[0] / eig[-1])


def runLmsSystemId(x: np.ndarray, d: np.ndarray, h_true: np.ndarray, mu: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  x, d, h_true = map(lambda a: np.asarray(a, dtype=np.complex64).reshape(-1), (x, d, h_true))
  if len(x) != len(d):
    raise ValueError("x and d must have the same length.")
  n_taps, n_samples = len(h_true), len(x)
  x_pad = np.concatenate([np.zeros(n_taps - 1, dtype=np.complex64), x])
  w = np.zeros(n_taps, dtype=np.complex64)
  mse, misalignment = np.zeros(n_samples, dtype=np.float64), np.zeros(n_samples, dtype=np.float64)
  h_norm = float(np.linalg.norm(h_true)) + 1e-12

  for k in range(n_samples):
    reg = x_pad[k:k + n_taps][::-1]
    e = d[k] - np.vdot(w, reg)
    w = (w + (2.0 * mu) * np.conj(e) * reg).astype(np.complex64)
    mse[k] = float(np.abs(e) ** 2)
    misalignment[k] = float(np.linalg.norm(w - h_true) / h_norm)
  return mse, misalignment, w


def runLmsMonteCarlo(config: WhiteningDemoConfig, h_true: np.ndarray) -> Dict[str, np.ndarray]:
  raw_mse_hist, white_mse_hist, raw_mis_hist, white_mis_hist = [], [], [], []
  for trial in range(config.lms_trials):
    x, _ = generateAr1(n_samples=config.lms_samples, pole=config.ar_pole, seed=config.seed + 1000 + trial)
    x = rmsNorm(x).astype(np.complex64)
    d = np.convolve(x, h_true, mode="full")[:len(x)]
    x_white, d_white, _ = adspwhitener.ArWhitener(ar_order=config.ar_order).fitWhitenPair(x=x, d=d, normalize=True)
    
    raw_mse, raw_mis, _ = runLmsSystemId(x=x, d=d, h_true=h_true, mu=config.lms_mu)
    white_mse, white_mis, _ = runLmsSystemId(x=x_white, d=d_white, h_true=h_true, mu=config.lms_mu)
    
    raw_mse_hist.append(raw_mse); white_mse_hist.append(white_mse)
    raw_mis_hist.append(raw_mis); white_mis_hist.append(white_mis)
  return {
    "raw_mse": np.mean(raw_mse_hist, axis=0), "white_mse": np.mean(white_mse_hist, axis=0),
    "raw_mis": np.mean(raw_mis_hist, axis=0), "white_mis": np.mean(white_mis_hist, axis=0),
  }


def drawSignalProperties(config: WhiteningDemoConfig, x: np.ndarray, innovation: np.ndarray, 
                         x_ar: np.ndarray, x_spectral: np.ndarray, spectral_gain: np.ndarray):
  fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.5))
  sample_stop = min(350, len(x))
  axes[0, 0].plot(x[:sample_stop], label="Colored AR(1) x[k]")
  axes[0, 0].plot(innovation[:sample_stop], label="White innovation u[k]", alpha=0.8)
  styleAxes(axes[0, 0], "Time-domain signal", ylabel="Amplitude"); axes[0, 0].legend()

  lag = np.arange(config.max_lag + 1)
  axes[0, 1].stem(lag, np.real(normalizedAutocorr(x, config.max_lag)), label="Colored input", basefmt=" ")
  axes[0, 1].stem(lag + 0.12, np.real(normalizedAutocorr(x_ar, config.max_lag)), label="AR-whitened", basefmt=" ")
  axes[0, 1].stem(lag + 0.24, np.real(normalizedAutocorr(x_spectral, config.max_lag)), label="Spectral-whitened", basefmt=" ")
  styleAxes(axes[0, 1], "Autocorrelation after whitening", xlabel="Lag", ylabel="Normalized Rxx"); axes[0, 1].legend()

  freq_x, psd_x = averagePsd(x, nfft=config.psd_nfft, seg_len=config.psd_seg_len)
  freq_ar, psd_ar = averagePsd(np.real(x_ar), nfft=config.psd_nfft, seg_len=config.psd_seg_len)
  freq_spec, psd_spec = averagePsd(np.real(x_spectral), nfft=config.psd_nfft, seg_len=config.psd_seg_len)
  axes[1, 0].plot(freq_x, psd_x, label="Colored input")
  axes[1, 0].plot(freq_ar, psd_ar, label="AR-whitened")
  axes[1, 0].plot(freq_spec, psd_spec, label="Spectral-whitened")
  styleAxes(axes[1, 0], "Power spectral density", xlabel="Normalized frequency", ylabel="PSD (dB, normalized)"); axes[1, 0].legend()

  axes[1, 1].plot(np.linspace(0.0, 0.5, len(spectral_gain)), spectral_gain)
  styleAxes(axes[1, 1], "Spectral-whitening gain", xlabel="Normalized frequency", ylabel="Gain")
  saveFigure(fig, config.out_dir / "01_signal_properties.png")


def drawWhiteningComparison(config: WhiteningDemoConfig, x: np.ndarray, x_diff: np.ndarray, x_ar: np.ndarray, x_spectral: np.ndarray):
  fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.5))
  lag = np.arange(config.max_lag + 1)
  axes[0, 0].stem(lag, np.real(normalizedAutocorr(x, config.max_lag)), label="Original", basefmt=" ")
  axes[0, 0].stem(lag + 0.12, np.real(normalizedAutocorr(x_diff, config.max_lag)), label="First difference", basefmt=" ")
  styleAxes(axes[0, 0], "Autocorrelation: first difference", xlabel="Lag", ylabel="Normalized Rxx"); axes[0, 0].legend()

  axes[0, 1].stem(lag, np.real(normalizedAutocorr(x, config.max_lag)), label="Original", basefmt=" ")
  axes[0, 1].stem(lag + 0.12, np.real(normalizedAutocorr(x_ar, config.max_lag)), label="AR whitening", basefmt=" ")
  styleAxes(axes[0, 1], "Autocorrelation: AR prediction-error whitening", xlabel="Lag", ylabel="Normalized Rxx"); axes[0, 1].legend()

  freq_x, psd_x = averagePsd(x, nfft=config.psd_nfft, seg_len=config.psd_seg_len)
  freq_diff, psd_diff = averagePsd(np.real(x_diff), nfft=config.psd_nfft, seg_len=config.psd_seg_len)
  freq_ar, psd_ar = averagePsd(np.real(x_ar), nfft=config.psd_nfft, seg_len=config.psd_seg_len)
  freq_spec, psd_spec = averagePsd(np.real(x_spectral), nfft=config.psd_nfft, seg_len=config.psd_seg_len)

  axes[1, 0].plot(freq_x, psd_x, label="Original")
  axes[1, 0].plot(freq_diff, psd_diff, label="First difference")
  styleAxes(axes[1, 0], "PSD: first-difference whitening", xlabel="Normalized frequency", ylabel="PSD (dB, normalized)"); axes[1, 0].legend()

  axes[1, 1].plot(freq_x, psd_x, label="Original")
  axes[1, 1].plot(freq_ar, psd_ar, label="AR whitening")
  axes[1, 1].plot(freq_spec, psd_spec, label="Spectral whitening")
  styleAxes(axes[1, 1], "PSD: AR vs spectral whitening", xlabel="Normalized frequency", ylabel="PSD (dB, normalized)"); axes[1, 1].legend()
  saveFigure(fig, config.out_dir / "02_whitening_comparison.png")


def drawLmsResults(config: WhiteningDemoConfig, lms_result: Dict[str, np.ndarray]):
  fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.7))
  axes[0].plot(powerToDb(smoothCurve(lms_result["raw_mse"])), label="Colored input")
  axes[0].plot(powerToDb(smoothCurve(lms_result["white_mse"])), label="AR-prewhitened input")
  styleAxes(axes[0], "LMS learning curve", xlabel="Iteration", ylabel="Instantaneous MSE (dB)"); axes[0].legend()

  axes[1].plot(powerToDb(smoothCurve(lms_result["raw_mis"] ** 2)), label="Colored input")
  axes[1].plot(powerToDb(smoothCurve(lms_result["white_mis"] ** 2)), label="AR-prewhitened input")
  styleAxes(axes[1], "LMS system-identification misalignment", xlabel="Iteration", ylabel="Relative parameter-error power (dB)"); axes[1].legend()
  saveFigure(fig, config.out_dir / "03_lms_convergence.png")


def drawCovarianceEigenvalues(config: WhiteningDemoConfig, eig_raw: np.ndarray, eig_white: np.ndarray):
  fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.7))
  tap_index = np.arange(1, len(eig_raw) + 1)
  axes[0].bar(tap_index, eig_raw)
  styleAxes(axes[0], "Eigenvalues of R_x: colored input", xlabel="Eigenvalue index", ylabel="Eigenvalue")
  axes[1].bar(tap_index, eig_white)
  styleAxes(axes[1], "Eigenvalues of R_x: AR-prewhitened input", xlabel="Eigenvalue index", ylabel="Eigenvalue")
  saveFigure(fig, config.out_dir / "04_covariance_eigenvalues.png")


def makeMetrics(config: WhiteningDemoConfig, x: np.ndarray, x_diff: np.ndarray, x_ar: np.ndarray, 
                x_spectral: np.ndarray, ar_whitener, cond_raw: float, cond_white: float, 
                lms_result: Dict[str, np.ndarray]) -> Dict[str, object]:
  tail = 200
  return {
    "ar1_true_pole": float(config.ar_pole),
    "estimated_ar_coef": [complex(v) for v in ar_whitener.ar_coef],
    "ar_whitening_coef": [complex(v) for v in ar_whitener.coef],
    "raw_lag1_autocorr": float(np.real(normalizedAutocorr(x, 1)[1])),
    "difference_lag1_autocorr": float(np.real(normalizedAutocorr(x_diff, 1)[1])),
    "ar_lag1_autocorr": float(np.real(normalizedAutocorr(x_ar, 1)[1])),
    "spectral_lag1_autocorr": float(np.real(normalizedAutocorr(x_spectral, 1)[1])),
    "colored_condition_number": float(cond_raw),
    "prewhitened_condition_number": float(cond_white),
    "final_colored_mse_db": float(powerToDb(np.mean(lms_result["raw_mse"][-tail:]))),
    "final_prewhitened_mse_db": float(powerToDb(np.mean(lms_result["white_mse"][-tail:]))),
    "final_colored_misalignment_db": float(powerToDb(np.mean(lms_result["raw_mis"][-tail:] ** 2))),
    "final_prewhitened_misalignment_db": float(powerToDb(np.mean(lms_result["white_mis"][-tail:] ** 2))),
  }


def writeMetrics(metrics: Dict[str, object], path: Path):
  path.write_text("\n".join(f"{k}={v}" for k, v in metrics.items()) + "\n", encoding="utf-8")


def runDemo(config: WhiteningDemoConfig):
  config.out_dir.mkdir(parents=True, exist_ok=True)
  x, innovation = generateAr1(n_samples=config.n_samples, pole=config.ar_pole, seed=config.seed)

  x_diff = adspwhitener.FirWhitener(init_coef=np.array([1.0, -1.0], dtype=np.complex64))(x)
  ar_whitener = adspwhitener.ArWhitener(ar_order=config.ar_order)
  x_ar, _ = ar_whitener.fitTransform(x, normalize=True)

  spectral_whitener = adspwhitener.SpectralWhitener(
    fft_len=config.n_samples, smooth_len=config.spectral_smooth_len,
    smooth_psd=True, max_gain=32.0, normalize_output=True
  )
  x_spectral = spectral_whitener.fitTransform(x)
  spectral_gain = spectral_whitener.getGain()

  cov_raw = estimateCovariance(rmsNorm(x), n_taps=config.lms_taps)
  cov_white = estimateCovariance(rmsNorm(x_ar), n_taps=config.lms_taps)
  eig_raw, cond_raw = condnumFromCovar(cov_raw)
  eig_white, cond_white = condnumFromCovar(cov_white)

  h_true = np.array([0.45, -0.25, 0.18, 0.12, -0.08, 0.05, 0.03, -0.02,
                     0.015, -0.01, 0.008, 0.0, -0.006, 0.004, 0.002, -0.001], dtype=np.complex64)
  lms_result = runLmsMonteCarlo(config=config, h_true=h_true)

  drawSignalProperties(config, x, innovation, x_ar, x_spectral, spectral_gain)
  drawWhiteningComparison(config, x, x_diff, x_ar, x_spectral)
  drawLmsResults(config, lms_result)
  drawCovarianceEigenvalues(config, eig_raw, eig_white)

  metrics = makeMetrics(config, x, x_diff, x_ar, x_spectral, ar_whitener, cond_raw, cond_white, lms_result)
  writeMetrics(metrics, config.out_dir / "metrics.txt")

  print("\n" + "=" * 72 + "\nWHITENING DEMO SUMMARY\n" + "=" * 72)
  print(f"AR(1) true pole:               {config.ar_pole:.6f}")
  print(f"Estimated AR coefficient:      {np.real(ar_whitener.ar_coef[0]):.6f}")
  print(f"Raw lag-1 autocorrelation:     {metrics['raw_lag1_autocorr']:.6f}")
  print(f"AR-white lag-1 autocorrelation:{metrics['ar_lag1_autocorr']:.6f}")
  print(f"Spectral lag-1 autocorrelation:{metrics['spectral_lag1_autocorr']:.6f}")
  print(f"Condition number, colored:     {cond_raw:.3f}\nCondition number, whitened:    {cond_white:.3f}")
  print(f"Final raw LMS misalignment:    {metrics['final_colored_misalignment_db']:.2f} dB")
  print(f"Final white LMS misalignment:  {metrics['final_prewhitened_misalignment_db']:.2f} dB")
  print(f"\nAssets written to: {config.out_dir.resolve()}\n" + "=" * 72)
  return metrics


def parseArgs() -> WhiteningDemoConfig:
  parser = argparse.ArgumentParser(description="Illustrate whitening and its effect on LMS convergence.")
  parser.add_argument("--out-dir", default="whitening_assets", help="Output directory for files.")
  parser.add_argument("--n-samples", type=int, default=4096, help="Samples for correlation/PSD figures.")
  parser.add_argument("--lms-trials", type=int, default=24, help="Monte-Carlo trials for LMS curves.")
  parser.add_argument("--lms-samples", type=int, default=1800, help="Samples per LMS trial.")
  parser.add_argument("--ar-pole", type=float, default=0.92, help="AR(1) pole to generate colored input.")
  parser.add_argument("--lms-mu", type=float, default=0.004, help="LMS step size.")
  args = parser.parse_args()
  return WhiteningDemoConfig(n_samples=args.n_samples, ar_pole=args.ar_pole, lms_trials=args.lms_trials,
                             lms_samples=args.lms_samples, lms_mu=args.lms_mu, out_dir=Path(args.out_dir))


if __name__ == "__main__":
  runDemo(parseArgs())