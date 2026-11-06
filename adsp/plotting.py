"""Dark plotting with consistent labels, colors, scales, and display-only output."""

import hashlib

import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from scipy.signal import freqz

from . import statistics as stats

BACKGROUND = "#0e1117"
PANEL = "#161b22"
FOREGROUND = "#e6edf3"
GRID = "#303745"
PALETTE = ["#58a6ff", "#2dd4bf", "#f472b6", "#facc15", "#a3e635", "#fb7185", "#c084fc", "#fb923c"]


def use_dark(*, font_size=11):
  """Call once in setup; every helper also applies the theme before making a figure."""
  plt.style.use("dark_background")
  plt.rcParams.update({
    "figure.facecolor": BACKGROUND,
    "axes.facecolor": BACKGROUND,
    "axes.edgecolor": GRID,
    "axes.labelcolor": FOREGROUND,
    "text.color": FOREGROUND,
    "xtick.color": FOREGROUND,
    "ytick.color": FOREGROUND,
    "grid.color": GRID,
    "legend.facecolor": PANEL,
    "legend.edgecolor": GRID,
    "font.size": font_size,
    "axes.titlesize": font_size + 1,
    "axes.labelsize": font_size,
    "legend.fontsize": max(font_size - 1, 8),
    "axes.prop_cycle": cycler(color=PALETTE),
    "lines.linewidth": 1.6,
    "figure.constrained_layout.use": True,
    "figure.max_open_warning": 0,
  })


def color(name):
  """The same model label always has the same visible color, independent of plot order."""
  preferred = {
    "LMS": 0,
    "NLMS": 1,
    "RLS": 2,
    "RLSAlt": 3,
    "Sign-Error LMS": 4,
    "Sign-Data LMS": 5,
    "Sign-Sign LMS": 6,
    "Power-of-Two LMS": 7
  }
  index = preferred.get(name, int(hashlib.sha256(name.encode()).hexdigest()[:8], 16) % len(PALETTE))
  return PALETTE[index]


def axes(*, rows=1, cols=1, figsize=None):
  use_dark()
  return plt.subplots(rows, cols, figsize=figsize or (7.5 * cols, 4.0 * rows), squeeze=False)


def _legend(ax):
  labels = ax.get_legend_handles_labels()[1]
  if len(labels) > 6:
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
  elif labels:
    ax.legend()


def style(ax, *, title="", xlabel="Sample index", ylabel="Amplitude", legend=True):
  ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
  ax.grid(True, which="both", alpha=0.5)
  if legend:
    _legend(ax)
  return ax


def lines(series, *, x=None, ax=None, title="", xlabel="Sample index", ylabel="Amplitude", scale="linear", window=1,
          component="real"):
  """series={label:y} or {label:(x,y)}; linear complex data use the selected component."""
  if ax is None:
    _, grid = axes()
    ax = grid[0, 0]
  if scale not in ("linear", "log", "db", "amplitude_db"):
    raise ValueError("scale must be linear, log, db, or amplitude_db.")
  if component not in ("real", "imag", "magnitude"):
    raise ValueError("component must be real, imag, or magnitude.")
  used_styles = set()
  for label, value in series.items():
    if isinstance(value, tuple):
      axis, values = value
    else:
      values = np.asarray(value)
      axis = np.arange(len(values)) if x is None else x
    values = stats.smooth(values, window=window) if window > 1 else np.asarray(values)
    if scale == "db":
      values = stats.power_db(values)
    elif scale == "amplitude_db":
      values = stats.amplitude_db(values)
    elif scale == "log":
      values = np.maximum(np.real(values), 1e-18)
    elif np.iscomplexobj(values):
      values = {"real": np.real, "imag": np.imag, "magnitude": np.abs}[component](values)
    line_color = color(label)
    line_style = next((candidate for candidate in ("-", "--", "-.", ":") if (line_color, candidate) not in used_styles),
                      ":")
    used_styles.add((line_color, line_style))
    ax.plot(axis, values, label=label, color=line_color, linestyle=line_style)
  if scale == "log":
    ax.set_yscale("log")
  return style(ax, title=title, xlabel=xlabel, ylabel=ylabel)


def learning(results, *, quantity="mse", ax=None, title="Learning curves", scale="db", window=31, noise_floor=None,
             subtract_floor=0.0):
  """Plot physical MSE or direct-form MSD at each model's actual output index."""
  series = {}
  for label, result in results.items():
    values = getattr(result, quantity, None)
    if values is None:
      continue
    series[label] = (result.index[result.valid], values[result.valid] - subtract_floor)
  if not series:
    raise ValueError(f"No result supplies {quantity}.")
  ylabel = quantity.upper() + (" (dB)" if scale == "db" else "")
  ax = lines(series, ax=ax, title=title, xlabel="Iteration k", ylabel=ylabel, scale=scale, window=window)
  if noise_floor is not None:
    level = stats.power_db(noise_floor) if scale == "db" else noise_floor
    ax.axhline(level, color=FOREGROUND, linestyle=":", alpha=0.8, label="Reference floor")
    _legend(ax)
  return ax


def stem(series, *, x=None, ax=None, title="", xlabel="Tap index", ylabel="Amplitude"):
  if ax is None:
    _, grid = axes()
    ax = grid[0, 0]
  for i, (label, values) in enumerate(series.items()):
    values = np.asarray(values)
    index = np.arange(len(values)) if x is None else np.asarray(x)
    offset = i * 0.12 if len(series) > 1 else 0.0
    marker, sticks, baseline = ax.stem(index + offset, values.real, label=label, basefmt=" ")
    plt.setp(marker, color=color(label), markersize=4)
    plt.setp(sticks, color=color(label), alpha=0.8)
  return style(ax, title=title, xlabel=xlabel, ylabel=ylabel)


def coefficients(result, *, reference=None, title="Coefficient trajectories", component="real", ax=None):
  weights = getattr(result, "w_mean", None)
  if weights is None:
    weights = result.w_hist
  if weights is None:
    raise ValueError("This result does not expose physical coefficient trajectories.")
  if component not in ("real", "imag", "magnitude"):
    raise ValueError("component must be real, imag, or magnitude.")
  transform = {"real": np.real, "imag": np.imag, "magnitude": np.abs}[component]
  axis = np.r_[result.index[0] - 1, result.index]
  ax = lines({f"tap {i}": (axis, transform(weights[:, i]))
              for i in range(weights.shape[1])}, ax=ax, title=title, xlabel="Iteration k",
             ylabel=f"Coefficient ({component})")
  if reference is not None:
    for i, value in enumerate(transform(np.asarray(reference))):
      ax.axhline(value, color=color(f"tap {i}"), linestyle=":", alpha=0.8)
  return ax


def constellation(series, *, start=0, title="Constellation", limit=6000):
  fig, grid = axes(cols=len(series), figsize=(5.0 * len(series), 4.6))
  for ax, (label, values) in zip(grid[0], series.items()):
    values = np.asarray(values)[start:]
    stride = max(1, len(values) // limit)
    ax.scatter(values.real[::stride], values.imag[::stride], s=4, alpha=0.35, color=color(label))
    ax.set_aspect("equal", adjustable="datalim")
    style(ax, title=label, xlabel="In-phase", ylabel="Quadrature", legend=False)
  fig.suptitle(title)
  return fig


def frequency_response(taps, *, title="Frequency response", ax=None, n_fft=2048):
  series = {}
  for label, value in taps.items():
    freq, response = freqz(value, worN=n_fft, fs=1.0)
    series[label] = (freq, response)
  return lines(series, ax=ax, title=title, xlabel="Frequency (cycles/sample)", ylabel="Magnitude (dB)",
               scale="amplitude_db")


def spectrum(series, *, title="Power spectral density", ax=None, n_fft=1024, segment_length=256):
  powers = {label: stats.psd(value, n_fft=n_fft, segment_length=segment_length) for label, value in series.items()}
  return lines(powers, ax=ax, title=title, xlabel="Frequency (cycles/sample)", ylabel="PSD (dB / cycles/sample)",
               scale="db")


def zeros(taps, *, title="Residual-filter zeros", ax=None):
  if ax is None:
    _, grid = axes(figsize=(5.5, 5.0))
    ax = grid[0, 0]
  theta = np.linspace(0, 2 * np.pi, 512)
  ax.plot(np.cos(theta), np.sin(theta), color=FOREGROUND, linestyle=":", alpha=0.5)
  for label, values in taps.items():
    roots = np.roots(values)
    ax.scatter(roots.real, roots.imag, color=color(label), label=label, s=35)
  ax.set_aspect("equal")
  return style(ax, title=title, xlabel="Real", ylabel="Imaginary")


def show(fig=None):
  """Display figures. This module deliberately provides no result-export API."""
  if fig is not None:
    fig.canvas.draw_idle()
  plt.show()
