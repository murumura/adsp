"""One calling style for direct, block, DFE, transform, and lattice experiments."""

from dataclasses import dataclass, field

import numpy as np

from .datasets import SampleBatch
from .statistics import normalized_lattice_error, tail_mean


@dataclass
class RunResult:
  name: str
  index: np.ndarray
  x: np.ndarray
  d: np.ndarray
  y: np.ndarray
  e: np.ndarray
  metric_error: np.ndarray
  valid: np.ndarray
  w_hist: np.ndarray | None = None
  w_final: np.ndarray | None = None
  msd: np.ndarray | None = None
  coefficient_space: str = "direct"
  extras: dict = field(default_factory=dict)

  @property
  def mse(self):
    return np.abs(self.metric_error)**2


@dataclass
class EnsembleResult:
  name: str
  index: np.ndarray
  mse: np.ndarray
  valid: np.ndarray
  n_trials: int
  representative: RunResult
  msd: np.ndarray | None = None
  w_mean: np.ndarray | None = None
  extras: dict = field(default_factory=dict)


def _lattice_run(model, batch, *, trace_keys=()):
  """Convert error definitions without altering the algorithm recursions."""
  from .filters.lattice import NormalizedLatticeRLS, APrioriLatticeRLS
  normalized = isinstance(model, NormalizedLatticeRLS)
  apriori = isinstance(model, APrioriLatticeRLS)
  x, desired = model.checkRealInput(batch.x, batch.d)
  y, error = np.empty(len(x)), np.empty(len(x))
  weights = None if normalized else np.empty((len(x) + 1, model.n_coef), dtype=np.complex128)
  if weights is not None:
    weights[0] = model.w
  traces = {key: [] for key in trace_keys}
  post_errors, normalized_errors = [], []
  for k, (xk, dk) in enumerate(zip(x, desired)):
    state = model.step(float(xk), float(dk))
    if normalized:
      error[k] = normalized_lattice_error(state, eps=model.eps)
      normalized_errors.append(state["e_bar"])
    elif apriori:
      error[k] = state["e"]
    else:
      post_errors.append(state["e"])
      quantizer = getattr(model, "coef_q", None)
      guard = max(model.eps, 1.0 / quantizer.scale if quantizer is not None else 0.0)
      error[k] = state["e"] / max(float(state["gamma"][-1]), guard)
    y[k] = dk - error[k]
    if weights is not None:
      weights[k + 1] = model.w
    for key in trace_keys:
      if key not in state:
        raise KeyError(f"{type(model).__name__} does not expose trace {key!r}.")
      traces[key].append(np.asarray(state[key]).copy())
  extras = {key: np.asarray(value) for key, value in traces.items()}
  if post_errors:
    extras["e_posterior"] = np.asarray(post_errors)
    extras["mse_posterior"] = np.abs(extras["e_posterior"])**2
  if normalized_errors:
    extras["e_normalized"] = np.asarray(normalized_errors)
    extras["mse_normalized"] = np.abs(extras["e_normalized"])**2
  return y, error, weights, extras


def _sliding_run(model, batch, *, adapt):
  y, error = np.empty(len(batch.d)), np.empty(len(batch.d))
  weights = np.empty((len(batch.d) + 1, model.n_coef))
  transformed = np.empty((len(batch.d), model.n_coef))
  weights[0] = model.w
  for k in range(len(batch.d)):
    y[k], error[k], transformed[k] = model.step(batch.x[k], batch.d[k], train=adapt)
    weights[k + 1] = model.w
  return y, error, weights, {"transform_samples": transformed}


def run(model, batch, *, name=None, adapt=True, metric="prediction", trace_keys=()):
  """Run one initialized model and return one result type.

  metric='prediction': |d-y|^2.
  metric='reconstruction': |(d-y)-clean|^2, for a noise canceller.
  Results use original sample indices; APA warm-up samples are omitted.
  Every MSE exposed here uses a physical a-priori error.
  Instances retain their learned state; compare() creates fresh instances per trial.
  """
  from .filters.lattice.base import BaseLatticeRLS
  from .filters.transform import SlidingDctLMS, TransformDomain
  from .filters.set_membership.partial_update import SMPartialUpdateAffineProjection
  from .filters.frequency_domain import OverLapSaveFDAF
  from .equalization.lattice import ErrorFeedbackLatticeRlsEq
  if not isinstance(batch, SampleBatch):
    raise TypeError("batch must be a SampleBatch.")
  if batch.input_rate != 1:
    raise ValueError("Adaptive models require input_rate=1; use evaluate() for fixed FSE models.")
  if metric not in ("prediction", "reconstruction"):
    raise ValueError("metric must be prediction or reconstruction.")
  coefficient_space = "direct"
  extras = {}
  if isinstance(model, BaseLatticeRLS):
    if not adapt:
      raise ValueError("Frozen lattice evaluation requires a reconstructed direct-form equalizer.")
    y, error, weights, extras = _lattice_run(model, batch, trace_keys=trace_keys)
    coefficient_space = "normalized_lattice" if weights is None else "lattice"
  elif isinstance(model, SlidingDctLMS):
    y, error, weights, extras = _sliding_run(model, batch, adapt=adapt)
    coefficient_space = "sliding_dct"
  else:
    if isinstance(model, ErrorFeedbackLatticeRlsEq):
      raw = model(batch.x.copy(), batch.d.copy(), train=adapt, reset=False)
    else:
      raw = model(batch.x.copy(), batch.d.copy(), train=adapt)
    if isinstance(raw, dict):
      y, error = raw["y"], raw["e"]
      weights = raw.get("w_hist")
      extras = {key: value for key, value in raw.items() if key not in ("y", "e", "w_hist", "w_final")}
    else:
      y, error = raw[:2]
      weights = raw[2] if len(raw) > 2 else getattr(model, "w_hist", None)
    if isinstance(model, SMPartialUpdateAffineProjection):
      # The legacy PUAP returns x^H w and conj(d)-x^H w.
      y, error = np.conj(y), np.conj(error)
    if hasattr(model, "q"):
      # Retain the quantized arithmetic error while evaluating the physical target.
      extras["e_quantized"] = np.asarray(error).copy()
      error = batch.d[len(batch.d) - len(y):] - y
    if isinstance(model, TransformDomain):
      # Its history already contains TH @ w_hat, i.e. direct-form stored weights.
      coefficient_space = "direct"
    if hasattr(model, "post_e_hist"):
      extras["e_posterior"] = np.asarray(model.post_e_hist).copy()
    if hasattr(model, "update_flags"):
      extras["update_flags"] = np.asarray(model.update_flags).copy()
    if hasattr(model, "selected_count_hist"):
      extras["selected_count"] = np.asarray(model.selected_count_hist).copy()
  y, error = np.asarray(y), np.asarray(error)
  if y.ndim != 1 or y.shape != error.shape or len(y) > len(batch.d):
    raise ValueError("Model output and error must be matching vectors no longer than the input.")
  offset = len(batch.d) - len(y)
  index = np.arange(offset, len(batch.d))
  desired = batch.d[index]
  if not np.all(np.isfinite(y)) or not np.all(np.isfinite(error)):
    raise FloatingPointError(f"{name or type(model).__name__} produced nonfinite output.")
  if not np.allclose(desired - y, error, rtol=2e-4, atol=2e-5):
    raise ValueError("Output/error convention is inconsistent with d-y.")
  if weights is not None:
    weights = np.asarray(weights)
    if weights.shape[0] != len(y) + 1 or not np.all(np.isfinite(weights)):
      raise ValueError("Weight history must contain the initial state and one state per output sample.")
  metric_error = error.copy()
  if metric == "reconstruction":
    if batch.clean is None:
      raise ValueError("reconstruction requires batch.clean.")
    metric_error -= batch.clean[index]
  final_weights = None if weights is None else weights[-1].copy()
  msd = None
  if isinstance(model, OverLapSaveFDAF):
    coefficient_space = "physical_fir"
  if batch.true_weights is not None and coefficient_space == "direct" and weights is not None:
    reference = batch.true_weights
    if reference.ndim == 2:
      reference = reference[index]
    if reference.shape[-1] != weights.shape[-1]:
      raise ValueError("true_weights and model coefficients have different lengths.")
    msd = np.sum(np.abs(weights[1:] - reference)**2, axis=1)
  return RunResult(name or type(model).__name__, index, batch.x[index], desired, y, error, metric_error,
                   batch.valid[index], weights, final_weights, msd, coefficient_space, extras)


def compare(models, make_data, *, n_trials=1, seed=0, metric="prediction", trace_keys=(), transforms=None):
  """Fair Monte Carlo comparison with fresh model state and shared data for every trial.

  models: {label: zero-argument factory}.
  make_data: callable(rng=generator) -> SampleBatch.
  transforms: optional {label: callable(batch)->SampleBatch}, e.g. prewhitening.
  Stores aggregate curves and one representative trial, never files or all trials.
  """
  if not models or n_trials < 1:
    raise ValueError("Provide at least one model factory and a positive trial count.")
  if not all(callable(factory) for factory in models.values()):
    raise TypeError("Every model entry must be a zero-argument factory.")
  children = np.random.SeedSequence(seed).spawn(n_trials)
  aggregate = {}
  for trial, child in enumerate(children):
    batch = make_data(rng=np.random.default_rng(child))
    for label, factory in models.items():
      model = factory()
      prepared = transforms[label](batch) if transforms and label in transforms else batch
      result = run(model, prepared, name=label, metric=metric, trace_keys=trace_keys)
      if trial == 0:
        aggregate[label] = EnsembleResult(
          label,
          result.index.copy(),
          np.zeros_like(result.mse),
          result.valid.copy(),
          n_trials,
          result,
          None if result.msd is None else np.zeros_like(result.msd),
          None if result.w_hist is None else np.zeros_like(result.w_hist),
          {},
        )
      total = aggregate[label]
      if not np.array_equal(result.index, total.index) or not np.array_equal(result.valid, total.valid):
        raise ValueError("Each trial must have the same output index and validity mask for a given model.")
      total.mse += result.mse / n_trials
      if total.msd is not None:
        total.msd += result.msd / n_trials
      if total.w_mean is not None:
        total.w_mean += result.w_hist / n_trials
      for key, value in result.extras.items():
        values = np.asarray(value)
        if values.dtype.kind not in "biufc":
          continue
        if key not in total.extras:
          total.extras[key] = np.zeros_like(values, dtype=np.result_type(values, np.float64))
        total.extras[key] += values / n_trials
  return aggregate


def evaluate(model, batch, *, name=None, guard=0):
  """Fixed SSE/FSE application using the same RunResult and plotting API."""
  from .equalization.fse import BaseFse
  if batch.input_rate != 1 and not isinstance(model, BaseFse):
    raise ValueError("Only FSE models support multirate batches.")
  if isinstance(model, BaseFse) and batch.input_rate != model.oversampling:
    raise ValueError("The batch input_rate must match the FSE oversampling factor.")
  output = model(batch.x, n_symbols=len(batch.d)) if isinstance(model, BaseFse) else model(batch.x)
  output = np.asarray(output)
  if output.shape != batch.d.shape or guard < 0 or 2 * guard >= len(output):
    raise ValueError("Output must match d and guard must leave at least one valid sample.")
  valid = batch.valid.copy()
  valid[:guard] = False
  if guard:
    valid[-guard:] = False
  error = batch.d - output
  return RunResult(name or type(model).__name__, np.arange(len(output)), batch.x[::batch.input_rate], batch.d, output,
                   error, error, valid, coefficient_space="physical_fir")


def fit_inband(model, i_stream, q_stream, desired, *, start=140, stop=3200, epochs=3, name="Nyquist inband"):
  """Supervised inband training followed by fixed-coefficient evaluation of the full record.

  valid marks only held-out symbols after stop. This result reports evaluation
  error, while extras['training_mse'] contains the sparse update trace.
  """
  desired = np.asarray(desired)
  model.fitNlms(i_stream, q_stream, desired, start=start, stop=stop, epochs=epochs)
  output = model.apply(i_stream, q_stream, len(desired))
  error = desired - output
  index = np.arange(len(desired))
  valid = (index >= stop + 200) & (index < len(desired) - 200)
  return RunResult(name, index, np.column_stack([i_stream[:len(desired)],
                                                 q_stream[:len(desired)]]), desired, output, error, error, valid,
                   coefficient_space="inband_real_paths", extras={"training_mse": np.asarray(model.trace)})


def summary(results, *, tail=200, noise_floor=None):
  """Print measured statistics; normalized errors are kept separate from physical MSE."""
  print(f"{'Model':<34} {'Tail MSE':>12} {'Tail MSD':>12} {'Update rate':>12}")
  for name, result in results.items():
    values = result.mse[result.valid]
    if not len(values):
      raise ValueError(f"{name} has no valid samples.")
    mse = tail_mean(values, tail=tail)
    msd = "—" if result.msd is None else f"{tail_mean(result.msd[result.valid], tail=tail):.5g}"
    update = result.extras.get("update_flags")
    update_rate = "—" if update is None else f"{100 * np.mean(update):.2f}%"
    print(f"{name:<34} {mse:12.5g} {msd:>12} {update_rate:>12}")
    if noise_floor is not None:
      if noise_floor <= 0:
        raise ValueError("noise_floor must be positive.")
      print(f"  measured misadjustment: {(mse - noise_floor) / noise_floor:.6g}")
