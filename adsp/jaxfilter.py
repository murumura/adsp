import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import scipy
from scipy import signal
from typing import Callable, Tuple, Dict, Any, Optional, Union
# Plotting and Analysis Functions
import matplotlib.pyplot as plt
from matplotlib import gridspec
from abc import ABC, abstractmethod

_Array = Union[np.ndarray, jnp.ndarray]


def dtype_to_real(dtype):
  return dtype.type(0).real.dtype


def dtype_to_complex(dtype):
  return (1j * dtype.type(0)).dtype


def build_regressor(x: _Array, order: int):
  """
  Builds a regressor matrix where each row corresponds to a time instant
  and contains the current and past input samples.
  
  Args:
      x: Input signal (1D array)
      order: Number of filter coefficients (regressor length)
  
  Returns:
      X: Regressor matrix of shape (N-order+1, order)
          Each row: [x(k), x(k-1), ..., x(k-order+1)]
  """
  N = len(x)
  n_coef = order
  # Create regressor matrix using slicing
  # For i=0: x[n_coef-1: N]        -> x(k) at each time
  # For i=1: x[n_coef-2: N-1]      -> x(k-1) at each time
  # For i=order-1: x[0: N-n_coef+1] -> x(k-order+1) at each time
  X = jnp.stack([x[n_coef - 1 - i:N - i] for i in range(n_coef)], axis=1)
  return X  # shape (N-n_coef+1, n_coef)


# Helper: compute empirical R and eigenvalue spread for regressor of length M
# This is a theoretical implmentation for all kind of problems
def empirical_R_and_spread(x: _Array, order: int):
  """
  Compute empirical autocorrelation matrix and its eigenvalue spread.
  where matrix R = E[x(k)x^T(k)]
  Args:
    x: Input signal
    order: Filter order
  
  Returns:
    R: Autocorrelation matrix (order x order)
    eigs: Eigenvalues of R (sorted)
    cond: Condition number = max(eig)/min(eig) = eigenvalue spread
  """
  X = build_regressor(x, order)
  R = (X.T @ X) / X.shape[0]  # Empirical autocorrelation
  eigs = jnp.linalg.eigvalsh(R)  # Eigenvalues (hermitian, so real)
  cond = float(jnp.max(eigs) / np.min(eigs))  # Eigenvalue spread
  return R, eigs, cond


def theoretical_R_ar1(a: float, order: int, sigma_v2: float = 1.0):
  """
  Compute theoretical autocorrelation matrix for AR(1) process.
  Assume the input x(n) is WSS, this is an implmentation of eq 2.83
  From equation (2.83): R = σ_v²/(1-a²) * Toeplitz([1, a, a², ..., a^(order-1)])
  
  This matches the definition R = E[x(k)x^T(k)] where x(k) = [x(k), x(k-1), ..., x(k-order+1)]^T
  """
  # Create the first row: [1, a, a², ..., a^(order-1)]
  first_row = np.array([a**k for k in range(order)])

  # Scale by σ_v²/(1-a²)
  scale = sigma_v2 / (1 - a**2)
  R = scale * jax.scipy.linalg.toeplitz(first_row)

  return R


# Search for AR(1) pole 'a' producing approximate eigenvalue spread
def find_ar1_for_spread(target_spread: float,
                        order: int,
                        key: jax.random.PRNGKey,
                        n_samples: int = 20000,
                        tol: float = 0.2):
  """Return a, example x sequence and R giving spread close to target_spread.
  Uses coarse grid search over a in [0, 0.99] and returns best a."""
  best = None
  a_grid = np.linspace(0.0, 0.99, 101)
  best_diff = np.inf
  best_info = None
  for a in a_grid:
    # white gaussian noise
    u = jax.random.normal(key, (n_samples,), dtype=jnp.float64)
    # Generate AR(1) process: x[n] = a*x[n-1] + u[n]
    x = jnp.zeros_like(u)
    for n in range(1, len(u)):
      x[n] = a * x[n - 1] + u[n]
    R, eigs, cond = empirical_R_and_spread(x, order)
    diff = abs(cond - target_spread)
    if diff < best_diff:
      best_diff = diff
      best_info = (a, x, R, eigs, cond)
    if best_diff <= tol:
      break
  return best_info  # (a, x, R, eigs, cond)


def find_ar1_pole_for_spread_theoretical(target_spread: float,
                                         order: int,
                                         sigma_v2: float = 1.0,
                                         tol: float = 0.01) -> float:
  """Find AR(1) pole that produces target eigenvalue spread using theoretical formula"""
  a_grid = np.linspace(0.0, 0.999, 1000)
  best_diff = np.inf
  best_a = 0.0

  for a in a_grid:
    R = theoretical_R_ar1(a, order, sigma_v2)  # Pass sigma_v2 here
    eigs = jnp.linalg.eigvalsh(R)
    spread = float(jnp.max(eigs) / jnp.min(eigs))
    diff = abs(spread - target_spread)

    if diff < best_diff:
      best_diff = diff
      best_a = a

    if best_diff <= tol:
      break

  return best_a


def create_spread_controlled_generator(original_gen_fn: Callable,
                                       target_spread: float,
                                       filter_order: int,
                                       sigma_v2: float = 1.0) -> Callable:
  """Create a wrapped signal generator that controls eigenvalue spread"""

  def spread_controlled_gen(n_samples: int, key: jnp.ndarray, **kwargs):
    # First call original generator to get the true system
    h_true, _, _ = original_gen_fn(n_samples, key)

    # Generate new input signal with desired spread
    a = find_ar1_pole_for_spread_theoretical(target_spread, filter_order, sigma_v2)
    x_controlled = generate_ar1_signal(n_samples, a, sigma_v2=sigma_v2, key=key)

    # Re-generate desired signal using the controlled input
    d_controlled = jnp.convolve(x_controlled, h_true, mode='same')

    # Add measurement noise (preserve original noise characteristics)
    if 'sigma_n2' in kwargs:
      key, subkey = jax.random.split(key)
      noise = jax.random.normal(subkey, d_controlled.shape) * jnp.sqrt(kwargs['sigma_n2'])
      d_controlled = d_controlled + noise

    return h_true, x_controlled, d_controlled

  return spread_controlled_gen


def generate_ar1_signal(n_samples: int,
                        a: float,
                        sigma_v2: float = 1.0,
                        burn_in: int = 1000,
                        key: jax.random.PRNGKey = None):
  """
    Generate AR(1) signal: x[n] = a*x[n-1] + v[n], v[n] ~ N(0, σ_v²)
  """
  key, subkey = jax.random.split(key)
  v = jax.random.normal(subkey, (n_samples + burn_in,)) * jnp.sqrt(sigma_v2)
  x = jnp.zeros_like(v)
  for n in range(1, len(v)):
    x = x.at[n].set(a * x[n - 1] + v[n])

  return x[burn_in:]


def find_convergence_iteration(mse_curve: _Array, threshold_db: float) -> int:
  """Find iteration where MSE drops below threshold (in dB relative to initial)"""
  initial_mse = jnp.mean(mse_curve[:100])
  threshold = initial_mse * (10**(threshold_db / 10))

  for i in range(100, len(mse_curve)):
    if jnp.mean(mse_curve[i:i + 50]) < threshold:
      return i
  return len(mse_curve)


def verify_R_calculation(a: float,
                         order: int,
                         n_samples: int = 10000,
                         key: jax.random.PRNGKey = None):
  """
  Verify that empirical R matches theoretical R
  """

  print(f"\n--- Verification for AR(1) with a = {a:.3f}, order = {order} ---")

  # Theoretical R
  R_theoretical = theoretical_R_ar1(a, order)

  # Generate AR(1) signal
  x = generate_ar1_signal(n_samples, a, burn_in=1000, key=key)

  # Empirical R using corrected implementation
  R_empirical, eigs_empirical, cond_empirical = empirical_R_and_spread(x, order)

  # Compare
  print("Theoretical R (first 3x3 block):")
  print(R_theoretical[:3, :3])
  print("\nEmpirical R (first 3x3 block):")
  print(R_empirical[:3, :3])

  mse = np.mean((R_empirical - R_theoretical)**2)
  print(f"\nMSE between theoretical and empirical R: {mse:.6e}")

  # Verify specific elements
  print(f"R[0,0]: theoretical={R_theoretical[0,0]:.4f}, empirical={R_empirical[0,0]:.4f}")
  print(f"R[0,1]: theoretical={R_theoretical[0,1]:.4f}, empirical={R_empirical[0,1]:.4f}")
  print(f"R[1,2]: theoretical={R_theoretical[1,2]:.4f}, empirical={R_empirical[1,2]:.4f}")


class LMS(nn.Module):
  """Complex LMS adaptive filter (Algorithm 3.2, Diniz)."""
  mu: float
  filter_order: int
  init_coef: Optional[_Array] = None

  def setup(self):
    self.n_coef = self.filter_order + 1
    if self.init_coef is not None:
      init_w = jnp.array(self.init_coef, dtype=jnp.complex64)
    else:
      init_w = jnp.zeros((self.n_coef,), dtype=jnp.complex64)
    self.w = self.variable('state', 'w', lambda: init_w)

  @nn.compact
  def __call__(self, x: _Array, d: _Array, train: bool = True):
    """
      Run LMS adaptation over an entire sequence using lax.scan.
      Args:
          x: Input signal, shape (N,)
          d: Desired signal, shape (N,)
          train: If True, update coefficients; otherwise freeze them.
      Returns:
          y_hist: Outputs over time, shape (N,)
          e_hist: Errors over time, shape (N,)
          w_hist: Coefficient history, shape (N+1, n_coef)
    """
    #n_coef = self.filter_order + 1
    N = x.shape[0]

    # Pad input
    x_padded = jnp.concatenate([jnp.zeros(self.n_coef - 1, dtype=x.dtype), x])

    # Initialize weights
    if self.init_coef is not None:
      init_w = jnp.array(self.init_coef, dtype=jnp.complex64)
    else:
      init_w = jnp.zeros((self.n_coef,), dtype=jnp.complex64)

    w0 = self.param("w", nn.initializers.zeros, (self.n_coef,), x.dtype)
    w0 = init_w.astype(x.dtype)  # override with init if provided

    def step_fn(cur_coef, k):
      # Regressor slice
      regressor = jax.lax.dynamic_slice(x_padded, (k,), (self.n_coef,))[::-1]

      # Output
      y_k = jnp.vdot(cur_coef, regressor)
      e_k = d[k] - y_k

      # Update coefficients if train
      w_new = jax.lax.cond(
          train,
          lambda _: cur_coef + self.mu * jnp.conj(e_k) * regressor,
          lambda _: cur_coef,
          operand=None,
      )

      return w_new, (y_k, e_k, w_new)

    # Run scan
    w_final, (y_hist, e_hist, w_hist) = jax.lax.scan(step_fn, w0, jnp.arange(N))

    # Prepend initial weights to history
    w_hist = jnp.vstack([ w0, w_hist ])

    return y_hist, e_hist, w_hist


# --------------------------------------------------
# Base Adaptive Problem Class
# --------------------------------------------------
class AdaptiveProblem(ABC):

  def __init__(self,
               lms: Any,
               gen_signals_fn: Callable,
               wiener_sol_fn: Optional[Callable] = None,
               seed: int = 42):
    """
      Base class for adaptive filtering problems.
      
      Args:
          lms: LMS filter instance
          gen_signals_fn: Function that generates signals for the specific problem
          wiener_sol_fn: Function to compute Wiener solution (optional)
          seed: Random seed
    """
    self.lms = lms
    self.gen_signals_fn = gen_signals_fn
    self.key = jax.random.PRNGKey(seed)
    self.wiener_sol_fn = wiener_sol_fn

    # Initialize with dummy signals
    self.vars = self.initialize_filter()

  def initialize_filter(self) -> _Array:
    """Initialize filter variables with dummy signals"""
    dummy_x = jnp.zeros(10, dtype=jnp.float64)
    dummy_d = jnp.zeros(10, dtype=jnp.float64)
    return self.lms.init(self.key, dummy_x, dummy_d)

  def generate_signals(self, n_samples: int, *args, **kwargs) -> Tuple:
    """Generate signals for the specific problem"""
    self.key, subkey = jax.random.split(self.key)
    return self.gen_signals_fn(n_samples, key=subkey)

  def run_adaptation(self,
                     x: _Array,
                     d: _Array,
                     training: bool = True) -> Tuple[_Array, _Array, _Array]:
    """Run LMS adaptation on given signals"""
    return self.lms.apply(self.vars, x, d, train=training)

  @abstractmethod
  def run(self, n_samples: int, training: bool = True):
    """Run the complete adaptive filtering experiment"""
    pass

  def default_wiener_solution(self, x: _Array, d: _Array) -> Tuple[_Array, _Array, _Array]:
    """Default Wiener solution implementation using time averages"""
    N_coef = self.lms.filter_order + 1  # Number of coefficients
    N = len(x)

    if N < N_coef:
      raise ValueError(f"Need at least {N_coef} samples for Wiener solution")

    # Estimate autocorrelation matrix R
    R = jnp.zeros((N_coef, N_coef), dtype=x.dtype)
    for i in range(N_coef):
      for j in range(i, N_coef):  # Only compute upper triangle
        lag = abs(i - j)
        corr = jnp.mean(jnp.conj(x[lag:]) * x[:N - lag])
        R = R.at[i, j].set(corr)
        if i != j:
          R = R.at[j, i].set(jnp.conj(corr))  # Make Hermitian

    # Estimate cross-correlation vector p
    p = jnp.zeros(N_coef, dtype=x.dtype)
    for i in range(N_coef):
      corr = jnp.mean(jnp.conj(d[i:]) * x[:N - i])
      p = p.at[i].set(corr)

    # Add regularization and solve
    R_reg = R + 1e-8 * jnp.eye(N_coef, dtype=x.dtype)
    w_opt = jnp.linalg.solve(R_reg, p)

    return w_opt, R, p

  def wiener_solution(self, x: _Array, d: _Array) -> Tuple[_Array, _Array, _Array]:
    """Compute Wiener solution using provided function or default implementation"""
    if self.wiener_sol_fn is not None:
      return self.wiener_sol_fn(x, d, self.lms.filter_order)
    else:
      return self.default_wiener_solution(x, d)

  def compute_basic_metrics(self, w_hist: _Array, e_hist: _Array, x: _Array,
                            d: _Array) -> Dict[str, Any]:
    """Compute basic performance metrics common to all adaptive problems"""
    w_opt, R, p = self.wiener_solution(x, d)

    mse_curve = jnp.abs(e_hist)**2
    coeff_error = jnp.linalg.norm(w_hist - w_opt, axis=1)

    return {
        "wiener_solution": w_opt,
        "R": R,
        "p": p,
        "final_coeff": w_hist[-1],
        "final_mse": jnp.mean(mse_curve[-100:]),
        "final_coeff_error": jnp.mean(coeff_error[-100:]),
        "mse_curve": mse_curve,
        "coeff_error_curve": coeff_error,
    }

  @abstractmethod
  def analyze_performance(self, w_hist: _Array, e_hist: _Array, x: _Array,
                          d: _Array) -> Dict[str, Any]:
    """Analyze problem-specific performance metrics"""
    pass

  def get_problem_type(self) -> str:
    """Return the type of adaptive problem"""
    return self.__class__.__name__

  def step_size(self):
    return self.lms.mu


# --------------------------------------------------
# Channel Equalization Problem
# --------------------------------------------------
class ChannelEqualization(AdaptiveProblem):

  def __init__(self, lms: LMS, gen_signals_fn, wiener_sol_fn=None, seed: int = 42):
    super().__init__(lms, gen_signals_fn, wiener_sol_fn, seed)

  def run(self, n_samples: int, training: bool = True):
    """Run channel equalization experiment"""
    x, d = self.generate_signals(n_samples)

    # Run LMS adaptation
    y_hist, e_hist, w_hist = self.run_adaptation(x, d, training)

    # Analyze performance
    metrics = self.analyze_performance(w_hist, e_hist, x, d)
    return w_hist, e_hist, metrics

  def analyze_performance(self, w_hist, e_hist, x, d):
    return self.compute_basic_metrics(w_hist, e_hist, x, d)


# ---------------------------------------------------------
# Enhanced System Identification Problem with Spread Control
# ----------------------------------------------------------
class SystemIdentification(AdaptiveProblem):

  def __init__(self,
               lms: Any,
               gen_signals_fn: Callable,
               wiener_sol_fn: Optional[Callable] = None,
               seed: int = 42):
    """
      System Identification Problem
      
      Args:
          lms: LMS filter instance
          gen_signals_fn: Function that returns (unknown_system_taps, x, d)
                        This function should handle any spread control internally
          wiener_sol_fn: Function to compute Wiener solution (optional)
          seed: Random seed
      """
    super().__init__(lms, gen_signals_fn, wiener_sol_fn, seed)

  def run(self, n_samples: int, training: bool = True, **kwargs):
    """Run system identification experiment"""
    # Generate unknown system and signals
    # Pass any additional kwargs to the signal generator
    unknown_system, x, d, measured_noise = self.generate_signals(n_samples, **kwargs)

    # Run LMS adaptation
    y_hist, e_hist, w_hist = self.run_adaptation(x, d, training)

    # Analyze performance
    metrics = self.analyze_performance(w_hist, e_hist, x, d, unknown_system, measured_noise)

    return w_hist, e_hist, metrics, unknown_system

  def analyze_performance(self, w_hist: _Array, e_hist: _Array, x: _Array, d: _Array,
                          h_true: _Array, measured_noise: _Array) -> Dict[str, Any]:
    """Fixed version with reliable min MSE calculation"""

    # Get basic metrics
    metrics = self.compute_basic_metrics(w_hist, e_hist, x, d)

    # Use KNOWN noise variance instead of measured (more reliable)
    min_mse_reliable = jnp.mean(jnp.abs(measured_noise)**2)

    # System identification specific metrics
    coeff_error_true = jnp.linalg.norm(w_hist - h_true, axis=1)

    # Compute theoretical parameters
    R = metrics["R"]
    eigenvals = jnp.linalg.eigvals(R)
    max_eigenval = jnp.max(jnp.real(eigenvals))
    min_eigenval = jnp.min(jnp.real(eigenvals))
    trace_R = jnp.trace(R)

    # MSE calculations
    mse_curve = jnp.abs(e_hist)**2

    # Use only steady-state for final calculations
    steady_state_start = len(mse_curve) // 2
    if 'convergence_20dB_iter' in metrics:
      steady_state_start = metrics['convergence_20dB_iter']

    mse_steady = mse_curve[steady_state_start:]
    final_mse = jnp.mean(mse_steady[-100:])
    mean_mse_steady = jnp.mean(mse_steady)

    # Misadjustment calculations
    misadjustment_curve = (mse_curve - min_mse_reliable) / min_mse_reliable
    final_misadjustment = (final_mse - min_mse_reliable) / min_mse_reliable
    misadjustment_avg = (mean_mse_steady - min_mse_reliable) / min_mse_reliable

    # Theoretical
    mu = self.step_size()
    theoretical_misadjustment = (mu * trace_R) / (1 - mu * trace_R)

    # Convergence analysis
    convergence_metrics = self.analyze_convergence(e_hist, w_hist, h_true,
                                                   metrics["wiener_solution"])

    # Print diagnostic information
    print(f"\n=== PERFORMANCE DIAGNOSTICS ===")
    print(f"Min MSE (σₙ²): {min_mse_reliable:.6f}")
    print(f"Final MSE: {final_mse:.6f}")
    print(f"Mean MSE (steady): {mean_mse_steady:.6f}")
    print(f"Final Misadjustment: {final_misadjustment:.6f}")
    print(f"Theoretical Misadjustment: {theoretical_misadjustment:.6f}")
    print(f"Steady-state start: {steady_state_start}")
    print("=" * 40)

    metrics.update({
        "unknown_system": h_true,
        "mean_mse": mean_mse_steady,
        "min_mse": min_mse_reliable,
        "misadjustment_curve": misadjustment_curve,
        "misadjustment": misadjustment_avg,  # Use steady-state average
        "misadjustment_theoretical": theoretical_misadjustment,
        "final_misadjustment": final_misadjustment,
        "final_coeff_error_true": jnp.mean(coeff_error_true[-100:]),
        "coeff_error_true_curve": coeff_error_true,
        "eigenvalues": eigenvals,
        "max_eigenvalue": max_eigenval,
        "min_eigenvalue": min_eigenval,
        "trace_R": trace_R,
        "problem_type": "system_identification",
        **convergence_metrics
    })

    return metrics
  
  def analyze_convergence(self, e_hist: _Array, w_hist: _Array, h_true: _Array,
                          w_opt: _Array) -> Dict[str, Any]:
    """Analyze convergence behavior for system identification"""
    mse_curve = jnp.abs(e_hist)**2

    # Algorithm performance: how close to Wiener solution
    coeff_error_wiener = jnp.linalg.norm(w_hist - w_opt, axis=1)

    # Identification accuracy: how close to true system
    coeff_error_true = jnp.linalg.norm(w_hist - h_true, axis=1)

    # Find convergence iterations
    conv_iter_10dB = find_convergence_iteration(mse_curve, -10)
    conv_iter_20dB = find_convergence_iteration(mse_curve, -20)
    conv_iter_30dB = find_convergence_iteration(mse_curve, -30)

    return {
        "convergence_10dB_iter": conv_iter_10dB,
        "convergence_20dB_iter": conv_iter_20dB,
        "convergence_30dB_iter": conv_iter_30dB,
        "final_coeff_error_wiener": jnp.mean(coeff_error_wiener[-100:]),
        "final_coeff_error_true": jnp.mean(coeff_error_true[-100:]),
        "coeff_error_wiener_curve": coeff_error_wiener,
        "coeff_error_true_curve": coeff_error_true,
    }

class TimeVaryingSystemIdentification(SystemIdentification):
  """Extended for time-varying system analysis"""

  def __init__(
      self,
      lms: Any,
      gen_signals_fn: Callable,
      wiener_sol_fn: Optional[Callable] = None,
      sigma_w2: float = 0.0015,  # Default from problem 3.6.2(e)
      sigma_n2: float = 0.01,  # Default from problem 3.6.2(e)  
      seed: int = 42):
    """
        Time-Varying System Identification Problem
        
        Args:
            lms: LMS filter instance
            gen_signals_fn: Function that returns (unknown_system_taps, x, d, measured_noise)
            wiener_sol_fn: Function to compute Wiener solution (optional)
            sigma_w2: Variance of coefficient innovations
            sigma_n2: Measurement noise variance
            seed: Random seed
        """
    super().__init__(lms, gen_signals_fn, wiener_sol_fn, seed)
    self.sigma_w2 = sigma_w2
    self.sigma_n2 = sigma_n2
  
  def analyze_performance(self, w_hist: _Array, e_hist: _Array, x: _Array, d: _Array,
                          h_true_time_varying: _Array, measured_noise: _Array) -> Dict[str, Any]:
    """Analyze performance for time-varying system with complete metrics"""

    # Get basic metrics from parent class
    metrics = super().analyze_performance(w_hist, e_hist, x, d, h_true_time_varying, measured_noise)

    # Time-varying specific metrics
    n_iters = len(e_hist)
    n_coeffs = w_hist.shape[1]

    # Calculate lag error vector: l_w(k) = w(k) - w_o(k)
    l_w = w_hist - h_true_time_varying[:n_iters + 1]

    # Get R matrix (computed in parent class)
    R = metrics["R"]
    trace_R = metrics["trace_R"]

    # Calculate lag-induced excess MSE
    lag_excess_mse = self.calculate_lag_excess_mse(l_w, R)

    # Theoretical calculations
    mu = self.step_size()

    # 1. Gradient noise component (stationary case misadjustment)
    misadjustment_gradient = (mu * trace_R) / (1 - mu * trace_R)
    excess_mse_gradient = misadjustment_gradient * self.sigma_n2

    # 2. Tracking lag components
    # Simplified theoretical approximation
    theoretical_lag_excess_mse_simple = (self.sigma_w2 / (4 * mu)) * n_coeffs

    # More accurate theoretical calculation from equation (3.69)
    eigenvals = jnp.linalg.eigvals(R)
    theoretical_lag_excess_mse_accurate = 0.0
    for i in range(n_coeffs):
      lambda_i = jnp.real(eigenvals[i])
      theoretical_lag_excess_mse_accurate += self.sigma_w2 / (4 * mu * (1 - mu * lambda_i))

    # 3. Total theoretical excess MSE
    total_theoretical_excess_mse = excess_mse_gradient + theoretical_lag_excess_mse_accurate
    total_theoretical_mse = self.sigma_n2 + total_theoretical_excess_mse

    # 4. Experimental excess MSE calculations
    mse_curve = jnp.abs(e_hist)**2
    final_mse = metrics["final_mse"]
    experimental_excess_mse = final_mse - self.sigma_n2

    # 5. Coefficient tracking performance
    coeff_tracking_error = jnp.linalg.norm(l_w, axis=1)
    final_coeff_tracking_error = jnp.mean(coeff_tracking_error[-100:])

    # 6. Calculate optimal μ for comparison
    mu_opt = self.calculate_optimal_mu(n_coeffs, trace_R)

    # Update metrics with time-varying specific results
    metrics.update({
        "h_true_time_varying": h_true_time_varying,
        "l_w": l_w,  # Lag error vector history

        # Lag excess MSE calculations
        "lag_excess_mse_experimental": lag_excess_mse,
        "lag_excess_mse_theoretical_simple": theoretical_lag_excess_mse_simple,
        "lag_excess_mse_theoretical_accurate": theoretical_lag_excess_mse_accurate,

        # Gradient noise components
        "misadjustment_gradient_theoretical": misadjustment_gradient,
        "excess_mse_gradient_theoretical": excess_mse_gradient,

        # Total excess MSE
        "excess_mse_experimental": experimental_excess_mse,
        "excess_mse_theoretical": total_theoretical_excess_mse,
        "total_mse_theoretical": total_theoretical_mse,

        # Tracking performance
        "coeff_tracking_error_curve": coeff_tracking_error,
        "final_coeff_tracking_error": final_coeff_tracking_error,

        # Optimal μ analysis
        "mu_opt": mu_opt,
        "mu_used": mu,
        "is_optimal_mu": jnp.abs(mu - mu_opt) < 0.001,

        # Problem parameters
        "sigma_w2": self.sigma_w2,
        "sigma_n2": self.sigma_n2,
        "problem_type": "time_varying_system_identification",
    })

    # Print comprehensive results
    self._print_time_varying_analysis(metrics)

    return metrics

  def calculate_lag_excess_mse(self, l_w: _Array, R: _Array) -> float:
    """
        Calculate excess MSE due to tracking lag in nonstationary environments.
        
        From equation (3.69): ξ_lag = E[l_w^T(k) R l_w(k)]
        """
    # Calculate covariance of lag error
    l_w_cov = jnp.cov(l_w.T)  # Shape (N, N)

    # ξ_lag = trace(R * E[l_w l_w^T])
    lag_excess_mse = jnp.trace(R @ l_w_cov)

    return lag_excess_mse

  def calculate_optimal_mu(self, n_coeffs: int, tr_R: float) -> float:
    """
        Calculate optimal step size μ for nonstationary environments.
        
        μ_opt = √[ (N) * σ_w² / (4 * σ_n² * tr[R]) ]
        """
    numerator = n_coeffs * self.sigma_w2
    denominator = 4 * self.sigma_n2 * tr_R
    return jnp.sqrt(numerator / denominator)

  def print_time_varying_analysis(self, metrics: Dict[str, Any]):
    """Print comprehensive time-varying analysis results"""

    print(f"\n{'='*80}")
    print("TIME-VARYING SYSTEM IDENTIFICATION ANALYSIS")
    print(f"{'='*80}")

    print(f"\nParameters:")
    print(f"  Step size μ: {metrics['mu_used']:.6f}")
    print(f"  Optimal μ:   {metrics['mu_opt']:.6f}")
    print(f"  Using optimal μ: {metrics['is_optimal_mu']}")
    print(f"  σ_w²: {self.sigma_w2:.6f} (coefficient innovation)")
    print(f"  σ_n²: {self.sigma_n2:.6f} (measurement noise)")
    print(f"  Filter coefficients: {metrics['w_hist'].shape[1]}")

    print(f"\nPerformance Results:")
    print(f"  Final MSE: {metrics['final_mse']:.6f}")
    print(f"  Theoretical MSE: {metrics['total_mse_theoretical']:.6f}")

    print(f"\nExcess MSE Breakdown:")
    print(f"  Total Experimental: {metrics['excess_mse_experimental']:.6f}")
    print(f"  Total Theoretical:  {metrics['excess_mse_theoretical']:.6f}")

    print(f"\n  Experimental Components:")
    print(
        f"    - Gradient Noise: {metrics['excess_mse_experimental'] - metrics['lag_excess_mse_experimental']:.6f}"
    )
    print(f"    - Tracking Lag:   {metrics['lag_excess_mse_experimental']:.6f}")

    print(f"\n  Theoretical Components:")
    print(f"    - Gradient Noise: {metrics['excess_mse_gradient_theoretical']:.6f}")
    print(f"    - Tracking Lag:   {metrics['lag_excess_mse_theoretical_accurate']:.6f}")

    print(f"\nTracking Performance:")
    print(f"  Final Coefficient Error: {metrics['final_coeff_tracking_error']:.6f}")

    # Calculate agreement percentages
    total_agreement = 1 - abs(metrics['excess_mse_experimental'] -
                              metrics['excess_mse_theoretical']) / metrics['excess_mse_theoretical']
    lag_agreement = 1 - abs(metrics['lag_excess_mse_experimental'] -
                            metrics['lag_excess_mse_theoretical_accurate']
                           ) / metrics['lag_excess_mse_theoretical_accurate']

    print(f"\nAgreement with Theory:")
    print(f"  Total Excess MSE: {total_agreement:.2%}")
    print(f"  Lag Component:    {lag_agreement:.2%}")

    if total_agreement > 0.8:
      print("  ✓ Good agreement with theoretical predictions")
    elif total_agreement > 0.6:
      print("  ~ Reasonable agreement with theoretical predictions")
    else:
      print("  ✗ Poor agreement with theoretical predictions")


def plot_results(w_hist: _Array, e_hist: _Array, metrics: Dict[str, Any]):
  """
    Plot comprehensive system identification results with new metrics and 
    return the matplotlib Figure object.
    """

  n_iters = len(e_hist)
  n_coeffs = w_hist.shape[1]

  # Create figure with more subplots
  fig = plt.figure(figsize=(18, 20))  # Increased height for new plot
  gs = gridspec.GridSpec(5, 3, figure=fig)  # Changed to 5 rows

  # Plot 1: Learning curve (MSE)
  ax1 = fig.add_subplot(gs[0, 0])
  # Use .get with a default for robust access
  mse_curve = metrics.get('mse_curve', jnp.abs(e_hist)**2)
  # Ensure mse_curve is a numpy array for plotting if it was a jnp array
  mse_curve_np = np.asarray(mse_curve)

  ax1.semilogy(mse_curve_np, 'b-', alpha=0.7, linewidth=1)

  # Mark convergence points if available
  if 'convergence_10dB_iter' in metrics:
    ax1.axvline(x=metrics['convergence_10dB_iter'],
                color='r',
                linestyle='--',
                alpha=0.7,
                label='-10dB')
  if 'convergence_20dB_iter' in metrics:
    ax1.axvline(x=metrics['convergence_20dB_iter'],
                color='g',
                linestyle='--',
                alpha=0.7,
                label='-20dB')
  if 'convergence_30dB_iter' in metrics:
    ax1.axvline(x=metrics['convergence_30dB_iter'],
                color='m',
                linestyle='--',
                alpha=0.7,
                label='-30dB')

  ax1.set_ylabel('MSE')
  ax1.set_xlabel('Iteration')
  ax1.set_title('(a) Learning Curve with Convergence Points')
  ax1.grid(True, alpha=0.3)
  if any(key in metrics
         for key in [ 'convergence_10dB_iter', 'convergence_20dB_iter', 'convergence_30dB_iter']):
    ax1.legend()

  # Plot 2: Misadjustment Learning Curve (NEW PLOT)
  ax2 = fig.add_subplot(gs[0, 1])
  if 'misadjustment_curve' in metrics:
    misadjustment_curve = np.asarray(metrics['misadjustment_curve'])
    ax2.plot(misadjustment_curve, 'g-', alpha=0.7, linewidth=1, label='Empirical Misadjustment')

    # Plot theoretical misadjustment line if available
    if 'misadjustment_theoretical' in metrics:
      theoretical_m = metrics['misadjustment_theoretical']
      ax2.axhline(y=theoretical_m,
                  color='r',
                  linestyle='--',
                  linewidth=2,
                  label=f'Theoretical M = {theoretical_m:.3f}')

    # Plot final steady-state misadjustment if available
    if 'misadjustment' in metrics:
      final_m = metrics['misadjustment']
      ax2.axhline(y=final_m,
                  color='b',
                  linestyle=':',
                  linewidth=2,
                  label=f'Final M = {final_m:.3f}')

    ax2.set_ylabel('Misadjustment M(k)')
    ax2.set_xlabel('Iteration')
    ax2.set_title('(b) Misadjustment Learning Curve')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
  else:
    ax2.text(0.5,
             0.5,
             'No misadjustment curve data',
             ha='center',
             va='center',
             transform=ax2.transAxes)
    ax2.set_title('(b) Misadjustment Learning Curve')
    ax2.grid(True, alpha=0.3)

  # Plot 3: Coefficient convergence vs TRUE system
  ax3 = fig.add_subplot(gs[0, 2])
  if 'unknown_system' in metrics:
    h_true = metrics['unknown_system']
    for i in range(min(4, n_coeffs)):
      # Plot real part of coefficients
      ax3.plot(jnp.real(w_hist[:, i]), label=f'Re(w[{i}])', alpha=0.7)
      # Plot real part of true system
      ax3.axhline(y=jnp.real(h_true[i]),
                  color=f'C{i}',
                  linestyle='--',
                  linewidth=2,
                  label=f'Re(h_true[{i}])',
                  alpha=0.8)
  else:
    for i in range(min(4, n_coeffs)):
      ax3.plot(jnp.real(w_hist[:, i]), label=f'Re(w[{i}])', alpha=0.7)

  ax3.set_ylabel('Coefficient Value')
  ax3.set_xlabel('Iteration')
  ax3.set_title('(c) Coefficient Convergence vs True System (Real)')
  ax3.legend(ncol=2, fontsize=8)
  ax3.grid(True, alpha=0.3)

  # Plot 4: Dual coefficient errors (Wiener vs True)
  ax4 = fig.add_subplot(gs[1, 0])
  if 'coeff_error_curve' in metrics and 'coeff_error_true_curve' in metrics:
    ax4.semilogy(metrics['coeff_error_curve'], 'r-', label='Error vs Wiener', alpha=0.7)
    ax4.semilogy(metrics['coeff_error_true_curve'], 'b-', label='Error vs True', alpha=0.7)
    ax4.set_ylabel('Coefficient Error Norm')
    ax4.legend()
  elif 'coeff_error_curve' in metrics:
    ax4.semilogy(metrics['coeff_error_curve'], 'r-', label='Error vs Wiener')
    ax4.set_ylabel('Coefficient Error Norm')
  ax4.set_xlabel('Iteration')
  ax4.set_title('(d) Coefficient Error Comparison')
  ax4.grid(True, alpha=0.3)

  # Plot 5: Eigenvalue analysis
  ax5 = fig.add_subplot(gs[1, 1])
  if 'eigenvalues' in metrics:
    # Use real part for plotting magnitude in stem plot context
    eigenvals = np.abs(metrics['eigenvalues'])
    ax5.stem(np.real(eigenvals), basefmt=" ")
    ax5.set_ylabel('Eigenvalue Magnitude')
    ax5.set_xlabel('Eigenvalue Index')

    # Add eigenvalue spread info
    max_e = np.max(eigenvals)
    min_e = np.min(eigenvals)
    # Handle the case of min_e being zero or near zero to avoid division by zero
    spread = max_e / min_e if min_e > 1e-10 else np.inf
    spread_str = f"Spread: {spread:.2f}" if spread != np.inf else "Spread: Inf"

    ax5.set_title(f'(e) Eigenvalue Distribution\n{spread_str}')
  else:
    ax5.text(0.5, 0.5, 'No eigenvalue data', ha='center', va='center', transform=ax5.transAxes)
    ax5.set_title('(e) Eigenvalue Distribution')
  ax5.grid(True, alpha=0.3)

  # Plot 6: Misadjustment and performance metrics (Bar chart)
  ax6 = fig.add_subplot(gs[1, 2])
  performance_data = []
  performance_labels = []

  if 'misadjustment' in metrics:
    performance_data.append(metrics['misadjustment'])
    performance_labels.append('Final Misadjustment')

  if 'misadjustment_theoretical' in metrics:
    performance_data.append(metrics['misadjustment_theoretical'])
    performance_labels.append('Theoretical Misadjustment')

  if 'final_mse' in metrics:
    performance_data.append(metrics['final_mse'])
    performance_labels.append('Final MSE')

  if 'min_mse' in metrics:
    performance_data.append(metrics['min_mse'])
    performance_labels.append('Min MSE')

  if performance_data:
    bars = ax6.bar(range(len(performance_data)),
                   performance_data,
                   color=[ 'skyblue', 'lightcoral', 'lightgreen', 'lightyellow'],
                   width=0.4,
                   edgecolor='black',
                   linewidth=0.3)
    ax6.set_xticks(range(len(performance_data)))
    ax6.set_xticklabels(performance_labels, rotation=45, ha='right')

    # Add value labels on bars
    for bar, value in zip(bars, performance_data):
      ax6.text(bar.get_x() + bar.get_width() / 2,
               bar.get_height(),
               f'{value:.2e}',
               ha='center',
               va='bottom',
               fontsize=9)

    ax6.ticklabel_format(axis='y', style='sci',
                         scilimits=(-2, 3))  # Use scientific notation for Y-axis

  ax6.set_ylabel('Value')
  ax6.set_title('(f) Performance Metrics')
  ax6.grid(True, alpha=0.3, axis='y')

  # Plot 7: Spread control information
  ax7 = fig.add_subplot(gs[2, 0])
  spread_info = []
  if 'target_eigenvalue_spread' in metrics:
    spread_info.append(f"Target Spread: {metrics['target_eigenvalue_spread']:.1f}")
  if 'actual_eigenvalue_spread' in metrics:
    spread_info.append(f"Actual Spread: {metrics['actual_eigenvalue_spread']:.1f}")
  if 'ar1_pole_used' in metrics:
    spread_info.append(f"AR(1) Pole: {metrics['ar1_pole_used']:.3f}")

  if spread_info:
    ax7.text(0.5,
             0.7,
             '\n'.join(spread_info),
             ha='center',
             va='center',
             transform=ax7.transAxes,
             fontsize=12,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
  else:
    ax7.text(0.5, 0.5, 'No spread control', ha='center', va='center', transform=ax7.transAxes)

  ax7.set_title('(g) Eigenvalue Spread Control')
  ax7.axis('off')

  # Plot 8: Frequency response comparison
  ax8 = fig.add_subplot(gs[2, 1])
  w_final = w_hist[-1, :]
  n_fft = 1024

  if 'unknown_system' in metrics:
    h_true = metrics['unknown_system']
    # True system frequency response
    w_freq, h_true_response = signal.freqz(np.array(jnp.real(h_true)), worN=n_fft)
    # Estimated system frequency response
    _, h_est_response = signal.freqz(np.array(jnp.real(w_final)), worN=n_fft)

    ax8.plot(w_freq / np.pi,
             20 * np.log10(np.abs(h_true_response + 1e-10)),
             'b-',
             label='True System',
             linewidth=2)
    ax8.plot(w_freq / np.pi,
             20 * np.log10(np.abs(h_est_response + 1e-10)),
             'r--',
             label='Estimated',
             linewidth=2)
    ax8.set_ylabel('Magnitude (dB)')
    ax8.set_xlabel('Normalized Frequency (×π rad/sample)')
    ax8.set_title('(h) Frequency Response Comparison')
    ax8.legend()
  else:
    w_freq, h_response = signal.freqz(np.array(jnp.real(w_final)), worN=n_fft)
    ax8.plot(w_freq / np.pi, 20 * np.log10(np.abs(h_response + 1e-10)))
    ax8.set_ylabel('Magnitude (dB)')
    ax8.set_xlabel('Normalized Frequency (×π rad/sample)')
    ax8.set_title('(h) Final Filter Frequency Response')
  ax8.grid(True, alpha=0.3)

  # Plot 9: Error distribution analysis
  ax9 = fig.add_subplot(gs[2, 2])
  # Take the last 1000 samples for better statistics, if available
  n_samples_hist = min(1000, len(e_hist))
  last_errors = e_hist[-n_samples_hist:]

  # Ensure data is numpy array for hist
  real_errors = np.asarray(jnp.real(last_errors))
  imag_errors = np.asarray(jnp.imag(last_errors))

  ax9.hist(real_errors, bins=20, alpha=0.7, label='Real', density=True)

  is_complex = not np.allclose(imag_errors, 0)
  if is_complex:
    ax9.hist(imag_errors, bins=20, alpha=0.7, label='Imag', density=True)

  # Add Gaussian fit to the real part
  mu, std = scipy.stats.norm.fit(real_errors)
  x = np.linspace(ax9.get_xlim()[0], ax9.get_xlim()[1], 100)
  ax9.plot(x,
           scipy.stats.norm.pdf(x, mu, std),
           'k-',
           linewidth=2,
           label=f'Normal fit\nμ={mu:.3f}, σ={std:.3f}')

  ax9.set_ylabel('Probability Density')
  ax9.set_xlabel('Error Value')
  ax9.set_title(f'(i) Error Distribution (last {n_samples_hist} samples)')
  ax9.legend()
  ax9.grid(True, alpha=0.3)

  # Plot 10: Convergence speed analysis
  ax10 = fig.add_subplot(gs[3, 0])
  # Moving average of MSE with different window sizes
  for window in [ 10, 50, 100 ]:
    if len(mse_curve_np) > window:
      mse_smooth = np.convolve(mse_curve_np, np.ones(window) / window, mode='valid')
      # Plot against the central point of the window for better alignment
      x_axis = np.arange(len(mse_smooth)) + window // 2
      ax10.semilogy(x_axis, mse_smooth, label=f'Window={window}', alpha=0.8)

  ax10.set_ylabel('Smoothed MSE')
  ax10.set_xlabel('Iteration')
  ax10.set_title('(j) Multi-Scale Learning Curves')
  ax10.legend()
  ax10.grid(True, alpha=0.3)

  # Plot 11: Misadjustment vs Theoretical (Scatter/Line comparison)
  ax11 = fig.add_subplot(gs[3, 1])
  if 'misadjustment_curve' in metrics and 'misadjustment_theoretical' in metrics:
    misadjustment_curve = np.asarray(metrics['misadjustment_curve'])
    theoretical_m = metrics['misadjustment_theoretical']

    # Plot the ratio of empirical to theoretical
    ratio_curve = misadjustment_curve / theoretical_m

    ax11.plot(ratio_curve, 'purple', alpha=0.7, linewidth=1)
    ax11.axhline(y=1.0, color='r', linestyle='--', linewidth=2, label='Theoretical Reference (1.0)')

    # Mark convergence region (last 20% of iterations)
    convergence_start = int(0.8 * len(ratio_curve))
    avg_convergence_ratio = np.mean(ratio_curve[convergence_start:])
    ax11.axhline(y=avg_convergence_ratio,
                 color='g',
                 linestyle=':',
                 linewidth=2,
                 label=f'Avg Convergence: {avg_convergence_ratio:.3f}')

    ax11.set_ylabel('Empirical / Theoretical')
    ax11.set_xlabel('Iteration')
    ax11.set_title('(k) Misadjustment Ratio vs Theory')
    ax11.grid(True, alpha=0.3)
    ax11.legend()
  else:
    ax11.text(0.5,
              0.5,
              'No misadjustment comparison data',
              ha='center',
              va='center',
              transform=ax11.transAxes)
    ax11.set_title('(k) Misadjustment Ratio vs Theory')
    ax11.grid(True, alpha=0.3)

  # Plot 12: Summary statistics
  ax12 = fig.add_subplot(gs[3, 2])
  summary_text = []

  # Basic metrics
  summary_text.append(f"Algorithm: {metrics.get('algorithm_name', 'LMS/RLS')}")
  summary_text.append(f"Iterations: {n_iters}")
  summary_text.append("-" * 30)
  if 'mean_mse' in metrics:
    summary_text.append(f"Mean MSE: {metrics['mean_mse']:.2e}")
  if 'final_mse' in metrics:
    summary_text.append(f"Final MSE: {metrics['final_mse']:.2e}")
  if 'min_mse' in metrics:
    summary_text.append(f"Minimum MSE: {metrics['min_mse']:.2e}")
  if 'misadjustment' in metrics:
    summary_text.append(f"Final Misadjustment: {metrics['misadjustment']:.4f}")
  if 'misadjustment_theoretical' in metrics:
    summary_text.append(f"Theoretical M: {metrics['misadjustment_theoretical']:.4f}")

  summary_text.append("-" * 30)

  # Convergence metrics
  if 'final_coeff_error_true' in metrics:
    summary_text.append(f"Final Coeff Error (True): {metrics['final_coeff_error_true']:.2e}")
  if 'final_coeff_error_wiener' in metrics:
    summary_text.append(f"Final Coeff Error (Wiener): {metrics['final_coeff_error_wiener']:.2e}")

  summary_text.append("-" * 30)

  # Eigenvalue metrics
  if 'max_eigenvalue' in metrics:
    summary_text.append(f"Max Eigenvalue: {metrics['max_eigenvalue']:.4f}")
  if 'min_eigenvalue' in metrics:
    summary_text.append(f"Min Eigenvalue: {metrics['min_eigenvalue']:.4f}")
  if 'trace_R' in metrics:
    summary_text.append(f"Trace(R): {metrics['trace_R']:.4f}")

  ax12.text(0.02,
            0.5,
            '\n'.join(summary_text),
            va='center',
            ha='left',
            fontsize=9,
            family='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"))
  ax12.set_title('(l) Summary Statistics')
  ax12.axis('off')

  # Plot 13: Additional performance analysis
  ax13 = fig.add_subplot(gs[4, :])

  analysis_text = []
  analysis_text.append("PERFORMANCE ANALYSIS")
  analysis_text.append("=" * 40)

  # Misadjustment analysis
  if 'misadjustment' in metrics and 'misadjustment_theoretical' in metrics:
    empirical_m = metrics['misadjustment']
    theoretical_m = metrics['misadjustment_theoretical']
    error_percent = abs(empirical_m - theoretical_m) / theoretical_m * 100
    analysis_text.append(f"Misadjustment Error: {error_percent:.1f}%")

    if error_percent < 10:
      analysis_text.append("✓ Good agreement with theory")
    elif error_percent < 25:
      analysis_text.append("~ Moderate agreement with theory")
    else:
      analysis_text.append("✗ Poor agreement with theory")

  # Convergence analysis
  if 'convergence_20dB_iter' in metrics:
    conv_iter = metrics['convergence_20dB_iter']
    analysis_text.append(f"Convergence (-20dB): {conv_iter} iterations")
    if conv_iter < n_iters * 0.3:
      analysis_text.append("✓ Fast convergence")
    elif conv_iter < n_iters * 0.7:
      analysis_text.append("~ Moderate convergence speed")
    else:
      analysis_text.append("✗ Slow convergence")

  # Coefficient accuracy
  if 'final_coeff_error_true' in metrics:
    coeff_error = metrics['final_coeff_error_true']
    analysis_text.append(f"Coefficient Error: {coeff_error:.2e}")
    if coeff_error < 0.1:
      analysis_text.append("✓ Good system identification")
    elif coeff_error < 0.3:
      analysis_text.append("~ Moderate identification")
    else:
      analysis_text.append("✗ Poor identification")

  ax13.text(0.02,
            0.5,
            '\n'.join(analysis_text),
            va='center',
            ha='left',
            fontsize=10,
            family='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow"))
  ax13.set_title('(m) Performance Analysis')
  ax13.axis('off')

  plt.tight_layout()

  return fig


def theoretical_fixed_point_lms_performance(mu: float, sigma_x2: float, sigma_n2: float,
                                            sigma_e2: float, sigma_w2: float, flter_ord: int):
  """Calculate theoretical performance for fixed-point LMS"""

  N = flter_ord
  denom = 1 - mu * (N + 1) * sigma_x2

  # Equation B.26: Expected coefficient error norm squared
  E_delta_w_norm_sq = (mu * (sigma_n2 + sigma_e2) * (N + 1)) / denom + \
                     ((N + 1) * sigma_w2) / (4 * mu * sigma_x2 * denom)

  # Equation B.32: Excess MSE
  xi_Q = (sigma_e2 + sigma_n2) / denom + \
        ((N + 1) * sigma_w2) / (4 * mu * sigma_x2 * denom)

  return E_delta_w_norm_sq, xi_Q


def analyze_convergence_path(w_hist: _Array, metrics: Dict[str, Any]):
  """
  Analyzes the convergence path on the MSE surface (for the first two coefficients)
  and returns the matplotlib Figure object while printing convergence statistics.
  """
  if w_hist.shape[1] < 2:
    print("Need at least 2 coefficients for convergence path analysis")
    return None

  # Extract first two coefficients (real parts for 2D visualization)
  w0_hist = jnp.real(w_hist[:, 0])
  w1_hist = jnp.real(w_hist[:, 1])

  # Create appropriate ranges for the MSE surface
  w0_min, w0_max = jnp.min(w0_hist), jnp.max(w0_hist)
  w1_min, w1_max = jnp.min(w1_hist), jnp.max(w1_hist)

  # Add some padding around the convergence path
  padding = 0.2
  w0_range = jnp.linspace(w0_min - padding, w0_max + padding, 100)
  w1_range = jnp.linspace(w1_min - padding, w1_max + padding, 100)
  W0, W1 = jnp.meshgrid(w0_range, w1_range)

  # Calculate actual MSE surface using Wiener solution
  is_wiener_info_available = ('wiener_solution' in metrics and
                              'autocorrelation_matrix' in metrics and
                              'crosscorrelation_vector' in metrics and 'desired_signal' in metrics)

  if is_wiener_info_available:
    w_wiener_sol = metrics['wiener_solution']
    w_wiener = jnp.real(w_wiener_sol[:2])  # First 2 coefficients
    R = metrics['autocorrelation_matrix'][:2, :2]  # 2x2 submatrix
    p = metrics['crosscorrelation_vector'][:2]  # First 2 elements

    # MSE = σ_d² - 2Re(w^H p) + w^H R w
    sigma_d2 = jnp.mean(jnp.abs(metrics.get('desired_signal', 1.0))**2)

    # Vectorized MSE calculation
    W0_flat = W0.flatten()
    W1_flat = W1.flatten()
    W_stack = jnp.column_stack([ W0_flat, W1_flat ])

    # Calculate MSE for each point
    mse_values = []
    for i in range(len(W_stack)):
      w_vec = W_stack[i]
      mse = sigma_d2 - 2 * jnp.real(jnp.vdot(w_vec, p)) + jnp.real(jnp.vdot(w_vec, R @ w_vec))
      mse_values.append(mse)

    MSE_surface = jnp.array(mse_values).reshape(W0.shape)

  else:
    # Fallback: simple quadratic surface centered at (0,0)
    MSE_surface = W0**2 + W1**2
    if 'wiener_solution' in metrics and 'autocorrelation_matrix' in metrics:
      print(
          "Warning: Missing 'crosscorrelation_vector' or 'desired_signal' for full Wiener MSE calculation. Using quadratic fallback."
      )

  # Convert to numpy for plotting
  W0_np = np.array(W0)
  W1_np = np.array(W1)
  MSE_surface_np = np.array(MSE_surface)
  w0_hist_np = np.array(w0_hist)
  w1_hist_np = np.array(w1_hist)

  # Create the plot
  fig = plt.figure(figsize=(12, 10))

  # Plot 1: Contour plot with convergence path
  plt.subplot(2, 2, 1)

  # Use logarithmic spacing for contour levels to better show the bowl shape
  min_mse = np.min(MSE_surface_np)
  max_mse = np.max(MSE_surface_np)
  levels = np.logspace(np.log10(min_mse + 1e-10), np.log10(max_mse), 15)

  contour = plt.contour(W0_np,
                        W1_np,
                        MSE_surface_np,
                        levels=levels,
                        alpha=0.6,
                        colors='gray',
                        linewidths=1)
  plt.clabel(contour, inline=True, fontsize=8, fmt='%.2f')

  # Plot convergence path with color indicating iteration
  scatter = plt.scatter(w0_hist_np,
                        w1_hist_np,
                        c=np.arange(len(w0_hist_np)),
                        cmap='viridis',
                        s=20,
                        alpha=0.6)
  plt.colorbar(scatter, label='Iteration')

  # Mark important points
  plt.plot(w0_hist_np[0],
           w1_hist_np[0],
           'go',
           markersize=10,
           markeredgecolor='black',
           label='Start')
  plt.plot(w0_hist_np[-1],
           w1_hist_np[-1],
           'bo',
           markersize=10,
           markeredgecolor='black',
           label='End')

  if 'wiener_solution' in metrics:
    w_wiener_plot = metrics['wiener_solution']
    plt.plot(jnp.real(w_wiener_plot[0]),
             jnp.real(w_wiener_plot[1]),
             'rx',
             markersize=12,
             markeredgewidth=3,
             label='Wiener Solution')

  plt.xlabel('Re(w₀)')
  plt.ylabel('Re(w₁)')
  plt.title('Convergence Path on MSE Surface')
  plt.legend()
  plt.grid(True, alpha=0.3)

  # Plot 2: 3D surface plot
  ax = plt.subplot(2, 2, 2, projection='3d')
  ax.plot_surface(W0_np,
                  W1_np,
                  np.log10(MSE_surface_np + 1e-10),
                  cmap='viridis',
                  alpha=0.7,
                  linewidth=0,
                  antialiased=True)

  # Plot convergence path in 3D
  path_mse = []
  if is_wiener_info_available:
    # Use the calculated Wiener parameters
    for i in range(len(w0_hist_np)):
      w_vec = jnp.array([w0_hist_np[i], w1_hist_np[i]])
      mse = sigma_d2 - 2 * jnp.real(jnp.vdot(w_vec, p)) + jnp.real(jnp.vdot(w_vec, R @ w_vec))
      path_mse.append(float(mse))
  else:
    # Use fallback MSE for path if Wiener info is missing
    for i in range(len(w0_hist_np)):
      path_mse.append(w0_hist_np[i]**2 + w1_hist_np[i]**2)

  path_mse_np_log = np.log10(np.array(path_mse) + 1e-10)

  ax.plot(w0_hist_np, w1_hist_np, path_mse_np_log, 'r-', linewidth=2, alpha=0.8)
  ax.scatter(w0_hist_np[0],
             w1_hist_np[0],
             path_mse_np_log[0],
             color='green',
             s=50,
             edgecolor='black')
  ax.scatter(w0_hist_np[-1],
             w1_hist_np[-1],
             path_mse_np_log[-1],
             color='blue',
             s=50,
             edgecolor='black')

  ax.set_xlabel('Re(w₀)')
  ax.set_ylabel('Re(w₁)')
  ax.set_zlabel('log10(MSE)')
  ax.set_title('3D MSE Surface with Convergence Path')

  # Plot 3: Distance to Wiener solution over time
  plt.subplot(2, 2, 3)
  distances = []
  if 'wiener_solution' in metrics:
    w_wiener_dist = jnp.real(metrics['wiener_solution'][:2])
    for i in range(len(w0_hist_np)):
      dist = jnp.sqrt((w0_hist_np[i] - w_wiener_dist[0])**2 + (w1_hist_np[i] - w_wiener_dist[1])**2)
      distances.append(dist)

    plt.semilogy(distances, 'r-', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Distance to Wiener Solution')
    plt.title('Convergence to Optimal Solution')
    plt.grid(True, alpha=0.3)
  else:
    plt.text(0.5,
             0.5,
             'Wiener Solution not available for this plot.',
             horizontalalignment='center',
             verticalalignment='center',
             transform=plt.gca().transAxes)
    plt.title('Convergence to Optimal Solution')

  # Plot 4: Step size analysis
  plt.subplot(2, 2, 4)
  steps = []
  for i in range(1, len(w0_hist_np)):
    step_size = jnp.sqrt((w0_hist_np[i] - w0_hist_np[i - 1])**2 +
                         (w1_hist_np[i] - w1_hist_np[i - 1])**2)
    steps.append(step_size)

  plt.semilogy(steps, 'g-', linewidth=2)
  plt.xlabel('Iteration')
  plt.ylabel('Step Size')
  plt.title('Adaptation Step Size Over Time')
  plt.grid(True, alpha=0.3)

  plt.tight_layout()

  # Print convergence statistics (KEPT AS REQUESTED)
  print("\n=== CONVERGENCE ANALYSIS ===")
  print(f"Total iterations: {len(w0_hist_np)}")
  print(f"Initial coefficients: w0={w0_hist_np[0]:.4f}, w1={w1_hist_np[0]:.4f}")
  print(f"Final coefficients: w0={w0_hist_np[-1]:.4f}, w1={w1_hist_np[-1]:.4f}")

  if 'wiener_solution' in metrics:
    w_wiener_stats = metrics['wiener_solution']
    print(
        f"Wiener solution: w0={jnp.real(w_wiener_stats[0]):.4f}, w1={jnp.real(w_wiener_stats[1]):.4f}"
    )
    if distances:  # Check if distances were calculated
      final_distance = distances[-1]
      print(f"Final distance to Wiener solution: {final_distance:.6f}")
    else:
      final_distance = jnp.sqrt((w0_hist_np[-1] - jnp.real(w_wiener_stats[0]))**2 +
                                (w1_hist_np[-1] - jnp.real(w_wiener_stats[1]))**2)
      print(f"Final distance to Wiener solution: {final_distance:.6f}")

  if steps:  # Check if steps list is not empty
    print(f"Total path length: {np.sum(steps):.4f}")
    print(f"Average step size: {np.mean(steps):.6f}")
  else:
    print("Not enough iterations to calculate path length or average step size.")

  return fig


def analyze_convergence_path_2d(w_hist: _Array, metrics: Dict[str, Any]):
  """
  Simplified version of convergence path analysis (2D) that returns the Figure.
  """
  if w_hist.shape[1] < 2:
    print("Need at least 2 coefficients for convergence path analysis")
    return None  # Return None if the condition is not met

  # Extract first two coefficients
  w0_hist = jnp.real(w_hist[:, 0])
  w1_hist = jnp.real(w_hist[:, 1])

  # Create appropriate ranges
  padding = 0.3
  w0_range = jnp.linspace(jnp.min(w0_hist) - padding, jnp.max(w0_hist) + padding, 50)
  w1_range = jnp.linspace(jnp.min(w1_hist) - padding, jnp.max(w1_hist) + padding, 50)
  W0, W1 = jnp.meshgrid(w0_range, w1_range)

  # Calculate MSE surface
  if 'wiener_solution' in metrics:
    w_wiener = metrics['wiener_solution']
    # Simple quadratic surface centered at Wiener solution (Approximation)
    MSE_surface = (W0 - jnp.real(w_wiener[0]))**2 + (W1 - jnp.real(w_wiener[1]))**2
  else:
    # Fallback surface centered at origin
    MSE_surface = W0**2 + W1**2

  fig = plt.figure(figsize=(10, 8))

  # Use better contour levels
  # Convert to numpy for min/max
  MSE_surface_np = np.array(MSE_surface)
  levels = np.linspace(np.min(MSE_surface_np), np.max(MSE_surface_np), 10)

  # Ensure W0, W1 are numpy arrays for contour plotting if they were JAX arrays
  W0_np = np.array(W0)
  W1_np = np.array(W1)

  contour = plt.contour(W0_np, W1_np, MSE_surface_np, levels=levels, alpha=0.6, colors='gray')
  plt.clabel(contour, inline=True, fontsize=9, fmt='%.1f')

  # Convert path to numpy for plotting robustness
  w0_hist_np = np.array(w0_hist)
  w1_hist_np = np.array(w1_hist)

  # Plot convergence path
  plt.plot(w0_hist_np, w1_hist_np, 'r-', linewidth=2, alpha=0.7, label='Convergence Path')
  plt.plot(w0_hist_np[0],
           w1_hist_np[0],
           'go',
           markersize=10,
           label='Start',
           markeredgecolor='black')
  plt.plot(w0_hist_np[-1],
           w1_hist_np[-1],
           'bo',
           markersize=10,
           label='End',
           markeredgecolor='black')

  if 'wiener_solution' in metrics:
    w_wiener_plot = metrics['wiener_solution']
    plt.plot(jnp.real(w_wiener_plot[0]),
             jnp.real(w_wiener_plot[1]),
             'rx',
             markersize=15,
             markeredgewidth=2,
             label='Wiener Solution')

  plt.xlabel('Re(w0)')
  plt.ylabel('Re(w1)')
  plt.title('Convergence Path on MSE Surface')
  plt.legend()
  plt.grid(True, alpha=0.3)

  return fig


def run_multiple_trials(eq: ChannelEqualization,
                        n_samples: int,
                        n_trials: int = 10,
                        use_scan: bool = True) -> Tuple[_Array, _Array]:
  """Run multiple trials and average results."""
  all_w_hist = []
  all_e_hist = []
  all_metrics = []

  for trial in range(n_trials):
    print(f"Running trial {trial + 1}/{n_trials}...")

    # Reset the equalizer state for each trial
    dummy_in = jnp.zeros(eq.lms.filter_order + 1, dtype=jnp.complex64)
    eq.key, subkey = jax.random.split(eq.key)
    eq.init_vars = eq.lms.init(subkey, dummy_in)

    # Run simulation
    w_hist, e_hist, metrics = eq.run(n_samples, use_scan=use_scan)

    all_w_hist.append(w_hist)
    all_e_hist.append(e_hist)
    all_metrics.append(metrics)

  # Average results
  avg_w_hist = jnp.mean(jnp.array(all_w_hist), axis=0)
  avg_e_hist = jnp.mean(jnp.array(all_e_hist), axis=0)
  avg_mse = jnp.mean(jnp.array([m['mse_curve'] for m in all_metrics]), axis=0)

  # Create averaged metrics
  avg_metrics = {
      'mse_curve': avg_mse,
      'wiener_solution': all_metrics[0]['wiener_solution'],  # Same for all trials
      'final_mse': jnp.mean(jnp.array([m['final_mse'] for m in all_metrics])),
      'final_coeff_error': jnp.mean(jnp.array([m['final_coeff_error'] for m in all_metrics]))
  }

  return avg_w_hist, avg_e_hist, avg_metrics, all_metrics
