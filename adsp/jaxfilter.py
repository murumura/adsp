import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import scipy
from scipy import signal
from typing import Callable, Tuple, Dict, Any, Optional, Union, List
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


def verify_R_calculation(a: float, order: int, n_samples: int = 10000, key: jax.random.PRNGKey = None):
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

  def excess_mse(self, R: jnp.ndarray, delta_w_cov: jnp.ndarray) -> float:
    """
    Calculate excess MSE using Equation 3.42
    
    Args:
        R: Input correlation matrix, shape (N+1, N+1)
        delta_w_cov: Covariance of coefficient errors, shape (N+1, N+1)
        
    Returns:
        Excess MSE value
    """
    # Method 1: tr[R * cov[Δw]]
    excess = jnp.trace(R @ delta_w_cov)
    return excess

  def is_stable(self, R: jnp.ndarray) -> bool:
    """Check stability condition by Equation 3.19 and Equation 3.30"""
    eigenvalues = jnp.linalg.eigvals(R)
    max_eigenvalue = jnp.max(jnp.real(eigenvalues))
    trace_R = jnp.trace(R)
    return 0 < self.mu < 1.0 / max_eigenvalue and 0 < self.mu < 1.0 / trace_R

  def reason_mu(self, R: jnp.ndarray, verbose: bool = False) -> Dict[str, Any]:
    """Analyze step size and suggest improvements"""
    eigenvalues = jnp.linalg.eigvals(R)
    lambda_max = jnp.max(jnp.real(eigenvalues))
    lambda_min = jnp.min(jnp.real(eigenvalues))
    trace_R = jnp.trace(R)
    
    # Stability bounds from textbook
    bound_eigenvalue = 1.0 / lambda_max
    bound_trace = 1.0 / trace_R  # More practical bound (Equation 3.30)
    
    # Current stability
    is_stable = self.is_stable(R)
    
    # Suggested ranges
    safe_mu = 0.1 * bound_trace  # Conservative choice
    aggressive_mu = 0.5 * bound_trace  # More aggressive but potentially unstable
    
    # Eigenvalue spread analysis
    eigenvalue_spread = lambda_max / lambda_min if lambda_min > 0 else jnp.inf
    
    # Calculate theoretical misadjustment (Equation 3.50)
    theoretical_misadjustment = (self.mu * trace_R) / (1 - self.mu * trace_R)
    
    result = {
        'is_stable': is_stable,
        'current_mu': self.mu,
        'max_stable_mu_eigenvalue': bound_eigenvalue,
        'max_stable_mu_trace': bound_trace,
        'suggested_conservative_mu': safe_mu,
        'suggested_aggressive_mu': aggressive_mu,
        'eigenvalue_spread': eigenvalue_spread,
        'convergence_warning': eigenvalue_spread > 100,  # High spread slows convergence
        'theoretical_misadjustment': theoretical_misadjustment,
        'lambda_max': lambda_max,
        'lambda_min': lambda_min,
        'trace_R': trace_R
    }
    
    if verbose:
        self._print_mu_analysis(result)
    
    return result

  def _print_mu_analysis(self, analysis: Dict[str, Any]):
    """Print detailed analysis of step size configuration"""
    print("\n" + "="*60)
    print("LMS STEP SIZE ANALYSIS")
    print("="*60)
    
    print(f"Current μ: {analysis['current_mu']:.6f}")
    print(f"Stability: {'✓ STABLE' if analysis['is_stable'] else '✗ UNSTABLE'}")
    
    print(f"\nEIGENVALUE ANALYSIS:")
    print(f"  Max eigenvalue (λ_max): {analysis['lambda_max']:.6f}")
    print(f"  Min eigenvalue (λ_min): {analysis['lambda_min']:.6f}")
    print(f"  Eigenvalue spread: {analysis['eigenvalue_spread']:.2f}")
    if analysis['convergence_warning']:
        print(f"  ⚠ WARNING: High eigenvalue spread will slow convergence")
    
    print(f"\nSTABILITY BOUNDS:")
    print(f"  From eigenvalues (1/λ_max): {analysis['max_stable_mu_eigenvalue']:.6f}")
    print(f"  From trace (1/tr[R]): {analysis['max_stable_mu_trace']:.6f}")
    
    print(f"\nPERFORMANCE PREDICTIONS:")
    print(f"  Theoretical misadjustment: {analysis['theoretical_misadjustment']:.4f}")
    
    # Calculate approximate convergence iterations
    if analysis['lambda_min'] > 0:
        slowest_time_constant = 1 / (2 * analysis['current_mu'] * analysis['lambda_min'])
        approx_iterations = 4.6 * slowest_time_constant  # ~100x attenuation
        print(f"  Approx. convergence iterations: {approx_iterations:.0f}")
    
    print(f"\nRECOMMENDED μ RANGES:")
    print(f"  Conservative (good stability): {analysis['suggested_conservative_mu']:.6f}")
    print(f"  Aggressive (faster convergence): {analysis['suggested_aggressive_mu']:.6f}")
    
    # Safety margin analysis
    margin_eigenvalue = analysis['max_stable_mu_eigenvalue'] / analysis['current_mu'] if analysis['current_mu'] > 0 else float('inf')
    margin_trace = analysis['max_stable_mu_trace'] / analysis['current_mu'] if analysis['current_mu'] > 0 else float('inf')
    
    print(f"\nSAFETY MARGINS:")
    print(f"  Eigenvalue bound margin: {margin_eigenvalue:.2f}x")
    print(f"  Trace bound margin: {margin_trace:.2f}x")
    
    if margin_eigenvalue < 2:
        print(f"  ⚠ WARNING: Small stability margin")
    
    print("="*60)

class AdaptiveProblem(ABC):

  def __init__(self,
               lms: Any,
               gen_signals_fn: Callable,
               n_ensemble: int = 1,
               wiener_sol_fn: Optional[Callable] = None,
               seed: int = 42,
               **kwargs):
    """
      Base class for adaptive filtering problems.
      
      Args:
          lms: LMS filter instance
          gen_signals_fn: Function that generates signals for the specific problem
          n_ensemble: Number of ensemble members
          wiener_sol_fn: Function to compute Wiener solution (optional)
          seed: Random seed
    """
    self.lms = lms
    self.gen_signals_fn = gen_signals_fn
    self.key = jax.random.PRNGKey(seed)
    self.wiener_sol_fn = wiener_sol_fn
    self.n_ensemble = n_ensemble
    self.vars = self.initialize_filter()
    # Fixed parameters for misadjustment calculation
    self.sigma_x2 = kwargs.get('sigma_x2', 1.0)

    # Get theoretical misadjustment value from kwargs or use default
    # function to compute it
    self.theoretical_misadjustment_val = kwargs.get('theoretical_misadjustment_val', None)

  def initialize_filter(self) -> _Array:
    """Initialize filter variables with dummy signals"""
    dummy_x = jnp.zeros(10, dtype=jnp.float64)
    dummy_d = jnp.zeros(10, dtype=jnp.float64)
    return self.lms.init(self.key, dummy_x, dummy_d)

  def generate_signals(self, n_samples: int, ensemble_idx: int = 0, **kwargs) -> Tuple:
    """Generate signals for the specific problem"""
    ensemble_key = jax.random.fold_in(self.key, ensemble_idx)
    signals = self.gen_signals_fn(n_samples, key=ensemble_key, **kwargs)

    # Handle different signal return formats
    if len(signals) == 2:
      x, d = signals
      return None, x, d, None
    elif len(signals) == 3:
      x, d, channel_info = signals
      return None, x, d, None
    else:
      return signals

  def run_adaptation(self, x: _Array, d: _Array, training: bool = True) -> Tuple[_Array, _Array, _Array]:
    """Run LMS adaptation on given signals"""
    return self.lms.apply(self.vars, x, d, train=training)

  def _stack_h_true_ensemble(self, all_h_true: List) -> Any:
    """Stack h_true ensemble with proper shape handling"""
    if all_h_true[0] is None:
      return None

    first_h = all_h_true[0]
    if first_h.ndim == 1:  # Stationary channel
      return first_h
    elif first_h.ndim == 2:  # Time-varying
      return jnp.stack(all_h_true)
    else:
      return first_h

  def run(self, n_samples: int, training: bool = True, **kwargs):
    """Run the complete adaptive filtering experiment"""
    ensemble_data = self._collect_ensemble_data(n_samples, training, **kwargs)
    ensemble_metrics = self._compute_ensemble_metrics(ensemble_data)

    return (ensemble_data['w_hist'], ensemble_data['e_hist'], ensemble_metrics, ensemble_data['return_h_true'])

  def _collect_ensemble_data(self, n_samples: int, training: bool = True, **kwargs):
    """Collect ensemble data efficiently"""
    all_w_hist, all_e_hist, all_h_true, all_x_hist, all_d_hist, all_measured_noise = [], [], [], [], [], []

    for i in range(self.n_ensemble):
      h_true, x, d, measured_noise = self.generate_signals(n_samples, ensemble_idx=i, **kwargs)
      _, e_hist, w_hist = self.run_adaptation(x, d, training)

      all_w_hist.append(w_hist)
      all_e_hist.append(e_hist)
      all_h_true.append(h_true)
      all_x_hist.append(x)
      all_d_hist.append(d)
      all_measured_noise.append(measured_noise)

    return {
        'w_hist': jnp.stack(all_w_hist),
        'e_hist': jnp.stack(all_e_hist),
        'x_hist': jnp.stack(all_x_hist),
        'd_hist': jnp.stack(all_d_hist),
        'h_true': all_h_true,
        'measured_noise': all_measured_noise,
        'return_h_true': self._stack_h_true_ensemble(all_h_true)
    }

  def _compute_ensemble_metrics(self, ensemble_data: Dict) -> Dict[str, Any]:
    """Compute all ensemble metrics in one place"""
    all_x_hist, all_d_hist = ensemble_data['x_hist'], ensemble_data['d_hist']
    all_w_hist, all_e_hist = ensemble_data['w_hist'], ensemble_data['e_hist']
    all_h_true, all_measured_noise = ensemble_data['h_true'], ensemble_data['measured_noise']

    n_ensemble, n_samples = all_e_hist.shape

    # Core metrics
    avg_mse_curve = jnp.mean(jnp.abs(all_e_hist)**2, axis=0)
    w_opt_ensemble, R_ensemble, p_ensemble = self.ensemble_wiener_solution(all_x_hist, all_d_hist)
    avg_w_error = jnp.mean(jnp.linalg.norm(all_w_hist - w_opt_ensemble, axis=2), axis=0)

    # Final metrics
    last_n = min(100, n_samples)
    final_mse = jnp.mean(avg_mse_curve[-last_n:])
    #final_mse = jnp.sum(avg_mse_curve[-last_n:]) / n_ensemble
    final_coeff = jnp.mean(all_w_hist[:, -1, :], axis=0)

    # Noise handling
    min_mse_ensemble = self._compute_min_mse(all_measured_noise, avg_mse_curve, last_n)

    # Misadjustment
    mu = self.step_size()
    trace_R = jnp.trace(R_ensemble)
    misadjustment = (final_mse - min_mse_ensemble) / min_mse_ensemble if min_mse_ensemble > 0 else 0.0

    if self.theoretical_misadjustment_val is None:
      theoretical_misadjustment = self._default_misadjustment_fn(mu, trace_R)
    else:
      theoretical_misadjustment = self.theoretical_misadjustment_val

    misadjustment_curve = (avg_mse_curve - min_mse_ensemble) / min_mse_ensemble

    # Convergence analysis
    convergence_metrics = self.analyze_convergence(all_e_hist, all_w_hist,
                                                   all_h_true[0] if all_h_true[0] is not None else None, w_opt_ensemble)

    # Base metrics
    ensemble_metrics = {
        "wiener_solution": w_opt_ensemble,
        "R": R_ensemble,
        "p": p_ensemble,
        "final_coeff": final_coeff,
        "final_mse": final_mse,
        "final_coeff_error": jnp.mean(avg_w_error[-last_n:]),
        "step_size": self.step_size(),
        "avg_mse_curve": avg_mse_curve,
        "avg_coeff_error_curve": avg_w_error,
        "min_mse": min_mse_ensemble,
        "misadjustment": misadjustment,
        "misadjustment_theoretical": theoretical_misadjustment,
        "misadjustment_curve": misadjustment_curve,
        "n_ensemble": self.n_ensemble,
        "eigenvalues": jnp.linalg.eigvals(R_ensemble),
        "max_eigenvalue": jnp.max(jnp.real(jnp.linalg.eigvals(R_ensemble))),
        "min_eigenvalue": jnp.min(jnp.real(jnp.linalg.eigvals(R_ensemble))),
        "trace_R": trace_R,
        **convergence_metrics  # Include convergence metrics
    }

    # Add h_true metrics if available
    ensemble_metrics.update(self._compute_h_true_metrics(all_w_hist, all_h_true, last_n))

    # Problem-specific metrics
    return self._add_problem_specific_metrics(ensemble_metrics, ensemble_data)

  def _default_misadjustment_fn(self, mu: float, trace_R: float) -> float:
    """
    Default misadjustment calculation with proper stability check
    
    Args:
        mu: Convergence factor
        trace_R: Trace of input correlation matrix
        sigma_n2: Measurement noise variance
        
    Returns:
        Theoretical misadjustment value
    """
    # Check stability condition
    if mu * trace_R >= 1:
      return float('inf')  # Unstable case

    return (mu * trace_R) / (1 - mu * trace_R)

  def _compute_min_mse(self, all_measured_noise: List, avg_mse_curve: _Array, last_n: int) -> float:
    """Compute minimum MSE from noise or steady-state"""
    if all_measured_noise[0] is not None:
      all_noise_stacked = jnp.stack([jnp.abs(noise)**2 for noise in all_measured_noise])
      return jnp.mean(all_noise_stacked)
    else:
      return jnp.mean(avg_mse_curve[-last_n:])

  def _compute_h_true_metrics(self, all_w_hist: _Array, all_h_true: List, last_n: int) -> Dict[str, Any]:
    """Compute metrics related to true system coefficients"""
    if all_h_true[0] is None:
      return {}

    first_h_true = all_h_true[0]
    metrics = {}

    if first_h_true.ndim == 1:  # Stationary
      avg_coeff_error_true = jnp.mean(jnp.linalg.norm(all_w_hist - first_h_true, axis=2), axis=0)
      metrics.update({
          "avg_coeff_error_true_curve": avg_coeff_error_true,
          "final_coeff_error_true": jnp.mean(avg_coeff_error_true[-last_n:]),
          "unknown_system": first_h_true,
      })
    elif first_h_true.ndim == 2:  # Time-varying
      all_h_true_stacked = jnp.stack(all_h_true)
      coeff_error_true = jnp.linalg.norm(all_w_hist - all_h_true_stacked, axis=2)
      avg_coeff_error_true = jnp.mean(coeff_error_true, axis=0)
      metrics.update({
          "avg_coeff_error_true_curve": avg_coeff_error_true,
          "final_coeff_error_true": jnp.mean(avg_coeff_error_true[-last_n:]),
          "unknown_system": all_h_true_stacked[0],
          "h_true_ensemble": all_h_true_stacked,
      })

    return metrics

  def analyze_convergence(self,
                          e_hist: _Array,
                          w_hist: _Array,
                          h_true: _Array = None,
                          w_opt: _Array = None) -> Dict[str, Any]:
    """Analyze convergence behavior for ensemble data"""
    # For ensemble mode, we always have ensemble data
    mse_curve = jnp.mean(jnp.abs(e_hist)**2, axis=0)
    coeff_error_wiener = jnp.mean(jnp.linalg.norm(w_hist - w_opt, axis=2), axis=0)

    # Find convergence iterations
    conv_iter_10dB = find_convergence_iteration(mse_curve, -10)
    conv_iter_20dB = find_convergence_iteration(mse_curve, -20)
    conv_iter_30dB = find_convergence_iteration(mse_curve, -30)

    convergence_metrics = {
        "convergence_10dB_iter": conv_iter_10dB,
        "convergence_20dB_iter": conv_iter_20dB,
        "convergence_30dB_iter": conv_iter_30dB,
        "final_coeff_error_wiener": jnp.mean(coeff_error_wiener[-100:]),
        "coeff_error_wiener_curve": coeff_error_wiener,
    }

    # Add true coefficient error if h_true is available
    if h_true is not None:
      if h_true.ndim == 1:  # Stationary system
        coeff_error_true = jnp.mean(jnp.linalg.norm(w_hist - h_true, axis=2), axis=0)
      else:  # Time-varying system
        # For ensemble analysis with time-varying h_true, we need to handle this case
        # This assumes h_true is the same for all ensemble members (first one)
        coeff_error_true = jnp.mean(jnp.linalg.norm(w_hist - h_true, axis=2), axis=0)

      convergence_metrics.update({
          "final_coeff_error_true": jnp.mean(coeff_error_true[-100:]),
          "coeff_error_true_curve": coeff_error_true,
      })

    return convergence_metrics

  def timeavg_wiener_solution(self, x: _Array, d: _Array) -> Tuple[_Array, _Array, _Array]:
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

  def ensemble_wiener_solution(self, all_x: _Array, all_d: _Array) -> Tuple[_Array, _Array, _Array]:
    """Compute Wiener solution using ensemble averages"""
    n_ensemble, n_samples = all_x.shape
    N_coef = self.lms.filter_order + 1

    # Ensemble average of autocorrelation matrix
    R_ensemble = jnp.zeros((N_coef, N_coef), dtype=all_x.dtype)
    for i in range(N_coef):
      for j in range(i, N_coef):
        lag = abs(i - j)
        # Average correlation across ensemble
        corr_sum = 0.0
        for k in range(n_ensemble):
          corr_sum += jnp.mean(jnp.conj(all_x[k, lag:]) * all_x[k, :n_samples - lag])
        R_ensemble = R_ensemble.at[i, j].set(corr_sum / n_ensemble)
        if i != j:
          R_ensemble = R_ensemble.at[j, i].set(jnp.conj(corr_sum / n_ensemble))

    # Ensemble average of cross-correlation vector
    p_ensemble = jnp.zeros(N_coef, dtype=all_x.dtype)
    for i in range(N_coef):
      corr_sum = 0.0
      for k in range(n_ensemble):
        corr_sum += jnp.mean(jnp.conj(all_d[k, i:]) * all_x[k, :n_samples - i])
      p_ensemble = p_ensemble.at[i].set(corr_sum / n_ensemble)

    # Add regularization and solve
    R_reg = R_ensemble + 1e-8 * jnp.eye(N_coef, dtype=all_x.dtype)
    w_ensemble = jnp.linalg.solve(R_reg, p_ensemble)

    return w_ensemble, R_ensemble, p_ensemble

  def wiener_solution(self, x: _Array, d: _Array) -> Tuple[_Array, _Array, _Array]:
    """Compute Wiener solution using appropriate method"""
    if self.wiener_sol_fn is not None:
      return self.wiener_sol_fn(x, d, self.lms.filter_order)
    elif hasattr(x, 'shape') and len(x.shape) > 1:
      # If we have ensemble data, use ensemble Wiener solution
      return self.ensemble_wiener_solution(x, d)
    else:
      # Default to time average for single realization
      return self.timeavg_wiener_solution(x, d)

  def get_problem_type(self) -> str:
    """Return the type of adaptive problem"""
    return self.__class__.__name__

  def step_size(self):
    return self.lms.mu


class ChannelEqualization(AdaptiveProblem):

  def _add_problem_specific_metrics(self, ensemble_metrics: Dict, ensemble_data: Dict) -> Dict[str, Any]:
    ensemble_metrics["problem_type"] = "channel_equalization"
    return ensemble_metrics


class SystemIdentification(AdaptiveProblem):

  def _add_problem_specific_metrics(self, ensemble_metrics: Dict, ensemble_data: Dict) -> Dict[str, Any]:
    ensemble_metrics["problem_type"] = "system_identification"
    return ensemble_metrics


def _generate_summary_text(metrics: Dict[str, Any], n_iters: int, problem_type: str, n_ensemble: int) -> List[str]:
  """Generate comprehensive summary text for the plot"""
  summary_text = []

  summary_text.append(f"Problem Type: {problem_type.replace('_', ' ').title()}")
  summary_text.append(f"Iterations: {n_iters}")
  summary_text.append(f"Ensemble Size: {n_ensemble}")
  summary_text.append("-" * 40)

  # Basic metrics - use ensemble metrics
  if 'final_mse' in metrics:
    summary_text.append(f"Final MSE: {metrics['final_mse']:.2e}")
  if 'min_mse' in metrics:
    summary_text.append(f"Minimum MSE: {metrics['min_mse']:.2e}")

  # Misadjustment
  if 'misadjustment' in metrics and 'misadjustment_theoretical' in metrics:
    summary_text.append(f"Misadjustment: {metrics['misadjustment']:.4f} (Exp)")
    summary_text.append(f"Misadjustment: {metrics['misadjustment_theoretical']:.4f} (Theo)")

  # Problem-specific metrics
  summary_text.append("-" * 40)
  if problem_type == 'time_varying_system_identification':
    summary_text.append("TIME-VARYING SYSTEM:")
    if 'final_coeff_tracking_error' in metrics:
      summary_text.append(f"Tracking Error: {metrics['final_coeff_tracking_error']:.4f}")

  elif problem_type == 'channel_equalization':
    summary_text.append("CHANNEL EQUALIZATION:")
    # Remove metrics that don't exist in ensemble mode
    summary_text.append("Equalizer Performance:")

  # Convergence metrics - use ensemble metrics
  summary_text.append("-" * 40)
  if 'final_coeff_error_true' in metrics:
    summary_text.append(f"Coeff Error (True): {metrics['final_coeff_error_true']:.2e}")
  if 'final_coeff_error' in metrics:  # This is Wiener error in ensemble mode
    summary_text.append(f"Coeff Error (Wiener): {metrics['final_coeff_error']:.2e}")

  # System parameters
  summary_text.append("-" * 40)
  if 'max_eigenvalue' in metrics and 'min_eigenvalue' in metrics:
    spread = metrics['max_eigenvalue'] / metrics['min_eigenvalue']
    summary_text.append(f"Eigenvalue Spread: {spread:.2f}")
  if 'trace_R' in metrics:
    summary_text.append(f"Trace(R): {metrics['trace_R']:.4f}")

  # Step size info
  mu = metrics.get('step_size', 'N/A')
  summary_text.append(f"Step Size μ: {mu}")

  return summary_text


def plot_results(w_hist: _Array, e_hist: _Array, metrics: Dict[str, Any]):
  """
    Plot comprehensive adaptive filtering results for ensemble-only mode
    """
  # Handle ensemble data shape: (n_ensemble, n_samples) for e_hist
  if e_hist.ndim == 2:  # Ensemble data
    n_ensemble, n_iters = e_hist.shape
    # Use first ensemble member for single trajectory plots
    e_hist_single = e_hist[0] if n_ensemble > 0 else e_hist
    w_hist_single = w_hist[0] if n_ensemble > 0 else w_hist
  else:  # Single realization (fallback)
    n_iters = len(e_hist)
    n_ensemble = 1
    e_hist_single = e_hist
    w_hist_single = w_hist

  n_coeffs = w_hist_single.shape[1] if w_hist_single.ndim > 1 else 1

  # Determine problem type
  problem_type = metrics.get('problem_type', 'unknown')
  n_ensemble = metrics.get('n_ensemble', n_ensemble)

  # Create figure layout
  fig = plt.figure(figsize=(18, 20))
  gs = gridspec.GridSpec(5, 3, figure=fig)

  # Plot 1: Learning curve (MSE) - Ensemble Average
  ax1 = fig.add_subplot(gs[0, 0])

  # Use ensemble average MSE curve
  if 'avg_mse_curve' in metrics:
    mse_curve = metrics['avg_mse_curve']
  else:
    # Fallback: average across ensemble
    mse_curve = jnp.mean(jnp.abs(e_hist)**2, axis=0)

  mse_curve_np = np.asarray(mse_curve)

  ax1.semilogy(mse_curve_np, 'b-', alpha=0.7, linewidth=1, label='Ensemble Average')

  # Mark convergence points
  convergence_colors = {'convergence_10dB_iter': 'r', 'convergence_20dB_iter': 'g', 'convergence_30dB_iter': 'm'}
  for conv_key, color in convergence_colors.items():
    if conv_key in metrics and metrics[conv_key] is not None:
      ax1.axvline(x=metrics[conv_key], color=color, linestyle='--', alpha=0.7, label=f'{conv_key.split("_")[1]}dB')

  ax1.set_ylabel('MSE')
  ax1.set_xlabel('Iteration')
  ax1.set_title(f'(a) Learning Curve ({problem_type.replace("_", " ").title()})')
  ax1.grid(True, alpha=0.3)
  if any(key in metrics for key in convergence_colors.keys()):
    ax1.legend()

  # Plot 2: Misadjustment Learning Curve
  ax2 = fig.add_subplot(gs[0, 1])
  if 'misadjustment_curve' in metrics:
    misadjustment_curve = np.asarray(metrics['misadjustment_curve'])
    ax2.plot(misadjustment_curve, 'g-', alpha=0.7, linewidth=1, label='Empirical Misadjustment')

    if 'misadjustment_theoretical' in metrics:
      theoretical_m = metrics['misadjustment_theoretical']
      ax2.axhline(y=theoretical_m, color='r', linestyle='--', linewidth=2, label=f'Theoretical M = {theoretical_m:.3f}')

    if 'misadjustment' in metrics:
      final_m = metrics['misadjustment']
      ax2.axhline(y=final_m, color='b', linestyle=':', linewidth=2, label=f'Final M = {final_m:.3f}')

    ax2.set_ylabel('Misadjustment M(k)')
    ax2.set_xlabel('Iteration')
    ax2.set_title('(b) Misadjustment Learning Curve')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
  else:
    ax2.text(0.5, 0.5, 'No misadjustment curve data', ha='center', va='center', transform=ax2.transAxes)
    ax2.set_title('(b) Misadjustment Learning Curve')
    ax2.grid(True, alpha=0.3)

  # Plot 3: Coefficient convergence - Show first ensemble member
  ax3 = fig.add_subplot(gs[0, 2])

  if problem_type == 'time_varying_system_identification' and 'h_true_ensemble' in metrics:
    # Time-varying coefficient tracking
    h_true_tv = metrics['h_true_ensemble'][0]  # First ensemble member
    for i in range(min(3, n_coeffs)):
      ax3.plot(w_hist_single[:, i], label=f'w[{i}]', alpha=0.7, linewidth=1)
      if i < h_true_tv.shape[1]:
        ax3.plot(h_true_tv[:n_iters, i], '--', label=f'h_true[{i}]', alpha=0.8, linewidth=1.5)
    ax3.set_title('(c) Time-Varying Coefficient Tracking')

  elif problem_type == 'channel_equalization':
    # Equalizer coefficients - first ensemble member
    for i in range(min(4, n_coeffs)):
      ax3.plot(w_hist_single[:, i], label=f'w[{i}]', alpha=0.7)
    if 'wiener_solution' in metrics:
      w_opt = metrics['wiener_solution']
      for i in range(min(4, len(w_opt))):
        ax3.axhline(y=w_opt[i], color=f'C{i}', linestyle='--', linewidth=2, label=f'w_opt[{i}]', alpha=0.8)
    ax3.set_title('(c) Equalizer Coefficients')

  elif 'unknown_system' in metrics:
    # Stationary system identification - first ensemble member
    h_true = metrics['unknown_system']
    for i in range(min(4, n_coeffs)):
      ax3.plot(w_hist_single[:, i], label=f'w[{i}]', alpha=0.7)
      if i < len(h_true):
        ax3.axhline(y=h_true[i], color=f'C{i}', linestyle='--', linewidth=2, label=f'h_true[{i}]', alpha=0.8)
    ax3.set_title('(c) Coefficient Convergence vs True System')
  else:
    # Generic coefficient plot
    for i in range(min(4, n_coeffs)):
      ax3.plot(w_hist_single[:, i], label=f'w[{i}]', alpha=0.7)
    ax3.set_title('(c) Coefficient Convergence')

  ax3.set_ylabel('Coefficient Value')
  ax3.set_xlabel('Iteration')
  ax3.legend(ncol=2, fontsize=8)
  ax3.grid(True, alpha=0.3)

  # Plot 4: Coefficient error comparison - Ensemble averages
  ax4 = fig.add_subplot(gs[1, 0])
  error_curves = []
  error_labels = []

  # Use ensemble average error curves
  if 'avg_coeff_error_curve' in metrics:
    error_curves.append(metrics['avg_coeff_error_curve'])
    error_labels.append('Ensemble Avg Error vs Wiener')

  if 'avg_coeff_error_true_curve' in metrics:
    error_curves.append(metrics['avg_coeff_error_true_curve'])
    error_labels.append('Ensemble Avg Error vs True')

  if problem_type == 'time_varying_system_identification' and 'coeff_tracking_error_curve' in metrics:
    error_curves.append(metrics['coeff_tracking_error_curve'])
    error_labels.append('Tracking Error')

  for curve, label in zip(error_curves, error_labels):
    ax4.semilogy(curve, label=label, alpha=0.7)

  ax4.set_ylabel('Coefficient Error Norm')
  ax4.set_xlabel('Iteration')
  ax4.set_title('(d) Coefficient Error Comparison')
  if error_curves:
    ax4.legend()
  ax4.grid(True, alpha=0.3)

  # Plot 5: Eigenvalue analysis
  ax5 = fig.add_subplot(gs[1, 1])
  if 'eigenvalues' in metrics:
    eigenvals = np.abs(metrics['eigenvalues'])
    ax5.stem(np.real(eigenvals), basefmt=" ")
    ax5.set_ylabel('Eigenvalue Magnitude')
    ax5.set_xlabel('Eigenvalue Index')

    max_e = np.max(eigenvals)
    min_e = np.min(eigenvals)
    spread = max_e / min_e if min_e > 1e-10 else np.inf
    spread_str = f"Spread: {spread:.2f}" if spread != np.inf else "Spread: Inf"

    ax5.set_title(f'(e) Eigenvalue Distribution\n{spread_str}')
  else:
    ax5.text(0.5, 0.5, 'No eigenvalue data', ha='center', va='center', transform=ax5.transAxes)
    ax5.set_title('(e) Eigenvalue Distribution')
  ax5.grid(True, alpha=0.3)

  # Plot 6: Performance metrics (Bar chart)
  ax6 = fig.add_subplot(gs[1, 2])
  performance_data = []
  performance_labels = []
  performance_colors = []

  # Basic metrics for all problem types
  metric_config = [
      ('misadjustment', 'Final Misadjustment', 'skyblue'),
      ('misadjustment_theoretical', 'Theoretical Misadjustment', 'lightcoral'),
      ('final_mse', 'Final MSE', 'lightgreen'),
      ('min_mse', 'Min MSE', 'lightyellow'),
      ('final_coeff_error', 'Coeff Error (Wiener)', 'lightpink'),
      ('final_coeff_error_true', 'Coeff Error (True)', 'lightcyan'),
  ]

  # Problem-specific metrics
  if problem_type == 'time_varying_system_identification':
    metric_config.extend([
        ('final_coeff_tracking_error', 'Tracking Error', 'brown'),
    ])

  for key, label, color in metric_config:
    if key in metrics and metrics[key] is not None:
      performance_data.append(metrics[key])
      performance_labels.append(label)
      performance_colors.append(color)

  if performance_data:
    bars = ax6.bar(range(len(performance_data)),
                   performance_data,
                   color=performance_colors,
                   width=0.4,
                   edgecolor='black',
                   linewidth=0.3)
    ax6.set_xticks(range(len(performance_data)))
    ax6.set_xticklabels(performance_labels, rotation=45, ha='right')

    for bar, value in zip(bars, performance_data):
      ax6.text(bar.get_x() + bar.get_width() / 2,
               bar.get_height(),
               f'{value:.2e}',
               ha='center',
               va='bottom',
               fontsize=8)

    ax6.ticklabel_format(axis='y', style='sci', scilimits=(-2, 3))

  ax6.set_ylabel('Value')
  ax6.set_title('(f) Performance Metrics')
  ax6.grid(True, alpha=0.3, axis='y')

  # Plot 7: Configuration and Parameters
  ax7 = fig.add_subplot(gs[2, 2])
  config_info = []

  # Basic configuration
  config_info.append(f"Problem: {problem_type}")
  config_info.append(f"Filter order: {n_coeffs-1}")
  config_info.append(f"Iterations: {n_iters}")
  config_info.append(f"Ensemble size: {n_ensemble}")

  # Step size info
  mu = metrics.get('step_size', 'N/A')
  config_info.append(f"Step size μ: {mu}")

  # Eigenvalue info
  if 'max_eigenvalue' in metrics and 'min_eigenvalue' in metrics:
    spread = metrics['max_eigenvalue'] / metrics['min_eigenvalue']
    config_info.append(f"Eigenvalue spread: {spread:.2f}")

  if 'trace_R' in metrics:
    config_info.append(f"Trace(R): {metrics['trace_R']:.4f}")

  ax7.text(0.5,
           0.7,
           '\n'.join(config_info),
           ha='center',
           va='center',
           transform=ax7.transAxes,
           fontsize=10,
           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcyan"))
  ax7.set_title('(g) Configuration & Parameters')
  ax7.axis('off')

  # Plot 8: Frequency response comparison
  ax8 = fig.add_subplot(gs[2, 0])
  w_final = metrics.get('final_coeff', w_hist_single[-1, :])
  n_fft = 1024

  # Get true system response if available
  unknown_system = metrics.get('unknown_system', None)
  if unknown_system is not None:
    h_true = unknown_system
  elif problem_type == 'time_varying_system_identification' and 'h_true_ensemble' in metrics:
    h_true = metrics['h_true_ensemble'][0, -1]  # First ensemble, last time step
  else:
    h_true = None

  if h_true is not None:
    h_true_1d = np.asarray(jnp.real(h_true)).flatten()
    w_final_1d = np.asarray(jnp.real(w_final)).flatten()

    # Ensure same length for frequency response
    max_len = max(len(h_true_1d), len(w_final_1d))
    h_true_padded = np.zeros(max_len)
    w_final_padded = np.zeros(max_len)
    h_true_padded[:len(h_true_1d)] = h_true_1d
    w_final_padded[:len(w_final_1d)] = w_final_1d

    w_freq, h_true_response = signal.freqz(h_true_padded, worN=n_fft)
    _, h_est_response = signal.freqz(w_final_padded, worN=n_fft)

    ax8.plot(w_freq / np.pi, 20 * np.log10(np.abs(h_true_response + 1e-10)), 'b-', label='True System', linewidth=2)
    ax8.plot(w_freq / np.pi, 20 * np.log10(np.abs(h_est_response + 1e-10)), 'r--', label='Estimated', linewidth=2)
    ax8.set_ylabel('Magnitude (dB)')
    ax8.legend()
  else:
    w_final_1d = np.asarray(jnp.real(w_final)).flatten()
    w_freq, h_response = signal.freqz(w_final_1d, worN=n_fft)
    ax8.plot(w_freq / np.pi, 20 * np.log10(np.abs(h_response + 1e-10)), 'b-', label='Estimated Response')
    ax8.set_ylabel('Magnitude (dB)')
    ax8.legend()

  ax8.set_xlabel('Normalized Frequency (×π rad/sample)')
  ax8.set_title('(h) Frequency Response Comparison')
  ax8.grid(True, alpha=0.3)

  # Plot 9: Error distribution - use last errors from first ensemble member
  ax9 = fig.add_subplot(gs[2, 1])
  n_samples_hist = min(1000, len(e_hist_single))
  last_errors = e_hist_single[-n_samples_hist:]
  real_errors = np.asarray(jnp.real(last_errors))

  if len(real_errors) > 0:
    ax9.hist(real_errors, bins=20, alpha=0.7, label='Real', density=True)
    try:
      mu, std = scipy.stats.norm.fit(real_errors)
      x = np.linspace(ax9.get_xlim()[0], ax9.get_xlim()[1], 100)
      ax9.plot(x, scipy.stats.norm.pdf(x, mu, std), 'k-', linewidth=2, label=f'Normal fit\nμ={mu:.3f}, σ={std:.3f}')
    except:
      pass  # Skip normal fit if it fails

    ax9.set_ylabel('Probability Density')
    ax9.set_xlabel('Error Value')
    ax9.legend()
  else:
    ax9.text(0.5, 0.5, 'No error data', ha='center', va='center', transform=ax9.transAxes)

  ax9.set_title('(i) Error Distribution')
  ax9.grid(True, alpha=0.3)

  # Plot 10: Multi-scale learning curves
  ax10 = fig.add_subplot(gs[3, 0])
  for window in [ 10, 50, 100 ]:
    if len(mse_curve_np) > window:
      mse_smooth = np.convolve(mse_curve_np, np.ones(window) / window, mode='valid')
      x_axis = np.arange(len(mse_smooth)) + window // 2
      ax10.semilogy(x_axis, mse_smooth, label=f'Window={window}', alpha=0.8)

  ax10.set_ylabel('Smoothed MSE')
  ax10.set_xlabel('Iteration')
  ax10.set_title('(j) Multi-Scale Learning Curves')
  ax10.legend()
  ax10.grid(True, alpha=0.3)

  # Summary statistics (always last row)
  ax_summary = fig.add_subplot(gs[4, :])
  summary_text = _generate_summary_text(metrics, n_iters, problem_type, n_ensemble)

  ax_summary.text(0.02,
                  0.5,
                  '\n'.join(summary_text),
                  va='center',
                  ha='left',
                  fontsize=9,
                  family='monospace',
                  bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"))
  ax_summary.set_title('(k) Summary Statistics')
  ax_summary.axis('off')

  plt.tight_layout()
  return fig


def _calculate_mse_surface(w0_hist, w1_hist, metrics):
  """Calculate MSE surface using your actual metric names"""
  w0_min, w0_max = jnp.min(w0_hist), jnp.max(w0_hist)
  w1_min, w1_max = jnp.min(w1_hist), jnp.max(w1_hist)
  padding = 0.2
  w0_range = jnp.linspace(w0_min - padding, w0_max + padding, 50)
  w1_range = jnp.linspace(w1_min - padding, w1_max + padding, 50)
  W0, W1 = jnp.meshgrid(w0_range, w1_range)

  # Use your actual metric names: metrics['R'] and metrics['p']
  has_wiener_info = ('wiener_solution' in metrics and 'R' in metrics and 'p' in metrics)

  if has_wiener_info:
    w_wiener = metrics['wiener_solution']
    w_wiener_2d = jnp.real(w_wiener[:2])
    R = metrics['R'][:2, :2]  # 2x2 submatrix
    p = metrics['p'][:2]  # First 2 elements
    # Estimate signal power from available data
    sigma_d2 = 1.0  # Default, you might want to calculate this from your signals

    W0_flat, W1_flat = W0.flatten(), W1.flatten()
    W_stack = jnp.column_stack([ W0_flat, W1_flat ])

    mse_values = []
    for i in range(len(W_stack)):
      w_vec = W_stack[i]
      mse = sigma_d2 - 2 * jnp.real(jnp.vdot(w_vec, p)) + jnp.real(jnp.vdot(w_vec, R @ w_vec))
      mse_values.append(mse)

    MSE_surface = jnp.array(mse_values).reshape(W0.shape)
  else:
    # Fallback quadratic surface - ensure positive values
    MSE_surface = W0**2 + W1**2 + 1e-10  # Add small constant to avoid zeros

  return W0, W1, MSE_surface


def _calculate_path_mse(w0_hist_np, w1_hist_np, metrics):
  """Calculate MSE along the convergence path using your actual metric names"""
  path_mse = []
  has_wiener_info = ('wiener_solution' in metrics and 'R' in metrics and 'p' in metrics)

  if has_wiener_info:
    R = metrics['R'][:2, :2]
    p = metrics['p'][:2]
    sigma_d2 = 1.0  # Default signal power

    for i in range(len(w0_hist_np)):
      w_vec = jnp.array([w0_hist_np[i], w1_hist_np[i]])
      mse = sigma_d2 - 2 * jnp.real(jnp.vdot(w_vec, p)) + jnp.real(jnp.vdot(w_vec, R @ w_vec))
      path_mse.append(float(mse))
  else:
    # Fallback - ensure positive values
    for i in range(len(w0_hist_np)):
      path_mse.append(w0_hist_np[i]**2 + w1_hist_np[i]**2 + 1e-10)

  return path_mse


def plot_convergence_path_3d(w_hist: _Array, metrics: Dict[str, Any], ax=None, ensemble_idx: int = 0):
  """
    Plot 3D convergence path on MSE surface as a separate function
    
    Args:
        w_hist: Coefficient history array with shape (n_ensemble, n_samples+1, n_coef) or (n_samples+1, n_coef)
        metrics: Performance metrics dictionary
        ax: Matplotlib 3D axis to plot on (if None, creates new figure)
        ensemble_idx: Which ensemble member to plot (default: 0 for first member, -1 for ensemble average)
    
    Returns:
        matplotlib.Figure or None
    """
  # Handle ensemble data
  if w_hist.ndim == 3:  # Ensemble data: (n_ensemble, n_samples+1, n_coef)
    n_ensemble = w_hist.shape[0]

    if ensemble_idx == -1:
      # Use ensemble average
      w_hist_single = jnp.mean(w_hist, axis=0)  # Average across ensemble
      ensemble_label = "Ensemble Average"
    else:
      # Use specified ensemble member
      w_hist_single = w_hist[ensemble_idx]
      ensemble_label = f"Ensemble {ensemble_idx+1}/{n_ensemble}"

  else:  # Single realization: (n_samples+1, n_coef)
    w_hist_single = w_hist
    n_ensemble = 1
    ensemble_label = "Single Realization"

  n_coeffs = w_hist_single.shape[1]

  if n_coeffs < 2:
    if ax is None:
      fig = plt.figure(figsize=(8, 6))
      ax = fig.add_subplot(111, projection='3d')
    ax.text(0.5, 0.5, 0, 'Need ≥2 coefficients\nfor 3D surface', ha='center', va='center', transform=ax.transAxes)
    ax.set_title(f'3D MSE Surface ({ensemble_label})')
    ax.set_xlabel('Re(w₀)')
    ax.set_ylabel('Re(w₁)')
    ax.set_zlabel('log10(MSE)')
    return fig if ax is None else None

  # Extract first two coefficients
  w0_hist = jnp.real(w_hist_single[:, 0])
  w1_hist = jnp.real(w_hist_single[:, 1])

  # Calculate MSE surface
  W0, W1, MSE_surface = _calculate_mse_surface(w0_hist, w1_hist, metrics)
  W0_np, W1_np, MSE_surface_np = np.array(W0), np.array(W1), np.array(MSE_surface)
  w0_hist_np, w1_hist_np = np.array(w0_hist), np.array(w1_hist)

  # Create figure if no axis provided
  if ax is None:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    return_fig = True
  else:
    fig = None
    return_fig = False

  # Handle invalid values for log10
  mse_surface_safe = np.where(MSE_surface_np <= 0, 1e-10, MSE_surface_np)
  surface_z = np.log10(mse_surface_safe)

  # Check for any remaining invalid values
  if np.any(~np.isfinite(surface_z)):
    surface_z = np.where(~np.isfinite(surface_z), np.nanmin(surface_z[np.isfinite(surface_z)]), surface_z)

  ax.plot_surface(W0_np, W1_np, surface_z, cmap='viridis', alpha=0.7, linewidth=0, antialiased=True)

  # Add convergence path to 3D plot
  path_mse = _calculate_path_mse(w0_hist_np, w1_hist_np, metrics)

  # Handle invalid values for path MSE
  path_mse_safe = np.where(np.array(path_mse) <= 0, 1e-10, np.array(path_mse))
  path_mse_np_log = np.log10(path_mse_safe)

  # Check for any remaining invalid values
  if np.any(~np.isfinite(path_mse_np_log)):
    path_mse_np_log = np.where(~np.isfinite(path_mse_np_log), np.nanmin(path_mse_np_log[np.isfinite(path_mse_np_log)]),
                               path_mse_np_log)

  # Plot convergence path
  ax.plot(w0_hist_np, w1_hist_np, path_mse_np_log, 'r-', linewidth=2, alpha=0.8, label='Convergence Path')
  ax.scatter(w0_hist_np[0], w1_hist_np[0], path_mse_np_log[0], color='green', s=50, edgecolor='black', label='Start')
  ax.scatter(w0_hist_np[-1], w1_hist_np[-1], path_mse_np_log[-1], color='blue', s=50, edgecolor='black', label='End')

  # Mark Wiener solution if available
  if 'wiener_solution' in metrics:
    w_wiener = metrics['wiener_solution']
    if len(w_wiener) >= 2:
      w0_opt = jnp.real(w_wiener[0])
      w1_opt = jnp.real(w_wiener[1])

      # Calculate MSE at Wiener solution
      w_opt_mse = _calculate_mse_at_point(w0_opt, w1_opt, metrics)
      w_opt_mse_safe = max(w_opt_mse, 1e-10)
      w_opt_mse_log = np.log10(w_opt_mse_safe)

      ax.scatter(w0_opt,
                 w1_opt,
                 w_opt_mse_log,
                 color='red',
                 s=100,
                 marker='*',
                 edgecolor='black',
                 label='Wiener Solution')

  ax.set_xlabel('Re(w₀)')
  ax.set_ylabel('Re(w₁)')
  ax.set_zlabel('log10(MSE)')
  ax.set_title(f'3D MSE Surface with Convergence Path\n({ensemble_label})')
  ax.legend()

  return fig if return_fig else None


def _calculate_mse_at_point(w0: float, w1: float, metrics: Dict[str, Any]) -> float:
  """Calculate MSE at a specific point in coefficient space"""
  if 'wiener_solution' in metrics and 'R' in metrics and 'p' in metrics:
    w_wiener_sol = metrics['wiener_solution']
    R = metrics['R'][:2, :2]  # 2x2 submatrix
    p = metrics['p'][:2]  # First 2 elements

    # MSE = σ_d² - 2Re(w^H p) + w^H R w
    if 'min_mse' in metrics:
      sigma_d2 = metrics['min_mse']
    else:
      sigma_d2 = 1.0  # Fallback

    w_vec = jnp.array([ w0, w1 ])
    mse = sigma_d2 - 2 * jnp.real(jnp.vdot(w_vec, p)) + jnp.real(jnp.vdot(w_vec, R @ w_vec))
    return float(mse)
  else:
    # Fallback: quadratic surface
    return w0**2 + w1**2


def analyze_convergence_path(w_hist: _Array, metrics: Dict[str, Any], ensemble_idx: int = -1):
  """
  Analyzes the convergence path on the MSE surface (for the first two coefficients)
  and returns the matplotlib Figure object while printing convergence statistics.
  
  Args:
      w_hist: Coefficient history with shape (n_ensemble, n_samples+1, n_coef) or (n_samples+1, n_coef)
      metrics: Performance metrics dictionary
      ensemble_idx: Which ensemble member to plot 
                   (default: 0 for first member, -1 for ensemble average)
  """
  # Handle ensemble data
  if w_hist.ndim == 3:  # Ensemble data: (n_ensemble, n_samples+1, n_coef)
    n_ensemble = w_hist.shape[0]

    if ensemble_idx == -1:
      # Use ensemble average
      w_hist_single = jnp.mean(w_hist, axis=0)  # Average across ensemble
      ensemble_label = "Ensemble Average"
    else:
      # Use specified ensemble member
      w_hist_single = w_hist[ensemble_idx]
      ensemble_label = f"Ensemble {ensemble_idx+1}/{n_ensemble}"

  else:  # Single realization: (n_samples+1, n_coef)
    w_hist_single = w_hist
    n_ensemble = 1
    ensemble_label = "Single Realization"

  if w_hist_single.shape[1] < 2:
    print("Need at least 2 coefficients for convergence path analysis")
    return None

  # Extract first two coefficients (real parts for 2D visualization)
  w0_hist = jnp.real(w_hist_single[:, 0])
  w1_hist = jnp.real(w_hist_single[:, 1])

  # Create appropriate ranges for the MSE surface
  w0_min, w0_max = jnp.min(w0_hist), jnp.max(w0_hist)
  w1_min, w1_max = jnp.min(w1_hist), jnp.max(w1_hist)

  # Add some padding around the convergence path
  padding = 0.2
  w0_range = jnp.linspace(w0_min - padding, w0_max + padding, 100)
  w1_range = jnp.linspace(w1_min - padding, w1_max + padding, 100)
  W0, W1 = jnp.meshgrid(w0_range, w1_range)

  # Calculate actual MSE surface using Wiener solution
  is_wiener_info_available = ('wiener_solution' in metrics and 'R' in metrics and 'p' in metrics)

  if is_wiener_info_available:
    w_wiener_sol = metrics['wiener_solution']
    w_wiener = jnp.real(w_wiener_sol[:2])  # First 2 coefficients
    R = metrics['R'][:2, :2]  # 2x2 submatrix
    p = metrics['p'][:2]  # First 2 elements

    # MSE = σ_d² - 2Re(w^H p) + w^H R w
    # Estimate σ_d² from min_mse or use theoretical value
    if 'min_mse' in metrics:
      sigma_d2 = metrics['min_mse']
    else:
      sigma_d2 = 1.0  # Fallback

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
    if 'wiener_solution' in metrics:
      print("Warning: Missing 'R' or 'p' for full Wiener MSE calculation. Using quadratic fallback.")

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
  min_mse = np.maximum(0.0, np.min(MSE_surface_np))
  max_mse = np.maximum(0.0, np.max(MSE_surface_np))
  levels = np.logspace(np.log10(min_mse + 1e-10), np.log10(max_mse), 15)

  contour = plt.contour(W0_np, W1_np, MSE_surface_np, levels=levels, alpha=0.6, colors='gray', linewidths=1)
  plt.clabel(contour, inline=True, fontsize=8, fmt='%.2f')

  # Plot convergence path with color indicating iteration
  scatter = plt.scatter(w0_hist_np, w1_hist_np, c=np.arange(len(w0_hist_np)), cmap='viridis', s=20, alpha=0.6)
  plt.colorbar(scatter, label='Iteration')

  # Mark important points
  plt.plot(w0_hist_np[0], w1_hist_np[0], 'go', markersize=10, markeredgecolor='black', label='Start')
  plt.plot(w0_hist_np[-1], w1_hist_np[-1], 'bo', markersize=10, markeredgecolor='black', label='End')

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
  plt.title(f'Convergence Path on MSE Surface ({ensemble_label})')
  plt.legend()
  plt.grid(True, alpha=0.3)

  # Plot 2: Individual coefficient evolution
  plt.subplot(2, 2, 2)
  iterations = np.arange(len(w0_hist_np))
  plt.plot(iterations, w0_hist_np, 'b-', label='w[0]', alpha=0.7, linewidth=2)
  plt.plot(iterations, w1_hist_np, 'r-', label='w[1]', alpha=0.7, linewidth=2)

  # Mark Wiener solutions
  if 'wiener_solution' in metrics:
    w_wiener = metrics['wiener_solution']
    plt.axhline(y=jnp.real(w_wiener[0]), color='b', linestyle='--', alpha=0.5, label='w_opt[0]')
    plt.axhline(y=jnp.real(w_wiener[1]), color='r', linestyle='--', alpha=0.5, label='w_opt[1]')

  # Mark true system if available
  if 'unknown_system' in metrics:
    h_true = metrics['unknown_system']
    if len(h_true) >= 2:
      plt.axhline(y=jnp.real(h_true[0]), color='b', linestyle=':', alpha=0.5, label='h_true[0]')
      plt.axhline(y=jnp.real(h_true[1]), color='r', linestyle=':', alpha=0.5, label='h_true[1]')

  plt.xlabel('Iteration')
  plt.ylabel('Coefficient Value')
  plt.title('Coefficient Evolution')
  plt.legend()
  plt.grid(True, alpha=0.3)

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
    step_size = jnp.sqrt((w0_hist_np[i] - w0_hist_np[i - 1])**2 + (w1_hist_np[i] - w1_hist_np[i - 1])**2)
    steps.append(step_size)

  plt.semilogy(steps, 'g-', linewidth=2)
  plt.xlabel('Iteration')
  plt.ylabel('Step Size')
  plt.title('Adaptation Step Size Over Time')
  plt.grid(True, alpha=0.3)

  plt.tight_layout()

  # Print convergence statistics
  print(f"\n=== CONVERGENCE ANALYSIS ({ensemble_label}) ===")
  print(f"Total iterations: {len(w0_hist_np)}")
  print(f"Initial coefficients: w0={w0_hist_np[0]:.4f}, w1={w1_hist_np[0]:.4f}")
  print(f"Final coefficients: w0={w0_hist_np[-1]:.4f}, w1={w1_hist_np[-1]:.4f}")

  if 'wiener_solution' in metrics:
    w_wiener_stats = metrics['wiener_solution']
    print(f"Wiener solution: w0={jnp.real(w_wiener_stats[0]):.4f}, w1={jnp.real(w_wiener_stats[1]):.4f}")
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
