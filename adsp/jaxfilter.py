import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
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

    def step_fn(current_coef, k):
      # Regressor slice
      regressor = jax.lax.dynamic_slice(x_padded, (k,), (self.n_coef,))[::-1]

      # Output
      y_k = jnp.vdot(current_coef, regressor)
      e_k = d[k] - y_k

      # Update coefficients if train
      w_new = jax.lax.cond(
          train,
          lambda _: current_coef + self.mu * jnp.conj(e_k) * regressor,
          lambda _: current_coef,
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

  def generate_signals(self, n_samples: int) -> Tuple:
    """Generate signals for the specific problem"""
    self.key, subkey = jax.random.split(self.key)
    return self.gen_signals_fn(n_samples, subkey)

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
    order = self.lms.filter_order + 1  # Number of coefficients
    N = len(x)

    if N < order:
      raise ValueError(f"Need at least {order} samples for Wiener solution")

    # Estimate autocorrelation matrix R
    R = jnp.zeros((order, order), dtype=x.dtype)
    for i in range(order):
      for j in range(i, order):  # Only compute upper triangle
        lag = abs(i - j)
        corr = jnp.mean(jnp.conj(x[lag:]) * x[:N - lag])
        R = R.at[i, j].set(corr)
        if i != j:
          R = R.at[j, i].set(jnp.conj(corr))  # Make Hermitian

    # Estimate cross-correlation vector p
    p = jnp.zeros(order, dtype=x.dtype)
    for i in range(order):
      corr = jnp.mean(jnp.conj(d[i:]) * x[:N - i])
      p = p.at[i].set(corr)

    # Add regularization and solve
    R_reg = R + 1e-8 * jnp.eye(order, dtype=x.dtype)
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


# --------------------------------------------------
# Channel Equalization Problem
# --------------------------------------------------
class ChannelEqualization(AdaptiveProblem):

  def __init__(self, lms: LMS, gen_signals_fn, wiener_sol_fn=None, seed: int = 42):
    super().__init__(lms, gen_signals_fn, wiener_sol_fn, seed)

  def gen_signals(self, n_samples: int):
    self.key, subkey = jax.random.split(self.key)
    return self.gen_d_x_fn(n_samples, subkey)

  def run(self, n_samples: int, training: bool = True):
    x, d = self.gen_signals(n_samples)

    # Use apply with the initialized params
    y_hist, e_hist, w_hist = self.lms.apply(self.vars, x, d, train=training)

    metrics = self.analyze_performance(w_hist, e_hist, x, d)
    return w_hist, e_hist, metrics

  def wiener_solution(self, x: _Array, d: _Array):
    """Estimate Wiener solution for WSS given signals."""
    if self.wiener_sol_fn is not None:
      w_opt, R, p = self.wiener_sol_fn(x, d, self.lms.filter_order)
    else:
      order = self.lms.filter_order + 1  # Number of coefficients
      N = len(x)

      if N < order:
        raise ValueError(f"Need at least {order} samples for Wiener solution")

      # Estimate autocorrelation matrix R
      R = jnp.zeros((order, order), dtype=x.dtype)
      for i in range(order):
        for j in range(i, order):  # Only compute upper triangle
          lag = abs(i - j)
          corr = jnp.mean(jnp.conj(x[lag:]) * x[:N - lag])
          R = R.at[i, j].set(corr)
          if i != j:
            R = R.at[j, i].set(jnp.conj(corr))  # Make Hermitian

      # Estimate cross-correlation vector p
      p = jnp.zeros(order, dtype=x.dtype)
      for i in range(order):
        corr = jnp.mean(jnp.conj(d[i:]) * x[:N - i])
        p = p.at[i].set(corr)

      # Add regularization and solve
      R_reg = R + 1e-8 * jnp.eye(order, dtype=x.dtype)
      w_opt = jnp.linalg.solve(R_reg, p)

    return w_opt, R, p

  def analyze_performance(self, w_hist, e_hist, x, d):
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
    }


def plot_results(w_hist: _Array,
                 e_hist: _Array,
                 metrics: Dict[str, Any],
                 example_name: str = "lms_simulation"):
  """Plot the LMS simulation results."""

  n_iterations = len(e_hist)
  n_coeffs = w_hist.shape[1]

  # Create figure with subplots
  fig = plt.figure(figsize=(15, 12))
  gs = gridspec.GridSpec(3, 2, figure=fig)

  # Plot 1: Learning curve (MSE)
  ax1 = fig.add_subplot(gs[0, 0])
  mse_curve = jnp.abs(e_hist)**2
  ax1.semilogy(mse_curve, 'b-', alpha=0.7)
  ax1.set_ylabel('Instantaneous Squared Error')
  ax1.set_xlabel('Iteration')
  ax1.set_title('(a) Learning Curve')
  ax1.grid(True, alpha=0.3)

  # Plot 2: Coefficient convergence
  ax2 = fig.add_subplot(gs[0, 1])
  for i in range(min(4, n_coeffs)):  # Plot first 4 coefficients to avoid clutter
    ax2.plot(jnp.real(w_hist[:, i]), label=f'Re(w[{i}])')
    ax2.plot(jnp.imag(w_hist[:, i]), '--', label=f'Im(w[{i}])')

  # Plot Wiener solution
  if 'wiener_solution' in metrics:
    w_wiener = metrics['wiener_solution']
    for i in range(min(3, len(w_wiener))):  # Plot first 3 coefficients
      ax2.axhline(y=jnp.real(w_wiener[i]), color=f'C{i}', linestyle='-', alpha=0.5, linewidth=2)
      ax2.axhline(y=jnp.imag(w_wiener[i]), color=f'C{i}', linestyle='--', alpha=0.5, linewidth=2)

  ax2.set_ylabel('Coefficient Value')
  ax2.set_xlabel('Iteration')
  ax2.set_title('(b) Coefficient Convergence')
  ax2.legend(ncol=2, fontsize=8)
  ax2.grid(True, alpha=0.3)

  # Plot 3: Coefficient error
  ax3 = fig.add_subplot(gs[1, 0])
  if 'coeff_error_curve' in metrics:
    ax3.semilogy(metrics['coeff_error_curve'], 'r-')
    ax3.set_ylabel('Coefficient Error (Norm)')
    ax3.set_xlabel('Iteration')
    ax3.set_title('(c) Coefficient Error vs Wiener Solution')
    ax3.grid(True, alpha=0.3)

  # Plot 4: Error histogram (last 100 samples)
  ax4 = fig.add_subplot(gs[1, 1])
  last_errors = e_hist[-100:]
  ax4.hist(jnp.real(last_errors), bins=20, alpha=0.7, label='Real', density=True)
  ax4.hist(jnp.imag(last_errors), bins=20, alpha=0.7, label='Imag', density=True)
  ax4.set_ylabel('Probability Density')
  ax4.set_xlabel('Error Value')
  ax4.set_title('(d) Error Distribution (last 100 samples)')
  ax4.legend()
  ax4.grid(True, alpha=0.3)

  # Plot 5: Frequency response (final filter)
  ax5 = fig.add_subplot(gs[2, 0])
  w_final = w_hist[-1, :]
  n_fft = 1024
  w_freq, h_response = signal.freqz(np.array(jnp.real(w_final)), worN=n_fft)
  ax5.plot(w_freq / np.pi, 20 * np.log10(np.abs(h_response + 1e-10)))
  ax5.set_ylabel('Magnitude (dB)')
  ax5.set_xlabel('Normalized Frequency (×π rad/sample)')
  ax5.set_title('(e) Final Filter Frequency Response')
  ax5.grid(True, alpha=0.3)

  # Plot 6: Convergence speed
  ax6 = fig.add_subplot(gs[2, 1])
  # Moving average of MSE
  window_size = 50
  mse_smooth = np.convolve(mse_curve, np.ones(window_size) / window_size, mode='valid')
  ax6.semilogy(mse_smooth, 'g-', linewidth=2)
  ax6.set_ylabel('Smoothed MSE')
  ax6.set_xlabel('Iteration')
  ax6.set_title('(f) Smoothed Learning Curve')
  ax6.grid(True, alpha=0.3)

  plt.tight_layout()
  plt.savefig(f'{example_name}_results.png', dpi=300, bbox_inches='tight')
  plt.show()

  return fig


def analyze_convergence_path(w_hist: _Array, metrics: Dict[str, Any]):
  """Analyze the convergence path on the MSE surface (for 3D case)."""
  if w_hist.shape[1] < 2:
    print("Need at least 2 coefficients for convergence path analysis")
    return

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
  if 'wiener_solution' in metrics and 'autocorrelation_matrix' in metrics:
    w_wiener = jnp.real(metrics['wiener_solution'][:2])  # First 2 coefficients
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

  # Convert to numpy for plotting
  W0_np = np.array(W0)
  W1_np = np.array(W1)
  MSE_surface_np = np.array(MSE_surface)
  w0_hist_np = np.array(w0_hist)
  w1_hist_np = np.array(w1_hist)

  # Create the plot
  plt.figure(figsize=(12, 10))

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
    w_wiener = metrics['wiener_solution']
    plt.plot(jnp.real(w_wiener[0]),
             jnp.real(w_wiener[1]),
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
  for i in range(len(w0_hist_np)):
    if 'wiener_solution' in metrics and 'autocorrelation_matrix' in metrics:
      w_vec = jnp.array([w0_hist_np[i], w1_hist_np[i]])
      mse = sigma_d2 - 2 * jnp.real(jnp.vdot(w_vec, p)) + jnp.real(jnp.vdot(w_vec, R @ w_vec))
      path_mse.append(float(mse))
    else:
      path_mse.append(w0_hist_np[i]**2 + w1_hist_np[i]**2)

  ax.plot(w0_hist_np,
          w1_hist_np,
          np.log10(np.array(path_mse) + 1e-10),
          'r-',
          linewidth=2,
          alpha=0.8)
  ax.scatter(w0_hist_np[0],
             w1_hist_np[0],
             np.log10(path_mse[0] + 1e-10),
             color='green',
             s=50,
             edgecolor='black')
  ax.scatter(w0_hist_np[-1],
             w1_hist_np[-1],
             np.log10(path_mse[-1] + 1e-10),
             color='blue',
             s=50,
             edgecolor='black')

  ax.set_xlabel('Re(w₀)')
  ax.set_ylabel('Re(w₁)')
  ax.set_zlabel('log10(MSE)')
  ax.set_title('3D MSE Surface with Convergence Path')

  # Plot 3: Distance to Wiener solution over time
  plt.subplot(2, 2, 3)
  if 'wiener_solution' in metrics:
    w_wiener = jnp.real(metrics['wiener_solution'][:2])
    distances = []
    for i in range(len(w0_hist_np)):
      dist = jnp.sqrt((w0_hist_np[i] - w_wiener[0])**2 + (w1_hist_np[i] - w_wiener[1])**2)
      distances.append(dist)

    plt.semilogy(distances, 'r-', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Distance to Wiener Solution')
    plt.title('Convergence to Optimal Solution')
    plt.grid(True, alpha=0.3)

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
  plt.savefig('convergence_path_detailed.png', dpi=300, bbox_inches='tight')
  plt.show()

  # Print convergence statistics
  print("\n=== CONVERGENCE ANALYSIS ===")
  print(f"Total iterations: {len(w0_hist_np)}")
  print(f"Initial coefficients: w0={w0_hist_np[0]:.4f}, w1={w1_hist_np[0]:.4f}")
  print(f"Final coefficients: w0={w0_hist_np[-1]:.4f}, w1={w1_hist_np[-1]:.4f}")

  if 'wiener_solution' in metrics:
    w_wiener = metrics['wiener_solution']
    print(f"Wiener solution: w0={jnp.real(w_wiener[0]):.4f}, w1={jnp.real(w_wiener[1]):.4f}")
    final_distance = jnp.sqrt((w0_hist_np[-1] - jnp.real(w_wiener[0]))**2 +
                              (w1_hist_np[-1] - jnp.real(w_wiener[1]))**2)
    print(f"Final distance to Wiener solution: {final_distance:.6f}")

  print(f"Total path length: {np.sum(steps):.4f}")
  print(f"Average step size: {np.mean(steps):.6f}")


# Alternative simpler version if you prefer the original style but better
def analyze_convergence_path_2d(w_hist: _Array, metrics: Dict[str, Any]):
  """Simplified version with better contour levels."""
  if w_hist.shape[1] < 2:
    print("Need at least 2 coefficients for convergence path analysis")
    return

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
    # Simple quadratic surface centered at Wiener solution
    MSE_surface = (W0 - jnp.real(w_wiener[0]))**2 + (W1 - jnp.real(w_wiener[1]))**2
  else:
    MSE_surface = W0**2 + W1**2

  plt.figure(figsize=(10, 8))

  # Use better contour levels
  levels = np.linspace(np.min(MSE_surface), np.max(MSE_surface), 10)
  contour = plt.contour(W0, W1, MSE_surface, levels=levels, alpha=0.6, colors='gray')
  plt.clabel(contour, inline=True, fontsize=9, fmt='%.1f')

  # Plot convergence path
  plt.plot(w0_hist, w1_hist, 'r-', linewidth=2, alpha=0.7, label='Convergence Path')
  plt.plot(w0_hist[0], w1_hist[0], 'go', markersize=10, label='Start', markeredgecolor='black')
  plt.plot(w0_hist[-1], w1_hist[-1], 'bo', markersize=10, label='End', markeredgecolor='black')

  if 'wiener_solution' in metrics:
    w_wiener = metrics['wiener_solution']
    plt.plot(jnp.real(w_wiener[0]),
             jnp.real(w_wiener[1]),
             'rx',
             markersize=15,
             markeredgewidth=2,
             label='Wiener Solution')

  plt.xlabel('Re(w0)')
  plt.ylabel('Re(w1)')
  plt.title('Convergence Path on MSE Surface')
  plt.legend()
  plt.grid(True, alpha=0.3)
  plt.savefig('convergence_path_simple.png', dpi=300, bbox_inches='tight')
  plt.show()


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
