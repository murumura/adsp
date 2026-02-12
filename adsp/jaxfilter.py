import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
from typing import Dict, Any, Optional, Union
from abc import ABC

_Array = Union[jnp.ndarray]

def dtype2real(dtype):
  return dtype.type(0).real.dtype

def dtype2complex(dtype):
  return (1j * dtype.type(0)).dtype

def realsign(x):
  """Scalar sign applied to the real part, as in Diniz Eq. (4.5)."""
  xr = jnp.real(x)
  return jnp.where(xr > 0, 1.0, jnp.where(xr < 0, -1.0, 0.0))

def csignvec(x, eps: float = 1e-12):
  """Elementwise complex sign for a vector."""
  x = x.astype(jnp.complex64)
  mag = jnp.abs(x)
  zero = jnp.zeros_like(x)
  return jnp.where(mag > eps, x / mag, zero)

def csignscalar(e, eps=1e-12):
  """complex sign scalar estimation"""
  e = e.astype(jnp.complex64)
  mag = jnp.abs(e)
  return jnp.where(mag > eps, e / mag, jnp.array(0+0j, dtype=jnp.complex64))

def ar1theoR(a: float, order: int, sigma_v2: float = 1.0):
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

def ar1poleSpreadSearch(target_spread: float,
                        order: int,
                        sigma_v2: float = 1.0,
                        tol: float = 0.01) -> float:
  """Find AR(1) pole that produces target eigenvalue spread using theoretical formula"""
  a_grid = np.linspace(0.0, 0.999, 1000)
  best_diff = np.inf
  best_a = 0.0

  for a in a_grid:
    R = ar1theoR(a, order, sigma_v2)  # Pass sigma_v2 here
    eigs = jnp.linalg.eigvalsh(R)
    spread = float(jnp.max(eigs) / jnp.min(eigs))
    diff = abs(spread - target_spread)

    if diff < best_diff:
      best_diff = diff
      best_a = a

    if best_diff <= tol:
      break

  return best_a

def dctOrthoMatrix(M: int, dtype=jnp.float32) -> _Array:
  """
  Orthonormal DCT-II matrix T (M x M):
    s = T x
  """
  n = jnp.arange(M, dtype=dtype)[:, None]   # (M,1)
  k = jnp.arange(M, dtype=dtype)[None, :]   # (1,M)
  # DCT-II basis: cos(pi/M * (n+0.5) * k)
  T = jnp.cos(jnp.pi / M * (n + 0.5) * k)   # (M,M)
  # Orthonormal scaling
  T = T.at[:, 0].set(T[:, 0] / jnp.sqrt(M))
  T = T.at[:, 1:].set(T[:, 1:] * jnp.sqrt(2.0 / M))
  return T

def dftUnitaryMatrix(M: int, dtype=jnp.complex64) -> _Array:
  """
  Unitary DFT matrix T (M x M):
    s = T x
  with 1/sqrt(M) normalization.
  """
  n = jnp.arange(M)[:, None]
  k = jnp.arange(M)[None, :]
  W = jnp.exp(-1j * 2.0 * jnp.pi * n * k / M).astype(dtype)
  return W / jnp.sqrt(M)


class BaseLMS(nn.Module):
  """Base class: persistent weight vector w[k] in 'state' collection."""
  mu: float
  filter_order: int
  init_coef: Optional[jnp.ndarray] = None

  def setup(self):
    self.n_coef = self.filter_order + 1
    if self.init_coef is None:
      init_w = jnp.zeros((self.n_coef,), dtype=jnp.complex64)
    else:
      init_w = jnp.array(self.init_coef, dtype=jnp.complex64)
    self.w = self.variable("state", "w", lambda: init_w)

  def prepad(self, x):
    return jnp.concatenate([jnp.zeros(self.n_coef - 1, dtype=x.dtype), x])

class LMS(BaseLMS):
  """
  Complex LMS adaptive filter (Diniz Algorithm 3.2)
  w(k+1) = w(k) + 2 μ e*(k) x(k)
  """
  @nn.compact
  def __call__(self, x, d, train: bool = True):
    N = x.shape[0]
    x_pad = self.prepad(x)
    w0 = self.w.value  # persistent initial weights

    def stepfn(w, k):
      reg = jax.lax.dynamic_slice(x_pad, (k,), (self.n_coef,))[::-1]
      y = jnp.vdot(w, reg)
      e = d[k] - y

      upd = 2.0 * self.mu * jnp.conj(e) * reg
      upd = upd.astype(w.dtype)

      w_new = jax.lax.cond(
        train,
        lambda _: w + upd,
        lambda _: w,
        operand=None,
      )
      return w_new, (y, e, w_new)

    w_final, (y_hist, e_hist, w_hist) = jax.lax.scan(stepfn, w0, jnp.arange(N))

    if train:
      self.w.value = w_final

    w_hist = jnp.vstack([w0, w_hist])
    return y_hist, e_hist, w_hist


  def excessMSE(self, R: _Array, delta_w_cov: _Array) -> float:
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

  def isStableR(self, R: _Array) -> bool:
    """Check stability condition by Equation 3.19 and Equation 3.30"""
    eigenvalues = jnp.linalg.eigvals(R)
    max_eigenvalue = jnp.max(jnp.real(eigenvalues))
    trace_R = jnp.trace(R)
    return 0 < self.mu < 1.0 / max_eigenvalue and 0 < self.mu < 1.0 / trace_R

  def analyze(self, R: _Array, verbose: bool = False) -> Dict[str, Any]:
    """Analyze step size and suggest improvements"""
    eigenvalues = jnp.linalg.eigvals(R)
    lambda_max = jnp.max(jnp.real(eigenvalues))
    lambda_min = jnp.min(jnp.real(eigenvalues))
    trace_R = jnp.trace(R)
  
    # Stability bounds from textbook
    bound_eigenvalue = 1.0 / lambda_max
    bound_trace = 1.0 / trace_R  # More practical bound (Equation 3.30)
    
    # Current stability
    is_stable_R = self.isStableR(R)
    
    # Suggested ranges
    safe_mu = 0.1 * bound_trace  # Conservative choice
    aggressive_mu = 0.5 * bound_trace  # More aggressive but potentially unstable
    
    # Eigenvalue spread analysis
    eigenvalue_spread = lambda_max / lambda_min if lambda_min > 0 else jnp.inf
    
    # Calculate theoretical misadjustment (Equation 3.50)
    theoretical_misadjustment = (self.mu * trace_R) / (1 - self.mu * trace_R)
    
    result = {
      'is_stable_R': is_stable_R,
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
        self.printMuAnalysis(result)
    
    return result

  def printMuAnalysis(self, analysis: Dict[str, Any]):
    """Print detailed analysis of step size configuration"""
    print("\n" + "="*60)
    print("LMS STEP SIZE ANALYSIS")
    print("="*60)
    
    print(f"Current μ: {analysis['current_mu']:.6f}")
    print(f"Stability: {'✓ STABLE' if analysis['is_stable_R'] else '✗ UNSTABLE'}")
    
    print(f"\nEIGENVALUE ANALYSIS:")
    print(f"  Max eigenvalue (λ_max): {analysis['lambda_max']:.6f}")
    print(f"  Min eigenvalue (λ_min): {analysis['lambda_min']:.6f}")
    print(f"  Eigenvalue spread: {analysis['eigenvalue_spread']:.2f}")
    if analysis['convergence_warning']:
        print(f"  WARNING: High eigenvalue spread will slow convergence")
    
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
        print(f"  WARNING: Small stability margin")
    
    print("="*60)


class SignErrorLMS(BaseLMS):
  """
  Complex Sign-Error LMS (Diniz Algorithm 4.1, complex extension)
  w(k+1) = w(k) + 2 μ sgn[e(k)] x(k)
  """

  @nn.compact
  def __call__(self, x, d, train: bool = True):
    N = x.shape[0]
    x_pad = self.prepad(x)
    w0 = self.w.value

    def stepfn(w, k):
      reg = jax.lax.dynamic_slice(x_pad, (k,), (self.n_coef,))[::-1]
      reg = reg.astype(w.dtype)       

      y = jnp.vdot(w, reg)
      e = d[k] - y

      sigma_e = csignscalar(e).astype(w.dtype)
      upd = (2.0 * self.mu) * sigma_e * reg
      upd = upd.astype(w.dtype)               

      w_new = jax.lax.cond(
        train,
        lambda _: w + upd,
        lambda _: w,
        None
      )

      return w_new, (y, e, w_new)

    w_final, (y_hist, e_hist, w_hist) = jax.lax.scan(stepfn, w0, jnp.arange(N))
    if train:
      self.w.value = w_final

    w_hist = jnp.vstack([w0, w_hist])
    return y_hist, e_hist, w_hist
  
class SignDataLMS(BaseLMS):
  """
  Complex Sign-Data LMS (Diniz Algorithm 4.2, complex extension)
  w(k+1) = w(k) + 2 μ e*(k) sgn[x(k)]
  """

  @nn.compact
  def __call__(self, x, d, train=True):
    N = x.shape[0]
    x_pad = self.prepad(x)
    w0 = self.w.value

    def stepfn(w, k):
      reg = jax.lax.dynamic_slice(x_pad, (k,), (self.n_coef,))[::-1]
      reg = reg.astype(jnp.complex64)

      y = jnp.vdot(w, reg)
      e = d[k] - y

      sign_reg = csignvec(reg).astype(w.dtype)
      upd = 2.0 * self.mu * jnp.conj(e) * sign_reg
      upd = upd.astype(w.dtype)

      w_new = jax.lax.cond(train, lambda _: w + upd, lambda _: w, None)
      return w_new, (y, e, w_new)

    w_final, (y_hist, e_hist, w_hist) = jax.lax.scan(stepfn, w0, jnp.arange(N))
    if train:
      self.w.value = w_final

    w_hist = jnp.vstack([w0, w_hist])
    return y_hist, e_hist, w_hist

class SignSignLMS(BaseLMS):
  """
  Complex Sign-Sign LMS (Diniz Sec. 4.2.4 extended)
  w(k+1) = w(k) + 2 μ sgn[e(k)] sgn[x(k)]
  """

  @nn.compact
  def __call__(self, x, d, train: bool = True):
    N = x.shape[0]
    x_pad = self.prepad(x)
    w0 = self.w.value

    def stepfn(w, k):
      reg = jax.lax.dynamic_slice(x_pad, (k,), (self.n_coef,))[::-1]
      y = jnp.vdot(w, reg)
      e = d[k] - y

      sigma_e = csignscalar(e).astype(w.dtype)
      sign_reg = csignvec(reg).astype(w.dtype)
      upd = 2.0 * self.mu * sigma_e * sign_reg
      w_new = jax.lax.cond(
          train,
          lambda _: w + upd,
          lambda _: w,
          operand=None,
      )
      return w_new, (y, e, w_new)

    w_final, (y_hist, e_hist, w_hist) = jax.lax.scan(stepfn, w0, jnp.arange(N))
    if train:
      self.w.value = w_final

    w_hist = jnp.vstack([w0, w_hist])
    return y_hist, e_hist, w_hist

class DualSignLMS(BaseLMS):
  """
  Complex Dual-Sign LMS (Diniz Sec. 4.2.3 style)
  if |e(k)| > rho:
      w(k+1) = w(k) + μ ε sgn[e(k)] x(k)
  else:
      w(k+1) = w(k) + μ sgn[e(k)] x(k)
  """
  epsilon: float = 2.0
  rho: float = 1.0

  @nn.compact
  def __call__(self, x, d, train: bool = True):
    N = x.shape[0]
    x_pad = self.prepad(x)
    w0 = self.w.value

    def stepfn(w, k):
      reg = jax.lax.dynamic_slice(x_pad, (k,), (self.n_coef,))[::-1]
      y = jnp.vdot(w, reg)
      e = d[k] - y

      sigma_e = csignscalar(e).astype(w.dtype)
      gain = jnp.where(jnp.abs(e) > self.rho, self.epsilon, 1.0)
      upd = 2.0 * self.mu * gain * sigma_e * reg
      w_new = jax.lax.cond(
        train,
        lambda _: w + upd,
        lambda _: w,
        operand=None,
      )
      return w_new, (y, e, w_new)

    w_final, (y_hist, e_hist, w_hist) = jax.lax.scan(stepfn, w0, jnp.arange(N))
    if train:
      self.w.value = w_final

    w_hist = jnp.vstack([w0, w_hist])
    return y_hist, e_hist, w_hist

class PowerOfTwoErrorLMS(BaseLMS):
  """
  Power-of-Two Error LMS (Diniz Sec. 4.2.3)
  pe[e] as in Eq. (4.40)
  w(k+1) = w(k) + 2 μ pe[e(k)] x(k)
  """
  bd: int = 8     # data wordlength (magnitude bits, no sign)
  tau: float = 0.0  # usually 0 or 2^-bd

  def p2e(self, e):
    """Power of 2 error"""
    abs_e = jnp.abs(e)
    s = realsign(e)
    thresh = 2.0 ** (-(self.bd - 1))

    region1 = abs_e >= 1.0
    region_mid = (abs_e >= thresh) & (abs_e < 1.0)

    # avoid log2(0)
    mag = abs_e + 1e-12
    pow_term = jnp.exp2(jnp.floor(jnp.log2(mag)))

    return jnp.where(
      region1,
      s,
      jnp.where(region_mid, pow_term * s, self.tau * s)
    )

  @nn.compact
  def __call__(self, x, d, train: bool = True):
    N = x.shape[0]
    x_pad = self.prepad(x)
    w0 = self.w.value

    def stepfn(w, k):
      reg = jax.lax.dynamic_slice(x_pad, (k,), (self.n_coef,))[::-1]
      y = jnp.vdot(w, reg)
      e = d[k] - y

      pe = self.p2e(e).astype(w.dtype)
      upd = 2.0 * self.mu * pe * reg
      upd = upd.astype(w.dtype)

      w_new = jax.lax.cond(
        train,
        lambda _: w + upd,
        lambda _: w,
        operand=None,
      )
      return w_new, (y, e, w_new)

    w_final, (y_hist, e_hist, w_hist) = jax.lax.scan(stepfn, w0, jnp.arange(N))

    if train:
      self.w.value = w_final

    w_hist = jnp.vstack([w0, w_hist])
    return y_hist, e_hist, w_hist

class TransformDomain(BaseLMS):
  """
  Transform-Domain (Power-Normalized) NLMS / TD-LMS (Diniz Sec. 4.5)

  s(k) = T x_reg(k)
  sigma2_i(k) = alpha |s_i(k)|^2 + (1-alpha) sigma2_i(k-1)
  w_hat(k+1) = w_hat(k) + 2*mu * e*(k) * ( s(k) / (tau + sigma2(k)) )

  Notes:
  - self.w stores transform-domain coefficients w_hat.
  - For DCT: T is real orthonormal; for DFT: T is unitary complex.
  """

  matrix: str = "dct" # "dct" or "dft"
  alpha: float = 0.01 # power smooth factor
  gamma: float = 1e-8 # safeguard
  init_power: float = 1.0     

  def setup(self):
    super().setup()
    M = self.n_coef

    mat = self.matrix.lower()
    if mat == "dct":
      T = dctOrthoMatrix(M, dtype=jnp.float32).astype(jnp.complex64)
    elif mat == "dft":
      T = dftUnitaryMatrix(M, dtype=jnp.complex64)
    else:
      raise ValueError(f"Unknown transform '{self.matrix}', use 'dct' or 'dft'.")

    self.T = T
    self.TH = jnp.conj(T).T

    # power_vector state (real)
    init_p = (self.init_power * jnp.ones((M,), dtype=jnp.float32))
    self.power = self.variable("state", "power", lambda: init_p)

  @nn.compact
  def __call__(self, x, d, train: bool = True):
    N = x.shape[0]
    M = self.n_coef

    x_pad = self.prepad(x)
    w0_hat = self.w.value                     # w_hat(0) stored in self.w
    p0 = self.power.value                     # power_vector(0)

    alpha = jnp.asarray(self.alpha, jnp.float32)
    gamma = jnp.asarray(self.gamma, jnp.float32)

    def stepfn(carry, k):
      w_hat, p = carry

      # tapped regressor (time-domain)
      reg = jax.lax.dynamic_slice(x_pad, (k,), (M,))[::-1].astype(jnp.complex64)

      # transform regressor: s(k)
      s = (self.T @ reg).astype(jnp.complex64)

      # power update (EMA)
      p_new = alpha * (jnp.abs(s) ** 2).astype(jnp.float32) + (1.0 - alpha) * p

      # output/error
      y = jnp.vdot(w_hat, s)
      e = d[k].astype(jnp.complex64) - y

      # TD normalized update 
      denom = gamma + p_new
      upd = 2.0 * self.mu * (jnp.conj(e) * s / denom.astype(s.dtype))  # self.mu == step
      upd = upd.astype(w_hat.dtype)

      w_new = jax.lax.cond(train, lambda _: w_hat + upd, lambda _: w_hat, None)

      # convert to time domain for history/plot
      w_tnew = self.TH @ w_new

      return (w_new, p_new), (y, e, w_tnew)

    (w_final, p_final), (y_hist, e_hist, w_time_hist) = jax.lax.scan(
      stepfn, (w0_hat, p0), jnp.arange(N)
    )

    if train:
      self.w.value = w_final
      self.power.value = p_final

    # prepend initial time-domain weight too
    w_time0 = self.TH @ w0_hat
    w_hist = jnp.vstack([w_time0, w_time_hist])
    return y_hist, e_hist, w_hist

class NLMS(BaseLMS):
  """
  Complex Normalized LMS (Diniz Sec. 4.3)

  e(k) = d(k) - w^H(k) x(k)
  μ(k) = μ_n / (τ + ||x(k)||^2)
  w(k+1) = w(k) + μ(k) e*(k) x(k)
  For NLMS mu means mu_n and should be bound for 0 < mu_n < 2
  for stability
  """

  tau: float = 1e-3

  @nn.compact
  def __call__(self, x, d, train=True):
    N = x.shape[0]
    x_pad = self.prepad(x)
    w0 = self.w.value

    def stepfn(w, k):
      reg = jax.lax.dynamic_slice(x_pad, (k,), (self.n_coef,))[::-1]
      reg = reg.astype(w.dtype)

      y = jnp.vdot(w, reg)
      e = d[k].astype(w.dtype) - y

      # instantaneous power (force float32)
      norm = jnp.real(jnp.vdot(reg, reg)).astype(jnp.float32)

      # normalized step (force float32)
      mu_k = (self.mu / (self.tau + norm)).astype(jnp.float32)

      # update (force same dtype as w)
      upd = (mu_k * jnp.conj(e) * reg).astype(w.dtype)

      w_new = jax.lax.cond(
        train,
        lambda _: w + upd,
        lambda _: w,
        operand=None
      )

      return w_new, (y, e, w_new)

    w_final, (y_hist, e_hist, w_hist) = jax.lax.scan(
      stepfn, w0, jnp.arange(N)
    )

    if train:
      self.w.value = w_final

    w_hist = jnp.vstack([w0, w_hist])
    return y_hist, e_hist, w_hist


class AffineProjection(BaseLMS):
  """
  Complex Affine Projection Algorithm (APA)
  Diniz Sec. 4.6
  """

  P: int = 3
  delta: float = 1e-3

  @nn.compact
  def __call__(self, x, d, train: bool = True):
    N = x.shape[0]
    x_pad = self.prepad(x)
    w0 = self.w.value
    M = self.n_coef
    Pp = self.P + 1

    def stepfn(w, k):
      # Build X_ap(k): shape (M, P+1)
      Xap = jnp.stack([
        jax.lax.dynamic_slice(x_pad, (k - i,), (M,))[::-1] for i in range(Pp)
      ], axis=1).astype(jnp.complex64)

      d_ap = jnp.stack([d[k - i] for i in range(Pp)])

      # a-priori output & error
      y_ap = Xap.conj().T @ w          # (P+1,)
      e_ap = d_ap - y_ap               # (P+1,)

      # APA update
      R = Xap.conj().T @ Xap
      R_reg = R + self.delta * jnp.eye(Pp, dtype=R.dtype)
      g = jnp.linalg.solve(R_reg, e_ap)
      upd = (self.mu * (Xap @ g)).astype(w.dtype)

      w_new = jax.lax.cond(
        train,
        lambda _: w + upd,
        lambda _: w,
        operand=None
      )

      # to align with lms only take current k as output
      y0 = y_ap[0]
      e0 = e_ap[0]

      return w_new, (y0, e0, w_new)

    w_final, (y_hist, e_hist, w_hist) = jax.lax.scan(
      stepfn, w0, jnp.arange(self.P, N)
    )

    if train:
      self.w.value = w_final

    # prepend initial weight to align shapes
    w_hist = jnp.vstack([w0, w_hist])

    return y_hist, e_hist, w_hist

