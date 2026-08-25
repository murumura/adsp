import numpy as np
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union, Literal, TypeAlias
from . import adspquant

Array: TypeAlias = np.ndarray

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def dtype2real(dtype: np.dtype) -> np.dtype:
  return np.array(0, dtype=dtype).real.dtype


def dtype2complex(dtype: np.dtype) -> np.dtype:
  return np.array(1j, dtype=dtype).dtype


def realsign(x: Array) -> Array:
  """Scalar sign applied to the real part, as in Diniz Eq. (4.5)."""
  xr = np.real(x)
  out = np.zeros_like(xr, dtype=np.float32)
  out[xr > 0] = 1.0
  out[xr < 0] = -1.0
  return out


def csignvec(x: Array, eps: float = 1e-12) -> Array:
  """Elementwise complex sign for a vector."""
  x = np.asarray(x, dtype=np.complex64)
  mag = np.abs(x)
  out = np.zeros_like(x)
  nz = mag > eps
  out[nz] = x[nz] / mag[nz]
  return out

def olsFFTConv(x, h, N, debug=False):
  x = np.asarray(x, dtype=float)
  h = np.asarray(h, dtype=float)

  L = len(x)
  M = len(h)

  if N < M:
    raise ValueError("FFT size must be >= filter length")

  P = N - (M - 1)

  if debug:
    print(f"[OLS] L={L} M={M} N={N} P={P}")

  # pad input with M-1 zeros at beginning
  xpad = np.concatenate((np.zeros(M-1), x))

  # FFT of filter
  hpad = np.zeros(N)
  hpad[:M] = h
  H = np.fft.fft(hpad)
  print(H)
  out = []

  for start in range(0, len(x), P):
    block = xpad[start:start + N]

    if len(block) < N:
      block = np.pad(block, (0, N - len(block)))

    X = np.fft.fft(block)
    Y = X * H
    y = np.fft.ifft(Y).real

    # discard corrupted samples
    out.extend(y[M-1:])

  return np.array(out[:L + M - 1])

def csignscalar(e: Union[complex, Array], eps: float = 1e-12) -> np.complex64:
  """Complex sign scalar estimation."""
  e = np.asarray(e, dtype=np.complex64)
  mag = np.abs(e)
  if np.ndim(mag) != 0:
    raise ValueError("csignscalar expects scalar")
  return np.conj(e) / (mag + eps)

def toeplitzFromFirstRow(first_row: Array) -> Array:
  """
  Toeplitz with first row = [r0, r1, ..., r_{M-1}] 
  and Hermitian-symmetric for real AR(1).
  """
  first_row = np.asarray(first_row)
  M = first_row.size
  T = np.empty((M, M), dtype=first_row.dtype)
  for i in range(M):
    for j in range(M):
      k = abs(i - j)
      T[i, j] = first_row[k]
  return T


def ar1theoR(a: float, order: int, sigma_v2: float = 1.0) -> Array:
  """
  Compute theoretical autocorrelation matrix for AR(1) process.
  Assume the input x(n) is WSS, this is an implmentation of eq 2.83
  From equation (2.83): R = sigma_v²/(1-a²) * Toeplitz([1, a, a², ..., a^(order-1)])
  
  This matches the definition R = E[x(k)x^T(k)] where x(k) = [x(k), x(k-1), ..., x(k-order+1)]^T
  """
  first_row = np.array([a**k for k in range(order)], dtype=np.float64)
  scale = sigma_v2 / (1.0 - a**2)
  R = scale * toeplitzFromFirstRow(first_row)
  return R


def ar1poleSpreadSearch(
  target_spread: float,
  order: int,
  sigma_v2: float = 1.0,
  tol: float = 0.01,
  ngrid: int = 1000,
) -> float:
  """Find AR(1) pole a that produces target eigenvalue spread (max/min) using theoretical formula."""
  a_grid = np.linspace(0.0, 0.999, ngrid)
  best_diff = np.inf
  best_a = 0.0

  for a in a_grid:
    R = ar1theoR(float(a), order, sigma_v2)
    eigs = np.linalg.eigvalsh(R)
    spread = float(np.max(eigs) / np.min(eigs))
    diff = abs(spread - target_spread)

    if diff < best_diff:
      best_diff = diff
      best_a = float(a)

    if best_diff <= tol:
      break

  return best_a

def arCorrMatrix(a, u_var, order=1):
  """
  Correlation matrix R for AR(1):
    x(n) = -a x(n-1) + u(n)
  u_var = var{u(n)}
  order = filter length M
  """

  if abs(a) >= 1:
    raise ValueError("AR(1) is not stationary unless |a| < 1")

  r0 = u_var / (1 - a**2)

  # autocorrelation sequence r(0)...r(M-1)
  r = np.array([r0 * (-a)**k for k in range(order)])

  # Toeplitz matrix
  R = np.empty((order, order))
  for i in range(order):
    for j in range(order):
      R[i, j] = r[abs(i - j)]

  return R

def dctOrthoMatrix(M: int, dtype=np.float32) -> Array:
  """
  Orthonormal DCT-II matrix T (M x M):
    s = T x
  """
  n = np.arange(M, dtype=dtype)[:, None]     # (M,1)
  k = np.arange(M, dtype=dtype)[None, :]     # (1,M)
  T = np.cos(np.pi / M * (n + 0.5) * k).astype(dtype)  # (M,M)

  T[:, 0] *= (1.0 / np.sqrt(M))
  if M > 1:
    T[:, 1:] *= np.sqrt(2.0 / M)
  return T


def dftUnitaryMatrix(M: int, dtype=np.complex64) -> Array:
  """
  Unitary DFT matrix T (M x M):
    s = T x
  with 1/sqrt(M) normalization.
  """
  n = np.arange(M)[:, None]
  k = np.arange(M)[None, :]
  W = np.exp(-1j * 2.0 * np.pi * n * k / M).astype(dtype)
  return W / np.sqrt(M)


def _prepad(x: Array, n_coef: int) -> Array:
  x_flat = np.asarray(x).ravel()
  return np.concatenate([np.zeros(n_coef - 1, dtype=x_flat.dtype), x_flat])

def eqfirDesired(s, n_samples, delay, dtype=np.complex64):
  """
  Desired signal for causal FIR equalizer.

  Equalizer regressor:
      reg[k] = [r[k], r[k-1], ..., r[k-M+1]]

  Therefore:
      y[k] ≈ s[k-delay]

  So:
      d[k] = s[k-delay]

  This means the first `delay` desired samples are invalid and set to zero.
  Ignore them during final constellation / MSE judgment.
  """

  if delay < 0:
    raise ValueError("delay must be >= 0")

  s = np.asarray(s, dtype=dtype)
  d = np.zeros(n_samples, dtype=dtype)

  if delay >= n_samples:
    return d

  n_valid = min(n_samples - delay, len(s))
  d[delay:delay + n_valid] = s[:n_valid]

  return d

def eqResultAlign(r, d, y, e):
  """
  Align r, d, y, e for plotting.

  NLMS:
      len(y) == len(r), so start = 0

  APA:
      your class outputs from k = P, so len(y) == len(r) - P,
      therefore start = len(r) - len(y)
  """

  r = np.asarray(r)
  d = np.asarray(d)
  y = np.asarray(y)
  e = np.asarray(e)

  start = len(r) - len(y)
  stop = start + len(y)

  if start < 0:
    raise ValueError("len(y) cannot be longer than len(r)")

  r_aligned = r[start:stop]
  d_aligned = d[start:stop]

  if len(e) != len(y):
    raise ValueError("e and y should have same length")

  return r_aligned, d_aligned, y, e, start

def makeEqW0(h, n_taps, delay):
  """
  Build a reference equalizer w0 for comparison.

  We solve for physical FIR equalizer q0 such that:
      h * q0 ≈ delayed impulse

  Since your adaptive filters store coefficients w such that:
      y = vdot(w, reg) = conj(w)^T reg

  the physical FIR taps are:
      q = conj(w)

  Therefore:
      w0 = conj(q0)
  """
  h = np.asarray(h, dtype=np.complex64)

  Lh = len(h)
  M = n_taps

  # Convolution matrix A so that:
  #   A @ q0 = h * q0
  A = np.zeros((Lh + M - 1, M), dtype=np.complex64)
  for i in range(M):
    A[i:i + Lh, i] = h

  # Desired delayed impulse
  desired = np.zeros(Lh + M - 1, dtype=np.complex64)
  desired[delay] = 1.0 + 0j

  # Least-squares solution for physical equalizer q0
  q0, _, _, _ = np.linalg.lstsq(A, desired, rcond=None)

  # Convert to your adaptive-filter stored coefficient convention
  w0 = np.conj(q0)

  return w0, q0, desired, A

# -----------------------------------------------------------------------------
# Base
# -----------------------------------------------------------------------------
@dataclass
class BaseLMS:
  filter_order: int
  mu: Optional[float] = None  # Default to None for non-LMS types
  init_coef: Optional[Array] = None
  w: Array = field(init=False)

  def __post_init__(self):
    self.n_coef = self.filter_order + 1
    if self.init_coef is None:
      self.w = np.zeros((self.n_coef,), dtype=np.complex64)
    else:
      self.w = np.asarray(self.init_coef, dtype=np.complex64).copy()
      if self.w.shape != (self.n_coef,):
        raise ValueError(f"init_coef must have shape {(self.n_coef,)}, got {self.w.shape}.")

@dataclass
class LMS(BaseLMS):
  """
  Complex LMS (Diniz Algorithm 3.2)
    w(k+1) = w(k) + 2 mu e*(k) x(k)
  """

  def __call__(self, x: Array, d: Array, train: bool = True) -> Tuple[Array, Array, Array]:
    x = np.asarray(x)
    d = np.asarray(d)
    N = x.shape[0]

    x_pad = _prepad(x, self.n_coef)
    w = self.w.copy()

    y_hist = np.empty((N,), dtype=np.complex64)
    e_hist = np.empty((N,), dtype=np.complex64)
    w_hist = np.empty((N + 1, self.n_coef), dtype=np.complex64)
    w_hist[0] = w

    for k in range(N):
      reg = x_pad[k:k + self.n_coef][::-1].astype(np.complex64)
      y = np.vdot(w, reg)  # w^H reg
      e = np.complex64(d[k]) - y
      if train:
        w = (w + (2.0 * self.mu) * np.conj(e) * reg).astype(np.complex64)
        
      y_hist[k] = y
      e_hist[k] = e
      w_hist[k + 1] = w

    if train:
      self.w = w

    return y_hist, e_hist, w_hist

  def excessMSE(self, R: Array, delta_w_cov: Array) -> float:
    """Excess MSE = tr[R * cov(Δw)]."""
    return float(np.trace(R @ delta_w_cov).real)

  def isStableR(self, R: Array) -> bool:
    """Check stability condition by Equation 3.19 and Equation 3.30"""
    eig = np.linalg.eigvals(R)
    lam_max = np.max(np.real(eig))
    tr = np.trace(R).real
    return (0.0 < self.mu < 1.0 / lam_max) and (0.0 < self.mu < 1.0 / tr)

  def analyze(self, R: Array, verbose: bool = False) -> Dict[str, Any]:
    """Analyze step size and suggest improvements"""
    eig = np.linalg.eigvals(R)
    lam_max = float(np.max(np.real(eig)))
    lam_min = float(np.min(np.real(eig)))
    tr = float(np.trace(R).real)

    bound_eig = 1.0 / lam_max
    bound_tr = 1.0 / tr
    is_stable = self.isStableR(R)

    safe_mu = 0.1 * bound_tr
    aggressive_mu = 0.5 * bound_tr

    spread = (lam_max / lam_min) if lam_min > 0 else float("inf")
    misadj = (self.mu * tr) / (1.0 - self.mu * tr) if (1.0 - self.mu * tr) != 0 else float("inf")

    res = dict(
      is_stable_R=is_stable,
      current_mu=float(self.mu),
      max_stable_mu_eigenvalue=float(bound_eig),
      max_stable_mu_trace=float(bound_tr),
      suggested_conservative_mu=float(safe_mu),
      suggested_aggressive_mu=float(aggressive_mu),
      eigenvalue_spread=float(spread),
      convergence_warning=bool(spread > 100.0),
      theoretical_misadjustment=float(misadj),
      lambda_max=float(lam_max),
      lambda_min=float(lam_min),
      trace_R=float(tr),
    )

    if verbose:
      self.printMuAnalysis(res)
    return res

  def printMuAnalysis(self, a: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("LMS STEP SIZE ANALYSIS")
    print("=" * 60)
    print(f"Current mu: {a['current_mu']:.6f}")
    print(f"Stability: {'O STABLE' if a['is_stable_R'] else 'X UNSTABLE'}")

    print("\nEIGENVALUE ANALYSIS:")
    print(f"  Max eigenvalue (lambda_max): {a['lambda_max']:.6f}")
    print(f"  Min eigenvalue (lambda_min): {a['lambda_min']:.6f}")
    print(f"  Eigenvalue spread: {a['eigenvalue_spread']:.2f}")
    if a["convergence_warning"]:
      print("  WARNING: High eigenvalue spread will slow convergence")

    print("\nSTABILITY BOUNDS:")
    print(f"  From eigenvalues (1/lambda_max): {a['max_stable_mu_eigenvalue']:.6f}")
    print(f"  From trace (1/tr[R]): {a['max_stable_mu_trace']:.6f}")

    print("\nPERFORMANCE PREDICTIONS:")
    print(f"  Theoretical misadj: {a['theoretical_misadjustment']:.4f}")

    if a["lambda_min"] > 0:
      slow_tc = 1.0 / (2.0 * a["current_mu"] * a["lambda_min"])
      approx_iters = 4.6 * slow_tc
      print(f"  Approx. convergence iterations: {approx_iters:.0f}")

    print("\nRECOMMENDED mu RANGES:")
    print(f"  Conservative: {a['suggested_conservative_mu']:.6f}")
    print(f"  Aggressive:   {a['suggested_aggressive_mu']:.6f}")

    m1 = a["max_stable_mu_eigenvalue"] / a["current_mu"] if a["current_mu"] > 0 else float("inf")
    m2 = a["max_stable_mu_trace"] / a["current_mu"] if a["current_mu"] > 0 else float("inf")
    print("\nSAFETY MARGINS:")
    print(f"  Eigenvalue bound margin: {m1:.2f}x")
    print(f"  Trace bound margin:      {m2:.2f}x")
    if m1 < 2:
      print("  WARNING: Small stability margin")
    print("=" * 60)


@dataclass
class QuantizedLMS(BaseLMS):
  """
  Quantized complex LMS.

  Update:
    y_Q(k) = Q{ w_Q^H(k) x(k) }
    e_Q(k) = Q{ d(k) - y_Q(k) }
    w_Q(k+1) = Q{ w_Q(k) + mu * conj(e_Q(k)) * x_Q(k) }
  """

  q_total_bits: int = 16
  q_frac_bits: int = 12
  q_store_inputs: bool = False
  q_inner_mode: Literal["after_add", "after_product"] = "after_add"

  q: adspquant.FixedQuantizer = field(init=False)

  def __post_init__(self) -> None:
    super().__post_init__()
    self.q = adspquant.FixedQuantizer(
      total_bits=self.q_total_bits,
      frac_bits=self.q_frac_bits,
      signed=True,
    )
    self.w = self.q(np.asarray(self.w, dtype=np.complex64))

  def resetState(self) -> None:
    self.w = self.q(np.zeros((self.n_coef,), dtype=np.complex64))

  def __call__(self, x: Array, d: Array, train: bool = True) -> Dict[str, Array]:
    x_arr = np.asarray(x)
    d_arr = np.asarray(d)

    n_samples = x_arr.shape[0]
    x_pad = _prepad(x_arr, self.n_coef)

    w = self.q(self.w.copy())

    y_hist = np.empty((n_samples,), dtype=np.complex64)
    e_hist = np.empty((n_samples,), dtype=np.complex64)
    w_hist = np.empty((n_samples + 1, self.n_coef), dtype=np.complex64)
    w_hist[0] = w

    for k_idx in range(n_samples):
      x_k = x_pad[k_idx:k_idx + self.n_coef][::-1].astype(np.complex64)
      if self.q_store_inputs:
        x_k = self.q(x_k)

      y_q = adspquant.qdot(w, x_k, self.q, conj_a=True, mode=self.q_inner_mode)
      d_q = np.complex64(self.q(np.array(d_arr[k_idx], dtype=np.complex64))[()])
      e_q = np.complex64(self.q(d_q - y_q))

      if train:
        update_q = self.q(self.mu * np.conj(e_q) * x_k)
        w = self.q(w + update_q)

      y_hist[k_idx] = y_q
      e_hist[k_idx] = e_q
      w_hist[k_idx + 1] = w

    if train:
      self.w = w

    return {
      "y": y_hist,
      "e": e_hist,
      "w_hist": w_hist,
      "w_final": w,
    }

  def analyze(self, R: Optional[Array] = None, verbose: bool = False):
    """
    Quantized LMS analysis based on Diniz Chapter 3 & 4.

    Key relations:
      Stability:
        0 < μ < 1 / λ_max

      Misadjustment:
        M ≈ μ tr[R] / (1 - μ tr[R])

      Quantization:
        introduces additional steady-state error floor
    """

    mu = float(self.mu)

    if R is not None:
      eig = np.linalg.eigvals(R)
      lam_max = float(np.max(np.real(eig)))
      lam_min = float(np.min(np.real(eig)))
      tr = float(np.trace(R).real)

      is_stable = (0.0 < mu < 1.0 / lam_max)

      # misadjustment (Diniz Ch.3)
      if (1.0 - mu * tr) > 0:
        misadj = (mu * tr) / (1.0 - mu * tr)
      else:
        misadj = float("inf")

      spread = lam_max / lam_min if lam_min > 0 else float("inf")

    else:
      lam_max = None
      lam_min = None
      tr = None
      is_stable = None
      misadj = None
      spread = None

    # quantization noise proxy (very useful in practice)
    q_step = 2.0 ** (-self.q_frac_bits)
    q_noise_power = q_step**2 / 12.0

    res = dict(
      mu_value=mu,
      is_stable=is_stable,
      lambda_max=lam_max,
      lambda_min=lam_min,
      eigenvalue_spread=spread,
      trace_R=tr,
      theoretical_misadjustment=misadj,
      quantization_step=q_step,
      quantization_noise_power=q_noise_power,
    )

    if verbose:
      self.printAnalyzeLMSQuant(res)

    return res

  def printAnalyzeLMSQuant(self, a: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("QUANTIZED LMS ANALYSIS (Diniz Ch.3 / Ch.4)")
    print("=" * 60)

    print(f"Step size (μ): {a['mu_value']:.6f}")

    if a["is_stable"] is not None:
      print(f"Stability: {'O STABLE' if a['is_stable'] else 'X UNSTABLE'}")

    if a["lambda_max"] is not None:
      print("\nEIGENVALUE PROPERTIES:")
      print(f"  λ_max: {a['lambda_max']:.6f}")
      print(f"  λ_min: {a['lambda_min']:.6f}")
      print(f"  Spread: {a['eigenvalue_spread']:.2f}")

    if a["trace_R"] is not None:
      print("\nSTEADY-STATE PERFORMANCE:")
      print(f"  trace[R]: {a['trace_R']:.6f}")
      print(f"  Misadjustment ≈ {a['theoretical_misadjustment']:.6e}")

    print("\nQUANTIZATION EFFECT:")
    print(f"  Δ (LSB): {a['quantization_step']:.3e}")
    print(f"  Noise power ≈ {a['quantization_noise_power']:.3e}")

    print("\nDESIGN INSIGHT:")
    print("  • Larger μ → faster convergence, higher misadjustment")
    print("  • Quantization adds noise floor")
    print("  • High eigenvalue spread → slow LMS")

    print("=" * 60)

@dataclass
class QuantizedSharedLMS(BaseLMS):
  """
  Shared analysis for all sign-* / quantized-error LMS algorithms.
  Implements Chapter 4 analysis (Eq 4.13, 4.14, 4.21, 4.28).
  """

  xi_ema_beta: float = 0.99
  xi_hat: float = field(default=1.0, init=False)  # E|e|^2 estimate

  # xi tracking (shared)
  def update_xi_hat(self, e: np.complex64) -> None:
    p = float(np.abs(e)**2)
    self.xi_hat = (self.xi_ema_beta * self.xi_hat + (1.0 - self.xi_ema_beta) * p)

  def excessMSE(self, R: Array, delta_w_cov: Array) -> float:
    return float(np.trace(R @ delta_w_cov).real)

  def analyze(self, R: Array, verbose: bool = False) -> Dict[str, Any]:
    """
    Sign-Error LMS analysis based strictly on Diniz Chapter 4.
    Uses Eq. (4.13), (4.14), (4.21), and (4.28).
    """
    eig = np.linalg.eigvals(R)
    lam = np.real(eig)
    lam_max = float(np.max(lam))
    lam_min = float(np.min(lam))
    tr = float(np.trace(R).real)

    xi = float(self.xi_hat)

    # ---- Bounds from textbook ----
    # Eq (4.13) – mean convergence
    mu_max_mean = (1.0 / lam_max) * np.sqrt(np.pi * xi / 2.0)

    # Eq (4.14) – practical mean bound
    mu_max_trace = (1.0 / tr) * np.sqrt(np.pi * xi / 2.0)

    # Eq (4.21) – MSE (second-order) bound
    mu_max_mse = (1.0 / (2.0 * tr)) * np.sqrt(np.pi * xi / 2.0)

    is_stable_mean = 0.0 < self.mu < mu_max_mean
    is_stable_mse = 0.0 < self.mu < mu_max_mse

    # Eq (4.28)
    misadj = (self.mu * np.sqrt(np.pi / (2.0 * max(xi, 1e-20))) * tr)
    spread = lam_max / lam_min if lam_min > 0 else float("inf")

    res = dict(
      current_mu=float(self.mu),
      xi_hat=float(xi),
      mu_max_mean=float(mu_max_mean),
      mu_max_trace=float(mu_max_trace),
      mu_max_mse=float(mu_max_mse),
      is_stable_mean=is_stable_mean,
      is_stable_mse=is_stable_mse,
      theoretical_misadjustment=float(misadj),
      lambda_max=float(lam_max),
      lambda_min=float(lam_min),
      trace_R=float(tr),
      eigenvalue_spread=float(spread),
    )

    if verbose:
      self.printMuAnalysisSign(res)

    return res

  def printMuAnalysisSign(self, a: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("QUANTIZED / SIGN LMS ANALYSIS (Ch.4)")
    print("=" * 60)
    print(f"mu: {a['current_mu']:.6e}")
    print(f"xi_hat: {a['xi_hat']:.6e}")
    print("\nBounds:")
    print(f"Mean bound (4.13): {a['mu_max_mean']:.6e}")
    print(f"MSE bound  (4.21): {a['mu_max_mse']:.6e}")
    print(f"Mean stable: {'O' if a['is_stable_mean'] else 'X'}")
    print(f"MSE stable:  {'O' if a['is_stable_mse'] else 'X'}")
    print("\nMisadjustment (4.28):")
    print(f"M = {a['theoretical_misadjustment']:.6e}")
    print("=" * 60)

# -----------------------------------------------------------------------------
# Sign-* LMS family
# -----------------------------------------------------------------------------
@dataclass
class SignErrorLMS(QuantizedSharedLMS):
  """
  Complex Sign-Error LMS (Diniz Algorithm 4.1, complex extension)
    w(k+1) = w(k) + 2 mu sgn[e(k)] x(k)
  """

  def __call__(self, x: Array, d: Array, train: bool = True) -> Tuple[Array, Array, Array]:
    x = np.asarray(x)
    d = np.asarray(d)
    N = x.shape[0]
    x_pad = _prepad(x, self.n_coef)
    w = self.w.copy()

    y_hist = np.empty((N,), dtype=np.complex64)
    e_hist = np.empty((N,), dtype=np.complex64)
    w_hist = np.empty((N + 1, self.n_coef), dtype=np.complex64)
    w_hist[0] = w

    for k in range(N):
      reg = x_pad[k:k + self.n_coef][::-1].astype(np.complex64)
      y = np.vdot(w, reg)
      e = np.complex64(d[k]) - y
      if train:
        # track xi_hat = E|e|^2 (for effective-mu analysis)
        self.update_xi_hat(e)
        sigma_e = csignscalar(e)
        w = (w + (2.0 * self.mu) * sigma_e * reg).astype(np.complex64)
      y_hist[k] = y
      e_hist[k] = e
      w_hist[k + 1] = w

    if train:
      self.w = w

    return y_hist, e_hist, w_hist
  
@dataclass
class SignDataLMS(BaseLMS):
  """
  Complex Sign-Data LMS (Diniz Algorithm 4.2, complex extension)
    w(k+1) = w(k) + 2 mu e*(k) sgn[x(k)]
  """

  def __call__(self, x: Array, d: Array, train: bool = True) -> Tuple[Array, Array, Array]:
    x = np.asarray(x)
    d = np.asarray(d)
    N = x.shape[0]
    x_pad = _prepad(x, self.n_coef)
    w = self.w.copy()

    y_hist = np.empty((N,), dtype=np.complex64)
    e_hist = np.empty((N,), dtype=np.complex64)
    w_hist = np.empty((N + 1, self.n_coef), dtype=np.complex64)
    w_hist[0] = w

    for k in range(N):
      reg = x_pad[k:k + self.n_coef][::-1].astype(np.complex64)
      y = np.vdot(w, reg)
      e = np.complex64(d[k]) - y
      if train:
        sreg = csignvec(reg)
        w = (w + (2.0 * self.mu) * np.conj(e) * sreg).astype(np.complex64)
      y_hist[k] = y
      e_hist[k] = e
      w_hist[k + 1] = w

    if train:
      self.w = w
    return y_hist, e_hist, w_hist


@dataclass
class SignSignLMS(QuantizedSharedLMS):
  """
  Complex Sign-Sign LMS (Diniz Sec. 4.2.4 extended)
    w(k+1) = w(k) + 2 mu sgn[e(k)] sgn[x(k)]
  """

  def __call__(self, x: Array, d: Array, train: bool = True) -> Tuple[Array, Array, Array]:
    x = np.asarray(x)
    d = np.asarray(d)
    N = x.shape[0]
    x_pad = _prepad(x, self.n_coef)
    w = self.w.copy()

    y_hist = np.empty((N,), dtype=np.complex64)
    e_hist = np.empty((N,), dtype=np.complex64)
    w_hist = np.empty((N + 1, self.n_coef), dtype=np.complex64)
    w_hist[0] = w

    for k in range(N):
      reg = x_pad[k:k + self.n_coef][::-1].astype(np.complex64)
      y = np.vdot(w, reg)
      e = np.complex64(d[k]) - y
      if train:
        # track xi_hat = E|e|^2 (for effective-mu analysis)
        self.update_xi_hat(e)
        sigma_e = csignscalar(e)
        sreg = csignvec(reg)
        w = (w + (2.0 * self.mu) * sigma_e * sreg).astype(np.complex64)

      y_hist[k] = y
      e_hist[k] = e
      w_hist[k + 1] = w

    if train:
      self.w = w
    return y_hist, e_hist, w_hist

@dataclass
class DualSignLMS(QuantizedSharedLMS):
  """
  Complex Dual-Sign LMS (Diniz Sec. 4.2.3 style)
  if |e(k)| > rho:
      w(k+1) = w(k) + 2 mu ε sgn[e(k)] x(k)
  else:
      w(k+1) = w(k) + 2 mu sgn[e(k)] x(k)
  """
  epsilon: float = 2.0
  rho: float = 1.0

  def __call__(self, x: Array, d: Array, train: bool = True) -> Tuple[Array, Array, Array]:
    x = np.asarray(x)
    d = np.asarray(d)
    N = x.shape[0]
    x_pad = _prepad(x, self.n_coef)
    w = self.w.copy()

    y_hist = np.empty((N,), dtype=np.complex64)
    e_hist = np.empty((N,), dtype=np.complex64)
    w_hist = np.empty((N + 1, self.n_coef), dtype=np.complex64)
    w_hist[0] = w

    for k in range(N):
      reg = x_pad[k:k + self.n_coef][::-1].astype(np.complex64)
      y = np.vdot(w, reg)
      e = np.complex64(d[k]) - y
      if train:
        # track xi_hat = E|e|^2 (for effective-mu analysis)
        self.update_xi_hat(e)
        sigma_e = csignscalar(e)
        gain = self.epsilon if (np.abs(e) > self.rho) else 1.0
        w = (w + (2.0 * self.mu) * gain * sigma_e * reg).astype(np.complex64)

      y_hist[k] = y
      e_hist[k] = e
      w_hist[k + 1] = w

    if train:
      self.w = w
    return y_hist, e_hist, w_hist

@dataclass
class PowerOfTwoErrorLMS(QuantizedSharedLMS):
  """
  Power-of-Two Error LMS (Diniz Sec. 4.2.3)
    pe[e] as in Eq. (4.40)
    w(k+1) = w(k) + 2 mu pe[e(k)] x(k)
  """
  bd: int = 8
  tau: float = 0.0

  def p2e(self, e: Union[complex, Array], eps: float = 1e-12) -> np.complex64:
    e = np.asarray(e, dtype=np.complex64)
    abs_e = np.abs(e)
    s = csignscalar(e, eps=eps)  # already conj(e)/(abs+eps)
    thresh = 2.0 ** (-(self.bd - 1))

    if abs_e >= 1.0:
      return s
    elif abs_e >= thresh:
      pow_mag = np.exp2(np.floor(np.log2(abs_e + eps))).astype(np.float32)
      return (pow_mag * s).astype(np.complex64)
    else:
      return np.complex64(self.tau) * s

  def __call__(self, x: Array, d: Array, train: bool = True) -> Tuple[Array, Array, Array]:
    x = np.asarray(x)
    d = np.asarray(d)
    N = x.shape[0]
    x_pad = _prepad(x, self.n_coef)
    w = self.w.copy()

    y_hist = np.empty((N,), dtype=np.complex64)
    e_hist = np.empty((N,), dtype=np.complex64)
    w_hist = np.empty((N + 1, self.n_coef), dtype=np.complex64)
    w_hist[0] = w

    for k in range(N):
      reg = x_pad[k:k + self.n_coef][::-1].astype(np.complex64)
      y = np.vdot(w, reg)
      e = np.complex64(d[k]) - y
      if train:
        # track xi_hat = E|e|^2 (for effective-mu analysis)
        self.update_xi_hat(e)
        pe = np.conj(self.p2e(e)).astype(np.complex64)
        w = (w + (2.0 * self.mu) * pe * reg).astype(np.complex64)

      y_hist[k] = y
      e_hist[k] = e
      w_hist[k + 1] = w

    if train:
      self.w = w
    return y_hist, e_hist, w_hist

@dataclass
class LMSNewton(BaseLMS):
  """
  Complex LMS-Newton adaptive filter.
  Implements the LMS-Newton algorithm as described in:
  Diniz, Adaptive Filtering: Algorithms and Practical Implementation.
  Update equations:
      y(k) = w^H(k) x(k)
      e(k) = d(k) - y(k)
      p(k) = P(k) x(k)
      phi(k) = x^H(k) p(k)
      k(k) = p(k) / (alpha + phi(k))
      w(k+1) = w(k) + 2 mu e*(k) p(k)
      P(k+1) = ( P(k) - k(k) x^H(k) P(k) ) / alpha
  where:
      alpha ∈ (0, 1] is the forgetting factor
      P(0) = (1/delta) I
  """
  alpha: float = 0.99           # forgetting factor (≈ 1)
  delta: float = 1e-2           # P(0) = (1/delta) I
  eps: float = 1e-12            # small numerical guard
  R_hat_inv: Array = field(init=False)  # P(k)

  def __post_init__(self):
    super().__post_init__()
    if not (0.0 < self.alpha <= 1.0):
      raise ValueError("alpha must satisfy 0 < alpha <= 1.")
    # P(0) = (1/delta) I
    self.R_hat_inv = (1.0 / self.delta) * np.eye(self.n_coef, dtype=np.complex64)

  def __call__(self, x: Array, d: Array, train: bool = True) -> Tuple[Array, Array, Array]:
    x = np.asarray(x)
    d = np.asarray(d)
    N = x.shape[0]
    x_pad = _prepad(x, self.n_coef)
    w = self.w.copy()
    P = self.R_hat_inv.copy()
    y_hist = np.empty((N,), dtype=np.complex64)
    e_hist = np.empty((N,), dtype=np.complex64)
    w_hist = np.empty((N + 1, self.n_coef), dtype=np.complex64)
    w_hist[0] = w

    for k in range(N):
      reg = x_pad[k : k + self.n_coef][::-1].astype(np.complex64)

      y = np.vdot(w, reg)          # w^H x
      e = np.complex64(d[k]) - y

      if train:
        x_col = reg.reshape(-1, 1)
        # p = P x
        p = P @ x_col
        phi = (x_col.conj().T @ p).item()
        denom = ((1.0 - self.alpha) / self.alpha) + phi
        if abs(denom) < self.eps:
          denom = denom + (self.eps + 0.0j)

        P = (P - (p @ p.conj().T) / denom) / (1.0 - self.alpha)
        w = (w + 2.0 * self.mu * e * (P @ x_col).ravel()).astype(np.complex64)

      y_hist[k] = y
      e_hist[k] = e
      w_hist[k + 1] = w

    if train:
      self.w = w
      self.R_hat_inv = P

    return y_hist, e_hist, w_hist

# Transform-domain (power-normalized) TD-LMS / TD-NLMS
@dataclass
class TransformDomain(BaseLMS):
  """
  Transform-Domain (Power-Normalized) NLMS / TD-LMS (Diniz Sec. 4.5)

    s(k) = T x_reg(k)
    p_i(k) = alpha |s_i(k)|^2 + (1-alpha) p_i(k-1)
    w_hat(k+1) = w_hat(k) + 2*mu * e*(k) * ( s(k) / (gamma + p(k)) )

  self.w stores transform-domain coefficients w_hat.
  """
  matrix: str = "dct"     # "dct" or "dft"
  alpha: float = 0.01
  gamma: float = 1e-8
  init_power: float = 1.0

  def __post_init__(self):
    super().__post_init__()

    M = self.n_coef
    mat = self.matrix.lower()
    if mat == "dct":
      T = dctOrthoMatrix(M, dtype=np.float32).astype(np.complex64)
    elif mat == "dft":
      T = dftUnitaryMatrix(M, dtype=np.complex64)
    else:
      raise ValueError(f"Unknown transform '{self.matrix}', use 'dct' or 'dft'.")

    self.T = T
    self.TH = np.conj(T).T
    self.power = (self.init_power * np.ones((M,), dtype=np.float32))

  def __call__(self, x: Array, d: Array, train: bool = True) -> Tuple[Array, Array, Array]:
    x = np.asarray(x)
    d = np.asarray(d)
    N = x.shape[0]
    M = self.n_coef

    x_pad = _prepad(x, M)
    w_hat = self.w.copy()              # transform-domain coeffs
    p = self.power.copy()              # real power vector

    y_hist = np.empty((N,), dtype=np.complex64)
    e_hist = np.empty((N,), dtype=np.complex64)
    w_time_hist = np.empty((N + 1, M), dtype=np.complex64)

    w_time0 = self.TH @ w_hat
    w_time_hist[0] = w_time0

    for k in range(N):
      reg = x_pad[k:k + M][::-1].astype(np.complex64)
      s = (self.T @ reg).astype(np.complex64)

      p_new = (self.alpha * (np.abs(s) ** 2).astype(np.float32) + (1.0 - self.alpha) * p).astype(np.float32)

      y = np.vdot(w_hat, s)
      e = np.complex64(d[k]) - y

      if train:
        denom = (self.gamma + p_new).astype(np.float32)
        w_hat = (w_hat + (2.0 * self.mu) * (np.conj(e) * s / denom.astype(np.complex64))).astype(np.complex64)
        p = p_new
      # if not training, still advance y/e histories; keep p unchanged

      y_hist[k] = y
      e_hist[k] = e
      w_time_hist[k + 1] = (self.TH @ w_hat)

    if train:
      self.w = w_hat
      self.power = p

    return y_hist, e_hist, w_time_hist

@dataclass
class NLMSController:
  """
  Variable Step-Size Controller for NLMS
  ---------------------------------------
  Diniz, Adaptive Filtering: Algorithms and Practical Implementation
      Sec. 4.3 (Normalized LMS)
      Variable step-size discussion

  Theory
  ------
  Under Assumptions 1-3 (independence, small step-size,
  and flat spectral input assumption), the optimal
  normalized step size is:

      u_opt(n) = E{|w_delta_u(n)|^2} / E{|e(n)|^2}

  where
      w_delta_u(n) = ε^H(n) u(n)  (noise-free a priori error)
      e(n)   = w_delta_u(n) + v(n)

  Since:
      E{|e(n)|^2} = E{|w_delta_u(n)|^2} + sigma_n^2

  We estimate:

      w_delta_u^2 ≈ e_power - sigma_n^2

  Therefore:

      u_opt ≈ (e_power - sigma_n^2) / e_power

  This matches Haykin Eq. (7.19) after applying
  Assumptions 1-3.

  Practical Notes
  ---------------
  • u_opt → 1 during initial convergence
  • u_opt → 0 near steady state
  • u_opt automatically adapts to SNR
  • Must satisfy stability condition: 0 < u < 2
  """

  # Short-term power smoothing factors
  # First-order recursive estimators:
  #   u_power(n) = γ_u u_power(n-1) + (1-γ_u) ||u(n)||^2
  #   e_power(n) = γ_e e_power(n-1) + (1-γ_e) |e(n)|^2
  #
  # (Haykin Sec. 7.4 practical implementation)
  gamma_u: float = 0.95
  gamma_e: float = 0.95

  # Noise variance estimate sigma_n^2
  # Can be obtained during silence periods (echo canceller practice)
  sigma_v2: float = 0.0

  # Upper bound to enforce NLMS stability (Haykin Eq. 7.18)
  mu_max: float = 1.0

  # Numerical safeguard
  eps: float = 1e-12

  # Internal state (running power estimates)
  u_power: float = 0.0
  e_power: float = 0.0

  def update(self, reg: np.ndarray, e: complex) -> float:
    """
    Update short-term power estimates and compute adaptive u.

    Parameters
    ----------
    reg : ndarray
        Regressor vector u(n)
    e : complex
        A priori error e(n)

    Returns
    -------
    mu_opt : float
        Adaptive normalized step size μ_n
    """

    # 1) Instantaneous power measurements
    # ||u(n)||^2
    u_inst = float(np.real(np.vdot(reg, reg)))

    # |e(n)|^2
    e_inst = float(np.abs(e)**2)

    # 2) First-order exponential smoothing (MSD estimation approach)
    self.u_power = (self.gamma_u * self.u_power + (1 - self.gamma_u) * u_inst)
    self.e_power = (self.gamma_e * self.e_power + (1 - self.gamma_e) * e_inst)

    # 3) Estimate noise-free error power
    #    w_delta_u^2 ≈ e_power - sigma_n^2
    #
    # From:
    #   E{|e|^2} = E{|w_delta_u|^2} + sigma_n^2
    xi_u2 = max(self.e_power - self.sigma_v2, 0.0)

    # 4) Haykin optimal u (Eq. 7.19 under assumptions)
    #   u_opt = E{|w_delta_u|^2} / E{|e|^2}
    if self.e_power > self.eps:
      mu_opt = xi_u2 / self.e_power
    else:
      mu_opt = 0.0

    # 5) Enforce stability bound
    #    0 < u < 2  (Haykin Eq. 7.18)
    mu_opt = min(max(mu_opt, 0.0), self.mu_max)

    return mu_opt

@dataclass
class NLMS(BaseLMS):
  """
  Complex NLMS (Diniz Sec. 4.3)
    mu_k = mu_n / (tau + ||x||^2)
    w(k+1) = w(k) + mu_k e*(k) x(k)
  Here mu means mu_n (should satisfy 0 < mu_n < 2 for stability in the standard case).
  """
  tau: float = 1e-3
  controller: NLMSController | None = None

  def __call__(self, x: Array, d: Array, train: bool = True) -> Tuple[Array, Array, Array]:
    x = np.asarray(x)
    d = np.asarray(d)
    N = x.shape[0]

    x_pad = _prepad(x, self.n_coef)
    w = self.w.copy()

    y_hist = np.empty((N,), dtype=np.complex64)
    e_hist = np.empty((N,), dtype=np.complex64)
    w_hist = np.empty((N + 1, self.n_coef), dtype=np.complex64)
    w_hist[0] = w

    for k in range(N):
      reg = x_pad[k:k + self.n_coef][::-1].astype(np.complex64)
      y = np.vdot(w, reg)
      e = np.complex64(d[k]) - y
      
      if train:
        # step size control
        if self.controller is not None:
          mu_n = self.controller.update(reg, e)
        else:
          mu_n = self.mu
        norm = float(np.real(np.vdot(reg, reg)))
        mu_k = mu_n / (self.tau + norm)

        w = (w + mu_k * np.conj(e) * reg).astype(np.complex64)

      y_hist[k] = y
      e_hist[k] = e
      w_hist[k + 1] = w

    if train:
      self.w = w
    return y_hist, e_hist, w_hist
  
  def analyze(self, R: Array, verbose: bool = False):

    tr = float(np.trace(R).real)

    # If controller exists, u is time-varying
    if self.controller is not None:
      mu_current = None
      mu_bound = self.controller.mu_max
    else:
      mu_current = float(self.mu)
      mu_bound = 2.0

    is_stable = True if mu_bound <= 2.0 else False

    misadj_small_mu = None
    if mu_current is not None:
        misadj_small_mu = mu_current / 2.0

    res = dict(
      tau=self.tau,
      trace_R=tr,
      controller_enabled=self.controller is not None,
      stability_bound=2.0,
      mu_current=mu_current,
      mu_max=mu_bound,
      theoretical_misadjustment=misadj_small_mu,
    )

    if verbose:
      self.printMuAnalysisNLMS(res)

    return res

  def printMuAnalysisNLMS(self, a: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("NLMS STEP SIZE ANALYSIS (Haykin Ch.7 / Diniz Sec.4.3)")
    print("=" * 60)

    print(f"tau: {a['tau']:.6e}")
    print(f"trace[R]: {a['trace_R']:.6e}")

    print("\nStability Condition (Haykin Eq. 7.18):")
    print("  0 < u(n) < 2")

    if a["controller_enabled"]:
      print("\nStep-size Mode: VARIABLE (controller enabled)")
      print(f"  u_max enforced: {a['mu_max']:.6f}")
      print("  Stability guaranteed if mu_max < 2")

      if a["mu_current"] is not None:
        print(f"  Current μ̃: {a['mu_current']:.6f}")

    else:
      print("\nStep-size Mode: CONSTANT")

      print(f"  u_n(mu_n): {a['mu_current']:.6f}")
      print(f"  Stability: {'O STABLE' if a['mu_current'] < 2.0 else 'X UNSTABLE'}")

      print("\nEffective LMS-equivalent step size:")
      mu_eff = a['mu_current'] / (2.0 * a['trace_R'])
      print(f"  u_eff ≈ u_opt/ (2 tr[R]) = {mu_eff:.6e}")

      if a["theoretical_misadjustment"] is not None:
        print("\nTheoretical steady-state misadjustment (small u):")
        print(f"  M ≈ u_n / 2 = {a['theoretical_misadjustment']:.6f}")

      print("\nRecommended u_n ranges (practical):")
      print("  Conservative: 0.3-0.5")
      print("  Aggressive:   0.7-1.0")

    print("=" * 60)

@dataclass
class AffineProjection(BaseLMS):
  """
  Complex Affine Projection Algorithm

  P = projection order in Diniz problem notation L
  This implementation uses P+1 reused data vectors.
  """
  P: int = 3
  delta: float = 1e-3

  def __call__(self, x: Array, d: Array, train: bool = True) -> Tuple[Array, Array, Array]:
    x = np.asarray(x, dtype=np.complex64)
    d = np.asarray(d, dtype=np.complex64)

    N = x.shape[0]
    M = self.n_coef
    Pp = self.P + 1

    x_pad = _prepad(x, M)
    w = self.w.copy()

    n_out = N - self.P
    y_hist = np.empty((n_out,), dtype=np.complex64)
    e_hist = np.empty((n_out,), dtype=np.complex64)
    w_hist = np.empty((n_out + 1, M), dtype=np.complex64)
    w_hist[0] = w

    out_idx = 0
    for k in range(self.P, N):
      # Xap(k): shape (M, P+1)
      cols = []
      for i in range(Pp):
        kk = k - i
        # reverse to get [x(k-i), x(k-i-1), ..., x(k-i-M+1)]
        if kk < 0:
          # pad zero for current time < 0 for aligning
          reg = np.zeros((M,), dtype=np.complex64)
        else:
          reg = x_pad[kk:kk + M][::-1].astype(np.complex64)
        cols.append(reg)

      Xap = np.stack(cols, axis=1)  # (M, P+1)
      
      d_ap = np.array([d[k - i] if (k - i) >= 0 else 0.0 for i in range(Pp)], dtype=np.complex64)
      # Diniz complex notation: y_ap = X_ap^T w*
      y_ap = Xap.T @ np.conj(w)     # (P+1,)
      e_ap = d_ap - y_ap            # (P+1,)

      R = Xap.conj().T @ Xap
      R_reg = R + self.delta * np.eye(Pp, dtype=np.complex64)

      try:
        g = np.linalg.solve(R_reg, np.conj(e_ap))
      except np.linalg.LinAlgError:
        g = (np.linalg.pinv(R_reg) @ np.conj(e_ap)).astype(np.complex64)

      upd = (self.mu * (Xap @ g)).astype(np.complex64)

      if train:
        w = (w + upd).astype(np.complex64)

      y_hist[out_idx] = y_ap[0]
      e_hist[out_idx] = e_ap[0]
      w_hist[out_idx + 1] = w
      out_idx += 1

    if train:
      self.w = w

    return y_hist, e_hist, w_hist

  def analyze(self, verbose=False):
    mu = float(self.mu)
    Lp = self.P + 1

    is_stable = (0.0 < mu < 2.0)

    num1 = (Lp * mu) / (2.0 - mu)
    num2 = 1.0 - (1.0 - mu) ** 2
    den2 = 1.0 - (1.0 - mu) ** (2 * Lp)

    if abs(den2) < 1e-12:
      M_exact = np.nan
    else:
      M_exact = num1 * (num2 / den2)

    M_approx = (Lp * mu) / (2.0 - mu)

    res = dict(
      mu=mu,
      projection_order=self.P,
      projection_dimension=Lp,
      is_stable=is_stable,
      misadjustment_exact=M_exact,
      misadjustment_approx=M_approx,
      stability_bound_upper=2.0,
    )

    if verbose:
      self.printAnalyzeAPA(res)

    return res

  def printAnalyzeAPA(self, a):
    print("\n" + "=" * 60)
    print("AFFINE PROJECTION ANALYSIS")
    print("=" * 60)
    print(f"Projection order P: {a['projection_order']}")
    print(f"Projection dimension P+1: {a['projection_dimension']}")
    print(f"mu: {a['mu']:.6f}")
    print(f"Stability: {'O STABLE' if a['is_stable'] else 'X UNSTABLE'}")
    print(f"Exact misadjustment: {a['misadjustment_exact']:.6f}")
    print(f"Approx misadjustment: {a['misadjustment_approx']:.6f}")
    print("=" * 60)

  
@dataclass
class OverLapSaveFDAF(BaseLMS):

  alpha: float = 0.9   # power smoothing factor
  eps: float = 1e-8    # numerical safety

  def __post_init__(self):
    self.fft_size = 2 * (self.filter_order + 1)
    super().__post_init__()
    self.Wf = np.zeros(self.fft_size, dtype=np.complex64)

    # initialize first M taps in frequency domain
    w_time = np.zeros(self.fft_size, dtype=np.complex64)
    w_time[:self.n_coef] = self.w
    self.Wf = np.fft.fft(w_time)

    self.pow_est = np.ones(self.fft_size, dtype=np.float32) * self.eps

  @property
  def w(self):
    return np.fft.ifft(self.Wf).real[:self.n_coef]

  @w.setter
  def w(self, value):
    value = np.asarray(value, dtype=np.complex64)
    if value.shape != (self.n_coef,):
      raise ValueError("Wrong shape")

    w_time = np.zeros(self.fft_size, dtype=np.complex64)
    w_time[:self.n_coef] = value
    self.Wf = np.fft.fft(w_time)

  def __call__(self, x: Array, d: Array, train: bool = True) -> Tuple[Array, Array, Array]:

    x = np.asarray(x, dtype=np.complex64)
    d = np.asarray(d, dtype=np.complex64)

    N = x.shape[0]
    M = self.n_coef
    L = self.fft_size

    if N < M:
      raise ValueError("Input must be longer than filter order")

    # Pad input for overlap-save
    x_pad = np.concatenate([np.zeros(M, dtype=np.complex64), x])

    y_hist = np.zeros(N, dtype=np.complex64)
    e_hist = np.zeros(N, dtype=np.complex64)
    w_hist = np.zeros((N + 1, M), dtype=np.complex64)

    # time-domain weights (for history)
    w_time = np.fft.ifft(self.Wf).real[:M]
    w_hist[0] = w_time

    idx = 0
    while idx + M <= N:

      # ---- Overlap-Save block ----
      x_block = x_pad[idx:idx + L]
      X = np.fft.fft(x_block)

      # Frequency-domain convolution
      Yf = X * self.Wf
      y_block = np.fft.ifft(Yf)

      # discard first half (circular part)
      y_valid = y_block[M:].real
      d_block = d[idx:idx + M]

      e_block = d_block - y_valid

      y_hist[idx:idx + M] = y_valid
      e_hist[idx:idx + M] = e_block

      if train:

        # zero pad error (alignment)
        e_pad = np.concatenate([np.zeros(M), e_block])
        Ef = np.fft.fft(e_pad)

        # power normalization
        self.pow_est = (
          self.alpha * self.pow_est
          + (1 - self.alpha) * (np.abs(X) ** 2)
        )

        Ef /= (self.pow_est + self.eps)

        # gradient
        Gf = X.conj() * Ef

        # ---- linear constraint ----
        g_time = np.fft.ifft(Gf)
        g_time[M:] = 0   # enforce FIR length M
        Gf = np.fft.fft(g_time)

        # update
        self.Wf += self.mu * Gf

      # update weight history
      w_time = np.fft.ifft(self.Wf).real[:M]
      w_hist[idx + M] = w_time

      idx += M

    # save back time-domain weights
    if train:
      self.w = np.fft.ifft(self.Wf).real[:M]

    return y_hist, e_hist, w_hist

@dataclass
class SlidingDctLMS(BaseLMS):
  beta: float = 0.99
  gamma: float = 0.98
  eps: float = 1e-8

  def __post_init__(self):
    super().__post_init__()

    M = self.n_coef
    self.m = np.arange(M, dtype=np.float64)

    # Table 8.2: alpha = 1/(2M)
    self.alpha = 1.0 / (2.0 * M)

    # W_{2M} = exp(-j2π/(2M))
    self.W2M = np.exp(-1j * 2.0 * np.pi / (2.0 * M))

    # W_{2M}^m and W_{2M}^{-m}
    self.Wm = self.W2M ** self.m
    self.Wm_inv = np.conj(self.Wm)

    # (-1)^m and k_m
    self.sign = (-1.0) ** self.m
    self.km = np.ones(M, dtype=np.float64)
    self.km[0] = 1.0 / np.sqrt(2.0)

    # States (Table 8.2 initialization)
    self.A1 = np.zeros(M, dtype=np.complex128)
    self.A2 = np.zeros(M, dtype=np.complex128)
    self.lambda_hat = np.zeros(M, dtype=np.float64)

    # delay line holds u(n-1)...u(n-M)
    self.delay_line = np.zeros(M, dtype=np.float64)

    # IMPORTANT: Table 8.2 uses real w_hat (DCT-domain weights)
    self.w = np.zeros(M, dtype=np.float64)

    self.n_iter = 1

  def estR(self, x):
    """Sample correlation"""
    x = np.asarray(x)
    return np.outer(x, x.conj())

  def step(self, u: float, d: float | None = None, train: bool = True):
    """
    One-sample update of Table 8.2 DCT-LMS.
    Returns: y, e, C  (C is the DCT-domain input vector)
    """

    M = self.n_coef
    u = np.float64(u)

    # sliding window delay
    u_old = self.delay_line[-1]
    self.delay_line[1:] = self.delay_line[:-1]
    self.delay_line[0] = u

    # sliding DCT innovation term: u(n) - beta^M (-1)^m u(n-M) 
    innovation = u - (self.beta ** M) * self.sign * u_old   # vector over m

    # Fig 8.6 / Table 8.2 recursions
    self.A1 = self.beta * self.Wm     * self.A1 + innovation
    self.A2 = self.beta * self.Wm_inv * self.A2 + self.Wm_inv * innovation

    A = self.A1 + self.A2

    # C_m(n)
    C = 0.5 * self.km * self.sign * (self.W2M ** (self.m / 2.0)) * A
    C = C.real

    # filter output
    y = float(np.dot(C, self.w))

    if d is None:
      # no training signal
      self.n_iter += 1
      return y, 0.0, C

    d = np.float64(d)
    e = float(d - y)

    # eigenvalue estimation Eq (8.74)
    self.lambda_hat = (
      self.gamma * self.lambda_hat
      + (1.0 / self.n_iter) * (C*C - self.gamma * self.lambda_hat)
    )
    self.lambda_hat = np.maximum(self.lambda_hat, self.eps)

    # ---- weight update Table 8.2 ----
    if train:
      step_size = self.mu if self.mu is not None else self.alpha
      self.w += (step_size / self.lambda_hat) * C * e

    self.n_iter += 1
    return y, e, C

  def __call__(self, x, d=None, train=True):
    """Run DCT-LMS over a sequence."""

    x = np.asarray(x, dtype=np.float64)
    N = len(x)

    if d is not None:
      d = np.asarray(d, dtype=np.float64)
      if len(d) != N:
        raise ValueError("x and d must have same length")

    y_hist = np.zeros(N, dtype=np.float64)
    e_hist = np.zeros(N, dtype=np.float64)
    C_hist = np.zeros((N, self.n_coef), dtype=np.float64)

    for n in range(N):
      if d is None:
        y, e, C = self.step(x[n], None, train=False)
      else:
        y, e, C = self.step(x[n], d[n], train=train)

      y_hist[n] = y
      e_hist[n] = e
      C_hist[n] = C

    return y_hist, e_hist, C_hist


def rlsMisadj2Lambda(M_desired: float, filter_order: int, K: float = 2.0,) -> float:
  """
  Solve for λ (forgetting factor) from desired RLS misadjustment using Diniz Example 5.3.

  Uses:
    a = (1 - λ) / (1 + λ)
    M = (N+1) a (1 + 2 a K)

  Args:
    M_desired: desired misadjustment M
    filter_order: adaptive filter order N
    K: ratio term used in the textbook expression

  Returns:
    λ in (0, 1]
  """

  if M_desired <= 0:
    raise ValueError("M_desired must be > 0")
  if filter_order < 0:
    raise ValueError("filter_order must be >= 0")
  if K < 0:
    raise ValueError("K must be >= 0")

  L = filter_order + 1  # number of taps

  # 2K a^2 + a - M/L = 0
  c2 = 2.0 * K
  c1 = 1.0
  c0 = -M_desired / L

  disc = c1 * c1 - 4.0 * c2 * c0
  if disc < 0:
    raise ValueError("No real solution for a")

  sqrt_disc = np.sqrt(disc)

  # physical valid root: a > 0
  a1 = (-c1 + sqrt_disc) / (2.0 * c2)
  a2 = (-c1 - sqrt_disc) / (2.0 * c2)

  a = a1 if a1 > 0 else a2
  if a <= 0:
    raise ValueError("No positive solution for a")

  lam = (1.0 - a) / (1.0 + a)

  if not (0.0 < lam <= 1.0):
    raise ValueError("Computed λ is out of valid range")

  return float(lam)

def rlsMisadj(lam: float, filter_order: int, K: float = 2.0,) -> float:
  """
  Compute RLS misadjustment:
    M = (N+1) a (1 + 2 a K),
    a = (1-λ)/(1+λ)
  """
  if not (0.0 < lam <= 1.0):
    raise ValueError("lam must satisfy 0 < lam <= 1")
  if filter_order < 0:
    raise ValueError("filter_order must be >= 0")
  if K < 0:
    raise ValueError("K must be >= 0")

  a = (1.0 - lam) / (1.0 + lam)
  L = filter_order + 1
  return float(L * a * (1.0 + 2.0 * a * K))

def rlsMisadj2Ord(M_desired: float, lam: float, K: float = 2.0,) -> int:
  """
  Solve for filter order N from desired misadjustment:
    M = (N+1) a (1 + 2 a K)
  """
  if M_desired <= 0:
    raise ValueError("M_desired must be > 0")
  if not (0.0 < lam <= 1.0):
    raise ValueError("lam must satisfy 0 < lam <= 1")
  if K < 0:
    raise ValueError("K must be >= 0")

  a = (1.0 - lam) / (1.0 + lam)
  denom = a * (1.0 + 2.0 * a * K)
  if denom <= 0:
    raise ValueError("Invalid denominator")

  N = M_desired / denom - 1.0
  return int(np.round(N))

def rlsSnrInitParams(
    snr_db: float,
    n_coef: int,
    sigma_u2: float = 1.0,
    lam: float = 0.99,
    delta_high_snr: float = 100.0,
    delta_mid_snr: float = 1.0,
    delta_low_snr: float = 0.01,
) -> dict:
  """
  Intuitive SNR-based initialization for a Diniz-style RLS class.

  Your class uses:
      S_D(-1) = delta * I

  Intuition:
    high SNR -> trust data early -> weaker prior protection
             -> larger Diniz delta

    low SNR  -> trust data less early -> stronger prior protection
             -> smaller Diniz delta

  lambda is kept fixed by default because Haykin's SNR discussion is mainly
  about initialization / regularization, not about changing lambda a lot.
  """
  if n_coef <= 0:
    raise ValueError("n_coef must be > 0")
  if sigma_u2 <= 0.0:
    raise ValueError("sigma_u2 must be > 0")
  if not (0.0 < lam <= 1.0):
    raise ValueError("lam must satisfy 0 < lam <= 1")
  if not (delta_high_snr > 0.0 and delta_mid_snr > 0.0 and delta_low_snr > 0.0):
    raise ValueError("all deltas must be > 0")

  # piecewise-linear interpolation in SNR
  if snr_db >= 30.0:
    delta = delta_high_snr
    mode = "high_snr"
  elif snr_db <= -10.0:
    delta = delta_low_snr
    mode = "low_snr"
  elif snr_db >= 10.0:
    t = (snr_db - 10.0) / 20.0
    delta = delta_mid_snr + t * (delta_high_snr - delta_mid_snr)
    mode = "mid_to_high_snr"
  else:
    t = (snr_db + 10.0) / 20.0
    delta = delta_low_snr + t * (delta_mid_snr - delta_low_snr)
    mode = "low_to_mid_snr"

  effective_memory = 1.0 / (1.0 - lam) if lam < 1.0 else float("inf")

  return dict(
    snr_db=float(snr_db),
    n_coef=int(n_coef),
    sigma_u2=float(sigma_u2),
    lambda_value=float(lam),
    delta=float(delta),
    effective_memory=float(effective_memory),
    mode=mode,
  )

@dataclass
class RLS(BaseLMS):
  """
  Conventional Complex RLS (Diniz Algorithm 5.3)

    S_D(k) = R_D^{-1}(k)

    k(k) = S_D(k-1)x(k) / (λ + x^H(k)S_D(k-1)x(k))

    w(k) = w(k-1) + k(k)e*(k)

    S_D(k) = (1/λ)[ S_D(k-1) - k(k)x^H(k)S_D(k-1) ]

  Initialization:
    S_D(-1) = δ I
    w(-1)   = 0
  """
  mu = None
  lam: float = 0.99
  delta: float = 1e-2
  eps: float = 1e-12
  dtype: object = np.complex128
  S_D: Array = field(init=False)

  def __post_init__(self):
    self.mu = None
    super().__post_init__()
    self.S_D = self.delta * np.eye(self.n_coef, dtype=self.dtype)

  def __call__(self, x: Array, d: Array, train: bool = True):
    x = np.asarray(x, dtype=self.dtype)
    d = np.asarray(d, dtype=self.dtype)

    N = x.shape[0]
    x_pad = _prepad(x, self.n_coef)

    w = self.w.copy()
    S_D = self.S_D.copy()

    y_hist = np.empty((N,), dtype=self.dtype)
    e_hist = np.empty((N,), dtype=self.dtype)
    w_hist = np.empty((N + 1, self.n_coef), dtype=self.dtype)
    w_hist[0] = w

    for k_idx in range(N):
      xk = x_pad[k_idx:k_idx + self.n_coef][::-1].reshape(-1, 1)

      y = np.vdot(w, xk.ravel())
      e = d[k_idx] - y

      if train:
        psi = S_D @ xk
        denom = self.lam + np.real((xk.conj().T @ psi).item())
        if abs(denom) < self.eps:
          denom += self.eps

        k_vec = psi / denom
        w = w + k_vec.ravel() * np.conj(e)

        S_D = (S_D - k_vec @ (xk.conj().T @ S_D)) / self.lam

        # numerical hygiene
        S_D = 0.5 * (S_D + S_D.conj().T)

      y_hist[k_idx] = y
      e_hist[k_idx] = e
      w_hist[k_idx + 1] = w

    if train:
      self.w = w
      self.S_D = S_D

    return y_hist, e_hist, w_hist

  def analyze(self, R: Optional[Array] = None, verbose: bool = False):
    """
    RLS analysis based on Diniz Chapter 5.

    Structural points:
      Stability condition: 0 < λ ≤ 1
      Effective memory:    N_eff ≈ 1 / (1 - λ)

    Common first-order approximation for λ ≈ 1:
      M ≈ ((1 - λ) / 2) * tr[R]

    If R is not provided, only structural info is returned.
    """

    lam = float(self.lam)
    is_stable = (0.0 < lam <= 1.0)

    if lam < 1.0:
      N_eff = 1.0 / (1.0 - lam)
    else:
      N_eff = float("inf")

    if R is not None:
      tr = float(np.trace(R).real)
      misadj = (1.0 - lam) / 2.0 * tr
    else:
      tr = None
      misadj = None

    eigvals = np.linalg.eigvals(self.S_D)
    abs_eigs = np.abs(eigvals)
    min_abs = float(np.min(abs_eigs))
    max_abs = float(np.max(abs_eigs))
    cond_P = float(max_abs / min_abs) if min_abs > 0.0 else float("inf")

    res = dict(
      lambda_value=lam,
      is_stable=is_stable,
      effective_memory=float(N_eff),
      trace_R=tr,
      theoretical_misadjustment=misadj,
      P_condition_number=cond_P,
    )

    if verbose:
      self.printAnalyzeRLS(res)

    return res

  def printAnalyzeRLS(self, a: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("RLS ANALYSIS (Diniz Chapter 5)")
    print("=" * 60)

    print(f"Lambda (λ): {a['lambda_value']:.6f}")
    print(f"Stability: {'O STABLE' if a['is_stable'] else 'X UNSTABLE'}")

    print("\nMEMORY / TRACKING:")
    print(f"  Effective memory length ≈ {a['effective_memory']:.2f} samples")

    if a["effective_memory"] < 20:
      print("  WARNING: Very short memory -> noisy estimate")
    elif a["effective_memory"] > 1000:
      print("  WARNING: Very long memory -> slow tracking")

    print("\nSTEADY-STATE PERFORMANCE:")
    if a["trace_R"] is not None:
      print(f"  trace[R]: {a['trace_R']:.6f}")
      print(f"  Misadjustment ≈ {a['theoretical_misadjustment']:.6e}")
    else:
      print("  (R not provided -> misadjustment unavailable)")

    print("\nNUMERICAL CONDITIONING:")
    print(f"  cond(S_D): {a['P_condition_number']:.2e}")
    if a["P_condition_number"] > 1e6:
      print("  WARNING: S_D is ill-conditioned -> numerical instability risk")

    print("\nDESIGN INSIGHT:")
    print("  • λ -> 1     -> lower misadjustment, slower tracking")
    print("  • λ smaller  -> faster tracking, noisier estimate")
    print("  • Effective memory ≈ 1 / (1 - λ)")
    print("=" * 60)

@dataclass
class RLSAlt(BaseLMS):
  """
  Alternative Complex RLS (Diniz Algorithm 5.4)

    e(k) = d(k) - w^H(k-1)x(k)
    ψ(k) = S_D(k-1)x(k)
    S_D(k) = (1/λ)[ S_D(k-1) - ψ(k)ψ^H(k)/(λ + ψ^H(k)x(k)) ]
    w(k) = w(k-1) + e*(k) S_D(k)x(k)
  """
  mu = None
  lam: float = 0.99
  delta: float = 1e-2
  eps: float = 1e-12
  dtype: object = np.complex128

  S_D: Array = field(init=False)

  def __post_init__(self):
    self.mu = None
    super().__post_init__()
    self.S_D = self.delta * np.eye(self.n_coef, dtype=self.dtype)

  def __call__(self, x: Array, d: Array, train: bool = True):
    x = np.asarray(x, dtype=self.dtype)
    d = np.asarray(d, dtype=self.dtype)

    N = x.shape[0]
    x_pad = _prepad(x, self.n_coef)

    w = self.w.copy()
    S_D = self.S_D.copy()

    y_hist = np.empty((N,), dtype=self.dtype)
    e_hist = np.empty((N,), dtype=self.dtype)
    w_hist = np.empty((N + 1, self.n_coef), dtype=self.dtype)
    w_hist[0] = w

    for k_idx in range(N):
      xk = x_pad[k_idx:k_idx + self.n_coef][::-1].reshape(-1, 1)

      y = np.vdot(w, xk.ravel())
      e = d[k_idx] - y

      if train:
        psi = S_D @ xk
        denom = self.lam + np.real((psi.conj().T @ xk).item())
        if abs(denom) < self.eps:
          denom += self.eps

        S_D = (S_D - (psi @ psi.conj().T) / denom) / self.lam
        S_D = 0.5 * (S_D + S_D.conj().T)

        w = w + np.conj(e) * (S_D @ xk).ravel()

      y_hist[k_idx] = y
      e_hist[k_idx] = e
      w_hist[k_idx + 1] = w

    if train:
      self.w = w
      self.S_D = S_D

    return y_hist, e_hist, w_hist

@dataclass
class QuantizedRLSAlt(BaseLMS):
  """
  Quantized alternative complex RLS.

  Exact-form backbone:
    e(k)   = d(k) - w^H(k-1)x(k)
    psi(k) = S_D(k-1)x(k)
    S_D(k) = (1/lambda)[S_D(k-1) - psi(k)psi^H(k)/(lambda + psi^H(k)x(k))]
    w(k)   = w(k-1) + conj(e(k)) S_D(k)x(k)

  Quantized form:
    quantizes y, e, psi, denominator, S_D, gain, update, and w.
  """

  mu: Optional[float] = None
  lam: float = 0.99
  delta: float = 0.5
  eps: float = 1e-12

  q_total_bits: int = 16
  q_frac_bits: int = 12
  q_inner_mode: Literal["after_add", "after_product"] = "after_add"
  q_store_inputs: bool = False
  enforce_hermitian: bool = True
  gain_clip: Optional[float] = None
  update_clip: Optional[float] = None
  diagonal_loading: float = 0.0

  s_d: Array = field(init=False)
  q: adspquant.FixedQuantizer = field(init=False)

  def __post_init__(self) -> None:
    super().__post_init__()

    self.q = adspquant.FixedQuantizer(
      total_bits=self.q_total_bits,
      frac_bits=self.q_frac_bits,
      signed=True,
    )

    init_val = 1.0 / self.delta
    self.w = self.q(np.zeros((self.n_coef,), dtype=np.complex64))
    self.s_d = self.q(init_val * np.eye(self.n_coef, dtype=np.complex64))

  def resetState(self) -> None:
    init_val = 1.0 / self.delta
    self.w = self.q(np.zeros((self.n_coef,), dtype=np.complex64))
    self.s_d = self.q(init_val * np.eye(self.n_coef, dtype=np.complex64))

  def clip(self, x: Array, clip_value: Optional[float]) -> Array:
    if clip_value is None:
      return x
    x_arr = np.asarray(x)
    x_real = np.clip(np.real(x_arr), -clip_value, clip_value)
    x_imag = np.clip(np.imag(x_arr), -clip_value, clip_value)
    return (x_real + 1j * x_imag).astype(np.complex64)

  def makeRegressor(self, x_pad: Array, k_idx: int) -> Array:
    x_k = x_pad[k_idx:k_idx + self.n_coef][::-1].astype(np.complex64)
    if self.q_store_inputs:
      x_k = self.q(x_k)
    return x_k

  def __call__(self, x: Array, d: Array, train: bool = True) -> Dict[str, Array]:
    x_arr = np.asarray(x)
    d_arr = np.asarray(d)

    n_samples = x_arr.shape[0]
    x_pad = _prepad(x_arr, self.n_coef)

    w = self.q(self.w.copy())
    s_d = self.q(self.s_d.copy())

    y_hist = np.empty((n_samples,), dtype=np.complex64)
    e_hist = np.empty((n_samples,), dtype=np.complex64)
    eps_hist = np.empty((n_samples,), dtype=np.complex64)
    psi_hist = np.empty((n_samples, self.n_coef), dtype=np.complex64)
    denom_hist = np.empty((n_samples,), dtype=np.float32)
    w_hist = np.empty((n_samples + 1, self.n_coef), dtype=np.complex64)
    w_hist[0] = w

    for k_idx in range(n_samples):
      x_k = self.makeRegressor(x_pad, k_idx)

      y_q = adspquant.qdot(w, x_k, self.q, conj_a=True, mode=self.q_inner_mode)
      d_q = np.complex64(self.q(np.array(d_arr[k_idx], dtype=np.complex64))[()])
      e_q = np.complex64(self.q(d_q - y_q))

      eps_q = e_q
      psi_q = np.zeros((self.n_coef,), dtype=np.complex64)
      denom_q = np.float32(0.0)

      if train:
        psi_q = adspquant.qmatVec(s_d, x_k, self.q, mode=self.q_inner_mode)
        psi_q = self.q(psi_q)

        psi_h_x_q = adspquant.qdot(psi_q, x_k, self.q, conj_a=True, mode=self.q_inner_mode)
        denom_val = float(np.real(psi_h_x_q) + self.lam)
        denom_val = max(denom_val, self.eps)
        denom_q = np.float32(self.q.quantizeReal(np.array(denom_val))[()])

        if abs(denom_q) < self.eps:
          denom_q = np.float32(self.eps)

        outer_q = adspquant.qouter(psi_q, np.conj(psi_q), self.q)
        corr_q = self.q(outer_q / denom_q)

        s_d = self.q((s_d - corr_q) / self.lam)

        if self.diagonal_loading > 0.0:
          s_d = self.q(s_d + self.diagonal_loading * np.eye(self.n_coef, dtype=np.complex64))

        if self.enforce_hermitian:
          s_d = 0.5 * (s_d + np.conj(s_d.T))
          s_d = self.q(s_d)

        gain_q = adspquant.qmatVec(s_d, x_k, self.q, mode=self.q_inner_mode)
        gain_q = self.q(gain_q)
        gain_q = self.clip(gain_q, self.gain_clip)
        gain_q = self.q(gain_q)

        update_q = self.q(np.conj(e_q) * gain_q)
        update_q = self.clip(update_q, self.update_clip)
        update_q = self.q(update_q)

        w = self.q(w + update_q)

        y_post_q = adspquant.qdot(w, x_k, self.q, conj_a=True, mode=self.q_inner_mode)
        eps_q = np.complex64(self.q(d_q - y_post_q))

      y_hist[k_idx] = y_q
      e_hist[k_idx] = e_q
      eps_hist[k_idx] = eps_q
      psi_hist[k_idx] = psi_q
      denom_hist[k_idx] = denom_q
      w_hist[k_idx + 1] = w

    if train:
      self.w = w
      self.s_d = s_d

    return {
      "y": y_hist,
      "e": e_hist,          # a priori error
      "eps": eps_hist,      # a posteriori error
      "psi_hist": psi_hist,
      "denom_hist": denom_hist,
      "w_hist": w_hist,
      "w_final": w,
      "s_d_final": s_d,
    }
  
  def analyze(self, R: Optional[Array] = None, verbose: bool = False):
    """
    Quantized RLS analysis based strictly on Diniz Chapter 5.

    Key relations:
      Stability:
        always stable if 0 < λ ≤ 1

      Effective memory:
        N_eff ≈ 1 / (1 - λ)

      Misadjustment:
        M ≈ (1 - λ)/2 * tr[R]

    Quantization effects:
      finite precision causes P-conditioning degradation
      and additional steady-state error
    """

    lam = float(self.lam)

    is_stable = (0.0 < lam <= 1.0)

    # ---- memory ----
    if lam < 1.0:
      N_eff = 1.0 / (1.0 - lam)
    else:
      N_eff = float("inf")

    # ---- misadjustment ----
    if R is not None:
      tr = float(np.trace(R).real)
      misadj = (1.0 - lam) / 2.0 * tr
    else:
      tr = None
      misadj = None

    # ---- conditioning of P ----
    eigvals = np.linalg.eigvals(self.S_D)
    cond_P = float(np.max(np.abs(eigvals)) / np.min(np.abs(eigvals)))

    # ---- quantization ----
    q_step = 2.0 ** (-self.q_frac_bits)
    q_noise_power = q_step**2 / 12.0

    # sensitivity: RLS more sensitive than LMS
    quant_sensitivity = cond_P * q_noise_power

    res = dict(
      lambda_value=lam,
      is_stable=is_stable,
      effective_memory=float(N_eff),
      trace_R=tr,
      theoretical_misadjustment=misadj,
      P_condition_number=cond_P,
      quantization_step=q_step,
      quantization_noise_power=q_noise_power,
      quantization_sensitivity=quant_sensitivity,
    )

    if verbose:
      self.printAnalyzeRLSQuant(res)

    return res


  def printAnalyzeRLSQuant(self, a: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("QUANTIZED RLS ANALYSIS (Diniz Chapter 5)")
    print("=" * 60)

    print(f"Lambda (λ): {a['lambda_value']:.6f}")
    print(f"Stability: {'O STABLE' if a['is_stable'] else 'X UNSTABLE'}")

    print("\nMEMORY / TRACKING:")
    print(f"  Effective memory ≈ {a['effective_memory']:.2f}")

    if a["effective_memory"] < 20:
      print("  WARNING: Very short memory → noisy")
    elif a["effective_memory"] > 1000:
      print("  WARNING: Very long memory → slow tracking")

    if a["trace_R"] is not None:
      print("\nSTEADY-STATE PERFORMANCE:")
      print(f"  trace[R]: {a['trace_R']:.6f}")
      print(f"  Misadjustment ≈ {a['theoretical_misadjustment']:.6e}")

    print("\nNUMERICAL CONDITIONING:")
    print(f"  cond(P): {a['P_condition_number']:.2e}")

    if a["P_condition_number"] > 1e6:
      print("  WARNING: Ill-conditioned → high quantization sensitivity")

    print("\nQUANTIZATION EFFECT:")
    print(f"  Δ (LSB): {a['quantization_step']:.3e}")
    print(f"  Noise power ≈ {a['quantization_noise_power']:.3e}")
    print(f"  Sensitivity ≈ {a['quantization_sensitivity']:.3e}")

    print("\nDESIGN INSIGHT:")
    print("  • λ → 1 → low misadjustment, high sensitivity")
    print("  • Smaller λ → robust but noisier")
    print("  • RLS highly sensitive to quantization of P")

    print("=" * 60)


@dataclass
class SMPNLMS(BaseLMS):
  """
  Set-Membership NLMS.

  Complex convention:
    y(k) = w^H(k) x(k)
    e(k) = d(k) - y(k)

  Update:
    if |e(k)| <= gamma_bar:
      w(k+1) = w(k)
    else:
      w(k+1) = w(k) + mu_sm e*(k) x(k) / ||x(k)||^2

  where:
    mu_sm = 1 - gamma_bar / |e(k)|
  """

  gamma_bar: float = 0.0
  eps: float = 1e-12
  init_coef: Optional[Array] = None
  mu: Optional[float] = None

  update_flags: Array = field(init=False, repr=False)
  mu_hist: Array = field(init=False, repr=False)
  delta: float = 1e-2

  def __post_init__(self):
    super().__post_init__()

    if self.gamma_bar < 0.0:
      raise ValueError(f"gamma_bar must be non-negative, got {self.gamma_bar}.")

    if self.eps <= 0.0:
      raise ValueError(f"eps must be positive, got {self.eps}.")

  def __call__(self, x: Array, d: Array, train: bool = True) -> Tuple[Array, Array, Array]:
    x = np.asarray(x, dtype=np.complex64)
    d = np.asarray(d, dtype=np.complex64)

    if x.ndim != 1:
      raise ValueError(f"x must be 1-D, got shape {x.shape}.")

    if d.ndim != 1:
      raise ValueError(f"d must be 1-D, got shape {d.shape}.")

    if x.shape[0] != d.shape[0]:
      raise ValueError(f"x and d must have same length, got {x.shape[0]} and {d.shape[0]}.")

    N = x.shape[0]
    x_pad = _prepad(x, self.n_coef)
    w = self.w.copy()

    y_hist = np.empty((N,), dtype=np.complex64)
    e_hist = np.empty((N,), dtype=np.complex64)
    w_hist = np.empty((N + 1, self.n_coef), dtype=np.complex64)

    update_flags = np.zeros((N,), dtype=bool)
    mu_hist = np.zeros((N,), dtype=np.float32)

    w_hist[0] = w

    for k in range(N):
      reg = x_pad[k:k + self.n_coef][::-1].astype(np.complex64)

      y = np.vdot(w, reg)
      e = d[k] - y
      abs_e = float(np.abs(e))

      if train and abs_e > self.gamma_bar:
        norm_x = float(np.vdot(reg, reg).real)

        if norm_x > self.eps:
          mu_sm = 1.0 - self.gamma_bar / max(abs_e, self.eps)
          w = (w + mu_sm * np.conj(e) * reg / (norm_x + self.delta)).astype(np.complex64)

          update_flags[k] = True
          mu_hist[k] = np.float32(mu_sm)

      y_hist[k] = y
      e_hist[k] = e
      w_hist[k + 1] = w

    if train:
      self.w = w

    self.update_flags = update_flags
    self.mu_hist = mu_hist

    return y_hist, e_hist, w_hist

  def getUpdateRate(self) -> float:
    if not hasattr(self, "update_flags") or self.update_flags.size == 0:
      return 0.0
    return float(np.mean(self.update_flags))

  def updateRate(self) -> float:
    return self.getUpdateRate()

  def analyzeUpdates(self) -> Dict[str, Any]:
    if not hasattr(self, "update_flags"):
      return dict(update_rate=0.0, n_updates=0, mean_mu_sm=0.0)

    n_updates = int(np.sum(self.update_flags))
    update_rate = float(np.mean(self.update_flags)) if self.update_flags.size else 0.0

    if n_updates > 0:
      active_mu = self.mu_hist[self.update_flags]
      mean_mu = float(np.mean(active_mu))
      max_mu = float(np.max(active_mu))
      min_mu = float(np.min(active_mu))
    else:
      mean_mu = 0.0
      max_mu = 0.0
      min_mu = 0.0

    return dict(
      gamma_bar=float(self.gamma_bar),
      n_updates=n_updates,
      update_rate=update_rate,
      mean_mu_sm=mean_mu,
      max_mu_sm=max_mu,
      min_mu_sm=min_mu,
    )


# ============================================================
# SMAffineProjection
# ============================================================

@dataclass
class SMAffineProjection(BaseLMS):
  """
  Simplified Set-Membership Affine Projection.

  P = projection order.
  This implementation uses P+1 reused vectors.

  Simplified SM-AP RHS:
    [mu_sm e(k), 0, ..., 0]^T
  """

  P: int = 3
  gamma_bar: float = 0.0
  delta: float = 1e-3
  eps: float = 1e-12
  mu: Optional[float] = None

  update_flags: Array = field(init=False, repr=False)
  mu_sm_hist: Array = field(init=False, repr=False)
  post_e_hist: Array = field(init=False, repr=False)

  def __post_init__(self):
    super().__post_init__()

    if self.P < 0:
      raise ValueError(f"P must be non-negative, got {self.P}.")

    if self.gamma_bar < 0.0:
      raise ValueError(f"gamma_bar must be non-negative, got {self.gamma_bar}.")

    if self.delta < 0.0:
      raise ValueError(f"delta must be non-negative, got {self.delta}.")

    if self.eps <= 0.0:
      raise ValueError(f"eps must be positive, got {self.eps}.")

  def __call__(self, x: Array, d: Array, train: bool = True) -> Tuple[Array, Array, Array]:
    x = np.asarray(x, dtype=np.complex64)
    d = np.asarray(d, dtype=np.complex64)

    if x.ndim != 1:
      raise ValueError(f"x must be 1-D, got shape {x.shape}.")

    if d.ndim != 1:
      raise ValueError(f"d must be 1-D, got shape {d.shape}.")

    if x.shape[0] != d.shape[0]:
      raise ValueError(f"x and d must have same length, got {x.shape[0]} and {d.shape[0]}.")

    N = x.shape[0]
    M = self.n_coef
    Pp = self.P + 1

    if N < Pp:
      raise ValueError(f"Need at least P+1={Pp} samples, got N={N}.")

    x_pad = _prepad(x, M)
    w = self.w.copy()

    n_out = N - self.P

    y_hist = np.empty((n_out,), dtype=np.complex64)
    e_hist = np.empty((n_out,), dtype=np.complex64)
    w_hist = np.empty((n_out + 1, M), dtype=np.complex64)

    update_flags = np.zeros((n_out,), dtype=bool)
    mu_sm_hist = np.zeros((n_out,), dtype=np.float32)
    post_e_hist = np.empty((n_out, Pp), dtype=np.complex64)

    eye = np.eye(Pp, dtype=np.complex64)

    w_hist[0] = w

    out_idx = 0

    for k in range(self.P, N):
      cols = []

      for i in range(Pp):
        kk = k - i
        reg = x_pad[kk:kk + M][::-1].astype(np.complex64)
        cols.append(reg)

      Xap = np.stack(cols, axis=1)
      d_ap = np.array([d[k - i] for i in range(Pp)], dtype=np.complex64)

      y_ap = Xap.T @ np.conj(w)
      e_ap = d_ap - y_ap

      e0 = e_ap[0]
      abs_e0 = float(np.abs(e0))

      if train and abs_e0 > self.gamma_bar:
        mu_sm = 1.0 - self.gamma_bar / max(abs_e0, self.eps)

        rhs = np.zeros((Pp,), dtype=np.complex64)
        rhs[0] = np.complex64(mu_sm) * e0

        R = Xap.conj().T @ Xap
        R_reg = R + self.delta * eye

        try:
          g = np.linalg.solve(R_reg, np.conj(rhs))
        except np.linalg.LinAlgError:
          g = (np.linalg.pinv(R_reg) @ np.conj(rhs)).astype(np.complex64)

        upd = (Xap @ g).astype(np.complex64)
        w = (w + upd).astype(np.complex64)

        update_flags[out_idx] = True
        mu_sm_hist[out_idx] = np.float32(mu_sm)

      y_post_ap = Xap.T @ np.conj(w)
      post_e_ap = d_ap - y_post_ap

      y_hist[out_idx] = y_ap[0]
      e_hist[out_idx] = e_ap[0]
      post_e_hist[out_idx] = post_e_ap
      w_hist[out_idx + 1] = w

      out_idx += 1

    if train:
      self.w = w

    self.update_flags = update_flags
    self.mu_sm_hist = mu_sm_hist
    self.post_e_hist = post_e_hist

    return y_hist, e_hist, w_hist

  def getUpdateRate(self) -> float:
    if not hasattr(self, "update_flags") or self.update_flags.size == 0:
      return 0.0
    return float(np.mean(self.update_flags))

  def updateRate(self) -> float:
    return self.getUpdateRate()

  def analyzeUpdates(self) -> Dict[str, Any]:
    if not hasattr(self, "update_flags"):
      return dict(
        gamma_bar=float(self.gamma_bar),
        n_updates=0,
        update_rate=0.0,
        mean_mu_sm=0.0,
        min_mu_sm=0.0,
        max_mu_sm=0.0,
      )

    n_updates = int(np.sum(self.update_flags))
    update_rate = float(np.mean(self.update_flags)) if self.update_flags.size else 0.0

    if n_updates > 0:
      active_mu = self.mu_sm_hist[self.update_flags]
      mean_mu = float(np.mean(active_mu))
      min_mu = float(np.min(active_mu))
      max_mu = float(np.max(active_mu))
    else:
      mean_mu = 0.0
      min_mu = 0.0
      max_mu = 0.0

    return dict(
      gamma_bar=float(self.gamma_bar),
      projection_order=int(self.P),
      projection_dimension=int(self.P + 1),
      delta=float(self.delta),
      n_updates=n_updates,
      update_rate=update_rate,
      mean_mu_sm=mean_mu,
      min_mu_sm=min_mu,
      max_mu_sm=max_mu,
    )

  def printUpdateAnalysis(self) -> None:
    a = self.analyzeUpdates()

    print("\n" + "=" * 60)
    print("SIMPLIFIED SM-AP UPDATE ANALYSIS")
    print("=" * 60)
    print(f"Projection order P:        {a['projection_order']}")
    print(f"Projection dimension P+1:  {a['projection_dimension']}")
    print(f"gamma_bar:                 {a['gamma_bar']:.6g}")
    print(f"delta:                     {a['delta']:.6g}")
    print(f"Number of updates:         {a['n_updates']}")
    print(f"Update rate:               {a['update_rate']:.4f}")
    print(f"Mean mu_sm:                {a['mean_mu_sm']:.6f}")
    print(f"Min mu_sm:                 {a['min_mu_sm']:.6f}")
    print(f"Max mu_sm:                 {a['max_mu_sm']:.6f}")
    print("=" * 60)


# ============================================================
# SMBNLMS
# ============================================================

@dataclass
class SMBNLMS(BaseLMS):
  """
  Set-Membership Binormalized LMS.

  algorithm = 1:
    Two-stage SM-BNLMS:
      1. SM-NLMS update for H(k)
      2. If H(k-1) is violated, correct along x(k-1) orthogonal to x(k)

  algorithm = 2:
    One-shot SM-BNLMS:
      update direction preserves previous a posteriori error.
  """

  gamma_bar: float = 0.0
  eps: float = 1e-12
  delta: float = 1e-2
  algorithm: int = 2
  init_coef: Optional[Array] = None
  mu: Optional[float] = None

  update_flags: Array = field(init=False, repr=False)
  correction_flags: Array = field(init=False, repr=False)
  singular_flags: Array = field(init=False, repr=False)
  mu_hist: Array = field(init=False, repr=False)
  mu2_hist: Array = field(init=False, repr=False)

  def __post_init__(self):
    super().__post_init__()
    if self.mu is None:
      self.mu = 1.0
    if self.gamma_bar < 0.0:
      raise ValueError(f"gamma_bar must be non-negative, got {self.gamma_bar}.")

    if self.eps <= 0.0:
      raise ValueError(f"eps must be positive, got {self.eps}.")

    if self.delta < 0.0:
      raise ValueError(f"delta must be non-negative, got {self.delta}.")

    if self.algorithm not in (1, 2):
      raise ValueError(f"algorithm must be 1 or 2, got {self.algorithm}.")

  def __call__(self, x: Array, d: Array, train: bool = True) -> Tuple[Array, Array, Array]:
    x = np.asarray(x, dtype=np.complex64)
    d = np.asarray(d, dtype=np.complex64)

    if x.ndim != 1:
      raise ValueError(f"x must be 1-D, got shape {x.shape}.")

    if d.ndim != 1:
      raise ValueError(f"d must be 1-D, got shape {d.shape}.")

    if x.shape[0] != d.shape[0]:
      raise ValueError(f"x and d must have same length, got {x.shape[0]} and {d.shape[0]}.")

    N = x.shape[0]
    M = self.n_coef

    x_pad = _prepad(x, M)
    w = self.w.copy()

    y_hist = np.empty((N,), dtype=np.complex64)
    e_hist = np.empty((N,), dtype=np.complex64)
    w_hist = np.empty((N + 1, M), dtype=np.complex64)

    update_flags = np.zeros((N,), dtype=bool)
    correction_flags = np.zeros((N,), dtype=bool)
    singular_flags = np.zeros((N,), dtype=bool)

    mu_hist = np.zeros((N,), dtype=np.float32)
    mu2_hist = np.zeros((N,), dtype=np.float32)

    w_hist[0] = w

    for k in range(N):
      x_cur = x_pad[k:k + M][::-1].astype(np.complex64)

      y = np.vdot(w, x_cur)
      e = d[k] - y
      abs_e = float(np.abs(e))

      if train and abs_e > self.gamma_bar:
        if self.algorithm == 1:
          w, did_update, did_correct, is_singular, mu1, mu2 = self.updateAlg1(
            w=w,
            x_pad=x_pad,
            d=d,
            k=k,
            x_cur=x_cur,
            e=e,
          )
        else:
          w, did_update, did_correct, is_singular, mu1, mu2 = self.updateAlg2(
            w=w,
            x_pad=x_pad,
            k=k,
            x_cur=x_cur,
            e=e,
          )

        update_flags[k] = did_update
        correction_flags[k] = did_correct
        singular_flags[k] = is_singular
        mu_hist[k] = np.float32(mu1)
        mu2_hist[k] = np.float32(mu2)

      y_hist[k] = y
      e_hist[k] = e
      w_hist[k + 1] = w

    if train:
      self.w = w

    self.update_flags = update_flags
    self.correction_flags = correction_flags
    self.singular_flags = singular_flags
    self.mu_hist = mu_hist
    self.mu2_hist = mu2_hist

    return y_hist, e_hist, w_hist

  def updateAlg1(self, w, x_pad, d, k, x_cur, e):
    abs_e = float(np.abs(e))
    norm_cur = float(np.vdot(x_cur, x_cur).real)

    if norm_cur <= self.eps:
      return w, False, False, True, 0.0, 0.0

    mu1 = self.mu * (1.0 - self.gamma_bar / max(abs_e, self.eps))

    w_hat = (w + mu1 * np.conj(e) * x_cur / (norm_cur + self.delta)).astype(np.complex64)

    did_update = True
    did_correct = False
    is_singular = False
    mu2 = 0.0

    if k == 0:
      return w_hat, did_update, did_correct, is_singular, mu1, mu2

    M = self.n_coef
    x_prev = x_pad[k - 1:k - 1 + M][::-1].astype(np.complex64)

    eps_prev = d[k - 1] - np.vdot(w_hat, x_prev)
    abs_eps_prev = float(np.abs(eps_prev))

    if abs_eps_prev <= self.gamma_bar:
      return w_hat, did_update, did_correct, is_singular, mu1, mu2

    c_cur_prev = np.vdot(x_cur, x_prev)
    x_prev_perp = x_prev - (c_cur_prev / max(norm_cur, self.eps)) * x_cur

    norm_perp = float(np.vdot(x_prev_perp, x_prev_perp).real)

    if norm_perp <= self.eps:
      return w_hat, did_update, did_correct, True, mu1, mu2

    mu2 = self.mu * (1.0 - self.gamma_bar / max(abs_eps_prev, self.eps))

    w_new = (w_hat + mu2 * np.conj(eps_prev) * x_prev_perp / (norm_perp + self.delta)).astype(np.complex64)
    did_correct = True

    return w_new, did_update, did_correct, is_singular, mu1, mu2

  def updateAlg2(self, w, x_pad, k, x_cur, e):
    abs_e = float(np.abs(e))
    
    mu1 = self.mu * (1.0 - self.gamma_bar / max(abs_e, self.eps))

    if k == 0:
      return self.updateSmnlmsFallback(
        w=w,
        x_cur=x_cur,
        e=e,
        mu1=mu1,
      )

    M = self.n_coef
    x_prev = x_pad[k - 1:k - 1 + M][::-1].astype(np.complex64)

    A = float(np.vdot(x_cur, x_cur).real)
    B = float(np.vdot(x_prev, x_prev).real)
    c_cur_prev = np.vdot(x_cur, x_prev)

    D = A * B - float(np.abs(c_cur_prev) ** 2)

    if A <= self.eps or B <= self.eps or D <= self.eps:
      return self.updateSmnlmsFallback(
        w=w,
        x_cur=x_cur,
        e=e,
        mu1=mu1,
        is_singular=True,
      )

    q = (B * x_cur - np.conj(c_cur_prev) * x_prev) / (D + self.delta)
    w_new = (w + mu1 * np.conj(e) * q).astype(np.complex64)

    return w_new, True, False, False, mu1, 0.0

  def updateSmnlmsFallback(
    self,
    w,
    x_cur,
    e,
    mu1,
    did_correct: bool = False,
    is_singular: bool = False,
  ):
    """
    Fallback to ordinary SM-NLMS when the SM-BNLMS update is singular.

    Complex convention:
      y(k) = w^H x(k)
      e(k) = d(k) - y(k)

    SM-NLMS fallback:
      w(k+1) = w(k) + mu1 * e*(k) x(k) / ||x(k)||^2

    If self.delta > 0, the boundary equality becomes approximate.
    """

    norm_cur = float(np.vdot(x_cur, x_cur).real)

    if norm_cur <= self.eps:
      return w, False, did_correct, True, 0.0, 0.0

    denom = norm_cur + self.delta

    w_new = (w + mu1 * np.conj(e) * x_cur / denom).astype(np.complex64)

    return w_new, True, did_correct, is_singular, mu1, 0.0

  def getUpdateRate(self) -> float:
    if not hasattr(self, "update_flags") or self.update_flags.size == 0:
      return 0.0
    return float(np.mean(self.update_flags))

  def updateRate(self) -> float:
    return self.getUpdateRate()

  def getCorrectionRate(self) -> float:
    if not hasattr(self, "correction_flags") or self.correction_flags.size == 0:
      return 0.0
    return float(np.mean(self.correction_flags))

  def getSingularRate(self) -> float:
    if not hasattr(self, "singular_flags") or self.singular_flags.size == 0:
      return 0.0
    return float(np.mean(self.singular_flags))

  def analyzeUpdates(self) -> Dict[str, Any]:
    if not hasattr(self, "update_flags"):
      return dict(
        gamma_bar=float(self.gamma_bar),
        algorithm=int(self.algorithm),
        n_updates=0,
        update_rate=0.0,
        correction_rate=0.0,
        singular_rate=0.0,
        mean_mu_sm=0.0,
        min_mu_sm=0.0,
        max_mu_sm=0.0,
        mean_mu2_sm=0.0,
      )

    n_updates = int(np.sum(self.update_flags))
    n_corrections = int(np.sum(self.correction_flags))

    update_rate = float(np.mean(self.update_flags)) if self.update_flags.size else 0.0
    correction_rate = float(np.mean(self.correction_flags)) if self.correction_flags.size else 0.0
    singular_rate = float(np.mean(self.singular_flags)) if self.singular_flags.size else 0.0

    if n_updates > 0:
      active_mu = self.mu_hist[self.update_flags]
      mean_mu = float(np.mean(active_mu))
      min_mu = float(np.min(active_mu))
      max_mu = float(np.max(active_mu))
    else:
      mean_mu = 0.0
      min_mu = 0.0
      max_mu = 0.0

    if n_corrections > 0:
      active_mu2 = self.mu2_hist[self.correction_flags]
      mean_mu2 = float(np.mean(active_mu2))
    else:
      mean_mu2 = 0.0

    return dict(
      gamma_bar=float(self.gamma_bar),
      algorithm=int(self.algorithm),
      delta=float(self.delta),
      n_updates=n_updates,
      n_corrections=n_corrections,
      update_rate=update_rate,
      correction_rate=correction_rate,
      singular_rate=singular_rate,
      mean_mu_sm=mean_mu,
      min_mu_sm=min_mu,
      max_mu_sm=max_mu,
      mean_mu2_sm=mean_mu2,
    )

  def printUpdateAnalysis(self) -> None:
    a = self.analyzeUpdates()

    print("\n" + "=" * 60)
    print("SM-BNLMS UPDATE ANALYSIS")
    print("=" * 60)
    print(f"Algorithm:                 {a['algorithm']}")
    print(f"gamma_bar:                 {a['gamma_bar']:.6g}")
    print(f"delta:                     {a['delta']:.6g}")
    print(f"Number of updates:         {a['n_updates']}")
    print(f"Number of corrections:     {a['n_corrections']}")
    print(f"Update rate:               {a['update_rate']:.4f}")
    print(f"Correction rate:           {a['correction_rate']:.4f}")
    print(f"Singular rate:             {a['singular_rate']:.4f}")
    print(f"Mean mu_sm:                {a['mean_mu_sm']:.6f}")
    print(f"Min mu_sm:                 {a['min_mu_sm']:.6f}")
    print(f"Max mu_sm:                 {a['max_mu_sm']:.6f}")
    print(f"Mean mu2_sm:               {a['mean_mu2_sm']:.6f}")
    print("=" * 60)


@dataclass
class SMPartialUpdateAffineProjection(BaseLMS):
  """
  Faithful Diniz-style Simplified SM-PUAP.

  This follows Simp_SM_PUAP / Algorithm 6.6 style.

  Textbook update:

    w(k+1) = w(k)
           + C(k) Xap(k)
             [Xap^H(k) C(k) Xap(k) + gamma I]^{-1}
             mu(k) e*(k) u1

  where:

    mu(k) = 1 - gamma_bar / |e(k)|, if |e(k)| > gamma_bar
          = 0, otherwise

  P:
    memory length L in the textbook.

  max_update:
    Mbar. Used when selector_mode="topEnergy".

  selector_mode:
    "topEnergy":
      Section 6.9.2 style. Select Mbar rows of Xap with largest row energy.

    "random":
      MATLAB demo style. Random 0/1 selector at each iteration.

    "external":
      Use up_selector[:, k] exactly.

  up_selector:
    Optional selector matrix with shape (n_coef, n_iterations).
    Each column is the 0/1 coefficient update selector C(k).
  """

  P: int = 3
  gamma_bar: float = 0.0
  max_update: int = 5
  delta: float = 1e-3
  eps: float = 1e-12
  mu: Optional[float] = None

  selector_mode: str = "topEnergy"
  up_selector: Optional[Array] = None

  update_flags: Array = field(init=False, repr=False)
  selected_mask_hist: Array = field(init=False, repr=False)
  selected_count_hist: Array = field(init=False, repr=False)
  mu_sm_hist: Array = field(init=False, repr=False)
  post_e_hist: Array = field(init=False, repr=False)

  def __post_init__(self):
    super().__post_init__()

    if self.P < 0:
      raise ValueError(f"P must be non-negative, got {self.P}.")

    if self.gamma_bar < 0.0:
      raise ValueError(f"gamma_bar must be non-negative, got {self.gamma_bar}.")

    if self.max_update <= 0:
      raise ValueError(f"max_update must be positive, got {self.max_update}.")

    if self.delta < 0.0:
      raise ValueError(f"delta must be non-negative, got {self.delta}.")

    if self.eps <= 0.0:
      raise ValueError(f"eps must be positive, got {self.eps}.")

    if self.selector_mode not in ("topEnergy", "random", "external"):
      raise ValueError(
        "selector_mode must be 'topEnergy', 'random', or 'external', "
        f"got {self.selector_mode}."
      )

  def __call__(self, x: Array, d: Array, train: bool = True) -> Tuple[Array, Array, Array]:
    x = np.asarray(x, dtype=np.complex128)
    d = np.asarray(d, dtype=np.complex128)

    if x.ndim != 1:
      raise ValueError(f"x must be 1-D, got shape {x.shape}.")

    if d.ndim != 1:
      raise ValueError(f"d must be 1-D, got shape {d.shape}.")

    if x.shape[0] != d.shape[0]:
      raise ValueError(f"x and d must have same length, got {x.shape[0]} and {d.shape[0]}.")

    K = x.shape[0]
    N = self.n_coef
    L = self.P
    Pp = L + 1

    if self.max_update > N:
      raise ValueError(f"max_update={self.max_update} cannot exceed n_coef={N}.")

    if self.selector_mode == "external":
      if self.up_selector is None:
        raise ValueError("up_selector must be provided when selector_mode='external'.")

      self.up_selector = np.asarray(self.up_selector)

      if self.up_selector.shape != (N, K):
        raise ValueError(
          f"up_selector must have shape {(N, K)}, got {self.up_selector.shape}."
        )

    # Diniz/MATLAB-style regular prefixing.
    x_pref = np.concatenate([np.zeros(N - 1, dtype=np.complex128), x])

    d_pref = np.concatenate([np.zeros(L, dtype=np.complex128), d])

    # Internal state.
    w = self.w.astype(np.complex128).copy()
    Xap = np.zeros((N, Pp), dtype=np.complex128)
    u1 = np.zeros(Pp, dtype=np.complex128)
    u1[0] = 1.0

    eye = np.eye(Pp, dtype=np.complex128)

    y_hist = np.zeros(K, dtype=np.complex128)
    e_hist = np.zeros(K, dtype=np.complex128)
    w_hist = np.zeros((K + 1, N), dtype=np.complex128)

    update_flags = np.zeros(K, dtype=bool)
    selected_mask_hist = np.zeros((K, N), dtype=bool)
    selected_count_hist = np.zeros(K, dtype=np.int32)
    mu_sm_hist = np.zeros(K, dtype=np.float64)
    post_e_hist = np.zeros((K, Pp), dtype=np.complex128)

    w_hist[0] = w

    for k in range(K):
      # Shift AP matrix and insert newest regressor.
      Xap[:, 1:] = Xap[:, :-1]
      Xap[:, 0] = x_pref[k:k + N][::-1]

      # Diniz convention:
      # output_ap = Xap^H w
      # error_ap_conj = conj(d_ap) - output_ap
      d_ap = d_pref[k:k + Pp][::-1]
      y_ap = Xap.conj().T @ w
      e_ap_conj = np.conj(d_ap) - y_ap

      y_hist[k] = y_ap[0]
      e_hist[k] = e_ap_conj[0]

      e0 = e_ap_conj[0]
      abs_e0 = float(np.abs(e0))

      if train and abs_e0 > self.gamma_bar:
        update_flags[k] = True
        mu_sm = 1.0 - self.gamma_bar / max(abs_e0, self.eps)
      else:
        mu_sm = 0.0

      mu_sm_hist[k] = mu_sm

      # Build C(k) selector.
      mask = self.selectMask(Xap=Xap, k=k)

      selected_mask_hist[k] = mask.astype(bool)
      selected_count_hist[k] = int(np.sum(mask))

      if train and mu_sm > 0.0:
        Xsel = Xap * mask[:, None]

        R = Xap.conj().T @ Xsel
        R_reg = R + self.delta * eye

        rhs = mu_sm * e0 * u1

        # Diniz MATLAB uses inv(...). Solve is numerically cleaner but equivalent.
        try:
          g = np.linalg.solve(R_reg, rhs)
        except np.linalg.LinAlgError:
          g = np.linalg.pinv(R_reg) @ rhs

        w = w + Xsel @ g

      y_post_ap = Xap.conj().T @ w
      post_e_hist[k] = np.conj(d_ap) - y_post_ap

      w_hist[k + 1] = w

    if train:
      self.w = w.astype(np.complex64)

    self.update_flags = update_flags
    self.selected_mask_hist = selected_mask_hist
    self.selected_count_hist = selected_count_hist
    self.mu_sm_hist = mu_sm_hist
    self.post_e_hist = post_e_hist

    return (
      y_hist.astype(np.complex64),
      e_hist.astype(np.complex64),
      w_hist.astype(np.complex64),
    )

  def selectMask(self, Xap: Array, k: int) -> Array:
    """
    Build selector mask C(k).

    topEnergy:
      choose Mbar rows of Xap with largest row energy.

    random:
      random 0/1 selector, like MATLAB randint(N,K,[0 1]).

    external:
      use self.up_selector[:, k].
    """
    N = Xap.shape[0]

    if self.selector_mode == "external":
      mask = np.asarray(self.up_selector[:, k], dtype=np.float64)
      return np.clip(mask, 0.0, 1.0)

    if self.selector_mode == "random":
      return np.random.randint(0, 2, size=N).astype(np.float64)

    # selector_mode == "topEnergy"
    row_energy = np.sum(np.abs(Xap) ** 2, axis=1)
    mbar = min(self.max_update, N)

    idx = np.argsort(row_energy)[-mbar:]

    mask = np.zeros(N, dtype=np.float64)
    mask[idx] = 1.0

    return mask

  def getUpdateRate(self) -> float:
    if not hasattr(self, "update_flags") or self.update_flags.size == 0:
      return 0.0
    return float(np.mean(self.update_flags))

  def updateRate(self) -> float:
    return self.getUpdateRate()

  def getAverageSelectedCount(self) -> float:
    if not hasattr(self, "selected_count_hist") or self.selected_count_hist.size == 0:
      return 0.0
    return float(np.mean(self.selected_count_hist))

  def analyzeUpdates(self) -> Dict[str, Any]:
    if not hasattr(self, "update_flags"):
      return dict(
        gamma_bar=float(self.gamma_bar),
        projection_order=int(self.P),
        projection_dimension=int(self.P + 1),
        max_update=int(self.max_update),
        selector_mode=self.selector_mode,
        n_updates=0,
        update_rate=0.0,
        average_selected_count=0.0,
        mean_mu_sm=0.0,
        min_mu_sm=0.0,
        max_mu_sm=0.0,
      )

    active_mu = self.mu_sm_hist[self.update_flags]

    if active_mu.size > 0:
      mean_mu = float(np.mean(active_mu))
      min_mu = float(np.min(active_mu))
      max_mu = float(np.max(active_mu))
    else:
      mean_mu = 0.0
      min_mu = 0.0
      max_mu = 0.0

    return dict(
      gamma_bar=float(self.gamma_bar),
      projection_order=int(self.P),
      projection_dimension=int(self.P + 1),
      max_update=int(self.max_update),
      selector_mode=self.selector_mode,
      delta=float(self.delta),
      n_updates=int(np.sum(self.update_flags)),
      update_rate=self.getUpdateRate(),
      average_selected_count=self.getAverageSelectedCount(),
      mean_mu_sm=mean_mu,
      min_mu_sm=min_mu,
      max_mu_sm=max_mu,
    )

  def printUpdateAnalysis(self) -> None:
    a = self.analyzeUpdates()

    print("\n" + "=" * 64)
    print("DINIZ-STYLE SM-PUAP UPDATE ANALYSIS")
    print("=" * 64)
    print(f"Projection order P/L:          {a['projection_order']}")
    print(f"Projection dimension L+1:      {a['projection_dimension']}")
    print(f"max_update Mbar:               {a['max_update']}")
    print(f"selector_mode:                 {a['selector_mode']}")
    print(f"gamma_bar:                     {a['gamma_bar']:.6g}")
    print(f"delta/gamma:                   {a['delta']:.6g}")
    print(f"Number of updates:             {a['n_updates']}")
    print(f"Update rate:                   {a['update_rate']:.4f}")
    print(f"Average selected count:        {a['average_selected_count']:.3f}")
    print(f"Mean mu_sm:                    {a['mean_mu_sm']:.6f}")
    print(f"Min mu_sm:                     {a['min_mu_sm']:.6f}")
    print(f"Max mu_sm:                     {a['max_mu_sm']:.6f}")
    print("=" * 64)

# -----------------------------------------------------------------------------
# Diniz Chapter 7 - Lattice RLS
# -----------------------------------------------------------------------------

@dataclass
class BaseLatticeRLS(BaseLMS):
  """Shared base for Diniz Algorithm 7.1 and 7.2."""

  lam: float = 0.99
  epsilon: float = 1e-3
  eps: float = 1e-12
  dtype: object = np.float64

  def __post_init__(self):
    self.mu = None
    super().__post_init__()

    if not (0.0 < self.lam <= 1.0):
      raise ValueError(f"lam must satisfy 0 < lam <= 1, got {self.lam}.")
    if self.epsilon <= 0.0:
      raise ValueError(f"epsilon must be > 0, got {self.epsilon}.")
    if self.eps <= 0.0:
      raise ValueError(f"eps must be > 0, got {self.eps}.")

    self.resetState()

  def checkRealInput(self, x: Array, d: Array) -> Tuple[Array, Array]:
    x = np.asarray(x)
    d = np.asarray(d)

    if np.iscomplexobj(x) and np.any(np.abs(np.imag(x)) > self.eps):
      raise ValueError("Diniz Algorithm 7.1/7.2 here implements the real-valued textbook form.")
    if np.iscomplexobj(d) and np.any(np.abs(np.imag(d)) > self.eps):
      raise ValueError("Diniz Algorithm 7.1/7.2 here implements the real-valued textbook form.")

    x = np.asarray(np.real(x), dtype=self.dtype)
    d = np.asarray(np.real(d), dtype=self.dtype)

    if x.ndim != 1 or d.ndim != 1:
      raise ValueError(f"x and d must be 1-D, got {x.shape} and {d.shape}.")
    if len(x) != len(d):
      raise ValueError(f"x and d must have same length, got {len(x)} and {len(d)}.")

    return x, d

  def posGuard(self, x: float) -> float:
    return max(float(x), self.eps)

  def sqrtOneMinusSquare(self, x: float) -> float:
    return math.sqrt(max(1.0 - float(x) * float(x), self.eps))


@dataclass
class LatticeRLS(BaseLatticeRLS):
  """
  Diniz Algorithm 7.1 - Lattice RLS based on a posteriori errors.

  self.w stores the lattice feedforward coefficients:
      self.w[i] = v_i(k)

  These are lattice-basis coefficients, not direct-form transversal FIR taps.
  """

  delta: Array = field(init=False, repr=False)
  delta_d: Array = field(init=False, repr=False)
  xi_b: Array = field(init=False, repr=False)
  xi_f: Array = field(init=False, repr=False)
  gamma: Array = field(init=False, repr=False)
  eb_prev: Array = field(init=False, repr=False)

  def resetState(self) -> None:
    M = self.n_coef

    self.delta = np.zeros(M, dtype=self.dtype)
    self.delta_d = np.zeros(M, dtype=self.dtype)

    self.xi_b = np.full(M + 1, self.epsilon, dtype=self.dtype)
    self.xi_f = np.full(M + 1, self.epsilon, dtype=self.dtype)

    self.gamma = np.ones(M + 1, dtype=self.dtype)
    self.eb_prev = np.zeros(M + 1, dtype=self.dtype)

    self.w = np.zeros(M, dtype=np.complex64)

  def step(self, xk: float, dk: float) -> Dict[str, Array]:
    M = self.n_coef

    delta_prev = self.delta.copy()
    delta_d_prev = self.delta_d.copy()
    xi_b_prev = self.xi_b.copy()
    xi_f_prev = self.xi_f.copy()
    gamma_prev = self.gamma.copy()
    eb_prev = self.eb_prev.copy()

    ef = np.zeros(M + 1, dtype=self.dtype)
    eb = np.zeros(M + 1, dtype=self.dtype)
    error = np.zeros(M + 1, dtype=self.dtype)
    gamma = np.ones(M + 1, dtype=self.dtype)
    xi_b = np.zeros(M + 1, dtype=self.dtype)
    xi_f = np.zeros(M + 1, dtype=self.dtype)

    delta = np.empty(M, dtype=self.dtype)
    delta_d = np.empty(M, dtype=self.dtype)
    kappa_b = np.empty(M, dtype=self.dtype)
    kappa_f = np.empty(M, dtype=self.dtype)
    v = np.empty(M, dtype=self.dtype)

    # Algorithm 7.1 order-0 initialization, Eqs. (7.35)-(7.36).
    gamma[0] = 1.0
    eb[0] = ef[0] = xk
    xi_b[0] = xi_f[0] = xk * xk + self.lam * xi_f_prev[0]
    error[0] = dk

    for i in range(M):
      # Eq. (7.51): time update of prediction cross-correlation.
      delta[i] = self.lam * delta_prev[i] + eb_prev[i] * ef[i] / self.posGuard(gamma_prev[i])

      # Eq. (7.60): order update of conversion factor.
      gamma[i + 1] = gamma[i] - eb[i] * eb[i] / self.posGuard(xi_b[i])

      # Reflection coefficients.
      kappa_b[i] = delta[i] / self.posGuard(xi_f[i])
      kappa_f[i] = delta[i] / self.posGuard(xi_b_prev[i])

      # Eqs. (7.34), (7.33): backward/forward a posteriori errors.
      eb[i + 1] = eb_prev[i] - kappa_b[i] * ef[i]
      ef[i + 1] = ef[i] - kappa_f[i] * eb_prev[i]

      # Eqs. (7.27), (7.31): minimum LS error order updates.
      xi_b[i + 1] = xi_b_prev[i] - delta[i] * kappa_b[i]
      xi_f[i + 1] = xi_f[i] - delta[i] * kappa_f[i]

      # Eqs. (7.64), (7.67), (7.68): joint-process feedforward section.
      delta_d[i] = self.lam * delta_d_prev[i] + eb[i] * error[i] / self.posGuard(gamma[i])
      v[i] = delta_d[i] / self.posGuard(xi_b[i])
      error[i + 1] = error[i] - v[i] * eb[i]

    self.delta = delta
    self.delta_d = delta_d
    self.xi_b = xi_b
    self.xi_f = xi_f
    self.gamma = gamma
    self.eb_prev = eb
    self.w = v.astype(np.complex64)

    return {
      "y": np.asarray(dk - error[-1], dtype=self.dtype)[()],
      "e": np.asarray(error[-1], dtype=self.dtype)[()],
      "e_order": error,
      "ef": ef,
      "eb": eb,
      "delta": delta,
      "delta_d": delta_d,
      "gamma": gamma,
      "xi_f": xi_f,
      "xi_b": xi_b,
      "kappa_f": kappa_f,
      "kappa_b": kappa_b,
      "v": v,
    }

  def __call__(self, x: Array, d: Array, train: bool = True) -> Tuple[Array, Array, Array]:
    x, d = self.checkRealInput(x, d)

    N = len(x)
    M = self.n_coef

    saved_state = None
    if not train:
      saved_state = (
        self.delta.copy(), self.delta_d.copy(), self.xi_b.copy(), self.xi_f.copy(), self.gamma.copy(),
        self.eb_prev.copy(), self.w.copy(),
      )

    y_hist = np.empty(N, dtype=self.dtype)
    e_hist = np.empty(N, dtype=self.dtype)
    w_hist = np.empty((N + 1, M), dtype=np.complex64)
    w_hist[0] = self.w

    for k in range(N):
      result = self.step(x[k], d[k])
      y_hist[k] = result["y"]
      e_hist[k] = result["e"]
      w_hist[k + 1] = result["v"]

    if not train:
      self.delta, self.delta_d, self.xi_b, self.xi_f, self.gamma, self.eb_prev, self.w = saved_state

    return y_hist, e_hist, w_hist


@dataclass
class NormalizedLatticeRLS(BaseLatticeRLS):
  """
  Diniz Algorithm 7.2 - Normalized lattice RLS based on a posteriori errors.

  The algorithm directly produces normalized errors and normalized correlations:
      delta_bar, delta_d_bar, ef_bar, eb_bar, e_bar.

  There is no physical direct-form FIR coefficient vector in Algorithm 7.2, so inherited self.w is not used.
  """

  delta_bar: Array = field(init=False, repr=False)
  delta_d_bar: Array = field(init=False, repr=False)
  eb_bar_prev: Array = field(init=False, repr=False)

  sigma_x2: float = field(init=False)
  sigma_d2: float = field(init=False)

  def resetState(self) -> None:
    M = self.n_coef

    self.delta_bar = np.zeros(M, dtype=self.dtype)
    self.delta_d_bar = np.zeros(M, dtype=self.dtype)
    self.eb_bar_prev = np.zeros(M + 1, dtype=self.dtype)

    self.sigma_x2 = float(self.epsilon)
    self.sigma_d2 = float(self.epsilon)

    self.w = np.zeros(M, dtype=np.complex64)

  def step(self, xk: float, dk: float) -> Dict[str, Array]:
    M = self.n_coef

    delta_prev = self.delta_bar.copy()
    delta_d_prev = self.delta_d_bar.copy()
    eb_prev = self.eb_bar_prev.copy()

    # Algorithm 7.2 input/reference power recursions.
    self.sigma_x2 = self.lam * self.sigma_x2 + xk * xk
    self.sigma_d2 = self.lam * self.sigma_d2 + dk * dk

    sigma_x = math.sqrt(self.posGuard(self.sigma_x2))
    sigma_d = math.sqrt(self.posGuard(self.sigma_d2))

    ef = np.zeros(M + 1, dtype=self.dtype)
    eb = np.zeros(M + 1, dtype=self.dtype)
    error = np.zeros(M + 1, dtype=self.dtype)

    delta = np.empty(M, dtype=self.dtype)
    delta_d = np.empty(M, dtype=self.dtype)
    eta_f = np.empty(M, dtype=self.dtype)
    eta_b = np.empty(M, dtype=self.dtype)
    eta_d = np.empty(M, dtype=self.dtype)

    # Order-0 normalized innovations.
    eb[0] = ef[0] = xk / sigma_x
    error[0] = dk / sigma_d

    for i in range(M):
      sqrt_eb_prev = self.sqrtOneMinusSquare(eb_prev[i])
      sqrt_ef = self.sqrtOneMinusSquare(ef[i])

      # Eq. (7.81): normalized forward/backward correlation.
      delta[i] = delta_prev[i] * sqrt_eb_prev * sqrt_ef + eb_prev[i] * ef[i]

      sqrt_delta = self.sqrtOneMinusSquare(delta[i])

      # Eqs. (7.91)-(7.92).
      eta_f[i] = 1.0 / (sqrt_delta * sqrt_eb_prev)
      eta_b[i] = 1.0 / (sqrt_delta * sqrt_ef)

      # Eqs. (7.84), (7.83).
      eb[i + 1] = eta_b[i] * (eb_prev[i] - delta[i] * ef[i])
      ef[i + 1] = eta_f[i] * (ef[i] - delta[i] * eb_prev[i])

      sqrt_eb = self.sqrtOneMinusSquare(eb[i])
      sqrt_error = self.sqrtOneMinusSquare(error[i])

      # Eq. (7.89): normalized joint-process correlation.
      delta_d[i] = delta_d_prev[i] * sqrt_eb * sqrt_error + error[i] * eb[i]

      # Eq. (7.93) and Eq. (7.88).
      eta_d[i] = 1.0 / (sqrt_eb * self.sqrtOneMinusSquare(delta_d[i]))
      error[i + 1] = eta_d[i] * (error[i] - delta_d[i] * eb[i])

    self.delta_bar = delta
    self.delta_d_bar = delta_d
    self.eb_bar_prev = eb

    return {
      "e_bar": np.asarray(error[-1], dtype=self.dtype)[()],
      "e_bar_order": error,
      "ef_bar": ef,
      "eb_bar": eb,
      "delta_bar": delta,
      "delta_d_bar": delta_d,
      "eta_f": eta_f,
      "eta_b": eta_b,
      "eta_d": eta_d,
      "sigma_x": np.asarray(sigma_x, dtype=self.dtype)[()],
      "sigma_d": np.asarray(sigma_d, dtype=self.dtype)[()],
    }

  def __call__(self, x: Array, d: Array, train: bool = True) -> Dict[str, Array]:
    x, d = self.checkRealInput(x, d)

    N = len(x)
    M = self.n_coef

    saved_state = None
    if not train:
      saved_state = (
        self.delta_bar.copy(), self.delta_d_bar.copy(), self.eb_bar_prev.copy(), self.sigma_x2, self.sigma_d2,
      )

    e_bar_hist = np.empty(N, dtype=self.dtype)
    ef_bar_hist = np.empty((N, M + 1), dtype=self.dtype)
    eb_bar_hist = np.empty((N, M + 1), dtype=self.dtype)
    delta_bar_hist = np.empty((N, M), dtype=self.dtype)
    delta_d_bar_hist = np.empty((N, M), dtype=self.dtype)
    eta_f_hist = np.empty((N, M), dtype=self.dtype)
    eta_b_hist = np.empty((N, M), dtype=self.dtype)
    eta_d_hist = np.empty((N, M), dtype=self.dtype)

    for k in range(N):
      result = self.step(x[k], d[k])

      e_bar_hist[k] = result["e_bar"]
      ef_bar_hist[k] = result["ef_bar"]
      eb_bar_hist[k] = result["eb_bar"]
      delta_bar_hist[k] = result["delta_bar"]
      delta_d_bar_hist[k] = result["delta_d_bar"]
      eta_f_hist[k] = result["eta_f"]
      eta_b_hist[k] = result["eta_b"]
      eta_d_hist[k] = result["eta_d"]

    if not train:
      self.delta_bar, self.delta_d_bar, self.eb_bar_prev, self.sigma_x2, self.sigma_d2 = saved_state

    return {
      "e_bar": e_bar_hist,
      "ef_bar_hist": ef_bar_hist,
      "eb_bar_hist": eb_bar_hist,
      "delta_bar_hist": delta_bar_hist,
      "delta_d_bar_hist": delta_d_bar_hist,
      "eta_f_hist": eta_f_hist,
      "eta_b_hist": eta_b_hist,
      "eta_d_hist": eta_d_hist,
    }

@dataclass
class ErrorFeedbackLatticeRLS(LatticeRLS):
  """
  Diniz Algorithm 7.3 - Error-feedback LRLS based on a posteriori errors.

  Differences from Algorithm 7.1:
    kappa_f(k,i) is recursively updated by Eq. (7.94).
    kappa_b(k,i) is recursively updated by Eq. (7.95).
    v_i(k)       is recursively updated by Eq. (7.96).

  self.w stores the feedforward lattice coefficients v_i(k).
  """

  kappa_f: Array = field(init=False, repr=False)
  kappa_b: Array = field(init=False, repr=False)
  xi_b_prev2: Array = field(init=False, repr=False)

  def resetState(self) -> None:
    # Reuse Algorithm 7.1 initialization for delta, xi_f, xi_b, gamma, eb_prev, and v.
    super().resetState()

    M = self.n_coef
    self.kappa_f = np.zeros(M, dtype=self.dtype)
    self.kappa_b = np.zeros(M, dtype=self.dtype)

    # Algorithm 7.3 additionally requires xi_b(-2,i) because Eq. (7.94) uses xi_b(k-2,i).
    self.xi_b_prev2 = np.full(M + 1, self.epsilon, dtype=self.dtype)

  def step(self, xk: float, dk: float) -> Dict[str, Array]:
    M = self.n_coef

    # Saved states correspond to time k-1, except xi_b_prev2 = xi_b(k-2,i).
    delta_prev = self.delta.copy()
    kappa_f_prev = self.kappa_f.copy()
    kappa_b_prev = self.kappa_b.copy()
    v_prev = np.asarray(self.w.real, dtype=self.dtype)

    gamma_prev = self.gamma.copy()
    eb_prev = self.eb_prev.copy()

    xi_f_prev = self.xi_f.copy()
    xi_b_prev = self.xi_b.copy()
    xi_b_prev2 = self.xi_b_prev2.copy()

    ef = np.zeros(M + 1, dtype=self.dtype)
    eb = np.zeros(M + 1, dtype=self.dtype)
    error = np.zeros(M + 1, dtype=self.dtype)
    gamma = np.ones(M + 1, dtype=self.dtype)

    xi_f = np.zeros(M + 1, dtype=self.dtype)
    xi_b = np.zeros(M + 1, dtype=self.dtype)

    delta = np.empty(M, dtype=self.dtype)
    kappa_f = np.empty(M, dtype=self.dtype)
    kappa_b = np.empty(M, dtype=self.dtype)
    v = np.empty(M, dtype=self.dtype)

    # Algorithm 7.3 order-0 initialization, Eqs. (7.35)-(7.36).
    gamma[0] = 1.0
    eb[0] = ef[0] = xk
    xi_f[0] = xi_b[0] = xk * xk + self.lam * xi_f_prev[0]
    error[0] = dk

    for i in range(M):
      gamma_prev_i = self.posGuard(gamma_prev[i])
      gamma_i = self.posGuard(gamma[i])

      xi_f_i = self.posGuard(xi_f[i])
      xi_f_prev_i = self.posGuard(xi_f_prev[i])

      xi_b_i = self.posGuard(xi_b[i])
      xi_b_prev_i = self.posGuard(xi_b_prev[i])
      xi_b_prev2_i = self.posGuard(xi_b_prev2[i])

      # Eq. (7.51): prediction cross-correlation time update.
      pred_corr = eb_prev[i] * ef[i] / gamma_prev_i
      delta[i] = self.lam * delta_prev[i] + pred_corr

      # Eq. (7.60): conversion-factor order update.
      gamma[i + 1] = gamma[i] - eb[i] * eb[i] / xi_b_i

      # Eq. (7.94): direct recursive forward reflection-coefficient update.
      kappa_f[i] = gamma_prev[i + 1] / gamma_prev_i * (
        kappa_f_prev[i] + pred_corr / (self.lam * xi_b_prev2_i)
      )

      # Eq. (7.95): direct recursive backward reflection-coefficient update.
      kappa_b[i] = gamma[i + 1] / gamma_prev_i * (
        kappa_b_prev[i] + pred_corr / (self.lam * xi_f_prev_i)
      )

      # Eqs. (7.34), (7.33): prediction-error order updates.
      eb[i + 1] = eb_prev[i] - kappa_b[i] * ef[i]
      ef[i + 1] = ef[i] - kappa_f[i] * eb_prev[i]

      # Eqs. (7.31), (7.27): minimum LS prediction-error updates.
      xi_f[i + 1] = xi_f[i] - delta[i] * delta[i] / xi_b_prev_i
      xi_b[i + 1] = xi_b_prev[i] - delta[i] * delta[i] / xi_f_i

      # Eq. (7.96): direct recursive feedforward-coefficient update.
      ff_corr = error[i] * eb[i] / gamma_i
      v[i] = gamma[i + 1] / gamma_i * (
        v_prev[i] + ff_corr / (self.lam * xi_b_prev_i)
      )

      # Eq. (7.68): joint-process a posteriori output-error update.
      error[i + 1] = error[i] - v[i] * eb[i]

    # Shift time states: k-2 <- k-1, k-1 <- k.
    self.xi_b_prev2 = xi_b_prev
    self.xi_b = xi_b
    self.xi_f = xi_f

    self.delta = delta
    self.gamma = gamma
    self.eb_prev = eb

    self.kappa_f = kappa_f
    self.kappa_b = kappa_b
    self.w = v.astype(np.complex64)

    return {
      "y": np.asarray(dk - error[-1], dtype=self.dtype)[()],
      "e": np.asarray(error[-1], dtype=self.dtype)[()],
      "e_order": error,
      "ef": ef,
      "eb": eb,
      "delta": delta,
      "gamma": gamma,
      "xi_f": xi_f,
      "xi_b": xi_b,
      "kappa_f": kappa_f,
      "kappa_b": kappa_b,
      "v": v,
    }

  def __call__(self, x: Array, d: Array, train: bool = True) -> Tuple[Array, Array, Array]:
    # LatticeRLS.__call__ already provides y/e/v histories. Only preserve the extra Algorithm 7.3 states for train=False.
    if train:
      return super().__call__(x, d, train=True)

    extra_state = (self.kappa_f.copy(), self.kappa_b.copy(), self.xi_b_prev2.copy())
    result = super().__call__(x, d, train=False)
    self.kappa_f, self.kappa_b, self.xi_b_prev2 = extra_state

    return result

@dataclass
class APrioriLatticeRLS(LatticeRLS):
  """
  Diniz Algorithm 7.4 - LRLS based on a priori errors.

  Main difference from Algorithm 7.1:
    - Current prediction errors use kappa_f(k-1,i), kappa_b(k-1,i).
    - Current joint-process error uses v_i(k-1).
    - Current a priori errors are then used to update delta, delta_D, kappa, and v.

  self.w stores the current feedforward lattice coefficients v_i(k).
  """

  kappa_f: Array = field(init=False, repr=False)
  kappa_b: Array = field(init=False, repr=False)

  def resetState(self) -> None:
    super().resetState()

    M = self.n_coef

    # Algorithm 7.4:
    #   kappa_f(-1,i) = kappa_b(-1,i) = 0
    self.kappa_f = np.zeros(M, dtype=self.dtype)
    self.kappa_b = np.zeros(M, dtype=self.dtype)

    # Eq. (7.103) with delta_D(-1,i)=0 implies v_i(-1)=0.
    self.w = np.zeros(M, dtype=np.complex64)

  def step(self, xk: float, dk: float) -> Dict[str, Array]:
    M = self.n_coef

    # States available at the beginning of time k.
    delta_prev = self.delta.copy()
    delta_d_prev = self.delta_d.copy()

    xi_b_prev = self.xi_b.copy()
    xi_f_prev = self.xi_f.copy()
    gamma_prev = self.gamma.copy()

    eb_prev = self.eb_prev.copy()

    kappa_f_prev = self.kappa_f.copy()
    kappa_b_prev = self.kappa_b.copy()
    v_prev = np.asarray(self.w.real, dtype=self.dtype)

    # Current-time order variables, i = 0 ... N+1.
    ef = np.zeros(M + 1, dtype=self.dtype)
    eb = np.zeros(M + 1, dtype=self.dtype)
    error = np.zeros(M + 1, dtype=self.dtype)

    gamma = np.ones(M + 1, dtype=self.dtype)
    xi_b = np.zeros(M + 1, dtype=self.dtype)
    xi_f = np.zeros(M + 1, dtype=self.dtype)

    # Current-time per-stage variables, i = 0 ... N.
    delta = np.empty(M, dtype=self.dtype)
    delta_d = np.empty(M, dtype=self.dtype)
    kappa_f = np.empty(M, dtype=self.dtype)
    kappa_b = np.empty(M, dtype=self.dtype)
    v = np.empty(M, dtype=self.dtype)

    # -------------------------------------------------------------------------
    # Algorithm 7.4 order-0 initialization.
    # -------------------------------------------------------------------------

    gamma[0] = 1.0
    eb[0] = ef[0] = xk
    xi_b[0] = xi_f[0] = xk * xk + self.lam * xi_f_prev[0]
    error[0] = dk

    for i in range(M):
      gamma_prev_i = self.posGuard(gamma_prev[i])
      gamma_i = self.posGuard(gamma[i])

      xi_b_i = self.posGuard(xi_b[i])
      xi_b_prev_i = self.posGuard(xi_b_prev[i])

      xi_f_i = self.posGuard(xi_f[i])

      # -----------------------------------------------------------------------
      # Eq. (7.97)
      #
      # delta(k,i) = lambda delta(k-1,i)
      #            + gamma(k-1,i) e_b(k-1,i) e_f(k,i)
      # -----------------------------------------------------------------------

      delta[i] = (self.lam * delta_prev[i] + gamma_prev_i * eb_prev[i] * ef[i])

      # -----------------------------------------------------------------------
      # Eq. (7.100), shifted to current k.
      #
      # gamma(k,i+1) = gamma(k,i)
      #              - gamma^2(k,i) e_b^2(k,i) / xi_b(k,i)
      # -----------------------------------------------------------------------

      gamma[i + 1] = (gamma[i] - gamma_i * gamma_i * eb[i] * eb[i] / xi_b_i)

      # -----------------------------------------------------------------------
      # Eqs. (7.99), (7.98)
      #
      # Important: current a priori errors use PREVIOUS-TIME coefficients.
      # -----------------------------------------------------------------------

      eb[i + 1] = eb_prev[i] - kappa_b_prev[i] * ef[i]
      ef[i + 1] = ef[i] - kappa_f_prev[i] * eb_prev[i]

      # -----------------------------------------------------------------------
      # Current reflection coefficients.
      #
      # These are calculated AFTER the current a priori errors and are stored
      # for use at time k+1.
      # -----------------------------------------------------------------------

      kappa_f[i] = delta[i] / xi_b_prev_i
      kappa_b[i] = delta[i] / xi_f_i

      # -----------------------------------------------------------------------
      # Eqs. (7.31), (7.27)
      # -----------------------------------------------------------------------

      xi_f[i + 1] = (xi_f[i] - delta[i] * kappa_f[i])

      xi_b[i + 1] = (xi_b_prev[i] - delta[i] * kappa_b[i])

      # -----------------------------------------------------------------------
      # Feedforward filtering.
      #
      # Eq. (7.101):
      #   delta_D(k,i) = lambda delta_D(k-1,i)
      #                + gamma(k,i) e_b(k,i) e(k,i)
      # -----------------------------------------------------------------------

      delta_d[i] = (self.lam * delta_d_prev[i]+ gamma_i * eb[i] * error[i])

      # -----------------------------------------------------------------------
      # Eq. (7.102):
      #
      #   e(k,i+1) = e(k,i) - v_i(k-1) e_b(k,i)
      #
      # Important: use v_prev here, NOT the newly calculated v_i(k).
      # -----------------------------------------------------------------------

      error[i + 1] = (error[i]- v_prev[i] * eb[i])

      # -----------------------------------------------------------------------
      # Eq. (7.103), current-time form used in Algorithm 7.4:
      #
      #   v_i(k) = delta_D(k,i) / xi_b(k,i)
      #
      # It is stored for use as v_i(k-1) at the next sample.
      # -----------------------------------------------------------------------

      v[i] = delta_d[i] / xi_b_i

    # Commit k states. At the next call they become the k-1 states.
    self.delta = delta
    self.delta_d = delta_d

    self.xi_b = xi_b
    self.xi_f = xi_f
    self.gamma = gamma

    self.eb_prev = eb

    self.kappa_f = kappa_f
    self.kappa_b = kappa_b

    self.w = v.astype(np.complex64)

    # Algorithm 7.4 output error is already the a priori error.
    e_final = error[-1]
    y_final = dk - e_final

    return {
      "y": np.asarray(y_final, dtype=self.dtype)[()],
      "e": np.asarray(e_final, dtype=self.dtype)[()],
      "e_order": error,

      "ef": ef,
      "eb": eb,

      "delta": delta,
      "delta_d": delta_d,

      "gamma": gamma,

      "xi_f": xi_f,
      "xi_b": xi_b,

      "kappa_f": kappa_f,
      "kappa_b": kappa_b,

      "v": v,

      # Useful for debugging Algorithm 7.4 causality.
      "kappa_f_prev": kappa_f_prev,
      "kappa_b_prev": kappa_b_prev,
      "v_prev": v_prev,
    }

  def __call__(self, x: Array, d: Array, train: bool = True) -> Tuple[Array, Array, Array]:
    # LatticeRLS.__call__ already provides y/e/v histories and calls self.step().
    if train:
      return super().__call__(x, d, train=True)

    # Parent train=False restores the common LatticeRLS states. Preserve the
    # Algorithm 7.4-specific previous reflection-coefficient states as well.
    extra_state = (
      self.kappa_f.copy(),
      self.kappa_b.copy(),
    )

    result = super().__call__(x, d, train=False)

    self.kappa_f, self.kappa_b = extra_state

    return result