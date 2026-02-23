import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union

Array = Union[np.ndarray]

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
  return np.concatenate([np.zeros(n_coef - 1, dtype=x.dtype), x])


# -----------------------------------------------------------------------------
# Base
# -----------------------------------------------------------------------------
@dataclass
class BaseLMS:
  mu: float | None
  filter_order: int
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
    print(f"Stability: {'✓ STABLE' if a['is_stable_R'] else '✗ UNSTABLE'}")

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
    print(f"Mean stable: {'✓' if a['is_stable_mean'] else '✗'}")
    print(f"MSE stable:  {'✓' if a['is_stable_mse'] else '✗'}")
    print("\nMisadjustment (4.28):")
    print(f"M = {a['theoretical_misadjustment']:.6e}")
    print("=" * 60)

# -----------------------------------------------------------------------------
# Sign-* LMS family
# -----------------------------------------------------------------------------
@dataclass
class SignErrorLMS(QuantizedLMS):
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
class SignSignLMS(QuantizedLMS):
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
class DualSignLMS(QuantizedLMS):
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
class PowerOfTwoErrorLMS(QuantizedLMS):
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

        # p(k) = P(k) x(k)
        p = P @ x_col

        # phi(k) = x^H P x
        phi = (x_col.conj().T @ p).item()

        # denom = alpha + phi(k)
        denom = self.alpha + phi
        if abs(denom) < self.eps:
          denom = denom + (self.eps + 0.0j)

        # k(k) = p(k) / denom
        k_vec = p / denom

        # Weight update (uses P(k))
        # w(k+1) = w(k) + 2 mu e*(k) p(k)
        w = (w + (2.0 * self.mu) * np.conj(e) * p.ravel()).astype(np.complex64)

        # P update
        # P(k+1) = (P(k) - k(k) x^H(k) P(k)) / alpha
        P = (P - k_vec @ (x_col.conj().T @ P)) / self.alpha

        # Optional: enforce Hermitian symmetry (numerical hygiene)
        # P = 0.5 * (P + P.conj().T)

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
class NLMS(BaseLMS):
  """
  Complex NLMS (Diniz Sec. 4.3)
    mu_k = mu_n / (tau + ||x||^2)
    w(k+1) = w(k) + mu_k e*(k) x(k)
  Here mu means mu_n (should satisfy 0 < mu_n < 2 for stability in the standard case).
  """
  tau: float = 1e-3

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

      norm = float(np.real(np.vdot(reg, reg)))  # ||reg||^2
      mu_k = float(self.mu / (self.tau + norm))

      if train:
        w = (w + mu_k * np.conj(e) * reg).astype(np.complex64)

      y_hist[k] = y
      e_hist[k] = e
      w_hist[k + 1] = w

    if train:
      self.w = w
    return y_hist, e_hist, w_hist
  
  def analyze(self, R: Array, verbose: bool = False) -> Dict[str, Any]:
    tr = float(np.trace(R).real)

    mu_n = float(self.mu)

    # Stability condition
    is_stable = (0.0 < mu_n < 2.0)

    # Effective LMS-equivalent step size
    mu_eff = mu_n / (2.0 * tr)

    # Approx steady-state misadjustment
    misadj = mu_n / 2.0

    res = dict(
      current_mu_n=mu_n,
      tau=float(self.tau),
      is_stable=is_stable,
      effective_mu=float(mu_eff),
      trace_R=float(tr),
      theoretical_misadjustment=float(misadj),
      upper_bound_mu_n=2.0,
      suggested_conservative_mu_n=0.5,
      suggested_aggressive_mu_n=1.0,
    )

    if verbose:
      self.printMuAnalysisNLMS(res)

    return res

  def printMuAnalysisNLMS(self, a: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("NLMS STEP SIZE ANALYSIS (Sec. 4.3)")
    print("=" * 60)

    print(f"mu_n: {a['current_mu_n']:.6f}")
    print(f"tau:  {a['tau']:.6e}")
    print(f"Stability: {'✓ STABLE' if a['is_stable'] else '✗ UNSTABLE'}")

    print("\nEffective LMS-equivalent step size:")
    print(f"  mu_eff ≈ mu_n / (2 tr[R]) = {a['effective_mu']:.6e}")

    print("\nTheoretical steady-state misadjustment:")
    print(f"  M ≈ mu_n / 2 = {a['theoretical_misadjustment']:.6f}")

    print("\nStability Bound:")
    print("  0 < mu_n < 2")

    print("\nRecommended mu_n ranges:")
    print(f"  Conservative: {a['suggested_conservative_mu_n']:.2f}")
    print(f"  Aggressive:   {a['suggested_aggressive_mu_n']:.2f}")

    print("=" * 60)

# Affine Projection (APA)
@dataclass
class AffineProjection(BaseLMS):
  """
  Complex Affine Projection Algorithm (APA), Diniz Sec. 4.6

  Uses projection order P (your code uses P+1 samples).
  """
  P: int = 3
  delta: float = 1e-3

  def __call__(self, x: Array, d: Array, train: bool = True) -> Tuple[Array, Array, Array]:
    x = np.asarray(x)
    d = np.asarray(d)
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
      # Build Xap(k): shape (M, P+1)
      cols = []
      for i in range(Pp):
        kk = k - i
        reg = x_pad[kk:kk + M][::-1].astype(np.complex64)
        cols.append(reg)
      Xap = np.stack(cols, axis=1)  # (M, P+1)

      d_ap = np.array([d[k - i] for i in range(Pp)], dtype=np.complex64)  # (P+1,)

      y_ap = Xap.conj().T @ w
      e_ap = d_ap - y_ap

      R = Xap.conj().T @ Xap
      R_reg = R + (self.delta * np.eye(Pp, dtype=np.complex64))
      g = np.linalg.solve(R_reg, e_ap)          # (P+1,)
      upd = (self.mu * (Xap @ g)).astype(np.complex64)

      if train:
        w = (w + upd).astype(np.complex64)

      y0 = y_ap[0]
      e0 = e_ap[0]

      y_hist[out_idx] = y0
      e_hist[out_idx] = e0
      w_hist[out_idx + 1] = w
      out_idx += 1

    if train:
      self.w = w
    return y_hist, e_hist, w_hist

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
