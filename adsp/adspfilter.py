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
    raise ValueError("csignscalar expects a scalar (0-d).")
  return np.complex64(e / mag) if mag > eps else np.complex64(0.0 + 0.0j)


def toeplitzFromFirstRow(first_row: Array) -> Array:
  """Toeplitz with first row = [r0, r1, ..., r_{M-1}] and Hermitian-symmetric for real AR(1)."""
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
  From equation (2.83): R = σ_v²/(1-a²) * Toeplitz([1, a, a², ..., a^(order-1)])
  
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
  mu: float
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


# -----------------------------------------------------------------------------
# LMS
# -----------------------------------------------------------------------------
@dataclass
class LMS(BaseLMS):
  """
  Complex LMS (Diniz Algorithm 3.2)
    w(k+1) = w(k) + 2 μ e*(k) x(k)
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
    print(f"Current μ: {a['current_mu']:.6f}")
    print(f"Stability: {'✓ STABLE' if a['is_stable_R'] else '✗ UNSTABLE'}")

    print("\nEIGENVALUE ANALYSIS:")
    print(f"  Max eigenvalue (λ_max): {a['lambda_max']:.6f}")
    print(f"  Min eigenvalue (λ_min): {a['lambda_min']:.6f}")
    print(f"  Eigenvalue spread: {a['eigenvalue_spread']:.2f}")
    if a["convergence_warning"]:
      print("  WARNING: High eigenvalue spread will slow convergence")

    print("\nSTABILITY BOUNDS:")
    print(f"  From eigenvalues (1/λ_max): {a['max_stable_mu_eigenvalue']:.6f}")
    print(f"  From trace (1/tr[R]): {a['max_stable_mu_trace']:.6f}")

    print("\nPERFORMANCE PREDICTIONS:")
    print(f"  Theoretical misadjustment: {a['theoretical_misadjustment']:.4f}")

    if a["lambda_min"] > 0:
      slow_tc = 1.0 / (2.0 * a["current_mu"] * a["lambda_min"])
      approx_iters = 4.6 * slow_tc
      print(f"  Approx. convergence iterations: {approx_iters:.0f}")

    print("\nRECOMMENDED μ RANGES:")
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


# -----------------------------------------------------------------------------
# Sign-* LMS family
# -----------------------------------------------------------------------------
@dataclass
class SignErrorLMS(BaseLMS):
  """
  Complex Sign-Error LMS (Diniz Algorithm 4.1, complex extension)
    w(k+1) = w(k) + 2 μ sgn[e(k)] x(k)
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
    w(k+1) = w(k) + 2 μ e*(k) sgn[x(k)]
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
class SignSignLMS(BaseLMS):
  """
  Complex Sign-Sign LMS (Diniz Sec. 4.2.4 extended)
    w(k+1) = w(k) + 2 μ sgn[e(k)] sgn[x(k)]
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
class DualSignLMS(BaseLMS):
  """
  Complex Dual-Sign LMS (Diniz Sec. 4.2.3 style)
  if |e(k)| > rho:
      w(k+1) = w(k) + 2 μ ε sgn[e(k)] x(k)
  else:
      w(k+1) = w(k) + 2 μ sgn[e(k)] x(k)
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
        sigma_e = csignscalar(e)
        gain = self.epsilon if (np.abs(e) > self.rho) else 1.0
        w = (w + (2.0 * self.mu) * gain * sigma_e * reg).astype(np.complex64)

      y_hist[k] = y
      e_hist[k] = e
      w_hist[k + 1] = w

    if train:
      self.w = w
    return y_hist, e_hist, w_hist


# -----------------------------------------------------------------------------
# Power-of-two error LMS
# -----------------------------------------------------------------------------
@dataclass
class PowerOfTwoErrorLMS(BaseLMS):
  """
  Power-of-Two Error LMS (Diniz Sec. 4.2.3)
    pe[e] as in Eq. (4.40)
    w(k+1) = w(k) + 2 μ pe[e(k)] x(k)
  """
  bd: int = 8
  tau: float = 0.0

  def p2e(self, e: Union[complex, Array]) -> Union[np.float32, Array]:
    e = np.asarray(e)
    abs_e = np.abs(e)
    s = realsign(e).astype(np.float32)
    thresh = 2.0 ** (-(self.bd - 1))

    region1 = abs_e >= 1.0
    region_mid = (abs_e >= thresh) & (abs_e < 1.0)

    mag = abs_e + 1e-12
    pow_term = np.exp2(np.floor(np.log2(mag))).astype(np.float32)

    return np.where(region1, s, np.where(region_mid, pow_term * s, self.tau * s)).astype(np.float32)

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
        pe = np.complex64(self.p2e(e))  # real scalar (float) but lift to complex
        w = (w + (2.0 * self.mu) * pe * reg).astype(np.complex64)

      y_hist[k] = y
      e_hist[k] = e
      w_hist[k + 1] = w

    if train:
      self.w = w
    return y_hist, e_hist, w_hist


# -----------------------------------------------------------------------------
# Transform-domain (power-normalized) TD-LMS / TD-NLMS
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# NLMS
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# Affine Projection (APA)
# -----------------------------------------------------------------------------
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
