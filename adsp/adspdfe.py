from dataclasses import dataclass, field
import numpy as np
from typing import Optional, TypeAlias, Tuple
from . import adspfilter


Array: TypeAlias = np.ndarray

@dataclass
class BaseDfe:
  n_ff: int                    # Feedforward taps
  n_fb: int                    # Feedback taps
  constellation: np.ndarray = field(default_factory=lambda: np.array([-1.0, 1.0]))
  init_coef: Optional[np.ndarray] = None
   
  # Structural state
  n_taps: int = field(init=False)
  w: np.ndarray = field(init=False)
  ff_buf: np.ndarray = field(init=False)
  fb_buf: np.ndarray = field(init=False)

  def __post_init__(self):
    self.n_taps = self.n_ff + self.n_fb
    self.ff_buf = np.zeros(self.n_ff, dtype=np.complex64)
    self.fb_buf = np.zeros(self.n_fb, dtype=np.complex64)
    
    if self.init_coef is None:
      self.w = np.zeros(self.n_taps, dtype=np.complex64)
    else:
      self.w = np.asarray(self.init_coef, dtype=np.complex64).copy()

  def decision(self, y: complex) -> complex:
    """Generalized Nearest-Neighbor decision."""
    idx = np.abs(self.constellation - y).argmin()
    return self.constellation[idx]

  def makeRegressor(self, new_u: complex) -> np.ndarray:
    """Update FF buffer and return the concatenated [u_vec; d_hat_vec]."""
    # Shift and insert new sample into Feedforward buffer
    self.ff_buf[1:] = self.ff_buf[:-1]
    self.ff_buf[0] = new_u
    # x(k) = [u(k), ..., u(k-Nff+1), d_hat(k-1), ..., d_hat(k-Nfb)]
    return np.concatenate([self.ff_buf, self.fb_buf]).astype(np.complex64)

  def updateFeedback(self, y: complex, d_true: complex, training_mode: bool):
    """Update the feedback buffer based on mode."""
    if self.n_fb > 0:
      self.fb_buf[1:] = self.fb_buf[:-1]
      if training_mode:
        self.fb_buf[0] = d_true
      else:
        self.fb_buf[0] = self.decision(y)

  def resetBuffers(self):
    """Zero out the tapped-delay lines."""
    self.ff_buf.fill(0)
    self.fb_buf.fill(0)


@dataclass
class RLS(BaseLMS):
  """
  Conventional Complex RLS (Diniz Algorithm 5.3).

    S_D(k) = R_D^{-1}(k)

    psi(k) = S_D(k-1) x(k)

    k(k) = psi(k) / [lambda + x^H(k) psi(k)]

    w(k) = w(k-1) + k(k) e*(k)

    S_D(k) = (1/lambda)
             [S_D(k-1) - k(k) x^H(k) S_D(k-1)]

  Initialization:
    S_D(-1) = delta I
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

    if not (0.0 < self.lam <= 1.0):
      raise ValueError(f"lam must satisfy 0 < lam <= 1, got {self.lam}.")

    if self.delta <= 0.0:
      raise ValueError(f"delta must be > 0, got {self.delta}.")

    self.resetState()

  def resetState(self) -> None:
    self.w = np.zeros(self.n_coef, dtype=self.dtype)
    self.S_D = self.delta * np.eye(self.n_coef, dtype=self.dtype)

  def stepRegressor(
    self,
    reg: Array,
    d: complex,
    train: bool = True,
  ) -> Tuple[complex, complex, Array]:
    """
    One RLS iteration using an externally prepared regressor.

    Parameters
    ----------
    reg:
      Full regressor x(k), shape (n_coef,).

    d:
      Desired sample d(k).

    train:
      If True, update w(k) and S_D(k).

    Returns
    -------
    y:
      A priori output:
        y(k) = w^H(k-1) x(k)

    e:
      A priori error:
        e(k) = d(k) - y(k)

    w:
      Current coefficient vector after the optional update.
    """

    reg = np.asarray(reg, dtype=self.dtype)

    if reg.ndim != 1:
      raise ValueError(f"reg must be 1-D, got shape {reg.shape}.")

    if reg.shape[0] != self.n_coef:
      raise ValueError(
        f"reg must have length {self.n_coef}, got {reg.shape[0]}."
      )

    xk = reg.reshape(-1, 1)

    y = np.vdot(self.w, reg)
    e = np.asarray(d, dtype=self.dtype)[()] - y

    if train:
      psi = self.S_D @ xk

      denom = self.lam + np.real((xk.conj().T @ psi).item())

      if abs(denom) < self.eps:
        denom = self.eps

      k_vec = psi / denom

      self.w = (self.w + k_vec.ravel() * np.conj(e)).astype(self.dtype)

      self.S_D = (self.S_D - k_vec @ (xk.conj().T @ self.S_D)) / self.lam

      # Numerical symmetry cleanup.
      self.S_D = 0.5 * (self.S_D + self.S_D.conj().T)

    return y, e, self.w.copy()

  def __call__(
    self,
    x: Array,
    d: Array,
    train: bool = True,
  ) -> Tuple[Array, Array, Array]:
    x = np.asarray(x, dtype=self.dtype)
    d = np.asarray(d, dtype=self.dtype)

    if x.ndim != 1 or d.ndim != 1:
      raise ValueError(f"x and d must be 1-D, got {x.shape} and {d.shape}.")

    if len(x) != len(d):
      raise ValueError(f"x and d must have same length, got {len(x)} and {len(d)}.")

    N = len(x)
    x_pad = adspfilter._prepad(x, self.n_coef)

    y_hist = np.empty(N, dtype=self.dtype)
    e_hist = np.empty(N, dtype=self.dtype)
    w_hist = np.empty((N + 1, self.n_coef), dtype=self.dtype)

    w_hist[0] = self.w

    for k in range(N):
      reg = x_pad[k:k + self.n_coef][::-1]

      y, e, w = self.stepRegressor(reg, d[k], train=train,)

      y_hist[k] = y
      e_hist[k] = e
      w_hist[k + 1] = w

    return y_hist, e_hist, w_hist

@dataclass
class ErrorFeedbackLatticeRlsEq(BaseDfe):
  """
  Diniz Example 7.2 linear equalizer using Algorithm 7.3.

  Example 7.2:
    channel          : h(k) = 0.1 * 0.5^k, k = 0,...,8
    equalizer order  : 25
    equalizer taps   : 26
    lambda           : 0.99
    epsilon          : 0.1
    equalizer delay L: 18

  This is a linear equalizer, not a decision-feedback equalizer.
  Therefore n_fb must be zero.

  Notes:
    lrls_engine.w : lattice feedforward coefficients v_i(k)
    self.w        : equivalent direct-form FIR equalizer coefficients
  """

  n_ff: int = 26
  n_fb: int = 0
  lam: float = 0.99
  epsilon: float = 0.1
  eps: float = 1e-12
  verify_direct: bool = False

  lrls_engine: adspfilter.ErrorFeedbackLatticeRLS = field(init=False, repr=False)
  wb_prev: list[np.ndarray] = field(init=False, repr=False)

  post_y_hist: np.ndarray = field(init=False, repr=False)
  post_e_hist: np.ndarray = field(init=False, repr=False)
  w_hist: np.ndarray = field(init=False, repr=False)

  def __post_init__(self):
    super().__post_init__()

    if self.n_fb != 0:
      raise ValueError("Diniz Example 7.2 is a linear equalizer; n_fb must be 0.")

    if self.n_ff <= 0:
      raise ValueError("n_ff must be positive.")

    if self.init_coef is not None and np.any(np.abs(self.w) > self.eps):
      raise ValueError("Nonzero direct-form init_coef is not supported for the LRLS initialization.")

    self.lrls_engine = adspfilter.ErrorFeedbackLatticeRLS(
      filter_order=self.n_ff - 1,
      lam=self.lam,
      epsilon=self.epsilon,
      eps=self.eps,
    )

    self.resetStates()

  @property
  def v(self) -> np.ndarray:
    """Lattice feedforward coefficients v_i(k), Eq. (7.96)."""
    return np.asarray(np.real(self.lrls_engine.w), dtype=np.float64)

  def resetStates(self):
    """Reset the equalizer buffers, LRLS states, and direct-form reconstruction states."""
    self.resetBuffers()
    self.lrls_engine.resetState()

    self.w = np.zeros(self.n_ff, dtype=np.complex64)

    # wb_prev[i] = w_b(k-1, i), a vector with i coefficients.
    self.wb_prev = [np.zeros(i, dtype=np.float64) for i in range(self.n_ff + 1)]

  def latticeToDirect(self):
    """
    Convert the current lattice representation to the direct FIR coefficient vector.

    Predictor order updates:
      w_f(k,i+1) = [w_f(k,i); 0] - kappa_f(k,i) [w_b(k-1,i); -1]       (7.32)

      w_b(k,i+1) = [0; w_b(k-1,i)] - kappa_b(k,i) [-1; w_f(k,i)]       (7.28)

    Joint-process direct-form order update:
      w(k,i+1) = [w(k,i); 0] + v_i(k) [-w_b(k,i); 1]                    (7.63)
    """

    kappa_f = np.asarray(self.lrls_engine.kappa_f, dtype=np.float64)
    kappa_b = np.asarray(self.lrls_engine.kappa_b, dtype=np.float64)
    v = self.v

    wf_cur = [np.zeros(0, dtype=np.float64)]
    wb_cur = [np.zeros(0, dtype=np.float64)]
    w_direct = np.zeros(0, dtype=np.float64)

    for i in range(self.n_ff):
      # Eq. (7.63): direct joint-process coefficient vector of order i+1.
      basis_i = np.concatenate([-wb_cur[i], [1.0]])
      w_direct = np.concatenate([w_direct, [0.0]]) + v[i] * basis_i

      # Eqs. (7.32), (7.28): predictor coefficients required by the next section.
      wf_aug = np.concatenate([wf_cur[i], [0.0]])
      wb_aug = np.concatenate([self.wb_prev[i], [-1.0]])
      wf_next = wf_aug - kappa_f[i] * wb_aug

      wb_time_aug = np.concatenate([[0.0], self.wb_prev[i]])
      wf_aug_b = np.concatenate([[-1.0], wf_cur[i]])
      wb_next = wb_time_aug - kappa_b[i] * wf_aug_b

      wf_cur.append(wf_next)
      wb_cur.append(wb_next)

    self.wb_prev = wb_cur
    self.w = w_direct.astype(np.complex64)

  def __call__(
    self,
    u: np.ndarray,
    d: np.ndarray,
    train: bool = True,
    reset: bool = True,
  ):
    """
    Process a received sequence.

    Returned y/e follow the normal adaptive-filter convention:
      y(k) = output produced by coefficients available before sample k
      e(k) = d(k) - y(k)

    During training, post_y_hist/post_e_hist contain the a posteriori LRLS results.
    """

    u = np.asarray(u, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)

    if u.ndim != 1 or d.ndim != 1:
      raise ValueError("u and d must be one-dimensional.")

    if len(u) != len(d):
      raise ValueError(f"u and d must have equal length, got {len(u)} and {len(d)}.")

    if reset:
      self.resetStates()

    n_samples = len(u)

    y_hist = np.zeros(n_samples, dtype=np.float64)
    e_hist = np.zeros(n_samples, dtype=np.float64)
    post_y_hist = np.zeros(n_samples, dtype=np.float64)
    post_e_hist = np.zeros(n_samples, dtype=np.float64)
    w_hist = np.zeros((n_samples + 1, self.n_ff), dtype=np.float64)

    w_hist[0] = np.real(self.w)

    for k in range(n_samples):
      # BaseDfe supplies the direct-form delay-line regressor.
      xk = self.makeRegressor(u[k])

      # A priori output/error: coefficients learned through k-1.
      y_prior = float(np.real(np.vdot(self.w, xk)))
      e_prior = float(d[k] - y_prior)

      y_hist[k] = y_prior
      e_hist[k] = e_prior

      if train:
        # LRLS itself accepts the scalar process u(k), not a prebuilt regressor.
        result = self.lrls_engine.step(float(u[k]), float(d[k]))

        # Recover the equivalent direct-form FIR weights.
        self.latticeToDirect()

        y_post = float(np.real(np.vdot(self.w, xk)))
        e_post = float(d[k] - y_post)

        post_y_hist[k] = y_post
        post_e_hist[k] = e_post

        if self.verify_direct:
          lattice_y = float(np.real(result["y"]))
          lattice_e = float(np.real(result["e"]))

          if not np.isclose(y_post, lattice_y, rtol=1e-5, atol=1e-9):
            raise RuntimeError(f"Direct/lattice output mismatch at k={k}: {y_post} vs {lattice_y}")

          if not np.isclose(e_post, lattice_e, rtol=1e-5, atol=1e-9):
            raise RuntimeError(f"Direct/lattice error mismatch at k={k}: {e_post} vs {lattice_e}")

      else:
        post_y_hist[k] = y_prior
        post_e_hist[k] = e_prior

      w_hist[k + 1] = np.real(self.w)

    self.post_y_hist = post_y_hist
    self.post_e_hist = post_e_hist
    self.w_hist = w_hist

    return y_hist, e_hist