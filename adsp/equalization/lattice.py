"""equalization / lattice algorithms and helpers."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from adsp.filters.lattice import error_feedback as adspfilter
from adsp.equalization.dfe import BaseDfe


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

  dtype: object = np.complex128
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

    if adspfilter is None:
      raise ImportError("ErrorFeedbackLatticeRlsEq requires the adspfilter module.")

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

    self.w = np.zeros(self.n_ff, dtype=self.dtype)

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
    self.w = w_direct.astype(self.dtype)

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


__all__ = ['ErrorFeedbackLatticeRlsEq']
