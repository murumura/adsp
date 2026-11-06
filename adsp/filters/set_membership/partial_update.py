"""filters / set membership / partial update algorithms and helpers."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
from adsp.core import Array
from adsp.filters.base import BaseLMS


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
      raise ValueError("selector_mode must be 'topEnergy', 'random', or 'external', "
                       f"got {self.selector_mode}.")

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
        raise ValueError(f"up_selector must have shape {(N, K)}, got {self.up_selector.shape}.")

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
    row_energy = np.sum(np.abs(Xap)**2, axis=1)
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


__all__ = ['SMPartialUpdateAffineProjection']
