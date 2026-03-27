from dataclasses import dataclass, field
import numpy as np
from typing import Optional
from . import adspfilter

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

  def getRegressor(self, new_u: complex) -> np.ndarray:
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
class RlsDfe(BaseDfe):
  lam: float = 0.99
  delta: float = 1e-2
  eps: float = 1e-12
   
  # Core RLS engine instance
  rls_engine: adspfilter.RLS = field(init=False)

  def __post_init__(self):
    super().__post_init__()
    # RLS filter_order N results in N+1 taps
    self.rls_engine = adspfilter.RLS(
      filter_order=self.n_taps - 1,
      lam=self.lam,
      delta=self.delta,
      eps=self.eps
    )

  def __call__(self, u: np.ndarray, d: np.ndarray, train: bool = True):
    u = np.asarray(u)
    d = np.asarray(d)
    N = len(u)
    
    y_hist = np.zeros(N, dtype=np.complex64)
    e_hist = np.zeros(N, dtype=np.complex64)
    
    # Clear buffers for new signal sequence
    self.resetAllStates()
    
    for k in range(N):
      # 1. Get the hybrid regressor x(k)
      xk = self.getRegressor(u[k])
      
      yk, ek, _ = self.rls_engine(
        xk.reshape(1, -1),
        np.array([d[k]]),
        train=train
      )
      
      y_hist[k] = yk[0]
      e_hist[k] = ek[0]
      
      # 3. Update feedback logic
      self.updateFeedback(y_hist[k], d[k], training_mode=train)
    
    return y_hist, e_hist

  def resetAllStates(self):
    """Resets both the DFE buffers and the RLS weights/P-matrix."""
    self.resetBuffers()
    # Direct reset of RLS class members
    self.rls_engine.w[:] = 0
    self.rls_engine.S_D = (1.0 / self.delta) * np.eye(self.n_taps, dtype=np.complex64)

  def analyze(self, R: Optional[np.ndarray] = None):
    """Analyze the RLS portion using your existing Diniz analysis code."""
    return self.rls_engine.analyze(R=R, verbose=True)