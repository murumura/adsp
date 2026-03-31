import numpy as np
from typing import Literal, Union
Array = Union[np.ndarray]

class FixedQuantizer:
  """
  Uniform signed fixed-point quantizer.

  Notes:
    - real and imag parts are quantized separately
    - rounding = nearest
    - overflow = saturation
  """

  def __init__(self, total_bits: int, frac_bits: int, signed: bool = True) -> None:
    if total_bits <= 0:
      raise ValueError("total_bits must be > 0")
    if frac_bits < 0:
      raise ValueError("frac_bits must be >= 0")
    if frac_bits >= total_bits:
      raise ValueError("frac_bits must be < total_bits")

    self.total_bits = int(total_bits)
    self.frac_bits = int(frac_bits)
    self.signed = bool(signed)

    self.scale = float(2 ** self.frac_bits)

    if self.signed:
      self.qmin = -(2 ** (self.total_bits - 1))
      self.qmax = (2 ** (self.total_bits - 1)) - 1
    else:
      self.qmin = 0
      self.qmax = (2 ** self.total_bits) - 1

    self.vmin = self.qmin / self.scale
    self.vmax = self.qmax / self.scale

  def quantizeReal(self, x: Array | float | complex) -> Array:
    x_arr = np.asarray(x, dtype=np.float64)
    q = np.round(x_arr * self.scale)
    q = np.clip(q, self.qmin, self.qmax)
    return (q / self.scale).astype(np.float32)

  def quantizeComplex(self, x: Array | complex) -> Array:
    x_arr = np.asarray(x)
    x_real = self.quantizeReal(np.real(x_arr))
    x_imag = self.quantizeReal(np.imag(x_arr))
    return (x_real + 1j * x_imag).astype(np.complex64)

  def quantizeValue(self, x: Array | float | complex) -> Array:
    x_arr = np.asarray(x)
    if np.iscomplexobj(x_arr):
      return self.quantizeComplex(x_arr)
    return self.quantizeReal(x_arr)

  def __call__(self, x: Array | float | complex) -> Array:
    return self.quantizeValue(x)

def qdot(
  a: Array,
  b: Array,
  q: FixedQuantizer,
  *,
  conj_a: bool = True,
  mode: Literal["after_add", "after_product"] = "after_add",
) -> np.complex64:
  a_arr = np.asarray(a).ravel()
  b_arr = np.asarray(b).ravel()

  if a_arr.shape != b_arr.shape:
    raise ValueError("qdot shape mismatch")

  lhs = np.conj(a_arr) if conj_a else a_arr

  if mode == "after_add":
    acc = np.sum(lhs * b_arr)
    return np.complex64(q(acc))
  if mode == "after_product":
    acc = np.complex64(0.0 + 0.0j)
    for lhs_i, rhs_i in zip(lhs, b_arr):
      prod_q = np.complex64(q(lhs_i * rhs_i))
      acc = np.complex64(q(acc + prod_q))
    return np.complex64(q(acc))

  raise ValueError(f"unsupported qdot mode: {mode}")


def qmatVec(
  a: Array,
  x: Array,
  q: FixedQuantizer,
  *,
  mode: Literal["after_add", "after_product"] = "after_add",
) -> Array:
  a_arr = np.asarray(a)
  x_arr = np.asarray(x).ravel()

  if a_arr.shape[1] != x_arr.shape[0]:
    raise ValueError("qmatVec shape mismatch")

  y = np.empty((a_arr.shape[0],), dtype=np.complex64)
  for row_idx in range(a_arr.shape[0]):
    y[row_idx] = qdot(a_arr[row_idx, :], x_arr, q, conj_a=False, mode=mode)

  return q(y)


def qouter(u: Array, v: Array, q: FixedQuantizer) -> Array:
  u_arr = np.asarray(u).ravel()
  v_arr = np.asarray(v).ravel()

  out = np.empty((u_arr.size, v_arr.size), dtype=np.complex64)
  for row_idx in range(u_arr.size):
    for col_idx in range(v_arr.size):
      out[row_idx, col_idx] = np.complex64(q(u_arr[row_idx] * v_arr[col_idx]))

  return q(out)