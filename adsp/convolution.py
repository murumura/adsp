"""Complex-safe overlap-save convolution."""

import numpy as np


def olsFFTConv(x, h, N, debug=False):
  """Return full linear convolution, including the final M-1 tail samples."""
  x, h = np.asarray(x), np.asarray(h)
  if x.ndim != 1 or h.ndim != 1 or not len(x) or not len(h) or N < len(h):
    raise ValueError("Nonempty vectors and FFT size >= filter length are required.")
  block_size = N - len(h) + 1
  length = len(x) + len(h) - 1
  padded = np.pad(x, (len(h) - 1, len(h) - 1 + N))
  response = np.fft.fft(h, N)
  output = np.empty(length, dtype=np.complex128)
  for start in range(0, length, block_size):
    stop = min(start + block_size, length)
    result = np.fft.ifft(np.fft.fft(padded[start:start + N]) * response)
    output[start:stop] = result[len(h) - 1:len(h) - 1 + stop - start]
  if debug:
    print(f"OLS samples={len(x)}, taps={len(h)}, fft={N}, output={length}")
  return output if np.iscomplexobj(x) or np.iscomplexobj(h) else output.real


__all__ = ["olsFFTConv"]
