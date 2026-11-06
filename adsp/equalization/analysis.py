"""Pole/residue checks of ideal two-sided inverse equalizers."""

import numpy as np


def inverse_polynomial_residues(h: np.ndarray, q: np.ndarray, reg: float):
  """W(D) = ||h|| D^2 / P(D), where P(D) has descending D powers 4,...,0.

  For each simple pole p of P, Res[W,p] = ||h|| p^2 / P'(p).
  The stable two-sided ROC retains only |p|>1 terms for w[0].
  """
  energy = float(np.vdot(h, h).real)
  poly = energy * np.asarray(q, dtype=np.complex128)[::-1].copy()
  poly[2] += energy * reg
  h_norm = np.sqrt(energy)
  roots = np.roots(poly)
  deriv = np.polyder(poly)
  residues = np.array([h_norm * p**2 / np.polyval(deriv, p) for p in roots])
  order = sorted(range(len(roots)), key=lambda i: (abs(roots[i]) < 1.0, np.angle(roots[i])))
  roots, residues = (roots[order], residues[order])
  center = -sum((residue / pole for pole, residue in zip(roots, residues) if abs(pole) > 1.0))
  return (poly, roots, residues, center)
