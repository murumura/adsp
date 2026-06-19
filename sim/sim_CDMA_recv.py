#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sim_CDMA_recv.py

Diniz Section 6.9.4 style wireless-channel simulation.

Naming style:
  - classes use UpperCamelCase
  - functions, methods, fields, and local variables use lowerCamelCase

Simulation chain:
  DS-CDMA/QPSK composite chip signal
        ->
  multipath time-varying Jakes fading channel
        ->
  additive noise
        ->
  set-membership NLMS channel tracker
        ->
  set-membership NLMS linear equalizer

Internal adaptive-filter convention:
  y(k) = w^H(k) xVec(k)
  e(k) = d(k) - y(k)
  w(k+1) = w(k) + mu(k) xVec(k) e*(k) / ||xVec(k)||^2

The channel simulator generates:
  d(k) = hTrue^H(k) xVec(k) + n(k)

So the channel-tracking weight vector w(k) should track hTrue(k).

The equalizer estimates:
  x(k - delay) from received signal d(k)
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, TypeAlias

import numpy as np
import matplotlib.pyplot as plt


Array: TypeAlias = np.ndarray


# =============================================================================
# Utility functions
# =============================================================================

def genQpskSymbols(nSym: int, rng: np.random.Generator) -> Array:
  """Generate unit-power QPSK symbols."""
  re = 2 * rng.integers(0, 2, nSym) - 1
  im = 2 * rng.integers(0, 2, nSym) - 1
  return (re + 1j * im).astype(np.complex128) / math.sqrt(2.0)


def hardQpsk(x: Array) -> Array:
  """Hard-slice complex samples to unit-power QPSK."""
  x = np.asarray(x)
  re = np.where(np.real(x) >= 0.0, 1.0, -1.0)
  im = np.where(np.imag(x) >= 0.0, 1.0, -1.0)
  return (re + 1j * im).astype(np.complex128) / math.sqrt(2.0)


def smoothCurve(x: Array, winLen: int = 256) -> Array:
  """Edge-safe moving-average smoothing."""
  x = np.asarray(x, dtype=np.float64)

  if winLen <= 1:
    return x

  padLeft = winLen // 2
  padRight = winLen - 1 - padLeft

  xPad = np.pad(x, (padLeft, padRight), mode="edge")
  win = np.ones(winLen, dtype=np.float64) / float(winLen)

  return np.convolve(xPad, win, mode="valid")


def toDb10(x: Array, floor: float = 1e-15) -> Array:
  """10 log10 with floor."""
  return 10.0 * np.log10(np.asarray(x) + floor)


def toDb20(x: Array, floor: float = 1e-15) -> Array:
  """20 log10 with floor."""
  return 20.0 * np.log10(np.asarray(x) + floor)


def makeHadamard(order: int) -> Array:
  """Build a Hadamard matrix recursively. order must be a power of two."""
  if order < 1 or (order & (order - 1)) != 0:
    raise ValueError(f"order must be a positive power of two, got {order}.")

  h = np.array([[1.0]], dtype=np.float64)

  while h.shape[0] < order:
    h = np.block([[h, h], [h, -h]])

  return h


# =============================================================================
# DS-CDMA source model
# =============================================================================

@dataclass
class DsCdmaSource:
  """
  Simple synchronous DS-CDMA composite source.

  numUsers:
    Number of synchronous users.

  spreadingLen:
    Number of chips per symbol.

  useWalshCodes:
    If possible, use orthogonal Walsh-Hadamard spreading codes.
    Otherwise, use random +/-1 spreading codes.

  Output composite chip sequence is normalized to unit average power.
  """

  numUsers: int = 4
  spreadingLen: int = 16
  seed: int = 123
  useWalshCodes: bool = True

  def generateCodes(self) -> Array:
    if (
      self.useWalshCodes
      and self.spreadingLen >= self.numUsers
      and (self.spreadingLen & (self.spreadingLen - 1)) == 0
    ):
      codes = makeHadamard(self.spreadingLen)[: self.numUsers, :]
    else:
      rng = np.random.default_rng(self.seed)

      codes = 2 * rng.integers(
        0,
        2,
        size=(self.numUsers, self.spreadingLen),
      ) - 1

    return codes.astype(np.float64) / math.sqrt(float(self.spreadingLen))

  def generate(self, numSymbols: int) -> Tuple[Array, Array, Array, float]:
    """
    Return:
      xChip:       composite chip-rate transmit sequence, shape (K,)
      symbols:     user QPSK symbols, shape (numSymbols, numUsers)
      codes:       spreading codes, shape (numUsers, spreadingLen)
      normScale:   scale used to normalize composite chip power
    """
    rng = np.random.default_rng(self.seed + 999)
    codes = self.generateCodes()

    symbols = np.zeros((numSymbols, self.numUsers), dtype=np.complex128)

    for userIdx in range(self.numUsers):
      symbols[:, userIdx] = genQpskSymbols(numSymbols, rng)

    nChips = numSymbols * self.spreadingLen
    xChip = np.zeros(nChips, dtype=np.complex128)

    for symIdx in range(numSymbols):
      chipBlock = np.zeros(self.spreadingLen, dtype=np.complex128)

      for userIdx in range(self.numUsers):
        chipBlock += symbols[symIdx, userIdx] * codes[userIdx, :]

      xChip[
        symIdx * self.spreadingLen : (symIdx + 1) * self.spreadingLen
      ] = chipBlock

    normScale = math.sqrt(float(np.mean(np.abs(xChip) ** 2)) + 1e-15)
    xChip = xChip / normScale

    return xChip, symbols, codes, normScale


# =============================================================================
# Jakes fading model
# =============================================================================

@dataclass
class JakesFadingGenerator:
  """
  Frequency-domain Jakes/Clarke fading generator.

  The generator approximates:

      R_a(f) = 1 / [pi fd sqrt(1 - (f/fd)^2)], |f| < fd

  by frequency-domain shaping:

      white Gaussian -> FFT -> multiply sqrt(R_a(f)) -> IFFT

  Output is complex Rayleigh fading with approximately E{|a[k]|^2}=1.
  """

  fd: float
  fs: float
  seed: int = 0
  eps: float = 1e-12

  def generate(self, nSamples: int) -> Array:
    if nSamples <= 0:
      raise ValueError(f"nSamples must be positive, got {nSamples}.")

    if self.fd <= 0.0:
      return np.ones(nSamples, dtype=np.complex128)

    if self.fs <= 2.0 * self.fd:
      raise ValueError(
        f"Fading sample rate fs={self.fs} must be larger than 2*fd={2*self.fd}."
      )

    rng = np.random.default_rng(self.seed)

    nFft = 1

    while nFft < max(nSamples, 16):
      nFft *= 2

    freqs = np.fft.fftfreq(nFft, d=1.0 / self.fs)
    fAbs = np.abs(freqs)

    psd = np.zeros(nFft, dtype=np.float64)
    inside = fAbs < self.fd

    ratio = np.clip(fAbs[inside] / self.fd, 0.0, 1.0 - 1e-8)
    psd[inside] = 1.0 / (
      np.pi * self.fd * np.sqrt(1.0 - ratio * ratio)
    )

    hDoppler = np.sqrt(psd)

    white = (
      rng.normal(size=nFft) + 1j * rng.normal(size=nFft)
    ) / math.sqrt(2.0)

    shaped = np.fft.ifft(np.fft.fft(white) * hDoppler)
    fading = shaped[:nSamples]

    fading = fading / math.sqrt(float(np.mean(np.abs(fading) ** 2)) + self.eps)

    return fading.astype(np.complex128)


# =============================================================================
# Multipath Jakes channel
# =============================================================================

@dataclass
class MultipathJakesChannel:
  """
  Time-varying multipath channel with Jakes fading on each path.

  delaysSec:
    Physical path delays in seconds.

  powersDb:
    Average relative path powers in dB.

  fsSim:
    Main system simulation sampling rate.

  fd:
    Maximum Doppler frequency.

  alpha1:
    Fading oversampling factor. Fading is generated at:

        fsChannel = alpha1 * fd

    and interpolated to fsSim.
  """

  delaysSec: Array
  powersDb: Array
  fsSim: float
  fd: float
  alpha1: float = 10.0
  seed: int = 42
  normalizeTotalPower: bool = True

  def __post_init__(self) -> None:
    self.delaysSec = np.asarray(self.delaysSec, dtype=np.float64)
    self.powersDb = np.asarray(self.powersDb, dtype=np.float64)

    if self.delaysSec.ndim != 1 or self.powersDb.ndim != 1:
      raise ValueError("delaysSec and powersDb must be 1-D arrays.")

    if self.delaysSec.shape[0] != self.powersDb.shape[0]:
      raise ValueError("delaysSec and powersDb must have the same length.")

    if self.fsSim <= 0.0:
      raise ValueError(f"fsSim must be positive, got {self.fsSim}.")

    if self.fd < 0.0:
      raise ValueError(f"fd must be non-negative, got {self.fd}.")

    self.delaySamples = np.rint(self.delaysSec * self.fsSim).astype(np.int64)

    powers = 10.0 ** (self.powersDb / 10.0)

    if self.normalizeTotalPower:
      powers = powers / np.sum(powers)

    self.pathPowers = powers.astype(np.float64)

    if self.fd > 0.0:
      self.fsChannel = max(float(self.alpha1 * self.fd), 2.1 * self.fd)
    else:
      self.fsChannel = 1.0

  @staticmethod
  def channelA(
    fsSim: float,
    fd: float,
    alpha1: float = 10.0,
    seed: int = 42,
  ) -> "MultipathJakesChannel":
    delaysNs = np.array([0.0, 110.0, 190.0, 410.0])
    powersDb = np.array([0.0, -9.7, -19.2, -22.8])

    return MultipathJakesChannel(
      delaysSec=delaysNs * 1e-9,
      powersDb=powersDb,
      fsSim=fsSim,
      fd=fd,
      alpha1=alpha1,
      seed=seed,
    )

  @staticmethod
  def channelB(
    fsSim: float,
    fd: float,
    alpha1: float = 10.0,
    seed: int = 42,
  ) -> "MultipathJakesChannel":
    """
    With fsSim = 3.84 MHz, delays map approximately to:
      [0, 1, 3, 5, 9, 14] samples/chips
    """
    delaysNs = np.array([0.0, 200.0, 800.0, 1200.0, 2300.0, 3700.0])
    powersDb = np.array([0.0, -0.9, -4.9, -8.0, -7.8, -23.9])

    return MultipathJakesChannel(
      delaysSec=delaysNs * 1e-9,
      powersDb=powersDb,
      fsSim=fsSim,
      fd=fd,
      alpha1=alpha1,
      seed=seed,
    )

  def interpToFastRate(self, fadingSlow: Array, nFast: int) -> Array:
    if len(fadingSlow) == nFast:
      return fadingSlow.astype(np.complex128)

    tSlow = np.arange(len(fadingSlow), dtype=np.float64) / self.fsChannel
    tFast = np.arange(nFast, dtype=np.float64) / self.fsSim

    realFast = np.interp(tFast, tSlow, fadingSlow.real)
    imagFast = np.interp(tFast, tSlow, fadingSlow.imag)

    fadingFast = realFast + 1j * imagFast
    fadingFast = fadingFast / math.sqrt(
      float(np.mean(np.abs(fadingFast) ** 2)) + 1e-12
    )

    return fadingFast.astype(np.complex128)

  def generatePathFading(self, nFast: int) -> Array:
    durationSec = nFast / self.fsSim

    if self.fd <= 0.0:
      return np.ones((nFast, len(self.delaysSec)), dtype=np.complex128)

    nSlow = int(math.ceil(durationSec * self.fsChannel)) + 16

    fades = []

    for pathIdx in range(len(self.delaysSec)):
      gen = JakesFadingGenerator(
        fd=self.fd,
        fs=self.fsChannel,
        seed=self.seed + 1009 * pathIdx,
      )

      fadingSlow = gen.generate(nSlow)
      fadingFast = self.interpToFastRate(fadingSlow, nFast)

      fades.append(fadingFast)

    return np.stack(fades, axis=1)

  def makeTrueImpulseResponse(self, nFast: int) -> Array:
    if nFast <= 0:
      raise ValueError(f"nFast must be positive, got {nFast}.")

    nTaps = int(np.max(self.delaySamples)) + 1
    hTrue = np.zeros((nFast, nTaps), dtype=np.complex128)

    fades = self.generatePathFading(nFast)

    for pathIdx, delay in enumerate(self.delaySamples):
      gain = math.sqrt(float(self.pathPowers[pathIdx]))
      hTrue[:, delay] += gain * fades[:, pathIdx]

    return hTrue

  def filter(self, x: Array, noiseVar: float = 0.0) -> Tuple[Array, Array, Array]:
    """
    Return:
      d:       received signal
      hTrue:   adaptive-filter true vector, shape (K, nTaps)
      noise:   additive noise sequence

    Uses:
      d(k) = hTrue^H(k) xVec(k) + n(k)
    """
    x = np.asarray(x, dtype=np.complex128)
    nFast = len(x)
    hTrue = self.makeTrueImpulseResponse(nFast)

    d = np.zeros(nFast, dtype=np.complex128)

    for tapIdx in range(hTrue.shape[1]):
      if tapIdx == 0:
        d += np.conj(hTrue[:, tapIdx]) * x
      else:
        d[tapIdx:] += np.conj(hTrue[tapIdx:, tapIdx]) * x[: nFast - tapIdx]

    if noiseVar > 0.0:
      rng = np.random.default_rng(self.seed + 99991)
      noise = math.sqrt(noiseVar / 2.0) * (
        rng.normal(size=nFast) + 1j * rng.normal(size=nFast)
      )
    else:
      noise = np.zeros(nFast, dtype=np.complex128)

    d = d + noise

    return d, hTrue, noise


# =============================================================================
# Set-membership NLMS channel tracker
# =============================================================================

@dataclass
class SmNlmsChannelTracker:
  """
  Set-membership NLMS tracker for the generated channel.

  Model:
    y(k) = w^H(k) xVec(k)
    e(k) = d(k) - y(k)
  """

  nCoef: int
  gammaBar: float
  eps: float = 1e-9

  def run(self, x: Array, d: Array) -> Dict[str, Array]:
    x = np.asarray(x, dtype=np.complex128)
    d = np.asarray(d, dtype=np.complex128)

    if x.ndim != 1 or d.ndim != 1:
      raise ValueError("x and d must be 1-D arrays.")

    if len(x) != len(d):
      raise ValueError("x and d must have same length.")

    nSamples = len(x)
    w = np.zeros(self.nCoef, dtype=np.complex128)

    y = np.zeros(nSamples, dtype=np.complex128)
    e = np.zeros(nSamples, dtype=np.complex128)
    wHist = np.zeros((nSamples, self.nCoef), dtype=np.complex128)
    updateFlags = np.zeros(nSamples, dtype=bool)
    muHist = np.zeros(nSamples, dtype=np.float64)

    xPad = np.concatenate([
      np.zeros(self.nCoef - 1, dtype=np.complex128),
      x,
    ])

    for sampleIdx in range(nSamples):
      reg = xPad[sampleIdx : sampleIdx + self.nCoef][::-1]

      y[sampleIdx] = np.vdot(w, reg)
      e[sampleIdx] = d[sampleIdx] - y[sampleIdx]

      absE = float(abs(e[sampleIdx]))

      if absE > self.gammaBar:
        muSm = 1.0 - self.gammaBar / max(absE, self.eps)
        denom = float(np.vdot(reg, reg).real) + self.eps

        w = w + muSm * reg * np.conj(e[sampleIdx]) / denom

        updateFlags[sampleIdx] = True
        muHist[sampleIdx] = muSm

      wHist[sampleIdx] = w

    return dict(
      y=y,
      e=e,
      wHist=wHist,
      updateFlags=updateFlags,
      muHist=muHist,
    )


# =============================================================================
# Set-membership NLMS linear equalizer
# =============================================================================

@dataclass
class SmNlmsLinearEqualizer:
  """
  SM-NLMS adaptive linear equalizer.

  It receives the channel output r(k), builds a received-signal regressor,
  and estimates the transmitted chip x(k - decisionDelay).

  Model:
    z(k) = q^H(k) rVec(k)
    eEq(k) = x(k - decisionDelay) - z(k)

  Update only when:
    |eEq(k)| > gammaBar
  """

  nCoef: int
  gammaBar: float
  decisionDelay: int = 0
  eps: float = 1e-9

  def run(self, r: Array, xRef: Array) -> Dict[str, Array]:
    r = np.asarray(r, dtype=np.complex128)
    xRef = np.asarray(xRef, dtype=np.complex128)

    if r.ndim != 1 or xRef.ndim != 1:
      raise ValueError("r and xRef must be 1-D arrays.")

    if len(r) != len(xRef):
      raise ValueError("r and xRef must have same length.")

    if self.decisionDelay < 0:
      raise ValueError("decisionDelay must be non-negative.")

    nSamples = len(r)
    q = np.zeros(self.nCoef, dtype=np.complex128)

    z = np.zeros(nSamples, dtype=np.complex128)
    eEq = np.zeros(nSamples, dtype=np.complex128)
    qHist = np.zeros((nSamples, self.nCoef), dtype=np.complex128)
    updateFlags = np.zeros(nSamples, dtype=bool)
    muHist = np.zeros(nSamples, dtype=np.float64)

    # xHatAligned[targetIdx] stores the equalizer output that estimates xRef[targetIdx].
    xHatAligned = np.zeros(nSamples, dtype=np.complex128)
    validMask = np.zeros(nSamples, dtype=bool)

    rPad = np.concatenate([
      np.zeros(self.nCoef - 1, dtype=np.complex128),
      r,
    ])

    for sampleIdx in range(nSamples):
      targetIdx = sampleIdx - self.decisionDelay

      reg = rPad[sampleIdx : sampleIdx + self.nCoef][::-1]
      z[sampleIdx] = np.vdot(q, reg)

      if targetIdx < 0 or targetIdx >= nSamples:
        qHist[sampleIdx] = q
        continue

      desired = xRef[targetIdx]
      eEq[sampleIdx] = desired - z[sampleIdx]

      absE = float(abs(eEq[sampleIdx]))

      if absE > self.gammaBar:
        muSm = 1.0 - self.gammaBar / max(absE, self.eps)
        denom = float(np.vdot(reg, reg).real) + self.eps

        q = q + muSm * reg * np.conj(eEq[sampleIdx]) / denom

        updateFlags[sampleIdx] = True
        muHist[sampleIdx] = muSm

      xHatAligned[targetIdx] = z[sampleIdx]
      validMask[targetIdx] = True
      qHist[sampleIdx] = q

    return dict(
      z=z,
      eEq=eEq,
      qHist=qHist,
      updateFlags=updateFlags,
      muHist=muHist,
      xHatAligned=xHatAligned,
      validMask=validMask,
    )


# =============================================================================
# Metrics and plotting
# =============================================================================

def computeTrackingMetrics(
  hTrue: Array,
  wHist: Array,
  e: Array,
  updateFlags: Array,
) -> Dict[str, Array]:
  n = min(hTrue.shape[0], wHist.shape[0], len(e))

  hUse = hTrue[:n, :]
  wUse = wHist[:n, :]

  msd = np.sum(np.abs(hUse - wUse) ** 2, axis=1)
  hPower = np.sum(np.abs(hUse) ** 2, axis=1)
  nmsd = msd / (hPower + 1e-15)
  mse = np.abs(e[:n]) ** 2

  return dict(
    mse=mse,
    msd=msd,
    nmsd=nmsd,
    mseDb=toDb10(mse),
    msdDb=toDb10(msd),
    nmsdDb=toDb10(nmsd),
    updateRate=float(np.mean(updateFlags[:n])),
  )


def computeEqualizerMetrics(
  xRef: Array,
  equalizerOut: Dict[str, Array],
) -> Dict[str, object]:
  xRef = np.asarray(xRef, dtype=np.complex128)

  validMask = np.asarray(equalizerOut["validMask"], dtype=bool)
  xHatAligned = np.asarray(equalizerOut["xHatAligned"], dtype=np.complex128)
  eEq = np.asarray(equalizerOut["eEq"], dtype=np.complex128)

  validX = xRef[validMask]
  validHat = xHatAligned[validMask]

  if validX.size == 0:
    chipMse = np.inf
    evm = np.inf
  else:
    chipErr = validX - validHat
    chipMse = float(np.mean(np.abs(chipErr) ** 2))
    evm = float(chipMse / (np.mean(np.abs(validX) ** 2) + 1e-15))

  return dict(
    eqMse=np.abs(eEq) ** 2,
    eqMseDb=toDb10(np.abs(eEq) ** 2),
    chipMse=chipMse,
    chipMseDb=float(toDb10(chipMse)),
    evm=evm,
    evmDb=float(toDb10(evm)),
    updateRate=float(np.mean(equalizerOut["updateFlags"])),
  )


def despreadUserSymbols(
  chipSeq: Array,
  validMask: Array,
  codes: Array,
  userIdx: int,
  spreadingLen: int,
) -> Tuple[Array, Array]:
  """
  Despread a recovered chip sequence.

  Return:
    symbolEst: estimated symbols
    symbolMask: True when the corresponding symbol used all valid chips
  """
  chipSeq = np.asarray(chipSeq, dtype=np.complex128)
  validMask = np.asarray(validMask, dtype=bool)

  nSymbols = len(chipSeq) // spreadingLen
  symbolEst = np.zeros(nSymbols, dtype=np.complex128)
  symbolMask = np.zeros(nSymbols, dtype=bool)

  code = np.asarray(codes[userIdx, :], dtype=np.float64)

  for symIdx in range(nSymbols):
    start = symIdx * spreadingLen
    stop = start + spreadingLen

    block = chipSeq[start:stop]
    blockValid = validMask[start:stop]

    if len(block) != spreadingLen:
      continue

    if np.all(blockValid):
      symbolEst[symIdx] = np.sum(block * code)
      symbolMask[symIdx] = True

  return symbolEst, symbolMask


def computeUserSer(
  symbolsTrue: Array,
  symbolEst: Array,
  symbolMask: Array,
  userIdx: int,
) -> float:
  nSym = min(symbolsTrue.shape[0], len(symbolEst), len(symbolMask))

  valid = symbolMask[:nSym]

  if not np.any(valid):
    return float("nan")

  decided = hardQpsk(symbolEst[:nSym][valid])
  truth = symbolsTrue[:nSym, userIdx][valid]

  return float(np.mean(decided != truth))


def plotChannelTaps(
  hTrue: Array,
  fsSim: float,
  outDir: Path,
  prefix: str,
  show: bool,
) -> Path:
  outPath = outDir / f"{prefix}_channel_taps.png"

  n = hTrue.shape[0]
  timeMs = np.arange(n) / fsSim * 1000.0
  activeTaps = np.where(np.mean(np.abs(hTrue), axis=0) > 1e-8)[0]

  plt.figure(figsize=(11, 5))

  for tapIdx in activeTaps:
    plt.plot(
      timeMs,
      toDb20(np.abs(hTrue[:, tapIdx])),
      linewidth=1.1,
      label=f"tap {tapIdx}",
    )

  plt.title("Time-varying Jakes multipath channel taps")
  plt.xlabel("Time [ms]")
  plt.ylabel("Tap magnitude [dB]")
  plt.grid(True, linestyle=":", alpha=0.6)
  plt.legend(fontsize=8)
  plt.tight_layout()
  plt.savefig(outPath, dpi=150)

  if show:
    plt.show()
  else:
    plt.close()

  return outPath


def plotTrackingMetrics(
  metrics: Dict[str, Array],
  fsSim: float,
  outDir: Path,
  prefix: str,
  show: bool,
) -> Path:
  outPath = outDir / f"{prefix}_tracking_metrics.png"

  n = len(metrics["mse"])
  timeMs = np.arange(n) / fsSim * 1000.0
  winLen = max(16, min(1024, n // 50))

  plt.figure(figsize=(11, 6))

  plt.plot(
    timeMs,
    smoothCurve(metrics["mseDb"], winLen),
    linewidth=1.2,
    label="Channel ID MSE",
  )

  plt.plot(
    timeMs,
    smoothCurve(metrics["nmsdDb"], winLen),
    linewidth=1.2,
    label="Channel ID normalized MSD",
  )

  plt.title("SM-NLMS channel tracking")
  plt.xlabel("Time [ms]")
  plt.ylabel("Metric [dB]")
  plt.grid(True, linestyle=":", alpha=0.6)
  plt.legend()
  plt.tight_layout()
  plt.savefig(outPath, dpi=150)

  if show:
    plt.show()
  else:
    plt.close()

  return outPath


def plotEqualizerMetrics(
  eqMetrics: Dict[str, object],
  fsSim: float,
  outDir: Path,
  prefix: str,
  show: bool,
) -> Path:
  outPath = outDir / f"{prefix}_equalizer_metrics.png"

  eqMseDb = np.asarray(eqMetrics["eqMseDb"], dtype=np.float64)
  n = len(eqMseDb)
  timeMs = np.arange(n) / fsSim * 1000.0
  winLen = max(16, min(1024, n // 50))

  plt.figure(figsize=(11, 5))

  plt.plot(
    timeMs,
    smoothCurve(eqMseDb, winLen),
    linewidth=1.2,
    label="Equalizer training MSE",
  )

  plt.title("SM-NLMS linear equalizer")
  plt.xlabel("Time [ms]")
  plt.ylabel("MSE [dB]")
  plt.grid(True, linestyle=":", alpha=0.6)
  plt.legend()
  plt.tight_layout()
  plt.savefig(outPath, dpi=150)

  if show:
    plt.show()
  else:
    plt.close()

  return outPath


# =============================================================================
# Main simulation
# =============================================================================

def runSimulation(args: argparse.Namespace) -> Dict[str, object]:
  fd = args.mobileSpeed * args.carrierHz / 3.0e8

  chipRate = args.chipRate
  fsSim = chipRate * args.samplesPerChip

  source = DsCdmaSource(
    numUsers=args.numUsers,
    spreadingLen=args.spreadingLen,
    seed=args.seed,
    useWalshCodes=not args.randomCodes,
  )

  numSymbols = int(math.ceil(args.duration * chipRate / args.spreadingLen))
  xChip, symbols, codes, normScale = source.generate(numSymbols)

  nSamples = int(round(args.duration * fsSim))

  if args.samplesPerChip == 1:
    x = xChip[:nSamples]
  else:
    x = np.repeat(xChip, args.samplesPerChip)[:nSamples]

  x = x / math.sqrt(float(np.mean(np.abs(x) ** 2)) + 1e-15)

  if args.channel.upper() == "A":
    channel = MultipathJakesChannel.channelA(
      fsSim=fsSim,
      fd=fd,
      alpha1=args.alpha1,
      seed=args.seed + 2000,
    )
  elif args.channel.upper() == "B":
    channel = MultipathJakesChannel.channelB(
      fsSim=fsSim,
      fd=fd,
      alpha1=args.alpha1,
      seed=args.seed + 2000,
    )
  else:
    raise ValueError("channel must be A or B.")

  d, hTrue, noise = channel.filter(x, noiseVar=args.noiseVar)

  gammaBar = math.sqrt(args.gammaScale * args.noiseVar)

  tracker = SmNlmsChannelTracker(
    nCoef=hTrue.shape[1],
    gammaBar=gammaBar,
    eps=args.eps,
  )

  trackerOut = tracker.run(x, d)

  trackingMetrics = computeTrackingMetrics(
    hTrue=hTrue,
    wHist=trackerOut["wHist"],
    e=trackerOut["e"],
    updateFlags=trackerOut["updateFlags"],
  )

  equalizerDelay = args.equalizerDelay

  if equalizerDelay < 0:
    equalizerDelay = max(0, channel.delaySamples.max())

  equalizerGammaBar = math.sqrt(args.equalizerGammaScale * args.noiseVar)

  equalizer = SmNlmsLinearEqualizer(
    nCoef=args.equalizerLen,
    gammaBar=equalizerGammaBar,
    decisionDelay=equalizerDelay,
    eps=args.eps,
  )

  equalizerOut = equalizer.run(d, x)

  equalizerMetrics = computeEqualizerMetrics(
    xRef=x,
    equalizerOut=equalizerOut,
  )

  symbolEst, symbolMask = despreadUserSymbols(
    chipSeq=equalizerOut["xHatAligned"],
    validMask=equalizerOut["validMask"],
    codes=codes,
    userIdx=args.userIdx,
    spreadingLen=args.spreadingLen,
  )

  userSer = computeUserSer(
    symbolsTrue=symbols,
    symbolEst=symbolEst,
    symbolMask=symbolMask,
    userIdx=args.userIdx,
  )

  outDir = Path(args.outDir)
  outDir.mkdir(parents=True, exist_ok=True)

  prefix = args.prefix

  tapPlot = plotChannelTaps(
    hTrue=hTrue,
    fsSim=fsSim,
    outDir=outDir,
    prefix=prefix,
    show=not args.noShow,
  )

  trackingPlot = plotTrackingMetrics(
    metrics=trackingMetrics,
    fsSim=fsSim,
    outDir=outDir,
    prefix=prefix,
    show=not args.noShow,
  )

  equalizerPlot = plotEqualizerMetrics(
    eqMetrics=equalizerMetrics,
    fsSim=fsSim,
    outDir=outDir,
    prefix=prefix,
    show=not args.noShow,
  )

  lastStart = int(0.8 * len(trackingMetrics["nmsdDb"]))

  summary = dict(
    channel=args.channel.upper(),
    chipRate=chipRate,
    samplesPerChip=args.samplesPerChip,
    fsSim=fsSim,
    carrierHz=args.carrierHz,
    mobileSpeed=args.mobileSpeed,
    fd=fd,
    alpha1=args.alpha1,
    fsChannel=channel.fsChannel,
    duration=args.duration,
    nSamples=len(x),
    numUsers=args.numUsers,
    spreadingLen=args.spreadingLen,
    delaySamples=channel.delaySamples.tolist(),
    pathPowers=channel.pathPowers.tolist(),
    channelNCoef=hTrue.shape[1],
    noiseVar=args.noiseVar,
    gammaBar=gammaBar,
    trackingUpdateRate=trackingMetrics["updateRate"],
    meanNmsdDbLast20pct=float(np.mean(trackingMetrics["nmsdDb"][lastStart:])),
    meanMseDbLast20pct=float(np.mean(trackingMetrics["mseDb"][lastStart:])),
    equalizerLen=args.equalizerLen,
    equalizerDelay=equalizerDelay,
    equalizerGammaBar=equalizerGammaBar,
    equalizerUpdateRate=equalizerMetrics["updateRate"],
    equalizerChipMseDb=equalizerMetrics["chipMseDb"],
    equalizerEvmDb=equalizerMetrics["evmDb"],
    userIdx=args.userIdx,
    userSer=userSer,
    tapPlot=str(tapPlot),
    trackingPlot=str(trackingPlot),
    equalizerPlot=str(equalizerPlot),
  )

  print("\n" + "=" * 78)
  print("6.9.4 Wireless Channel Environment Simulation")
  print("=" * 78)
  print(f"Channel model:                 {summary['channel']}")
  print(f"Chip rate:                     {summary['chipRate']:.6g} chips/s")
  print(f"Samples per chip:              {summary['samplesPerChip']}")
  print(f"Simulation fs:                 {summary['fsSim']:.6g} samples/s")
  print(f"Carrier frequency:             {summary['carrierHz']:.6g} Hz")
  print(f"Mobile speed:                  {summary['mobileSpeed']:.6g} m/s")
  print(f"Maximum Doppler fd:            {summary['fd']:.6g} Hz")
  print(f"Channel fading fs:             {summary['fsChannel']:.6g} Hz")
  print(f"Duration:                      {summary['duration']:.6g} s")
  print(f"Number of samples:             {summary['nSamples']}")
  print(f"DS-CDMA users:                 {summary['numUsers']}")
  print(f"Spreading length:              {summary['spreadingLen']}")
  print(f"Delay samples:                 {summary['delaySamples']}")
  print(f"Channel identifier nCoef:      {summary['channelNCoef']}")
  print(f"noiseVar:                      {summary['noiseVar']:.6g}")
  print(f"Channel ID gammaBar:           {summary['gammaBar']:.6g}")
  print(f"Channel ID update rate:        {100.0 * summary['trackingUpdateRate']:.2f}%")
  print(f"Last 20% mean NMSD:            {summary['meanNmsdDbLast20pct']:.3f} dB")
  print(f"Last 20% mean MSE:             {summary['meanMseDbLast20pct']:.3f} dB")
  print("-" * 78)
  print(f"Equalizer length:              {summary['equalizerLen']}")
  print(f"Equalizer decision delay:      {summary['equalizerDelay']}")
  print(f"Equalizer gammaBar:            {summary['equalizerGammaBar']:.6g}")
  print(f"Equalizer update rate:         {100.0 * summary['equalizerUpdateRate']:.2f}%")
  print(f"Equalizer chip MSE:            {summary['equalizerChipMseDb']:.3f} dB")
  print(f"Equalizer EVM:                 {summary['equalizerEvmDb']:.3f} dB")
  print(f"User {summary['userIdx']} SER:                  {summary['userSer']:.6f}")
  print("-" * 78)
  print(f"Saved tap plot:                {summary['tapPlot']}")
  print(f"Saved tracking plot:           {summary['trackingPlot']}")
  print(f"Saved equalizer plot:          {summary['equalizerPlot']}")
  print("=" * 78)

  return summary


def buildArgParser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description="Diniz Section 6.9.4 style DS-CDMA + Jakes channel/equalizer simulation."
  )

  parser.add_argument("--channel", type=str, default="B", choices=["A", "B"])
  parser.add_argument("--duration", type=float, default=0.04)
  parser.add_argument("--chip-rate", dest="chipRate", type=float, default=3.84e6)
  parser.add_argument("--samples-per-chip", dest="samplesPerChip", type=int, default=1)

  parser.add_argument("--carrier-hz", dest="carrierHz", type=float, default=1.0e9)
  parser.add_argument("--mobile-speed", dest="mobileSpeed", type=float, default=30.0)
  parser.add_argument("--alpha1", type=float, default=10.0)

  parser.add_argument("--num-users", dest="numUsers", type=int, default=4)
  parser.add_argument("--spreading-len", dest="spreadingLen", type=int, default=16)
  parser.add_argument("--random-codes", dest="randomCodes", action="store_true")

  parser.add_argument("--noise-var", dest="noiseVar", type=float, default=1e-4)
  parser.add_argument("--gamma-scale", dest="gammaScale", type=float, default=5.0)
  parser.add_argument("--eps", type=float, default=1e-9)

  parser.add_argument("--equalizer-len", dest="equalizerLen", type=int, default=31)
  parser.add_argument(
    "--equalizer-delay",
    dest="equalizerDelay",
    type=int,
    default=-1,
    help="Equalizer decision delay. Use -1 to select max channel delay sample.",
  )
  parser.add_argument(
    "--equalizer-gamma-scale",
    dest="equalizerGammaScale",
    type=float,
    default=5.0,
  )
  parser.add_argument("--user-idx", dest="userIdx", type=int, default=0)

  parser.add_argument("--seed", type=int, default=1234)
  parser.add_argument("--out-dir", dest="outDir", type=str, default=".")
  parser.add_argument("--prefix", type=str, default="cdma_jakes")
  parser.add_argument("--no-show", dest="noShow", action="store_true")

  return parser


def main() -> None:
  parser = buildArgParser()
  args = parser.parse_args()
  runSimulation(args)


if __name__ == "__main__":
  main()