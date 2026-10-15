#!/usr/bin/env python3
"""Rigorous check of Pintz Lemma 7: |zeta(1/4 + it)| > e^-150 for |t| <= 4.
Covers [0, 4] by balls of radius 1/2^11 (zeta(1/4 - it) = conj zeta(1/4 + it))."""
import flint
from flint import arb, acb
flint.ctx.prec = 64
n = 2 ** 13
lo = None
for j in range(n):
    t = arb(4) * (2 * j + 1) / (2 * n)                 # centre
    t = arb(t.mid(), arb(4) / (2 * n))                 # ball covering [4j/n, 4(j+1)/n]
    m = abs(acb(arb("0.25"), t).zeta()).lower()
    lo = m if lo is None or m < lo else lo
print(f"min |zeta(1/4+it)| over |t| <= 4  >=  {lo.str(8)}  (> e^-150: {lo > (-arb(150)).exp()})")
