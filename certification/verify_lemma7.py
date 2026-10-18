#!/usr/bin/env python3
"""Rigorous check of Lemma 2.3 of the paper (a strong form of Pintz Lemma 7):

    |zeta(1/4 + it)| >= 0.4556   for |t| <= 4   (hence > e^-150).

Covers [0, 4] by 2^13 balls of radius 2^-11 (zeta(1/4 - it) = conj zeta(1/4 + it)).
Writes lemma7_certificate.json and exits nonzero unless every enclosure is finite and the
smallest lower bound exceeds THRESHOLD.

Usage: verify_lemma7.py [--threshold 0.4556] [--out lemma7_certificate.json]
"""
import argparse
import json
import os
import sys

import flint
from flint import acb, arb

from certio import ball, fail, provenance

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", default="0.4556")
    ap.add_argument("--out", default="lemma7_certificate.json")
    a = ap.parse_args()
    flint.ctx.prec = 64
    n = 2 ** 13
    lo = None
    for j in range(n):
        t = arb(4) * (2 * j + 1) / (2 * n)                 # centre
        t = arb(t.mid(), arb(4) / (2 * n))                 # ball covering [4j/n, 4(j+1)/n]
        z = acb(arb("0.25"), t).zeta()
        if not z.is_finite():
            fail(f"non-finite enclosure of zeta(1/4+it) on cell {j}")
        m = abs(z).lower()
        lo = m if lo is None or m < lo else lo
    thr = arb(a.threshold)
    ok = bool(lo > thr)
    weaker = bool(lo > (-arb(150)).exp())
    print(f"min |zeta(1/4+it)| over |t| <= 4  >=  {lo.str(10)}  (> {a.threshold}: {ok}; > e^-150: {weaker})")
    cert = provenance({"cells": n, "prec_bits": 64, "threshold": a.threshold}, [os.path.abspath(__file__)])
    cert.update({"min_abs_zeta_lower": ball(lo, 20), "threshold": a.threshold, "passed": ok and weaker})
    with open(os.path.join(HERE, a.out), "w") as f:
        json.dump(cert, f, indent=2)
    if not (ok and weaker):
        fail(f"Lemma 2.3 not certified: lower bound {lo.str(10)} <= {a.threshold}")


if __name__ == "__main__":
    main()
