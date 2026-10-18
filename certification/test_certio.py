#!/usr/bin/env python3
"""Regression checks for the failure modes in the 2026-10-18 review (C1-C3). Run: python test_certio.py"""
import json
import os
import subprocess
import sys
import tempfile

from flint import arb

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from certio import CertificationError, ball, exact_int, legacy_ball  # noqa: E402


def expect_error(f, *a):
    try:
        f(*a)
    except CertificationError:
        return
    raise AssertionError(f"{f.__name__}{a} did not fail")


def main():
    # C1: radius-free decimals are widened outward; balls round-trip as enclosures
    x = arb("-8.76706559692302356")
    b = legacy_ball(x.str(12, radius=False))
    if not (b.lower() < x < b.upper()):
        raise AssertionError("legacy_ball does not enclose the true value")
    if not arb(ball(x, 12)).contains(x):
        raise AssertionError("ball string does not enclose")
    # C3: counts must be unique integers
    if exact_int(arb(2325572), "N") != 2325572:
        raise AssertionError("exact_int")
    expect_error(exact_int, arb(arb("2325572.5"), arb("0.6")), "N")
    # C3: a failing Lemma 2.3 threshold exits nonzero
    with tempfile.TemporaryDirectory() as d:
        r = subprocess.run([sys.executable, os.path.join(HERE, "verify_lemma7.py"), "--threshold", "0.5",
                            "--out", os.path.join(d, "l7.json")], capture_output=True, text=True)
        if r.returncode == 0:
            raise AssertionError("verify_lemma7 with an unattainable threshold exited 0")
    # C2: a cached K2 payload whose key does not match is rejected
    import certify_K2 as C
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "00000001_250_45.json")
        json.dump({"tag": "other", "k": "1.5e-10", "v": "0", "n0": 1, "num": 250, "dps": 45}, open(path, "w"))
        expect_error(C.work, (1, 250, 45, "1.5e-10", "0", d, "this-tag", False))
    print("all regression checks passed")


if __name__ == "__main__":
    main()
