#!/usr/bin/env python3
"""
Final step: combine the certificates and check the hypothesis of the criterion (paper Theorem 3.7,
THEOREM.md),

    |K2(u + k, T0, k)| > 1 + E(u),   u = v - k,

then print the resulting bound  x_min < exp(u + 2 sqrt(k u)).

Inputs: k2_certificate.json or another certify_K2.py --out file (argv[1]), table1_certificate.json
(or argv[2]) and lemma7_certificate.json. Schema-2 certificates store enclosing balls. Schema-1
certificates (initial release) are accepted with their radius-free decimals widened outward by
one unit in the last place (certio.legacy_ball). Every failed check exits nonzero.
"""
import json
import os
import sys

import flint
from flint import arb

import error_bounds as EB
from certio import CertificationError, fail, legacy_ball, require

HERE = os.path.dirname(os.path.abspath(__file__))
flint.ctx.dps = 60


def load(name):
    path = os.path.join(HERE, name)
    require(os.path.exists(path), f"missing certificate {name}")
    with open(path) as f:
        return json.load(f)


def table1_M(t1):
    """Upper bound for the 1/zeta constant M on [T0, T1]."""
    if t1.get("schema") == 2:
        return arb(t1["M_ball"]).upper()
    # schema 1: radius-free decimals; take the more conservative of the two stored fields
    from_M = legacy_ball(t1["M = 14 - min_N"]).upper()
    from_N = (14 - legacy_ball(t1["min_N"]).lower()).upper()
    return from_M if from_M > from_N else from_N


def main():
    k2_file = sys.argv[1] if len(sys.argv) > 1 else "k2_certificate.json"
    t1_file = sys.argv[2] if len(sys.argv) > 2 else "table1_certificate.json"
    k2, t1, l7 = load(k2_file), load(t1_file), load("lemma7_certificate.json")

    require(l7.get("passed") is True, "Lemma 2.3 certificate does not record a pass")
    l7_lo = arb(l7["min_abs_zeta_lower"]).lower()
    require(l7_lo > arb(l7["threshold"]) and l7_lo > (-arb(150)).exp(), "Lemma 2.3 bound insufficient")
    require(t1["T0"] == k2["T0"], "Table-1 recomputation must start at the same T0")
    for c, name in ((k2, k2_file), (t1, t1_file)):
        if c.get("schema") == 2 and l7.get("schema") == 2:
            require(c["versions"]["flint"] == l7["versions"]["flint"], f"{name}: FLINT version differs from Lemma 2.3 run")

    k, v, T0 = arb(k2["k"]), arb(k2["v"]), arb(k2["T0"])
    u = v - k
    delta = arb(k2["delta_lower"]).lower()
    K2 = arb(k2["K2"])
    require(K2.is_finite(), "K2 enclosure not finite")
    T1 = arb(t1["T1"])
    M = table1_M(t1)
    schemas = (k2.get("schema", 1), t1.get("schema", 1), l7.get("schema", 1))

    print(f"certificate schemas (K2, Table 1, Lemma 2.3) = {schemas}")
    print(f"k = {k2['k']}, v = {k2['v']}, T0 = {k2['T0']} (N(T0) = {k2['N_T0']}, delta >= {delta.str(8)}), "
          f"T1 = {t1['T1']}, M <= {M.str(15, radius=False)}, Lemma 2.3: |zeta(1/4+it)| >= {l7_lo.str(8)}")
    print("Error terms:")
    E = EB.total_error(u, k, T0, T1, delta, M=M)          # also checks the finite hypotheses of the criterion
    margin = abs(K2) - 1 - E
    print(f"|K2(v, T0, k)|    = {abs(K2).str(15)}")
    print(f"|K2| - 1 - E     >= {margin.lower().str(10, radius=False)}")
    require(margin > 0, "hypothesis |K2| > 1 + E of the criterion not certified")
    X = u + 2 * (k * u).sqrt()
    bound = X.upper().ceil()
    print(f"\nCERTIFIED (subject to the cited inputs in THEOREM.md section 2):")
    print(f"  smallest counterexample < exp(u + 2 sqrt(k u)) <= exp({bound.str(30, radius=False).split('.')[0]})")
    print(f"  exponent ratio vs Kim-Nguyen exp(1.9571878850562201959e19): {(arb('19571878850562201959.215107') / X).str(6)}")


if __name__ == "__main__":
    try:
        main()
    except CertificationError as e:
        fail(str(e))
