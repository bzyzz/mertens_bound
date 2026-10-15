#!/usr/bin/env python3
"""
Final step: combine the certificates and check the hypothesis of the Theorem in THEOREM.md,

    |K2(u + k, T0, k)| > 1 + E(u),   u = v - k,

then print the resulting bound  x_min < exp(u + 2 sqrt(k u)).

Inputs: k2_certificate.json (certify_K2.py) and table1_certificate.json (verify_table1.py).
"""
import json, os, sys
import flint
from flint import arb
import error_bounds as EB

HERE = os.path.dirname(os.path.abspath(__file__))
flint.ctx.dps = 60

k2 = json.load(open(os.path.join(HERE, "k2_certificate.json")))
t1 = json.load(open(os.path.join(HERE, "table1_certificate.json")))

k, v, T0 = arb(k2["k"]), arb(k2["v"]), arb(k2["T0"])
u = v - k
delta = arb(k2["delta_lower"]).lower()
K2 = arb(k2["K2"])
assert t1["T0"] == k2["T0"], "Table-1 recomputation must start at the same T0"
T1 = arb(t1["T1"])
M = arb(t1["M = 14 - min_N"])

print(f"k = {k2['k']}, v = {k2['v']}, T0 = {k2['T0']} (N(T0) = {k2['N_T0']}, delta >= {delta.str(8)}), T1 = {t1['T1']}, M = {t1['M = 14 - min_N']}")
print("Error terms:")
E = EB.total_error(u, k, T0, T1, delta, M=M)
margin = abs(K2) - 1 - E
print(f"|K2(v, T0, k)|    = {abs(K2).str(15)}")
print(f"|K2| - 1 - E     >= {margin.lower().str(10, radius=False)}")
if not margin > 0:
    sys.exit("FAIL: hypothesis of the Theorem not certified")
X = u + 2 * (k * u).sqrt()
bound = X.upper().ceil()
print(f"\nCERTIFIED (subject to the cited inputs in THEOREM.md section 2):")
print(f"  smallest counterexample < exp(u + 2 sqrt(k u)) <= exp({bound.str(30, radius=False).split('.')[0]})")
print(f"  exponent ratio vs Kim-Nguyen exp(1.9571878850562201959e19): {(arb('19571878850562201959.215107') / X).str(6)}")
