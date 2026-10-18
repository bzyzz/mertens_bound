#!/usr/bin/env python3
"""
Rigorous (Arb ball arithmetic) evaluation of the error term E(u, k, T0, T1) in the
Saouter-te Riele (2014) form of Pintz's effective disproof, for general (k, T0, T1).

    |M2(u) - K2(u + k, T0, k)| <= (|J1| + ... + |J6|) / pi           (StR (4.1))
    |M2(u) - M1(u)|            <= 2 exp(-u/2 + 2 sqrt(k u))           (Lemma F', see THEOREM.md)

Each J_i bound below is re-derived in THEOREM.md, Section 3; the inputs that are
*cited* rather than re-derived are Pintz's Lemmas I, J, K and StR Table 1 (M = 23.05).
All quantities are returned as natural logarithms (arb) so that J1 and J6, which are
of size exp(-10^17) and smaller, stay representable.
"""
import flint
from flint import arb

from certio import require

flint.ctx.dps = 60

C0 = arb(1).tan()                                   # c0 = tan(1) from Pintz Lemma I
M_STR = arb("23.05")                                # StR Table 1: log|1/zeta(1/2+1/u+it)| <= log u + 11 log t + 23.05
T_MAX_TABLE1 = arb("10379424.827854489")            # range of validity of StR Table 1


def log_add(a, b):
    """log(e^a + e^b) for arb a, b."""
    m = a if a > b else b
    return m + ((a - m).exp() + (b - m).exp()).log()


def log_J1(u, k, T0):
    # L1: s = 1/4 + it, 0 <= t <= T0. |1/s| <= 4, |exp(k(s^2-1/4) + u(s-1/2))| <= exp(-u/4) (k(s^2-1/4) has real part < 0).
    # t <= 4: Lemma K, |1/zeta| <= e^150.  t >= 4: Lemma J with v = 4 (zeros on the critical line up to T0).
    a = arb(4).log() + arb(4).log() + 150                       # 4 * 4 * e^150
    b = arb(4).log() + T0.log() + (C0 * arb(4).log() + 11) * T0.log() + 14
    return -u / 4 + log_add(a, b)


def log_J2(u, k, T0, delta):
    # L2: s = sigma + i T0, 1/4 <= sigma <= 1/2 + 1/u. Length <= 1, |1/s| <= 1,
    # |exp(...)| <= exp(-k T0^2 + 1 + k/u + k/u^2) <= exp(-k T0^2 + 2), Lemma J with v = 1/delta.
    return -k * T0 * T0 + C0 * T0.log() * (1 / delta).log() + 11 * T0.log() + 14 + 2


def log_J3(u, k, T0, T1, M=None):
    # L3: s = 1/2 + 1/u + it, T0 <= t <= T1. |exp(...)| = exp(k(1/u + 1/u^2 - t^2) + 1) <= exp(-k t^2 + 2),
    # |1/s| <= 1, StR Table 1: |1/zeta(s)| <= u t^11 e^M.  int_T0^inf t^11 e^{-k t^2} dt = Gamma(6, k T0^2) / (2 k^6).
    # M=None: StR Table 1 (valid up to T_MAX_TABLE1). Otherwise M must come from verify_table1.py on [T0, T1].
    if M is None:
        require(T1 <= T_MAX_TABLE1, "StR Table 1 is valid only up to 10379424.827854489")
        M = M_STR
    g = (k * T0 * T0).gamma_upper(arb(6))
    return u.log() + M + 2 + g.log() - arb(2).log() - 6 * k.log()


def log_J4(u, k, T1):
    # L4: s = sigma + i T1, 1/2 + 1/u <= sigma <= 1/2 + (5 log T1 + 3)/u. Length <= 1, |1/s| <= 1,
    # |exp(...)| <= exp(-k T1^2 + 1 + 5 log T1 + 3), Lemma J with v = u.
    require((5 * T1.log() + 3) / u < arb("0.5"), "hypothesis (5 log T1 + 3)/u < 1/2 fails")
    return -k * T1 * T1 + 5 * T1.log() + 4 + C0 * T1.log() * u.log() + 11 * T1.log() + 14


def log_J5(u, k, T1):
    # L5: s = 1/2 + (5 log t + 3)/u + it, T1 <= t <= exp(3u/10 - 3/5). |ds| <= 2 dt, |1/s| <= 1,
    # |exp(...)| <= exp(-k t^2 + 1 + 5 log t + 3), Lemma J with v = u  =>  integrand <= 2 e^18 exp(-k t^2 + A log t),
    # A = c0 log u + 16.  If k t^2 / 2 >= A log t on [T1, inf) the integrand is <= 2 e^18 exp(-k t^2 / 2).
    A = C0 * u.log() + 16
    require(T1 >= (A / k).sqrt(), "f(t) = k t^2/2 - A log t must be increasing on [T1, inf)")
    require(k / 2 * T1 * T1 >= A * T1.log(), "f(T1) >= 0 fails")
    return arb(2).log() + 18 - k * T1 * T1 / 2 - (k * T1).log()


def log_J6(u, k):
    # L6: s = 2 + it, t >= t6 = exp(3u/10 - 3/5). |1/s| <= 1, |1/zeta(2+it)| <= zeta(2)/zeta(4) = 15/pi^2 < 1.52,
    # |exp(...)| = exp(k(4 - t^2 - 1/4) + 3u/2) <= exp(-k t^2 + 1 + 3u/2).
    # int_{t6}^inf e^{-k t^2} dt <= e^{-k t6^2} / (2 k t6).
    log_t6 = 3 * u / 10 - arb("0.6")
    return arb("1.52").log() + 1 + 3 * u / 2 - k * (2 * log_t6).exp() - arb(2).log() - k.log() - log_t6


def log_lemmaF(u, k):
    # |M2(u) - M1(u)| <= 2 exp(-u/2 + 2 sqrt(k u))  (re-derived; implies StR Lemma F for k <= u/64)
    return arb(2).log() - u / 2 + 2 * (k * u).sqrt()


def check_hypotheses(u, k, T0, T1, delta):
    """The finite hypotheses of the criterion (paper Theorem 3.7), checked in ball arithmetic."""
    require(u > 16, "u > 16 fails")
    require(k > 0 and k < arb("0.25"), "0 < k < 1/4 fails")
    require(T0 >= 4 and T0 < T1, "4 <= T0 < T1 fails")
    require(T1 + C0 <= arb("3e12"), "T1 + c0 <= 3e12 (Platt-Trudgian range) fails")
    require(delta > 0 and delta <= 1, "0 < delta <= 1 fails")
    require(u >= k, "u >= k (Gaussian-tail lemma) fails")


def total_error(u, k, T0, T1, delta, verbose=True, M=None):
    """Upper bound (an arb) for E = (B1 + ... + B6)/pi + 2 exp(-u/2 + 2 sqrt(ku)). M must be an upper
    bound for the 1/zeta constant on [T0, T1] (or None for StR's 23.05, valid up to 10379424.83)."""
    check_hypotheses(u, k, T0, T1, delta)
    parts = {
        "J1": log_J1(u, k, T0),
        "J2": log_J2(u, k, T0, delta),
        "J3": log_J3(u, k, T0, T1, M),
        "J4": log_J4(u, k, T1),
        "J5": log_J5(u, k, T1),
        "J6": log_J6(u, k),
    }
    log_sumJ = None
    for v in parts.values():
        log_sumJ = v if log_sumJ is None else log_add(log_sumJ, v)
    log_E = log_add(log_sumJ - arb.pi().log(), log_lemmaF(u, k))
    if verbose:
        for name, v in parts.items():
            print(f"    log|{name}| <= {v.upper().str(8, radius=False)}")
        print(f"    log(2 e^(-u/2 + 2 sqrt(ku))) <= {log_lemmaF(u, k).upper().str(8, radius=False)}")
        print(f"    E = sum|J|/pi + 2e^(-u/2+2sqrt(ku)) <= {log_E.exp().upper().str(6, radius=False)}")
    return log_E.exp().upper()


if __name__ == "__main__":
    import sys
    k = arb(sys.argv[1]) if len(sys.argv) > 1 else arb("1.5e-10")
    v = arb("3484971332733132396.5383341489440461")
    u = v - k
    T1 = arb(10**7)
    for T0 in [1000000, 1200000, 1250000, 1300000]:
        print(f"k = {k.str(5)}, T0 = {T0}, T1 = 1e7, delta = 0.05:")
        total_error(u, k, arb(T0), T1, arb("0.05"))
