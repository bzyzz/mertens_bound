# A smaller upper bound for the first counterexample to the Mertens conjecture

The Mertens conjecture states that |M(x)| < √x for all x > 1, where M(x) = Σ_{n≤x} μ(n). Odlyzko and te Riele disproved it in 1985, but their proof gives no location for a counterexample. Let 𝔵 be the smallest counterexample. Pintz (1987) gave the first explicit upper bound on 𝔵, and later work has lowered it.

This repository contains a computer-assisted proof of

$$\mathfrak{x} \;<\; \exp(3\,484\,971\,332\,733\,178\,124) \;\approx\; \exp(3.485\cdot 10^{18}),$$

an exponent about 5.6 times smaller than the previous record, exp(1.96·10¹⁹) (Kim & Nguyen).

> **Status: not peer-reviewed.** The argument extends a published method (Pintz; Saouter–te Riele) to new parameters, and every numerical step is certified with interval arithmetic. It still depends on the published lemmas listed under [What the result depends on](#what-the-result-depends-on), and the re-derived error bounds in [`THEOREM.md`](THEOREM.md) §3 have not yet been checked by a specialist. Please treat the bound as a claim under review.

## Previous bounds

| Work | Bound on 𝔵 |
|---|---|
| Pintz (1987) | exp(3.21·10⁶⁴) |
| Kotnik & te Riele (2006) | exp(1.59·10⁴⁰) |
| Saouter & te Riele (2014) | exp(1.004·10³³) |
| Rozmarynowycz & Kim (2023) | exp(1.017·10²⁹) |
| Kim & Nguyen (2024/25) | exp(1.96·10¹⁹) |
| **This repository** | **exp(3.485·10¹⁸)** |

## How the bound is obtained

All previous bounds since Pintz use the same criterion. Let

K₂(v, T, k) = 2 Σ_{0<γ<T} e^{−kγ²} Re( e^{iγv} / (ρ ζ′(ρ)) ),

summed over the zeros ρ = ½ + iγ of ζ. Pintz's argument, as refined by Saouter and te Riele, shows: if |K₂(u+k, T₀, k)| > 1 + E for an explicit error term E, then 𝔵 < exp(u + 2√(ku)). The step that converts a large value of K₂ into this bound is a contour-integral estimate. Its error terms depend on the Gaussian width k and the cutoff height T₀.

Earlier records used Saouter–te Riele's fixed parameters k = 3·10⁻⁹ and T₀ = 3·10⁵. At those parameters, our best candidate point (Peak2, found by the lattice search in `search/`) has K₂ = 0.9957, below the threshold of 1.

This work keeps the proof structure and changes the parameters:

1. **Smaller Gaussian width.** With k = 1.5·10⁻¹⁰, high zeros are damped less, and K₂ at Peak2 rises above 1.
2. **Error terms re-derived for the new k.** The error integrals J₁…J₆ are re-derived for general (k, T₀, T₁). They are small enough only once T₀ ≳ 1.27·10⁶. We take T₀ = 1.3·10⁶ and T₁ = 4.3·10⁶.
3. **Every zero below T₀ is included.** K₂ is summed over all 2,325,572 zeros below T₀, each computed from scratch with certified isolation in Arb. The zero count is the exact N(T₀).
4. **The bound on 1/ζ is recomputed.** The error term J₃ needs an upper bound for 1/ζ on the line Re s = ½ + 1/u. Saouter and te Riele computed one from their own database of zeros, which is not published. We recompute it on the range we need from 6.2 million certified zeros. The result agrees with theirs and is slightly sharper (constant 22.767 against their 23.05).

The full statement, proof and certified numbers are in **[`THEOREM.md`](THEOREM.md)**.

### Certified values

| Quantity | Value |
|---|---|
| Evaluation point v (Peak2, re-optimised for k = 1.5·10⁻¹⁰); u = v − k | 3484971332733132396.5383341489440461 |
| K₂(v, 1.3·10⁶, 1.5·10⁻¹⁰), all 2,325,572 zeros | [1.0003052504 ± 2.3·10⁻¹¹] |
| Error term E (J₁…J₆ and the smoothing tail) | ≤ 2.4·10⁻¹¹ |
| Margin \|K₂\| − 1 − E | ≥ 3.05·10⁻⁴ |
| Conclusion: 𝔵 < exp(u + 2√(ku)) | ≤ exp(3484971332733178124) |

## What the result depends on

The result rests on four things:

- **Pintz (1987), Theorem A, Corollary A and Lemmas 4–5.** These are published, but the lemmas are given only as proof sketches. The same lemmas underlie the Saouter–te Riele and Kim–Nguyen bounds. Pintz's Lemma 7 is stated without proof; we verify it numerically (`certification/verify_lemma7.py`).
- **Platt & Trudgian (2021).** RH has been verified, with simple zeros, up to height 3·10¹².
- **FLINT/Arb's certified zero routines** (`acb_dirichlet_zeta_zeros`, `zeta_nzeros`).
- **The derivations in [`THEOREM.md`](THEOREM.md) §3**, which still need a specialist's check.

## Reproducing the certificate

```bash
conda env create -f environment.yml && conda activate mertens   # python-flint >= 0.9 (FLINT 3)
make certify WORKERS=24
```

`make certify` runs four steps in `certification/`:

| Step | Script | What it does | Time |
|---|---|---|---|
| `lemma7` | `verify_lemma7.py` | certifies \|ζ(¼+it)\| ≥ 0.4556 for \|t\| ≤ 4 | seconds |
| `k2` | `certify_K2.py` | sums K₂ over all zeros below T₀ → `k2_certificate.json` | ~45 min on 24 cores |
| `table1` | `verify_table1.py` | recomputes the bound on 1/ζ over [T₀, T₁] → `table1_certificate.json` | ~50 min on 16 cores |
| `check` | `check_theorem.py` | combines both certificates with the error bounds in `error_bounds.py` and prints the conclusion | seconds |

The two long steps cache their work per batch, so they can be interrupted and resumed. `make clean-cache` forces a full recomputation. The certificate files from our run are committed, so `make check` reproduces the final verdict in seconds.

## Repository layout

```text
THEOREM.md            statement, proof, inputs, certified numbers, open items
certification/        rigorous scripts and certificates (everything needed to verify the bound)
search/               lattice (CVP) search for candidate points y; not needed to verify the bound
  *.cpp, *.cu         OpenMP/CUDA enumerators   (make search / make gpu)
  make_*.py, build_*  lattice basis generators
  eval_*.py, path1_*  candidate re-scoring on 100k–600k floating-point zeros
  logs/, results/     search logs; results/subrecord_peaks_summary.{csv,json} lists the top 25 candidates
kim_nguyen_ref/       reference code by P. Q. Nguyen (public domain), for comparison
legacy/               early prototypes; legacy/superseded/ holds an earlier, invalid certification attempt
```

### Notes on the search results

- `search/results/subrecord_peaks_summary.*` lists candidate points y below Kim–Nguyen's y = 1.957·10¹⁹. The values there come from truncated floating-point sums. Only Peak2 has been certified, and only through `certification/`.
- Later sweeps (`search/logs/tech1_*`, `tech2_*`, `tech3_*`) found no candidate better than Peak2.
- The large search inputs are regenerable and not tracked: zero tables, basis files and chunks (`make zeros`, `search/make_*.py`). Odlyzko's `zeros6.gz` can be downloaded from his [zeta tables page](https://www-users.cse.umn.edu/~odlyzko/zeta_tables/).

## Contributions

The project was directed by the repository author. Most of the code, derivations and text were produced by two AI models.

**Gemini 4 Argon** built the candidate search in `search/`:
- the OpenMP and CUDA lattice (CVP) enumerators and the lattice basis generators;
- refinement of the first 600,000 zeta zeros, and the re-scoring of candidates on them;
- the search that found the candidate point Peak2.

It also wrote the first certification attempt, now in `legacy/superseded/`. That attempt evaluated a smaller Gaussian width k at Peak2, but the error bound it used was not valid (see [`THEOREM.md`](THEOREM.md) §4).

**Claude Opus 5.5** turned the Peak2 observation into a proof:
- found that the first certification attempt was invalid.
- read the original Pintz and Saouter–te Riele papers and re-derived the error bounds J₁…J₆ for general (k, T₀, T₁). This showed that T₀ ≳ 1.27·10⁶ is needed at k = 1.5·10⁻¹⁰, and confirmed the Y = eᵘ reading of Pintz's Corollary A.
- wrote the scripts in `certification/`, including the certified sum over all 2,325,572 zeros, the recomputed bound on 1/ζ, and the numerical check of Pintz's Lemma 7.
- wrote [`THEOREM.md`](THEOREM.md).

Because these derivations were produced by an AI model, they need checking by a human specialist before the bound is cited. This is the main reason for the status note at the top.

## References

1. A. M. Odlyzko, H. J. J. te Riele, *Disproof of the Mertens conjecture*, J. reine angew. Math. 357 (1985), 138–160.
2. J. Pintz, *An effective disproof of the Mertens conjecture*, Astérisque 147–148 (1987), 325–333. [Numdam](https://www.numdam.org/item/AST_1987__147-148__325_0/)
3. T. Kotnik, H. te Riele, *The Mertens conjecture revisited*, ANTS-VII, LNCS 4076 (2006), 156–167.
4. Y. Saouter, H. te Riele, *Improved results on the Mertens conjecture*, Math. Comp. 83 (2014), 421–433.
5. J. Rozmarynowycz, S. Kim, *A new upper bound on the smallest counterexample to the Mertens conjecture*, [arXiv:2305.00345](https://arxiv.org/abs/2305.00345) (2023).
6. S. Kim, P. Q. Nguyen, *On counterexamples to the Mertens conjecture*, ANTS XVI (2024); [arXiv:2502.21021](https://arxiv.org/abs/2502.21021).
7. D. J. Platt, T. S. Trudgian, *The Riemann hypothesis is true up to 3·10¹²*, Bull. LMS 53 (2021), 792–797.
8. The FLINT team, *FLINT: Fast Library for Number Theory* (Arb ball arithmetic), https://flintlib.org.

## License

MIT, except `kim_nguyen_ref/`, which is public domain (see [`LICENSE`](LICENSE)).
