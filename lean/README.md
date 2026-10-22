# mertens_verify

A Lean 4 + Mathlib (v4.35.0) check of the deduction in [THEOREM.md](../THEOREM.md): from its cited
and Arb-certified inputs to the conclusion 𝔵 < exp(3484971332733178124).

```
lake exe cache get && lake build
lake env lean AxiomCheck.lean     # prints the axioms each theorem depends on
```

## The main theorem

`MertensVerify.Main.smallest_counterexample_lt` (in `MertensVerify/Main.lean`):

```lean
theorem smallest_counterexample_lt
    (hC : Case1 u₀ → ∃ x : ℝ, x ≤ exp u₀ ∧ exp 1 * √x < |(M x : ℝ)|)
    (K J : ℝ)
    (hcontour : ¬ Case1 u₀ → |M₂ u₀ k₀ - K| ≤ J)
    (hK : K2Lower ≤ |K|)
    (hJ : J ≤ JUpper) :
    ∃ x : ℝ, 1 ≤ x ∧ x < exp 3484971332733178124 ∧ Violates x
```

All the objects are genuine Mathlib objects:

- `M x = ∑_{1 ≤ n ≤ x} μ(n)`, using `ArithmeticFunction.moebius`;
- `Violates x` means `√x < |M x|`;
- `M₂` is the Lebesgue integral of THEOREM.md §3, in the variable w = log x;
- `Case1 u` says: some ρ with `riemannZeta ρ = 0` and Im ρ > 0 has Re ρ > ½ + (5 log Im ρ + 1)/u.

`#print axioms` reports only `propext`, `Classical.choice` and `Quot.sound` for the main theorem (and for `lemmaE`, `lemmaF`). Its dependency closure contains no `sorry` and no `native_decide`. These standard axioms do not discharge the four hypotheses below, which remain inputs.

## What is proved in Lean, and what is assumed

| Step | Status |
|---|---|
| \|M(x)\| ≤ x, M(x) = 0 for x < 1, measurability (`Mobius.lean`) | proved |
| Gaussian completion of squares; ∫ exp(−(w−a)²/4k) = 2√(πk); normalisation of the weight (`Gaussian.lean`) | proved |
| **Lemma E**: \|M(x)\| ≤ √x on the window ⇒ \|M₁\| < 1 (`LemmaE.lean`) | proved |
| **Tail bound**: \|M₂ − M₁\| ≤ 2 exp(u/2 + 3k/4 − 3(D−2k)²/16k) (`LemmaF.lean`). This is weaker than THEOREM.md's Lemma F′ but enough here: it is ≤ 10⁻¹⁶. | proved |
| Case analysis, window arithmetic, all numerical inequalities (`Main.lean`, kernel-checked `norm_num`) | proved |
| `hC`: Pintz 1987 Thm A + Cor. A with Y = eᵘ, ε = 1 | **hypothesis** (cited) |
| `hcontour`: residue theorem on the Saouter–te Riele contour with the bounds Jᵢ ≤ Bᵢ (Lemma G; Pintz Lemmas 4, 5, 7; Table 1 on [T₀, T₁]; Platt–Trudgian). Here K = K₂(u+k, T₀, k) and J = (B₁+⋯+B₆)/π, the explicit bounds of the paper's Lemma 3.5. | **hypothesis** (cited + derived on paper) |
| `hK`: \|K₂\| ≥ 1.0003052504 − 2.3·10⁻¹¹ | **hypothesis** (Arb certificate `k2_certificate.json`) |
| `hJ`: J = ΣBᵢ/π ≤ 2.40·10⁻¹¹ (computed value ≈ 2.39071·10⁻¹¹; the hypothesis uses the safe bound 2.40·10⁻¹¹) | **hypothesis** (Arb, `error_bounds.py`) |

So Lean certifies that the paper's conclusion follows from these four inputs. It does not certify the inputs. Also not formalized: the identification of K and J above, and of Lean's `M₂` (written in w = log x) and `Case1` with the paper's M₂ and ¬Q(u).

Formalizing them would need, in Lean: Perron/Mellin inversion for 1/(sζ(s)), contour integration with residues along the StR contour, Pintz's bounds on 1/ζ, and a verified evaluation of ζ′ at 2.3 million zeros. That is far beyond current Mathlib.

The window here is |log x − u| ≤ 2s, with the rational s > √(ku) from `Basic.lean`, rather than 2√(ku). The upper end u + 2s < 3484971332733178124 is the same one THEOREM.md uses, so the conclusion is identical.

`Basic.lean`, from an earlier session, checks the §6 arithmetic over `Rat` with `native_decide`. The main theorem does not depend on it: `Main.lean` re-proves those inequalities over ℝ with `norm_num`.
