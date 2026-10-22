import MertensVerify.LemmaF

/-!
# The theorem of THEOREM.md, from its cited and certified inputs

Everything below is proved in Lean from Mathlib, **except** four hypotheses, each a precise statement
about Mathlib objects (`riemannZeta`, the Lebesgue integral `M₂`, real numbers):

* `hC` **[cited]** Pintz 1987, Theorem A and Corollary A (= Saouter–te Riele Thm C / Cor D), with
  `Y = exp u` and `ε = 1`: if some zero has `β > 1/2 + (5 log γ + 1)/u`, then `|M(x)| > e √x` for some
  `x ≤ exp u`.
* `hcontour` **[cited + derived]** the residue theorem on the Saouter–te Riele contour (THEOREM.md §3,
  Lemma G, the contour `L₁…L₆`, Pintz Lemmas 4–5, Pintz Lemma 7, Table 1 on `[T₀, T₁]`,
  Platt–Trudgian): if no zero is that far right, then `|M₂(u) - K| ≤ J`, where `K = K₂(u+k, T₀, k)`
  and `J = (B₁ + ⋯ + B₆) / π`, the explicit bounds of the paper's Lemma 3.5 (so that
  `|M₂ - K| ≤ Σ Jᵢ / π ≤ Σ Bᵢ / π = J` in that case).
* `hK` **[certified with Arb]** `certification/k2_certificate.json`: `|K| ≥ 1.0003052504 - 2.3·10⁻¹¹`.
* `hJ` **[certified with Arb]** `certification/error_bounds.py`: `J ≤ 2.40·10⁻¹¹` (computed value ≈ 2.39071·10⁻¹¹; 2.40·10⁻¹¹ is the bound used).

Proved here: Lemma E, a tail bound in place of Lemma F′, the case split, and all arithmetic (kernel
checked with `norm_num`). The main theorem's dependency closure contains no `native_decide`; the separate `Basic.lean` does use it. The window is `|log x - u| ≤ 2s` with the rational
`s > √(ku)` of `Basic.lean`, so the conclusion is the same as THEOREM.md's.
-/

open Real

namespace MertensVerify.Main

open MertensVerify

/-- `v = u + k` from THEOREM.md §1 (the evaluation point of `K₂`). -/
noncomputable def v₀ : ℝ := 34849713327331323965383341489440461 / 10000000000000000
/-- `k = 1.5·10⁻¹⁰`. -/
noncomputable def k₀ : ℝ := 3 / 20000000000
/-- `u = v - k`. -/
noncomputable def u₀ : ℝ := v₀ - k₀
/-- A rational `s` with `s² > k u` (so `√(ku) < s`). -/
noncomputable def s₀ : ℝ := 1143181634639 / 50000000
/-- Window half-width `D = 2 s`. -/
noncomputable def D₀ : ℝ := 2 * s₀
/-- Lower end of the certified enclosure of `K₂`. -/
noncomputable def K2Lower : ℝ := 10003052504 / 10000000000 - 23 / 1000000000000
/-- Certified upper bound for `J = Σ Bᵢ / π`. -/
noncomputable def JUpper : ℝ := 24 / 1000000000000

/-- Case 1 of the dichotomy: a zero of `ζ` lies to the right of `1/2 + (5 log γ + 1)/u`. -/
def Case1 (u : ℝ) : Prop :=
  ∃ ρ : ℂ, riemannZeta ρ = 0 ∧ 0 < ρ.im ∧ 1 / 2 + (5 * Real.log ρ.im + 1) / u < ρ.re

theorem k₀_pos : 0 < k₀ := by norm_num [k₀]

theorem two_k_le_D : 2 * k₀ ≤ D₀ := by norm_num [k₀, D₀, s₀]

theorem s_sq_gt : s₀ ^ 2 > k₀ * u₀ := by norm_num [s₀, k₀, u₀, v₀]

theorem u_add_D_lt : u₀ + D₀ < 3484971332733178124 := by norm_num [u₀, v₀, k₀, D₀, s₀]

theorem u_sub_D_pos : 0 < u₀ - D₀ := by norm_num [u₀, v₀, k₀, D₀, s₀]

/-- The exponent of the tail bound is below `-10¹⁷`. -/
theorem tail_exponent :
    3 * (D₀ - 2 * k₀) ^ 2 / (16 * k₀) - u₀ / 2 - 3 * k₀ / 4 ≥ 10 ^ 17 := by
  norm_num [u₀, v₀, k₀, D₀, s₀]

theorem tail_small : 2 * exp (u₀ / 2 + 3 * k₀ / 4 - 3 * (D₀ - 2 * k₀) ^ 2 / (16 * k₀)) ≤ 1 / 10 ^ 16 := by
  set X := 3 * (D₀ - 2 * k₀) ^ 2 / (16 * k₀) - u₀ / 2 - 3 * k₀ / 4
  have hX := tail_exponent
  have h1 : u₀ / 2 + 3 * k₀ / 4 - 3 * (D₀ - 2 * k₀) ^ 2 / (16 * k₀) = -X := by simp only [X]; ring
  rw [h1, exp_neg]
  have h2 : X + 1 ≤ exp X := add_one_le_exp X
  have h3 : (0 : ℝ) < X + 1 := by linarith
  have h4 : (exp X)⁻¹ ≤ (X + 1)⁻¹ := inv_anti₀ h3 h2
  have h5 : (X + 1)⁻¹ ≤ (10 ^ 17 + 1 : ℝ)⁻¹ := inv_anti₀ (by positivity) (by linarith)
  have h6 : 2 * (10 ^ 17 + 1 : ℝ)⁻¹ ≤ 1 / 10 ^ 16 := by norm_num
  linarith

theorem margin : K2Lower - JUpper - 1 / 10 ^ 16 > 1 := by norm_num [K2Lower, JUpper]

/-- **Theorem.** Under the cited and certified inputs, `𝔵 < exp 3484971332733178124`: some
`1 ≤ x < exp 3484971332733178124` has `|M(x)| > √x`. -/
theorem smallest_counterexample_lt
    (hC : Case1 u₀ → ∃ x : ℝ, x ≤ exp u₀ ∧ exp 1 * √x < |(M x : ℝ)|)
    (K J : ℝ)
    (hcontour : ¬ Case1 u₀ → |M₂ u₀ k₀ - K| ≤ J)
    (hK : K2Lower ≤ |K|)
    (hJ : J ≤ JUpper) :
    ∃ x : ℝ, 1 ≤ x ∧ x < exp 3484971332733178124 ∧ Violates x := by
  have hu : u₀ < 3484971332733178124 := by
    have h1 := u_add_D_lt
    have h2 : 0 < D₀ := by norm_num [D₀, s₀]
    linarith
  by_cases h1 : Case1 u₀
  · -- Case 1: Pintz's Theorem A gives a violation below exp u.
    obtain ⟨x, hx, hv⟩ := hC h1
    have hviol : Violates x := by
      unfold Violates
      have : √x ≤ exp 1 * √x := le_mul_of_one_le_left (sqrt_nonneg x) (one_le_exp (by norm_num))
      linarith
    exact ⟨x, one_le_of_violates hviol, lt_of_le_of_lt hx (exp_lt_exp.mpr hu), hviol⟩
  · -- Case 2: |M₁| > 1, so Lemma E gives a violation in the window.
    have hM₂ : K2Lower - JUpper ≤ |M₂ u₀ k₀| := by
      have := hcontour h1
      have := abs_sub_abs_le_abs_sub K (M₂ u₀ k₀)
      rw [abs_sub_comm K] at this
      linarith
    have hF := lemmaF (u := u₀) k₀_pos two_k_le_D
    have hT := tail_small
    have hM₁ : 1 ≤ |M₁ u₀ k₀ D₀| := by
      have := abs_sub_abs_le_abs_sub (M₂ u₀ k₀) (M₁ u₀ k₀ D₀)
      have := margin
      linarith
    obtain ⟨w, hw, hv⟩ := exists_violation_of_M₁ k₀_pos hM₁
    refine ⟨exp w, one_le_of_violates hv, ?_, hv⟩
    exact exp_lt_exp.mpr (lt_of_le_of_lt hw.2 u_add_D_lt)

end MertensVerify.Main
