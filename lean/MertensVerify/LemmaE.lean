import MertensVerify.Gaussian

/-!
# Lemma E (THEOREM.md §3)

If `|M(x)| ≤ √x` whenever `|log x - u| ≤ D`, then `|M₁(u)| < 1`.

With `x = exp w`, the hypothesis reads `|M (exp w)| ≤ exp (w/2)` on the window. The weight
`c * exp (w/2) * g` integrates to `1` over `ℝ` (`c_mul_integral_half_weight`) and is positive,
so its mass on the window is `< 1`.
-/

open Real MeasureTheory Set

namespace MertensVerify

variable {u k : ℝ}

theorem measurable_integrand (u k : ℝ) :
    Measurable fun w : ℝ => (M (exp w) : ℝ) * g u k w := by
  refine measurable_M_exp.mul ?_
  unfold g
  fun_prop

theorem integrableOn_window {D : ℝ} (hk : 0 < k)
    (h : ∀ w ∈ window u D, |(M (exp w) : ℝ)| ≤ exp (w / 2)) :
    IntegrableOn (fun w => (M (exp w) : ℝ) * g u k w) (window u D) := by
  refine Integrable.mono' (integrable_half_weight (u := u) hk).integrableOn
    (measurable_integrand u k).aestronglyMeasurable ?_
  refine ae_restrict_of_forall_mem measurableSet_Icc fun w hw => ?_
  rw [Real.norm_eq_abs, abs_mul, abs_of_pos (g_pos u k w)]
  exact mul_le_mul_of_nonneg_right (h w hw) (g_pos u k w).le

/-- The weight has positive mass outside any window. -/
theorem integral_compl_window_pos (D : ℝ) (hk : 0 < k) :
    0 < ∫ w in (window u D)ᶜ, exp (w / 2) * g u k w := by
  have hint := (integrable_half_weight (u := u) hk).integrableOn (s := (window u D)ᶜ)
  rw [setIntegral_pos_iff_support_of_nonneg_ae
    (Filter.Eventually.of_forall fun w => (mul_pos (exp_pos _) (g_pos u k w)).le) hint]
  have hsupp : Function.support (fun w : ℝ => exp (w / 2) * g u k w) = univ := by
    ext w; simp [Function.mem_support, (g_pos u k w).ne']
  rw [hsupp, univ_inter]
  have hsub : Ioi (u + D) ⊆ (window u D)ᶜ := by
    intro w hw hw'
    exact absurd hw'.2 (not_le.mpr hw)
  have : volume (Ioi (u + D)) ≤ volume (window u D)ᶜ := measure_mono hsub
  rw [Real.volume_Ioi] at this
  exact lt_of_lt_of_le (by simp) this

/-- **Lemma E.** -/
theorem lemmaE {D : ℝ} (hk : 0 < k)
    (h : ∀ w ∈ window u D, |(M (exp w) : ℝ)| ≤ exp (w / 2)) : |M₁ u k D| < 1 := by
  set f := fun w : ℝ => (M (exp w) : ℝ) * g u k w
  set F := fun w : ℝ => exp (w / 2) * g u k w
  have hF : Integrable F := integrable_half_weight hk
  have hf : IntegrableOn f (window u D) := integrableOn_window hk h
  -- |∫_win f| ≤ ∫_win F
  have h1 : |∫ w in window u D, f w| ≤ ∫ w in window u D, F w := by
    refine (abs_integral_le_integral_abs).trans ?_
    refine setIntegral_mono_on hf.abs hF.integrableOn measurableSet_Icc fun w hw => ?_
    simp only [f, F, abs_mul, abs_of_pos (g_pos u k w)]
    exact mul_le_mul_of_nonneg_right (h w hw) (g_pos u k w).le
  -- ∫_win F < ∫ F
  have hmeas : MeasurableSet (window u D) := measurableSet_Icc
  have h2 : ∫ w in window u D, F w < ∫ w, F w := by
    rw [← integral_add_compl hmeas hF]
    have hpos : 0 < ∫ w in (window u D)ᶜ, F w := integral_compl_window_pos (u := u) D hk
    linarith
  have hc := c_pos (u := u) hk
  have h3 : c u k * ∫ w, F w = 1 := c_mul_integral_half_weight hk
  rw [M₁, abs_mul, abs_of_pos hc]
  calc c u k * |∫ w in window u D, f w| ≤ c u k * ∫ w in window u D, F w :=
        mul_le_mul_of_nonneg_left h1 hc.le
    _ < c u k * ∫ w, F w := mul_lt_mul_of_pos_left h2 hc
    _ = 1 := h3

/-- Contrapositive form used in the proof of the theorem: if `|M₁(u)| ≥ 1` then some `x` in the
window violates the Mertens inequality. -/
theorem exists_violation_of_M₁ {D : ℝ} (hk : 0 < k) (h : 1 ≤ |M₁ u k D|) :
    ∃ w ∈ window u D, Violates (exp w) := by
  by_contra hne
  push_neg at hne
  refine absurd (lemmaE hk fun w hw => ?_) (not_lt.mpr h)
  have := hne w hw
  unfold Violates at this
  push_neg at this
  rwa [sqrt_eq_rpow, ← exp_mul, show w * (1 / 2) = w / 2 by ring] at this

end MertensVerify
