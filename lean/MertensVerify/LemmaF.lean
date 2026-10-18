import MertensVerify.LemmaE

/-!
# Tail bound (the role of Lemma F′ in THEOREM.md §3)

  `|M₂(u) - M₁(u)| ≤ 2 exp (u/2 + 3k/4 - 3 (D - 2k)² / (16 k))`   for `D ≥ 2k`.

Proof: `|M (exp w)| ≤ exp w`, and `exp w * g u k w = exp (u + k) * exp (-(w - b)² / (4k))` with
`b = u + 2k`. Off the window `|w - b| ≥ a := D - 2k`, where
`exp (-(w-b)²/(4k)) ≤ exp (-3a²/(16k)) * exp (-(w-b)²/(16k))`, and the last Gaussian integrates to
`2 √(4πk)` over `ℝ`. (THEOREM.md's Lemma F′ is sharper, via erfc; this weaker form needs only the full
Gaussian integral and is still `≈ exp (-u/4)` at the relevant `u`.)
-/

open Real MeasureTheory Set

namespace MertensVerify

variable {u k : ℝ}

theorem integrable_integrand (hk : 0 < k) :
    Integrable fun w : ℝ => (M (exp w) : ℝ) * g u k w := by
  refine Integrable.mono' (integrable_full_weight (u := u) hk)
    (measurable_integrand u k).aestronglyMeasurable (Filter.Eventually.of_forall fun w => ?_)
  rw [Real.norm_eq_abs, abs_mul, abs_of_pos (g_pos u k w)]
  exact mul_le_mul_of_nonneg_right (abs_M_le (exp_pos w).le) (g_pos u k w).le

theorem M₂_sub_M₁ (D : ℝ) (hk : 0 < k) :
    M₂ u k - M₁ u k D = c u k * ∫ w in (window u D)ᶜ, (M (exp w) : ℝ) * g u k w := by
  have hmeas : MeasurableSet (window u D) := measurableSet_Icc
  rw [M₂, M₁, ← integral_add_compl hmeas (integrable_integrand (u := u) hk)]
  ring

/-- Off the window the shifted Gaussian is dominated by a wider one times `exp (-3a²/(16k))`. -/
theorem gauss_off_window {D w : ℝ} (hk : 0 < k) (hD : 2 * k ≤ D) (hw : w ∈ (window u D)ᶜ) :
    exp (-(w - (u + 2 * k)) ^ 2 / (4 * k)) ≤
      exp (-3 * (D - 2 * k) ^ 2 / (16 * k)) * exp (-(w - (u + 2 * k)) ^ 2 / (4 * (4 * k))) := by
  rw [← exp_add]
  apply exp_le_exp.mpr
  have ha : 0 ≤ D - 2 * k := by linarith
  have hsq : (D - 2 * k) ^ 2 ≤ (w - (u + 2 * k)) ^ 2 := by
    simp only [window, mem_compl_iff, mem_Icc, not_and_or, not_le] at hw
    rcases hw with h | h
    · nlinarith
    · nlinarith
  rw [div_add_div _ _ (by positivity) (by positivity), div_le_div_iff₀ (by positivity) (by positivity)]
  have hk2 : k ^ 2 * (D - 2 * k) ^ 2 ≤ k ^ 2 * (w - (u + 2 * k)) ^ 2 :=
    mul_le_mul_of_nonneg_left hsq (sq_nonneg k)
  nlinarith [hk2]

/-- **Tail bound.** -/
theorem lemmaF {D : ℝ} (hk : 0 < k) (hD : 2 * k ≤ D) :
    |M₂ u k - M₁ u k D| ≤ 2 * exp (u / 2 + 3 * k / 4 - 3 * (D - 2 * k) ^ 2 / (16 * k)) := by
  set b := u + 2 * k
  set A := exp (-3 * (D - 2 * k) ^ 2 / (16 * k))
  have hc := c_pos (u := u) hk
  have hint := integrable_integrand (u := u) hk
  have hG : Integrable fun w : ℝ => exp (-(w - b) ^ 2 / (4 * (4 * k))) :=
    integrable_gauss b (by positivity)
  rw [M₂_sub_M₁ D hk, abs_mul, abs_of_pos hc]
  -- |∫_{off} M g| ≤ ∫_{off} exp w * g = exp (u+k) ∫_{off} exp (-(w-b)²/(4k)) ≤ exp (u+k) A ∫ G
  have h1 : |∫ w in (window u D)ᶜ, (M (exp w) : ℝ) * g u k w|
      ≤ ∫ w in (window u D)ᶜ, exp (u + k) * (A * exp (-(w - b) ^ 2 / (4 * (4 * k)))) := by
    refine (abs_integral_le_integral_abs).trans ?_
    refine setIntegral_mono_on hint.integrableOn.abs
      ((hG.const_mul A).const_mul (exp (u + k))).integrableOn measurableSet_Icc.compl fun w hw => ?_
    rw [abs_mul, abs_of_pos (g_pos u k w)]
    calc |(M (exp w) : ℝ)| * g u k w ≤ exp w * g u k w :=
          mul_le_mul_of_nonneg_right (abs_M_le (exp_pos w).le) (g_pos u k w).le
      _ = exp (u + k) * exp (-(w - b) ^ 2 / (4 * k)) := full_weight u k w hk.ne'
      _ ≤ exp (u + k) * (A * exp (-(w - b) ^ 2 / (4 * (4 * k)))) :=
          mul_le_mul_of_nonneg_left (gauss_off_window hk hD hw) (exp_pos _).le
  have h2 : ∫ w in (window u D)ᶜ, exp (u + k) * (A * exp (-(w - b) ^ 2 / (4 * (4 * k))))
      ≤ ∫ w, exp (u + k) * (A * exp (-(w - b) ^ 2 / (4 * (4 * k)))) :=
    setIntegral_le_integral ((hG.const_mul A).const_mul _)
      (Filter.Eventually.of_forall fun w => by positivity)
  have h3 : ∫ w, exp (u + k) * (A * exp (-(w - b) ^ 2 / (4 * (4 * k))))
      = exp (u + k) * (A * (2 * √(π * (4 * k)))) := by
    rw [integral_const_mul, integral_const_mul, integral_gauss b (by positivity)]
  have h4 : c u k * (exp (u + k) * (A * (2 * √(π * (4 * k)))))
      = 2 * exp (u / 2 + 3 * k / 4 - 3 * (D - 2 * k) ^ 2 / (16 * k)) := by
    have hs : √(π * (4 * k)) = 2 * √(π * k) := by
      rw [show π * (4 * k) = 4 * (π * k) by ring, sqrt_mul (by norm_num : (0 : ℝ) ≤ 4),
        show (4 : ℝ) = 2 ^ 2 by norm_num, sqrt_sq (by norm_num : (0 : ℝ) ≤ 2)]
    have hpos : 0 < √(π * k) := sqrt_pos.mpr (by positivity)
    rw [hs, c]
    have he : exp (-k / 4 - u / 2) * exp (u + k) * A =
        exp (u / 2 + 3 * k / 4 - 3 * (D - 2 * k) ^ 2 / (16 * k)) := by
      simp only [A, ← exp_add]
      congr 1
      ring
    calc exp (-k / 4 - u / 2) / (2 * √(π * k)) * (exp (u + k) * (A * (2 * (2 * √(π * k)))))
        = 2 * (exp (-k / 4 - u / 2) * exp (u + k) * A) * (√(π * k) / √(π * k)) := by ring
      _ = 2 * exp (u / 2 + 3 * k / 4 - 3 * (D - 2 * k) ^ 2 / (16 * k)) := by
          rw [he, div_self hpos.ne', mul_one]
  calc c u k * |∫ w in (window u D)ᶜ, (M (exp w) : ℝ) * g u k w|
      ≤ c u k * ∫ w in (window u D)ᶜ, exp (u + k) * (A * exp (-(w - b) ^ 2 / (4 * (4 * k)))) :=
        mul_le_mul_of_nonneg_left h1 hc.le
    _ ≤ c u k * ∫ w, exp (u + k) * (A * exp (-(w - b) ^ 2 / (4 * (4 * k)))) :=
        mul_le_mul_of_nonneg_left h2 hc.le
    _ = 2 * exp (u / 2 + 3 * k / 4 - 3 * (D - 2 * k) ^ 2 / (16 * k)) := by rw [h3, h4]

end MertensVerify
