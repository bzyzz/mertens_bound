import MertensVerify.Mobius

/-!
# Gaussian identities

Completing the square,
  `exp (w/2) * g u k w = exp (u/2 + k/4) * exp (-(w - (u+k))² / (4k))`,
  `exp w     * g u k w = exp (u + k)     * exp (-(w - (u+2k))² / (4k))`,
and `∫ exp (-(w - a)² / (4k)) dw = 2 √(π k)`. Hence `c u k * ∫ exp (w/2) * g u k w = 1`, the
normalisation behind Lemma E.
-/

open Real MeasureTheory

namespace MertensVerify

variable {u k : ℝ}

theorem gauss_fun_eq (a k : ℝ) :
    (fun w : ℝ => exp (-(w - a) ^ 2 / (4 * k))) = fun w => exp (-(1 / (4 * k)) * (w - a) ^ 2) := by
  funext w
  congr 1
  ring

theorem integrable_gauss (a : ℝ) (hk : 0 < k) :
    Integrable fun w : ℝ => exp (-(w - a) ^ 2 / (4 * k)) := by
  rw [gauss_fun_eq]
  have h : Integrable fun x : ℝ => exp (-(1 / (4 * k)) * x ^ 2) :=
    integrable_exp_neg_mul_sq (by positivity)
  exact h.comp_sub_right a

theorem integral_gauss (a : ℝ) (hk : 0 < k) :
    ∫ w : ℝ, exp (-(w - a) ^ 2 / (4 * k)) = 2 * √(π * k) := by
  rw [gauss_fun_eq]
  rw [integral_sub_right_eq_self (fun x : ℝ => exp (-(1 / (4 * k)) * x ^ 2)) a, integral_gaussian]
  have h1 : π / (1 / (4 * k)) = 4 * (π * k) := by field_simp
  rw [h1, sqrt_mul (by norm_num : (0 : ℝ) ≤ 4)]
  have h4 : √(4 : ℝ) = 2 := by
    rw [show (4 : ℝ) = 2 ^ 2 by norm_num, sqrt_sq (by norm_num : (0 : ℝ) ≤ 2)]
  rw [h4]

theorem half_weight (u k w : ℝ) (hk : k ≠ 0) :
    exp (w / 2) * g u k w = exp (u / 2 + k / 4) * exp (-(w - (u + k)) ^ 2 / (4 * k)) := by
  rw [g, ← exp_add, ← exp_add]
  congr 1
  field_simp
  ring

theorem full_weight (u k w : ℝ) (hk : k ≠ 0) :
    exp w * g u k w = exp (u + k) * exp (-(w - (u + 2 * k)) ^ 2 / (4 * k)) := by
  rw [g, ← exp_add, ← exp_add]
  congr 1
  field_simp
  ring

theorem g_pos (u k w : ℝ) : 0 < g u k w := exp_pos _

theorem integrable_half_weight (hk : 0 < k) :
    Integrable fun w : ℝ => exp (w / 2) * g u k w := by
  have : (fun w : ℝ => exp (w / 2) * g u k w) =
      fun w => exp (u / 2 + k / 4) * exp (-(w - (u + k)) ^ 2 / (4 * k)) := by
    funext w; exact half_weight u k w hk.ne'
  rw [this]
  exact (integrable_gauss (u + k) hk).const_mul _

theorem integrable_full_weight (hk : 0 < k) :
    Integrable fun w : ℝ => exp w * g u k w := by
  have : (fun w : ℝ => exp w * g u k w) =
      fun w => exp (u + k) * exp (-(w - (u + 2 * k)) ^ 2 / (4 * k)) := by
    funext w; exact full_weight u k w hk.ne'
  rw [this]
  exact (integrable_gauss (u + 2 * k) hk).const_mul _

theorem integral_half_weight (hk : 0 < k) :
    ∫ w : ℝ, exp (w / 2) * g u k w = exp (u / 2 + k / 4) * (2 * √(π * k)) := by
  have : (fun w : ℝ => exp (w / 2) * g u k w) =
      fun w => exp (u / 2 + k / 4) * exp (-(w - (u + k)) ^ 2 / (4 * k)) := by
    funext w; exact half_weight u k w hk.ne'
  rw [this, integral_const_mul, integral_gauss (u + k) hk]

theorem c_pos (hk : 0 < k) : 0 < c u k := by
  unfold c
  positivity

/-- The weight `c * exp (w/2) * g` is a probability density. -/
theorem c_mul_integral_half_weight (hk : 0 < k) :
    c u k * ∫ w : ℝ, exp (w / 2) * g u k w = 1 := by
  rw [integral_half_weight hk, c]
  have hs : 0 < √(π * k) := sqrt_pos.mpr (by positivity)
  have he : exp (-k / 4 - u / 2) * exp (u / 2 + k / 4) = 1 := by
    rw [← exp_add]; simp [show -k / 4 - u / 2 + (u / 2 + k / 4) = 0 by ring]
  have h2 : (2 * √(π * k)) / (2 * √(π * k)) = 1 := div_self (by positivity)
  calc exp (-k / 4 - u / 2) / (2 * √(π * k)) * (exp (u / 2 + k / 4) * (2 * √(π * k)))
      = exp (-k / 4 - u / 2) * exp (u / 2 + k / 4) * ((2 * √(π * k)) / (2 * √(π * k))) := by ring
    _ = 1 := by rw [he, h2, one_mul]

end MertensVerify
