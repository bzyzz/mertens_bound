import MertensVerify.Defs

/-!
# Elementary facts about `M`

* `M x = 0` for `x < 1`;
* the trivial bound `|M x| ≤ x` for `x ≥ 0`;
* `w ↦ M (exp w)` is measurable.
-/

open Real MeasureTheory

namespace MertensVerify

theorem M_of_lt_one {x : ℝ} (hx : x < 1) : M x = 0 := by
  have : ⌊x⌋₊ = 0 := Nat.floor_eq_zero.mpr hx
  simp [M, this]

/-- `|M x| ≤ ⌊x⌋₊`, since `|μ n| ≤ 1`. -/
theorem abs_M_le_floor (x : ℝ) : |(M x : ℝ)| ≤ (⌊x⌋₊ : ℝ) := by
  have h : |M x| ≤ ∑ n ∈ Finset.Icc 1 ⌊x⌋₊, |ArithmeticFunction.moebius n| :=
    Finset.abs_sum_le_sum_abs _ _
  have h1 : ∀ n ∈ Finset.Icc 1 ⌊x⌋₊, |ArithmeticFunction.moebius n| ≤ 1 := fun n _ =>
    ArithmeticFunction.abs_moebius_le_one
  have h2 := Finset.sum_le_sum h1
  simp only [Finset.sum_const, Nat.card_Icc, add_tsub_cancel_right, nsmul_eq_mul, mul_one] at h2
  have h3 : |M x| ≤ (⌊x⌋₊ : ℤ) := h.trans h2
  have : ((|M x| : ℤ) : ℝ) ≤ ((⌊x⌋₊ : ℤ) : ℝ) := by exact_mod_cast h3
  simpa [Int.cast_abs] using this

/-- The trivial bound `|M(x)| ≤ x`. -/
theorem abs_M_le {x : ℝ} (hx : 0 ≤ x) : |(M x : ℝ)| ≤ x :=
  (abs_M_le_floor x).trans (Nat.floor_le hx)

theorem measurable_M_exp : Measurable fun w : ℝ => (M (exp w) : ℝ) := by
  have hf : Measurable fun n : ℕ =>
      ((∑ m ∈ Finset.Icc 1 n, ArithmeticFunction.moebius m : ℤ) : ℝ) := measurable_from_nat
  exact hf.comp (Measurable.nat_floor measurable_exp)

/-- A violation forces `x ≥ 1` (for `x < 1` the sum is empty). -/
theorem one_le_of_violates {x : ℝ} (h : Violates x) : 1 ≤ x := by
  by_contra hx
  push_neg at hx
  have := M_of_lt_one hx
  unfold Violates at h
  rw [this] at h
  simp at h
  exact absurd h (not_lt.mpr (sqrt_nonneg x))

end MertensVerify
