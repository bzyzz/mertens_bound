import Mathlib

/-!
# Definitions

`M x = ∑_{n ≤ x} μ(n)` with Mathlib's Möbius function, and the Gaussian averages `M₁`, `M₂` of
THEOREM.md §3, written in the variable `w = log x` (so `dx / x = dw`, and `M (exp w) = 0` for `w < 0`):

  `M₂ u = c u k * ∫ w, M (exp w) * g u k w`,
  `M₁ u = c u k * ∫ w in [u - D, u + D], M (exp w) * g u k w`,

with `g u k w = exp (-(u - w)² / (4k))` and `c u k = exp (-k/4 - u/2) / (2 √(π k))`.
-/

open Real MeasureTheory

namespace MertensVerify

/-- The Mertens function `M(x) = ∑_{1 ≤ n ≤ x} μ(n)`. -/
noncomputable def M (x : ℝ) : ℤ := ∑ n ∈ Finset.Icc 1 ⌊x⌋₊, ArithmeticFunction.moebius n

/-- `x` violates the Mertens inequality. -/
def Violates (x : ℝ) : Prop := √x < |(M x : ℝ)|

/-- Gaussian weight `exp (-(u - w)² / (4k))`. -/
noncomputable def g (u k w : ℝ) : ℝ := exp (-(u - w) ^ 2 / (4 * k))

/-- Normalising constant `exp (-k/4 - u/2) / (2 √(π k))`. -/
noncomputable def c (u k : ℝ) : ℝ := exp (-k / 4 - u / 2) / (2 * √(π * k))

/-- The window `|w - u| ≤ D`. -/
def window (u D : ℝ) : Set ℝ := Set.Icc (u - D) (u + D)

/-- `M₂(u)`: the Gaussian average of `M` over all `x ≥ 1`. -/
noncomputable def M₂ (u k : ℝ) : ℝ := c u k * ∫ w, (M (exp w) : ℝ) * g u k w

/-- `M₁(u)`: the same average restricted to the window `|log x - u| ≤ D`. -/
noncomputable def M₁ (u k D : ℝ) : ℝ := c u k * ∫ w in window u D, (M (exp w) : ℝ) * g u k w

end MertensVerify
