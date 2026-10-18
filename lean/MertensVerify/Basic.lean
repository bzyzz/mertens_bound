/-
  What this file proves, and what it does not.

  Proved, from no axioms:
  * the exponent cancellation in the Gaussian weight of `M₁` (Lemma 3.1 of the paper);
  * the completion-of-square identity behind the upper tail of `|M₂ − M₁|`;
  * the matching identity for the lower-tail endpoint, and that this endpoint is
    strictly smaller whenever `√(ku) > 0`;
  * the arithmetic in Section 6: from the certificate enclosures
        `K₂ ∈ [1.0003052504 ± 2.30e-11]`, `E ≤ 2.40e-11`,
        `k = 1.5e-10`, `v = 3484971332733132396.5383341489440461`,
    one has `|K₂| − 1 − E > 0`, `|K₂| − E > 1.0003`, and
        `u + 2s < 3484971332733178124`,
    where `u = v − k` and `s` is a positive rational with `s² > ku`.
    Any real square root of `ku` is then strictly less than `s`, so
        `u + 2√(ku) < 3484971332733178124`.

  Not formalized, and not claimed here: Pintz's Lemmas 4 and 5 and Theorem A,
  the contour integral and the bounds on `J₁`…`J₆`, Platt–Trudgian, and the
  Arb evaluation of `K₂` over the zeros below `T₀`. Those are the hypotheses
  the numerical step consumes. The integral comparison in the Gaussian tail
  (as opposed to the exponent identity) is also not formalized.
-/

namespace MertensVerify

/-! ### Data from the certificate -/

/-- Evaluation point `v` from Section 4 of the paper. -/
def v : Rat := 34849713327331323965383341489440461 / 10000000000000000

/-- Gaussian width `k = 1.5 × 10⁻¹⁰`. -/
def k : Rat := 3 / 20000000000

/-- `u = v − k`, the centre of the logarithmic window. -/
def u : Rat := v - k

/-- A rational upper bound for `√(ku)`. -/
def s : Rat := 1143181634639 / 50000000

/-- Centre of the certified enclosure `K₂ ∈ [1.0003052504 ± 2.30e-11]`. -/
def K2Centre : Rat := 10003052504 / 10000000000

/-- Radius of that enclosure. -/
def K2Radius : Rat := 23 / 1000000000000

/-- Lower endpoint of the enclosure. -/
def K2Lower : Rat := K2Centre - K2Radius

/-- The paper's upper bound `E ≤ 2.40 × 10⁻¹¹`. -/
def EUpper : Rat := 24 / 1000000000000

/-- The integer in the statement of the main theorem. -/
def exponentBound : Rat := 3484971332733178124

/-! ### Algebra of the Gaussian weight -/

/-- Exponents in the prefactor and in the full-line Gaussian integral cancel. -/
theorem prefactor_exponent (u k : Rat) : -k / 4 - u / 2 + (u / 2 + k / 4) = 0 := by
  grind

/-- Completion of the square for the upper tail.

With `a = √u` and `b = √k`, the left-hand side is the exponent
`u/2 + 3k/4 − (√u − √k)²` produced by the Gaussian tail bound, and the
right-hand side is `-u/2 + 2√(ku) − k/4`. -/
theorem upper_tail_exponent (a b : Rat) :
    a * a / 2 + 3 * (b * b) / 4 - (a - b) * (a - b)
      = -(a * a) / 2 + 2 * a * b - b * b / 4 := by
  grind

/-- The same completion at the lower endpoint `d = −2√(ku)`.

The last summand is `D²/(4k)` with `D = 2ab`. The hypothesis `b ≠ 0` is
`k ≠ 0`. -/
theorem lower_tail_endpoint (a b : Rat) (hb : b ≠ 0) :
    a * a / 2 - b * b / 4 - 2 * a * b - (2 * a * b) * (2 * a * b) / (4 * (b * b))
      = -(a * a) / 2 - 2 * a * b - b * b / 4 := by
  grind

/-- That lower endpoint is strictly below the upper-tail exponent when `√(ku) > 0`. -/
theorem lower_tail_endpoint_lt_upper (a b : Rat) (h : 0 < a * b) :
    -(a * a) / 2 - 2 * a * b - b * b / 4
      < -(a * a) / 2 + 2 * a * b - b * b / 4 := by
  grind

/-! ### The numerical step in Section 6 -/

theorem s_pos : 0 < s := by native_decide

/-- `s² > ku`, so `√(ku) < s` for the positive real square root. -/
theorem s_sq_gt_ku : s * s > k * u := by native_decide

theorem exponent_lt : u + 2 * s < exponentBound := by native_decide

/-- The matching lower end of the window, used in the introduction. -/
theorem window_lower : u - 2 * s > 3484971332733086669 := by native_decide

theorem margin_pos : K2Lower - 1 - EUpper > 0 := by native_decide

theorem exceeds_one_point_0003 : K2Lower - EUpper > 10003 / 10000 := by native_decide

/-- The inequalities Section 6 needs from the certificate. -/
theorem section6 :
    K2Lower > 1 + EUpper ∧
    K2Lower - EUpper > 10003 / 10000 ∧
    0 < s ∧
    s * s > k * u ∧
    u + 2 * s < exponentBound ∧
    u - 2 * s > 3484971332733086669 := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩
  · native_decide
  · native_decide
  · native_decide
  · native_decide
  · native_decide
  · native_decide

end MertensVerify
