# A Pintz-type criterion with weight e^{-1.5·10⁻¹⁰ γ²} and height T₀ = 1.3·10⁶

Status: **draft for expert checking.** Each step below is either *cited* from a published source (marked **[cited]**), *re-derived here* (marked **[derived]**), or *certified numerically here* (marked **[verified here]**). The certified numbers come from these scripts in [`certification/`](certification/): `certify_K2.py` (K₂), `verify_table1.py` (bound on 1/ζ), `verify_lemma7.py`, and `error_bounds.py` (E).

Notation: 𝔵 is the smallest x ≥ 1 with |M(x)| > √x. Zeros are ρ = β + iγ with γ > 0. c₀ = tan 1.

$$K_2(v,T,k) := 2\sum_{0<\gamma<T} e^{-k\gamma^2}\,\frac{\cos(\gamma v - \arg(\rho\zeta'(\rho)))}{|\rho\zeta'(\rho)|}
\;=\; 2\sum_{0<\gamma<T} e^{-k\gamma^2}\,\mathrm{Re}\frac{e^{i\gamma v}}{\rho\zeta'(\rho)} .$$

## 1. Statement

**Theorem.** Let k = 1.5·10⁻¹⁰, T₀ = 1.3·10⁶ and T₁ = 4.3·10⁶. Let u satisfy 10⁶ ≤ u and the side conditions checked in `error_bounds.py`. Let E(u) = Σ|Jᵢ|/π + 2e^{−u/2+2√(ku)}, with the J-bounds of Section 3. If

$$|K_2(u+k,\,T_0,\,k)| \;>\; 1 + E(u),$$

then 𝔵 < exp(u + 2√(ku)). At u = v − k with v as below, E(u) ≤ 2.391·10⁻¹¹.

A variant with T₁ = 10⁷ that uses StR's published Table 1 constant (23.05) instead of the recomputed one (22.7671) gives E ≤ 3.2·10⁻¹¹. Both variants work.

**Corollary (numerical).** Take v = 3484971332733132396.5383341489440461 and u = v − k. Then the certified enclosure from `certification/k2_certificate.json` gives K₂(v, T₀, k) > 1 + E (Section 5). Hence

$$\mathfrak{x} < \exp(u + 2\sqrt{ku}) < \exp(3484971332733178124) \approx \exp(3.4849713\cdot 10^{18}),$$

which is 5.616× smaller in the exponent than Kim–Nguyen's exp(1.9572·10¹⁹).

Note: J₃ grows linearly in u, because of the 1/u distance to zeros in (Tab1). So E(u) must be recomputed for every u with `error_bounds.total_error`. For u ≲ 10²⁰, T₀ = 1.3·10⁶ keeps E(u) below 10⁻⁹.

## 2. Inputs that are cited, not re-derived

None of these depend on k or T₀. They are exactly the inputs used by Saouter–te Riele (StR, Math. Comp. 83 (2014) 421–433) and, through StR, by Kim–Nguyen.

- **(C) [cited] Pintz 1987, Thm A + Cor. A (= StR Thm C / Cor. D).** Suppose some zero has β₀ > ½ + (5 log γ₀ + ε)/log Y. Then D(Y) := Y⁻¹∫₀^Y |M(x)| dx > e^ε √Y. It follows that |M(x)| > e^ε √x for some x ≤ Y.
- **(I, J) [cited] Pintz Lemmas 4, 5 (= StR Lemmas I, J).** Take 0 ≤ σ ≤ 2, t ≥ 4, and suppose |s − ρ| ≥ 1/v for every zero ρ. Then log|1/ζ(s)| ≤ c₀ log t log v + 11 log t + 14.
- **(K) [verified here] Pintz Lemma 7 (= StR Lemma K).** |ζ(¼ + it)| ≥ e⁻¹⁵⁰ for |t| ≤ 4. Pintz states it without proof; `verify_lemma7.py` certifies min |ζ(¼+it)| ≥ 0.4556 on |t| ≤ 4 with Arb, using 8192 covering balls.
- **(Tab1) [verified here on [T₀, T₁]] StR §5, Table 1.** For s = ½ + 1/u + it, log|1/ζ(s)| ≤ log u + 11 log t + M.
  - StR prove this with M = 23.05 for 4 ≤ t ≤ 10379424.83, from their unpublished database of 22·10⁶ zeros.
  - `verify_table1.py` recomputes it from Pintz Lemma 4 and certified Arb zeros, over all 6,185,380 intervals [θ_{ν−1}, θ_ν] covering [1.3·10⁶, 4.3·10⁶]. The result is **M ≤ 22.7671** (certified 22.7670656, rounded up; `certification/table1_certificate.json`).
  - The extremal interval is I_ν = [θ_{ν−1}, θ_ν] = [2650984.27884, 2650984.61215] for ν = 5,042,995 (γ_ν = 2650984.61067, with the next zero only 0.003 higher), with N = −8.767. StR's Table 1 lists N = −8.77 at the same zero, on [γ_{ν−1}, γ_ν] = [2650983.947, 2650984.611]. The partitions differ, so this is a consistency check at the same location, not an independent confirmation of their minimisation.
- **(RH) [cited] Platt–Trudgian (Bull. LMS 2021).** All zeros with 0 < γ ≤ 3·10¹² are simple and lie on the critical line.
- **(Arb) [cited] Arb / FLINT 3 `acb_dirichlet_zeta_zeros`, `acb_dirichlet_zeta_nzeros`.** These give certified enclosures of the n-th zero and the zero count N(T). For the indices used here (below 1.4·10⁷), FLINT evaluates Hardy's Z by the Riemann–Siegel formula and isolates zeros on Gram/Rosser blocks, with Turing's method certifying the count; its Platt large-height method is used only for n ≥ 10¹¹ and was not invoked.

## 3. Lemmas re-derived here

Fix u > 16 and 0 < k < ¼. Define

$$M_1(u) = \frac{e^{-k/4-u/2}}{2\sqrt{\pi k}}\int_{|\log x-u|\le 2\sqrt{ku}} \frac{M(x)}{x}\,e^{-(u-\log x)^2/4k}\,dx, \qquad M_2(u) = \text{same integral over } x\in[1,\infty).$$

**Lemma E [derived].** If |M(x)| ≤ √x whenever |log x − u| < 2√(ku), then |M₁(u)| < 1. (The open window suffices: its endpoints have measure zero.)

*Proof.* Write w = log x. Then
|M₁| ≤ e^{−k/4−u/2}(2√(πk))⁻¹ ∫ e^{w/2 − (u−w)²/4k} dw,
taken over the window. The same integral over all of ℝ equals e^{u/2+k/4}·2√(πk). So the bound is at most 1, and it is strictly less than 1 because the window is a proper subset of ℝ. ∎

**Lemma F′ [derived].** |M₂(u) − M₁(u)| ≤ 2 exp(−u/2 + 2√(ku)). In particular this is ≤ 2e^{−u/4} when k ≤ u/64, which is StR Lemma F.

*Proof.* Use the trivial bound |M(e^w)| ≤ e^w and substitute w = u + d. The integrand of M₂ − M₁ is then at most
(2√(πk))⁻¹ e^{u/2 − k/4 + d − d²/4k} = (2√(πk))⁻¹ e^{u/2 + 3k/4 − (d−2k)²/4k}.

- **Upper tail** (d > D = 2√(ku)). The integral is ≤ e^{u/2+3k/4}·½erfc((D−2k)/2√k) ≤ e^{u/2+3k/4−(√u−√k)²} = e^{−u/2+2√(ku)−k/4}.
- **Lower tail** (d < −D). Here d − d²/4k ≤ −D − D²/4k, so this tail is smaller than the upper one.

Summing the two tails proves the bound. ∎

**Lemma G [derived].** M₂(u) = (2πi)⁻¹∫_{(2)} e^{k(s²−¼)+u(s−½)}/(sζ(s)) ds.

*Proof.* Start from 1/(sζ(s)) = ∫₁^∞ M(x) x^{−s−1} dx for Re s > 1, which is Mellin inversion. Use ∫ e^{ws − (u−w)²/4k} dw = 2√(πk) e^{us+ks²}. The Gaussian factor makes everything absolutely convergent, which justifies Fubini. ∎

Next, note that e^{k(s²−¼)+u(s−½)} = e^{k(s−½)² + (u+k)(s−½)}. So the residue at a simple zero ρ = ½ + iγ is e^{−kγ²}e^{iγ(u+k)}/(ρζ'(ρ)). This is why K₂ is evaluated at **v = u + k**.

**Dichotomy.** One of two cases holds.

- **Case 1.** Some zero has β > ½ + (5 log γ + 1)/u. Apply (C) with Y = eᵘ and ε = 1. This gives x ≤ eᵘ with |M(x)| > e√x, so 𝔵 ≤ eᵘ < exp(u + 2√(ku)) and we are done.
- **Case 2.** Every zero has β ≤ ½ + (5 log γ + 1)/u. Then the contour L of StR Lemma H, made of L₁…L₆ and their mirror images, has these properties:
  - It runs to the right of every zero with γ ≥ T₁, since σ(L₅) = ½ + (5 log t + 3)/u.
  - It runs to the right of the on-line zeros with T₀ < γ < T₁, along L₃ at σ = ½ + 1/u.
  - It runs to the left of the zeros with γ < T₀, along L₁ at σ = ¼, and so excludes s = 0.
  - Its distance to every zero is ≥ 1/u along L₃ through L₅. On L₅, with σ(t) = ½ + (5 log t + 3)/u, a zero with |t − γ| < 1/u has σ(t) − β ≥ (2 − 5 log(1 + 1/(ut)))/u ≥ (2 − 5/(uT₁))/u > 1/u.

  By (RH) every zero with γ < T₁ + c₀ is simple and on the line. The residue theorem then gives
  M₂(u) = K₂(u+k, T₀, k) + (2πi)⁻¹∫_L, and so |M₂ − K₂| ≤ (|J₁|+…+|J₆|)/π, where the reflected path accounts for the factor 2.

**Bounds on J₁…J₆ [derived].** These are implemented line by line in `error_bounds.py`. On every piece, the modulus of the exponential factor is e^{k(σ²−t²−¼)+u(σ−½)}.

| Piece | Path | Bound used |
|---|---|---|
| J₁ | σ = ¼, 0 ≤ t ≤ T₀ | ≤ 4e^{−u/4}[4e^{150} + T₀·T₀^{c₀ log 4 + 11}e^{14}], using (K) for t ≤ 4 and (J) with v = 4 |
| J₂ | t = T₀, ¼ ≤ σ ≤ ½+1/u | ≤ exp(−kT₀² + c₀ log T₀ log(1/δ) + 11 log T₀ + 16), using (J) with v = 1/δ, where δ = dist(T₀, {γ}) ≥ 0.18286 |
| J₃ | σ = ½+1/u, T₀ ≤ t ≤ T₁ | ≤ u·e^{M+2}∫_{T₀}^∞ t¹¹e^{−kt²}dt = u e^{M+2} Γ(6, kT₀²)/(2k⁶), using (Tab1) with M = 22.7671 |
| J₄ | t = T₁, ½+1/u ≤ σ ≤ ½+(5 log T₁+3)/u | ≤ exp(−kT₁² + 16 log T₁ + c₀ log T₁ log u + 18), using (J) with v = u |
| J₅ | σ = ½+(5 log t+3)/u, T₁ ≤ t ≤ t₆ | ≤ 2e^{18} e^{−kT₁²/2}/(kT₁), using (J) with v = u and \|ds\| ≤ 2dt; this needs T₁ ≥ √(A/k) and kT₁²/2 ≥ A log T₁, where A = c₀ log u + 16 (both checked) |
| J₆ | σ = 2, t ≥ t₆ = e^{3u/10−3/5} | ≤ 1.52·e^{1+3u/2}e^{−kt₆²}/(2kt₆), using \|1/ζ(2+it)\| ≤ ζ(2)/ζ(4) < 1.52 |

**Values at k = 1.5·10⁻¹⁰, T₀ = 1.3·10⁶, T₁ = 4.3·10⁶, M = 22.7671, u ≈ 3.485·10¹⁸** (upper bounds, from `error_bounds.py`):

- log J₁ ≤ −8.7·10¹⁷
- log J₂ ≤ −45.39
- log J₃ ≤ −23.31
- log J₄ ≤ −1495
- log J₅ ≤ −1360
- log J₆ ≤ −10^(9·10¹⁷)
- Lemma F′ term ≤ e^(−1.7·10¹⁸)

Total: **E ≤ 2.391·10⁻¹¹** (computed value ≈ 2.39071·10⁻¹¹; use the rounded-up bound). J₃ dominates; J₄ and J₅ need T₁ ≥ 4.2·10⁶.

**Proof of the Theorem.** Case 1 gives 𝔵 < eᵘ. In Case 2:

|M₁(u)| ≥ |K₂(u+k, T₀, k)| − Σ|J|/π − 2e^{−u/2+2√(ku)} > 1.

By Lemma E, some x with |log x − u| < 2√(ku) has |M(x)| > √x, so 𝔵 < exp(u + 2√(ku)). (𝔵 exists and is an integer: M is constant on each [n, n+1), so ⌊x⌋ is also a counterexample.) Cases 1 and 2 are disjoint proof branches; the two conclusions are not exclusive. ∎

## 4. Differences from Saouter–te Riele and from the superseded `legacy/superseded/certify_peak2_600k.py`

- **Smaller k means larger T₀.** In StR's J₃ bound, the factor e^{−kT₀²} has to overcome u·T₀¹¹e^{25}. At k = 3·10⁻⁹, T₀ = 3·10⁵ was enough. At k = 1.5·10⁻¹⁰ this requires T₀ ≳ 1.27·10⁶. At T₀ = 376,918, the height of the previous 600k-zero computation, the same bound gives E ≈ 10⁷⁹, so that computation cannot certify anything at this k.
- **The tail of K₂ for 376,918 < γ < T₀ is no longer optional.** By the residue theorem it is part of K₂, so it must be computed. `certify_K2.py` sums all N(T₀) = 2,325,572 zeros, each a certified Arb enclosure. Zeros n > 600,000 are summed at 20 digits; there the phase ball is [0, 2π] and each term contributes its full magnitude bound 2e^{−kγ²}/|ρζ'(ρ)|. This is rigorous and needs no assumed lower bound on |ζ'(ρ)|, unlike StR's "1/|ζ'| ≤ 9".
- **The old `eps_Mertens` and `eps_emp` play no role.** Neither is a valid bound: both ignore the weights 1/|ρζ'(ρ)| (no lower bound on |ζ'(ρ)| is known), and neither is the error term of any proved theorem.
- **Two small corrections to StR's printed bounds.** J₅ now includes the factor 2 from |ds| ≤ 2dt. J₆ is bounded with |1/ζ(2+it)| < 1.52 instead of Lemma J. Both are irrelevant numerically.

## 5. Certified numbers

From `certification/k2_certificate.json`, produced by `certification/certify_K2.py`. The run took 2,701 s with 24 workers. All zeros were recomputed with Arb; nothing from `search/chunks` was used.

| Quantity | Certified value |
|---|---|
| N(T₀) for T₀ = 1.3·10⁶ (exact, `arb.zeta_nzeros`) | 2,325,572 |
| δ = dist(T₀, {γ}) | ≥ 0.1828652 (γ_N = 1299999.5363, γ_{N+1} = 1300000.1829) |
| K₂ main part, n ≤ 600,000, at 45 digits | [1.000305250386499012059 ± 3.21·10⁻²²] |
| K₂ tail, 600,000 < n ≤ 2,325,572 | \|·\| ≤ 9.43·10⁻¹² (ball); Σ 2e^{−kγ²}/\|ρζ'(ρ)\| ≤ 7.39·10⁻¹² |
| **K₂(v, T₀, k)** | **[1.0003052504 ± 2.30·10⁻¹¹], lower bound 1.00030525037707** |
| M (`certification/table1_certificate.json`, 6,185,380 intervals, 3,000 s with 16 workers) | ≤ 22.7671 |
| E(u) (T₁ = 4.3·10⁶, M = 22.7671) | ≤ 2.391·10⁻¹¹ |
| E(u), StR variant (T₁ = 10⁷, M = 23.05) | ≤ 3.18·10⁻¹¹ |
| min \|ζ'(ρ)\| over all 2,325,572 zeros (diagnostic only) | 0.109 |

Therefore K₂ − 1 − E ≥ 3.05·10⁻⁴ > 0, and the Corollary follows: **𝔵 < exp(3484971332733178124)**.

As a cross-check, the earlier `certify_peak2_600k.py` value at n ≤ 600,000 was 1.000305250386499007…. It differs from the certified value by 5·10⁻¹⁸, which comes from the 26-digit coefficients in `search/chunks`.

## 6. What a referee must still check (open items)

1. ~~**Reading of (C).**~~ **Resolved.** StR's text states the Case-2 hypothesis with **log u** and "D(u) > e√u". Pintz's original paper (Astérisque 147–148, §5) instead assumes |M(x)| < √x for x < e^{u+2√(ku)} and deduces from Corollary A that β₀ ≤ ½ + 5 log|γ₀|/**u**. That is Corollary A with **Y = eᵘ**, the reading used here, and it is consistent with the contour L₅. The side condition of Theorem A, Y > γ₀⁵, holds automatically in Case 1 because β₀ < 1 forces γ₀⁵ < e^{u/2}. StR's "log u" is a notational slip.
2. **Constants in (I)/(J)/(K).** These are Pintz's Lemmas 4, 5 and 7, with the same constants as transcribed by StR. Pintz gives only proof sketches for Lemmas 4 and 5, and these have not been re-proved here. Lemma 7 is now verified numerically (see (K)).
3. ~~**(Tab1).**~~ **Resolved** on the range needed, [T₀, T₁] = [1.3·10⁶, 4.3·10⁶], by `verify_table1.py` (M ≤ 22.7671; its extremal interval is at the same zero as StR's). It still rests on Pintz Lemma 4 (item 2). Optional improvement, not done: integrate ∏|s−ρ|⁻¹ along L₃ instead of using its supremum. That would replace the factor u in J₃ by roughly log u and lower the required T₀.
4. **Arb's zero certification** (Riemann–Siegel with Gram/Rosser isolation and Turing's method) is trusted as implemented in FLINT 3.
5. **Peer review.** This is a computer-assisted extension of a published argument with new parameters. The remaining trust base is:
   - Pintz's Theorem A and Lemmas 4 and 5 (published, but only sketched);
   - Platt–Trudgian's RH verification;
   - FLINT/Arb's certified zero routines;
   - the derivations in Section 3.

   Before it is presented as a record, a specialist should check Section 3 and the use of Pintz's lemmas.
