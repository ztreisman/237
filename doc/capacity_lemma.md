# The Capacity Lemma: what is proven, what is conditional, what remains
# {3,7} project — 2026-06-05

Throughout, P_r is the simplicial disk of rings 0..r of the {3,7} tiling,
realized in R^d with rigid unit equilateral faces; N_r = ring sizes
(7, 21, 56, 147, 385, 1008, ... with N_{r+1} = 3N_r − N_{r−1});
V_r = Σ_{k≤r} N_k; F_r = number of faces; φ = (1+√5)/2.

We use the Euler-count identities (triangulated disk, χ = 1):
  F_r = 2V_r − N_r − 2,     band faces B_r := F_r − F_{r−1} = N_r + N_{r−1}.
(Check at r = 5: F_5 = 2·617 − 385 − 2 = 847 ✓; B_5 = 385+147 = 532 ✓,
which also equals the radial-edge count 2n_A + n_B = 2·147 + 238 = 532 ✓.)

---

## Theorem 1 (Demand identity — unconditional, exact)

The per-sector fold demand converges to (π/3)·φ. Precisely: the total
intrinsic turning of ∂P_{r+1} is (π/3)·n_B(r+1) with n_B(r+1) = 6 + V_r
(Gauss–Bonnet, proven earlier), and

    (6 + V_r)/N_r  →  φ,        so   demand per ring-r sector → (π/3)·φ.

*Proof.* Set x_r = N_{r+1}/N_r. The recurrence gives x_r = 3 − 1/x_{r−1},
whose iteration map g(x) = 3 − 1/x is increasing on (1, ∞) with fixed points
(3±√5)/2. From x_2 = 56/21 = 8/3 ∈ (φ⁻², φ²), monotone convergence gives
x_r ↑ φ² = (3+√5)/2. (Closed form: N_r = Aφ^{2r} + Bφ^{−2r}.) Then

  V_r/N_r = Σ_{k≤r} N_k/N_r → Σ_{j≥0} φ^{−2j} = 1/(1−φ^{−2}) = φ²/(φ²−1) = φ,

using φ² − 1 = φ. Hence (6+V_r)/N_r → φ. Equivalently, since
6 + V_r = n_B(r+1) = N_{r+1} − N_r, the same limit reads
n_B(r+1)/N_r → φ² − 1 = φ. ∎

Convergence is fast (error ~ φ^{−4r}): at r = 5, (6+V_5)/N_5-type ratio
238/147 = 1.61905 vs φ = 1.61803. Measured per-sector total D4+D5 = 1.757
(CV 0.17) vs theoretical (π/3)(238/147) = 1.696 — agreement to 3.6%.

**Consequence.** Demand per sector is asymptotically CONSTANT. Any
impossibility proof must therefore come from capacity decay, not demand
growth.

---

## Theorem 2 (The tetrahedral wall — unconditional trichotomy)

Let v be a type-B boundary vertex with boundary neighbors u, w and parent a
(fan = two unit triangles uva, avw over the parent edge va). Let κ = κ_ext(v),
let δ = the dihedral angle along va, and let s = |f(u) − f(w)|. Then the
following are EQUIVALENT, by exact identities (no inequalities):

    s = 2cos(κ/2)            and          s = √3 · sin(δ/2),

hence
    κ ≤ 2π/3   ⟺   s ≥ 1   ⟺   δ ≥ arccos(1/3) = 70.53°.

*Proof.* Law of cosines in triangle u,v,w with |uv| = |vw| = 1 and interior
angle β = π − κ at v: s = 2sin(β/2) = 2cos(κ/2). For the fan: u and w both
lie at distance 1 from both endpoints of the unit edge va, hence on the
circle of radius √3/2 in the bisecting plane of va; the angle between their
positions on that circle is the dihedral δ, so s = 2·(√3/2)·sin(δ/2). ∎

The configuration at the wall (s = 1, δ = arccos(1/3)) is exactly the
REGULAR TETRAHEDRON on {u, v, a, w}. Folding past the wall means the two
outer triangles of the fan pass through the tetrahedral configuration —
beyond it, u and w are at distance < 1 and the fan is collapsing toward
self-contact (faces coincide at δ = 0).

**Empirical confirmation (209 trials).** The wall is not imposed by the
problem (non-adjacent vertices MAY be closer than 1) — yet the relaxer
respects it statistically:
  p90 extrinsic turning = 2.082–2.096 rad across ALL word classes
    (wall: 2π/3 = 2.0944);
  p90 band-5 fold = 1.9194–1.9287 rad
    (wall: π − arccos(1/3) = 1.9106) — agreement within 1%.
The "dihedral wall" in the headroom analysis h(r) = π − fold_max is hereby
identified: it is the tetrahedral dihedral arccos(1/3), i.e. the
second-neighbor unit-distance shell. All three measured ceilings (turning
2π/3, fold 1.91, distance 1) are the same wall in three coordinates.

---

## Lemma 3 (Confinement inequality — unconditional, but not binding)

Any closed polygon γ ⊂ B(0, R) with edge lengths ℓ_i and turning angles κ_i
satisfies
    L = Σℓ_i  ≤  R · Σ 2sin(κ_i/2)  ≤  R · K_ext.

*Proof.* L = ∮⟨γ', T⟩ds = −∮⟨γ, dT⟩ ≤ max|γ| · ∮|dT|, and for a polygon
∮|dT| = Σ|ΔT_i| = Σ 2sin(κ_i/2). ∎  (Sharp for circles: L = 2πR, K = 2π.)

Applied to ∂P_r ⊂ B(0, r): K_ext ≥ N_r/r. Honest remark: Gauss–Bonnet
already forces K_ext ≥ (π/3)(6+V_{r−1}) ≈ 0.647·N_r, which exceeds N_r/r for
r ≥ 2, so confinement adds nothing here. The turning BUDGET also never
binds: even under the tetrahedral cap κ ≤ 2π/3, budget (2π/3)N_r exceeds
demand (π/3)n_B ≈ (π/3φ)... demand/budget → 1/(2φ)·... = 0.309 < 1.
The obstruction is not in the turning ledger; it is spatial. This matches
the data (failure mode = dihedrals at the wall, not turning saturation).

---

## Theorem 4 (Conditional capacity — finiteness under quantitative reach)

Say a realization f of P_r has **reach λ** if any two faces that do not
share a vertex are at Euclidean distance ≥ λ. Let C₀ = 15 (a face has 3
vertices, each in ≤ 7 faces; faces sharing a vertex with a fixed face
number ≤ 15 — generous).

**(a) R³:** If P_r admits a realization with reach λ contained in B(0, r),
then
    F_r · (√3/4) · λ  ≤  C₀ · (4/3)π (r + λ/2)³.

**(b) R⁴:**  F_r · (√3/4) · πλ²  ≤  C₀ · (π²/2)(r + λ/2)⁴.

Since F_r ~ c·φ^{2r} (exponential) and the right sides are polynomial, both
yield an explicit finite r₀(λ): **realizations with any fixed positive reach
fail at finite radius.** Numerically (C₀ = 15):

    R³:  λ = 0.5 → r₀ = 12;   λ = 0.25 → r₀ = 13
    R⁴:  λ = 0.5 → r₀ = 15;   λ = 0.25 → r₀ = 17

*Proof of (a).* For each face F_i let U_i be its open λ/2-neighborhood;
vol(U_i) ≥ area(F_i)·λ = (√3/4)λ (the prism over the face alone). If
x ∈ U_i ∩ U_j then dist(F_i, F_j) < λ, so by reach F_i and F_j share a
vertex; hence every point lies in U_i for faces from a single vertex-sharing
cluster, of size ≤ C₀. So Σ vol(U_i) ≤ C₀ · vol(B(0, r + λ/2)). Sum the
lower bounds. (b) identical with the codim-2 tube volume πλ²·area. ∎

**Honesty about the gap.** r₀(λ=0.25) = 13 in R³, but the true threshold is
6. The volume argument charges each unit of AREA a thickness λ; the data
says the actual mechanism is more expensive: the demand (π/3)φ per sector
must be executed as FOLDS, and a fold near the tetrahedral wall sweeps a
three-dimensional region (the fan's swept wedge), consuming volume per
SECTOR, not per unit area. Note also r₀(λ) → ∞ as λ → 0: the conditional
theorem does NOT rule out near-collision realizations going deeper — and
this is exactly the regime of the existing cascade analysis ("ε → 0 forces
the contradiction at ring 6"). The two analyses are the same boundary
layer approached from opposite sides.

---

## Lemma 5 (OPEN — the sharp capacity lemma, precisely stated)

Let f realize P_r in R³ with reach λ, and let p be a ring-(r−1) vertex with
sector demand d(p) (the fold total over p's band edges; Σ_p d(p) =
(π/3)n_B(r) + o(·), and empirically d(p) ≈ (π/3)φ with CV 0.17). 

CLAIM TO PROVE: there is a universal v₀ > 0 such that the sector's swept
region (the union of the λ/2-neighborhoods of p's band faces) has volume
≥ v₀ · min(d(p), π − arccos(1/3))^α for some α ≥ 1 — i.e., executing fold
demand costs VOLUME PER SECTOR bounded below, not merely volume per area.
Then Σ_p over N_{r−1} sectors against vol B(0,r) gives

    N_{r−1} · v₀ · ((π/3)φ)^α  ≤  C₀ · (4/3)πr³,

and with v₀ calibrated by the tetrahedral-wall geometry the threshold lands
at the first r with N_{r−1} ≳ r³ — i.e. between N_4 = 147 vs ~3·6³ and
N_5 = 385: **r₀ = 6** falls out if v₀·((π/3)φ)^α ≈ 0.6–1.2, which is the
volume of a unit-triangle fan swept through ~1 radian. This is the lemma
the data is pointing at; proving the per-sector volume floor (the
"fold-displacement inequality") is the single remaining gap between the
conditional theorem and r(2,3) = 5.

Supporting empirical facts for Lemma 5: per-sector totals are conserved
(CV 0.17 — demand can't be dodged by uneven sharing); folds at p90 sit ON
the tetrahedral wall (the floor is active); stress concentrates at lone-B
positions of the substitution word σ: A→AB, B→ABB (density φ⁻⁴), giving the
localization needed to keep v₀ uniform.

## R⁴ prediction (the falsifiability check)

In R⁴ the same Lemma-5 form reads N_{r−1}·v₀' ≤ C·r⁴ with v₀' ~ λ·(swept
3-volume): threshold at N_{r−1} ~ r⁴/λ. For λ in the 0.1–0.3 range this
lands r₀(R⁴) in 10–12 — consistent with (and just above) the observed
r(2,4) ≥ 9. Measuring fold statistics in the R⁴ grower output
(grow_37_4d.py) and checking whether its tetrahedral-wall analogue
(the 4d fan wall: same arccos(1/3)? or the 5-cell dihedral arccos(1/4)?)
is respected at p90 would test the whole mechanism off-sample.

## Status summary

  PROVEN:      demand identity (π/3)φ exact; tetrahedral wall trichotomy;
               confinement inequality; conditional finiteness with explicit
               r₀(λ) in R³ and R⁴.
  IDENTIFIED:  the single open gap (Lemma 5, per-sector volume floor) with
               a precise statement and the empirical facts that support it.
  NOT CLAIMED: an unconditional proof of r(2,3) = 5. The λ → 0 boundary
               layer is exactly the cascade-contradiction regime; closing
               Lemma 5 there is the remaining mathematics.
