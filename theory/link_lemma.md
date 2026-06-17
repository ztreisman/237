# The Link Lemma: transverse fold coupling at constant negative curvature
# {3,7} project — 2026-06-05

## Setup

At every interior vertex v of P_r, seven unit equilateral triangles meet.
Intersect the star of v with the unit sphere centered at v. Each face cuts a
geodesic arc of length π/3 (its angle at v); consecutive arcs meet at the
dihedral angle along the shared edge. Therefore:

**The link of every interior vertex is a closed spherical heptagon with all
seven sides equal to π/3, whose interior angles are the seven dihedral
angles θ_1, …, θ_7 at the edges incident to v.**

Perimeter = 7π/3 = 2π + π/3. Two immediate structural facts:

(i) **The link is never convex.** A convex closed curve on the unit sphere
has length ≤ 2π; the link exceeds this by exactly π/3 — which is exactly
−(discrete intrinsic curvature) at v. The perimeter excess IS the negative
curvature, in link coordinates. Every vertex star of {3,7} in R³ is
non-convexly folded; there is no flat or convex configuration at all.

(ii) The configuration space of closed spherical heptagons with sides π/3
is 4-dimensional (7 angles − 3 closure conditions); embeddedness of the
star requires the link to be a SIMPLE closed curve.

## Lemma (per-vertex fold conservation)

For every interior vertex v, with signed folds f_i = π − θ_i (sign by the
coherent orientation of the surface):

    Σ_{i=1}^{7} f_i  =  2π − A(v)   ∈ (−2π, 2π),

where A(v) is the spherical area enclosed by the link.

*Proof.* Spherical Gauss–Bonnet for a simple geodesic n-gon:
Σθ_i = (n−2)π + A. With n = 7: Σθ_i = 5π + A(v); substitute θ_i = π − f_i. ∎

This is an exact local conservation law at every interior vertex — the
pointwise version of the global demand identity. Note what it does and does
not constrain: the SIGNED total is pinned to 2π − A(v); the UNSIGNED total
Σ|f_i| is free to exceed it only through alternation (pleating). This is
precisely the empirically observed compensation structure: the embedding
meets large unsigned demand with anti-correlated folds because the signed
budget per vertex is conserved.

## Transversality suppression (the user's observation, quantified)

A crease passing through v uses two roughly opposite edges with sharp folds
(θ ≈ 0). Two transverse creases crossing at or near v use four edges with
θ_i ≈ 0. Gauss–Bonnet then forces the remaining three angles to satisfy

    θ_remaining (avg)  ≈  (5π + A)/3  >  5π/3 = 300°,

i.e. reflex folds |π − θ| ≥ 2.26 rad on EVERY other edge at v (verified:
A = 0.5 → 309.5°, fold 2.261; A = 1 → 319.1°, fold 2.428). Folds crossing
near a sharp fold cannot themselves be sharp in the same sense — the link
must pay them back as reflex folds, and the side-length rigidity (every arc
exactly π/3) plus simplicity bound how much can accumulate. Constant
negative curvature couples transverse folds through link closure.

## The tetrahedral wall in link coordinates

A spike of the link at vertex i with angle θ has span d between its two
neighboring link vertices given by the spherical law of cosines:

    cos d = cos²(π/3) + sin²(π/3)·cos θ = 1/4 + (3/4)cos θ.

At the tetrahedral wall θ = arccos(1/3): cos d = 1/4 + 1/4 = 1/2, so
**d = π/3 exactly — the spike's span equals one mesh side.** Passing the
wall means the spike becomes narrower than a side of the polygon itself
(a "virtual side" degeneration). The three measured ceilings (turning 2π/3,
fold 1.91, second-neighbor distance 1) and now the link degeneration are
one wall in four coordinate systems.

## Capacity, reformulated (replacing the volume-packing argument)

Define the per-vertex unsigned fold capacity

    C_link  =  max  Σ_{i=1}^{7} |π − θ_i|
              over closed spherical heptagons with all sides π/3,

and C_link^simple = the same max over SIMPLE (embedded-link) heptagons.

Computed: the unconstrained maximum is attained at the regular spherical
heptagram {7/3} with side π/3 — all seven angles equal arccos-form value
29.78° (verified numerically to 6 digits and in closed form:
sin ρ = sin(π/6)/sin(3π/7), cot(θ/2) = cos ρ · tan(3π/7)):

    C_link = 7·(π − 0.5198) = 18.353 rad = 5.842π.

The heptagram self-intersects, so this is a strict upper bound for embedded
stars. C_link^simple is strictly smaller and is the honest per-vertex
capacity; its computation is pending (optimizer note: random starts wind —
seed instead from MEASURED links: every ring-5 trial PLY supplies 232
genuine simple links; maximize from those).

The comparison ladder so far (rad per vertex):

    demand at band 5 (measured):        ≈ 7.1
    7 × tetrahedral wall (heuristic):     13.4
    C_link^simple:                        pending (between the two?)
    C_link (heptagram, non-simple):       18.35

## Status of the proof architecture

- Theorem 4 of capacity_lemma.md (volume packing under reach λ) is hereby
  demoted to MOTIVATION: its hypothesis is alien to the problem, its
  threshold (r₀ ≈ 13) is far from 6, and it vanishes as λ → 0. The
  conceptual content it provided — exponential demand vs polynomial room —
  survives, but the proof-bearing object is the link.
- The capacity question is now finite-dimensional and local: per-vertex
  demand (from the global Gauss–Bonnet identity, (π/3)φ per sector,
  distributed over links) versus C_link^simple (a computable constant of
  the {3,7} mesh), with the coupling between neighboring links (shared
  edges → shared angles) supplying the propagation that turns local
  saturation into global failure.
- Open: (a) compute C_link^simple; (b) the demand-routing lemma: confinement
  in B(0,r) forces per-vertex unsigned demand approaching C_link^simple by
  ring 6 — this replaces Lemma 5's volume floor with an angle floor.
