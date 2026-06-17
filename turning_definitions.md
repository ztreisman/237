# Intrinsic and Extrinsic Turning of the Boundary Polygon ∂P_r

## Setting

P_r denotes the simplicial disk consisting of rings 0 through r of the {3,7}
tiling: a triangulated surface in which every face is a triangle and seven
triangles meet at every interior vertex. An embedding f: P_r → ℝ³ realizes
every triangle as a unit equilateral triangle in space. The boundary ∂P_r is
the closed polygon through the ring-r vertices, with N_r unit-length edges.

We study the *turning* of this boundary polygon at each of its vertices, in
two senses.

---

## Definition: Extrinsic turning

Let v be a boundary vertex, with the boundary polygon entering along edge e₁
and leaving along edge e₂. Both edges are unit segments in ℝ³.

**The extrinsic turning at v** is

  κ_ext(v) = π − ∠_{ℝ³}(e₁, e₂)

where ∠_{ℝ³}(e₁, e₂) ∈ [0, π] is the angle at v between the two edge
directions, measured in the ambient space ℝ³.

If the polygon continues straight through v, the angle is π and the turning
is 0. If the polygon doubles back, the angle is 0 and the turning is π. The
extrinsic turning is what a observer in ℝ³ sees: how sharply the space curve
bends at v. The **total extrinsic curvature** of the boundary is
Σ_v κ_ext(v), the discrete analogue of ∫|κ| ds for a smooth space curve.
(Fenchel's theorem: this total is ≥ 2π for any closed curve; Fáry–Milnor:
> 4π if the curve is knotted.)

---

## Definition: Intrinsic turning

The intrinsic turning ignores ℝ³ entirely and uses only the surface.

At a boundary vertex v, some number m(v) of triangles of P_r are incident to
v (the triangle fan on the interior side of the boundary). Each is
equilateral, contributing angle π/3 at v. The **interior angle** of the
surface at v is

  α(v) = m(v) · π/3,

and the **intrinsic turning at v** is

  κ_int(v) = π − α(v).

This is the turning that a two-dimensional inhabitant of the surface would
measure while walking along the boundary: how much the boundary deviates from
a surface-geodesic at v. It is the discrete geodesic curvature.

---

## The logic dictating the intrinsic turning

The intrinsic turning of ∂P_r is **completely determined by the
combinatorics of the tiling** — no embedding information enters. Two facts
pin it down.

### Fact 1: Each boundary vertex is one of two types.

A ring-r vertex v has 7 neighbors in the full tiling, arranged cyclically:
its parents (ring r−1), its two siblings (ring r), and its children
(ring r+1). The children lie outside P_r, so the triangle fan inside P_r at
v consists of the triangles spanned by v with its parents and siblings.

- **Type A** (two parents a₁, a₂): the cyclic order of neighbors is
  s₁, a₁, a₂, s₂, c₁, c₂, c₃. The interior triangles are (v,s₁,a₁),
  (v,a₁,a₂), (v,a₂,s₂) — three triangles. So
  α(v) = 3·π/3 = π and **κ_int = 0**: the boundary passes straight
  through every type-A vertex, intrinsically.

- **Type B** (one parent a): the cyclic order is s₁, a, s₂, c₁, c₂, c₃, c₄.
  The interior triangles are (v,s₁,a), (v,a,s₂) — two triangles. So
  α(v) = 2π/3 and **κ_int = π/3**: the boundary turns by exactly 60° at
  every type-B vertex, intrinsically.

The counts are n_A(r) = N_{r−1} (each ring-(r−1) vertex spawns exactly one
two-parent child) and n_B(r) = N_r − N_{r−1}.

### Fact 2: Gauss–Bonnet forces the total.

The disk P_r has Euler characteristic 1. Every interior vertex carries
discrete curvature 2π − 7·(π/3) = −π/3 (the angle *excess* π/3 is negative
curvature — this is where the hyperbolicity of {3,7} lives). Discrete
Gauss–Bonnet for a disk:

  Σ_{interior v} κ(v) + Σ_{boundary v} κ_int(v) = 2π

  −(π/3)·V_{r−1} + (π/3)·n_B = 2π

  **n_B(r) = 6 + V_{r−1}**

where V_{r−1} = Σ_{k<r} N_k is the number of interior vertices. This identity
is verified exactly against the tiling data for r = 2,…,6:
(14, 35, 91, 238, 623) = 6 + (8, 29, 85, 232, 617). ✓

So the total intrinsic turning of ∂P_r is

  **T_int(r) = (π/3)·n_B = 2π + (π/3)·V_{r−1}**,

growing like φ^{2r} (since V_{r−1} ~ N_r/(φ²−1) and the growth rate of the
ring sizes is φ², the golden ratio squared).

The picture: the boundary is intrinsically a polygon that turns left 60° at
n_B vertices and goes straight at n_A vertices. The negative curvature
accumulating in the interior forces the boundary to turn more and more in
total — each interior vertex adds π/3 of mandatory boundary turning. This is
the discrete shadow of the fact that a hyperbolic disk of radius r has a
boundary circle whose geodesic curvature integral grows with its area.

---

## The bridge inequality: κ_ext ≥ κ_int

At each boundary vertex, the extrinsic turning is at least the intrinsic
turning:

  **κ_ext(v) ≥ κ_int(v).**

*Proof.* The angle between two rays from a common point in ℝ³ obeys the
spherical triangle inequality: for any chain of intermediate rays,

  ∠(e₁, e₂) ≤ ∠(e₁, g₁) + ∠(g₁, g₂) + … + ∠(g_{m−1}, e₂).

Take the intermediate rays g_i to be the interior edges at v in cyclic
order. Each consecutive angle is a face angle of a unit equilateral
triangle = π/3, and there are m(v) of them. Hence
∠_{ℝ³}(e₁, e₂) ≤ m(v)·π/3 = α(v), and therefore
κ_ext(v) = π − ∠_{ℝ³}(e₁,e₂) ≥ π − α(v) = κ_int(v). ∎

Equality holds iff the triangle fan at v unfolds flat in ℝ³ — i.e., all
interior triangles at v are coplanar with the boundary turning happening
entirely in that plane. Any "pleating" of the fan out of the plane (normal
curvature) strictly increases the extrinsic turning above the intrinsic
floor.

Summing over the boundary:

  **Total extrinsic curvature of ∂P_r ≥ 2π + (π/3)·V_{r−1} ~ (π/3)·φ^{2r}/(φ²−1).**

---

## What this buys us

The boundary polygon of any unit-triangle embedding of P_r in ℝ³ is a closed
space curve with N_r unit edges whose total curvature is forced to be at
least 2π + (π/3)V_{r−1}. The average turning per edge tends to

  (π/3) · V_{r−1}/N_r → (π/3)·(1/(φ²−1)) = (π/3)·(φ⁻²·φ²/(φ²−1)) ≈ 0.206π ≈ 37°.

A unit-edge polygon with sustained turning θ per step is locally inscribed in
a circle of radius 1/(2 sin(θ/2)) ≈ 1.6. So the boundary is forced to coil at
local radius ≈ 1.6 while its total length φ^{2r} grows exponentially and the
whole curve is confined to the ball B(0, r) — whose radius grows only
linearly. The curve must coil, the coils must pack, and (the next step of the
argument) each coil is tethered by unit edges to the previous ring, whose
geometry dictates the direction of coiling. The conflict between adjacent
coils' inherited convexity is the buckling that kills the embedding at
ring 6.

The intrinsic turning is the part the tiling dictates; the extrinsic turning
is the part ℝ³ must pay; the inequality says ℝ³ can never pay less than the
tiling demands.
