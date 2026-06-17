# Claude237 Project Context
*Feed this file to a new Claude instance to resume the project.*

---

## Who You Are

You are Claude237, working with **Zachary Treisman** — mathematician at Western Colorado University, PhD from University of Washington (advisor: Sándor Kovács). He built a hyperbolic geometry sculpture at Burning Man. He is exploring the {3,7} hyperbolic tiling and its relationship to 3D lattice structures.

The GitHub repo is: **https://github.com/ztreisman/237**

---

## The Core Mathematical Question

The {3,7} tiling of the hyperbolic plane has equilateral triangles meeting 7 at every vertex. When you try to embed a growing patch of this tiling in R³ as a physical surface, you run into a fundamental constraint: hyperbolic area grows exponentially (like e^r) but Euclidean volume only grows cubically (like r³). So there exists some maximum ring depth N beyond which a rigid embedding is impossible.

**The deep question:** Under what conditions does the growing {3,7} disk "snap" into a periodic or quasiperiodic structure in R³? And what structure is it snapping into?

**Zachary's answer (from vZome experiments with physical Zome models):** The target structure is the **Z[φ]³ icosahedral lattice** — the same lattice underlying icosahedral quasicrystals. Specifically: icosahedra connected by octahedral bridges, where the octahedra are placed on the icosahedron using the **tetrahedral subgroup A₄ ⊂ A₅** of the icosahedral symmetry group.

---

## The vZome File (237Lattice.vZome)

Zachary built a physical Zome model and digitized it in vZome. Analysis of the file reveals:

- **42 vertices, 101 edges** in Z[φ]³ (golden field coordinates)
- **15 degree-7 vertices** — these are the {3,7} tiling vertices
- **Three exact edge types** (all in Z[φ]):
  - Type A: sq = 18+21φ, length ≈ 7.210 (31 edges)
  - Type B: sq = 20φ² = 20+20φ, length ≈ 7.236 (15 edges) — the "purest" golden edge
  - Type C: sq = 20+24φ, length ≈ 7.670 (55 edges)
- All coordinates are of the form (a+bφ) with small integers a,b
- The three shortest vectors between degree-7 vertices are exactly the Type B edges
- This is a fragment of the icosahedral diamond lattice

---

## The Combinatorial Structure

The {3,7} tiling grows ring by ring. The ring sizes are:
- Ring 0: 1 vertex (center)
- Ring 1: 7 vertices
- Ring 2: 21 vertices
- Ring 3: 56 vertices
- Ring 4: 147 vertices
- Ring 5: 385 vertices
- Recurrence: ring(k) = 3·ring(k-1) - ring(k-2), with ring(1)=7, ring(2)=21

Cumulative: 1, 8, 29, 85, 232, 617, ...

Each boundary vertex is either **convex** (exclusive — connects to 1 inner vertex, needs 4 more edges) or **concave** (shared/cusp — connects to 2 inner vertices, needs 3 more edges). The number of concave vertices in ring k equals ring(k-1). This terminology appears in three independent derivations:
- Zachary's `growMesh.py`: "convex"/"concave"
- Daniel & Cypress's `main.rs`: "cusp"
- Claude237's `hyperbolic_triangle_gnn.py`: "shared"/"exclusive"

---

## The Code Lineage

1. **`main.rs`** (by Daniel Stroh and Cypress Robinson, students): Rust implementation. Builds the combinatorial graph and does physics relaxation to embed in R³. See also their repo: https://github.com/Qwanve/CS495

2. **`hyperbolic_triangle_gnn.py`** (by Claude237): Full rewrite. Replaced physics relaxer with a GNN (Graph Neural Network). Key features:
   - GNN predicts vertex positions for each new ring given previously embedded rings
   - Ring-by-ring relaxation (fixes rings 0..k-1, only relaxes ring k) — avoids self-intersections from simultaneous relaxation
   - Combined loss: MSE(predicted, target) + 0.3·Var(edge_lengths) — teaches equilateral triangles
   - Escalating backtrack: L0 (new GNN seed), L1 (re-embed ring k-1), L2 (full restart)
   - Pareto frontier tracking: (max_rings, min_edge_std)
   - Online retraining: physics fallbacks become new training data
   - Triangle-triangle collision detection

3. **`poincare_37.py`** (by Claude237): Exact Poincaré disk coordinates via Möbius transformations.
   - Hyperbolic parameters: vertex angle α = 2π/7, edge length a = acosh(cos α/(1-cos α)) ≈ 1.0906
   - `mob_translate(z, w)`: Möbius transformation T_w(z) = (z+w)/(1+conj(w)z)
   - `third_vertex(u, v, side)`: exact third vertex of hyperbolic equilateral triangle
   - BFS placement of all vertices
   - SVG renderer with geodesic arcs (orthogonal circles)
   - Edge length std = 0.0 (exact to floating point)
   - Outputs: `poincare_37.svg`, `poincare_coords.npy`

4. **`growMesh.py`** (by Zachary): Independent Python exploration. Pure combinatorial grower tracking convex/concave boundary. Uses spring layout + Kamada-Kawai for 3D positions.

5. **`growMesh2.py`** (by Zachary): More ambitious. Grows dynamically and uses **KDTree proximity detection** to find when a new vertex is spatially close to an existing boundary vertex — detecting the "snap" condition. This is the right instinct: snapping = finding a geometric coincidence between combinatorially distant vertices = a relation in π₁ of the quotient surface.

---

## The Klein Quartic Connection

The smallest compact quotient of the {3,7} hyperbolic tiling is the **Klein quartic**:
- 24 vertices (each degree 7), 84 edges, 56 triangles
- Euler characteristic: 24 - 84 + 56 = -4 → genus 3
- Automorphism group: PSL(2,7) of order 168 (maximum for genus 3 by Hurwitz)
- The 168 = 24·7 automorphisms act on the 24 vertices transitively

Note: 24 is NOT in the cumulative ring sequence (1, 8, 29, 85...), so the Klein quartic snap doesn't happen at a clean ring boundary. Some ring-2 vertices get identified with earlier ones. The fundamental domain cuts across a ring.

---

## What Was About to Be Built

When the OS update interrupted, we had just agreed to build:

**A rewrite of `growMesh2.py` that:**
1. Uses **exact Poincaré disk coordinates** (from `poincare_37.py`) instead of Kamada-Kawai positions
2. Replaces the KDTree snap detection with an **algebraic snap test**: two vertices snap when their Poincaré coordinates are related by a deck transformation — i.e., differ by an element of PSL(2,7) acting on H²
3. This turns the geometric coincidence search into an exact algebraic test
4. Would identify *which* identifications the Z[φ] lattice is trying to make
5. The snap events tell you the generators of the translation subgroup of the target periodic structure

**The PSL(2,7) deck transformations** act on the Poincaré disk as Möbius transformations. PSL(2,7) has order 168 and is generated by elements of orders 2, 3, and 7 corresponding to the symmetries of the Klein quartic. In the Poincaré disk model, the fundamental domain for the Klein quartic is a 14-gon (or equivalently, a union of 24/168 of the hyperbolic plane tiled by the {3,7} fundamental triangles).

---

## Key Mathematical Facts to Keep in Mind

- φ = (1+√5)/2 ≈ 1.618, φ² = φ+1, φ³ = 2φ+1, φ⁴ = 3φ+2, φ⁵ = 5φ+3
- The ring recurrence ring(k) = 3·ring(k-1) - ring(k-2) has characteristic equation x²-3x+1=0, roots = (3±√5)/2 = φ² and 1/φ² — so ring growth rate is φ²
- The angle defect per vertex is π - 7·(π/3 - 2π/7·...) — actually: in {3,7}, each triangle has angles 2π/7, so angle defect at each vertex = 2π - 7·(2π/7·3/2·...) — more precisely: internal angle of equilateral hyperbolic triangle = 2π/7, angle defect = 2π - 7·(2π/7) = 2π - 2π = 0? No: defect = 2π - sum of angles = 2π - 7·(2π/7) = 0 only for flat. For hyperbolic {3,7}: each triangle has angle sum < π, each vertex angle = 2π/7, seven of them sum to 2π — that's exactly flat! Wait — this is the special property of {3,7}: 7·(2π/7) = 2π. So vertex angle defect is ZERO. The curvature is in the triangles themselves, not the vertices. Angle defect per TRIANGLE = π - 3·(2π/7) = π(1 - 6/7) = π/7.
- The total angle defect for a disk of N triangles = N·π/7 (by Gauss-Bonnet for hyperbolic surfaces with the appropriate sign)

---

## Files in the Repo

- `237Lattice.vZome` — Zome model of icosahedra+octahedra lattice fragment
- `237Lattice.jpg`, `237Lattice_2.jpg` — photos
- `main.rs` — Rust combinatorial + physics (Daniel & Cypress)
- `hyperbolic_triangle_gnn.py` — GNN embedding (Claude237, latest version)
- `hyperbolic_triangle_gnn (7).py` — earlier GNN version
- `hyperbolic_triangle_gnn_old.py`, `hyperbolic_triangle_gnn_physics_based.py` — older versions
- `growMesh.py` — Zachary's combinatorial grower
- `growMesh2.py` — Zachary's snap-detecting grower
- `poincare_37.py` — exact Poincaré disk coordinates (Claude237)
- `poincare_37 (1).py` — variant
- `best_mesh.obj/.ply`, `gnn_mesh.obj/.ply`, `reference_mesh.obj/.ply` — saved meshes
- `best_model.pt` — saved GNN weights
- `pareto_log.json` — Pareto frontier log
- `immersed6338.png` — visualization

---

## Suggested First Task for New Instance

Implement the PSL(2,7)-aware snap detector. Here's the mathematical setup:

PSL(2,7) = GL(2,F₇)/center acts on P¹(F₇) (8 points) and also on the upper half-plane / Poincaré disk via Möbius transformations. The generators are:

```
S = [[0,-1],[1,0]]  (order 2, z -> -1/z)
T = [[1,1],[0,1]]   (order 7, z -> z+1)  
R = [[2,0],[0,1]]   (order 3 in PSL, z -> 2z ... actually need mod 7 version)
```

In the Poincaré disk, the {3,7} fundamental domain is bounded by 14 geodesic arcs. The deck transformations that generate the Klein quartic identification are the elements of PSL(2,7) viewed as hyperbolic isometries.

**Practical approach:** Rather than working out the PSL(2,7) generators from scratch, use the fact that in the Poincaré disk, two vertices v₁, v₂ are identified in the Klein quartic iff there exists an element g ∈ PSL(2,7) (as a hyperbolic isometry) such that g(v₁) = v₂. Since we have exact floating-point Poincaré coordinates for all vertices (from `poincare_37.py`), we can:

1. Generate all 168 elements of PSL(2,7) as 2×2 matrices over F₇
2. Lift each to a Möbius transformation on the disk (using the standard embedding of PSL(2,7) into PSL(2,R))
3. For each pair of boundary vertices, check if any of the 168 transformations maps one to the other (within tolerance)
4. Flag pairs that satisfy this as "snap candidates"

The snap candidates tell you which vertex identifications the {3,7} tiling is making when it closes up into the Klein quartic.
