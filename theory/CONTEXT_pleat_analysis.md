# CONTEXT: {3,7} Pleat Capacity Analysis — Continue in Claude Code
# Created: 2026-06-04 (claude.ai session)
# User: Zack Treisman (ztreisman), WCU mathematics

## IMMEDIATE TASK

Adapt `pleat_analysis.py` to read PLY trial files, then run it against the
500-trial library and interpret the results against the buckling hypothesis.

The script is complete and tested EXCEPT `load_trial()` assumes .npy/.json.
The library files are **PLY format**. Fix: parse PLY vertex positions
(ascii or binary little-endian; check header). Use plyfile
(`pip install plyfile`) or write a minimal parser. The positions array must
be (617, 3) for a complete ring-5 trial, **in the relaxer's vertex order**.

⚠ CRITICAL CORRECTNESS ISSUE — VERTEX ORDERING:
The script builds its own combinatorial {3,7} tiling (ring sizes
[1,7,21,56,147,385], verified correct, 847 faces, all interior degree 7).
The analysis is only valid if trial vertex i = combinatorial vertex i.
The relaxer (physics_grow.py / Rust predecessor) may use a different
creation order. VERIFY before trusting results:
  - Check edge lengths: for every edge (u,v) in the script's adjacency,
    |P[u]-P[v]| should be ≈ 1.0 (edge_std ≈ 0.047 for good ring-5 trials).
    If many edges have wildly wrong lengths → ordering mismatch.
  - If mismatched: recover the permutation by matching graph structure
    (ring 0 = vertex nearest centroid / origin; grow outward by nearest-
    neighbor matching at unit distance), or read the relaxer source to get
    its creation order. The relaxer repo: github.com/ztreisman/237.
  - PLY files may also contain face data — if so, USE THE PLY FACES to
    build adjacency instead of the script's combinatorial model. That
    sidesteps the ordering problem entirely and is the preferred fix.

Library location: lambda01 (`fac_treisman@lambda01.wsc.western.edu`),
built by `build_library.py`, likely under ~/237/. ~500 trials; 34.6% reach
ring 5 (best edge_std ≈ 0.047); 0% reach ring 6. Trial 196 = lowest ring-6
collision edge_std (0.04671); trial 63 = best spherical gap (16.8°).
Colleague Andy has sudo on lambda01.

## THE THEORY BEING TESTED (developed today — this is the new part)

### Established framework (full writeup: turning_definitions.md)

Definitions, for the boundary polygon ∂P_r of the embedded disk P_r
(rings 0..r, unit equilateral triangles in R³):

- **Extrinsic turning** at boundary vertex v: κ_ext(v) = π − ∠_R³(e₁,e₂),
  the actual bend of the space polygon at v.
- **Intrinsic turning**: κ_int(v) = π − m(v)·π/3 where m(v) = # interior
  triangles at v. Pure combinatorics:
  - Type A vertex (2 parents in ring r−1): m=3 → κ_int = 0 (straight)
  - Type B vertex (1 parent): m=2 → κ_int = π/3 (60° turn)
- **Bridge inequality**: κ_ext(v) ≥ κ_int(v) at every boundary vertex
  (spherical triangle inequality on the fan of interior edges at v).
  Equality iff the fan unfolds flat; pleating ⇒ strict excess.

Key combinatorial facts (all verified against tiling data):
- n_A(r) = N_{r−1}; n_B(r) = N_r − N_{r−1}
- Gauss–Bonnet (interior curvature −π/3 per vertex, disk χ=1):
  **n_B(r) = 6 + V_{r−1}** where V_{r−1} = Σ_{k<r} N_k. Verified exactly
  for r=2..6: (14,35,91,238,623) = 6+(8,29,85,232,617).
- Total intrinsic turning T(r) = 2π + (π/3)V_{r−1} ~ φ^{2r}. Exponential.
- Ring sizes: 1,7,21,56,147,385,1008,... recurrence a_r = 3a_{r−1} − a_{r−2},
  growth rate exactly φ² = (3+√5)/2.
- **Boundary word is a substitution sequence**: σ: A→AB, B→ABB.
  Ring 2 = (ABB)⁷, ring 3 = σ(ring 2) up to rotation (verified).
  Substitution matrix [[1,1],[1,2]], leading eigenvalue φ². A:B ratio → 1:φ.
  B-runs have length exactly 1 or 2, never 0, never ≥3.

### The buckling hypothesis (TO TEST)

The annular strip between ring r−1 and ring r connects an inner polygon of
length N_{r−1} to an outer polygon of length φ²·N_{r−1}. A developable
(unpleated) strip can lengthen its outer boundary only by ~2π. Therefore the
strip must absorb an excess fraction of exactly

    1 − φ⁻² = **1/φ ≈ 0.618**

of its outer length via pleating (dihedral folds), EVERY ring. Pleat capacity
c(r) is finite and DECREASES with r because folds at ring r stack on folds
already used at inner rings. Hypothesis:

    **c(5) ≥ 1/φ > c(6)** — the golden threshold is crossed at ring 6,
    which is why r(2,3) = 5.

Honest caveats (raised by Zack, incorporated):
- Solid-angle counting alone is NOT a proof — non-adjacent vertices can be
  arbitrarily close; only edge-connected pairs are distance-constrained.
- The boundary curve can wander anywhere in B(0,r)\{0}; no normal field is
  canonical on a polygon. The dihedral angles along interior edges are the
  honest degrees of freedom. The triangle normals + Gauss map are the
  right smooth proxies if needed.
- Naive constraint counting gives ratio ~0.47–0.50 always (underdetermined);
  the obstruction is feasibility (no REAL solutions), not overdetermination.
  Local turn directions are coupled through shared interior triangles —
  zigzag pleating is constrained, and conflicting convexity between adjacent
  "coils" is the buckling mechanism.

## WHAT pleat_analysis.py MEASURES

Per trial, per ring band r=1..5:
1. **fold_used**: mean |π − dihedral| over band edges (pleat budget consumed)
2. **fold_max**: max fold in band
3. **turn_excess**: mean (κ_ext − κ_int) on ring-r boundary — extrinsic
   curvature paid ABOVE the Gauss–Bonnet floor; this is normal curvature
   i.e. pleating, measured a second independent way
4. **absorption**: 1 − 2πR_r/N_r where R_r = mean distance of ring-r
   vertices from their centroid — fraction of unit-edge length "eaten"
   vs presented circumference. Compare against 1/φ = 0.618.

Expected signature if hypothesis is right: absorption ≥ 1/φ with margin for
small r, margin shrinking toward 0 at r=5; fold_used rising toward a ceiling;
turn_excess growing. If absorption at r=5 is well above 1/φ with no trend,
the hypothesis needs revision.

Analysis ideas beyond the script: distribution (not just mean) of dihedrals
per band; spatial autocorrelation of fold direction around the band (pleats
should alternate — check coupling); compare trial 196 / trial 63 (the best
failures) against typical ring-5 successes; absorption using best-fit-plane
projected perimeter instead of radius-of-gyration proxy (more honest).

## BROADER PROJECT CONTEXT

r(n,d) = max rings of a hyperbolic tiling in H^n isometrically (unit-edge)
embeddable in R^d. Established: r(2,3)=5 (empirical, 500 trials);
r(2,4)≥9; r(3,4)=4 ({3,3,5}); r(4,5)≤1 (Gram matrix has 29 negative
eigenvalues — impossible); no compact regular honeycombs in H⁵⁺.

Three proof threads, now unified:
1. **Algebraic mismatch**: Δ(2,3,7) = <a,b,c | a²=b³=c⁷=abc=1> has no
   faithful rep in O(3) (it's infinite; surjects PSL(2,7), order 168, whose
   smallest faithful real rep is 6-dim). No equivariant embedding exists.
2. **Ratner/equidistribution**: Δ(2,3,7) cocompact in PSL(2,R) → every
   horocycle equidistributes (Dani–Margulis); ring vertices approach
   horocycles; in R^d the directions equidistribute on S^{d−1}. Explains
   WHY the constraints fill out. (Motivating, not directly the obstruction.)
3. **Gauss–Bonnet/buckling** (TODAY, the most promising): intrinsic turning
   forced combinatorially, extrinsic ≥ intrinsic, pleating must absorb 1/φ
   per ring, capacity decays. This is the thread the PLY analysis tests.

People to contact once the story is solid: Anna Gilbert (Yale, "Shedding
Light on Problems with Hyperbolic Graph Learning" TMLR 2025 w/ Isay
Katsman — their critique: no good theory of when hyperbolic helps; r(k,d)
is that theory). Frederic Sala (Wisconsin, ICML 2018 tree-embedding
constructions). Cold-email hook: growth rate φ², obstruction exactly ring 6
in R³, Gauss–Bonnet mechanism.

## FILES

- pleat_analysis.py — the script (tiling builder verified ✓; fix load_trial
  for PLY; prefer PLY faces for adjacency if present)
- turning_definitions.md — full writeup of today's definitions/lemma
- solid_angle_theorem.md — earlier (superseded in approach but the capacity
  framing survives)
- GitHub: github.com/ztreisman/237
- Animation: boundary_racing.html (linked radial-sweep racers + Poincaré
  disk inset, working)

## NEXT STEPS AFTER THE ANALYSIS RUNS

1. If signature confirms: formalize c(r) — define pleat capacity precisely
   (sup of absorbable length given inner-ring dihedral state), prove
   monotone decay, prove threshold crossing. This + Gauss–Bonnet floor =
   the theorem.
2. Same analysis in R⁴ (grow_37_4d.py output): absorption requirement is
   identical (1/φ) but capacity is larger — measure how much larger, predict
   r(2,4) and check against ≥9.
3. Write the paper: Gauss–Bonnet section is the spine; Ratner as context;
   algebraic mismatch as the equivariant special case.

## DIRECTION 4 (new, 2026-06-04): Learning unit-distance embeddings

Full sketch in unit_distance_learning_proposal.md. Core idea: relaxer
embeddings of {p,q} disks in R^n + chord augmentation (add edges between
vertex pairs at Euclidean distance ≈ prescribed length L that are graph-
distant) = factory for SOLVED unit-distance embedding instances. The chords
camouflage the hyperbolic origin. Train an E(n)-equivariant GNN to map
abstract graphs → unit-edge embeddings, minimal n. Goal: beat the physics
relaxer (baselines: 34.6% ring-5 success, edge_std 0.047).

First concrete step (can be done alongside the pleat analysis, same PLY
loading code): chord-augmentation script — kd-tree pair scan at target
lengths {1.0, 2.0} ± 0.02 over library trials, emit (graph, embedding)
pairs, then short relaxer re-polish with chords as constraints to certify.
Note the unit-distance decision problem is ∃R-complete (Schaefer) — this is
why manufactured solved instances are valuable. AlphaFold analogy: distance
constraints → structure, with a data trick replacing evolutionary couplings.
