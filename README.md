# 237

A question: What's the largest piece of the densest hyperbolic tiling (the {3,7} pattern, where seven equilateral triangles meet at every vertex) that can be built rigidly in 3D without self-intersection? Computational evidence and a proof sketch in progress point to 5 rings. The tiling is also a concrete example of a computationally accessible structure that encodes a hierarchical, exponentially-growing tree in a low-dimensional embedding. This makes the question of *learning* its embeddings (see "Learned embedding experiments" below), not just constructing them, worth asking.

## Background

The curvature trichotomy in mathematics connects a wide-ranging collection of ideas. The simplest case comes from 2-D geometry, where a surface can have positive curvature like a sphere, zero curvature like a Euclidean plane, or negative curvature like a saddle. By considering structures associated to these geometric objects, the trichotomy reveals facts and illustrative examples not only in geometry but also in algebra, number theory, analysis, mathematical physics, and beyond.

One aspect of the trichotomy that manifests in many contexts is visible when considering the three example surfaces of sphere, flat plane, saddle. The sphere is compact and finite in area. The flat plane is potentially infinite in extent, but it can be conveniently described by a well-behaved coordinate system. The saddle is another potentially infinite surface, and coordinate systems are more complicated. Answers to numerical questions posed of structures sorted into this trichotomy often come back with answers that are either explicitly "none", "finitely many" and "infinitely many" or can be rephrased as such.

This project investigates a structure that is minimal among an interesting category of discrete symmetric structures in the negative curvature sector of the trichotomy. The minimality is interesting because it distinguishes this structure among an infinite variety. Its simplest illustration consists of congruent equilateral triangles, joined edge to edge, with seven meeting at every vertex. I call this the {3,7} hyperbolic tiling.

The {3,7} tiling is a natural place to begin investigating this minimal structure. The minimality is evident in the hyperbolic area of the constituent triangles. All other symmetric tilings of the hyperbolic plane have a larger fundamental domain. Another wonderful and closely related example is the algebraic curve known as Klein's Quartic. Building physical models led me to a question: a physical model of a periodic tiling of the hyperbolic plane, like [this one](https://publicartarchive.org/art/Hyperbolic-Immersion/005697b6), will always run up against the fact that an exponentially growing area (the area of a hyperbolic disk grows exponentially in its radius) must fit in a volume that only grows cubically. This is almost certainly a reflection of a mathematical restriction. I assert that the rigidity of the model means there exists an N such that it is impossible to embed the surface consisting of all triangles within distance N from a given central triangle. I say almost certainly because a *continuous* embedding of the hyperbolic plane in R³ can be done (Nash–Kuiper), so any obstruction here has to come from rigidity and the discrete, non-self-intersecting structure of the tiling — not from continuity per se. But if N > 5, it would be very surprising.

For building intuition about this tiling, you might try playing [HyperRogue](https://en.wikipedia.org/wiki/HyperRogue) ([Steam](https://store.steampowered.com/app/342610/HyperRogue/)).

## The embedding question

The central question is: what is the maximum number of rings of equilateral triangles that can be embedded in R³ following the {3,7} pattern without self-intersection?

Computational experiments using a physics-based relaxer (spring forces between vertices, repulsion between non-adjacent faces) suggest the answer is **N = 5**. Across 500 independent trials:

- Rings 1–3 embed with edge standard deviation ≈ 0.000 (essentially perfect equilateral triangles)
- Ring 4 embeds with edge std ≈ 0.005
- Ring 5 embeds in 34.6% of trials, with edge std ≈ 0.047
- Ring 6 collides in 100% of trials

A proof sketch for the existence of a finite N, with a conjectured N = 5, is under development, based on:

- Physical realization with ε-thick tiles, where the volume constraint becomes active
- A **snapping** mechanism: when two vertices come within ε with compatible edge directions, they merge, inheriting all edges from both — potentially triggering a cascade of further snaps. This is to deal with potential embeddings with near-foliated regions.
- The cascade is constrained combinatorially and algebraically. Coxeter group structures related to the hyperbolic triangle group Δ(2,3,7) appear incompatible with Euclidean symmetries.
- All structures satisfying the algebraic/combinatorial constraint develop a contradiction as N increases, and this contradiction persists as ε → 0.

## Graph realization vs. surface embedding

These are different problems with different answers, and the distinction matters for what kind of proof this is. The {3,7} graph's *unit-distance realization* — satisfying every edge-length constraint with no restriction on self-intersection — certifies well past ring 6 in R³; vertices and faces simply pass through each other where the surface would collide. Only when self-intersection is forbidden does the ring-5 ceiling appear. The same distinction in R⁴ gives a graph realization certified to ring 11, versus a self-intersection-respecting surface estimate around ring 10.

This was checked deliberately rather than assumed, because a purely algebraic failure was a live alternative hypothesis: in R⁵, the {3,3,5} polytope's distance system already produces a Gram matrix with 29 negative eigenvalues, ruling out *any* real point configuration past ring 1 — crossing or not. The {3,7} disk's metric system shows no such pathology at any depth tested. The obstruction is specifically about embeddedness, which is what justifies building the capacity- and link-lemma machinery below rather than looking for a Cayley–Menger-style rigidity argument instead. (This embedded-vs-immersed split is the same territory Connelly's work on flexible polyhedra and generic rigidity theory addresses more generally: the same distance data can force a unique shape, or not, depending on whether self-intersection is allowed.)

## The lattice connection

Gluing icosahedra together using octahedra as bridges produces a diamond-like lattice living in Z[φ]³ (coordinates with entries of the form a + bφ, where φ is the golden ratio). This leads to further questions:

- Under what rules will these lattices spontaneously emerge from a growing shape?
- Can the lattice structure and representations of Fuchsian groups — in particular PSL(2,7), the minimal algebraic example associated with Klein's Quartic curve — be used to efficiently work with hierarchical data in relatively low dimensions?

## Visualization

The interactive animation `boundary_racing.html` shows points sweeping out the ring boundary cycles, with a synchronized Poincaré-disk view of the same race.

[![boundary racing screenshot](examples/boundary_racing.png)](https://ztreisman.github.io/237/examples/boundary_racing.html)

## Learned embedding experiments

`ml_experiments/` documents an attempt to train a model to predict valid {3,7} embeddings directly from combinatorial structure alone, with no geometric input. Two architectures were tried — direct cold-start prediction, and a diffusion-style denoising approach with noise-level conditioning, curriculum training, and iterative refinement — and neither beat a physics-relaxer baseline from random initialization. The likely cause is architectural rather than a training failure: local message-passing GNNs propagate information a fixed number of hops per layer, while valid {3,7} embeddings require globally coordinated folding across the whole graph. No amount of training fixes a receptive field that's structurally too small for the problem.

A more useful result came out of the same investigation: a reliably reproducible population of graphs — the {3,7} ring-5 disk with length-2 chords at ≥60% density — where a certified unit-distance embedding exists, but cold-start L-BFGS fails to find it 100% of the time across many random seeds. See `ml_experiments/udg_cold_start_eval.py`.

## Contents

| Path | Description |
|---|---|
| `theory/capacity_lemma.md` | Demand identity, tetrahedral wall, confinement lemma (proven); a conditional capacity bound, demoted to motivational only — see the file for why |
| `theory/link_lemma.md` | Spherical link analysis at each vertex; per-vertex turning conservation (proven) |
| `theory/turning_definitions.md` | Intrinsic/extrinsic turning, the bridge inequality |
| `theory/CLAUDE237_CONTEXT.md`, `CONTEXT_pleat_analysis.md` | Background context notes from exploratory sessions |
| `embedding/probe_depth.py` | Resumable ring-depth certification probe, with an optional collision-repulsion mode for distinguishing graph realization from surface embedding |
| `embedding/generate_unit_distance_library.py` | {p,q} disk factory with chord augmentation, used for the cold-start experiments |
| `embedding/run_campaign.sh` | Unattended multi-stage calibration and measurement orchestration |
| `python/build_library.py` | Large-trial {3,7} library generator with gap statistics — produced the headline 500-trial, ring-5 result |
| `python/physics_grow.py`, `physics_grow_snap.py` | Physics-based ring grower (spring relaxer), and a variant with vertex snapping (ε-merging) |
| `python/physics_grow_zip.py` | Zip-cascade variant: snaps boundary *edge pairs* rather than individual vertices, so each step is a manifold-safe double merge; committed pairs attract rather than repel |
| `python/poincare_37.py` | Exact Poincaré disk coordinates and SVG rendering for the {3,7} tiling; source of the coordinate data used by `boundary_racing.html` |
| `python/gap_analysis.py`, `gap_geometry.py` | Spherical gap and azimuthal supply/demand profiling |
| `python/pleat_analysis.py`, `fold_propagation.py`, `fold_ab_analysis.py` | Fold/headroom metrics, the compensation law, and A/B boundary-word structure |
| `python/hyperbolic_triangle_gnn.py` | Early GNN-assisted ring embedder; superseded by the work in `ml_experiments/` but kept for the record |
| `python/growMesh.py`, `growMesh2.py`, `growMesh3.py` | Earliest precursor scripts, predating the Rust implementation below |
| `python/crystal_neighborhoods*.py`, `analyze_snaps.py` | Exploratory work on the vertex-snapping mechanism; not concluded |
| `analysis/` | Generated data: per-trial library results and gap-geometry measurements, produced by `python/build_library.py`, `python/gap_analysis.py`, and `python/poincare_37.py` |
| `ml_experiments/` | Learned embedding attempts and the cold-start difficulty characterization — see above |
| `rust/main.rs` | Original Rust relaxer, the starting point before the Python ports |
| `boundary_racing.html` *(in `examples/`)* | Interactive boundary-cycle racing animation |
| `vZome/` | Hand-colored {3,7} disk and extended-disk files in the icosahedral lattice |
| `examples/` | Renders, animations, and supporting visual assets |

## Related work

- Student work by Daniel Stroh and Cypress Robinson: [github.com/Qwanve/CS495](https://github.com/Qwanve/CS495)
