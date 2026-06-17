# Learning Unit-Distance Graph Embeddings from Hyperbolic Surface Data
# Research proposal sketch — Zack Treisman, 2026-06-04
# (Direction 4 of the {3,7} project)

## The idea in one paragraph

Use physics-relaxer embeddings of hyperbolic tiling disks ({3,7} and other
{p,q}) in R^n as a factory for SOLVED instances of the unit-distance graph
embedding problem. Augment each embedded surface with chord edges — pairs of
vertices that happen to lie at distance ≈1 (or 2, or any prescribed length)
in R^n but are far apart in the graph — producing graphs whose abstract
structure no longer reveals their hyperbolic origin but which carry a known
unit-distance embedding by construction. Train a model (GNN → coordinate
regression) on these (graph, embedding) pairs to map abstractly defined
graphs to unit-length-edge embeddings in R^n with n as small as possible.
Goal: beat the physics relaxer.

## Why the data trick matters

Unit-distance graph recognition/embedding is hard in the formal sense
(∃R-complete in the plane — Schaefer), and like most hard combinatorial
geometry problems, its ML bottleneck is the absence of solved instances at
scale. The hyperbolic disk + chord construction manufactures them:

1. **Base surfaces.** Relaxer embeddings of {p,q} disks: {3,7} in R³
   (rings 0–5, 617 vertices, 500-trial library exists), {3,7} in R⁴
   (rings 0–9, ~29k vertices — sizes 1,7,21,56,147,385,1008,2639,6909,18088),
   plus {3,8}, {4,5}, {4,6}, {5,5}, ... each with its own r(p,q,n) frontier.
2. **Chord augmentation.** The embedded surface approaches itself in R^n.
   Scan vertex pairs at Euclidean distance within tolerance of a target
   length L ∈ {1, 2, ...} that are graph-distant; add them as edges
   (optionally labeled with L). Each base embedding yields combinatorially
   many augmented graphs (sample chord subsets).
3. **Camouflage.** Dense chords destroy the recognizable tree-like /
   hyperbolic signature — the model cannot succeed by classifying the graph
   as "a {3,7} disk" and pasting a memorized layout. Mixing {p,q} families,
   gluing patches, and varying chord density gives a difficulty dial.

Scale estimate: 500 R³ trials + R⁴ runs, × hundreds of chord subsets each
→ 10⁴–10⁵ solved instances, 600 to ~30k vertices.

## The model

Input: abstract graph (adjacency; optional per-edge prescribed lengths).
Output: coordinates in R^n. Loss: per-edge length error (relaxer's edge_std
as the metric), plus optional collision/self-intersection penalties.

The closest analogy is AlphaFold: protein structure determination IS a
prescribed-distance embedding problem (NMR constraints → conformation), and
its breakthrough was a data trick (evolutionary couplings) plus iterative
refinement. Here the data trick is hyperbolic manufacture. Related but
weaker precedents: neural graph drawing (DeepGD), learned distance-geometry
/ SDP-initialization solvers.

Architectural notes:
- Equivariance: target is defined up to E(n); use an E(n)-equivariant GNN
  (EGNN-style) or predict a Gram matrix / distance matrix and factor.
- Dimension minimization: train with an n-penalty (e.g., predict in
  generous n, penalize variance in trailing coordinates; or train a family
  of fixed-n heads and select).
- Iterative refinement head mirroring relaxer dynamics may help; or use the
  model purely as the relaxer's initializer (hybrid).

## What "beat the relaxer" means (metrics)

Baselines from the existing library: 34.6% of trials reach ring 5;
edge_std ≈ 0.047 at best; 0% reach ring 6.
- Success rate at fixed ring depth / fixed graph family
- Final edge_std (and max edge error) at convergence
- Wall-clock to solution
- Coverage: does the model find ring-5 embeddings in basins the relaxer
  misses? (Compare embedding diversity, e.g., via gap statistics from
  gap_analysis.py.)
- Generalization: performance on graphs NOT from the generator — random
  unit-distance graphs, matchstick graphs, classical hard instances
  (Moser spindle, Harborth graph), DIMACS-style distance-geometry sets.

## Connection to the theory (why this isn't a separate project)

- The pleat-capacity story (Gauss–Bonnet floor: total intrinsic turning
  2π + (π/3)V_{r−1}; absorption requirement exactly 1/φ per ring; capacity
  c(r) decaying) describes WHAT a successful embedding must do: distribute
  folds. A model that learns to embed {3,7} disks has implicitly learned
  discrete pleating — where to put the folds the relaxer finds only by
  trial and error. Interpretability question: do the model's internal
  representations track fold placement / the A–B substitution sequence
  (σ: A→AB, B→ABB)?
- Chords have a group-theoretic reading: chords between Δ(2,3,7)-orbit-
  related patches encode partial quotients — the graph "remembers" the
  surface's self-approach. Curriculum from sparse to dense chords
  interpolates from disk toward quotient-like structures.
- The r(p,q,n) frontier provides the difficulty ladder: training near the
  frontier (ring 5 in R³, ring 9 in R⁴) is exactly where the relaxer
  struggles, so that's where learned global structure should pay off most.
- Direction 3 (LLM embedding capacity) gets an empirical sibling: if token
  graphs are effectively hyperbolic (ICLR 2025 hyperbolicity findings),
  a model that embeds hyperbolic-with-chords graphs well is a candidate
  architecture for token-geometry-aware embedding layers.

## Concrete first steps

1. Chord-augmentation script: load library PLY → kd-tree pair scan at
   target lengths {1.0, 2.0} with tolerance (start ±0.02) → emit
   (graph, embedding) JSON/PT pairs. Cheap; runs on laptop.
2. Dataset v0: ring-4 and ring-5 {3,7} disks from the existing library,
   3 chord densities × 100 subsets each. Train/val split by base trial
   (no leakage of the same surface across splits).
3. Model v0: EGNN, ~6 layers, edge-length loss; n=3 fixed. Compare against
   relaxer-from-random-init on held-out graphs: success rate and time.
4. Only after v0 works: R⁴ data, mixed {p,q}, dimension-minimization,
   and the hard external benchmarks.

## Honest risks

- The model may just learn the generator's distribution and fail off it —
  the generalization tests in the metrics section are the real bar.
- Chord tolerance creates slightly-non-unit edges in the "ground truth";
  decide whether to re-relax after chord addition (recommended: yes, a short
  relaxer polish with chords as constraints, keeping only instances that
  converge — this also certifies the augmented graph IS unit-distance).
- Equivariant GNNs on 30k-vertex graphs are trainable but not trivial;
  start small (ring 4, ~230 vertices).
