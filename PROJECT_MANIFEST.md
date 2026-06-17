# 237 Project Manifest — Repo Cleanup Brief

This document exists because the work was developed across many conversations
with an assistant that has full context; whoever reorganizes this repo
(a fresh Claude Code session, most likely) does not have that context by
default. Read this fully before moving/deleting anything.

## 0. Consolidation status (pinyon is the target machine)

Confirmed: cold-start ML work already ran on **pinyon** (Zack's laptop,
Linux), which is also the machine being used to edit this manifest. That
makes pinyon the natural consolidation point. Remaining steps to get
everything in one place:

1. **Already on pinyon**: `udg_cold_start_eval.py`, `udg_library/` (411
   instances), `cold_start_aug.jsonl`, `cold_start_aug.log`,
   `train_udg_embedder.py`, `train_denoising.py`.
2. **Download to pinyon now** (available from claude.ai chat outputs):
   `probe_depth.py`, `run_campaign.sh`, `generate_unit_distance_library.py`,
   plus — if not already local — the theory docs (`turning_definitions.md`,
   `capacity_lemma.md`, `link_lemma.md`, `RESULTS_addendum.md`),
   `boundary_racing.html`, and `unit_distance_learning_proposal.md`.
3. **Pull from the existing GitHub repo** onto pinyon (`git clone` or
   `git pull` github.com/ztreisman/237): `physics_grow.py`,
   `physics_grow_snap.py`, `build_library.py`, `gap_analysis.py`,
   `gap_geometry.py`, `color_disk.py`, `hyperbolic_triangle_gnn.py`,
   `new_journal_entry*.txt`, vZome files, the current README.
4. **Stays on lambda01, does not need to move**: the raw probe/campaign
   output (`probe_q7_d*.jsonl`, `campaign_*/` directories). Per §2 below,
   these are excluded from git anyway. EXCEPTION: the `.jsonl` result logs
   themselves are tiny (one line per ring) and are the actual evidence
   behind the ring-11 and graph-vs-surface claims in `RESULTS_addendum.md`
   — worth `scp`-ing those specific small files down even though the large
   `campaign_*/` directories and raw library `.npz` files are not.

Once steps 2–3 are done, pinyon has everything needed for the
reorganization in §3, and a fresh Claude Code session pointed at pinyon's
checkout can do the actual `mv`/`git mv` work directly.

## 1. Known naming collisions / ambiguities to resolve first

- **`physics_grow.py` vs `physics_grow_snap.py`** — the snap version (with
  ε-vertex-merging, `SNAP_EPSILON=0.25`) is the later, more developed one.
  Confirm which is still in active use; consider archiving the other under
  `legacy/` rather than deleting (it's the historical baseline the proof's
  empirical claims were partly built on).
- **`build_library.py` vs `generate_unit_distance_library.py`** — these are
  NOT duplicates despite similar names. `build_library.py` grows the bare
  {3,7} disk and computes gap statistics (this produced the 500-trial,
  34.6%-ring-5-success result the whole proof program rests on).
  `generate_unit_distance_library.py` is the later ML-oriented tool: same
  disk factory, generalized to {p,q}, plus chord augmentation for the
  unit-distance-graph learning experiments. Rename for clarity if convenient
  — e.g. `grow_tiling_library.py` and `generate_chord_library.py` — but
  document the distinction in the README regardless.
- **`hyperbolic_triangle_gnn.py`** — this is an early, more primitive GNN
  experiment, predating the EGNN work below. Confirm whether it's superseded
  by `train_udg_embedder.py`/`train_denoising.py` or explores something
  different; if superseded, move to `legacy/` rather than the main ML folder.
- **`boundary_racing.html`** — had a `buildParam`/`buildS` typo bug (fixed),
  plus fixes for Poincaré-disk/3D-position mismatch, disk placement, and
  removal of phantom non-surface edges. Confirm the copy going into the repo
  is the patched version, not an earlier one.

## 2. File-by-file status and recommended disposition

### Theory (core proof documents — polish, keep prominent)
| File | Status |
|---|---|
| `turning_definitions.md` | Core. Defines intrinsic/extrinsic turning, the bridge inequality. |
| `capacity_lemma.md` | Core. Demand identity (proven), tetrahedral wall (proven trichotomy), confinement lemma (proven, not binding), conditional finiteness (demoted to motivational). |
| `link_lemma.md` | Core. Spherical link analysis, per-vertex conservation (proven), C_link bound. |
| `RESULTS_addendum.md` | Core. All empirical findings: fold/compensation analysis, the graph-vs-surface separation (ring(2,3) graph realizes past 6, surface doesn't), the r(2,4) measurement. **This is the file that ties theory to experiment — make sure it's current and linked from the README, not buried.** |

### Core embedding pipeline (canonical scripts)
| File | Status |
|---|---|
| `physics_grow_snap.py` | Canonical grower. Keep. |
| `physics_grow.py` | Superseded — archive under `legacy/` with one-line note. |
| `build_library.py` | Canonical — produced the headline 500-trial result. Keep prominent. |
| `generate_unit_distance_library.py` | Canonical for the ML-data side. Keep, but rename/clarify per §1. |
| `probe_depth.py` | Canonical — the resumable, repulsion-capable depth probe used for the ring-11 (R⁴ graph) and the R³/R⁴ graph-vs-surface control runs. Keep prominent. |
| `run_campaign.sh` | Canonical — the unattended lambda01 orchestration script (calibration → measurement stages). Keep. |

### Analysis scripts (supporting evidence for the proof)
| File | Status |
|---|---|
| `gap_analysis.py` | Keep. |
| `gap_geometry.py` | Keep. |
| `pleat_analysis.py` | Keep — fold/absorption metrics, headroom collapse at ring 5. |
| `fold_propagation.py` | Keep — refutes naive crease-propagation, establishes compensation. |
| `fold_ab_analysis.py` | Keep — A/B word structure, conservation law, per-sector demand. |

### Visualization
| File | Status |
|---|---|
| `boundary_racing.html` | Keep — confirm patched version (see §1). |
| `color_disk.py` | Keep. |
| `lattice_unit_colored.vZome` | Keep. |
| `disk_extended_*.vZome` (rings 0–17) | Keep, but consider whether all ring depths need to ship in the repo or just a representative few — these can be large. |

### ML experiments — mixed results, organize to tell an honest story
This is the section most likely to confuse an outside reader if dumped
flat into one folder. There are two genuinely different outcomes here and
the repo structure should make the distinction visible at a glance:

**Negative result (learned embedding from cold start does not work):**
- `hyperbolic_triangle_gnn.py` — early attempt, status TBD (see §1).
- `train_udg_embedder.py` — EGNN v0, cold-init, 0% cold-start success vs.
  93% physics-relaxer baseline. Wrong training domain (data was too easy).
- `train_denoising.py` — EGNN + diffusion-style denoising + noise-level
  conditioning + curriculum + T=8 iterative refinement at eval. Still 0%
  cold-start success. Root cause: 6-layer message passing cannot coordinate
  vertices across a graph diameter of 8–10 hops; the fold structure
  requires genuinely global coordination that no fixed-depth local GNN can
  represent, regardless of training procedure.

**Positive result (characterizing where the physics relaxer itself struggles):**
- `udg_cold_start_eval.py` — this is the actually-successful piece of the
  ML direction. Found a clean, reproducible population of graphs (ring-5
  {3,7} disk + length-2 chords at ≥60% density) where a certified
  unit-distance embedding exists but cold-start L-BFGS fails 100% of the
  time (32/32 instances, 6 cold starts each). This is a real result about
  the difficulty landscape of unit-distance realization, independent of
  whether the learned embedder ever works.
- `unit_distance_learning_proposal.md` — the original proposal; keep as
  historical record of the plan, with a note pointing to what actually
  happened.

Recommend a short `ml_experiments/README.md` stating this distinction
plainly: the learned embedder failed and here's the architectural reason
why; the cold-start characterization succeeded and here's what it found.
Stating a negative result clearly reads as more credible than omitting it.

### Logs / notes
| File | Status |
|---|---|
| `new_journal_entry*.txt` | Probably exclude from the public repo, or fold the proof-relevant content into the theory docs and keep the raw logs in a private/local-only journal directory (not pushed). |

### Data libraries — exclude from git, document regeneration instead
`udg_library/`, `cold_start_aug.jsonl`, `probe_q7_d*.jsonl`, `campaign_*/`
directories: these are regenerable from the scripts above and some are
sized awkwardly for git (the udg_library is ~14MB; campaign directories
could be larger). Add a `.gitignore` entry and instead document the exact
commands used to regenerate each, e.g. in a `DATA.md`:
```
# regenerate the cold-start hard-instance population
python3 generate_unit_distance_library.py --q 7 --dims 3 --rings 5 \
    --chord-lengths 2 --densities 0.6
```
If a specific result needs to ship with the repo for reproducibility
(e.g. the exact 411-instance library that produced the cold-start table),
consider a GitHub Release attachment rather than committing it to the
tree.

## 3. Suggested target structure

```
237/
├── README.md
├── theory/
│   ├── turning_definitions.md
│   ├── capacity_lemma.md
│   ├── link_lemma.md
│   └── RESULTS_addendum.md
├── embedding/
│   ├── physics_grow_snap.py
│   ├── build_library.py
│   ├── generate_unit_distance_library.py   (or renamed, see §1)
│   ├── probe_depth.py
│   └── run_campaign.sh
├── analysis/
│   ├── gap_analysis.py
│   ├── gap_geometry.py
│   ├── pleat_analysis.py
│   ├── fold_propagation.py
│   └── fold_ab_analysis.py
├── visualization/
│   ├── boundary_racing.html
│   └── color_disk.py
├── vzome/
│   ├── lattice_unit_colored.vZome
│   └── disk_extended_*.vZome
├── ml_experiments/
│   ├── README.md                 (state the negative/positive split plainly)
│   ├── udg_cold_start_eval.py
│   ├── train_udg_embedder.py
│   ├── train_denoising.py
│   └── unit_distance_learning_proposal.md
├── legacy/
│   ├── physics_grow.py
│   └── hyperbolic_triangle_gnn.py   (if confirmed superseded)
├── examples/
│   └── boundary_racing.png
├── DATA.md                        (regeneration commands for excluded data)
└── .gitignore                     (udg_library/, *.jsonl, campaign_*/, etc.)
```

## 4. What NOT to do

- Don't delete anything during this pass — move to `legacy/` instead.
  Several "superseded" scripts (e.g. `physics_grow.py`) produced results
  the proof currently cites; deleting them would orphan that provenance.
- Don't flatten the ML experiments into a single undifferentiated folder —
  the negative/positive distinction is the whole point, see §2.
- Don't commit the data libraries directly (§2, data libraries) — they're
  regenerable and some are large.
