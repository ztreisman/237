#!/usr/bin/env python3
"""
pleat_analysis.py — Measure pleat capacity c(r) from the ring-5 trial library.

THEORY (2026-06-04 session):
The boundary of P_r must absorb a fraction 1/φ ≈ 0.618 of its length via
pleating (dihedral folds), every ring. The pleat capacity c(r) decreases as
inner-ring folds accumulate. Embedding survives iff c(r) ≥ 1/φ.
Hypothesis: c(5) ≥ 1/φ > c(6) — the golden threshold is crossed at ring 6.

WHAT THIS SCRIPT MEASURES, per trial and per ring-band:
1. Dihedral angle along every interior edge (fold angle distribution)
2. "Fold budget used": deviation of dihedrals from flat (π)
3. Boundary turning: extrinsic turning at each ring-r vertex vs the
   intrinsic floor (0 for type A, π/3 for type B)
4. Length absorption: how much outer-boundary length each band's pleating
   actually absorbs, compared to the required 1/φ fraction.

USAGE:
  python3 pleat_analysis.py /path/to/library_dir
  # library_dir contains trial files: trial_NNN.json or .npy with vertex
  # positions. Adjust load_trial() below to match the actual format.

OUTPUT:
  pleat_report.txt — per-ring summary stats across all trials
  pleat_data.npz   — raw arrays for further analysis

Runs in seconds. numpy only.
"""

import sys, os, json, glob, math
import numpy as np
from collections import defaultdict, deque

PHI = (1 + math.sqrt(5)) / 2
INV_PHI = 1 / PHI  # 0.618... the critical absorption fraction

# ----------------------------------------------------------------------------
# 1. Build the {3,7} combinatorial structure (rings 0..5, 617 vertices)
# ----------------------------------------------------------------------------

def build_37_tiling(max_ring=5):
    """
    Build the {3,7} tiling graph by breadth-first growth.
    Returns: adjacency dict, ring assignment, ordered boundary cycles,
             and the triangle (face) list.

    Construction: standard layer-by-layer growth for {3,7}.
    Each new ring vertex is type A (2 parents) or B (1 parent),
    following the substitution structure A→AB, B→ABB.
    """
    # We grow the tiling explicitly. Vertices are integers in creation order.
    # State per ring: the cyclic list of ring-r vertices.
    adj = defaultdict(set)

    def connect(u, v):
        adj[u].add(v); adj[v].add(u)

    ring_of = {0: 0}
    rings = {0: [0]}
    next_id = 1

    # Ring 1: 7 vertices in a cycle, each connected to center
    ring1 = list(range(next_id, next_id + 7))
    next_id += 7
    for i, v in enumerate(ring1):
        ring_of[v] = 1
        connect(v, 0)
        connect(v, ring1[(i + 1) % 7])
    rings[1] = ring1

    # Each ring-1 vertex has: 1 parent (center), 2 siblings, 4 children → type B
    # Grow ring by ring.
    # For each ring-r vertex (in cyclic order), it contributes children:
    #   - It SHARES one child with each cyclic neighbor (the type-A children).
    #   - Type A vertex (2 parents): 3 children = shared_left, own(1), shared_right
    #   - Type B vertex (1 parent): 4 children = shared_left, own(2), shared_right
    # We build ring r+1 by walking ring r and emitting [shared, own...] blocks.

    vertex_type = {v: 'B' for v in ring1}  # ring-1 all type B

    for r in range(1, max_ring):
        cur = rings[r]
        n = len(cur)
        new_ring = []
        shared_child = {}  # (i) -> child shared between cur[i] and cur[i+1]

        # First create all shared children (one per ring-r edge)
        for i in range(n):
            c = next_id; next_id += 1
            ring_of[c] = r + 1
            vertex_type[c] = 'A'
            shared_child[i] = c
            connect(c, cur[i])
            connect(c, cur[(i + 1) % n])

        # Then create own children and assemble the cyclic order of ring r+1
        for i in range(n):
            v = cur[i]
            n_own = 1 if vertex_type[v] == 'A' else 2
            own = []
            for _ in range(n_own):
                c = next_id; next_id += 1
                ring_of[c] = r + 1
                vertex_type[c] = 'B'
                connect(c, v)
                own.append(c)
            # cyclic order around ring r+1: shared(i-1,i), own children of v, ...
            new_ring.append(shared_child[(i - 1) % n])
            new_ring.extend(own)

        # Connect consecutive ring-(r+1) vertices (sibling edges)
        m = len(new_ring)
        for i in range(m):
            connect(new_ring[i], new_ring[(i + 1) % m])

        rings[r + 1] = new_ring

    # Build triangle list: every 3-clique
    # Efficient: for each edge (u,v), common neighbors w>v>u
    faces = set()
    for u in adj:
        for v in adj[u]:
            if v < u: continue
            common = adj[u] & adj[v]
            for w in common:
                if w > v:
                    faces.add((u, v, w))
    faces = sorted(faces)

    return adj, ring_of, rings, faces, vertex_type


# ----------------------------------------------------------------------------
# 2. Load trial data — ADJUST THIS to match the library format
# ----------------------------------------------------------------------------

def load_ply(path):
    """
    Minimal PLY reader (ascii or binary_little_endian/big_endian).
    Returns an (N, 3) array of vertex x,y,z, ignoring other per-vertex
    properties (e.g. color).
    """
    type_map = {
        'float': 'f4', 'float32': 'f4', 'double': 'f8', 'float64': 'f8',
        'char': 'i1', 'int8': 'i1', 'uchar': 'u1', 'uint8': 'u1',
        'short': 'i2', 'int16': 'i2', 'ushort': 'u2', 'uint16': 'u2',
        'int': 'i4', 'int32': 'i4', 'uint': 'u4', 'uint32': 'u4',
    }
    with open(path, 'rb') as f:
        fmt = None
        n_verts = 0
        props = []
        in_vertex_element = False
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: missing end_header")
            tokens = line.decode('ascii').split()
            if not tokens:
                continue
            if tokens[0] == 'format':
                fmt = tokens[1]
            elif tokens[0] == 'element':
                in_vertex_element = (tokens[1] == 'vertex')
                if in_vertex_element:
                    n_verts = int(tokens[2])
            elif tokens[0] == 'property' and in_vertex_element:
                props.append((tokens[2], tokens[1]))  # (name, ply_type)
            elif tokens[0] == 'end_header':
                break

        xyz_idx = [i for i, (name, _) in enumerate(props) if name in ('x', 'y', 'z')]
        if len(xyz_idx) != 3:
            raise ValueError(f"{path}: could not find x/y/z vertex properties")

        if fmt == 'ascii':
            pos = np.zeros((n_verts, 3))
            for i in range(n_verts):
                vals = f.readline().decode('ascii').split()
                pos[i] = [float(vals[j]) for j in xyz_idx]
            return pos
        elif fmt in ('binary_little_endian', 'binary_big_endian'):
            endian = '<' if fmt == 'binary_little_endian' else '>'
            dtype = np.dtype([(name, endian + type_map[t]) for name, t in props])
            data = np.frombuffer(f.read(n_verts * dtype.itemsize), dtype=dtype, count=n_verts)
            xyz_names = [props[i][0] for i in xyz_idx]
            return np.stack([data[name].astype(float) for name in xyz_names], axis=1)
        raise ValueError(f"{path}: unsupported PLY format {fmt!r}")


def load_trial(path):
    """
    Return an (N_verts, 3) numpy array of vertex positions.
    Handles .ply (ascii or binary), .npy, .json, .txt/.csv.

    Vertex order: creation order (ring 0, ring 1, ..., ring 5) matching
    build_37_tiling. Verified that physics_grow.py's build_combinatorial
    produces an identical adjacency/ring/face structure and writes PLY
    vertices in that same creation order, so trial vertex i ==
    build_37_tiling vertex i with no permutation needed. main() also runs
    a per-library edge-length sanity check against `adj` at runtime.
    """
    if path.endswith('.ply'):
        return load_ply(path)
    if path.endswith('.npy'):
        return np.load(path)
    elif path.endswith('.json'):
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, dict):
            for key in ('positions', 'verts', 'vertices', 'X'):
                if key in data:
                    return np.asarray(data[key], dtype=float)
            raise ValueError(f"No position key found in {path}; keys: {list(data.keys())}")
        return np.asarray(data, dtype=float)
    elif path.endswith('.txt') or path.endswith('.csv'):
        return np.loadtxt(path, delimiter=None)
    raise ValueError(f"Unknown format: {path}")


# ----------------------------------------------------------------------------
# 3. Geometry: dihedral angles, turning angles, absorption
# ----------------------------------------------------------------------------

def triangle_normal(P, f):
    a, b, c = P[f[0]], P[f[1]], P[f[2]]
    n = np.cross(b - a, c - a)
    norm = np.linalg.norm(n)
    return n / norm if norm > 1e-12 else n

def dihedral_angles(P, faces, adj):
    """
    For every interior edge (shared by exactly 2 faces), compute the dihedral
    angle in [0, π]: π = flat, deviations = folding.
    Returns dict: (u,v) -> dihedral angle.
    """
    edge_faces = defaultdict(list)
    for fi, f in enumerate(faces):
        for e in ((f[0], f[1]), (f[0], f[2]), (f[1], f[2])):
            edge_faces[e].append(fi)

    dihedrals = {}
    for e, fl in edge_faces.items():
        if len(fl) != 2:
            continue  # boundary edge
        f1, f2 = faces[fl[0]], faces[fl[1]]
        n1, n2 = triangle_normal(P, f1), triangle_normal(P, f2)
        u, v = e
        # opposite vertices
        w1 = [x for x in f1 if x not in e][0]
        w2 = [x for x in f2 if x not in e][0]
        # signed convention: angle between half-planes
        edge_vec = P[v] - P[u]
        edge_vec /= max(np.linalg.norm(edge_vec), 1e-12)
        d1 = P[w1] - P[u]; d1 -= edge_vec * (d1 @ edge_vec); d1 /= max(np.linalg.norm(d1), 1e-12)
        d2 = P[w2] - P[u]; d2 -= edge_vec * (d2 @ edge_vec); d2 /= max(np.linalg.norm(d2), 1e-12)
        cosang = np.clip(d1 @ d2, -1, 1)
        dihedrals[e] = math.acos(cosang)  # π = flat (opposite verts far), 0 = folded shut
    return dihedrals

def boundary_turning(P, cycle):
    """Extrinsic turning angle at each vertex of the boundary polygon."""
    m = len(cycle)
    out = np.zeros(m)
    for i in range(m):
        prev_v, v, next_v = cycle[(i - 1) % m], cycle[i], cycle[(i + 1) % m]
        e1 = P[v] - P[prev_v]; e1 /= max(np.linalg.norm(e1), 1e-12)
        e2 = P[next_v] - P[v]; e2 /= max(np.linalg.norm(e2), 1e-12)
        cosang = np.clip(e1 @ e2, -1, 1)
        out[i] = math.acos(cosang)  # 0 = straight, π = reversal
    return out

def ring_band_edges(ring_of, dihedrals, r):
    """Interior edges of the band between ring r-1 and ring r:
    edges where min ring = r-1 or both endpoints in ring r (sibling edges
    interior to P_r when ring r+1 exists are boundary at the P_r level —
    here we just classify by ring labels)."""
    band = {}
    for (u, v), ang in dihedrals.items():
        ru, rv = ring_of[u], ring_of[v]
        lo, hi = min(ru, rv), max(ru, rv)
        if hi == r and lo in (r - 1, r):
            band[(u, v)] = ang
    return band


# ----------------------------------------------------------------------------
# 4. Main analysis
# ----------------------------------------------------------------------------

def main(library_dir):
    print(f"Building {{3,7}} tiling to ring 5...")
    adj, ring_of, rings, faces, vtype = build_37_tiling(5)
    n_verts = len(ring_of)
    print(f"  {n_verts} vertices, {len(faces)} faces")
    for r in range(6):
        print(f"  ring {r}: {len(rings[r])} vertices")
    expected = [1, 7, 21, 56, 147, 385]
    actual = [len(rings[r]) for r in range(6)]
    if actual != expected:
        print(f"  WARNING: ring sizes {actual} != expected {expected}")
        print(f"  The growth rule in build_37_tiling may need adjustment.")

    # Find trial files
    patterns = ['trial*.ply', '*.ply', 'trial_*.npy', 'trial_*.json', '*.npy', '*.json']
    files = []
    for p in patterns:
        files = sorted(glob.glob(os.path.join(library_dir, p)))
        if files:
            break
    if not files:
        print(f"No trial files found in {library_dir}")
        print("Expected trial*.ply (or .npy/.json) with (617,3) positions.")
        sys.exit(1)
    print(f"\nFound {len(files)} trial files. Analyzing...")

    edge_list = [(u, v) for u in adj for v in adj[u] if v > u]

    # Per-ring accumulators across trials
    fold_used = defaultdict(list)     # ring -> [mean |π - dihedral| per trial]
    fold_max = defaultdict(list)      # ring -> [max fold per trial]
    turn_excess = defaultdict(list)   # ring -> [mean (κ_ext - κ_int) per trial]
    absorption = defaultdict(list)    # ring -> [length absorption fraction per trial]

    n_loaded = 0
    for path in files:
        try:
            P = load_trial(path)
        except Exception as e:
            print(f"  skip {os.path.basename(path)}: {e}")
            continue
        if P.shape[0] < n_verts:
            continue  # trial didn't reach ring 5
        P = P[:n_verts]

        if n_loaded == 0:
            lens = np.array([np.linalg.norm(P[u] - P[v]) for u, v in edge_list])
            print(f"  vertex-order check ({os.path.basename(path)}): "
                  f"edge length mean={lens.mean():.4f} std={lens.std():.4f} "
                  f"(expect mean~1.0, std~0.05)")
            if abs(lens.mean() - 1.0) > 0.3 or lens.std() > 0.3:
                print("  WARNING: edge lengths far from 1.0 -- vertex ordering may "
                      "not match build_37_tiling; results below would be unreliable.")

        n_loaded += 1

        dih = dihedral_angles(P, faces, adj)

        for r in range(1, 6):
            band = ring_band_edges(ring_of, dih, r)
            if band:
                folds = np.array([abs(math.pi - a) for a in band.values()])
                fold_used[r].append(folds.mean())
                fold_max[r].append(folds.max())

            # boundary turning for the ring-r cycle
            cyc = rings[r]
            kext = boundary_turning(P, cyc)
            kint = np.array([0.0 if vtype[v] == 'A' else math.pi / 3 for v in cyc])
            turn_excess[r].append(float((kext - kint).mean()))

            # length absorption: chord span of ring r vs ring r-1
            # "absorbed" length = how much shorter the effective circumference is
            # than the unit-edge count. Effective circumference: perimeter of the
            # projection onto the best-fit plane? Simpler robust proxy:
            # radius of gyration ratio.
            cyc_prev = rings[r - 1]
            R_r = np.linalg.norm(P[cyc] - P[cyc].mean(0), axis=1).mean()
            R_prev = np.linalg.norm(P[cyc_prev] - P[cyc_prev].mean(0), axis=1).mean() if r > 1 else 0.0
            # planar circumference the ring "presents": 2πR_r; absorbed fraction:
            n_r = len(cyc)
            absorbed = 1.0 - (2 * math.pi * R_r) / n_r
            absorption[r].append(absorbed)

    print(f"Loaded {n_loaded} complete ring-5 trials.\n")

    # Report
    lines = []
    lines.append("PLEAT CAPACITY ANALYSIS")
    lines.append(f"critical absorption threshold 1/φ = {INV_PHI:.6f}")
    lines.append(f"trials analyzed: {n_loaded}")
    lines.append("")
    lines.append(f"{'ring':>5} | {'mean fold (rad)':>16} | {'max fold':>9} | "
                 f"{'turn excess':>12} | {'absorption':>11} | {'vs 1/φ':>8}")
    lines.append("-" * 75)
    for r in range(1, 6):
        mf = np.mean(fold_used[r]) if fold_used[r] else float('nan')
        xf = np.mean(fold_max[r]) if fold_max[r] else float('nan')
        te = np.mean(turn_excess[r]) if turn_excess[r] else float('nan')
        ab = np.mean(absorption[r]) if absorption[r] else float('nan')
        flag = "ABOVE" if ab >= INV_PHI else "below"
        lines.append(f"{r:>5} | {mf:>16.4f} | {xf:>9.4f} | {te:>12.4f} | "
                     f"{ab:>11.4f} | {flag:>8}")
    lines.append("")
    lines.append("Interpretation:")
    lines.append("- 'mean fold': average |π - dihedral| in the band ending at ring r.")
    lines.append("  Growth in r = accumulating pleat usage.")
    lines.append("- 'turn excess': mean (κ_ext - κ_int) on the ring-r boundary.")
    lines.append("  Positive = the embedding turns MORE than the intrinsic floor")
    lines.append("  (paying extra extrinsic curvature = normal curvature = pleating).")
    lines.append("- 'absorption': fraction of unit-edge length the ring 'eats'")
    lines.append("  relative to its presented circumference 2πR.")
    lines.append("  Theory: must be ≥ 1/φ = 0.618 for the NEXT ring to fit;")
    lines.append("  capacity c(r) declining through 1/φ between r=5 and r=6 is the")
    lines.append("  buckling hypothesis.")

    report = "\n".join(lines)
    print(report)
    with open('pleat_report.txt', 'w') as f:
        f.write(report + "\n")
    np.savez('pleat_data.npz',
             **{f'fold_used_r{r}': np.array(fold_used[r]) for r in range(1, 6)},
             **{f'fold_max_r{r}': np.array(fold_max[r]) for r in range(1, 6)},
             **{f'turn_excess_r{r}': np.array(turn_excess[r]) for r in range(1, 6)},
             **{f'absorption_r{r}': np.array(absorption[r]) for r in range(1, 6)})
    print("\nWrote pleat_report.txt and pleat_data.npz")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
