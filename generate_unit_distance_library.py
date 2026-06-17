#!/usr/bin/env python3
"""
generate_unit_distance_library.py — Direction 4 training-data factory.

Produces a library of SOLVED unit-distance graph instances:
  1. Build the combinatorial disk of a hyperbolic triangle tiling {3,q}
     (q >= 7), ring by ring, up to --rings or --max-verts.
  2. Embed in R^n (n in --dims) with a spring relaxer (unit target edges),
     grown ring by ring from the center.
  3. Scan the embedding for chord candidates: vertex pairs at Euclidean
     distance ~= L (L in --chord-lengths), graph distance >= --min-graph-dist.
  4. Sample chord subsets at several densities, add them as edges with
     prescribed target length L, RE-RELAX, and keep only instances whose
     final max edge error passes --certify-tol (the instance is then a
     certified realization of the augmented graph).
  5. Save each instance as NPZ (positions, edges, target lengths) plus a
     JSONL manifest line. Resumable: existing outputs are skipped.

USAGE (lambda01):
  nohup python3 generate_unit_distance_library.py \
      --out ~/237/udg_library \
      --q 7 8 9 --dims 3 4 --trials 40 \
      --chord-lengths 1.0 2.0 --densities 0.15 0.4 1.0 --subsets 3 \
      > udg_gen.log 2>&1 &

  Smoke test (laptop, ~1 min):
  python3 generate_unit_distance_library.py --selftest

NOTES
- Ring growth for {3,q}: type-A vertices (2 parents) have q-6 "own"
  children, type-B (1 parent) have q-5; one shared (type-A) child per
  ring edge. For q=7 this is the verified {3,7} builder (sizes
  1,7,21,56,147,385). Interior degree == q is checked at build time.
- The relaxer certifies EDGE LENGTHS only (the training target). It does
  not enforce face non-intersection; --repulse adds short-range repulsion
  between non-adjacent vertices to discourage degenerate collapse.
- Train/val splits should be made on base_id (one embedded surface spawns
  many chord-augmented instances; do not let the same surface cross the
  split). base_id is recorded in the manifest.
"""

import argparse, json, math, os, sys, time
from collections import defaultdict, deque

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# 1. Combinatorial {3,q} disk
# ---------------------------------------------------------------------------

def build_3q_disk(q, max_ring, max_verts=None):
    """Grow the {3,q} triangulated disk. Returns (adj, ring_of, rings,
    achieved_ring). Vertices are integers in creation order."""
    assert q >= 7, "hyperbolic triangle tilings need q >= 7"
    adj = defaultdict(set)

    def connect(u, v):
        adj[u].add(v); adj[v].add(u)

    ring_of = {0: 0}
    rings = {0: [0]}
    nid = 1
    ring1 = list(range(nid, nid + q)); nid += q
    for i, v in enumerate(ring1):
        ring_of[v] = 1
        connect(v, 0)
        connect(v, ring1[(i + 1) % q])
    rings[1] = ring1
    vtype = {v: 'B' for v in ring1}

    own = {'A': q - 6, 'B': q - 5}
    r = 1
    while r < max_ring:
        cur = rings[r]
        n = len(cur)
        next_size = n + sum(own[vtype[v]] for v in cur)
        if max_verts and nid + next_size > max_verts:
            break
        shared = {}
        for i in range(n):
            c = nid; nid += 1
            ring_of[c] = r + 1; vtype[c] = 'A'
            shared[i] = c
            connect(c, cur[i]); connect(c, cur[(i + 1) % n])
        new_ring = []
        for i in range(n):
            v = cur[i]
            new_ring.append(shared[(i - 1) % n])
            for _ in range(own[vtype[v]]):
                c = nid; nid += 1
                ring_of[c] = r + 1; vtype[c] = 'B'
                connect(c, v)
                new_ring.append(c)
        m = len(new_ring)
        for i in range(m):
            connect(new_ring[i], new_ring[(i + 1) % m])
        rings[r + 1] = new_ring
        r += 1

    # interior degree check
    for k in range(r):
        for v in rings[k]:
            if len(adj[v]) != q:
                raise RuntimeError(f"degree check failed at ring {k}: "
                                   f"deg={len(adj[v])} != q={q}")
    return adj, ring_of, rings, r


def edges_from_adj(adj):
    return np.array(sorted((u, v) for u in adj for v in adj[u] if v > u),
                    dtype=np.int64)


# ---------------------------------------------------------------------------
# 2. Relaxer
# ---------------------------------------------------------------------------

def relax(P, edges, lengths, iters=400, repulse_pairs=None, repulse_r=0.6,
          repulse_w=0.2):
    """L-BFGS on spring energy sum (|xi-xj| - L)^2 (+ optional repulsion).
    P: (V,d) initial positions. Returns relaxed P."""
    V, d = P.shape
    ea, eb = edges[:, 0], edges[:, 1]
    L = lengths

    if repulse_pairs is not None and len(repulse_pairs):
        ra = repulse_pairs[:, 0]; rb = repulse_pairs[:, 1]
    else:
        ra = rb = None

    def f(x):
        Q = x.reshape(V, d)
        dv = Q[ea] - Q[eb]
        dist = np.sqrt(np.sum(dv * dv, axis=1)) + 1e-12
        diff = dist - L
        E = float(np.sum(diff * diff))
        g = np.zeros_like(Q)
        coef = (2.0 * diff / dist)[:, None] * dv
        np.add.at(g, ea, coef)
        np.add.at(g, eb, -coef)
        if ra is not None:
            rv = Q[ra] - Q[rb]
            rd = np.sqrt(np.sum(rv * rv, axis=1)) + 1e-12
            mask = rd < repulse_r
            if mask.any():
                pen = repulse_r - rd[mask]
                E += repulse_w * float(np.sum(pen * pen))
                rc = (-2.0 * repulse_w * pen / rd[mask])[:, None] * rv[mask]
                np.add.at(g, ra[mask], rc)
                np.add.at(g, rb[mask], -rc)
        return E, g.ravel()

    res = minimize(f, P.ravel(), jac=True, method='L-BFGS-B',
                   options={'maxiter': iters, 'maxcor': 20})
    return res.x.reshape(V, d)


def edge_stats(P, edges, lengths):
    d = np.linalg.norm(P[edges[:, 0]] - P[edges[:, 1]], axis=1)
    err = d - lengths
    return float(d.mean()), float(err.std()), float(np.abs(err).max())


def grow_embedding(adj, rings, achieved, dim, seed, iters_per_stage=350,
                   repulse=False):
    """Stage-wise growth: place ring k from parents, relax all placed."""
    rng = np.random.default_rng(seed)
    Vtot = sum(len(rings[k]) for k in range(achieved + 1))
    P = np.zeros((Vtot, dim))
    placed = [0]
    # ring 1 on a unit circle in the first two coords (+ noise)
    q = len(rings[1])
    for i, v in enumerate(rings[1]):
        ang = 2 * math.pi * i / q
        P[v, 0] = math.cos(ang); P[v, 1] = math.sin(ang)
        P[v] += rng.normal(0, 0.05, dim)
    placed += rings[1]

    for k in range(2, achieved + 1):
        for v in rings[k]:
            parents = [u for u in adj[v] if u in set(placed)]
            if parents:
                m = P[parents].mean(axis=0)
                nm = np.linalg.norm(m)
                outward = m / nm if nm > 1e-6 else rng.normal(0, 1, dim)
                P[v] = m + 0.9 * outward + rng.normal(0, 0.08, dim)
            else:
                P[v] = rng.normal(0, k, dim)
            placed.append(v)
        sub = np.array(placed)
        idx = {v: i for i, v in enumerate(sub)}
        E = np.array([(idx[u], idx[v]) for u in sub for v in adj[u]
                      if v in idx and idx[v] > idx[u]], dtype=np.int64)
        L = np.ones(len(E))
        rp = None
        if repulse:
            tree = cKDTree(P[sub])
            cand = np.array(sorted(tree.query_pairs(0.6)), dtype=np.int64)
            if len(cand):
                aset = {(min(a, b), max(a, b)) for a, b in E}
                rp = np.array([(a, b) for a, b in cand
                               if (min(a, b), max(a, b)) not in aset],
                              dtype=np.int64)
                if len(rp) == 0:
                    rp = None
        P[sub] = relax(P[sub], E, L, iters=iters_per_stage + 40 * k,
                       repulse_pairs=rp)
    return P


# ---------------------------------------------------------------------------
# 3. Chord scanning and certification
# ---------------------------------------------------------------------------

def graph_dist_lt(adj, src, dst, bound):
    """True if graph distance(src,dst) < bound (BFS capped at bound-1)."""
    if src == dst:
        return True
    seen = {src}
    frontier = [src]
    for _ in range(bound - 1):
        nxt = []
        for u in frontier:
            for w in adj[u]:
                if w == dst:
                    return True
                if w not in seen:
                    seen.add(w); nxt.append(w)
        frontier = nxt
        if not frontier:
            break
    return False


def chord_candidates(P, adj, edges_set, target_L, tol, min_gd, cap, rng):
    tree = cKDTree(P)
    pairs = sorted(tree.query_pairs(target_L + tol))
    rng.shuffle(pairs)
    out = []
    for (i, j) in pairs:
        d = np.linalg.norm(P[i] - P[j])
        if d < target_L - tol:
            continue
        if (i, j) in edges_set or (j, i) in edges_set:
            continue
        if graph_dist_lt(adj, i, j, min_gd):
            continue
        out.append((i, j))
        if len(out) >= cap:
            break
    return out


# ---------------------------------------------------------------------------
# 4. Main pipeline
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='udg_library')
    ap.add_argument('--q', type=int, nargs='+', default=[7])
    ap.add_argument('--dims', type=int, nargs='+', default=[3, 4])
    ap.add_argument('--rings', type=int, default=0,
                    help='0 = auto (limited by --max-verts and dim defaults)')
    ap.add_argument('--max-verts', type=int, default=8000)
    ap.add_argument('--trials', type=int, default=20)
    ap.add_argument('--edge-std-max', type=float, default=0.06,
                    help='base-embedding acceptance threshold')
    ap.add_argument('--chord-lengths', type=float, nargs='+',
                    default=[1.0, 2.0])
    ap.add_argument('--tol', type=float, default=0.03,
                    help='chord candidate window around target length')
    ap.add_argument('--min-graph-dist', type=int, default=4)
    ap.add_argument('--densities', type=float, nargs='+',
                    default=[0.15, 0.4, 1.0])
    ap.add_argument('--subsets', type=int, default=2,
                    help='random subsets per density')
    ap.add_argument('--chord-cap', type=int, default=4000)
    ap.add_argument('--certify-tol', type=float, default=0.02,
                    help='max |edge error| for a kept instance')
    ap.add_argument('--repulse', action='store_true')
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    os.makedirs(args.out, exist_ok=True)
    manifest = os.path.join(args.out, 'manifest.jsonl')

    def auto_rings(q, dim):
        # stay below known frontiers; deeper in R^4
        base = {7: (5, 9), 8: (4, 7), 9: (3, 6), 10: (3, 5)}
        lo = base.get(q, (3, 5))
        return lo[0] if dim == 3 else lo[1]

    for q in args.q:
        for dim in args.dims:
            rings_target = args.rings or auto_rings(q, dim)
            for trial in range(args.trials):
                base_id = f"q{q}_n{dim}_r{rings_target}_t{trial:03d}"
                base_npz = os.path.join(args.out, base_id + '_base.npz')
                if os.path.exists(base_npz):
                    data = np.load(base_npz)
                    P, edges = data['positions'], data['edges']
                    adj, ring_of, rings, ach = build_3q_disk(
                        q, rings_target, args.max_verts)
                else:
                    adj, ring_of, rings, ach = build_3q_disk(
                        q, rings_target, args.max_verts)
                    t0 = time.time()
                    P = grow_embedding(adj, rings, ach, dim, seed=trial,
                                       repulse=args.repulse)
                    edges = edges_from_adj(adj)
                    L = np.ones(len(edges))
                    mean, std, mx = edge_stats(P, edges, L)
                    print(f"[{base_id}] rings={ach} V={len(P)} "
                          f"std={std:.4f} max={mx:.4f} "
                          f"({time.time()-t0:.0f}s)", flush=True)
                    if std > args.edge_std_max:
                        print(f"[{base_id}] rejected (std)", flush=True)
                        continue
                    np.savez_compressed(
                        base_npz, positions=P, edges=edges,
                        lengths=L, ring_of=np.array(
                            [ring_of[v] for v in range(len(P))]))
                    with open(manifest, 'a') as f:
                        f.write(json.dumps({
                            'file': os.path.basename(base_npz),
                            'base_id': base_id, 'kind': 'base', 'q': q,
                            'dim': dim, 'rings': ach, 'V': len(P),
                            'E': len(edges), 'edge_std': std,
                            'edge_max_err': mx}) + '\n')

                # chord augmentation
                edges_set = {(int(a), int(b)) for a, b in edges}
                rng = np.random.default_rng(1000 + trial)
                for target_L in args.chord_lengths:
                    cands = chord_candidates(
                        P, adj, edges_set, target_L, args.tol,
                        args.min_graph_dist, args.chord_cap, rng)
                    if not cands:
                        continue
                    for dens in args.densities:
                        nsub = args.subsets if dens < 1.0 else 1
                        for s in range(nsub):
                            tag = (f"{base_id}_L{target_L:g}"
                                   f"_d{int(100*dens):03d}_s{s}")
                            out_npz = os.path.join(args.out, tag + '.npz')
                            if os.path.exists(out_npz):
                                continue
                            k = max(1, int(dens * len(cands)))
                            sel = rng.choice(len(cands), size=k,
                                             replace=False)
                            chords = np.array([cands[i] for i in sel],
                                              dtype=np.int64)
                            E2 = np.vstack([edges, chords])
                            L2 = np.concatenate([
                                np.ones(len(edges)),
                                np.full(len(chords), target_L)])
                            P2 = relax(P.copy(), E2, L2, iters=600)
                            mean, std, mx = edge_stats(P2, E2, L2)
                            if mx > args.certify_tol:
                                print(f"[{tag}] uncertified "
                                      f"(max err {mx:.4f})", flush=True)
                                continue
                            np.savez_compressed(
                                out_npz, positions=P2, edges=E2, lengths=L2)
                            with open(manifest, 'a') as f:
                                f.write(json.dumps({
                                    'file': os.path.basename(out_npz),
                                    'base_id': base_id, 'kind': 'augmented',
                                    'q': q, 'dim': dim, 'V': len(P2),
                                    'E': len(E2), 'n_chords': len(chords),
                                    'chord_L': target_L, 'density': dens,
                                    'edge_std': std,
                                    'edge_max_err': mx}) + '\n')
                            print(f"[{tag}] saved: +{len(chords)} chords, "
                                  f"max err {mx:.4f}", flush=True)


# ---------------------------------------------------------------------------
# 5. Self test
# ---------------------------------------------------------------------------

def selftest():
    print("SELFTEST: {3,7} rings 0-4 in R^3, chords at L=1, re-relax")
    adj, ring_of, rings, ach = build_3q_disk(7, 4)
    sizes = [len(rings[k]) for k in range(ach + 1)]
    print(f"  ring sizes: {sizes} (expect [1,7,21,56,147])")
    assert sizes == [1, 7, 21, 56, 147]
    # also verify q=8 builder shape
    _, _, r8, a8 = build_3q_disk(8, 3)
    print(f"  {{3,8}} sizes: {[len(r8[k]) for k in range(a8+1)]} "
          f"(expect [1,8,32,120])")
    P = grow_embedding(adj, rings, ach, dim=3, seed=0)
    edges = edges_from_adj(adj)
    L = np.ones(len(edges))
    mean, std, mx = edge_stats(P, edges, L)
    print(f"  base embed: V={len(P)} E={len(edges)} mean={mean:.4f} "
          f"std={std:.4f} max|err|={mx:.4f}")
    edges_set = {(int(a), int(b)) for a, b in edges}
    rng = np.random.default_rng(0)
    cands = chord_candidates(P, adj, edges_set, 1.0, 0.05, 4, 500, rng)
    print(f"  chord candidates @ L=1±0.05, graph-dist>=4: {len(cands)}")
    if cands:
        chords = np.array(cands[:max(1, len(cands)//2)], dtype=np.int64)
        E2 = np.vstack([edges, chords])
        L2 = np.concatenate([L, np.ones(len(chords))])
        P2 = relax(P.copy(), E2, L2, iters=600)
        m2, s2, x2 = edge_stats(P2, E2, L2)
        print(f"  re-relaxed with {len(chords)} chords: std={s2:.4f} "
              f"max|err|={x2:.4f} "
              f"({'CERTIFIED' if x2 < 0.02 else 'not certified'})")
    print("SELFTEST done")


if __name__ == '__main__':
    main()
