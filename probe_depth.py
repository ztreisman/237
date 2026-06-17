#!/usr/bin/env python3
"""
probe_depth.py — How many rings of {3,q} embed in R^d?

Grows the tiling ring by ring with the spring relaxer, retrying each ring
with perturbed restarts, and stops at the first ring that fails to certify.
Reports per-ring diagnostics that feed the capacity-lemma theory:

  - edge_std / max |edge err|  (certification)
  - band fold stats: mean, p90, max |pi - dihedral|, and HEADROOM
    h(r) = pi - fold_max   [dihedral between two faces sharing an edge is
    well-defined in any R^d: angle between opposite vertices projected
    perpendicular to the shared edge]
  - p90 fold vs the TETRAHEDRAL WALL pi - arccos(1/3) = 1.9106
    (off-sample test: does the wall govern R^4 as it does R^3?)
  - ring radius profile mean|x| (non-collapse / H1 check)
  - min non-adjacent vertex distance (collision proxy)

USAGE (lambda01):
  # the R^4 depth question (expect ring 9 ok; 10-11 are the frontier):
  nohup python3 probe_depth.py --q 7 --dim 4 --max-ring 11 \
      --max-verts 210000 --retries 3 --out probe_q7_d4 > probe_q7_d4.log 2>&1 &

  # quick check / other tilings:
  python3 probe_depth.py --q 7 --dim 4 --max-ring 6 --retries 1
  python3 probe_depth.py --q 8 --dim 4 --max-ring 6

Vertex counts {3,7}: r9=29k, r10=77k, r11=201k (L-BFGS on 4V params:
minutes / tens of minutes / hours on CPU). Requires
generate_unit_distance_library.py in the same directory.

Output: <out>.jsonl (one line per ring) and <out>_deepest.npz
(positions/edges of the deepest certified embedding).
"""

import argparse, json, math, os, sys, time
from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_unit_distance_library import (build_3q_disk, edges_from_adj,
                                            relax, edge_stats)

WALL = math.pi - math.acos(1/3)   # 1.9106 — tetrahedral fold wall


def faces_of(adj, nmax):
    """All triangles (3-cliques) among vertices < nmax."""
    out = []
    for u in range(nmax):
        for v in adj[u]:
            if v <= u or v >= nmax:
                continue
            for w in adj[u] & adj[v]:
                if w > v and w < nmax:
                    out.append((u, v, w))
    return out


def band_folds(P, adj, ring_of, faces, r):
    """|pi - dihedral| for interior edges of band r (max endpoint ring == r).
    Works in any dimension."""
    edge_faces = defaultdict(list)
    for f in faces:
        for e in ((f[0], f[1]), (f[0], f[2]), (f[1], f[2])):
            edge_faces[e].append(f)
    folds = []
    for (u, v), fl in edge_faces.items():
        if len(fl) != 2:
            continue
        if max(ring_of[u], ring_of[v]) != r:
            continue
        w1 = [x for x in fl[0] if x != u and x != v][0]
        w2 = [x for x in fl[1] if x != u and x != v][0]
        evec = P[v] - P[u]
        evec = evec / max(np.linalg.norm(evec), 1e-12)
        d1 = P[w1] - P[u]; d1 = d1 - evec * (d1 @ evec)
        d2 = P[w2] - P[u]; d2 = d2 - evec * (d2 @ evec)
        n1, n2 = np.linalg.norm(d1), np.linalg.norm(d2)
        if n1 < 1e-9 or n2 < 1e-9:
            continue
        dih = math.acos(float(np.clip(d1 @ d2 / (n1 * n2), -1, 1)))
        folds.append(abs(math.pi - dih))
    return np.array(folds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--q', type=int, default=7)
    ap.add_argument('--dim', type=int, default=4)
    ap.add_argument('--max-ring', type=int, default=11)
    ap.add_argument('--max-verts', type=int, default=210000)
    ap.add_argument('--retries', type=int, default=3)
    ap.add_argument('--certify', type=float, default=0.02,
                    help='max |edge err| to accept a ring')
    ap.add_argument('--iters', type=int, default=400,
                    help='base L-BFGS iters per stage (scales with ring)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='')
    ap.add_argument('--repulse', action='store_true',
                    help='short-range repulsion between non-adjacent '
                         'vertices (collision proxy -> surface embedding '
                         'mode rather than graph realization mode)')
    ap.add_argument('--repulse-r', type=float, default=0.6,
                    help='repulsion radius (vertex pairs closer than this '
                         'are pushed apart)')
    ap.add_argument('--repulse-w', type=float, default=0.2,
                    help='repulsion weight relative to edge springs')
    ap.add_argument('--resume-from', default='',
                    help='path to a *_deepest.npz written by this script; '
                         'continues growth from the ring after its deepest')
    args = ap.parse_args()

    out = args.out or f"probe_q{args.q}_d{args.dim}"
    logf = out + '.jsonl'

    adj, ring_of, rings, ach = build_3q_disk(args.q, args.max_ring,
                                             args.max_verts)
    print(f"{{3,{args.q}}} built to ring {ach}; sizes "
          f"{[len(rings[k]) for k in range(ach+1)]}", flush=True)
    ring_arr = {v: ring_of[v] for v in ring_of}

    rng = np.random.default_rng(args.seed)
    dim = args.dim

    # incremental growth with per-ring certification
    Vtot = sum(len(rings[k]) for k in range(ach + 1))
    P = np.zeros((Vtot, dim))
    start_k = 2
    if args.resume_from:
        data = np.load(args.resume_from)
        if 'vids' not in data:
            sys.exit('resume file lacks vertex ids (written by an older '
                     'version) — cannot resume safely')
        vids = data['vids']
        if data['positions'].shape[1] != dim:
            sys.exit('resume file dimension mismatch')
        P[vids] = data['positions']
        placed = sorted(int(v) for v in vids)
        deepest = max(ring_of[v] for v in placed)
        start_k = deepest + 1
        print(f"resumed from {args.resume_from}: deepest certified ring "
              f"{deepest}, continuing at ring {start_k}", flush=True)
    else:
        q = len(rings[1])
        for i, v in enumerate(rings[1]):
            ang = 2 * math.pi * i / q
            P[v, 0] = math.cos(ang); P[v, 1] = math.sin(ang)
            P[v] += rng.normal(0, 0.05, dim)
        placed = [0] + rings[1]
        deepest = 1

    for k in range(start_k, ach + 1):
        t0 = time.time()
        placed_set = set(placed)
        for v in rings[k]:
            parents = [u for u in adj[v] if u in placed_set]
            m = P[parents].mean(axis=0)
            nm = np.linalg.norm(m)
            outward = m / nm if nm > 1e-6 else rng.normal(0, 1, dim)
            P[v] = m + 0.9 * outward + rng.normal(0, 0.08, dim)
            placed.append(v); placed_set.add(v)

        sub = np.array(placed)
        idx = {v: i for i, v in enumerate(sub)}
        E = np.array([(idx[u], idx[v]) for u in sub for v in adj[u]
                      if v in idx and idx[v] > idx[u]], dtype=np.int64)
        L = np.ones(len(E))

        ok = False
        eset_local = {(int(a), int(b)) for a, b in E}

        def build_rp(seed):
            tree0 = cKDTree(P[sub])
            cand = np.array(sorted(tree0.query_pairs(args.repulse_r)),
                            dtype=np.int64)
            if not len(cand):
                return None
            keep = [(a, b) for a, b in cand
                    if (a, b) not in eset_local and (b, a) not in eset_local]
            if not keep:
                return None
            if len(keep) > 500000:
                sel = np.random.default_rng(seed).choice(
                    len(keep), 500000, replace=False)
                keep = [keep[i] for i in sel]
            return np.array(keep, dtype=np.int64)

        for attempt in range(args.retries):
            total = args.iters + 80 * k + 400 * attempt
            if args.repulse:
                # rebuild the repulsion pair list during relaxation so
                # pairs collapsing mid-relax are caught (MD neighbor list)
                cycles = 5
                for c in range(cycles):
                    rp = build_rp(attempt * 10 + c)
                    P[sub] = relax(P[sub], E, L,
                                   iters=max(80, total // cycles),
                                   repulse_pairs=rp,
                                   repulse_r=args.repulse_r,
                                   repulse_w=args.repulse_w)
            else:
                P[sub] = relax(P[sub], E, L, iters=total)
            mean, std, mx = edge_stats(P[sub], E, L)
            if mx <= args.certify:
                ok = True
                break
            # perturb outer two rings and retry
            outer = [idx[v] for v in placed if ring_arr[v] >= k - 1]
            P[sub[outer]] += rng.normal(0, 0.05 * (attempt + 1),
                                        (len(outer), dim))
        # diagnostics
        faces = faces_of(adj, int(sub.max()) + 1)
        folds = band_folds(P, adj, ring_arr, faces, k)
        radius = float(np.linalg.norm(P[rings[k]], axis=1).mean())
        tree = cKDTree(P[sub])
        pairs = tree.query_pairs(0.8)
        eset = {(int(a), int(b)) for a, b in E}
        min_nonadj = min((float(np.linalg.norm(P[sub[a]] - P[sub[b]]))
                          for a, b in pairs
                          if (a, b) not in eset and (b, a) not in eset),
                         default=float('nan'))
        rec = {
            'ring': k, 'V': len(sub), 'E': len(E),
            'certified': bool(ok), 'edge_std': std, 'edge_max_err': mx,
            'fold_mean': float(folds.mean()) if len(folds) else None,
            'fold_p90': float(np.percentile(folds, 90)) if len(folds) else None,
            'fold_max': float(folds.max()) if len(folds) else None,
            'headroom': float(math.pi - folds.max()) if len(folds) else None,
            'wall_1.9106_p90_ratio':
                float(np.percentile(folds, 90) / WALL) if len(folds) else None,
            'ring_radius_mean': radius,
            'radius_per_ring': radius / k,
            'min_nonadj_dist': min_nonadj,
            'seconds': round(time.time() - t0, 1),
        }
        with open(logf, 'a') as f:
            f.write(json.dumps(rec) + '\n')
        print(f"ring {k}: certified={ok} std={std:.4f} max={mx:.4f} "
              f"fold_p90={rec['fold_p90'] and round(rec['fold_p90'],3)} "
              f"headroom={rec['headroom'] and round(rec['headroom'],3)} "
              f"R/k={radius/k:.3f} ({rec['seconds']}s)", flush=True)

        if ok:
            deepest = k
            np.savez_compressed(out + '_deepest.npz',
                                positions=P[sub], edges=E, vids=sub,
                                rings=np.array([ring_arr[v] for v in sub]))
        else:
            print(f"FAILED at ring {k} after {args.retries} retries. "
                  f"Deepest certified: ring {deepest}.", flush=True)
            break
    else:
        print(f"Reached build limit. Deepest certified: ring {deepest} "
              f"(raise --max-ring / --max-verts to push further).",
              flush=True)


if __name__ == '__main__':
    main()
