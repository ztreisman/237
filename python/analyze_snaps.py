"""
analyze_snaps.py — Run snap detection on an existing PLY mesh.

Loads a physics_grow PLY, reconstructs adjacency from faces and ring
assignment from vertex colors, then runs snap_cascade with a sweep of
epsilon values to map out snap sensitivity.

Usage:
    python analyze_snaps.py <file.ply> [--eps 0.05 0.10 0.15 0.20 0.25]
"""

import sys, json, math, argparse
import numpy as np
from collections import defaultdict
from scipy.spatial import cKDTree

# ── Ring color map (same as physics_grow) ─────────────────────────────────────
RING_COLORS = [
    (255, 255, 255),
    ( 68,   1,  84),
    ( 71,  44, 122),
    ( 59,  81, 139),
    ( 44, 113, 142),
    ( 33, 144, 141),
    ( 39, 173, 129),
    ( 92, 200, 100),
    (170, 220,  50),
    (234, 229,  26),
    (253, 231,  37),
]

def closest_ring(rgb):
    return min(range(len(RING_COLORS)),
               key=lambda r: sum((a-b)**2 for a,b in zip(rgb, RING_COLORS[r])))

def load_ply(path):
    with open(path) as f:
        lines = f.readlines()
    i = 0; nv = nf = 0
    while lines[i].strip() != 'end_header':
        if lines[i].startswith('element vertex'): nv = int(lines[i].split()[-1])
        if lines[i].startswith('element face'):   nf = int(lines[i].split()[-1])
        i += 1
    i += 1
    pos = []; rings = []
    for j in range(nv):
        p = lines[i+j].split()
        pos.append([float(p[0]), float(p[1]), float(p[2])])
        rings.append(closest_ring((int(p[3]), int(p[4]), int(p[5]))))
    i += nv
    faces = []
    for j in range(nf):
        p = lines[i+j].split()
        faces.append((int(p[1]), int(p[2]), int(p[3])))
    pos = np.array(pos)
    return pos, rings, faces

def build_adj(faces, n_verts):
    adj = defaultdict(set)
    for a, b, c in faces:
        adj[a].add(b); adj[b].add(a)
        adj[a].add(c); adj[c].add(a)
        adj[b].add(c); adj[c].add(b)
    return adj

# ── Snap functions (from physics_grow_snap.py) ────────────────────────────────

def edge_directions(v, pos, adj):
    nbrs = list(adj[v])
    if not nbrs: return np.zeros((0, 3))
    vecs = pos[np.array(nbrs)] - pos[v]
    norms = np.linalg.norm(vecs, axis=1, keepdims=True).clip(1e-12)
    return vecs / norms

def edges_align(v_a, v_b, pos, adj, angle_tol):
    if v_b in adj[v_a]: return 0
    dirs_a = edge_directions(v_a, pos, adj)
    dirs_b = edge_directions(v_b, pos, adj)
    if len(dirs_a) == 0 or len(dirs_b) == 0: return 0
    ab = pos[v_b] - pos[v_a]
    ab_norm = np.linalg.norm(ab)
    if ab_norm < 1e-12: return 0
    ab_unit = ab / ab_norm
    dots_a = dirs_a @ ab_unit
    dots_b = dirs_b @ (-ab_unit)
    dirs_a_filt = dirs_a[dots_a < 0.7]
    dirs_b_filt = dirs_b[dots_b < 0.7]
    if len(dirs_a_filt) == 0 or len(dirs_b_filt) == 0: return 0
    cos_mat = dirs_a_filt @ dirs_b_filt.T
    return int((cos_mat > np.cos(angle_tol)).sum())

def find_snaps(pos, adj, vertex_ring, epsilon, min_edges, angle_tol):
    N = len(pos)
    tree = cKDTree(pos)
    pairs = tree.query_pairs(r=epsilon, output_type='ndarray')
    snaps = []
    for pa, pb in pairs:
        if pb in adj[pa]: continue
        combined_deg = len(adj[pa]) + len(adj[pb])
        if combined_deg > 7: continue
        n_aligned = edges_align(pa, pb, pos, adj, angle_tol)
        if n_aligned >= min_edges:
            dist = float(np.linalg.norm(pos[pb] - pos[pa]))
            snaps.append((dist, int(pa), int(pb), n_aligned))
    return sorted(snaps)

def summarize_snaps(snaps, vertex_ring):
    if not snaps:
        print("  (none)")
        return
    for dist, va, vb, na in snaps:
        ra = vertex_ring[va]; rb = vertex_ring[vb]
        print(f"  v{va}(r{ra}) ↔ v{vb}(r{rb})  dist={dist:.4f}  "
              f"aligned_edges={na}  combined_deg={len_adj[va]+len_adj[vb]}")


def analyze(path, epsilons, min_edges=2, angle_tol=0.3):
    print(f"\nLoading {path}")
    pos, ring_list, faces = load_ply(path)
    adj = build_adj(faces, len(pos))
    vertex_ring = {i: ring_list[i] for i in range(len(pos))}
    max_ring = max(ring_list)

    # Degree histogram
    degrees = {r: [] for r in range(max_ring+1)}
    for v in range(len(pos)):
        degrees[vertex_ring[v]].append(len(adj[v]))

    print(f"  Vertices: {len(pos)}   Faces: {len(faces)}   Max ring: {max_ring}")
    for r in range(max_ring+1):
        degs = degrees[r]
        if degs:
            print(f"  Ring {r}: {len(degs)} verts  degree range [{min(degs)}, {max(degs)}]")

    # Edge length stats
    edge_lens = []
    seen = set()
    for v in range(len(pos)):
        for u in adj[v]:
            if u > v:
                edge_lens.append(np.linalg.norm(pos[v] - pos[u]))
                seen.add((v,u))
    el = np.array(edge_lens)
    print(f"  Edge lengths: mean={el.mean():.4f}  std={el.std():.4f}  "
          f"min={el.min():.4f}  max={el.max():.4f}")

    # Snap sweep
    print(f"\n  Snap sweep (min_edges={min_edges}, angle_tol={angle_tol:.2f} rad):")
    print(f"  {'epsilon':>8}  {'#pairs':>7}  {'snaps':>6}  snap detail")
    print(f"  {'-'*60}")

    all_results = {}
    for eps in epsilons:
        tree = cKDTree(pos)
        pairs = tree.query_pairs(r=eps, output_type='ndarray')
        n_pairs = len(pairs)
        snaps = find_snaps(pos, adj, vertex_ring, eps, min_edges, angle_tol)
        n_snaps = len(snaps)
        detail = ""
        if snaps:
            ring_pairs = sorted(set((vertex_ring[va], vertex_ring[vb])
                                    for _, va, vb, _ in snaps))
            detail = "  rings: " + ", ".join(f"({a},{b})" for a,b in ring_pairs)
        print(f"  {eps:8.3f}  {n_pairs:7d}  {n_snaps:6d}{detail}")
        all_results[eps] = snaps

    # Detailed print for largest epsilon with snaps
    for eps in reversed(epsilons):
        snaps = all_results[eps]
        if snaps:
            print(f"\n  Detailed snaps at epsilon={eps:.3f}:")
            for dist, va, vb, na in snaps[:20]:
                ra = vertex_ring[va]; rb = vertex_ring[vb]
                da = len(adj[va]); db = len(adj[vb])
                print(f"    v{va}(r{ra},deg{da}) ↔ v{vb}(r{rb},deg{db})  "
                      f"dist={dist:.4f}  aligned={na}")
            break

    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ply', nargs='?',
                        default='../physics_grow/physics_grow_seed03_ring05_std0.0415.ply')
    parser.add_argument('--eps', type=float, nargs='+',
                        default=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50])
    parser.add_argument('--min-edges', type=int, default=2)
    parser.add_argument('--angle-tol', type=float, default=0.3)
    args = parser.parse_args()

    len_adj = None  # populated in analyze

    pos, ring_list, faces = load_ply(args.ply)
    adj = build_adj(faces, len(pos))
    len_adj = {v: len(adj[v]) for v in range(len(pos))}

    analyze(args.ply, args.eps, args.min_edges, args.angle_tol)
