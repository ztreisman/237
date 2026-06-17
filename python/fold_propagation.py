#!/usr/bin/env python3
"""
fold_propagation.py — Does a bad fold on ring 4 make a worse fold on the
adjacent part of ring 5?

Tests radial inheritance of fold stress between band 4 and band 5, using the
parent structure of the {3,7} tiling to define adjacency.

DEPENDS ON pleat_analysis.py in the same directory for:
  build_37_tiling, load_trial (with the PLY loader), dihedral_angles,
  triangle_normal
Run AFTER pleat_analysis.py works (same library dir, same vertex ordering —
already verified byte-identical to physics_grow.py's build_combinatorial).

USAGE:
  python3 fold_propagation.py /path/to/library_dir

MEASURES (per trial, pooled across trials):
 A. Spearman correlation: fold(e5) vs parent_fold(e5)
    where e5 = radial 4→5 edge, parent_fold = max fold over band-4 edges
    incident to e5's ring-4 endpoint.
 B. Conditional contrast: E[fold(e5) | parent decile 10] vs decile 1.
 C. Stress profile: band-5 fold vs cyclic distance (along ring 5) from the
    child position of the trial's worst band-4 edge.
 D. Signed folds: same-sign radial inheritance (crease propagation) vs
    alternation (pleating), using consistently oriented triangle normals.

OUTPUT: propagation_report.txt, propagation_data.npz
"""

import sys, os, glob, math
import numpy as np
from collections import defaultdict

# Import shared machinery
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pleat_analysis import build_37_tiling, load_trial, dihedral_angles

def spearman(x, y):
    """Spearman rank correlation without scipy."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    denom = math.sqrt((rx @ rx) * (ry @ ry))
    return float(rx @ ry / denom) if denom > 0 else 0.0


def signed_dihedrals(P, faces, face_orient):
    """
    Signed fold angle per interior edge: positive = fold toward the
    outward orientation (convex), negative = concave. Uses consistently
    oriented faces (face_orient: list of faces as oriented vertex triples).
    Returns dict edge -> signed fold (fold = pi - dihedral, sign from
    relative normal geometry).
    """
    edge_faces = defaultdict(list)
    for fi, f in enumerate(face_orient):
        for k in range(3):
            e = (f[k], f[(k+1) % 3])
            edge_faces[tuple(sorted(e))].append((fi, e))

    out = {}
    for e_sorted, lst in edge_faces.items():
        if len(lst) != 2:
            continue
        (f1i, e1), (f2i, e2) = lst
        f1, f2 = face_orient[f1i], face_orient[f2i]
        n1 = np.cross(P[f1[1]] - P[f1[0]], P[f1[2]] - P[f1[0]])
        n2 = np.cross(P[f2[1]] - P[f2[0]], P[f2[2]] - P[f2[0]])
        n1 /= max(np.linalg.norm(n1), 1e-12)
        n2 /= max(np.linalg.norm(n2), 1e-12)
        u, v = e_sorted
        w1 = [x for x in f1 if x != u and x != v][0]
        w2 = [x for x in f2 if x != u and x != v][0]
        evec = P[v] - P[u]; evec /= max(np.linalg.norm(evec), 1e-12)
        d1 = P[w1] - P[u]; d1 -= evec * (d1 @ evec); d1 /= max(np.linalg.norm(d1), 1e-12)
        d2 = P[w2] - P[u]; d2 -= evec * (d2 @ evec); d2 /= max(np.linalg.norm(d2), 1e-12)
        dihedral = math.acos(float(np.clip(d1 @ d2, -1, 1)))
        fold = math.pi - dihedral
        # Sign: does w2 lie on the +n1 side of f1's plane?
        side = float(n1 @ (P[w2] - P[u]))
        out[e_sorted] = fold * (1.0 if side >= 0 else -1.0)
    return out


def orient_faces(faces, adj, rings):
    """
    Produce a consistent orientation of all faces (so signed folds are
    comparable). Walk the dual graph from an arbitrary seed face, flipping
    orientations so shared edges are traversed oppositely (standard
    orientability propagation; the disk is orientable).
    """
    edge_to_faces = defaultdict(list)
    for fi, f in enumerate(faces):
        for e in ((f[0], f[1]), (f[0], f[2]), (f[1], f[2])):
            edge_to_faces[tuple(sorted(e))].append(fi)

    oriented = {}
    seed = 0
    oriented[seed] = tuple(faces[seed])
    stack = [seed]
    while stack:
        fi = stack.pop()
        f = oriented[fi]
        dir_edges = {(f[0], f[1]), (f[1], f[2]), (f[2], f[0])}
        for k in range(3):
            e = tuple(sorted((f[k], f[(k+1) % 3])))
            for gj in edge_to_faces[e]:
                if gj == fi or gj in oriented:
                    continue
                g = faces[gj]
                # orient g so that shared edge is traversed in the opposite direction
                a, b = f[k], f[(k+1) % 3]  # direction in f
                # candidate orientations of g
                cands = [(g[0], g[1], g[2]), (g[0], g[2], g[1])]
                chosen = None
                for cand in cands:
                    ed = {(cand[0], cand[1]), (cand[1], cand[2]), (cand[2], cand[0])}
                    if (b, a) in ed:  # opposite traversal
                        chosen = cand
                        break
                oriented[gj] = chosen if chosen else tuple(g)
                stack.append(gj)
    return [oriented[i] for i in range(len(faces))]


def main(library_dir):
    adj, ring_of, rings, faces, vtype = build_37_tiling(5)
    n_verts = len(ring_of)
    print(f"Tiling: {n_verts} verts, {len(faces)} faces")

    face_orient = orient_faces(faces, adj, rings)

    # Band-edge classification
    def band_of_edge(u, v):
        return max(ring_of[u], ring_of[v])

    # For each radial 4->5 edge, its ring-4 endpoint
    # For each ring-4 vertex p, its incident band-4 edges
    band4_edges_at = defaultdict(list)   # ring-4 vertex -> list of band-4 edges
    radial45 = []                        # list of (edge, ring4_endpoint, ring5_endpoint)
    for u in adj:
        for v in adj[u]:
            if v < u:
                continue
            b = band_of_edge(u, v)
            e = (u, v)
            if b == 4:
                if ring_of[u] == 4:
                    band4_edges_at[u].append(e)
                if ring_of[v] == 4:
                    band4_edges_at[v].append(e)
            elif b == 5:
                p = u if ring_of[u] == 4 else v
                c = v if ring_of[u] == 4 else u
                if ring_of[p] == 4 and ring_of[c] == 5:
                    radial45.append((e, p, c))

    # cyclic position of ring-5 vertices for the stress profile
    ring5_pos = {v: i for i, v in enumerate(rings[5])}
    N5 = len(rings[5])

    # Trial files
    patterns = ['trial*_ring5_*.ply', 'trial*.ply', '*.ply']
    files = []
    for p in patterns:
        files = sorted(glob.glob(os.path.join(library_dir, p)))
        if files:
            break
    if not files:
        print(f"No PLY files in {library_dir}")
        sys.exit(1)
    print(f"{len(files)} trial files")

    rho_list, contrast_list = [], []
    profile_accum = np.zeros(N5)
    profile_count = np.zeros(N5)
    sign_match_list = []   # fraction of radial pairs with same-sign folds
    all_e5_folds, all_parent_folds = [], []

    n_loaded = 0
    for path in files:
        try:
            P = load_trial(path)
        except Exception as ex:
            continue
        if P.shape[0] < n_verts:
            continue
        P = P[:n_verts]
        n_loaded += 1

        dih = dihedral_angles(P, faces, adj)
        sgn = signed_dihedrals(P, faces, face_orient)
        fold = {e: abs(math.pi - a) for e, a in dih.items()}

        # A/B: per radial45 edge, parent stress = max band-4 fold at p
        e5f, pf = [], []
        sign_pairs = []
        for e, p, c in radial45:
            if e not in fold:
                continue
            parent_edges = [pe for pe in band4_edges_at[p] if tuple(sorted(pe)) in fold or pe in fold]
            pf_vals = []
            for pe in band4_edges_at[p]:
                key = tuple(sorted(pe))
                if key in fold:
                    pf_vals.append(fold[key])
                elif pe in fold:
                    pf_vals.append(fold[pe])
            if not pf_vals:
                continue
            ekey = tuple(sorted(e))
            f5 = fold.get(ekey, fold.get(e))
            e5f.append(f5)
            pf.append(max(pf_vals))
            # signed comparison: sign of the child fold vs sign of worst parent fold
            s5 = sgn.get(ekey)
            worst_pe = max(band4_edges_at[p],
                           key=lambda pe: fold.get(tuple(sorted(pe)), 0.0))
            s4 = sgn.get(tuple(sorted(worst_pe)))
            if s5 is not None and s4 is not None and abs(s5) > 1e-6 and abs(s4) > 1e-6:
                sign_pairs.append(1.0 if (s5 > 0) == (s4 > 0) else 0.0)

        if len(e5f) < 10:
            continue
        rho_list.append(spearman(pf, e5f))
        all_e5_folds.extend(e5f)
        all_parent_folds.extend(pf)
        if sign_pairs:
            sign_match_list.append(float(np.mean(sign_pairs)))

        # B: decile contrast
        pf_arr, e5_arr = np.array(pf), np.array(e5f)
        lo_th, hi_th = np.quantile(pf_arr, 0.1), np.quantile(pf_arr, 0.9)
        lo_mean = e5_arr[pf_arr <= lo_th].mean()
        hi_mean = e5_arr[pf_arr >= hi_th].mean()
        contrast_list.append(hi_mean - lo_mean)

        # C: stress profile around the worst band-4 edge
        # find worst band-4 edge, take a ring-4 endpoint, find its children's
        # cyclic positions in ring 5, accumulate band-5 fold vs cyclic distance
        worst_e4, worst_val = None, -1
        for pvert, elist in band4_edges_at.items():
            for pe in elist:
                key = tuple(sorted(pe))
                fv = fold.get(key, 0.0)
                if fv > worst_val:
                    worst_val, worst_e4 = fv, key
        # anchor position: child of the band-4 edge's ring-4 endpoint(s)
        anchors = []
        for x in worst_e4:
            if ring_of[x] == 4:
                for nb in adj[x]:
                    if ring_of[nb] == 5:
                        anchors.append(ring5_pos[nb])
        if anchors:
            a0 = anchors[0]
            for e, p, c in radial45:
                ekey = tuple(sorted(e))
                if ekey not in fold:
                    continue
                d = abs(ring5_pos[c] - a0)
                d = min(d, N5 - d)
                profile_accum[d] += fold[ekey]
                profile_count[d] += 1

    print(f"Loaded {n_loaded} complete trials\n")

    rho = np.array(rho_list)
    contrast = np.array(contrast_list)
    pooled_rho = spearman(all_parent_folds, all_e5_folds)
    profile = np.where(profile_count > 0, profile_accum / np.maximum(profile_count, 1), np.nan)

    lines = []
    lines.append("FOLD PROPAGATION: ring 4 -> ring 5")
    lines.append(f"trials: {n_loaded}, radial 4->5 edges per trial: ~{len(radial45)}")
    lines.append("")
    lines.append("A. Spearman correlation fold(e5) vs parent_fold (max band-4 fold at parent):")
    lines.append(f"   per-trial mean rho = {rho.mean():.4f}  (sd {rho.std():.4f}, "
                 f"min {rho.min():.4f}, max {rho.max():.4f})")
    lines.append(f"   pooled rho = {pooled_rho:.4f}")
    lines.append("")
    lines.append("B. Decile contrast E[fold5 | parent top 10%] - E[fold5 | parent bottom 10%]:")
    lines.append(f"   mean = {contrast.mean():.4f} rad  (sd {contrast.std():.4f})")
    lines.append("")
    lines.append("D. Signed inheritance (fraction of same-sign parent/child folds):")
    if sign_match_list:
        sm = np.array(sign_match_list)
        lines.append(f"   mean = {sm.mean():.4f}  (0.5 = no relation, >0.5 = creases propagate")
        lines.append(f"    radially with the same sign, <0.5 = alternation/pleating)")
    lines.append("")
    lines.append("C. Stress profile vs cyclic distance from worst band-4 edge's child")
    lines.append("   (band-5 fold, averaged over trials; first 30 lags):")
    for d in range(0, 30):
        if not math.isnan(profile[d]):
            bar = '#' * int(profile[d] * 40)
            lines.append(f"   d={d:3d}: {profile[d]:.4f}  {bar}")
    lines.append("")
    lines.append("Interpretation guide:")
    lines.append(" - rho > 0 and contrast > 0: bad ring-4 folds DO make adjacent ring-5")
    lines.append("   folds worse (radial stress inheritance — supports compounding h(r)).")
    lines.append(" - profile peaked at d=0 and decaying: stress is LOCALIZED radially.")
    lines.append("   Decay length = how far a crease's influence spreads along the ring.")
    lines.append(" - sign fraction >> 0.5: a crease at ring 4 continues as a crease at")
    lines.append("   ring 5 (same bend direction) — a propagating fold line.")
    lines.append("   << 0.5: the surface accommodates by counter-folding (pleating).")

    report = "\n".join(lines)
    print(report)
    with open('propagation_report.txt', 'w') as f:
        f.write(report + "\n")
    np.savez('propagation_data.npz',
             rho=rho, contrast=contrast, profile=profile,
             profile_count=profile_count,
             pooled_parent=np.array(all_parent_folds),
             pooled_child=np.array(all_e5_folds),
             sign_match=np.array(sign_match_list))
    print("\nWrote propagation_report.txt and propagation_data.npz")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
