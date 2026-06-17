#!/usr/bin/env python3
"""
fold_ab_analysis.py — Two follow-ups to the propagation result:

1. COMPENSATION TEST (the propagation result reinterpreted):
   For each ring-4 vertex p (a radial "sector"), let
     D4(p) = mean fold over band-4 edges incident to p
     D5(p) = mean fold over radial 4->5 edges incident to p
   The propagation run showed corr(D4, D5) < 0. Here we test the stronger
   conservation claim: Var(D4 + D5) << Var(D4) + Var(D5)
   (i.e., the SUM is much more uniform than the parts — work is shared).
   Also reports corr, and the distribution of the per-sector total.

2. A/B WORD ANALYSIS:
   Fold statistics conditioned on the substitution word position.
   Per radial 4->5 edge, classify by:
     - child type: A (2 parents, intrinsic turn 0) vs B (1 parent, turn π/3)
     - B-run context: is a B child in a run of length 1 (ABA) or 2 (ABBA)?
       and for length-2 runs, first-B vs second-B position.
     - A context: A between two B's of the same run vs A flanked by
       different run patterns (the word has limited local types; we
       enumerate the 5-letter window around each position).
   Per ring-5 BOUNDARY vertex, the extrinsic turning vs type
   (already know mean excess; here we want the full distribution per class).

USAGE: python3 fold_ab_analysis.py /path/to/library_dir
OUTPUT: ab_report.txt, ab_data.npz
Depends on pleat_analysis.py (tiling + PLY loader + dihedrals).
"""

import sys, os, glob, math
import numpy as np
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pleat_analysis import build_37_tiling, load_trial, dihedral_angles


def main(library_dir):
    adj, ring_of, rings, faces, vtype = build_37_tiling(5)
    n_verts = len(ring_of)

    # ---- word structure of ring 5 ----
    cyc5 = rings[5]
    N5 = len(cyc5)
    word = ''.join(vtype[v] for v in cyc5)
    # rotate to start at an A for run parsing
    i0 = word.index('A')
    rot_idx = list(range(i0, N5)) + list(range(0, i0))
    word_rot = ''.join(word[i] for i in rot_idx)

    # classify every position: 'A1' (A before single B), 'A2' (A before BB),
    # 'B1' (lone B), 'B2a'/'B2b' (first/second of BB)
    cls = [''] * N5
    i = 0
    while i < N5:
        assert word_rot[i] == 'A'
        j = i + 1
        run = 0
        while j < N5 and word_rot[j] == 'B':
            run += 1; j += 1
        a_label = 'A1' if run == 1 else 'A2'
        cls[rot_idx[i]] = a_label
        if run == 1:
            cls[rot_idx[i+1]] = 'B1'
        elif run == 2:
            cls[rot_idx[i+1]] = 'B2a'
            cls[rot_idx[i+2]] = 'B2b'
        i = j
    class_counts = Counter(cls)

    # ---- radial 4->5 edges and sector structure ----
    def band_of_edge(u, v): return max(ring_of[u], ring_of[v])
    band4_at = defaultdict(list)
    radial45 = []
    for u in adj:
        for v in adj[u]:
            if v < u: continue
            b = band_of_edge(u, v)
            if b == 4:
                if ring_of[u] == 4: band4_at[u].append((u, v))
                if ring_of[v] == 4: band4_at[v].append((u, v))
            elif b == 5:
                p, c = (u, v) if ring_of[u] == 4 else (v, u)
                radial45.append(((u, v), p, c))

    pos5 = {v: i for i, v in enumerate(cyc5)}

    # boundary turning helper (extrinsic)
    def boundary_turning(P, cycle):
        m = len(cycle); out = np.zeros(m)
        for i in range(m):
            a, b, c = cycle[(i-1) % m], cycle[i], cycle[(i+1) % m]
            e1 = P[b]-P[a]; e1 /= max(np.linalg.norm(e1), 1e-12)
            e2 = P[c]-P[b]; e2 /= max(np.linalg.norm(e2), 1e-12)
            out[i] = math.acos(float(np.clip(e1 @ e2, -1, 1)))
        return out

    # trial files
    for p in ['trial*_ring5_*.ply', 'trial*.ply', '*.ply']:
        files = sorted(glob.glob(os.path.join(library_dir, p)))
        if files: break
    if not files:
        print(f"No PLY files in {library_dir}"); sys.exit(1)

    # accumulators
    fold_by_class = defaultdict(list)      # radial-edge fold by child class
    turn_by_class = defaultdict(list)      # boundary κ_ext by vertex class
    sector_corr, var_ratio = [], []
    sector_sum_all = []

    n_loaded = 0
    for path in files:
        try:
            P = load_trial(path)
        except Exception:
            continue
        if P.shape[0] < n_verts: continue
        P = P[:n_verts]; n_loaded += 1

        dih = dihedral_angles(P, faces, adj)
        fold = {tuple(sorted(e)): abs(math.pi - a) for e, a in dih.items()}

        # 1. compensation per sector
        D4, D5 = [], []
        for p4 in rings[4]:
            f4 = [fold[tuple(sorted(e))] for e in band4_at[p4]
                  if tuple(sorted(e)) in fold]
            f5 = [fold[tuple(sorted(e))] for (e, pp, c) in radial45 if pp == p4
                  and tuple(sorted(e)) in fold]
            if f4 and f5:
                D4.append(np.mean(f4)); D5.append(np.mean(f5))
        D4, D5 = np.array(D4), np.array(D5)
        if len(D4) > 10:
            c = np.corrcoef(D4, D5)[0, 1]
            sector_corr.append(c)
            vr = np.var(D4 + D5) / (np.var(D4) + np.var(D5))
            var_ratio.append(vr)
            sector_sum_all.append(D4 + D5)

        # 2. fold by child class
        for (e, pp, c) in radial45:
            k = tuple(sorted(e))
            if k in fold:
                fold_by_class[cls[pos5[c]]].append(fold[k])

        # boundary turning by class
        kext = boundary_turning(P, cyc5)
        for i, v in enumerate(cyc5):
            turn_by_class[cls[i]].append(kext[i])

    print(f"Loaded {n_loaded} trials")
    sector_corr = np.array(sector_corr)
    var_ratio = np.array(var_ratio)

    lines = []
    lines.append("A/B WORD & COMPENSATION ANALYSIS")
    lines.append(f"trials: {n_loaded}")
    lines.append("")
    lines.append(f"Ring-5 word classes: {dict(class_counts)}")
    lines.append("  (A1: A before lone B; A2: A before BB; B1: lone B;")
    lines.append("   B2a/B2b: first/second B of a BB run)")
    lines.append("")
    lines.append("1. COMPENSATION (per ring-4 sector: D4 = mean band-4 fold,")
    lines.append("   D5 = mean radial 4->5 fold at the same vertex)")
    lines.append(f"   corr(D4, D5):       mean {sector_corr.mean():+.4f}  (sd {sector_corr.std():.4f})")
    lines.append(f"   Var(D4+D5)/(Var D4 + Var D5): mean {var_ratio.mean():.4f}")
    lines.append("   (ratio < 1 = anti-correlated sharing; << 1 = strong conservation;")
    lines.append("    = 1 = independent; > 1 = reinforcing)")
    if sector_sum_all:
        allsum = np.concatenate(sector_sum_all)
        lines.append(f"   per-sector total D4+D5: mean {allsum.mean():.4f}, "
                     f"sd {allsum.std():.4f}, CV {allsum.std()/allsum.mean():.4f}")
    lines.append("")
    lines.append("2. RADIAL FOLD BY CHILD WORD CLASS (rad):")
    lines.append(f"   {'class':>5} | {'n':>8} | {'mean':>7} | {'sd':>6} | {'p90':>6} | {'max':>6}")
    for k in ['A1', 'A2', 'B1', 'B2a', 'B2b']:
        a = np.array(fold_by_class[k]) if fold_by_class[k] else np.array([np.nan])
        lines.append(f"   {k:>5} | {len(fold_by_class[k]):>8} | {np.nanmean(a):>7.4f} | "
                     f"{np.nanstd(a):>6.4f} | {np.nanpercentile(a, 90):>6.4f} | {np.nanmax(a):>6.4f}")
    lines.append("")
    lines.append("3. BOUNDARY EXTRINSIC TURNING BY VERTEX CLASS (rad):")
    lines.append("   (intrinsic floor: A* = 0, B* = π/3 ≈ 1.0472)")
    lines.append(f"   {'class':>5} | {'n':>8} | {'mean':>7} | {'mean-floor':>10} | {'p90':>6}")
    floors = {'A1': 0.0, 'A2': 0.0, 'B1': math.pi/3, 'B2a': math.pi/3, 'B2b': math.pi/3}
    for k in ['A1', 'A2', 'B1', 'B2a', 'B2b']:
        a = np.array(turn_by_class[k]) if turn_by_class[k] else np.array([np.nan])
        lines.append(f"   {k:>5} | {len(turn_by_class[k]):>8} | {np.nanmean(a):>7.4f} | "
                     f"{np.nanmean(a)-floors[k]:>10.4f} | {np.nanpercentile(a, 90):>6.4f}")
    lines.append("")
    lines.append("Interpretation:")
    lines.append(" - Strong conservation (var ratio << 1) + low CV of D4+D5 means the")
    lines.append("   total per-sector fold demand is a near-constant — measure it, and")
    lines.append("   capacity exhaustion becomes: (demand growth φ²/ring) vs (bounded")
    lines.append("   per-sector capacity). That is the provable form.")
    lines.append(" - If B2a/B2b classes carry systematically larger folds/turning than")
    lines.append("   B1, the BB runs of the substitution word are the stress carriers,")
    lines.append("   and σ: A→AB, B→ABB guarantees their density — the localization")
    lines.append("   lemma becomes combinatorial.")

    report = "\n".join(lines)
    print(report)
    with open('ab_report.txt', 'w') as f:
        f.write(report + "\n")
    np.savez('ab_data.npz',
             sector_corr=sector_corr, var_ratio=var_ratio,
             sector_sums=np.concatenate(sector_sum_all) if sector_sum_all else np.array([]),
             **{f'fold_{k}': np.array(v) for k, v in fold_by_class.items()},
             **{f'turn_{k}': np.array(v) for k, v in turn_by_class.items()})
    print("\nWrote ab_report.txt and ab_data.npz")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(__doc__); sys.exit(1)
    main(sys.argv[1])
