#!/usr/bin/env python3
"""
udg_cold_start_eval.py — Does the physics relaxer reliably find a
unit-distance embedding from a COLD start (random positions, no
information from the ground-truth embedding — only the combinatorial
graph + target edge lengths)?

For each NPZ instance in a library, run N independent cold-start
relaxations (random initial positions at a given spatial scale) and
record the resulting max|edge error|. Reports success rate at the
proposal's thresholds (0.05 loose, 0.02 certify-tol), per group
(base vs augmented, ring depth, chord length/density).

This is the first test in the "do hard cold-start instances exist"
question: if cold start reliably converges across these families,
there is no gap for a learned model to fill on THIS data; if it
doesn't, the failing instances define the training/eval target.

Usage:
  python3 udg_cold_start_eval.py --lib udg_library --pattern '*_base.npz' \
      --seeds 5 --iters 3000 --scale 1.0 3.8
"""

import argparse, glob, json, os, time
from collections import defaultdict

import numpy as np

from generate_unit_distance_library import relax, edge_stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lib', default='udg_library')
    ap.add_argument('--pattern', default='*_base.npz')
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--iters', type=int, default=3000)
    ap.add_argument('--scale', type=float, nargs='+', default=[1.0])
    ap.add_argument('--limit', type=int, default=0,
                    help='0 = no limit')
    ap.add_argument('--out', default='')
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.lib, args.pattern)))
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise SystemExit(f"no files matching {args.pattern} in {args.lib}")

    print(f"{len(files)} files, {args.seeds} seeds, "
          f"{args.iters} iters, scales={args.scale}")

    results = []
    t0 = time.time()
    for fi, fname in enumerate(files):
        data = np.load(fname)
        edges = data['edges']
        L = data['lengths']
        V = data['positions'].shape[0]
        dim = data['positions'].shape[1]
        base = os.path.basename(fname)

        for scale in args.scale:
            mxs = []
            stds = []
            for seed in range(args.seeds):
                rng = np.random.default_rng(hash((base, scale, seed)) & 0xffffffff)
                P0 = rng.normal(0, scale, (V, dim))
                P = relax(P0, edges, L, iters=args.iters)
                _, std, mx = edge_stats(P, edges, L)
                mxs.append(mx); stds.append(std)
            mxs = np.array(mxs)
            results.append({
                'file': base, 'V': V, 'E': len(edges), 'dim': dim,
                'scale': scale,
                'mx_per_seed': mxs.tolist(),
                'best_mx': float(mxs.min()),
                'succ005_rate': float((mxs < 0.05).mean()),
                'succ002_rate': float((mxs < 0.02).mean()),
                'any_succ005': bool((mxs < 0.05).any()),
            })
            print(f"[{fi+1}/{len(files)}] {base} scale={scale}: "
                  f"best_mx={mxs.min():.4f} mean_mx={mxs.mean():.4f} "
                  f"succ@0.05={results[-1]['succ005_rate']:.2f} "
                  f"succ@0.02={results[-1]['succ002_rate']:.2f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    if args.out:
        with open(args.out, 'w') as f:
            for r in results:
                f.write(json.dumps(r) + '\n')

    # summary by scale
    print("\n=== SUMMARY ===")
    by_scale = defaultdict(list)
    for r in results:
        by_scale[r['scale']].append(r)
    for scale, rs in sorted(by_scale.items()):
        s005 = np.mean([r['succ005_rate'] for r in rs])
        s002 = np.mean([r['succ002_rate'] for r in rs])
        any5 = np.mean([r['any_succ005'] for r in rs])
        print(f"scale={scale}: mean per-seed succ@0.05={s005:.3f}  "
              f"succ@0.02={s002:.3f}  any-seed succ@0.05={any5:.3f}  "
              f"(n={len(rs)})")


if __name__ == '__main__':
    main()
