"""
gap_analysis.py  —  compute spherical gap analysis for all physics_grow PLY files.

Run on lambda01:
    python gap_analysis.py [directory]

Saves results to gap_analysis.json in the same directory.
"""

import sys, os, json, math, glob
import numpy as np

RING_COLORS = [
    (255,255,255),(68,1,84),(71,44,122),(59,81,139),(44,113,142),(33,144,141),
    (39,173,129),(92,200,100),(170,220,50),(234,229,26),(253,231,37),
]

def closest_ring(rgb):
    return min(range(len(RING_COLORS)),
               key=lambda r: sum((a-b)**2 for a,b in zip(rgb, RING_COLORS[r])))

def load_ply(path):
    with open(path) as f:
        lines = f.readlines()
    i = 0; n_verts = 0
    while lines[i].strip() != 'end_header':
        if lines[i].startswith('element vertex'):
            n_verts = int(lines[i].split()[-1])
        i += 1
    i += 1
    verts = []; colors = []
    for j in range(n_verts):
        p = lines[i+j].split()
        verts.append([float(p[0]),float(p[1]),float(p[2])])
        colors.append((int(p[3]),int(p[4]),int(p[5])))
    return np.array(verts), [closest_ring(c) for c in colors]

# Fibonacci sphere sample — computed once
N_SAMP = 10000
golden = (1 + math.sqrt(5)) / 2
_CANDS = np.array([
    [math.sin(math.acos(1-2*(k+.5)/N_SAMP))*math.cos(2*math.pi*k/golden),
     math.sin(math.acos(1-2*(k+.5)/N_SAMP))*math.sin(2*math.pi*k/golden),
     math.cos(math.acos(1-2*(k+.5)/N_SAMP))]
    for k in range(N_SAMP)])

def gap_radius_deg(unit_vecs):
    """Largest angular radius of a spherical cap containing no unit_vecs."""
    dots = np.clip(_CANDS @ unit_vecs.T, -1, 1)
    min_angs = np.degrees(np.arccos(dots.max(axis=1)))
    return float(min_angs.max())

def analyze_ply(path):
    verts, vring = load_ply(path)
    vring = np.array(vring)

    # Center on ring-0 vertex
    r0 = verts[vring == 0]
    if len(r0) == 0:
        return None
    center = r0[0]
    v = verts - center

    max_ring = int(vring.max())
    result = {'file': os.path.basename(path), 'max_ring': max_ring, 'rings': {}}

    for r in range(1, max_ring+1):
        rv = v[vring == r]
        dists = np.linalg.norm(rv, axis=1)
        good = dists > 1e-6
        rv = rv[good]; dists = dists[good]
        if len(rv) == 0:
            continue
        units = rv / dists[:,None]

        # All verts through ring r (for "disk" gap)
        allv = v[np.logical_and(vring >= 1, vring <= r)]
        alld = np.linalg.norm(allv, axis=1)
        allg = alld > 1e-6
        allv = allv[allg]; alld = alld[allg]
        all_units = allv / alld[:,None]

        gap_bdy  = gap_radius_deg(units)
        gap_disk = gap_radius_deg(all_units)

        r_mean   = float(dists.mean())
        r_min    = float(dists.min())
        r_max    = float(dists.max())
        strip    = math.degrees(math.asin(min(1.0, 1.0 / r_mean)))
        margin   = gap_disk - strip

        result['rings'][r] = {
            'n_verts':    int((vring == r).sum()),
            'radius_mean': round(r_mean, 4),
            'radius_min':  round(r_min, 4),
            'radius_max':  round(r_max, 4),
            'gap_bdy_deg':  round(gap_bdy, 4),
            'gap_disk_deg': round(gap_disk, 4),
            'strip_deg':    round(strip, 4),
            'margin_deg':   round(margin, 4),
        }

    return result

if __name__ == '__main__':
    directory = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser('~')
    pattern = os.path.join(directory, 'physics_grow_*.ply')
    files = sorted(glob.glob(pattern))
    if not files:
        pattern = os.path.join(directory, 'physics_best.ply')
        files = glob.glob(pattern)
    if not files:
        print(f"No physics_grow_*.ply files found in {directory}")
        sys.exit(1)

    print(f"Found {len(files)} PLY files. Analyzing...")
    results = []
    for i, path in enumerate(files):
        print(f"  [{i+1}/{len(files)}] {os.path.basename(path)} ...", flush=True)
        r = analyze_ply(path)
        if r:
            # Print key numbers
            rings = r['rings']
            max_r = r['max_ring']
            if max_r in rings:
                info = rings[max_r]
                print(f"    ring {max_r}: gap={info['gap_disk_deg']:.2f}°  "
                      f"strip={info['strip_deg']:.2f}°  margin={info['margin_deg']:+.2f}°")
            results.append(r)

    out = os.path.join(directory, 'gap_analysis.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} results to {out}")

    # Print summary
    print("\n=== Summary: margin at max ring ===")
    print(f"{'File':45s} {'max_ring':>9} {'gap':>8} {'strip':>8} {'margin':>8}")
    for r in results:
        mr = r['max_ring']
        if mr in r['rings']:
            info = r['rings'][mr]
            print(f"  {r['file']:43s}  ring {mr}  "
                  f"{info['gap_disk_deg']:6.2f}°  {info['strip_deg']:6.2f}°  "
                  f"{info['margin_deg']:+6.2f}°")
