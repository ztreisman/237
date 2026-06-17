"""
build_library.py  —  generate a large library of {3,7} physics embeddings
                     and record their geometric statistics.

Each trial grows a disk ring by ring until collision, then records:
  - per-ring edge_std, convergence time, vertex count
  - spherical gap radius and margin at each ring
  - the ring-5 PLY file (if reached), for visualization

Results are appended to library.jsonl (one JSON object per trial) so the
run can be interrupted and resumed. PLY files saved to ply_library/.

Usage:
    python build_library.py [--trials N] [--max-rings R] [--steps S] [--seeds-per-trial K]

Defaults: 500 trials, max 6 rings, 20000 steps, 1 seed per trial (random).
"""

import argparse, json, math, os, time, sys
import numpy as np
from collections import defaultdict
from scipy.spatial import cKDTree

PHI        = (1 + math.sqrt(5)) / 2
EDGE_LENGTH = 1.0

# ── Fibonacci sphere sample (computed once) ───────────────────────────────────
N_SAMP = 5000
_golden = (1 + math.sqrt(5)) / 2
_CANDS  = np.array([
    [math.sin(math.acos(1-2*(k+.5)/N_SAMP))*math.cos(2*math.pi*k/_golden),
     math.sin(math.acos(1-2*(k+.5)/N_SAMP))*math.sin(2*math.pi*k/_golden),
     math.cos(math.acos(1-2*(k+.5)/N_SAMP))]
    for k in range(N_SAMP)])

# ── Combinatorial structure ───────────────────────────────────────────────────
def build_combinatorial(num_rings):
    adj = defaultdict(set); vertex_ring = {}; faces = []; _id = [0]
    def new_v(r):
        v = _id[0]; _id[0] += 1; vertex_ring[v] = r; return v
    seen = set()
    def add_face(a, b, c):
        key = frozenset([a, b, c])
        if key in seen: return
        seen.add(key); faces.append((a, b, c))
        for u, w in [(a,b),(b,c),(a,c)]: adj[u].add(w); adj[w].add(u)
    center = new_v(0); ring1 = [new_v(1) for _ in range(7)]
    for i in range(7): add_face(center, ring1[i], ring1[(i+1)%7])
    boundary = {0: [center], 1: ring1}
    for r in range(2, num_rings+1):
        prev = boundary[r-1]; n = len(prev)
        needed = [7 - len(adj[v]) for v in prev]
        shared    = [new_v(r) for _ in range(n)]
        exclusive = [[new_v(r) for _ in range(needed[i]-2)] for i in range(n)]
        for i in range(n):
            v = prev[i]; v_next = prev[(i+1)%n]
            fan = [shared[(i-1)%n]] + exclusive[i] + [shared[i]]
            for j in range(len(fan)-1): add_face(v, fan[j], fan[j+1])
            add_face(v, v_next, shared[i])
        new_bdy = []
        for i in range(n):
            new_bdy.append(shared[(i-1)%n]); new_bdy.extend(exclusive[i])
        boundary[r] = new_bdy
    return adj, vertex_ring, faces, boundary

def initial_positions(vertex_ring, boundary, rng, z_noise=0.1):
    N   = max(vertex_ring) + 1
    pos = np.zeros((N, 3))
    for r, verts in boundary.items():
        nv = len(verts)
        for i, v in enumerate(verts):
            angle = 2*math.pi*i / max(nv, 1)
            pos[v] = [r*math.cos(angle), r*math.sin(angle), rng.uniform(-z_noise, z_noise)]
    return pos

def edge_std_val(pos, adj):
    u_idx = np.array([u for u in adj for v in adj[u] if v > u])
    v_idx = np.array([v for u in adj for v in adj[u] if v > u])
    if len(u_idx) == 0: return 0.0
    lens = np.linalg.norm(pos[v_idx] - pos[u_idx], axis=1)
    return float(np.std(lens))

# ── Relaxer ───────────────────────────────────────────────────────────────────
def relax(pos, adj, free_verts=None, max_steps=20000,
          converge_window=500, converge_tol=1e-5,
          attract=0.1, repel=0.6):
    pos = pos.copy()
    N   = len(pos)
    if free_verts is None:
        free_verts = list(range(N))
    free_arr = np.zeros(N, dtype=bool)
    for v in free_verts: free_arr[v] = True

    adj_set = {(min(u,v), max(u,v)) for v in range(N) for u in adj[v]}
    adj_mat = np.zeros((N, N), dtype=bool)
    for u, v in adj_set: adj_mat[u,v] = adj_mat[v,u] = True
    adj_arr = np.array(list(adj_set))
    U, V    = adj_arr[:,0], adj_arr[:,1]
    fm      = free_arr[U] | free_arr[V]
    Uf, Vf  = U[fm], V[fm]
    Ur = Vr = np.array([], dtype=int)

    def rebuild():
        nonlocal Ur, Vr
        spread = np.max(np.linalg.norm(pos - pos.mean(axis=0), axis=1))
        if spread < 0.5: Ur = Vr = np.array([], dtype=int); return
        tree  = cKDTree(pos)
        pairs = tree.query_pairs(r=1.5, output_type='ndarray')
        if not len(pairs): Ur = Vr = np.array([], dtype=int); return
        pa, pb = pairs[:,0], pairs[:,1]
        if len(pa) > 5000:
            idx = np.random.choice(len(pa), 5000, replace=False)
            pa, pb = pa[idx], pb[idx]
        keep = ~adj_mat[pa, pb] & (free_arr[pa] | free_arr[pb])
        if not keep.any(): Ur = Vr = np.array([], dtype=int); return
        Ur, Vr = pa[keep], pb[keep]

    def cur_std():
        if len(Uf) == 0: return 0.0
        return float(np.std(np.linalg.norm(pos[Vf]-pos[Uf], axis=1)))

    rebuild()
    std_history = []
    for step in range(max_steps):
        delta = np.zeros((N, 3))
        d  = pos[Vf] - pos[Uf]
        dn = np.linalg.norm(d, axis=1, keepdims=True).clip(1e-12)
        np.add.at(delta, Uf, -d*(1-dn)*attract)
        np.add.at(delta, Vf,  d*(1-dn)*attract)
        if len(Ur):
            dr  = pos[Vr] - pos[Ur]
            drn = np.linalg.norm(dr, axis=1, keepdims=True).clip(1e-12)
            cl  = (drn < 1.0).flatten()
            if cl.any():
                np.add.at(delta, Vr[cl],  dr[cl]*(1-drn[cl])*repel)
                np.add.at(delta, Ur[cl], -dr[cl]*(1-drn[cl])*repel)
        delta[~free_arr] = 0.0
        pos += delta
        if not np.isfinite(pos).all(): return None, step
        if step % 100 == 99:
            rebuild()
            std_history.append(cur_std())
            if len(std_history) >= converge_window // 100:
                window = std_history[-(converge_window // 100):]
                if (window[0] - window[-1]) < converge_tol and window[-1] < 0.5:
                    return pos, step
    return pos, max_steps

# ── Collision detection (vectorized, cached) ──────────────────────────────────
_PAIR_CACHE = {}

def _precompute_pairs(faces, vertex_ring, max_ring):
    faces_arr = np.array(faces); F = len(faces_arr)
    mv = int(faces_arr.max()) + 1
    mem = np.zeros((F, mv), dtype=np.uint8)
    mem[np.arange(F)[:,None], faces_arr] = 1
    shares = (mem @ mem.T) > 0
    vr_arr = np.array([vertex_ring[v] for v in range(max(vertex_ring)+1)])
    fvr    = vr_arr[faces_arr]; aidx = np.arange(F, dtype=np.int32)
    result = {}
    for r in range(1, max_ring+1):
        nm = np.any(fvr == r, axis=1); ni = np.where(nm)[0]
        II, JJ = [], []
        for fi in ni:
            valid = ~shares[fi]; valid[fi] = False
            jj = aidx[valid]
            II.append(np.full(len(jj), fi, dtype=np.int32)); JJ.append(jj)
        result[r] = (
            np.concatenate(II) if II else np.array([], dtype=np.int32),
            np.concatenate(JJ) if JJ else np.array([], dtype=np.int32))
    return result

def _brt(o, d, v0, v1, v2):
    e1=v1-v0; e2=v2-v0; h=np.cross(d,e2)
    a=np.einsum('ij,ij->i',e1,h); ok=np.abs(a)>1e-10
    f=np.where(ok,1/np.where(ok,a,1.),0.); s=o-v0
    u=f*np.einsum('ij,ij->i',s,h); ok&=(u>=0)&(u<=1)
    q=np.cross(s,e1); v=f*np.einsum('ij,ij->i',d,q)
    ok&=(v>=0)&(u+v<=1); t=f*np.einsum('ij,ij->i',e2,q)
    return ok&(t>1e-10)

def has_collision(pos, faces, check_rings, vertex_ring):
    faces_arr = np.array(faces)
    mr  = max(vertex_ring.values())
    key = (len(faces), mr)
    if key not in _PAIR_CACHE:
        _PAIR_CACHE[key] = _precompute_pairs(faces, vertex_ring, mr)
    cache = _PAIR_CACHE[key]
    for r in check_rings:
        if r not in cache: continue
        II, JJ = cache[r]
        if not len(II): continue
        t1 = pos[faces_arr[II]]; t2 = pos[faces_arr[JJ]]
        for s, g in ((t2,t1),(t1,t2)):
            for ei, ej in ((0,1),(0,2),(1,2)):
                p=s[:,ei,:]; q=s[:,ej,:]; d=q-p
                if np.any(_brt(p,d,g[:,0,:],g[:,1,:],g[:,2,:]) &
                           _brt(q,-d,g[:,0,:],g[:,1,:],g[:,2,:])): return True
    return False

# ── Spherical gap analysis ────────────────────────────────────────────────────
def gap_radius_deg(unit_vecs):
    if len(unit_vecs) == 0: return 180.0
    dots = np.clip(_CANDS @ unit_vecs.T, -1, 1)
    return float(np.degrees(np.arccos(dots.max(axis=1))).max())

def gap_center_and_radius(unit_vecs):
    """Return (gap_center unit vec, gap_radius_deg)."""
    if len(unit_vecs) == 0: return np.array([0,0,1.0]), 180.0
    dots    = np.clip(_CANDS @ unit_vecs.T, -1, 1)
    min_ang = np.degrees(np.arccos(dots.max(axis=1)))
    idx     = int(np.argmax(min_ang))
    return _CANDS[idx], float(min_ang[idx])

def gap_profile(gap_center, boundary_units, n_az=36):
    """Angular supply profile around gap center at n_az azimuth samples."""
    z = gap_center
    x = np.array([1,0,0]) if abs(z[0]) < 0.9 else np.array([0,1,0])
    x = x - np.dot(x,z)*z; x /= np.linalg.norm(x)
    y = np.cross(z, x)
    dots_z  = np.clip(boundary_units @ z, -1, 1)
    colats  = np.degrees(np.arccos(dots_z))
    proj    = boundary_units - dots_z[:,None]*z[None,:]
    pnorm   = np.linalg.norm(proj, axis=1, keepdims=True).clip(1e-8)
    tang_dirs = proj / pnorm
    radii = []
    step  = 360.0 / n_az
    for k in range(n_az):
        rad  = math.radians(k * step)
        tang = math.cos(rad)*x + math.sin(rad)*y
        az_dots = tang_dirs @ tang
        sector  = az_dots > math.cos(math.radians(step))
        radii.append(float(colats[sector].min() if sector.any() else colats.min()))
    return radii

def ring_gap_stats(pos, vertex_ring, r, adj=None, n_az=36):
    center = pos[next(v for v,rv in vertex_ring.items() if rv == 0)]
    v = pos - center
    vr_arr = np.array([vertex_ring[i] for i in range(len(pos))])
    rv = v[vr_arr == r]
    dists = np.linalg.norm(rv, axis=1)
    good  = dists > 1e-6
    if not good.any(): return None
    rv = rv[good]; dists = dists[good]
    units  = rv / dists[:,None]
    gc, gr = gap_center_and_radius(units)
    r_mean = float(dists.mean())
    r_std  = float(dists.std())
    strip  = math.degrees(math.asin(min(1.0, 1.0/r_mean)))

    # Gap profile
    profile = gap_profile(gc, units, n_az)
    p_mean  = float(np.mean(profile))
    p_std   = float(np.std(profile))
    p_min   = float(np.min(profile))

    # Local demand: arcsin(1/radius) per boundary vertex
    demands = [math.degrees(math.asin(min(1.0, 1.0/d))) for d in dists]
    d_mean  = float(np.mean(demands))
    d_std   = float(np.std(demands))
    d_min   = float(np.min(demands))
    d_max   = float(np.max(demands))

    # Curvature of boundary on sphere (mean colat of neighbors vs vertex)
    if adj is not None:
        bdy_idx = [i for i,rv2 in enumerate(vr_arr) if rv2 == r and dists[list(np.where(vr_arr==r)[0]).index(i)] > 1e-6] if False else []
        # simpler: skip detailed curvature here, add in post-processing
        pass

    # Supply-demand balance across azimuths
    n_neg = sum(1 for p in profile if p < d_mean)

    return {
        'gap_deg':      round(gr, 4),
        'gap_center':   [round(x,4) for x in gc.tolist()],
        'strip_deg':    round(strip, 4),
        'margin_deg':   round(gr - strip, 4),
        'radius_mean':  round(r_mean, 4),
        'radius_std':   round(r_std, 4),
        'profile_mean': round(p_mean, 4),
        'profile_std':  round(p_std, 4),
        'profile_min':  round(p_min, 4),
        'demand_mean':  round(d_mean, 4),
        'demand_std':   round(d_std, 4),
        'demand_min':   round(d_min, 4),
        'demand_max':   round(d_max, 4),
        'supply_deficit_frac': round(n_neg / n_az, 4),
    }

# ── PLY export ────────────────────────────────────────────────────────────────
VIRIDIS = [(255,255,255),(68,1,84),(71,44,122),(59,81,139),(44,113,142),
           (33,144,141),(39,173,129),(92,200,100),(170,220,50),(234,229,26),(253,231,37)]

def save_ply(path, pos, faces, vertex_ring):
    with open(path, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pos)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\nend_header\n")
        for i, p in enumerate(pos):
            r, g, b = VIRIDIS[min(vertex_ring.get(i,0), len(VIRIDIS)-1)]
            f.write(f"{p[0]:.5f} {p[1]:.5f} {p[2]:.5f} {r} {g} {b}\n")
        for a, b, c in faces:
            f.write(f"3 {a} {b} {c}\n")

# ── Main trial runner ─────────────────────────────────────────────────────────
def run_trial(trial_id, rng, max_rings, max_steps, save_ply_rings, ply_dir):
    """
    Grow one disk, return a dict of results.
    """
    result = {
        'trial_id':  trial_id,
        'seed':      int(rng.integers(0, 2**31)),
        'timestamp': time.time(),
        'rings':     {},
        'max_ring_reached': 0,
        'collision_at':     None,
    }

    adj, vertex_ring, faces, boundary = build_combinatorial(1)
    pos = initial_positions(vertex_ring, boundary, rng)

    # Relax seed
    t0  = time.time()
    pos, steps = relax(pos, adj, max_steps=max_steps)
    if pos is None:
        result['error'] = 'nan_seed'
        return result

    std = edge_std_val(pos, adj)
    result['rings'][1] = {
        'n_verts': int((np.array([vertex_ring[i] for i in range(len(pos))]) == 1).sum()),
        'edge_std': round(std, 6),
        'steps':    steps,
        'time_s':   round(time.time()-t0, 2),
        **({} if True else {}),   # gap added below
    }
    result['rings'][1].update(ring_gap_stats(pos, vertex_ring, 1) or {})

    valid_ring = 1

    for r in range(2, max_rings+1):
        t0 = time.time()
        adj_new, vr_new, faces_new, bdy_new = build_combinatorial(r)
        N_new = max(vr_new) + 1

        # Warm-start new ring at neighbor centroids
        pos_new = np.zeros((N_new, 3))
        for v in range(len(pos)): pos_new[v] = pos[v]
        new_verts = [v for v, rv in vr_new.items() if rv == r]
        for v in new_verts:
            nbrs = [u for u in adj_new[v] if u < len(pos)]
            if nbrs:
                pos_new[v] = np.mean([pos[u] for u in nbrs], axis=0)
                pos_new[v][2] += rng.uniform(-0.1, 0.1)

        # Relax full mesh
        pos_new, steps = relax(pos_new, adj_new, max_steps=max_steps)
        if pos_new is None:
            result['error'] = f'nan_ring{r}'
            break

        collides = has_collision(pos_new, faces_new, {r}, vr_new)
        std      = edge_std_val(pos_new, adj_new)
        elapsed  = round(time.time()-t0, 2)

        ring_info = {
            'n_verts':  int((np.array([vr_new[i] for i in range(N_new)]) == r).sum()),
            'edge_std': round(std, 6),
            'steps':    steps,
            'time_s':   elapsed,
            'collision': collides,
        }
        gap = ring_gap_stats(pos_new, vr_new, r)
        if gap: ring_info.update(gap)

        result['rings'][r] = ring_info

        if collides:
            result['collision_at'] = r
            break

        # Accept ring
        adj, vertex_ring, faces, boundary = adj_new, vr_new, faces_new, bdy_new
        pos = pos_new
        valid_ring = r

        # Re-relax full mesh (option: skip to save time)
        t1 = time.time()
        pos_rr, steps_rr = relax(pos, adj, max_steps=max_steps)
        if pos_rr is not None:
            std_rr = edge_std_val(pos_rr, adj)
            result['rings'][r]['edge_std_rerelax'] = round(std_rr, 6)
            result['rings'][r]['rerelax_time_s']   = round(time.time()-t1, 2)
            gap_rr = ring_gap_stats(pos_rr, vertex_ring, r)
            if gap_rr:
                result['rings'][r]['gap_deg_rerelax']    = gap_rr['gap_deg']
                result['rings'][r]['margin_deg_rerelax']  = gap_rr['margin_deg']
            pos = pos_rr

        # Save PLY if requested
        if r in save_ply_rings:
            fname = os.path.join(ply_dir, f"trial{trial_id:05d}_ring{r}_std{std:.4f}.ply")
            save_ply(fname, pos, faces, vertex_ring)
            result['rings'][r]['ply'] = fname

    result['max_ring_reached'] = valid_ring
    return result


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials',    type=int, default=500)
    parser.add_argument('--max-rings', type=int, default=6)
    parser.add_argument('--steps',     type=int, default=20000)
    parser.add_argument('--save-ply',  type=str, default='5',
                        help='comma-separated ring numbers to save PLY for (default: 5)')
    parser.add_argument('--outdir',    type=str, default='.')
    parser.add_argument('--resume',    action='store_true',
                        help='append to existing library.jsonl instead of starting fresh')
    args = parser.parse_args()

    save_ply_rings = set(int(x) for x in args.save_ply.split(',') if x.strip())
    ply_dir  = os.path.join(args.outdir, 'ply_library')
    jsonl_path = os.path.join(args.outdir, 'library.jsonl')
    os.makedirs(ply_dir, exist_ok=True)

    # Find starting trial_id if resuming
    start_id = 0
    if args.resume and os.path.exists(jsonl_path):
        with open(jsonl_path) as f:
            lines = [l for l in f if l.strip()]
        start_id = len(lines)
        print(f"Resuming from trial {start_id}")

    mode = 'a' if args.resume else 'w'
    rng  = np.random.default_rng()

    print(f"{'='*60}")
    print(f"  Building library: {args.trials} trials, max {args.max_rings} rings")
    print(f"  Steps: {args.steps}, output: {jsonl_path}")
    print(f"{'='*60}")

    t_total = time.time()
    with open(jsonl_path, mode) as out:
        for i in range(args.trials):
            trial_id = start_id + i
            t0 = time.time()
            result = run_trial(trial_id, rng, args.max_rings,
                               args.steps, save_ply_rings, ply_dir)
            elapsed = time.time() - t0

            out.write(json.dumps(result) + '\n')
            out.flush()

            mr   = result['max_ring_reached']
            coll = result.get('collision_at', '—')
            r5   = result['rings'].get(5, {})
            gap5 = f"{r5.get('gap_deg_rerelax', r5.get('gap_deg', '—')):.2f}°" if 5 in result['rings'] else '—'
            print(f"  [{trial_id:5d}] max_ring={mr}  collision_at={coll}  "
                  f"ring5_gap={gap5}  t={elapsed:.0f}s", flush=True)

    total = time.time() - t_total
    print(f"\nDone. {args.trials} trials in {total/60:.1f} min -> {jsonl_path}")
