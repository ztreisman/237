"""
physics_grow.py  —  grow a {3,7} hyperbolic mesh ring by ring using
pure physics relaxation, saving each valid ring as a .ply/.obj file.
Runs until it hits a collision or reaches MAX_RINGS.

Usage:  python physics_grow.py
"""

import sys, math, time
import numpy as np
from collections import defaultdict
from scipy.spatial import cKDTree

# ── Paste-in of the core functions from hyperbolic_triangle_gnn.py ───────────
# (so this script is self-contained and can run standalone on lambda01)

PHI        = (1 + math.sqrt(5)) / 2
EDGE_LENGTH = 1.0
MAX_RINGS   = 10          # keep growing until collision
RELAX_STEPS = 100000      # max steps per relax (usually converges much sooner)
ATTRACT     = 0.1
REPEL       = 0.6
NUM_SEEDS   = 20          # try this many random seeds, keep the best

# ── Combinatorial structure ───────────────────────────────────────────────────
def build_combinatorial(num_rings):
    adj = defaultdict(set); vertex_ring = {}; faces = []; _id = [0]
    def new_v(r):
        v = _id[0]; _id[0] += 1; vertex_ring[v] = r; return v
    seen_faces = set()
    def add_face(a, b, c):
        key = frozenset([a, b, c])
        if key in seen_faces: return
        seen_faces.add(key); faces.append((a, b, c))
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

def initial_positions(vertex_ring, boundary, rng):
    N   = max(vertex_ring) + 1
    pos = np.zeros((N, 3))
    for r, verts in boundary.items():
        nv = len(verts)
        for i, v in enumerate(verts):
            angle = 2*math.pi*i / max(nv, 1)
            pos[v] = [r*math.cos(angle), r*math.sin(angle), rng.uniform(-0.1,0.1)]
    return pos

def edge_std(pos, adj):
    lens = [np.linalg.norm(pos[v]-pos[u]) for v in adj for u in adj[v] if u > v]
    return float(np.std(lens))

# ── Vectorized relaxer ────────────────────────────────────────────────────────
def relax(pos, adj, free_verts=None, steps=10000,
          converge_window=500, converge_tol=1e-5):
    """
    Relax until convergence or max steps.
    Convergence: edge_std improves by less than converge_tol
    over the last converge_window steps.
    steps is treated as a maximum; pass steps=None for unlimited.
    """
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
        lens = np.linalg.norm(pos[V] - pos[U], axis=1)
        return float(np.std(lens))

    rebuild()
    std_history = []
    max_steps = steps if steps is not None else 10_000_000

    for step in range(max_steps):
        delta = np.zeros((N, 3))
        d     = pos[Vf] - pos[Uf]
        dn    = np.linalg.norm(d, axis=1, keepdims=True).clip(1e-12)
        np.add.at(delta, Uf, -d*(1-dn)*ATTRACT)
        np.add.at(delta, Vf,  d*(1-dn)*ATTRACT)
        if len(Ur):
            dr = pos[Vr] - pos[Ur]
            drn = np.linalg.norm(dr, axis=1, keepdims=True).clip(1e-12)
            cl  = (drn < 1.0).flatten()
            if cl.any():
                np.add.at(delta, Vr[cl],  dr[cl]*(1-drn[cl])*REPEL)
                np.add.at(delta, Ur[cl], -dr[cl]*(1-drn[cl])*REPEL)
        delta[~free_arr] = 0.0
        pos += delta
        if not np.isfinite(pos).all():
            print(f"    [relax] NaN/inf at step {step}, aborting")
            return None
        if step % 100 == 99:
            rebuild()
            std_history.append(cur_std())
            # Check convergence over last window
            if len(std_history) >= converge_window // 100:
                window = std_history[-(converge_window // 100):]
                improvement = window[0] - window[-1]
                if improvement < converge_tol and window[-1] < 0.5:
                    return pos  # converged
    return pos

# ── Collision detection (vectorized) ─────────────────────────────────────────
_PAIR_CACHE = {}

def _precompute_pairs(faces, vertex_ring, max_ring):
    faces_arr = np.array(faces); F = len(faces_arr)
    mv = int(faces_arr.max())+1
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
            v = ~shares[fi]; v[fi] = False; jj = aidx[v]
            II.append(np.full(len(jj), fi, dtype=np.int32)); JJ.append(jj)
        result[r] = (np.concatenate(II) if II else np.array([], dtype=np.int32),
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
            for ei,ej in ((0,1),(0,2),(1,2)):
                p=s[:,ei,:]; q=s[:,ej,:]; d=q-p
                if np.any(_brt(p,d,g[:,0,:],g[:,1,:],g[:,2,:]) &
                           _brt(q,-d,g[:,0,:],g[:,1,:],g[:,2,:])): return True
    return False

# ── PLY / OBJ export ──────────────────────────────────────────────────────────
# Viridis colormap sampled at 11 points (ring 0 = center = white)
RING_COLORS = [
    (255, 255, 255),  # 0  white (center)
    ( 68,   1,  84),  # 1  viridis 0.0  deep purple
    ( 71,  44, 122),  # 2  viridis 0.1
    ( 59,  81, 139),  # 3  viridis 0.2
    ( 44, 113, 142),  # 4  viridis 0.3
    ( 33, 144, 141),  # 5  viridis 0.4
    ( 39, 173, 129),  # 6  viridis 0.5
    ( 92, 200, 100),  # 7  viridis 0.6
    (170, 220,  50),  # 8  viridis 0.7
    (234, 229,  26),  # 9  viridis 0.8
    (253, 231,  37),  # 10 viridis 0.9  yellow
]
def _col(r): return RING_COLORS[min(r, len(RING_COLORS)-1)]

def save_ply(path, pos, faces, vertex_ring):
    with open(path, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pos)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\nend_header\n")
        for i, p in enumerate(pos):
            r,g,b = _col(vertex_ring.get(i,0))
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {r} {g} {b}\n")
        for a,b,c in faces:
            f.write(f"3 {a} {b} {c}\n")

def save_obj(path, pos, faces):
    with open(path,'w') as f:
        f.write("# {3,7} hyperbolic mesh\n")
        for p in pos: f.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        for a,b,c in faces: f.write(f"f {a+1} {b+1} {c+1}\n")

# ── Main: grow ring by ring ───────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("  {3,7} Physics-only mesh grower")
    print("=" * 60)

    best_ever_ring = 0
    best_ever_pos  = None
    best_ever_data = None

    for seed_idx in range(NUM_SEEDS):
        rng = np.random.default_rng(seed_idx * 17 + 42)
        print(f"\n── Seed {seed_idx+1}/{NUM_SEEDS} ──")

        # Start with just rings 0+1, extend combinatorially as we grow
        adj, vertex_ring, faces, boundary = build_combinatorial(1)
        pos = initial_positions(vertex_ring, boundary, rng)

        # Relax seed (rings 0+1) simultaneously
        t0 = time.time()
        pos = relax(pos, adj, free_verts=None, steps=RELAX_STEPS)
        print(f"  Ring 1 seed: edge_std={edge_std(pos,adj):.4f}  t={time.time()-t0:.1f}s")

        valid_ring = 1
        best_pos   = pos.copy()

        for r in range(2, MAX_RINGS+1):
            t0 = time.time()

            # Extend combinatorial structure by one ring
            adj_new, vr_new, faces_new, bdy_new = build_combinatorial(r)

            # Copy existing positions into new (larger) pos array
            N_new = max(vr_new)+1
            pos_new = np.zeros((N_new, 3))
            for v in range(len(pos)):   # copy known verts
                pos_new[v] = pos[v]

            # Warm-start new ring vertices: place each one at the centroid
            # of its already-placed neighbors, plus small z noise.
            # This prevents the force explosion from starting far off.
            new_verts = [v for v, rv in vr_new.items() if rv == r]
            for v in new_verts:
                nbrs = [u for u in adj_new[v] if u < len(pos)]
                if nbrs:
                    pos_new[v] = np.mean([pos[u] for u in nbrs], axis=0)
                    pos_new[v][2] += rng.uniform(-0.1, 0.1)
                else:
                    # fallback: concentric ring position
                    angle = 2*math.pi * v / max(len(new_verts), 1)
                    pos_new[v] = [r*math.cos(angle), r*math.sin(angle),
                                  rng.uniform(-0.1, 0.1)]

            # Relax FULL mesh simultaneously — ALL rings free.
            # Inner rings re-adjust to accommodate the new ring,
            # which is the key difference from ring-by-ring freezing.
            pos_new = relax(pos_new, adj_new, free_verts=None,
                            steps=RELAX_STEPS)

            if pos_new is None:
                print(f"  Ring {r}: NaN divergence, stopping seed")
                break

            collides = has_collision(pos_new, faces_new,
                                     check_rings={r},
                                     vertex_ring=vr_new)
            std = edge_std(pos_new, adj_new)
            elapsed = time.time() - t0

            if collides:
                print(f"  Ring {r}: COLLISION  (edge_std={std:.4f}, {elapsed:.1f}s)")
                print(f"  → Stopped at ring {valid_ring}")
                break
            else:
                # Accept this ring, then re-relax the ENTIRE mesh
                # (all rings free) so inner rings can adjust before
                # the next ring is added.
                t1 = time.time()
                pos_rerelax = relax(pos_new, adj_new, free_verts=None,
                                    steps=RELAX_STEPS)
                if pos_rerelax is not None:
                    std2 = edge_std(pos_rerelax, adj_new)
                    print(f"    re-relax: edge_std {std:.4f} → {std2:.4f}  t={time.time()-t1:.1f}s")
                    pos_new = pos_rerelax
                    std = std2

                adj, vertex_ring, faces, boundary = adj_new, vr_new, faces_new, bdy_new
                pos        = pos_new
                valid_ring = r
                best_pos   = pos.copy()
                print(f"  Ring {r}: OK  edge_std={std:.4f}  N={N_new}  t={elapsed:.1f}s")

                fname = f"physics_grow_seed{seed_idx+1:02d}_ring{r:02d}_std{std:.4f}"
                save_ply(fname + ".ply", pos, faces, vertex_ring)
                save_obj(fname + ".obj", pos, faces)
                print(f"    saved {fname}.ply")

        if valid_ring > best_ever_ring:
            best_ever_ring = valid_ring
            best_ever_pos  = best_pos.copy()
            best_ever_data = (adj, vertex_ring, faces)
            print(f"  ★ New best: {best_ever_ring} rings!")

    print(f"\n{'='*60}")
    print(f"Best result: {best_ever_ring} rings across {NUM_SEEDS} seeds")
    if best_ever_pos is not None:
        adj, vr, f = best_ever_data
        save_ply("physics_best.ply", best_ever_pos, f, vr)
        save_obj("physics_best.obj", best_ever_pos, f)
        print(f"Best mesh saved: physics_best.ply / .obj")
    print("Done.")
