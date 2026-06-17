"""
physics_grow_zip.py  —  {3,7} mesh grower with edge-identification snap.

Key difference from physics_grow_snap.py:
  - Snap unit is a BOUNDARY EDGE PAIR, not a vertex pair.
  - Identifying edge (u1,v1) with (u2,v2) does two vertex merges simultaneously,
    which IS a manifold operation (no shared-neighbor non-manifold issue).
  - Cascade = zipper: after one edge identification, the adjacent boundary
    edge pairs become the next candidates. The repulsion between committed
    zip partners is reversed to attraction.
  - Physics: zip_force() applies an attractive spring between each committed
    vertex pair, letting the relaxer actively pull the seam closed before
    snapping discrete steps.

Usage:  python physics_grow_zip.py [--rings N] [--seeds N]
"""

import sys, math, time, json, argparse
import numpy as np
from collections import defaultdict
from scipy.spatial import cKDTree

# ── Constants ─────────────────────────────────────────────────────────────────
EDGE_LENGTH  = 1.0
MAX_RINGS    = 8
RELAX_STEPS  = 10000
ATTRACT      = 0.10
REPEL        = 0.60
ZIP_STRENGTH = 0.40   # attraction force for committed zip pairs
NUM_SEEDS    = 3

ZIP_EPSILON  = 0.50   # boundary edge midpoint distance to trigger zip candidate
ZIP_PARALLEL = 0.70   # |cos θ| threshold — accept near-parallel AND near-antiparallel

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
    return float(np.std(lens)) if lens else 0.0

# ── Boundary structure ────────────────────────────────────────────────────────
def get_boundary_edges(faces):
    """Return set of (min,max) vertex pairs that are boundary (valence-1) edges."""
    ec = defaultdict(int)
    for a,b,c in faces:
        for e in [(min(a,b),max(a,b)),(min(b,c),max(b,c)),(min(a,c),max(a,c))]:
            ec[e] += 1
    return {e for e,cnt in ec.items() if cnt == 1}

def get_boundary_loop(faces):
    """
    Return ordered boundary vertices as a loop (or multiple loops as list of lists).
    Each boundary edge has exactly one face; we chain them into loops.
    """
    bdy = get_boundary_edges(faces)
    adj_bdy = defaultdict(list)
    for a,b in bdy:
        adj_bdy[a].append(b); adj_bdy[b].append(a)

    visited = set(); loops = []
    for start in sorted(adj_bdy):
        if start in visited: continue
        loop = [start]; visited.add(start)
        cur = start
        while True:
            nbrs = [v for v in adj_bdy[cur] if v not in visited]
            if not nbrs: break
            nxt = nbrs[0]; visited.add(nxt); loop.append(nxt); cur = nxt
        loops.append(loop)
    return loops

# ── Edge-snap detection ───────────────────────────────────────────────────────
def _interior_edges(faces):
    """Return set of (min,max) vertex pairs that are interior (valence >= 2)."""
    ec = defaultdict(int)
    for a,b,c in faces:
        for e in [(min(a,b),max(a,b)),(min(b,c),max(b,c)),(min(a,c),max(a,c))]:
            ec[e] += 1
    return {e for e,v in ec.items() if v >= 2}


def _merge_manifold_safe(va, vb, adj, interior):
    """
    True iff merging vb into va creates no non-manifold edges.

    After the merge, for every shared neighbour p of va and vb, edge (va,p)
    gains one more face.  If that edge is already interior (valence 2), it
    would become valence 3 — non-manifold.  So we require that every shared
    neighbour p has edge (va,p) currently boundary (valence 1).
    """
    for p in adj[va] & adj[vb]:
        if (min(va,p), max(va,p)) in interior:
            return False
    return True


def find_edge_snap_candidates(pos, faces, adj, epsilon, parallel_thresh):
    """
    Find pairs of boundary edges whose midpoints are within epsilon and whose
    directions are nearly parallel or anti-parallel.

    For each candidate pair ((u1,v1),(u2,v2)):
      - Choose orientation (parallel/antiparallel) by smaller vertex displacement.
      - Full cascade-safe manifold check for both sequential merges.

    Returns list of (midpoint_dist, (u1,v1), (u2,v2), orientation, vertex_pairs)
    sorted by midpoint distance.
    """
    bdy = get_boundary_edges(faces)
    if not bdy: return []

    interior = _interior_edges(faces)

    edges  = list(bdy)
    mids   = np.array([(pos[a]+pos[b])/2 for a,b in edges])
    dirs   = np.array([pos[b]-pos[a] for a,b in edges])
    norms  = np.linalg.norm(dirs, axis=1, keepdims=True).clip(1e-12)
    dirs   = dirs / norms

    tree  = cKDTree(mids)
    pairs = tree.query_pairs(r=epsilon, output_type='ndarray')

    candidates = []
    for i,j in pairs:
        u1,v1 = edges[i]; u2,v2 = edges[j]
        if u1 in (u2,v2) or v1 in (u2,v2): continue

        cos_theta = float(dirs[i] @ dirs[j])
        if abs(cos_theta) < parallel_thresh: continue

        # Choose orientation by smaller total vertex displacement
        dist_par  = (np.linalg.norm(pos[u1]-pos[u2]) +
                     np.linalg.norm(pos[v1]-pos[v2]))
        dist_anti = (np.linalg.norm(pos[u1]-pos[v2]) +
                     np.linalg.norm(pos[v1]-pos[u2]))

        if dist_anti <= dist_par:
            a1,a2,b1,b2 = u1,v2,v1,u2
            orient = 'antiparallel'
        else:
            a1,a2,b1,b2 = u1,u2,v1,v2
            orient = 'parallel'

        # Full cascade-safe manifold check.
        # Merge 1 (a1←a2): safe iff no shared neighbour of a1,a2 has (a1,p) interior.
        if not _merge_manifold_safe(a1, a2, adj, interior): continue

        # Simulate adj[a1] after merge 1 to check merge 2.
        # After a1←a2: adj[a1] grows to adj[a1] | adj[a2].
        # Merge 2 (b1←b2): shared neighbours are (adj[b1] | something) & adj[b2].
        # We only need to check with the updated adj[a1] for neighbours of b1 that
        # came from a2 — but b1 itself doesn't change from merge 1.
        # Simpler sufficient condition: use the union adj[a1]|adj[a2] as the new adj[a1].
        sim_adj_a1 = adj[a1] | adj[a2]
        sim_adj    = defaultdict(set, {a1: sim_adj_a1, **{v: adj[v] for v in adj}})
        if not _merge_manifold_safe(b1, b2, sim_adj, interior): continue

        mid_dist = float(np.linalg.norm(mids[i]-mids[j]))
        candidates.append((mid_dist, (u1,v1), (u2,v2), orient, ((a1,a2),(b1,b2))))

    return sorted(candidates)

# ── Edge-snap execution ───────────────────────────────────────────────────────
def execute_vertex_merge(va, vb, pos, adj, vertex_ring, faces):
    """Merge vb into va (midpoint). Manifold-safe if va,vb share no common neighbors."""
    pos = pos.copy()
    pos[va] = (pos[va] + pos[vb]) / 2.0
    for nb in list(adj[vb]):
        if nb != va:
            adj[va].add(nb); adj[nb].discard(vb); adj[nb].add(va)
    adj[vb].clear(); adj[va].discard(vb)
    new_faces = []
    for face in faces:
        f = tuple(va if v==vb else v for v in face)
        if len(set(f)) == 3: new_faces.append(f)
    faces[:] = new_faces
    vertex_ring[vb] = -1
    return pos, adj, vertex_ring, faces

def execute_edge_snap(u1, v1, u2, v2, orient, pos, adj, vertex_ring, faces):
    """
    Identify boundary edge (u1,v1) with (u2,v2).
    Orientation determines which endpoints merge:
      parallel:     u1↔u2, v1↔v2
      antiparallel: u1↔v2, v1↔u2
    Returns updated (pos, adj, vertex_ring, faces) and the two merge pairs used.
    """
    if orient == 'antiparallel':
        a1,b1, a2,b2 = u1,v1, v2,u2   # merge a1↔a2, b1↔b2
    else:
        a1,b1, a2,b2 = u1,v1, u2,v2

    pos, adj, vertex_ring, faces = execute_vertex_merge(a1, a2, pos, adj, vertex_ring, faces)
    # b2 may have been renumbered to a2 during first merge — no, they're distinct vertices
    pos, adj, vertex_ring, faces = execute_vertex_merge(b1, b2, pos, adj, vertex_ring, faces)
    return pos, adj, vertex_ring, faces, (a1,a2), (b1,b2)

# ── Zip cascade ───────────────────────────────────────────────────────────────
def zip_cascade(pos, adj, vertex_ring, faces, epsilon, parallel_thresh,
                max_zips=200, verbose=True):
    """
    Iteratively find the closest edge-pair snap, execute it, and repeat.
    Each execution exposes new boundary edges that become the next candidates.
    Returns (pos, adj, vertex_ring, faces, zip_log).
    zip_log: list of dicts, one per executed zip.
    """
    zip_log = []
    merged  = set()

    for step in range(max_zips):
        cands = find_edge_snap_candidates(pos, faces, adj, epsilon, parallel_thresh)
        # Filter out any involving already-merged vertices
        cands = [(d,e1,e2,ori,vp) for d,e1,e2,ori,vp in cands
                 if not (e1[0] in merged or e1[1] in merged or
                         e2[0] in merged or e2[1] in merged)]
        if not cands:
            break

        d, (u1,v1), (u2,v2), orient, vertex_pairs = cands[0]
        (a1,a2),(b1,b2) = vertex_pairs

        if verbose:
            r1a = vertex_ring.get(u1,-1); r1b = vertex_ring.get(v1,-1)
            r2a = vertex_ring.get(u2,-1); r2b = vertex_ring.get(v2,-1)
            print(f"  [zip {step+1}] ({u1}r{r1a},{v1}r{r1b}) <-> ({u2}r{r2a},{v2}r{r2b})"
                  f"  {orient}  mid_dist={d:.4f}")

        pos, adj, vertex_ring, faces, (a1,a2), (b1,b2) = execute_edge_snap(
            u1, v1, u2, v2, orient, pos, adj, vertex_ring, faces)
        merged.update([a2, b2])

        zip_log.append({
            'step': step, 'mid_dist': round(d,5),
            'e1': (u1,v1), 'e2': (u2,v2), 'orient': orient,
            'merged_pairs': [(a1,a2), (b1,b2)],  # (kept, removed) for each merge
        })

    return pos, adj, vertex_ring, faces, zip_log

# ── Physics relaxer with zip force ───────────────────────────────────────────
def relax(pos, adj, faces, free_verts=None, steps=10000,
          zip_pairs=None,
          converge_window=500, converge_tol=1e-5):
    """
    Relax until convergence.
    zip_pairs: list of (va, vb) vertex pairs that should attract (zip force).
               For these pairs the normal repulsion is suppressed and replaced
               by a stronger attraction proportional to distance.
    """
    pos = pos.copy()
    N   = len(pos)
    if free_verts is None:
        free_verts = list(range(N))
    free_arr = np.zeros(N, dtype=bool)
    for v in free_verts: free_arr[v] = True

    adj_set = {(min(u,v), max(u,v)) for v in range(N) for u in adj[v]}
    adj_mat = np.zeros((N, N), dtype=bool)
    for u,v in adj_set: adj_mat[u,v] = adj_mat[v,u] = True

    adj_arr = np.array(list(adj_set)) if adj_set else np.zeros((0,2),dtype=int)
    if len(adj_arr):
        U, V = adj_arr[:,0], adj_arr[:,1]
        fm   = free_arr[U] | free_arr[V]
        Uf, Vf = U[fm], V[fm]
    else:
        Uf = Vf = np.array([], dtype=int)

    # Zip pair arrays
    if zip_pairs:
        za = np.array([a for a,b in zip_pairs], dtype=int)
        zb = np.array([b for a,b in zip_pairs], dtype=int)
    else:
        za = zb = np.array([], dtype=int)

    Ur = Vr = np.array([], dtype=int)

    def rebuild():
        nonlocal Ur, Vr
        spread = np.max(np.linalg.norm(pos - pos.mean(axis=0), axis=1)) if N > 0 else 0
        if spread < 0.5: Ur = Vr = np.array([], dtype=int); return
        tree  = cKDTree(pos)
        pairs = tree.query_pairs(r=1.5, output_type='ndarray')
        if not len(pairs): Ur = Vr = np.array([], dtype=int); return
        pa, pb = pairs[:,0], pairs[:,1]
        if len(pa) > 5000:
            idx = np.random.choice(len(pa), 5000, replace=False)
            pa, pb = pa[idx], pb[idx]
        keep = ~adj_mat[pa,pb] & (free_arr[pa] | free_arr[pb])
        # Don't repel zip partners
        if len(za):
            zip_mat = np.zeros((N,N), dtype=bool)
            zip_mat[za,zb] = zip_mat[zb,za] = True
            keep &= ~zip_mat[pa,pb]
        if not keep.any(): Ur = Vr = np.array([], dtype=int); return
        Ur, Vr = pa[keep], pb[keep]

    def cur_std():
        if not len(adj_arr): return 0.0
        lens = np.linalg.norm(pos[V] - pos[U], axis=1)
        return float(np.std(lens))

    rebuild()
    std_history = []
    max_steps = steps if steps is not None else 10_000_000

    for step in range(max_steps):
        delta = np.zeros((N, 3))

        # Spring force on edges
        if len(Uf):
            d  = pos[Vf] - pos[Uf]
            dn = np.linalg.norm(d, axis=1, keepdims=True).clip(1e-12)
            np.add.at(delta, Uf, -d*(1-dn)*ATTRACT)
            np.add.at(delta, Vf,  d*(1-dn)*ATTRACT)

        # Repulsion for non-adjacent, non-zip pairs
        if len(Ur):
            dr  = pos[Vr] - pos[Ur]
            drn = np.linalg.norm(dr, axis=1, keepdims=True).clip(1e-12)
            cl  = (drn < 1.0).flatten()
            if cl.any():
                np.add.at(delta, Vr[cl],  dr[cl]*(1-drn[cl])*REPEL)
                np.add.at(delta, Ur[cl], -dr[cl]*(1-drn[cl])*REPEL)

        # Zip force: attract committed pairs toward each other
        if len(za):
            dz  = pos[zb] - pos[za]
            dzn = np.linalg.norm(dz, axis=1, keepdims=True).clip(1e-12)
            # Pull proportional to distance (simple attractive spring, no equilibrium)
            np.add.at(delta, za,  dz * ZIP_STRENGTH)
            np.add.at(delta, zb, -dz * ZIP_STRENGTH)

        delta[~free_arr] = 0.0
        pos += delta
        if not np.isfinite(pos).all():
            return None

        if step % 100 == 99:
            rebuild()
            std_history.append(cur_std())
            if len(std_history) >= converge_window // 100:
                window = std_history[-(converge_window//100):]
                if window[0] - window[-1] < converge_tol and window[-1] < 0.5:
                    return pos

    return pos

# ── Collision detection ───────────────────────────────────────────────────────
_PAIR_CACHE = {}

def _precompute_pairs(faces, vertex_ring, max_ring):
    faces_arr = np.array(faces); F = len(faces_arr)
    if F == 0: return {}
    mv = int(faces_arr.max())+1
    mem = np.zeros((F, mv), dtype=np.uint8)
    mem[np.arange(F)[:,None], faces_arr] = 1
    shares = (mem @ mem.T) > 0
    vr_arr = np.array([vertex_ring.get(v,0) for v in range(max(vertex_ring)+1)])
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
    if not faces: return False
    faces_arr = np.array(faces)
    mr  = max(vertex_ring.values())
    key = (id(faces), mr)
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
RING_COLORS = [
    (255,255,255),(68,1,84),(71,44,122),(59,81,139),(44,113,142),(33,144,141),
    (39,173,129),(92,200,100),(170,220,50),(234,229,26),(253,231,37),
]
def _col(r): return RING_COLORS[min(r, len(RING_COLORS)-1)]

def save_ply(path, pos, faces, vertex_ring):
    active = {v for v,rv in vertex_ring.items() if rv >= 0}
    # Remap vertices
    verts = sorted(active)
    remap = {v:i for i,v in enumerate(verts)}
    with open(path,'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(verts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        valid_faces = [face for face in faces
                       if all(v in active for v in face) and len(set(face))==3]
        f.write(f"element face {len(valid_faces)}\n")
        f.write("property list uchar int vertex_indices\nend_header\n")
        for v in verts:
            p = pos[v]; r,g,b = _col(vertex_ring.get(v,0))
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {r} {g} {b}\n")
        for a,b,c in valid_faces:
            f.write(f"3 {remap[a]} {remap[b]} {remap[c]}\n")

# ── Boundary topology summary ─────────────────────────────────────────────────
def boundary_summary(faces, vertex_ring):
    bdy = get_boundary_edges(faces)
    adj_bdy = defaultdict(set)
    for a,b in bdy:
        adj_bdy[a].add(b); adj_bdy[b].add(a)
    visited = set(); components = []
    for start in sorted(adj_bdy):
        if start in visited: continue
        comp=[]; queue=[start]
        while queue:
            v=queue.pop()
            if v in visited: continue
            visited.add(v); comp.append(v); queue.extend(adj_bdy[v]-visited)
        components.append(comp)

    active = sum(1 for rv in vertex_ring.values() if rv >= 0)
    ec = defaultdict(int)
    for a,b,c in faces:
        for e in [(min(a,b),max(a,b)),(min(b,c),max(b,c)),(min(a,c),max(a,c))]:
            ec[e] += 1
    nm = sum(1 for cnt in ec.values() if cnt > 2)
    V = active; E = len(ec); F = len(faces)
    chi = V - E + F
    b   = len(components)
    g   = (2 - chi - b) / 2
    return {
        'V': V, 'E': E, 'F': F, 'chi': chi,
        'boundary_components': b,
        'component_sizes': sorted([len(c) for c in components], reverse=True),
        'non_manifold_edges': nm,
        'genus': g,
    }

# ── Main loop ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rings', type=int, default=MAX_RINGS)
    parser.add_argument('--seeds', type=int, default=NUM_SEEDS)
    parser.add_argument('--epsilon', type=float, default=ZIP_EPSILON)
    parser.add_argument('--no-zip-force', action='store_true',
                        help='Disable zip force in relaxer (zip only at end of ring)')
    args = parser.parse_args()

    print("=" * 60)
    print("  {3,7} edge-zip mesh grower")
    print(f"  rings={args.rings}  seeds={args.seeds}  epsilon={args.epsilon:.2f}")
    print("=" * 60)

    best_ever_ring = 0
    all_seed_logs  = []

    for seed_idx in range(args.seeds):
        rng = np.random.default_rng(seed_idx * 17 + 42)
        print(f"\n-- Seed {seed_idx+1}/{args.seeds} --")

        adj, vr, faces, boundary = build_combinatorial(1)
        pos = initial_positions(vr, boundary, rng)
        pos = relax(pos, adj, faces, steps=RELAX_STEPS)
        if pos is None: continue

        valid_ring = 1; seed_log = []

        for r in range(2, args.rings+1):
            t0 = time.time()
            adj_new, vr_new, faces_new, bdy_new = build_combinatorial(r)
            N_new = max(vr_new)+1
            pos_new = np.zeros((N_new, 3))
            for v in range(len(pos)): pos_new[v] = pos[v]
            new_verts = [v for v,rv in vr_new.items() if rv == r]
            for v in new_verts:
                nbrs = [u for u in adj_new[v] if u < len(pos)]
                if nbrs:
                    pos_new[v] = np.mean([pos[u] for u in nbrs], axis=0) + rng.uniform(-0.05,0.05,3)
                else:
                    angle = 2*math.pi*v/max(len(new_verts),1)
                    pos_new[v] = [r*math.cos(angle), r*math.sin(angle), rng.uniform(-0.1,0.1)]

            pos_new = relax(pos_new, adj_new, faces_new, steps=RELAX_STEPS)
            if pos_new is None:
                print(f"  Ring {r}: NaN, stopping"); break

            collides = has_collision(pos_new, faces_new, {r}, vr_new)
            std = edge_std(pos_new, adj_new)
            elapsed = time.time()-t0

            # Boundary topology before any zip
            topo = boundary_summary(faces_new, vr_new)
            print(f"  Ring {r}: {'COLLISION' if collides else 'ok'}  "
                  f"std={std:.4f}  N={N_new}  t={elapsed:.1f}s")
            print(f"    boundary: {topo['boundary_components']} component(s) "
                  f"{topo['component_sizes']}  chi={topo['chi']}  "
                  f"genus={topo['genus']:.1f}  nm={topo['non_manifold_edges']}")

            if collides:
                print(f"    attempting zip cascade (epsilon={args.epsilon:.2f})...")

                pos_z = pos_new; adj_z = adj_new
                vr_z  = dict(vr_new); faces_z = list(faces_new)

                if not args.no_zip_force:
                    # Phase 1: zip-force pre-relax.
                    # Find candidates, extract their vertex pairs as attractive springs,
                    # run the relaxer so physics pulls the seam closed, then re-detect.
                    cands0 = find_edge_snap_candidates(
                        pos_z, faces_z, adj_z, args.epsilon, ZIP_PARALLEL)
                    if cands0:
                        zip_pairs0 = [vp for _,_,_,_,vps in cands0 for vp in vps]
                        print(f"    zip-force pre-relax: {len(cands0)} candidate edges, "
                              f"{len(zip_pairs0)} vertex pairs attracted...")
                        pos_pre = relax(pos_z, adj_z, faces_z,
                                        zip_pairs=zip_pairs0, steps=RELAX_STEPS)
                        if pos_pre is not None:
                            pos_z = pos_pre
                            std_pre = edge_std(pos_z, adj_z)
                            cands1 = find_edge_snap_candidates(
                                pos_z, faces_z, adj_z, args.epsilon, ZIP_PARALLEL)
                            print(f"    after pre-relax: std={std_pre:.4f}, "
                                  f"candidates {len(cands0)} -> {len(cands1)}")

                # Phase 2: discrete zip cascade
                pos_z, adj_z, vr_z, faces_z, zlog = zip_cascade(
                    pos_z, adj_z, vr_z, faces_z,
                    args.epsilon, ZIP_PARALLEL, verbose=True)

                if zlog:
                    # Phase 3: post-zip relax with zip forces on merged pairs
                    active_merges = [vp for entry in zlog for vp in entry['merged_pairs']
                                     if vr_z.get(vp[0],-1) >= 0]
                    if active_merges and not args.no_zip_force:
                        print(f"    post-zip relax with {len(active_merges)} zip pairs...")
                        pos_post = relax(pos_z, adj_z, faces_z,
                                         zip_pairs=active_merges, steps=RELAX_STEPS)
                        if pos_post is not None:
                            pos_z = pos_post

                    topo_z = boundary_summary(faces_z, vr_z)
                    std_z  = edge_std(pos_z, adj_z)
                    col_z  = has_collision(pos_z, faces_z, {r}, vr_z)
                    print(f"    after {len(zlog)} zips: std={std_z:.4f}  "
                          f"collision={'YES' if col_z else 'NO'}")
                    print(f"    boundary: {topo_z['boundary_components']} component(s) "
                          f"{topo_z['component_sizes']}  chi={topo_z['chi']}  "
                          f"genus={topo_z['genus']:.1f}  nm={topo_z['non_manifold_edges']}")

                    save_ply(f"zip_seed{seed_idx+1:02d}_ring{r:02d}_zips{len(zlog)}.ply",
                             pos_z, faces_z, vr_z)
                    seed_log.append({'ring': r, 'zips': len(zlog),
                                     'topo': topo_z, 'collision_resolved': not col_z})

                print(f"    stopping at ring {r} (collision)")
                break

            # No collision: re-relax, then try opportunistic zip
            pos_r2 = relax(pos_new, adj_new, faces_new, steps=RELAX_STEPS)
            if pos_r2 is not None:
                std2 = edge_std(pos_r2, adj_new)
                pos_new = pos_r2
                print(f"    re-relax: std {std:.4f} -> {std2:.4f}")

            # Opportunistic zip on non-colliding ring (should find nothing)
            cands = find_edge_snap_candidates(pos_new, faces_new, adj_new,
                                              args.epsilon, ZIP_PARALLEL)
            if cands:
                print(f"    {len(cands)} zip candidate(s) on non-collision ring — unexpected")

            adj, vr, faces, boundary = adj_new, vr_new, faces_new, bdy_new
            pos = pos_new; valid_ring = r

            save_ply(f"zip_seed{seed_idx+1:02d}_ring{r:02d}_std{std2:.4f}.ply",
                     pos, faces, vr)
            print(f"    Ring {r} valid OK  (best so far: {valid_ring})")
            if valid_ring > best_ever_ring:
                best_ever_ring = valid_ring

        all_seed_logs.append({'seed': seed_idx+1, 'valid_ring': valid_ring,
                              'events': seed_log})

    with open('zip_log.json','w') as f: json.dump(all_seed_logs, f, indent=2)
    print(f"\n{'='*60}")
    print(f"Best ring across {args.seeds} seeds: {best_ever_ring}")
    print("Log saved: zip_log.json")
