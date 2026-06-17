"""
Hyperbolic Triangle Mesh — GNN Embedder
========================================
Goal: find the largest self-intersection-free embedding of the {3,7}
hyperbolic tiling (7 equilateral triangles at every interior vertex) in R³.

Pipeline
--------
1. COMBINATORICS  — build the abstract {3,7} graph ring by ring (no geometry).
2. PHYSICS        — L-BFGS spring relaxer; used ONLY to generate training data.
3. GNN            — replaces the relaxer at inference time; predicts final
                    embedded positions for ring k given rings 0..k-1.
4. COLLISION TEST — triangle-triangle intersection test (ported from Rust).
5. SEARCH         — escalating backtrack on collision:
                      Level 0: retry ring k with a new GNN sample (fast)
                      Level 1: re-embed ring k-1 and redo ring k
                      Level 2: restart the whole mesh
6. PARETO LOG     — tracks (ring_depth, edge_std, valid) across all attempts.
7. ONLINE LOOP    — physics solutions from successful backtracks become new
                    training data; GNN is periodically retrained.

Backend
-------
Set USE_DGL = True  for DGL (faster batching, built-in layers).
Set USE_DGL = False for pure PyTorch (no extra dependencies).
Weights in best_model.pt are identical either way.

Dependencies: torch numpy scipy
Optional:     dgl  (pip install dgl -f https://data.dgl.ai/wheels/torch-X.Y/repo.html)
"""

import math, random, time, json, copy
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import cKDTree

# Crystal neighborhood scoring (Z[phi]^3 and other lattices).
# Requires crystal_neighborhoods_ascii.py in the same directory.
try:
    from crystal_neighborhoods_ascii import (
        crystallinity_score, align_to_lattice, crystal_snap_forces,
        compute_neighborhood_features, neighborhood_feature_dim,
        full_relaxer_step, LATTICE_DB,
    )
    CRYSTAL_AVAILABLE = True
except ImportError:
    CRYSTAL_AVAILABLE = False
    print("WARNING: crystal_neighborhoods_ascii.py not found. "
          "Crystal forces and features disabled.")

# ── Backend ────────────────────────────────────────────────────────────────────
USE_DGL = False
if USE_DGL:
    try:
        import dgl
        from dgl.nn import SAGEConv as DGLSAGEConv
        print("DGL backend active.")
    except ImportError:
        print("WARNING: USE_DGL=True but DGL not found — falling back to pure PyTorch.")
        USE_DGL = False

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Combinatorics  — {3,7} disk, topology only
# ═══════════════════════════════════════════════════════════════════════════════
#
# Rules:
#   • Every interior vertex (ring < current max) has degree exactly 7.
#   • Every face is a triangle.
#   • We grow ring by ring from a central vertex (ring 0).
#
# When adding ring r, each boundary vertex v needs (7 - deg(v)) new neighbours.
# Of those, exactly 2 are "shared" with adjacent boundary vertices;
# the rest are "exclusive" to v.  This gives ring sizes 1,7,21,56,147,385,...
# (OEIS A001354, matching the Rust implementation's recurrence).
#
# Verified: χ = V - E + F = 1 (disk) and all interior vertices have degree 7.

def build_combinatorial(num_rings: int) -> tuple:
    """
    Build the {3,7} tiling as a pure graph (integers and sets, no coordinates).

    Returns
    -------
    adj         : dict[int, set[int]]
    vertex_ring : dict[int, int]
    faces       : list[(int,int,int)]
    boundary    : dict[int, list[int]]  ordered boundary vertices per ring
    """
    adj = defaultdict(set)
    vertex_ring: dict[int,int] = {}
    faces: list[tuple] = []
    _id = [0]

    def new_v(r):
        v = _id[0]; _id[0] += 1
        vertex_ring[v] = r
        return v

    seen_faces: set[frozenset] = set()
    def add_face(a, b, c):
        key = frozenset([a, b, c])
        if key in seen_faces: return
        seen_faces.add(key)
        faces.append((a, b, c))
        for u, w in [(a,b),(b,c),(a,c)]:
            adj[u].add(w); adj[w].add(u)

    center = new_v(0)
    ring1  = [new_v(1) for _ in range(7)]
    for i in range(7):
        add_face(center, ring1[i], ring1[(i+1) % 7])
    boundary = {0: [center], 1: ring1}

    for r in range(2, num_rings + 1):
        prev    = boundary[r-1]
        n       = len(prev)
        needed  = [7 - len(adj[v]) for v in prev]
        assert all(k >= 2 for k in needed), \
            f"Ring {r-1}: some vertex needs <2 new neighbours — combinatorics bug"

        shared    = [new_v(r) for _ in range(n)]
        exclusive = [[new_v(r) for _ in range(needed[i] - 2)] for i in range(n)]

        for i in range(n):
            v      = prev[i]
            v_next = prev[(i+1) % n]
            fan    = [shared[(i-1) % n]] + exclusive[i] + [shared[i]]
            for j in range(len(fan) - 1):
                add_face(v, fan[j], fan[j+1])
            add_face(v, v_next, shared[i])

        new_bdy = []
        for i in range(n):
            new_bdy.append(shared[(i-1) % n])
            new_bdy.extend(exclusive[i])
        boundary[r] = new_bdy

    return adj, vertex_ring, faces, boundary


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Geometry helpers
# ═══════════════════════════════════════════════════════════════════════════════

EDGE_LENGTH = 1.0


def initial_positions(vertex_ring: dict, boundary: dict,
                      z_noise: float = 0.1, rng=None) -> np.ndarray:
    """
    Concentric-ring flat layout with small random z perturbation.

    The Rust code adds uniform noise in [-0.1, 0.1] to z (main.rs line 195).
    This breaks the flat symmetry so L-BFGS can find a buckled solution.
    """
    if rng is None:
        rng = np.random.default_rng()
    N   = max(vertex_ring) + 1
    pos = np.zeros((N, 3))
    for r, verts in boundary.items():
        nv = len(verts)
        for i, v in enumerate(verts):
            angle = 2.0 * math.pi * i / max(nv, 1)
            radius = r * EDGE_LENGTH
            pos[v] = [radius * math.cos(angle),
                      radius * math.sin(angle),
                      rng.uniform(-z_noise, z_noise)]
    return pos


def relax(pos: np.ndarray, adj: dict,
          free_verts=None, steps: int = 600,
          attract: float = 0.1,
          repel: float = 0.6,
          crystal_align: float = 0.0,
          crystal_snap: float = 0.0,
          crystal_threshold: float = 0.25,
          lattice_key: str = 'zphi',
          rep_cutoff: float = 1.5,
          rebuild_rep_every: int = 100,
          vertex_ring: dict = None) -> np.ndarray:
    """
    Vectorized iterative relaxer matching Rust main.rs physics.

    Two base forces applied simultaneously to all vertices each step:
      - Attraction: (1 - dist) * attract  along connected edges  (move_points)
      - Repulsion:  (1 - dist) * repel    for non-adjacent pairs closer than 1.0
                    (move_duals via collision_groups in Rust)

    Optional crystal forces (requires crystal_neighborhoods_ascii.py):
      - Alignment: pulls neighbors toward ideal Z[phi]^3 lattice positions,
                   gated by crystallinity score (turns on near convergence).
      - Snap:      pulls spatially close, crystalline vertex pairs together,
                   enabling distant stitching along lattice seams.

    steps: number of iterations.
    free_verts: if given, only those vertices are updated (others frozen).
    """
    pos      = pos.copy()
    N        = len(pos)
    if free_verts is None:
        free_verts = list(range(N))
    # Build adjacency as a boolean matrix for fast vectorized lookup
    adj_set  = {(min(u,v), max(u,v)) for v in range(N) for u in adj[v]}
    adj_mat  = np.zeros((N, N), dtype=bool)
    for u, v in adj_set:
        adj_mat[u, v] = True
        adj_mat[v, u] = True

    adj_arr  = np.array(list(adj_set))
    U, V     = adj_arr[:, 0], adj_arr[:, 1]
    # Vectorized free_mask: no Python loop
    free_arr = np.zeros(N, dtype=bool)
    for v in free_verts:
        free_arr[v] = True
    free_mask = free_arr[U] | free_arr[V]
    Uf, Vf   = U[free_mask], V[free_mask]

    CUTOFF = rep_cutoff
    Ur = Vr = np.array([], dtype=int)

    def rebuild_rep():
        nonlocal Ur, Vr
        # Skip repulsion while mesh is still collapsed — query_pairs is O(pairs)
        # and catastrophically slow when all vertices are near the origin.
        spread = np.max(np.linalg.norm(pos - pos.mean(axis=0), axis=1))
        if spread < 0.5:
            Ur = Vr = np.array([], dtype=int)
            return
        tree  = cKDTree(pos)
        pairs = tree.query_pairs(r=CUTOFF, output_type='ndarray')
        if not len(pairs):
            Ur = Vr = np.array([], dtype=int)
            return
        pa, pb = pairs[:, 0], pairs[:, 1]
        # Cap pair count to avoid O(N²) work on dense configurations
        if len(pa) > 5000:
            idx  = np.random.choice(len(pa), 5000, replace=False)
            pa, pb = pa[idx], pb[idx]
        not_adj  = ~adj_mat[pa, pb]
        has_free = free_arr[pa] | free_arr[pb]
        keep     = not_adj & has_free
        if not keep.any():
            Ur = Vr = np.array([], dtype=int)
            return
        Ur, Vr = pa[keep], pb[keep]

    rebuild_rep()

    for step in range(steps):
        delta = np.zeros((N, 3))

        # -- Attraction along edges
        d_vec  = pos[Vf] - pos[Uf]
        dist   = np.linalg.norm(d_vec, axis=1, keepdims=True).clip(min=1e-12)
        scalar = (1.0 - dist) * attract
        np.add.at(delta, Uf, -d_vec * scalar)
        np.add.at(delta, Vf,  d_vec * scalar)

        # -- Repulsion between non-adjacent nearby pairs
        if len(Ur):
            dr_vec = pos[Vr] - pos[Ur]
            dr     = np.linalg.norm(dr_vec, axis=1, keepdims=True).clip(min=1e-12)
            close  = (dr < 1.0).flatten()
            if close.any():
                sc = (1.0 - dr[close]) * repel
                np.add.at(delta, Vr[close],  dr_vec[close] * sc)
                np.add.at(delta, Ur[close], -dr_vec[close] * sc)

        # -- Crystal forces (alignment + snap), gated by crystallinity score
        if CRYSTAL_AVAILABLE and (crystal_align > 0.0 or crystal_snap > 0.0):
            vr = vertex_ring if vertex_ring is not None else {}
            cf = crystal_snap_forces(
                pos, adj, vr,
                lattice_key=lattice_key,
                score_threshold=crystal_threshold,
                snap_strength=crystal_snap,
                align_strength=crystal_align,
            )
            cf[~free_arr] = 0.0   # zero fixed verts (vectorized)
            delta += cf

        # -- Apply: only free vertices move (vectorized mask)
        delta[~free_arr] = 0.0
        pos += delta

        if step % rebuild_rep_every == rebuild_rep_every - 1:
            rebuild_rep()

    return pos


def edge_std(pos: np.ndarray, adj: dict) -> float:
    """Standard deviation of edge lengths (0 = perfectly equilateral)."""
    lengths = [np.linalg.norm(pos[v] - pos[u])
               for v in adj for u in adj[v] if u > v]
    return float(np.std(lengths)) if lengths else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: Collision detection  (ported from Rust main.rs)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Uses Möller–Trumbore ray-triangle intersection.
# Two triangles collide if any edge of one pierces the other.
# Triangles sharing a vertex are skipped (they meet at an edge, not a crossing).
# A spatial grid prunes the O(F²) search to only nearby triangle pairs.

# ── Vectorized Möller–Trumbore (batch) ────────────────────────────────────────
def _batch_ray_triangle(origs: np.ndarray, dirs: np.ndarray,
                        v0s: np.ndarray, v1s: np.ndarray,
                        v2s: np.ndarray) -> np.ndarray:
    """Batched Möller–Trumbore: P rays vs P triangles → bool array (P,)."""
    e1 = v1s - v0s;  e2 = v2s - v0s
    h  = np.cross(dirs, e2)
    a  = np.einsum('ij,ij->i', e1, h)
    ok = np.abs(a) > 1e-10
    f  = np.where(ok, 1.0 / np.where(ok, a, 1.0), 0.0)
    s  = origs - v0s
    u  = f * np.einsum('ij,ij->i', s, h)
    ok &= (u >= 0.0) & (u <= 1.0)
    q  = np.cross(s, e1)
    v  = f * np.einsum('ij,ij->i', dirs, q)
    ok &= (v >= 0.0) & (u + v <= 1.0)
    t  = f * np.einsum('ij,ij->i', e2, q)
    return ok & (t > 1e-10)


# Cache of precomputed non-adjacent pair indices, keyed by num_rings.
# Topology is deterministic so we compute once and reuse.
_COLLISION_PAIR_CACHE: dict = {}


def _precompute_pairs(faces: list, vertex_ring: dict, max_ring: int) -> dict:
    """
    For each ring r in 1..max_ring, precompute arrays (II, JJ) of face-index
    pairs where face II contains a ring-r vertex and faces II,JJ share no vertex.
    Fully vectorized — no Python loops over face pairs.
    """
    faces_arr = np.array(faces)          # (F, 3)
    F         = len(faces_arr)
    max_vert  = int(faces_arr.max()) + 1

    # Build (F, V) boolean membership matrix: membership[fi, v] = v in face fi
    membership = np.zeros((F, max_vert), dtype=np.uint8)
    membership[np.arange(F)[:, None], faces_arr] = 1

    # shares[i,j] = True iff faces i and j share at least one vertex
    shares = (membership @ membership.T) > 0  # (F, F)

    # ring of each face vertex
    vr_arr     = np.array([vertex_ring[v] for v in range(max(vertex_ring) + 1)])
    face_vring = vr_arr[faces_arr]       # (F, 3)
    all_idx    = np.arange(F, dtype=np.int32)

    result = {}
    for r in range(1, max_ring + 1):
        new_mask = np.any(face_vring == r, axis=1)
        new_idx  = np.where(new_mask)[0]
        II_list, JJ_list = [], []
        for fi in new_idx:
            valid      = ~shares[fi]
            valid[fi]  = False
            jj = all_idx[valid]
            II_list.append(np.full(len(jj), fi, dtype=np.int32))
            JJ_list.append(jj)
        result[r] = (
            np.concatenate(II_list) if II_list else np.array([], dtype=np.int32),
            np.concatenate(JJ_list) if JJ_list else np.array([], dtype=np.int32),
        )
    return result


def has_collision(pos: np.ndarray, faces: list,
                  check_rings: set = None,
                  vertex_ring: dict = None) -> bool:
    """
    Return True if any two non-adjacent triangles intersect.
    Uses fully-vectorized batched Möller–Trumbore — ~100x faster than the
    previous Python loop over pairs.
    Pair indices are precomputed per mesh topology and cached.
    """
    global _COLLISION_PAIR_CACHE

    if not faces:
        return False

    faces_arr = np.array(faces)

    if check_rings and vertex_ring:
        # Use cached precomputed pairs
        max_ring = max(vertex_ring.values())
        cache_key = (len(faces), max_ring)
        if cache_key not in _COLLISION_PAIR_CACHE:
            _COLLISION_PAIR_CACHE[cache_key] = _precompute_pairs(
                faces, vertex_ring, max_ring)
        pairs_cache = _COLLISION_PAIR_CACHE[cache_key]

        for r in check_rings:
            if r not in pairs_cache:
                continue
            II, JJ = pairs_cache[r]
            if len(II) == 0:
                continue
            t1 = pos[faces_arr[II]]   # (P,3,3)
            t2 = pos[faces_arr[JJ]]   # (P,3,3)
            for src, tgt in ((t2, t1), (t1, t2)):
                for ei, ej in ((0,1),(0,2),(1,2)):
                    p = src[:, ei, :]; q = src[:, ej, :]; d = q - p
                    if np.any(_batch_ray_triangle(p,  d, tgt[:,0,:], tgt[:,1,:], tgt[:,2,:]) &
                              _batch_ray_triangle(q, -d, tgt[:,0,:], tgt[:,1,:], tgt[:,2,:])):
                        return True
        return False

    else:
        # Full check: spatial grid to limit pairs, then vectorized test
        tris     = pos[faces_arr]                # (F,3,3)
        centroids = tris.mean(axis=1)             # (F,3)
        cell_size = 2.0
        grid: dict = defaultdict(list)
        for i, c in enumerate(centroids):
            grid[tuple((c / cell_size).astype(int))].append(i)

        ii_list, jj_list = [], []
        seen: set = set()
        for i in range(len(faces)):
            c = tuple((centroids[i] / cell_size).astype(int))
            nbrs = []
            for dc in np.ndindex(3, 3, 3):
                nc = (c[0]+dc[0]-1, c[1]+dc[1]-1, c[2]+dc[2]-1)
                nbrs.extend(grid.get(nc, []))
            fa = set(faces_arr[i])
            for j in nbrs:
                if j <= i or (i,j) in seen: continue
                seen.add((i,j))
                if fa & set(faces_arr[j]): continue
                ii_list.append(i); jj_list.append(j)

        if not ii_list:
            return False
        II = np.array(ii_list, dtype=np.int32)
        JJ = np.array(jj_list, dtype=np.int32)
        t1 = pos[faces_arr[II]]; t2 = pos[faces_arr[JJ]]
        for src, tgt in ((t2, t1), (t1, t2)):
            for ei, ej in ((0,1),(0,2),(1,2)):
                p = src[:, ei, :]; q = src[:, ej, :]; d = q - p
                if np.any(_batch_ray_triangle(p,  d, tgt[:,0,:], tgt[:,1,:], tgt[:,2,:]) &
                          _batch_ray_triangle(q, -d, tgt[:,0,:], tgt[:,1,:], tgt[:,2,:])):
                    return True
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: Graph data  (dual-backend: DGL or pure PyTorch)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Node features (7-dimensional):
#   pos_xyz      (3) — 3D position
#   ring_norm    (1) — ring / max_ring
#   is_boundary  (1) — 1 if on the innermost ring being predicted FROM
#   deg_norm     (1) — degree / 7
#   ring_abs     (1) — ring index (unscaled, helps GNN sense depth)

def _node_features(pos, adj, vertex_ring, input_rings, predict_ring, verts):
    p   = np.array([pos[v]         for v in verts], dtype=np.float32)
    r   = np.array([vertex_ring[v] for v in verts], dtype=np.float32)
    d   = np.array([len(adj[v])    for v in verts], dtype=np.float32)
    bdy = (r == input_rings).astype(np.float32)
    mr  = max(predict_ring, 1)
    base = np.column_stack([p, r[:,None]/mr, bdy[:,None],
                             d[:,None]/7.0, r[:,None]]).astype(np.float32)

    if not CRYSTAL_AVAILABLE or not USE_CRYSTAL_FEATURES:
        return base

    # Append crystal neighborhood features for each vertex:
    # [zphi_score, fcc_score, sc_score, spoke_mean, spoke_std,
    #  pw_mean, pw_std, k2_size, align_rmsd, planarity]  (10 dims)
    crystal_feats = np.array([
        compute_neighborhood_features(v, pos, adj, vertex_ring)
        for v in verts
    ], dtype=np.float32)
    return np.column_stack([base, crystal_feats]).astype(np.float32)


# Feature dimensions:
#   base (7): pos_xyz, ring_norm, is_boundary, deg_norm, ring_abs
#   crystal (10, if enabled): zphi/fcc/sc scores, spoke stats,
#                             pairwise stats, k2 size, rmsd, planarity
_CRYSTAL_FEAT_DIM = 10  # neighborhood_feature_dim() from crystal_neighborhoods_ascii
USE_CRYSTAL_FEATURES = CRYSTAL_AVAILABLE  # set False to disable even if module present
IN_FEATS = 7 + (_CRYSTAL_FEAT_DIM if USE_CRYSTAL_FEATURES else 0)


class TorchMeshData:
    """Pure-PyTorch graph snapshot. Interface is the same as the DGL path."""
    __slots__ = ['node_feat','edge_index','mask','target_pos','num_nodes','valid']

    def __init__(self, pos, adj, vertex_ring, input_rings, predict_ring):
        verts = [v for v, r in vertex_ring.items() if r <= predict_ring]
        pred  = [v for v in verts if vertex_ring[v] == predict_ring]
        self.valid = bool(pred)
        if not self.valid: return

        v2l = {v: i for i, v in enumerate(verts)}
        N   = len(verts)
        self.num_nodes = N

        src, dst = [], []
        for v in verts:
            for u in adj[v]:
                if u in v2l:
                    src.append(v2l[v]); dst.append(v2l[u])
        self.edge_index = torch.tensor([src, dst], dtype=torch.long)

        feat = _node_features(pos, adj, vertex_ring, input_rings, predict_ring, verts)
        self.node_feat  = torch.from_numpy(feat)

        self.mask = torch.zeros(N, dtype=torch.bool)
        for v in pred: self.mask[v2l[v]] = True

        self.target_pos = torch.from_numpy(
            np.array([pos[v] for v in pred], dtype=np.float32))

    def to(self, device):
        self.node_feat  = self.node_feat.to(device)
        self.edge_index = self.edge_index.to(device)
        self.mask       = self.mask.to(device)
        self.target_pos = self.target_pos.to(device)
        return self


def make_dgl_data(pos, adj, vertex_ring, input_rings, predict_ring):
    verts = [v for v, r in vertex_ring.items() if r <= predict_ring]
    pred  = [v for v in verts if vertex_ring[v] == predict_ring]
    if not pred: return None
    v2l   = {v: i for i, v in enumerate(verts)}
    N     = len(verts)
    src, dst = [], []
    for v in verts:
        for u in adj[v]:
            if u in v2l: src.append(v2l[v]); dst.append(v2l[u])
    g = dgl.graph((src, dst), num_nodes=N)
    feat = _node_features(pos, adj, vertex_ring, input_rings, predict_ring, verts)
    g.ndata['feat']   = torch.from_numpy(feat)
    mask              = torch.zeros(N, dtype=torch.bool)
    target            = torch.zeros(N, 3)
    for v in pred:
        mask[v2l[v]]   = True
        target[v2l[v]] = torch.from_numpy(pos[v].astype(np.float32))
    g.ndata['mask']   = mask
    g.ndata['target'] = target
    return g


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: GNN model  (replaces the relaxer at inference time)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Architecture: 6-layer GraphSAGE-mean with residual connections and LayerNorm.
# Wider and deeper than before because the task is harder — we're asking it to
# predict final positions, not just refine physics positions.
#
# Training targets: physics-relaxed positions from valid (collision-free)
# embeddings. The GNN learns the mapping:
#   (embedded rings 0..k-1)  →  embedded positions of ring k
#
# At inference: no relaxer. The GNN's prediction IS the embedding.
# The collision detector then validates it.

class TorchSAGEConv(nn.Module):
    """GraphSAGE-mean via scatter_add — pure PyTorch, runs on CPU or CUDA."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W_self  = nn.Linear(in_dim, out_dim, bias=False)
        self.W_neigh = nn.Linear(in_dim, out_dim, bias=False)
        self.bias    = nn.Parameter(torch.zeros(out_dim))

    def forward(self, h, edge_index, num_nodes):
        src, dst = edge_index[0], edge_index[1]
        agg = torch.zeros(num_nodes, h.size(1), device=h.device, dtype=h.dtype)
        agg.scatter_add_(0, dst.unsqueeze(1).expand(-1, h.size(1)), h[src])
        cnt = torch.zeros(num_nodes, 1, device=h.device, dtype=h.dtype)
        cnt.scatter_add_(0, dst.unsqueeze(1),
                         torch.ones(dst.size(0), 1, device=h.device))
        return self.W_self(h) + self.W_neigh(agg / cnt.clamp(min=1)) + self.bias


class DGLSAGEWrapper(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = DGLSAGEConv(in_dim, out_dim, 'mean')
    def forward(self, h, g, num_nodes=None):
        return self.conv(g, h)


class HyperbolicGNN(nn.Module):
    """
    Predicts 3D positions for the next ring given the current embedded mesh.

    Deeper and wider than the previous version because it now has to learn
    the full geometry, not just refine a physics solution.

    At inference: call predict_ring_positions() rather than forward() directly.
    """
    def __init__(self, in_feats=IN_FEATS, hidden=192, num_layers=6,
                 use_dgl=False):
        super().__init__()
        self.use_dgl    = use_dgl
        self.input_proj = nn.Linear(in_feats, hidden)

        Conv = DGLSAGEWrapper if use_dgl else TorchSAGEConv
        self.convs = nn.ModuleList([Conv(hidden, hidden) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(num_layers)])

        # Output head: predict absolute 3D positions
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),   nn.GELU(),
            nn.Linear(hidden, hidden//2), nn.GELU(),
            nn.Linear(hidden//2, 3),
        )

    def forward(self, data):
        if self.use_dgl:
            h, grp, N = data.ndata['feat'], data, data.num_nodes()
        else:
            h, grp, N = data.node_feat, data.edge_index, data.num_nodes

        h = F.gelu(self.input_proj(h))
        for conv, norm in zip(self.convs, self.norms):
            h = norm(h + F.gelu(conv(h, grp, N)))
        return self.head(h)   # (N, 3)

    @torch.no_grad()
    def predict_ring_positions(self, pos: np.ndarray, adj: dict,
                               vertex_ring: dict, predict_ring: int,
                               device: torch.device) -> np.ndarray:
        """
        Given the embedded mesh up to ring predict_ring-1, return predicted
        3D positions for ring predict_ring vertices (in vertex-index order).
        """
        self.eval()
        input_rings = predict_ring - 1
        if self.use_dgl:
            g = make_dgl_data(pos, adj, vertex_ring, input_rings, predict_ring)
            if g is None: return None
            g   = g.to(device)
            out = self(g).cpu().numpy()
            msk = g.ndata['mask'].cpu().numpy().astype(bool)
        else:
            d = TorchMeshData(pos, adj, vertex_ring, input_rings, predict_ring)
            if not d.valid: return None
            d   = d.to(device)
            out = self(d).cpu().numpy()
            msk = d.mask.cpu().numpy()
        return out[msk]  # (M, 3) for M new vertices in order


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: Training
# ═══════════════════════════════════════════════════════════════════════════════

def relax_ring_by_ring(adj: dict, vertex_ring: dict, faces: list,
                       boundary: dict, rng=None,
                       steps_per_ring: int = 600,
                       collision_check: bool = True,
                       attract: float = 0.1,
                       repel: float = 0.6,
                       crystal_align: float = 0.0,
                       crystal_snap: float = 0.0,
                       crystal_threshold: float = 0.25,
                       lattice_key: str = 'zphi',
                       simultaneous: bool = True) -> Optional[np.ndarray]:
    """
    Relax the mesh, optionally ring-by-ring (freezing inner rings) or
    simultaneously (all vertices free at once, matching Rust main.rs).

    simultaneous=True  : all vertices relaxed together each step.
                         Finds the cooperative buckled basin that ring-by-ring
                         misses; within-ring edges stay near 1.0 instead of
                         compressing. Matches reference_mesh.ply quality.
    simultaneous=False : ring-by-ring freeze (legacy; faster but lower quality).

    crystal_align > 0  : pull neighbors toward Z[phi]^3 lattice ideal positions
                         (requires crystal_neighborhoods_ascii.py).
    crystal_snap  > 0  : attract crystalline vertices toward each other across
                         distant graph regions, enabling lattice stitching.

    Returns the position array, or None if a collision is found.
    """
    if rng is None:
        rng = np.random.default_rng()

    pos      = initial_positions(vertex_ring, boundary, rng=rng)
    max_ring = max(vertex_ring.values())
    relax_kw = dict(
        adj=adj, attract=attract, repel=repel,
        crystal_align=crystal_align, crystal_snap=crystal_snap,
        crystal_threshold=crystal_threshold, lattice_key=lattice_key,
        vertex_ring=vertex_ring,
    )

    if simultaneous:
        # -- Simultaneous: all vertices free, matching Rust physics
        # Phase 1: short warm-up with all verts to find the buckled basin
        pos = relax(pos, steps=steps_per_ring, **relax_kw)

        if collision_check and has_collision(pos, faces, vertex_ring=vertex_ring):
            return None

    else:
        # -- Legacy ring-by-ring (inner rings frozen)
        seed_verts = [v for v, r in vertex_ring.items() if r <= 1]
        pos = relax(pos, free_verts=seed_verts, steps=steps_per_ring, **relax_kw)

        for r in range(2, max_ring + 1):
            ring_verts = [v for v, rv in vertex_ring.items() if rv == r]
            pos = relax(pos, free_verts=ring_verts, steps=steps_per_ring, **relax_kw)

            if collision_check:
                if has_collision(pos, faces,
                                 check_rings={r}, vertex_ring=vertex_ring):
                    return None

    return pos


# Module-level physics defaults used by generate_training_sample.
# Overridden by __main__ globals when running as a script; importers can
# set these directly before calling generate_training_sample().
SIMULTANEOUS_RELAX = True
ATTRACT            = 0.1
REPEL              = 0.6
CRYSTAL_ALIGN      = 0.0   # disabled: too slow per step
CRYSTAL_SNAP       = 0.0   # disabled: too slow per step


def generate_training_sample(num_rings: int, relax_steps: int = 600,
                              rng=None) -> Optional[tuple]:
    """
    Generate one physics-relaxed valid embedding using ring-by-ring relaxation.
    Returns (pos, adj, vertex_ring, faces, boundary) or None if any ring collides.

    Ring-by-ring relaxation is used here because:
      - It produces much better-quality embeddings (closer to equilateral).
      - It catches collisions early (per ring) rather than discovering them
        at the end of an expensive full-mesh relaxation.
      - The GNN learns from cleaner targets, which directly improves its
        ability to predict equilateral embeddings.
    """
    import time as _time
    if rng is None:
        rng = np.random.default_rng()
    _t0 = _time.time()
    adj, vertex_ring, faces, boundary = build_combinatorial(num_rings)
    _t1 = _time.time()
    pos = relax_ring_by_ring(adj, vertex_ring, faces, boundary,
                             rng=rng, steps_per_ring=relax_steps,
                             simultaneous=SIMULTANEOUS_RELAX,
                             attract=ATTRACT, repel=REPEL,
                             crystal_align=CRYSTAL_ALIGN,
                             crystal_snap=CRYSTAL_SNAP)
    _t2 = _time.time()
    if pos is None:
        return None
    if _t2 - _t0 > 5.0:
        print(f"    [timing] rings={num_rings} build={_t1-_t0:.1f}s relax={_t2-_t1:.1f}s total={_t2-_t0:.1f}s", flush=True)
    return pos, adj, vertex_ring, faces, boundary


def build_dataset(num_samples: int = 200, max_rings: int = 4,
                  relax_steps: int = 400) -> list:
    """
    Generate physics-relaxed training samples (collision-free only).
    Each sample predicts the outermost ring given all inner rings.
    """
    import time
    samples   = []
    attempts  = 0
    rng       = np.random.default_rng()
    print(f"Building dataset (target {num_samples} valid samples, "
          f"max_rings={max_rings})...")
    t0 = time.time()
    raw_meshes = []   # keep raw (pos, adj, vertex_ring, faces) for saving

    while len(samples) < num_samples:
        attempts += 1
        num_rings = random.randint(2, max_rings)
        t1 = time.time()
        result    = generate_training_sample(num_rings, relax_steps, rng=rng)
        elapsed   = time.time() - t1
        if result is None:
            if attempts % 10 == 0:
                print(f"  attempt {attempts}: collision (rings={num_rings}, {elapsed:.1f}s)")
            continue
        pos, adj, vertex_ring, faces, boundary = result
        predict_ring = num_rings
        input_rings  = num_rings - 1
        if USE_DGL:
            g = make_dgl_data(pos, adj, vertex_ring, input_rings, predict_ring)
            if g: samples.append(g)
        else:
            d = TorchMeshData(pos, adj, vertex_ring, input_rings, predict_ring)
            if d.valid: samples.append(d)
        raw_meshes.append((pos.copy(), adj, vertex_ring, faces,
                           edge_std(pos, adj), num_rings))
        if len(samples) % 10 == 0 or len(samples) <= 5:
            rate = len(samples) / max(time.time()-t0, 1)
            eta  = (num_samples - len(samples)) / max(rate, 1e-6)
            print(f"  {len(samples):3d}/{num_samples}  "
                  f"attempts={attempts}  "
                  f"yield={100*len(samples)/attempts:.0f}%  "
                  f"last={elapsed:.1f}s  "
                  f"ETA={eta/60:.1f}min")
    total = time.time() - t0
    print(f"Dataset ready: {len(samples)} samples from {attempts} attempts "
          f"in {total/60:.1f}min.\n")
    return samples, raw_meshes



def _combined_loss(out, mask, target_pos, node_feat,
                   data, device, use_dgl=False,
                   equilateral_weight=0.3):
    """
    Loss = MSE(predicted positions, physics targets)
           + equilateral_weight * edge_length_variance

    The equilateral term directly penalises the GNN for predicting positions
    that result in non-uniform edge lengths — which is exactly what we want
    to minimise. It is differentiable because it operates on the predicted
    positions directly.

    We compute it only for edges where BOTH endpoints are predicted (new ring),
    because those are the edges the GNN fully controls. Edges between the
    new ring and the fixed inner rings are also included since their lengths
    depend on the predicted positions.
    """
    # Position MSE
    pos_loss = F.mse_loss(out[mask], target_pos)

    # Edge length variance over edges incident to predicted vertices
    # node_feat[:, 0:3] are the input positions for fixed vertices;
    # for masked (new) vertices we use the predicted positions.
    pred_pos = node_feat[:, :3].clone().detach()
    pred_pos[mask] = out[mask]  # substitute predictions

    if use_dgl:
        src, dst = data.edges()
        src, dst = src.to(device), dst.to(device)
    else:
        src, dst = data.edge_index[0], data.edge_index[1]

    # Only edges where at least one endpoint is a predicted vertex
    incident = mask[src] | mask[dst]
    if incident.sum() < 2:
        return pos_loss

    s = src[incident]
    d_idx = dst[incident]
    edge_vecs = pred_pos[d_idx] - pred_pos[s]
    edge_lens  = edge_vecs.norm(dim=1)
    # Penalise variance of edge lengths (target = all equal = EDGE_LENGTH)
    eq_loss = edge_lens.var()

    return pos_loss + equilateral_weight * eq_loss


def train_model(model: HyperbolicGNN, dataset: list,
                epochs: int = 60, lr: float = 1e-3,
                device: torch.device = None) -> dict:
    """Train (or fine-tune) the GNN on a dataset of TorchMeshData / DGL graphs."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    random.shuffle(dataset)
    split     = int(0.85 * len(dataset))
    train_set = dataset[:split]
    val_set   = dataset[split:]

    model     = model.to(device)
    opt       = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched     = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_val  = float('inf')
    history   = {'train': [], 'val': []}

    print(f"Training: {len(train_set)} train / {len(val_set)} val  |  "
          f"{epochs} epochs  |  device={device}")

    for epoch in range(1, epochs + 1):
        model.train()
        random.shuffle(train_set)
        tloss = 0.0
        BATCH = 8

        if USE_DGL:
            for b in range(0, len(train_set), BATCH):
                bg  = dgl.batch(train_set[b:b+BATCH]).to(device)
                out = model(bg)
                msk = bg.ndata['mask']
                tgt = bg.ndata['target'][msk].to(device)
                loss = _combined_loss(out, msk, tgt, bg.ndata['feat'],
                                      bg, device, use_dgl=True)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tloss += loss.item()
            tloss /= math.ceil(len(train_set) / BATCH)
        else:
            for d in train_set:
                d    = d.to(device)
                out  = model(d)
                loss = _combined_loss(out, d.mask, d.target_pos,
                                      d.node_feat, d, device, use_dgl=False)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tloss += loss.item()
            tloss /= len(train_set)

        model.eval()
        vloss = 0.0
        with torch.no_grad():
            if USE_DGL:
                for g in val_set:
                    g    = g.to(device)
                    out  = model(g)
                    msk  = g.ndata['mask']
                    tgt  = g.ndata['target'][msk]
                    vloss += F.mse_loss(out[msk], tgt).item()
            else:
                for d in val_set:
                    d     = d.to(device)
                    out   = model(d)
                    vloss += F.mse_loss(out[d.mask], d.target_pos).item()
        vloss /= max(len(val_set), 1)

        sched.step()
        history['train'].append(tloss)
        history['val'].append(vloss)
        if vloss < best_val:
            best_val = vloss
            torch.save(model.state_dict(), 'best_model.pt')
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs}  "
                  f"train={tloss:.5f}  val={vloss:.5f}  best={best_val:.5f}")

    model.load_state_dict(torch.load('best_model.pt',
                                     map_location=device, weights_only=True))
    print(f"Training done. Best val loss: {best_val:.5f}\n")
    return history


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: Pareto log
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class EmbeddingResult:
    ring_depth:  int
    edge_std:    float
    valid:       bool        # collision-free?
    source:      str         # 'physics' | 'gnn' | 'gnn+physics'
    attempt:     int
    backtrack:   int         # how many backtracks were needed

@dataclass
class ParetoLog:
    results: list = field(default_factory=list)

    def add(self, r: EmbeddingResult):
        self.results.append(r)

    def pareto_frontier(self) -> list:
        """
        Return the Pareto-optimal results by (ring_depth ↑, edge_std ↓).
        A result is Pareto-optimal if no other result is better on both axes.
        Only considers valid (collision-free) results.
        """
        valid = [r for r in self.results if r.valid]
        if not valid: return []
        frontier = []
        for r in sorted(valid, key=lambda x: (-x.ring_depth, x.edge_std)):
            dominated = any(
                (p.ring_depth >= r.ring_depth and p.edge_std <= r.edge_std
                 and (p.ring_depth, p.edge_std) != (r.ring_depth, r.edge_std))
                for p in frontier
            )
            if not dominated:
                frontier.append(r)
        return frontier

    def summary(self) -> str:
        lines = ["=== Pareto Frontier (ring_depth ↑, edge_std ↓) ==="]
        for r in self.pareto_frontier():
            lines.append(f"  rings={r.ring_depth:2d}  "
                         f"edge_std={r.edge_std:.4f}  "
                         f"source={r.source}  "
                         f"attempt={r.attempt}  "
                         f"backtracks={r.backtrack}")
        lines.append(f"\nValid embeddings per ring depth:")
        from collections import Counter
        counts = Counter(r.ring_depth for r in self.results if r.valid)
        for depth in sorted(counts):
            lines.append(f"  ring {depth}: {counts[depth]} valid embedding(s)")
        return "\n".join(lines)

    def save(self, path: str = "pareto_log.json"):
        import dataclasses
        with open(path, 'w') as f:
            json.dump([dataclasses.asdict(r) for r in self.results], f, indent=2)
        print(f"Pareto log saved to {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8: Search with escalating backtrack
# ═══════════════════════════════════════════════════════════════════════════════
#
# Strategy: grow the mesh ring by ring using GNN predictions.
# When a collision occurs at ring k:
#
#   Level 0 (fast):   retry ring k with a fresh random GNN context
#                     (same ring k-1 embedding, just re-run GNN prediction
#                     — useful because the GNN may have stochastic elements
#                     or we can perturb the input slightly)
#
#   Level 1 (medium): re-embed ring k-1 from scratch (new random seed),
#                     then redo ring k with GNN
#
#   Level 2 (slow):   restart the entire mesh from scratch
#
# After MAX_ATTEMPTS at each level we escalate. If all levels fail,
# we record the best ring depth achieved and stop.
#
# Successful GNN embeddings (and physics fallbacks) are added to the
# training dataset for periodic GNN retraining.

MAX_L0_RETRIES = 5    # retries before escalating to L1
MAX_L1_RETRIES = 3    # retries before escalating to L2
MAX_L2_RETRIES = 3    # full restarts before giving up


def _apply_gnn_ring(pos: np.ndarray, adj: dict, vertex_ring: dict,
                    faces: list, boundary: dict,
                    predict_ring: int, predictions: np.ndarray) -> np.ndarray:
    """Write GNN predictions into the position array for ring predict_ring."""
    pos = pos.copy()
    ring_verts = [v for v, r in vertex_ring.items() if r == predict_ring]
    for i, v in enumerate(ring_verts):
        if i < len(predictions):
            pos[v] = predictions[i]
    return pos


def _physics_ring(pos: np.ndarray, adj: dict, vertex_ring: dict,
                  predict_ring: int, relax_steps: int) -> np.ndarray:
    """Relax only the vertices of predict_ring (keep all others fixed)."""
    new_verts = [v for v, r in vertex_ring.items() if r == predict_ring]
    return relax(pos, adj, free_verts=new_verts, steps=relax_steps)


def search(model: HyperbolicGNN,
           max_rings: int = 6,
           relax_steps: int = 500,
           device: torch.device = None,
           pareto_log: ParetoLog = None,
           new_training_data: list = None,
           attempt_number: int = 0) -> tuple:
    """
    Try to build the largest collision-free {3,7} embedding.

    Returns (pos, max_valid_ring, num_backtracks) where max_valid_ring is the
    deepest ring that was collision-free.

    Side effects:
      - Appends EmbeddingResults to pareto_log
      - Appends new TorchMeshData / DGL graphs to new_training_data
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if pareto_log is None:
        pareto_log = ParetoLog()
    if new_training_data is None:
        new_training_data = []

    rng          = np.random.default_rng()
    total_backs  = 0

    # Build the full combinatorial structure once
    adj, vertex_ring, faces, boundary = build_combinatorial(max_rings)
    N = max(vertex_ring) + 1

    # ── Seed: rings 0 and 1 via physics ──────────────────────────────────
    seed_verts = [v for v, r in vertex_ring.items() if r <= 1]
    pos        = initial_positions(vertex_ring, boundary, rng=rng)
    pos        = relax(pos, adj, free_verts=seed_verts, steps=relax_steps)

    best_valid_ring = 1
    best_pos        = pos.copy()

    # ── Grow ring by ring ─────────────────────────────────────────────────
    for r in range(2, max_rings + 1):
        ring_faces = [(a,b,c) for a,b,c in faces
                      if any(vertex_ring[v] == r for v in (a,b,c))]
        succeeded  = False

        for l2 in range(MAX_L2_RETRIES):
            if l2 > 0:
                # Level 2: full restart, rebuilding rings 1..r-1 ring-by-ring
                print(f"    [L2 restart {l2}] ring {r}: full restart")
                total_backs += 1
                best_valid_ring = max(1, r - 2)
                seed_v = [v for v, rr in vertex_ring.items() if rr <= 1]
                pos    = initial_positions(vertex_ring, boundary, rng=rng)
                pos    = relax(pos, adj, free_verts=seed_v, steps=relax_steps)
                l2_ok  = True
                for rr in range(2, r):
                    p_rr = model.predict_ring_positions(
                        pos, adj, vertex_ring, rr, device)
                    if p_rr is not None:
                        pos = _apply_gnn_ring(pos, adj, vertex_ring,
                                              faces, boundary, rr, p_rr)
                    else:
                        rr_v = [v for v, rv in vertex_ring.items() if rv == rr]
                        pos  = relax(pos, adj, free_verts=rr_v,
                                     steps=relax_steps)
                    if has_collision(pos, faces, check_rings={rr},
                                     vertex_ring=vertex_ring):
                        l2_ok = False; break
                if not l2_ok:
                    continue

            for l1 in range(MAX_L1_RETRIES):
                if l1 > 0:
                    # Level 1: fresh seed for ring r-1, keep 0..r-2 fixed
                    print(f"    [L1 retry {l1}] ring {r}: re-seeding ring {r-1}")
                    total_backs += 1
                    prev_v  = [v for v, rr in vertex_ring.items() if rr == r-1]
                    fresh   = initial_positions(vertex_ring, boundary, rng=rng)
                    for v in prev_v: pos[v] = fresh[v]
                    pos = relax(pos, adj, free_verts=prev_v, steps=relax_steps)
                    # If re-seeded ring r-1 itself collides, try again
                    if has_collision(pos, faces, check_rings={r-1},
                                     vertex_ring=vertex_ring):
                        continue

                # Level 0: try GNN first, then physics if GNN fails
                # Collect all GNN candidates; if all collide, try physics
                gnn_failed = False
                for l0 in range(MAX_L0_RETRIES):
                    pred = model.predict_ring_positions(
                        pos, adj, vertex_ring, r, device)
                    if pred is not None:
                        candidate = _apply_gnn_ring(
                            pos, adj, vertex_ring, faces, boundary, r, pred)
                        source = 'gnn'
                    else:
                        candidate = None

                    if candidate is not None:
                        collides = has_collision(
                            candidate, faces,
                            check_rings={r},
                            vertex_ring=vertex_ring)
                        if not collides:
                            std = edge_std(candidate, adj)
                            # Reject poor quality even if collision-free:
                            # a ring with edge_std > 0.15 won't support the next ring
                            if std > 0.15:
                                print(f"    [L0 quality reject] ring {r}: "
                                      f"edge_std={std:.3f} too poor, retrying")
                                total_backs += 1
                                if l0 < MAX_L0_RETRIES - 1:
                                    continue
                                else:
                                    gnn_failed = True; break
                            pareto_log.add(EmbeddingResult(
                                ring_depth=r, edge_std=std, valid=True,
                                source=source, attempt=attempt_number,
                                backtrack=total_backs))
                            if not USE_DGL:
                                d = TorchMeshData(candidate, adj, vertex_ring, r-1, r)
                                if d.valid: new_training_data.append(d)
                            else:
                                g = make_dgl_data(candidate, adj, vertex_ring, r-1, r)
                                if g: new_training_data.append(g)
                            pos             = candidate
                            best_valid_ring = r
                            best_pos        = pos.copy()
                            succeeded       = True
                            break
                        else:
                            std = edge_std(candidate, adj)
                            pareto_log.add(EmbeddingResult(
                                ring_depth=r, edge_std=std, valid=False,
                                source=source, attempt=attempt_number,
                                backtrack=total_backs))
                            if l0 < MAX_L0_RETRIES - 1:
                                print(f"    [L0 retry {l0+1}/{MAX_L0_RETRIES}] "
                                      f"ring {r}: collision, retrying GNN")
                                total_backs += 1
                            else:
                                gnn_failed = True
                    else:
                        gnn_failed = True; break

                # If GNN failed all retries, fall back to physics relaxer
                if not succeeded and gnn_failed:
                    print(f"    [physics fallback] ring {r}: GNN exhausted, using physics")
                    ring_verts = [v for v, rv in vertex_ring.items() if rv == r]
                    candidate  = relax(pos, adj, free_verts=ring_verts,
                                       steps=relax_steps)
                    source = 'physics'
                    collides = has_collision(candidate, faces,
                                            check_rings={r},
                                            vertex_ring=vertex_ring)
                    if not collides:
                        std = edge_std(candidate, adj)
                        pareto_log.add(EmbeddingResult(
                            ring_depth=r, edge_std=std, valid=True,
                            source=source, attempt=attempt_number,
                            backtrack=total_backs))
                        if not USE_DGL:
                            d = TorchMeshData(candidate, adj, vertex_ring, r-1, r)
                            if d.valid: new_training_data.append(d)
                        else:
                            g = make_dgl_data(candidate, adj, vertex_ring, r-1, r)
                            if g: new_training_data.append(g)
                        pos             = candidate
                        best_valid_ring = r
                        best_pos        = pos.copy()
                        succeeded       = True

                if succeeded: break
            if succeeded: break

        if not succeeded:
            print(f"  Ring {r}: all retries exhausted. "
                  f"Best valid ring: {best_valid_ring}")
            break

        print(f"  Ring {r}: embedded  "
              f"(edge_std={edge_std(pos,adj):.4f}  source={source})")

    return best_pos, best_valid_ring, total_backs


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9: Export
# ═══════════════════════════════════════════════════════════════════════════════

_RING_COLORS = [
    (255,255,255), (253,231,37),  (122,209,81),
    (34, 168,132), (42, 120,142), (65, 68, 135),
    (68,  1,  84), (180, 50, 50), (220,100, 20),
]
def _color(r): return _RING_COLORS[min(r, len(_RING_COLORS)-1)]

def save_ply(path, pos, faces, vertex_ring):
    with open(path, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pos)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\nend_header\n")
        for i, p in enumerate(pos):
            r, g, b = _color(vertex_ring[i])
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {r} {g} {b}\n")
        for a,b,c in faces:
            f.write(f"3 {a} {b} {c}\n")

def save_obj(path, pos, faces):
    with open(path,'w') as f:
        f.write("# {3,7} hyperbolic triangle mesh\n")
        for p in pos: f.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        for a,b,c in faces: f.write(f"f {a+1} {b+1} {c+1}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10: Main loop  (train → search → retrain → search → ...)
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    print("=" * 65)
    print("  {3,7} Hyperbolic Triangle Mesh — GNN Search")
    print(f"  Backend: {'DGL' if USE_DGL else 'pure PyTorch'}")
    print("=" * 65)

    # ── Configuration ─────────────────────────────────────────────────────
    MAX_RINGS      = 4     # how far to try to grow
    NUM_SAMPLES    = 120   # initial training set size
    INITIAL_EPOCHS = 50    # initial training epochs
    FINETUNE_EPOCHS= 20    # epochs for online retraining
    RELAX_STEPS    = 10000 # steps to converge to edge_std ~0.02 (Rust quality)
    NUM_SEARCHES   = 5     # how many independent search attempts
    RETRAIN_EVERY  = 2     # retrain GNN after every N search attempts

    # ── Relaxer physics ────────────────────────────────────────────────────
    # simultaneous=True relaxes all vertices at once, matching Rust main.rs.
    # This is what achieves edge_std ~0.017 (near-equilateral).
    # Ring-by-ring (False) is faster but produces poor quality training data.
    SIMULTANEOUS_RELAX = True
    ATTRACT            = 0.1
    REPEL              = 0.6

    # ── Crystal forces ─────────────────────────────────────────────────────
    # crystal_align: pulls each vertex's neighbors toward their ideal Z[phi]^3
    #   positions, gated by crystallinity score (inactive until near convergence).
    # crystal_snap: pulls spatially close crystalline vertex pairs together,
    #   enabling distant stitching along lattice seams.
    # Set both to 0.0 to use pure spring+repulsion (Rust-equivalent baseline).
    CRYSTAL_ALIGN  = 0.0   # 0.0 to disable (re-enable after baseline confirmed)
    CRYSTAL_SNAP   = 0.0   # 0.0 to disable

    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pareto_log = ParetoLog()

    # ── Phase 1: Build initial training dataset (physics) ─────────────────
    print(f"\n[Phase 1] Generating {NUM_SAMPLES} physics-based training samples...")
    dataset, raw_meshes = build_dataset(NUM_SAMPLES, max_rings=MAX_RINGS,
                                        relax_steps=RELAX_STEPS)
    # Note: SIMULTANEOUS_RELAX / ATTRACT / REPEL / CRYSTAL_* are read from
    # module-level globals inside generate_training_sample.

    # ── Save best physics meshes ──────────────────────────────────────────
    # Sort by (max rings DESC, edge_std ASC) and save top 5
    raw_meshes.sort(key=lambda x: (-x[5], x[4]))
    print(f"Saving top physics meshes...")
    for i, (p, a, vr, f, std, nr) in enumerate(raw_meshes[:5]):
        fname = f"physics_mesh_{i+1:02d}_rings{nr}_std{std:.4f}"
        save_ply(fname + ".ply", p, f, vr)
        save_obj(fname + ".obj", p, f)
        print(f"  {fname}.ply  (rings={nr}, edge_std={std:.4f})")

    # ── Phase 2: Initial GNN training ─────────────────────────────────────
    print(f"\n[Phase 2] Initial GNN training ({INITIAL_EPOCHS} epochs)...")
    model = HyperbolicGNN(in_feats=IN_FEATS, hidden=192, num_layers=6,
                          use_dgl=USE_DGL)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    train_model(model, dataset, epochs=INITIAL_EPOCHS, lr=1e-3, device=device)

    # ── Phase 3: Search loop with online retraining ───────────────────────
    print(f"\n[Phase 3] Search loop ({NUM_SEARCHES} attempts)...")
    new_data      = []
    best_ever     = 0
    best_ever_pos = None
    best_ever_adj = None
    best_ever_vr  = None
    best_ever_f   = None

    for attempt in range(NUM_SEARCHES):
        print(f"\n── Attempt {attempt+1}/{NUM_SEARCHES} ──")

        # Retrain periodically on accumulated successful embeddings
        if attempt > 0 and attempt % RETRAIN_EVERY == 0 and new_data:
            print(f"  [Online retrain] {len(new_data)} new samples collected")
            combined = dataset + new_data
            train_model(model, combined, epochs=FINETUNE_EPOCHS,
                        lr=3e-4, device=device)
            new_data = []

        adj, vertex_ring, faces, boundary = build_combinatorial(MAX_RINGS)

        pos, best_ring, n_backs = search(
            model, max_rings=MAX_RINGS, relax_steps=RELAX_STEPS,
            device=device, pareto_log=pareto_log,
            new_training_data=new_data, attempt_number=attempt)

        print(f"  Result: best_ring={best_ring}  backtracks={n_backs}")

        if best_ring > best_ever:
            best_ever     = best_ring
            best_ever_pos = pos
            best_ever_adj = adj
            best_ever_vr  = vertex_ring
            best_ever_f   = faces
            print(f"  ★ New best: {best_ever} rings!")

    # ── Results ───────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(pareto_log.summary())
    pareto_log.save("pareto_log.json")

    if best_ever_pos is not None:
        save_ply("best_mesh.ply", best_ever_pos, best_ever_f, best_ever_vr)
        save_obj("best_mesh.obj", best_ever_pos, best_ever_f)
        print(f"\nBest mesh saved: best_mesh.ply / .obj  ({best_ever} rings)")

    print("\nDone!")
    print("  best_mesh.ply     — best collision-free embedding found")
    print("  pareto_log.json   — full search history")
    print("  best_model.pt     — trained GNN weights")
    print("\nView .ply in MeshLab, Blender, or https://3dviewer.net")
    print("To switch to DGL: set USE_DGL = True at the top of this file.")
