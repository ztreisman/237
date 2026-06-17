"""
crystal_neighborhoods.py
========================
Multi-lattice crystalline neighborhood scoring and snap-force infrastructure
for the {3,7} hyperbolic tiling relaxer.

Architecture
------------
Each vertex v carries a CrystallineNeighborhood object that knows:
  - Its k-hop subgraph (graph metric balls B(v,1), B(v,2), B(v,3))
  - Scores against multiple lattices (Z[phi]^3, FCC, E8 shadow, ...)
  - Snap candidates: other vertices whose B(v,1) matches v's against the
    same lattice, weighted by graph distance and holonomy
  - A force contribution that activates as crystallinity score rises

The relaxer then calls:
  1. spring_repulsion_step()   -- existing attraction + repulsion
  2. crystal_snap_step()       -- additional force from this module
  
The snap force is gated by crystallinity_score, so it is essentially
zero far from any lattice basin and grows dominant near convergence.

Lattice conventions
-------------------
All lattice configurations are stored in *normalized* units where the
shortest edge = 1.0, matching EDGE_LENGTH in the relaxer.

Z[phi]^3 fingerprint extracted directly from 237Lattice.vZome:
  - Every degree-7 vertex has spoke signature AABCCCC
    A: |e|~=1.000 (2 neighbors), type sq=(18,21) in Z[phi]
    B: |e|~=1.004 (1 neighbor),  type sq=(20,20) = pure phi power
    C: |e|~=1.064 (4 neighbors), type sq=(20,24)
  - Two ring-1 sub-configurations:
    Class I  (12/15 vertices): ring1 = AABCCCCC
    Class II  (3/15 vertices): ring1 = AABBCCCC
  - Canonical pairwise distances among the 7 neighbors (21 values, sorted):
    [1.000, 1.000, 1.004, 1.064?5, 1.417?2, 1.460?2, 1.694?4, 1.730, 1.935, 1.967?2, 1.998]
"""

import numpy as np
from collections import defaultdict
from scipy.spatial import cKDTree
from typing import Optional
import itertools


# ---------------------------------------------------------------------------
# Z[phi]^3 lattice constants
# ---------------------------------------------------------------------------

PHI = (1 + 5**0.5) / 2

# Squared lengths in Z[phi]: (a,b) means a + b*phi
# In normalized units (A=1.0):
ZPHI_TYPE_SQ = {
    'A': (18, 21),   # sq ~= 51.979, |e| ~= 7.210 ? normalized 1.000
    'B': (20, 20),   # sq ~= 52.361, |e| ~= 7.236 ? normalized 1.004
    'C': (20, 24),   # sq ~= 58.833, |e| ~= 7.670 ? normalized 1.064
}
ZPHI_LATTICE_SCALE = (18 + 21 * PHI) ** 0.5  # = 7.2096, type-A edge in lattice units

def zphi_length(type_key):
    a, b = ZPHI_TYPE_SQ[type_key]
    return ((a + b * PHI) / (18 + 21 * PHI)) ** 0.5   # normalized

ZPHI_LENGTHS = {k: zphi_length(k) for k in 'ABC'}

# Canonical sorted spoke distances for a degree-7 vertex (normalized, 7 values)
ZPHI_SPOKE_CANONICAL = np.array(sorted(
    [ZPHI_LENGTHS['A']] * 2 +
    [ZPHI_LENGTHS['B']] * 1 +
    [ZPHI_LENGTHS['C']] * 4
))

# Canonical pairwise distances among the 7 neighbors (21 values)
# Two sub-classes; we store both and score against the nearer one
# Class I:  ring-1 edges = AABCCCCC (8 edges for 7 vertices in a cycle... 
#           actually ring-1 has 8 edges among 7 neighbors in a heptagon)
# Rather than hardcode, we compute from the canonical edge-type sequence.
# Neighbor-ring edge types for class I and II:
ZPHI_RING1_CLASS_I  = ['A','A','B','C','C','C','C','C']  # 8 edges in ring-1
ZPHI_RING1_CLASS_II = ['A','A','B','B','C','C','C','C']

def _ring1_pairwise(ring_types):
    """
    Given the cyclic edge-type sequence around the 7-cycle of neighbors,
    produce the 21 sorted pairwise distances.
    We need to know the actual geometry, not just the types.
    Since we have the canonical offsets from the vZome file we use those.
    This is a placeholder; the actual offsets are in ZPHI_NEIGHBOR_OFFSETS.
    """
    pass  # filled below after defining the offsets

# Canonical neighbor offset vectors for deg-7 vertex v=12 from vZome,
# normalized to EDGE_LENGTH=1. Order: sorted by neighbor index.
# Types: [C, C, B, C, A, C, A] for neighbors [2,6,13,14,16,22,24]
ZPHI_NEIGHBOR_OFFSETS_RAW = np.array([
    [-0.865 , -0.3631,  0.5018],   # type C
    [-0.5018, -0.865 , -0.3631],   # type C
    [ 0.    ,  0.    ,  1.0037],   # type B
    [ 0.5018, -0.865 , -0.3631],   # type C
    [ 0.5018,  0.    , -0.865 ],   # type A
    [ 0.865 , -0.3631,  0.5018],   # type C
    [ 0.865 ,  0.5018,  0.    ],   # type A
])
ZPHI_NEIGHBOR_TYPES_RAW = ['C','C','B','C','A','C','A']

# Sorted by length (A,A,B,C,C,C,C)
_order = np.argsort(np.linalg.norm(ZPHI_NEIGHBOR_OFFSETS_RAW, axis=1))
ZPHI_NEIGHBOR_OFFSETS = ZPHI_NEIGHBOR_OFFSETS_RAW[_order]
ZPHI_NEIGHBOR_TYPES   = [ZPHI_NEIGHBOR_TYPES_RAW[i] for i in _order]

# 21 canonical pairwise distances
_pw21 = sorted(
    np.linalg.norm(ZPHI_NEIGHBOR_OFFSETS[i] - ZPHI_NEIGHBOR_OFFSETS[j])
    for i in range(7) for j in range(i+1, 7)
)
ZPHI_PAIRWISE_CANONICAL = np.array(_pw21)


# ---------------------------------------------------------------------------
# Additional lattice fingerprints (FCC, SC, HCP as placeholders)
# These are expressed as sorted spoke distances + pairwise distances
# for a degree-k vertex, normalized to shortest edge = 1.
# ---------------------------------------------------------------------------

def _fcc_canonical(degree=12):
    """FCC: 12 nearest neighbors at distance 1, pairwise at 1 or ?2."""
    # FCC neighbor offsets: all permutations of (?1,?1,0)/?2
    offsets = []
    for i, j, k in itertools.permutations([0,1,2]):
        for s1, s2 in itertools.product([-1,1], [-1,1]):
            v = [0,0,0]; v[i]=s1; v[j]=s2
            offsets.append(v)
    offsets = np.unique(np.array(offsets, dtype=float) / 2**0.5, axis=0)
    pw = sorted(np.linalg.norm(offsets[i]-offsets[j])
                for i in range(len(offsets)) for j in range(i+1,len(offsets))
                if np.linalg.norm(offsets[i]-offsets[j]) < 1.5)
    return {
        'name': 'FCC',
        'degree': 12,
        'offsets': offsets,
        'spoke_canonical': np.ones(12),
        'pairwise_canonical': np.array(sorted(pw)),
    }

def _sc_canonical():
    """Simple cubic: 6 nearest neighbors."""
    offsets = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]], dtype=float)
    pw = sorted(np.linalg.norm(offsets[i]-offsets[j])
                for i in range(6) for j in range(i+1,6))
    return {
        'name': 'SC',
        'degree': 6,
        'offsets': offsets,
        'spoke_canonical': np.ones(6),
        'pairwise_canonical': np.array(sorted(pw)),
    }

LATTICE_DB = {
    'zphi': {
        'name': 'Z[phi]^3 icosahedral',
        'degree': 7,
        'offsets': ZPHI_NEIGHBOR_OFFSETS,
        'spoke_canonical': ZPHI_SPOKE_CANONICAL,
        'pairwise_canonical': ZPHI_PAIRWISE_CANONICAL,
        'type_lengths': ZPHI_LENGTHS,
    },
    'fcc': _fcc_canonical(),
    'sc': _sc_canonical(),
}


# ---------------------------------------------------------------------------
# Core scoring functions
# ---------------------------------------------------------------------------

def spoke_score(neighbor_offsets: np.ndarray, canonical: np.ndarray) -> float:
    """
    Score how close the sorted spoke distances are to the canonical pattern.
    Returns value in [0, 1] where 1 = perfect match.
    canonical: sorted spoke distances (length = degree)
    neighbor_offsets: (degree, 3) array of vectors from center to neighbors
    """
    if len(neighbor_offsets) != len(canonical):
        return 0.0
    actual = np.sort(np.linalg.norm(neighbor_offsets, axis=1))
    residual = np.mean((actual - canonical)**2)
    return float(np.exp(-residual / 0.05))   # sigma=0.22 in distance units


def pairwise_score(neighbor_offsets: np.ndarray, canonical_pw: np.ndarray) -> float:
    """
    Score the sorted pairwise distance distribution among neighbors
    against the canonical fingerprint.
    """
    n = len(neighbor_offsets)
    if n < 2:
        return 0.0
    pw = sorted(
        np.linalg.norm(neighbor_offsets[i] - neighbor_offsets[j])
        for i in range(n) for j in range(i+1, n)
    )
    npw = len(canonical_pw)
    # Only compare up to min length (in case of degree mismatch)
    m = min(len(pw), npw)
    residual = np.mean((np.array(pw[:m]) - canonical_pw[:m])**2)
    return float(np.exp(-residual / 0.05))


def crystallinity_score(
    center_pos: np.ndarray,
    neighbor_positions: np.ndarray,
    lattice_key: str = 'zphi',
    weights: tuple = (0.4, 0.6),   # (spoke_weight, pairwise_weight)
) -> float:
    """
    Combined crystallinity score in [0,1] for a vertex and its 1-hop neighbors.
    Higher = more like the target lattice local configuration.
    """
    L = LATTICE_DB[lattice_key]
    offsets = neighbor_positions - center_pos
    if len(offsets) != L['degree']:
        return 0.0
    s = spoke_score(offsets, L['spoke_canonical'])
    p = pairwise_score(offsets, L['pairwise_canonical'])
    return weights[0] * s + weights[1] * p


def best_lattice_score(
    center_pos: np.ndarray,
    neighbor_positions: np.ndarray,
) -> tuple:
    """
    Score against all known lattices. Returns (best_key, best_score, all_scores).
    """
    scores = {}
    for key, L in LATTICE_DB.items():
        if len(neighbor_positions) == L['degree']:
            scores[key] = crystallinity_score(center_pos, neighbor_positions, key)
    if not scores:
        return None, 0.0, {}
    best = max(scores, key=scores.get)
    return best, scores[best], scores


# ---------------------------------------------------------------------------
# Optimal rotation alignment (Kabsch algorithm)
# ---------------------------------------------------------------------------

def kabsch_align(P: np.ndarray, Q: np.ndarray) -> tuple:
    """
    Find rotation R minimizing ||P - Q@R||_F.
    Returns (R, rmsd).
    P, Q: (n, 3) arrays -- both centered at origin.
    """
    H = P.T @ Q
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    P_rot = P @ R
    rmsd = np.sqrt(np.mean(np.sum((P_rot - Q)**2, axis=1)))
    return R, rmsd


def align_to_lattice(
    center_pos: np.ndarray,
    neighbor_positions: np.ndarray,
    lattice_key: str = 'zphi',
) -> tuple:
    """
    Find the best rotation aligning the neighbor cloud to the canonical
    lattice frame. Returns (R, rmsd, canonical_targets).
    
    canonical_targets: the lattice neighbor positions in the current frame
    (i.e., what the neighbors *should* be at, in R^3).
    """
    L = LATTICE_DB[lattice_key]
    offsets = neighbor_positions - center_pos

    if len(offsets) != L['degree']:
        return None, np.inf, None

    # Normalize actual spoke lengths to match canonical pattern
    # (we want the rotation, not the scale)
    actual_norms = np.linalg.norm(offsets, axis=1, keepdims=True).clip(min=1e-12)
    canon_norms  = L['spoke_canonical'][np.argsort(actual_norms.flatten())][:, None]

    # Match actual neighbors to canonical by greedy nearest-neighbor on sorted norms
    actual_sorted_idx = np.argsort(np.linalg.norm(offsets, axis=1))
    canon_sorted_idx  = np.argsort(L['spoke_canonical'])
    P = offsets[actual_sorted_idx]   # actual offsets sorted by length
    Q = L['offsets'][canon_sorted_idx]  # canonical offsets sorted by length

    # Normalize each to unit length for rotation alignment
    P_hat = P / np.linalg.norm(P, axis=1, keepdims=True).clip(min=1e-12)
    Q_hat = Q / np.linalg.norm(Q, axis=1, keepdims=True).clip(min=1e-12)

    R, rmsd = kabsch_align(P_hat, Q_hat)

    # Canonical targets: canonical offsets rotated into current frame
    # (R maps canonical ? current frame? No: Kabsch aligns P?Q, so R maps P to Q)
    # We want canonical in current frame: Q = P @ R, so canonical = Q_hat*norms
    # targets in current frame = L['offsets'][canon_sorted_idx] @ R.T
    targets = (L['offsets'][canon_sorted_idx]) @ R.T + center_pos

    # Re-index: targets[i] is the ideal position for actual neighbor actual_sorted_idx[i]
    target_by_neighbor = np.empty_like(neighbor_positions)
    target_by_neighbor[actual_sorted_idx] = targets

    return R, rmsd, target_by_neighbor


# ---------------------------------------------------------------------------
# Graph metric balls
# ---------------------------------------------------------------------------

def graph_balls(adj: dict, v: int, max_k: int = 3) -> dict:
    """
    BFS from v. Returns {k: set_of_vertices_at_distance_k}.
    """
    visited = {v: 0}
    frontier = [v]
    balls = {0: {v}}
    for k in range(1, max_k + 1):
        next_frontier = []
        for u in frontier:
            for w in adj[u]:
                if w not in visited:
                    visited[w] = k
                    next_frontier.append(w)
        balls[k] = set(next_frontier)
        frontier = next_frontier
        if not frontier:
            break
    return balls


def graph_distance(adj: dict, u: int, v: int, max_dist: int = 50) -> int:
    """BFS shortest path from u to v."""
    if u == v:
        return 0
    visited = {u}
    frontier = [u]
    for d in range(1, max_dist + 1):
        next_f = []
        for w in frontier:
            for x in adj[w]:
                if x == v:
                    return d
                if x not in visited:
                    visited.add(x)
                    next_f.append(x)
        frontier = next_f
        if not frontier:
            break
    return max_dist + 1


# ---------------------------------------------------------------------------
# Holonomy tracking along graph paths
# ---------------------------------------------------------------------------

def path_holonomy(
    path_vertices: list,
    pos: np.ndarray,
    adj: dict,
) -> np.ndarray:
    """
    Approximate parallel transport of the local icosahedral frame along a
    graph path. Returns the accumulated rotation matrix (3?3).
    
    Method: at each step u?w, compute the rotation that maps the
    local normal at u (estimated from neighbors) to that at w.
    This is the discrete connection on the surface.
    """
    R_total = np.eye(3)
    for i in range(len(path_vertices) - 1):
        u = path_vertices[i]
        w = path_vertices[i + 1]
        nu = local_normal(u, pos, adj)
        nw = local_normal(w, pos, adj)
        if nu is None or nw is None:
            continue
        # Rotation mapping nu ? nw
        R_step = rotation_between(nu, nw)
        R_total = R_step @ R_total
    return R_total


def local_normal(v: int, pos: np.ndarray, adj: dict) -> Optional[np.ndarray]:
    """Estimate local surface normal at v from neighbor cross products."""
    nbrs = list(adj[v])
    if len(nbrs) < 2:
        return None
    center = pos[v]
    offsets = pos[np.array(nbrs)] - center
    # SVD: smallest singular vector is the normal
    _, _, Vt = np.linalg.svd(offsets)
    return Vt[-1]   # (3,)


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation matrix mapping unit vector a to unit vector b."""
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    v = np.cross(a, b)
    c = np.dot(a, b)
    if abs(c + 1) < 1e-9:   # anti-parallel
        return -np.eye(3)
    s = np.linalg.norm(v)
    if s < 1e-9:
        return np.eye(3)
    K = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    return np.eye(3) + K + K @ K * (1 - c) / (s * s)


# ---------------------------------------------------------------------------
# Snap candidate detection
# ---------------------------------------------------------------------------

MEANINGFUL_GRAPH_DISTANCES = {
    # Klein quartic relations: word lengths in ?1 of the {3,7}/PSL(2,7) quotient
    # 24 degree-7 vertices, minimal identifying distances
    7: 1.0,    # one full heptagonal cycle
    14: 0.9,
    21: 0.8,
    3: 0.3,    # short -- probably accidental, low weight
    4: 0.3,
    # Multiples related to ring sizes 7,21,56,147,...
    28: 0.7,
    42: 0.6,
}

def snap_candidates(
    v: int,
    pos: np.ndarray,
    adj: dict,
    vertex_ring: dict,
    spatial_tree: cKDTree,
    all_vertices: np.ndarray,   # indices of all vertices
    score_threshold: float = 0.3,
    spatial_radius: float = 2.0,
    lattice_key: str = 'zphi',
    max_graph_dist_check: int = 50,
) -> list:
    """
    Find snap candidates for vertex v:
    Vertices u (u?v) such that:
      1. |pos[v] - pos[u]| < spatial_radius  (close in R^3)
      2. crystallinity_score(v) > score_threshold
      3. crystallinity_score(u) > score_threshold
      4. Both score best against the same lattice
    
    Returns list of dicts:
      {u, graph_dist, spatial_dist, score_v, score_u,
       lattice, graph_dist_weight, holonomy_R}
    """
    nbrs_v = np.array(list(adj[v]))
    if len(nbrs_v) == 0:
        return []

    L = LATTICE_DB[lattice_key]
    if len(nbrs_v) != L['degree']:
        return []

    score_v = crystallinity_score(pos[v], pos[nbrs_v], lattice_key)
    if score_v < score_threshold:
        return []

    # Find spatially close vertices
    idxs = spatial_tree.query_ball_point(pos[v], r=spatial_radius)
    candidates = []
    for u in idxs:
        u = int(u)
        if u == v or u in set(adj[v]):
            continue

        nbrs_u = np.array(list(adj[u]))
        if len(nbrs_u) != L['degree']:
            continue

        score_u = crystallinity_score(pos[u], pos[nbrs_u], lattice_key)
        if score_u < score_threshold:
            continue

        spatial_dist = np.linalg.norm(pos[v] - pos[u])
        graph_dist   = graph_distance(adj, v, u, max_dist=max_graph_dist_check)
        gd_weight    = MEANINGFUL_GRAPH_DISTANCES.get(graph_dist, 0.1)

        # Holonomy: find BFS path and compute parallel transport
        # (expensive for long paths; skip if graph_dist > 15)
        hol_R = None
        if graph_dist <= 15:
            path = _bfs_path(adj, v, u)
            if path:
                hol_R = path_holonomy(path, pos, adj)

        candidates.append({
            'u': u,
            'graph_dist': graph_dist,
            'spatial_dist': spatial_dist,
            'score_v': score_v,
            'score_u': score_u,
            'lattice': lattice_key,
            'graph_dist_weight': gd_weight,
            'holonomy_R': hol_R,
            'combined_score': score_v * score_u * gd_weight / (spatial_dist + 0.1),
        })

    return sorted(candidates, key=lambda x: -x['combined_score'])


def _bfs_path(adj: dict, start: int, end: int) -> list:
    """BFS to find shortest path from start to end."""
    prev = {start: None}
    queue = [start]
    while queue:
        v = queue.pop(0)
        if v == end:
            path = []
            while v is not None:
                path.append(v)
                v = prev[v]
            return path[::-1]
        for u in adj[v]:
            if u not in prev:
                prev[u] = v
                queue.append(u)
    return []


# ---------------------------------------------------------------------------
# Crystal snap force
# ---------------------------------------------------------------------------

def crystal_snap_forces(
    pos: np.ndarray,
    adj: dict,
    vertex_ring: dict,
    lattice_key: str = 'zphi',
    score_threshold: float = 0.25,
    snap_strength: float = 0.05,
    align_strength: float = 0.03,
) -> np.ndarray:
    """
    Compute additional forces pulling vertices toward crystalline configurations.
    
    Two components:
    
    1. ALIGNMENT force: for each vertex v with crystallinity > threshold,
       pull its neighbors toward their ideal lattice positions (from align_to_lattice).
       Strength scales with crystallinity_score.
    
    2. IDENTIFICATION force: for each snap candidate pair (v, u),
       pull v and u toward each other with weight = combined_score.
       This is what enables distant stitching.
    
    Returns force array of shape (N, 3).
    """
    N = len(pos)
    forces = np.zeros((N, 3))
    L = LATTICE_DB[lattice_key]

    # Build spatial tree
    tree = cKDTree(pos)

    for v in range(N):
        nbrs = np.array(list(adj[v]))
        if len(nbrs) != L['degree']:
            continue

        score = crystallinity_score(pos[v], pos[nbrs], lattice_key)
        if score < score_threshold:
            continue

        # --- Alignment force ---
        R, rmsd, targets = align_to_lattice(pos[v], pos[nbrs], lattice_key)
        if targets is not None:
            gate = score * align_strength
            for i, n in enumerate(nbrs):
                f = (targets[i] - pos[n]) * gate
                forces[n] += f
                forces[v] -= f * 0.5   # reaction: center moves opposite

        # --- Identification force ---
        # Only look for snap candidates when score is high enough
        if score > 0.5:
            cands = snap_candidates(
                v, pos, adj, vertex_ring, tree,
                np.arange(N), lattice_key=lattice_key,
                score_threshold=score_threshold,
                spatial_radius=3.0,
            )
            for c in cands[:3]:   # top 3 candidates only
                u = c['u']
                direction = pos[u] - pos[v]
                dist = np.linalg.norm(direction)
                if dist < 1e-9:
                    continue
                f_mag = snap_strength * c['combined_score']
                f = direction / dist * f_mag
                forces[v] += f
                forces[u] -= f   # equal and opposite

    return forces


# ---------------------------------------------------------------------------
# Higher-order feature computation for GNN
# ---------------------------------------------------------------------------

def compute_neighborhood_features(
    v: int,
    pos: np.ndarray,
    adj: dict,
    vertex_ring: dict,
    lattice_keys: list = ['zphi', 'fcc', 'sc'],
    k_max: int = 2,
) -> np.ndarray:
    """
    Compute a feature vector for vertex v encoding:
      - Crystallinity scores against each lattice (len = n_lattices)
      - Best-lattice alignment RMSD
      - Graph ball sizes at k=1,2 (already implicit in degree but useful)
      - Spoke distance mean/std at k=1
      - Pairwise distance mean/std at k=1
      - Snap readiness (max crystallinity product over spatially close vertices)
    
    Returns: np.ndarray of shape (n_features,)
    """
    nbrs = np.array(list(adj[v]))
    features = []

    # 1. Per-lattice crystallinity scores
    for key in lattice_keys:
        L = LATTICE_DB[key]
        if len(nbrs) == L['degree']:
            s = crystallinity_score(pos[v], pos[nbrs], key)
        else:
            s = 0.0
        features.append(s)

    # 2. Spoke distance statistics (k=1)
    if len(nbrs) > 0:
        spokes = np.linalg.norm(pos[nbrs] - pos[v], axis=1)
        features.extend([float(np.mean(spokes)), float(np.std(spokes))])
    else:
        features.extend([0.0, 0.0])

    # 3. Pairwise distance stats among k=1 neighbors
    if len(nbrs) >= 2:
        pw = [np.linalg.norm(pos[nbrs[i]]-pos[nbrs[j]])
              for i in range(len(nbrs)) for j in range(i+1,len(nbrs))]
        features.extend([float(np.mean(pw)), float(np.std(pw))])
    else:
        features.extend([0.0, 0.0])

    # 4. k=2 ring size (number of vertices at graph distance exactly 2)
    balls = graph_balls(adj, v, max_k=k_max)
    k2_size = len(balls.get(2, set()))
    features.append(float(k2_size))

    # 5. Alignment RMSD to best lattice
    best_rmsd = np.inf
    for key in lattice_keys:
        L = LATTICE_DB[key]
        if len(nbrs) == L['degree']:
            _, rmsd, _ = align_to_lattice(pos[v], pos[nbrs], key)
            best_rmsd = min(best_rmsd, rmsd)
    features.append(float(best_rmsd) if best_rmsd < np.inf else 5.0)

    # 6. Local normal estimate reliability (condition number of neighbor matrix)
    if len(nbrs) >= 3:
        offsets = pos[nbrs] - pos[v]
        sv = np.linalg.svd(offsets, compute_uv=False)
        # High ratio = flat/planar, low ratio = 3D spread
        planarity = sv[-1] / (sv[0] + 1e-9)
        features.append(float(planarity))
    else:
        features.append(0.0)

    return np.array(features, dtype=np.float32)


def neighborhood_feature_dim(lattice_keys: list = ['zphi', 'fcc', 'sc']) -> int:
    """Returns the dimension of the feature vector from compute_neighborhood_features."""
    # lattice_scores + spoke(2) + pairwise(2) + k2_size + best_rmsd + planarity
    return len(lattice_keys) + 2 + 2 + 1 + 1 + 1


# ---------------------------------------------------------------------------
# Full relaxer step combining spring + crystal forces
# ---------------------------------------------------------------------------

def full_relaxer_step(
    pos: np.ndarray,
    adj: dict,
    vertex_ring: dict,
    free_verts: Optional[list] = None,
    attract: float = 0.1,
    repel: float = 0.6,
    crystal_align: float = 0.03,
    crystal_snap: float = 0.01,
    lattice_key: str = 'zphi',
    crystal_threshold: float = 0.25,
    rep_cutoff: float = 1.5,
) -> np.ndarray:
    """
    One full step combining:
      1. Spring attraction along edges
      2. Non-adjacent repulsion (Rust-style)
      3. Crystal alignment force (pulls neighbors to lattice targets)
      4. Crystal snap force (pulls crystalline vertices toward each other)
    
    Crystal forces scale with crystallinity score -- zero when amorphous,
    dominant when nearly converged.
    """
    N = len(pos)
    if free_verts is None:
        free_verts = list(range(N))
    free_set = set(free_verts)

    adj_set  = {(min(u,v), max(u,v)) for v in range(N) for u in adj[v]}
    adj_arr  = np.array(list(adj_set))
    U, V     = adj_arr[:,0], adj_arr[:,1]
    free_mask = np.array([u in free_set or v in free_set for u,v in zip(U,V)])
    Uf, Vf   = U[free_mask], V[free_mask]

    delta = np.zeros((N, 3))

    # 1. Spring attraction
    d_vec  = pos[Vf] - pos[Uf]
    dist   = np.linalg.norm(d_vec, axis=1, keepdims=True).clip(min=1e-12)
    scalar = (1.0 - dist) * attract
    np.add.at(delta, Uf, -d_vec * scalar)
    np.add.at(delta, Vf,  d_vec * scalar)

    # 2. Non-adjacent repulsion
    tree   = cKDTree(pos)
    pairs  = tree.query_pairs(r=rep_cutoff, output_type='ndarray')
    if len(pairs):
        mask   = np.array([(min(int(a),int(b)),max(int(a),int(b))) not in adj_set
                           and (a in free_set or b in free_set)
                           for a,b in pairs])
        pairs  = pairs[mask]
        if len(pairs):
            Ur, Vr = pairs[:,0], pairs[:,1]
            dr_vec = pos[Vr] - pos[Ur]
            dr     = np.linalg.norm(dr_vec, axis=1, keepdims=True).clip(min=1e-12)
            close  = (dr < 1.0).flatten()
            if close.any():
                sc = (1.0 - dr[close]) * repel
                np.add.at(delta, Vr[close],  dr_vec[close] * sc)
                np.add.at(delta, Ur[close], -dr_vec[close] * sc)

    # 3+4. Crystal forces (only if any strength is nonzero)
    if crystal_align > 0 or crystal_snap > 0:
        cf = crystal_snap_forces(
            pos, adj, vertex_ring, lattice_key=lattice_key,
            score_threshold=crystal_threshold,
            snap_strength=crystal_snap,
            align_strength=crystal_align,
        )
        delta += cf

    # Zero out fixed vertices
    for v in range(N):
        if v not in free_set:
            delta[v] = 0.0

    return pos + delta


# ---------------------------------------------------------------------------
# Quick test / demo
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys, math
    from collections import defaultdict

    EDGE_LENGTH = 1.0

    def build_small():
        adj = defaultdict(set); vertex_ring = {}; _id=[0]
        def nv(r): v=_id[0];_id[0]+=1;vertex_ring[v]=r;return v
        center=nv(0); ring1=[nv(1) for _ in range(7)]
        for i in range(7):
            adj[center].add(ring1[i]); adj[ring1[i]].add(center)
            adj[ring1[i]].add(ring1[(i+1)%7]); adj[ring1[(i+1)%7]].add(ring1[i])
        return adj, vertex_ring, ring1

    adj, vertex_ring, ring1 = build_small()
    N = max(vertex_ring)+1
    pos = np.zeros((N,3))
    for i,v in enumerate(ring1):
        a = 2*math.pi*i/7
        pos[v] = [math.cos(a), math.sin(a), np.random.uniform(-0.1,0.1)]

    print("=== Crystal Neighborhood Test ===")
    print(f"N={N}, lattice_keys={list(LATTICE_DB.keys())}")
    print(f"Z[phi] spoke canonical: {np.round(ZPHI_SPOKE_CANONICAL,4)}")
    print(f"Z[phi] pairwise canonical (first 7): {np.round(ZPHI_PAIRWISE_CANONICAL[:7],4)}")
    print(f"Feature dim: {neighborhood_feature_dim()}")

    # Score center vertex (degree=7 but all neighbors at distance 1, not A/B/C mix)
    v = 0
    nbrs = np.array(list(adj[v]))
    score = crystallinity_score(pos[v], pos[nbrs], 'zphi')
    print(f"\nCrystallinity score of center (uniform ring, not lattice-like): {score:.4f}")

    # Feature vector
    feats = compute_neighborhood_features(v, pos, adj, vertex_ring)
    print(f"Feature vector: {np.round(feats,4)}")

    print("\nDone. Import this module and call full_relaxer_step() in your main relaxer.")
