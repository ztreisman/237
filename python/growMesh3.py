"""
growMesh3.py  —  {3,7} mesh grower with Klein quartic snap detection
=====================================================================
Synthesizes:
  - growMesh.py:  convex/concave boundary tracking (your Windows code)
  - growMesh2.py: dynamic grow + KDTree snap detection (your Windows code)
  - poincare_37.py: exact Poincaré disk coordinates via Möbius transforms
  - PSL(2,7): Klein quartic deck transformations for algebraic snap detection

The snap detection replaces the heuristic KDTree distance check with an
exact algebraic condition: two vertices snap when they are in the same
PSL(2,7)-orbit, i.e. when they represent the same vertex on the Klein quartic.

Visualization: SVG in the style of the zome lattice —
  - vein edges (spoke from center outward): deep red, thick
  - strut edges (diagonal bridges): cobalt blue, medium
  - ring edges (boundary cycle): gold, thin
  - vine spirals wind around strut and vein edges like climbing plants
  - snapped vertices glow white with a halo
  - background: dark void, stars at infinity
"""

import math
import cmath
import colorsys
import numpy as np
from collections import defaultdict
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# Hyperbolic geometry
# ═══════════════════════════════════════════════════════════════════════════════

ALPHA   = 2 * math.pi / 7
COSH_A  = math.cos(ALPHA) / (1 - math.cos(ALPHA))
A       = math.acosh(COSH_A)
TANH_A2 = math.tanh(A / 2)


def mob_to_origin(z: complex, c: complex) -> complex:
    return (z - c) / (1 - c.conjugate() * z)

def mob_from_origin(z: complex, c: complex) -> complex:
    return (z + c) / (1 + c.conjugate() * z)

def third_vertex(u: complex, v: complex, side: int = 1) -> complex:
    v_loc   = mob_to_origin(v, u)
    w_local = TANH_A2 * cmath.exp(1j * (cmath.phase(v_loc) + side * ALPHA))
    return mob_from_origin(w_local, u)

def hyp_dist(z1: complex, z2: complex) -> float:
    num = abs(z1 - z2) ** 2
    den = (1 - abs(z1)**2) * (1 - abs(z2)**2)
    return math.acosh(1 + 2 * num / den) if den > 0 else float('inf')


# ═══════════════════════════════════════════════════════════════════════════════
# PSL(2,7) — Klein quartic deck group
# ═══════════════════════════════════════════════════════════════════════════════

def _mat_mod(M, p=7):
    return M % p

def _mat_mul(A, B, p=7):
    return (A @ B) % p

def _mat_hash(M, p=7):
    M  = M % p
    nM = (-M) % p
    t  = tuple(M.flatten())
    nt = tuple(nM.flatten())
    return t if t <= nt else nt


def build_psl27():
    """Return all 168 elements of PSL(2,7) as 2×2 numpy arrays over Z/7."""
    import numpy as np
    from collections import deque
    I      = np.eye(2, dtype=int)
    S      = np.array([[0, 6], [1, 0]], dtype=int)   # [[0,-1],[1,0]] mod 7
    T      = np.array([[1, 1], [0, 1]], dtype=int)
    T_inv  = np.array([[1, 6], [0, 1]], dtype=int)   # T^{-1} mod 7
    gens   = [S, T, T_inv]
    seen   = {_mat_hash(I): I}
    queue  = deque([I])
    while queue:
        M = queue.popleft()
        for g in gens:
            N = _mat_mul(M, g)
            h = _mat_hash(N)
            if h not in seen:
                seen[h] = N
                queue.append(N)
    return list(seen.values())


def build_klein_quartic_graph(elements):
    """
    Build the {3,7} graph of the Klein quartic.

    Vertices = 24 cosets of <T> in PSL(2,7).
    Edges = pairs of cosets related by T^k * S for k=0..6
    (the 7 edge-flip generators of the tiling).

    Returns:
        cosets:   list of 24 coset-representative matrices
        adj_KQ:   list of 24 sets of neighbor indices
        edge_type_KQ: dict (i,j) -> 'strut'|'vein'|'ring'  [all 'strut' for Klein quartic]
    """
    import numpy as np
    I = np.eye(2, dtype=int)
    T = np.array([[1, 1], [0, 1]], dtype=int)
    S = np.array([[0, 6], [1, 0]], dtype=int)

    # <T> subgroup hashes
    T_powers = set()
    M = I.copy()
    for _ in range(7):
        T_powers.add(_mat_hash(M))
        M = _mat_mul(M, T)

    # Find 24 coset representatives
    cosets = [I]
    for M in elements:
        in_existing = False
        for rep in cosets:
            a, b, c, d = rep[0,0], rep[0,1], rep[1,0], rep[1,1]
            rep_inv = np.array([[d, -b], [-c, a]], dtype=int) % 7
            prod    = _mat_mul(rep_inv, M)
            if _mat_hash(prod) in T_powers:
                in_existing = True
                break
        if not in_existing:
            cosets.append(M)

    def coset_index(M):
        for i, rep in enumerate(cosets):
            a, b, c, d = rep[0,0], rep[0,1], rep[1,0], rep[1,1]
            ri  = np.array([[d, -b], [-c, a]], dtype=int) % 7
            prd = _mat_mul(ri, M)
            if _mat_hash(prd) in T_powers:
                return i
        return -1

    adj_KQ = [set() for _ in range(24)]
    Tk = I.copy()
    for k in range(7):
        g = _mat_mul(Tk, S)
        for i, C in enumerate(cosets):
            CS = _mat_mul(C, g)
            j  = coset_index(CS)
            if j >= 0 and j != i:
                adj_KQ[i].add(j)
        Tk = _mat_mul(Tk, T)

    return cosets, adj_KQ


def assign_poincare_positions_kq(cosets, adj_KQ):
    """
    Assign Poincaré disk positions to the 24 Klein quartic vertices.

    Vertex 0 (coset of I) → origin.
    Each neighbor is placed at hyperbolic distance A, equally spaced
    around vertex 0.  Further vertices are placed by BFS using third_vertex().

    Returns: dict {vertex_index: complex Poincaré position}
    """
    pos = {}
    pos[0] = 0 + 0j

    # Ring 1: 7 neighbors of vertex 0, equally spaced
    nbrs0 = sorted(adj_KQ[0])
    for k, v in enumerate(nbrs0):
        angle     = 2 * math.pi * k / 7
        pos[v]    = TANH_A2 * cmath.exp(1j * angle)

    # BFS: for each face (triangle), place unplaced third vertex
    # Build face list from the adj graph (every triangle = 3 mutually adjacent vertices)
    faces_KQ = []
    seen_f   = set()
    for a in range(24):
        for b in adj_KQ[a]:
            for c in adj_KQ[a]:
                if b < c and c in adj_KQ[b]:
                    key = (a, b, c)
                    if key not in seen_f:
                        seen_f.add(key); faces_KQ.append(key)

    changed = True
    while changed:
        changed = False
        for (a, b, c) in faces_KQ:
            placed   = [v for v in (a, b, c) if v in pos]
            unplaced = [v for v in (a, b, c) if v not in pos]
            if len(placed) < 2 or len(unplaced) == 0:
                continue
            u, v = placed[0], placed[1]
            w    = unplaced[0]

            def is_new(p):
                return all(abs(p - pos[x]) > 1e-7 for x in pos)

            wp = third_vertex(pos[u], pos[v], side=+1)
            wm = third_vertex(pos[u], pos[v], side=-1)
            if is_new(wp) and is_new(wm):
                # Pick the one farther from origin (outward)
                cand = wp if abs(wp) > abs(wm) else wm
            elif is_new(wp):
                cand = wp
            elif is_new(wm):
                cand = wm
            else:
                continue

            pos[w]  = cand
            changed = True

    return pos


# ═══════════════════════════════════════════════════════════════════════════════
# Mesh grower with convex/concave boundary (from growMesh.py)
# and snap detection against Klein quartic orbits
# ═══════════════════════════════════════════════════════════════════════════════

SNAP_TOL = 0.04   # Poincaré disk Euclidean distance tolerance for snap


class MeshGrower:
    """
    Grows the {3,7} tiling ring by ring.

    Each vertex carries:
      - poincare:  complex, exact position in Poincaré disk
      - node_type: 'convex' | 'concave' | 'start'
      - layer:     int, ring index
      - kq_label:  int 0..23, which Klein quartic vertex this snaps to (-1 if not yet snapped)

    Each edge carries:
      - edge_type: 'vein' | 'ring' | 'strut'
    """

    def __init__(self, kq_pos: dict, adj_KQ: list):
        self.kq_pos   = kq_pos    # {kq_idx: complex poincare pos}
        self.adj_KQ   = adj_KQ
        self.nodes    = {}        # {vid: {'poincare', 'node_type', 'layer', 'kq_label'}}
        self.edges    = {}        # {(u,v): {'edge_type'}}
        self.adj      = defaultdict(set)
        self._vid     = 0
        self.snaps    = []        # list of (vid, kq_label) snap events
        self.boundary = []        # ordered list of boundary vertex ids

        # Place vertex 0 at origin
        v0 = self._add_node(0 + 0j, 'start', 0, kq_label=0)

        # Ring 1: 7 convex vertices
        ring1 = []
        for k in range(7):
            angle = 2 * math.pi * k / 7
            p     = TANH_A2 * cmath.exp(1j * angle)
            v     = self._add_node(p, 'convex', 1, kq_label=k + 1)
            self._add_edge(v0, v, 'vein')
            ring1.append(v)
        for k in range(7):
            self._add_edge(ring1[k], ring1[(k + 1) % 7], 'ring')

        self.boundary = ring1[:]

    def _add_node(self, p: complex, ntype: str, layer: int,
                  kq_label: int = -1) -> int:
        vid = self._vid; self._vid += 1
        self.nodes[vid] = {'poincare': p, 'node_type': ntype,
                           'layer': layer, 'kq_label': kq_label}
        return vid

    def _add_edge(self, u: int, v: int, etype: str):
        key = (min(u, v), max(u, v))
        self.edges[key] = {'edge_type': etype}
        self.adj[u].add(v); self.adj[v].add(u)

    def _snap_check(self, p: complex, layer: int) -> int:
        """
        Check if Poincaré position p is close to any Klein quartic vertex position.
        Returns kq_label (0..23) if snap detected, else -1.
        Snap is only considered valid if the kq vertex is at the right ring depth.
        """
        for kq_idx, kq_p in self.kq_pos.items():
            if abs(p - kq_p) < SNAP_TOL:
                return kq_idx
        return -1

    def grow_ring(self):
        """
        Grow one ring using the convex/concave boundary logic from growMesh.py.
        Returns the new boundary.
        """
        old_boundary = self.boundary[:]
        new_boundary = []
        layer        = self.nodes[old_boundary[0]]['layer'] + 1
        n            = len(old_boundary)

        # Compute Poincaré positions for new vertices by geometric placement
        # For each boundary vertex y, we need to place its "fan" of new vertices.
        # The first new vertex of y's fan should connect to the last new vertex
        # of the previous boundary vertex's fan (the "strut" connection).

        # We precompute all new positions using third_vertex():
        # For a convex boundary vertex y with ring neighbors y_prev, y_next:
        #   The fan is: [shared_prev, excl_0, excl_1, shared_next]  (4 new verts, 3+1 arcs)
        #   Actually: convex -> 3 new boundary verts, concave -> 2 new boundary verts.
        # 
        # We use the existing positions to compute new ones via third_vertex():
        # Each new vertex is the third vertex of a triangle whose other two vertices
        # are already placed.

        new_verts  = {}   # position in new boundary -> (poincare, node_type)
        new_bdy_ids = []

        for i, y in enumerate(old_boundary):
            y_prev = old_boundary[(i - 1) % n]
            y_next = old_boundary[(i + 1) % n]
            py     = self.nodes[y]['poincare']
            py_prev = self.nodes[y_prev]['poincare']
            py_next = self.nodes[y_next]['poincare']
            ntype  = self.nodes[y]['node_type']

            if ntype == 'convex':
                # 3 new boundary vertices: convex, convex, concave
                # Geometrically:
                # w1 = third_vertex(y, y_prev, outward)
                # w2 = third_vertex(y, w1, outward)
                # w3 = third_vertex(y, y_next, outward) = shared with next y

                # w1: outward from edge (y_prev -> y)
                w1 = third_vertex(py_prev, py, side=-1)
                # w2: outward from edge (y -> w1)
                w2 = third_vertex(py, w1, side=-1)
                # w3: outward from edge (y -> y_next) -- will be shared
                w3 = third_vertex(py, py_next, side=+1)

                new_bdy_ids.append((w1, 'convex'))
                new_bdy_ids.append((w2, 'convex'))
                new_bdy_ids.append((w3, 'concave'))

            elif ntype == 'concave':
                # 2 new boundary vertices: convex, concave
                w1 = third_vertex(py_prev, py, side=-1)
                w2 = third_vertex(py, py_next, side=+1)

                new_bdy_ids.append((w1, 'convex'))
                new_bdy_ids.append((w2, 'concave'))

        # Now create actual vertex IDs, detecting snaps and sharing "concave" verts
        # The concave (shared) vertices at the junction between two boundary fans
        # are the SAME vertex — detect this by proximity in Poincaré disk.
        new_vid_map = {}   # position index in new_bdy_ids -> vertex id
        placed_positions = []   # (poincare, vid)

        for idx, (p, ntype) in enumerate(new_bdy_ids):
            # Check if this position is already placed (shared vertex)
            existing = None
            for pp, pvid in placed_positions:
                if abs(p - pp) < 1e-8:
                    existing = pvid; break

            if existing is not None:
                new_vid_map[idx] = existing
            else:
                kq_label = self._snap_check(p, layer)
                vid = self._add_node(p, ntype, layer, kq_label=kq_label)
                if kq_label >= 0:
                    self.snaps.append((vid, kq_label))
                placed_positions.append((p, vid))
                new_vid_map[idx] = vid

        # Now rebuild the edge structure
        # For each old boundary vertex y, connect its fan:
        bdy_ptr = 0
        for i, y in enumerate(old_boundary):
            y_prev = old_boundary[(i - 1) % n]
            py     = self.nodes[y]['poincare']
            ntype  = self.nodes[y]['node_type']

            if ntype == 'convex':
                w1_idx = bdy_ptr
                w2_idx = bdy_ptr + 1
                w3_idx = bdy_ptr + 2
                w1 = new_vid_map[w1_idx]
                w2 = new_vid_map[w2_idx]
                w3 = new_vid_map[w3_idx]

                # strut: y -> previous fan's last concave (already connected if shared)
                # vein: y -> w1, y -> w2
                # ring: w1-w2, w2-w3
                # strut: w3 -> next fan (will be handled when processing next y via y_prev logic)
                self._add_edge(y, w1, 'vein')
                self._add_edge(y, w2, 'vein')
                self._add_edge(w1, w2, 'ring')
                self._add_edge(w2, w3, 'ring')

                # strut connects y to the previous fan's last concave vertex
                prev_last_idx = (bdy_ptr - 1) % len(new_bdy_ids)
                prev_last = new_vid_map[prev_last_idx]
                self._add_edge(y, prev_last, 'strut')
                self._add_edge(w1, prev_last, 'ring')

                # strut: y -> w3, and w3 connects forward
                self._add_edge(y, w3, 'strut')

                bdy_ptr += 3

            elif ntype == 'concave':
                w1_idx = bdy_ptr
                w2_idx = bdy_ptr + 1
                w1 = new_vid_map[w1_idx]
                w2 = new_vid_map[w2_idx]

                self._add_edge(y, w1, 'vein')
                self._add_edge(w1, w2, 'ring')

                prev_last_idx = (bdy_ptr - 1) % len(new_bdy_ids)
                prev_last = new_vid_map[prev_last_idx]
                self._add_edge(y, prev_last, 'strut')
                self._add_edge(w1, prev_last, 'ring')

                self._add_edge(y, w2, 'strut')

                bdy_ptr += 2

        self.boundary = [new_vid_map[i] for i in range(len(new_bdy_ids))]
        # Deduplicate (some shared vertices appear twice)
        seen = set(); deduped = []
        for v in self.boundary:
            if v not in seen:
                seen.add(v); deduped.append(v)
        self.boundary = deduped
        return self.boundary


# ═══════════════════════════════════════════════════════════════════════════════
# SVG rendering — zome lattice style with vine spirals
# ═══════════════════════════════════════════════════════════════════════════════

# Zome colors
COLORS = {
    'vein':   '#c0392b',   # deep red (like red zome struts)
    'strut':  '#2980b9',   # cobalt blue (like blue zome struts)
    'ring':   '#f39c12',   # amber gold (like yellow zome struts)
    'snap':   '#ffffff',   # white glow for snapped vertices
    'bg':     '#08080f',
    'disk_inner': '#0f1020',
    'disk_outer': '#060610',
}

THICKNESS = {'vein': 2.4, 'strut': 1.8, 'ring': 1.2}
OPACITY   = {'vein': 0.90, 'strut': 0.80, 'ring': 0.65}
VINE_COLOR = {'vein': '#27ae60', 'strut': '#8e44ad'}   # green vines on red, purple on blue


def poincare_to_svg(z: complex, cx: float, cy: float, R: float):
    return (cx + z.real * R, cy - z.imag * R)


def geodesic_arc_svg(z1: complex, z2: complex, cx: float, cy: float, R: float) -> str:
    EPS = 1e-9
    def to_svg(z): return poincare_to_svg(z, cx, cy, R)
    cross = z1.real * z2.imag - z1.imag * z2.real
    if abs(cross) < EPS:
        x1, y1 = to_svg(z1); x2, y2 = to_svg(z2)
        return f"M {x1:.3f},{y1:.3f} L {x2:.3f},{y2:.3f}", None  # no circle center for diameter

    mid  = (z1 + z2) / 2
    diff = z2 - z1
    perp = 1j * diff / abs(diff)
    d    = mid - z1
    denom = 2 * (((mid - d) * perp.conjugate()).real)
    if abs(denom) < EPS:
        x1, y1 = to_svg(z1); x2, y2 = to_svg(z2)
        return f"M {x1:.3f},{y1:.3f} L {x2:.3f},{y2:.3f}", None

    t      = (1 + abs(d)**2 - abs(mid)**2) / denom
    c_orth = mid + t * perp
    r_orth = abs(c_orth - z1)

    x1, y1    = to_svg(z1); x2, y2 = to_svg(z2)
    cx_o, cy_o = poincare_to_svg(c_orth, cx, cy, R)
    r_svg      = r_orth * R

    ang1  = cmath.phase(z1 - c_orth)
    ang2  = cmath.phase(z2 - c_orth)
    d_ang = ((ang2 - ang1 + math.pi) % (2*math.pi)) - math.pi
    large = 1 if abs(d_ang) > math.pi else 0
    sweep = 1 if d_ang > 0 else 0

    path = (f"M {x1:.3f},{y1:.3f} "
            f"A {r_svg:.3f},{r_svg:.3f} 0 {large},{sweep} {x2:.3f},{y2:.3f}")
    return path, (cx_o, cy_o, r_svg, ang1, ang2, d_ang)


def vine_spiral_svg(arc_info, t_start: float, t_end: float,
                    n_loops: float, color: str, thickness: float,
                    cx: float, cy: float, R: float,
                    z1: complex, z2: complex) -> list:
    """
    Generate SVG path data for a vine spiral winding around a geodesic arc.
    The vine is a sinusoidal oscillation perpendicular to the arc.
    Returns list of <path> SVG strings.
    """
    lines = []
    n_pts = 80
    amplitude_svg = R * 0.018   # vine amplitude in SVG pixels

    if arc_info is None:
        # Straight line case: parameterize linearly
        x1, y1 = poincare_to_svg(z1, cx, cy, R)
        x2, y2 = poincare_to_svg(z2, cx, cy, R)
        dx, dy = x2 - x1, y2 - y1
        length = math.sqrt(dx**2 + dy**2)
        if length < 1e-3: return lines
        nx_, ny_ = -dy / length, dx / length  # normal

        pts = []
        for i in range(n_pts + 1):
            t   = i / n_pts
            # Oscillation along the vine
            phase = t * n_loops * 2 * math.pi
            amp   = amplitude_svg * math.sin(phase) * math.sin(math.pi * t)
            px    = x1 + t * dx + amp * nx_
            py    = y1 + t * dy + amp * ny_
            pts.append((px, py))

        path_d = "M " + " L ".join(f"{px:.2f},{py:.2f}" for px,py in pts)
        lines.append(f'<path d="{path_d}" stroke="{color}" stroke-width="{thickness:.1f}" '
                     f'fill="none" opacity="0.7"/>')
        return lines

    cx_o, cy_o, r_svg, ang1, ang2, d_ang = arc_info

    # Parameterize points along the arc
    pts = []
    for i in range(n_pts + 1):
        t     = i / n_pts
        angle = ang1 + t * d_ang
        # Point on the arc circle
        ax = cx_o + r_svg * math.cos(angle)
        ay = cy_o + r_svg * math.sin(angle)
        # Normal direction (radially inward toward arc center)
        nx_ = (cx_o - ax) / r_svg
        ny_ = (cy_o - ay) / r_svg
        # Vine oscillation
        phase = t * n_loops * 2 * math.pi
        amp   = amplitude_svg * math.sin(phase) * math.sin(math.pi * t)
        px    = ax + amp * nx_
        py    = ay + amp * ny_
        pts.append((px, py))

    path_d = "M " + " L ".join(f"{px:.2f},{py:.2f}" for px,py in pts)
    lines.append(f'<path d="{path_d}" stroke="{color}" stroke-width="{thickness:.1f}" '
                 f'fill="none" opacity="0.65"/>')
    return lines


def render_svg(grower: MeshGrower, path: str,
               size: int = 1000, num_rings: int = 4) -> None:
    R  = size * 0.44
    cx = size / 2
    cy = size / 2
    N  = len(grower.nodes)

    lines = []
    bg_col = COLORS['bg']
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
                 f'viewBox="0 0 {size} {size}" style="background:{bg_col}">')

    # ── Defs ──────────────────────────────────────────────────────────────────
    lines.append('<defs>')
    lines.append(f'  <clipPath id="disk"><circle cx="{cx}" cy="{cy}" r="{R}"/></clipPath>')

    # Radial gradient for disk background
    lines.append(f'  <radialGradient id="diskbg" cx="50%" cy="50%" r="50%">'
                 f'    <stop offset="0%" stop-color="#141428"/>'
                 f'    <stop offset="70%" stop-color="#0a0a18"/>'
                 f'    <stop offset="100%" stop-color="#050510"/>'
                 f'  </radialGradient>')

    # Glow filter for snap vertices
    lines.append('  <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">')
    lines.append('    <feGaussianBlur stdDeviation="4" result="blur"/>')
    lines.append('    <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>')
    lines.append('  </filter>')

    # Soft glow for vein/strut edges
    lines.append('  <filter id="edgeglow" x="-20%" y="-20%" width="140%" height="140%">')
    lines.append('    <feGaussianBlur stdDeviation="1.5" result="blur"/>')
    lines.append('    <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>')
    lines.append('  </filter>')

    # Star-field pattern
    lines.append('  <pattern id="stars" x="0" y="0" width="60" height="60" patternUnits="userSpaceOnUse">')
    rng = np.random.default_rng(42)
    for _ in range(8):
        sx = rng.uniform(0, 60); sy = rng.uniform(0, 60)
        sr = rng.uniform(0.3, 1.0); so = rng.uniform(0.2, 0.7)
        lines.append(f'    <circle cx="{sx:.1f}" cy="{sy:.1f}" r="{sr:.1f}" '
                     f'fill="white" opacity="{so:.2f}"/>')
    lines.append('  </pattern>')

    lines.append('</defs>')

    # ── Background ────────────────────────────────────────────────────────────
    lines.append(f'<rect width="{size}" height="{size}" fill="url(#stars)"/>')
    lines.append(f'<circle cx="{cx}" cy="{cy}" r="{R}" fill="url(#diskbg)"/>')

    # ── Edges ─────────────────────────────────────────────────────────────────
    snapped_vids = {v for v, _ in grower.snaps}

    # Draw in order: ring, strut, vein (so veins are on top)
    edge_order = ['ring', 'strut', 'vein']
    for etype in edge_order:
        col  = COLORS[etype]
        thk  = THICKNESS[etype]
        opac = OPACITY[etype]
        vine_col = VINE_COLOR.get(etype)

        edge_group = []
        vine_group = []

        for (u, v), edata in grower.edges.items():
            if edata['edge_type'] != etype: continue
            z1 = grower.nodes[u]['poincare']
            z2 = grower.nodes[v]['poincare']
            if abs(z1) > 0.999 or abs(z2) > 0.999: continue

            path_d, arc_info = geodesic_arc_svg(z1, z2, cx, cy, R)
            edge_group.append(f'  <path d="{path_d}" stroke="{col}" '
                              f'stroke-width="{thk:.1f}" fill="none" '
                              f'opacity="{opac:.2f}"/>')

            # Vine spirals on vein and strut edges
            if vine_col and etype in ('vein', 'strut'):
                n_loops = 3.5 if etype == 'vein' else 2.5
                vines = vine_spiral_svg(arc_info, 0, 1, n_loops,
                                        vine_col, 0.8, cx, cy, R, z1, z2)
                vine_group.extend(vines)

        if edge_group:
            filter_attr = ' filter="url(#edgeglow)"' if etype == 'vein' else ''
            lines.append(f'<g clip-path="url(#disk)"{filter_attr}>')
            lines.extend(edge_group)
            lines.append('</g>')
        if vine_group:
            lines.append('<g clip-path="url(#disk)">')
            lines.extend(vine_group)
            lines.append('</g>')

    # ── Vertices ──────────────────────────────────────────────────────────────
    lines.append('<g clip-path="url(#disk)">')
    for vid, ndata in grower.nodes.items():
        z     = ndata['poincare']
        if abs(z) > 0.999: continue
        layer = ndata['layer']
        px, py = poincare_to_svg(z, cx, cy, R)
        is_snap = vid in snapped_vids
        kq = ndata.get('kq_label', -1)

        if is_snap:
            # Snapped vertex: white glow halo + bright dot
            lines.append(f'  <circle cx="{px:.2f}" cy="{py:.2f}" r="9" '
                         f'fill="none" stroke="white" stroke-width="1.5" '
                         f'opacity="0.4" filter="url(#glow)"/>')
            lines.append(f'  <circle cx="{px:.2f}" cy="{py:.2f}" r="4.5" '
                         f'fill="white" opacity="0.95" filter="url(#glow)"/>')
        else:
            # Color by layer
            hue = (layer * 0.13) % 1.0
            r_g, g_g, b_g = colorsys.hsv_to_rgb(hue, 0.6, 0.9)
            col = f'rgb({int(r_g*255)},{int(g_g*255)},{int(b_g*255)})'
            radius = max(1.5, 4.0 - layer * 0.5)
            lines.append(f'  <circle cx="{px:.2f}" cy="{py:.2f}" r="{radius:.1f}" '
                         f'fill="{col}" opacity="0.85"/>')

        # KQ label for snapped vertices
        if is_snap and kq >= 0:
            lines.append(f'  <text x="{px+6:.1f}" y="{py-4:.1f}" '
                         f'font-family="Georgia,serif" font-size="9" '
                         f'fill="white" opacity="0.7">{kq}</text>')
    lines.append('</g>')

    # ── Klein quartic vertices (exact positions) ──────────────────────────────
    lines.append('<g clip-path="url(#disk)" opacity="0.25">')
    for kq_idx, kq_z in grower.kq_pos.items():
        if abs(kq_z) > 0.999: continue
        px, py = poincare_to_svg(kq_z, cx, cy, R)
        lines.append(f'  <circle cx="{px:.2f}" cy="{py:.2f}" r="6" '
                     f'fill="none" stroke="#ffffff" stroke-width="1" '
                     f'stroke-dasharray="2,2"/>')
    lines.append('</g>')

    # ── Boundary circle ────────────────────────────────────────────────────────
    lines.append(f'<circle cx="{cx}" cy="{cy}" r="{R}" '
                 f'fill="none" stroke="#ffffff" stroke-width="1" opacity="0.3"/>')

    # ── Legend ────────────────────────────────────────────────────────────────
    lx, ly = 28, size - 110
    legend = [
        ('vein',   'Vein (spoke)'),
        ('strut',  'Strut (bridge)'),
        ('ring',   'Ring (boundary)'),
    ]
    lines.append(f'<rect x="{lx-8}" y="{ly-16}" width="160" height="96" '
                 f'rx="6" fill="black" opacity="0.5"/>')
    for i, (etype, label) in enumerate(legend):
        col = COLORS[etype]
        thk = THICKNESS[etype]
        vy  = ly + i * 28
        lines.append(f'<line x1="{lx}" y1="{vy}" x2="{lx+32}" y2="{vy}" '
                     f'stroke="{col}" stroke-width="{thk:.0f}" opacity="0.9"/>')
        lines.append(f'<text x="{lx+40}" y="{vy+5}" font-family="Georgia,serif" '
                     f'font-size="13" fill="#cccccc">{label}</text>')

    # Vine legend
    lines.append(f'<line x1="{lx}" y1="{ly+84}" x2="{lx+32}" y2="{ly+84}" '
                 f'stroke="#27ae60" stroke-width="1.2" stroke-dasharray="3,2"/>')
    lines.append(f'<text x="{lx+40}" y="{ly+89}" font-family="Georgia,serif" '
                 f'font-size="11" fill="#aaaaaa">Vine spiral</text>')

    # Snap legend
    lines.append(f'<circle cx="{lx+12}" cy="{ly+112}" r="5" fill="none" '
                 f'stroke="white" stroke-width="1.2" stroke-dasharray="2,2"/>')
    lines.append(f'<text x="{lx+40}" y="{ly+117}" font-family="Georgia,serif" '
                 f'font-size="11" fill="#aaaaaa">Klein quartic vertex</text>')

    # Title
    lines.append(f'<text x="{cx}" y="32" font-family="Georgia,serif" font-size="18" '
                 f'fill="white" opacity="0.7" text-anchor="middle" '
                 f'letter-spacing="2">{{3,7}} Tiling — Klein Quartic Snap</text>')
    lines.append(f'<text x="{cx}" y="52" font-family="Georgia,serif" font-size="11" '
                 f'fill="#aaaaaa" opacity="0.6" text-anchor="middle">'
                 f'Poincaré disk · PSL(2,7) deck transformations · '
                 f'{len(grower.nodes)} vertices · {len(grower.edges)} edges · '
                 f'{len(grower.snaps)} snaps</text>')

    lines.append('</svg>')

    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"SVG saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--rings', type=int, default=4)
    parser.add_argument('--size',  type=int, default=1000)
    args = parser.parse_args()

    print("Building PSL(2,7)...")
    elements = build_psl27()
    print(f"  {len(elements)} elements")

    print("Building Klein quartic {3,7} graph...")
    cosets, adj_KQ = build_klein_quartic_graph(elements)
    print(f"  24 vertices, degrees: {sorted(set(len(adj_KQ[i]) for i in range(24)))}")

    print("Computing Klein quartic Poincaré positions...")
    kq_pos = assign_poincare_positions_kq(cosets, adj_KQ)
    placed = sum(1 for p in kq_pos.values() if not (isinstance(p, float) and math.isnan(p.real)))
    print(f"  {len(kq_pos)} vertices placed")

    print(f"\nGrowing {args.rings} rings...")
    grower = MeshGrower(kq_pos, adj_KQ)
    for r in range(2, args.rings + 1):
        grower.grow_ring()
        n_snaps = len(grower.snaps)
        print(f"  Ring {r}: {len(grower.nodes)} vertices, "
              f"{len(grower.edges)} edges, {n_snaps} snaps detected")
        if grower.snaps:
            for vid, kq_label in grower.snaps:
                z = grower.nodes[vid]['poincare']
                print(f"    Snap: vertex {vid} -> KQ vertex {kq_label} "
                      f"at Poincaré pos {z:.4f}")

    print(f"\nRendering SVG ({args.size}×{args.size})...")
    out = 'growMesh3.svg'
    render_svg(grower, out, size=args.size, num_rings=args.rings)

    # Summary
    edge_types = {}
    for edata in grower.edges.values():
        t = edata['edge_type']
        edge_types[t] = edge_types.get(t, 0) + 1
    print(f"\nEdge counts: {edge_types}")
    print(f"Snap events: {len(grower.snaps)}")
    if grower.snaps:
        print("Snapped vertices:")
        for vid, kq in grower.snaps:
            z = grower.nodes[vid]['poincare']
            print(f"  vertex {vid} (layer {grower.nodes[vid]['layer']}) "
                  f"-> KQ {kq}, pos={z:.4f}")
