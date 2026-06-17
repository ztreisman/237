"""
Poincare Disk -- exact {3,7} tiling
====================================
Computes exact hyperbolic positions for every vertex of the {3,7} tiling
in the Poincare disk model using Mobius transformations.

The key formula: for an equilateral hyperbolic triangle with all vertex
angles alpha = 2pi/7 (since 7 triangles meet at each vertex), the hyperbolic
side length a satisfies:

    cosh(a) = cos(alpha) / (1 - cos(alpha))

Vertex positions in the Poincare disk are complex numbers z with |z| < 1,
where Euclidean distance |z| = tanh(d/2) for hyperbolic distance d from origin.

Mobius transformations (Mobius group PSL(2,R)) act as isometries of the
hyperbolic plane and are used to place each new triangle exactly.

Outputs
-------
  poincare_37.svg     -- exact tiling rendered as SVG, colored by ring
  poincare_coords.npy -- (N, 2) array of Poincare coordinates per vertex,
                        in the same vertex order as build_combinatorial()
                        -> use as additional node features for the GNN

Usage
-----
  python poincare_37.py [--rings N]  (default 4)

The coordinates in poincare_coords.npy can be loaded in the main GNN script:
  coords = np.load('poincare_coords.npy')   # shape (N, 2)
  # Add as extra node features alongside pos_xyz, ring_norm, etc.
"""

import math
from typing import Dict, List, Set, Tuple, Optional
import cmath
import argparse
import numpy as np
from collections import defaultdict
from pathlib import Path

# ===============================================================================
# Hyperbolic geometry constants for the {3,7} tiling
# ===============================================================================

ALPHA   = 2 * math.pi / 7          # vertex angle (7 triangles per vertex)
COSH_A  = math.cos(ALPHA) / (1 - math.cos(ALPHA))
A       = math.acosh(COSH_A)       # hyperbolic edge length
TANH_A2 = math.tanh(A / 2)         # Poincare disk radius for one edge length


# ===============================================================================
# Mobius transformation utilities
# ===============================================================================

def mob_translate(z: complex, w: complex) -> complex:
    """
    Hyperbolic translation: move point z by the displacement w in the
    Poincare disk. w is a complex number with |w| < 1 encoding direction
    and distance (|w| = tanh(d/2) for hyperbolic distance d).

    This is the standard Mobius transformation for the Poincare disk:
        T_w(z) = (z + w) / (1 + conj(w) * z)
    """
    return (z + w) / (1 + w.conjugate() * z)


def mob_to_origin(z: complex, center: complex) -> complex:
    """Map the Poincare disk so that `center` goes to the origin."""
    return (z - center) / (1 - center.conjugate() * z)


def mob_from_origin(z: complex, center: complex) -> complex:
    """Inverse: map origin back to `center`."""
    return (z + center) / (1 + center.conjugate() * z)


def hyp_dist(z1: complex, z2: complex) -> float:
    """Hyperbolic distance between two Poincare disk points."""
    num = abs(z1 - z2) ** 2
    den = (1 - abs(z1)**2) * (1 - abs(z2)**2)
    if den <= 0:
        return float('inf')
    return math.acosh(1 + 2 * num / den)


def third_vertex(u: complex, v: complex, side: int = 1) -> complex:
    """
    Given two adjacent vertices u, v of the {3,7} tiling (one edge of an
    equilateral hyperbolic triangle), return the third vertex on the specified
    side (+1 = left of the directed edge u->v, -1 = right).

    Method: translate so u is at the origin, rotate so v is on the positive
    real axis, place the third vertex at angle +-ALPHA from that axis at
    hyperbolic distance A, then undo the translation and rotation.
    """
    # Map u to origin
    v_local = mob_to_origin(v, u)
    angle_v  = cmath.phase(v_local)

    # Third vertex: same hyperbolic distance A from origin, at angle +-ALPHA
    w_angle = angle_v + side * ALPHA
    w_local = TANH_A2 * cmath.exp(1j * w_angle)

    # Map back to original frame
    return mob_from_origin(w_local, u)


# ===============================================================================
# Build combinatorial structure (same as main GNN script)
# ===============================================================================

def build_combinatorial(num_rings: int) -> Tuple:
    adj = defaultdict(set)
    vertex_ring: Dict[int, int] = {}
    faces = []
    seen_faces: Set[frozenset] = set()
    _id = [0]

    def new_v(r):
        v = _id[0]; _id[0] += 1
        vertex_ring[v] = r
        return v

    def add_face(a, b, c):
        key = frozenset([a, b, c])
        if key in seen_faces: return
        seen_faces.add(key); faces.append((a, b, c))
        for u, w in [(a,b),(b,c),(a,c)]:
            adj[u].add(w); adj[w].add(u)

    center = new_v(0)
    ring1  = [new_v(1) for _ in range(7)]
    for i in range(7):
        add_face(center, ring1[i], ring1[(i+1) % 7])
    boundary = {0: [center], 1: ring1}

    for r in range(2, num_rings + 1):
        prev   = boundary[r-1]
        n      = len(prev)
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
            new_bdy.append(shared[(i-1)%n])
            new_bdy.extend(exclusive[i])
        boundary[r] = new_bdy

    return adj, vertex_ring, faces, boundary


# ===============================================================================
# Compute exact Poincare coordinates for every vertex
# ===============================================================================

def compute_poincare_coords(adj: dict, vertex_ring: dict,
                            faces: list, boundary: dict) -> np.ndarray:
    """
    Assign exact Poincare disk coordinates to every vertex.

    Strategy: BFS from the center, placing each new vertex as the
    third vertex of an equilateral hyperbolic triangle given two
    already-placed neighbors.

    For each face (a, b, c), if two vertices are placed and one is not,
    place the unplaced one using third_vertex(). The orientation (+1/-1)
    is chosen to keep the new vertex on the correct side -- we pick the
    side that is farther from the center (outward).

    Returns an (N, 2) float32 array of (Re(z), Im(z)) Poincare coordinates.
    """
    N   = max(vertex_ring) + 1
    pos = np.full((N, 2), np.nan, dtype=np.float64)

    # Place center at origin
    center_v = [v for v, r in vertex_ring.items() if r == 0][0]
    pos[center_v] = [0.0, 0.0]

    # Place ring 1: evenly spaced around origin
    ring1 = boundary[1]
    for i, v in enumerate(ring1):
        angle = 2 * math.pi * i / 7
        pos[v] = [TANH_A2 * math.cos(angle), TANH_A2 * math.sin(angle)]

    # BFS: for each face with exactly 2 placed vertices, place the third
    placed  = set(v for v in range(N) if not np.isnan(pos[v, 0]))
    changed = True
    while changed:
        changed = False
        for (a, b, c) in faces:
            verts    = [a, b, c]
            is_placed = [v in placed for v in verts]
            n_placed  = sum(is_placed)
            if n_placed == 3:
                continue
            if n_placed < 2:
                continue
            # Exactly 2 placed -- find which is unplaced
            unplaced_idx = is_placed.index(False)
            placed_pair  = [verts[i] for i in range(3) if i != unplaced_idx]
            u_idx, v_idx = placed_pair
            unplaced     = verts[unplaced_idx]

            u_c = complex(*pos[u_idx])
            v_c = complex(*pos[v_idx])

            # Try both sides, pick the one farther from origin (outward)
            w_plus  = third_vertex(u_c, v_c, side=+1)
            w_minus = third_vertex(u_c, v_c, side=-1)

            # Verify distances -- should both be at distance A from u and v
            # Pick the one that is NOT already placed (not coinciding with
            # an existing vertex) and is farther from origin if ambiguous
            def is_new(w):
                for p in placed:
                    if abs(w - complex(*pos[p])) < 1e-8:
                        return False
                return True

            if is_new(w_plus) and is_new(w_minus):
                # Both new -- pick the one farther from origin (outward)
                w = w_plus if abs(w_plus) > abs(w_minus) else w_minus
            elif is_new(w_plus):
                w = w_plus
            elif is_new(w_minus):
                w = w_minus
            else:
                # Both coincide with existing vertices -- this face is done
                continue

            pos[unplaced] = [w.real, w.imag]
            placed.add(unplaced)
            changed = True

    # Verify: all vertices should be placed
    unplaced = [v for v in range(N) if np.isnan(pos[v, 0])]
    if unplaced:
        print(f"WARNING: {len(unplaced)} vertices could not be placed: {unplaced[:5]}")

    # Verify edge lengths
    placed_set = set(v for v in range(N) if not np.isnan(pos[v, 0]))
    lengths = []
    seen_e  = set()
    for v in placed_set:
        for u in adj[v]:
            key = (min(u,v), max(u,v))
            if key in seen_e or u not in placed_set: continue
            seen_e.add(key)
            d = hyp_dist(complex(*pos[v]), complex(*pos[u]))
            lengths.append(d)
    lengths = np.array(lengths)
    print(f"Placed {len(placed_set)}/{N} vertices")
    print(f"Edge lengths: mean={lengths.mean():.6f} std={lengths.std():.8f} "
          f"(should be {A:.6f}, std~=0)")

    return pos.astype(np.float32)


# ===============================================================================
# SVG rendering
# ===============================================================================

# Ring colors (same palette as main GNN script)
_RING_COLORS = [
    '#ffffff',  # 0 -- white
    '#fde725',  # 1 -- yellow
    '#7ad151',  # 2 -- green
    '#22a884',  # 3 -- teal
    '#2a778e',  # 4 -- blue-teal
    '#414487',  # 5 -- indigo
    '#440154',  # 6 -- purple
    '#b43232',  # 7+ -- red
]

def _ring_color(r: int) -> str:
    return _RING_COLORS[min(r, len(_RING_COLORS) - 1)]


def poincare_to_svg(z: complex, cx: float, cy: float, R: float) -> Tuple:
    """Convert Poincare disk coordinate to SVG pixel coordinate."""
    return (cx + z.real * R, cy - z.imag * R)


def geodesic_arc_svg(z1: complex, z2: complex,
                     cx: float, cy: float, R: float) -> str:
    """
    Return an SVG path for the hyperbolic geodesic between z1 and z2
    in the Poincare disk.

    Geodesics in the Poincare disk are either:
      a) Diameters (if z1, z2, and origin are collinear)
      b) Circular arcs perpendicular to the boundary circle

    For case (b), we find the center and radius of the orthogonal circle,
    then output an SVG arc.
    """
    EPS = 1e-9

    def to_svg(z):
        return poincare_to_svg(z, cx, cy, R)

    # Check if the geodesic passes through (or near) the origin
    # i.e., if z1/z2 are antipodal or collinear with 0
    cross = z1.real * z2.imag - z1.imag * z2.real
    if abs(cross) < EPS:
        # Diameter -- straight line
        x1, y1 = to_svg(z1)
        x2, y2 = to_svg(z2)
        return f"M {x1:.3f},{y1:.3f} L {x2:.3f},{y2:.3f}"

    # Find the circle orthogonal to the unit disk passing through z1 and z2.
    # Its center c_orth lies on the perpendicular bisector of z1,z2
    # AND on the real/imaginary extended line satisfying |c|^2 = 1 + r^2
    # (orthogonality condition with unit circle).
    #
    # Let c = (cx_orth, cy_orth). Then:
    #   |c - z1|^2 = |c - z2|^2   (equidistant from z1 and z2)
    #   |c|^2 = 1 + |c - z1|^2   (orthogonal to unit circle)
    #
    # From the first condition: c lies on the perpendicular bisector of z1,z2.
    # Parameterise: c = mid + t * perp  where mid=(z1+z2)/2, perp=i*(z2-z1)/|z2-z1|

    mid  = (z1 + z2) / 2
    diff = z2 - z1
    perp = 1j * diff / abs(diff)   # unit perpendicular

    # |c|^2 = 1 + |c-z1|^2 gives:
    # |mid + t*perp|^2 = 1 + |mid + t*perp - z1|^2
    # Let m = mid, p = perp, d = mid - z1
    # |m|^2+2t Re(m conj(p))+t^2 = 1 + |d|^2+2t Re(d conj(p))+t^2
    # 2t Re((m-d) conj(p)) = 1 + |d|^2 - |m|^2
    # t = (1 + |d|^2 - |m|^2) / (2 Re((m-d) conj(p)))
    d     = mid - z1
    denom = 2 * (((mid - d) * perp.conjugate()).real)
    if abs(denom) < EPS:
        # Degenerate -- use straight line
        x1, y1 = to_svg(z1)
        x2, y2 = to_svg(z2)
        return f"M {x1:.3f},{y1:.3f} L {x2:.3f},{y2:.3f}"

    t       = (1 + abs(d)**2 - abs(mid)**2) / denom
    c_orth  = mid + t * perp
    r_orth  = abs(c_orth - z1)

    # Convert to SVG
    x1, y1    = to_svg(z1)
    x2, y2    = to_svg(z2)
    cx_o, cy_o = poincare_to_svg(c_orth, cx, cy, R)
    r_svg      = r_orth * R

    # Determine arc direction (large-arc=0, sweep depends on orientation)
    # The shorter arc between z1 and z2 on the orthogonal circle.
    # Angular span determines large-arc flag.
    # NOTE: SVG y-axis is flipped (y increases downward), so we conjugate
    # the Poincare coordinates when computing the sweep direction.
    ang1   = cmath.phase((z1 - c_orth).conjugate())
    ang2   = cmath.phase((z2 - c_orth).conjugate())
    d_ang  = ((ang2 - ang1 + math.pi) % (2*math.pi)) - math.pi
    large  = 1 if abs(d_ang) > math.pi else 0
    sweep  = 1 if d_ang > 0 else 0

    return (f"M {x1:.3f},{y1:.3f} "
            f"A {r_svg:.3f},{r_svg:.3f} 0 {large},{sweep} {x2:.3f},{y2:.3f}")


def render_svg(pos: np.ndarray, adj: dict, vertex_ring: dict, faces: list,
               num_rings: int, path: str,
               size: int = 900, show_vertices: bool = True) -> None:
    """
    Render the {3,7} Poincare disk tiling as SVG.

    Edges are drawn as geodesic arcs (circles perpendicular to boundary).
    Vertices are colored by ring index.
    """
    R  = size * 0.46          # disk radius in pixels
    cx = size / 2             # disk center x
    cy = size / 2             # disk center y

    N       = len(pos)
    placed  = [v for v in range(N) if not np.isnan(pos[v, 0])]
    seen_e  = set()

    # Collect edge paths grouped by min ring (for layered rendering)
    edge_paths_by_ring = defaultdict(list)
    for v in placed:
        for u in adj[v]:
            if u not in set(placed): continue
            key = (min(u,v), max(u,v))
            if key in seen_e: continue
            seen_e.add(key)
            z1 = complex(pos[v, 0], pos[v, 1])
            z2 = complex(pos[u, 0], pos[u, 1])
            r  = min(vertex_ring[v], vertex_ring[u])
            edge_paths_by_ring[r].append(geodesic_arc_svg(z1, z2, cx, cy, R))

    lines = []
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" '
                 f'width="{size}" height="{size}" '
                 f'viewBox="0 0 {size} {size}" '
                 f'style="background:#0a0a0f">')

    # Defs: clip path so nothing renders outside the disk
    lines.append(f'<defs>')
    lines.append(f'  <clipPath id="disk">')
    lines.append(f'    <circle cx="{cx}" cy="{cy}" r="{R}"/>')
    lines.append(f'  </clipPath>')
    # Radial gradient for the disk background
    lines.append(f'  <radialGradient id="bg" cx="50%" cy="50%" r="50%">')
    lines.append(f'    <stop offset="0%" stop-color="#1a1a2e"/>')
    lines.append(f'    <stop offset="85%" stop-color="#0d0d1a"/>')
    lines.append(f'    <stop offset="100%" stop-color="#050508"/>')
    lines.append(f'  </radialGradient>')
    lines.append(f'</defs>')

    # Disk background
    lines.append(f'<circle cx="{cx}" cy="{cy}" r="{R}" fill="url(#bg)"/>')

    # Edge group (clipped to disk)
    lines.append(f'<g clip-path="url(#disk)">')
    for r in sorted(edge_paths_by_ring.keys()):
        # Outer rings slightly thinner and more transparent
        opacity   = max(0.25, 1.0 - r * 0.12)
        thickness = max(0.4, 1.8 - r * 0.25)
        color     = _ring_color(r)
        lines.append(f'  <g stroke="{color}" stroke-width="{thickness:.2f}" '
                     f'fill="none" opacity="{opacity:.2f}">')
        for path_d in edge_paths_by_ring[r]:
            lines.append(f'    <path d="{path_d}"/>')
        lines.append(f'  </g>')
    lines.append(f'</g>')

    # Vertices
    if show_vertices:
        lines.append(f'<g clip-path="url(#disk)">')
        for v in placed:
            z      = complex(pos[v, 0], pos[v, 1])
            px, py = poincare_to_svg(z, cx, cy, R)
            r      = vertex_ring[v]
            color  = _ring_color(r)
            radius = max(1.0, 4.0 - r * 0.5)
            lines.append(f'  <circle cx="{px:.2f}" cy="{py:.2f}" '
                         f'r="{radius:.1f}" fill="{color}" '
                         f'opacity="0.9"/>')
        lines.append(f'</g>')

    # Boundary circle
    lines.append(f'<circle cx="{cx}" cy="{cy}" r="{R}" '
                 f'fill="none" stroke="#ffffff" stroke-width="1.5" '
                 f'opacity="0.4"/>')

    # Label
    lines.append(f'<text x="{cx}" y="{size - 18}" '
                 f'font-family="Georgia, serif" font-size="13" '
                 f'fill="#ffffff" opacity="0.5" text-anchor="middle">'
                 f'{{3,7}} tiling -- Poincare disk -- {num_rings} rings'
                 f'</text>')

    lines.append('</svg>')

    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"SVG saved: {path}  ({len(placed)} vertices, {len(seen_e)} edges)")


# ===============================================================================
# Main
# ===============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Poincare disk {3,7} tiling')
    parser.add_argument('--rings', type=int, default=4,
                        help='Number of rings (default 4; '
                             'ring 5 has 617 vertices, ring 6 has 1625)')
    parser.add_argument('--size',  type=int, default=900,
                        help='SVG size in pixels (default 900)')
    parser.add_argument('--no-vertices', action='store_true',
                        help='Omit vertex dots (cleaner for many rings)')
    args = parser.parse_args()

    print(f"{{3,7}} Poincare disk tiling -- {args.rings} rings")
    print(f"  Vertex angle:          2pi/7 = {math.degrees(ALPHA):.4f} ")
    print(f"  Hyperbolic edge length: a  = {A:.6f}")
    print(f"  Poincare radius (ring 1):  = {TANH_A2:.6f}")
    print()

    print("Building combinatorial structure...")
    adj, vertex_ring, faces, boundary = build_combinatorial(args.rings)
    N = max(vertex_ring) + 1
    ring_sizes = [sum(1 for v,r in vertex_ring.items() if r==k)
                  for k in range(args.rings + 1)]
    print(f"  Vertices: {N}  Faces: {len(faces)}")
    print(f"  Ring sizes: {ring_sizes}")
    print()

    print("Computing exact Poincare coordinates...")
    coords = compute_poincare_coords(adj, vertex_ring, faces, boundary)
    print()

    print("Rendering SVG...")
    svg_path = 'poincare_37.svg'
    render_svg(coords, adj, vertex_ring, faces,
               num_rings=args.rings,
               path=svg_path,
               size=args.size,
               show_vertices=not args.no_vertices)

    print("Saving coordinates for GNN...")
    npy_path = 'poincare_coords.npy'
    np.save(npy_path, coords)
    print(f"  Saved: {npy_path}  shape={coords.shape} dtype={coords.dtype}")

    print()
    print("To use Poincare coordinates as GNN node features:")
    print("  coords = np.load('poincare_coords.npy')  # (N, 2)")
    print("  # coords[v] = (Re(z), Im(z)) in Poincare disk for vertex v")
    print("  # |coords[v]| = tanh(hyp_dist(v, center) / 2)")
    print("  # Append to node_feat in _node_features() in main GNN script")
    print()
    print(f"  Ring sizes: {ring_sizes}")
    print(f"  Poincare radii by ring:")
    for r in range(args.rings + 1):
        verts_r = [v for v,rv in vertex_ring.items() if rv==r]
        if verts_r:
            v = verts_r[0]
            if not np.isnan(coords[v, 0]):
                poincare_r = math.sqrt(coords[v,0]**2 + coords[v,1]**2)
                hyp_r = 2 * math.atanh(poincare_r) if poincare_r < 1 else float('inf')
                print(f"    ring {r}: |z| ~= {poincare_r:.4f}  "
                      f"(hyp dist from center ~= {hyp_r:.4f} = {r} x {A:.4f})")
