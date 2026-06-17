"""
gap_geometry.py  —  detailed spherical geometry analysis of the ring-5 boundary.

For each ring-5 PLY file, computes:

1. Gap boundary profile: angular distance from gap center to boundary
   as a function of azimuth around the gap center (not just the max).

2. Local demand at each ring-5 boundary vertex: the angular footprint
   that a ring-6 triangle attaching there would need to occupy.

3. Local curvature of the ring-5 boundary curve on the sphere.

4. Supply vs demand: at each attachment point, gap_supply(theta) vs demand(theta).

Usage:
    python gap_geometry.py physics_best.ply [output.json]
"""

import sys, json, math
import numpy as np
from collections import defaultdict

RING_COLORS = [
    (255,255,255),(68,1,84),(71,44,122),(59,81,139),(44,113,142),(33,144,141),
    (39,173,129),(92,200,100),(170,220,50),(234,229,26),(253,231,37),
]

def closest_ring(rgb):
    return min(range(len(RING_COLORS)),
               key=lambda r: sum((a-b)**2 for a,b in zip(rgb, RING_COLORS[r])))

def load_ply(path):
    with open(path) as f: lines = f.readlines()
    i = 0; nv = nf = 0
    while lines[i].strip() != 'end_header':
        if lines[i].startswith('element vertex'): nv = int(lines[i].split()[-1])
        if lines[i].startswith('element face'):   nf = int(lines[i].split()[-1])
        i += 1
    i += 1
    verts = []; colors = []
    for j in range(nv):
        p = lines[i+j].split()
        verts.append([float(p[0]),float(p[1]),float(p[2])])
        colors.append((int(p[3]),int(p[4]),int(p[5])))
    i += nv
    faces = []
    for j in range(nf):
        p = lines[i+j].split()
        faces.append([int(p[1]),int(p[2]),int(p[3])])
    return np.array(verts), [closest_ring(c) for c in colors], np.array(faces)

# Fibonacci sphere sample
N_SAMP = 10000
_g = (1+math.sqrt(5))/2
CANDS = np.array([
    [math.sin(math.acos(1-2*(k+.5)/N_SAMP))*math.cos(2*math.pi*k/_g),
     math.sin(math.acos(1-2*(k+.5)/N_SAMP))*math.sin(2*math.pi*k/_g),
     math.cos(math.acos(1-2*(k+.5)/N_SAMP))]
    for k in range(N_SAMP)])

def project_to_sphere(v, center):
    """Project vertices onto unit sphere centered at center."""
    rel  = v - center
    dist = np.linalg.norm(rel, axis=1)
    mask = dist > 1e-6
    units = np.zeros_like(rel)
    units[mask] = rel[mask] / dist[mask, None]
    return units, dist

def gap_profile(gap_center, boundary_units, n_azimuths=72):
    """
    Angular profile of the gap: for each azimuth direction around the gap center,
    find the angular distance to the nearest boundary vertex.

    gap_center: unit vector pointing to gap center
    boundary_units: (N,3) unit vectors of boundary vertices on sphere
    n_azimuths: number of azimuth samples around the gap center

    Returns: azimuths (degrees), radii (degrees)
    """
    # Build a local coordinate frame at gap_center
    # x-axis: arbitrary perpendicular to gap_center
    z = gap_center
    x = np.array([1,0,0]) if abs(z[0]) < 0.9 else np.array([0,1,0])
    x = x - np.dot(x, z)*z; x /= np.linalg.norm(x)
    y = np.cross(z, x)

    azimuths = np.linspace(0, 360, n_azimuths, endpoint=False)
    radii    = []
    for az in azimuths:
        # Direction in the tangent plane at gap_center
        rad = math.radians(az)
        tang = math.cos(rad)*x + math.sin(rad)*y

        # For each boundary vertex, find its angular distance from gap_center
        # and its azimuth in the local frame
        dots_z    = np.clip(boundary_units @ z, -1, 1)
        colats    = np.degrees(np.arccos(dots_z))   # angular distance from gap center

        # Project boundary vertices onto tangent plane and find azimuth
        proj      = boundary_units - dots_z[:,None]*z[None,:]  # tangent components
        pnorm     = np.linalg.norm(proj, axis=1, keepdims=True).clip(1e-8)
        tang_dirs = proj / pnorm

        # Find vertices within 60° of this azimuth direction
        az_dots   = tang_dirs @ tang
        in_sector = az_dots > math.cos(math.radians(360/n_azimuths))
        if in_sector.any():
            radii.append(float(colats[in_sector].min()))
        else:
            radii.append(float(colats.min()))

    return list(azimuths), radii

def local_demand_at_vertex(v_idx, boundary_units, boundary_dists,
                           adj, vertex_ring, r_boundary,
                           center, gap_center):
    """
    Estimate the angular footprint on the sphere that a ring-(r+1) triangle
    attaching at boundary vertex v_idx would need.

    The demand at vertex v is:
      - The vertex v itself is on the gap boundary (at angular distance d from gap center)
      - Ring-(r+1) attaches here with edge length ~1
      - The new vertex is ~1 edge-length away from v, roughly in the direction of the gap
      - Angular footprint = arcsin(edge / radius) at that point

    But more precisely: the new vertex's direction on the sphere depends on
    the local normal to the boundary curve at v. Where the boundary curves
    toward the gap, the new vertex lands closer to the gap center (smaller demand);
    where it curves away, the new vertex lands farther into the gap (larger demand).
    """
    v_unit = boundary_units[v_idx]
    v_dist = boundary_dists[v_idx]

    # Angular distance of v from gap center
    colat_v = math.degrees(math.acos(np.clip(np.dot(v_unit, gap_center), -1, 1)))

    # Local curvature: look at neighbors in same ring on the sphere
    ring_r  = vertex_ring[v_idx]
    nbrs    = [u for u in adj[v_idx] if vertex_ring.get(u, -1) == ring_r]

    # Outward normal to boundary curve at v (perpendicular to curve, pointing into gap)
    # = direction toward gap center, projected onto sphere tangent at v
    to_gap   = gap_center - np.dot(gap_center, v_unit)*v_unit
    tog_norm = np.linalg.norm(to_gap)
    if tog_norm < 1e-8:
        outward = np.array([0,0,0])
    else:
        outward = to_gap / tog_norm   # unit tangent pointing toward gap

    # Tangent to boundary curve at v (perpendicular to outward and to v)
    if nbrs:
        nbr_units = np.array([boundary_units[u] for u in nbrs if u < len(boundary_units)])
        if len(nbr_units):
            # Average direction to neighbors in tangent plane
            tang_vecs = nbr_units - np.dot(nbr_units, v_unit[:,None]) * v_unit
            tang_mean = tang_vecs.mean(axis=0)
            tn = np.linalg.norm(tang_mean)
            tangent = tang_mean / tn if tn > 1e-8 else np.cross(v_unit, outward)
        else:
            tangent = np.cross(v_unit, outward)
    else:
        tangent = np.cross(v_unit, outward)

    # Curvature sign: if boundary neighbors are more toward gap_center than v,
    # the curve is concave (good for ring 6); if less, convex (bad)
    if nbrs:
        nbr_colats = [math.degrees(math.acos(np.clip(np.dot(boundary_units[u], gap_center),-1,1)))
                      for u in nbrs if u < len(boundary_units)]
        mean_nbr_colat = np.mean(nbr_colats) if nbr_colats else colat_v
        # curvature > 0: neighbors closer to gap center (concave boundary = good)
        # curvature < 0: neighbors farther from gap center (convex = bad for ring 6)
        curvature = colat_v - mean_nbr_colat
    else:
        curvature = 0.0

    # Estimated demand: angular width of ring-(r+1) edge at this radius
    base_demand = math.degrees(math.asin(min(1.0, 1.0/v_dist)))

    # Adjust for local curvature: convex boundary pushes ring-(r+1) closer to
    # gap center, reducing effective room. Concave gives a bit more room.
    # This is a first-order correction.
    adjusted_demand = base_demand - curvature * 0.3  # heuristic factor

    return {
        'colat_from_gap': round(colat_v, 3),
        'base_demand':    round(base_demand, 3),
        'curvature':      round(float(curvature), 3),
        'adjusted_demand':round(adjusted_demand, 3),
        'radius':         round(float(v_dist), 4),
    }

def analyze(path, output_path=None, n_azimuths=72):
    verts, vring_list, faces = load_ply(path)
    vring = {i: vring_list[i] for i in range(len(verts))}

    # Build adjacency
    adj = defaultdict(set)
    for a, b, c in faces:
        for u, w in [(a,b),(b,c),(a,c)]:
            adj[u].add(w); adj[w].add(u)

    center = verts[vring_list.index(0)]
    units, dists = project_to_sphere(verts, center)

    max_ring = max(vring_list)
    print(f"File: {path}")
    print(f"Max ring: {max_ring}, vertices: {len(verts)}")

    # Find gap center using ring-max boundary vertices
    bdy_idx  = [i for i,r in enumerate(vring_list) if r == max_ring]
    bdy_u    = units[bdy_idx]
    bdy_d    = dists[bdy_idx]

    dots     = np.clip(CANDS @ bdy_u.T, -1, 1)
    min_angs = np.degrees(np.arccos(dots.max(axis=1)))
    gap_idx  = int(np.argmax(min_angs))
    gap_r    = float(min_angs[gap_idx])
    gap_c    = CANDS[gap_idx]
    print(f"Gap radius: {gap_r:.3f}°  center direction: {np.round(gap_c,3)}")

    # Gap boundary profile
    azimuths, profile = gap_profile(gap_c, bdy_u, n_azimuths)
    print(f"Gap profile: min={min(profile):.2f}°  max={max(profile):.2f}°  "
          f"mean={np.mean(profile):.2f}°  std={np.std(profile):.2f}°")

    # Local demand at each boundary vertex
    demands = []
    for i, vi in enumerate(bdy_idx):
        d = local_demand_at_vertex(vi, units, dists, adj, vring,
                                   max_ring, center, gap_c)
        demands.append(d)

    demand_vals = [d['adjusted_demand'] for d in demands]
    base_vals   = [d['base_demand']     for d in demands]
    curv_vals   = [d['curvature']       for d in demands]
    print(f"Demand at boundary vertices:")
    print(f"  base (global):    mean={np.mean(base_vals):.3f}°  std={np.std(base_vals):.3f}°  "
          f"range=[{min(base_vals):.3f}°, {max(base_vals):.3f}°]")
    print(f"  adjusted (local): mean={np.mean(demand_vals):.3f}°  std={np.std(demand_vals):.3f}°  "
          f"range=[{min(demand_vals):.3f}°, {max(demand_vals):.3f}°]")
    print(f"  curvature:        mean={np.mean(curv_vals):.3f}°  std={np.std(curv_vals):.3f}°")

    # Supply vs demand: for each azimuth, gap_profile[az] vs local demand at
    # the boundary vertex nearest to that azimuth
    supply_minus_demand = [p - d for p, d in zip(profile, [np.mean(demand_vals)]*len(profile))]
    print(f"\nSupply (profile) − mean demand:")
    print(f"  min={min(supply_minus_demand):.3f}°  max={max(supply_minus_demand):.3f}°")
    n_neg = sum(1 for x in supply_minus_demand if x < 0)
    print(f"  Azimuths where supply < demand: {n_neg}/{n_azimuths} ({100*n_neg/n_azimuths:.0f}%)")

    result = {
        'file':          path,
        'max_ring':      max_ring,
        'n_verts':       len(verts),
        'gap_radius':    round(gap_r, 4),
        'gap_center':    list(np.round(gap_c, 6)),
        'gap_profile': {
            'azimuths': [round(a,1) for a in azimuths],
            'radii':    [round(r,4) for r in profile],
            'mean':     round(float(np.mean(profile)), 4),
            'std':      round(float(np.std(profile)), 4),
            'min':      round(float(np.min(profile)), 4),
            'max':      round(float(np.max(profile)), 4),
        },
        'demand': {
            'base_mean':      round(float(np.mean(base_vals)), 4),
            'base_std':       round(float(np.std(base_vals)), 4),
            'base_min':       round(float(np.min(base_vals)), 4),
            'base_max':       round(float(np.max(base_vals)), 4),
            'adjusted_mean':  round(float(np.mean(demand_vals)), 4),
            'adjusted_std':   round(float(np.std(demand_vals)), 4),
            'adjusted_min':   round(float(np.min(demand_vals)), 4),
            'adjusted_max':   round(float(np.max(demand_vals)), 4),
            'curvature_mean': round(float(np.mean(curv_vals)), 4),
            'curvature_std':  round(float(np.std(curv_vals)), 4),
            'per_vertex':     demands,
        },
        'supply_minus_demand': {
            'values':  [round(x,4) for x in supply_minus_demand],
            'min':     round(float(np.min(supply_minus_demand)), 4),
            'max':     round(float(np.max(supply_minus_demand)), 4),
            'n_negative': n_neg,
            'frac_negative': round(n_neg/n_azimuths, 4),
        }
    }

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved to {output_path}")

    return result

if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'physics_best.ply'
    out  = sys.argv[2] if len(sys.argv) > 2 else path.replace('.ply', '_gap_geometry.json')
    analyze(path, out)
