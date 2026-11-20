"""Matching measured reciprocal-lattice vectors to CIF phases."""
from dataclasses import dataclass, field

import numpy as np

from .crystal import Phase, preference_key, reduce_int
from .pattern import screen_cross


@dataclass
class Candidate:
    phase: Phase
    h1: np.ndarray            # indices of the measured g1
    h2: np.ndarray            # indices of the measured g2
    zone: tuple
    d1_err: float             # relative length errors (measured / CIF - 1)
    d2_err: float
    angle_meas: float
    angle_theo: float
    forbidden: float = 0.0    # weighted share of observed spots that are kinematically forbidden
    missing: float = 0.0      # weighted share of strong CIF reflections that were not observed
    observed: int = 0
    cost: float = 0.0
    scale: float = 1.0        # camera-constant correction suggested by this match
    dynamic: float = 0.0      # weighted share of spots explained only by double diffraction
    notes: list = field(default_factory=list)
    cost_parts: dict = field(default_factory=dict)

    @property
    def label(self):
        return f"{self.phase.name} {self.phase.uvw_text(self.zone)}"

    @property
    def scale_err(self):
        """Common length error (camera constant / lattice-parameter mismatch)."""
        return (self.d1_err + self.d2_err) / 2

    @property
    def ratio_err(self):
        return self.d1_err - self.d2_err


def zone_of(h1, h2):
    return reduce_int(np.cross(np.asarray(h1, int), np.asarray(h2, int)))


def _angle(a, b):
    c = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def candidate_pairs(phase, g1_len, g2_len, angle, length_tol=0.08, angle_tol=3.0,
                    include_forbidden=False, max_pairs=4000):
    """All (h1, h2) with |g| close to the measured lengths and the measured angle."""
    usable = np.ones(len(phase.hkl), bool) if include_forbidden else phase.possible
    gn = phase.gn
    a_idx = np.flatnonzero(usable & (np.abs(gn / g1_len - 1) < length_tol))
    b_idx = np.flatnonzero(usable & (np.abs(gn / g2_len - 1) < length_tol))
    if not len(a_idx) or not len(b_idx):
        return []
    ga = phase.g[a_idx] / gn[a_idx, None]
    gb = phase.g[b_idx] / gn[b_idx, None]
    cos = np.clip(ga @ gb.T, -1, 1)
    angles = np.degrees(np.arccos(cos))
    # the measured length ratio must survive any common scale error
    ratio = (gn[a_idx][:, None] / gn[b_idx][None, :]) / (g1_len / g2_len)
    ia, ib = np.nonzero((np.abs(angles - angle) < angle_tol) & (np.abs(ratio - 1) < 0.6 * length_tol))
    pairs = [(a_idx[i], b_idx[j]) for i, j in zip(ia, ib)]
    return pairs[:max_pairs]


def evaluate(phase, h1, h2, measured, length_tol=0.03, angle_tol=3.0, scale_tol=0.06,
             geometry_only=False):
    """Pattern consistency of one assignment.

    measured: dict with g1_len, g2_len, angle, nodes (list of (n1, n2, weight)),
    rmax (1/nm) and handed (+1 if g2 is counter-clockwise from g1).
    length_tol applies to the d1/d2 ratio, scale_tol to the common length error
    (camera constant drift or lattice-parameter mismatch with the CIF).
    geometry_only drops the kinematic-intensity term (HRTEM FFT peaks follow the
    contrast transfer function, not diffraction intensities).
    """
    h1, h2 = np.asarray(h1, int), np.asarray(h2, int)
    zone = zone_of(h1, h2)
    if measured.get("handed", 1) < 0:
        zone = tuple(-v for v in zone)
    t1, t2 = phase.g_vector(h1), phase.g_vector(h2)
    T1, T2 = np.linalg.norm(t1), np.linalg.norm(t2)
    d1_err = measured["g1_len"] / T1 - 1
    d2_err = measured["g2_len"] / T2 - 1
    theo_angle = _angle(t1, t2)
    weights_total, forbidden_weight, dynamic_weight, observed = 0.0, 0.0, 0.0, 0
    seen = set()
    for n1, n2, weight in measured["nodes"]:
        hkl = n1 * h1 + n2 * h2
        seen.add((n1, n2))
        weights_total += weight
        observed += 1
        kind = phase.kind_of(hkl)
        if kind == "absent":
            forbidden_weight += weight
        elif kind == "dynamic":
            dynamic_weight += weight
    forbidden = forbidden_weight / weights_total if weights_total else 0.0
    dynamic = dynamic_weight / weights_total if weights_total else 0.0
    # Strong CIF reflections inside the measured radius that were not seen.
    rmax = measured["rmax"]
    idx = phase.zone_reflections(zone, True, gmax=rmax * (1 + abs(d1_err) + 0.02))
    basis = np.stack([t1, t2], 1)
    missing_weight, expected_weight = 0.0, 0.0
    if len(idx):
        coeff = np.linalg.lstsq(basis, phase.g[idx].T, rcond=None)[0].T
        strength = np.sqrt(phase.intensity[idx])
        on_node = np.all(np.abs(coeff - np.round(coeff)) < 0.02, axis=1)
        for k, i in enumerate(idx):
            expected_weight += strength[k]
            if not on_node[k] or tuple(np.round(coeff[k]).astype(int)) not in seen:
                missing_weight += strength[k]
    missing = missing_weight / expected_weight if expected_weight else 0.0
    scale_err = (d1_err + d2_err) / 2
    parts = {"ratio": abs(d1_err - d2_err) / max(length_tol, 1e-6) * .6,
             "scale": abs(scale_err) / max(scale_tol, 1e-6) * .5,
             "angle": abs(measured["angle"] - theo_angle) / max(angle_tol, 1e-6) * .6,
             "forbidden": 2.5 * forbidden, "dynamic": .4 * dynamic,
             "missing": 0. if geometry_only else 2. * missing, "zone": zone_penalty(zone)}
    cost = sum(parts.values())
    return Candidate(phase, h1, h2, zone, d1_err, d2_err, measured["angle"], theo_angle,
                     forbidden, missing, observed, cost,
                     scale=1.0 / (1 + scale_err), dynamic=dynamic, cost_parts=parts)


def zone_penalty(zone):
    """Strong SAED patterns almost always come from low-index zones."""
    return 0.06 * max(0, sum(abs(v) for v in zone) - 3)


def nicest(phase, h1, h2, handed=1):
    """Symmetry-equivalent labelling with the most readable zone axis and indices."""
    options = []
    for m in phase.operations:
        for sign in (1, -1):
            a, b = sign * (np.asarray(h1) @ m), sign * (np.asarray(h2) @ m)
            zone = zone_of(a, b)
            if handed < 0:
                zone = tuple(-v for v in zone)
            options.append((preference_key(zone), preference_key(a), preference_key(b),
                            tuple(a), tuple(b)))
    best = max(options)
    return np.array(best[3]), np.array(best[4])


def equivalent_labellings(phase, h1, h2, handed=1):
    seen, result = set(), []
    for m in phase.operations:
        for sign in (1, -1):
            a, b = tuple(sign * (np.asarray(h1) @ m)), tuple(sign * (np.asarray(h2) @ m))
            if (a, b) not in seen:
                seen.add((a, b))
                result.append((np.array(a), np.array(b)))
    result.sort(key=lambda ab: (preference_key(zone_of(*ab) if handed > 0 else
                                               tuple(-v for v in zone_of(*ab))),
                                preference_key(ab[0]), preference_key(ab[1])), reverse=True)
    return result


def measured_from_lattice(lattice, spots, px_per_unit, max_nodes=80):
    g1, g2 = lattice.g1, lattice.g2
    nodes = []
    rmax = 0.0
    for i, (n1, n2) in lattice.members.items():
        if (n1, n2) == (0, 0):
            continue
        spot = spots[i]
        weight = np.sqrt(max(spot.intensity, 1e-9))
        nodes.append((n1, n2, weight))
        rmax = max(rmax, np.linalg.norm(n1 * g1 + n2 * g2) / px_per_unit)
    nodes.sort(key=lambda t: -t[2])
    if nodes:
        top = nodes[0][2]
        nodes = [(a, b, w / top) for a, b, w in nodes[:max_nodes]]
    return {"g1_len": np.linalg.norm(g1) / px_per_unit, "g2_len": np.linalg.norm(g2) / px_per_unit,
            "angle": _angle(g1, g2), "handed": 1 if screen_cross(g1, g2) > 0 else -1,
            "nodes": nodes, "rmax": max(rmax, 1.05 * max(np.linalg.norm(g1), np.linalg.norm(g2)) / px_per_unit)}


def index_lattice(phases, measured, length_tol=0.03, angle_tol=3.0, scale_tol=0.06,
                  include_forbidden=False, top=40, zone_limit=None, geometry_only=False):
    """Ranked candidates (one per phase and zone family)."""
    results = []
    window = length_tol + scale_tol
    for phase in phases:
        pairs = candidate_pairs(phase, measured["g1_len"], measured["g2_len"], measured["angle"],
                                window, angle_tol, include_forbidden)
        best_by_family = {}
        checked = set()
        for a, b in pairs:
            h1, h2 = phase.hkl[a], phase.hkl[b]
            zone = zone_of(h1, h2)
            if zone_limit and max(abs(v) for v in zone) > zone_limit:
                continue
            family = phase.family_key(zone)
            # symmetry-equivalent pairs give identical scores; evaluate each
            # distinct (|h1|, |h2|, family) once
            key = (family, round(float(phase.gn[a]), 5), round(float(phase.gn[b]), 5),
                   round(phase.intensity[a], 3), round(phase.intensity[b], 3))
            if key in checked:
                continue
            checked.add(key)
            candidate = evaluate(phase, h1, h2, measured, length_tol, angle_tol, scale_tol,
                                 geometry_only)
            if family not in best_by_family or candidate.cost < best_by_family[family].cost:
                best_by_family[family] = candidate
        for candidate in best_by_family.values():
            n1, n2 = nicest(phase, candidate.h1, candidate.h2, measured.get("handed", 1))
            candidate.h1, candidate.h2 = n1, n2
            candidate.zone = zone_of(n1, n2) if measured.get("handed", 1) > 0 else \
                tuple(-v for v in zone_of(n1, n2))
            results.append(candidate)
    results.sort(key=lambda c: c.cost)
    return results[:top]


def pair_candidates(phases, len1, len2, angle, handed, length_tol=0.03, angle_tol=3.0,
                    scale_tol=0.06, include_forbidden=False, top=60):
    """Classical two-spot indexing without pattern information."""
    measured = {"g1_len": len1, "g2_len": len2, "angle": angle, "handed": handed,
                "nodes": [(1, 0, 1.0), (0, 1, 1.0)], "rmax": max(len1, len2) * 1.01}
    results = []
    for phase in phases:
        seen = set()
        for a, b in candidate_pairs(phase, len1, len2, angle, length_tol + scale_tol, angle_tol,
                                    include_forbidden):
            h1, h2 = phase.hkl[a], phase.hkl[b]
            family = phase.family_key(zone_of(h1, h2))
            key = (family, round(float(phase.gn[a]), 5), round(float(phase.gn[b]), 5))
            if key in seen:
                continue
            seen.add(key)
            c = evaluate(phase, h1, h2, measured, length_tol, angle_tol, scale_tol)
            c.missing = 0.0
            c.cost_parts["missing"] = 0.0
            c.cost = sum(c.cost_parts.values())
            n1, n2 = nicest(phase, h1, h2, handed)
            c.h1, c.h2 = n1, n2
            c.zone = zone_of(n1, n2) if handed > 0 else tuple(-v for v in zone_of(n1, n2))
            results.append(c)
    results.sort(key=lambda c: c.cost)
    return results[:top]


# ------------------------------------------------------------- simulation

def simulate_zone(phase, zone, h1, g1_px, g2_px, center, px_per_unit, gmax, fit_points=None,
                  free_scale=False, include_forbidden=False):
    """Ideal CIF pattern of a zone placed on the image.

    The in-plane rotation is fitted to the measured g1/g2 (or to `fit_points`
    given as (hkl, xy) pairs); the scale is the calibration unless free_scale.
    Returns list of (hkl, x, y, intensity, allowed).
    """
    e1, e2, _ = phase.plane_frame(zone, h1)
    idx = np.flatnonzero((phase.hkl @ np.asarray(zone, int) == 0) & (phase.gn <= gmax))
    if not include_forbidden:
        idx = idx[phase.allowed[idx]]
    if fit_points is None:
        fit_points = []
    q, p = [], []
    for hkl, xy in fit_points:
        g = phase.g_vector(hkl)
        q.append([g @ e1, g @ e2])
        p.append([xy[0] - center[0], -(xy[1] - center[1])])   # y up
    q, p = np.array(q, float).reshape(-1, 2), np.array(p, float).reshape(-1, 2)
    if len(q):
        dot = (q * p).sum()
        cross = (q[:, 0] * p[:, 1] - q[:, 1] * p[:, 0]).sum()
        theta = np.arctan2(cross, dot)
        scale = np.hypot(dot, cross) / (q * q).sum() if free_scale else px_per_unit
    else:
        theta, scale = 0.0, px_per_unit
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    out = []
    for i in idx:
        g = phase.g[i]
        v = rot @ np.array([g @ e1, g @ e2]) * scale
        out.append((tuple(int(x) for x in phase.hkl[i]), center[0] + v[0], center[1] - v[1],
                    float(phase.intensity[i]), bool(phase.allowed[i])))
    return out, float(np.degrees(theta)), float(scale)
