"""Spot detection, pattern centre and 2D reciprocal-lattice fitting.

Image coordinates: x to the right, y down (pixels). Lattice vectors are
stored in image coordinates as well.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree


@dataclass
class Spot:
    x: float
    y: float
    intensity: float
    peak: float = 0.0
    manual: bool = False
    diameter: float = 0.0

    @property
    def xy(self):
        return np.array([self.x, self.y])


@dataclass
class Lattice:
    g1: np.ndarray            # px, image coordinates
    g2: np.ndarray
    center: np.ndarray
    members: dict = field(default_factory=dict)   # spot index -> (n1, n2)
    rms: float = 0.0

    def position(self, n1, n2):
        return self.center + n1 * self.g1 + n2 * self.g2


def robust_sigma(values):
    values = np.asarray(values, float).ravel()
    median = np.median(values)
    return 1.4826 * np.median(np.abs(values - median)) or float(values.std()) or 1.0


def _mask_rects(shape, rects):
    valid = np.ones(shape, bool)
    for x1, y1, x2, y2 in rects or []:
        valid[max(0, int(y1)):int(y2), max(0, int(x1)):int(x2)] = False
    return valid


def remove_line_artifacts(data, valid=None):
    """Subtract bright detector columns/rows (bleeding, dead-column streaks)."""
    data = np.asarray(data, np.float32)
    out = data.copy()
    for axis in (0, 1):
        profile = np.median(data, axis=axis)
        smooth = ndi.median_filter(profile, size=15, mode="nearest")
        excess = profile - smooth
        noise = robust_sigma(excess)
        excess[excess < 4 * noise] = 0
        out -= excess[None, :] if axis == 0 else excess[:, None]
    return out


def detect_spots(image, sensitivity=6.0, min_distance=6, exclude=None, smoothing=1.2,
                 min_diameter=0., max_diameter=0.):
    """Bright diffraction spots with sub-pixel centroids.

    `sensitivity` is the threshold in robust noise units of a band-pass image.
    Size is the equivalent diameter of the connected half-height peak area,
    measured before detection smoothing. Zero disables each size bound.
    """
    values = (sensitivity, min_distance, smoothing, min_diameter, max_diameter)
    if (not all(np.isfinite(v) for v in values) or sensitivity <= 0 or min_distance < 1
            or smoothing <= 0 or min_diameter < 0 or max_diameter < 0
            or (max_diameter and min_diameter > max_diameter)):
        raise ValueError("亮点参数无效：尺寸须非负，最大直径不能小于最小直径。")
    data = remove_line_artifacts(image)
    h, w = data.shape
    valid = _mask_rects(data.shape, exclude)
    band = ndi.gaussian_filter(data, smoothing) - ndi.gaussian_filter(data, 10 + 2 * smoothing)
    noise = robust_sigma(band[valid][:: max(1, valid.sum() // 200000)])
    peaks = (band == ndi.maximum_filter(band, size=2 * int(min_distance) + 1)) & (band > sensitivity * noise)
    peaks &= valid
    border = 3
    peaks[:border], peaks[-border:], peaks[:, :border], peaks[:, -border:] = False, False, False, False
    ys, xs = np.nonzero(peaks)
    order = np.argsort(-band[ys, xs])[:3000]
    spots = []
    size_signal = data - ndi.gaussian_filter(data, 10 + 2 * smoothing)
    saturation = .98 * float(np.max(data))
    for y, x in zip(ys[order], xs[order]):
        diameter = _spot_diameter(size_signal, x, y)
        if diameter < min_diameter or (max_diameter and diameter > max_diameter):
            continue
        radius = max(3, int(min_distance * 0.8))
        patch = data[max(0, y-radius):y+radius+1, max(0, x-radius):x+radius+1]
        if np.mean(patch >= saturation) > .35:
            radius = max(radius, 16)
        spot = None
        # saturated discs are wider than the default window: grow it
        for r in (radius, 2 * radius, 4 * radius):
            spot = refine_centroid(data, x, y, radius=r)
            if spot is not None and np.hypot(spot.x - x, spot.y - y) < r:
                break
            spot = None
        if spot is None:
            continue
        if _elongation(band, x, y, max(3, int(min_distance * 0.8))) > 3.0:
            continue
        spot.peak = float(band[y, x] / noise)
        spot.diameter = diameter
        spots.append(spot)
    return merge_spots(spots, min_distance * 0.75)


def _spot_diameter(signal, x, y):
    """Connected half-height area; expanding crops avoid clipping broad discs."""
    h, w = signal.shape
    half = float(signal[y, x]) * .5
    if half <= 0:
        return 0.
    radius = 8
    while True:
        x1, x2 = max(0, x-radius), min(w, x+radius+1)
        y1, y2 = max(0, y-radius), min(h, y+radius+1)
        labels, _ = ndi.label(signal[y1:y2, x1:x2] >= half)
        component = labels == labels[y-y1, x-x1]
        touches = ((x1 > 0 and component[:, 0].any())
                   or (x2 < w and component[:, -1].any())
                   or (y1 > 0 and component[0].any())
                   or (y2 < h and component[-1].any()))
        if not touches:
            return float(2 * np.sqrt(np.count_nonzero(component) / np.pi))
        radius *= 2


def _elongation(band, x, y, radius):
    h, w = band.shape
    x1, x2, y1, y2 = max(0, x - radius), min(w, x + radius + 1), max(0, y - radius), min(h, y + radius + 1)
    window = np.clip(band[y1:y2, x1:x2], 0, None)
    window = np.where(window >= 0.3 * window.max(), window, 0)
    total = window.sum()
    if total <= 0:
        return 1.0
    yy, xx = np.mgrid[y1:y2, x1:x2]
    mx, my = (window * xx).sum() / total, (window * yy).sum() / total
    cxx = (window * (xx - mx) ** 2).sum() / total + 0.08
    cyy = (window * (yy - my) ** 2).sum() / total + 0.08
    cxy = (window * (xx - mx) * (yy - my)).sum() / total
    eig = np.linalg.eigvalsh(np.array([[cxx, cxy], [cxy, cyy]]))
    return float(np.sqrt(eig[1] / eig[0]))


def refine_centroid(data, x, y, radius=5, iterations=3):
    h, w = data.shape
    cx, cy = float(x), float(y)
    total = 0.0
    for _ in range(iterations):
        x1, x2 = int(max(0, round(cx) - radius)), int(min(w, round(cx) + radius + 1))
        y1, y2 = int(max(0, round(cy) - radius)), int(min(h, round(cy) + radius + 1))
        window = data[y1:y2, x1:x2]
        if window.size < 9:
            return None
        yy, xx = np.mgrid[y1:y2, x1:x2]
        inside = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius * radius
        ring = ~inside | ((xx - cx) ** 2 + (yy - cy) ** 2 >= (radius - 1.5) ** 2)
        background = np.median(window[ring]) if ring.any() else window.min()
        weights = np.clip(window - background, 0, None) * inside
        total = float(weights.sum())
        if total <= 0:
            return None
        nx, ny = float((weights * xx).sum() / total), float((weights * yy).sum() / total)
        if abs(nx - cx) < 0.02 and abs(ny - cy) < 0.02:
            cx, cy = nx, ny
            break
        cx, cy = nx, ny
    return Spot(cx, cy, total)


def merge_spots(spots, distance):
    spots = sorted(spots, key=lambda s: -s.intensity)
    kept = []
    for spot in spots:
        if all((spot.x - k.x) ** 2 + (spot.y - k.y) ** 2 > distance ** 2 for k in kept):
            kept.append(spot)
    return kept


def find_center(image, spots, exclude=None):
    """Transmitted-beam position refined from Friedel (centro-symmetric) pairs."""
    data = np.asarray(image, np.float32)
    h, w = data.shape
    valid = _mask_rects(data.shape, exclude)
    blurred = ndi.gaussian_filter(np.where(valid, data, 0), 8)
    guess = np.array(np.unravel_index(np.argmax(blurred), blurred.shape)[::-1], float)
    if spots:
        pts = np.array([[s.x, s.y] for s in spots])
        weights = np.array([s.intensity for s in spots])
        near = np.linalg.norm(pts - guess, axis=1) < 0.05 * max(h, w)
        if near.any():
            i = np.flatnonzero(near)[np.argmax(weights[near])]
            guess = pts[i].copy()
        if len(pts) >= 6:
            # Bound pair search, and prefer symmetry centres near the detector centre.
            # A single bright reflection is not reliable evidence of the transmitted beam.
            ids = np.argsort(-weights)[:400]
            pair_pts = pts[ids]
            a, b = np.triu_indices(len(pair_pts), 1)
            mids = (pair_pts[a] + pair_pts[b]) / 2
            image_center = np.array([w / 2, h / 2])
            good = np.linalg.norm(pair_pts[a] - pair_pts[b], axis=1) > 20
            good &= np.all(np.abs(mids - image_center) < np.array([w, h]) * .4, axis=1)
            mids = mids[good]
            if len(mids):
                tree = cKDTree(mids)
                counts = tree.query_ball_point(mids, 2.5, return_length=True)
                prior = np.exp(-.5 * np.sum(((mids-image_center) / (.12 * np.array([w, h]))) ** 2, axis=1))
                mx = np.clip(np.round(mids[:, 0]).astype(int), 0, w - 1)
                my = np.clip(np.round(mids[:, 1]).astype(int), 0, h - 1)
                local_signal = np.clip(blurred[my, mx] - np.median(blurred[valid]), 0, None)
                signal = np.sqrt(local_signal / max(float(local_signal.max()), 1e-9))
                # Inversion centres halfway between reflections are common. Prefer
                # a centre carrying real beam intensity, not an empty half-node.
                best = int(np.argmax(counts * (.15 + .85 * prior) * (.1 + .9 * signal)))
                if counts[best] >= 3:
                    guess = np.median(mids[tree.query_ball_point(mids[best], 2.5)], axis=0)
    return refine_center(spots, guess, radius=0.06 * max(h, w))


def refine_center(spots, guess, radius=60.0, tolerance=1.5):
    if len(spots) < 3:
        return np.asarray(guess, float)
    pts = np.array([[s.x, s.y] for s in spots])
    r = np.linalg.norm(pts - guess, axis=1)
    use = np.flatnonzero(r > 8)
    pts_use = pts[use]
    mids = (pts_use[:, None, :] + pts_use[None, :, :]) / 2
    iu = np.triu_indices(len(pts_use), 1)
    mids = mids[iu]
    # Friedel partners lie at opposite sides: the pair vectors are antiparallel.
    d1 = pts_use[iu[0]] - guess
    d2 = pts_use[iu[1]] - guess
    cos = (d1 * d2).sum(1) / (np.linalg.norm(d1, axis=1) * np.linalg.norm(d2, axis=1) + 1e-9)
    good = (np.linalg.norm(mids - guess, axis=1) < radius) & (cos < -0.95)
    mids = mids[good]
    if len(mids) < 2:
        return np.asarray(guess, float)
    # densest cluster of midpoints
    tree = cKDTree(mids)
    counts = tree.query_ball_point(mids, tolerance * 2, return_length=True)
    best = mids[tree.query_ball_point(mids[np.argmax(counts)], tolerance * 2)]
    for _ in range(3):
        center = np.median(best, axis=0)
        best = mids[np.linalg.norm(mids - center, axis=1) < tolerance * 1.5]
        if not len(best):
            break
    return np.median(best, axis=0) if len(best) else center


# ------------------------------------------------------------ lattice fit

def _tolerance(vectors, absolute=2.5, relative=0.02):
    return np.maximum(absolute, relative * np.linalg.norm(vectors, axis=-1))


def assign(points, center, g1, g2, absolute=2.5, relative=0.02):
    basis = np.stack([g1, g2], 1)
    rel = points - center
    coords = np.linalg.solve(basis, rel.T).T
    n = np.round(coords)
    residual = np.linalg.norm(rel - n @ basis.T, axis=1)
    ok = residual < _tolerance(rel, absolute, relative)
    return n.astype(int), residual, ok


def gauss_reduce(g1, g2):
    g1, g2 = np.array(g1, float), np.array(g2, float)
    for _ in range(50):
        if g2 @ g2 < g1 @ g1:
            g1, g2 = g2, g1
        m = round(float(g1 @ g2 / (g1 @ g1)))
        if m == 0:
            break
        g2 = g2 - m * g1
    # acute angle, g2 counter-clockwise from g1 on screen
    if g1 @ g2 < 0:
        g2 = -g2
    if screen_cross(g1, g2) < 0:
        g1, g2 = g2, g1
    return g1, g2


def screen_cross(a, b):
    """z-component of a x b with y pointing up (counter-clockwise > 0)."""
    return float(a[0] * -b[1] - (-a[1]) * b[0])


def _difference_vectors(points, weights, available, minimum, tolerance):
    """Repeated translations recover a basis even when its first-order spot is absent."""
    ids = np.flatnonzero(available)
    ids = ids[np.argsort(-weights[ids], kind="stable")[:60]]
    a, b = np.triu_indices(len(ids), 1)
    differences = points[ids[a]] - points[ids[b]]
    differences[differences[:, 0] < 0] *= -1
    differences = differences[np.linalg.norm(differences, axis=1) > minimum]
    if not len(differences):
        return []
    tree = cKDTree(differences)
    counts = tree.query_ball_point(differences, tolerance, return_length=True)
    order = np.argsort(-counts / np.sqrt(np.linalg.norm(differences, axis=1)), kind="stable")
    vectors = []
    for i in order:
        if counts[i] < 3:
            continue
        if any(np.linalg.norm(differences[i] - v) < tolerance for v in vectors):
            continue
        vectors.append(np.median(differences[tree.query_ball_point(differences[i], tolerance)], axis=0))
        if len(vectors) == 12:
            break
    return vectors


def fit_lattice(spots, center, weights=None, candidates=24, min_radius=10, exclude_ids=(),
                absolute=2.5, relative=0.02, refine_center_too=True,
                min_length=0., min_occupancy=0.):
    """Best 2D lattice through the spots, refined by least squares."""
    pts = np.array([[s.x, s.y] for s in spots], float)
    if len(pts) < 3:
        return None
    if weights is None:
        weights = np.sqrt(np.maximum([s.intensity for s in spots], 1e-9))
    weights = np.asarray(weights, float) / max(np.max(weights), 1e-12)
    rel = pts - center
    radius = np.linalg.norm(rel, axis=1)
    available = np.array([i not in exclude_ids for i in range(len(pts))])
    pool = np.flatnonzero(available & (radius > min_radius))
    if len(pool) < 2:
        return None
    # short, bright vectors are the best basis candidates
    pool = pool[np.argsort(radius[pool] / (0.3 + weights[pool]))][:candidates]
    vectors = list(rel[pool]) + _difference_vectors(pts, weights, available, min_radius, absolute * 2)
    rmax = np.percentile(radius[radius > min_radius], 90) if (radius > min_radius).any() else radius.max()
    hypotheses = []
    for a_i in range(len(vectors)):
        for b_i in range(a_i + 1, len(vectors)):
            a, b = vectors[a_i], vectors[b_i]
            cross = abs(screen_cross(a, b))
            if cross < 0.25 * np.linalg.norm(a) * np.linalg.norm(b):
                continue
            g1, g2 = gauss_reduce(a, b)
            area = abs(screen_cross(g1, g2))
            if area < 4 * min_radius ** 2 / 4:
                continue
            n, residual, ok = assign(pts, center, g1, g2, absolute, relative)
            ok &= radius > min_radius
            if ok.sum() < 3:
                continue
            provisional = Lattice(g1, g2, np.asarray(center),
                                  {int(i): tuple(n[i]) for i in np.flatnonzero(ok)})
            if min(np.linalg.norm(g1), np.linalg.norm(g2)) < min_length or \
                    lattice_occupancy(provisional, spots) < min_occupancy * .7:
                continue
            explained = weights[ok & available].sum()
            # A too fine lattice explains everything; penalise empty nodes.
            nodes = _node_count(g1, g2, rmax)
            occupied = len({tuple(v) for v in n[ok & (radius <= rmax)]})
            occupancy = occupied / max(nodes, 1)
            score = explained * (0.35 + 0.65 * min(1.0, occupancy)) ** 1.5
            hypotheses.append((score, g1, g2))
    if not hypotheses:
        return None
    # Initial spot pairs can represent a sublattice when spots are missing.
    # Compare full consensus fits before committing to a basis.
    best, best_score = None, -np.inf
    valid_fits = 0
    for _, g1, g2 in sorted(hypotheses, key=lambda item: -item[0])[:96]:
        lattice = refine_lattice(spots, center, g1, g2, absolute, relative,
                                 refine_center_too, exclude_ids=exclude_ids)
        ids = np.array([i for i in lattice.members if available[i]], dtype=int)
        if len(ids) < 3:
            continue
        if min(np.linalg.norm(lattice.g1), np.linalg.norm(lattice.g2)) < min_length or \
                lattice_occupancy(lattice, spots) < min_occupancy:
            continue
        occupied = len({lattice.members[i] for i in ids if radius[i] <= rmax})
        nodes = _node_count(lattice.g1, lattice.g2, rmax)
        occupancy = min(1., occupied / nodes)
        score = weights[ids].sum() * (.35 + .65 * occupancy) ** 1.5
        if score > best_score:
            best, best_score = lattice, score
        valid_fits += 1
        if valid_fits >= 16:
            break
    return best


def _node_count(g1, g2, rmax):
    area = abs(screen_cross(g1, g2))
    return max(1, int(np.pi * rmax ** 2 / area) - 1)


def refine_lattice(spots, center, g1, g2, absolute=2.5, relative=0.02, refine_center_too=True,
                   members=None, exclude_ids=()):
    pts = np.array([[s.x, s.y] for s in spots], float)
    center = np.asarray(center, float)
    g1, g2 = np.asarray(g1, float), np.asarray(g2, float)
    available = np.array([i not in exclude_ids for i in range(len(pts))])
    for _ in range(8):
        if members is None:
            n, residual, ok = assign(pts, center, g1, g2, absolute, relative)
            ok &= np.linalg.norm(pts - center, axis=1) > 3
            ok &= available
            ids = np.flatnonzero(ok)
            nn = n[ok]
        else:
            ids = np.array(list(members.keys()), int)
            nn = np.array([members[i] for i in ids], float)
        if len(ids) < 2:
            break
        if np.linalg.matrix_rank(nn) < 2:
            break
        predicted = center + nn @ np.stack([g1, g2])
        errors = np.linalg.norm(pts[ids] - predicted, axis=1)
        cutoff = max(absolute * .5, float(np.median(errors)) * 1.5)
        weights = np.sqrt(np.minimum(1., cutoff / np.maximum(errors, 1e-9)))
        if refine_center_too and len(ids) >= 4:
            design = np.column_stack([np.ones(len(ids)), nn])
            solution, *_ = np.linalg.lstsq(design * weights[:, None],
                                           pts[ids] * weights[:, None], rcond=None)
            center, g1, g2 = solution[0], solution[1], solution[2]
        else:
            solution, *_ = np.linalg.lstsq(nn * weights[:, None],
                                           (pts[ids] - center) * weights[:, None], rcond=None)
            g1, g2 = solution[0], solution[1]
        if members is not None:
            break
    n, residual, ok = assign(pts, center, g1, g2, absolute, relative)
    ok &= np.linalg.norm(pts - center, axis=1) > 3
    lattice = Lattice(g1, g2, center)
    lattice.members = {int(i): (int(n[i, 0]), int(n[i, 1])) for i in np.flatnonzero(ok)}
    lattice.rms = float(np.sqrt(np.mean(residual[ok] ** 2))) if ok.any() else 0.0
    return lattice


def lattice_occupancy(lattice, spots):
    """Share of lattice nodes inside the spot cloud that carry a spot."""
    pts = np.array([[s.x, s.y] for s in spots], float)
    if not lattice.members:
        return 0.0
    r = np.linalg.norm(pts[list(lattice.members)] - lattice.center, axis=1)
    rmax = np.percentile(r, 85)
    occupied = sum(1 for i, n in lattice.members.items()
                   if np.linalg.norm(lattice.position(*n) - lattice.center) <= rmax)
    return occupied / _node_count(lattice.g1, lattice.g2, rmax)


def find_lattices(spots, center, max_sets=3, min_spots=6, absolute=2.5, relative=0.02,
                  min_length=20.0, min_occupancy=0.2, report=None, min_radius=10,
                  fixed_center=False):
    """Successive lattices; later sets are built from spots not yet explained."""
    lattices, used = [], set()
    attempts = 0
    while len(lattices) < max_sets and attempts < max_sets + 3:
        attempts += 1
        lattice = fit_lattice(spots, center, exclude_ids=used, absolute=absolute, relative=relative,
                              refine_center_too=not lattices and not fixed_center,
                              min_radius=min_radius,
                              min_length=min_length, min_occupancy=min_occupancy)
        if lattice is None:
            break
        new = set(lattice.members) - used
        if len(new) < min_spots:
            if report is not None:
                report.append({"accepted": False, "new_spots": len(new),
                               "reason": "新增独立斑点不足，未自动增加组"})
            break
        short = min(np.linalg.norm(lattice.g1), np.linalg.norm(lattice.g2))
        if short < min_length or lattice_occupancy(lattice, spots) < min_occupancy:
            # random weak spots make a fine meaningless mesh; drop them and retry
            used |= new
            continue
        if lattices:
            # a "new" lattice that is just a sub-lattice of an earlier one adds nothing
            duplicate = False
            for prev in lattices:
                _, _, inside = assign(np.array([lattice.center + lattice.g1, lattice.center + lattice.g2]),
                                      prev.center, prev.g1, prev.g2, absolute, relative)
                if inside.all():
                    duplicate = True
                    break
            if duplicate:
                if report is not None:
                    report.append({"accepted": False, "new_spots": len(new),
                                   "reason": "与已有晶格重复，未自动增加组"})
                used |= new
                continue
        directions = np.array([spots[i].xy - lattice.center for i in new])
        singular = np.linalg.svd(directions, compute_uv=False)
        if singular[-1] < .08 * singular[0]:
            if report is not None:
                report.append({"accepted": False, "new_spots": len(new),
                               "reason": "新增斑点近共线，二维晶格证据不足"})
            used |= new
            continue
        nodes = set(lattice.members.values())
        pairs = sum(tuple(-v for v in n) in nodes for n in nodes if n != (0, 0)) // 2
        if len(new) < 10 and pairs < 2:
            if report is not None:
                report.append({"accepted": False, "new_spots": len(new),
                               "reason": "稀疏候选缺少两对中心对称斑点，保留检测点供手工核对"})
            used |= new
            continue
        if report is not None:
            report.append({"accepted": True, "new_spots": len(new),
                           "reason": "独立斑点支持二维晶格", "rms_px": lattice.rms,
                           "occupancy": min(1., lattice_occupancy(lattice, spots))})
        lattices.append(lattice)
        if not lattices[1:]:
            center = lattice.center
        used |= set(lattice.members)
    return lattices
