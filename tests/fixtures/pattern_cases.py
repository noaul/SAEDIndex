"""Synthetic spot clouds with known lattices; no fitter is used to label them."""
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter

from saedindex.pattern import Spot


@dataclass
class PatternCase:
    name: str
    spots: list
    center: np.ndarray
    bases: list
    truth: list
    seed: int
    expected_groups: int


def make_case(name, bases, seed=17, missing=0., jitter=.3, noise=0, extent=3):
    rng = np.random.default_rng(seed)
    center = np.array([256., 256.])
    spots, truth = [], []
    for group, basis in enumerate(bases):
        for a in range(-extent, extent + 1):
            for b in range(-extent, extent + 1):
                if (a, b) == (0, 0) or rng.random() < missing:
                    continue
                xy = center + np.asarray(basis).T @ [a, b]
                if np.any(xy < 12) or np.any(xy > 500):
                    continue
                existing = next((i for i, p in enumerate(spots)
                                 if np.linalg.norm(p.xy - xy) < 1.5), None)
                if existing is not None:
                    truth[existing].add(group)
                else:
                    xy = xy + rng.normal(0, jitter, 2)
                    spots.append(Spot(*xy, float(rng.uniform(80, 120))))
                    truth.append({group})
    for xy in rng.uniform(15, 497, (noise, 2)):
        if all(np.linalg.norm(p.xy - xy) > 5 for p in spots):
            spots.append(Spot(*xy, float(rng.uniform(10, 30))))
            truth.append(set())
    return PatternCase(name, spots, center, [np.asarray(b, float) for b in bases],
                       truth, seed, len(bases))


def cases():
    base = np.array([[44., 3.], [10., -51.]])
    theta = np.deg2rad(27)
    rotation = np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])
    result = [
        make_case("single_complete", [base], jitter=0),
        make_case("single_missing", [base], seed=23, missing=.35),
        make_case("single_noisy", [base], seed=41, noise=50),
        make_case("single_position_noise", [base], seed=51, jitter=1.1, noise=15),
        make_case("double_rotated", [base, base @ rotation.T], seed=33),
        make_case("double_shared_row", [[[44., 0.], [11., -46.]],
                                        [[44., 0.], [-9., -58.]]], seed=71, jitter=.1),
        make_case("double_missing", [base, base @ rotation.T], seed=83, missing=.2),
        make_case("noise_only", [], seed=97, noise=65),
    ]
    collinear = make_case("collinear", [], seed=5)
    collinear.spots = [Spot(256 + 37 * i, 256 + .05 * i, 100.)
                       for i in (-4, -3, -2, -1, 1, 2, 3, 4)]
    collinear.truth = [set() for _ in collinear.spots]
    result.append(collinear)
    for seed in (1, 4, 11, 15):
        result.append(make_case(f"single_sparse_noise_{seed}", [base], seed=seed,
                                missing=.35, jitter=1.1, noise=45))
    angle = np.deg2rad(17)
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
    for seed in (0, 5, 16):
        result.append(make_case(f"double_sparse_noise_{seed}", [base, base @ rotation.T],
                                seed=seed, missing=.35, jitter=1.1, noise=45))
    return result


def image_case(obscured=False):
    case = make_case("obscured_center" if obscured else "saturated_center",
                     [[[44., 0.], [0., -51.]]], jitter=0)
    image = np.zeros((512, 512), float)
    for spot in case.spots:
        image[round(spot.y), round(spot.x)] = 8000
    image = gaussian_filter(image, 1.8)
    yy, xx = np.mgrid[:512, :512]
    image += 1800 * np.exp(-((xx - 256) ** 2 + (yy - 256) ** 2) / 35.)
    image = np.clip(image + 4, 0, 255)
    if obscured:
        image[244:269, 244:269] = 4
    return case, image.astype(np.float32)


def truth_match(case, lattice):
    """Return the best group overlap; truth comes from generation, not fitting."""
    actual = set(lattice.members)
    values = []
    for group in range(len(case.bases)):
        expected = {i for i, labels in enumerate(case.truth) if group in labels}
        both = len(actual & expected)
        values.append((both / max(len(actual), 1), both / max(len(expected), 1)))
    return values
