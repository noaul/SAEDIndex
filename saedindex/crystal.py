"""CIF reading and kinematic electron-diffraction geometry.

Units: direct lattice in nm, reciprocal vectors in 1/nm (no 2*pi).
Zone-axis convention: [uvw] = h1 x h2 when g2 is counter-clockwise from g1 on
screen, i.e. the zone axis points from the specimen towards the viewer
(antiparallel to the electron beam).
"""
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import product
from math import gcd
from pathlib import Path
import re

import numpy as np

ELEMENTS = ("H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni "
            "Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe "
            "Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg "
            "Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf").split()
ATOMIC_NUMBER = {symbol: i + 1 for i, symbol in enumerate(ELEMENTS)}
ATOMIC_NUMBER["D"] = 1


# ---------------------------------------------------------------- CIF parsing

_TOKEN = re.compile(r"""'(?:[^']|'(?=\S))*'(?=\s|$)|"(?:[^"]|"(?=\S))*"(?=\s|$)|#.*|\S+""")


def _tokens(text):
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith(";"):
            block = [line[1:]]
            i += 1
            while i < len(lines) and not lines[i].startswith(";"):
                block.append(lines[i])
                i += 1
            i += 1
            yield "\n".join(block).strip(), True
            continue
        for match in _TOKEN.finditer(line):
            token = match.group(0)
            if token.startswith("#"):
                break
            if token[0] in "'\"" and len(token) >= 2:
                yield token[1:-1], True
            else:
                yield token, False
        i += 1


def parse_cif_text(text):
    """Tags and loops of the first data block (tag names lower-cased)."""
    tokens = list(_tokens(text))
    tags, loops = {}, []
    i, started = 0, False
    while i < len(tokens):
        token, quoted = tokens[i]
        low = token.lower()
        if not quoted and low.startswith("data_"):
            if started:
                break
            started = True
            i += 1
        elif not quoted and low == "loop_":
            i += 1
            names = []
            while i < len(tokens) and not tokens[i][1] and tokens[i][0].startswith("_"):
                names.append(tokens[i][0].lower())
                i += 1
            values = []
            while i < len(tokens):
                value, q = tokens[i]
                if not q and (value.startswith("_") or value.lower() == "loop_"
                              or value.lower().startswith("data_")):
                    break
                values.append(value)
                i += 1
            if names:
                rows = len(values) // len(names)
                loops.append({name: values[k::len(names)][:rows] for k, name in enumerate(names)})
        elif not quoted and token.startswith("_"):
            if i + 1 < len(tokens):
                tags[low] = tokens[i + 1][0]
            i += 2
        else:
            i += 1
    return tags, loops


def cif_number(value, default=None):
    if value is None:
        return default
    value = re.sub(r"\(.*?\)", "", str(value)).strip()
    if value in ("", ".", "?"):
        return default
    try:
        return float(Fraction(value)) if "/" in value else float(value)
    except ValueError:
        return default


def parse_symop(text):
    """Rotation matrix and translation of an operation such as '-x+1/2, y, z'."""
    text = re.sub(r"\s+", "", text.lower())
    parts = text.split(",")
    if len(parts) != 3:
        raise ValueError(f"Bad symmetry operation: {text}")
    rotation, translation = np.zeros((3, 3)), np.zeros(3)
    for row, expression in enumerate(parts):
        for sign, term in re.findall(r"([+-]?)([^+-]+)", expression):
            factor = -1.0 if sign == "-" else 1.0
            if term[-1] in "xyz":
                coefficient = term[:-1].rstrip("*")
                coefficient = float(Fraction(coefficient)) if coefficient else 1.0
                rotation[row, "xyz".index(term[-1])] += factor * coefficient
            elif term[0] in "xyz" and "/" in term:
                rotation[row, "xyz".index(term[0])] += factor / float(term.split("/")[1])
            else:
                translation[row] += factor * float(Fraction(term))
    return rotation, translation


def element_of(text):
    match = re.match(r"([A-Z][a-z]?)", text.strip().capitalize() if text[:1].islower() else text.strip())
    if not match:
        return "X"
    symbol = match.group(1)
    if symbol not in ATOMIC_NUMBER and symbol[:1] in ATOMIC_NUMBER:
        symbol = symbol[:1]
    return symbol


def lattice_vectors(a, b, c, alpha, beta, gamma):
    """Rows a, b, c in Cartesian coordinates (a along x, b in the xy plane)."""
    al, be, ga = np.radians([alpha, beta, gamma])
    va = np.array([a, 0.0, 0.0])
    vb = np.array([b * np.cos(ga), b * np.sin(ga), 0.0])
    cx = c * np.cos(be)
    cy = c * (np.cos(al) - np.cos(be) * np.cos(ga)) / np.sin(ga)
    vc = np.array([cx, cy, np.sqrt(max(c * c - cx * cx - cy * cy, 0.0))])
    return np.vstack([va, vb, vc])


def _subscript(formula):
    table = str.maketrans("0123456789.", "\u2080\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089.")
    return re.sub(r"(?<=[A-Za-z)])(\d+(?:\.\d+)?)", lambda m: m.group(1).translate(table), formula)


# ------------------------------------------------------------------- Phase

@dataclass
class Atom:
    element: str
    xyz: np.ndarray
    occupancy: float = 1.0
    b_iso: float = 0.5   # A^2


@dataclass
class Phase:
    name: str
    cell: tuple                      # a, b, c (nm), alpha, beta, gamma (deg)
    atoms: list
    space_group: str = ""
    formula: str = ""
    path: str = ""
    gmax: float = 14.0               # 1/nm
    threshold: float = 1e-4          # |F|^2 / max below this counts as forbidden
    _cache: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self.direct = lattice_vectors(*self.cell)
        self.reciprocal = np.linalg.inv(self.direct).T
        self.metric = self.reciprocal @ self.reciprocal.T
        a, b, c, al, be, ga = self.cell
        self.hexagonal = (abs(a - b) < 1e-4 * a and abs(al - 90) < 0.05 and abs(be - 90) < 0.05
                          and abs(ga - 120) < 0.05)
        self._reflections()
        self._laue_operations()

    @property
    def display_name(self):
        return self.name

    # -- reflections
    def _scattering(self, s2):
        """Approximate electron scattering amplitude (relative intensities only)."""
        values = []
        for atom in self.atoms:
            z = ATOMIC_NUMBER.get(atom.element, 6)
            values.append(atom.occupancy * z ** 0.8 * np.exp(-(atom.b_iso + 2.0) * s2))
        return np.array(values)

    def _reflections(self):
        lengths = np.linalg.norm(self.direct, axis=1)
        limits = np.ceil(self.gmax * lengths).astype(int)
        ranges = [np.arange(-n, n + 1) for n in limits]
        hkl = np.stack(np.meshgrid(*ranges, indexing="ij"), -1).reshape(-1, 3)
        g = hkl @ self.reciprocal
        gn = np.linalg.norm(g, axis=1)
        keep = (gn > 0) & (gn <= self.gmax)
        hkl, g, gn = hkl[keep], g[keep], gn[keep]
        xyz = np.array([atom.xyz for atom in self.atoms]) if self.atoms else np.zeros((0, 3))
        intensity = np.zeros(len(hkl))
        s2 = (gn / 20.0) ** 2              # (sin(theta)/lambda)^2 in 1/A^2
        for start in range(0, len(hkl), 4096):
            part = slice(start, start + 4096)
            if not self.atoms:
                intensity[part] = 1.0
                continue
            phase = np.exp(2j * np.pi * hkl[part] @ xyz.T)
            f = self._scattering(s2[part]).T           # (n, atoms)
            intensity[part] = np.abs((phase * f).sum(axis=1)) ** 2
        peak = intensity.max() if len(intensity) and intensity.max() > 0 else 1.0
        self.hkl = hkl.astype(int)
        self.g = g
        self.gn = gn
        self.intensity = 100.0 * intensity / peak
        self.allowed = self.intensity > 100.0 * self.threshold
        self._lookup = {tuple(h): i for i, h in enumerate(self.hkl)}
        self._dynamic_reflections()

    def _dynamic_reflections(self):
        """Forbidden reflections reachable by double diffraction (g = g' + g'').

        Glide/screw absences can appear dynamically; lattice-centring absences
        never do, because sums of allowed reflections stay allowed.
        """
        allowed_set = {tuple(h) for h in self.hkl[self.allowed]}
        short = self.hkl[self.allowed][np.argsort(self.gn[self.allowed])][:300]
        possible = self.allowed.copy()
        for i in np.flatnonzero(~self.allowed):
            h = self.hkl[i]
            for s in short:
                if tuple(h - s) in allowed_set:
                    possible[i] = True
                    break
        self.possible = possible

    def kind_of(self, hkl):
        """'allowed', 'dynamic' (double diffraction only) or 'absent'."""
        i = self.index_of(hkl)
        if i is None:
            intensity = self.structure_intensity(hkl)
            return "allowed" if intensity > 100 * self.threshold else "dynamic"
        if self.allowed[i]:
            return "allowed"
        return "dynamic" if self.possible[i] else "absent"

    def index_of(self, hkl):
        return self._lookup.get(tuple(int(v) for v in hkl))

    def intensity_of(self, hkl):
        i = self.index_of(hkl)
        if i is None:
            return self.structure_intensity(hkl)
        return float(self.intensity[i])

    def structure_intensity(self, hkl):
        hkl = np.asarray(hkl, float)
        gn = np.linalg.norm(hkl @ self.reciprocal)
        if gn == 0:
            return 100.0
        xyz = np.array([atom.xyz for atom in self.atoms])
        f = self._scattering((gn / 20.0) ** 2)
        value = abs((f * np.exp(2j * np.pi * xyz @ hkl)).sum()) ** 2
        reference = self._unnormalised_peak()
        return float(100.0 * value / reference)

    def _unnormalised_peak(self):
        if "peak" not in self._cache:
            i = int(np.argmax(self.intensity))
            xyz = np.array([atom.xyz for atom in self.atoms])
            f = self._scattering((self.gn[i] / 20.0) ** 2)
            self._cache["peak"] = abs((f * np.exp(2j * np.pi * xyz @ self.hkl[i])).sum()) ** 2
        return self._cache["peak"]

    def d_spacing(self, hkl):
        return 1.0 / np.linalg.norm(np.asarray(hkl, float) @ self.reciprocal)

    def g_vector(self, hkl):
        return np.asarray(hkl, float) @ self.reciprocal

    def angle(self, h1, h2):
        a, b = self.g_vector(h1), self.g_vector(h2)
        return float(np.degrees(np.arccos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1))))

    # -- symmetry
    def _laue_operations(self):
        """Integer matrices M (h' = h M) that keep the metric and |F|^2."""
        mats = np.array(list(product((-1, 0, 1), repeat=9))).reshape(-1, 3, 3)
        det = np.round(np.linalg.det(mats)).astype(int)
        mats = mats[np.abs(det) == 1]
        g = self.metric
        transformed = np.einsum("nij,jk,nlk->nil", mats, g, mats)
        scale = np.abs(g).max()
        mats = mats[np.all(np.abs(transformed - g) < 2e-3 * scale, axis=(1, 2))]
        strong = np.flatnonzero(self.intensity > 1e-3)
        sample = strong[np.argsort(self.gn[strong])][:400]
        operations = []
        for m in mats:
            ok = True
            for i in sample:
                j = self._lookup.get(tuple(self.hkl[i] @ m))
                if j is None:
                    continue
                if abs(self.intensity[i] - self.intensity[j]) > 1e-3 * max(self.intensity[i], 1e-2) + 1e-4:
                    ok = False
                    break
            if ok:
                operations.append(m.astype(int))
        self.operations = operations or [np.eye(3, dtype=int), -np.eye(3, dtype=int)]
        self.operations_inverse = [np.round(np.linalg.inv(m)).astype(int) for m in self.operations]

    def equivalent_directions(self, uvw):
        uvw = np.asarray(uvw, int)
        return {tuple(int(v) for v in mi @ uvw) for mi in self.operations_inverse}

    def family_key(self, uvw):
        options = self.equivalent_directions(uvw) | self.equivalent_directions(-np.asarray(uvw))
        return max(options, key=preference_key)

    # -- zones
    def zone_axes(self, limit=5):
        """One representative per symmetry family, low indices first."""
        key = ("zones", limit)
        if key not in self._cache:
            seen, zones = set(), []
            for uvw in product(range(-limit, limit + 1), repeat=3):
                if uvw == (0, 0, 0) or gcd(gcd(abs(uvw[0]), abs(uvw[1])), abs(uvw[2])) != 1:
                    continue
                family = self.family_key(uvw)
                if family in seen:
                    continue
                seen.add(family)
                zones.append(family)
            zones.sort(key=lambda z: (sum(abs(v) for v in z), z))
            self._cache[key] = zones
        return self._cache[key]

    def zone_reflections(self, uvw, only_allowed=True, gmax=None):
        mask = self.hkl @ np.asarray(uvw, int) == 0
        if only_allowed:
            mask &= self.allowed
        if gmax is not None:
            mask &= self.gn <= gmax
        return np.flatnonzero(mask)

    def zone_basis(self, uvw):
        """Reduced basis (h1, h2, generated_by_allowed) of the reflections in a zone."""
        key = ("basis", tuple(uvw))
        if key in self._cache:
            return self._cache[key]
        result = None
        for only_allowed in (True, False):
            idx = self.zone_reflections(uvw, only_allowed)
            if len(idx) < 2:
                continue
            idx = idx[np.argsort(self.gn[idx], kind="stable")]
            members = self.hkl[idx].astype(float)
            a = self.hkl[idx[0]]
            for j in idx[1:]:
                b = self.hkl[j]
                if not np.any(np.cross(a, b)):
                    continue
                basis = np.stack([a, b], 1).astype(float)
                coefficients = np.linalg.lstsq(basis, members.T, rcond=None)[0]
                if np.allclose(coefficients, np.round(coefficients), atol=1e-6):
                    result = (*self.reduce_pair(a, b), only_allowed)
                    break
            if result:
                break
        self._cache[key] = result
        return result

    def reduce_pair(self, h1, h2):
        h1, h2 = np.array(h1, int), np.array(h2, int)
        norm = lambda h: float(h @ self.metric @ h)
        for _ in range(100):
            if norm(h2) < norm(h1):
                h1, h2 = h2, h1
            m = int(round(float(h1 @ self.metric @ h2) / norm(h1)))
            if m == 0:
                break
            h2 = h2 - m * h1
        return h1, h2

    def plane_frame(self, uvw, h1):
        """Orthonormal in-plane axes e1 (along g(h1)) and e2 = n x e1."""
        n = np.asarray(uvw, float) @ self.direct
        n /= np.linalg.norm(n)
        e1 = self.g_vector(h1)
        e1 = e1 - n * (e1 @ n)
        e1 /= np.linalg.norm(e1)
        return e1, np.cross(n, e1), n

    # -- index notation
    def hkl_text(self, hkl):
        return format_indices(hkl, "(", self.hexagonal, plane=True)

    def uvw_text(self, uvw):
        return format_indices(uvw, "[", self.hexagonal, plane=False)


def preference_key(v):
    v = tuple(int(x) for x in v)
    return (-sum(1 for x in v if x < 0), -sum(1 for x in v if x != 0), v)


def reduce_int(v):
    v = [int(round(x)) for x in v]
    divisor = 0
    for x in v:
        divisor = gcd(divisor, abs(x))
    return tuple(x // divisor for x in v) if divisor else tuple(v)


def four_index_plane(hkl):
    h, k, l = hkl
    return (h, k, -(h + k), l)


def four_index_direction(uvw):
    u, v, w = uvw
    four = (2 * u - v, 2 * v - u, -(u + v), 3 * w)
    return reduce_int(four)


def format_indices(values, bracket="(", hexagonal=False, plane=True):
    """Plain-text form; negative numbers carry a leading minus."""
    values = tuple(int(v) for v in values)
    if hexagonal:
        values = four_index_plane(values) if plane else four_index_direction(values)
    closing = {"(": ")", "[": "]", "{": "}", "<": ">"}[bracket]
    separator = " " if any(abs(v) >= 10 for v in values) else ""
    return bracket + separator.join(str(v) for v in values) + closing


def display_indices(values, hexagonal=False, plane=True):
    values = tuple(int(v) for v in values)
    if hexagonal:
        return four_index_plane(values) if plane else four_index_direction(values)
    return values


# ----------------------------------------------------------------- loading

def load_cif(path, gmax=14.0):
    path = Path(path)
    for encoding in ("utf-8", "gbk", "latin-1"):
        try:
            text = path.read_text(encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    tags, loops = parse_cif_text(text)
    cell = [cif_number(tags.get(f"_cell_length_{x}")) for x in "abc"]
    angles = [cif_number(tags.get(f"_cell_angle_{x}"), 90.0) for x in ("alpha", "beta", "gamma")]
    if any(v is None for v in cell):
        raise ValueError("CIF 中缺少晶胞参数 _cell_length_a/b/c。")
    operations = []
    for loop in loops:
        for name in ("_space_group_symop_operation_xyz", "_symmetry_equiv_pos_as_xyz",
                     "_space_group_symop.operation_xyz"):
            if name in loop:
                operations = [parse_symop(op) for op in loop[name]]
                break
        if operations:
            break
    if not operations:
        operations = [(np.eye(3), np.zeros(3))]
    atoms = []
    for loop in loops:
        if "_atom_site_fract_x" not in loop:
            continue
        n = len(loop["_atom_site_fract_x"])
        symbols = loop.get("_atom_site_type_symbol") or loop.get("_atom_site_label") or ["X"] * n
        occupancy = loop.get("_atom_site_occupancy", ["1"] * n)
        b_values = loop.get("_atom_site_b_iso_or_equiv")
        u_values = loop.get("_atom_site_u_iso_or_equiv")
        for i in range(n):
            xyz = np.array([cif_number(loop[f"_atom_site_fract_{c}"][i], 0.0) for c in "xyz"])
            b = cif_number(b_values[i]) if b_values else None
            if b is None and u_values:
                u = cif_number(u_values[i])
                b = 8 * np.pi ** 2 * u if u is not None else None
            atoms.append(Atom(element_of(symbols[i]), xyz, cif_number(occupancy[i], 1.0),
                              b if b is not None and 0 < b < 10 else 0.5))
    if not atoms:
        raise ValueError("CIF 中没有原子坐标 (_atom_site_fract_x/y/z)。")
    expanded = []
    for atom in atoms:
        for rotation, translation in operations:
            position = np.mod(rotation @ atom.xyz + translation, 1.0)
            duplicate = False
            for other in expanded:
                delta = position - other.xyz
                delta -= np.round(delta)
                if other.element == atom.element and np.all(np.abs(delta) < 1e-3):
                    duplicate = True
                    break
            if not duplicate:
                expanded.append(Atom(atom.element, position, atom.occupancy, atom.b_iso))
    formula = (tags.get("_chemical_formula_structural") or tags.get("_chemical_formula_sum")
               or path.stem).replace(" ", "")
    group = tags.get("_symmetry_space_group_name_h-m") or tags.get("_space_group_name_h-m_alt") or ""
    name = re.sub(r"_(mp|cod|icsd)-?\d+.*$", "", path.stem, flags=re.I) or formula
    return Phase(name=name, cell=(cell[0] / 10, cell[1] / 10, cell[2] / 10, *angles),
                 atoms=expanded, space_group=group.strip(), formula=_subscript(formula),
                 path=str(path), gmax=gmax)
