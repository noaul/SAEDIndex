"""HRTEM Fourier analysis: square ROI FFT, Friedel peaks and masked IFFT (Qt-free).

Coordinates follow pattern.py: x to the right, y down. A real-space fringe
with wave vector (kx, ky) cycles/px appears at center + N * (kx, ky) in the
shifted N x N spectrum, so 1 FFT pixel = 1 / (N * nm_per_px) 1/nm.
With a display zoom z the same N x N grid samples only the central band
(chirp-z transform, equivalent to zero padding), so 1 pixel = 1/(z N nm_per_px).
"""
from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from scipy.signal import zoom_fft

from .pattern import Spot, robust_sigma

MIN_ROI = 32
ZOOMS = (1, 2, 4, 8)
REAL_UNITS = {"nm": 1.0, "a": .1, "\u00e5": .1, "ang": .1, "angstrom": .1, "pm": 1e-3,
              "um": 1e3, "\u00b5m": 1e3}


@dataclass
class FFTResult:
    roi: tuple                 # x, y, size in source pixels
    nm_per_px: float | None
    spectrum: np.ndarray       # windowed, shifted complex spectrum (detection/display)
    plain: np.ndarray          # unwindowed, shifted complex spectrum (reconstruction)
    mean: float
    zoom: int = 1

    @property
    def size(self):
        return int(self.roi[2])

    @property
    def center(self):
        c = float(self.size // 2)
        return np.array([c, c])

    @property
    def px_per_unit(self):
        """FFT pixels per 1/nm, the quantity used by reciprocal indexing."""
        return self.size * self.zoom * self.nm_per_px if self.nm_per_px else None

    def to_plain(self, xy):
        """Displayed (zoomed) spectrum coordinates to the unzoomed FFT grid."""
        return self.center + (np.asarray(xy, float) - self.center) / self.zoom

    @property
    def magnitude(self):
        return np.abs(self.spectrum).astype(np.float32)

    def display(self):
        """Log-magnitude image for display only."""
        log = np.log1p(np.abs(self.spectrum))
        low, high = np.percentile(log, [1., 99.97])
        return np.clip((log - low) / max(high - low, 1e-12) * 255, 0, 255).astype(np.uint8)


def real_scale_from_axes(axes):
    """(nm_per_px, anisotropic) from DM per-axis calibrations, or None if not real space."""
    values = []
    for scale, unit in list(axes)[:2]:
        factor = REAL_UNITS.get(str(unit).replace(" ", "").lower())
        if factor is None or not np.isfinite(scale) or scale <= 0:
            return None
        values.append(float(scale) * factor)
    if not values:
        return None
    anisotropic = len(values) > 1 and abs(values[1] / values[0] - 1) > .005
    return round(values[0], 12), anisotropic


def validate_roi(shape, roi):
    if len(roi) != 3:
        raise ValueError("ROI 须为 x、y、边长。")
    x, y, size = (int(round(v)) for v in roi)
    h, w = shape
    if size < MIN_ROI or x < 0 or y < 0 or x + size > w or y + size > h:
        raise ValueError(f"ROI 须为图内正方形，边长不少于 {MIN_ROI} px。")
    return x, y, size


def default_roi(shape, exclude=None):
    """Largest centred square; keep clear of a burnt-in scale bar when possible."""
    h, w = shape
    size = min(h, w)
    x, y = (w - size) // 2, (h - size) // 2
    if exclude:
        x1, y1, x2, y2 = exclude
        if y1 > .5 * h and int(y1) >= max(MIN_ROI, .5 * size):
            size = min(w, int(y1))
            x, y = (w - size) // 2, 0
    return x, y, size


def _band_spectrum(data, zoom):
    n = data.shape[0]
    if zoom == 1:
        return np.fft.fftshift(np.fft.fft2(data))
    f1 = -(n // 2) / (n * zoom)
    band = [f1, f1 + 1 / zoom]
    rows = zoom_fft(data, band, m=n, fs=1, axis=1)
    return zoom_fft(rows, band, m=n, fs=1, axis=0)


def compute_fft(raw, roi, nm_per_px, zoom=1):
    raw = np.asarray(raw, float)
    x, y, size = validate_roi(raw.shape, roi)
    if nm_per_px is not None and (not np.isfinite(nm_per_px) or nm_per_px <= 0):
        raise ValueError("实空间比例须为正数 nm/px。")
    if zoom not in ZOOMS:
        raise ValueError("FFT 显示倍率须为 1、2、4 或 8。")
    crop = raw[y:y + size, x:x + size]
    if not np.isfinite(crop).all():
        raise ValueError("ROI 内含无效像素 (NaN/Inf)。")
    mean = float(crop.mean())
    centered = crop - mean
    window = np.outer(np.hanning(size), np.hanning(size))
    spectrum = _band_spectrum(centered * window, zoom)
    plain = np.fft.fftshift(np.fft.fft2(centered))
    return FFTResult((x, y, size), float(nm_per_px) if nm_per_px else None,
                     spectrum, plain, mean, zoom)


def auto_zoom(raw, roi, fill=.42):
    """Largest zoom that keeps every detected lattice peak inside `fill` of the half width."""
    result = compute_fft(raw, roi, None)
    peaks = detect_peaks(result)
    if not peaks:
        return 1
    rmax = max(np.linalg.norm(p.xy - result.center) for p in peaks)
    return max(z for z in ZOOMS if z == 1 or rmax * z <= fill * result.size)


def _subpixel(m, x, y):
    """Gaussian three-point interpolation along each axis."""
    out = []
    for dx, dy in ((1, 0), (0, 1)):
        values = [m[y - dy, x - dx], m[y, x], m[y + dy, x + dx]]
        if min(values) <= 0:
            out.append(0.)
            continue
        a, b, c = np.log(values)
        denominator = a - 2 * b + c
        out.append(float(np.clip(.5 * (a - c) / denominator, -.5, .5)) if denominator < 0 else 0.)
    return x + out[0], y + out[1]


def detect_peaks(result, sensitivity=6.0, max_peaks=60, relative=.05, min_radius=None):
    """Friedel-symmetric FFT maxima as Spot objects (centre fixed at zero frequency)."""
    m = result.magnitude
    n = result.size
    c = result.center
    yy, xx = np.mgrid[:n, :n]
    r = np.hypot(xx - c[0], yy - c[1])
    min_radius = max(3., n / 120) if min_radius is None else min_radius
    usable = (r > min_radius) & (xx > 0) & (yy > 0) & (xx < n - 1) & (yy < n - 1)
    # one half plane; partners are added by symmetry
    half = (yy < c[1]) | ((yy == c[1]) & (xx > c[0]))
    sample = m[usable]
    floor = float(np.median(sample) + sensitivity * robust_sigma(sample))
    peaks = (m == ndi.maximum_filter(m, size=5)) & usable & half & (m > floor)
    ys, xs = np.nonzero(peaks)
    if not len(xs):
        return []
    values = m[ys, xs]
    keep = values >= relative * values.max()
    ys, xs, values = ys[keep], xs[keep], values[keep]
    order = np.argsort(-values)[:max(1, max_peaks // 2)]
    noise = max(robust_sigma(sample), 1e-12)
    spots = []
    for i in order:
        px, py = _subpixel(m, int(xs[i]), int(ys[i]))
        for sx, sy in ((px, py), (2 * c[0] - px, 2 * c[1] - py)):
            spot = Spot(float(sx), float(sy), float(values[i]))
            spot.peak = float(values[i] / noise)
            spots.append(spot)
    return spots


def soft_mask(n, points, radius, soft):
    """Union of cosine-edged discs at each point and its Friedel partner."""
    if radius <= 0 or soft < 0:
        raise ValueError("掩膜半径须为正数，软边不能为负。")
    yy, xx = np.mgrid[:n, :n]
    c = float(n // 2)
    mask = np.zeros((n, n))
    for x, y in points:
        for px, py in ((x, y), (2 * c - x, 2 * c - y)):
            d = np.hypot(xx - px, yy - py)
            if soft > 0:
                edge = .5 * (1 + np.cos(np.pi * np.clip((d - radius) / soft, 0, 1)))
            else:
                edge = (d <= radius).astype(float)
            mask = np.maximum(mask, edge)
    return mask


def inverse_fft(result, points, radius=4., soft=2.):
    """Filtered real-space image built from the retained complex spectrum."""
    points = [tuple(map(float, p)) for p in points]
    if not points:
        raise ValueError("请先选择用于 IFFT 的 FFT 斑点。")
    mask = soft_mask(result.size, [tuple(result.to_plain(p)) for p in points], radius, soft)
    image = np.fft.ifft2(np.fft.ifftshift(result.plain * mask)).real
    return (image + result.mean).astype(np.float32)


def to_uint8(data, low=.5, high=99.5):
    lo, hi = np.percentile(data, [low, high])
    return np.clip((data - lo) / max(hi - lo, 1e-12) * 255, 0, 255).astype(np.uint8)
