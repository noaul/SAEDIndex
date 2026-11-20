"""Image loading and automatic reciprocal-space calibration.

Order of preference:
1. DigitalMicrograph metadata (the opened .dm3/.dm4, or one with the same name
   next to an exported image) when its unit is 1/nm.
2. Scale bar found in the image plus its label read by template matching.
3. Manual input.
"""
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from scipy import ndimage as ndi

from .dmfile import read_dm, find_companion

DM_SUFFIXES = (".dm3", ".dm4")
FONT_DIR = Path("C:/Windows/Fonts")
OCR_FONTS = ("lucon.ttf", "consola.ttf", "cour.ttf", "courbd.ttf", "micross.ttf", "tahoma.ttf",
             "arial.ttf", "arialbd.ttf", "segoeui.ttf", "calibri.ttf", "times.ttf", "timesbd.ttf",
             "verdana.ttf")
NICE_VALUES = [m * 10.0 ** e for e in range(-2, 4) for m in (1, 2, 2.5, 5)]


@dataclass
class Calibration:
    px_per_unit: float | None = None      # pixels per 1/nm
    source: str = "未标定"
    confident: bool = False
    bar: dict | None = None               # x1, x2, y, top, bottom, pixels
    label_rect: list | None = None        # area covered by the old bar and label
    label_value: float | None = None
    label_unit: str = ""
    label_text: str = ""
    notes: list = field(default_factory=list)
    proposed_px_per_unit: float | None = None
    reference_points: list | None = None
    reference_value: float | None = None
    reference_unit: str = ""
    metadata_px_per_unit: float | None = None
    label_px_per_unit: float | None = None
    label_confidence: float = 0.0
    conflict: bool = False

    @property
    def unit_per_px(self):
        return 1.0 / self.px_per_unit if self.px_per_unit else None


@dataclass
class LoadedImage:
    path: Path
    raw: np.ndarray            # float32, original values
    gray: np.ndarray           # uint8 for display and detection
    calibration: Calibration
    meta: dict


def _to_uint8(data, low_pct=0.05, high_pct=99.95):
    low, high = np.nanpercentile(data, [low_pct, high_pct])
    return np.clip((data - low) / max(high - low, 1e-12) * 255, 0, 255).astype(np.uint8)


def _dm_calibration(path):
    info = {}
    _, scale, unit, shape = read_dm(path, load_data=False, info=info)
    info["dm_shape"] = list(shape)
    unit_text = unit.replace(" ", "").lower()
    factor = None
    if unit_text in ("1/nm", "nm^-1", "nm-1"):
        factor = 1.0
    elif unit_text in ("1/a", "1/\u00e5", "a^-1"):
        factor = 10.0
    elif unit_text in ("1/um", "1/\u00b5m"):
        factor = 1e-3
    if factor is None or not scale:
        return None, info
    per_px = scale * factor
    return Calibration(px_per_unit=1.0 / per_px, source=f"DM 元数据 ({Path(path).name})",
                       confident=True, notes=[f"{per_px:.6g} 1/nm/px"]), info


def load_image(path):
    path = Path(path)
    meta = {"path": str(path)}
    calibration = Calibration()
    if path.suffix.lower() in DM_SUFFIXES:
        data, _, _, _ = read_dm(path, info=meta)
        raw = data.astype(np.float32)
        gray = _to_uint8(raw, 0.1, 99.97)
        cal, _ = _dm_calibration(path)
        if cal:
            calibration = cal
    else:
        with Image.open(path) as source:
            source.load()
            image = ImageOps.exif_transpose(source)
            array = np.asarray(image)
        if array.ndim == 3:
            array = np.asarray(image.convert("L"))
        raw = array.astype(np.float32)
        gray = array if array.dtype == np.uint8 else _to_uint8(raw, 0.0, 100.0)
        companion = find_companion(path)
        if companion:
            try:
                cal, info = _dm_calibration(companion)
                meta.update(info)
                if cal and tuple(info.get("dm_shape", ())) == raw.shape:
                    calibration = cal
                elif cal:
                    calibration.notes.append("图片与同名 DM 尺寸不同，未沿用原 DM 比例；请核对是否缩放或裁剪")
            except (OSError, ValueError, KeyError) as error:
                meta["dm_error"] = str(error)
    calibration.metadata_px_per_unit = calibration.px_per_unit
    bar = detect_scale_bar(gray)
    if bar:
        label = read_bar_label(gray, bar)
        calibration.bar = bar
        calibration.label_rect = label["rect"]
        calibration.label_value = label["value"]
        calibration.label_unit = label["unit"]
        calibration.label_text = label["text"]
        calibration.label_confidence = float(label["confidence"])
        reciprocal_label = label["unit"] in ("1/nm", "1/A", "1/\u00c5")
        if label["value"] and reciprocal_label:
            factor = 10.0 if label["unit"] in ("1/A", "1/\u00c5") else 1.0
            calibration.label_px_per_unit = bar["pixels"] / (label["value"] * factor)
        if calibration.px_per_unit is None and label["value"] and reciprocal_label:
            factor = 10.0 if label["unit"] in ("1/A", "1/\u00c5") else 1.0
            calibration.px_per_unit = bar["pixels"] / (label["value"] * factor)
            calibration.source = f"标尺自动识别 ({bar['pixels']:.0f} px = {label['text']})"
            calibration.confident = label["confidence"] > 0.55 and label["unit"].startswith("1/")
            if not calibration.confident:
                calibration.notes.append("标尺数字识别置信度偏低，请核对数值")
                calibration.proposed_px_per_unit = calibration.px_per_unit
                calibration.px_per_unit = None
                calibration.source = f"标尺待核对 ({label['text']})"
        elif calibration.px_per_unit and label["value"] and reciprocal_label:
            # Cross-check the DM metadata with the burnt-in bar.
            factor = 10.0 if label["unit"] in ("1/A", "1/\u00c5") else 1.0
            expected = calibration.px_per_unit * label["value"] * factor
            calibration.notes.append(f"标尺 {bar['pixels']:.1f} px，元数据预期 {expected:.1f} px "
                                     f"({(bar['pixels'] / expected - 1) * 100:+.2f}%)")
            if label["confidence"] > .55 and abs(bar["pixels"] / expected - 1) > .03:
                calibration.conflict = True
                calibration.confident = False
                calibration.proposed_px_per_unit = calibration.px_per_unit
                calibration.px_per_unit = None
                calibration.source = "DM 与图中标尺冲突，待人工核对"
                calibration.notes.append("两个来源相差超过 3%，未自动使用任一比例")
            elif label["confidence"] <= .55:
                calibration.notes.append("图中文字置信度偏低，保留 DM 比例；请核对图中标尺")
        elif label["value"] and not reciprocal_label:
            calibration.notes.append("识别到实空间标尺或单位不明，未用于衍射标定")
    return LoadedImage(path, raw, np.ascontiguousarray(gray), calibration, meta)


# ------------------------------------------------------------- scale bar


# Empirical offsets from 155 Gatan SAED exports against their DM3 calibration:
# the dark core of an outlined bar is 0.73 +- 0.24 px shorter than the nominal
# length, the sub-pixel length of a solid bar 0.88 +- 0.19 px longer.
OUTLINED_OFFSET = 0.73
SOLID_OFFSET = -0.88


def _bar_result(top, y, x1, inner, left, right, bottom):
    offset = OUTLINED_OFFSET if left + right > 0 else SOLID_OFFSET
    return {"x1": int(x1 - left), "x2": int(round(x1 + inner - 1 + right)), "y": float(y),
            # pixel-edge start of the measured length, centred on the bar core
            "start": float(x1) - offset / 2,
            "pixels": float(inner) + offset, "measured": float(inner),
            "outer": float(inner + left + right),
            "outlined": left + right > 0, "top": int(top), "bottom": int(bottom)}


def _framed_runs(row_dark, row_bright):
    """Dark runs whose both neighbours are bright pixels."""
    edges = np.diff(np.concatenate(([0], row_dark.view(np.int8), [0])))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    runs = []
    for s, e in zip(starts, stops):
        if s > 0 and e < len(row_dark) and row_bright[s - 1] and row_bright[e]:
            runs.append((int(s), int(e) - 1))
    return runs


def detect_scale_bar(gray):
    """Horizontal scale bar in the bottom 30 % of the image.

    DM bars are a dark core framed by a bright outline; the core length is
    used (within 0.6 % of the DM calibration on 262 Gatan exports). Solid
    bright bars are measured end to end with sub-pixel ends.
    """
    gray = np.asarray(gray, dtype=np.float32)
    h, w = gray.shape
    if h < 40 or w < 60:
        return None
    start = int(h * 0.7)
    region = gray[start:]
    minimum = max(30, w * 0.06)
    dark, bright = region <= 70, region >= 170
    candidates = []
    open_runs = {}
    rows_runs = [[r for r in _framed_runs(dark[y], bright[y]) if r[1] - r[0] + 1 >= minimum]
                 for y in range(region.shape[0])]
    for y, runs in enumerate(rows_runs):
        for run in runs:
            key = next((k for k in open_runs if abs(k[0] - run[0]) <= 1 and abs(k[1] - run[1]) <= 1
                        and open_runs[k][1] == y - 1), None)
            if key:
                first, _ = open_runs.pop(key)
                open_runs[run] = (first, y)
            else:
                open_runs[run] = (y, y)
    for (left, right), (top, bottom) in open_runs.items():
        thickness = bottom - top + 1
        core = right - left + 1
        if thickness < 2 or core / thickness < 5 or thickness > h * 0.05:
            continue
        span = slice(left, right + 1)
        # bright outline rows above and below
        up = 0
        while up < 6 and top - up - 1 >= 0 and bright[top - up - 1, span].mean() > 0.9:
            up += 1
        down = 0
        while down < 6 and bottom + down + 1 < region.shape[0] and bright[bottom + down + 1, span].mean() > 0.9:
            down += 1
        if not up or not down:
            continue
        side = 0
        while side < 6 and left - side - 1 >= 0 and bright[top:bottom + 1, left - side - 1].mean() > 0.8:
            side += 1
        candidates.append((core, _bar_result(start + top - up, start + (top + bottom) / 2, left, core,
                                             side, side, start + bottom + down)))
    if candidates:
        return max(candidates, key=lambda item: item[0])[1]
    labels, _ = ndi.label(region >= 200)
    for i, box in enumerate(ndi.find_objects(labels), 1):
        part = labels[box] == i
        rows = np.flatnonzero(part.mean(axis=1) > 0.9)
        if not len(rows):
            continue
        y1, y2 = box[0].start + rows[0], box[0].start + rows[-1] + 1
        cols = np.flatnonzero(part[rows].all(axis=0))
        if not len(cols):
            continue
        x1, x2 = box[1].start + cols[0], box[1].start + cols[-1] + 1
        length, thickness = x2 - x1, y2 - y1
        if length < minimum or length / max(thickness, 1) < 8 or thickness > h * 0.06:
            continue
        profile = region[y1:y2, max(0, x1 - 3):min(w, x2 + 3)].mean(axis=0)
        background = float(np.median(region))
        level = (profile.max() + background) / 2
        exact = float(np.clip((profile - background) / max(profile.max() - background, 1), 0, 1).sum())
        if abs(exact - length) <= 2:
            length = exact
        candidates.append((length, _bar_result(start + y1, start + (y1 + y2 - 1) / 2, x1, length,
                                               0, 0, start + y2 - 1)))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


# ------------------------------------------------------------ label OCR

def _normalise(mask, size=(24, 32)):
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return np.zeros(size[::-1], np.float32)
    crop = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.uint8) * 255
    h, w = crop.shape
    scale = min(size[0] / w, size[1] / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    image = Image.fromarray(crop).resize((nw, nh), Image.Resampling.BILINEAR)
    canvas = Image.new("L", size, 0)
    canvas.paste(image, ((size[0] - nw) // 2, (size[1] - nh) // 2))
    array = ndi.gaussian_filter(np.asarray(canvas, np.float32), 1.6)
    array -= array.mean()
    norm = np.linalg.norm(array)
    return array / norm if norm else array


@lru_cache(maxsize=None)
def _templates():
    templates = []
    for font_name in OCR_FONTS:
        path = FONT_DIR / font_name
        if not path.exists():
            continue
        for size in (40, 64):
            font = ImageFont.truetype(str(path), size)
            for char in "0123456789":
                image = Image.new("L", (110, 120), 0)
                ImageDraw.Draw(image).text((20, 10), char, font=font, fill=255)
                filled = np.asarray(image) > 110
                templates.append((char, _normalise(filled)))
                thin = ndi.binary_erosion(filled, iterations=max(1, size // 30))
                if thin.sum() > 20:
                    templates.append((char, _normalise(thin)))
    return templates


def _classify(mask):
    vector = _normalise(mask)
    best = {}
    for char, template in _templates():
        score = float((vector * template).sum())
        best[char] = max(best.get(char, -1), score)
    # topology: closed loops distinguish 0/6/8/9 from 1/2/3/5/7
    filled = ndi.binary_fill_holes(mask)
    holes = ndi.label(filled & ~mask)[1]
    expected = {"0": 1, "4": (0, 1), "6": 1, "8": 2, "9": 1}
    for char in best:
        want = expected.get(char, 0)
        ok = holes in want if isinstance(want, tuple) else holes == want
        if not ok:
            best[char] -= 0.15
    ranked = sorted(best.items(), key=lambda item: -item[1])
    return ranked[0][0], ranked[0][1], ranked[1][1]


def _framed(core, halo, minimum=0.6):
    """Components of `core` whose outer border lies mostly on `halo` pixels."""
    labels, count = ndi.label(core, np.ones((3, 3)))
    if not count:
        return core
    ring = ndi.binary_dilation(core, np.ones((3, 3))) & ~core
    ring_labels = ndi.grey_dilation(labels, size=(3, 3)) * ring
    total = np.bincount(ring_labels.ravel(), minlength=count + 1)
    hits = np.bincount(ring_labels[halo].ravel(), minlength=count + 1)
    ratio = hits / np.maximum(total, 1)
    keep = np.flatnonzero(ratio >= minimum)
    keep = keep[keep > 0]
    return np.isin(labels, keep)


def _slanted(piece):
    rows = np.array([np.flatnonzero(r).mean() if r.any() else np.nan for r in piece])
    valid = ~np.isnan(rows)
    if valid.sum() < 5:
        return False
    slope = np.polyfit(np.flatnonzero(valid), rows[valid], 1)[0]
    return slope < -0.25


def read_bar_label(gray, bar):
    """Read the number and unit printed next to a scale bar."""
    gray = np.asarray(gray, dtype=np.float32)
    h, w = gray.shape
    bar_len = bar["x2"] - bar["x1"]
    height = max(24, int(0.5 * bar_len), int(0.08 * h))
    x1, x2 = max(0, bar["x1"] - int(0.3 * bar_len)), min(w, bar["x2"] + int(1.2 * bar_len))
    result = {"value": None, "unit": "", "text": "", "confidence": 0.0,
              "rect": [max(0, bar["x1"] - 6), max(0, bar["top"] - 6), min(w, bar["x2"] + 7),
                       min(h, bar["bottom"] + 7)]}
    best = None
    attempts = []
    for y1, y2 in ((max(0, bar["top"] - height), bar["top"] - 1),
                   (bar["bottom"] + 2, min(h, bar["bottom"] + 2 + height))):
        if y2 - y1 < 8:
            continue
        region = gray[y1:y2, x1:x2]
        # Labels are white or black glyphs with a contrasting rim; both differ
        # strongly from the local background, so take the absolute contrast.
        small = ndi.median_filter(region[::3, ::3], size=(3, 55), mode="nearest")
        background = np.repeat(np.repeat(small, 3, 0), 3, 1)[:region.shape[0], :region.shape[1]]
        contrast = np.abs(region - background)
        level = max(70.0, 0.45 * float(np.percentile(contrast, 99.7)))
        attempts.append((y1, region, contrast >= level))
    for y1, region, mask in attempts:
        if mask.sum() < 20 or mask.mean() > 0.5:
            continue
        labels, _ = ndi.label(mask, np.ones((3, 3)))
        comps = []
        for i, box in enumerate(ndi.find_objects(labels), 1):
            if (labels[box] == i).sum() < 4:
                continue
            comps.append({"ids": [i], "y1": box[0].start, "y2": box[0].stop,
                          "x1": box[1].start, "x2": box[1].stop})
        # Merge broken strokes of one glyph (horizontal overlap).
        comps.sort(key=lambda c: c["x1"])
        merged = []
        for c in comps:
            m = next((m for m in merged if min(m["x2"], c["x2"]) - max(m["x1"], c["x1"])
                      > 0.5 * min(m["x2"] - m["x1"], c["x2"] - c["x1"])), None)
            if m:
                m.update(x1=min(m["x1"], c["x1"]), x2=max(m["x2"], c["x2"]), y1=min(m["y1"], c["y1"]),
                         y2=max(m["y2"], c["y2"]), ids=m["ids"] + c["ids"])
            else:
                merged.append(dict(c))
        for m in merged:
            m["h"] = m["y2"] - m["y1"]
        if not merged:
            continue
        # Digit height: the label number starts at the left end of the bar.
        bar_left = bar["x1"] - x1
        lead = [m for m in merged if m["h"] >= 8 and m["x2"] >= bar_left - 4
                and m["x1"] <= bar_left + 0.25 * bar_len]
        if not lead:
            continue
        first = min(lead, key=lambda m: (m["x1"] if m["h"] >= 0.6 * max(c["h"] for c in lead) else 1e9))
        tall = float(first["h"])
        base = first["y2"]
        glyphs = [m for m in merged if 0.5 * tall <= m["h"] <= 1.5 * tall and m["x1"] >= first["x1"]
                  and (abs(m["y2"] - base) < 0.25 * tall
                       or (m["h"] >= 1.05 * tall and -0.1 * tall <= m["y2"] - base < 0.45 * tall))]
        glyphs.sort(key=lambda m: m["x1"])
        # Stop at the first large gap: the label is one line of text.
        line = [glyphs[0]]
        for m in glyphs[1:]:
            if m["x1"] - line[-1]["x2"] > 1.6 * tall:
                break
            line.append(m)
        kinds = []
        for m in line:
            piece = np.isin(labels[m["y1"]:m["y2"], m["x1"]:m["x2"]], m["ids"])
            if m["h"] >= 0.9 * tall and _slanted(piece):
                kinds.append("/")
            elif m["h"] < 0.82 * tall:
                kinds.append("a")          # lower-case letters of the unit
            else:
                kinds.append("d")
        slash = kinds.index("/") if "/" in kinds else None
        unit = "1/nm" if slash is not None else "nm"
        end = slash - 1 if slash is not None else (kinds.index("a") if "a" in kinds else len(kinds))
        for k in range(1, max(end, 0)):
            if line[k]["x1"] - line[k - 1]["x2"] > 0.6 * tall:   # word space
                end = k
                break
        number = line[:max(end, 0)]
        if not number or len(number) > 4:
            continue
        digits, scores = "", []
        for k, m in enumerate(number):
            piece = np.isin(labels[m["y1"]:m["y2"], m["x1"]:m["x2"]], m["ids"])
            char, score, second = _classify(piece)
            if k:
                small = [c for c in merged if c not in glyphs and c["x1"] >= number[k - 1]["x2"] - 1
                         and c["x2"] <= m["x1"] + 1 and c["y2"] >= base - 0.3 * tall
                         and c["h"] < 0.35 * tall]
                if small:
                    digits += "."
            digits += char
            scores.append(score - max(0.0, 0.05 - (score - second)))
        try:
            value = float(digits)
        except ValueError:
            continue
        confidence = float(np.mean(scores))
        if value <= 0 or min(abs(value / v - 1) for v in NICE_VALUES) > 1e-6:
            confidence *= 0.7
        box = [x1 + min(m["x1"] for m in line), y1 + min(m["y1"] for m in line),
               x1 + max(m["x2"] for m in line), y1 + max(m["y2"] for m in line)]
        candidate = {"value": value, "unit": unit, "confidence": confidence,
                     "text": f"{digits} {unit}", "box": box}
        if best is None or candidate["confidence"] > best["confidence"]:
            best = candidate
    if best:
        result.update({k: best[k] for k in ("value", "unit", "text", "confidence")})
        bx1, by1, bx2, by2 = best["box"]
        r = result["rect"]
        result["rect"] = [max(0, min(r[0], bx1 - 6)), max(0, min(r[1], by1 - 6)),
                          min(w, max(r[2], bx2 + 6)), min(h, max(r[3], by2 + 6))]
    return result


def nice_bar_value(px_per_unit, image_width, fraction=0.2):
    target = fraction * image_width / px_per_unit
    options = [v for v in NICE_VALUES if v * px_per_unit <= 0.4 * image_width]
    return min(options or NICE_VALUES, key=lambda v: abs(np.log(v / target)))
