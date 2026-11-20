"""Minimal reader for Gatan DigitalMicrograph DM3/DM4 files.

Reads the main image array and its spatial calibration (scale and unit).
Large arrays that are not needed are skipped by seeking.
"""
from pathlib import Path
import struct

import numpy as np

SIMPLE = {2: "h", 3: "i", 4: "H", 5: "I", 6: "f", 7: "d", 8: "?", 9: "b", 10: "B",
          11: "q", 12: "Q"}
DATA_TYPES = {1: np.int16, 2: np.float32, 3: np.complex64, 6: np.uint8, 7: np.int32,
              9: np.int8, 10: np.uint16, 11: np.uint32, 12: np.float64,
              13: np.complex128, 14: np.uint8, 23: np.uint32, 39: np.int64, 40: np.uint64}


class _Reader:
    def __init__(self, handle):
        self.f = handle
        version, = struct.unpack(">I", self.f.read(4))
        if version not in (3, 4):
            raise ValueError("Not a DM3/DM4 file.")
        self.dm4 = version == 4
        self.f.read(8 if self.dm4 else 4)
        order, = struct.unpack(">I", self.f.read(4))
        self.endian = "<" if order == 1 else ">"

    def _count(self):
        if self.dm4:
            return struct.unpack(">Q", self.f.read(8))[0]
        return struct.unpack(">I", self.f.read(4))[0]

    def group(self):
        self.f.read(2)
        result, index = {}, 0
        for _ in range(self._count()):
            kind = self.f.read(1)[0]
            length, = struct.unpack(">H", self.f.read(2))
            name = self.f.read(length).decode("latin-1") or f"[{index}]"
            if not length:
                index += 1
            if self.dm4:
                self.f.read(8)
            if kind == 20:
                result[name] = self.group()
            elif kind == 21:
                result[name] = self.data()
            else:
                raise ValueError("Corrupt DM tag directory.")
        return result

    def _read(self, code):
        fmt = self.endian + SIMPLE[code]
        return struct.unpack(fmt, self.f.read(struct.calcsize(fmt)))[0]

    def data(self):
        if self.f.read(4) != b"%%%%":
            raise ValueError("Corrupt DM data tag.")
        info = [self._count() for _ in range(self._count())]
        code = info[0]
        if code in SIMPLE:
            return self._read(code)
        if code == 18:
            return self.f.read(info[1] * 2).decode("utf-16-le", "replace")
        if code == 15:
            return tuple(self._read(info[4 + 2 * i]) for i in range(info[2]))
        if code == 20:
            element, = info[1:2]
            if element == 15:
                fields = [info[5 + 2 * i] for i in range(info[3])]
                count = info[-1]
                size = sum(struct.calcsize(SIMPLE[f]) for f in fields)
                self.f.seek(size * count, 1)
                return None
            count = info[2]
            dtype = np.dtype(SIMPLE[element]).newbyteorder(self.endian)
            position = self.f.tell()
            self.f.seek(dtype.itemsize * count, 1)
            if element in (4,) and count < 512:
                self.f.seek(position)
                raw = np.frombuffer(self.f.read(dtype.itemsize * count), dtype)
                return "".join(map(chr, raw))
            return ("array", position, element, count)
        raise ValueError(f"Unsupported DM data type {code}.")


def _search(tags, name):
    for key, value in tags.items():
        if key == name and not isinstance(value, dict):
            return value
        if isinstance(value, dict):
            found = _search(value, name)
            if found is not None:
                return found
    return None


def read_dm(path, load_data=True, info=None):
    """Return (image or None, scale, unit, shape) for the main image of a DM3/DM4 file.

    When `info` is a dict it receives the accelerating voltage (V) and the
    operation mode found in the image tags.
    """
    with open(path, "rb") as handle:
        reader = _Reader(handle)
        tags = reader.group()
        images = list(tags.get("ImageList", {}).values())
        if not images:
            raise ValueError("The DM file contains no image.")
        # The first image is usually a thumbnail; the main image is the largest.
        def pixels(entry):
            dims = entry.get("ImageData", {}).get("Dimensions", {})
            values = [v for v in dims.values() if isinstance(v, int)]
            return int(np.prod(values)) if values else 0
        entry = max(images, key=pixels)
        main = entry["ImageData"]
        if info is not None:
            image_tags = entry.get("ImageTags", {})
            voltage = _search(image_tags, "Voltage")
            info.update(voltage=float(voltage) if isinstance(voltage, (int, float)) else None,
                        mode=_search(image_tags, "Operation Mode"))
        dims = [main["Dimensions"][key] for key in sorted(main["Dimensions"],
                key=lambda k: int(k.strip("[]")))]
        calibration = main.get("Calibrations", {}).get("Dimension", {})
        first = calibration.get("[0]", {})
        scale = float(first.get("Scale", 1.0))
        unit = first.get("Units", "") or ""
        unit = unit.replace("\u00b5", "u").replace("\u03bc", "u")
        if info is not None:
            axes = []
            for key in sorted(calibration, key=lambda k: int(k.strip("[]"))):
                axis_scale = float(calibration[key].get("Scale", 1.0))
                axis_unit = (calibration[key].get("Units", "") or "").replace(
                    "\u00b5", "u").replace("\u03bc", "u")
                if axis_unit == "um":
                    axis_scale, axis_unit = axis_scale * 1000, "nm"
                axes.append((axis_scale, axis_unit))
            info["axis_scales"] = axes
        image = None
        if load_data:
            _, position, element, count = main["Data"]
            dtype = np.dtype(DATA_TYPES.get(main.get("DataType"), SIMPLE[element]))
            dtype = dtype.newbyteorder(reader.endian)
            handle.seek(position)
            raw = np.frombuffer(handle.read(dtype.itemsize * count), dtype)
            width, height = dims[0], dims[1] if len(dims) > 1 else 1
            image = raw[: width * height].reshape(height, width).astype(np.float64)
    shape = (dims[1], dims[0]) if len(dims) > 1 else (1, dims[0])
    if unit == "um":
        scale, unit = scale * 1000, "nm"
    return image, scale, unit, shape


def find_companion(path):
    """Original DM file saved next to an exported image with the same name."""
    path = Path(path)
    for suffix in (".dm4", ".dm3", ".DM4", ".DM3"):
        candidate = path.with_suffix(suffix)
        if candidate.exists():
            return candidate
    return None
