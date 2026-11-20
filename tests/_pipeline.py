"""Diagnostic: calibration, spots, lattices and top candidates on real images."""
import sys
import time

import numpy as np

from saedindex.calibration import load_image
from saedindex.crystal import load_cif
from saedindex.indexing import index_lattice, measured_from_lattice
from saedindex.pattern import detect_spots, find_center, find_lattices

ROOT = r"F:\E D\整理中-第四章稳定性及力学\00-热暴露\03-TEM\850-10000"
NAMES = sys.argv[1:] or ["1-0004", "1-0005", "1-0006", "1-0009", "1-0010", "1-0011",
                         "1-0012", "1-0013", "1-0014"]
CIFS = ["Ni_mp-23", "NbNi3_mp-11513", "达尔塔NbNi3_mp-1451", "NbC_mp-910"]

phases = [load_cif(rf"D:\cif\{n}_conventional_standard.cif") for n in CIFS]
for name in NAMES:
    t = time.time()
    img = load_image(rf"{ROOT}\{name}.jpg")
    cal = img.calibration
    exclude = [cal.label_rect] if cal.label_rect else []
    spots = detect_spots(img.raw, exclude=exclude)
    center = find_center(img.raw, spots, exclude)
    lattices = find_lattices(spots, center)
    print(f"== {name}  {cal.px_per_unit:.3f} px/(1/nm) [{cal.source}]  spots={len(spots)}  "
          f"center={np.round(center, 1)}")
    for k, lat in enumerate(lattices):
        m = measured_from_lattice(lat, spots, cal.px_per_unit)
        print(f"  set{k}: n={len(lat.members)} d1={1/m['g1_len']:.4f} d2={1/m['g2_len']:.4f} "
              f"ang={m['angle']:.1f} rms={lat.rms:.2f}")
        for c in index_lattice(phases, m)[:4]:
            print(f"     {c.label:28s} {c.phase.hkl_text(c.h1)} {c.phase.hkl_text(c.h2)} "
                  f"err={c.d1_err*100:+.1f}%/{c.d2_err*100:+.1f}% ang={c.angle_theo:.1f} "
                  f"forb={c.forbidden:.2f} miss={c.missing:.2f} cost={c.cost:.2f}")
    print(f"  ({time.time() - t:.1f}s)")
