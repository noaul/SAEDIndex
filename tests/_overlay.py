"""Debug overlay: detected spots and fitted lattices drawn onto the image."""
import sys

import numpy as np
from PIL import Image, ImageDraw

from saedindex.calibration import load_image
from saedindex.pattern import detect_spots, find_center, find_lattices

ROOT = r"F:\E D\整理中-第四章稳定性及力学\00-热暴露\03-TEM\850-10000"
COLORS = [(255, 60, 60), (60, 200, 255), (255, 220, 0)]
for name in sys.argv[1:]:
    img = load_image(rf"{ROOT}\{name}.jpg")
    cal = img.calibration
    exclude = [cal.label_rect] if cal.label_rect else []
    spots = detect_spots(img.raw, exclude=exclude)
    center = find_center(img.raw, spots, exclude)
    lattices = find_lattices(spots, center)
    g = img.gray.astype(float)
    g = np.clip((g - np.percentile(g, 1)) / (np.percentile(g, 99.7) - np.percentile(g, 1)) * 255, 0, 255) ** 1
    canvas = Image.fromarray(g.astype(np.uint8)).convert("RGB")
    d = ImageDraw.Draw(canvas)
    for s in spots:
        d.ellipse([s.x - 7, s.y - 7, s.x + 7, s.y + 7], outline=(0, 255, 0))
    for k, lat in enumerate(lattices):
        c = COLORS[k % 3]
        for i in lat.members:
            s = spots[i]
            d.rectangle([s.x - 3 - k, s.y - 3 - k, s.x + 3 + k, s.y + 3 + k], outline=c)
        for v in (lat.g1, lat.g2):
            d.line([*lat.center, *(lat.center + v)], fill=c, width=2)
    d.ellipse([center[0] - 4, center[1] - 4, center[0] + 4, center[1] + 4], fill=(255, 0, 255))
    canvas.save(rf"D:\科研论文\SAEDIndex\tests\out_{name}.png")
    print(name, len(spots))
