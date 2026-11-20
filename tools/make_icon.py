"""Render assets/app.ico and assets/app.png (a stylised indexed SAED pattern).

Run: .\\venv\\Scripts\\python.exe tools\\make_icon.py
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ASSETS = Path(__file__).resolve().parents[1] / "assets"
BG_TOP, BG_BOTTOM = (24, 36, 38), (10, 15, 17)
TEAL = (52, 196, 176)
AMBER = (255, 196, 64)


def render(size, simple=False):
    ss = 4  # supersampling for anti-aliased geometry
    n = size * ss
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32) / n

    # Rounded-square plate with a soft vertical gradient.
    t = yy[..., None]
    rgb = (1 - t) * np.array(BG_TOP) + t * np.array(BG_BOTTOM)

    cx = cy = 0.5
    angle = np.deg2rad(-16)
    step = 0.25 if simple else 0.2
    g1 = step * np.array([np.cos(angle), np.sin(angle)])
    g2 = step * 0.82 * np.array([-np.sin(angle), np.cos(angle)])

    light = np.zeros((n, n), np.float32)
    rng = range(-1, 2) if simple else range(-4, 5)
    spots = []
    for i in rng:
        for j in rng:
            px, py = cx + i * g1[0] + j * g2[0], cy + i * g1[1] + j * g2[1]
            r = np.hypot(px - cx, py - cy)
            if r > 0.43 or min(px, py) < 0.07 or max(px, py) > 0.93:
                continue
            spots.append((i, j, px, py))
            if i == 0 and j == 0:
                sigma, amp = (0.05 if simple else 0.03), 1.0
            else:
                sigma = 0.034 if simple else 0.013
                amp = (0.95 if simple else 0.9) * np.exp(-(r / 0.36) ** 2) + 0.25
            light += amp * np.exp(-((xx - px) ** 2 + (yy - py) ** 2) / (2 * sigma ** 2))
    # Faint diffuse halo around the transmitted beam.
    light += 0.18 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 0.16 ** 2))
    light = np.clip(light, 0, 1)[..., None]
    rgb = rgb * (1 - light) + np.array([246, 250, 250]) * light

    img = Image.fromarray(rgb.astype(np.uint8), "RGB").convert("RGBA")
    draw = ImageDraw.Draw(img)

    # Indexing parallelogram: 000 -> g1 -> g1+g2 -> g2 (teal), g1 and g2 ringed.
    p = lambda v: (float(v[0] * n), float(v[1] * n))
    o = np.array([cx, cy])
    corners = [o, o + g1, o + g1 + g2, o + g2]
    width = int(n * (0.05 if simple else 0.026))
    draw.line([p(c) for c in corners + [o]], fill=TEAL + (255,), width=width, joint="curve")
    ring = n * (0.075 if simple else 0.048)
    for c in (o + g1, o + g2):
        x, y = p(c)
        draw.ellipse([x - ring, y - ring, x + ring, y + ring], outline=AMBER + (255,),
                     width=max(1, int(width * 0.75)))

    # Rounded-square alpha mask.
    mask = Image.new("L", (n, n), 0)
    radius = int(n * 0.2)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, n - 1, n - 1], radius=radius, fill=255)
    img.putalpha(mask)
    return img.resize((size, size), Image.Resampling.LANCZOS)


def main():
    sizes = [16, 20, 24, 32, 40, 48, 64, 128, 256]
    frames = [render(s, simple=s <= 32) for s in sizes]
    big = render(512)
    big.save(ASSETS / "app.png")
    frames[-1].save(ASSETS / "app.ico", format="ICO", sizes=[(s, s) for s in sizes],
                    append_images=frames[:-1])
    preview = Image.new("RGBA", (sum(sizes) + 10 * len(sizes) + 10, 276), (255, 255, 255, 255))
    x = 10
    for s, f in zip(sizes, frames):
        preview.alpha_composite(f, (x, 266 - s))
        x += s + 10
    preview.save(ASSETS.parent / "verification" / "icon-sizes.png")


if __name__ == "__main__":
    main()
