import tempfile
import shutil
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from saedindex.calibration import load_image
from saedindex.crystal import Phase, Atom
from saedindex.pattern import Lattice, Spot
from saedindex.session import Session
from saedindex.indexing import evaluate


def cubic_session():
    session = Session()
    phase = Phase("test", (.4, .4, .4, 90, 90, 90), [Atom("Ni", np.zeros(3))])
    session.phases = [phase]
    session.px_per_unit = 40.0
    session.center = np.array([256., 256.])
    # Deliberate 10% image/CIF discrepancy
    lat = Lattice(np.array([110., 0.]), np.array([0., -110.]), session.center.copy())
    for n in ((1, 0), (0, 1), (1, 1), (-1, 0), (0, -1), (-1, -1)):
        xy = lat.position(*n)
        lat.members[len(session.spots)] = n
        session.spots.append(Spot(*xy, 100.))
    s = session._append_set(lat)
    c = evaluate(phase, (1, 0, 0), (0, 1, 0), session.measured(s))
    s.custom = c
    from saedindex.calibration import LoadedImage, Calibration
    session.image = LoadedImage(Path("synthetic.png"), np.zeros((512, 512), np.float32),
                                np.zeros((512, 512), np.uint8), Calibration(40.), {})
    return session, s


class CoreTests(unittest.TestCase):
    def test_supplied_pattern_center_avoids_half_lattice_translation(self):
        from saedindex.pattern import detect_spots, find_center
        source = Path("C:/Users/aodo/AppData/Local/Temp/codex-clipboard-b218fabb-5c4c-4ecd-8986-febb475dc307.jpg")
        if not source.exists():
            self.skipTest("User reference image unavailable")
        image = load_image(source)
        exclude = [image.calibration.label_rect]
        spots = detect_spots(image.raw, exclude=exclude)
        actual = find_center(image.raw, spots, exclude)
        np.testing.assert_allclose(actual, [493., 480.5], atol=2.)

    def test_delta_pattern_center_remains_on_transmitted_beam(self):
        from saedindex.pattern import detect_spots, find_center
        source = Path(r"F:\E D\整理中-第四章稳定性及力学\00-热暴露\03-TEM\850-10000\1-0013.jpg")
        if not source.exists():
            self.skipTest("Local development dataset unavailable")
        image = load_image(source)
        exclude = [image.calibration.label_rect]
        actual = find_center(image.raw, detect_spots(image.raw, exclude=exclude), exclude)
        np.testing.assert_allclose(actual, [510.7, 468.7], atol=2.)

    def test_saturated_disc_centroid_uses_whole_disc_not_plateau_noise(self):
        from saedindex.pattern import detect_spots
        yy, xx = np.mgrid[:128, :128]
        rng = np.random.default_rng(7)
        data = np.clip(8 + 1300 * np.exp(-((xx - 64.3) ** 2 + (yy - 63.7) ** 2) / 35.)
                       + rng.normal(0, 1.5, xx.shape), 0, 255).astype(np.float32)
        spots = detect_spots(data)
        strongest = max(spots, key=lambda p: p.intensity)
        self.assertLess(np.linalg.norm(strongest.xy - [64.3, 63.7]), .5)
        self.assertGreater(strongest.intensity, 10000)

    def test_center_uses_friedel_pairs_despite_brighter_off_axis_reflection(self):
        from saedindex.pattern import find_center
        center = np.array([500., 470.])
        spots = [Spot(*(center + [90*a, 130*b]), 100.)
                 for a in range(-2, 3) for b in range(-2, 3)]
        spots.append(Spot(650., 350., 5000.))
        data = np.zeros((1024, 1024), np.float32)
        for p in spots:
            data[int(p.y), int(p.x)] = p.intensity
        actual = find_center(data, spots)
        np.testing.assert_allclose(actual, center, atol=1.)

    def test_resized_export_does_not_reuse_unscaled_companion_metadata(self):
        root = Path(r"F:\E D\整理中-第四章稳定性及力学\00-热暴露\03-TEM\850-10000")
        if not (root / "1-0013.dm3").exists():
            self.skipTest("Local development dataset unavailable")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "resized.jpg"
            with Image.open(root / "1-0013.jpg") as image:
                image.resize((512, 512)).save(path)
            shutil.copyfile(root / "1-0013.dm3", path.with_suffix(".dm3"))
            calibration = load_image(path).calibration
            if calibration.px_per_unit is not None:
                self.assertAlmostEqual(calibration.px_per_unit, 38.154 / 2, delta=.3)
            self.assertFalse(calibration.source.startswith("DM 元数据"))

    def test_known_low_confidence_misread_is_not_applied(self):
        source = Path(r"F:\E D\整理中-第四章稳定性及力学\00-热暴露\03-TEM\850-100\20W-4Nb-0.05\1-0012.jpg")
        if not source.exists():
            self.skipTest("Local development dataset unavailable")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "image_only.jpg"
            shutil.copyfile(source, path)
            calibration = load_image(path).calibration
            self.assertFalse(calibration.confident)
            self.assertIsNone(calibration.px_per_unit)
            self.assertIsNotNone(calibration.proposed_px_per_unit)

    def test_real_space_scale_must_not_be_used_for_diffraction(self):
        image = Image.new("L", (1024, 1024), 4)
        d = ImageDraw.Draw(image)
        font = ImageFont.truetype("C:/Windows/Fonts/timesbd.ttf", 60)
        d.text((30, 900), "5 nm", font=font, fill=255)
        d.rectangle((30, 980, 220, 992), fill=255)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "real_space.png"
            image.save(path)
            loaded = load_image(path)
            self.assertIsNone(loaded.calibration.px_per_unit)

    def test_simulation_preserves_image_calibration_by_default(self):
        session, s = cubic_session()
        reflections, _, scale = session.simulation(s)
        p = next(r for r in reflections if r[0] == tuple(s.indices()[0]))
        self.assertAlmostEqual(np.hypot(p[1] - 256, p[2] - 256), 100., places=5)
        self.assertAlmostEqual(scale, 1.0)

    def test_free_scale_is_explicit_and_reports_correction(self):
        session, s = cubic_session()
        reflections, _, scale = session.simulation(s, free_scale=True)
        p = next(r for r in reflections if r[0] == tuple(s.indices()[0]))
        self.assertAlmostEqual(np.hypot(p[1] - 256, p[2] - 256), 110., places=5)
        self.assertAlmostEqual(scale, 1.1)

    def test_manual_indices_reject_zero_and_collinear_pairs(self):
        session, s = cubic_session()
        for a, b in [((0, 0, 0), (1, 0, 0)), ((1, 0, 0), (2, 0, 0))]:
            with self.subTest(a=a), self.assertRaises(ValueError):
                session.set_custom(s, a, b)

    def test_vector_closure_is_measured_not_constructed(self):
        session, s = cubic_session()
        session.spots[2].x += 2
        rows = session.vector_rows(s)
        self.assertAlmostEqual(rows[2]["closure_px"], 2.)


if __name__ == "__main__":
    unittest.main()
