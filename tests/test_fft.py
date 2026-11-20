import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from saedindex.crystal import Atom, Phase
from saedindex.fft import auto_zoom, compute_fft, detect_peaks, inverse_fft, real_scale_from_axes
from saedindex.session import Session


def fringes(shape, waves, phase=0.):
    """Sum of cosines; each wave is (kx, ky) in cycles per pixel and an amplitude."""
    yy, xx = np.mgrid[:shape[0], :shape[1]].astype(float)
    image = np.full(shape, 100.)
    for (kx, ky), amplitude in waves:
        image += amplitude * np.cos(2 * np.pi * (kx * xx + ky * yy) + phase)
    return image.astype(np.float32)


def peak_frequencies(result, peaks):
    v = np.array([p.xy for p in peaks]) - result.center
    return np.linalg.norm(v, axis=1) / result.px_per_unit, v


class FFTTests(unittest.TestCase):
    def test_acceptance_vector_spacings_and_angle(self):
        raw = fringes((256, 256), [((16 / 256, 0), 30), ((0, 32 / 256), 30)])
        result = compute_fft(raw, (0, 0, 256), 0.02)
        self.assertAlmostEqual(result.px_per_unit, 5.12)
        peaks = detect_peaks(result)
        g, v = peak_frequencies(result, peaks)
        for target in (3.125, 6.25):
            self.assertTrue(np.any(np.abs(g / target - 1) < .005), (target, g))
        a = v[np.argmin(np.abs(g - 3.125))]
        b = v[np.argmin(np.abs(g - 6.25))]
        angle = np.degrees(np.arccos(abs(a @ b) / np.linalg.norm(a) / np.linalg.norm(b)))
        self.assertAlmostEqual(angle, 90., delta=.5)
        self.assertAlmostEqual(1 / 3.125, .32)

    def test_peaks_are_friedel_pairs(self):
        raw = fringes((256, 256), [((16 / 256, 8 / 256), 30)])
        result = compute_fft(raw, (0, 0, 256), .02)
        peaks = detect_peaks(result)
        self.assertEqual(len(peaks), 2)
        np.testing.assert_allclose(peaks[0].xy + peaks[1].xy, 2 * result.center, atol=1e-6)

    def test_odd_roi_and_crop_keep_real_spacing(self):
        raw = fringes((400, 420), [((1 / 16, 0), 30)])
        for roi in ((0, 0, 255), (37, 61, 128), (100, 20, 301)):
            result = compute_fft(raw, roi, .02)
            g, _ = peak_frequencies(result, detect_peaks(result))
            self.assertTrue(np.all(np.abs(1 / g / .32 - 1) < .02), (roi, 1 / g))

    def test_diagonal_fringes_give_direction(self):
        k = np.array([12, 20]) / 256
        raw = fringes((256, 256), [((k[0], k[1]), 30)])
        result = compute_fft(raw, (0, 0, 256), .02)
        _, v = peak_frequencies(result, detect_peaks(result))
        expected = k * 256
        best = min(v, key=lambda x: np.linalg.norm(x - expected))
        np.testing.assert_allclose(best, expected, atol=.3)

    def test_no_scale_and_invalid_input(self):
        raw = fringes((128, 128), [((1 / 16, 0), 30)])
        self.assertIsNone(compute_fft(raw, (0, 0, 128), None).px_per_unit)
        for value in (0., -1., float("nan")):
            with self.assertRaises(ValueError):
                compute_fft(raw, (0, 0, 128), value)
        bad = raw.copy()
        bad[5, 5] = np.nan
        with self.assertRaises(ValueError):
            compute_fft(bad, (0, 0, 128), .02)
        for roi in ((0, 0, 16), (100, 0, 64), (-1, 0, 64)):
            with self.assertRaises(ValueError):
                compute_fft(raw, roi, .02)

    def test_axis_metadata_units_and_anisotropy(self):
        self.assertEqual(real_scale_from_axes([(.02, "nm"), (.02, "nm")]), (.02, False))
        self.assertEqual(real_scale_from_axes([(.2, "A"), (.2, "A")]), (.02, False))
        self.assertIsNone(real_scale_from_axes([(.1, "1/nm"), (.1, "1/nm")]))
        self.assertIsNone(real_scale_from_axes([]))
        value, anisotropic = real_scale_from_axes([(.02, "nm"), (.021, "nm")])
        self.assertTrue(anisotropic)

    def test_ifft_recovers_selected_fringe_set(self):
        a, b = (1 / 16, 0), (0, 1 / 10)
        raw = fringes((256, 256), [(a, 30), (b, 30)])
        result = compute_fft(raw, (0, 0, 256), .02)
        peaks = detect_peaks(result)
        g, _ = peak_frequencies(result, peaks)
        chosen = [p.xy for p, value in zip(peaks, g) if abs(value - 1 / .32) < .1]
        self.assertEqual(len(chosen), 2)
        out = inverse_fft(result, chosen[:1], radius=4, soft=2)
        only_a = fringes((256, 256), [(a, 30)])[32:-32, 32:-32]
        only_b = fringes((256, 256), [(b, 30)])[32:-32, 32:-32]
        core = out[32:-32, 32:-32]
        self.assertGreater(np.corrcoef(core.ravel(), only_a.ravel())[0, 1], .97)
        self.assertLess(abs(np.corrcoef(core.ravel(), only_b.ravel())[0, 1]), .05)
        with self.assertRaises(ValueError):
            inverse_fft(result, [], radius=4, soft=2)

    def test_zoomed_spectrum_keeps_reciprocal_scale(self):
        raw = fringes((256, 256), [((16 / 256, 0), 30), ((0, 32 / 256), 30)])
        for zoom, targets in ((2, (3.125, 6.25)), (4, (3.125,))):
            result = compute_fft(raw, (0, 0, 256), .02, zoom)
            self.assertAlmostEqual(result.px_per_unit, 5.12 * zoom)
            g, _ = peak_frequencies(result, detect_peaks(result))
            for target in targets:
                self.assertTrue(np.any(np.abs(g / target - 1) < .003), (zoom, target, g))
        with self.assertRaises(ValueError):
            compute_fft(raw, (0, 0, 256), .02, 3)

    def test_auto_zoom_enlarges_low_frequency_lattice(self):
        raw = fringes((512, 512), [((1 / 20, 0), 30), ((0, 1 / 20), 30)])
        self.assertEqual(auto_zoom(raw, (0, 0, 512)), 8)
        fine = fringes((256, 256), [((1 / 3, 0), 30)])
        self.assertEqual(auto_zoom(fine, (0, 0, 256)), 1)

    def test_zoomed_ifft_matches_unzoomed(self):
        a, b = (1 / 16, 0), (0, 1 / 10)
        raw = fringes((256, 256), [(a, 30), (b, 30)])
        plain = compute_fft(raw, (0, 0, 256), .02, 1)
        zoomed = compute_fft(raw, (0, 0, 256), .02, 4)
        point = [plain.center + np.array([16., 0.])]
        expected = inverse_fft(plain, point, 4, 2)
        actual = inverse_fft(zoomed, [zoomed.center + np.array([64., 0.])], 4, 2)
        np.testing.assert_allclose(actual, expected, atol=1e-6)


class HRTEMSessionTests(unittest.TestCase):
    def lattice_image(self, root, size=300, period=20):
        a, b = (1 / period, 0), (0, 1 / period)
        raw = fringes((size, size), [(a, 40), (b, 40), ((a[0], b[1]), 15), ((a[0], -b[1]), 15)])
        raw = (raw - raw.min()) / np.ptp(raw) * 255
        path = Path(root) / "hrtem.png"
        Image.fromarray(raw.astype(np.uint8)).save(path)
        return path

    def session(self, root):
        session = Session()
        session.phases = [Phase("cubic", (.4, .4, .4, 90, 90, 90), [Atom("Ni", np.zeros(3))])]
        session.open_hrtem(self.lattice_image(root))
        return session

    def test_unscaled_image_keeps_fft_but_blocks_indexing(self):
        with tempfile.TemporaryDirectory() as temp:
            session = self.session(temp)
            session.analyse()
            self.assertEqual(session.mode, "hrtem")
            self.assertIsNone(session.px_per_unit)
            self.assertTrue(session.spots)
            self.assertTrue(all(not s.candidates for s in session.sets))

    def test_scaled_fft_indexes_cubic_zone_with_fixed_center(self):
        with tempfile.TemporaryDirectory() as temp:
            session = self.session(temp)
            session.set_real_scale(.02, "人工输入")
            session.analyse()
            np.testing.assert_allclose(session.center, session.hrtem.fft.center)
            self.assertGreater(session.hrtem.fft.zoom, 1)
            self.assertAlmostEqual(session.px_per_unit, session.hrtem.fft.size * session.hrtem.fft.zoom * .02)
            self.assertTrue(session.sets)
            best = session.sets[0].candidates[0]
            self.assertEqual(sorted(abs(v) for v in best.zone), [0, 0, 1])
            self.assertLess(abs(best.scale_err), .02)
            self.assertEqual(best.cost_parts["missing"], 0.)
            with self.assertRaises(ValueError):
                session.set_center(10, 10)

    def test_roi_change_and_ifft_selection(self):
        with tempfile.TemporaryDirectory() as temp:
            session = self.session(temp)
            session.set_real_scale(.02, "人工输入")
            session.set_roi([10, 20, 200])
            self.assertEqual(session.hrtem.fft.size, 200)
            self.assertEqual(session.image.gray.shape, (200, 200))
            image = session.compute_ifft()
            self.assertEqual(image.shape, (200, 200))
            self.assertTrue(session.ifft_peaks())
            with self.assertRaises(ValueError):
                session.set_roi([150, 150, 200])

    def test_saed_open_resets_hrtem_state(self):
        with tempfile.TemporaryDirectory() as temp:
            session = self.session(temp)
            session.open(Path(temp) / "hrtem.png")
            self.assertEqual(session.mode, "saed")
            self.assertIsNone(session.hrtem)

    def test_project_round_trip_rebuilds_fft_from_source(self):
        from saedindex.project import load_project, save_project
        from saedindex.rendering import FigureStyle
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            session = self.session(temp)
            cif = root / "cubic.cif"
            cif.write_text("data_t\n_cell_length_a 4\n_cell_length_b 4\n_cell_length_c 4\n"
                           "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n"
                           "_symmetry_space_group_name_H-M 'P 1'\nloop_\n_atom_site_label\n"
                           "_atom_site_type_symbol\n_atom_site_fract_x\n_atom_site_fract_y\n"
                           "_atom_site_fract_z\nNi1 Ni 0 0 0\n", encoding="utf-8")
            session.phases = []
            session.add_phase(cif)
            session.set_real_scale(.02, "人工输入")
            session.set_roi([0, 0, 256])
            session.hrtem.ifft_points = [tuple(session.spots[0].xy)]
            session.hrtem.ifft_radius = 6.
            session.sets[0].confirmed = True
            save_project(root / "p.saed.json", session, FigureStyle())
            loaded, _ = load_project(root / "p.saed.json")
            self.assertEqual(loaded.mode, "hrtem")
            self.assertEqual(tuple(loaded.hrtem.roi), (0, 0, 256))
            self.assertEqual(loaded.hrtem.nm_per_px, .02)
            self.assertEqual(loaded.hrtem.ifft_radius, 6.)
            self.assertEqual(loaded.px_per_unit, session.px_per_unit)
            self.assertTrue(loaded.sets[0].confirmed)
            self.assertEqual(loaded.sets[0].candidate.cost_parts["missing"], 0.)
            np.testing.assert_allclose(loaded.image.raw, session.image.raw)
            self.assertEqual(loaded.compute_ifft().shape, (256, 256))


if __name__ == "__main__":
    unittest.main()
