"""Native HRTEM FFT/IFFT workflow check on a synthetic calibrated lattice image."""
from pathlib import Path
import time

import numpy as np
from PIL import Image
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from saedindex.project import load_project, save_project
from saedindex.rendering import export_figure
from saedindex.window import MainWindow

APP = QApplication([])
out = Path(__file__).resolve().parent.parent / 'verification'
out.mkdir(exist_ok=True)
saed = Path(r'F:\E D\整理中-第四章稳定性及力学\00-热暴露\03-TEM\850-10000\1-0013.jpg')

# simple cubic a = 0.4 nm viewed along [001] at 0.02 nm/px: (100) fringes every 20 px
rng = np.random.default_rng(7)
yy, xx = np.mgrid[:1024, :1024].astype(float)
theta = np.radians(17)
u = xx * np.cos(theta) + yy * np.sin(theta)
v = -xx * np.sin(theta) + yy * np.cos(theta)
raw = np.cos(2 * np.pi * u / 20) + np.cos(2 * np.pi * v / 20) + .35 * np.cos(2 * np.pi * (u + v) / 20) \
    + .35 * np.cos(2 * np.pi * (u - v) / 20) + rng.normal(0, .6, xx.shape)
raw = (raw - raw.min()) / np.ptp(raw) * 230 + 10
image_path = out / 'fft-synthetic-hrtem.png'
Image.fromarray(raw.astype(np.uint8)).save(image_path)
cif = out / 'fft-cubic.cif'
cif.write_text("data_t\n_cell_length_a 4\n_cell_length_b 4\n_cell_length_c 4\n"
               "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n"
               "_symmetry_space_group_name_H-M 'P 1'\nloop_\n_atom_site_label\n"
               "_atom_site_type_symbol\n_atom_site_fract_x\n_atom_site_fract_y\n"
               "_atom_site_fract_z\nNi1 Ni 0 0 0\n", encoding='utf-8')

window = MainWindow(load_defaults=False)


def wait_for_worker():
    deadline = time.monotonic() + 90
    while window.worker is not None and time.monotonic() < deadline:
        APP.processEvents()
        QTest.qWait(20)
    assert window.worker is None, 'Analysis timed out'


try:
    window.show()
    window.session.add_phase(cif)
    window.open_image(image_path, kind='hrtem')
    wait_for_worker()
    s = window.session
    assert s.mode == 'hrtem' and not window.view_bar.isHidden() and not window.hrtem_box.isHidden()
    assert not window.mode_actions['center'].isVisible() and window.mode_actions['roi'].isVisible()
    assert s.px_per_unit is None and s.spots, 'unscaled image should still show FFT peaks'
    assert window.reference_unit.itemText(0) == 'nm'
    # real-space scale from two reviewed endpoints: 500 px = 10 nm
    window.set_view('source')
    assert window.canvas.view == 'source' and window.canvas.background.width() == 1024
    window.stage_scale([[100., 1000.], [600., 1000.]])
    window.reference_value.setValue(10.)
    window.apply_reference()
    assert abs(s.hrtem.nm_per_px - .02) < 1e-12, s.hrtem.nm_per_px
    window.apply_roi([200, 150, 512])
    s = window.session
    assert tuple(s.hrtem.roi) == (200, 150, 512) and s.sets
    best = s.sets[0].candidate
    assert sorted(abs(x) for x in best.zone) == [0, 0, 1], best.label
    assert abs(best.scale_err) < .01 and best.cost_parts['missing'] == 0.
    assert np.allclose(s.center, [256, 256])
    window.undo()
    assert tuple(window.session.hrtem.roi) != (200, 150, 512)
    window.redo()
    assert tuple(window.session.hrtem.roi) == (200, 150, 512)
    assert window.session.hrtem.fft.zoom == 4, window.session.hrtem.fft.zoom
    window.fft_zoom.setCurrentIndex(window.fft_zoom.findData(1))
    assert window.session.hrtem.fft.zoom == 1
    unzoomed = window.session.sets[0].candidate
    assert abs(unzoomed.scale_err) < .01 and sorted(abs(x) for x in unzoomed.zone) == [0, 0, 1]
    window.undo()
    assert window.session.hrtem.fft.zoom == 4 and window.fft_zoom.currentData() == 0
    window.ifft_use_group()
    selected = len(window.session.ifft_peaks())
    assert window.undo_action.toolTip().startswith('撤销：IFFT 使用')
    window.toggle_ifft_spot(next(iter(window.session.sets[0].lattice.members)))
    assert len(window.session.ifft_peaks()) < selected
    assert window.undo_action.toolTip().startswith('撤销：IFFT 选点')
    window.undo()
    assert len(window.session.ifft_peaks()) == selected
    assert window.session.hrtem.ifft_points is not None
    window.session.sets[0].confirmed = True
    window.refresh()
    for width, height in ((1480, 960), (1100, 800)):
        window.resize(width, height)
        for view in ('source', 'pattern', 'ifft', 'compare'):
            window.set_view(view)
            APP.processEvents()
            assert window.canvas.width() >= 360
            assert window.right_panel.horizontalScrollBar().maximum() == 0
            for tab in range(3):
                window.sidebar_tabs.setCurrentIndex(tab)
                APP.processEvents()
                assert window.setting_scrolls[tab].horizontalScrollBar().maximum() == 0, (width, tab)
            window.sidebar_tabs.setCurrentIndex(1 if view != 'source' else 0)
            APP.processEvents()
            window.grab().save(str(out / f'fft-{width}-{view}.png'))
    window.resize(1480, 960)
    ifft = window.session.hrtem.ifft
    assert ifft.shape == (512, 512) and np.isfinite(ifft).all()
    style = window.style
    export_figure(window.session, style, out / 'fft-figure.png')
    window.show_spots.setChecked(not style.show_spots)
    export_figure(window.session, style, out / 'fft-figure-check.png')
    assert (out / 'fft-figure.png').read_bytes() == (out / 'fft-figure-check.png').read_bytes()
    for view in ('ifft', 'compare', 'source'):
        export_figure(window.session, style, out / f'fft-export-{view}.png', view=view)
    assert Image.open(out / 'fft-export-compare.png').size == (512 * 2 + 8, 512)
    save_project(out / 'fft.saed.json', window.session, style)
    loaded, _ = load_project(out / 'fft.saed.json')
    assert loaded.mode == 'hrtem' and tuple(loaded.hrtem.roi) == (200, 150, 512)
    assert loaded.sets[0].confirmed and len(loaded.ifft_peaks()) == selected
    if saed.exists():
        window.open_image(saed)
        wait_for_worker()
        assert window.session.mode == 'saed' and window.view_bar.isHidden()
        assert window.mode_actions['center'].isVisible() and not window.mode_actions['roi'].isVisible()
        assert window.reference_unit.itemText(0) == '1/nm' and window.canvas.view == 'pattern'
    print(f'FFT OK: zone {best.label}, scale error {best.scale_err * 100:+.2f}%, '
          f'{len(window.session.spots) if saed.exists() else 0} SAED spots after switching back; '
          'ROI/IFFT undo, views, clean exports and project v2 round trip.')
finally:
    if window.worker:
        window.worker.wait()
        APP.processEvents()
    window.close()
    APP.processEvents()
