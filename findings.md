# Findings

## Recovered State
- Calibration, crystallography, spot fitting, indexing and session model already exist.
- Real images: F:\E D\整理中-第四章稳定性及力学\00-热暴露\03-TEM\850-10000.
- CIF library: D:\cif. Default comparison subset: Ni, NbNi3, delta NbNi3, NbC.
- PySide6 6.12, scipy, numpy and Pillow work in the project venv.
- Existing companion-DM calibration is 38.154 px/(1/nm); JPEG-only bar reading must
  be tested separately from this metadata path.
- Reference figure: unframed image with white parallelogram, hkl text, zone axis,
  overbar negatives; solid white scale line with bold Times text at lower left.

## Source Checks
- IUCr reciprocal lattice teaching pamphlet and Online Dictionary:
  zone law hu+kv+lw=0; d=1/|g| with reciprocal convention without 2*pi.
- Official Qt for Python Graphics View and thread signal documentation consulted.

## Issues Found During Recovery
- Session.simulation currently always fits scale, hiding calibration/CIF disagreement.
- load_image assigns reciprocal calibration even when OCR unit is real-space nm.
- vector_rows calls missing predicted nodes "d_meas"; UI/export must flag these.
- No GUI, figure export, test suite or launch entrypoint exists yet.
- CIF parsing/scattering is custom and approximate; no validated dynamical engine.

## Implemented And Verified
- Added PySide6 workbench, fixed-scale default simulation, font registration,
  project JSON round-trip, source hashing, CSV and PNG/TIFF exports.
- Qt offscreen platform has an empty font database on this machine. Explicitly
  registering installed Microsoft YaHei and Times New Roman resolves missing glyphs.
- Reject low-confidence OCR rather than silently indexing a wrong number.
- JPEG/DM dimension mismatch reproduced by resizing a real image to 512 square:
  old code incorrectly retained 38.154 instead of 19.077 px/(1/nm). Fixed by rejecting
  companion metadata when dimensions differ and trying the image bar.
- Both supplied reference attachments independently read "5 1/nm", about
  38.146 px/(1/nm), without companion metadata.
- Latest 155-image image-only audit: 150 accepted with correct lengths, 5 rejected
  or held for review, 0 accepted errors >3%. Mean accepted error 0.043327%, max 0.200768%.
- After saturated-centroid and centre fixes, 1-0010 remains ambiguous: sparse 8-spot
  set, Ni [521] (symmetry-equivalent to [215]) now first, very close to NbNi3.
  UI warns and leaves results unconfirmed.
- Latest 1-0013 delta [001] group has 89 spots, RMS 2.03 px, measured closure
  0.85 px (0.68%). This supersedes the earlier 0.13 px diagnostic.

## Editing Revision Findings
- Reproduced deleting a non-first SpotSet: list.remove invoked dataclass equality
  on numpy arrays and raised an ambiguous-truth ValueError. Removal now uses identity.
- Two-click scale tool discarded endpoint geometry; now retains and displays
  editable endpoints, with explicit scientific application and a reversible draft.
- Cosmetic masking previously rewrote calibration.label_rect, changing scientific
  exclusion. FigureStyle.cover_rect now stores figure-only masking independently.
- Scientific preview previously masked the original bar. Publication preview is
  now explicit; calibration evidence includes raw crop, OCR number/unit and length.
- Default pan intercepted all label drags; press/release handling now prioritizes
  text and commits final mouse positions for text, spots, endpoints, masks and pan.
- CIF markers were 1.4 source-image pixels wide and became faint at fit zoom.
  Their size is now screen-fixed with cosmetic pens and black contrast halos.
- Native compact checks exposed clipped endpoint fields and a right-panel scrollbar.
  Endpoint input rows now span the sidebar; right content can shrink within its scroll area.
- Export excludes all editing layers and unmatched-group draft outlines. Labels
  retain their dragged positions, font weight and line-width style after project restore.
- Review: the bar detector measures the core from x1 plus half the edge offset, while
  auto endpoints started at outer x1, shifting the drawn reference by a few pixels.
  Endpoints now use bar["start"]; the calibrated scale value itself was unaffected.
