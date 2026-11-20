# Progress

## Dark Workbench Update
- User selected a dark image-workbench theme. Added grouped settings and dedicated
  calibration, detection and display pages with an independent image-header ring toggle.
- Added optional min/max peak diameter and smoothing controls. Detection remains
  based on raw image data; size limits default to disabled. Chemical formulas use
  display-only subscripts, without modifying names stored in CIF/project data.
- Reanalysis now preserves calibration and publication masking and is undoable,
  including detection settings and the earlier confirmation state. Pending scale
  drafts block reanalysis. Project files retain optional new fields compatibly.
- Full suite: 97 tests passed. Native revision and reliability regressions passed.
  New native verification covers actual worker reanalysis, undo/redo, independent
  ring pixels, chemical labels, clean export and two window sizes.
- Source and previous package backed up before replacing the release. Build passed;
  updated dist/SAEDIndex/SAEDIndex.exe and dist/SAEDIndex-win64.zip. SHA256 confirmed
  the zipped executable matches the tested executable. Packaged startup loaded the
  example project, a full-window screenshot was inspected and errors.log is empty.
  The new packaged interactive window is left open. No clean-machine test performed.
  Details: verification/dark-workbench-validation.md.

## Reliability Upgrade Delivery
- User authorized the first delivery in the CrysTBox-inspired plan: P0/P1.
- Backed up source and prior portable zip before editing; fresh baseline 61 tests passed.
- Added 16 labelled synthetic patterns and an 8-image series-separated real manifest.
  Real group/phase truth remains unreviewed, so those images are comparison cases,
  not a measured identification-accuracy dataset.
- Reproduced seven sparse/noisy lattice splits; fixed using repeated translations,
  robust consensus refinement and candidate validation before acceptance.
- Added geometric diagnostics, score components, fixed-scale CIF residual overlays,
  metadata/bar conflict gating, explicit undoable CIF reference-scale application.
- Added regression coverage for six-point noise meshes, low-occupancy hypotheses,
  provenance/legacy project loading, clean export and missing angle references.
- Final verification: 84/84 unit tests passed. Native verify_reliability.py,
  verify_revision.py and verify_native.py passed, including clean export,
  annotation dragging, calibration history and project restoration.
- Fixed and stress synthetic comparisons: original 30/46 versus revised 46/46
  correct group counts. This is not phase-identification accuracy; one stress
  group has 84.375% membership recall. Real-image truth still needs human review.
- Rebuilt dist/SAEDIndex/SAEDIndex.exe and dist/SAEDIndex-win64.zip. The executable
  inside the zip has the same SHA256 as the tested executable.
- Packaged application loaded the portable smoke project with copied image/CIFs;
  inspected its full main-window screenshot (verification/packaged-reliability.png).
  Application error log remained empty. No isolated clean Windows machine tested.
- This delivery covers P0/P1 software work, not the remaining manual scientific
  validation. FFT, orientation relationships and tilt planning remain future work.

## 2026-10-29 Resume
- Read existing session, calibration, crystallography and pattern/indexing code.
- Recovered previous diagnostics; no previous work reverted.
- Read TDD, verification, planning and systematic-debugging skills.
- Current stage: test-first safety checks, rendering and desktop implementation.
- No new completion claims; previous calibration statistics require fresh verification.

## Implementation Checkpoint
- 21 tests now cover numeric scale safety, simulation, custom-index validation,
  measured vector closure, rendering, fonts, project state and GUI interactions.
- Fixed reproduced failures: automatic free-scale simulation, real-space label accepted
  as reciprocal, low-confidence OCR applied, rescaled JPEG/DM mismatch, predicted
  coordinates exported as measurements, disabled phase still indexed, compact canvas.
- Real-image verification generated pipeline.json, image-only-calibration.csv,
  calibration-summary.json and screenshots/exports in verification/.
- First screenshots had missing-font boxes; root cause was empty offscreen font db.
  Explicit application-font loading was added and glyph tests now pass.
- Remaining: asynchronous worker checks, final screenshot pass, README and native launch.

## Final Integration Corrections
- Native Windows smoke test exposed cross-volume relative-path failure. Added
  cross-drive regression; references now use absolute paths across Windows volumes.
- Reproduced saturated-disc centroid bias: a tiny window measures plateau noise,
  producing weak/mislocated "bright" spots. Saturated peaks now use a full 16px window.
- Added global Friedel midpoint search with cKDTree (bounded to 400 points), detector
  centre prior and beam-intensity evidence. Empty half-lattice centres must not win
  solely because of symmetric spot count. User reference and delta image centre
  regression tests added; transmitted-beam estimate remains explicitly heuristic.
- Local reference image now estimates centre near (492.6, 481.2) px.
- Native style emits libpng iCCP profile warnings; no rendering or operation failure.

## Delivery Verification
- Final `python -m unittest discover -s tests -v`: 28/28 pass.
- `tests/verify_native.py` on Windows: image-only scale 38.14654, 122 spots, 3 sets,
  project saved/restored; explicit centre assertions pass.
- `tests/verify_real.py`: pipeline, 1440px/1100px screenshots, PNG/CSV and project
  round-trip pass. Final image-only 155-image calibration summary unchanged:
  150 accepted correct, 5 held, 0 wrong accepted scales.
- `compileall` and `main.py --help` pass.
- Native desktop launched via pythonw on 1-0013.jpg; SAEDIndex window Responding=True,
  errors.log empty. Background GUI remains open intentionally for user testing.
- README, requirements and Start-SAEDIndex.cmd provided.

## User Feedback Revision
- Added tests before changes; reproduced group-array equality failure, missing
  endpoint retention, scientific/cosmetic exclusion coupling, masked scale preview,
  default-pan text drag failure, absent styling controls and faint simulations.
- Added explicit mouse group add/delete, checked scale evidence and source crop,
  editable endpoints/value/unit/length, restore-draft control and commit gating.
- Split figure-only bar replacement/cover from calibration. Added clean publication
  preview, adjustable font weight/size/line width and screen-fixed CIF markers.
- Applied native screenshot feedback to compact coordinate fields and panel sizing.
- Fresh verification: 48/48 unittest cases pass. Windows tests/verify_revision.py
  passes on 1-0013.jpg, including mouse group creation/deletion, scale staging/apply,
  g1/zone dragging, 1480x960 and 1100x800 layouts, PNG export and project round-trip.
- Screenshots: verification/revision-calibration-simulation.png,
  revision-annotation-controls.png, revision-compact.png, revision-publication-preview.png.
- Example output: verification/revision-publication-draft.png and
  verification/revision-example.saed.json. Results remain unconfirmed drafts.
- Existing native Qt libpng iCCP warnings remain non-fatal. No detector/indexing
  algorithm changes or new scientific-accuracy claims in this revision.

## Review pass
- Added 6 regression tests first, then fixed: cover rect leaking into the next image,
  plain clicks in move mode clearing confirmations, missing undo for group deletion,
  publication preview leaving hidden spot tools active, endpoint drags not updating
  the panel live, tiny mask drags never clearing, auto endpoints using the outer bar edge.
- Verification: 54/54 unittest cases, verify_revision.py and verify_native.py pass
  on Windows, compileall clean.

## Edit history and marker toggle
- Replaced group-only undo with snapshot undo/redo (toolbar, Ctrl+Z / Ctrl+Y,
  limit 60): spot add/delete/move, center, group add/delete, label drags,
  reference and pixel-scale application. Phases are shared, not copied.
- Undo with a pending scale draft discards the draft first. History clears on new
  image, re-analysis, project load and CIF changes.
- Toolbar eye toggle (H) hides spot rings, center, endpoints and simulation marks.
- Verification: 61/61 unittest cases, both native scripts pass, compileall clean.
  Screenshot: verification/history-markers-hidden.png.

## HRTEM FFT / IFFT (P2)
- New saedindex/fft.py (Qt-free): square ROI, mean removal, Hann window, fftshift,
  chirp-z zoom 1/2/4/8x with auto choice, Friedel peak detection with sub-pixel
  fit, soft Friedel mask and IFFT from the unwindowed complex spectrum.
- Session HRTEM mode: real-space scale from DM axes / nm bar / reviewed endpoints,
  fixed zero-frequency centre, geometry-only CIF ranking, undoable ROI, zoom,
  IFFT selection and mask; project version 2 (version 1 still loads).
- UI: toolbar HRTEM import, view switch (原图 / FFT / IFFT / 对照), FFT 区域与 IFFT
  panel, nm/A reference endpoints on the original image, nm bar for real-space exports.
- Verification: 112/112 unittest (offscreen), verify_fft.py plus the three native
  regression scripts pass. Details: verification/fft-validation.md.
- Package: dist/SAEDIndex-FFT/SAEDIndex.exe and dist/SAEDIndex-win64.zip, EXE
  SHA256 matches; packaged startup check clean (empty errors.log).

## White theme
- Palette, stylesheet, icon stroke and canvas surround switched to a white UI
  (teal accent kept). Spec filters PATH-sourced DLLs that broke QtCore import.
- 112/112 unittest, verify_dark_workbench.py and verify_fft.py pass; rebuilt
  package starts cleanly.

## App icon
- tools/make_icon.py renders assets/app.ico (16-256 px, simplified lattice at
  <= 32 px) and assets/app.png: indexed SAED spots with g1/g2 parallelogram.
- Used for the window/taskbar (main.py, own AppUserModelID) and the EXE (spec).
