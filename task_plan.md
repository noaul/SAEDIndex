# SAEDIndex Desktop

## Goal
Import diffraction images, automatically read reciprocal scale calibration,
detect/index spots against CIF, compare simulations and export annotated figures.

## Phases
- [x] Recover existing geometry/calibration/pattern code and real-image diagnostics.
- [x] Establish desktop architecture and testable rendering/model boundaries.
- [x] Add calibration safety checks and fixed-scale simulation regression tests.
- [x] Build image canvas, Chinese workbench, candidate comparison and manual correction.
- [x] Add publication-style overbar labels, scale replacement and CSV/PNG/TIFF export.
- [x] Verify real images, image-only scale reading, desktop interactions and launch.

## Delivery
- 28 automated tests pass; native Windows import/save/restore smoke test passes.
- Real-image pipeline and PNG/CSV/project exports regenerated after centroid fixes.
- Desktop launched on 1-0013.jpg; window is responding, error log empty.
- Delivered as a runnable local desktop workbench, not a validated fully autonomous
  phase-identification instrument. Confirmation and limitations remain explicit.

## Decisions
- PySide6 desktop; numpy/scipy/Pillow; use installed dependencies.
- Calibration is authoritative: do not silently refit it to a CIF.
- Automatic candidates are provisional until a user confirms them.
- Numerical comparisons must distinguish detected spots from predicted nodes.
- Approximate kinematic intensity is not quantitative dynamical diffraction.
- Keep source images untouched; scale replacement only affects exported/rendered copies.

## Known Risks
- Saturated transmitted beam, weak streaks, overlapping phases, and symmetry ambiguity.
- Prior diagnostics recognize delta [001] but some images still produce uncertain matches.
- Existing 153/155 calibration statistic was measured on the development dataset,
  not an independent accuracy guarantee.
- Fresh image-only safety-gated audit: 150/155 accepted, 5 require checking,
  no wrong accepted scales in this dataset. Mean accepted error 0.0433%.

## Editing Revision (2026-10-29)
- [x] Reproduce group deletion, toolbar actions and label dragging with Qt tests.
- [x] Separate calibration evidence and editable reference endpoints from figure cosmetics.
- [x] Add explicit mouse group controls, annotation styling and visible CIF overlays.
- [x] Verify clean export, project round-trip and native Windows interactions.

Revision verification: 48 tests pass; native real-image mouse workflow,
1480/1100px layouts, clean export and project restore pass. Automatic grouping
remains provisional; no new autonomous identification accuracy claim.
