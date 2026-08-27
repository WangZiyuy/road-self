# Stage S1 trajectory-raster data contract

## Contract result

`data_contract_ready_for_model = BLOCKED` and `GO / NO-GO = NO_GO`.

This contract is deliberately independent of model code, DSF, and the trajectory-sequence loader. It establishes file/grid/value requirements before any `raster_seg_only` implementation may begin.

## Manifest schema

`artifacts/stage_s1_pair_manifest.json` uses schema version `1.0.0`. Every region/split entry exposes both portable paths and measured facts, including:

- `region`, `split`, `aerial_path`, `raster_path`, and `source_identity`;
- flat SHA-256, shape, dtype, and unique-value fields plus the complete nested file facts;
- channel counts, value ranges, allowed values, and loader normalization;
- `crs`, `geotransform`, `pixel_size`, `pixel_origin`, coordinate-axis contract, and y direction;
- source lineage and value-semantics status;
- `shape_match`, `grid_registration_proven`, registration evidence, and entry status.

Paths and source labels are repository-relative or symbolic. The committed JSON contains no real absolute path, username, credential, image bytes, or raw trajectory rows.

## Fail-fast validation

`tools/seg_raster/validate_pair_manifest.py` checks:

1. schema and required fields;
2. missing aerial/raster members;
3. filename/region mismatch;
4. aerial/raster spatial-shape mismatch;
5. aerial/raster channel contracts;
6. raw uint8 dtype and range contracts;
7. observed trajectory values against the declared allowed values;
8. nested and flat SHA-256 values against source bytes;
9. source-backed value semantics;
10. disk/decoded/legacy-loader axis order, pixel origin, and y direction;
11. non-empty registration evidence independent of shape equality;
12. train/validation/test source-identity overlap and explicit non-leak proof.

The validator returns non-zero on violations by default. `--allow-blocked` exists only for audit capture: it preserves every blocking issue in JSON without turning a known blocked real-data audit into a shell failure.

## Axis and normalization contract

The proven loader layout is:

`disk image (width,height) → decoded array (height,width,channels) → swapaxes(0,1) → legacy internal (width,height,channels)`.

Window selection then indexes internal x followed by y. Both aerial and raster windows are converted to float32 and divided by 255 (`utils/model_utils.py:917-926`). The decoded image-array origin is upper-left and array y increases downward. The legacy raster's cross-source geographic origin, scale, and transform remain unproven, so the real manifest's pair-level `pixel_origin`, `crs`, `geotransform`, and `pixel_size` remain null.

## Pair inventory

| Profile / region | Split | Aerial | Trajectory raster | Shape/grid result | Status |
|---|---|---|---|---|---|
| Xi'an / `xian` | train | 4096×4096×3 present | 5000×4300×1 present | spatial shape mismatch; registration unproven | BLOCKED |
| legacy hard-coded `chicago` | validation | absent | absent | pair and source lineage absent | BLOCKED |
| default / `20` | test | present | absent | cannot check grid | BLOCKED |
| Xi'an / `xian` | test | 8192×8192×3 present | absent | cannot check grid | BLOCKED |

The default config also points `TILE_DIR` at `data_self/input/tile/`, while the audited Xi'an aerial files are under `data_self/input/imagery/`; the Xi'an-specific config points `TILE_DIR` at imagery. This configuration divergence is recorded rather than normalized by S1.

The validation loader hard-codes `['chicago']` (`utils/tileloader.py:210-233`), but no audited paired files or split lineage supports it. Therefore validation completeness and train/validation/test separation cannot be proven.

## Xi'an reconstruction gate

A real rebuild may run only after all of these are source-backed:

- the rasterization algorithm and its exact inputs;
- meanings and overwrite behavior for values 0/128/255;
- coordinate reference, source extent, pixel origin, x/y order, y direction, and target crop;
- identical code/parameters for train, validation, and test;
- explicit split provenance with no train/test leakage.

The current evidence proves a coordinate formula for trajectory points and an aerial crop chain, but it does not prove that the legacy PNG used that formula or specify how points became pixels. Consequently no `rebuild_aligned_raster.py` was created, no target data directory was written, and no real-data output checksum exists.

The synthetic reproducibility unit test writes two tiny identical PNG outputs and verifies byte identity, plus a differing output as a negative control. That validates the comparator only; it is not Xi'an reconstruction evidence.

## Tests

`tests/test_seg_raster_data_contract.py` covers the fourteen required categories with 18 tests: schema, spatial-shape rejection, missing pair, checksum mismatch, invalid value, unknown semantics, missing registration evidence, axis contract, deterministic byte comparison, split leakage, commit exclusions, no trajectory-sequence imports, no DSF/torch imports, and an empty production-file diff.

The tested production paths include the model/train/infer files and the pre-existing loader/config/test files in the explicit protection list. No production file is part of the S1 deliverable.

The required targeted command passed 18 tests. The first full-suite run inside the workspace sandbox reached `258 passed, 7 failed, 4 subtests passed`; all seven failures were access-denied errors from `tempfile.TemporaryDirectory`, not assertion failures. The unchanged full command was rerun with that sandbox restriction lifted and passed `265 tests, 4 subtests` with 9 warnings. Both executions and the seven original test identifiers are preserved in `artifacts/stage_s1_test_results.json`.

## Gate evaluation

| GO requirement | Result |
|---|---|
| Same Xi'an aerial/raster pixel grid | FAIL/BLOCKED |
| Reproducible source-backed registration | BLOCKED |
| Known 0/128/255 semantics | BLOCKED |
| Complete train/validation/test pairs | BLOCKED |
| All manifest checksums pass | BLOCKED by missing pair members |
| Explicit loader axis/crop contract | Loader axes known; cross-source grid still BLOCKED |
| Two real rebuilds yield identical SHA-256 | NOT_EXECUTED |
| No train/test leakage | UNPROVEN |

Because multiple mandatory conditions are unmet, model readiness cannot be `GO`. The only safe next action is to obtain authoritative raster-generation/value/registration and test-split sources, then rerun Stage S1. S1 does not authorize model, fusion, training, inference, anchor, loss, checkpoint, or sequence-branch changes.
