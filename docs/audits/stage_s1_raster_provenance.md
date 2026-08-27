# Stage S1 trajectory-raster provenance audit

## Decision

Stage S1 is **BLOCKED**. The aerial preparation chain and the raw-trajectory-to-pixel point chain are reproducible, but no checked source or history establishes how `data_self/input/traj/xian_0_0.png` was rendered, registered, or assigned values 0/128/255. Equal raw dimensions do not bridge that missing provenance.

- S1 base: `23285e5bc6515ca88a3121d2547aa9ab0476a7ad`
- Branch: `feat/seg-raster-only`
- Source boundary: `CURRENT_DIRTY_SNAPSHOT_READ_ONLY`
- Model implementation: not started
- Real raster rebuild: not executed

The source-provenance gate was declared passed before S1. The dirty snapshot was used only for reads; no checkout, stash, reset, clean, commit, or file write was performed there.

## Audited flows

### Xi'an aerial chain: PASS

`scripts/prepare_xian_image.py:35-42` rejects inputs larger than the configured canvas, creates a zero-filled RGB canvas, and pastes the source image at `(0,0)`. Lines 45-49 then crop tiles at explicit multiples of `tile_size`; lines 55-71 persist the bbox and size metadata.

The source-backed chain is:

`${RAW_AERIAL_SOURCE}` (4300×5000 RGB) → upper-left paste on 8192×8192 zero canvas → `data_self/input/imagery_8192/xian.png` → upper-left 4096×4096 crop `data_self/input/imagery/xian_0_0.png`.

Byte-derived pixel checks prove all four statements:

- the 4096 tile equals the raw image's upper-left 4096×4096 pixels;
- the full canvas's 4300×5000 source extent equals the raw image;
- right padding is all zero;
- bottom padding is all zero.

Therefore 4300 and 5000 are the raw aerial width and height, while the configured train aerial is the upper-left 4096 tile. This explains the aerial size transition without resize, center crop, transpose, flip, or inferred offset.

### Raw trajectory point chain: PASS for points only

The read-only trajectory-piece manifest matches `${RAW_TRAJECTORY_SOURCE}` by SHA-256 and records 14,267 trajectories and 305,484 points. `scripts/prepare_xian_traj_piece.py:141-149` gives the explicit GCJ-02 bbox-to-pixel formula:

`x = (lon - lon_min) * width / (lon_max - lon_min)`

`y = height - (lat - lat_min) * height / (lat_max - lat_min)`

It records the input SHA, original size, and observed pixel bounds at lines 222-243. `utils/gis_to_graph.py:292-299` and `utils/gis_to_graph.py:168-170` use the same scale and y inversion. These scripts generate per-trajectory/pixel-coordinate intermediates and graph inputs; they do not write the audited legacy trajectory PNG.

### Legacy Xi'an trajectory raster: BLOCKED

The observed file is single-channel uint8 with shape 5000×4300×1 and exactly these values:

| Raw value | Pixel count | Source-backed meaning |
|---:|---:|---|
| 0 | 19,937,247 | UNKNOWN |
| 128 | 385,526 | UNKNOWN |
| 255 | 1,177,227 | UNKNOWN |

Its SHA-256 is recorded in `artifacts/stage_s1_provenance_trace.json`. It has no embedded CRS, geotransform, pixel size, or other PNG metadata. The audit found no checked generator, original command, intermediate chain, crop extent, affine transform, drawing parameters, anti-aliasing rule, overwrite order, or code line that assigns 0/128/255.

Three external raster candidates were inspected read-only. Their dimensions are 14063×16525, 14063×16525, and 5625×6610; none matches the legacy file's size or SHA, and none contains embedded spatial metadata. They are retained as separate candidates, not silently selected.

`data_self/gen_dataset.py` is a separate hard-coded legacy dataset slicing flow. It reads trajectory-like grayscale inputs but does not establish that it created this Xi'an PNG or explain its values. `scripts/prepare_xian_traj.py`, `utils/OSMDataset.py`, and the GIS graph conversion path operate on trajectory points/NPZ/graph structures rather than writing this raster.

## Registration answers

| Question | Evidence-backed answer |
|---|---|
| Why is the aerial 4096×4096? | It is the proven upper-left tile of the 8192 canvas. |
| Why is the raster 4300×5000? | It matches the raw aerial dimensions, but the raster's producing extent/recipe is unknown. |
| Do raster and aerial share the same upper-left? | UNKNOWN. Image decoding uses an upper-left array origin, but cross-source geographic origin is unproven. |
| Are x/y swapped? | The loader explicitly swaps decoded H/W to internal W/H (`utils/tileloader.py:20-30`); the producer's raster axes are unknown. |
| Is there flip, offset, crop, or scale? | Aerial crop is proven; raster transform is UNKNOWN. |
| Does equal 4300×5000 size prove registration to the raw aerial? | No. Shape equality is only `SHAPE_MATCH`, never registration proof. |
| May the legacy raster be cropped upper-left to 4096? | No. That is only a candidate operation until raster origin/extent is source-backed. |

No resize, visual alignment, GT-driven offset/flip selection, or inferred crop was adopted.

## Loader and pairing evidence

`utils/tileloader.py:145-160` uses the same requested rectangle for aerial and trajectory loads but does not validate whole-image shapes or source registration. `utils/model_utils.py:917-926` applies the same window indices and divides both inputs by 255. `infer.py:573-585` reads each configured test aerial and, for the trajectory path, requires the corresponding file under `TEST_TRAJ_DIR`.

The configured `data_self/input/traj_test/` directory is absent in the audited snapshot:

- default profile configures region `20`; its aerial exists, but `traj_test/20.png` does not;
- Xi'an profile configures region `xian`; its 8192 aerial exists, but `traj_test/xian.png` does not;
- `653.png` is present in the default test aerial directory but is not listed by the default test-region file;
- no source-backed test-raster generator or train/test non-leak proof was found.

Status: `BLOCKED_TEST_RASTER_SOURCE_MISSING`. A train raster was not copied into the test split.

## Value-semantics boundary

The only proven runtime transformation is `float32(value)/255.0`, so the observed values become 0, approximately 0.5019607843, and 1. This mathematical normalization does not reveal whether the levels mean points, lines, density, coverage, unknown area, overwrite priority, or something else.

There is insufficient evidence to preserve a formal ternary semantic contract, binarize, threshold, or convert to two channels. Status: `BLOCKED_VALUE_SEMANTICS_UNKNOWN`.

## Reproduction and unknowns

The aerial and point-coordinate chains can be replayed from labeled source inputs and recorded checksums. The legacy raster cannot be deterministically reproduced because these required facts remain unknown:

- authoritative raster generator and exact command;
- raw/intermediate inputs used by that generator;
- 0/128/255 assignment and overwrite rules;
- line/point width, interpolation, morphology, or anti-aliasing parameters;
- raster CRS/extent/geotransform and its relationship to the aerial grid;
- validation/test source lineage and split-separation proof.

Accordingly, `artifacts/stage_s1_rebuild_manifest.json` and `artifacts/stage_s1_reproducibility_check.json` exist with `status=NOT_EXECUTED`; no corrected data was created.
