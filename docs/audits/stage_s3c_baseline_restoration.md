# Stage S3C baseline restoration

The selected baseline is the official release `vecroad.pth.tar`, SHA256 `498abc76e4ea461040b2b4ce69dc3d896cb975e93aae58815fe76443f6acf7c3` (130,206,305 bytes).

Its only wrapper field is `state_dict`. All 648 checkpoint tensors match the 648 official RPNet keys by name and shape and load with `strict=True`. There are no trajectory, Transformer, DSF, `fuse_module_traj`, or raster-adapter keys. The extracted release source tree matches official revision `ffcb47e50e48ced717b2ac0e0f8c720ffc083441`, and its official `prepare_data.sh` places this artifact at `data/ckpt/vecroad.pth.tar`.

The current image-only RPNet also loads all 648 keys strictly. The raster-capable model adds only 18 `segmentation_raster_fusion.*` parameters; the loader preserves those explicitly new parameters, merges every official shared tensor, and then performs a final strict load. The residual projection remains exactly zero-initialized.

Remote frozen validation and resource-capped closed-loop results are recorded separately in `artifacts/stage_s3c_baseline_evaluation.json` after the run-code freeze.
