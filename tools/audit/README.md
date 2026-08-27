# Seg-Raster Stage S0 audit tools

These tools inspect the legacy DSF/raster integration without modifying the
production model, training, or inference paths.

Final bookkeeping can be refreshed without repeating runtime, gradient, or
checkpoint probes:

```text
python tools/audit/audit_stage_s0_static.py --inventory-only --source-repo <dirty-worktree>
python tools/audit/sanitize_stage_s0_artifacts.py --dirty-repo <dirty-worktree>
python tools/audit/build_stage_s0_commit_manifest.py
```

The commit manifest deliberately excludes itself because a file cannot embed a
stable SHA-256 of its own bytes. It explicitly validates that no checkpoint,
dataset, model-weight, cache, production-model, train, or infer file is listed.
