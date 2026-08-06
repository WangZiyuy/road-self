# road_self Stage 3F-A (execution pending)

Stage 3F-A adds an isolated, zero-initialized trajectory-evidence residual to
the two original VecRoad anchor pre-head features. It does not alter road or
junction prediction, recursive graph exploration, `Path.push`, anchor
thresholds, NMS, or `map_to_coordinate`.

## Commands

```bash
python scripts/build_stage3fa_anchor_cache.py \
  --config configs/stage3fa_seed20260724.yml \
  --device cuda --overwrite

python train_stage3fa_anchor_fusion.py \
  --config configs/stage3fa_seed20260724.yml --device cuda
python scripts/evaluate_stage3fa_anchor_fusion.py \
  --config configs/stage3fa_seed20260724.yml --device cuda

python scripts/summarize_stage3fa.py \
  --root data_self/stage3fa \
  --output-dir docs/stage3fa_20260805
```

Repeat the train/evaluate commands for seeds 20260725 and 20260726. The cache
is shared because sample IDs, frozen inputs, trajectory evidence and targets
are immutable across the three fusion-initialization seeds.

Results have not yet been inserted into this file. The summarizer will replace
this execution-pending report only after all three evaluations exist.

This remains teacher-forced anchor validation. It is not a complete VecRoad
closed-loop experiment and does not claim APLS, TOPO, or complete road-network
improvements.
