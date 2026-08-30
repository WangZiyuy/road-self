# Stage S3C official VecRoad contract

The read-only source reference is the official [`tansor/VecRoad`](https://github.com/tansor/VecRoad) repository at revision `ffcb47e50e48ced717b2ac0e0f8c720ffc083441`. The audited files are [`train.py`](https://github.com/tansor/VecRoad/blob/master/train.py), [`configs/default.yml`](https://github.com/tansor/VecRoad/blob/master/configs/default.yml), and [`model/model.py`](https://github.com/tansor/VecRoad/blob/master/model/model.py).

## OFFICIAL_SOURCE_CONTRACT

Official RPNet uses a pretrained Res2Net-50 backbone, 256×256 windows, four recursive targets, `follow_target`, batch size 20, and `DataParallel=True` over the configured two GPUs. The training set contains 25 named cities. Adam uses betas `(0.9, 0.99)` and weight decay `2e-4`. Outer iterations 1–20 use LR `1e-3`; 21–50 use `1e-4`. There are 2048 optimizer updates per outer iteration and 50 outer iterations, for 102400 updates.

Road, junction, main anchor, and middle anchor losses all use `BCEWithLogits` with sum reduction. The two anchor losses are added, then total loss is `anchor + road + junction`. Gradients are value-clipped to `1e4`. The anchor fuse consumes `road_fts` and `junc_fts`. The official script has no validation early stopping.

## CURRENT_REPOSITORY_CONTRACT

The origin RPNet path retains all 648 official state keys and strictly loads the official release checkpoint. The current output name corresponding to official `anchor_middle` is `anchor_lowrs`. Raster mode adds exactly 18 keys under `segmentation_raster_fusion.`; sequence, Transformer, legacy DSF, and `fuse_module_traj` remain excluded from `raster_seg_only`.

## STAGE_S3C_ADAPTATION_CONTRACT

Stage S3C is not a full reproduction of official multi-city training. It is Xi’an small-region adaptation from the trained official checkpoint. It freezes the image backbone, recursive explorer, anchor heads, and original BatchNorm running statistics. Only road/junction segmentation heads and, for raster controls, the segmentation-only raster adapter are optimized. Anchor loss is never backpropagated. Micro-batch 10 with accumulation 2 preserves the official 20-sample sum-loss magnitude without division; LR is fixed at `1e-5` for adaptation.
