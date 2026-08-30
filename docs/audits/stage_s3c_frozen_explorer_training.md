# Stage S3C frozen-explorer training contract

The original image feature path (`stage_1`–`stage_5`, side projections, and `conv_fuse`) and the full anchor/explorer path (`fuse_module`, recursive feedback, decoders, and final anchor heads) are frozen. Original BatchNorm modules remain in evaluation mode and their running-buffer checksum must not change. Road/junction head BatchNorm affine parameters remain trainable while their running statistics stay frozen. The raster adapter uses GroupNorm, which has no running statistics.

Each run uses one physical remote GPU, micro-batch 10, accumulation 2, and an effective 20 samples per optimizer update. Because losses use sum reduction, the two micro-batch losses are accumulated directly without dividing by two. Adam uses LR `1e-5`, betas `(0.9, 0.99)`, and weight decay `2e-4`.

Xi'an supplies two eligible 2048x2048 subtiles in each frozen spatial half. Five independent teacher-forced `Path` replicas per subtile recover the official per-replica local batch of 10 without sampling the same mutable explorer twice in one micro-batch. This preserves the spatial split while making `micro_batch=10, accumulation=2` executable.

Training backward is restricted to road and junction segmentation losses. The model's `segmentation_only=True` forward returns native 64×64 logits and skips the frozen recursive anchor computation. Anchor evaluation is forward-only and conditional on the preregistered segmentation gate.

The budget is 40,960 samples with immutable checkpoints at 0, 2,560, 5,120, 7,680, 10,240, 12,800, 15,360, 17,920, 20,480, 25,600, 30,720, 35,840, and 40,960 samples.
