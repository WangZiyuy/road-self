# Stage S3D Controlled Comparison

Formal run code SHA: `68566e6319d59bf9726beda656c6abbe372b5be4`. All five runs used the N0-selected common sample point of `40960` samples.

| Run | Control | Road F1 | Road IoU | Road AUPRC | Junction F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| N0 | image-only | 0.242529 | 0.137999 | 0.263079 | 0.168573 |
| N1 | aligned raster | 0.207447 | 0.115727 | 0.252732 | 0.168573 |
| N2 | zero raster | 0.242529 | 0.137999 | 0.263079 | 0.168573 |
| N3 | shift +512/+512 | 0.233420 | 0.132131 | 0.256317 | 0.168573 |
| N4 | deterministic permutation | 0.209112 | 0.116764 | 0.250258 | 0.168573 |

The strict null-parity gate is `FAIL`: N0 and N2 retained identical shared trainable tensors, road/junction predictions, road features, and reported metrics at every checkpoint, but their raw gradient SHA256 values diverged beginning at 7680 samples. The protocol requires gradient checksum equality, so this was not relaxed or reinterpreted.

The spatial-causal gate is `FAIL`. N1 improved none of road F1, road IoU, or road AUPRC versus N0, N2, N3, or N4. The paired mean composite delta versus N0 was `-0.0276049`, with a descriptive bootstrap interval of `[-0.0409259, -0.0143337]`; the last three checkpoint directions were all unfavorable. This single-seed screen is not a significance claim.

Junction parity passed. Because the segmentation gate failed, frozen-anchor and resource-capped graph evaluation were `NOT_EXECUTED_BY_GATE`.

Final raster branch decision: `STOP_AFTER_NEGATIVE_CONTROL_RESULT`.

All formal CUDA evidence was produced on REMOTE_TRAINING_SERVER.
