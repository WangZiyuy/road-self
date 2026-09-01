# Stage S3E Final Report

Stage S3E closes the observed failure as a distributed learned-path problem rather than a lack of raster information. Phase A proves road-head drift begins in the `0_TO_2560` sample interval. At the final checkpoint, the drift alone costs F1 0.03494, IoU 0.02218, and AUPRC 0.03057. The trained adapter on the clean head is also harmful at final time, while the factorial interaction term is positive rather than destructively negative.

Projection zero initialization restores exact sample-0 and first-step parity but does not repair final degradation. C1 then shows that a trained raster adapter can still cause recall/F1 degradation with a bitwise frozen road head. Road-head co-adaptation is therefore falsified as a necessary condition, although road-head parameter drift remains a proven additional damage path.

C2 shows the support multiplier is not primary. C3 provides the only positive mechanism result: freezing the raster encoder improves aligned F1 by 0.00537, IoU by 0.00337, AUPRC by 0.00364, and reduces the null AUPRC drift gap by 21.64%. This supports raster-encoder learning as a causal contributor, but not as the sole cause. C4 is inconclusive because a global nonnegative background weight cannot reach the predeclared residual-gradient balance target.

Root-cause classification:

- random initial residual: `AMPLIFIER`;
- trained raster-adapter forward path: `PROVEN_CAUSAL`;
- road-head parameter drift: `PROVEN_CAUSAL`;
- road-head co-adaptation as a necessary condition: `FALSIFIED`;
- support multiplication: `FALSIFIED`;
- raster-encoder learning: `SUPPORTED_CAUSAL`;
- background-gradient dominance: `INCONCLUSIVE`.

No complete fix is justified. The best tested partial mitigation is projection zero-init plus a frozen raster encoder; it still requires a dedicated next-stage validation. The only allowed next experiment is a single-variable encoder learning-rate/freeze validation. Anchor, graph, multiseed, combined fixes, density raster, trajectory sequence, and new fusion architecture remain outside the Stage S3E boundary.
