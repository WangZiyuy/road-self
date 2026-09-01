# Stage S3E Mechanism Ablations

All comparisons use the zero-initialized Z2 path as the reference and the same 40,960-sample plan.

| Ablation | Result | Final aligned delta versus Z2 (F1 / IoU / AUPRC) |
|---|---|---:|
| C1 frozen road head | `NOT_NECESSARY` | -0.013357 / -0.008295 / -0.020462 |
| C2 no support multiplier | `NOT_PRIMARY` | -0.000471 / -0.000295 / +0.000001 |
| C3 frozen raster encoder | `SUPPORTED` | +0.005374 / +0.003373 / +0.003637 |
| C4 gradient balance | `INCONCLUSIVE_CALIBRATION_TARGET_INFEASIBLE` | not run |

C1 kept the road-head checksum unchanged, yet recall fell by 0.04145 and F1 by 0.02096 from its own sample-0 state. Thus road-head updates are not necessary for functional degradation, although Phase A independently proves road-head parameter drift is an additional damage path.

C2 was effectively identical to Z2, falsifying hard support multiplication as the primary cause. C3 reduced the null AUPRC head-drift gap by 21.64% and improved all three primary aligned metrics, so encoder learning is a supported causal contributor, but the remaining gap means freezing the encoder is only a partial mitigation.

C4 was not trained. With the lowest predeclared nonnegative weight (0.005), the residual negative/positive gradient-mass ratio remained 2.081747, outside the frozen [0.8, 1.2] gate. Reporting a C4 performance result would therefore violate the single-variable contract.
