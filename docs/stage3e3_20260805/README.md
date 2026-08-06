# Stage 3E-3 single-token trajectory evidence validation

## Scope and code facts

This report covers teacher-forced validation of the auxiliary branch head. It does not report complete VecRoad graph metrics. RPNet, the fragment and graph encoders, E4 branch decoder/output heads, support head, anchor path, and `Path.push` remain frozen or untouched. Only the M=1 `TrajectoryEvidenceEncoder` was trained.

## Experimental results

- Seeds: 20260724, 20260725, 20260726.
- Mean full minus image_graph Branch AP: +0.017796 (std 0.002026).
- Mean wrong-sample minus full Branch AP: -0.020718.
- Preflight M1 reproduction passed: True.
- Exact no-trajectory equivalence passed: True.
- Acceptance gate passed: True.

| Mode | Branch AP mean | std |
| --- | ---: | ---: |
| image_graph | 0.895513 | 0.000000 |
| original_fragment | 0.900514 | 0.000000 |
| full_trajectory | 0.913309 | 0.002026 |
| no_trajectory | 0.895513 | 0.000000 |
| retain_75 | 0.913755 | 0.001234 |
| retain_50 | 0.912222 | 0.001263 |
| retain_25 | 0.912466 | 0.001153 |
| wrong_sample_trajectory | 0.892591 | 0.002570 |

Category-wise full minus image_graph Branch AP: ordinary -0.001176, T-junction +0.087159, multi-branch +0.073214.

Thinning minus full Branch AP: retain-75 +0.000446, retain-50 -0.001087, retain-25 -0.000843.

For full trajectory evidence, normalized attention entropy was 0.959074, effective fragment count 49.988, top-1 mass 0.068878, and top-8 mass 0.315936 (three-seed means).

| Seed | Best epoch | Full AP | Full-image delta | Checkpoint SHA256 |
| --- | ---: | ---: | ---: | --- |
| 20260724 | 30 | 0.913849 | +0.018336 | `d7f00a12a10c8e80687945f0f68596883bdd4b038893d828bba297601b451ff6` |
| 20260725 | 28 | 0.910601 | +0.015088 | `a552f0c89949f9c1acbb71ab3d661660a9a615881b60267e06e29f74835f5f68` |
| 20260726 | 30 | 0.915475 | +0.019962 | `ae4a8157885a89dcbfe596123dc974b2ac00d27aae7923229b227be9827b3dae` |

See `comparison.json`, `per_seed_results.json`, and `robustness_results.json` for complete numerical results.

## Interpretive inference

The single-token trajectory pathway passed the predefined teacher-forced stability gate. This is not evidence yet of an improvement in closed-loop road-graph extraction.
Wrong-sample trajectories reduced Branch AP by 0.020718 versus full evidence on average and fell below image_graph, so this control does not support a pure trajectory-presence shortcut.
Retaining only 25% of fragments changed Branch AP by less than 0.001 on average. The learned pooling appears redundant or driven by a relatively small useful subset; this is a robustness observation, not proof that all discarded fragments are useless.
The gain is concentrated at junctions: ordinary-node AP changed slightly negatively, while T-junction and multi-branch gains were positive for every seed.

## Reproduction

```bash
python train_trajectory_evidence.py --config configs/stage3e3_seed20260724.yml --device cuda
python scripts/evaluate_stage3e3_robustness.py --config configs/stage3e3_seed20260724.yml --checkpoint data_self/stage3e3/seed20260724/checkpoints/stage3e0.best.pth.tar --device cuda
python scripts/summarize_stage3e3.py --root data_self/stage3e3
```
