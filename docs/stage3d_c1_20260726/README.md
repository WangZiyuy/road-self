# Stage 3D-C1 support-guided trajectory fusion

All results use the same E4 branch decoder, RPNet checkpoint, teacher-forced split, seed, and bounded 64-fragment inputs. The branch head remains auxiliary and never feeds `Path.push`.

| variant | branch AP | no-traj AP | full-no-traj | endpoint px | direction deg | exact count | oracle distinct coverage | oracle duplicate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| original_attention | 0.900514 | 0.895513 | +0.005001 | 5.019 | 6.088 | 0.7051 | 0.8522 | 0.0000 |
| support_aggregation | 0.907397 | 0.895499 | +0.011897 | 4.407 | 5.351 | 0.7129 | 0.8583 | 0.0000 |
| support_topk_8 | 0.907702 | 0.895499 | +0.012203 | 3.922 | 4.978 | 0.7129 | 0.8623 | 0.0000 |
| support_topk_16 | 0.906697 | 0.895499 | +0.011198 | 4.389 | 5.162 | 0.7148 | 0.8623 | 0.0069 |
| random_fragment_aggregation | 0.905318 | 0.895499 | +0.009819 | 4.654 | 5.229 | 0.7148 | 0.8563 | 0.0000 |
| c1_b_topk8 | 0.908223 | 0.895513 | +0.012709 | 3.835 | 5.078 | 0.7148 | 0.8623 | 0.0000 |

## Results by node type

| variant | ordinary AP | T-junction AP | multi-branch AP |
|---|---:|---:|---:|
| original_attention | 0.987483 | 0.670718 | 0.809277 |
| support_aggregation | 0.986933 | 0.688886 | 0.827637 |
| support_topk_8 | 0.987020 | 0.684510 | 0.797289 |
| support_topk_16 | 0.986761 | 0.665159 | 0.792957 |
| random_fragment_aggregation | 0.984519 | 0.691979 | 0.835412 |
| c1_b_topk8 | 0.986813 | 0.672928 | 0.800598 |

## Support selection

| variant | support AP | Precision@8 | Recall@8 | nDCG@8 | top-8 query Jaccard |
|---|---:|---:|---:|---:|---:|
| support_aggregation | 0.776658 | 0.632135 | 0.304048 | 0.580267 | 0.293243 |
| support_topk_8 | 0.769491 | 0.633721 | 0.304242 | 0.585599 | 0.288167 |
| support_topk_16 | 0.772651 | 0.633985 | 0.306365 | 0.586911 | 0.299691 |
| random_fragment_aggregation | 0.771835 | 0.634514 | 0.303766 | 0.589502 | 0.281221 |
| c1_b_topk8 | 0.788302 | 0.670983 | 0.318795 | 0.646790 | 0.295405 |

## C1-b decision

- Best support variant: `support_topk_8`.
- Branch AP gain over original: `+0.007188`.
- Full minus no-trajectory AP: `+0.012203`.
- C1-a gate passed: `True`.
- Run C1-b: `True`.
- C1-b completed: `True`.
- C1-b AP change over best C1-a: `+0.000520`.
- Reason: C1-a improved validation branch AP and retained the configured full-vs-no-trajectory gain.

## Reproduction commands

```bash
python train_support_fusion.py --config configs/stage3d_c1_a_original.yml --mode evaluate
python train_support_fusion.py --config configs/stage3d_c1_b_support.yml --mode train
python train_support_fusion.py --config configs/stage3d_c1_c_topk8.yml --mode train
python train_support_fusion.py --config configs/stage3d_c1_d_topk16.yml --mode train
python train_support_fusion.py --config configs/stage3d_c1_e_random.yml --mode train
python train_support_fusion.py --config configs/stage3d_c1_b_finetune_topk8.yml --mode train
python scripts/summarize_stage3d_c1.py --root data_self/stage3d_c1 --output-dir docs/stage3d_c1_20260726
```

C1-b is launched from the selected C1-a best checkpoint only when the machine-readable gate above reports `run_c1_b=true`.

C1-a caches frozen RPNet, graph-state, trajectory-encoder and pre-trajectory decoder tensors once. A regression test verifies that this cache produces the same support-fusion outputs as the dynamic frozen-model path. C1-b keeps the trajectory encoder dynamic and trainable.

The best C1-a gain over original attention is larger than zero, but its margin over the random-fragment control is only `+0.002384`. C1-b adds only `+0.000520` AP over C1-a. These results support the implementation gate, but are not yet strong evidence of seed-stable generalization.

Support ranking metrics (Precision/Recall/nDCG at K and top-8 Jaccard) and ordinary/T-junction/multi-branch metrics are retained in `comparison.json`.
