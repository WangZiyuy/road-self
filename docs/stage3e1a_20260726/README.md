# Stage 3E-1A evidence-token necessity and diversity

M=1/4/8 use the same teacher-forced split, seed, frozen E4, fragment tokens, masks, sample order, optimizer settings and shared evidence-encoder initialization.

| setting | Branch AP | endpoint px | direction deg | ordinary AP | T-junction AP | multi-branch AP |
|---|---:|---:|---:|---:|---:|---:|
| M1 | 0.913849 | 5.175 | 6.146 | 0.985625 | 0.711826 | 0.846087 |
| M4 | 0.914417 | 5.117 | 5.968 | 0.985627 | 0.711192 | 0.851257 |
| M8 | 0.914600 | 5.156 | 5.955 | 0.985572 | 0.714702 | 0.846325 |

| setting | token cosine | attention cosine | attention entropy | top-8 Jaccard |
|---|---:|---:|---:|---:|
| M1 | n/a | n/a | 0.958971 | n/a |
| M4 | 0.999235 | 0.999906 | 0.967392 | 0.982407 |
| M8 | 0.999184 | 0.999918 | 0.967934 | 0.987686 |

## Controlled-comparison checks

- All controls identical: `True`.
- M4 - M1 Branch AP: `+0.000567`.
- M8 - M1 Branch AP: `+0.000751`.
- M8 - M4 Branch AP: `+0.000183`.
- M4 collapsed: `True`.
- M8 collapsed: `True`.

## Conclusion

M=1 and M=4 are AP-equivalent within tolerance; the current pathway behaves like global trajectory evidence.

No diversity, reliability, count, support, anchor, or Path.push loss/path was added in this diagnostic stage.

## Reproduction

```bash
python train_trajectory_evidence.py --config configs/stage3e1a_m1.yml --mode train
python train_trajectory_evidence.py --config configs/stage3e1a_m4.yml --mode train
python train_trajectory_evidence.py --config configs/stage3e1a_m8.yml --mode train
python scripts/summarize_stage3e1a.py --root data_self/stage3e1a --output-dir docs/stage3e1a_20260726
```
