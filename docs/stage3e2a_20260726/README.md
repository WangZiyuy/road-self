# Stage 3E-2A trajectory evidence capacity diagnosis

Mean, attention M=1, and latent M=4 use the same frozen E4, teacher-forced split, seed, fragment tokens, masks, and sample order. Mean is a strict parameter-free masked mean; no learned adapter is hidden in that baseline.

| setting | Branch AP | endpoint px | direction deg | ordinary AP | T-junction AP | multi-branch AP |
|---|---:|---:|---:|---:|---:|---:|
| Mean | 0.900643 | 4.985 | 6.121 | 0.987445 | 0.671664 | 0.810224 |
| Attention M1 | 0.913849 | 5.175 | 6.146 | 0.985625 | 0.711826 | 0.846087 |
| Latent M4 | 0.914417 | 5.117 | 5.968 | 0.985627 | 0.711192 | 0.851257 |

| setting | trainable params | token norm | token cosine | attention entropy |
|---|---:|---:|---:|---:|
| Mean | 0 | 9.622530 | n/a | 1.000000 |
| Attention M1 | 66688 | 11.544702 | n/a | 0.958971 |
| Latent M4 | 67072 | 11.547259 | 0.999235 | 0.967392 |

## GT branch-count groups

| setting | group | samples | Branch AP | endpoint px | direction deg | exact count |
|---|---|---:|---:|---:|---:|---:|
| Mean | count_0 | 124 | 0.000000 | n/a | n/a | 0.5403 |
| Mean | count_1 | 317 | 0.987445 | 4.051 | 4.371 | 0.9180 |
| Mean | count_2 | 39 | 0.671664 | 6.848 | 10.378 | 0.0513 |
| Mean | count_ge3 | 32 | 0.810224 | 6.487 | 8.318 | 0.0625 |
| Attention M1 | count_0 | 124 | 0.000000 | n/a | n/a | 0.5484 |
| Attention M1 | count_1 | 317 | 0.985625 | 4.267 | 4.726 | 0.9306 |
| Attention M1 | count_2 | 39 | 0.711826 | 6.930 | 8.985 | 0.0513 |
| Attention M1 | count_ge3 | 32 | 0.846087 | 6.696 | 8.449 | 0.0625 |
| Latent M4 | count_0 | 124 | 0.000000 | n/a | n/a | 0.5484 |
| Latent M4 | count_1 | 317 | 0.985627 | 4.176 | 4.423 | 0.9274 |
| Latent M4 | count_2 | 39 | 0.711192 | 6.967 | 8.970 | 0.0513 |
| Latent M4 | count_ge3 | 32 | 0.851257 | 6.658 | 8.531 | 0.0625 |

## Road categories

| setting | category | samples | Branch AP | endpoint px | direction deg |
|---|---|---:|---:|---:|---:|
| Mean | ordinary | 317 | 0.987445 | 4.051 | 4.371 |
| Mean | t_junction | 39 | 0.671664 | 6.848 | 10.378 |
| Mean | multi_branch | 32 | 0.810224 | 6.487 | 8.318 |
| Attention M1 | ordinary | 317 | 0.985625 | 4.267 | 4.726 |
| Attention M1 | t_junction | 39 | 0.711826 | 6.930 | 8.985 |
| Attention M1 | multi_branch | 32 | 0.846087 | 6.696 | 8.449 |
| Latent M4 | ordinary | 317 | 0.985627 | 4.176 | 4.423 |
| Latent M4 | t_junction | 39 | 0.711192 | 6.967 | 8.970 |
| Latent M4 | multi_branch | 32 | 0.851257 | 6.658 | 8.531 |

## Controlled-comparison checks

- Frozen inputs and split hashes identical: `True`.
- Attention M1/M4 shared initialization identical: `True`.
- Attention M1 - Mean Branch AP: `+0.013206`.
- Latent M4 - Mean Branch AP: `+0.013773`.
- Latent M4 - Attention M1 Branch AP: `+0.000567`.
- M4 collapsed: `True`.

## Conclusion

Learned attention exceeds strict masked mean; attention-based trajectory aggregation adds value.

No diversity/reliability loss, support replacement, anchor fusion, or Path.push change was introduced.

## Reproduction

```bash
python train_trajectory_evidence.py --config configs/stage3e2a_mean.yml --mode train
python train_trajectory_evidence.py --config configs/stage3e2a_attention_m1.yml --mode train
python train_trajectory_evidence.py --config configs/stage3e2a_latent_m4.yml --mode train
python scripts/summarize_stage3e2a.py --root data_self/stage3e2a --output-dir docs/stage3e2a_20260726
```
