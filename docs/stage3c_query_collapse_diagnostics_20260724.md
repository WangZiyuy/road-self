# road_self Stage 3C existence / query-collapse diagnosis

## Scope

This work changes only the Stage 3C auxiliary branch head, its loss,
evaluation, experiment configuration, and tests. RPNet, the VecRoad anchor
head, `NUM_TARGETS`, `map_to_coordinate`, and the graph-growth state machine
are unchanged. Auxiliary branch predictions are not passed to `Path.push`.

The evaluated auxiliary checkpoint is:

`/home/wangziyu/VecRoad_self/data_self/stage3c_branch_aux/checkpoints/stage3c_aux.best.pth.tar`

It reports epoch 16. The strictly loaded, frozen RPNet checkpoint is:

`/home/wangziyu/VecRoad-master/data/ckpt/vecroad.pth.tar`

## Pre-change call-chain audit

- `model/branch_query_decoder.py`
  - six learned queries are initialized with standard deviation 0.02;
  - one shared graph-state token is added to every learned query and the sum
    is passed through `LayerNorm`;
  - the resulting queries independently read image/walked-path tokens and
    trajectory fragment tokens through cross-attention;
  - existence is a linear head; endpoint is an MLP followed by `tanh`;
  - direction is the normalized predicted endpoint offset with epsilon
    `1e-6`.
- `model/branch_set_loss.py`
  - the old Hungarian cost is endpoint L1 plus direction cosine distance;
  - matching does not use existence;
  - matched query slots become positive existence targets;
  - all other slots, including all slots at an empty-GT node, are negative;
  - only matched slots enter endpoint/direction losses.
- `utils/branch_metrics.py`
  - the historical duplicate metric only examines queries passing the chosen
    existence threshold. It cannot by itself distinguish activated unmatched
    slots from internal multi-query collapse.
- `train_branch_aux.py` and `utils/stage3c_checkpoint.py`
  - RPNet loads with `strict=True`, is put in eval mode, and all of its
    parameters have `requires_grad=False`;
  - the three auxiliary modules load with `strict=True`;
  - no auxiliary branch output is connected to graph growth.
- `scripts/prepare_stage3c_branch_dataset.py`
  - its `Path.push` is teacher-forced dataset-state collection, not predicted
    branch insertion.

## E0 results

The complete machine-readable output is in
`docs/stage3c_e0_diagnostics_20260724/`.

Validation has 512 states, 3,072 query slots, and 494 GT branches.

### Existence separation

| metric | value |
|---|---:|
| slot AP | 0.2333 |
| slot AUROC | 0.6613 |
| ECE (15 bins) | 0.0393 |
| matched probability mean / median | 0.2311 / 0.1469 |
| unmatched probability mean / median | 0.1670 / 0.1393 |
| P(p > 0.1 \| matched) | 0.9879 |
| P(p > 0.1 \| unmatched) | 0.8301 |
| median matched probability rank among six queries | 4 |
| matched query is top-probability query | 0.1154 |

The low ECE does not mean that branch detection is good. The predicted
average probability is close to the imbalanced slot prevalence, while
matched and unmatched distributions still overlap heavily.

At threshold 0.10 the model predicts 2,628 branches for 494 GT branches:
exact-count accuracy is 0.1328, missed-branch rate is 0.3563, and extra-branch
rate is 0.8790.

### Threshold-free branch quality

| modality | branch AP |
|---|---:|
| full | 0.0897 |
| no trajectory | 0.1022 |
| trajectory + graph | 0.0408 |

On this checkpoint the trajectory modality does not provide a stable
increment under the threshold-free metric.

### Collapse decomposition

- thresholded duplicate-pair ratio: 1.0;
- oracle-K duplicate-pair ratio: 1.0;
- actual-matched duplicate-pair ratio: 1.0;
- geometry-reference-matched duplicate-pair ratio: 1.0.

For all 71 validation nodes with at least two GT branches, oracle-K,
actual-matched, and geometry-reference-matched queries contain duplicates.
Their oracle-K recall is 0.2599. Therefore the observed collapse is not only
caused by activating unmatched queries at a low threshold.

### Representation homogenization

Full-modality mean pairwise query cosine:

| stage | cosine |
|---|---:|
| learned query embedding | -0.0008 |
| after shared graph-state addition | 0.9996 |
| image cross-attention output | 0.9999995 |
| trajectory cross-attention output | 0.999999997 |
| final fused query | 0.9999563 |

The learned-query norm is about 0.222, while the repeated graph-state token
norm is about 11.264. The expression
`LayerNorm(learned_query + shared_state)` therefore nearly removes query
identity before either cross-attention. This is direct evidence for a real
decoder representation bottleneck.

### Matcher scale and stability

Across all query-to-valid-GT pairs:

| component | median | P90 |
|---|---:|---:|
| endpoint L1 | 0.1296 | 0.2078 |
| direction cosine distance | 0.0101 | 1.1623 |
| existence cost `-p` | -0.1467 | -0.1308 |

The median best-versus-second-best total geometry-cost margin is only
0.00139. Match frequency is also uneven across query IDs:
`[42, 66, 153, 41, 160, 32]`. Existence matching weight 1.0 is not larger
than the normal geometry scale, but at the old checkpoint the existence
scores themselves are almost query-independent. It should be evaluated
together with retraining, not assumed to fix a frozen checkpoint.

### Offset/direction numerical checks

- no NaN or Inf directions;
- no offset has norm below `1e-6`;
- no `tanh` component has absolute value above 0.95;
- matched/unmatched median offset norms are 0.247 / 0.250.

The failure is not explained by `tanh` saturation or near-zero direction
normalization.

## Configurable changes

The base configuration exactly preserves the historical behavior:

- `STAGE3C.MATCHING.EXISTENCE_COST_WEIGHT: 0.0`;
- `STAGE3C.LOSS.EXIST_NO_OBJECT_COEF: 1.0`;
- `STAGE3C.MODEL.QUERY_SELF_ATTENTION_LAYERS: 0`;
- model selection remains validation F1.

Experiment configurations explicitly enable:

- E1: existence matching only;
- E2: no-object coefficient 0.2 only;
- E3: both;
- E4: both plus one query self-attention layer.

When enabled, the one self-attention layer is applied to learned queries
before the large shared graph-state token is added. This lets its residual
LayerNorm bring query-specific features to a comparable scale. It is a
single optional layer, not a multi-layer DETR decoder. It creates no state
dict keys when disabled, so the epoch-16 auxiliary checkpoint remains
strict-load compatible.

The no-object loss uses per-slot BCE. Positive slots have weight 1.0,
negative slots have the configured coefficient, and the result is normalized
by total weight. Coefficient 1.0 calls the historical unweighted BCE
reduction directly.

## Commands

E0:

```bash
CUDA_VISIBLE_DEVICES=4 python scripts/diagnose_stage3c_branch_aux.py \
  --config configs/stage3c_e0_current_checkpoint_diagnostics.yml
```

Fresh E1-E4 runs (do not pass `--resume`):

```bash
python train_branch_aux.py --config configs/stage3c_e1_existence_matching_only.yml
python train_branch_aux.py --config configs/stage3c_e2_no_object_weight_only.yml
python train_branch_aux.py --config configs/stage3c_e3_matching_plus_no_object.yml
python train_branch_aux.py --config configs/stage3c_e4_matching_no_object_self_attn.yml
```

The formal initialization seed is reset after the sanity gate. Shared
parameters therefore start identically in E1-E4; the optional E4 parameters
are constructed last and have their own deterministic initialization.

Multi-branch sanity:

```bash
python train_branch_aux.py \
  --config configs/stage3c_multibranch_overfit.yml \
  --mode sanity
```

This selects 32 cached states with GT branch count 2-4, disables modality and
model dropout, and records branch AP, fixed-threshold exact-count metrics,
oracle-K duplicate/coverage, and matched/unmatched probability separation.

## Verification

On the 237 server:

```bash
python -m unittest discover -s tests -v
```

The final discovery run passed all 134 tests: the previous 119 tests and 15
new matcher, loss, diagnostics, compatibility, initialization, and
forward/backward tests.

E1-E4 and the multi-branch overfit experiment have not been trained. No
claim is made that query collapse is solved. E0 shows that E4 is a justified
experiment; only retrained oracle-K duplicate, multi-branch coverage, branch
AP, probability separation, and extra-branch rate can establish a fix.
