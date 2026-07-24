# Stage 3B multimodal auxiliary branch decoder

Stage 3B adds an independent, unordered immediate-branch prediction side
head. It does not modify RPNet, the recursive anchor head, `NUM_TARGETS`,
`map_to_coordinate`, `Path.push`, `train.py`, or `infer.py`.

## Data flow

```text
frozen image-only RPNet feature_maps["stage_fuse"]
                                      \
Stage 3A graph state -> GraphStateEncoder -> state token
                                      /
Stage 1B full query
-> bounded_near_diverse(K=64, prepool=8, near_fraction=0.5)
-> Stage 1C batch
-> Stage 2A TrajectoryFragmentEncoder
-> unordered fragment tokens
                                      \
                    MultiModalBranchQueryDecoder
                    -> 6 unordered auxiliary slots
                    -> Hungarian matching
                    -> existence + endpoint + direction losses
```

The six learned queries are independent of VecRoad's four recursive
`NUM_TARGETS` maps. Predicted branch endpoints are never sent to `Path.push`.

## Modules

- `model/graph_state_encoder.py`
  - preserves continuous directions;
  - applies a shared edge MLP;
  - uses masked mean and max pooling;
  - produces a finite token for an empty explored-edge set.
- `model/branch_query_decoder.py`
  - pools `stage_fuse` to 16x16 and adds a two-dimensional positional
    encoding;
  - treats trajectory fragments as an unordered set without fragment-index
    positional embeddings;
  - performs separate image and trajectory cross-attention;
  - fuses both contexts with the graph state;
  - derives branch direction by normalizing the predicted endpoint offset;
  - bypasses all-masked trajectory attention and uses an exact zero
    trajectory context for no-trajectory samples.
- `model/branch_set_loss.py`
  - matches slots to valid GT branches with endpoint L1 and direction cosine
    costs;
  - supervises all unmatched slots as no-branch;
  - applies Smooth-L1 endpoint and cosine direction losses only to matched
    slots.

Returned attention weights are diagnostic attention allocation only. They are
not trajectory-support probabilities and are not trained with support labels.

## Real-data smoke

Command executed:

```bash
python scripts/smoke_multimodal_branch_decoder.py \
  --config configs/baseline_image_only.yml \
  --cache-dir data_self/input/traj_structured/xian/v1 \
  --device cpu \
  --max-attempts 1024 \
  --output data_self/output/stage3b_smoke/stage3b_multimodal_branch_smoke.json
```

The configured image-only checkpoint was not present on the local machine:

```text
data_self/baseline_image_only_original/ckpt/image_only_original.latest.pth.tar
```

Consequently, the recorded run uses a randomly initialized but frozen RPNet
for interface validation, as allowed by the Stage 3B specification. The
auxiliary heads are also randomly initialized. The loss values below are not
accuracy or convergence results.

| Node type | GT branches | Full fragments | Kept fragments |
| --- | ---: | ---: | ---: |
| Ordinary | 1 | 1949 | 64 |
| T junction | 3 | 1908 | 64 |
| Multi-branch | 3 | 1981 | 64 |

Observed tensor shapes:

| Tensor | Shape |
| --- | --- |
| `stage_fuse` | `[3, 128, 64, 64]` |
| `fragment_tokens` | `[3, 64, 128]` |
| `state_token` | `[3, 128]` |
| `branch_exist_logits` | `[3, 6]` |
| `branch_offsets_norm` | `[3, 6, 2]` |
| `branch_directions` | `[3, 6, 2]` |

CPU smoke observations:

- frozen RPNet feature extraction: 4141.97 ms;
- auxiliary forward: 1215.38 ms;
- auxiliary backward: 170.66 ms;
- total loss: 1.774513;
- existence loss: 0.763092;
- endpoint loss: 0.013107;
- direction loss: 0.998314;
- matched GT branches: 7;
- every trainable auxiliary parameter received a finite gradient;
- frozen RPNet received no gradient;
- all outputs were finite;
- the explicit no-trajectory pass was finite.

These are one-run interface-smoke timings on CPU, not benchmark results.

To validate a compatible trained image-only RPNet explicitly:

```bash
python scripts/smoke_multimodal_branch_decoder.py \
  --config configs/baseline_image_only.yml \
  --cache-dir data_self/input/traj_structured/xian/v1 \
  --checkpoint /path/to/image_only_original.latest.pth.tar \
  --device cuda
```

An explicitly requested missing or incompatible checkpoint raises an error;
the script does not silently substitute another checkpoint.

## Tests

Executed:

```bash
python -m unittest discover -s tests -v
```

Result: 108 tests passed.

Stage 3B coverage includes output shapes, target-order-invariant matching,
zero/one/multiple target sets, unmatched-query supervision, no-trajectory
samples, fragment permutation and padding invariance, empty graph state,
finite gradients for all three modalities, and finite backward propagation.

No trajectory-support labels, reliability loss, trajectory spatial
projection, anchor residual enhancement, formal joint training, or
`follow_output` curriculum is implemented.
