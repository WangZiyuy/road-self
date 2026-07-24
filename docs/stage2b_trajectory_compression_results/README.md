# Stage 2B trajectory candidate compression results

This directory contains portable validation artifacts for the deterministic,
non-learned compression step inserted between Stage 1B fragment retrieval and
the Stage 1C batch builder.

The online-oriented `bounded_near_diverse` strategy was added without changing
the existing `nearest` and full `near_diverse` baselines.

## Inputs and settings

- Cache: `data_self/input/traj_structured/xian/v1`
- GT graph: `data_self/input/graphs/xian.graph`
- Cache basis:
  `trajectory_sample_points_and_segments_supercover_cells`
- Window size: 256 pixels
- Context points: 2
- Maximum time gap: 60 seconds
- Maximum spatial gap: 256 pixels
- Fragment budgets: 32, 64, and 128
- Full `near_diverse` nearest fraction: 0.25
- Bounded nearest fraction: 0.5
- Bounded prepool multiplier: 8
- GT evaluation probe distance: 40 pixels
- GT evidence threshold: 20 pixels

GT information is used only for evaluation after selection. It is not an input
to any compression strategy.

The 8192x8192 background is top-left aligned zero padding of the original
4300x5000 image. No coordinate scaling or offset is applied.

## Bounded strategy

For `N` high-recall fragments and a final budget `K`,
`bounded_near_diverse`:

1. computes exact polyline-to-node distance for all `N` fragments;
2. creates a deterministic distance shortlist of at most
   `M = min(N, 8K)` fragments;
3. keeps the nearest `ceil(0.5K)` fragments;
4. builds only the four-dimensional
   `[closest_x, closest_y, cos(2 theta), sin(2 theta)]` descriptor for the
   shortlist;
5. fills the remaining budget with deterministic greedy max-min selection
   over fragments with a valid nonzero tangent;
6. falls back to distance order if valid tangent candidates are insufficient.

It does not build the full nine-dimensional descriptor for every candidate and
does not perform the full `N`-to-`K` support assignment. Consequently its
`support_count` is invalid by design; the Stage 1C batch keeps a stable tensor
schema by filling ones and setting `fragment_support_count_valid=False`.

## Real GT-node evaluation

The completed analysis covers 639 selected GT road nodes and 1,607 incident
GT branches:

- 369 ordinary degree-2 nodes
- 214 T junctions
- 56 degree-4-or-higher multi-branch junctions

The timings below include each strategy's complete standalone compression call:
distance computation, shortlist construction, descriptor construction,
selection, and support assignment where applicable.

| Budget | Nearest coverage | Near-diverse coverage | Bounded coverage | Near-diverse mean time | Bounded mean time | Bounded speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 89.05% | 92.72% | 92.53% | 199.51 ms | 57.45 ms | 3.47x |
| 64 | 90.85% | 93.03% | 92.97% | 202.48 ms | 67.09 ms | 3.02x |
| 128 | 92.16% | 93.03% | 93.03% | 207.14 ms | 83.17 ms | 2.49x |

The bounded strategy is above `nearest` at all three budgets. Its branch
coverage gap relative to full `near_diverse` is 0.19, 0.06, and 0.00 percentage
points at budgets 32, 64, and 128 respectively.

Descriptor evaluation respects the explicit `8K` limit:

| Budget | Mean descriptor count | Maximum descriptor count |
| ---: | ---: | ---: |
| 32 | 214.1 | 256 |
| 64 | 385.5 | 512 |
| 128 | 651.7 | 1024 |

The selected-fragment mean minimum distance for bounded versus full
`near_diverse` is 30.06 versus 62.89 pixels at `K=32`, 37.91 versus 64.74 at
`K=64`, and 47.33 versus 63.69 at `K=128`. This is consistent with the visual
reduction in far-away abnormal polylines.

The full run took 1122.17 seconds, including fragment queries, all strategy and
budget evaluations, metric aggregation, and six figures. Exact fragment query
time averaged 274.92 ms per node and used the segment-aware grid index rather
than a full trajectory scan.

`compression_summary.json` contains aggregate distributions and per-node-type
results. The full per-node report remains at
`data_self/output/stage2b_bounded_compression/trajectory_compression_analysis.json`
in the local and 237 workspaces. Its SHA-256 is recorded in the portable
summary.

## Visual comparison

Each figure contains:

1. every Stage 1B high-recall candidate;
2. the 64 nearest candidates;
3. the 64 full `near_diverse` representatives;
4. the 64 `bounded_near_diverse` representatives.

Cyan dashed lines are incident GT edges, yellow diamonds are evaluation probe
points, and the green star is the current node. Support labels are shown only
for strategies with a valid support assignment.

- `multi_branch_vertex_0118_all_nearest64_diverse64_bounded64.png`
- `multi_branch_vertex_0307_all_nearest64_diverse64_bounded64.png`
- `multi_branch_vertex_0489_all_nearest64_diverse64_bounded64.png`
- `multi_branch_vertex_0666_all_nearest64_diverse64_bounded64.png`
- `t_junction_vertex_0092_all_nearest64_diverse64_bounded64.png`
- `t_junction_vertex_0116_all_nearest64_diverse64_bounded64.png`

This remains a compute-budget compressor rather than a final learned quality
selector. Reliability and task-conditioned use of fragments remain later
modeling problems.

## Reproduction

```bash
python scripts/analyze_trajectory_compression.py \
  --cache-dir data_self/input/traj_structured/xian/v1 \
  --graph data_self/input/graphs/xian.graph \
  --output-dir data_self/output/stage2b_bounded_compression \
  --background-image data_self/input/imagery_8192/xian.png \
  --max-time-gap-seconds 60 \
  --max-spatial-gap-pixels 256 \
  --budgets 32 64 128 \
  --near-fraction 0.25 \
  --bounded-near-fraction 0.5 \
  --prepool-multiplier 8 \
  --visualization-count 6
```

The complete Stage 0 through Stage 2B regression suite contains 82 tests and
passes in the local environment.
