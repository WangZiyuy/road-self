# Stage 3A graph state and immediate branch results

This directory contains portable validation artifacts for the Stage 3A
non-model interfaces:

- one queued VecRoad search visit, including its parent and incoming edge;
- continuous incoming and explored-neighbor directions;
- the immediate GT branch set derived from `target_poses[0]` only.

No branch-query model, matching, trajectory fusion, or new loss is used.

## Data flow

The analysis calls the existing training implementation directly:

```text
Path.pop_state
-> Path.make_path_input
-> Path.get_target_poses
-> build_graph_state
-> build_immediate_branch_targets
-> Path.push(follow_target)
```

The complete immediate branch set is measured before the existing
`follow_target` key-point subsampling is applied for graph progression.

`NUM_TARGETS` remains 4. It is not reinterpreted as four branches:

- `target_poses[0]` contains endpoints reached immediately from the current
  node and is the only source of Stage 3A branch targets;
- `target_poses[1:]` contains recursive future points and is excluded.

## Settings

- Config: `configs/baseline_image_only.yml`
- Region: Xian
- Trajectory mode: `none`
- Window size: 256
- Step length: 20
- Maximum explored neighbors retained: 8
- Fixed seed: 20260724
- Sampled training search states: 512

The 512 states contain:

- 352 ordinary-road states;
- 103 GT degree-3 junction states;
- 34 GT degree-4-or-higher junction states;
- 23 other states, including unmatched/random starts or GT endpoints not in
  the three requested categories.

## Immediate branch statistics

Overall branch-count histogram:

| Immediate branches | State count |
| ---: | ---: |
| 0 | 49 |
| 1 | 334 |
| 2 | 53 |
| 3 | 57 |
| 4 | 19 |

The maximum observed immediate branch count is 4.

By GT node type:

| Type | States | Branch-count histogram | Mean branches |
| --- | ---: | --- | ---: |
| Ordinary | 352 | `0:16, 1:323, 2:13` | 0.991 |
| T junction | 103 | `0:5, 1:8, 2:39, 3:51` | 2.320 |
| Multi-branch | 34 | `0:5, 1:3, 2:1, 3:6, 4:19` | 2.912 |

An ordinary GT degree-2 starting node can expose two immediate directions
when it has no incoming edge. An entered T junction usually exposes two
remaining branches, while a T-junction starting state can expose all three.
This is why immediate branch count is conditioned on exploration state rather
than being identical to static GT degree.

## Exploration-state statistics

- Incoming direction valid: 400/512 = 78.13%
- Explored-neighbor histogram:
  `0:112, 1:378, 2:18, 3:2, 4:2`
- Maximum observed explored-neighbor count: 4
- States with nonempty future target slots: 314

The 314 nonempty future-slot cases are ordinary recursive road supervision;
they are deliberately not counted as additional immediate branches.

## Visualizations

Each figure shows:

- current node: green star;
- incoming direction: blue arrow pointing from the parent into the node;
- generated-graph neighbor direction: orange line;
- immediate GT branch: magenta dashed line and yellow endpoint;
- local 256x256 window: red rectangle.

Files:

- `ordinary_sample_0004.png`
- `ordinary_sample_0005.png`
- `t_junction_sample_0023.png`
- `t_junction_sample_0040.png`
- `multi_branch_sample_0016.png`
- `multi_branch_sample_0082.png`

`stage3a_summary.json` contains the portable aggregate report. The full
per-state report remains under
`data_self/output/stage3a_graph_branch_state/stage3a_graph_branch_state.json`;
its SHA-256 is recorded in the summary.

## Reproduction

```bash
python scripts/analyze_stage3a_graph_branch_state.py \
  --config configs/baseline_image_only.yml \
  --output-dir data_self/output/stage3a_graph_branch_state \
  --max-states 512 \
  --max-attempts 8192 \
  --max-explored-edges 8 \
  --seed 20260724 \
  --background-image data_self/input/imagery_8192/xian.png \
  --visualizations-per-type 2
```
