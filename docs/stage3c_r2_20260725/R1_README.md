# Stage 3C-R1 query identity comparison

All variants use the same 32 teacher-forced training states, trajectory fragments, seed, and strictly loaded frozen RPNet.

| Variant | Branch AP | Slot AP | Mean probability gap | Exact count | Oracle recall | Distinct coverage | Oracle duplicate | Graph cosine | Final cosine | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| M0 | 0.9972 | 0.9994 | 0.9523 | 0.9688 | 0.9759 | 0.9759 | 0.0135 | 0.9845 | 0.1941 | True |
| M3 | 1.0000 | 1.0000 | 0.9954 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.9841 | 0.1424 | True |
| M4 | 1.0000 | 1.0000 | 0.9795 | 0.9688 | 1.0000 | 1.0000 | 0.0000 | 0.3526 | 0.1716 | True |

## Best/final checkpoint lifecycle

| Variant | Best epoch | Final epoch | Best AP | Final AP | Best coverage | Final coverage | Best duplicate | Final duplicate | Late regression |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| M0 | 550 | 600 | 0.9972 | 0.2787 | 0.9759 | 0.3976 | 0.0135 | 0.0000 | True |
| M3 | 430 | 600 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0270 | True |
| M4 | 210 | 600 | 1.0000 | 0.4169 | 1.0000 | 0.5904 | 0.0000 | 0.0135 | True |

## Decisions

1. Did M3 improve probability separation over M0? **Yes**
2. Does M3 still have query collapse? **No**
   Early graph-conditioned queries remain homogeneous? **Yes**
3. Does M4 restore query identity? **Yes**
4. Does M4 cover distinct GT branches? **Yes**
5. Run full E1-E4 training? **Yes**

Qualifying-epoch counts (before the required final-checkpoint decision): M0=5, M3=23, M4=35.
Final-checkpoint regression after a qualifying epoch: M0=True, M3=True, M4=True.

No branch prediction was passed to `Path.push`; no anchor or RPNet architecture was changed.
