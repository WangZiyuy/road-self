# Stage 3C-R1 query identity comparison

All variants use the same 32 teacher-forced training states, trajectory fragments, seed, and strictly loaded frozen RPNet.

| Variant | Branch AP | Slot AP | Mean probability gap | Exact count | Oracle recall | Distinct coverage | Oracle duplicate | Graph cosine | Final cosine | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| M0 | 0.2787 | 0.8321 | 0.4173 | 0.4688 | 0.3976 | 0.3976 | 0.0000 | 0.9838 | 0.3489 | False |
| M3 | 1.0000 | 1.0000 | 0.9983 | 1.0000 | 1.0000 | 1.0000 | 0.0270 | 0.9844 | 0.1172 | True |
| M4 | 0.4169 | 0.9783 | 0.9129 | 0.8125 | 0.5904 | 0.5904 | 0.0135 | 0.3327 | 0.2368 | False |

## Decisions

1. Did M3 improve probability separation over M0? **Yes**
2. Does M3 still have query collapse? **No**
   Early graph-conditioned queries remain homogeneous? **Yes**
3. Does M4 restore query identity? **Yes**
4. Does M4 cover distinct GT branches? **No**
5. Run full E1-E4 training? **No**

Qualifying-epoch counts (before the required final-checkpoint decision): M0=5, M3=23, M4=35.
Final-checkpoint regression after a qualifying epoch: M0=True, M3=False, M4=True.

No branch prediction was passed to `Path.push`; no anchor or RPNet architecture was changed.
