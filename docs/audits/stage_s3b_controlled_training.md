# Stage S3B Controlled Training

## Status

NOT_EXECUTED_BY_LR_STABILITY_GATE

## Boundary

Phase C aligned/zero/shifted controls, conditional anchor evaluation, and graph evaluation were not executed.

All six Phase A runs retained the common nine-checkpoint grid from step `0` to
`20480`, and their first 100 sample identities and common tensors matched.
This does not create a baseline-controlled R0/R1/R2/R3 common step: the
predeclared gate stopped the protocol before Phase B and Phase C.

Consequently, segmentation causal, anchor specificity, multistep-anchor, graph
stability, and computational-feasibility statuses are all
`NOT_EXECUTED_BY_LR_STABILITY_GATE`, not `PASS` or ordinary cancellation.
