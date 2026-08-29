# Stage S3B Junction-Loss Screen

## Status

NOT_EXECUTED_BY_LR_STABILITY_GATE

## Reason

The preregistered LR stability gate failed, so no Phase B loss run was launched and no junction-loss claim is made.

## Audited but not selected

The frozen data audit found `1016` positive and `130056` negative junction
pixels across the diagnostic plan (`0.007751` positive ratio). It prepared a
capped `pos_weight=32.0` and deterministic gradient-matching scales for both
balanced candidates. These are implementation evidence only: neither balanced
BCE nor balanced BCE plus Dice received optimizer steps, so
`JUNCTION_LOSS_REPAIR` remains `NOT_EXECUTED_BY_LR_STABILITY_GATE`.
