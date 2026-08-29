# Stage S3B Optimizer and Loss Contract

## Historical contract

The historical optimizer is Adam with base LR `1e-4`, betas `0.9/0.99`,
weight decay `2e-4`, one parameter group, no scheduler, and no warmup.
Gradient values are clipped at `10000`. Road and junction use
`BCEWithLogits(reduction=sum)`; anchor loss sums the full- and low-resolution
BCE terms over valid target channels. The three task weights are all `1.0`.

## Diagnostic scope

Sixteen frozen remote CUDA batches were audited with independent road,
junction, anchor, and total backward passes. Positive pixel ratios were
`0.046257` for road, `0.007751` for junction, and `0.003452` for anchor.
The junction imbalance audit produced raw positive weight `128.007874`, capped
at the preregistered value `32.0`.

The initial mean junction-head gradient norm was `36766.956609` for the legacy
loss and `69246.169308` for balanced BCE. The frozen matching scale was
`0.530960152407` for balanced BCE and `0.530959993978` for balanced BCE plus
Dice. These values were audited but were not used in training because the
Phase A LR stability gate failed before Phase B.

## Provenance

Formal training code: `02068550f2acd97de6ae44dc97ffde0ae72344ff`; result reducer: `5357f44baf94b07051d25d3f5d1ab0a2ce87d144`.
