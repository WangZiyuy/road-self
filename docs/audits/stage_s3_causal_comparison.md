# Stage S3R causal comparison

## Screening result

This is a single-seed screen, not a statistical-significance claim.

- Segmentation causal screen: PROMISING.
- Indirect anchor screen: PROMISING.
- Joint screen: PROMISING, with an anchor regression caveat.
- Multi-seed recommendation: GO, subject to separate authorization.

## Segmentation

At threshold 0.3, C1-C0 road F1 was +0.0091886922, road IoU +0.0046717191, and junction F1 0. Their mean was +0.0046201371, versus -0.0011309744 for C2 and -0.0009052068 for C3. Correct alignment passed the declared spatial-control screen. Junction F1 remained zero in every group and absolute scores were low. J1-J0 segmentation composite was +0.0264674319. Coverage-stratified metrics and positive-ratio histograms are in the JSON artifact.

## Anchor

C1-C0 top-K recall was +0.0833333333; localization error improved by 35.7711569787 and missed branches by 2. Isolation tests still prove there is no direct raw-raster path to anchor. J1-J0 top-K recall regressed by 0.0416666667, so joint optimization is not evidence of anchor improvement.

## Conditional graph evaluation

The gate passed, so C0/C1/J0/J1 ran on one frozen 4096x4096 split canvas and validation extent [2176,0,4096,4096]. Approximate APLS was C0=0.2572694597, C1=0.2782527173, J0=0.2253922483, J1=0.2348697375.

The server lacked the official APLS jar. These values are a declared deterministic pixel-graph approximation, not official SpaceNet APLS. The initial 8192 raster graph attempt lacked required raster tiles and is excluded from formal comparison.

## Interpretation

The aligned binary raster produced the strongest detach-mode segmentation change and passed the declared indirect-anchor screen for this seed. A preregistered multi-seed replication is warranted, but low scores, zero junction F1, approximate graph metrics, and post-launch GPU contention remain risks.
