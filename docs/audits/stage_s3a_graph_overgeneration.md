# Stage S3A graph controls and overgeneration

Graph audit status: **PARTIAL_TERMINATED_PATHOLOGICAL_EXPANSION**. Nine jobs completed successfully. C1-best was terminated as `TERMINATED_PATHOLOGICAL_EXPANSION` after its forensic snapshot; its incomplete partial graph is not reported as a completed graph metric result. C1-latest and the completed C0/C2/C3 controls remain valid. The C0-best common-step matrix is unavailable because C1's step-5120 weights were not retained.

All available APLS values are `deterministic_pixel_graph_approximation`, not official SpaceNet APLS. Latest-checkpoint candidate/undirected/dangling/duplicate edge counts are reported with C1−C0, C1−C2 and C1−C3 deltas so connectivity gains cannot be separated from overgeneration by APLS alone. The best-checkpoint comparison is explicitly incomplete rather than populated from the C1-best partial graph.
