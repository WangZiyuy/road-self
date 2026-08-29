# Stage S3A anchor forensics

Historical anchor provenance: **PASS**, latest step 102400.

Multistep validity: **FAIL**. The fixed-threshold and top-K metrics answer different questions: top-K bypasses 0.3, while threshold recall/false-positive counts can be zero when all probabilities remain below 0.3. The aligned result is not specific against shifted control, so the indirect-anchor causal claim is not retained.
