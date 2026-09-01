# Stage S3E Projection Zero-Init

Z1 exactly reproduced the existing default-initialized N1 path at sample 0 and the common nonzero checkpoint. Z2 satisfied exact sample-0 image identity, and its first-step road-head gradient and optimizer update matched Z0. The projection received a nonzero loss gradient while the encoder loss gradient was zero at the first step, as expected for a zero projection.

Zero initialization removed the initial ranking damage but did not repair the final learned failure. Relative to Z1, Z2 improved final aligned F1 by 0.004659, IoU by 0.002907, and AUPRC by 0.001173. The final null AUPRC head-drift gap shrank by only 5.29%; F1 and IoU gap magnitudes became worse.

Therefore `ZERO_INIT_ROOT_CAUSE_STATUS = INITIAL_HARM_CONFIRMED_BUT_NOT_MAIN_FINAL_CAUSE`. Zero initialization is a justified safety property, not a complete fix.
