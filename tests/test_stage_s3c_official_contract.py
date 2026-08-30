from pathlib import Path

import yaml

from utils.seg_raster.stage_s3c import (
    EFFECTIVE_SAMPLES_PER_UPDATE, GRADIENT_ACCUMULATION,
    MAX_SAMPLES_SEEN, MICRO_BATCH_PER_GPU, OFFICIAL_SOURCE_SHA,
    OFFICIAL_STATE_KEY_COUNT, SAMPLE_GRID,
)


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_official_and_adaptation_constants() -> None:
    assert OFFICIAL_SOURCE_SHA == "ffcb47e50e48ced717b2ac0e0f8c720ffc083441"
    assert OFFICIAL_STATE_KEY_COUNT == 648
    assert MICRO_BATCH_PER_GPU == 10
    assert GRADIENT_ACCUMULATION == 2
    assert EFFECTIVE_SAMPLES_PER_UPDATE == 20
    assert MAX_SAMPLES_SEEN == 40960
    assert SAMPLE_GRID[0] == 0 and SAMPLE_GRID[-1] == MAX_SAMPLES_SEEN


def test_stage_s3c_config_preserves_sum_loss_adaptation_contract() -> None:
    config = yaml.safe_load((ROOT / "configs/stage_s3c_common.yml").read_text())
    assert config["TRAIN"]["BATCH_SIZE"] == 10
    assert config["S3"]["GRADIENT_ACCUMULATION"] == 2
    assert config["TRAIN"]["SOLVER"]["LEARNING_RATE"] == 1e-5
    assert config["TRAIN"]["SOLVER"]["METHOD"] == "Adam"
    assert config["S3"]["ANCHOR_GRAD_TO_SEG"] is False
