import pytest

from utils.seg_raster.stage_s3c import (
    GraphResourceSnapshot, assert_json_finite, graph_resource_status,
    validate_commit_paths,
)


def test_uniform_graph_cap_is_not_reported_as_pass() -> None:
    result = graph_resource_status(GraphResourceSnapshot(
        iterations=3000, vertices=100, directed_edges=200,
        elapsed_seconds=10.0))
    assert result["status"] == "RESOURCE_CAP_REACHED"
    assert result["natural_termination"] is False
    assert result["reached_caps"] == ["MAX_GRAPH_ITERATIONS"]


def test_finite_json_and_commit_manifest_exclude_large_runtime_files() -> None:
    assert_json_finite({"value": [0.0, 1.0]})
    with pytest.raises(ValueError, match="NaN"):
        assert_json_finite({"value": float("nan")})
    validate_commit_paths([
        "artifacts/stage_s3c_conclusion.json",
        "docs/audits/stage_s3c_final_report.md",
        "tests/test_stage_s3c_graph_caps.py",
    ])
    with pytest.raises(ValueError):
        validate_commit_paths(["data_self/stage_s3c/checkpoints/latest.pth.tar"])
