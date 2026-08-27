"""Audit assertions for the unmodified legacy DSF integration.

These tests document production behavior; they do not repair or normalize it.
"""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8", errors="replace")


def payload(path: str):
    return json.loads(text(path))


def test_model2_is_a_compatibility_import_not_a_second_rpnet():
    source = text("model/model2.py")
    assert "Compatibility import" in source
    assert "from .model import" in source
    assert "class RPNet" not in source


def test_no_checked_in_config_selects_dsfnet():
    selected = []
    for path in (ROOT / "configs").glob("*.yml"):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            stripped = line.split("#", 1)[0].strip().lower().replace('"', "").replace("'", "")
            if stripped == "model: dsfnet":
                selected.append(path.name)
    assert selected == []


def test_dsf_probability_heads_conflict_with_training_logits_loss():
    dsf = text("model/DSFNet.py")
    train = text("train.py")
    assert dsf.count("nn.Sigmoid()") >= 3
    assert "binary_cross_entropy_with_logits" in train


def test_anchor_inference_does_not_fetch_or_supply_dsf_raster():
    infer = text("infer.py")
    assert "include_raster=False" in infer
    assert "traj_image=None" in infer


def test_dsf_test_path_returns_quarter_resolution_heads_without_upsample():
    model = text("model/model.py")
    assert "return {'road': road_final, 'junc': junc_final}" in model
    assert "return {\n                    'road': upsample(road_final, 4)" in model


def test_traj_road_is_not_part_of_production_total_loss():
    train = text("train.py")
    loss_block = train[train.index("loss = anchor_loss + road_loss + junc_loss") - 200:
                       train.index("loss = anchor_loss + road_loss + junc_loss") + 100]
    assert "traj" not in loss_block.lower()


def test_dsf_full_resolution_decoder_does_not_consume_next_step():
    model = text("model/model.py")
    start = model.index("if model == 'origin':", model.index("for index in range(num_targets):"))
    end = model.index("feature_maps['decoded_ft_4_step_{}'", start)
    branch = model[start:end]
    dsf_branch = branch[branch.index("else:"):]
    assert "next_step" not in dsf_branch


def test_repository_inventory_counts_and_stage_s0_file_origins_reconcile():
    inventory = payload("artifacts/stage_s0_repository_inventory.json")
    candidate = inventory["dirty_untracked_candidate_code_count"]
    relevant = inventory["dirty_untracked_relevant_after_review_count"]
    irrelevant = inventory["dirty_untracked_irrelevant_after_review_count"]
    assert candidate == relevant + irrelevant
    assert (candidate, relevant, irrelevant) == (537, 63, 474)
    assert inventory["count_consistency"]["dirty_candidates_equal_relevant_plus_irrelevant"]

    generated = [item for item in inventory["files"] if item["generated_by_stage_s0"]]
    assert len(generated) == inventory["stage_s0_generated_file_count"]
    assert all(item["repository_file_origin"] == "stage_s0_generated_file" for item in generated)
    assert all(not item["category"].startswith("dead/legacy") for item in generated)
    required_generated = [
        item for item in inventory["files"]
        if item["snapshot_id"] == "BASELINE_13488c7" and (
            item["path"].startswith("artifacts/stage_s0_")
            or item["path"].startswith("tools/audit/")
            or item["path"] == "tests/test_stage_s0_audit.py"
        )
    ]
    assert required_generated
    assert all(item["generated_by_stage_s0"] for item in required_generated)


def test_checkpoint_conclusion_preserves_scope_and_current_model_mismatch():
    conclusion = payload("artifacts/stage_s0_conclusion.json")
    required = (
        "在所比较的 epoch40/50 checkpoint 对中，全部 368 个 DSF tensors 和全部 74 个 "
        "DSF anchor decoder tensors 均逐元素完全相同；Transformer、fuse_module_traj 和 "
        "Res2Net 等模块存在变化。"
    )
    assert conclusion["checkpoint_scope"]["required_conclusion"] == required
    assert conclusion["checkpoint_scope"]["checkpoint_source_commit_provenance"] == "UNKNOWN"
    alignment = conclusion["checkpoint_scope"]["current_model_key_alignment"]
    assert alignment == {
        "missing_expected_key_count": 16,
        "extra_checkpoint_key_count": 28,
        "fully_compatible": False,
    }


def test_source_provenance_scope_is_limited_to_twelve_critical_files():
    provenance = payload("artifacts/stage_s0_source_provenance.json")
    assert provenance["summary"]["critical_file_count"] == 12
    assert provenance["summary"]["content_different_after_line_ending_normalization_count"] == 0
    assert all(item["identical_after_line_ending_normalization"] for item in provenance["files"])
    assert "does not assert" in provenance["summary"]["scope_warning"]


def test_commit_manifest_hashes_and_exclusions_are_exact():
    manifest = payload("artifacts/stage_s0_commit_manifest.json")
    assert not manifest["manifest_self_policy"]["listed_in_entries"]
    assert manifest["summary"]["listed_entry_count"] == len(manifest["entries"])
    for item in manifest["entries"]:
        path = ROOT / item["path"]
        assert item["generated_by_stage_s0"]
        assert item["intended_for_commit"]
        assert path.stat().st_size == item["size_bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
    assert manifest["summary"]["checkpoint_or_model_weight_file_count"] == 0
    assert manifest["summary"]["dataset_file_count"] == 0
    assert manifest["summary"]["cache_file_count"] == 0
    assert manifest["summary"]["production_file_count"] == 0


def test_redaction_json_counts_include_the_report_itself():
    report = payload("artifacts/stage_s0_redaction_audit.json")
    assert report["redaction_target_count"] == len(report["files"])
    assert report["total_stage_s0_json_count_including_redaction_report"] == len(report["files"]) + 1
