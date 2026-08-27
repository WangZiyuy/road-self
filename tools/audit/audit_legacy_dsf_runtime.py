#!/usr/bin/env python3
"""Runtime, shape, anchor-step, parameter, and gradient audit for legacy RPNet.

The harness imports and executes production modules without editing them.  Its
synthetic loss mirrors the set of production outputs but uses mean reduction
to keep the diagnostic numerically bounded.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Any


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat()


def clean_float(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace", check=False,
    )
    return proc.stdout.strip() if proc.returncode == 0 else "UNKNOWN"


def redacted_path(value: str | Path, source_repo: Path, process_cwd: Path) -> str:
    path = Path(value).resolve()
    mappings = (
        (source_repo, "${SOURCE_WORKTREE}"),
        (process_cwd, "${PROCESS_CWD}"),
        (Path(sys.prefix).resolve(), "${PYTHON_PREFIX}"),
        (Path.home().resolve(), "${USER_HOME}"),
    )
    for root, label in mappings:
        try:
            relative = path.relative_to(root)
        except ValueError:
            continue
        return label if str(relative) == "." else label + "/" + relative.as_posix()
    return "${ABSOLUTE_PATH_REDACTED}/" + path.name


def source_provenance(
        repo: Path, snapshot_id: str, model_path: Path, dsf_path: Path) -> dict[str, Any]:
    process_cwd = Path.cwd().resolve()
    return {
        "snapshot_id": snapshot_id,
        "cwd": redacted_path(process_cwd, repo, process_cwd),
        "source_worktree": "${SOURCE_WORKTREE}",
        "branch": git_value(repo, "branch", "--show-current"),
        "HEAD": git_value(repo, "rev-parse", "HEAD"),
        "model.model.__file__": redacted_path(model_path, repo, process_cwd),
        "model.DSFNet.__file__": redacted_path(dsf_path, repo, process_cwd),
        "sys.path": [redacted_path(item or process_cwd, repo, process_cwd) for item in sys.path],
        "imported_source_sha256": {
            "model/model.py": sha256_file(model_path),
            "model/DSFNet.py": sha256_file(dsf_path),
        },
        "monkeypatch_used": False,
        "stub_used": False,
        "audit_harness_used": True,
        "production_model_modified": False,
        "redaction": "Absolute paths are represented by logical labels.",
    }


def tensor_stats(tensor: Any) -> dict[str, Any] | None:
    if tensor is None:
        return None
    detached = tensor.detach()
    finite = detached.isfinite()
    finite_values = detached[finite]
    if finite_values.numel() == 0:
        minimum = maximum = mean = None
    else:
        minimum = clean_float(finite_values.min().item())
        maximum = clean_float(finite_values.max().item())
        mean = clean_float(finite_values.float().mean().item())
    return {
        "shape": list(detached.shape), "dtype": str(detached.dtype),
        "device": str(detached.device), "min": minimum, "max": maximum,
        "mean": mean, "finite_ratio": clean_float(finite.float().mean().item()),
    }


def step_differences(tensor: Any) -> list[dict[str, Any]]:
    if tensor is None or tensor.ndim < 2 or tensor.shape[1] < 4:
        return []
    results = []
    for other in (1, 2, 3):
        delta = (tensor[:, 0] - tensor[:, other]).detach().abs()
        results.append({
            "pair": f"0_vs_{other}", "max_abs": clean_float(delta.max().item()),
            "mean_abs": clean_float(delta.float().mean().item()),
            "exactly_equal": bool((delta == 0).all().item()),
        })
    return results


def module_group(parameter_name: str) -> str:
    if parameter_name.startswith("DSF."):
        sub = parameter_name.split(".")[1]
        if "src_traj" in sub:
            return "DSF.image_unet"
        if sub.startswith("down") and "traj" in sub or sub == "center_traj":
            return "DSF.trajectory_encoder"
        if sub.startswith("up") and "traj" in sub or sub.startswith("trans") and "traj" in sub:
            return "DSF.trajectory_decoder"
        if sub.startswith("traj_"):
            return "DSF.traj_head"
        if sub in {"W_b", "W_s", "W_t"} or sub.startswith("ca_"):
            return "DSF.co_attention"
        if sub.startswith("sfw"):
            return "DSF.sfw"
        if sub.startswith("road") or sub.startswith("conv_road"):
            return "DSF.road_head"
        if sub.startswith("junc") or sub.startswith("conv_junc"):
            return "DSF.junction_head"
        return "DSF.other"
    if parameter_name.startswith("transformer."):
        return "Transformer"
    if parameter_name.startswith("fuse_module_traj."):
        return "fuse_module_traj"
    if parameter_name.startswith("cross_attention."):
        return "cross_attention"
    if parameter_name.startswith("traj_to_img_fc."):
        return "traj_to_img_fc"
    if parameter_name.startswith("stage_1_traj_aerial."):
        return "stage_1_traj_aerial"
    if parameter_name.startswith("stage_1_traj."):
        return "stage_1_traj"
    if parameter_name.startswith("up") and "anchor" in parameter_name or parameter_name.startswith("trans") and "anchor" in parameter_name:
        return "DSF_anchor_decoder"
    if parameter_name == "missing_traj_feature":
        return "missing_traj_feature"
    if parameter_name.startswith(("stage_1.", "stage_2.", "stage_3.", "stage_4.", "stage_5.")):
        return "Res2Net"
    if parameter_name.startswith("fuse_module."):
        return "origin_fuse_module"
    if parameter_name.startswith("decoders."):
        return "origin_anchor_decoder"
    if parameter_name.startswith(("road_seg.", "conv_road_final.")):
        return "origin_road_head"
    if parameter_name.startswith(("junc_seg.", "conv_junc_final.")):
        return "origin_junction_head"
    return parameter_name.split(".")[0]


def owning_module_reached(parameter_name: str, reached: set[str]) -> bool:
    parts = parameter_name.split(".")[:-1]
    for length in range(len(parts), 0, -1):
        if ".".join(parts[:length]) in reached:
            return True
    return False


def synthetic_inputs(torch: Any, device: Any, use_traj: bool) -> dict[str, Any]:
    generator = torch.Generator(device=device)
    generator.manual_seed(20260827)
    aerial = torch.randn((1, 3, 256, 256), generator=generator, device=device)
    raster = torch.rand((1, 1, 256, 256), generator=generator, device=device) if use_traj else None
    walked = torch.zeros((1, 1, 64, 64), device=device)
    walked[:, :, 32, 32] = 1.0
    sequence = torch.tensor(
        [[[[0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5]]]],
        dtype=torch.float32, device=device,
    ) if use_traj else None
    valid = torch.ones((1, 1, 4), dtype=torch.bool, device=device) if use_traj else None
    return {
        "aerial_image": aerial, "traj_image": raster,
        "aerial_traj_image": None, "neighborhood_trajectory_norm": sequence,
        "valid_mask": valid, "walked_path": walked,
        "NUM_TARGETS": 4, "test": False,
    }


def run_one_mode(torch: Any, RPNet: Any, device: Any, *, label: str, model_name: str,
                 enable_trajectory_modules: bool, use_traj: bool, train_mode: bool,
                 backward: bool) -> dict[str, Any]:
    result: dict[str, Any] = {
        "label": label, "model": model_name,
        "enable_trajectory_modules": enable_trajectory_modules,
        "use_traj": use_traj, "module_mode": "train" if train_mode else "eval",
        "backward_requested": backward, "status": "NOT_RUN",
        "production_or_harness": "PRODUCTION_MODEL_WITH_SYNTHETIC_INPUTS",
    }
    model = None
    hooks = []
    try:
        torch.manual_seed(20260827)
        model = RPNet(
            num_targets=4, backbone_pretrained=False,
            enable_trajectory_modules=enable_trajectory_modules,
        ).to(device)
        model.train(train_mode)
        reached: set[str] = set()
        for module_name, module in model.named_modules():
            if not module_name:
                continue
            hooks.append(module.register_forward_hook(
                lambda _module, _inputs, _output, name=module_name: reached.add(name)
            ))
        inputs = synthetic_inputs(torch, device, use_traj)
        inputs["model"] = model_name
        inputs["use_traj"] = use_traj
        context = torch.enable_grad() if backward else torch.no_grad()
        with context:
            outputs = model(**inputs)
            loss = None
            if backward:
                import torch.nn.functional as F
                loss_terms = []
                for key in ("road", "junc", "anchor", "anchor_lowrs"):
                    value = outputs[key]
                    loss_terms.append(F.binary_cross_entropy_with_logits(value, torch.zeros_like(value), reduction="mean"))
                loss = sum(loss_terms)
                loss.backward()
        result["status"] = "PASS"
        result["outputs"] = {key: tensor_stats(outputs.get(key)) for key in ("road", "junc", "anchor", "anchor_lowrs", "traj_road")}
        result["anchor_step_differences"] = step_differences(outputs.get("anchor"))
        result["anchor_lowrs_step_differences"] = step_differences(outputs.get("anchor_lowrs"))
        result["loss"] = clean_float(loss.item()) if loss is not None else None
        result["forward_reached_modules"] = sorted(reached)
        result["plain_tensor_attributes"] = {
            "DSF.temp": {
                "present": bool(hasattr(getattr(model, "DSF", None), "temp")),
                "registered_buffer": "DSF.temp" in dict(model.named_buffers()),
                "device": str(model.DSF.temp.device) if hasattr(getattr(model, "DSF", None), "temp") else None,
            }
        }
        if backward:
            state_keys = set(model.state_dict())
            params = []
            for name, parameter in model.named_parameters():
                grad = parameter.grad
                grad_finite = None
                grad_norm = None
                grad_max = None
                grad_all_zero = None
                if grad is not None:
                    grad_finite = bool(grad.isfinite().all().item())
                    grad_norm = clean_float(grad.float().norm().item())
                    grad_max = clean_float(grad.detach().abs().max().item())
                    grad_all_zero = bool((grad == 0).all().item())
                params.append({
                    "module": module_group(name), "parameter_name": name,
                    "shape": list(parameter.shape), "numel": parameter.numel(),
                    "requires_grad": bool(parameter.requires_grad),
                    "forward_reached": bool(grad is not None or owning_module_reached(name, reached)),
                    "grad_is_none": grad is None, "grad_finite": grad_finite,
                    "grad_norm": grad_norm, "grad_max_abs": grad_max,
                    "grad_all_zero": grad_all_zero,
                    "optimizer_included": bool(parameter.requires_grad),
                    "checkpoint_included": name in state_keys,
                })
            result["parameters"] = params
    except Exception as exc:
        result["status"] = "FAIL"
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc()
    finally:
        for hook in hooks:
            hook.remove()
        if model is not None:
            del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return result


def aggregate_parameters(mode: dict[str, Any]) -> dict[str, Any]:
    params = mode.get("parameters", [])
    total = sum(item["numel"] for item in params)
    trainable = sum(item["numel"] for item in params if item["requires_grad"])
    registered_not_executed = sum(item["numel"] for item in params if not item["forward_reached"])
    executed_grad_none = sum(item["numel"] for item in params if item["forward_reached"] and item["grad_is_none"])
    zero_grad = sum(item["numel"] for item in params if item["grad_all_zero"] is True)
    groups: dict[str, dict[str, Any]] = {}
    for item in params:
        group = groups.setdefault(item["module"], {
            "numel": 0, "forward_reached_numel": 0, "grad_none_numel": 0,
            "zero_grad_numel": 0, "parameter_count": 0,
        })
        group["numel"] += item["numel"]
        group["parameter_count"] += 1
        if item["forward_reached"]:
            group["forward_reached_numel"] += item["numel"]
        if item["grad_is_none"]:
            group["grad_none_numel"] += item["numel"]
        if item["grad_all_zero"] is True:
            group["zero_grad_numel"] += item["numel"]
    for values in groups.values():
        values["fraction_of_total"] = clean_float(values["numel"] / total) if total else 0.0
    return {
        "label": mode["label"], "status": mode["status"], "total_parameters": total,
        "trainable_parameters": trainable, "registered_not_executed_parameters": registered_not_executed,
        "executed_but_grad_none_parameters": executed_grad_none,
        "gradient_all_zero_parameters": zero_grad, "groups": groups,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--snapshot-id", default="BASELINE_13488c7")
    parser.add_argument("--artifact-suffix", default="")
    args = parser.parse_args()
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo))

    import torch
    import_state_before = {
        "CUDA_LAUNCH_BLOCKING": os.environ.get("CUDA_LAUNCH_BLOCKING"),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
    }
    from model import DSFNet as dsf_module
    from model import model as model_module
    RPNet = model_module.RPNet
    import_state_after = {
        "CUDA_LAUNCH_BLOCKING": os.environ.get("CUDA_LAUNCH_BLOCKING"),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
    }
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    runtime: dict[str, Any] = {
        "stage": "seg_raster_stage_s0", "generated_at": now_iso(),
        "source_provenance": source_provenance(
            repo, args.snapshot_id, Path(model_module.__file__).resolve(),
            Path(dsf_module.__file__).resolve()),
        "torch_version": torch.__version__, "cuda_available": bool(torch.cuda.is_available()),
        "requested_device": args.device, "actual_device": str(device),
        "import_side_effect": {"before": import_state_before, "after": import_state_after},
        "runs": [],
    }
    cases = [
        dict(label="origin_none_train", model_name="origin", enable_trajectory_modules=False, use_traj=False, train_mode=True, backward=True),
        dict(label="origin_none_eval", model_name="origin", enable_trajectory_modules=False, use_traj=False, train_mode=False, backward=False),
        dict(label="origin_legacy_train", model_name="origin", enable_trajectory_modules=True, use_traj=True, train_mode=True, backward=True),
        dict(label="DSFNet_legacy_train", model_name="DSFNet", enable_trajectory_modules=True, use_traj=True, train_mode=True, backward=True),
        dict(label="DSFNet_legacy_eval", model_name="DSFNet", enable_trajectory_modules=True, use_traj=True, train_mode=False, backward=False),
    ]
    for case in cases:
        print(f"running {case['label']}", flush=True)
        runtime["runs"].append(run_one_mode(torch, RPNet, device, **case))

    artifacts = args.output_dir.resolve() if args.output_dir else repo / "artifacts"
    suffix = args.artifact_suffix
    write_json(artifacts / f"stage_s0_runtime_audit{suffix}.json", runtime)
    anchor_payload = {
        "stage": "seg_raster_stage_s0", "generated_at": now_iso(),
        "source_provenance": runtime["source_provenance"],
        "runs": [{
            "label": item["label"], "status": item["status"], "model": item["model"],
            "module_mode": item["module_mode"],
            "anchor": item.get("anchor_step_differences", []),
            "anchor_lowrs": item.get("anchor_lowrs_step_differences", []),
            "causal_trace": (
                "DSF full-resolution decoder does not consume next_step; next_step only produces "
                "anchor_lowrs and recursive slot feedback."
                if item["model"] == "DSFNet"
                else "Origin full-resolution decoder consumes next_step at every recursive step."
            ),
        } for item in runtime["runs"]],
    }
    write_json(artifacts / f"stage_s0_anchor_step_audit{suffix}.json", anchor_payload)
    gradient_runs = [item for item in runtime["runs"] if item.get("backward_requested")]
    parameter_payload = {
        "stage": "seg_raster_stage_s0", "generated_at": now_iso(),
        "source_provenance": runtime["source_provenance"],
        "summaries": [aggregate_parameters(item) for item in gradient_runs],
        "runs": [{"label": item["label"], "status": item["status"], "parameters": item.get("parameters", [])} for item in gradient_runs],
    }
    gradient_payload = {
        "stage": "seg_raster_stage_s0", "generated_at": now_iso(),
        "source_provenance": runtime["source_provenance"],
        "loss_contract": "Production output set and BCEWithLogits semantics; audit uses mean rather than production sum reduction.",
        "summaries": [aggregate_parameters(item) for item in gradient_runs],
        "runs": [{
            "label": item["label"], "status": item["status"], "loss": item.get("loss"),
            "parameters": item.get("parameters", []), "error": item.get("error", ""),
        } for item in gradient_runs],
    }
    write_json(artifacts / f"stage_s0_parameter_inventory{suffix}.json", parameter_payload)
    write_json(artifacts / f"stage_s0_gradient_inventory{suffix}.json", gradient_payload)
    print(json.dumps({
        "device": str(device), "runs": [{"label": item["label"], "status": item["status"]} for item in runtime["runs"]]
    }, allow_nan=False))
    return 0 if all(item["status"] == "PASS" for item in runtime["runs"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
