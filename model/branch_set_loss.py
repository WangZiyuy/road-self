"""Hungarian set matching and auxiliary immediate-branch losses."""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


MatchIndices = Tuple[torch.Tensor, torch.Tensor]


def _validate_inputs(
    predictions: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
) -> Tuple[int, int, int]:
    required_predictions = (
        "branch_exist_logits",
        "branch_offsets_norm",
        "branch_directions",
    )
    required_targets = (
        "branch_offsets_norm",
        "branch_directions",
        "branch_mask",
    )
    for key in required_predictions:
        if key not in predictions:
            raise KeyError("predictions are missing {!r}".format(key))
    for key in required_targets:
        if key not in targets:
            raise KeyError("targets are missing {!r}".format(key))

    logits = predictions["branch_exist_logits"]
    offsets = predictions["branch_offsets_norm"]
    directions = predictions["branch_directions"]
    if logits.ndim != 2:
        raise ValueError("branch_exist_logits must have shape [B, K]")
    batch_size, query_count = logits.shape
    if tuple(offsets.shape) != (batch_size, query_count, 2):
        raise ValueError("predicted offsets must have shape [B, K, 2]")
    if tuple(directions.shape) != (batch_size, query_count, 2):
        raise ValueError("predicted directions must have shape [B, K, 2]")

    target_offsets = targets["branch_offsets_norm"]
    target_directions = targets["branch_directions"]
    target_mask = targets["branch_mask"]
    if target_offsets.ndim != 3 or target_offsets.shape[-1] != 2:
        raise ValueError("target offsets must have shape [B, M, 2]")
    target_count = target_offsets.shape[1]
    if tuple(target_offsets.shape[:1]) != (batch_size,):
        raise ValueError("prediction and target batch sizes differ")
    if tuple(target_directions.shape) != (
            batch_size, target_count, 2):
        raise ValueError("target directions must have shape [B, M, 2]")
    if tuple(target_mask.shape) != (batch_size, target_count):
        raise ValueError("branch_mask must have shape [B, M]")
    return batch_size, query_count, target_count


def hungarian_match_branches(
    predictions: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    endpoint_cost_weight: float = 1.0,
    direction_cost_weight: float = 1.0,
    existence_cost_weight: float = 0.0,
) -> List[MatchIndices]:
    """Match unordered predicted slots to valid immediate GT branches."""

    batch_size, _, _ = _validate_inputs(predictions, targets)
    if (
            endpoint_cost_weight < 0.0
            or direction_cost_weight < 0.0
            or existence_cost_weight < 0.0):
        raise ValueError("matching cost weights must be non-negative")

    components = branch_matching_cost_components(
        predictions, targets)
    pred_offsets = predictions["branch_offsets_norm"]
    target_mask = targets["branch_mask"].to(
        device=pred_offsets.device, dtype=torch.bool)

    # Build every small K x M cost matrix on the accelerator first and
    # transfer the batch once. The previous per-sample `.cpu()` call forced
    # B device synchronizations per optimizer step even though SciPy itself
    # only solves tiny matrices.
    batched_cost = (
        float(endpoint_cost_weight) * components["endpoint"]
        + float(direction_cost_weight) * components["direction"]
        + float(existence_cost_weight) * components["existence"]
    ).detach().cpu().numpy()
    target_mask_cpu = target_mask.detach().cpu().numpy()

    matches = []
    for batch_index in range(batch_size):
        valid_target_columns = target_mask_cpu[batch_index].nonzero()[0]
        if valid_target_columns.size == 0:
            empty = torch.empty(
                0, dtype=torch.long, device=pred_offsets.device)
            matches.append((empty, empty))
            continue
        prediction_rows, compact_target_columns = linear_sum_assignment(
            batched_cost[batch_index][:, valid_target_columns])
        prediction_indices = torch.as_tensor(
            prediction_rows,
            dtype=torch.long,
            device=pred_offsets.device,
        )
        matched_target_indices = torch.as_tensor(
            valid_target_columns[compact_target_columns],
            dtype=torch.long,
            device=pred_offsets.device,
        )
        matches.append((prediction_indices, matched_target_indices))
    return matches


def branch_matching_cost_components(
    predictions: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Return unweighted [B, K, M] matching-cost components.

    Existence is broadcast over every target branch because it is a property
    of a query slot, not of a particular GT endpoint.  Keeping this helper
    separate makes the matching scale observable without changing assignment.
    """

    _validate_inputs(predictions, targets)
    pred_offsets = predictions["branch_offsets_norm"]
    pred_directions = F.normalize(
        predictions["branch_directions"], p=2, dim=-1, eps=1e-6)
    target_offsets = targets["branch_offsets_norm"].to(
        device=pred_offsets.device, dtype=pred_offsets.dtype)
    target_directions = F.normalize(
        targets["branch_directions"].to(
            device=pred_offsets.device, dtype=pred_offsets.dtype),
        p=2,
        dim=-1,
        eps=1e-6,
    )
    endpoint_cost = torch.cdist(
        pred_offsets, target_offsets, p=1)
    direction_cost = 1.0 - torch.matmul(
        pred_directions, target_directions.transpose(1, 2))
    existence_cost = -torch.sigmoid(
        predictions["branch_exist_logits"]
    ).unsqueeze(-1).expand_as(endpoint_cost)
    return {
        "endpoint": endpoint_cost,
        "direction": direction_cost,
        "existence": existence_cost,
    }


class BranchSetCriterion(nn.Module):
    """Existence, endpoint, and direction losses after Hungarian matching."""

    def __init__(
        self,
        existence_weight: float = 1.0,
        endpoint_weight: float = 1.0,
        direction_weight: float = 1.0,
        endpoint_cost_weight: float = 1.0,
        direction_cost_weight: float = 1.0,
        match_cost_exist_weight: float = 0.0,
        exist_no_object_coef: float = 1.0,
        debug_cost_statistics: bool = False,
    ) -> None:
        super().__init__()
        weights = (
            existence_weight,
            endpoint_weight,
            direction_weight,
            endpoint_cost_weight,
            direction_cost_weight,
            match_cost_exist_weight,
            exist_no_object_coef,
        )
        if any(weight < 0.0 for weight in weights):
            raise ValueError("loss and matching weights must be non-negative")
        self.existence_weight = float(existence_weight)
        self.endpoint_weight = float(endpoint_weight)
        self.direction_weight = float(direction_weight)
        self.endpoint_cost_weight = float(endpoint_cost_weight)
        self.direction_cost_weight = float(direction_cost_weight)
        self.match_cost_exist_weight = float(
            match_cost_exist_weight)
        self.exist_no_object_coef = float(exist_no_object_coef)
        self.debug_cost_statistics = bool(debug_cost_statistics)
        self._printed_debug_statistics = False

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, object]:
        batch_size, query_count, _ = _validate_inputs(
            predictions, targets)
        logits = predictions["branch_exist_logits"]
        pred_offsets = predictions["branch_offsets_norm"]
        pred_directions = predictions["branch_directions"]
        target_offsets = targets["branch_offsets_norm"].to(
            device=logits.device, dtype=pred_offsets.dtype)
        target_directions = targets["branch_directions"].to(
            device=logits.device, dtype=pred_directions.dtype)

        matches = hungarian_match_branches(
            predictions,
            targets,
            endpoint_cost_weight=self.endpoint_cost_weight,
            direction_cost_weight=self.direction_cost_weight,
            existence_cost_weight=self.match_cost_exist_weight,
        )
        existence_targets = torch.zeros_like(logits)
        matched_pred_offsets = []
        matched_target_offsets = []
        matched_pred_directions = []
        matched_target_directions = []
        for batch_index, (
                prediction_indices, target_indices) in enumerate(matches):
            existence_targets[batch_index, prediction_indices] = 1.0
            if prediction_indices.numel() == 0:
                continue
            matched_pred_offsets.append(
                pred_offsets[batch_index].index_select(
                    0, prediction_indices))
            matched_target_offsets.append(
                target_offsets[batch_index].index_select(
                    0, target_indices))
            matched_pred_directions.append(
                pred_directions[batch_index].index_select(
                    0, prediction_indices))
            matched_target_directions.append(
                target_directions[batch_index].index_select(
                    0, target_indices))

        if batch_size == 0 or query_count == 0:
            existence_loss = logits.sum() * 0.0
            existence_weights = torch.zeros_like(logits)
        else:
            loss_raw = F.binary_cross_entropy_with_logits(
                logits, existence_targets, reduction="none")
            existence_weights = torch.full_like(
                logits, self.exist_no_object_coef)
            existence_weights = torch.where(
                existence_targets > 0.5,
                torch.ones_like(existence_weights),
                existence_weights,
            )
            if self.exist_no_object_coef == 1.0:
                # Keep the original reduction path bit-for-bit for old
                # configurations and checkpoints.
                existence_loss = F.binary_cross_entropy_with_logits(
                    logits, existence_targets)
            else:
                weight_sum = existence_weights.sum().clamp_min(
                    torch.finfo(logits.dtype).eps)
                existence_loss = (
                    loss_raw * existence_weights).sum() / weight_sum

        if matched_pred_offsets:
            selected_pred_offsets = torch.cat(
                matched_pred_offsets, dim=0)
            selected_target_offsets = torch.cat(
                matched_target_offsets, dim=0)
            endpoint_loss = F.smooth_l1_loss(
                selected_pred_offsets,
                selected_target_offsets,
            )
            selected_pred_directions = F.normalize(
                torch.cat(matched_pred_directions, dim=0),
                p=2,
                dim=-1,
                eps=1e-6,
            )
            selected_target_directions = F.normalize(
                torch.cat(matched_target_directions, dim=0),
                p=2,
                dim=-1,
                eps=1e-6,
            )
            direction_loss = (
                1.0
                - (
                    selected_pred_directions
                    * selected_target_directions
                ).sum(dim=-1)
            ).mean()
            matched_count = selected_pred_offsets.shape[0]
        else:
            endpoint_loss = pred_offsets.sum() * 0.0
            direction_loss = pred_directions.sum() * 0.0
            matched_count = 0

        total_loss = (
            self.existence_weight * existence_loss
            + self.endpoint_weight * endpoint_loss
            + self.direction_weight * direction_loss
        )
        if (
                self.debug_cost_statistics
                and not self._printed_debug_statistics):
            components = branch_matching_cost_components(
                predictions, targets)
            target_mask = targets["branch_mask"].to(
                device=logits.device, dtype=torch.bool)
            expanded_mask = target_mask.unsqueeze(1).expand_as(
                components["endpoint"])
            statistics = []
            for name in ("endpoint", "direction", "existence"):
                values = components[name][expanded_mask].detach()
                if values.numel() == 0:
                    statistics.append("{}=empty".format(name))
                else:
                    median = torch.quantile(values.float(), 0.5)
                    p90 = torch.quantile(values.float(), 0.9)
                    statistics.append(
                        "{} median={:.6f} p90={:.6f}".format(
                            name, float(median), float(p90)))
            print(
                "branch matcher cost scales: {}".format(
                    "; ".join(statistics)),
                flush=True,
            )
            self._printed_debug_statistics = True
        return {
            "loss": total_loss,
            "existence_loss": existence_loss,
            "endpoint_loss": endpoint_loss,
            "direction_loss": direction_loss,
            "existence_targets": existence_targets,
            "existence_weights": existence_weights,
            "matched_count": torch.tensor(
                matched_count, dtype=torch.int64, device=logits.device),
            "matches": matches,
        }
