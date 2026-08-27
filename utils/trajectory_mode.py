"""Central trajectory-mode resolution and stage-0 gating helpers."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping
from typing import Any


TRAJ_MODE_NONE = "none"
TRAJ_MODE_LEGACY = "legacy_current"
TRAJ_MODE_RASTER_SEG_ONLY = "raster_seg_only"
VALID_TRAJ_MODES = frozenset({
    TRAJ_MODE_NONE,
    TRAJ_MODE_LEGACY,
    TRAJ_MODE_RASTER_SEG_ONLY,
})

_MISSING = object()
_WARNED_CONFLICTS: set[tuple[str, bool]] = set()


def _get_value(container: Any, key: str, default: Any = _MISSING) -> Any:
    if container is None:
        return default
    if isinstance(container, Mapping):
        return container.get(key, default)
    return getattr(container, key, default)


def validate_trajectory_mode(mode: str) -> str:
    """Normalize and validate one of the trajectory modes implemented today."""
    if not isinstance(mode, str):
        raise ValueError(
            "TRAJ.MODE must be a string; supported modes are: "
            + ", ".join(sorted(VALID_TRAJ_MODES))
        )
    normalized = mode.strip().lower()
    if normalized not in VALID_TRAJ_MODES:
        raise ValueError(
            "Unknown TRAJ.MODE={!r}; supported modes are: {}. "
            "structured_all and branch_slot are reserved for later stages and "
            "are not implemented in stage 0.".format(
                mode, ", ".join(sorted(VALID_TRAJ_MODES))
            )
        )
    return normalized


def resolve_trajectory_mode(cfg: Any) -> str:
    """Resolve new and legacy configuration with ``TRAJ.MODE`` taking priority.

    Older configs map ``TRAIN.USE_TRAJ=False`` to ``none`` and ``True`` to
    ``legacy_current``. If neither field exists, the original image-only mode is
    the conservative default.
    """
    train_cfg = _get_value(cfg, "TRAIN", None)
    legacy_value = _get_value(train_cfg, "USE_TRAJ", _MISSING)
    legacy_mode = (
        TRAJ_MODE_LEGACY
        if legacy_value is not _MISSING and bool(legacy_value)
        else TRAJ_MODE_NONE
    )

    traj_cfg = _get_value(cfg, "TRAJ", None)
    configured_mode = _get_value(traj_cfg, "MODE", _MISSING)
    if configured_mode is _MISSING:
        return legacy_mode

    mode = validate_trajectory_mode(configured_mode)
    if (
        legacy_value is not _MISSING
        and mode != TRAJ_MODE_RASTER_SEG_ONLY
        and mode != legacy_mode
    ):
        conflict = (mode, bool(legacy_value))
        if conflict not in _WARNED_CONFLICTS:
            warnings.warn(
                "TRAJ.MODE={!r} conflicts with TRAIN.USE_TRAJ={!r}; "
                "TRAJ.MODE takes precedence.".format(mode, legacy_value),
                RuntimeWarning,
                stacklevel=2,
            )
            _WARNED_CONFLICTS.add(conflict)
    return mode


def trajectory_enabled(cfg: Any) -> bool:
    return resolve_trajectory_mode(cfg) != TRAJ_MODE_NONE


def trajectory_uses_sequence(mode: str) -> bool:
    """Return whether the legacy coordinate-sequence branch is required."""
    return validate_trajectory_mode(mode) == TRAJ_MODE_LEGACY


def trajectory_uses_raster(mode: str) -> bool:
    """Return whether a spatial trajectory raster must be loaded."""
    return validate_trajectory_mode(mode) in {
        TRAJ_MODE_LEGACY,
        TRAJ_MODE_RASTER_SEG_ONLY,
    }


def validate_trajectory_model_compatibility(
    cfg: Any, mode: str | None = None
) -> None:
    """Reject a trajectory-dependent segmentation model in image-only mode."""
    resolved_mode = resolve_trajectory_mode(cfg) if mode is None else validate_trajectory_mode(mode)
    train_cfg = _get_value(cfg, "TRAIN", None)
    model_name = _get_value(train_cfg, "MODEL", _MISSING)
    if (
        resolved_mode == TRAJ_MODE_NONE
        and model_name is not _MISSING
        and str(model_name).lower() != "origin"
    ):
        raise ValueError(
            "TRAJ.MODE='none' requires TRAIN.MODEL='origin'; got {!r}. "
            "The DSFNet path consumes a trajectory raster and cannot be used "
            "as the image-only baseline.".format(model_name)
        )
    if (
        resolved_mode == TRAJ_MODE_RASTER_SEG_ONLY
        and model_name is not _MISSING
        and str(model_name).lower() != "origin"
    ):
        raise ValueError(
            "TRAJ.MODE='raster_seg_only' requires TRAIN.MODEL='origin'; "
            "legacy DSFNet is not a formal raster-seg-only path. Got {!r}."
            .format(model_name)
        )

    if resolved_mode == TRAJ_MODE_RASTER_SEG_ONLY:
        traj_cfg = _get_value(cfg, "TRAJ", None)
        raster_cfg = _get_value(traj_cfg, "RASTER", None)
        semantics = _get_value(raster_cfg, "INPUT_SEMANTICS", _MISSING)
        if semantics is not _MISSING and semantics != "binary_presence":
            raise ValueError(
                "raster_seg_only supports only "
                "TRAJ.RASTER.INPUT_SEMANTICS='binary_presence'; got {!r}."
                .format(semantics)
            )
        sequence_cfg = _get_value(traj_cfg, "SEQUENCE", None)
        sequence_enabled = _get_value(sequence_cfg, "ENABLED", False)
        if bool(sequence_enabled):
            raise ValueError(
                "raster_seg_only cannot be combined with trajectory sequence "
                "loading or a sequence Transformer."
            )


def trajectory_fetch_fields(mode: str, *, include_raster: bool) -> tuple[str, ...]:
    """Return the legacy fields a caller may request for the resolved mode."""
    mode = validate_trajectory_mode(mode)
    if mode == TRAJ_MODE_NONE:
        return ()
    fields = []
    if trajectory_uses_sequence(mode):
        fields.append("valid_trajectories")
    if include_raster:
        fields[0:0] = [
            "traj_image_chw",
            "traj_image_hwc",
            "traj_valid_mask_chw",
            "traj_valid_mask_hwc",
        ]
    return tuple(fields)


def load_region_trajectory_inputs_for_mode(
    mode: str,
    region: str,
    cfg: Any,
    loader: Callable[[str, Any], tuple[Any, Any, Any, Any]],
) -> tuple[Any, Any, Any, Any]:
    """Call the legacy loader only when trajectory input is enabled.

    The empty return value matches ``Path``'s existing constructor contract and
    avoids touching raw trajectory files, prepared caches, or CUDA tensors.
    """
    mode = validate_trajectory_mode(mode)
    if not trajectory_uses_sequence(mode):
        return None, [], None, None
    return loader(region, cfg)


def prepare_trajectory_sequence_batch(
    mode: str,
    batch_trajectories: Any,
    pad_to_device: Callable[[Any], Any],
    normalize: Callable[[Any], tuple[Any, Any]],
) -> tuple[Any, Any]:
    """Prepare legacy sequence tensors, or return two ``None`` values.

    Keeping the gate here makes it straightforward to prove that stage-0
    image-only training and inference never invoke padding or normalization.
    """
    mode = validate_trajectory_mode(mode)
    if not trajectory_uses_sequence(mode):
        return None, None
    padded = pad_to_device(batch_trajectories)
    return normalize(padded)
