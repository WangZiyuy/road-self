"""Fail-fast contract checks for paired aerial and trajectory rasters.

This module is deliberately independent of the model and trajectory-sequence
loaders.  It deals only with files and machine-readable metadata so Stage S1
can establish (or block) the raster contract before any model integration.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set

import numpy as np
from PIL import Image


PAIR_MANIFEST_SCHEMA_VERSION = "1.0.0"
AXIS_ORDER_CONTRACT = {
    "disk": "image_width_height",
    "decoded_array": "height_width_channels",
    "legacy_tileloader_internal": "width_height_channels",
    "legacy_tileloader_transform": "swapaxes(0,1)",
}


@dataclass(frozen=True)
class ContractIssue:
    """One actionable contract violation."""

    code: str
    location: str
    message: str
    severity: str = "ERROR"

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_png(path: Path) -> Dict[str, Any]:
    """Return deterministic raster facts without retaining image bytes."""

    with Image.open(path) as image:
        array = np.asarray(image)
        mode = image.mode
        width, height = image.size
        metadata_keys = sorted(str(key) for key in image.info)

    channels = 1 if array.ndim == 2 else int(array.shape[2])
    values = counts = None
    if channels == 1:
        values, counts = np.unique(array, return_counts=True)
    return {
        "exists": True,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "mode": mode,
        "shape_hwc": [height, width, channels],
        "channels": channels,
        "dtype": str(array.dtype),
        "value_range": [int(array.min()), int(array.max())],
        "unique_value_count": int(values.size) if values is not None else None,
        "unique_value_counts": (
            {str(int(value)): int(count) for value, count in zip(values, counts)}
            if values is not None and counts is not None
            else None
        ),
        "embedded_metadata_keys": metadata_keys,
    }


def _issue(
    issues: List[ContractIssue], code: str, location: str, message: str
) -> None:
    issues.append(ContractIssue(code=code, location=location, message=message))


def _required_keys(
    value: Mapping[str, Any],
    keys: Iterable[str],
    location: str,
    issues: List[ContractIssue],
) -> None:
    for key in keys:
        if key not in value:
            _issue(
                issues,
                "SCHEMA_REQUIRED_FIELD_MISSING",
                f"{location}.{key}",
                "Required field is absent.",
            )


def _as_set(values: Optional[Sequence[Any]]) -> Set[int]:
    if values is None:
        return set()
    return {int(value) for value in values}


def validate_pair_manifest(
    manifest: Mapping[str, Any], source_root: Optional[Path] = None
) -> List[ContractIssue]:
    """Validate a manifest, returning every blocking issue found.

    ``source_root`` is optional so committed artifacts remain portable.  When
    supplied, existence and checksums are verified against that read-only
    source tree in addition to the recorded facts.
    """

    issues: List[ContractIssue] = []
    _required_keys(
        manifest,
        [
            "schema_version",
            "manifest_kind",
            "entries",
            "split_separation",
        ],
        "$",
        issues,
    )
    if manifest.get("schema_version") != PAIR_MANIFEST_SCHEMA_VERSION:
        _issue(
            issues,
            "SCHEMA_VERSION_UNSUPPORTED",
            "$.schema_version",
            f"Expected {PAIR_MANIFEST_SCHEMA_VERSION!r}.",
        )

    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        _issue(
            issues,
            "SCHEMA_ENTRIES_EMPTY",
            "$.entries",
            "At least one region/split entry is required.",
        )
        return issues

    required_entry_keys = [
        "region",
        "split",
        "aerial_path",
        "raster_path",
        "shape",
        "channels",
        "dtype",
        "value_range",
        "allowed_values",
        "normalization",
        "coordinate_reference",
        "pixel_origin",
        "axis_order",
        "y_direction",
        "source_lineage",
        "checksum",
        "grid_registration_proven",
        "registration_evidence",
        "value_semantics",
        "source_identity",
        "aerial_sha256",
        "raster_sha256",
        "aerial_shape_hwc",
        "raster_shape_hwc",
        "aerial_dtype",
        "raster_dtype",
        "raster_unique_values",
        "crs",
        "geotransform",
        "pixel_size",
        "coordinate_axis_contract",
        "shape_match",
        "status",
    ]

    split_identities: Dict[str, Set[str]] = {"train": set(), "validation": set(), "test": set()}
    for index, entry in enumerate(entries):
        location = f"$.entries[{index}]"
        if not isinstance(entry, Mapping):
            _issue(issues, "SCHEMA_ENTRY_NOT_OBJECT", location, "Entry must be an object.")
            continue
        _required_keys(entry, required_entry_keys, location, issues)

        split = entry.get("split")
        if split not in split_identities:
            _issue(
                issues,
                "INVALID_SPLIT",
                f"{location}.split",
                "Split must be train, validation, or test.",
            )
        identity = entry.get("source_identity")
        if split in split_identities and isinstance(identity, str) and identity:
            split_identities[split].add(identity)

        region = entry.get("region")
        if isinstance(region, str) and region:
            for side in ("aerial", "raster"):
                recorded_path = entry.get(f"{side}_path")
                if isinstance(recorded_path, str):
                    stem = Path(recorded_path).stem
                    if stem != region and not stem.startswith(f"{region}_"):
                        _issue(
                            issues,
                            "REGION_NAME_MISMATCH",
                            f"{location}.{side}_path",
                            f"Filename stem {stem!r} does not identify region {region!r}.",
                        )

        for side in ("aerial", "raster"):
            path_key = f"{side}_path"
            facts = entry.get(side)
            recorded_path = entry.get(path_key)
            if not isinstance(facts, Mapping) or not facts.get("exists", False):
                _issue(
                    issues,
                    "MISSING_PAIR_MEMBER",
                    f"{location}.{path_key}",
                    (
                        "Required aerial raster is missing."
                        if side == "aerial"
                        else "Required trajectory raster is missing."
                    ),
                )
            if source_root is not None and isinstance(recorded_path, str):
                resolved = source_root / Path(recorded_path)
                if not resolved.is_file():
                    _issue(
                        issues,
                        "SOURCE_FILE_MISSING",
                        f"{location}.{path_key}",
                        "Recorded source file does not exist below source_root.",
                    )
                else:
                    expected = (entry.get("checksum") or {}).get(side)
                    actual = sha256_file(resolved)
                    if expected != actual:
                        _issue(
                            issues,
                            "CHECKSUM_MISMATCH",
                            f"{location}.checksum.{side}",
                            "Recorded checksum differs from source bytes.",
                        )
                    flat_expected = entry.get(f"{side}_sha256")
                    if flat_expected != actual:
                        _issue(
                            issues,
                            "CHECKSUM_MISMATCH",
                            f"{location}.{side}_sha256",
                            "Flat recorded checksum differs from source bytes.",
                        )

        shapes = entry.get("shape") or {}
        aerial_shape = shapes.get("aerial")
        raster_shape = shapes.get("raster")
        if (
            aerial_shape is not None
            and raster_shape is not None
            and list(aerial_shape[:2]) != list(raster_shape[:2])
        ):
            _issue(
                issues,
                "GRID_SHAPE_MISMATCH",
                f"{location}.shape",
                "Aerial and trajectory raster spatial shapes are not identical.",
            )

        channels = entry.get("channels") or {}
        if channels.get("aerial") not in (None, 3):
            _issue(
                issues,
                "AERIAL_CHANNEL_COUNT_INVALID",
                f"{location}.channels.aerial",
                "Aerial raster must have three channels for this contract.",
            )
        if channels.get("raster") not in (None, 1):
            _issue(
                issues,
                "RASTER_CHANNEL_COUNT_INVALID",
                f"{location}.channels.raster",
                "Trajectory raster must be single-channel.",
            )

        dtypes = entry.get("dtype") or {}
        for side in ("aerial", "raster"):
            if dtypes.get(side) not in (None, "uint8"):
                _issue(
                    issues,
                    "DTYPE_INVALID",
                    f"{location}.dtype.{side}",
                    f"{side} dtype must be uint8 before normalization.",
                )

        value_ranges = entry.get("value_range") or {}
        for side in ("aerial", "raster"):
            value_range = value_ranges.get(side)
            if value_range is not None and (
                not isinstance(value_range, Sequence)
                or len(value_range) != 2
                or value_range[0] < 0
                or value_range[1] > 255
                or value_range[0] > value_range[1]
            ):
                _issue(
                    issues,
                    "VALUE_RANGE_INVALID",
                    f"{location}.value_range.{side}",
                    "Raw uint8 value range must be ordered and within [0,255].",
                )

        allowed_values = _as_set((entry.get("allowed_values") or {}).get("raster"))
        observed_values = _as_set(
            ((entry.get("raster") or {}).get("unique_value_counts") or {}).keys()
        )
        if allowed_values and observed_values and not observed_values.issubset(allowed_values):
            unexpected = sorted(observed_values - allowed_values)
            _issue(
                issues,
                "RASTER_VALUE_NOT_ALLOWED",
                f"{location}.allowed_values.raster",
                f"Observed disallowed values: {unexpected}.",
            )

        semantics = entry.get("value_semantics") or {}
        if semantics.get("status") != "KNOWN_AND_SOURCE_BACKED":
            _issue(
                issues,
                "VALUE_SEMANTICS_UNKNOWN",
                f"{location}.value_semantics",
                "Raster classes/levels lack source-backed semantic assignments.",
            )

        if entry.get("axis_order") != AXIS_ORDER_CONTRACT:
            _issue(
                issues,
                "AXIS_ORDER_CONTRACT_INVALID",
                f"{location}.axis_order",
                "Axis order must describe disk, decoded-array, and loader-internal layouts.",
            )
        if entry.get("pixel_origin") not in ("upper_left",):
            _issue(
                issues,
                "PIXEL_ORIGIN_UNKNOWN",
                f"{location}.pixel_origin",
                "Pixel origin must be proven as upper_left for this loader contract.",
            )
        if entry.get("y_direction") not in ("down",):
            _issue(
                issues,
                "Y_DIRECTION_UNKNOWN",
                f"{location}.y_direction",
                "Pixel y direction must be proven as down for this loader contract.",
            )
        evidence = entry.get("registration_evidence")
        if entry.get("grid_registration_proven") is not True or not isinstance(evidence, list) or not evidence:
            _issue(
                issues,
                "GRID_REGISTRATION_UNPROVEN",
                f"{location}.registration_evidence",
                "Registration needs non-empty source-backed transform evidence; equal shape is insufficient.",
            )

    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = sorted(split_identities[left] & split_identities[right])
        if overlap:
            _issue(
                issues,
                "SPLIT_SOURCE_LEAKAGE",
                "$.split_separation",
                f"{left}/{right} share source identities: {overlap}.",
            )

    split_separation = manifest.get("split_separation") or {}
    if split_separation.get("status") != "PROVEN_NO_LEAKAGE":
        _issue(
            issues,
            "SPLIT_SEPARATION_UNPROVEN",
            "$.split_separation",
            "Train/validation/test source separation has not been proven.",
        )
    return issues


def verify_reproducible_files(first: Path, second: Path) -> Dict[str, Any]:
    """Compare two independently written outputs byte-for-byte."""

    first_hash = sha256_file(first)
    second_hash = sha256_file(second)
    return {
        "first_sha256": first_hash,
        "second_sha256": second_hash,
        "byte_identical": first_hash == second_hash and first.read_bytes() == second.read_bytes(),
    }
