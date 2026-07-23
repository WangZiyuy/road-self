"""Visualize Stage 1B trajectory fragments around one global pixel position."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.structured_trajectory_store import open_structured_trajectory_store


def _meta_image_size(meta: Dict[str, Any]) -> Tuple[int, int]:
    image_size = meta.get("image_size")
    if not isinstance(image_size, (list, tuple)) or len(image_size) != 2:
        raise ValueError("cache meta.json must contain image_size [width, height]")
    width, height = map(int, image_size)
    if width <= 0 or height <= 0:
        raise ValueError("cache image_size must be positive")
    return width, height


def _uniform_pixel_value(image: Image.Image) -> Optional[Tuple[int, ...]]:
    if image.width == 0 or image.height == 0:
        return None
    extrema = image.getextrema()
    if extrema and isinstance(extrema[0], int):
        extrema = (extrema,)
    if not all(lower == upper for lower, upper in extrema):
        return None
    return tuple(int(lower) for lower, _ in extrema)


def verify_background_alignment(
    image: Image.Image,
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    """Verify exact-size or demonstrable top-left padding alignment."""

    original_width, original_height = _meta_image_size(meta)
    actual_width, actual_height = image.size
    if image.size == (original_width, original_height):
        return {
            "status": "aligned_original_size",
            "original_size": [original_width, original_height],
            "background_size": [actual_width, actual_height],
            "coordinate_offset_xy": [0, 0],
            "coordinate_scale_xy": [1.0, 1.0],
        }

    canvas_size = meta.get("canvas_size")
    if isinstance(canvas_size, int):
        expected_canvas = (canvas_size, canvas_size)
    elif isinstance(canvas_size, (list, tuple)) and len(canvas_size) == 2:
        expected_canvas = tuple(map(int, canvas_size))
    else:
        raise ValueError(
            "background size {} differs from cache image_size {}, and cache "
            "metadata does not describe a canvas".format(
                image.size, (original_width, original_height))
        )
    if image.size != expected_canvas:
        raise ValueError(
            "background size {} does not match metadata canvas {}".format(
                image.size, expected_canvas))
    if (
        actual_width < original_width
        or actual_height < original_height
    ):
        raise ValueError(
            "metadata canvas is smaller than the original image extent")

    padding_regions = []
    if actual_width > original_width:
        padding_regions.append(
            image.crop(
                (original_width, 0, actual_width, original_height)
            )
        )
    if actual_height > original_height:
        padding_regions.append(
            image.crop(
                (0, original_height, actual_width, actual_height)
            )
        )
    padding_values = [
        _uniform_pixel_value(region) for region in padding_regions
    ]
    if (
        not padding_values
        or any(value is None for value in padding_values)
        or len(set(padding_values)) != 1
    ):
        raise ValueError(
            "larger background is not demonstrably uniform right/bottom "
            "padding; refusing to guess a coordinate transform"
        )
    return {
        "status": "aligned_top_left_padding",
        "original_size": [original_width, original_height],
        "background_size": [actual_width, actual_height],
        "coordinate_offset_xy": [0, 0],
        "coordinate_scale_xy": [1.0, 1.0],
        "padding_pixel": list(padding_values[0]),
    }


def _display_crop(
    image: Image.Image,
    center_xy: Sequence[float],
    window_size: float,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    margin = max(32.0, window_size * 0.25)
    half_extent = window_size / 2.0 + margin
    left = max(0, int(math.floor(center_xy[0] - half_extent)))
    top = max(0, int(math.floor(center_xy[1] - half_extent)))
    right = min(
        image.width, int(math.ceil(center_xy[0] + half_extent)))
    bottom = min(
        image.height, int(math.ceil(center_xy[1] + half_extent)))
    if left >= right or top >= bottom:
        raise ValueError("requested display area is outside the background")
    return np.asarray(image.crop((left, top, right, bottom))), (
        left,
        top,
        right,
        bottom,
    )


def visualize_fragments(
    cache_dir: Path,
    center_xy: Sequence[float],
    window_size: float,
    context_points: int,
    output_path: Path,
    background_image: Optional[Path] = None,
    max_time_gap_seconds: Optional[float] = None,
    max_spatial_gap_pixels: Optional[float] = None,
    reference_segments_xy: Optional[Sequence[np.ndarray]] = None,
    max_fragments: Optional[int] = None,
) -> Dict[str, Any]:
    store = open_structured_trajectory_store(str(cache_dir))
    fragments = store.query_trajectory_fragments(
        center_xy=center_xy,
        window_size=window_size,
        context_points=context_points,
        max_time_gap_seconds=max_time_gap_seconds,
        max_spatial_gap_pixels=max_spatial_gap_pixels,
    )
    total_fragment_count = len(fragments)
    if max_fragments is not None:
        from utils.trajectory_batch import build_trajectory_batch

        selection = build_trajectory_batch(
            [fragments],
            center_xy=[center_xy],
            window_size=window_size,
            max_fragments=max_fragments,
        )
        selected_indices = selection[
            "source_fragment_indices"
        ][0][selection["fragment_mask"][0]].tolist()
        fragments = [
            fragments[int(fragment_index)]
            for fragment_index in selected_indices
        ]
    center_x, center_y = map(float, center_xy)
    half_window = float(window_size) / 2.0
    display_margin = max(32.0, window_size * 0.25)
    display_bounds = (
        center_x - half_window - display_margin,
        center_y - half_window - display_margin,
        center_x + half_window + display_margin,
        center_y + half_window + display_margin,
    )

    figure, axis = plt.subplots(figsize=(9, 9))
    alignment = None
    if background_image is not None:
        with Image.open(background_image) as image_file:
            image = image_file.convert("RGB")
            alignment = verify_background_alignment(image, store.meta)
            crop, (left, top, right, bottom) = _display_crop(
                image, center_xy, window_size)
        axis.imshow(
            crop,
            extent=(left, right, bottom, top),
            origin="upper",
        )

    reference_segments = (
        [] if reference_segments_xy is None else reference_segments_xy
    )
    reference_segment_count = 0
    for segment in reference_segments:
        segment_array = np.asarray(segment, dtype=np.float64)
        if segment_array.shape != (2, 2):
            raise ValueError(
                "each reference segment must have shape [2, 2]")
        axis.plot(
            segment_array[:, 0],
            segment_array[:, 1],
            color="cyan",
            linewidth=3.0,
            linestyle="--",
            alpha=0.95,
            zorder=3,
            label=(
                "incident GT edge"
                if reference_segment_count == 0
                else None
            ),
        )
        reference_segment_count += 1

    track_ids = sorted({fragment.track_index for fragment in fragments})
    color_map = plt.get_cmap("tab20")
    color_by_track = {
        track_index: color_map(index % 20)
        for index, track_index in enumerate(track_ids)
    }
    labeled_tracks = set()
    for fragment in fragments:
        points = fragment.points_global_xy
        color = color_by_track[fragment.track_index]
        label = None
        if fragment.track_index not in labeled_tracks:
            label = "track {}".format(fragment.track_index)
            labeled_tracks.add(fragment.track_index)
        axis.plot(
            points[:, 0],
            points[:, 1],
            color=color,
            linewidth=2.0,
            alpha=0.9,
            label=label,
            zorder=4,
        )
        axis.scatter(
            points[0, 0],
            points[0, 1],
            color=[color],
            marker="o",
            s=28,
            edgecolors="white",
            linewidths=0.5,
            zorder=4,
        )
        axis.scatter(
            points[-1, 0],
            points[-1, 1],
            color=[color],
            marker="s",
            s=28,
            edgecolors="black",
            linewidths=0.5,
            zorder=4,
        )

    axis.add_patch(
        Rectangle(
            (center_x - half_window, center_y - half_window),
            window_size,
            window_size,
            fill=False,
            edgecolor="yellow",
            linewidth=2.0,
            linestyle="--",
            zorder=5,
        )
    )
    axis.scatter(
        [center_x],
        [center_y],
        color="red",
        marker="*",
        s=160,
        edgecolors="white",
        linewidths=0.8,
        label="current node",
        zorder=6,
    )
    axis.set_xlim(display_bounds[0], display_bounds[2])
    axis.set_ylim(display_bounds[3], display_bounds[1])
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("global pixel x")
    axis.set_ylabel("global pixel y")
    axis.set_title(
        "{} tracks / {} of {} fragments around ({:.1f}, {:.1f})".format(
            len(track_ids),
            len(fragments),
            total_fragment_count,
            center_x,
            center_y,
        )
    )
    if len(track_ids) <= 20:
        axis.legend(loc="upper right", fontsize=8)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(str(output_path), dpi=160, bbox_inches="tight")
    plt.close(figure)

    return {
        "cache_dir": str(cache_dir.resolve()),
        "center_xy": [center_x, center_y],
        "window_size": float(window_size),
        "context_points": int(context_points),
        "trajectory_count": len(track_ids),
        "fragment_count": len(fragments),
        "total_fragment_count": total_fragment_count,
        "kept_fragment_count": len(fragments),
        "truncated_fragment_count": (
            total_fragment_count - len(fragments)),
        "max_fragments": max_fragments,
        "track_indices_preview": track_ids[:50],
        "track_indices_truncated": len(track_ids) > 50,
        "background_alignment": alignment,
        "reference_segment_count": reference_segment_count,
        "output_path": str(output_path.resolve()),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize exact Stage 1B local trajectory fragments.")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--center-x", type=float, required=True)
    parser.add_argument("--center-y", type=float, required=True)
    parser.add_argument("--window-size", type=float, default=256.0)
    parser.add_argument("--context-points", type=int, default=2)
    parser.add_argument("--max-time-gap-seconds", type=float, default=None)
    parser.add_argument("--max-spatial-gap-pixels", type=float, default=None)
    parser.add_argument("--max-fragments", type=int, default=None)
    parser.add_argument("--background-image", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = visualize_fragments(
        cache_dir=args.cache_dir,
        center_xy=(args.center_x, args.center_y),
        window_size=args.window_size,
        context_points=args.context_points,
        output_path=args.output,
        background_image=args.background_image,
        max_time_gap_seconds=args.max_time_gap_seconds,
        max_spatial_gap_pixels=args.max_spatial_gap_pixels,
        max_fragments=args.max_fragments,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
