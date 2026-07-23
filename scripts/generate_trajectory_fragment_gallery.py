"""Select and render diverse real Stage 1B trajectory-fragment cases."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.visualize_trajectory_fragments import visualize_fragments
from utils.structured_trajectory_store import open_structured_trajectory_store


def _fragment_metrics(
    fragments,
    center_xy: Sequence[float],
    window_size: float,
) -> Dict[str, Any]:
    center_x, center_y = map(float, center_xy)
    half_window = window_size / 2.0
    bounds = (
        center_x - half_window,
        center_y - half_window,
        center_x + half_window,
        center_y + half_window,
    )
    track_counts = Counter(fragment.track_index for fragment in fragments)
    segment_only_count = 0
    point_counts = []
    for fragment in fragments:
        points = fragment.points_global_xy
        has_sampled_point_inside = bool(
            (
                (points[:, 0] >= bounds[0])
                & (points[:, 0] <= bounds[2])
                & (points[:, 1] >= bounds[1])
                & (points[:, 1] <= bounds[3])
            ).any()
        )
        if not has_sampled_point_inside:
            segment_only_count += 1
        point_counts.append(len(fragment))
    return {
        "trajectory_count": len(track_counts),
        "fragment_count": len(fragments),
        "multi_visit_track_count": sum(
            count > 1 for count in track_counts.values()),
        "segment_only_fragment_count": segment_only_count,
        "mean_fragment_point_count": (
            float(np.mean(point_counts)) if point_counts else 0.0
        ),
        "max_fragment_point_count": max(point_counts, default=0),
    }


def _sample_records(
    store,
    sample_count: int,
    seed: int,
    window_size: float,
    context_points: int,
    max_time_gap_seconds: Optional[float],
    max_spatial_gap_pixels: Optional[float],
) -> List[Dict[str, Any]]:
    image_width, image_height = map(int, store.meta["image_size"])
    half_window = window_size / 2.0
    if image_width <= window_size or image_height <= window_size:
        raise ValueError("window_size must be smaller than the cache image")
    random = np.random.default_rng(seed)
    centers = np.column_stack(
        (
            random.uniform(
                half_window, image_width - half_window, sample_count),
            random.uniform(
                half_window, image_height - half_window, sample_count),
        )
    )

    records = []
    for sample_index, center_xy in enumerate(centers):
        filtered = store.query_trajectory_fragments(
            center_xy=center_xy,
            window_size=window_size,
            context_points=context_points,
            max_time_gap_seconds=max_time_gap_seconds,
            max_spatial_gap_pixels=max_spatial_gap_pixels,
        )
        no_gap = store.query_trajectory_fragments(
            center_xy=center_xy,
            window_size=window_size,
            context_points=context_points,
        )
        filtered_metrics = _fragment_metrics(
            filtered, center_xy, window_size)
        no_gap_metrics = _fragment_metrics(
            no_gap, center_xy, window_size)
        records.append(
            {
                "sample_index": sample_index,
                "center_xy": [float(center_xy[0]), float(center_xy[1])],
                "filtered": filtered_metrics,
                "no_gap": no_gap_metrics,
                "fragment_count_delta_no_gap_minus_filtered": (
                    no_gap_metrics["fragment_count"]
                    - filtered_metrics["fragment_count"]
                ),
                "trajectory_count_removed_by_gap": (
                    no_gap_metrics["trajectory_count"]
                    - filtered_metrics["trajectory_count"]
                ),
            }
        )
    return records


def _select_unique_cases(
    records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    selected = []
    used_sample_indices = set()

    def add_case(label, ordered_records, predicate=lambda record: True):
        for record in ordered_records:
            if (
                record["sample_index"] not in used_sample_indices
                and predicate(record)
            ):
                selected_record = dict(record)
                selected_record["case_label"] = label
                selected.append(selected_record)
                used_sample_indices.add(record["sample_index"])
                return

    by_fragment_count = sorted(
        records, key=lambda record: record["filtered"]["fragment_count"])
    add_case(
        "empty_or_minimum",
        by_fragment_count,
    )
    add_case(
        "sparse_nonzero",
        by_fragment_count,
        lambda record: record["filtered"]["fragment_count"] > 0,
    )
    nonzero = [
        record for record in records
        if record["filtered"]["fragment_count"] > 0
    ]
    if nonzero:
        for label, quantile in (
            ("lower_quartile", 0.25),
            ("median_density", 0.50),
            ("upper_quartile", 0.75),
            ("high_density", 0.90),
        ):
            target = float(np.quantile(
                [
                    record["filtered"]["fragment_count"]
                    for record in nonzero
                ],
                quantile,
            ))
            add_case(
                label,
                sorted(
                    nonzero,
                    key=lambda record: abs(
                        record["filtered"]["fragment_count"] - target),
                ),
            )
    add_case(
        "maximum_density",
        sorted(
            records,
            key=lambda record: record["filtered"]["fragment_count"],
            reverse=True,
        ),
    )
    add_case(
        "multiple_window_visits",
        sorted(
            records,
            key=lambda record: (
                record["filtered"]["multi_visit_track_count"],
                record["filtered"]["fragment_count"],
            ),
            reverse=True,
        ),
        lambda record: record["filtered"]["multi_visit_track_count"] > 0,
    )
    add_case(
        "segment_only_crossings",
        sorted(
            records,
            key=lambda record: (
                record["filtered"]["segment_only_fragment_count"],
                record["filtered"]["fragment_count"],
            ),
            reverse=True,
        ),
        lambda record: (
            record["filtered"]["segment_only_fragment_count"] > 0
        ),
    )
    add_case(
        "gap_sensitive",
        sorted(
            records,
            key=lambda record: (
                record["trajectory_count_removed_by_gap"],
                abs(
                    record[
                        "fragment_count_delta_no_gap_minus_filtered"
                    ]
                ),
            ),
            reverse=True,
        ),
        lambda record: (
            record["trajectory_count_removed_by_gap"] > 0
            or record["fragment_count_delta_no_gap_minus_filtered"] != 0
        ),
    )
    return selected


def _captioned_thumbnail(
    image_path: Path,
    caption: str,
    width: int = 600,
) -> Image.Image:
    with Image.open(image_path) as image_file:
        image = image_file.convert("RGB")
        height = max(1, round(image.height * width / image.width))
        image.thumbnail((width, height), Image.Resampling.LANCZOS)
        canvas = Image.new(
            "RGB", (width, image.height + 36), (255, 255, 255))
        canvas.paste(image, ((width - image.width) // 2, 36))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 10), caption, fill=(0, 0, 0))
    return canvas


def _write_contact_sheet(
    rendered_cases: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    thumbnails = []
    for case in rendered_cases:
        metrics = case["filtered"]
        caption = (
            "{label} | tracks={tracks} fragments={fragments} "
            "multi={multi} segment-only={segment_only}"
        ).format(
            label=case["case_label"],
            tracks=metrics["trajectory_count"],
            fragments=metrics["fragment_count"],
            multi=metrics["multi_visit_track_count"],
            segment_only=metrics["segment_only_fragment_count"],
        )
        thumbnails.append(
            _captioned_thumbnail(Path(case["filtered_image"]), caption))
    if not thumbnails:
        raise ValueError("no cases were rendered")
    column_count = 2
    row_count = math.ceil(len(thumbnails) / column_count)
    cell_width = max(image.width for image in thumbnails)
    cell_height = max(image.height for image in thumbnails)
    sheet = Image.new(
        "RGB",
        (column_count * cell_width, row_count * cell_height),
        (230, 230, 230),
    )
    for index, image in enumerate(thumbnails):
        column = index % column_count
        row = index // column_count
        sheet.paste(
            image,
            (column * cell_width, row * cell_height),
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def generate_gallery(
    cache_dir: Path,
    background_image: Path,
    output_dir: Path,
    sample_count: int,
    seed: int,
    window_size: float,
    context_points: int,
    max_time_gap_seconds: Optional[float],
    max_spatial_gap_pixels: Optional[float],
) -> Dict[str, Any]:
    store = open_structured_trajectory_store(str(cache_dir))
    records = _sample_records(
        store=store,
        sample_count=sample_count,
        seed=seed,
        window_size=window_size,
        context_points=context_points,
        max_time_gap_seconds=max_time_gap_seconds,
        max_spatial_gap_pixels=max_spatial_gap_pixels,
    )
    selected = _select_unique_cases(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered_cases = []
    for case_index, case in enumerate(selected):
        center_x, center_y = case["center_xy"]
        stem = "{:02d}_{}".format(case_index + 1, case["case_label"])
        filtered_path = output_dir / "{}_filtered.png".format(stem)
        visualize_fragments(
            cache_dir=cache_dir,
            center_xy=(center_x, center_y),
            window_size=window_size,
            context_points=context_points,
            output_path=filtered_path,
            background_image=background_image,
            max_time_gap_seconds=max_time_gap_seconds,
            max_spatial_gap_pixels=max_spatial_gap_pixels,
        )
        rendered = dict(case)
        rendered["filtered_image"] = str(filtered_path.resolve())
        if case["case_label"] == "gap_sensitive":
            no_gap_path = output_dir / "{}_no_gap.png".format(stem)
            visualize_fragments(
                cache_dir=cache_dir,
                center_xy=(center_x, center_y),
                window_size=window_size,
                context_points=context_points,
                output_path=no_gap_path,
                background_image=background_image,
            )
            rendered["no_gap_image"] = str(no_gap_path.resolve())
        rendered_cases.append(rendered)

    contact_sheet = output_dir / "contact_sheet.png"
    _write_contact_sheet(rendered_cases, contact_sheet)
    manifest = {
        "cache_dir": str(cache_dir.resolve()),
        "background_image": str(background_image.resolve()),
        "sample_count": sample_count,
        "seed": seed,
        "window_size": window_size,
        "context_points": context_points,
        "max_time_gap_seconds": max_time_gap_seconds,
        "max_spatial_gap_pixels": max_spatial_gap_pixels,
        "selected_case_count": len(rendered_cases),
        "cases": rendered_cases,
        "contact_sheet": str(contact_sheet.resolve()),
    }
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
        file.write("\n")
    manifest["manifest_path"] = str(manifest_path.resolve())
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a deterministic gallery of diverse Stage 1B cases.")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--background-image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--window-size", type=float, default=256.0)
    parser.add_argument("--context-points", type=int, default=2)
    parser.add_argument("--max-time-gap-seconds", type=float, default=None)
    parser.add_argument("--max-spatial-gap-pixels", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.sample_count <= 0:
        raise ValueError("sample-count must be positive")
    report = generate_gallery(
        cache_dir=args.cache_dir,
        background_image=args.background_image,
        output_dir=args.output_dir,
        sample_count=args.sample_count,
        seed=args.seed,
        window_size=args.window_size,
        context_points=args.context_points,
        max_time_gap_seconds=args.max_time_gap_seconds,
        max_spatial_gap_pixels=args.max_spatial_gap_pixels,
    )
    summary = {
        "selected_case_count": report["selected_case_count"],
        "contact_sheet": report["contact_sheet"],
        "manifest_path": report["manifest_path"],
        "cases": [
            {
                "case_label": case["case_label"],
                "center_xy": case["center_xy"],
                "filtered": case["filtered"],
                "no_gap": case["no_gap"],
                "fragment_count_delta_no_gap_minus_filtered": case[
                    "fragment_count_delta_no_gap_minus_filtered"
                ],
                "trajectory_count_removed_by_gap": case[
                    "trajectory_count_removed_by_gap"
                ],
                "filtered_image": case["filtered_image"],
                "no_gap_image": case.get("no_gap_image"),
            }
            for case in report["cases"]
        ],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
