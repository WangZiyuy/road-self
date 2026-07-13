import argparse
import json
from pathlib import Path

from PIL import Image


def main():
    parser = argparse.ArgumentParser(
        description="Prepare a small xian image for VecRoad_self inference."
    )
    parser.add_argument("image", help="Path to the source xian image.")
    parser.add_argument("--region", default="xian", help="Region name used by VecRoad.")
    parser.add_argument("--data-root", default="data_self", help="VecRoad_self data root.")
    parser.add_argument("--canvas-size", type=int, default=8192, help="Padded test image size.")
    parser.add_argument("--tile-size", type=int, default=4096, help="Tile size expected by VecRoad.")
    parser.add_argument("--lat-min", type=float, default=34.22484722131834)
    parser.add_argument("--lon-min", type=float, default=108.94460164474442)
    parser.add_argument("--lat-max", type=float, default=34.24707831919142)
    parser.add_argument("--lon-max", type=float, default=108.9677436888106)
    args = parser.parse_args()

    source = Path(args.image)
    data_root = Path(args.data_root)
    full_dir = data_root / "input" / "imagery_8192"
    tile_dir = data_root / "input" / "imagery"
    region_dir = data_root / "input" / "regions"

    full_dir.mkdir(parents=True, exist_ok=True)
    tile_dir.mkdir(parents=True, exist_ok=True)
    region_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(source).convert("RGB")
    width, height = image.size
    if width > args.canvas_size or height > args.canvas_size:
        raise ValueError(
            f"Input image is {width}x{height}, larger than canvas "
            f"{args.canvas_size}x{args.canvas_size}."
        )

    canvas = Image.new("RGB", (args.canvas_size, args.canvas_size), (0, 0, 0))
    canvas.paste(image, (0, 0))
    canvas.save(full_dir / f"{args.region}.png")

    for x in range(args.canvas_size // args.tile_size):
        for y in range(args.canvas_size // args.tile_size):
            left = x * args.tile_size
            upper = y * args.tile_size
            tile = canvas.crop((left, upper, left + args.tile_size, upper + args.tile_size))
            tile.save(tile_dir / f"{args.region}_{x}_{y}.png")

    region_file = region_dir / f"{args.region}_regions.txt"
    region_file.write_text(f"{args.region} 0 0\n", encoding="utf-8")

    metadata = {
        "region": args.region,
        "source": str(source),
        "original_size": [width, height],
        "bbox_gcj02": {
            "lat_min": args.lat_min,
            "lon_min": args.lon_min,
            "lat_max": args.lat_max,
            "lon_max": args.lon_max,
        },
        "canvas_size": args.canvas_size,
        "tile_size": args.tile_size,
        "full_image": str(full_dir / f"{args.region}.png"),
        "region_file": str(region_file),
    }
    (region_dir / f"{args.region}_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
