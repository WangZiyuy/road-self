import argparse
import json
import struct
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Point:
    x: float
    y: float


@dataclass(frozen=True)
class Edge:
    src: Point
    dst: Point


WGS84_PRJ = (
    'GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",'
    'SPHEROID["WGS_1984",6378137.0,298.257223563]],'
    'PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]]'
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert VecRoad .graph pixel coordinates to a polyline shapefile."
    )
    parser.add_argument("--graph", default="data_self/graphs/50.2047_4/post/xian.graph")
    parser.add_argument("--metadata", default="data_self/input/regions/xian_metadata.json")
    parser.add_argument("--output", default="data_self/output/xian_roads.shp")
    parser.add_argument(
        "--bounds",
        nargs=4,
        type=float,
        required=True,
        metavar=("LAT_MIN", "LON_MIN", "LAT_MAX", "LON_MAX"),
        help="GCJ02 geographic bounds of the original image.",
    )
    parser.add_argument(
        "--no-clip",
        action="store_true",
        help="Keep graph edges outside the original image extent.",
    )
    return parser.parse_args()


def pixel_to_lonlat(x, y, width, height, lat_min, lon_min, lat_max, lon_max):
    lon = lon_min + (x / width) * (lon_max - lon_min)
    lat = lat_max - (y / height) * (lat_max - lat_min)
    return lon, lat


def segment_inside(p1, p2, width, height):
    return (
        0 <= p1.x <= width
        and 0 <= p2.x <= width
        and 0 <= p1.y <= height
        and 0 <= p2.y <= height
    )


def read_graph_edges(path):
    vertices = {}
    vertex_section = True
    next_vertex_id = 0
    edges = []

    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if not parts:
            vertex_section = False
            continue

        if vertex_section:
            if len(parts) == 2:
                vertex_id = next_vertex_id
                x, y = map(float, parts)
            elif len(parts) == 3:
                vertex_id = int(parts[0])
                x, y = map(float, parts[1:])
            else:
                raise ValueError(f"Invalid vertex line: {line}")
            vertices[vertex_id] = Point(x, y)
            next_vertex_id += 1
        else:
            if len(parts) == 2:
                src_id, dst_id = map(int, parts)
            elif len(parts) == 3:
                src_id, dst_id = map(int, parts[1:])
            else:
                raise ValueError(f"Invalid edge line: {line}")
            if src_id in vertices and dst_id in vertices:
                src = vertices[src_id]
                dst = vertices[dst_id]
                if src != dst:
                    edges.append(Edge(src, dst))
    return edges


def shp_record_content(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    content = struct.pack("<i4d2i", 3, min(xs), min(ys), max(xs), max(ys), 1, len(points))
    content += struct.pack("<i", 0)
    for x, y in points:
        content += struct.pack("<2d", x, y)
    return content


def write_shp(path, lines):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    shp_path = path.with_suffix(".shp")
    shx_path = path.with_suffix(".shx")
    dbf_path = path.with_suffix(".dbf")
    prj_path = path.with_suffix(".prj")
    cpg_path = path.with_suffix(".cpg")

    contents = [shp_record_content(line) for line in lines]
    offsets = []
    offset_words = 50
    for content in contents:
        offsets.append(offset_words)
        offset_words += 4 + len(content) // 2

    all_x = [x for line in lines for x, _ in line]
    all_y = [y for line in lines for _, y in line]
    bbox = (
        min(all_x) if all_x else 0.0,
        min(all_y) if all_y else 0.0,
        max(all_x) if all_x else 0.0,
        max(all_y) if all_y else 0.0,
    )

    def header(file_length_words):
        return (
            struct.pack(">7i", 9994, 0, 0, 0, 0, 0, file_length_words)
            + struct.pack("<2i4d4d", 1000, 3, *bbox, 0.0, 0.0, 0.0, 0.0)
        )

    with shp_path.open("wb") as f:
        f.write(header(offset_words))
        for i, content in enumerate(contents, start=1):
            f.write(struct.pack(">2i", i, len(content) // 2))
            f.write(content)

    shx_length_words = 50 + len(contents) * 4
    with shx_path.open("wb") as f:
        f.write(header(shx_length_words))
        for offset, content in zip(offsets, contents):
            f.write(struct.pack(">2i", offset, len(content) // 2))

    write_dbf(dbf_path, len(lines))
    prj_path.write_text(WGS84_PRJ, encoding="ascii")
    cpg_path.write_text("UTF-8\n", encoding="ascii")


def write_dbf(path, record_count):
    now = __import__("datetime").date.today()
    fields = [("id", "N", 10, 0)]
    header_length = 32 + len(fields) * 32 + 1
    record_length = 1 + sum(field[2] for field in fields)

    with Path(path).open("wb") as f:
        f.write(
            struct.pack(
                "<BBBBIHH20x",
                3,
                now.year - 1900,
                now.month,
                now.day,
                record_count,
                header_length,
                record_length,
            )
        )
        for name, field_type, size, decimals in fields:
            raw_name = name.encode("ascii")[:10].ljust(11, b"\x00")
            f.write(raw_name)
            f.write(field_type.encode("ascii"))
            f.write(struct.pack("<I", 0))
            f.write(struct.pack("BB14x", size, decimals))
        f.write(b"\r")
        for idx in range(1, record_count + 1):
            f.write(b" ")
            f.write(str(idx).rjust(10).encode("ascii"))
        f.write(b"\x1a")


def main():
    args = parse_args()
    lat_min, lon_min, lat_max, lon_max = args.bounds

    metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    width, height = metadata["original_size"]

    lines = []
    seen = set()
    for edge in read_graph_edges(args.graph):
        src = edge.src
        dst = edge.dst
        if not args.no_clip and not segment_inside(src, dst, width, height):
            continue
        edge_key = ((src.x, src.y), (dst.x, dst.y))
        if edge_key in seen:
            continue
        seen.add(edge_key)
        lines.append(
            [
                pixel_to_lonlat(src.x, src.y, width, height, lat_min, lon_min, lat_max, lon_max),
                pixel_to_lonlat(dst.x, dst.y, width, height, lat_min, lon_min, lat_max, lon_max),
            ]
        )

    write_shp(args.output, lines)
    print(f"wrote {len(lines)} road segments to {Path(args.output).with_suffix('.shp')}")
    print("coordinates are GCJ02 lon/lat; .prj is WGS84-like only for GIS display compatibility")


if __name__ == "__main__":
    main()
