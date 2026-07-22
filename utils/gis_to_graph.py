import argparse
import hashlib
import sys
sys.path.append('.')
import json
import os
from pathlib import Path
import geopandas as gpd
import shapely
from lib.graph import Graph
from lib import geom
from PIL import Image
from shapely.geometry import box, LineString, MultiLineString
import matplotlib.pyplot as plt
import networkx as nx


XIAN_BBOX_GCJ02 = {
    "lat_min": 34.22484722131834,
    "lon_min": 108.94460164474442,
    "lat_max": 34.24707831919142,
    "lon_max": 108.9677436888106,
}


def _repo_path(*parts):
    return os.path.abspath(os.path.join(os.getcwd(), *parts))


def _load_region_metadata(region_num, data_root="data_self"):
    candidates = [
        _repo_path(data_root, "input", "regions", f"{region_num}_metadata.json"),
        _repo_path("data_self", "input", "regions", f"{region_num}_metadata.json"),
        _repo_path("data", "input", "regions", f"{region_num}_metadata.json"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f), path
    return None, None


def _metadata_bbox(meta, region_num):
    bbox = meta.get("bbox_gcj02") or meta.get("bbox_wgs84") or meta.get("bbox")
    if isinstance(bbox, dict):
        return bbox
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        return {
            "lat_min": bbox[0],
            "lon_min": bbox[1],
            "lat_max": bbox[2],
            "lon_max": bbox[3],
        }
    if str(region_num).lower() == "xian":
        return XIAN_BBOX_GCJ02
    return None


def _metadata_size(meta):
    size = meta.get("original_size") or meta.get("image_size") or meta.get("valid_size")
    if isinstance(size, (list, tuple)) and len(size) == 2:
        return int(size[0]), int(size[1])
    return None


class GisToGraphConverter:
    def __init__(self, region_num, trajectory, data_root="data_self"):
        """
        初始化 GisToGraphConverter 类，设置转换参数。
        :param min_lat: 经度最小值，用于映射到像素坐标
        :param max_lat: 经度最大值，用于映射到像素坐标
        :param min_lng: 纬度最小值，用于映射到像素坐标
        :param max_lng: 纬度最大值，用于映射到像素坐标
        """
        self.trajectory = trajectory
        self.region_num = str(region_num)
        self.data_root = data_root

    def get_trans_para(self):
        meta, meta_path = _load_region_metadata(self.region_num, self.data_root)
        if meta is not None:
            bbox = _metadata_bbox(meta, self.region_num)
            size = _metadata_size(meta)
            if bbox is None or size is None:
                raise ValueError(
                    f"metadata {meta_path} must contain bbox and original_size for trajectory conversion"
                )
            min_lat = float(bbox["lat_min"])
            max_lat = float(bbox["lat_max"])
            min_lng = float(bbox["lon_min"])
            max_lng = float(bbox["lon_max"])
            nb_cols, nb_rows = size
            yscale = nb_rows / (max_lat - min_lat)
            xscale = nb_cols / (max_lng - min_lng)
            return min_lat, max_lat, min_lng, max_lng, nb_rows, nb_cols, yscale, xscale

        sta_path = _repo_path(self.data_root, "shp_files", "sta_mbrs.csv")
        if not os.path.isfile(sta_path):
            raise FileNotFoundError(
                f"missing region metadata and station bounds: {sta_path}"
            )
        if os.path.isfile(sta_path):
            sta_mbr_init = load_sta_data(sta_path)[self.region_num]
            min_lat = sta_mbr_init[0]
            max_lat = sta_mbr_init[2]
            min_lng = sta_mbr_init[1]
            max_lng = sta_mbr_init[3]
            img_candidates = [
                _repo_path(self.data_root, "input", "imagery_8192", f"{self.region_num}.png"),
                _repo_path(self.data_root, "input", "imagery", f"{self.region_num}_0_0.png"),
            ]
            img_dir = next((path for path in img_candidates if os.path.isfile(path)), None)
            if img_dir is None:
                raise FileNotFoundError(
                    f"cannot find imagery for trajectory conversion: {img_candidates}"
                )
            nb_cols, nb_rows = Image.open(img_dir).size
            yscale = nb_rows / (max_lat - min_lat)
            xscale = nb_cols / (max_lng - min_lng)
            return min_lat, max_lat, min_lng, max_lng, nb_rows, nb_cols, yscale, xscale

        sta_mbr_init = load_sta_data('/home/wangziyu/VecRoad/data_self/shp_files/sta_mbrs.csv')[self.region_num]
        min_lat = sta_mbr_init[0]
        max_lat = sta_mbr_init[2]
        min_lng = sta_mbr_init[1]
        max_lng = sta_mbr_init[3]

        # 计算经纬度到像素坐标的转换比例
        img_dir = f"/home/wangziyu/VecRoad/data_self/input/imagery/{self.region_num}_0_0.png"
        nb_cols, nb_rows = Image.open(img_dir).size
        yscale = nb_rows / (max_lat - min_lat)
        xscale = nb_cols / (max_lng - min_lng)

        return min_lat, max_lat, min_lng, max_lng, nb_rows, nb_cols, yscale, xscale


    def convert_trajectories_to_pixels(self):
        """
        将轨迹点（经纬度）转换为像素坐标。
        :return: 转换后的像素坐标集合
        """
        pixel_trajectories = []
        min_lat, max_lat, min_lng, max_lng, nb_rows, nb_cols, yscale, xscale = self.get_trans_para()
        # print('self.sta_mbr_init:', min_lat, max_lat, min_lng, max_lng)
        # print("nb_rows, nb_cols, yscale, xscale: ", nb_rows, nb_cols, yscale, xscale)

        for i, point in enumerate(self.trajectory):
            j = float(nb_rows - (point[0] - min_lat) * yscale)
            i = float((point[1] - min_lng) * xscale)
            pixel_trajectories.append((i, j))
        return pixel_trajectories


def load_sta_data(path):
    id2info = {}
    with open(path, 'r') as f:
        f.readline()
        for line in f.readlines():
            arr = line.strip().split(';')
            sta_id = arr[0]
            mbr_poly = shapely.wkt.loads(arr[1])
            sta_mbr = mbr_poly.bounds[1], mbr_poly.bounds[0], mbr_poly.bounds[3], mbr_poly.bounds[2]
            # depot_pt = SPoint(float(arr[2]), float(arr[3]))
            id2info[sta_id] = sta_mbr
    return id2info


def latlng_to_pixel(lat, lng, nb_rows, min_lat, min_lng, xscale, yscale):
    j = float(nb_rows - (lat - min_lat) * yscale)
    i = float((lng - min_lng) * xscale)
    return i, j


def _round_coord(pt, ndigits=9):
    # shapely 坐标是 (lng, lat)
    return (round(pt[0], ndigits), round(pt[1], ndigits))


def _iter_lines(geom_clip):
    """把裁剪后的几何展开成若干条 LineString 的坐标列表"""
    if geom_clip is None or geom_clip.is_empty:
        return
    if isinstance(geom_clip, LineString):
        coords = list(geom_clip.coords)
        if len(coords) >= 2 and geom_clip.length > 0:
            yield coords
    elif isinstance(geom_clip, MultiLineString):
        for g in geom_clip.geoms:
            yield from _iter_lines(g)
    elif hasattr(geom_clip, "geoms"):
        for g in geom_clip.geoms:
            yield from _iter_lines(g)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_edge(p0, p1):
    return (p0, p1) if p0 < p1 else (p1, p0)


def _graph_metadata(metadata_path):
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    region = metadata.get("region", "xian")
    bbox = _metadata_bbox(metadata, region)
    size = _metadata_size(metadata)
    if bbox is None or size is None:
        raise ValueError(
            f"metadata {metadata_path} must contain bbox and original_size"
        )
    width, height = map(int, size)
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid original_size in {metadata_path}: {size}")
    required = ("lat_min", "lon_min", "lat_max", "lon_max")
    missing = [key for key in required if key not in bbox]
    if missing:
        raise ValueError(f"metadata bbox is missing keys: {missing}")
    bbox = {key: float(bbox[key]) for key in required}
    if bbox["lat_min"] >= bbox["lat_max"] or bbox["lon_min"] >= bbox["lon_max"]:
        raise ValueError(f"invalid bbox in {metadata_path}: {bbox}")
    return metadata, bbox, width, height


def build_graph_from_shapefile(
        shp_path, metadata_path, output_path, coord_round_digits=9):
    """Convert geographic GT lines into a deterministic VecRoad graph.

    The shapefile and metadata bbox must contain coordinates in the same
    geographic system. Xian stores GCJ02 degree values in an EPSG:4326-labelled
    shapefile, so this conversion deliberately does not reproject the values.
    """
    shp_path = Path(shp_path).resolve(strict=True)
    metadata_path = Path(metadata_path).resolve(strict=True)
    output_path = Path(output_path).resolve(strict=False)
    metadata, bbox, width, height = _graph_metadata(metadata_path)
    roads = gpd.read_file(shp_path)
    if roads.crs is None:
        raise ValueError(f"shapefile has no CRS declaration: {shp_path}")

    aoi = box(
        bbox["lon_min"], bbox["lat_min"],
        bbox["lon_max"], bbox["lat_max"])
    unique_edges = set()
    raw_segment_count = 0
    clipped_line_parts = 0
    skipped = {
        "null_geometry": 0,
        "empty_geometry": 0,
        "zero_length": 0,
        "non_line_geometry": 0,
    }

    for geometry in roads.geometry:
        if geometry is None:
            skipped["null_geometry"] += 1
            continue
        if geometry.is_empty:
            skipped["empty_geometry"] += 1
            continue
        if geometry.geom_type not in ("LineString", "MultiLineString"):
            skipped["non_line_geometry"] += 1
            continue
        if geometry.length <= 0:
            skipped["zero_length"] += 1
            continue

        clipped = geometry.intersection(aoi)
        for coordinates in _iter_lines(clipped):
            clipped_line_parts += 1
            rounded = [
                _round_coord(point, ndigits=coord_round_digits)
                for point in coordinates
            ]
            for p0, p1 in zip(rounded, rounded[1:]):
                if p0 == p1:
                    skipped["zero_length"] += 1
                    continue
                raw_segment_count += 1
                unique_edges.add(_canonical_edge(p0, p1))

    if not unique_edges:
        raise ValueError(
            f"no road segments remain after clipping {shp_path} to {bbox}"
        )

    xscale = width / (bbox["lon_max"] - bbox["lon_min"])
    yscale = height / (bbox["lat_max"] - bbox["lat_min"])
    geographic_points = sorted({point for edge in unique_edges for point in edge})
    geographic_to_pixel = {}
    for lon, lat in geographic_points:
        pixel_x, pixel_y = latlng_to_pixel(
            lat, lon, height, bbox["lat_min"], bbox["lon_min"],
            xscale, yscale)
        # Rounded intersections may exceed a bbox by a tiny floating error.
        pixel_x = int(min(max(pixel_x, 0.0), float(width)))
        pixel_y = int(min(max(pixel_y, 0.0), float(height)))
        geographic_to_pixel[(lon, lat)] = (pixel_x, pixel_y)

    pixel_edges = set()
    collapsed_pixel_segments = 0
    for p0, p1 in unique_edges:
        pixel_p0 = geographic_to_pixel[p0]
        pixel_p1 = geographic_to_pixel[p1]
        if pixel_p0 == pixel_p1:
            collapsed_pixel_segments += 1
            continue
        pixel_edges.add(_canonical_edge(pixel_p0, pixel_p1))

    pixel_points = sorted({point for edge in pixel_edges for point in edge})
    point_ids = {point: idx for idx, point in enumerate(pixel_points)}
    graph = Graph()
    for vertex_id, (pixel_x, pixel_y) in enumerate(pixel_points):
        graph.add_vertex(geom.Point(pixel_x, pixel_y), vertex_id=vertex_id)

    for p0, p1 in sorted(pixel_edges):
        graph.add_bidirectional_edge(point_ids[p0], point_ids[p1])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    graph.save_gis_to_graph(temporary_path)
    os.replace(temporary_path, output_path)

    nx_graph = graph.convert_to_networkx()
    component_sizes = sorted(
        (len(component) for component in nx.connected_components(nx_graph)),
        reverse=True)
    xs = [point[0] for point in pixel_points]
    ys = [point[1] for point in pixel_points]
    report = {
        "region": metadata.get("region", output_path.stem),
        "source_shapefile": os.fspath(shp_path),
        "source_shapefile_sha256": _sha256(shp_path),
        "metadata": os.fspath(metadata_path),
        "metadata_sha256": _sha256(metadata_path),
        "output_graph": os.fspath(output_path),
        "crs": str(roads.crs),
        "bbox": bbox,
        "original_size": [width, height],
        "pixel_coordinate_quantization": "integer truncation (lib.geom.Point)",
        "source_feature_count": int(len(roads)),
        "clipped_line_parts": clipped_line_parts,
        "raw_segment_count": raw_segment_count,
        "duplicate_geographic_segments_removed": raw_segment_count - len(unique_edges),
        "collapsed_pixel_segments_removed": collapsed_pixel_segments,
        "duplicate_pixel_segments_removed": (
            len(unique_edges) - collapsed_pixel_segments - len(pixel_edges)),
        "vertex_count": len(pixel_points),
        "undirected_edge_count": len(pixel_edges),
        "directed_edge_count": len(graph.edges),
        "connected_component_count": len(component_sizes),
        "largest_component_vertex_count": component_sizes[0],
        "component_sizes": component_sizes,
        "pixel_bounds": {
            "x_min": min(xs), "y_min": min(ys),
            "x_max": max(xs), "y_max": max(ys),
        },
        "skipped": skipped,
    }
    if report["directed_edge_count"] != 2 * report["undirected_edge_count"]:
        raise AssertionError("graph does not contain exactly two directions per edge")
    return report


def gis_to_graph(
        shp_path="data_self/shp_files/xian/xian.shp",
        metadata_path="data_self/input/regions/xian_metadata.json",
        output_path="data_self/input/graphs/xian.graph"):
    return build_graph_from_shapefile(shp_path, metadata_path, output_path)


def _parse_graph_args():
    parser = argparse.ArgumentParser(
        description="Convert a geographic GT road shapefile to a VecRoad graph."
    )
    parser.add_argument(
        "--shp", default="data_self/shp_files/xian/xian.shp")
    parser.add_argument(
        "--metadata", default="data_self/input/regions/xian_metadata.json")
    parser.add_argument(
        "--output", default="data_self/input/graphs/xian.graph")
    parser.add_argument(
        "--report", default=None,
        help="Optional JSON path for conversion provenance and validation stats.")
    parser.add_argument("--coord-round-digits", type=int, default=9)
    return parser.parse_args()


def visualize_graph(graph_file):
    graph = nx.Graph()  # 创建一个无向图

    # 读取文件并解析
    with open(graph_file, 'r') as f:
        lines = f.readlines()

    # 获取节点坐标和边的索引
    nodes = []
    edges = []
    is_edges_section = False  # 标志是否进入边的部分

    for line in lines:
        line = line.strip()
        if line == "":  # 空行，标志着进入边部分
            is_edges_section = True
            continue
        if not is_edges_section:
            # 第一部分：节点坐标
            x, y = map(float, line.split())
            y = 6144 - y  # 在这里翻转 y 坐标 因为在graph坐标系中y轴向下是越来越大的
            nodes.append((x, y))
        else:
            # 第二部分：边的索引
            start_id, end_id = map(int, line.split())
            edges.append((start_id, end_id))

    # 添加节点
    for i, (x, y) in enumerate(nodes):
        graph.add_node(i, pos=(x, y))  # 添加节点，并且保存位置坐标

    # 添加边
    for start_id, end_id in edges:
        graph.add_edge(start_id, end_id)

    # 绘制图形
    pos = nx.get_node_attributes(graph, 'pos')  # 获取节点位置
    plt.figure(figsize=(10, 10))  # 设置图的大小
    nx.draw(graph, pos, with_labels=False, node_size=5, node_color='skyblue', font_size=8)

    plt.title("Graph Visualization")
    plt.show()


if __name__ == "__main__":
    args = _parse_graph_args()
    conversion_report = build_graph_from_shapefile(
        args.shp,
        args.metadata,
        args.output,
        coord_round_digits=args.coord_round_digits,
    )
    if args.report:
        report_path = Path(args.report).resolve(strict=False)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_report = report_path.with_suffix(report_path.suffix + ".tmp")
        temporary_report.write_text(
            json.dumps(conversion_report, indent=2), encoding="utf-8")
        os.replace(temporary_report, report_path)
    print(json.dumps(conversion_report, indent=2))
