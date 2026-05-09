# -*- coding: UTF-8 -*-
'''
@Project ：VecRoad
@File    ：graph2shp.py
@IDE     ：PyCharm
@Author  ：wzy
@Date    ：2025/4/30 11:00:46
'''
import math
DEGREES_TO_RADIANS = math.pi / 180
RADIANS_TO_DEGREES = 1 / DEGREES_TO_RADIANS
EARTH_MEAN_RADIUS_METER = 6371008.7714
DEG_TO_KM = DEGREES_TO_RADIANS * EARTH_MEAN_RADIUS_METER
LAT_PER_METER = 8.993203677616966e-06
LNG_PER_METER = 1.1700193970443768e-05

import sys
sys.path.append('.')
import geopandas as gpd
from shapely.geometry import LineString
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--graph_dir", type=str, help="input graph dir", default="/home/wangziyu/VecRoad/data_self/graphs/vecroad_4/post/")
parser.add_argument(
    "--save_dir", type=str, help="save csv dir", default="/home/wangziyu/VecRoad/data_self/graphs/vecroad_4/graphs_shp/")
args = parser.parse_args()


def convert_graph_to_shp(graph_path, output_shp_path, full_img_size, geo_bounds, patch_size, patch_origin, crs="EPSG:4326"):
    """
       将 .graph 文件转换为 .shp 线段文件（edges），并输出当前 patch 的经纬度范围
       Parameters:
           - full_img_size: tuple(W, H) → 原始图像大小
           - geo_bounds: tuple(lon_min, lon_max, lat_min, lat_max) → 原图地理范围
           - patch_origin: tuple(x_offset, y_offset) → 当前 patch 左上角在原图中的像素坐标
           - patch_size: tuple(w, h) → patch 尺寸（默认4096×4096）
       """

    lon_min, lon_max, lat_min, lat_max = geo_bounds
    W, H = full_img_size

    # 每个像素代表的经纬度间隔
    res_x = (lon_max - lon_min) / W
    res_y = (lat_max - lat_min) / H
    print('每个像素代表的经纬度间隔: ', res_x, res_y)

    x_offset, y_offset = patch_origin
    patch_w, patch_h = patch_size

    # 计算当前 patch 的地理范围
    patch_lon_min = lon_min + x_offset * res_x
    patch_lat_max = lat_max - y_offset * res_y
    patch_lon_max = lon_min + (x_offset + patch_w) * res_x
    patch_lat_min = lat_max - (y_offset + patch_h) * res_y
    print("🌍 当前 patch 经纬度范围：")
    print(f"  - lon_min: {patch_lon_min:.8f}")
    print(f"  - lon_max: {patch_lon_max:.8f}")
    print(f"  - lat_min: {patch_lat_min:.8f}")
    print(f"  - lat_max: {patch_lat_max:.8f}")

    with open(graph_path, 'r') as f:
        lines = f.readlines()

    # 分割成点段与边段
    node_lines, edge_lines = [], []
    in_edge_section = False

    for line in lines:
        line = line.strip()
        if line == "":
            in_edge_section = True
            continue
        if not in_edge_section:
            node_lines.append(line)
        else:
            edge_lines.append(line)

    # 解析点
    points = []
    x_offset, y_offset = patch_origin

    for line in node_lines:
        parts = line.strip().split()
        if len(parts) != 2:
            continue
        x, y = map(float, parts)
        lon = lon_min + (x + x_offset) * res_x
        lat = lat_max - (y + y_offset) * res_y  # 纬度方向是向下递减的
        points.append((lon, lat))

    # 解析边
    edges = []
    for line in edge_lines:
        parts = line.strip().split()
        if len(parts) != 2:
            continue
        a1, a2 = map(int, parts)
        if a1 < len(points) and a2 < len(points):
            pt1 = points[a1]
            pt2 = points[a2]
            edges.append(LineString([pt1, pt2]))

    # 构建 GeoDataFrame
    gdf = gpd.GeoDataFrame({'id': range(len(edges))}, geometry=edges, crs=crs)

    # 写入 shapefile
    os.makedirs(os.path.dirname(output_shp_path), exist_ok=True)
    gdf.to_file(output_shp_path)

    print(f"✅ Saved shapefile to {output_shp_path}")

# 示例用法
if __name__ == "__main__":
    os.makedirs(args.save_dir, exist_ok=True)
    files = [f for f in os.listdir(args.graph_dir) if f.endswith('.graph')]
    for file in files:
        graph_path = os.path.join(args.graph_dir, file)
        shp_path = os.path.join(args.save_dir, file.replace('.graph', '.shp'))

        convert_graph_to_shp(graph_path, shp_path,
                             full_img_size=(5465, 4367),
                             geo_bounds=(116.30278723404255, 116.33204255319149,  # lon_min, lon_max
                                         39.87206896551724, 39.890098522167484),  # lat_min, lat_max
                             patch_size=(4096, 4096),
                             patch_origin=(0, 0))