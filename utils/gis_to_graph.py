import sys
sys.path.append('.')
import os
import geopandas as gpd
import shapely
from lib.graph import Graph
from lib import geom
from PIL import Image
from shapely.geometry import box, LineString, MultiLineString
import matplotlib.pyplot as plt
import networkx as nx


class GisToGraphConverter:
    def __init__(self, region_num, trajectory):
        """
        初始化 GisToGraphConverter 类，设置转换参数。
        :param min_lat: 经度最小值，用于映射到像素坐标
        :param max_lat: 经度最大值，用于映射到像素坐标
        :param min_lng: 纬度最小值，用于映射到像素坐标
        :param max_lng: 纬度最大值，用于映射到像素坐标
        """
        self.trajectory = trajectory
        self.region_num = str(region_num)

    def get_trans_para(self):
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


def traverse_geometry_elements(folder_path, file_path, points, REGION_NUM, aoi_poly):

    input_file = os.path.join(folder_path, file_path)
    data = gpd.read_file(input_file)
    # 存储点id和坐标信息的字典

    # 打开文件，并写入内容
    file = open(f"/home/wangziyu/VecRoad/data_self/shp_files/{REGION_NUM}/shp_file.txt", "w")

    for index, data in data.iterrows():

        geomdata = data['geometry']
        file.write(str(geomdata))
        file.write("\n")
        # 如果是点，处理节点信息
        if geomdata is None:
            continue

        elif geomdata.geom_type == 'LineString':
            edge_props = data['eid']
            # edge_coords = list(geomdata.coords)
            # for i in edge_coords:
            #     points.add(i)

            clipped = geomdata.intersection(aoi_poly)
            for coords in _iter_lines(clipped):
                # 把每条线的所有顶点加入集合（已在 AOI 内）
                for pt in coords:
                    points.add(_round_coord(pt))  # (lng, lat)(108.976844, 34.2628499)

    file.close()


def construct_graph_content(folder_path, file_path, point_dict, graph, aoi_poly):

    input_file = os.path.join(folder_path, file_path)
    data = gpd.read_file(input_file)

    for index, data in data.iterrows():

        geomdata = data['geometry']

        # 如果是点，处理节点信息
        if geomdata is None:
            continue

        elif geomdata.geom_type == 'LineString':
            edge_props = data['eid']

            edge_id = edge_props  # 或者使用其他方式获取边ID

            # edge_coords_start = geomdata.coords[0]  # 边的坐标序列
            # edge_coords_end = geomdata.coords[1]  # 边的坐标序列

            edge_coords = list(geomdata.coords)
            # for i in range(len(edge_coords) - 1):
            #     edge_coords_start = edge_coords[i]
            #     edge_coords_end = edge_coords[i + 1]
            #
            #     if edge_coords_end not in point_dict.keys() or edge_coords_start not in point_dict.keys():
            #         continue
            #
            #     start_id = point_dict[edge_coords_start]
            #     end_id = point_dict[edge_coords_end]
            #
            #     if start_id != end_id:
            #         graph.add_bidirectional_edge(start_id, end_id)
            # 裁剪线段并建立边
            clipped = geomdata.intersection(aoi_poly)
            for coords in _iter_lines(clipped):
                for i in range(len(coords) - 1):
                    p0 = _round_coord(coords[i])
                    p1 = _round_coord(coords[i + 1])

                    if p0 in point_dict and p1 in point_dict and p0 != p1:
                        u = point_dict[p0]
                        v = point_dict[p1]
                        graph.add_bidirectional_edge(u, v)  # 添加双向边

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
    if geom_clip.is_empty:
        return
    if isinstance(geom_clip, LineString):
        yield list(geom_clip.coords)
    elif isinstance(geom_clip, MultiLineString):
        for g in geom_clip.geoms:
            if not g.is_empty:
                yield list(g.coords)


def gis_to_graph():
    REGION_NUM = 'xian'

    # 指定文件夹路径
    folder_path = f"../data_self/shp_files/{REGION_NUM}/"

    shp_file = f"{REGION_NUM}.shp"

    # 读取shp文件
    graph = Graph()

    points = set()
    point_dict = {}

    # 设置经纬度坐标点向像素坐标点转化所需参数
    # sta_id = str(REGION_NUM)
    # id2info = load_sta_data('../data_self/shp_files/sta_mbrs.csv')
    # sta_mbr_init = id2info[sta_id]
    # max_lat, min_lat, max_lng, min_lng = sta_mbr_init[2], sta_mbr_init[0], sta_mbr_init[3], sta_mbr_init[1]
    # print("max_lat, min_lat, max_lng, min_lng: ", max_lat, min_lat, max_lng, min_lng)
    # min_lat, min_lng, max_lat, max_lng = 34.204950, 108.921860, 34.278600, 109.008830, # 4*4
    min_lat, min_lng, max_lat, max_lng = 34.223382, 108.921860, 34.278678, 108.988720, # 3*3
    aoi_poly = box(min_lng, min_lat, max_lng, max_lat)
    # 因为遥感图像的范围稍微大于轨迹数据的范围，所以会出现轨迹的最下面和最右边一条是黑的，所以截取了左上角的3*3

    # img_dir = f"../data_self/input/imagery/{REGION_NUM}_0_0.png"
    # img = Image.open(img_dir)
    # nb_cols, nb_rows = img.size

    nb_cols, nb_rows = 6144, 6144
    yscale = nb_rows / (max_lat - min_lat)
    xscale = nb_cols / (max_lng - min_lng)
    print("nb_rows, nb_cols, yscale, xscale: ", nb_rows, nb_cols, yscale, xscale)

    # 手动标注后的shp文件只有边信息
    # 需要先读取出点信息
    traverse_geometry_elements(folder_path, shp_file, points, REGION_NUM, aoi_poly)
    # 向图中添加点信息
    # 构造点信息字典{点：id}
    #
    for i, point in enumerate(points):
        point_pixel = latlng_to_pixel(point[1], point[0], nb_rows, min_lat, min_lng, xscale, yscale)
        graph.add_vertex(geom.FPoint(point_pixel[0], point_pixel[1]), vertex_id=i)
        point_dict[point] = i

    # 再次遍历，向图中添加边信息
    construct_graph_content(folder_path, shp_file, point_dict, graph, aoi_poly)

    graph.save_gis_to_graph(os.path.join(folder_path, f"{REGION_NUM}.graph"))
    print(f"Trans {REGION_NUM} from shp to graph, saved in {folder_path}")
    print(f"Trans {len(points)} points to graph")
    print(f"Trans {len(graph.edges)} edges to graph")


# gis_to_graph()


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


# gis_to_graph()
# visualize_graph("../data_self/shp_files/xian/xian.graph")