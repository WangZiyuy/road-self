# -*- coding: UTF-8 -*-
'''
@File: Class:TileCache,Tile
       Function:load_tile, load_tile_part, save_tile, load_rect
'''


from lib import geom
from lib import graph as graph_helper

import numpy as np
import os
import random
from PIL import Image
import time
import pickle
from utils.regions import get_regions


def load_tile(tile_dir, region, i, j):
    prefix = '{}/{}_{}_{}'.format(tile_dir, region, i, j)
    sat_im = np.array(Image.open(prefix + '.png'))
    # 如果是灰度图，增加一个通道维度，使其与RGB图像形状一致
    if len(sat_im.shape) == 2:  # 灰度图像 (height, width)
        sat_im = np.expand_dims(sat_im, axis=-1)  # 增加一个维度，变成 (height, width, 1)
    # 如果是RGBA图像（有4个通道），去掉Alpha通道(height, width, channels)
    elif sat_im.shape[2] == 4:
        sat_im = sat_im[:, :, 0:3]
    # Image.open 读取的维度是(height, width, channels)感觉不需要进行这一步了
    sat_im = sat_im.swapaxes(0, 1)
    return {
        'input': sat_im,
    }


def load_tile_part(tile_dir, region, i, j, start, end):
    prefix = '{}/{}_{}_{}'.format(tile_dir, region, i, j)
    sat_im = np.array(Image.open(prefix + '.png'))
    # 如果是灰度图，增加一个通道维度，使其与RGB图像形状一致
    if len(sat_im.shape) == 2:  # 灰度图像 (height, width)
        sat_im = np.expand_dims(sat_im, axis=-1)  # 增加一个维度，变成 (height, width, 1)
    # 如果是RGBA图像（有4个通道），去掉Alpha通道
    elif sat_im.shape[2] == 4:
        sat_im = sat_im[:, :, 0:3]
    sat_im = sat_im.swapaxes(0, 1)
    sat_im = sat_im[start[0]:end[0], start[1]:end[1], :]
    return {
        'input': sat_im,
    }


def save_tile(tile, save_dir, region, i, j, start, end):
    sat_im = tile['input']
    sat_im = sat_im.swapaxes(0, 1)  # Swap axes back to original if needed
    img = Image.fromarray(sat_im)
    save_path = '{}/{}_{}_{}_{}_{}.png'.format(save_dir, region, i, j, start[0], start[1])
    img.save(save_path)


def load_rect(tile_dir, region, rect, tile_size, window_size):
    # special case for fast load: rect is single tile
    if rect.start.x % tile_size == 0 and rect.start.y % tile_size == 0 \
            and rect.end.x % tile_size == 0 and rect.end.y % tile_size == 0 \
            and rect.end.x - rect.start.x == tile_size and rect.end.y - rect.start.y == tile_size:
        sat_im = load_tile(tile_dir, region, rect.start.x // tile_size, rect.start.y // tile_size)
        sat_im['input'] = np.pad(
            sat_im['input'], 
            ((window_size // 2, window_size // 2), (window_size // 2, window_size // 2), (0, 0)),
            mode='constant')
        return sat_im

    if rect.start.x % (tile_size//2) == 0 and rect.start.y % (tile_size//2) == 0 \
            and rect.end.x % (tile_size//2) == 0 and rect.end.y % (tile_size//2) == 0 \
            and rect.end.x - rect.start.x == (tile_size//2) and rect.end.y - rect.start.y == (tile_size//2):
        start = (rect.start.x % tile_size, rect.start.y % tile_size)
        end = (start[0] + tile_size//2, start[1] + tile_size//2)
        sat_im = load_tile_part(
            tile_dir, region, rect.start.x // tile_size, rect.start.y // tile_size, start, end)
        sat_im['input'] = np.pad(
            sat_im['input'], 
            ((window_size // 2, window_size // 2), (window_size // 2, window_size // 2), (0, 0)),
            mode='constant')
        return sat_im

    tile_rect = geom.Rectangle(
        geom.Point(rect.start.x / tile_size, rect.start.y / tile_size),
        geom.Point((rect.end.x - 1) / tile_size + 1, (rect.end.y - 1) / tile_size + 1)
    )
    full_rect = geom.Rectangle(
        tile_rect.start.scale(tile_size),
        tile_rect.end.scale(tile_size)
    )
    full_ims = {}

    for i in range(tile_rect.start.x, tile_rect.end.x):
        for j in range(tile_rect.start.y, tile_rect.end.y):
            p = geom.Point(i - tile_rect.start.x, j - tile_rect.start.y).scale(tile_size)
            tile_ims = load_tile(tile_dir, region, i, j)
            for k, im in tile_ims.items():
                scale = tile_size // im.shape[0]
                if k not in full_ims:
                    full_ims[k] = np.zeros(
                        (full_rect.lengths().x // scale, full_rect.lengths().y // scale, im.shape[2]),
                        dtype=np.uint8)
                full_ims[k][p.x // scale:(p.x + tile_size) // scale, 
                            p.y // scale:(p.y + tile_size) // scale, :] = im

    crop_rect = geom.Rectangle(
        rect.start.sub(full_rect.start),
        rect.end.sub(full_rect.start)
    )
    for k in full_ims:
        scale = (full_rect.end.x - full_rect.start.x) // full_ims[k].shape[0]
        full_ims[k] = full_ims[k][crop_rect.start.x // scale:crop_rect.end.x // scale,
                                  crop_rect.start.y // scale:crop_rect.end.y // scale, :]
        full_ims[k] = np.pad(
            full_ims[k],
            ((window_size // 2, window_size // 2), (window_size // 2, window_size // 2), (0, 0)),
            mode='constant')
    return full_ims


class TileCache(object):
    def __init__(self, tile_dir, traj_dir, tile_size, window_size, limit=128):
        self.limit = limit
        self.cache = {} # 存储缓存数据（遥感图像）
        self.cache_traj = {} # 存储缓存数据（轨迹图像）
        self.last_used = {}
        self.tile_dir = tile_dir
        self.traj_dir = traj_dir
        self.tile_size = tile_size
        self.window_size = window_size

    def reduce_to(self, limit):
        while len(self.cache) > limit:
            best_k = None
            best_used = None
            for k in self.cache:
                if best_k is None or self.last_used.get(k, 0) < best_used:
                    best_k = k
                    best_used = self.last_used.get(k, 0)
            del self.cache[best_k]

    def get(self, region, rect):
        k = '{}.{}.{}.{}.{}'.format(
            region, rect.start.x, rect.start.y, rect.end.x, rect.end.y)
        if k not in self.cache:
            self.reduce_to(self.limit - 1)
            self.cache[k] = load_rect(self.tile_dir, region, rect, self.tile_size, self.window_size)
        self.last_used[k] = time.time()
        return self.cache[k]

    #新添加，用于读取轨迹图像数据
    def get_traj(self, region, rect):
        k = '{}.{}.{}.{}.{}'.format(
            region, rect.start.x, rect.start.y, rect.end.x, rect.end.y)
        if k not in self.cache_traj:
            self.reduce_to(self.limit - 1)
            self.cache_traj[k] = load_rect(self.traj_dir, region, rect, self.tile_size, self.window_size)
        self.last_used[k] = time.time()
        return self.cache_traj[k]

    #新添加，用于读取轨迹成条数据
    def load_traj_pieces(self, region, rect):
        k = '{}.{}.{}.{}.{}'.format(
            region, rect.start.x, rect.start.y, rect.end.x, rect.end.y)
        if k not in self.cache:
            self.reduce_to(self.limit - 1)
            self.cache[k] = load_rect(self.traj_dir, region, rect, self.tile_size, self.window_size)
        self.last_used[k] = time.time()
        return self.cache_traj[k]

    def get_window(self, region, big_rect, small_rect):
        big_dict = self.get(region, big_rect)
        small_dict = {}
        for k, v in big_dict.items():
            small_dict[k] = v[small_rect.start.x:small_rect.end.x,
                              small_rect.start.y:small_rect.end.y, :]
        return small_dict


def get_starting_locations(gc):
    """
    如果顶点的入边数量大于或等于 3，则认为它是一个交叉点，且设置为关键点；
    如果顶点的入边数量等于 2，并且随机数小于 1/10，添加中间点；
    如果中间点的出边是环路，则跳过这个点

    """
    g = gc.graph
    starting_locations = {'middle': [], 'junction': []}
    for vertex in g.vertices.values():
        if len(vertex.in_edges_id) >= 3 and len(vertex.out_edges_id) > 0:
            starting_locations['junction'].append([{
                'point': vertex.point,
                'key_point': True,
                'edge_pos': graph_helper.EdgePos(vertex.out_edges_id[0], 0)
            }])
        elif len(vertex.in_edges_id) == 2 and random.random() < 1/10 and len(vertex.out_edges_id) > 0:
            if gc.edge_id_to_rs(vertex.out_edges_id[0]).is_loop:
                # TODO: prevent loops !! there are many fail cases: milwaukee_1_1 left top
                continue
            starting_locations['middle'].append([{
                'point': vertex.point,
                'key_point': False,
                'edge_pos': graph_helper.EdgePos(vertex.out_edges_id[0], 0)
            }])
    return starting_locations


class Tiles(object):
    def __init__(self, training_regions, parallel_tiles, region_path, graph_dir, tile_dir, traj_dir,
                 tile_size=4096, train_tile_size=2048, window_size=256, validate=False):
        self.parallel_tiles = parallel_tiles
        """
        load tile list
        this is a list of point dicts (a point dict has keys 'x', 'y')
        don't include test tiles
        """
        # print('reading tiles')
        self.training_regions = training_regions
        self.validate_regions = ['chicago']

        self.graph_dir = graph_dir
        self.tile_dir = tile_dir
        self.traj_dir = traj_dir
        self.tile_size = tile_size
        self.train_tile_size = train_tile_size
        self.window_size = window_size
        self.validate = validate
        if self.validate:
            self.all_regions = self.training_regions + self.validate_regions
        else:
            self.all_regions = self.training_regions

        self.all_tiles = get_regions(region_path) # {"city name": Region(city, int(radius_x), int(radius_y)) ,  }

        # utils.regions.Region object
        self.train_tiles = [tile for tile in self.all_tiles.values() if tile.name in self.training_regions]
        self.cache = TileCache(
            limit=self.parallel_tiles, tile_dir=self.tile_dir, traj_dir=self.traj_dir,
            tile_size=self.tile_size, window_size=self.window_size)

        self.gcs = {}
        self.subtiles = []
        self.all_starting_locations = None

    def get_gc(self, region):
        if region in self.gcs:
            return self.gcs[region]

        fname = os.path.join(self.graph_dir, region + '.graph')
        g = graph_helper.read_graph(fname)
        #lib.graph.Graph object
        gc = graph_helper.GraphContainer(g)
        #lib.graph.GraphContainer
        # region: san antonio

        self.gcs[region] = gc
        return gc

    def prepare_training(self):
        for region in self.training_regions:
            print('reading graph for region {}'.format(region))
            self.get_gc(region)

        random.shuffle(self.train_tiles)  # TODO
        print("self.train_tiles: ", self.train_tiles)

        print("split regions to 2048x2048 parallel tiles")
        for tile in self.train_tiles:
            # tile = geom.Point(tile.x, tile.y)
            # 创建一个 geom.Point 对象，作为基准点

            # 从代码看，不管区域原本像素有多大， 此处都是只取四个rect
            # 但是由于graph中的像素点坐标都很大，所以起始点定在4096，4096也不会影响subgraph抽样

            ptile = geom.Point(tile.radius_x, tile.radius_y)
            print("ptile: ", ptile)
            # Point(0, 0)

            # 创建一个 big_rect 对象，表示一个大的矩形区域。tile.scale(self.tile_size) 和
            # tile.add(geom.Point(1, 1)).scale(self.tile_size) 分别是这个矩形的起点和终点坐标
            big_rect = geom.Rectangle(
                ptile.scale(self.tile_size),
                ptile.add(geom.Point(1, 1)).scale(self.tile_size)
            )  # start:(0,0) end:(4096,4096) 但实际上是 Point(4096, 4096) Point(8192, 8192)

            # 遍历四个不同的偏移量，以创建四个平行的子矩形。0,2048
            for offset in [geom.Point(0, 0), 
                           geom.Point(0, self.train_tile_size),
                           geom.Point(self.train_tile_size, 0), 
                           geom.Point(self.train_tile_size, self.train_tile_size)]:

                # 根据当前的 offset 计算子矩形的起点 start，然后创建一个 search_rect，表示一个搜索区域。

                start = big_rect.start.add(offset)

                # lib.geom.Rectangle object
                search_rect = geom.Rectangle(start, 
                    start.add(geom.Point(self.train_tile_size, self.train_tile_size)))

                # print("tileloader  search_rect", search_rect.start, search_rect.end)
                # tileloader  search_rect Point(4096, 4096) Point(6144, 6144)
                # tileloader  search_rect Point(4096, 6144) Point(6144, 8192)
                # tileloader  search_rect Point(6144, 4096) Point(8192, 6144)
                # tileloader  search_rect Point(6144, 6144) Point(8192, 8192)
                # 四个偏移量 从4096大矩形的四个点作为起始点

                # 获取当前切片所在区域的图形容器 gc，然后在该区域内提取出一个子图 sub_graph，并将其封装成 sub_gc 对象。
                #####################这里的问题，为什么要从一个点中间提取区域啊
                #gc = self.gcs[tile.region]

                gc = self.gcs[tile.name]
                sub_graph = gc.edge_index.subgraph(search_rect)
                sub_gc = graph_helper.GraphContainer(sub_graph)

                # 获取子图的起始位置，如果起始位置（交点+中间点）的数量少于5，则删除子图和子图容器，并跳过当前循环。
                starting_locations = get_starting_locations(sub_gc)
                if len(starting_locations["junction"]) + len(starting_locations["middle"]) < 5:
                    del sub_graph, sub_gc
                    continue

                self.subtiles.append({
                    # "region": tile.region,
                    "region": tile.name,
                    "search_rect": search_rect,
                    "cache": self.cache,
                    "starting_locations": starting_locations,
                    "gc": sub_gc,
                })

        print("regions: {}, parallel tiles: {}".format(len(self.gcs), len(self.subtiles)))

        return self.subtiles

    def num_tiles(self):
        return len(self.train_tiles)
