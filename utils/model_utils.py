import os
import torch
from lib import geom, graph as graph_helper
import numpy as np
import pandas as pd
import math
from PIL import Image
import random
import rtree
import sys
import time
import cv2 as cv
from skimage import measure
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from lib import geom
from utils.gis_to_graph import GisToGraphConverter
from configs.config import config
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


class Path(object):
    def __init__(self, idx, training, gc, tile_data, all_trajectories, all_pixel_trajectories,
                 net=None, graph=None, road_seg=None, WINDOW_SIZE=256,
                 traj_grid_index=None, traj_grid_cell_size=None):
        """
        graph container contains the total graph of the region, not only the search rectangle,
        so it records the road segment info of the total graph, when reset the gragh container(gc),
        be careful not to affect other path belonging to the same graph

        path路径中包含了id、gragh container（这个意思是一个图容器中包含很多路径？）、tile_data（）

        图容器包含该区域的总图，而不仅仅是搜索矩形，那么在这里初始化总轨迹数据也合理吧
        """

        self.idx = idx
        self.gc = gc
        self.tile_data = tile_data
        self.road_seg = road_seg
        self.road_seg_origin = self.tile_data['search_rect'].start
        self.ROADSEG_OVERWRITE_THICKNESS = 20
        self.ROADSEG_OVERWRITE_RADIUS = 20
        if graph is None:
            self.graph = graph_helper.Graph()
        else:
            self.graph = graph
            self.remove_graph_from_road_seg()
        self.tile_size = self.tile_data['search_rect'].lengths().x
        # print("self.tile_size: " ,self.tile_data['search_rect'].lengths()) # 这里的区域应该是2048*2048

        self.unmatched_vertices = 0

        self._load_edge_rtree()
        self._load_key_point_rtree()

        self.not_explored_starting_points = self.tile_data['starting_locations']['junction'].copy() + \
            self.tile_data['starting_locations']['middle'].copy()

        self.search_vertices = []

        self.anchor_point_rtree = rtree.index.Index()
        self.indexed_anchor_points = dict()
        self.is_training = training
        if self.is_training:
            self.rs_exploration = graph_helper.RoadSegmentExplorationDict(
                gc=gc, rect=tile_data['search_rect'])

        self.all_trajectories = [] if all_trajectories is None else all_trajectories
        self.all_pixel_trajectories = [] if all_pixel_trajectories is None else all_pixel_trajectories
        self.traj_grid_index = traj_grid_index or {}
        self.traj_grid_cell_size = traj_grid_cell_size
        self.use_traj_grid_index = bool(self.traj_grid_index) and self.traj_grid_cell_size is not None
        self.all_pixel_trajectories_gpu = []
        if not self.use_traj_grid_index:
            self.all_pixel_trajectories_gpu = [
                torch.tensor(traj, device="cuda", dtype=torch.float32)
                for traj in self.all_pixel_trajectories
            ]

        self.net = net #为了传递两个半径的自定义参数
        if self.net is not None:
            # 自动剥离DataParallel包装
            self.net = net.module if hasattr(net, 'module') else net

        self.valid_trajectories = []
        self.circles = []
        # self._build_rtree_for_trajectories()

    def remove_graph_from_road_seg(self):
        if self.graph is not None and self.road_seg is not None:
            for edge in self.graph.edges.values():
                src = edge.src(self.graph).point.sub(self.road_seg_origin)
                dst = edge.dst(self.graph).point.sub(self.road_seg_origin)
                cv.line(self.road_seg, (src.y, src.x), (dst.y, dst.x),
                        color=0, thickness=self.ROADSEG_OVERWRITE_THICKNESS)

    def _load_key_point_rtree(self):
        self.key_point_rtree = rtree.index.Index()
        self.indexed_key_points = dict()
        starting_locations = self.tile_data['starting_locations']['junction']
        for i, item in enumerate(starting_locations):
            pnt = item[0]['point']
            self.key_point_rtree.insert(i, (pnt.x, pnt.y, pnt.x, pnt.y))
            self.indexed_key_points[i] = pnt

    def _load_edge_rtree(self):
        self.indexed_edges = set()
        # rtree.index.Index类主要用于数据操作
        self.edge_rtree = rtree.index.Index()
        for edge in self.graph.edges.values():
            self._add_edge_to_rtree(edge)

    def _add_edge_to_rtree(self, edge):
        if edge.id in self.indexed_edges:
            return
        self.indexed_edges.add(edge.id)
        bounds = edge.segment(self.graph).bounds().add_tol(1)
        self.edge_rtree.insert(
            edge.id, (bounds.start.x, bounds.start.y, bounds.end.x, bounds.end.y))

    def _add_bidirectional_edge(self, src, dst, prob=1.0):
        edges = self.graph.add_bidirectional_edge(src.id, dst.id)
        edges[0].prob = prob
        edges[1].prob = prob
        self._add_edge_to_rtree(edges[0])
        self._add_edge_to_rtree(edges[1])

    def _build_rtree_for_trajectories(self):
        """
            为所有轨迹构建 R-tree 索引
            :param self.all_trajectories: List，每条轨迹是点的集合 [(lat, lng), ...]
            :return: R-tree 索引对象
            """
        self.trajectories_rtree = rtree.index.Index()
        self.trajectories_map = {}
        # 遍历所有轨迹，将其点加入 R-tree
        for traj_idx, trajectory in enumerate(tqdm(self.all_pixel_trajectories, desc="Building rtree for trajectories")):
            # 将经纬度的轨迹点转化为像素坐标
            for point_idx, point in enumerate(trajectory):
                # 用点的坐标作为边界创建索引项
                if self.tile_data['search_rect'].contains(geom.Point(point[0], point[1])):
                    id = traj_idx * 10**6 + point_idx
                    self.trajectories_rtree.insert(id, (float(point[0]), float(point[1]), float(point[0]), float(point[1])))
                    self.trajectories_map[id] = (traj_idx, point_idx)

    def prepend_search_vertex(self, vertex, is_key_point):
        if self.tile_data['search_rect'].contains(vertex.point):
            self.search_vertices.append((vertex, is_key_point))
            return True
        else:
            return False

    def mark_rs_explored_part(self, rs, edge_pos):
        rs_exp = self.rs_exploration[rs.id]
        if rs_exp.is_explored():
            return
        edge = edge_pos.edge(self.gc.graph)
        curr_distance = rs.edge_distances[edge.id] + edge_pos.distance
        if curr_distance > rs_exp.explored_start_dis:
            if round(curr_distance)+1 >= round(rs_exp.explored_end_dis):
                self.mark_rs_explored(rs)
                return
            rs_exp.explored_start_dis = curr_distance
            rs_exp.explored_start_pos = edge_pos
            opposite_rs = rs.get_opposite_rs(
                self.gc.edge_id_to_rs_id, self.gc.road_segments, self.gc.graph)
            if opposite_rs is not None:
                opposite_rs_exp = self.rs_exploration[opposite_rs.id]
                opposite_rs_exp.explored_end_dis = opposite_rs.marked_length - curr_distance
                opposite_rs_exp.explored_end_pos = edge_pos.reverse(
                    self.gc.graph)
            else:
                print('err!1')
        else:
            # meet a self-crossed overpass
            pass

    def mark_rs_explored(self, rs):
        self.rs_exploration[rs.id].explored = True
        oppo_rs = rs.get_opposite_rs(
            self.gc.edge_id_to_rs_id, self.gc.road_segments, self.gc.graph)
        if oppo_rs is not None:
            self.rs_exploration[oppo_rs.id].explored = True
        else:
            print('err!2')

    def is_explored_edge_pos(self, edge_pos):
        edge = edge_pos.edge(self.gc.graph)
        rs = self.gc.edge_id_to_rs(edge.id)
        if self.rs_exploration[rs.id].is_explored():
            return True
        curr_distance = rs.edge_distances[edge.id] + edge_pos.distance
        if self.rs_exploration[rs.id].explored_start_dis <= curr_distance < self.rs_exploration[rs.id].explored_end_dis:
            return False
        return True

    def get_vertex_from_point_in_graph(self, pnt):
        lst = list(self.edge_rtree.intersection((pnt.x, pnt.y, pnt.x, pnt.y)))
        for edge_id in lst:
            edge = self.graph.edges[edge_id]
            for vertex in [edge.src(self.graph), edge.dst(self.graph)]:
                if vertex.point == pnt:
                    return vertex
        return None

    def get_rs_from_next_point(self, target_poses, next_point):
        target_poses_len = len(target_poses)
        assert target_poses_len >= 1
        if target_poses_len == 1:
            nearest_target_pos = target_poses[0]
        else:
            nearest_target_pos = None
            for pos in target_poses:
                if nearest_target_pos is None or \
                        pos.point(self.gc.graph).distance(next_point) < nearest_target_pos.point(self.gc.graph).distance(next_point):
                    nearest_target_pos = pos
        rs = self.gc.edge_id_to_rs(nearest_target_pos.edge_id)
        return rs, nearest_target_pos

    def generate_new_vertex(self, next_point):
        next_vertex = self.graph.add_vertex(next_point)
        next_vertex.edge_pos = None  # if TRAIN, set value below
        new_vertex_index = next_vertex.id
        self.anchor_point_rtree.insert(
            new_vertex_index, (next_point.x, next_point.y, next_point.x, next_point.y))
        self.indexed_anchor_points[new_vertex_index] = next_point
        return next_vertex

    # if at key point, target_poses contains position of next points in different road segment with one time step
    # [[.....], [], [], []]
    # else, target_poses contains position of next points in one road segment with num_targets time step
    # [[.], [.], [.], []] or [[.], [.....], [], []] (end with a junction)
    def push(self, extension_vertex, is_key_point, follow_mode, target_poses, output_points, RECT_RADIUS=10,
             road_segmentation=None, NUM_TARGETS=4, WINDOW_SIZE=256, STEP_LENGTH=20, AVG_CONFIDENCE_THRESHOLD=0.2):

        def proc(target_pos_lst, explored_points, curr_vertex, curr_pnt, next_point, next_pos, prepend_flag):
            new_vertex_flag, key_point_flag, end_flag, next_vertex, next_key_point = \
                self._follow_graph_one_step(
                    mode='push',
                    curr_pnt=curr_pnt, curr_rs=None,
                    next_point=next_point, next_pos=next_pos,
                    road_segmentation=road_segmentation,
                    origin_pnt=extension_vertex.point.sub(
                        geom.Point(WINDOW_SIZE//2, WINDOW_SIZE//2)),
                    explored_points=explored_points,
                    STEP_LENGTH=STEP_LENGTH, RECT_RADIUS=RECT_RADIUS, WINDOW_SIZE=WINDOW_SIZE)
            if next_vertex is not None and curr_vertex == next_vertex:
                return True, None
            # new_vertex_flag
            if new_vertex_flag == 0:
                next_vertex = self.generate_new_vertex(next_point)
            elif new_vertex_flag == 1:
                if next_vertex is None:
                    next_vertex = self.generate_new_vertex(next_key_point)
            elif new_vertex_flag == 2:
                pnt = next_vertex.point
                self.anchor_point_rtree.delete(
                    next_vertex.id, (pnt.x, pnt.y, pnt.x, pnt.y))
                self.anchor_point_rtree.insert(
                    next_vertex.id, (next_point.x, next_point.y, next_point.x, next_point.y))
                next_vertex.point = next_point
            # key_point_flag
            if prepend_flag is True or end_flag is True:
                self.prepend_search_vertex(
                    next_vertex, is_key_point=key_point_flag)
            if end_flag:
                key_pnts = get_points_from_rtree(point_rtree=self.key_point_rtree,
                                                 index2point=self.indexed_key_points, center_point=next_vertex.point, RECT_RADIUS=0)
                if len(key_pnts) == 0:
                    pnt = next_vertex.point
                    index = len(self.indexed_key_points)
                    self.key_point_rtree.insert(
                        index, (pnt.x, pnt.y, pnt.x, pnt.y))
                    self.indexed_key_points[index] = pnt
            # end_flag
            if self.is_training:
                rs, tp = self.get_rs_from_next_point(
                    target_pos_lst, next_point)
                if next_vertex.edge_pos is None:
                    next_vertex.edge_pos = tp
                if end_flag:
                    self.mark_rs_explored(rs=rs)
                else:
                    self.mark_rs_explored_part(rs=rs, edge_pos=tp)
            # add bidirectional edge
            self._add_bidirectional_edge(curr_vertex, next_vertex)
            if self.road_seg is not None:
                src = curr_vertex.point.sub(self.road_seg_origin)
                dst = next_vertex.point.sub(self.road_seg_origin)
                cv.line(self.road_seg, (src.y, src.x), (dst.y, dst.x),
                        color=0, thickness=self.ROADSEG_OVERWRITE_THICKNESS)
            explored_points.append(next_vertex.point)
            return end_flag, next_vertex

        if follow_mode == 'follow_target':
            next_points = [pos.point(
                self.gc.graph) for pos in target_poses.get_single_lst_without_junction_end()]
        elif follow_mode == 'follow_output':
            next_points = output_points
        else:
            raise NotImplementedError
        if len(next_points) == 0:
            return
        origin_point = extension_vertex.point.sub(
            geom.Point(WINDOW_SIZE//2, WINDOW_SIZE//2))

        if is_key_point:
            """
            check for loop pushing
            ---|---
               | /  <==
               |/   <==
            """
            if road_segmentation is not None:
                nearby_edge_segments = graph_helper.get_nearby_edge_segments(
                    extension_vertex, 1, self.graph)
                for pnt in next_points.copy():
                    for segment in nearby_edge_segments:
                        if segment.distance(pnt) < RECT_RADIUS and \
                                get_avg_between_pnts_in_map(
                                    im_map=road_segmentation,
                                    pnt1=segment.start.sub(origin_point),
                                    pnt2=segment.end.sub(origin_point),
                                    WINDOW_SIZE=WINDOW_SIZE
                        ) < AVG_CONFIDENCE_THRESHOLD:
                            next_points.remove(pnt)
                            break
            """
            check if end
            end_flag[]:
              1: not end;
              2: end at key points;
              3: end at anchor points
                (a)---+----
                      |
                   ==>
                      |
                      |
                (b)--------
                   ==>
                      |
            """
            explored_points = [x.point for x in graph_helper.get_nearby_vertices(
                extension_vertex, 2, self.graph)]
            for next_point in next_points:
                proc(target_pos_lst=target_poses[0] if target_poses is not None else None,
                     explored_points=explored_points,
                     curr_vertex=extension_vertex, curr_pnt=extension_vertex.point,
                     next_point=next_point, next_pos=None, prepend_flag=True)
        else:
            # cannot add edge between same vertex
            if len(next_points) == 1 and self.get_vertex_from_point_in_graph(next_points[0]) == extension_vertex:
                return
            # recurrent set curr_vertex
            curr_vertex = extension_vertex
            # -1: not end; >=0: end at key points; -2: end at anchor points
            # end_flag = False
            # explored_points = [extension_vertex.point]
            explored_points = [x.point for x in graph_helper.get_nearby_vertices(
                extension_vertex, 2, self.graph)]
            for i, next_point in enumerate(next_points):
                end_flag, next_vertex = proc(
                    target_pos_lst=target_poses[i] if target_poses is not None else None,
                    explored_points=explored_points,
                    curr_vertex=curr_vertex, curr_pnt=curr_vertex.point,
                    next_point=next_point, next_pos=None,
                    prepend_flag=True if i == len(next_points)-1 else False)
                if end_flag:
                    break
                # recurrent set curr_vertex
                curr_vertex = next_vertex

    def pop(self, follow_order=True, probs=[0.15, 0.8, 0.05], WINDOW_SIZE=256):
        """

        :param follow_order:
        :param probs: {
                "pop_unexplored_starting_point": 0.15,
                "pop_search_vertices": 0.8,
                "pop_random_starting_point": 0.05
            }
        :param WINDOW_SIZE:
        :return:
        """

        def _pop_search_vertices():
            if len(self.search_vertices) > 0:
                if follow_order:
                    _vertex, _is_key_point = self.search_vertices.pop()
                else:
                    _vertex, _is_key_point = self.search_vertices.pop(
                        random.randint(0, len(self.search_vertices)-1))
                return _vertex, _is_key_point
            return None, None

        def _pop_not_explored_starting_points():
            if len(self.not_explored_starting_points) > 0:
                index = random.randint(
                    0, len(self.not_explored_starting_points) - 1)
                start_loc = self.not_explored_starting_points.pop(index)
                if not self.is_training or start_loc[0]['key_point']:
                    _vertex = self.graph.add_vertex(start_loc[0]['point'])
                    _vertex.edge_pos = start_loc[0]['edge_pos']
                    _is_key_point = start_loc[0]['key_point']
                    return _vertex, True
                elif self.is_training:  # starting point at the middle of the road
                    split_point = start_loc[0]['point']
                    edge_pos = start_loc[0]['edge_pos']
                    opposite_edge = edge_pos.edge(self.gc.graph).get_opposite_edge(self.gc.graph)
                    edge_ids = [edge_pos.edge_id]
                    if opposite_edge is not None:
                        edge_ids.append(opposite_edge.id)

                    # for edge_id in [start_loc[0]['edge_pos'].edge_id, start_loc[0]['edge_pos'].edge(self.gc.graph).get_opposite_edge(self.gc.graph).id]:
                    for edge_id in edge_ids:
                        rs = self.gc.edge_id_to_rs(edge_id)
                        rs_exp = self.rs_exploration[rs.id]
                        rs_edges = rs.edges(self.gc.graph)
                        break_index = 0
                        for edge in rs_edges:
                            if edge.dst(self.gc.graph).point == split_point:
                                break_index = rs.edges_id.index(edge.id) + 1
                                break
                        if break_index == 0 or break_index == len(rs_edges):
                            return _pop_not_explored_starting_points()
                        if rs_exp.is_explored() or \
                                rs_exp.explored_start_dis + 1 >= \
                                rs.edge_distances[rs.edges_id[break_index]] \
                                or \
                                rs_exp.explored_end_dis - 1 <= \
                                rs.edge_distances[rs.edges_id[break_index]]:
                            return _pop_not_explored_starting_points()
                        new_rs = graph_helper.RoadSegment(
                            len(self.gc.road_segments))
                        new_rs.edges_id = rs.edges_id[break_index:]
                        rs.marked_length = rs.edge_distances[rs.edges_id[break_index]]
                        rs.edges_id = rs.edges_id[:break_index]
                        for k, v in rs.edge_distances.copy().items():
                            if k not in rs.edges_id:
                                del rs.edge_distances[k]
                        new_rs.compute_edge_distances(self.gc.graph)
                        for new_edge_id in new_rs.edges_id:
                            self.gc.edge_id_to_rs_id[new_edge_id] = new_rs.id
                        self.gc.road_segments.append(new_rs)
                        new_rs_exp = graph_helper.new_road_segment_exploration(
                            new_rs, self.gc.graph)
                        self.rs_exploration.data[new_rs.id] = new_rs_exp
                        new_rs_exp.explored_end_dis = new_rs.marked_length - \
                            (rs_exp.marked_length - rs_exp.explored_end_dis)
                        new_rs_exp.explored_end_pos = rs_exp.explored_end_pos
                        rs_exp.explored_end_pos = graph_helper.EdgePos(
                            rs.edges_id[-1], distance=self.gc.graph.edges[rs.edges_id[-1]].segment(self.gc.graph).length())
                        rs_exp.explored_end_dis = rs.marked_length
                        rs_exp.marked_length = rs.marked_length
                    _vertex = self.graph.add_vertex(start_loc[0]['point'])
                    _vertex.edge_pos = start_loc[0]['edge_pos']
                    return _vertex, True
            return None, None

        def _pop_road_seg_peak():
            if self.road_seg.max() > 0:
                peak = self.road_seg.argmax()
                peak = geom.Point(peak % self.tile_size, peak / self.tile_size)
                cv.circle(self.road_seg, (peak.x, peak.y),
                          radius=self.ROADSEG_OVERWRITE_RADIUS, color=0, thickness=-1)
                _vertex = graph_helper.Vertex(self.graph.vertices_index, peak.add(
                    self.tile_data['search_rect'].start))
                _vertex.edge_pos = None
                _vertex.from_road_seg = True
                _is_key_point = False
                return _vertex, _is_key_point
            return None, None

        def _pop_random_patch(max_iter=5):
            random_rect = get_random_rect_padding(
                self.tile_data['search_rect'], WINDOW_SIZE)
            small_rect = random_rect.add_tol(-WINDOW_SIZE//3)
            if len(self.gc.edge_index.search(small_rect)) > 0:
                if max_iter == 0:
                    return None, None
                return _pop_random_patch(max_iter-1)
            else:
                _vertex = graph_helper.Vertex(-1, random_rect.start.add(
                    geom.Point(WINDOW_SIZE//2, WINDOW_SIZE//2)))
                _vertex.edge_pos = None
                return _vertex, False

        vertex, is_key_point = None, None
        if follow_order:
            if len(self.search_vertices) > 0:
                vertex, is_key_point = _pop_search_vertices()
            elif len(self.not_explored_starting_points) > 0:
                vertex, is_key_point = _pop_not_explored_starting_points()
            elif self.road_seg is not None:
                vertex, is_key_point = _pop_road_seg_peak()
            return vertex, is_key_point
        else:
            choice = random_sample_given_probs(
                ["pop_unexplored_starting_point", "pop_search_vertices", "pop_random_starting_point"], probs=probs)
            if choice == "pop_unexplored_starting_point" and len(self.not_explored_starting_points) > 0:
                vertex, is_key_point = _pop_not_explored_starting_points()
            elif choice == "pop_search_vertices" and len(self.search_vertices) > 0:
                vertex, is_key_point = _pop_search_vertices()
            elif choice == "pop_random_starting_point":
                vertex, is_key_point = _pop_random_patch(max_iter=5)
            if vertex is None and (len(self.not_explored_starting_points) > 0 or len(self.search_vertices) > 0):
                if len(self.search_vertices) > 0:
                    vertex, is_key_point = _pop_search_vertices()
                if vertex is None:
                    vertex, is_key_point = _pop_not_explored_starting_points()
            return vertex, is_key_point

    def _follow_graph_one_step(self, mode, curr_pnt, curr_rs, next_point, next_pos, road_segmentation,
                               origin_pnt, explored_points=list(),
                               AVG_CONFIDENCE_THRESHOLD=0.2, STEP_LENGTH=20, RECT_RADIUS=10, WINDOW_SIZE=256):

        if mode == 'push':
            # 0: self.generate_new_vertex; 1: exist vertex; 2: move vertex
            new_vertex_flag = 0
            # False: anchor point; True: key point
            key_point_flag = False
            # False: part; True: end
            end_flag = False
            next_vertex = None
            next_key_point = None
        elif mode == 'pop':
            new_edge_pos = next_pos

        key_points = get_points_from_rtree(point_rtree=self.key_point_rtree,
                                           index2point=self.indexed_key_points, center_point=next_point, RECT_RADIUS=RECT_RADIUS)
        for pnt in explored_points:
            if pnt in key_points:
                del key_points[pnt]
        anchor_points = get_points_from_rtree(point_rtree=self.anchor_point_rtree,
                                              index2point=self.indexed_anchor_points, center_point=next_point, RECT_RADIUS=RECT_RADIUS)
        for pnt in explored_points:
            if pnt in anchor_points:
                del anchor_points[pnt]
        if len(anchor_points) == 0 and len(key_points) == 0:  # 0: not end
            if mode == 'push':
                new_vertex_flag = 0
                key_point_flag = False
                end_flag = False
            elif mode == 'pop':
                new_edge_pos = next_pos
        else:  # len(anchor_points) > 0 or len(key_points) > 0
            if len(anchor_points) > 0:
                anchor_pnt = get_nearest_end_point(
                    anchor_points.keys(), next_point)
                anchor_pnt_dis = anchor_pnt.distance(next_point)
            else:
                anchor_pnt_dis = math.inf
            if len(key_points) > 0:
                key_pnt = get_nearest_end_point(key_points.keys(), next_point)
                key_pnt_dis = key_pnt.distance(next_point)
            else:
                key_pnt_dis = math.inf
            if anchor_pnt_dis < key_pnt_dis:  # anchor_pnt
                avg_conf = get_avg_between_pnts_in_map(
                    im_map=road_segmentation,
                    pnt1=curr_pnt.sub(origin_pnt),
                    pnt2=anchor_pnt.sub(origin_pnt),
                    WINDOW_SIZE=WINDOW_SIZE)
                if anchor_pnt_dis > 8 and avg_conf < AVG_CONFIDENCE_THRESHOLD:  # do not select the anchor point
                    if mode == 'push':
                        new_vertex_flag = 0
                        key_point_flag = False
                        end_flag = False
                    elif mode == 'pop':
                        new_edge_pos = next_pos
                else:  # select the anchor point
                    end_flag = True
                    next_vertex = self.graph.vertices[anchor_points[anchor_pnt]]
                    if anchor_pnt_dis > 8 and len(next_vertex.in_edges_id) > 1:
                        if mode == 'push':
                            new_vertex_flag = 2
                            key_point_flag = True
                            end_flag = False
                        elif mode == 'pop':
                            new_edge_pos = curr_rs.closest_pos(
                                anchor_pnt, self.gc.graph)
                    else:  # case(a) or case(b)
                        if len(next_vertex.in_edges_id) > 1:
                            if mode == 'push':
                                new_vertex_flag = 1
                                key_point_flag = True
                                end_flag = False
                            elif mode == 'pop':
                                new_edge_pos = curr_rs.closest_pos(
                                    anchor_pnt, self.gc.graph)
                        else:
                            if mode == 'push':
                                new_vertex_flag = 1
                                key_point_flag = False
                                end_flag = False
                            elif mode == 'pop':
                                new_edge_pos = curr_rs.closest_pos(
                                    anchor_pnt, self.gc.graph)
            else:  # key_pnt
                avg_conf = get_avg_between_pnts_in_map(
                    im_map=road_segmentation,
                    pnt1=curr_pnt.sub(origin_pnt),
                    pnt2=key_pnt.sub(origin_pnt),
                    WINDOW_SIZE=WINDOW_SIZE)
                if key_pnt_dis > 8 and avg_conf < AVG_CONFIDENCE_THRESHOLD:  # do not select the key point
                    if mode == 'push':
                        new_vertex_flag = 0
                        key_point_flag = False
                        end_flag = False
                    elif mode == 'pop':
                        new_edge_pos = next_pos
                else:  # select the key point
                    if mode == 'push':
                        next_vertex = self.get_vertex_from_point_in_graph(
                            key_pnt)
                        if next_vertex is None:
                            new_vertex_flag = 0
                            key_point_flag = True
                            end_flag = True
                            next_key_point = key_pnt
                        else:
                            new_vertex_flag = 1
                            key_point_flag = True
                            end_flag = True
                    elif mode == 'pop':
                        # assert curr_rs.explored_end_pos.point() == key_pnt
                        new_edge_pos = curr_rs.closest_pos(
                            key_pnt, self.gc.graph)
        if mode == 'push':
            return new_vertex_flag, key_point_flag, end_flag, next_vertex, next_key_point
        elif mode == 'pop':
            return new_edge_pos

    def get_target_poses(self, extension_vertex, road_segmentation, STEP_LENGTH=20, is_key_point=False,
                         NUM_TARGETS=4, RECT_RADIUS=10, WINDOW_SIZE=256):
        """
        :return target_poses: [[], [], [], []]
        """

        def append_next_starting_pos(curr_rs, target_poses, curr_pnt, curr_vertex, explored_points, potential_rs_list):
            target_poses_len = len(target_poses)
            if target_poses_len >= NUM_TARGETS:
                return
            if potential_rs_list is None:
                potential_rs_list = []
                for edge in curr_vertex.out_edges(self.gc.graph):
                    next_rs = self.gc.edge_id_to_rs(edge.id)
                    if curr_rs is not None and (next_rs.id == curr_rs.id or next_rs.is_opposite(curr_rs, self.gc.graph)):
                        continue
                    next_rs_exp = self.rs_exploration[next_rs.id]
                    if next_rs_exp.is_explored() or next_rs_exp.explored_start_dis > 0:
                        continue
                    potential_rs_list.append(next_rs)
                # detect very short road segment
                for next_rs in potential_rs_list.copy():
                    if next_rs.marked_length < 5:
                        potential_rs_list.remove(next_rs)
                        for edge in next_rs.dst(self.gc.graph).out_edges(self.gc.graph):
                            next_next_rs = self.gc.edge_id_to_rs(edge.id)
                            if next_rs.id == next_next_rs.id or next_rs.is_opposite(next_next_rs, self.gc.graph):
                                continue
                            if self.rs_exploration[next_next_rs.id].is_explored():
                                continue
                            potential_rs_list.append(next_next_rs)
            for next_rs in potential_rs_list:
                next_rs_exp = self.rs_exploration[next_rs.id]
                # next_starting_pos = next_rs.closest_pos(curr_pnt)
                next_starting_pos = next_rs_exp.explored_start_pos
                # assert next_starting_pos.point(self.gc.graph) == curr_pnt  # TODO
                if self.is_explored_edge_pos(next_starting_pos):
                    continue
                if next_rs_exp.get_unexplored_dis() <= STEP_LENGTH * 1.5:
                    target_poses[target_poses_len].append(
                        next_rs_exp.explored_end_pos)
                    continue
                rs_follow_positions = self.gc.graph.follow_graph(
                    next_starting_pos, STEP_LENGTH)
                # assert len(rs_follow_positions) == 1 and \
                #     next_rs.id == self.gc.edge_id_to_rs_id[rs_follow_positions[0].edge.id].id
                next_pos = rs_follow_positions[0]
                new_edge_pos = self._follow_graph_one_step(
                    mode='pop',
                    curr_pnt=curr_pnt, curr_rs=next_rs,
                    next_point=next_pos.point(self.gc.graph), next_pos=next_pos,
                    road_segmentation=road_segmentation,
                    origin_pnt=extension_vertex.point.sub(
                        geom.Point(WINDOW_SIZE//2, WINDOW_SIZE//2)),
                    explored_points=explored_points,
                    STEP_LENGTH=STEP_LENGTH, RECT_RADIUS=RECT_RADIUS, WINDOW_SIZE=WINDOW_SIZE)
                if not self.tile_data['search_rect'].contains(new_edge_pos.point(self.gc.graph)):
                    continue
                if new_edge_pos.point(self.gc.graph) not in [x.point(self.gc.graph) for x in target_poses[target_poses_len]]:
                    # prevent same point but different rs, just waste one
                    target_poses[target_poses_len].append(new_edge_pos)
                explored_points.append(new_edge_pos.point(self.gc.graph))

        target_poses = TargetPosesContainer(NUM_TARGETS)
        if extension_vertex.edge_pos is None:
            return target_poses

        # avoid getting into another road
        if extension_vertex.edge_pos.point(self.gc.graph).distance(extension_vertex.point) > 2 * STEP_LENGTH:
            # map_match_pos()
            # if extension_vertex.edge_pos is None:
            #     return target_poses
            return target_poses

        curr_edge = extension_vertex.edge_pos.edge(self.gc.graph)
        curr_rs = self.gc.edge_id_to_rs(curr_edge.id)

        # if at key point, target_poses contains position of next points in different road segment with one time step
        # else, target_poses contains position of next points in one road segment with num_targets time step

        if is_key_point:  # more than one rs, just one time step
            extension_vertex_gt_graph = None
            potential_rs_list = None
            if extension_vertex.point == curr_edge.src(self.gc.graph).point:
                extension_vertex_gt_graph = curr_edge.src(self.gc.graph)
            elif extension_vertex.point == curr_edge.dst(self.gc.graph).point:
                extension_vertex_gt_graph = curr_edge.dst(self.gc.graph)
            else:
                # walk into a viaduct which do not cross but look like a key point in 2D
                potential_rs_list = [curr_rs]

            explored_points = [x.point for x in graph_helper.get_nearby_vertices(
                extension_vertex, 2, self.graph)]
            append_next_starting_pos(curr_rs=None, target_poses=target_poses,
                                     curr_pnt=extension_vertex.edge_pos.point(
                                         self.gc.graph),
                                     curr_vertex=extension_vertex_gt_graph,
                                     explored_points=explored_points, potential_rs_list=potential_rs_list)
        else:  # only one rs, more than one time step
            rs = curr_rs
            rs_exp = self.rs_exploration[rs.id]
            curr_pos = extension_vertex.edge_pos
            explored_points = [x.point for x in graph_helper.get_nearby_vertices(
                extension_vertex, 2, self.graph)]
            if not self.is_explored_edge_pos(curr_pos):
                for i in range(NUM_TARGETS):
                    if rs_exp.get_unexplored_dis() - i * STEP_LENGTH <= STEP_LENGTH * 1.5:
                        target_poses[i].append(rs_exp.explored_end_pos)
                        if rs_exp.explored_end_dis >= rs_exp.marked_length - 1:
                            append_next_starting_pos(
                                rs, target_poses, curr_pnt=rs.dst(
                                    self.gc.graph).point,
                                curr_vertex=rs.dst(self.gc.graph), explored_points=explored_points,
                                potential_rs_list=None)
                        break
                    rs_follow_positions = self.gc.graph.follow_graph(
                        extension_vertex.edge_pos, STEP_LENGTH * (i+1))
                    next_pos = rs_follow_positions[0]
                    new_edge_pos = self._follow_graph_one_step(
                        mode='pop',
                        curr_pnt=curr_pos.point(self.gc.graph), curr_rs=rs,
                        next_point=next_pos.point(self.gc.graph), next_pos=next_pos,
                        road_segmentation=road_segmentation,
                        origin_pnt=extension_vertex.point.sub(
                            geom.Point(WINDOW_SIZE//2, WINDOW_SIZE//2)),
                        explored_points=explored_points,
                        STEP_LENGTH=STEP_LENGTH, RECT_RADIUS=RECT_RADIUS, WINDOW_SIZE=WINDOW_SIZE)
                    target_poses[i].append(new_edge_pos)
                    explored_points.append(new_edge_pos.point(self.gc.graph))
                    curr_pos = new_edge_pos
        return target_poses

    def make_path_input(self, extension_vertex, fetch_list, traj_filter, is_key_point=False, WINDOW_SIZE=256):
        """
        :param extension_vertex: 
        :param fetch_list: 
            'aerial_image_chw':
            'aerial_image_hwc':
            'traj_image_hwc':
            'traj_image_chw':
            'walked_path_small':
            'road_seg_small':
            'road_seg_thick3':
            'junc_seg_small':
        :param is_key_point: 
        :param WINDOW_SIZE:
        :return:
        """
        search_rect = self.tile_data['search_rect']
        big_origin = search_rect.start  # (0,0) 这里也对应前面 不是从0 而是4096开始

        # 遥感图像是通过tile_data['cache']来访问的
        big_img = self.tile_data['cache'].get(
            self.tile_data['region'], search_rect)
        # {'input': img.shape==(4096,4096,3)}
        need_traj_image = 'traj_image_hwc' in fetch_list or 'traj_image_chw' in fetch_list
        big_traj_img = None
        if need_traj_image:
            big_traj_img = self.tile_data['cache'].get_traj(
                self.tile_data['region'], search_rect)

        if not search_rect.contains(extension_vertex.point):
            # (top_left:(128, 128), buttom_right:(1920, 1920))
            raise Exception('bad path {}'.format(self))
        origin = extension_vertex.point.sub(
            geom.Point(WINDOW_SIZE // 2, WINDOW_SIZE // 2))
        tile_origin = origin.sub(big_origin).add(
            geom.Point(WINDOW_SIZE // 2, WINDOW_SIZE // 2))
        rect = origin.bounds().extend(origin.add(geom.Point(WINDOW_SIZE, WINDOW_SIZE)))
        safe_rect = search_rect.add_tol(-WINDOW_SIZE // 2)

        ################ walked_path ################
        walked_path_small = None
        walked_path = None
        if 'walked_path_small' in fetch_list or 'walked_path' in fetch_list:
            walked_path_small = np.zeros(
                (WINDOW_SIZE // 4, WINDOW_SIZE // 4), dtype=np.float32)
            walked_path = np.zeros(
                (WINDOW_SIZE, WINDOW_SIZE), dtype=np.float32)
            for edge_id in self.edge_rtree.intersection((rect.start.x, rect.start.y, rect.end.x, rect.end.y)):
                edge = self.graph.edges[edge_id]
                start = edge.src(self.graph).point.sub(origin)
                end = edge.dst(self.graph).point.sub(origin)
                cv.line(walked_path_small, (start.y // 4, start.x // 4),
                        (end.y // 4, end.x // 4), 1, 1)
                cv.line(walked_path, (start.y, start.x),
                        (end.y, end.x), 1, 1)
            # 中心点设置为1
            walked_path_small[WINDOW_SIZE // 8, WINDOW_SIZE // 8] = 1.0
            walked_path[WINDOW_SIZE // 2, WINDOW_SIZE // 2] = 1.0

        ################ road_seg & junc_seg ################
        road_seg_small = junc_seg_small = None
        if 'road_seg_small' in fetch_list or 'junc_seg_small' in fetch_list or 'road_seg_thick3' in fetch_list:
            seg_rect = rect
            seg_origin = seg_rect.start
            if self.is_training:
                road_seg_small = np.zeros(
                    (WINDOW_SIZE // 4, WINDOW_SIZE // 4), dtype=np.float32)
                junc_seg_small = np.zeros(
                    (WINDOW_SIZE // 4, WINDOW_SIZE // 4), dtype=np.float32)
                road_seg_thick3 = np.zeros((WINDOW_SIZE, WINDOW_SIZE), dtype=np.float32)
                junc_seg_thick3 = np.zeros((WINDOW_SIZE, WINDOW_SIZE), dtype=np.float32)

                for edge in self.gc.edge_index.search(seg_rect):
                    if 'junc_seg_small' in fetch_list or 'junc_seg_thick3' in fetch_list:
                        for vertex in [edge.src(self.gc.graph), edge.dst(self.gc.graph)]:
                            pnt = vertex.point
                            if len(vertex.out_edges(self.gc.graph)) > 2 and seg_rect.contains(pnt) and search_rect.contains(pnt):
                                pnt = pnt.sub(seg_origin)
                                cv.circle(junc_seg_small, (pnt.y // 4, pnt.x // 4), radius=2, color=1, thickness=-1)
                                cv.circle(junc_seg_thick3, (pnt.y, pnt.x), radius=5, color=1, thickness=-1)
                    if 'road_seg_small' in fetch_list or 'road_seg_thick3' in fetch_list:
                        start = edge.src(self.gc.graph).point
                        end = edge.dst(self.gc.graph).point
                        if search_rect.contains(start) or search_rect.contains(end):
                            start = start.sub(seg_origin)
                            end = end.sub(seg_origin)
                            cv.line(road_seg_small, (start.y // 4, start.x // 4), (end.y // 4, end.x // 4), color=1, thickness=1)
                            cv.line(road_seg_thick3, (start.y, start.x), (end.y, end.x), color=1, thickness=5)
                if not safe_rect.contains(extension_vertex.point):
                    clip_rect = search_rect.clip_rect(seg_rect)
                    start, end = clip_rect.start.sub(
                        seg_origin), clip_rect.end.sub(seg_origin)
                    new_road_seg_small = np.zeros(
                        (WINDOW_SIZE // 4, WINDOW_SIZE // 4), dtype=np.float32)
                    new_road_seg_small[start.x // 4:end.x // 4, start.y // 4:end.y //
                                       4] = road_seg_small[start.x // 4:end.x // 4, start.y // 4:end.y // 4]
                    del road_seg_small
                    road_seg_small = new_road_seg_small

        aerial_image_hwc = big_img['input'][tile_origin.x:tile_origin.x + WINDOW_SIZE,
                                            tile_origin.y:tile_origin.y + WINDOW_SIZE, :].astype('float32') / 255.0
        aerial_image_chw = aerial_image_hwc.swapaxes(0, 2).swapaxes(1, 2)

        traj_image_hwc = None
        traj_image_chw = None
        if need_traj_image:
            traj_image_hwc = big_traj_img['input'][tile_origin.x:tile_origin.x + WINDOW_SIZE,
                                                tile_origin.y:tile_origin.y + WINDOW_SIZE, :].astype('float32') / 255.0
            traj_image_chw = traj_image_hwc.swapaxes(0, 2).swapaxes(1, 2)

        # ################ traj in pieces ################
        if 'valid_trajectories' in fetch_list:
            # 存储通过过滤器的邻域轨迹点
            num_circles = 8
            # # 创建多个小圆圈，这些圆圈的圆心为待处理节点
            self.circles = []
            circle_radius = 20
            neighborhood_radius = 50

            # # # 计算每个圆圈的中心位置（以目标节点为圆心，间隔一定角度的圆）
            for i in range(num_circles):
                angle = 2 * np.pi * i / num_circles  # 分布在360度范围内
                circle_center = extension_vertex.point.add(geom.Point(neighborhood_radius * np.cos(angle), neighborhood_radius * np.sin(angle)))
                circle = {'center': circle_center, 'radius': circle_radius}
                self.circles.append(circle)

            if traj_filter:
                GPU_valid_trajectories2 = self.filter_trajectories_on_gpu2(rect, extension_vertex)
                if GPU_valid_trajectories2.size(0) > 100:
                    GPU_valid_trajectories2 = GPU_valid_trajectories2[:100, :, :]
                # (n, win_len, 2) 当没有有效轨迹rect_trajectory_segments2为空时返回占位向量n=1
                self.valid_trajectories = GPU_valid_trajectories2
            else:
                GPU_valid_trajectories2 = self.filter_trajectories_on_gpu2(rect, extension_vertex)
                if GPU_valid_trajectories2.size(0) > 100:
                    GPU_valid_trajectories2 = GPU_valid_trajectories2[:100, :, :]
                self.valid_trajectories = GPU_valid_trajectories2

        ret_dict = {
            'aerial_image_chw':  aerial_image_chw if 'aerial_image_chw' in fetch_list else None,
            'aerial_image_hwc':  aerial_image_hwc if 'aerial_image_hwc' in fetch_list else None,
            'traj_image_chw':    traj_image_chw if 'traj_image_chw' in fetch_list else None,
            'traj_image_hwc':    traj_image_hwc if 'traj_image_hwc' in fetch_list else None,
            'walked_path_small': walked_path_small[np.newaxis, :, :] if 'walked_path_small' in fetch_list else None,
            'walked_path':       walked_path[np.newaxis, :, :] if 'walked_path' in fetch_list else None,
            'road_seg_small':    road_seg_small[np.newaxis, :, :] if 'road_seg_small' in fetch_list else None,
            'road_seg_thick3':   road_seg_thick3[np.newaxis, :, :] if 'road_seg_thick3' in fetch_list else None,
            'junc_seg_small':    junc_seg_small[np.newaxis, :, :] if 'junc_seg_small' in fetch_list else None,
            'junc_seg_thick3':   junc_seg_thick3[np.newaxis, :, :] if 'road_seg_thick3' in fetch_list else None,
            'valid_trajectories': self.valid_trajectories if 'valid_trajectories' in fetch_list else None,
        }

        return ret_dict

    def visualize_output(self, fname_prefix, extension_vertex, aerial_image, target_poses=None, pred_gt_pair_list=None,
                         WINDOW_SIZE=256):
        """
        :param
            aerial_image: PIL
            pred_gt_pair_list: list of tuples of predicted maps and ground truth maps
                [('anchor', anchor_output_map, anchor_target_map),  # anchor maps must be at the first index
                 ('road', road_segmentation_output_map, road_segmentation_target_map),
                 ('junc', junc_segmentation_output_map, junc_segmentation_output_map),
                 ('res_seg', residual_road_segment_segmentation_output_map, residual_road_segment_segmentation_target_map)]
            target_poses: TargetPosesContainer

        """
        ################ aerial_image ################
        if aerial_image is not None:
            aerial_image = aerial_image.swapaxes(0, 1)

        if self.gc is not None:
            if aerial_image is None:
                aerial_image = np.zeros((WINDOW_SIZE, WINDOW_SIZE, 3))

            # 将其转化为连续数组，有啥用？
            aerial_image = np.ascontiguousarray(aerial_image)
            explored_edge = []
            # 绘制原点，通常用于将绘图的原点移到窗口的中心
            # origin:  Point(4466, 4092) 4466 4092
            origin = extension_vertex.point.sub(
                geom.Point(WINDOW_SIZE // 2, WINDOW_SIZE // 2))

            # Legacy trajectory overlays are optional. Image-only paths never
            # request trajectory filtering, so both collections remain empty.
            valid_trajectories = self.valid_trajectories
            circles = getattr(self, 'circles', [])
            for trajectory in valid_trajectories:
                for i in range(len(trajectory) - 1):
                    start = geom.Point(trajectory[i][0],trajectory[i][1]).sub(origin)  # 转为窗口坐标
                    end = geom.Point(trajectory[i+1][0], trajectory[i+1][1]).sub(origin)  # 转为窗口坐标
                    # cv.line(aerial_image, (start.x, start.y),
                    #          (end.x, end.y), color=(1., 1., 1.), thickness=1)
                    cv.circle(aerial_image, (start.x, start.y),color=(1., 1., 1.), radius=1)
                    # print('valid_trajectories:', start)
            # The radius-50 marker belongs to the legacy trajectory filter and
            # must not appear in image-only visualizations.
            if len(valid_trajectories) > 0 or len(circles) > 0:
                cv.circle(aerial_image, (WINDOW_SIZE // 2, WINDOW_SIZE // 2), 50, (1., 0., 0.), 1)
            if len(circles) > 0:
                for i, circle in enumerate(circles):
                    circle_center = circle['center'].sub(origin)
                    circle_radius = circle['radius']
                    cv.circle(aerial_image, (circle_center.x, circle_center.y), circle_radius, (1., 0., 0.), 1)
                    # print('circle:', circle_center)

            # draw road segments, green 这里可视化的应该是真值
            # 绘制区域
            rect = origin.bounds().extend(origin.add(geom.Point(WINDOW_SIZE, WINDOW_SIZE)))

            for edge in self.gc.edge_index.search(rect):
                if edge in explored_edge:
                    continue
                explored_edge.append(edge)
                explored_edge.append(edge.get_opposite_edge(self.gc.graph))

                # 获取边的源点和目标点，并减去原点 origin，得到实际绘制时需要使用的坐标（为什么要减去？？？）
                start = edge.src(self.gc.graph).point.sub(origin)
                end = edge.dst(self.gc.graph).point.sub(origin)
                cv.line(aerial_image, (start.x, start.y),
                        (end.x, end.y), color=(0., 1., 0.), thickness=2)


            # draw intersections 紫色
            # 这里的像素坐标很小point(10,43) 是因为这个图维度是64*64？
            for edge in self.gc.edge_index.search(rect):
                start = edge.src(self.gc.graph)
                end = edge.dst(self.gc.graph)
                for pnt in [start, end]:
                    # if len(pnt.in_edges) >= 3:
                    if True:  # TODO
                        pnt = pnt.point.sub(origin)
                        cv.circle(aerial_image, center=(pnt.x, pnt.y),
                                  radius=3, color=(1., 0., 1.), thickness=-1)

            # draw already walked path, red
            # 没看懂这个已经探索过的是怎么记录的
            explored_edge = []
            for edge_id in self.edge_rtree.intersection((rect.start.x, rect.start.y, rect.end.x, rect.end.y)):
                edge = self.graph.edges[edge_id]
                if edge in explored_edge:
                    continue
                explored_edge.append(edge)
                explored_edge.append(edge.get_opposite_edge(self.graph))
                start = edge.src(self.graph).point.sub(origin)
                end = edge.dst(self.graph).point.sub(origin)
                cv.line(aerial_image, (start.x, start.y),
                        (end.x, end.y), color=(1., 0., 0.), thickness=1)

            # draw extension vertex point, half WINDOW_SIZE, blue
            cv.circle(aerial_image, center=(WINDOW_SIZE // 2, WINDOW_SIZE //
                                            2), radius=3, color=(0., 0., 1.), thickness=-1)

            # draw target position
            if target_poses is not None:
                if extension_vertex.edge_pos is not None:
                    # 如果 extension_vertex 的 edge_pos 不为 None，
                    # 则表示扩展顶点有一个相关的边位置
                    # pp 变量通过调用 edge_pos.point(self.gc.graph) 来获取该边的位置

                    # point 方法的作用是根据给定的边或线段，计算从起始点出发在沿着线段方向的某个特定距离处的点。
                    # 如果线段非常短（小于 1），则返回起始点；
                    # 否则，它返回一个沿着线段方向、距离起始点 self.distance 的新点
                    pp = extension_vertex.edge_pos.point(
                        self.gc.graph).sub(origin)
                    # extension vertex's edge_pos, cyan, "青色"
                    cv.circle(aerial_image, (pp.x, pp.y), radius=2,
                              color=(0., 1., 1.), thickness=-1)
                for p in target_poses.get_single_lst():
                    pp = p.point(self.gc.graph).sub(origin)
                    # target points next step, white
                    cv.circle(aerial_image, (pp.x, pp.y), radius=2,
                              color=(1., 1., 1.), thickness=-1)

        if aerial_image is not None:
            Image.fromarray((aerial_image * 255.0).astype('uint8')
                            ).save(fname_prefix + 'ai.png')

        ################ anchor_output_map ################

        # pred_gt_pair_list =
        # [
        #     ("anchor", res_dict.batch_output_anchor_maps[x], res_dict.batch_target_maps[x]),
        #     ("road", res_dict.batch_output_road[x, 0], res_dict.batch_road_segmentation[x, 0]),
        #     ("junc", res_dict.batch_output_junc[x, 0], res_dict.batch_junction_segmentation[x, 0])
        # ])

        anchor_fname_suffix, anchor_output_map, anchor_target_map = pred_gt_pair_list.pop(0)
        # 应该是将多个锚点的图进行合并
        anchor_output_map = np.sum(anchor_output_map, axis=0)
        if anchor_target_map is not None:
            anchor_target_map = np.sum(anchor_target_map, axis=0)
        pred_gt_pair_list.append(
            (anchor_fname_suffix, anchor_output_map, anchor_target_map))

        ################ pred_gt_pair_list ################

        for fname_suffix, output_map, target_map in pred_gt_pair_list:
            if target_map is not None:
                # print(output_map.shape, target_map.shape)
                # 为啥 为啥要把输出和gt拼接？？
                res = np.stack(
                    [output_map, target_map, np.zeros(output_map.shape)], axis=-1)
                # print(res.shape)
                Image.fromarray((res * 255.0).swapaxes(0, 1).astype('uint8')
                                ).save(fname_prefix + fname_suffix + '.png')
            else:
                Image.fromarray((output_map * 255.0).swapaxes(0, 1).astype('uint8')).save(
                    fname_prefix + fname_suffix + '.png')

    def visualize_and_save_path(self, i, tile_data, save_path, img=None):
        """
        可视化路径，绘制节点和边到图像上，并保存图像
        :param img: 背景图像，如果为空则创建一个空白图像
        :param save_path: 保存路径图像的文件路径
        :param window_size: 窗口大小，用于限定区域

        将某个region大图的四个2048小图中的路径进行可视化
        :return: 绘制的图像
        """

        fold = './data_self/coincidence_test/'
        save_path = os.path.join(fold, save_path + '_' + str(i) + ".png")

        # 如果没有传入图像，则创建一个空白图像
        if img is None:
            tile_size = self.tile_size
            img = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)

        origin = tile_data['search_rect'].start

        # 遍历所有边，并在图像上绘制
        for edge in self.gc.edge_index.search(tile_data['search_rect']):
            src_point = edge.src(self.gc.graph).point.sub(origin)
            dst_point = edge.dst(self.gc.graph).point.sub(origin)

            # 转换坐标到图像坐标系（例如从真实坐标映射到像素坐标）
            start_pixel = src_point.x, src_point.y
            end_pixel = dst_point.x, dst_point.y
            # print("start_end_pixel: ", start_pixel, end_pixel)
            # 在图像上绘制边
            cv.line(img, start_pixel, end_pixel, (0, 255, 0), thickness=2)

        # 保存图像到指定路径
        cv.imwrite(save_path, img)
        print(f"visualize_and_save_path 图像已保存至: {save_path}")
        return img

    def generate_target_maps(self, extension_vertex, target_poses, NUM_TARGETS=4, WINDOW_SIZE=224, is_key_point=False):
        """
        :param target_poses [[], [], [], []]
        :return: ndarray (NUM_TARGETS, WINDOW_SIZE, WINDOW_SIZE)
        """

        def generate_target(target_pnts, image_shape, target_shape):
            """
            :param joints:  [num_joints, 3]
            :return: target, target_weight(1: visible, 0: invisible)
            """
            sigma = 3
            num_joints = len(target_pnts)
            heatmap_size = np.array(target_shape)
            image_size = np.array(image_shape)

            target_weight = np.ones((num_joints, 1), dtype=np.float32)
            # target_weight[:, 0] = joints_vis[:, 0]

            target = np.zeros((num_joints,
                               heatmap_size[1],
                               heatmap_size[0]),
                              dtype=np.float32)

            tmp_size = sigma * 3

            for joint_id in range(num_joints):
                feat_stride = image_size / heatmap_size
                mu_x = int(target_pnts[joint_id].x / feat_stride[0] + 0.5)
                mu_y = int(target_pnts[joint_id].y / feat_stride[1] + 0.5)
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if ul[0] >= heatmap_size[0] or ul[1] >= heatmap_size[1] \
                        or br[0] < 0 or br[1] < 0:
                    # If not, just return the image as is
                    target_weight[joint_id] = 0
                    continue

                # # Generate gaussian
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2
                # The gaussian is not normalized, we want the center value to equal 1
                g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) /
                           (2 * sigma ** 2))
                # print(g)

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], heatmap_size[1]) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], heatmap_size[1])

                v = target_weight[joint_id]
                if v > 0.5:
                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

            return target, target_weight

        origin_point = extension_vertex.point.sub(
            geom.Point(WINDOW_SIZE // 2, WINDOW_SIZE // 2))
        target_maps = []
        for poses in target_poses:
            poses_len = len(poses)
            if poses_len == 1:
                lst = [pos.point(self.gc.graph).sub(origin_point)
                       for pos in poses]
                target, _ = generate_target(
                    lst, (WINDOW_SIZE, WINDOW_SIZE), (WINDOW_SIZE, WINDOW_SIZE))
                target_maps.append(target[0].swapaxes(0, 1))
            elif poses_len > 1:
                lst = [pos.point(self.gc.graph).sub(origin_point)
                       for pos in poses]
                target, _ = generate_target(
                    lst, (WINDOW_SIZE, WINDOW_SIZE), (WINDOW_SIZE, WINDOW_SIZE))
                target_map = np.sum(target, axis=0).swapaxes(0, 1)
                target_map[np.where(target_map > 1)] = 1
                target_maps.append(target_map)
            else:  # poses_len == 0:
                target_maps.append(np.zeros((WINDOW_SIZE, WINDOW_SIZE)))
        target_maps = np.stack(target_maps, axis=0)
        return target_maps

    def alltraj_to_allpixeltraj(self):
        """
        将csv文件中存储的实际经纬度坐标转换为像素坐标
        """
        all_pixel_trajectories = []

        for _, traj in enumerate(tqdm(self.all_trajectories, desc="trans all traj to all pixel traj")):
            converter = GisToGraphConverter(self.tile_data['region'], traj)
            pixel_trajectories = converter.convert_trajectories_to_pixels()
            all_pixel_trajectories.append(pixel_trajectories)
        return all_pixel_trajectories
        all_pixel_trajectories = np.array(all_pixel_trajectories)

        return all_pixel_trajectories

    # def filter_trajectories_on_gpu(self, rect, extension_vertex, device="cuda"):
    #     start = time.time()
    #
    #     # GPU 参数
    #     window_size = 10
    #     num_circles = 8
    #
    #     circle_radius = config.circle_radius.to(device)
    #     neighborhood_radius = config.neighborhood_radius.to(device)
    #
    #     print(config.circle_radius.is_leaf)  # 输出应为True
    #     print(circle_radius.is_leaf)  # 输出应为False
    #     print(circle_radius.grad_fn)  # 显示ToDevice操作节点
    #
    #     # 强制转换vertex坐标到GPU张量
    #     ext_x = torch.tensor(extension_vertex.point.x, device=device, dtype=torch.float32)
    #     ext_y = torch.tensor(extension_vertex.point.y, device=device, dtype=torch.float32)
    #
    #     # 生成圆心信息
    #     # TODO 生成小波式多尺度滤波器组 (参考小波散射网络思想) ds起的名字还挺高端
    #     angles = torch.linspace(0, 2 * torch.pi, steps=num_circles, device=device)
    #     circle_centers = torch.stack([
    #         ext_x + neighborhood_radius * torch.cos(angles),
    #         ext_y + neighborhood_radius * torch.sin(angles)
    #     ], dim=1).to(device)
    #
    #     # 遍历 rect 范围内的轨迹数据
    #     rect_trajectory_segments2 = []
    #     for traj_idx, trajectory in enumerate(self.all_pixel_trajectories_gpu):
    #         if len(trajectory) > 0:
    #             traj_gpu = torch.as_tensor(trajectory, device=device, dtype=torch.float32)
    #             mask = ((traj_gpu[:, 0] >= rect.start.x) & (traj_gpu[:, 0] <= rect.end.x) &
    #                     (traj_gpu[:, 1] >= rect.start.y) & (traj_gpu[:, 1] <= rect.end.y))
    #             if True in mask:
    #                 rect_trajectory_segments2.append(traj_gpu[mask])
    #     # 初始化存储
    #     circle_trajectories = {i: [] for i in range(num_circles)}
    #     trajectory_passes = torch.zeros(num_circles, device=device)
    #
    #     for trajectory in rect_trajectory_segments2:
    #         if trajectory.shape[0] == 0:
    #             continue
    #
    #         for i in range(num_circles):
    #             # 逐点计算距离
    #             # TODO 本来简单计算结果ds说可能会梯度断裂
    #             distances = torch.sqrt(
    #                 (trajectory[:, 0] - circle_centers[i, 0]) ** 2 +
    #                 (trajectory[:, 1] - circle_centers[i, 1]) ** 2
    #             )
    #             inside_circle = distances <= circle_radius
    #             points_inside = torch.where(inside_circle)[0]
    #
    #             last_processed_idx = -1
    #             for idx in points_inside:
    #                 if idx <= last_processed_idx:
    #                     continue
    #
    #                 start_idx = max(0, idx - window_size)
    #                 end_idx = min(trajectory.size(0), idx + window_size + 1)
    #                 segment = trajectory[start_idx:end_idx]
    #                 if len(segment) > 0:
    #                     circle_trajectories[i].append(segment)
    #                 last_processed_idx = end_idx - 1
    #             trajectory_passes[i] += len(points_inside)
    #
    #     max_passed_circle_idx = torch.argmax(trajectory_passes).item()
    #     if torch.max(trajectory_passes) > 0:
    #         print(max_passed_circle_idx)
    #     # print(f"GPU filtering completed in {time.time() - start:.2f} seconds.")
    #     return circle_trajectories[max_passed_circle_idx]

    def _candidate_trajectories_for_rect(self, rect, device="cuda"):
        if not self.use_traj_grid_index:
            return self.all_pixel_trajectories_gpu

        cell_size = self.traj_grid_cell_size
        min_cx = int(math.floor(rect.start.x / cell_size))
        max_cx = int(math.floor(rect.end.x / cell_size))
        min_cy = int(math.floor(rect.start.y / cell_size))
        max_cy = int(math.floor(rect.end.y / cell_size))

        candidate_ids = set()
        for cx in range(min_cx, max_cx + 1):
            for cy in range(min_cy, max_cy + 1):
                ids = self.traj_grid_index.get((cx, cy))
                if ids is not None:
                    candidate_ids.update(int(i) for i in ids)

        if len(candidate_ids) == 0:
            return []

        candidate_trajectories = []
        for traj_id in candidate_ids:
            if traj_id >= len(self.all_pixel_trajectories):
                continue
            traj = self.all_pixel_trajectories[traj_id]
            if len(traj) > 0:
                candidate_trajectories.append(
                    torch.as_tensor(traj, device=device, dtype=torch.float32))
        return candidate_trajectories

    def filter_trajectories_on_gpu2(self, rect, extension_vertex, device="cuda"):
        # GPU 参数

        window_size = 5
        num_circles = 4
        win_len = window_size * 2 + 1
        empty_result = torch.zeros(1, win_len, 2, device=device)
        candidate_trajectories = self._candidate_trajectories_for_rect(rect, device=device)
        if len(candidate_trajectories) == 0:
            return empty_result

        circle_radius = config.circle_radius
        neighborhood_radius = config.neighborhood_radius

        # 生成圆心信息
        # TODO 生成小波式多尺度滤波器组 (参考小波散射网络思想) ds起的名字还挺高端
        angles = torch.linspace(0, 2 * torch.pi, steps=num_circles, device=device)
        ext_x = torch.tensor(extension_vertex.point.x, device=device, dtype=torch.float32)
        ext_y = torch.tensor(extension_vertex.point.y, device=device, dtype=torch.float32)
        circle_centers = torch.stack([ext_x + neighborhood_radius * torch.cos(angles), ext_y + neighborhood_radius * torch.sin(angles)], dim=1)

        # 掩码+索引法 完成遍历 rect 范围内的轨迹数据
        traj_meta = [(i, len(traj)) for i, traj in enumerate(candidate_trajectories) if len(traj) > 0]
        if len(traj_meta) == 0:
            return empty_result
        indices = torch.cat([torch.full((length,), fill_value=i, device=device) for i, length in traj_meta])
        # 扁平化所有轨迹点
        # all_points = torch.cat([torch.as_tensor(traj, device=device) for traj in self.all_pixel_trajectories_gpu if len(traj) > 0])
        all_points = torch.cat([traj for traj in candidate_trajectories if len(traj) > 0])

        # 向量化过滤，判断哪些轨迹点在rect范围内
        rect_mask = (all_points[:, 0] >= rect.start.x) & (all_points[:, 0] <= rect.end.x) & (all_points[:, 1] >= rect.start.y) & (all_points[:, 1] <= rect.end.y)
        # 所有在ext为中心矩形框内的轨迹点及其索引
        valid_points = all_points[rect_mask]
        valid_indices = indices[rect_mask]
        if valid_points.size(0) == 0:
            return empty_result

        rect_trajectory_segments2 = []
        for traj_idx in torch.unique(valid_indices):
            mask1 = (valid_indices == traj_idx)
            if valid_points[mask1].size(0) >= win_len:
                rect_trajectory_segments2.append(valid_points[mask1])
        if len(rect_trajectory_segments2) == 0:
            return empty_result

        # 使用 pad_sequence 将轨迹补齐到同一长度(轨迹对齐)
        # padded_segments: [B, T, 2]，B:轨迹数量，T:最大轨迹长度
        padded_segments = pad_sequence(rect_trajectory_segments2, batch_first=True, padding_value=0)
        B, T, _ = padded_segments.shape
        # 构造 mask，指示哪些点是真实数据（True）而哪些是填充值（False）
        lengths = torch.tensor([seg.size(0) for seg in rect_trajectory_segments2], device=device)
        mask2 = torch.arange(T, device=device).expand(B, T) < lengths.unsqueeze(1)  # [B, T]

        # 对 padded_segments 应用 unfold 操作，窗口大小为 (window_size*2 + 1)

        # 轨迹太短时进行填充（保证滑动窗口长度）
        if T < win_len:
            pad_amount = win_len - T
            padded_segments = F.pad(padded_segments, (0, 0, 0, pad_amount), value=0)
            mask2 = F.pad(mask2, (0, pad_amount), value=False)  # [B, T + pad_amount]
        # all_segments: [B, S, win_len, 2]，其中 S = T - win_len + 1
        all_segments = padded_segments.unfold(1, win_len, 5).permute(0, 1, 3, 2)
        # 对 mask 也进行 unfold，得到每个窗口的有效性 mask，形状 [B, S, win_len]
        mask_unfold = mask2.unfold(1, win_len, 5)
        # valid_window_mask: [B, S]，只有当窗口内所有点都是真实数据时，才为 True
        valid_window_mask = (mask_unfold.sum(dim=2) == win_len)

        # 计算每个窗口中各圆的激活得分
        B_S = all_segments.size(0) * all_segments.size(1)
        all_segments_flat = all_segments.contiguous().view(B_S, win_len, 2) # 展开为 [B*S, win_len, 2]
        # 计算每个窗口中每个点与各圆心的距离（填充的轨迹点置为inf，后续计算sigmoid归0）
        distances = torch.cdist(all_segments_flat, circle_centers) # [B*S, win_len, 8]
        # 使用距离的倒数（距离越小得分越大），而不是直接用距离差
        inverse_distances = 1.0 / (distances + 1e-6)  # 加上小的正数以避免除零
        inverse_distances[~valid_window_mask.view(-1, 1, 1).expand(-1, win_len, num_circles)] = float('-inf')
        # 激活函数：在距离越近时激活更强，使用倒数距离
        activation = torch.sigmoid(inverse_distances * 100)  # 放大值以增强变化
        # 对窗口内的激活值取平均，得到每个窗口对各圆的平均激活
        window_scores_flat = activation.mean(dim=1) # [B*S, 8]
        window_scores = window_scores_flat.view(B, -1, circle_centers.size(0)) # 重塑为 [B, S, 8]
        # 使用 valid_window_mask 将无效窗口得分置零
        window_scores = window_scores * valid_window_mask.unsqueeze(2).float()

        # 动态权重分配（类似小波散射的路径整合）
        circle_weights = F.softmax(window_scores.view(-1, circle_centers.size(0)), dim=1)  # [B*S, 8]

        # 聚合窗口片段：按照每个圆分组
        circle_trajectories = []

        # 方案1 阈值筛选
        for i in range(circle_centers.size(0)):  # 对每个圆
            # 取出每个窗口对应的第 i 个圆的权重，形状 [B*S]
            weights_i = circle_weights[:, i]
            # 选择权重大于某个阈值的窗口，例如 > 0.5
            threshold = 0.25
            sel_mask = weights_i > threshold
            selected = all_segments_flat[sel_mask]
            circle_trajectories.append(selected)

        # 频次统计：将所有窗口的权重对每个圆求和，然后选择最大
        pass_counts = circle_weights.sum(dim=0)  # [num_circles]
        max_idx = torch.argmax(pass_counts).item()

        if circle_trajectories[max_idx].size(0) == 0:
            return empty_result
        circle_trajectories[max_idx] -= torch.tensor([rect.start.x, rect.start.y], device=circle_trajectories[max_idx].device)
        return circle_trajectories[max_idx]

        # 对每个圆心的轨迹都进行可视化
        import matplotlib.pyplot as plt
        # 创建新的 figure 和 axes
        # 创建一个2列子图 (左：线图；右：点图)
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        ax_line, ax_scatter = axes
        colors = plt.cm.get_cmap("tab10", circle_centers.size(0))  # 每个圆分配不同颜色
        # 对每个圆心的轨迹都进行可视化
        # for i in range(num_circles):  # 遍历所有圆心
            # 获取当前圆心的轨迹
            # circle_trajectories[i] -= torch.tensor([rect.start.x, rect.start.y], device=circle_trajectories[max_idx].device)
            # traj_data = circle_trajectories[i]  # 当前圆心的所有轨迹
            # color = colors(i)
            # traj_np = traj_data.cpu().numpy()  # 转换为 numpy 格式
            # for traj in traj_np:
            #     ax_line.plot(traj[:, 1], traj[:, 0], color=color, linewidth=1)  # lat_lon[:, 0] -> 纬度, lat_lon[:, 1] -> 经度

        # # 设置标题和坐标轴
        # ax_line.set_title("All Trajectories around 8 Circle Centers")
        # ax_line.set_xlim(0, 256)
        # ax_line.set_ylim(0, 256)
        # ax_line.axis("off")
        # ax_line.set_aspect('equal')  # 确保坐标轴比例一致
        # ax_line.axis("off")

        for i in range(num_circles):  # 遍历所有圆心
            # 获取当前圆心的轨迹
            traj_data = circle_trajectories[i]  # 当前圆心的所有轨迹
            color = colors(i)
            all_points = traj_data.view(-1, 2).cpu().numpy()  # 把所有轨迹点拼在一起
            ax_scatter.scatter(all_points[:, 1], all_points[:, 0], s=5, color=color, alpha=0.7, label=f"circle {i}")

        # 设置标题和坐标轴
        ax_scatter.set_title("All Trajectories points around 8 Circle Centers")
        ax_scatter.set_xlim(0, 256)
        ax_scatter.set_ylim(0, 256)
        ax_scatter.set_aspect('equal')  # 确保坐标轴比例一致
        ax_scatter.axis("off")

        # 保存图像到文件
        plt.tight_layout()
        plt.savefig("2all_trajectories_vis.png", bbox_inches="tight", pad_inches=0.1)
        ax_line.invert_yaxis()
        ax_scatter.invert_yaxis()
        plt.savefig("2all_trajectories_vis_y.png", bbox_inches="tight", pad_inches=0.1)
        ax_line.invert_xaxis()
        ax_line.invert_yaxis()
        ax_scatter.invert_yaxis()
        plt.savefig("2all_trajectories_vis_yx.png", bbox_inches="tight", pad_inches=0.1)

        return circle_trajectories[max_idx]

    #条件分支是啥 是不是我model里面的判断？？？？？？？？？？？？？？？？？？？？？
    # TODO 试试三种不同的circle_weights处理方案


def random_sample_given_probs(seq, probs):
    sum_probs = sum(probs)
    if sum_probs != 1:
        probs = [x/sum_probs for x in probs]
    probs.insert(0, 0)
    for i in range(len(probs)-1):
        probs[i+1] = probs[i] + probs[i+1]
    rand = random.random()
    for i in range(len(probs)-1):
        if probs[i] < rand < probs[i+1]:
            break
    return seq[i]


def get_avg_between_pnts_in_map(im_map, pnt1, pnt2, WINDOW_SIZE=256):
    if im_map is None:
        return 0
    pnts = geom.draw_line(pnt1, pnt2, geom.Point(WINDOW_SIZE, WINDOW_SIZE))
    lst = [im_map[pnt.x, pnt.y] for pnt in pnts]
    return np.mean(lst) if len(lst) != 0 else 0


def get_points_from_rtree(point_rtree, index2point, center_point, RECT_RADIUS) -> dict:
    points = dict()
    start = geom.Point(center_point.x - RECT_RADIUS,
                       center_point.y - RECT_RADIUS)
    end = geom.Point(center_point.x + RECT_RADIUS,
                     center_point.y + RECT_RADIUS)
    for point_id in point_rtree.intersection((start.x, start.y, end.x, end.y)):
        points[index2point[point_id]] = point_id
    return points


def get_random_rect(big_rect, WINDOW_SIZE=256):
    x = random.randint(big_rect.start.x, big_rect.end.x-WINDOW_SIZE)
    y = random.randint(big_rect.start.y, big_rect.end.y-WINDOW_SIZE)
    return geom.Rectangle(geom.Point(x, y), geom.Point(x+WINDOW_SIZE, y+WINDOW_SIZE))


def get_random_rect_padding(big_rect, WINDOW_SIZE=256):
    x = random.randint(big_rect.start.x, big_rect.end.x-1)
    y = random.randint(big_rect.start.y, big_rect.end.y-1)
    sz = WINDOW_SIZE // 2
    return geom.Rectangle(geom.Point(x-sz, y-sz), geom.Point(x+sz, y+sz))


def get_nearest_end_point(key_points, point):
    if type(key_points) is not list:
        key_points = list(key_points)
    if len(key_points) == 1:
        return key_points[0]
    nearest_key_point = key_points[0]
    for pnt in key_points[1:]:
        if pnt.distance(point) < nearest_key_point.distance(point):
            nearest_key_point = pnt
    return nearest_key_point


# 新加，同于读取成条轨迹数据
def get_alltraj_pieces_from_txt(trajectory_dir):
    """
        读取多个轨迹数据文件，返回所有轨迹的坐标数据。
        每个文件代表一条轨迹，每条轨迹的数据存储为 [latitude, longitude]。
        """
    all_trajectories = []

    print("loading traj in pieces")
    total = sum(len(files) for _, _, files in os.walk(trajectory_dir))
    for root, dirs, files in tqdm(os.walk(trajectory_dir), total=total):
        # 遍历当前文件夹下的所有文件
        for file_name in files:
            file_path = os.path.join(root, file_name)
            data = pd.read_csv(file_path, header=None, skiprows=1, names=['latitude', 'longitude'], usecols=[1, 2], dtype=np.float64)
            coordinates = data[['latitude', 'longitude']].to_numpy()
            all_trajectories.append(coordinates)
    all_trajectories = np.array(all_trajectories)
    print(f"trajectory length: {all_trajectories.shape}")

    return all_trajectories


def valid_trajectory_input(batch_valid_trajectories):
    # 如果输入为空或所有簇均为空，则返回占位张量
    if not batch_valid_trajectories or all(len(cluster) == 0 for cluster in batch_valid_trajectories):
        print("Warning: No valid trajectory data in the batch. Returning zero placeholder.")
        return np.zeros((len(batch_valid_trajectories), 1, 1, 2))  # 返回一个最小尺寸的占位张量
    if any(len(cluster) == 0 for cluster in batch_valid_trajectories):
        print("Warning: No valid trajectory data in the batch element")

    # if batch_valid_trajectories is not None and any(len(cluster) > 0 for cluster in batch_valid_trajectories):        # 计算批次中最大轨迹长度和最大轨迹簇数量
    max_seq_len = max([max([len(trajectory) for trajectory in cluster]) for cluster in batch_valid_trajectories if cluster and len(cluster) > 0])
    max_num_trajectories = max([len(cluster) for cluster in batch_valid_trajectories])
    # 初始化一个全零的数组，用来存储填充后的轨迹数据
    padded_batch = np.zeros((len(batch_valid_trajectories), max_num_trajectories, max_seq_len, 2))

    # 填充所有轨迹簇中的轨迹，使其具有相同的最大长度
    for i, cluster in enumerate(batch_valid_trajectories):
        padded_cluster = []
        for j, trajectory in enumerate(cluster):
            if len(trajectory) < max_seq_len:
                padding = max_seq_len - len(trajectory)
                # 使用0进行填充，保持轨迹点的经纬度结构
                padded_trajectory = np.pad(trajectory, ((0, padding), (0, 0)), mode='constant', constant_values=0)
            else:
                padded_trajectory = trajectory
            padded_batch[i, j, :, :] = np.array(padded_trajectory)
    # 将填充后的批次转换为四维数组，形状为 [batch_size, num_trajectories_in_cluster, max_seq_len, 2]
    return np.array(padded_batch)


def valid_trajectory_input_GPU(batch_valid_trajectories):
    # 如果输入是GPU张量则跳过转换
    if isinstance(batch_valid_trajectories, torch.Tensor):
        return batch_valid_trajectories

    # 处理空数据的情况
    if not batch_valid_trajectories or all(len(cluster) == 0 for cluster in batch_valid_trajectories):
        return torch.zeros((len(batch_valid_trajectories), 1, 1, 2), device='cuda')

    # 直接处理GPU张量
    max_seq_len = max([traj.size(0) for cluster in batch_valid_trajectories for traj in cluster])
    max_num_trajectories = max([len(cluster) for cluster in batch_valid_trajectories])

    # 使用torch.zeros_like保持设备一致性
    padded_batch = torch.zeros((len(batch_valid_trajectories), max_num_trajectories, max_seq_len, 2),device='cuda')

    for i, cluster in enumerate(batch_valid_trajectories):
        for j, trajectory in enumerate(cluster):
            if trajectory.size(0) < max_seq_len:
                pad_size = max_seq_len - trajectory.size(0)
                padded_trajectory = torch.cat([trajectory,
                                               torch.zeros(pad_size, 2, device=trajectory.device)], dim=0)
            else:
                padded_trajectory = trajectory[:max_seq_len]
            padded_batch[i, j] = padded_trajectory

    return padded_batch


def normalize_trajectory_batch(traj_batch, eps=1e-6):
    """
    对形状为 (B, n_cluster, seq_len, 2) 的轨迹数据进行归一化。
    只对有效轨迹点（非零填充值）计算均值和标准差，填充值保持为0。

    Args:
        traj_batch: Tensor, 形状为 (B, n_cluster, seq_len, 2)
        eps: 防止除零的小常数

    Returns:
        norm_traj: 归一化后的轨迹数据，形状 (B, n_cluster, seq_len, 2)
        mask: 布尔型 mask，形状 (B, n_cluster, seq_len)，True 表示真实数据，False 表示填充值
    """
    # 计算每个轨迹点是否有效（这里假设填充值为0）
    mask = (traj_batch.abs().sum(dim=-1) != 0).unsqueeze(-1)  # [B, n_cluster, seq_len, 1]
    mask_float = mask.float()
    # 计算每个轨迹序列中有效点的数量，形状 [B, n_cluster, 1, 1]
    valid_counts = mask_float.sum(dim=2, keepdim=True)
    # 计算均值（仅对有效数据进行统计），形状 [B, n_cluster, 1, 2]
    mean = (traj_batch * mask_float).sum(dim=2, keepdim=True) / (valid_counts + eps)
    # 计算方差（仅对有效数据进行统计），形状 [B, n_cluster, 1, 2]
    var = (((traj_batch - mean) ** 2) * mask_float).sum(dim=2, keepdim=True) / (valid_counts + eps)
    std = torch.sqrt(var + eps)
    # 归一化：对于有效数据进行 (x - mean) / std，不改变填充值
    norm_traj = (traj_batch - mean) / std
    norm_traj = norm_traj * mask_float  # 保持填充值为0
    # 返回归一化后的轨迹和对应的 mask（去掉最后一个维度）
    return norm_traj, mask.squeeze(-1)


class TargetPosesContainer:
    def __init__(self, NUM_TARGETS=4):
        self.target_poses = [[] for _ in range(NUM_TARGETS)]
        self.NUM_TARGETS = NUM_TARGETS

    def get_all_target_poses(self):
        res = []
        for poses in self.target_poses:
            res.extend(poses)
        return res

    def __getitem__(self, index):
        return self.target_poses[index]

    def __len__(self):
        for i, poses in enumerate(self.target_poses):
            if len(poses) == 0:
                return i
        return self.NUM_TARGETS  # == NUM_TARGETS

    def is_end_with_key_point(self):
        end_index = self.__len__()
        if end_index > 0 and len(self.target_poses[end_index - 1]) > 1:
            return True
        return False

    def get_single_lst(self):
        # to get target_poses without junction end
        res = []
        for poses in self.target_poses:
            if len(poses) == 0:
                break
            res.extend(poses)
        return res

    def get_single_lst_without_junction_end(self):
        # to get target_poses without junction end
        res = []
        for i, poses in enumerate(self.target_poses):
            if i == 0 and len(poses) > 1:
                return poses
            elif i > 0 and len(poses) > 1:
                return res
            else:
                res.extend(poses)
        return res

    def len_without_junction_end(self):
        for i, poses in enumerate(self.target_poses):
            if i != 0 and len(poses) > 1:
                return i
            if len(poses) == 0:
                return i
        return self.NUM_TARGETS

    def get_supervision_end_index(self):
        for i, poses in enumerate(self.target_poses):
            if len(poses) > 1:
                return i + 1
        return self.NUM_TARGETS

    def str(self, graph):
        string = "["
        for index, item in enumerate(self.target_poses):
            string += "["
            for i, x in enumerate(item):
                pnt = x.point(graph)
                string += "({},{})".format(pnt.x, pnt.y)
                if i != len(item)-1:
                    string += ", "
            string += "]"
            if index != len(self.target_poses)-1:
                string += ", "
        string += "]"
        return string


def map_to_coordinate(batch_output_maps, batch_is_key_point, batch_extension_vertices, ROAD_SEG_THRESHOLE=0.2,
                      STEP_LENGTH=20, JUNC_MAX_REGION_AREA=200):
    """
    其主要目的是将一批输出的地图数据（batch_output_anchor_maps）转换为坐标点。
    这个函数通过分析地图数据中的区域特征，识别出关键点或锚点，并根据这些特征生成相应的坐标点

    return:
        if is_key_point:
            res == [(x,y), ..., (x,y)]  # time_step == 1 # +
        else:
            res == [(x,y), ..., (x,y)]  # time_step  > 1 # ----
    """
    def _frame_to_coordinate(frame, origin_point, channel_index, previous_pnt=None):
        # 用于处理单个帧（地图）
        # 遍历每个区域，检查其面积是否超过 JUNC_MAX_REGION_AREA，如果超过则跳过。
        # 计算区域质心的距离，并根据距离和条件决定是否将其加入结果列表 res
        frame[np.where(frame < ROAD_SEG_THRESHOLE)] = 0
        frame[np.where(frame)] = 1
        labels = measure.label(frame, connectivity=2)
        props = measure.regionprops(labels)
        res = []
        for region in props:
            if region.area > JUNC_MAX_REGION_AREA:
                continue
            offset = geom.Point(
                int(region.centroid[0]), int(region.centroid[1]))
            distance = offset.distance(center_pnt)
            if distance > (channel_index + 2) * STEP_LENGTH:
                continue
            if previous_pnt is not None:
                distance = offset.distance(previous_pnt.sub(origin_point))
                if distance > 2 * STEP_LENGTH:
                    continue
            res.append(origin_point.add(offset))
        return res

    _, NUM_TARGETS, WINDOW_SIZE, _ = batch_output_maps.shape #（20，4，256，256）
    batch_size = len(batch_extension_vertices)
    batch_res = []
    center_pnt = geom.Point(WINDOW_SIZE // 2, WINDOW_SIZE // 2)
    for batch_idx in range(batch_size):
        origin_point = batch_extension_vertices[batch_idx].point.sub(
            geom.Point(WINDOW_SIZE // 2, WINDOW_SIZE // 2))
        previous_pnt = None
        res = []
        for i in range(NUM_TARGETS):
            frame = batch_output_maps[batch_idx, i, :, :]
            frame_coordinate = _frame_to_coordinate(
                frame, origin_point, i, previous_pnt)
            # judged by junction segmentation to be a keypoint
            if batch_is_key_point[batch_idx]:
                res.extend(frame_coordinate)
                break
            # batch_is_key_point[batch_idx]==False, judged by anchor to be a keypoint
            elif len(frame_coordinate) > 1:
                if i == 0:
                    batch_is_key_point[batch_idx] = True
                    res.extend(frame_coordinate)
                break
            elif len(frame_coordinate) == 0:
                break
            else:  # len(frame_coordinate) == 1:
                previous_pnt = frame_coordinate[0]
                res.extend(frame_coordinate)
        batch_res.append(res)
    return batch_res
