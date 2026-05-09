from tptk.common.road_network import load_rn_shp
from tptk.common.mbr import MBR
from tptk.common.grid import Grid
from tptk.common.spatial_func import LAT_PER_METER, LNG_PER_METER
import random
from metrics.geo_metric import cal_geometry_metrics
import numpy as np
import cv2
import os
from regions import get_regions

def cal_topological_metrics(nb_samples, mbr, sample_interval, search_radius, pred_rn, gt_rn):
    row_span = sample_interval * LAT_PER_METER
    col_span = sample_interval * LNG_PER_METER
    row_num = int((mbr.max_lat - mbr.min_lat) / row_span)
    col_num = int((mbr.max_lng - mbr.min_lng) / col_span)
    sample_region_delta_row = search_radius * 2 * LAT_PER_METER
    sample_region_delta_col = search_radius * 2 * LNG_PER_METER
    print('{}x{}'.format(row_num, col_num))
    coarse_grid_idx = Grid(mbr, row_num, col_num)
    gt_non_empty_grid = set(get_non_empty_grid(gt_rn, coarse_grid_idx).keys())
    pred_non_empty_grid = set(get_non_empty_grid(pred_rn, coarse_grid_idx).keys())
    non_empty_grids = list(gt_non_empty_grid.intersection(pred_non_empty_grid))
    precision_all = 0.0
    recall_all = 0.0
    f1_all = 0.0
    invalid_cnt = 0
    for i in range(nb_samples):
        start_grid = random.choice(non_empty_grids)
        start_pt = coarse_grid_idx.get_mbr_by_matrix_idx(start_grid[0], start_grid[1]).center()
        sample_region_grid_idx = get_sample_region_grid_idx(start_pt, sample_region_delta_row, sample_region_delta_col, mbr, row_span, col_span)
        gt_reachable_segments = get_reachable_segments(start_grid, gt_rn, coarse_grid_idx, sample_region_grid_idx.mbr)
        pred_reachable_segments = get_reachable_segments(start_grid, pred_rn, coarse_grid_idx, sample_region_grid_idx.mbr)
        gt_data = np.asarray(generate_image(gt_reachable_segments, sample_region_grid_idx, gt_rn, str(i) + '_gt'))
        pred_data = np.asarray(generate_image(pred_reachable_segments, sample_region_grid_idx, pred_rn, str(i) + '_pred'))
        precision, recall, f1 = cal_geometry_metrics(pred_data > 0, gt_data > 0)
        if np.isnan(precision) or np.isnan(recall) or np.isnan(f1):
            invalid_cnt += 1
            continue
        precision_all += precision
        recall_all += recall
        f1_all += f1
    precision_all /= (nb_samples - invalid_cnt)
    recall_all /= (nb_samples - invalid_cnt)
    f1_all /= (nb_samples - invalid_cnt)
    print('precision:{}, recall:{}, f1:{}'.format(precision_all, recall_all, f1_all))
    return precision_all, recall_all, f1_all


def get_non_empty_grid(gt_rn, grid_idx):
    grid2segments = {}
    for i in range(grid_idx.row_num):
        for j in range(grid_idx.col_num):
            query_mbr = grid_idx.get_mbr_by_matrix_idx(i, j)
            result_segments = gt_rn.range_query(query_mbr)
            if len(result_segments) > 0:
                grid2segments[(i, j)] = result_segments
    return grid2segments


def get_reachable_segments(start_grid, rn, grid_idx, sample_region_mbr):
    reachable_segments = set()
    start_grid_mbr = grid_idx.get_mbr_by_matrix_idx(start_grid[0], start_grid[1])
    start_segments = rn.range_query(start_grid_mbr)
    open_node = set()
    close_node = set()
    for u, v in start_segments:
        for nei in rn.neighbors(v):
            if sample_region_mbr.contains(nei[1], nei[0]):
                reachable_segments.add((v, nei))
                open_node.add(nei)
    while len(open_node) > 0:
        v = open_node.pop()
        close_node.add(v)
        for nei in rn.neighbors(v):
            if nei not in close_node and sample_region_mbr.contains(nei[1], nei[0]):
                reachable_segments.add((v, nei))
                open_node.add(nei)
    return reachable_segments


def get_sample_region_grid_idx(start_pt, sample_region_delta_row, sample_region_delta_col, mbr, row_span, col_span):
    sample_region_min_lat = max(start_pt[0] - sample_region_delta_row / 2.0, mbr.min_lat)
    sample_region_min_lng = max(start_pt[1] - sample_region_delta_col / 2.0, mbr.min_lng)
    sample_region_max_lat = min(start_pt[0] + sample_region_delta_row / 2.0, mbr.max_lat)
    sample_region_max_lng = min(start_pt[1] + + sample_region_delta_col / 2.0, mbr.max_lng)
    sample_region_mbr = MBR(sample_region_min_lat, sample_region_min_lng, sample_region_max_lat, sample_region_max_lng)
    sample_region_row_num = int((sample_region_mbr.max_lat - sample_region_mbr.min_lat) / row_span)
    sample_region_col_num = int((sample_region_mbr.max_lng - sample_region_mbr.min_lng) / col_span)
    sample_region_grid_idx = Grid(sample_region_mbr, sample_region_row_num, sample_region_col_num)
    return sample_region_grid_idx


def generate_image(segments, sample_region_grid_idx, rn, filename):
    img = np.zeros((sample_region_grid_idx.row_num, sample_region_grid_idx.col_num), dtype=np.uint8)
    # print("img: {}".format(img.shape))
    for eid in segments:
        # eid: [lng, lat]
        coords = rn.edges[eid]['coords']
        for i in range(len(coords) - 1):
            start_node, end_node = coords[i], coords[i + 1]
            try:
                y1, x1 = sample_region_grid_idx.get_matrix_idx(start_node.lat, start_node.lng)
                y2, x2 = sample_region_grid_idx.get_matrix_idx(end_node.lat, end_node.lng)
                cv2.line(img, (x1, y1), (x2, y2), 255, 1, lineType=cv2.LINE_8)
            except IndexError:
                continue
    cv2.imwrite('./test/{}.png'.format(filename), img)
    return img


if __name__ == '__main__':

    # TODO: to be replaced
    # 20;POLYGON ((116.30278723404255 39.87206896551724	 116.30278723404255 39.890098522167484	 116.33204255319149 39.890098522167484	 116.33204255319149 39.87206896551724	 116.30278723404255 39.87206896551724))
    eval_mbr = MBR(39.87206896551724, 116.30278723404255, 39.890098522167484, 116.33204255319149)
    gt_rn_path = '/home/wangziyu/VecRoad/data_self/shp_files/'
    pred_rn_path = '/home/wangziyu/VecRoad/data_self/graphs/vecroad_4/graphs_shp/'
    region_file = '/home/wangziyu/VecRoad/data_self/input/regions/test_regions.txt'
    is_directed = False
    region_dict = get_regions(region_file)
    region = list(region_dict.values())[0]
    shp_path = os.path.join(pred_rn_path, region.name + ".shp")
    shp_path_gt = os.path.join(gt_rn_path, region.name, region.name + ".shp")

    search_radius = 2000
    nb_samples = 200
    sample_intervals = [5, 10, 15, 20]
    gt_rn = load_rn_shp(shp_path_gt, is_directed=is_directed)
    pred_rn = load_rn_shp(shp_path, is_directed=is_directed)

    for sample_interval in sample_intervals:
        print(sample_interval)
        cal_topological_metrics(nb_samples, eval_mbr, sample_interval, search_radius, pred_rn, gt_rn)
