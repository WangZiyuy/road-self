import sys
sys.path.append('.')

import numpy as np
from tptk.common.mbr import MBR
from tptk.common.grid import Grid
from tptk.common.spatial_func import LAT_PER_METER, LNG_PER_METER
from metrics.pure_image_metric import cal_geometry_metrics
from metrics.utils import generate_rn_image
from tptk.common.road_network import load_rn_shp
import os
from regions import get_regions

def cal_geometric_metrics(mbr, sample_interval, pred, gt):
    print('distance matching threshold:{}'.format(sample_interval))
    row_span = sample_interval * LAT_PER_METER
    col_span = sample_interval * LNG_PER_METER
    row_num = int((mbr.max_lat - mbr.min_lat) / row_span)
    col_num = int((mbr.max_lng - mbr.min_lng) / col_span)
    print('{}x{}'.format(row_num, col_num))

    precision, recall, f1 = cal_geometry_metrics(pred, gt)
    print('precision:{}, recall:{}, f1:{}'.format(precision, recall, f1))
    return precision, recall, f1


if __name__ == '__main__':

    # TODO: to be replaced
    # 20;POLYGON ((116.30278723404255 39.87206896551724	 116.30278723404255 39.890098522167484	 116.33204255319149 39.890098522167484	 116.33204255319149 39.87206896551724	 116.30278723404255 39.87206896551724))
    eval_mbr = MBR(39.87206896551724, 116.30278723404255, 39.890098522167484, 116.33204255319149)
    gt_rn_path = '/home/wangziyu/VecRoad/data_self/shp_files/'
    pred_rn_path = '/home/wangziyu/VecRoad/data_self/graphs/vecroad_4/graphs_shp/'
    region_file = '/home/wangziyu/VecRoad/data_self/input/regions/test_regions.txt'

    region_dict = get_regions(region_file)
    region = list(region_dict.values())[0]
    shp_path = os.path.join(pred_rn_path, region.name + ".shp")
    shp_path_gt = os.path.join(gt_rn_path, region.name, region.name + ".shp")
    print(shp_path)
    print(shp_path_gt)

    gt_rn = load_rn_shp(shp_path_gt)
    pred_rn = load_rn_shp(shp_path)

    sample_intervals = [5, 10, 15, 20]
    for sample_interval in sample_intervals:
        print(sample_interval)

        row_span = sample_interval * LAT_PER_METER
        col_span = sample_interval * LNG_PER_METER
        row_num = int((eval_mbr.max_lat - eval_mbr.min_lat) / row_span)
        col_num = int((eval_mbr.max_lng - eval_mbr.min_lng) / col_span)
        eval_region_grid_idx = Grid(eval_mbr, row_num, col_num)
        gt_image = generate_rn_image(gt_rn, eval_region_grid_idx, region.name+'ground truth')
        gt_data = np.copy(np.asarray(gt_image))
        gt_data[gt_data > 0] = 1

        pred_image = generate_rn_image(pred_rn, eval_region_grid_idx, region.name+'pred')
        pred_data = np.copy(np.asarray(pred_image))
        pred_data[pred_data > 0] = 1
        cal_geometric_metrics(eval_mbr, sample_interval, pred_data, gt_data)
