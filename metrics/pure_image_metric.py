import numpy as np
from tptk.common.mbr import MBR
from tptk.common.grid import Grid
from tptk.common.spatial_func import LAT_PER_METER, LNG_PER_METER
from PIL import Image


def cal_geometry_metrics(pred, gt):
    tp = np.sum(pred * gt)
    fn = np.sum((pred == 0) * (gt == 1))
    fp = np.sum((pred == 1) * (gt == 0))
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def cal_geometric_metrics(filename, mbr, sample_interval, pred, gt):
    print('distance matching threshold:{}'.format(sample_interval))
    row_span = sample_interval * LAT_PER_METER
    col_span = sample_interval * LNG_PER_METER
    row_num = int((mbr.max_lat - mbr.min_lat) / row_span)
    col_num = int((mbr.max_lng - mbr.min_lng) / col_span)
    print('{}x{}'.format(row_num, col_num))
    coarse_grid_idx = Grid(mbr, row_num, col_num)
    fine_grid_idx = Grid(mbr, pred.shape[0], pred.shape[1])
    coarse_pred = get_coarse_data(pred, fine_grid_idx, coarse_grid_idx)

    img = Image.fromarray(coarse_pred * 255, 'L')
    img.save('coarse_{}_{}.png'.format(filename[:-4], sample_interval))

    coarse_gt = get_coarse_data(gt, fine_grid_idx, coarse_grid_idx)

    img = Image.fromarray(coarse_gt * 255, 'L')
    img.save('coarse_gt_{}.png'.format(sample_interval))

    precision, recall, f1 = cal_geometry_metrics(coarse_pred, coarse_gt)
    print('precision:{}, recall:{}, f1:{}'.format(precision, recall, f1))
    return precision, recall, f1


def get_coarse_data(fine_data, fine_grid_idx, coarse_grid_idx):
    coarse_data = np.zeros((coarse_grid_idx.row_num, coarse_grid_idx.col_num), dtype=np.uint8)
    for i in range(coarse_grid_idx.row_num):
        for j in range(coarse_grid_idx.col_num):
            coarse_mbr = coarse_grid_idx.get_mbr_by_matrix_idx(i, j)
            fine_idxes = fine_grid_idx.range_query(coarse_mbr, type='matrix')
            fine_idxes = [fine_idx for fine_idx in fine_idxes if coarse_mbr.contains(*fine_grid_idx.get_mbr_by_matrix_idx(fine_idx[0], fine_idx[1]).center())]
            if sum([fine_data[y][x] for y, x in fine_idxes]) > 0:
                coarse_data[i, j] = 1
    return coarse_data


if __name__ == '__main__':
    eval_mbr = MBR(39.89006875, 116.29856875, 39.9350375, 116.35713125000001)
    sample_intervals = [5, 10, 15, 20]
    print(eval_mbr)

    # gt_image_path = './data/gt_rn.png'
    gt_image_path = '../data/eval_data/uic/rn_edge_1280_2048_line_8.png'
    gt_image = Image.open(gt_image_path).convert('L')
    gt_data = np.copy(np.asarray(gt_image))
    gt_data[gt_data > 0] = 1

    # filename = 'dlinknet34_pred_rn.png'
    filename = 'dlinknet34_thinned.png'
    # filename = 'gis12_rn.png'
    # filename = 'traj2rn_trajnet_no_da.png'
    # pred_image_path = './data/{}'.format(filename)
    pred_image_path = '../data/eval_data/uic/pred.png'
    pred_image = Image.open(pred_image_path).convert('L')
    pred_data = np.copy(np.asarray(pred_image))
    pred_data[pred_data > 0] = 1
    print(pred_data.shape)

    cal_geometric_metrics(filename, eval_mbr, 5, pred_data, gt_data)
    cal_geometric_metrics(filename, eval_mbr, 10, pred_data, gt_data)
    cal_geometric_metrics(filename, eval_mbr, 15, pred_data, gt_data)
    cal_geometric_metrics(filename, eval_mbr, 20, pred_data, gt_data)
