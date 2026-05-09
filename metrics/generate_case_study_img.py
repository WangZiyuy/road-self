from tptk.common.grid import Grid
from tptk.common.mbr import MBR
import cv2
import numpy as np
from metrics.utils import generate_rn_image
from tptk.common.road_network import load_rn_shp

if __name__ == '__main__':
    mbr_bj = MBR(39.8451, 116.2810, 39.9890, 116.4684)
    h_res = 1.9531
    w_res = 1.9529
    h_tiles = int(mbr_bj.get_h() / h_res)
    w_tiles = int(mbr_bj.get_w() / w_res)
    tile_idx = Grid(mbr_bj, h_tiles, w_tiles)
    method = 'Biagioni'
    # method = 'Kharita'
    # method = 'Chen'
    # method = 'Cao'
    # method = 'Edelkamp'
    # method = 'DeepMG'
    alpha = 1.4
    rel_path = None
    if method == 'Biagioni':
        # rel_path = 'Biagioni'
        rel_path = 'Biagioni_draft'
    elif method == 'Kharita':
        # rel_path = 'SDM18/kharita'
        rel_path = 'kharita'
    elif method == 'Cao':
        rel_path = '09_result'
    elif method == 'Edelkamp':
        rel_path = '03_result'
    elif method == 'Chen':
        rel_path = 'KDD16/add_links'
    if method != 'DeepMG':
        # rn_path = '/Users/sjruan/OneDrive/MR-GAN/baselines/bj/{}/'.format(rel_path)
        rn_path = 'D:/OneDrive/MR-GAN/baselines/bj/{}/'.format(rel_path)
    else:
        # rn_path = '../data/aaai20/TaxiBJ/filtered_r100_s5_a{}_condi/'.format(alpha)
        rn_path = '../topology_construction/topology_refinement/aaai_refine_test_a{}/'.format(alpha)

    generated_map = load_rn_shp(rn_path)
    pred_image = generate_rn_image(generated_map, tile_idx)
    TILE_PIXEL_SIZE = 256

    row_min = 17.8
    row_max = 19
    col_min = 3.35
    col_max = 4.55
    slices = (slice(int(row_min * TILE_PIXEL_SIZE), int(row_max * TILE_PIXEL_SIZE)),
              slice(int(col_min * TILE_PIXEL_SIZE), int(col_max * TILE_PIXEL_SIZE)))
    moti_region_rn = pred_image[slices]
    moti_region_rn_disp = np.full((moti_region_rn.shape[0], moti_region_rn.shape[1], 3), 255, dtype=np.uint8)
    moti_region_rn_disp[moti_region_rn > 0] = (255, 0, 0)
    if method != 'DeepMG':
        cv2.imwrite('./case_study_{}.png'.format(method), moti_region_rn_disp)
    else:
        cv2.imwrite('./case_study_{}_a{}_new.png'.format(method, str(alpha).replace('.', '_')), moti_region_rn_disp)
