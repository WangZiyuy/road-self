import cv2
import numpy as np


def generate_rn_image(rn, grid_idx, filename=None):
    img = np.zeros((grid_idx.row_num, grid_idx.col_num), dtype=np.uint8)
    # print("img: {}".format(img.shape))
    for eid in rn.edges():
        # eid: [lng, lat]
        coords = rn.edges[eid]['coords']
        for i in range(len(coords) - 1):
            start_node, end_node = coords[i], coords[i + 1]
            try:
                y1, x1 = grid_idx.get_matrix_idx(start_node.lat, start_node.lng)
                y2, x2 = grid_idx.get_matrix_idx(end_node.lat, end_node.lng)
                cv2.line(img, (x1, y1), (x2, y2), 255, 1, lineType=cv2.LINE_8)
            except IndexError:
                continue
    if filename is not None:
        cv2.imwrite('../data_self/graphs/vecroad_4/test/{}.png'.format(filename), img)
    return img
