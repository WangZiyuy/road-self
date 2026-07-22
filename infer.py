import argparse
import os.path
import time
from multiprocessing import Pool

import cv2 as cv
import torchvision.transforms as transforms
import yaml
from easydict import EasyDict
from PIL import Image
from skimage import measure
from tqdm import tqdm

import utils.model_utils as model_utils
import utils.tileloader as tileloader
import utils.OSMDataset as OSMDataset
from lib import geom, graph as graph_helper
from model.model import RPNet, upsample
from utils.regions import Region, get_regions
from utils.utils import load_pretrained, numpy2tensor2cuda, MapContainer
import numpy as np
import torch
from utils.additional_methods import analyze_checkpoint
from utils.trajectory_mode import (
    TRAJ_MODE_NONE,
    load_region_trajectory_inputs_for_mode,
    prepare_trajectory_sequence_batch,
    resolve_trajectory_mode,
    trajectory_fetch_fields,
    validate_trajectory_model_compatibility,
)


parser = argparse.ArgumentParser(description="VecRoad Pytorch Test")
parser.add_argument(
    "--config",
    default="configs/default_self.yml",
    metavar="FILE",
    help="path to config file",
    type=str,
)
args = parser.parse_args()

assert os.path.isfile(args.config)
config_file = open(args.config, "r")
cfg = yaml.load(config_file, Loader=yaml.UnsafeLoader)
config_file.close()
cfg = EasyDict(cfg)
TRAJECTORY_MODE = resolve_trajectory_mode(cfg)
validate_trajectory_model_compatibility(cfg, TRAJECTORY_MODE)
USE_TRAJECTORY = TRAJECTORY_MODE != TRAJ_MODE_NONE

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.TEST.GPU_ID

def main():
    test_regions = get_regions(cfg.DIR.TEST_REGION_PATH)
    if cfg.TEST.SINGLE_REGION != "":
        test_regions = {
            cfg.TEST.SINGLE_REGION: test_regions[cfg.TEST.SINGLE_REGION]}

    net = prepare_net().eval()

    junction_nms_res = dict()
    junction_nms_res_vis = dict()
    road_seg_filter_dict = dict()
    graph_dict = dict()
    for region_name in test_regions.keys():
        graph_dict[region_name] = None

    ######################## 这一部分都是前处理，对于节点和路段分割##############################
    if cfg.TEST.INFER_STEP == "start":
        os.makedirs(os.path.join(cfg.DIR.SAVE_SEG_DIR,
                                 cfg.TEST.CKPT, "junction"), exist_ok=True)
        os.makedirs(os.path.join(cfg.DIR.SAVE_SEG_DIR,
                                 cfg.TEST.CKPT, "road"), exist_ok=True)
        print("infer segmentation start, INPUT_SIZE:{}".format(cfg.TEST.CROP_SZ))
        road_map_dict, junc_map_dict = infer_segmentation(
            net, list(test_regions.keys()))
        print("infer segmentation done")


        print("junction nms start")
        os.makedirs(os.path.join(cfg.DIR.SAVE_SEG_DIR,
                                 cfg.TEST.CKPT, "junc_nms"), exist_ok=True)
        os.makedirs(os.path.join(cfg.DIR.SAVE_SEG_DIR,
                                 cfg.TEST.CKPT, "junc_nms_vis"), exist_ok=True)
        os.makedirs(os.path.join(cfg.DIR.SAVE_SEG_DIR, cfg.TEST.CKPT,
                                 "road_seg_region_filter"), exist_ok=True)
        pool = Pool(cfg.TEST.CPU_WORKER)
        # 交叉点非极大值抑制(NMS)：通过junction_nms进一步提取交叉点信息，去除噪声区域。
        # 道路分割区域过滤：调用road_seg_region_filter删除面积过小或非有效的道路区域。
        if cfg.TEST.START_FROM_JUNC_PEAK:
            for region_name in test_regions.keys():
                # apply_async并行运行
                junction_nms_res[region_name] = pool.apply_async(
                    junction_nms, args=(region_name, junc_map_dict[region_name]))
            for region_name in test_regions.keys():
                junction_nms_res[region_name] = junction_nms_res[region_name].get()
            del junc_map_dict

        if cfg.TEST.START_FROM_ROAD_PEAK:
            for region_name in test_regions.keys():
                road_seg_filter_dict[region_name] = pool.apply_async(
                    road_seg_region_filter, args=(region_name, road_map_dict[region_name]))
            for region_name in test_regions.keys():
                road_seg_filter_dict[region_name] = road_seg_filter_dict[region_name].get(
                )
        del road_map_dict
        pool.close()
        pool.join()
        print("junction nms done")
    elif cfg.TEST.INFER_STEP == "after_seg":
        print("junction nms start")
        os.makedirs(os.path.join(cfg.DIR.SAVE_SEG_DIR,
                                 cfg.TEST.CKPT, "junc_nms"), exist_ok=True)
        os.makedirs(os.path.join(cfg.DIR.SAVE_SEG_DIR,
                                 cfg.TEST.CKPT, "junc_nms_vis"), exist_ok=True)
        os.makedirs(os.path.join(cfg.DIR.SAVE_SEG_DIR, cfg.TEST.CKPT,
                                 "road_seg_region_filter"), exist_ok=True)
        pool = Pool(cfg.TEST.CPU_WORKER)
        if cfg.TEST.START_FROM_JUNC_PEAK:
            for region_name in test_regions.keys():
                junc_nms_map = cv.imread(os.path.join(
                    cfg.DIR.SAVE_SEG_DIR, cfg.TEST.CKPT, "junction", region_name + '.png'), 0)
                junction_nms_res[region_name] = pool.apply_async(
                    junction_nms, args=(region_name, junc_nms_map,))
            for region_name in test_regions.keys():
                junction_nms_res[region_name] = junction_nms_res[region_name].get()
        if cfg.TEST.START_FROM_ROAD_PEAK:
            for region_name in test_regions.keys():
                road_seg_map = cv.imread(os.path.join(
                    cfg.DIR.SAVE_SEG_DIR, cfg.TEST.CKPT, "road", region_name + '.png'), 0) / 255.
                road_seg_filter_dict[region_name] = pool.apply_async(
                    road_seg_region_filter, args=(region_name, road_seg_map,))
            for region_name in test_regions.keys():
                road_seg_filter_dict[region_name] = road_seg_filter_dict[region_name].get(
                )
        pool.close()
        pool.join()
        print("junction nms end")
    elif cfg.TEST.INFER_STEP in ["after_junc_nms", "given_junc_nms"] and cfg.TEST.START_FROM_ROAD_PEAK:
        for region_name in test_regions.keys():
            road_seg_filter_dict[region_name] = cv.imread(
                os.path.join(cfg.DIR.SAVE_SEG_DIR, cfg.TEST.CKPT, "road_seg_region_filter", region_name + ".png"), 0) / 255.
    elif cfg.TEST.INFER_STEP == "after_graph_from_junc" and cfg.TEST.START_FROM_ROAD_PEAK:
        for region_name in test_regions.keys():
            road_seg_filter_dict[region_name] = cv.imread(os.path.join(cfg.DIR.SAVE_SEG_DIR, cfg.TEST.CKPT,
                                                                       "road_seg_region_filter", region_name + ".png"), 0) / 255.
            if cfg.TEST.START_FROM_JUNC_PEAK:
                graph_dict[region_name] = graph_helper.read_graph(os.path.join(
                    cfg.DIR.SAVE_GRAPH_DIR, '{}_{}'.format(
                        cfg.TEST.CKPT, cfg.TEST.NUM_TARGETS), 'graphs_junc',
                    '{}.graph'.format(region_name)))


    ######################## 这一部分是道路网络生成##############################
    img_cache = tileloader.TileCache(
        tile_dir=cfg.DIR.IMAGERY_DIR,
        tile_size=cfg.TRAIN.IMG_SZ,
        window_size=cfg.TEST.WINDOW_SIZE,
        limit=cfg.TRAIN.PARALLEL_TILES,
        traj_dir=(cfg.DIR.get("TRAJ_DIR", None) if USE_TRAJECTORY else None),)
    paths = []
    region_lst = list(test_regions.keys())

    # START_FROM_JUNC_PEAK的模式可能会受生成的junc_nms节点数量影响，而我之前的junc确实很少
    if not cfg.TEST.INFER_STEP == "after_graph_from_junc" and cfg.TEST.START_FROM_JUNC_PEAK:
        for i, region_name in enumerate(region_lst):
            tile_data = get_tile_data(
                test_regions[region_name], img_cache, junction_nms_res, get_starting_locations=True)
            all_trajectories = None
            all_pixel_trajectories = None
            traj_grid_index = None
            traj_grid_cell_size = None
            all_trajectories, all_pixel_trajectories, traj_grid_index, traj_grid_cell_size = \
                load_region_trajectory_inputs_for_mode(
                    TRAJECTORY_MODE,
                    tile_data["region"],
                    cfg,
                    OSMDataset.load_region_trajectory_inputs,
                )

            paths.append(model_utils.Path(i, training=False, gc=None, tile_data=tile_data, graph=None, road_seg=None,
                                          all_trajectories=all_trajectories, all_pixel_trajectories=all_pixel_trajectories,
                                          traj_grid_index=traj_grid_index, traj_grid_cell_size=traj_grid_cell_size))

        save_graph_dir = os.path.join(cfg.DIR.SAVE_GRAPH_DIR, '{}_{}'.format(cfg.TEST.CKPT, cfg.TEST.NUM_TARGETS),
                                      'graphs_junc')
        os.makedirs(save_graph_dir, exist_ok=True)
        try:
            iters, graph_dict = infer_anchor(paths, net, region_lst=region_lst, save_graph_dir=save_graph_dir,
                                             batch_size=cfg.TEST.BATCH_SIZE_ANCHOR)
            print('iters: ', iters)
        except:
            for path in paths:
                path.graph.save(os.path.join(
                    save_graph_dir, 'except_{}.graph'.format(region_lst[path.idx])))
                print("    Except save graph {}".format(region_lst[path.idx]))

    if cfg.TEST.START_FROM_ROAD_PEAK:
        if len(paths) == 0:
            for i, region_name in enumerate(region_lst):
                tile_data = get_tile_data(
                    test_regions[region_name], img_cache, junction_nms_res, get_starting_locations=False)
                all_trajectories = None
                all_pixel_trajectories = None
                traj_grid_index = None
                traj_grid_cell_size = None
                all_trajectories, all_pixel_trajectories, traj_grid_index, traj_grid_cell_size = \
                    load_region_trajectory_inputs_for_mode(
                        TRAJECTORY_MODE,
                        tile_data["region"],
                        cfg,
                        OSMDataset.load_region_trajectory_inputs,
                    )
                path = model_utils.Path(i, training=False, gc=None, tile_data=tile_data,
                                        graph=graph_dict[region_name],
                                        road_seg=np.ascontiguousarray(road_seg_filter_dict[region_name].swapaxes(0, 1)),
                                        all_trajectories=all_trajectories, all_pixel_trajectories=all_pixel_trajectories,
                                        traj_grid_index=traj_grid_index, traj_grid_cell_size=traj_grid_cell_size)
                paths.append(path)
        else:
            for i, region_name in enumerate(region_lst):
                paths[i].road_seg = np.ascontiguousarray(
                    road_seg_filter_dict[region_name].swapaxes(0, 1))
                paths[i].remove_graph_from_road_seg()

        if cfg.TEST.START_FROM_JUNC_PEAK:
            save_graph_dir = os.path.join(cfg.DIR.SAVE_GRAPH_DIR,
                                          '{}_{}'.format(cfg.TEST.CKPT, cfg.TEST.NUM_TARGETS),
                                          'graphs_junc_road')
        else:
            save_graph_dir = os.path.join(cfg.DIR.SAVE_GRAPH_DIR,
                                          '{}_{}'.format(cfg.TEST.CKPT, cfg.TEST.NUM_TARGETS),
                                          'graphs_road')
        os.makedirs(save_graph_dir, exist_ok=True)
        try:
            iters, graph_dict = infer_anchor(paths, net, region_lst=region_lst, save_graph_dir=save_graph_dir,
                                             batch_size=cfg.TEST.BATCH_SIZE_ANCHOR)
            print(iters)
        except:
            for path in paths:
                path.graph.save(os.path.join(
                    save_graph_dir, 'except_{}.graph'.format(region_lst[path.idx])))
                print("    Except save graph {}".format(region_lst[path.idx]))

    post_process_graph(graph_dict)



#删除标记长度过短的道路段。重新生成图以确保拓扑结构的正确性。
def infer_anchor(paths, net, region_lst, save_graph_dir, batch_size=2, save_pic=True,
                 max_iteration=99999999, verbose=True):
    print("infer anchor start")
    net.eval()
    if len(paths) < batch_size:
        batch_size = len(paths)
    print("batch_size:" + str(batch_size))
    output_flag_list = [False for _ in range(len(paths))]
    graph_dict = dict()

    iteration = 0
    pbar = tqdm(total=None, desc="graph exploration", unit="iter")

    try:
        for iteration in range(max_iteration):
            path_indices = []
            batch_extension_vertices = []
            batch_is_key_point = np.empty(batch_size)
            batch_inputs = np.empty(
                (batch_size, 3, cfg.TEST.WINDOW_SIZE, cfg.TEST.WINDOW_SIZE))
            batch_walked_path = np.empty(
                (batch_size, 1, cfg.TEST.WINDOW_SIZE, cfg.TEST.WINDOW_SIZE))
            batch_valid_trajectory_inputs = []

            for path_idx in range(len(paths)):
                if output_flag_list[path_idx]:
                    continue

                extension_vertex, is_key_point = paths[path_idx].pop(
                    follow_order=True)
                if extension_vertex is None:
                    output_flag_list[path_idx] = True
                    paths[path_idx].graph.save(os.path.join(
                        save_graph_dir,
                        '{}.graph'.format(region_lst[path_idx])))
                    tqdm.write("    save graph {}".format(region_lst[path_idx]))
                    graph_dict[region_lst[path_idx]] = paths[path_idx].graph
                    continue

                i = len(path_indices)
                path_indices.append(path_idx)
                batch_extension_vertices.append(extension_vertex)
                batch_is_key_point[i] = is_key_point

                fetch_list = ['aerial_image_chw', 'walked_path']
                fetch_list.extend(trajectory_fetch_fields(
                    TRAJECTORY_MODE, include_raster=False))
                if cfg.TEST.SAVE_EXAMPLES:
                    fetch_list += ['aerial_image_hwc']

                data_dict = paths[path_idx].make_path_input(
                    extension_vertex=extension_vertex,
                    fetch_list=fetch_list,
                    traj_filter=cfg.TRAIN.get("TRAJ_FILTER", False),
                    is_key_point=is_key_point,
                    WINDOW_SIZE=cfg.TEST.WINDOW_SIZE)
                data_dict = EasyDict(data_dict)
                batch_inputs[i] = data_dict.aerial_image_chw
                batch_walked_path[i] = data_dict.walked_path
                if USE_TRAJECTORY:
                    batch_valid_trajectory_inputs.append(data_dict.valid_trajectories)
                if len(path_indices) >= batch_size:
                    break

            if len(path_indices) == 0:
                pbar.set_postfix({
                    "active": 0,
                    "done": "{}/{}".format(sum(output_flag_list), len(paths)),
                    "vertices": sum(len(path.graph.vertices) for path in paths),
                    "edges": sum(len(path.graph.edges) for path in paths),
                })
                break

            length_path_indices = len(path_indices)
            batch_is_key_point = batch_is_key_point[:length_path_indices]
            batch_inputs = batch_inputs[:length_path_indices]
            batch_walked_path = batch_walked_path[:length_path_indices]
            batch_valid_trajectory_inputs = batch_valid_trajectory_inputs[:length_path_indices]

            batch_inputs_cuda = numpy2tensor2cuda(batch_inputs)
            batch_walked_path_cuda = numpy2tensor2cuda(batch_walked_path)
            batch_normalized_traj, batch_valid_mask = prepare_trajectory_sequence_batch(
                TRAJECTORY_MODE,
                batch_valid_trajectory_inputs if USE_TRAJECTORY else None,
                model_utils.valid_trajectory_input_GPU,
                model_utils.normalize_trajectory_batch,
            )

            batch_output_cuda_dict = net(
                aerial_image=batch_inputs_cuda,
                traj_image=None,
                aerial_traj_image=None,
                neighborhood_trajectory_norm=batch_normalized_traj,
                valid_mask=batch_valid_mask,
                walked_path=batch_walked_path_cuda,
                NUM_TARGETS=cfg.TEST.NUM_TARGETS,
                model=cfg.TRAIN.MODEL,
                use_traj=USE_TRAJECTORY)
            batch_output_road_cuda = batch_output_cuda_dict['road']
            batch_output_junc_cuda = batch_output_cuda_dict['junc']
            batch_output_anchor_maps_cuda = batch_output_cuda_dict['anchor']

            if batch_output_road_cuda.shape[-1] != cfg.TEST.WINDOW_SIZE:
                scale = cfg.TEST.WINDOW_SIZE / batch_output_road_cuda.shape[-1]
                batch_output_road_cuda = upsample(batch_output_road_cuda, scale)
            batch_output_road = torch.sigmoid(
                batch_output_road_cuda).detach().cpu().numpy()

            batch_output_anchor_maps = torch.sigmoid(
                batch_output_anchor_maps_cuda).detach().cpu().numpy()

            if cfg.TEST.SAVE_EXAMPLES and cfg.TEST.START_FROM_JUNC_PEAK:
                batch_output_junc = torch.sigmoid(
                    batch_output_junc_cuda).detach().cpu().numpy()

            batch_output_points = model_utils.map_to_coordinate(
                batch_output_maps=batch_output_anchor_maps.copy(),
                batch_is_key_point=batch_is_key_point,
                batch_extension_vertices=batch_extension_vertices,
                ROAD_SEG_THRESHOLE=cfg.TEST.BINARIZE_MAP.ROAD_SEG_THRESHOLE,
                STEP_LENGTH=cfg.TEST.STEP_LENGTH,
                JUNC_MAX_REGION_AREA=cfg.TEST.BINARIZE_MAP.JUNC_MAX_REGION_AREA)

            if verbose and iteration % cfg.TEST.PRINT_ITERATION == 0:
                tqdm.write('  iter:{} len(paths):{}'.format(
                    iteration, len(path_indices)))

            save_idx = cfg.TEST.SAVE_IDX
            if cfg.TEST.SAVE_EXAMPLES and save_idx in path_indices:
                for i in range(len(path_indices)):
                    region_name = region_lst[path_indices[i]]
                    os.makedirs(os.path.join(cfg.DIR.INFER_STEP_DIR,
                                             region_name), exist_ok=True)
                    fname = os.path.join(cfg.DIR.INFER_STEP_DIR,
                                         region_name, '{}_'.format(iteration))
                    pred_gt_pair_list = [
                        ("anchor", batch_output_anchor_maps[save_idx], None)]
                    pred_gt_pair_list.append(
                        ("road", batch_output_road[save_idx, 0], None))
                    pred_gt_pair_list.append(
                        ("junc", batch_output_junc[save_idx, 0], None))
                    paths[path_indices[save_idx]].visualize_output(
                        fname_prefix=fname,
                        extension_vertex=batch_extension_vertices[save_idx],
                        aerial_image=data_dict.aerial_image_hwc, target_poses=None,
                        pred_gt_pair_list=pred_gt_pair_list)

            for i in range(len(path_indices)):
                path_idx = path_indices[i]
                if len(batch_output_points[i]) > 0:
                    if hasattr(batch_extension_vertices[i], 'from_road_seg'):
                        batch_extension_vertices[i] = paths[path_idx].graph.add_vertex(
                            batch_extension_vertices[i].point)
                    paths[path_idx].push(
                        extension_vertex=batch_extension_vertices[i],
                        is_key_point=batch_is_key_point[i],
                        follow_mode=cfg.TEST.FOLLOW_MODE,
                        target_poses=None,
                        output_points=batch_output_points[i],
                        RECT_RADIUS=cfg.TEST.RECT_RADIUS,
                        road_segmentation=batch_output_road[i, 0],
                        NUM_TARGETS=cfg.TEST.NUM_TARGETS,
                        WINDOW_SIZE=cfg.TEST.WINDOW_SIZE,
                        STEP_LENGTH=cfg.TEST.STEP_LENGTH,
                        AVG_CONFIDENCE_THRESHOLD=cfg.TEST.AVG_CONFIDENCE_THRESHOLD)

            pbar.set_postfix({
                "active": len(path_indices),
                "done": "{}/{}".format(sum(output_flag_list), len(paths)),
                "vertices": sum(len(path.graph.vertices) for path in paths),
                "edges": sum(len(path.graph.edges) for path in paths),
            })
            pbar.update(1)
    finally:
        pbar.close()

    return iteration, graph_dict


def get_tile_data(region, cache, junction_nms_res=None, get_starting_locations=True):
    print('  region: {}'.format(region.name))
    num_tiles = cfg.TEST.get("NUM_TILES", 1)
    TILE_START = geom.Point(
        region.radius_x, region.radius_y).scale(cfg.TRAIN.IMG_SZ)
    # TILE_END = TILE_START.add(geom.Point(2, 2).scale(cfg.TRAIN.IMG_SZ))
    TILE_END = TILE_START.add(geom.Point(num_tiles, num_tiles).scale(cfg.TRAIN.IMG_SZ))
    search_rect = geom.Rectangle(TILE_START, TILE_END)
    starting_locations = []

    if get_starting_locations:
        pnts = list()
        if cfg.TEST.INFER_STEP == "given_junc_nms":
            for x in range(region.radius_x, region.radius_x + num_tiles):
                for y in range(region.radius_y, region.radius_y + num_tiles):
                    fname = '{}_{}_{}.png'.format(region.name, x, y)
                    junc_nms_map = cv.imread(os.path.join(
                        cfg.DIR.PRE_JUNC_NMS_DIR, fname), 0)
                    tmp_pnts = list(zip(*np.where(junc_nms_map > 0)))
                    tmp_pnts = [geom.Point(pnt[1] + x * cfg.TRAIN.IMG_SZ, pnt[0] + y * cfg.TRAIN.IMG_SZ)
                                for pnt in tmp_pnts]
                    pnts.extend(tmp_pnts)
        elif cfg.TEST.INFER_STEP == "after_junc_nms":
            junc_nms_map = cv.imread(os.path.join(
                cfg.DIR.SAVE_SEG_DIR, cfg.TEST.CKPT, "junc_nms", region.name + '.png'), 0).astype(np.float32)
            pnts = list(zip(*np.where(junc_nms_map > 0)))
            pnts = [geom.Point(pnt[1] + region.radius_x * cfg.TRAIN.IMG_SZ,
                               pnt[0] + region.radius_y * cfg.TRAIN.IMG_SZ)
                    for pnt in pnts]
        elif cfg.TEST.INFER_STEP == "start" or cfg.TEST.INFER_STEP == "after_seg":
            pnts = [geom.Point(pnt[1] + region.radius_x * cfg.TRAIN.IMG_SZ, pnt[0] + region.radius_y * cfg.TRAIN.IMG_SZ)
                    for pnt in junction_nms_res[region.name]]

        for pnt in pnts:
            if not search_rect.contains(pnt):
                continue
            starting_locations.append([{
                'point': pnt,
                'edge_pos': None,
                'key_point': True
            }])

    return {
        'region': region.name,
        'search_rect': search_rect,
        'cache': cache,
        'starting_locations': {
            'junction': starting_locations,
            'middle': []
        },
        'gc': None
    }


def post_process_graph(graph_dict):
    save_dir = os.path.join(
        cfg.DIR.SAVE_GRAPH_DIR,
        '{}_{}'.format(cfg.TEST.CKPT, cfg.TEST.NUM_TARGETS),
        'post'
    )
    os.makedirs(save_dir, exist_ok=True)
    for region_name, g in graph_dict.items():
        bad_edges = set()
        road_segments, _ = graph_helper.get_graph_road_segments(g)
        for rs in road_segments:
            if rs.marked_length < 2 * cfg.TEST.STEP_LENGTH and \
                    (len(rs.src(g).in_edges_id) <= 1 or len(rs.dst(g).in_edges_id) <= 1):
                for edge in rs.edges(g):
                    bad_edges.add(edge)
        ng = graph_helper.Graph()
        seen_pnts = dict()
        for edge in g.edges.values():
            if edge in bad_edges:
                continue
            if edge.src(g).point == edge.dst(g).point:
                continue
            src_dst = []
            for pnt in [edge.src(g).point, edge.dst(g).point]:
                if pnt not in seen_pnts:
                    v = ng.add_vertex(pnt)
                    seen_pnts[pnt] = v.id
                src_dst.append(seen_pnts[pnt])
            ng.add_edge(src_dst[0], src_dst[1])
        ng.save(os.path.join(save_dir, '{}.graph'.format(
            region_name)), clear_self=False)


def prepare_net():
    print('initializing model')
    net = RPNet(cfg.TRAIN.NUM_TARGETS)
    net = net.cuda()
    file_name = os.path.join(cfg.DIR.CHECK_POINT_DIR,
                             '{}.pth.tar'.format(cfg.TEST.CKPT))

    # analyze_checkpoint(file_name)

    if os.path.isfile(file_name):
        net = load_pretrained(net, file_name)
    else:
        raise FileNotFoundError("checkpoint not found: {}".format(file_name))
    if cfg.TEST.DATA_PARALLEL:
        net = torch.nn.DataParallel(net)
    return net


def generate_sample_lst(IMG_SZ, CROP_SZ, SAMPLE_STEP=2):
    CROP_SAMPLE_LST = []
    rows = list(range(0, IMG_SZ - CROP_SZ + 1, CROP_SZ // SAMPLE_STEP))
    cols = list(range(0, IMG_SZ - CROP_SZ + 1, CROP_SZ // SAMPLE_STEP))
    for r in rows:
        for c in cols:
            CROP_SAMPLE_LST.append((r, c))
    return CROP_SAMPLE_LST


def infer_segmentation(net, region_names):
    start_time = time.time()
    trans = transforms.ToTensor()
    CROP_SAMPLE_LST = generate_sample_lst(
        cfg.TEST.TEST_IMG_SZ, cfg.TEST.CROP_SZ, cfg.TEST.get("SAMPLE_STEP", 2))
    cuda_device_num = torch.cuda.device_count()
    road_map_dict = dict()
    junc_map_dict = dict()
    batch_input_dict = dict()
    for num, region_name in enumerate(region_names):
        print("[{:2d}/{:2d}] {}".format(num, len(region_names), region_name))
        img_map = np.array(Image.open(os.path.join(
            cfg.DIR.TEST_IMAGERY_DIR, region_name + ".png")))
        img_map = img_map.swapaxes(0, 1)
        img_map = trans(img_map)
        img_map = torch.unsqueeze(img_map, 0) # (b,c,w,h)
        traj_map = None
        if USE_TRAJECTORY and cfg.TRAIN.MODEL == 'DSFNet':
            traj_path = os.path.join(cfg.DIR.TEST_TRAJ_DIR, region_name + ".png")
            if not os.path.isfile(traj_path):
                raise FileNotFoundError("trajectory test image not found: {}".format(traj_path))
            traj_map = np.array(Image.open(traj_path))
            traj_map = traj_map.swapaxes(0, 1)
            traj_map = trans(traj_map)
            traj_map = torch.unsqueeze(traj_map, 0)

        container = {}
        container['road'] = MapContainer(os.path.join(cfg.DIR.SAVE_SEG_DIR, cfg.TEST.CKPT, "road"),
                                         region_name, cfg.TEST.TEST_IMG_SZ)
        container['junc'] = MapContainer(os.path.join(cfg.DIR.SAVE_SEG_DIR, cfg.TEST.CKPT, "junction"),
                                         region_name, cfg.TEST.TEST_IMG_SZ)

        # TODO 这里为了符合模型训练时256的输入，要把切片改为256
        pnt_index = 0
        skipped_empty_crops = 0
        pbar = tqdm(total=len(CROP_SAMPLE_LST))
        while pnt_index < len(CROP_SAMPLE_LST):
            input_var, input_traj_var = None, None

            pnt_lst = CROP_SAMPLE_LST[pnt_index:pnt_index + cfg.TEST.BATCH_SIZE_SEG]
            # bug: DataParallel, must feed something into every gpu
            if len(pnt_lst) < cuda_device_num:
                pnt_lst = CROP_SAMPLE_LST[:-cuda_device_num]
            batch_input = []
            batch_traj_input = []
            batch_pnt_lst = []
            blank_pnt_lst = []

            for pnt in pnt_lst:
                crop_img = img_map[:, :, pnt[0]:pnt[0] + cfg.TEST.CROP_SZ, pnt[1]:pnt[1] + cfg.TEST.CROP_SZ]
                if cfg.TEST.get("SKIP_EMPTY_CROP", False) and crop_img.sum().item() == 0:
                    blank_pnt_lst.append(pnt)
                else:
                    batch_input.append(crop_img)
                    batch_pnt_lst.append(pnt)

            if len(blank_pnt_lst) > 0:
                blank_maps = np.zeros((len(blank_pnt_lst), 1, cfg.TEST.CROP_SZ, cfg.TEST.CROP_SZ), dtype=np.float32)
                container['road'].add_batch_cpu(blank_pnt_lst, blank_maps, cfg.TEST.CROP_SZ)
                container['junc'].add_batch_cpu(blank_pnt_lst, blank_maps, cfg.TEST.CROP_SZ)
                skipped_empty_crops += len(blank_pnt_lst)

            if len(batch_input) == 0:
                pnt_index += len(pnt_lst)
                pbar.update(len(pnt_lst))
                continue

            batch_input = torch.cat(batch_input, dim=0)
            input_var = torch.autograd.Variable(batch_input).cuda()

            if USE_TRAJECTORY and cfg.TRAIN.MODEL == 'DSFNet':
                for pnt in batch_pnt_lst:
                    crop_traj = traj_map[:, :, pnt[0]:pnt[0] + cfg.TEST.CROP_SZ, pnt[1]:pnt[1] + cfg.TEST.CROP_SZ]
                    batch_traj_input.append(crop_traj)
                batch_traj_input = torch.cat(batch_traj_input, dim=0)
                input_traj_var = torch.autograd.Variable(batch_traj_input).cuda()

            res = net(aerial_image=input_var, traj_image=input_traj_var, aerial_traj_image=None, walked_path=None, neighborhood_trajectory_norm=None, valid_mask=None, test=True, model=cfg.TRAIN.MODEL, use_traj=USE_TRAJECTORY)
            # TODO 这里推理的过程需不需要加上轨迹数据
            road, junc = res['road'], res['junc']

            container['road'].add_batch_gpu(batch_pnt_lst, road, cfg.TEST.CROP_SZ)
            container['junc'].add_batch_gpu(batch_pnt_lst, junc, cfg.TEST.CROP_SZ)

            pnt_index += len(pnt_lst)
            pbar.update(len(pnt_lst))
        pbar.close()
        if skipped_empty_crops > 0:
            print("  skipped {} all-zero crops".format(skipped_empty_crops))
        for item in container.values():
            item.close()
            item.save_map()
        road_map_dict[region_name] = container['road'].get_map().swapaxes(0, 1)
        print('road_map_dict[region_name] max min', road_map_dict[region_name].max(), road_map_dict[region_name].min())
        junc_map_dict[region_name] = container['junc'].get_map().swapaxes(0, 1)

    duration = time.time() - start_time
    print('{} images, img_sz: {}, infer time: {}, speed: {}fps'.format(
        len(region_names), cfg.TEST.TEST_IMG_SZ, duration, len(region_names) / duration))
    return road_map_dict, junc_map_dict


def junction_nms(region_name, junc_map):
    print("  region: {}".format(region_name))
    junc_pnts = list()
    res_map = np.zeros(junc_map.shape)
    vis_map = np.zeros((junc_map.shape[0], junc_map.shape[1], 3))
    # 叠加可视化 观察junc结果
    junc_map = (junc_map * 255.).astype(np.uint8)
    # heat_color = cv.applyColorMap(junc_map, cv.COLORMAP_JET)
    # batch_input = batch_input.squeeze()
    # batch_input = batch_input.transpose(1, 2, 0)
    # batch_input = (batch_input * 255.).swapaxes(0, 1).astype(np.uint8)
    # fused_img = cv.addWeighted(batch_input, 0.5, heat_color, 0.5, 0)
    # 阈值筛选
    if np.max(junc_map) > 1:
        vis_map[:, :, 1] = junc_map
        junc_map[np.where(
            junc_map < cfg.TEST.BINARIZE_MAP.JUNC_SEG_THRESHOLE * 255)] = 0
    else:
        vis_map[:, :, 1] = junc_map * 255
        junc_map[np.where(
            junc_map < cfg.TEST.BINARIZE_MAP.JUNC_SEG_THRESHOLE)] = 0
    junc_map[np.where(junc_map)] = 1
    # 标记连通区域然后提取质心
    labels = measure.label(junc_map, connectivity=2)
    props = measure.regionprops(labels)
    for region in props:
        if region.area > cfg.TEST.BINARIZE_MAP.ANCHOR_MAX_REGION_AREA:
            continue
        center = (int(region.centroid[0]), int(region.centroid[1]))
        res_map[center] = 255
        cv.circle(vis_map, (center[1], center[0]),
                  radius=7, color=(0, 0, 255), thickness=-1)
        # cv.circle(fused_img, (center[1], center[0]),
        #           radius=7, color=(0, 0, 255), thickness=-1)
        junc_pnts.append(center)
    cv.imwrite(os.path.join(cfg.DIR.SAVE_SEG_DIR, cfg.TEST.CKPT,
                            "junc_nms", region_name + ".png"), res_map)
    cv.imwrite(os.path.join(cfg.DIR.SAVE_SEG_DIR, cfg.TEST.CKPT,
                            "junc_nms_vis", region_name + ".png"), vis_map)
    # cv.imwrite(os.path.join(cfg.DIR.SAVE_SEG_DIR, cfg.TEST.CKPT,
    #                         "junc_nms_vis", region_name + "add.png"), fused_img)
    print("  region: {}, junction starts: {}".format(region_name, len(junc_pnts)))
    return junc_pnts


def road_seg_region_filter(region_name, road_seg):
    frame = road_seg.copy()
    frame[np.where(frame < cfg.TEST.BINARIZE_MAP.ROAD_SEG_THRESHOLE)] = 0
    frame[np.where(frame)] = 1
    frame = frame.astype(np.uint8)
    labels = measure.label(frame, connectivity=2)
    props = measure.regionprops(labels)
    for region in props.copy():
        print("region.area:", region.area)
        if region.area < cfg.TEST.BINARIZE_MAP.MIN_BAD_ROAD_AREA:
            frame[tuple(region.coords.swapaxes(0, 1))] = 0
            props.remove(region)
    frame = cv.bitwise_and(road_seg, road_seg, mask=frame)

    frame[np.where(frame)] = 1

    cv.imwrite(os.path.join(cfg.DIR.SAVE_SEG_DIR, cfg.TEST.CKPT, "road_seg_region_filter", region_name + ".png"),frame*255.)

    # # 叠加可视化 观察junc结果
    # road_seg_vis = road_seg.copy()
    # road_seg_vis = np.clip(road_seg_vis, 0, 1)
    # vis_img = batch_input.copy()
    # vis_img = vis_img.squeeze().transpose(1, 2, 0).swapaxes(0, 1)
    # # 使用灰度叠加在 G 通道（绿色）上
    # vis_img[:, :, 1] = np.clip(vis_img[:, :, 1] + 0.4 * road_seg_vis, 0, 1)
    # # 将 frame_filtered 叠加为红色主干道
    # frame_mask = frame.copy()
    # frame_mask = np.clip(frame_mask, 0, 1)
    # # 红色通道叠加高亮主干道
    # vis_img[:, :, 0] = np.clip(vis_img[:, :, 0] + 0.6 * frame_mask, 0, 1)
    # # 保存最终合成图像
    # cv.imwrite(os.path.join(cfg.DIR.SAVE_SEG_DIR, cfg.TEST.CKPT, "road_seg_region_filter", region_name + "add.png"), vis_img * 255.)
    return frame


if __name__ == "__main__":
    with torch.no_grad():
        main()
