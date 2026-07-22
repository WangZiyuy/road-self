import json
import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import utils.model_utils as model_utils
from utils.tileloader import Tiles
from easydict import EasyDict
from utils.gis_to_graph import GisToGraphConverter
from utils.trajectory_mode import (
    TRAJ_MODE_NONE,
    load_region_trajectory_inputs_for_mode,
    resolve_trajectory_mode,
    trajectory_fetch_fields,
)
import time

class OSMDataset:

    def __init__(self, cfg, net=None, training=True, seg_input=None):
        self.cfg = cfg
        self.trajectory_mode = resolve_trajectory_mode(cfg)
        self.use_trajectory = self.trajectory_mode != TRAJ_MODE_NONE
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.window_size = cfg.TRAIN.WINDOW_SIZE
        self.input_channels = cfg.TRAIN.NUM_INPUT_CHANNELS
        self.input_traj_channels = (
            cfg.TRAIN.get("NUM_INPUT_TRAJECTORY_CHANNELS", 1)
            if self.use_trajectory else 0
        )
        self.seg_input = seg_input
        self.num_targets = cfg.TRAIN.NUM_TARGETS
        self.paths = []
        self.tiles = Tiles(training_regions=self.cfg.TRAIN.TRAINING_REGIONS,
                           parallel_tiles=self.cfg.TRAIN.PARALLEL_TILES,
                           region_path=cfg.DIR.ALL_REGION_PATH,
                           graph_dir=cfg.DIR.GRAPH_DIR,
                           tile_dir=cfg.DIR.TILE_DIR,
                           traj_dir=(cfg.DIR.get("TRAJ_DIR", None)
                                     if self.use_trajectory else None), )
        self.save_idx = 0
        self.training = training
        self.net = net # 用于传递给path从而传递给model_utils文件中的轨迹过滤方法传递两个半径自定义参数

        self.subtiles = self.tiles.prepare_training()
        print("extracted {} subtiles from {} tiles (missing {})".format(
            len(self.subtiles), len(self.tiles.train_tiles), 4 * len(self.tiles.train_tiles) - len(self.subtiles)))

        print("loading initial paths")
        for i, subtile in enumerate(self.subtiles):
            print("In region:{}, in tile {}{}".format(subtile["region"], subtile['search_rect'].start, subtile['search_rect'].end))

            self.all_trajectories, self.all_pixel_trajectories, self.traj_grid_index, self.traj_grid_cell_size = \
                load_region_trajectory_inputs_for_mode(
                    self.trajectory_mode,
                    subtile["region"],
                    self.cfg,
                    load_region_trajectory_inputs,
                )

            path = model_utils.Path(
                i, training, subtile["gc"].clone(), subtile,
                self.all_trajectories, self.all_pixel_trajectories,
                net=self.net,
                traj_grid_index=self.traj_grid_index,
                traj_grid_cell_size=self.traj_grid_cell_size)
            self.paths.append(path)

            # 将某个region大图的四个2048小图中的路径进行可视化
            path.visualize_and_save_path(i, subtile, save_path=subtile["region"])

        # # 可视化某个大图代码，验证路网生成的效果，以及和遥感图像的契合程度
        # # 将上述四个拼接成4096的大图
        # # 此处以Columbus的（1，1）大图为例
        # big_rect_path_vis = np.zeros((4096, 4096, 3))
        # fold = f'./data_self/coincidence_test /'
        # region_fold = f'./data_self/coincidence_test/{subtile["region"]}/'
        # tile_size = 2048
        # positions = [
        #     (0, 0),
        #     (tile_size, 0),
        #     (0, tile_size),
        #     (tile_size, tile_size),
        # ]
        #
        # for j, filename in enumerate(os.listdir(fold)):
        #     # 读取当前图像
        #     if filename.split('_')[0] == subtile["region"]:
        #         file_path = os.path.join(fold, filename)
        #         if os.path.isfile(file_path):
        #             img = cv.imread(os.path.join(fold, filename))
        #             # 将图片填充到大图的相应位置
        #             k = int(filename.split('_')[1][0])
        #             start_x, start_y = positions[k]
        #             print("da tu tian chong : ", filename, start_x, start_y)
        #             big_rect_path_vis[start_x:start_x + tile_size, start_y:start_y + tile_size] = img
        #
        # save_path = os.path.join(region_fold, subtile["region"] + 'big.png')
        # cv.imwrite(save_path, big_rect_path_vis)
        # print(f"整合4096大图像已保存至: {save_path}")
        #
        # # 将遥感图和path重合
        # remote_sensing_image = cv.imread(f'./data_self/input/imagery/{subtile["region"]}_0_0.png')[:4096, :4096, :]
        # print('\n')
        # print(big_rect_path_vis.shape, remote_sensing_image.shape)
        # overlay_image = cv.addWeighted(big_rect_path_vis.astype(np.float32), 1.0, remote_sensing_image.astype(np.float32), 0.6, 0)
        # cv.imwrite(os.path.join(region_fold, subtile["region"] + 'overlay_image.png'), overlay_image)
        # print(f"路径和遥感叠加大图像已保存至: {save_path}")

    def warm_up(self):
        print("warm up now:")
        for path_idx in tqdm(range(len(self.paths))):
            path = self.paths[path_idx]
            for i in range(random.randint(self.cfg.TRAIN.MAX_PATH_LENGTH//4, self.cfg.TRAIN.MAX_PATH_LENGTH)):
                while True:
                    extension_vertex, is_key_point = path.pop(follow_order=False, probs=[0.2, 0.8, 0],
                                                              WINDOW_SIZE=self.window_size)
                    if extension_vertex is None or len(path.graph.vertices) >= self.cfg.TRAIN.MAX_PATH_LENGTH:
                        self.paths[path_idx] = model_utils.Path(
                            idx=path_idx, training=self.training, gc=self.subtiles[path_idx]["gc"].clone(),
                            tile_data=self.subtiles[path_idx],
                            all_trajectories=path.all_trajectories,
                            all_pixel_trajectories=path.all_pixel_trajectories,
                            net=self.net,
                            traj_grid_index=path.traj_grid_index,
                            traj_grid_cell_size=path.traj_grid_cell_size)
                        path = self.paths[path_idx]
                        continue
                    break
                target_poses = path.get_target_poses(
                    extension_vertex=extension_vertex, road_segmentation=None,
                    STEP_LENGTH=self.cfg.TRAIN.STEP_LENGTH, is_key_point=is_key_point,
                    NUM_TARGETS=self .num_targets, RECT_RADIUS=self.cfg.TRAIN.RECT_RADIUS,
                    WINDOW_SIZE=self.window_size)
                if extension_vertex.edge_pos is None:
                    continue
                if len(target_poses) == 0:
                    continue
                if is_key_point:
                    length = len(target_poses.target_poses[0])
                    if length > 0:
                        target_poses.target_poses[0] = \
                            random.sample(target_poses.target_poses[0], random.randint(1, length))
                path.push(
                    extension_vertex=extension_vertex, is_key_point=is_key_point,
                    follow_mode=self.cfg.TRAIN.FOLLOW_MODE, target_poses=target_poses,
                    output_points=None,
                    RECT_RADIUS=self.cfg.TRAIN.RECT_RADIUS,
                    road_segmentation=None,
                    NUM_TARGETS=self.cfg.TRAIN.NUM_TARGETS, WINDOW_SIZE=self.cfg.TRAIN.WINDOW_SIZE,
                    STEP_LENGTH=self.cfg.TRAIN.STEP_LENGTH,
                    AVG_CONFIDENCE_THRESHOLD=self.cfg.TRAIN.AVG_CONFIDENCE_THRESHOLD)

    def get_batch(self):
        """
        Returns：
        返回一个batch需要的数据“
        主要有输入遥感图像、道路分割结果、下一顶点位置、已有路径等

        """
        path_indices = random.sample(range(len(self.paths)), self.batch_size)
        # len(self.paths)=96 中取20
        # 这里有一个问题，为什么subtile的范围是2048*2048（也就是在这样的范围内取path）
        # 但是后面的batch数据的范围都是256*256，64*64

        batch_extension_vertices = []
        batch_inputs = np.zeros((self.batch_size, self.input_channels, self.window_size, self.window_size))
        batch_traj_inputs = None
        batch_aerial_traj = None
        if self.use_trajectory:
            batch_traj_inputs = np.zeros(
                (self.batch_size, self.input_traj_channels,
                 self.window_size, self.window_size))
            batch_aerial_traj = np.zeros(
                (self.batch_size, self.input_channels + self.input_traj_channels,
                 self.window_size, self.window_size))
        batch_target_maps = np.zeros((self.batch_size, self.num_targets, self.window_size, self.window_size))
        batch_is_key_point = np.zeros(self.batch_size)
        batch_end_index = np.zeros(self.batch_size, dtype=np.int64)
        batch_target_poses = []
        default_shape = (self.batch_size, 1, self.window_size, self.window_size)
        batch_walked_path_small = np.zeros((self.batch_size, 1, self.window_size // 4, self.window_size // 4))
        batch_walked_path = np.zeros((self.batch_size, 1, self.window_size, self.window_size))
        batch_road_segmentation = np.zeros((self.batch_size, 1, self.window_size // 4, self.window_size // 4))
        batch_road_segmentation_thick3 = np.zeros(default_shape)
        batch_junction_segmentation = np.zeros((self.batch_size, 1, self.window_size // 4, self.window_size // 4))
        batch_junction_segmentation_thick3 = np.zeros(default_shape)
        batch_aerial_images_hwc = []
        batch_traj_images_hwc = []
        batch_valid_trajectory_inputs = []

        # 遍历每个路径索引(随机遍历)，从路径列表中获取相应的路径，路径列表由subtile产生,来源是tileloader的prepare_training
        for i in range(len(path_indices)):
            path_idx = path_indices[i]
            path = self.paths[path_idx]

            # 使用 path.pop 方法生成一个扩展顶点。
            # 如果扩展顶点为空或路径的顶点数量超过最大长度，则重新初始化该路径并继续生成顶点
            while True:
                extension_vertex, is_key_point = path.pop(follow_order=False, probs=[0.15, 0.8, 0.05],
                                                          WINDOW_SIZE=self.window_size)

                if extension_vertex is None or len(path.graph.vertices) >= self.cfg.TRAIN.MAX_PATH_LENGTH:
                    self.paths[path_idx] = model_utils.Path(
                        idx=path_idx, training=self.training, gc=self.subtiles[path_idx]["gc"].clone(),
                        tile_data=self.subtiles[path_idx], all_trajectories=self.all_trajectories,
                        all_pixel_trajectories=self.all_pixel_trajectories, net=self.net,
                        traj_grid_index=getattr(self, "traj_grid_index", None),
                        traj_grid_cell_size=getattr(self, "traj_grid_cell_size", None))
                    path = self.paths[path_idx]
                    continue
                break

            fetch_list = ['aerial_image_chw',
                          'aerial_image_hwc',
                          'walked_path_small',
                          'walked_path',
                          'road_seg_small',
                          'road_seg_thick3',
                          'junc_seg_small',
                          'junc_seg_thick3']
            fetch_list.extend(trajectory_fetch_fields(
                self.trajectory_mode, include_raster=True))

            data_dict = path.make_path_input(extension_vertex=extension_vertex,
                                             fetch_list=fetch_list,
                                             traj_filter=self.cfg.TRAIN.get("TRAJ_FILTER", False),
                                             is_key_point=is_key_point,
                                             WINDOW_SIZE=self.window_size,
                                             )

            data_dict = EasyDict(data_dict)

            # 获取目标位置
            target_poses = self.paths[path_idx].get_target_poses(
                extension_vertex=extension_vertex, road_segmentation=data_dict.road_seg_thick3[0],
                STEP_LENGTH=self.cfg.TRAIN.STEP_LENGTH, is_key_point=is_key_point,
                NUM_TARGETS=self.num_targets, RECT_RADIUS=self.cfg.TRAIN.RECT_RADIUS,
                WINDOW_SIZE=self.window_size)  # edge_pos list

            batch_aerial_images_hwc.append(data_dict.aerial_image_hwc)
            batch_extension_vertices.append(extension_vertex)

            # 输入确定
            batch_inputs[i] = data_dict.aerial_image_chw
            if self.use_trajectory:
                batch_traj_images_hwc.append(data_dict.traj_image_hwc)
                batch_traj_inputs[i] = data_dict.traj_image_chw
                batch_aerial_traj[i] = np.concatenate(
                    (batch_inputs[i], batch_traj_inputs[i]), axis=0)
            batch_walked_path_small[i] = data_dict.walked_path_small
            batch_walked_path[i] = data_dict.walked_path
            batch_road_segmentation[i] = data_dict.road_seg_small
            batch_road_segmentation_thick3[i] = data_dict.road_seg_thick3
            batch_junction_segmentation[i] = data_dict.junc_seg_small
            batch_junction_segmentation_thick3[i] = data_dict.junc_seg_thick3
            batch_target_poses.append(target_poses)
            batch_is_key_point[i] = is_key_point
            batch_end_index[i] = 1 if is_key_point else target_poses.get_supervision_end_index()
            target_maps = path.generate_target_maps(extension_vertex, target_poses, self.num_targets,
                                                    self.window_size,
                                                    is_key_point)
            batch_target_maps[i] = target_maps
            # batch_valid_trajectory_inputs.append(model_utils.valid_trajectory_input(data_dict.valid_trajectories))
            if self.use_trajectory:
                batch_valid_trajectory_inputs.append(data_dict.valid_trajectories)

        data = EasyDict({
            'path_indices': path_indices,
            'batch_extension_vertices': batch_extension_vertices,
            'batch_inputs': batch_inputs,
            'batch_target_maps': batch_target_maps,
            'batch_is_key_point': batch_is_key_point,
            'batch_end_index': batch_end_index,
            'batch_target_poses': batch_target_poses,
            'batch_walked_path_small': batch_walked_path_small,
            'batch_walked_path': batch_walked_path,
            'batch_road_segmentation': batch_road_segmentation,
            'batch_road_segmentation_thick3': batch_road_segmentation_thick3,
            'batch_junction_segmentation': batch_junction_segmentation,
            'batch_junction_segmentation_thick3': batch_junction_segmentation_thick3,
            'batch_aerial_images_hwc': batch_aerial_images_hwc,
        })
        if self.use_trajectory:
            data.update({
                'batch_traj_inputs': batch_traj_inputs,
                'batch_aerial_traj': batch_aerial_traj,
                'batch_traj_images_hwc': batch_traj_images_hwc,
                'batch_valid_trajectory_inputs': batch_valid_trajectory_inputs,
            })
        return data

    def push_and_vis_batch(self, res_dict, outer_it, path_it):

        """
        筛选关键点；
        进行绘图输出；
        改变了path中batch_target_poses和batch_output_points属性？

         """

        if self.cfg.TRAIN.FOLLOW_MODE == "follow_output":
            # 一批地图数据中提取有效的坐标点，判断哪些点是关键点，并根据一定的规则进行过滤和选择
            batch_output_points = \
                model_utils.map_to_coordinate(
                    batch_output_maps=res_dict.batch_output_anchor_maps.copy(),
                    batch_is_key_point=res_dict.batch_is_key_point,
                    batch_extension_vertices=res_dict.batch_extension_vertices,
                    SEGMENTATION_THRESHOLD=self.cfg.TRAIN.BINARIZE_MAP.SEGMENTATION_THRESHOLD,
                    STEP_LENGTH=self.cfg.TRAIN.STEP_LENGTH,
                    MAX_REGION_AREA=self.cfg.TRAIN.BINARIZE_MAP.MAX_REGION_AREA)

        if self.cfg.TRAIN.SAVE_EXAMPLES and self.save_idx in res_dict.path_indices:
            x = res_dict.path_indices.index(self.save_idx)
            fname = os.path.join(self.cfg.DIR.SHORTCUT_DIR,
                                 "{}_{}_{}_".format(res_dict.path_indices[x], outer_it, path_it))

            self.paths[res_dict.path_indices[x]].visualize_output(
                fname_prefix=fname,
                extension_vertex=res_dict.batch_extension_vertices[x],
                aerial_image=res_dict.batch_aerial_images_hwc[x], target_poses=res_dict.batch_target_poses[x],
                pred_gt_pair_list=[
                    ("anchor", res_dict.batch_output_anchor_maps[x], res_dict.batch_target_maps[x]),
                    ("road", res_dict.batch_output_road[x, 0], res_dict.batch_road_segmentation_thick3[x, 0]),
                    ("junc", res_dict.batch_output_junc[x, 0], res_dict.batch_junction_segmentation_thick3[x, 0])
                ])

        for i in range(len(res_dict.path_indices)):
            if res_dict.batch_extension_vertices[i].edge_pos is None:
                continue
            if len(res_dict.batch_target_poses[i]) == 0:
                continue
            path_idx = res_dict.path_indices[i]
            if res_dict.batch_is_key_point[i]:
                if self.cfg.TRAIN.FOLLOW_MODE == "follow_target":
                    length = len(res_dict.batch_target_poses[i].target_poses[0])
                    if length > 0:
                        res_dict.batch_target_poses[i].target_poses[0] = \
                            random.sample(res_dict.batch_target_poses[i].target_poses[0], random.randint(1, length))
                elif self.cfg.TRAIN.FOLLOW_MODE == "follow_output":
                    length = len(batch_output_points[i])
                    if length > 0:
                        batch_output_points[i] = \
                            random.sample(batch_output_points[i], random.randint(1, length))
            self.paths[path_idx].push(
                extension_vertex=res_dict.batch_extension_vertices[i], is_key_point=res_dict.batch_is_key_point[i],
                follow_mode=self.cfg.TRAIN.FOLLOW_MODE, target_poses=res_dict.batch_target_poses[i],
                output_points=batch_output_points[i] if self.cfg.TRAIN.FOLLOW_MODE == "follow_output" else None,
                RECT_RADIUS=self.cfg.TRAIN.RECT_RADIUS, road_segmentation=res_dict.batch_road_segmentation_thick3[i, 0],
                NUM_TARGETS=self.cfg.TRAIN.NUM_TARGETS, WINDOW_SIZE=self.cfg.TRAIN.WINDOW_SIZE,
                STEP_LENGTH=self.cfg.TRAIN.STEP_LENGTH,
                AVG_CONFIDENCE_THRESHOLD=self.cfg.TRAIN.AVG_CONFIDENCE_THRESHOLD)
        return

# 新加，用于读取成条轨迹数据。原始轨迹允许变长，进入模型前会在
# valid_trajectory_input_GPU 中 pad 成固定 batch tensor。
def _iter_traj_files(trajectory_dir, suffixes):
    if not os.path.isdir(trajectory_dir):
        raise FileNotFoundError(f"trajectory directory not found: {trajectory_dir}")
    result = []
    for root, _, files in os.walk(trajectory_dir):
        for file_name in files:
            if file_name.lower().endswith(suffixes):
                result.append(os.path.join(root, file_name))
    return sorted(result)


def _filter_traj_points(coordinates, bbox=None, min_points=2):
    if len(coordinates) == 0:
        return None
    coordinates = np.asarray(coordinates, dtype=np.float32)
    finite_mask = np.isfinite(coordinates).all(axis=1)
    coordinates = coordinates[finite_mask]
    if bbox is not None and len(coordinates) > 0:
        lat_min, lon_min, lat_max, lon_max = bbox
        bbox_mask = (
            (coordinates[:, 0] >= lat_min) & (coordinates[:, 0] <= lat_max) &
            (coordinates[:, 1] >= lon_min) & (coordinates[:, 1] <= lon_max)
        )
        coordinates = coordinates[bbox_mask]
    if len(coordinates) < min_points:
        return None
    return coordinates


def read_legacy_csv_traj(file_path, bbox=None):
    data = pd.read_csv(
        file_path,
        header=None,
        skiprows=[0],
        usecols=[1, 2],
        dtype=str,
        on_bad_lines="skip",
    )
    data = data.apply(pd.to_numeric, errors="coerce")
    return _filter_traj_points(data.dropna().to_numpy(), bbox=bbox)


def read_didi_gaia_txt_traj(file_path, bbox=None, point_source="raw"):
    usecols = [1, 2] if point_source == "raw" else [4, 5]
    data = pd.read_csv(
        file_path,
        header=None,
        skiprows=[0],
        usecols=usecols,
        dtype=str,
        engine="c",
        on_bad_lines="skip",
    )
    data = data.apply(pd.to_numeric, errors="coerce")
    return _filter_traj_points(data.dropna().to_numpy(), bbox=bbox)


def get_region_bbox_for_traj(region_num, data_root="data_self"):
    min_lat, max_lat, min_lng, max_lng, _, _, _, _ = GisToGraphConverter(
        region_num, [], data_root=data_root).get_trans_para()
    return min_lat, min_lng, max_lat, max_lng


def get_all_traj_pieces_from_txt(trajectory_dir, bbox=None, txt_point_source="raw"):
    """
    Read trajectory files as a list of [lat, lon] arrays.

    CSV format: header + rows where columns 1/2 are lat/lon.
    Didi/Gaia TXT format: header + rows where columns 1/2 are raw GCJ02
    lat/lon and columns 4/5 are map-matched lat/lon.
    """
    print("loading traj in pieces")
    traj_files = _iter_traj_files(trajectory_dir, (".csv", ".txt"))
    all_trajectories = []
    skipped = 0
    for file_path in tqdm(traj_files, desc="load trajectory files"):
        ext = os.path.splitext(file_path)[1].lower()
        try:
            if ext == ".txt":
                coordinates = read_didi_gaia_txt_traj(
                    file_path, bbox=bbox, point_source=txt_point_source)
            else:
                coordinates = read_legacy_csv_traj(file_path, bbox=bbox)
        except Exception:
            skipped += 1
            continue
        if coordinates is None:
            skipped += 1
            continue
        all_trajectories.append(coordinates)
    print(f"trajectory count: {len(all_trajectories)}, skipped files: {skipped}")
    return all_trajectories


def all_traj_to_all_pixel_traj(all_trajectories, region_num, data_root="data_self"):
    """
    将轨迹经纬度坐标转换为像素坐标。这里保持 list，因为每条原始轨迹
    长度不同；局部窗口轨迹会在模型输入前统一 pad。
    """
    all_pixel_trajectories = []

    for _, traj in enumerate(tqdm(all_trajectories, desc="trans all traj to all pixel traj")):
        converter = GisToGraphConverter(region_num, traj, data_root=data_root)
        pixel_trajectories = converter.convert_trajectories_to_pixels()
        all_pixel_trajectories.append(pixel_trajectories)

    return all_pixel_trajectories


def _default_prepared_traj_dir(region_num, data_root="data_self"):
    return os.path.join(data_root, "input", "traj_prepared", region_num)


def _prepared_traj_dir(region_num, cfg):
    prepared_root = cfg.TRAIN.get("TRAJ_PREPARED_DIR", None)
    if prepared_root:
        if os.path.basename(os.path.normpath(prepared_root)) == region_num:
            return prepared_root
        return os.path.join(prepared_root, region_num)
    return _default_prepared_traj_dir(region_num, cfg.DIR.DATA_ROOT)


def prepared_traj_cache_exists(prepared_dir):
    return (
        os.path.isfile(os.path.join(prepared_dir, "pixel_trajs.npz")) and
        os.path.isfile(os.path.join(prepared_dir, "grid_index.npz")) and
        os.path.isfile(os.path.join(prepared_dir, "meta.json"))
    )


def load_prepared_traj_cache(prepared_dir):
    pixel_npz = np.load(os.path.join(prepared_dir, "pixel_trajs.npz"), allow_pickle=False)
    points = pixel_npz["points"].astype(np.float32, copy=False)
    offsets = pixel_npz["offsets"].astype(np.int64, copy=False)
    all_pixel_trajectories = [
        points[offsets[i]:offsets[i + 1]]
        for i in range(len(offsets) - 1)
    ]

    grid_npz = np.load(os.path.join(prepared_dir, "grid_index.npz"), allow_pickle=False)
    cells = grid_npz["cells"].astype(np.int32, copy=False)
    cell_offsets = grid_npz["cell_offsets"].astype(np.int64, copy=False)
    traj_ids = grid_npz["traj_ids"].astype(np.int32, copy=False)
    grid_index = {
        (int(cells[i, 0]), int(cells[i, 1])): traj_ids[cell_offsets[i]:cell_offsets[i + 1]]
        for i in range(len(cells))
    }

    with open(os.path.join(prepared_dir, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    cell_size = int(meta["cell_size"])
    return all_pixel_trajectories, grid_index, cell_size


def load_region_trajectory_inputs(region_num, cfg):
    traj_source = cfg.TRAIN.get("TRAJ_SOURCE", "raw")
    prepared_dir = _prepared_traj_dir(region_num, cfg)
    if traj_source in ("prepared", "auto") and prepared_traj_cache_exists(prepared_dir):
        all_pixel_trajectories, grid_index, cell_size = load_prepared_traj_cache(prepared_dir)
        print(f"loaded prepared trajectories: {prepared_dir}, count: {len(all_pixel_trajectories)}")
        return None, all_pixel_trajectories, grid_index, cell_size

    if traj_source == "prepared":
        raise FileNotFoundError(
            f"prepared trajectory cache not found in {prepared_dir}; run scripts/prepare_xian_traj.py first"
        )

    traj_bbox = get_region_bbox_for_traj(region_num, data_root=cfg.DIR.DATA_ROOT)
    all_trajectories = get_all_traj_pieces_from_txt(
        os.path.join(cfg.DIR.DATA_ROOT, "input", "traj_piece", region_num),
        bbox=traj_bbox,
        txt_point_source=cfg.TRAIN.get("TRAJ_TXT_POINT_SOURCE", "raw"))
    all_pixel_trajectories = all_traj_to_all_pixel_traj(
        all_trajectories, region_num, data_root=cfg.DIR.DATA_ROOT)
    return all_trajectories, all_pixel_trajectories, None, None
