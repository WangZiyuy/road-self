import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import utils.model_utils as model_utils
from utils.tileloader import Tiles
from easydict import EasyDict
from utils.gis_to_graph import GisToGraphConverter
import time

class OSMDataset:

    def __init__(self, cfg, net=None, training=True, seg_input=None):
        self.cfg = cfg
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.window_size = cfg.TRAIN.WINDOW_SIZE
        self.input_channels = cfg.TRAIN.NUM_INPUT_CHANNELS
        self.input_traj_channels = cfg.TRAIN.NUM_INPUT_TRAJECTORY_CHANNELS
        self.seg_input = seg_input
        self.num_targets = cfg.TRAIN.NUM_TARGETS
        self.paths = []
        self.tiles = Tiles(training_regions=self.cfg.TRAIN.TRAINING_REGIONS,
                           parallel_tiles=self.cfg.TRAIN.PARALLEL_TILES,
                           region_path=cfg.DIR.ALL_REGION_PATH,
                           graph_dir=cfg.DIR.GRAPH_DIR,
                           tile_dir=cfg.DIR.TILE_DIR,
                           traj_dir=cfg.DIR.TRAJ_DIR, )
        self.save_idx = 0
        self.training = training
        self.net = net # 用于传递给path从而传递给model_utils文件中的轨迹过滤方法传递两个半径自定义参数

        self.subtiles = self.tiles.prepare_training()
        print("extracted {} subtiles from {} tiles (missing {})".format(
            len(self.subtiles), len(self.tiles.train_tiles), 4 * len(self.tiles.train_tiles) - len(self.subtiles)))

        print("loading initial paths")
        for i, subtile in enumerate(self.subtiles):
            print("In region:{}, in tile {}{}".format(subtile["region"], subtile['search_rect'].start, subtile['search_rect'].end))

            # 传入整个站点的轨迹数据（因为不想写切分2048的代码了。。。）
            self.all_trajectories = get_all_traj_pieces_from_txt(f'./data_self/input/traj_piece/{subtile["region"]}')
            self.all_pixel_trajectories = all_traj_to_all_pixel_traj(self.all_trajectories, subtile["region"])

            path = model_utils.Path(i, training, subtile["gc"].clone(), subtile, self.all_trajectories, self.all_pixel_trajectories, net=self.net)
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
                            tile_data=self.subtiles[path_idx])
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
        batch_traj_inputs = np.zeros((self.batch_size, self.input_traj_channels, self.window_size, self.window_size))
        batch_aerial_traj = np.zeros((self.batch_size, self.input_channels + self.input_traj_channels, self.window_size, self.window_size))
        batch_target_maps = np.zeros((self.batch_size, self.num_targets, self.window_size, self.window_size))
        batch_is_key_point = np.zeros(self.batch_size)
        batch_end_index = np.zeros(self.batch_size, dtype=np.int)
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
                        all_pixel_trajectories=self.all_pixel_trajectories, net=self.net)
                    path = self.paths[path_idx]
                    continue
                break

            fetch_list = ['aerial_image_chw',
                          'aerial_image_hwc',
                          'traj_image_chw',
                          'traj_image_hwc',
                          'walked_path_small',
                          'walked_path',
                          'road_seg_small',
                          'road_seg_thick3',
                          'junc_seg_small',
                          'junc_seg_thick3',
                          'valid_trajectories',]

            data_dict = path.make_path_input(extension_vertex=extension_vertex,
                                             fetch_list=fetch_list,
                                             traj_filter=self.cfg.TRAIN.TRAJ_FILTER,
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
            batch_traj_images_hwc.append(data_dict.traj_image_hwc)
            batch_extension_vertices.append(extension_vertex)

            # 输入确定
            batch_inputs[i] = data_dict.aerial_image_chw
            batch_traj_inputs[i] = data_dict.traj_image_chw
            batch_aerial_traj[i] = np.concatenate((batch_inputs[i], batch_traj_inputs[i]), axis=0)
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
            batch_valid_trajectory_inputs.append(data_dict.valid_trajectories)

        data = EasyDict({
            'path_indices': path_indices,
            'batch_extension_vertices': batch_extension_vertices,
            'batch_inputs': batch_inputs,
            'batch_traj_inputs': batch_traj_inputs,
            'batch_aerial_traj': batch_aerial_traj,
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
            'batch_traj_images_hwc': batch_traj_images_hwc,
            'batch_valid_trajectory_inputs': batch_valid_trajectory_inputs
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

# 新加，同于读取成条轨迹数据
def get_all_traj_pieces_from_txt(trajectory_dir):
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

def all_traj_to_all_pixel_traj(all_trajectories, region_num):
    """
    将csv文件中存储的实际经纬度坐标转换为像素坐标
    """
    all_pixel_trajectories = []

    for _, traj in enumerate(tqdm(all_trajectories, desc="trans all traj to all pixel traj")):
        converter = GisToGraphConverter(region_num, traj)
        pixel_trajectories = converter.convert_trajectories_to_pixels()
        all_pixel_trajectories.append(pixel_trajectories)
    all_pixel_trajectories = np.array(all_pixel_trajectories)

    return all_pixel_trajectories