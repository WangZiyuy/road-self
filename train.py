import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from easydict import EasyDict
import utils.model_utils as model_utils

import model.model as model
# from utils import crash_on_ipy
from utils.utils import AverageMeter, load_pretrained, numpy2tensor2cuda, get_logger, dilate_label_batch
from utils.OSMDataset import OSMDataset
from torch.utils.tensorboard import SummaryWriter
from model.losses import BCEDiceLoss, BCE_Loss
from configs.config import config
from utils.additional_methods import visualize_batch_data_grid, assert_finite
from utils.trajectory_mode import (
    TRAJ_MODE_NONE,
    prepare_trajectory_sequence_batch,
    resolve_trajectory_mode,
    validate_trajectory_model_compatibility,
)

import torch
import cv2

torch.set_num_threads(1)
torch.multiprocessing.set_start_method('spawn', force=True)

cv2.setNumThreads(0)   # 禁用 OpenCV 多线程
cv2.ocl.setUseOpenCL(False)


def epoch_to_learning_rate(epoch):
    if epoch <= 20:
        return 5e-4
    elif 20 < epoch <= 40:
        return 1e-4
    else:
        return 5e-5

def main():
    parser = argparse.ArgumentParser(description="VecRoad Pytorch Train")
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
    trajectory_mode = resolve_trajectory_mode(cfg)
    validate_trajectory_model_compatibility(cfg, trajectory_mode)
    use_trajectory = trajectory_mode != TRAJ_MODE_NONE

    #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.TRAIN.GPU_ID

    os.makedirs(cfg.DIR.DATA_ROOT, exist_ok=True)
    os.makedirs(cfg.DIR.LOG_DIR, exist_ok=True)
    os.makedirs(cfg.DIR.CHECK_POINT_DIR, exist_ok=True)
    os.makedirs(cfg.DIR.SHORTCUT_DIR, exist_ok=True)
    os.makedirs("visualization", exist_ok=True)

    logger = get_logger(logger_name="logtrain", log_dir=cfg.DIR.LOG_DIR)
    logger.info("trajectory mode: %s", trajectory_mode)
    summary_writer = SummaryWriter(log_dir=os.path.join(cfg.DIR.LOG_DIR))

    losses = AverageMeter()
    anchor_losses = AverageMeter()
    road_losses = AverageMeter()
    junc_losses = AverageMeter()
    time_meter = AverageMeter()

    # device = torch.device("cuda:%s" % cfg.TRAIN.GPU_ID if torch.cuda.is_available() else "cpu")
    net = model.RPNet(num_targets=cfg.TRAIN.NUM_TARGETS)

    if cfg.TRAIN.DATA_PARALLEL:
        print('Training in DataParallel')
        net = torch.nn.DataParallel(net, device_ids=list(range(torch.cuda.device_count())))
    net = net.cuda()
    # net = net.to(device)

    osm = OSMDataset(cfg, net=net)

    criteria = lambda a, b: F.binary_cross_entropy_with_logits(a, b, reduction='sum')
    # 将road loss修改为BCEDice
    criterion_road = BCEDiceLoss()

    # if cfg.TRAIN.SOLVER.METHOD == "Adam":
    #     optimizer = torch.optim.Adam(
    #         filter(lambda p: p.requires_grad, net.parameters()),
    #         lr=cfg.TRAIN.SOLVER.LEARNING_RATE,
    #         betas=(0.9, 0.99),
    #         weight_decay=cfg.TRAIN.SOLVER.WEIGHT_DECAY)

    if cfg.TRAIN.SOLVER.METHOD == "Adam":
        # 将网络参数和全局配置参数一起加入优化器
        optimizer = torch.optim.Adam(
            list(filter(lambda p: p.requires_grad, net.parameters())),
            lr=cfg.TRAIN.SOLVER.LEARNING_RATE,
            betas=(0.9, 0.99),
            weight_decay=cfg.TRAIN.SOLVER.WEIGHT_DECAY
        )


    if cfg.TRAIN.LOAD_CHECK_POINT:
        file_name = os.path.join(cfg.DIR.CHECK_POINT_DIR, cfg.TRAIN.CHECK_POINT_NAME)
        if os.path.isfile(file_name):
            net, optimizer = load_pretrained(net, file_name, optimizer, strict=True)

    start_epoch = 1 if cfg.TRAIN.START_EPOCH == 0 else cfg.TRAIN.START_EPOCH

    # 在定义模型和损失函数后，开启异常检测模式
    # torch.autograd.set_detect_anomaly(True)

    # 外循环，进行训练迭代的次数
    for outer_it in range(start_epoch, cfg.TRAIN.TOTAL_ITERATION + 1):
        # adjust learning rate
        current_lr = optimizer.param_groups[0]["lr"]
        expected_lr = epoch_to_learning_rate(outer_it)
        if current_lr != expected_lr:
            msg = "adjust learning rate: {}".format(expected_lr)
            logger.info(msg)
            for param_group in optimizer.param_groups:
                param_group["lr"] = epoch_to_learning_rate(outer_it)
        else:
            msg = "current learning rate: {}".format(current_lr)
            logger.info(msg)

        # if outer_it > 10:
        #    FOLLOW_MODE = "follow_output"

        net.train()
        # 内循环，进行路径循环，2048次，不是很理解
        for path_it in range(2048):

            stage_time = time.time()

            data_dict = osm.get_batch()
            #data_dict:
            # path_indices:                     20 elements         [78, 89, 10, 3, 53, 55, 49, 31, 24, 83, 33, 39, 58, 80, 59, 64, 54, 85, 13, 34],
            # batch_extension_vertices:         20 elements         [Vertex:{Point(7124, 5447), in:[290], out:[291]}, Vertex:{Point(5658, 7431), in:[462], out:[463]},...
            # batch_inputs:                     (20, 3 , 256, 256)   array([[[[0.09803922, 0.04705882, 0.06666667, ...,
            # batch_traj_inputs:                (20, 1 , 256, 256)
            # batch_target_maps:                (20, 4, 256, 256)   array([[[[0., 0., 0., ..., 0., 0., 0.],...,
            # batch_is_key_point:               (20,)               array([0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 1., 0.,0., 0., 1.]),
            # batch_end_index:                  (20,)               array([2, 1, 2, 4, 4, 2, 4, 4, 4, 4, 1, 4, 4, 4, 1, 1, 4, 4, 4, 1]),
            # batch_target_poses:               20 elements         [<utils.model_utils.TargetPosesContainer object at 0x7f2f766050d0>, ...
            # batch_walked_path_small:          (20, 1, 64, 64)     array([[[[[0., 0., 0., ..., 0., 0., 0.],...
            # batch_road_segmentation:          (20, 1, 64, 64)
            # batch_road_segmentation_thick3:   (20, 1, 256, 256)
            # batch_junction_segmentation:      (20, 1, 64, 64)
            # batch_aerial_images_hwc:          20 elements
            # batch_valid_trajectory_inputs_cuda:(2, 5, 11, 2) batch size 、轨迹条数、轨迹点数量、xy
            # 这里面的步长的20，首先是batch size是20；其次下次移动的固定步长是20

            os.makedirs(f"visualization/{outer_it}_{path_it}/", exist_ok=True)

            # batch内容可视化
            if use_trajectory and path_it % cfg.TRAIN.PRINT_ITERATION == 0 and data_dict.batch_valid_trajectory_inputs[0].size(0) > 1:
                visualize_batch_data_grid(
                    data_dict=data_dict,
                    batch_index=1,
                    fields=[
                        "batch_inputs",
                        "batch_traj_inputs",
                        "batch_road_segmentation",
                        "batch_road_segmentation_thick3",
                        "batch_junction_segmentation",
                        "batch_junction_segmentation_thick3",
                        "batch_aerial_images_hwc",
                        "batch_traj_images_hwc",
                        "batch_walked_path_small",
                        "batch_walked_path",
                        "batch_valid_trajectory_inputs",
                    ],
                    title="Batch Input Visualization",
                    save_path=f"visualization/{outer_it}_{path_it}/batch_input_{outer_it}_{path_it}.png"
                )

            # 遥感图像替换轨迹图像以及叠加图像
            batch_inputs_cuda = numpy2tensor2cuda(data_dict.batch_inputs)
            batch_walked_path_small_cuda = numpy2tensor2cuda(data_dict.batch_walked_path_small)
            batch_walked_path_cuda = numpy2tensor2cuda(data_dict.batch_walked_path)
            batch_target_maps_cuda = numpy2tensor2cuda(data_dict.batch_target_maps)

            batch_road_segmentation_dilated = dilate_label_batch(data_dict.batch_road_segmentation, kernel_size=3, iterations=1)
            batch_junction_segmentation_dilated = dilate_label_batch(data_dict.batch_junction_segmentation, kernel_size=3, iterations=1)
            batch_road_segmentation_cuda = numpy2tensor2cuda(batch_road_segmentation_dilated)
            batch_junction_segmentation_cuda = numpy2tensor2cuda(batch_junction_segmentation_dilated)

            batch_road_segmentation_thick3_dilated = dilate_label_batch(data_dict.batch_road_segmentation_thick3, kernel_size=3, iterations=1)
            batch_junction_segmentation_thick3_dilated = dilate_label_batch(data_dict.batch_junction_segmentation_thick3, kernel_size=3, iterations=1)
            batch_road_segmentation_thick3_cuda = numpy2tensor2cuda(batch_road_segmentation_thick3_dilated)
            batch_junction_segmentation_thick3_cuda = numpy2tensor2cuda(batch_junction_segmentation_thick3_dilated)

            batch_traj_inputs_cuda = None
            batch_aerial_traj_cuda = None
            if use_trajectory:
                batch_traj_inputs_cuda = numpy2tensor2cuda(data_dict.batch_traj_inputs)
                batch_aerial_traj_cuda = numpy2tensor2cuda(data_dict.batch_aerial_traj)
            batch_normalized_traj, batch_valid_mask = prepare_trajectory_sequence_batch(
                trajectory_mode,
                data_dict.batch_valid_trajectory_inputs if use_trajectory else None,
                model_utils.valid_trajectory_input_GPU,
                model_utils.normalize_trajectory_batch,
            )

            optimizer.zero_grad()

            with torch.autograd.set_detect_anomaly(True):
                """
                Net Processing
                """
                # 遥感图像 + 轨迹过滤模块
                batch_output_cuda_dict = net(
                    batch_inputs_cuda,
                    batch_traj_inputs_cuda,
                    batch_aerial_traj_cuda,
                    batch_normalized_traj,
                    batch_valid_mask,
                    batch_walked_path_cuda,
                    NUM_TARGETS=None,
                    test=False,
                    model=cfg.TRAIN.MODEL,
                    use_traj=use_trajectory)

                # # 为了比较轨迹的影响，也运行一次不使用轨迹的版本
                # if data_dict.batch_valid_trajectory_inputs[0].size(0) > 1:
                #     with torch.no_grad():
                #         batch_output_no_traj_dict = net(batch_inputs_cuda, batch_traj_inputs_cuda, batch_aerial_traj_cuda, batch_normalized_traj, batch_valid_mask, batch_walked_path_cuda, NUM_TARGETS=None, test=False, model=cfg.TRAIN.MODEL, use_traj=False)
                #
                #         # 比较有轨迹和无轨迹的特征图差异
                #         if 'feature_maps' in batch_output_cuda_dict and 'feature_maps' in batch_output_no_traj_dict:
                #             from utils.additional_methods import visualize_feature_comparison
                #             visualize_feature_comparison(
                #                 feature_maps_with_traj=batch_output_cuda_dict['feature_maps'],
                #                 feature_maps_without_traj=batch_output_no_traj_dict['feature_maps'],
                #                 batch_index=1,
                #                 save_path=f"visualization/{outer_it}_{path_it}/feature_comparison_{outer_it}_{path_it}.png",
                #                 title=f"Feature Comparison: With vs Without Trajectory (Epoch {outer_it}, Iter {path_it})"
                #             )

                # 网络的四个输出
                batch_output_road_cuda = batch_output_cuda_dict['road'] # 'road' torch.Size([20, 1, 64, 64])
                batch_output_junc_cuda = batch_output_cuda_dict['junc'] # 'junc' torch.Size([20, 1, 64, 64])
                batch_output_anchor_maps_cuda = batch_output_cuda_dict['anchor'] # 'anchor' torch.Size([20, 4, 256, 256])
                batch_output_anchor_step_maps_cuda = batch_output_cuda_dict['anchor_lowrs'] # 'anchor_lowrs' torch.Size([20, 4, 256, 256])
                batch_output_traj_road_cuda = batch_output_cuda_dict['traj_road'] # 'junc' torch.Size([20, 1, 64, 64])

                # 输出结果内容可视化
                if use_trajectory and path_it % cfg.TRAIN.PRINT_ITERATION == 0 and data_dict.batch_valid_trajectory_inputs[0].size(0) > 1:
                    # 可视化网络输出
                    visualize_batch_data_grid(
                        data_dict=batch_output_cuda_dict,
                        batch_index=1,
                        fields=[
                            "road",
                            "junc",
                            "anchor",
                            "anchor_lowrs",
                            "traj_road",
                        ],
                        title="Network Output Visualization",
                        save_path=f"visualization/{outer_it}_{path_it}/network_output_{outer_it}_{path_it}.png"
                    )

                    # 可视化中间特征图
                    if 'feature_maps' in batch_output_cuda_dict:
                        from utils.additional_methods import visualize_feature_maps
                        visualize_feature_maps(
                            feature_maps=batch_output_cuda_dict['feature_maps'],
                            batch_index=1,
                            save_path=f"visualization/{outer_it}_{path_it}/feature_maps_{outer_it}_{path_it}.png",
                            title=f"Feature Maps Visualization (Epoch {outer_it}, Iter {path_it})"
                        )

                """
                Loss Calculation
                """
                anchor_loss = 0
                for i in range(cfg.TRAIN.BATCH_SIZE):
                    inp = batch_output_anchor_maps_cuda[i, :data_dict.batch_end_index[i], :, :]
                    target = batch_target_maps_cuda[i, :data_dict.batch_end_index[i], :, :]
                    anchor_loss += criteria(inp, target).cuda()

                anchor_mid_loss = 0
                for i in range(cfg.TRAIN.BATCH_SIZE):
                    inp = batch_output_anchor_step_maps_cuda[i, :data_dict.batch_end_index[i], :, :]
                    target = batch_target_maps_cuda[i, :data_dict.batch_end_index[i], :, :]
                    anchor_mid_loss += criteria(inp, target).cuda()

                anchor_loss += anchor_mid_loss

                # road_loss = junc_loss = 0
                # for item in batch_output_road_cuda:
                road_loss = criteria(batch_output_road_cuda, batch_road_segmentation_thick3_cuda).cuda()
                # road_loss = criterion_road(batch_road_segmentation_thick3_cuda, batch_output_road_cuda).cuda()
                # for item in batch_output_junc_cuda:
                junc_loss = criteria(batch_output_junc_cuda, batch_junction_segmentation_thick3_cuda).cuda()
                # junc_loss = criterion_road(batch_junction_segmentation_thick3_cuda, batch_output_junc_cuda).cuda()

                # traj_road_loss  = criteria(batch_output_traj_road_cuda, batch_road_segmentation_cuda).cuda()

                # 确保 outer_it(外层循环此处应该是50次) 和 path_it（内层循环此处应该是2048） 在整个训练过程中唯一标识每一步
                loss = anchor_loss + 10 * road_loss + 10 * junc_loss

                if path_it % cfg.TRAIN.PRINT_ITERATION == 0:
                    # Log anchor loss to TensorBoard
                    summary_writer.add_scalar('anchor_loss', anchor_loss, outer_it * 2048 + path_it)
                    summary_writer.add_scalar('anchor_mid_loss', anchor_mid_loss, outer_it * 2048 + path_it)
                    summary_writer.add_scalar('road_loss', road_loss, outer_it * 2048 + path_it)
                    summary_writer.add_scalar('junc_loss', junc_loss, outer_it * 2048 + path_it)
                    summary_writer.add_scalar('total_loss', loss, outer_it * 2048 + path_it)

                # optimizer.zero_grad()
                loss.backward()

            torch.nn.utils.clip_grad_value_(net.parameters(), 1e4)
            # TODO ds说梯度裁剪过大 建议torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
            optimizer.step()

            """
            Data post-processing
            """
            data_dict.batch_output_road = torch.sigmoid(batch_output_road_cuda).detach().cpu().numpy()
            data_dict.batch_output_junc = torch.sigmoid(batch_output_junc_cuda).detach().cpu().numpy()
            data_dict.batch_output_anchor_maps = torch.sigmoid(batch_output_anchor_maps_cuda).detach().cpu().numpy()

            losses.update(loss.data.item())
            anchor_losses.update(anchor_loss.data.item())
            road_losses.update(road_loss.data.item())
            junc_losses.update(junc_loss.data.item())

            time_meter.update(time.time() - stage_time)

            if path_it % cfg.TRAIN.PRINT_ITERATION == 0:
                msg = "iter:[{0}]-[{1}/2048] " \
                      "Time: {time_meter.val:.3f} ({time_meter.avg:.3f}) " \
                      "Anchor: {anchor_loss.val:.3f} ({anchor_loss.avg:.3f}) " \
                      "Road: {road_loss_val:.3f} ({road_loss_avg:.3f}) " \
                      "Junc: {junc_loss_val:.3f} ({junc_loss_avg:.3f}) " \
                      "Total: {total_loss.val:.3f} ({total_loss.avg:.3f})" \
                    .format(
                    outer_it, path_it,
                    time_meter=time_meter,
                    anchor_loss=anchor_losses,
                    road_loss_val=road_losses.val * 10,
                    road_loss_avg=road_losses.avg * 10,
                    junc_loss_val=junc_losses.val * 10,
                    junc_loss_avg=junc_losses.avg * 10,
                    total_loss=losses
                )
                logger.info(msg)

            osm.push_and_vis_batch(data_dict, outer_it, path_it)

            # 所以在这里进行重置损失是为了存储每次迭代的结果（大循环 50那个）
            if (path_it + 1) % cfg.TRAIN.SAVE_ITERATIONS == 0:
                msg = "iter:[{0}]-[{1}/2048] " \
                        "Time: {time_meter.sum:.3f} " \
                        "Anchor: {anchor_loss.avg:.3f} " \
                        "Road: {road_loss:.3f} " \
                        "Junc: {junc_loss:.3f} " \
                        "Total: {total_loss.avg:.3f}" \
                    .format(outer_it, path_it, time_meter=time_meter, anchor_loss=anchor_losses,
                            road_loss=road_losses.avg * 10, junc_loss=junc_losses.avg * 10, total_loss=losses)
                logger.info(msg)

                time_meter.reset()
                losses.reset()
                anchor_losses.reset()
                road_losses.reset()
                junc_losses.reset()
                if outer_it >= 10 and outer_it % 10 == 0:
                    train_name = '新有轨迹_anchor*0.1ALLbce_256_thick5'
                    os.makedirs(os.path.join(cfg.DIR.CHECK_POINT_DIR, train_name), exist_ok=True)
                    save_file = os.path.join(cfg.DIR.CHECK_POINT_DIR, train_name, "{}.{}.pth.tar".format(outer_it, path_it))
                    torch.save({
                        "outer_it": outer_it,
                        "path_it": path_it,
                        "state_dict": net.state_dict(),
                        "optimizer": optimizer.state_dict()
                    }, save_file)

            print('Time taken for ne ITERATIONS = {} sec \n'.format(time.time() - stage_time))

    summary_writer.close()


if __name__ == '__main__':
    main()
