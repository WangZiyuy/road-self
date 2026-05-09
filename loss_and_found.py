# -*- coding: UTF-8 -*-
'''
@Project ：VecRoad 
@File    ：loss_and_found.py
@IDE     ：PyCharm 
@Author  ：wzy
@Date    ：2025/4/2 23:49:34
'''

# 获取输出和注意力权重
# output, attention_weights = model(trajectory_input_tensor)
#
# # 可视化注意力权重
# attention_weights = attention_weights[0]  # 选择第一层的注意力权重
# sns.heatmap(attention_weights[0].detach().numpy(), cmap="Blues",
#             xticklabels=range(len(trajectory_input_tensor[0])),
#             yticklabels=range(len(trajectory_input_tensor[0])))
# plt.xlabel('Trajectory Points')
# plt.ylabel('Trajectory Points')
# plt.title('Attention Weights for Trajectory')
# plt.show()

# 在此处加入处理轨迹图的过滤模块(代替原来的conv_fuse）
# stage_fuse = self.directional_filtering(stage_fuse)

# elif model == 'DSFNet':
# # TODO 这里也改成Unet的解码器结构 应该更合理
# decoded_ft_4 = self.decoders_DFS[0](upsample(stage_4_DFS, 2), next_step)
# decoded_ft_3 = self.decoders_DFS[1](upsample(stage_3_DFS, 1), decoded_ft_4)
# decoded_ft_2 = self.decoders_DFS[2](upsample(stage_2_DFS, 1), upsample(decoded_ft_3, 2))
# decoded_ft_1 = self.decoders_DFS[3](upsample(stage_1_DFS, 1), upsample(decoded_ft_2, 2))


# 每次更新后，限制这两个参数的范围
# config.circle_radius.data = torch.clamp(config.circle_radius, min=4.0, max=40)
# config.neighborhood_radius.data = torch.clamp(config.neighborhood_radius, min=10.0, max=100)
# with torch.no_grad():
#     # TODO 到底要不要no grad
#     net.circle_radius.data.clamp_(min=4.0, max=40)
#     net.neighborhood_radius.data.clamp_(min=10.0, max=100)


# 特征为64，64的backbone
# stage_1 = self.stage_1(aerial_image)  # 1/2
# stage_1_down = self.maxpool(stage_1)     #1/4
#
# stage_2 = self.stage_2(stage_1_down)     #1/4
# stage_2_side = self.conv_2_side(stage_2) #1/4
#
# stage_3 = self.stage_3(stage_2)          #1/8
# stage_3_side = self.conv_3_side(stage_3) #1/8
# stage_3_side = upsample(stage_3_side, 2) #1/4
#
# stage_4 = self.stage_4(stage_3)          #1/8
# stage_4_side = self.conv_4_side(stage_4) #1/8
# stage_4_side = upsample(stage_4_side, 2) #1/4
#
# stage_5 = self.stage_5(stage_4)          #1/8
# stage_5_side = self.conv_5_side(stage_5) #1/8
# stage_5_side = upsample(stage_5_side, 2) #1/4
# side都为torch.Size([5, 128, 64, 64])

# for name, param in net.named_parameters():
#     print(name, param.size())

# if path_it % cfg.TRAIN.PRINT_ITERATION == 0:
#     msg = "iter:[{0}]-[{1}/2048] " \
#           "Time: {time_meter.val:.3f} ({time_meter.avg:.3f}) " \
#           "Anchor: {anchor_loss.val:.3f} ({anchor_loss.avg:.3f}) " \
#           "Road: {road_loss.val:.3f} ({road_loss.avg:.3f}) " \
#           "Junc: {junc_loss.val:.3f} ({junc_loss.avg:.3f}) " \
#           "Total: {total_loss.val:.3f} ({total_loss.avg:.3f})" \
#         .format(outer_it, path_it, time_meter=time_meter, anchor_loss=anchor_losses,
#                 road_loss=road_losses, junc_loss=junc_losses, total_loss=losses)
#     logger.info(msg)

# distances[~valid_window_mask.view(-1, 1, 1).expand(-1, win_len, num_circles)] = float('inf')
        # 激活函数：当距离小于 circle_radius 时激活接近1，否则趋近0；乘以10使得变化更明显
        # activation = torch.sigmoid((circle_radius - distances) * 100)  # [B*S, win_len, 8]

#
# # 遍历所有rect范围内的轨迹数据
# for trajectory in rect_trajectory_segments:
#     # 遍历每个圆圈，检查轨迹是否通过
#     for i, circle in enumerate(self.circles):
#         circle_center = circle['center']
#         circle_radius = circle['radius']
#         last_processed_idx = -1  # 记录上一次处理的最后一个轨迹点索引
#         # 判断轨迹中的每个点是否在圆圈内
#         passed_circle = False
#         for idx, point in enumerate(trajectory):
#             if idx <= last_processed_idx:
#                 continue
#             trajectory_segment = []  # 用于保存有效轨迹片段
#             # point的真实经纬度： latitude', 'longitude'
#             # point的像素经纬度：y, x
#             # circles的像素经纬度同origin：x, y
#             # point = (point[1] - origin.x, point[0] - origin.y)
#             distance = np.sqrt((point[0] - circle_center.x) ** 2 + (point[1] - circle_center.y) ** 2)
#
#             if distance <= circle_radius:
#                 passed_circle = True
#                 trajectory_passes[i] += 1
#                 # 保存该轨迹点以及前后 window_size 个轨迹点
#                 start_idx = max(0, idx - window_size)
#                 end_idx = min(len(trajectory), idx + window_size + 1)
#                 trajectory_segment.extend(trajectory[start_idx:end_idx])  # 将这些轨迹点加入到片段中
#
#                 # 更新 last_processed_idx
#                 last_processed_idx = end_idx - 1
#
#                 # 如果轨迹通过了当前圆圈，保存轨迹片段
#                 if passed_circle and len(trajectory_segment) > 0:
#                     circle_trajectories[i].append(trajectory_segment)
#
# # 找到穿过最多轨迹的圆圈
# max_passed_circle_idx = np.argmax(trajectory_passes)
# self.valid_trajectories = circle_trajectories[max_passed_circle_idx]
# # print('Time taken for traj in pieces = {} sec \n'.format(time.time() - start))


# circle_trajectories = {i: [] for i in range(num_circles)}  # 记录每个圆圈穿过的轨迹
# trajectory_passes = np.zeros(num_circles)  # 用于记录轨迹穿过每个圆圈的次数
#
# # 使用 R-tree 查找在 rect 范围内的轨迹点
# rect_candidate_points = list(self.trajectories_rtree.intersection((rect.start.x, rect.start.y, rect.end.x, rect.end.y)))
# # 用于恢复rect中属于同一条轨迹片段的轨迹点
# recover_points = {}
# # 根据查找到的点提取相关轨迹
# for id in rect_candidate_points:
#     traj_idx, point_idx = self.trajectories_map[id]
#     if traj_idx not in recover_points:
#         recover_points[traj_idx] = []
#     recover_points[traj_idx].append(point_idx)
# # 将轨迹片段按顺序整理
# rect_trajectory_segments = []
# for traj_idx, point_indices in recover_points.items():
#     # 获取原始轨迹
#     # TODO 这里到底还需不需要进行排序什么的(需要排序，但不知道顺序对不对
#     trajectory = self.all_pixel_trajectories[traj_idx]
#     # 按轨迹点索引排序，确保顺序
#     point_indices.sort()
#     # 提取轨迹片段
#     segment = [trajectory[idx] for idx in point_indices]
#     rect_trajectory_segments.append(segment)


# 遥感图像 + 轨迹图像
# （遥感图像 + 轨迹图像） + 轨迹过滤模块
# for name, param in net.named_parameters():
#     if torch.isnan(param).any() or torch.isinf(param).any():
#         print(name)

# for name, param in net.named_parameters():
#     if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
#         print(name)


# # 方案2：对于每个窗口，选择权重最大的圆，然后聚合属于该圆的窗口
# # 先计算每个窗口的最大权重索引和对应的权重
# max_weights, max_indices = torch.max(circle_weights, dim=1)  # [B*S]，max_indices: 对应圆的索引
#
# # 设置一个较低的权重阈值，只有当窗口的最大权重超过该值时，才认为该窗口具有可信的方向信息
# threshold_weight = 0.25  # 可根据实验调整
#
# circle_trajectories = []
# for i in range(circle_centers.size(0)):  # 对每个圆
#     # 选择窗口：要求窗口的最大权重指向当前圆，同时其最大权重必须大于阈值
#     sel_mask = (max_indices == i) & (max_weights > threshold_weight)
#     selected = all_segments_flat[sel_mask]  # [n_i, win_len, 2]
#     circle_trajectories.append(selected)
#
# # 频次统计：统计每个圆被选中的有效窗口数量
# pass_counts = torch.stack([(max_indices == i).sum() for i in range(circle_centers.size(0))])
# max_idx = torch.argmax(pass_counts).item()


# # 对每个圆心的轨迹都进行可视化
# import matplotlib.pyplot as plt
# # 创建新的 figure 和 axes
# # 创建一个2列子图 (左：线图；右：点图)
# fig, axes = plt.subplots(1, 2, figsize=(12, 6))
# ax_line, ax_scatter = axes
# colors = plt.cm.get_cmap("tab10", circle_centers.size(0))  # 每个圆分配不同颜色
# # 对每个圆心的轨迹都进行可视化
# for i in range(num_circles):  # 遍历所有圆心
#     # 获取当前圆心的轨迹
#     circle_trajectories[i] -= torch.tensor([rect.start.x, rect.start.y], device=circle_trajectories[max_idx].device)
#     # 将Y轴翻转
#     # circle_trajectories[i][:, :, 1] = 256 - circle_trajectories[i][:, :, 1]
#     traj_data = circle_trajectories[i]  # 当前圆心的所有轨迹
#     color = colors(i)
#     traj_np = traj_data.cpu().numpy()  # 转换为 numpy 格式
#     for traj in traj_np:
#         ax_line.plot(traj[:, 1], traj[:, 0], color=color, linewidth=1)  # lat_lon[:, 0] -> 纬度, lat_lon[:, 1] -> 经度
#
# # 设置标题和坐标轴
# ax_line.set_title("All Trajectories around 8 Circle Centers")
# ax_line.set_xlim(0, 256)
# ax_line.set_ylim(0, 256)
# ax_line.axis("off")
# ax_line.set_aspect('equal')  # 确保坐标轴比例一致
# ax_line.axis("off")
#
# for i in range(num_circles):  # 遍历所有圆心
#     # 获取当前圆心的轨迹
#     circle_trajectories[i] -= torch.tensor([rect.start.x, rect.start.y],
#                                            device=circle_trajectories[max_idx].device)
#     traj_data = circle_trajectories[i]  # 当前圆心的所有轨迹
#     color = colors(i)
#     all_points = traj_data.view(-1, 2).cpu().numpy()  # 把所有轨迹点拼在一起
#     ax_scatter.scatter(all_points[:, 1], all_points[:, 0], s=5, color=color, alpha=0.7, label=f"circle {i}")
#
# # 设置标题和坐标轴
# ax_scatter.set_title("All Trajectories points around 8 Circle Centers")
# ax_scatter.set_xlim(0, 256)
# ax_scatter.set_ylim(0, 256)
# ax_scatter.set_aspect('equal')  # 确保坐标轴比例一致
# ax_scatter.axis("off")
#
# # 保存图像到文件
# plt.tight_layout()
# plt.savefig("2all_trajectories_vis.png", bbox_inches="tight", pad_inches=0.1)
# ax_line.invert_yaxis()
# ax_scatter.invert_yaxis()
# plt.savefig("2all_trajectories_vis_y.png", bbox_inches="tight", pad_inches=0.1)
# ax_line.invert_xaxis()
# ax_line.invert_yaxis()
# ax_scatter.invert_yaxis()
# plt.savefig("2all_trajectories_vis_yx.png", bbox_inches="tight", pad_inches=0.1)