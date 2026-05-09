from PIL import Image, ImageOps
import os
import re


def assert_finite(name, x):
    if not torch.isfinite(x).all():
        bad = (~torch.isfinite(x)).nonzero(as_tuple=False)
        raise RuntimeError(f"{name} contains non-finite values at {bad[:5]} | "
                           f"min={x.nanmin().item() if x.numel() else 'n/a'} "
                           f"max={x.nanmax().item() if x.numel() else 'n/a'}")


# 从文件名中解析坐标
def parse_coordinates(file_name):
    match = re.search(r'columbus_(-?\d+)_(-?\d+)', file_name)
    if match:
        return int(match.group(1)), int(match.group(2))
    else:
        raise ValueError(f"Filename {file_name} does not contain valid coordinates.")


# 拼接图像函数
def concat_images(image_files, border_size=15, border_color=(255, 255, 255)):
    """
    拼接图像
    :param image_files: 图像文件列表，每个图像对应文件路径
    :param rows: 行数
    :param cols: 列数
    :param border_size: 边界的像素宽度
    :param border_color: 边界颜色，默认为黑色
    :return: 拼接后的大图像
    """
    # 打开所有图像并记录对应坐标
    images = {}
    for img_file in image_files:
        img = Image.open(img_file)
        x, y = parse_coordinates(os.path.basename(img_file))
        images[(x, y)] = img

    # 获取单个图像的大小，假设所有图像的大小相同
    width, height = next(iter(images.values())).size

    # 计算整张图像的尺寸
    min_x = min(x for x, y in images.keys())
    max_x = max(x for x, y in images.keys())
    min_y = min(y for x, y in images.keys())
    max_y = max(y for x, y in images.keys())

    cols = max_x - min_x + 1
    rows = max_y - min_y + 1

    # 创建一个新的空白图像，大小为(cols * width + 边界, rows * height + 边界)
    new_width = cols * (width + border_size) - border_size
    new_height = rows * (height + border_size) - border_size
    combined_image = Image.new('RGB', (new_width, new_height), color=border_color)

    # 将每个图像按照其坐标放入大图中，并且添加边界
    for (x, y), img in images.items():
        img_with_border = ImageOps.expand(img, border=border_size, fill=border_color)
        col = x - min_x  # 转换为从0开始的索引
        row = y - min_y
        combined_image.paste(img_with_border, (col * (width + border_size), row * (height + border_size)))
        print(col * (width + border_size), row * (height + border_size))
    return combined_image


def get_file_list(directory):
    file_list = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if filepath[-3:] == 'png':
            file_list.append(filepath)
    return file_list


# # 示例使用
# image_dir = './columbus/train/'
# image_files = get_file_list(image_dir)
#
# combined_image = concat_images(image_files)
#
# # 保存拼接后的图像
# combined_image.save('combined_image.png')
# # combined_image.show()


import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免GUI相关错误
import matplotlib.pyplot as plt
import numpy as np

# def visualize_batch_data_grid(data_dict, batch_index=0, fields=None, save_path=None, title=None, max_per_row=4):
#     """
#     可视化 EasyDict 中多个 batch 字段的图像内容。
#
#     参数：
#         data_dict (dict or EasyDict): 包含多个图像字段的 dict
#         batch_index (int): 可视化第几个 batch（默认取第0个）
#         fields (list): 指定展示的字段名（默认自动选择常见图像字段）
#         save_path (str): 是否保存为文件
#         title (str): 整体图标题
#         max_per_row (int): 每行显示多少张图
#     """
#
#     num_fields = len(fields)
#     n_cols = min(num_fields, max_per_row)
#     n_rows = int(np.ceil(num_fields / n_cols))
#
#     fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
#
#     # 如果只有1行1列，axes 不是数组，需要转成列表处理
#     if num_fields == 1:
#         axes = [axes]
#     else:
#         axes = axes.flatten()
#
#     for i, field in enumerate(fields):
#         ax = axes[i]
#         value = data_dict[field]
#
#         try:
#             if isinstance(value, torch.Tensor):
#                 value = value.detach().cpu().numpy()
#             img = value[batch_index]
#             print("field value:", field, img.min().item(), img.max().item())
#         except:
#             ax.set_title(f"{field}\n(no data)")
#             ax.axis("off")
#             continue
#
#         # 转换维度或处理
#         if isinstance(img, np.ndarray):
#             if img.ndim == 3 and img.shape[0] in [1, 3]:  # C, H, W -> H, W, C
#                 img = img.transpose(1, 2, 0)
#             elif img.ndim == 2:
#                 pass  # 灰度图
#             elif img.ndim == 3 and img.shape[2] in [1, 3]:
#                 pass  # HWC 格式
#             else:
#                 img = np.mean(img, axis=0)  # 降维
#
#             # 归一化到 [0,1] 范围便于可视化
#             if img.max() > 1.0:
#                 img = img / 255.0
#                 ax.set_title(f"{field}\n(no norm)")
#
#             ax.imshow(img, cmap='gray' if img.ndim == 2 else None)
#             ax.set_title(field)
#             ax.axis("off")
#         else:
#             ax.set_title(f"{field}\n(invalid)")
#             ax.axis("off")
#
#     # 清空多余 subplot
#     for j in range(i + 1, len(axes)):
#         axes[j].axis("off")
#
#     if title:
#         plt.suptitle(title, fontsize=16)
#
#     plt.tight_layout()
#     if save_path:
#         plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
#         print(f"Saved to {save_path}")
#     else:
#         plt.show()
#
#     plt.close()


def visualize_batch_data_grid(data_dict, batch_index=0, fields=None, save_path=None, title=None, max_per_row=4):
    """
    可视化 EasyDict 中多个 batch 字段的图像内容。
    对 raw logits 的分割结果自动加 sigmoid。
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    if fields is None:
        fields = list(data_dict.keys())

    sigmoid_fields = ["road", "junc", "anchor", "anchor_lowrs", "traj_road"]

    num_fields = len(fields)
    n_cols = min(num_fields, max_per_row)
    n_rows = int(np.ceil(num_fields / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))

    # 如果只有1行1列，axes 不是数组，需要转成列表处理
    if num_fields == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, field in enumerate(fields):
        ax = axes[i]
        value = data_dict[field]

        # 处理轨迹数据（直接在字段循环中）
        if field == "batch_valid_trajectory_inputs":
            traj_cluster = value[batch_index]  # 直接从 value 获取轨迹数据

            for traj in traj_cluster:  # 绘制每条轨迹
                lat_lon = traj.cpu().numpy()  # 轨迹的经纬度数据
                # ax.plot(lat_lon[:, 1], lat_lon[:, 0], color='blue', linewidth=1)  # lat_lon[:, 0] -> 纬度, lat_lon[:, 1] -> 经度
                ax.scatter(lat_lon[:, 1], lat_lon[:, 0], color='red', s=5)

            import matplotlib.cm as cm
            # colors = cm.rainbow(np.linspace(0, 1, len(traj_cluster)))
            # for traj, c in zip(traj_cluster, colors):
            #     lat_lon = traj.detach().cpu().numpy()
            #     ax.plot(lat_lon[:, 0], lat_lon[:, 1], color=c, linewidth=1)

            ax.set_xlim(0, 256)
            ax.set_ylim(256, 0)
            ax.set_title("batch_valid_trajectory_inputs")
            ax.axis("off")
            continue  # 跳过后续处理，直接跳到下一字段

        try:
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu()
                if field in sigmoid_fields:
                    value = torch.sigmoid(value)
                value = value.numpy()

            img = value[batch_index]
            # print("field value:", field, img.min().item(), img.max().item())
        except:
            ax.set_title(f"{field}\n(no data)")
            ax.axis("off")
            continue

        # 转换维度或处理
        if isinstance(img, np.ndarray):
            if img.ndim == 3 and img.shape[0] in [1, 3]:  # C, H, W -> H, W, C
                img = img.transpose(1, 2, 0)
            elif img.ndim == 2:
                pass  # 灰度图
            elif img.ndim == 3 and img.shape[2] in [1, 3]:
                pass  # HWC 格式
            else:
                img = np.mean(img, axis=0)  # 降维

            # 归一化到 [0,1] 范围便于可视化
            if img.max() > 1.0:
                img = img / 255.0
                ax.set_title(f"{field}\n(no norm)")

            ax.imshow(img, cmap='gray' if img.ndim == 2 else None)
            ax.set_title(field)
            ax.axis("off")
        else:
            ax.set_title(f"{field}\n(invalid)")
            ax.axis("off")

    # 清空多余 subplot
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    if title:
        plt.suptitle(title, fontsize=16)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        print(f"Saved to {save_path}")
    else:
        plt.show()

    plt.close()


import torch


def analyze_checkpoint(path):
    print(f"分析模型权重文件：{path}")
    checkpoint = torch.load(path, map_location='cpu')

    # 查看整体结构
    print("\n=== checkpoint keys ===")
    for key in checkpoint.keys():
        print(f"  {key}")

    # 重点分析 state_dict
    if 'state_dict' in checkpoint:
        print("\n=== state_dict keys (共 {} 项) ===".format(len(checkpoint['state_dict'])))
        all_keys = list(checkpoint['state_dict'].keys())

        seg_keys = [k for k in all_keys if 'road_seg' in k or 'conv_road_final' in k or 'junc_seg' in k]
        anchor_keys = [k for k in all_keys if 'anchor' in k]

        print(f"\n✅ 包含分割模块的参数（共 {len(seg_keys)} 项）：")
        for k in seg_keys:
            print(f"  {k}")

        print(f"\n📦 包含 anchor 模块参数（共 {len(anchor_keys)} 项）：")
        for k in anchor_keys:
            print(f"  {k}")


def visualize_feature_maps(feature_maps, batch_index=0, save_path=None, title=None, max_per_row=4):
    """
    可视化模型中间特征图

    参数：
        feature_maps (dict): 包含特征图的字典
        batch_index (int): 可视化第几个batch
        save_path (str): 保存路径
        title (str): 图像标题
        max_per_row (int): 每行最大显示数量
    """
    import numpy as np
    import torch

    # 过滤出有效的特征图
    valid_features = {}
    for name, tensor in feature_maps.items():
        if isinstance(tensor, torch.Tensor) and tensor.dim() == 4:  # 确保是4D张量 [B, C, H, W]
            valid_features[name] = tensor

    if not valid_features:
        print("没有找到有效的特征图")
        return

    # 选择要显示的特征图（避免太多）
    feature_names = list(valid_features.keys())
    if len(feature_names) > 20:  # 如果特征图太多，只显示前20个
        feature_names = feature_names[:20]
        print(f"特征图数量过多，只显示前20个: {feature_names}")

    num_features = len(feature_names)
    n_cols = min(num_features, max_per_row)
    n_rows = int(np.ceil(num_features / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))

    # 如果只有1行1列，axes 不是数组，需要转成列表处理
    if num_features == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, feature_name in enumerate(feature_names):
        ax = axes[i]
        tensor = valid_features[feature_name]

        try:
            # 获取指定batch的特征图
            feature_map = tensor[batch_index].detach().cpu().numpy()

            # 如果是多通道，取平均值或选择前几个通道
            if feature_map.shape[0] > 3:
                # 多通道特征图，取平均值
                feature_map = np.mean(feature_map, axis=0)
                cmap = 'viridis'  # 使用viridis颜色映射
            elif feature_map.shape[0] == 3:
                # 3通道，直接显示
                feature_map = feature_map.transpose(1, 2, 0)
                cmap = None
            elif feature_map.shape[0] == 1:
                # 单通道
                feature_map = feature_map[0]
                cmap = 'gray'
            else:
                # 其他情况，取平均值
                feature_map = np.mean(feature_map, axis=0)
                cmap = 'viridis'

            # 归一化到[0,1]范围
            if feature_map.max() > feature_map.min():
                feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min())

            # 显示特征图
            im = ax.imshow(feature_map, cmap=cmap)
            ax.set_title(f"{feature_name}\n{tensor.shape[1]}C,{tensor.shape[2]}x{tensor.shape[3]}", fontsize=8)
            ax.axis('off')

            # 添加颜色条（对于多通道特征图）
            if feature_map.ndim == 2 and cmap != 'gray':
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        except Exception as e:
            ax.set_title(f"{feature_name}\n(error: {str(e)})", fontsize=8)
            ax.axis('off')

    # 清空多余的subplot
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    if title:
        plt.suptitle(title, fontsize=16)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=150)
        print(f"特征图已保存到: {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_feature_comparison(feature_maps_with_traj, feature_maps_without_traj, batch_index=0, save_path=None, title=None, topk_channels=0, joint_normalize=True):
    """
    比较有轨迹和无轨迹时的特征图差异

    参数：
        feature_maps_with_traj (dict): 有轨迹时的特征图
        feature_maps_without_traj (dict): 无轨迹时的特征图
        batch_index (int): 可视化第几个batch
        save_path (str): 保存路径
        title (str): 图像标题
    """
    import numpy as np
    import torch

    # 找到共同的特征图
    common_features = set(feature_maps_with_traj.keys()) & set(feature_maps_without_traj.keys())
    common_features = [f for f in common_features if isinstance(feature_maps_with_traj[f], torch.Tensor) and
                      isinstance(feature_maps_without_traj[f], torch.Tensor)]

    if not common_features:
        print("没有找到共同的特征图")
        return

    # 选择要比较的特征图（优先展示融合预览）
    key_features = ['stage_fuse_after', 'valid_traj_feature', 'stage_fuse', 'road_fts', 'junc_fts']
    selected_features = [f for f in key_features if f in common_features]
    if len(selected_features) < len(key_features):
        # 如果关键特征不完整，添加其他特征
        remaining = [f for f in common_features if f not in selected_features]
        selected_features.extend(remaining[:4])  # 最多添加4个其他特征

    num_features = len(selected_features)
    fig, axes = plt.subplots(num_features, 3, figsize=(12, 4 * num_features))

    if num_features == 1:
        axes = axes.reshape(1, -1)

    for i, feature_name in enumerate(selected_features):
        with_traj_t = feature_maps_with_traj[feature_name][batch_index].detach().cpu()
        without_traj_t = feature_maps_without_traj[feature_name][batch_index].detach().cpu()

        # 可选：选取方差最大的前K个通道，避免通道均值稀释差异
        if topk_channels and with_traj_t.dim() == 3:
            c = with_traj_t.shape[0]
            k = min(topk_channels, c)
            var_scores = with_traj_t.view(c, -1).var(dim=1)
            top_idx = torch.topk(var_scores, k=k).indices
            with_traj_t = with_traj_t[top_idx]
            without_traj_t = without_traj_t[top_idx]

        with_traj = with_traj_t.numpy()
        without_traj = without_traj_t.numpy()

        # 处理多通道特征图
        if with_traj.shape[0] > 3:
            with_traj = np.mean(with_traj, axis=0)
            without_traj = np.mean(without_traj, axis=0)
            cmap = 'viridis'
        elif with_traj.shape[0] == 3:
            with_traj = with_traj.transpose(1, 2, 0)
            without_traj = without_traj.transpose(1, 2, 0)
            cmap = None
        elif with_traj.shape[0] == 1:
            with_traj = with_traj[0]
            without_traj = without_traj[0]
            cmap = 'gray'
        else:
            with_traj = np.mean(with_traj, axis=0)
            without_traj = np.mean(without_traj, axis=0)
            cmap = 'viridis'

        # 归一化：联合归一可避免把差异洗掉
        if joint_normalize:
            mn = min(with_traj.min(), without_traj.min())
            mx = max(with_traj.max(), without_traj.max())
            if mx > mn:
                with_traj = (with_traj - mn) / (mx - mn)
                without_traj = (without_traj - mn) / (mx - mn)
        else:
            if with_traj.max() > with_traj.min():
                with_traj = (with_traj - with_traj.min()) / (with_traj.max() - with_traj.min())
            if without_traj.max() > without_traj.min():
                without_traj = (without_traj - without_traj.min()) / (without_traj.max() - without_traj.min())

        # 计算差异
        diff = with_traj - without_traj
        vmax = np.abs(diff).max()
        vmin = -vmax

        # 显示
        axes[i, 0].imshow(with_traj, cmap=cmap)
        axes[i, 0].set_title(f"{feature_name}\n(有轨迹)", fontsize=10)
        axes[i, 0].axis('off')

        axes[i, 1].imshow(without_traj, cmap=cmap)
        axes[i, 1].set_title(f"{feature_name}\n(无轨迹)", fontsize=10)
        axes[i, 1].axis('off')

        axes[i, 2].imshow(diff, cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axes[i, 2].set_title(f"{feature_name}\n(差异)", fontsize=10)
        axes[i, 2].axis('off')

    if title:
        plt.suptitle(title, fontsize=16)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=150)
        print(f"特征图对比已保存到: {save_path}")
    else:
        plt.show()

    plt.close()



def visualize_feature_maps_with_gt(feature_maps, batch_index=0, save_path=None, title=None,
                                   input_data=None, output_data=None, max_per_row=4):
    """
    可视化模型中间特征图，并添加 ground truth 信息（输入数据和输出结果）

    参数：
        feature_maps (dict): 包含特征图的字典
        batch_index (int): 可视化第几个batch
        save_path (str): 保存路径
        title (str): 图像标题
        input_data (dict): 输入数据字典，包含 'aerial_image', 'traj_image' 等
        output_data (dict): 输出数据字典，包含 'road', 'junc', 'anchor' 等
        max_per_row (int): 每行最大显示数量
    """
    import numpy as np
    import torch

    # 过滤出有效的特征图
    valid_features = {}
    for name, tensor in feature_maps.items():
        if isinstance(tensor, torch.Tensor) and tensor.dim() == 4:  # 确保是4D张量 [B, C, H, W]
            valid_features[name] = tensor

    if not valid_features:
        print("没有找到有效的特征图")
        return

    # 选择要显示的特征图（避免太多）
    feature_names = list(valid_features.keys())
    if len(feature_names) > 20:  # 如果特征图太多，只显示前20个
        feature_names = feature_names[:20]
        print(f"特征图数量过多，只显示前20个: {feature_names}")

    # 计算总行数：输入数据 + 特征图 + 输出数据
    total_rows = 0
    if input_data:
        total_rows += 1  # 输入数据行
    total_rows += int(np.ceil(len(feature_names) / max_per_row))  # 特征图行数
    if output_data:
        total_rows += 1  # 输出数据行

    n_cols = max_per_row
    fig, axes = plt.subplots(total_rows, n_cols, figsize=(4 * n_cols, 4 * total_rows))

    # 如果只有1行1列，axes 不是数组，需要转成列表处理
    if total_rows == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    current_ax = 0

    # 1. 显示输入数据
    if input_data:
        for i, (key, value) in enumerate(input_data.items()):
            if i >= max_per_row:
                break
            ax = axes[current_ax]
            current_ax += 1

            try:
                if isinstance(value, torch.Tensor):
                    img = value[batch_index].detach().cpu().numpy()
                else:
                    img = value[batch_index] if hasattr(value, '__getitem__') else value

                # 处理图像数据
                if isinstance(img, np.ndarray):
                    if img.ndim == 3 and img.shape[0] in [1, 3]:  # C, H, W -> H, W, C
                        img = img.transpose(1, 2, 0)
                    elif img.ndim == 2:
                        pass  # 灰度图
                    elif img.ndim == 3 and img.shape[2] in [1, 3]:
                        pass  # HWC 格式
                    else:
                        img = np.mean(img, axis=0)  # 降维

                    # 归一化到 [0,1] 范围便于可视化
                    if img.max() > 1.0:
                        img = img / 255.0

                    ax.imshow(img, cmap='gray' if img.ndim == 2 else None)
                    ax.set_title(f"Input: {key}", fontsize=10)
                else:
                    ax.set_title(f"Input: {key}\n(no data)", fontsize=10)
                ax.axis("off")
            except Exception as e:
                ax.set_title(f"Input: {key}\n(error: {str(e)})", fontsize=10)
                ax.axis("off")

        # 清空多余的输入subplot
        for j in range(current_ax, max_per_row):
            axes[j].axis("off")

    # 2. 显示特征图
    for i, feature_name in enumerate(feature_names):
        ax = axes[current_ax]
        current_ax += 1

        tensor = valid_features[feature_name]

        try:
            # 获取指定batch的特征图
            feature_map = tensor[batch_index].detach().cpu().numpy()

            # 如果是多通道，取平均值或选择前几个通道
            if feature_map.shape[0] > 3:
                # 多通道特征图，取平均值
                feature_map = np.mean(feature_map, axis=0)
                cmap = 'viridis'  # 使用viridis颜色映射
            elif feature_map.shape[0] == 3:
                # 3通道，直接显示
                feature_map = feature_map.transpose(1, 2, 0)
                cmap = None
            elif feature_map.shape[0] == 1:
                # 单通道
                feature_map = feature_map[0]
                cmap = 'gray'
            else:
                # 其他情况，取平均值
                feature_map = np.mean(feature_map, axis=0)
                cmap = 'viridis'

            # 归一化到[0,1]范围
            if feature_map.max() > feature_map.min():
                feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min())

            # 显示特征图
            im = ax.imshow(feature_map, cmap=cmap)
            ax.set_title(f"{feature_name}\n{tensor.shape[1]}C,{tensor.shape[2]}x{tensor.shape[3]}", fontsize=8)
            ax.axis('off')

            # 添加颜色条（对于多通道特征图）
            if feature_map.ndim == 2 and cmap != 'gray':
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        except Exception as e:
            ax.set_title(f"{feature_name}\n(error: {str(e)})", fontsize=8)
            ax.axis('off')

    # 3. 显示输出数据
    if output_data:
        for i, (key, value) in enumerate(output_data.items()):
            if i >= max_per_row:
                break
            ax = axes[current_ax]
            current_ax += 1

            try:
                if isinstance(value, torch.Tensor):
                    img = value[batch_index].detach().cpu().numpy()
                else:
                    img = value[batch_index] if hasattr(value, '__getitem__') else value

                # 处理图像数据
                if isinstance(img, np.ndarray):
                    if img.ndim == 3 and img.shape[0] in [1, 3]:  # C, H, W -> H, W, C
                        img = img.transpose(1, 2, 0)
                    elif img.ndim == 2:
                        pass  # 灰度图
                    elif img.ndim == 3 and img.shape[2] in [1, 3]:
                        pass  # HWC 格式
                    else:
                        img = np.mean(img, axis=0)  # 降维

                    # 归一化到 [0,1] 范围便于可视化
                    if img.max() > 1.0:
                        img = img / 255.0

                    # 对于输出数据，使用更明显的颜色映射
                    if img.ndim == 2:
                        cmap = 'hot' if 'anchor' in key else 'gray'
                    else:
                        cmap = None

                    ax.imshow(img, cmap=cmap)
                    ax.set_title(f"Output: {key}", fontsize=10)

                    # 为输出数据添加颜色条
                    if img.ndim == 2 and cmap != 'gray':
                        plt.colorbar(ax.imshow(img, cmap=cmap), ax=ax, fraction=0.046, pad=0.04)
                else:
                    ax.set_title(f"Output: {key}\n(no data)", fontsize=10)
                ax.axis("off")
            except Exception as e:
                ax.set_title(f"Output: {key}\n(error: {str(e)})", fontsize=10)
                ax.axis("off")

        # 清空多余的输出subplot
        for j in range(current_ax, max_per_row):
            axes[j].axis("off")

    # 清空多余的subplot
    for j in range(current_ax, len(axes)):
        axes[j].axis("off")

    if title:
        plt.suptitle(title, fontsize=16)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=150)
        print(f"特征图（含GT）已保存到: {save_path}")
    else:
        plt.show()

    plt.close()



def visualize_anchor_feature_comparison(data_dict, batch_index=0, save_path=None, title=None):
    """
    可视化轨迹特征融合前后的对比分析
    5行3列布局：
    5行：decoded_ft_1, decoded_ft_2, decoded_ft_3, decoded_ft_4, anchor_fts
    3列：融合前特征图、融合后特征图、差值图
    """
    import numpy as np
    import torch
    import matplotlib.pyplot as plt

    # 定义要分析的特征
    feature_names = ['decoded_ft_1', 'decoded_ft_2', 'decoded_ft_3', 'decoded_ft_4', 'anchor_fts']

    # 创建5行3列的布局
    fig, axes = plt.subplots(5, 3, figsize=(12, 20))

    for row, feature_name in enumerate(feature_names):
        # 获取融合前后的特征图
        no_traj_feature = None
        with_traj_feature = None

        # 查找无轨迹的特征图
        for key in data_dict.keys():
            if f'no_traj_{feature_name}' in key:
                no_traj_feature = data_dict[key][batch_index]
                break

        # 查找有轨迹的特征图
        for key in data_dict.keys():
            if f'with_traj_{feature_name}' in key:
                with_traj_feature = data_dict[key][batch_index]
                break

        # 如果没找到带前缀的，尝试直接查找（但不要用同一个特征图）
        if no_traj_feature is None:
            for key in data_dict.keys():
                if feature_name in key and 'no_traj' not in key and 'with_traj' not in key:
                    # 只取第一个找到的作为无轨迹特征
                    no_traj_feature = data_dict[key][batch_index]
                    break

        if with_traj_feature is None:
            # 如果没找到有轨迹的特征，跳过这个特征
            pass

        # 处理特征图
        if no_traj_feature is not None:
            if isinstance(no_traj_feature, torch.Tensor):
                no_traj_feature = no_traj_feature.detach().cpu().numpy()

            # 如果是多通道，取平均值
            if no_traj_feature.ndim == 3 and no_traj_feature.shape[0] > 1:
                no_traj_feature = np.mean(no_traj_feature, axis=0)

        if with_traj_feature is not None:
            if isinstance(with_traj_feature, torch.Tensor):
                with_traj_feature = with_traj_feature.detach().cpu().numpy()

            # 如果是多通道，取平均值
            if with_traj_feature.ndim == 3 and with_traj_feature.shape[0] > 1:
                with_traj_feature = np.mean(with_traj_feature, axis=0)

        # 计算差值图
        diff_feature = None
        if with_traj_feature is not None and no_traj_feature is not None:
            # 确保尺寸一致
            if with_traj_feature.shape != no_traj_feature.shape:
                import cv2
                no_traj_feature = cv2.resize(no_traj_feature, with_traj_feature.shape[::-1])
            # 正确的差值计算：With Traj - No Traj
            diff_feature = with_traj_feature - no_traj_feature

        # 绘制三列图像
        for col in range(3):
            ax = axes[row, col]

            if col == 0:  # 融合前
                img = no_traj_feature
                title_text = f"{feature_name}\n(No Traj)"
            elif col == 1:  # 融合后
                img = with_traj_feature
                title_text = f"{feature_name}\n(With Traj)"
            else:  # 差值
                img = diff_feature
                title_text = f"{feature_name}\n(Diff)"

            if img is not None:
                if col == 2:  # 差值图使用不同的可视化方法
                    # 差值图：使用 RdBu_r 颜色映射，对称范围
                    vmax = np.abs(img).max()
                    vmin = -vmax
                    im = ax.imshow(img, cmap='RdBu_r', vmin=vmin, vmax=vmax)
                    ax.set_title(title_text, fontsize=10)
                    ax.axis('off')
                    # 为差值图添加颜色条
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                else:  # 前两列使用相同的可视化方法
                    # 归一化到[0,1]范围
                    if img.max() > img.min():
                        img = (img - img.min()) / (img.max() - img.min())

                    # 显示图像
                    if img.ndim == 2:
                        im = ax.imshow(img, cmap='viridis')
                    else:
                        im = ax.imshow(img, cmap='viridis')

                    ax.set_title(title_text, fontsize=10)
                    ax.axis('off')
            else:
                ax.set_title(f"{title_text}\n(No Data)", fontsize=10)
                ax.axis('off')

    # 添加总标题
    if title:
        plt.suptitle(title, fontsize=16)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=150)
        print(f"Saved anchor feature comparison to {save_path}")
    else:
        plt.show()

    plt.close()