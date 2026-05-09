# coding=utf-8

import cv2
import random
import os
import numpy as np
from tqdm import tqdm

img_w = 256
img_h = 256

image_sets = ['20.png', '518.png', '653.png']


def gamma_transform(img, gamma):
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img, gamma_table)


def random_gamma_transform(img, gamma_vari):
    log_gamma_vari = np.log(gamma_vari)
    alpha = np.random.uniform(-log_gamma_vari, log_gamma_vari)
    gamma = np.exp(alpha)
    return gamma_transform(img, gamma)


def rotate(xb, yb, angle):
    M_rotate = cv2.getRotationMatrix2D((img_w / 2, img_h / 2), angle, 1)
    xb = cv2.warpAffine(xb, M_rotate, (img_w, img_h))
    yb = cv2.warpAffine(yb, M_rotate, (img_w, img_h))
    return xb, yb


def blur(img):
    img = cv2.blur(img, (3, 3))
    return img


def add_noise(img):
    for i in range(200):  # 添加点噪声
        temp_x = np.random.randint(0, img.shape[0])
        temp_y = np.random.randint(0, img.shape[1])
        img[temp_x][temp_y] = 255
    return img


def data_augment(xb, yb):
    if np.random.random() < 0.25:
        xb, yb = rotate(xb, yb, 90)
    if np.random.random() < 0.25:
        xb, yb = rotate(xb, yb, 180)
    if np.random.random() < 0.25:
        xb, yb = rotate(xb, yb, 270)
    if np.random.random() < 0.25:
        xb = cv2.flip(xb, 1)  # flipcode > 0：沿y轴翻转
        yb = cv2.flip(yb, 1)

    if np.random.random() < 0.25:
        xb = random_gamma_transform(xb, 1.0)

    if np.random.random() < 0.25:
        xb = blur(xb)

    if np.random.random() < 0.2:
        xb = add_noise(xb)

    return xb, yb


def multi_data_concat(i):
    # 三种模态数：分别是原始路网数据、原始遥感RGB影像、轨迹空间特征
    basemap = cv2.imread('D:/DataSet/multi_data_down/basemap/' + image_sets[i], cv2.IMREAD_GRAYSCALE)  # 像素中用1表示道路
    basemap = (np.array(basemap, dtype='float')).astype(np.uint8)
    src = cv2.imread('D:/DataSet/multi_data_down/src/' + image_sets[i])
    traj = cv2.imread('D:/DataSet/multi_data_down/traj/' + image_sets[i], cv2.IMREAD_GRAYSCALE)  # 这里的灰度是1~几百的灰度表示
    traj_point = cv2.imread('D:/DataSet/multi_data_down/trajpoint/' + image_sets[i], cv2.IMREAD_GRAYSCALE)  # 这里的灰度是1~几百的灰度表示
    basemap = np.reshape(basemap, (basemap.shape[0], basemap.shape[1], 1))
    traj = np.reshape(traj, (traj.shape[0], traj.shape[1], 1))
    traj_point = np.reshape(traj_point, (traj_point.shape[0], traj_point.shape[1], 1))
    # 按照通道拼接
    multi_feature = np.dstack((basemap, traj, traj_point, src))

    # 两种真实数据：区域建筑物轮廓、区域完整路网
    # 调整为只用路网一种
    building_label = cv2.imread('D:/DataSet/multi_data_down/building_label/' + image_sets[i], cv2.IMREAD_GRAYSCALE)  # 像素值中2表示建筑
    map_label = cv2.imread('D:/DataSet/multi_data_down/map_label/label_width2/' + image_sets[i], cv2.IMREAD_GRAYSCALE)  # 像素值中1表示路网
    # 两个label叠加，像素点同时出现两个则并以建筑覆盖
    # label = cv2.add(building_label, map_label)
    # ret, thresh = cv2.threshold(label, 2, 2, cv2.THRESH_TRUNC)

    map_label = np.reshape(map_label, (map_label.shape[0], map_label.shape[1], 1))
    return multi_feature, map_label, building_label


def multi_data_concat_test(i):
    # 三种模态数：分别是原始路网数据、原始遥感RGB影像、轨迹空间特征
    src = cv2.imread('D:/DataSet/multi_data_down/src/' + image_test_sets[i])
    traj = cv2.imread('D:/DataSet/multi_data_down/traj/' + image_test_sets[i], cv2.IMREAD_GRAYSCALE)  # 这里的灰度是1~几百的灰度表示
    traj_point = cv2.imread('D:/DataSet/multi_data_down/trajpoint/' + image_test_sets[i], cv2.IMREAD_GRAYSCALE)  # 这里的灰度是1~几百的灰度表示
    traj = np.reshape(traj, (traj.shape[0], traj.shape[1], 1))
    traj_point = np.reshape(traj_point, (traj_point.shape[0], traj_point.shape[1], 1))
    # 按照通道拼接
    multi_feature = np.dstack((traj, traj_point, src))
    map_label = cv2.imread('D:/DataSet/multi_data_down/map_label/label_width2/' + image_test_sets[i], cv2.IMREAD_GRAYSCALE)  # 像素值中1表示路网
    map_label = np.reshape(map_label, (map_label.shape[0], map_label.shape[1], 1))
    return multi_feature, map_label


def creat_dataset(image_num=75, mode='norm'):
    # mode: oroginal, augment
    print('creating dataset...')
    image_each = image_num / len(image_sets)
    g_count = 0
    for i in tqdm(range(len(image_sets))):
        count = 0
        # src_img = cv2.imread('D:/DataSet/building_recognition/unet_buildings/src/' + image_sets[i])  # 3 channels
        # label_img = cv2.imread('D:/DataSet/building_recognition/unet_buildings/all_label/' + image_sets[i], cv2.IMREAD_GRAYSCALE)  # single channel
        src_img, label_img, label_building = multi_data_concat(i)
        X_height, X_width, _ = src_img.shape
        # (5485, 5444, 5)(5485, 5444, 1)
        while count < image_each:
            # random_width = random.randint(0, X_width - img_w - 1)
            # random_height = random.randint(0,  X_height - img_h - 1)
            # # 切分训练集
            # random_width = random.randint(0, X_width - img_w - 1)
            # random_height = random.randint(0, 7*X_height // 8)
            # 切分验证集
            random_width = random.randint(0, X_width - img_w - 1)
            random_height = random.randint(7*X_height // 8, X_height - img_h - 1)

            print(random_width, random_height)
            # 256,256,5
            src_roi = src_img[random_height: random_height + img_h, random_width: random_width + img_w, :]
            label_roi = label_img[random_height: random_height + img_h, random_width: random_width + img_w]
            building_roi = label_building[random_height: random_height + img_h, random_width: random_width + img_w]

            if mode == 'augment':
                src_roi, label_roi = data_augment(src_roi, label_roi)

            visualize = np.zeros((256, 256)).astype(np.uint8)
            visualize = label_roi * 50
            # 存储完整特征和label,
            # 计算出去building区域的label
            cv2.imwrite(('D:/DataSet/multi_data_down/train_log_GKS/visualize/%d.png' % g_count), visualize)
            np.save(('D:/DataSet/multi_data_down/train_log_GKS/src/%d.npy' % g_count), src_roi)
            cv2.imwrite(('D:/DataSet/multi_data_down/train_log_GKS/label/%d.png' % g_count), label_roi)

            traj_for_mask = np.reshape(src_roi[:, :, 1], (src_roi.shape[0], src_roi.shape[1]))
            trajpoint_for_mask = np.reshape(src_roi[:, :, 2], (src_roi.shape[0], src_roi.shape[1]))
            building_mask_traj = np.where(building_roi[:, :] == 2, 0, traj_for_mask[:, :])
            building_mask_trajpoint = np.where(building_roi[:, :] == 2, 0, trajpoint_for_mask[:, :])
            building_mask_traj_and_point = np.dstack((building_mask_traj, building_mask_trajpoint))
            building_mask_PLI = np.dstack((building_mask_traj, building_mask_trajpoint, src_roi[:, :, 3:6]))

            # 看一下切出来的片 其中三个特征分别存储
            cv2.imwrite(('D:/DataSet/multi_data_down/train_log_GKS/basemap_split/%d.png' % g_count), src_roi[:, :, 0])
            cv2.imwrite(('D:/DataSet/multi_data_down/train_log_GKS/traj_split/%d.png' % g_count), src_roi[:, :, 1])
            cv2.imwrite(('D:/DataSet/multi_data_down/train_log_GKS/trajpoint_split/%d.png' % g_count), src_roi[:, :, 2])
            np.save(('D:/DataSet/multi_data_down/train_log_GKS/traj_and_point_split/%d.npy' % g_count), src_roi[:, :, 1:3])
            np.save(('D:/DataSet/multi_data_down/train_log_GKS/traj_and_point_and_img_split/%d.npy' % g_count), src_roi[:, :, 1:6])
            cv2.imwrite(('D:/DataSet/multi_data_down/train_log_GKS/src_split/%d.png' % g_count), src_roi[:, :, 3:6])
            cv2.imwrite(('D:/DataSet/multi_data_down/train_log_GKS/building_label/%d.png' % g_count), building_roi)
            np.save(('D:/DataSet/multi_data_down/train_log_GKS/building_mask_traj_and_point_split/%d.npy' % g_count), building_mask_traj_and_point)
            np.save(('D:/DataSet/multi_data_down/train_log_GKS/building_mask_PLI_split/%d.npy' % g_count), building_mask_PLI)
            count += 1
            g_count += 1
            print(g_count)


def creat_dataset_test(image_num=50, mode='norm'):
    # mode: oroginal, augment
    print('creating dataset...')
    image_each = image_num / len(image_test_sets)
    g_count = 0
    for i in tqdm(range(len(image_test_sets))):
        count = 0
        src_img, label_img = multi_data_concat_test(i)
        X_height, X_width, _ = src_img.shape
        while count < image_each:
            random_width = random.randint(0, X_width - img_w - 1)
            random_height = random.randint(0,  X_height - img_h - 1)

            # 256,256,5
            src_roi = src_img[random_height: random_height + img_h, random_width: random_width + img_w, :]
            label_roi = label_img[random_height: random_height + img_h, random_width: random_width + img_w]

            if mode == 'augment':
                src_roi, label_roi = data_augment(src_roi, label_roi)

            visualize = np.zeros((256, 256)).astype(np.uint8)
            visualize = label_roi * 50

            cv2.imwrite(('D:/DataSet/multi_data_down/train_log_GKS/visualize/%d.png' % g_count), visualize)
            np.save(('D:/DataSet/multi_data_down/train_log_GKS/src/%d.npy' % g_count), src_roi)
            cv2.imwrite(('D:/DataSet/multi_data_down/train_log_GKS/label/%d.png' % g_count), label_roi)

            # 看一下切出来的片 其中三个特征分别存储
            cv2.imwrite(('D:/DataSet/multi_data_down/train_log_GKS/traj_split/%d.png' % g_count), src_roi[:, :, 0])
            cv2.imwrite(('D:/DataSet/multi_data_down/train_log_GKS/trajpoint_split/%d.png' % g_count), src_roi[:, :, 1])
            np.save(('D:/DataSet/multi_data_down/train_log_GKS/traj_and_point_split/%d.npy' % g_count), src_roi[:, :, 0:2])
            np.save(('D:/DataSet/multi_data_down/train_log_GKS/traj_and_point_and_img_split/%d.npy' % g_count), src_roi[:, :, :])
            cv2.imwrite(('D:/DataSet/multi_data_down/train_log_GKS/src_split/%d.png' % g_count), src_roi[:, :, 2:5])
            count += 1
            g_count += 1
            print(g_count)


def find_all_data_and_extract(path):
    """
    将traj和trajpoint合并
    :param path:
    :return:
    """
    if not os.path.exists(path):
        print('路径存在问题：', path)
        return None

    for i in os.listdir(path):
        if os.path.isfile(path + "/" + i):
            if 'npy' in i:
                traj_and_point = np.load(('D:/DataSet/multi_data/train_expriment/val/src/' + i))
                # cv2.imwrite(('D:/DataSet/multi_data/train_expriment/train/traj_and_point_split/%d.png' % int(i.split('.')[0])), traj_and_point[:, :, 1:2])
                np.save(('D:/DataSet/multi_data/train_expriment/val/traj_and_point_and_img_split/%d.npy' % int(i.split('.')[0])), traj_and_point[:, :, 1:])
                print(i)
        else:
            find_all_data_and_extract(path + "/" + i)


if __name__ == '__main__':
    creat_dataset(mode='norm')
    # creat_dataset_test(mode='norm')

    # find_all_data_and_extract('D:/DataSet/multi_data/train_expriment/val/src/')
    #
    # path = 'D:/DataSet/multi_data_down/train_log_GKS/'
    # img_num = 8
    # traj = cv2.imread(path + 'traj_split/%d.png' % img_num, cv2.IMREAD_GRAYSCALE)
    # trajpoint = cv2.imread(path + 'trajpoint_split/%d.png' % img_num, cv2.IMREAD_GRAYSCALE)
    # trajandpoint = np.load(path + 'traj_and_point_split/%d.npy' % img_num)
    # building_label = cv2.imread(path + 'building_label/%d.png' % img_num, cv2.IMREAD_GRAYSCALE)
    # label = cv2.imread(path + 'label/%d.png' % img_num, cv2.IMREAD_GRAYSCALE)
    # building_mask_PL = np.load(path + '/building_mask_traj_and_point_split/%d.npy' % img_num)
    # PLI = np.load(path + '/traj_and_point_and_img_split/%d.npy' % img_num)
    # building_mask_PLI = np.load(path + '/building_mask_PLI_split/%d.npy' % img_num)
    # all = np.load(path + '/src/%d.npy' % img_num)

    # print(traj[0, :100], trajpoint[0, :100], trajandpoint[0, :100, :], label[0, :100])
    # print(traj[50, 50:150], trajpoint[50, 50:150], building_mask_PL[50, 50:150], building_label[50, 50:150], label[50, 50:150])
    # print(traj[100, 10:25], '\n', trajpoint[100, 10:25])
    # print(building_label[100, 10:25], '\n', label[100, 10:25])
    # print(PLI[100, 10:25])
    # print(building_mask_PL[100, 10:25], end=' \n')
    # print(building_mask_PLI[100, 10:25], end=' \n')



