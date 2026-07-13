#!/user/bin/python
# coding=utf-8

import os
import torch
from lib import geom, graph as graph_helper
import numpy as np
import logging
import time
import cv2 as cv
import torch.distributed as dist
import numpy as np
import cv2


def _print_key_list(title, keys, max_items=80):
    keys = list(keys)
    if not keys:
        return
    print("=> {}:".format(title))
    for key in keys[:max_items]:
        print("   - {}".format(key))
    if len(keys) > max_items:
        print("   ... {} more".format(len(keys) - max_items))


def load_pretrained(model, fname, optimizer=None, strict=True):
    """
    resume training from previous checkpoint
    :param fname: filename(with path) of checkpoint file
    :return: model, optimizer, checkpoint epoch
    """
    if os.path.isfile(fname):
        print("=> loading checkpoint '{}'".format(fname))

        checkpoint = torch.load(fname)
        state_dict = checkpoint['state_dict']
        model_state = model.state_dict()
        shape_mismatch_keys = []
        for key, value in state_dict.items():
            if key in model_state and model_state[key].shape != value.shape:
                shape_mismatch_keys.append(
                    "{}: checkpoint {} vs model {}".format(
                        key, tuple(value.shape), tuple(model_state[key].shape)))
        _print_key_list("checkpoint shape mismatch keys", shape_mismatch_keys)
        # model = torch.nn.DataParallel(model).cuda()
        model = model.cuda()
        incompatible = model.load_state_dict(state_dict, strict=False)
        print("=> checkpoint missing keys: {}, unexpected keys: {}".format(
            len(incompatible.missing_keys), len(incompatible.unexpected_keys)))
        _print_key_list("checkpoint missing key names", incompatible.missing_keys)
        _print_key_list("checkpoint unexpected key names", incompatible.unexpected_keys)

        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
            return model, optimizer
        else:
            return model
    else:
        print("=> no checkpoint found at '{}'".format(fname))


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def numpy2tensor2cuda(batch_inputs):
    return torch.autograd.Variable(torch.from_numpy(batch_inputs).float()).cuda()
    # return torch.from_numpy(batch_inputs).float().cuda()

def dilate_label_batch(label, kernel_size=3, iterations=1):
    """
    对输入的 (B, 1, H, W) 标签 tensor 执行 OpenCV 膨胀操作，返回膨胀后的新 tensor。

    参数:
        label_tensor: torch.Tensor, 形状为 (B, 1, H, W)，数值范围应为 [0, 1]
        kernel_size: 卷积核尺寸，默认 3 表示 3x3 膨胀
        iterations: 膨胀迭代次数，默认 1 次

    返回:
        torch.Tensor: 相同形状的 tensor，值仍为 float32 格式
    """
    B, C, H, W = label.shape
    new_labels = np.zeros_like(label)

    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    for i in range(B):
        # 输入图像需为 uint8
        label_img = (label[i, 0] > 0.5).astype(np.uint8) * 255
        dilated = cv2.dilate(label_img, kernel, iterations=iterations)
        new_labels[i, 0] = dilated.astype(np.float32) / 255.0

    return new_labels


def get_logger(logger_name="logtrain", log_dir="data/logs/"):
    timenow = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

    logging_dir = os.path.join(log_dir)
    if not os.path.isdir(logging_dir):
        os.makedirs(logging_dir, exist_ok=True)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(fmt))
    logger.addHandler(console)

    logname = logger_name + "_" + timenow + '.txt'
    handler = logging.FileHandler(os.path.join(logging_dir, logname))
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)

    return logger


class MapContainer(object):
    def __init__(self, path, region_name, IMG_SZ):
        self.map = np.zeros((2, IMG_SZ, IMG_SZ))
        self.path = path
        self.region_name = region_name

    def add_map(self, pnt, map, CROP_SZ):
        # map[map > 0.5] = 1
        # map[map <= 0.5] = 0
        self.map[0, pnt[0]:pnt[0] + CROP_SZ, pnt[1]:pnt[1] + CROP_SZ] += map
        self.map[1, pnt[0]:pnt[0] + CROP_SZ, pnt[1]:pnt[1] + CROP_SZ] += 1

    def add_batch_gpu(self, pnt_lst, maps_cuda, CROP_SZ):
        maps_np = torch.sigmoid(maps_cuda).data.cpu().numpy()
        for batch_i, pnt in enumerate(pnt_lst):
            self.add_map(pnt, maps_np[batch_i, 0, :, :], CROP_SZ)

    def add_batch_cpu(self, pnt_lst, maps_np, CROP_SZ):
        for batch_i, pnt in enumerate(pnt_lst):
            self.add_map(pnt, maps_np[batch_i, 0, :, :], CROP_SZ)

    def close(self):
        self.map[0] /= self.map[1]

    def save_map(self):
        cv.imwrite(os.path.join(self.path, self.region_name + ".png"), self.map[0].swapaxes(0, 1) * 255)

    def get_map(self):
        return self.map[0]

def random_sample_given_probs(seq, probs):
    sum_probs = sum(probs)
    if sum_probs != 1:
        probs = [x/sum_probs for x in probs]
    probs.insert(0, 0)
    for i in range(len(probs)-1):
        probs[i+1] = probs[i] + probs[i+1]
    rand = random.random()
    for i in range(len(probs)-1):
        if probs[i] < rand < probs[i+1]:
            break
    return seq[i]

def split_tile_img():
    img = cv.imread("/home/wangziyu/VecRoad/data_self/input/imagery/653_0_0.png")
    if img is None:
        raise FileNotFoundError("hard-coded source image for split_tile_img was not found")
    # 将图片填充到大图的相应位置
    split_img = img[:4096, :4096, :]
    cv.imwrite("/home/wangziyu/VecRoad/data_self/input/tile/653_0_0.png", split_img)

if __name__ == "__main__":
    split_tile_img()
