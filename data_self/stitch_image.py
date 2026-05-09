from PIL import Image
import os


def stitch_images(image_paths, rows=4, cols=4):
    # 计算每幅图像的尺寸
    first_image = Image.open(image_paths[0])
    img_width, img_height = first_image.size

    # 创建一个空的拼接图像，尺寸为4x4矩阵的总尺寸
    total_width = img_width * cols
    total_height = img_height * rows
    stitched_image = Image.new("RGB", (total_width, total_height))

    # 遍历每一幅图像并放到合适的位置
    for i, image_path in enumerate(image_paths):
        img = Image.open(image_path)
        # 如果图像是二值图像，转换为黑白模式并确保是二值图

        # if img.mode != '1':  # 检查是否是二值模式
        #     img = img.convert('L')  # 转换为灰度图
        #     img = img.point(lambda p: p > 128 and 255)  # 只保留0和255，确保是二值图

        # 计算当前图像的放置位置
        row = i // cols  # 当前图像的行
        col = i % cols  # 当前图像的列

        # 计算该位置左上角的坐标
        x_offset = col * img_width
        y_offset = (rows - 1 - row) * img_height  # 从左下角开始，所以要调整行的顺序

        # 将图像粘贴到合适的位置
        stitched_image.paste(img, (x_offset, y_offset))

    return stitched_image


def crop_slices(stitched_image, slice_size=2048, num_slices=3):
    slices = []
    img_width, img_height = stitched_image.size

    for row in range(num_slices):
        for col in range(num_slices):
            # 计算当前切片的坐标
            left = col * slice_size
            top = row * slice_size
            right = left + slice_size
            bottom = top + slice_size

            # 裁剪并添加到切片列表
            cropped_slice = stitched_image.crop((left, top, right, bottom))
            slices.append(cropped_slice)

    return slices


# # 假设您的图像路径是从 'image_0.png' 到 'image_15.png'
# # image_paths = [f"../data_self/input/imagery/xian/{i}/point.png" for i in range(16)]
# image_paths = [f"../data_self/input/imagery/xian/{i}/point.png" for i in range(16)]
#
# # 拼接图像
# result_image = stitch_images(image_paths)
#
# # 保存拼接后的图像
# result_image.save("../data_self/input/imagery/stitched_image.png")

result_image = Image.open("../data_self/input/imagery/xian.png")
slices = result_image.crop((0, 0, 3*2048, 3*2048))
slices.save("../data_self/input/imagery/stitched_image_slices.png")
