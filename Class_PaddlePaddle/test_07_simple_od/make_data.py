# Author:  Acer Zhang
# Datetime:2019/9/11
# Copyright belongs to the author.
# Please indicate the source for reprinting.

# 该程序为生产数据集而生

# import block
import PIL.Image as Image
import os
import random
from os_tools import mkdir

with open("./oridata/OCR_100P.txt", "r") as f:
    labels = f.read()


def random_size(pil_obj):
    """随机目标大小"""
    tmp = random.uniform(1, 3)
    pil_obj = pil_obj.resize((int(14 * tmp), int(24 * tmp)), Image.ANTIALIAS)
    return pil_obj


def paste_all_in_one():
    """
    随机将多个基础验证码合并到一张大图中
    :return:Pil_obj,label_info
    label_info Type: Ox(中心W坐标), Oy(中心H坐标), W(距离中心点W方向距离)， H(距离中心点H方向距离)， label
    """
    data_path = "./oridata/"
    sum_base_img = random.randint(3, 10)
    base_img_list = []
    floor_color = (255, 255, 255)
    label_info_list = []
    img_info_list = []
    for i in range(sum_base_img):
        send_img = random.randint(random.randint(i, 1000), random.randint(1000, 1900))
        label = labels[send_img - 1]
        send_img = Image.open(data_path + str(send_img) + ".jpg")
        send_img = send_img.crop((1, 3, 15, 27))
        send_img = random_size(send_img)
        base_img_list.append(send_img)
        label_info_list.append(label)
        floor_color = send_img.getpixel((2, 2))
    floor_img = Image.new('RGB', (512, 512), floor_color)
    for i, img, label in zip(range(10), base_img_list, label_info_list):
        local_w = random.randint(43 * i, 70 + 43 * i - img.size[0])
        local_h = random.randint(1, 500 - img.size[1])
        box = (local_w, local_h, local_w + img.size[0], local_h + img.size[1])
        info = [local_w + 0.5 * img.size[0],
                local_h + 0.5 * img.size[1],
                0.5 * img.size[0],
                0.5 * img.size[1],
                int(label)]
        img_info_list.append(info)
        floor_img.paste(img, box)
    return floor_img, img_info_list


def make_img(make_num, save_path):
    """
    制作数据集，保存形式为: "./img/" + {ID}.jpg, "./info/" + {ID}.info
    :param make_num: Type int
    :param save_path: "./xxx"
    :return:
    """
    img_path = os.path.join(save_path, "img").replace("\\", "/")
    info_path = os.path.join(save_path, "info").replace("\\", "/")
    mkdir(img_path, de=True)
    mkdir(info_path, de=True)
    for i in range(make_num):
        floor_img, label_info_list = paste_all_in_one()
        floor_img.save(img_path + "/" + str(i) + ".jpg")
        with open(info_path + "/" + str(i) + ".info", "a") as f:
            for info in label_info_list:
                info = str(info)[1:-1]
                f.writelines(info + "\n")


path = "./data"
make_img(10000, save_path=path)