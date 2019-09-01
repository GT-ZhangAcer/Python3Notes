# Author:  Acer Zhang
# Datetime:2019/8/25
# Copyright belongs to the author.
# Please indicate the source for reprinting.

import matplotlib.pyplot as plt
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 用来显示中文


def make_classify_info(file_path):
    """
    :param file_path:log file path

    Data Example:

    2019-08-31-21-01
    0.13099-0.54513-2.31339_0.16342-0.58067-2.27263_2.5705785751342773
    0.17225-0.60655-2.25609_0.18672-0.62579-2.24045_2.529569625854492

    First line:Date
    Other :Data

    data Type: train_acc1 - train_acc5 - train_loss _ test_acc1 - test_acc5 - test_loss _ cost_time

    """
    train_acc1 = []
    train_acc5 = []
    train_loss = []
    test_acc1 = []
    test_acc5 = []
    time_list = []
    with open(file_path, "r") as f:
        data = f.readlines()
        for line_id, info in enumerate(data):
            if line_id == 0:
                continue
            info = info.split("_")
            train_acc1.append([float(info[0].split("-")[0])])
            train_acc5.append([float(info[0].split("-")[1])])
            try:
                train_loss.append([float(info[0].split("-")[2])])
            except ValueError:
                train_loss.append([0])
            test_acc1.append([float(info[1].split("-")[0])])
            test_acc5.append([float(info[1].split("-")[1])])
            time_list.append(float(info[2].split("-")[0].replace("\n", "")))

    print("The time cost on this train is : " + str(round(sum(time_list), 3)))
    epoch_list = [i + 1 for i in range(len(time_list))]
    return epoch_list, train_acc1, train_acc5, train_loss, test_acc1, test_acc5, time_list


def make_2label(epoch_list, a_list, b_list, title, y_label='accuracy rate', a_label='train', b_label='validation'):
    """
    制作出图
    :param epoch_list: X轴数据
    :param a_list: 数据A
    :param b_list: 数据B
    :param title: 标题
    :param y_label: Y轴标签
    :param a_label: 图例A
    :param b_label: 图例B
    :return: epoch_list, train_acc1, train_acc5, train_loss, test_acc1, test_acc5, time_list
    """
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(y_label)
    plt.ylim(0, 1)
    plt.plot(epoch_list, a_list, 'r', label=a_label)
    plt.plot(epoch_list, b_list, 'b', label=b_label)
    plt.legend(bbox_to_anchor=[1, 1])
    plt.grid()


dataC1 = make_classify_info(
    r'F:\Python3Notes\Class_PaddlePaddle\test06_Parameter_Adjustment\test01_Optimizer\2019-08-31-22-23C1.log')
dataC2 = make_classify_info(
    r'F:\Python3Notes\Class_PaddlePaddle\test06_Parameter_Adjustment\test01_Optimizer\2019-08-31-22-22C2.log')

# epoch_list, train_acc1, train_acc5, train_loss, test_acc1, test_acc5, time_list
# Top1
make_2label(epoch_list=dataC1[0], a_list=dataC1[1], b_list=dataC2[1], title="Cifar-10 Top1 Train 准确率", a_label="SGD",
            b_label="Adam")
plt.show()

make_2label(epoch_list=dataC1[0], a_list=dataC1[4], b_list=dataC2[4], title="Cifar-10 Top1 Validation 准确率", a_label="SGD",
            b_label="Adam")
plt.show()
# Top5
make_2label(epoch_list=dataC1[0], a_list=dataC1[5], b_list=dataC2[5], title="Cifar-10 Top5 Validation 准确率", a_label="SGD",
            b_label="Adam")
plt.show()
# Loss
make_2label(epoch_list=dataC1[0], a_list=dataC1[3], b_list=dataC2[3], title="Cifar-10 loss", a_label="SGD",
            b_label="Adam")
plt.show()
