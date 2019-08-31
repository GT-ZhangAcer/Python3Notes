# Author:  Acer Zhang
# Datetime:2019/8/25
# Copyright belongs to the author.
# Please indicate the source for reprinting.

import matplotlib.pyplot as plt
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 用来显示中文


def make_classify_train_plt(file_path):
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
    with open(file_path, "w") as f:
        data = f.readlines()
        for line_id, info in enumerate(data):
            if line_id == 1:
                continue
            info = info.split("_")
            train_acc1.append([info[0].split("-")][0])
            train_acc5.append([info[0].split("-")][1])
            train_loss.append([info[0].split("-")][2])
            test_acc1.append([info[1].split("-")][0])
            test_acc5.append([info[1].split("-")][1])
            time_list.append([info[2].split("-")][0])

    print("The time cost on this train is : " + str(round(sum(time_list), 3)))
    epoch_list = [i + 1 for i in range(len(time_list))]
    fig2 = plt.figure()
    fig2.set_ylabel()
    plt.title('分类准确率指标')
    plt.ylabel('accuracy rate')
    plt.xlabel('Epoch percentage')
    plt.ylim(0, 1)
    plt.plot(epoch_list, y1, 'r', label='train')
    plt.plot(epoch_list, y2, 'b', label='test')

    plt.legend(bbox_to_anchor=[1, 1])
    plt.grid()
    plt.show()
