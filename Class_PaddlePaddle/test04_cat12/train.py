import os
from PIL import Image
import cv2 as cv
import numpy
import traceback
import paddle.fluid as fluid
import paddle
import numpy as np
from PIL import Image

import reader
from se_resnext import SE_ResNeXt152_32x4d


def readLabel(path):
    '''
    猫12标签读取工具
    :param path:
    :return:
    '''
    with open(path, "r") as f:
        info = f.read().split("\n")
        info = [i.split("\t") for i in info]
        imgPL = [i[0] for i in info]
        labelL = [i[1] for i in info]
    return imgPL, labelL


# imgPL, labelL=readLabel("train_list.txt")

place = fluid.CUDAPlace(0)
# place = fluid.CPUPlace()
exe = fluid.Executor(place)
# 新建项目
defectProgram = fluid.Program()  # 主程序
startup = fluid.Program()  # 默认启动程序

train_reader = paddle.batch(reader.train(), batch_size=8)
val_reader = paddle.batch(reader.val(), batch_size=8)

with fluid.program_guard(main_program=defectProgram, startup_program=startup):
    img = fluid.layers.data(name="img", shape=[3, 224, 224], dtype="float32")
    label = fluid.layers.data(name="label", shape=[1], dtype="int64")

    net_x = SE_ResNeXt152_32x4d().net(ipt=img, class_dim=12)

    net_x.stop_gradient=True
    net_x = fluid.layers.fc(input=net_x, size=12, act='softmax')
    # 定义损失函数 此处应设置为软标签
    cost = fluid.layers.cross_entropy(input=net_x, label=label)
    avg_cost = fluid.layers.mean(cost)
    acc_1 = fluid.layers.accuracy(input=net_x, label=label, k=1)
    testProgram = defectProgram.clone(for_test=True)

    # 定义优化方法
    sgd_optimizer = fluid.optimizer.Adam(learning_rate=0.001)
    sgd_optimizer.minimize(avg_cost)

feeder = fluid.DataFeeder(place=place, feed_list=[img, label])

exe.run(startup)

trainNum = 100
maxacc = 0
for i in range(trainNum):
    accL = []
    taccL = []
    for batch_id, data in enumerate(train_reader()):
        outs = exe.run(program=defectProgram,
                       feed=feeder.feed(data),
                       fetch_list=[avg_cost, acc_1])
        try:
            accL.append(float(outs[1]))
        except:
            pass
    print("TRAIN", i, sum(accL) / len(accL))
    for batch_id, data in enumerate(val_reader()):
        outs = exe.run(program=testProgram,
                       feed=feeder.feed(data),
                       fetch_list=[avg_cost, acc_1])
        try:
            taccL.append(float(outs[1]))
        except:
            pass
    finalacc = sum(taccL) / len(taccL)
    print("TEST", i, finalacc)
    if finalacc > maxacc:
        fluid.io.save_inference_model("./data/" + str(i), ['img'], [net_x], exe, main_program=defectProgram)
