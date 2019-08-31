# Author:  Acer Zhang
# Datetime:2019/8/25
# Copyright belongs to the author.
# Please indicate the source for reprinting.

# Tips : If you want use this py in AI Studio GPU, Please edit rootPath to /home/aistudio/work and make big batch_size
import os
import sys

# 切换工作目录
rootPath = os.path.dirname(sys.path[0])
os.chdir(rootPath)

import paddle.fluid as fluid
from script.log_Script import WriteLog
from script.reader_Script import data_cifar10
from test01_Optimizer.mini_restnet import resnet_cifar10

# Hyper parameter
use_cuda = True  # Whether to use GPU or not
batch_size = 64  # Number of incoming batches of data
epochs = 1000  # Number of training rounds
save_model_path = "./model"
learning_rate = 0.001


# normalization
def img_normalization(img):
    img = img / 255
    return img


# Initialization
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
exe = fluid.Executor(place)

# Program
main_program = fluid.Program()
startup = fluid.Program()

# Edit Program
with fluid.program_guard(main_program=main_program, startup_program=startup):
    """Tips:Symbol * stands for Must"""
    # * Define data types
    input_img = fluid.layers.data(name="input_img", shape=[3, 32, 32])
    label = fluid.layers.data(name="label", shape=[1], dtype="int64")
    # * Access to the Network
    net = resnet_cifar10(input_img)
    # fluid.io.load_params(exe, "./test01_Optimizer/params", main_program=main_program)
    # * Define loss function
    loss = fluid.layers.cross_entropy(input=net, label=label)
    #  Access to statistical information
    loss = fluid.layers.mean(loss)
    acc1 = fluid.layers.accuracy(input=net, label=label, k=1)
    acc5 = fluid.layers.accuracy(input=net, label=label, k=5)
    # Clone program
    test_program = main_program.clone()
    # * Define the optimizer

    fluid.optimizer.SGD(learning_rate=learning_rate).minimize(loss)
    # fluid.optimizer.Adam(learning_rate=learning_rate).minimize(loss)

# Feed configure
# if you want to shuffle "reader=paddle.reader.shuffle(dataReader(), buf_size)"
train_reader, test_reader = data_cifar10(batch_size=batch_size, modifier_def=img_normalization)
train_feeder = fluid.DataFeeder(place=place, feed_list=[input_img, label])

# if you want to asynchronous reading
# batch_reader = fluid.io.PyReader(feed_list=[x, y], capacity=64)
# batch_reader.decorate_sample_list_generator(paddle.batch(data_reader(), batch_size=batch_size),place)

# Train Process
exe.run(startup)
log_obj = WriteLog(path="./test01_Optimizer")

for epoch in range(epochs):
    for step, data in enumerate(train_reader()):
        outs = exe.run(program=main_program,
                       feed=train_feeder.feed(data),
                       fetch_list=[acc1, acc5, loss])
        log_obj.add_batch_train_value(outs[0], outs[1], outs[2])

    for step, data in enumerate(test_reader()):
        outs = exe.run(program=test_program,
                       feed=train_feeder.feed(data),
                       fetch_list=[acc1, acc5, loss])
        log_obj.add_batch_test_value(outs[0], outs[1], outs[2])
    train_print, test_print = log_obj.write_and_req()
    print(epoch, "Train acc1 ", train_print["acc1"], "acc5 ", train_print["acc5"], "loss ", train_print["loss"])
    print(epoch, "Test  acc1 ", test_print["acc1"], "acc5 ", test_print["acc5"], "loss ", test_print["loss"])
    # fluid.io.save_params(executor=exe, dirname="./test01_Optimizer/params", main_program=main_program)
    # break
