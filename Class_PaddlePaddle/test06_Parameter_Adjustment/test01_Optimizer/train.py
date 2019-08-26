# Author: Acer Zhang
# Datetime:2019/8/25
# Copyright belongs to the author.
# Please indicate the source for reprinting.

import paddle.fluid as fluid
import paddle
from script.log_Script import WriteLog
from script.reader_Script import data_normal_id_img
from net.resnet_vd import ResNet50_vd

import os
import sys

# 切换工作目录
rootPath = os.path.dirname(sys.path[0])
os.chdir(rootPath)

# Hyper parameter
use_cuda = False  # Whether to use GPU or not
batch_size = 512  # Number of incoming batches of data
epochs = 1000  # Number of training rounds
save_model_path = "./model"
learning_rate = 0.001
data_num_rate = 1  # Reader data rate

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
    input_img = fluid.layers.data(name="input_img", shape=[1, 30, 15])
    label = fluid.layers.data(name="label", shape=[1], dtype="int64")
    # * Access to the Network
    net = ResNet50_vd().net(input=input_img, class_dim=10)
    fluid.io.load_params(exe, "./net/ResNet50_vd_v2_pretrained", main_program=main_program)
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
train_reader, test_reader = data_normal_id_img(batch_size=batch_size)
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

    fluid.io.save_persistables(dirname=save_model_path + "/" + str(epoch) + "persistables", executor=exe,
                               main_program=main_program)
    fluid.io.save_inference_model(dirname=save_model_path + "/" + str(epoch),
                                  feeded_var_names=["input_img"], target_vars=[net], main_program=main_program,
                                  executor=exe)
