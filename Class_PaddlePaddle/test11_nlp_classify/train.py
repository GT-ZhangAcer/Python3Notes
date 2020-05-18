# Author: Acer Zhang
# Datetime:2020/5/18 13:09
# Copyright belongs to the author.
# Please indicate the source for reprinting.

from __future__ import print_function
import paddle
import paddle.fluid as fluid
import numpy as np
import sys
import math

CLASS_DIM = 2  # 情感分类的类别数
EMB_DIM = 128  # 词向量的维度
HID_DIM = 512  # 隐藏层的维度
STACKED_NUM = 3  # LSTM双向栈的层数
BATCH_SIZE = 128  # batch的大小


def convolution_net(data, input_dim, class_dim, emb_dim, hid_dim):
    emb = fluid.embedding(
        input=data, size=[input_dim, emb_dim], is_sparse=True)
    conv_3 = fluid.nets.sequence_conv_pool(
        input=emb,
        num_filters=hid_dim,
        filter_size=3,
        act="tanh",
        pool_type="sqrt")
    conv_4 = fluid.nets.sequence_conv_pool(
        input=emb,
        num_filters=hid_dim,
        filter_size=4,
        act="tanh",
        pool_type="sqrt")
    prediction = fluid.layers.fc(
        input=[conv_3, conv_4], size=class_dim, act="softmax")
    return prediction


def stacked_lstm_net(data, input_dim, class_dim, emb_dim, hid_dim, stacked_num):
    # 计算词向量
    emb = fluid.embedding(
        input=data, size=[input_dim, emb_dim], is_sparse=True)

    # 第一层栈
    # 全连接层
    fc1 = fluid.layers.fc(input=emb, size=hid_dim)
    # lstm层
    lstm1, cell1 = fluid.layers.dynamic_lstm(input=fc1, size=hid_dim)

    inputs = [fc1, lstm1]

    # 其余的所有栈结构
    for i in range(2, stacked_num + 1):
        fc = fluid.layers.fc(input=inputs, size=hid_dim)
        lstm, cell = fluid.layers.dynamic_lstm(
            input=fc, size=hid_dim, is_reverse=(i % 2) == 0)
        inputs = [fc, lstm]

    # 池化层
    fc_last = fluid.layers.sequence_pool(input=inputs[0], pool_type='max')
    lstm_last = fluid.layers.sequence_pool(input=inputs[1], pool_type='max')

    # 全连接层，softmax预测
    prediction = fluid.layers.fc(
        input=[fc_last, lstm_last], size=class_dim, act='softmax')
    return prediction


def inference_program(word_dict):
    data = fluid.data(
        name="words", shape=[None], dtype="int64", lod_level=1)
    dict_dim = len(word_dict)
    net = convolution_net(data, dict_dim, CLASS_DIM, EMB_DIM, HID_DIM)
    # net = stacked_lstm_net(data, dict_dim, CLASS_DIM, EMB_DIM, HID_DIM, STACKED_NUM)
    return net


def train_program(prediction):
    label = fluid.data(name="label", shape=[None, 1], dtype="int64")
    cost = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_cost = fluid.layers.mean(cost)
    accuracy = fluid.layers.accuracy(input=prediction, label=label)
    return [avg_cost, accuracy]  # 返回平均cost和准确率acc


# 优化函数
def optimizer_func():
    return fluid.optimizer.Adam(learning_rate=0.002)


use_cuda = False  # 在cpu上进行训练
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

print("Loading IMDB word dict....")
word_dict = paddle.dataset.imdb.word_dict()

print("Reading training data....")
train_reader = fluid.io.batch(
    fluid.io.shuffle(
        paddle.dataset.imdb.train(word_dict), buf_size=9),
    batch_size=1)
print("Reading testing data....")
test_reader = fluid.io.batch(
    paddle.dataset.imdb.test(word_dict), batch_size=BATCH_SIZE)

exe = fluid.Executor(place)
prediction = inference_program(word_dict)
[avg_cost, accuracy] = train_program(prediction)  # 训练程序
sgd_optimizer = optimizer_func()  # 训练优化函数
sgd_optimizer.minimize(avg_cost)
# Specify the directory path to save the parameters
params_dirname = "./understand_sentiment_conv.inference.model"

feed_order = ['words', 'label']
pass_num = 1  # 训练循环的轮数


# 程序主循环部分
def train_loop(main_program):
    # 启动上文构建的训练器
    exe.run(fluid.default_startup_program())

    feed_var_list_loop = [
        main_program.global_block().var(var_name) for var_name in feed_order
    ]
    feeder = fluid.DataFeeder(
        feed_list=feed_var_list_loop, place=place)

    test_program = fluid.default_main_program().clone(for_test=True)

    # 训练循环
    for epoch_id in range(pass_num):
        for step_id, data in enumerate(train_reader()):
            # 运行训练器
            metrics = exe.run(main_program,
                              feed=feeder.feed(data),
                              fetch_list=[avg_cost, accuracy])

            # 测试结果
            avg_cost_test, acc_test = train_test(test_program, test_reader)
            print('Step {0}, Test Loss {1:0.2}, Acc {2:0.2}'.format(
                step_id, avg_cost_test, acc_test))

            print("Step {0}, Epoch {1} Metrics {2}".format(
                step_id, epoch_id, list(map(np.array,
                                            metrics))))

            if params_dirname is not None:
                fluid.io.save_inference_model(params_dirname, ["words"],
                                              prediction, exe)  # 保存模型
            return


def train_test(program, reader):
    count = 0
    feed_var_list = [
        program.global_block().var(var_name) for var_name in feed_order
    ]
    feeder_test = fluid.DataFeeder(feed_list=feed_var_list, place=place)
    test_exe = fluid.Executor(place)
    accumulated = len([avg_cost, accuracy]) * [0]
    for test_data in reader():
        avg_cost_np = test_exe.run(
            program=program,
            feed=feeder_test.feed(test_data),
            fetch_list=[avg_cost, accuracy])
        accumulated = [
            x[0] + x[1][0] for x in zip(accumulated, avg_cost_np)
        ]
        count += 1
    return [x / count for x in accumulated]


train_loop(fluid.default_main_program())
