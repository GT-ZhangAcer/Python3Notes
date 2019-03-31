import os
from PIL import Image # 导入图像处理模块
import matplotlib.pyplot as plt
import numpy
import paddle # 导入paddle模块
import paddle.fluid as fluid

#定义softmax分类器
def softmax_regression():
    """
    定义softmax分类器：
        一个以softmax为激活函数的全连接层
    Return:
        predict_image -- 分类的结果
    """
    # 输入的原始图像数据，大小为28*28*1
    img = fluid.layers.data(name='img', shape=[1, 28, 28], dtype='float32')
    # 以softmax为激活函数的全连接层，输出层的大小必须为数字的个数10
    predict = fluid.layers.fc(
        input=img, size=10, act='softmax')
    return predict

def train_program():
    """
    配置train_program

    Return:
        predict -- 分类的结果
        avg_cost -- 平均损失
        acc -- 分类的准确率

    """
    # 标签层，名称为label,对应输入图片的类别标签
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    predict = softmax_regression() # 取消注释将使用 Softmax回归
    # predict = multilayer_perceptron() # 取消注释将使用 多层感知器
    #predict = convolutional_neural_network() # 取消注释将使用 LeNet5卷积神经网络

    # 使用类交叉熵函数计算predict和label之间的损失函数
    cost = fluid.layers.cross_entropy(input=predict, label=label)
    # 计算平均损失
    avg_cost = fluid.layers.mean(cost)
    # 计算分类准确率
    acc = fluid.layers.accuracy(input=predict, label=label)
    return predict, [avg_cost, acc]

def optimizer_program():
    return fluid.optimizer.Adam(learning_rate=0.001)

# 一个minibatch中有64个数据
BATCH_SIZE = 64

# 每次读取训练集中的500个数据并随机打乱，传入batched reader中，batched reader 每次 yield 64个数据
train_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.mnist.train(), buf_size=500),
        batch_size=BATCH_SIZE)
# 读取测试集的数据，每次 yield 64个数据
test_reader = paddle.batch(
            paddle.dataset.mnist.test(), batch_size=BATCH_SIZE)

def event_handler(pass_id, batch_id, cost):
    # 打印训练的中间结果，训练轮次，batch数，损失函数
    print("Pass %d, Batch %d, Cost %f" % (pass_id,batch_id, cost))

place=fluid.CPUPlace()
