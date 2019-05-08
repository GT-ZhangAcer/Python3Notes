import paddle.fluid as fluid
import paddle
import numpy as np
from PIL import Image
import Class_OS.o1_获得当前工作目录
import paddle.dataset.mnist as mnist
# 绘图
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import random
from Class_PaddlePaddle.test03_Autoimg2.torNN import TorNN

# 参数表
CLASS_NUM = 10  # 类别数目
PRE_IMG_NUM = 5  # 准备单类带标签图片数目
LABEL_DIM = 3  # 分类维度，越高越精确

# 指定路径
path = Class_OS.o1_获得当前工作目录.main()
params_dirname = path + "test02.inference.model"
print("训练后文件夹路径" + params_dirname)
# 参数初始化
place = fluid.CUDAPlace(0)
# place=fluid.CPUPlace()
exe = fluid.Executor(place)

# 加载数据
datatype = 'float32'

with open(path + "data/ocrData.txt", 'rt') as f:
    a = f.read()


def preDataReader():
    def reader():
        for i in range(CLASS_NUM * PRE_IMG_NUM):
            im = Image.open("./OCRData/" + str(i) + ".jpg").convert("L")
            im = np.array(im).reshape(1, 30, 15).astype(np.float32)
            im = im / 255.0 * 2.0 - 1.0
            labelInfo = i // 5
            yield im, labelInfo

    return reader


def dataReader():
    def redaer():
        READ_IMG_NUM = 1024  # 原始图片读取个数
        for i in range(1, READ_IMG_NUM):
            im = Image.open(path + "data/" + str(i) + ".jpg").convert('L')
            im = np.array(im).reshape(1, 30, 15).astype(np.float32)
            labelInfo = a[i - 1]
            yield im, labelInfo  # 返回一个的话竟然会报错，好像是拆分了一个 啊啊啊！

    return redaer


# 定义基准网络模型

def convolutional_neural_network(img):
    # 第一个卷积-池化层
    # 使用20个5*5的滤波器，池化大小为2，池化步长为2，激活函数为Relu
    ipt = fluid.layers.reshape(x=img, shape=[-1, 1, 30, 15])
    conv1 = fluid.layers.conv2d(input=ipt,
                                num_filters=32,
                                filter_size=3,
                                padding=1,
                                stride=1,
                                name='conv1',
                                act='relu')

    pool1 = fluid.layers.pool2d(input=conv1,
                                pool_size=2,
                                pool_stride=2,
                                pool_type='max',
                                name='pool1')

    bn1 = fluid.layers.batch_norm(input=pool1, name='bn1')

    fc1 = fluid.layers.fc(input=bn1, size=1024, act='relu', name='fc1')

    fc2 = fluid.layers.fc(input=fc1, size=10, act='softmax', name='fc2')
    pltdata = fluid.layers.fc(input=fc2, size=3, act=None)
    return fc2, pltdata


# 创建分支程序用于TorNN初始化
torNNBase = fluid.Program()  # 基准元训练
torNNBigMeta = fluid.Program()  # 元训练
startup = fluid.Program()  # 默认启动程序



# torNN基准元训练项目
with fluid.program_guard(main_program=torNNBase, startup_program=startup):
    x = fluid.layers.data(name="x", shape=[1, 30, 15], dtype='float32')
    label = fluid.layers.data(name="label", shape=[1], dtype="int64")
    net_x_Base, _ = convolutional_neural_network(x)  # 获取网络
    # 定义损失函数
    cost_Base = fluid.layers.cross_entropy(input=net_x_Base, label=label)
    avg_cost_Base = fluid.layers.mean(cost_Base)
    acc = fluid.layers.accuracy(input=net_x_Base, label=label, k=1)
    # 定义优化方法
    sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.01)
    sgd_optimizer.minimize(avg_cost_Base)

# 数据传入设置

# 人工标签传入
prebatch_reader = paddle.batch(
    reader=preDataReader(),
    batch_size=512)
prefeeder = fluid.DataFeeder(place=place, feed_list=[x, label])
# 原始数据传入
batch_reader = paddle.batch(
    reader=dataReader(),
    batch_size=512)
feeder = fluid.DataFeeder(place=place, feed_list=[x, label])

exe.run(startup)

testdata=[]
# 预训练-TorNNBase
for batch_id, data in enumerate(prebatch_reader()):
    outs = exe.run(program=torNNBase,
                   feed=prefeeder.feed(data),
                   fetch_list=[label,net_x_Base, avg_cost_Base])
    label_data,net_x_Base_data, avg_cost_Base_data=outs
    for i in range(len(label_data)):
        testdata.append([label_data[i],net_x_Base_data[i]])
pass