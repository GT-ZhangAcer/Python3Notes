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
            im = Image.open(path + "data/" + str(i) + ".jpg").convert("L")
            im = np.array(im).reshape(1, 30, 15).astype(np.float32)
            im = im / 255.0 * 2.0 - 1.0
            yield im, -1  # 返回一个的话竟然会报错，好像是拆分了一个 啊啊啊！

    return redaer


def dataReader2(load_list):
    def redaer():
        for i in load_list:
            im = Image.open(path + "data/" + str(i[0] + 1) + ".jpg").convert('L')
            im = np.array(im).reshape(1, 30, 15).astype(np.float32)
            label = i[1]
            yield im, label

    return redaer


# 定义基准网络模型

def convolutional_neural_network(img,name):
    # 第一个卷积-池化层
    # 使用20个5*5的滤波器，池化大小为2，池化步长为2，激活函数为Relu
    with fluid.unique_name.guard(name + "/"):
        ipt = fluid.layers.reshape(x=img, shape=[-1, 1, 30, 15])
        conv1 = fluid.layers.conv2d(input=ipt,
                                    num_filters=32,
                                    filter_size=3,
                                    padding=1,
                                    stride=1,
                                    act='relu')

        pool1 = fluid.layers.pool2d(input=conv1,
                                    pool_size=2,
                                    pool_stride=2,
                                    pool_type='max')

        bn1 = fluid.layers.batch_norm(input=pool1)

        fc2 = fluid.layers.fc(input=bn1, size=10, act='relu')
        pltdata = fluid.layers.fc(input=fc2, size=3, act=None)
        return fc2, pltdata

def convolutional_neural_network2(img,name):
    # 第一个卷积-池化层
    # 使用20个5*5的滤波器，池化大小为2，池化步长为2，激活函数为Relu
    with fluid.unique_name.guard(name + "/"):
        ipt2 = fluid.layers.reshape(x=img, shape=[-1, 1, 30, 15])
        conv21 = fluid.layers.conv2d(input=ipt2,
                                    num_filters=32,
                                    filter_size=3,
                                    padding=1,
                                    stride=1,
                                    act='relu')

        pool21 = fluid.layers.pool2d(input=conv21,
                                    pool_size=2,
                                    pool_stride=2,
                                    pool_type='max')

        bn21 = fluid.layers.batch_norm(input=pool21)

        fc22 = fluid.layers.fc(input=bn21, size=10, act='relu')

        pltdata2 = fluid.layers.fc(input=fc22, size=3, act=None)
        return fc22, pltdata2

# 创建分支程序用于TorNN初始化
torNNBase = fluid.Program()  # 基准元训练
final_program = fluid.Program()
startup = fluid.Program()  # 默认启动程序

# torNN基准元训练项目
with fluid.program_guard(main_program=torNNBase, startup_program=startup):
    x = fluid.layers.data(name="x", shape=[1, 30, 15], dtype='float32')
    label = fluid.layers.data(name="label", shape=[1], dtype="int64")
    net_x_Base, _ = convolutional_neural_network(x,"torNNBase")  # 获取网络
    # 定义损失函数
    cost_Base = fluid.layers.cross_entropy(input=net_x_Base, label=label)
    avg_cost_Base = fluid.layers.mean(cost_Base)
    torNNBaseT = torNNBase.clone(for_test=True)
    # 定义优化方法
    sgd_optimizer = fluid.optimizer.Adam(learning_rate=0.01)
    sgd_optimizer.minimize(avg_cost_Base)

with fluid.program_guard(main_program=final_program, startup_program=startup):
    x_f = fluid.layers.data(name="x_f", shape=[1, 30, 15], dtype='float32')
    label_f = fluid.layers.data(name="label_f", shape=[1], dtype="int64")
    net_x_Base_f, pltdata = convolutional_neural_network2(x_f,"final_program")  # 获取网络
    # 定义损失函数
    cost_Base_f = fluid.layers.cross_entropy(input=net_x_Base_f, label=label_f)
    avg_cost_Base_f = fluid.layers.mean(cost_Base_f)
    final_programT = final_program.clone(for_test=True)
    # 定义优化方法
    sgd_optimizer_f = fluid.optimizer.Adam(learning_rate=0.01)
    sgd_optimizer_f.minimize(avg_cost_Base_f)
# 数据传入设置

# 人工标签传入
prebatch_reader = paddle.batch(
    reader=preDataReader(),
    batch_size=512)
prefeeder = fluid.DataFeeder(place=place, feed_list=[x, label])
batch_reader = paddle.batch(
    reader=dataReader(),
    batch_size=2048)
feeder = fluid.DataFeeder(place=place, feed_list=[x, label])

exe.run(startup)

# 预训练-TorNNBase
TRAINNUM = 10

# 初始化分类列表

basedata = []
oridata = []
for i in range(CLASS_NUM):
    basedata.append([])

for train_num,i in enumerate(range(TRAINNUM)):
    '''
    basedata = []
    oridata = []
    for i in range(CLASS_NUM):
        basedata.append([])
    '''
    for batch_id, data in enumerate(prebatch_reader()):
        # 获取训练数据
        outs = exe.run(program=torNNBase,
                       feed=prefeeder.feed(data),
                       fetch_list=[label, net_x_Base, avg_cost_Base])
        # 按格式记录数据
        label_data, net_x_Base_data, avg_cost_Base_data = outs
        for i in range(len(label_data)):
            index = label_data[i].tolist()[0]
            basedata[index].append(net_x_Base_data[i].tolist())


for batch_id, data in enumerate(batch_reader()):
    # 获取训练数据
    outs = exe.run(program=torNNBaseT,
                   feed=feeder.feed(data),
                   fetch_list=[net_x_Base])
    oridata = outs[0].tolist()

baseTor = TorNN(oridata, basedata)
baseTorT, _ = baseTor.classsify(expansion_rate=1,debug=True,classify_True_rate=2)
baseTorTrueNum = 0
final_id_list = []
for i in baseTorT:
    labeli = a[i[1]]
    final_id_list.append([i[1], i[0]])
    if str(labeli) == str(i[0]):
        baseTorTrueNum += 1
        #print("True")
    #else:
        #print("False")

print("True rate:",str(baseTorTrueNum / len(baseTorT) * 100)[:4])


# 原始数据传入
batch_reader2 = paddle.batch(
    reader=dataReader2(final_id_list),
    batch_size=1024)
feeder = fluid.DataFeeder(place=place, feed_list=[x_f, label_f])

TRAINNUM = 10
basedata = []
oridata = []
for i in range(CLASS_NUM):
    basedata.append([])
for i in range(TRAINNUM):
    for batch_id, data in enumerate(batch_reader2()):
        # 获取训练数据
        outs = exe.run(program=final_program,
                       feed=feeder.feed(data),
                       fetch_list=[label_f, net_x_Base_f])
        if i == TRAINNUM-1:
            # 按格式记录数据
            label_data, net_x_Base_data = outs
            for i in range(len(label_data)):
                index = label_data[i].tolist()[0]
                basedata[index].append(net_x_Base_data[i].tolist())


for batch_id, data in enumerate(batch_reader()):
    # 获取训练数据
    outs = exe.run(program=final_program,
                   feed=feeder.feed(data),
                   fetch_list=[net_x_Base_f,pltdata])
    oridata = outs[0].tolist()

baseTor2 = TorNN(oridata, basedata)
baseTorT, _ = baseTor2.classsify(expansion_rate=1, debug=True)
baseTorTrueNum = 0
final_id_list = []
for i in baseTorT:
    labeli = a[i[1]]
    final_id_list.append([i[1], a[i[1]]])
    if str(labeli) == str(i[0]):
        baseTorTrueNum += 1
        print("True")
    else:
        print("False")
print("True rate2:", str(baseTorTrueNum / len(baseTorT) * 100)[:4])



