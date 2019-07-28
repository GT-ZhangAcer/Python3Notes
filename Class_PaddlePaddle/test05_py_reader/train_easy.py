import paddle.fluid as fluid
import paddle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 用来显示中文

path = "./"
params_dirname = path + "test.inference.model"
print("训练后文件夹路径" + params_dirname)
# 参数初始化
place = fluid.CUDAPlace(0)
# place = fluid.CPUPlace()


# 加载数据
datatype = 'float32'
with open(path + "data/ocrData.txt", 'rt') as f:
    a = f.read()


def dataReader():
    def redaer():
        for i in range(1, 1501):
            im = Image.open(path + "data/" + str(i) + ".jpg").convert('L')
            im = np.array(im).reshape(1, 30, 15).astype(np.float32)
            im = im / 255.0 * 2.0 - 1.0

            labelInfo = a[i - 1]
            yield im, labelInfo

    return redaer


# 定义网络
x = fluid.layers.data(name="x", shape=[1, 30, 15], dtype=datatype)
label = fluid.layers.data(name='label', shape=[1], dtype='int64')

# 异步读取

# pyreader = fluid.io.PyReader(feed_list=[x, label], capacity=64)
# 非可迭代读取
pyreader = fluid.io.PyReader(feed_list=[x, label], capacity=4, iterable=False)

pyreader.decorate_sample_list_generator(
    paddle.batch(dataReader(), batch_size=1500),
    place)

'''
# 同步数据传入设置
batch_reader = paddle.batch(reader=dataReader(), batch_size=2048)
feeder = fluid.DataFeeder(place=place, feed_list=[x, label])
'''

def cnn(ipt):
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

    conv2 = fluid.layers.conv2d(input=bn1,
                                num_filters=64,
                                filter_size=3,
                                padding=1,
                                stride=1,
                                name='conv2',
                                act='relu')

    pool2 = fluid.layers.pool2d(input=conv2,
                                pool_size=2,
                                pool_stride=2,
                                pool_type='max',
                                name='pool2')

    bn2 = fluid.layers.batch_norm(input=pool2, name='bn2')

    fc1 = fluid.layers.fc(input=bn2, size=1024, act='relu', name='fc1')

    fc2 = fluid.layers.fc(input=fc1, size=10, act='softmax', name='fc2')

    return fc2


net = cnn(x)  # CNN模型

cost = fluid.layers.cross_entropy(input=net, label=label)
avg_cost = fluid.layers.mean(cost)
acc = fluid.layers.accuracy(input=net, label=label, k=1)
# 定义优化方法
sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.01)
sgd_optimizer.minimize(avg_cost)


exe = fluid.Executor(place)
prog = fluid.default_startup_program()
exe.run(prog)

trainNum = 50
accL = []

import time

'''
# 可迭代读取
starttime = time.time()
for i in range(trainNum):
    for batch_id, data in enumerate(pyreader()):
        outs = exe.run(
            feed=data,
            fetch_list=[label, avg_cost, acc])  # feed为数据表 输入数据和标签数据
        accL.append(outs)
    pross = float(i) / trainNum
    print("当前训练进度百分比为：" + str(pross * 100)[:3].strip(".") + "%")
sumtime = time.time() - starttime
print("Cost Time:", sumtime)
'''

# 不可迭代读取
starttime = time.time()
for i in range(trainNum):
    pyreader.start()
    try:
        while True:
            outs = exe.run(
                fetch_list=[label, avg_cost, acc])  # feed为数据表 输入数据和标签数据
            accL.append(outs[2])
    except fluid.core.EOFException:
        print('End of epoch')
        pyreader.reset()

    pross = float(i) / trainNum
    print("当前训练进度百分比为：" + str(pross * 100)[:3].strip(".") + "%")
sumtime = time.time() - starttime
print("Cost Time:", sumtime)


fluid.io.save_inference_model(params_dirname, ['x'], [net], exe)