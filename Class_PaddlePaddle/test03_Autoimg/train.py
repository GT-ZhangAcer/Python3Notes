import paddle.fluid as fluid
import paddle
import numpy as np
from PIL import Image
import Class_OS.o1_获得当前工作目录

# 绘图
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import random

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
shape = [1, 30, 15]

with open(path + "data/ocrData.txt", 'rt') as f:
    a = f.read()


def trainReader():
    def redaer():
        for i in range(1, 1501):
            im = Image.open(path + "data/" + str(i) + ".jpg").convert('L')
            im = np.array(im).reshape(shape).astype(np.float32)

            labelInfo = a[i + 1]
            yield im, im, labelInfo  # 返回一个的话竟然会报错，好像是拆分了一个 啊啊啊！

    return redaer


def testReader():
    def redaer():
        for i in range(1500, 2000):
            im = Image.open(path + "data/" + str(i) + ".jpg").convert('L')
            im = np.array(im).reshape(shape).astype(np.float32)
            labelInfo = a[i + 1]
            yield im, labelInfo

    return redaer


# 定义网络
x = fluid.layers.data(name="x", shape=shape, dtype=datatype)
y = fluid.layers.data(name="y", shape=shape, dtype=datatype)
label = fluid.layers.data(name="label", shape=[1], dtype="int64")


def net(input):
    img = fluid.layers.fc(input=input, size=128, act="relu")
    img = fluid.layers.fc(input=img, size=64, act="relu")
    img = fluid.layers.fc(input=img, size=32, act="relu")
    pltdata = fluid.layers.fc(img, size=3, act="relu")
    img = fluid.layers.fc(input=pltdata, size=32, act="relu")
    img = fluid.layers.fc(input=img, size=64, act="relu")
    img = fluid.layers.fc(input=img, size=128, act="relu")
    img = fluid.layers.fc(input=img, size=shape[2] * shape[1], act="relu")
    img = fluid.layers.reshape(x=img, shape=[-1, shape[1], shape[2]])

    return img, pltdata


net_x, pltdata = net(x)
net_y = fluid.layers.reshape(x=y, shape=[-1, shape[1], shape[2]])

# 定义损失函数
cost = fluid.layers.square_error_cost(input=net_x, label=net_y)
# cost= fluid.layers.smooth_l1(x=net_x,y=net_y)
avg_cost = fluid.layers.reduce_mean(cost)
# 定义优化方法
adam_optimizer = fluid.optimizer.Adam(learning_rate=0.01)
adam_optimizer.minimize(avg_cost)

# 数据传入设置
batch_reader = paddle.batch(
    reader=trainReader(),
    batch_size=2000)

feeder = fluid.DataFeeder(place=place, feed_list=[x, y, label])
prog = fluid.default_startup_program()
exe.run(prog)

trainNum = 1000

for i in range(trainNum):
    outc = 0
    for batch_id, data in enumerate(batch_reader()):
        outs = exe.run(
            feed=feeder.feed(data),
            fetch_list=[avg_cost, pltdata, label])  # feed为数据表 输入数据和标签数据

        outc = outs[0]
        print(str(i + 1) + "次训练后损失值为：" + str(outc))

    if outc <= 5:
        break
    # 绘图-3D
    fig = plt.figure()
    ax = Axes3D(fig)
    X_MIT = []
    Y_MIT = []
    Z_MIT = []
    Value_MIT = []
    for ii in range(200):
        X_MIT.append(outs[1][ii][0])
        Y_MIT.append(outs[1][ii][1])
        Z_MIT.append(outs[1][ii][2])
        Value_MIT.append(outs[2][ii][0])  # label数据
    X_MIT = np.array(X_MIT)
    Y_MIT = np.array(Y_MIT)
    Z_MIT = np.array(Z_MIT)
    Value_MIT = np.array(Value_MIT)
    for x, y, z, s in zip(X_MIT, Y_MIT, Z_MIT, Value_MIT):
        c = cm.rainbow(int(255 * int(s) / 9))  # 上色
        ax.text(x, y, z, s, backgroundcolor=c)  # 标位子
    ax.set_xlim(X_MIT.min(), X_MIT.max())
    ax.set_ylim(Y_MIT.min(), Y_MIT.max())
    ax.set_zlim(Z_MIT.min(), Z_MIT.max())
    plt.show()
    pross = float(i) / trainNum
    print("当前训练进度百分比为：" + str(pross * 100)[:3].strip(".") + "%")

# 保存预测模型


fluid.io.save_inference_model(params_dirname, ['x'], [net_x], exe)

print(params_dirname)
