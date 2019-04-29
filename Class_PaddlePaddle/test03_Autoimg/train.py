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
shape = [30, 15]

with open(path + "data/ocrData.txt", 'rt') as f:
    a = f.read()


def dataReader():
    def redaer():
        for i in range(1, 1501):
            im = Image.open(path + "data/" + str(i) + ".jpg").convert('L')
            im_x = np.array(im).reshape(shape).astype(np.float32)
            im_x = im_x / 255.0 * 2.0 - 1.0
            '''
            自解码
            
            im = im.resize((shape[2] // 3, shape[1] // 3), Image.ANTIALIAS)
            im = im.resize((shape[2], shape[1]), Image.ANTIALIAS)

            im_y = np.array(im).reshape(shape).astype(np.float32)
            im_y = im_y / 255.0 * 2.0 - 1.0
            
            '''
            labelInfo = a[i - 1]
            yield im_x, labelInfo  # 返回一个的话竟然会报错，好像是拆分了一个 啊啊啊！

    return redaer


# 定义网络
x = fluid.layers.data(name="x", shape=shape, dtype=datatype)

y = fluid.layers.data(name="y", shape=[1], dtype=datatype)
label = fluid.layers.data(name="label", shape=[1], dtype=datatype)


def net(input):
    """
    自解码网络

    :param input: 图像张量
    :return: 自解码后数据，原图数据
    """

    img0 = fluid.layers.fc(input=input, size=shape[0] * shape[1], act="tanh")

    img = fluid.layers.fc(input=img0, size=128, act="tanh")

    img = fluid.layers.fc(input=img, size=64, act="tanh")

    img = fluid.layers.fc(input=img, size=32, act="tanh")

    pltdata = fluid.layers.fc(img, size=3, act="softmax")  # 输出 XYZ坐标

    img = fluid.layers.fc(input=pltdata, size=32, act="tanh")

    img = fluid.layers.fc(input=img, size=64, act="tanh")

    img = fluid.layers.fc(input=img, size=128, act="tanh")

    img = fluid.layers.fc(input=img, size=shape[0] * shape[1], act="sigmoid")

    img = fluid.layers.reshape(x=img, shape=[-1, shape[0], shape[1]])



    return img,pltdata


# 获取网络后数据
net_x,pltdata = net(x)
net_y=x
# 定义损失函数 此处应设置为软标签
# cost = fluid.layers.cross_entropy(input=net_x, label=net_y, soft_label=True)
cost = fluid.layers.square_error_cost(input=net_x, label=net_y)
avg_cost = fluid.layers.mean(cost)
# 定义优化方法
sgd_optimizer = fluid.optimizer.Adam(learning_rate=0.01)
sgd_optimizer.minimize(avg_cost)

# 数据传入设置
batch_reader = paddle.batch(reader=dataReader(), batch_size=1)
# batch_reader = paddle.batch(mnist.train(), batch_size=128)
feeder = fluid.DataFeeder(place=place, feed_list=[x, label])  # V1.4版本 不可以只传入一个数据
prog = fluid.default_startup_program()
exe.run(prog)

trainNum = 50
for i in range(trainNum):
    outs = []
    for batch_id, data in enumerate(batch_reader()):

        outs = exe.run(
            feed=feeder.feed(data),
            fetch_list=[avg_cost, pltdata])  # feed为数据表 输入数据和标签数据

    print(str(i + 1) + "次训练后损失值为：" + str(outs))

    if i % 10 == 0:
        # 绘图
        fig = plt.figure()
        ax = Axes3D(fig)
        X_MIT = []
        Y_MIT = []
        Z_MIT = []
        Value_MIT = []
        for ii in range(100):
            X_MIT.append(outs[1][ii][0])
            Y_MIT.append(outs[1][ii][1])
            Z_MIT.append(outs[1][ii][2])
            Value_MIT.append(a[ii])  # label数据
        X_MIT = np.array(X_MIT)
        Y_MIT = np.array(Y_MIT)
        Z_MIT = np.array(Z_MIT)
        Value_MIT = np.array(Value_MIT)
        for x, y, z, s in zip(X_MIT, Y_MIT, Z_MIT, Value_MIT):
            c = cm.rainbow(int(255 * int(s) / 9))  # 上色
            ax.text(x, y, z, s, backgroundcolor=c)  # 标位子
        ax.set_zlim(-1, 1)
        ax.set_xlim(-1, 0)
        ax.set_ylim(-1, 0)
        plt.show()
        pross = float(i) / trainNum
        print("当前训练进度百分比为：" + str(pross * 100)[:3].strip(".") + "%")

# 保存预测模型
path = params_dirname

fluid.io.save_inference_model(params_dirname, ['x'], [net_x], exe)

print(params_dirname)
