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
        READ_IMG_NUM=256 #原始图片读取个数
        IMG_RANDOM_NUM=10#产出图片个数

        for i in range(1,READ_IMG_NUM):
            im = Image.open(path + "data/" + str(i) + ".jpg").convert('1')
            im_y = np.array(im).reshape(shape).astype(np.float32)
            #im_x = im_x / 255.0 * 2.0 - 1.0

            #自解码
            for ii in range(1,IMG_RANDOM_NUM):

                k=random.uniform(1.0,3.0)
                imx = im.resize((int(shape[1] / k), int(shape[0] / k)), Image.ANTIALIAS)
                imx = imx.resize((shape[1], shape[0]), Image.ANTIALIAS)

                im_x = np.array(imx).reshape(shape).astype(np.float32)
                #im_y = im_y / 255.0 * 2.0 - 1.0


                labelInfo = a[i - 1]
                yield im_x, im_y,labelInfo  # 返回一个的话竟然会报错，好像是拆分了一个 啊啊啊！

    return redaer


# 定义网络
x = fluid.layers.data(name="x", shape=shape, dtype=datatype)

y = fluid.layers.data(name="y", shape=shape, dtype=datatype)
label = fluid.layers.data(name="label", shape=[1], dtype="int64")

'''
def net(input):
    """
    自解码网络

    :param input: 图像张量
    :return: 自解码后数据，原图数据
    """

    img0 = fluid.layers.fc(input=input, size=shape[0] * shape[1], act=None)

    img = fluid.layers.fc(input=img0, size=128, act="tanh")
    
    img = fluid.layers.fc(input=img, size=64, act="tanh")

    img = fluid.layers.fc(input=img, size=32, act="tanh")
    
    pltdata = fluid.layers.fc(img, size=3, act=None)  # 输出 XYZ坐标
    
    img = fluid.layers.fc(input=pltdata, size=32, act="tanh")

    img = fluid.layers.fc(input=img, size=64, act="tanh")
    
    img = fluid.layers.fc(input=img, size=128, act="tanh")

    img = fluid.layers.fc(input=img, size=shape[0] * shape[1], act="sigmoid")

    img = fluid.layers.reshape(x=img, shape=[-1, shape[0], shape[1]])



    return img,input,pltdata
    
'''
def convolutional_neural_network(img_x,img_y):

    def conv(img):
        # 第一个卷积-池化层
        # 使用20个5*5的滤波器，池化大小为2，池化步长为2，激活函数为Relu
        ipt = fluid.layers.reshape(x=img, shape=[-1,1, shape[0], shape[1]])
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

        #fc2 = fluid.layers.fc(input=fc1, size=10, act='softmax', name='fc2')
        return fc1

    prediction_x=conv(img_x)
    prediction_y = conv(img_y)

    pltdata=fluid.layers.fc(input=prediction_x,size=3,act=None)
    return prediction_x,prediction_y,pltdata


# 获取网络后数据
#net_x,net_y,pltdata = net(x)
net_x,net_y,pltdata=convolutional_neural_network(x,y)

# 定义损失函数 此处应设置为软标签
cost = fluid.layers.cross_entropy(input=net_x, label=net_x, soft_label=True)
# cost = fluid.layers.square_error_cost(input=net_x, label=net_y)
cost=fluid.layers.abs(cost)
avg_cost = fluid.layers.mean(cost)
# 定义优化方法
sgd_optimizer = fluid.optimizer.Adam(learning_rate=0.005)
sgd_optimizer.minimize(avg_cost)


# 数据传入设置
batch_reader = paddle.batch(
    reader=paddle.reader.shuffle(reader=dataReader(),buf_size=3000),
    batch_size=512)
# batch_reader = paddle.batch(mnist.train(), batch_size=128)
feeder = fluid.DataFeeder(place=place, feed_list=[x, y,label])  # V1.4版本 不可以只传入一个数据
prog = fluid.default_startup_program()
exe.run(prog)

trainNum = 200
for i in range(trainNum):
    outs = []
    for batch_id, data in enumerate(batch_reader()):

        outs = exe.run(
            feed=feeder.feed(data),
            fetch_list=[avg_cost, pltdata,label])  # feed为数据表 输入数据和标签数据

    print(str(i + 1) + "次训练后损失值为：" + str(outs[0]))


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
path = params_dirname

fluid.io.save_inference_model(params_dirname, ['x'], [net_x], exe)

print(params_dirname)
