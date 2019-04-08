import paddle.fluid as fluid
import paddle
import numpy as np
from PIL import Image
import Class_OS.o1_获得当前工作目录
import linecache  # 读取指定行
import os
import visualdl

# 指定路径
path = Class_OS.o1_获得当前工作目录.main()
params_dirname = path + "test02.inference.model"
print("训练后文件夹路径" + params_dirname)
# 参数初始化
place = fluid.CUDAPlace(0)
# place=fluid.CPUPlace()
exe = fluid.Executor(place)

# visualdl
logw = visualdl.LogWriter("g:/log/main_log", sync_cycle=10)
# Scalar-损失指标
with logw.mode('train') as logger:
    trainTag = logger.scalar("损失指标")
with logw.mode('test') as logger:
    testTag = logger.scalar("损失指标")
# Image-卷积图片
with logw.mode("train") as writer:
    conv_image = writer.image("转换后图片", 3, 1)  # 每一次训练仅展示3张图片
    input_image = writer.image("input_image", 3, 1)

# visualDL --logdir g:/log/main_log --port 8080 --host 127.0.0.10

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
            '''
            img = paddle.dataset.image.load_image(path + "data/" + str(i+1) + ".jpg")'''
            labelInfo = a[i - 1]
            yield im, labelInfo

    return redaer


def testReader():
    def redaer():
        for i in range(1501, 1951):
            im = Image.open(path + "data/" + str(i) + ".jpg").convert('L')
            im = np.array(im).reshape(1, 30, 15).astype(np.float32)
            im = im / 255.0 * 2.0 - 1.0
            '''
            img = paddle.dataset.image.load_image(path + "data/" + str(i+1) + ".jpg")
            img=np.transpose(img, (2, 0, 1))'''
            labelInfo = a[i - 1]
            yield im, labelInfo

    return redaer


# 定义网络
x = fluid.layers.data(name="x", shape=[1, 30, 15], dtype=datatype)
label = fluid.layers.data(name='label', shape=[1], dtype='int64')

sss=0
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


def multilayer_perceptron(img):
    """
    定义多层感知机分类器：
        含有两个隐藏层（全连接层）的多层感知器
        其中前两个隐藏层的激活函数采用 ReLU，输出层的激活函数用 Softmax

    Return:
        predict_image -- 分类的结果
    """
    # 输入的原始图像数据，大小为28*28*1
    # 第一个全连接层，激活函数为ReLU
    hidden = fluid.layers.fc(input=img, size=200, act='relu')
    # 第二个全连接层，激活函数为ReLU
    hidden = fluid.layers.fc(input=hidden, size=200, act='relu')
    # 以softmax为激活函数的全连接输出层，输出层的大小必须为数字的个数10
    prediction = fluid.layers.fc(input=hidden, size=10, act='softmax')
    return prediction


def convolutional_neural_network(img):
    """
    定义卷积神经网络分类器：
        输入的二维图像，经过两个卷积-池化层，使用以softmax为激活函数的全连接层作为输出层

    Return:
        predict -- 分类的结果
    """
    # 第一个卷积-池化层
    # 使用20个5*5的滤波器，池化大小为2，池化步长为2，激活函数为Relu
    conv_pool_0 = fluid.nets.simple_img_conv_pool(
        input=img,
        filter_size=5,
        num_filters=20,
        pool_size=2,
        pool_stride=2,
        act="relu")
    conv_pool_1 = fluid.layers.batch_norm(conv_pool_0)
    # 第二个卷积-池化层
    # 使用50个5*5的滤波器，池化大小为2，池化步长为2，激活函数为Relu
    conv_pool_2 = fluid.nets.simple_img_conv_pool(
        input=conv_pool_1,
        filter_size=5,
        num_filters=50,
        pool_size=2,
        pool_stride=2,
        act="relu")
    # 以softmax为激活函数的全连接输出层，输出层的大小必须为数字的个数10
    prediction = fluid.layers.fc(input=conv_pool_2, size=10, act='softmax')
    return prediction


#net = cnn(x)  # CNN模型
net = multilayer_perceptron(x)  # 多层感知机
#net=convolutional_neural_network(x)#官方的CNN
# 定义损失函数
cost = fluid.layers.cross_entropy(input=net, label=label)
avg_cost = fluid.layers.mean(cost)
acc = fluid.layers.accuracy(input=net, label=label, k=1)
# 定义优化方法
sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.01)
sgd_optimizer.minimize(avg_cost)
# 数据传入设置
batch_reader = paddle.batch(reader=dataReader(), batch_size=1024)
testb_reader = paddle.batch(reader=testReader(), batch_size=1024)
feeder = fluid.DataFeeder(place=place, feed_list=[x, label])
prog = fluid.default_startup_program()
exe.run(prog)

trainNum = 50
for i in range(trainNum):
    for batch_id, data in enumerate(batch_reader()):
        outs = exe.run(
            feed=feeder.feed(data),
            fetch_list=[label, avg_cost])  # feed为数据表 输入数据和标签数据

        trainTag.add_record(i, outs[1])
        # print(str(i + 1) + "次训练后损失值为：" + str(outs[1]))
        #
    for batch_id, data in enumerate(testb_reader()):
        test_acc, test_cost = exe.run(
            feed=feeder.feed(data),
            fetch_list=[label, avg_cost])  # feed为数据表 输入数据和标签数据
        test_costs = []
        test_costs.append(test_cost[0])
        testcost = (sum(test_costs) / len(test_costs))
        testTag.add_record(i, testcost)
        # print("预测损失为：", testcost, "\n")
    pross = float(i) / trainNum
    print("当前训练进度百分比为：" + str(pross * 100)[:3].strip(".") + "%")

# 保存预测模型
path = params_dirname


def del_file(path):
    for i in os.listdir(path):
        path_file = os.path.join(path, i)
        if os.path.isfile(path_file):
            os.remove(path_file)
        else:
            del_file(path_file)


fluid.io.save_inference_model(params_dirname, ['x'], [net], exe)

print(params_dirname)
