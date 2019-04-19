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
            im = im / 255.0
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
    return prediction, conv_pool_0


net, conv0 = convolutional_neural_network(x)  # 官方的CNN
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

trainNum = 3
for i in range(trainNum):
    for batch_id, data in enumerate(batch_reader()):
        outs = exe.run(
            feed=feeder.feed(data),
            fetch_list=[label, avg_cost, conv0])  # feed为数据表 输入数据和标签数据

        # 打印输出面板
        trainTag.add_record(i, outs[1])
        # print(str(i + 1) + "次训练后损失值为：" + str(outs[1]))

        if i == 0:
            # 图像拉伸
            for ii in range(5):
                # 获取图像
                pic0 = (outs[2][ii][0] * 255).astype(int)  # 取第一张图片第一个组的滤波器过滤数据
                picMax = np.max(pic0)  # 求最值
                picMin = np.min(pic0)
                # 获取颜色拉伸后图像
                pic1 = pic0 // (picMax - picMin) * 255  # 整除会向下取整
                '''
                #Debug
                picMax = np.max(pic)  # 求最值
                picMin = np.min(pic)
                print(str(picMax)+"-"+ str(picMin))
                '''
                # 将滤波器过虑数据转成PIL
                pic = Image.fromarray(pic1.reshape(pic1.shape[-2], pic1.shape[-1]).astype('uint8'))
                pic.show()


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
