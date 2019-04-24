import paddle.fluid as fluid
import paddle
import numpy as np
from PIL import Image
import Class_OS.o1_获得当前工作目录

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
shape = [3, 30, 15]


def dataReader():
    def redaer():
        for i in range(1, 1501):
            im = Image.open(path + "data/" + str(i) + ".jpg")
            im_x = np.array(im).reshape(shape).astype(np.float32)
            im_x = im_x / 255.0 * 2.0 - 1.0
            yield im_x

    return redaer


# 定义网络
x = fluid.layers.data(name="x", shape=shape, dtype=datatype)
y = fluid.layers.data(name="y", shape=shape, dtype=datatype)


def net(input):
    """
    自解码网络
    等比例使用双线性内插扩大倍数缩小倍数
    :param input: 图像张量
    :return: 图像张量
    """
    img_y = fluid.layers.fill_constant_batch_size_like(input=input, shape=[1,3,30, 15], value=0, dtype=datatype)
    img = fluid.layers.resize_bilinear(input, out_shape=[shape[1] // 4, shape[2] // 4])
    img = fluid.layers.resize_bilinear(img, out_shape=[shape[1] * 4, shape[2] * 4])
    img = fluid.layers.resize_bilinear(img, out_shape=[shape[1] // 4, shape[2] // 4])
    img = fluid.layers.resize_bilinear(img, out_shape=[shape[1] * 4, shape[2] * 4])
    # img=fluid.layers.fc(input=img, size=10, act=None)
    return img, img_y


net_x, net_y = net(x)

# 定义损失函数
cost = fluid.layers.cross_entropy(input=net_x, label=net_y, soft_label=True)
avg_cost = fluid.layers.mean(cost)
#acc = fluid.layers.accuracy(input=net_x, label=net_y, k=1)
# 定义优化方法
sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.01)
sgd_optimizer.minimize(avg_cost)
# 数据传入设置
batch_reader = paddle.batch(reader=dataReader(), batch_size=1024)
feeder = fluid.DataFeeder(place=place, feed_list=[x])
prog = fluid.default_startup_program()
exe.run(prog)

trainNum = 50
for i in range(trainNum):
    for batch_id, data in enumerate(batch_reader()):
        outs = exe.run(
            feed=feeder.feed(data),
            fetch_list=[avg_cost])  # feed为数据表 输入数据和标签数据

        print(str(i + 1) + "次训练后损失值为：" + str(outs[1]))

    pross = float(i) / trainNum
    print("当前训练进度百分比为：" + str(pross * 100)[:3].strip(".") + "%")

# 保存预测模型
path = params_dirname

fluid.io.save_inference_model(params_dirname, ['x'], [net_x], exe)

print(params_dirname)
