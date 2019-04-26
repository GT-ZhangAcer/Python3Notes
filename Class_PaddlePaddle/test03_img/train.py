import paddle.fluid as fluid
import paddle
import numpy as np
from PIL import Image
import Class_OS.o1_获得当前工作目录
import paddle.dataset.mnist as mnist

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
            '''
            自解码
            
            im = im.resize((shape[2] // 3, shape[1] // 3), Image.ANTIALIAS)
            im = im.resize((shape[2], shape[1]), Image.ANTIALIAS)

            im_y = np.array(im).reshape(shape).astype(np.float32)
            im_y = im_y / 255.0 * 2.0 - 1.0
            
            '''

            yield im_x,1

    return redaer


# 定义网络
x = fluid.layers.data(name="x", shape=shape, dtype=datatype)


y = fluid.layers.data(name="y", shape=[1], dtype=datatype)


def net(input):
    """
    自解码网络
    :param input: 图像张量
    :return: 图像张量
    """
    img0 = fluid.layers.fc(input=input, size=shape[0] * shape[1] * shape[2])
    bn1 = fluid.layers.batch_norm(input=img0, name='bn1')
    img = fluid.layers.fc(input=bn1, size=(shape[0] * shape[1] * shape[2]) // 27)
    bn2 = fluid.layers.batch_norm(input=img, name='bn2')
    img = fluid.layers.fc(input=bn2, size=shape[0] * shape[1] * shape[2], act="relu")

    return img, img0


net_x, net_y = net(x)

# 定义损失函数
cost = fluid.layers.cross_entropy(input=net_x, label=net_y, soft_label=True)
avg_cost = fluid.layers.mean(cost)

# 定义优化方法
sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.01)
sgd_optimizer.minimize(avg_cost)
# 数据传入设置
batch_reader = paddle.batch(reader=dataReader(), batch_size=1024)
# batch_reader = paddle.batch(mnist.train(), batch_size=128)
feeder = fluid.DataFeeder(place=place, feed_list=[x,y])
prog = fluid.default_startup_program()
exe.run(prog)

trainNum = 10
for i in range(trainNum):
    for batch_id, data in enumerate(batch_reader()):
        outs = exe.run(
            feed=feeder.feed(data),
            fetch_list=[avg_cost])  # feed为数据表 输入数据和标签数据

        print(str(i + 1) + "次训练后损失值为：" + str(outs))

    pross = float(i) / trainNum
    print("当前训练进度百分比为：" + str(pross * 100)[:3].strip(".") + "%")

# 保存预测模型
path = params_dirname

fluid.io.save_inference_model(params_dirname, ['x'], [net_x], exe)

print(params_dirname)
