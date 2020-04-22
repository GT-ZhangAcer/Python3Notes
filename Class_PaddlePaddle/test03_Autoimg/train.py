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
# place = fluid.CUDAPlace(0)
place = fluid.CPUPlace()
exe = fluid.Executor(place)

# 加载数据
datatype = 'float32'
shape = [1, 30, 15]

with open(path + "data/ocrData.txt", 'rt') as f:
    a = f.read()


def generate_reader():
    def reader():
        for index in range(1, 1501):
            im = Image.open(path + "data/" + str(index) + ".jpg").convert('L')
            im = np.array(im).reshape(shape).astype(np.float32)
            label_info = a[index + 1]
            yield im, label_info

    return reader


# 定义网络
x = fluid.layers.data(name="x", shape=shape, dtype=datatype)
label = fluid.layers.data(name="label", shape=[1], dtype="int64")


def net(ipt):
    img = fluid.layers.fc(input=ipt, size=450, act="relu")
    img = fluid.layers.fc(input=img, size=300, act="relu")
    img = fluid.layers.fc(input=img, size=100, act="relu")
    img_o = fluid.layers.fc(input=img, size=10)
    img = fluid.layers.fc(input=img_o, size=100, act="relu")
    img = fluid.layers.fc(input=img, size=300, act="relu")
    img = fluid.layers.fc(input=img, size=450, act="relu")
    out = fluid.layers.softmax(img_o)
    return img, out


net_x, net_out = net(x)
net_y = fluid.layers.reshape(x=x, shape=[-1, 450])

# 定义损失函数
cost = fluid.layers.square_error_cost(input=net_x, label=net_y)
# cost= fluid.layers.smooth_l1(x=net_x,y=net_y)
avg_cost = fluid.layers.reduce_mean(cost)
# 定义优化方法
adam_optimizer = fluid.optimizer.Adam(learning_rate=0.01)
adam_optimizer.minimize(avg_cost)

# 数据传入设置
batch_reader = paddle.batch(
    reader=fluid.io.shuffle(generate_reader(), buf_size=1024),
    batch_size=64)

feeder = fluid.DataFeeder(place=place, feed_list=[x, label])
prog = fluid.default_startup_program()
exe.run(prog)

trainNum = 1000

for i in range(trainNum):

    for batch_id, data in enumerate(batch_reader()):
        outs = exe.run(
            feed=feeder.feed(data),
            fetch_list=[net_out, label])  # feed为数据表 输入数据和标签数据
        dic = dict()
        for lab0, lab1 in zip(outs[0], outs[1]):
            lab0 = np.argsort(lab0)[-1]
            if lab1[0] not in dic:
                dic[int(lab0[0])] = []
            dic[int(lab0[0])].append(lab0)
        print(dic)
        # 保存预测模型

        fluid.io.save_inference_model(params_dirname, ['x'], [net_x], exe)

        print(params_dirname)
