# 加载库
import paddle.fluid as fluid
import numpy
import Class_OS.o1_获得当前工作目录

fluid.install_check.run_check()
# 指定路径
path = Class_OS.o1_获得当前工作目录.main()
params_dirname = path + "test01.inference.model/"
print("训练后文件夹路径" + params_dirname)
# 参数初始化
gpu = fluid.CUDAPlace(0)
exe = fluid.Executor(gpu)

# 定义数据
train_data = [[0], [1], [2], [3], [4], [5], [10]]
y_true = [[3], [13], [23], [33], [43], [53], [103]]

# 定义网络
x = fluid.layers.data(name="x", shape=[1], dtype="float32")
y = fluid.layers.data(name="y", shape=[1], dtype="float32")
y_predict = fluid.layers.fc(input=x, size=1, act=None)  # 定义x与其有关系
# 定义损失函数
cost = fluid.layers.square_error_cost(input=y_predict, label=y)
avg_cost = fluid.layers.mean(cost)
# 定义优化方法
sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.01)
sgd_optimizer.minimize(avg_cost)

# 开始训练，迭代100次
prog = fluid.default_startup_program()
exe.run(prog)

for i in range(50):
    for data_id in range(len(y_true)):
        data_x = numpy.array(train_data[data_id]).astype("float32").reshape((1, 1))
        data_y = numpy.array(y_true[data_id]).astype("float32").reshape((1, 1))
        outs = exe.run(
            feed={'x': data_x, 'y': data_y},
            fetch_list=[y_predict.name, avg_cost])  # feed为数据表 输入数据和标签数据
        print("正在训练第" + str(i + 1) + "次")
        # 观察结果
        print(outs)

data_x = numpy.array([[input("请输入数值")]]).astype("float32").reshape((1, 1))
data_y = numpy.array(0).astype("float32").reshape((1, 1))
outs = exe.run(
            feed={'x': data_x, 'y': data_y},
            fetch_list=[y_predict.name])  # feed为数据表 输入数据和标签数据
print(outs)