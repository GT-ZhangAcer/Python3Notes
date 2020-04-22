import paddle.fluid as fluid
import paddle
import numpy as np

# 参数初始化
cpu = fluid.CPUPlace()
exe = fluid.Executor(cpu)

datatype = 'float32'


# 加载数据

def data_reader():
    def reader():
        for i in range(1, 15):
            x = i
            y = 10 * i
            yield int(x), int(y)

    return reader


# 定义网络
x = fluid.layers.data(name="x", shape=[1], dtype=datatype)
y = fluid.layers.data(name="y", shape=[1], dtype=datatype)
y_predict = fluid.layers.fc(input=x, size=1, act=None)  # 定义x与其有关系
# 定义损失函数
cost = fluid.layers.square_error_cost(input=y_predict, label=y)
avg_cost = fluid.layers.mean(cost)
# 定义优化方法
sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.01)
sgd_optimizer.minimize(avg_cost)
# 数据传入设置
batch_reader = paddle.batch(reader=data_reader(), batch_size=10)
feeder = fluid.DataFeeder(place=cpu, feed_list=[x, y])
prog = fluid.default_startup_program()
exe.run(prog)

for i in range(500):
    for batch_id, data in enumerate(batch_reader()):
        train_cost = exe.run(
            feed=feeder.feed(data),
            fetch_list=[avg_cost, y_predict])  # feed为数据表 输入数据和标签数据
        print(train_cost)

