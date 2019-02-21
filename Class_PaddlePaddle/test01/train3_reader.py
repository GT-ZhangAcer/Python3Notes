import paddle.fluid as fluid
import paddle
import numpy as np
from PIL import Image
import os
import Class_OS.o1_获得当前工作目录
import linecache#读取指定行

# 指定路径
path = Class_OS.o1_获得当前工作目录.main()
params_dirname = path + "test02.inference.model"
print("训练后文件夹路径" + params_dirname)
# 参数初始化
cpu = fluid.CPUPlace()
exe = fluid.Executor(cpu)

datatype='float32'
#加载数据
def dataReader():
    def redaer():
        for i in range(5):
            x=i
            y=3*i+4
            yield x,y
    return redaer

#定义网络
x = fluid.layers.data(name="x",shape=[1],dtype=datatype)
y = fluid.layers.data(name="y",shape=[1],dtype=datatype)
y_predict = fluid.layers.fc(input=x,size=1,act=None)#定义x与其有关系
#定义损失函数
cost = fluid.layers.square_error_cost(input=y_predict,label=y)
avg_cost = fluid.layers.mean(cost)
#定义优化方法
sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.01)
sgd_optimizer.minimize(avg_cost)
#数据传入设置
batch_reader = paddle.batch(reader=dataReader(), batch_size=5)
feeder = fluid.DataFeeder(place=cpu, feed_list=[x, y])
prog=fluid.default_startup_program()
exe.run(prog)


for batch_id,data in enumerate(batch_reader()):
    print(batch_id)
    print(data)
    outs = exe.run(
        feed=feeder.feed(data),
        fetch_list=[y_predict.name,avg_cost.name])#feed为数据表 输入数据和标签数据


#保存预测模型
fluid.io.save_inference_model(params_dirname, ['x'],[y_predict], exe)

print(params_dirname)
