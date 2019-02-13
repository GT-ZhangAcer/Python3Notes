#加载库
import paddle.fluid as fluid
import numpy
#定义数据
train_data=numpy.array([[1.0],[2.0],[3.5],[4.0],[5.0]]).astype('float32')
y_true = numpy.array([[1],[1],[0],[0],[0]]).astype('float32')
#定义网络
x = fluid.layers.data(name="x",shape=[1],dtype='float32')
y = fluid.layers.data(name="y",shape=[1],dtype='float32')
y_predict = fluid.layers.fc(input=x,size=1,act=None)
#定义损失函数
cost = fluid.layers.square_error_cost(input=y_predict,label=y)
avg_cost = fluid.layers.mean(cost)
#定义优化方法
sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.01)
sgd_optimizer.minimize(avg_cost)
#参数初始化
cpu = fluid.core.CPUPlace()
exe = fluid.Executor(cpu)
exe.run(fluid.default_startup_program())
##开始训练，迭代100次
params_dirname = "fit_a_line.inference.model"
for i in range(50):
    outs = exe.run(
        feed={'x':train_data,'y':y_true},
        fetch_list=[y_predict.name,avg_cost.name])
    print("正在训练第"+str(i+1)+"次")
#观察结果
    print(outs)
    fluid.io.save_inference_model(params_dirname, ['x'],[y_predict], exe)

print(params_dirname)