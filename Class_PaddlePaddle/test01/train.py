#加载库
import paddle.fluid as fluid
import numpy
import Class_OS.o1_获得当前工作目录

#指定路径
path=Class_OS.o1_获得当前工作目录.main()
params_dirname = path+"test01.inference.model"
print("训练后文件夹路径"+params_dirname)
#参数初始化
gpu = fluid.CUDAPlace(0)
exe = fluid.Executor(gpu)


#定义数据
datatype="float32"
train_data=numpy.array([[0],[1],[2],[3],[4],[5],[10]]).astype(datatype)#10倍缩放 此处数据类型尽可能与网格类型相似
y_true = numpy.array([[3],[13],[23],[33],[43],[53],[103]]).astype(datatype)


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

##开始训练，迭代100次
prog=fluid.default_startup_program()
exe.run(prog)

for i in range(5000):
    outs = exe.run(
        feed={'x':train_data,'y':y_true},
        fetch_list=[y_predict.name,avg_cost])#feed为数据表 输入数据和标签数据
    print("正在训练第"+str(i+1)+"次")
#观察结果
    print(outs)
#保存预测模型
fluid.io.save_inference_model(params_dirname, ['x'],[y_predict], exe)

print(params_dirname)