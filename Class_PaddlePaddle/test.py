import paddle.fluid as fluid
a = fluid.layers.data(name="a",shape=[1],dtype='float32')
b = fluid.layers.data(name="b",shape=[1],dtype='float32')

result = fluid.layers.elementwise_add(a,b)

#定义Exector
cpu = fluid.core.CPUPlace() #定义运算场所，这里选择在CPU下训练
exe = fluid.Executor(cpu) #创建执行器
exe.run(fluid.default_startup_program()) #网络参数初始化

#准备数据
import numpy
data_1 = input("a=")
data_2 = input("b=")
x = numpy.array([[data_1]])
y = numpy.array([[data_2]])

#执行计算
outs = exe.run(
feed={'a':x,'b':y},
fetch_list=[result.name])

#验证结果
print(outs)