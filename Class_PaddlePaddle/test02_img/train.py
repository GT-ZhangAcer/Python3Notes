#加载库
import paddle.fluid as fluid
import numpy
import Class_OS.o1_获得当前工作目录

#指定路径
path=Class_OS.o1_获得当前工作目录.main()
params_dirname = path+"test01.inference.model"
print("训练后文件夹路径"+params_dirname)
#参数初始化
cpu = fluid.CPUPlace()
exe = fluid.Executor(cpu)

# 卷积神经网络
def convolutional_neural_network(input):
    # 第一个卷积层，卷积核大小为3*3，一共有32个卷积核
    conv1 = fluid.layers.conv2d(input=input,
                                num_filters=32,
                                filter_size=3,
                                stride=1)

    # 第一个池化层，池化大小为2*2，步长为1，最大池化
    pool1 = fluid.layers.pool2d(input=conv1,
                                pool_size=2,
                                pool_stride=1,
                                pool_type='max')

    # 第二个卷积层，卷积核大小为3*3，一共有64个卷积核
    conv2 = fluid.layers.conv2d(input=pool1,
                                num_filters=64,
                                filter_size=3,
                                stride=1)

    # 第二个池化层，池化大小为2*2，步长为1，最大池化
    pool2 = fluid.layers.pool2d(input=conv2,
                                pool_size=2,
                                pool_stride=1,
                                pool_type='max')

    # 以softmax为激活函数的全连接输出层，大小为label大小
    fc = fluid.layers.fc(input=pool2, size=10, act='softmax')
    return fc
