# 加载库
import paddle.fluid as fluid
import numpy as np
import Class_OS.o1_获得当前工作目录
from PIL import Image

# 指定路径
path = Class_OS.o1_获得当前工作目录.main()
params_dirname = path + "test02.inference.model"
print("训练后文件夹路径" + params_dirname)
# 参数初始化
cpu = fluid.CPUPlace()
exe = fluid.Executor(cpu)

# 数据处理
imglst = []  # 图片列表
textlst = []  # 文字列表
# textlst = np.loadtxt(path + "data/ocrData.txt", dtype=int)  # 文字列表 因为含有0开头的数字 所以不能是数值类型

for i in range(3):
    image = Image.open(path + "datawb/" + str(i) + ".jpg")
    image = np.array(image).reshape(70, 30).astype(np.float32)
    imglst.append(image)
imglst = np.array(imglst, dtype=float)
imglst.astype('float32')
'''
with open(path + "data/ocrData.txt") as f:
    for line in f:
        temp=[]
        temp.append(line[0])
        temp=np.array(temp,dtype=int)
        textlst.append(temp)#找了一晚上为什么出现 could not convert string to float: 'B' 发现原识别中出现了非数字的字符
textlst=np.array(textlst,dtype=int)
textlst.astype('int64')
'''
textlst = np.array([[0],[1],[2]]).astype('int64')
print(imglst)
# 定义输入层
image = fluid.layers.data(name='image', shape=[1, 70, 30], dtype='float32')  # 单通道 70x30图片
label = fluid.layers.data(name='label', shape=[1], dtype='int64')


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


# 获取分类器
model = convolutional_neural_network(image)

# 获取损失函数和准确率函数
cost = fluid.layers.cross_entropy(input=model, label=label)
avg_cost = fluid.layers.mean(cost)
acc = fluid.layers.accuracy(input=model, label=label)

# 定义优化方法
optimizer = fluid.optimizer.AdamOptimizer(learning_rate=0.001)
opts = optimizer.minimize(avg_cost)

# 获取测试程序
test_program = fluid.default_main_program().clone(for_test=True)
# 进行参数初始化
exe.run(fluid.default_startup_program())
# 定义输入数据维度
feeder = {'image': imglst, 'label': textlst}
for i in range(5):
    results = exe.run(program=fluid.default_main_program(),
                      feed=feeder,
                      fetch_list=[avg_cost])
    print(results[0])
