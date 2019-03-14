import paddle.fluid as fluid
import paddle
import numpy as np
from PIL import Image
import os

#定义网络
def vgg_bn_drop(input):
    def conv_block(ipt, num_filter, groups, dropouts):
        return fluid.nets.img_conv_group(
            input=ipt,
            pool_size=2,
            pool_stride=2,
            conv_num_filter=[num_filter] * groups,
            conv_filter_size=3,
            conv_act='relu',
            conv_with_batchnorm=True,
            conv_batchnorm_drop_rate=dropouts,
            pool_type='max')

    conv1 = conv_block(input, 64, 2, [0.3, 0])
    conv2 = conv_block(conv1, 128, 2, [0.4, 0])
    conv3 = conv_block(conv2, 256, 3, [0.4, 0.4, 0])
    conv4 = conv_block(conv3, 512, 3, [0.4, 0.4, 0])
    conv5 = conv_block(conv4, 512, 3, [0.4, 0.4, 0])

    drop = fluid.layers.dropout(x=conv5, dropout_prob=0.5)
    fc1 = fluid.layers.fc(input=drop, size=512, act=None)
    bn = fluid.layers.batch_norm(input=fc1, act='relu')
    drop2 = fluid.layers.dropout(x=bn, dropout_prob=0.5)
    fc2 = fluid.layers.fc(input=drop2, size=512, act=None)
    predict = fluid.layers.fc(input=fc2, size=10, act='softmax')
    return predict

params_dirname ="./test02.inference.model"
print("训练后文件夹路径" + params_dirname)
# 参数初始化
cpu = fluid.CPUPlace()
exe = fluid.Executor(cpu)

datatype='float32'
#加载数据

with open("./data/ocrData.txt", 'rt') as f:
    a=f.read()
def dataReader():
    def redaer():
        for i in range(1500):
            im = Image.open("./data/" + str(i+1) + ".jpg").convert('L')
            im = np.array(im).reshape(30, 15).astype(np.float32)
            im = im / 255.0 * 2.0 - 1.0
            '''
            img = paddle.dataset.image.load_image(path + "data/" + str(i+1) + ".jpg")
            img=np.transpose(img, (2, 0, 1))'''
            labelInfo=a[i]
            yield im,labelInfo
    return redaer

def testReader():
    def redaer():
        for i in range(1501,1800):
            im = Image.open("./data/" + str(i+1) + ".jpg").convert('L')
            im = np.array(im).reshape(30, 15).astype(np.float32)
            im = im / 255.0 * 2.0 - 1.0
            '''
            img = paddle.dataset.image.load_image(path + "data/" + str(i+1) + ".jpg")
            img=np.transpose(img, (2, 0, 1))'''
            labelInfo=a[i]
            yield im,labelInfo
    return redaer
#定义网络
x = fluid.layers.data(name="x",shape=[1,30,15],dtype=datatype)
label = fluid.layers.data(name='label', shape=[1], dtype='int64')

net=vgg_bn_drop(x)
#定义损失函数
cost = fluid.layers.cross_entropy(input=net,label=label)
avg_cost = fluid.layers.mean(cost)
acc = fluid.layers.accuracy(input=net, label=label, k=1)
#定义优化方法
sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.01)
sgd_optimizer.minimize(avg_cost)
#数据传入设置
batch_reader = paddle.batch(reader=dataReader(), batch_size=1024)
testb_reader=paddle.batch(reader=testReader(), batch_size=1024)
feeder = fluid.DataFeeder(place=cpu, feed_list=[x, label])
prog=fluid.default_startup_program()
exe.run(prog)

trainNum=10
for i in range(trainNum):
    for batch_id,data in enumerate(batch_reader()):
        outs = exe.run(
            feed=feeder.feed(data),
            fetch_list=[label,avg_cost])#feed为数据表 输入数据和标签数据
        pross=float(i)/trainNum
        #打印输出面板
        print(str(i+1)+"次训练后损失值为："+str(outs[1]))
        print("当前训练进度百分比为："+str(pross*100)[:3].strip(".")+"%")
    for batch_id,data in enumerate(testb_reader()):
        test_acc,test_cost = exe.run(
            feed=feeder.feed(data),
            fetch_list=[label, avg_cost])  # feed为数据表 输入数据和标签数据
        test_costs=[]
        test_costs.append(test_cost[0])
        testcost = (sum(test_costs) / len(test_costs))
        print("预测损失为：",testcost,"\n")

#保存预测模型
path=params_dirname
def del_file(path):
    for i in os.listdir(path):
        path_file = os.path.join(path, i)
        if os.path.isfile(path_file):
            os.remove(path_file)
        else:
            del_file(path_file)
fluid.io.save_inference_model(params_dirname, ['x'],[net], exe)