import paddle.fluid as fluid
import paddle
import numpy as np
from PIL import Image
from GSODNet import BGSODNet

# Hyper parameter
use_cuda = True  # Whether to use GPU or not
batch_size = 2  # Number of incoming batches of data
epochs = 1  # Number of training rounds
save_model_path = "./model"
data_path = "./data"
img_size = [512, 512]
block_num = 16  # 分块个数
view_pix = 32  # 感受野
learning_rate = 0.001


def info_read(info_path):
    """
    读取info文件下数据
    :param info_path: info文件路径
    :return: np Array :box, label, label
    """
    # 创建一个 H x W x len(box + label) 形状的数组来存储数据
    info_array = np.zeros([block_num, block_num, 5])
    with open(info_path, "r") as f:
        infos = f.read().replace(" ", "").split("\n")
        for info in infos:
            info = info.split(",")
            mini_data = [float(i) for i in info]
            h = mini_data[0] // view_pix
            w = mini_data[1] // view_pix
            info_array[int(h)][int(w)] = mini_data
            return info_array


# Reader

def data_reader(for_test=False):
    start_num = 0
    end_num = 300
    if for_test:
        start_num = 300
        end_num = 500

    def reader():
        for i in range(start_num, end_num):
            im = Image.open(data_path + '/img/' + str(i) + ".jpg")
            im = np.array(im).reshape(1, 3, 512, 512).astype(np.float32)
            info_path = data_path + '/info/' + str(i) + ".info"
            info_array = info_read(info_path)
            box_array = info_array[:, :, :4]
            box_array = box_array.reshape(1, block_num ** 2, 4)
            label_array = info_array[:, :, 4:5]
            label_array = label_array.reshape(1, block_num ** 2)
            img_size_array = np.array(img_size).reshape(1, 2).astype(np.int32)
            yield im, box_array, label_array, img_size_array

    return reader


# Initialization
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
exe = fluid.Executor(place)

# Program
main_program = fluid.Program()
# 如果不定义上方的那行代码则默认执行fluid.default_main_program()为主程序，同时也不需要使用with关键字编辑项目了
startup = fluid.Program()
# 这里使用 startup =fluid.default_startup_program()作为缺省的启动程序也是可以的


# Edit Program
with fluid.program_guard(main_program=main_program, startup_program=startup):
    """Tips:Symbol * stands for Must"""
    # * Define data types
    img = fluid.layers.data(name="img", shape=[3, 512, 512], dtype="float32")
    box = fluid.layers.data(name="box", shape=[block_num ** 2, 4], dtype="float32")
    label = fluid.layers.data(name="label", shape=[block_num ** 2], dtype="int32")
    img_size_2d = fluid.layers.data(name='img_size', shape=[2], dtype='int32')
    # * Access to the Network
    scores, loss = BGSODNet(10).net(img, box, label, img_size_2d)

    #  Access to statistical information

    # Clone program
    evl_program = main_program.clone(for_test=True)
    # * Define the optimizer
    opt = fluid.optimizer.SGD(learning_rate=learning_rate)
    opt.minimize(loss)

# Feed configure
# if you want to shuffle "reader=paddle.reader.shuffle(dataReader(), buf_size)"
train_reader = paddle.batch(reader=paddle.reader.shuffle(data_reader(), 500), batch_size=batch_size)
test_reader = paddle.batch(reader=data_reader(for_test=True), batch_size=batch_size)
train_feeder = fluid.DataFeeder(place=place, feed_list=[img, box, label, img_size_2d])

# Train Process
exe.run(startup)
# log_obj = WriteLog()
print("Start!")
for epoch in range(epochs):
    for step, data in enumerate(train_reader()):
        outs = exe.run(program=main_program,
                       feed=train_feeder.feed(data),
                       fetch_list=[scores, loss])
        print(outs[0])

    for step, data in enumerate(test_reader()):
        outs = exe.run(program=evl_program,
                       feed=train_feeder.feed(data),
                       fetch_list=[scores, loss])

    fluid.io.save_params(executor=exe, dirname=save_model_path + "/One_Epoch", main_program=main_program)
