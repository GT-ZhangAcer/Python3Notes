import paddle.fluid as fluid
import paddle
import numpy as np
from PIL import Image
from GSODNet import BGSODNet

# Hyper parameter
use_cuda = False  # Whether to use GPU or not
batch_size = 10  # Number of incoming batches of data
epochs = 10  # Number of training rounds
save_model_path = "./model"
data_path = "./data"
block_num = 16  # 分块个数
view_pix = 32  # 感受野
learning_rate = 0.0001


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
    end_num = 6000
    if for_test:
        start_num = 6000
        end_num = 8000

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
            yield im, box_array, label_array

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
    # * Access to the Network
    result_list, loss = BGSODNet(10).net(img, box, label)

    #  Access to statistical information

    # Clone program
    evl_program = main_program.clone(for_test=True)
    # * Define the optimizer
    opt = fluid.optimizer.Adam(learning_rate=learning_rate)
    opt.minimize(loss)

# Feed configure
# if you want to shuffle "reader=paddle.reader.shuffle(dataReader(), buf_size)"
train_reader = paddle.batch(reader=data_reader(), batch_size=batch_size)
test_reader = paddle.batch(reader=data_reader(for_test=True), batch_size=batch_size)
train_feeder = fluid.DataFeeder(place=place, feed_list=[img, box, label])

# if you want to asynchronous reading
# batch_reader = fluid.io.PyReader(feed_list=[x, y], capacity=64)
# batch_reader.decorate_sample_list_generator(paddle.batch(data_reader(), batch_size=batch_size),place)

# Train Process
exe.run(startup)
# log_obj = WriteLog()
print("Start!")
for epoch in range(epochs):
    for step, data in enumerate(train_reader()):
        outs = exe.run(program=main_program,
                       feed=train_feeder.feed(data),
                       fetch_list=[result_list, loss])
        # log_obj.add_batch_train_value(outs[0], outs[1], outs[2])
        print(outs[1])

    for step, data in enumerate(test_reader()):
        outs = exe.run(program=evl_program,
                       feed=train_feeder.feed(data),
                       fetch_list=[result_list, loss])
        # log_obj.add_batch_test_value(outs[0], outs[1], outs[2])
    # train_print, test_print = log_obj.write_and_req()
    # print(epoch, "Train acc1 ", train_print["acc1"], "acc5 ", train_print["acc5"], "loss ", train_print["loss"])
    # print(epoch, "Test  acc1 ", test_print["acc1"], "acc5 ", test_print["acc5"], "loss ", test_print["loss"])

    fluid.io.save_persistables(dirname=save_model_path + "/" + str(epoch) + "persistables", executor=exe,
                               main_program=main_program)
    fluid.io.save_inference_model(dirname=save_model_path + "/" + str(epoch),
                                  feeded_var_names=["input_img"], target_vars=[result_list], main_program=main_program,
                                  executor=exe)
