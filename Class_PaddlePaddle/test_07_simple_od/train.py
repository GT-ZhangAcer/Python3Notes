import paddle.fluid as fluid
import paddle
import numpy as np
from PIL import Image
import json
from mbnet import MobileNetSSD
from GSODNet import BGSODNet

# Hyper parameter
use_cuda = True  # Whether to use GPU or not
batch_size = 4  # Number of incoming batches of data
epochs = 100  # Number of training rounds
data_path = "./data"
save_model_path = "./model"
img_size = [512, 512]
block_num = 10  # 目标最大数量
learning_rate = 0.001


# def data_reader():
#     def reader():
#         with open("./lslm_data/train.txt", "r") as f:
#             infos = f.read().split("\n")
#             for line in infos:
#                 info = line.split("\t")
#                 if info[-1] is "":
#                     info.pop(-1)
#                 img_name = info[0]
#                 label_infos = info[2:]
#                 box_list = []
#                 label_list = []
#                 for label_info in label_infos:
#                     label_info = json.loads(label_info)
#                     if label_info["value"] == "bolt":
#                         this_label = 1
#                     else:
#                         this_label = 2
#                     up_x, up_y = label_info["coordinate"][0]
#                     down_x, down_y = label_info["coordinate"][1]
#                     # this_box = [((up_x + down_x) * 0.5 - 360) / 1440,
#                     #             ((up_y + down_y) * 0.5) / 1080,
#                     #             (down_x - up_x) / 1440,
#                     #             (down_y - up_y) / 1080]
#                     this_box = [up_x / 1440, up_y / 1080, down_x / 1440, down_y / 1080]
#                     box_list.append(this_box)
#                     label_list.append(this_label)
#                 im = Image.open(data_path + "/" + img_name)
#                 im = im.crop((360, 0, 1440, 1080))
#                 im = im.resize((300, 300), Image.LANCZOS)
#                 im = np.array(im).transpose((2, 0, 1)).reshape(1, 3, 300, 300) * 0.007843
#                 box_list = np.array(box_list)
#                 label_list = np.array(label_list)
#                 yield im, box_list, label_list
#
#     return reader

def reader():
    def yield_data():
        for index in range(500):
            box_list = []
            label_list = []
            with open(data_path + "/info/" + str(index) + ".info") as f:
                lines = f.read()
                for line in lines.split("\n"):
                    info = line.split(", ")
                    box_list.append([float(size) / 512 for size in info[:4]])
                    label_list.append(info[-1])
            im = Image.open(data_path + "/img/" + str(index) + ".jpg")
            # im = im.resize((300, 300), Image.LANCZOS)
            im = np.array(im).transpose((2, 0, 1)).reshape(1, 3, 512, 512) / 255
            box_list = np.array(box_list)
            label_list = np.array(label_list)
            yield im, box_list, label_list

    return yield_data


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
    img = fluid.layers.data(name="img", shape=[3] + img_size, dtype="float32")
    box = fluid.layers.data(name="box", shape=[4], dtype="float32", lod_level=1)
    label = fluid.layers.data(name="label", shape=[1], dtype="int32", lod_level=1)
    # * Access to the Network
    loss, cur_map, accum_map, map_eval = BGSODNet(10).net(img, box, label)
    # loss, cur_map, _ = MobileNetSSD().net(img, box_ipt_list=box, label_list=label)

    #  Access to statistical information

    # Clone program
    # evl_program = main_program.clone(for_test=True)
    # * Define the optimizer
    opt = fluid.optimizer.Adam(learning_rate=learning_rate)
    opt.minimize(loss)

# fluid.io.load_persistables(executor=exe, dirname=save_model_path + "/Epoch_60", main_program=main_program)

# Feed configure
# if you want to shuffle "reader=paddle.reader.shuffle(dataReader(), buf_size)"
train_reader = paddle.batch(reader=paddle.reader.shuffle(reader(), 500), batch_size=batch_size)
train_feeder = fluid.DataFeeder(place=place, feed_list=[img, box, label])

# Train Process
exe.run(startup)
# log_obj = WriteLog()
print("Start!")
for epoch in range(epochs):
    for step, data in enumerate(train_reader()):
        outs = exe.run(program=main_program,
                       feed=train_feeder.feed(data),
                       fetch_list=[loss, cur_map, accum_map])

        if step == 0:
            print(outs[0], outs[1], outs[2])
            map_eval.reset(exe)

    fluid.io.save_persistables(executor=exe, dirname=save_model_path + "/OCR_" + str(epoch), main_program=main_program)
