import paddle.fluid as fluid
import paddle
import numpy as np
from PIL import Image
from script.log_Script import WriteLog
from GSODNet import BGSODNet

# Hyper parameter
use_cuda = True  # Whether to use GPU or not
batch_size = 10  # Number of incoming batches of data
epochs = 10  # Number of training rounds
save_model_path = "./model"
data_path = "./data"
block_num = 10  # Block_num


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
            im = np.array(im).reshape(3, 512, 512).astype(np.float32)
            with open(data_path + '/info/' + str(i) + ".info", "r") as f:
                info = f.read().replace(" ", "").split(",")
                box = info[:3]
                box = np.array(box).reshape(4).astype(np.float32)
                label = info[3]
                label = np.array(label).reshape(1).astype(np.int64)
            return im, box, label

    return reader


# Initialization
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
exe = fluid.Executor(place)

# Program
main_program = fluid.Program()
startup = fluid.Program()

# Edit Program
with fluid.program_guard(main_program=main_program, startup_program=startup):
    """Tips:Symbol * stands for Must"""
    # * Define data types
    img = fluid.layers.data(name="img", shape=[3, 512, 512], dtype="float32")
    box = fluid.layers.data(name="box", shape=[4], dtype="float32")
    label = fluid.layers.data(name="label", shape=[1], dtype="int64")
    # * Access to the Network
    net = BGSODNet(10).net(img, box, label)
    # * Define loss function

    #  Access to statistical information

    # Clone program

    # * Define the optimizer

    pass

# Feed configure
# if you want to shuffle "reader=paddle.reader.shuffle(dataReader(), buf_size)"
batch_reader = paddle.batch(reader=data_reader(), batch_size=batch_size)
test_batch_reader = paddle.batch(reader=data_reader(for_test=True), batch_size=batch_size)
train_feeder = fluid.DataFeeder(place=place, feed_list=[x, y])

# if you want to asynchronous reading
# batch_reader = fluid.io.PyReader(feed_list=[x, y], capacity=64)
# batch_reader.decorate_sample_list_generator(paddle.batch(data_reader(), batch_size=batch_size),place)

# Train Process
exe.run(startup)
log_obj = WriteLog()

for epoch in range(epochs):
    for step, data in enumerate(train_reader()):
        outs = exe.run(program=main_program,
                       feed=train_feeder.feed(data),
                       fetch_list=[acc1, acc5, loss])
        log_obj.add_batch_train_value(outs[0], outs[1], outs[2])

    for step, data in enumerate(test_reader()):
        outs = exe.run(program=test_program,
                       feed=train_feeder.feed(data),
                       fetch_list=[acc1, acc5, loss])
        log_obj.add_batch_test_value(outs[0], outs[1], outs[2])
    train_print, test_print = log_obj.write_and_req()
    print(epoch, "Train acc1 ", train_print["acc1"], "acc5 ", train_print["acc5"], "loss ", train_print["loss"])
    print(epoch, "Test  acc1 ", test_print["acc1"], "acc5 ", test_print["acc5"], "loss ", test_print["loss"])

    fluid.io.save_persistables(dirname=save_model_path + "/" + str(epoch) + "persistables", executor=exe,
                               main_program=main_program)
    fluid.io.save_inference_model(dirname=save_model_path + "/" + str(epoch),
                                  feeded_var_names=["input_img"], target_vars=[net], main_program=main_program,
                                  executor=exe)
