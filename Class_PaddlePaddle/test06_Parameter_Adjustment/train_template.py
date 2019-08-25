import paddle.fluid as fluid
import paddle
import numpy as np
from PIL import Image
from script.log_Script import WriteLog

# Hyper parameter
use_cuda = True  # Whether to use GPU or not
batch_size = 10  # Number of incoming batches of data
epochs = 10  # Number of training rounds
save_model_path = "./model"


# Reader
def data_reader(for_test=False):
    def reader():
        pass

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

    # * Access to the Network

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

log_obj = WriteLog()

for epoch in range(epochs):
    for step, data in enumerate(batch_reader()):
        outs = exe.run(program=main_program,
                       feed=train_feeder.feed(data),
                       fetch_list=[acc_1, acc_5, loss],
                       return_numpy=False)
        log_obj.add_batch_train_value(outs[1], outs[2], outs[3])

    train_print, _ = log_obj.write_and_req()
    print("Avg acc1 ", train_print["acc1"], "acc5 ", train_print["acc5"], "loss ", train_print["loss"])

    fluid.io.save_persistables(dirname=save_model_path + "/" + str(epoch) + "persistables", executor=exe,
                               main_program=main_program)
    fluid.io.save_inference_model(dirname=save_model_path + "/" + str(epoch),
                                  feeded_var_names=["y"], target_vars=[net_out], main_program=main_program,
                                  executor=exe)
