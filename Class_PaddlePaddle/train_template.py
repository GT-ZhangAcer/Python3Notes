import paddle.fluid as fluid
import paddle
import numpy as np
from PIL import Image
import numpy

# Hyper parameter
use_cuda = True  # Whether to use GPU or not
batch_size = 10  # Number of incoming batches of data
epochs = 10  # Number of training rounds
save_model_path = "./model"


# Reader
def data_reader(for_test=False):
    def reader():
        pass

    return reader()


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
for epoch in range(epochs):
    sum_acc1 = []
    sum_acc5 = []
    sum_cost = []
    test_sum_cost = []
    test_sum_acc1 = []
    test_sum_acc5 = []
    for step, data in enumerate(batch_reader()):
        outs = exe.run(program=main_program,
                       feed=train_feeder.feed(data),
                       fetch_list=[acc_1, acc_5, cost])
        # print("Train step:", train_num, batch_id, "Acc", outs[0], outs[1], 'Cost:', sum(outs[2]) / len(outs[2]))
        try:
            sum_acc1.append(float(outs[0]))
            sum_acc5.append(float(outs[1]))
            sum_cost.append(sum(outs[2]) / len(outs[2]))
        except:
            pass
    print(epoch, "Acc", avgacc, avgacc5, "Test Acc", t_avgacc, t_avgacc5, "cost", tcost)
    if epoch % 100 == 99:
        fluid.io.save_persistables(dirname=save_model_path + "/" + str(epoch) + "persistables", executor=exe,
                                   main_program=main_program)
    fluid.io.save_inference_model(dirname=save_model_path + "/" + str(epoch),
                                  feeded_var_names=["y"], target_vars=[net_out], main_program=main_program,
                                  executor=exe)
