from __future__ import print_function

import paddle
import paddle.fluid as fluid
import numpy
import math
import sys

BATCH_SIZE = 20

train_reader = paddle.batch(
    paddle.reader.shuffle(
        paddle.dataset.uci_housing.train(), buf_size=500),
        batch_size=BATCH_SIZE)

test_reader = paddle.batch(
    paddle.reader.shuffle(
        paddle.dataset.uci_housing.test(), buf_size=500),
        batch_size=BATCH_SIZE)

x = fluid.layers.data(name='x', shape=[13], dtype='float32')
y = fluid.layers.data(name='y', shape=[1], dtype='float32')
y_predict = fluid.layers.fc(input=x, size=1, act=None)

main_program = fluid.default_main_program()
startup_program = fluid.default_startup_program()

cost = fluid.layers.square_error_cost(input=y_predict, label=y)
avg_loss = fluid.layers.mean(cost)

sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.001)
sgd_optimizer.minimize(avg_loss)

#clone a test_program
test_program = main_program.clone(for_test=True)

use_cuda = False
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

exe = fluid.Executor(place)

# Plot data
from paddle.utils.plot import Ploter

train_prompt = "Train cost"
test_prompt = "Test cost"
plot_prompt = Ploter(train_prompt, test_prompt)

num_epochs = 100

# For training test cost
def train_test(executor, program, reader, feeder, fetch_list):
    accumulated = 1 * [0]
    count = 0
    for data_test in reader():
        outs = executor.run(program=program,
                            feed=feeder.feed(data_test),
                            fetch_list=fetch_list)
        accumulated = [x_c[0] + x_c[1][0] for x_c in zip(accumulated, outs)]
        count += 1
    return [x_d / count for x_d in accumulated]


# Specify the directory to save the parameters
params_dirname = "fit_a_line.inference.model"
feeder = fluid.DataFeeder(place=place, feed_list=[x, y])
naive_exe = fluid.Executor(place)
naive_exe.run(startup_program)
step = 0

exe_test = fluid.Executor(place)

# main train loop.
for pass_id in range(num_epochs):
    for data_train in train_reader():
        avg_loss_value, = exe.run(main_program,
                                  feed=feeder.feed(data_train),
                                  fetch_list=[avg_loss])
        if step % 10 == 0:  # record a train cost every 10 batches
            plot_prompt.append(train_prompt, step, avg_loss_value[0])
            plot_prompt.plot()
        if step % 100 == 0:  # record a test cost every 100 batches
            test_metics = train_test(executor=exe_test,
                                     program=test_program,
                                     reader=test_reader,
                                     fetch_list=[avg_loss.name],
                                     feeder=feeder)
            plot_prompt.append(test_prompt, step, test_metics[0])
            plot_prompt.plot()
            # If the accuracy is good enough, we can stop the training.
            if test_metics[0] < 10.0:
                break

        step += 1

        if math.isnan(float(avg_loss_value[0])):
            sys.exit("got NaN loss, training failed.")
    if params_dirname is not None:
        # We can save the trained parameters for the inferences later
        fluid.io.save_inference_model(params_dirname, ['x'],
                                      [y_predict], exe)