# Author:  Acer Zhang
# Datetime:2019/9/23
# Copyright belongs to the author.
# Please indicate the source for reprinting.

import paddle.fluid as fluid
import paddle
import numpy as np
from PIL import Image

# Hyper parameter
use_cuda = True  # Whether to use GPU or not
batch_size = 10  # Number of incoming batches of data
model_path = "./model/infer"  # infer model path

# Reader
img_size = np.array([512, 512])


def data_reader(index):
    def reader():
        im = Image.open("./data/img/" + str(index) + ".jpg")
        im = np.array(im).reshape(1, 3, 512, 512).astype(np.float32)
        yield im, img_size

    return reader


# Initialization

place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
exe = fluid.Executor(place)
startup = fluid.Program()

# load infer model
[infer_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(model_path, exe)

# Feed configure
# if you want to shuffle "reader=paddle.reader.shuffle(dataReader(), buf_size)"

# if you want to asynchronous reading
# batch_reader = fluid.io.PyReader(feed_list=[x, y], capacity=64)
# batch_reader.decorate_sample_list_generator(paddle.batch(data_reader(), batch_size=batch_size),place)

exe.run(startup)
# Start infer
infer_reader = paddle.batch(reader=data_reader(0), batch_size=batch_size)
infer_feeder = fluid.DataFeeder(place=place, feed_list=feed_target_names, program=infer_program)
for data in infer_reader():
    results = exe.run(infer_program, feed=infer_feeder.feed(data),
                      fetch_list=fetch_targets, return_numpy=False)
    bboxes = np.array(results[0])
    print(bboxes)
