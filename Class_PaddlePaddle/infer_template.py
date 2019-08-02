import paddle.fluid as fluid
import paddle
import numpy as np
from PIL import Image

# Hyper parameter
use_cuda = True  # Whether to use GPU or not
batch_size = 10  # Number of incoming batches of data
model_path = ""  # infer model path


# Reader
def data_reader():
    def reader():
        pass

    return reader


# Initialization

place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
exe = fluid.Executor(place)
startup = fluid.Program()


# load infer model
[infer_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(model_path, exe)

# Feed configure
# if you want to shuffle "reader=paddle.reader.shuffle(dataReader(), buf_size)"
batch_reader = paddle.batch(reader=data_reader(), batch_size=batch_size)

# if you want to asynchronous reading
# batch_reader = fluid.io.PyReader(feed_list=[x, y], capacity=64)
# batch_reader.decorate_sample_list_generator(paddle.batch(data_reader(), batch_size=batch_size),place)

exe.run(startup)
# Start infer
for id_, data in enumerate(batch_reader()):
    results = exe.run(infer_program, feed={feed_target_names[0]: x}, fetch_list=fetch_targets)
    lab = np.argsort(results)[0][0][-1]
    print(lab)
