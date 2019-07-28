import paddle.fluid as fluid
import paddle
import numpy as np
from PIL import Image
import numpy

# Hyper parameter
use_cuda = True  # Whether to use GPU or not
model_path = ""  # infer model path

# Make infer datas

# Initialization

cpu = fluid.CUDAPlace(0)
exe = fluid.Executor(cpu)
startup = fluid.Program()
exe.run(startup)

# load infer model
[infer_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(model_path, exe)

# Start infer
results = exe.run(infer_program, feed={feed_target_names[0]: x}, fetch_list=fetch_targets)
lab = np.argsort(results)[0][0][-1]
print(lab)
