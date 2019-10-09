# Author:  Acer Zhang
# Datetime:2019/9/23
# Copyright belongs to the author.
# Please indicate the source for reprinting.

import paddle.fluid as fluid
from mbnet import MobileNetSSD

# Hyper parameter
use_cuda = True  # Whether to use GPU or not
save_model_path = "./model"
img_size = [300, 300]

# Initialization
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
exe = fluid.Executor(place)

# Program
main_program = fluid.Program()
startup = fluid.Program()

# Edit Program
with fluid.program_guard(main_program=main_program, startup_program=startup):
    # * Define data types
    img = fluid.layers.data(name="img", shape=[3] + img_size, dtype="float32")
    # * Access to the Network
    # loss = BGSODNet(10).net(img, box, label)
    nms_out = MobileNetSSD().net(img, for_test=True)

# Train Process
exe.run(startup)

fluid.io.load_persistables(executor=exe, dirname=save_model_path + "/Epoch_74", main_program=main_program)
fluid.io.save_inference_model(dirname=save_model_path + "/infer", feeded_var_names=[img.name],
                              target_vars=[nms_out],
                              executor=exe, main_program=main_program)
